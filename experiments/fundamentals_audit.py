"""Do the features carry signal, and is any of them leaking the future? (T6 audit)

    python -m experiments.fundamentals_audit --panel data_fundamentals/panel_hybrid.parquet

For every feature: daily rank-IC against the 21-day excess return (the league's label), by era, with a t-stat and how many
years had the right sign. Also the IC against the NEXT-DAY return as a leak check: a feature known at the close of day t should
have almost no power over tomorrow's return, so a big number there means the feature saw the future (a filing-date or
look-ahead bug). Rows with fund_known == -1 (no real fundamentals) are excluded from the fundamental features' IC, so filler
zeros cannot dilute or inflate them.

The last --holdout (default 500) labelled dates are NEVER used, same as the league, so the audit does not peek at the test set.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from env.features import FUND_FEATURES, feature_columns

ERAS = [("2001-08", 2001, 2008), ("2009-14", 2009, 2014), ("2015-19", 2015, 2019), ("2020+", 2020, 2100)]


def _ic(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return np.nan
    return float(np.corrcoef(rankdata(a[ok]), rankdata(b[ok]))[0, 1])


def forward_excess(panel: pd.DataFrame, horizon: int) -> pd.Series:
    """Excess forward return over `horizon` trading days, aligned to the panel rows: compounded next-day returns of days
    t+1..t+horizon minus that day's cross-sectional mean. NaN if any of those days is missing for the stock (it left the index
    window or the data ends). horizon 21 uses the panel's own fwd_excess_21 (built from the full price table)."""
    if horizon == 21 and "fwd_excess_21" in panel.columns:
        return panel["fwd_excess_21"]
    fr = panel["fwd_ret"].unstack("ticker").sort_index()
    roll = np.expm1(np.log1p(fr).rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1)))
    exc = roll.sub(roll.mean(axis=1), axis=0).stack()
    exc = exc[~exc.index.duplicated()]
    return exc.reindex(panel.index)


def audit(panel: pd.DataFrame, holdout: int = 500, step: int = 5, horizon: int = 21) -> pd.DataFrame:
    """Per-feature IC against the `horizon`-day label. The label of the last audited date reaches `horizon` days ahead, so
    the last `horizon` dates before the holdout are dropped too: no audited label ever touches a holdout return."""
    feats = [c for c in feature_columns(panel) if c not in ("fund_known", "_y")]
    panel = panel.assign(_y=forward_excess(panel, horizon))
    dates = sorted(panel["_y"].dropna().index.get_level_values("date").unique())
    dates = dates[:len(dates) - holdout - horizon] if holdout else dates
    sample = dates[::step]
    d = panel.index.get_level_values("date")
    ics = {f: [] for f in feats}
    leak = {f: [] for f in feats}
    when = []
    has_known = "fund_known" in panel.columns
    for dt in sample:
        g = panel[d == dt]
        y21, y1 = g["_y"].to_numpy(float), g["fwd_excess"].to_numpy(float)
        known = (g["fund_known"].to_numpy() > -1) if has_known else np.ones(len(g), bool)
        for f in feats:
            x = g[f].to_numpy(float)
            m = known if f in FUND_FEATURES else np.ones(len(g), bool)
            ics[f].append(_ic(x[m], y21[m]))
            leak[f].append(_ic(x[m], y1[m]))
        when.append(dt)
    when = pd.DatetimeIndex(when)
    rows = []
    for f in feats:
        s = pd.Series(ics[f], index=when).dropna()
        l = pd.Series(leak[f], index=when).dropna()
        n_eff = max(len(s) * step / horizon, 1.0)                 # 21-day labels overlap: far fewer independent samples
        r = dict(feature=f, kind="fundamental" if f in FUND_FEATURES else "price", n_dates=len(s),
                 ic_21d=s.mean(), t_stat=(s.mean() / s.std() * np.sqrt(n_eff)) if s.std() > 0 else 0.0,
                 ic_1d_leakcheck=l.mean())
        for name, a, b in ERAS:
            r[name] = s[(s.index.year >= a) & (s.index.year <= b)].mean()
        yr = s.groupby(s.index.year).mean()
        r["years_right_sign"] = f"{int((np.sign(yr) == np.sign(s.mean())).sum())}/{len(yr)}"
        rows.append(r)
    out = pd.DataFrame(rows).set_index("feature")
    out["leak_flag"] = (out.ic_1d_leakcheck.abs() > 0.03) & (out.ic_1d_leakcheck.abs() > 0.5 * out.ic_21d.abs())
    out = out.sort_values("t_stat", key=lambda x: -x.abs())
    out.attrs["last_date"] = sample[-1] if len(sample) else None
    return out


def compare_horizons(panel, horizons, holdout=500, step=5):
    """-> (ic, t) DataFrames: features x horizons."""
    ics, ts = {}, {}
    for h in horizons:
        r = audit(panel, holdout, step, h)
        ics[h], ts[h] = r["ic_21d"], r["t_stat"]
    order = audit(panel, holdout, step, horizons[0]).index if False else list(ics[horizons[0]].index)
    return pd.DataFrame(ics).loc[order], pd.DataFrame(ts).loc[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data_fundamentals/panel_hybrid.parquet")
    ap.add_argument("--holdout", type=int, default=500)
    ap.add_argument("--step", type=int, default=5, help="use every k-th date (speed)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--horizon", type=int, default=21, help="label horizon in trading days")
    ap.add_argument("--horizons", default="", help="comma list, e.g. 5,21,63,126,252: one comparison table instead of the full audit")
    a = ap.parse_args()
    panel = pd.read_parquet(a.panel)
    if a.horizons:
        hs = [int(x) for x in a.horizons.split(",")]
        print(f"{len(panel):,} rows; holdout excluded; horizons {hs} ...", flush=True)
        ic, t = compare_horizons(panel, hs, a.holdout, a.step)
        cell = lambda i, j: f"{ic.iloc[i, j]:+.3f} ({t.iloc[i, j]:+.1f})"
        tab = pd.DataFrame({f"{h}d": [cell(i, j) for i in range(len(ic))] for j, h in enumerate(hs)}, index=ic.index)
        tab.insert(0, "kind", ["fundamental" if f in FUND_FEATURES else "price" for f in ic.index])
        print(tab.to_string())
        print("\ncells are IC (t-stat). t uses independent samples ~ n/horizon, so long horizons have wide error bars; |t| < 2 is noise.")
        out = a.out or a.panel.replace(".parquet", "_horizons.csv")
        pd.concat({"ic": ic, "t": t}, axis=1).to_csv(out)
        print(f"saved {out}")
        return
    print(f"{len(panel):,} rows; holdout (last {a.holdout} labelled dates) excluded; {a.horizon}-day label; every {a.step}th date used ...", flush=True)
    res = audit(panel, a.holdout, a.step, a.horizon)
    pd.set_option("display.width", 220, "display.float_format", lambda x: f"{x:+.3f}")
    show = res.drop(columns=["n_dates"]).assign(t_stat=res.t_stat.map(lambda x: f"{x:+.1f}"))
    print(show.to_string())
    print("\nt_stat is approximate (21-day labels overlap, so independent samples are n/21). |t| < 2 means indistinguishable from noise.")
    print("ic_1d_leakcheck should be near 0; leak_flag=True means check that feature's timing before trusting anything.")
    out = a.out or a.panel.replace(".parquet", "_audit.csv")
    res.to_csv(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
