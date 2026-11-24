"""F3: does any of the 9 new SEC features carry sector-neutral skill? One test per feature, rules fixed BEFORE looking.

    python -m experiments.screen_fund_v2 --panel data_fundamentals/panel_v2.parquet --sectors data_fundamentals/universe/sectors.csv --out screen_v2

Rules (pre-registered):
  * Label: the league's own, "5+21+63" sector-neutral (tournament.labels). A single 21-day sector-neutral label is shown as a second view.
  * Dates: 2011-01-01 .. 2023-12-31 only (coverage is ~0 before 2010; 2024 on stays untouched, the league holdout starts 2024-08-12),
    one date in every 21 so the labels barely overlap. Only rows where the feature is really known (its within-sector rank is not the 0 filler).
  * Per date: Spearman rank correlation of the feature with the label across stocks (at least 30 stocks). Per calendar year: the mean of those.
  * t = mean of the yearly means / its standard error (13 years). A feature SURVIVES if |t| >= 2.5 (9 features tested, so the usual 2 is too
    lenient) AND its yearly means have the same sign in at least 10 of 13 years. |t| 2 to 2.5 is only WEAK. Everything else FAILS.
  * The 11 old fundamentals and 'fund_known' are printed as context, not tested for survival.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from experiments.add_fund_v2 import NEW

START, END, STEP = "2011-01-01", "2023-12-31", 21
MIN_STOCKS = 30
T_SURVIVE, T_WEAK, SIGN_SHARE = 2.5, 2.0, 10 / 13


def rank_ic_rows(F: pd.DataFrame, Y: pd.DataFrame, min_n: int = MIN_STOCKS) -> pd.Series:
    """Spearman IC per row (date) of two dates x tickers frames, using only cells where BOTH are finite."""
    ok = F.notna() & Y.notna()
    f, y = F.where(ok).rank(axis=1), Y.where(ok).rank(axis=1)
    n = ok.sum(axis=1)
    fm, ym = f.sub(f.mean(axis=1), axis=0), y.sub(y.mean(axis=1), axis=0)
    cov = (fm * ym).sum(axis=1)
    den = np.sqrt((fm ** 2).sum(axis=1) * (ym ** 2).sum(axis=1))
    ic = cov / den.where(den > 0)
    return ic.where(n >= min_n).dropna()


def by_year(ic: pd.Series) -> pd.DataFrame:
    """Yearly means of the daily ICs -> (mean, t across years, years positive, years)."""
    ym = ic.groupby(ic.index.year).mean()
    n = len(ym)
    se = ym.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return pd.DataFrame(dict(ic=ym.mean(), t=ym.mean() / se if se and se > 0 else np.nan, pos=int((ym > 0).sum()), years=n,
                             dates=len(ic)), index=[0]).iloc[0]


def verdict(t: float, pos: int, years: int) -> str:
    if not np.isfinite(t):
        return "n/a"
    share = max(pos, years - pos) / years if years else 0.0
    if abs(t) >= T_SURVIVE and share >= SIGN_SHARE - 1e-9 and (pos > years / 2) == (t > 0):
        return "SURVIVES"
    if abs(t) >= T_WEAK:
        return "weak"
    return "fails"


def screen(panel: pd.DataFrame, labels: dict, cols: list[str], start=START, end=END, step=STEP) -> pd.DataFrame:
    """labels: name -> Series aligned to the panel rows. -> one row per (feature, label)."""
    dates = panel.index.get_level_values("date").unique().sort_values()
    use = dates[(dates >= start) & (dates <= end)][::step]
    sub = panel[panel.index.get_level_values("date").isin(use)]
    rows = []
    for lname, y in labels.items():
        Y = y.reindex(sub.index).unstack("ticker")
        for c in cols:
            F = sub[c].where(sub[c] != 0).unstack("ticker").reindex(index=Y.index, columns=Y.columns)
            ic = rank_ic_rows(F, Y)
            if ic.empty:
                continue
            r = by_year(ic)
            rows.append(dict(feature=c, label=lname, ic=r["ic"], t=r["t"], years_pos=int(r["pos"]), years=int(r["years"]),
                             dates=int(r["dates"]), verdict=verdict(r["t"], int(r["pos"]), int(r["years"]))))
    return pd.DataFrame(rows)


def main():
    from tournament.labels import load_sectors, make_labels
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default="data_fundamentals/panel_v2.parquet")
    ap.add_argument("--sectors", default="data_fundamentals/universe/sectors.csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    print("loading panel ...", flush=True)
    panel = pd.read_parquet(a.panel)
    sec = load_sectors(a.sectors)
    labels = {"league 5+21+63": make_labels(panel, "5+21+63", sec)[0], "21d only": make_labels(panel, "21", sec)[0]}
    new = [f"s_{m}" for m in NEW]
    old = [c for c in panel.columns if c in ("earnings_yield", "book_to_price", "fcf_yield", "roe", "roa", "gross_margin", "rev_growth_yoy",
                                             "eps_growth_yoy", "debt_to_equity", "accruals", "days_since_filing")]
    res = screen(panel, labels, new)
    ctx = screen(panel, labels, old)
    pd.set_option("display.width", 200)
    print(f"\nNEW FEATURES ({len(new)} tested; survive = |t| >= {T_SURVIVE} and the same sign in >= 10 of 13 years)\n")
    print(res.round(4).to_string(index=False))
    print("\nCONTEXT: the 11 old fundamentals on the same rules (not tested for survival)\n")
    print(ctx.round(4).to_string(index=False))
    n_sur = int((res[res.label == 'league 5+21+63'].verdict == "SURVIVES").sum())
    print(f"\n{n_sur} of {len(new)} new features survive on the league label.")
    if a.out:
        res.to_csv(f"{a.out}_new.csv", index=False)
        ctx.to_csv(f"{a.out}_old.csv", index=False)


if __name__ == "__main__":
    main()
