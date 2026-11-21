"""Were the fly accounts' sells (and buys) any good? Read-only; uses the saved scores, no retraining.

    python -m experiments.sell_quality --root league_v7 --panel data_fundamentals/panel_hybrid.parquet --flies 5 --out sellq

For every fill of the default hold policy (accounts.hold_policy.MIX) in every saved round it records
  * the return of that stock over the next 21 and 63 trading days, AFTER the fill (the fill is at that day's close),
  * the same horizon for three yardsticks: all stocks equal-weight (uni), SPY (spy) and, for a sell, the stocks the account kept
    holding that day (kept: other stocks only, ETF core excluded),
  * for a sell, the gain or loss on the position at the sale (price / average cost - 1),
  * where the price sat in its trailing 126-day range at the fill (0 = at the low, 1 = at the high): the direct "buy high, sell low" check.
A good sell is followed by the stock UNDERperforming the yardstick; a good buy by OUTperforming it. Fills too close to the end of the data
to have a full horizon are left out of that horizon. Numbers are averaged per fly-round first, then across them; flies in the same round
share one market, so treat the t-stats as optimistic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from accounts.instruments import APPROVED_ETFS, is_core

HORIZONS = (21, 63)
LOOKBACK = 126


def fill_table(fills, prices: pd.DataFrame, stocks, hs=HORIZONS, lookback: int = LOOKBACK) -> pd.DataFrame:
    """One row per STOCK fill. prices must run past the last fill (by max(hs) days when possible). stocks: the stock tickers."""
    stocks = [c for c in stocks if c in prices.columns]
    fwd = {h: prices.shift(-h) / prices - 1.0 for h in hs}
    uni = {h: fwd[h][stocks].mean(axis=1) for h in hs}
    spy = {h: fwd[h]["SPY"] if "SPY" in prices.columns else pd.Series(np.nan, index=prices.index) for h in hs}
    lo = prices.rolling(lookback, min_periods=20).min()
    hi = prices.rolling(lookback, min_periods=20).max()
    held, basis, rows = {}, {}, []
    by_day = {}
    for f in fills:
        by_day.setdefault(pd.Timestamp(f.date), []).append(f)
    for day in sorted(by_day):
        if day not in prices.index:
            continue
        for f in by_day[day]:
            t, n, px = f.ticker, float(f.shares), float(f.price)
            pnl = np.nan
            if f.side == "BUY":
                held[t] = held.get(t, 0.0) + n
                basis[t] = basis.get(t, 0.0) + n * px
            else:
                avg = basis.get(t, 0.0) / held[t] if held.get(t, 0.0) > 0 else 0.0
                pnl = px / avg - 1.0 if avg > 0 else np.nan
                basis[t] = basis.get(t, 0.0) * (1.0 - n / held[t]) if held.get(t, 0.0) > 0 else 0.0
                held[t] = held.get(t, 0.0) - n
                if held[t] <= 1e-9:
                    held.pop(t, None)
                    basis.pop(t, None)
            if t not in stocks:
                continue
            span = hi.at[day, t] - lo.at[day, t]
            row = dict(date=day, ticker=t, side=f.side, pnl=pnl,
                       range_pos=(px - lo.at[day, t]) / span if np.isfinite(span) and span > 0 else np.nan)
            kept = [k for k in held if k in stocks and k != t and not is_core(k) and k not in APPROVED_ETFS]
            for h in hs:
                row[f"fwd{h}"] = fwd[h].at[day, t]
                row[f"uni{h}"] = uni[h].at[day]
                row[f"spy{h}"] = spy[h].at[day]
                row[f"kept{h}"] = float(np.nanmean(fwd[h].loc[day, kept].to_numpy(float))) if f.side == "SELL" and kept else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, hs=HORIZONS) -> pd.DataFrame:
    """df: fill_table rows plus 'round' and 'fly'. -> one line per (kind, horizon, yardstick); per fly-round means first."""
    out = []
    kinds = {"sells": df[df.side == "SELL"], "sells at a gain": df[(df.side == "SELL") & (df.pnl > 0)],
             "sells at a loss": df[(df.side == "SELL") & (df.pnl <= 0)], "buys": df[df.side == "BUY"]}
    for kind, d in kinds.items():
        if d.empty:
            continue
        for h in hs:
            for ref in ("uni", "spy", "kept"):
                if ref == "kept" and kind == "buys":
                    continue
                x = d[[f"fwd{h}", f"{ref}{h}", "round", "fly"]].dropna()
                if x.empty:
                    continue
                x = x.assign(ex=x[f"fwd{h}"] - x[f"{ref}{h}"])
                per = x.groupby(["round", "fly"])["ex"].mean()
                good = (x["ex"] < 0) if kind.startswith("sells") else (x["ex"] > 0)
                se = per.std(ddof=1) / np.sqrt(len(per)) if len(per) > 1 else np.nan
                out.append(dict(kind=kind, horizon=h, vs=ref, n_fills=len(x), n_fly_rounds=len(per), stock_fwd=x[f"fwd{h}"].mean(),
                                yardstick=x[f"{ref}{h}"].mean(), excess=per.mean(), t=per.mean() / se if se and se > 0 else np.nan,
                                pct_good=float(good.mean())))
    pos = []
    for kind, d in (("buys", df[df.side == "BUY"]), ("sells", df[df.side == "SELL"])):
        x = d.dropna(subset=["range_pos"])
        if len(x):
            pos.append(dict(kind=kind, horizon=0, vs="range_pos", n_fills=len(x), n_fly_rounds=x.groupby(["round", "fly"]).ngroups,
                            stock_fwd=np.nan, yardstick=np.nan, excess=x["range_pos"].mean(), t=np.nan, pct_good=np.nan))
    return pd.DataFrame(out + pos).set_index(["kind", "horizon", "vs"]) if (out or pos) else pd.DataFrame()


def main():
    from accounts.etfs import mask_before_listing, with_etfs
    from accounts.league_report import prices_from_returns
    from dashboard import state, trades
    from tournament.store import Store

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--etf-root", default=None)
    ap.add_argument("--rounds", default=None, help="e.g. 4-18 (default: every round with saved scores)")
    ap.add_argument("--flies", type=int, default=5)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = Store(a.root).load_config()
    if a.panel:
        cfg["panel"] = a.panel
    print("reading panel (aligned to the league's own label)...", flush=True)
    index, fwd1 = trades.league_index(cfg)
    stocks = list(fwd1.columns)
    fwd1 = with_etfs(fwd1, a.etf_root or Path(cfg["panel"]).parent)
    root = Path(a.root)
    rounds = sorted({int(p.name[1:3]) for p in (root / "rounds" / "scores").glob("r*_*.npz")})
    if a.rounds:
        lo, _, hi = a.rounds.partition("-")
        rounds = [r for r in rounds if int(lo) <= r <= int(hi or lo)]
    frames = []
    for r in rounds:
        ids = sorted({p.name[4:-4] for p in (root / "rounds" / "scores").glob(f"r{r:02d}_*.npz")})
        for aid in [ids[int(j * len(ids) / a.flies)] for j in range(min(a.flies, len(ids)))]:
            sc = state.load_scores(a.root, r, aid)
            idxs = [i for i in sorted(sc) if i < len(index)]
            if not idxs or all(np.nanstd(sc[i]) < 1e-9 for i in idxs):
                continue
            acct, _, prices, _ = trades.run_policy(sc, index, fwd1, k=a.k)
            if not len(prices):
                continue
            w0 = fwd1.index.get_loc(prices.index[0])
            w1 = min(fwd1.index.get_loc(prices.index[-1]) + max(HORIZONS) + 1, len(fwd1))
            ext = mask_before_listing(prices_from_returns(fwd1, list(range(w0, w1))), fwd1.iloc[w0:w1])
            t = fill_table(acct.filled, ext, stocks)
            if len(t):
                frames.append(t.assign(round=r, fly=aid))
        print(f"round {r} done", flush=True)
    if not frames:
        print("no fills found")
        return
    df = pd.concat(frames, ignore_index=True)
    pd.set_option("display.width", 200)
    s = summarize(df)
    print("\nexcess = stock's forward return minus the yardstick (negative is GOOD for a sell, positive is GOOD for a buy);"
          "\npct_good = share of fills that went the right way; range_pos rows: 0 = at the 126-day low, 1 = at the high\n")
    print(s.round(3).to_string())
    if a.out:
        df.to_csv(f"{a.out}_fills.csv", index=False)
        s.to_csv(f"{a.out}_summary.csv")


if __name__ == "__main__":
    main()
