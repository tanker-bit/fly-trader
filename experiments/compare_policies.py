"""Does the calmer hold policy beat the old one on your rounds? Rule by rule, with and without the SPY sleeve.

    python -m experiments.compare_policies --root league_v7 --flies 5 --out compare

For each finished round, a few flies (--flies, spread evenly over the round's flies) each run a $1M long-only account under every
variant below on the SAME saved scores and the SAME prices; nothing is trained and no holdout is touched. Prices have a 125-day run-up
before the first trade so the stop/filters have history. Variants:

  old             the original policy (rank hysteresis, rebalance every 21 days)
  hold            the hold policy with every rule on (stops, trailing stop, falling-stock and jump filters, concentration cap)
  hold plain      only the slow rebalance and the rank rules (all risk rules off)
  hold -stops / -cap   everything on except the stops / the concentration cap
  hold +core ...  hold with a core of broad ETFs that is at least 30% of the account at all times (target 35%), tried with four cores:
                  SPY alone, broad (VTI/RSP/QQQ), defensive (SPY/XLV/XLP), dividend (VTI/SCHD). Needs the ETF download.
  SPY buy&hold    just SPY, for reference (needs the ETF download)

Columns: mean return, mean excess over the equal-weight stock benchmark, rounds beating it, the worst round, mean max drawdown, mean number
of fills, mean fees, and a paired t of (variant - old) across rounds.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from accounts.hold_policy import CORES, HoldParams, MIX
from accounts.instruments import Universe
from accounts.league_report import prices_from_returns
from accounts.sim import metrics, run_account

PRE = 125            # trading days of price history before the first trade (125 + 1 = 126 = 2 x 63 = 6 x 21: rebalance days line up)

BASE = HoldParams()
VARIANTS = {
    "old": None,
    "hold": BASE,
    "hold plain": replace(BASE, stop_loss=None, trail=None, knife=None, jump=None, max_weight=None),
    "hold -stops": replace(BASE, stop_loss=None, trail=None),
    "hold -cap": replace(BASE, max_weight=None),
    "hold +core SPY": replace(BASE, core=CORES["spy"]),
    "hold +core broad": replace(BASE, core=CORES["broad"]),
    "hold +core defensive": replace(BASE, core=CORES["defensive"]),
    "hold +core dividend": replace(BASE, core=CORES["dividend"]),
    "mix (broad+sleeve)": MIX,
    "mix -sleeve": replace(MIX, tactical_k=0),
    "mix -long-term": replace(MIX, lt_sell_rank=None),
}


def window_rows(prices: pd.DataFrame, sc: pd.DataFrame, stock_cols, variants=VARIANTS, k: int = 20, etf_ok: bool = True) -> list[dict]:
    """Run every variant on one fly-round. prices: PRE run-up days + the window. -> one dict per variant."""
    uni = Universe(stocks=stock_cols)
    r = prices[list(stock_cols)].pct_change(fill_method=None).iloc[PRE + 1:].mean(axis=1)
    bench = float((1.0 + r.fillna(0.0)).prod() - 1.0)
    out = []
    for name, hold in variants.items():
        if hold is not None and hold.core and not etf_ok:
            continue
        if hold is None:
            acct, eq = run_account(prices, sc, uni, k=k, every=21)
        else:
            acct, eq = run_account(prices, sc, uni, hold=replace(hold, k=k))
        m = metrics(eq)
        out.append(dict(variant=name, total_return=m["total_return"], max_drawdown=m["max_drawdown"], bench_return=bench,
                        excess=m["total_return"] - bench, fills=len(acct.filled), fees=acct.fees))
    if etf_ok and "SPY" in prices.columns:
        spy = prices["SPY"]
        tr = float(spy.iloc[-1] / spy.iloc[PRE] - 1.0)
        dd = float((spy.iloc[PRE:] / spy.iloc[PRE:].cummax() - 1.0).min())
        out.append(dict(variant="SPY buy&hold", total_return=tr, max_drawdown=dd, bench_return=bench, excess=tr - bench, fills=1, fees=0.0))
    return out


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """rows: one line per (round, fly, variant). Averages over flies inside a round first, then across rounds."""
    per = rows.groupby(["round", "variant"]).agg(total_return=("total_return", "mean"), excess=("excess", "mean"),
                                                 max_drawdown=("max_drawdown", "mean"), fills=("fills", "mean"),
                                                 fees=("fees", "mean")).reset_index()
    old = per[per.variant == "old"].set_index("round")["total_return"]
    out = []
    for v in list(dict.fromkeys(rows["variant"])):                 # keep the order the variants were run in
        g = per[per.variant == v]
        g = g.set_index("round")
        d = (g["total_return"] - old.reindex(g.index)).dropna()
        t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 and d.std(ddof=1) > 0 else np.nan
        out.append(dict(variant=v, rounds=len(g), mean_return=g.total_return.mean(), mean_excess=g.excess.mean(),
                        rounds_beat_bench=int((g.excess > 0).sum()), worst_round=g.total_return.min(),
                        mean_max_drawdown=g.max_drawdown.mean(), mean_fills=g.fills.mean(), mean_fees=g.fees.mean(),
                        vs_old=float(d.mean()) if len(d) else np.nan, t_vs_old=t))
    return pd.DataFrame(out).set_index("variant")


def main():
    from accounts.etfs import mask_before_listing, with_etfs
    from dashboard import state, trades
    from tournament.store import Store

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--etf-root", default=None, help="folder holding prices_etf (default: the panel's folder)")
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
    has_spy = "SPY" in fwd1.columns
    if not has_spy:
        print("note: no SPY prices found (run experiments.fetch_etfs): the ETF variants are skipped", flush=True)
    root = Path(a.root)
    rounds = sorted({int(p.name[1:3]) for p in (root / "rounds" / "scores").glob("r*_*.npz")})
    if a.rounds:
        lo, _, hi = a.rounds.partition("-")
        rounds = [r for r in rounds if int(lo) <= r <= int(hi or lo)]
    rows = []
    for r in rounds:
        ids = sorted({p.name[4:-4] for p in (root / "rounds" / "scores").glob(f"r{r:02d}_*.npz")})
        pick = [ids[int(j * len(ids) / a.flies)] for j in range(min(a.flies, len(ids)))]
        for aid in pick:
            sc = state.load_scores(a.root, r, aid)
            idxs = [i for i in sorted(sc) if i < len(index)]
            if not idxs or idxs[0] < PRE + 1:
                continue
            win = list(range(idxs[0] - PRE, idxs[-1] + 1))
            prices = mask_before_listing(prices_from_returns(fwd1, win), fwd1.iloc[win])
            scdf = pd.DataFrame({index[i][0]: pd.Series(np.asarray(sc[i], float), index=index[i][1]) for i in idxs}).T.sort_index()
            ok = has_spy and prices["SPY"].iloc[PRE:].notna().all() and prices["SPY"].nunique() > 5
            for row in window_rows(prices, scdf, [c for c in stocks if c in prices.columns], k=a.k, etf_ok=bool(ok)):
                rows.append(dict(round=r, fly=aid, **row))
        print(f"round {r} done", flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    s = summarize(df)
    print("\nAcross rounds (returns are per round; t_vs_old pairs each variant with the old policy)\n")
    print(s.round(3).to_string())
    if a.out:
        df.to_csv(f"{a.out}_rows.csv", index=False)
        s.to_csv(f"{a.out}_summary.csv")


if __name__ == "__main__":
    main()
