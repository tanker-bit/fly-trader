"""Give every league fly a $1,000,000 long-only account and report real portfolio results next to IC.

    python -m accounts.league_report --root league_data --round 3          # every fly alive that round + the committee
    python -m accounts.league_report --root league_data --round 3 --k 20 --every 21 --cost-bps 10

Uses the scores the league already saved (rounds/scores/rXX_<id>.npz), so nothing is retrained. Each fly starts the round's
validation window with $1M cash, buys the top-K scored stocks, sells only what it owns, and is marked to market daily.
Prices are rebuilt from the panel's next-day returns (fwd_ret), so no price download is needed. The benchmark is the equal-weight
average of the same stock universe (the panel has no SPY), rebalanced daily and cost-free.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from accounts.instruments import Universe, START_CASH
from accounts.sim import run_account, metrics


def prices_from_returns(fwd1: pd.DataFrame, idxs) -> pd.DataFrame:
    """Price paths (start 100) for dates[idxs] from next-day returns: price[t+1] = price[t] * (1 + fwd_ret[t]).
    A ticker with no return on a date has no price that day (the account keeps its last known price)."""
    r = fwd1.iloc[list(idxs)]
    px = 100.0 * (1.0 + r.fillna(0.0)).cumprod().shift(1).fillna(1.0)
    return px.where(r.notna().shift(1, fill_value=True))


def scores_frame(score_dict: dict, data, idxs) -> pd.DataFrame:
    """{date_idx: score array aligned with data.day[i][0]} -> DataFrame(date x ticker)."""
    rows = {data.dates[i]: pd.Series(np.asarray(v, float), index=data.day[i][0]) for i, v in score_dict.items() if i in set(idxs)}
    return pd.DataFrame(rows).T.sort_index()


def equal_weight_benchmark(prices: pd.DataFrame) -> pd.Series:
    r = prices.pct_change().mean(axis=1).fillna(0.0)
    return START_CASH * (1 + r).cumprod()


def account_report(data, score_dict: dict, v0: int, v1: int, k=20, every=21, cost_bps=10.0, idle_etf=None, sell_rank=None):
    idxs = list(range(v0, v1))
    prices = prices_from_returns(data.fwd1, idxs)
    sc = scores_frame(score_dict, data, idxs)
    uni = Universe(stocks=prices.columns)
    acct, eq = run_account(prices, sc, uni, k=k, every=every, cost_bps=cost_bps, idle_etf=idle_etf, sell_rank=sell_rank)
    m = metrics(eq, equal_weight_benchmark(prices))
    m.update(trades=len(acct.filled), rejected=len(acct.rejected), fees=acct.fees, final_equity=float(eq.iloc[-1]),
             positions=len(acct.pos))
    return acct, eq, m


def main():
    from tournament.data import Data, rank_ic
    from tournament.league import committee_scores
    from tournament.store import Store

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="league_data")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--every", type=int, default=21)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--sell-rank", type=int, default=None, help="keep a held stock while its rank is within this (default 2*k; = k means no hysteresis)")
    a = ap.parse_args()
    store = Store(a.root)
    cfg = store.load_config()
    import time
    t0 = time.time()
    print("loading panel (same slow start as run_league, runs once)...", flush=True)
    data = Data(cfg["panel"], cfg["train_days"], cfg["val_days"], cfg["holdout"])
    print(f"panel loaded in {time.time() - t0:.0f}s", flush=True)
    c, v0, v1 = data.round_geometry(a.round)
    print(f"round {a.round}: {data.dates[v0].date()} -> {data.dates[v1 - 1].date()}  "
          f"(each fly: ${START_CASH:,.0f}, long-only, top {a.k}, rebalance every {a.every} days, {a.cost_bps:g} bps, sell rank {a.sell_rank or 2 * a.k})\n")
    dicts, rows = {}, []
    for aid in store.agent_ids():
        f = store.root / "rounds" / "scores" / f"r{a.round:02d}_{aid}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        dicts[aid] = {int(n[1:]): z[n].astype(float) for n in z.files}
    if not dicts:
        raise SystemExit(f"no saved scores for round {a.round} in {store.root / 'rounds' / 'scores'}")
    dicts["COMMITTEE"] = committee_scores(list(dicts.values()))
    for n, (aid, d) in enumerate(dicts.items(), 1):
        print(f"  account {n}/{len(dicts)}: {aid}", flush=True)
        _, _, m = account_report(data, d, v0, v1, a.k, a.every, a.cost_bps, sell_rank=a.sell_rank)
        m["id"] = aid
        m["ic"] = float(np.mean([rank_ic(d[i], data.day[i][2]) for i in d]))
        rows.append(m)
    df = pd.DataFrame(rows).set_index("id").sort_values("excess_return", ascending=False)
    cols = ["ic", "final_equity", "total_return", "bench_return", "excess_return", "max_drawdown", "sharpe", "trades", "fees"]
    pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x:,.3f}")
    print(df[cols].to_string())
    out = store.root / "rounds" / f"accounts_r{a.round:02d}.csv"
    df.to_csv(out)
    print(f"\nsaved {out}\nbenchmark = equal-weight average of the same stocks (panel has no SPY). Excess return is vs that.")


if __name__ == "__main__":
    main()
