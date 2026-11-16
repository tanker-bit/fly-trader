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


_C = {}


def _init_ctl(prices, shapes, kw):
    _C.update(prices=prices, shapes=shapes, kw=kw)


def _ctl_job(seed):
    """Total return of one random-score fly: same account rules, scores are pure noise."""
    rng = np.random.default_rng(seed)
    rows = {d: pd.Series(rng.normal(size=len(t)), index=t) for d, t in _C["shapes"]}
    sc = pd.DataFrame(rows).T.sort_index()
    prices, kw = _C["prices"], _C["kw"]
    _, eq = run_account(prices, sc, Universe(stocks=prices.columns), **kw)
    return float(eq.iloc[-1] / eq.iloc[0] - 1)


def random_control(data, score_dict, v0, v1, n=200, workers=1, seed0=1000, **kw):
    """Returns of n random-score accounts over the same window and rules (shape of the scores = the fly's saved score dates)."""
    idxs = list(range(v0, v1))
    prices = prices_from_returns(data.fwd1, idxs)
    shapes = [(data.dates[i], data.day[i][0]) for i in sorted(score_dict)]
    seeds = list(range(seed0, seed0 + n))
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers, initializer=_init_ctl, initargs=(prices, shapes, kw)) as pool:
            return np.array(pool.map(_ctl_job, seeds, chunksize=max(1, n // (workers * 4))))
    _init_ctl(prices, shapes, kw)
    return np.array([_ctl_job(s) for s in seeds])


def momentum_scores(data, score_dict):
    """12-1 momentum as a 'fly': score = the panel's mom_12_1 feature on the same dates."""
    return {i: np.nan_to_num(data.day[i][1][:, data.i_mom].astype(float)) for i in score_dict}


def main():
    from tournament.data import Data, rank_ic
    from tournament.league import committee_scores
    from tournament.store import Store
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="league_data")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--every", type=int, default=21)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--sell-rank", type=int, default=None, help="keep a held stock while its rank is within this (default 2*k; = k means no hysteresis)")
    ap.add_argument("--controls", type=int, default=200, help="number of random-score control accounts (0 = skip)")
    ap.add_argument("--workers", type=int, default=1, help="processes for the random controls")
    a = ap.parse_args()
    store = Store(a.root)
    cfg = store.load_config()
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
    rk = dict(k=a.k, every=a.every, cost_bps=a.cost_bps, sell_rank=a.sell_rank)
    if a.controls > 0:
        any_fly = next(d for aid, d in dicts.items() if aid != "COMMITTEE")
        t1 = time.time()
        print(f"\nrunning {a.controls} random-score control accounts on {a.workers} worker(s)...", flush=True)
        rnd = random_control(data, any_fly, v0, v1, n=a.controls, workers=a.workers, **rk)
        _, _, mm = account_report(data, momentum_scores(data, any_fly), v0, v1, a.k, a.every, a.cost_bps, sell_rank=a.sell_rank)
        print(f"controls done in {time.time() - t1:.0f}s")
        p5, p50, p95 = np.percentile(rnd, [5, 50, 95])
        bench = float(df["bench_return"].iloc[0])
        print(f"\nrandom picks, same rules ({len(rnd)} draws): mean {rnd.mean():+.3f}  median {p50:+.3f}  5%..95% [{p5:+.3f}, {p95:+.3f}]"
              f"   benchmark {bench:+.3f}  beat benchmark: {(rnd > bench).mean():.0%} of draws")
        print(f"momentum 12-1 top-{a.k}, same rules:      {mm['total_return']:+.3f}   (fees {mm['fees']:,.0f})\n")
        df["pct_vs_random"] = [float((rnd < r).mean()) for r in df["total_return"]]
        print("share of random draws each fly beats (0.50 = typical blind pick; 0.95+ = clearly better):")
        print(df[["total_return", "pct_vs_random"]].to_string())
    out = store.root / "rounds" / f"accounts_r{a.round:02d}.csv"
    df.to_csv(out)
    print(f"\nsaved {out}\nbenchmark = equal-weight average of the same stocks (panel has no SPY). Excess return is vs that.")


if __name__ == "__main__":
    main()
