"""Dollar test of plain rules (T5 step 6, part 1: core).

Each rule runs a $1M long-only account (top-K, whole shares, 10 bps, sell only what it owns, slow rebalance) over each league
validation window, using the same account engine as the flies. Fundamentals rules score only stocks whose fundamentals are real
(fund_known > 0); the others get a score so low they are never bought. Nothing is trained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from accounts.league_report import account_report
from experiments.fund_rules import RULES as FUND_RULES, rule_score

NEVER = -1e9
DOLLAR_RULES = {**{k: FUND_RULES[k] for k in ["fcf_yield", "-accruals", "roe", "value+quality"]},
                "reversal -ma50_dist": ["-ma50_dist"], "momentum 12-1": ["+mom_12_1"]}


def rule_score_dict(data, terms, idxs, need_fundamentals=True) -> dict:
    """{date_idx: score array aligned with data.day[i][0]}. Rows without real fundamentals score NEVER (when required)."""
    fk = data.feats.index("fund_known") if need_fundamentals else None
    out = {}
    for i in idxs:
        X = data.day[i][1]
        if need_fundamentals:
            real = X[:, fk] > 0
            s = np.full(len(X), NEVER)
            if real.sum() >= 20:
                s[real] = rule_score(data, X[real], terms)
        else:
            s = rule_score(data, X, terms)
        out[i] = s
    return out


def round_accounts(data, r, score_every, rules=DOLLAR_RULES, k=20, every=63, cost_bps=10.0, sell_rank=None) -> pd.DataFrame:
    """One row per rule: total_return, bench_return, excess_return, max_drawdown, sharpe, trades, fees for round r."""
    c, v0, v1 = data.round_geometry(r)
    idxs = list(range(v0, v1, score_every))
    rows = {}
    for name, terms in rules.items():
        need = not name.startswith(("reversal", "momentum"))
        sc = rule_score_dict(data, terms, idxs, need_fundamentals=need)
        _, _, m = account_report(data, sc, v0, v1, k=k, every=every, cost_bps=cost_bps, sell_rank=sell_rank)
        rows[name] = {x: m[x] for x in ["total_return", "bench_return", "excess_return", "max_drawdown", "sharpe", "trades", "fees"]}
    return pd.DataFrame(rows).T


def summarize(excess: pd.DataFrame, total: pd.DataFrame, rand_mean: pd.Series) -> pd.DataFrame:
    """Across rounds. excess/total: rounds x rules. rand_mean: random-pick mean return per round (or NaN)."""
    vs_rand = total.sub(rand_mean, axis=0)
    n = excess.count()
    return pd.DataFrame({"mean_return": total.mean(), "mean_excess": excess.mean(), "rounds_beat_bench": (excess > 0).sum(),
                         "mean_vs_random": vs_rand.mean(), "rounds_beat_random": (vs_rand > 0).sum(), "rounds": n,
                         "t_vs_random": vs_rand.mean() / (vs_rand.std(ddof=1) / np.sqrt(n))})


def main():
    import argparse
    import time

    from accounts.league_report import random_control
    from tournament.data import Data
    from tournament.store import Store

    ap = argparse.ArgumentParser(description="$1M long-only accounts for plain rules on the league's rounds, vs benchmark and random picks.")
    ap.add_argument("--root", default="league_data")
    ap.add_argument("--first-round", type=int, default=4, help="fundamentals are real only from about 2010 (round 4 on the current league)")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--every", type=int, default=63, help="rebalance every N days (slow, to match the long horizon)")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--sell-rank", type=int, default=None)
    ap.add_argument("--controls", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None, help="optional CSV prefix")
    a = ap.parse_args()
    store = Store(a.root)
    cfg = store.load_config()
    print("loading panel...", flush=True)
    data = Data.from_cfg(cfg)
    rounds = [r for r in range(a.first_round, 200) if data.round_valid(r)]
    rk = dict(k=a.k, every=a.every, cost_bps=a.cost_bps, sell_rank=a.sell_rank)
    tot, exc, fees, rnd_mean = {}, {}, {}, {}
    for r in rounds:
        t0 = time.time()
        c, v0, v1 = data.round_geometry(r)
        t = round_accounts(data, r, cfg["score_every"], **rk)
        tot[r], exc[r], fees[r] = t["total_return"], t["excess_return"], t["fees"]
        if a.controls > 0:
            shape = {i: np.zeros(len(data.day[i][0])) for i in range(v0, v1, cfg["score_every"])}
            rnd_mean[r] = float(random_control(data, shape, v0, v1, n=a.controls, workers=a.workers, **rk).mean())
        else:
            rnd_mean[r] = np.nan
        print(f"round {r} done ({time.time() - t0:.0f}s)", flush=True)
    tot, exc, fees = pd.DataFrame(tot).T, pd.DataFrame(exc).T, pd.DataFrame(fees).T
    rm = pd.Series(rnd_mean)
    pd.set_option("display.width", 220)
    print(f"\nTotal return per round (k={a.k}, rebalance every {a.every}d, {a.cost_bps:g} bps)\n")
    print(pd.concat([tot, rm.rename("random picks")], axis=1).round(3).to_string())
    print("\nAcross rounds (t is optimistic: long windows overlap)\n")
    print(summarize(exc, tot, rm).round(3).to_string())
    print(f"\nmean fees per round: {fees.mean().round(0).to_dict()}")
    if a.out:
        tot.to_csv(f"{a.out}_total.csv"); exc.to_csv(f"{a.out}_excess.csv")


if __name__ == "__main__":
    main()
