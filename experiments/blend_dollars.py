"""Dollar test of the blends (A2, part 2): do fly + fundamentals make money in a $1M long-only account?

    python -m experiments.blend_dollars --root league_v7 --first-round 4 --workers 4 --out blenddollars

Same account engine and rules as experiments.rules_dollars (top-K, whole shares, 10 bps, sell only what it owns, slow rebalance), run
over each league validation window with the blend's scores. Only stocks with real fundamentals can be bought (all others score NEVER),
so every variant picks from the same universe. The parts are the ones of experiments.blend_test: fly (committee of every fly that
scored the round), ridge (fitted on that round's training dates), fund (value+quality). Blends are equal-weight averages of z-scores,
nothing fitted. Compared with the equal-weight benchmark of the same stocks and with random picks. Nothing is trained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from accounts.league_report import account_report
from experiments.blend_test import components_for_day
from experiments.rules_dollars import NEVER, summarize
from learning.ridge import fit_ridge

VARIANTS = {"fly": ("fly",), "ridge": ("ridge",), "fund": ("fund",), "fly+fund": ("fly", "fund"),
            "ridge+fund": ("ridge", "fund"), "fly+ridge+fund": ("fly", "ridge", "fund")}


def blend_score_dicts(data, r, dicts, score_every: int, variants=VARIANTS) -> dict:
    """{variant: {date_idx: score array aligned with data.day[i][0]}}; stocks without real fundamentals score NEVER; unusable days are
    left out (the account keeps its last scores)."""
    c, v0, v1 = data.round_geometry(r)
    ridge = fit_ridge(*data.stack_train(c))
    out = {v: {} for v in variants}
    for i in range(v0, v1, score_every):
        got = components_for_day(data, i, dicts, ridge)
        if got is None:
            continue
        comps, _, real = got
        for v, parts in variants.items():
            if not all(p in comps for p in parts):
                continue
            s = np.full(len(real), NEVER)
            s[real] = np.mean([comps[p] for p in parts], axis=0)
            out[v][i] = s
    return {v: d for v, d in out.items() if d}


def round_blend_accounts(data, r, dicts, score_every, variants=VARIANTS, k=20, every=63, cost_bps=10.0, sell_rank=None) -> pd.DataFrame:
    """One row per variant: total_return, bench_return, excess_return, max_drawdown, sharpe, trades, fees."""
    c, v0, v1 = data.round_geometry(r)
    rows = {}
    for v, sc in blend_score_dicts(data, r, dicts, score_every, variants).items():
        _, _, m = account_report(data, sc, v0, v1, k=k, every=every, cost_bps=cost_bps, sell_rank=sell_rank)
        rows[v] = {x: m[x] for x in ["total_return", "bench_return", "excess_return", "max_drawdown", "sharpe", "trades", "fees"]}
    return pd.DataFrame(rows).T


def main():
    import argparse
    import time

    from accounts.league_report import random_control
    from experiments.blend_test import load_flies
    from tournament.data import Data
    from tournament.store import Store

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="league_data")
    ap.add_argument("--first-round", type=int, default=4)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--every", type=int, default=63)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--sell-rank", type=int, default=None)
    ap.add_argument("--controls", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None, help="CSV prefix")
    a = ap.parse_args()
    store = Store(a.root)
    cfg = store.load_config()
    print("loading panel...", flush=True)
    data = Data.from_cfg(cfg)
    rounds = [r for r in range(a.first_round, 200) if data.round_valid(r)
              and any((store.root / "rounds" / "scores").glob(f"r{r:02d}_*.npz"))]
    flies = load_flies(store, rounds)
    rk = dict(k=a.k, every=a.every, cost_bps=a.cost_bps, sell_rank=a.sell_rank)
    tot, exc, fees, rnd_mean = {}, {}, {}, {}
    for r in rounds:
        t0 = time.time()
        c, v0, v1 = data.round_geometry(r)
        t = round_blend_accounts(data, r, flies[r], cfg["score_every"], **rk)
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
    print("\nExcess return vs the equal-weight benchmark per round\n")
    print(exc.round(3).to_string())
    print("\nAcross rounds (t is optimistic: long windows overlap; the same 8-ish variants were looked at)\n")
    print(summarize(exc, tot, rm).round(3).to_string())
    print(f"\nmean fees per round: {fees.mean().round(0).to_dict()}")
    if a.out:
        tot.to_csv(f"{a.out}_total.csv")
        exc.to_csv(f"{a.out}_excess.csv")


if __name__ == "__main__":
    main()
