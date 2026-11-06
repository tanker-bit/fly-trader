"""Stage 0: evolve flies on synthetic markets with a planted pattern.

    python -m experiments.stage0_evolve --quick                  # smoke test, a few minutes
    python -m experiments.stage0_evolve --pattern linear         # one pattern
    python -m experiments.stage0_evolve --workers 10             # full ladder, parallel (Windows: ok)

Patterns (see evolution/synthetic.py): none, linear, sparse2, xor. 'none' is the overfitting
detector: there is nothing to find, so any 'winner' on validation is luck and must fail the holdout.
A fly only counts as winning if its HOLDOUT IC beats the null (random), the untrained fly and the
default genome, and its validation IC cleared the luck bar for the number of flies tried.
Results are written to results/stage0_<pattern>.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from evolution.evaluate import DEFAULT_CFG, holdout_report, null_bar
from evolution.evolve import evolve
from evolution.synthetic import PATTERNS, make_splits


def run_pattern(pattern, a):
    cfg = dict(DEFAULT_CFG)
    n_train, n_val, n_hold = (300, 200, 600) if a.quick else (a.n_train, a.n_val, a.n_hold)
    splits = make_splits(pattern, seed=a.seed, n_train=n_train, n_val=n_val, n_hold=n_hold, snr=a.snr)
    print(f"\n=== pattern: {pattern}  (snr {a.snr}, train {n_train}, val {n_val} x2, holdout {n_hold}) ===")
    res = evolve(splits, cfg, pop=a.pop, gens=a.gens, workers=a.workers, seed=a.seed)
    rep = holdout_report(res["champion"], res["default"], splits, cfg, seed=a.seed)

    replay = rep.pop("_replay_val_fitness")
    if abs(replay - res["champion_fitness"]) > 1e-9:
        print(f"WARNING: champion replay gave val IC {replay:+.3f}, evolution recorded "
              f"{res['champion_fitness']:+.3f}; the holdout numbers below are not for the same fly")
    se = 1.0 / np.sqrt(n_hold)
    bar = null_bar(res["n_trials"], res["n_val_total"])
    print(f"\nchampion genome: {res['champion'].to_dict()}")
    print(f"champion validation IC {res['champion_fitness']:+.3f}   luck bar after {res['n_trials']} flies: {bar:.3f}"
          f"   -> {'cleared' if res['champion_fitness'] > bar else 'NOT cleared'}")
    print(f"\nHOLDOUT IC (rank correlation with next-day return; noise ~ +/-{1.96 * se:.3f})")
    for k, v in sorted(rep.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<26}{v:+.3f}")
    gap = res["champion_fitness"] - rep["fly champion"]
    print(f"selection bias (validation IC minus holdout IC of the champion): {gap:+.3f}")

    os.makedirs("results", exist_ok=True)
    out = {"pattern": pattern, "snr": a.snr, "champion": res["champion"].to_dict(),
           "champion_val_ic": res["champion_val_ic"], "luck_bar": bar, "n_trials": res["n_trials"],
           "holdout_ic": rep, "history": res["history"]}
    with open(f"results/stage0_{pattern}.json", "w") as f:
        json.dump(out, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="all", choices=("all",) + PATTERNS)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--snr", type=float, default=0.3, help="learnable signal vs noise (0.3 -> IC about 0.29)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-hold", type=int, default=2000)
    ap.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    a = ap.parse_args()
    if a.quick:
        a.pop, a.gens = min(a.pop, 6), min(a.gens, 2)
    for p in (PATTERNS if a.pattern == "all" else (a.pattern,)):
        run_pattern(p, a)


if __name__ == "__main__":
    main()
