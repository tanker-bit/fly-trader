"""Sensitivity test: how weak can the planted pattern get before the fly stops learning it?

    python -m experiments.sensitivity

Repeats the learnability test at several momentum strengths (phi). Real daily stock returns
have almost none (roughly 0.0-0.05), so the realistic question is where the fly's edge
disappears. phi=0 is a control: with no pattern, trained and untrained should look the same.
Prints Sharpe (mean +/- std over seeds) for the trained fly, untrained fly and momentum rule.
"""
from __future__ import annotations

import numpy as np

from env.trading_env import TradingEnv
from experiments.learnability import (
    EPOCHS, EPS_END, EPS_START, TEST_DAYS, TRAIN_DAYS, WINDOW, MomentumRule,
)
from experiments.train import run_episode
from learning.fly_agent import FlyAgent
import pandas as pd

PHIS = [0.0, 0.03, 0.06, 0.10, 0.20]
SEEDS = [0, 1, 2]


def make_prices(phi: float, seed: int, n: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + rng.normal(0.0003, 0.01)
    return pd.Series(100 * np.exp(np.cumsum(r)))


def main():
    n = TRAIN_DAYS + TEST_DAYS + WINDOW + 1
    print(f"{'phi':>5}  {'fly trained':>14}  {'fly untrained':>14}  {'momentum rule':>14}  gap(trained-untrained)")
    for phi in PHIS:
        tr, un, mo = [], [], []
        for seed in SEEDS:
            prices = make_prices(phi, 200 + seed, n)
            train_env = TradingEnv(prices.iloc[: TRAIN_DAYS + WINDOW + 1])
            test_env = TradingEnv(prices.iloc[TRAIN_DAYS:])
            trained = FlyAgent(seed=seed)
            steps = EPOCHS * TRAIN_DAYS
            for ep in range(EPOCHS):
                fn = lambda s, b=ep * TRAIN_DAYS: max(EPS_END, EPS_START - (EPS_START - EPS_END) * (b + s) / steps)
                run_episode(trained, train_env, epsilon_fn=fn, learn=True)
            tr.append(run_episode(trained, test_env)["sharpe"])
            un.append(run_episode(FlyAgent(seed=seed), test_env)["sharpe"])
            mo.append(run_episode(MomentumRule(), test_env)["sharpe"])
        f = lambda v: f"{np.mean(v):+.2f} +/- {np.std(v):.2f}"
        print(f"{phi:>5}  {f(tr):>14}  {f(un):>14}  {f(mo):>14}  {np.mean(tr) - np.mean(un):+.2f}", flush=True)


if __name__ == "__main__":
    main()
