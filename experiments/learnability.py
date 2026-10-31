"""Learnability test: can the real-connectome fly learn a pattern that we KNOW is there?

    python -m experiments.learnability

Prices are synthetic, with planted momentum (today's return partly predicts tomorrow's).
No internet needed. Compares the trained fly with an untrained fly, random trading, a simple
momentum rule (the practical ceiling for this pattern) and buy-and-hold.

If the trained fly beats the untrained one here, the dopamine learning mechanism works and
the problem on real markets is data/features. If it doesn't, the mechanism needs fixing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from baselines.buy_and_hold import evaluate as bh_eval
from env.trading_env import TradingEnv
from experiments.evaluate import RandomAgent
from experiments.train import run_episode
from learning.fly_agent import FlyAgent

PHI = 0.30            # strength of the planted momentum
TRAIN_DAYS, TEST_DAYS, WINDOW = 1000, 500, 20
EPOCHS = 2
SEEDS = [0, 1, 2, 3, 4]
EPS_START, EPS_END = 0.30, 0.02


class MomentumRule:
    """Long if yesterday was up, short if down. Ceiling for the planted pattern."""
    def reset(self): pass
    def act(self, obs, epsilon=0.0): return 2 if obs[-2] > 0 else 0


def make_prices(seed: int, n: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = PHI * r[i - 1] + rng.normal(0.0003, 0.01)
    return pd.Series(100 * np.exp(np.cumsum(r)))


def main():
    total_days = TRAIN_DAYS + TEST_DAYS + WINDOW + 1
    res = {k: [] for k in ("fly trained", "fly untrained", "random", "momentum rule")}
    bh = []
    for seed in SEEDS:
        prices = make_prices(100 + seed, total_days)
        train_env = TradingEnv(prices.iloc[: TRAIN_DAYS + WINDOW + 1])
        test_prices = prices.iloc[TRAIN_DAYS:]
        test_env = TradingEnv(test_prices)
        bh.append(bh_eval(TradingEnv(test_prices), lambda o: 2)["sharpe"])

        trained = FlyAgent(seed=seed)
        n_steps = EPOCHS * TRAIN_DAYS
        for ep in range(EPOCHS):
            fn = lambda s, b=ep * TRAIN_DAYS: max(EPS_END, EPS_START - (EPS_START - EPS_END) * (b + s) / n_steps)
            run_episode(trained, train_env, epsilon_fn=fn, learn=True)

        agents = {"fly trained": trained, "fly untrained": FlyAgent(seed=seed),
                  "random": RandomAgent(seed), "momentum rule": MomentumRule()}
        for name, ag in agents.items():
            res[name].append(run_episode(ag, test_env)["sharpe"])
        print(f"seed {seed} done", flush=True)

    print(f"\nPlanted momentum phi={PHI}; {len(SEEDS)} seeds; test = {TEST_DAYS} days (Sharpe)")
    for name, v in res.items():
        print(f"{name:<16} {np.mean(v):+.2f} +/- {np.std(v):.2f}")
    print(f"{'buy & hold':<16} {np.mean(bh):+.2f}")


if __name__ == "__main__":
    main()
