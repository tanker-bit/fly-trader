"""Honest evaluation: many seeds, several rolling test windows, real baselines.

    python -m experiments.evaluate

For each rolling window (train on TRAIN_DAYS, test on the next TEST_DAYS) and each seed:
  - fly (trained)    dopamine-gated learning on the train span, then frozen
  - fly (untrained)  same wiring, no learning
  - random           uniformly random actions
and buy-and-hold once per window. Prints mean +/- std of test Sharpe and the share of
runs that beat buy-and-hold, so one lucky window or seed can't fool us.
"""
from __future__ import annotations

import time

import numpy as np

from baselines.buy_and_hold import evaluate as bh_eval
from env.trading_env import TradingEnv, load_prices
from experiments.train import run_episode
from learning.fly_agent import FlyAgent

TRAIN_DAYS = 500
TEST_DAYS = 250
STEP_DAYS = 250        # roll forward by one test window
N_WINDOWS = 3
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 1             # one pass: avoids replaying (memorizing) the same days
EPS_START, EPS_END = 0.30, 0.02
WINDOW = 20


class RandomAgent:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, obs, epsilon=0.0):
        return int(self.rng.integers(3))


def train_agent(agent, env):
    total = EPOCHS * TRAIN_DAYS
    for ep in range(EPOCHS):
        fn = lambda s, b=ep * TRAIN_DAYS: max(EPS_END, EPS_START - (EPS_START - EPS_END) * (b + s) / total)
        run_episode(agent, env, epsilon_fn=fn, learn=True)


def summarize(name, sharpes, equities, bh_sharpes):
    s, e = np.array(sharpes), np.array(equities)
    beat = np.mean(e > np.array(bh_sharpes["equity"]))
    print(f"{name:<18} sharpe {s.mean():+.2f} +/- {s.std():.2f}   equity {e.mean():.3f}   beats buy&hold: {beat:.0%}")


def main():
    span = TRAIN_DAYS + TEST_DAYS + (N_WINDOWS - 1) * STEP_DAYS + WINDOW + 1
    prices = load_prices("SPY", start="2008-01-01").iloc[-span:]
    results = {k: ([], []) for k in ("fly trained", "fly untrained", "random")}
    bh = {"sharpe": [], "equity": []}
    t0 = time.time()

    for w in range(N_WINDOWS):
        a = w * STEP_DAYS
        train_env = TradingEnv(prices.iloc[a : a + TRAIN_DAYS + WINDOW + 1])
        test_prices = prices.iloc[a + TRAIN_DAYS : a + TRAIN_DAYS + TEST_DAYS + WINDOW + 1]
        test_env = TradingEnv(test_prices)
        b = bh_eval(TradingEnv(test_prices), lambda o: 2)
        for seed in SEEDS:
            agents = {
                "fly trained": FlyAgent(seed=seed),
                "fly untrained": FlyAgent(seed=seed),
                "random": RandomAgent(seed),
            }
            train_agent(agents["fly trained"], train_env)
            for name, ag in agents.items():
                r = run_episode(ag, test_env)
                results[name][0].append(r["sharpe"])
                results[name][1].append(r["equity"])
            bh["sharpe"].append(b["sharpe"])
            bh["equity"].append(b["final_equity"])
        print(f"window {w + 1}/{N_WINDOWS} done ({time.time() - t0:.0f}s)  buy&hold sharpe={b['sharpe']:+.2f}")

    n = len(SEEDS) * N_WINDOWS
    print(f"\n{n} runs per strategy, test window = {TEST_DAYS} days")
    for name, (sh, eq) in results.items():
        summarize(name, sh, eq, bh)
    print(f"{'buy & hold':<18} sharpe {np.mean(bh['sharpe']):+.2f}   equity {np.mean(bh['equity']):.3f}")


if __name__ == "__main__":
    main()
