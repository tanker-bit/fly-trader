"""Train the fly brain with dopamine-gated learning, then test it out-of-sample.

    python -m experiments.train

Walk-forward: learn on the first TRAIN_DAYS, then freeze the synapses and run the
following TEST_DAYS untouched. Compared against an untrained fly and buy-and-hold on
the same test window. Markets are mostly noise: do not expect a big edge.
"""
from __future__ import annotations

import time

import numpy as np

from baselines.buy_and_hold import evaluate
from env.trading_env import TradingEnv, load_prices
from learning.fly_agent import FlyAgent

TRAIN_DAYS = 750
TEST_DAYS = 250
EPOCHS = 3
EPS_START, EPS_END = 0.30, 0.02   # exploration, decays across all training steps
WINDOW = 20


def run_episode(agent: FlyAgent, env: TradingEnv, epsilon_fn=None, learn=False):
    obs, _ = env.reset()
    agent.reset()
    rewards, actions, das = [], [], []
    done, step = False, 0
    while not done:
        eps = epsilon_fn(step) if epsilon_fn else 0.0
        a = agent.act(obs, epsilon=eps)
        obs, r, done, _, info = env.step(a)
        if learn:
            das.append(agent.learn(r))
        rewards.append(r)
        actions.append(a)
        step += 1
    rewards = np.array(rewards)
    return {
        "equity": info["equity"],
        "sharpe": float(rewards.mean() / (rewards.std() + 1e-12) * np.sqrt(252)),
        "actions": np.bincount(actions, minlength=3),
        "mean_da": float(np.mean(das)) if das else 0.0,
    }


def fmt(name, s):
    c = s["actions"]
    return f"{name:<22} equity={s['equity']:.3f}  sharpe={s['sharpe']:+.2f}  sell/hold/buy={c[0]}/{c[1]}/{c[2]}"


def main():
    prices = load_prices("SPY", start="2012-01-01").iloc[-(TRAIN_DAYS + TEST_DAYS + WINDOW + 1):]
    train_env = TradingEnv(prices.iloc[: TRAIN_DAYS + WINDOW + 1])
    test_env = TradingEnv(prices.iloc[TRAIN_DAYS:])

    agent = FlyAgent()
    untrained = FlyAgent()  # same seed -> same wiring, no learning
    total = EPOCHS * TRAIN_DAYS
    eps_fn_factory = lambda base: (lambda s: max(EPS_END, EPS_START - (EPS_START - EPS_END) * (base + s) / total))

    t0 = time.time()
    for ep in range(EPOCHS):
        s = run_episode(agent, train_env, epsilon_fn=eps_fn_factory(ep * TRAIN_DAYS), learn=True)
        print(fmt(f"train epoch {ep + 1}", s) + f"  mean dopamine={s['mean_da']:+.2f}  ({time.time() - t0:.0f}s)")

    print("\n--- out-of-sample test (synapses frozen, no exploration) ---")
    print(fmt("fly (trained)", run_episode(agent, test_env)))
    print(fmt("fly (untrained)", run_episode(untrained, test_env)))
    bh = evaluate(TradingEnv(prices.iloc[TRAIN_DAYS:]), lambda o: 2)
    print(f"{'buy & hold':<22} equity={bh['final_equity']:.3f}  sharpe={bh['sharpe']:+.2f}")


if __name__ == "__main__":
    main()
