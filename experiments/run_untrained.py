"""Run the untrained fly brain through the trading env (pipeline sanity check).

    python -m experiments.run_untrained

Prints equity / Sharpe next to buy-and-hold on the same window, the action mix, and basic
network activity stats. No learning happens here.
"""
from __future__ import annotations

import time

import numpy as np

from baselines.buy_and_hold import evaluate
from env.trading_env import TradingEnv, load_prices
from learning.fly_agent import FlyAgent

DAYS = 500  # most recent trading days; the spiking sim is slow


def main():
    prices = load_prices("SPY", start="2015-01-01").iloc[-(DAYS + 21):]
    env = TradingEnv(prices)
    agent = FlyAgent()
    print(f"KC={len(agent.kc)}  MBON={len(agent.mbon)}  DAN={len(agent.dan)}  "
          f"active KCs/step={agent.n_winners}")

    obs, _ = env.reset()
    agent.reset()
    rewards, actions, kc_active, mbon_rate = [], [], [], []
    done, t0 = False, time.time()
    while not done:
        a = agent.act(obs)
        obs, r, done, _, info = env.step(a)
        rewards.append(r)
        actions.append(a)
        kc_active.append(int((agent.last_pre > 0).sum()))
        mbon_rate.append(agent.last_rates.mean())
    rewards = np.array(rewards)
    sharpe = rewards.mean() / (rewards.std() + 1e-12) * np.sqrt(252)
    counts = np.bincount(actions, minlength=3)

    print(f"\nfly (untrained): equity={info['equity']:.3f}  sharpe={sharpe:.2f}  "
          f"({len(rewards)} steps, {time.time() - t0:.0f}s)")
    print(f"actions: sell={counts[0]}  hold={counts[1]}  buy={counts[2]}")
    print(f"mean KCs spiking/step={np.mean(kc_active):.0f}   mean MBON rate={np.mean(mbon_rate):.1f} Hz")
    print("buy & hold, same window:", evaluate(TradingEnv(prices), lambda o: 2))


if __name__ == "__main__":
    main()
