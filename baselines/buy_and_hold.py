"""Buy-and-hold baseline: always long. Run: python -m baselines.buy_and_hold"""
from __future__ import annotations

import numpy as np

from env.trading_env import TradingEnv, load_prices


def evaluate(env: TradingEnv, policy) -> dict:
    obs, _ = env.reset()
    rewards, done = [], False
    while not done:
        obs, r, done, _, info = env.step(policy(obs))
        rewards.append(r)
    rewards = np.array(rewards)
    sharpe = rewards.mean() / (rewards.std() + 1e-12) * np.sqrt(252)
    return {"final_equity": info["equity"], "sharpe": float(sharpe), "steps": len(rewards)}


if __name__ == "__main__":
    prices = load_prices("SPY", start="2015-01-01")
    env = TradingEnv(prices)
    print("buy & hold:", evaluate(env, lambda obs: 2))
