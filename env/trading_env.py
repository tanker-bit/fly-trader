"""Gymnasium trading environment with transaction costs.

Observation: window of recent log returns (+ current position).
Action: 0 = short/flat target -1 ... mapped: 0 -> -1, 1 -> 0, 2 -> +1 (target position).
Reward: position * next return - cost * |change in position|  (can be used as dopamine).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


def load_prices(ticker: str = "SPY", start: str = "2015-01-01", end: str | None = None) -> pd.Series:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, prices: pd.Series, window: int = 20, cost: float = 0.0005, allow_short: bool = True):
        super().__init__()
        self.returns = np.log(prices).diff().dropna().values.astype(np.float32)
        self.window = window
        self.cost = cost
        self.allow_short = allow_short
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(window + 1,), dtype=np.float32)
        self.reset()

    def _obs(self):
        w = self.returns[self.t - self.window : self.t]
        return np.append(w, self.position).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window
        self.position = 0.0
        self.equity = 1.0
        return self._obs(), {}

    def step(self, action: int):
        target = {0: -1.0 if self.allow_short else 0.0, 1: 0.0, 2: 1.0}[int(action)]
        trade_cost = self.cost * abs(target - self.position)
        r = target * float(self.returns[self.t]) - trade_cost
        self.equity *= float(np.exp(r))
        self.position = target
        self.t += 1
        terminated = self.t >= len(self.returns)
        obs = self._obs() if not terminated else np.zeros(self.window + 1, dtype=np.float32)
        return obs, r, terminated, False, {"equity": self.equity}
