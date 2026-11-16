"""Run one fly's $1M long-only account through history.

prices: DataFrame, index = dates, columns = tickers (adjusted close). scores: DataFrame, same layout, the fly's score per
ticker per date (NaN = not scored). Scores may be sparse (e.g. every 5th date): the latest score on or before date t is used. Scores from date t are traded at the price of the NEXT date (exec_lag=1) so a fly can never
trade on a price it has not yet seen. Rebalances every `every` trading days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from accounts.account import Account
from accounts.instruments import Universe, START_CASH
from accounts.policy import rebalance


def run_account(prices: pd.DataFrame, scores: pd.DataFrame, universe: Universe | None = None, k: int = 20, every: int = 21,
                cost_bps: float = 10.0, idle_etf: str | None = None, exec_lag: int = 1, cash: float = START_CASH, sell_rank: int | None = None):
    dates = list(prices.index)
    if universe is None:
        universe = Universe(stocks=[c for c in prices.columns])
    acct = Account(universe, cash=cash, cost_bps=cost_bps)
    curve = []
    for i, d in enumerate(dates):
        px = prices.iloc[i].to_dict()
        if i % every == 0 and i - exec_lag >= 0:
            j = scores.index.searchsorted(dates[i - exec_lag], side="right") - 1     # latest score at or before that date
            if j >= 0:
                rebalance(acct, d, scores.iloc[j], px, k=k, idle_etf=idle_etf, sell_rank=sell_rank)
        curve.append((d, acct.equity(px)))
    eq = pd.Series(dict(curve)).astype(float)
    return acct, eq


def metrics(eq: pd.Series, bench: pd.Series | None = None, periods: int = 252) -> dict:
    r = eq.pct_change().dropna()
    out = dict(total_return=float(eq.iloc[-1] / eq.iloc[0] - 1),
               ann_return=float((eq.iloc[-1] / eq.iloc[0]) ** (periods / max(len(r), 1)) - 1),
               ann_vol=float(r.std() * np.sqrt(periods)),
               max_drawdown=float((eq / eq.cummax() - 1).min()))
    out["sharpe"] = float(r.mean() / r.std() * np.sqrt(periods)) if r.std() > 0 else 0.0
    if bench is not None:
        b = bench.reindex(eq.index).ffill()
        out["bench_return"] = float(b.iloc[-1] / b.iloc[0] - 1)
        out["excess_return"] = out["total_return"] - out["bench_return"]
    return out
