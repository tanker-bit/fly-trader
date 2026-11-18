"""Real trades for the dashboard: run each fly's $1M long-only account over a saved round and turn its actual fills into the
decision stream the scene plays. Read-only; imports the account engine, never writes to league folders.

BUY  = the account really bought that stock (whole shares, 10 bps fee, cash permitting)
SELL = the account really sold shares it owned (never a short, never a call)
HOLD = a position the account kept through that rebalance
SILENT = constant score (dead circuit), as before; no account is run
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from accounts.instruments import Universe
from accounts.league_report import prices_from_returns
from accounts.sim import run_account
from dashboard.state import Decision


def load_fwd1(panel_path, dates) -> pd.DataFrame:
    """Next-day returns, dates x tickers, reindexed to `dates` (the labelled dates the score keys refer to). Only fwd_ret is read."""
    p = str(panel_path)
    df = pd.read_pickle(p) if p.endswith(".pkl") else pd.read_parquet(p, columns=["fwd_ret"])
    return df["fwd_ret"].unstack("ticker").reindex(list(dates))


def _score_of(sc: pd.DataFrame, date, ticker) -> float:
    j = sc.index.searchsorted(pd.Timestamp(date), side="right") - 1
    if j < 0 or ticker not in sc.columns:
        return 0.0
    v = sc.iloc[j].get(ticker)
    return float(v) if v is not None and np.isfinite(v) else 0.0


def replay_trades(scores: dict, index, fwd1: pd.DataFrame, agent: str, rnd: int, k: int = 20, every: int = 21,
                  cost_bps: float = 10.0, sell_rank=None) -> list:
    """{date_idx: score array} -> chronological [Decision] from the account's real fills (BUY/SELL) plus a HOLD per rebalance."""
    idxs = [i for i in sorted(scores) if i < len(index)]
    if not idxs:
        return []
    if all(np.nanstd(scores[i]) < 1e-9 for i in idxs):
        return [Decision(str(index[i][0].date()), "", "SILENT", float(np.nanmean(scores[i])), agent, rnd) for i in idxs]
    win = list(range(idxs[0], idxs[-1] + 1))
    prices = prices_from_returns(fwd1, win)
    sc = pd.DataFrame({index[i][0]: pd.Series(np.asarray(scores[i], float), index=index[i][1]) for i in idxs}).T.sort_index()
    acct, _ = run_account(prices, sc, Universe(stocks=prices.columns), k=k, every=every, cost_bps=cost_bps, sell_rank=sell_rank)
    out, held, by_day = [], {}, {}
    for f in acct.filled:
        by_day.setdefault(pd.Timestamp(f.date), []).append(f)
    for day in sorted(by_day):
        traded = set()
        for f in by_day[day]:
            held[f.ticker] = held.get(f.ticker, 0.0) + (f.shares if f.side == "BUY" else -f.shares)
            if held[f.ticker] <= 1e-9:
                held.pop(f.ticker, None)
            traded.add(f.ticker)
            out.append(Decision(str(day.date()), f.ticker, f.side, _score_of(sc, day, f.ticker), agent, rnd))
        keep = [t for t in held if t not in traded]
        if keep:
            t = max(keep, key=lambda x: _score_of(sc, day, x))
            out.append(Decision(str(day.date()), t, "HOLD", _score_of(sc, day, t), agent, rnd))
    return out
