"""Turns a fly's scores into long-only orders on an Account.

Top-K equal weight: own the K highest-scored stocks, sell anything that falls out, never short. Sells run before buys so the
cash is there. Whatever cannot be invested (rounding to whole shares, the fee reserve) is parked in `idle_etf` if given.
"""
from __future__ import annotations

import math

import pandas as pd


def pick_top_k(scores: pd.Series, universe, k: int) -> list[str]:
    """K best-scored tradable stocks (ties and NaN scores dropped)."""
    s = pd.Series(scores).dropna()
    s = s[[universe.kind(t) == "STOCK" for t in s.index]]
    return list(s.sort_values(ascending=False, kind="mergesort").index[:k])


def rebalance(acct, date, scores, prices, k: int = 20, idle_etf: str | None = None, reserve: float = 0.002, band: float = 0.25,
              sell_rank: int | None = None):
    """One rebalance. `prices` is a ticker->price mapping for today. Returns the list of Fills made.

    sell_rank: a stock already held is kept while its score rank is within sell_rank (default 2*k); new money goes to the best-ranked
        names not yet held until k are owned. sell_rank=k gives the plain top-K with no hysteresis. This cuts turnover and fees.
    band: leave a holding alone if it is within +-band of its target value (saves fees on tiny trims).
    reserve: fraction of equity held back so fees never make a buy fail.
    """
    start = len(acct.log)
    acct.mark(prices)
    sell_rank = 2 * k if sell_rank is None else max(sell_rank, k)
    ranked = [t for t in pick_top_k(scores, acct.universe, len(pd.Series(scores).dropna())) if _ok(prices.get(t))]
    pos_rank = {t: i for i, t in enumerate(ranked)}
    kept = [t for t in ranked if t in acct.pos and pos_rank[t] < sell_rank][:k]
    want = kept + [t for t in ranked if t not in kept][: max(k - len(kept), 0)]
    equity = acct.equity()
    invest = equity * (1.0 - reserve)
    per = invest / max(len(want), 1) if want else 0.0
    keep = set(want) | ({idle_etf.upper()} if idle_etf else set())

    # 1) sell everything we no longer want, then trim big overweights
    for t in list(acct.pos):
        if t in keep:
            continue
        if _ok(prices.get(t)):
            acct.sell_all(date, t, prices[t])
    for t in want:
        px = prices[t]
        held_val = acct.pos.get(t, 0.0) * px
        if held_val > per * (1 + band):
            acct.sell_shares(date, t, math.floor((held_val - per) / px), px) if (held_val - per) >= px else None

    # 2) buy the shortfall in the wanted names
    for t in want:
        px = prices[t]
        gap = per - acct.pos.get(t, 0.0) * px
        if gap > per * band and gap >= px:
            acct.buy_dollars(date, t, min(gap, acct.cash), px)

    # 3) park leftover cash in the idle ETF (or release it if cash is needed: handled next rebalance)
    if idle_etf and _ok(prices.get(idle_etf)):
        spare = acct.cash - equity * reserve
        if spare > prices[idle_etf]:
            acct.buy_dollars(date, idle_etf, spare, prices[idle_etf])
    return acct.log[start:]


def _ok(p) -> bool:
    try:
        return math.isfinite(float(p)) and float(p) > 0
    except (TypeError, ValueError):
        return False
