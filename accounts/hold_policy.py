"""A calmer long-only policy: buy the best-ranked stocks, then HOLD them.

Compared with accounts.policy.rebalance (which re-equalises every holding on every rebalance), this one
  * rebalances rarely (default every 63 trading days) and never trims or tops up a name it already owns;
  * sells a stock for rank reasons only after a minimum holding period (min_hold), and only when its rank has fallen out of sell_rank;
  * cuts a loss early: a stop-loss from the purchase price AND a trailing stop from the highest price since purchase (so a big winner
    cannot ride all the way back down), checked every day, then a cooldown before the name may be bought again;
  * caps concentration: a holding above max_weight of equity is trimmed back to trim_to (winners still run, but not to 40% of the account);
  * refills empty slots every fill_every days instead of waiting for the next full rebalance, so stopped-out cash does not sit idle;
  * does not BUY a stock in free fall (63-day return below -knife) or with a suspicious one-day move (|ret| > jump in the last 60
    days: bad data, splits, delisting stubs). Names already owned are never force-sold by these two filters;
  * can keep a core ETF sleeve (etf, etf_weight) next to the stocks.
Everything is long-only, whole shares, via accounts.account.Account (which refuses shorts, options, inverse and leveraged ETFs).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from accounts.policy import _ok


@dataclass(frozen=True)
class HoldParams:
    k: int = 20
    every: int = 63                  # trading days between rebalances
    sell_rank: int = 60              # an owned stock is only a sell candidate once its rank is worse than this
    min_hold: int = 63               # trading days an owned stock is protected from rank-based selling
    stop_loss: float | None = 0.25   # sell when the price is this far below the average cost (None = off)
    trail: float | None = 0.30       # sell when the price is this far below its highest close since purchase (None = off)
    cooldown: int = 63               # trading days before a stopped-out name may be bought again
    max_weight: float | None = 0.12  # trim a holding that grows beyond this share of equity (None = never trim)
    trim_to: float = 0.08            # ... back down to this share
    fill_every: int = 21             # refill empty slots this often (full rank-based selling still happens every `every` days)
    knife: float | None = 0.30       # do not buy when the 63-day return is below -knife (None = off)
    jump: float | None = 0.35        # do not buy when any 1-day move in the last 60 days exceeds this (None = off)
    etf: str | None = None           # core ETF held next to the stocks (e.g. "SPY")
    etf_weight: float = 0.0          # its target share of equity
    etf_band: float = 0.03           # only trade the ETF when it is off target by more than this share of equity
    reserve: float = 0.002           # equity held back so fees never make a buy fail


class HoldPolicy:
    def __init__(self, params: HoldParams, prices: pd.DataFrame):
        self.p = params
        r1 = prices.pct_change(fill_method=None)
        self.jump = r1.abs().rolling(60, min_periods=1).max()
        base = prices.shift(63)
        if len(prices):
            base.iloc[:63] = prices.iloc[0].to_numpy()          # early days: measure from the first day we have
        self.ret63 = prices / base - 1.0
        self.entry: dict[str, int] = {}      # ticker -> index of the day it was first bought
        self.until: dict[str, int] = {}      # ticker -> index until which it may not be bought again
        self.peak: dict[str, float] = {}     # ticker -> highest close since it was bought

    # ------------------------------------------------------------------ guards
    def blocked(self, i: int, t: str) -> bool:
        """True if a NEW buy of t on day i must be skipped. Uses data up to the day before (what the fly could have seen)."""
        if self.until.get(t, -1) > i:
            return True
        j = i - 1
        if j < 0 or t not in self.jump.columns:
            return False
        if self.p.jump is not None:
            v = self.jump.iloc[j][t]
            if pd.notna(v) and v > self.p.jump:
                return True
        if self.p.knife is not None:
            v = self.ret63.iloc[j][t]
            if pd.notna(v) and v < -self.p.knife:
                return True
        return False

    # ------------------------------------------------------------------ daily
    def day(self, acct, i: int, date, px: dict, scores: pd.Series | None, mode: str | None):
        """mode: 'full' = rank-based selling + refilling + ETF sleeve; 'fill' = refill empty slots + ETF sleeve only; None = nothing."""
        acct.mark(px)
        self._stops(acct, i, date, px)
        if mode and scores is not None:
            self._rebalance(acct, i, date, px, scores, sells=(mode == "full"))
        for t in list(self.entry):
            if t not in acct.pos:
                self.entry.pop(t)
                self.peak.pop(t, None)
        for t in acct.pos:
            self.entry.setdefault(t, i)
            if _ok(px.get(t)):
                self.peak[t] = max(self.peak.get(t, 0.0), float(px[t]))

    def _stops(self, acct, i, date, px):
        if self.p.stop_loss is None and self.p.trail is None:
            return
        for t in list(acct.pos):
            if acct.universe.kind(t) != "STOCK" or not _ok(px.get(t)):
                continue
            avg = acct.avg_cost.get(t, 0.0)
            hit = self.p.stop_loss is not None and avg > 0 and px[t] / avg - 1.0 <= -self.p.stop_loss
            pk = self.peak.get(t, 0.0)
            hit = hit or (self.p.trail is not None and pk > 0 and px[t] / pk - 1.0 <= -self.p.trail)
            if hit:
                acct.sell_all(date, t, px[t])
                self.until[t] = i + self.p.cooldown

    # ------------------------------------------------------------------ rebalance
    def _rebalance(self, acct, i, date, px, scores, sells: bool = True):
        p = self.p
        s = pd.Series(scores).dropna()
        ranked = [t for t in s.sort_values(ascending=False, kind="mergesort").index
                  if acct.universe.kind(t) == "STOCK" and _ok(px.get(t))]
        rank = {t: n for n, t in enumerate(ranked)}
        equity = acct.equity()

        # 1) sell owned stocks that fell out of the ranking (only once held long enough); trim any that grew too big
        for t in [t for t in acct.pos if acct.universe.kind(t) == "STOCK"]:
            if not _ok(px.get(t)):
                continue
            held_days = i - self.entry.get(t, i)
            if sells and rank.get(t, 10 ** 9) >= p.sell_rank and held_days >= p.min_hold:
                acct.sell_all(date, t, px[t])
            elif p.max_weight is not None and acct.pos[t] * px[t] > p.max_weight * equity:
                n = math.floor((acct.pos[t] * px[t] - p.trim_to * equity) / px[t])
                if n >= 1:
                    acct.sell_shares(date, t, n, px[t])
        equity = acct.equity()
        self._etf(acct, date, px, equity, sells_only=True)          # an ETF trim frees cash before the stock buys

        # 2) fill the empty slots with the best-ranked names that pass the guards; never touch names already owned
        owned = [t for t in acct.pos if acct.universe.kind(t) == "STOCK"]
        slots = max(p.k - len(owned), 0)
        per = equity * (1.0 - p.reserve) * (1.0 - p.etf_weight) / max(p.k, 1)
        for t in ranked:
            if slots <= 0:
                break
            if t in acct.pos or self.blocked(i, t):
                continue
            budget = min(per, acct.cash - equity * p.reserve)
            if budget > 0 and acct.buy_dollars(date, t, budget, px[t]).status == "FILLED":
                slots -= 1

        # 3) ETF sleeve: buy it up to target with whatever cash is left (never by selling a stock)
        self._etf(acct, date, px, acct.equity(), sells_only=False)

    def _etf(self, acct, date, px, equity, sells_only: bool):
        p = self.p
        if not (p.etf and _ok(px.get(p.etf)) and p.etf_weight > 0):
            return
        etf, price = p.etf.upper(), float(px[p.etf])
        diff = p.etf_weight * equity - acct.pos.get(etf, 0.0) * price
        if abs(diff) <= p.etf_band * equity:
            return
        if diff < 0 and sells_only:
            n = math.floor(-diff / price)
            if n >= 1:
                acct.sell_shares(date, etf, n, price)
        elif diff > 0 and not sells_only:
            budget = min(diff, acct.cash - equity * p.reserve)
            if budget >= price * (1.0 + acct.cost):                  # no cash -> no order (nothing is liquidated, nothing is rejected)
                acct.buy_dollars(date, etf, budget, price)
