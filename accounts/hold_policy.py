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
  * keeps a CORE of broad ETFs (core, core_weight) and enforces a hard floor: broad-market / multi-sector ETFs are at least core_floor
    (30%) of the account at all times once trading starts. Checked every day; if the core slips below the floor (stocks rallied, the ETF
    fell) it is topped back up to core_weight from spare cash first and, if there is none, by trimming the largest stocks. The rest of
    the account (up to 70%) is the stock picks.
  * short-term sleeve (tactical_k, tactical_weight): a few quick positions with NO profit target and NO fixed exit date. The fly decides:
    a sleeve name is sold when the fly's own ranking of it falls below tactical_sell_rank (checked on every fill day). Only a downside
    stop (tactical_stop) limits losses. Structural limits stay: 30% core floor, sleeve cap, long-only.
Everything is long-only, whole shares, via accounts.account.Account (which refuses shorts, options, inverse and leveraged ETFs).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from accounts.instruments import is_core
from accounts.policy import _ok

# Ready-made cores: (ticker, share of the core). Any broad / multi-sector ETF from accounts.instruments.CORE_ETFS can be used.
CORES = {
    "spy": (("SPY", 1.0),),
    "broad": (("VTI", 0.5), ("RSP", 0.25), ("QQQ", 0.25)),
    "defensive": (("SPY", 0.5), ("XLV", 0.25), ("XLP", 0.25)),
    "dividend": (("VTI", 0.5), ("SCHD", 0.5)),
}


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
    core: tuple = ()                 # ((ticker, share), ...) the broad ETFs that make up the core; () = no core
    core_weight: float = 0.35        # target share of equity in the core (a buffer above the floor, so small drifts do not trigger trades)
    core_floor: float = 0.30         # hard minimum share of equity in the core, enforced every day
    core_band: float = 0.03          # a core member is only topped up when it is this far (share of equity) below its target
    reserve: float = 0.002           # equity held back so fees never make a buy fail
    lt_days: int = 252               # a stock held this long counts as a long-term holding ...
    lt_sell_rank: int | None = None  # ... and is only rank-sold when its rank is worse than this (None = same as sell_rank)
    tactical_k: int = 0              # short-term sleeve: at most this many quick trades at once (0 = no sleeve)
    tactical_weight: float = 0.10    # share of equity the whole sleeve may use (each slot gets an equal part)
    tactical_sell_rank: int = 40     # the fly itself decides: a sleeve name is sold once the fly ranks it worse than this (no clock, no profit cap)
    tactical_stop: float | None = 0.08   # downside risk limit only (None = off); there is NO profit ceiling and NO time limit


# The mix the dashboard and the audit use: medium/long-term stock book + a 35% broad core + a small short-term sleeve.
MIX = HoldParams(core=CORES["broad"], lt_sell_rank=100, tactical_k=3, tactical_weight=0.05)


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
        self.tact: dict[str, int] = {}       # tactical (short-term) names -> index of the day bought
        self.started = False                 # the floor applies once the first rebalance has run

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
        if self.started:
            self._floor(acct, i, date, px)
        if mode and scores is not None:
            self.started = True
            self._rebalance(acct, i, date, px, scores, sells=(mode == "full"))
        for t in list(self.tact):
            if t not in acct.pos:
                self.tact.pop(t)
        for t in list(self.entry):
            if t not in acct.pos:
                self.entry.pop(t)
                self.peak.pop(t, None)
        for t in acct.pos:
            self.entry.setdefault(t, i)
            if _ok(px.get(t)):
                self.peak[t] = max(self.peak.get(t, 0.0), float(px[t]))

    def _stops(self, acct, i, date, px):
        p = self.p
        for t in list(acct.pos):
            if t not in self.tact or not _ok(px.get(t)):
                continue
            avg = acct.avg_cost.get(t, 0.0)
            gain = px[t] / avg - 1.0 if avg > 0 else 0.0
            if p.tactical_stop is not None and gain <= -p.tactical_stop:
                acct.sell_all(date, t, px[t])
                self.until[t] = i + p.cooldown
        if self.p.stop_loss is None and self.p.trail is None:
            return
        for t in list(acct.pos):
            if t in self.tact or acct.universe.kind(t) != "STOCK" or not _ok(px.get(t)):
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
        for t in [t for t in acct.pos if acct.universe.kind(t) == "STOCK" and t not in self.tact]:
            if not _ok(px.get(t)):
                continue
            held_days = i - self.entry.get(t, i)
            limit = p.lt_sell_rank if (p.lt_sell_rank and held_days >= p.lt_days) else p.sell_rank     # long-term holdings are kept longer
            if sells and rank.get(t, 10 ** 9) >= limit and held_days >= p.min_hold:
                acct.sell_all(date, t, px[t])
            elif p.max_weight is not None and acct.pos[t] * px[t] > p.max_weight * equity:
                n = math.floor((acct.pos[t] * px[t] - p.trim_to * equity) / px[t])
                if n >= 1:
                    acct.sell_shares(date, t, n, px[t])
        equity = acct.equity()

        # 1b) short-term sleeve: no fixed exit date and no profit target. The fly decides: sold when its own ranking turns against the name
        for t in [t for t in self.tact if t in acct.pos]:
            if _ok(px.get(t)) and rank.get(t, 10 ** 9) >= p.tactical_sell_rank:
                acct.sell_all(date, t, px[t])
        equity = acct.equity()

        # 2) fill the empty slots with the best-ranked names that pass the guards; never touch names already owned
        owned = [t for t in acct.pos if acct.universe.kind(t) == "STOCK" and t not in self.tact]
        slots = max(p.k - len(owned), 0)
        cw = p.core_weight if self._core_members(px) else 0.0           # no tradable core ETF today -> the stocks may use the whole account
        tw = p.tactical_weight if p.tactical_k > 0 else 0.0
        per = equity * (1.0 - p.reserve) * (1.0 - cw - tw) / max(p.k, 1)
        for t in ranked:
            if slots <= 0:
                break
            if t in acct.pos or self.blocked(i, t):
                continue
            budget = min(per, acct.cash - equity * p.reserve)
            if budget >= float(px[t]) * (1.0 + acct.cost) and acct.buy_dollars(date, t, budget, px[t]).status == "FILLED":
                slots -= 1                                              # too little cash for one share -> no order at all

        # 2b) short-term sleeve: the next-best names (the long book already took the best); it exits when the fly's rank for the name falls
        if p.tactical_k > 0:
            tslots = p.tactical_k - len(self.tact)
            tper = equity * (1.0 - p.reserve) * tw / p.tactical_k
            for t in ranked:
                if tslots <= 0:
                    break
                if t in acct.pos or self.blocked(i, t):
                    continue
                budget = min(tper, acct.cash - equity * p.reserve)
                if budget >= float(px[t]) * (1.0 + acct.cost) and acct.buy_dollars(date, t, budget, px[t]).status == "FILLED":
                    self.tact[t] = i
                    tslots -= 1

        # 3) core ETFs: top up to the target with whatever cash is left
        self._core_buy(acct, date, px, acct.equity())

    # ------------------------------------------------------------------ core ETFs
    def _core_members(self, px):
        """[(ticker, share)] of the core members that can be traded today (valid price), shares renormalised to sum to 1."""
        mem = [(t.upper(), w) for t, w in self.p.core if is_core(t) and _ok(px.get(t.upper())) and w > 0]
        tot = sum(w for _, w in mem)
        return [(t, w / tot) for t, w in mem] if tot > 0 else []

    def core_value(self, acct, px) -> float:
        return sum(acct.pos[t] * float(px[t]) for t in acct.pos if is_core(t) and _ok(px.get(t)))

    def _core_buy(self, acct, date, px, equity):
        p = self.p
        mem = self._core_members(px)
        if not mem or p.core_weight <= 0:
            return
        for t, w in mem:
            price = float(px[t])
            diff = p.core_weight * equity * w - acct.pos.get(t, 0.0) * price
            if diff <= p.core_band * equity * w:
                continue
            budget = min(diff, acct.cash - equity * p.reserve)
            if budget >= price * (1.0 + acct.cost):                  # no cash -> no order (nothing is rejected)
                acct.buy_dollars(date, t, budget, price)

    def _floor(self, acct, i, date, px):
        """Daily check of the hard floor. Below it: restore the core to core_weight from spare cash, then by trimming the biggest stocks."""
        p = self.p
        if not p.core or p.core_floor <= 0 or not self._core_members(px):
            return
        equity = acct.equity()
        have = self.core_value(acct, px)
        if equity <= 0 or have >= p.core_floor * equity:
            return
        need = p.core_weight * equity - have
        short = need - (acct.cash - equity * p.reserve)
        if short > 0:
            stocks = sorted([t for t in acct.pos if acct.universe.kind(t) == "STOCK" and _ok(px.get(t))],
                            key=lambda t: -acct.pos[t] * px[t])
            for t in stocks:
                if short <= 0:
                    break
                n = min(acct.pos[t], math.ceil(short / (float(px[t]) * (1.0 - acct.cost))))
                if n >= 1:
                    f = acct.sell_shares(date, t, n, px[t])
                    short -= f.shares * float(px[t]) * (1.0 - acct.cost) if f.status == "FILLED" else 0.0
        self._core_buy(acct, date, px, acct.equity())
