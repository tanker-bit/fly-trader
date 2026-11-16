"""A long-only cash account.

The rules, enforced here and nowhere else, so no policy can break them:
  * starts with START_CASH (1,000,000) and never owes money: cash >= 0 after every trade (no margin, no leverage);
  * BUY needs enough cash for shares * price + fee;
  * SELL can only sell shares the account holds (no shorting, no naked anything);
  * only instruments in the Universe can be traded (stocks and approved plain ETFs; no options, no inverse/leveraged ETFs);
  * whole shares unless fractional=True; a trading cost of cost_bps of the traded value is charged on every fill.
A trade that breaks a rule is REJECTED (logged, account unchanged) rather than raising, so a buggy policy cannot crash a league
run; pass strict=True to turn rejections into RuleViolation errors while developing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

from accounts.instruments import START_CASH, Universe

EPS = 1e-9


class RuleViolation(Exception):
    pass


@dataclass(frozen=True)
class Fill:
    date: str
    ticker: str
    side: str            # BUY | SELL
    shares: float
    price: float
    fee: float
    status: str          # FILLED | REJECTED
    reason: str = ""
    cash_after: float = 0.0
    realized: float = 0.0   # profit realised by a SELL, before fees


class Account:
    def __init__(self, universe: Universe, cash: float = START_CASH, cost_bps: float = 10.0, fractional: bool = False,
                 strict: bool = False):
        self.universe = universe
        self.start_cash = float(cash)
        self.cash = float(cash)
        self.cost = float(cost_bps) / 1e4
        self.fractional = fractional
        self.strict = strict
        self.pos: dict[str, float] = {}          # ticker -> shares (> 0 only)
        self.avg_cost: dict[str, float] = {}     # ticker -> average purchase price (fees excluded)
        self.last_px: dict[str, float] = {}      # last price seen per ticker (for marking when a price is missing)
        self.realized = 0.0                      # realised trading profit, before fees
        self.fees = 0.0
        self.log: list[Fill] = []

    # ---------------------------------------------------------------- internals
    def _reject(self, date, t, side, shares, price, reason) -> Fill:
        f = Fill(str(date), str(t).upper(), side, float(shares) if _num(shares) else 0.0, float(price) if _num(price) else 0.0,
                 0.0, "REJECTED", reason, self.cash)
        self.log.append(f)
        if self.strict:
            raise RuleViolation(f"{side} {t}: {reason}")
        return f

    def _check_common(self, date, t, side, shares, price):
        t = str(t).upper()
        if not _num(price) or price <= 0:
            return self._reject(date, t, side, shares, price, "bad price")
        if not _num(shares) or shares <= 0:
            return self._reject(date, t, side, shares, price, "shares must be positive")
        if not self.fractional and abs(shares - round(shares)) > EPS:
            return self._reject(date, t, side, shares, price, "whole shares only")
        why = self.universe.check(t)
        if why:
            return self._reject(date, t, side, shares, price, why)
        return None

    # ---------------------------------------------------------------- trading
    def buy_shares(self, date, ticker, shares, price) -> Fill:
        t = str(ticker).upper()
        bad = self._check_common(date, t, "BUY", shares, price)
        if bad:
            return bad
        shares = float(round(shares)) if not self.fractional else float(shares)
        value = shares * price
        fee = value * self.cost
        if value + fee > self.cash + EPS:
            return self._reject(date, t, "BUY", shares, price, "not enough cash")
        old = self.pos.get(t, 0.0)
        self.avg_cost[t] = (self.avg_cost.get(t, 0.0) * old + value) / (old + shares)
        self.pos[t] = old + shares
        self.cash -= value + fee
        self.fees += fee
        self.last_px[t] = float(price)
        f = Fill(str(date), t, "BUY", shares, float(price), fee, "FILLED", "", self.cash)
        self.log.append(f)
        return f

    def buy_dollars(self, date, ticker, dollars, price) -> Fill:
        """Spend up to `dollars` (fee included) on whole shares. Rejected if not even one share fits."""
        if not _num(price) or price <= 0 or not _num(dollars) or dollars <= 0:
            return self._reject(date, ticker, "BUY", 0.0, price, "bad price" if not _num(price) or price <= 0 else "bad amount")
        per = price * (1.0 + self.cost)
        shares = dollars / per if self.fractional else math.floor(dollars / per + EPS)
        if shares <= 0:
            return self._reject(date, ticker, "BUY", 0.0, price, "amount below one share")
        return self.buy_shares(date, ticker, shares, price)

    def sell_shares(self, date, ticker, shares, price) -> Fill:
        t = str(ticker).upper()
        bad = self._check_common(date, t, "SELL", shares, price)
        if bad:
            return bad
        shares = float(round(shares)) if not self.fractional else float(shares)
        held = self.pos.get(t, 0.0)
        if held <= EPS:
            return self._reject(date, t, "SELL", shares, price, "nothing held: shorting is not allowed")
        if shares > held + EPS:
            return self._reject(date, t, "SELL", shares, price, "cannot sell more than held: shorting is not allowed")
        shares = min(shares, held)
        value = shares * price
        fee = value * self.cost
        realized = (price - self.avg_cost[t]) * shares
        self._reduce(t, shares)
        self.cash += value - fee
        self.fees += fee
        self.realized += realized
        self.last_px[t] = float(price)
        f = Fill(str(date), t, "SELL", shares, float(price), fee, "FILLED", "", self.cash, realized)
        self.log.append(f)
        return f

    def sell_all(self, date, ticker, price) -> Fill:
        held = self.pos.get(str(ticker).upper(), 0.0)
        return self.sell_shares(date, ticker, held if held > 0 else 0.0, price)

    def delist(self, date, ticker, price: float = 0.0) -> Fill | None:
        """A holding stops trading: it is cashed out at `price` (0 = bankruptcy, worthless) with no fee. No-op if not held."""
        t = str(ticker).upper()
        held = self.pos.get(t, 0.0)
        if held <= EPS:
            return None
        price = max(float(price), 0.0) if _num(price) else 0.0
        realized = (price - self.avg_cost[t]) * held
        self._reduce(t, held)
        self.cash += held * price
        self.realized += realized
        f = Fill(str(date), t, "SELL", held, price, 0.0, "FILLED", "delisted", self.cash, realized)
        self.log.append(f)
        return f

    def _reduce(self, t, shares):
        left = self.pos[t] - shares
        if left <= EPS:
            self.pos.pop(t, None)
            self.avg_cost.pop(t, None)
        else:
            self.pos[t] = left

    # ---------------------------------------------------------------- valuation
    def mark(self, prices) -> None:
        """Remember today's prices (ignores missing / non-positive / NaN ones: those tickers keep their last price)."""
        for t, p in prices.items():
            if _num(p) and p > 0:
                self.last_px[str(t).upper()] = float(p)

    def equity(self, prices=None) -> float:
        if prices:
            self.mark(prices)
        return self.cash + sum(sh * self.last_px[t] for t, sh in self.pos.items())

    def stale(self, prices) -> list[str]:
        """Held tickers with no usable price today (they are valued at their last known price)."""
        return [t for t in self.pos if not (_num(prices.get(t)) and prices.get(t) > 0)]

    def weights(self, prices=None) -> dict[str, float]:
        eq = self.equity(prices)
        w = {t: sh * self.last_px[t] / eq for t, sh in self.pos.items()} if eq > 0 else {}
        w["CASH"] = self.cash / eq if eq > 0 else 1.0
        return w

    def unrealized(self, prices=None) -> float:
        if prices:
            self.mark(prices)
        return sum((self.last_px[t] - self.avg_cost[t]) * sh for t, sh in self.pos.items())

    @property
    def filled(self) -> list[Fill]:
        return [f for f in self.log if f.status == "FILLED"]

    @property
    def rejected(self) -> list[Fill]:
        return [f for f in self.log if f.status == "REJECTED"]

    # ---------------------------------------------------------------- saving
    def to_dict(self) -> dict:
        return dict(start_cash=self.start_cash, cash=self.cash, cost=self.cost, fractional=self.fractional, pos=dict(self.pos),
                    avg_cost=dict(self.avg_cost), last_px=dict(self.last_px), realized=self.realized, fees=self.fees,
                    log=[asdict(f) for f in self.log])

    @classmethod
    def from_dict(cls, d: dict, universe: Universe, strict: bool = False) -> "Account":
        a = cls(universe, d["start_cash"], d["cost"] * 1e4, d.get("fractional", False), strict)
        a.cash, a.pos, a.avg_cost, a.last_px = float(d["cash"]), dict(d["pos"]), dict(d["avg_cost"]), dict(d["last_px"])
        a.realized, a.fees = float(d["realized"]), float(d["fees"])
        a.log = [Fill(**f) for f in d["log"]]
        return a


def _num(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False
