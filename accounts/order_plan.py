"""V4c: turn today's fly scores and the real game account into a list of orders for a person to review. Places nothing.

    plan = plan_day(prices, scores, Holdings(cash=100_000.0), params=MIX, state=load_state(path), buyable=offered)
    plan.orders        # [Order(side, symbol, shares, est_price, est_value, reason, rank, score), ...]  sells first
    plan.state         # the policy memory AFTER these orders; save it only once they have really been placed

How: a shadow accounts.account.Account is seeded with the game's cash, shares and purchase prices, accounts.hold_policy runs ONE day on
it (stops and the 30% ETF floor every day; refills / full rebalances when accounts.policy_state.mode_today says they are due), and every
simulated fill becomes an Order with the reason it happened. All account rules still apply (long-only, cash only, whole shares, approved
ETFs only). The game charges no commission, so the shadow account charges none.

prices: dates x tickers, ending on the day the scores were made (adjusted closes; on the last day they equal the real close). It must
cover every stock that can be bought, the core ETFs and the current holdings, with about 64+ days of history for the policy's guards.
scores: one number per stock, higher = better (tournament.live COMMITTEE column). Ranks are taken among the stocks that can be bought.
buyable: tickers the game offers (None = no limit). Held names can always be sold. Orders are sized at the last close; real market
orders fill at a different price, so the list is an estimate. Symbols are in our (Yahoo) form, e.g. BRK-B; the game form is BRK.B.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from accounts.account import Account
from accounts.hold_policy import MIX, HoldParams, HoldPolicy
from accounts.instruments import FORBIDDEN_ETFS, Universe, is_core
from accounts.policy import _ok
from accounts.policy_state import PolicyState, capture, mode_today, restore


@dataclass
class Holdings:
    """The game account as it is now. shares and avg_cost use our ticker form."""
    cash: float
    shares: dict = field(default_factory=dict)         # ticker -> whole shares (> 0)
    avg_cost: dict = field(default_factory=dict)       # ticker -> average purchase price (missing -> today's price)
    prices: dict = field(default_factory=dict)         # ticker -> the game's current price (used where our data has none)


@dataclass
class Order:
    side: str                  # BUY | SELL
    symbol: str
    shares: int
    est_price: float
    est_value: float
    reason: str
    rank: int | None = None    # 1 = the stock the flies like most among the buyable ones today
    score: float | None = None


@dataclass
class Plan:
    asof: pd.Timestamp
    mode: str | None           # 'full' | 'fill' | None (only stops and the ETF floor ran)
    orders: list
    state: PolicyState
    before: dict
    after: dict
    notes: list = field(default_factory=list)


def _snapshot(acct, pol, px) -> dict:
    eq = acct.equity()
    core = pol.core_value(acct, px)
    return dict(cash=acct.cash, equity=eq, positions={t: int(round(s)) for t, s in sorted(acct.pos.items())},
                core_share=core / eq if eq > 0 else 0.0)


def _ranking(scores: pd.Series, acct, px) -> dict:
    """1-based rank among stocks with a usable price, the same order the policy uses."""
    s = scores.dropna()
    order = [t for t in s.sort_values(ascending=False, kind="mergesort").index if acct.universe.kind(t) == "STOCK" and _ok(px.get(t))]
    return {t: n + 1 for n, t in enumerate(order)}


def _sell_reason(t, f, p: HoldParams, pre, rank, held_days, px) -> str:
    full = f.shares >= pre["pos"].get(t, 0) - 1e-9
    avg = pre["avg"].get(t, 0.0)
    gain = px / avg - 1.0 if avg > 0 else 0.0
    r = rank.get(t)
    rtxt = f"rank {r}" if r else "not ranked today"
    if not full:
        w = pre["pos"].get(t, 0) * px / pre["equity"] if pre["equity"] > 0 else 0.0
        if p.max_weight is not None and w > p.max_weight:
            return f"trim: {w:.1%} of the account, back to {p.trim_to:.0%}"
        return f"trim: raise cash to keep the ETF core at {p.core_floor:.0%}"
    if t in pre["tact"]:
        if p.tactical_stop is not None and gain <= -p.tactical_stop:
            return f"short-term sleeve stop: {gain:+.1%} since purchase"
        return f"short-term sleeve exit: fly {rtxt} (sell when worse than {p.tactical_sell_rank})"
    if p.stop_loss is not None and avg > 0 and gain <= -p.stop_loss:
        return f"stop-loss: {gain:+.1%} vs average cost"
    pk = pre["peak"].get(t, 0.0)
    if p.trail is not None and pk > 0 and px / pk - 1.0 <= -p.trail:
        return f"trailing stop: {px / pk - 1.0:+.1%} from its high since purchase"
    limit = p.lt_sell_rank if (p.lt_sell_rank and held_days >= p.lt_days) else p.sell_rank
    return f"rank fell: {rtxt} (sell when worse than {limit}; held {held_days} trading days)"


def _guard(pol, i, t) -> str:
    """Which of HoldPolicy.blocked's rules stops a new buy of t today."""
    if pol.until.get(t, -1) > i:
        return f"cooldown after a stop, {pol.until[t] - i} more trading days"
    j = i - 1
    if pol.p.jump is not None and j >= 0 and t in pol.jump.columns:
        v = pol.jump.iloc[j][t]
        if pd.notna(v) and v > pol.p.jump:
            return f"a one-day move of {v:.0%} in the last 60 days"
    if pol.p.knife is not None and j >= 0 and t in pol.ret63.columns:
        v = pol.ret63.iloc[j][t]
        if pd.notna(v) and v < -pol.p.knife:
            return f"down {-v:.0%} over 63 trading days"
    return "blocked"


def plan_day(prices: pd.DataFrame, scores: pd.Series, holdings: Holdings, params: HoldParams = MIX,
             state: PolicyState | None = None, buyable=None) -> Plan:
    if prices.empty:
        raise ValueError("no prices")
    prices = prices.sort_index()
    i = len(prices) - 1
    asof = prices.index[i]
    notes = []
    held = {str(t).upper(): int(q) for t, q in holdings.shares.items() if int(q) > 0}

    s = pd.Series(scores, dtype=float).dropna()
    s.index = [str(t).upper() for t in s.index]
    if buyable is not None:
        ok = {str(t).upper() for t in buyable}
        s = s[[t in ok or t in held for t in s.index]]
    universe = Universe(stocks=set(s.index) | {t for t in held if t not in FORBIDDEN_ETFS})
    for t in held:
        if universe.check(t):
            notes.append(f"{t} is held but cannot be traded by these rules ({universe.check(t)}): left alone")

    px = {str(k).upper(): v for k, v in prices.iloc[i].to_dict().items()}
    for t in held:
        if not _ok(px.get(t)):
            g = holdings.prices.get(t)
            if _ok(g):
                px[t] = float(g)
                notes.append(f"{t}: no price in our data on {asof.date()}; using the game's {float(g):.2f}")
            else:
                notes.append(f"{t}: no price anywhere; it is valued at its purchase price and cannot be traded today")
    missing_core = [t for t, _ in params.core if not _ok(px.get(t.upper()))]
    if missing_core:
        notes.append(f"no price for core ETF(s) {missing_core} on {asof.date()}: they cannot be bought today")

    acct = Account(universe, cash=float(holdings.cash), cost_bps=0.0)
    for t, q in held.items():
        acct.pos[t] = float(q)
        acct.avg_cost[t] = float(holdings.avg_cost.get(t) or px.get(t) or 0.0)
        acct.last_px[t] = float(px[t]) if _ok(px.get(t)) else acct.avg_cost[t]

    pol = HoldPolicy(params, prices)
    restore(state, pol, prices.index, i)
    if state is None and held:
        notes.append("no saved policy state: current holdings are treated as bought today (protected from rank sells for "
                     f"{params.min_hold} trading days)")
    mode = mode_today(state, prices.index, i, params)
    rank = _ranking(s, acct, px)
    pre = dict(pos=dict(acct.pos), avg=dict(acct.avg_cost), peak=dict(pol.peak), tact=set(pol.tact), equity=acct.equity(),
               entry=dict(pol.entry))
    before = _snapshot(acct, pol, px)
    n0 = len(acct.log)

    if mode and not len(s):
        notes.append("no stock scores could be used today: only the ETF core was managed")
    pol.day(acct, i, asof, px, s if mode else None, mode)
    ran = mode

    orders = []
    for f in acct.log[n0:]:
        if f.status != "FILLED":
            notes.append(f"{f.side} {f.ticker} refused by the account rules: {f.reason}")
            continue
        t = f.ticker
        if f.side == "SELL":
            held_days = i - pre["entry"].get(t, i)
            why = _sell_reason(t, f, params, pre, rank, held_days, f.price)
        elif is_core(t) or universe.kind(t) == "ETF":
            why = f"ETF core: top up to the {params.core_weight:.0%} target (floor {params.core_floor:.0%})"
        elif t in pol.tact and t not in pre["tact"]:
            why = f"short-term sleeve: rank {rank.get(t)}"
        else:
            why = f"new holding: rank {rank.get(t)}"
        orders.append(Order(f.side, t, int(round(f.shares)), float(f.price), float(f.shares * f.price), why, rank.get(t),
                            float(s[t]) if t in s.index else None))
    orders.sort(key=lambda o: 0 if o.side == "SELL" else 1)        # sells first: their cash pays for the buys

    bought = [o.rank for o in orders if o.side == "BUY" and o.rank]
    if bought:
        skipped = [t for t, r in sorted(rank.items(), key=lambda x: x[1])
                   if r < max(bought) and t not in pre["pos"] and t not in acct.pos and pol.blocked(i, t)]
        if skipped:
            notes.append("passed over by the buy guards: " + ", ".join(f"{t} (rank {rank[t]}, {_guard(pol, i, t)})" for t in skipped))

    after = _snapshot(acct, pol, px)
    if after["cash"] < -1e-6:
        raise AssertionError("the plan would leave negative cash")       # the account forbids it; this is a guard against a bug
    return Plan(asof, ran, orders, capture(pol, prices.index, i, ran, prev=state), before, after, notes)


def orders_frame(plan: Plan, to_game=lambda t: t.replace("-", ".")) -> pd.DataFrame:
    rows = [dict(side=o.side, symbol=o.symbol, game_symbol=to_game(o.symbol), shares=o.shares, est_price=round(o.est_price, 4),
                 est_value=round(o.est_value, 2), rank=o.rank, score=None if o.score is None else round(o.score, 4), reason=o.reason)
            for o in plan.orders]
    return pd.DataFrame(rows, columns=["side", "symbol", "game_symbol", "shares", "est_price", "est_value", "rank", "score", "reason"])


def plan_text(plan: Plan) -> str:
    b, a = plan.before, plan.after
    L = [f"Order plan for the next market open, based on prices and scores of {plan.asof.date()}",
         f"rebalance today: {plan.mode or 'none (only stops and the ETF floor were checked)'}",
         f"before: cash {b['cash']:,.2f}, value {b['equity']:,.2f}, {len(b['positions'])} positions, ETF core {b['core_share']:.1%}",
         f"after:  cash {a['cash']:,.2f}, value {a['equity']:,.2f}, {len(a['positions'])} positions, ETF core {a['core_share']:.1%}", ""]
    if not plan.orders:
        L.append("no orders today")
    for o in plan.orders:
        L.append(f"{o.side:<4} {o.shares:>6} {o.symbol:<7} ~{o.est_price:>10,.2f}  = {o.est_value:>11,.2f}   {o.reason}")
    if plan.notes:
        L += ["", "notes:"] + [f"  - {n}" for n in plan.notes]
    L += ["", "Market orders fill at the next open, not at these estimated prices. Nothing has been sent to the game."]
    return "\n".join(L)


def apply_fills(holdings: Holdings, orders) -> Holdings:
    """The holdings after these orders filled at their estimated prices (for tests and what-if runs)."""
    sh, avg, cash = dict(holdings.shares), dict(holdings.avg_cost), float(holdings.cash)
    for o in orders:
        if o.side == "BUY":
            old = sh.get(o.symbol, 0)
            avg[o.symbol] = (avg.get(o.symbol, 0.0) * old + o.est_value) / (old + o.shares)
            sh[o.symbol] = old + o.shares
            cash -= o.est_value
        else:
            sh[o.symbol] = sh.get(o.symbol, 0) - o.shares
            cash += o.est_value
            if sh[o.symbol] <= 0:
                sh.pop(o.symbol)
                avg.pop(o.symbol, None)
    if cash < -1e-6 or any(v < 0 for v in sh.values()) or not math.isfinite(cash):
        raise ValueError("these orders cannot all fill: cash or shares would go negative")
    return Holdings(cash, sh, avg, dict(holdings.prices))
