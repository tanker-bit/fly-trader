"""READ-ONLY: the game account as accounts.order_plan.Holdings, from the live game (through the browser bridge) or a snapshot file.

Snapshot file (JSON), handy for dry runs and tests; read_live() can also write one:
    {"cash": 100000.0, "positions": [{"symbol": "BRK.B", "quantity": 3, "purchase_price": 410.5, "current_price": 455.2}]}
Symbols are converted from the game's form (BRK.B) to ours (BRK-B). Short and option positions are not expected (the account is
long-only); if the game reports any, they are listed in the notes and left alone. No trade class is imported, nothing is ordered.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from accounts.order_plan import Holdings


def to_ours(game_symbol: str) -> str:
    return str(game_symbol).strip().upper().replace(".", "-")


def to_game(symbol: str) -> str:
    return str(symbol).strip().upper().replace("-", ".")


@dataclass
class GameAccount:
    holdings: Holdings
    source: str
    notes: list = field(default_factory=list)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def from_dict(d: dict, source="snapshot") -> GameAccount:
    cash = _num(d.get("cash"))
    if cash is None or cash < 0:
        raise ValueError(f"snapshot has no valid cash: {d.get('cash')!r}")
    sh, avg, px, notes = {}, {}, {}, []
    for p in d.get("positions", []):
        t, q = to_ours(p["symbol"]), _num(p.get("quantity"))
        if q is None or q <= 0 or q != int(q):
            notes.append(f"{t}: quantity {p.get('quantity')!r} is not a positive whole number; ignored")
            continue
        sh[t] = sh.get(t, 0) + int(q)
        if _num(p.get("purchase_price")):
            avg[t] = _num(p["purchase_price"])
        if _num(p.get("current_price")):
            px[t] = _num(p["current_price"])
        if t in avg and t in px and not 0.2 <= avg[t] / px[t] <= 5.0:
            notes.append(f"{t}: purchase price {avg[t]:,.2f} vs current {px[t]:,.2f} looks odd; check it is a per-share price "
                         f"(the stop-loss uses it)")
    for k in ("short_positions", "option_positions"):
        if d.get(k):
            notes.append(f"the game reports {len(d[k])} {k.replace('_', ' ')}; this long-only runner leaves them alone")
    return GameAccount(Holdings(cash, sh, avg, px), source, notes)


def read_snapshot(path) -> GameAccount:
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")), source=f"snapshot {path}")


def portfolio_to_dict(p) -> dict:
    """A library Portfolio object -> the snapshot format (no ids: they identify the user's game)."""
    pos = [dict(symbol=x.symbol, quantity=int(x.quantity), purchase_price=_num(x.purchase_price),
                current_price=_num(getattr(x, "current_price", None))) for x in p.stock_portfolio]
    return dict(cash=_num(p.cash), account_value=_num(p.account_value), positions=pos,
                short_positions=[x.symbol for x in p.short_portfolio], option_positions=[x.symbol for x in p.option_portfolio],
                open_orders=len(p.open_orders))


def read_live(lib_dir, game_name=None) -> GameAccount:
    """Needs node token_keeper.mjs running and logged in (read-only mode is enough). Restores the working folder afterwards."""
    here = os.getcwd()
    try:
        from broker.session_guard import DUMMY_CREDENTIALS, prepare_session
        prepare_session(lib_dir)
        from investopedia_api import InvestopediaApi        # noqa: E402  (the library's api class only; no trade classes)
        api = InvestopediaApi(DUMMY_CREDENTIALS, game_name=game_name) if game_name else InvestopediaApi(DUMMY_CREDENTIALS)
        d = portfolio_to_dict(api.portfolio)
    finally:
        os.chdir(here)
    acc = from_dict(d, source=f"live game{f' [{game_name}]' if game_name else ''}")
    if d.get("open_orders"):
        acc.notes.append(f"the game has {d['open_orders']} open order(s): they are not in this plan; check them before trading")
    return acc


def fresh(cash: float = 100_000.0) -> GameAccount:
    return GameAccount(Holdings(float(cash)), source=f"fresh account with {cash:,.0f} cash (no game read)")
