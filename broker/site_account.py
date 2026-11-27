"""READ-ONLY: the fly reads its own game account from the simulator website (Portfolio and Order Status pages).

Layout seen on 2026-11-27 (data-cy names): Portfolio page /simulator/portfolio: account-value-text, buying-power-text, cash-text, and
the holdings table inside `holdings-table` (Symbol, Description, Current Price, Today's Change, Purchase Price, QTY, Total Value,
Total Gain/Loss, Trade Actions; "you have no stock holdings yet" when empty). Order Status page /simulator/trade/order-status:
`open-trade-table` (pending orders) and `failed-trade-table`.

The numbers are cross-checked: cash + the holdings' total value must equal the account value within 1% (or $50), otherwise the read is
refused (a changed page or a half-loaded table must never be traded on). Only the STOCKS & ETFS tab is read: the account is long-only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from broker import game_portfolio as gp
from broker.trade_ui import money

PORTFOLIO_URL = "https://www.investopedia.com/simulator/portfolio"
ORDER_STATUS_URL = "https://www.investopedia.com/simulator/trade/order-status"

TABLES_JS = r"""
(root) => {
  const box = document.querySelector(root);
  if (!box) return null;
  return [...box.querySelectorAll('table')].map(t => ({head: [...t.querySelectorAll('thead th')].map(h => h.innerText.trim()),
    rows: [...t.querySelectorAll('tbody tr')].map(r => [...r.querySelectorAll('td')].map(c => c.innerText.trim()))}));
}
"""


class AccountReadError(RuntimeError):
    pass


@dataclass
class SiteAccount:
    account_value: float
    buying_power: float
    cash: float
    positions: list = field(default_factory=list)       # [{symbol, quantity, purchase_price, current_price, total_value}]
    open_orders: list = field(default_factory=list)     # [{date, symbol, status, transaction, quantity, ...}]
    failed: list = field(default_factory=list)

    def as_game_account(self) -> gp.GameAccount:
        acc = gp.from_dict(dict(cash=self.cash, positions=self.positions), source="website (Portfolio page)")
        if self.open_orders:
            acc.notes.append(f"{len(self.open_orders)} open order(s) on the website: "
                             + ", ".join(f"{o.get('transaction', '?')} {o.get('quantity', '?')} {o.get('symbol', '?')}" for o in self.open_orders))
        return acc


def _key(h: str) -> str:
    return re.sub(r"[^a-z]+", "_", h.strip().lower()).strip("_")


def rows_as_dicts(head: list, rows: list) -> list:
    """Table rows -> dicts keyed by the header (lower_snake). Rows that are a single 'no ... yet' message are dropped."""
    keys = [_key(h) for h in head]
    out = []
    for r in rows:
        if len(r) < len(keys) or (len(r) == 1 and re.search(r"\bno\b|yet|empty", r[0], re.I)):
            continue
        out.append(dict(zip(keys, r)))
    return out


def first_word(cell: str) -> str:
    return re.split(r"\s+", (cell or "").strip(), maxsplit=1)[0].upper() if (cell or "").strip() else ""


def parse_holdings(head: list, rows: list) -> list:
    need = {"symbol", "qty", "purchase_price", "current_price", "total_value"}
    keys = {_key(h) for h in head}
    if not need <= keys:
        raise AccountReadError(f"holdings table headers changed: {head}")
    out = []
    for d in rows_as_dicts(head, rows):
        q = money(d["qty"])
        if q is None:
            raise AccountReadError(f"cannot read the quantity in {d}")
        out.append(dict(symbol=first_word(d["symbol"]), quantity=int(q), purchase_price=money(d["purchase_price"]),
                        current_price=money(d["current_price"]), total_value=money(d["total_value"])))
    return out


def parse_orders(head: list, rows: list) -> list:
    out = []
    for d in rows_as_dicts(head, rows):
        d = dict(d)
        d["symbol"] = first_word(d.get("symbol", ""))
        out.append(d)
    return out


def check_totals(acct: SiteAccount, tol=0.01, floor=50.0):
    held = sum(p.get("total_value") or 0.0 for p in acct.positions)
    gap = abs(acct.cash + held - acct.account_value)
    if gap > max(tol * acct.account_value, floor):
        raise AccountReadError(f"cash {acct.cash:,.2f} + holdings {held:,.2f} does not match the account value "
                               f"{acct.account_value:,.2f} (gap {gap:,.2f}): refusing to trade on this read")


def read_account(page, wait=8.0, log=print, mirror=None) -> SiteAccount:
    import time
    page.goto(PORTFOLIO_URL)
    try:
        page.wait_for_selector('[data-cy="holdings-table"]', timeout=45000)
    except Exception as e:
        if "auth." in page.url:
            from broker.trade_ui import NotSignedIn
            raise NotSignedIn("not signed in: run  python -m broker.trade_ui --login") from e
        raise AccountReadError("the Portfolio page has no holdings table") from e
    time.sleep(wait)
    if mirror:
        mirror(page, force=True)

    def num(cy):
        loc = page.locator(f'[data-cy="{cy}"]')
        v = money(loc.first.inner_text()) if loc.count() else None
        if v is None:
            raise AccountReadError(f"cannot read {cy} on the Portfolio page")
        return v

    acct = SiteAccount(num("account-value-text"), num("buying-power-text"), num("cash-text"))
    tabs = page.evaluate(TABLES_JS, '[data-cy="holdings-table"]') or []
    if not tabs:
        raise AccountReadError("no table inside holdings-table")
    acct.positions = parse_holdings(tabs[0]["head"], tabs[0]["rows"])

    page.goto(ORDER_STATUS_URL)
    try:
        page.wait_for_selector('[data-cy="open-trade-table"]', timeout=45000)
    except Exception as e:
        raise AccountReadError("the Order Status page has no open-trade table") from e
    time.sleep(wait / 2)
    if mirror:
        mirror(page, force=True)
    ot = page.evaluate(TABLES_JS, '[data-cy="open-trade-table"]') or []
    ft = page.evaluate(TABLES_JS, '[data-cy="failed-trade-table"]') or []
    acct.open_orders = parse_orders(ot[0]["head"], ot[0]["rows"]) if ot else []
    acct.failed = parse_orders(ft[0]["head"], ft[0]["rows"]) if ft else []
    check_totals(acct)
    log(f"website account: value {acct.account_value:,.2f}, cash {acct.cash:,.2f}, buying power {acct.buying_power:,.2f}, "
        f"{len(acct.positions)} holdings, {len(acct.open_orders)} open orders, {len(acct.failed)} failed trades listed")
    return acct
