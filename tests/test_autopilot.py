import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from accounts.order_plan import Order
from accounts.policy_state import PolicyState, load_state, save_state
from broker import autopilot as ap
from broker import site_account as sa

HEAD = ["Symbol", "Description", "Current Price", "Today's Change", "Purchase Price", "QTY", "Total Value", "Total Gain/Loss",
        "Trade Actions"]                                             # the real holdings table headers, 2026-11-27


# ---------------------------------------------------------------- reading the website's tables
def test_holdings_rows_become_positions():
    rows = [["BRK.B", "Berkshire Hathaway Inc", "$455.20", "+$1.10 (0.24%)", "$410.50", "3", "$1,365.60", "+$134.10", "BUY SELL"],
            ["GOOGL\nAlphabet", "Alphabet Inc", "$273.19", "-$0.50", "$270.00", "10", "$2,731.90", "+$31.90", ""]]
    pos = sa.parse_holdings(HEAD, rows)
    assert pos[0] == dict(symbol="BRK.B", quantity=3, purchase_price=410.5, current_price=455.2, total_value=1365.6)
    assert pos[1]["symbol"] == "GOOGL" and pos[1]["quantity"] == 10
    assert sa.parse_holdings(HEAD, [["you have no stock holdings yet"]]) == []
    with pytest.raises(sa.AccountReadError):
        sa.parse_holdings(["Ticker", "Shares"], rows)              # the page changed: never guess


def test_totals_must_add_up():
    ok = sa.SiteAccount(100_000.0, 95_000.0, 95_000.0, [dict(symbol="A", quantity=1, total_value=5_000.0)])
    sa.check_totals(ok)
    with pytest.raises(sa.AccountReadError):
        sa.check_totals(sa.SiteAccount(100_000.0, 95_000.0, 95_000.0, []))                  # a holding was not read
    acc = ok.as_game_account()
    assert acc.holdings.cash == 95_000.0 and acc.holdings.shares == {"A": 1}


def test_open_orders_are_parsed():
    head = ["Order Date & Time", "Symbol", "Status", "Transaction", "Quantity", "Order Price", "Processed at", "Expire On", "Order #",
            "Action"]
    o = sa.parse_orders(head, [["11/27/2026 4:30 PM", "JBL", "Open", "Buy", "13", "Market", "", "11/28/2026", "123", "CANCEL"]])
    assert o[0]["symbol"] == "JBL" and o[0]["transaction"] == "Buy" and o[0]["quantity"] == "13"


# ---------------------------------------------------------------- read_account on stand-in pages (no network)
def _edge_ok():
    return importlib.util.find_spec("playwright") is not None and any(
        Path(p).exists() for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                                   r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"))


PORTFOLIO = """<html><body><div data-cy="account-value-text">$101,000.00</div><div data-cy="buying-power-text">$96,000.00</div>
<div data-cy="cash-text">$96,000.00</div><div data-cy="holdings-table"><table><thead><tr>{h}</tr></thead><tbody>
<tr><td>BRK.B</td><td>Berkshire</td><td>$500.00</td><td>+$1</td><td>$480.00</td><td>10</td><td>$5,000.00</td><td>+$200</td><td></td></tr>
</tbody></table></div></body></html>""".format(h="".join(f"<th>{x}</th>" for x in HEAD))
STATUS = """<html><body><div data-cy="open-trade-table"><table><thead><tr><th>Order Date &amp; Time</th><th>Symbol</th><th>Status</th>
<th>Transaction</th><th>Quantity</th></tr></thead><tbody></tbody></table></div><div data-cy="failed-trade-table"><table><thead>
<tr><th>Failed On</th><th>Symbol</th></tr></thead><tbody></tbody></table></div></body></html>"""


@pytest.mark.skipif(not _edge_ok(), reason="playwright or Microsoft Edge not installed")
def test_read_account_from_stand_in_pages():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        page = b.new_page()
        page.route(sa.PORTFOLIO_URL, lambda r: r.fulfill(status=200, content_type="text/html", body=PORTFOLIO))
        page.route(sa.ORDER_STATUS_URL, lambda r: r.fulfill(status=200, content_type="text/html", body=STATUS))
        acct = sa.read_account(page, wait=0.1, log=lambda m: None)
        b.close()
    assert (acct.account_value, acct.cash) == (101_000.0, 96_000.0) and acct.open_orders == [] and acct.failed == []
    assert acct.positions == [dict(symbol="BRK.B", quantity=10, purchase_price=480.0, current_price=500.0, total_value=5000.0)]
    assert acct.as_game_account().holdings.shares == {"BRK-B": 10}


# ---------------------------------------------------------------- the autopilot's decisions (website and planner replaced by fakes)
class FakeBrowser:
    def __init__(self):
        self.opened = 0

    def __call__(self, profile, headless=False):
        self.opened += 1
        close = SimpleNamespace(close=lambda: None, stop=lambda: None)
        return close, close, SimpleNamespace()


def setup(monkeypatch, tmp_path, *, open_orders=(), notes=(), orders=1, results=("submitted",)):
    out = tmp_path / "orders"
    day = out / "2026-11-25"
    day.mkdir(parents=True)
    save_state(day / "state_after.json", PolicyState(asof="2026-11-25", started=True))
    plan_orders = [Order("BUY", "JBL", 13, 228.0, 2964.0, "new holding: rank 3", 3, 1.2)][:orders]
    (day / "orders.csv").write_text("side,symbol,game_symbol,shares,est_price,est_value,rank,score,reason\n" +
                                    "".join("BUY,JBL,JBL,13,228.0,2964.0,3,1.2,new holding\n" for _ in plan_orders))
    acct = sa.SiteAccount(100_000.0, 100_000.0, 100_000.0, [], list(open_orders), [])
    monkeypatch.setattr(sa, "read_account", lambda page, log=print: acct)
    calls = dict(run_day=0, run_orders=0)

    def fake_run_day(a, today=None, log=print):
        calls["run_day"] += 1
        calls["portfolio"] = a.portfolio
        return dict(out=str(day), asof="2026-11-25", notes=list(notes), plan=SimpleNamespace(orders=plan_orders))

    def fake_run_orders(site, orders, preview_only=False, max_drift=0.05, stop_file=None, log=print):
        calls["run_orders"] += 1
        calls["preview_only"] = preview_only
        return [dict(status=s, side="BUY", symbol="JBL", game_symbol="JBL", shares=13, plan_price=228.0, detail="", reason="")
                for s in results]

    monkeypatch.setattr(ap.dr, "run_day", fake_run_day)
    monkeypatch.setattr(ap.tu, "run_orders", fake_run_orders)
    a = ap.parser().parse_args(["--out", str(out), "--state", str(out / "policy_state.json"), "--profile", str(tmp_path / "prof")])
    return a, out, calls


def test_a_normal_day_trades_and_accepts(monkeypatch, tmp_path):
    a, out, calls = setup(monkeypatch, tmp_path)
    fb = FakeBrowser()
    assert ap.run(a, lambda m: None, browser=fb) == "traded"
    assert calls["run_orders"] == 1 and not calls["preview_only"] and fb.opened == 2
    assert load_state(out / "policy_state.json").asof == "2026-11-25"
    assert Path(calls["portfolio"]).name == "web_account.json"                   # the plan used the account read from the website


def test_open_orders_on_the_website_mean_no_trading(monkeypatch, tmp_path):
    a, out, calls = setup(monkeypatch, tmp_path, open_orders=[dict(symbol="JBL", transaction="Buy", quantity="13")])
    r = ap.run(a, lambda m: None, browser=FakeBrowser())
    assert r.startswith("skipped: open orders") and calls["run_day"] == 0 and calls["run_orders"] == 0


def test_stale_prices_mean_no_trading(monkeypatch, tmp_path):
    a, out, calls = setup(monkeypatch, tmp_path, notes=["WARNING: prices are from 2026-11-06, 21 days before today"])
    assert ap.run(a, lambda m: None, browser=FakeBrowser()).startswith("skipped: prices are stale") and calls["run_orders"] == 0


def test_the_same_price_date_is_never_traded_twice(monkeypatch, tmp_path):
    a, out, calls = setup(monkeypatch, tmp_path)
    (out / "2026-11-25" / "trade_log.csv").write_text("time,side,symbol,status\nx,BUY,JBL,submitted\n")
    assert ap.run(a, lambda m: None, browser=FakeBrowser()).startswith("skipped: orders for 2026-11-25 were already submitted")
    assert calls["run_orders"] == 0


def test_preview_only_never_accepts(monkeypatch, tmp_path):
    a, out, calls = setup(monkeypatch, tmp_path, results=("previewed",))
    a.preview_only = True
    assert ap.run(a, lambda m: None, browser=FakeBrowser()) == "previewed" and calls["preview_only"]
    assert not (out / "policy_state.json").exists()


def test_errors_leave_the_day_unaccepted(monkeypatch, tmp_path):
    a, out, calls = setup(monkeypatch, tmp_path, results=("submitted", "error"))
    assert ap.run(a, lambda m: None, browser=FakeBrowser()) == "traded" and not (out / "policy_state.json").exists()


def test_a_stop_file_stops_everything(monkeypatch, tmp_path):
    a, out, calls = setup(monkeypatch, tmp_path)
    (out / "STOP").write_text("")
    assert ap.run(a, lambda m: None, browser=FakeBrowser()).startswith("skipped:") and calls["run_day"] == 0
