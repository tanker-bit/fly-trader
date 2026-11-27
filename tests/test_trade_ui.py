import importlib.util
from pathlib import Path

import pytest

from broker.trade_ui import (TRADE_URL, check_before_submit, money, parse_panel, parse_preview, pick_suggestion, run_orders,
                             should_accept)

# the stock panel text exactly as the real site showed it on 2026-11-27 for GOOGL
REAL_PANEL = ("Symbol\nGOOGL\nVolume (current)\t17.490m\nDay's High ($)\t279.58\nDay's Low ($)\t270.03\n52 Week High ($)\t408.61\n"
              "Bid/Ask price ($)\t273.17/273.21\n52 Week Low ($)\t262.73\n")


def order(side="BUY", sym="GOOGL", shares=2, price=100.05):
    return dict(side=side, symbol=sym.replace(".", "-"), game_symbol=sym, shares=str(shares), est_price=str(price), reason="test")


# ---------------------------------------------------------------- parsing and guards
def test_money_and_panel_parsing():
    assert money("$1,234.50") == 1234.5 and money("Estimated Total $273.19") == 273.19 and money("n/a") is None
    p = parse_panel(REAL_PANEL)
    assert p["bid"] == 273.17 and p["ask"] == 273.21 and p["day_high"] == 279.58 and p["low_52w"] == 262.73 and p["volume"] == "17.490m"


def test_preview_parsing():
    pv = parse_preview("Stock: Buy at Market", "GOOGL", "1", ["Duration Day Only", "Estimated Price $273.19", "Quantity 1",
                                                              "Commission $0.00", "Estimated Total\n$1,273.19"])
    assert pv == dict(type="Stock: Buy at Market", symbol="GOOGL", qty=1, est_price=273.19, est_total=1273.19, commission=0.0)


def test_only_the_exact_ticker_is_picked():
    texts = ["GOOG | Alphabet Inc - Class C", "GOOGL | Alphabet Inc - Ordinary Shares - Class A"]
    assert pick_suggestion(texts, "GOOGL") == 1 and pick_suggestion(texts, "goog") == 0
    assert pick_suggestion(["BRK.A | Berkshire", "BRK.B | Berkshire B"], "BRK.B") == 1
    assert pick_suggestion(["GOOGL | Alphabet"], "GOO") is None


def test_guards_before_submit():
    o, panel = order(price=100.0), dict(bid=100.0, ask=100.1)
    good = dict(type="Stock: Buy at Market", symbol="GOOGL", qty=2, est_price=100.1, est_total=200.2)
    assert check_before_submit(o, panel, good, 100_000.0, 0.05) is None
    assert "expected 'Buy at Market'" in check_before_submit(o, panel, dict(good, type="Stock: Short at Market"), 1e5, 0.05)
    assert "symbol" in check_before_submit(o, panel, dict(good, symbol="GOOG"), 1e5, 0.05)
    assert "quantity" in check_before_submit(o, panel, dict(good, qty=20), 1e5, 0.05)
    assert "price moved +10.0%" in check_before_submit(o, dict(bid=110.0, ask=110.0), good, 1e5, 0.05)
    assert "buying power" in check_before_submit(o, panel, good, 150.0, 0.05)
    sell = order(side="SELL", price=100.0)
    assert check_before_submit(sell, dict(bid=97.0, ask=99.0), dict(good, type="Stock: Sell at Market"), 0.0, 0.05) is None  # bid used


def test_accept_rule():
    assert should_accept([dict(status="submitted"), dict(status="skipped")])
    assert not should_accept([dict(status="submitted"), dict(status="error")])
    assert not should_accept([dict(status="skipped")])                    # nothing went through: do not mark the rebalance as done
    assert not should_accept([dict(status="stopped")]) and should_accept([])


# ---------------------------------------------------------------- the whole flow on a local stand-in page (no network, no account)
def _edge_ok():
    if importlib.util.find_spec("playwright") is None:
        return False
    return any(Path(p).exists() for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                                          r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"))


browser = pytest.mark.skipif(not _edge_ok(), reason="playwright or Microsoft Edge not installed")
FAKE = (Path(__file__).parent / "fake_trade_page.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def edge():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        yield b
        b.close()


@pytest.fixture
def site(edge, tmp_path):
    from broker.trade_ui import TradeSite
    page = edge.new_page()
    page.route(TRADE_URL, lambda r: r.fulfill(status=200, content_type="text/html", body=FAKE))   # the real site is never contacted
    page.route("https://**", lambda r: r.abort() if r.request.url != TRADE_URL else r.fallback())
    s = TradeSite(page, shots=tmp_path / "screens", pause=0.05, log=lambda m: None)
    yield s
    page.close()


@browser
def test_buy_and_sell_are_submitted_exactly_as_planned(site, tmp_path):
    res = run_orders(site, [order("BUY", "GOOGL", 2, 100.10), order("SELL", "BRK.B", 3, 450.0)], log=lambda m: None)
    assert [r["status"] for r in res] == ["submitted", "submitted"]
    assert [r["game_symbol"] for r in res] == ["BRK.B", "GOOGL"]                # sells first
    assert res[1]["ask"] == 100.10 and res[0]["bid"] == 450.0 and "placed" in res[1]["detail"]
    assert len(list((tmp_path / "screens").glob("*.png"))) >= 4


@browser
def test_the_form_is_filled_with_buy_or_sell_market_day_only(site):
    from broker.trade_ui import TradeSite
    site.open_trade()
    site.enter(order("SELL", "AAPL", 5, 200.0))
    pv = site.preview()
    assert pv["type"] == "Stock: Sell at Market" and pv["symbol"] == "AAPL" and pv["qty"] == 5 and pv["est_price"] == 200.0
    site.submit()
    assert site.page.evaluate("window.submitted") == [dict(side="Sell", symbol="AAPL", qty=5, type="Market", dur="Day Only")]


@browser
def test_preview_only_never_submits(site):
    site.open_trade()
    res = run_orders(site, [order()], preview_only=True, log=lambda m: None)
    assert res[0]["status"] == "previewed" and site.page.evaluate("window.submitted") == []


@browser
def test_guards_skip_bad_orders_and_the_game_keeps_nothing(site):
    res = run_orders(site, [order(price=80.0), order(sym="ZZZZ"), order(shares=5000)], log=lambda m: None)
    assert [r["status"] for r in res] == ["skipped"] * 3
    assert "price moved" in res[0]["detail"] and "no exact match" in res[1]["detail"] and "buying power" in res[2]["detail"]
    assert site.page.evaluate("window.submitted") == []


@browser
def test_a_preview_that_does_not_match_is_not_submitted(site):
    site.open_trade()
    site.page.evaluate("window.QTY_BUG = 1")                                  # the page shows 3 shares when 2 were entered
    panel = site.enter(order())
    pv = site.preview()
    assert check_before_submit(order(), panel, pv, site.buying_power(), 0.05).startswith("preview quantity 3")
    assert site.page.evaluate("window.submitted") == []


@browser
def test_a_stop_file_halts_before_the_next_order(site, tmp_path):
    stop = tmp_path / "STOP"
    stop.write_text("stop")
    res = run_orders(site, [order(), order(sym="AAPL", price=200.2)], stop_file=stop, log=lambda m: None)
    assert [r["status"] for r in res] == ["stopped"]
