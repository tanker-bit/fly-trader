"""Tests for the long-only $1M account: no shorting, no leverage, no options, stocks + approved ETFs only."""
import random

import pytest

from accounts import Account, RuleViolation, Universe, START_CASH

PX = {"AAPL": 100.0, "MSFT": 50.0, "SPY": 400.0}


def mk(**kw):
    return Account(Universe(stocks=["AAPL", "MSFT"]), **kw)


def test_starts_with_one_million():
    a = mk()
    assert a.cash == START_CASH == 1_000_000.0 and a.pos == {} and a.equity() == START_CASH


def test_buy_stock_and_etf():
    a = mk(cost_bps=0)
    assert a.buy_shares("d1", "AAPL", 10, 100).status == "FILLED"
    assert a.buy_shares("d1", "SPY", 5, 400).status == "FILLED"
    assert a.pos == {"AAPL": 10, "SPY": 5} and a.cash == START_CASH - 1000 - 2000


def test_cannot_short_or_oversell():
    a = mk()
    f = a.sell_shares("d1", "AAPL", 1, 100)
    assert f.status == "REJECTED" and "shorting" in f.reason and a.pos == {} and a.cash == START_CASH
    a.buy_shares("d1", "AAPL", 10, 100)
    f = a.sell_shares("d2", "AAPL", 11, 100)
    assert f.status == "REJECTED" and a.pos["AAPL"] == 10


def test_no_leverage():
    a = mk(cost_bps=0)
    assert a.buy_shares("d1", "AAPL", 10_001, 100).status == "REJECTED"   # 1,000,100 > cash
    assert a.buy_shares("d1", "AAPL", 10_000, 100).status == "REJECTED" or a.cash == 0
    assert a.cash >= 0


def test_fee_counts_toward_cash_check():
    a = mk(cost_bps=10)
    assert a.buy_shares("d1", "AAPL", 10_000, 100).status == "REJECTED"   # 1,000,000 + 1,000 fee > cash
    assert a.buy_shares("d1", "AAPL", 9_990, 100).status == "FILLED"
    assert a.cash >= 0


def test_buy_dollars_whole_shares():
    a = mk(cost_bps=10)
    f = a.buy_dollars("d1", "AAPL", 1_000, 100)
    assert f.status == "FILLED" and f.shares == 9            # 10 shares would cost 1,001
    assert a.buy_dollars("d1", "AAPL", 50, 100).status == "REJECTED"


def test_forbidden_and_unknown_instruments():
    a = mk()
    for t in ["SQQQ", "TQQQ", "UVXY", "SH", "AAPL240621C00190000", "ZZZZ"]:
        f = a.buy_shares("d1", t, 1, 10)
        assert f.status == "REJECTED", t
    assert a.pos == {} and a.cash == START_CASH
    assert "SQQQ" not in Universe(stocks=["SQQQ"])              # cannot be sneaked in via the stock list


def test_whole_shares_and_bad_input():
    a = mk()
    assert a.buy_shares("d1", "AAPL", 1.5, 100).status == "REJECTED"
    for sh, px in [(0, 100), (-1, 100), (1, 0), (1, -5), (1, float("nan")), (float("nan"), 100)]:
        assert a.buy_shares("d1", "AAPL", sh, px).status == "REJECTED"
    assert Account(Universe(["AAPL"]), fractional=True).buy_shares("d1", "AAPL", 1.5, 100).status == "FILLED"


def test_strict_mode_raises():
    a = mk(strict=True)
    with pytest.raises(RuleViolation):
        a.sell_shares("d1", "AAPL", 1, 100)


def test_pnl_and_avg_cost():
    a = mk(cost_bps=0)
    a.buy_shares("d1", "AAPL", 10, 100)
    a.buy_shares("d2", "AAPL", 10, 120)
    assert a.avg_cost["AAPL"] == 110
    f = a.sell_shares("d3", "AAPL", 5, 130)
    assert f.realized == pytest.approx(100) and a.realized == pytest.approx(100)
    assert a.unrealized({"AAPL": 130}) == pytest.approx(15 * 20)
    assert a.equity() == pytest.approx(START_CASH + 100 + 300)


def test_round_trip_loses_only_fees():
    a = mk(cost_bps=10)
    a.buy_shares("d1", "AAPL", 100, 100)
    a.sell_all("d2", "AAPL", 100)
    assert a.pos == {} and a.cash == pytest.approx(START_CASH - a.fees) and a.fees == pytest.approx(20)


def test_missing_price_uses_last_known():
    a = mk(cost_bps=0)
    a.buy_shares("d1", "AAPL", 10, 100)
    a.mark({"AAPL": 110})
    assert a.equity({"AAPL": float("nan"), "MSFT": 50}) == pytest.approx(START_CASH + 100)
    assert a.stale({"MSFT": 50}) == ["AAPL"]


def test_delist_bankruptcy():
    a = mk(cost_bps=0)
    a.buy_shares("d1", "AAPL", 10, 100)
    f = a.delist("d2", "AAPL", 0.0)
    assert f.status == "FILLED" and a.pos == {} and a.cash == START_CASH - 1000 and a.delist("d3", "AAPL") is None


def test_weights_sum_to_one():
    a = mk(cost_bps=0)
    a.buy_shares("d1", "AAPL", 100, 100)
    w = a.weights(PX)
    assert sum(w.values()) == pytest.approx(1.0) and w["AAPL"] == pytest.approx(0.01)


def test_save_and_load_round_trip():
    a = mk()
    a.buy_shares("d1", "AAPL", 10, 100)
    a.sell_shares("d2", "AAPL", 99, 100)       # rejected, still logged
    b = Account.from_dict(a.to_dict(), a.universe)
    assert b.to_dict() == a.to_dict() and len(b.rejected) == 1


def test_fuzz_invariants():
    rnd = random.Random(0)
    names = ["AAPL", "MSFT", "SPY", "SQQQ", "ZZZZ"]
    px = {"AAPL": 100.0, "MSFT": 50.0, "SPY": 400.0}
    a = mk(cost_bps=10)
    for i in range(3000):
        t = rnd.choice(names)
        p = px.get(t, 10.0) * rnd.uniform(0.9, 1.1)
        side = rnd.random()
        if side < 0.45:
            a.buy_shares(i, t, rnd.randint(-2, 3000), p)
        elif side < 0.8:
            a.sell_shares(i, t, rnd.randint(-2, 3000), p)
        elif side < 0.9:
            a.buy_dollars(i, t, rnd.uniform(-10, 300_000), p)
        else:
            a.sell_all(i, t, p)
        assert a.cash >= -1e-6
        assert all(s > 0 for s in a.pos.values())
        assert set(a.pos) <= {"AAPL", "MSFT", "SPY"}
        assert a.equity() == pytest.approx(a.start_cash + a.realized + a.unrealized() - a.fees, rel=1e-9, abs=1e-4)
