from accounts.account import Fill
from experiments.audit_trades import audit_fills, run
from tests.test_dashboard_trades import make


def F(side, t, sh, px=10.0, fee=0.0, d="2021-01-04"):
    return Fill(date=d, ticker=t, side=side, shares=sh, price=px, fee=fee, status="FILLED")


def test_clean_sequence_passes():
    assert audit_fills([F("BUY", "A", 100), F("SELL", "A", 40), F("SELL", "A", 60)], {"A"}) == []


def test_selling_unowned_or_oversold_is_flagged():
    assert any("short" in p for p in audit_fills([F("SELL", "A", 5)], {"A"}))
    assert any("short" in p for p in audit_fills([F("BUY", "A", 10), F("SELL", "A", 11)], {"A"}))


def test_overspending_and_foreign_ticker_flagged():
    assert any("cash" in p for p in audit_fills([F("BUY", "A", 10_000_000)], {"A"}))
    assert any("universe" in p for p in audit_fills([F("BUY", "ZZZ", 1)], {"A"}))


def test_real_engine_fills_are_clean():
    index, fwd1, scores = make()
    acct, eq, uni = run(scores, index, fwd1, k=5, every=21, cost_bps=10.0)
    assert acct.filled and audit_fills(acct.filled, uni) == []
