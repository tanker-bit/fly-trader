import numpy as np
import pandas as pd
import pytest

from accounts.hold_policy import CORES, MIX, HoldParams
from accounts.order_plan import Holdings, apply_fills, orders_frame, plan_day, plan_text
from accounts.policy_state import PolicyState

N_DAYS, N = 200, 40
TICK = [f"S{i:02d}" for i in range(N)]
P = HoldParams(k=5, core=CORES["spy"], tactical_k=2, tactical_weight=0.10, knife=None, jump=None)


def world(seed=0, vol=0.004):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=N_DAYS)
    r = rng.normal(0.0002, vol, (N_DAYS, N + 1))
    return pd.DataFrame(100.0 * np.cumprod(1 + r, axis=0), index=dates, columns=TICK + ["SPY"])


def scores(order=TICK):
    return pd.Series({t: float(len(order) - n) for n, t in enumerate(order)})


def test_first_day_with_fresh_cash_buys_the_top_picks_the_sleeve_and_the_core():
    px = world()
    plan = plan_day(px, scores(), Holdings(cash=100_000.0), P)
    assert plan.mode == "full" and plan.before["equity"] == 100_000.0
    buys = {o.symbol: o for o in plan.orders}
    assert all(o.side == "BUY" and o.shares >= 1 and o.shares == int(o.shares) for o in plan.orders)
    assert {t for t in buys if buys[t].reason.startswith("new holding")} == set(TICK[:5])         # the 5 best
    assert {t for t in buys if buys[t].reason.startswith("short-term sleeve")} == set(TICK[5:7])  # the next 2
    assert buys["SPY"].reason.startswith("ETF core") and buys["S00"].rank == 1
    assert plan.after["cash"] >= 0 and plan.after["core_share"] >= P.core_floor
    assert abs(sum(o.est_value for o in plan.orders) + plan.after["cash"] - 100_000.0) < 1e-6        # no commission
    assert plan.state.started and plan.state.last_full == px.index[-1].strftime("%Y-%m-%d") and set(plan.state.tact) == set(TICK[5:7])


def test_the_next_day_after_the_fills_is_quiet():
    px = world()
    day1 = plan_day(px.iloc[:-1], scores(), Holdings(cash=100_000.0), P)
    h = apply_fills(Holdings(cash=100_000.0), day1.orders)
    day2 = plan_day(px, scores(), h, P, state=day1.state)
    assert day2.mode is None and day2.orders == []
    assert day2.after["positions"] == {t: q for t, q in sorted(h.shares.items())}


def test_a_stock_down_30_percent_is_sold_by_the_stop_loss():
    px = world()
    day1 = plan_day(px.iloc[:-5], scores(), Holdings(cash=100_000.0), P)
    h = apply_fills(Holdings(cash=100_000.0), day1.orders)
    px.loc[px.index[-3]:, "S02"] *= 0.70
    day2 = plan_day(px, scores(), h, P, state=day1.state)
    sells = [o for o in day2.orders if o.side == "SELL"]
    assert [o.symbol for o in sells] == ["S02"] and sells[0].reason.startswith("stop-loss") and sells[0].shares == h.shares["S02"]
    assert "S02" in day2.state.cooldown


def test_buys_only_what_the_game_offers():
    px = world()
    plan = plan_day(px, scores(), Holdings(cash=100_000.0), P, buyable=TICK[3:] + ["SPY"])
    bought = {o.symbol for o in plan.orders if o.reason.startswith("new holding")}
    assert bought == set(TICK[3:8]) and not bought & set(TICK[:3])
    assert plan.orders[0].rank is not None and min(o.rank for o in plan.orders if o.rank) == 1      # ranks among buyable names


def test_full_rebalance_sells_a_held_stock_that_fell_out_of_the_ranking():
    px = world()
    old = PolicyState(asof=px.index[-64].strftime("%Y-%m-%d"), started=True, entry={"S39": px.index[-130].strftime("%Y-%m-%d")},
                      peak={"S39": float(px["S39"].iloc[-64])}, last_full=px.index[-64].strftime("%Y-%m-%d"),
                      last_fill=px.index[-64].strftime("%Y-%m-%d"))
    h = Holdings(cash=50_000.0, shares={"S39": 100, "SPY": 300}, avg_cost={"S39": 90.0, "SPY": 95.0})
    p = HoldParams(k=5, sell_rank=10, core=CORES["spy"], stop_loss=None, trail=None, knife=None, jump=None)    # 40 stocks: limit 10
    plan = plan_day(px, scores(), h, p, state=old)
    assert plan.mode == "full"
    s39 = [o for o in plan.orders if o.symbol == "S39"]
    assert len(s39) == 1 and s39[0].side == "SELL" and s39[0].shares == 100
    assert s39[0].reason.startswith("rank fell: rank 40 (sell when worse than 10; held 129 trading days)")


def test_unknown_and_forbidden_holdings_are_reported_not_crashed_on():
    px = world()
    h = Holdings(cash=10_000.0, shares={"SQQQ": 10, "ZZZZ": 5}, avg_cost={"ZZZZ": 20.0}, prices={"ZZZZ": 21.5})
    plan = plan_day(px, scores(), h, P)
    text = " ".join(plan.notes)
    assert "SQQQ" in text and "cannot be traded" in text
    assert "ZZZZ: no price in our data" in text and "21.50" in text
    assert not any(o.symbol in ("SQQQ", "ZZZZ") for o in plan.orders)


def test_stocks_passed_over_by_the_guards_are_explained():
    px = world()
    px.loc[px.index[-40]:, "S01"] *= np.linspace(1.0, 0.5, 40)       # the #2 pick slid 50% over 40 days: a falling knife
    px.loc[px.index[-10]:, "S03"] *= 0.5                              # the #4 pick halved in one day: a suspicious jump
    plan = plan_day(px, scores(), Holdings(cash=100_000.0), HoldParams(k=5, core=CORES["spy"]))
    assert not {"S01", "S03"} & {o.symbol for o in plan.orders}
    note = [n for n in plan.notes if n.startswith("passed over")]
    assert note and "S01 (rank 2, down " in note[0] and "over 63 trading days)" in note[0]
    assert "S03 (rank 4, a one-day move of 50% in the last 60 days)" in note[0]


def test_no_state_with_existing_holdings_is_flagged():
    px = world()
    plan = plan_day(px, scores(), Holdings(cash=1_000.0, shares={"S00": 10}, avg_cost={"S00": 100.0}), P)
    assert any("no saved policy state" in n for n in plan.notes)


def test_report_and_csv_frame():
    px = world()
    plan = plan_day(px, scores(), Holdings(cash=100_000.0), P)
    df = orders_frame(plan)
    assert list(df.columns) == ["side", "symbol", "game_symbol", "shares", "est_price", "est_value", "rank", "score", "reason"]
    assert len(df) == len(plan.orders)
    assert orders_frame(plan_day(px, pd.Series({"BRK-B": 1.0}), Holdings(cash=0.0), P)).empty
    txt = plan_text(plan)
    assert "Nothing has been sent to the game" in txt and "BUY" in txt


def test_apply_fills_refuses_impossible_orders():
    from accounts.order_plan import Order
    with pytest.raises(ValueError):
        apply_fills(Holdings(cash=10.0), [Order("BUY", "S00", 1, 100.0, 100.0, "x")])


def test_mix_preset_runs_on_a_plain_world():
    """MIX wants VTI/RSP/QQQ; without their prices it must still plan (no core today) and say so."""
    px = world()
    plan = plan_day(px, scores(), Holdings(cash=100_000.0), MIX)
    assert any("core ETF" in n for n in plan.notes) and plan.after["cash"] >= 0
