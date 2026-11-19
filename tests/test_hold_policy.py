import numpy as np
import pandas as pd

from accounts.hold_policy import HoldParams
from accounts.instruments import Universe
from accounts.sim import run_account
from experiments.audit_trades import audit_fills

N_DAYS, N = 300, 30
TICK = [f"S{i:02d}" for i in range(N)]


def world(seed=0, drift=0.0003, vol=0.01):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=N_DAYS)
    r = rng.normal(drift, vol, (N_DAYS, N + 1))
    px = pd.DataFrame(100.0 * np.cumprod(1 + r, axis=0), index=dates, columns=TICK + ["SPY"])
    return px


def const_scores(px, order):
    """Same ranking every day: stocks listed in `order` first (best first)."""
    s = pd.Series({t: float(len(order) - n) for n, t in enumerate(order)})
    return pd.DataFrame([s] * len(px), index=px.index)


def run(px, sc, **kw):
    kw.setdefault("trail", None)                    # the older tests isolate one rule at a time
    kw.setdefault("max_weight", None)
    kw.setdefault("fill_every", kw.get("every", 63))
    return run_account(px, sc, Universe(stocks=TICK), hold=HoldParams(**kw))


def test_buys_k_names_then_holds_without_trimming_or_topping_up():
    px = world()
    sc = const_scores(px, TICK)
    acct, eq = run(px, sc, k=5, every=21, stop_loss=None, knife=None, jump=None)
    buys = [f for f in acct.filled if f.side == "BUY"]
    assert len(buys) == 5 and not [f for f in acct.filled if f.side == "SELL"]      # winners are left alone, nothing is trimmed
    assert audit_fills(acct.filled, set(px.columns)) == []


def test_min_hold_protects_a_fallen_rank_then_releases_it():
    px = world()
    order1, order2 = TICK, TICK[::-1]                                 # day 63: the ranking flips completely
    sc = pd.DataFrame([pd.Series({t: float(len(order1) - n) for n, t in enumerate(order1)})] * 40 +
                      [pd.Series({t: float(len(order2) - n) for n, t in enumerate(order2)})] * (N_DAYS - 40), index=px.index)
    early, _ = run(px, sc, k=5, every=21, min_hold=63, sell_rank=10, stop_loss=None, knife=None, jump=None)
    sold_days = sorted({pd.Timestamp(f.date) for f in early.filled if f.side == "SELL"})
    first_buy = min(pd.Timestamp(f.date) for f in early.filled if f.side == "BUY")
    assert sold_days and (px.index.get_loc(sold_days[0]) - px.index.get_loc(first_buy)) >= 63    # not sold before 63 trading days
    free, _ = run(px, sc, k=5, every=21, min_hold=0, sell_rank=10, stop_loss=None, knife=None, jump=None)
    assert min(pd.Timestamp(f.date) for f in free.filled if f.side == "SELL") < sold_days[0]


def test_stop_loss_sells_a_crash_and_the_cooldown_blocks_a_rebuy():
    px = world(drift=0.0, vol=0.002)
    crash_day = 100
    px.loc[px.index[crash_day:], "S00"] *= 0.6                       # S00 (top ranked) drops 40% and stays there
    sc = const_scores(px, TICK)
    acct, _ = run(px, sc, k=5, every=21, stop_loss=0.25, cooldown=63, knife=None, jump=None)
    sells = [f for f in acct.filled if f.side == "SELL" and f.ticker == "S00"]
    assert len(sells) == 1 and pd.Timestamp(sells[0].date) == px.index[crash_day]
    rebuys = [f for f in acct.filled if f.side == "BUY" and f.ticker == "S00" and pd.Timestamp(f.date) > px.index[crash_day]]
    assert all(px.index.get_loc(pd.Timestamp(f.date)) >= crash_day + 63 for f in rebuys)             # not before the cooldown ends


def test_falling_knife_and_jump_filters_block_buys_not_holdings():
    px = world(drift=0.0, vol=0.002)
    px["S00"] *= 0.985 ** np.arange(N_DAYS)                        # S00 falls steadily: already below -30% over 63 days at the first rebalance
    px.loc[px.index[100:], "S01"] *= 1.6                            # S01 gaps up 60% in one day: suspicious history
    sc = const_scores(px, TICK)
    acct, _ = run(px, sc, k=5, every=63, stop_loss=None, knife=0.30, jump=0.35, min_hold=0)
    first_day = {f.ticker for f in acct.filled if f.side == "BUY" and pd.Timestamp(f.date) == px.index[63]}
    assert "S00" not in first_day and "S01" in first_day            # S00 is a falling knife at day 63 (-30%+); S01's jump is still ahead
    acct2, _ = run(px, sc, k=5, every=63, stop_loss=None, knife=0.30, jump=0.35, min_hold=0, sell_rank=1)
    sold = [f for f in acct2.filled if f.side == "SELL" and f.ticker == "S01" and pd.Timestamp(f.date) == px.index[126]]
    rebought = [f for f in acct2.filled if f.side == "BUY" and f.ticker == "S01" and pd.Timestamp(f.date) == px.index[126]]
    assert sold and not rebought                                    # sold on rank, and the jump 26 days earlier blocks the same-day re-buy


def test_new_buys_skip_a_stock_that_already_crashed():
    px = world(drift=0.0, vol=0.002)
    px.loc[px.index[40:], "S00"] *= 0.55                            # -45% before the first rebalance
    sc = const_scores(px, TICK)
    acct, _ = run(px, sc, k=5, every=63, stop_loss=None, knife=0.30, jump=None)
    assert "S00" not in {f.ticker for f in acct.filled if f.side == "BUY"}
    acct_off, _ = run(px, sc, k=5, every=63, stop_loss=None, knife=None, jump=None)
    assert "S00" in {f.ticker for f in acct_off.filled if f.side == "BUY"}


def test_etf_sleeve_is_held_near_its_target_and_never_shorted():
    px = world()
    sc = const_scores(px, TICK)
    acct, eq = run(px, sc, k=5, every=63, etf="SPY", etf_weight=0.30, stop_loss=None, knife=None, jump=None)
    assert "SPY" in acct.pos
    first = pd.Timestamp([f for f in acct.filled if f.ticker == "SPY"][0].date)
    row = px.loc[first]
    assert abs(acct.log[0].shares * 0) == 0                       # (log is not empty)
    spy_val = sum(f.shares * f.price for f in acct.filled if f.ticker == "SPY" and f.side == "BUY" and pd.Timestamp(f.date) == first)
    eq_first = eq.loc[first]
    assert 0.25 < spy_val / eq_first < 0.32
    assert audit_fills(acct.filled, set(px.columns)) == []


def test_far_fewer_fills_than_the_old_policy_on_noisy_scores():
    px = world(seed=3)
    rng = np.random.default_rng(1)
    sc = pd.DataFrame(rng.normal(size=(N_DAYS, N)), index=px.index, columns=TICK)       # a restless signal: new ranking every day
    old, _ = run_account(px, sc, Universe(stocks=TICK), k=10, every=21)
    new, _ = run(px, sc, k=10, every=63)
    assert len(new.filled) < 0.5 * len(old.filled)
    assert audit_fills(new.filled, set(px.columns)) == []


def test_trailing_stop_sells_a_big_winner_that_gives_the_gains_back():
    px = world(drift=0.0, vol=0.002)
    up = np.concatenate([np.linspace(1, 3.0, 120), np.full(N_DAYS - 120, 3.0)])
    px["S00"] *= up                                                    # tripled by day 120 ...
    px.loc[px.index[150:], "S00"] *= 0.65                              # ... then -35% from the peak (still far above the purchase price)
    sc = const_scores(px, TICK)
    old, _ = run(px, sc, k=5, every=21, stop_loss=0.25, trail=None, knife=None, jump=None)
    assert not [f for f in old.filled if f.side == "SELL" and f.ticker == "S00"]       # the cost-based stop never fires
    acct, _ = run(px, sc, k=5, every=21, stop_loss=0.25, trail=0.30, knife=None, jump=None)
    sells = [f for f in acct.filled if f.side == "SELL" and f.ticker == "S00"]
    assert len(sells) == 1 and pd.Timestamp(sells[0].date) == px.index[150]


def test_concentration_cap_trims_a_runaway_winner():
    px = world(drift=0.0, vol=0.002)
    px["S00"] *= np.concatenate([np.linspace(1, 4.0, 150), np.full(N_DAYS - 150, 4.0)])
    sc = const_scores(px, TICK)
    acct, eq = run(px, sc, k=10, every=21, max_weight=0.12, trim_to=0.08, stop_loss=None, knife=None, jump=None)
    trims = [f for f in acct.filled if f.side == "SELL" and f.ticker == "S00"]
    assert trims and "S00" in acct.pos                                  # trimmed, not dumped
    last = px.iloc[-1]
    assert acct.pos["S00"] * last["S00"] / eq.iloc[-1] < 0.14
    capless, eq2 = run(px, sc, k=10, every=21, max_weight=None, stop_loss=None, knife=None, jump=None)
    assert capless.pos["S00"] * last["S00"] / eq2.iloc[-1] > 0.20


def test_stopped_out_cash_is_refilled_at_the_next_fill_day_not_the_next_rebalance():
    px = world(drift=0.0, vol=0.002)
    px.loc[px.index[130:], "S00"] *= 0.6                               # crash on day 130, between fill days 126 and 147
    sc = const_scores(px, TICK)
    acct, _ = run(px, sc, k=5, every=63, fill_every=21, stop_loss=0.25, knife=None, jump=None)
    stop_day = px.index[130]
    after = [pd.Timestamp(f.date) for f in acct.filled if f.side == "BUY" and pd.Timestamp(f.date) > stop_day]
    assert after and px.index.get_loc(min(after)) == 147                # refilled on the next fill day (147), not at 189
    slow, _ = run(px, sc, k=5, every=63, fill_every=63, stop_loss=0.25, knife=None, jump=None)
    after2 = [pd.Timestamp(f.date) for f in slow.filled if f.side == "BUY" and pd.Timestamp(f.date) > stop_day]
    assert px.index.get_loc(min(after2)) == 189


def test_etf_with_no_cash_places_no_order_and_sells_no_stock():
    px = world()
    sc = const_scores(px, TICK)
    acct, _ = run(px, sc, k=5, every=21, etf="SPY", etf_weight=0.30, stop_loss=None, knife=None, jump=None)
    assert not [f for f in acct.log if f.status == "REJECTED"]
    assert not [f for f in acct.filled if f.side == "SELL" and f.ticker != "SPY"]
