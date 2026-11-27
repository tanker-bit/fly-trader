import json

import numpy as np
import pandas as pd
import pytest

from accounts.account import Account
from accounts.hold_policy import CORES, HoldParams, HoldPolicy
from accounts.instruments import Universe
from accounts.policy_state import (PolicyState, VERSION, _date_at, _pos, capture, load_state, mode_today, restore,
                                   save_state)

N_DAYS, N = 260, 30
TICK = [f"S{i:02d}" for i in range(N)]

# Every rule that keeps memory is switched on and tight enough to fire: stops, trailing stops, cooldowns, trims, the sleeve, the core.
BUSY = HoldParams(k=5, every=42, fill_every=21, sell_rank=10, min_hold=21, stop_loss=0.05, trail=0.06, cooldown=10,
                  max_weight=0.30, trim_to=0.20, core=CORES["spy"], lt_days=60, lt_sell_rank=15,
                  tactical_k=2, tactical_weight=0.10, tactical_sell_rank=8, tactical_stop=0.04)


def world(seed=0, drift=0.0003, vol=0.02):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=N_DAYS)
    r = rng.normal(drift, vol, (N_DAYS, N + 1))
    return pd.DataFrame(100.0 * np.cumprod(1 + r, axis=0), index=dates, columns=TICK + ["SPY"])


def noisy_scores(px, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(len(px), N)), index=px.index, columns=TICK)


def todays_order(px, sc, hold, i):
    """The backtest's schedule (same as accounts.sim.run_account with exec_lag 1): scores from the day before, fixed grid."""
    if i >= 1 and (i % hold.every == 0 or i % hold.fill_every == 0):
        j = sc.index.searchsorted(px.index[i - 1], side="right") - 1
        if j >= 0:
            return sc.iloc[j], ("full" if i % hold.every == 0 else "fill")
    return None, None


def run_continuous(px, sc, hold):
    acct, pol = Account(Universe(stocks=TICK)), HoldPolicy(hold, px)
    seen = dict(cooldown=False, sleeve=False, sells=0)
    for i, d in enumerate(px.index):
        row, mode = todays_order(px, sc, hold, i)
        pol.day(acct, i, d, px.iloc[i].to_dict(), row, mode)
        seen["cooldown"] |= any(u > i for u in pol.until.values())
        seen["sleeve"] |= bool(pol.tact)
    seen["sells"] = sum(f.side == "SELL" for f in acct.filled)
    return acct, seen


def run_restarted(px, sc, hold, path, window=None):
    """Same days, but the policy is rebuilt from the saved file every morning, on a price frame that may start later."""
    acct = Account(Universe(stocks=TICK))
    for i, d in enumerate(px.index):
        lo = 0 if window is None else max(0, i - window + 1)
        frame = px.iloc[lo:i + 1]
        k = i - lo
        pol = HoldPolicy(hold, frame)
        state = load_state(path)
        restore(state, pol, frame.index, k)
        row, mode = todays_order(px, sc, hold, i)
        pol.day(acct, k, d, px.iloc[i].to_dict(), row, mode)
        save_state(path, capture(pol, frame.index, k, mode, prev=state))
    return acct


def test_restarting_every_day_from_the_file_makes_exactly_the_same_trades(tmp_path):
    px, sc = world(), noisy_scores(world())
    ref, seen = run_continuous(px, sc, BUSY)
    assert seen["cooldown"] and seen["sleeve"] and seen["sells"] > 0          # the memory really was used
    again = run_restarted(px, sc, BUSY, tmp_path / "state.json")
    assert again.log == ref.log


def test_a_shorter_price_window_each_day_still_makes_the_same_trades(tmp_path):
    """Entry dates older than the frame are counted back in weekdays; the window is long enough for the 60/63-day guards."""
    px, sc = world(seed=3), noisy_scores(world(seed=3), seed=4)
    ref, _ = run_continuous(px, sc, BUSY)
    again = run_restarted(px, sc, BUSY, tmp_path / "state.json", window=100)
    assert again.log == ref.log


def test_cooldowns_count_down_across_frames():
    px = world()
    calm = HoldParams(knife=None, jump=None)                  # so blocked() answers only about the cooldown
    pol = HoldPolicy(calm, px.iloc[:100])
    pol.until = {"S01": 99 + 10, "S02": 99}                    # S02's cooldown ends today: not saved
    pol.entry = {"S03": 5}
    pol.started = True
    s = capture(pol, px.index[:100], 99)
    assert s.cooldown == {"S01": 10} and s.entry == {"S03": px.index[5].strftime("%Y-%m-%d")}

    later = px.iloc[50:104]                                    # 4 trading days later, frame starts on day 50
    pol2 = HoldPolicy(calm, later)
    restore(s, pol2, later.index, 53)
    assert pol2.until["S01"] - 53 == 6 and pol2.blocked(53, "S01")
    assert pol2.entry["S03"] == 5 - 50                         # bought before this frame: negative index
    assert pol2.started

    much_later = px.iloc[60:110]                               # 10 trading days after the save: cooldown over
    pol3 = HoldPolicy(calm, much_later)
    restore(s, pol3, much_later.index, 49)
    assert "S01" not in pol3.until and not pol3.blocked(49, "S01")


def test_mode_today_follows_the_rebalance_rhythm_and_catches_up_after_missed_days():
    dates = pd.bdate_range("2021-01-04", periods=200)
    p = HoldParams(every=63, fill_every=21)
    day = lambda k: dates[k].strftime("%Y-%m-%d")
    assert mode_today(None, dates, 0, p) == "full"
    assert mode_today(PolicyState(asof=day(3)), dates, 5, p) == "full"                     # never started yet
    s = PolicyState(asof=day(10), started=True, last_full=day(10), last_fill=day(10))
    assert mode_today(s, dates, 10, p) is None                                             # same day again: nothing new
    assert mode_today(s, dates, 30, p) is None
    assert mode_today(s, dates, 31, p) == "fill"
    assert mode_today(s, dates, 73, p) == "full"
    assert mode_today(s, dates, 150, p) == "full"                                          # long gap: catch up
    s2 = PolicyState(asof=day(31), started=True, last_full=day(10), last_fill=day(31))
    assert mode_today(s2, dates, 51, p) is None and mode_today(s2, dates, 52, p) == "fill"


def test_capture_records_rebalance_dates():
    px = world()
    pol = HoldPolicy(HoldParams(), px)
    full = capture(pol, px.index, 40, "full")
    assert full.last_full == full.last_fill == full.asof == px.index[40].strftime("%Y-%m-%d")
    fill = capture(pol, px.index, 61, "fill", prev=full)
    assert fill.last_full == full.last_full and fill.last_fill == px.index[61].strftime("%Y-%m-%d")
    quiet = capture(pol, px.index, 62, None, prev=fill)
    assert (quiet.last_full, quiet.last_fill) == (fill.last_full, fill.last_fill)
    with pytest.raises(ValueError):
        capture(pol, px.index, 62, "sometimes")


def test_file_round_trip_and_safety(tmp_path):
    path = tmp_path / "sub" / "state.json"
    assert load_state(path) is None                                                        # first day: no file yet
    s = PolicyState(asof="2024-03-01", started=True, entry={"AAPL": "2024-01-02", "BRK.B": "2023-12-29"},
                    tact={"NA": "2024-02-27"}, peak={"AAPL": 191.25}, cooldown={"XYZ": 7},
                    last_full="2024-01-02", last_fill="2024-02-01")
    save_state(path, s)
    assert load_state(path) == s
    assert list(load_state(path).entry) == ["AAPL", "BRK.B"]                               # order kept
    assert "NA" in load_state(path).tact                                                   # the ticker NA is not read as missing
    assert not (tmp_path / "sub" / "state.json.tmp").exists()
    assert json.loads(path.read_text())["version"] == VERSION

    path.write_text("{ half a fi")
    with pytest.raises(ValueError):
        load_state(path)
    path.write_text(json.dumps(dict(s.to_dict(), version=VERSION + 1)))
    with pytest.raises(ValueError):
        load_state(path)


def test_restore_refuses_a_state_from_the_future():
    px = world()
    pol = HoldPolicy(HoldParams(), px.iloc[:50])
    future = PolicyState(asof=px.index[60].strftime("%Y-%m-%d"), started=True)
    with pytest.raises(ValueError):
        restore(future, pol, px.index[:50], 49)


def test_restore_of_nothing_leaves_a_fresh_policy():
    px = world()
    pol = HoldPolicy(HoldParams(), px)
    restore(None, pol, px.index, 10)
    assert pol.entry == {} and pol.until == {} and not pol.started


def test_dates_before_the_frame_round_trip():
    dates = pd.bdate_range("2022-06-01", periods=30)
    for k in (0, 7, 29, -1, -7, -300):
        assert _pos(dates, _date_at(dates, k)) == k
    assert _pos(dates, "2022-06-04") == _pos(dates, "2022-06-06")                          # a Saturday maps to the next trading day
