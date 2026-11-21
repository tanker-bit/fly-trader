import numpy as np
import pandas as pd

from accounts.account import Account
from accounts.hold_policy import CORES, HoldParams, HoldPolicy
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
    acct, eq = run(px, sc, k=5, every=63, core=CORES["spy"], core_weight=0.30, core_floor=0.0, stop_loss=None, knife=None, jump=None)
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
    acct, _ = run(px, sc, k=5, every=21, core=CORES["spy"], core_weight=0.30, core_floor=0.0, stop_loss=None, knife=None, jump=None)
    assert not [f for f in acct.log if f.status == "REJECTED"]
    assert not [f for f in acct.filled if f.side == "SELL" and f.ticker != "SPY"]


def daily_core_share(px, sc, hold, start_day=126):
    """Run the policy day by day (same loop as run_account) and record the core's share of equity after each day once trading started."""
    uni = Universe(stocks=TICK)
    acct, pol = Account(uni), HoldPolicy(hold, px)
    shares = []
    for i, d in enumerate(px.index):
        row, mode = None, None
        if i >= 1 and (i % hold.every == 0 or i % hold.fill_every == 0):
            j = sc.index.searchsorted(px.index[i - 1], side="right") - 1
            row = sc.iloc[j] if j >= 0 else None
            mode = "full" if i % hold.every == 0 else "fill"
        pxd = px.iloc[i].to_dict()
        pol.day(acct, i, d, pxd, row, mode if row is not None else None)
        if pol.started:
            shares.append(pol.core_value(acct, pxd) / acct.equity(pxd))
    return np.array(shares), acct


def core_hold(**kw):
    base = dict(k=5, every=63, fill_every=21, core=CORES["spy"], core_weight=0.35, core_floor=0.30, stop_loss=None, trail=None,
                knife=None, jump=None, max_weight=None)
    base.update(kw)
    return HoldParams(**base)


def test_core_is_at_least_the_floor_every_day_after_trading_starts():
    px = world(seed=5)
    px.loc[:, TICK[:5]] *= np.linspace(1.0, 4.0, N_DAYS)[:, None]           # the stocks rally hard: the core's share would drift far below 30%
    sc = const_scores(px, TICK)
    shares, acct = daily_core_share(px, sc.iloc[:], core_hold())
    assert len(shares) > 100 and shares.min() >= 0.30 - 1e-3 and shares[0] > 0.30
    assert [f for f in acct.filled if f.side == "SELL" and f.ticker in TICK], "the floor was kept by trimming stocks, not by magic"
    assert not [f for f in acct.log if f.status == "REJECTED"]


def test_without_the_floor_the_same_rally_breaks_30_percent():
    px = world(seed=5)
    px.loc[:, TICK[:5]] *= np.linspace(1.0, 4.0, N_DAYS)[:, None]
    shares, _ = daily_core_share(px, const_scores(px, TICK), core_hold(core_floor=0.0))
    assert shares.min() < 0.25


def test_a_basket_core_follows_its_shares_and_renormalises_when_a_member_is_missing():
    px = world()
    for t in ("VTI", "RSP", "QQQ"):
        px[t] = 100.0 * np.cumprod(1 + np.random.default_rng(len(t)).normal(0.0003, 0.01, N_DAYS))
    sc = const_scores(px, TICK)
    shares, acct = daily_core_share(px, sc, core_hold(core=CORES["broad"]))
    spot = {t: acct.pos[t] * px[t].iloc[-1] for t in ("VTI", "RSP", "QQQ")}
    tot = sum(spot.values())
    assert 0.35 < spot["VTI"] / tot < 0.65 and spot["RSP"] / tot > 0.12 and spot["QQQ"] / tot > 0.12
    px2 = px.copy()
    px2["QQQ"] = np.nan                                                    # one member has no price (not listed yet)
    _, acct2 = daily_core_share(px2, sc, core_hold(core=CORES["broad"]))
    assert "QQQ" not in acct2.pos and "VTI" in acct2.pos and "RSP" in acct2.pos


def test_only_broad_equity_etfs_count_toward_the_core():
    px = world()
    for t in ("TLT", "GLD"):
        px[t] = 100.0
    acct = Account(Universe(stocks=TICK))
    acct.buy_shares("d", "TLT", 1000, 100.0)
    acct.buy_shares("d", "GLD", 1000, 100.0)
    acct.buy_shares("d", "SPY", 10, 100.0)
    pol = HoldPolicy(core_hold(), px)
    assert abs(pol.core_value(acct, px.iloc[0].to_dict()) - 10 * px["SPY"].iloc[0]) < 1e-6           # only SPY counts


def test_a_leveraged_etf_can_never_be_the_core():
    px = world()
    px["TQQQ"] = 100.0
    sc = const_scores(px, TICK)
    shares, acct = daily_core_share(px, sc, core_hold(core=(("TQQQ", 1.0),)))
    assert "TQQQ" not in acct.pos and not [f for f in acct.log if f.status == "REJECTED"]
    assert len([f for f in acct.filled if f.side == "BUY"]) == 5                    # no tradable core -> the stocks use the whole account


def test_no_etf_prices_means_no_core_and_no_error():
    px = world().drop(columns=["SPY"])
    sc = const_scores(px, TICK)
    shares, acct = daily_core_share(px, sc, core_hold())
    assert len(acct.pos) == 5 and len(shares) and shares.max() == 0.0


# ---------------------------------------------------------------- the mix: long-term book + core + short-term sleeve
def mix_hold(**kw):
    base = dict(k=5, every=63, fill_every=21, stop_loss=None, trail=None, knife=None, jump=None, max_weight=None,
                tactical_k=2, tactical_weight=0.10, tactical_sell_rank=12, tactical_stop=None)
    base.update(kw)
    return HoldParams(**base)


def sleeve_names(acct):
    return {f.ticker for f in acct.filled if f.side == "BUY"} - set(TICK[:5])


def test_the_sleeve_has_no_clock_and_no_profit_cap_while_the_fly_still_likes_the_name():
    px = world(seed=3, drift=0.003)                   # strong uptrend: every name gains far more than 12%
    sc = const_scores(px, TICK)
    acct, _ = run_account(px, sc, Universe(stocks=TICK), hold=mix_hold())
    assert sleeve_names(acct), "the sleeve bought something"
    assert not [f for f in acct.filled if f.side == "SELL"]       # held for the whole run: no take-profit, no time exit


def test_the_sleeve_sells_when_the_fly_ranking_turns_against_the_name():
    px = world(seed=3)
    order = TICK[:]
    sc = const_scores(px, order)
    sleeve = order[5]                                   # a sleeve name (6th best)
    worse = [x for x in order if x != sleeve] + [sleeve]  # from day 130 the fly ranks it last
    sc.iloc[130:] = const_scores(px, worse).reindex(columns=sc.columns).iloc[130:].to_numpy()
    acct, _ = run_account(px, sc, Universe(stocks=TICK), hold=mix_hold())
    sells = [f for f in acct.filled if f.side == "SELL" and f.ticker == sleeve]
    assert sells and str(sells[0].date)[:10] >= str(px.index[130].date())     # sold only after the fly changed its mind


def test_sleeve_downside_stop_still_works_when_enabled():
    px = world(seed=6)
    sc = const_scores(px, TICK)
    px.loc[px.index[100]:, TICK[5]] *= 0.5
    acct, _ = run_account(px, sc, Universe(stocks=TICK), hold=mix_hold(tactical_stop=0.08))
    sells = [f for f in acct.filled if f.side == "SELL" and f.ticker == TICK[5]]
    assert sells and str(sells[0].date)[:10] <= str(px.index[102].date())


def test_long_term_holdings_survive_a_rank_drop_that_a_new_holding_would_not():
    px = world(seed=7)
    order = TICK[:]
    sc = const_scores(px, order)
    worse = order[1:] + order[:1]
    sc.iloc[130:] = const_scores(px, worse).reindex(columns=sc.columns).iloc[130:].to_numpy()
    kw = dict(tactical_k=0, sell_rank=10, min_hold=10, lt_days=100)
    a_short, _ = run_account(px, sc, Universe(stocks=TICK), hold=mix_hold(lt_sell_rank=None, **kw))
    a_long, _ = run_account(px, sc, Universe(stocks=TICK), hold=mix_hold(lt_sell_rank=999, **kw))
    sold = lambda a: [f for f in a.filled if f.side == "SELL" and f.ticker == TICK[0]]
    assert sold(a_short) and not sold(a_long)


def test_mix_preset_has_core_floor_and_sleeve_and_audit_passes():
    from accounts.hold_policy import MIX
    assert MIX.core and MIX.tactical_k > 0 and MIX.core_floor >= 0.30
    assert not hasattr(MIX, "tactical_take") and not hasattr(MIX, "tactical_days")      # no profit ceiling, no forced exit date
    px = world(seed=8)
    sc = const_scores(px, TICK)
    acct, _ = run_account(px, sc, Universe(stocks=TICK), hold=MIX)
    assert audit_fills(acct.filled, set(px.columns)) == []


def test_mix_never_sends_an_order_it_cannot_afford():
    from accounts.hold_policy import MIX
    from dataclasses import replace
    px = world(seed=9)
    px.loc[:, TICK[:10]] *= 400.0                    # pricey shares and a fully invested account: the old loop kept firing 0-share orders
    for e in ("VTI", "RSP", "QQQ"):
        px[e] = px["SPY"] * (1.0 + 0.0001 * np.arange(N_DAYS))
    sc = const_scores(px, TICK)
    acct, _ = run_account(px, sc, Universe(stocks=TICK), hold=replace(MIX, k=8))
    assert acct.filled and not acct.rejected
