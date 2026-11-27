import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from accounts.order_plan import apply_fills
from accounts.policy_state import PolicyState, load_state, save_state
from broker import game_portfolio as gp
from broker.daily_run import accept, buyable_set, parser, planner_prices, run_day
from broker.live_prices import overlay, refresh_one
from tournament.live_features import Inputs


# ---------------------------------------------------------------- availability
def test_buyable_set_excludes_no_and_optionally_unknown(tmp_path):
    f = tmp_path / "avail.csv"
    pd.DataFrame(dict(symbol=["AAA", "BBB", "NA"], game_symbol=["AAA", "BBB", "NA"], status=["yes", "no", "yes"])).to_csv(f, index=False)
    allowed, n = buyable_set(f, ["AAA", "BBB", "CCC", "NA"])
    assert allowed == {"AAA", "CCC", "NA"} and n == dict(yes=2, no=1, unknown=1, offered_only=False)   # NA is a ticker, not a gap
    allowed, _ = buyable_set(f, ["AAA", "BBB", "CCC", "NA"], offered_only=True)
    assert allowed == {"AAA", "NA"}
    allowed, n = buyable_set(tmp_path / "missing.csv", ["AAA"])
    assert allowed == {"AAA"} and n["unknown"] == 1


# ---------------------------------------------------------------- the game account
def test_snapshot_converts_symbols_and_flags_oddities():
    acc = gp.from_dict(dict(cash=1234.5, positions=[
        dict(symbol="BRK.B", quantity=3, purchase_price=400.0, current_price=450.0),
        dict(symbol="brk.b", quantity=2, purchase_price=410.0, current_price=450.0),
        dict(symbol="XYZ", quantity=1.5),
        dict(symbol="ODD", quantity=10, purchase_price=5000.0, current_price=50.0)], short_positions=["TSLA"]))
    h = acc.holdings
    assert h.cash == 1234.5 and h.shares == {"BRK-B": 5, "ODD": 10} and h.prices["BRK-B"] == 450.0
    text = " ".join(acc.notes)
    assert "XYZ" in text and "ODD" in text and "short positions" in text
    with pytest.raises(ValueError):
        gp.from_dict(dict(cash=-1, positions=[]))


def test_library_portfolio_object_becomes_a_snapshot_without_ids():
    pos = SimpleNamespace(symbol="AAPL", quantity=4, purchase_price="150.25", current_price=170.0)
    p = SimpleNamespace(cash="9000.5", account_value="9700", stock_portfolio=[pos], short_portfolio=[], option_portfolio=[],
                        open_orders=[1], portfolio_id=123, game_id=456)
    d = gp.portfolio_to_dict(p)
    assert d["cash"] == 9000.5 and d["positions"][0] == dict(symbol="AAPL", quantity=4, purchase_price=150.25, current_price=170.0)
    assert "portfolio_id" not in d and "game_id" not in d and d["open_orders"] == 1


# ---------------------------------------------------------------- live prices
def hist(n=5, start="2026-11-02", close=10.0):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Close": close, "Adj Close": close, "Volume": 1000.0, "Stock Splits": 0.0}, index=idx)


def test_refresh_keeps_the_old_file_when_the_download_fails(tmp_path):
    assert refresh_one("AAA", tmp_path, fetch=lambda s: hist(close=11.0)) == "ok"
    assert pd.read_pickle(tmp_path / "AAA.pkl")["Close"].iloc[-1] == 11.0

    def boom(s):
        raise ConnectionError("rate limited")

    assert refresh_one("AAA", tmp_path, fetch=boom, sleep=lambda s: None) == "error"
    assert pd.read_pickle(tmp_path / "AAA.pkl")["Close"].iloc[-1] == 11.0
    assert refresh_one("ZZZ", tmp_path, fetch=lambda s: None) == "empty" and not (tmp_path / "ZZZ.pkl").exists()
    bar = hist(close=12.0)
    bar.iloc[-1, 0] = np.nan                                       # today's half-formed bar
    refresh_one("AAA", tmp_path, fetch=lambda s: bar)
    assert pd.read_pickle(tmp_path / "AAA.pkl").index[-1] == bar.index[-2]


def test_todays_bar_is_dropped_until_after_the_close(tmp_path):
    """Seen on 2026-11-27 at 12:47 New York time: Yahoo already returned a partial bar for that day."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    ny = ZoneInfo("America/New_York")
    h = hist(4, start="2026-11-24")                                # 24, 25, 26, 27 November
    refresh_one("AAA", tmp_path, fetch=lambda s: h, now=datetime(2026, 11, 27, 12, 47, tzinfo=ny))
    assert pd.read_pickle(tmp_path / "AAA.pkl").index[-1] == pd.Timestamp("2026-11-26")
    refresh_one("AAA", tmp_path, fetch=lambda s: h, now=datetime(2026, 11, 27, 16, 30, tzinfo=ny))
    assert pd.read_pickle(tmp_path / "AAA.pkl").index[-1] == pd.Timestamp("2026-11-27")
    refresh_one("AAA", tmp_path, fetch=lambda s: h, now=datetime(2026, 11, 28, 9, 0, tzinfo=ny))     # next morning: yesterday is final
    assert pd.read_pickle(tmp_path / "AAA.pkl").index[-1] == pd.Timestamp("2026-11-27")


def test_overlay_replaces_and_extends_only_the_refreshed_columns(tmp_path):
    base = hist(5)
    old = pd.DataFrame({"AAA": base["Close"], "BBB": base["Close"] * 2})
    inp = Inputs(tmp_path, old, old, old * 100, {}, pd.DataFrame(columns=["symbol", "start", "end"]), {})
    live = tmp_path / "live"
    live.mkdir()
    hist(8, close=20.0).to_pickle(live / "AAA.pkl")
    got = overlay(inp, live)
    assert len(got.adj) == 8 and (got.adj["AAA"] == 20.0).all()
    assert got.adj["BBB"].iloc[:5].tolist() == [20.0] * 5 and got.adj["BBB"].iloc[5:].isna().all()
    assert overlay(inp, tmp_path / "nothing") is inp


def test_planner_prices_end_on_the_real_close(tmp_path):
    idx = pd.bdate_range("2026-01-01", periods=4)
    adj = pd.DataFrame({"AAA": [9.0, 9.5, 9.8, 9.9]}, index=idx)
    cs = pd.DataFrame({"AAA": [10.0, 10.5, 10.8, 11.0]}, index=idx)
    inp = Inputs(tmp_path, adj, cs, adj, {}, None, {})
    px = planner_prices(inp, ["AAA", "MISSING"], idx[-1])
    assert px["AAA"].tolist() == [9.0, 9.5, 9.8, 11.0] and list(px.columns) == ["AAA"]


# ---------------------------------------------------------------- accept
def test_accept_copies_the_days_state_and_never_goes_back(tmp_path):
    save_state(tmp_path / "out" / "2026-11-06" / "state_after.json", PolicyState(asof="2026-11-06", started=True))
    save_state(tmp_path / "out" / "2026-11-05" / "state_after.json", PolicyState(asof="2026-11-05", started=True))
    real = tmp_path / "out" / "policy_state.json"
    accept(tmp_path / "out", "2026-11-06", real, log=lambda s: None)
    assert load_state(real).asof == "2026-11-06"
    with pytest.raises(SystemExit):
        accept(tmp_path / "out", "2026-11-05", real, log=lambda s: None)
    with pytest.raises(SystemExit):
        accept(tmp_path / "out", "2026-11-07", real, log=lambda s: None)


# ---------------------------------------------------------------- end to end on the real caches (skipped where absent)
NEEDED = [Path("data_fundamentals/panel_hybrid.parquet"), Path("data_fundamentals/prices_etf"), Path("league_v8/config.json"),
          Path("data/mb_neurons.csv")]
real = pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="research caches or league_v8 not on this PC")


@real
def test_dry_run_two_days_with_accept(tmp_path):
    out, state = tmp_path / "orders", tmp_path / "orders" / "policy_state.json"
    base = ["--flies", "a0192", "--out", str(out), "--state", str(state), "--live-prices", str(tmp_path / "nolive"),
            "--availability", str(tmp_path / "none.csv")]
    a = parser().parse_args(base + ["--asof", "2026-11-05"])
    r1 = run_day(a, today=pd.Timestamp("2026-11-27").date(), log=lambda s: None)
    day1 = out / "2026-11-05"
    for f in ("orders.csv", "plan.txt", "scores.csv", "run.json", "state_after.json"):
        assert (day1 / f).exists()
    assert not state.exists()                                      # a dry run never touches the real memory
    meta = json.loads((day1 / "run.json").read_text())
    assert meta["mode"] == "full" and meta["orders"] > 5 and meta["after"]["cash"] >= 0
    assert meta["after"]["core_share"] >= 0.30 and meta["notes"][0].startswith("WARNING: prices are from 2026-11-05")
    orders = pd.read_csv(day1 / "orders.csv", keep_default_na=False)
    assert (orders.side == "BUY").all() and {"VTI", "RSP", "QQQ"} <= set(orders.symbol)

    # the orders "filled": the account and the accepted memory go into the next day
    h = apply_fills(gp.fresh(100_000.0).holdings, r1["plan"].orders)
    snap = tmp_path / "game_snapshot.json"
    snap.write_text(json.dumps(dict(cash=h.cash, positions=[dict(symbol=gp.to_game(t), quantity=q, purchase_price=h.avg_cost[t])
                                                            for t, q in h.shares.items()])))
    accept(out, "2026-11-05", state, log=lambda s: None)
    a2 = parser().parse_args(base + ["--asof", "2026-11-06", "--portfolio", str(snap)])
    r2 = run_day(a2, today=pd.Timestamp("2026-11-07").date(), log=lambda s: None)
    assert r2["mode"] is None and r2["before"]["positions"] == {t: q for t, q in sorted(h.shares.items())}
    assert all(o.reason.split(":")[0] in ("stop-loss", "trailing stop", "short-term sleeve stop", "trim", "ETF core")
               for o in r2["plan"].orders)                        # no rebalance due the next day: only stops / trims / floor
    assert not any(n.startswith("WARNING") for n in r2["notes"])
