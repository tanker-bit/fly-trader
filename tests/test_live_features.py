from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tournament.live_features import Inputs, feature_rows

FUND = ["earnings_yield", "book_to_price", "fcf_yield", "roe", "roa", "gross_margin", "rev_growth_yoy", "eps_growth_yoy",
        "debt_to_equity", "accruals", "days_since_filing"]


# ---------------------------------------------------------------- synthetic market (no files needed)
def synthetic(tmp_path, n_days=330, n=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    tick = [f"T{k:02d}" for k in range(n)]
    px = pd.DataFrame(50.0 * np.cumprod(1 + rng.normal(0.0003, 0.02, (n_days, n)), axis=0), index=dates, columns=tick)
    px.iloc[: n_days - 100, 0] = np.nan                       # T00: listed only 100 days ago -> too little history
    px["T01"] = px["T01"] / 200.0                             # T01: a penny stock -> no row
    vol = pd.DataFrame(rng.integers(1e5, 1e6, (n_days, n)).astype(float), index=dates, columns=tick)
    iv = pd.DataFrame({"symbol": tick, "start": pd.NaT, "end": pd.NaT})
    iv.loc[iv.symbol == "T29", "end"] = dates[100]            # T29 left the index long ago: not a member today
    return Inputs(Path(tmp_path), px, px, vol, {}, iv, {})


def test_synthetic_rows_are_ranks_among_todays_members(tmp_path):
    inp = synthetic(tmp_path)
    res = feature_rows(inp)
    assert res.asof == inp.adj.index[-1]
    assert "T29" not in res.members and set(res.dropped) == {"T00", "T01"}
    assert len(res.rows) == 27 and not res.rows.isna().any().any()
    assert res.rows.abs().max().max() <= 1.0
    assert (res.rows[FUND] == 0.0).all().all() and (res.rows["fund_known"] == -1.0).all()    # no SEC data: middle + flag
    assert list(res.rows.columns[:13]) == ["ret_1", "ret_5", "ret_20", "ret_60", "ret_120", "mom_12_1", "vol_20", "vol_ratio",
                                           "volume_ratio", "ma50_dist", "ma200_dist", "rsi_14", "dist_52w_high"]


def test_synthetic_result_does_not_depend_on_the_window(tmp_path):
    inp = synthetic(tmp_path)
    a, b = feature_rows(inp, window=300), feature_rows(inp, window=330)
    pd.testing.assert_frame_equal(a.rows, b.rows)


def test_a_partly_refreshed_cache_falls_back_to_the_last_complete_day(tmp_path):
    """Seen in the real cache: after 2026-11-06 only about 100 of 600 tickers had new prices."""
    inp = synthetic(tmp_path)
    inp.adj.iloc[-2:, 5:] = np.nan                            # the last 2 days: only 5 of 30 tickers refreshed
    res = feature_rows(inp)
    assert res.asof == inp.adj.index[-3] and res.skipped_dates == list(inp.adj.index[-2:][::-1])
    assert res.coverage == 1.0 and res.no_price_today == [] and len(res.rows) == 27
    with pytest.raises(ValueError, match="incomplete"):
        feature_rows(inp, asof=inp.adj.index[-1])             # asking for the thin day explicitly is refused


def test_one_missing_price_is_kept_like_the_panel_but_reported(tmp_path):
    inp = synthetic(tmp_path)
    inp.adj.iloc[-1, 10] = np.nan                             # T10 has no quote today: the panel's 3-day forward fill covers it
    res = feature_rows(inp)
    assert res.asof == inp.adj.index[-1] and res.no_price_today == ["T10"] and "T10" in res.rows.index


def test_synthetic_bad_requests_are_refused(tmp_path):
    inp = synthetic(tmp_path)
    with pytest.raises(ValueError, match="too short"):
        feature_rows(inp, window=100)
    with pytest.raises(ValueError, match="not a trading day"):
        feature_rows(inp, asof="2021-01-09")                  # a Saturday
    with pytest.raises(ValueError, match="features need"):
        feature_rows(inp, asof=inp.adj.index[50])


# ---------------------------------------------------------------- the real caches (skipped where absent)
ROOT = Path("data_fundamentals")
NEEDED = [ROOT / "panel_hybrid.parquet", ROOT / "universe" / "intervals.csv", ROOT / "prices", ROOT / "sec"]
real = pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="data_fundamentals caches not on this PC")


@pytest.fixture(scope="module")
def caches():
    from tournament.live_features import load_inputs
    return load_inputs(ROOT), pd.read_parquet(ROOT / "panel_hybrid.parquet")


def _check_against_panel(inp, panel, day):
    res = feature_rows(inp, asof=day)
    want = panel.xs(day, level="date")
    want = want[[c for c in want.columns if not c.startswith("fwd_")]].sort_index()
    assert list(res.rows.index) == list(want.index)
    assert list(res.rows.columns) == list(want.columns)
    assert np.array_equal(res.rows.to_numpy(), want.to_numpy())


@real
def test_rebuilds_the_last_panel_day_exactly(caches):
    inp, panel = caches
    _check_against_panel(inp, panel, panel.index.get_level_values("date").max())


@real
def test_rebuilds_an_old_panel_day_exactly_despite_later_splits(caches):
    inp, panel = caches
    _check_against_panel(inp, panel, pd.Timestamp("2023-04-14"))


@real
def test_end_to_end_the_flies_score_rebuilt_rows_as_in_the_league(caches):
    """Rebuilt feature rows -> tournament.live -> exactly the scores league_v8 saved for that day in round 18."""
    if not all(p.exists() for p in (Path("league_v8/config.json"), Path("data/mb_neurons.csv"), Path("data/mb_edges.csv"))):
        pytest.skip("league_v8 or the connectome csvs not on this PC")
    from tournament.live import load_flies, pick_flies, score_rows
    from tournament.store import Store
    inp, _ = caches
    store = Store("league_v8")
    cfg = store.load_config()
    res = feature_rows(inp, asof="2023-04-14")
    flies = load_flies(store, cfg, pick_flies(store, ["a0192"]))
    got = score_rows(flies, res.rows, cfg["feats"])
    z = np.load("league_v8/rounds/scores/r18_a0192.npz")
    key = z.files[0]                                          # d<i>: the first validation day of round 18 = 2023-04-14
    assert np.array_equal(got["a0192"].to_numpy().astype(np.float32), z[key])
