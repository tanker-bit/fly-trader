"""Run: python -m pytest tests/test_fetch_us_prices.py -q"""
import numpy as np
import pandas as pd

from experiments.fetch_us_prices import fetch_all, liquidity, real_price_and_dollar_volume


def hist(n=200, price=10.0, vol=200_000, splits=None, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=n)
    h = pd.DataFrame(dict(Close=price, **{"Adj Close": price}, Volume=float(vol), **{"Stock Splits": 0.0}), index=idx)
    for d, r in (splits or {}).items():
        h.loc[pd.Timestamp(d), "Stock Splits"] = r
    return h


def test_fetch_all_runs_every_symbol_and_reports_statuses():
    seen = []
    def fake(s, d):
        seen.append(s)
        return {"A": "ok", "B": "cached", "C": "empty"}.get(s, "error")
    out = fetch_all(["A", "B", "C", "D"], "x", workers=3, downloader=fake, every=2)
    assert out == {"A": "ok", "B": "cached", "C": "empty", "D": "error"} and sorted(seen) == ["A", "B", "C", "D"]


def test_real_price_undoes_later_splits_and_dollar_volume_is_unchanged():
    # a 4:1 split on day 100: Yahoo shows earlier closes divided by 4 and earlier volume times 4
    h = hist(price=10.0, vol=200_000, splits={"2020-05-20": 4.0})
    split_day = pd.Timestamp("2020-05-20")
    h.loc[h.index < split_day, "Close"] = 10.0 / 4
    h.loc[h.index < split_day, "Volume"] = 200_000 * 4.0
    px, dv = real_price_and_dollar_volume(h)
    assert np.isclose(px[h.index < split_day], 10.0).all() and np.isclose(px[h.index >= split_day], 10.0).all()
    assert np.isclose(dv.iloc[0], 2.5 * 800_000) and np.isclose(dv.iloc[-1], 10.0 * 200_000)


def test_liquidity_rule(tmp_path):
    hist(price=10, vol=200_000).to_pickle(tmp_path / "GOOD.pkl")           # $2M a day, $10
    hist(price=1.5, vol=2_000_000).to_pickle(tmp_path / "PENNY.pkl")       # liquid but under $2
    hist(price=10, vol=20_000).to_pickle(tmp_path / "THIN.pkl")            # $200k a day
    tr, summ = liquidity(tmp_path, ["GOOD", "PENNY", "THIN", "MISSING"], window=60)
    assert list(tr.columns) == ["GOOD", "PENNY", "THIN"]
    assert tr["GOOD"].iloc[-1] and not tr["PENNY"].iloc[-1] and not tr["THIN"].iloc[-1]
    assert not tr["GOOD"].iloc[:30].any(), "needs 40 of the last 60 days before it can pass"
    assert tr["GOOD"].iloc[45:].all()
    assert set(summ.symbol) == {"GOOD", "PENNY", "THIN"} and summ.set_index("symbol").loc["GOOD", "tradable_days"] > 100


def test_liquidity_has_no_lookahead(tmp_path):
    h = hist(n=250, price=10, vol=200_000)
    h.to_pickle(tmp_path / "X.pkl")
    a, _ = liquidity(tmp_path, ["X"])
    h2 = h.copy()
    h2.iloc[200:, h2.columns.get_loc("Volume")] = 1.0                      # the volume collapses late
    h2.to_pickle(tmp_path / "X.pkl")
    b, _ = liquidity(tmp_path, ["X"])
    assert (a["X"].iloc[:200] == b["X"].iloc[:200]).all()
    assert not b["X"].iloc[-1]


def test_a_trailing_bar_without_a_close_does_not_wipe_out_the_last_day(tmp_path):
    h = hist(n=120, price=10, vol=200_000)
    h.loc[h.index[-1] + pd.offsets.BDay(1)] = [np.nan, np.nan, np.nan, 0.0]          # today's half-formed row
    h.to_pickle(tmp_path / "X.pkl")
    tr, summ = liquidity(tmp_path, ["X"])
    assert tr["X"].iloc[-1] and summ.iloc[0]["last_price"] == 10.0
