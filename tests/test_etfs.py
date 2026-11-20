import numpy as np
import pandas as pd

from accounts.etfs import etf_fwd1, with_etfs
from accounts.instruments import FORBIDDEN_ETFS
from experiments.fetch_etfs import fetch_all, summarize


def write_cache(root, ticker, dates, prices):
    d = root / "prices_etf"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Close": prices, "Adj Close": prices, "Volume": 1.0, "Stock Splits": 0.0}, index=dates).to_pickle(d / f"{ticker}.pkl")


def test_etf_fwd1_is_next_day_return_on_the_given_dates(tmp_path):
    dates = pd.bdate_range("2021-01-04", periods=6)
    write_cache(tmp_path, "SPY", dates, [100.0, 110.0, 99.0, 99.0, 108.9, 108.9])
    got = etf_fwd1(tmp_path, dates)
    assert list(got.columns) == ["SPY"] and np.allclose(got["SPY"].iloc[:4], [0.10, -0.10, 0.0, 0.10])
    assert np.isnan(got["SPY"].iloc[-1])                                     # no next day for the last date
    sub = etf_fwd1(tmp_path, dates[1:4])
    assert len(sub) == 3 and np.isclose(sub["SPY"].iloc[0], -0.10)


def test_missing_cache_and_dates_the_etf_did_not_trade(tmp_path):
    dates = pd.bdate_range("2021-01-04", periods=5)
    assert etf_fwd1(tmp_path, dates).shape == (5, 0)                         # no cache folder: nothing, no error
    write_cache(tmp_path, "GLD", dates[2:], [50.0, 51.0, 52.0])
    got = etf_fwd1(tmp_path, dates)
    assert got["GLD"].iloc[:2].isna().all() and got["GLD"].iloc[2:4].notna().all()


def test_with_etfs_adds_columns_and_keeps_stocks(tmp_path):
    dates = pd.bdate_range("2021-01-04", periods=5)
    write_cache(tmp_path, "SPY", dates, [100.0] * 5)
    stocks = pd.DataFrame(0.01, index=dates, columns=["AAA", "BBB"])
    out = with_etfs(stocks, tmp_path)
    assert list(out.columns) == ["AAA", "BBB", "SPY"] and (out[["AAA", "BBB"]] == 0.01).all().all()
    assert with_etfs(out, tmp_path).shape == out.shape                        # a second call adds nothing


def test_fetch_all_skips_forbidden_and_reports_status(tmp_path):
    calls = []

    def fake(t, cache):
        calls.append(t)
        return "empty" if t == "GLD" else "ok"

    st = fetch_all(tmp_path, tickers=["SPY", "GLD", "SQQQ", "UVXY"], downloader=fake)
    assert set(st) == {"SPY", "GLD"} and st["GLD"] == "empty" and not (set(calls) & FORBIDDEN_ETFS)


def test_summarize_reads_the_cache(tmp_path):
    dates = pd.bdate_range("2021-01-04", periods=4)
    write_cache(tmp_path, "SPY", dates, [1.0, 2.0, 3.0, 4.0])
    s = summarize(tmp_path, {"SPY": "ok", "GLD": "empty"})
    assert s.loc["SPY", "days"] == 4 and str(s.loc["SPY", "first"]) == "2021-01-04" and s.loc["GLD", "days"] == 0


def test_mask_before_listing_blanks_days_before_the_etf_existed():
    from accounts.etfs import mask_before_listing
    dates = pd.bdate_range("2021-01-04", periods=6)
    f = pd.DataFrame({"AAA": 0.01, "QQQM": [np.nan, np.nan, 0.01, 0.01, 0.01, 0.01], "SPY": 0.0}, index=dates)
    p = pd.DataFrame(100.0, index=dates, columns=["AAA", "QQQM", "SPY"])
    out = mask_before_listing(p, f)
    assert out["QQQM"].iloc[:2].isna().all() and out["QQQM"].iloc[2:].notna().all()
    assert out["AAA"].notna().all() and out["SPY"].notna().all()
    assert p["QQQM"].notna().all()                                   # the input is not modified
