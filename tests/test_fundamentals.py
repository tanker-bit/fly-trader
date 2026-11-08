"""Run: python -m tests.test_fundamentals

Proves the point-in-time rules of the SEC fundamentals pipeline on synthetic filings (no network):
  1. no filing is visible before its filed date (and, with the 1-day lag, not on it either)
  2. a restatement filed later never leaks into earlier dates
  3. TTM / YoY arithmetic is right
  4. truncation invariance: deleting every filing after a cutoff changes nothing on or before the cutoff
  5. a stock split cannot create a jump in market-value based features
  6. the panel built with fundamentals has no look-ahead (same truncation test, end to end)
  7. SEC client: needs a User-Agent with an email, never exceeds 8 requests/second, caches, stops on 403
  8. Wikipedia parsing and point-in-time membership
"""
import io
import json
import tempfile
import urllib.error

import numpy as np
import pandas as pd

from env import fundamentals as fnd
from env import universe_us as uni
from env.features import FUND_FEATURES, build_panel, feature_columns


def fact(end, val, filed, form, start=None):
    d = {"end": end, "val": val, "filed": filed, "form": form, "accn": "x"}
    if start:
        d["start"] = start
    return d


def company_json(extra=None):
    """FY2019 10-K, Q1/Q2 2020 10-Qs, FY2020 10-K that RESTATES FY2019 net income from 100 to 90."""
    usd = lambda tag, rows: {tag: {"units": {"USD": rows}}}
    f = {
        **usd("NetIncomeLoss", [
            fact("2019-12-31", 100, "2020-02-20", "10-K", "2019-01-01"),
            fact("2020-03-31", 30, "2020-05-05", "10-Q", "2020-01-01"), fact("2019-03-31", 20, "2020-05-05", "10-Q", "2019-01-01"),
            fact("2020-06-30", 70, "2020-08-05", "10-Q", "2020-01-01"), fact("2019-06-30", 45, "2020-08-05", "10-Q", "2019-01-01"),
            fact("2019-12-31", 90, "2021-03-01", "10-K", "2019-01-01"), fact("2020-12-31", 150, "2021-03-01", "10-K", "2020-01-01")]),
        **usd("Revenues", [
            fact("2019-12-31", 1000, "2020-02-20", "10-K", "2019-01-01"),
            fact("2020-03-31", 300, "2020-05-05", "10-Q", "2020-01-01"), fact("2019-03-31", 200, "2020-05-05", "10-Q", "2019-01-01"),
            fact("2020-06-30", 700, "2020-08-05", "10-Q", "2020-01-01"), fact("2019-06-30", 450, "2020-08-05", "10-Q", "2019-01-01"),
            fact("2020-12-31", 1600, "2021-03-01", "10-K", "2020-01-01")]),
        **usd("StockholdersEquity", [fact("2019-12-31", 500, "2020-02-20", "10-K"), fact("2020-03-31", 520, "2020-05-05", "10-Q"),
                                      fact("2020-06-30", 560, "2020-08-05", "10-Q"), fact("2020-12-31", 700, "2021-03-01", "10-K")]),
        **usd("Assets", [fact("2019-12-31", 2000, "2020-02-20", "10-K"), fact("2020-03-31", 2100, "2020-05-05", "10-Q"),
                         fact("2020-06-30", 2200, "2020-08-05", "10-Q"), fact("2020-12-31", 2500, "2021-03-01", "10-K")]),
        "EarningsPerShareDiluted": {"units": {"USD/shares": [
            fact("2019-12-31", 1.0, "2020-02-20", "10-K", "2019-01-01"), fact("2020-12-31", 1.5, "2021-03-01", "10-K", "2020-01-01"),
            fact("2019-12-31", 0.9, "2021-03-01", "10-K", "2019-01-01")]}},
    }
    js = {"facts": {"us-gaap": f, "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        fact("2020-02-10", 100, "2020-02-20", "10-K"), fact("2020-05-01", 100, "2020-05-05", "10-Q"),
        fact("2020-08-01", 100, "2020-08-05", "10-Q"), fact("2021-02-20", 100, "2021-03-01", "10-K")]}}}}}
    return js


def events(js=None, lag=1):
    return fnd.build_events(fnd.compact_companyfacts(js or company_json()), lag)


def trading_dates():
    return pd.bdate_range("2020-01-01", "2021-06-30")


def metric_on(ev, dates, name, close=None, splits=None):
    close = close if close is not None else pd.Series(10.0, index=dates)
    return pd.Series(fnd.align_metrics(ev, dates, close, splits)[name], index=dates)


def test_nothing_visible_before_filed_date():
    ev, dates = events(), trading_dates()
    first = ev.iloc[0]
    assert first["filed"] == pd.Timestamp("2020-02-20") and first["avail"] == pd.Timestamp("2020-02-21")
    roa = metric_on(ev, dates, "roa")
    assert roa[:pd.Timestamp("2020-02-20")].isna().all(), "metric visible on or before the filing date"
    assert roa[pd.Timestamp("2020-02-21")] == 100 / 2000
    for _, e in ev.iterrows():                      # every date only ever sees an event filed strictly before it
        shown = metric_on(ev, dates, "days_since_filing")
        assert (shown.dropna() >= 1).all(), "a filing was used on its own filed date (1-day lag violated)"
    ev0 = events(lag=0)                             # with lag 0 the filing is usable ON its filed date but never before
    roa0 = metric_on(ev0, dates, "roa")
    assert roa0[:pd.Timestamp("2020-02-19")].isna().all() and not np.isnan(roa0[pd.Timestamp("2020-02-20")])


def test_restatement_is_not_leaked_before_it_is_filed():
    ev, dates = events(), trading_dates()
    # FY2019 net income was 100 when first reported, 90 in the restatement filed 2021-03-01
    first_report = ev[ev["filed"] == pd.Timestamp("2020-02-20")].iloc[0]
    assert first_report["ni_ttm"] == 100
    roe = metric_on(ev, dates, "roe")
    assert np.isclose(roe[pd.Timestamp("2020-03-02")], 100 / 500)                    # original, pre-restatement
    assert np.isclose(roe[pd.Timestamp("2020-06-01")], (100 + 30 - 20) / 520)        # Q1 filing; still original FY2019
    assert np.isclose(roe[pd.Timestamp("2021-03-02")], 150 / 700)                    # only now is the restated year known
    assert all(ev[ev["filed"] < pd.Timestamp("2021-03-01")]["ni_ttm"] != 90)


def test_ttm_and_yoy_arithmetic():
    ev = events()
    q2 = ev[ev["filed"] == pd.Timestamp("2020-08-05")].iloc[0]
    assert q2["ni_ttm"] == 100 + 70 - 45                                              # FY + YTD - prior YTD
    assert q2["rev_ttm"] == 1000 + 700 - 450
    assert (q2["rev_cur"], q2["rev_prev"]) == (700 - 0, 450) or (q2["rev_cur"], q2["rev_prev"]) == (700, 450)
    fy = ev[ev["filed"] == pd.Timestamp("2021-03-01")].iloc[0]
    assert (fy["eps_cur"], fy["eps_prev"]) == (1.5, 0.9), "EPS growth must compare like with like (same filing, restated prior year)"
    dates = trading_dates()
    g = metric_on(ev, dates, "eps_growth_yoy")
    assert np.isclose(g[pd.Timestamp("2021-03-02")], (1.5 - 0.9) / 0.9)


def test_truncation_invariance():
    """Deleting every filing after a cutoff must not change any metric on or before the cutoff."""
    js, dates = company_json(), trading_dates()
    cutoff = pd.Timestamp("2020-09-01")
    full = events(js)
    cut = fnd.compact_companyfacts(js)
    cut = cut[cut["filed"] <= cutoff]
    part = fnd.build_events(cut)
    px = pd.Series(np.linspace(8, 14, len(dates)), index=dates)
    a, b = fnd.align_metrics(full, dates, px), fnd.align_metrics(part, dates, px)
    m = dates <= cutoff + pd.Timedelta(days=0)
    for k in fnd.METRICS:
        assert np.allclose(a[k][m], b[k][m], equal_nan=True), f"{k} changed when future filings were removed"
    assert not np.allclose(a["roe"][~m], b["roe"][~m], equal_nan=True), "test is vacuous: the future should differ"


def test_split_does_not_create_a_jump():
    ev, dates = events(), trading_dates()
    V = pd.Series(np.linspace(10, 12, len(dates)), index=dates)                      # value per ORIGINAL share
    s = pd.Timestamp("2020-10-01")
    split = pd.Series({s: 2.0})                                                      # 2-for-1 on 2020-10-01
    close_split = V / np.where(dates < s, 2.0, 2.0)                                  # Yahoo 'Close': split-adjusted (continuous)
    before = metric_on(ev, dates, "earnings_yield", close_split, split)
    day_before, day_after = before[pd.Timestamp("2020-09-30")], before[pd.Timestamp("2020-10-01")]
    assert abs(day_after / day_before - 1) < 0.01, f"split created a jump in earnings yield ({day_before} -> {day_after})"
    mcap_event = 100 * (V[pd.Timestamp("2020-08-06")] / 2.0) * 2.0                   # shares x actual price at the 2020-08 filing
    assert np.isclose(before[pd.Timestamp("2020-08-06")], (100 + 70 - 45) / mcap_event)


def synthetic_prices(n_days=600, n_stocks=14, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    tick = [f"S{i}" for i in range(n_stocks)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (n_days, n_stocks)), axis=0)), index=dates, columns=tick)
    vol = pd.DataFrame(rng.integers(1e5, 1e6, (n_days, n_stocks)).astype(float), index=dates, columns=tick)
    return close, vol


def raw_metrics(close, js_by_ticker, cutoff=None):
    raw = {m: pd.DataFrame(np.nan, index=close.index, columns=close.columns) for m in fnd.METRICS}
    for t, js in js_by_ticker.items():
        facts = fnd.compact_companyfacts(js)
        if cutoff is not None:
            facts = facts[facts["filed"] <= cutoff]
        for m, v in fnd.align_metrics(fnd.build_events(facts), close.index, close[t]).items():
            raw[m][t] = v
    return raw


def test_panel_with_fundamentals_has_no_lookahead():
    close, vol = synthetic_prices()
    rng = np.random.default_rng(1)
    jss = {}
    for t in close.columns:                          # each stock: the same story scaled randomly, filings shifted a little
        js = company_json()
        k = float(rng.uniform(0.5, 2.0))
        for tag in js["facts"]["us-gaap"].values():
            for rows in tag["units"].values():
                for d in rows:
                    d["val"] = d["val"] * k
        jss[t] = js
    member = pd.DataFrame(True, index=close.index, columns=close.columns)
    cutoff = pd.Timestamp("2020-09-01")
    full = build_panel(close, vol, None, raw_metrics(close, jss), member, min_fund=3, horizon21=True)
    part = build_panel(close, vol, None, raw_metrics(close, jss, cutoff), member, min_fund=3, horizon21=True)
    feats = feature_columns(full)
    assert set(FUND_FEATURES) <= set(feats) and not any(c.startswith("fwd_") for c in feats), "labels leaked into features"
    common = full.index.intersection(part.index)
    common = common[common.get_level_values("date") <= cutoff]
    assert len(common) > 500, "test is vacuous"
    diff = (full.loc[common, feats] - part.loc[common, feats]).abs().max().max()
    assert diff < 1e-9, f"features changed when future filings were removed (max diff {diff})"
    f_ok = full[FUND_FEATURES].ne(0).any(axis=1).sum()
    assert f_ok > 1000, "fundamentals are empty: the end-to-end test proves nothing"
    assert full.index.get_level_values("date").min() >= pd.Timestamp("2020-02-21"), "rows before the first filing survived min_fund"


# ---------------------------------------------------------------- SEC client
class FakeResp:
    def __init__(self, body, headers=None):
        self.body, self.headers = body, headers or {}

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_sec_client_rules():
    for bad in ("", "just a name"):
        try:
            fnd.SecClient(bad, tempfile.mkdtemp())
            raise AssertionError("client started without a contact email in the User-Agent")
        except ValueError:
            pass
    t = [0.0]
    lim = fnd.RateLimiter(clock=lambda: t[0], sleep=lambda d: t.__setitem__(0, t[0] + d))
    stamps = []
    for _ in range(40):
        lim.wait()
        stamps.append(t[0])
    gaps = np.diff(stamps)
    assert gaps.min() >= 1 / fnd.MAX_REQUESTS_PER_SEC - 1e-9 and 39 / (stamps[-1] - stamps[0]) <= fnd.MAX_REQUESTS_PER_SEC
    try:
        fnd.RateLimiter(min_interval=0.05)
        raise AssertionError("a limiter faster than 8/s was accepted")
    except ValueError:
        pass
    seen = []

    def opener(req, timeout=0):
        seen.append(req)
        if "CIK0000000002" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, io.BytesIO(b""))
        if "CIK0000000003" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 403, "blocked", {}, io.BytesIO(b""))
        return FakeResp(json.dumps({"facts": {}}).encode())

    d = tempfile.mkdtemp()
    c = fnd.SecClient("Test Person test@example.com", d, limiter=fnd.RateLimiter(clock=lambda: t[0], sleep=lambda x: None), opener=opener)
    assert c.fetch_companyfacts(1).exists() and c.fetch_companyfacts(1).exists()
    assert len(seen) == 1, "second call should come from the cache"
    assert seen[0].get_header("User-agent") == "Test Person test@example.com"
    assert c.fetch_companyfacts(2) is None and c.fetch_companyfacts(2) is None and len(seen) == 2, "404 should be remembered"
    try:
        c.fetch_companyfacts(3)
        raise AssertionError("403 must stop the run")
    except fnd.SecBlocked:
        pass


# ---------------------------------------------------------------- universe
HTML = """<html><table class="wikitable" id="constituents"><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>Date added</th><th>CIK</th></tr>
<tr><td><a>AAA</a></td><td>Aaa Inc</td><td>Tech</td><td>2015-01-05</td><td>0000000111</td></tr>
<tr><td>BRK.B</td><td>Berkshire</td><td>Fin</td><td>2010-02-16</td><td>0001067983</td></tr></table>
<table class="wikitable" id="changes"><tr><th>Date</th><th>Added</th><th>Removed</th><th>Reason</th></tr>
<tr><th></th><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th><th></th></tr>
<tr><td>January 5, 2015</td><td>AAA</td><td>Aaa Inc</td><td>OLD</td><td>Old Corp</td><td>market cap<sup>[1]</sup></td></tr>
<tr><td>June 1, 2012</td><td>OLD</td><td>Old Corp</td><td>GONE</td><td>Gone Ltd</td><td>x</td></tr></table></html>"""


def test_universe_parsing_and_point_in_time_membership():
    cur, chg = uni.parse_wikipedia(HTML)
    assert list(cur["symbol"]) == ["AAA", "BRK-B"] and cur["cik"].tolist() == [111, 1067983]
    assert len(chg) == 2 and chg["removed"].tolist() == ["OLD", "GONE"]
    iv, bad = uni.build_intervals(cur, chg)
    assert bad == 0
    dates = pd.bdate_range("2010-01-01", "2016-12-30")
    m = uni.membership_matrix(iv, dates, ["AAA", "OLD", "GONE", "BRK-B"])
    on = lambda t, d: bool(m.loc[pd.Timestamp(d), t])
    assert not on("AAA", "2014-12-31") and on("AAA", "2015-01-05"), "AAA must enter the universe only when it joined"
    assert on("OLD", "2014-12-31") and not on("OLD", "2015-01-06") and not on("OLD", "2012-05-31") and on("OLD", "2012-06-01")
    assert on("GONE", "2011-01-03") and not on("GONE", "2012-06-01"), "removed company must be a member BEFORE its removal"
    assert on("BRK-B", "2010-03-01") and not on("BRK-B", "2010-02-01")
    assert uni.cik_map(cur, iv) == {"AAA": 111, "BRK-B": 1067983}, "removed tickers must not be auto-mapped to a CIK"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all fundamentals tests passed")
