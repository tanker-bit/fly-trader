"""Run: python -m pytest tests/test_fundamentals_v2.py -q

The nine richer SEC features: arithmetic, no look-ahead (truncation invariance), SUE, the within-sector rank, the panel merge, and
that the original metrics and old cached event tables are unaffected."""
import numpy as np
import pandas as pd

from env import fundamentals as fnd
from env.features import sector_xs_rank
from experiments.add_fund_v2 import NEW, add_v2
from tests.test_fundamentals import company_json, fact, trading_dates


def rich_json():
    js = company_json()
    usd = lambda rows: {"units": {"USD": rows}}
    f = js["facts"]["us-gaap"]
    f["OperatingIncomeLoss"] = usd([
        fact("2019-12-31", 200, "2020-02-20", "10-K", "2019-01-01"),
        fact("2020-03-31", 70, "2020-05-05", "10-Q", "2020-01-01"), fact("2019-03-31", 50, "2020-05-05", "10-Q", "2019-01-01")])
    f["InterestExpense"] = usd([
        fact("2019-12-31", 20, "2020-02-20", "10-K", "2019-01-01"),
        fact("2020-03-31", 6, "2020-05-05", "10-Q", "2020-01-01"), fact("2019-03-31", 4, "2020-05-05", "10-Q", "2019-01-01")])
    f["PaymentsForRepurchaseOfCommonStock"] = usd([
        fact("2019-12-31", 40, "2020-02-20", "10-K", "2019-01-01"),
        fact("2020-03-31", 15, "2020-05-05", "10-Q", "2020-01-01"), fact("2019-03-31", 5, "2020-05-05", "10-Q", "2019-01-01")])
    f["AssetsCurrent"] = usd([fact("2020-03-31", 900, "2020-05-05", "10-Q")])
    f["LiabilitiesCurrent"] = usd([fact("2020-03-31", 600, "2020-05-05", "10-Q")])
    f["Assets"]["units"]["USD"].append(fact("2019-03-31", 1800, "2019-05-05", "10-Q"))     # one year before the Q1-2020 balance sheet
    f["LongTermDebt"] = usd([fact("2020-03-31", 420, "2020-05-05", "10-Q")])
    return js


def events(js):
    return fnd.build_events(fnd.compact_companyfacts(js), 1)


def on(ev, dates, name, close=None):
    close = close if close is not None else pd.Series(10.0, index=dates)
    return pd.Series(fnd.align_metrics(ev, dates, close, None, extended=True)[name], index=dates)


def test_new_metrics_arithmetic():
    ev, dates = events(rich_json()), trading_dates()
    d = pd.Timestamp("2020-05-06")                       # the day after the Q1-2020 10-Q is filed
    # TTM: FY2019 + Q1-2020 - Q1-2019 ; revenue 1000 + 300 - 200 = 1100 ; opinc 200 + 70 - 50 = 220 ; interest 20 + 6 - 4 = 22
    assert np.isclose(on(ev, dates, "op_margin")[d], 220 / 1100)
    assert np.isclose(on(ev, dates, "interest_cover")[d], 220 / 22)
    assert np.isclose(on(ev, dates, "current_ratio")[d], 900 / 600)
    assert np.isclose(on(ev, dates, "asset_growth")[d], 2100 / 1800 - 1)
    assert np.isclose(on(ev, dates, "debt_to_assets")[d], 420 / 2100)
    assert np.isclose(on(ev, dates, "margin_change")[d], 70 / 300 - 50 / 200)
    mcap = 100 * 10.0                                    # shares 100 x price 10
    assert np.isclose(on(ev, dates, "buyback_yield")[d], (40 + 15 - 5) / mcap)


def test_nothing_new_is_visible_before_it_is_filed():
    ev, dates = events(rich_json()), trading_dates()
    assert on(ev, dates, "current_ratio")[:pd.Timestamp("2020-05-05")].isna().all(), "balance-sheet ratio seen on or before its filing date"
    assert on(ev, dates, "op_margin")[:pd.Timestamp("2020-02-20")].isna().all()


def test_truncation_invariance_for_the_new_metrics():
    js, dates = rich_json(), trading_dates()
    cutoff = pd.Timestamp("2020-06-15")
    full = events(js)
    cut = fnd.compact_companyfacts(js)
    part = fnd.build_events(cut[cut["filed"] <= cutoff])
    px = pd.Series(np.linspace(8, 14, len(dates)), index=dates)
    a, b = fnd.align_metrics(full, dates, px, None, extended=True), fnd.align_metrics(part, dates, px, None, extended=True)
    m = dates <= cutoff
    for k in fnd.METRICS + fnd.METRICS_V2:
        assert np.allclose(a[k][m], b[k][m], equal_nan=True), f"{k} changed when later filings were removed"


def test_extended_flag_leaves_the_original_metrics_alone_and_old_caches_work():
    js, dates = rich_json(), trading_dates()
    ev = events(js)
    px = pd.Series(10.0, index=dates)
    base, ext = fnd.align_metrics(ev, dates, px), fnd.align_metrics(ev, dates, px, None, extended=True)
    assert set(base) == set(fnd.METRICS) and set(ext) == set(fnd.METRICS + fnd.METRICS_V2)
    for k in fnd.METRICS:
        assert np.allclose(base[k], ext[k], equal_nan=True)
    old = ev.drop(columns=[c for c in ("opinc_ttm", "interest_ttm", "repurch_ttm", "cur_assets", "cur_liab", "assets_prev",
                                       "opinc_cur", "opinc_prev", "sue") if c in ev.columns])
    out = fnd.align_metrics(old, dates, px, None, extended=True)           # an event table from before this change
    assert all(np.isnan(out[k]).all() for k in ("op_margin", "current_ratio", "sue"))


def eps_json(values):
    """Quarterly EPS facts, 10-Qs filed 35 days after each quarter end (three quarters a year; here simply every quarter)."""
    ends = pd.date_range("2018-03-31", periods=len(values), freq="QE")
    rows = []
    for e, v in zip(ends, values):
        rows.append(fact(e.strftime("%Y-%m-%d"), v, (e + pd.Timedelta(days=35)).strftime("%Y-%m-%d"), "10-Q",
                         (e - pd.Timedelta(days=89)).strftime("%Y-%m-%d")))
    return {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": rows}}}}}


EPS = [1.0, 1.1, 1.2, 1.3, 1.5, 1.5, 1.7, 1.6, 1.9, 1.8, 2.2, 2.5]


def test_sue_matches_the_definition_and_ignores_the_future():
    ev = events(eps_json(EPS))
    ch = np.array([EPS[i] - EPS[i - 4] for i in range(4, 12)])
    want = (ch[-1] - ch[:-1].mean()) / ch[:-1].std(ddof=1)
    assert np.isclose(ev["sue"].iloc[-1], want), (ev["sue"].iloc[-1], want)
    ev2 = events(eps_json(EPS[:11]))                       # drop the last quarter: the earlier snapshot must not change
    assert np.isclose(ev["sue"].iloc[-2], ev2["sue"].iloc[-1])
    assert ev["sue"].iloc[:5].isna().all(), "too little history must give NaN, not a made-up number"


def test_sector_rank_is_within_the_sector_and_needs_enough_peers():
    idx = pd.bdate_range("2020-01-01", periods=2)
    cols = [f"A{i}" for i in range(6)] + [f"B{i}" for i in range(3)]
    df = pd.DataFrame([list(range(6)) + [100, 200, 300]] * 2, index=idx, columns=cols, dtype=float)
    sec = {c: ("tech" if c[0] == "A" else "oil") for c in cols}
    r = sector_xs_rank(df, sec, min_group=6)
    assert r[[f"A{i}" for i in range(6)]].iloc[0].is_monotonic_increasing and r["A5"].iloc[0] == 1.0 and r["A0"].iloc[0] < -0.6
    assert r[["B0", "B1", "B2"]].isna().all().all(), "a 3-stock sector must not be ranked"
    unk = sector_xs_rank(df[["A0", "A1"]], {}, min_group=2)                # names with no sector share the 'unknown' group
    assert unk["A1"].iloc[0] > unk["A0"].iloc[0]


def test_panel_merge_keeps_old_columns_and_zero_fills_unknowns():
    dates = pd.bdate_range("2020-01-01", periods=5)
    tick = [f"S{i}" for i in range(8)]
    idx = pd.MultiIndex.from_product([dates, tick], names=["date", "ticker"])
    panel = pd.DataFrame({"ret_1": np.linspace(-1, 1, len(idx)), "fwd_ret": 0.01}, index=idx)
    raw = {m: pd.DataFrame(np.arange(8, dtype=float)[None, :].repeat(5, 0), index=dates, columns=tick) for m in NEW}
    raw["sue"].iloc[:, :] = np.nan                                         # one metric unknown everywhere
    out = add_v2(panel, raw, {t: "x" for t in tick}, min_group=6)
    assert (out["ret_1"] == panel["ret_1"]).all() and (out["fwd_ret"] == panel["fwd_ret"]).all()
    assert (out["s_sue"] == 0).all() and out["s_op_margin"].abs().max() <= 1.0
    top = out.xs("S7", level="ticker")["s_op_margin"]
    assert (top == 1.0).all()
    assert np.allclose(out["fund2_known"], 8 / 9 * 2 - 1)
