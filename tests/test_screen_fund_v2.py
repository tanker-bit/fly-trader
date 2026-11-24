"""Run: python -m pytest tests/test_screen_fund_v2.py -q"""
import numpy as np
import pandas as pd

from experiments.screen_fund_v2 import by_year, rank_ic_rows, screen, verdict


def panel_and_label(n_dates=800, n=60, seed=0, skill=0.5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-06-01", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(n)]], names=["date", "ticker"])
    signal = rng.normal(size=len(idx))
    y = skill * signal + rng.normal(size=len(idx))
    p = pd.DataFrame({"good": np.clip(signal / 3, -1, 1), "noise": np.clip(rng.normal(size=len(idx)) / 3, -1, 1)}, index=idx)
    return p, pd.Series(y, index=idx)


def test_rank_ic_rows_matches_scipy_style_spearman():
    rng = np.random.default_rng(1)
    F = pd.DataFrame(rng.normal(size=(5, 50))); Y = pd.DataFrame(0.7 * F.to_numpy() + rng.normal(size=(5, 50)))
    ic = rank_ic_rows(F, Y, min_n=10)
    want = [F.iloc[i].corr(Y.iloc[i], method="spearman") for i in range(5)]
    assert np.allclose(ic.to_numpy(), want)


def test_a_planted_feature_survives_and_noise_fails():
    p, y = panel_and_label()
    res = screen(p, {"lab": y}, ["good", "noise"], start="2010-06-01", end="2013-12-31", step=5)
    r = res.set_index("feature")
    assert r.loc["good", "verdict"] == "SURVIVES" and r.loc["good", "ic"] > 0.15
    assert r.loc["noise", "verdict"] != "SURVIVES" and abs(r.loc["noise", "ic"]) < 0.05


def test_only_dates_inside_the_window_are_used():
    p, y = panel_and_label()
    y2 = y.copy()
    late = y2.index.get_level_values("date") > pd.Timestamp("2012-12-31")
    y2[late] = np.random.default_rng(3).normal(size=int(late.sum()))               # wreck the label after the window
    a = screen(p, {"l": y}, ["good"], start="2010-06-01", end="2012-12-31", step=5)
    b = screen(p, {"l": y2}, ["good"], start="2010-06-01", end="2012-12-31", step=5)
    assert np.isclose(a["ic"].iloc[0], b["ic"].iloc[0])


def test_filler_zeros_are_not_scored():
    p, y = panel_and_label()
    p2 = p.copy()
    p2.loc[p2.index.get_level_values("date") < pd.Timestamp("2011-06-01"), "good"] = 0.0     # "unknown" early on
    res = screen(p2, {"l": y}, ["good"], start="2010-06-01", end="2013-12-31", step=5)
    full = screen(p, {"l": y}, ["good"], start="2010-06-01", end="2013-12-31", step=5)
    assert res["dates"].iloc[0] < full["dates"].iloc[0]


def test_verdict_rules():
    assert verdict(3.0, 11, 13) == "SURVIVES"
    assert verdict(-3.0, 2, 13) == "SURVIVES"                  # a reliably NEGATIVE feature is also a signal (use the opposite sign)
    assert verdict(3.0, 8, 13) == "weak"                       # big t but inconsistent across years
    assert verdict(2.2, 12, 13) == "weak"
    assert verdict(1.0, 13, 13) == "fails"
