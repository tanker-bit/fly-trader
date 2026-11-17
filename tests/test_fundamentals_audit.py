import numpy as np
import pandas as pd

from experiments.fundamentals_audit import audit


def synth(n_dates=420, n_stocks=60, seed=0):
    rnd = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(n_stocks)]], names=["date", "ticker"])
    n = len(idx)
    y21 = rnd.normal(size=n)
    y1 = rnd.normal(size=n)
    p = pd.DataFrame({
        "earnings_yield": 0.3 * y21 + rnd.normal(size=n),       # real signal for the 21-day label
        "roe": rnd.normal(size=n),                              # pure noise
        "book_to_price": 0.6 * y1 + rnd.normal(size=n),         # leaks tomorrow's return
        "ret_1": rnd.normal(size=n),
        "fund_known": 1.0,
        "fwd_excess": y1, "fwd_excess_21": y21}, index=idx)
    return p


def test_audit_finds_signal_ignores_noise_and_flags_leak():
    r = audit(synth(), holdout=20, step=3)
    assert r.loc["earnings_yield", "ic_21d"] > 0.1 and r.loc["earnings_yield", "t_stat"] > 3
    assert abs(r.loc["roe", "ic_21d"]) < 0.05
    assert r.loc["book_to_price", "leak_flag"] and not r.loc["earnings_yield", "leak_flag"]
    assert r.loc["earnings_yield", "kind"] == "fundamental" and r.loc["ret_1", "kind"] == "price"


def test_holdout_dates_are_never_used():
    p = synth(n_dates=200)
    d = p.index.get_level_values("date")
    cut = sorted(d.unique())[-60]
    p.loc[d >= cut, "earnings_yield"] = np.random.default_rng(1).normal(size=int((d >= cut).sum()))   # destroy the signal only in the last 60 dates
    r = audit(p, holdout=60, step=2)
    assert r.loc["earnings_yield", "ic_21d"] > 0.1                          # unaffected: those dates were excluded


def test_unknown_fundamentals_are_excluded_from_fundamental_ic():
    p = synth()
    d = p.index.get_level_values("date")
    p.loc[d >= d.unique()[200], "fund_known"] = -1.0
    p.loc[d >= d.unique()[200], "earnings_yield"] = 0.0                      # filler zeros
    r = audit(p, holdout=20, step=3)
    assert r.loc["earnings_yield", "ic_21d"] > 0.1                          # filler rows did not dilute it


def synth_trend(n_dates=900, n_stocks=80, seed=2):
    """Each stock has a fixed feature x and a daily drift proportional to x: the signal builds up with the horizon."""
    rnd = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_dates)
    x = np.linspace(-1, 1, n_stocks)
    r = 0.0008 * x[None, :] + rnd.normal(0, 0.01, (n_dates, n_stocks))
    idx = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(n_stocks)]], names=["date", "ticker"])
    fr = pd.DataFrame(r, dates, [f"S{i}" for i in range(n_stocks)])
    p = pd.DataFrame({"trend": np.tile(x, n_dates), "noise": rnd.normal(size=n_dates * n_stocks), "fund_known": 1.0,
                      "fwd_ret": fr.to_numpy().ravel(), "fwd_excess": fr.sub(fr.mean(axis=1), axis=0).to_numpy().ravel()}, index=idx)
    return p


def test_signal_grows_with_horizon_and_noise_stays_flat():
    from experiments.fundamentals_audit import compare_horizons
    ic, t = compare_horizons(synth_trend(), [5, 63], holdout=50, step=4)
    assert ic.loc["trend", 63] > ic.loc["trend", 5] > 0.02
    assert abs(ic.loc["noise", 63]) < 0.05


def test_long_horizon_labels_never_reach_into_the_holdout():
    from experiments.fundamentals_audit import forward_excess
    p = synth_trend(n_dates=700)
    for holdout, h in [(100, 126), (60, 21), (50, 5)]:
        r = audit(p, holdout=holdout, step=3, horizon=h)
        lab = sorted(forward_excess(p, h).dropna().index.get_level_values("date").unique())
        pos = lab.index(r.attrs["last_date"])
        # the label of the last audited date uses returns up to date pos+h; the holdout starts at len(lab)-holdout
        assert pos + h <= len(lab) - holdout, (holdout, h)
