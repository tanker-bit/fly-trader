import types

import numpy as np
import pandas as pd

from accounts.league_report import prices_from_returns, scores_frame, account_report, equal_weight_benchmark


def fake_data(n=120, m=25, seed=1):
    rnd = np.random.default_rng(seed)
    dates = list(pd.bdate_range("2021-01-01", periods=n))
    tick = np.array([f"T{i}" for i in range(m)])
    drift = np.linspace(-0.001, 0.001, m)
    fwd1 = pd.DataFrame(drift + rnd.normal(0, 0.01, (n, m)), dates, tick)
    d = types.SimpleNamespace(dates=dates, fwd1=fwd1, day=[(tick, None, None)] * n)
    return d, drift


def test_prices_from_returns_compounds_and_masks_gaps():
    f = pd.DataFrame({"A": [0.1, 0.1, np.nan], "B": [0.0, 0.0, 0.0]}, pd.bdate_range("2021-01-01", periods=3))
    p = prices_from_returns(f, [0, 1, 2])
    assert p["A"].iloc[0] == 100 and abs(p["A"].iloc[1] - 110) < 1e-9 and abs(p["A"].iloc[2] - 121) < 1e-9
    f2 = f.copy(); f2.iloc[1, 0] = np.nan
    assert np.isnan(prices_from_returns(f2, [0, 1, 2])["A"].iloc[2])


def test_report_good_scores_beat_benchmark_and_rules_hold():
    d, drift = fake_data()
    idxs = range(0, 120, 5)
    good = {i: drift for i in idxs}
    bad = {i: -drift for i in idxs}
    ag, eg, mg = account_report(d, good, 0, 120, k=5, every=10)
    ab, eb, mb = account_report(d, bad, 0, 120, k=5, every=10)
    for a in (ag, ab):
        assert a.cash >= 0 and not a.rejected and all(s > 0 for s in a.pos.values())
    assert mg["excess_return"] > mb["excess_return"] and mg["trades"] > 0 and mg["fees"] > 0
    assert mg["bench_return"] == mb["bench_return"]


def test_scores_frame_shape():
    d, drift = fake_data()
    f = scores_frame({0: drift, 5: drift}, d, range(0, 10))
    assert f.shape == (2, 25)


def test_random_control_is_noise_and_rules_hold():
    from accounts.league_report import random_control, momentum_scores
    d, drift = fake_data()
    d.i_mom = 0
    sd = {i: drift for i in range(0, 120, 5)}
    r = random_control(d, sd, 0, 120, n=12, workers=1, k=5, every=10)
    assert r.shape == (12,) and np.isfinite(r).all() and r.std() > 0
    r2 = random_control(d, sd, 0, 120, n=12, workers=1, k=5, every=10)
    assert np.allclose(r, r2)                       # seeded, reproducible
    good = account_report(d, sd, 0, 120, k=5, every=10)[2]["total_return"]
    assert good > np.percentile(r, 50)              # real signal beats typical noise
