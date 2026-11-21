import types

import numpy as np
import pandas as pd

from experiments.blend_test import BLENDS, blend_ics, summarize

FEATS = ["earnings_yield", "fcf_yield", "roe", "roa", "accruals", "book_to_price", "ma50_dist", "f1", "fund_known"]


def make(n_days=120, n=60, seed=0, w_fund=0.5, w_rev=0.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tick = np.array([f"T{i}" for i in range(n)])
    day, lab = [], {}
    for d in dates:
        X = rng.normal(size=(n, 9)).astype(np.float32)
        X[:, 8] = 1.0
        y = w_fund * (X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3] - X[:, 4]) - w_rev * X[:, 6] + rng.normal(size=n)
        day.append((tick, X, y))
        lab[d] = pd.Series(y, index=tick)
    Xtr = np.vstack([x[1] for x in day]); ytr = np.concatenate([x[2] for x in day])
    data = types.SimpleNamespace(day=day, dates=list(dates), feats=FEATS, n_usable=100,
                                 round_geometry=lambda r: (40 + 10 * r, 45 + 10 * r, 75 + 10 * r),
                                 stack_train=lambda upto: (np.vstack([x[1] for x in day[:upto + 1]]), np.concatenate([x[2] for x in day[:upto + 1]])))
    return data, pd.DataFrame(lab).T


def fly_dicts(data, rounds, noise, seed=1, se=3):
    """Fake flies: the true label plus noise."""
    rng = np.random.default_rng(seed)
    out = {}
    for r in rounds:
        c, v0, v1 = data.round_geometry(r)
        out[r] = [{i: data.day[i][2] + noise * rng.normal(size=len(data.day[i][2])) for i in range(v0, v1, se)} for _ in range(4)]
    return out


def test_blend_with_a_good_part_beats_a_pure_noise_part_and_shapes_are_right():
    data, lw = make()
    flies = fly_dicts(data, [0, 1, 2], noise=6.0)                    # weak flies
    tabs, corr = blend_ics(data, flies, [0, 1, 2], 3, {"fwd5d": (lw, 5)})
    t = tabs["league"]
    assert list(t.columns) == list(BLENDS) and len(t) == 3
    assert t["fund"].mean() > 0.2 and t["rev"].abs().mean() < 0.2    # fundamentals carry the planted signal, reversal none
    assert t["fly+fund"].mean() > t["fly"].mean()                    # mixing a good part into a weak one helps
    assert tabs["fwd5d"].shape == t.shape
    assert corr.shape == (4, 4) and np.allclose(np.diag(corr.to_numpy()), 1.0, atol=1e-6)


def test_dead_flies_are_left_out_and_missing_flies_skip_fly_blends():
    data, lw = make()
    flies = {0: [{i: np.zeros(60) for i in range(45, 75, 3)}], 1: []}
    tabs, _ = blend_ics(data, flies, [0, 1], 3)
    t = tabs["league"]
    assert t["fly"].isna().all() and t["fly+ridge+fund"].isna().all()
    assert t["ridge"].notna().all() and t["fund"].notna().all()


def test_horizon_past_the_holdout_is_skipped():
    data, lw = make()
    tabs, _ = blend_ics(data, {}, [1], 3, {"fwd60d": (lw, 60)})      # 55.. + 60 > 100 for every date
    assert tabs["fwd60d"].isna().all().all() and tabs["league"]["fund"].notna().all()


def test_summarize_compares_with_the_best_single_component():
    t = pd.DataFrame({"fly": [0.01, 0.02, 0.0, 0.01], "ridge": [0.05, 0.04, 0.06, 0.05], "fund": [0.02, 0.0, 0.03, 0.01],
                      "rev": [0.0, 0.01, 0.0, 0.0], "fly+ridge": [0.06, 0.05, 0.07, 0.06]})
    s = summarize(t)
    assert s.loc["ridge", "paired_t"] != s.loc["ridge", "paired_t"]            # NaN for the best single itself
    assert s.loc["fly+ridge", "minus ridge"] > 0 and s.loc["fly+ridge", "rounds_positive"] == 4
