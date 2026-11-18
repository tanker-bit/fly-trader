import numpy as np

from experiments.decode_fly import decode_scores

FEATS = ["a", "b", "c", "d"]


def _days(n_days=20, n=80, seed=0):
    rng = np.random.default_rng(seed)
    Xs = [rng.uniform(-1, 1, (n, 4)) for _ in range(n_days)]
    ys = [x[:, 2] * 0.3 + rng.normal(size=n) for x in Xs]
    return rng, Xs, ys


def test_a_fly_that_relays_one_feature_is_detected():
    rng, Xs, ys = _days()
    sc = [x[:, 2] + rng.normal(0, 0.05, len(x)) for x in Xs]
    d = decode_scores(Xs, sc, ys, FEATS)
    assert d["feature_corr"].abs().idxmax() == "c" and d["feature_corr"]["c"] > 0.95
    assert d["r2_all_features"] > 0.9 and d["ic"] > 0.1


def test_a_fly_unrelated_to_the_inputs_has_low_r2_and_low_correlations():
    rng, Xs, ys = _days()
    sc = [rng.normal(size=len(x)) for x in Xs]
    d = decode_scores(Xs, sc, ys, FEATS)
    assert d["feature_corr"].abs().max() < 0.15 and d["r2_all_features"] < 0.15


def test_ridge_correlation_and_signs():
    rng, Xs, ys = _days()
    ridge = [x @ np.array([0.0, 1.0, -1.0, 0.0]) for x in Xs]
    sc = [r + rng.normal(0, 0.05, len(r)) for r in ridge]
    d = decode_scores(Xs, sc, ys, FEATS, ridge)
    assert d["ridge_corr"] > 0.95 and d["feature_corr"]["c"] < -0.3 < 0.3 < d["feature_corr"]["b"]
