import types

import numpy as np

from experiments.rules_vs_flies import rule_ics, rule_scores

FEATS = ["ma50_dist", "ret_20", "rsi_14", "mom_12_1"]


def make_data(n_days=200, n=60, seed=0, signal=0.5):
    rng = np.random.default_rng(seed)
    day = []
    for _ in range(n_days):
        X = rng.uniform(-1, 1, (n, 4)).astype(np.float32)
        y = -signal * X[:, 0] + rng.normal(size=n)           # reversal: low ma50_dist -> high label
        day.append((np.array([f"T{i}" for i in range(n)]), X, y))

    def geom(r):
        c = 59 + 20 * r
        return c, c + 5, c + 5 + 20

    def stack(upto):
        return np.vstack([day[i][1] for i in range(upto + 1)]), np.concatenate([day[i][2] for i in range(upto + 1)])

    return types.SimpleNamespace(day=day, feats=FEATS, round_geometry=geom, stack_train=stack)


def test_rule_scores_signs():
    d = make_data()
    X = d.day[0][1]
    assert np.allclose(rule_scores(d, "-ma50_dist", X), -X[:, 0])
    assert np.allclose(rule_scores(d, "momentum 12-1", X), X[:, 3])


def test_reversal_rule_finds_planted_signal_and_momentum_does_not():
    d = make_data(signal=0.8)
    t = rule_ics(d, [0, 1, 2], score_every=2)
    assert list(t.index) == [0, 1, 2] and (t["-ma50_dist"] > 0.15).all()
    assert abs(t["momentum 12-1"]).max() < 0.15
    assert (t["ridge"] > 0.1).all()                        # ridge fitted on the past finds it too


def test_uses_the_leagues_validation_dates_only():
    d = make_data(signal=0.8)
    seen = []
    orig = d.day
    class Spy(list):
        def __getitem__(self, i):
            seen.append(i)
            return orig[i]
    d.day = Spy(orig)
    rule_ics(d, [1], score_every=4, rules=["-ma50_dist"])
    c, v0, v1 = d.round_geometry(1)
    assert sorted(set(seen)) == list(range(v0, v1, 4))     # touches no other date (ridge not requested here)


def test_summarize_paired_t_sign():
    import pandas as pd
    from experiments.rules_vs_flies import summarize
    t = pd.DataFrame({"fly mean": [0.02, 0.03, 0.01, 0.04], "committee": [0.02] * 4, "-ma50_dist": [0.05, 0.06, 0.04, 0.07]})
    s = summarize(t, ["-ma50_dist"])
    assert s.loc["-ma50_dist", "fly_minus_this"] < 0 and s.loc["-ma50_dist", "paired_t"] < 0
    assert s.loc["fly mean", "rounds_positive"] == 4
