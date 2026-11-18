import types

import numpy as np
import pandas as pd

from experiments.fund_rules import fund_rule_ics, rule_score

FEATS = ["earnings_yield", "fcf_yield", "roe", "roa", "accruals", "book_to_price", "fund_known"]


def make(n_days=80, n=50, seed=0, planted=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tick = np.array([f"T{i}" for i in range(n)])
    day, lab = [], {}
    for d in dates:
        X = rng.normal(size=(n, 7)).astype(np.float32)
        X[:, 6] = 1.0                                              # fundamentals real
        y = (X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3] - X[:, 4]) * (0.5 if planted else 0) + rng.normal(size=n)
        day.append((tick, X, y)); lab[d] = pd.Series(y, index=tick)
    data = types.SimpleNamespace(day=day, dates=list(dates), feats=FEATS, n_usable=60,
                                 round_geometry=lambda r: (20 + 10 * r, 25 + 10 * r, 45 + 10 * r))
    return data, pd.DataFrame(lab).T


def test_planted_value_quality_signal_found():
    d, lw = make()
    t = fund_rule_ics(d, lw, [0, 1], score_every=2, horizon=5)
    assert (t["value+quality"] > 0.3).all() and (t["-accruals"] > 0.1).all()


def test_no_signal_gives_near_zero():
    d, lw = make(planted=False)
    t = fund_rule_ics(d, lw, [0, 1], score_every=2, horizon=5)
    assert abs(t["value+quality"]).max() < 0.2


def test_dates_reaching_holdout_are_skipped():
    d, lw = make()
    t = fund_rule_ics(d, lw, [1], score_every=2, horizon=30)       # 35.. + 30 > 60 for most of the window
    assert t.shape == (1, 7)
    assert fund_rule_ics(d, lw, [1], score_every=2, horizon=100).isna().all().all()


def test_rule_score_sign():
    d, _ = make()
    X = d.day[0][1]
    assert np.corrcoef(rule_score(d, X, ["-accruals"]), X[:, 4])[0, 1] < -0.99


def test_summarize():
    from experiments.fund_rules import summarize
    t = pd.DataFrame({"a": [0.1, 0.2, 0.15, 0.12], "b": [-0.1, 0.05, -0.2, 0.0]})
    s = summarize(t)
    assert s.loc["a", "t"] > 3 and s.loc["b", "rounds_positive"] == 1 and (s["rounds"] == 4).all()
