import types

import numpy as np
import pandas as pd

from experiments.rules_dollars import NEVER, round_accounts, rule_score_dict

FEATS = ["earnings_yield", "fcf_yield", "roe", "roa", "accruals", "book_to_price", "ma50_dist", "mom_12_1", "fund_known"]


def make(n_days=140, n=40, seed=0, signal=0.004):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tick = np.array([f"T{i}" for i in range(n)])
    quality = rng.normal(size=n)                                  # fcf_yield is a persistent trait that drives returns
    day, rets = [], {}
    for d in dates:
        X = rng.normal(size=(n, 9)).astype(np.float32)
        X[:, 1] = quality
        X[:, 8] = 1.0
        X[:5, 8] = -1.0                                           # first five stocks have no real fundamentals
        day.append((tick, X, np.zeros(n)))
        rets[d] = pd.Series(signal * quality + rng.normal(0, 0.01, n), index=tick)
    fwd1 = pd.DataFrame(rets).T
    return types.SimpleNamespace(day=day, dates=list(dates), feats=FEATS, fwd1=fwd1, n_usable=n_days,
                                 round_geometry=lambda r: (20 + 30 * r, 25 + 30 * r, 85 + 30 * r))


def test_unreal_fundamentals_never_picked():
    d = make()
    s = rule_score_dict(d, ["+fcf_yield"], [30, 31])
    assert (s[30][:5] == NEVER).all() and (s[30][5:] > NEVER).all()


def test_planted_fundamentals_rule_makes_money_vs_benchmark():
    d = make()
    t = round_accounts(d, 0, score_every=5, k=8, every=21)
    assert list(t.columns[:3]) == ["total_return", "bench_return", "excess_return"]
    assert "reversal -ma50_dist" in t.index and "momentum 12-1" in t.index
    assert t.loc["fcf_yield", "excess_return"] > 0.02
    assert abs(t.loc["momentum 12-1", "excess_return"]) < t.loc["fcf_yield", "excess_return"]
    assert (t["fees"] >= 0).all()


def test_summarize_vs_random():
    from experiments.rules_dollars import summarize
    tot = pd.DataFrame({"a": [0.10, 0.12, 0.08, 0.11], "b": [0.02, 0.0, 0.03, 0.01]})
    exc = tot - 0.05
    s = summarize(exc, tot, pd.Series([0.05] * 4))
    assert s.loc["a", "rounds_beat_random"] == 4 and s.loc["b", "rounds_beat_bench"] == 0
    assert s.loc["a", "t_vs_random"] > 3 and s.loc["b", "mean_vs_random"] < 0
