import numpy as np

from experiments.blend_dollars import blend_score_dicts, round_blend_accounts
from experiments.rules_dollars import NEVER
from tests.test_rules_dollars import make as _make


def make(**kw):
    d = _make(**kw)
    d.stack_train = lambda upto: (np.vstack([x[1] for x in d.day[:upto + 1]]), np.concatenate([x[2] for x in d.day[:upto + 1]]))
    return d


def fly_dicts(data, r, se=5, seed=3):
    c, v0, v1 = data.round_geometry(r)
    rng = np.random.default_rng(seed)
    return [{i: rng.normal(size=len(data.day[i][0])) for i in range(v0, v1, se)} for _ in range(3)]


def test_scores_never_pick_stocks_without_real_fundamentals_and_cover_all_variants():
    d = make()
    sd = blend_score_dicts(d, 0, fly_dicts(d, 0), 5)
    assert set(sd) == {"fly", "ridge", "fund", "fly+fund", "ridge+fund", "fly+ridge+fund"}
    for v, dd in sd.items():
        i = next(iter(dd))
        assert (dd[i][:5] == NEVER).all() and (dd[i][5:] > NEVER).all()


def test_missing_flies_drop_the_fly_variants_only():
    d = make()
    sd = blend_score_dicts(d, 0, [], 5)
    assert set(sd) == {"ridge", "fund", "ridge+fund"}


def test_planted_fundamentals_make_the_fund_blends_money_and_noise_flies_do_not():
    d = make(signal=0.004)
    t = round_blend_accounts(d, 0, fly_dicts(d, 0), 5, k=8, every=21)
    assert list(t.columns[:3]) == ["total_return", "bench_return", "excess_return"]
    assert t.loc["fund", "excess_return"] > 0.02
    assert t.loc["fly+fund", "excess_return"] > t.loc["fly", "excess_return"]
    assert (t["fees"] >= 0).all()
