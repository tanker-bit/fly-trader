"""Fast checks for the evolution pieces (no simulation needed)."""
import numpy as np

from evolution.evaluate import ic, null_bar
from evolution.genome import BOUNDS, KC_INPUTS, Genome, crossover, mutate, random_genome
from evolution.synthetic import PATTERNS, make_block, make_splits


def test_mutation_respects_bounds():
    rng = np.random.default_rng(0)
    g = Genome()
    for _ in range(200):
        g = mutate(g, rng, p=1.0, sigma=1.0)
        for k, (lo, hi, _) in BOUNDS.items():
            assert lo <= getattr(g, k) <= hi
        assert KC_INPUTS[0] <= g.kc_inputs <= KC_INPUTS[1]


def test_crossover_genes_come_from_parents():
    rng = np.random.default_rng(1)
    a, b = random_genome(rng), random_genome(rng)
    c = crossover(a, b, rng)
    for k, v in c.to_dict().items():
        assert v in (a.to_dict()[k], b.to_dict()[k])


def test_planted_signal_is_unit_std_and_xor_is_not_linear():
    X, y, f = make_block("xor", 20000, 0)
    assert abs(np.std(f) - 1.0) < 0.05
    assert abs(np.corrcoef(X[:, 0], y)[0, 1]) < 0.03   # no single-feature (linear) signal
    for p in ("linear", "sparse2"):
        _, _, f = make_block(p, 20000, 0)
        assert abs(np.std(f) - 1.0) < 0.05
    assert np.all(make_block("none", 10, 0)[2] == 0)


def test_oracle_ic_is_positive_for_planted_patterns():
    for p in PATTERNS:
        X, y, f = make_block(p, 5000, 3)
        if p != "none":
            assert ic(f, y) > 0.2


def test_luck_bar_grows_with_trials_and_shrinks_with_data():
    assert null_bar(500, 800) > null_bar(10, 800)
    assert null_bar(100, 4000) < null_bar(100, 800)


def test_splits_are_disjoint_draws():
    s = make_splits("linear")
    assert not np.allclose(s["train"][0][:5], s["holdout"][0][:5])
