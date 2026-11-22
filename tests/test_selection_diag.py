import types

import numpy as np
import pandas as pd

from experiments.selection_diag import diversity, genome_effects, lift_detection


def make_day(n_dates=12, n=80, seed=0):
    rng = np.random.default_rng(seed)
    return [(np.arange(n), None, rng.normal(size=n)) for _ in range(n_dates)]


def test_diversity_is_one_for_copies_and_near_zero_for_independent_flies():
    rng = np.random.default_rng(1)
    base = {i: rng.normal(size=60) for i in range(8)}
    copies = {f"a{k}": {i: base[i] * (1 + 0.1 * k) for i in base} for k in range(5)}
    indep = {f"a{k}": {i: rng.normal(size=60) for i in base} for k in range(5)}
    assert diversity(copies)["mean_corr"] > 0.99
    assert abs(diversity(indep)["mean_corr"]) < 0.1
    dead = dict(copies, dead={i: np.zeros(60) for i in base})
    assert diversity(dead)["flies"] == 5                                     # the silent fly is left out


def test_genome_effects_finds_a_planted_setting_and_ignores_a_noise_setting():
    rng = np.random.default_rng(2)
    ids = [f"a{i}" for i in range(30)]
    lr = pd.Series(rng.uniform(size=30), index=ids)
    noise = pd.Series(rng.uniform(size=30), index=ids)
    m = pd.DataFrame({r: 0.05 * lr + rng.normal(0, 0.01, 30) for r in range(8)})
    m.index = ids
    ge = genome_effects(m, {a: dict(lr0=float(lr[a]), kc_drive=float(noise[a]), wiring_seed=0) for a in ids})
    assert ge.loc["lr0", "mean_corr"] > 0.6 and ge.loc["lr0", "t"] > 5
    assert abs(ge.loc["kc_drive", "mean_corr"]) < 0.25
    assert "wiring_seed" not in ge.index                                     # a constant gene has nothing to test


def test_planted_skill_is_detected_more_often_as_the_dose_grows():
    day = make_day()
    data = types.SimpleNamespace(day=day)
    rng = np.random.default_rng(3)
    flies = {f"a{k}": {i: day[i][2] * 0.02 + rng.normal(size=80) for i in range(len(day))} for k in range(10)}   # near-pure noise flies
    lf = lift_detection(data, flies)
    assert list(lf.index) == [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
    assert lf["ic_gain"].is_monotonic_increasing and lf.loc[0.0, "ic_gain"] == 0
    assert lf.loc[0.4, "first"] == 1.0 and lf.loc[0.4, "top_half"] == 1.0
    assert lf.loc[0.0, "first"] == 0.0 or lf.loc[0.0, "first"] == 1.0         # one round: a 0/1 outcome


def test_too_few_flies_gives_an_empty_table():
    assert lift_detection(types.SimpleNamespace(day=make_day()), {"a": {0: np.zeros(80)}}).empty
