import types

import numpy as np
import pandas as pd

from experiments.selection_noise import ic_matrix, pair_stats, summarize


def test_persistence_is_high_when_skill_is_persistent_and_zero_when_it_is_luck():
    rng = np.random.default_rng(0)
    ids = [f"a{i}" for i in range(20)]
    skill = pd.Series(rng.normal(0, 0.03, 20), index=ids)
    persistent = pd.DataFrame({r: skill + rng.normal(0, 0.005, 20) for r in range(6)})
    persistent.index = ids
    luck = pd.DataFrame({r: rng.normal(0, 0.03, 20) for r in range(6)}, index=ids)
    assert summarize(pair_stats(persistent)).loc["persistence", "mean"] > 0.8
    assert abs(summarize(pair_stats(luck)).loc["persistence", "mean"]) < 0.3


def test_kept_vs_new_and_top_vs_bottom_use_only_flies_present_in_both_rounds():
    m = pd.DataFrame({0: [0.05, 0.04, 0.03, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan],
                      1: [0.05, 0.04, 0.03, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02, -0.03]},
                     index=[f"a{i}" for i in range(12)])
    p = pair_stats(m, min_flies=6)
    row = p.loc["0->1"]
    assert row["n_both"] == 10 and row["n_new"] == 2
    assert row["kept_minus_new"] > 0.02 and row["top_minus_bottom"] > 0.029 and row["persistence"] > 0.8


def test_rounds_that_are_not_consecutive_are_skipped_and_summary_counts_pairs():
    m = pd.DataFrame({0: [0.01] * 8, 2: [0.02] * 8}, index=[f"a{i}" for i in range(8)])
    assert pair_stats(m).empty
    p = pd.DataFrame({"persistence": [0.2, 0.3, np.nan], "kept_minus_new": [0.01, -0.01, 0.02], "top_minus_bottom": [0.0, 0.0, 0.0]})
    s = summarize(p)
    assert s.loc["persistence", "pairs"] == 2 and s.loc["kept_minus_new", "pairs_positive"] == 2


def test_ic_matrix_reads_saved_scores(tmp_path):
    root = tmp_path
    sd = root / "rounds" / "scores"
    sd.mkdir(parents=True)
    rng = np.random.default_rng(1)
    day = [(np.arange(30), None, rng.normal(size=30)) for _ in range(6)]
    for i in range(6):
        day[i] = (day[i][0], None, day[i][2])
    good = {f"d{i}": day[i][2] + 0.1 * rng.normal(size=30) for i in range(3)}
    dead = {f"d{i}": np.zeros(30) for i in range(3)}
    np.savez(sd / "r00_a0001.npz", **good)
    np.savez(sd / "r00_a0002.npz", **dead)
    m = ic_matrix(types.SimpleNamespace(root=root), types.SimpleNamespace(day=day), [0])
    assert m.loc["a0001", 0] > 0.8 and abs(m.loc["a0002", 0]) < 1e-9 or np.isnan(m.loc["a0002", 0])
