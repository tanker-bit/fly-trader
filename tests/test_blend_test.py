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


def test_horizon_table_lines_up_components_and_horizons():
    from experiments.blend_test import horizon_table
    a = pd.DataFrame({"fly": [0.02, 0.03, 0.01], "ridge": [0.01, 0.0, 0.02], "fund": [0.0, 0.01, 0.0], "rev": [0.02, 0.01, 0.03],
                      "fly+fund": [0.01, 0.02, 0.0]})
    b = a.iloc[:2] * 2
    h = horizon_table({"league": a, "fwd63d": b})
    assert list(h.index) == ["fly", "ridge", "fund", "rev"]                  # blends are left out, components kept in order
    assert ("league", "IC") in h.columns and ("63 d", "t") in h.columns and h.loc["fly", ("league", "n")] == 3 and h.loc["fly", ("63 d", "n")] == 2
    assert abs(h.loc["fly", ("63 d", "IC")] - 0.05) < 1e-9


def test_label_frame_sector_neutral_removes_the_sector_mean_and_horizon_table_names_it():
    from experiments.blend_test import horizon_table, label_frame
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=40)
    tick = [f"T{i}" for i in range(12)]
    sectors = {t: ("A" if i < 4 else "B" if i < 8 else "C") for i, t in enumerate(tick)}
    sec_ret = {s: rng.normal(0, 0.01, len(dates)) for s in "ABC"}
    rows = {(d, t): sec_ret[sectors[t]][j] + rng.normal(0, 0.002) for j, d in enumerate(dates) for t in tick}
    panel = pd.DataFrame({"fwd_ret": pd.Series(rows)})
    panel.index.names = ["date", "ticker"]
    plain = label_frame(panel, 5)
    neutral = label_frame(panel, 5, sectors)
    d = plain.dropna().index[3]
    by_sec = neutral.loc[d].groupby(pd.Series(sectors)).mean()
    assert by_sec.abs().max() < 1e-12 and plain.loc[d].groupby(pd.Series(sectors)).mean().abs().max() > 1e-4
    t = pd.DataFrame({"fly": [0.01, 0.02, 0.0], "ridge": [0.0, 0.01, 0.0], "fund": [0.02, 0.0, 0.01], "rev": [0.0, 0.0, 0.01]})
    h = horizon_table({"league": t, "fwd63d": t, "sn63d": t})
    assert ("63 d sector-neutral", "IC") in h.columns and ("63 d", "IC") in h.columns
