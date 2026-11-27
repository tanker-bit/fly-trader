from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tournament.league import _z
from tournament.live import COMMITTEE, check_features, load_flies, pick_flies, score_rows


# ---------------------------------------------------------------- pick_flies on a fake league (no files needed)
class FakeStore:
    def __init__(self, recs, weights):
        self.recs, self.weights = recs, set(weights)

    def alive(self):
        return [r for r in self.recs if r["status"] == "alive"]

    def load_weights(self, aid):
        return np.zeros((2, 2)) if aid in self.weights else None


def rec(aid, ics, last_round, status="alive", silent=False):
    h = [dict(round=last_round - len(ics) + 1 + n, ic=ic) for n, ic in enumerate(ics)]
    if h and silent:
        h[-1]["silent"] = True
    return dict(id=aid, status=status, history=h, genome={})


def fake_league():
    recs = [rec("a1", [0.01, 0.02], 5), rec("a2", [0.05], 5), rec("a3", [0.09], 4),          # a3 was not scored in round 5
            rec("a4", [0.03], 5, silent=True), rec("a5", [], 5), rec("a6", [0.2], 5, status="culled"),
            rec("a7", [0.04], 5)]                                                              # a7: scored but its weights are gone
    return FakeStore(recs, ["a1", "a2", "a3", "a4", "a5", "a6"])


def test_kept_means_scored_in_the_latest_round_with_weights_and_not_silent():
    assert [r["id"] for r in pick_flies(fake_league(), "kept")] == ["a1", "a2"]


def test_best_is_the_highest_selection_score_among_the_kept():
    assert [r["id"] for r in pick_flies(fake_league(), "best")] == ["a2"]


def test_explicit_ids_must_be_alive_and_trained():
    assert [r["id"] for r in pick_flies(fake_league(), ["a3", "a1"])] == ["a1", "a3"]
    with pytest.raises(ValueError):
        pick_flies(fake_league(), ["a6"])                    # culled
    with pytest.raises(ValueError):
        pick_flies(fake_league(), ["a7"])                    # no weights
    with pytest.raises(ValueError):
        pick_flies(fake_league(), "everyone")
    with pytest.raises(ValueError):
        pick_flies(FakeStore([], []), "kept")


# ---------------------------------------------------------------- feature checks (no files needed)
FEATS = ["f1", "f2", "f3"]


def test_check_features_reorders_columns_and_refuses_bad_rows():
    X = pd.DataFrame({"f3": [0.5, -0.5], "extra": [9, 9], "f1": [0.1, 0.2], "f2": [-1.0, 1.0]}, index=["AAA", "NA"])
    A = check_features(X, FEATS)
    assert A.dtype == np.float32 and A.tolist() == np.array([[0.1, -1.0, 0.5], [0.2, 1.0, -0.5]], np.float32).tolist()
    with pytest.raises(ValueError, match="lack"):
        check_features(X.drop(columns="f2"), FEATS)
    with pytest.raises(ValueError, match="missing or infinite"):
        check_features(X.assign(f1=[0.1, np.nan]), FEATS)
    with pytest.raises(ValueError, match="outside"):
        check_features(X.assign(f1=[0.1, 3.0]), FEATS)       # raw values, not ranks
    with pytest.raises(ValueError, match="duplicate"):
        check_features(pd.concat([X, X.iloc[:1]]), FEATS)
    with pytest.raises(ValueError, match="at least 2"):
        check_features(X.iloc[:1], FEATS)


# ---------------------------------------------------------------- the real v8 league (skipped where its files are absent)
ROOT = Path("league_v8")
NEEDED = [ROOT / "config.json", Path("data/mb_neurons.csv"), Path("data/mb_edges.csv"),
          Path("data_fundamentals/panel_hybrid.parquet")]
real = pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="league_v8, connectome csvs or the hybrid panel not on this PC")
KEPT_R18 = ["a0180", "a0182", "a0184", "a0185", "a0187", "a0191", "a0192", "a0193", "a0195", "a0196"]   # league_v8/LEADERBOARD.md


@pytest.fixture(scope="module")
def v8():
    from tournament.data import Data
    from tournament.store import Store
    store = Store(ROOT)
    cfg = store.load_config()
    return store, cfg, Data.from_cfg(cfg)


@real
def test_v8_kept_flies_are_the_round_18_survivors(v8):
    store, _, _ = v8
    assert [r["id"] for r in pick_flies(store, "kept")] == KEPT_R18
    assert [r["id"] for r in pick_flies(store, "best")] == ["a0192"]


@real
def test_v8_live_scores_reproduce_the_scores_the_league_saved(v8):
    """Scoring a panel day through tournament.live gives exactly the round-18 scores saved by the league run."""
    store, cfg, data = v8
    flies = load_flies(store, cfg, pick_flies(store, ["a0182", "a0192"]))          # two slices: price+risk and risk+fund
    saved = {f.id: np.load(ROOT / "rounds" / "scores" / f"r18_{f.id}.npz") for f in flies}
    key = saved["a0192"].files[0]
    i = int(key[1:])
    tickers, X, _ = data.day[i]
    rows = pd.DataFrame(X, index=tickers, columns=data.feats)
    got = score_rows(flies, rows.sample(frac=1.0, random_state=0).iloc[:, ::-1], cfg["feats"]).reindex(tickers)   # order must not matter
    for f in flies:
        assert np.array_equal(got[f.id].to_numpy().astype(np.float32), saved[f.id][key])
    want = np.mean([_z(got[f.id].to_numpy()) for f in flies], axis=0)
    assert np.allclose(got[COMMITTEE].to_numpy(), want)
