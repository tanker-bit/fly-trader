"""Run: python -m tests.test_league

Checks the honesty rules of the league: the 21-day label uses only the next 21 returns, splits leave a
21-day gap, genomes stay in bounds, and selection is idempotent (re-running after a crash is safe).
"""
import tempfile

import numpy as np
import pandas as pd

from tournament.data import GAP, Data, monthly_labels
from tournament.genome import INIT_GENES, SCALARS, mutate, random_genome
from tournament.league import new_record, select_and_breed
from tournament.store import Store


def make_panel(n_days=400, n_stocks=15, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=n_days),
                                      [f"T{i}" for i in range(n_stocks)]], names=["date", "ticker"])
    df = pd.DataFrame(rng.uniform(-1, 1, (len(idx), 13)), index=idx,
                      columns=[f"f{i}" for i in range(11)] + ["ret_1", "mom_12_1"])
    df["fwd_ret"] = rng.normal(0, 0.01, len(idx))
    df.iloc[-n_stocks:, df.columns.get_loc("fwd_ret")] = np.nan      # last day has no next-day return
    df["fwd_excess"] = df.fwd_ret - df.groupby(level="date").fwd_ret.transform("mean")
    return df


def test_label_matches_manual_and_ignores_the_future():
    p = make_panel()
    y = monthly_labels(p).unstack("ticker")
    fr = p["fwd_ret"].unstack("ticker")
    for t in (0, 50, 200):
        manual = (1 + fr.iloc[t:t + 21]).prod() - 1
        manual = manual - manual.mean()
        assert np.allclose(y.iloc[t].to_numpy(), manual.to_numpy(), atol=1e-12), "label != product of next 21 days"
    assert y.iloc[-20:].isna().all().all(), "last 20 dates cannot have a 21-day label"
    p2 = p.copy()
    d = p2.index.get_level_values("date").unique()
    p2.loc[d[260]:, "fwd_ret"] = 0.5                                  # change everything after date 260
    y2 = monthly_labels(p2).unstack("ticker")
    assert np.allclose(y.iloc[:240].to_numpy(), y2.iloc[:240].to_numpy(), equal_nan=True), "label depends on far future"


def test_splits_leave_a_gap_and_never_touch_holdout():
    p = make_panel(700)
    path = tempfile.mkdtemp() + "/p.pkl"
    p.to_pickle(path)
    d = Data(path, train_days=150, val_days=60, holdout=80)
    r = 0
    while d.round_valid(r):
        c, v0, v1 = d.round_geometry(r)
        assert v0 - c >= GAP and v1 <= d.n_usable and c >= 149
        r += 1
    assert r >= 3, "expected several rounds"
    assert not d.round_valid(r)


def test_genome_bounds_and_frozen_genes():
    rng = np.random.default_rng(0)
    g = random_genome(rng)
    for _ in range(300):
        g2 = mutate(g, rng, rewire=False, freeze_init=True)
        for k, (lo, hi, _) in SCALARS.items():
            assert lo - 1e-12 <= g2[k] <= hi + 1e-12
        for k in INIT_GENES:
            assert g2[k] == g[k]
        assert (g2["wiring_seed"], g2["kc_inputs"]) == (g["wiring_seed"], g["kc_inputs"])
        g = g2
    g3 = mutate(g, rng, rewire=True, freeze_init=False)
    assert g3["wiring_seed"] != g["wiring_seed"]


def test_selection_is_idempotent_and_culls_the_worst():
    store = Store(tempfile.mkdtemp())
    cfg = dict(seed=1, pop=6, keep=3)
    store.save_state(dict(round=0, next_id=6, trials=6))
    rng = np.random.default_rng(0)
    for i in range(6):
        rec = new_record(f"a{i:04d}", random_genome(rng), born_round=0)
        rec["last_round_done"] = 0
        rec["history"] = [dict(round=0, ic=0.01 * i, sd=0.1, n=10)]
        rec["trained_idx"], rec["n_seen"] = 99, 800
        store.save_agent(rec)
        store.save_weights(rec["id"], np.ones((3, 4)))
    keep, culled, kids = select_and_breed(store, cfg, 0)
    assert [a["id"] for a in keep] == ["a0005", "a0004", "a0003"]
    assert len(culled) == 3 and len(kids) == 3 and len(store.alive()) == 6
    assert store.load_weights("a0000") is None, "culled agents should drop their weights"
    keep2, culled2, kids2 = select_and_breed(store, cfg, 0)           # simulate a re-run after a crash
    assert len(kids2) == 0 and len(culled2) == 0 and len(store.alive()) == 6
    assert all(k["born_round"] == 1 for k in kids)


if __name__ == "__main__":
    test_label_matches_manual_and_ignores_the_future()
    test_splits_leave_a_gap_and_never_touch_holdout()
    test_genome_bounds_and_frozen_genes()
    test_selection_is_idempotent_and_culls_the_worst()
    print("all league tests passed")
