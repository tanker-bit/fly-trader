import types

import numpy as np

import tournament.league as L
from tournament.genome import kc_inputs_bounds, mutate, random_genome
from tournament.groups import GROUPS, MIN_VISIBLE, n_view, usable_sets, view_cols, visible_columns

FEATS = list(GROUPS["price"]) + list(GROUPS["risk"]) + list(GROUPS["fund"]) + ["other_feature"]


def test_visible_columns_follow_the_groups_and_ungrouped_features_stay_visible():
    cols = visible_columns(("risk",), FEATS)
    names = [FEATS[i] for i in cols]
    assert set(names) == set(GROUPS["risk"]) | {"other_feature"}
    assert len(visible_columns(("price", "risk", "fund"), FEATS)) == len(FEATS)
    assert not set(GROUPS["fund"]) & {FEATS[i] for i in visible_columns(("price",), FEATS)}


def test_usable_sets_drop_slices_that_leave_too_few_features():
    all_sets = usable_sets(FEATS)
    assert ("risk",) not in all_sets and ("price", "fund") in all_sets            # 3 risk features + 1 other < MIN_VISIBLE
    no_fund = [f for f in FEATS if f not in GROUPS["fund"]]
    assert all("fund" not in s or len(visible_columns(s, no_fund)) >= MIN_VISIBLE for s in usable_sets(no_fund))
    assert ("fund",) not in usable_sets(no_fund)


def test_default_genome_is_the_v7_genome_and_sees_everything():
    g = random_genome(np.random.default_rng(0))
    assert "groups" not in g and view_cols(g, FEATS) is None and n_view(g, FEATS) == len(FEATS)
    assert random_genome(np.random.default_rng(0), feats=FEATS, diverse=False) == g          # same draws, nothing added


def test_diverse_genomes_get_a_slice_and_inputs_that_fit_it():
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(60):
        g = random_genome(rng, len(FEATS), feats=FEATS, diverse=True)
        nv = n_view(g, FEATS)
        lo, hi = kc_inputs_bounds(nv)
        assert nv >= MIN_VISIBLE and lo <= g["kc_inputs"] <= hi and g["kc_inputs"] <= nv
        seen.add(tuple(g["groups"]))
    assert len(seen) >= 4                                                                    # the draw really spreads over slices


def test_mutation_keeps_the_slice_and_clips_inputs_to_what_the_fly_sees():
    rng = np.random.default_rng(2)
    g = random_genome(rng, len(FEATS), feats=FEATS, diverse=True)
    nv = n_view(g, FEATS)
    for _ in range(40):
        c = mutate(g, rng, rewire=True, freeze_init=False, n_obs=nv)
        assert c["groups"] == g["groups"] and kc_inputs_bounds(nv)[0] <= c["kc_inputs"] <= kc_inputs_bounds(nv)[1]


class FakeAgent:
    seen_dims = []

    def __init__(self, n_obs):
        self.n_obs, self.r_mean, self.r_var = n_obs, 0.0, 1e-4
        self.syn = types.SimpleNamespace(lr=0.0, W=np.zeros((2, 2)))
        self.mask = 1.0
        self.last_score, self.last_pre = 0.0, np.zeros(1)

    def reset_state(self): pass

    def act(self, x):
        assert len(x) == self.n_obs                       # the fly is fed exactly the features it was built for
        FakeAgent.seen_dims.append(len(x))
        self.last_score = float(np.sum(x))

    def learn_counterfactual(self, pre, y): pass


def make_data(n_days=6, n=12):
    rng = np.random.default_rng(0)
    day = [(np.arange(n), rng.normal(size=(n, len(FEATS))).astype(np.float32), rng.normal(size=n)) for _ in range(n_days)]
    return types.SimpleNamespace(feats=FEATS, day=day)


def test_agent_is_built_and_trained_on_its_slice_only(monkeypatch):
    built = []
    monkeypatch.setattr(L, "build_agent", lambda g, cfg, n_obs: built.append(n_obs) or FakeAgent(n_obs))
    data = make_data()
    store = types.SimpleNamespace(load_weights=lambda aid: None)
    for groups in (None, ["price", "risk"]):
        g = dict(wiring_seed=1, lr0=0.01, tau_anneal=100.0)
        if groups:
            g["groups"] = groups
        rec = dict(id="a0", genome=g, n_seen=0, trained_idx=-1, r_mean=0.0, r_var=1e-4)
        FakeAgent.seen_dims.clear()
        ag = L.load_agent_obj(rec, store, {}, data)
        L.train_through(ag, rec, data, 3, dict(stocks_per_day=4))
        want = len(FEATS) if groups is None else len(visible_columns(groups, FEATS))
        assert built[-1] == want and set(FakeAgent.seen_dims) == {want}
