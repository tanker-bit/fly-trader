"""Run: python -m pytest tests/test_diverse_selection.py -q

Diverse league, part 2: niche-aware keeping, immigrants, structural mutation, and that v7 behaviour is unchanged when they are off."""
import tempfile

import numpy as np

from tournament.genome import random_genome
from tournament.groups import GROUPS, n_view, niche, usable_sets
from tournament.league import choose_keep, new_record, select_and_breed
from tournament.store import Store

FEATS = list(GROUPS["price"]) + list(GROUPS["risk"]) + list(GROUPS["fund"])


def make_store(slices, ics, feats=FEATS, weights=True):
    store = Store(tempfile.mkdtemp())
    store.save_state(dict(round=0, next_id=len(slices), trials=len(slices)))
    rng = np.random.default_rng(0)
    for i, (s, ic) in enumerate(zip(slices, ics)):
        g = random_genome(rng, feats=feats, diverse=True, groups=list(s))
        rec = new_record(f"a{i:04d}", g, born_round=0)
        rec["last_round_done"] = 0
        rec["history"] = [dict(round=0, ic=ic, sd=0.1, n=10)]
        rec["trained_idx"], rec["n_seen"] = 99, 800
        store.save_agent(rec)
        if weights:
            store.save_weights(rec["id"], np.ones((3, 4)))
    return store


def cfg(**kw):
    c = dict(seed=1, pop=10, keep=5, diverse=True, feats=FEATS, n_obs=len(FEATS))
    c.update(kw)
    return c


def test_niching_keeps_the_best_of_each_slice_instead_of_one_crowd():
    price, fund, pf = ("price",), ("fund",), ("price", "fund")
    slices = [price] * 6 + [fund, pf, ("price", "risk"), ("risk", "fund")]
    ics = [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.045, 0.044, 0.043, 0.042]
    store = make_store(slices, ics)
    elig = sorted(store.alive(), key=lambda a: -a["history"][-1]["ic"])
    plain, _ = choose_keep(elig, dict(keep=5))
    assert sum(niche(a["genome"]) == price for a in plain) == 5
    kept, culled = choose_keep(elig, cfg(niching=True, niche_cap=2))
    assert len(kept) == 5 and len(kept) + len(culled) == 10
    assert len({niche(a["genome"]) for a in kept}) >= 3
    assert max(sum(niche(a["genome"]) == n for a in kept) for n in {niche(a["genome"]) for a in kept}) <= 2


def test_niching_relaxes_the_cap_when_there_is_no_one_else():
    store = make_store([("price",)] * 10, [0.01 * i for i in range(10)])
    elig = sorted(store.alive(), key=lambda a: -a["history"][-1]["ic"])
    kept, culled = choose_keep(elig, cfg(niching=True, niche_cap=2))
    assert len(kept) == 5 and len(culled) == 5


def test_niching_never_protects_a_silent_or_below_median_fly():
    store = make_store([("price",)] * 4 + [("fund",)], [0.05, 0.04, 0.03, 0.02, -0.5])
    elig = sorted(store.alive(), key=lambda a: -a["history"][-1]["ic"])
    kept, _ = choose_keep(elig, cfg(niching=True, niche_cap=3, keep=2))
    assert all(niche(a["genome"]) == ("price",) for a in kept)


def test_immigrants_are_fresh_flies_in_underrepresented_slices():
    slices = [("price",)] * 6 + [("price", "fund")] * 4
    store = make_store(slices, [0.1 - 0.01 * i for i in range(10)])
    keep, culled, kids = select_and_breed(store, cfg(niching=True, niche_cap=3, immigrants=2), 0)
    imm = [k for k in kids if k.get("immigrant")]
    assert len(imm) == 2 and len(kids) == len(culled)
    for k in imm:
        assert k["parent"] is None and k["gen"] == 0 and not k["inherited_weights"]
        assert store.load_weights(k["id"]) is None
        assert tuple(k["genome"]["groups"]) in {tuple(s) for s in usable_sets(FEATS)}
        assert k["genome"]["kc_inputs"] <= n_view(k["genome"], FEATS)
    held = {niche(a["genome"]) for a in keep}
    assert any(niche(k["genome"]) not in held for k in imm), "an immigrant should fill a slice nobody kept"


def test_structural_mutation_changes_the_slice_by_one_group_and_drops_old_weights():
    store = make_store([("price", "risk")] * 10, [0.1 - 0.01 * i for i in range(10)])
    keep, culled, kids = select_and_breed(store, cfg(p_struct=1.0), 0)
    assert kids
    for k in kids:
        assert len(set(k["genome"]["groups"]) ^ {"price", "risk"}) == 1
        assert not k["inherited_weights"]
        assert k["genome"]["kc_inputs"] <= n_view(k["genome"], FEATS)
        assert k["parent"] is not None


def test_options_off_gives_the_v7_children():
    a = make_store([("price", "risk")] * 10, [0.1 - 0.01 * i for i in range(10)])
    b = make_store([("price", "risk")] * 10, [0.1 - 0.01 * i for i in range(10)])
    _, _, ka = select_and_breed(a, cfg(), 0)
    _, _, kb = select_and_breed(b, cfg(immigrants=0, p_struct=0.0, niching=False), 0)
    assert [k["genome"] for k in ka] == [k["genome"] for k in kb] and [k["parent"] for k in ka] == [k["parent"] for k in kb]
    assert all(tuple(k["genome"]["groups"]) == ("price", "risk") for k in ka)


def test_breeding_is_idempotent_with_all_options_on():
    store = make_store([("price",)] * 5 + [("fund",)] * 3 + [("price", "fund")] * 2, [0.1 - 0.01 * i for i in range(10)])
    c = cfg(niching=True, niche_cap=2, immigrants=2, p_struct=0.3)
    _, _, k1 = select_and_breed(store, c, 0)
    assert len(store.alive()) == 10
    _, culled2, k2 = select_and_breed(store, c, 0)
    assert not k2 and not culled2 and len(store.alive()) == 10
