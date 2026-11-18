import types

import numpy as np

from experiments.ablation import VARIANTS, run_variant, scramble_block, summarize


class FakeSyn:
    def __init__(self, W):
        self.W, self.lr = W, 0.0


class FakeAgent:
    """Stands in for the fly: records every training call so the harness logic can be checked without the connectome."""
    log = []

    def __init__(self):
        rng = np.random.default_rng(0)
        W = rng.uniform(0.1, 1.0, (6, 40)) * (rng.uniform(size=(6, 40)) < 0.4)
        self.syn, self.mask, self.last_score, self.last_pre = FakeSyn(W), (W > 0).astype(float), 0.0, np.ones(40)

    def reset_state(self): pass

    def act(self, x):
        self.last_score = float(x[0])

    def learn_counterfactual(self, pre, y):
        FakeAgent.log.append((self._cur, y))


def _data(n_days=30, n=20, seed=1):
    rng = np.random.default_rng(seed)
    day = []
    for _ in range(n_days):
        X = rng.normal(size=(n, 3))
        day.append((np.array([f"T{i}" for i in range(n)]), X, X[:, 0] * 0.5 + rng.normal(size=n)))
    return types.SimpleNamespace(day=day)


def test_scramble_keeps_counts_and_weights():
    ag = FakeAgent()
    W0 = ag.syn.W.copy()
    scramble_block(ag, "row", 3)
    assert (ag.syn.W > 0).sum(axis=1).tolist() == (W0 > 0).sum(axis=1).tolist()        # per-output-neuron in-degree kept
    assert np.allclose(np.sort(ag.syn.W.ravel()), np.sort(W0.ravel())) and not np.allclose(ag.syn.W, W0)
    assert (ag.mask == (ag.syn.W > 0)).all()
    scramble_block(ag, "all", 3)
    assert np.allclose(np.sort(ag.syn.W.ravel()), np.sort(W0.ravel()))
    assert (ag.syn.W > 0).sum(axis=1).tolist() != (W0 > 0).sum(axis=1).tolist()        # in-degrees no longer preserved


def test_variants_train_on_the_same_samples_and_only_the_label_differs():
    d = _data()
    cfg = dict(stocks_per_day=5, score_every=2)
    g = dict(wiring_seed=7, lr0=0.01, tau_anneal=100.0)
    logs, scores = {}, {}
    for v in ("real", "shuffled_reward", "untrained"):
        FakeAgent.log = []
        agent_cls = FakeAgent

        def build(gg, cc, n, agent_cls=agent_cls):
            a = agent_cls()
            orig = a.act
            a.act = lambda x, a=a, orig=orig: (orig(x), setattr(a, "_cur", float(x[0])))[0]
            return a

        def scores_fn(ag, X):
            return X[:, 0]

        idx, ics = run_variant(g, cfg, 3, d, 19, 20, 30, v, build, scores_fn)
        logs[v], scores[v] = list(FakeAgent.log), ics
    assert len(logs["untrained"]) == 0                                          # no learning at all
    assert len(logs["real"]) == len(logs["shuffled_reward"]) == 20 * 5          # same number of updates
    assert [x for x, _ in logs["real"]] == [x for x, _ in logs["shuffled_reward"]]   # same stocks shown, same order
    assert [y for _, y in logs["real"]] != [y for _, y in logs["shuffled_reward"]]   # but different labels
    assert np.allclose(sorted(y for _, y in logs["real"]), sorted(y for _, y in logs["real"]))
    assert np.allclose(scores["real"], scores["untrained"])                     # the fake scorer ignores weights: scoring path is shared


def test_summarize_paired_difference():
    rng = np.random.default_rng(0)
    base = rng.normal(0.04, 0.05, 40)
    res = {"real": {"a": base, "b": base + 0.001},
           "untrained": {"a": base - 0.03 + rng.normal(0, 0.005, 40), "b": base - 0.03},
           "shuffled_reward": {"a": base, "b": base}}
    s = summarize(res, n_eff=10)
    assert s.loc["untrained", "real_minus_variant"] > 0.02 and s.loc["untrained", "t_paired"] > 3
    assert abs(s.loc["shuffled_reward", "real_minus_variant"]) < 0.001
    assert list(s.index) == [v for v in VARIANTS if v in res]
