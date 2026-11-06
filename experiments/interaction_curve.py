"""How many samples does the fly need to learn an interaction (XOR-like) pattern?

    python -m experiments.interaction_curve                      # about 2 minutes
    python -m experiments.interaction_curve --pattern linear     # control: should match ridge
    python -m experiments.interaction_curve --ic 0.4             # stronger planted pattern

Plants a pattern in 13 random features in [-1, 1] and measures out-of-sample rank-IC as the training set
grows, for: the oracle (ceiling), linear ridge, QUADRATIC ridge (a model that CAN learn
interactions), and one default fly trained online with dopamine. The quadratic ridge answers the
question "is this pattern learnable at this sample size at all?". Without it, a fly that scores
zero could mean "the circuit cannot do it" or just "the test was too small for anyone".

  xor pattern:     signal = x0 * x1      (linear models see nothing)
  linear pattern:  signal = -x0
Test-set IC noise is about 1/sqrt(n_test), roughly +/-0.03 at the default 1500.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.stats import rankdata

from learning.fly_agent import FlyAgent
from learning.ridge import fit_ridge, predict

POS = (-1.0, 0.0, 1.0)  # actions 0 / 1 / 2 = sell / hold / buy
NOISE = 0.03
SIGNAL_STD = {"xor": 1.0 / 3.0, "linear": 1.0 / np.sqrt(3.0)}  # of the raw signal for U(-1,1) inputs


def rank_ic(s, y):
    if np.std(s) < 1e-12:
        return 0.0
    return float(np.corrcoef(rankdata(s), rankdata(y))[0, 1])


def make(n, pattern, ic, rng):
    X = rng.uniform(-1, 1, (n, 13))
    raw = X[:, 0] * X[:, 1] if pattern == "xor" else -X[:, 0]
    f = raw / SIGNAL_STD[pattern]
    y = ic / np.sqrt(1 - ic ** 2) * NOISE * f + rng.normal(0, NOISE, n)
    return X, y, raw


def quad(X):
    iu = np.triu_indices(X.shape[1])
    return np.hstack([X, X[:, iu[0]] * X[:, iu[1]]])


def ridge_ic(feat, Xtr, ytr, Xte, yte):
    coef, xm = fit_ridge(feat(Xtr), ytr)
    return rank_ic(predict(feat(Xte), coef, xm), yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", choices=["xor", "linear"], default="xor")
    ap.add_argument("--ic", type=float, default=0.25, help="planted pattern strength (correlation)")
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[300, 1000, 3000, 10000])
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--w", type=float, default=0.08, help="w_kc_mbon (0.08 was the calibrated value on your connectome)")
    ap.add_argument("--neurons", default="data/mb_neurons.csv")
    ap.add_argument("--edges", default="data/mb_edges.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    n_max = max(a.checkpoints)
    Xtr, ytr, _ = make(n_max, a.pattern, a.ic, rng)
    Xte, yte, raw_te = make(a.n_test, a.pattern, a.ic, rng)
    oracle = rank_ic(raw_te, yte)

    fly = FlyAgent(neurons_csv=a.neurons, edges_csv=a.edges, n_obs=13, sim_ms=20, warm_ms=15,
                   input_scale=0.5, has_position=False, w_kc_mbon=a.w, seed=a.seed)

    def fly_ic():
        s = np.empty(len(Xte))
        for i, x in enumerate(Xte):
            fly.reset_state()
            fly.act(x)
            s[i] = fly.last_score
        return rank_ic(s, yte)

    print(f"pattern={a.pattern}  planted IC={a.ic}  test samples={a.n_test}  (IC noise about +/-{1 / np.sqrt(a.n_test):.3f})")
    print(f"{'train n':>8}{'oracle':>9}{'linear':>9}{'quadratic':>11}{'fly':>8}   time")
    done, t0 = 0, time.time()
    for cp in sorted(a.checkpoints):
        for i in range(done, cp):  # online learning, one sample at a time
            eps = max(0.02, 0.30 - 0.28 * i / n_max)
            fly.reset_state()
            act = fly.act(Xtr[i], epsilon=eps)
            fly.learn_sample(fly.last_pre.copy(), act, POS[act] * ytr[i] - 0.0005 * abs(POS[act]))
        done = cp
        lin = ridge_ic(lambda X: X, Xtr[:cp], ytr[:cp], Xte, yte)
        qd = ridge_ic(quad, Xtr[:cp], ytr[:cp], Xte, yte)
        print(f"{cp:>8}{oracle:>9.3f}{lin:>9.3f}{qd:>11.3f}{fly_ic():>8.3f}   {time.time() - t0:.0f}s", flush=True)

    print("\nHow to read this:")
    print(" * quadratic well above 0 and the fly near 0: the pattern is learnable, but this circuit")
    print("   + learning rule is not extracting it. Evolving hyperparameters may or may not fix that.")
    print(" * quadratic also near 0: the test is too small or too weak for anyone; raise --ic or the samples.")
    print(" * fly rising toward quadratic as samples grow: it can learn it, just slowly (sample-hungry).")


if __name__ == "__main__":
    main()
