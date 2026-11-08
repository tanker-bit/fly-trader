"""Why does the fly fail the XOR test and get WORSE with more data?  (v2 diagnostic)

    python -m experiments.diagnose_fly                       # about 5-8 minutes
    python -m experiments.diagnose_fly --n-train 1500 --lrs 0.01 0.001   # quicker

Two questions, answered in order:

 A. REPRESENTATION. Is the XOR information present in the fly's KC code at all?
    A ridge model is fitted on the fly's own KC spike pattern (no dopamine learning). If this
    scores well, the encoder is fine and the dopamine learning rule is the problem.
    If it scores ~0, the encoder (random projection + winner-take-all) is the problem.

 B. LEARNING RULE. The dopamine rule is run online at several learning rates. For each, it
    prints test IC, IC on its own training samples (is it fitting at all?), and what happened to
    the plastic weights: how many are pinned at 0 or at w_max (saturated = learning has
    turned into noise), and how far they moved relative to their starting size.

Same data, same seed and same training loop as experiments.interaction_curve.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.interaction_curve import POS, make, quad, rank_ic, ridge_ic
from learning.fly_agent import FlyAgent
from learning.ridge import fit_ridge, predict


def new_fly(a, lr=None):
    fly = FlyAgent(neurons_csv=a.neurons, edges_csv=a.edges, n_obs=13, sim_ms=20, warm_ms=15,
                   input_scale=0.5, has_position=False, w_kc_mbon=a.w, seed=a.seed)
    if lr is not None:
        fly.syn.lr = lr
    return fly


def kc_code(fly, X):
    """Binary KC spike pattern per sample, from the full circuit simulation (no learning)."""
    out = np.empty((len(X), len(fly.kc)), dtype=np.float32)
    for i, x in enumerate(X):
        fly.reset_state()
        fly.act(x)
        out[i] = fly.last_pre > 0
    return out


def fly_ic(fly, X, y):
    s = np.empty(len(X))
    for i, x in enumerate(X):
        fly.reset_state()
        fly.act(x)
        s[i] = fly.last_score
    return rank_ic(s, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", type=float, default=0.25)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.01, 0.003, 0.001, 0.0003])
    ap.add_argument("--w", type=float, default=0.08)
    ap.add_argument("--neurons", default="data/mb_neurons.csv")
    ap.add_argument("--edges", default="data/mb_edges.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    Xtr, ytr, _ = make(a.n_train, "xor", a.ic, rng)
    Xte, yte, raw_te = make(a.n_test, "xor", a.ic, rng)
    n = a.n_train
    print(f"xor, planted IC={a.ic}, train={n}, test={a.n_test}  (IC noise about +/-{1 / np.sqrt(a.n_test):.3f})")
    print(f"oracle {rank_ic(raw_te, yte):.3f}   linear ridge {ridge_ic(lambda X: X, Xtr, ytr, Xte, yte):.3f}"
          f"   quadratic ridge {ridge_ic(quad, Xtr, ytr, Xte, yte):.3f}\n")

    # ---- A. is the XOR information in the KC code? ----
    t0 = time.time()
    fly = new_fly(a)
    Ctr, Cte = kc_code(fly, Xtr), kc_code(fly, Xte)
    code_ic = {}
    for lam in (0.01, 0.1, 1.0):
        coef, xm = fit_ridge(Ctr, ytr, lam_rel=lam)
        code_ic[lam] = rank_ic(predict(Cte, coef, xm), yte)
    print("A. REPRESENTATION: ridge fitted on the fly's own KC spike code (no dopamine learning)")
    print("   " + "   ".join(f"penalty {k}: IC {v:+.3f}" for k, v in code_ic.items()) + f"   ({time.time() - t0:.0f}s)")
    best = max(code_ic.values())
    print("   -> " + ("XOR information IS in the KC code: the dopamine learning rule is what fails."
                      if best > 0.08 else
                      "KC code carries little XOR information: the encoder is a problem, not just the learning rule.")
          + "\n")

    # ---- B. the learning rule at several learning rates ----
    print("B. LEARNING RULE (online dopamine learning, one pass over the training set)")
    print(f"{'lr':>8}{'test IC':>9}{'train IC':>10}{'at 0':>8}{'at w_max':>10}{'moved':>8}   time")
    for lr in a.lrs:
        t0 = time.time()
        fly = new_fly(a, lr)
        m = fly.mask > 0
        w0 = fly.syn.W[m].copy()
        for i in range(n):
            eps = max(0.02, 0.30 - 0.28 * i / n)
            fly.reset_state()
            act = fly.act(Xtr[i], epsilon=eps)
            fly.learn_sample(fly.last_pre.copy(), act, POS[act] * ytr[i] - 0.0005 * abs(POS[act]))
        w = fly.syn.W[m]
        at0 = float(np.mean(w <= 1e-9))
        atmax = float(np.mean(w >= fly.syn.w_max - 1e-9))
        moved = float(np.mean(np.abs(w - w0)) / np.mean(w0))
        tr_ic = fly_ic(fly, Xtr[-a.n_test:], ytr[-a.n_test:])
        te_ic = fly_ic(fly, Xte, yte)
        print(f"{lr:>8g}{te_ic:>9.3f}{tr_ic:>10.3f}{at0:>8.0%}{atmax:>10.0%}{moved:>7.1f}x   {time.time() - t0:.0f}s", flush=True)

    print("\nHow to read B: 'at 0' / 'at w_max' = share of plastic synapses pinned at a bound; 'moved' = mean")
    print("absolute weight change as a multiple of the starting mean weight.")
    print(" * high at-0 or at-w_max and moved >> 1 at large lr: updates are far too big, weights random-walk")
    print("   into the bounds (this would explain getting worse with more data).")
    print(" * a smaller lr gives a higher test IC: the rule can learn, it just needs a much smaller step.")
    print(" * train IC high but test IC ~0: overfitting/memorising. Both ~0 at every lr: the rule itself is the limit.")


if __name__ == "__main__":
    main()
