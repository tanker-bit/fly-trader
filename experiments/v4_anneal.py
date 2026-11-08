"""v4: annealed learning rate + several seeds, on the XOR test.

    python -m experiments.v4_anneal                       # about 15 minutes
    python -m experiments.v4_anneal --seeds 0 1           # quicker

v3 (counterfactual dopamine) reached test IC ~0.10 at 3000 samples, then FELL with more data while its
training IC stayed high (0.25-0.35): with a constant step, the weights keep chasing the noise of the
latest samples. v4 shrinks the step over time:   lr_t = lr0 / (1 + t / tau)   (tau = 0 means constant).
Each setting is run on several seeds (new data AND new random wiring per seed) and reported as
mean +/- standard deviation of the test IC, because one run has about +/-0.026 noise.
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np

from experiments.diagnose_fly import fly_ic
from experiments.interaction_curve import make, quad, rank_ic, ridge_ic
from experiments.v3_counterfactual import CounterfactualFly, new_fly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", type=float, default=0.25)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[1000, 3000, 6000])
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--configs", nargs="+", default=["0.03:0", "0.03:1000", "0.1:1000"],
                    help="lr0:tau pairs, tau 0 = constant lr")
    ap.add_argument("--w", type=float, default=0.08)
    ap.add_argument("--neurons", default="data/mb_neurons.csv")
    ap.add_argument("--edges", default="data/mb_edges.csv")
    a = ap.parse_args()

    cps = sorted(a.checkpoints)
    configs = [(float(c.split(":")[0]), float(c.split(":")[1])) for c in a.configs]
    res = {(c, cp): [] for c in configs for cp in cps}
    oracle, quadr = [], []
    t_start = time.time()
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        Xtr, ytr, _ = make(cps[-1], "xor", a.ic, rng)
        Xte, yte, raw_te = make(a.n_test, "xor", a.ic, rng)
        oracle.append(rank_ic(raw_te, yte))
        quadr.append(ridge_ic(quad, Xtr, ytr, Xte, yte))
        for (lr0, tau) in configs:
            sa = copy.copy(a)
            sa.seed = seed
            fly = new_fly(sa, CounterfactualFly, lr0)
            done = 0
            for cp in cps:
                for i in range(done, cp):
                    fly.syn.lr = lr0 / (1.0 + i / tau) if tau > 0 else lr0
                    fly.reset_state()
                    fly.act(Xtr[i])
                    fly.learn_counterfactual(fly.last_pre.copy(), ytr[i])
                done = cp
                res[((lr0, tau), cp)].append(fly_ic(fly, Xte, yte))
            print(f"seed {seed} lr0 {lr0:g} tau {tau:g}: " +
                  "  ".join(f"{cp}:{res[((lr0, tau), cp)][-1]:+.3f}" for cp in cps) +
                  f"   ({time.time() - t_start:.0f}s)", flush=True)

    print(f"\nXOR, planted IC {a.ic}, {len(a.seeds)} seeds, test {a.n_test} each. Test IC, mean +/- sd over seeds:")
    print(f"oracle {np.mean(oracle):.3f}   quadratic ridge (n={cps[-1]}) {np.mean(quadr):.3f}\n")
    print(f"{'lr0':>6}{'tau':>7}" + "".join(f"{('n=' + str(cp)):>16}" for cp in cps))
    for (lr0, tau) in configs:
        print(f"{lr0:>6g}{tau:>7g}" + "".join(
            f"{np.mean(res[((lr0, tau), cp)]):>+9.3f} +/-{np.std(res[((lr0, tau), cp)]):.3f}" for cp in cps))
    print("\nReference: v2 bandit ~0.07 at n=3000; v3 constant lr peaked ~0.10 at n=3000 then fell.")
    print(" * annealed rows that keep rising (or hold) at larger n while the constant row falls: annealing fixes it.")
    print(" * all rows within about 0.03 of each other: the differences are noise; the readout is the limit.")


if __name__ == "__main__":
    main()
