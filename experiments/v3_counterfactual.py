"""v3: counterfactual dopamine learning (full-information update) on the XOR test.

    python -m experiments.v3_counterfactual                     # about 10-12 minutes
    python -m experiments.v3_counterfactual --checkpoints 1000 3000 --lrs 0.03

Why: the v2 diagnostic showed the KC code DOES contain the XOR information (ridge on it: IC ~0.19,
quadratic ridge 0.23) but the bandit-style dopamine rule extracts only ~0.07. The bandit rule
only learns from the one action the fly happened to take, so most samples teach it little.
In stock trading the next-day return does not depend on what we do (we are small), so the reward
of EVERY action is known after the fact: buy earns +y, sell earns -y. This update uses both:

    buy group  : weights of the KCs that fired change by  +lr * dopamine
    sell group : weights of the KCs that fired change by  -lr * dopamine
    dopamine   = (y - running mean) / running std, clipped to +-3

No exploration is needed. Nothing else in the circuit changes (same encoder, same readout).
Compared against the v2 bandit rule (same data, same seed) and ridge on the same KC code.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.diagnose_fly import fly_ic, kc_code
from experiments.interaction_curve import POS, make, quad, rank_ic, ridge_ic
from learning.fly_agent import FlyAgent
from learning.ridge import fit_ridge, predict


class CounterfactualFly(FlyAgent):
    def learn_counterfactual(self, pre_counts: np.ndarray, y: float) -> float:
        da = float(np.clip((y - self.r_mean) / np.sqrt(self.r_var + 1e-12), -3.0, 3.0))
        self.r_mean += self.r_beta * (y - self.r_mean)
        self.r_var += self.r_beta * ((y - self.r_mean) ** 2 - self.r_var)
        pre = (pre_counts > 0).astype(np.float64)
        step = self.syn.lr * da * pre
        W = self.syn.W
        W[self.groups[2]] += step      # buy group: strengthened when y is high
        W[self.groups[0]] -= step      # sell group: strengthened when y is low
        self.syn.W = np.clip(W, 0.0, self.syn.w_max) * self.mask
        return da


def new_fly(a, cls, lr):
    fly = cls(neurons_csv=a.neurons, edges_csv=a.edges, n_obs=13, sim_ms=20, warm_ms=15,
              input_scale=0.5, has_position=False, w_kc_mbon=a.w, seed=a.seed)
    fly.syn.lr = lr
    return fly


def weight_stats(fly, w0, m):
    w = fly.syn.W[m]
    return (float(np.mean(w <= 1e-9)), float(np.mean(w >= fly.syn.w_max - 1e-9)),
            float(np.mean(np.abs(w - w0)) / np.mean(w0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", type=float, default=0.25)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[1000, 3000, 10000])
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    ap.add_argument("--w", type=float, default=0.08)
    ap.add_argument("--neurons", default="data/mb_neurons.csv")
    ap.add_argument("--edges", default="data/mb_edges.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cps = sorted(a.checkpoints)
    n_max = cps[-1]
    rng = np.random.default_rng(a.seed)
    Xtr, ytr, _ = make(n_max, "xor", a.ic, rng)
    Xte, yte, raw_te = make(a.n_test, "xor", a.ic, rng)
    print(f"xor, planted IC={a.ic}, test={a.n_test}  (IC noise about +/-{1 / np.sqrt(a.n_test):.3f})")
    print(f"oracle {rank_ic(raw_te, yte):.3f}")

    fly0 = new_fly(a, FlyAgent, 0.01)
    Ctr = kc_code(fly0, Xtr[:cps[-1] if cps[-1] <= 3000 else 3000])
    Cte = kc_code(fly0, Xte)
    n_c = len(Ctr)
    coef, xm = fit_ridge(Ctr, ytr[:n_c], lam_rel=0.1)
    print(f"reference ceilings: ridge on this fly's KC code (n={n_c}) {rank_ic(predict(Cte, coef, xm), yte):.3f}   "
          f"quadratic ridge (n={n_max}) {ridge_ic(quad, Xtr, ytr, Xte, yte):.3f}\n")

    print(f"{'rule':>16}{'lr':>7}{'train n':>9}{'test IC':>9}{'train IC':>10}{'at 0':>7}{'at max':>8}{'moved':>7}   time")
    for lr in a.lrs:
        fly = new_fly(a, CounterfactualFly, lr)
        m = fly.mask > 0
        w0 = fly.syn.W[m].copy()
        t0, done = time.time(), 0
        for cp in cps:
            for i in range(done, cp):
                fly.reset_state()
                fly.act(Xtr[i])                     # no exploration needed
                fly.learn_counterfactual(fly.last_pre.copy(), ytr[i])
            done = cp
            te = fly_ic(fly, Xte, yte)
            tr = fly_ic(fly, Xtr[max(0, cp - a.n_test):cp], ytr[max(0, cp - a.n_test):cp])
            at0, atmax, moved = weight_stats(fly, w0, m)
            print(f"{'v3 counterfact.':>16}{lr:>7g}{cp:>9}{te:>9.3f}{tr:>10.3f}{at0:>7.0%}{atmax:>8.0%}{moved:>6.1f}x   {time.time() - t0:.0f}s", flush=True)

    print("\nBaseline to beat: v2 bandit rule at 3000 samples scored test IC 0.069 (lr 0.01), 0.061 (lr 0.003).")
    print("Ceiling from the same KC code: about 0.19.")
    print(" * test IC clearly above 0.07 and rising with train n: the full-information update fixes the rule.")
    print(" * still ~0.07: the bottleneck is the readout (one linear layer through noisy spiking), not the credit signal.")
    print(" * test IC falls at large train n or lr: weights saturate ('at 0' / 'at max' high); lower lr.")


if __name__ == "__main__":
    main()
