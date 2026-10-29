"""Find a KC->MBON weight scale where the MBONs respond to input without saturating.

    python -m experiments.tune_drive

For each candidate scale, drives the circuit with random market-like inputs and prints:
  mean MBON rate    -> want roughly 5-40 Hz (not 0, not ~100+)
  rate variability  -> want clearly above 0 (circuit responds to the input)
  action mix        -> want all three actions used, not 100% one action
"""
from __future__ import annotations

import time

import numpy as np

from learning.fly_agent import FlyAgent

SCALES = [0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02]
STEPS = 150


def main():
    rng = np.random.default_rng(0)
    obs = rng.normal(0.0, 0.01, size=(STEPS, 21))
    obs[:, -1] = rng.choice([-1.0, 0.0, 1.0], size=STEPS)  # position slot
    print(f"{'scale':>7} {'mean Hz':>8} {'std Hz':>7} {'sell/hold/buy':>16}  time")
    for w in SCALES:
        t0 = time.time()
        agent = FlyAgent(w_kc_mbon=w)
        agent.reset()
        means, acts = [], []
        for x in obs:
            acts.append(agent.act(x))
            means.append(agent.last_rates.mean())
        c = np.bincount(acts, minlength=3)
        print(f"{w:>7} {np.mean(means):>8.1f} {np.std(means):>7.2f} {c[0]:>5}/{c[1]}/{c[2]:<5}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
