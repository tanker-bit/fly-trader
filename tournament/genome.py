"""A fly's settings (genome) as a plain dict, so it can be saved as JSON.

Scalar genes are mutated in a bounded way. Two genes define the WIRING (kc_inputs, wiring_seed): the trained
KC->MBON weights only make sense for the wiring they were trained with. So a child either
  * keeps the parent's wiring and INHERITS its trained weights (only non-initial genes mutate), or
  * gets new wiring and starts from scratch (all genes mutate).
"""
from __future__ import annotations

import math

import numpy as np

SCALARS = {  # name: (low, high, "lin" | "log")
    "kc_sparsity": (0.01, 0.10, "lin"),
    "kc_drive": (0.9, 3.0, "lin"),         # floor raised from 0.6: in the 15-round baseline every agent below ~0.82 was silent (IC exactly 0)
    "w_kc_mbon": (0.04, 0.25, "log"),      # sets the INITIAL weight scale: frozen when weights are inherited
    "tau_plastic": (2.0, 20.0, "log"),
    "lr0": (0.003, 0.1, "log"),
    "tau_anneal": (2000.0, 100000.0, "log"),
}
INIT_GENES = ("w_kc_mbon",)


def kc_inputs_bounds(n_obs: int = 13) -> tuple[int, int]:
    """Each KC samples `kc_inputs` of the `n_obs` features, so a given feature (or a pair of them, which XOR-like
    interactions need) is seen by a share of KCs that shrinks as n_obs grows. Let the upper bound grow with n_obs
    (13 inputs -> 3..12, 24 inputs -> 3..16)."""
    return 3, max(12, int(round(2 * n_obs / 3)))


def default_genome() -> dict:
    return dict(kc_sparsity=0.05, kc_drive=1.5, w_kc_mbon=0.08, tau_plastic=5.0, lr0=0.03,
                tau_anneal=10000.0, kc_inputs=6, wiring_seed=0)


def _fwd(v, kind):
    return math.log(v) if kind == "log" else v


def _back(t, kind):
    return math.exp(t) if kind == "log" else t


def random_genome(rng, n_obs: int = 13) -> dict:
    g = {k: float(min(hi, max(lo, _back(rng.uniform(_fwd(lo, kind), _fwd(hi, kind)), kind))))
         for k, (lo, hi, kind) in SCALARS.items()}
    lo_k, hi_k = kc_inputs_bounds(n_obs)
    g["kc_inputs"] = int(rng.integers(lo_k, hi_k + 1))
    g["wiring_seed"] = int(rng.integers(0, 2 ** 31 - 1))
    return g


def mutate(g: dict, rng, rewire: bool, freeze_init: bool, sigma: float = 0.1, p: float = 0.6, n_obs: int = 13) -> dict:
    """Gaussian step (sigma = fraction of each gene's range, in log space for log genes), clipped to bounds."""
    out = dict(g)
    names = [k for k in SCALARS if not (freeze_init and k in INIT_GENES)]
    chosen = [k for k in names if rng.random() < p] or [names[int(rng.integers(len(names)))]]
    for k in chosen:
        lo, hi, kind = SCALARS[k]
        span = _fwd(hi, kind) - _fwd(lo, kind)
        t = float(np.clip(_fwd(g[k], kind) + rng.normal(0, sigma * span), _fwd(lo, kind), _fwd(hi, kind)))
        out[k] = float(min(hi, max(lo, _back(t, kind))))
    if rewire:
        out["wiring_seed"] = int(rng.integers(0, 2 ** 31 - 1))
        out["kc_inputs"] = int(np.clip(g["kc_inputs"] + rng.integers(-2, 3), *kc_inputs_bounds(n_obs)))
    return out
