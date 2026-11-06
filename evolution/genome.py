"""A fly 'genome': the heritable knobs of a FlyAgent.

The connectome itself stays fixed. What evolves is everything around it: how sensory input is
wired into the Kenyon cells, how sparse the code is, how strong the KC->MBON synapses start,
how fast dopamine learning is, and which MBONs form the sell/hold/buy readout groups.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

# name: (low, high, log_scale). Floats mutate multiplicatively when log_scale is True.
BOUNDS = {
    "kc_sparsity": (0.01, 0.20, True),
    "kc_drive": (0.8, 4.0, True),
    "w_kc_mbon": (0.03, 0.30, True),   # below ~0.03 the MBONs stay silent, above ~0.3 they saturate
    "tau_plastic": (2.0, 20.0, True),
    "lr": (0.001, 0.1, True),
}
KC_INPUTS = (2, 13)   # random input features per Kenyon cell (integer)


@dataclass(frozen=True)
class Genome:
    kc_sparsity: float = 0.05
    kc_drive: float = 1.5
    w_kc_mbon: float = 0.08
    tau_plastic: float = 5.0
    lr: float = 0.01
    kc_inputs: int = 6
    wiring_seed: int = 0   # seeds the input projection, readout groups and initial weights

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Genome":
        return Genome(**d)


def random_genome(rng: np.random.Generator) -> Genome:
    vals = {}
    for k, (lo, hi, log) in BOUNDS.items():
        vals[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if log else float(rng.uniform(lo, hi))
    return Genome(**vals, kc_inputs=int(rng.integers(KC_INPUTS[0], KC_INPUTS[1] + 1)),
                  wiring_seed=int(rng.integers(0, 2**31 - 1)))


def mutate(g: Genome, rng: np.random.Generator, p: float = 0.4, sigma: float = 0.3,
           p_rewire: float = 0.15) -> Genome:
    """Each float gene mutates with probability p; the wiring seed is resampled with p_rewire."""
    d = g.to_dict()
    for k, (lo, hi, log) in BOUNDS.items():
        if rng.random() < p:
            d[k] = float(np.clip(d[k] * np.exp(rng.normal(0, sigma)), lo, hi))
    if rng.random() < p:
        d["kc_inputs"] = int(np.clip(d["kc_inputs"] + rng.choice([-2, -1, 1, 2]), *KC_INPUTS))
    if rng.random() < p_rewire:
        d["wiring_seed"] = int(rng.integers(0, 2**31 - 1))
    return Genome(**d)


def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    """Uniform crossover: each gene comes from one parent."""
    da, db = a.to_dict(), b.to_dict()
    return Genome(**{k: (da[k] if rng.random() < 0.5 else db[k]) for k in da})


def clone_with(g: Genome, **kw) -> Genome:
    return replace(g, **kw)
