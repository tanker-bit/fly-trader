"""Dopamine-gated (three-factor) plasticity for KC -> MBON synapses.

dW = lr * dopamine * eligibility
eligibility decays over time and accumulates pre * post coincidences.
Positive dopamine (gain) potentiates recently active synapses; negative dopamine
(loss) depresses them. Weights are clipped to stay non-negative and bounded.
"""
from __future__ import annotations

import numpy as np


class DopamineGatedSynapses:
    def __init__(
        self,
        n_pre: int,
        n_post: int,
        lr: float = 0.01,
        tau_elig: float = 50.0,   # steps
        w_max: float = 1.0,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(0.0, 0.1, size=(n_post, n_pre))
        self.elig = np.zeros_like(self.W)
        self.lr = lr
        self.decay = np.exp(-1.0 / tau_elig)
        self.w_max = w_max

    def update_eligibility(self, pre: np.ndarray, post: np.ndarray):
        self.elig = self.elig * self.decay + np.outer(post.astype(float), pre.astype(float))

    def apply_dopamine(self, dopamine: float):
        self.W = np.clip(self.W + self.lr * dopamine * self.elig, 0.0, self.w_max)
