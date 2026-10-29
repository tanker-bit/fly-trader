"""Leaky integrate-and-fire simulator over a sparse signed weight matrix."""
from __future__ import annotations

import numpy as np
from scipy import sparse


class LIFNetwork:
    def __init__(
        self,
        W: sparse.spmatrix,
        dt: float = 1.0,          # ms
        tau_m: float = 20.0,      # membrane time constant, ms
        tau_syn: float = 5.0,     # synaptic current decay, ms
        v_rest: float = 0.0,
        v_thresh: float = 1.0,
        v_reset: float = 0.0,
        refractory_ms: float = 2.0,
    ):
        self.W = W.tocsr()
        self.n = W.shape[0]
        self.dt = dt
        self.tau_m = tau_m
        self.tau_syn = tau_syn
        self.v_rest, self.v_thresh, self.v_reset = v_rest, v_thresh, v_reset
        self.ref_steps = int(round(refractory_ms / dt))
        self.reset()

    def reset(self):
        self.v = np.full(self.n, self.v_rest, dtype=np.float64)
        self.i_syn = np.zeros(self.n)
        self.ref = np.zeros(self.n, dtype=np.int32)
        self.spikes = np.zeros(self.n, dtype=bool)

    def step(self, i_ext: np.ndarray | None = None) -> np.ndarray:
        """Advance one dt. i_ext is an external input current per neuron."""
        self.i_syn += -self.i_syn * (self.dt / self.tau_syn) + self.W @ self.spikes.astype(np.float64)
        drive = self.i_syn if i_ext is None else self.i_syn + i_ext
        active = self.ref <= 0
        dv = (-(self.v - self.v_rest) + drive) * (self.dt / self.tau_m)
        self.v = np.where(active, self.v + dv, self.v_reset)
        self.spikes = (self.v >= self.v_thresh) & active
        self.v[self.spikes] = self.v_reset
        self.ref = np.where(self.spikes, self.ref_steps, np.maximum(self.ref - 1, 0))
        return self.spikes

    def run(self, i_ext: np.ndarray, steps: int):
        """Run for `steps` with constant input; return (rates in Hz, spike matrix)."""
        counts = np.zeros(self.n)
        record = np.zeros((steps, self.n), dtype=bool)
        for t in range(steps):
            s = self.step(i_ext)
            counts += s
            record[t] = s
        rates = counts / (steps * self.dt / 1000.0)
        return rates, record
