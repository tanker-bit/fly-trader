"""FlyAgent: mushroom-body connectome wired into the LIF simulator.

Pipeline per trading step:
  market observation -> sparse random projection -> k-winners-take-all onto Kenyon cells
  -> LIF simulation for `sim_ms` ms -> MBON firing rates -> buy / hold / sell.

The connectome supplies all fixed wiring (KC, MBON, DAN and their mutual connections).
The KC -> MBON block is pulled out of the fixed matrix and held as plastic weights
(`self.syn`), which the dopamine-gated rule will train in the next step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from connectome.load import load_edges, to_weight_matrix
from learning.three_factor import DopamineGatedSynapses
from sim.lif import LIFNetwork


class FlyAgent:
    def __init__(
        self,
        neurons_csv: str = "data/mb_neurons.csv",
        edges_csv: str = "data/mb_edges.csv",
        n_obs: int = 21,          # TradingEnv: window(20) + position
        sim_ms: int = 50,         # LIF steps (1 ms each) per trading step
        kc_sparsity: float = 0.05,  # fraction of KCs active per input (APL-style sparse code)
        kc_inputs: int = 6,       # random input features per KC
        kc_drive: float = 1.5,    # input current to winning KCs
        w_kc_mbon: float = 0.03,  # mean initial KC->MBON weight (tuned on hemibrain: ~25 Hz MBON rate)
        tau_plastic: float = 5.0,  # ms decay of the KC->MBON synaptic current
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.sim_ms = sim_ms
        self.n_obs = n_obs
        self.kc_drive = kc_drive
        self.tau_plastic = tau_plastic

        neurons = pd.read_csv(neurons_csv)
        edges = load_edges(edges_csv)
        W, index = to_weight_matrix(edges)
        n = W.shape[0]

        ids = np.empty(n, dtype=np.int64)
        for body_id, i in index.items():
            ids[i] = body_id
        cls = neurons.drop_duplicates("bodyId").set_index("bodyId").cell_class.reindex(ids).values
        self.kc = np.where(cls == "KC")[0]
        self.mbon = np.where(cls == "MBON")[0]
        self.dan = np.where(cls == "DAN")[0]
        if len(self.kc) == 0 or len(self.mbon) == 0:
            raise ValueError("No KC / MBON neurons found in the edge table")

        # Pull the KC -> MBON block out of the fixed matrix; it becomes plastic.
        block = W[self.mbon][:, self.kc].toarray()
        coo = W.tocoo()
        is_block = np.isin(coo.row, self.mbon) & np.isin(coo.col, self.kc)
        W_fixed = sparse.csr_matrix(
            (coo.data[~is_block], (coo.row[~is_block], coo.col[~is_block])), shape=W.shape
        )

        nz = np.abs(block[block != 0])
        scale = w_kc_mbon / nz.mean() if nz.size else 1.0
        self.syn = DopamineGatedSynapses(len(self.kc), len(self.mbon), seed=seed)
        self.syn.W = np.clip(np.abs(block) * scale, 0.0, self.syn.w_max)
        self.mask = (self.syn.W > 0).astype(np.float64)  # only real connectome synapses may change
        self.syn.decay = 0.0  # eligibility lasts one trading step (reward arrives immediately)
        self.rng = np.random.default_rng(seed + 1)
        # Running reward baseline for the dopamine reward-prediction-error signal.
        self.r_mean, self.r_var, self.r_beta = 0.0, 1e-4, 0.01

        self.net = LIFNetwork(W_fixed)

        # Fixed sparse random projection: observation -> KC drive (like PN -> KC in the fly).
        proj = np.zeros((len(self.kc), n_obs))
        for k in range(len(self.kc)):
            dims = rng.choice(n_obs, size=min(kc_inputs, n_obs), replace=False)
            proj[k, dims] = rng.normal(size=len(dims))
        self.proj = proj
        self.n_winners = max(1, int(kc_sparsity * len(self.kc)))

        # Arbitrary fixed readout groups: sell / hold / buy -> actions 0 / 1 / 2.
        perm = rng.permutation(len(self.mbon))
        self.groups = np.array_split(perm, 3)

        self.reset()

    def reset(self):
        self.net.reset()
        self.i_plastic = np.zeros(len(self.mbon))
        self.last_pre = np.zeros(len(self.kc))
        self.last_post = np.zeros(len(self.mbon))
        self.last_rates = np.zeros(len(self.mbon))
        self.group_base = None  # running mean firing rate of each readout group

    # ---- input encoding ----
    def encode(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64).copy()
        x[:-1] = np.clip(x[:-1] / 0.01, -5, 5)  # daily log returns ~ 1% -> unit scale
        drive = self.proj @ x
        winners = np.argpartition(drive, -self.n_winners)[-self.n_winners:]
        i_kc = np.zeros(len(self.kc))
        i_kc[winners] = self.kc_drive * (1.0 + 0.5 * np.tanh(drive[winners]))
        return i_kc

    # ---- one trading step ----
    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """Run the circuit; with probability `epsilon` take a random action instead (exploration)."""
        i_kc = self.encode(obs)
        i_ext = np.zeros(self.net.n)
        i_ext[self.kc] = i_kc
        kc_counts = np.zeros(len(self.kc))
        mbon_counts = np.zeros(len(self.mbon))
        decay = np.exp(-1.0 / self.tau_plastic)
        for _ in range(self.sim_ms):
            kc_spk = self.net.spikes[self.kc].astype(np.float64)
            self.i_plastic = self.i_plastic * decay + self.syn.W @ kc_spk
            i_ext[self.mbon] = self.i_plastic
            spikes = self.net.step(i_ext)
            kc_counts += spikes[self.kc]
            mbon_counts += spikes[self.mbon]
        self.last_pre, self.last_post = kc_counts, mbon_counts
        self.last_rates = mbon_counts / (self.sim_ms / 1000.0)
        rates = np.array([self.last_rates[g].mean() for g in self.groups])
        # Compare each group to its own running baseline, so the action reflects how the
        # input changed the circuit, not which group happens to be wired stronger.
        if self.group_base is None:
            self.group_base = rates.copy()
        scores = rates - self.group_base
        self.group_base += 0.05 * (rates - self.group_base)
        action = 1 if np.allclose(scores, 0.0) else int(np.argmax(scores))  # no change -> hold
        if epsilon > 0 and self.rng.random() < epsilon:
            action = int(self.rng.integers(3))
        self.last_action = action
        return action

    # ---- dopamine-gated learning ----
    def learn(self, reward: float) -> float:
        """Three-factor update after the env has paid out `reward` for the last action.

        dopamine = (reward - running mean) / running std   (reward prediction error, clipped)
        eligibility = (KCs that spiked) x (MBONs of the action actually taken)
        Gains above expectation strengthen those KC->MBON synapses, losses weaken them.
        """
        da = float(np.clip((reward - self.r_mean) / np.sqrt(self.r_var + 1e-12), -3.0, 3.0))
        self.r_mean += self.r_beta * (reward - self.r_mean)
        self.r_var += self.r_beta * ((reward - self.r_mean) ** 2 - self.r_var)

        pre = (self.last_pre > 0).astype(np.float64)
        post = np.zeros(len(self.mbon))
        post[self.groups[self.last_action]] = 1.0  # efference copy of the chosen action
        self.syn.update_eligibility(pre, post)
        self.syn.apply_dopamine(da)
        self.syn.W *= self.mask
        return da
