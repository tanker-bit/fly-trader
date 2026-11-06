"""Build a FlyAgent from a Genome, train it with dopamine, and score it. Used by the evolution loop."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from evolution.genome import Genome
from learning.fly_agent import FlyAgent
from learning.ridge import fit_ridge, predict

POS = (-1.0, 0.0, 1.0)  # actions 0 / 1 / 2 = sell / hold / buy

DEFAULT_CFG = dict(
    epochs=2, batch=8, cost=0.0005, sim_ms=20, warm_ms=15,
    neurons_csv="data/mb_neurons.csv", edges_csv="data/mb_edges.csv",
)


def build_agent(g: Genome, n_obs: int, cfg: dict) -> FlyAgent:
    agent = FlyAgent(
        neurons_csv=cfg["neurons_csv"], edges_csv=cfg["edges_csv"], n_obs=n_obs,
        sim_ms=cfg["sim_ms"], warm_ms=cfg["warm_ms"], input_scale=0.5, has_position=False,
        kc_sparsity=g.kc_sparsity, kc_inputs=g.kc_inputs, kc_drive=g.kc_drive,
        w_kc_mbon=g.w_kc_mbon, tau_plastic=g.tau_plastic, seed=g.wiring_seed,
    )
    agent.syn.lr = g.lr
    return agent


def train_agent(agent: FlyAgent, X: np.ndarray, y: np.ndarray, cfg: dict, seed: int = 0) -> None:
    """Online dopamine learning: act (with exploration), then learn from the reward of that action."""
    rng = np.random.default_rng(seed + 100)
    n, total, step = len(X), cfg["epochs"] * len(X), 0
    for _ in range(cfg["epochs"]):
        order = rng.permutation(n)
        for s in range(0, n, cfg["batch"]):
            eps = max(0.02, 0.30 - 0.28 * step / total)
            pending = []
            for i in order[s:s + cfg["batch"]]:
                agent.reset_state()
                a = agent.act(X[i], epsilon=eps)
                pending.append((agent.last_pre.copy(), a, y[i]))
            for pre, a, ret in pending:   # reward arrives once the next-day return is known
                agent.learn_sample(pre, a, POS[a] * ret - cfg["cost"] * abs(POS[a]))
            step += len(pending)


def score_agent(agent: FlyAgent, X: np.ndarray) -> np.ndarray:
    """Continuous readout (buy-group minus sell-group MBON rate) for each sample, synapses frozen."""
    out = np.empty(len(X))
    for i, x in enumerate(X):
        agent.reset_state()
        agent.act(x)
        out[i] = agent.last_score
    return out


def ic(score: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation between a score and the realised return (0 if the score is constant)."""
    if np.std(score) < 1e-12:
        return 0.0
    r = spearmanr(score, y)[0]
    return 0.0 if np.isnan(r) else float(r)


def evaluate_genome(g: Genome, splits: dict, cfg: dict) -> dict:
    """Train on the train block, score on every validation block. Fitness = mean validation IC."""
    Xtr, ytr, _ = splits["train"]
    agent = build_agent(g, Xtr.shape[1], cfg)
    train_agent(agent, Xtr, ytr, cfg, seed=g.wiring_seed)
    val_ic = [ic(score_agent(agent, Xv), yv) for Xv, yv, _ in splits["val"]]
    return {"fitness": float(np.mean(val_ic)), "val_ic": val_ic}


def null_bar(n_trials: int, n_val_total: int) -> float:
    """Roughly the best validation IC that pure luck produces after `n_trials` tries.

    For a random score the IC over n samples has std ~ 1/sqrt(n); the max of N such draws is about
    sqrt(2 ln N) standard deviations. A champion's validation IC must clear this to mean anything.
    """
    return float(np.sqrt(2.0 * np.log(max(n_trials, 2))) / np.sqrt(n_val_total))


def holdout_report(champion: Genome, default: Genome, splits: dict, cfg: dict, seed: int = 0) -> dict:
    """Score everything on the holdout block, which evolution never saw. Returns {name: IC}."""
    Xtr, ytr, _ = splits["train"]
    Xh, yh, fh = splits["holdout"]
    rng = np.random.default_rng(seed)
    out = {}

    coef, xm = fit_ridge(Xtr, ytr)
    out["ridge (same train)"] = ic(predict(Xh, coef, xm), yh)
    out["random"] = ic(rng.normal(size=len(yh)), yh)
    out["oracle (true pattern)"] = ic(fh, yh) if np.std(fh) > 0 else 0.0

    # The untrained control gets its OWN agent: scoring mutates an agent's internal state (action
    # baseline), so reusing it for training would make the trained fly differ from the one evolution picked.
    out["fly champion UNTRAINED"] = ic(score_agent(build_agent(champion, Xtr.shape[1], cfg), Xh), yh)

    for name, g in (("fly default genome", default), ("fly champion", champion)):
        agent = build_agent(g, Xtr.shape[1], cfg)
        train_agent(agent, Xtr, ytr, cfg, seed=g.wiring_seed)
        if name == "fly champion":   # replay check: must equal the fitness evolution recorded
            out["_replay_val_fitness"] = float(np.mean([ic(score_agent(agent, Xv), yv) for Xv, yv, _ in splits["val"]]))
        out[name] = ic(score_agent(agent, Xh), yh)
    return out
