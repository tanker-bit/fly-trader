"""Fly vs ridge on the TSX panel: one train/test split on the most recent data.

    python -m experiments.fly_panel            # about 15 minutes
    python -m experiments.fly_panel --quick    # about 4 minutes, rougher
    python -m experiments.fly_panel --seeds 0 1 2

Each (stock, day) is an independent sample: the 13 ranked stock features go into the Kenyon
cells, the circuit is reset, and the MBON readout gives a buy/hold/sell action. Dopamine is the
reward prediction error of  position * next-day relative return - trading cost.  After training the
synapses are frozen, and the continuous score (buy-group minus sell-group firing rate) ranks the
stocks each day in the same cost-aware long-short backtest used for the ridge baseline.

The comparison rows printed at the end:
  ridge (same train)   plain linear model, fit on exactly the same training window
  reversal / momentum  the two simple signals found earlier (nothing fitted)
  random               pure noise: shows what costs alone do
  fly untrained        same wiring, no learning (control, first seed only)
  fly seed N           trained fly
The fly only adds value if it is clearly above 'fly untrained' AND competitive with ridge.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from env.features import feature_columns
from env.portfolio import report
from experiments.backtest import MARKET_COLS
from learning.fly_agent import FlyAgent
from learning.ridge import fit_ridge, predict

POS = (-1.0, 0.0, 1.0)   # actions 0 / 1 / 2 = sell / hold / buy
CAL_SCALES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3]
TARGET_HZ = 25.0


def make_agent(seed, scale, n_obs, cfg):
    return FlyAgent(neurons_csv=cfg["neurons_csv"], edges_csv=cfg["edges_csv"], n_obs=n_obs,
                    sim_ms=cfg["sim_ms"], warm_ms=cfg["warm_ms"], input_scale=0.5,
                    has_position=False, w_kc_mbon=scale, seed=seed)


def fly_scores(agent, X):
    out = np.empty(len(X))
    for i, x in enumerate(X):
        agent.reset_state()
        agent.act(x)
        out[i] = agent.last_score
    return out


def calibrate(samples, n_obs, cfg):
    """Pick the KC->MBON weight scale whose untrained MBON firing is closest to ~25 Hz."""
    print(f"{'scale':>7}{'mean Hz':>9}{'score std':>11}")
    best, best_gap = None, 1e9
    for sc in cfg["scales"]:
        ag = make_agent(0, sc, n_obs, cfg)
        hz, s = [], []
        for x in samples:
            ag.reset_state()
            ag.act(x)
            hz.append(ag.last_rates.mean())
            s.append(ag.last_score)
        m, sd = float(np.mean(hz)), float(np.std(s))
        print(f"{sc:>7}{m:>9.1f}{sd:>11.2f}")
        gap = abs(m - TARGET_HZ) + (1000 if sd < 1e-9 else 0)
        if gap < best_gap:
            best, best_gap = sc, gap
    print(f"-> using w_kc_mbon = {best}\n")
    return best


def train_fly(agent, days, day, cfg, seed):
    rng = np.random.default_rng(seed + 100)
    total = cfg["epochs"] * len(days)
    step, t0 = 0, time.time()
    for ep in range(cfg["epochs"]):
        for d in days:
            _, X, y = day[d]
            eps = max(0.02, 0.30 - 0.28 * step / total)
            idx = rng.choice(len(X), size=min(cfg["stocks_per_day"], len(X)), replace=False)
            batch = []
            for i in idx:
                agent.reset_state()
                a = agent.act(X[i], epsilon=eps)
                batch.append((agent.last_pre.copy(), a, y[i]))
            for pre, a, ex in batch:      # reward arrives after the day's return is known
                agent.learn_sample(pre, a, POS[a] * ex - cfg["cost"] * abs(POS[a]))
            step += 1
            if step % 250 == 0:
                print(f"    seed {seed} train {step}/{total} days  ({time.time() - t0:.0f}s)", flush=True)


def run(panel, seeds=(0,), train_days=1000, test_days=500, stocks_per_day=8, epochs=2,
        sim_ms=20, warm_ms=15, cost=0.0005, neurons_csv="data/mb_neurons.csv",
        edges_csv="data/mb_edges.csv", scales=CAL_SCALES, control=True):
    cfg = dict(stocks_per_day=stocks_per_day, epochs=epochs, sim_ms=sim_ms, warm_ms=warm_ms, cost=cost,
               neurons_csv=neurons_csv, edges_csv=edges_csv, scales=scales)
    feats = [c for c in feature_columns(panel) if c not in MARKET_COLS]
    lab = panel[panel["fwd_excess"].notna()]
    dates = lab.index.get_level_values("date").unique().sort_values()[-(train_days + test_days):]
    train_dates, test_dates = dates[:-test_days], dates[-test_days:]
    lab = lab[lab.index.get_level_values("date").isin(dates)]
    day = {d: (g.index.get_level_values("ticker").to_numpy(), g[feats].to_numpy(float),
               g["fwd_excess"].to_numpy(float)) for d, g in lab.groupby(level="date")}
    print(f"train {train_dates[0].date()} -> {train_dates[-1].date()} ({len(train_dates)} days)   "
          f"test {test_dates[0].date()} -> {test_dates[-1].date()} ({len(test_dates)} days)\n")

    def frame(fn):
        return pd.DataFrame({d: pd.Series(fn(day[d]), index=day[d][0]) for d in test_dates}).T

    # ---- baselines ----
    Xtr = np.vstack([day[d][1] for d in train_dates])
    ytr = np.concatenate([day[d][2] for d in train_dates])
    coef, xm = fit_ridge(Xtr, ytr)
    rng = np.random.default_rng(0)
    signals = {
        "ridge (same train)": frame(lambda t: predict(t[1], coef, xm)),
        "reversal (-ret_1)": frame(lambda t: -t[1][:, feats.index("ret_1")]),
        "momentum 12-1": frame(lambda t: t[1][:, feats.index("mom_12_1")]),
        "random": frame(lambda t: rng.normal(size=len(t[1]))),
    }

    # ---- fly ----
    pick = np.random.default_rng(1)
    cal = [day[d][1][pick.integers(len(day[d][1]))] for d in pick.choice(train_dates, 100)]
    scale = calibrate(np.array(cal), len(feats), cfg)
    for k, seed in enumerate(seeds):
        agent = make_agent(seed, scale, len(feats), cfg)
        if control and k == 0:
            t0 = time.time()
            signals["fly untrained"] = frame(lambda t: fly_scores(agent, t[1]))
            print(f"  untrained control scored ({time.time() - t0:.0f}s)", flush=True)
        train_fly(agent, train_dates, day, cfg, seed)
        t0 = time.time()
        signals[f"fly seed {seed}"] = frame(lambda t: fly_scores(agent, t[1]))
        print(f"  seed {seed}: test scored ({time.time() - t0:.0f}s)\n", flush=True)

    fwd = lab["fwd_ret"].unstack("ticker").reindex(test_dates)
    exc = lab["fwd_excess"].unstack("ticker").reindex(test_dates)
    print("=== Long-short, Sharpe after costs (S@c = c basis points per unit traded) ===")
    report(signals, fwd, exc, rebalances=(1, 5))
    return signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--train-days", type=int, default=1000)
    ap.add_argument("--test-days", type=int, default=500)
    ap.add_argument("--stocks-per-day", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--sim-ms", type=int, default=20)
    ap.add_argument("--warm-ms", type=int, default=15)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.train_days, a.test_days, a.epochs, a.stocks_per_day = 500, 200, 1, 6
    panel = pd.read_parquet("data/panel.parquet")
    run(panel, seeds=a.seeds, train_days=a.train_days, test_days=a.test_days,
        stocks_per_day=a.stocks_per_day, epochs=a.epochs, sim_ms=a.sim_ms, warm_ms=a.warm_ms)


if __name__ == "__main__":
    main()
