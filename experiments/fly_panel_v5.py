"""v5: the v4 fly (counterfactual dopamine, annealed learning rate) on the real TSX panel.

    python -m experiments.fly_panel_v5                 # about 8-10 minutes per seed
    python -m experiments.fly_panel_v5 --quick         # about 3 minutes, rough
    python -m experiments.fly_panel_v5 --seeds 0 1 2

Same data split, same baselines and same cost-aware backtest as experiments.fly_panel, so the rows are
directly comparable with the earlier v1 results (fly IC 0.018-0.024, untrained 0.003-0.005, ridge 0.007,
reversal 0.021, momentum 0.037). Only the learning rule changes:

  v1 (fly_panel):  bandit rule - learns only from the action it took, epsilon-greedy exploration.
  v5 (this file):  every stock-day updates BOTH groups (buy gets +dopamine, sell gets -dopamine, where
                   dopamine = the stock's next-day excess return, normalised), with lr_t = lr0 / (1 + t / tau).
                   No exploration. Valid because our trades do not move the next-day return.

lr0 = 0.03 and tau = 1000 were tuned on the synthetic XOR test (experiments.v4_anneal), NOT on this data.
Real returns are far weaker (IC ~0.02 vs 0.25 planted), so these settings may be off; do not tune them on the
test period here. The IC and its t-stat are the numbers to trust; Sharpe over 500 days is mostly noise.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from env.features import feature_columns
from env.portfolio import report
from experiments.backtest import MARKET_COLS
from experiments.fly_panel import CAL_SCALES, calibrate, fly_scores
from experiments.v3_counterfactual import CounterfactualFly
from learning.ridge import fit_ridge, predict


def make_agent(seed, scale, n_obs, a):
    return CounterfactualFly(neurons_csv=a.neurons, edges_csv=a.edges, n_obs=n_obs, sim_ms=a.sim_ms,
                             warm_ms=a.warm_ms, input_scale=0.5, has_position=False,
                             w_kc_mbon=scale, seed=seed)


def train(agent, days, day, a, seed):
    rng = np.random.default_rng(seed + 100)
    t0, step = time.time(), 0
    for n_day, d in enumerate(days):
        _, X, y = day[d]
        idx = rng.choice(len(X), size=min(a.stocks_per_day, len(X)), replace=False)
        for i in idx:
            agent.syn.lr = a.lr0 / (1.0 + step / a.tau) if a.tau > 0 else a.lr0
            agent.reset_state()
            agent.act(X[i])
            agent.learn_counterfactual(agent.last_pre.copy(), y[i])
            step += 1
        if (n_day + 1) % 250 == 0:
            print(f"    seed {seed} trained {n_day + 1}/{len(days)} days  ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--train-days", type=int, default=1000)
    ap.add_argument("--test-days", type=int, default=500)
    ap.add_argument("--stocks-per-day", type=int, default=8)
    ap.add_argument("--lr0", type=float, default=0.03)
    ap.add_argument("--tau", type=float, default=1000.0)
    ap.add_argument("--sim-ms", type=int, default=20)
    ap.add_argument("--warm-ms", type=int, default=15)
    ap.add_argument("--neurons", default="data/mb_neurons.csv")
    ap.add_argument("--edges", default="data/mb_edges.csv")
    ap.add_argument("--panel", default="data/panel.parquet")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.train_days, a.test_days, a.stocks_per_day = 500, 200, 6

    panel = pd.read_parquet(a.panel)
    feats = [c for c in feature_columns(panel) if c not in MARKET_COLS]
    lab = panel[panel["fwd_excess"].notna()]
    dates = lab.index.get_level_values("date").unique().sort_values()[-(a.train_days + a.test_days):]
    train_dates, test_dates = dates[:-a.test_days], dates[-a.test_days:]
    lab = lab[lab.index.get_level_values("date").isin(dates)]
    day = {d: (g.index.get_level_values("ticker").to_numpy(), g[feats].to_numpy(float),
               g["fwd_excess"].to_numpy(float)) for d, g in lab.groupby(level="date")}
    print(f"train {train_dates[0].date()} -> {train_dates[-1].date()} ({len(train_dates)} days)   "
          f"test {test_dates[0].date()} -> {test_dates[-1].date()} ({len(test_dates)} days)\n")

    def frame(fn):
        return pd.DataFrame({d: pd.Series(fn(day[d]), index=day[d][0]) for d in test_dates}).T

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

    pick = np.random.default_rng(1)
    cal = [day[d][1][pick.integers(len(day[d][1]))] for d in pick.choice(train_dates, 100)]
    cfg = dict(sim_ms=a.sim_ms, warm_ms=a.warm_ms, neurons_csv=a.neurons, edges_csv=a.edges, scales=CAL_SCALES)
    scale = calibrate(np.array(cal), len(feats), cfg)
    for k, seed in enumerate(a.seeds):
        agent = make_agent(seed, scale, len(feats), a)
        if k == 0:
            t0 = time.time()
            signals["fly untrained"] = frame(lambda t: fly_scores(agent, t[1]))
            print(f"  untrained control scored ({time.time() - t0:.0f}s)", flush=True)
        train(agent, train_dates, day, a, seed)
        t0 = time.time()
        signals[f"fly v5 seed {seed}"] = frame(lambda t: fly_scores(agent, t[1]))
        print(f"  seed {seed}: test scored ({time.time() - t0:.0f}s)\n", flush=True)

    fwd = lab["fwd_ret"].unstack("ticker").reindex(test_dates)
    exc = lab["fwd_excess"].unstack("ticker").reindex(test_dates)
    print("=== Long-short, Sharpe after costs (S@c = c basis points per unit traded) ===")
    report(signals, fwd, exc, rebalances=(1, 5, 21))
    print("\nWhat to look at: the IC and t columns. Noise on the mean daily IC over 500 days is about +/-0.006")
    print("(t about 2 needs IC ~0.012). The fly matters only if it is clearly above 'fly untrained'")
    print("and 'ridge', and consistent across seeds. v1 (bandit) fly IC was 0.018-0.024.")


if __name__ == "__main__":
    main()
