"""Control experiment: does the connectome add anything beyond a random sparse code with a fitted readout?

    python -m experiments.control_features --panel data_fundamentals/panel.parquet --league-root league_data_fundamentals

Same rounds, same train/validation split (21-day gap) and same validation dates as the league. For each round, every model
is fit ONLY on dates <= the round's training cutoff and scored on that round's validation window:

  ridge raw        linear ridge on the raw 24 features (the boring baseline)
  random KC code   fly-style encoder (sparse random projection + top-k winners, NO connectome, no spiking) + ridge readout.
                   This is the best the fly's Kenyon-cell code could do with an optimal linear readout.
  random dense     dense random tanh features + ridge readout (same width): a nonlinear model with no fly structure at all
  fly (league)     mean IC over all agents that were scored that round (read from --league-root, if given)

If the random models match or beat the fly rows, the connectome and the evolution add nothing. A control model that does
not beat plain ridge means the nonlinearity itself has nothing to find on these inputs. The final holdout is never touched.
"""
from __future__ import annotations

import argparse

import numpy as np

from tournament.data import GAP, Data, rank_ic


def kc_code(X, proj, k, scale=0.5):
    x = np.clip(X.astype(np.float32) / scale, -5, 5)
    drive = x @ proj.T
    thr = np.partition(drive, -k, axis=1)[:, -k][:, None]
    return np.where(drive >= thr, 1.0 + 0.5 * np.tanh(drive), 0.0).astype(np.float32)


def dense_code(X, proj, scale=0.5):
    return np.tanh(np.clip(X.astype(np.float32) / scale, -5, 5) @ proj.T / np.sqrt(proj.shape[1])).astype(np.float32)


def sparse_proj(rng, n_units, n_obs, n_in):
    P = np.zeros((n_units, n_obs), np.float32)
    for u in range(n_units):
        dims = rng.choice(n_obs, size=min(n_in, n_obs), replace=False)
        P[u, dims] = rng.normal(size=len(dims))
    return P


def fit(F, y, lam_rel=0.01):
    fm, ym = F.mean(0), y.mean()
    Fc = F - fm
    G = (Fc.T @ Fc).astype(np.float64)
    coef = np.linalg.solve(G + lam_rel * len(F) * np.eye(G.shape[0]), (Fc.T @ (y - ym)).astype(np.float64))
    return coef.astype(np.float32), fm


def score(F, model):
    coef, fm = model
    return (F - fm) @ coef


def stats(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    return v.mean(), se, (v.mean() / se if se and se > 0 else np.nan)


def fly_round_means(root):
    from tournament.store import Store
    s = Store(root)
    by = {}
    for aid in s.agent_ids():
        for h in s.load_agent(aid)["history"]:
            by.setdefault(h["round"], []).append(h["ic"])
    return {r: float(np.mean(v)) for r, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data_fundamentals/panel.parquet")
    ap.add_argument("--league-root", default=None, help="read the fly agents' per-round mean IC from here")
    ap.add_argument("--train-days", type=int, default=1000)
    ap.add_argument("--val-days", type=int, default=252)
    ap.add_argument("--holdout", type=int, default=500)
    ap.add_argument("--score-every", type=int, default=5)
    ap.add_argument("--units", type=int, default=1000)
    ap.add_argument("--kc-inputs", type=int, default=8)
    ap.add_argument("--sparsity", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=100000)
    ap.add_argument("--train-every", type=int, default=3, help="use every k-th training date")
    a = ap.parse_args()

    d = Data(a.panel, a.train_days, a.val_days, a.holdout)
    n_obs = len(d.feats)
    k = max(1, int(a.sparsity * a.units))
    fly = fly_round_means(a.league_root) if a.league_root else {}
    names = ["ridge raw", "random KC code", "random dense"]
    rows, r = [], 0
    while d.round_valid(r):
        c, v0, v1 = d.round_geometry(r)
        tr = range(0, c + 1, a.train_every)
        X = np.vstack([d.day[i][1] for i in tr]).astype(np.float32)
        y = np.concatenate([d.day[i][2] for i in tr]).astype(np.float32)
        ok = np.isfinite(y)
        X, y = X[ok], y[ok]
        if len(X) > a.max_rows:
            sel = np.random.default_rng(r).choice(len(X), a.max_rows, replace=False)
            X, y = X[sel], y[sel]
        val = list(range(v0, v1, a.score_every))
        out = {}
        m = fit(X, y)
        out["ridge raw"] = np.mean([rank_ic(score(d.day[i][1].astype(np.float32), m), d.day[i][2]) for i in val])
        for name in names[1:]:
            ics = []
            for s in range(a.seeds):
                rng = np.random.default_rng([r, s, 7])
                if name == "random KC code":
                    P = sparse_proj(rng, a.units, n_obs, a.kc_inputs)
                    enc = lambda Z: kc_code(Z, P, k)
                else:
                    P = rng.normal(size=(a.units, n_obs)).astype(np.float32)
                    enc = lambda Z: dense_code(Z, P)
                mm = fit(enc(X), y)
                ics.append(np.mean([rank_ic(score(enc(d.day[i][1]), mm), d.day[i][2]) for i in val]))
            out[name] = float(np.mean(ics))
        out["fly (league)"] = fly.get(r, np.nan)
        rows.append(out)
        print(f"round {r:>2}  validate {d.dates[v0].date()} -> {d.dates[v1 - 1].date()}   " +
              "  ".join(f"{k_}: {v:+.3f}" for k_, v in out.items()), flush=True)
        r += 1

    cols = names + ["fly (league)"]
    print("\nMean IC over rounds (se across rounds):")
    for c_ in cols:
        m, se, t = stats([x[c_] for x in rows])
        print(f"  {c_:<16} {m:+.4f}  se {se:.4f}  t {t:+.2f}")
    print("\nPaired differences across rounds (what matters):")
    for a_, b_ in [("random KC code", "ridge raw"), ("random dense", "ridge raw"), ("fly (league)", "random KC code"),
                   ("fly (league)", "ridge raw")]:
        m, se, t = stats([x[a_] - x[b_] for x in rows])
        print(f"  {a_} - {b_:<16} {m:+.4f}  se {se:.4f}  t {t:+.2f}")
    print("\nReading: |t| < 2 is no evidence. 'fly - random KC code' near zero means the connectome adds nothing over a random")
    print("sparse code. 'random ... - ridge raw' near zero means nonlinear features find nothing that a linear model misses.")


if __name__ == "__main__":
    main()
