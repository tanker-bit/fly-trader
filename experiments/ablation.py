"""Which part of the fly does the work? Paired ablations on the league's own genomes (T5a).

    python -m experiments.ablation --root league_v7 --workers 8                 # last finished round, top 4 flies
    python -m experiments.ablation --root league_v7 --round 5 --top 6 --workers 8

For each of the top flies (by the league's own selection score) the SAME genome, wiring seed and training samples are run as:
  real             trained from scratch through the round's training date with the true label (the control)
  untrained        no learning at all (what the wiring + random projection give for free)
  shuffled_reward  trained, but the label is randomly reassigned among that day's stocks (dopamine carries no information)
  scrambled_row    trained, KC->MBON wiring randomly permuted within each output neuron (same synapse count, in-degree, weights)
  scrambled_all    trained, KC->MBON wiring permuted over the whole block (same synapse count and weights, no structure at all)
then scored on the round's validation dates. Only one thing differs between a variant and `real`, so the paired difference
says what that thing contributes. Error bars use the league's own rule: independent samples ~ n_dates * score_every / gap.
Nothing is saved in the league; results go to <root>/ablation_rXX.csv.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from tournament.data import rank_ic

VARIANTS = ["real", "untrained", "shuffled_reward", "scrambled_row", "scrambled_all"]


def scramble_block(ag, mode: str, seed: int):
    """Randomly rewire the plastic KC->MBON block (rows = MBON, columns = KC). 'row': permute within each row; 'all': over the block."""
    rng = np.random.default_rng([int(seed), 777])
    W = ag.syn.W.copy()
    if mode == "row":
        for r in range(W.shape[0]):
            W[r] = W[r][rng.permutation(W.shape[1])]
    elif mode == "all":
        W = W.ravel()[rng.permutation(W.size)].reshape(W.shape)
    else:
        raise ValueError(mode)
    ag.syn.W = W
    ag.mask = (W > 0).astype(np.float64)


def run_variant(genome, cfg, n_obs, data, upto, v0, v1, variant, build, fly_scores):
    """Train (if the variant learns) on dates 0..upto, score validation dates. -> (date indices, IC per date)."""
    ag = build(genome, cfg, n_obs)
    if variant.startswith("scrambled"):
        scramble_block(ag, variant.split("_")[1], genome["wiring_seed"])
    if variant != "untrained":
        rng = np.random.default_rng([int(genome["wiring_seed"]), 0])           # same sample draws in every variant
        shuf = np.random.default_rng([int(genome["wiring_seed"]), 12345])
        n_seen = 0
        for i in range(0, upto + 1):
            _, X, y = data.day[i]
            if variant == "shuffled_reward":
                y = y[shuf.permutation(len(y))]
            for j in rng.choice(len(X), size=min(cfg["stocks_per_day"], len(X)), replace=False):
                ag.syn.lr = genome["lr0"] / (1.0 + n_seen / genome["tau_anneal"])
                ag.reset_state()
                ag.act(X[j])
                ag.learn_counterfactual(ag.last_pre.copy(), y[j])
                n_seen += 1
    idx = list(range(v0, v1, cfg["score_every"]))
    return idx, np.array([rank_ic(fly_scores(ag, data.day[i][1]), data.day[i][2]) for i in idx])


def summarize(res: dict, n_eff: float) -> pd.DataFrame:
    """res[variant][fly id] = IC per validation date (same dates for all). Paired comparison against `real`."""
    base = np.mean([res["real"][a] for a in res["real"]], axis=0)
    rows = []
    for v in [v for v in VARIANTS if v in res]:
        per_fly = [float(np.mean(x)) for x in res[v].values()]
        s = np.mean([res[v][a] for a in res[v]], axis=0)
        d = base - s
        sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
        rows.append(dict(variant=v, ic=float(np.mean(s)), ic_min_fly=min(per_fly), ic_max_fly=max(per_fly),
                         real_minus_variant=float(d.mean()),
                         t_paired=float(d.mean() / (sd / np.sqrt(max(n_eff, 1.0)))) if sd > 0 else 0.0))
    return pd.DataFrame(rows).set_index("variant")


# ---------------------------------------------------------------- league plumbing (needs the real connectome files)
def _job(aid, variant, r):
    from tournament.league import G, build_agent, fly_scores
    cfg, data, store = G["cfg"], G["data"], G["store"]
    g = store.load_agent(aid)["genome"]
    c, v0, v1 = data.round_geometry(r)
    t0 = time.time()
    idx, ics = run_variant(g, cfg, len(data.feats), data, c, v0, v1, variant, build_agent, fly_scores)
    return aid, variant, idx, ics, time.time() - t0


def main():
    from tournament.league import _map, sel_score
    from tournament.store import Store
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="league_v7")
    ap.add_argument("--round", type=int, default=-1, help="-1 = the last finished round")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    store = Store(a.root)
    cfg = store.load_config()
    r = a.round if a.round >= 0 else store.load_state()["round"] - 1
    flies = sorted((x for x in store.alive() if x["history"]), key=lambda x: -sel_score(x))[:a.top]
    variants = [v for v in a.variants.split(",") if v in VARIANTS]
    if "real" not in variants:
        variants.insert(0, "real")
    jobs = [(f["id"], v, r) for f in flies for v in variants]
    print(f"round {r}: {len(flies)} flies x {len(variants)} variants = {len(jobs)} runs on {a.workers} workers "
          f"(each trains from scratch; the slow ones take a few minutes)", flush=True)
    res, idxs = {}, None
    for k, (aid, v, idx, ics, secs) in enumerate(_map(_job, jobs, cfg, a.workers), 1):
        res.setdefault(v, {})[aid] = ics
        idxs = idx
        print(f"  [{k}/{len(jobs)}] {aid} {v:<16} IC {np.mean(ics):+.3f}  ({secs:.0f}s)", flush=True)
    from tournament.data import Data
    gap = Data.from_cfg(cfg).gap
    df = summarize(res, len(idxs) * cfg["score_every"] / gap)
    pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x:+.3f}")
    print(f"\n=== ablation, round {r}, validation IC (mean over {len(flies)} flies) ===")
    print(df.to_string())
    print("\nreal_minus_variant > 0 with |t| > 2: the removed part was helping. About 0 or negative: it was doing nothing.\n"
          f"(t uses ~{len(idxs) * cfg['score_every'] / gap:.0f} independent samples; {len(flies)} flies share one validation window, so treat t as a rough guide.)")
    out = store.root / f"ablation_r{r:02d}.csv"
    df.to_csv(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
