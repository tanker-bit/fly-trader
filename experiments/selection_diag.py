"""Why is the league's selection noise? (A5) Read-only diagnosis from the saved scores and agent records, no training.

    python -m experiments.selection_diag --root league_v7 --out seldiag

Three questions:
  1. diversity: how alike are the 20 flies of a round? Mean rank correlation between their scores (1 = copies, 0 = unrelated).
  2. settings: does any setting (kc_sparsity, kc_drive, lr0, ...) predict a fly's IC inside a round? Rank correlation of the setting with
     IC across the round's flies, averaged over rounds. Near 0 for every setting means the settings we evolve do not matter.
  3. detectable skill: if one fly knew a little more than the rest, would the league see it? A copy of the median fly of each round gets
     a planted dose of the label (score + c * label, z-scored per date); we record the IC gain that gives and whether the planted fly
     would have ranked first or in the kept (top) half. c = 0 is the control: a copy of the median fly should land in the top half
     about half the time.
The t-stats treat rounds as independent; they share a market, so they are optimistic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from tournament.data import rank_ic

LIFTS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)


def _z(x) -> np.ndarray:
    x = np.asarray(x, float)
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if sd > 1e-12 else np.zeros_like(x)


def load_round(store, r) -> dict:
    """{agent id: {date idx: score array}} for one round."""
    out = {}
    for f in sorted((store.root / "rounds" / "scores").glob(f"r{r:02d}_*.npz")):
        z = np.load(f)
        out[f.name[4:-4]] = {int(k[1:]): z[k].astype(float) for k in z.files}
    return out


def diversity(flies: dict, max_dates: int = 40) -> dict:
    """Mean and minimum pairwise rank correlation of the flies' scores (dead circuits left out), over up to max_dates dates."""
    ids = [a for a, d in flies.items() if d and all(np.nanstd(v) > 1e-9 for v in d.values())]
    if len(ids) < 3:
        return dict(mean_corr=np.nan, min_corr=np.nan, flies=len(ids))
    dates = sorted(set.intersection(*[set(flies[a]) for a in ids]))
    dates = dates[:: max(len(dates) // max_dates, 1)]
    cs = []
    for i in dates:
        M = np.vstack([rankdata(np.nan_to_num(flies[a][i])) for a in ids])
        C = np.corrcoef(M)
        cs.append(C[np.triu_indices(len(ids), 1)])
    cs = np.vstack(cs).mean(axis=0) if cs else np.array([np.nan])
    return dict(mean_corr=float(cs.mean()), min_corr=float(cs.min()), flies=len(ids))


def genome_effects(m: pd.DataFrame, genomes: dict, min_flies: int = 8) -> pd.DataFrame:
    """Per gene: the mean over rounds of the Spearman correlation between the gene and the fly's IC in that round (+ t, rounds used)."""
    g = pd.DataFrame(genomes).T.apply(pd.to_numeric, errors="coerce")
    rows = {}
    for gene in g.columns:
        cs = []
        for r in m.columns:
            x = pd.concat([g[gene], m[r]], axis=1, keys=["g", "ic"]).dropna()
            if len(x) >= min_flies and x["g"].nunique() > 2:
                cs.append(float(x["g"].rank().corr(x["ic"].rank())))
        if cs:
            s = pd.Series(cs)
            rows[gene] = dict(mean_corr=s.mean(), t=s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 and s.std(ddof=1) > 0 else np.nan,
                              rounds=len(s), spread=float(g[gene].std() / (abs(g[gene].mean()) + 1e-12)))
    return pd.DataFrame(rows).T


def lift_detection(data, flies: dict, lifts=LIFTS) -> pd.DataFrame:
    """One round: for each lift c, the IC gain of the planted copy of the median fly, and whether it ranks first / in the top half."""
    dates = sorted(set.intersection(*[set(d) for d in flies.values()])) if flies else []
    dates = [i for i in dates if i < len(data.day)]
    if len(dates) < 3 or len(flies) < 4:
        return pd.DataFrame()

    def ic(score_fn):
        return float(np.mean([rank_ic(score_fn(i), data.day[i][2]) for i in dates]))

    real = {a: ic(lambda i, a=a: flies[a][i]) for a in flies}
    ids = sorted(real, key=real.get)
    base = ids[len(ids) // 2]
    out = {}
    for c in lifts:
        planted = ic(lambda i: _z(flies[base][i]) + c * np.nan_to_num(_z(data.day[i][2])))
        others = [v for a, v in real.items() if a != base]
        out[c] = dict(ic_gain=planted - real[base], first=float(planted > max(others)),
                      top_half=float(planted >= np.median(list(real.values()))))
    return pd.DataFrame(out).T


def main():
    import argparse

    from experiments.selection_noise import ic_matrix
    from tournament.data import Data
    from tournament.store import Store

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="league_data")
    ap.add_argument("--out", default=None, help="CSV prefix")
    a = ap.parse_args()
    store = Store(a.root)
    cfg = store.load_config()
    print("loading panel...", flush=True)
    data = Data.from_cfg(cfg)
    rounds = sorted({int(f.name[1:3]) for f in (store.root / "rounds" / "scores").glob("r*_*.npz")})
    pd.set_option("display.width", 200)

    div, lifts = {}, []
    for r in rounds:
        flies = load_round(store, r)
        div[r] = diversity(flies)
        lf = lift_detection(data, flies)
        if len(lf):
            lifts.append(lf)
        print(f"round {r} done", flush=True)
    d = pd.DataFrame(div).T
    print("\n1. DIVERSITY: rank correlation between the flies' scores in each round (1 = copies)\n")
    print(d.round(3).T.to_string())
    print(f"\nmean over rounds {d['mean_corr'].mean():.3f}; rounds 0-4: {d.loc[d.index <= 4, 'mean_corr'].mean():.3f}; "
          f"rounds 5+: {d.loc[d.index >= 5, 'mean_corr'].mean():.3f}")

    m = ic_matrix(store, data, rounds)
    genomes = {aid: store.load_agent(aid)["genome"] for aid in m.index}
    ge = genome_effects(m, genomes)
    print("\n2. SETTINGS: does a setting predict a fly's IC within a round? (mean rank corr over rounds; |t| > 2 would be notable)\n")
    print(ge.round(3).to_string() if len(ge) else "not enough variation")

    if lifts:
        allr = pd.concat(lifts, keys=range(len(lifts)))
        s = allr.groupby(level=1).mean()
        print(f"\n3. DETECTABLE SKILL: a copy of the median fly with a planted dose of the label, {len(lifts)} rounds\n"
              "   ic_gain = IC points added; first = share of rounds it would rank #1 of the round; top_half = share it would be KEPT\n"
              "   (c = 0 is the control: about 0.5 expected)\n")
        print(s.round(3).to_string())
    if a.out:
        d.to_csv(f"{a.out}_diversity.csv")
        ge.to_csv(f"{a.out}_settings.csv")
        if lifts:
            s.to_csv(f"{a.out}_detectable.csv")


if __name__ == "__main__":
    main()
