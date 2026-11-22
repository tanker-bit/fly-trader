"""Is the league's selection mostly luck? (A4) Read-only: uses the flies' saved scores, no training.

    python -m experiments.selection_noise --root league_v7 --out selnoise

Builds the matrix of every fly's mean rank IC (against the league's own label) in every round it scored, then asks three questions:
  1. persistence: among flies that scored both round r and r+1, does a high IC in r go with a high IC in r+1? (rank correlation per
     round pair, averaged). Near 0 means the ranking is noise: the league keeps and culls on luck.
  2. kept vs new: in round r+1, do the flies kept from round r score higher than the children that replaced the culled ones?
     A positive gap is the evidence that keeping the better half helps.
  3. top vs bottom half: among the flies present in both rounds, does the better half in r beat the worse half in r+1?
The t-stats treat round pairs as independent; they share a market, so they are optimistic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tournament.data import rank_ic


def ic_matrix(store, data, rounds) -> pd.DataFrame:
    """agent x round mean rank IC from rounds/scores/rXX_<id>.npz (dead circuits with constant scores count as IC 0)."""
    cols = {}
    for r in rounds:
        col = {}
        for f in sorted((store.root / "rounds" / "scores").glob(f"r{r:02d}_*.npz")):
            z = np.load(f)
            ics = [rank_ic(z[k].astype(float), data.day[int(k[1:])][2]) for k in z.files if int(k[1:]) < len(data.day)]
            col[f.name[4:-4]] = float(np.nanmean(ics)) if ics else np.nan
        cols[r] = pd.Series(col, dtype=float)
    return pd.DataFrame(cols)


def _t(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


def pair_stats(m: pd.DataFrame, min_flies: int = 6) -> pd.DataFrame:
    """One row per round pair (r, r+1): persistence (Spearman of IC_r and IC_r+1 among flies in both), kept-minus-new gap in r+1,
    and top-half-minus-bottom-half gap in r+1 among flies in both."""
    rows = {}
    rounds = sorted(m.columns)
    for a, b in zip(rounds, rounds[1:]):
        if b != a + 1:
            continue
        both = m[[a, b]].dropna()
        in_b = m[b].dropna()
        new = in_b[~in_b.index.isin(both.index)]
        row = {"n_both": len(both), "n_new": len(new)}
        row["persistence"] = float(both[a].rank().corr(both[b].rank())) if len(both) >= min_flies else np.nan
        row["kept_minus_new"] = float(both[b].mean() - new.mean()) if len(both) and len(new) else np.nan
        if len(both) >= min_flies:
            top = both[a] >= both[a].median()
            row["top_minus_bottom"] = float(both.loc[top, b].mean() - both.loc[~top, b].mean())
        else:
            row["top_minus_bottom"] = np.nan
        rows[f"{a}->{b}"] = row
    return pd.DataFrame(rows).T


def summarize(p: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in ["persistence", "kept_minus_new", "top_minus_bottom"]:
        x = p[c].dropna()
        out[c] = {"mean": float(x.mean()) if len(x) else np.nan, "t": _t(x), "pairs": len(x), "pairs_positive": int((x > 0).sum())}
    return pd.DataFrame(out).T


def main():
    import argparse

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
    m = ic_matrix(store, data, rounds)
    p = pair_stats(m)
    pd.set_option("display.width", 200)
    print(f"\n{m.shape[0]} flies over {len(rounds)} rounds. Mean rank IC of the flies each round (spread = how much flies differ):\n")
    print(pd.DataFrame({"mean": m.mean(), "std across flies": m.std(), "best": m.max(), "worst": m.min(), "flies": m.count()}).round(3).T.to_string())
    print("\nPer round pair\n")
    print(p.round(3).to_string())
    print("\nAcross round pairs (positive = selection is working; t is optimistic)\n")
    print(summarize(p).round(3).to_string())
    if a.out:
        m.to_csv(f"{a.out}_ic_matrix.csv")
        p.to_csv(f"{a.out}_pairs.csv")


if __name__ == "__main__":
    main()
