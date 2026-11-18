"""What is the fly actually computing? (T5 step 3, no retraining)

    python -m experiments.decode_fly --root league_v7                 # last finished round, every fly that has saved scores
    python -m experiments.decode_fly --root league_v7 --round 5

Uses the scores the league already saved (rounds/scores/rXX_<id>.npz). For each fly and each scored date, across that day's
stocks: the rank correlation of the fly's score with every input feature, with the ridge model's prediction (ridge fitted on the
round's training dates, same as the league's control), and the cross-sectional R^2 of the score's rank on ALL features together.
If one feature (or ridge) explains most of a fly's score, the fly is relaying that feature, not computing something of its own.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from tournament.data import rank_ic


def _rank_corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def decode_scores(Xs, scores, ys, feats, ridge_preds=None) -> dict:
    """Xs / scores / ys / ridge_preds: lists with one array per scored date (same stocks). Returns per-feature mean rank correlation
    (signed), mean |corr|, the share of the score's rank variance the whole feature set explains (R^2), IC, and ridge correlation."""
    corr = np.full((len(Xs), len(feats)), np.nan)
    r2, ic, rc = [], [], []
    for d, (X, s, y) in enumerate(zip(Xs, scores, ys)):
        rs = rankdata(s)
        for k in range(len(feats)):
            corr[d, k] = _rank_corr(s, X[:, k])
        A = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(A, rs, rcond=None)
        sst = ((rs - rs.mean()) ** 2).sum()
        r2.append(1.0 - ((rs - A @ beta) ** 2).sum() / sst if sst > 0 else np.nan)
        ic.append(rank_ic(s, y))
        if ridge_preds is not None:
            rc.append(_rank_corr(s, ridge_preds[d]))
    m = pd.Series(np.nanmean(corr, axis=0), index=feats)
    return dict(feature_corr=m, abs_corr=pd.Series(np.nanmean(np.abs(corr), axis=0), index=feats),
                r2_all_features=float(np.nanmean(r2)), ic=float(np.nanmean(ic)),
                ridge_corr=float(np.nanmean(rc)) if rc else np.nan)


def main():
    from learning.ridge import fit_ridge, predict
    from tournament.data import Data
    from tournament.store import Store
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="league_v7")
    ap.add_argument("--round", type=int, default=-1, help="-1 = the last finished round")
    a = ap.parse_args()
    store = Store(a.root)
    cfg = store.load_config()
    r = a.round if a.round >= 0 else store.load_state()["round"] - 1
    data = Data.from_cfg(cfg)
    c, v0, v1 = data.round_geometry(r)
    files = sorted((store.root / "rounds" / "scores").glob(f"r{r:02d}_*.npz"))
    if not files:
        raise SystemExit(f"no saved scores for round {r} in {store.root / 'rounds' / 'scores'}")
    X, y = data.stack_train(c)
    coef, xm = fit_ridge(X, y)
    rows, fc = [], {}
    for f in files:
        aid = f.stem.split("_", 1)[1]
        z = np.load(f)
        idx = sorted(int(n[1:]) for n in z.files)
        Xs = [data.day[i][1] for i in idx]
        d = decode_scores(Xs, [z[f"d{i}"].astype(float) for i in idx], [data.day[i][2] for i in idx], data.feats,
                          [predict(x, coef, xm) for x in Xs])
        top = d["feature_corr"].reindex(d["feature_corr"].abs().sort_values(ascending=False).index)[:3]
        rows.append(dict(id=aid, ic=d["ic"], corr_ridge=d["ridge_corr"], r2_all_features=d["r2_all_features"],
                         top_feature_1=f"{top.index[0]} {top.iloc[0]:+.2f}", top_feature_2=f"{top.index[1]} {top.iloc[1]:+.2f}",
                         top_feature_3=f"{top.index[2]} {top.iloc[2]:+.2f}"))
        fc[aid] = d["feature_corr"]
    df = pd.DataFrame(rows).set_index("id").sort_values("ic", ascending=False)
    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:+.3f}")
    print(f"=== round {r}: what each fly's score tracks (rank correlation across stocks, mean over {len(idx)} dates) ===")
    print(df.to_string())
    avg = pd.DataFrame(fc).mean(axis=1)
    print("\naverage over flies, per feature (signed rank correlation with the fly's score):")
    print(avg.reindex(avg.abs().sort_values(ascending=False).index).head(10).to_string())
    print("\nr2_all_features: share of the score's rank variance that a linear model on all features explains (1.0 = the fly is just a "
          "linear function of the inputs).\ncorr_ridge: how closely the fly follows the ridge model. Both near 1 = the fly is ridge in disguise.")
    out = store.root / f"decode_r{r:02d}.csv"
    df.to_csv(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
