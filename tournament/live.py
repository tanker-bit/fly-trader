"""V4b-1: score stocks for a live trading day with a league's trained flies (no training, no labels, no holdout).

    store, cfg = Store("league_v8"), Store("league_v8").load_config()
    flies = load_flies(store, cfg, pick_flies(store, "kept"))
    scores = score_rows(flies, X, cfg["feats"])        # X: one row per ticker, the league's feature columns

Which flies (`pick_flies`):
  "kept"  every living fly that was scored in the league's latest round, has saved weights and was not silent: the survivors of
          the last selection (league_v8: the 10 KEPT in round 18). Newborn children that were never scored are left out.
  "best"  the single fly with the best selection score (mean validation IC over its last 3 rounds).
  [ids]   exactly these flies (each must be alive and have weights).

`score_rows` returns one column per fly plus "committee": the average of the flies' scores after z-scoring each fly across today's
stocks, the same committee the league reports. A higher score = the fly likes the stock more. Scores only rank stocks against each
other on the same day; their size means nothing on its own.

X must hold exactly what the league trained on: every column in cfg["feats"], already ranked across stocks to [-1, 1], no gaps
(unknown fundamentals as 0, as in the hybrid panel). Anything else raises instead of quietly scoring garbage. Building X for today
is V4b-2. Reminder from the research: no tradable edge has been shown (see PROJECT_HANDOVER_CURRENT.md section 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tournament.groups import view_cols
from tournament.league import _z, fly_scores, load_agent_obj, sel_score

COMMITTEE = "committee"


@dataclass
class Fly:
    id: str
    agent: object
    cols: np.ndarray | None          # visible feature indices (None = sees every feature)
    rec: dict


def _eligible(store) -> list[dict]:
    """Alive flies scored (not silent) in the latest round that any fly was scored in, with weights on disk."""
    scored = [r for r in store.alive() if r.get("history")]
    if not scored:
        return []
    latest = max(r["history"][-1]["round"] for r in scored)
    return [r for r in scored
            if r["history"][-1]["round"] == latest and not r["history"][-1].get("silent") and store.load_weights(r["id"]) is not None]


def pick_flies(store, which="kept") -> list[dict]:
    """Agent records to use (see the module docstring). Raises ValueError when nothing usable is found."""
    if isinstance(which, str) and which == "kept":
        recs = _eligible(store)
    elif isinstance(which, str) and which == "best":
        el = _eligible(store)
        recs = [max(el, key=sel_score)] if el else []
    elif isinstance(which, str):
        raise ValueError(f"unknown fly choice {which!r}: use 'kept', 'best' or a list of ids")
    else:
        alive = {r["id"]: r for r in store.alive()}
        recs = []
        for aid in which:
            if aid not in alive:
                raise ValueError(f"fly {aid} is not alive in this league")
            if store.load_weights(aid) is None:
                raise ValueError(f"fly {aid} has no saved weights (never trained)")
            recs.append(alive[aid])
    if not recs:
        raise ValueError("no usable flies: none alive with weights and a scored, non-silent latest round")
    return sorted(recs, key=lambda r: r["id"])


def load_flies(store, cfg, recs) -> list[Fly]:
    """Rebuild each fly's circuit and load its trained weights. Paths in cfg (connectome csvs) are relative to the project root."""
    feats = list(cfg["feats"])
    shim = SimpleNamespace(feats=feats)                  # load_agent_obj only needs the feature list
    out = []
    for r in recs:
        ag = load_agent_obj(r, store, cfg, shim)
        W = store.load_weights(r["id"])
        if W is None or W.shape != ag.syn.W.shape:
            raise ValueError(f"fly {r['id']}: saved weights missing or the wrong shape {None if W is None else W.shape} "
                             f"for its circuit {ag.syn.W.shape}")
        out.append(Fly(r["id"], ag, view_cols(r["genome"], feats), r))
    return out


def check_features(X: pd.DataFrame, feats) -> np.ndarray:
    """X as a float32 array in the league's column order, after checking it is what the flies were trained on."""
    feats = list(feats)
    missing = [f for f in feats if f not in X.columns]
    if missing:
        raise ValueError(f"feature rows lack {len(missing)} column(s) the flies need: {missing}")
    if len(X) < 2:
        raise ValueError("need at least 2 stocks to rank")
    if X.index.has_duplicates:
        raise ValueError(f"duplicate tickers: {sorted(set(X.index[X.index.duplicated()]))[:10]}")
    A = X[feats].to_numpy(np.float64)
    bad = ~np.isfinite(A)
    if bad.any():
        r, c = np.argwhere(bad)[0]
        raise ValueError(f"{int(bad.sum())} missing or infinite value(s), e.g. {X.index[r]} / {feats[c]}")
    if (np.abs(A) > 1.0 + 1e-6).any():
        r, c = np.argwhere(np.abs(A) > 1.0 + 1e-6)[0]
        raise ValueError(f"values outside [-1, 1] (features must be cross-sectional ranks), e.g. {X.index[r]} / {feats[c]} = {A[r, c]}")
    return A.astype(np.float32)


def score_rows(flies, X: pd.DataFrame, feats) -> pd.DataFrame:
    """Scores for every row (ticker) of X: one column per fly, then the committee. Row order follows X."""
    A = check_features(X, feats)
    out = {}
    for f in flies:
        out[f.id] = fly_scores(f.agent, A if f.cols is None else A[:, f.cols])
    df = pd.DataFrame(out, index=X.index)
    df[COMMITTEE] = np.mean([_z(df[f.id].to_numpy()) for f in flies], axis=0)
    return df
