"""Ridge regression baseline: the boring linear model the fly has to beat.

Fits next-day relative return on the (cross-sectionally ranked) stock features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_ridge(X: np.ndarray, y: np.ndarray, lam_rel: float = 0.01):
    """Closed-form ridge with centering. lam_rel scales the penalty with sample size."""
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    n, d = Xc.shape
    coef = np.linalg.solve(Xc.T @ Xc + lam_rel * n * np.eye(d), Xc.T @ yc)
    return coef, xm


def predict(X: np.ndarray, coef: np.ndarray, xm: np.ndarray) -> np.ndarray:
    return (X - xm) @ coef


def walk_forward_ridge(panel: pd.DataFrame, features: list[str], target: str = "fwd_excess",
                       min_train_days: int = 1000, refit_every: int = 250, lam_rel: float = 0.01):
    """Out-of-sample scores: refit every `refit_every` days on ALL earlier days only.

    Returns (scores aligned to panel.index, DataFrame of coefficients per refit date).
    """
    dates = panel.index.get_level_values("date")
    uniq = dates.unique().sort_values()
    X = panel[features].to_numpy(float)
    y = panel[target].to_numpy(float)
    d = dates.to_numpy()
    scores = np.full(len(panel), np.nan)
    coefs, when = [], []
    for i in range(min_train_days, len(uniq), refit_every):
        t0 = uniq[i]
        t1 = uniq[i + refit_every] if i + refit_every < len(uniq) else uniq[-1] + pd.Timedelta(days=1)
        tr = (d < t0) & ~np.isnan(y)       # only information available before t0
        te = (d >= t0) & (d < t1)
        coef, xm = fit_ridge(X[tr], y[tr], lam_rel)
        scores[te] = predict(X[te], coef, xm)
        coefs.append(coef)
        when.append(t0)
    return pd.Series(scores, index=panel.index), pd.DataFrame(coefs, index=when, columns=features)
