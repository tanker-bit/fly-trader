"""Do the flies beat one-line rules? (T5 step 4)

Core: score simple rules on exactly the validation dates the league used, round by round, with no training. Part 2 adds the
command line and the summary against the flies' own saved results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from learning.ridge import fit_ridge, predict
from tournament.data import rank_ic

RULES = ["-ma50_dist", "-ret_20", "-rsi_14", "momentum 12-1", "ridge"]


def rule_scores(data, name: str, Xi, ridge=None):
    """Score array for one day's stocks. `-name` = minus that feature (so a reversal rule); 'ridge' uses the fitted (coef, xm)."""
    if name == "ridge":
        return predict(Xi, *ridge)
    if name == "momentum 12-1":
        return Xi[:, data.feats.index("mom_12_1")]
    if name.startswith("-"):
        return -Xi[:, data.feats.index(name[1:])]
    return Xi[:, data.feats.index(name)]


def rule_ics(data, rounds, score_every: int, rules=RULES) -> pd.DataFrame:
    """Mean rank IC of each rule over each round's validation dates (the same dates the league scored: range(v0, v1, score_every)).
    Ridge is fitted on the round's training dates only (<= c), as the league's own ridge control is. -> DataFrame rounds x rules."""
    rows = {}
    for r in rounds:
        c, v0, v1 = data.round_geometry(r)
        idx = list(range(v0, v1, score_every))
        ridge = None
        if "ridge" in rules:
            X, y = data.stack_train(c)
            ridge = fit_ridge(X, y)
        row = {}
        for name in rules:
            row[name] = float(np.nanmean([rank_ic(rule_scores(data, name, data.day[i][1], ridge), data.day[i][2]) for i in idx]))
        rows[r] = row
    return pd.DataFrame(rows).T
