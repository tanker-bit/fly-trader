"""Monthly labels, the data splits, and rank-IC helpers for the league.

Label: forward 21-trading-day return minus the cross-sectional mean (excess), built from the panel's daily
next-day returns. The label at date t uses returns of days t+1 ... t+21, so it is only KNOWN 21 days later.
That is why every split leaves a 21-day GAP between the last training date and the first validation date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from env.features import feature_columns
from experiments.backtest import MARKET_COLS

HORIZON = 21
GAP = 21


def load_panel(path):
    return pd.read_pickle(path) if str(path).endswith(".pkl") else pd.read_parquet(path)


def monthly_labels(panel: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    """Excess 21-day forward return per (date, ticker); NaN where the next 21 daily returns are not all known.
    If the panel already carries `fwd_excess_21` (built from the full price table, env/features.build_panel horizon21=True)
    that column is used: it keeps the label for stocks that leave the index inside the window."""
    if "fwd_excess_21" in panel.columns:
        return panel["fwd_excess_21"]
    fr = panel["fwd_ret"].unstack("ticker").sort_index()
    roll = np.log1p(fr).rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))
    ret = np.expm1(roll)
    exc = ret.sub(ret.mean(axis=1), axis=0)
    s = exc.stack()
    s = s[~s.index.duplicated()]
    return s.reindex(panel.index)


def rank_ic(s, y) -> float:
    s, y = np.asarray(s, float), np.asarray(y, float)
    ok = np.isfinite(s) & np.isfinite(y)
    if ok.sum() < 10 or np.std(s[ok]) < 1e-12:
        return 0.0
    return float(np.corrcoef(rankdata(s[ok]), rankdata(y[ok]))[0, 1])


class Data:
    """dates[i] / day[i] = (tickers, X, y21) for the i-th labelled date. The last `holdout` dates are never used by the league."""

    def __init__(self, panel_path, train_days, val_days, holdout, min_stocks=10):
        panel = load_panel(panel_path)
        self.feats = [c for c in feature_columns(panel) if c not in MARKET_COLS]
        y = monthly_labels(panel)
        ok = y.notna()
        sub = panel.loc[ok, self.feats].copy()
        sub["_y"] = y[ok].to_numpy()
        self.dates, self.day = [], []
        for d, g in sub.groupby(level="date"):
            if len(g) < min_stocks:
                continue
            self.dates.append(d)
            self.day.append((g.index.get_level_values("ticker").to_numpy(), g[self.feats].to_numpy(np.float32),
                             g["_y"].to_numpy(float)))
        self.fwd1 = panel["fwd_ret"].unstack("ticker").reindex(self.dates)
        self.train_days, self.val_days, self.holdout = train_days, val_days, holdout
        self.i_mom, self.i_rev = self.feats.index("mom_12_1"), self.feats.index("ret_1")

    @property
    def n_usable(self):
        return len(self.dates) - self.holdout

    def round_geometry(self, r):
        """(c, v0, v1): train on dates[..c]; validate on dates[v0:v1] with v0 = c + GAP."""
        c = self.train_days - 1 + r * self.val_days
        return c, c + GAP, c + GAP + self.val_days

    def round_valid(self, r):
        return self.round_geometry(r)[2] <= self.n_usable

    def stack_train(self, upto):
        X = np.vstack([self.day[i][1] for i in range(upto + 1)])
        y = np.concatenate([self.day[i][2] for i in range(upto + 1)])
        return X, y

    def y_frame(self, idxs):
        return pd.DataFrame({self.dates[i]: pd.Series(self.day[i][2], index=self.day[i][0]) for i in idxs}).T
