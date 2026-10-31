"""Do the features contain ANY signal about next-day returns? (information coefficient)

    python -m experiments.feature_ic

For each stock feature: every day, rank-correlate the feature with the next-day return across
stocks (the daily IC). Report the mean IC, its t-statistic, and the hit rate (share of days with
IC > 0). For market-wide features (same for all stocks) we instead correlate the feature
with the next-day average market return over time.

How to read it:
  * Real, tradable signals in liquid stocks are tiny: mean IC of 0.01-0.03 is typical.
  * |t| above ~3 is a meaningful result; |t| below ~2 is noise. With dozens of features,
    expect one or two |t| > 2 purely by luck.
  * A feature needs to stay significant in BOTH halves of the data to be trusted.
Uses the "fwd_excess" label (return relative to that day's average stock) for stock features,
so market direction cannot leak in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from env.features import feature_columns

MARKET_COLS = {"mkt_ret_1", "mkt_ret_5", "mkt_ret_20", "vix_z", "vix_chg_5", "usdcad_ret_20", "oil_ret_20"}


def daily_ic(panel: pd.DataFrame, col: str) -> pd.Series:
    sub = panel[[col, "fwd_excess"]].dropna()
    g = sub.groupby(level="date")
    # rank both within each day (Spearman), then Pearson on the ranks
    a = g[col].rank()
    b = g["fwd_excess"].rank()
    df = pd.DataFrame({"a": a, "b": b})
    def corr(x):
        return x["a"].corr(x["b"]) if len(x) >= 10 else np.nan
    return df.groupby(level="date").apply(corr).dropna()


def summarize(ic: pd.Series) -> dict:
    n = len(ic)
    sd = ic.std()
    return {
        "mean_ic": ic.mean(),
        "t": ic.mean() / sd * np.sqrt(n) if sd > 0 else np.nan,
        "hit": (ic > 0).mean(),
        "n_days": n,
    }


def main():
    panel = pd.read_parquet("data/panel.parquet")
    panel = panel[panel.fwd_ret.notna()]
    feats = feature_columns(panel)
    stock_cols = [c for c in feats if c not in MARKET_COLS]
    dates = panel.index.get_level_values("date")
    mid = dates.unique().sort_values()[len(dates.unique()) // 2]

    rows = []
    for c in stock_cols:
        ic = daily_ic(panel, c)
        s1, s2 = summarize(ic[ic.index < mid]), summarize(ic[ic.index >= mid])
        s = summarize(ic)
        rows.append({"feature": c, "mean_IC": s["mean_ic"], "t": s["t"], "hit%": 100 * s["hit"],
                     "t_first_half": s1["t"], "t_second_half": s2["t"]})
    out = pd.DataFrame(rows).set_index("feature").sort_values("t", key=abs, ascending=False)
    print("Stock features: cross-sectional IC vs next-day relative return")
    print(out.round(3).to_string())

    mkt_ret = panel.groupby(level="date").fwd_ret.mean()
    print("\nMarket features: correlation with next-day average return (time series)")
    rows = []
    for c in sorted(MARKET_COLS & set(feats)):
        x = panel[c].groupby(level="date").first()
        j = pd.concat([x, mkt_ret], axis=1).dropna()
        r = j.iloc[:, 0].corr(j.iloc[:, 1])
        rows.append({"feature": c, "corr": r, "t": r * np.sqrt((len(j) - 2) / (1 - r ** 2)), "n_days": len(j)})
    print(pd.DataFrame(rows).set_index("feature").round(3).to_string())

    n_sig = int((out.t.abs() > 3).sum())
    print(f"\n{n_sig} of {len(out)} stock features have |t| > 3 over the full sample.")


if __name__ == "__main__":
    main()
