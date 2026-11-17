"""Training labels for the league: any horizon, blends of horizons, and sector-neutral versions (T6c).

spec is "21" (the original 21-day label, the default), "63", or a blend such as "5+21+63".
  * single horizon: excess forward return = compounded next-day returns of days t+1..t+h minus that day's cross-sectional mean.
    "21" with no sectors returns exactly what tournament.data.monthly_labels always returned (existing leagues are unchanged).
  * blend: each horizon's label is z-scored across stocks per day and the z-scores are averaged, so short and long horizons count
    equally. A row needs ALL horizons known.
  * sectors (ticker -> sector): the day's sector mean is subtracted too, so a fly cannot score well just by tilting to a sector.
    Groups with fewer than 3 stocks that day (and tickers with no sector, grouped as "unknown") fall back to market-neutral.
The league's train/validation GAP must be at least the longest horizon: make_labels returns it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def parse_label(spec) -> list[int]:
    hs = [int(x) for x in str(spec).replace(",", "+").split("+") if x.strip()]
    if not hs or any(h < 1 for h in hs):
        raise ValueError(f"bad label spec {spec!r}: use e.g. 21 or 5+21+63")
    return hs


def horizon_excess(panel: pd.DataFrame, h: int) -> pd.Series:
    if h == 21 and "fwd_excess_21" in panel.columns:
        return panel["fwd_excess_21"]
    fr = panel["fwd_ret"].unstack("ticker").sort_index()
    roll = np.expm1(np.log1p(fr).rolling(h, min_periods=h).sum().shift(-(h - 1)))
    s = roll.sub(roll.mean(axis=1), axis=0).stack()
    return s[~s.index.duplicated()].reindex(panel.index)


def sector_neutral(y: pd.Series, sectors: dict, min_group: int = 3) -> pd.Series:
    d = y.rename("y").reset_index()
    d["sec"] = d["ticker"].map(sectors).fillna("unknown")
    g = d.groupby(["date", "sec"])["y"]
    mean, n = g.transform("mean"), g.transform("count")
    out = d["y"] - mean.where(n >= min_group, 0.0)
    return pd.Series(out.to_numpy(), index=y.index)


def _z(y: pd.Series) -> pd.Series:
    g = y.groupby(level="date")
    return (y - g.transform("mean")) / (g.transform("std") + 1e-12)


def make_labels(panel: pd.DataFrame, spec="21", sectors: dict | None = None) -> tuple[pd.Series, int]:
    """-> (label aligned to the panel rows, gap in trading days = the longest horizon)."""
    hs = parse_label(spec)
    ys = [horizon_excess(panel, h) for h in hs]
    if sectors:
        ys = [sector_neutral(y, sectors) for y in ys]
    if len(ys) == 1:
        return ys[0], hs[0]
    zs = pd.concat([_z(y) for y in ys], axis=1)
    y = zs.mean(axis=1, skipna=False)                       # NaN if any horizon is unknown
    return y, max(hs)


def load_sectors(path) -> dict:
    """ticker -> sector from universe/constituents.csv (columns symbol, sector). Removed names have no sector."""
    df = pd.read_csv(path)
    col = "sector" if "sector" in df.columns else "sub_industry"
    return {str(r.symbol): str(getattr(r, col)) for r in df.itertuples() if pd.notna(getattr(r, col))}
