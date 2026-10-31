"""Feature panel for the multi-stock fly trader.

Rules that keep this honest:
  * Every feature at date t uses only data up to and including the close of day t.
  * Stock features are ranked ACROSS stocks on the same day (scaled to [-1, 1]). That
    removes scale issues and market-wide moves, and uses nothing from the future.
  * Market features are rolling z-scores over the past year only.
  * The target (next-day return) is returned separately and never mixed into features.
tests/test_features.py checks the no-look-ahead rule directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_STOCKS_PER_DAY = 10      # need a real cross-section to rank
MIN_HISTORY = 260            # trading days of history before a stock can be used
MIN_PRICE = 1.0              # drop penny prices
ZWIN = 252


def _ret(close, n):
    return close / close.shift(n) - 1.0


def _xs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank scaled to [-1, 1]; NaN on days with too few stocks."""
    r = df.rank(axis=1, pct=True) * 2.0 - 1.0
    return r.where(df.notna().sum(axis=1) >= MIN_STOCKS_PER_DAY)


def _rolling_z(s: pd.Series) -> pd.Series:
    m = s.rolling(ZWIN, min_periods=ZWIN // 2).mean()
    sd = s.rolling(ZWIN, min_periods=ZWIN // 2).std()
    return ((s - m) / sd).clip(-4, 4)


def stock_features(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    close = close.ffill(limit=3)                      # tolerate a few missing quotes
    lr = np.log(close / close.shift(1))
    vol20, vol60 = lr.rolling(20).std(), lr.rolling(60).std()
    gain = close.diff().clip(lower=0).rolling(14).mean()
    loss = (-close.diff().clip(upper=0)).rolling(14).mean()
    dvol = volume.rolling(5).mean() / volume.rolling(60).mean()

    raw = {
        "ret_1": _ret(close, 1),
        "ret_5": _ret(close, 5),
        "ret_20": _ret(close, 20),
        "ret_60": _ret(close, 60),
        "ret_120": _ret(close, 120),
        "mom_12_1": close.shift(21) / close.shift(252) - 1.0,   # classic 12-1 momentum
        "vol_20": vol20,
        "vol_ratio": vol20 / vol60,
        "volume_ratio": np.log(dvol.where(dvol > 0)),
        "ma50_dist": close / close.rolling(50).mean() - 1.0,
        "ma200_dist": close / close.rolling(200).mean() - 1.0,
        "rsi_14": gain / (gain + loss),
        "dist_52w_high": close / close.rolling(252).max() - 1.0,
    }
    valid = (close.notna().rolling(MIN_HISTORY).sum() >= MIN_HISTORY) & (close >= MIN_PRICE)
    return {k: _xs_rank(v.where(valid)) for k, v in raw.items()}


def market_features(market: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    m = market.reindex(market.index.union(index)).ffill(limit=3).reindex(index)
    out = pd.DataFrame(index=index)
    out["mkt_ret_1"] = _ret(m["tsx"], 1) / 0.01
    out["mkt_ret_5"] = _ret(m["tsx"], 5) / 0.02
    out["mkt_ret_20"] = _ret(m["tsx"], 20) / 0.04
    out["vix_z"] = _rolling_z(m["vix"])
    out["vix_chg_5"] = _rolling_z(m["vix"].diff(5))
    out["usdcad_ret_20"] = _rolling_z(_ret(m["usdcad"], 20))
    out["oil_ret_20"] = _rolling_z(_ret(m["oil"], 20))
    return out.clip(-4, 4)


def forward_return(close: pd.DataFrame) -> pd.DataFrame:
    """Next-day close-to-close return (the label). NaN for the last day."""
    close = close.ffill(limit=3)
    return close.shift(-1) / close - 1.0


def build_panel(close, volume, market) -> pd.DataFrame:
    """Long table indexed by (date, ticker): feature columns + fwd_ret + fwd_excess."""
    close, volume = close.sort_index(), volume.reindex_like(close)
    feats = stock_features(close, volume)
    mkt = market_features(market, close.index)
    fwd = forward_return(close)

    dates, tickers = close.index, close.columns
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    cols = {k: v.to_numpy().ravel() for k, v in feats.items()}
    for name in mkt.columns:
        cols[name] = np.repeat(mkt[name].to_numpy(), len(tickers))
    cols["fwd_ret"] = fwd.to_numpy().ravel()
    cols["fwd_excess"] = fwd.sub(fwd.mean(axis=1), axis=0).to_numpy().ravel()
    panel = pd.DataFrame(cols, index=idx)

    feature_cols = list(feats) + list(mkt.columns)
    # keep rows where every feature exists; the label may be NaN only on the final day
    panel = panel.dropna(subset=feature_cols)
    return panel


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in ("fwd_ret", "fwd_excess")]
