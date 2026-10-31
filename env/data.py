"""Download and cache daily price data (adjusted for splits and dividends).

Raw downloads are cached as parquet in data/raw/ so re-running is instant and offline.
Use refresh=True to download again.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from env.universe import MARKET_TICKERS, TSX_STOCKS

RAW = Path("data/raw")
CHUNK = 15  # tickers per request: smaller batches fail less often


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    df.index.name = "date"
    return df[~df.index.duplicated(keep="last")].sort_index()


def _download(tickers: list[str], start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    closes, volumes = [], []
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i : i + CHUNK]
        raw = yf.download(chunk, start=start, auto_adjust=True, progress=False, threads=True)
        if raw is None or raw.empty:
            continue
        close, vol = raw["Close"], raw["Volume"]
        if isinstance(close, pd.Series):  # single ticker comes back flat
            close, vol = close.to_frame(chunk[0]), vol.to_frame(chunk[0])
        closes.append(close)
        volumes.append(vol)
        print(f"  downloaded {min(i + CHUNK, len(tickers))}/{len(tickers)}", flush=True)
    if not closes:
        raise RuntimeError("Nothing downloaded. Check your internet connection.")
    close = _clean_index(pd.concat(closes, axis=1))
    volume = _clean_index(pd.concat(volumes, axis=1))
    return close.dropna(axis=1, how="all"), volume.dropna(axis=1, how="all")


def load_universe(start: str = "2000-01-01", refresh: bool = False):
    """Return (close, volume, market): DataFrames indexed by date.

    close / volume have one column per stock; market has one column per MARKET_TICKERS key.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    files = {k: RAW / f"{k}.parquet" for k in ("close", "volume", "market")}
    if not refresh and all(f.exists() for f in files.values()):
        return tuple(pd.read_parquet(files[k]) for k in ("close", "volume", "market"))

    print(f"Downloading {len(TSX_STOCKS)} stocks from {start} ...")
    close, volume = _download(TSX_STOCKS, start)
    missing = sorted(set(TSX_STOCKS) - set(close.columns))
    if missing:
        print(f"WARNING: no data for {len(missing)} tickers (skipped): {', '.join(missing)}")

    print("Downloading market context ...")
    mkt_close, _ = _download(list(MARKET_TICKERS.values()), start)
    inv = {v: k for k, v in MARKET_TICKERS.items()}
    market = mkt_close.rename(columns=inv)

    close.to_parquet(files["close"])
    volume.to_parquet(files["volume"])
    market.to_parquet(files["market"])
    return close, volume, market
