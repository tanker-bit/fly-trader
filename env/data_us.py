"""Yahoo Finance price download for the US universe, one cached file per ticker (resumable).

Per ticker we keep four columns from Ticker.history(auto_adjust=False, actions=True):
  'Adj Close'   split- and dividend-adjusted: used for returns, features and labels (total return)
  'Close'       split-adjusted only: used with the split factors to turn SEC share counts into market value
  'Volume', 'Stock Splits'
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

KEEP = ["Close", "Adj Close", "Volume", "Stock Splits"]


def download_one(ticker: str, cache_dir, retries: int = 3, pause: float = 0.3) -> str:
    """-> 'cached' | 'ok' | 'empty' | 'error'. Network code: needs yfinance and internet."""
    import yfinance as yf
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f, empty = cache_dir / f"{ticker}.pkl", cache_dir / f"{ticker}.empty"
    if f.exists():
        return "cached"
    if empty.exists():
        return "empty"
    for attempt in range(retries):
        try:
            h = yf.Ticker(ticker).history(period="max", auto_adjust=False, actions=True)
            time.sleep(pause)
            if h is None or h.empty or "Adj Close" not in h.columns:
                empty.write_text("empty")
                return "empty"
            h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
            if "Stock Splits" not in h.columns:
                h["Stock Splits"] = 0.0
            h[KEEP].to_pickle(f)
            return "ok"
        except Exception:
            time.sleep(2 ** attempt)
    return "error"


def load_prices(tickers, cache_dir, start: str = "2000-01-01"):
    """-> (adj_close, close_split, volume, splits_by_ticker) for the tickers that have a cached file."""
    cache_dir = Path(cache_dir)
    adj, cs, vol, spl = {}, {}, {}, {}
    for t in tickers:
        f = cache_dir / f"{t}.pkl"
        if not f.exists():
            continue
        h = pd.read_pickle(f)
        h = h[h.index >= pd.Timestamp(start)]
        h = h[~h.index.duplicated(keep="last")]
        adj[t], cs[t], vol[t] = h["Adj Close"], h["Close"], h["Volume"]
        s = h["Stock Splits"]
        spl[t] = s[s > 0]
    frame = lambda d: pd.DataFrame(d).sort_index()
    return frame(adj), frame(cs), frame(vol), spl
