"""Fresh daily prices for the live run, kept in a SEPARATE folder (default data_live/prices).

The research caches (data_fundamentals/prices, prices_etf) are never rewritten: the training panel and its exactness tests depend on
them. refresh_prices() downloads each symbol's full Yahoo history again (same columns as env.data_us: Close, Adj Close, Volume,
Stock Splits) into the live folder, replacing the old file only when the new download worked. overlay() then puts the live columns
over the research ones for the live run. Network: Yahoo only, no personal data is sent.
"""
from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd

KEEP = ["Close", "Adj Close", "Volume", "Stock Splits"]
CLOSED_AFTER = (16, 15)        # New York time after which today's bar counts as final (the close is 16:00; half days close earlier)


def now_new_york():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York"))


def drop_unfinished_bar(h: pd.DataFrame, now) -> pd.DataFrame:
    """Yahoo returns today's bar while the market is still open (a partial volume and a moving 'close'). Drop it until CLOSED_AFTER.
    Seen on 2026-11-27 at 12:47 New York time: AAPL's bar for that day had a quarter of a normal day's volume."""
    today = pd.Timestamp(now.date())
    if (now.hour, now.minute) < CLOSED_AFTER and len(h) and h.index[-1] >= today:
        return h[h.index < today]
    return h


def yahoo_history(symbol: str) -> pd.DataFrame | None:
    import yfinance as yf
    h = yf.Ticker(symbol).history(period="max", auto_adjust=False, actions=True)
    if h is None or h.empty or "Adj Close" not in h.columns:
        return None
    h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
    if "Stock Splits" not in h.columns:
        h["Stock Splits"] = 0.0
    return h[KEEP]


def refresh_one(symbol: str, live_dir, fetch=yahoo_history, tries: int = 3, sleep=None, now=None) -> str:
    """-> 'ok' | 'empty' | 'error'. The old live file stays when the download fails. A bar for today is kept only after the close."""
    import time
    sleep = sleep or time.sleep
    now = now or now_new_york()
    live_dir = Path(live_dir)
    live_dir.mkdir(parents=True, exist_ok=True)
    for n in range(tries):
        try:
            h = fetch(symbol)
            if h is None or h.empty:
                return "empty"
            h = h[~h.index.duplicated(keep="last")].sort_index()
            h = h[h["Close"].notna() & (h["Close"] > 0)]               # Yahoo sometimes adds today's bar with no close yet
            h = drop_unfinished_bar(h, now)
            if h.empty:
                return "empty"
            tmp = live_dir / f"{symbol}.pkl.tmp"
            h[KEEP].to_pickle(tmp)
            os.replace(tmp, live_dir / f"{symbol}.pkl")
            return "ok"
        except Exception:
            sleep(2 ** n)
    return "error"


def refresh_prices(symbols, live_dir, fetch=yahoo_history, workers: int = 4, log=print, now=None) -> dict:
    symbols = sorted(set(symbols))
    now = now or now_new_york()
    out = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for s, r in zip(symbols, ex.map(lambda s: refresh_one(s, live_dir, fetch, now=now), symbols)):
            out[s] = r
    log(f"price refresh: {dict(Counter(out.values()))} for {len(symbols)} symbols into {live_dir}")
    return out


def load_frames(symbols, folder):
    """(adj, close_split, volume, splits) from one folder; symbols without a file are left out."""
    from env.data_us import load_prices
    return load_prices([s for s in symbols if (Path(folder) / f"{s}.pkl").exists()], folder)


def overlay(inp, live_dir):
    """A copy of tournament.live_features.Inputs whose price columns come from live_dir wherever a live file exists."""
    live_dir = Path(live_dir)
    syms = sorted(p.stem for p in live_dir.glob("*.pkl")) if live_dir.exists() else []
    if not syms:
        return inp
    adj, cs, vol, spl = load_frames(syms, live_dir)
    idx = inp.adj.index.union(adj.index)

    def put(base, new):
        b = base.reindex(idx)
        for c in new.columns:
            b[c] = new[c].reindex(idx)
        return b

    return replace(inp, adj=put(inp.adj, adj), close_split=put(inp.close_split, cs), volume=put(inp.volume, vol),
                   splits={**inp.splits, **spl})
