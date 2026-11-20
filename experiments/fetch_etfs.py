"""Download the approved long-only ETFs from Yahoo (adjusted close) so the accounts can hold them.

    python -m experiments.fetch_etfs --root data_fundamentals --workers 4

Caches one file per ETF in <root>/prices_etf (re-running only fetches what is missing), then prints and saves <root>/etf_summary.csv:
first and last date and number of days per ETF. Nothing else changes. Needs yfinance and internet (run it on your PC).
The approved list is accounts.instruments.APPROVED_ETFS: plain unleveraged long-only funds; inverse, leveraged and volatility products
are never included.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from accounts.instruments import APPROVED_ETFS, FORBIDDEN_ETFS
from env import data_us


def fetch_all(root, tickers=APPROVED_ETFS, workers: int = 4, downloader=None) -> dict:
    """-> {ticker: 'cached' | 'ok' | 'empty' | 'error'}. `downloader(ticker, cache_dir)` defaults to env.data_us.download_one."""
    cache = Path(root) / "prices_etf"
    down = downloader or data_us.download_one
    todo = [t for t in tickers if t not in FORBIDDEN_ETFS]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return dict(zip(todo, ex.map(lambda t: down(t, cache), todo)))


def summarize(root, status: dict) -> pd.DataFrame:
    rows = []
    for t, st in status.items():
        f = Path(root) / "prices_etf" / f"{t}.pkl"
        if f.exists():
            h = pd.read_pickle(f)
            rows.append(dict(ticker=t, status=st, first=h.index.min().date(), last=h.index.max().date(), days=len(h)))
        else:
            rows.append(dict(ticker=t, status=st, first=None, last=None, days=0))
    return pd.DataFrame(rows).set_index("ticker")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data_fundamentals")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    status = fetch_all(a.root, workers=a.workers)
    s = summarize(a.root, status)
    s.to_csv(Path(a.root) / "etf_summary.csv")
    pd.set_option("display.width", 160)
    print(s.to_string())
    bad = s[s.days == 0]
    print(f"\n{len(s) - len(bad)} of {len(s)} ETFs have history" + (f"; none for: {', '.join(bad.index)}" if len(bad) else ""))


if __name__ == "__main__":
    main()
