"""U2: download daily prices for the US universe (resumable) and work out, day by day, which stocks were liquid enough to trade.

    python -m experiments.fetch_us_prices --out data_us --limit 50        # a small test run first
    python -m experiments.fetch_us_prices --out data_us                   # everything (rerun it after a crash or rate limit: it resumes)

Reads <out>/universe_us.csv (from build_us_universe), keeps the stocks (not the ETFs), and downloads each one's full Yahoo history to
<out>/prices/<SYMBOL>.pkl with the same downloader and file format as the S&P 500 runs (env.data_us.download_one). A symbol Yahoo does not
know gets an empty marker file and is skipped next time. Then it writes
  <out>/tradable.parquet        dates x symbols, True on days a stock passes the liquidity rule below (point-in-time, no look-ahead)
  <out>/liquidity_summary.csv   one row per symbol: first/last date, days of history, last price, recent median dollar volume, tradable days
  <out>/price_status.csv        what the download did for each symbol (ok / cached / empty / error)

Liquidity rule (a stock may be bought or held on a day only if): the REAL traded price that day (split-adjusted close times the splits that came
after) is at least --min-price (default $2) and the median of the last --window (default 60) days of dollar volume (price x volume, which a split
does not change) is at least --min-dollar-volume (default $1,000,000). It needs at least 2/3 of the window to exist. Yahoo history only
exists for stocks that trade today, so delisted companies are missing (survivorship bias, as for the universe list).
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def fetch_all(symbols, cache_dir, workers: int = 4, downloader=None, every: int = 100) -> dict:
    """-> {symbol: 'cached' | 'ok' | 'empty' | 'error'}. `downloader(symbol, cache_dir)` defaults to env.data_us.download_one."""
    if downloader is None:
        from env import data_us
        downloader = data_us.download_one
    symbols = list(symbols)
    out = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for k, (s, r) in enumerate(zip(symbols, ex.map(lambda s: downloader(s, cache_dir), symbols)), 1):
            out[s] = r
            if k % every == 0 or k == len(symbols):
                print(f"  {k}/{len(symbols)}  {dict(Counter(out.values()))}", flush=True)
    return out


def real_price_and_dollar_volume(h: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """h: one cached Yahoo history (Close = split-adjusted, Volume, Stock Splits). -> (price as actually quoted that day, dollar volume)."""
    h = h[~h.index.duplicated(keep="last")].sort_index()
    spl = h["Stock Splits"].where(h["Stock Splits"] > 0, 1.0).astype(float)
    after = spl[::-1].cumprod()[::-1].shift(-1).fillna(1.0)          # product of the splits that happened AFTER each day
    return h["Close"] * after, h["Close"] * h["Volume"]


def liquidity(prices_dir, symbols, start="2000-01-01", min_price=2.0, min_dollar_volume=1e6, window=60):
    """-> (tradable bool frame dates x symbols, summary frame). Only symbols with a cached file are included."""
    prices_dir = Path(prices_dir)
    flags, rows = {}, []
    need = int(round(window * 2 / 3))
    for s in symbols:
        f = prices_dir / f"{s}.pkl"
        if not f.exists():
            continue
        h = pd.read_pickle(f)
        h = h[h.index >= pd.Timestamp(start)]
        h = h[h["Close"].notna() & (h["Close"] > 0)].copy()                  # Yahoo sometimes adds today's bar with no close yet
        h["Volume"] = h["Volume"].fillna(0.0)
        if h.empty:
            continue
        px, dv = real_price_and_dollar_volume(h)
        med = dv.rolling(window, min_periods=need).median()
        ok = (px >= min_price) & (med >= min_dollar_volume)
        flags[s] = ok
        rows.append(dict(symbol=s, first=px.index[0].date(), last=px.index[-1].date(), days=len(px), last_price=float(px.iloc[-1]),
                         median_dollar_volume=float(med.iloc[-1]) if np.isfinite(med.iloc[-1]) else np.nan, tradable_days=int(ok.sum())))
    tradable = pd.DataFrame(flags).sort_index().fillna(False).astype(bool) if flags else pd.DataFrame()
    return tradable, pd.DataFrame(rows)


def names_by_year(tradable: pd.DataFrame, years=(2005, 2010, 2015, 2020, 2024)) -> str:
    L = ["tradable stocks on the last trading day of:"]
    for y in years:
        d = tradable.index[tradable.index.year == y]
        if len(d):
            L.append(f"  {y}: {int(tradable.loc[d[-1]].sum()):,}")
    L.append(f"  latest ({tradable.index[-1].date()}): {int(tradable.iloc[-1].sum()):,}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_us")
    ap.add_argument("--limit", type=int, default=None, help="only the first N symbols (a test run)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--min-price", type=float, default=2.0)
    ap.add_argument("--min-dollar-volume", type=float, default=1e6)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--skip-download", action="store_true", help="only recompute the liquidity files from the cached prices")
    a = ap.parse_args()
    out = Path(a.out)
    uni = pd.read_csv(out / "universe_us.csv", keep_default_na=False)
    symbols = [s for s in uni.loc[~uni["etf"].astype(str).str.lower().eq("true"), "symbol"]]
    if a.limit:
        symbols = symbols[: a.limit]
    if not a.skip_download:
        print(f"downloading {len(symbols):,} stocks (resumable) ...", flush=True)
        status = fetch_all(symbols, out / "prices", a.workers)
        pd.Series(status, name="status").rename_axis("symbol").to_csv(out / "price_status.csv")
        print(f"done: {dict(Counter(status.values()))}. Rerun the same command to retry the errors.")
    tradable, summ = liquidity(out / "prices", symbols, a.start, a.min_price, a.min_dollar_volume, a.window)
    if tradable.empty:
        print("no prices cached yet")
        return
    tradable.to_parquet(out / "tradable.parquet")
    summ.to_csv(out / "liquidity_summary.csv", index=False)
    print(f"\n{len(summ):,} stocks have price history; {int((summ.tradable_days > 0).sum()):,} were tradable on at least one day")
    print(names_by_year(tradable))
    stale = summ[pd.to_datetime(summ["last"]) < tradable.index[-1] - pd.Timedelta(days=10)]
    print(f"{len(stale)} symbols stopped updating more than 10 days before the latest date (halted or delisted)")


if __name__ == "__main__":
    main()
