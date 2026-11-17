"""Which of the removed S&P 500 names without a SEC CIK still have Yahoo price history? (T3c, read-only for the panel)

    python -m experiments.probe_prices --root data_fundamentals --workers 4

Downloads into <root>/prices_extra (NOT the panel's price cache), then writes <root>/probe_prices.csv and prints a summary:
for each missing name, whether Yahoo returned history, and whether it covers the years the stock was in the index.
Nothing else changes. Needs yfinance + internet (run it on your PC).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


def summarize(tickers, cache_dir, intervals: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker: Yahoo status, first/last price date, and the share of its index years that the history covers."""
    cache = Path(cache_dir)
    iv = intervals.assign(start=pd.to_datetime(intervals["start"]), end=pd.to_datetime(intervals["end"]))
    rows = []
    for t in tickers:
        f = cache / f"{t}.pkl"
        mem = iv[iv.symbol == t]
        today = pd.Timestamp.today().normalize()
        lo = mem.start.fillna(pd.Timestamp("2004-01-01")).clip(lower=pd.Timestamp("2004-01-01"))
        hi = mem.end.fillna(today)
        years = float(((hi - lo).dt.days.clip(lower=0) / 365.25).sum())
        if not f.exists():
            rows.append((t, "no history" if (cache / f"{t}.empty").exists() else "not downloaded", None, None, 0, years, 0.0))
            continue
        h = pd.read_pickle(f)
        first, last = h.index.min(), h.index.max()
        cov = 0.0
        for a, b in zip(lo, hi):
            cov += max((min(b, last) - max(a, first)).days, 0) / 365.25
        rows.append((t, "has history", first, last, len(h), years, cov))
    out = pd.DataFrame(rows, columns=["ticker", "yahoo", "first", "last", "rows", "member_years", "years_covered"])
    out["usable"] = (out.yahoo == "has history") & (out.years_covered >= 0.8 * out.member_years) & (out.rows >= 252)
    return out


def main():
    from env import data_us
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_fundamentals")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    root = Path(a.root)
    gaps = pd.read_csv(root / "universe_gaps.csv")
    tick = sorted(gaps.loc[gaps.status.astype(str).str.startswith("no SEC CIK"), "symbol"])
    iv = pd.read_csv(root / "universe" / "intervals.csv")
    cache = root / "prices_extra"
    print(f"probing Yahoo for {len(tick)} removed names without a CIK ({a.workers} workers) ...", flush=True)
    res = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for k, (t, r) in enumerate(zip(tick, ex.map(lambda t: data_us.download_one(t, cache), tick)), 1):
            res[r] = res.get(r, 0) + 1
            if k % 50 == 0 or k == len(tick):
                print(f"  {k}/{len(tick)} {res}", flush=True)
    df = summarize(tick, cache, iv)
    df.to_csv(root / "probe_prices.csv", index=False)
    tot = df.member_years.sum()
    print(f"\nYahoo has some history for {int((df.yahoo == 'has history').sum())} of {len(df)} names; "
          f"{int(df.usable.sum())} cover >=80% of their index years ({df.loc[df.usable, 'member_years'].sum():.0f} of {tot:.0f} member-years)")
    print("\nlargest names Yahoo has NO usable history for:")
    print(df[~df.usable].sort_values("member_years", ascending=False).head(20)[["ticker", "yahoo", "member_years", "years_covered"]]
          .round(1).to_string(index=False))
    print(f"\nsaved {root / 'probe_prices.csv'}")


if __name__ == "__main__":
    main()
