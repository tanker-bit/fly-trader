"""Price-only panel (T6a): no fundamentals, so no SEC CIK is needed and every member with usable prices gets in.

    python -m experiments.build_dataset_price --root data_fundamentals

--fundamentals builds the HYBRID panel: the 11 SEC fundamentals stay wherever they exist (they are NOT dropped), names and
years without them stay in as price-only rows, flagged by a `fund_known` feature.

Why: the fundamentals panel keeps a stock-day only with >= 7 of 11 fundamentals and only for tickers with a SEC CIK, which
thins the early years (2010: about half the index). This panel uses only the 13 price/volume features, so all index
members with price history are in, from 2005 on, including removed names Yahoo still has (verified by probe_prices).

Reads  <root>/prices/*.pkl (panel cache) and <root>/prices_extra/*.pkl + probe_prices.csv (only names marked usable),
       <root>/universe/intervals.csv (point-in-time membership).
Writes <root>/panel_price.parquet (or panel_hybrid.parquet with --fundamentals) (same format as panel.parquet, so the league reads it with --panel) and prints how many
members are in the panel per year. The existing panel.parquet is not touched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from env import data_us, universe_us as uni
from env.features import build_panel


def load_all_prices(root: Path, start="2000-01-01", full=False):
    """Prices from the panel cache plus the probed extras that passed the usable check. -> (adj, vol, sources);
    with full=True -> (adj, close_split, vol, splits, sources) (the fundamentals alignment needs the split-adjusted close)."""
    root = Path(root)
    main = sorted(p.stem for p in (root / "prices").glob("*.pkl"))
    adj, cs, vol, spl = data_us.load_prices(main, root / "prices", start)
    extra = []
    pf, ed = root / "probe_prices.csv", root / "prices_extra"
    if pf.exists() and ed.exists():
        pr = pd.read_csv(pf)
        extra = [t for t in pr.loc[pr.usable, "ticker"] if t not in adj.columns]
        if extra:
            a2, c2, v2, s2 = data_us.load_prices(extra, ed, start)
            adj, vol = pd.concat([adj, a2], axis=1).sort_index(), pd.concat([vol, v2], axis=1).sort_index()
            cs, spl = pd.concat([cs, c2], axis=1).sort_index(), {**s2, **spl}
    src = dict(main=len(main), extra=len(extra))
    return (adj, cs, vol, spl, src) if full else (adj, vol, src)


def raw_fundamentals(root: Path, ciks: dict, adj, close_split, splits, lag_days=1) -> dict:
    """Point-in-time raw metrics for every ticker that has a CIK and SEC facts; all others stay NaN (price-only rows)."""
    from env import fundamentals as fnd
    dates, tick = adj.index, list(adj.columns)
    raw = {m: pd.DataFrame(np.nan, index=dates, columns=tick) for m in fnd.METRICS}
    n = 0
    for t in tick:
        if t not in ciks:
            continue
        ev = fnd.events_for_cik(Path(root) / "sec", ciks[t], lag_days)
        if ev is None:
            continue
        n += 1
        for m, v in fnd.align_metrics(ev, dates, close_split[t], splits.get(t)).items():
            raw[m][t] = v
    print(f"  fundamentals aligned for {n} of {len(tick)} tickers; the rest are price-only")
    return raw


def build_price_panel(adj, vol, intervals: pd.DataFrame, raw=None) -> pd.DataFrame:
    """raw=None: price-only panel. raw=<metric frames>: HYBRID panel: all members kept; fundamentals real where known,
    0 (mid rank) where not, and a `fund_known` feature says which."""
    tick = [t for t in adj.columns if t in set(intervals.symbol)]
    adj, vol = adj[tick], vol[tick]
    member = uni.membership_matrix(intervals, adj.index, tick)
    if raw is None:
        return build_panel(adj, vol, None, fundamentals=None, member=member, horizon21=True)
    raw = {k: v[tick] for k, v in raw.items()}
    return build_panel(adj, vol, None, fundamentals=raw, member=member, min_fund=0, horizon21=True, fund_flag=True)


def yearly(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.reset_index()
    g = d.groupby(d["date"].dt.year).agg(rows=("ticker", "size"), stocks=("ticker", "nunique"))
    g["avg_per_day"] = (g.rows / d.groupby(d["date"].dt.year)["date"].nunique()).round(0)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_fundamentals")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--fundamentals", action="store_true",
                    help="HYBRID panel: keep the 11 fundamentals wherever SEC data exists (needs the sec/ cache), price-only elsewhere")
    ap.add_argument("--lag-days", type=int, default=1)
    a = ap.parse_args()
    root = Path(a.root)
    iv = pd.read_csv(root / "universe" / "intervals.csv", parse_dates=["start", "end"])
    print("loading prices ...", flush=True)
    adj, cs, vol, spl, src = load_all_prices(root, a.start, full=True)
    print(f"  {src['main']} tickers from the panel cache + {src['extra']} usable extras = {adj.shape[1]} columns")
    raw = None
    if a.fundamentals:
        cur = pd.read_csv(root / "universe" / "constituents.csv", parse_dates=["date_added"])
        ciks = uni.cik_map(cur, iv, root / "universe" / "cik_overrides.csv")
        raw = raw_fundamentals(root, ciks, adj, cs, spl, a.lag_days)
    panel = build_price_panel(adj, vol, iv, raw)
    out = root / ("panel_hybrid.parquet" if a.fundamentals else "panel_price.parquet")
    panel.to_parquet(out)
    print("\nstocks per day in the panel (index members with 260+ days of history):")
    print(yearly(panel).to_string())
    if a.fundamentals:
        yk = panel["fund_known"].groupby(panel.index.get_level_values("date").year).apply(lambda x: float((x > -1).mean()))
        print("\nshare of rows with at least one real fundamental, by year:")
        print((yk * 100).round(0).astype(int).astype(str).add("%").to_string())
    print(f"\nwrote {out}: {len(panel):,} rows, {panel.index.get_level_values('date').nunique()} dates, "
          f"{panel.index.get_level_values('ticker').nunique()} tickers")


if __name__ == "__main__":
    main()
