"""Download the TSX universe + market context, build the feature panel, print diagnostics.

    python -m experiments.build_dataset              # uses cache if present
    python -m experiments.build_dataset --refresh    # re-download everything

Output: data/panel.parquet  (gitignored)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from env.data import load_universe
from env.features import build_panel, feature_columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    close, volume, market = load_universe(start=args.start, refresh=args.refresh)
    print(f"\nstocks with data: {close.shape[1]}   days: {close.shape[0]}   "
          f"{close.index.min().date()} -> {close.index.max().date()}")
    first = close.apply(lambda s: s.first_valid_index())
    print(f"stocks with 10+ years of history: {(first < close.index.max() - __import__('pandas').Timedelta(days=3650)).sum()}")
    print(f"market series: {', '.join(market.columns)}   ({market.index.min().date()} -> {market.index.max().date()})")

    panel = build_panel(close, volume, market)
    feats = feature_columns(panel)
    days = panel.index.get_level_values("date")
    print(f"\npanel rows: {len(panel):,}   features: {len(feats)}   "
          f"{days.min().date()} -> {days.max().date()}")
    print(f"stocks per day: median {panel.groupby('date').size().median():.0f}, "
          f"min {panel.groupby('date').size().min()}")
    print(f"labelled rows (have next-day return): {panel.fwd_ret.notna().sum():,}")

    Path("data").mkdir(exist_ok=True)
    panel.to_parquet("data/panel.parquet")
    print("saved data/panel.parquet")


if __name__ == "__main__":
    main()
