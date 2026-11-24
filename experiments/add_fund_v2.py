"""Add the richer SEC fundamentals (9 new features, ranked WITHIN each sector) to a copy of the hybrid panel.

    python -m experiments.add_fund_v2 --root data_fundamentals --sectors data_fundamentals/universe/sectors.csv

Reads  <root>/panel_hybrid.parquet (never changed), the SEC cache <root>/sec/raw (no download: it only re-reads the files already
there; the per-company event tables are rebuilt in the new format the first time) and the price cache (for market values).
Writes <root>/panel_v2.parquet = panel_hybrid + 9 columns `s_<metric>` + `fund2_known`, and <root>/panel_v2_coverage.txt.

Each new column is the stock's rank among the stocks of the SAME sector that day, in [-1, 1]; unknown values (and sectors with fewer
than 6 stocks that day) are 0, the middle, exactly as for the old fundamentals; `fund2_known` in [-1, 1] says how many of the 9 are real.
Features (all point-in-time, filed + 1 day): gp_to_assets, op_margin, margin_change, asset_growth, debt_to_assets, interest_cover,
current_ratio, sue (standardised unexpected EPS from the company's own history), buyback_yield.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from env import fundamentals as fnd, universe_us as uni
from env.features import sector_xs_rank

NEW = fnd.METRICS_V2


def add_v2(panel: pd.DataFrame, raw: dict, sector: dict, min_group: int = 6) -> pd.DataFrame:
    """panel: (date, ticker) rows. raw: metric -> dates x tickers frame (point-in-time). Returns a copy with the new columns."""
    dates = panel.index.get_level_values("date").unique().sort_values()
    tick = sorted(panel.index.get_level_values("ticker").unique())
    di = dates.get_indexer(panel.index.get_level_values("date"))
    ti = pd.Index(tick).get_indexer(panel.index.get_level_values("ticker"))
    member = np.zeros((len(dates), len(tick)), bool)
    member[di, ti] = True                                          # the panel's rows are the index members of that day
    member = pd.DataFrame(member, index=dates, columns=tick)
    out = panel.copy()
    known = np.zeros(len(panel))
    for m in NEW:
        v = raw[m].reindex(index=dates, columns=tick).replace([np.inf, -np.inf], np.nan).where(member)
        r = sector_xs_rank(v, sector, min_group).to_numpy()[di, ti]
        known += np.isfinite(r)
        out[f"s_{m}"] = np.nan_to_num(r, nan=0.0)
    out["fund2_known"] = known / len(NEW) * 2.0 - 1.0
    return out


def coverage(panel: pd.DataFrame) -> str:
    yrs = panel.index.get_level_values("date").year
    L = ["share of panel rows with each new feature known (rank not at the 0 filler), by year", ""]
    cols = [f"s_{m}" for m in NEW]
    head = "year " + " ".join(f"{m[:12]:>12}" for m in NEW)
    L.append(head)
    for y, g in panel.groupby(yrs):
        L.append(f"{y} " + " ".join(f"{(g[c] != 0).mean():>12.0%}" for c in cols))
    return "\n".join(L)


def main():
    from experiments.build_dataset_price import load_all_prices, raw_fundamentals
    from tournament.labels import load_sectors
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data_fundamentals")
    ap.add_argument("--sectors", default="data_fundamentals/universe/sectors.csv")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--lag-days", type=int, default=1)
    ap.add_argument("--min-group", type=int, default=6)
    a = ap.parse_args()
    root = Path(a.root)
    panel = pd.read_parquet(root / "panel_hybrid.parquet")
    iv = pd.read_csv(root / "universe" / "intervals.csv", parse_dates=["start", "end"])
    cur = pd.read_csv(root / "universe" / "constituents.csv", parse_dates=["date_added"])
    ciks = uni.cik_map(cur, iv, root / "universe" / "cik_overrides.csv")
    print("loading prices ...", flush=True)
    adj, cs, vol, spl, src = load_all_prices(root, a.start, full=True)
    print("aligning fundamentals (rebuilds each company's event table once) ...", flush=True)
    raw = raw_fundamentals(root, ciks, adj, cs, spl, a.lag_days, extended=True)
    out = add_v2(panel, raw, load_sectors(a.sectors), a.min_group)
    path = root / "panel_v2.parquet"
    out.to_parquet(path)
    rep = coverage(out)
    (root / "panel_v2_coverage.txt").write_text(rep)
    print(rep)
    print(f"\nwrote {path}: {len(out):,} rows, {out.shape[1]} columns ({len(NEW) + 1} new)")


if __name__ == "__main__":
    main()
