"""How survivorship-biased is the panel? Compares who was in the S&P 500 over time with who actually has prices / is in the panel.

    python -m experiments.universe_coverage --root data_fundamentals

Reads (nothing is downloaded, nothing is changed):
  <root>/universe/intervals.csv   point-in-time membership intervals (symbol,start,end)
  <root>/universe/constituents.csv, cik_overrides.csv   who has a SEC CIK
  <root>/prices/<T>.pkl | <T>.empty   Yahoo price cache (.empty = Yahoo had nothing)
  <root>/panel.parquet            the finished panel (optional)
Writes <root>/universe_coverage.txt and <root>/universe_gaps.csv (every ever-member missing from the panel, with the reason).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from env import universe_us as uni


def _status(root: Path, t: str, ciks: dict, panel_cols: set | None) -> str:
    pr = root / "prices"
    if panel_cols is not None and t in panel_cols:
        return "in panel"
    if t not in ciks:
        return "no SEC CIK (dropped before prices were tried)"
    if (pr / f"{t}.pkl").exists():
        return "has prices + CIK but not in panel" if panel_cols is not None else "has prices"
    if (pr / f"{t}.empty").exists():
        return "Yahoo has no prices"
    return "prices not downloaded"


def coverage(root, today=None):
    root = Path(root)
    udir = root / "universe"
    iv = pd.read_csv(udir / "intervals.csv", parse_dates=["start", "end"])
    cur = pd.read_csv(udir / "constituents.csv", parse_dates=["date_added"])
    ciks = uni.cik_map(cur, iv, udir / "cik_overrides.csv")
    panel_cols = None
    pf, today = root / "panel.parquet", pd.Timestamp(today or pd.Timestamp.today().normalize())
    first = pd.Timestamp("2004-01-01")
    if pf.exists():
        p = pd.read_parquet(pf)
        panel_cols = set(p.index.get_level_values("ticker").unique())
        first = pd.Timestamp(p.index.get_level_values("date").min())
    iv = iv.copy()
    iv["lo"] = iv["start"].fillna(first).clip(lower=first)
    iv["hi"] = iv["end"].fillna(today)
    iv = iv[iv["hi"] > iv["lo"]]
    iv["years"] = (iv["hi"] - iv["lo"]).dt.days / 365.25
    per = iv.groupby("symbol").agg(years=("years", "sum"), removed=("end", lambda s: s.notna().any())).reset_index()
    per["status"] = [_status(root, t, ciks, panel_cols) for t in per.symbol]
    per["kind"] = per.removed.map({True: "removed", False: "current"})
    by_year = []
    in_panel = set(per.symbol[per.status == "in panel"]) if panel_cols is not None else set(per.symbol[per.status.str.startswith("has")])
    for y in range(first.year, today.year + 1):
        d = pd.Timestamp(f"{y}-06-30")
        if d < first or d > today:
            continue
        mem = set(iv.symbol[(iv.lo <= d) & (iv.hi > d)])
        by_year.append((y, len(mem), len(mem & in_panel)))
    yr = pd.DataFrame(by_year, columns=["year", "members", "covered"])
    yr["pct"] = (yr.covered / yr.members.where(yr.members > 0)).fillna(0.0)
    return per, yr


def report(per, yr) -> str:
    L = ["S&P 500 coverage: who was a member vs who we can actually trade", ""]
    t = per.groupby(["kind", "status"]).agg(tickers=("symbol", "count"), member_years=("years", "sum"))
    L += [t.round(0).to_string(), ""]
    tot, ok = per.years.sum(), per.years[per.status.isin(["in panel", "has prices"])].sum()
    L += [f"member-years covered: {ok / tot:.0%} of {tot:,.0f}  (the missing {1 - ok / tot:.0%} are stocks that left the index: "
          f"mostly the losers, so this inflates results)", "", "members on 30 June each year, and how many are in the data:",
          yr.assign(pct=(yr.pct * 100).round(0).astype(int).astype(str) + "%").to_string(index=False), "",
          "largest gaps (removed names missing from the data, by years of membership):"]
    gap = per[(per.kind == "removed") & ~per.status.isin(["in panel", "has prices"])].sort_values("years", ascending=False)
    L += [gap.head(25)[["symbol", "years", "status"]].round(1).to_string(index=False)]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_fundamentals")
    a = ap.parse_args()
    per, yr = coverage(a.root)
    text = report(per, yr)
    root = Path(a.root)
    (root / "universe_coverage.txt").write_text(text)
    per[~per.status.isin(["in panel", "has prices"])].sort_values("years", ascending=False).to_csv(root / "universe_gaps.csv", index=False)
    print(text)
    print(f"\nsaved {root / 'universe_coverage.txt'} and {root / 'universe_gaps.csv'}")


if __name__ == "__main__":
    main()
