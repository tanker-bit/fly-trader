"""Find SEC CIKs for removed S&P 500 names so they stop being dropped (T3b).

    python -m experiments.resolve_ciks --root data_fundamentals --user-agent "Your Name you@example.com"           # dry run
    python -m experiments.resolve_ciks --root data_fundamentals --user-agent "Your Name you@example.com" --apply   # write overrides

Reads universe_gaps.csv (from experiments.universe_coverage) and SEC's company_tickers.json (companies that still file today).
A removed ticker is matched only if (a) it is in that file and (b) it left the index within --max-age years, because old
tickers get reused by other companies and a wrong match would attach the wrong financials. Matches are listed with the SEC
company name so you can eyeball them; --apply appends them to <root>/universe/cik_overrides.csv (ticker,cik). Companies that
no longer file (acquired, bankrupt) are NOT in that file; they stay missing and need a different data source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def propose(gaps: pd.DataFrame, intervals: pd.DataFrame, sec_json: dict, today, max_age: float | None = 6.0) -> pd.DataFrame:
    """gaps: symbol,years,status. intervals: symbol,start,end. sec_json: SEC company_tickers.json content."""
    sec = {}
    for v in sec_json.values():
        sec.setdefault(str(v["ticker"]).upper(), (int(v["cik_str"]), v["title"]))
    end = intervals.assign(end=pd.to_datetime(intervals["end"])).groupby("symbol")["end"].max()
    today = pd.Timestamp(today)
    rows = []
    for r in gaps.itertuples():
        if not str(r.status).startswith("no SEC CIK"):
            continue
        hit = sec.get(r.symbol.upper()) or sec.get(r.symbol.upper().replace(".", "-"))
        if hit is None:
            continue
        e = end.get(r.symbol, pd.NaT)
        age = (today - e).days / 365.25 if pd.notna(e) else None
        if max_age is not None and (age is None or age > max_age):
            continue
        rows.append((r.symbol, hit[0], hit[1], e, round(age, 1) if age is not None else None, round(float(r.years), 1)))
    return pd.DataFrame(rows, columns=["ticker", "cik", "sec_name", "left_index", "years_since_left", "member_years"])


def apply_overrides(path: Path, new: pd.DataFrame) -> int:
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["ticker", "cik"])
    have = set(old["ticker"].astype(str).str.upper())
    add = new[~new["ticker"].str.upper().isin(have)][["ticker", "cik"]]
    if len(add):
        pd.concat([old[["ticker", "cik"]], add]).to_csv(path, index=False)
    return len(add)


def main():
    from env import universe_us as uni
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_fundamentals")
    ap.add_argument("--user-agent", required=True)
    ap.add_argument("--max-age", type=float, default=6.0, help="only match names that left the index within this many years (0 = no limit)")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    gaps = pd.read_csv(root / "universe_gaps.csv")
    iv = pd.read_csv(root / "universe" / "intervals.csv", parse_dates=["start", "end"])
    print("downloading SEC company_tickers.json ...", flush=True)
    sec = json.loads(uni.fetch_text(SEC_TICKERS_URL, a.user_agent))
    prop = propose(gaps, iv, sec, pd.Timestamp.today().normalize(), a.max_age or None)
    n_gap = int(gaps.status.astype(str).str.startswith("no SEC CIK").sum())
    print(f"{n_gap} removed names without a CIK; {len(prop)} matched a company that still files with the SEC "
          f"({prop.member_years.sum():.0f} member-years)\n")
    pd.set_option("display.width", 200, "display.max_rows", 500)
    print(prop.to_string(index=False))
    prop.to_csv(root / "cik_proposed.csv", index=False)
    if a.apply:
        n = apply_overrides(root / "universe" / "cik_overrides.csv", prop)
        print(f"\nadded {n} rows to {root / 'universe' / 'cik_overrides.csv'}")
    else:
        print(f"\ndry run: nothing written except {root / 'cik_proposed.csv'}. Check the names, then rerun with --apply.")


if __name__ == "__main__":
    main()
