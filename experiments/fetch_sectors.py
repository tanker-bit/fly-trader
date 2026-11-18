"""Download the GICS sector of every current S&P 500 member from Wikipedia and save universe/sectors.csv.

    python -m experiments.fetch_sectors --root data_fundamentals --user-agent "Your Name you@example.com"

Only this one file is written (symbol, sector, sub_industry). The membership files (constituents.csv, changes, intervals.csv)
are NOT touched, so the panel and every saved league stay valid. Use it as:  --sectors data_fundamentals/universe/sectors.csv
Removed companies have no sector on that page; the sector-neutral label puts them in an 'unknown' group.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from env import universe_us as uni


def sectors_from_html(html: str):
    cur, _ = uni.parse_wikipedia(html)
    out = cur[["symbol", "sector", "sub_industry"]]
    return out[out["sector"].astype(str).str.strip() != ""].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_fundamentals")
    ap.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT", ""))
    a = ap.parse_args()
    if not a.user_agent:
        raise SystemExit('give --user-agent "Your Name you@example.com" (Wikipedia asks for a contact string)')
    df = sectors_from_html(uni.fetch_text(uni.WIKI_URL, a.user_agent))
    out = Path(a.root) / "universe" / "sectors.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} current members with a sector -> {out}\n")
    print(df.groupby("sector").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
