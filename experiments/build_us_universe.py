"""U1: list every NYSE / Nasdaq / NYSE American common stock (and, separately flagged, ETF) that trades today.

    python -m experiments.build_us_universe --out data_us

Downloads the two symbol directories the exchanges publish (nasdaqtrader.com): nasdaqlisted.txt (Nasdaq) and otherlisted.txt (NYSE,
NYSE American, NYSE Arca, Cboe, IEX). Saves the raw files, dated, under <out>/raw/ and writes <out>/universe_us.csv:

    symbol (Yahoo style: BRK.B -> BRK-B), exchange (NASDAQ / NYSE / NYSE American / NYSE Arca / ...), name, etf (True/False), listed_on_raw

Kept as stocks: common shares (and ADRs/class shares) on Nasdaq, NYSE and NYSE American, not test issues, not on the exchanges' deficiency /
delinquency lists. Dropped: warrants, rights, units, preferred and other debt-like lines, NYSE Arca/Cboe/IEX-only lines (mostly ETFs), and any
symbol containing $ or ^. ETFs are listed with etf=True and are NOT stocks here (the approved core ETFs live in accounts/etfs.py).

LIMITS: these files only list what trades TODAY. Companies that were delisted, bought or went bankrupt are not in them, so any backtest on
this list suffers survivorship bias (the S&P 500 runs rebuilt membership history; this one cannot). No fundamentals or prices here.
If the download is blocked, save both files by hand and pass --nasdaq-file / --other-file.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import re
import urllib.request
from pathlib import Path

import pandas as pd

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
EXCHANGES = {"N": "NYSE", "A": "NYSE American", "P": "NYSE Arca", "Z": "Cboe BZX", "V": "IEX"}
STOCK_EXCHANGES = {"NASDAQ", "NYSE", "NYSE American"}
# whole words / patterns that mark a line as something other than ordinary shares (word boundaries: "Community" is not a unit)
NOT_COMMON = re.compile(r"\b(warrants?|rights?|preferred|pref|notes?|debentures?|subordinated|bonds?|etns?)\b|\d%|exchange traded note|"
                        r"when issued|contingent value|trust certificate|depositary shares, each representing", re.I)
UNITS = re.compile(r"\bunits?\b", re.I)
PARTNER_UNITS = re.compile(r"common units|limited partner|ordinary units", re.I)


def _table(text: str) -> pd.DataFrame:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("File Creation Time")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str, keep_default_na=False)


def parse_nasdaq(text: str) -> pd.DataFrame:
    t = _table(text)
    return pd.DataFrame(dict(raw=t["Symbol"], name=t["Security Name"], exchange="NASDAQ", etf=t["ETF"].eq("Y"),
                             test=t["Test Issue"].eq("Y"), status_ok=t["Financial Status"].isin(["N", ""])))


def parse_other(text: str) -> pd.DataFrame:
    t = _table(text)
    return pd.DataFrame(dict(raw=t["ACT Symbol"], name=t["Security Name"], exchange=t["Exchange"].map(EXCHANGES).fillna("other"),
                             etf=t["ETF"].eq("Y"), test=t["Test Issue"].eq("Y"), status_ok=True))


def looks_like_common(name: str) -> bool:
    if NOT_COMMON.search(name):
        return False
    if UNITS.search(name) and not PARTNER_UNITS.search(name):          # SPAC units yes, "Common Units" of a partnership no
        return False
    return True


def build(nasdaq_text: str, other_text: str) -> pd.DataFrame:
    df = pd.concat([parse_nasdaq(nasdaq_text), parse_other(other_text)], ignore_index=True)
    df = df[~df["test"] & df["status_ok"]]
    df = df[~df["raw"].str.contains(r"[\$\^]", regex=True)]
    df["symbol"] = df["raw"].str.strip().str.upper().str.replace(".", "-", regex=False)
    stock = df["exchange"].isin(STOCK_EXCHANGES) & ~df["etf"] & df["name"].map(looks_like_common)
    keep = stock | df["etf"]
    out = df[keep][["symbol", "exchange", "name", "etf", "raw"]].rename(columns={"raw": "listed_on_raw"})
    out = out.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    return out


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (fly-trader research)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_us")
    ap.add_argument("--nasdaq-file", default=None)
    ap.add_argument("--other-file", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    stamp = dt.date.today().isoformat()
    texts = []
    for url, path, name in ((NASDAQ_URL, a.nasdaq_file, "nasdaqlisted"), (OTHER_URL, a.other_file, "otherlisted")):
        text = Path(path).read_text(encoding="utf-8", errors="replace") if path else fetch(url)
        (out / "raw" / f"{name}_{stamp}.txt").write_text(text, encoding="utf-8")
        texts.append(text)
    uni = build(*texts)
    uni.to_csv(out / "universe_us.csv", index=False)
    st = uni[~uni.etf]
    print(f"{len(st):,} stocks and {int(uni.etf.sum()):,} ETFs written to {out / 'universe_us.csv'}")
    print(st.groupby("exchange").size().to_string())
    print("\nsurvivorship: only names listed today are here; delisted companies are missing from every backtest on this list.")


if __name__ == "__main__":
    main()
