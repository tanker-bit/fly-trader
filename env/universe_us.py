"""S&P 500 universe with point-in-time membership, rebuilt from Wikipedia's constituents + changes tables.

Method: start from today's members and walk the change log backwards. A change on date D "adds A, removes R" means R was a
member just before D and A was not. This gives membership intervals per ticker, so a stock enters the cross-section only
from the day it joined the index, which removes the bias of ranking today's winners over years when they were not yet
large. What it can NOT fix (be honest about it):
  * removed companies often have no Yahoo price history (delisted) and no SEC mapping, so they silently drop out;
  * Wikipedia's change log is "selected changes" and tickers get renamed or reused, so a few intervals will be wrong.
Removed tickers are NOT auto-mapped to SEC companies (a reused ticker would attach the wrong company's financials); add them
by hand in universe/cik_overrides.csv (columns: ticker,cik) if you want them.
"""
from __future__ import annotations

import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def tls_context():
    """Certificate checking stays ON. Prefer the certifi bundle (installed with yfinance/requests) over the Windows store,
    which can hold an expired or missing root and make Python fail with 'certificate has expired'."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def fetch_text(url: str, user_agent: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout, context=tls_context()) as r:
        return r.read().decode("utf-8", errors="replace")


class _Tables(HTMLParser):
    """Collects EVERY table (nested ones too) as a list of rows of (is_header, text) cells.

    A stack keeps each table's rows separate, so a table sitting inside another table or wrapper is still found
    (the earlier top-level-only version missed the 'changes' table on the real page)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.stack = [], []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.stack.append(dict(rows=[], row=None, cell=None, hdr=False))
        elif self.stack:
            t = self.stack[-1]
            if tag == "tr":
                t["row"] = []
            elif tag in ("td", "th") and t["row"] is not None:
                t["cell"], t["hdr"] = [], tag == "th"

    def handle_endtag(self, tag):
        if not self.stack:
            return
        t = self.stack[-1]
        if tag == "table":
            self.tables.append(self.stack.pop()["rows"])
        elif tag in ("td", "th") and t["cell"] is not None:
            t["row"].append((t["hdr"], re.sub(r"\[[^\]]*\]", "", " ".join("".join(t["cell"]).split()))))
            t["cell"] = None
        elif tag == "tr" and t["row"] is not None:
            t["rows"].append(t["row"])
            t["row"] = None

    def handle_data(self, data):
        if self.stack and self.stack[-1]["cell"] is not None:
            self.stack[-1]["cell"].append(data)


def _head(rows, n=2):
    return " ".join(t.lower() for r in rows[:n] for _, t in r)


def _sym(x: str) -> str:
    return x.strip().upper().replace(".", "-")


HIST_URLS = ["https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500",
             "https://en.wikipedia.org/wiki/Selected_changes_to_the_list_of_S%26P_500_companies"]


def parse_changes(html: str) -> pd.DataFrame:
    """Index-change log (date, added, removed) from any page holding a table headed Added/Removed. Empty frame if none."""
    p = _Tables()
    p.feed(html)
    chg = []
    for t in p.tables:
        if not ("added" in _head(t, 3) and "removed" in _head(t, 3)):
            continue
        for r in t:
            cells = [x for _, x in r]
            if len(cells) >= 5:
                d = pd.to_datetime(cells[0], errors="coerce")
                if pd.notna(d):
                    chg.append((d, _sym(cells[1]) if cells[1] else "", _sym(cells[3]) if cells[3] else ""))
    return pd.DataFrame(chg, columns=["date", "added", "removed"])


def parse_wikipedia(html: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (current: symbol, security, date_added, cik ; changes: date, added, removed).
    The constituents table is required. The changes table used to be on this page; Wikipedia moved it, so it may be
    empty here (the caller then fetches HIST_URLS)."""
    p = _Tables()
    p.feed(html)
    cons = next((t for t in p.tables if t and "symbol" in _head(t, 1) and "security" in _head(t, 1)), None)
    if cons is None:
        seen = [(len(t), _head(t, 1)[:60]) for t in p.tables[:6]]
        raise ValueError(f"could not find the constituents table (page has {len(p.tables)} tables, {len(html)} characters; "
                         f"first headers: {seen}).")
    head = [t.lower() for _, t in cons[0]]
    col = lambda name: next(i for i, h in enumerate(head) if h.startswith(name))
    ic, isec, ida, icik = col("symbol"), col("security"), col("date added"), col("cik")
    igs = next((i for i, h in enumerate(head) if h == "gics sector"), None)
    isub = next((i for i, h in enumerate(head) if h == "gics sub-industry"), None)
    cur = []
    for r in cons[1:]:
        if len(r) > max(ic, isec, ida, icik):
            cik = re.sub(r"\D", "", r[icik][1])
            cur.append((_sym(r[ic][1]), r[isec][1], pd.to_datetime(r[ida][1], errors="coerce"), int(cik) if cik else np.nan,
                        r[igs][1] if igs is not None and len(r) > igs else "", r[isub][1] if isub is not None and len(r) > isub else ""))
    return (pd.DataFrame(cur, columns=["symbol", "security", "date_added", "cik", "sector", "sub_industry"]),
            parse_changes(html))


def build_intervals(current: pd.DataFrame, changes: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Membership intervals [start, end) per ticker (NaT = unbounded). Returns (intervals, number of inconsistent changes)."""
    state = {s: pd.NaT for s in current["symbol"]}          # ticker -> end date of the interval being traced backwards
    iv, bad = [], 0
    for _, c in changes.sort_values("date", ascending=False, kind="stable").iterrows():
        if c["removed"]:
            if c["removed"] in state:
                bad += 1
            else:
                state[c["removed"]] = c["date"]
        if c["added"]:
            if c["added"] in state:
                iv.append((c["added"], c["date"], state.pop(c["added"])))
            else:
                bad += 1
    added = dict(zip(current["symbol"], current["date_added"]))
    for sym, end in state.items():
        iv.append((sym, added.get(sym, pd.NaT) if pd.isna(end) else pd.NaT, end))
    return pd.DataFrame(iv, columns=["symbol", "start", "end"]), bad


def membership_matrix(intervals: pd.DataFrame, dates: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    out = pd.DataFrame(False, index=dates, columns=list(tickers))
    d = dates.to_numpy("datetime64[ns]")
    for sym, g in intervals.groupby("symbol"):
        if sym not in out.columns:
            continue
        m = np.zeros(len(dates), bool)
        for _, r in g.iterrows():
            lo = np.ones(len(dates), bool) if pd.isna(r["start"]) else d >= r["start"].to_datetime64()
            hi = np.ones(len(dates), bool) if pd.isna(r["end"]) else d < r["end"].to_datetime64()
            m |= lo & hi
        out[sym] = m
    return out


def cik_map(current: pd.DataFrame, intervals: pd.DataFrame, overrides_csv: Path | None = None) -> dict[str, int]:
    """symbol -> CIK. Current members use the CIK from Wikipedia; others only through the hand-written overrides file."""
    m = {r.symbol: int(r.cik) for r in current.itertuples() if pd.notna(r.cik)}
    if overrides_csv is not None and Path(overrides_csv).exists():
        for r in pd.read_csv(overrides_csv).itertuples():
            m[_sym(r.ticker)] = int(r.cik)
    return m
