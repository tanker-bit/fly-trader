"""Fresh S&P 500 membership for the live run, kept in a SEPARATE folder (default data_live/universe).

The research list (data_fundamentals/universe) is never rewritten: the training panel and its exactness tests depend on it.
refresh_membership() reads Wikipedia's current constituents and its change log (the same parser that built the research list,
env.universe_us), rebuilds the point-in-time intervals, copies the hand-made CIK overrides, and writes:
    data_live/universe/constituents.csv, changes.csv, intervals.csv, cik_overrides.csv, fetched_at.txt
tournament.live_features.load_inputs(..., universe_dir=...) then uses it. Network: Wikipedia only, with a plain project
User-Agent; no personal data is sent. A suspicious page (too few or too many members) is refused and nothing is written.
"""
from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import pandas as pd

from env import universe_us as uni

USER_AGENT = "fly-trader research script (S&P 500 membership; personal non-commercial use)"
DEFAULT_DIR = Path("data_live") / "universe"
RESEARCH_DIR = Path("data_fundamentals") / "universe"
STALE_DAYS = 7
MIN_MEMBERS, MAX_MEMBERS = 480, 520            # the S&P 500 has about 503 lines (some companies have two share classes)


def fetch_membership(fetch=uni.fetch_text) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (current constituents, change log). The change log moved off the main page, so the history pages are tried too."""
    cur, chg = uni.parse_wikipedia(fetch(uni.WIKI_URL, USER_AGENT))
    if chg.empty:
        for url in uni.HIST_URLS:
            try:
                chg = uni.parse_changes(fetch(url, USER_AGENT))
            except Exception:
                continue
            if len(chg):
                break
    return cur, chg


def refresh_membership(out_dir=DEFAULT_DIR, research_dir=RESEARCH_DIR, fetch=uni.fetch_text, now=None, log=print) -> dict:
    out_dir, research_dir = Path(out_dir), Path(research_dir)
    cur, chg = fetch_membership(fetch)
    if not MIN_MEMBERS <= len(cur) <= MAX_MEMBERS:
        raise ValueError(f"Wikipedia gave {len(cur)} constituents (expected {MIN_MEMBERS}-{MAX_MEMBERS}); nothing was written")
    if chg.empty:
        old = research_dir / "changes.csv"
        if not old.exists():
            raise ValueError("no change log on Wikipedia and none in the research folder; nothing was written")
        chg = pd.read_csv(old, parse_dates=["date"]).fillna("")
        log("membership: Wikipedia's change log was not found; the research change log is used")
    iv, bad = uni.build_intervals(cur, chg)
    out_dir.mkdir(parents=True, exist_ok=True)
    cur[["symbol", "security", "date_added", "cik"]].to_csv(out_dir / "constituents.csv", index=False)
    chg.to_csv(out_dir / "changes.csv", index=False)
    iv.to_csv(out_dir / "intervals.csv", index=False)
    if (research_dir / "cik_overrides.csv").exists():
        shutil.copyfile(research_dir / "cik_overrides.csv", out_dir / "cik_overrides.csv")
    stamp = (now or dt.datetime.now()).isoformat(timespec="seconds")
    (out_dir / "fetched_at.txt").write_text(stamp + "\n", encoding="utf-8")
    old_members = set(pd.read_csv(research_dir / "constituents.csv")["symbol"]) if (research_dir / "constituents.csv").exists() else set()
    new = set(cur["symbol"])
    res = dict(members=len(cur), changes=len(chg), newest_change=str(pd.to_datetime(chg["date"]).max().date()) if len(chg) else None,
               inconsistent=bad, added_since_research=sorted(new - old_members), removed_since_research=sorted(old_members - new))
    log(f"membership: {res['members']} members, newest index change {res['newest_change']}; since the research list: "
        f"added {res['added_since_research'] or 'none'}, removed {res['removed_since_research'] or 'none'} -> {out_dir}")
    return res


def fetched_at(out_dir=DEFAULT_DIR):
    f = Path(out_dir) / "fetched_at.txt"
    try:
        return dt.datetime.fromisoformat(f.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def is_stale(out_dir=DEFAULT_DIR, days: int = STALE_DAYS, now=None) -> bool:
    t = fetched_at(out_dir)
    return t is None or not (Path(out_dir) / "intervals.csv").exists() or ((now or dt.datetime.now()) - t).days >= days


def universe_dir(out_dir=DEFAULT_DIR):
    """The live membership folder if it has been fetched, else None (the research list is used)."""
    d = Path(out_dir)
    return d if (d / "intervals.csv").exists() and (d / "constituents.csv").exists() else None
