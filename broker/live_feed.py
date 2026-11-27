"""A live event file for the 3D dashboard: what the fly is doing on the website, one JSON object per line.

The autopilot and broker.trade_ui append events (default orders/live_feed.jsonl); experiments/visual_dashboard.py --trade-feed
tails the file and animates them. Writing never fails a trading run: a problem with the file is ignored.

Events (every one also has "t" = ISO time and "run" = the run's id):
  start    {mode: "trading" | "preview", asof?}           a new run begins
  phase    {text, asof?, n_stocks?, n_flies?}              e.g. READING ACCOUNT, REFRESHING PRICES, SCORING, PLANNED 26 ORDERS
  account  {equity, cash, buying_power, positions: [[ticker, shares, price, value, pnl_fraction], ...]}
  search   {symbol, side, shares, rank, score, reason}      the fly types a ticker into the Trade tab
  quote    {symbol, bid, ask, day_high, day_low, high_52w, low_52w, volume}
  preview  {symbol, side, shares, est_price, est_total}
  result   {symbol, side, shares, status, detail, score}    status: submitted | skipped | previewed | error | stopped
  done     {text}
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

DEFAULT = Path("orders") / "live_feed.jsonl"


class Feed:
    def __init__(self, path=DEFAULT, run=None):
        self.path = Path(path)
        self.run = run or dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    def __call__(self, kind: str, **data):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            rec = dict(t=dt.datetime.now().isoformat(timespec="seconds"), run=self.run, kind=kind, **data)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass                                  # the dashboard is optional; trading must not depend on it


def nothing(kind: str, **data):
    """The default when no dashboard feed is wanted."""


def read_new(path, offset: int = 0):
    """-> (events, new_offset). Reads complete lines written since `offset` (bytes). A missing file gives ([], 0)."""
    p = Path(path)
    if not p.exists():
        return [], 0
    size = p.stat().st_size
    if size < offset:                             # the file was replaced: start again
        offset = 0
    with open(p, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    end = chunk.rfind(b"\n")
    if end < 0:
        return [], offset
    out = []
    for line in chunk[: end + 1].splitlines():
        try:
            out.append(json.loads(line.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            continue
    return out, offset + end + 1


def last_run(path) -> list:
    """All events of the newest run in the file (for a replay)."""
    ev, _ = read_new(path, 0)
    starts = [i for i, e in enumerate(ev) if e.get("kind") == "start"]
    return ev[starts[-1]:] if starts else ev


SCREEN = Path("orders") / "live_screen.jpg"


class Mirror:
    """Saves what the fly's browser shows to a JPEG (default orders/live_screen.jpg) for the dashboard's website monitor.
    At most one picture every `every` seconds unless forced; written to a temporary file and swapped in, so the dashboard never
    reads half a picture. Never fails a trading run."""

    def __init__(self, path=SCREEN, every: float = 0.6, quality: int = 60):
        import time as _time
        self.path, self.every, self.quality, self._clock, self._last = Path(path), every, quality, _time.monotonic, -1e9

    def __call__(self, page, force: bool = False):
        now = self._clock()
        if not force and now - self._last < self.every:
            return False
        self._last = now
        try:
            import os
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            page.screenshot(path=str(tmp), type="jpeg", quality=self.quality)
            os.replace(tmp, self.path)
            return True
        except Exception:
            return False


def no_mirror(page, force: bool = False):
    return False

