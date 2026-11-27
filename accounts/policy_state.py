"""V4a: remember the hold policy between daily runs.

HoldPolicy keeps its memory (the day each stock was first bought, its highest close since, the cooldown after a stop, which names
are in the short-term sleeve) as integer day indices into ONE price frame, and that frame only lives for one backtest. A daily
runner starts fresh every day with a new price frame, so this module saves the memory to a small JSON file and puts it back.

Dates, not indices, are saved, so the file stays valid when tomorrow's price frame starts on a different day:
  * entry / tact: the date a name was first bought (held days are recounted in the new frame's trading days; a date before the
    frame's first day is counted back in weekdays);
  * cooldown: trading days a stopped-out name was still blocked at the save date (days that pass in the new frame are subtracted);
  * peak: the highest close since purchase (a price, saved as is);
  * last_full / last_fill: dates of the last full rebalance and the last refill, so the runner knows what is due today: a full
    rebalance every `every` trading days and a refill every `fill_every`, the rhythm of accounts.sim.run_account. Unlike the
    backtest's fixed grid, a missed day does not skip a rebalance: it is simply done on the next run.

Typical daily use (prices: date x ticker, last row = today):

    state = load_state(path)                                   # None on the very first day
    pol = HoldPolicy(params, prices)
    i = len(prices) - 1
    restore(state, pol, prices.index, i)
    mode = mode_today(state, prices.index, i, params)
    pol.day(acct, i, prices.index[i], prices.iloc[i].to_dict(), scores, mode)
    new = capture(pol, prices.index, i, mode if scores is not None else None, prev=state)
    save_state(path, new)        # only once the day's orders really happened; a dry run should save elsewhere or not at all

Only the policy's memory lives here. Holdings and cash come from the game itself each day. No network, no orders.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 1


@dataclass
class PolicyState:
    asof: str | None = None                                   # trading date of the run that saved this (YYYY-MM-DD)
    started: bool = False                                     # the first rebalance has run (the core floor is enforced from then on)
    entry: dict[str, str] = field(default_factory=dict)       # ticker -> date first bought
    tact: dict[str, str] = field(default_factory=dict)        # short-term sleeve ticker -> date bought
    peak: dict[str, float] = field(default_factory=dict)      # ticker -> highest close since bought
    cooldown: dict[str, int] = field(default_factory=dict)    # ticker -> trading days still blocked from re-buying, counted at `asof`
    last_full: str | None = None                              # date of the last full rebalance
    last_fill: str | None = None                              # date of the last refill (a full rebalance also counts)

    def to_dict(self) -> dict:
        return dict(version=VERSION, **asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyState":
        if not isinstance(d, dict):
            raise ValueError("policy state must be a JSON object")
        if d.get("version") != VERSION:
            raise ValueError(f"policy state version {d.get('version')!r} is not supported (expected {VERSION})")
        return cls(
            asof=_opt_date(d.get("asof")),
            started=bool(d.get("started", False)),
            entry={str(t): _date(v) for t, v in d.get("entry", {}).items()},
            tact={str(t): _date(v) for t, v in d.get("tact", {}).items()},
            peak={str(t): float(v) for t, v in d.get("peak", {}).items()},
            cooldown={str(t): int(v) for t, v in d.get("cooldown", {}).items()},
            last_full=_opt_date(d.get("last_full")),
            last_fill=_opt_date(d.get("last_fill")),
        )


# ---------------------------------------------------------------------- policy <-> state
def capture(policy, dates, i: int, mode: str | None = None, prev: PolicyState | None = None) -> PolicyState:
    """The policy's memory after day i of the frame `dates`, as dates. `mode` is what the policy actually did today
    ('full', 'fill' or None: pass None when it had no scores and so did nothing); `prev` carries the earlier rebalance dates."""
    dates = _norm(dates)
    _check_i(dates, i)
    today = _date_at(dates, i)
    last_full = prev.last_full if prev else None
    last_fill = prev.last_fill if prev else None
    if mode == "full":
        last_full = last_fill = today
    elif mode == "fill":
        last_fill = today
    elif mode is not None:
        raise ValueError(f"unknown mode {mode!r}")
    return PolicyState(
        asof=today,
        started=bool(policy.started),
        entry={t: _date_at(dates, k) for t, k in policy.entry.items()},      # insertion order kept: the policy sells in this order
        tact={t: _date_at(dates, k) for t, k in policy.tact.items()},
        peak={t: float(v) for t, v in policy.peak.items()},
        cooldown={t: int(u - i) for t, u in policy.until.items() if u > i},
        last_full=last_full,
        last_fill=last_fill,
    )


def restore(state: PolicyState | None, policy, dates, i: int) -> None:
    """Put a saved memory into a fresh HoldPolicy whose price frame is `dates` and whose today is index i. None = first day."""
    if state is None:
        return
    dates = _norm(dates)
    _check_i(dates, i)
    if state.asof is not None:
        asof = pd.Timestamp(state.asof)
        if asof > dates[i]:
            raise ValueError(f"policy state is from {state.asof}, after today ({_date_at(dates, i)}): wrong file or wrong prices")
        base = _pos(dates, asof)
    else:
        base = i
    policy.entry = {t: _pos(dates, d) for t, d in state.entry.items()}
    policy.tact = {t: _pos(dates, d) for t, d in state.tact.items()}
    policy.peak = {t: float(v) for t, v in state.peak.items()}
    policy.until = {t: base + int(n) for t, n in state.cooldown.items() if base + int(n) > i}
    policy.started = bool(state.started)


def mode_today(state: PolicyState | None, dates, i: int, params) -> str | None:
    """'full' when a full rebalance is due (first run, or `params.every` trading days since the last one), 'fill' when a refill is
    due (`params.fill_every` days since the last refill or full rebalance), otherwise None (only stops and the core floor run)."""
    if state is None or not state.started or state.last_full is None:
        return "full"
    dates = _norm(dates)
    _check_i(dates, i)
    if i - _pos(dates, state.last_full) >= params.every:
        return "full"
    if i - _pos(dates, state.last_fill or state.last_full) >= params.fill_every:
        return "fill"
    return None


# ---------------------------------------------------------------------- file
def save_state(path, state: PolicyState) -> Path:
    """Write the state as JSON. Written to a temporary file first and then swapped in, so a crash never leaves half a file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=1)
        f.write("\n")
    _replace(tmp, p)
    return p


def _replace(src: Path, dst: Path, tries: int = 20, wait: float = 0.05, sleep=time.sleep) -> None:
    """os.replace, retried briefly on Windows' 'Access is denied': a virus scanner or the search indexer often holds a file that
    was written a moment ago. Gives up (re-raises) after about 10 seconds in total."""
    for n in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if n == tries - 1:
                raise
            sleep(wait * (n + 1))


def load_state(path) -> PolicyState | None:
    """The saved state, or None if there is no file yet (first day). A damaged file raises ValueError rather than starting over,
    because starting over would trigger a full rebalance and forget every stop and cooldown."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"policy state file {p} is damaged: {e}") from e
    return PolicyState.from_dict(d)


# ---------------------------------------------------------------------- dates
def _norm(dates) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(dates)
    if d.tz is not None:
        d = d.tz_localize(None)
    d = d.normalize()
    if len(d) == 0:
        raise ValueError("empty price frame")
    if not d.is_monotonic_increasing:
        raise ValueError("price frame dates must be sorted")
    return d


def _check_i(dates: pd.DatetimeIndex, i: int) -> None:
    if not 0 <= i < len(dates):
        raise IndexError(f"day index {i} is outside the price frame (0..{len(dates) - 1})")


def _pos(dates: pd.DatetimeIndex, d) -> int:
    """Index of date d in the trading-day frame. A date before the frame gets a negative index counted in weekdays; a date missing
    from the frame (a holiday or a gap in the data) maps to the next trading day in it."""
    d = pd.Timestamp(d).normalize()
    first = dates[0]
    if d < first:
        return -int(np.busday_count(np.datetime64(d.date(), "D"), np.datetime64(first.date(), "D")))
    return int(dates.searchsorted(d, side="left"))


def _date_at(dates: pd.DatetimeIndex, k: int) -> str:
    """The date of index k; a negative k is counted back from the first day in weekdays (the inverse of _pos)."""
    k = int(k)
    if k >= 0:
        return dates[min(k, len(dates) - 1)].strftime("%Y-%m-%d")
    return str(np.busday_offset(np.datetime64(dates[0].date(), "D"), k, roll="forward"))


def _date(v) -> str:
    return pd.Timestamp(v).strftime("%Y-%m-%d")


def _opt_date(v) -> str | None:
    return None if v is None else _date(v)
