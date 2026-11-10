"""The dashboard's data layer: reads a league folder without writing to it, survives half-written files, and its replay maps
saved scores to the same dates and tickers as tournament.data.Data."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard import state
from tests.test_league import make_panel
from tournament.data import Data
from tournament.store import Store


def build_league(root):
    s = Store(root)
    s.save_state(dict(round=3))
    for aid, ics, silent in (("a1", [0.05, 0.03, 0.04], False), ("a2", [0.02, 0.01, 0.0], True), ("a3", [0.08, 0.02, -0.01], False)):
        h = [dict(round=i, ic=v, sd=0.1, n=10, **({"silent": True} if (silent and i == 2) else {})) for i, v in enumerate(ics)]
        s.save_agent(dict(id=aid, status="alive", gen=1, last_round_done=2, history=h))
    s.save_agent(dict(id="a9", status="culled", gen=0, last_round_done=1, history=[]))
    for r, ridge in enumerate([0.01, 0.02, 0.03]):
        s._write(Path(root) / "rounds" / f"round_{r:02d}.json",
                 dict(round=r, controls={"ridge": dict(ic=ridge, sd=0.1), "fly untrained (default)": dict(ic=-0.01, sd=0.1)}))
    return s


def test_snapshot_ranks_agents_and_reads_controls_without_writing():
    root = tempfile.mkdtemp()
    build_league(root)
    before = sorted((str(p), p.stat().st_mtime_ns) for p in Path(root).rglob("*") if p.is_file())
    snap = state.snapshot(root, n_universe=499)
    after = sorted((str(p), p.stat().st_mtime_ns) for p in Path(root).rglob("*") if p.is_file())
    assert before == after                                   # strictly read-only
    assert snap.round == 3 and snap.n_universe == 499
    assert {a.id for a in snap.agents} == {"a1", "a2", "a3"}   # culled agents are not shown
    assert snap.top.id == "a1"                                # avg3 0.04 beats a3's 0.0300 and the silent a2
    assert abs(snap.ridge_avg3 - 0.02) < 1e-9 and abs(snap.untrained_avg3 + 0.01) < 1e-9
    a2 = next(a for a in snap.agents if a.id == "a2")
    assert a2.silent


def test_half_written_file_is_survived():
    root = tempfile.mkdtemp()
    build_league(root)
    (Path(root) / "agents" / "a1" / "state.json").write_text('{"id": "a1", "hist')   # truncated mid-write
    snap = state.snapshot(root)
    assert {a.id for a in snap.agents} == {"a2", "a3"}          # skipped for now, no crash


def test_mode_detection():
    root, sec = tempfile.mkdtemp(), Path(tempfile.mkdtemp())
    build_league(root)
    (sec / "raw").mkdir()
    f = sec / "raw" / "CIK0000000001.json"
    f.write_text("{}")
    now = time.time()
    assert state.detect_mode(root, sec, now=now) == "research"
    old = now - 500
    os.utime(f, (old, old))
    for p in Path(root).glob("agents/*/state.json"):
        os.utime(p, (old, old))
    assert state.detect_mode(root, sec, now=now) == "idle"
    os.utime(Path(root) / "agents" / "a3" / "state.json", (now, now))
    assert state.detect_mode(root, sec, now=now) == "league"


def test_replay_matches_data_ordering_and_flags_silent():
    p = make_panel(700, 15)
    fr = p["fwd_ret"].unstack("ticker")
    r21 = fr.rolling(21).sum().shift(-20)
    p["fwd_excess_21"] = r21.sub(r21.mean(axis=1), axis=0).stack().reindex(p.index)
    f = os.path.join(tempfile.mkdtemp(), "p.pkl")
    p.to_pickle(f)
    d = Data(f, 300, 100, 50)
    idx = state.date_ticker_index(f)
    assert len(idx) == len(d.dates)
    for i in (0, 5, len(d.dates) - 1):
        assert idx[i][0] == d.dates[i] and list(idx[i][1]) == list(d.day[i][0])
    i0 = 330
    live = np.linspace(-1, 1, len(d.day[i0][0]))
    dec = state.replay_decisions({i0: live, i0 + 5: np.zeros(len(d.day[i0 + 5][0]))}, idx, "a1", 0)
    assert [x.action for x in dec] == ["BUY", "HOLD", "SELL", "SILENT"]
    assert dec[0].ticker == d.day[i0][0][-1] and dec[2].ticker == d.day[i0][0][0]
    assert dec[0].score == 1.0 and dec[2].score == -1.0


def test_research_feed_reads_real_cache_files():
    sec = Path(tempfile.mkdtemp())
    (sec / "raw").mkdir()
    js = dict(cik=320193, facts={"us-gaap": {"Revenues": {"units": {"USD": [
        dict(val=100, fy=2022, filed="2022-11-01"), dict(val=383285000000, fy=2023, filed="2023-11-03")]}}}})
    (sec / "raw" / "CIK0000320193.json").write_text(json.dumps(js))
    lines, cache = state.research_feed(sec, {320193: "AAPL"})
    assert len(lines) == 1 and "AAPL" in lines[0] and "383285000000" in lines[0] and "2023-11-03" in lines[0]
    lines2, _ = state.research_feed(sec, {320193: "AAPL"}, cache=cache)
    assert lines2 == lines


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
            print("ok", n)
    print("all dashboard state tests passed")
