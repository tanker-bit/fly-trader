"""Read-only view of the league on disk. Everything here only READS files the league already writes, tolerates files
that are half-written (retries, then skips), and does its slow work (the ticker universe, SEC cache sampling) outside the
draw loop so it cannot slow the multiprocessing workers or the window.

What is real and what is not (the dashboard must not pretend otherwise):
  * league round / agents / IC history / ridge control        -> real, from <root>/state.json, agents/*/state.json, rounds/round_XX.json
  * universe size                                             -> real, tickers on the panel's last date
  * research feed (tickers + accounting facts)                -> real, sampled from the SEC cache files being written
  * trade decisions                                           -> a REPLAY of the saved out-of-sample scores of one agent
    (rounds/scores/rXX_<id>.npz): top scores are shown as BUY, bottom as SELL, middle as HOLD, flat scores as SILENT.
    The league places no real trades and does not stream decisions live; the replay is clearly labelled as such.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _read_json(path, tries=3):
    for k in range(tries):
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, PermissionError, OSError):
            time.sleep(0.05 * (k + 1))                      # being replaced right now: try again
    return None


@dataclass
class AgentView:
    id: str
    status: str
    gen: int
    last_round: int
    last_ic: float
    avg3: float
    silent: bool
    rounds: int


@dataclass
class Snapshot:
    root: str
    round: int = -1
    last_done_round: int = -1
    agents: list = field(default_factory=list)
    top: AgentView | None = None
    ridge_avg3: float = float("nan")
    untrained_avg3: float = float("nan")
    n_universe: int = 0
    mode: str = "idle"                 # 'research' | 'league' | 'idle'
    taken_at: float = 0.0


def _mean(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def read_agents(root):
    out = []
    adir = Path(root) / "agents"
    if not adir.exists():
        return out
    for p in sorted(adir.iterdir()):
        rec = _read_json(p / "state.json")
        if not rec:
            continue
        h = rec.get("history", [])
        live = [x["ic"] for x in h[-3:] if not x.get("silent")]
        out.append(AgentView(id=rec["id"], status=rec.get("status", "?"), gen=rec.get("gen", 0),
                             last_round=rec.get("last_round_done", -1),
                             last_ic=h[-1]["ic"] if h else float("nan"), avg3=_mean(live),
                             silent=bool(h and h[-1].get("silent")), rounds=len(h)))
    return out


def read_controls(root, n_last=3):
    """Mean over the last `n_last` finished rounds of the ridge and untrained-fly controls on each round's validation window."""
    rd = Path(root) / "rounds"
    files = sorted(rd.glob("round_*.json")) if rd.exists() else []
    ridge, untr = [], []
    for f in files[-n_last:]:
        js = _read_json(f)
        if not js:
            continue
        for name, v in js.get("controls", {}).items():
            if name.lower().startswith("ridge"):
                ridge.append(v["ic"])
            if "untrained" in name.lower():
                untr.append(v["ic"])
    return _mean(ridge), _mean(untr)


def detect_mode(league_root, sec_root=None, now=None, research_secs=20, league_secs=120):
    """'research' while SEC cache files are being written, 'league' while agent files are being saved, else 'idle'."""
    now = time.time() if now is None else now

    def newest(paths):
        t = 0.0
        for p in paths:
            try:
                t = max(t, os.path.getmtime(p))
            except OSError:
                pass
        return t

    if sec_root:
        raw = Path(sec_root) / "raw"
        if raw.exists():
            # only look at a few of the most recently listed entries: cheap even with 500 files
            ents = sorted(os.scandir(raw), key=lambda e: e.stat().st_mtime, reverse=True)[:3]
            if ents and now - ents[0].stat().st_mtime < research_secs:
                return "research"
    ag = Path(league_root) / "agents"
    if ag.exists():
        t = newest(list(ag.glob("*/state.json")))
        if now - t < league_secs:
            return "league"
    return "idle"


def snapshot(league_root, sec_root=None, n_universe=0, now=None):
    root = Path(league_root)
    st = _read_json(root / "state.json") or {}
    agents = read_agents(root)
    alive = [a for a in agents if a.status == "alive"]
    rank = sorted([a for a in alive if np.isfinite(a.avg3) and not a.silent], key=lambda a: -a.avg3)
    ridge, untr = read_controls(root)
    s = Snapshot(root=str(root), round=int(st.get("round", -1)), agents=alive, top=rank[0] if rank else None,
                 ridge_avg3=ridge, untrained_avg3=untr, n_universe=n_universe,
                 mode=detect_mode(root, sec_root, now), taken_at=time.time())
    done = [a.last_round for a in alive]
    s.last_done_round = max(done) if done else -1
    return s


class UniverseCounter:
    """Number of distinct stocks on the panel's last date. Reads only the label column + index, in a background thread."""

    def __init__(self, panel_path):
        self.n = 0
        self.error = None
        threading.Thread(target=self._run, args=(panel_path,), daemon=True).start()

    def _run(self, path):
        try:
            self.n = len(date_ticker_index(path)[-1][1])
        except Exception as e:           # keep the dashboard alive without it
            self.error = str(e)


def _load_label_frame(panel_path):
    import pandas as pd
    p = str(panel_path)
    if p.endswith(".pkl"):
        df = pd.read_pickle(p)
        return df[["fwd_excess_21"]] if "fwd_excess_21" in df.columns else df[["fwd_excess"]]
    try:
        return pd.read_parquet(p, columns=["fwd_excess_21"])
    except Exception:
        return pd.read_parquet(p, columns=["fwd_excess"])


def date_ticker_index(panel_path, min_stocks=10):
    """[(date, tickers array)] in EXACTLY the order tournament.data.Data indexes its labelled dates, so a saved score
    array dN.npz-key N maps to this list's N-th entry. Only the label column and the (date, ticker) index are read."""
    df = _load_label_frame(panel_path)
    y = df.iloc[:, 0]
    ok = y.notna()
    out = []
    for d, g in y[ok].groupby(level="date"):
        if len(g) < min_stocks:
            continue
        out.append((d, g.index.get_level_values("ticker").to_numpy()))
    return out


@dataclass
class Decision:
    when: str
    ticker: str
    action: str            # BUY | SELL | HOLD | SILENT
    score: float
    agent: str
    round: int


def load_scores(root, rnd, aid):
    f = Path(root) / "rounds" / "scores" / f"r{rnd:02d}_{aid}.npz"
    if not f.exists():
        return None
    try:
        z = np.load(f)
        return {int(k[1:]): z[k].astype(float) for k in z.files}
    except Exception:
        return None


def replay_decisions(scores, index, agent, rnd, q=0.1):
    """Turn one agent's saved validation scores into a chronological decision stream: per date, the top-scored stock is a BUY,
    the bottom-scored a SELL, and a mid-ranked one a HOLD. A constant score (std ~ 0) means the circuit is dead -> SILENT."""
    out = []
    for i in sorted(scores):
        s = scores[i]
        if i >= len(index):
            continue
        date, tick = index[i]
        if len(s) != len(tick):
            continue
        if np.nanstd(s) < 1e-9:
            out.append(Decision(str(date.date()), "", "SILENT", float(np.nanmean(s)), agent, rnd))
            continue
        order = np.argsort(s, kind="stable")
        out.append(Decision(str(date.date()), str(tick[order[-1]]), "BUY", float(s[order[-1]]), agent, rnd))
        out.append(Decision(str(date.date()), str(tick[order[len(order) // 2]]), "HOLD", float(s[order[len(order) // 2]]), agent, rnd))
        out.append(Decision(str(date.date()), str(tick[order[0]]), "SELL", float(s[order[0]]), agent, rnd))
    return out


def research_feed(sec_root, cik_to_ticker, n=40, cache=None, max_files=6):
    """Real tickers and raw accounting facts taken from the newest SEC cache files (the ones the build is writing now).
    Returns display lines such as 'AAPL  Revenues 383285000000 USD  FY2023 filed 2023-11-03'. `cache` (dict) avoids rereading."""
    cache = {} if cache is None else cache
    raw = Path(sec_root) / "raw"
    if not raw.exists():
        return [], cache
    ents = sorted((e for e in os.scandir(raw) if e.name.endswith(".json")), key=lambda e: e.stat().st_mtime, reverse=True)[:max_files]
    lines = []
    for e in ents:
        key = (e.name, int(e.stat().st_mtime))
        if key not in cache:
            cache[key] = _facts_from_file(e.path, cik_to_ticker)
        lines += cache[key]
    return lines[:n], cache


def _facts_from_file(path, cik_to_ticker, per_file=8):
    try:
        with open(path) as f:
            js = json.load(f)
    except Exception:
        return []
    cik = int(js.get("cik", 0))
    tk = cik_to_ticker.get(cik, f"CIK{cik}")
    out = []
    gaap = js.get("facts", {}).get("us-gaap", {})
    for tag in ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "NetIncomeLoss", "Assets",
                "StockholdersEquity", "OperatingIncomeLoss", "GrossProfit", "NetCashProvidedByUsedInOperatingActivities"):
        units = gaap.get(tag, {}).get("units", {})
        rows = next(iter(units.values()), None) if units else None
        if rows:
            r = rows[-1]
            out.append(f"{tk:<6} {tag[:34]:<34} {r.get('val')}  FY{r.get('fy', '')} filed {r.get('filed', '')}")
        if len(out) >= per_file:
            break
    return out


def cik_ticker_map(universe_dir):
    """{cik: ticker} from the constituents CSV written by build_dataset_fund (current members only)."""
    import pandas as pd
    f = Path(universe_dir) / "constituents.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    return {int(c): s for c, s in zip(df["cik"], df["symbol"]) if np.isfinite(c)}
