"""Controller between the league folders and the scene. No pygame here (so it is testable). A background thread polls the
league (read-only) and prepares data; update()/view() are cheap and never touch the disk, so the window stays smooth and the
multiprocessing workers are never slowed down."""
from __future__ import annotations

import threading
import time
from collections import deque

from dashboard import state

try:
    from accounts.instruments import START_CASH
except Exception:                       # pragma: no cover
    START_CASH = 1_000_000.0

try:                                    # real account fills; if it cannot import, the old score replay is used
    from dashboard import trades
except Exception:                       # pragma: no cover
    trades = None
from dashboard.scene import FlyAnimator, View

STATUS = ("SEARCHING SEC EDGAR...", "DOWNLOADING SEC XBRL...", "PARSING TICKER {tk}...")


class DashboardApp:
    def __init__(self, league_root, panel=None, sec_root=None, universe_dir=None, agent=None, poll=2.0, start_threads=True):
        self.league_root, self.panel, self.sec_root, self.universe_dir = league_root, panel, sec_root, universe_dir
        self.pin, self.poll = agent, poll
        self.animator = FlyAnimator()
        self.snap = state.snapshot(league_root, sec_root)
        self.index = None                       # [(date, tickers)] once the panel labels are read
        self.fwd1 = None                        # next-day returns for the real-trade replay (None -> old score replay)
        self.etf_ok = False                     # True when ETF prices were found, so the 30% broad-ETF core can be held
        self.real = False                       # True while the decisions come from real account fills
        self.n_universe = 0
        self.index_error = None
        self.cik = {}
        self.research_lines, self._rcache = [], {}
        self.decisions, self.cursor, self.src = [], 0, None     # replay source: (agent, round)
        self.tape = deque(maxlen=12)
        self.last_decision = None
        self.account = None                     # (cash, equity, positions) right after the last real trade
        self.paused = False
        self.clock = 0.0
        self.quit = False
        self._agent_ix = 0
        self._lock = threading.Lock()
        if start_threads:
            threading.Thread(target=self._load_index, daemon=True).start()
            threading.Thread(target=self._poll_loop, daemon=True).start()

    # ------------------------------------------------------------ background work
    def _load_index(self):
        if not self.panel:
            return
        try:
            idx = None
            if trades is not None:
                try:                                # aligned with the league's own label (blends, sector-neutral)
                    from tournament.store import Store
                    cfg = Store(self.league_root).load_config()
                    cfg["panel"] = self.panel
                    idx, self.fwd1 = trades.league_index(cfg)
                    try:                            # approved ETFs (fetch_etfs cache next to the panel) join the stocks
                        from pathlib import Path
                        from accounts.etfs import with_etfs
                        self.fwd1 = with_etfs(self.fwd1, Path(self.panel).parent)
                        self.etf_ok = "SPY" in self.fwd1.columns
                    except Exception:
                        self.etf_ok = False
                except Exception:
                    idx, self.fwd1 = None, None     # cannot align: old replay
            if idx is None:
                idx = state.date_ticker_index(self.panel)
            self.n_universe = len(idx[-1][1]) if idx else 0
            self.index = idx                        # set last, so the replay never starts before fwd1 is ready
        except Exception as e:
            self.index_error = str(e)

    def _poll_loop(self):
        while not self.quit:
            try:
                self.poll_once()
            except Exception:
                pass                              # never let a bad file kill the dashboard
            time.sleep(self.poll)

    def candidates(self):
        return sorted([a for a in self.snap.agents], key=lambda a: -(a.avg3 if a.avg3 == a.avg3 else -9))

    def poll_once(self):
        snap = state.snapshot(self.league_root, self.sec_root, self.n_universe)
        if self.universe_dir and not self.cik:
            self.cik = state.cik_ticker_map(self.universe_dir)
        if self.sec_root:
            self.research_lines, self._rcache = state.research_feed(self.sec_root, self.cik, cache=self._rcache)
        with self._lock:
            self.snap = snap
        self._refresh_replay(snap)

    def _refresh_replay(self, snap):
        if self.index is None:
            return
        cands = self.candidates()
        aid = self.pin or (snap.top.id if snap.top else (cands[0].id if cands else None))
        if aid is None:
            return
        for r in range(max(snap.last_done_round, snap.round), -1, -1):
            sc = state.load_scores(self.league_root, r, aid)
            if sc:
                if self.src != (aid, r):
                    dec, real = None, False
                    if self.fwd1 is not None:
                        try:
                            dec, real = trades.replay_trades(sc, self.index, self.fwd1, aid, r), True
                        except Exception:
                            dec = None              # fall back rather than show nothing
                    if dec is None:
                        dec = state.replay_decisions(sc, self.index, aid, r)
                    with self._lock:
                        self.decisions, self.cursor, self.src, self.real = dec, 0, (aid, r), real
                        self.account = None             # a new replay starts from the $1,000,000 start state
                return

    # ------------------------------------------------------------ foreground (cheap)
    def next_agent(self):
        c = self.candidates()
        if c:
            self._agent_ix = (self._agent_ix + 1) % len(c)
            self.pin = c[self._agent_ix].id

    def toggle_pause(self):
        self.paused = not self.paused

    def update(self, dt):
        if self.paused:
            return
        self.clock += dt
        a = self.animator
        a.set_research(self.snap.mode == "research")
        if a.cur is None and not a.queue:
            with self._lock:
                if self.decisions:
                    d = self.decisions[self.cursor % len(self.decisions)]
                    self.cursor += 1
                else:
                    d = None
            if d is not None:
                a.push(d)
                self.last_decision = d
                self._apply_account(d)
                if d.action != "SILENT":
                    self.tape.append((d.ticker, d.score))
        a.update(dt)

    def _apply_account(self, d):
        """Remember the account as it is right after this decision (only real-trade decisions carry one)."""
        if self.real and hasattr(d, "equity"):
            self.account = (d.cash, d.equity, list(d.positions))

    def _account_header(self):
        """Header keys for the money panel: cash, equity, positions. Empty when the decisions are not real trades (panel stays hidden)."""
        if not self.real:
            return {}
        cash, eq, pos = self.account or (START_CASH, START_CASH, [])
        return dict(cash=cash, equity=eq, positions=list(pos), start_cash=START_CASH)

    def _date_header(self):
        """The trading date of the decision being shown (ISO), so the monitors can print dd/mm/yyyy."""
        d = self.last_decision
        return {"trade_date": d.when} if d is not None and getattr(d, "when", "") else {}

    def replay_label(self):
        if self.index is None:
            return "LOADING PANEL INDEX..." if not self.index_error else "PANEL INDEX FAILED: " + self.index_error[:60]
        if not self.src:
            return "NO SAVED SCORES YET (agents from rounds before v6 saved none)"
        d = self.last_decision
        what = "real long-only account fills, simulated $1M, no real money" if self.real else "saved out-of-sample scores, no real trades"
        if self.real and not self.etf_ok:
            what += "; NO ETF DATA, core floor off (run fetch_etfs)"
        return f"REPLAY r{self.src[1]:02d} {self.src[0]} {d.when if d else ''}  ({what})"

    def view(self):
        s = self.snap
        lines = self.research_lines
        tk = lines[int(self.clock * 12) % len(lines)][:6].strip() if lines else "..."
        status = STATUS[int(self.clock * 1.2) % 3].format(tk=tk)
        hdr = dict(round=s.round, universe=self.n_universe or s.n_universe, top_id=s.top.id if s.top else None,
                   top_ic=s.top.avg3 if s.top else None, ridge_ic=s.ridge_avg3, untrained_ic=s.untrained_avg3, mode=s.mode)
        hdr.update(self._account_header())
        hdr.update(self._date_header())
        return View(header=hdr, pose=self.animator.pose(), log=list(self.animator.log), tape=list(self.tape),
                    research_lines=lines, research_status=status, clock=self.clock, replay_label=self.replay_label())
