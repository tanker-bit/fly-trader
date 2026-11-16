"""Controller (replay feed, pause, agent cycling) and the pygame canvas wrapper, tested with a recording stand-in for pygame."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from dashboard import state
from dashboard.app import DashboardApp
from dashboard.canvas_pygame import PygameCanvas
from dashboard.scene import FlyAnimator, View, draw_scene
from tests.test_dashboard_state import build_league
from tests.test_league import make_panel


def make_world():
    root = tempfile.mkdtemp()
    build_league(root)
    p = make_panel(700, 15)
    fr = p["fwd_ret"].unstack("ticker")
    r21 = fr.rolling(21).sum().shift(-20)
    p["fwd_excess_21"] = r21.sub(r21.mean(axis=1), axis=0).stack().reindex(p.index)
    panel = os.path.join(tempfile.mkdtemp(), "p.pkl")
    p.to_pickle(panel)
    idx = state.date_ticker_index(panel)
    return root, panel, idx


def save_scores(root, aid, rnd, idx, dates, flat=False):
    d = Path(root) / "rounds" / "scores"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    z = {f"d{i}": (np.zeros(len(idx[i][1]), np.float32) if flat else rng.normal(size=len(idx[i][1])).astype(np.float32)) for i in dates}
    np.savez_compressed(d / f"r{rnd:02d}_{aid}.npz", **z)


def test_app_replays_saved_scores_and_header_is_real():
    root, panel, idx = make_world()
    save_scores(root, "a1", 2, idx, [330, 335, 340])
    app = DashboardApp(root, panel, start_threads=False)
    assert "LOADING" in app.replay_label()
    app._load_index()
    app.poll_once()
    assert app.src == ("a1", 2) and len(app.decisions) == 9           # BUY/HOLD/SELL for 3 dates
    seen = set()
    for _ in range(600):
        app.update(0.02)
        seen.add(app.animator.state)
    assert {"BUY", "SELL", "HOLD"} <= seen
    v = app.view()
    assert v.header["round"] == 3 and v.header["top_id"] == "a1" and abs(v.header["top_ic"] - 0.04) < 1e-9
    assert v.header["universe"] == 15 and v.tape and any("[FLY BUY]" in t for _, t in v.log)
    assert v.replay_label.startswith("REPLAY r02 a1") and "no real trades" in v.replay_label


def test_pause_cycle_agent_and_silent_replay_and_no_scores_label():
    root, panel, idx = make_world()
    app = DashboardApp(root, panel, start_threads=False)
    app._load_index()
    app.poll_once()
    assert app.src is None and "NO SAVED SCORES" in app.replay_label()
    save_scores(root, "a1", 2, idx, [330, 335])
    save_scores(root, "a3", 2, idx, [330], flat=True)                 # a dead circuit
    app.poll_once()
    for _ in range(100):
        app.update(0.02)
    t = app.clock
    app.toggle_pause()
    for _ in range(100):
        app.update(0.02)
    assert app.clock == t                                              # frozen while paused
    app.toggle_pause()
    app.next_agent()
    first = app.pin
    app.next_agent()
    assert app.pin != first and app.pin in {"a1", "a2", "a3"}
    app.pin = "a3"
    app.poll_once()
    assert app.src == ("a3", 2) and app.decisions[0].action == "SILENT"
    for _ in range(300):
        app.update(0.02)
    assert any("[FLY SILENT]" in t for _, t in app.view().log)


class _Surf:
    def __init__(self, size=(1, 1), log=None):
        self.log, self.size = log if log is not None else [], size

    def fill(self, c):
        self.log.append(("fill", c))

    def blit(self, img, pos):
        self.log.append(("blit", pos))

    def get_width(self):
        return self.size[0]


class _Draw:
    def __init__(self, log):
        self.log = log

    def __getattr__(self, name):
        def f(surf, color, *a):
            assert isinstance(color, tuple) and len(color) in (3, 4), (name, color)
            self.log.append((name, a))
        return f


class _Font:
    def size(self, s):
        return (8 * len(s), 14)

    def render(self, s, aa, color):
        return _Surf((8 * len(s), 14))


class FakePygame:
    SRCALPHA = 1

    def __init__(self):
        self.log = []
        self.draw = _Draw(self.log)
        self.font = type("F", (), {"SysFont": staticmethod(lambda names, size: _Font())})

    def Surface(self, size, flags=0):
        return _Surf(size, self.log)


def test_pygame_canvas_wrapper_draws_every_state_with_valid_arguments():
    pg = FakePygame()
    cv = PygameCanvas(_Surf(log=pg.log), pg)
    from dashboard.state import Decision
    for act in ("BUY", "SELL", "HOLD", "SILENT"):
        a = FlyAnimator()
        a.push(Decision("2022-03-15", "AAPL", act, 3.0, "a1", 8))
        for _ in range(40):
            a.update(0.02)
        draw_scene(cv, View(header=dict(round=1, universe=499, top_id="a1", top_ic=0.05, ridge_ic=0.02, mode="league"),
                            pose=a.pose(), log=list(a.log), tape=[("AAPL", 1.0), ("MSFT", -1.0)], clock=a.clock,
                            research_lines=["AAPL   Revenues                          383285000000  FY2023 filed 2023-11-03"] * 3))
    kinds = {k for k, _ in pg.log}
    assert {"rect", "ellipse", "line", "blit"} <= kinds
    assert cv.text_width("ABC", 14) == 24


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
            print("ok", n)
    print("all dashboard app tests passed")
