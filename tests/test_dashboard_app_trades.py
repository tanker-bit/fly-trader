import importlib
import sys
import types

import numpy as np
import pandas as pd


def load_app(monkeypatch):
    scene = types.ModuleType("dashboard.scene")
    scene.FlyAnimator = type("FlyAnimator", (), {"__init__": lambda self: None})
    scene.View = object
    monkeypatch.setitem(sys.modules, "dashboard.scene", scene)
    monkeypatch.delitem(sys.modules, "dashboard.app", raising=False)
    return importlib.import_module("dashboard.app")


def fake_app(app_mod, monkeypatch, fwd1):
    a = app_mod.DashboardApp.__new__(app_mod.DashboardApp)
    import threading
    a.league_root, a.pin, a._lock = "x", None, threading.Lock()
    a.index, a.fwd1, a.real, a.src, a.decisions, a.cursor, a.last_decision = None, fwd1, False, None, [], 0, None
    a.index_error = None
    a.snap = types.SimpleNamespace(top=types.SimpleNamespace(id="f1"), last_done_round=0, round=0, agents=[])
    return a


def data(n_days=60, n=20):
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    tick = np.array([f"T{i}" for i in range(n)])
    index = [(d, tick) for d in dates]
    fwd1 = pd.DataFrame(rng.normal(0, 0.01, (n_days, n)), index=dates, columns=tick)
    return index, fwd1, {i: rng.normal(size=n) for i in range(0, n_days, 5)}


def test_real_trades_used_and_labelled(monkeypatch):
    app = load_app(monkeypatch)
    index, fwd1, sc = data()
    monkeypatch.setattr(app.state, "load_scores", lambda root, r, aid: sc)
    a = fake_app(app, monkeypatch, fwd1)
    a.index = index
    a._refresh_replay(a.snap)
    assert a.real and a.decisions and {d.action for d in a.decisions} <= {"BUY", "SELL", "HOLD"}
    assert "real long-only account fills" in a.replay_label() and "no real money" in a.replay_label()


def test_falls_back_to_score_replay_without_returns(monkeypatch):
    app = load_app(monkeypatch)
    index, _, sc = data()
    monkeypatch.setattr(app.state, "load_scores", lambda root, r, aid: sc)
    a = fake_app(app, monkeypatch, None)
    a.index = index
    a._refresh_replay(a.snap)
    assert not a.real and a.decisions
    assert "no real trades" in a.replay_label()


def test_a_failing_account_run_falls_back(monkeypatch):
    app = load_app(monkeypatch)
    index, fwd1, sc = data()
    monkeypatch.setattr(app.state, "load_scores", lambda root, r, aid: sc)
    monkeypatch.setattr(app.trades, "replay_trades", lambda *a, **k: 1 / 0)
    a = fake_app(app, monkeypatch, fwd1)
    a.index = index
    a._refresh_replay(a.snap)
    assert not a.real and a.decisions
