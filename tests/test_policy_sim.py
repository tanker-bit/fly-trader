import numpy as np
import pandas as pd
import pytest

from accounts import Account, Universe
from accounts.policy import pick_top_k, rebalance
from accounts.sim import run_account, metrics


def synth(n=300, m=30, seed=0):
    rnd = np.random.default_rng(seed)
    d = pd.bdate_range("2020-01-01", periods=n)
    t = [f"S{i}" for i in range(m)]
    drift = np.linspace(-0.001, 0.001, m)
    px = pd.DataFrame(100 * np.cumprod(1 + drift + rnd.normal(0, 0.01, (n, m)), axis=0), d, t)
    return px, drift, t


def test_pick_top_k_ignores_nan_and_non_stocks():
    u = Universe(["A", "B", "C"])
    s = pd.Series({"A": 1, "B": 3, "C": np.nan, "SPY": 9, "ZZZ": 8})
    assert pick_top_k(s, u, 2) == ["B", "A"]


def test_rebalance_long_only_and_sells_dropouts():
    u = Universe(["A", "B", "C", "D"])
    a = Account(u)
    px = {"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0}
    rebalance(a, "d1", pd.Series({"A": 4, "B": 3, "C": 2, "D": 1}), px, k=2)
    assert set(a.pos) == {"A", "B"} and a.cash >= 0 and not a.rejected
    rebalance(a, "d2", pd.Series({"A": 1, "B": 2, "C": 4, "D": 3}), px, k=2, sell_rank=2)
    assert set(a.pos) == {"C", "D"} and a.cash >= 0 and not a.rejected     # only ever sold what it held


def test_idle_etf_parks_cash():
    u = Universe(["A"])
    a = Account(u)
    rebalance(a, "d1", pd.Series({"A": 1}), {"A": 100.0, "SHY": 80.0}, k=1, idle_etf="SHY")
    assert a.cash < 0.01 * a.equity() + 100 and "A" in a.pos


def test_sim_never_breaks_rules_and_good_scores_beat_bad():
    px, drift, t = synth()
    good = pd.DataFrame(np.tile(drift, (len(px), 1)), px.index, t)
    bad = -good
    ag, eg = run_account(px, good, k=5, every=21)
    ab, eb = run_account(px, bad, k=5, every=21)
    for a in (ag, ab):
        assert a.cash >= 0 and all(s > 0 for s in a.pos.values()) and not a.rejected
    assert eg.iloc[-1] > eb.iloc[-1]
    m = metrics(eg, px.mean(axis=1) / px.mean(axis=1).iloc[0] * 1e6)
    assert {"total_return", "max_drawdown", "sharpe", "excess_return"} <= set(m)


def test_no_lookahead():
    px, drift, t = synth(n=60)
    sc = pd.DataFrame(np.tile(drift, (len(px), 1)), px.index, t)
    a, _ = run_account(px, sc, k=5, every=21)
    first = a.filled[0]
    assert first.date == str(px.index[21])       # scores from day 20 traded at day 21's price


def test_hysteresis_cuts_turnover_and_stays_long_only():
    px, drift, t = synth(n=400)
    rnd = np.random.default_rng(5)
    noisy = pd.DataFrame(np.tile(drift, (len(px), 1)) + rnd.normal(0, 0.002, px.shape), px.index, t)
    a0, _ = run_account(px, noisy, k=5, every=10, sell_rank=5)
    a1, _ = run_account(px, noisy, k=5, every=10, sell_rank=15)
    assert len(a1.filled) < len(a0.filled) and a1.fees < a0.fees
    assert a1.cash >= 0 and not a1.rejected and len(a1.pos) <= 5
