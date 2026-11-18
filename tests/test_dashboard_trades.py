import numpy as np
import pandas as pd

from dashboard.trades import load_fwd1, replay_trades


def make(n_days=70, n=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    tick = np.array([f"T{i}" for i in range(n)])
    index = [(d, tick) for d in dates]
    fwd1 = pd.DataFrame(rng.normal(0, 0.01, (n_days, n)), index=dates, columns=tick)
    scores = {i: rng.normal(size=n) for i in range(0, n_days, 5)}
    return index, fwd1, scores


def test_decisions_are_real_fills_long_only():
    index, fwd1, scores = make()
    dec = replay_trades(scores, index, fwd1, "f1", 3, k=5, every=21)
    assert dec and {d.action for d in dec} <= {"BUY", "SELL", "HOLD"}
    assert dec == sorted(dec, key=lambda d: d.when)
    held = {}
    for d in dec:
        if d.action == "BUY":
            held[d.ticker] = True
        if d.action == "SELL":
            assert held.pop(d.ticker, None), "a SELL only ever sells something the fly bought"
    assert all(d.agent == "f1" and d.round == 3 for d in dec)


def test_first_rebalance_buys_k_names_and_scores_are_attached():
    index, fwd1, scores = make()
    dec = replay_trades(scores, index, fwd1, "f1", 0, k=5, every=21)
    first_day = dec[0].when
    buys = [d for d in dec if d.when == first_day and d.action == "BUY"]
    assert len(buys) == 5 and all(np.isfinite(d.score) for d in buys)


def test_hold_is_a_position_kept_not_traded_that_day():
    index, fwd1, scores = make()
    dec = replay_trades(scores, index, fwd1, "f1", 0, k=5, every=21)
    for h in [d for d in dec if d.action == "HOLD"]:
        assert not [d for d in dec if d.when == h.when and d.ticker == h.ticker and d.action != "HOLD"]


def test_dead_circuit_is_silent_and_places_no_trades():
    index, fwd1, _ = make()
    dec = replay_trades({0: np.zeros(30), 5: np.zeros(30)}, index, fwd1, "f1", 0)
    assert [d.action for d in dec] == ["SILENT", "SILENT"]


def test_load_fwd1_reads_only_returns(tmp_path):
    index, fwd1, _ = make(n_days=10, n=4)
    long = fwd1.stack().rename("fwd_ret").to_frame()
    long.index.names = ["date", "ticker"]
    f = tmp_path / "p.pkl"
    long.assign(other=1.0).to_pickle(f)
    got = load_fwd1(f, [d for d, _ in index])
    assert got.shape == fwd1.shape and np.allclose(got.to_numpy(), fwd1.to_numpy())
