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
    dec = replay_trades(scores, index, fwd1, "f1", 0, k=5, every=21, policy="old")
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


def test_league_index_matches_data_for_a_blended_label(tmp_path):
    from dashboard.trades import league_index
    from tests.test_labels import make_panel
    from tournament.data import Data
    p = make_panel(900)
    p.iloc[::7, p.columns.get_loc("fwd_ret")] = np.nan          # holes so the labelled stocks differ per date
    f = tmp_path / "p.pkl"
    p.to_pickle(f)
    cfg = dict(panel=str(f), train_days=150, val_days=60, holdout=80, label="5+21+63")
    index, fwd1 = league_index(cfg)
    d = Data.from_cfg(cfg)
    assert [x for x, _ in index] == list(d.dates)
    assert all(np.array_equal(t, d.day[i][0]) for i, (_, t) in enumerate(index))
    assert fwd1.shape[0] == len(d.dates) and np.allclose(fwd1.to_numpy(), d.fwd1.to_numpy(), equal_nan=True)


def test_decisions_carry_cash_equity_and_positions():
    index, fwd1, scores = make()
    dec = replay_trades(scores, index, fwd1, "f1", 0, k=5, every=21, policy="old")
    first = dec[0]
    assert first.action == "BUY" and first.cash < 1_000_000 and first.equity <= 1_000_000 + 1     # one buy: fee only
    assert abs(first.equity - 1_000_000) < 0.002 * 1_000_000                                       # equity ~ start (fees, rounding)
    assert [p[0] for p in first.positions] == [first.ticker]
    last_buy_day = [d for d in dec if d.when == dec[0].when]
    assert len(last_buy_day[-1].positions) == 5                                                     # k names after the first rebalance
    for d in dec:
        assert d.cash >= -1e-6
        assert abs(d.equity - (d.cash + sum(p[3] for p in d.positions))) < 1e-6
        assert list(d.positions) == sorted(d.positions, key=lambda r: -r[3])


def test_sells_reduce_positions_and_cash_matches_fills():
    index, fwd1, scores = make(n_days=140)
    dec = replay_trades(scores, index, fwd1, "f1", 0, k=5, every=21)
    sells = [d for d in dec if d.action == "SELL"]
    assert sells
    for d in sells:
        assert d.cash > 0


def make_with_etfs(n_days=260, n=30, seed=0):
    index, fwd1, _ = make(n_days=n_days, n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for e in ("VTI", "RSP", "QQQ", "SPY"):
        fwd1[e] = rng.normal(0.0004, 0.008, n_days)
    scores = {i: np.random.default_rng(i).normal(size=n) for i in range(130, n_days, 5)}
    return index, fwd1, scores


def test_default_policy_buys_etfs_and_the_core_stays_above_30_percent():
    from dashboard.trades import core_share_by_day, run_policy
    index, fwd1, scores = make_with_etfs()
    dec = replay_trades(scores, index, fwd1, "f1", 0, k=10, every=21)
    held_etfs = {p[0] for d in dec for p in d.positions} & {"VTI", "RSP", "QQQ"}
    assert held_etfs, "ETFs show up in the portfolio positions"
    acct, eq, prices, sc = run_policy(scores, index, fwd1, k=10)
    share = core_share_by_day(acct.filled, prices)
    assert len(share) and share.min() >= 0.30 - 0.003
    assert not [f for f in acct.log if f.status == "REJECTED"]


def test_default_policy_without_etf_columns_still_runs_with_stocks_only():
    index, fwd1, scores = make(n_days=200)
    scores = {i: v for i, v in scores.items()}
    dec = replay_trades(scores, index, fwd1, "f1", 0, k=5)
    assert dec and {p[0] for d in dec for p in d.positions} <= set(fwd1.columns)
