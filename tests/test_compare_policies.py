import numpy as np
import pandas as pd

from experiments.compare_policies import PRE, VARIANTS, summarize, window_rows

N = 20
TICK = [f"S{i:02d}" for i in range(N)]


def frame(days=PRE + 1 + 130, seed=0, spy=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=days)
    cols = TICK + (["SPY"] if spy else [])
    return pd.DataFrame(100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, (days, len(cols))), axis=0), index=dates, columns=cols)


def scores(px, seed=1):
    rng = np.random.default_rng(seed)
    start = PRE                                            # the saved scores begin the day before the first trade
    return pd.DataFrame(rng.normal(size=(len(px) - start, N)), index=px.index[start:], columns=TICK)


def test_every_variant_runs_and_hold_trades_less_than_old():
    px = frame()
    rows = pd.DataFrame(window_rows(px, scores(px), TICK, k=5))
    assert set(rows.variant) == set(VARIANTS) | {"SPY buy&hold"}
    f = rows.set_index("variant").fills
    assert f["hold"] < f["old"] and f["hold plain"] <= f["old"]
    assert (rows.max_drawdown <= 0).all() and rows.fees.ge(0).all()
    assert np.allclose(rows.excess, rows.total_return - rows.bench_return)


def test_etf_variants_are_skipped_without_spy():
    px = frame(spy=False)
    rows = pd.DataFrame(window_rows(px, scores(px), TICK, k=10, etf_ok=False))
    assert not [v for v in set(rows.variant) if "core" in v] and "SPY buy&hold" not in set(rows.variant) and "old" in set(rows.variant)


def test_spy_sleeve_actually_holds_spy():
    from dataclasses import replace
    from accounts.instruments import Universe
    from accounts.sim import run_account
    px = frame()
    acct, _ = run_account(px, scores(px), Universe(stocks=TICK), hold=replace(VARIANTS["hold +core SPY"], k=10))
    assert "SPY" in acct.pos and not [f for f in acct.log if f.status == "REJECTED"]
    v = acct.pos["SPY"] * px["SPY"].iloc[-1] / acct.equity(px.iloc[-1].to_dict())
    assert 0.3 < v < 0.5


def test_first_trade_day_is_the_first_scored_day():
    px = frame()
    from accounts.instruments import Universe
    from accounts.sim import run_account
    acct, _ = run_account(px, scores(px), Universe(stocks=TICK), hold=VARIANTS["hold"])
    assert pd.Timestamp(acct.filled[0].date) == px.index[PRE + 1]


def test_summarize_pairs_with_old_across_rounds():
    rows = []
    for r in range(4):
        for fly in "ab":
            rows += [dict(round=r, fly=fly, variant="old", total_return=0.00 + 0.01 * r, excess=-0.02, max_drawdown=-0.2, fills=300, fees=3000.0),
                     dict(round=r, fly=fly, variant="hold", total_return=0.05 + 0.01 * r + 0.01 * (r % 2), excess=0.03, max_drawdown=-0.1,
                          fills=80, fees=800.0)]
    s = summarize(pd.DataFrame(rows))
    assert list(s.index) == ["old", "hold"] and s.loc["hold", "rounds"] == 4 and s.loc["hold", "rounds_beat_bench"] == 4
    assert abs(s.loc["hold", "vs_old"] - 0.055) < 1e-9 and s.loc["hold", "t_vs_old"] > 5
