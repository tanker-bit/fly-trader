"""Run: python -m tests.test_features   (or pytest)

Checks the one rule that matters: features at date t must not depend on anything after t.
"""
import numpy as np
import pandas as pd

from env.features import build_panel, feature_columns


def make_data(n_days=700, n_stocks=25, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    tickers = [f"S{i}.TO" for i in range(n_stocks)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (n_days, n_stocks)), axis=0)),
                         index=dates, columns=tickers)
    volume = pd.DataFrame(rng.integers(1e5, 1e6, (n_days, n_stocks)).astype(float), index=dates, columns=tickers)
    market = pd.DataFrame({
        "tsx": 20000 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n_days))),
        "vix": 18 + np.cumsum(rng.normal(0, 0.3, n_days)).clip(-8, 20),
        "usdcad": 1.3 * np.exp(np.cumsum(rng.normal(0, 0.004, n_days))),
        "oil": 70 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days))),
    }, index=dates)
    return close, volume, market


def test_no_lookahead():
    close, volume, market = make_data()
    full = build_panel(close, volume, market)
    cut = close.index[500]
    part = build_panel(close.loc[:cut], volume.loc[:cut], market.loc[:cut])
    feats = feature_columns(full)
    common = part.index.intersection(full.index)
    assert len(common) > 1000, "test is vacuous"
    a, b = full.loc[common, feats], part.loc[common, feats]
    diff = (a - b).abs().max().max()
    assert diff < 1e-9, f"features changed when the future was removed (max diff {diff})"


def test_label_is_next_day():
    close, volume, market = make_data()
    p = build_panel(close, volume, market)
    d, t = p.index[len(p) // 2]
    pos = close.index.get_loc(d)
    expect = close[t].iloc[pos + 1] / close[t].iloc[pos] - 1
    assert abs(p.loc[(d, t), "fwd_ret"] - expect) < 1e-12


def test_ranks_in_range():
    close, volume, market = make_data()
    p = build_panel(close, volume, market)
    assert p["ret_5"].between(-1, 1).all()


if __name__ == "__main__":
    test_no_lookahead(); test_label_is_next_day(); test_ranks_in_range()
    print("all feature tests passed")
