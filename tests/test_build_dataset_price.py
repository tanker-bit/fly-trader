import numpy as np
import pandas as pd

from experiments.build_dataset_price import build_price_panel, load_all_prices, yearly


def _write(d, t, n=700, seed=0, start="2018-01-01"):
    rnd = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    px = 50 * np.cumprod(1 + rnd.normal(0, 0.01, n))
    pd.DataFrame({"Close": px, "Adj Close": px, "Volume": 1e6, "Stock Splits": 0.0}, index=idx).to_pickle(d / f"{t}.pkl")


def test_price_only_panel_includes_removed_name_only_while_member(tmp_path):
    (tmp_path / "prices").mkdir()
    (tmp_path / "prices_extra").mkdir()
    for i in range(12):
        _write(tmp_path / "prices", f"S{i}", seed=i)
    _write(tmp_path / "prices_extra", "GONE", seed=99)
    _write(tmp_path / "prices_extra", "JUNK", seed=98, n=2)
    pd.DataFrame({"ticker": ["GONE", "JUNK"], "usable": [True, False]}).to_csv(tmp_path / "probe_prices.csv", index=False)
    iv = pd.DataFrame({"symbol": [f"S{i}" for i in range(12)] + ["GONE", "JUNK"],
                       "start": pd.NaT, "end": [pd.NaT] * 12 + [pd.Timestamp("2019-12-31"), pd.NaT]})
    adj, vol, src = load_all_prices(tmp_path, "2000-01-01")
    assert src == {"main": 12, "extra": 1} and "JUNK" not in adj.columns          # stub history is never used
    p = build_price_panel(adj, vol, iv)
    d = p.index.get_level_values("date"); t = p.index.get_level_values("ticker")
    assert "GONE" in set(t) and (d[t == "GONE"] < pd.Timestamp("2020-01-01")).all()   # only while it was a member
    y = yearly(p)
    assert y.stocks.max() == 13 and set(p.columns) >= {"mom_12_1", "fwd_excess_21"}
    assert not any(c in p.columns for c in ["earnings_yield", "roe"])


def test_hybrid_keeps_fundamentals_and_flags_price_only_rows(tmp_path):
    from env import fundamentals as fnd
    (tmp_path / "prices").mkdir()
    for i in range(16):
        _write(tmp_path / "prices", f"S{i}", seed=i)
    iv = pd.DataFrame({"symbol": [f"S{i}" for i in range(16)], "start": pd.NaT, "end": pd.NaT})
    adj, vol, _ = load_all_prices(tmp_path, "2000-01-01")
    rnd = np.random.default_rng(3)
    raw = {m: pd.DataFrame(np.nan, adj.index, adj.columns) for m in fnd.METRICS}
    for m in fnd.METRICS:                                   # S0..S11 have fundamentals (ranking needs >= 10 per day, with headroom), S12..S15 do not
        raw[m][[f"S{i}" for i in range(12)]] = rnd.normal(size=(len(adj), 12))
    p = build_price_panel(adj, vol, iv, raw)
    t = p.index.get_level_values("ticker")
    has = t.isin([f"S{i}" for i in range(12)])
    assert set(t) == {f"S{i}" for i in range(16)}            # price-only names are NOT dropped
    assert (p.loc[has, "fund_known"] == 1.0).all() and (p.loc[~has, "fund_known"] == -1.0).all()
    assert (p.loc[~has, "earnings_yield"] == 0.0).all()      # filler for unknowns
    assert p.loc[t == "S0", "earnings_yield"].abs().sum() > 0   # real ranks where known
