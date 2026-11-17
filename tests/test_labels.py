import tempfile

import numpy as np
import pandas as pd

from tournament.data import Data, monthly_labels
from tournament.labels import horizon_excess, make_labels, parse_label, sector_neutral

def make_panel(n_days=400, n_stocks=15, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=n_days),
                                      [f"T{i}" for i in range(n_stocks)]], names=["date", "ticker"])
    df = pd.DataFrame(rng.uniform(-1, 1, (len(idx), 13)), index=idx,
                      columns=[f"f{i}" for i in range(11)] + ["ret_1", "mom_12_1"])
    df["fwd_ret"] = rng.normal(0, 0.01, len(idx))
    df.iloc[-n_stocks:, df.columns.get_loc("fwd_ret")] = np.nan
    df["fwd_excess"] = df.fwd_ret - df.groupby(level="date").fwd_ret.transform("mean")
    return df



def test_parse_label():
    assert parse_label("21") == [21] and parse_label("5+21+63") == [5, 21, 63]


def test_default_label_is_unchanged_and_other_horizons_match_manual():
    p = make_panel()
    y, gap = make_labels(p, "21")
    assert gap == 21 and np.allclose(y.to_numpy(), monthly_labels(p).to_numpy(), equal_nan=True)
    fr = p["fwd_ret"].unstack("ticker")
    y63 = horizon_excess(p, 63).unstack("ticker")
    for t in (0, 40, 150):
        m = (1 + fr.iloc[t:t + 63]).prod() - 1
        assert np.allclose(y63.iloc[t].to_numpy(), (m - m.mean()).to_numpy(), atol=1e-12)
    assert y63.iloc[-62:].isna().all().all()                      # last 62 dates cannot have a 63-day label


def test_sector_neutral_zero_mean_per_sector_and_small_groups_fall_back():
    p = make_panel(n_stocks=15)
    sec = {f"T{i}": ("A" if i < 6 else "B" if i < 12 else "C") for i in range(15)}     # C has only T12..T14 -> 3 stocks
    sec["T14"] = "D"                                                                 # now C has 2 stocks (< 3): falls back
    y = horizon_excess(p, 21)
    n = sector_neutral(y, sec)
    d = pd.DataFrame({"y": y, "n": n, "sec": [sec[t] for t in y.index.get_level_values("ticker")]}).dropna()
    m = d.groupby([d.index.get_level_values("date"), "sec"])["n"].mean()
    assert abs(m.xs("A", level="sec")).max() < 1e-12 and abs(m.xs("B", level="sec")).max() < 1e-12
    small = d[d.sec.isin(["C", "D"])]
    assert np.allclose(small.n, small.y)                          # unchanged (market-neutral only)


def test_blend_is_mean_of_zscores_and_gap_is_longest_horizon():
    p = make_panel(n_days=300)
    y, gap = make_labels(p, "5+21+63")
    assert gap == 63
    zs = []
    for h in (5, 21, 63):
        e = horizon_excess(p, h)
        g = e.groupby(level="date")
        zs.append((e - g.transform("mean")) / (g.transform("std") + 1e-12))
    manual = pd.concat(zs, axis=1).mean(axis=1, skipna=False)
    assert np.allclose(y.to_numpy(), manual.to_numpy(), equal_nan=True)
    assert y.unstack("ticker").iloc[-62:].isna().all().all()


def test_data_uses_label_gap_and_old_configs_still_work():
    p = make_panel(900)
    path = tempfile.mkdtemp() + "/p.pkl"
    p.to_pickle(path)
    cfg = dict(panel=path, train_days=150, val_days=60, holdout=80)
    old = Data.from_cfg(cfg)                                      # no label / sectors keys
    assert old.gap == 21 and old.label == "21"
    new = Data.from_cfg({**cfg, "label": "5+21+63"})
    assert new.gap == 63
    r = 0
    while new.round_valid(r):
        c, v0, v1 = new.round_geometry(r)
        assert v0 - c >= 63 and v1 <= new.n_usable
        r += 1
    assert r >= 2 and len(new.dates) < len(old.dates)             # last 62 dates lose their 63-day label
