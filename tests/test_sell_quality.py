import numpy as np
import pandas as pd

from accounts.account import Fill
from experiments.sell_quality import fill_table, summarize


def F(side, t, sh, px, d):
    return Fill(date=d, ticker=t, side=side, shares=sh, price=px, fee=0.0, status="FILLED")


def prices(n=300):
    dates = pd.bdate_range("2020-01-01", periods=n)
    base = np.arange(n)
    return pd.DataFrame({
        "AAA": 100.0 + 0.0 * base,            # flat
        "BBB": 100.0 * 1.002 ** base,         # rises 0.2% a day
        "CCC": 100.0 * 0.998 ** base,         # falls 0.2% a day
        "SPY": 100.0 * 1.0005 ** base,
        "VTI": 100.0 * 1.0005 ** base,
    }, index=dates), dates


def test_forward_returns_pnl_and_yardsticks():
    px, d = prices()
    stocks = ["AAA", "BBB", "CCC"]
    fills = [F("BUY", "BBB", 10, float(px.at[d[50], "BBB"]), d[50]), F("BUY", "AAA", 10, 100.0, d[50]),
             F("BUY", "VTI", 10, float(px.at[d[50], "VTI"]), d[50]),
             F("SELL", "BBB", 10, float(px.at[d[100], "BBB"]), d[100])]
    t = fill_table(fills, px, stocks)
    assert set(t.ticker) == {"BBB", "AAA"}                                  # the ETF fill is not a stock fill
    s = t[(t.side == "SELL")].iloc[0]
    assert abs(s.pnl - (1.002 ** 50 - 1)) < 1e-9                             # gain from the average cost
    assert abs(s.fwd21 - (1.002 ** 21 - 1)) < 1e-9
    assert abs(s.spy21 - (1.0005 ** 21 - 1)) < 1e-9
    assert abs(s.kept21 - 0.0) < 1e-9                                        # AAA (flat) is the only other stock kept; VTI is excluded
    assert 0.9 < s.range_pos <= 1.0                                          # a riser sells near its 126-day high


def test_horizon_cut_off_at_the_end_and_summary_shape():
    px, d = prices(120)
    stocks = ["AAA", "BBB", "CCC"]
    fills = [F("BUY", "CCC", 5, float(px.at[d[10], "CCC"]), d[10]), F("SELL", "CCC", 5, float(px.at[d[110], "CCC"]), d[110])]
    t = fill_table(fills, px, stocks)
    sell = t[t.side == "SELL"].iloc[0]
    assert np.isnan(sell.fwd21)                                              # no 21 days left after day 110 of 120
    assert sell.pnl < 0


def test_summary_marks_a_correct_sell_as_good():
    px, d = prices()
    stocks = ["AAA", "BBB", "CCC"]
    rows = []
    for r, fly in ((0, "a"), (0, "b"), (1, "a")):
        fills = [F("BUY", "CCC", 5, float(px.at[d[50], "CCC"]), d[50]), F("SELL", "CCC", 5, float(px.at[d[100], "CCC"]), d[100]),
                 F("BUY", "BBB", 5, float(px.at[d[120], "BBB"]), d[120])]
        rows.append(fill_table(fills, px, stocks).assign(round=r, fly=fly))
    s = summarize(pd.concat(rows, ignore_index=True))
    sells = s.loc[("sells at a loss", 21, "uni")]
    assert sells.excess < 0 and sells.pct_good == 1.0 and sells.n_fly_rounds == 3
    buys = s.loc[("buys", 21, "uni")]
    assert buys.pct_good == 0.5                                              # the CCC buy lost, the BBB buy won
    assert ("sells", 0, "range_pos") in s.index and ("buys", 0, "range_pos") in s.index
