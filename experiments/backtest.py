"""Walk-forward backtest with trading costs: simple signals vs a ridge model.

    python -m experiments.backtest

Needs data/panel.parquet (python -m experiments.build_dataset).

Everything printed is out-of-sample: the ridge model is refit every year using only earlier
data, and the simple signals have no fitted parameters at all.

How to read the tables:
  IC / t      average rank correlation of the score with next-day relative return, and its t-stat
  rebal       hold each portfolio for this many days before rebalancing (1 = daily)
  turn/d      average sum of weight changes per day (gross exposure is 2.0, so the maximum is 4.0
              when every position flips; 0.5 means a quarter of the book turns over daily)
  S@c         Sharpe ratio AFTER paying c basis points (0.01%) per unit traded, one way
  ret%@10     annual return after 10 bps costs;  maxDD% worst peak-to-trough loss
Long-short = long the top 20% of stocks, short the bottom 20% (market-neutral).
Costs are guesses: 5-10 bps is plausible for liquid TSX names; shorting adds borrow fees we
do not model. Survivorship bias (today's big stocks) still flatters every number.
"""
from __future__ import annotations

import pandas as pd

from env.features import feature_columns
from env.portfolio import report, report_long_only
from learning.ridge import walk_forward_ridge

MARKET_COLS = {"mkt_ret_1", "mkt_ret_5", "mkt_ret_20", "vix_z", "vix_chg_5", "usdcad_ret_20", "oil_ret_20"}


def build_signals(panel: pd.DataFrame, min_train_days: int = 1000):
    feats = [c for c in feature_columns(panel) if c not in MARKET_COLS]
    ridge, coefs = walk_forward_ridge(panel, feats, min_train_days=min_train_days)
    first_oos = ridge.dropna().index.get_level_values("date").min()
    un = lambda s: s.unstack("ticker")
    signals = {
        "reversal (-ret_1)": un(-panel["ret_1"]),
        "reversal 5d (-ret_5)": un(-panel["ret_5"]),
        "momentum 12-1": un(panel["mom_12_1"]),
        f"ridge ({len(feats)} feats)": un(ridge),
    }
    signals = {k: v.loc[v.index >= first_oos] for k, v in signals.items()}
    return signals, coefs, first_oos


def main():
    panel = pd.read_parquet("data/panel.parquet")
    signals, coefs, first_oos = build_signals(panel)
    fwd = panel["fwd_ret"].unstack("ticker")
    exc = panel["fwd_excess"].unstack("ticker")
    last = next(iter(signals.values())).index.max()
    print(f"Out-of-sample: {first_oos.date()} -> {last.date()}   "
          f"({signals['momentum 12-1'].shape[0]} days, up to {signals['momentum 12-1'].shape[1]} stocks)\n")

    print("=== Long-short (market-neutral), Sharpe after costs ===")
    report(signals, fwd, exc)

    print("\n=== Long-only top 20% after 10 bps costs vs equal-weight universe ===")
    report_long_only(signals, fwd)

    print("\n=== Ridge coefficients (average over yearly refits; + = higher rank predicts higher return) ===")
    print(coefs.mean().sort_values(key=abs, ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
