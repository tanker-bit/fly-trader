"""Portfolio backtest with trading costs, information coefficient, and performance stats.

Scores are DataFrames (dates x tickers): the score on date t is known at the close of t, the
position is held for the next day, and `fwd` holds the next-day return of each stock.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_ic(score: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """Spearman rank correlation between score and target across stocks, for each day."""
    cols = score.columns.intersection(target.columns)
    a, b = score[cols].rank(axis=1), target.reindex(score.index)[cols].rank(axis=1)
    ok = a.notna() & b.notna()
    a, b = a.where(ok), b.where(ok)
    a, b = a.sub(a.mean(axis=1), axis=0), b.sub(b.mean(axis=1), axis=0)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    return (num / den).replace([np.inf, -np.inf], np.nan).dropna()


def target_weights(s: np.ndarray, q: float, mode: str) -> np.ndarray:
    """Equal-weight the top q of stocks (long) and, for long_short, the bottom q (short)."""
    w = np.zeros(len(s))
    valid = ~np.isnan(s)
    n = int(valid.sum())
    if n < 10:
        return w
    k = max(1, int(round(q * n)))
    order = np.argsort(np.where(valid, s, -np.inf))  # invalid first, then ascending
    first = len(s) - n
    w[order[-k:]] = 1.0 / k
    if mode == "long_short":
        w[order[first:first + k]] = -1.0 / k
    return w


def backtest(score: pd.DataFrame, fwd: pd.DataFrame, q: float = 0.2, rebalance_every: int = 1,
             cost_bps: float = 10.0, mode: str = "long_short") -> pd.DataFrame:
    """Daily gross / net returns and turnover. cost_bps is charged per unit of weight traded."""
    cols = score.columns.intersection(fwd.columns)
    S = score[cols].to_numpy(float)
    R = fwd.reindex(index=score.index, columns=cols).to_numpy(float)
    w = np.zeros(len(cols))
    gross, turn = np.zeros(len(S)), np.zeros(len(S))
    for i in range(len(S)):
        if i % rebalance_every == 0:
            nw = target_weights(S[i], q, mode)
            turn[i] = np.abs(nw - w).sum()
            w = nw
        gross[i] = (w * np.nan_to_num(R[i])).sum()
    net = gross - turn * cost_bps / 1e4
    return pd.DataFrame({"gross": gross, "net": net, "turnover": turn}, index=score.index)


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    ann, vol = r.mean() * TRADING_DAYS, r.std() * np.sqrt(TRADING_DAYS)
    eq = (1 + r).cumprod()
    return {"ann_ret": ann, "vol": vol, "sharpe": ann / vol if vol > 0 else np.nan,
            "max_dd": float((eq / eq.cummax() - 1).min())}


def report(signals: dict[str, pd.DataFrame], fwd: pd.DataFrame, excess: pd.DataFrame,
           rebalances=(1, 5, 21), costs=(0, 5, 10, 20), q: float = 0.2) -> None:
    """Print IC and long-short Sharpe at several trading costs for each signal."""
    head = "".join(f"{'S@' + str(c):>7}" for c in costs)
    print(f"{'signal':<22}{'IC':>7}{'t':>6}{'rebal':>6}{'turn/d':>7}{head}{'ret%@10':>9}{'maxDD%':>8}")
    for name, sc in signals.items():
        ic = daily_ic(sc, excess)
        ic_m = ic.mean()
        ic_t = ic_m / ic.std() * np.sqrt(len(ic)) if len(ic) > 1 and ic.std() > 0 else np.nan
        for reb in rebalances:
            res = {c: backtest(sc, fwd, q, reb, c) for c in costs}
            sharpes = "".join(f"{perf(res[c]['net'])['sharpe']:>7.2f}" for c in costs)
            ref = res[10] if 10 in res else res[costs[-1]]
            p = perf(ref["net"])
            print(f"{name:<22}{ic_m:>7.3f}{ic_t:>6.1f}{reb:>6}{ref['turnover'].mean():>7.2f}"
                  f"{sharpes}{100 * p['ann_ret']:>9.1f}{100 * p['max_dd']:>8.1f}")


def report_long_only(signals: dict[str, pd.DataFrame], fwd: pd.DataFrame, cost_bps: float = 10.0,
                     rebalances=(1, 5, 21), q: float = 0.2) -> None:
    """Long-only top-q portfolio vs the equal-weight universe (what a no-shorting account can do)."""
    bench = perf(fwd.reindex(signals[next(iter(signals))].index).mean(axis=1))
    print(f"equal-weight universe: ann return {100 * bench['ann_ret']:.1f}%, Sharpe {bench['sharpe']:.2f}")
    print(f"{'signal':<22}{'rebal':>6}{'ann%':>8}{'Sharpe':>8}{'vs univ%':>10}")
    for name, sc in signals.items():
        for reb in rebalances:
            p = perf(backtest(sc, fwd, q, reb, cost_bps, "long_only")["net"])
            print(f"{name:<22}{reb:>6}{100 * p['ann_ret']:>8.1f}{p['sharpe']:>8.2f}"
                  f"{100 * (p['ann_ret'] - bench['ann_ret']):>10.1f}")
