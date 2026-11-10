"""Cost-aware long-short and long-only portfolios from scores.

Each decision date i: rank stocks by score, go long the top fraction q and short the bottom fraction q, equal weight,
hold for the label horizon (21 days, the label's own window), so returns are the stocks' forward 21-day EXCESS returns
and consecutive decisions never overlap. Costs: `bps` per unit of one-way turnover on each leg (the first date pays for
building the whole book)."""
from __future__ import annotations

import numpy as np


def _legs(s, y, q):
    ok = np.isfinite(s) & np.isfinite(y)
    idx = np.where(ok)[0]
    k = max(1, int(q * len(idx)))
    o = idx[np.argsort(s[idx], kind="stable")]
    return o[-k:], o[:k]


def long_short(signal, labels, tickers, dates, q=0.1, bps=(0, 10, 25), periods_per_year=12):
    """Returns {bps: dict(mean_m, sharpe, maxdd, turnover, long_only_excess)} plus 'gross_by_date'."""
    rows, prev_l, prev_s = [], None, None
    for i in dates:
        s, y, tk = np.asarray(signal[i], float), np.asarray(labels[i], float), np.asarray(tickers[i])
        L, S = _legs(s, y, q)
        lt, st = set(tk[L]), set(tk[S])
        tl = 1.0 if prev_l is None else 1 - len(lt & prev_l) / max(len(lt), 1)
        ts = 1.0 if prev_s is None else 1 - len(st & prev_s) / max(len(st), 1)
        rows.append((y[L].mean(), y[S].mean(), tl, ts))
        prev_l, prev_s = lt, st
    r = np.array(rows)
    out = {}
    for b in bps:
        net = r[:, 0] - r[:, 1] - b * 1e-4 * (r[:, 2] + r[:, 3])
        cum = np.cumsum(net)
        out[b] = dict(mean_m=float(net.mean()), sharpe=float(net.mean() / net.std(ddof=1) * np.sqrt(periods_per_year))
                      if net.std(ddof=1) > 0 else np.nan, maxdd=float((cum - np.maximum.accumulate(cum)).min()),
                      turnover=float(r[1:, 2:].mean()) if len(r) > 1 else np.nan,
                      long_only_excess=float((r[:, 0] - b * 1e-4 * r[:, 2]).mean()))
    out["gross_by_date"] = r[:, 0] - r[:, 1]
    return out
