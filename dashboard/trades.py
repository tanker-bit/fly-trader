"""Real trades for the dashboard: run each fly's $1M long-only account over a saved round and turn its actual fills into the
decision stream the scene plays. Read-only; imports the account engine, never writes to league folders.

BUY  = the account really bought that stock (whole shares, 10 bps fee, cash permitting)
SELL = the account really sold shares it owned (never a short, never a call)
HOLD = a position the account kept through that rebalance
SILENT = constant score (dead circuit), as before; no account is run
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from accounts.etfs import mask_before_listing
from accounts.hold_policy import MIX
from accounts.instruments import APPROVED_ETFS, Universe, START_CASH, is_core
from accounts.league_report import prices_from_returns
from accounts.sim import run_account
from dashboard.state import Decision


def load_fwd1(panel_path, dates) -> pd.DataFrame:
    """Next-day returns, dates x tickers, reindexed to `dates` (the labelled dates the score keys refer to). Only fwd_ret is read."""
    p = str(panel_path)
    df = pd.read_pickle(p) if p.endswith(".pkl") else pd.read_parquet(p, columns=["fwd_ret"])
    return df["fwd_ret"].unstack("ticker").reindex(list(dates))


def _score_of(sc: pd.DataFrame, date, ticker) -> float:
    j = sc.index.searchsorted(pd.Timestamp(date), side="right") - 1
    if j < 0 or ticker not in sc.columns:
        return 0.0
    v = sc.iloc[j].get(ticker)
    return float(v) if v is not None and np.isfinite(v) else 0.0


@dataclass
class TradeDecision(Decision):
    """A Decision that also carries the account right after it: cash, equity (cash + holdings at that day's prices) and the
    positions [(ticker, shares, price, value, pnl_fraction)] sorted by value. The scene ignores the extras; the HUD shows them."""
    cash: float = START_CASH
    equity: float = START_CASH
    positions: tuple = ()


def _snapshot(held, basis, cash, prices_row):
    pos = []
    for t, sh in held.items():
        px = prices_row.get(t)
        px = float(px) if px is not None and np.isfinite(px) else 0.0
        avg = basis[t] / sh if sh > 0 else 0.0
        pos.append((t, sh, px, sh * px, (px / avg - 1.0) if avg > 0 and px > 0 else 0.0))
    pos.sort(key=lambda r: -r[3])
    return cash + sum(r[3] for r in pos), tuple(pos)


PRE = 125       # price days before the first score, so the hold policy's 63-day filters and the first rebalance line up with the window


def run_policy(scores: dict, index, fwd1: pd.DataFrame, policy: str = "hold", hold=None, k: int = 20, every: int = 21,
               cost_bps: float = 10.0, sell_rank=None, pre: int = PRE):
    """Run one fly's account over a saved round. policy 'hold' (default) = the MIX: a 35% broad-ETF core with a hard 30% floor, a
    medium/long-term stock book and a small short-term sleeve; 'old' = the first top-K re-equalising policy. fwd1 may carry ETF columns.
    -> (acct, eq, prices, sc)."""
    idxs = [i for i in sorted(scores) if i < len(index)]
    stocks = [c for c in fwd1.columns if c not in APPROVED_ETFS]
    sc = pd.DataFrame({index[i][0]: pd.Series(np.asarray(scores[i], float), index=index[i][1]) for i in idxs}).T.sort_index()
    if policy == "old":
        win = list(range(idxs[0], idxs[-1] + 1))
        prices = prices_from_returns(fwd1[stocks], win).ffill()
        acct, eq = run_account(prices, sc, Universe(stocks=prices.columns), k=k, every=every, cost_bps=cost_bps, sell_rank=sell_rank)
        return acct, eq, prices, sc
    pre = max(min(pre, idxs[0]), 0)
    pre = max((pre + 1) // 21 * 21 - 1, 0) if pre else 0           # (pre + 1) a multiple of 21 so the first rebalance falls on the window start
    win = list(range(idxs[0] - pre, idxs[-1] + 1))
    prices = mask_before_listing(prices_from_returns(fwd1, win), fwd1.iloc[win])
    h = replace(hold or MIX, k=k)
    acct, eq = run_account(prices, sc, Universe(stocks=[c for c in stocks if c in prices.columns]), cost_bps=cost_bps, hold=h)
    return acct, eq, prices, sc


def core_share_by_day(fills, prices: pd.DataFrame) -> pd.Series:
    """Broad-ETF core as a share of equity at the end of every day from the first fill on (re-built from the fills alone)."""
    held, cash, out = {}, START_CASH, {}
    by_day = {}
    for f in fills:
        by_day.setdefault(pd.Timestamp(f.date), []).append(f)
    if not by_day:
        return pd.Series(dtype=float)
    for day in prices.index[prices.index >= min(by_day)]:
        for f in by_day.get(day, []):
            sign = 1.0 if f.side == "BUY" else -1.0
            held[f.ticker] = held.get(f.ticker, 0.0) + sign * f.shares
            cash -= sign * (f.shares * f.price) + f.fee
        row = prices.loc[day]
        vals = {t: n * float(row[t]) for t, n in held.items() if n > 1e-9 and np.isfinite(row.get(t, np.nan))}
        eq = cash + sum(vals.values())
        out[day] = sum(v for t, v in vals.items() if is_core(t)) / eq if eq > 0 else 0.0
    return pd.Series(out)


def replay_trades(scores: dict, index, fwd1: pd.DataFrame, agent: str, rnd: int, k: int = 20, every: int = 21,
                  cost_bps: float = 10.0, sell_rank=None, policy: str = "hold", hold=None) -> list:
    """{date_idx: score array} -> chronological [TradeDecision] from the account's real fills (BUY/SELL) plus a HOLD per rebalance.
    Each decision carries the cash, equity and positions right after it (ETFs included)."""
    idxs = [i for i in sorted(scores) if i < len(index)]
    if not idxs:
        return []
    if all(np.nanstd(scores[i]) < 1e-9 for i in idxs):
        return [TradeDecision(str(index[i][0].date()), "", "SILENT", float(np.nanmean(scores[i])), agent, rnd) for i in idxs]
    acct, _, prices, sc = run_policy(scores, index, fwd1, policy=policy, hold=hold, k=k, every=every, cost_bps=cost_bps, sell_rank=sell_rank)
    out, held, basis, cash, by_day = [], {}, {}, START_CASH, {}
    for f in acct.filled:
        by_day.setdefault(pd.Timestamp(f.date), []).append(f)
    for day in sorted(by_day):
        row = prices.loc[day].to_dict() if day in prices.index else {}
        traded = set()
        for f in by_day[day]:
            if f.side == "BUY":
                held[f.ticker] = held.get(f.ticker, 0.0) + f.shares
                basis[f.ticker] = basis.get(f.ticker, 0.0) + f.shares * f.price
                cash -= f.shares * f.price + f.fee
            else:
                old = held.get(f.ticker, 0.0)
                basis[f.ticker] = basis.get(f.ticker, 0.0) * (1.0 - f.shares / old) if old > 0 else 0.0
                held[f.ticker] = old - f.shares
                cash += f.shares * f.price - f.fee
                if held[f.ticker] <= 1e-9:
                    held.pop(f.ticker, None)
                    basis.pop(f.ticker, None)
            traded.add(f.ticker)
            eq, pos = _snapshot(held, basis, cash, row)
            out.append(TradeDecision(str(day.date()), f.ticker, f.side, _score_of(sc, day, f.ticker), agent, rnd, cash, eq, pos))
        keep = [t for t in held if t not in traded]
        if keep:
            t = max(keep, key=lambda x: _score_of(sc, day, x))
            eq, pos = _snapshot(held, basis, cash, row)
            out.append(TradeDecision(str(day.date()), t, "HOLD", _score_of(sc, day, t), agent, rnd, cash, eq, pos))
    return out


def league_index(cfg: dict, min_stocks: int = 10):
    """([(date, tickers)], fwd1) aligned EXACTLY with tournament.data.Data for this league config, so a saved score array d<N> maps to
    index[N]. The league's label can be a blend (e.g. 5+21+63) with sector neutrality, which drops different stocks than the plain
    21-day label does, so the old date_ticker_index (21-day label only) does not line up for such leagues. Only fwd_ret (and the
    21-day column when it exists) are read, so this is light enough for the dashboard's background thread."""
    from tournament.data import HORIZON, load_panel, monthly_labels
    from tournament.labels import load_sectors, make_labels

    path, label = cfg["panel"], str(cfg.get("label", "21"))
    if str(path).endswith(".pkl"):
        panel = load_panel(path)
    else:
        cols = ["fwd_ret"]
        try:
            import pyarrow.parquet as pq
            if "fwd_excess_21" in pq.read_schema(path).names:
                cols.append("fwd_excess_21")
        except Exception:
            pass
        panel = pd.read_parquet(path, columns=cols)
    sectors = load_sectors(cfg["sectors"]) if cfg.get("sectors") else None
    y = monthly_labels(panel) if (label == "21" and not sectors) else make_labels(panel, label, sectors)[0]
    y = y[y.notna()]
    index = []
    for d, g in y.groupby(level="date"):
        if len(g) >= min_stocks:
            index.append((d, g.index.get_level_values("ticker").to_numpy()))
    fwd1 = panel["fwd_ret"].unstack("ticker").reindex([d for d, _ in index])
    return index, fwd1
