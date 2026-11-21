"""Check the no-short / no-sell-what-you-don't-own rule on the real fills a fly's account makes.

    python -m experiments.audit_trades --root league_v7 --panel data_fundamentals/panel_hybrid.parquet --round 10
    python -m experiments.audit_trades --root league_v7 --panel ... --round 10 --agent a0089

For every fly with saved scores that round it replays the $1M account and then re-checks the fills from scratch, independent of the
engine: every SELL is covered by shares bought earlier, holdings and cash never go negative, only the universe's stocks traded.
With the default hold policy it also rebuilds the broad-ETF core's share of the account every day from the fills and fails a fly
whose core ever dips under 30%. Rejected orders (the engine refusing a rule-breaking trade) are listed. Prints PASS or FAIL per fly; exit code 1 if any fails.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from accounts.instruments import Universe, START_CASH
from accounts.league_report import prices_from_returns
from accounts.sim import run_account
from dashboard import state, trades

FLOOR = 0.30      # broad-ETF core: at least this share of equity at the end of every day


def audit_fills(fills, universe_tickers) -> list[str]:
    """Problems found in a list of Fill (empty list = clean). Independent of Account's own checks."""
    problems, held, cash = [], {}, START_CASH
    for n, f in enumerate(fills, 1):
        if f.ticker not in universe_tickers:
            problems.append(f"fill {n}: {f.ticker} is not in the universe")
        if f.shares <= 0:
            problems.append(f"fill {n}: non-positive shares {f.shares}")
        if f.side == "BUY":
            held[f.ticker] = held.get(f.ticker, 0.0) + f.shares
            cash -= f.shares * f.price + f.fee
        elif f.side == "SELL":
            if held.get(f.ticker, 0.0) + 1e-9 < f.shares:
                problems.append(f"fill {n}: SELL {f.shares:g} {f.ticker} on {f.date} but only {held.get(f.ticker, 0.0):g} owned (a short)")
            held[f.ticker] = held.get(f.ticker, 0.0) - f.shares
            cash += f.shares * f.price - f.fee
        else:
            problems.append(f"fill {n}: unknown side {f.side}")
        if cash < -1e-6:
            problems.append(f"fill {n}: cash went negative ({cash:,.2f}) on {f.date}")
    if any(v < -1e-9 for v in held.values()):
        problems.append("a holding went negative")
    return problems


def run(scores, index, fwd1, k, every, cost_bps, policy="hold"):
    acct, eq, prices, _ = trades.run_policy(scores, index, fwd1, policy=policy, k=k, every=every, cost_bps=cost_bps)
    return acct, eq, set(prices.columns), prices


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--panel", default=None, help="override the panel path in the league config")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--agent", default=None, help="one fly id (default: every fly with saved scores that round)")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--every", type=int, default=21)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--policy", choices=["hold", "old"], default="hold", help="hold = core ETFs + stock book + sleeve (default); old = first policy")
    ap.add_argument("--etf-root", default=None, help="folder holding prices_etf (default: the panel's folder)")
    a = ap.parse_args()
    from tournament.store import Store
    cfg = Store(a.root).load_config()
    if a.panel:
        cfg["panel"] = a.panel
    print("reading panel (aligned to the league's own label)...", flush=True)
    index, fwd1 = trades.league_index(cfg)
    if a.policy == "hold":
        from pathlib import Path
        from accounts.etfs import with_etfs
        fwd1 = with_etfs(fwd1, a.etf_root or Path(cfg["panel"]).parent)
        if "SPY" not in fwd1.columns:
            print("note: no ETF prices found (run experiments.fetch_etfs): the core floor cannot be tested", flush=True)
    ids = [a.agent] if a.agent else sorted(p.name for p in (__import__("pathlib").Path(a.root) / "agents").iterdir() if p.is_dir())
    bad = 0
    for aid in ids:
        sc = state.load_scores(a.root, a.round, aid)
        if not sc:
            continue
        acct, eq, uni, prices = run(sc, index, fwd1, a.k, a.every, a.cost_bps, a.policy)
        probs = audit_fills(acct.filled, uni)
        share = trades.core_share_by_day(acct.filled, prices) if a.policy == "hold" else None
        core_note = ""
        if share is not None and len(share) and share.max() > 0:
            core_note = f", core ETFs min {share.min():.1%} / max {share.max():.1%} of equity"
            if share.min() < FLOOR - 0.003:
                probs.append(f"the broad-ETF core fell to {share.min():.1%}, below the {FLOOR:.0%} floor")
        sells = sum(f.side == "SELL" for f in acct.filled)
        print(f"{aid}: {len(acct.filled)} fills ({sells} sells), {len(acct.rejected)} rejected, final ${eq.iloc[-1]:,.0f}"
              f"{core_note} -> {'PASS' if not probs else 'FAIL'}")
        for r in acct.rejected[:5]:
            print(f"    rejected: {r.side} {r.shares:g} {r.ticker} on {r.date}: {r.reason}")
        for p in probs[:10]:
            print("    PROBLEM:", p)
        bad += bool(probs)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
