r"""The fly's daily autopilot: it reads its own account on the website, decides, and places its own orders. No person in the loop.

    python -m broker.autopilot                    # the full day: refresh prices, read the account, plan, trade, accept
    python -m broker.autopilot --preview-only     # the same, but stops at each order's preview (nothing is submitted)

One run, in order:
  1. read the account on the website (broker.site_account): cash, holdings with purchase prices, open and failed orders;
     cash + holdings must add up to the account value, or the run stops;
  2. if any order is still OPEN on the website (e.g. placed after the close and waiting for the next open), do not trade today:
     new orders could double them;
  3. refresh prices from Yahoo into data_live\prices (after 16:15 New York time today's bar is used, before that yesterday's);
  4. score the S&P 500 with the league's kept flies and plan one hold-policy day on the real account (broker.daily_run);
  5. refuse to trade on stale prices (more than 4 days old) or if this price date was already traded;
  6. place the orders on the website (broker.trade_ui): sells first, each checked against its preview, the live price (5% limit)
     and the buying power; a file orders\STOP halts it between orders;
  7. accept the day's policy memory when nothing went wrong.
Everything is appended to orders\autopilot.log. Run it once per trading day after the close (see run_autopilot.cmd and the handover
for a Windows scheduled task). Reminder: no tradable edge has been shown; this is a play-money research run.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

from broker import daily_run as dr
from broker.live_feed import Feed
from broker import site_account as sa
from broker import trade_ui as tu


class Log:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg=""):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')}  {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def already_traded(day_dir: Path) -> bool:
    f = day_dir / "trade_log.csv"
    if not f.exists():
        return False
    with open(f, newline="", encoding="utf-8") as fh:
        return any(r.get("status") == "submitted" for r in csv.DictReader(fh))


def snapshot(acct: sa.SiteAccount, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(cash=acct.cash, account_value=acct.account_value, buying_power=acct.buying_power,
                                    positions=acct.positions, open_orders=len(acct.open_orders)), indent=1), encoding="utf-8")
    return path


def run(a, log, browser=tu.browser, today=None, feed=None) -> str:
    """-> 'traded' | 'previewed' | 'no orders' | 'skipped: <why>'."""
    out = Path(a.out)
    profile = Path(a.profile).resolve()
    feed = feed or Feed(out / "live_feed.jsonl")
    if (out / "STOP").exists():
        return f"skipped: {out / 'STOP'} exists"
    feed("start", mode="preview" if a.preview_only else "trading")
    feed("phase", text="READING ACCOUNT ON THE WEBSITE")

    # 1-2) the account, as the website shows it
    pw, ctx, page = browser(profile, a.headless)
    try:
        acct = sa.read_account(page, log=log)
    finally:
        ctx.close()
        pw.stop()
    feed("account", equity=acct.account_value, cash=acct.cash, buying_power=acct.buying_power,
         positions=[[p["symbol"], p["quantity"], p.get("current_price"), p.get("total_value"),
                     (p["current_price"] / p["purchase_price"] - 1.0) if p.get("current_price") and p.get("purchase_price") else 0.0]
                    for p in acct.positions])
    if acct.failed:
        log(f"the website lists {len(acct.failed)} failed trade(s): " + "; ".join(
            f"{f.get('transaction', '?')} {f.get('quantity', '?')} {f.get('symbol', '?')} ({f.get('status', '')})" for f in acct.failed[:10]))
    if acct.open_orders:
        feed("done", text="OPEN ORDERS WAITING: NO TRADING TODAY")
        return "skipped: open orders are still waiting on the website (" + ", ".join(
            f"{o.get('transaction', '?')} {o.get('quantity', '?')} {o.get('symbol', '?')}" for o in acct.open_orders[:10]) + ")"
    snap = snapshot(acct, out / "web_account.json")

    # 3-4) prices, scores, plan
    args = ["--portfolio", str(snap), "--out", str(out), "--state", str(a.state), "--league", a.league, "--flies", a.flies,
            "--data", a.data, "--live-prices", a.live_prices]
    if not a.no_refresh:
        args.append("--refresh-prices")
    if a.offered_only:
        args.append("--offered-only")
    feed("phase", text="SCORING THE S&P 500" if a.no_refresh else "REFRESHING PRICES AND SCORING")
    res = dr.run_day(dr.parser().parse_args(args), today=today, log=log)
    day_dir, plan = Path(res["out"]), res["plan"]
    feed("phase", text=f"PLANNED {len(plan.orders)} ORDERS", asof=res["asof"], n_flies=len(res.get("flies", [])))

    # 5) refuse what must not be traded
    if any(n.startswith("WARNING") for n in res["notes"]) and not a.allow_stale:
        return f"skipped: prices are stale ({res['asof']}); nothing was traded"
    if already_traded(day_dir):
        return f"skipped: orders for {res['asof']} were already submitted (see {day_dir / 'trade_log.csv'})"
    if not plan.orders:
        if not a.preview_only:
            dr.accept(out, res["asof"], a.state, log=log)              # a quiet day still moves the policy memory forward
        return "no orders"

    # 6) the website
    orders = tu.read_orders(day_dir / "orders.csv")
    pw, ctx, page = browser(profile, a.headless)
    try:
        site = tu.TradeSite(page, shots=day_dir / "screens", log=log, emit=feed)
        results = tu.run_orders(site, orders, a.preview_only, a.max_drift, out / "STOP", log=log)
    finally:
        ctx.close()
        pw.stop()
    tu.write_log(day_dir / "trade_log.csv", results)
    counts = {s: sum(r["status"] == s for r in results) for s in ("submitted", "skipped", "previewed", "error", "stopped")}
    log(f"orders: {counts}")
    feed("done", text=" ".join(f"{k.upper()} {v}" for k, v in counts.items() if v))

    # 7) memory
    if a.preview_only:
        return "previewed"
    if tu.should_accept(results):
        dr.accept(out, res["asof"], a.state, log=log)
    else:
        log("the day was NOT accepted (errors, a stop, or nothing went through); check the log, then broker.daily_run --accept")
    return "traded"


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview-only", action="store_true", help="stop at each order's preview; submit nothing")
    ap.add_argument("--no-refresh", action="store_true", help="use the price files as they are")
    ap.add_argument("--allow-stale", action="store_true", help="trade even if the newest prices are more than 4 days old")
    ap.add_argument("--offered-only", action="store_true")
    ap.add_argument("--max-drift", type=float, default=0.05)
    ap.add_argument("--league", default="league_v8")
    ap.add_argument("--flies", default="kept")
    ap.add_argument("--data", default="data_fundamentals")
    ap.add_argument("--live-prices", default="data_live/prices")
    ap.add_argument("--out", default="orders")
    ap.add_argument("--state", default="orders/policy_state.json")
    ap.add_argument("--profile", default=str(tu.DEFAULT_PROFILE))
    ap.add_argument("--headless", action="store_true", help="no visible window (for an unattended machine)")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    log = Log(Path(a.out) / "autopilot.log")
    log(f"autopilot start ({'PREVIEW ONLY' if a.preview_only else 'trading'})")
    try:
        result = run(a, log)
    except Exception as e:
        log(f"autopilot FAILED: {type(e).__name__}: {e}")
        raise SystemExit(1)
    log(f"autopilot done: {result}")


if __name__ == "__main__":
    main()
