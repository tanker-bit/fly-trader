r"""V4d: the daily DRY RUN. Scores today's S&P 500 stocks with the league's flies and writes the orders the hold policy would place
at the next market open, for a person to review. It NEVER places an order (no trade code is imported; that is V5).

    python -m broker.daily_run                                   # fresh $100,000 account, local price files as they are
    python -m broker.daily_run --refresh-prices                  # first download fresh Yahoo prices into data_live\prices
    python -m broker.daily_run --portfolio game_snapshot.json    # plan against a saved copy of the game account
    python -m broker.daily_run --portfolio live --lib-dir D:\github\investopedia_simulator_api     (needs node token_keeper.mjs running)
    python -m broker.daily_run --accept 2026-11-06               # AFTER you placed that day's orders: make its policy memory the real one

Steps: (1) optional price refresh (Yahoo; into a separate folder, the research caches are never rewritten); (2) today's feature rows
(tournament.live_features: the newest day with prices for 95% of the members); (3) scores from the league's kept flies and their
committee (tournament.live); (4) the game account: a fresh one, a snapshot file, or read live through the bridge (read-only);
(5) one day of the hold policy (accounts.order_plan, MIX preset) on a shadow copy of that account, buying only stocks the game is not
known to refuse; (6) files in <out>\<date>\: orders.csv, plan.txt, scores.csv, run.json, state_after.json.

The real policy memory (--state, default orders\policy_state.json) is only replaced by --accept, so a dry run never makes tomorrow
believe today's orders happened. Availability: data_us\game_availability.csv says yes / no per stock; stocks never checked are
'unknown' and allowed unless --offered-only. Reminder: the research found no tradable edge; this is play money.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import pandas as pd

from accounts.hold_policy import MIX
from accounts.instruments import APPROVED_ETFS
from accounts.order_plan import orders_frame, plan_day, plan_text
from accounts.policy_state import load_state, save_state
from broker import game_portfolio as gp
from broker.live_prices import load_frames, overlay, refresh_prices

STALE_DAYS = 4          # calendar days between the price date and today before the run warns


# ---------------------------------------------------------------- market data
def add_frames(inp, adj, cs, vol, spl):
    """Inputs with extra price columns (e.g. ETFs) added where the research caches have none."""
    from dataclasses import replace
    idx = inp.adj.index.union(adj.index)

    def put(base, new):
        b = base.reindex(idx)
        for c in new.columns:
            if c not in base.columns:
                b[c] = new[c].reindex(idx)
        return b

    return replace(inp, adj=put(inp.adj, adj), close_split=put(inp.close_split, cs), volume=put(inp.volume, vol),
                   splits={**spl, **inp.splits})


def load_market(root, live_dir=None, etfs=APPROVED_ETFS):
    from tournament.live_features import load_inputs
    inp = load_inputs(root)
    etf_dir = Path(root) / "prices_etf"
    if etf_dir.exists():
        inp = add_frames(inp, *load_frames(etfs, etf_dir))
    return overlay(inp, live_dir) if live_dir else inp


def planner_prices(inp, tickers, asof, days: int = 300) -> pd.DataFrame:
    """Adjusted closes up to asof for the policy's guards; the LAST row is the real close of asof (split-adjusted only), because the
    orders are sized with it and the stops compare it with real purchase prices."""
    cols = [t for t in dict.fromkeys(tickers) if t in inp.adj.columns]
    px = inp.adj.loc[:asof, cols].tail(days).copy()
    real = inp.close_split.loc[asof, cols]
    px.iloc[-1] = real.where(real.notna(), px.iloc[-1])
    return px


def buyable_set(avail_csv, symbols, offered_only=False):
    """-> (set of our symbols allowed to be bought, counts). 'no' is always excluded; 'unknown' only with offered_only."""
    status = {}
    p = Path(avail_csv) if avail_csv else None
    if p and p.exists():
        d = pd.read_csv(p, keep_default_na=False, dtype=str)
        status = dict(zip(d["symbol"].str.upper(), d["status"].str.lower()))
    yes = {s for s in symbols if status.get(s) == "yes"}
    no = {s for s in symbols if status.get(s) == "no"}
    unknown = set(symbols) - yes - no
    allowed = yes if offered_only else yes | unknown
    return allowed, dict(yes=len(yes), no=len(no), unknown=len(unknown), offered_only=bool(offered_only))


# ---------------------------------------------------------------- the run
def run_day(a, today=None, log=print) -> dict:
    from tournament.live import COMMITTEE, load_flies, pick_flies, score_rows
    from tournament.live_features import _members_on, feature_rows
    from tournament.store import Store
    today = today or dt.date.today()
    params = MIX
    core = [t for t, _ in params.core]

    # 1) the game account first: its holdings decide which prices must be refreshed
    if a.portfolio == "fresh":
        acc = gp.fresh(a.cash)
    elif a.portfolio == "live":
        if not a.lib_dir:
            raise SystemExit("--portfolio live needs --lib-dir (the investopedia_simulator_api folder)")
        acc = gp.read_live(a.lib_dir, a.game_name)
        if a.save_snapshot:
            Path(a.save_snapshot).write_text(json.dumps(dict(cash=acc.holdings.cash, positions=[
                dict(symbol=gp.to_game(t), quantity=q, purchase_price=acc.holdings.avg_cost.get(t),
                     current_price=acc.holdings.prices.get(t)) for t, q in acc.holdings.shares.items()]), indent=1))
    else:
        acc = gp.read_snapshot(a.portfolio)
    log(f"account: {acc.source}: cash {acc.holdings.cash:,.2f}, {len(acc.holdings.shares)} positions")

    # 2) prices
    if a.refresh_prices:
        base = load_market(a.data)
        members = _members_on(base, base.adj.index[-1])
        refresh_prices(set(members) | set(core) | set(acc.holdings.shares), a.live_prices, log=log)
    inp = load_market(a.data, a.live_prices)

    # 3) features and scores
    feats = feature_rows(inp, asof=a.asof, log=log)
    asof = feats.asof
    store = Store(a.league)
    cfg = store.load_config()
    which = a.flies if a.flies in ("kept", "best") else [x.strip() for x in a.flies.split(",") if x.strip()]
    flies = load_flies(store, cfg, pick_flies(store, which))
    log(f"scoring {len(feats.rows)} stocks with {len(flies)} flies ({', '.join(f.id for f in flies)}) ...")
    sc = score_rows(flies, feats.rows, cfg["feats"])

    # 4) plan
    allowed, avail = buyable_set(a.availability, list(sc.index), a.offered_only)
    prices = planner_prices(inp, list(sc.index) + core + list(acc.holdings.shares), asof)
    state = load_state(a.state)
    plan = plan_day(prices, sc[COMMITTEE], acc.holdings, params, state=state, buyable=allowed)

    notes = list(acc.notes) + list(plan.notes)
    age = (today - asof.date()).days
    if age > STALE_DAYS:
        notes.insert(0, f"WARNING: prices are from {asof.date()}, {age} days before today ({today}). Refresh them "
                        f"(--refresh-prices) before acting on this list.")
    if feats.no_price_today:
        notes.append(f"no price on {asof.date()} for {', '.join(feats.no_price_today)}: scored on the last known price "
                     f"(halted, delisted or a data gap)")
    if feats.skipped_dates:
        notes.append(f"newer dates with too few prices were skipped: {', '.join(str(d.date()) for d in feats.skipped_dates)}")
    if avail["unknown"] and not a.offered_only:
        notes.append(f"{avail['unknown']} scored stocks were never checked against the game (allowed); run broker.availability "
                     f"or use --offered-only")
    plan.notes = notes

    # 5) files
    out = Path(a.out) / str(asof.date())
    out.mkdir(parents=True, exist_ok=True)
    orders_frame(plan, gp.to_game).to_csv(out / "orders.csv", index=False)
    text = plan_text(plan)
    (out / "plan.txt").write_text(text + "\n", encoding="utf-8")
    sc.assign(game_symbol=[gp.to_game(t) for t in sc.index], buyable=[t in allowed for t in sc.index]) \
      .sort_values(COMMITTEE, ascending=False).to_csv(out / "scores.csv", index_label="symbol")
    save_state(out / "state_after.json", plan.state)
    meta = dict(asof=str(asof.date()), run_on=str(today), league=str(a.league), flies=[f.id for f in flies], mode=plan.mode,
                orders=len(plan.orders), account=acc.source, before=plan.before, after=plan.after, coverage=feats.coverage,
                dropped=feats.dropped, no_price_today=feats.no_price_today, availability=avail, state_used=str(a.state) if state else None, notes=notes)
    (out / "run.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    log("\n" + text)
    log(f"\nfiles in {out}. After placing these orders in the game, run:  python -m broker.daily_run --accept {asof.date()}")
    return dict(meta, out=str(out), plan=plan)


def accept(out_root, day, state_path, log=print) -> Path:
    """Make <out>/<day>/state_after.json the real policy memory (after that day's orders were placed)."""
    src = Path(out_root) / str(day) / "state_after.json"
    if not src.exists():
        raise SystemExit(f"{src} not found: run the dry run for {day} first")
    new = load_state(src)
    cur = load_state(state_path)
    if cur is not None and cur.asof and new.asof and pd.Timestamp(cur.asof) > pd.Timestamp(new.asof):
        raise SystemExit(f"the current policy memory is from {cur.asof}, newer than {new.asof}: refusing to go back")
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, Path(state_path))
    log(f"policy memory updated from {src} -> {state_path} (as of {new.asof})")
    return Path(state_path)


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", default="league_v8")
    ap.add_argument("--flies", default="kept", help="kept | best | comma-separated fly ids")
    ap.add_argument("--data", default="data_fundamentals", help="research caches: prices, prices_etf, universe, sec")
    ap.add_argument("--live-prices", default="data_live/prices", help="folder for refreshed prices (never the research caches)")
    ap.add_argument("--refresh-prices", action="store_true", help="download fresh Yahoo prices first (a few minutes)")
    ap.add_argument("--portfolio", default="fresh", help="fresh | live | path to a snapshot .json")
    ap.add_argument("--cash", type=float, default=100_000.0, help="cash of a fresh account (the game starts with 100,000)")
    ap.add_argument("--lib-dir", default=None)
    ap.add_argument("--game-name", default=None)
    ap.add_argument("--save-snapshot", default=None, help="with --portfolio live: also save the account to this .json")
    ap.add_argument("--availability", default="data_us/game_availability.csv")
    ap.add_argument("--offered-only", action="store_true", help="buy only stocks the game is known to offer")
    ap.add_argument("--asof", default=None, help="score this date instead of the newest complete one")
    ap.add_argument("--out", default="orders")
    ap.add_argument("--state", default="orders/policy_state.json", help="the real policy memory (changed only by --accept)")
    ap.add_argument("--accept", default=None, metavar="DATE", help="make that day's state_after.json the real policy memory")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    if a.accept:
        accept(a.out, a.accept, a.state)
        return
    run_day(a)


if __name__ == "__main__":
    main()
