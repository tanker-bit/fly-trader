"""Build the US (S&P 500, point-in-time) + fundamentals panel into its own folder. Existing data\\ and league_data\\ are untouched.

    python -m experiments.build_dataset_fund --user-agent "Your Name you@example.com" --limit 5      # dry run, 5 tickers
    python -m experiments.build_dataset_fund --user-agent "Your Name you@example.com"               # full build
    python -m experiments.build_dataset_fund --step panel                                           # rebuild only the panel

Steps (all resumable; every download is cached): universe -> sec -> prices -> panel.
SEC needs a User-Agent with your name and email (also read from the SEC_USER_AGENT environment variable); about 500 requests at
<= 7 per second. Yahoo takes roughly 10-15 minutes. Output: <root>/panel.parquet and <root>/coverage_report.txt.

Not tested against the live services (the development sandbox cannot reach SEC, Wikipedia or Yahoo): run with --limit 5 first.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from env import data_us, fundamentals as fnd, universe_us as uni
from env.features import FUND_FEATURES, build_panel


def step_universe(a, root):
    udir = root / "universe"
    udir.mkdir(parents=True, exist_ok=True)
    cur_f, chg_f = udir / "constituents.csv", udir / "changes.csv"
    if not (cur_f.exists() and chg_f.exists()) or a.refresh:
        if a.wiki_html:
            print(f"reading the S&P 500 tables from your saved page {a.wiki_html} ...")
            html = Path(a.wiki_html).read_text(encoding="utf-8", errors="replace")
        else:
            print("fetching the Wikipedia S&P 500 tables ...")
            html = uni.fetch_text(uni.WIKI_URL, a.user_agent)
        try:
            cur, chg = uni.parse_wikipedia(html)
        except ValueError:
            (udir / "wiki_page.html").write_text(html, encoding="utf-8")
            print(f"saved what Wikipedia returned to {udir / 'wiki_page.html'} (open it in a browser to see what it is)")
            raise
        if chg.empty and not a.wiki_html:
            for url in uni.HIST_URLS:
                try:
                    page = uni.fetch_text(url, a.user_agent)
                except Exception as e:
                    print(f"  could not fetch {url}: {e}")
                    continue
                (udir / "history_page.html").write_text(page, encoding="utf-8")
                chg = uni.parse_changes(page)
                print(f"  {url}: {len(chg)} change rows")
                if len(chg):
                    break
        if chg.empty:
            print("WARNING: no index-change history found. Membership falls back to TODAY's members only, which is "
                  "survivorship-biased. Put a changes.csv (date,added,removed) in the universe folder to fix this.")
        cur.to_csv(cur_f, index=False)
        chg.to_csv(chg_f, index=False)
    cur = pd.read_csv(cur_f, parse_dates=["date_added"])
    chg = pd.read_csv(chg_f, parse_dates=["date"]).fillna("")
    iv, bad = uni.build_intervals(cur, chg)
    iv.to_csv(udir / "intervals.csv", index=False)
    ciks = uni.cik_map(cur, iv, udir / "cik_overrides.csv")
    print(f"universe: {len(cur)} current members, {len(chg)} logged changes ({bad} inconsistent and skipped), "
          f"{iv.symbol.nunique()} tickers ever members, {len(ciks)} with a SEC CIK "
          f"({iv.symbol.nunique() - len(set(iv.symbol) & set(ciks))} without: mostly removed companies)")
    return iv, ciks


def step_sec(a, root, tickers, ciks):
    client = fnd.SecClient(a.user_agent, root / "sec")
    todo = [(t, ciks[t]) for t in tickers if t in ciks]
    got = missing = 0
    for k, (t, cik) in enumerate(todo, 1):
        p = client.fetch_companyfacts(cik)
        got += p is not None
        missing += p is None
        if k % 50 == 0 or k == len(todo):
            print(f"  sec {k}/{len(todo)}  ({client.n_requests} requests, {missing} without XBRL facts)", flush=True)
    return got, missing


def step_prices(a, root, tickers):
    res = {}
    for k, t in enumerate(tickers, 1):
        r = data_us.download_one(t, root / "prices")
        res[r] = res.get(r, 0) + 1
        if k % 50 == 0 or k == len(tickers):
            print(f"  prices {k}/{len(tickers)} {res}", flush=True)
    return res


def step_panel(a, root, iv, ciks, tickers):
    adj, close_split, vol, splits = data_us.load_prices(tickers, root / "prices", a.start)
    tick = [t for t in adj.columns if t in ciks]
    adj, close_split, vol = adj[tick], close_split[tick], vol[tick]
    dates = adj.index
    member = uni.membership_matrix(iv, dates, tick)
    raw = {m: pd.DataFrame(np.nan, index=dates, columns=tick) for m in fnd.METRICS}
    n_ev = 0
    for t in tick:
        ev = fnd.events_for_cik(root / "sec", ciks[t], a.lag_days)
        if ev is None:
            continue
        n_ev += 1
        for m, v in fnd.align_metrics(ev, dates, close_split[t], splits.get(t)).items():
            raw[m][t] = v
    print(f"fundamentals aligned for {n_ev}/{len(tick)} tickers (lag {a.lag_days} day)")
    panel = build_panel(adj, vol, None, fundamentals=raw, member=member, min_fund=a.min_fund, horizon21=True)
    out = root / "panel.parquet"
    panel.to_parquet(out)
    report(panel, member, raw, root, a)
    print(f"wrote {out}: {len(panel):,} rows, {panel.index.get_level_values('date').nunique()} dates, "
          f"{panel.index.get_level_values('ticker').nunique()} tickers")


def report(panel, member, raw, root, a):
    yrs = panel.index.get_level_values("date").year
    L = [f"min_fund={a.min_fund}, lag_days={a.lag_days}", "", "rows / distinct stocks per year in the panel:"]
    for y, g in panel.groupby(yrs):
        L.append(f"  {y}  rows {len(g):>7}  stocks {g.index.get_level_values('ticker').nunique():>4}")
    L += ["", "share of member stock-days with each raw fundamental available (before rank/fill):"]
    mem = member.to_numpy()
    for m in FUND_FEATURES:
        v = raw[m].reindex(index=member.index, columns=member.columns).to_numpy()
        L.append(f"  {m:<20} {np.isfinite(v[mem]).mean():.1%}" if mem.sum() else f"  {m}")
    (root / "coverage_report.txt").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_fundamentals")
    ap.add_argument("--step", default="all", choices=["all", "universe", "sec", "prices", "panel"])
    ap.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT", ""))
    ap.add_argument("--limit", type=int, default=0, help="only the first N current members (dry run)")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--lag-days", type=int, default=1, help="a filing is usable from filed + this many days")
    ap.add_argument("--min-fund", type=int, default=7, help="a stock-day needs at least this many of the 11 fundamentals")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--wiki-html", default="", help="path of the Wikipedia 'List of S&P 500 companies' page saved from your browser (use if the download fails)")
    a = ap.parse_args()
    if "@" not in a.user_agent:
        raise SystemExit("Give --user-agent \"Your Name you@example.com\" (or set SEC_USER_AGENT): SEC and Wikipedia require it.")
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    iv, ciks = step_universe(a, root)
    tickers = sorted(set(iv.symbol) & set(ciks))
    if a.limit:
        cur = pd.read_csv(root / "universe" / "constituents.csv")["symbol"].tolist()
        tickers = [t for t in cur if t in ciks][:a.limit]
    if a.step in ("all", "sec"):
        print(f"SEC companyfacts for {len(tickers)} companies ...")
        print("  done: %d fetched or cached, %d without XBRL facts" % step_sec(a, root, tickers, ciks))
    if a.step in ("all", "prices"):
        print(f"Yahoo prices for {len(tickers)} tickers ...")
        step_prices(a, root, tickers)
    if a.step in ("all", "panel"):
        step_panel(a, root, iv, ciks, tickers)


if __name__ == "__main__":
    main()
