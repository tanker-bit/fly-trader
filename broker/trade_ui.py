r"""V5: place a day's orders on the Investopedia simulator website, the way a person would, in the bot's own visible Edge window.

    python -m broker.trade_ui --login                       # once: opens the bot's Edge window; sign in there (kept afterwards)
    python -m broker.trade_ui --date 2026-11-25             # place orders\2026-11-25\orders.csv (SUBMITS by default)
    python -m broker.trade_ui --date 2026-11-25 --preview-only     # fill in and preview each order, never submit

For every order (sells first): open the Trade tab, type the symbol, pick the exact match from the suggestions, read the stock panel
(bid/ask, day and 52-week range, volume), choose Buy or Sell, enter the shares (Market, Day Only), press PREVIEW ORDER, check the
preview, then press SUBMIT ORDER. Checked before every submit, otherwise the order is skipped and logged:
  * the preview shows exactly this symbol, side ("Buy at Market" / "Sell at Market") and number of shares;
  * the live price (ask for a buy, bid for a sell) is within --max-drift (5%) of the price the plan was sized with;
  * a buy's estimated total fits in the buying power shown on the page;
  * no file named STOP exists in the orders folder (create one to halt the run between orders).
Only Buy and Sell are ever chosen (never Short / Buy To Cover); only Market orders, Day Only.

Everything goes to orders\<date>\trade_log.csv plus a screenshot per step in orders\<date>\screens\. If every order was submitted or
skipped by a guard (no errors), that day is accepted automatically (accounts.policy_state memory, as broker.daily_run --accept).
The window uses its own Edge profile (default ..\data_live\edge_profile, outside the repository: it holds the live login).
Layout found on 2026-11-27 (data-cy names below); if the site changes them the run stops with an error instead of guessing.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

TRADE_URL = "https://www.investopedia.com/simulator/trade/stocks"
SEL = dict(
    symbol='[data-cy="symbol-search-field"] input', options='[role="option"], .v-list-item', form='[data-cy="symbol-info-form"]',
    action='[data-cy="action-select"]', qty='[data-cy="quantity-input"] input', order_type='[data-cy="order-select"]',
    duration='[data-cy="duration-select"]', preview='[data-cy="preview-button"]', dialog='[data-cy="preview-order-dialog"]',
    pv_type='[data-cy="preview-order-type"]', pv_symbol='[data-cy="preview-stock-symbol"]', pv_qty='[data-cy="preview-stock-quantity"]',
    pv_items='[data-cy="preview-list-item"]', submit='[data-cy="submit-order-button"]', change='[data-cy="change-order-button"]',
    buying_power='[data-cy="buying-power"]', cash='[data-cy="cash"]', dismiss=".pushly-prompt-btn-dismiss",
)
SIDES = {"BUY": "Buy", "SELL": "Sell"}
ERROR_WORDS = re.compile(r"error|insufficient|not enough|unable|failed|rejected|not allowed|invalid", re.I)
DEFAULT_PROFILE = Path("..") / "data_live" / "edge_profile"


class SiteChanged(RuntimeError):
    """The page does not look like the layout this module was written for: stop rather than guess."""


class NotSignedIn(RuntimeError):
    pass


# ---------------------------------------------------------------- parsing and checks (no browser needed)
def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def money(text) -> float | None:
    m = re.search(r"-?\$?\s*([\d,]+(?:\.\d+)?)", str(text or ""))
    return float(m.group(1).replace(",", "")) if m else None


def parse_panel(text: str) -> dict:
    """The stock panel under the symbol box, e.g. 'Bid/Ask price ($)\t273.17/273.21'."""
    out = {}
    m = re.search(r"Bid/Ask price[^\d]*([\d,.]+)\s*/\s*([\d,.]+)", text or "")
    if m:
        out["bid"], out["ask"] = float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))
    for key, label in (("day_high", "Day's High"), ("day_low", "Day's Low"), ("high_52w", "52 Week High"), ("low_52w", "52 Week Low")):
        m = re.search(re.escape(label) + r"[^\d]*([\d,.]+)", text or "")
        if m:
            out[key] = float(m.group(1).replace(",", ""))
    m = re.search(r"Volume \(current\)\s*([\d.,]+\s*[kmb]?)", text or "", re.I)
    if m:
        out["volume"] = m.group(1).strip()
    return out


def parse_preview(order_type: str, symbol: str, qty: str, items: list) -> dict:
    d = dict(type=(order_type or "").strip(), symbol=(symbol or "").strip().upper(), qty=None, est_price=None, est_total=None,
             commission=None)
    try:
        d["qty"] = int(float(str(qty).replace(",", "").strip()))
    except ValueError:
        pass
    for it in items:
        t = " ".join(str(it).split())
        if t.startswith("Estimated Price"):
            d["est_price"] = money(t)
        elif t.startswith("Estimated Total"):
            d["est_total"] = money(t)
        elif t.startswith("Commission"):
            d["commission"] = money(t)
    return d


def pick_suggestion(texts: list, game_symbol: str) -> int | None:
    """Index of the suggestion whose ticker is exactly this symbol (the first word of the line)."""
    want = game_symbol.strip().upper()
    for i, t in enumerate(texts):
        first = re.split(r"[\s|]+", t.strip(), maxsplit=1)[0].upper() if t.strip() else ""
        if first == want:
            return i
    return None


def check_before_submit(order: dict, panel: dict, pv: dict, buying_power: float | None, max_drift: float) -> str | None:
    """None if the order may be submitted, otherwise the reason it must be skipped."""
    side, sym, qty = order["side"], order["game_symbol"].upper(), int(order["shares"])
    want_type = f"{SIDES[side]} at Market"
    if want_type.lower() not in pv.get("type", "").lower():
        return f"preview says {pv.get('type')!r}, expected {want_type!r}"
    if pv.get("symbol") != sym:
        return f"preview symbol {pv.get('symbol')!r} is not {sym}"
    if pv.get("qty") != qty:
        return f"preview quantity {pv.get('qty')} is not {qty}"
    live = panel.get("ask") if side == "BUY" else panel.get("bid")
    live = live or pv.get("est_price")
    plan = float(order["est_price"])
    if not live or plan <= 0:
        return "no live price to compare with the plan"
    drift = live / plan - 1.0
    if abs(drift) > max_drift:
        return f"price moved {drift:+.1%} since the plan ({plan:.2f} -> {live:.2f}), limit {max_drift:.0%}"
    if side == "BUY" and buying_power is not None and pv.get("est_total") and pv["est_total"] > buying_power:
        return f"estimated total {pv['est_total']:,.2f} is more than the buying power {buying_power:,.2f}"
    return None


def should_accept(results: list) -> bool:
    """Accept the day's policy memory when nothing went wrong: every order was submitted or skipped by a guard."""
    return all(r["status"] in ("submitted", "skipped") for r in results) and (
        any(r["status"] == "submitted" for r in results) or not results)


# ---------------------------------------------------------------- the website
class TradeSite:
    def __init__(self, page, shots: Path | None = None, pause: float = 1.5, log=print, emit=None):
        from broker.live_feed import nothing
        self.page, self.shots, self.pause, self.log, self.n = page, shots, pause, log, 0
        self.emit = emit or nothing                   # broker.live_feed.Feed: the 3D dashboard shows what happens here

    def shot(self, name):
        if self.shots:
            self.shots.mkdir(parents=True, exist_ok=True)
            self.n += 1
            self.page.screenshot(path=str(self.shots / f"{self.n:03d}_{name}.png"))

    def wait(self, k=1.0):
        time.sleep(self.pause * k)

    def open_trade(self, url=TRADE_URL):
        self.page.goto(url)
        try:
            self.page.wait_for_selector(SEL["symbol"], timeout=45000)
        except Exception as e:
            if "auth." in self.page.url or "Sign in" in (self.page.title() or ""):
                raise NotSignedIn("not signed in: run  python -m broker.trade_ui --login  and sign in in the window") from e
            raise SiteChanged(f"no symbol box on {self.page.url}") from e
        self.wait()
        self.dismiss_popups()

    def text(self, key) -> str:
        loc = self.page.locator(SEL[key])
        return loc.first.inner_text().strip() if loc.count() else ""

    def choose(self, key, label):
        """Open a dropdown (action / order type / duration) and click the option with exactly this text."""
        if self.text(key).strip().lower().startswith(label.lower()):
            return
        self.page.locator(SEL[key]).first.click()
        self.wait()
        opts = self.page.locator(SEL["options"])
        for i in range(opts.count()):
            if opts.nth(i).inner_text().strip().lower() == label.lower():
                opts.nth(i).click()
                self.wait()
                if not self.text(key).strip().lower().startswith(label.lower()):
                    raise SiteChanged(f"chose {label!r} but the field shows {self.text(key)!r}")
                return
        raise SiteChanged(f"no option {label!r} in {key}")

    def enter(self, order: dict) -> dict:
        """Fill the form for one order and return the stock panel's numbers. Raises LookupError if the game has no such symbol."""
        sym = order["game_symbol"].upper()
        self.emit("search", symbol=sym, side=order["side"], shares=int(order["shares"]), rank=_num(order.get("rank")),
                  score=_num(order.get("score")), reason=order.get("reason", ""))
        box = self.page.locator(SEL["symbol"]).first
        box.click()
        box.fill("")
        box.type(sym, delay=90)
        self.wait(2)
        opts = self.page.locator(SEL["options"])
        texts = [opts.nth(i).inner_text().replace("\n", " | ") for i in range(min(opts.count(), 15))]
        k = pick_suggestion(texts, sym)
        if k is None:
            raise LookupError(f"the game offers no exact match for {sym} (suggestions: {texts[:5]})")
        opts.nth(k).click()
        try:
            self.page.wait_for_function("t => document.querySelector(t) && /Bid\\/Ask/.test(document.querySelector(t).innerText)",
                                        arg=SEL["form"], timeout=20000)
        except Exception as e:
            raise SiteChanged("the stock panel with Bid/Ask did not appear") from e
        panel = parse_panel(self.text("form"))
        self.emit("quote", symbol=sym, **panel)
        self.choose("action", SIDES[order["side"]])
        self.choose("order_type", "Market")
        self.choose("duration", "Day Only")
        q = self.page.locator(SEL["qty"]).first
        q.fill(str(int(order["shares"])))
        self.wait()
        self.shot(f"{sym}_filled")
        return panel

    def dismiss_popups(self):
        d = self.page.locator(SEL["dismiss"])
        if d.count() and d.first.is_visible():
            d.first.click()
            self.wait(0.5)

    def preview(self) -> dict:
        b = self.page.locator(SEL["preview"]).first
        if not b.is_enabled():
            raise SiteChanged("PREVIEW ORDER stays disabled after filling the form")
        for attempt in range(2):            # seen on the real site: the first click sometimes opened nothing (a late pop-up took it)
            self.dismiss_popups()
            b.scroll_into_view_if_needed()
            b.click()
            try:
                self.page.wait_for_selector(SEL["dialog"], timeout=10000 if attempt == 0 else 15000)
                break
            except Exception as e:
                if attempt == 1:                # logged as an error for this order; the run goes on with the next one
                    raise RuntimeError("no preview dialog appeared after two clicks") from e
        self.wait()
        items = [x.inner_text() for x in self.page.locator(SEL["pv_items"]).all()]
        pv = parse_preview(self.text("pv_type"), self.text("pv_symbol"), self.text("pv_qty"), items)
        side = "SELL" if "sell" in pv["type"].lower() else "BUY"
        self.emit("preview", symbol=pv["symbol"], side=side, shares=pv["qty"], est_price=pv["est_price"], est_total=pv["est_total"])
        return pv

    def buying_power(self) -> float | None:
        return money(self.text("buying_power"))

    def back_out(self):
        c = self.page.locator(SEL["change"])
        if c.count():
            c.first.click()
            self.wait()

    def submit(self) -> str:
        """Press SUBMIT ORDER and return whatever message the page shows next ('' if none)."""
        s = self.page.locator(SEL["submit"]).first
        if not re.search(r"submit", s.inner_text(), re.I):
            raise SiteChanged(f"the submit button reads {s.inner_text()!r}")
        s.click()
        self.wait(3)
        msgs = []
        for sel in ('[role="alert"]', ".v-snackbar", '[role="dialog"]', ".v-overlay__content"):
            for x in self.page.locator(sel).all():
                try:
                    t = " ".join(x.inner_text().split())
                except Exception:
                    continue
                if t and t not in msgs:
                    msgs.append(t[:300])
        return " / ".join(msgs)


# ---------------------------------------------------------------- the run
LOG_COLS = ["time", "side", "symbol", "game_symbol", "shares", "plan_price", "bid", "ask", "est_price", "est_total", "status", "detail",
            "reason"]


def run_orders(site, orders: list, preview_only=False, max_drift=0.05, stop_file: Path | None = None, log=print) -> list:
    results = []
    for o in sorted(orders, key=lambda o: 0 if o["side"] == "SELL" else 1):
        row = dict(time=dt.datetime.now().isoformat(timespec="seconds"), side=o["side"], symbol=o["symbol"], game_symbol=o["game_symbol"],
                   shares=int(o["shares"]), plan_price=float(o["est_price"]), reason=o.get("reason", ""), bid=None, ask=None,
                   est_price=None, est_total=None)
        if stop_file and stop_file.exists():
            results.append(dict(row, status="stopped", detail=f"{stop_file} exists"))
            site.emit("result", symbol=o["game_symbol"], side=o["side"], shares=int(o["shares"]), status="stopped",
                      detail="STOP file", score=_num(o.get("score")))
            log(f"STOP file found: {stop_file}; the remaining orders were not touched")
            break
        try:
            site.open_trade()
            panel = site.enter(o)
            row.update(bid=panel.get("bid"), ask=panel.get("ask"))
            pv = site.preview()
            row.update(est_price=pv.get("est_price"), est_total=pv.get("est_total"))
            why = check_before_submit(o, panel, pv, site.buying_power(), max_drift)
            site.shot(f"{o['game_symbol']}_preview")
            if why:
                site.back_out()
                results.append(dict(row, status="skipped", detail=why))
            elif preview_only:
                site.back_out()
                results.append(dict(row, status="previewed", detail="--preview-only: not submitted"))
            else:
                msg = site.submit()
                site.shot(f"{o['game_symbol']}_submitted")
                bad = ERROR_WORDS.search(msg or "")
                results.append(dict(row, status="error" if bad else "submitted", detail=msg or "no message shown"))
        except LookupError as e:
            results.append(dict(row, status="skipped", detail=str(e)))
        except (SiteChanged, NotSignedIn):
            raise
        except Exception as e:                                         # a timeout or a page glitch on one order: log it, go on
            results.append(dict(row, status="error", detail=f"{type(e).__name__}: {str(e)[:200]}"))
            try:
                site.shot(f"{o['game_symbol']}_error")
            except Exception:
                pass
        r = results[-1]
        site.emit("result", symbol=r["game_symbol"], side=r["side"], shares=r["shares"], status=r["status"], detail=r["detail"],
                  score=_num(o.get("score")))
        log(f"{r['status']:>9}  {r['side']:<4} {r['shares']:>6} {r['game_symbol']:<7} {r['detail'][:110]}")
    return results


def write_log(path: Path, results: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in results:
            w.writerow(r)


def read_orders(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["side"] not in SIDES:
            raise ValueError(f"unexpected side {r['side']!r} in {path}")
        if int(float(r["shares"])) <= 0:
            raise ValueError(f"non-positive shares for {r['symbol']} in {path}")
    return rows


def browser(profile: Path, headless=False):
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    ctx = pw.chromium.launch_persistent_context(str(profile), channel="msedge", headless=headless, viewport={"width": 1280, "height": 900})
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    return pw, ctx, page


def login(profile: Path, wait_s=900, log=print) -> bool:
    pw, ctx, page = browser(profile)
    try:
        page.goto(TRADE_URL)
        log("sign in in the Edge window that just opened (it waits up to 15 minutes) ...")
        t0, ok = time.time(), 0
        while time.time() - t0 < wait_s:
            hit = any(p.url.startswith("https://www.investopedia.com/simulator/") and "Sign in" not in (p.title() or "")
                      for p in list(ctx.pages))
            ok = ok + 1 if hit else 0
            if ok >= 4:
                log("signed in; the login is saved in the bot's profile")
                return True
            time.sleep(2)
        log("timed out waiting for the sign-in")
        return False
    finally:
        ctx.close()
        pw.stop()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="the orders\\<date> folder made by broker.daily_run")
    ap.add_argument("--orders-dir", default="orders")
    ap.add_argument("--state", default="orders/policy_state.json")
    ap.add_argument("--profile", default=str(DEFAULT_PROFILE), help="the bot's Edge profile (holds the live login; keep it out of git)")
    ap.add_argument("--login", action="store_true", help="open the window to sign in once, then exit")
    ap.add_argument("--preview-only", action="store_true", help="fill in and preview every order, never submit")
    ap.add_argument("--max-drift", type=float, default=0.05, help="skip an order if the live price moved more than this from the plan")
    ap.add_argument("--no-accept", action="store_true", help="do not accept the day's policy memory automatically")
    ap.add_argument("--headless", action="store_true", help="no visible window (not recommended: you cannot watch it)")
    a = ap.parse_args(argv)
    profile = Path(a.profile).resolve()
    if a.login:
        raise SystemExit(0 if login(profile) else 1)
    if not a.date:
        raise SystemExit("--date is required (the folder broker.daily_run wrote, e.g. 2026-11-25)")
    day = Path(a.orders_dir) / a.date
    orders = read_orders(day / "orders.csv")
    print(f"{len(orders)} orders for {a.date} from {day / 'orders.csv'}; {'PREVIEW ONLY' if a.preview_only else 'SUBMITTING'}; "
          f"create {Path(a.orders_dir) / 'STOP'} to halt between orders")
    from broker.live_feed import Feed
    feed = Feed(Path(a.orders_dir) / "live_feed.jsonl")
    feed("start", mode="preview" if a.preview_only else "trading", asof=a.date)
    pw, ctx, page = browser(profile, a.headless)
    try:
        site = TradeSite(page, shots=day / "screens", emit=feed)
        results = run_orders(site, orders, a.preview_only, a.max_drift, Path(a.orders_dir) / "STOP")
    finally:
        ctx.close()
        pw.stop()
    write_log(day / "trade_log.csv", results)
    counts = {s: sum(r["status"] == s for r in results) for s in ("submitted", "skipped", "previewed", "error", "stopped")}
    feed("done", text=str(counts))
    print(f"\n{counts}; log: {day / 'trade_log.csv'}; screenshots: {day / 'screens'}")
    if not a.preview_only and not a.no_accept and should_accept(results):
        from broker.daily_run import accept
        accept(a.orders_dir, a.date, a.state)
    elif not a.preview_only:
        print("the day was NOT accepted (errors or a stop); check the log, then run broker.daily_run --accept when it is right")


if __name__ == "__main__":
    main()
