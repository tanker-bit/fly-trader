"""Point-in-time company fundamentals from SEC EDGAR XBRL "companyfacts".

How look-ahead is prevented (tests/test_fundamentals.py checks each of these):
  * Every fact carries the date it was FILED. A "snapshot" of a company at filing date F uses only facts with filed <= F.
    Later restatements are therefore invisible until they are filed (as-of semantics: if a restated number was filed on or
    before F, it is what an investor could see at F; nothing filed after F is ever used).
  * Snapshots are taken only at the dates of periodic reports (10-K, 10-Q, 20-F, 40-F and amendments). Facts filed in other
    forms (8-K ...) become visible at the next periodic report, which can only make the data later, never earlier.
  * A snapshot becomes usable on filed + `lag_days` (default 1 calendar day, i.e. a trading date t sees filings with
    filed < t): SEC dates carry no time of day and many filings arrive after the close, while the panel decides at the close.
  * Market value is shares (as reported in the snapshot) x price, with a split correction (see align_metrics).

SEC fair-access rules: a User-Agent that identifies you (name + email) is mandatory, and at most 10 requests/second;
this client stays at about 7 per second (hard cap 8) and refuses to start without a User-Agent containing an email.
"""
from __future__ import annotations

import gzip
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
MAX_REQUESTS_PER_SEC = 8
MIN_INTERVAL = 0.14            # about 7.1 requests/second, safely under both our cap (8) and SEC's limit (10)
PERIODIC_FORMS = ("10-K", "10-Q", "10-KT", "10-QT", "20-F", "40-F")   # prefix match, so amendments (10-K/A) count too
NAT = -10 ** 9                 # sentinel for "no start date" (instant facts)

# concept -> priority list of (taxonomy, tag, unit); the first tag that has a value for a period wins
CONCEPTS: dict[str, list[tuple[str, str, str]]] = {
    "ni": [("us-gaap", "NetIncomeLoss", "USD"), ("us-gaap", "ProfitLoss", "USD")],
    "rev": [("us-gaap", "Revenues", "USD"),
            ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
            ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax", "USD"),
            ("us-gaap", "SalesRevenueNet", "USD"), ("us-gaap", "SalesRevenueGoodsNet", "USD")],
    "cogs": [("us-gaap", "CostOfRevenue", "USD"), ("us-gaap", "CostOfGoodsAndServicesSold", "USD"),
             ("us-gaap", "CostOfGoodsSold", "USD")],
    "gp": [("us-gaap", "GrossProfit", "USD")],
    "ocf": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities", "USD")],
    "capex": [("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment", "USD"),
              ("us-gaap", "PaymentsToAcquireProductiveAssets", "USD")],
    "eps": [("us-gaap", "EarningsPerShareDiluted", "USD/shares"), ("us-gaap", "EarningsPerShareBasic", "USD/shares")],
    "equity": [("us-gaap", "StockholdersEquity", "USD"),
               ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD")],
    "assets": [("us-gaap", "Assets", "USD")],
    "debt_total": [("us-gaap", "LongTermDebt", "USD")],
    "debt_nc": [("us-gaap", "LongTermDebtNoncurrent", "USD")],
    "debt_cur": [("us-gaap", "LongTermDebtCurrent", "USD"), ("us-gaap", "DebtCurrent", "USD")],
    "shares": [("dei", "EntityCommonStockSharesOutstanding", "shares"),
               ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares")],
}
FLOWS = ("ni", "rev", "cogs", "gp", "ocf", "capex", "eps")
METRICS = ["earnings_yield", "book_to_price", "fcf_yield", "roe", "roa", "gross_margin", "rev_growth_yoy",
           "eps_growth_yoy", "debt_to_equity", "accruals", "days_since_filing"]


# ------------------------------------------------------------------ SEC client
class SecError(RuntimeError):
    pass


class SecBlocked(SecError):
    pass


class RateLimiter:
    """Thread-safe: consecutive calls to wait() are at least `min_interval` seconds apart."""

    def __init__(self, min_interval=MIN_INTERVAL, clock=time.monotonic, sleep=time.sleep):
        if min_interval < 1.0 / MAX_REQUESTS_PER_SEC:
            raise ValueError(f"min_interval {min_interval}s would exceed {MAX_REQUESTS_PER_SEC} requests/second")
        self.min_interval, self.clock, self.sleep = min_interval, clock, sleep
        self._next, self._lock = 0.0, threading.Lock()

    def wait(self):
        with self._lock:
            now = self.clock()
            start = max(now, self._next)
            self._next = start + self.min_interval
        if start > now:
            self.sleep(start - now)


class SecClient:
    def __init__(self, user_agent: str, cache_dir, limiter: RateLimiter | None = None, opener=None, max_retries=4,
                 sleep=time.sleep):
        if not user_agent or "@" not in user_agent:
            raise ValueError("SEC requires a User-Agent that identifies you, with contact info, e.g. "
                             "'Jane Doe jane@example.com'. Pass --user-agent or set SEC_USER_AGENT.")
        self.user_agent, self.cache = user_agent, Path(cache_dir)
        if opener is None:
            from env.universe_us import tls_context
            ctx = tls_context()
            opener = lambda req, timeout=60: urllib.request.urlopen(req, timeout=timeout, context=ctx)
        self.limiter, self.opener = limiter or RateLimiter(), opener
        self.max_retries, self.sleep = max_retries, sleep
        (self.cache / "raw").mkdir(parents=True, exist_ok=True)
        self.n_requests = 0

    def get(self, url: str) -> bytes | None:
        """Bytes of the response, or None for 404. Retries 429/5xx with backoff; stops hard on 403."""
        for attempt in range(self.max_retries):
            self.limiter.wait()
            self.n_requests += 1
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip"})
            try:
                with self.opener(req, timeout=60) as resp:
                    data = resp.read()
                    if (resp.headers.get("Content-Encoding") or "").lower() == "gzip":
                        data = gzip.decompress(data)
                    return data
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.code == 403:
                    raise SecBlocked("SEC answered 403 (blocked). Check the User-Agent, wait a while, do not retry in a loop.")
                if e.code in (429, 500, 502, 503, 504):
                    self.sleep(2 ** attempt * 2)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                self.sleep(2 ** attempt * 2)
        raise SecError(f"giving up on {url} after {self.max_retries} tries")

    def fetch_companyfacts(self, cik: int) -> Path | None:
        """Download (once) and cache the raw companyfacts JSON. None if SEC has no XBRL facts for this CIK."""
        raw = self.cache / "raw" / f"CIK{cik:010d}.json"
        missing = self.cache / "raw" / f"CIK{cik:010d}.404"
        if raw.exists():
            return raw
        if missing.exists():
            return None
        data = self.get(COMPANYFACTS_URL.format(cik=cik))
        if data is None:
            missing.write_text("404")
            return None
        tmp = str(raw) + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, raw)
        return raw


# ------------------------------------------------------------------ parsing
def compact_companyfacts(js: dict) -> pd.DataFrame:
    """Keep only the tags we use: columns concept, prio, start, end, val, filed, form."""
    rows = []
    facts = js.get("facts", {})
    for concept, specs in CONCEPTS.items():
        for prio, (tax, tag, unit) in enumerate(specs):
            for d in facts.get(tax, {}).get(tag, {}).get("units", {}).get(unit, []):
                if "end" in d and "filed" in d and d.get("val") is not None:
                    rows.append((concept, prio, d.get("start"), d["end"], float(d["val"]), d["filed"], d.get("form", "")))
    df = pd.DataFrame(rows, columns=["concept", "prio", "start", "end", "val", "filed", "form"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df.dropna(subset=["end", "filed"]).reset_index(drop=True)


def _days(s: pd.Series) -> np.ndarray:
    d = s.to_numpy("datetime64[D]").astype("int64")
    return np.where(s.isna().to_numpy(), NAT, d)


def _arrays(facts: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for concept, g in facts.groupby("concept"):
        g = g.sort_values("filed")
        out[concept] = dict(S=_days(g["start"]), E=_days(g["end"]), V=g["val"].to_numpy(float),
                            P=g["prio"].to_numpy(int), F=_days(g["filed"]))
    return out


# ------------------------------------------------------------------ as-of snapshot
def _asof(a, F: int):
    """Facts with filed <= F, one per (start, end): best tag priority, then latest filing. Returns (S, E, V) or None."""
    if a is None:
        return None
    m = a["F"] <= F
    if not m.any():
        return None
    S, E, V, P, Fl = a["S"][m], a["E"][m], a["V"][m], a["P"][m], a["F"][m]
    order = np.lexsort((-Fl, P))                    # primary key P ascending, then filed descending
    S, E, V = S[order], E[order], V[order]
    _, first = np.unique(np.stack([S, E], 1), axis=0, return_index=True)
    return S[first], E[first], V[first]


def _ttm(S, E, V):
    """Trailing-twelve-month value = latest fiscal year + current YTD - prior-year YTD (all as known). Returns (value, end)."""
    d = E - S
    ann = (d >= 350) & (d <= 380)
    if not ann.any():
        return np.nan, None
    ia = np.flatnonzero(ann)[np.argmax(E[ann])]
    aE, aV = E[ia], V[ia]
    cand = (d >= 60) & (d <= 300) & (S >= aE - 3) & (S <= aE + 10) & (E > aE)
    if not cand.any():
        return aV, aE
    ic = np.flatnonzero(cand)[np.argmax(E[cand])]
    prior = (np.abs(S - (S[ic] - 365)) <= 10) & (np.abs(E - (E[ic] - 365)) <= 10)
    if not prior.any():
        return aV, aE
    ip = np.flatnonzero(prior)[np.argmin(np.abs(E[prior] - (E[ic] - 365)))]
    return aV + V[ic] - V[ip], E[ic]


def _yoy(S, E, V):
    """(current, same period a year earlier) for the most recent period that has a comparable, both as known now."""
    d = E - S
    for i in np.argsort(-E, kind="stable"):
        if d[i] < 60 or d[i] > 380:
            continue
        j = np.flatnonzero((np.abs(S - (S[i] - 365)) <= 10) & (np.abs(E - (E[i] - 365)) <= 10))
        if len(j):
            return V[i], V[j[0]]
    return np.nan, np.nan


def _instant(S, E, V):
    i = int(np.argmax(E))
    return V[i], E[i]


def snapshot(arrs, F: int) -> dict:
    """Everything an investor could compute from facts filed on or before day F (days since 1970-01-01)."""
    g = {c: _asof(arrs.get(c), F) for c in CONCEPTS}
    out = dict(filed=F)
    for c in ("ni", "rev", "cogs", "gp", "ocf", "capex"):
        out[c + "_ttm"] = _ttm(*g[c])[0] if g[c] is not None else np.nan
    for c in ("equity", "assets", "shares"):
        out[c] = _instant(*g[c])[0] if g[c] is not None else np.nan
    out["rev_cur"], out["rev_prev"] = _yoy(*g["rev"]) if g["rev"] is not None else (np.nan, np.nan)
    out["eps_cur"], out["eps_prev"] = _yoy(*g["eps"]) if g["eps"] is not None else (np.nan, np.nan)
    dt = _instant(*g["debt_total"])[0] if g["debt_total"] is not None else np.nan
    dn = _instant(*g["debt_nc"])[0] if g["debt_nc"] is not None else np.nan
    dc = _instant(*g["debt_cur"])[0] if g["debt_cur"] is not None else 0.0
    out["debt"] = dt if not np.isnan(dt) else (dn + dc if not np.isnan(dn) else np.nan)
    return out


def build_events(facts: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """One row per periodic filing date, with every metric input as known on that date. `avail` = filed + lag_days."""
    if facts.empty:
        return pd.DataFrame()
    arrs = _arrays(facts)
    periodic = facts["form"].fillna("").str.upper().str.startswith(PERIODIC_FORMS)
    days = sorted(set(_days(facts.loc[periodic, "filed"])))
    rows = [snapshot(arrs, int(F)) for F in days]
    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev
    ev["filed"] = pd.to_datetime(ev["filed"].astype("int64"), unit="D")
    ev["avail"] = ev["filed"] + pd.Timedelta(days=lag_days)
    return ev.sort_values("avail").reset_index(drop=True)


# ------------------------------------------------------------------ alignment to trading dates
def split_after(dates: pd.DatetimeIndex, splits: pd.Series | None) -> np.ndarray:
    """For each date i, the product of split ratios that happen AFTER i (ratio 2 = 2-for-1, 0.1 = 1-for-10)."""
    sr = np.ones(len(dates))
    if splits is not None and len(splits):
        for sd, ratio in splits.items():
            pos = int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(sd).to_datetime64()), side="left"))
            if pos < len(dates) and ratio > 0:
                sr[pos] *= float(ratio)
    return np.cumprod(sr[::-1])[::-1] / sr


def align_metrics(events: pd.DataFrame, dates: pd.DatetimeIndex, close_split: pd.Series,
                  splits: pd.Series | None = None) -> dict[str, np.ndarray]:
    """Raw (unranked) fundamental metrics on each trading date, using the latest event whose `avail` <= that date.

    `close_split` is the Yahoo 'Close' (split-adjusted, NOT dividend-adjusted). Market value at the event is
    shares_reported x (Close x later split factors) = shares x the actual price then; afterwards it moves with Close,
    which is unaffected by splits, so a split can never create a jump.
    """
    n = len(dates)
    nan = np.full(n, np.nan)
    out = {m: nan.copy() for m in METRICS}
    if events is None or len(events) == 0:
        return out
    cf = close_split.reindex(dates).ffill().to_numpy(float)
    sa = split_after(dates, splits)
    avail = events["avail"].to_numpy("datetime64[ns]")
    k = np.searchsorted(avail, dates.to_numpy("datetime64[ns]"), side="right") - 1
    iF = np.searchsorted(dates.to_numpy("datetime64[ns]"), avail, side="left")      # first trading day on/after avail
    okF = iF < n
    iFc = np.where(okF, iF, 0)
    mF = events["shares"].to_numpy(float) * cf[iFc] * sa[iFc]                         # market value at the event
    mF = np.where(okF, mF, np.nan)
    cF = np.where(okF, cf[iFc], np.nan)
    has = k >= 0
    kk = np.where(has, k, 0)
    ev = {c: events[c].to_numpy(float)[kk] for c in events.columns if events[c].dtype.kind in "fi"}
    mcap = mF[kk] * cf / cF[kk]
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = lambda x: np.where(x > 0, x, np.nan)
        gp = np.where(np.isnan(ev["gp_ttm"]), ev["rev_ttm"] - ev["cogs_ttm"], ev["gp_ttm"])
        res = dict(
            earnings_yield=ev["ni_ttm"] / pos(mcap),
            book_to_price=ev["equity"] / pos(mcap),
            fcf_yield=(ev["ocf_ttm"] - ev["capex_ttm"]) / pos(mcap),
            roe=ev["ni_ttm"] / pos(ev["equity"]),
            roa=ev["ni_ttm"] / pos(ev["assets"]),
            gross_margin=gp / pos(ev["rev_ttm"]),
            rev_growth_yoy=ev["rev_cur"] / pos(ev["rev_prev"]) - 1.0,
            eps_growth_yoy=(ev["eps_cur"] - ev["eps_prev"]) / np.where(ev["eps_prev"] != 0, np.abs(ev["eps_prev"]), np.nan),
            debt_to_equity=ev["debt"] / pos(ev["equity"]),
            accruals=(ev["ni_ttm"] - ev["ocf_ttm"]) / pos(ev["assets"]),
        )
    filed_days = events["filed"].to_numpy("datetime64[D]").astype("int64")[kk]
    res["days_since_filing"] = dates.to_numpy("datetime64[D]").astype("int64") - filed_days
    for m, v in res.items():
        out[m] = np.where(has, v, np.nan).astype(float)
    return out


def events_for_cik(sec_dir, cik: int, lag_days: int = 1, refresh: bool = False) -> pd.DataFrame | None:
    """Cached events table for one company (None if it has no usable XBRL facts)."""
    sec_dir = Path(sec_dir)
    (sec_dir / "events").mkdir(parents=True, exist_ok=True)
    cache = sec_dir / "events" / f"CIK{cik:010d}_lag{lag_days}.pkl"
    if cache.exists() and not refresh:
        return pd.read_pickle(cache)
    raw = sec_dir / "raw" / f"CIK{cik:010d}.json"
    if not raw.exists():
        return None
    with open(raw) as f:
        ev = build_events(compact_companyfacts(json.load(f)), lag_days)
    ev = ev if len(ev) else None
    pd.to_pickle(ev, cache)
    return ev
