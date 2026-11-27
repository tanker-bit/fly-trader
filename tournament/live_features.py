"""V4b-2: today's feature rows for the flies, built exactly the way data_fundamentals/panel_hybrid.parquet was built.

    inp = load_inputs("data_fundamentals")                  # S&P 500 prices, membership, CIKs (local caches, no network)
    res = feature_rows(inp)                                  # the newest date with prices for 95% of the members
    res.rows                                                 # ticker x league features, ready for tournament.live.score_rows

Same code path as the training panel (experiments.build_dataset_price.build_price_panel + raw_fundamentals), so nothing is
re-implemented: 13 price/volume features ranked across that day's S&P 500 members, 11 point-in-time SEC fundamentals ranked the
same way (unknown = 0, the middle), and `fund_known`. Two details that keep the rows identical to the panel:
  * features need about 260 trading days of history, so only the last `window` days of prices are used (default 400; the result
    does not change with a longer window);
  * the fundamentals are aligned on dates running to the END of the price cache, not cut at `asof`: Yahoo's 'Close' is adjusted
    for every split up to the download, and the market-value step (env.fundamentals.split_after) only undoes splits it can see.
    Cutting the dates at `asof` changes earnings_yield, book_to_price and fcf_yield whenever a split came later.

Ranking is among ALL index members of the day, as in training. Filtering to the stocks the game offers happens afterwards (V4c),
never before, or the ranks would mean something else. The rows are only as fresh as the caches: refreshing prices, membership and
SEC filings is a separate step (V4d). Members without a row (under 260 days of history, price under $1, no price that day) are
listed in `dropped`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from env import universe_us as uni
from env.features import MIN_HISTORY, feature_columns

DEFAULT_WINDOW = 400
MIN_WINDOW = MIN_HISTORY + 40          # 260 days of history plus room for the 252-day features and the 3-day forward fill
MIN_COVERAGE = 0.95                    # share of the day's index members that must have a real (not forward-filled) price


@dataclass
class Inputs:
    root: Path                         # holds sec/ (the SEC cache)
    adj: pd.DataFrame                  # dividend-adjusted close, dates x tickers
    close_split: pd.DataFrame          # split-adjusted close (Yahoo 'Close')
    volume: pd.DataFrame
    splits: dict                       # ticker -> Series of split ratios
    intervals: pd.DataFrame            # point-in-time index membership: symbol, start, end
    ciks: dict                         # ticker -> SEC CIK


@dataclass
class FeatureRows:
    asof: pd.Timestamp
    rows: pd.DataFrame                 # ticker x features (the league's columns, panel order)
    members: list = field(default_factory=list)       # index members that day with prices in the cache
    dropped: list = field(default_factory=list)       # members with no row
    coverage: float = 1.0                             # share of members with a real price on asof
    no_price_today: list = field(default_factory=list)   # members whose row uses a forward-filled price (as in the panel)
    skipped_dates: list = field(default_factory=list)    # newer dates in the cache left out because too few members had prices


def load_inputs(root="data_fundamentals", start="2000-01-01") -> Inputs:
    """The same local caches the hybrid panel was built from. No downloads."""
    from experiments.build_dataset_price import load_all_prices
    root = Path(root)
    iv = pd.read_csv(root / "universe" / "intervals.csv", parse_dates=["start", "end"])
    cur = pd.read_csv(root / "universe" / "constituents.csv", parse_dates=["date_added"])
    ciks = uni.cik_map(cur, iv, root / "universe" / "cik_overrides.csv")
    adj, cs, vol, spl, _ = load_all_prices(root, start, full=True)
    return Inputs(root, adj, cs, vol, spl, iv, ciks)


def _members_on(inp: Inputs, d) -> list:
    known = [t for t in inp.adj.columns if t in set(inp.intervals.symbol)]
    on = uni.membership_matrix(inp.intervals, pd.DatetimeIndex([d]), known).iloc[0]
    return sorted(t for t in known if on[t])


def coverage(inp: Inputs, d) -> float:
    """Share of the index members on date d that have a real price that day (0 when there are none)."""
    m = _members_on(inp, d)
    return float(inp.adj.loc[d, m].notna().mean()) if m else 0.0


def latest_complete_date(inp: Inputs, min_coverage: float = MIN_COVERAGE, look_back: int = 30):
    """The newest date on which at least `min_coverage` of the members have a real price, and the newer dates passed over."""
    dates = inp.adj.index
    skipped = []
    for d in dates[::-1][:look_back]:
        if coverage(inp, d) >= min_coverage:
            return d, skipped
        skipped.append(d)
    raise ValueError(f"none of the last {look_back} dates in the price cache has prices for {min_coverage:.0%} of the members: "
                     f"refresh the prices")


def feature_rows(inp: Inputs, asof=None, window: int = DEFAULT_WINDOW, lag_days: int = 1, min_coverage: float = MIN_COVERAGE,
                 log=lambda s: None) -> FeatureRows:
    """Feature rows of every index member on `asof`. Default: the newest date with real prices for `min_coverage` of the members
    (a partly refreshed cache would otherwise give an almost empty day, or rows built from forward-filled stale prices)."""
    from experiments.build_dataset_price import build_price_panel, raw_fundamentals
    if window < MIN_WINDOW:
        raise ValueError(f"window {window} is too short: features need at least {MIN_WINDOW} trading days")
    dates = inp.adj.index
    skipped = []
    if asof is None:
        asof, skipped = latest_complete_date(inp, min_coverage)
    asof = pd.Timestamp(asof)
    if asof not in dates:
        raise ValueError(f"{asof.date()} is not a trading day in the price cache ({dates[0].date()} .. {dates[-1].date()})")
    i = dates.get_loc(asof)
    if i + 1 < MIN_HISTORY:
        raise ValueError(f"only {i + 1} days of prices up to {asof.date()}; features need {MIN_HISTORY}")
    lo = max(0, i - window + 1)

    members = _members_on(inp, asof)
    if not members:
        raise ValueError(f"no index members with prices on {asof.date()}")
    cov = coverage(inp, asof)
    if cov < min_coverage:
        raise ValueError(f"only {cov:.0%} of the {len(members)} members have a price on {asof.date()} (need {min_coverage:.0%}): "
                         f"the cache is incomplete for that day")
    no_price = sorted(t for t in members if pd.isna(inp.adj.at[asof, t]))

    # fundamentals: aligned from the window start to the END of the cache (see the module docstring), then cut to the window
    span = slice(lo, len(dates))
    raw = raw_fundamentals(inp.root, inp.ciks, inp.adj.iloc[span][members], inp.close_split.iloc[span][members], inp.splits, lag_days)
    win = dates[lo:i + 1]
    raw = {k: v.loc[win] for k, v in raw.items()}

    panel = build_price_panel(inp.adj.loc[win, members], inp.volume.loc[win, members], inp.intervals, raw)
    day = panel.xs(asof, level="date") if asof in panel.index.get_level_values("date") else panel.iloc[0:0].droplevel("date")
    rows = day[feature_columns(day)].sort_index()
    dropped = sorted(set(members) - set(rows.index))
    log(f"{asof.date()}: {len(rows)} of {len(members)} index members have feature rows ({len(dropped)} dropped, "
        f"{len(no_price)} on a forward-filled price)" + (f"; {len(skipped)} newer incomplete date(s) skipped" if skipped else ""))
    return FeatureRows(asof, rows, members, dropped, cov, no_price, list(skipped))
