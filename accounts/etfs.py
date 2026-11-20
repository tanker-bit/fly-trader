"""ETF price data for the accounts. The league panel holds stocks only, so the approved ETFs come from their own cache
(<root>/prices_etf, filled by experiments.fetch_etfs) and join the stock returns as extra columns."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounts.instruments import APPROVED_ETFS


def etf_fwd1(root, dates, tickers=None) -> pd.DataFrame:
    """Next-day return of each cached ETF, dates x ETFs, on the given dates: fwd[t] = adj[t+1] / adj[t] - 1 (the same convention as
    the panel's fwd_ret, so accounts.league_report.prices_from_returns turns it into a price path). ETFs with no cache are skipped;
    dates an ETF did not trade are NaN."""
    cache = Path(root) / "prices_etf"
    cols = {}
    for t in (tickers or APPROVED_ETFS):
        f = cache / f"{t}.pkl"
        if not f.exists():
            continue
        h = pd.read_pickle(f)
        adj = h["Adj Close"].astype(float)
        adj = adj[~adj.index.duplicated(keep="last")].sort_index()
        cols[t] = (adj.shift(-1) / adj - 1.0).reindex(pd.DatetimeIndex(dates))
    return pd.DataFrame(cols, index=pd.DatetimeIndex(dates))


def with_etfs(fwd1: pd.DataFrame, root, tickers=None) -> pd.DataFrame:
    """The stock return frame plus the cached ETFs as extra columns (a ticker already in the frame is left alone)."""
    e = etf_fwd1(root, fwd1.index, tickers)
    e = e[[c for c in e.columns if c not in fwd1.columns]]
    return pd.concat([fwd1, e], axis=1) if len(e.columns) else fwd1


def mask_before_listing(prices: pd.DataFrame, fwd1: pd.DataFrame) -> pd.DataFrame:
    """prices_from_returns fills a missing return with 0, so an ETF that did not exist yet would show a flat fake price. Blank those days
    (NaN) so the policy cannot buy an ETF before it was listed. Only the ETF columns of `fwd1` are touched (stocks are left alone)."""
    out = prices.copy()
    for c in out.columns:
        if c in APPROVED_ETFS and c in fwd1.columns:
            first = fwd1[c].first_valid_index()
            out.loc[:, c] = out[c].where(out.index >= first) if first is not None else float("nan")
    return out
