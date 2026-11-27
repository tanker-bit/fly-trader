"""US stock market (NYSE) full-day holidays, from the exchange's rules, for counting trading days outside a price frame.

Rules: New Year's Day, Martin Luther King Jr. Day (3rd Monday of January, from 1998), Washington's Birthday (3rd Monday of
February), Good Friday, Memorial Day (last Monday of May), Juneteenth (June 19, from 2022), Independence Day, Labor Day (1st
Monday of September), Thanksgiving (4th Thursday of November), Christmas. A holiday on a Sunday is observed on the Monday; on a
Saturday, on the Friday before, except New Year's Day (the exchange then stays open on December 31). Plus one-off closures
(national days of mourning, 9/11, Hurricane Sandy). Half days (e.g. the day after Thanksgiving) are trading days.
Checked against SPY's actual trading days 2001-2026 in tests/test_market_calendar.py.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

import numpy as np

SPECIAL_CLOSURES = (
    "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14",        # September 11
    "2004-06-11",                                                    # Ronald Reagan, day of mourning
    "2007-01-02",                                                    # Gerald Ford
    "2012-10-29", "2012-10-30",                                      # Hurricane Sandy
    "2018-12-05",                                                    # George H. W. Bush
    "2025-01-09",                                                    # Jimmy Carter
)


def _easter(y: int) -> dt.date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return dt.date(y, month, day)


def _nth_weekday(y: int, month: int, weekday: int, n: int) -> dt.date:
    """n-th (1-based) weekday (Mon=0) of the month; n=-1 = the last one."""
    if n > 0:
        d = dt.date(y, month, 1)
        d += dt.timedelta(days=(weekday - d.weekday()) % 7)
        return d + dt.timedelta(weeks=n - 1)
    d = dt.date(y + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: dt.date, saturday_to_friday: bool = True):
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    if d.weekday() == 5:
        return d - dt.timedelta(days=1) if saturday_to_friday else None
    return d


def holidays_of_year(y: int) -> list:
    out = [_observed(dt.date(y, 1, 1), saturday_to_friday=False)]
    if y >= 1998:
        out.append(_nth_weekday(y, 1, 0, 3))
    out += [_nth_weekday(y, 2, 0, 3), _easter(y) - dt.timedelta(days=2), _nth_weekday(y, 5, 0, -1)]
    if y >= 2022:
        out.append(_observed(dt.date(y, 6, 19)))
    out += [_observed(dt.date(y, 7, 4)), _nth_weekday(y, 9, 0, 1), _nth_weekday(y, 11, 3, 4), _observed(dt.date(y, 12, 25))]
    out += [dt.date.fromisoformat(s) for s in SPECIAL_CLOSURES if int(s[:4]) == y]
    return sorted(d for d in out if d is not None and d.year == y)


@lru_cache(maxsize=4)
def nyse_holidays(start_year: int = 1990, end_year: int = 2100) -> np.ndarray:
    """Sorted datetime64[D] array of full-day closures, for numpy's busday_count / busday_offset (holidays=...)."""
    return np.array([np.datetime64(d, "D") for y in range(start_year, end_year + 1) for d in holidays_of_year(y)], dtype="datetime64[D]")
