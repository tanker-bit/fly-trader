import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from accounts.market_calendar import holidays_of_year, nyse_holidays


def test_known_holidays():
    h = set(holidays_of_year(2026))
    for d in ("2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
              "2026-11-26", "2026-12-25"):
        assert dt.date.fromisoformat(d) in h, d
    assert dt.date(2026, 11, 27) not in h                                  # the day after Thanksgiving is a half day, open
    assert dt.date(2021, 12, 31) not in set(holidays_of_year(2021))         # New Year's on a Saturday: no Friday closure
    assert dt.date(2022, 6, 20) in set(holidays_of_year(2022))              # Juneteenth on a Sunday -> Monday
    assert dt.date(2021, 6, 18) not in set(holidays_of_year(2021))          # before 2022 no Juneteenth closure
    assert dt.date(2025, 1, 9) in set(holidays_of_year(2025))               # one-off closure


def test_busday_counting_skips_holidays():
    hol = nyse_holidays()
    assert np.busday_count("2026-11-25", "2026-11-30", holidays=hol) == 2   # Wed 25 and Fri 27 (Thanksgiving closed)
    assert str(np.busday_offset("2026-11-27", -1, holidays=hol)) == "2026-11-25"


SPY = Path("data_fundamentals/prices_etf/SPY.pkl")


@pytest.mark.skipif(not SPY.exists(), reason="SPY price history not on this PC")
def test_calendar_matches_the_real_trading_days_2001_to_now():
    days = pd.DatetimeIndex(pd.read_pickle(SPY).index).normalize()
    days = days[(days >= "2001-01-01")]
    last = days.max()
    weekdays = pd.bdate_range("2001-01-01", last)
    hol = pd.DatetimeIndex(nyse_holidays().astype("datetime64[ns]"))
    predicted = weekdays.difference(hol)
    missing_from_data = predicted.difference(days)                          # we say open, the data has no bar
    traded_on_holiday = days.intersection(hol)                              # we say closed, the data has a bar
    assert list(traded_on_holiday) == []
    assert len(missing_from_data) <= 2, list(missing_from_data)             # allow a stray gap in Yahoo's data
