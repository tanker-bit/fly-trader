"""Run: python -m pytest tests/test_build_us_universe.py -q"""
import pandas as pd

from experiments.build_us_universe import build, looks_like_common, parse_nasdaq, parse_other

NASDAQ = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust, Series 1|G|N|N|100|Y|N
ZZZT|Test Co - Common Stock|Q|Y|N|100|N|N
BADX|Distressed Inc. - Common Stock|S|N|D|100|N|N
ABCDW|ABC Acquisition Corp - Warrant|S|N|N|100|N|N
ABCDR|ABC Acquisition Corp - Right|S|N|N|100|N|N
ABCDU|ABC Acquisition Corp - Unit|S|N|N|100|N|N
FITBP|Fifth Third Bancorp - Depositary Shares, each representing 1/40th Preferred|Q|N|N|100|N|N
File Creation Time: 1122202618:00|||||||
"""
OTHER = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
IBM|International Business Machines Corporation Common Stock|N|IBM|N|100|N|IBM
BRK.B|Berkshire Hathaway Inc. Class B Common Stock|N|BRK.B|N|100|N|BRK.B
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
GSAT|Globalstar, Inc. Common Stock|A|GSAT|N|100|N|GSAT
NYSEONLY$|Something Preferred|N|X|N|100|N|X
JPM$C|JPMorgan Chase Preferred Series C|N|JPM$C|N|100|N|JPM$C
CBOEX|Odd Cboe Only Line Common Stock|Z|CBOEX|N|100|N|CBOEX
File Creation Time: 1122202618:00||||||
"""


def test_nasdaq_and_other_parsing_ignore_footers():
    assert len(parse_nasdaq(NASDAQ)) == 8 and len(parse_other(OTHER)) == 7
    assert "File Creation Time" not in " ".join(parse_nasdaq(NASDAQ)["raw"])


def test_only_common_stocks_on_the_three_exchanges_survive_and_etfs_are_flagged():
    u = build(NASDAQ, OTHER)
    stocks = set(u[~u.etf].symbol)
    assert stocks == {"AAPL", "IBM", "BRK-B", "GSAT"}, stocks
    assert set(u[u.etf].symbol) == {"QQQ", "SPY"}
    assert u.set_index("symbol").loc["BRK-B", "exchange"] == "NYSE" and u.set_index("symbol").loc["GSAT", "exchange"] == "NYSE American"
    assert u.symbol.is_unique


def test_dropped_lines():
    u = set(build(NASDAQ, OTHER).symbol)
    for bad in ("ZZZT", "BADX", "ABCDW", "ABCDR", "ABCDU", "FITBP", "NYSEONLY$", "JPM$C", "CBOEX"):
        assert bad not in u, bad


def test_name_rule():
    assert looks_like_common("Apple Inc. - Common Stock")
    assert looks_like_common("Taiwan Semiconductor Manufacturing Company Ltd. American Depositary Shares")
    assert not looks_like_common("Foo Corp - Warrant")
    assert not looks_like_common("Foo Corp Common Stock and Warrant units")
    assert not looks_like_common("Bar Inc 6.5% Notes due 2030")
    assert looks_like_common("Community Health Systems, Inc. Common Stock")           # "unit" inside a word is fine
    assert looks_like_common("Senior Housing Properties Trust Common Shares of Beneficial Interest")
    assert looks_like_common("Copyright Clearance Corp Common Stock")
    assert looks_like_common("Energy Transfer LP Common Units representing limited partner interests")
    assert not looks_like_common("SPAC Acquisition Corp Units, each consisting of one share")
