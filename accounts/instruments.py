"""What a fly is allowed to own.

Allowed: common stocks (the league's universe) and plain long-only ETFs from an approved list.
Never allowed: options and other derivatives, shorting, margin, and inverse / leveraged / volatility ETFs
(they are shorting or leverage in disguise). Anything not on the lists is simply not tradable.
"""
from __future__ import annotations

START_CASH = 1_000_000.0

KIND_STOCK, KIND_ETF = "STOCK", "ETF"

# Plain, unleveraged, long-only ETFs: broad market, size, sectors, bonds, gold.
APPROVED_ETFS = (
    "SPY", "VOO", "IVV", "VTI", "QQQ", "DIA", "IWM", "MDY", "IJR",                      # US equity, broad / size
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",       # S&P sector SPDRs
    "TLT", "IEF", "SHY", "BIL", "AGG", "LQD", "HYG", "TIP",                             # Treasuries, bonds, cash-like
    "GLD", "IAU", "EFA", "EEM", "VNQ",                                                  # gold, international, real estate
)

# Inverse, leveraged and volatility products: never tradable, even if someone adds them to a list by mistake.
FORBIDDEN_ETFS = frozenset({
    "SH", "PSQ", "DOG", "SDS", "QID", "DXD", "SPXU", "SPXS", "SQQQ", "SDOW", "TZA", "SRTY", "SOXS", "FAZ", "SKF", "TBT", "TMV",
    "SSO", "QLD", "UWM", "UDOW", "SPXL", "UPRO", "TQQQ", "TNA", "SOXL", "FAS", "ERX", "LABU", "TMF", "UGL", "NUGT", "JNUG",
    "UVXY", "SVXY", "VXX", "VIXY", "UVIX", "SVIX",
})


def looks_like_option(ticker: str) -> bool:
    """OCC-style option symbols (e.g. AAPL240621C00190000): root + 6-digit date + C/P + 8-digit strike."""
    t = str(ticker).strip().upper()
    return len(t) >= 15 and t[-9] in "CP" and t[-8:].isdigit() and t[-15:-9].isdigit()


class Universe:
    """The set of tradable tickers and what kind each is. Stocks come from the league's panel; ETFs from the approved list."""

    def __init__(self, stocks=(), etfs=APPROVED_ETFS):
        self.stocks = {str(t).upper() for t in stocks}
        self.etfs = {str(t).upper() for t in etfs} - FORBIDDEN_ETFS
        self.stocks -= FORBIDDEN_ETFS
        self.stocks -= self.etfs                       # a ticker is one kind only (ETF wins)

    def kind(self, ticker: str):
        t = str(ticker).upper()
        if t in self.etfs:
            return KIND_ETF
        if t in self.stocks:
            return KIND_STOCK
        return None

    def check(self, ticker: str):
        """None if tradable, otherwise the reason it is not."""
        t = str(ticker).upper()
        if t in FORBIDDEN_ETFS:
            return "inverse / leveraged / volatility product"
        if looks_like_option(t):
            return "options are not allowed"
        if self.kind(t) is None:
            return "not in the tradable universe"
        return None

    def __contains__(self, ticker) -> bool:
        return self.check(ticker) is None
