"""TSX stock universe and market-context series.

IMPORTANT: this is a list of large, well-known TSX names as of writing, not the historical
index membership. Using today's winners to test the past builds in survivorship bias (stocks
that later failed or were dropped are missing), which flatters backtests. Treat results as
optimistic. Tickers that no longer resolve are skipped and reported by build_dataset.
Edit this file freely: it is plain data.
"""

TSX_STOCKS = [
    # Banks / insurers / financials
    "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO",
    "MFC.TO", "SLF.TO", "GWO.TO", "IFC.TO", "POW.TO", "BN.TO", "BAM.TO",
    # Energy / pipelines
    "CNQ.TO", "SU.TO", "CVE.TO", "IMO.TO", "TOU.TO", "ARX.TO",
    "ENB.TO", "TRP.TO", "PPL.TO",
    # Industrials / transport
    "CNR.TO", "CP.TO", "WCN.TO", "TFII.TO", "WSP.TO", "STN.TO",
    # Tech
    "SHOP.TO", "CSU.TO", "OTEX.TO", "GIB-A.TO",
    # Telecom / media
    "BCE.TO", "T.TO", "RCI-B.TO", "TRI.TO",
    # Consumer / staples
    "ATD.TO", "L.TO", "MRU.TO", "DOL.TO", "QSR.TO", "CTC-A.TO", "WN.TO", "SAP.TO",
    # Materials / mining
    "FNV.TO", "WPM.TO", "AEM.TO", "ABX.TO", "K.TO", "AGI.TO",
    "NTR.TO", "TECK-B.TO", "FM.TO", "CCO.TO",
    # Utilities
    "FTS.TO", "EMA.TO", "H.TO", "AQN.TO", "BIP-UN.TO",
]

# Market-wide context (same value for every stock on a given day).
MARKET_TICKERS = {
    "tsx": "^GSPTSE",    # TSX composite index
    "vix": "^VIX",       # US volatility index
    "usdcad": "CAD=X",   # USD/CAD exchange rate
    "oil": "CL=F",       # WTI crude oil futures
}
