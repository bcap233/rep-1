"""
Configuration for the covered call / yield max backtest analysis.
"""

# Yield Max / Covered Call ETFs to analyze
YIELD_MAX_TICKERS = {
    # Global X Covered Call ETFs
    "QYLD": "NASDAQ 100 Covered Call",
    "XYLD": "S&P 500 Covered Call",
    "RYLD": "Russell 2000 Covered Call",

    # JPMorgan Premium Income
    "JEPI": "JPM Equity Premium Income",
    "JEPQ": "JPM NASDAQ Equity Premium",

    # Other yield-focused
    "DIVO": "Amplify CWP Enhanced Dividend",
    "NUSI": "Nationwide Risk-Managed Income",
}

# Underlying indices/ETFs for comparison
UNDERLYING_TICKERS = {
    "QYLD": "QQQ",   # NASDAQ 100
    "XYLD": "SPY",   # S&P 500
    "RYLD": "IWM",   # Russell 2000
    "JEPI": "SPY",   # S&P 500
    "JEPQ": "QQQ",   # NASDAQ 100
    "DIVO": "SPY",   # S&P 500
    "NUSI": "QQQ",   # NASDAQ 100
}

# SMA periods to analyze
SMA_PERIODS = [10, 20, 50, 100, 200]

# Decline detection thresholds
DECLINE_THRESHOLDS = {
    "minor": -0.05,      # 5% decline
    "moderate": -0.10,   # 10% decline
    "steep": -0.15,      # 15% decline
    "severe": -0.20,     # 20% decline
    "crash": -0.30,      # 30% decline
}

# Default analysis period
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = None  # None = today

# Output directories
OUTPUT_DIR = "output"
CHARTS_DIR = "output/charts"
REPORTS_DIR = "output/reports"
