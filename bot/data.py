"""
Data fetcher for the V7 rotation bot.

Pulls daily price + dividend data from Yahoo Finance for each pair.
Returns DataFrames in the same format the backtest engine expects:
  - Index: DatetimeIndex named "Date"
  - Columns: close, dividend, split (all lowercase)
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# CSV file mapping for local/fallback data
LOCAL_CSV_MAP = {
    "MSTY": "msty_data_with_dividends.csv",
    "WNTR": "wntr_data_with_dividends.csv",
    "NVDY": "nvdy_data_with_dividends.csv",
    "DIPS": "dips_data_with_dividends.csv",
    "CONY": "cony_data_with_dividends.csv",
    "FIAT": "fiat_data_with_dividends.csv",
    "TSLY": "tsly_data_with_dividends.csv",
    "CRSH": "crsh_data_with_dividends.csv",
}


def load_local_csv(ticker: str) -> pd.DataFrame:
    """
    Load data from local CSV files (backtest data).
    Used as fallback when yfinance is unavailable.
    """
    csv_name = LOCAL_CSV_MAP.get(ticker.upper())
    if not csv_name:
        logger.error(f"No local CSV mapping for {ticker}")
        return pd.DataFrame()

    csv_path = Path(csv_name)
    if not csv_path.exists():
        # Try relative to project root
        csv_path = Path(__file__).parent.parent / csv_name

    if not csv_path.exists():
        logger.error(f"Local CSV not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date").sort_index()
    df.columns = [c.lower().strip() for c in df.columns]
    logger.info(f"  {ticker}: loaded {len(df)} rows from {csv_path.name}")
    return df


def fetch_etf_data(
    ticker: str,
    lookback_days: int = 200,
    end_date: Optional[str] = None,
    use_local: bool = False,
) -> pd.DataFrame:
    """
    Fetch daily close, dividends, and splits for one ETF.

    Returns DataFrame with columns: close, dividend, split
    Index: DatetimeIndex (tz-naive).
    """
    # Use local CSV if requested or as fallback
    if use_local:
        return load_local_csv(ticker)

    if end_date:
        end = pd.Timestamp(end_date)
    else:
        end = pd.Timestamp.now()

    # Fetch extra buffer for SMA warm-up
    start = end - timedelta(days=int(lookback_days * 1.8))

    logger.info(f"Fetching {ticker} from {start.date()} to {end.date()}")

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)

        # Price history
        hist = stock.history(start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"))

        if hist.empty:
            logger.error(f"No price data for {ticker}")
            return pd.DataFrame()

        # Normalize timezone
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        hist.index.name = "Date"

        # Build output DataFrame
        df = pd.DataFrame(index=hist.index)
        df["close"] = hist["Close"]

        # Dividends — yfinance includes them in history
        df["dividend"] = hist["Dividends"] if "Dividends" in hist.columns else 0.0

        # Stock splits
        df["split"] = hist["Stock Splits"] if "Stock Splits" in hist.columns else 0.0
        # Convert split column: 0 means no split (we use 1 for "no split" in our system)
        df["split"] = df["split"].replace(0, 1)

        df = df.sort_index()
        logger.info(f"  {ticker}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
        return df

    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        # Fall back to local CSV
        logger.info(f"Falling back to local CSV for {ticker}")
        return load_local_csv(ticker)


def fetch_pair_data(
    bull_ticker: str,
    bear_ticker: str,
    lookback_days: int = 200,
    use_local: bool = False,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Fetch data for a bull/bear pair. Trims both to their overlapping date range.
    """
    bull_df = fetch_etf_data(bull_ticker, lookback_days, use_local=use_local)
    bear_df = fetch_etf_data(bear_ticker, lookback_days, use_local=use_local)

    if bull_df.empty or bear_df.empty:
        return None, None

    # Trim to overlapping dates
    start = max(bull_df.index.min(), bear_df.index.min())
    end = min(bull_df.index.max(), bear_df.index.max())
    bull_df = bull_df.loc[start:end]
    bear_df = bear_df.loc[start:end]

    return bull_df, bear_df
