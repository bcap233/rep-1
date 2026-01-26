"""
Data fetcher module for retrieving historical price data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and manages OHLCV data from Yahoo Finance."""

    def __init__(self, cache_data: bool = True):
        """
        Initialize the data fetcher.

        Args:
            cache_data: Whether to cache fetched data in memory
        """
        self.cache_data = cache_data
        self._cache: Dict[str, pd.DataFrame] = {}

    def fetch_ticker(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "5y",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single ticker.

        Args:
            ticker: Stock/ETF ticker symbol
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            period: Alternative to dates, e.g., "1y", "5y", "max"

        Returns:
            DataFrame with OHLCV data and ticker column
        """
        cache_key = f"{ticker}_{start_date}_{end_date}_{period}"

        if self.cache_data and cache_key in self._cache:
            logger.info(f"Using cached data for {ticker}")
            return self._cache[cache_key].copy()

        logger.info(f"Fetching data for {ticker}...")

        try:
            stock = yf.Ticker(ticker)

            if start_date and end_date:
                df = stock.history(start=start_date, end=end_date)
            elif start_date:
                df = stock.history(start=start_date)
            else:
                df = stock.history(period=period)

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            # Standardize column names
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })

            # Keep only OHLCV columns
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            df = df[[col for col in ohlcv_cols if col in df.columns]]

            # Add ticker column
            df["ticker"] = ticker

            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)  # Remove timezone info

            if self.cache_data:
                self._cache[cache_key] = df.copy()

            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()

    def fetch_multiple(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "5y",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            period: Alternative to dates

        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}

        for ticker in tickers:
            df = self.fetch_ticker(ticker, start_date, end_date, period)
            if not df.empty:
                results[ticker] = df

        return results

    def fetch_yield_max_with_underlying(
        self,
        yield_max_tickers: Dict[str, str],
        underlying_map: Dict[str, str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch data for yield max ETFs along with their underlying indices.

        Args:
            yield_max_tickers: Dict of ticker -> description
            underlying_map: Dict of yield_max_ticker -> underlying_ticker
            start_date: Start date
            end_date: End date

        Returns:
            Nested dict: {yield_max_ticker: {"etf": df, "underlying": df}}
        """
        results = {}

        # Get unique tickers to fetch
        all_tickers = set(yield_max_tickers.keys()) | set(underlying_map.values())
        all_data = self.fetch_multiple(list(all_tickers), start_date, end_date)

        for ym_ticker in yield_max_tickers.keys():
            if ym_ticker not in all_data:
                logger.warning(f"No data for {ym_ticker}, skipping")
                continue

            underlying_ticker = underlying_map.get(ym_ticker)
            if not underlying_ticker or underlying_ticker not in all_data:
                logger.warning(f"No underlying data for {ym_ticker}, skipping")
                continue

            results[ym_ticker] = {
                "etf": all_data[ym_ticker],
                "underlying": all_data[underlying_ticker],
                "underlying_ticker": underlying_ticker,
            }

        return results

    def get_dividend_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetch dividend history for a ticker.

        Args:
            ticker: Stock/ETF ticker symbol

        Returns:
            DataFrame with dividend data
        """
        try:
            stock = yf.Ticker(ticker)
            dividends = stock.dividends

            if dividends.empty:
                return pd.DataFrame()

            df = dividends.to_frame(name="dividend")
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)

            return df

        except Exception as e:
            logger.error(f"Error fetching dividends for {ticker}: {e}")
            return pd.DataFrame()

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
        logger.info("Cache cleared")
