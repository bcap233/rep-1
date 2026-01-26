"""
Technical indicators module for SMA and related calculations.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union


class TechnicalIndicators:
    """Calculate technical indicators for price analysis."""

    @staticmethod
    def sma(
        df: pd.DataFrame,
        periods: Union[int, List[int]],
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate Simple Moving Average(s).

        Args:
            df: DataFrame with price data
            periods: Single period or list of periods
            column: Column to calculate SMA on

        Returns:
            DataFrame with SMA columns added
        """
        result = df.copy()

        if isinstance(periods, int):
            periods = [periods]

        for period in periods:
            col_name = f"sma_{period}"
            result[col_name] = result[column].rolling(window=period).mean()

        return result

    @staticmethod
    def ema(
        df: pd.DataFrame,
        periods: Union[int, List[int]],
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate Exponential Moving Average(s).

        Args:
            df: DataFrame with price data
            periods: Single period or list of periods
            column: Column to calculate EMA on

        Returns:
            DataFrame with EMA columns added
        """
        result = df.copy()

        if isinstance(periods, int):
            periods = [periods]

        for period in periods:
            col_name = f"ema_{period}"
            result[col_name] = result[column].ewm(span=period, adjust=False).mean()

        return result

    @staticmethod
    def price_to_sma_ratio(
        df: pd.DataFrame,
        sma_periods: List[int],
        price_column: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate the ratio of price to various SMAs.
        Values > 1 mean price is above SMA, < 1 means below.

        Args:
            df: DataFrame with price and SMA data
            sma_periods: List of SMA periods to calculate ratios for
            price_column: Price column to use

        Returns:
            DataFrame with price/SMA ratio columns
        """
        result = df.copy()

        for period in sma_periods:
            sma_col = f"sma_{period}"
            if sma_col not in result.columns:
                result = TechnicalIndicators.sma(result, period, price_column)

            ratio_col = f"price_sma_{period}_ratio"
            result[ratio_col] = result[price_column] / result[sma_col]

        return result

    @staticmethod
    def sma_distance_percent(
        df: pd.DataFrame,
        sma_periods: List[int],
        price_column: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate percentage distance from price to SMAs.
        Positive = above SMA, Negative = below SMA.

        Args:
            df: DataFrame with price data
            sma_periods: List of SMA periods
            price_column: Price column to use

        Returns:
            DataFrame with distance percentage columns
        """
        result = df.copy()

        for period in sma_periods:
            sma_col = f"sma_{period}"
            if sma_col not in result.columns:
                result = TechnicalIndicators.sma(result, period, price_column)

            dist_col = f"sma_{period}_dist_pct"
            result[dist_col] = ((result[price_column] - result[sma_col]) / result[sma_col]) * 100

        return result

    @staticmethod
    def sma_crossover_signals(
        df: pd.DataFrame,
        fast_period: int,
        slow_period: int,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Detect SMA crossover signals.

        Args:
            df: DataFrame with price data
            fast_period: Fast SMA period
            slow_period: Slow SMA period
            column: Price column

        Returns:
            DataFrame with crossover signal columns
        """
        result = df.copy()

        fast_sma = f"sma_{fast_period}"
        slow_sma = f"sma_{slow_period}"

        if fast_sma not in result.columns:
            result = TechnicalIndicators.sma(result, fast_period, column)
        if slow_sma not in result.columns:
            result = TechnicalIndicators.sma(result, slow_period, column)

        # 1 when fast > slow, -1 when fast < slow
        result["sma_position"] = np.where(
            result[fast_sma] > result[slow_sma], 1,
            np.where(result[fast_sma] < result[slow_sma], -1, 0)
        )

        # Crossover signal: 1 = golden cross, -1 = death cross
        result["sma_crossover"] = result["sma_position"].diff()
        result["golden_cross"] = result["sma_crossover"] == 2  # Changed from -1 to 1
        result["death_cross"] = result["sma_crossover"] == -2  # Changed from 1 to -1

        return result

    @staticmethod
    def returns(
        df: pd.DataFrame,
        periods: List[int] = [1, 5, 10, 20, 60],
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate returns over various periods.

        Args:
            df: DataFrame with price data
            periods: List of periods for return calculation
            column: Price column

        Returns:
            DataFrame with return columns
        """
        result = df.copy()

        for period in periods:
            # Forward returns (what happens AFTER this point)
            result[f"fwd_return_{period}d"] = (
                result[column].shift(-period) / result[column] - 1
            ) * 100

            # Backward returns (what happened BEFORE this point)
            result[f"return_{period}d"] = (
                result[column] / result[column].shift(period) - 1
            ) * 100

        return result

    @staticmethod
    def drawdown(df: pd.DataFrame, column: str = "close") -> pd.DataFrame:
        """
        Calculate drawdown from rolling maximum.

        Args:
            df: DataFrame with price data
            column: Price column

        Returns:
            DataFrame with drawdown columns
        """
        result = df.copy()

        # Rolling maximum
        result["rolling_max"] = result[column].expanding().max()

        # Drawdown (negative percentage from peak)
        result["drawdown"] = (result[column] / result["rolling_max"] - 1) * 100

        # Drawdown duration (days since last peak)
        result["at_peak"] = result[column] >= result["rolling_max"]
        result["days_since_peak"] = (~result["at_peak"]).groupby(
            result["at_peak"].cumsum()
        ).cumsum()

        return result

    @staticmethod
    def volatility(
        df: pd.DataFrame,
        windows: List[int] = [10, 20, 60],
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate rolling volatility (standard deviation of returns).

        Args:
            df: DataFrame with price data
            windows: Rolling window sizes
            column: Price column

        Returns:
            DataFrame with volatility columns
        """
        result = df.copy()

        # Daily returns
        if "daily_return" not in result.columns:
            result["daily_return"] = result[column].pct_change() * 100

        for window in windows:
            result[f"volatility_{window}d"] = (
                result["daily_return"].rolling(window=window).std()
            )
            # Annualized volatility
            result[f"volatility_{window}d_ann"] = (
                result[f"volatility_{window}d"] * np.sqrt(252)
            )

        return result

    @staticmethod
    def add_all_indicators(
        df: pd.DataFrame,
        sma_periods: List[int] = [10, 20, 50, 100, 200],
        return_periods: List[int] = [1, 5, 10, 20, 60],
        volatility_windows: List[int] = [10, 20, 60],
    ) -> pd.DataFrame:
        """
        Add all technical indicators to a DataFrame.

        Args:
            df: DataFrame with OHLCV data
            sma_periods: SMA periods to calculate
            return_periods: Return periods to calculate
            volatility_windows: Volatility windows to calculate

        Returns:
            DataFrame with all indicators added
        """
        result = df.copy()

        result = TechnicalIndicators.sma(result, sma_periods)
        result = TechnicalIndicators.price_to_sma_ratio(result, sma_periods)
        result = TechnicalIndicators.sma_distance_percent(result, sma_periods)
        result = TechnicalIndicators.returns(result, return_periods)
        result = TechnicalIndicators.drawdown(result)
        result = TechnicalIndicators.volatility(result, volatility_windows)

        return result
