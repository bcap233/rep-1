"""
Analysis engine for detecting declines and analyzing SMA relationships.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .indicators import TechnicalIndicators


@dataclass
class DeclinePeriod:
    """Represents a period of decline."""
    start_date: datetime
    end_date: Optional[datetime]
    peak_price: float
    trough_price: float
    max_drawdown: float
    duration_days: int
    recovery_date: Optional[datetime] = None
    recovery_days: Optional[int] = None


@dataclass
class SMACondition:
    """Represents market condition relative to SMA."""
    date: datetime
    price: float
    sma_value: float
    distance_pct: float
    position: str  # "above" or "below"


class DeclineAnalyzer:
    """Analyzes price declines and their characteristics."""

    def __init__(self, thresholds: Dict[str, float]):
        """
        Initialize with decline thresholds.

        Args:
            thresholds: Dict mapping severity name to threshold (e.g., {"steep": -0.15})
        """
        self.thresholds = thresholds

    def identify_decline_periods(
        self,
        df: pd.DataFrame,
        threshold: float = -0.10,
        min_duration: int = 1,
    ) -> List[DeclinePeriod]:
        """
        Identify periods where drawdown exceeded the threshold.

        Args:
            df: DataFrame with price data (needs 'close' column)
            threshold: Drawdown threshold (e.g., -0.10 for 10% decline)
            min_duration: Minimum days in decline to count

        Returns:
            List of DeclinePeriod objects
        """
        # Ensure we have drawdown calculated
        if "drawdown" not in df.columns:
            df = TechnicalIndicators.drawdown(df)

        # Convert threshold to percentage if needed
        threshold_pct = threshold * 100 if threshold > -1 else threshold

        declines = []
        in_decline = False
        current_decline = None

        for i, (date, row) in enumerate(df.iterrows()):
            drawdown = row["drawdown"]

            if not in_decline and drawdown <= threshold_pct:
                # Starting a new decline period
                in_decline = True
                # Find the peak (where drawdown was 0)
                peak_idx = df.index.get_loc(date)
                peak_search = df.iloc[:peak_idx + 1]
                peak_row = peak_search[peak_search["drawdown"] == 0].iloc[-1] if len(peak_search[peak_search["drawdown"] == 0]) > 0 else peak_search.iloc[0]
                peak_date = peak_row.name if hasattr(peak_row, 'name') else peak_search.index[-1]

                current_decline = {
                    "start_date": peak_date,
                    "peak_price": row["rolling_max"],
                    "trough_price": row["close"],
                    "max_drawdown": drawdown,
                    "trough_date": date,
                }

            elif in_decline:
                if drawdown <= current_decline["max_drawdown"]:
                    # Decline is deepening
                    current_decline["max_drawdown"] = drawdown
                    current_decline["trough_price"] = row["close"]
                    current_decline["trough_date"] = date

                if drawdown == 0 or i == len(df) - 1:
                    # Recovery or end of data
                    in_decline = False
                    duration = (current_decline["trough_date"] - current_decline["start_date"]).days

                    if duration >= min_duration:
                        recovery_date = date if drawdown == 0 else None
                        recovery_days = (date - current_decline["trough_date"]).days if drawdown == 0 else None

                        declines.append(DeclinePeriod(
                            start_date=current_decline["start_date"],
                            end_date=current_decline["trough_date"],
                            peak_price=current_decline["peak_price"],
                            trough_price=current_decline["trough_price"],
                            max_drawdown=current_decline["max_drawdown"],
                            duration_days=duration,
                            recovery_date=recovery_date,
                            recovery_days=recovery_days,
                        ))

        return declines

    def classify_decline_severity(self, drawdown: float) -> str:
        """
        Classify a drawdown by severity.

        Args:
            drawdown: Drawdown value (negative percentage)

        Returns:
            Severity label
        """
        # Convert to decimal if percentage
        if drawdown < -1:
            drawdown = drawdown / 100

        for severity, threshold in sorted(self.thresholds.items(), key=lambda x: x[1]):
            if drawdown <= threshold:
                return severity

        return "minimal"

    def analyze_declines_summary(
        self,
        df: pd.DataFrame,
        ticker: str = "",
    ) -> pd.DataFrame:
        """
        Create a summary of all decline periods.

        Args:
            df: DataFrame with price data
            ticker: Ticker symbol for labeling

        Returns:
            DataFrame summarizing decline periods
        """
        summaries = []

        for severity, threshold in self.thresholds.items():
            declines = self.identify_decline_periods(df, threshold)

            if declines:
                avg_drawdown = np.mean([d.max_drawdown for d in declines])
                avg_duration = np.mean([d.duration_days for d in declines])
                avg_recovery = np.mean([d.recovery_days for d in declines if d.recovery_days])

                summaries.append({
                    "ticker": ticker,
                    "severity": severity,
                    "threshold_pct": threshold * 100,
                    "num_occurrences": len(declines),
                    "avg_drawdown_pct": avg_drawdown,
                    "avg_duration_days": avg_duration,
                    "avg_recovery_days": avg_recovery if avg_recovery else None,
                    "worst_drawdown_pct": min([d.max_drawdown for d in declines]),
                    "longest_duration": max([d.duration_days for d in declines]),
                })

        return pd.DataFrame(summaries)


class SMARelationshipAnalyzer:
    """Analyzes relationships between price action, SMAs, and declines."""

    def __init__(self, sma_periods: List[int] = [10, 20, 50, 100, 200]):
        """
        Initialize with SMA periods to analyze.

        Args:
            sma_periods: List of SMA periods
        """
        self.sma_periods = sma_periods

    def analyze_sma_breakdown_events(
        self,
        df: pd.DataFrame,
        sma_period: int = 50,
    ) -> pd.DataFrame:
        """
        Analyze what happens after price breaks below an SMA.

        Args:
            df: DataFrame with price and indicator data
            sma_period: SMA period to analyze

        Returns:
            DataFrame with breakdown event analysis
        """
        result = df.copy()
        sma_col = f"sma_{sma_period}"

        if sma_col not in result.columns:
            result = TechnicalIndicators.sma(result, sma_period)

        # Ensure we have forward returns
        result = TechnicalIndicators.returns(result, [5, 10, 20, 60])

        # Identify breakdown events (price crosses below SMA)
        result["above_sma"] = result["close"] > result[sma_col]
        result["breakdown"] = (result["above_sma"].shift(1) == True) & (result["above_sma"] == False)

        # Filter to breakdown events
        breakdowns = result[result["breakdown"]].copy()

        if breakdowns.empty:
            return pd.DataFrame()

        # Analyze outcomes
        analysis = breakdowns[[
            "close", sma_col, "fwd_return_5d", "fwd_return_10d",
            "fwd_return_20d", "fwd_return_60d"
        ]].copy()

        analysis["sma_period"] = sma_period
        analysis["distance_below_sma_pct"] = (
            (analysis["close"] - analysis[sma_col]) / analysis[sma_col] * 100
        )

        return analysis

    def analyze_sma_conditions_during_declines(
        self,
        df: pd.DataFrame,
        decline_periods: List[DeclinePeriod],
    ) -> pd.DataFrame:
        """
        Analyze SMA conditions at the start of each decline.

        Args:
            df: DataFrame with price and indicator data
            decline_periods: List of DeclinePeriod objects

        Returns:
            DataFrame with SMA conditions at decline starts
        """
        results = []

        for period in self.sma_periods:
            sma_col = f"sma_{period}"
            dist_col = f"sma_{period}_dist_pct"

            if sma_col not in df.columns:
                df = TechnicalIndicators.sma(df, period)
            if dist_col not in df.columns:
                df = TechnicalIndicators.sma_distance_percent(df, [period])

        for decline in decline_periods:
            if decline.start_date not in df.index:
                continue

            row = df.loc[decline.start_date]

            result = {
                "decline_start": decline.start_date,
                "max_drawdown_pct": decline.max_drawdown,
                "duration_days": decline.duration_days,
                "price_at_start": row["close"],
            }

            for period in self.sma_periods:
                sma_col = f"sma_{period}"
                dist_col = f"sma_{period}_dist_pct"

                if sma_col in row.index and not pd.isna(row[sma_col]):
                    result[f"sma_{period}"] = row[sma_col]
                    result[f"dist_sma_{period}_pct"] = row[dist_col]
                    result[f"above_sma_{period}"] = row["close"] > row[sma_col]

            results.append(result)

        return pd.DataFrame(results)

    def analyze_forward_returns_by_sma_position(
        self,
        df: pd.DataFrame,
        sma_period: int = 50,
        forward_days: List[int] = [5, 10, 20, 60],
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze forward returns based on position relative to SMA.

        Args:
            df: DataFrame with price data
            sma_period: SMA period to analyze
            forward_days: Forward return periods

        Returns:
            Dict with analysis for above/below SMA conditions
        """
        result = df.copy()
        sma_col = f"sma_{sma_period}"

        if sma_col not in result.columns:
            result = TechnicalIndicators.sma(result, sma_period)

        result = TechnicalIndicators.returns(result, forward_days)

        # Split by position
        above_sma = result[result["close"] > result[sma_col]]
        below_sma = result[result["close"] <= result[sma_col]]

        # Calculate statistics
        stats = {}
        for position, data in [("above", above_sma), ("below", below_sma)]:
            position_stats = {"observations": len(data)}

            for days in forward_days:
                col = f"fwd_return_{days}d"
                if col in data.columns:
                    returns = data[col].dropna()
                    position_stats[f"fwd_{days}d_mean"] = returns.mean()
                    position_stats[f"fwd_{days}d_median"] = returns.median()
                    position_stats[f"fwd_{days}d_std"] = returns.std()
                    position_stats[f"fwd_{days}d_positive_pct"] = (returns > 0).mean() * 100

            stats[position] = position_stats

        return stats

    def compare_etf_vs_underlying(
        self,
        etf_df: pd.DataFrame,
        underlying_df: pd.DataFrame,
        etf_ticker: str,
        underlying_ticker: str,
    ) -> pd.DataFrame:
        """
        Compare yield max ETF performance vs underlying during different conditions.

        Args:
            etf_df: ETF price data
            underlying_df: Underlying index price data
            etf_ticker: ETF ticker symbol
            underlying_ticker: Underlying ticker symbol

        Returns:
            DataFrame comparing performance in different market conditions
        """
        # Align dates
        common_dates = etf_df.index.intersection(underlying_df.index)
        etf = etf_df.loc[common_dates].copy()
        underlying = underlying_df.loc[common_dates].copy()

        # Add indicators to underlying
        for period in self.sma_periods:
            underlying = TechnicalIndicators.sma(underlying, period)

        underlying = TechnicalIndicators.drawdown(underlying)

        # Calculate returns
        etf["etf_return"] = etf["close"].pct_change() * 100
        underlying["underlying_return"] = underlying["close"].pct_change() * 100

        # Combine
        combined = pd.DataFrame({
            "etf_close": etf["close"],
            "etf_return": etf["etf_return"],
            "underlying_close": underlying["close"],
            "underlying_return": underlying["underlying_return"],
            "underlying_drawdown": underlying["drawdown"],
        })

        for period in self.sma_periods:
            sma_col = f"sma_{period}"
            combined[f"underlying_{sma_col}"] = underlying[sma_col]
            combined[f"underlying_above_sma_{period}"] = underlying["close"] > underlying[sma_col]

        # Analyze different market conditions
        conditions = {
            "all_periods": combined,
            "bull_market": combined[combined["underlying_return"] > 0],
            "bear_market": combined[combined["underlying_return"] < 0],
            "above_50sma": combined[combined["underlying_above_sma_50"] == True],
            "below_50sma": combined[combined["underlying_above_sma_50"] == False],
            "above_200sma": combined[combined["underlying_above_sma_200"] == True],
            "below_200sma": combined[combined["underlying_above_sma_200"] == False],
            "drawdown_gt_10pct": combined[combined["underlying_drawdown"] <= -10],
            "drawdown_gt_20pct": combined[combined["underlying_drawdown"] <= -20],
        }

        results = []
        for condition_name, data in conditions.items():
            if len(data) < 10:
                continue

            result = {
                "condition": condition_name,
                "num_days": len(data),
                "etf_ticker": etf_ticker,
                "underlying_ticker": underlying_ticker,
                "etf_avg_daily_return": data["etf_return"].mean(),
                "underlying_avg_daily_return": data["underlying_return"].mean(),
                "etf_total_return": (1 + data["etf_return"] / 100).prod() - 1,
                "underlying_total_return": (1 + data["underlying_return"] / 100).prod() - 1,
                "etf_volatility": data["etf_return"].std(),
                "underlying_volatility": data["underlying_return"].std(),
                "correlation": data["etf_return"].corr(data["underlying_return"]),
            }

            # Capture ratio (ETF return / underlying return)
            if result["underlying_total_return"] != 0:
                result["capture_ratio"] = result["etf_total_return"] / result["underlying_total_return"]
            else:
                result["capture_ratio"] = np.nan

            results.append(result)

        return pd.DataFrame(results)

    def find_steep_decline_entry_points(
        self,
        df: pd.DataFrame,
        decline_threshold: float = -0.15,
        lookback_days: int = 5,
    ) -> pd.DataFrame:
        """
        Identify conditions present when steep declines begin.

        Args:
            df: DataFrame with price and indicator data
            decline_threshold: What constitutes a "steep" decline
            lookback_days: Days to look back for entry conditions

        Returns:
            DataFrame with entry point analysis
        """
        result = df.copy()

        # Ensure indicators
        result = TechnicalIndicators.add_all_indicators(result, self.sma_periods)

        # Find steep decline starts
        result["in_steep_decline"] = result["drawdown"] <= (decline_threshold * 100)
        result["decline_start"] = (
            (result["in_steep_decline"]) &
            (~result["in_steep_decline"].shift(1).fillna(False))
        )

        decline_starts = result[result["decline_start"]]

        if decline_starts.empty:
            return pd.DataFrame()

        # Analyze conditions at entry
        entries = []
        for date, row in decline_starts.iterrows():
            entry = {
                "date": date,
                "price": row["close"],
                "drawdown_at_signal": row["drawdown"],
            }

            # SMA conditions
            for period in self.sma_periods:
                sma_col = f"sma_{period}"
                dist_col = f"sma_{period}_dist_pct"

                if sma_col in row.index:
                    entry[f"above_sma_{period}"] = row["close"] > row[sma_col]
                    entry[f"dist_sma_{period}"] = row[dist_col] if dist_col in row.index else np.nan

            # Momentum
            if "return_5d" in row.index:
                entry["momentum_5d"] = row["return_5d"]
            if "return_20d" in row.index:
                entry["momentum_20d"] = row["return_20d"]

            # Volatility
            if "volatility_20d" in row.index:
                entry["volatility_20d"] = row["volatility_20d"]

            entries.append(entry)

        return pd.DataFrame(entries)
