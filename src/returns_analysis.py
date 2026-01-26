"""
Returns regime analysis module.

Analyzes how total returns shift over time based on position relative to the 50 SMA.
Key insight: Covered call products show asymmetric return profiles:
- Near/above SMA: Returns capped but stable
- Below SMA trending down: Returns skew heavily negative (reflexive collapse)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


@dataclass
class RegimeStats:
    """Statistics for a particular SMA regime."""
    regime: str
    num_days: int
    total_return_pct: float
    avg_daily_return: float
    median_daily_return: float
    std_daily_return: float
    skewness: float
    worst_day: float
    best_day: float
    pct_positive_days: float
    avg_5d_return: float
    avg_10d_return: float
    avg_20d_return: float


class ReturnsRegimeAnalyzer:
    """
    Analyzes returns based on position relative to the 50 SMA.

    Key questions answered:
    1. How do returns differ when above vs below the 50 SMA?
    2. How does return skew change as distance from SMA increases?
    3. What's the cumulative return in each regime?
    """

    def __init__(self, sma_period: int = 50):
        self.sma_period = sma_period

    def prepare_returns_data(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
    ) -> pd.DataFrame:
        """
        Prepare dataframe with returns and SMA regime classifications.

        Args:
            df: DataFrame with price data
            price_col: Column name for price

        Returns:
            DataFrame with returns and regime data
        """
        result = df.copy()

        # Ensure we have price column
        if price_col not in result.columns:
            if "price" in result.columns:
                price_col = "price"
            else:
                raise ValueError(f"Price column '{price_col}' not found")

        # Calculate SMA if not present
        sma_col = f"sma_{self.sma_period}"
        if sma_col not in result.columns:
            result[sma_col] = result[price_col].rolling(window=self.sma_period).mean()

        # Daily returns
        result["daily_return"] = result[price_col].pct_change() * 100

        # Rolling returns for various periods
        for period in [5, 10, 20, 60]:
            result[f"return_{period}d"] = (
                result[price_col].pct_change(period) * 100
            )

        # Distance from SMA
        result["sma_distance_pct"] = (
            (result[price_col] - result[sma_col]) / result[sma_col] * 100
        )

        # SMA position regimes
        result["above_sma"] = result[price_col] > result[sma_col]

        # More granular regimes based on distance
        result["regime"] = pd.cut(
            result["sma_distance_pct"],
            bins=[-np.inf, -20, -10, -5, 0, 5, 10, 20, np.inf],
            labels=[
                "deep_below_20",   # < -20%
                "below_10_20",     # -20% to -10%
                "below_5_10",      # -10% to -5%
                "below_0_5",       # -5% to 0%
                "above_0_5",       # 0% to 5%
                "above_5_10",      # 5% to 10%
                "above_10_20",     # 10% to 20%
                "far_above_20",    # > 20%
            ]
        )

        # Simplified regime (3 categories)
        result["regime_simple"] = np.where(
            result["sma_distance_pct"] > 5, "trending_up",
            np.where(result["sma_distance_pct"] < -5, "trending_down", "near_sma")
        )

        # Trend direction (is price moving toward or away from SMA?)
        result["sma_distance_change"] = result["sma_distance_pct"].diff()
        result["moving_toward_sma"] = (
            (result["sma_distance_pct"] > 0) & (result["sma_distance_change"] < 0) |
            (result["sma_distance_pct"] < 0) & (result["sma_distance_change"] > 0)
        )

        # Cumulative return
        result["cumulative_return"] = (1 + result["daily_return"] / 100).cumprod() - 1
        result["cumulative_return_pct"] = result["cumulative_return"] * 100

        return result

    def calculate_regime_stats(
        self,
        df: pd.DataFrame,
        regime_col: str = "regime_simple",
    ) -> Dict[str, RegimeStats]:
        """
        Calculate detailed statistics for each regime.

        Args:
            df: Prepared DataFrame from prepare_returns_data
            regime_col: Column to use for regime classification

        Returns:
            Dict mapping regime name to RegimeStats
        """
        stats = {}

        for regime in df[regime_col].dropna().unique():
            regime_data = df[df[regime_col] == regime]

            if len(regime_data) < 5:
                continue

            daily_returns = regime_data["daily_return"].dropna()

            # Calculate skewness manually to avoid scipy dependency issues
            if len(daily_returns) > 2:
                mean = daily_returns.mean()
                std = daily_returns.std()
                if std > 0:
                    skew = ((daily_returns - mean) ** 3).mean() / (std ** 3)
                else:
                    skew = 0
            else:
                skew = 0

            # Total return in this regime
            regime_returns = regime_data["daily_return"].dropna() / 100
            total_return = (1 + regime_returns).prod() - 1

            stats[regime] = RegimeStats(
                regime=regime,
                num_days=len(regime_data),
                total_return_pct=total_return * 100,
                avg_daily_return=daily_returns.mean(),
                median_daily_return=daily_returns.median(),
                std_daily_return=daily_returns.std(),
                skewness=skew,
                worst_day=daily_returns.min(),
                best_day=daily_returns.max(),
                pct_positive_days=(daily_returns > 0).mean() * 100,
                avg_5d_return=regime_data["return_5d"].dropna().mean(),
                avg_10d_return=regime_data["return_10d"].dropna().mean(),
                avg_20d_return=regime_data["return_20d"].dropna().mean(),
            )

        return stats

    def calculate_rolling_regime_returns(
        self,
        df: pd.DataFrame,
        window: int = 20,
    ) -> pd.DataFrame:
        """
        Calculate rolling returns segmented by current regime.
        Shows how returns accumulate differently in each regime over time.

        Args:
            df: Prepared DataFrame
            window: Rolling window for return calculation

        Returns:
            DataFrame with rolling regime returns
        """
        result = df.copy()

        # Rolling return
        result[f"rolling_{window}d_return"] = (
            result["daily_return"].rolling(window=window).sum()
        )

        # Rolling return by regime
        for regime in ["trending_up", "near_sma", "trending_down"]:
            mask = result["regime_simple"] == regime
            result[f"return_in_{regime}"] = np.where(
                mask, result["daily_return"], 0
            )
            result[f"cumul_return_{regime}"] = (
                result[f"return_in_{regime}"].cumsum()
            )

        return result

    def analyze_return_distribution_by_sma_distance(
        self,
        df: pd.DataFrame,
        distance_bins: List[float] = [-25, -15, -10, -5, 0, 5, 10, 15],
    ) -> pd.DataFrame:
        """
        Analyze how return distribution changes at different distances from SMA.

        Args:
            df: Prepared DataFrame
            distance_bins: Distance thresholds to analyze

        Returns:
            DataFrame with distribution stats by distance
        """
        results = []

        for i in range(len(distance_bins) - 1):
            lower = distance_bins[i]
            upper = distance_bins[i + 1]

            mask = (df["sma_distance_pct"] >= lower) & (df["sma_distance_pct"] < upper)
            subset = df[mask]

            if len(subset) < 5:
                continue

            daily_returns = subset["daily_return"].dropna()

            # Calculate percentiles
            results.append({
                "distance_range": f"{lower}% to {upper}%",
                "lower_bound": lower,
                "upper_bound": upper,
                "num_days": len(subset),
                "avg_daily_return": daily_returns.mean(),
                "median_daily_return": daily_returns.median(),
                "std_return": daily_returns.std(),
                "p10_return": daily_returns.quantile(0.10),
                "p25_return": daily_returns.quantile(0.25),
                "p75_return": daily_returns.quantile(0.75),
                "p90_return": daily_returns.quantile(0.90),
                "pct_negative": (daily_returns < 0).mean() * 100,
                "avg_negative_return": daily_returns[daily_returns < 0].mean() if (daily_returns < 0).any() else 0,
                "avg_positive_return": daily_returns[daily_returns > 0].mean() if (daily_returns > 0).any() else 0,
            })

        return pd.DataFrame(results)

    def analyze_regime_transitions(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Analyze what happens during regime transitions.

        Args:
            df: Prepared DataFrame

        Returns:
            DataFrame with transition analysis
        """
        result = df.copy()

        # Identify regime changes
        result["regime_change"] = result["regime_simple"].ne(result["regime_simple"].shift())
        result["prev_regime"] = result["regime_simple"].shift()

        # Forward returns after transition
        for days in [5, 10, 20]:
            result[f"fwd_return_{days}d"] = result["daily_return"].rolling(days).sum().shift(-days)

        # Filter to transition days
        transitions = result[result["regime_change"] & result["prev_regime"].notna()].copy()

        if transitions.empty:
            return pd.DataFrame()

        # Group by transition type
        transition_stats = []
        for (prev, curr), group in transitions.groupby(["prev_regime", "regime_simple"]):
            if len(group) < 2:
                continue

            transition_stats.append({
                "from_regime": prev,
                "to_regime": curr,
                "num_transitions": len(group),
                "avg_5d_fwd_return": group["fwd_return_5d"].mean(),
                "avg_10d_fwd_return": group["fwd_return_10d"].mean(),
                "avg_20d_fwd_return": group["fwd_return_20d"].mean(),
            })

        return pd.DataFrame(transition_stats)


class ReturnsVisualizer:
    """Visualization for returns regime analysis."""

    def __init__(self, output_dir: str = "output/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_returns_by_regime(
        self,
        df: pd.DataFrame,
        regime_stats: Dict[str, RegimeStats],
        ticker: str = "MSTY",
    ) -> plt.Figure:
        """
        Comprehensive visualization of returns by SMA regime.
        """
        fig, axes = plt.subplots(3, 2, figsize=(14, 14))

        # 1. Price with regime shading
        ax1 = axes[0, 0]
        ax1.plot(df.index, df["close"] if "close" in df.columns else df["price"],
                 color="black", linewidth=1)

        # Shade by regime
        for regime, color in [("trending_up", "green"), ("near_sma", "yellow"), ("trending_down", "red")]:
            mask = df["regime_simple"] == regime
            if mask.any():
                ax1.fill_between(df.index, df["close"].min(), df["close"].max(),
                               where=mask, alpha=0.2, color=color, label=regime)

        ax1.set_title(f"{ticker} Price with SMA Regime Shading", fontweight="bold")
        ax1.set_ylabel("Price ($)")
        ax1.legend(loc="upper left", fontsize=8)

        # 2. Cumulative returns by regime
        ax2 = axes[0, 1]
        for regime, color in [("trending_up", "green"), ("near_sma", "gold"), ("trending_down", "red")]:
            col = f"cumul_return_{regime}"
            if col in df.columns:
                ax2.plot(df.index, df[col], label=regime, color=color, linewidth=1.5)

        ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax2.set_title("Cumulative Return by Regime", fontweight="bold")
        ax2.set_ylabel("Cumulative Return (%)")
        ax2.legend()

        # 3. Return distribution comparison (box plot style)
        ax3 = axes[1, 0]
        regime_returns = {}
        for regime in ["trending_down", "near_sma", "trending_up"]:
            returns = df[df["regime_simple"] == regime]["daily_return"].dropna()
            if len(returns) > 0:
                regime_returns[regime] = returns

        if regime_returns:
            positions = list(range(len(regime_returns)))
            bp = ax3.boxplot(regime_returns.values(), positions=positions,
                           patch_artist=True, widths=0.6)
            colors = ["red", "gold", "green"]
            for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

            ax3.set_xticks(positions)
            ax3.set_xticklabels(regime_returns.keys())
            ax3.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
            ax3.set_title("Daily Return Distribution by Regime", fontweight="bold")
            ax3.set_ylabel("Daily Return (%)")

        # 4. Return stats bar chart
        ax4 = axes[1, 1]
        if regime_stats:
            regimes = list(regime_stats.keys())
            x = np.arange(len(regimes))
            width = 0.25

            avg_returns = [regime_stats[r].avg_daily_return for r in regimes]
            pct_positive = [regime_stats[r].pct_positive_days for r in regimes]

            colors = ["red" if r == "trending_down" else "gold" if r == "near_sma" else "green"
                     for r in regimes]

            ax4.bar(x, avg_returns, width=0.6, color=colors, alpha=0.7)
            ax4.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
            ax4.set_xticks(x)
            ax4.set_xticklabels(regimes)
            ax4.set_title("Average Daily Return by Regime", fontweight="bold")
            ax4.set_ylabel("Avg Daily Return (%)")

            # Add pct positive as text
            for i, (ret, pct) in enumerate(zip(avg_returns, pct_positive)):
                ax4.annotate(f"{pct:.0f}% +", (i, ret), ha="center",
                           va="bottom" if ret > 0 else "top", fontsize=9)

        # 5. SMA distance vs next day return scatter
        ax5 = axes[2, 0]
        sample = df.dropna(subset=["sma_distance_pct", "daily_return"])
        if len(sample) > 0:
            colors = sample["sma_distance_pct"].apply(
                lambda x: "green" if x > 5 else "red" if x < -5 else "gold"
            )
            ax5.scatter(sample["sma_distance_pct"], sample["daily_return"].shift(-1),
                       alpha=0.4, c=colors, s=10)
            ax5.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
            ax5.axvline(x=0, color="blue", linestyle="--", linewidth=0.5)
            ax5.axvline(x=-5, color="orange", linestyle="--", alpha=0.5)
            ax5.axvline(x=-10, color="red", linestyle="--", alpha=0.5)
            ax5.set_xlabel("SMA Distance (%)")
            ax5.set_ylabel("Next Day Return (%)")
            ax5.set_title("SMA Distance vs Next Day Return", fontweight="bold")

        # 6. Total return breakdown
        ax6 = axes[2, 1]
        if regime_stats:
            regimes = list(regime_stats.keys())
            total_returns = [regime_stats[r].total_return_pct for r in regimes]
            num_days = [regime_stats[r].num_days for r in regimes]

            colors = ["red" if r == "trending_down" else "gold" if r == "near_sma" else "green"
                     for r in regimes]

            bars = ax6.bar(regimes, total_returns, color=colors, alpha=0.7)
            ax6.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
            ax6.set_title("Total Return Earned in Each Regime", fontweight="bold")
            ax6.set_ylabel("Total Return (%)")

            # Add day count labels
            for bar, days in zip(bars, num_days):
                height = bar.get_height()
                ax6.annotate(f"{days}d", (bar.get_x() + bar.get_width()/2, height),
                           ha="center", va="bottom" if height > 0 else "top", fontsize=9)

        plt.tight_layout()

        filepath = self.output_dir / f"{ticker}_returns_regime_analysis.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")

        return fig

    def plot_return_distribution_by_distance(
        self,
        distribution_df: pd.DataFrame,
        ticker: str = "MSTY",
    ) -> plt.Figure:
        """
        Visualize how return distribution changes at different SMA distances.
        """
        if distribution_df.empty:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        df = distribution_df.sort_values("lower_bound")
        x = range(len(df))

        # 1. Average return by distance
        ax1 = axes[0, 0]
        colors = ["red" if v < -5 else "gold" if v < 5 else "green"
                 for v in df["lower_bound"]]
        ax1.bar(x, df["avg_daily_return"], color=colors, alpha=0.7)
        ax1.axhline(y=0, color="black", linestyle="--")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df["distance_range"], rotation=45, ha="right")
        ax1.set_title("Average Daily Return by SMA Distance", fontweight="bold")
        ax1.set_ylabel("Avg Return (%)")

        # 2. Return range (p10 to p90)
        ax2 = axes[0, 1]
        ax2.fill_between(x, df["p10_return"], df["p90_return"], alpha=0.3, color="blue", label="10th-90th pctl")
        ax2.fill_between(x, df["p25_return"], df["p75_return"], alpha=0.5, color="blue", label="25th-75th pctl")
        ax2.plot(x, df["median_daily_return"], color="black", marker="o", label="Median")
        ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(df["distance_range"], rotation=45, ha="right")
        ax2.set_title("Return Distribution Range by SMA Distance", fontweight="bold")
        ax2.set_ylabel("Return (%)")
        ax2.legend(fontsize=8)

        # 3. % Negative days
        ax3 = axes[1, 0]
        colors = ["red" if v > 50 else "gold" if v > 45 else "green"
                 for v in df["pct_negative"]]
        ax3.bar(x, df["pct_negative"], color=colors, alpha=0.7)
        ax3.axhline(y=50, color="black", linestyle="--", label="50%")
        ax3.set_xticks(x)
        ax3.set_xticklabels(df["distance_range"], rotation=45, ha="right")
        ax3.set_title("% of Negative Days by SMA Distance", fontweight="bold")
        ax3.set_ylabel("% Negative Days")
        ax3.set_ylim(0, 100)

        # 4. Asymmetry: avg positive vs avg negative return
        ax4 = axes[1, 1]
        width = 0.35
        ax4.bar([i - width/2 for i in x], df["avg_positive_return"], width,
               label="Avg Positive Day", color="green", alpha=0.7)
        ax4.bar([i + width/2 for i in x], df["avg_negative_return"], width,
               label="Avg Negative Day", color="red", alpha=0.7)
        ax4.axhline(y=0, color="black", linestyle="--")
        ax4.set_xticks(x)
        ax4.set_xticklabels(df["distance_range"], rotation=45, ha="right")
        ax4.set_title("Return Asymmetry by SMA Distance", fontweight="bold")
        ax4.set_ylabel("Return (%)")
        ax4.legend()

        plt.suptitle(f"{ticker} - Return Characteristics by Distance from 50 SMA",
                    fontsize=14, fontweight="bold")
        plt.tight_layout()

        filepath = self.output_dir / f"{ticker}_returns_by_distance.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")

        return fig


@dataclass
class RollingReturnSignal:
    """Represents a rolling return signal that preceded a selloff."""
    date: pd.Timestamp
    window: int
    rolling_return: float
    slope: float
    days_to_selloff: int
    selloff_magnitude: float


class RollingReturnsAnalyzer:
    """
    Analyzes rolling total returns to identify patterns that precede selloffs.

    Key hypothesis: Certain rolling return levels, slopes, or timeframes
    reveal themselves as leading indicators before large selloffs.
    """

    def __init__(self, windows: List[int] = None):
        """
        Initialize with rolling windows to analyze.

        Args:
            windows: List of rolling window periods (default: [5, 10, 20, 30, 60])
        """
        self.windows = windows or [5, 10, 20, 30, 60]

    def calculate_rolling_returns(
        self,
        df: pd.DataFrame,
        price_col: str = "price",
    ) -> pd.DataFrame:
        """
        Calculate rolling returns for multiple windows.

        Args:
            df: DataFrame with price data
            price_col: Column name for price

        Returns:
            DataFrame with rolling returns added
        """
        result = df.copy()

        if price_col not in result.columns:
            if "close" in result.columns:
                price_col = "close"

        for window in self.windows:
            # Rolling return (percent change over window)
            result[f"rolling_{window}d_return"] = (
                result[price_col].pct_change(window) * 100
            )

            # Slope of rolling return (is it accelerating/decelerating?)
            result[f"rolling_{window}d_slope"] = (
                result[f"rolling_{window}d_return"].diff(5)  # 5-day change in rolling return
            )

            # Rolling return momentum (second derivative - acceleration)
            result[f"rolling_{window}d_accel"] = (
                result[f"rolling_{window}d_slope"].diff(3)
            )

            # Normalized rolling return (z-score over trailing 60 days)
            rolling_mean = result[f"rolling_{window}d_return"].rolling(60).mean()
            rolling_std = result[f"rolling_{window}d_return"].rolling(60).std()
            result[f"rolling_{window}d_zscore"] = (
                (result[f"rolling_{window}d_return"] - rolling_mean) / rolling_std
            )

        return result

    def identify_selloff_events(
        self,
        df: pd.DataFrame,
        threshold: float = -15,
        price_col: str = "price",
    ) -> List[Dict]:
        """
        Identify significant selloff events.

        Args:
            df: DataFrame with price data
            threshold: Drawdown threshold to qualify as selloff (e.g., -15%)
            price_col: Price column name

        Returns:
            List of selloff events with details
        """
        if price_col not in df.columns:
            if "close" in df.columns:
                price_col = "close"

        result = df.copy()

        # Calculate drawdown
        result["rolling_max"] = result[price_col].expanding().max()
        result["drawdown"] = (
            (result[price_col] - result["rolling_max"]) / result["rolling_max"] * 100
        )

        # Find selloff starts (first day crossing threshold)
        result["in_selloff"] = result["drawdown"] <= threshold
        result["selloff_start"] = (
            result["in_selloff"] & ~result["in_selloff"].shift(1).fillna(False)
        )

        selloffs = []
        for date in result[result["selloff_start"]].index:
            # Find the trough
            idx = result.index.get_loc(date)
            future = result.iloc[idx:min(idx + 60, len(result))]
            trough_idx = future["drawdown"].idxmin()
            trough_dd = future.loc[trough_idx, "drawdown"]

            selloffs.append({
                "start_date": date,
                "trough_date": trough_idx,
                "max_drawdown": trough_dd,
                "duration_to_trough": (trough_idx - date).days,
            })

        return selloffs

    def analyze_returns_before_selloffs(
        self,
        df: pd.DataFrame,
        selloffs: List[Dict],
        lookback_days: List[int] = [5, 10, 20, 30],
    ) -> pd.DataFrame:
        """
        Analyze what rolling returns looked like before each selloff.

        Args:
            df: DataFrame with rolling returns calculated
            selloffs: List of selloff events
            lookback_days: Days before selloff to analyze

        Returns:
            DataFrame with pre-selloff conditions
        """
        results = []

        for selloff in selloffs:
            start_date = selloff["start_date"]

            if start_date not in df.index:
                continue

            start_idx = df.index.get_loc(start_date)

            for lookback in lookback_days:
                if start_idx - lookback < 0:
                    continue

                pre_selloff_date = df.index[start_idx - lookback]
                pre_selloff_row = df.loc[pre_selloff_date]

                result = {
                    "selloff_date": start_date,
                    "lookback_days": lookback,
                    "max_drawdown": selloff["max_drawdown"],
                }

                # Capture rolling returns at each window
                for window in self.windows:
                    ret_col = f"rolling_{window}d_return"
                    slope_col = f"rolling_{window}d_slope"
                    zscore_col = f"rolling_{window}d_zscore"

                    if ret_col in pre_selloff_row.index:
                        result[f"ret_{window}d"] = pre_selloff_row[ret_col]
                    if slope_col in pre_selloff_row.index:
                        result[f"slope_{window}d"] = pre_selloff_row[slope_col]
                    if zscore_col in pre_selloff_row.index:
                        result[f"zscore_{window}d"] = pre_selloff_row[zscore_col]

                results.append(result)

        return pd.DataFrame(results)

    def find_predictive_thresholds(
        self,
        df: pd.DataFrame,
        selloffs: List[Dict],
        forward_days: int = 20,
    ) -> pd.DataFrame:
        """
        Find rolling return thresholds that are predictive of selloffs.

        For each window, find the threshold level that, when crossed,
        most often preceded a selloff within forward_days.

        Args:
            df: DataFrame with rolling returns
            selloffs: List of selloff events
            forward_days: Days ahead to look for selloff

        Returns:
            DataFrame with predictive threshold analysis
        """
        results = []

        # Create selloff indicator
        selloff_dates = set(s["start_date"] for s in selloffs)

        # For each day, mark if selloff occurred within forward_days
        df_analysis = df.copy()
        df_analysis["selloff_ahead"] = False

        for date in selloff_dates:
            if date in df_analysis.index:
                idx = df_analysis.index.get_loc(date)
                start_idx = max(0, idx - forward_days)
                df_analysis.iloc[start_idx:idx, df_analysis.columns.get_loc("selloff_ahead")] = True

        # Test different thresholds for each window
        for window in self.windows:
            ret_col = f"rolling_{window}d_return"

            if ret_col not in df_analysis.columns:
                continue

            # Test thresholds from 0 to -30%
            for threshold in range(0, -31, -2):
                below_threshold = df_analysis[ret_col] <= threshold
                below_count = below_threshold.sum()

                if below_count < 5:
                    continue

                # Of days below threshold, what % had selloff ahead?
                below_with_selloff = (below_threshold & df_analysis["selloff_ahead"]).sum()
                accuracy = below_with_selloff / below_count if below_count > 0 else 0

                # False positive rate: below threshold but no selloff
                false_positive_rate = 1 - accuracy

                results.append({
                    "window": window,
                    "threshold": threshold,
                    "signals": below_count,
                    "true_positives": below_with_selloff,
                    "accuracy": accuracy * 100,
                    "false_positive_rate": false_positive_rate * 100,
                })

        results_df = pd.DataFrame(results)

        # Find optimal threshold for each window (highest accuracy with reasonable signals)
        optimal = []
        for window in self.windows:
            window_results = results_df[results_df["window"] == window]
            if window_results.empty:
                continue

            # Filter to thresholds with at least 5 signals
            valid = window_results[window_results["signals"] >= 5]
            if valid.empty:
                continue

            # Find best accuracy
            best = valid.loc[valid["accuracy"].idxmax()]
            optimal.append({
                "window": window,
                "optimal_threshold": best["threshold"],
                "accuracy": best["accuracy"],
                "signals": best["signals"],
            })

        return pd.DataFrame(optimal)

    def analyze_rolling_return_levels(
        self,
        df: pd.DataFrame,
        price_col: str = "price",
    ) -> pd.DataFrame:
        """
        Analyze what typically happens at different rolling return levels.

        Shows forward returns from different starting rolling return levels.

        Args:
            df: DataFrame with rolling returns
            price_col: Price column name

        Returns:
            DataFrame with analysis by rolling return level
        """
        if price_col not in df.columns:
            price_col = "close" if "close" in df.columns else "price"

        results = []

        # Calculate forward returns
        df_analysis = df.copy()
        for fwd in [5, 10, 20]:
            df_analysis[f"fwd_{fwd}d"] = df_analysis[price_col].pct_change(fwd).shift(-fwd) * 100

        # For each window, bin by rolling return level
        for window in self.windows:
            ret_col = f"rolling_{window}d_return"

            if ret_col not in df_analysis.columns:
                continue

            # Create bins
            bins = [-np.inf, -20, -15, -10, -5, 0, 5, 10, 15, 20, np.inf]
            labels = ["<-20", "-20 to -15", "-15 to -10", "-10 to -5", "-5 to 0",
                     "0 to 5", "5 to 10", "10 to 15", "15 to 20", ">20"]

            df_analysis["ret_bin"] = pd.cut(df_analysis[ret_col], bins=bins, labels=labels)

            for label in labels:
                subset = df_analysis[df_analysis["ret_bin"] == label]

                if len(subset) < 5:
                    continue

                results.append({
                    "window": window,
                    "rolling_return_range": label,
                    "num_observations": len(subset),
                    "avg_fwd_5d": subset["fwd_5d"].mean(),
                    "avg_fwd_10d": subset["fwd_10d"].mean(),
                    "avg_fwd_20d": subset["fwd_20d"].mean(),
                    "pct_negative_fwd_10d": (subset["fwd_10d"] < 0).mean() * 100,
                    "worst_fwd_20d": subset["fwd_20d"].min(),
                })

        return pd.DataFrame(results)

    def get_current_rolling_status(
        self,
        df: pd.DataFrame,
    ) -> Dict:
        """
        Get the current rolling return status across all windows.

        Args:
            df: DataFrame with rolling returns calculated

        Returns:
            Dict with current status
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]

        status = {
            "date": df.index[-1],
            "windows": {},
        }

        warning_signals = []

        for window in self.windows:
            ret_col = f"rolling_{window}d_return"
            slope_col = f"rolling_{window}d_slope"
            zscore_col = f"rolling_{window}d_zscore"

            window_status = {}

            if ret_col in latest.index and not pd.isna(latest[ret_col]):
                window_status["return"] = latest[ret_col]

                # Flag concerning levels
                if latest[ret_col] < -10:
                    warning_signals.append(f"{window}d return at {latest[ret_col]:.1f}%")

            if slope_col in latest.index and not pd.isna(latest[slope_col]):
                window_status["slope"] = latest[slope_col]

                # Flag accelerating decline
                if latest[slope_col] < -3:
                    warning_signals.append(f"{window}d slope accelerating: {latest[slope_col]:.1f}")

            if zscore_col in latest.index and not pd.isna(latest[zscore_col]):
                window_status["zscore"] = latest[zscore_col]

                # Flag extreme z-scores
                if latest[zscore_col] < -2:
                    warning_signals.append(f"{window}d z-score extreme: {latest[zscore_col]:.1f}")

            status["windows"][window] = window_status

        status["warning_signals"] = warning_signals
        status["alert_level"] = len(warning_signals)

        return status


class RollingReturnsVisualizer:
    """Visualization for rolling returns analysis."""

    def __init__(self, output_dir: str = "output/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_rolling_returns_with_selloffs(
        self,
        df: pd.DataFrame,
        selloffs: List[Dict],
        windows: List[int] = [10, 20, 30],
        ticker: str = "MSTY",
    ) -> plt.Figure:
        """
        Plot rolling returns over time with selloff events marked.
        """
        fig, axes = plt.subplots(len(windows) + 1, 1, figsize=(14, 4 * (len(windows) + 1)))

        price_col = "price" if "price" in df.columns else "close"

        # Price chart with selloff shading
        ax0 = axes[0]
        ax0.plot(df.index, df[price_col], color="black", linewidth=1)

        for selloff in selloffs:
            ax0.axvspan(selloff["start_date"], selloff["trough_date"],
                       alpha=0.3, color="red")

        ax0.set_title(f"{ticker} Price with Selloff Periods", fontweight="bold")
        ax0.set_ylabel("Price ($)")

        # Rolling returns for each window
        for i, window in enumerate(windows):
            ax = axes[i + 1]
            ret_col = f"rolling_{window}d_return"

            if ret_col not in df.columns:
                continue

            # Plot rolling return
            ax.plot(df.index, df[ret_col], color="blue", linewidth=1, alpha=0.8)
            ax.fill_between(df.index, df[ret_col], 0,
                           where=df[ret_col] >= 0, color="green", alpha=0.3)
            ax.fill_between(df.index, df[ret_col], 0,
                           where=df[ret_col] < 0, color="red", alpha=0.3)

            # Mark selloff starts
            for selloff in selloffs:
                if selloff["start_date"] in df.index:
                    ax.axvline(selloff["start_date"], color="red",
                              linestyle="--", alpha=0.5)

            # Threshold lines
            ax.axhline(y=0, color="black", linewidth=1)
            ax.axhline(y=-10, color="orange", linestyle="--", alpha=0.7)
            ax.axhline(y=-20, color="red", linestyle="--", alpha=0.7)

            ax.set_ylabel(f"{window}D Return (%)")
            ax.set_title(f"Rolling {window}-Day Return", fontweight="bold")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        filepath = self.output_dir / f"{ticker}_rolling_returns_selloffs.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")

        return fig

    def plot_pre_selloff_patterns(
        self,
        pre_selloff_df: pd.DataFrame,
        ticker: str = "MSTY",
    ) -> plt.Figure:
        """
        Visualize rolling return patterns before selloffs.
        """
        if pre_selloff_df.empty:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Group by lookback period
        lookbacks = pre_selloff_df["lookback_days"].unique()

        # 1. Average rolling returns before selloffs by lookback
        ax1 = axes[0, 0]
        for lb in sorted(lookbacks):
            lb_data = pre_selloff_df[pre_selloff_df["lookback_days"] == lb]
            windows = [10, 20, 30]
            values = [lb_data[f"ret_{w}d"].mean() for w in windows if f"ret_{w}d" in lb_data.columns]
            if values:
                ax1.plot(windows[:len(values)], values, marker="o", label=f"{lb}d before")

        ax1.axhline(y=0, color="black", linestyle="--")
        ax1.set_xlabel("Rolling Window (days)")
        ax1.set_ylabel("Avg Rolling Return (%)")
        ax1.set_title("Average Rolling Return Before Selloffs", fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Distribution of 20d rolling return before selloffs
        ax2 = axes[0, 1]
        if "ret_20d" in pre_selloff_df.columns:
            for lb in sorted(lookbacks):
                lb_data = pre_selloff_df[pre_selloff_df["lookback_days"] == lb]["ret_20d"].dropna()
                if len(lb_data) > 0:
                    ax2.hist(lb_data, bins=15, alpha=0.5, label=f"{lb}d before")

        ax2.axvline(x=0, color="black", linestyle="--")
        ax2.set_xlabel("20-Day Rolling Return (%)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of 20D Return Before Selloffs", fontweight="bold")
        ax2.legend()

        # 3. Slope patterns before selloffs
        ax3 = axes[1, 0]
        if "slope_20d" in pre_selloff_df.columns:
            for lb in sorted(lookbacks):
                lb_data = pre_selloff_df[pre_selloff_df["lookback_days"] == lb]
                if "slope_20d" in lb_data.columns:
                    slopes = lb_data["slope_20d"].dropna()
                    if len(slopes) > 0:
                        ax3.hist(slopes, bins=15, alpha=0.5, label=f"{lb}d before")

        ax3.axvline(x=0, color="black", linestyle="--")
        ax3.set_xlabel("20-Day Return Slope")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Slope (Momentum) Before Selloffs", fontweight="bold")
        ax3.legend()

        # 4. Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis("off")

        summary_data = []
        for lb in sorted(lookbacks):
            lb_data = pre_selloff_df[pre_selloff_df["lookback_days"] == lb]
            if "ret_20d" in lb_data.columns:
                ret_20 = lb_data["ret_20d"]
                summary_data.append([
                    f"{lb}d before",
                    f"{ret_20.mean():.1f}%",
                    f"{ret_20.median():.1f}%",
                    f"{(ret_20 < 0).mean() * 100:.0f}%",
                    f"{ret_20.min():.1f}%",
                ])

        if summary_data:
            table = ax4.table(
                cellText=summary_data,
                colLabels=["Lookback", "Avg 20D Ret", "Median", "% Negative", "Worst"],
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax4.set_title("20-Day Rolling Return Before Selloffs", fontweight="bold", pad=20)

        plt.suptitle(f"{ticker} - Rolling Return Patterns Before Selloffs",
                    fontsize=14, fontweight="bold")
        plt.tight_layout()

        filepath = self.output_dir / f"{ticker}_pre_selloff_patterns.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")

        return fig

    def plot_predictive_thresholds(
        self,
        thresholds_df: pd.DataFrame,
        ticker: str = "MSTY",
    ) -> plt.Figure:
        """
        Visualize the predictive power of different thresholds.
        """
        if thresholds_df.empty:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        windows = thresholds_df["window"].unique()
        x = np.arange(len(windows))
        width = 0.6

        thresholds = thresholds_df["optimal_threshold"].values
        accuracies = thresholds_df["accuracy"].values

        bars = ax.bar(x, thresholds, width, color="steelblue", alpha=0.7)

        # Add accuracy labels
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.annotate(f"{acc:.0f}% acc",
                       (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       ha="center", va="bottom" if bar.get_height() < 0 else "top",
                       fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{w}D" for w in windows])
        ax.set_xlabel("Rolling Window")
        ax.set_ylabel("Optimal Threshold (%)")
        ax.set_title(f"{ticker} - Optimal Rolling Return Thresholds for Selloff Prediction",
                    fontweight="bold")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        filepath = self.output_dir / f"{ticker}_predictive_thresholds.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")

        return fig
