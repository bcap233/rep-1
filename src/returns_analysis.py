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
