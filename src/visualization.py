"""
Visualization module for backtest charts and reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


class BacktestVisualizer:
    """Create visualizations for backtest analysis."""

    def __init__(self, output_dir: str = "output/charts"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_price_with_smas(
        self,
        df: pd.DataFrame,
        ticker: str,
        sma_periods: List[int] = [20, 50, 200],
        figsize: Tuple[int, int] = (14, 8),
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot price with multiple SMAs.

        Args:
            df: DataFrame with price and SMA data
            ticker: Ticker symbol for title
            sma_periods: SMA periods to plot
            figsize: Figure size
            save: Whether to save the figure

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

        # Price and SMAs
        ax1 = axes[0]
        ax1.plot(df.index, df["close"], label="Price", linewidth=1.5, color="black")

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sma_periods)))
        for i, period in enumerate(sma_periods):
            sma_col = f"sma_{period}"
            if sma_col in df.columns:
                ax1.plot(
                    df.index, df[sma_col],
                    label=f"SMA {period}",
                    linewidth=1,
                    alpha=0.8,
                    color=colors[i]
                )

        ax1.set_title(f"{ticker} Price with SMAs", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Price ($)")
        ax1.legend(loc="upper left")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # Drawdown
        ax2 = axes[1]
        if "drawdown" in df.columns:
            ax2.fill_between(
                df.index, df["drawdown"], 0,
                alpha=0.5, color="red", label="Drawdown"
            )
            ax2.axhline(y=-10, color="orange", linestyle="--", alpha=0.7, label="-10%")
            ax2.axhline(y=-20, color="red", linestyle="--", alpha=0.7, label="-20%")

        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.legend(loc="lower left")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f"{ticker}_price_sma.png"
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            print(f"Saved: {filepath}")

        return fig

    def plot_sma_distance_analysis(
        self,
        df: pd.DataFrame,
        ticker: str,
        sma_periods: List[int] = [50, 200],
        figsize: Tuple[int, int] = (14, 10),
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot distance from SMAs over time.

        Args:
            df: DataFrame with distance data
            ticker: Ticker symbol
            sma_periods: SMA periods to analyze
            figsize: Figure size
            save: Whether to save

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(len(sma_periods) + 1, 1, figsize=figsize)

        # Price
        axes[0].plot(df.index, df["close"], color="black", linewidth=1)
        axes[0].set_title(f"{ticker} - SMA Distance Analysis", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Price ($)")

        # Distance plots
        for i, period in enumerate(sma_periods):
            ax = axes[i + 1]
            dist_col = f"sma_{period}_dist_pct"

            if dist_col in df.columns:
                # Color based on above/below
                colors = ["green" if x > 0 else "red" for x in df[dist_col]]
                ax.bar(df.index, df[dist_col], color=colors, alpha=0.6, width=1)
                ax.axhline(y=0, color="black", linewidth=0.5)
                ax.axhline(y=5, color="green", linestyle="--", alpha=0.5)
                ax.axhline(y=-5, color="red", linestyle="--", alpha=0.5)

                ax.set_ylabel(f"Dist SMA{period} (%)")

        axes[-1].set_xlabel("Date")

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f"{ticker}_sma_distance.png"
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            print(f"Saved: {filepath}")

        return fig

    def plot_etf_vs_underlying_comparison(
        self,
        etf_df: pd.DataFrame,
        underlying_df: pd.DataFrame,
        etf_ticker: str,
        underlying_ticker: str,
        figsize: Tuple[int, int] = (14, 12),
        save: bool = True,
    ) -> plt.Figure:
        """
        Compare yield max ETF vs its underlying index.

        Args:
            etf_df: ETF price data
            underlying_df: Underlying price data
            etf_ticker: ETF ticker
            underlying_ticker: Underlying ticker
            figsize: Figure size
            save: Whether to save

        Returns:
            matplotlib Figure
        """
        # Align and normalize
        common_dates = etf_df.index.intersection(underlying_df.index)
        etf = etf_df.loc[common_dates].copy()
        underlying = underlying_df.loc[common_dates].copy()

        # Normalize to 100
        etf_norm = etf["close"] / etf["close"].iloc[0] * 100
        underlying_norm = underlying["close"] / underlying["close"].iloc[0] * 100

        fig, axes = plt.subplots(3, 1, figsize=figsize)

        # Normalized price comparison
        ax1 = axes[0]
        ax1.plot(etf_norm.index, etf_norm, label=etf_ticker, linewidth=1.5)
        ax1.plot(underlying_norm.index, underlying_norm, label=underlying_ticker, linewidth=1.5)
        ax1.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
        ax1.set_title(
            f"{etf_ticker} vs {underlying_ticker} - Normalized Performance",
            fontsize=14, fontweight="bold"
        )
        ax1.set_ylabel("Value (Base=100)")
        ax1.legend()

        # Spread (ETF - Underlying performance)
        ax2 = axes[1]
        spread = etf_norm - underlying_norm
        colors = ["green" if x > 0 else "red" for x in spread]
        ax2.fill_between(spread.index, spread, 0, alpha=0.5,
                         color="green", where=spread >= 0, label="ETF outperforms")
        ax2.fill_between(spread.index, spread, 0, alpha=0.5,
                         color="red", where=spread < 0, label="Underlying outperforms")
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.set_ylabel("Performance Spread")
        ax2.set_title("Relative Performance (ETF - Underlying)")
        ax2.legend()

        # Rolling correlation
        ax3 = axes[2]
        etf_returns = etf["close"].pct_change()
        underlying_returns = underlying["close"].pct_change()
        rolling_corr = etf_returns.rolling(60).corr(underlying_returns)
        ax3.plot(rolling_corr.index, rolling_corr, color="purple", linewidth=1)
        ax3.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax3.set_ylabel("60-Day Rolling Correlation")
        ax3.set_xlabel("Date")
        ax3.set_ylim(0, 1.1)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f"{etf_ticker}_vs_{underlying_ticker}.png"
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            print(f"Saved: {filepath}")

        return fig

    def plot_decline_analysis(
        self,
        df: pd.DataFrame,
        decline_periods: List,
        ticker: str,
        figsize: Tuple[int, int] = (14, 10),
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot price with decline periods highlighted.

        Args:
            df: DataFrame with price data
            decline_periods: List of DeclinePeriod objects
            ticker: Ticker symbol
            figsize: Figure size
            save: Whether to save

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])

        # Price with decline shading
        ax1 = axes[0]
        ax1.plot(df.index, df["close"], color="black", linewidth=1)

        # Shade decline periods
        for decline in decline_periods:
            ax1.axvspan(
                decline.start_date, decline.end_date,
                alpha=0.3, color="red",
                label=f"Decline: {decline.max_drawdown:.1f}%"
            )

        ax1.set_title(f"{ticker} - Decline Periods Analysis", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Price ($)")

        # Drawdown
        ax2 = axes[1]
        if "drawdown" in df.columns:
            ax2.fill_between(df.index, df["drawdown"], 0, alpha=0.5, color="red")

        for decline in decline_periods:
            ax2.axvline(decline.start_date, color="red", linestyle="--", alpha=0.5)
            ax2.axvline(decline.end_date, color="green", linestyle="--", alpha=0.5)

        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f"{ticker}_decline_analysis.png"
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            print(f"Saved: {filepath}")

        return fig

    def plot_sma_position_returns(
        self,
        analysis_results: Dict[str, Dict],
        ticker: str,
        sma_period: int,
        figsize: Tuple[int, int] = (12, 8),
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot forward returns by SMA position.

        Args:
            analysis_results: Results from SMARelationshipAnalyzer
            ticker: Ticker symbol
            sma_period: SMA period analyzed
            figsize: Figure size
            save: Whether to save

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        periods = [5, 10, 20, 60]
        x = np.arange(len(periods))
        width = 0.35

        # Mean returns
        ax1 = axes[0]
        above_means = [analysis_results["above"].get(f"fwd_{p}d_mean", 0) for p in periods]
        below_means = [analysis_results["below"].get(f"fwd_{p}d_mean", 0) for p in periods]

        bars1 = ax1.bar(x - width / 2, above_means, width, label="Above SMA", color="green", alpha=0.7)
        bars2 = ax1.bar(x + width / 2, below_means, width, label="Below SMA", color="red", alpha=0.7)

        ax1.axhline(y=0, color="black", linewidth=0.5)
        ax1.set_ylabel("Mean Forward Return (%)")
        ax1.set_title(f"Mean Forward Returns by SMA{sma_period} Position")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{p}D" for p in periods])
        ax1.legend()

        # Positive probability
        ax2 = axes[1]
        above_pos = [analysis_results["above"].get(f"fwd_{p}d_positive_pct", 0) for p in periods]
        below_pos = [analysis_results["below"].get(f"fwd_{p}d_positive_pct", 0) for p in periods]

        bars3 = ax2.bar(x - width / 2, above_pos, width, label="Above SMA", color="green", alpha=0.7)
        bars4 = ax2.bar(x + width / 2, below_pos, width, label="Below SMA", color="red", alpha=0.7)

        ax2.axhline(y=50, color="black", linewidth=0.5, linestyle="--")
        ax2.set_ylabel("Probability of Positive Return (%)")
        ax2.set_title(f"Win Rate by SMA{sma_period} Position")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{p}D" for p in periods])
        ax2.set_ylim(0, 100)
        ax2.legend()

        plt.suptitle(f"{ticker} - SMA{sma_period} Position Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            filepath = self.output_dir / f"{ticker}_sma{sma_period}_returns.png"
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            print(f"Saved: {filepath}")

        return fig

    def plot_market_condition_comparison(
        self,
        comparison_df: pd.DataFrame,
        figsize: Tuple[int, int] = (14, 10),
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot ETF vs underlying performance across market conditions.

        Args:
            comparison_df: DataFrame from compare_etf_vs_underlying
            figsize: Figure size
            save: Whether to save

        Returns:
            matplotlib Figure
        """
        if comparison_df.empty:
            return None

        etf_ticker = comparison_df["etf_ticker"].iloc[0]
        underlying_ticker = comparison_df["underlying_ticker"].iloc[0]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Filter for relevant conditions
        conditions = [
            "all_periods", "bull_market", "bear_market",
            "above_50sma", "below_50sma", "above_200sma", "below_200sma"
        ]
        plot_df = comparison_df[comparison_df["condition"].isin(conditions)].copy()

        if plot_df.empty:
            return None

        # Total returns comparison
        ax1 = axes[0, 0]
        x = np.arange(len(plot_df))
        width = 0.35
        ax1.bar(x - width / 2, plot_df["etf_total_return"] * 100, width,
                label=etf_ticker, color="blue", alpha=0.7)
        ax1.bar(x + width / 2, plot_df["underlying_total_return"] * 100, width,
                label=underlying_ticker, color="orange", alpha=0.7)
        ax1.axhline(y=0, color="black", linewidth=0.5)
        ax1.set_ylabel("Total Return (%)")
        ax1.set_title("Total Returns by Market Condition")
        ax1.set_xticks(x)
        ax1.set_xticklabels(plot_df["condition"], rotation=45, ha="right")
        ax1.legend()

        # Capture ratio
        ax2 = axes[0, 1]
        colors = ["green" if x > 0 else "red" for x in plot_df["capture_ratio"]]
        ax2.bar(x, plot_df["capture_ratio"], color=colors, alpha=0.7)
        ax2.axhline(y=1, color="black", linestyle="--", linewidth=0.5)
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.set_ylabel("Capture Ratio")
        ax2.set_title("Capture Ratio (ETF/Underlying)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(plot_df["condition"], rotation=45, ha="right")

        # Volatility comparison
        ax3 = axes[1, 0]
        ax3.bar(x - width / 2, plot_df["etf_volatility"], width,
                label=etf_ticker, color="blue", alpha=0.7)
        ax3.bar(x + width / 2, plot_df["underlying_volatility"], width,
                label=underlying_ticker, color="orange", alpha=0.7)
        ax3.set_ylabel("Daily Volatility (%)")
        ax3.set_title("Volatility by Market Condition")
        ax3.set_xticks(x)
        ax3.set_xticklabels(plot_df["condition"], rotation=45, ha="right")
        ax3.legend()

        # Correlation
        ax4 = axes[1, 1]
        ax4.bar(x, plot_df["correlation"], color="purple", alpha=0.7)
        ax4.axhline(y=1, color="black", linestyle="--", linewidth=0.5)
        ax4.set_ylabel("Correlation")
        ax4.set_title("Return Correlation by Condition")
        ax4.set_xticks(x)
        ax4.set_xticklabels(plot_df["condition"], rotation=45, ha="right")
        ax4.set_ylim(0, 1.1)

        plt.suptitle(
            f"{etf_ticker} vs {underlying_ticker} - Market Condition Analysis",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        if save:
            filepath = self.output_dir / f"{etf_ticker}_market_conditions.png"
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            print(f"Saved: {filepath}")

        return fig

    def create_summary_dashboard(
        self,
        data: Dict[str, pd.DataFrame],
        analysis_results: Dict,
        figsize: Tuple[int, int] = (18, 14),
        save: bool = True,
    ) -> plt.Figure:
        """
        Create a comprehensive summary dashboard.

        Args:
            data: Dict of ticker -> DataFrame
            analysis_results: Analysis results dictionary
            figsize: Figure size
            save: Whether to save

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)

        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Performance comparison (top row, spanning 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        for ticker, df in data.items():
            if "close" in df.columns:
                normalized = df["close"] / df["close"].iloc[0] * 100
                ax1.plot(normalized.index, normalized, label=ticker, linewidth=1.5)
        ax1.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
        ax1.set_title("Normalized Performance Comparison", fontweight="bold")
        ax1.set_ylabel("Value (Base=100)")
        ax1.legend(loc="upper left")

        # Drawdown comparison (middle left)
        ax2 = fig.add_subplot(gs[1, 0])
        for ticker, df in data.items():
            if "drawdown" in df.columns:
                ax2.plot(df.index, df["drawdown"], label=ticker, alpha=0.7)
        ax2.set_title("Drawdowns", fontweight="bold")
        ax2.set_ylabel("Drawdown (%)")
        ax2.legend(loc="lower left", fontsize=8)

        # Max drawdown table (middle center)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis("off")
        drawdown_data = []
        for ticker, df in data.items():
            if "drawdown" in df.columns:
                max_dd = df["drawdown"].min()
                drawdown_data.append([ticker, f"{max_dd:.1f}%"])

        if drawdown_data:
            table = ax3.table(
                cellText=drawdown_data,
                colLabels=["Ticker", "Max Drawdown"],
                loc="center",
                cellLoc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax3.set_title("Maximum Drawdowns", fontweight="bold")

        # Volatility comparison (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        vol_data = []
        for ticker, df in data.items():
            if "volatility_20d_ann" in df.columns:
                avg_vol = df["volatility_20d_ann"].mean()
                vol_data.append((ticker, avg_vol))

        if vol_data:
            tickers, vols = zip(*vol_data)
            ax4.barh(tickers, vols, color="steelblue", alpha=0.7)
            ax4.set_title("Avg Annual Volatility", fontweight="bold")
            ax4.set_xlabel("Volatility (%)")

        # Stats summary (bottom row)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("off")

        stats_data = []
        for ticker, df in data.items():
            if "close" in df.columns:
                total_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
                max_dd = df["drawdown"].min() if "drawdown" in df.columns else np.nan
                vol = df["volatility_20d_ann"].mean() if "volatility_20d_ann" in df.columns else np.nan
                sharpe = (total_return / 100) / (vol / 100) if vol and vol > 0 else np.nan

                stats_data.append([
                    ticker,
                    f"{total_return:.1f}%",
                    f"{max_dd:.1f}%",
                    f"{vol:.1f}%",
                    f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A"
                ])

        if stats_data:
            table2 = ax5.table(
                cellText=stats_data,
                colLabels=["Ticker", "Total Return", "Max Drawdown", "Avg Volatility", "Sharpe-like"],
                loc="center",
                cellLoc="center"
            )
            table2.auto_set_font_size(False)
            table2.set_fontsize(10)
            table2.scale(1.2, 1.5)
            ax5.set_title("Performance Summary", fontweight="bold", pad=20)

        plt.suptitle("Yield Max / Covered Call ETF Analysis Dashboard", fontsize=16, fontweight="bold")

        if save:
            filepath = self.output_dir / "summary_dashboard.png"
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            print(f"Saved: {filepath}")

        return fig
