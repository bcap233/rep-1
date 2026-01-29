"""
Counterparty analysis module.

Compares MSTY (long/bull covered call) vs WNTR (short/bear inverse) rolling total returns
to identify divergence patterns that signal reflexive collapse or expansion.

Key hypothesis: As MSTY's rolling total returns go negative on increasingly more timeframes
(5d -> 10d -> 20d), WNTR's rolling total returns should go positive on those same timeframes.
This divergence cascade is a signal of reflexive collapse in MSTY.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class TotalReturnCalculator:
    """
    Calculates total return index from price + dividend data.

    Total return accounts for:
    - Price appreciation/depreciation
    - Dividend income (assumed reinvested)
    - Stock splits (adjusts historical prices)
    """

    def calculate_total_return_index(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        dividend_col: str = "dividend",
        split_col: str = "split",
    ) -> pd.Series:
        """
        Create a total return index that accounts for dividends.

        Starts at 1.0 on the first day. Each day:
        - Adjusts for splits
        - Adds dividend yield (dividend / price on ex-date)
        - Compounds price returns

        Args:
            df: DataFrame with price, dividend, and split data (oldest first)
            price_col: Column name for closing price
            dividend_col: Column name for dividend payments
            split_col: Column name for split ratios

        Returns:
            Series with total return index values
        """
        result = df.copy()
        result = result.sort_index()  # Ensure oldest first

        # First, create split-adjusted prices working backwards
        # The split column contains the split ratio on the split date
        if split_col in result.columns:
            cumulative_split = result[split_col].replace(0, 1).cumprod()
            # Normalize so the last value = 1 (current prices are "real")
            final_split = cumulative_split.iloc[-1]
            split_adjustment = cumulative_split / final_split
            adjusted_price = result[price_col] / split_adjustment
        else:
            adjusted_price = result[price_col].copy()

        # Calculate daily price returns
        daily_price_return = adjusted_price.pct_change()

        # Calculate dividend yield on each day
        if dividend_col in result.columns:
            # Dividend yield = dividend / price on ex-date
            # For split-adjusted: use adjusted price
            dividend_yield = result[dividend_col] / adjusted_price
            dividend_yield = dividend_yield.fillna(0)
        else:
            dividend_yield = pd.Series(0, index=result.index)

        # Total daily return = price return + dividend yield
        total_daily_return = daily_price_return + dividend_yield
        total_daily_return.iloc[0] = 0  # First day has no return

        # Compound into total return index
        total_return_index = (1 + total_daily_return).cumprod()

        return total_return_index

    def calculate_rolling_total_returns(
        self,
        total_return_index: pd.Series,
        windows: List[int] = None,
    ) -> pd.DataFrame:
        """
        Calculate rolling total returns over multiple windows.

        Args:
            total_return_index: Total return index series
            windows: List of rolling window periods (default: [5, 10, 20])

        Returns:
            DataFrame with rolling total return columns
        """
        windows = windows or [5, 10, 20]
        result = pd.DataFrame(index=total_return_index.index)

        for window in windows:
            result[f"total_return_{window}d"] = (
                total_return_index / total_return_index.shift(window) - 1
            ) * 100

        return result


class CounterpartyAnalyzer:
    """
    Analyzes the relationship between MSTY and WNTR rolling total returns.

    Core analysis:
    - Side-by-side rolling total returns across 5/10/20 day windows
    - Divergence scoring: how strongly are they moving in opposite directions
    - Timeframe cascade: as MSTY goes negative on more timeframes, does WNTR go positive
    """

    def __init__(self, windows: List[int] = None):
        self.windows = windows or [5, 10, 20]
        self.calc = TotalReturnCalculator()

    def prepare_comparison_data(
        self,
        msty_df: pd.DataFrame,
        wntr_df: pd.DataFrame,
        msty_price_col: str = "close",
        wntr_price_col: str = "close",
        msty_div_col: str = "dividend",
        wntr_div_col: str = "dividend",
        msty_split_col: str = "split",
        wntr_split_col: str = "split",
    ) -> pd.DataFrame:
        """
        Prepare side-by-side comparison dataframe.

        Args:
            msty_df: MSTY data with price, dividend, split columns
            wntr_df: WNTR data with price, dividend, split columns

        Returns:
            DataFrame with both instruments' total returns aligned by date
        """
        # Calculate total return indices
        msty_tri = self.calc.calculate_total_return_index(
            msty_df, msty_price_col, msty_div_col, msty_split_col
        )
        wntr_tri = self.calc.calculate_total_return_index(
            wntr_df, wntr_price_col, wntr_div_col, wntr_split_col
        )

        # Calculate rolling total returns
        msty_rolling = self.calc.calculate_rolling_total_returns(msty_tri, self.windows)
        wntr_rolling = self.calc.calculate_rolling_total_returns(wntr_tri, self.windows)

        # Build comparison dataframe on intersection of dates
        common_dates = msty_df.sort_index().index.intersection(wntr_df.sort_index().index)
        comparison = pd.DataFrame(index=common_dates)

        # Prices
        comparison["msty_price"] = msty_df.sort_index()[msty_price_col]
        comparison["wntr_price"] = wntr_df.sort_index()[wntr_price_col]

        # Total return indices
        comparison["msty_tri"] = msty_tri
        comparison["wntr_tri"] = wntr_tri

        # Rolling total returns for each window
        for window in self.windows:
            col = f"total_return_{window}d"
            comparison[f"msty_{col}"] = msty_rolling[col]
            comparison[f"wntr_{col}"] = wntr_rolling[col]

            # Spread (WNTR - MSTY): positive = diverging in WNTR's favor
            comparison[f"spread_{window}d"] = (
                wntr_rolling[col] - msty_rolling[col]
            )

        # Sign analysis for each window
        for window in self.windows:
            col = f"total_return_{window}d"
            comparison[f"msty_neg_{window}d"] = comparison[f"msty_{col}"] < 0
            comparison[f"wntr_pos_{window}d"] = comparison[f"wntr_{col}"] > 0
            comparison[f"diverging_{window}d"] = (
                comparison[f"msty_neg_{window}d"] & comparison[f"wntr_pos_{window}d"]
            )

        # Count how many timeframes show MSTY negative
        comparison["msty_neg_count"] = sum(
            comparison[f"msty_neg_{w}d"].astype(int) for w in self.windows
        )

        # Count how many timeframes show WNTR positive
        comparison["wntr_pos_count"] = sum(
            comparison[f"wntr_pos_{w}d"].astype(int) for w in self.windows
        )

        # Count how many timeframes are diverging (MSTY neg AND WNTR pos)
        comparison["divergence_count"] = sum(
            comparison[f"diverging_{w}d"].astype(int) for w in self.windows
        )

        # Divergence score: weighted by timeframe (longer = more significant)
        weight_sum = sum(self.windows)
        comparison["divergence_score"] = sum(
            comparison[f"diverging_{w}d"].astype(int) * (w / weight_sum)
            for w in self.windows
        )

        return comparison

    def analyze_divergence_patterns(
        self,
        comparison: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Analyze what happens at different divergence levels.

        Groups by how many timeframes show divergence (0, 1, 2, 3)
        and shows what typically follows.

        Returns:
            DataFrame with divergence pattern statistics
        """
        results = []

        for count in range(len(self.windows) + 1):
            mask = comparison["divergence_count"] == count
            subset = comparison[mask].dropna(
                subset=[f"msty_total_return_{self.windows[0]}d"]
            )

            if len(subset) < 3:
                continue

            row = {
                "divergence_level": count,
                "description": self._divergence_description(count),
                "num_days": len(subset),
                "pct_of_total": len(subset) / len(comparison.dropna(
                    subset=[f"msty_total_return_{self.windows[0]}d"]
                )) * 100,
            }

            # Average returns at each level
            for window in self.windows:
                msty_col = f"msty_total_return_{window}d"
                wntr_col = f"wntr_total_return_{window}d"
                spread_col = f"spread_{window}d"

                if msty_col in subset.columns:
                    row[f"avg_msty_{window}d"] = subset[msty_col].mean()
                    row[f"avg_wntr_{window}d"] = subset[wntr_col].mean()
                    row[f"avg_spread_{window}d"] = subset[spread_col].mean()

            results.append(row)

        return pd.DataFrame(results)

    def _divergence_description(self, count: int) -> str:
        if count == 0:
            return "No divergence"
        elif count == 1:
            return "Early divergence (1 timeframe)"
        elif count == 2:
            return "Building divergence (2 timeframes)"
        else:
            return "Full divergence (all timeframes)"

    def analyze_timeframe_cascade(
        self,
        comparison: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Track how signals cascade across timeframes.

        Shows progression:
        1. MSTY 5d goes negative -> what's WNTR doing?
        2. MSTY 5d+10d negative -> what's WNTR doing on both?
        3. MSTY 5d+10d+20d negative -> full cascade

        Returns:
            DataFrame with cascade analysis
        """
        results = []

        # Sort windows shortest to longest
        sorted_windows = sorted(self.windows)

        for depth in range(1, len(sorted_windows) + 1):
            active_windows = sorted_windows[:depth]

            # Mask: MSTY negative on all active windows
            msty_mask = pd.Series(True, index=comparison.index)
            for w in active_windows:
                msty_mask = msty_mask & comparison[f"msty_neg_{w}d"]

            subset = comparison[msty_mask].dropna(
                subset=[f"msty_total_return_{sorted_windows[0]}d"]
            )

            if len(subset) < 2:
                continue

            cascade_label = " + ".join(f"{w}d" for w in active_windows)

            row = {
                "cascade_depth": depth,
                "msty_negative_on": cascade_label,
                "num_days": len(subset),
            }

            # For each window, show WNTR's average return and % positive
            for w in self.windows:
                wntr_col = f"wntr_total_return_{w}d"
                msty_col = f"msty_total_return_{w}d"

                if wntr_col in subset.columns:
                    row[f"wntr_avg_{w}d"] = subset[wntr_col].mean()
                    row[f"wntr_pct_pos_{w}d"] = (
                        (subset[wntr_col] > 0).mean() * 100
                    )
                    row[f"msty_avg_{w}d"] = subset[msty_col].mean()
                    row[f"spread_avg_{w}d"] = subset[f"spread_{w}d"].mean()

            results.append(row)

        return pd.DataFrame(results)

    def analyze_reflexive_divergence(
        self,
        comparison: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Analyze the reflexive nature: as MSTY's returns worsen,
        does WNTR's improvement accelerate?

        Bins MSTY's 20d return and shows corresponding WNTR behavior.

        Returns:
            DataFrame with reflexive divergence analysis
        """
        ref_window = max(self.windows)
        msty_col = f"msty_total_return_{ref_window}d"
        results = []

        bins = [(-100, -30), (-30, -20), (-20, -10), (-10, -5), (-5, 0),
                (0, 5), (5, 10), (10, 20), (20, 100)]

        for lower, upper in bins:
            mask = (comparison[msty_col] >= lower) & (comparison[msty_col] < upper)
            subset = comparison[mask].dropna(subset=[msty_col])

            if len(subset) < 3:
                continue

            row = {
                "msty_return_range": f"{lower}% to {upper}%",
                "num_days": len(subset),
            }

            for w in self.windows:
                wntr_col = f"wntr_total_return_{w}d"
                m_col = f"msty_total_return_{w}d"

                if wntr_col in subset.columns:
                    row[f"wntr_avg_{w}d"] = subset[wntr_col].mean()
                    row[f"msty_avg_{w}d"] = subset[m_col].mean()
                    row[f"spread_{w}d"] = subset[f"spread_{w}d"].mean()
                    row[f"wntr_pct_pos_{w}d"] = (
                        (subset[wntr_col] > 0).mean() * 100
                    )

            results.append(row)

        return pd.DataFrame(results)

    def get_current_status(
        self,
        comparison: pd.DataFrame,
    ) -> Dict:
        """
        Get the current divergence status.

        Returns:
            Dict with current status across all windows
        """
        latest = comparison.dropna(
            subset=[f"msty_total_return_{self.windows[0]}d"]
        )

        if latest.empty:
            return {}

        latest_row = latest.iloc[-1]
        date = latest.index[-1]

        status = {
            "date": date,
            "msty_price": latest_row.get("msty_price", None),
            "wntr_price": latest_row.get("wntr_price", None),
            "windows": {},
            "divergence_count": int(latest_row["divergence_count"]),
            "divergence_score": latest_row["divergence_score"],
        }

        for w in self.windows:
            msty_ret = latest_row.get(f"msty_total_return_{w}d", None)
            wntr_ret = latest_row.get(f"wntr_total_return_{w}d", None)
            spread = latest_row.get(f"spread_{w}d", None)

            status["windows"][w] = {
                "msty_return": msty_ret,
                "wntr_return": wntr_ret,
                "spread": spread,
                "msty_negative": bool(latest_row.get(f"msty_neg_{w}d", False)),
                "wntr_positive": bool(latest_row.get(f"wntr_pos_{w}d", False)),
                "diverging": bool(latest_row.get(f"diverging_{w}d", False)),
            }

        return status


class CounterpartyVisualizer:
    """Visualization for counterparty analysis."""

    def __init__(self, output_dir: str = "output/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_rolling_returns_comparison(
        self,
        comparison: pd.DataFrame,
        windows: List[int] = None,
    ) -> plt.Figure:
        """
        Side-by-side rolling total returns for MSTY vs WNTR.
        Each window gets its own panel showing both instruments.
        """
        windows = windows or [5, 10, 20]
        num_panels = len(windows) + 1  # +1 for price chart

        fig, axes = plt.subplots(num_panels, 1, figsize=(16, 4 * num_panels))

        # Panel 0: Price comparison (normalized)
        ax0 = axes[0]
        if "msty_tri" in comparison.columns and "wntr_tri" in comparison.columns:
            msty_tri = comparison["msty_tri"].dropna()
            wntr_tri = comparison["wntr_tri"].dropna()

            ax0.plot(msty_tri.index, (msty_tri / msty_tri.iloc[0] - 1) * 100,
                     color="red", linewidth=1.5, label="MSTY Total Return")
            ax0.plot(wntr_tri.index, (wntr_tri / wntr_tri.iloc[0] - 1) * 100,
                     color="blue", linewidth=1.5, label="WNTR Total Return")

            ax0.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
            ax0.fill_between(msty_tri.index,
                             (msty_tri / msty_tri.iloc[0] - 1) * 100, 0,
                             alpha=0.15, color="red")
            ax0.fill_between(wntr_tri.index,
                             (wntr_tri / wntr_tri.iloc[0] - 1) * 100, 0,
                             alpha=0.15, color="blue")

        ax0.set_title("MSTY vs WNTR — Cumulative Total Return (%)", fontweight="bold", fontsize=12)
        ax0.set_ylabel("Total Return (%)")
        ax0.legend(loc="best")
        ax0.grid(True, alpha=0.3)

        # Panels 1-N: Rolling returns for each window
        for i, window in enumerate(windows):
            ax = axes[i + 1]
            msty_col = f"msty_total_return_{window}d"
            wntr_col = f"wntr_total_return_{window}d"

            if msty_col in comparison.columns:
                msty_data = comparison[msty_col].dropna()
                wntr_data = comparison[wntr_col].dropna()

                ax.plot(msty_data.index, msty_data,
                        color="red", linewidth=1.2, alpha=0.9, label="MSTY")
                ax.plot(wntr_data.index, wntr_data,
                        color="blue", linewidth=1.2, alpha=0.9, label="WNTR")

                ax.axhline(y=0, color="black", linewidth=1, linestyle="-")

                # Shade divergence zones (MSTY neg + WNTR pos)
                div_col = f"diverging_{window}d"
                if div_col in comparison.columns:
                    div_mask = comparison[div_col].fillna(False)
                    if div_mask.any():
                        ymin = min(msty_data.min(), wntr_data.min()) * 1.1
                        ymax = max(msty_data.max(), wntr_data.max()) * 1.1
                        ax.fill_between(
                            comparison.index, ymin, ymax,
                            where=div_mask, alpha=0.15, color="purple",
                            label="Divergence zone"
                        )

            ax.set_title(
                f"Rolling {window}-Day Total Return — MSTY vs WNTR",
                fontweight="bold", fontsize=11,
            )
            ax.set_ylabel(f"{window}D Return (%)")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Date")
        plt.tight_layout()

        filepath = self.output_dir / "msty_vs_wntr_rolling_returns.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")

        return fig

    def plot_divergence_dashboard(
        self,
        comparison: pd.DataFrame,
        windows: List[int] = None,
    ) -> plt.Figure:
        """
        Dashboard showing divergence intensity over time.
        """
        windows = windows or [5, 10, 20]

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        # Panel 1: Divergence count over time
        ax1 = axes[0]
        div_count = comparison["divergence_count"].dropna()
        colors_map = {0: "green", 1: "gold", 2: "orange", 3: "red"}

        ax1.bar(div_count.index, div_count,
                color=[colors_map.get(int(v), "gray") for v in div_count],
                width=1.5, alpha=0.8)
        ax1.set_title("Divergence Intensity — # of Timeframes with MSTY Negative + WNTR Positive",
                       fontweight="bold", fontsize=11)
        ax1.set_ylabel("# Timeframes Diverging")
        ax1.set_yticks(range(len(windows) + 1))
        ax1.grid(True, alpha=0.3, axis="y")

        # Panel 2: Spread (WNTR return - MSTY return) for each window
        ax2 = axes[1]
        for w in windows:
            spread_col = f"spread_{w}d"
            if spread_col in comparison.columns:
                spread = comparison[spread_col].dropna()
                ax2.plot(spread.index, spread, linewidth=1.2,
                         label=f"{w}D Spread", alpha=0.8)

        ax2.axhline(y=0, color="black", linewidth=1, linestyle="-")
        ax2.fill_between(comparison.index,
                         comparison.get(f"spread_{windows[-1]}d",
                                        pd.Series(0, index=comparison.index)),
                         0, alpha=0.1, color="purple")
        ax2.set_title("Return Spread (WNTR − MSTY) — Positive = WNTR Outperforming",
                       fontweight="bold", fontsize=11)
        ax2.set_ylabel("Spread (%)")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)

        # Panel 3: MSTY negative count vs WNTR positive count
        ax3 = axes[2]
        msty_neg = comparison["msty_neg_count"].dropna()
        wntr_pos = comparison["wntr_pos_count"].dropna()

        ax3.plot(msty_neg.index, msty_neg, color="red", linewidth=1.5,
                 label="MSTY # negative timeframes", alpha=0.8)
        ax3.plot(wntr_pos.index, wntr_pos, color="blue", linewidth=1.5,
                 label="WNTR # positive timeframes", alpha=0.8)
        ax3.fill_between(msty_neg.index, msty_neg, alpha=0.15, color="red")
        ax3.fill_between(wntr_pos.index, wntr_pos, alpha=0.15, color="blue")

        ax3.set_title(
            "Timeframe Cascade — MSTY Negative Count vs WNTR Positive Count",
            fontweight="bold", fontsize=11,
        )
        ax3.set_ylabel("# of Timeframes")
        ax3.set_yticks(range(len(windows) + 1))
        ax3.legend(loc="best")
        ax3.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Date")
        plt.tight_layout()

        filepath = self.output_dir / "msty_vs_wntr_divergence.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")

        return fig

    def plot_reflexive_analysis(
        self,
        reflexive_df: pd.DataFrame,
        windows: List[int] = None,
    ) -> plt.Figure:
        """
        Visualize the reflexive relationship:
        As MSTY's returns worsen, how does WNTR respond?
        """
        windows = windows or [5, 10, 20]

        if reflexive_df.empty:
            return None

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, w in enumerate(windows):
            ax = axes[i]
            wntr_col = f"wntr_avg_{w}d"
            msty_col = f"msty_avg_{w}d"

            if wntr_col not in reflexive_df.columns:
                continue

            x = range(len(reflexive_df))
            x_labels = reflexive_df["msty_return_range"]

            # MSTY bars (negative direction)
            ax.bar([xi - 0.2 for xi in x], reflexive_df[msty_col],
                   width=0.35, color="red", alpha=0.7, label="MSTY avg return")
            # WNTR bars (positive direction)
            ax.bar([xi + 0.2 for xi in x], reflexive_df[wntr_col],
                   width=0.35, color="blue", alpha=0.7, label="WNTR avg return")

            ax.axhline(y=0, color="black", linewidth=1)
            ax.set_xticks(list(x))
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{w}-Day Window", fontweight="bold")
            ax.set_ylabel("Average Total Return (%)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            "Reflexive Divergence — WNTR Returns by MSTY Return Level\n"
            "(When MSTY declines, does WNTR gain proportionally?)",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()

        filepath = self.output_dir / "msty_vs_wntr_reflexive.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")

        return fig
