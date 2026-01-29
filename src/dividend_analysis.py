"""
Dividend growth/decline analysis module.

Compares MSTY and WNTR dividend trajectories on a weekly basis to reveal
the reflexive unwind: as MSTY's dividends shrink (reflecting covered call
premium erosion on declining equity), WNTR's dividends should grow (reflecting
gains from the inverse position on the same underlying).

Key considerations:
- MSTY had a 1:5 stock split on Dec 8, 2025. Post-split per-share dividends
  are 1/5 of what they'd be pre-split, so we normalize to "per original share"
  by multiplying post-split dividends by the cumulative split factor.
- Both instruments switched from monthly to weekly dividends around Oct 2025.
  All comparisons use weekly aggregation for apples-to-apples comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


@dataclass
class DividendPayment:
    """A single dividend payment, possibly split-adjusted."""
    date: pd.Timestamp
    raw_amount: float
    split_adjusted_amount: float
    price_on_date: float
    yield_pct: float  # dividend / price * 100


class DividendAnalyzer:
    """
    Analyzes dividend growth/decline patterns for MSTY vs WNTR.

    Handles:
    - Split adjustment for MSTY (normalizes post-split dividends to pre-split equivalent)
    - Weekly aggregation of dividends from both instruments
    - Growth rate calculation (week-over-week, rolling averages)
    - Cross-instrument comparison of dividend trajectories
    """

    def extract_dividends(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        dividend_col: str = "dividend",
        split_col: str = "split",
    ) -> pd.DataFrame:
        """
        Extract dividend-only rows and compute split-adjusted amounts.

        For an instrument with a split (e.g., MSTY 1:5), dividends paid after
        the split are per-new-share. To compare fairly to pre-split dividends,
        multiply post-split dividends by the cumulative split factor so they
        represent income per original share.

        Returns:
            DataFrame with columns: raw_amount, split_adjusted_amount,
            price, yield_pct, cumulative_split
        """
        data = df.copy().sort_index()

        # Calculate cumulative split factor (forward-looking)
        if split_col in data.columns:
            cumulative_split = data[split_col].replace(0, 1).cumprod()
        else:
            cumulative_split = pd.Series(1, index=data.index)

        # Filter to dividend days only
        div_mask = data[dividend_col] > 0
        div_days = data[div_mask].copy()

        if div_days.empty:
            return pd.DataFrame()

        result = pd.DataFrame(index=div_days.index)
        result["raw_amount"] = div_days[dividend_col]
        result["cumulative_split"] = cumulative_split.loc[div_days.index]

        # Split-adjusted = raw_amount * cumulative_split_at_that_date
        # This gives "income per original share" — if you held 1 share pre-split,
        # you now hold N shares, so total income = per-share div * N
        result["split_adjusted_amount"] = (
            result["raw_amount"] * result["cumulative_split"]
        )

        result["price"] = div_days[price_col]
        result["yield_pct"] = (result["raw_amount"] / result["price"]) * 100

        # For split-adjusted yield, use the split-adjusted dividend
        # against the price (which is already post-split)
        # Actually, yield per original investment = split_adj_div / (price * cumulative_split)
        # which simplifies back to raw_div / price. So yield_pct is the same either way.

        return result

    def aggregate_weekly(
        self,
        dividends: pd.DataFrame,
        amount_col: str = "split_adjusted_amount",
    ) -> pd.DataFrame:
        """
        Aggregate dividends to weekly totals.

        Uses ISO week-year grouping. Weeks with no dividends get 0.

        Returns:
            DataFrame indexed by week-start date with weekly totals
        """
        if dividends.empty:
            return pd.DataFrame()

        # Create a week label (Monday of each week)
        div_data = dividends.copy()
        div_data["week_start"] = div_data.index - pd.to_timedelta(
            div_data.index.dayofweek, unit="d"
        )

        weekly = div_data.groupby("week_start").agg(
            weekly_amount=(amount_col, "sum"),
            num_payments=("raw_amount", "count"),
            avg_price=("price", "mean"),
        )

        weekly.index = pd.to_datetime(weekly.index)
        weekly["weekly_yield_pct"] = (weekly["weekly_amount"] / weekly["avg_price"]) * 100

        return weekly

    def build_weekly_comparison(
        self,
        msty_df: pd.DataFrame,
        wntr_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build a side-by-side weekly dividend comparison.

        Aligns both instruments to the same weekly timeline, filling weeks
        with no dividend as 0.

        Returns:
            DataFrame with msty and wntr weekly dividends aligned
        """
        msty_divs = self.extract_dividends(msty_df)
        wntr_divs = self.extract_dividends(wntr_df)

        msty_weekly = self.aggregate_weekly(msty_divs)
        wntr_weekly = self.aggregate_weekly(wntr_divs)

        # Get full weekly range spanning both instruments
        all_weeks = pd.date_range(
            start=min(
                msty_weekly.index.min() if not msty_weekly.empty else pd.Timestamp.max,
                wntr_weekly.index.min() if not wntr_weekly.empty else pd.Timestamp.max,
            ),
            end=max(
                msty_weekly.index.max() if not msty_weekly.empty else pd.Timestamp.min,
                wntr_weekly.index.max() if not wntr_weekly.empty else pd.Timestamp.min,
            ),
            freq="W-MON",
        )

        comparison = pd.DataFrame(index=all_weeks)
        comparison.index.name = "week_start"

        # MSTY columns
        if not msty_weekly.empty:
            comparison["msty_weekly_div"] = msty_weekly["weekly_amount"]
            comparison["msty_num_payments"] = msty_weekly["num_payments"]
            comparison["msty_avg_price"] = msty_weekly["avg_price"]
            comparison["msty_yield_pct"] = msty_weekly["weekly_yield_pct"]
        else:
            comparison["msty_weekly_div"] = 0
            comparison["msty_num_payments"] = 0

        # WNTR columns
        if not wntr_weekly.empty:
            comparison["wntr_weekly_div"] = wntr_weekly["weekly_amount"]
            comparison["wntr_num_payments"] = wntr_weekly["num_payments"]
            comparison["wntr_avg_price"] = wntr_weekly["avg_price"]
            comparison["wntr_yield_pct"] = wntr_weekly["weekly_yield_pct"]
        else:
            comparison["wntr_weekly_div"] = 0
            comparison["wntr_num_payments"] = 0

        # Fill NaN (weeks with no dividend) with 0
        comparison["msty_weekly_div"] = comparison["msty_weekly_div"].fillna(0)
        comparison["wntr_weekly_div"] = comparison["wntr_weekly_div"].fillna(0)
        comparison["msty_num_payments"] = comparison["msty_num_payments"].fillna(0).astype(int)
        comparison["wntr_num_payments"] = comparison["wntr_num_payments"].fillna(0).astype(int)

        # Cumulative dividends
        comparison["msty_cumulative_div"] = comparison["msty_weekly_div"].cumsum()
        comparison["wntr_cumulative_div"] = comparison["wntr_weekly_div"].cumsum()

        return comparison

    def calculate_dividend_growth(
        self,
        weekly_comparison: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate dividend growth rates on the weekly comparison.

        Only computes growth between weeks that actually had dividend payments.
        Adds rolling averages for smoothing.

        Returns:
            The input DataFrame augmented with growth columns
        """
        result = weekly_comparison.copy()

        for instrument in ["msty", "wntr"]:
            div_col = f"{instrument}_weekly_div"
            # Filter to weeks with actual payments for growth calc
            has_payment = result[div_col] > 0

            # Week-over-week growth (only between payment weeks)
            payment_amounts = result.loc[has_payment, div_col]
            wow_growth = payment_amounts.pct_change() * 100
            result[f"{instrument}_wow_growth"] = np.nan
            result.loc[wow_growth.index, f"{instrument}_wow_growth"] = wow_growth

            # Rolling 4-week average dividend (using payment weeks only)
            result[f"{instrument}_4wk_avg_div"] = (
                payment_amounts.rolling(4, min_periods=2).mean()
            )
            # Reindex to fill in for all weeks
            result[f"{instrument}_4wk_avg_div"] = (
                result[f"{instrument}_4wk_avg_div"].ffill()
            )

            # Rolling 4-week average growth rate
            result[f"{instrument}_4wk_avg_growth"] = (
                wow_growth.rolling(4, min_periods=2).mean()
            )
            result[f"{instrument}_4wk_avg_growth"] = (
                result[f"{instrument}_4wk_avg_growth"].ffill()
            )

        # Dividend spread: WNTR weekly div - MSTY weekly div
        # (only meaningful when both have payments in the same week)
        both_paid = (result["msty_weekly_div"] > 0) & (result["wntr_weekly_div"] > 0)
        result["div_spread"] = np.nan
        result.loc[both_paid, "div_spread"] = (
            result.loc[both_paid, "wntr_weekly_div"]
            - result.loc[both_paid, "msty_weekly_div"]
        )

        # Ratio: WNTR div / MSTY div (when both pay)
        result["div_ratio"] = np.nan
        valid_ratio = both_paid & (result["msty_weekly_div"] > 0)
        result.loc[valid_ratio, "div_ratio"] = (
            result.loc[valid_ratio, "wntr_weekly_div"]
            / result.loc[valid_ratio, "msty_weekly_div"]
        )

        return result

    def analyze_frequency_shift(
        self,
        dividends: pd.DataFrame,
        instrument_name: str,
    ) -> pd.DataFrame:
        """
        Analyze the transition from monthly to weekly dividends.

        Identifies the frequency regime (monthly vs weekly) and computes
        statistics for each regime.

        Returns:
            DataFrame with regime analysis
        """
        if dividends.empty:
            return pd.DataFrame()

        # Calculate days between consecutive dividends
        div_dates = dividends.index.sort_values()
        gaps = pd.Series(
            [(div_dates[i] - div_dates[i - 1]).days for i in range(1, len(div_dates))],
            index=div_dates[1:],
        )

        # Classify: weekly = gap <= 10 days, monthly = gap > 10 days
        regime = pd.DataFrame(index=div_dates)
        regime["amount"] = dividends["split_adjusted_amount"]
        regime["raw_amount"] = dividends["raw_amount"]
        regime["price"] = dividends["price"]
        regime["gap_days"] = np.nan
        regime.loc[gaps.index, "gap_days"] = gaps.values
        regime["frequency"] = "monthly"
        regime.loc[regime["gap_days"] <= 10, "frequency"] = "weekly"
        # First payment has no gap - classify based on next payment
        if len(regime) > 1 and pd.isna(regime["gap_days"].iloc[0]):
            regime.iloc[0, regime.columns.get_loc("frequency")] = regime["frequency"].iloc[1]

        # Summary by regime
        results = []
        for freq in ["monthly", "weekly"]:
            subset = regime[regime["frequency"] == freq]
            if subset.empty:
                continue

            results.append({
                "instrument": instrument_name,
                "frequency": freq,
                "period": f"{subset.index[0].date()} to {subset.index[-1].date()}",
                "num_payments": len(subset),
                "avg_amount_split_adj": subset["amount"].mean(),
                "total_amount_split_adj": subset["amount"].sum(),
                "avg_yield_pct": (subset["amount"] / subset["price"]).mean() * 100,
                "avg_gap_days": subset["gap_days"].mean(),
                "min_amount": subset["amount"].min(),
                "max_amount": subset["amount"].max(),
            })

        return pd.DataFrame(results)

    def analyze_weekly_period_trends(
        self,
        weekly_comparison: pd.DataFrame,
    ) -> Dict:
        """
        Analyze trends specifically during the weekly dividend period (Oct 2025+).

        Returns:
            Dict with trend statistics for each instrument
        """
        # Filter to weekly period (Oct 2025 onward)
        weekly_period = weekly_comparison.loc["2025-10-01":]

        trends = {}
        for instrument in ["msty", "wntr"]:
            div_col = f"{instrument}_weekly_div"
            has_payment = weekly_period[div_col] > 0
            payments = weekly_period.loc[has_payment, div_col]

            if len(payments) < 2:
                continue

            # Linear regression for trend
            x = np.arange(len(payments))
            y = payments.values
            slope, intercept = np.polyfit(x, y, 1)

            # First half vs second half
            mid = len(payments) // 2
            first_half_avg = payments.iloc[:mid].mean()
            second_half_avg = payments.iloc[mid:].mean()

            trends[instrument] = {
                "num_weekly_payments": len(payments),
                "first_payment": payments.iloc[0],
                "last_payment": payments.iloc[-1],
                "change_pct": (payments.iloc[-1] / payments.iloc[0] - 1) * 100,
                "avg_amount": payments.mean(),
                "trend_slope": slope,
                "trend_direction": "declining" if slope < 0 else "growing",
                "first_half_avg": first_half_avg,
                "second_half_avg": second_half_avg,
                "half_change_pct": (second_half_avg / first_half_avg - 1) * 100,
                "min_payment": payments.min(),
                "max_payment": payments.max(),
            }

        return trends


class DividendVisualizer:
    """Visualization for dividend growth/decline analysis."""

    def __init__(self, output_dir: str = "output/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_weekly_dividends_comparison(
        self,
        weekly_comparison: pd.DataFrame,
        msty_divs: pd.DataFrame,
        wntr_divs: pd.DataFrame,
    ) -> plt.Figure:
        """
        4-panel chart:
        1. Individual dividend payments (split-adjusted) timeline
        2. Weekly dividend amounts side-by-side bars
        3. Cumulative dividend income
        4. Weekly dividend yield %
        """
        fig, axes = plt.subplots(4, 1, figsize=(16, 18))

        # --- Panel 1: Individual dividend payments (scatter/stem) ---
        ax1 = axes[0]

        if not msty_divs.empty:
            ax1.stem(
                msty_divs.index, msty_divs["split_adjusted_amount"],
                linefmt="r-", markerfmt="ro", basefmt="k-",
                label="MSTY (split-adjusted)",
            )
        if not wntr_divs.empty:
            ax1.stem(
                wntr_divs.index, wntr_divs["split_adjusted_amount"],
                linefmt="b-", markerfmt="bs", basefmt="k-",
                label="WNTR",
            )

        # Mark the split date
        ax1.axvline(
            pd.Timestamp("2025-12-08"), color="gray", linestyle="--",
            linewidth=1.5, alpha=0.7, label="MSTY 1:5 Split (Dec 8)"
        )
        # Mark frequency change
        ax1.axvline(
            pd.Timestamp("2025-10-16"), color="orange", linestyle=":",
            linewidth=1.5, alpha=0.7, label="Weekly dividends begin"
        )

        ax1.set_title(
            "Individual Dividend Payments — Split-Adjusted Per Original Share",
            fontweight="bold", fontsize=12,
        )
        ax1.set_ylabel("Dividend Amount ($)")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # --- Panel 2: Weekly dividend bars (only weeks with payments) ---
        ax2 = axes[1]

        # Filter to weeks where at least one instrument paid
        has_any = (weekly_comparison["msty_weekly_div"] > 0) | (
            weekly_comparison["wntr_weekly_div"] > 0
        )
        weekly_paid = weekly_comparison[has_any].copy()

        if not weekly_paid.empty:
            bar_width = pd.Timedelta(days=2)
            ax2.bar(
                weekly_paid.index - bar_width / 2,
                weekly_paid["msty_weekly_div"],
                width=bar_width,
                color="red", alpha=0.7, label="MSTY (split-adj)",
            )
            ax2.bar(
                weekly_paid.index + bar_width / 2,
                weekly_paid["wntr_weekly_div"],
                width=bar_width,
                color="blue", alpha=0.7, label="WNTR",
            )

            # Add trend lines for weekly period (Oct 2025+)
            weekly_only = weekly_paid.loc["2025-10-01":]
            for instrument, color in [("msty", "red"), ("wntr", "blue")]:
                col = f"{instrument}_weekly_div"
                paid = weekly_only[weekly_only[col] > 0]
                if len(paid) >= 3:
                    x_num = mdates.date2num(paid.index)
                    z = np.polyfit(x_num, paid[col].values, 1)
                    p = np.poly1d(z)
                    ax2.plot(
                        paid.index, p(x_num),
                        color=color, linewidth=2, linestyle="--", alpha=0.8,
                    )

        ax2.axvline(
            pd.Timestamp("2025-10-16"), color="orange", linestyle=":",
            linewidth=1.5, alpha=0.7,
        )
        ax2.set_title(
            "Weekly Dividend Amounts — MSTY (Split-Adjusted) vs WNTR",
            fontweight="bold", fontsize=12,
        )
        ax2.set_ylabel("Weekly Dividend ($)")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)

        # --- Panel 3: Cumulative dividend income ---
        ax3 = axes[2]

        ax3.plot(
            weekly_comparison.index, weekly_comparison["msty_cumulative_div"],
            color="red", linewidth=2, label="MSTY cumulative (split-adj)",
        )
        ax3.plot(
            weekly_comparison.index, weekly_comparison["wntr_cumulative_div"],
            color="blue", linewidth=2, label="WNTR cumulative",
        )
        ax3.fill_between(
            weekly_comparison.index,
            weekly_comparison["msty_cumulative_div"],
            weekly_comparison["wntr_cumulative_div"],
            where=weekly_comparison["msty_cumulative_div"] >= weekly_comparison["wntr_cumulative_div"],
            alpha=0.15, color="red", label="MSTY leading",
        )
        ax3.fill_between(
            weekly_comparison.index,
            weekly_comparison["msty_cumulative_div"],
            weekly_comparison["wntr_cumulative_div"],
            where=weekly_comparison["msty_cumulative_div"] < weekly_comparison["wntr_cumulative_div"],
            alpha=0.15, color="blue", label="WNTR leading",
        )

        ax3.axvline(
            pd.Timestamp("2025-10-16"), color="orange", linestyle=":",
            linewidth=1.5, alpha=0.7,
        )
        ax3.set_title(
            "Cumulative Dividend Income — Split-Adjusted Per Original Share",
            fontweight="bold", fontsize=12,
        )
        ax3.set_ylabel("Cumulative Dividends ($)")
        ax3.legend(loc="upper left", fontsize=9)
        ax3.grid(True, alpha=0.3)

        # --- Panel 4: Dividend yield per payment ---
        ax4 = axes[3]

        if not msty_divs.empty:
            ax4.plot(
                msty_divs.index, msty_divs["yield_pct"],
                "ro-", markersize=5, linewidth=1, alpha=0.8,
                label="MSTY yield per payment",
            )
        if not wntr_divs.empty:
            ax4.plot(
                wntr_divs.index, wntr_divs["yield_pct"],
                "bs-", markersize=5, linewidth=1, alpha=0.8,
                label="WNTR yield per payment",
            )

        ax4.axvline(
            pd.Timestamp("2025-10-16"), color="orange", linestyle=":",
            linewidth=1.5, alpha=0.7,
        )
        ax4.axvline(
            pd.Timestamp("2025-12-08"), color="gray", linestyle="--",
            linewidth=1.5, alpha=0.7,
        )
        ax4.set_title(
            "Dividend Yield Per Payment (Dividend / Price %)",
            fontweight="bold", fontsize=12,
        )
        ax4.set_ylabel("Yield (%)")
        ax4.set_xlabel("Date")
        ax4.legend(loc="upper right", fontsize=9)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / "dividend_growth_comparison.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")
        plt.close()
        return fig

    def plot_weekly_period_deep_dive(
        self,
        weekly_comparison: pd.DataFrame,
        msty_divs: pd.DataFrame,
        wntr_divs: pd.DataFrame,
    ) -> plt.Figure:
        """
        Focused chart on the weekly dividend period (Oct 2025+).

        3 panels:
        1. Weekly dividends side-by-side with trend lines
        2. Week-over-week growth rates
        3. WNTR/MSTY dividend ratio evolution
        """
        # Filter to weekly period
        wc = weekly_comparison.loc["2025-10-01":].copy()
        msty_w = msty_divs.loc["2025-10-01":] if not msty_divs.empty else msty_divs
        wntr_w = wntr_divs.loc["2025-10-01":] if not wntr_divs.empty else wntr_divs

        fig, axes = plt.subplots(3, 1, figsize=(16, 14))

        # --- Panel 1: Weekly dividends with annotations ---
        ax1 = axes[0]

        if not msty_w.empty:
            ax1.plot(
                msty_w.index, msty_w["split_adjusted_amount"],
                "ro-", markersize=7, linewidth=1.5, label="MSTY (split-adj)",
            )
            # Annotate amounts
            for idx, row in msty_w.iterrows():
                ax1.annotate(
                    f"${row['split_adjusted_amount']:.2f}",
                    (idx, row["split_adjusted_amount"]),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=7, color="red", ha="center",
                )

        if not wntr_w.empty:
            ax1.plot(
                wntr_w.index, wntr_w["split_adjusted_amount"],
                "bs-", markersize=7, linewidth=1.5, label="WNTR",
            )
            for idx, row in wntr_w.iterrows():
                ax1.annotate(
                    f"${row['split_adjusted_amount']:.2f}",
                    (idx, row["split_adjusted_amount"]),
                    textcoords="offset points", xytext=(0, -15),
                    fontsize=7, color="blue", ha="center",
                )

        ax1.axvline(
            pd.Timestamp("2025-12-08"), color="gray", linestyle="--",
            linewidth=1.5, alpha=0.7, label="MSTY 1:5 Split"
        )
        ax1.set_title(
            "Weekly Period Dividends — Per Original Share (Split-Adjusted)",
            fontweight="bold", fontsize=12,
        )
        ax1.set_ylabel("Dividend Amount ($)")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # --- Panel 2: Week-over-week growth rates ---
        ax2 = axes[1]

        for instrument, color, marker in [("msty", "red", "o"), ("wntr", "blue", "s")]:
            growth_col = f"{instrument}_wow_growth"
            if growth_col in wc.columns:
                growth_data = wc[growth_col].dropna()
                if not growth_data.empty:
                    ax2.plot(
                        growth_data.index, growth_data,
                        f"{color[0]}{marker}-", markersize=6, linewidth=1.2,
                        label=f"{instrument.upper()} WoW growth",
                        color=color, alpha=0.8,
                    )

        ax2.axhline(y=0, color="black", linewidth=1)
        ax2.axvline(
            pd.Timestamp("2025-12-08"), color="gray", linestyle="--",
            linewidth=1.5, alpha=0.7,
        )
        ax2.set_title(
            "Week-over-Week Dividend Growth Rate (%)",
            fontweight="bold", fontsize=12,
        )
        ax2.set_ylabel("Growth Rate (%)")
        ax2.legend(loc="best", fontsize=9)
        ax2.grid(True, alpha=0.3)

        # --- Panel 3: WNTR/MSTY dividend ratio ---
        ax3 = axes[2]

        if "div_ratio" in wc.columns:
            ratio_data = wc["div_ratio"].dropna()
            if not ratio_data.empty:
                ax3.plot(
                    ratio_data.index, ratio_data,
                    "go-", markersize=7, linewidth=1.5, label="WNTR / MSTY ratio",
                )
                ax3.axhline(y=1.0, color="black", linewidth=1, linestyle="--",
                            label="Parity (1.0)")

                # Annotate ratio values
                for idx, val in ratio_data.items():
                    ax3.annotate(
                        f"{val:.2f}x",
                        (idx, val),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=8, ha="center", color="green",
                    )

        ax3.axvline(
            pd.Timestamp("2025-12-08"), color="gray", linestyle="--",
            linewidth=1.5, alpha=0.7, label="MSTY 1:5 Split"
        )
        ax3.set_title(
            "WNTR / MSTY Weekly Dividend Ratio — Above 1.0 = WNTR Paying More",
            fontweight="bold", fontsize=12,
        )
        ax3.set_ylabel("Ratio (WNTR / MSTY)")
        ax3.set_xlabel("Date")
        ax3.legend(loc="best", fontsize=9)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / "dividend_weekly_deep_dive.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")
        plt.close()
        return fig

    def plot_dividend_vs_price(
        self,
        msty_df: pd.DataFrame,
        wntr_df: pd.DataFrame,
        msty_divs: pd.DataFrame,
        wntr_divs: pd.DataFrame,
    ) -> plt.Figure:
        """
        Show price and dividends together to visualize the reflexive unwind:
        - MSTY: price down + dividends down = reflexive collapse
        - WNTR: price stable/up + dividends up = inverse beneficiary

        2 panels (one per instrument) with dual y-axes.
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # --- MSTY ---
        ax1 = axes[0]
        msty_sorted = msty_df.sort_index()

        # Split-adjust MSTY prices for continuity
        if "split" in msty_sorted.columns:
            cum_split = msty_sorted["split"].replace(0, 1).cumprod()
            final_split = cum_split.iloc[-1]
            # Normalize older prices down to post-split equivalent
            adj_price = msty_sorted["close"] / (cum_split / final_split)
        else:
            adj_price = msty_sorted["close"]

        ax1.plot(adj_price.index, adj_price, "r-", linewidth=1.5, label="MSTY Price (split-adj)")
        ax1.set_ylabel("Price ($)", color="red")
        ax1.tick_params(axis="y", labelcolor="red")

        ax1_div = ax1.twinx()
        if not msty_divs.empty:
            ax1_div.bar(
                msty_divs.index, msty_divs["split_adjusted_amount"],
                width=3, color="red", alpha=0.4, label="MSTY Dividend (split-adj)"
            )
        ax1_div.set_ylabel("Dividend ($)", color="darkred")
        ax1_div.tick_params(axis="y", labelcolor="darkred")

        ax1.axvline(pd.Timestamp("2025-10-16"), color="orange", linestyle=":", linewidth=1.5, alpha=0.7)
        ax1.axvline(pd.Timestamp("2025-12-08"), color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

        ax1.set_title(
            "MSTY — Price + Dividends (Both Declining = Reflexive Collapse)",
            fontweight="bold", fontsize=12,
        )
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_div.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # --- WNTR ---
        ax2 = axes[1]
        wntr_sorted = wntr_df.sort_index()
        ax2.plot(wntr_sorted.index, wntr_sorted["close"], "b-", linewidth=1.5, label="WNTR Price")
        ax2.set_ylabel("Price ($)", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")

        ax2_div = ax2.twinx()
        if not wntr_divs.empty:
            ax2_div.bar(
                wntr_divs.index, wntr_divs["split_adjusted_amount"],
                width=3, color="blue", alpha=0.4, label="WNTR Dividend"
            )
        ax2_div.set_ylabel("Dividend ($)", color="darkblue")
        ax2_div.tick_params(axis="y", labelcolor="darkblue")

        ax2.axvline(pd.Timestamp("2025-10-16"), color="orange", linestyle=":", linewidth=1.5, alpha=0.7)

        ax2.set_title(
            "WNTR — Price + Dividends (Inverse Beneficiary)",
            fontweight="bold", fontsize=12,
        )
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_div.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("Date")

        plt.tight_layout()
        filepath = self.output_dir / "dividend_vs_price.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")
        plt.close()
        return fig
