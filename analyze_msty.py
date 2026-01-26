#!/usr/bin/env python3
"""
MSTY Collapse Detection Analysis

Analyzes MSTY (YieldMax MSTR Option Income Strategy) to understand:
1. At what point does price moving away from the 50 SMA indicate a true collapse?
2. What distinguishes a true reflexive collapse from normal volatility?

Usage:
    python analyze_msty.py
    python analyze_msty.py --days 365
    python analyze_msty.py --current  # Just show current status
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent))

from src.data_fetcher import DataFetcher
from src.collapse_detector import CollapseDetector, CollapseEvent
from src.returns_analysis import ReturnsRegimeAnalyzer, ReturnsVisualizer


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def print_subheader(text: str):
    """Print a formatted subheader."""
    print(f"\n--- {text} ---\n")


def plot_msty_collapse_analysis(
    df: pd.DataFrame,
    collapses: list,
    optimal_threshold: float,
    output_dir: str = "output/charts",
):
    """
    Create comprehensive visualization of MSTY collapse analysis.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), height_ratios=[2, 1, 1, 1])

    # 1. Price with 50 SMA and collapse zones
    ax1 = axes[0]
    ax1.plot(df.index, df["price"], label="MSTY", linewidth=1.5, color="black")
    ax1.plot(df.index, df["sma_50"], label="50 SMA", linewidth=1.5, color="blue", alpha=0.7)

    # Shade collapse periods
    for collapse in collapses:
        end = collapse.end_date or df.index[-1]
        color = "red" if collapse.was_true_collapse else "orange"
        alpha = 0.3 if collapse.was_true_collapse else 0.2
        ax1.axvspan(collapse.start_date, end, alpha=alpha, color=color)

    ax1.set_title("MSTY Price vs 50-Day SMA with Collapse Periods", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. Distance from 50 SMA with threshold zones
    ax2 = axes[1]
    colors = ["green" if x > 0 else "red" for x in df["sma_distance_pct"]]

    ax2.fill_between(df.index, df["sma_distance_pct"], 0,
                     where=df["sma_distance_pct"] >= 0, color="green", alpha=0.3)
    ax2.fill_between(df.index, df["sma_distance_pct"], 0,
                     where=df["sma_distance_pct"] < 0, color="red", alpha=0.3)

    ax2.plot(df.index, df["sma_distance_pct"], color="black", linewidth=0.8)

    # Threshold lines
    ax2.axhline(y=0, color="black", linewidth=1)
    ax2.axhline(y=-5, color="yellow", linestyle="--", alpha=0.7, label="Warning (-5%)")
    ax2.axhline(y=-10, color="orange", linestyle="--", alpha=0.7, label="Caution (-10%)")
    ax2.axhline(y=-15, color="red", linestyle="--", alpha=0.7, label="Danger (-15%)")
    ax2.axhline(y=optimal_threshold, color="purple", linestyle="-", linewidth=2,
                label=f"Optimal ({optimal_threshold}%)")

    ax2.set_ylabel("Distance from SMA (%)")
    ax2.set_title("Distance from 50-Day SMA (Key Collapse Indicator)", fontweight="bold")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Velocity of SMA distance change
    ax3 = axes[2]
    velocity = df["sma_distance_velocity_3d"].fillna(0)

    ax3.fill_between(df.index, velocity, 0,
                     where=velocity >= 0, color="green", alpha=0.4, label="Moving toward SMA")
    ax3.fill_between(df.index, velocity, 0,
                     where=velocity < 0, color="red", alpha=0.4, label="Moving away from SMA")

    ax3.plot(df.index, velocity, color="black", linewidth=0.5)
    ax3.axhline(y=0, color="black", linewidth=1)
    ax3.axhline(y=-2, color="orange", linestyle="--", alpha=0.7, label="Accelerating (-2%/day)")
    ax3.axhline(y=-5, color="red", linestyle="--", alpha=0.7, label="Rapid (-5%/day)")

    ax3.set_ylabel("Velocity (%/day)")
    ax3.set_title("Rate of Change in SMA Distance (Collapse Acceleration)", fontweight="bold")
    ax3.legend(loc="lower left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Collapse probability over time
    ax4 = axes[3]
    detector = CollapseDetector()
    probabilities = df.apply(lambda row: detector.calculate_collapse_probability(row), axis=1)

    ax4.fill_between(df.index, probabilities * 100, 0, alpha=0.5, color="purple")
    ax4.plot(df.index, probabilities * 100, color="purple", linewidth=1)

    ax4.axhline(y=30, color="yellow", linestyle="--", alpha=0.7, label="Low Risk (30%)")
    ax4.axhline(y=50, color="orange", linestyle="--", alpha=0.7, label="Elevated (50%)")
    ax4.axhline(y=70, color="red", linestyle="--", alpha=0.7, label="High Risk (70%)")

    ax4.set_ylabel("Collapse Probability (%)")
    ax4.set_xlabel("Date")
    ax4.set_title("Real-Time Collapse Probability Score", fontweight="bold")
    ax4.set_ylim(0, 100)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = Path(output_dir) / "msty_collapse_analysis.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {filepath}")

    return fig


def plot_threshold_effectiveness(
    effectiveness_df: pd.DataFrame,
    output_dir: str = "output/charts",
):
    """
    Visualize threshold effectiveness analysis.
    """
    if effectiveness_df.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Filter to 10-day forward analysis
    df_10d = effectiveness_df[effectiveness_df["forward_days"] == 10]

    if df_10d.empty:
        df_10d = effectiveness_df

    # 1. Accuracy by threshold
    ax1 = axes[0, 0]
    ax1.bar(df_10d["threshold_pct"], df_10d["pct_continued_decline"], color="steelblue", alpha=0.7)
    ax1.axhline(y=70, color="red", linestyle="--", label="70% target")
    ax1.set_xlabel("SMA Distance Threshold (%)")
    ax1.set_ylabel("% Leading to Further Decline")
    ax1.set_title("Accuracy: How Often Does Crossing Lead to More Decline?")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Average forward return by threshold
    ax2 = axes[0, 1]
    colors = ["red" if x < 0 else "green" for x in df_10d["avg_forward_return"]]
    ax2.bar(df_10d["threshold_pct"], df_10d["avg_forward_return"], color=colors, alpha=0.7)
    ax2.axhline(y=0, color="black", linewidth=1)
    ax2.set_xlabel("SMA Distance Threshold (%)")
    ax2.set_ylabel("Avg Forward Return (%)")
    ax2.set_title("Average 10-Day Return After Crossing Threshold")
    ax2.grid(True, alpha=0.3)

    # 3. Number of signals by threshold
    ax3 = axes[1, 0]
    ax3.bar(df_10d["threshold_pct"], df_10d["num_crosses"], color="purple", alpha=0.7)
    ax3.set_xlabel("SMA Distance Threshold (%)")
    ax3.set_ylabel("Number of Signals")
    ax3.set_title("Signal Frequency by Threshold")
    ax3.grid(True, alpha=0.3)

    # 4. Risk/Reward tradeoff
    ax4 = axes[1, 1]
    ax4.scatter(df_10d["pct_continued_decline"], df_10d["avg_forward_return"],
                s=df_10d["num_crosses"] * 20, alpha=0.6, c=df_10d["threshold_pct"], cmap="RdYlGn_r")
    ax4.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax4.axvline(x=70, color="red", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Accuracy (% Continued Decline)")
    ax4.set_ylabel("Avg Forward Return (%)")
    ax4.set_title("Threshold Tradeoff: Accuracy vs Return\n(size = signal frequency)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = Path(output_dir) / "msty_threshold_effectiveness.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Saved: {filepath}")

    return fig


def run_msty_analysis(lookback_days: int = 365, show_current_only: bool = False):
    """
    Run the MSTY collapse detection analysis.

    Args:
        lookback_days: Number of days of history to analyze
        show_current_only: Only show current signal status
    """
    print_header("MSTY Collapse Detection Analysis")
    print(f"Analyzing: MSTY (YieldMax MSTR Option Income Strategy)")
    print(f"Underlying: MSTR (MicroStrategy)")
    print(f"Focus: 50-Day SMA")
    print(f"Lookback: {lookback_days} days")

    # Fetch data
    print_subheader("Fetching Data")
    fetcher = DataFetcher()

    start_date = (datetime.now() - timedelta(days=lookback_days + 60)).strftime("%Y-%m-%d")

    msty_df = fetcher.fetch_ticker("MSTY", start_date=start_date)
    mstr_df = fetcher.fetch_ticker("MSTR", start_date=start_date)

    if msty_df.empty or mstr_df.empty:
        print("ERROR: Could not fetch data for MSTY or MSTR")
        print("MSTY may be a newer ETF - try a shorter lookback period")
        return

    print(f"MSTY: {len(msty_df)} days ({msty_df.index[0].date()} to {msty_df.index[-1].date()})")
    print(f"MSTR: {len(mstr_df)} days ({mstr_df.index[0].date()} to {mstr_df.index[-1].date()})")

    # Initialize detector and prepare data
    detector = CollapseDetector(sma_period=50)
    df = detector.prepare_data(msty_df, mstr_df)

    # Trim to lookback period (after SMA calculation)
    df = df.iloc[-lookback_days:] if len(df) > lookback_days else df

    print(f"Analysis period: {df.index[0].date()} to {df.index[-1].date()}")

    # ================================================================
    # CURRENT STATUS
    # ================================================================
    print_header("Current Collapse Status")

    signal = detector.get_current_signal(df)

    if signal:
        print(f"Date: {signal.date.date()}")
        print(f"Signal Type: {signal.signal_type.upper()}")
        print(f"Distance from 50 SMA: {signal.distance_from_sma:.2f}%")
        print(f"Velocity (3-day avg): {signal.velocity:.2f}%/day")
        print(f"Days Below Threshold: {signal.days_below_threshold}")
        print(f"Current Drawdown: {signal.current_drawdown:.2f}%")
        print(f"\nCOLLAPSE PROBABILITY: {signal.probability_true_collapse * 100:.1f}%")
        print(f"\nRECOMMENDATION: {signal.recommended_action}")

        # Current price context
        latest = df.iloc[-1]
        print(f"\nCurrent Price: ${latest['price']:.2f}")
        print(f"50 SMA: ${latest['sma_50']:.2f}")
        print(f"MSTR Distance from 50 SMA: {latest['underlying_sma_distance_pct']:.2f}%")

    if show_current_only:
        return

    # ================================================================
    # FIND OPTIMAL THRESHOLD
    # ================================================================
    print_header("Optimal Threshold Analysis")
    print("Finding the SMA distance threshold that best predicts true collapses...")

    optimal = detector.find_optimal_threshold(df, target_accuracy=0.7)

    print(f"\nOPTIMAL THRESHOLD: {optimal['optimal_threshold']}% below 50 SMA")
    print(f"Accuracy: {optimal['accuracy'] * 100:.1f}% of crosses led to further decline")
    print(f"Number of signals in period: {optimal['num_signals']}")
    print(f"Average further decline: {optimal['avg_further_decline']:.2f}%")

    print("\nInterpretation:")
    print(f"  When MSTY crosses {optimal['optimal_threshold']}% below its 50 SMA,")
    print(f"  there's a {optimal['accuracy'] * 100:.0f}% chance it will decline further.")
    print(f"  This is your KEY TRIGGER LEVEL for identifying true collapses.")

    # Full threshold results
    print("\nAll Threshold Results:")
    all_results = optimal["all_results"].sort_values("threshold", ascending=False)
    print(tabulate(
        all_results[["threshold", "num_signals", "accuracy", "avg_further_decline"]].head(15),
        headers=["Threshold %", "Signals", "Accuracy", "Avg Decline %"],
        tablefmt="simple",
        floatfmt=(".0f", ".0f", ".2f", ".2f")
    ))

    # ================================================================
    # THRESHOLD EFFECTIVENESS ANALYSIS
    # ================================================================
    print_header("Threshold Effectiveness by Forward Period")

    effectiveness = detector.analyze_threshold_effectiveness(df, [5, 10, 20])

    if not effectiveness.empty:
        # Show 10-day effectiveness
        eff_10d = effectiveness[effectiveness["forward_days"] == 10].sort_values("threshold_pct", ascending=False)
        print("\n10-Day Forward Analysis:")
        print(tabulate(
            eff_10d[["threshold_pct", "num_crosses", "pct_continued_decline", "avg_forward_return", "worst_forward_return"]],
            headers=["Threshold", "Signals", "% Decline", "Avg Return", "Worst Return"],
            tablefmt="simple",
            floatfmt=(".0f", ".0f", ".1f", ".2f", ".2f")
        ))

    # ================================================================
    # HISTORICAL COLLAPSE EVENTS
    # ================================================================
    print_header("Historical Collapse Events")

    collapses = detector.identify_historical_collapses(df, min_drawdown=-15, min_duration=3)

    if collapses:
        print(f"Found {len(collapses)} collapse events:\n")

        collapse_data = []
        for i, c in enumerate(collapses, 1):
            end_str = c.end_date.date() if c.end_date else "ONGOING"
            collapse_data.append([
                i,
                c.start_date.date(),
                end_str,
                f"{c.trigger_distance:.1f}%",
                f"{c.max_distance:.1f}%",
                f"{c.max_drawdown:.1f}%",
                c.duration_days,
                f"{c.velocity_at_trigger:.2f}",
                "YES" if c.was_true_collapse else "no",
            ])

        print(tabulate(
            collapse_data,
            headers=["#", "Start", "End", "Trigger Dist", "Max Dist", "Max DD", "Days", "Velocity", "True Collapse?"],
            tablefmt="simple"
        ))

        # True collapse statistics
        true_collapses = [c for c in collapses if c.was_true_collapse]
        if true_collapses:
            print(f"\n{len(true_collapses)} TRUE COLLAPSES identified:")
            print(f"  Average trigger distance: {np.mean([c.trigger_distance for c in true_collapses]):.1f}%")
            print(f"  Average max drawdown: {np.mean([c.max_drawdown for c in true_collapses]):.1f}%")
            print(f"  Average duration: {np.mean([c.duration_days for c in true_collapses]):.0f} days")
            print(f"  Average velocity at trigger: {np.mean([c.velocity_at_trigger for c in true_collapses]):.2f}%/day")
    else:
        print("No collapse events meeting criteria found in this period.")

    # ================================================================
    # RETURNS BY SMA REGIME
    # ================================================================
    print_header("Returns Analysis by SMA Regime")
    print("How do total returns shift based on position relative to the 50 SMA?")

    returns_analyzer = ReturnsRegimeAnalyzer(sma_period=50)
    returns_viz = ReturnsVisualizer()

    # Prepare returns data
    returns_df = returns_analyzer.prepare_returns_data(df, price_col="price")
    returns_df = returns_analyzer.calculate_rolling_regime_returns(returns_df, window=20)

    # Calculate regime statistics
    regime_stats = returns_analyzer.calculate_regime_stats(returns_df, "regime_simple")

    print_subheader("Return Statistics by Regime")
    if regime_stats:
        regime_table = []
        for regime in ["trending_up", "near_sma", "trending_down"]:
            if regime in regime_stats:
                s = regime_stats[regime]
                regime_table.append([
                    regime,
                    s.num_days,
                    f"{s.total_return_pct:.2f}%",
                    f"{s.avg_daily_return:.3f}%",
                    f"{s.pct_positive_days:.1f}%",
                    f"{s.skewness:.2f}",
                    f"{s.worst_day:.2f}%",
                    f"{s.best_day:.2f}%",
                ])

        print(tabulate(
            regime_table,
            headers=["Regime", "Days", "Total Return", "Avg Daily", "% Positive", "Skew", "Worst Day", "Best Day"],
            tablefmt="simple"
        ))

        # Key insight about regimes
        print("\nKEY FINDING:")
        if "trending_down" in regime_stats and "trending_up" in regime_stats:
            down_stats = regime_stats["trending_down"]
            up_stats = regime_stats["trending_up"]
            print(f"  When TRENDING DOWN (>5% below SMA):")
            print(f"    - Total return: {down_stats.total_return_pct:.2f}%")
            print(f"    - Only {down_stats.pct_positive_days:.1f}% of days are positive")
            print(f"    - Return skew: {down_stats.skewness:.2f} (negative = more extreme losses)")
            print(f"\n  When TRENDING UP (>5% above SMA):")
            print(f"    - Total return: {up_stats.total_return_pct:.2f}%")
            print(f"    - {up_stats.pct_positive_days:.1f}% of days are positive")
            print(f"    - Return skew: {up_stats.skewness:.2f}")

    # Analyze return distribution by distance
    print_subheader("Return Distribution by SMA Distance")
    distribution = returns_analyzer.analyze_return_distribution_by_sma_distance(
        returns_df,
        distance_bins=[-30, -20, -10, -5, 0, 5, 10, 20, 30]
    )

    if not distribution.empty:
        print(tabulate(
            distribution[["distance_range", "num_days", "avg_daily_return", "pct_negative", "p10_return", "p90_return"]],
            headers=["Distance Range", "Days", "Avg Return", "% Negative", "10th Pctl", "90th Pctl"],
            tablefmt="simple",
            floatfmt=("", ".0f", ".3f", ".1f", ".2f", ".2f")
        ))

        # Find the inflection point where returns turn negative
        negative_zones = distribution[distribution["avg_daily_return"] < 0]
        if not negative_zones.empty:
            worst_zone = negative_zones.loc[negative_zones["avg_daily_return"].idxmin()]
            print(f"\n  WORST ZONE: {worst_zone['distance_range']}")
            print(f"    Average daily return: {worst_zone['avg_daily_return']:.3f}%")
            print(f"    {worst_zone['pct_negative']:.0f}% of days are negative")

    # Cumulative return breakdown
    print_subheader("Cumulative Return Breakdown")
    if "cumul_return_trending_down" in returns_df.columns:
        final_down = returns_df["cumul_return_trending_down"].iloc[-1]
        final_near = returns_df["cumul_return_near_sma"].iloc[-1]
        final_up = returns_df["cumul_return_trending_up"].iloc[-1]
        total = final_down + final_near + final_up

        print(f"  Return earned while TRENDING DOWN: {final_down:.2f}%")
        print(f"  Return earned while NEAR SMA:      {final_near:.2f}%")
        print(f"  Return earned while TRENDING UP:   {final_up:.2f}%")
        print(f"  -----------------------------------")
        print(f"  TOTAL CUMULATIVE RETURN:           {total:.2f}%")

        if total != 0:
            print(f"\n  Contribution breakdown:")
            print(f"    Trending down: {(final_down/total)*100:.1f}% of total return")
            print(f"    Near SMA:      {(final_near/total)*100:.1f}% of total return")
            print(f"    Trending up:   {(final_up/total)*100:.1f}% of total return")

    # Generate returns visualizations
    print_subheader("Generating Returns Charts")
    returns_viz.plot_returns_by_regime(returns_df, regime_stats, ticker="MSTY")
    returns_viz.plot_return_distribution_by_distance(distribution, ticker="MSTY")

    # ================================================================
    # KEY INSIGHTS
    # ================================================================
    print_header("Key Insights for MSTY Collapse Detection")

    print("""
COLLAPSE IDENTIFICATION RULES:

1. PRIMARY TRIGGER:
   - Watch for MSTY crossing {threshold}% below the 50 SMA
   - This is the optimal threshold based on historical accuracy

2. CONFIRMATION SIGNALS (increases confidence):
   - Velocity < -2%/day (price accelerating away from SMA)
   - 3+ consecutive days below danger level
   - MSTR also below its 50 SMA
   - Drawdown already > 10% from peak

3. FALSE POSITIVE INDICATORS (may recover):
   - Velocity slowing (becoming less negative)
   - MSTR holding above its 50 SMA
   - Quick bounce back above warning level
   - Low overall collapse probability score

4. EXIT COLLAPSE DETECTION:
   - Price returns above the -5% (warning) level
   - Velocity turns positive
   - Collapse probability drops below 30%

RECOMMENDED MONITORING:
   - Run this analysis daily during volatile periods
   - Use --current flag for quick status check
   - Pay special attention when probability > 50%
""".format(threshold=optimal['optimal_threshold']))

    # ================================================================
    # GENERATE CHARTS
    # ================================================================
    print_header("Generating Visualizations")

    plot_msty_collapse_analysis(df, collapses, optimal["optimal_threshold"])
    plot_threshold_effectiveness(effectiveness)

    print("\nAnalysis complete. Check output/charts/ for visualizations.")


def main():
    parser = argparse.ArgumentParser(
        description="MSTY Collapse Detection Analysis"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=365,
        help="Number of days to analyze (default: 365)"
    )
    parser.add_argument(
        "--current", "-c",
        action="store_true",
        help="Only show current collapse status"
    )

    args = parser.parse_args()

    run_msty_analysis(
        lookback_days=args.days,
        show_current_only=args.current,
    )


if __name__ == "__main__":
    main()
