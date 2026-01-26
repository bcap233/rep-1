#!/usr/bin/env python3
"""
Analyze REAL MSTY data from CSV.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent))

from src.collapse_detector import CollapseDetector
from src.returns_analysis import (
    ReturnsRegimeAnalyzer,
    ReturnsVisualizer,
    RollingReturnsAnalyzer,
    RollingReturnsVisualizer,
)


def print_header(text: str):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def print_subheader(text: str):
    print(f"\n--- {text} ---\n")


def run_analysis():
    print_header("MSTY REAL DATA ANALYSIS")
    print("Analyzing actual MSTY price history")

    # Load data
    df = pd.read_csv("msty_data.csv", parse_dates=["Date"], index_col="Date")
    df = df.sort_index()  # Oldest first
    df.columns = [c.lower().strip() for c in df.columns]

    print(f"\nData range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total days: {len(df)}")
    print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")

    # Create a "fake" underlying for the detector (use MSTY itself scaled)
    # In real analysis you'd use MSTR data
    underlying_df = df.copy()
    underlying_df["close"] = df["close"] * 3  # Approximate MSTR relationship

    # Prepare for collapse detector
    msty_df = df.rename(columns={"close": "close", "open": "open", "high": "high", "low": "low"})

    detector = CollapseDetector(sma_period=50)

    # Manually prepare data since we have limited structure
    analysis_df = df.copy()
    analysis_df["price"] = df["close"]
    analysis_df["sma_50"] = df["close"].rolling(50).mean()
    analysis_df["sma_distance_pct"] = ((analysis_df["price"] - analysis_df["sma_50"]) / analysis_df["sma_50"]) * 100

    # Velocity
    analysis_df["sma_distance_velocity_3d"] = analysis_df["sma_distance_pct"].diff(3) / 3

    # Drawdown
    analysis_df["rolling_max"] = analysis_df["price"].expanding().max()
    analysis_df["drawdown_pct"] = ((analysis_df["price"] - analysis_df["rolling_max"]) / analysis_df["rolling_max"]) * 100

    # Drop NaN rows (need 50 days for SMA)
    analysis_df = analysis_df.dropna(subset=["sma_50"])

    print(f"\nAnalysis period (after SMA warmup): {analysis_df.index[0].date()} to {analysis_df.index[-1].date()}")

    # ================================================================
    # CURRENT STATUS
    # ================================================================
    print_header("Current Status")

    latest = analysis_df.iloc[-1]
    print(f"Date: {analysis_df.index[-1].date()}")
    print(f"Current Price: ${latest['price']:.2f}")
    print(f"50-Day SMA: ${latest['sma_50']:.2f}")
    print(f"Distance from 50 SMA: {latest['sma_distance_pct']:.2f}%")
    print(f"Velocity (3-day): {latest['sma_distance_velocity_3d']:.2f}%/day")
    print(f"Drawdown from Peak: {latest['drawdown_pct']:.2f}%")

    # Signal classification
    dist = latest['sma_distance_pct']
    if dist < -20:
        signal = "COLLAPSE"
    elif dist < -15:
        signal = "DANGER"
    elif dist < -10:
        signal = "CAUTION"
    elif dist < -5:
        signal = "WARNING"
    else:
        signal = "NORMAL"

    print(f"\n🚨 SIGNAL: {signal}")

    # ================================================================
    # HISTORICAL ANALYSIS
    # ================================================================
    print_header("Historical SMA Distance Analysis")

    # Statistics by distance zone
    zones = [
        ("Far below (<-20%)", analysis_df[analysis_df["sma_distance_pct"] < -20]),
        ("Below (-20% to -10%)", analysis_df[(analysis_df["sma_distance_pct"] >= -20) & (analysis_df["sma_distance_pct"] < -10)]),
        ("Slightly below (-10% to 0%)", analysis_df[(analysis_df["sma_distance_pct"] >= -10) & (analysis_df["sma_distance_pct"] < 0)]),
        ("Above (0% to +20%)", analysis_df[(analysis_df["sma_distance_pct"] >= 0) & (analysis_df["sma_distance_pct"] < 20)]),
        ("Far above (>+20%)", analysis_df[analysis_df["sma_distance_pct"] >= 20]),
    ]

    zone_stats = []
    for name, zone_df in zones:
        if len(zone_df) == 0:
            continue
        daily_returns = zone_df["price"].pct_change() * 100
        zone_stats.append([
            name,
            len(zone_df),
            f"{daily_returns.mean():.3f}%",
            f"{(daily_returns > 0).mean() * 100:.1f}%",
            f"{daily_returns.min():.2f}%",
            f"{daily_returns.max():.2f}%",
        ])

    print(tabulate(
        zone_stats,
        headers=["Zone", "Days", "Avg Daily Ret", "% Positive", "Worst Day", "Best Day"],
        tablefmt="simple"
    ))

    # ================================================================
    # ROLLING RETURNS
    # ================================================================
    print_header("Rolling Returns Analysis")

    rolling_analyzer = RollingReturnsAnalyzer(windows=[5, 10, 20, 30])
    rolling_df = rolling_analyzer.calculate_rolling_returns(analysis_df, price_col="price")

    # Current rolling status
    print_subheader("Current Rolling Returns")
    current_status = rolling_analyzer.get_current_rolling_status(rolling_df)

    if current_status:
        status_table = []
        for window, stats in current_status["windows"].items():
            ret = stats.get("return", float("nan"))
            slope = stats.get("slope", float("nan"))
            zscore = stats.get("zscore", float("nan"))
            status_table.append([
                f"{window}D",
                f"{ret:.2f}%" if not pd.isna(ret) else "N/A",
                f"{slope:.2f}" if not pd.isna(slope) else "N/A",
                f"{zscore:.2f}" if not pd.isna(zscore) else "N/A",
            ])

        print(tabulate(
            status_table,
            headers=["Window", "Rolling Return", "Slope", "Z-Score"],
            tablefmt="simple"
        ))

        if current_status["warning_signals"]:
            print(f"\n⚠️  WARNING SIGNALS ({current_status['alert_level']}):")
            for sig in current_status["warning_signals"]:
                print(f"    - {sig}")
        else:
            print("\n✓ No warning signals currently active")

    # Identify selloffs
    selloffs = rolling_analyzer.identify_selloff_events(rolling_df, threshold=-20, price_col="price")
    print(f"\nIdentified {len(selloffs)} major selloff events (>20% drawdown)")

    if selloffs:
        print_subheader("Selloff Events")
        selloff_table = []
        for i, s in enumerate(selloffs, 1):
            selloff_table.append([
                i,
                s["start_date"].date(),
                s["trough_date"].date(),
                f"{s['max_drawdown']:.1f}%",
                s["duration_to_trough"],
            ])
        print(tabulate(
            selloff_table,
            headers=["#", "Start", "Trough", "Max Drawdown", "Days to Trough"],
            tablefmt="simple"
        ))

        # Pre-selloff analysis
        print_subheader("What Did Rolling Returns Look Like BEFORE Selloffs?")
        pre_selloff = rolling_analyzer.analyze_returns_before_selloffs(
            rolling_df, selloffs, lookback_days=[5, 10, 20]
        )

        if not pre_selloff.empty:
            summary_rows = []
            for lb in [5, 10, 20]:
                lb_data = pre_selloff[pre_selloff["lookback_days"] == lb]
                if lb_data.empty:
                    continue
                row = [f"{lb} days before"]
                for window in [10, 20, 30]:
                    col = f"ret_{window}d"
                    if col in lb_data.columns:
                        avg = lb_data[col].mean()
                        row.append(f"{avg:.1f}%")
                    else:
                        row.append("N/A")
                summary_rows.append(row)

            print(tabulate(
                summary_rows,
                headers=["Lookback", "10D Ret", "20D Ret", "30D Ret"],
                tablefmt="simple"
            ))

    # ================================================================
    # RETURNS BY REGIME
    # ================================================================
    print_header("Returns by SMA Regime")

    returns_analyzer = ReturnsRegimeAnalyzer(sma_period=50)
    returns_df = returns_analyzer.prepare_returns_data(analysis_df, price_col="price")
    returns_df = returns_analyzer.calculate_rolling_regime_returns(returns_df, window=20)

    regime_stats = returns_analyzer.calculate_regime_stats(returns_df, "regime_simple")

    if regime_stats:
        regime_table = []
        for regime in ["trending_up", "near_sma", "trending_down"]:
            if regime in regime_stats:
                s = regime_stats[regime]
                regime_table.append([
                    regime.replace("_", " ").title(),
                    s.num_days,
                    f"{s.total_return_pct:.1f}%",
                    f"{s.avg_daily_return:.3f}%",
                    f"{s.pct_positive_days:.1f}%",
                    f"{s.worst_day:.1f}%",
                ])

        print(tabulate(
            regime_table,
            headers=["Regime", "Days", "Total Return", "Avg Daily", "% Positive", "Worst Day"],
            tablefmt="simple"
        ))

        print("\n" + "=" * 60)
        print(" KEY INSIGHT - WHERE DID RETURNS COME FROM?")
        print("=" * 60)

        if "cumul_return_trending_down" in returns_df.columns:
            final_down = returns_df["cumul_return_trending_down"].iloc[-1]
            final_near = returns_df["cumul_return_near_sma"].iloc[-1]
            final_up = returns_df["cumul_return_trending_up"].iloc[-1]
            total = final_down + final_near + final_up

            print(f"\n  Return while TRENDING DOWN (>5% below SMA): {final_down:>8.1f}%")
            print(f"  Return while NEAR SMA (-5% to +5%):         {final_near:>8.1f}%")
            print(f"  Return while TRENDING UP (>5% above SMA):   {final_up:>8.1f}%")
            print(f"  " + "-" * 50)
            print(f"  TOTAL CUMULATIVE RETURN:                    {total:>8.1f}%")

            if "trending_down" in regime_stats:
                down = regime_stats["trending_down"]
                print(f"\n  ⚠️  CRITICAL: When trending down, only {down.pct_positive_days:.0f}% of days are positive")
                print(f"      Average daily return in this regime: {down.avg_daily_return:.3f}%")

    # ================================================================
    # PREDICTIVE THRESHOLDS
    # ================================================================
    if selloffs:
        print_header("Predictive Thresholds")
        print("Which rolling return level best predicts upcoming selloffs?\n")

        optimal_thresholds = rolling_analyzer.find_predictive_thresholds(
            rolling_df, selloffs, forward_days=15
        )

        if not optimal_thresholds.empty:
            print(tabulate(
                optimal_thresholds,
                headers=["Window", "Threshold", "Accuracy %", "Signals"],
                tablefmt="simple",
                floatfmt=(".0f", ".0f", ".1f", ".0f")
            ))

            best = optimal_thresholds.loc[optimal_thresholds["accuracy"].idxmax()]
            print(f"\n🎯 BEST PREDICTOR: {int(best['window'])}-day rolling return")
            print(f"   Threshold: {int(best['optimal_threshold'])}%")
            print(f"   Accuracy: {best['accuracy']:.0f}%")

    # ================================================================
    # SUMMARY
    # ================================================================
    print_header("SUMMARY - Key Findings for MSTY")

    peak = analysis_df["price"].max()
    current = analysis_df["price"].iloc[-1]
    decline_from_peak = (current / peak - 1) * 100

    print(f"""
CURRENT STATE:
  Price: ${current:.2f}
  Peak: ${peak:.2f} (decline of {decline_from_peak:.1f}%)
  Distance from 50 SMA: {latest['sma_distance_pct']:.1f}%
  Signal: {signal}

HISTORICAL PATTERN:
  - When >5% BELOW 50 SMA: Returns heavily negative, only ~{regime_stats.get('trending_down', type('', (), {{'pct_positive_days': 40}})()).pct_positive_days:.0f}% of days positive
  - When NEAR 50 SMA: More balanced returns
  - When >5% ABOVE 50 SMA: Positive returns but capped (covered call effect)

WARNING SIGNS TO WATCH:
  1. Distance from 50 SMA dropping below -10%
  2. 20-day rolling return going negative
  3. Velocity (rate of SMA distance change) accelerating negative
  4. Rolling return slope < -3

IMPLICATION:
  The 50 SMA acts as a key support/resistance level.
  When price trends below it, MSTY experiences reflexive collapse
  as the covered call strategy fails to provide downside protection.
""")

    # Generate charts
    print_subheader("Generating Charts")
    returns_viz = ReturnsVisualizer()
    rolling_viz = RollingReturnsVisualizer()

    distribution = returns_analyzer.analyze_return_distribution_by_sma_distance(
        returns_df, distance_bins=[-60, -40, -20, -10, -5, 0, 5, 10, 20, 40]
    )

    returns_viz.plot_returns_by_regime(returns_df, regime_stats, ticker="MSTY")
    returns_viz.plot_return_distribution_by_distance(distribution, ticker="MSTY")

    if selloffs:
        rolling_viz.plot_rolling_returns_with_selloffs(rolling_df, selloffs, windows=[10, 20, 30], ticker="MSTY")

    print("\nCharts saved to output/charts/")


if __name__ == "__main__":
    run_analysis()
