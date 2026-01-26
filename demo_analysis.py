#!/usr/bin/env python3
"""
Demo analysis with synthetic data mimicking MSTY behavior.
Run this to see exactly what the analysis output looks like.
"""

import sys
from datetime import datetime, timedelta
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


def generate_msty_like_data(days: int = 365) -> tuple:
    """
    Generate synthetic data that mimics MSTY's behavior:
    - High volatility (tied to MSTR/Bitcoin)
    - Periods of stability near 50 SMA
    - Sharp selloffs when below 50 SMA
    - Capped upside (covered call effect)
    """
    np.random.seed(42)  # Reproducible

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Generate MSTR-like underlying (high volatility, trending)
    mstr_returns = np.random.normal(0.001, 0.04, days)  # 4% daily vol

    # Add some trending periods
    mstr_returns[50:100] = np.random.normal(0.015, 0.03, 50)   # Bull run
    mstr_returns[150:200] = np.random.normal(-0.02, 0.05, 50)  # Crash
    mstr_returns[250:280] = np.random.normal(-0.015, 0.04, 30) # Another selloff
    mstr_returns[320:350] = np.random.normal(0.012, 0.03, 30)  # Recovery

    mstr_price = 100 * np.cumprod(1 + mstr_returns)

    # Generate MSTY (covered call on MSTR)
    # - Captures ~70% of upside (capped by calls)
    # - Captures ~90% of downside (some premium cushion)
    # - Has high yield (simulated as small daily boost when flat)
    msty_returns = np.where(
        mstr_returns > 0.02,
        mstr_returns * 0.5 + 0.003,  # Heavily capped on big up days
        np.where(
            mstr_returns > 0,
            mstr_returns * 0.7 + 0.002,  # Moderately capped on small up days
            mstr_returns * 0.95  # Nearly full downside exposure
        )
    )

    # Add some income effect on flat days
    msty_returns = np.where(
        abs(mstr_returns) < 0.01,
        msty_returns + 0.001,  # Small daily income
        msty_returns
    )

    msty_price = 25 * np.cumprod(1 + msty_returns)

    msty_df = pd.DataFrame({
        'open': msty_price * (1 + np.random.normal(0, 0.005, days)),
        'high': msty_price * (1 + abs(np.random.normal(0.01, 0.01, days))),
        'low': msty_price * (1 - abs(np.random.normal(0.01, 0.01, days))),
        'close': msty_price,
        'volume': np.random.randint(1000000, 5000000, days),
    }, index=dates)

    mstr_df = pd.DataFrame({
        'open': mstr_price * (1 + np.random.normal(0, 0.005, days)),
        'high': mstr_price * (1 + abs(np.random.normal(0.01, 0.01, days))),
        'low': mstr_price * (1 - abs(np.random.normal(0.01, 0.01, days))),
        'close': mstr_price,
        'volume': np.random.randint(5000000, 20000000, days),
    }, index=dates)

    return msty_df, mstr_df


def run_demo():
    print_header("MSTY Collapse Detection Analysis (DEMO DATA)")
    print("Using synthetic data that mimics MSTY/MSTR behavior")
    print("Run with real data locally: python analyze_msty.py")

    # Generate data
    print_subheader("Generating Synthetic Data")
    msty_df, mstr_df = generate_msty_like_data(365)
    print(f"MSTY: {len(msty_df)} days ({msty_df.index[0].date()} to {msty_df.index[-1].date()})")
    print(f"MSTR: {len(mstr_df)} days")

    # Initialize detector
    detector = CollapseDetector(sma_period=50)
    df = detector.prepare_data(msty_df, mstr_df)

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

        latest = df.iloc[-1]
        print(f"\nCurrent Price: ${latest['price']:.2f}")
        print(f"50 SMA: ${latest['sma_50']:.2f}")

    # ================================================================
    # OPTIMAL THRESHOLD
    # ================================================================
    print_header("Optimal Threshold Analysis")

    optimal = detector.find_optimal_threshold(df, target_accuracy=0.7)

    print(f"\nOPTIMAL THRESHOLD: {optimal['optimal_threshold']}% below 50 SMA")
    print(f"Accuracy: {optimal['accuracy'] * 100:.1f}% of crosses led to further decline")
    print(f"Number of signals in period: {optimal['num_signals']}")
    print(f"Average further decline: {optimal['avg_further_decline']:.2f}%")

    print("\nAll Threshold Results:")
    all_results = optimal["all_results"].sort_values("threshold", ascending=False)
    print(tabulate(
        all_results[["threshold", "num_signals", "accuracy", "avg_further_decline"]].head(15),
        headers=["Threshold %", "Signals", "Accuracy", "Avg Decline %"],
        tablefmt="simple",
        floatfmt=(".0f", ".0f", ".2f", ".2f")
    ))

    # ================================================================
    # HISTORICAL COLLAPSES
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
                "YES" if c.was_true_collapse else "no",
            ])

        print(tabulate(
            collapse_data,
            headers=["#", "Start", "End", "Trigger Dist", "Max Dist", "Max DD", "Days", "True?"],
            tablefmt="simple"
        ))

    # ================================================================
    # RETURNS BY SMA REGIME
    # ================================================================
    print_header("Returns Analysis by SMA Regime")

    returns_analyzer = ReturnsRegimeAnalyzer(sma_period=50)
    returns_viz = ReturnsVisualizer()

    returns_df = returns_analyzer.prepare_returns_data(df, price_col="price")
    returns_df = returns_analyzer.calculate_rolling_regime_returns(returns_df, window=20)

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

        print("\n" + "=" * 50)
        print(" KEY FINDING - RETURN ASYMMETRY")
        print("=" * 50)
        if "trending_down" in regime_stats and "trending_up" in regime_stats:
            down = regime_stats["trending_down"]
            up = regime_stats["trending_up"]
            near = regime_stats.get("near_sma")

            print(f"\n  TRENDING DOWN (>5% below SMA): {down.num_days} days")
            print(f"    Total return: {down.total_return_pct:.2f}%")
            print(f"    Only {down.pct_positive_days:.1f}% of days positive")
            print(f"    Skewness: {down.skewness:.2f} (negative = fat left tail)")

            print(f"\n  NEAR SMA (-5% to +5%): {near.num_days if near else 0} days")
            if near:
                print(f"    Total return: {near.total_return_pct:.2f}%")
                print(f"    {near.pct_positive_days:.1f}% of days positive")

            print(f"\n  TRENDING UP (>5% above SMA): {up.num_days} days")
            print(f"    Total return: {up.total_return_pct:.2f}%")
            print(f"    {up.pct_positive_days:.1f}% of days positive")

    # Cumulative return breakdown
    print_subheader("Cumulative Return Breakdown")
    if "cumul_return_trending_down" in returns_df.columns:
        final_down = returns_df["cumul_return_trending_down"].iloc[-1]
        final_near = returns_df["cumul_return_near_sma"].iloc[-1]
        final_up = returns_df["cumul_return_trending_up"].iloc[-1]
        total = final_down + final_near + final_up

        print(f"  Return earned while TRENDING DOWN: {final_down:>10.2f}%")
        print(f"  Return earned while NEAR SMA:      {final_near:>10.2f}%")
        print(f"  Return earned while TRENDING UP:   {final_up:>10.2f}%")
        print(f"  " + "-" * 40)
        print(f"  TOTAL CUMULATIVE RETURN:           {total:>10.2f}%")

    # ================================================================
    # ROLLING RETURNS ANALYSIS
    # ================================================================
    print_header("Rolling Total Returns Analysis")

    rolling_analyzer = RollingReturnsAnalyzer(windows=[5, 10, 20, 30, 60])
    rolling_viz = RollingReturnsVisualizer()

    rolling_df = rolling_analyzer.calculate_rolling_returns(df, price_col="price")

    selloffs = rolling_analyzer.identify_selloff_events(rolling_df, threshold=-15, price_col="price")
    print(f"\nIdentified {len(selloffs)} selloff events (>15% drawdown)")

    if selloffs:
        print_subheader("Selloff Events")
        selloff_table = []
        for i, s in enumerate(selloffs[:10], 1):
            selloff_table.append([
                i,
                s["start_date"].date(),
                s["trough_date"].date(),
                f"{s['max_drawdown']:.1f}%",
                s["duration_to_trough"],
            ])
        print(tabulate(
            selloff_table,
            headers=["#", "Start", "Trough", "Max DD", "Days"],
            tablefmt="simple"
        ))

    # Current rolling status
    print_subheader("Current Rolling Returns Status")
    current_status = rolling_analyzer.get_current_rolling_status(rolling_df)

    if current_status:
        print(f"Date: {current_status['date'].date()}\n")
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

    # Pre-selloff patterns
    if selloffs:
        print_subheader("Rolling Returns BEFORE Selloffs")
        pre_selloff = rolling_analyzer.analyze_returns_before_selloffs(
            rolling_df, selloffs, lookback_days=[5, 10, 20]
        )

        if not pre_selloff.empty:
            print("Average rolling returns X days BEFORE selloff started:\n")
            summary_rows = []
            for lb in [5, 10, 20]:
                lb_data = pre_selloff[pre_selloff["lookback_days"] == lb]
                if lb_data.empty:
                    continue
                row = [f"{lb}d before"]
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

            print("\n" + "=" * 50)
            print(" KEY INSIGHT - PRE-SELLOFF PATTERNS")
            print("=" * 50)
            lb_10 = pre_selloff[pre_selloff["lookback_days"] == 10]
            if not lb_10.empty and "ret_20d" in lb_10.columns:
                avg_ret = lb_10["ret_20d"].mean()
                pct_neg = (lb_10["ret_20d"] < 0).mean() * 100
                print(f"\n  10 days before selloffs:")
                print(f"    - 20D rolling return averaged: {avg_ret:.1f}%")
                print(f"    - Was already negative {pct_neg:.0f}% of the time")
                print(f"\n  IMPLICATION: Monitor when 20D return goes negative!")

    # Predictive thresholds
    if selloffs:
        print_subheader("Predictive Rolling Return Thresholds")
        print("Finding the threshold that best predicts upcoming selloffs...\n")

        optimal_thresholds = rolling_analyzer.find_predictive_thresholds(
            rolling_df, selloffs, forward_days=20
        )

        if not optimal_thresholds.empty:
            print(tabulate(
                optimal_thresholds,
                headers=["Window", "Threshold", "Accuracy %", "Signals"],
                tablefmt="simple",
                floatfmt=(".0f", ".0f", ".1f", ".0f")
            ))

            best = optimal_thresholds.loc[optimal_thresholds["accuracy"].idxmax()]
            print(f"\n" + "=" * 50)
            print(f" 🎯 BEST PREDICTOR")
            print("=" * 50)
            print(f"\n  {int(best['window'])}-day rolling return")
            print(f"  When it drops below {int(best['optimal_threshold'])}%,")
            print(f"  there's a {best['accuracy']:.0f}% chance of selloff within 20 days")

    # Forward returns by level
    print_subheader("Forward Returns by Rolling Return Level")
    level_analysis = rolling_analyzer.analyze_rolling_return_levels(rolling_df, price_col="price")

    if not level_analysis.empty:
        analysis_20d = level_analysis[level_analysis["window"] == 20].copy()
        if not analysis_20d.empty:
            print("When 20-day rolling return is at these levels, what happens next?\n")
            print(tabulate(
                analysis_20d[["rolling_return_range", "num_observations", "avg_fwd_10d", "pct_negative_fwd_10d", "worst_fwd_20d"]],
                headers=["20D Return Level", "# Obs", "Avg 10D Fwd", "% Neg 10D", "Worst 20D"],
                tablefmt="simple",
                floatfmt=("", ".0f", ".2f", ".1f", ".1f")
            ))

            danger = analysis_20d[analysis_20d["avg_fwd_10d"] < -1]
            if not danger.empty:
                worst = danger.loc[danger["avg_fwd_10d"].idxmin()]
                print(f"\n  ⚠️  DANGER ZONE: 20D return in range {worst['rolling_return_range']}")
                print(f"      Avg next 10 days: {worst['avg_fwd_10d']:.2f}%")
                print(f"      {worst['pct_negative_fwd_10d']:.0f}% of time, next 10 days negative")

    # Generate charts
    print_subheader("Generating Charts")
    returns_viz.plot_returns_by_regime(returns_df, regime_stats, ticker="MSTY_DEMO")
    returns_viz.plot_return_distribution_by_distance(
        returns_analyzer.analyze_return_distribution_by_sma_distance(returns_df),
        ticker="MSTY_DEMO"
    )

    if selloffs:
        rolling_viz.plot_rolling_returns_with_selloffs(rolling_df, selloffs, windows=[10, 20, 30], ticker="MSTY_DEMO")
        rolling_viz.plot_pre_selloff_patterns(pre_selloff, ticker="MSTY_DEMO")
        if not optimal_thresholds.empty:
            rolling_viz.plot_predictive_thresholds(optimal_thresholds, ticker="MSTY_DEMO")

    # ================================================================
    # SUMMARY
    # ================================================================
    print_header("SUMMARY - Key Findings")

    print("""
COLLAPSE DETECTION SIGNALS:

1. SMA DISTANCE THRESHOLD:
   - Optimal trigger: {threshold}% below 50 SMA
   - Accuracy: {accuracy:.0f}%

2. ROLLING RETURN THRESHOLDS:
   - Best predictor: {window}D rolling return below {ret_threshold}%
   - Accuracy: {ret_accuracy:.0f}%

3. RETURN REGIME ASYMMETRY:
   - Returns are heavily negative when trending below 50 SMA
   - Returns are capped but more stable near/above 50 SMA
   - Most losses occur in "trending down" regime

4. EARLY WARNING SIGNS:
   - 20D rolling return turns negative 10-20 days before major selloffs
   - Accelerating slope (velocity) confirms collapse
   - Z-score < -2 indicates extreme conditions

RECOMMENDED MONITORING:
   - Daily check of distance from 50 SMA
   - Track 20D rolling return level and slope
   - Alert when collapse probability > 50%
""".format(
        threshold=optimal['optimal_threshold'],
        accuracy=optimal['accuracy'] * 100,
        window=int(best['window']) if not optimal_thresholds.empty else 20,
        ret_threshold=int(best['optimal_threshold']) if not optimal_thresholds.empty else -10,
        ret_accuracy=best['accuracy'] if not optimal_thresholds.empty else 70,
    ))

    print("\nCharts saved to: output/charts/")
    print("Run with real MSTY data: python analyze_msty.py")


if __name__ == "__main__":
    run_demo()
