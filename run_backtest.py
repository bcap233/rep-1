#!/usr/bin/env python3
"""
Main backtest runner for Yield Max / Covered Call ETF analysis.

This script analyzes the relationship between SMAs and yield max/covered call
products, particularly focusing on when these products enter steep declines.

Usage:
    python run_backtest.py [--tickers QYLD,XYLD] [--start 2020-01-01] [--end 2024-12-31]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    YIELD_MAX_TICKERS,
    UNDERLYING_TICKERS,
    SMA_PERIODS,
    DECLINE_THRESHOLDS,
    DEFAULT_START_DATE,
    OUTPUT_DIR,
)
from src.data_fetcher import DataFetcher
from src.indicators import TechnicalIndicators
from src.analysis import DeclineAnalyzer, SMARelationshipAnalyzer
from src.visualization import BacktestVisualizer


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60 + "\n")


def run_analysis(
    tickers: list = None,
    start_date: str = None,
    end_date: str = None,
    generate_charts: bool = True,
):
    """
    Run the full backtest analysis.

    Args:
        tickers: List of yield max tickers to analyze (default: all)
        start_date: Start date for analysis
        end_date: End date for analysis
        generate_charts: Whether to generate visualization charts
    """
    # Setup
    start_date = start_date or DEFAULT_START_DATE
    tickers = tickers or list(YIELD_MAX_TICKERS.keys())

    print_section("Yield Max / Covered Call SMA Analysis")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Period: {start_date} to {end_date or 'present'}")
    print(f"SMA Periods: {SMA_PERIODS}")

    # Initialize components
    fetcher = DataFetcher()
    decline_analyzer = DeclineAnalyzer(DECLINE_THRESHOLDS)
    sma_analyzer = SMARelationshipAnalyzer(SMA_PERIODS)
    visualizer = BacktestVisualizer()

    # Filter to requested tickers
    yield_max_subset = {k: v for k, v in YIELD_MAX_TICKERS.items() if k in tickers}
    underlying_subset = {k: v for k, v in UNDERLYING_TICKERS.items() if k in tickers}

    # Fetch data
    print_section("Fetching Data from Yahoo Finance")
    data = fetcher.fetch_yield_max_with_underlying(
        yield_max_subset, underlying_subset, start_date, end_date
    )

    if not data:
        print("ERROR: No data fetched. Check ticker symbols and date range.")
        return

    # Store all processed DataFrames
    all_data = {}
    all_comparisons = []
    all_decline_summaries = []
    all_sma_condition_analyses = []

    # Process each ticker
    for ym_ticker, ticker_data in data.items():
        print_section(f"Analyzing {ym_ticker} ({YIELD_MAX_TICKERS.get(ym_ticker, '')})")

        etf_df = ticker_data["etf"]
        underlying_df = ticker_data["underlying"]
        underlying_ticker = ticker_data["underlying_ticker"]

        print(f"ETF data: {len(etf_df)} days ({etf_df.index[0].date()} to {etf_df.index[-1].date()})")
        print(f"Underlying ({underlying_ticker}): {len(underlying_df)} days")

        # Add all indicators
        etf_df = TechnicalIndicators.add_all_indicators(etf_df, SMA_PERIODS)
        underlying_df = TechnicalIndicators.add_all_indicators(underlying_df, SMA_PERIODS)

        all_data[ym_ticker] = etf_df
        all_data[underlying_ticker] = underlying_df

        # ============================================================
        # 1. DECLINE ANALYSIS
        # ============================================================
        print("\n--- Decline Analysis ---")

        decline_summary = decline_analyzer.analyze_declines_summary(etf_df, ym_ticker)
        if not decline_summary.empty:
            all_decline_summaries.append(decline_summary)
            print(tabulate(decline_summary, headers="keys", tablefmt="simple", floatfmt=".2f"))

        # Identify steep declines
        steep_declines = decline_analyzer.identify_decline_periods(etf_df, threshold=-0.15)
        print(f"\nSteep decline periods (>15%): {len(steep_declines)}")

        for i, decline in enumerate(steep_declines[:5], 1):  # Show first 5
            print(f"  {i}. {decline.start_date.date()} to {decline.end_date.date()}: "
                  f"{decline.max_drawdown:.1f}% over {decline.duration_days} days")

        # ============================================================
        # 2. SMA RELATIONSHIP ANALYSIS
        # ============================================================
        print("\n--- SMA Position Analysis ---")

        for sma_period in [50, 200]:
            print(f"\nSMA {sma_period} Analysis:")
            sma_stats = sma_analyzer.analyze_forward_returns_by_sma_position(
                etf_df, sma_period, [5, 10, 20, 60]
            )

            above = sma_stats.get("above", {})
            below = sma_stats.get("below", {})

            print(f"  Days above SMA{sma_period}: {above.get('observations', 0)}")
            print(f"  Days below SMA{sma_period}: {below.get('observations', 0)}")

            if above.get("fwd_20d_mean") is not None and below.get("fwd_20d_mean") is not None:
                print(f"  Avg 20-day forward return when ABOVE: {above['fwd_20d_mean']:.2f}%")
                print(f"  Avg 20-day forward return when BELOW: {below['fwd_20d_mean']:.2f}%")
                print(f"  Win rate ABOVE: {above.get('fwd_20d_positive_pct', 0):.1f}%")
                print(f"  Win rate BELOW: {below.get('fwd_20d_positive_pct', 0):.1f}%")

            # Visualize
            if generate_charts:
                visualizer.plot_sma_position_returns(sma_stats, ym_ticker, sma_period)

        # ============================================================
        # 3. SMA CONDITIONS AT DECLINE ENTRY
        # ============================================================
        print("\n--- SMA Conditions at Decline Entry Points ---")

        if steep_declines:
            sma_conditions = sma_analyzer.analyze_sma_conditions_during_declines(
                etf_df, steep_declines
            )

            if not sma_conditions.empty:
                all_sma_condition_analyses.append(sma_conditions)

                # Summary statistics
                for sma_period in [50, 200]:
                    col = f"above_sma_{sma_period}"
                    if col in sma_conditions.columns:
                        above_pct = sma_conditions[col].mean() * 100
                        print(f"  % of declines starting above SMA{sma_period}: {above_pct:.1f}%")

                    dist_col = f"dist_sma_{sma_period}_pct"
                    if dist_col in sma_conditions.columns:
                        avg_dist = sma_conditions[dist_col].mean()
                        print(f"  Avg distance from SMA{sma_period} at decline start: {avg_dist:.2f}%")

        # ============================================================
        # 4. ETF VS UNDERLYING COMPARISON
        # ============================================================
        print("\n--- ETF vs Underlying Comparison ---")

        comparison = sma_analyzer.compare_etf_vs_underlying(
            etf_df, underlying_df, ym_ticker, underlying_ticker
        )

        if not comparison.empty:
            all_comparisons.append(comparison)

            # Key metrics
            all_period = comparison[comparison["condition"] == "all_periods"]
            if not all_period.empty:
                row = all_period.iloc[0]
                print(f"  Total Period ({row['num_days']} days):")
                print(f"    {ym_ticker} return: {row['etf_total_return']*100:.2f}%")
                print(f"    {underlying_ticker} return: {row['underlying_total_return']*100:.2f}%")
                print(f"    Capture ratio: {row['capture_ratio']:.2f}")
                print(f"    Correlation: {row['correlation']:.3f}")

            # During drawdowns
            dd_condition = comparison[comparison["condition"] == "drawdown_gt_10pct"]
            if not dd_condition.empty:
                row = dd_condition.iloc[0]
                print(f"\n  During >10% Drawdowns ({row['num_days']} days):")
                print(f"    {ym_ticker} return: {row['etf_total_return']*100:.2f}%")
                print(f"    {underlying_ticker} return: {row['underlying_total_return']*100:.2f}%")
                print(f"    Capture ratio: {row['capture_ratio']:.2f}")

        # ============================================================
        # 5. GENERATE CHARTS
        # ============================================================
        if generate_charts:
            print("\n--- Generating Charts ---")
            visualizer.plot_price_with_smas(etf_df, ym_ticker, [20, 50, 200])
            visualizer.plot_sma_distance_analysis(etf_df, ym_ticker, [50, 200])
            visualizer.plot_etf_vs_underlying_comparison(
                etf_df, underlying_df, ym_ticker, underlying_ticker
            )
            if steep_declines:
                visualizer.plot_decline_analysis(etf_df, steep_declines, ym_ticker)
            visualizer.plot_market_condition_comparison(comparison)

    # ============================================================
    # SUMMARY ACROSS ALL TICKERS
    # ============================================================
    print_section("Cross-Ticker Summary")

    if all_decline_summaries:
        combined_declines = pd.concat(all_decline_summaries, ignore_index=True)
        print("Decline Summary (All Tickers):")
        print(tabulate(
            combined_declines[["ticker", "severity", "num_occurrences", "avg_drawdown_pct", "avg_duration_days"]],
            headers="keys", tablefmt="simple", floatfmt=".2f"
        ))

    if all_comparisons:
        combined_comparisons = pd.concat(all_comparisons, ignore_index=True)

        # Pivot for easier reading
        all_periods = combined_comparisons[combined_comparisons["condition"] == "all_periods"]
        if not all_periods.empty:
            print("\nOverall Performance Comparison:")
            summary_table = all_periods[["etf_ticker", "underlying_ticker",
                                          "etf_total_return", "underlying_total_return",
                                          "capture_ratio", "correlation"]].copy()
            summary_table["etf_total_return"] = summary_table["etf_total_return"] * 100
            summary_table["underlying_total_return"] = summary_table["underlying_total_return"] * 100
            print(tabulate(summary_table, headers="keys", tablefmt="simple", floatfmt=".2f"))

    # Generate summary dashboard
    if generate_charts and all_data:
        print("\n--- Generating Summary Dashboard ---")
        visualizer.create_summary_dashboard(all_data, {})

    print_section("Analysis Complete")
    print(f"Charts saved to: {OUTPUT_DIR}/charts/")
    print("\nKey Insights:")
    print("1. Check SMA distance charts to see how far from SMAs declines typically start")
    print("2. Market condition charts show ETF vs underlying performance in different regimes")
    print("3. Decline analysis shows severity and duration of historical drawdowns")
    print("4. SMA position returns show predictive value of being above/below key SMAs")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Backtest analysis for yield max / covered call ETFs"
    )
    parser.add_argument(
        "--tickers", "-t",
        type=str,
        default=None,
        help="Comma-separated list of tickers to analyze (default: all configured)"
    )
    parser.add_argument(
        "--start", "-s",
        type=str,
        default=DEFAULT_START_DATE,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START_DATE})"
    )
    parser.add_argument(
        "--end", "-e",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation"
    )

    args = parser.parse_args()

    tickers = args.tickers.split(",") if args.tickers else None

    run_analysis(
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        generate_charts=not args.no_charts,
    )


if __name__ == "__main__":
    main()
