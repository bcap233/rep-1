#!/usr/bin/env python3
"""
MSTY vs WNTR — Dividend Growth/Decline Analysis.

Compares dividend trajectories on a weekly basis to reveal the reflexive
unwind: MSTY's dividends shrink as its equity erodes, while WNTR's
dividends grow as its inverse position benefits.

Key handling:
- MSTY 1:5 split on Dec 8, 2025: normalizes to per-original-share basis
- Oct 2025 frequency shift to weekly: compares on weekly timeframe
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dividend_analysis import DividendAnalyzer, DividendVisualizer


def print_header(title: str):
    print(f"\n{'=' * 75}")
    print(f"  {title}")
    print(f"{'=' * 75}\n")


def print_subheader(title: str):
    print(f"\n--- {title} ---\n")


def main():
    print_header("MSTY vs WNTR — DIVIDEND GROWTH/DECLINE ANALYSIS")
    print("Tracking dividend trajectories to identify the reflexive unwind.")
    print("MSTY dividends normalized to per-original-share (split-adjusted).")
    print("Comparison on weekly timeframe since Oct 2025 frequency shift.\n")

    # ================================================================
    # LOAD DATA
    # ================================================================
    print_subheader("Loading Data")

    msty_csv = Path("msty_data_with_dividends.csv")
    wntr_csv = Path("wntr_data_with_dividends.csv")

    if not msty_csv.exists():
        print("ERROR: msty_data_with_dividends.csv not found.")
        return
    if not wntr_csv.exists():
        print("ERROR: wntr_data_with_dividends.csv not found.")
        return

    msty_df = pd.read_csv(msty_csv, parse_dates=["Date"], index_col="Date")
    msty_df = msty_df.sort_index()
    msty_df.columns = [c.lower().strip() for c in msty_df.columns]

    wntr_df = pd.read_csv(wntr_csv, parse_dates=["Date"], index_col="Date")
    wntr_df = wntr_df.sort_index()
    wntr_df.columns = [c.lower().strip() for c in wntr_df.columns]

    print(f"MSTY: {msty_df.index[0].date()} to {msty_df.index[-1].date()} ({len(msty_df)} days)")
    print(f"  Dividends: {(msty_df['dividend'] > 0).sum()} payments, ${msty_df['dividend'].sum():.2f} total")
    print(f"  Split: 1:5 on 2025-12-08")

    print(f"\nWNTR: {wntr_df.index[0].date()} to {wntr_df.index[-1].date()} ({len(wntr_df)} days)")
    print(f"  Dividends: {(wntr_df['dividend'] > 0).sum()} payments, ${wntr_df['dividend'].sum():.2f} total")

    # ================================================================
    # EXTRACT DIVIDENDS
    # ================================================================
    analyzer = DividendAnalyzer()

    print_header("INDIVIDUAL DIVIDEND PAYMENTS — SPLIT-ADJUSTED")

    msty_divs = analyzer.extract_dividends(msty_df)
    wntr_divs = analyzer.extract_dividends(wntr_df)

    print("MSTY Dividends (per original share, split-adjusted):")
    print(f"{'Date':>12}  {'Raw':>8}  {'Split-Adj':>10}  {'Price':>8}  {'Yield':>7}  {'Split':>6}")
    print("-" * 60)
    for idx, row in msty_divs.iterrows():
        marker = " *" if row["cumulative_split"] > 1 else ""
        print(
            f"{idx.date()!s:>12}  ${row['raw_amount']:>6.3f}  "
            f"${row['split_adjusted_amount']:>8.3f}  "
            f"${row['price']:>6.2f}  "
            f"{row['yield_pct']:>5.2f}%  "
            f"{row['cumulative_split']:>4.0f}x{marker}"
        )

    print(f"\n  * = post-split (raw amount x{msty_divs['cumulative_split'].max():.0f} = split-adjusted)")
    print(f"  Total split-adjusted dividends: ${msty_divs['split_adjusted_amount'].sum():.2f}")

    print(f"\nWNTR Dividends:")
    print(f"{'Date':>12}  {'Amount':>8}  {'Price':>8}  {'Yield':>7}")
    print("-" * 42)
    for idx, row in wntr_divs.iterrows():
        print(
            f"{idx.date()!s:>12}  ${row['raw_amount']:>6.3f}  "
            f"${row['price']:>6.2f}  "
            f"{row['yield_pct']:>5.2f}%"
        )
    print(f"\n  Total dividends: ${wntr_divs['split_adjusted_amount'].sum():.2f}")

    # ================================================================
    # FREQUENCY SHIFT ANALYSIS
    # ================================================================
    print_header("FREQUENCY SHIFT — MONTHLY TO WEEKLY")

    msty_freq = analyzer.analyze_frequency_shift(msty_divs, "MSTY")
    wntr_freq = analyzer.analyze_frequency_shift(wntr_divs, "WNTR")

    freq_combined = pd.concat([msty_freq, wntr_freq], ignore_index=True)
    if not freq_combined.empty:
        print(tabulate(
            freq_combined[[
                "instrument", "frequency", "period", "num_payments",
                "avg_amount_split_adj", "total_amount_split_adj",
                "avg_yield_pct", "avg_gap_days",
            ]].round(3),
            headers=[
                "Instrument", "Frequency", "Period", "# Payments",
                "Avg Div ($)", "Total Div ($)", "Avg Yield %", "Avg Gap (days)",
            ],
            tablefmt="simple",
            floatfmt=(".0f", "", "", ".0f", ".3f", ".2f", ".2f", ".1f"),
        ))

    # ================================================================
    # WEEKLY COMPARISON
    # ================================================================
    print_header("WEEKLY DIVIDEND COMPARISON — SIDE BY SIDE")

    weekly = analyzer.build_weekly_comparison(msty_df, wntr_df)
    weekly = analyzer.calculate_dividend_growth(weekly)

    # Show only weeks where at least one instrument paid
    has_payment = (weekly["msty_weekly_div"] > 0) | (weekly["wntr_weekly_div"] > 0)
    weekly_paid = weekly[has_payment].copy()

    print(f"{'Week Start':>12}  {'MSTY Div':>9}  {'WNTR Div':>9}  {'Spread':>8}  {'Ratio':>7}  {'MSTY Chg':>9}  {'WNTR Chg':>9}")
    print("-" * 78)
    for idx, row in weekly_paid.iterrows():
        msty_div = row["msty_weekly_div"]
        wntr_div = row["wntr_weekly_div"]
        spread = row.get("div_spread", np.nan)
        ratio = row.get("div_ratio", np.nan)
        msty_g = row.get("msty_wow_growth", np.nan)
        wntr_g = row.get("wntr_wow_growth", np.nan)

        spread_str = f"${spread:>+6.2f}" if not pd.isna(spread) else "    n/a"
        ratio_str = f"{ratio:>5.2f}x" if not pd.isna(ratio) else "   n/a"
        msty_g_str = f"{msty_g:>+7.1f}%" if not pd.isna(msty_g) else "     n/a"
        wntr_g_str = f"{wntr_g:>+7.1f}%" if not pd.isna(wntr_g) else "     n/a"

        print(
            f"{idx.date()!s:>12}  "
            f"${msty_div:>7.3f}  "
            f"${wntr_div:>7.3f}  "
            f"{spread_str}  "
            f"{ratio_str}  "
            f"{msty_g_str}  "
            f"{wntr_g_str}"
        )

    # ================================================================
    # WEEKLY PERIOD TREND ANALYSIS
    # ================================================================
    print_header("WEEKLY DIVIDEND PERIOD — TREND ANALYSIS (Oct 2025+)")

    trends = analyzer.analyze_weekly_period_trends(weekly)

    for instrument, data in trends.items():
        name = instrument.upper()
        print(f"  {name} Weekly Dividend Trend:")
        print(f"    Payments:         {data['num_weekly_payments']}")
        print(f"    First payment:    ${data['first_payment']:.3f}")
        print(f"    Last payment:     ${data['last_payment']:.3f}")
        print(f"    Change:           {data['change_pct']:+.1f}%")
        print(f"    Average:          ${data['avg_amount']:.3f}")
        print(f"    Min / Max:        ${data['min_payment']:.3f} / ${data['max_payment']:.3f}")
        print(f"    Trend direction:  {data['trend_direction'].upper()}")
        print(f"    Trend slope:      {data['trend_slope']:+.4f} $/week")
        print(f"    1st half avg:     ${data['first_half_avg']:.3f}")
        print(f"    2nd half avg:     ${data['second_half_avg']:.3f}")
        print(f"    Half-over-half:   {data['half_change_pct']:+.1f}%")
        print()

    # ================================================================
    # KEY FINDINGS
    # ================================================================
    print_header("KEY FINDINGS — DIVIDEND DIVERGENCE")

    msty_trend = trends.get("msty", {})
    wntr_trend = trends.get("wntr", {})

    if msty_trend and wntr_trend:
        print("  THE REFLEXIVE UNWIND IN DIVIDENDS:")
        print()
        print(f"    MSTY weekly dividends are {msty_trend['trend_direction'].upper()}")
        print(f"      From ${msty_trend['first_payment']:.3f} to ${msty_trend['last_payment']:.3f} ({msty_trend['change_pct']:+.1f}%)")
        print(f"      Trend slope: {msty_trend['trend_slope']:+.4f} $/week")
        print()
        print(f"    WNTR weekly dividends are {wntr_trend['trend_direction'].upper()}")
        print(f"      From ${wntr_trend['first_payment']:.3f} to ${wntr_trend['last_payment']:.3f} ({wntr_trend['change_pct']:+.1f}%)")
        print(f"      Trend slope: {wntr_trend['trend_slope']:+.4f} $/week")
        print()
        print("  WHY THIS MATTERS:")
        print("    MSTY sells covered calls on MSTR. As MSTR declines:")
        print("      -> Call premiums shrink (less volatility value to harvest)")
        print("      -> MSTY's NAV erodes (underlying asset declining)")
        print("      -> Dividends decline BECAUSE of both factors")
        print("      -> This confirms the reflexive collapse: equity down -> income down")
        print()
        print("    WNTR holds inverse/put positions on MSTR. As MSTR declines:")
        print("      -> Inverse position gains value")
        print("      -> Higher NAV supports larger distributions")
        print("      -> Dividends can grow as the position profits")
        print()

        # Compute the crossover analysis
        weekly_oct = weekly.loc["2025-10-01":]
        both = weekly_oct[(weekly_oct["msty_weekly_div"] > 0) & (weekly_oct["wntr_weekly_div"] > 0)]
        if not both.empty:
            wntr_larger = (both["wntr_weekly_div"] > both["msty_weekly_div"]).sum()
            total_both = len(both)
            print(f"  WNTR DIVIDEND > MSTY DIVIDEND (split-adjusted):")
            print(f"    {wntr_larger} out of {total_both} weeks ({wntr_larger / total_both * 100:.0f}%)")

            if "div_ratio" in both.columns:
                avg_ratio = both["div_ratio"].mean()
                latest_ratio = both["div_ratio"].iloc[-1]
                print(f"    Average WNTR/MSTY ratio: {avg_ratio:.2f}x")
                print(f"    Latest WNTR/MSTY ratio:  {latest_ratio:.2f}x")
        print()

    # ================================================================
    # CHARTS
    # ================================================================
    print_header("GENERATING CHARTS")

    viz = DividendVisualizer()

    viz.plot_weekly_dividends_comparison(weekly, msty_divs, wntr_divs)
    viz.plot_weekly_period_deep_dive(weekly, msty_divs, wntr_divs)
    viz.plot_dividend_vs_price(msty_df, wntr_df, msty_divs, wntr_divs)

    print("\nCharts saved to output/charts/")
    print("  - dividend_growth_comparison.png")
    print("  - dividend_weekly_deep_dive.png")
    print("  - dividend_vs_price.png")


if __name__ == "__main__":
    main()
