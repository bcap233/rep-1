#!/usr/bin/env python3
"""
MSTY vs WNTR Counterparty Rolling Total Return Analysis.

Compares MSTY (bullish covered call on MSTR) against WNTR (bearish inverse)
to identify divergence patterns. As MSTY's rolling total returns go negative
on increasingly more timeframes (5d -> 10d -> 20d), WNTR's should go positive —
revealing the reflexive collapse in action.

WNTR DATA NOTE:
    WNTR data in this script is SYNTHETIC — modeled as an inverse instrument
    to MSTY based on MSTY's daily returns. Replace wntr_data_with_dividends.csv
    with real market data when available.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent))

from src.counterparty_analysis import (
    TotalReturnCalculator,
    CounterpartyAnalyzer,
    CounterpartyVisualizer,
)


def print_header(text: str):
    print("\n" + "=" * 75)
    print(f"  {text}")
    print("=" * 75)


def print_subheader(text: str):
    print(f"\n--- {text} ---\n")


def generate_synthetic_wntr(msty_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic WNTR data as the inverse counterparty to MSTY.

    Model assumptions:
    - WNTR is a bearish options strategy on MSTR (inverse of MSTY)
    - When MSTY's underlying (MSTR) goes down, WNTR benefits
    - WNTR collects option premium (dividends) similar to MSTY
    - The inverse beta is not perfect (-0.82) due to options structure differences
    - WNTR has its own premium decay and vol drag

    This is synthetic data for analysis framework development.
    Replace with real WNTR data when available.
    """
    msty = msty_df.sort_index().copy()

    # Calculate MSTY daily price returns
    msty_returns = msty["close"].pct_change().fillna(0)

    # WNTR parameters
    inverse_beta = -0.82       # How strongly WNTR inverses MSTY price moves
    daily_premium = 0.0003     # Daily option premium drift (~7.5% annualized)
    noise_std = 0.005          # Small idiosyncratic noise
    starting_price = 22.50     # WNTR starting price

    np.random.seed(42)  # Reproducible

    # Generate WNTR daily returns
    wntr_returns = (
        msty_returns * inverse_beta
        + daily_premium
        + np.random.normal(0, noise_std, len(msty_returns))
    )

    # Build WNTR price series
    wntr_prices = [starting_price]
    for ret in wntr_returns.iloc[1:]:
        new_price = wntr_prices[-1] * (1 + ret)
        new_price = max(new_price, 0.50)  # Floor
        wntr_prices.append(new_price)

    # Generate WNTR dividends
    # WNTR pays weekly-ish dividends (similar cadence to MSTY)
    # Dividend amount scales with price and recent performance
    dividends = []
    for i, (date, price) in enumerate(zip(msty.index, wntr_prices)):
        # Pay dividend roughly every 7 trading days
        if i > 0 and i % 7 == 0:
            # Base dividend yield ~ 0.8-1.5% of price per payment
            # Higher when WNTR is doing well (MSTY declining)
            recent_return = (wntr_prices[i] / wntr_prices[max(0, i - 7)] - 1)
            base_yield = 0.008 + max(0, recent_return * 0.3)
            div_amount = round(price * base_yield, 3)
            dividends.append(div_amount)
        else:
            dividends.append(0.0)

    wntr_df = pd.DataFrame({
        "close": wntr_prices,
        "dividend": dividends,
        "split": 1,
    }, index=msty.index)

    return wntr_df


def run_analysis():
    print_header("MSTY vs WNTR — COUNTERPARTY DIVERGENCE ANALYSIS")
    print("\nComparing rolling total returns to identify reflexive collapse signals")
    print("When MSTY goes negative on a timeframe, does WNTR go positive?\n")

    # ================================================================
    # LOAD DATA
    # ================================================================
    print_subheader("Loading Data")

    # Load MSTY with dividends
    msty_df = pd.read_csv(
        "msty_data_with_dividends.csv", parse_dates=["Date"], index_col="Date"
    )
    msty_df = msty_df.sort_index()
    msty_df.columns = [c.lower().strip() for c in msty_df.columns]

    print(f"MSTY: {msty_df.index[0].date()} to {msty_df.index[-1].date()} ({len(msty_df)} days)")
    print(f"  Price range: ${msty_df['close'].min():.2f} - ${msty_df['close'].max():.2f}")
    print(f"  Total dividends: ${msty_df['dividend'].sum():.2f}")

    # Generate or load WNTR data
    wntr_csv = Path("wntr_data_with_dividends.csv")
    if wntr_csv.exists():
        wntr_df = pd.read_csv(wntr_csv, parse_dates=["Date"], index_col="Date")
        wntr_df = wntr_df.sort_index()
        wntr_df.columns = [c.lower().strip() for c in wntr_df.columns]
        print("\n  [Loaded WNTR data from wntr_data_with_dividends.csv]")
    else:
        print("\n  [No WNTR CSV found — generating SYNTHETIC inverse data]")
        print("  NOTE: Replace with real WNTR data for production analysis")
        wntr_df = generate_synthetic_wntr(msty_df)

        # Save for inspection
        save_df = wntr_df.copy()
        save_df.index.name = "Date"
        save_df.columns = ["Close", "Dividend", "Split"]
        save_df.to_csv("wntr_data_with_dividends.csv")
        print("  Saved synthetic data to wntr_data_with_dividends.csv")

    print(f"\nWNTR: {wntr_df.index[0].date()} to {wntr_df.index[-1].date()} ({len(wntr_df)} days)")
    print(f"  Price range: ${wntr_df['close'].min():.2f} - ${wntr_df['close'].max():.2f}")
    print(f"  Total dividends: ${wntr_df['dividend'].sum():.2f}")

    # ================================================================
    # TOTAL RETURN INDICES
    # ================================================================
    print_header("TOTAL RETURN COMPARISON")

    calc = TotalReturnCalculator()

    msty_tri = calc.calculate_total_return_index(msty_df)
    wntr_tri = calc.calculate_total_return_index(wntr_df)

    msty_total_return = (msty_tri.iloc[-1] / msty_tri.iloc[0] - 1) * 100
    wntr_total_return = (wntr_tri.iloc[-1] / wntr_tri.iloc[0] - 1) * 100
    msty_price_return = (msty_df["close"].iloc[-1] / msty_df["close"].iloc[0] - 1) * 100
    wntr_price_return = (wntr_df["close"].iloc[-1] / wntr_df["close"].iloc[0] - 1) * 100

    print(f"""
  {'':30s}  {'MSTY':>12s}  {'WNTR':>12s}
  {'—' * 58}
  Starting Price:               ${msty_df['close'].iloc[0]:>10.2f}  ${wntr_df['close'].iloc[0]:>10.2f}
  Ending Price:                 ${msty_df['close'].iloc[-1]:>10.2f}  ${wntr_df['close'].iloc[-1]:>10.2f}
  Price-Only Return:             {msty_price_return:>10.1f}%  {wntr_price_return:>10.1f}%
  Total Dividends Collected:    ${msty_df['dividend'].sum():>10.2f}  ${wntr_df['dividend'].sum():>10.2f}
  Total Return (w/ dividends):   {msty_total_return:>10.1f}%  {wntr_total_return:>10.1f}%
  {'—' * 58}
  Spread (WNTR − MSTY):         {wntr_total_return - msty_total_return:>10.1f}%
""")

    # ================================================================
    # ROLLING TOTAL RETURNS
    # ================================================================
    print_header("ROLLING TOTAL RETURNS — SIDE BY SIDE")

    windows = [5, 10, 20]
    analyzer = CounterpartyAnalyzer(windows=windows)
    comparison = analyzer.prepare_comparison_data(msty_df, wntr_df)

    # Show current rolling returns
    print_subheader("Current Rolling Total Returns")

    current = analyzer.get_current_status(comparison)
    if current:
        print(f"  Date: {current['date'].date()}")
        print(f"  MSTY Price: ${current['msty_price']:.2f}")
        print(f"  WNTR Price: ${current['wntr_price']:.2f}")
        print(f"  Divergence Score: {current['divergence_score']:.2f}")
        print(f"  Timeframes Diverging: {current['divergence_count']} / {len(windows)}\n")

        status_table = []
        for w in windows:
            ws = current["windows"][w]
            msty_r = ws["msty_return"]
            wntr_r = ws["wntr_return"]
            spread = ws["spread"]

            msty_sign = "NEG" if ws["msty_negative"] else "pos"
            wntr_sign = "POS" if ws["wntr_positive"] else "neg"
            div_flag = "DIVERGING" if ws["diverging"] else ""

            status_table.append([
                f"{w}D",
                f"{msty_r:+.2f}%" if msty_r is not None and not np.isnan(msty_r) else "N/A",
                msty_sign,
                f"{wntr_r:+.2f}%" if wntr_r is not None and not np.isnan(wntr_r) else "N/A",
                wntr_sign,
                f"{spread:+.2f}%" if spread is not None and not np.isnan(spread) else "N/A",
                div_flag,
            ])

        print(tabulate(
            status_table,
            headers=["Window", "MSTY Return", "MSTY", "WNTR Return", "WNTR", "Spread", "Signal"],
            tablefmt="simple",
        ))

    # ================================================================
    # DIVERGENCE PATTERN ANALYSIS
    # ================================================================
    print_header("DIVERGENCE PATTERN ANALYSIS")
    print("How do returns look at different divergence levels?\n")

    divergence_patterns = analyzer.analyze_divergence_patterns(comparison)

    if not divergence_patterns.empty:
        div_table = []
        for _, row in divergence_patterns.iterrows():
            level = int(row["divergence_level"])
            desc = row["description"]
            days = int(row["num_days"])
            pct = row["pct_of_total"]

            div_row = [
                f"{level}",
                desc,
                f"{days} ({pct:.0f}%)",
            ]

            for w in windows:
                msty_key = f"avg_msty_{w}d"
                wntr_key = f"avg_wntr_{w}d"
                spread_key = f"avg_spread_{w}d"

                if msty_key in row:
                    div_row.append(f"{row[msty_key]:+.1f}%")
                    div_row.append(f"{row[wntr_key]:+.1f}%")
                else:
                    div_row.append("N/A")
                    div_row.append("N/A")

            div_table.append(div_row)

        headers = ["Level", "Description", "Days"]
        for w in windows:
            headers.extend([f"MSTY {w}D", f"WNTR {w}D"])

        print(tabulate(div_table, headers=headers, tablefmt="simple"))

    # ================================================================
    # TIMEFRAME CASCADE
    # ================================================================
    print_header("TIMEFRAME CASCADE ANALYSIS")
    print("As MSTY goes negative on MORE timeframes, what does WNTR do?\n")

    cascade = analyzer.analyze_timeframe_cascade(comparison)

    if not cascade.empty:
        print_subheader("When MSTY's rolling total returns are negative...")

        cascade_table = []
        for _, row in cascade.iterrows():
            depth = int(row["cascade_depth"])
            msty_neg_on = row["msty_negative_on"]
            days = int(row["num_days"])

            c_row = [
                f"Depth {depth}",
                msty_neg_on,
                str(days),
            ]

            for w in windows:
                wntr_key = f"wntr_avg_{w}d"
                pct_key = f"wntr_pct_pos_{w}d"
                msty_key = f"msty_avg_{w}d"

                if wntr_key in row:
                    c_row.append(f"{row[msty_key]:+.1f}%")
                    c_row.append(f"{row[wntr_key]:+.1f}%")
                    c_row.append(f"{row[pct_key]:.0f}%")
                else:
                    c_row.append("N/A")
                    c_row.append("N/A")
                    c_row.append("N/A")

            cascade_table.append(c_row)

        headers = ["Cascade", "MSTY Neg On", "Days"]
        for w in windows:
            headers.extend([f"MSTY {w}D", f"WNTR {w}D", f"WNTR %Pos"])

        print(tabulate(cascade_table, headers=headers, tablefmt="simple"))

        # Key insight
        if len(cascade) > 0:
            deepest = cascade.iloc[-1]
            print(f"\n  KEY FINDING:")
            print(f"  When MSTY is negative on ALL {len(windows)} timeframes ({deepest['msty_negative_on']}):")
            print(f"    This occurred on {int(deepest['num_days'])} trading days")
            for w in windows:
                wntr_key = f"wntr_avg_{w}d"
                pct_key = f"wntr_pct_pos_{w}d"
                if wntr_key in deepest:
                    print(f"    WNTR {w}D avg return: {deepest[wntr_key]:+.1f}%  "
                          f"(positive {deepest[pct_key]:.0f}% of the time)")

    # ================================================================
    # REFLEXIVE DIVERGENCE
    # ================================================================
    print_header("REFLEXIVE DIVERGENCE — WNTR BEHAVIOR BY MSTY RETURN LEVEL")
    print("As MSTY's 20-day total return worsens, does WNTR improve proportionally?\n")

    reflexive = analyzer.analyze_reflexive_divergence(comparison)

    if not reflexive.empty:
        ref_table = []
        for _, row in reflexive.iterrows():
            r_row = [
                row["msty_return_range"],
                int(row["num_days"]),
            ]

            for w in windows:
                wntr_key = f"wntr_avg_{w}d"
                pct_key = f"wntr_pct_pos_{w}d"
                spread_key = f"spread_{w}d"

                if wntr_key in row:
                    r_row.append(f"{row[wntr_key]:+.1f}%")
                    r_row.append(f"{row[pct_key]:.0f}%")
                else:
                    r_row.append("N/A")
                    r_row.append("N/A")

            ref_table.append(r_row)

        headers = ["MSTY 20D Return", "Days"]
        for w in windows:
            headers.extend([f"WNTR {w}D Avg", f"WNTR %Pos"])

        print(tabulate(ref_table, headers=headers, tablefmt="simple"))

        print("\n  INTERPRETATION:")
        print("  Read left to right: as MSTY's 20-day return gets more negative,")
        print("  observe how WNTR's returns on each timeframe change.")
        print("  A strong inverse relationship = reflexive divergence at work.")

    # ================================================================
    # ROLLING SNAPSHOT TABLE
    # ================================================================
    print_header("RECENT ROLLING TOTAL RETURNS — DAY BY DAY")
    print("Last 20 trading days, showing both instruments side by side\n")

    recent = comparison.tail(20).copy()
    snapshot_table = []

    for date, row in recent.iterrows():
        s_row = [date.strftime("%Y-%m-%d")]

        for w in windows:
            msty_col = f"msty_total_return_{w}d"
            wntr_col = f"wntr_total_return_{w}d"

            msty_val = row.get(msty_col, np.nan)
            wntr_val = row.get(wntr_col, np.nan)

            if not np.isnan(msty_val):
                s_row.append(f"{msty_val:+.1f}%")
            else:
                s_row.append("—")

            if not np.isnan(wntr_val):
                s_row.append(f"{wntr_val:+.1f}%")
            else:
                s_row.append("—")

        div_count = row.get("divergence_count", 0)
        if not np.isnan(div_count):
            flags = ""
            if div_count >= 3:
                flags = "<<< FULL DIVERGENCE"
            elif div_count >= 2:
                flags = "<< BUILDING"
            elif div_count >= 1:
                flags = "< EARLY"
            s_row.append(f"{int(div_count)} {flags}")
        else:
            s_row.append("—")

        snapshot_table.append(s_row)

    headers = ["Date"]
    for w in windows:
        headers.extend([f"MSTY {w}D", f"WNTR {w}D"])
    headers.append("Divergence")

    print(tabulate(snapshot_table, headers=headers, tablefmt="simple"))

    # ================================================================
    # SUMMARY
    # ================================================================
    print_header("SUMMARY — MSTY vs WNTR COUNTERPARTY ANALYSIS")

    # Count divergence occurrences
    valid = comparison.dropna(subset=[f"msty_total_return_{windows[0]}d"])
    full_div_days = (valid["divergence_count"] == len(windows)).sum()
    partial_div_days = (valid["divergence_count"] >= 1).sum()
    total_days = len(valid)

    print(f"""
  INSTRUMENT COMPARISON:
    MSTY Total Return (with dividends):  {msty_total_return:+.1f}%
    WNTR Total Return (with dividends):  {wntr_total_return:+.1f}%
    Spread (WNTR - MSTY):               {wntr_total_return - msty_total_return:+.1f}%

  DIVERGENCE STATISTICS:
    Total analysis days:                  {total_days}
    Days with any divergence (>=1 TF):    {partial_div_days} ({partial_div_days/total_days*100:.0f}%)
    Days with FULL divergence (all TFs):  {full_div_days} ({full_div_days/total_days*100:.0f}%)

  WHAT THIS MEANS:
    When MSTY's rolling returns go negative due to reflexive collapse
    (shrinking equity + shrinking dividends), WNTR should be moving the
    opposite direction. The divergence count tells you how many timeframes
    confirm this pattern:

      0 = No divergence: both moving same direction or mixed
      1 = Early signal: shortest timeframe diverging
      2 = Building: short + medium timeframes diverging
      3 = Full divergence: ALL timeframes confirm MSTY collapsing, WNTR rising

    Full divergence on all timeframes is the strongest confirmation
    that the reflexive collapse mechanism is active.
""")

    # ================================================================
    # GENERATE CHARTS
    # ================================================================
    print_subheader("Generating Charts")

    viz = CounterpartyVisualizer()
    viz.plot_rolling_returns_comparison(comparison, windows=windows)
    viz.plot_divergence_dashboard(comparison, windows=windows)
    viz.plot_reflexive_analysis(reflexive, windows=windows)

    print("\nCharts saved to output/charts/")
    print("  - msty_vs_wntr_rolling_returns.png")
    print("  - msty_vs_wntr_divergence.png")
    print("  - msty_vs_wntr_reflexive.png")


if __name__ == "__main__":
    run_analysis()
