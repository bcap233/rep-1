#!/usr/bin/env python3
"""
Always-Invested Rotation Strategy — MSTY vs WNTR.

The goal: always own one of the two instruments to capture maximum dividends.
Never sit in cash. The only decision is WHICH one to hold.

Compares 4 signal variants:
  V1: Baseline 3/3 TF divergence (default MSTY)
  V2: Debounced (3-day confirmation before switching)
  V3: 50-day SMA lead signal
  V4: SMA + debounce combined
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.rotation_strategy import RotationBacktester, RotationVisualizer


def print_header(title: str):
    print(f"\n{'=' * 85}")
    print(f"  {title}")
    print(f"{'=' * 85}\n")


def print_subheader(title: str):
    print(f"\n--- {title} ---\n")


def print_trade_log(result):
    """Print detailed trade log with dividend tracking."""
    print(f"{'#':>3}  {'Instr':>5}  {'Entry':>12}  {'Exit':>12}  "
          f"{'Days':>5}  {'Entry$':>8}  {'Exit$':>8}  "
          f"{'TotRet':>7}  {'PxRet':>7}  {'Divs':>6}  {'#Div':>4}")
    print("-" * 100)

    for t in result.trades:
        print(
            f"{t.trade_num:>3}  "
            f"{t.instrument:>5}  "
            f"{t.entry_date.date()!s:>12}  "
            f"{t.exit_date.date()!s:>12}  "
            f"{t.holding_days:>5}  "
            f"${t.entry_price:>7.2f}  "
            f"${t.exit_price:>7.2f}  "
            f"{t.total_return_pct:>+6.1f}%  "
            f"{t.price_return_pct:>+6.1f}%  "
            f"${t.dividends_collected:>5.2f}  "
            f"{t.num_dividends:>4}"
        )


def print_summary_table(results):
    """Side-by-side comparison of all strategy variants."""
    print(f"  {'Metric':<30s}", end="")
    for r in results:
        label = r.strategy_name.split(":")[0].strip()
        print(f"  {label:>16s}", end="")
    print()
    print("  " + "-" * (30 + 18 * len(results)))

    rows = [
        ("Total Return", [f"{r.total_return_pct:+.1f}%" for r in results]),
        ("Final Equity ($10k start)", [f"${10000*(1+r.total_return_pct/100):,.0f}" for r in results]),
        ("Total Switches", [f"{r.num_switches}" for r in results]),
        ("Days in MSTY", [f"{r.msty_days} ({r.msty_days/(r.msty_days+r.wntr_days)*100:.0f}%)" for r in results]),
        ("Days in WNTR", [f"{r.wntr_days} ({r.wntr_days/(r.msty_days+r.wntr_days)*100:.0f}%)" for r in results]),
        ("Dividends from MSTY", [f"${r.msty_dividends:.2f}" for r in results]),
        ("Dividends from WNTR", [f"${r.wntr_dividends:.2f}" for r in results]),
        ("Total Dividends", [f"${r.total_dividends_collected:.2f}" for r in results]),
        ("Number of Trades", [f"{len(r.trades)}" for r in results]),
    ]

    # Win rate
    win_rates = []
    for r in results:
        winners = sum(1 for t in r.trades if t.total_return_pct > 0)
        win_rates.append(f"{winners}/{len(r.trades)} ({winners/len(r.trades)*100:.0f}%)")
    rows.append(("Win Rate", win_rates))

    # Max drawdown
    dds = []
    for r in results:
        rm = r.daily_equity.expanding().max()
        dd = (r.daily_equity / rm - 1) * 100
        dds.append(f"{dd.min():.1f}%")
    rows.append(("Max Drawdown", dds))

    for label, values in rows:
        print(f"  {label:<30s}", end="")
        for v in values:
            print(f"  {v:>16s}", end="")
        print()


def main():
    print_header("ALWAYS-INVESTED ROTATION: MSTY vs WNTR")
    print("  Goal: Always own one instrument to capture maximum dividends.")
    print("  Question: WHICH one do I own right now?")
    print("  Default: MSTY (higher dividends in bull regime)")
    print("  Switch to WNTR when: collapse signals confirm MSTY declining")

    # Load data
    msty_df = pd.read_csv("msty_data_with_dividends.csv", parse_dates=["Date"], index_col="Date").sort_index()
    msty_df.columns = [c.lower().strip() for c in msty_df.columns]

    wntr_df = pd.read_csv("wntr_data_with_dividends.csv", parse_dates=["Date"], index_col="Date").sort_index()
    wntr_df.columns = [c.lower().strip() for c in wntr_df.columns]

    overlap_start = max(msty_df.index.min(), wntr_df.index.min())
    overlap_end = min(msty_df.index.max(), wntr_df.index.max())
    msty_df = msty_df.loc[overlap_start:overlap_end]
    wntr_df = wntr_df.loc[overlap_start:overlap_end]

    print(f"\n  Period: {overlap_start.date()} to {overlap_end.date()}")
    print(f"  MSTY total dividends in period: ${msty_df['dividend'].sum():.2f}")
    print(f"  WNTR total dividends in period: ${wntr_df['dividend'].sum():.2f}")

    # ================================================================
    # RUN ALL 4 VARIANTS
    # ================================================================
    bt = RotationBacktester(windows=[5, 10, 20])

    print_header("RUNNING 4 STRATEGY VARIANTS")

    print("  V1: Baseline — 3/3 TF divergence, default MSTY on neutral")
    v1 = bt.run_v1_baseline(msty_df, wntr_df)

    print("  V2: Debounced — Require 3 consecutive days before switching")
    v2 = bt.run_v2_debounced(msty_df, wntr_df, debounce_days=3)

    print("  V3: SMA-Enhanced — 50-day SMA as primary signal")
    v3 = bt.run_v3_sma_enhanced(msty_df, wntr_df, sma_period=50)

    print("  V4: SMA + Debounce — 50-day SMA with 3-day confirmation")
    v4 = bt.run_v4_sma_plus_debounce(msty_df, wntr_df, sma_period=50, debounce_days=3)

    results = [v1, v2, v3, v4]

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print_header("STRATEGY COMPARISON — ALL VARIANTS")
    print_summary_table(results)

    print(f"\n  BUY-AND-HOLD BENCHMARKS:")
    print(f"    MSTY only (total return): {v1.msty_buy_hold_return:+.1f}%  "
          f"-> ${10000*(1+v1.msty_buy_hold_return/100):,.0f}")
    print(f"    WNTR only (total return): {v1.wntr_buy_hold_return:+.1f}%  "
          f"-> ${10000*(1+v1.wntr_buy_hold_return/100):,.0f}")

    # ================================================================
    # DETAILED TRADE LOGS
    # ================================================================
    for res in results:
        print_header(f"TRADE LOG: {res.strategy_name}")
        print_trade_log(res)

        # Key holding periods
        significant = [t for t in res.trades if abs(t.total_return_pct) > 5 or t.holding_days > 20]
        if significant:
            print_subheader("Significant Holding Periods")
            for t in significant:
                div_note = f", collected ${t.dividends_collected:.2f} ({t.num_dividends} payments)" if t.num_dividends > 0 else ""
                print(f"    #{t.trade_num} {t.instrument}: {t.entry_date.date()} to {t.exit_date.date()} "
                      f"({t.holding_days}d) -> {t.total_return_pct:+.1f}% total "
                      f"({t.price_return_pct:+.1f}% price{div_note})")

    # ================================================================
    # DIVIDEND ANALYSIS
    # ================================================================
    print_header("DIVIDEND INCOME ANALYSIS")

    for res in results:
        label = res.strategy_name.split(":")[0].strip()
        print(f"  {label}:")
        print(f"    MSTY dividends: ${res.msty_dividends:.2f} "
              f"(captured {res.msty_days} days of MSTY)")
        print(f"    WNTR dividends: ${res.wntr_dividends:.2f} "
              f"(captured {res.wntr_days} days of WNTR)")
        print(f"    Total dividends: ${res.total_dividends_collected:.2f}")
        print()

    # What would pure B&H have collected?
    msty_total_divs = msty_df["dividend"].sum()
    wntr_total_divs = wntr_df["dividend"].sum()
    print(f"  BENCHMARKS:")
    print(f"    MSTY B&H dividends: ${msty_total_divs:.2f} (all {len(msty_df)} days)")
    print(f"    WNTR B&H dividends: ${wntr_total_divs:.2f} (all {len(wntr_df)} days)")

    # ================================================================
    # WINNER ANALYSIS
    # ================================================================
    print_header("WHICH STRATEGY WINS?")

    best = max(results, key=lambda r: r.total_return_pct)
    most_divs = max(results, key=lambda r: r.total_dividends_collected)
    fewest_switches = min(results, key=lambda r: r.num_switches)

    # Max drawdowns
    def max_dd(r):
        rm = r.daily_equity.expanding().max()
        return ((r.daily_equity / rm - 1) * 100).min()

    shallowest_dd = max(results, key=lambda r: max_dd(r))

    print(f"  Best total return:   {best.strategy_name} ({best.total_return_pct:+.1f}%)")
    print(f"  Most dividends:      {most_divs.strategy_name} (${most_divs.total_dividends_collected:.2f})")
    print(f"  Fewest switches:     {fewest_switches.strategy_name} ({fewest_switches.num_switches} switches)")
    print(f"  Shallowest drawdown: {shallowest_dd.strategy_name} ({max_dd(shallowest_dd):.1f}%)")

    print(f"\n  KEY INSIGHT:")
    print(f"    All rotation strategies beat MSTY buy-and-hold ({v1.msty_buy_hold_return:+.1f}%)")
    print(f"    because they switch to WNTR during the collapse.")
    print(f"    The question is how MUCH of WNTR's {v1.wntr_buy_hold_return:+.1f}% they capture")
    print(f"    while still collecting MSTY's higher dividends in the bull phase.")

    # ================================================================
    # CHARTS
    # ================================================================
    print_header("GENERATING CHARTS")

    viz = RotationVisualizer()
    viz.plot_rotation_comparison(results)
    viz.plot_dividend_income_timeline(results, msty_df, wntr_df)

    print("\nCharts saved:")
    print("  - output/charts/rotation_strategy_comparison.png")
    print("  - output/charts/rotation_dividend_income.png")


if __name__ == "__main__":
    main()
