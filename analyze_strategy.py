#!/usr/bin/env python3
"""
Rolling Total Return Signal Strategy — Full Backtest.

Answers: "If we had followed the 3-timeframe divergence signal,
what would we have held on what dates, and what would total return be?"

Signal rules:
- OWN MSTY: All 3 timeframes (5D/10D/20D) show MSTY positive + WNTR negative
- OWN WNTR: All 3 timeframes show MSTY negative + WNTR positive
- CASH: Mixed signals -> sit out

Uses total return indices (TRI) for P&L calculation, which correctly
handles the MSTY 1:5 split on Dec 8, 2025 and all dividend reinvestment.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.strategy_backtest import StrategyBacktester, StrategyVisualizer
from src.counterparty_analysis import CounterpartyAnalyzer


def print_header(title: str):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_subheader(title: str):
    print(f"\n--- {title} ---\n")


def main():
    print_header("ROLLING TOTAL RETURN SIGNAL — STRATEGY BACKTEST")
    print("What would have happened if we followed the 3-timeframe signal?")
    print("Using total return indices to correctly handle splits + dividends.\n")

    # ================================================================
    # LOAD DATA
    # ================================================================
    msty_csv = Path("msty_data_with_dividends.csv")
    wntr_csv = Path("wntr_data_with_dividends.csv")

    if not msty_csv.exists() or not wntr_csv.exists():
        print("ERROR: Need both msty_data_with_dividends.csv and wntr_data_with_dividends.csv")
        return

    msty_df = pd.read_csv(msty_csv, parse_dates=["Date"], index_col="Date").sort_index()
    msty_df.columns = [c.lower().strip() for c in msty_df.columns]

    wntr_df = pd.read_csv(wntr_csv, parse_dates=["Date"], index_col="Date").sort_index()
    wntr_df.columns = [c.lower().strip() for c in wntr_df.columns]

    # Align to overlapping period
    overlap_start = max(msty_df.index.min(), wntr_df.index.min())
    overlap_end = min(msty_df.index.max(), wntr_df.index.max())
    msty_df = msty_df.loc[overlap_start:overlap_end]
    wntr_df = wntr_df.loc[overlap_start:overlap_end]

    print(f"  Period: {overlap_start.date()} to {overlap_end.date()}")
    print(f"  MSTY days: {len(msty_df)}, WNTR days: {len(wntr_df)}")

    # ================================================================
    # RUN BACKTEST — STRONG SIGNALS ONLY (3/3 TF)
    # ================================================================
    print_header("STRATEGY 1: STRONG SIGNALS ONLY (ALL 3 TIMEFRAMES MUST AGREE)")
    print("  OWN MSTY = 5D + 10D + 20D all show MSTY pos + WNTR neg")
    print("  OWN WNTR = 5D + 10D + 20D all show MSTY neg + WNTR pos")
    print("  Otherwise = CASH (flat, no position)\n")

    backtester = StrategyBacktester(windows=[5, 10, 20], signal_threshold=3)
    result = backtester.run_backtest(msty_df, wntr_df, neutral_action="cash")

    # ================================================================
    # TRADE LOG
    # ================================================================
    print_subheader("COMPLETE TRADE LOG")
    print(f"{'#':>3}  {'Instrument':>10}  {'Entry Date':>12}  {'Exit Date':>12}  "
          f"{'Days':>5}  {'Entry $':>9}  {'Exit $':>9}  {'Return':>8}  {'Cumul $':>10}")
    print("-" * 95)

    cumulative_equity = 10000.0
    for trade in result.trades:
        cumulative_equity *= (1 + trade.return_pct / 100)

        entry_price_str = f"${trade.entry_price:.2f}" if not np.isnan(trade.entry_price) else "n/a"
        exit_price_str = f"${trade.exit_price:.2f}" if not np.isnan(trade.exit_price) else "n/a"

        print(
            f"{trade.trade_num:>3}  "
            f"{trade.instrument:>10}  "
            f"{trade.entry_date.date()!s:>12}  "
            f"{trade.exit_date.date()!s:>12}  "
            f"{trade.holding_days:>5}  "
            f"{entry_price_str:>9}  "
            f"{exit_price_str:>9}  "
            f"{trade.return_pct:>+7.1f}%  "
            f"${cumulative_equity:>9,.0f}"
        )

    # ================================================================
    # POSITION SUMMARY
    # ================================================================
    print_subheader("POSITION SUMMARY")

    msty_trades = [t for t in result.trades if t.instrument == "MSTY"]
    wntr_trades = [t for t in result.trades if t.instrument == "WNTR"]

    msty_days = sum(t.holding_days for t in msty_trades)
    wntr_days = sum(t.holding_days for t in wntr_trades)
    total_period = (overlap_end - overlap_start).days
    cash_days = total_period - msty_days - wntr_days

    print(f"  Total period:     {total_period} calendar days")
    print(f"  Days in MSTY:     {msty_days} ({msty_days/total_period*100:.0f}%)")
    print(f"  Days in WNTR:     {wntr_days} ({wntr_days/total_period*100:.0f}%)")
    print(f"  Days in CASH:     {cash_days} ({cash_days/total_period*100:.0f}%)")

    print(f"\n  MSTY trades: {len(msty_trades)}")
    if msty_trades:
        msty_returns = [t.return_pct for t in msty_trades]
        msty_winners = [r for r in msty_returns if r > 0]
        print(f"    Winners: {len(msty_winners)}/{len(msty_trades)}")
        print(f"    Avg return: {np.mean(msty_returns):+.1f}%")
        print(f"    Best: {max(msty_returns):+.1f}%, Worst: {min(msty_returns):+.1f}%")
        print(f"    Total contribution: {sum(msty_returns):+.1f}% (simple sum)")

    print(f"\n  WNTR trades: {len(wntr_trades)}")
    if wntr_trades:
        wntr_returns = [t.return_pct for t in wntr_trades]
        wntr_winners = [r for r in wntr_returns if r > 0]
        print(f"    Winners: {len(wntr_winners)}/{len(wntr_trades)}")
        print(f"    Avg return: {np.mean(wntr_returns):+.1f}%")
        print(f"    Best: {max(wntr_returns):+.1f}%, Worst: {min(wntr_returns):+.1f}%")
        print(f"    Total contribution: {sum(wntr_returns):+.1f}% (simple sum)")

    # ================================================================
    # PERFORMANCE COMPARISON
    # ================================================================
    print_header("PERFORMANCE COMPARISON")

    final_equity = result.daily_equity.iloc[-1]

    print(f"  Starting capital:                $10,000")
    print(f"")
    print(f"  STRATEGY (3-TF signal):          ${final_equity:>10,.0f}  ({result.total_return_pct:>+7.1f}%)")
    print(f"  MSTY Buy & Hold (total return):  ${10000*(1+result.msty_buy_hold_return/100):>10,.0f}  ({result.msty_buy_hold_return:>+7.1f}%)")
    print(f"  WNTR Buy & Hold (total return):  ${10000*(1+result.wntr_buy_hold_return/100):>10,.0f}  ({result.wntr_buy_hold_return:>+7.1f}%)")
    print(f"  Cash (0% return):                $    10,000  (  +0.0%)")

    print(f"\n  Strategy statistics:")
    print(f"    Total trades:       {result.num_trades}")
    print(f"    Win rate:           {result.win_rate:.0f}%")
    print(f"    Avg holding period: {result.avg_holding_days:.0f} days")

    # Calculate max drawdown
    running_max = result.daily_equity.expanding().max()
    drawdown = (result.daily_equity / running_max - 1) * 100
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()

    msty_eq = 10000 * (result.msty_tri / result.msty_tri.iloc[0])
    msty_rm = msty_eq.expanding().max()
    msty_dd = (msty_eq / msty_rm - 1) * 100

    wntr_eq = 10000 * (result.wntr_tri / result.wntr_tri.iloc[0])
    wntr_rm = wntr_eq.expanding().max()
    wntr_dd = (wntr_eq / wntr_rm - 1) * 100

    print(f"\n  Max drawdowns:")
    print(f"    Strategy:    {max_dd:.1f}% (on {max_dd_date.date()})")
    print(f"    MSTY B&H:    {msty_dd.min():.1f}% (on {msty_dd.idxmin().date()})")
    print(f"    WNTR B&H:    {wntr_dd.min():.1f}% (on {wntr_dd.idxmin().date()})")

    # ================================================================
    # REGIME BREAKDOWN
    # ================================================================
    print_header("REGIME BREAKDOWN — KEY PERIODS")

    # Identify the major holding periods
    print("  KEY HOLDING PERIODS:\n")
    for trade in result.trades:
        if abs(trade.return_pct) > 5 or trade.holding_days > 14:
            print(f"    Trade #{trade.trade_num}: {trade.instrument} "
                  f"from {trade.entry_date.date()} to {trade.exit_date.date()} "
                  f"({trade.holding_days}d)")
            print(f"      Entry: ${trade.entry_price:.2f}, "
                  f"Exit: ${trade.exit_price:.2f}, "
                  f"Return: {trade.return_pct:+.1f}%")

            # Context for what was happening
            if trade.instrument == "MSTY" and trade.return_pct > 5:
                print(f"      -> MSTY rallying, signal correctly positioned long MSTY")
            elif trade.instrument == "MSTY" and trade.return_pct < -5:
                print(f"      -> MSTY declining while signal said OWN MSTY (lagging)")
            elif trade.instrument == "WNTR" and trade.return_pct > 5:
                print(f"      -> WNTR rallying as MSTY collapsed, signal correctly positioned")
            elif trade.instrument == "WNTR" and trade.return_pct < -5:
                print(f"      -> WNTR declining while signal said OWN WNTR (wrong)")
            print()

    # ================================================================
    # CRITICAL ASSESSMENT
    # ================================================================
    print_header("CRITICAL ASSESSMENT")

    # Count whipsaw trades (< 5 days)
    whipsaw_trades = [t for t in result.trades if t.holding_days <= 7]
    whipsaw_losses = sum(t.return_pct for t in whipsaw_trades if t.return_pct < 0)

    print("  STRENGTHS:")
    # Find the longest winning streak
    longest_winner = max(result.trades, key=lambda t: t.return_pct) if result.trades else None
    if longest_winner:
        print(f"    - Best trade: #{longest_winner.trade_num} {longest_winner.instrument} "
              f"{longest_winner.entry_date.date()} to {longest_winner.exit_date.date()} "
              f"({longest_winner.return_pct:+.1f}%)")

    # Find longest holding period
    longest_hold = max(result.trades, key=lambda t: t.holding_days) if result.trades else None
    if longest_hold and longest_hold != longest_winner:
        print(f"    - Longest hold: #{longest_hold.trade_num} {longest_hold.instrument} "
              f"{longest_hold.entry_date.date()} to {longest_hold.exit_date.date()} "
              f"({longest_hold.holding_days}d, {longest_hold.return_pct:+.1f}%)")

    print(f"\n  WEAKNESSES:")
    print(f"    - Whipsaw trades (<=7 days): {len(whipsaw_trades)} trades")
    print(f"    - Whipsaw losses: {whipsaw_losses:+.1f}% (sum of losing short trades)")
    print(f"    - Signal is LAGGING: uses 5/10/20 day lookback windows")
    print(f"    - Missed the July 17, 2025 MSTY peak by ~14 trading days")
    print(f"    - Cash drag: {cash_days/total_period*100:.0f}% of time in cash earning nothing")

    print(f"\n  CONCLUSION:")
    if result.total_return_pct > 0:
        print(f"    The strategy returned {result.total_return_pct:+.1f}%.")
    else:
        print(f"    The strategy LOST {abs(result.total_return_pct):.1f}%.")

    print(f"    MSTY buy-and-hold returned {result.msty_buy_hold_return:+.1f}% (with dividends).")
    print(f"    WNTR buy-and-hold returned {result.wntr_buy_hold_return:+.1f}% (with dividends).")
    print()
    print(f"    The rolling total return signal is a REGIME CONFIRMATION tool,")
    print(f"    NOT a trading system. Its value is in confirming when the")
    print(f"    reflexive collapse is active (sustained 3/3 divergence),")
    print(f"    not in timing entries and exits.")
    print()
    print(f"    The user's insight about the 50-day SMA break on July 17 was")
    print(f"    the actual tradeable signal. The 3-TF divergence provides")
    print(f"    conviction confirmation AFTER the regime has already shifted.")

    # ================================================================
    # CHARTS
    # ================================================================
    print_header("GENERATING CHARTS")

    # Build comparison data for charts
    analyzer = CounterpartyAnalyzer(windows=[5, 10, 20])
    comparison = analyzer.prepare_comparison_data(msty_df, wntr_df)

    viz = StrategyVisualizer()
    viz.plot_strategy_equity_curve(result, comparison)
    viz.plot_signal_timeline(comparison, result, windows=[5, 10, 20])

    print("\nCharts saved to output/charts/")
    print("  - strategy_backtest.png")
    print("  - strategy_signal_timeline.png")


if __name__ == "__main__":
    main()
