#!/usr/bin/env python3
"""
Multi-Pair Rotation Strategy Backtest.

Tests the always-invested SMA rotation strategy across multiple
bull/bear YieldMax instrument pairs:
  - MSTY / WNTR  (MSTR bull / bear)
  - NVDY / DIPS  (NVDA bull / bear)
  - COIN / FIAT  (Coinbase bull / bear)
  - TSLY / CRSH  (TSLA bull / bear)

For each pair, the strategy is:
  - Own the BULL instrument when it's above its 50-day SMA
  - Own the BEAR instrument when bull is below its 50-day SMA
  - Wait 3 days before switching (debounce)

Usage:
  python analyze_pairs.py

CSV files should be named: {ticker}_data_with_dividends.csv
Format: Date,Close,Dividend,Split (newest first or oldest first, both work)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.rotation_strategy import RotationBacktester, RotationVisualizer, RotationResult
from src.counterparty_analysis import TotalReturnCalculator

# ================================================================
# PAIR CONFIGURATION
# ================================================================
PAIRS = [
    {
        "name": "MSTR (BTC Proxy)",
        "bull": "MSTY",
        "bear": "WNTR",
        "bull_csv": "msty_data_with_dividends.csv",
        "bear_csv": "wntr_data_with_dividends.csv",
    },
    {
        "name": "NVIDIA",
        "bull": "NVDY",
        "bear": "DIPS",
        "bull_csv": "nvdy_data_with_dividends.csv",
        "bear_csv": "dips_data_with_dividends.csv",
    },
    {
        "name": "Coinbase",
        "bull": "CONY",
        "bear": "FIAT",
        "bull_csv": "cony_data_with_dividends.csv",
        "bear_csv": "fiat_data_with_dividends.csv",
    },
    {
        "name": "Tesla",
        "bull": "TSLY",
        "bear": "CRSH",
        "bull_csv": "tsly_data_with_dividends.csv",
        "bear_csv": "crsh_data_with_dividends.csv",
    },
]


def print_header(title: str):
    print(f"\n{'=' * 85}")
    print(f"  {title}")
    print(f"{'=' * 85}\n")


def print_subheader(title: str):
    print(f"\n--- {title} ---\n")


def load_pair_data(pair: dict) -> tuple:
    """Load and align data for a bull/bear pair. Returns (bull_df, bear_df) or (None, None)."""
    bull_csv = Path(pair["bull_csv"])
    bear_csv = Path(pair["bear_csv"])

    if not bull_csv.exists():
        print(f"  MISSING: {bull_csv} — skipping {pair['bull']}")
        return None, None
    if not bear_csv.exists():
        print(f"  MISSING: {bear_csv} — skipping {pair['bear']}")
        return None, None

    bull_df = pd.read_csv(bull_csv, parse_dates=["Date"], index_col="Date").sort_index()
    bull_df.columns = [c.lower().strip() for c in bull_df.columns]

    bear_df = pd.read_csv(bear_csv, parse_dates=["Date"], index_col="Date").sort_index()
    bear_df.columns = [c.lower().strip() for c in bear_df.columns]

    # Align to overlapping period
    overlap_start = max(bull_df.index.min(), bear_df.index.min())
    overlap_end = min(bull_df.index.max(), bear_df.index.max())

    bull_df = bull_df.loc[overlap_start:overlap_end]
    bear_df = bear_df.loc[overlap_start:overlap_end]

    return bull_df, bear_df


def run_pair_backtest(pair: dict, bull_df: pd.DataFrame, bear_df: pd.DataFrame) -> dict:
    """Run V4 (SMA + debounce) rotation on a single pair."""
    bt = RotationBacktester(windows=[5, 10, 20])

    # V4: SMA(50) + Debounce(3) — our best performer
    result = bt.run_v4_sma_plus_debounce(bull_df, bear_df, sma_period=50, debounce_days=3)

    # Also run V3 for comparison
    result_v3 = bt.run_v3_sma_enhanced(bull_df, bear_df, sma_period=50)

    # Buy-and-hold returns
    calc = TotalReturnCalculator()
    bull_tri = calc.calculate_total_return_index(bull_df)
    bear_tri = calc.calculate_total_return_index(bear_df)
    bull_bh = (bull_tri.iloc[-1] / bull_tri.iloc[0] - 1) * 100
    bear_bh = (bear_tri.iloc[-1] / bear_tri.iloc[0] - 1) * 100

    return {
        "pair": pair,
        "v4": result,
        "v3": result_v3,
        "bull_bh": bull_bh,
        "bear_bh": bear_bh,
        "bull_df": bull_df,
        "bear_df": bear_df,
    }


def print_pair_results(res: dict):
    """Print detailed results for one pair."""
    pair = res["pair"]
    v4 = res["v4"]
    v3 = res["v3"]

    print_header(f"{pair['bull']} / {pair['bear']} — {pair['name']}")

    period_start = v4.trades[0].entry_date.date()
    period_end = v4.trades[-1].exit_date.date()
    total_days = v4.msty_days + v4.wntr_days

    print(f"  Period: {period_start} to {period_end} ({total_days} trading days)")
    print(f"  {pair['bull']} dividends in period: ${res['bull_df']['dividend'].sum():.2f}")
    print(f"  {pair['bear']} dividends in period: ${res['bear_df']['dividend'].sum():.2f}")

    print_subheader("Performance")

    # Max drawdowns
    def max_dd(equity):
        rm = equity.expanding().max()
        return ((equity / rm - 1) * 100).min()

    print(f"  {'Strategy':<35s}  {'Return':>10s}  {'$10k ->':>10s}  {'MaxDD':>8s}  {'Trades':>7s}")
    print(f"  {'-' * 75}")
    print(f"  {'V4: SMA(50) + Debounce(3d)':<35s}  {v4.total_return_pct:>+9.1f}%  ${10000*(1+v4.total_return_pct/100):>9,.0f}  {max_dd(v4.daily_equity):>7.1f}%  {len(v4.trades):>7}")
    print(f"  {'V3: SMA(50) raw':<35s}  {v3.total_return_pct:>+9.1f}%  ${10000*(1+v3.total_return_pct/100):>9,.0f}  {max_dd(v3.daily_equity):>7.1f}%  {len(v3.trades):>7}")
    print(f"  {pair['bull'] + ' Buy & Hold':<35s}  {res['bull_bh']:>+9.1f}%  ${10000*(1+res['bull_bh']/100):>9,.0f}  {'':>8s}  {'':>7s}")
    print(f"  {pair['bear'] + ' Buy & Hold':<35s}  {res['bear_bh']:>+9.1f}%  ${10000*(1+res['bear_bh']/100):>9,.0f}  {'':>8s}  {'':>7s}")

    print_subheader("V4 Trade Log")
    print(f"  {'#':>3}  {'Instr':>6}  {'Entry':>12}  {'Exit':>12}  "
          f"{'Days':>5}  {'TotRet':>8}  {'PxRet':>8}  {'Divs':>7}  {'#Div':>5}")
    print(f"  {'-' * 80}")

    for t in v4.trades:
        inst_label = pair["bull"] if t.instrument == "MSTY" else pair["bear"]
        print(
            f"  {t.trade_num:>3}  "
            f"{inst_label:>6}  "
            f"{t.entry_date.date()!s:>12}  "
            f"{t.exit_date.date()!s:>12}  "
            f"{t.holding_days:>5}  "
            f"{t.total_return_pct:>+7.1f}%  "
            f"{t.price_return_pct:>+7.1f}%  "
            f"${t.dividends_collected:>6.2f}  "
            f"{t.num_dividends:>5}"
        )

    print_subheader("Allocation & Dividends")
    bull_pct = v4.msty_days / total_days * 100
    bear_pct = v4.wntr_days / total_days * 100
    print(f"  Days in {pair['bull']}: {v4.msty_days} ({bull_pct:.0f}%)")
    print(f"  Days in {pair['bear']}: {v4.wntr_days} ({bear_pct:.0f}%)")
    print(f"  {pair['bull']} dividends captured: ${v4.msty_dividends:.2f}")
    print(f"  {pair['bear']} dividends captured: ${v4.wntr_dividends:.2f}")
    print(f"  Total dividends captured: ${v4.total_dividends_collected:.2f}")

    # Current SMA status
    bull_df = res["bull_df"]
    sma50 = bull_df["close"].rolling(50).mean()
    last_price = bull_df["close"].iloc[-1]
    last_sma = sma50.iloc[-1]
    if pd.notna(last_sma):
        gap = (last_price / last_sma - 1) * 100
        signal = f"OWN {pair['bull']}" if last_price > last_sma else f"OWN {pair['bear']}"
        print(f"\n  CURRENT SIGNAL: {signal}")
        print(f"    {pair['bull']} price: ${last_price:.2f}, SMA50: ${last_sma:.2f} ({gap:+.1f}%)")


def print_cross_pair_summary(all_results: list):
    """Print comparison table across all pairs."""
    print_header("CROSS-PAIR COMPARISON — V4 SMA(50) + DEBOUNCE(3)")

    print(f"  {'Pair':<18s}  {'V4 Return':>10s}  {'Bull B&H':>10s}  {'Bear B&H':>10s}  "
          f"{'V4 vs Bull':>10s}  {'Trades':>7s}  {'MaxDD':>8s}  {'Divs':>8s}")
    print(f"  {'-' * 95}")

    for res in all_results:
        pair = res["pair"]
        v4 = res["v4"]
        label = f"{pair['bull']}/{pair['bear']}"

        rm = v4.daily_equity.expanding().max()
        dd = ((v4.daily_equity / rm - 1) * 100).min()
        edge = v4.total_return_pct - res["bull_bh"]

        print(
            f"  {label:<18s}  "
            f"{v4.total_return_pct:>+9.1f}%  "
            f"{res['bull_bh']:>+9.1f}%  "
            f"{res['bear_bh']:>+9.1f}%  "
            f"{edge:>+9.1f}%  "
            f"{len(v4.trades):>7}  "
            f"{dd:>7.1f}%  "
            f"${v4.total_dividends_collected:>7.2f}"
        )

    print()
    # Win rate across pairs
    wins = sum(1 for r in all_results if r["v4"].total_return_pct > r["bull_bh"])
    total = len(all_results)
    print(f"  Strategy beat bull B&H: {wins}/{total} pairs")

    wins_bear = sum(1 for r in all_results if r["v4"].total_return_pct > r["bear_bh"])
    print(f"  Strategy beat bear B&H: {wins_bear}/{total} pairs")

    avg_return = np.mean([r["v4"].total_return_pct for r in all_results])
    avg_bull = np.mean([r["bull_bh"] for r in all_results])
    avg_bear = np.mean([r["bear_bh"] for r in all_results])

    print(f"\n  Average V4 return:    {avg_return:+.1f}%")
    print(f"  Average bull B&H:     {avg_bull:+.1f}%")
    print(f"  Average bear B&H:     {avg_bear:+.1f}%")
    print(f"  Average edge vs bull: {avg_return - avg_bull:+.1f}%")


def main():
    print_header("MULTI-PAIR ROTATION STRATEGY BACKTEST")
    print("  Testing SMA(50) + Debounce(3) rotation across YieldMax pairs")
    print("  Rule: Own bull when bull > SMA50, own bear when bull < SMA50")
    print()

    # Check which pairs have data
    available = []
    missing = []

    for pair in PAIRS:
        bull_df, bear_df = load_pair_data(pair)
        if bull_df is not None and bear_df is not None:
            print(f"  FOUND: {pair['bull']}/{pair['bear']} — "
                  f"{bull_df.index[0].date()} to {bull_df.index[-1].date()} "
                  f"({len(bull_df)}/{len(bear_df)} days)")
            available.append((pair, bull_df, bear_df))
        else:
            missing.append(pair)

    if missing:
        print(f"\n  MISSING DATA ({len(missing)} pairs):")
        for pair in missing:
            print(f"    {pair['bull']}/{pair['bear']}: need {pair['bull_csv']} and {pair['bear_csv']}")

    if not available:
        print("\n  No pairs with complete data. Please add CSV files.")
        return

    # Run backtests
    print_header(f"RUNNING BACKTESTS — {len(available)} PAIRS")

    all_results = []
    for pair, bull_df, bear_df in available:
        print(f"  Testing {pair['bull']}/{pair['bear']}...")
        res = run_pair_backtest(pair, bull_df, bear_df)
        all_results.append(res)

    # Print individual results
    for res in all_results:
        print_pair_results(res)

    # Cross-pair comparison
    if len(all_results) > 1:
        print_cross_pair_summary(all_results)

    # Generate charts
    print_header("GENERATING CHARTS")

    viz = RotationVisualizer()

    # If multiple pairs, generate comparison chart
    v4_results = [r["v4"] for r in all_results]
    if len(v4_results) > 1:
        viz.plot_rotation_comparison(v4_results)
        print("  Saved: output/charts/rotation_strategy_comparison.png")

    # Individual pair dividend charts
    for res in all_results:
        pair = res["pair"]
        viz.plot_dividend_income_timeline(
            [res["v4"]], res["bull_df"], res["bear_df"]
        )

    print("\n  Done.")


if __name__ == "__main__":
    main()
