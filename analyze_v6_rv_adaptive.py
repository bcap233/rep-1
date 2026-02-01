#!/usr/bin/env python3
"""
V6 RV-Adaptive Band Analysis.

Tests V6 (volatility-scaled band) against V5 (fixed band) across all 4 pairs.

The core idea: a fixed ±5% band means different things at different vol levels.
At 30% RV, ±5% is a ~2.6σ event. At 80% RV, ±5% is only ~1σ.
V6 scales the band by realized volatility so a breach always carries the
same statistical significance.

Band width = RV(20d, ann) / scalar, clamped to [floor, cap].
With scalar=10: RV 30% → ±3%, RV 50% → ±5%, RV 80% → ±8%.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.rotation_strategy import RotationBacktester
from src.counterparty_analysis import TotalReturnCalculator

PAIRS = [
    {"name": "MSTR (BTC Proxy)", "bull": "MSTY", "bear": "WNTR",
     "bull_csv": "msty_data_with_dividends.csv", "bear_csv": "wntr_data_with_dividends.csv"},
    {"name": "NVIDIA", "bull": "NVDY", "bear": "DIPS",
     "bull_csv": "nvdy_data_with_dividends.csv", "bear_csv": "dips_data_with_dividends.csv"},
    {"name": "Coinbase", "bull": "CONY", "bear": "FIAT",
     "bull_csv": "cony_data_with_dividends.csv", "bear_csv": "fiat_data_with_dividends.csv"},
    {"name": "Tesla", "bull": "TSLY", "bear": "CRSH",
     "bull_csv": "tsly_data_with_dividends.csv", "bear_csv": "crsh_data_with_dividends.csv"},
]

SMA_PERIOD = 50
DEBOUNCE_DAYS = 3


def load_pair(pair):
    bull_csv = Path(pair["bull_csv"])
    bear_csv = Path(pair["bear_csv"])
    if not bull_csv.exists() or not bear_csv.exists():
        return None, None
    bull_df = pd.read_csv(bull_csv, parse_dates=["Date"], index_col="Date").sort_index()
    bull_df.columns = [c.lower().strip() for c in bull_df.columns]
    bear_df = pd.read_csv(bear_csv, parse_dates=["Date"], index_col="Date").sort_index()
    bear_df.columns = [c.lower().strip() for c in bear_df.columns]
    start = max(bull_df.index.min(), bear_df.index.min())
    end = min(bull_df.index.max(), bear_df.index.max())
    return bull_df.loc[start:end], bear_df.loc[start:end]


def max_dd(equity):
    rm = equity.expanding().max()
    return ((equity / rm - 1) * 100).min()


def get_adj_price_and_rv(df, sma_period=50, rv_window=20):
    """Get adjusted price, SMA, and RV for display."""
    df_sorted = df.sort_index()
    if "split" in df_sorted.columns:
        cum_split = df_sorted["split"].replace(0, 1).cumprod()
        final_split = cum_split.iloc[-1]
        split_adj = cum_split / final_split
        adj_price = df_sorted["close"] / split_adj
    else:
        adj_price = df_sorted["close"]
    sma = adj_price.rolling(window=sma_period, min_periods=sma_period).mean()
    log_ret = np.log(adj_price / adj_price.shift(1))
    rv = log_ret.rolling(rv_window).std() * np.sqrt(252) * 100
    return adj_price, sma, rv


def print_separator(char="=", width=100):
    print(char * width)


def main():
    print()
    print_separator()
    print("  V6 RV-ADAPTIVE BAND — COMPARISON WITH V5")
    print_separator()
    print()
    print("  V5 (fixed):    Band = ±5% always")
    print("  V6 (adaptive): Band = RV(20d) / 10, clamped to [±2%, ±12%]")
    print("                 Low vol (RV=30%) → ±3%  (tighter, faster switch)")
    print("                 Mid vol (RV=50%) → ±5%  (same as V5)")
    print("                 High vol (RV=80%) → ±8% (wider, more patience)")
    print()

    bt = RotationBacktester(windows=[5, 10, 20])
    calc = TotalReturnCalculator()

    # ================================================================
    # PART 1: V5 vs V6 head-to-head across all pairs
    # ================================================================
    print_separator()
    print("  PART 1: V5 vs V6 HEAD-TO-HEAD")
    print_separator()
    print()

    results = {}

    for pair in PAIRS:
        bull_df, bear_df = load_pair(pair)
        if bull_df is None:
            continue

        label = f"{pair['bull']}/{pair['bear']}"

        # V5 fixed band
        v5 = bt.run_v5_sma_hysteresis(
            bull_df, bear_df, sma_period=SMA_PERIOD,
            band_pct=5.0, debounce_days=DEBOUNCE_DAYS
        )

        # V6 RV-adaptive (scalar=10, floor=2, cap=12)
        v6 = bt.run_v6_rv_adaptive(
            bull_df, bear_df, sma_period=SMA_PERIOD,
            rv_window=20, band_scalar=10.0,
            band_floor=2.0, band_cap=12.0,
            debounce_days=DEBOUNCE_DAYS
        )

        results[label] = {
            "v5": v5, "v6": v6,
            "v5_dd": max_dd(v5.daily_equity),
            "v6_dd": max_dd(v6.daily_equity),
            "pair": pair,
            "bull_df": bull_df,
            "bear_df": bear_df,
        }

    # Summary table
    print(f"  {'Pair':<14}  {'V5 Return':>10}  {'V6 Return':>10}  {'Delta':>7}  "
          f"{'V5 Trades':>9}  {'V6 Trades':>9}  {'V5 MaxDD':>9}  {'V6 MaxDD':>9}")
    print(f"  {'-' * 90}")

    total_v5 = total_v6 = 0
    n = 0

    for label, data in results.items():
        v5, v6 = data["v5"], data["v6"]
        delta = v6.total_return_pct - v5.total_return_pct
        print(f"  {label:<14}  {v5.total_return_pct:>+9.1f}%  {v6.total_return_pct:>+9.1f}%  "
              f"{delta:>+6.1f}%  {len(v5.trades):>9}  {len(v6.trades):>9}  "
              f"{data['v5_dd']:>8.1f}%  {data['v6_dd']:>8.1f}%")
        total_v5 += v5.total_return_pct
        total_v6 += v6.total_return_pct
        n += 1

    print(f"  {'-' * 90}")
    print(f"  {'AVERAGE':<14}  {total_v5/n:>+9.1f}%  {total_v6/n:>+9.1f}%  "
          f"{(total_v6-total_v5)/n:>+6.1f}%")
    print()

    # ================================================================
    # PART 2: V6 trade detail with daily band width
    # ================================================================
    print_separator()
    print("  PART 2: V6 TRADE HISTORY")
    print_separator()

    for label, data in results.items():
        v6 = data["v6"]
        pair = data["pair"]
        bull_name = pair["bull"]
        bear_name = pair["bear"]
        bull_df = data["bull_df"]

        adj_price, sma, rv = get_adj_price_and_rv(bull_df, SMA_PERIOD, 20)

        print()
        print(f"  --- {label} ({pair['name']}) ---")
        print(f"  Total Return: {v6.total_return_pct:+.1f}% | Trades: {len(v6.trades)} | Max DD: {data['v6_dd']:.1f}%")
        print()
        print(f"  {'#':>3}  {'Instr':>6}  {'Entry':>12}  {'Exit':>12}  {'Days':>5}  "
              f"{'PxRet':>8}  {'TotRet':>8}  {'Cumul':>8}  "
              f"{'Entry RV':>9}  {'Band@Entry':>11}")
        print(f"  {'-' * 100}")

        cumulative = 100.0
        for t in v6.trades:
            cumulative *= (1 + t.total_return_pct / 100)
            cum_ret = cumulative - 100
            inst = bull_name if t.instrument == "MSTY" else bear_name

            # RV and band at entry
            rv_at_entry = rv[t.entry_date] if t.entry_date in rv.index else np.nan
            band_at_entry = max(2.0, min(12.0, rv_at_entry / 10.0)) if not pd.isna(rv_at_entry) else np.nan

            rv_str = f"{rv_at_entry:.1f}%" if not pd.isna(rv_at_entry) else "N/A"
            band_str = f"±{band_at_entry:.1f}%" if not pd.isna(band_at_entry) else "N/A"

            print(f"  {t.trade_num:>3}  {inst:>6}  {t.entry_date.date()!s:>12}  "
                  f"{t.exit_date.date()!s:>12}  {t.holding_days:>5}  "
                  f"{t.price_return_pct:>+7.1f}%  {t.total_return_pct:>+7.1f}%  "
                  f"{cum_ret:>+7.1f}%  {rv_str:>9}  {band_str:>11}")

        print()

    # ================================================================
    # PART 3: V6 daily band width evolution
    # ================================================================
    print_separator()
    print("  PART 3: DAILY BAND WIDTH — HOW V6 ADAPTED")
    print_separator()
    print()

    for label, data in results.items():
        pair = data["pair"]
        bull_df = data["bull_df"]
        bull_name = pair["bull"]

        adj_price, sma, rv = get_adj_price_and_rv(bull_df, SMA_PERIOD, 20)

        # Sample monthly
        rv_valid = rv.dropna()
        sma_valid = sma.dropna()
        dates = rv_valid.index.intersection(sma_valid.index)
        if len(dates) == 0:
            continue

        monthly = [dates[0]]
        for i in range(1, len(dates)):
            if dates[i].month != dates[i-1].month:
                monthly.append(dates[i])
        monthly.append(dates[-1])

        print(f"  {label} ({bull_name}) — Monthly band width snapshots:")
        print()
        print(f"  {'Date':>12}  {'Price':>8}  {'%SMA':>8}  {'RV(20d)':>8}  "
              f"{'V6 Band':>8}  {'V5 Band':>8}  {'Regime'}")
        print(f"  {'-' * 70}")

        for d in monthly:
            price = adj_price[d] if d in adj_price.index else np.nan
            sma_val = sma[d] if d in sma.index else np.nan
            pct_sma = (price / sma_val - 1) * 100 if not pd.isna(sma_val) else np.nan
            rv_val = rv[d]
            v6_band = max(2.0, min(12.0, rv_val / 10.0))

            regime = ""
            if not pd.isna(pct_sma):
                if abs(pct_sma) <= v6_band:
                    regime = "HOLD(V6)"
                elif pct_sma > v6_band:
                    regime = "BULL"
                else:
                    regime = "BEAR"

            pct_str = f"{pct_sma:+.1f}%" if not pd.isna(pct_sma) else "N/A"

            print(f"  {d.date()!s:>12}  ${price:>7.2f}  {pct_str:>8}  "
                  f"{rv_val:>7.1f}%  ±{v6_band:>5.1f}%  ± 5.0%  {regime}")

        print()

    # ================================================================
    # PART 4: Scalar sensitivity sweep
    # ================================================================
    print_separator()
    print("  PART 4: BAND SCALAR SENSITIVITY")
    print_separator()
    print()
    print("  Testing different scalars: band = RV / scalar")
    print("  Scalar 8 = wider bands (more patience)")
    print("  Scalar 10 = baseline")
    print("  Scalar 12 = tighter bands (faster switching)")
    print()

    scalars = [8, 10, 12]

    print(f"  {'Pair':<14}", end="")
    for s in scalars:
        print(f"  {'Sc=' + str(s):>10}", end="")
    print(f"  {'V5 (±5%)':>10}")
    print(f"  {'-' * 60}")

    for pair in PAIRS:
        bull_df, bear_df = load_pair(pair)
        if bull_df is None:
            continue

        label = f"{pair['bull']}/{pair['bear']}"
        v5 = results[label]["v5"]

        print(f"  {label:<14}", end="")
        for s in scalars:
            v6_test = bt.run_v6_rv_adaptive(
                bull_df, bear_df, sma_period=SMA_PERIOD,
                rv_window=20, band_scalar=s,
                band_floor=2.0, band_cap=12.0,
                debounce_days=DEBOUNCE_DAYS
            )
            print(f"  {v6_test.total_return_pct:>+9.1f}%", end="")
        print(f"  {v5.total_return_pct:>+9.1f}%")

    print()

    # ================================================================
    # PART 5: Key comparison — where V6 differs from V5
    # ================================================================
    print_separator()
    print("  PART 5: WHERE V6 DIFFERS FROM V5")
    print_separator()
    print()

    for label, data in results.items():
        v5 = data["v5"]
        v6 = data["v6"]
        pair = data["pair"]
        bull_name = pair["bull"]
        bear_name = pair["bear"]

        # Find dates where V5 and V6 hold different positions
        v5_pos = v5.daily_position
        v6_pos = v6.daily_position

        # Align date ranges
        common = v5_pos.index.intersection(v6_pos.index)
        disagree = [d for d in common if v5_pos[d] != v6_pos[d]]

        if not disagree:
            print(f"  {label}: V5 and V6 agree on every day — identical signals")
        else:
            # Group into contiguous periods
            periods = []
            start = disagree[0]
            prev = disagree[0]
            for d in disagree[1:]:
                if (d - prev).days > 5:  # Gap > 5 days = new period
                    periods.append((start, prev))
                    start = d
                prev = d
            periods.append((start, prev))

            print(f"  {label}: V5 and V6 DISAGREE on {len(disagree)} days across {len(periods)} period(s):")
            for s, e in periods:
                v5_side = bull_name if v5_pos[s] == "MSTY" else bear_name
                v6_side = bull_name if v6_pos[s] == "MSTY" else bear_name
                days = len([d for d in disagree if s <= d <= e])
                print(f"    {s.date()} to {e.date()} ({days} days): V5={v5_side}, V6={v6_side}")
        print()

    # ================================================================
    # PART 6: Current signals comparison
    # ================================================================
    print_separator()
    print("  PART 6: CURRENT SIGNALS")
    print_separator()
    print()

    for label, data in results.items():
        v5 = data["v5"]
        v6 = data["v6"]
        pair = data["pair"]
        bull_name = pair["bull"]
        bear_name = pair["bear"]
        bull_df = data["bull_df"]

        adj_price, sma, rv = get_adj_price_and_rv(bull_df, SMA_PERIOD, 20)
        last = adj_price.index[-1]
        rv_now = rv[last]
        band_now = max(2.0, min(12.0, rv_now / 10.0))
        pct_sma = (adj_price[last] / sma[last] - 1) * 100

        v5_pos = bull_name if v5.daily_position.iloc[-1] == "MSTY" else bear_name
        v6_pos = bull_name if v6.daily_position.iloc[-1] == "MSTY" else bear_name
        agree = "AGREE" if v5_pos == v6_pos else "DISAGREE"

        print(f"  {label:<14}  V5: {v5_pos:>5} (±5.0%)  |  V6: {v6_pos:>5} (±{band_now:.1f}%)  "
              f"|  RV: {rv_now:.1f}%  %SMA: {pct_sma:+.1f}%  [{agree}]")

    print()
    print_separator()
    print("  END OF V6 ANALYSIS")
    print_separator()
    print()


if __name__ == "__main__":
    main()
