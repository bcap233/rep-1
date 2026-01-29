#!/usr/bin/env python3
"""
V5 Hysteresis Band Analysis.

Tests the always-invested rotation strategy with a dead-zone band around the SMA.
The band prevents switches when price is just oscillating near the SMA (chop).

Sweeps band widths: 0% (=V4), 2%, 3%, 5%, 8%
Tests across all 4 YieldMax pairs.
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

BAND_WIDTHS = [0, 2, 3, 5, 8]  # 0% = same as V4


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


def main():
    print("=" * 100)
    print("  V5 HYSTERESIS BAND SWEEP — Does a Dead Zone Fix Chop?")
    print("=" * 100)
    print()
    print("  Concept: V4 switches when price crosses the SMA line.")
    print("  V5 adds a BAND around the SMA — only switch when price moves")
    print("  decisively OUTSIDE the band. Inside the band = stay put.")
    print()
    print("  Band widths tested: 0% (=V4), 2%, 3%, 5%, 8%")
    print()

    bt = RotationBacktester(windows=[5, 10, 20])
    calc = TotalReturnCalculator()

    # Store all results: results[pair_label][band_pct] = {...}
    all_results = {}

    for pair in PAIRS:
        bull_df, bear_df = load_pair(pair)
        if bull_df is None:
            print(f"  SKIP: {pair['bull']}/{pair['bear']} — data missing")
            continue

        label = f"{pair['bull']}/{pair['bear']}"
        print(f"  Running {label}...")
        all_results[label] = {}

        # Buy-and-hold baselines
        bull_tri = calc.calculate_total_return_index(bull_df)
        bear_tri = calc.calculate_total_return_index(bear_df)
        bull_bh = (bull_tri.iloc[-1] / bull_tri.iloc[0] - 1) * 100
        bear_bh = (bear_tri.iloc[-1] / bear_tri.iloc[0] - 1) * 100

        all_results[label]["bull_bh"] = bull_bh
        all_results[label]["bear_bh"] = bear_bh

        for bw in BAND_WIDTHS:
            if bw == 0:
                # 0% band = V4 (no hysteresis)
                res = bt.run_v4_sma_plus_debounce(bull_df, bear_df, sma_period=50, debounce_days=3)
            else:
                res = bt.run_v5_sma_hysteresis(bull_df, bear_df, sma_period=50, band_pct=bw, debounce_days=3)

            dd = max_dd(res.daily_equity)
            all_results[label][bw] = {
                "return": res.total_return_pct,
                "trades": len(res.trades),
                "max_dd": dd,
                "bull_days": res.msty_days,
                "bear_days": res.wntr_days,
                "divs": res.total_dividends_collected,
                "result": res,
            }

    # ================================================================
    # PRINT RESULTS
    # ================================================================

    # Per-pair band sweep tables
    for label, data in all_results.items():
        bull_bh = data["bull_bh"]
        bear_bh = data["bear_bh"]
        bull_name = label.split("/")[0]
        bear_name = label.split("/")[1]

        print()
        print("=" * 95)
        print(f"  {label} — Band Width Sweep")
        print("=" * 95)
        print(f"  Baselines: {bull_name} B&H = {bull_bh:+.1f}%, {bear_name} B&H = {bear_bh:+.1f}%")
        print()
        print(f"  {'Band':>6}  {'Return':>10}  {'vs V4':>8}  {'Trades':>7}  "
              f"{'MaxDD':>8}  {'Bull%':>7}  {'Bear%':>7}  {'Divs':>8}")
        print(f"  {'-' * 75}")

        v4_ret = data[0]["return"]

        for bw in BAND_WIDTHS:
            r = data[bw]
            total_days = r["bull_days"] + r["bear_days"]
            bull_pct = r["bull_days"] / total_days * 100 if total_days > 0 else 0
            bear_pct = r["bear_days"] / total_days * 100 if total_days > 0 else 0
            vs_v4 = r["return"] - v4_ret

            band_label = "V4 (0%)" if bw == 0 else f"±{bw}%"
            print(
                f"  {band_label:>6}  "
                f"{r['return']:>+9.1f}%  "
                f"{vs_v4:>+7.1f}%  "
                f"{r['trades']:>7}  "
                f"{r['max_dd']:>7.1f}%  "
                f"{bull_pct:>6.0f}%  "
                f"{bear_pct:>6.0f}%  "
                f"${r['divs']:>7.2f}"
            )

    # ================================================================
    # CROSS-PAIR SUMMARY — Best band per pair
    # ================================================================
    print()
    print("=" * 95)
    print("  CROSS-PAIR SUMMARY — Which Band Width Wins?")
    print("=" * 95)
    print()
    print(f"  {'Pair':<14}  {'V4 (0%)':>10}  {'±2%':>10}  {'±3%':>10}  {'±5%':>10}  {'±8%':>10}  {'Best':>8}  {'Bull BH':>10}")
    print(f"  {'-' * 95}")

    total_by_band = {bw: 0 for bw in BAND_WIDTHS}

    for label, data in all_results.items():
        row = f"  {label:<14}"
        best_bw = 0
        best_ret = -999

        for bw in BAND_WIDTHS:
            ret = data[bw]["return"]
            row += f"  {ret:>+9.1f}%"
            total_by_band[bw] += ret
            if ret > best_ret:
                best_ret = ret
                best_bw = bw

        best_label = "V4" if best_bw == 0 else f"±{best_bw}%"
        row += f"  {best_label:>8}"
        row += f"  {data['bull_bh']:>+9.1f}%"
        print(row)

    n_pairs = len(all_results)
    print(f"  {'-' * 95}")
    row = f"  {'AVERAGE':<14}"
    best_avg_bw = 0
    best_avg = -999
    for bw in BAND_WIDTHS:
        avg = total_by_band[bw] / n_pairs
        row += f"  {avg:>+9.1f}%"
        if avg > best_avg:
            best_avg = avg
            best_avg_bw = bw
    best_label = "V4" if best_avg_bw == 0 else f"±{best_avg_bw}%"
    row += f"  {best_label:>8}"
    print(row)

    # ================================================================
    # TRADE COUNT COMPARISON — Show the chop reduction
    # ================================================================
    print()
    print("=" * 95)
    print("  TRADE COUNT BY BAND WIDTH — Chop Reduction")
    print("=" * 95)
    print()
    print(f"  {'Pair':<14}  {'V4 (0%)':>8}  {'±2%':>8}  {'±3%':>8}  {'±5%':>8}  {'±8%':>8}")
    print(f"  {'-' * 60}")

    for label, data in all_results.items():
        row = f"  {label:<14}"
        for bw in BAND_WIDTHS:
            row += f"  {data[bw]['trades']:>8}"
        print(row)

    # ================================================================
    # DRAWDOWN COMPARISON
    # ================================================================
    print()
    print("=" * 95)
    print("  MAX DRAWDOWN BY BAND WIDTH")
    print("=" * 95)
    print()
    print(f"  {'Pair':<14}  {'V4 (0%)':>8}  {'±2%':>8}  {'±3%':>8}  {'±5%':>8}  {'±8%':>8}")
    print(f"  {'-' * 60}")

    for label, data in all_results.items():
        row = f"  {label:<14}"
        for bw in BAND_WIDTHS:
            row += f"  {data[bw]['max_dd']:>7.1f}%"
        print(row)

    # ================================================================
    # TSLY/CRSH DEEP DIVE — Trade log comparison
    # ================================================================
    if "TSLY/CRSH" in all_results:
        print()
        print("=" * 95)
        print("  TSLY/CRSH DEEP DIVE — V4 vs Best V5 Band")
        print("=" * 95)

        tsly_data = all_results["TSLY/CRSH"]

        # Find best band for TSLY
        best_bw = max(BAND_WIDTHS, key=lambda bw: tsly_data[bw]["return"])
        best_label = "V4" if best_bw == 0 else f"±{best_bw}%"

        print(f"\n  V4 (0% band): {tsly_data[0]['return']:+.1f}% return, {tsly_data[0]['trades']} trades, {tsly_data[0]['max_dd']:.1f}% max DD")
        print(f"  Best ({best_label}):    {tsly_data[best_bw]['return']:+.1f}% return, {tsly_data[best_bw]['trades']} trades, {tsly_data[best_bw]['max_dd']:.1f}% max DD")
        print(f"  Improvement:    {tsly_data[best_bw]['return'] - tsly_data[0]['return']:+.1f}%")

        # Print trade logs for V4 and best
        for bw_show in [0, best_bw]:
            if bw_show == 0 and best_bw == 0:
                continue  # Don't print twice
            res = tsly_data[bw_show]["result"]
            bw_label = "V4 (0%)" if bw_show == 0 else f"V5 (±{bw_show}%)"
            print(f"\n  --- {bw_label} Trade Log ---")
            print(f"  {'#':>3}  {'Instr':>6}  {'Entry':>12}  {'Exit':>12}  {'Days':>5}  {'TotRet':>8}  {'PxRet':>8}")
            print(f"  {'-' * 65}")
            for t in res.trades:
                inst = "TSLY" if t.instrument == "MSTY" else "CRSH"
                print(f"  {t.trade_num:>3}  {inst:>6}  {t.entry_date.date()!s:>12}  "
                      f"{t.exit_date.date()!s:>12}  {t.holding_days:>5}  "
                      f"{t.total_return_pct:>+7.1f}%  {t.price_return_pct:>+7.1f}%")

    print()
    print("=" * 95)
    print("  CONCLUSION")
    print("=" * 95)
    best_label = "V4" if best_avg_bw == 0 else f"V5 ±{best_avg_bw}%"
    print(f"\n  Best average band: {best_label} at {best_avg:+.1f}% avg return")
    v4_avg = total_by_band[0] / n_pairs
    if best_avg_bw != 0:
        print(f"  V4 average: {v4_avg:+.1f}% — improvement: {best_avg - v4_avg:+.1f}%")
    print()


if __name__ == "__main__":
    main()
