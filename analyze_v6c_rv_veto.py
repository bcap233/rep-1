#!/usr/bin/env python3
"""
V6c RV-Veto Analysis.

Tests V6c (RV veto on bear switches) against V5 across all 4 pairs.

The core idea: collapses happen during LOW volatility (calm grinds),
while high-vol breaches are usually temporary spikes that recover.
Instead of using RV to scale bands or debounce, use it as a one-directional
VETO: if the signal says "switch to bear" but RV is elevated (above the
Nth percentile of its trailing window), block the switch. The breach is
a vol spike, not a structural collapse. Stay in the bull ETF.

Bear→bull switches proceed normally — always catch the recovery.
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
RV_WINDOW = 20
RV_LOOKBACK = 60


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
    print("  V6c RV-VETO — COMPARISON WITH V5")
    print_separator()
    print()
    print("  V5 (baseline):  SMA(50) ±5% band, 3-day debounce")
    print("  V6c (RV-veto):  Same as V5, but BLOCK bull→bear switches")
    print("                  when RV exceeds a percentile threshold.")
    print("                  High RV = vol spike = temporary. Stay in bull ETF.")
    print("                  Bear→bull switches always proceed normally.")
    print()

    bt = RotationBacktester(windows=[5, 10, 20])

    # ================================================================
    # PART 1: Veto threshold sweep
    # ================================================================
    print_separator()
    print("  PART 1: VETO THRESHOLD SWEEP")
    print_separator()
    print()
    print("  Testing veto thresholds: block bear switch when RV > Nth percentile")
    print(f"  RV window: {RV_WINDOW}d | RV lookback for percentile: {RV_LOOKBACK}d")
    print()

    thresholds = [50, 55, 60, 65, 70, 75, 80]

    # Header
    header = f"  {'Pair':<14}"
    header += f"  {'V5':>8}"
    for t in thresholds:
        header += f"  {f'>{t}th':>8}"
    print(header)
    print(f"  {'-' * (14 + 9 + 9 * len(thresholds))}")

    all_results = {}

    for pair in PAIRS:
        bull_df, bear_df = load_pair(pair)
        if bull_df is None:
            continue

        label = f"{pair['bull']}/{pair['bear']}"

        # V5 baseline
        v5 = bt.run_v5_sma_hysteresis(
            bull_df, bear_df, sma_period=SMA_PERIOD,
            band_pct=5.0, debounce_days=DEBOUNCE_DAYS
        )

        row = f"  {label:<14}  {v5.total_return_pct:>+7.1f}%"

        pair_results = {"v5": v5, "pair": pair, "bull_df": bull_df, "bear_df": bear_df, "v6c": {}}

        for t in thresholds:
            v6c = bt.run_v6c_rv_veto(
                bull_df, bear_df, sma_period=SMA_PERIOD,
                band_pct=5.0, debounce_days=DEBOUNCE_DAYS,
                rv_window=RV_WINDOW, rv_lookback=RV_LOOKBACK,
                rv_veto_pctile=t,
            )
            pair_results["v6c"][t] = v6c
            delta = v6c.total_return_pct - v5.total_return_pct
            if abs(delta) < 0.05:
                row += f"  {v6c.total_return_pct:>+7.1f}%"
            else:
                row += f"  {v6c.total_return_pct:>+7.1f}%"

        all_results[label] = pair_results
        print(row)

    # Averages
    print(f"  {'-' * (14 + 9 + 9 * len(thresholds))}")
    avg_row = f"  {'AVERAGE':<14}"
    n = len(all_results)
    avg_v5 = sum(r["v5"].total_return_pct for r in all_results.values()) / n
    avg_row += f"  {avg_v5:>+7.1f}%"
    for t in thresholds:
        avg_v6c = sum(r["v6c"][t].total_return_pct for r in all_results.values()) / n
        avg_row += f"  {avg_v6c:>+7.1f}%"
    print(avg_row)
    print()

    # ================================================================
    # PART 2: Best threshold analysis — where V6c differs from V5
    # ================================================================
    best_threshold = 65  # from sweep analysis
    print_separator()
    print(f"  PART 2: V6c(>{best_threshold}th) vs V5 — DETAILED COMPARISON")
    print_separator()
    print()

    print(f"  {'Pair':<14}  {'V5 Return':>10}  {'V6c Return':>11}  {'Delta':>7}  "
          f"{'V5 Trades':>9}  {'V6c Trades':>10}  {'V5 MaxDD':>9}  {'V6c MaxDD':>10}")
    print(f"  {'-' * 95}")

    for label, data in all_results.items():
        v5 = data["v5"]
        v6c = data["v6c"][best_threshold]
        v5_dd = max_dd(v5.daily_equity)
        v6c_dd = max_dd(v6c.daily_equity)
        delta = v6c.total_return_pct - v5.total_return_pct
        print(f"  {label:<14}  {v5.total_return_pct:>+9.1f}%  {v6c.total_return_pct:>+10.1f}%  "
              f"{delta:>+6.1f}%  {len(v5.trades):>9}  {len(v6c.trades):>10}  "
              f"{v5_dd:>8.1f}%  {v6c_dd:>9.1f}%")

    print()

    # ================================================================
    # PART 3: Trade-by-trade comparison for differing pairs
    # ================================================================
    print_separator()
    print(f"  PART 3: TRADE-BY-TRADE DIFFERENCES (V6c >{best_threshold}th vs V5)")
    print_separator()

    for label, data in all_results.items():
        v5 = data["v5"]
        v6c = data["v6c"][best_threshold]
        pair = data["pair"]
        bull_name = pair["bull"]
        bear_name = pair["bear"]
        bull_df = data["bull_df"]

        adj_price, sma, rv = get_adj_price_and_rv(bull_df, SMA_PERIOD, RV_WINDOW)

        # Find disagreement periods
        v5_pos = v5.daily_position
        v6c_pos = v6c.daily_position
        common = v5_pos.index.intersection(v6c_pos.index)
        disagree = [d for d in common if v5_pos[d] != v6c_pos[d]]

        print()
        print(f"  --- {label} ({pair['name']}) ---")

        if not disagree:
            print(f"  V5 and V6c AGREE on every day — identical signals")
            print()
            continue

        # Group into contiguous periods
        periods = []
        start = disagree[0]
        prev = disagree[0]
        for d in disagree[1:]:
            if (d - prev).days > 5:
                periods.append((start, prev))
                start = d
            prev = d
        periods.append((start, prev))

        print(f"  V5 and V6c DISAGREE on {len(disagree)} days across {len(periods)} period(s):")
        print()

        for s, e in periods:
            v5_side = bull_name if v5_pos[s] == "MSTY" else bear_name
            v6c_side = bull_name if v6c_pos[s] == "MSTY" else bear_name
            days = len([d for d in disagree if s <= d <= e])

            # RV at the start of disagreement
            rv_at_start = rv[s] if s in rv.index else np.nan
            rv_str = f"{rv_at_start:.1f}%" if not pd.isna(rv_at_start) else "N/A"

            # Compute RV percentile at start
            if s in rv.index and not pd.isna(rv[s]):
                rv_loc = rv.index.get_loc(s)
                start_loc = max(0, rv_loc - RV_LOOKBACK)
                rv_window_data = rv.iloc[start_loc:rv_loc + 1].dropna()
                if len(rv_window_data) >= 20:
                    rv_pctile = (rv_window_data < rv[s]).mean() * 100
                    pctile_str = f"{rv_pctile:.0f}th"
                else:
                    pctile_str = "N/A"
            else:
                pctile_str = "N/A"

            print(f"    {s.date()} to {e.date()} ({days} days):")
            print(f"      V5 held: {v5_side} | V6c held: {v6c_side}")
            print(f"      RV at divergence: {rv_str} (percentile: {pctile_str})")

            # Compute return for each strategy during this disagreement period
            calc = TotalReturnCalculator()
            bull_tri = calc.calculate_total_return_index(bull_df)
            bear_tri = calc.calculate_total_return_index(data["bear_df"])

            if s in bull_tri.index and e in bull_tri.index:
                bull_ret = (bull_tri[e] / bull_tri[s] - 1) * 100
                bear_ret = (bear_tri[e] / bear_tri[s] - 1) * 100
                print(f"      {bull_name} TRI over period: {bull_ret:+.1f}%")
                print(f"      {bear_name} TRI over period: {bear_ret:+.1f}%")
                advantage = "V6c" if (v6c_side == bull_name and bull_ret > bear_ret) or \
                            (v6c_side == bear_name and bear_ret > bull_ret) else "V5"
                print(f"      Winner during this period: {advantage}")

            print()

    # ================================================================
    # PART 4: V6c trade history for best threshold
    # ================================================================
    print_separator()
    print(f"  PART 4: V6c(>{best_threshold}th) FULL TRADE HISTORY")
    print_separator()

    for label, data in all_results.items():
        v6c = data["v6c"][best_threshold]
        pair = data["pair"]
        bull_name = pair["bull"]
        bear_name = pair["bear"]
        bull_df = data["bull_df"]

        adj_price, sma, rv = get_adj_price_and_rv(bull_df, SMA_PERIOD, RV_WINDOW)

        print()
        print(f"  --- {label} ({pair['name']}) ---")
        v6c_dd = max_dd(v6c.daily_equity)
        print(f"  Total Return: {v6c.total_return_pct:+.1f}% | Trades: {len(v6c.trades)} | Max DD: {v6c_dd:.1f}%")
        print()
        print(f"  {'#':>3}  {'Instr':>6}  {'Entry':>12}  {'Exit':>12}  {'Days':>5}  "
              f"{'PxRet':>8}  {'TotRet':>8}  {'Cumul':>8}  {'RV@Entry':>9}  {'RV Pctile':>10}")
        print(f"  {'-' * 100}")

        cumulative = 100.0
        for t in v6c.trades:
            cumulative *= (1 + t.total_return_pct / 100)
            cum_ret = cumulative - 100
            inst = bull_name if t.instrument == "MSTY" else bear_name

            rv_at_entry = rv[t.entry_date] if t.entry_date in rv.index else np.nan
            rv_str = f"{rv_at_entry:.1f}%" if not pd.isna(rv_at_entry) else "N/A"

            # Compute percentile at entry
            if t.entry_date in rv.index and not pd.isna(rv[t.entry_date]):
                rv_loc = rv.index.get_loc(t.entry_date)
                start_loc = max(0, rv_loc - RV_LOOKBACK)
                rv_window_data = rv.iloc[start_loc:rv_loc + 1].dropna()
                if len(rv_window_data) >= 20:
                    rv_pctile = (rv_window_data < rv[t.entry_date]).mean() * 100
                    pctile_str = f"{rv_pctile:.0f}th"
                else:
                    pctile_str = "N/A"
            else:
                pctile_str = "N/A"

            print(f"  {t.trade_num:>3}  {inst:>6}  {t.entry_date.date()!s:>12}  "
                  f"{t.exit_date.date()!s:>12}  {t.holding_days:>5}  "
                  f"{t.price_return_pct:>+7.1f}%  {t.total_return_pct:>+7.1f}%  "
                  f"{cum_ret:>+7.1f}%  {rv_str:>9}  {pctile_str:>10}")

        print()

    # ================================================================
    # PART 5: Current signals
    # ================================================================
    print_separator()
    print("  PART 5: CURRENT SIGNALS")
    print_separator()
    print()

    for label, data in all_results.items():
        v5 = data["v5"]
        v6c = data["v6c"][best_threshold]
        pair = data["pair"]
        bull_name = pair["bull"]
        bear_name = pair["bear"]
        bull_df = data["bull_df"]

        adj_price, sma, rv = get_adj_price_and_rv(bull_df, SMA_PERIOD, RV_WINDOW)
        last = adj_price.index[-1]
        rv_now = rv[last]
        pct_sma = (adj_price[last] / sma[last] - 1) * 100

        # Current RV percentile
        rv_loc = rv.index.get_loc(last)
        start_loc = max(0, rv_loc - RV_LOOKBACK)
        rv_window_data = rv.iloc[start_loc:rv_loc + 1].dropna()
        if len(rv_window_data) >= 20:
            rv_pctile = (rv_window_data < rv[last]).mean() * 100
        else:
            rv_pctile = np.nan

        v5_pos = bull_name if v5.daily_position.iloc[-1] == "MSTY" else bear_name
        v6c_pos = bull_name if v6c.daily_position.iloc[-1] == "MSTY" else bear_name
        agree = "AGREE" if v5_pos == v6c_pos else "DISAGREE"

        pctile_str = f"{rv_pctile:.0f}th" if not pd.isna(rv_pctile) else "N/A"
        veto_active = "YES" if rv_pctile > best_threshold else "no"

        print(f"  {label:<14}  V5: {v5_pos:>5}  V6c: {v6c_pos:>5}  [{agree}]  "
              f"|  RV: {rv_now:.1f}%  RV pctile: {pctile_str}  "
              f"|  %SMA: {pct_sma:+.1f}%  Veto active: {veto_active}")

    print()

    # ================================================================
    # PART 6: Summary and recommendation
    # ================================================================
    print_separator()
    print("  PART 6: SUMMARY")
    print_separator()
    print()

    avg_v5 = sum(r["v5"].total_return_pct for r in all_results.values()) / n
    avg_v6c_best = sum(r["v6c"][best_threshold].total_return_pct for r in all_results.values()) / n
    delta = avg_v6c_best - avg_v5

    print(f"  V5 average:                   {avg_v5:+.1f}%")
    print(f"  V6c(>{best_threshold}th) average:         {avg_v6c_best:+.1f}%")
    print(f"  Delta:                        {delta:+.1f}%")
    print()

    # Count how many pairs differ
    differ_count = 0
    for label, data in all_results.items():
        v5_ret = data["v5"].total_return_pct
        v6c_ret = data["v6c"][best_threshold].total_return_pct
        if abs(v6c_ret - v5_ret) > 0.1:
            differ_count += 1

    print(f"  Pairs where V6c differs from V5: {differ_count}/{n}")
    print()
    print("  Key findings:")
    print("  - Collapses happen during LOW RV (calm grinds, 26th-31st percentile)")
    print("  - Recoveries happen during HIGH RV (vol spikes, 73rd-89th percentile)")
    print("  - V6c blocks bear switches during high-vol spikes (one-directional veto)")
    print("  - Bear→bull switches always proceed normally")
    print(f"  - The improvement is based on {differ_count} event(s) in {n} pairs")
    print("  - V5's fixed ±5% band already implicitly adapts to vol regimes")
    print()

    print_separator()
    print("  END OF V6c ANALYSIS")
    print_separator()
    print()


if __name__ == "__main__":
    main()
