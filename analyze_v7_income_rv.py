#!/usr/bin/env python3
"""
V7 Income/RV Ratio Filter Analysis.

Tests V7 (Income/RV filter on bear switches) against V5 across all 4 pairs.

The key finding from trend analysis: at bear switch points, the Income/RV
ratio perfectly separates good switches from bad ones:
  - BAD switches: ratio < 1.1 (income below RV → vol spike, not a collapse)
  - GOOD switches: ratio > 1.8 (income exceeds RV → healthy premium, real decline)

Correlation: r=+0.82, p=0.013 for bear switches (n=8).

V7 blocks bear switches when Income/RV < min_ratio. Bear-to-bull always proceeds.
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


def get_adj_price(df):
    df_sorted = df.sort_index()
    if "split" in df_sorted.columns:
        cum_split = df_sorted["split"].replace(0, 1).cumprod()
        final_split = cum_split.iloc[-1]
        split_adj = cum_split / final_split
        return df_sorted["close"] / split_adj
    return df_sorted["close"]


def compute_rv(adj_price, window=20):
    log_ret = np.log(adj_price / adj_price.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252) * 100


def compute_income_yield(df, rolling_weeks=4):
    adj_price = get_adj_price(df)
    divs = df["dividend"] if "dividend" in df.columns else pd.Series(0, index=df.index)
    if "split" in df.columns:
        cum_split = df["split"].replace(0, 1).cumprod()
        final_split = cum_split.iloc[-1]
        split_adj = cum_split / final_split
        adj_divs = divs / split_adj
    else:
        adj_divs = divs
    rolling_days = rolling_weeks * 5
    rolling_div = adj_divs.rolling(rolling_days, min_periods=1).sum()
    return (rolling_div / adj_price) * (52 / rolling_weeks) * 100


def print_separator(char="=", width=110):
    print(char * width)


def main():
    print()
    print_separator()
    print("  V7 INCOME/RV RATIO FILTER — COMPARISON WITH V5")
    print_separator()
    print()
    print("  V5 (baseline):  SMA(50) ±5% band, 3-day debounce")
    print("  V7 (filtered):  Same as V5, but BLOCK bull→bear switches")
    print("                  when Income Yield / RV < threshold.")
    print("                  Low ratio = vol spike = income can't keep up = wrong time to switch.")
    print("                  High ratio = healthy premium engine = real structural decline.")
    print()
    print("  In-sample finding: ratio < 1.1 at switch → all BAD (avg -21.4%)")
    print("                     ratio > 1.8 at switch → all GOOD (avg +112.4%)")
    print("                     Correlation: r=+0.82, p=0.013")
    print()

    bt = RotationBacktester(windows=[5, 10, 20])

    # ================================================================
    # PART 1: Threshold sweep
    # ================================================================
    print_separator()
    print("  PART 1: INCOME/RV RATIO THRESHOLD SWEEP")
    print_separator()
    print()

    thresholds = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]

    # Header
    header = f"  {'Pair':<14}  {'V5':>8}"
    for t in thresholds:
        header += f"  {f'>{t:.1f}':>8}"
    print(header)
    print(f"  {'-' * (14 + 9 + 9 * len(thresholds))}")

    all_results = {}

    for pair in PAIRS:
        bull_df, bear_df = load_pair(pair)
        if bull_df is None:
            continue

        label = f"{pair['bull']}/{pair['bear']}"

        v5 = bt.run_v5_sma_hysteresis(
            bull_df, bear_df, sma_period=SMA_PERIOD,
            band_pct=5.0, debounce_days=DEBOUNCE_DAYS
        )

        row = f"  {label:<14}  {v5.total_return_pct:>+7.1f}%"

        pair_results = {"v5": v5, "pair": pair, "bull_df": bull_df, "bear_df": bear_df, "v7": {}}

        for t in thresholds:
            v7 = bt.run_v7_income_rv_filter(
                bull_df, bear_df, sma_period=SMA_PERIOD,
                band_pct=5.0, debounce_days=DEBOUNCE_DAYS,
                rv_window=20, income_weeks=4, min_ratio=t,
            )
            pair_results["v7"][t] = v7
            row += f"  {v7.total_return_pct:>+7.1f}%"

        all_results[label] = pair_results
        print(row)

    # Averages
    print(f"  {'-' * (14 + 9 + 9 * len(thresholds))}")
    avg_row = f"  {'AVERAGE':<14}"
    n = len(all_results)
    avg_v5 = sum(r["v5"].total_return_pct for r in all_results.values()) / n
    avg_row += f"  {avg_v5:>+7.1f}%"
    for t in thresholds:
        avg_v7 = sum(r["v7"][t].total_return_pct for r in all_results.values()) / n
        avg_row += f"  {avg_v7:>+7.1f}%"
    print(avg_row)

    # Delta row
    delta_row = f"  {'DELTA vs V5':<14}"
    delta_row += f"  {'---':>8}"
    for t in thresholds:
        avg_v7 = sum(r["v7"][t].total_return_pct for r in all_results.values()) / n
        delta = avg_v7 - avg_v5
        delta_row += f"  {delta:>+7.1f}%"
    print(delta_row)
    print()

    # ================================================================
    # PART 2: Best threshold detailed analysis
    # ================================================================
    # Find best threshold
    best_t = max(thresholds,
                 key=lambda t: sum(r["v7"][t].total_return_pct for r in all_results.values()))
    best_avg = sum(r["v7"][best_t].total_return_pct for r in all_results.values()) / n

    print_separator()
    print(f"  PART 2: V7(>{best_t:.1f}) vs V5 — DETAILED COMPARISON")
    print_separator()
    print()

    print(f"  {'Pair':<14}  {'V5 Return':>10}  {'V7 Return':>10}  {'Delta':>7}  "
          f"{'V5 Trades':>9}  {'V7 Trades':>9}  {'V5 MaxDD':>9}  {'V7 MaxDD':>9}")
    print(f"  {'-' * 90}")

    for label, data in all_results.items():
        v5 = data["v5"]
        v7 = data["v7"][best_t]
        v5_dd = max_dd(v5.daily_equity)
        v7_dd = max_dd(v7.daily_equity)
        delta = v7.total_return_pct - v5.total_return_pct
        print(f"  {label:<14}  {v5.total_return_pct:>+9.1f}%  {v7.total_return_pct:>+9.1f}%  "
              f"{delta:>+6.1f}%  {len(v5.trades):>9}  {len(v7.trades):>9}  "
              f"{v5_dd:>8.1f}%  {v7_dd:>8.1f}%")

    print()

    # ================================================================
    # PART 3: Trade-by-trade for each pair (V7 best threshold)
    # ================================================================
    print_separator()
    print(f"  PART 3: TRADE-BY-TRADE (V7 >{best_t:.1f})")
    print_separator()

    for label, data in all_results.items():
        v5 = data["v5"]
        v7 = data["v7"][best_t]
        pair = data["pair"]
        bull_name = pair["bull"]
        bear_name = pair["bear"]
        bull_df = data["bull_df"]

        adj_price = get_adj_price(bull_df)
        rv = compute_rv(adj_price, 20)
        income = compute_income_yield(bull_df, 4)
        ratio = income / rv.replace(0, np.nan)

        print()
        print(f"  --- {label} ({pair['name']}) ---")
        v7_dd = max_dd(v7.daily_equity)
        print(f"  V7 Total Return: {v7.total_return_pct:+.1f}% | Trades: {len(v7.trades)} | Max DD: {v7_dd:.1f}%")
        print()
        print(f"  {'#':>3}  {'Instr':>6}  {'Entry':>12}  {'Exit':>12}  {'Days':>5}  "
              f"{'PxRet':>8}  {'TotRet':>8}  {'Cumul':>8}  "
              f"{'RV':>6}  {'Income':>7}  {'Inc/RV':>7}")
        print(f"  {'-' * 100}")

        cumulative = 100.0
        for t in v7.trades:
            cumulative *= (1 + t.total_return_pct / 100)
            cum_ret = cumulative - 100
            inst = bull_name if t.instrument == "MSTY" else bear_name

            rv_at = rv[t.entry_date] if t.entry_date in rv.index else np.nan
            inc_at = income[t.entry_date] if t.entry_date in income.index else np.nan
            rat_at = ratio[t.entry_date] if t.entry_date in ratio.index else np.nan

            rv_str = f"{rv_at:.0f}%" if not pd.isna(rv_at) else "N/A"
            inc_str = f"{inc_at:.0f}%" if not pd.isna(inc_at) else "N/A"
            rat_str = f"{rat_at:.2f}" if not pd.isna(rat_at) else "N/A"

            print(f"  {t.trade_num:>3}  {inst:>6}  {t.entry_date.date()!s:>12}  "
                  f"{t.exit_date.date()!s:>12}  {t.holding_days:>5}  "
                  f"{t.price_return_pct:>+7.1f}%  {t.total_return_pct:>+7.1f}%  "
                  f"{cum_ret:>+7.1f}%  {rv_str:>6}  {inc_str:>7}  {rat_str:>7}")

        print()

        # Show disagreement with V5
        v5_pos = v5.daily_position
        v7_pos = v7.daily_position
        common = v5_pos.index.intersection(v7_pos.index)
        disagree = [d for d in common if v5_pos[d] != v7_pos[d]]

        if disagree:
            periods = []
            start = disagree[0]
            prev = disagree[0]
            for d in disagree[1:]:
                if (d - prev).days > 5:
                    periods.append((start, prev))
                    start = d
                prev = d
            periods.append((start, prev))

            print(f"  DISAGREEMENT WITH V5: {len(disagree)} days across {len(periods)} period(s):")
            calc = TotalReturnCalculator()
            bull_tri = calc.calculate_total_return_index(bull_df)
            bear_tri = calc.calculate_total_return_index(data["bear_df"])

            for s, e in periods:
                v5_side = bull_name if v5_pos[s] == "MSTY" else bear_name
                v7_side = bull_name if v7_pos[s] == "MSTY" else bear_name
                days = len([d for d in disagree if s <= d <= e])

                rat_at_s = ratio[s] if s in ratio.index else np.nan
                rat_str = f"{rat_at_s:.2f}" if not pd.isna(rat_at_s) else "N/A"

                print(f"    {s.date()} to {e.date()} ({days}d): V5={v5_side}, V7={v7_side}  "
                      f"Inc/RV at start: {rat_str}")

                if s in bull_tri.index and e in bull_tri.index:
                    bull_ret = (bull_tri[e] / bull_tri[s] - 1) * 100
                    bear_ret = (bear_tri[e] / bear_tri[s] - 1) * 100
                    winner = "V7" if (v7_side == bull_name and bull_ret > bear_ret) or \
                                (v7_side == bear_name and bear_ret > bull_ret) else "V5"
                    print(f"      {bull_name} TRI: {bull_ret:+.1f}%  |  {bear_name} TRI: {bear_ret:+.1f}%  "
                          f"→ Winner: {winner}")
            print()
        else:
            print(f"  V5 and V7 AGREE on every day")
            print()

    # ================================================================
    # PART 4: What Inc/RV ratio is at each V5 bear switch
    # ================================================================
    print_separator()
    print("  PART 4: INCOME/RV RATIO AT EVERY V5 BEAR SWITCH")
    print_separator()
    print()
    print("  Mapping the ratio at each bear switch to its outcome.")
    print("  This shows the separation that drives V7's filter.")
    print()

    print(f"  {'Pair':<14}  {'Date':>12}  {'Inc/RV':>7}  {'Trade Return':>12}  {'Quality':>8}  "
          f"{'Blocked by':>12}")
    print(f"  {'-' * 80}")

    for label, data in all_results.items():
        v5 = data["v5"]
        pair = data["pair"]
        bull_name = pair["bull"]
        bear_name = pair["bear"]
        bull_df = data["bull_df"]

        adj_price = get_adj_price(bull_df)
        rv = compute_rv(adj_price, 20)
        income = compute_income_yield(bull_df, 4)
        ratio = income / rv.replace(0, np.nan)

        for t in v5.trades:
            if t.instrument == "WNTR":  # Bear trade
                d = t.entry_date
                rat = ratio[d] if d in ratio.index else np.nan
                rat_str = f"{rat:.2f}" if not pd.isna(rat) else "N/A"

                quality = "GOOD" if t.total_return_pct > 10 else "BAD" if t.total_return_pct < -5 else "NEUTRAL"

                # Which thresholds would block this switch?
                blocked = []
                for thresh in [1.0, 1.2, 1.4, 1.6, 1.8]:
                    if not pd.isna(rat) and rat < thresh:
                        blocked.append(f">{thresh:.1f}")
                blocked_str = ", ".join(blocked) if blocked else "none"

                print(f"  {label:<14}  {d.date()!s:>12}  {rat_str:>7}  "
                      f"{t.total_return_pct:>+11.1f}%  {quality:>8}  {blocked_str:>12}")

    # ================================================================
    # PART 5: Current signals
    # ================================================================
    print()
    print_separator()
    print("  PART 5: CURRENT SIGNALS")
    print_separator()
    print()

    for label, data in all_results.items():
        v5 = data["v5"]
        v7 = data["v7"][best_t]
        pair = data["pair"]
        bull_name = pair["bull"]
        bear_name = pair["bear"]
        bull_df = data["bull_df"]

        adj_price = get_adj_price(bull_df)
        sma = adj_price.rolling(SMA_PERIOD, min_periods=SMA_PERIOD).mean()
        rv = compute_rv(adj_price, 20)
        income = compute_income_yield(bull_df, 4)
        ratio = income / rv.replace(0, np.nan)

        last = adj_price.index[-1]
        pct_sma = (adj_price[last] / sma[last] - 1) * 100
        rv_now = rv[last]
        inc_now = income[last]
        rat_now = ratio[last] if last in ratio.index else np.nan

        v5_pos = bull_name if v5.daily_position.iloc[-1] == "MSTY" else bear_name
        v7_pos = bull_name if v7.daily_position.iloc[-1] == "MSTY" else bear_name
        agree = "AGREE" if v5_pos == v7_pos else "DISAGREE"

        rat_str = f"{rat_now:.2f}" if not pd.isna(rat_now) else "N/A"
        filter_active = "YES" if not pd.isna(rat_now) and rat_now < best_t else "no"

        print(f"  {label:<14}  V5: {v5_pos:>5}  V7: {v7_pos:>5}  [{agree}]  "
              f"|  RV: {rv_now:.0f}%  Income: {inc_now:.0f}%  Inc/RV: {rat_str}  "
              f"|  %SMA: {pct_sma:+.1f}%  Filter active: {filter_active}")

    # ================================================================
    # PART 6: Summary
    # ================================================================
    print()
    print_separator()
    print("  PART 6: SUMMARY")
    print_separator()
    print()

    print(f"  V5 average:                     {avg_v5:+.1f}%")
    print(f"  V7(>{best_t:.1f}) average:            {best_avg:+.1f}%")
    print(f"  Delta:                          {best_avg - avg_v5:+.1f}%")
    print()

    # Count improvements
    improved = sum(1 for r in all_results.values()
                   if r["v7"][best_t].total_return_pct > r["v5"].total_return_pct + 0.1)
    unchanged = sum(1 for r in all_results.values()
                    if abs(r["v7"][best_t].total_return_pct - r["v5"].total_return_pct) <= 0.1)
    worse = sum(1 for r in all_results.values()
                if r["v7"][best_t].total_return_pct < r["v5"].total_return_pct - 0.1)

    print(f"  Pairs improved: {improved}  |  Unchanged: {unchanged}  |  Worse: {worse}")
    print()
    print("  INCOME/RV RATIO SIGNIFICANCE:")
    print("  - The ratio measures 'premium health': income yield / realized volatility")
    print("  - High ratio (>1.5): income exceeds realized risk → premium engine is healthy")
    print("  - Low ratio (<1.0): RV exceeds income → vol spike, income can't compensate")
    print("  - At bear switch points: r=+0.82, p=0.013 (n=8 bear switches)")
    print("  - All bad switches had ratio < 1.1, all good switches had ratio > 1.8")
    print()

    print_separator()
    print("  END OF V7 ANALYSIS")
    print_separator()
    print()


if __name__ == "__main__":
    main()
