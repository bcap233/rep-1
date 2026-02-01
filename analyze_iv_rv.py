#!/usr/bin/env python3
"""
IV/RV Analysis for YieldMax Covered Call Rotation Strategy.

Computes:
- Realized Volatility (RV): 20-day rolling std of daily log returns, annualized
- Income Yield (IV proxy): Rolling 4-week dividend sum / price, annualized
  (dividend yield is a proxy for implied vol since premiums come from selling options)
- Income/RV ratio: When income yield > RV, the covered call engine is working.
  When RV overwhelms income yield, the structure is breaking down.

Overlays these metrics on:
- V5 strategy switch points
- SMA breach events
- Collapse episodes vs recoveries
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
BAND_PCT = 5
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


def compute_rv(prices, window=20):
    """Compute realized volatility: rolling std of log returns, annualized."""
    log_ret = np.log(prices / prices.shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(252) * 100
    return rv


def compute_income_yield(df, window_weeks=4):
    """
    Compute rolling income yield (IV proxy).
    Sum dividends over last N weeks, divide by current price, annualize.
    """
    divs = df["dividend"].copy()
    prices = df["close"].copy()

    # Rolling sum of dividends over window_weeks * 5 trading days
    window_days = window_weeks * 5
    rolling_divs = divs.rolling(window_days, min_periods=1).sum()

    # Annualize: (rolling_divs / window_weeks) * 52 / price * 100
    annualized_yield = (rolling_divs / window_weeks) * 52 / prices * 100

    return annualized_yield


def get_adj_price_and_sma(df, sma_period=50):
    """Get split-adjusted price and SMA for an ETF."""
    df_sorted = df.sort_index()
    if "split" in df_sorted.columns:
        cum_split = df_sorted["split"].replace(0, 1).cumprod()
        final_split = cum_split.iloc[-1]
        split_adj = cum_split / final_split
        adj_price = df_sorted["close"] / split_adj
    else:
        adj_price = df_sorted["close"]
    sma = adj_price.rolling(window=sma_period, min_periods=sma_period).mean()
    return adj_price, sma


def find_switch_dates(result):
    """Find dates where the strategy switched positions."""
    pos = result.daily_position
    switches = []
    for i in range(1, len(pos)):
        if pos.iloc[i] != pos.iloc[i - 1]:
            switches.append({
                "date": pos.index[i],
                "from": pos.iloc[i - 1],
                "to": pos.iloc[i],
            })
    return switches


def find_sma_breach_events(df, sma_period=50):
    """Find all dates where price crosses below SMA."""
    adj_price, sma = get_adj_price_and_sma(df, sma_period)
    sma_valid = sma.dropna()

    events = []
    for i in range(1, len(sma_valid)):
        date = sma_valid.index[i]
        prev_date = sma_valid.index[i - 1]
        if date not in adj_price.index or prev_date not in adj_price.index:
            continue

        price = adj_price[date]
        sma_val = sma_valid[date]
        prev_price = adj_price[prev_date]
        prev_sma = sma_valid[prev_date]

        if prev_price >= prev_sma and price < sma_val:
            pct_below = (price / sma_val - 1) * 100

            # Track forward for 60 days
            future = adj_price.index[adj_price.index > date]
            day10_pct = None
            min_pct = pct_below
            days_below = 0
            recovered = False

            for fd in future[:120]:
                if fd in sma.index and not pd.isna(sma[fd]):
                    fp = adj_price[fd]
                    fs = sma[fd]
                    pct = (fp / fs - 1) * 100
                    if pct < min_pct:
                        min_pct = pct
                    if fp >= fs:
                        # Check if sustained recovery
                        bounce_dates = adj_price.index[adj_price.index > fd][:10]
                        days_above = sum(1 for bd in bounce_dates
                                         if bd in sma.index and not pd.isna(sma[bd])
                                         and adj_price[bd] >= sma[bd])
                        if days_above >= 8:
                            recovered = True
                            break
                        continue
                    days_below += 1
                    if days_below == 10:
                        day10_pct = pct

            events.append({
                "breach_date": date,
                "entry_pct": pct_below,
                "day10_pct": day10_pct,
                "min_pct": min_pct,
                "recovered": recovered,
            })

    return events


def print_separator(char="=", width=100):
    print(char * width)


def main():
    print()
    print_separator()
    print("  IV/RV ANALYSIS — YieldMax Covered Call Rotation")
    print_separator()
    print()
    print("  Metrics:")
    print("  - RV (Realized Volatility): 20-day rolling std of daily log returns, annualized")
    print("  - Income Yield (IV proxy): Rolling 4-week dividend sum / price, annualized")
    print("  - Income/RV Ratio: Income yield / RV — when > 1, covered call engine is working")
    print()

    bt = RotationBacktester(windows=[5, 10, 20])

    for pair in PAIRS:
        bull_df, bear_df = load_pair(pair)
        if bull_df is None:
            continue

        bull_name = pair["bull"]
        bear_name = pair["bear"]
        label = f"{bull_name}/{bear_name}"

        print_separator()
        print(f"  {label} — {pair['name']}")
        print_separator()
        print()

        # Run V5 strategy
        res = bt.run_v5_sma_hysteresis(
            bull_df, bear_df, sma_period=SMA_PERIOD,
            band_pct=BAND_PCT, debounce_days=DEBOUNCE_DAYS
        )

        # Compute metrics for BOTH sides
        for df, name, side in [(bull_df, bull_name, "bull"), (bear_df, bear_name, "bear")]:
            rv = compute_rv(df["close"], window=20)
            income_yield = compute_income_yield(df, window_weeks=4)
            ratio = income_yield / rv

            adj_price, sma = get_adj_price_and_sma(df, SMA_PERIOD)
            sma_valid = sma.dropna()

            pct_vs_sma = pd.Series(np.nan, index=adj_price.index)
            for d in sma_valid.index:
                if d in adj_price.index:
                    pct_vs_sma[d] = (adj_price[d] / sma_valid[d] - 1) * 100

            print(f"  --- {name} ({side}) ---")
            print()

            # Summary stats
            rv_valid = rv.dropna()
            iy_valid = income_yield.dropna()
            ratio_valid = ratio.replace([np.inf, -np.inf], np.nan).dropna()

            print(f"  {'Metric':>25}  {'Min':>8}  {'25th':>8}  {'Median':>8}  {'75th':>8}  {'Max':>8}")
            print(f"  {'-' * 70}")
            print(f"  {'RV (20d, ann %)':>25}  {rv_valid.min():>7.1f}%  {rv_valid.quantile(0.25):>7.1f}%  "
                  f"{rv_valid.median():>7.1f}%  {rv_valid.quantile(0.75):>7.1f}%  {rv_valid.max():>7.1f}%")
            print(f"  {'Income Yield (ann %)':>25}  {iy_valid.min():>7.1f}%  {iy_valid.quantile(0.25):>7.1f}%  "
                  f"{iy_valid.median():>7.1f}%  {iy_valid.quantile(0.75):>7.1f}%  {iy_valid.max():>7.1f}%")
            if len(ratio_valid) > 0:
                print(f"  {'Income/RV Ratio':>25}  {ratio_valid.min():>8.2f}  {ratio_valid.quantile(0.25):>8.2f}  "
                      f"{ratio_valid.median():>8.2f}  {ratio_valid.quantile(0.75):>8.2f}  {ratio_valid.max():>8.2f}")
            print()

        # ============================================================
        # KEY ANALYSIS: IV/RV at strategy switch points
        # ============================================================
        switches = find_switch_dates(res)
        if switches:
            print(f"  IV/RV AT V5 STRATEGY SWITCH POINTS:")
            print()
            print(f"  {'Date':>12}  {'Switch':>15}  {'Bull RV':>8}  {'Bull IY':>8}  {'Bull I/RV':>9}  "
                  f"{'Bear RV':>8}  {'Bear IY':>8}  {'Bear I/RV':>9}  {'Bull %SMA':>10}")
            print(f"  {'-' * 105}")

            bull_rv = compute_rv(bull_df["close"], window=20)
            bear_rv = compute_rv(bear_df["close"], window=20)
            bull_iy = compute_income_yield(bull_df, window_weeks=4)
            bear_iy = compute_income_yield(bear_df, window_weeks=4)
            bull_ratio = bull_iy / bull_rv
            bear_ratio = bear_iy / bear_rv

            adj_price, sma = get_adj_price_and_sma(bull_df, SMA_PERIOD)

            for sw in switches:
                d = sw["date"]
                from_inst = bull_name if sw["from"] == "MSTY" else bear_name
                to_inst = bull_name if sw["to"] == "MSTY" else bear_name
                switch_label = f"{from_inst}->{to_inst}"

                brv = bull_rv[d] if d in bull_rv.index else np.nan
                biy = bull_iy[d] if d in bull_iy.index else np.nan
                br = bull_ratio[d] if d in bull_ratio.index else np.nan
                berv = bear_rv[d] if d in bear_rv.index else np.nan
                beiy = bear_iy[d] if d in bear_iy.index else np.nan
                ber = bear_ratio[d] if d in bear_ratio.index else np.nan

                pct_sma = np.nan
                if d in adj_price.index and d in sma.index and not pd.isna(sma.get(d)):
                    pct_sma = (adj_price[d] / sma[d] - 1) * 100

                print(f"  {d.date()!s:>12}  {switch_label:>15}  {brv:>7.1f}%  {biy:>7.1f}%  "
                      f"{br:>9.2f}  {berv:>7.1f}%  {beiy:>7.1f}%  {ber:>9.2f}  {pct_sma:>+9.1f}%")

            print()

        # ============================================================
        # KEY ANALYSIS: IV/RV at SMA breach events (bull side)
        # ============================================================
        breach_events = find_sma_breach_events(bull_df, SMA_PERIOD)

        # Deduplicate breaches within 30 days
        deduped = []
        for e in breach_events:
            if not deduped or (e["breach_date"] - deduped[-1]["breach_date"]).days >= 30:
                deduped.append(e)

        if deduped:
            print(f"  IV/RV AT {bull_name} SMA BREACH EVENTS:")
            print()
            print(f"  {'Date':>12}  {'%Below':>7}  {'Day10%':>8}  {'Outcome':>10}  "
                  f"{'RV':>8}  {'IncYld':>8}  {'I/RV':>7}  {'RV 5d Before':>12}  {'RV Change':>10}")
            print(f"  {'-' * 100}")

            bull_rv = compute_rv(bull_df["close"], window=20)
            bull_iy = compute_income_yield(bull_df, window_weeks=4)
            bull_ratio = bull_iy / bull_rv

            for e in deduped:
                d = e["breach_date"]
                outcome = "RECOVERED" if e["recovered"] else "COLLAPSED"

                brv = bull_rv[d] if d in bull_rv.index else np.nan
                biy = bull_iy[d] if d in bull_iy.index else np.nan
                br = bull_ratio[d] if d in bull_ratio.index else np.nan

                # RV 5 trading days before
                d_idx = bull_rv.index.get_loc(d) if d in bull_rv.index else None
                rv_before = np.nan
                rv_change = np.nan
                if d_idx is not None and d_idx >= 5:
                    rv_before = bull_rv.iloc[d_idx - 5]
                    if not pd.isna(rv_before) and rv_before > 0:
                        rv_change = (brv / rv_before - 1) * 100

                day10_str = f"{e['day10_pct']:+.1f}%" if e['day10_pct'] is not None else "  N/A"

                print(f"  {d.date()!s:>12}  {e['entry_pct']:>+6.1f}%  {day10_str:>8}  "
                      f"{outcome:>10}  {brv:>7.1f}%  {biy:>7.1f}%  {br:>7.2f}  "
                      f"{rv_before:>11.1f}%  {rv_change:>+9.1f}%")

            print()

        # ============================================================
        # IV/RV TIME SERIES: Monthly snapshots around key periods
        # ============================================================
        print(f"  MONTHLY IV/RV SNAPSHOT ({bull_name}):")
        print()

        bull_rv = compute_rv(bull_df["close"], window=20)
        bull_iy = compute_income_yield(bull_df, window_weeks=4)
        bull_ratio = bull_iy / bull_rv
        adj_price, sma = get_adj_price_and_sma(bull_df, SMA_PERIOD)

        # Sample monthly
        dates = bull_rv.dropna().index
        monthly = pd.DatetimeIndex([dates[0]] + [d for i, d in enumerate(dates) if
                                     i > 0 and d.month != dates[i-1].month] + [dates[-1]])

        print(f"  {'Date':>12}  {'Price':>8}  {'SMA':>8}  {'%SMA':>8}  "
              f"{'RV(20d)':>8}  {'IncYld':>8}  {'I/RV':>7}  {'Regime'}")
        print(f"  {'-' * 80}")

        for d in monthly:
            if d not in bull_rv.index:
                continue
            price = bull_df.loc[d, "close"] if d in bull_df.index else np.nan
            sma_val = sma[d] if d in sma.index else np.nan
            pct_sma = (adj_price[d] / sma_val - 1) * 100 if d in adj_price.index and not pd.isna(sma_val) else np.nan
            brv = bull_rv[d]
            biy = bull_iy[d] if d in bull_iy.index else np.nan
            br = bull_ratio[d] if d in bull_ratio.index else np.nan

            # Determine regime
            regime = ""
            if not pd.isna(pct_sma):
                if pct_sma > 5:
                    regime = "BULL"
                elif pct_sma < -5:
                    regime = "BEAR"
                else:
                    regime = "CHOP"
                if pct_sma < -10:
                    regime = "COLLAPSE"

            sma_str = f"${sma_val:.2f}" if not pd.isna(sma_val) else "  N/A"
            pct_str = f"{pct_sma:+.1f}%" if not pd.isna(pct_sma) else "  N/A"

            print(f"  {d.date()!s:>12}  ${price:>7.2f}  {sma_str:>8}  {pct_str:>8}  "
                  f"{brv:>7.1f}%  {biy:>7.1f}%  {br:>7.2f}  {regime}")

        print()
        print()

    # ================================================================
    # CROSS-PAIR SUMMARY: IV/RV at collapse vs recovery events
    # ================================================================
    print_separator()
    print("  CROSS-PAIR SUMMARY: IV/RV AT COLLAPSES vs RECOVERIES")
    print_separator()
    print()

    collapse_rvs = []
    collapse_ratios = []
    recovery_rvs = []
    recovery_ratios = []

    for pair in PAIRS:
        bull_df, bear_df = load_pair(pair)
        if bull_df is None:
            continue

        bull_rv = compute_rv(bull_df["close"], window=20)
        bull_iy = compute_income_yield(bull_df, window_weeks=4)
        bull_ratio = bull_iy / bull_rv

        breach_events = find_sma_breach_events(bull_df, SMA_PERIOD)
        deduped = []
        for e in breach_events:
            if not deduped or (e["breach_date"] - deduped[-1]["breach_date"]).days >= 30:
                deduped.append(e)

        for e in deduped:
            d = e["breach_date"]
            if d not in bull_rv.index:
                continue
            rv_val = bull_rv[d]
            ratio_val = bull_ratio[d] if d in bull_ratio.index else np.nan
            if pd.isna(rv_val) or pd.isna(ratio_val):
                continue

            if e["recovered"]:
                recovery_rvs.append(rv_val)
                recovery_ratios.append(ratio_val)
            else:
                collapse_rvs.append(rv_val)
                collapse_ratios.append(ratio_val)

    # Also check bear side collapses
    for pair in PAIRS:
        bull_df, bear_df = load_pair(pair)
        if bear_df is None:
            continue

        bear_rv = compute_rv(bear_df["close"], window=20)
        bear_iy = compute_income_yield(bear_df, window_weeks=4)
        bear_ratio = bear_iy / bear_rv

        breach_events = find_sma_breach_events(bear_df, SMA_PERIOD)
        deduped = []
        for e in breach_events:
            if not deduped or (e["breach_date"] - deduped[-1]["breach_date"]).days >= 30:
                deduped.append(e)

        for e in deduped:
            d = e["breach_date"]
            if d not in bear_rv.index:
                continue
            rv_val = bear_rv[d]
            ratio_val = bear_ratio[d] if d in bear_ratio.index else np.nan
            if pd.isna(rv_val) or pd.isna(ratio_val):
                continue

            if e["recovered"]:
                recovery_rvs.append(rv_val)
                recovery_ratios.append(ratio_val)
            else:
                collapse_rvs.append(rv_val)
                collapse_ratios.append(ratio_val)

    print(f"  {'':>25}  {'Collapses':>12}  {'Recoveries':>12}")
    print(f"  {'-' * 55}")
    print(f"  {'N events':>25}  {len(collapse_rvs):>12}  {len(recovery_rvs):>12}")

    if collapse_rvs:
        print(f"  {'Avg RV at breach':>25}  {np.mean(collapse_rvs):>11.1f}%  ", end="")
        if recovery_rvs:
            print(f"{np.mean(recovery_rvs):>11.1f}%")
        else:
            print("        N/A")

        print(f"  {'Median RV at breach':>25}  {np.median(collapse_rvs):>11.1f}%  ", end="")
        if recovery_rvs:
            print(f"{np.median(recovery_rvs):>11.1f}%")
        else:
            print("        N/A")

    if collapse_ratios:
        print(f"  {'Avg Income/RV at breach':>25}  {np.mean(collapse_ratios):>12.2f}  ", end="")
        if recovery_ratios:
            print(f"{np.mean(recovery_ratios):>12.2f}")
        else:
            print("        N/A")

        print(f"  {'Median Income/RV at breach':>25}  {np.median(collapse_ratios):>12.2f}  ", end="")
        if recovery_ratios:
            print(f"{np.median(recovery_ratios):>12.2f}")
        else:
            print("        N/A")

    print()

    # ================================================================
    # KEY FINDING: Does IV/RV predict collapse better than SMA alone?
    # ================================================================
    print_separator()
    print("  ANALYSIS: DOES IV/RV ADD PREDICTIVE VALUE?")
    print_separator()
    print()

    # For each pair, compute correlation between income/RV ratio and
    # forward 20-day returns (does low ratio predict poor forward returns?)
    print("  CORRELATION: Income/RV Ratio vs Forward 20-day Total Return")
    print()

    for pair in PAIRS:
        bull_df, bear_df = load_pair(pair)
        if bull_df is None:
            continue

        calc = TotalReturnCalculator()
        bull_tri = calc.calculate_total_return_index(bull_df)

        bull_rv = compute_rv(bull_df["close"], window=20)
        bull_iy = compute_income_yield(bull_df, window_weeks=4)
        bull_ratio = (bull_iy / bull_rv).replace([np.inf, -np.inf], np.nan)

        # Forward 20-day TRI return
        fwd_20d = (bull_tri.shift(-20) / bull_tri - 1) * 100

        # Align
        common = bull_ratio.dropna().index.intersection(fwd_20d.dropna().index)
        if len(common) < 30:
            continue

        x = bull_ratio[common]
        y = fwd_20d[common]
        corr = x.corr(y)

        print(f"  {pair['bull']:>6}: Income/RV vs Fwd 20d return correlation = {corr:+.3f}  "
              f"(N={len(common)})")

    print()

    # Correlation between RV level and forward returns
    print("  CORRELATION: RV Level vs Forward 20-day Total Return")
    print()

    for pair in PAIRS:
        bull_df, bear_df = load_pair(pair)
        if bull_df is None:
            continue

        calc = TotalReturnCalculator()
        bull_tri = calc.calculate_total_return_index(bull_df)
        bull_rv = compute_rv(bull_df["close"], window=20)
        fwd_20d = (bull_tri.shift(-20) / bull_tri - 1) * 100

        common = bull_rv.dropna().index.intersection(fwd_20d.dropna().index)
        if len(common) < 30:
            continue

        x = bull_rv[common]
        y = fwd_20d[common]
        corr = x.corr(y)

        print(f"  {pair['bull']:>6}: RV vs Fwd 20d return correlation = {corr:+.3f}  "
              f"(N={len(common)})")

    print()

    # RV percentile at breach events — are collapses associated with
    # RV being in a specific range?
    print("  RV PERCENTILE AT BREACH (relative to each ETF's own RV distribution):")
    print()

    for pair in PAIRS:
        bull_df, bear_df = load_pair(pair)
        if bull_df is None:
            continue

        bull_rv = compute_rv(bull_df["close"], window=20).dropna()

        breach_events = find_sma_breach_events(bull_df, SMA_PERIOD)
        deduped = []
        for e in breach_events:
            if not deduped or (e["breach_date"] - deduped[-1]["breach_date"]).days >= 30:
                deduped.append(e)

        for e in deduped:
            d = e["breach_date"]
            if d not in bull_rv.index:
                continue
            rv_val = bull_rv[d]
            pctile = (bull_rv < rv_val).mean() * 100
            outcome = "RECOVERED" if e["recovered"] else "COLLAPSED"

            print(f"    {pair['bull']:>6}  {d.date()}  RV={rv_val:.1f}%  "
                  f"({pctile:.0f}th percentile)  {outcome}")

    print()

    print_separator()
    print("  END OF IV/RV ANALYSIS")
    print_separator()
    print()


if __name__ == "__main__":
    main()
