#!/usr/bin/env python3
"""
Volatility Trend Analysis at Strategy Decision Points.

Prior analysis looked at RV LEVELS (percentiles, thresholds) and found
weak correlations. This analysis looks at RV and Income Yield TRENDS
(direction and rate of change) at V5 switch points.

The hypothesis: the Unusual Whales charts showed that collapses happen
when IV AND RV are both DECLINING (the calm grind), while recoveries
happen when they spike. The key metric may not be "is RV high or low"
but "is RV rising or falling."

Metrics computed:
  1. RV (20-day, annualized) from ETF price
  2. RV trend: 10-day slope of RV (via linear regression)
  3. Income yield: rolling 4-week dividend / price, annualized
  4. Income yield trend: 10-day slope
  5. Income/RV ratio and its trend
  6. Price momentum: 20-day rate of change
  7. SMA distance trend: is the gap widening or narrowing?
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

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


def get_adj_price(df):
    """Get split-adjusted price."""
    df_sorted = df.sort_index()
    if "split" in df_sorted.columns:
        cum_split = df_sorted["split"].replace(0, 1).cumprod()
        final_split = cum_split.iloc[-1]
        split_adj = cum_split / final_split
        return df_sorted["close"] / split_adj
    return df_sorted["close"]


def compute_rv(adj_price, window=20):
    """Realized volatility: annualized std of log returns."""
    log_ret = np.log(adj_price / adj_price.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252) * 100


def compute_income_yield(df, rolling_weeks=4):
    """Income yield: rolling dividend sum / price, annualized."""
    adj_price = get_adj_price(df)
    divs = df["dividend"] if "dividend" in df.columns else pd.Series(0, index=df.index)
    # Adjust dividends for splits same way as price
    if "split" in df.columns:
        cum_split = df["split"].replace(0, 1).cumprod()
        final_split = cum_split.iloc[-1]
        split_adj = cum_split / final_split
        adj_divs = divs / split_adj
    else:
        adj_divs = divs
    rolling_days = rolling_weeks * 5  # approx trading days per week
    rolling_div = adj_divs.rolling(rolling_days, min_periods=1).sum()
    income_yield = (rolling_div / adj_price) * (52 / rolling_weeks) * 100
    return income_yield


def rolling_slope(series, window=10):
    """Linear regression slope over rolling window (normalized: slope per day as % of mean)."""
    result = pd.Series(np.nan, index=series.index)
    values = series.values
    for i in range(window - 1, len(values)):
        y = values[i - window + 1:i + 1]
        if np.isnan(y).any():
            continue
        x = np.arange(window)
        slope, _, _, _, _ = scipy_stats.linregress(x, y)
        # Normalize: slope as fraction of current value per day
        if abs(values[i]) > 0.001:
            result.iloc[i] = slope / abs(values[i]) * 100  # percent per day
        else:
            result.iloc[i] = 0.0
    return result


def find_v5_switch_dates(bt, bull_df, bear_df):
    """Run V5 and extract switch dates with direction."""
    v5 = bt.run_v5_sma_hysteresis(
        bull_df, bear_df, sma_period=SMA_PERIOD,
        band_pct=5.0, debounce_days=DEBOUNCE_DAYS
    )
    switches = []
    for i, t in enumerate(v5.trades):
        switches.append({
            "date": t.entry_date,
            "direction": "BULL" if t.instrument == "MSTY" else "BEAR",
            "trade_num": t.trade_num,
            "holding_days": t.holding_days,
            "total_return": t.total_return_pct,
        })
    return switches, v5


def classify_trade_quality(trade):
    """Classify a trade as good, neutral, or bad based on total return."""
    if trade["total_return"] > 10:
        return "GOOD"
    elif trade["total_return"] > -5:
        return "NEUTRAL"
    else:
        return "BAD"


def print_separator(char="=", width=110):
    print(char * width)


def main():
    print()
    print_separator()
    print("  VOLATILITY TREND ANALYSIS AT V5 DECISION POINTS")
    print_separator()
    print()
    print("  Previous analysis: looked at RV levels (percentiles) → weak correlations.")
    print("  This analysis: looks at RV and income yield TRENDS (slopes) at switch points.")
    print()
    print("  Hypothesis: collapses coincide with declining RV+income (calm grind),")
    print("  recoveries coincide with rising RV+income (vol spike + rich premiums).")
    print()

    bt = RotationBacktester(windows=[5, 10, 20])

    # Collect all switch data across pairs
    all_switches = []

    for pair in PAIRS:
        bull_df, bear_df = load_pair(pair)
        if bull_df is None:
            continue

        label = f"{pair['bull']}/{pair['bear']}"
        bull_name = pair["bull"]
        bear_name = pair["bear"]

        adj_price = get_adj_price(bull_df)
        sma = adj_price.rolling(SMA_PERIOD, min_periods=SMA_PERIOD).mean()

        # Core metrics
        rv = compute_rv(adj_price, window=20)
        income = compute_income_yield(bull_df, rolling_weeks=4)
        ratio = income / rv.replace(0, np.nan)

        # Trends (10-day slopes, normalized)
        rv_slope = rolling_slope(rv, window=10)
        income_slope = rolling_slope(income, window=10)
        ratio_slope = rolling_slope(ratio.dropna(), window=10)

        # Price momentum
        price_mom = (adj_price / adj_price.shift(20) - 1) * 100

        # SMA distance and its trend
        sma_dist = (adj_price / sma - 1) * 100
        sma_dist_slope = rolling_slope(sma_dist.dropna(), window=10)

        # Get V5 switches
        switches, v5 = find_v5_switch_dates(bt, bull_df, bear_df)

        # Enrich each switch with metrics
        for sw in switches:
            d = sw["date"]
            sw["pair"] = label
            sw["bull"] = bull_name
            sw["bear"] = bear_name
            sw["quality"] = classify_trade_quality(sw)

            # Levels at switch
            sw["rv"] = rv[d] if d in rv.index else np.nan
            sw["income"] = income[d] if d in income.index else np.nan
            sw["ratio"] = ratio[d] if d in ratio.index else np.nan
            sw["sma_dist"] = sma_dist[d] if d in sma_dist.index else np.nan
            sw["price_mom"] = price_mom[d] if d in price_mom.index else np.nan

            # Trends at switch (rate of change)
            sw["rv_slope"] = rv_slope[d] if d in rv_slope.index else np.nan
            sw["income_slope"] = income_slope[d] if d in income_slope.index else np.nan
            sw["ratio_slope"] = ratio_slope[d] if d in ratio_slope.index else np.nan
            sw["sma_dist_slope"] = sma_dist_slope[d] if d in sma_dist_slope.index else np.nan

            # RV percentile (trailing 60d)
            if d in rv.index and not pd.isna(rv[d]):
                rv_loc = rv.index.get_loc(d)
                start_loc = max(0, rv_loc - 60)
                rv_win = rv.iloc[start_loc:rv_loc + 1].dropna()
                sw["rv_pctile"] = (rv_win < rv[d]).mean() * 100 if len(rv_win) >= 20 else np.nan
            else:
                sw["rv_pctile"] = np.nan

            all_switches.append(sw)

    # ================================================================
    # PART 1: Full metric snapshot at each V5 switch
    # ================================================================
    print_separator()
    print("  PART 1: COMPLETE METRIC SNAPSHOT AT EACH V5 SWITCH")
    print_separator()
    print()
    print(f"  {'Pair':<14}  {'Date':>12}  {'Dir':>5}  {'Quality':>7}  {'TotRet':>7}  "
          f"{'RV':>6}  {'RV%ile':>6}  {'RV Slope':>9}  {'Income':>7}  {'Inc Slope':>10}  "
          f"{'Ratio':>6}  {'PxMom':>6}  {'%SMA':>6}")
    print(f"  {'-' * 108}")

    for sw in all_switches:
        rv_str = f"{sw['rv']:.0f}%" if not pd.isna(sw['rv']) else "N/A"
        pctile_str = f"{sw['rv_pctile']:.0f}" if not pd.isna(sw['rv_pctile']) else "N/A"
        rv_sl_str = f"{sw['rv_slope']:+.2f}" if not pd.isna(sw['rv_slope']) else "N/A"
        inc_str = f"{sw['income']:.0f}%" if not pd.isna(sw['income']) else "N/A"
        inc_sl_str = f"{sw['income_slope']:+.2f}" if not pd.isna(sw['income_slope']) else "N/A"
        ratio_str = f"{sw['ratio']:.2f}" if not pd.isna(sw['ratio']) else "N/A"
        mom_str = f"{sw['price_mom']:+.0f}%" if not pd.isna(sw['price_mom']) else "N/A"
        sma_str = f"{sw['sma_dist']:+.0f}%" if not pd.isna(sw['sma_dist']) else "N/A"

        print(f"  {sw['pair']:<14}  {sw['date'].date()!s:>12}  "
              f"{'→'+sw['direction']:>5}  {sw['quality']:>7}  {sw['total_return']:>+6.1f}%  "
              f"{rv_str:>6}  {pctile_str:>6}  {rv_sl_str:>9}  {inc_str:>7}  {inc_sl_str:>10}  "
              f"{ratio_str:>6}  {mom_str:>6}  {sma_str:>6}")

    # ================================================================
    # PART 2: Separate bear switches — which are "good" vs "bad"
    # ================================================================
    print()
    print_separator()
    print("  PART 2: BEAR SWITCH ANALYSIS — RV TREND AS PREDICTOR")
    print_separator()
    print()
    print("  Bear switches (bull→bear): the critical decision. Is this a real collapse")
    print("  or a temporary spike? If RV is DECLINING at the switch, it's a calm grind")
    print("  (real collapse). If RV is RISING, it's a vol spike (may be temporary).")
    print()

    bear_switches = [sw for sw in all_switches if sw["direction"] == "BEAR"]

    if bear_switches:
        print(f"  {'Pair':<14}  {'Date':>12}  {'Quality':>7}  {'TotRet':>7}  "
              f"{'RV':>6}  {'RV%ile':>6}  {'RV Slope':>9}  {'Inc Slope':>10}  "
              f"{'Ratio':>6}  {'Rat Slope':>10}  {'Verdict'}")
        print(f"  {'-' * 110}")

        for sw in bear_switches:
            rv_sl = sw['rv_slope']
            inc_sl = sw['income_slope']
            ratio_sl = sw['ratio_slope']

            # Determine verdict based on trends
            if not pd.isna(rv_sl):
                if rv_sl < -0.5:
                    rv_verdict = "RV↓"
                elif rv_sl > 0.5:
                    rv_verdict = "RV↑"
                else:
                    rv_verdict = "RV→"
            else:
                rv_verdict = "?"

            if not pd.isna(inc_sl):
                if inc_sl < -0.5:
                    inc_verdict = "Inc↓"
                elif inc_sl > 0.5:
                    inc_verdict = "Inc↑"
                else:
                    inc_verdict = "Inc→"
            else:
                inc_verdict = "?"

            verdict = f"{rv_verdict} {inc_verdict}"

            rv_sl_str = f"{rv_sl:+.2f}" if not pd.isna(rv_sl) else "N/A"
            inc_sl_str = f"{inc_sl:+.2f}" if not pd.isna(inc_sl) else "N/A"
            ratio_str = f"{sw['ratio']:.2f}" if not pd.isna(sw['ratio']) else "N/A"
            ratio_sl_str = f"{ratio_sl:+.2f}" if not pd.isna(ratio_sl) else "N/A"

            print(f"  {sw['pair']:<14}  {sw['date'].date()!s:>12}  {sw['quality']:>7}  "
                  f"{sw['total_return']:>+6.1f}%  {sw['rv']:.0f}%  "
                  f"{sw['rv_pctile']:.0f}  {rv_sl_str:>9}  {inc_sl_str:>10}  "
                  f"{ratio_str:>6}  {ratio_sl_str:>10}  {verdict}")

    # ================================================================
    # PART 3: Bull switches — recovery timing
    # ================================================================
    print()
    print_separator()
    print("  PART 3: BULL SWITCH ANALYSIS")
    print_separator()
    print()

    bull_switches = [sw for sw in all_switches if sw["direction"] == "BULL"]

    if bull_switches:
        print(f"  {'Pair':<14}  {'Date':>12}  {'Quality':>7}  {'TotRet':>7}  "
              f"{'RV':>6}  {'RV%ile':>6}  {'RV Slope':>9}  {'Inc Slope':>10}  "
              f"{'Ratio':>6}  {'Verdict'}")
        print(f"  {'-' * 100}")

        for sw in bull_switches:
            rv_sl = sw['rv_slope']
            inc_sl = sw['income_slope']

            if not pd.isna(rv_sl):
                rv_verdict = "RV↓" if rv_sl < -0.5 else ("RV↑" if rv_sl > 0.5 else "RV→")
            else:
                rv_verdict = "?"

            if not pd.isna(inc_sl):
                inc_verdict = "Inc↓" if inc_sl < -0.5 else ("Inc↑" if inc_sl > 0.5 else "Inc→")
            else:
                inc_verdict = "?"

            verdict = f"{rv_verdict} {inc_verdict}"
            rv_sl_str = f"{rv_sl:+.2f}" if not pd.isna(rv_sl) else "N/A"
            inc_sl_str = f"{inc_sl:+.2f}" if not pd.isna(inc_sl) else "N/A"
            ratio_str = f"{sw['ratio']:.2f}" if not pd.isna(sw['ratio']) else "N/A"

            print(f"  {sw['pair']:<14}  {sw['date'].date()!s:>12}  {sw['quality']:>7}  "
                  f"{sw['total_return']:>+6.1f}%  {sw['rv']:.0f}%  "
                  f"{sw['rv_pctile']:.0f}  {rv_sl_str:>9}  {inc_sl_str:>10}  "
                  f"{ratio_str:>6}  {verdict}")

    # ================================================================
    # PART 4: Statistical tests — do trends predict trade quality?
    # ================================================================
    print()
    print_separator()
    print("  PART 4: STATISTICAL CORRELATION — TRENDS vs TRADE OUTCOME")
    print_separator()
    print()

    # Test: does RV slope at switch predict subsequent total return?
    valid = [sw for sw in all_switches if not pd.isna(sw['rv_slope']) and not pd.isna(sw['total_return'])]

    if len(valid) >= 5:
        metrics_to_test = [
            ("RV Slope", "rv_slope"),
            ("Income Slope", "income_slope"),
            ("RV Level", "rv"),
            ("RV Percentile", "rv_pctile"),
            ("Income/RV Ratio", "ratio"),
            ("Ratio Slope", "ratio_slope"),
            ("Price Momentum", "price_mom"),
            ("SMA Distance", "sma_dist"),
            ("SMA Dist Slope", "sma_dist_slope"),
        ]

        print(f"  Correlation of metric at switch → subsequent trade total return:")
        print()
        print(f"  {'Metric':<20}  {'Corr':>6}  {'p-value':>8}  {'n':>3}  {'Interpretation'}")
        print(f"  {'-' * 75}")

        for name, key in metrics_to_test:
            data = [(sw[key], sw['total_return']) for sw in valid
                    if not pd.isna(sw.get(key, np.nan))]
            if len(data) >= 4:
                x = [d[0] for d in data]
                y = [d[1] for d in data]
                corr, p = scipy_stats.pearsonr(x, y)
                strength = ""
                if abs(corr) > 0.5:
                    strength = "STRONG" if p < 0.1 else "moderate"
                elif abs(corr) > 0.3:
                    strength = "moderate" if p < 0.1 else "weak"
                else:
                    strength = "weak"
                direction = "positive" if corr > 0 else "negative"
                print(f"  {name:<20}  {corr:>+5.2f}  {p:>8.3f}  {len(data):>3}  "
                      f"{strength} {direction}")

        # Also test for bear switches only
        bear_valid = [sw for sw in bear_switches
                      if not pd.isna(sw.get('rv_slope', np.nan))]
        if len(bear_valid) >= 4:
            print()
            print(f"  BEAR SWITCHES ONLY (n={len(bear_valid)}):")
            print(f"  {'Metric':<20}  {'Corr':>6}  {'p-value':>8}  {'n':>3}")
            print(f"  {'-' * 50}")

            for name, key in metrics_to_test:
                data = [(sw[key], sw['total_return']) for sw in bear_valid
                        if not pd.isna(sw.get(key, np.nan))]
                if len(data) >= 3:
                    x = [d[0] for d in data]
                    y = [d[1] for d in data]
                    if len(set(x)) > 1:
                        corr, p = scipy_stats.pearsonr(x, y)
                        print(f"  {name:<20}  {corr:>+5.2f}  {p:>8.3f}  {len(data):>3}")

    # ================================================================
    # PART 5: Regime snapshots — monthly vol/income/ratio evolution
    # ================================================================
    print()
    print_separator()
    print("  PART 5: MONTHLY VOL/INCOME/RATIO EVOLUTION PER PAIR")
    print_separator()

    for pair in PAIRS:
        bull_df, bear_df = load_pair(pair)
        if bull_df is None:
            continue

        label = f"{pair['bull']}/{pair['bear']}"
        bull_name = pair["bull"]

        adj_price = get_adj_price(bull_df)
        sma = adj_price.rolling(SMA_PERIOD, min_periods=SMA_PERIOD).mean()
        rv = compute_rv(adj_price, 20)
        income = compute_income_yield(bull_df, 4)
        ratio = income / rv.replace(0, np.nan)
        rv_slope = rolling_slope(rv, 10)
        income_slope = rolling_slope(income, 10)

        # Monthly samples
        valid_dates = rv.dropna().index.intersection(income.dropna().index)
        if len(valid_dates) == 0:
            continue
        monthly = [valid_dates[0]]
        for i in range(1, len(valid_dates)):
            if valid_dates[i].month != valid_dates[i-1].month:
                monthly.append(valid_dates[i])
        monthly.append(valid_dates[-1])

        print()
        print(f"  --- {label} ({pair['name']}) ---")
        print()
        print(f"  {'Date':>12}  {'Price':>8}  {'%SMA':>8}  {'RV':>6}  {'RV Slope':>9}  "
              f"{'Income':>7}  {'Inc Slope':>10}  {'Inc/RV':>7}  {'Regime'}")
        print(f"  {'-' * 90}")

        # Get V5 switches for this pair
        switches, _ = find_v5_switch_dates(bt, bull_df, bear_df)
        switch_dates = {sw["date"]: sw["direction"] for sw in switches}

        for d in monthly:
            p = adj_price[d] if d in adj_price.index else np.nan
            s = sma[d] if d in sma.index else np.nan
            pct = (p / s - 1) * 100 if not pd.isna(s) else np.nan
            r = rv[d] if d in rv.index else np.nan
            inc = income[d] if d in income.index else np.nan
            rat = ratio[d] if d in ratio.index else np.nan
            r_sl = rv_slope[d] if d in rv_slope.index else np.nan
            i_sl = income_slope[d] if d in income_slope.index else np.nan

            # Determine regime
            regime = ""
            if not pd.isna(r_sl) and not pd.isna(i_sl):
                if r_sl < -0.5 and i_sl < -0.5:
                    regime = "DANGER: vol+income declining"
                elif r_sl > 0.5 and i_sl > 0.5:
                    regime = "SPIKE: vol+income rising"
                elif r_sl < -0.5:
                    regime = "vol declining"
                elif r_sl > 0.5:
                    regime = "vol rising"
                elif i_sl < -0.5:
                    regime = "income declining"
                elif i_sl > 0.5:
                    regime = "income rising"
                else:
                    regime = "stable"

            # Mark switch dates
            switch_marker = ""
            # Check if any switch is within 3 days of this monthly sample
            for sd, direction in switch_dates.items():
                if abs((sd - d).days) <= 3:
                    switch_marker = f"  *** SWITCH→{direction} ***"

            pct_str = f"{pct:+.1f}%" if not pd.isna(pct) else "N/A"
            r_str = f"{r:.0f}%" if not pd.isna(r) else "N/A"
            r_sl_str = f"{r_sl:+.1f}" if not pd.isna(r_sl) else "N/A"
            inc_str = f"{inc:.0f}%" if not pd.isna(inc) else "N/A"
            i_sl_str = f"{i_sl:+.1f}" if not pd.isna(i_sl) else "N/A"
            rat_str = f"{rat:.2f}" if not pd.isna(rat) else "N/A"

            print(f"  {d.date()!s:>12}  ${p:>7.2f}  {pct_str:>8}  {r_str:>6}  {r_sl_str:>9}  "
                  f"{inc_str:>7}  {i_sl_str:>10}  {rat_str:>7}  {regime}{switch_marker}")

    # ================================================================
    # PART 6: SMA breach events with trend context
    # ================================================================
    print()
    print_separator()
    print("  PART 6: SMA BREACH EVENTS WITH TREND CONTEXT")
    print_separator()
    print()
    print("  When price first crosses below SMA-5%, what are the trends?")
    print("  This is BEFORE debounce — the moment the threshold is first hit.")
    print()

    all_breach_events = []

    for pair in PAIRS:
        bull_df, bear_df = load_pair(pair)
        if bull_df is None:
            continue

        label = f"{pair['bull']}/{pair['bear']}"
        bull_name = pair["bull"]

        adj_price = get_adj_price(bull_df)
        sma = adj_price.rolling(SMA_PERIOD, min_periods=SMA_PERIOD).mean()
        rv = compute_rv(adj_price, 20)
        income = compute_income_yield(bull_df, 4)
        ratio = income / rv.replace(0, np.nan)
        rv_slope = rolling_slope(rv, 10)
        income_slope = rolling_slope(income, 10)
        price_mom = (adj_price / adj_price.shift(20) - 1) * 100

        sma_dist = (adj_price / sma - 1) * 100

        # Find first breach below -5% after being above -5%
        above = True
        for date in sma_dist.dropna().index:
            pct = sma_dist[date]
            if above and pct < -5.0:
                above = False
                # This is a breach event

                # Forward outcome: where is price in 20 trading days?
                future_dates = adj_price.index[adj_price.index > date][:20]
                if len(future_dates) >= 20:
                    fwd20 = (adj_price[future_dates[-1]] / adj_price[date] - 1) * 100
                else:
                    fwd20 = np.nan

                # Did this breach lead to a V5 bear switch?
                # (check if V5 held bear position within 5 days of this breach)
                # We'll just note the metrics

                rv_at = rv[date] if date in rv.index else np.nan
                rv_sl = rv_slope[date] if date in rv_slope.index else np.nan
                inc_at = income[date] if date in income.index else np.nan
                inc_sl = income_slope[date] if date in income_slope.index else np.nan
                rat_at = ratio[date] if date in ratio.index else np.nan
                mom_at = price_mom[date] if date in price_mom.index else np.nan

                # RV percentile
                if date in rv.index and not pd.isna(rv[date]):
                    rv_loc = rv.index.get_loc(date)
                    start_loc = max(0, rv_loc - 60)
                    rv_win = rv.iloc[start_loc:rv_loc + 1].dropna()
                    rv_pctile = (rv_win < rv[date]).mean() * 100 if len(rv_win) >= 20 else np.nan
                else:
                    rv_pctile = np.nan

                event = {
                    "pair": label, "etf": bull_name, "date": date,
                    "sma_dist": pct, "fwd20": fwd20,
                    "rv": rv_at, "rv_pctile": rv_pctile, "rv_slope": rv_sl,
                    "income": inc_at, "income_slope": inc_sl,
                    "ratio": rat_at, "price_mom": mom_at,
                }
                all_breach_events.append(event)

            elif not above and pct > -3.0:  # Need to recover above -3% to reset
                above = True

    if all_breach_events:
        print(f"  {'ETF':>6}  {'Date':>12}  {'%SMA':>6}  {'Fwd20':>7}  "
              f"{'RV':>6}  {'RV%ile':>6}  {'RV Slp':>7}  {'Income':>7}  {'Inc Slp':>8}  "
              f"{'Inc/RV':>7}  {'PxMom':>6}  {'Outcome'}")
        print(f"  {'-' * 105}")

        for e in sorted(all_breach_events, key=lambda x: x["date"]):
            outcome = ""
            if not pd.isna(e["fwd20"]):
                outcome = "RECOVERED" if e["fwd20"] > 0 else "CONTINUED ↓"

            rv_sl_str = f"{e['rv_slope']:+.1f}" if not pd.isna(e['rv_slope']) else "N/A"
            inc_sl_str = f"{e['income_slope']:+.1f}" if not pd.isna(e['income_slope']) else "N/A"
            fwd_str = f"{e['fwd20']:+.1f}%" if not pd.isna(e['fwd20']) else "N/A"
            rv_str = f"{e['rv']:.0f}%" if not pd.isna(e['rv']) else "N/A"
            pctile_str = f"{e['rv_pctile']:.0f}" if not pd.isna(e['rv_pctile']) else "N/A"
            inc_str = f"{e['income']:.0f}%" if not pd.isna(e['income']) else "N/A"
            rat_str = f"{e['ratio']:.2f}" if not pd.isna(e['ratio']) else "N/A"
            mom_str = f"{e['price_mom']:+.0f}%" if not pd.isna(e['price_mom']) else "N/A"

            print(f"  {e['etf']:>6}  {e['date'].date()!s:>12}  {e['sma_dist']:+.1f}%  "
                  f"{fwd_str:>7}  {rv_str:>6}  {pctile_str:>6}  {rv_sl_str:>7}  "
                  f"{inc_str:>7}  {inc_sl_str:>8}  {rat_str:>7}  {mom_str:>6}  {outcome}")

        # Statistical test on breach events
        valid_breach = [e for e in all_breach_events
                        if not pd.isna(e.get("fwd20")) and not pd.isna(e.get("rv_slope"))]
        if len(valid_breach) >= 4:
            print()
            print("  Correlation at BREACH moments (metric → 20-day forward return):")
            print()
            for name, key in [("RV Slope", "rv_slope"), ("Income Slope", "income_slope"),
                              ("RV Level", "rv"), ("RV Percentile", "rv_pctile"),
                              ("Income/RV Ratio", "ratio"), ("Price Momentum", "price_mom")]:
                data = [(e[key], e["fwd20"]) for e in valid_breach
                        if not pd.isna(e.get(key, np.nan))]
                if len(data) >= 4:
                    x = [d[0] for d in data]
                    y = [d[1] for d in data]
                    if len(set(x)) > 1:
                        corr, p = scipy_stats.pearsonr(x, y)
                        sig = "***" if p < 0.05 else "**" if p < 0.1 else "*" if p < 0.2 else ""
                        print(f"    {name:<20}  r={corr:+.3f}  p={p:.3f}  n={len(data)}  {sig}")

    # ================================================================
    # PART 7: Composite signal test
    # ================================================================
    print()
    print_separator()
    print("  PART 7: COMPOSITE TREND SIGNAL")
    print_separator()
    print()
    print("  Testing: at SMA breach, if BOTH RV slope AND income slope are negative")
    print("  (declining vol + declining income), does the breach always lead to collapse?")
    print("  If either slope is positive (rising vol or rising income), does it recover?")
    print()

    if all_breach_events:
        both_declining = [e for e in all_breach_events
                          if not pd.isna(e.get("rv_slope")) and not pd.isna(e.get("income_slope"))
                          and e["rv_slope"] < 0 and e["income_slope"] < 0]
        either_rising = [e for e in all_breach_events
                         if not pd.isna(e.get("rv_slope")) and not pd.isna(e.get("income_slope"))
                         and (e["rv_slope"] > 0 or e["income_slope"] > 0)]

        print(f"  BOTH declining (RV↓ + Income↓) at breach: {len(both_declining)} events")
        for e in both_declining:
            fwd_str = f"{e['fwd20']:+.1f}%" if not pd.isna(e['fwd20']) else "?"
            outcome = "CONTINUED ↓" if not pd.isna(e["fwd20"]) and e["fwd20"] < 0 else "RECOVERED"
            print(f"    {e['etf']:>6} {e['date'].date()}  RV slope={e['rv_slope']:+.1f}  "
                  f"Inc slope={e['income_slope']:+.1f}  Fwd20={fwd_str}  → {outcome}")

        print()
        print(f"  EITHER rising (RV↑ or Income↑) at breach: {len(either_rising)} events")
        for e in either_rising:
            fwd_str = f"{e['fwd20']:+.1f}%" if not pd.isna(e['fwd20']) else "?"
            outcome = "RECOVERED" if not pd.isna(e["fwd20"]) and e["fwd20"] > 0 else "CONTINUED ↓"
            rising = []
            if e["rv_slope"] > 0:
                rising.append("RV↑")
            if e["income_slope"] > 0:
                rising.append("Inc↑")
            print(f"    {e['etf']:>6} {e['date'].date()}  RV slope={e['rv_slope']:+.1f}  "
                  f"Inc slope={e['income_slope']:+.1f}  Fwd20={fwd_str}  → {outcome}  "
                  f"({'+'.join(rising)})")

        # Classification accuracy
        if both_declining or either_rising:
            correct_decline = sum(1 for e in both_declining
                                  if not pd.isna(e["fwd20"]) and e["fwd20"] < 0)
            correct_rise = sum(1 for e in either_rising
                               if not pd.isna(e["fwd20"]) and e["fwd20"] > 0)
            total = len([e for e in both_declining + either_rising if not pd.isna(e["fwd20"])])
            correct = correct_decline + correct_rise
            print()
            print(f"  Classification accuracy: {correct}/{total} "
                  f"({correct/total*100:.0f}%)" if total > 0 else "")

    print()
    print_separator()
    print("  END OF VOLATILITY TREND ANALYSIS")
    print_separator()
    print()


if __name__ == "__main__":
    main()
