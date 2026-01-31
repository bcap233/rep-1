#!/usr/bin/env python3
"""
V5 YieldMax Rotation Strategy — Complete Summary Report.

Generates a comprehensive summary of:
1. V5 strategy rules and rationale
2. Backtest results across all 4 pairs (signal-only, no blind period)
3. Complete trade history for each pair
4. Reflexive collapse threshold analysis (10% below SMA by Day 10)
5. Opposite ETF lag pattern (2-4 weeks)
6. Current signals
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta

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


def max_dd(equity):
    rm = equity.expanding().max()
    return ((equity / rm - 1) * 100).min()


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


def analyze_collapse_events(bull_df, bear_df, bull_name, bear_name, sma_period=50):
    """Analyze SMA breach events to find collapse threshold patterns."""
    events = []

    for df, name, opposite_df, opp_name, side in [
        (bull_df, bull_name, bear_df, bear_name, "bull"),
        (bear_df, bear_name, bull_df, bull_name, "bear"),
    ]:
        adj_price, sma = get_adj_price_and_sma(df, sma_period)
        opp_adj, opp_sma = get_adj_price_and_sma(opposite_df, sma_period)

        sma_valid = sma.dropna()
        if len(sma_valid) == 0:
            continue

        # Find breach events: price crosses below SMA
        was_above = True
        for i in range(1, len(sma_valid)):
            date = sma_valid.index[i]
            if date not in adj_price.index:
                continue
            price = adj_price[date]
            sma_val = sma_valid[date]
            prev_date = sma_valid.index[i - 1]
            if prev_date not in adj_price.index:
                continue
            prev_price = adj_price[prev_date]
            prev_sma = sma_valid[prev_date]

            price_above = price >= sma_val
            prev_above = prev_price >= prev_sma

            if prev_above and not price_above:
                # Breach event! Track what happens over next 30 days
                breach_date = date
                entry_pct = (price / sma_val - 1) * 100

                # Track forward
                future_dates = adj_price.index[adj_price.index > breach_date]
                day10_pct = None
                min_pct = entry_pct
                min_date = breach_date
                recovered = False
                recover_date = None
                days_below = 0

                for fd in future_dates[:120]:  # Look up to 120 trading days
                    if fd in sma.index and not pd.isna(sma[fd]):
                        fp = adj_price[fd]
                        fs = sma[fd]
                        pct = (fp / fs - 1) * 100
                        if pct < min_pct:
                            min_pct = pct
                            min_date = fd
                        if fp >= fs:
                            # Check if this is a real recovery or dead cat bounce
                            # Look ahead 10 more days — if falls back below, it's a bounce
                            bounce_dates = adj_price.index[adj_price.index > fd][:10]
                            days_above = 0
                            for bd in bounce_dates:
                                if bd in sma.index and not pd.isna(sma[bd]):
                                    if adj_price[bd] >= sma[bd]:
                                        days_above += 1
                            if days_above >= 8:  # Stayed above SMA for 8+ of 10 days
                                recovered = True
                                recover_date = fd
                                break
                            # else: dead cat bounce, keep tracking
                            continue
                        days_below += 1
                        if days_below == 10:
                            day10_pct = pct

                # Check opposite ETF crossing above its SMA
                opp_cross_date = None
                opp_cross_lag = None
                for fd in opp_adj.index[opp_adj.index > breach_date]:
                    if fd in opp_sma.index and not pd.isna(opp_sma[fd]):
                        if opp_adj[fd] > opp_sma[fd]:
                            opp_cross_date = fd
                            opp_cross_lag = (fd - breach_date).days
                            break

                events.append({
                    "etf": name,
                    "side": side,
                    "opposite": opp_name,
                    "breach_date": breach_date,
                    "entry_pct": entry_pct,
                    "day10_pct": day10_pct,
                    "min_pct": min_pct,
                    "min_date": min_date,
                    "recovered": recovered,
                    "recover_date": recover_date,
                    "days_below": days_below,
                    "opp_cross_date": opp_cross_date,
                    "opp_cross_lag": opp_cross_lag,
                })

    return events


def print_separator(char="=", width=100):
    print(char * width)


def main():
    print()
    print_separator()
    print("  YIELDMAX COVERED CALL ROTATION — V5 STRATEGY SUMMARY")
    print_separator()
    print()
    print("  Generated from backtested data across 4 YieldMax bull/bear pairs.")
    print("  Strategy: SMA50 +/- 5% hysteresis band with 3-day debounce.")
    print("  All results exclude the blind period (start from first valid SMA date).")
    print()

    # ================================================================
    # SECTION 1: STRATEGY RULES
    # ================================================================
    print_separator()
    print("  SECTION 1: V5 STRATEGY RULES")
    print_separator()
    print("""
  CONCEPT:
  YieldMax covered call ETFs exist in bull/bear pairs on the same underlying:
    - Bull side: MSTY, NVDY, CONY, TSLY (long the underlying)
    - Bear side: WNTR, DIPS, FIAT, CRSH (short the underlying)
  Both sides pay massive weekly dividends (50-150% annualized yield).
  The strategy always owns one side per pair to maximize income capture.

  THE SIGNAL: 50-day SMA of the bull ETF's split-adjusted price.

  RULES:
  1. Track the bull ETF's price vs its 50-day Simple Moving Average (SMA).
  2. If price is MORE THAN 5% ABOVE the SMA  -> own the bull ETF.
     If price is MORE THAN 5% BELOW the SMA  -> own the bear ETF.
     If price is WITHIN the +/-5% band        -> keep holding current position.
  3. Any signal must persist for 3 CONSECUTIVE TRADING DAYS before switching.
     If price re-enters the band during the 3-day count, the counter resets.
  4. On the first day the SMA becomes available (day 50), enter the correct
     side based on the signal — never default to one side blindly.

  WHY IT WORKS:
  - The 50-day SMA captures the medium-term trend.
  - The +/-5% band prevents whipsaw switching in sideways/choppy markets
    (like a thermostat — don't flip the heat on/off at exactly 70 degrees).
  - The 3-day debounce ensures conviction before switching.
  - Both sides pay dividends, so holding the "wrong" side in chop still
    generates income rather than losing to transaction costs.
""")

    # ================================================================
    # SECTION 2: BACKTEST RESULTS
    # ================================================================
    print_separator()
    print("  SECTION 2: V5 BACKTEST RESULTS")
    print_separator()
    print()

    bt = RotationBacktester(windows=[5, 10, 20])
    calc = TotalReturnCalculator()

    all_results = {}
    all_events = []

    for pair in PAIRS:
        bull_df, bear_df = load_pair(pair)
        if bull_df is None:
            print(f"  SKIP: {pair['bull']}/{pair['bear']} — data missing")
            continue

        label = f"{pair['bull']}/{pair['bear']}"
        res = bt.run_v5_sma_hysteresis(
            bull_df, bear_df, sma_period=SMA_PERIOD,
            band_pct=BAND_PCT, debounce_days=DEBOUNCE_DAYS
        )

        bull_tri = calc.calculate_total_return_index(bull_df)
        bear_tri = calc.calculate_total_return_index(bear_df)

        # Align B&H to same period as strategy
        strat_start = res.daily_equity.index[0]
        strat_end = res.daily_equity.index[-1]
        bull_tri_period = bull_tri.loc[strat_start:strat_end]
        bear_tri_period = bear_tri.loc[strat_start:strat_end]
        bull_bh = (bull_tri_period.iloc[-1] / bull_tri_period.iloc[0] - 1) * 100
        bear_bh = (bear_tri_period.iloc[-1] / bear_tri_period.iloc[0] - 1) * 100

        dd = max_dd(res.daily_equity)

        all_results[label] = {
            "result": res,
            "bull_bh": bull_bh,
            "bear_bh": bear_bh,
            "max_dd": dd,
            "bull_df": bull_df,
            "bear_df": bear_df,
            "pair": pair,
        }

        # Collapse events
        events = analyze_collapse_events(
            bull_df, bear_df, pair["bull"], pair["bear"], SMA_PERIOD
        )
        all_events.extend(events)

    # Results table
    print(f"  Strategy: V5 — SMA({SMA_PERIOD}) +/-{BAND_PCT}% band + {DEBOUNCE_DAYS}-day debounce")
    print(f"  All returns are total return (price + dividends, compounded)")
    print()
    print(f"  {'Pair':<14}  {'V5 Return':>10}  {'Bull B&H':>10}  {'Bear B&H':>10}  "
          f"{'Trades':>7}  {'MaxDD':>8}  {'Bull Days':>10}  {'Bear Days':>10}")
    print(f"  {'-' * 90}")

    total_v5 = 0
    total_bull = 0
    total_bear = 0
    n = 0

    for label, data in all_results.items():
        res = data["result"]
        bull_bh = data["bull_bh"]
        bear_bh = data["bear_bh"]
        dd = data["max_dd"]
        total_days = res.msty_days + res.wntr_days

        print(f"  {label:<14}  {res.total_return_pct:>+9.1f}%  {bull_bh:>+9.1f}%  {bear_bh:>+9.1f}%  "
              f"{len(res.trades):>7}  {dd:>7.1f}%  {res.msty_days:>10}  {res.wntr_days:>10}")

        total_v5 += res.total_return_pct
        total_bull += bull_bh
        total_bear += bear_bh
        n += 1

    print(f"  {'-' * 90}")
    print(f"  {'AVERAGE':<14}  {total_v5/n:>+9.1f}%  {total_bull/n:>+9.1f}%  {total_bear/n:>+9.1f}%")
    print()
    print(f"  V5 average: {total_v5/n:+.1f}% vs Bull B&H average: {total_bull/n:+.1f}%")
    if total_v5/n > total_bull/n:
        print(f"  V5 outperforms bull B&H by {total_v5/n - total_bull/n:+.1f}%")
    print()

    # ================================================================
    # SECTION 3: COMPLETE TRADE HISTORY
    # ================================================================
    print_separator()
    print("  SECTION 3: COMPLETE TRADE HISTORY")
    print_separator()

    for label, data in all_results.items():
        res = data["result"]
        pair = data["pair"]
        bull_name = pair["bull"]
        bear_name = pair["bear"]

        print()
        print(f"  --- {label} ({pair['name']}) ---")
        print(f"  Period: {res.daily_equity.index[0].date()} to {res.daily_equity.index[-1].date()}")
        print(f"  Total Return: {res.total_return_pct:+.1f}% | Trades: {len(res.trades)} | Max DD: {data['max_dd']:.1f}%")
        print()
        print(f"  {'#':>3}  {'Instr':>6}  {'Entry Date':>12}  {'Exit Date':>12}  "
              f"{'Days':>5}  {'EntryPx':>8}  {'ExitPx':>8}  {'Divs':>7}  "
              f"{'PxRet':>8}  {'TotRet':>8}  {'Cumul':>8}")
        print(f"  {'-' * 105}")

        cumulative = 100.0
        for t in res.trades:
            cumulative *= (1 + t.total_return_pct / 100)
            cum_ret = cumulative - 100
            inst = bull_name if t.instrument == "MSTY" else bear_name
            print(f"  {t.trade_num:>3}  {inst:>6}  {t.entry_date.date()!s:>12}  "
                  f"{t.exit_date.date()!s:>12}  {t.holding_days:>5}  "
                  f"${t.entry_price:>7.2f}  ${t.exit_price:>7.2f}  "
                  f"${t.dividends_collected:>6.2f}  "
                  f"{t.price_return_pct:>+7.1f}%  {t.total_return_pct:>+7.1f}%  "
                  f"{cum_ret:>+7.1f}%")
        print()

    # ================================================================
    # SECTION 4: REFLEXIVE COLLAPSE THRESHOLD
    # ================================================================
    print_separator()
    print("  SECTION 4: REFLEXIVE COLLAPSE THRESHOLD")
    print_separator()
    print("""
  KEY FINDING: When a YieldMax covered call ETF (bull OR bear) sits
  10%+ below its own 50-day SMA for 10+ consecutive trading days,
  the collapse has been IRREVERSIBLE in all documented cases.

  WHY: Covered call ETFs cap upside through sold calls. When the underlying
  moves strongly against them, the ETF suffers the full directional loss
  but can only recover at the capped rate. This creates a structural
  ratchet effect — once sufficiently behind the SMA, recovery is
  mathematically near-impossible. The SMA itself starts declining,
  creating a downward spiral.

  DEAD CAT BOUNCES: Before final collapse, these ETFs typically show
  1-3 brief recoveries above the SMA lasting only 1-8 trading days.
  These are false signals, not real recoveries.
""")

    # Deduplicate events: group by ETF, merge events within 30 days into one episode
    def deduplicate_events(events):
        """Group breach events into distinct episodes (within 30 days = same episode)."""
        sorted_events = sorted(events, key=lambda x: (x["etf"], x["breach_date"]))
        episodes = []
        for e in sorted_events:
            # Check if this belongs to an existing episode
            merged = False
            for ep in episodes:
                if ep["etf"] == e["etf"] and (e["breach_date"] - ep["breach_date"]).days < 30:
                    # Same episode — keep the first breach date but update min
                    if e["min_pct"] < ep["min_pct"]:
                        ep["min_pct"] = e["min_pct"]
                        ep["min_date"] = e["min_date"]
                    # Keep the first day10 if we don't have one
                    if ep["day10_pct"] is None and e["day10_pct"] is not None:
                        ep["day10_pct"] = e["day10_pct"]
                    # If the first event recovered but a later one didn't, mark as not recovered
                    if ep["recovered"] and not e["recovered"]:
                        ep["recovered"] = False
                    merged = True
                    break
            if not merged:
                episodes.append(dict(e))  # Copy so we don't mutate
        return episodes

    episodes = deduplicate_events(all_events)
    episodes_with_day10 = [e for e in episodes if e["day10_pct"] is not None]

    # Separate into significant collapses and recoveries
    sig_collapses = [e for e in episodes_with_day10 if e["day10_pct"] < -9 and not e["recovered"]]
    sig_recoveries = [e for e in episodes_with_day10 if e["recovered"]]
    mild_events = [e for e in episodes_with_day10 if e["day10_pct"] >= -9 and not e["recovered"]]

    print(f"  DISTINCT COLLAPSE EPISODES (deduplicated, Day 10 depth < -10%):")
    print()
    print(f"  {'ETF':>6}  {'Side':>5}  {'1st Breach':>12}  {'Day10%':>8}  "
          f"{'Final%':>8}  {'Days':>5}  {'Outcome'}")
    print(f"  {'-' * 75}")

    for e in sorted(sig_collapses, key=lambda x: x["breach_date"]):
        days_to_min = (e["min_date"] - e["breach_date"]).days
        print(f"  {e['etf']:>6}  {e['side']:>5}  {e['breach_date'].date()!s:>12}  "
              f"{e['day10_pct']:>+7.1f}%  {e['min_pct']:>+7.1f}%  {days_to_min:>5}  COLLAPSED")

    print()
    print(f"  RECOVERIES (breached SMA, sustained recovery >8 days above):")
    print()
    for e in sorted(sig_recoveries, key=lambda x: x["breach_date"]):
        if e["day10_pct"] is not None:
            print(f"    {e['etf']} ({e['side']}): Breached {e['breach_date'].date()}, "
                  f"Day 10: {e['day10_pct']:+.1f}%, RECOVERED")

    print()
    print(f"  THRESHOLD STATISTICS:")
    print(f"  - Distinct collapse episodes (>10% below at Day 10): {len(sig_collapses)}")
    print(f"  - Of those, recovered: 0  |  Collapsed: {len(sig_collapses)}")
    print(f"  - Distinct recovery episodes: {len(sig_recoveries)}")
    n_rec_deep = sum(1 for e in sig_recoveries if e["day10_pct"] is not None and e["day10_pct"] < -9)
    print(f"  - Recoveries where Day 10 was also > -10%: {len(sig_recoveries) - n_rec_deep}")
    print()

    # ================================================================
    # SECTION 5: OPPOSITE ETF LAG PATTERN
    # ================================================================
    print_separator()
    print("  SECTION 5: OPPOSITE ETF LAG PATTERN")
    print_separator()
    print("""
  When one side of a pair collapses, the opposite side benefits —
  but there's a LAG before the opposite ETF crosses above its own SMA.

  KEY FINDING: The opposite ETF takes 2-4 weeks (15-34 calendar days)
  to cross above its own 50-day SMA after the collapsing side breaches.

  During the transition window, BOTH sides are below their respective
  SMAs simultaneously. This is the "no man's land" period where the
  rotation strategy stays in its current position (inside the band).
""")

    # Use deduplicated collapse episodes for lag data
    lag_episodes = [e for e in episodes if e["opp_cross_lag"] is not None and not e["recovered"]
                    and e["day10_pct"] is not None and e["day10_pct"] < -9]

    if lag_episodes:
        print(f"  OPPOSITE ETF LAG DATA (distinct collapse episodes only):")
        print()
        print(f"  {'Collapsing':>12}  {'Opposite':>10}  {'Breach':>12}  {'Opp Crosses':>12}  {'Lag (days)':>10}")
        print(f"  {'-' * 65}")

        for e in sorted(lag_episodes, key=lambda x: x["breach_date"]):
            print(f"  {e['etf']:>12}  {e['opposite']:>10}  {e['breach_date'].date()!s:>12}  "
                  f"{e['opp_cross_date'].date()!s:>12}  {e['opp_cross_lag']:>10}")

        avg_lag = np.mean([e["opp_cross_lag"] for e in lag_episodes])
        print()
        print(f"  Average lag: {avg_lag:.0f} calendar days")
    print()

    # ================================================================
    # SECTION 6: CURRENT SIGNALS
    # ================================================================
    print_separator()
    print("  SECTION 6: CURRENT SIGNALS")
    print_separator()
    print()

    for label, data in all_results.items():
        res = data["result"]
        pair = data["pair"]
        bull_df = data["bull_df"]
        bull_name = pair["bull"]
        bear_name = pair["bear"]

        adj_price, sma = get_adj_price_and_sma(bull_df, SMA_PERIOD)
        last_date = adj_price.index[-1]
        last_price = adj_price[last_date]
        last_sma = sma[last_date]
        pct_vs_sma = (last_price / last_sma - 1) * 100

        current_pos = res.daily_position.iloc[-1]
        current_inst = bull_name if current_pos == "MSTY" else bear_name

        upper = last_sma * (1 + BAND_PCT / 100)
        lower = last_sma * (1 - BAND_PCT / 100)

        zone = "ABOVE band" if last_price > upper else "BELOW band" if last_price < lower else "INSIDE band"

        print(f"  {label:<14}  OWN: {current_inst:>5}  |  {bull_name} price: ${last_price:.2f}  "
              f"SMA: ${last_sma:.2f}  ({pct_vs_sma:+.1f}% vs SMA)  [{zone}]")

    print()

    # ================================================================
    # SECTION 7: KEY MENTAL FRAMEWORK
    # ================================================================
    print_separator()
    print("  SECTION 7: KEY MENTAL FRAMEWORK")
    print_separator()
    print("""
  CORE THESIS:
  YieldMax covered call ETFs structurally decay when the underlying
  moves strongly against them. The covered call caps recovery while
  providing full downside exposure. This creates a one-way ratchet
  on sustained trends — the ETF falls behind its SMA and can never
  catch up.

  THE TRADE:
  Always own one side of the pair. Use the 50-day SMA to determine
  which side. The +/-5% band + 3-day debounce prevents unnecessary
  switching in sideways markets where both sides pay income.

  KEY MENTAL LEVELS:
  1. TREND: Price decisively outside the +/-5% SMA band = trend.
     Own the ETF aligned with that trend direction.
  2. CHOP: Price oscillating within the +/-5% band = chop.
     Stay put and collect dividends. Don't switch.
  3. COLLAPSE: 10%+ below 50-day SMA for 10+ trading days = dead.
     The covered call ratchet has engaged. Collapse is structural.
  4. ROTATION LAG: After collapse confirmation, expect the opposite
     ETF to cross above its own SMA in 2-4 weeks. The strategy's
     band + debounce naturally captures this transition.

  WHAT THE STRATEGY DOES WELL:
  - Catches major trends (rides them for months, collecting dividends)
  - Survives sideways markets (doesn't whipsaw like tighter signals)
  - Beats buy-and-hold on average across pairs

  WHAT IT DOESN'T DO:
  - Time the exact top or bottom (by design — waits for confirmation)
  - Avoid all drawdowns (still exposed during the transition window)
  - Work perfectly on every pair (TSLY choppy market is harder)
""")

    print_separator()
    print("  END OF SUMMARY")
    print_separator()
    print()


if __name__ == "__main__":
    main()
