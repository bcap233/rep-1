#!/usr/bin/env python3
"""
V7 Rotation Bot — Daily Runner.

Usage:
    python -m bot                         # Run signals for all pairs
    python -m bot --pair MSTY             # Run for one pair only
    python -m bot --local                 # Use local CSV data (offline/testing)
    python -m bot --execute              # Execute trades via Alpaca
    python -m bot --status               # Show saved state
    python -m bot --notify               # Send webhook notifications

Run this daily after market close (e.g., 5pm ET via cron).
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from .config import STRATEGY, PAIRS, ALPACA, NOTIFICATIONS, DATA, EXECUTION
from .data import fetch_pair_data
from .signal import evaluate_pair, SignalState
from .notify import (
    send_webhook, format_switch_alert, format_filter_alert, format_daily_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_state(state_file: str) -> dict:
    """Load persisted state from disk."""
    path = Path(state_file)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_state(state_file: str, state: dict):
    """Persist state to disk."""
    path = Path(state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def run_signals(pairs_filter: str = None, use_local: bool = False) -> list[SignalState]:
    """
    Fetch data and compute V7 signals for all enabled pairs.
    Returns list of SignalState objects.
    """
    states = []

    for pair in PAIRS:
        if not pair.get("enabled", True):
            continue
        if pairs_filter and pair["bull"].upper() != pairs_filter.upper():
            continue

        name = pair["name"]
        bull = pair["bull"]
        bear = pair["bear"]

        logger.info(f"Processing {name} ({bull}/{bear})...")

        bull_df, bear_df = fetch_pair_data(
            bull, bear, lookback_days=DATA["lookback_days"],
            use_local=use_local,
        )

        if bull_df is None:
            logger.error(f"  Could not fetch data for {bull}/{bear}")
            continue

        state = evaluate_pair(
            bull_df=bull_df,
            pair_name=name,
            bull_ticker=bull,
            bear_ticker=bear,
            sma_period=STRATEGY["sma_period"],
            band_pct=STRATEGY["band_pct"],
            debounce_days=STRATEGY["debounce_days"],
            rv_window=STRATEGY["rv_window"],
            income_weeks=STRATEGY["income_weeks"],
            min_ratio=STRATEGY["min_income_rv_ratio"],
        )

        states.append(state)

    return states


def check_for_switches(
    states: list[SignalState],
    prev_state: dict,
) -> list[tuple[str, SignalState]]:
    """
    Compare current signals with previous state.
    Returns list of (old_holding, new_state) for pairs that switched.
    """
    switches = []

    for state in states:
        key = state.pair_name
        old_holding = prev_state.get(key, {}).get("holding")

        if old_holding and old_holding != state.holding:
            switches.append((old_holding, state))
            state.action = f"SWITCH_TO_{'BULL' if state.position == 'BULL' else 'BEAR'}"
            state.switch_to = state.holding
        else:
            state.action = "HOLD"

    return switches


def print_dashboard(states: list[SignalState]):
    """Print a formatted dashboard to the terminal."""
    print()
    print("=" * 90)
    print("  V7 ROTATION BOT — SIGNAL DASHBOARD")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    print()
    print(f"  Strategy: SMA({STRATEGY['sma_period']}) ±{STRATEGY['band_pct']}% "
          f"+ Debounce({STRATEGY['debounce_days']}d) "
          f"+ Income/RV > {STRATEGY['min_income_rv_ratio']}")
    print()

    print(f"  {'Pair':<12} {'Holding':>7} {'Price':>9} {'SMA':>9} {'%SMA':>7} "
          f"{'RV':>5} {'Inc':>5} {'I/RV':>5} {'Signal':>6} {'Pending':>10} {'Filter':>8}")
    print(f"  {'-' * 88}")

    for s in states:
        pend = f"{s.pending}({s.debounce_count}d)" if s.pending else "—"
        filt = "BLOCKED" if s.filter_blocked else ("ACTIVE" if s.filter_active else "—")

        print(f"  {s.pair_name:<12} {s.holding:>7} ${s.price:>8.2f} ${s.sma:>8.2f} "
              f"{s.pct_sma:>+6.1f}% {s.rv:>4.0f}% {s.income_yield:>4.0f}% "
              f"{s.income_rv_ratio:>5.2f} {s.raw_signal:>6} {pend:>10} {filt:>8}")

    print()

    # Show any pending switches
    for s in states:
        if s.action.startswith("SWITCH"):
            print(f"  *** ACTION: {s.pair_name} → Switch to {s.switch_to} ***")
        if s.filter_blocked:
            print(f"  *** FILTER: {s.pair_name} bear switch blocked "
                  f"(Inc/RV = {s.income_rv_ratio:.2f} < {STRATEGY['min_income_rv_ratio']}) ***")

    print()


def execute_trades(switches: list[tuple[str, SignalState]]):
    """Execute trades via the brokerage for any switches."""
    if not ALPACA["api_key"]:
        logger.warning("No Alpaca API key configured. Set ALPACA credentials in bot/config.py")
        return

    from .broker import AlpacaBroker

    broker = AlpacaBroker(
        api_key=ALPACA["api_key"],
        secret_key=ALPACA["secret_key"],
        base_url=ALPACA["base_url"],
    )

    # Verify account access
    acct = broker.get_account()
    if not acct:
        logger.error("Could not connect to Alpaca. Check credentials.")
        return

    logger.info(f"Alpaca account: equity=${float(acct.get('equity', 0)):,.2f} "
                f"buying_power=${float(acct.get('buying_power', 0)):,.2f}")

    for old_holding, state in switches:
        new_holding = state.holding
        logger.info(f"Executing: {old_holding} → {new_holding}")

        # Check if we actually hold the old position
        pos = broker.get_position(old_holding)
        if pos:
            # Close old position
            sell_result = broker.close_position(old_holding)
            if sell_result.success:
                logger.info(f"  Closed {old_holding}: order {sell_result.order_id}")
            else:
                logger.error(f"  Failed to close {old_holding}: {sell_result.error}")
                continue
        else:
            logger.info(f"  No existing position in {old_holding} to close")

        # Buy new position
        # Get current buying power for qty calculation
        acct = broker.get_account()
        if acct:
            buying_power = float(acct.get("buying_power", 0))
            # Use position_size fraction
            alloc = buying_power * EXECUTION["position_size"] / len(PAIRS)
            # Rough qty estimate (will need current price)
            est_price = state.price  # Use signal engine's last price
            if est_price > 0:
                qty = int(alloc / est_price)
                if qty > 0:
                    buy_result = broker.market_order(new_holding, "buy", qty)
                    if buy_result.success:
                        logger.info(f"  Bought {qty} {new_holding}: order {buy_result.order_id}")
                    else:
                        logger.error(f"  Failed to buy {new_holding}: {buy_result.error}")
                else:
                    logger.warning(f"  Calculated qty=0 for {new_holding} "
                                   f"(alloc=${alloc:.2f}, price=${est_price:.2f})")


def main():
    parser = argparse.ArgumentParser(description="V7 Rotation Bot")
    parser.add_argument("--pair", type=str, help="Run for a specific pair only (bull ticker)")
    parser.add_argument("--local", action="store_true", help="Use local CSV data instead of fetching")
    parser.add_argument("--execute", action="store_true", help="Execute trades (requires Alpaca config)")
    parser.add_argument("--status", action="store_true", help="Show current state without fetching new data")
    parser.add_argument("--notify", action="store_true", help="Send webhook notification with results")
    parser.add_argument("--quiet", action="store_true", help="Suppress dashboard output")
    args = parser.parse_args()

    state_file = DATA["state_file"]

    if args.status:
        prev = load_state(state_file)
        if not prev:
            print("No saved state. Run the bot first.")
            return
        print(json.dumps(prev, indent=2))
        return

    # Run signals
    source = "local CSV" if args.local else "Yahoo Finance"
    logger.info(f"Fetching data from {source} and computing signals...")
    states = run_signals(pairs_filter=args.pair, use_local=args.local)

    if not states:
        logger.error("No signals computed. Check data availability.")
        return

    # Load previous state and detect switches
    prev = load_state(state_file)
    switches = check_for_switches(states, prev)

    # Print dashboard
    if not args.quiet:
        print_dashboard(states)

    # Save current state
    new_state = {}
    for s in states:
        new_state[s.pair_name] = {
            "holding": s.holding,
            "position": s.position,
            "date": str(s.date.date()),
            "price": s.price,
            "pct_sma": round(s.pct_sma, 2),
            "rv": round(s.rv, 1),
            "income_yield": round(s.income_yield, 1),
            "income_rv_ratio": round(s.income_rv_ratio, 2),
            "raw_signal": s.raw_signal,
            "pending": s.pending,
            "debounce_count": s.debounce_count,
            "filter_active": s.filter_active,
            "action": s.action,
        }
    save_state(state_file, new_state)
    logger.info(f"State saved to {state_file}")

    # Notifications
    if switches:
        for old_holding, state in switches:
            msg = format_switch_alert(old_holding, state)
            logger.info(f"SWITCH: {old_holding} → {state.holding}")
            print(f"\n  *** SWITCH DETECTED: {old_holding} → {state.holding} ***\n")
            if args.notify and NOTIFICATIONS["webhook_url"]:
                send_webhook(NOTIFICATIONS["webhook_url"], msg,
                             title=f"Position Switch: {state.pair_name}")

    # Check for filter blocks
    for state in states:
        if state.filter_blocked:
            msg = format_filter_alert(state)
            logger.info(f"FILTER BLOCKED: {state.pair_name} bear switch blocked")
            if args.notify and NOTIFICATIONS["webhook_url"]:
                send_webhook(NOTIFICATIONS["webhook_url"], msg,
                             title=f"Filter Blocked: {state.pair_name}")

    # Daily summary notification
    if args.notify and NOTIFICATIONS["webhook_url"]:
        summary = format_daily_summary(states)
        send_webhook(NOTIFICATIONS["webhook_url"], summary, title="Daily Signals")

    # Execute trades if requested
    if args.execute or EXECUTION["mode"] == "live":
        if switches:
            if args.execute:
                logger.info("Executing trades...")
                execute_trades(switches)
            else:
                logger.info("Live mode enabled but --execute flag not passed. "
                            "Use --execute to place orders.")
        else:
            logger.info("No switches — no trades to execute.")


if __name__ == "__main__":
    main()
