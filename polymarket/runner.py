#!/usr/bin/env python3
"""
Polymarket Arbitrage Bot — Main Runner.

Usage:
    python -m polymarket                      # Run one scan cycle
    python -m polymarket --loop               # Run continuously
    python -m polymarket --loop --interval 15 # Custom poll interval (seconds)
    python -m polymarket --assets BTC,ETH     # Specific assets only
    python -m polymarket --markets            # List available markets
    python -m polymarket --status             # Show open positions
    python -m polymarket --close-all          # Close all positions
    python -m polymarket --paper              # Force paper mode
    python -m polymarket --live               # Force live mode (requires API keys)

The bot:
  1. Fetches spot prices from Binance/Coinbase/Kraken
  2. Builds 5/10/15 minute charts and computes momentum indicators
  3. Finds relevant Polymarket prediction markets
  4. Compares spot momentum vs Polymarket implied probabilities
  5. Places trades when mispricing exceeds threshold
"""

import argparse
import json
import logging
import signal as sys_signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import STRATEGY, ASSETS, EXECUTION, RISK, DATA, TIMEFRAMES
from .exchanges import get_composite_price
from .polymarket_client import PolymarketClient
from .signals import generate_signals
from .execution import ExecutionEngine
from .risk import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Graceful shutdown
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received, finishing current cycle...")
    _shutdown = True


sys_signal.signal(sys_signal.SIGINT, _handle_signal)
sys_signal.signal(sys_signal.SIGTERM, _handle_signal)


def print_banner():
    print()
    print("=" * 80)
    print("  POLYMARKET ARBITRAGE BOT")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80)
    print()
    print(f"  Strategy: Spot momentum vs Polymarket implied probability")
    print(f"  Timeframes: {', '.join(str(t) + 'm' for t in TIMEFRAMES)}")
    print(f"  Min edge: {STRATEGY['min_edge']*100:.0f}c | "
          f"Momentum threshold: {STRATEGY['momentum_threshold_pct']}%")
    print(f"  Assets: {', '.join(ASSETS.keys())}")
    print(f"  Mode: {EXECUTION['mode'].upper()}")
    print(f"  Risk: max ${RISK['max_position_usdc']}/trade, "
          f"${RISK['max_total_exposure_usdc']} total, "
          f"SL {RISK['stop_loss']*100:.0f}c / TP {RISK['take_profit']*100:.0f}c")
    print()


def print_prices(assets: list[str]):
    """Print current spot prices."""
    print(f"  {'Asset':<6} {'Spot Price':>12} {'Sources':>8}")
    print(f"  {'-' * 30}")
    for asset in assets:
        comp = get_composite_price(asset)
        if comp:
            print(f"  {asset:<6} ${comp.price:>11,.2f} {len(comp.sources):>6}")
        else:
            print(f"  {asset:<6} {'N/A':>12}")
    print()


def print_signals(signals):
    """Print detected signals."""
    if not signals:
        print("  No arbitrage signals detected.")
        print()
        return

    print(f"  Found {len(signals)} signal(s):")
    print()
    print(f"  {'Asset':<6} {'Side':>8} {'Market':>40} {'Edge':>6} {'Conf':>5} "
          f"{'Price':>6} {'Size':>5}")
    print(f"  {'-' * 80}")

    for s in signals:
        market_short = s.market.question[:38]
        print(f"  {s.asset:<6} {s.side + ' ' + s.token_side:>8} "
              f"{market_short:>40} {s.edge:>5.3f} {s.confidence:>5.2f} "
              f"${s.suggested_price:>4.2f} {s.suggested_size:>5.0f}")

    print()


def print_positions(engine: ExecutionEngine):
    """Print open positions."""
    summary = engine.get_summary()
    positions = summary["positions"]

    if not positions:
        print("  No open positions.")
    else:
        print(f"  Open positions: {summary['open_positions']}")
        print(f"  Total exposure: ${summary['total_exposure_usdc']:.2f}")
        print(f"  Unrealized PnL: ${summary['unrealized_pnl']:+.2f}")
        print()
        print(f"  {'ID':<16} {'Asset':<6} {'Side':>8} {'Entry':>6} {'Now':>6} "
              f"{'PnL':>8} {'SL':>5} {'TP':>5}")
        print(f"  {'-' * 70}")
        for p in positions:
            print(f"  {p['id']:<16} {p['asset']:<6} {p['side']:>8} "
                  f"${p['entry']:>4.2f} ${p['current']:>4.2f} "
                  f"${p['pnl']:>+6.2f} {p['stop']:>5.2f} {p['tp']:>5.2f}")

    print()
    print(f"  Daily realized PnL: ${summary['daily_realized_pnl']:+.2f}")
    print(f"  Daily trades: {summary['daily_trades']}")
    print()


def print_risk_report(risk_mgr: RiskManager, engine: ExecutionEngine):
    """Print risk summary."""
    report = risk_mgr.assess(engine.state)

    print(f"  Exposure: ${report.total_exposure:.2f} / ${report.max_exposure:.2f} "
          f"({report.utilization_pct:.0f}%)")
    print(f"  Positions: {report.open_positions} / {report.max_positions}")
    print(f"  Daily PnL: ${report.daily_pnl:+.2f} / "
          f"-${report.daily_loss_limit:.2f} limit")
    print(f"  Drawdown: ${report.drawdown_from_peak:.2f}")

    if report.kill_switch_active:
        print(f"  *** KILL SWITCH ACTIVE ***")

    for w in report.warnings:
        print(f"  WARNING: {w}")

    print()


def list_markets(client: PolymarketClient, assets: list[str]):
    """List available Polymarket markets for tracked assets."""
    for asset in assets:
        markets = client.find_crypto_price_markets(asset)
        print(f"\n  {asset} — {len(markets)} market(s):")
        for m in markets[:10]:
            book = client.get_order_book(m.yes_token_id) if m.yes_token_id else None
            price_str = f"${book.midpoint:.2f}" if book else "N/A"
            print(f"    {price_str:>6} | {m.question[:70]}")
            if book:
                print(f"           spread={book.spread:.3f} | "
                      f"vol=${m.volume:,.0f} | liq=${m.liquidity:,.0f}")
    print()


def run_cycle(client: PolymarketClient, engine: ExecutionEngine,
              risk_mgr: RiskManager, assets: list[str],
              quiet: bool = False) -> int:
    """
    Run one complete scan-and-trade cycle.

    Returns the number of trades executed.
    """
    trades_executed = 0

    # Step 1: Update existing positions (check stops/TPs)
    closed = engine.update_positions()
    if closed:
        for pos in closed:
            logger.info(f"Position closed: {pos.position_id} "
                         f"reason={pos.exit_reason} PnL=${pos.realized_pnl:+.2f}")

    # Step 2: Risk check
    report = risk_mgr.assess(engine.state)
    if report.kill_switch_active:
        logger.warning("Kill switch active — skipping signal generation")
        return 0

    # Step 3: Generate signals
    signals = generate_signals(client, assets)

    if not quiet:
        print_signals(signals)

    # Step 4: Execute signals (best first)
    for sig in signals:
        # Portfolio-level risk check
        allowed, reason = risk_mgr.pre_trade_check(sig, engine.state)
        if not allowed:
            logger.info(f"Signal filtered by risk manager: {reason}")
            continue

        # Adjust size
        adjusted_size = risk_mgr.adjust_size(sig, engine.state)
        sig.suggested_size = adjusted_size

        # Execute
        position = engine.execute_signal(sig)
        if position:
            trades_executed += 1

    return trades_executed


def main():
    parser = argparse.ArgumentParser(description="Polymarket Arbitrage Bot")
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously")
    parser.add_argument("--interval", type=int, default=None,
                        help="Poll interval in seconds (default: from config)")
    parser.add_argument("--assets", type=str, default=None,
                        help="Comma-separated list of assets (e.g., BTC,ETH)")
    parser.add_argument("--markets", action="store_true",
                        help="List available Polymarket markets and exit")
    parser.add_argument("--status", action="store_true",
                        help="Show current positions and exit")
    parser.add_argument("--close-all", action="store_true",
                        help="Close all open positions and exit")
    parser.add_argument("--paper", action="store_true",
                        help="Force paper trading mode")
    parser.add_argument("--live", action="store_true",
                        help="Force live trading mode")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress dashboard output")
    parser.add_argument("--prices", action="store_true",
                        help="Show current spot prices and exit")
    args = parser.parse_args()

    # Determine assets
    if args.assets:
        assets = [a.strip().upper() for a in args.assets.split(",")]
    else:
        assets = list(ASSETS.keys())

    # Override execution mode
    if args.paper:
        EXECUTION["mode"] = "paper"
    elif args.live:
        EXECUTION["mode"] = "live"

    # Initialize components
    client = PolymarketClient()
    engine = ExecutionEngine(client)
    risk_mgr = RiskManager()

    interval = args.interval or EXECUTION["poll_interval"]

    # Handle simple commands
    if args.prices:
        print_banner()
        print_prices(assets)
        return

    if args.markets:
        print_banner()
        list_markets(client, assets)
        return

    if args.status:
        print_banner()
        print_positions(engine)
        print_risk_report(risk_mgr, engine)
        return

    if args.close_all:
        engine.close_all()
        print("All positions closed.")
        return

    # Main execution
    print_banner()

    if not args.quiet:
        print_prices(assets)

    if args.loop:
        logger.info(f"Starting continuous loop (interval={interval}s)")
        cycle = 0

        while not _shutdown:
            cycle += 1
            logger.info(f"--- Cycle {cycle} ---")

            try:
                trades = run_cycle(client, engine, risk_mgr, assets,
                                   quiet=args.quiet)

                if not args.quiet:
                    print_positions(engine)
                    print_risk_report(risk_mgr, engine)

                if trades:
                    logger.info(f"Executed {trades} trade(s)")
                else:
                    logger.info("No trades this cycle")

            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)

            if not _shutdown:
                logger.info(f"Sleeping {interval}s...")
                # Sleep in small increments so we can catch shutdown signals
                for _ in range(interval):
                    if _shutdown:
                        break
                    time.sleep(1)

        logger.info("Shutting down...")
        engine.close_all(reason="shutdown")
        logger.info("All positions closed. Goodbye.")

    else:
        # Single cycle
        trades = run_cycle(client, engine, risk_mgr, assets,
                           quiet=args.quiet)

        if not args.quiet:
            print_positions(engine)
            print_risk_report(risk_mgr, engine)

        if trades:
            logger.info(f"Executed {trades} trade(s)")
        else:
            logger.info("No trades this cycle")


if __name__ == "__main__":
    main()
