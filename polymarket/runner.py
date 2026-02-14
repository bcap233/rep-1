#!/usr/bin/env python3
"""
Polymarket Arbitrage Bot — Main Runner.

Usage:
    python -m polymarket --setup                            # First-time setup wizard
    python -m polymarket                                    # Run one scan (all strategies)
    python -m polymarket --loop                             # Run continuously
    python -m polymarket --strategy spot_divergence --loop  # Single strategy
    python -m polymarket --strategy high_prob_grinder       # Grinder only
    python -m polymarket --strategy bilateral_arb           # Bilateral arb only
    python -m polymarket --strategy all --loop              # All strategies (default)
    python -m polymarket --assets BTC,ETH                   # Specific assets
    python -m polymarket --markets                          # List available markets
    python -m polymarket --status                           # Show open positions
    python -m polymarket --close-all                        # Close all positions
    python -m polymarket --strategies                       # List available strategies
    python -m polymarket --paper                            # Force paper mode
    python -m polymarket --live                             # Force live mode

Strategies:
    spot_divergence    - Spot momentum vs Polymarket probability lag
    high_prob_grinder  - Buy >90% events at scale, collect small edges
    bilateral_arb      - Buy both sides when YES+NO < $1.00
    all                - Run all strategies each cycle
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
from .signals import Signal, generate_signals
from .execution import ExecutionEngine
from .risk import RiskManager
from .strategies import get_strategy, list_strategies, STRATEGIES, BaseStrategy

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


# ============================================================
# Display helpers
# ============================================================

def print_banner(strategy_names: list[str]):
    print()
    print("=" * 80)
    print("  POLYMARKET ARBITRAGE BOT")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80)
    print()

    if len(strategy_names) == 1:
        name = strategy_names[0]
        if name in STRATEGIES:
            print(f"  Strategy: {name} — {STRATEGIES[name].description}")
        else:
            print(f"  Strategy: {name}")
    else:
        print(f"  Strategies: {', '.join(strategy_names)}")
        for name in strategy_names:
            if name in STRATEGIES:
                print(f"    - {name}: {STRATEGIES[name].description}")

    print(f"  Mode: {EXECUTION['mode'].upper()}")
    print(f"  Assets: {', '.join(ASSETS.keys())}")
    print(f"  Risk: max ${RISK['max_position_usdc']}/trade, "
          f"${RISK['max_total_exposure_usdc']} total, "
          f"SL {RISK['stop_loss']*100:.0f}c / TP {RISK['take_profit']*100:.0f}c")
    print()


def print_strategies():
    """Print all available strategies."""
    print()
    print("  Available Strategies:")
    print(f"  {'-' * 60}")
    for name, desc in list_strategies():
        print(f"  {name:<25} {desc}")
    print()
    print(f"  {'all':<25} Run all strategies each cycle")
    print()
    print("  Usage: python -m polymarket --strategy <name> [--loop]")
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


def print_signals(signals: list[Signal], strategy_name: str = ""):
    """Print detected signals."""
    if strategy_name:
        header = f"  [{strategy_name.upper()}] "
    else:
        header = "  "

    if not signals:
        print(f"{header}No signals detected.")
        print()
        return

    print(f"{header}Found {len(signals)} signal(s):")
    print()

    # Group by strategy for multi-strategy runs
    by_strategy: dict[str, list[Signal]] = {}
    for s in signals:
        strat = getattr(s, 'strategy', 'unknown')
        by_strategy.setdefault(strat, []).append(s)

    for strat, strat_signals in by_strategy.items():
        if len(by_strategy) > 1:
            print(f"  --- {strat.upper()} ({len(strat_signals)}) ---")

        # Adapt columns based on strategy type
        if strat == "bilateral_arb":
            print(f"  {'Asset':<6} {'Type':>12} {'Market':>35} {'Gap':>6} "
                  f"{'Cost':>6} {'Profit':>7}")
            print(f"  {'-' * 78}")
            for s in strat_signals:
                market_short = s.market.question[:33]
                profit = s.edge * s.suggested_size
                print(f"  {s.asset:<6} {s.token_side:>12} "
                      f"{market_short:>35} {s.edge:>5.3f} "
                      f"${s.suggested_price:>4.2f} ${profit:>+5.2f}")
                # Print legs
                for leg in s.arb_legs:
                    print(f"    -> {leg['label'][:30]:<32} @ ${leg['price']:.2f}")

        elif strat == "high_prob_grinder":
            print(f"  {'Asset':<6} {'Price':>6} {'EV/sh':>7} {'Market':>45} "
                  f"{'Size':>5}")
            print(f"  {'-' * 78}")
            for s in strat_signals:
                market_short = s.market.question[:43]
                print(f"  {s.asset:<6} ${s.suggested_price:>4.2f} "
                      f"${s.edge:>+5.4f} {market_short:>45} "
                      f"{s.suggested_size:>5.0f}")

        else:  # spot_divergence or unknown
            print(f"  {'Asset':<6} {'Side':>8} {'Market':>35} {'Edge':>6} "
                  f"{'Conf':>5} {'Price':>6} {'Size':>5}")
            print(f"  {'-' * 78}")
            for s in strat_signals:
                market_short = s.market.question[:33]
                print(f"  {s.asset:<6} "
                      f"{s.side + ' ' + s.token_side:>8} "
                      f"{market_short:>35} {s.edge:>5.3f} "
                      f"{s.confidence:>5.2f} "
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
        print(f"  {'ID':<16} {'Asset':<6} {'Side':>12} {'Entry':>6} {'Now':>6} "
              f"{'PnL':>8} {'SL':>5} {'TP':>5}")
        print(f"  {'-' * 70}")
        for p in positions:
            sl = f"{p['stop']:>5.2f}" if p['stop'] > 0 else "  n/a"
            tp = f"{p['tp']:>5.2f}" if p['tp'] > 0 else "  n/a"
            print(f"  {p['id']:<16} {p['asset']:<6} {p['side']:>12} "
                  f"${p['entry']:>4.2f} ${p['current']:>4.2f} "
                  f"${p['pnl']:>+6.2f} {sl} {tp}")

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


# ============================================================
# Core cycle
# ============================================================

def run_cycle(strategies: list[BaseStrategy], engine: ExecutionEngine,
              risk_mgr: RiskManager, assets: list[str],
              quiet: bool = False) -> int:
    """
    Run one complete scan-and-trade cycle across all active strategies.

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

    # Step 3: Collect signals from all strategies
    all_signals = []
    for strategy in strategies:
        try:
            logger.info(f"--- Running {strategy.name} ---")
            signals = strategy.scan(assets)
            all_signals.extend(signals)
            logger.info(f"  {strategy.name}: {len(signals)} signals")
        except Exception as e:
            logger.error(f"Strategy {strategy.name} error: {e}", exc_info=True)

    if not quiet:
        print_signals(all_signals)

    # Step 4: Execute signals (best first, across all strategies)
    # Sort all signals by edge * confidence for unified prioritization
    all_signals.sort(key=lambda s: s.edge * s.confidence, reverse=True)

    for sig in all_signals:
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


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Arbitrage Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
First-time setup:
  python -m polymarket --setup

Strategies:
  spot_divergence    Spot momentum vs Polymarket probability lag
  high_prob_grinder  Buy >90%% events at scale, collect small edges
  bilateral_arb      Buy both sides when YES+NO < $1.00
  all                Run all strategies each cycle (default)

Examples:
  python -m polymarket --setup                                   # Setup wizard
  python -m polymarket --strategy bilateral_arb --loop           # Run bilateral arb
  python -m polymarket --strategy high_prob_grinder --paper      # Paper trade grinder
  python -m polymarket --strategy all --assets BTC --loop --interval 15
        """,
    )
    parser.add_argument("--strategy", type=str, default="all",
                        help="Strategy to run: spot_divergence, high_prob_grinder, "
                             "bilateral_arb, all (default: all)")
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
    parser.add_argument("--strategies", action="store_true",
                        help="List available strategies and exit")
    parser.add_argument("--paper", action="store_true",
                        help="Force paper trading mode")
    parser.add_argument("--live", action="store_true",
                        help="Force live trading mode")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress dashboard output")
    parser.add_argument("--prices", action="store_true",
                        help="Show current spot prices and exit")
    parser.add_argument("--dashboard", action="store_true",
                        help="Run with live terminal dashboard")
    parser.add_argument("--setup", action="store_true",
                        help="Run interactive setup wizard (wallet, API creds, funding)")
    args = parser.parse_args()

    # Setup wizard
    if args.setup:
        from .setup import run_setup
        run_setup()
        return

    # List strategies
    if args.strategies:
        print_strategies()
        return

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

    # Resolve strategy selection
    strategy_name = args.strategy.lower()
    if strategy_name == "all":
        strategy_names = list(STRATEGIES.keys())
    else:
        # Support comma-separated strategies
        strategy_names = [s.strip() for s in strategy_name.split(",")]

    # Instantiate strategies
    active_strategies: list[BaseStrategy] = []
    for name in strategy_names:
        try:
            strat = get_strategy(name, client)
            active_strategies.append(strat)
        except ValueError as e:
            print(f"Error: {e}")
            print_strategies()
            return

    # Handle simple commands
    if args.prices:
        print_banner(strategy_names)
        print_prices(assets)
        return

    if args.markets:
        print_banner(strategy_names)
        list_markets(client, assets)
        return

    if args.status:
        print_banner(strategy_names)
        print_positions(engine)
        print_risk_report(risk_mgr, engine)
        return

    if args.close_all:
        engine.close_all()
        print("All positions closed.")
        return

    # Main execution
    if args.dashboard:
        from .dashboard import run_dashboard_with_bot
        run_dashboard_with_bot(run_cycle, active_strategies, engine, risk_mgr,
                               assets, interval)
        return

    print_banner(strategy_names)

    if not args.quiet:
        print_prices(assets)

    if args.loop:
        logger.info(f"Starting continuous loop (interval={interval}s, "
                     f"strategies={','.join(strategy_names)})")
        cycle = 0

        while not _shutdown:
            cycle += 1
            logger.info(f"=== Cycle {cycle} ===")

            try:
                trades = run_cycle(active_strategies, engine, risk_mgr,
                                   assets, quiet=args.quiet)

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
                for _ in range(interval):
                    if _shutdown:
                        break
                    time.sleep(1)

        logger.info("Shutting down...")
        engine.close_all(reason="shutdown")
        logger.info("All positions closed. Goodbye.")

    else:
        # Single cycle
        trades = run_cycle(active_strategies, engine, risk_mgr,
                           assets, quiet=args.quiet)

        if not args.quiet:
            print_positions(engine)
            print_risk_report(risk_mgr, engine)

        if trades:
            logger.info(f"Executed {trades} trade(s)")
        else:
            logger.info("No trades this cycle")


if __name__ == "__main__":
    main()
