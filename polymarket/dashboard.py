"""
Polymarket Arbitrage Bot — Live Terminal Dashboard.

A rich TUI that watches state.json and trades.jsonl to display
real-time trading activity alongside the running bot loop.

Usage:
    python -m polymarket --dashboard          # Dashboard + trading loop
    python -m polymarket.dashboard            # Standalone viewer (read-only)
"""

import json
import logging
import os
import signal as sys_signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import ASSETS, DATA, EXECUTION, RISK

logger = logging.getLogger(__name__)

# ============================================================
# State reader — polls state.json and trades.jsonl
# ============================================================


def _load_state() -> dict:
    """Load current state from disk."""
    path = Path(DATA["state_file"])
    if not path.exists():
        return {"positions": [], "daily_pnl": 0, "daily_trades": 0,
                "cooldowns": {}, "last_update": 0}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"positions": [], "daily_pnl": 0, "daily_trades": 0,
                "cooldowns": {}, "last_update": 0}


def _load_trades(limit: int = 20) -> list[dict]:
    """Load recent trades from the JSONL log."""
    path = Path(DATA["trade_log"])
    if not path.exists():
        return []
    try:
        lines = path.read_text().strip().splitlines()
        trades = []
        for line in lines[-limit:]:
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return trades
    except OSError:
        return []


# ============================================================
# Widget builders
# ============================================================


def _make_header(cycle: int, start_time: float, mode: str) -> Panel:
    """Top header bar."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    uptime_s = int(time.time() - start_time)
    h, rem = divmod(uptime_s, 3600)
    m, s = divmod(rem, 60)
    uptime = f"{h}h {m:02d}m {s:02d}s"

    title = Text()
    title.append("  POLYMARKET ARBITRAGE BOT  ", style="bold white")

    info = Text()
    info.append(f"  {now}  |  ", style="dim")
    if mode == "paper":
        info.append("PAPER", style="bold yellow")
    else:
        info.append("LIVE", style="bold red")
    info.append(f"  |  Cycle {cycle}  |  Uptime {uptime}", style="dim")

    return Panel(
        Group(title, info),
        style="blue",
        padding=(0, 1),
    )


def _make_portfolio(state: dict) -> Panel:
    """Portfolio summary panel."""
    positions = state.get("positions", [])
    open_pos = [p for p in positions if p.get("status") == "open"]
    total_exposure = sum(p.get("entry_cost", 0) for p in open_pos)
    unrealized = sum(p.get("unrealized_pnl", 0) for p in open_pos)
    daily_pnl = state.get("daily_pnl", 0)
    daily_trades = state.get("daily_trades", 0)
    max_exposure = RISK["max_total_exposure_usdc"]
    util_pct = (total_exposure / max_exposure * 100) if max_exposure > 0 else 0

    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_column("label", style="dim", min_width=14)
    tbl.add_column("value", min_width=18)
    tbl.add_column("label2", style="dim", min_width=14)
    tbl.add_column("value2", min_width=18)

    # Row 1
    pnl_style = "green" if daily_pnl >= 0 else "red"
    upnl_style = "green" if unrealized >= 0 else "red"
    tbl.add_row(
        "Exposure",
        f"${total_exposure:,.2f} / ${max_exposure:,.0f}  ({util_pct:.0f}%)",
        "Daily PnL",
        Text(f"${daily_pnl:+,.2f}", style=pnl_style),
    )
    # Row 2
    tbl.add_row(
        "Positions",
        f"{len(open_pos)} / {RISK['max_open_positions']}",
        "Unrealized",
        Text(f"${unrealized:+,.2f}", style=upnl_style),
    )
    # Row 3
    loss_limit = RISK["daily_loss_limit_usdc"]
    remaining = loss_limit + daily_pnl
    rem_style = "green" if remaining > loss_limit * 0.5 else ("yellow" if remaining > 0 else "red")
    tbl.add_row(
        "Trades Today",
        str(daily_trades),
        "Loss Limit",
        Text(f"${remaining:,.2f} remaining", style=rem_style),
    )

    return Panel(tbl, title="Portfolio", border_style="cyan", padding=(0, 1))


def _make_positions(state: dict) -> Panel:
    """Open positions table."""
    positions = state.get("positions", [])
    open_pos = [p for p in positions if p.get("status") == "open"]

    tbl = Table(box=None, padding=(0, 1), expand=True)
    tbl.add_column("ID", style="dim", max_width=16)
    tbl.add_column("Asset", style="bold", max_width=5)
    tbl.add_column("Side", max_width=10)
    tbl.add_column("Market", max_width=40)
    tbl.add_column("Entry", justify="right", max_width=7)
    tbl.add_column("Now", justify="right", max_width=7)
    tbl.add_column("PnL", justify="right", max_width=9)
    tbl.add_column("SL", justify="right", style="dim", max_width=6)
    tbl.add_column("TP", justify="right", style="dim", max_width=6)

    if not open_pos:
        tbl.add_row("", "", "", Text("No open positions", style="dim"),
                     "", "", "", "", "")
    else:
        for p in open_pos:
            pnl = p.get("unrealized_pnl", 0)
            pnl_style = "green" if pnl >= 0 else "red"
            side_str = f"{p.get('side', '')} {p.get('token_side', '')}"

            sl = p.get("stop_loss_price", 0)
            tp = p.get("take_profit_price", 0)
            sl_str = f"${sl:.2f}" if sl > 0 else "-"
            tp_str = f"${tp:.2f}" if tp > 0 else "-"

            market = p.get("market_question", "")[:38]

            tbl.add_row(
                p.get("position_id", "")[-12:],
                p.get("asset", ""),
                side_str,
                market,
                f"${p.get('entry_price', 0):.2f}",
                f"${p.get('current_price', 0):.2f}",
                Text(f"${pnl:+.2f}", style=pnl_style),
                sl_str,
                tp_str,
            )

    return Panel(tbl, title=f"Open Positions ({len(open_pos)})",
                 border_style="green", padding=(0, 0))


def _make_trades(trades: list[dict]) -> Panel:
    """Recent trades table."""
    tbl = Table(box=None, padding=(0, 1), expand=True)
    tbl.add_column("Time", style="dim", max_width=11)
    tbl.add_column("Action", max_width=6)
    tbl.add_column("Asset", style="bold", max_width=5)
    tbl.add_column("Side", max_width=10)
    tbl.add_column("Market", max_width=35)
    tbl.add_column("Price", justify="right", max_width=7)
    tbl.add_column("Size", justify="right", max_width=6)
    tbl.add_column("PnL", justify="right", max_width=9)
    tbl.add_column("Reason", style="dim", max_width=12)

    if not trades:
        tbl.add_row("", "", "", "", Text("No trades yet", style="dim"),
                     "", "", "", "")
    else:
        for t in reversed(trades[-12:]):
            ts = t.get("timestamp", 0)
            time_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S") if ts else ""

            action = t.get("action", "")
            action_style = "green" if action == "OPEN" else "red"

            pnl = t.get("pnl", 0)
            pnl_str = f"${pnl:+.2f}" if action == "CLOSE" else "-"
            pnl_style = "green" if pnl >= 0 else "red"

            side_str = f"{t.get('side', '')} {t.get('token_side', '')}"
            market = t.get("market", "")[:33]

            tbl.add_row(
                time_str,
                Text(action, style=action_style),
                t.get("asset", ""),
                side_str,
                market,
                f"${t.get('price', 0):.2f}",
                str(int(t.get("size", 0))),
                Text(pnl_str, style=pnl_style),
                t.get("reason", ""),
            )

    return Panel(tbl, title=f"Trade Log ({len(trades)} total)",
                 border_style="yellow", padding=(0, 0))


def _make_strategy_bar() -> Panel:
    """Strategy status bar."""
    strats = Text()
    strats.append("  spot_divergence ", style="bold cyan")
    strats.append("Spot momentum vs probability lag", style="dim")
    strats.append("  |  ", style="dim")
    strats.append("high_prob_grinder ", style="bold magenta")
    strats.append("Buy >90% events at scale", style="dim")
    strats.append("  |  ", style="dim")
    strats.append("bilateral_arb ", style="bold yellow")
    strats.append("YES+NO < $1 arbitrage", style="dim")

    return Panel(strats, title="Strategies", border_style="dim", padding=(0, 0))


def _make_risk_bar(state: dict) -> Panel:
    """Risk status footer bar."""
    daily_pnl = state.get("daily_pnl", 0)
    positions = state.get("positions", [])
    open_pos = [p for p in positions if p.get("status") == "open"]
    total_exposure = sum(p.get("entry_cost", 0) for p in open_pos)

    parts = Text()

    # Mode
    mode = EXECUTION["mode"].upper()
    mode_style = "yellow" if mode == "PAPER" else "red"
    parts.append(f"  [{mode}]  ", style=f"bold {mode_style}")

    # Risk settings
    parts.append(f"Max/trade: ${RISK['max_position_usdc']:.0f}", style="dim")
    parts.append("  |  ", style="dim")
    parts.append(f"SL: {RISK['stop_loss']*100:.0f}c  TP: {RISK['take_profit']*100:.0f}c", style="dim")
    parts.append("  |  ", style="dim")
    parts.append(f"Hold: {RISK['max_hold_minutes']}min", style="dim")
    parts.append("  |  ", style="dim")
    parts.append(f"Poll: {EXECUTION['poll_interval']}s", style="dim")

    # Kill switch
    loss_limit = RISK["daily_loss_limit_usdc"]
    if daily_pnl <= -loss_limit:
        parts.append("  |  ", style="dim")
        parts.append("KILL SWITCH ACTIVE", style="bold red blink")

    return Panel(parts, border_style="dim", padding=(0, 0))


# ============================================================
# Main dashboard loop
# ============================================================


def build_display(cycle: int, start_time: float, mode: str) -> Layout:
    """Build the full dashboard layout from current state files."""
    state = _load_state()
    trades = _load_trades(limit=50)

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=5),
        Layout(name="portfolio", size=6),
        Layout(name="strategies", size=3),
        Layout(name="middle"),
        Layout(name="footer", size=3),
    )

    # Middle: positions on top, trades on bottom
    layout["middle"].split_column(
        Layout(name="positions", ratio=1),
        Layout(name="trades", ratio=1),
    )

    layout["header"].update(_make_header(cycle, start_time, mode))
    layout["portfolio"].update(_make_portfolio(state))
    layout["strategies"].update(_make_strategy_bar())
    layout["positions"].update(_make_positions(state))
    layout["trades"].update(_make_trades(trades))
    layout["footer"].update(_make_risk_bar(state))

    return layout


def run_dashboard_standalone():
    """Run dashboard as a standalone viewer (no trading)."""
    console = Console()
    start_time = time.time()
    mode = EXECUTION["mode"]

    _stop = False

    def _handle_signal(signum, frame):
        nonlocal _stop
        _stop = True

    sys_signal.signal(sys_signal.SIGINT, _handle_signal)
    sys_signal.signal(sys_signal.SIGTERM, _handle_signal)

    cycle = 0
    try:
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while not _stop:
                cycle += 1
                display = build_display(cycle, start_time, mode)
                live.update(display)
                time.sleep(1)
    except KeyboardInterrupt:
        pass

    console.print("\n[dim]Dashboard stopped.[/dim]")


def run_dashboard_with_bot(run_cycle_fn, strategies, engine, risk_mgr,
                           assets, interval: int):
    """
    Run the dashboard with the trading loop integrated.

    The bot cycle runs every `interval` seconds, and the dashboard
    refreshes every second between cycles to show live state.
    """
    from .runner import _shutdown  # Import mutable flag

    console = Console()
    start_time = time.time()
    mode = EXECUTION["mode"]
    cycle = 0
    last_cycle_time = 0.0

    try:
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while True:
                # Check shutdown flag from runner module
                import polymarket.runner as runner_mod
                if runner_mod._shutdown:
                    break

                now = time.time()

                # Time to run a trading cycle?
                if now - last_cycle_time >= interval:
                    cycle += 1
                    try:
                        # Suppress normal print output — dashboard handles display
                        trades = run_cycle_fn(strategies, engine, risk_mgr,
                                              assets, quiet=True)
                        if trades:
                            logger.info(f"Cycle {cycle}: {trades} trade(s)")
                    except Exception as e:
                        logger.error(f"Cycle {cycle} error: {e}")
                    last_cycle_time = time.time()

                # Update display
                display = build_display(cycle, start_time, mode)
                live.update(display)
                time.sleep(1)

    except KeyboardInterrupt:
        pass

    # Clean shutdown
    console.print("\n[yellow]Shutting down...[/yellow]")
    engine.close_all(reason="shutdown")
    console.print("[green]All positions closed. Goodbye.[/green]")


# Allow running as: python -m polymarket.dashboard
if __name__ == "__main__":
    run_dashboard_standalone()
