"""
Trade Execution Engine for Polymarket.

Handles the lifecycle of trades:
  1. Receive a Signal from the signal generator
  2. Validate against risk limits
  3. Place orders on Polymarket CLOB
  4. Track open positions
  5. Manage exits (stop loss, take profit, time-based)

Supports paper trading (log only) and live execution.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import EXECUTION, RISK, DATA
from .polymarket_client import PolymarketClient, Order, OrderBook
from .signals import Signal

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """An open position on Polymarket."""
    position_id: str        # Internal tracking ID
    asset: str
    market_question: str
    condition_id: str
    token_id: str
    token_side: str         # "YES" or "NO"
    side: str               # "BUY" (long) or "SELL" (short)

    # Entry
    entry_price: float
    entry_size: float       # Shares
    entry_cost: float       # USDC spent
    entry_time: float       # Unix timestamp
    entry_order_id: str = ""

    # Current
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

    # Exit
    exit_price: float = 0.0
    exit_time: float = 0.0
    exit_reason: str = ""
    realized_pnl: float = 0.0
    exit_order_id: str = ""

    # Status
    status: str = "open"    # "open", "closing", "closed"

    # Risk levels
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    max_hold_until: float = 0.0  # Unix timestamp


@dataclass
class ExecutionState:
    """Global execution state."""
    positions: list[Position] = field(default_factory=list)
    closed_positions: list[Position] = field(default_factory=list)
    daily_pnl: float = 0.0
    daily_trades: int = 0
    total_exposure: float = 0.0
    cooldowns: dict = field(default_factory=dict)  # condition_id → cooldown_until
    last_update: float = 0.0


class ExecutionEngine:
    """
    Manages trade execution and position lifecycle.

    Usage:
        engine = ExecutionEngine(client)
        engine.execute_signal(signal)  # Opens a position
        engine.update_positions()       # Check stops/TPs
        engine.close_position(pos_id)  # Manual close
    """

    def __init__(self, client: PolymarketClient):
        self.client = client
        self.state = ExecutionState()
        self._position_counter = 0
        self._load_state()

    def _next_position_id(self) -> str:
        self._position_counter += 1
        return f"pos_{int(time.time())}_{self._position_counter}"

    # --------------------------------------------------------
    # State persistence
    # --------------------------------------------------------

    def _load_state(self):
        """Load state from disk."""
        path = Path(DATA["state_file"])
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                # Restore positions
                for p in data.get("positions", []):
                    self.state.positions.append(Position(**p))
                self.state.daily_pnl = data.get("daily_pnl", 0)
                self.state.daily_trades = data.get("daily_trades", 0)
                self.state.cooldowns = data.get("cooldowns", {})
                logger.info(f"Loaded {len(self.state.positions)} open positions from state")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")

    def _save_state(self):
        """Persist state to disk."""
        path = Path(DATA["state_file"])
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "positions": [vars(p) for p in self.state.positions],
            "daily_pnl": self.state.daily_pnl,
            "daily_trades": self.state.daily_trades,
            "cooldowns": self.state.cooldowns,
            "last_update": time.time(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _log_trade(self, position: Position, action: str):
        """Append trade to the trade log (JSONL)."""
        path = Path(DATA["trade_log"])
        path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": time.time(),
            "action": action,
            "position_id": position.position_id,
            "asset": position.asset,
            "market": position.market_question[:80],
            "token_side": position.token_side,
            "side": position.side,
            "price": position.entry_price if action == "OPEN" else position.exit_price,
            "size": position.entry_size,
            "pnl": position.realized_pnl if action == "CLOSE" else 0,
            "reason": position.exit_reason if action == "CLOSE" else "",
        }
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # --------------------------------------------------------
    # Risk checks
    # --------------------------------------------------------

    def _check_risk_limits(self, signal: Signal) -> tuple[bool, str]:
        """
        Validate a signal against risk limits before execution.

        Returns (allowed, reason).
        """
        # Daily loss limit
        if self.state.daily_pnl <= -RISK["daily_loss_limit_usdc"]:
            return False, f"Daily loss limit hit (${self.state.daily_pnl:.2f})"

        # Max open positions
        open_count = len([p for p in self.state.positions if p.status == "open"])
        if open_count >= RISK["max_open_positions"]:
            return False, f"Max open positions ({RISK['max_open_positions']})"

        # Max total exposure
        total_exposure = sum(p.entry_cost for p in self.state.positions if p.status == "open")
        new_cost = signal.suggested_price * signal.suggested_size
        if total_exposure + new_cost > RISK["max_total_exposure_usdc"]:
            return False, (f"Max exposure (${total_exposure:.2f} + ${new_cost:.2f} > "
                           f"${RISK['max_total_exposure_usdc']:.2f})")

        # Single position limit
        if new_cost > RISK["max_position_usdc"]:
            return False, f"Position too large (${new_cost:.2f} > ${RISK['max_position_usdc']:.2f})"

        # Cooldown check
        cooldown_until = self.state.cooldowns.get(signal.market.condition_id, 0)
        if time.time() < cooldown_until:
            remaining = cooldown_until - time.time()
            return False, f"Cooldown active ({remaining:.0f}s remaining)"

        # Don't double up on same market
        for pos in self.state.positions:
            if pos.condition_id == signal.market.condition_id and pos.status == "open":
                return False, f"Already have position in this market"

        return True, "OK"

    # --------------------------------------------------------
    # Execution
    # --------------------------------------------------------

    def execute_signal(self, signal: Signal) -> Optional[Position]:
        """
        Execute a signal: validate risk, place order, track position.

        Returns the Position if opened, None if rejected or failed.
        """
        # Risk check
        allowed, reason = self._check_risk_limits(signal)
        if not allowed:
            logger.info(f"Signal rejected: {reason}")
            return None

        mode = EXECUTION["mode"]
        price = signal.suggested_price
        size = signal.suggested_size

        # Apply limit offset for non-aggressive orders
        if EXECUTION["order_type"] == "limit":
            offset = EXECUTION["limit_offset"]
            if signal.side == "BUY":
                price = max(0.01, price - offset)
            else:
                price = min(0.99, price + offset)

        cost = price * size

        logger.info(f"{'[PAPER] ' if mode == 'paper' else ''}Executing: "
                     f"{signal.side} {size} {signal.token_side} @ ${price:.2f} "
                     f"(cost=${cost:.2f}) | {signal.market.question[:50]}...")

        order_id = ""

        if mode == "live":
            order = self.client.place_order(
                token_id=signal.token_id,
                side=signal.side,
                price=price,
                size=size,
            )
            if not order:
                logger.error("Order placement failed")
                return None
            order_id = order.order_id
            logger.info(f"Order placed: {order_id}")
        else:
            order_id = f"paper_{int(time.time())}"
            logger.info(f"Paper trade logged: {order_id}")

        # Create position
        now = time.time()
        position = Position(
            position_id=self._next_position_id(),
            asset=signal.asset,
            market_question=signal.market.question,
            condition_id=signal.market.condition_id,
            token_id=signal.token_id,
            token_side=signal.token_side,
            side=signal.side,
            entry_price=price,
            entry_size=size,
            entry_cost=cost,
            entry_time=now,
            entry_order_id=order_id,
            current_price=price,
            stop_loss_price=max(0.01, price - RISK["stop_loss"]) if signal.side == "BUY" else min(0.99, price + RISK["stop_loss"]),
            take_profit_price=min(0.99, price + RISK["take_profit"]) if signal.side == "BUY" else max(0.01, price - RISK["take_profit"]),
            max_hold_until=now + RISK["max_hold_minutes"] * 60,
        )

        self.state.positions.append(position)
        self.state.daily_trades += 1
        self._log_trade(position, "OPEN")
        self._save_state()

        return position

    def update_positions(self) -> list[Position]:
        """
        Update all open positions: fetch current prices, check stops/TPs.

        Returns list of positions that were closed.
        """
        closed = []

        for pos in self.state.positions:
            if pos.status != "open":
                continue

            # Fetch current price
            current_price = self.client.get_midpoint(pos.token_id)
            if current_price is None:
                book = self.client.get_order_book(pos.token_id)
                if book:
                    current_price = book.midpoint
                else:
                    continue

            pos.current_price = current_price

            # Calculate unrealized PnL
            if pos.side == "BUY":
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.entry_size
            else:
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.entry_size

            # Check stop loss
            if pos.side == "BUY" and current_price <= pos.stop_loss_price:
                self._close_position(pos, current_price, "stop_loss")
                closed.append(pos)
                continue

            if pos.side == "SELL" and current_price >= pos.stop_loss_price:
                self._close_position(pos, current_price, "stop_loss")
                closed.append(pos)
                continue

            # Check take profit
            if pos.side == "BUY" and current_price >= pos.take_profit_price:
                self._close_position(pos, current_price, "take_profit")
                closed.append(pos)
                continue

            if pos.side == "SELL" and current_price <= pos.take_profit_price:
                self._close_position(pos, current_price, "take_profit")
                closed.append(pos)
                continue

            # Check max hold time
            if time.time() >= pos.max_hold_until:
                self._close_position(pos, current_price, "max_hold_time")
                closed.append(pos)
                continue

        if closed:
            self._save_state()

        return closed

    def _close_position(self, pos: Position, exit_price: float, reason: str):
        """Close a position (internal)."""
        mode = EXECUTION["mode"]

        if pos.side == "BUY":
            pos.realized_pnl = (exit_price - pos.entry_price) * pos.entry_size
        else:
            pos.realized_pnl = (pos.entry_price - exit_price) * pos.entry_size

        pos.exit_price = exit_price
        pos.exit_time = time.time()
        pos.exit_reason = reason
        pos.status = "closed"

        self.state.daily_pnl += pos.realized_pnl

        logger.info(f"{'[PAPER] ' if mode == 'paper' else ''}Position closed: "
                     f"{pos.position_id} | reason={reason} | "
                     f"PnL=${pos.realized_pnl:+.2f}")

        # Place exit order on live
        if mode == "live":
            exit_side = "SELL" if pos.side == "BUY" else "BUY"
            order = self.client.place_order(
                token_id=pos.token_id,
                side=exit_side,
                price=exit_price,
                size=pos.entry_size,
            )
            if order:
                pos.exit_order_id = order.order_id

        # Apply cooldown on losses
        if pos.realized_pnl < 0:
            self.state.cooldowns[pos.condition_id] = (
                time.time() + RISK["loss_cooldown_seconds"]
            )

        # Move to closed list
        self.state.positions.remove(pos)
        self.state.closed_positions.append(pos)
        self._log_trade(pos, "CLOSE")

    def close_position(self, position_id: str, reason: str = "manual") -> bool:
        """Manually close a position by ID."""
        for pos in self.state.positions:
            if pos.position_id == position_id and pos.status == "open":
                current_price = self.client.get_midpoint(pos.token_id)
                if current_price is None:
                    current_price = pos.current_price or pos.entry_price
                self._close_position(pos, current_price, reason)
                self._save_state()
                return True
        return False

    def close_all(self, reason: str = "manual_close_all"):
        """Close all open positions."""
        to_close = [p for p in self.state.positions if p.status == "open"]
        for pos in to_close:
            current_price = self.client.get_midpoint(pos.token_id)
            if current_price is None:
                current_price = pos.current_price or pos.entry_price
            self._close_position(pos, current_price, reason)
        self._save_state()

    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------

    def get_summary(self) -> dict:
        """Get current execution state summary."""
        open_positions = [p for p in self.state.positions if p.status == "open"]
        total_exposure = sum(p.entry_cost for p in open_positions)
        total_unrealized = sum(p.unrealized_pnl for p in open_positions)

        return {
            "open_positions": len(open_positions),
            "total_exposure_usdc": round(total_exposure, 2),
            "unrealized_pnl": round(total_unrealized, 2),
            "daily_realized_pnl": round(self.state.daily_pnl, 2),
            "daily_trades": self.state.daily_trades,
            "positions": [
                {
                    "id": p.position_id,
                    "asset": p.asset,
                    "market": p.market_question[:60],
                    "side": f"{p.side} {p.token_side}",
                    "entry": p.entry_price,
                    "current": p.current_price,
                    "pnl": round(p.unrealized_pnl, 2),
                    "stop": p.stop_loss_price,
                    "tp": p.take_profit_price,
                }
                for p in open_positions
            ],
        }
