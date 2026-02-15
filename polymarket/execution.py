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
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .config import EXECUTION, RISK, DATA
from .polymarket_client import PolymarketClient, Order, OrderBook
from .signals import Signal

logger = logging.getLogger(__name__)

# --------------------------------------------------------
# Paper-mode slippage: walk the real order book
# --------------------------------------------------------
# Instead of guessing slippage, we fetch the actual CLOB order book
# and simulate filling the order through real price levels.
#
# Two execution modes:
#   MAKER: post a limit at our price, assume fill if price is within
#          the top 3 levels of the opposing book.  Zero slippage.
#   TAKER: walk through the book levels (market order).  Realistic slippage.
#
# MM strategy uses MAKER (that's how real MMs work — they post and wait).
# All other strategies use TAKER.


def _walk_book(levels: list[tuple[float, float]], size: int) -> tuple[float, int]:
    """Walk through order book levels to compute average fill price.

    Args:
        levels: [(price, size), ...] sorted best-first
                (ascending for asks, descending for bids)
        size: number of shares to fill

    Returns:
        (avg_fill_price, fillable_size)
        If the book can't fill the full order, fillable_size < size.
    """
    remaining = size
    total_cost = 0.0
    filled = 0

    for level_price, level_size in levels:
        if remaining <= 0:
            break
        take = min(remaining, int(level_size))
        total_cost += take * level_price
        filled += take
        remaining -= take

    if filled == 0:
        return 0.0, 0

    return total_cost / filled, filled


def _cap_to_book_depth(levels: list[tuple[float, float]]) -> int:
    """Return total depth available across all book levels."""
    return int(sum(sz for _, sz in levels))


def _estimate_round_trip_slippage(book: OrderBook, side: str, size: int) -> float:
    """Estimate total round-trip slippage cost for a taker order.

    Computes: (entry slippage) + (exit slippage) in price terms.
    Entry = walking asks (for BUY); Exit = walking bids (to sell back).
    """
    entry_levels = book.asks if side == "BUY" else book.bids
    exit_levels = book.bids if side == "BUY" else book.asks

    entry_fill, _ = _walk_book(entry_levels, size)
    exit_fill, _ = _walk_book(exit_levels, size)

    if entry_fill == 0 or exit_fill == 0:
        return 1.0  # No book = infinite slippage, skip trade

    best_entry = entry_levels[0][0] if entry_levels else entry_fill
    best_exit = exit_levels[0][0] if exit_levels else exit_fill

    entry_slip = abs(entry_fill - best_entry)
    exit_slip = abs(exit_fill - best_exit)

    return entry_slip + exit_slip


def _maker_fill_probable(book: OrderBook, side: str, price: float) -> bool:
    """Check if a passive limit order at `price` is likely to fill.

    A maker BUY fills when someone market-sells into us.
    On short-duration Polymarket markets (5m/15m), price swings of
    5-15c are normal within a single market window. So a bid that's
    a few cents below the current best_bid will still get hit.

    We allow bids up to 5c below best_bid (conservative — real swings
    are often larger).
    """
    if side == "BUY":
        return price >= (book.best_bid - 0.05) and book.spread <= 0.10
    else:
        return price <= (book.best_ask + 0.05) and book.spread <= 0.10


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

    # Strategy that opened this position
    strategy: str = ""


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
    daily_pnl_date: str = ""  # YYYY-MM-DD — tracks which day daily_pnl belongs to


def _calc_stop(signal, price: float) -> float:
    """Calculate stop loss price by strategy.

    MM: 8c stop — 5-min markets swing 5-10c within a window, so 5c
        was getting stopped out then reversing. 8c survives the noise.
        With 15c TP this is still ~2:1 reward/risk.
    Spot divergence: 10c
    High prob grinder: no stop (0.0) — hold to resolution.
        The whole thesis is that 90%+ events resolve YES ~96% of the time.
        Stopping out on noise defeats the strategy.
    Bilateral arb: no stop (handled separately in _execute_bilateral).
    Other: config default (18c)
    """
    strategy = getattr(signal, "strategy", "")
    if strategy == "market_maker":
        sl_dist = 0.08
    elif strategy == "spot_divergence":
        sl_dist = 0.10
    elif strategy in ("high_prob_grinder", "bilateral_arb"):
        return 0.0  # Hold to resolution — no stop loss
    else:
        sl_dist = RISK["stop_loss"]

    if signal.side == "BUY":
        sl = max(0.01, price - sl_dist)
        # Prevent stop == entry (would never trigger or trigger immediately)
        return min(sl, price - 0.01) if price > 0.02 else 0.01
    else:
        sl = min(0.99, price + sl_dist)
        return max(sl, price + 0.01) if price < 0.98 else 0.99


def _calc_tp(signal, price: float) -> float:
    """Calculate take profit price.

    MM: 15c TP with 8c SL = ~2:1 reward/risk.
    With maker fills (no slippage on wins), we keep the full TP.
    Need only ~35% win rate to break even.
    High prob grinder: no TP — hold to resolution for full $1.00 payout.
    Bilateral arb: no TP — hold to resolution (guaranteed payout).
    """
    strategy = getattr(signal, "strategy", "")
    if strategy in ("high_prob_grinder", "bilateral_arb"):
        return 0.0  # Hold to resolution
    if strategy == "market_maker":
        tp_dist = 0.15
    else:
        tp_dist = RISK["take_profit"]

    if signal.side == "BUY":
        return min(0.99, price + tp_dist)
    return max(0.01, price - tp_dist)


def _et_to_utc_offset_hours() -> int:
    """Return the current ET→UTC offset: 5 (EST, Nov-Mar) or 4 (EDT, Mar-Nov)."""
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime as dt
        et_now = dt.now(ZoneInfo("America/New_York"))
        return -et_now.utcoffset().total_seconds() // 3600
    except Exception:
        # Fallback: assume EST (correct Nov-Mar)
        return 5


def _market_end_timestamp(question: str, end_date: str = "") -> Optional[float]:
    """Parse the market close time.

    Tries two methods:
    1. BTC Up/Down question format: "February 14, 1:30PM-1:45PM ET" → timestamp for 1:45PM ET
    2. Gamma API end_date (ISO format): "2026-02-20T00:00:00Z" → timestamp

    Returns Unix timestamp of the market end, or None if unparseable.
    """
    # Method 1: parse from question text (BTC Up/Down markets)
    m = re.search(
        r'(\w+ \d+),?\s*\d{1,2}:\d{2}(?:am|pm)\s*-\s*(\d{1,2}):(\d{2})(am|pm)\s*et',
        question.lower(),
    )
    if m:
        date_str = m.group(1)  # "february 14"
        h, mi, p = int(m.group(2)), int(m.group(3)), m.group(4)

        if p == 'pm' and h != 12:
            h += 12
        if p == 'am' and h == 12:
            h = 0

        now = datetime.now(timezone.utc)
        try:
            end_dt = datetime.strptime(
                f"{date_str} {now.year} {h:02d}:{mi:02d}",
                "%B %d %Y %H:%M",
            ).replace(tzinfo=timezone.utc)
            end_dt = end_dt + timedelta(hours=_et_to_utc_offset_hours())
            ts = end_dt.timestamp()
            if ts > time.time():
                return ts
        except ValueError:
            pass

    # Method 2: parse end_date from Gamma API (grinder, bilateral, etc.)
    if end_date:
        for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ",
                    "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]:
            try:
                end_dt = datetime.strptime(end_date, fmt).replace(tzinfo=timezone.utc)
                ts = end_dt.timestamp()
                if ts > time.time():
                    return ts
            except ValueError:
                continue

    return None


class ExecutionEngine:
    """
    Manages trade execution and position lifecycle.

    Usage:
        engine = ExecutionEngine(client)
        engine.execute_signal(signal)  # Opens a position
        engine.update_positions()       # Check stops/TPs
        engine.close_position(pos_id)  # Manual close
    """

    def __init__(self, client: PolymarketClient, bankroll_mgr=None):
        self.client = client
        self.state = ExecutionState()
        self.bankroll = bankroll_mgr  # Optional BankrollManager for Kelly reinvestment
        self._position_counter = 0
        self._mm_strategy = None  # Set by runner to wire up inventory tracking
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
                self.state.daily_pnl_date = data.get("daily_pnl_date", "")
                self.state.cooldowns = data.get("cooldowns", {})
                logger.info(f"Loaded {len(self.state.positions)} open positions from state")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")

        # Reset daily counters if the date has changed (midnight rollover).
        # Also reconcile with trade log on load to fix any drift.
        self._check_daily_reset()
        self._reconcile_daily_from_trade_log()

    def _check_daily_reset(self):
        """Reset daily counters if the calendar date (UTC) has changed."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.daily_pnl_date and self.state.daily_pnl_date != today:
            logger.info(f"[DAILY RESET] New day {today} (was {self.state.daily_pnl_date}). "
                        f"Resetting daily_pnl=${self.state.daily_pnl:+.2f}, "
                        f"daily_trades={self.state.daily_trades}")
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
        self.state.daily_pnl_date = today

    def _reconcile_daily_from_trade_log(self):
        """Recompute today's daily_pnl from the trade log (source of truth).

        Fixes drift caused by state resets, crashes, or missed updates.
        Only called on startup — not every save cycle.
        """
        path = Path(DATA["trade_log"])
        if not path.exists():
            return

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_pnl = 0.0
        today_trades = 0

        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("action") != "CLOSE":
                        continue
                    ts = entry.get("timestamp", 0)
                    trade_date = datetime.fromtimestamp(
                        ts, tz=timezone.utc
                    ).strftime("%Y-%m-%d")
                    if trade_date == today:
                        today_pnl += entry.get("pnl", 0)
                        today_trades += 1
        except Exception as e:
            logger.warning(f"[RECONCILE] Could not read trade log: {e}")
            return

        if abs(today_pnl - self.state.daily_pnl) > 1.0:
            logger.info(f"[RECONCILE] Fixing daily_pnl: "
                        f"state=${self.state.daily_pnl:.2f} → "
                        f"trade_log=${today_pnl:.2f} "
                        f"(diff=${today_pnl - self.state.daily_pnl:+.2f})")
        self.state.daily_pnl = today_pnl
        self.state.daily_trades = today_trades
        self.state.daily_pnl_date = today

    def _save_state(self):
        """Persist state to disk atomically (write to temp, then rename)."""
        path = Path(DATA["state_file"])
        path.parent.mkdir(parents=True, exist_ok=True)

        # Check for midnight rollover on every save
        self._check_daily_reset()

        # Garbage-collect expired cooldowns
        now = time.time()
        self.state.cooldowns = {
            k: v for k, v in self.state.cooldowns.items() if v > now
        }

        data = {
            "positions": [vars(p) for p in self.state.positions],
            "daily_pnl": self.state.daily_pnl,
            "daily_trades": self.state.daily_trades,
            "daily_pnl_date": self.state.daily_pnl_date,
            "cooldowns": self.state.cooldowns,
            "last_update": now,
        }
        # Atomic write: temp file + rename prevents corruption on crash
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

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
            "strategy": position.strategy,
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
        # Get Kelly-scaled limits if bankroll manager is available
        if self.bankroll:
            limits = self.bankroll.get_effective_limits()
        else:
            limits = {
                "max_total_exposure_usdc": RISK["max_total_exposure_usdc"],
                "max_position_usdc": RISK["max_position_usdc"],
                "daily_loss_limit_usdc": RISK["daily_loss_limit_usdc"],
            }

        # Daily loss limit (Kelly-scaled)
        if self.state.daily_pnl <= -limits["daily_loss_limit_usdc"]:
            return False, f"Daily loss limit hit (${self.state.daily_pnl:.2f})"

        # Max open positions
        open_count = len([p for p in self.state.positions if p.status == "open"])
        if open_count >= RISK["max_open_positions"]:
            return False, f"Max open positions ({RISK['max_open_positions']})"

        # Max total exposure (Kelly-scaled)
        total_exposure = sum(p.entry_cost for p in self.state.positions if p.status == "open")
        new_cost = signal.suggested_price * signal.suggested_size
        max_exposure = limits["max_total_exposure_usdc"]
        if total_exposure + new_cost > max_exposure:
            return False, (f"Max exposure (${total_exposure:.2f} + ${new_cost:.2f} > "
                           f"${max_exposure:.2f})")

        # Single position limit (Kelly-scaled)
        max_pos = limits["max_position_usdc"]
        if new_cost > max_pos:
            return False, f"Position too large (${new_cost:.2f} > ${max_pos:.2f})"

        # Cooldown check
        cooldown_until = self.state.cooldowns.get(signal.market.condition_id, 0)
        if time.time() < cooldown_until:
            remaining = cooldown_until - time.time()
            return False, f"Cooldown active ({remaining:.0f}s remaining)"

        # Don't double up on same market
        # MM can hold both sides (Up + Down) of the same market, and can
        # stack up to 3 positions per side at different price levels.
        # Balance rule: one side can't exceed 2x the other in $ terms.
        is_mm = getattr(signal, "strategy", "") == "market_maker"
        has_same_market = any(
            p.condition_id == signal.market.condition_id and p.status == "open"
            for p in self.state.positions
        )

        if has_same_market:
            if is_mm:
                mm_positions = [
                    p for p in self.state.positions
                    if p.condition_id == signal.market.condition_id
                    and p.status == "open"
                    and p.strategy == "market_maker"
                ]
                same_side = [p for p in mm_positions
                             if p.token_side == signal.token_side]

                # Max 3 layers per side (was 5 — too many stacked losses)
                if len(same_side) >= 3:
                    return False, f"MM: max 3 layers on {signal.token_side} in this market"

                # Don't stack at the exact same price level (±1c)
                if any(abs(p.entry_price - signal.suggested_price) < 0.01
                       for p in same_side):
                    return False, f"MM: already have {signal.token_side} at this price"

                # Balance check: prevent directional overload.
                # If we already hold one side but not the other, block adding
                # more to the existing side until the other side catches up.
                up_positions = [p for p in mm_positions if p.token_side == "Up"]
                down_positions = [p for p in mm_positions if p.token_side == "Down"]
                up_cost = sum(p.entry_cost for p in up_positions)
                down_cost = sum(p.entry_cost for p in down_positions)
                new_cost = signal.suggested_price * signal.suggested_size

                if signal.token_side == "Up":
                    # Block adding more Up if Down side is empty but Up isn't
                    if len(up_positions) > 0 and len(down_positions) == 0:
                        return False, "MM: open Down side before adding more Up"
                    # Enforce max 2x dollar imbalance
                    if down_cost > 0 and (up_cost + new_cost) > down_cost * 2.0:
                        return False, (f"MM: Up ${up_cost + new_cost:.0f} would exceed "
                                       f"2x Down ${down_cost:.0f}")
                elif signal.token_side == "Down":
                    if len(down_positions) > 0 and len(up_positions) == 0:
                        return False, "MM: open Up side before adding more Down"
                    if up_cost > 0 and (down_cost + new_cost) > up_cost * 2.0:
                        return False, (f"MM: Down ${down_cost + new_cost:.0f} would exceed "
                                       f"2x Up ${up_cost:.0f}")
            else:
                return False, f"Already have position in this market"

        return True, "OK"

    # --------------------------------------------------------
    # Execution
    # --------------------------------------------------------

    def execute_signal(self, signal: Signal) -> Optional[Position]:
        """
        Execute a signal: validate risk, place order, track position.

        Handles both single-leg signals (spot divergence, grinder)
        and multi-leg bilateral arb signals.

        Returns the Position if opened, None if rejected or failed.
        """
        # Risk check
        allowed, reason = self._check_risk_limits(signal)
        if not allowed:
            logger.info(f"Signal rejected: {reason}")
            return None

        # Dispatch to bilateral handler if multi-leg
        if signal.is_bilateral:
            return self._execute_bilateral(signal)

        return self._execute_single_leg(signal)

    def _execute_single_leg(self, signal: Signal) -> Optional[Position]:
        """Execute a standard single-leg signal."""
        mode = EXECUTION["mode"]
        price = signal.suggested_price
        size = signal.suggested_size
        strategy = getattr(signal, "strategy", "")
        is_maker = strategy == "market_maker"

        # Apply limit offset for non-MM orders.
        # MM strategy already computes the exact bid price — applying
        # offset again would double-subtract and reduce edge.
        if EXECUTION["order_type"] == "limit" and not is_maker:
            offset = EXECUTION["limit_offset"]
            if signal.side == "BUY":
                price = max(0.01, price - offset)
            else:
                price = min(0.99, price + offset)

        # --- Paper-mode execution ---
        slip_info = ""
        if mode == "paper":
            book = self.client.get_order_book(signal.token_id)
            if not book:
                logger.warning(f"[PAPER] No book for {signal.token_id[:20]}... — skipping")
                return None

            if is_maker:
                # MAKER mode: post a passive limit at our price.
                # MM strategy already computes the right bid price.
                # No slippage — we get filled AT our price or not at all.
                if not _maker_fill_probable(book, signal.side, price):
                    logger.info(f"[PAPER] Maker fill unlikely: spread={book.spread:.3f} "
                                f"price={price:.2f} best_bid={book.best_bid:.2f} "
                                f"best_ask={book.best_ask:.2f} "
                                f"| {signal.market.question[:40]}")
                    return None
                # Hard cap: never risk more than $250 on a single maker order
                # regardless of Kelly/boost scaling
                max_maker_cost = RISK["max_position_usdc"]
                max_maker_shares = max(1, int(max_maker_cost / price))
                size = min(size, max_maker_shares)
                slip_info = f"MAKER, spread={book.spread:.3f}, no_slippage"
            else:
                # TAKER mode: walk the book for realistic fills
                # But first: check if the round-trip slippage exceeds the edge
                rt_slip = _estimate_round_trip_slippage(book, signal.side, size)
                if rt_slip > signal.edge * 0.7:
                    logger.info(f"[PAPER] SKIP: rt_slippage={rt_slip:.3f} > "
                                f"70% of edge={signal.edge:.3f} "
                                f"| {signal.market.question[:40]}")
                    return None

                levels = book.asks if signal.side == "BUY" else book.bids
                total_depth = _cap_to_book_depth(levels)
                size = min(size, max(1, total_depth))

                avg_fill, filled = _walk_book(levels, size)
                if filled > 0 and avg_fill > 0:
                    slip = abs(avg_fill - price)
                    price = round(avg_fill, 4)
                    size = filled
                    slip_info = (f"TAKER, slippage={slip:.3f}, "
                                 f"rt_slip={rt_slip:.3f}, "
                                 f"book_depth={total_depth}")
                else:
                    logger.warning(f"[PAPER] Book empty for {signal.token_side} "
                                   f"| {signal.market.question[:40]} — skipping")
                    return None

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
            logger.info(f"Paper trade logged ({slip_info}): {order_id}")

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
            stop_loss_price=_calc_stop(signal, price),
            take_profit_price=_calc_tp(signal, price),
            max_hold_until=_market_end_timestamp(signal.market.question, getattr(signal.market, "end_date", "")) or (now + RISK["max_hold_minutes"] * 60),
            strategy=getattr(signal, "strategy", ""),
        )

        self.state.positions.append(position)
        self.state.daily_trades += 1

        # Update MM inventory tracking on open
        if position.strategy == "market_maker" and self._mm_strategy is not None:
            try:
                self._mm_strategy.update_inventory(
                    position.condition_id, position.token_side, position.entry_size
                )
            except Exception:
                pass

        self._log_trade(position, "OPEN")
        self._save_state()

        return position

    def _execute_bilateral(self, signal: Signal) -> Optional[Position]:
        """
        Execute a bilateral arb: buy all legs simultaneously.

        For a YES+NO arb, we place two orders (buy YES, buy NO).
        For a multi-outcome arb, we place N orders (one per outcome).
        All legs must fill for the arb to be complete.
        """
        mode = EXECUTION["mode"]
        legs = signal.arb_legs
        size = signal.suggested_size

        # Paper mode: cap size to smallest book depth across all legs
        if mode == "paper":
            for leg in legs:
                book = self.client.get_order_book(leg["token_id"])
                if book and book.asks:
                    leg_depth = _cap_to_book_depth(book.asks)
                    size = min(size, max(1, leg_depth))
                    leg["_book"] = book  # stash for per-leg fill calc
                else:
                    logger.warning(f"[PAPER] No book for bilateral leg — skipping arb")
                    return None

        total_cost = sum(leg["price"] * size for leg in legs)
        guaranteed_payout = size  # One leg resolves to $1.00 * size

        logger.info(f"{'[PAPER] ' if mode == 'paper' else ''}"
                     f"Executing BILATERAL ({len(legs)} legs): "
                     f"{size} shares each | total cost=${total_cost:.2f} → "
                     f"guaranteed=${guaranteed_payout:.2f} | "
                     f"profit=${guaranteed_payout - total_cost:.2f}")

        order_ids = []
        failed = False

        for i, leg in enumerate(legs):
            leg_label = leg.get("label", f"Leg {i}")
            leg_price = leg["price"]
            leg_token = leg["token_id"]

            logger.info(f"  Leg {i+1}/{len(legs)}: BUY {size} {leg_label} "
                         f"@ ${leg_price:.2f}")

            # Paper mode: walk real book for this leg
            if mode == "paper":
                leg_book = leg.get("_book")
                if leg_book and leg_book.asks:
                    avg_fill, filled = _walk_book(leg_book.asks, size)
                    if filled > 0:
                        leg_price = round(avg_fill, 4)
                        size = min(size, filled)

            if mode == "live":
                order = self.client.place_order(
                    token_id=leg_token,
                    side="BUY",
                    price=leg_price,
                    size=size,
                )
                if not order:
                    logger.error(f"  Leg {i+1} FAILED — cancelling previous legs")
                    # Cancel all previously placed legs to avoid naked exposure
                    for prev_id in order_ids:
                        if not prev_id.startswith("FAILED"):
                            self.client.cancel_order(prev_id)
                            logger.info(f"  Cancelled leg: {prev_id}")
                    failed = True
                    break
                else:
                    order_ids.append(order.order_id)
            else:
                order_ids.append(f"paper_leg{i}_{int(time.time())}")

        if failed:
            logger.error("Bilateral arb aborted — all legs cancelled")
            return None

        # Track as a single combined position
        now = time.time()
        position = Position(
            position_id=self._next_position_id(),
            asset=signal.asset,
            market_question=signal.market.question,
            condition_id=signal.market.condition_id,
            token_id=signal.token_id,  # Primary leg token
            token_side=signal.token_side,
            side="BUY",
            entry_price=round(sum(l["price"] for l in legs), 4),
            entry_size=size,
            entry_cost=total_cost,
            entry_time=now,
            entry_order_id="|".join(order_ids),
            current_price=signal.suggested_price,
            # Bilateral arbs don't use stop/TP — they're held to resolution
            stop_loss_price=0.0,
            take_profit_price=0.0,
            max_hold_until=0,  # Hold until market resolves
            strategy=getattr(signal, "strategy", ""),
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

        for pos in list(self.state.positions):
            if pos.status != "open":
                continue

            # Fetch current price
            current_price = self.client.get_midpoint(pos.token_id)
            if current_price is None:
                book = self.client.get_order_book(pos.token_id)
                if book:
                    current_price = book.midpoint
                else:
                    # Both midpoint and orderbook unavailable — market has
                    # likely expired/resolved and the CLOB removed the book.
                    # Auto-close at last known price to free up capacity.
                    exit_price = pos.current_price if pos.current_price > 0 else pos.entry_price
                    logger.info(
                        f"Market gone (no orderbook) for {pos.position_id} "
                        f"({pos.market_question[:50]}) — auto-closing at "
                        f"${exit_price:.3f}"
                    )
                    self._close_position(pos, exit_price, "market_expired")
                    closed.append(pos)
                    continue

            pos.current_price = current_price

            # Calculate unrealized PnL
            if pos.side == "BUY":
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.entry_size
            else:
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.entry_size

            # Hold-to-resolution strategies (grinder, bilateral arb) have
            # stop=0 and tp=0. They resolve when the market closes.
            # Skip stop/TP/trailing logic for them — only time-based exit applies.
            has_stop = pos.stop_loss_price > 0
            has_tp = pos.take_profit_price > 0

            if not has_stop and not has_tp and pos.max_hold_until == 0:
                # Bilateral arb: no stop, no TP, no hold time → pure hold-to-resolution
                continue

            # Trailing stop: once in profit, move stop to breakeven.
            # Only applies to strategies with active stops (MM, spot_div).
            if has_stop:
                strategy = getattr(pos, "strategy", "")
                if strategy == "market_maker":
                    trail_threshold = 0.08   # 8c stop → trail at 8c profit
                elif strategy == "spot_divergence":
                    trail_threshold = 0.05   # 10c stop → trail at 5c
                else:
                    trail_threshold = 0.08   # 18c stop → trail at 8c

                if pos.side == "BUY":
                    profit = current_price - pos.entry_price
                    if profit >= trail_threshold:
                        new_sl = pos.entry_price + 0.01
                        if new_sl > pos.stop_loss_price:
                            pos.stop_loss_price = new_sl
                elif pos.side == "SELL":
                    profit = pos.entry_price - current_price
                    if profit >= trail_threshold:
                        new_sl = pos.entry_price - 0.01
                        if new_sl < pos.stop_loss_price:
                            pos.stop_loss_price = new_sl

                # Check stop loss
                if pos.side == "BUY" and current_price <= pos.stop_loss_price:
                    self._close_position(pos, current_price, "stop_loss")
                    closed.append(pos)
                    continue

                if pos.side == "SELL" and current_price >= pos.stop_loss_price:
                    self._close_position(pos, current_price, "stop_loss")
                    closed.append(pos)
                    continue

            # Check take profit (only if TP is set)
            if has_tp:
                if pos.side == "BUY" and current_price >= pos.take_profit_price:
                    self._close_position(pos, current_price, "take_profit")
                    closed.append(pos)
                    continue

                if pos.side == "SELL" and current_price <= pos.take_profit_price:
                    self._close_position(pos, current_price, "take_profit")
                    closed.append(pos)
                    continue

            # Near-expiry handling — strategy-dependent:
            # MM/spot_div: exit before resolution to avoid gap risk.
            #   5-min markets gap 30-40c at resolution (token → 0 or 1).
            # Grinder/bilateral: HOLD through resolution — resolution IS the profit.
            #   When the book disappears, the "market_expired" handler above closes
            #   at last known price (paper) or Polymarket pays out (live).
            if pos.max_hold_until > 0:
                time_left = pos.max_hold_until - time.time()
                is_hold_strat = pos.strategy in ("high_prob_grinder", "bilateral_arb")

                if not is_hold_strat:
                    # MM/spot_div: exit early to avoid gap risk
                    if time_left <= 120 and pos.unrealized_pnl > 0:
                        self._close_position(pos, current_price, "near_expiry_profit")
                        closed.append(pos)
                        continue
                    if time_left <= 30:
                        reason = "near_expiry_profit" if pos.unrealized_pnl > 0 else "near_expiry_exit"
                        self._close_position(pos, current_price, reason)
                        closed.append(pos)
                        continue

                # Max hold time (fallback for strategies using config default)
                if time.time() >= pos.max_hold_until and not is_hold_strat:
                    self._close_position(pos, current_price, "max_hold_time")
                    closed.append(pos)
                    continue

        if closed:
            self._save_state()

        return closed

    def _close_position(self, pos: Position, exit_price: float, reason: str):
        """Close a position (internal)."""
        # Idempotency guard: prevent double-counting PnL if called twice
        if pos.status == "closed":
            return

        mode = EXECUTION["mode"]

        # Paper mode: simulate exit fill.
        # Three exit modes:
        #   1. MM take-profit / near-expiry: MAKER (zero slippage, post limit)
        #   2. MM stop-loss / max-hold: HALF-SPREAD penalty (aggressive limit,
        #      fills quickly on active 5m markets — not a full book walk)
        #   3. Non-MM any exit: TAKER (walk the book, full slippage)
        is_mm = pos.strategy == "market_maker"
        is_urgent = reason in ("stop_loss", "max_hold_time", "market_expired", "near_expiry_exit")
        if mode == "paper":
            if is_mm and is_urgent:
                # Aggressive limit exit: apply half the spread as penalty
                book = self.client.get_order_book(pos.token_id)
                if book:
                    penalty = book.spread * 0.5
                    if pos.side == "BUY":
                        exit_price = round(exit_price - penalty, 4)
                    else:
                        exit_price = round(exit_price + penalty, 4)
            elif not is_mm:
                # Taker exit: walk the book (full slippage)
                book = self.client.get_order_book(pos.token_id)
                if book:
                    levels = book.bids if pos.side == "BUY" else book.asks
                    if levels:
                        avg_fill, filled = _walk_book(levels, int(pos.entry_size))
                        if filled > 0:
                            exit_price = round(avg_fill, 4)
            # else: MM non-urgent → maker exit, no slippage

        if pos.side == "BUY":
            pos.realized_pnl = (exit_price - pos.entry_price) * pos.entry_size
        else:
            pos.realized_pnl = (pos.entry_price - exit_price) * pos.entry_size

        pos.exit_price = exit_price
        pos.exit_time = time.time()
        pos.exit_reason = reason

        # Place exit order on live BEFORE marking closed — if order fails,
        # position stays "open" so we can retry next cycle
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
            else:
                logger.error(f"Exit order FAILED for {pos.position_id} — "
                             f"will retry next cycle")
                pos.exit_price = 0.0
                pos.exit_time = 0.0
                pos.exit_reason = ""
                return

        pos.status = "closed"
        pos.unrealized_pnl = 0.0

        self.state.daily_pnl += pos.realized_pnl

        # Record trade in bankroll for Kelly reinvestment
        if self.bankroll:
            self.bankroll.record_trade(pos.realized_pnl)

        logger.info(f"{'[PAPER] ' if mode == 'paper' else ''}Position closed: "
                     f"{pos.position_id} | reason={reason} | "
                     f"PnL=${pos.realized_pnl:+.2f}")

        # Apply cooldown on losses
        if pos.realized_pnl < 0:
            self.state.cooldowns[pos.condition_id] = (
                time.time() + RISK["loss_cooldown_seconds"]
            )

        # Update MM inventory tracking (negative shares = sold/closed)
        if pos.strategy == "market_maker" and self._mm_strategy is not None:
            try:
                self._mm_strategy.update_inventory(
                    pos.condition_id, pos.token_side, -pos.entry_size
                )
            except Exception:
                pass  # Don't let inventory tracking break closes

        # Move to closed list
        if pos in self.state.positions:
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
        to_close = list(self.state.positions)
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
