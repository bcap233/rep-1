"""
Strategy 4: Market Maker for BTC Up/Down Markets.

The concept:
  Post bids and asks on both the "Up" and "Down" sides of BTC short-duration
  markets. Profit comes from the spread, not from predicting direction.

  Example on a 15m BTC Up/Down market:
    Fair value of "Up" = 0.50 (neutral momentum)
    Post bid at 0.48, ask at 0.52 on "Up"
    Post bid at 0.48, ask at 0.52 on "Down"
    If both sides fill → paid 0.96, guaranteed payout $1.00 → profit $0.04

  When we accumulate inventory (e.g., long 50 shares of "Up"):
    Skew quotes: lower "Up" ask to offload, raise "Up" bid to slow buying
    This keeps inventory near neutral while still capturing spread

  Key edge vs directional betting:
    - Don't need to predict direction
    - Profit on EVERY round-trip, not just when we're right
    - Works best in range-bound / choppy markets (majority of the time)
    - Momentum bias lets us lean into trends when they appear

Filters:
  - Only 15m and 4h markets (5m too fast to manage inventory)
  - Min time to expiry (don't quote dying markets)
  - Inventory limits (cap net exposure)
  - Spread requirements (don't fight 1c spread markets)
"""

import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

from . import BaseStrategy, register_strategy
from ..config import MARKET_MAKER, STRATEGY
from ..exchanges import get_composite_price
from ..charts import analyze_all_timeframes, multi_timeframe_consensus
from ..exchanges import get_best_candles
from ..polymarket_client import PolymarketClient, Market, OrderBook
from ..signals import Signal, estimate_updown_fair_probability

logger = logging.getLogger(__name__)


def _parse_market_duration(question: str) -> Optional[str]:
    """Extract market duration tag from question text.

    Returns '5m', '15m', '4h', or None.
    """
    q = question.lower()
    # Match time spans like "12:00AM-12:05AM" (5m), "12:00AM-12:15AM" (15m),
    # "12:00AM-4:00AM" (4h)
    m = re.search(r'(\d{1,2}):(\d{2})(am|pm)-(\d{1,2}):(\d{2})(am|pm)', q)
    if not m:
        return None
    h1, m1, p1 = int(m.group(1)), int(m.group(2)), m.group(3)
    h2, m2, p2 = int(m.group(4)), int(m.group(5)), m.group(6)
    # Convert to 24h
    if p1 == 'pm' and h1 != 12:
        h1 += 12
    if p1 == 'am' and h1 == 12:
        h1 = 0
    if p2 == 'pm' and h2 != 12:
        h2 += 12
    if p2 == 'am' and h2 == 12:
        h2 = 0
    diff_min = (h2 * 60 + m2) - (h1 * 60 + m1)
    if diff_min < 0:
        diff_min += 24 * 60  # crosses midnight
    if diff_min <= 5:
        return "5m"
    elif diff_min <= 15:
        return "15m"
    elif diff_min <= 240:
        return "4h"
    return None


def _minutes_to_market_end(question: str) -> Optional[float]:
    """Estimate minutes until this market's window closes.

    Parses the end time from the question text and compares to now (UTC).
    Returns None if unparseable.
    """
    # Extract date and end time: "February 14, 1:30AM-1:45AM ET"
    m = re.search(
        r'(\w+ \d+),?\s*\d{1,2}:\d{2}(?:am|pm)\s*-\s*(\d{1,2}):(\d{2})(am|pm)\s*et',
        question.lower(),
    )
    if not m:
        return None

    date_str = m.group(1)  # "february 14"
    h, mi, p = int(m.group(2)), int(m.group(3)), m.group(4)

    if p == 'pm' and h != 12:
        h += 12
    if p == 'am' and h == 12:
        h = 0

    now = datetime.now(timezone.utc)
    # ET is UTC-5
    try:
        end_dt = datetime.strptime(
            f"{date_str} {now.year} {h:02d}:{mi:02d}",
            "%B %d %Y %H:%M",
        ).replace(tzinfo=timezone.utc)
        # Adjust for ET → UTC (+5 hours)
        from datetime import timedelta
        end_dt = end_dt + timedelta(hours=5)
    except ValueError:
        return None

    delta = (end_dt - now).total_seconds() / 60.0
    return delta if delta > 0 else None


@register_strategy
class MarketMaker(BaseStrategy):
    """
    Market-making strategy for BTC Up/Down short-duration markets.

    Posts quotes on both sides, profits from the spread.
    Uses spot momentum to bias fair value and skews quotes
    to manage inventory.

    Usage:
        python -m polymarket --strategy market_maker --loop
    """

    name = "market_maker"
    description = "Two-sided quotes on BTC Up/Down markets, profit from spread"

    def __init__(self, client: PolymarketClient):
        super().__init__(client)
        self.cfg = MARKET_MAKER
        # Track net inventory per market (condition_id → net_up_shares)
        # Positive = long Up, Negative = long Down
        self._inventory: dict[str, float] = {}

    def scan(self, assets: Optional[list[str]] = None) -> list[Signal]:
        """
        Scan BTC Up/Down markets and generate two-sided quotes.

        For each eligible market:
        1. Get order books for Up and Down
        2. Compute fair value (50/50 base + momentum bias)
        3. Apply inventory skew
        4. Generate BUY signals for the underpriced side
        """
        signals = []

        # Get spot momentum for fair value bias
        momentum_score = 0.0
        composite = get_composite_price("BTC")
        if composite:
            candles_by_tf = {}
            for tf in [5, 15]:
                candles = get_best_candles("BTC", tf, limit=50)
                if candles:
                    candles_by_tf[tf] = candles
            if candles_by_tf:
                analyses = analyze_all_timeframes("BTC", candles_by_tf)
                if analyses:
                    momentum_score, _ = multi_timeframe_consensus(analyses)

        # Find BTC Up/Down markets
        markets = self.client.find_btc_updown_markets()
        if not markets:
            logger.info("[MM] No BTC Up/Down markets found")
            return []

        eligible = 0
        for market in markets:
            q = market.question

            # Filter by duration
            duration = _parse_market_duration(q)
            if duration not in self.cfg["durations"]:
                continue

            # Check time to expiry
            mins_left = _minutes_to_market_end(q)
            if mins_left is not None and mins_left < self.cfg["min_minutes_to_expiry"]:
                continue

            # Need both token IDs
            if len(market.token_ids) < 2:
                continue

            up_token = market.token_ids[0]
            down_token = market.token_ids[1]

            # Get order books
            up_book = self.client.get_order_book(up_token)
            down_book = self.client.get_order_book(down_token)
            if not up_book or not down_book:
                continue

            # Check if spread is wide enough to make money
            up_spread = up_book.spread
            down_spread = down_book.spread
            if up_spread < self.cfg["min_book_spread"] and down_spread < self.cfg["min_book_spread"]:
                continue  # Market too tight, can't compete

            eligible += 1

            # Fair value: anchor on book midpoint + momentum bias
            book_mid_up = up_book.midpoint
            bias = momentum_score * self.cfg["momentum_bias_weight"]
            fair_up = max(0.05, min(0.95, book_mid_up + bias))
            fair_down = 1.0 - fair_up

            # Inventory skew
            inv = self._inventory.get(market.condition_id, 0.0)
            max_inv = self.cfg["max_inventory_shares"]
            skew = 0.0
            if max_inv > 0 and abs(inv) > 0:
                skew = (inv / max_inv) * self.cfg["inventory_skew_max"]

            half = self.cfg["half_spread"]
            size_usdc = self.cfg["size_per_side_usdc"]

            # ---- Market-making approach ----
            # We post passive limit bids that improve the current best bid
            # by 1 tick (0.01). This puts us at the front of the queue.
            # When someone market-sells into us, we buy cheap. Then we
            # post a sell (ask) at the midpoint or above to close.
            #
            # The edge per round-trip = spread we capture.
            # We generate BUY signals at our bid price — the execution
            # engine places them as limit orders.

            # Up side: post bid at best_bid + 0.01 (improve the book)
            our_up_bid = min(up_book.best_bid + 0.01, fair_up - half - skew)
            our_up_bid = max(0.01, min(0.99, round(our_up_bid, 2)))
            up_expected_profit = fair_up - our_up_bid  # profit if we buy here and it reverts to fair
            if up_expected_profit > 0.01 and inv < max_inv:
                up_size = max(1, int(size_usdc / our_up_bid))
                sig = self._make_signal(
                    market, up_token, "Up", our_up_bid, up_size,
                    up_expected_profit, fair_up, up_book.midpoint,
                    momentum_score, duration,
                )
                signals.append(sig)

            # Down side: post bid at best_bid + 0.01
            our_down_bid = min(down_book.best_bid + 0.01, fair_down - half + skew)
            our_down_bid = max(0.01, min(0.99, round(our_down_bid, 2)))
            down_expected_profit = fair_down - our_down_bid
            if down_expected_profit > 0.01 and -inv < max_inv:
                down_size = max(1, int(size_usdc / our_down_bid))
                sig = self._make_signal(
                    market, down_token, "Down", our_down_bid, down_size,
                    down_expected_profit, fair_down, down_book.midpoint,
                    momentum_score, duration,
                )
                signals.append(sig)

            # ---- Bilateral arb: buy BOTH sides for < $1.00 ----
            total_cost = up_book.best_ask + down_book.best_ask
            if total_cost < 0.98:
                arb_edge = 1.0 - total_cost
                cheaper_side = "Up" if up_book.best_ask <= down_book.best_ask else "Down"
                cheaper_token = up_token if cheaper_side == "Up" else down_token
                cheaper_price = up_book.best_ask if cheaper_side == "Up" else down_book.best_ask
                cheaper_size = max(1, int(size_usdc / cheaper_price))

                sig = Signal(
                    asset="BTC",
                    market=market,
                    token_id=cheaper_token,
                    side="BUY",
                    token_side=f"{cheaper_side}+arb",
                    spot_price=composite.price if composite else 0.0,
                    target_price=0.0,
                    implied_prob=cheaper_price,
                    fair_prob=0.50,
                    edge=arb_edge,
                    confidence=0.95,
                    momentum_score=momentum_score,
                    momentum_direction="n/a",
                    confirming_timeframes=0,
                    suggested_price=round(cheaper_price, 2),
                    suggested_size=cheaper_size,
                    timestamp=time.time(),
                    reason=(
                        f"MM-ARB: Up@{up_book.best_ask:.2f}+Down@{down_book.best_ask:.2f}"
                        f"={total_cost:.2f} < $1.00 | "
                        f"profit={arb_edge:.3f}/share"
                    ),
                    strategy="market_maker",
                    arb_legs=[
                        {"token_id": up_token, "side": "BUY",
                         "price": up_book.best_ask, "label": "Up"},
                        {"token_id": down_token, "side": "BUY",
                         "price": down_book.best_ask, "label": "Down"},
                    ],
                )
                signals.append(sig)
                logger.info(f"  MM-ARB: {q[:50]} | "
                            f"Up@{up_book.best_ask:.2f}+Down@{down_book.best_ask:.2f}"
                            f"={total_cost:.2f} → profit={arb_edge:.3f}")

        logger.info(f"[MM] Scanned {len(markets)} markets, "
                     f"{eligible} eligible, {len(signals)} signals")

        signals.sort(key=lambda s: s.edge, reverse=True)
        return signals

    def _make_signal(
        self, market: Market, token_id: str, side_label: str,
        price: float, size: int, edge: float, fair: float,
        implied: float, momentum: float, duration: str,
    ) -> Signal:
        """Create a market-making signal."""
        return Signal(
            asset="BTC",
            market=market,
            token_id=token_id,
            side="BUY",
            token_side=side_label,
            spot_price=0.0,
            target_price=0.0,
            implied_prob=implied,
            fair_prob=fair,
            edge=edge,
            confidence=0.70,
            momentum_score=momentum,
            momentum_direction="n/a",
            confirming_timeframes=0,
            suggested_price=round(price, 2),
            suggested_size=size,
            timestamp=time.time(),
            reason=(
                f"MM-{duration}: BUY {side_label}@{price:.2f} | "
                f"fair={fair:.2f} edge={edge:.3f} | "
                f"inv={self._inventory.get(market.condition_id, 0):.0f}"
            ),
            strategy="market_maker",
        )

    def update_inventory(self, condition_id: str, side: str, shares: float):
        """Track inventory after a fill."""
        current = self._inventory.get(condition_id, 0.0)
        if side == "Up":
            self._inventory[condition_id] = current + shares
        else:
            self._inventory[condition_id] = current - shares
