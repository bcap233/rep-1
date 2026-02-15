"""
Strategy 1: High Probability Grinder.

The concept:
  Buy YES tokens on events priced at 90%+ (i.e., $0.90+) at massive scale.
  These events resolve YES most of the time, so you collect the 5-10c
  spread on every win. The math favors you as long as your win rate
  exceeds the implied probability.

  Example:
    Buy YES at $0.93 → resolves YES → profit $0.07 per share (7.5% return)
    Buy YES at $0.93 → resolves NO  → loss  $0.93 per share

  Break-even: need >93% win rate at 93c entry price.
  If actual win rate is 96% (common for >90% events), expected value:
    EV = 0.96 * 0.07 - 0.04 * 0.93 = 0.0672 - 0.0372 = +$0.03/share

  At scale (1000+ trades), the law of large numbers kicks in.

Filters:
  - Minimum probability (default 90c)
  - Minimum liquidity (avoid illiquid traps)
  - Minimum volume (social proof of price discovery)
  - Maximum spread (tight markets only)
  - Time-to-expiry window (not too close, not too far)
  - Historical category win rates (some categories are more reliable)
  - Market size requirements (avoid tiny markets with no depth)
"""

import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

from . import BaseStrategy, register_strategy
from ..config import ASSETS, HIGH_PROB_GRINDER, EVENT_CATEGORIES, MARKET_MAKER
from ..polymarket_client import PolymarketClient, Market, OrderBook
from ..signals import Signal

logger = logging.getLogger(__name__)


def _classify_market(question_lower: str) -> str:
    """Classify a market into a category label based on its question text."""
    # Try crypto assets first (word-boundary match to avoid false positives
    # like "sol" matching "soliciting" or "eth" matching "Yesilgoz")
    for asset, cfg in ASSETS.items():
        tags = cfg.get("polymarket_tags", [])
        if any(re.search(rf'\b{re.escape(t)}\b', question_lower) for t in tags):
            return asset

    # Then try event categories
    for cat_name, cat_cfg in EVENT_CATEGORIES.items():
        for query in cat_cfg["queries"]:
            if query in question_lower:
                return cat_cfg["label"]

    return "MISC"


# Default config — merged with config.py overrides at init
_DEFAULTS = {
    "min_probability": 0.85,
    "max_probability": 0.97,
    "sweet_spot_low": 0.88,
    "sweet_spot_high": 0.95,
    "min_liquidity": 5_000,
    "min_volume": 10_000,
    "max_spread": 0.04,
    "min_ask_depth": 50,
    "size_per_trade_usdc": 20.0,
    "max_positions": 50,
    "max_exposure_usdc": 2_000.0,
    "min_hours_to_expiry": 0,
    "max_days_to_expiry": 365,
    "favored_keywords": [
        "will", "above", "below", "reach", "remain",
        "stay", "end", "close",
    ],
    "avoid_keywords": [
        "tweet", "say", "announce", "resign", "fire",
        "scandal", "hack", "exploit",
    ],
    "scan_limit": 200,
    "min_edge": 0.01,
    # Spread-lock: buy both sides of an Up/Down market for guaranteed profit.
    # lock_max_total = max you'll pay for Up + Down combined (< $1.00).
    # lock_min_profit = minimum net profit per share after fees to bother.
    "lock_max_total": 0.98,
    "lock_min_profit": 0.005,
    "lock_fee_rate": 0.02,
}


def _parse_end_date(date_str: str) -> Optional[datetime]:
    """Parse a market end date string into a datetime."""
    if not date_str:
        return None
    for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]:
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _hours_to_expiry(end_date_str: str) -> Optional[float]:
    """Calculate hours until market resolution."""
    dt = _parse_end_date(end_date_str)
    if not dt:
        return None
    now = datetime.now(timezone.utc)
    delta = dt - now
    return delta.total_seconds() / 3600


@register_strategy
class HighProbGrinder(BaseStrategy):
    """
    Buy high-probability YES tokens at scale.

    Scans all active Polymarket markets for YES tokens priced 90-97c.
    Filters for liquidity, volume, spread, and time-to-expiry.
    Places many small bets, relying on the law of large numbers.

    Usage:
        python -m polymarket --strategy high_prob_grinder --loop
    """

    name = "high_prob_grinder"
    description = "Buy 90%+ events at scale for recurring small profits"

    def __init__(self, client: PolymarketClient):
        super().__init__(client)
        # Merge: config.py overrides take precedence over defaults
        self.cfg = {**_DEFAULTS, **HIGH_PROB_GRINDER}
        # Track what we've already bought this session to avoid duplicates
        self._seen_conditions: set[str] = set()
        # Open legs on Up/Down markets, waiting for lock opportunities.
        # condition_id → {side, entry_price, other_token_id, other_side, market}
        self._open_legs: dict[str, dict] = {}

    def scan(self, assets: Optional[list[str]] = None) -> list[Signal]:
        """
        Scan all active markets for high-probability opportunities.

        Two market sources:
        1. Gamma API (event markets): politics, sports, economics, etc.
        2. BTC Up/Down 5m/15m markets: when momentum pushes one side to 90%+

        The grinder doesn't care what the market is about, only that
        the probability is high, the market is liquid, and the math works.
        """
        logger.info(f"[GRINDER] Scanning for >{self.cfg['min_probability']*100:.0f}% events...")

        signals = []

        # Source 1: BTC/ETH Up/Down short-duration markets
        updown_signals = self._scan_updown_markets()
        signals.extend(updown_signals)

        # Source 2: Gamma API broad market scan
        if self.cfg.get("category_scan", True):
            all_markets = self.client.search_all_categories(
                limit_per_query=self.cfg["scan_limit"])
        else:
            all_markets = self.client.search_markets(
                "", limit=self.cfg["scan_limit"], active_only=True)

        if not all_markets:
            logger.warning("[GRINDER] No event markets returned from search")
            if signals:
                signals.sort(key=lambda s: s.edge, reverse=True)
                return signals
            return []

        logger.info(f"[GRINDER] Fetched {len(all_markets)} event markets, filtering...")

        scanned = 0
        filtered_prob = 0
        filtered_liquidity = 0
        filtered_spread = 0
        filtered_expiry = 0
        filtered_keywords = 0

        for market in all_markets:
            # Skip already-seen markets
            if market.condition_id in self._seen_conditions:
                continue

            # Quick keyword filter
            q = market.question.lower()
            if any(kw in q for kw in self.cfg["avoid_keywords"]):
                filtered_keywords += 1
                continue

            # Check liquidity and volume thresholds
            if market.liquidity < self.cfg["min_liquidity"]:
                filtered_liquidity += 1
                continue

            if market.volume < self.cfg["min_volume"]:
                filtered_liquidity += 1
                continue

            # Check time to expiry (when end_date is available).
            # Many Polymarket markets are event-driven with no fixed end date
            # (e.g., "Will X happen before Y"). These are still good grinder
            # candidates if they pass quality filters.
            hours = _hours_to_expiry(market.end_date)
            if hours is not None:
                if hours < self.cfg["min_hours_to_expiry"]:
                    filtered_expiry += 1
                    continue
                if hours > self.cfg["max_days_to_expiry"] * 24:
                    filtered_expiry += 1
                    continue

            # Check both YES and NO tokens for high-probability side
            candidates = []
            if market.yes_token_id:
                candidates.append((market.yes_token_id, "YES"))
            if market.no_token_id:
                candidates.append((market.no_token_id, "NO"))

            if not candidates:
                continue

            for token_id, token_side in candidates:
                book = self.client.get_order_book(token_id)
                if not book:
                    continue

                scanned += 1
                mid = book.midpoint

                # Check probability range
                if mid < self.cfg["min_probability"] or mid > self.cfg["max_probability"]:
                    filtered_prob += 1
                    continue

                # Check spread
                if book.spread > self.cfg["max_spread"]:
                    filtered_spread += 1
                    continue

                # Check depth at best ask
                ask_depth = sum(size for _, size in book.asks[:3])
                if ask_depth < self.cfg["min_ask_depth"]:
                    continue

                # Calculate position sizing
                entry_price = book.best_ask
                if entry_price > self.cfg["max_probability"]:
                    continue

                profit_per_share = 1.0 - entry_price
                loss_per_share = entry_price
                implied_prob = entry_price
                estimated_true_prob = min(0.99, implied_prob + 0.02)

                ev_per_share = (estimated_true_prob * profit_per_share -
                                (1 - estimated_true_prob) * loss_per_share)

                if ev_per_share < self.cfg["min_edge"] * profit_per_share:
                    continue

                size = max(1, int(self.cfg["size_per_trade_usdc"] / entry_price))

                # Confidence: higher for sweet spot, lower at edges
                if self.cfg["sweet_spot_low"] <= entry_price <= self.cfg["sweet_spot_high"]:
                    confidence = 0.85
                else:
                    confidence = 0.65

                # Bonus confidence for favored keywords
                if any(kw in q for kw in self.cfg["favored_keywords"]):
                    confidence = min(1.0, confidence + 0.10)

                # Bonus for high volume (better price discovery)
                if market.volume > 100_000:
                    confidence = min(1.0, confidence + 0.05)

                edge = ev_per_share
                asset = _classify_market(q)

                signal = Signal(
                    asset=asset,
                    market=market,
                    token_id=token_id,
                    side="BUY",
                    token_side=token_side,
                    spot_price=0.0,
                    target_price=0.0,
                    implied_prob=implied_prob,
                    fair_prob=estimated_true_prob,
                    edge=edge,
                    confidence=confidence,
                    momentum_score=0.0,
                    momentum_direction="n/a",
                    confirming_timeframes=0,
                    suggested_price=round(entry_price, 2),
                    suggested_size=size,
                    timestamp=time.time(),
                    reason=(
                        f"GRINDER: {token_side}@{entry_price:.2f} | "
                        f"profit/loss={profit_per_share:.2f}/{loss_per_share:.2f} | "
                        f"EV={ev_per_share:+.4f}/share | "
                        f"liq=${market.liquidity:,.0f} vol=${market.volume:,.0f}"
                    ),
                    strategy="high_prob_grinder",
                )

                signals.append(signal)
                self._seen_conditions.add(market.condition_id)
                break  # One signal per market (best side wins)

        logger.info(f"[GRINDER] Scanned {scanned} books | "
                     f"Filtered: prob={filtered_prob} liq={filtered_liquidity} "
                     f"spread={filtered_spread} expiry={filtered_expiry} "
                     f"kw={filtered_keywords}")
        logger.info(f"[GRINDER] Found {len(signals)} opportunities")

        # Sort by EV per share (best first)
        signals.sort(key=lambda s: s.edge, reverse=True)
        return signals

    def _scan_updown_markets(self) -> list[Signal]:
        """Scan BTC Up/Down 5m/15m markets for high-prob AND lock setups.

        Two modes:
        1. DIRECTIONAL: Buy high-prob side at 85-97%, hold to resolution.
        2. SPREAD-LOCK: If we already hold one side (or both sides are
           cheap right now), buy the other side to lock in guaranteed
           profit. Total cost < $1.00 → riskless.

        When BTC has momentum, buy the strong side. If the spread
        tightens or reverses, lock in the gain instead of holding to
        resolution.
        """
        from .market_maker import _parse_market_duration, _minutes_to_market_end

        signals = []
        durations = MARKET_MAKER.get("durations", ["5m", "15m"])
        min_minutes = MARKET_MAKER.get("min_minutes_to_expiry", 2)

        # Clean up expired legs before scanning
        self._expire_stale_legs()

        markets = self.client.find_btc_updown_markets(durations)
        if not markets:
            return []

        scanned = 0
        locks_found = 0
        directional_found = 0

        for market in markets:
            q = market.question
            duration = _parse_market_duration(q)
            if duration not in durations:
                continue

            mins_left = _minutes_to_market_end(q)
            if mins_left is None or mins_left < min_minutes:
                continue

            if len(market.token_ids) < 2:
                continue

            up_token = market.token_ids[0]
            down_token = market.token_ids[1]

            # Fetch both order books — needed for lock check regardless
            up_book = self.client.get_order_book(up_token)
            down_book = self.client.get_order_book(down_token)
            if not up_book or not down_book:
                continue
            if not up_book.asks or not down_book.asks:
                continue
            scanned += 1

            # --- MODE 1: IMMEDIATE LOCK ---
            # Both sides available right now with total < lock threshold.
            # This is pure arb — guaranteed profit, no directional risk.
            up_ask = up_book.best_ask
            down_ask = down_book.best_ask
            total_cost = up_ask + down_ask

            max_profit_side = max(1.0 - up_ask, 1.0 - down_ask)
            fee = self.cfg["lock_fee_rate"] * max_profit_side
            net_profit = 1.0 - total_cost - fee

            if (total_cost <= self.cfg["lock_max_total"]
                    and net_profit >= self.cfg["lock_min_profit"]):
                up_depth = sum(s for _, s in up_book.asks[:3])
                down_depth = sum(s for _, s in down_book.asks[:3])
                min_depth = min(up_depth, down_depth)

                if min_depth >= self.cfg["min_ask_depth"]:
                    max_price = max(up_ask, down_ask)
                    size = max(1, int(self.cfg["size_per_trade_usdc"] / max_price))
                    size = min(size, int(min_depth))

                    signal = Signal(
                        asset="BTC",
                        market=market,
                        token_id=up_token,
                        side="BUY",
                        token_side="Up+Down",
                        spot_price=0.0,
                        target_price=0.0,
                        implied_prob=total_cost,
                        fair_prob=1.0,
                        edge=net_profit,
                        confidence=0.99,
                        momentum_score=0.0,
                        momentum_direction="n/a",
                        confirming_timeframes=0,
                        suggested_price=round(total_cost, 2),
                        suggested_size=size,
                        timestamp=time.time(),
                        reason=(
                            f"LOCK-IMMEDIATE: Up@{up_ask:.2f}+Down@{down_ask:.2f}"
                            f"={total_cost:.2f} ({duration}, {mins_left:.0f}m) | "
                            f"profit={net_profit:.3f}/sh (${net_profit*size:.2f})"
                        ),
                        strategy="high_prob_grinder",
                        arb_legs=[
                            {"token_id": up_token, "side": "BUY",
                             "price": up_ask, "label": "Up"},
                            {"token_id": down_token, "side": "BUY",
                             "price": down_ask, "label": "Down"},
                        ],
                    )
                    signals.append(signal)
                    locks_found += 1
                    # Don't mark as seen — we might also want a directional
                    # entry if the lock doesn't execute (size limited, etc.)
                    continue  # Lock takes priority over directional

            # --- MODE 2: DELAYED LOCK ---
            # We already hold one side from a previous scan. Check if the
            # other side is now cheap enough to lock in profit.
            cond = market.condition_id
            if cond in self._open_legs:
                leg = self._open_legs[cond]
                other_token = leg["other_token_id"]
                other_book = (down_book if leg["side"] == "Up" else up_book)
                other_ask = other_book.best_ask

                lock_total = leg["entry_price"] + other_ask
                lock_profit = 1.0 - lock_total - fee
                other_depth = sum(s for _, s in other_book.asks[:3])

                if (lock_total <= self.cfg["lock_max_total"]
                        and lock_profit >= self.cfg["lock_min_profit"]
                        and other_depth >= self.cfg["min_ask_depth"]):
                    size = max(1, int(self.cfg["size_per_trade_usdc"] / other_ask))
                    size = min(size, int(other_depth), leg["size"])

                    signal = Signal(
                        asset="BTC",
                        market=market,
                        token_id=other_token,
                        side="BUY",
                        token_side=leg["other_side"],
                        spot_price=0.0,
                        target_price=0.0,
                        implied_prob=other_ask,
                        fair_prob=1.0,
                        edge=lock_profit,
                        confidence=0.99,
                        momentum_score=0.0,
                        momentum_direction="n/a",
                        confirming_timeframes=0,
                        suggested_price=round(other_ask, 2),
                        suggested_size=size,
                        timestamp=time.time(),
                        reason=(
                            f"LOCK-DELAYED: held {leg['side']}@{leg['entry_price']:.2f}"
                            f" + buy {leg['other_side']}@{other_ask:.2f}"
                            f"={lock_total:.2f} ({duration}, {mins_left:.0f}m) | "
                            f"locked profit={lock_profit:.3f}/sh"
                        ),
                        strategy="high_prob_grinder",
                    )
                    signals.append(signal)
                    locks_found += 1
                    # Remove the tracked leg — it's now locked
                    del self._open_legs[cond]
                    continue

            # --- MODE 3: DIRECTIONAL ENTRY ---
            # One side is at 85-97%. Buy it and hold to resolution.
            # Also track it as an open leg for potential lock later.
            if cond in self._seen_conditions:
                continue

            sides = [
                (up_token, "Up", up_book, down_token, "Down"),
                (down_token, "Down", down_book, up_token, "Up"),
            ]

            for token_id, token_side, book, other_tok, other_side in sides:
                mid = book.midpoint
                if mid < self.cfg["min_probability"] or mid > self.cfg["max_probability"]:
                    continue
                if book.spread > self.cfg["max_spread"]:
                    continue

                ask_depth = sum(size for _, size in book.asks[:3])
                if ask_depth < self.cfg["min_ask_depth"]:
                    continue

                entry_price = book.best_ask
                if entry_price > self.cfg["max_probability"]:
                    continue

                profit_per_share = 1.0 - entry_price
                implied_prob = entry_price
                estimated_true_prob = min(0.99, implied_prob + 0.02)

                ev_per_share = (estimated_true_prob * profit_per_share -
                                (1 - estimated_true_prob) * entry_price)

                if ev_per_share < self.cfg["min_edge"] * profit_per_share:
                    continue

                size = max(1, int(self.cfg["size_per_trade_usdc"] / entry_price))

                if self.cfg["sweet_spot_low"] <= entry_price <= self.cfg["sweet_spot_high"]:
                    confidence = 0.85
                else:
                    confidence = 0.65

                if duration == "5m":
                    confidence = min(1.0, confidence + 0.05)

                signal = Signal(
                    asset="BTC",
                    market=market,
                    token_id=token_id,
                    side="BUY",
                    token_side=token_side,
                    spot_price=0.0,
                    target_price=0.0,
                    implied_prob=implied_prob,
                    fair_prob=estimated_true_prob,
                    edge=ev_per_share,
                    confidence=confidence,
                    momentum_score=0.0,
                    momentum_direction="n/a",
                    confirming_timeframes=0,
                    suggested_price=round(entry_price, 2),
                    suggested_size=size,
                    timestamp=time.time(),
                    reason=(
                        f"GRINDER-UPDOWN: {token_side}@{entry_price:.2f} "
                        f"({duration}, {mins_left:.0f}m left) | "
                        f"EV={ev_per_share:+.4f}/share | "
                        f"spread={book.spread:.3f} depth={ask_depth:.0f}"
                    ),
                    strategy="high_prob_grinder",
                )
                signals.append(signal)
                self._seen_conditions.add(cond)
                directional_found += 1

                # Track this leg for delayed lock on next scan
                self._open_legs[cond] = {
                    "side": token_side,
                    "entry_price": entry_price,
                    "other_token_id": other_tok,
                    "other_side": other_side,
                    "size": size,
                    "market": market,
                    "time": time.time(),
                }
                break  # One signal per market (best side wins)

        logger.info(f"[GRINDER] Up/Down scan: {scanned} markets | "
                     f"{locks_found} locks + {directional_found} directional")
        return signals

    def _expire_stale_legs(self):
        """Remove open legs older than 20 minutes (market has resolved)."""
        cutoff = time.time() - 20 * 60
        expired = [cid for cid, leg in self._open_legs.items()
                   if leg["time"] < cutoff]
        for cid in expired:
            del self._open_legs[cid]
        if expired:
            logger.debug(f"[GRINDER] Expired {len(expired)} stale legs")

    def reset_seen(self):
        """Clear the seen-conditions cache (e.g., new trading session)."""
        self._seen_conditions.clear()
