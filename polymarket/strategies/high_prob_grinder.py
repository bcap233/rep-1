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
from ..config import ASSETS, HIGH_PROB_GRINDER, EVENT_CATEGORIES
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

    def scan(self, assets: Optional[list[str]] = None) -> list[Signal]:
        """
        Scan all active markets for high-probability opportunities.

        This strategy ignores the `assets` parameter — it scans everything.
        The grinder doesn't care what the market is about, only that
        the probability is high, the market is liquid, and the math works.
        """
        logger.info(f"[GRINDER] Scanning for >{self.cfg['min_probability']*100:.0f}% events...")

        # Fetch markets: broad scan + category-specific searches
        if self.cfg.get("category_scan", True):
            all_markets = self.client.search_all_categories(
                limit_per_query=self.cfg["scan_limit"])
        else:
            all_markets = self.client.search_markets(
                "", limit=self.cfg["scan_limit"], active_only=True)

        if not all_markets:
            logger.warning("[GRINDER] No markets returned from search")
            return []

        logger.info(f"[GRINDER] Fetched {len(all_markets)} active markets, filtering...")

        signals = []
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

    def reset_seen(self):
        """Clear the seen-conditions cache (e.g., new trading session)."""
        self._seen_conditions.clear()
