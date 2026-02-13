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
import time
from datetime import datetime, timezone
from typing import Optional

from . import BaseStrategy, register_strategy
from ..config import ASSETS
from ..polymarket_client import PolymarketClient, Market, OrderBook
from ..signals import Signal

logger = logging.getLogger(__name__)

# ============================================================
# Strategy-specific config (overridable from config.py)
# ============================================================

GRINDER_CONFIG = {
    # Minimum YES price to consider (0.90 = 90c = 90% implied prob)
    "min_probability": 0.90,

    # Maximum YES price (don't buy at 99c — too little upside)
    "max_probability": 0.97,

    # Ideal sweet spot range for best risk/reward
    "sweet_spot_low": 0.91,
    "sweet_spot_high": 0.95,

    # Minimum liquidity in USDC
    "min_liquidity": 5_000,

    # Minimum total volume traded
    "min_volume": 10_000,

    # Maximum bid-ask spread
    "max_spread": 0.04,

    # Minimum depth: total size available at best ask (shares)
    "min_ask_depth": 50,

    # Size per trade (USDC). Small and repeated.
    "size_per_trade_usdc": 20.0,

    # Maximum concurrent positions from this strategy
    "max_positions": 50,

    # Maximum total exposure for this strategy
    "max_exposure_usdc": 2_000.0,

    # Don't buy markets expiring in less than N hours
    # (last-minute volatility risk)
    "min_hours_to_expiry": 2,

    # Don't buy markets expiring in more than N days
    # (capital tied up too long)
    "max_days_to_expiry": 30,

    # Categories to favor (these historically resolve as expected)
    "favored_keywords": [
        "will", "above", "below", "reach", "remain",
        "stay", "end", "close",
    ],

    # Categories to avoid (unpredictable events)
    "avoid_keywords": [
        "tweet", "say", "announce", "resign", "fire",
        "scandal", "hack", "exploit",
    ],

    # How many markets to scan per cycle (API rate limiting)
    "scan_limit": 200,

    # Minimum edge: implied prob must overstate true prob by at least this.
    # For a 93c contract, if we think true prob is 96%, edge = 0.03.
    # We set a small minimum since the strategy relies on volume, not big edges.
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
        self.cfg = GRINDER_CONFIG
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

        # Fetch a broad set of active markets
        all_markets = self.client.search_markets("", limit=self.cfg["scan_limit"],
                                                  active_only=True)

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

            # Check time to expiry
            hours = _hours_to_expiry(market.end_date)
            if hours is not None:
                if hours < self.cfg["min_hours_to_expiry"]:
                    filtered_expiry += 1
                    continue
                if hours > self.cfg["max_days_to_expiry"] * 24:
                    filtered_expiry += 1
                    continue

            # Get YES token order book
            if not market.yes_token_id:
                continue

            book = self.client.get_order_book(market.yes_token_id)
            if not book:
                continue

            scanned += 1
            yes_price = book.midpoint

            # Check probability range
            if yes_price < self.cfg["min_probability"] or yes_price > self.cfg["max_probability"]:
                filtered_prob += 1
                continue

            # Check spread
            if book.spread > self.cfg["max_spread"]:
                filtered_spread += 1
                continue

            # Check depth at best ask
            ask_depth = sum(size for _, size in book.asks[:3])  # Top 3 levels
            if ask_depth < self.cfg["min_ask_depth"]:
                continue

            # Calculate position sizing
            entry_price = book.best_ask  # We buy at the ask
            if entry_price > self.cfg["max_probability"]:
                continue

            profit_per_share = 1.0 - entry_price      # Profit if YES
            loss_per_share = entry_price               # Loss if NO
            implied_prob = entry_price
            # We assume true probability is slightly higher than implied
            # (market tends to under-price near-certainties due to opportunity cost)
            estimated_true_prob = min(0.99, implied_prob + 0.02)

            ev_per_share = (estimated_true_prob * profit_per_share -
                            (1 - estimated_true_prob) * loss_per_share)

            if ev_per_share < self.cfg["min_edge"] * profit_per_share:
                continue

            # Size: fixed USDC per trade
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

            # Determine asset tag (best effort)
            asset = "MISC"
            for a, cfg in ASSETS.items():
                tags = cfg.get("polymarket_tags", [])
                if any(t in q for t in tags):
                    asset = a
                    break

            signal = Signal(
                asset=asset,
                market=market,
                token_id=market.yes_token_id,
                side="BUY",
                token_side="YES",
                spot_price=0.0,  # N/A for this strategy
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
                    f"GRINDER: YES@{entry_price:.2f} | "
                    f"profit/loss={profit_per_share:.2f}/{loss_per_share:.2f} | "
                    f"EV={ev_per_share:+.4f}/share | "
                    f"liq=${market.liquidity:,.0f} vol=${market.volume:,.0f}"
                ),
                strategy="high_prob_grinder",
            )

            signals.append(signal)
            self._seen_conditions.add(market.condition_id)

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
