"""
Arbitrage Signal Generator.

Compares spot price momentum (from 5/10/15 min charts) against
Polymarket implied probabilities to detect mispricing.

Core logic:
  1. Fetch spot candles and run chart analysis
  2. Fetch Polymarket order books for relevant markets
  3. Estimate "fair" probability based on spot momentum
  4. Compare fair vs market probability → edge
  5. If edge > threshold and risk checks pass → signal

The key insight: Polymarket prediction markets for crypto price
targets are priced as probabilities (0-1). When spot price is
moving strongly toward a target on short timeframes, the market
often lags by a few cents. We capture this lag.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .charts import ChartAnalysis, analyze_all_timeframes, multi_timeframe_consensus
from .config import STRATEGY, TIMEFRAMES, ASSETS
from .exchanges import (
    CompositePrice, get_composite_price, get_best_candles,
)
from .polymarket_client import PolymarketClient, Market, OrderBook

logger = logging.getLogger(__name__)


@dataclass
class MarketContext:
    """A Polymarket market with its current pricing and the relevant target."""
    market: Market
    token_id: str           # The token we'd trade (YES or NO)
    side: str               # "YES" or "NO"
    book: Optional[OrderBook] = None
    implied_prob: float = 0.0
    target_price: Optional[float] = None  # Extracted target (e.g., 100000 for "BTC above $100k")
    direction: str = ""     # "above" or "below"


@dataclass
class Signal:
    """An arbitrage signal produced by any strategy."""
    asset: str
    market: Market
    token_id: str
    side: str               # "BUY" or "SELL"
    token_side: str         # "YES", "NO", "YES+NO" (bilateral), "ALL_N_OUTCOMES"

    # Pricing
    spot_price: float
    target_price: float
    implied_prob: float     # Current Polymarket probability
    fair_prob: float        # Our estimated fair probability
    edge: float             # fair_prob - implied_prob (positive = underpriced)

    # Confidence
    confidence: float       # 0-1 confidence in the signal
    momentum_score: float
    momentum_direction: str
    confirming_timeframes: int

    # Order details
    suggested_price: float  # Price to bid/ask
    suggested_size: float   # Size in shares

    # Metadata
    timestamp: float = 0.0
    reason: str = ""

    # Strategy identification
    strategy: str = "spot_divergence"

    # For bilateral arb: details of each leg to execute
    # List of dicts: [{"token_id": ..., "side": ..., "price": ..., "label": ...}]
    arb_legs: list = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.arb_legs is None:
            self.arb_legs = []

    @property
    def is_bilateral(self) -> bool:
        return len(self.arb_legs) >= 2


def is_updown_market(question: str) -> bool:
    """Check if a market is a Bitcoin Up or Down short-duration market."""
    return "up or down" in question.lower()


def extract_target_price(question: str) -> tuple[Optional[float], str]:
    """
    Extract the price target and direction from a Polymarket question.

    Examples:
        "Will Bitcoin be above $100,000 on March 1?" → (100000.0, "above")
        "Will ETH reach $5,000 by end of Q1?"        → (5000.0, "above")
        "Will BTC drop below $80,000?"                → (80000.0, "below")

    For "Up or Down" markets, returns (0.0, "updown") as a sentinel.
    These markets don't have a price target — they're pure directional bets.
    """
    q = question.lower()

    # Up or Down markets: no price target, pure directional
    if "up or down" in q:
        return 0.0, "updown"

    # Determine direction
    if any(kw in q for kw in ["above", "over", "higher than", "reach", "hit", "exceed"]):
        direction = "above"
    elif any(kw in q for kw in ["below", "under", "lower than", "drop", "fall"]):
        direction = "below"
    else:
        direction = "above"  # Default assumption

    # Extract dollar amounts
    # Matches: $100,000 | $100000 | $100k | 100,000 | 100000
    patterns = [
        r'\$?([\d,]+\.?\d*)\s*k\b',           # $100k or 100k
        r'\$([\d,]+\.?\d*)',                     # $100,000 or $100000
        r'([\d,]+\.?\d*)\s*(?:dollars|usd)',     # 100000 dollars
    ]

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            val_str = match.group(1).replace(",", "")
            val = float(val_str)
            if "k" in q[match.start():match.end() + 1].lower():
                val *= 1000
            return val, direction

    return None, direction


def estimate_updown_fair_probability(
    momentum_score: float,
    roc_pct: float,
    volume_ratio: float = 1.0,
) -> float:
    """
    Estimate the fair probability that BTC goes "Up" in a short-duration window.

    For Up/Down markets, there's no price target — it's simply whether
    the price is higher or lower at the end of the window vs the start.

    Inputs:
      - momentum_score: -1.0 (bearish) to +1.0 (bullish) from chart analysis
      - roc_pct: recent rate of change (%) — directional momentum
      - volume_ratio: current volume / average volume — conviction strength

    Returns probability of "Up" outcome (0.02 to 0.98).
    """
    # Base: 50/50 (no edge without signal)
    base = 0.50

    # Momentum drives direction: strong momentum = higher probability
    # Scale: ±0.20 max from momentum alone
    momentum_adj = momentum_score * 0.20

    # ROC provides extrapolation: if moving up fast, likely continues
    # Cap at ±0.10
    roc_adj = max(-0.10, min(0.10, roc_pct * 0.04))

    # High volume confirms the move
    if volume_ratio > 1.5:
        volume_adj = 0.03 * (1 if momentum_score > 0 else -1)
    else:
        volume_adj = 0.0

    fair = base + momentum_adj + roc_adj + volume_adj
    return max(0.02, min(0.98, fair))


def estimate_fair_probability(
    spot_price: float,
    target_price: float,
    direction: str,
    momentum_score: float,
    roc_pct: float,
    time_to_expiry_hours: Optional[float] = None,
) -> float:
    """
    Estimate the "fair" probability that spot price will be above/below
    the target, given current momentum.

    This is a simplified model based on:
    1. Distance from target (closer = higher probability if moving toward it)
    2. Momentum strength (stronger momentum = higher probability)
    3. Recent rate of change (extrapolate the move)

    For a proper implementation, you'd use a jump-diffusion model or
    implied volatility surface. This is a first-order approximation
    that's good enough for detecting clear mispricings.
    """
    if target_price == 0:
        return 0.5

    # Distance from target as % of spot
    distance_pct = (target_price - spot_price) / spot_price * 100

    if direction == "above":
        # Positive distance = target is above current price
        # Negative distance = target is below current price (already past it)

        if distance_pct < 0:
            # Already above target
            base_prob = 0.70 + min(0.25, abs(distance_pct) * 0.05)
        elif distance_pct < 0.5:
            # Very close to target
            base_prob = 0.50
        elif distance_pct < 2.0:
            # Moderate distance
            base_prob = 0.40 - distance_pct * 0.05
        else:
            # Far from target
            base_prob = max(0.05, 0.30 - distance_pct * 0.03)

        # Momentum adjustment
        if momentum_score > 0:
            # Bullish momentum increases probability of being above target
            momentum_adj = momentum_score * 0.15
        else:
            # Bearish momentum decreases it
            momentum_adj = momentum_score * 0.10

        # ROC extrapolation: if we're moving toward target at current rate,
        # how likely are we to reach it?
        if roc_pct > 0 and distance_pct > 0:
            # Moving toward target
            roc_adj = min(0.10, roc_pct * 0.05)
        elif roc_pct < 0 and distance_pct > 0:
            # Moving away from target
            roc_adj = max(-0.10, roc_pct * 0.03)
        else:
            roc_adj = 0.0

    else:  # direction == "below"
        if distance_pct > 0:
            # Target is above current price, we're already below it
            base_prob = 0.70 + min(0.25, distance_pct * 0.05)
        elif distance_pct > -0.5:
            base_prob = 0.50
        elif distance_pct > -2.0:
            base_prob = 0.40 + distance_pct * 0.05
        else:
            base_prob = max(0.05, 0.30 + distance_pct * 0.03)

        if momentum_score < 0:
            momentum_adj = abs(momentum_score) * 0.15
        else:
            momentum_adj = -momentum_score * 0.10

        if roc_pct < 0 and distance_pct < 0:
            roc_adj = min(0.10, abs(roc_pct) * 0.05)
        elif roc_pct > 0 and distance_pct < 0:
            roc_adj = max(-0.10, -roc_pct * 0.03)
        else:
            roc_adj = 0.0

    fair_prob = base_prob + momentum_adj + roc_adj
    return max(0.02, min(0.98, fair_prob))


def generate_signals(
    client: PolymarketClient,
    assets: Optional[list[str]] = None,
) -> list[Signal]:
    """
    Main signal generation loop.

    1. For each tracked asset, fetch spot prices and candles
    2. Run chart analysis on 5/10/15 min timeframes
    3. Find relevant Polymarket markets
    4. Compare spot momentum vs Polymarket implied probability
    5. Generate signals where edge exceeds threshold

    Args:
        client: Initialized PolymarketClient
        assets: List of assets to analyze (default: all from config)

    Returns:
        List of Signal objects, sorted by edge (best first)
    """
    if assets is None:
        assets = list(ASSETS.keys())

    all_signals = []

    for asset in assets:
        logger.info(f"Analyzing {asset}...")

        # Step 1: Get composite spot price
        composite = get_composite_price(asset)
        if not composite:
            logger.warning(f"  No spot price for {asset}, skipping")
            continue

        spot_price = composite.price
        logger.info(f"  Spot: ${spot_price:,.2f} (from {len(composite.sources)} exchanges)")

        # Step 2: Fetch candles and analyze charts
        candles_by_tf = {}
        for tf in TIMEFRAMES:
            candles = get_best_candles(asset, tf, limit=100)
            if candles:
                candles_by_tf[tf] = candles

        if not candles_by_tf:
            logger.warning(f"  No candle data for {asset}, skipping")
            continue

        analyses = analyze_all_timeframes(asset, candles_by_tf)
        if not analyses:
            logger.warning(f"  Chart analysis failed for {asset}, skipping")
            continue

        consensus_score, consensus_dir = multi_timeframe_consensus(analyses)
        confirming = sum(1 for a in analyses
                         if a.momentum_direction == consensus_dir
                         and consensus_dir != "neutral")

        logger.info(f"  Momentum: {consensus_dir} (score={consensus_score:+.3f}, "
                     f"{confirming}/{len(analyses)} timeframes)")

        # Skip if momentum is too weak
        if abs(consensus_score) < 0.10:
            logger.info(f"  Weak momentum, skipping signal generation")
            continue

        # Step 3: Find relevant Polymarket markets
        markets = client.find_crypto_price_markets(asset)
        if not markets:
            logger.info(f"  No relevant Polymarket markets for {asset}")
            continue

        logger.info(f"  Found {len(markets)} Polymarket markets")

        # Step 4: Evaluate each market for edge
        for market in markets:
            target_price, direction = extract_target_price(market.question)
            if target_price is None:
                continue

            # Use best available ROC for estimation
            best_roc = analyses[0].roc_3 if analyses else 0.0

            # --- Up/Down markets: pure directional bet ---
            if direction == "updown":
                signal = _evaluate_updown_signal(
                    client, asset, market, spot_price,
                    consensus_score, consensus_dir, confirming, analyses, best_roc,
                )
                if signal:
                    all_signals.append(signal)
                continue

            # --- Standard price-target markets ---
            # Get order book for YES token
            if not market.yes_token_id:
                continue

            book = client.get_order_book(market.yes_token_id)
            if not book:
                continue

            # Skip illiquid markets
            if book.spread > STRATEGY["max_spread"]:
                logger.debug(f"  Skipping {market.question[:60]}... (spread={book.spread:.3f})")
                continue

            implied_prob = book.midpoint

            # Estimate fair probability
            fair_prob = estimate_fair_probability(
                spot_price=spot_price,
                target_price=target_price,
                direction=direction,
                momentum_score=consensus_score,
                roc_pct=best_roc,
            )

            # Calculate edge
            # For "above" markets with bullish momentum → BUY YES if underpriced
            # For "above" markets with bearish momentum → BUY NO (SELL YES) if overpriced
            # For "below" markets, reverse the logic

            if direction == "above":
                if consensus_dir == "bullish":
                    # We think price will go up → YES is underpriced
                    edge = fair_prob - implied_prob
                    trade_side = "BUY"
                    token_side = "YES"
                    token_id = market.yes_token_id
                    suggested_price = min(book.best_ask, implied_prob + edge * 0.5)
                else:
                    # Bearish → NO is underpriced (YES overpriced)
                    edge = implied_prob - fair_prob
                    trade_side = "BUY"
                    token_side = "NO"
                    token_id = market.no_token_id
                    if not token_id:
                        continue
                    # NO price = 1 - YES price
                    no_book = client.get_order_book(token_id)
                    if not no_book:
                        continue
                    suggested_price = min(no_book.best_ask, (1 - fair_prob) + edge * 0.5)
            else:  # direction == "below"
                if consensus_dir == "bearish":
                    edge = fair_prob - implied_prob
                    trade_side = "BUY"
                    token_side = "YES"
                    token_id = market.yes_token_id
                    suggested_price = min(book.best_ask, implied_prob + edge * 0.5)
                else:
                    edge = implied_prob - fair_prob
                    trade_side = "BUY"
                    token_side = "NO"
                    token_id = market.no_token_id
                    if not token_id:
                        continue
                    no_book = client.get_order_book(token_id)
                    if not no_book:
                        continue
                    suggested_price = min(no_book.best_ask, (1 - fair_prob) + edge * 0.5)

            # Only signal if edge exceeds minimum
            if edge < STRATEGY["min_edge"]:
                continue

            # Confidence score
            confidence = min(1.0, (
                abs(consensus_score) * STRATEGY["weights"]["momentum"] +
                min(1, abs(analyses[0].vwap_deviation) / 0.5) * STRATEGY["weights"]["vwap_dev"] +
                (1 if analyses[0].rsi > 60 or analyses[0].rsi < 40 else 0.5) * STRATEGY["weights"]["rsi"] +
                min(1, analyses[0].volume_ratio / 2) * STRATEGY["weights"]["volume"] +
                (confirming / len(analyses)) * STRATEGY["weights"]["multi_tf"]
            ))

            # Suggested size based on edge and confidence
            from .config import RISK
            max_size = RISK["max_position_usdc"]
            suggested_size = max_size * confidence * min(1.0, edge / 0.10)
            suggested_size = max(1, round(suggested_size / suggested_price))

            signal = Signal(
                asset=asset,
                market=market,
                token_id=token_id,
                side=trade_side,
                token_side=token_side,
                spot_price=spot_price,
                target_price=target_price,
                implied_prob=implied_prob,
                fair_prob=fair_prob,
                edge=edge,
                confidence=confidence,
                momentum_score=consensus_score,
                momentum_direction=consensus_dir,
                confirming_timeframes=confirming,
                suggested_price=round(suggested_price, 2),
                suggested_size=suggested_size,
                timestamp=time.time(),
                reason=(
                    f"{consensus_dir} momentum ({consensus_score:+.2f}) on "
                    f"{confirming}/{len(analyses)} TFs | "
                    f"spot=${spot_price:,.0f} vs target=${target_price:,.0f} | "
                    f"edge={edge:.3f} ({edge*100:.1f}c)"
                ),
            )

            all_signals.append(signal)
            logger.info(f"  SIGNAL: {trade_side} {token_side} on "
                         f"\"{market.question[:50]}...\" | "
                         f"edge={edge:.3f} conf={confidence:.2f}")

    # Sort by edge * confidence (best opportunities first)
    all_signals.sort(key=lambda s: s.edge * s.confidence, reverse=True)
    return all_signals


def _evaluate_updown_signal(
    client: PolymarketClient,
    asset: str, market: Market, spot_price: float,
    consensus_score: float, consensus_dir: str,
    confirming: int, analyses: list, best_roc: float,
) -> Optional[Signal]:
    """
    Evaluate a Bitcoin "Up or Down" market for a trade signal.

    Outcomes: token_ids[0] = "Up", token_ids[1] = "Down".
    If momentum is bullish → buy "Up" when underpriced.
    If momentum is bearish → buy "Down" when underpriced.
    """
    if len(market.token_ids) < 2:
        return None

    up_token_id = market.token_ids[0]
    down_token_id = market.token_ids[1]

    volume_ratio = analyses[0].volume_ratio if analyses else 1.0
    fair_up = estimate_updown_fair_probability(
        momentum_score=consensus_score,
        roc_pct=best_roc,
        volume_ratio=volume_ratio,
    )

    if consensus_dir == "bullish":
        token_id = up_token_id
        token_side = "Up"
        fair_prob = fair_up
    elif consensus_dir == "bearish":
        token_id = down_token_id
        token_side = "Down"
        fair_prob = 1.0 - fair_up
    else:
        return None

    book = client.get_order_book(token_id)
    if not book:
        return None

    if book.spread > STRATEGY["max_spread"]:
        return None

    implied_prob = book.midpoint
    edge = fair_prob - implied_prob

    if edge < STRATEGY["min_edge"]:
        return None

    suggested_price = min(book.best_ask, implied_prob + edge * 0.5)

    confidence = min(1.0, (
        abs(consensus_score) * STRATEGY["weights"]["momentum"] +
        min(1, abs(analyses[0].vwap_deviation) / 0.5) * STRATEGY["weights"]["vwap_dev"] +
        (1 if analyses[0].rsi > 60 or analyses[0].rsi < 40 else 0.5) * STRATEGY["weights"]["rsi"] +
        min(1, analyses[0].volume_ratio / 2) * STRATEGY["weights"]["volume"] +
        (confirming / len(analyses)) * STRATEGY["weights"]["multi_tf"]
    ))

    from .config import RISK
    max_size = RISK["max_position_usdc"]
    suggested_size = max_size * confidence * min(1.0, edge / 0.10)
    suggested_size = max(1, round(suggested_size / suggested_price))

    signal = Signal(
        asset=asset,
        market=market,
        token_id=token_id,
        side="BUY",
        token_side=token_side,
        spot_price=spot_price,
        target_price=0.0,
        implied_prob=implied_prob,
        fair_prob=fair_prob,
        edge=edge,
        confidence=confidence,
        momentum_score=consensus_score,
        momentum_direction=consensus_dir,
        confirming_timeframes=confirming,
        suggested_price=round(suggested_price, 2),
        suggested_size=suggested_size,
        timestamp=time.time(),
        reason=(
            f"BTC-UPDOWN: {consensus_dir} ({consensus_score:+.2f}) on "
            f"{confirming}/{len(analyses)} TFs | "
            f"spot=${spot_price:,.0f} | "
            f"BUY {token_side} edge={edge:.3f} ({edge*100:.1f}c)"
        ),
    )

    logger.info(f"  SIGNAL: BUY {token_side} on "
                 f"\"{market.question[:50]}...\" | "
                 f"edge={edge:.3f} conf={confidence:.2f}")
    return signal
