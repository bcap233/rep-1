"""
Strategy 3: Spot-Implied Divergence.

The concept:
  Monitor spot prices across exchanges (Binance, Coinbase, Kraken) on
  5/10/15 minute charts. When spot momentum is strong and directional,
  Polymarket prediction market contract prices often lag by a few cents.

  Example:
    BTC rallies from $98k to $99.5k in 10 minutes (+1.5%).
    Polymarket "BTC above $100k by March 1" is still priced at $0.55
    but given the momentum, fair value is closer to $0.62.
    Buy YES at $0.55, capture the 7c lag as the market reprices.

This module wraps the existing signals.py logic into the strategy framework.
"""

import logging
import time
from typing import Optional

from . import BaseStrategy, register_strategy
from ..charts import analyze_all_timeframes, multi_timeframe_consensus
from ..config import STRATEGY, TIMEFRAMES, ASSETS, RISK
from ..exchanges import get_composite_price, get_best_candles
from ..polymarket_client import PolymarketClient, Market
from ..signals import Signal, extract_target_price, estimate_fair_probability

logger = logging.getLogger(__name__)


@register_strategy
class SpotDivergence(BaseStrategy):
    """
    Spot price momentum vs Polymarket implied probability divergence.

    Compares real-time exchange data across 5/10/15 minute timeframes
    against Polymarket contract prices to find momentum-driven mispricings.

    Usage:
        python -m polymarket --strategy spot_divergence --loop
        python -m polymarket --strategy spot_divergence --assets BTC,ETH
    """

    name = "spot_divergence"
    description = "Spot momentum vs Polymarket implied probability lag"

    def scan(self, assets: Optional[list[str]] = None) -> list[Signal]:
        """
        Scan for spot-vs-Polymarket divergence signals.

        For each asset:
        1. Fetch composite spot price from exchanges
        2. Build 5/10/15 min charts and compute momentum
        3. Find relevant Polymarket markets
        4. Compare fair probability (from momentum) vs implied (Polymarket)
        5. Signal when edge > threshold
        """
        if assets is None:
            assets = list(ASSETS.keys())

        all_signals = []

        for asset in assets:
            logger.info(f"[SPOT-DIV] Analyzing {asset}...")

            # Step 1: Composite spot price
            composite = get_composite_price(asset)
            if not composite:
                logger.warning(f"  No spot price for {asset}, skipping")
                continue

            spot_price = composite.price
            logger.info(f"  Spot: ${spot_price:,.2f} "
                         f"(from {len(composite.sources)} exchanges)")

            # Step 2: Chart analysis across timeframes
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
            confirming = sum(
                1 for a in analyses
                if a.momentum_direction == consensus_dir
                and consensus_dir != "neutral"
            )

            logger.info(f"  Momentum: {consensus_dir} "
                         f"(score={consensus_score:+.3f}, "
                         f"{confirming}/{len(analyses)} TFs)")

            # Skip weak momentum
            if abs(consensus_score) < 0.10:
                logger.info(f"  Weak momentum, skipping")
                continue

            # Step 3: Find relevant Polymarket markets
            markets = self.client.find_crypto_price_markets(asset)
            if not markets:
                logger.info(f"  No Polymarket markets for {asset}")
                continue

            logger.info(f"  Found {len(markets)} Polymarket markets")

            # Step 4: Evaluate each market
            for market in markets:
                target_price, direction = extract_target_price(market.question)
                if target_price is None:
                    continue

                if not market.yes_token_id:
                    continue

                book = self.client.get_order_book(market.yes_token_id)
                if not book:
                    continue

                if book.spread > STRATEGY["max_spread"]:
                    continue

                implied_prob = book.midpoint
                best_roc = analyses[0].roc_3 if analyses else 0.0

                fair_prob = estimate_fair_probability(
                    spot_price=spot_price,
                    target_price=target_price,
                    direction=direction,
                    momentum_score=consensus_score,
                    roc_pct=best_roc,
                )

                # Determine trade direction
                if direction == "above":
                    if consensus_dir == "bullish":
                        edge = fair_prob - implied_prob
                        trade_side = "BUY"
                        token_side = "YES"
                        token_id = market.yes_token_id
                        price = min(book.best_ask,
                                    implied_prob + edge * 0.5)
                    else:
                        edge = implied_prob - fair_prob
                        trade_side = "BUY"
                        token_side = "NO"
                        token_id = market.no_token_id
                        if not token_id:
                            continue
                        no_book = self.client.get_order_book(token_id)
                        if not no_book:
                            continue
                        price = min(no_book.best_ask,
                                    (1 - fair_prob) + edge * 0.5)
                else:
                    if consensus_dir == "bearish":
                        edge = fair_prob - implied_prob
                        trade_side = "BUY"
                        token_side = "YES"
                        token_id = market.yes_token_id
                        price = min(book.best_ask,
                                    implied_prob + edge * 0.5)
                    else:
                        edge = implied_prob - fair_prob
                        trade_side = "BUY"
                        token_side = "NO"
                        token_id = market.no_token_id
                        if not token_id:
                            continue
                        no_book = self.client.get_order_book(token_id)
                        if not no_book:
                            continue
                        price = min(no_book.best_ask,
                                    (1 - fair_prob) + edge * 0.5)

                if edge < STRATEGY["min_edge"]:
                    continue

                # Confidence
                confidence = min(1.0, (
                    abs(consensus_score) * STRATEGY["weights"]["momentum"]
                    + min(1, abs(analyses[0].vwap_deviation) / 0.5) * STRATEGY["weights"]["vwap_dev"]
                    + (1 if analyses[0].rsi > 60 or analyses[0].rsi < 40 else 0.5) * STRATEGY["weights"]["rsi"]
                    + min(1, analyses[0].volume_ratio / 2) * STRATEGY["weights"]["volume"]
                    + (confirming / len(analyses)) * STRATEGY["weights"]["multi_tf"]
                ))

                # Sizing
                max_size = RISK["max_position_usdc"]
                suggested_size = max_size * confidence * min(1.0, edge / 0.10)
                suggested_size = max(1, round(suggested_size / price))

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
                    suggested_price=round(price, 2),
                    suggested_size=suggested_size,
                    timestamp=time.time(),
                    reason=(
                        f"SPOT-DIV: {consensus_dir} ({consensus_score:+.2f}) "
                        f"{confirming}/{len(analyses)} TFs | "
                        f"spot=${spot_price:,.0f} vs target=${target_price:,.0f} | "
                        f"edge={edge:.3f}"
                    ),
                    strategy="spot_divergence",
                )

                all_signals.append(signal)
                logger.info(f"  SIGNAL: {trade_side} {token_side} | "
                             f"edge={edge:.3f} conf={confidence:.2f} | "
                             f"\"{market.question[:50]}...\"")

        all_signals.sort(key=lambda s: s.edge * s.confidence, reverse=True)
        return all_signals
