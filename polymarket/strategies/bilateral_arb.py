"""
Strategy 2: Bilateral Arbitrage.

The concept:
  On Polymarket, every binary market has a YES token and a NO token.
  Exactly one of them resolves to $1.00, the other to $0.00.
  Therefore: YES + NO = $1.00 always at resolution.

  If you can BUY YES at $0.48 and BUY NO at $0.49:
    Total cost: $0.97
    Guaranteed payout: $1.00 (one side MUST win)
    Risk-free profit: $0.03 per share pair

  This also works across correlated markets:
    Market A: "BTC above $100k on March 1" — YES @ $0.55
    Market B: "BTC below $100k on March 1" — YES @ $0.42
    Total: $0.97 for guaranteed $1.00 → 3c arb

  And on multi-outcome markets (brackets):
    "BTC price on March 1"
    Outcome 1: "Above $110k" — YES @ $0.20
    Outcome 2: "$100k-$110k" — YES @ $0.30
    Outcome 3: "$90k-$100k" — YES @ $0.25
    Outcome 4: "Below $90k" — YES @ $0.18
    Total: $0.93 for guaranteed $1.00 → 7c arb

Types of bilateral arb this strategy detects:

  1. INTRA-MARKET: YES_ask + NO_ask < 1.00 within the same market
     (simplest, most common, smallest edges)

  2. CROSS-MARKET: Two separate but logically complementary markets
     where YES_A + YES_B < 1.00 (or > 1.00 for sell-side)

  3. MULTI-OUTCOME: Markets with 3+ buckets where the sum of all
     best asks < 1.00

Fees:
  Polymarket charges ~2% on winnings. So the real break-even is:
  YES_ask + NO_ask < 1.00 - (0.02 * profit)
  In practice, need the gap to be > ~2c to be profitable after fees.
"""

import logging
import time
from typing import Optional

from . import BaseStrategy, register_strategy
from ..config import ASSETS, BILATERAL_ARB
from ..polymarket_client import PolymarketClient, Market, OrderBook
from ..signals import Signal

logger = logging.getLogger(__name__)

# Default config — merged with config.py overrides at init
_DEFAULTS = {
    "min_gap_intra": 0.02,
    "min_gap_cross": 0.03,
    "min_gap_multi": 0.04,
    "fee_rate": 0.02,
    "min_liquidity_per_side": 2_000,
    "min_depth_per_side": 20,
    "max_spread_per_side": 0.05,
    "max_positions": 20,
    "size_per_leg_usdc": 50.0,
    "max_exposure_usdc": 2_000.0,
    "scan_limit": 200,
    "max_expiry_gap_hours": 24,
    "complement_keywords": {
        "above": "below",
        "over": "under",
        "higher": "lower",
        "up": "down",
        "rise": "fall",
        "exceed": "drop",
    },
}


@register_strategy
class BilateralArb(BaseStrategy):
    """
    Buy both sides of a market when the total cost < $1.00.

    Scans for three types of arb:
    1. Intra-market: YES + NO < $1.00 in the same market
    2. Cross-market: Complementary markets with combined cost < $1.00
    3. Multi-outcome: Bracket markets where all outcomes sum < $1.00

    Usage:
        python -m polymarket --strategy bilateral_arb --loop
    """

    name = "bilateral_arb"
    description = "Buy both YES+NO (or all outcomes) when total cost < $1.00"

    def __init__(self, client: PolymarketClient):
        super().__init__(client)
        # Merge: config.py overrides take precedence over defaults
        self.cfg = {**_DEFAULTS, **BILATERAL_ARB}
        self._market_cache: dict[str, Market] = {}

    def scan(self, assets: Optional[list[str]] = None) -> list[Signal]:
        """
        Scan for bilateral arbitrage opportunities.

        Checks all three arb types: intra-market, cross-market, multi-outcome.
        """
        all_signals = []

        # Fetch markets — use asset-specific search if provided,
        # otherwise broad scan
        markets = self._fetch_markets(assets)
        if not markets:
            return []

        # Type 1: Intra-market arb (YES + NO in same market)
        intra_signals = self._scan_intra_market(markets)
        all_signals.extend(intra_signals)

        # Type 2: Cross-market arb (complementary separate markets)
        cross_signals = self._scan_cross_market(markets)
        all_signals.extend(cross_signals)

        # Type 3: Multi-outcome arb (bracket markets)
        multi_signals = self._scan_multi_outcome(markets)
        all_signals.extend(multi_signals)

        logger.info(f"[BILATERAL] Found {len(intra_signals)} intra + "
                     f"{len(cross_signals)} cross + {len(multi_signals)} multi = "
                     f"{len(all_signals)} total opportunities")

        # Sort by gap (risk-free profit) descending
        all_signals.sort(key=lambda s: s.edge, reverse=True)
        return all_signals

    def _fetch_markets(self, assets: Optional[list[str]]) -> list[Market]:
        """Fetch markets to scan."""
        if assets:
            all_markets = []
            for asset in assets:
                markets = self.client.find_crypto_price_markets(asset)
                all_markets.extend(markets)
            # Also do a broad search for more coverage
            broad = self.client.search_markets("", limit=self.cfg["scan_limit"],
                                                active_only=True)
            all_markets.extend(broad)
        else:
            all_markets = self.client.search_markets("", limit=self.cfg["scan_limit"],
                                                      active_only=True)

        # Deduplicate
        seen = set()
        unique = []
        for m in all_markets:
            if m.condition_id not in seen:
                seen.add(m.condition_id)
                unique.append(m)
                self._market_cache[m.condition_id] = m

        logger.info(f"[BILATERAL] Fetched {len(unique)} unique markets")
        return unique

    # --------------------------------------------------------
    # Type 1: Intra-market arb
    # --------------------------------------------------------

    def _scan_intra_market(self, markets: list[Market]) -> list[Signal]:
        """
        Check each market for YES_ask + NO_ask < 1.00.

        This is the simplest arb: buy both sides of the same market.
        One MUST resolve to $1.00, so if total cost < $1.00, it's free money.
        """
        signals = []
        checked = 0

        for market in markets:
            if not market.yes_token_id or not market.no_token_id:
                continue

            # Only binary (2-outcome) markets for intra-market
            if len(market.outcomes) != 2:
                continue

            # Get both order books
            yes_book = self.client.get_order_book(market.yes_token_id)
            if not yes_book or not yes_book.asks:
                continue

            no_book = self.client.get_order_book(market.no_token_id)
            if not no_book or not no_book.asks:
                continue

            checked += 1

            yes_ask = yes_book.best_ask
            no_ask = no_book.best_ask
            total_cost = yes_ask + no_ask

            # The arb: if total < 1.00, buy both
            gap = 1.0 - total_cost

            # Account for fees (fee on winnings only)
            # Winner pays fee_rate on (1.00 - cost_of_winning_side)
            # Worst case fee: fee_rate * max(1 - yes_ask, 1 - no_ask)
            max_profit_side = max(1 - yes_ask, 1 - no_ask)
            fee = self.cfg["fee_rate"] * max_profit_side
            net_gap = gap - fee

            if net_gap < self.cfg["min_gap_intra"]:
                continue

            # Check depth
            yes_depth = sum(s for _, s in yes_book.asks[:3])
            no_depth = sum(s for _, s in no_book.asks[:3])
            if (yes_depth < self.cfg["min_depth_per_side"] or
                    no_depth < self.cfg["min_depth_per_side"]):
                continue

            # Check spreads
            if (yes_book.spread > self.cfg["max_spread_per_side"] or
                    no_book.spread > self.cfg["max_spread_per_side"]):
                continue

            # Calculate size (must buy equal shares of both sides)
            max_shares = int(self.cfg["size_per_leg_usdc"] / max(yes_ask, no_ask))
            max_shares = min(max_shares, int(yes_depth), int(no_depth))
            if max_shares < 1:
                continue

            # Determine asset label
            asset = self._guess_asset(market.question)

            # Create signal for the YES leg
            # (the execution engine will see this as one position;
            #  we create a paired signal)
            signal = Signal(
                asset=asset,
                market=market,
                token_id=market.yes_token_id,
                side="BUY",
                token_side="YES+NO",  # Special marker for bilateral
                spot_price=0.0,
                target_price=0.0,
                implied_prob=yes_ask,
                fair_prob=1.0,  # Guaranteed payout
                edge=net_gap,
                confidence=0.99,  # Near-certain (execution risk only)
                momentum_score=0.0,
                momentum_direction="n/a",
                confirming_timeframes=0,
                suggested_price=round(total_cost, 2),  # Total cost for the pair
                suggested_size=max_shares,
                timestamp=time.time(),
                reason=(
                    f"INTRA-ARB: YES@{yes_ask:.2f} + NO@{no_ask:.2f} = "
                    f"{total_cost:.2f} | gap={gap:.3f} net={net_gap:.3f} | "
                    f"profit=${net_gap * max_shares:.2f} on {max_shares} shares"
                ),
                strategy="bilateral_arb",
                # Store leg details for the execution engine
                arb_legs=[
                    {"token_id": market.yes_token_id, "side": "BUY",
                     "price": yes_ask, "label": "YES"},
                    {"token_id": market.no_token_id, "side": "BUY",
                     "price": no_ask, "label": "NO"},
                ],
            )
            signals.append(signal)

            logger.info(f"  INTRA-ARB: YES@{yes_ask:.2f} + NO@{no_ask:.2f} = "
                         f"{total_cost:.3f} | gap={net_gap:.3f} | "
                         f"\"{market.question[:50]}\"")

        logger.info(f"[BILATERAL] Checked {checked} intra-market, "
                     f"found {len(signals)} arbs")
        return signals

    # --------------------------------------------------------
    # Type 2: Cross-market arb
    # --------------------------------------------------------

    def _scan_cross_market(self, markets: list[Market]) -> list[Signal]:
        """
        Find pairs of separate markets that are logically complementary.

        E.g., Market A: "BTC above $100k by March 1"
              Market B: "BTC below $100k by March 1"
        If YES_A + YES_B < 1.00, buy both YES tokens.
        """
        signals = []

        # Group markets by potential complement pairs
        # Look for "above X" / "below X" pairs with similar targets
        from ..signals import extract_target_price

        # Build index: (asset_tag, target_price, direction) → market
        market_index: dict[tuple[str, float, str], list[tuple[Market, OrderBook]]] = {}

        for market in markets:
            if not market.yes_token_id:
                continue

            target, direction = extract_target_price(market.question)
            if target is None:
                continue

            asset = self._guess_asset(market.question)

            book = self.client.get_order_book(market.yes_token_id)
            if not book or not book.asks:
                continue

            # Round target to group similar prices
            key = (asset, round(target, -2), direction)  # Round to nearest 100
            if key not in market_index:
                market_index[key] = []
            market_index[key].append((market, book))

        # Now find complement pairs
        for (asset, target, direction), entries in market_index.items():
            # Find the opposite direction at the same target
            opposite = "below" if direction == "above" else "above"
            comp_key = (asset, target, opposite)

            if comp_key not in market_index:
                continue

            comp_entries = market_index[comp_key]

            # Try all combinations
            for market_a, book_a in entries:
                for market_b, book_b in comp_entries:
                    if market_a.condition_id == market_b.condition_id:
                        continue

                    yes_a_ask = book_a.best_ask
                    yes_b_ask = book_b.best_ask
                    total = yes_a_ask + yes_b_ask

                    # For cross-market, the two events must be truly
                    # complementary (mutually exclusive and exhaustive).
                    # "Above X" + "Below X" covers everything except "exactly X",
                    # which has ~0 probability for continuous prices.
                    gap = 1.0 - total

                    # Higher fee estimate for cross (slightly more risk)
                    max_profit = max(1 - yes_a_ask, 1 - yes_b_ask)
                    fee = self.cfg["fee_rate"] * max_profit
                    net_gap = gap - fee

                    if net_gap < self.cfg["min_gap_cross"]:
                        continue

                    # Check depth on both sides
                    depth_a = sum(s for _, s in book_a.asks[:3])
                    depth_b = sum(s for _, s in book_b.asks[:3])
                    if (depth_a < self.cfg["min_depth_per_side"] or
                            depth_b < self.cfg["min_depth_per_side"]):
                        continue

                    max_shares = int(self.cfg["size_per_leg_usdc"] / max(yes_a_ask, yes_b_ask))
                    max_shares = min(max_shares, int(depth_a), int(depth_b))
                    if max_shares < 1:
                        continue

                    signal = Signal(
                        asset=asset,
                        market=market_a,  # Primary market
                        token_id=market_a.yes_token_id,
                        side="BUY",
                        token_side="YES+YES_CROSS",
                        spot_price=0.0,
                        target_price=target,
                        implied_prob=yes_a_ask,
                        fair_prob=1.0,
                        edge=net_gap,
                        confidence=0.90,  # Slightly less certain (correlation risk)
                        momentum_score=0.0,
                        momentum_direction="n/a",
                        confirming_timeframes=0,
                        suggested_price=round(total, 2),
                        suggested_size=max_shares,
                        timestamp=time.time(),
                        reason=(
                            f"CROSS-ARB: \"{market_a.question[:30]}\" YES@{yes_a_ask:.2f} + "
                            f"\"{market_b.question[:30]}\" YES@{yes_b_ask:.2f} = "
                            f"{total:.2f} | net gap={net_gap:.3f}"
                        ),
                        strategy="bilateral_arb",
                        arb_legs=[
                            {"token_id": market_a.yes_token_id, "side": "BUY",
                             "price": yes_a_ask, "label": f"YES ({market_a.question[:30]})"},
                            {"token_id": market_b.yes_token_id, "side": "BUY",
                             "price": yes_b_ask, "label": f"YES ({market_b.question[:30]})"},
                        ],
                    )
                    signals.append(signal)

                    logger.info(f"  CROSS-ARB: {asset} target=${target:,.0f} | "
                                 f"above@{yes_a_ask:.2f} + below@{yes_b_ask:.2f} = "
                                 f"{total:.3f} | gap={net_gap:.3f}")

        logger.info(f"[BILATERAL] Found {len(signals)} cross-market arbs")
        return signals

    # --------------------------------------------------------
    # Type 3: Multi-outcome arb
    # --------------------------------------------------------

    def _scan_multi_outcome(self, markets: list[Market]) -> list[Signal]:
        """
        Find multi-outcome (bracket) markets where the sum of all
        outcome asks < 1.00.

        E.g., "BTC price on March 1" with outcomes:
          >$110k @ 0.20, $100k-$110k @ 0.30, $90k-$100k @ 0.25, <$90k @ 0.18
          Total: 0.93 → buy all four → guaranteed 7c profit
        """
        signals = []

        for market in markets:
            # Multi-outcome markets have 3+ outcomes
            if len(market.outcomes) < 3 or len(market.token_ids) < 3:
                continue

            # Fetch order books for all outcome tokens
            legs = []
            total_cost = 0.0
            all_have_depth = True

            for i, token_id in enumerate(market.token_ids):
                book = self.client.get_order_book(token_id)
                if not book or not book.asks:
                    all_have_depth = False
                    break

                ask = book.best_ask
                depth = sum(s for _, s in book.asks[:3])

                if depth < self.cfg["min_depth_per_side"]:
                    all_have_depth = False
                    break

                outcome_label = market.outcomes[i] if i < len(market.outcomes) else f"Outcome {i}"
                legs.append({
                    "token_id": token_id,
                    "side": "BUY",
                    "price": ask,
                    "depth": depth,
                    "label": outcome_label,
                })
                total_cost += ask

            if not all_have_depth or not legs:
                continue

            gap = 1.0 - total_cost

            # Fee on multi-outcome: winning side pays fee
            max_profit = max(1 - leg["price"] for leg in legs)
            fee = self.cfg["fee_rate"] * max_profit
            net_gap = gap - fee

            if net_gap < self.cfg["min_gap_multi"]:
                continue

            # Size: limited by the shallowest leg
            max_price = max(leg["price"] for leg in legs)
            min_depth = min(leg["depth"] for leg in legs)
            max_shares = int(self.cfg["size_per_leg_usdc"] / max_price)
            max_shares = min(max_shares, int(min_depth))
            if max_shares < 1:
                continue

            asset = self._guess_asset(market.question)

            leg_desc = " + ".join(f"{l['label'][:15]}@{l['price']:.2f}" for l in legs)

            signal = Signal(
                asset=asset,
                market=market,
                token_id=market.token_ids[0],  # Primary token
                side="BUY",
                token_side=f"ALL_{len(legs)}_OUTCOMES",
                spot_price=0.0,
                target_price=0.0,
                implied_prob=total_cost,
                fair_prob=1.0,
                edge=net_gap,
                confidence=0.95,
                momentum_score=0.0,
                momentum_direction="n/a",
                confirming_timeframes=0,
                suggested_price=round(total_cost, 2),
                suggested_size=max_shares,
                timestamp=time.time(),
                reason=(
                    f"MULTI-ARB ({len(legs)} outcomes): {leg_desc} = "
                    f"{total_cost:.2f} | net gap={net_gap:.3f} | "
                    f"profit=${net_gap * max_shares:.2f}"
                ),
                strategy="bilateral_arb",
                arb_legs=legs,
            )
            signals.append(signal)

            logger.info(f"  MULTI-ARB: {len(legs)} outcomes | "
                         f"total={total_cost:.3f} | gap={net_gap:.3f} | "
                         f"\"{market.question[:50]}\"")

        logger.info(f"[BILATERAL] Found {len(signals)} multi-outcome arbs")
        return signals

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _guess_asset(self, question: str) -> str:
        """Best-effort asset tagging from market question text."""
        q = question.lower()
        for asset, cfg in ASSETS.items():
            tags = cfg.get("polymarket_tags", [])
            if any(t in q for t in tags):
                return asset
        return "MISC"
