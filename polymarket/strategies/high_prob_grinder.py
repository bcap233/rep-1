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
import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from . import BaseStrategy, register_strategy
from ..config import ASSETS, HIGH_PROB_GRINDER, EVENT_CATEGORIES, MARKET_MAKER
from ..exchanges import get_composite_price
from ..polymarket_client import PolymarketClient, Market, OrderBook
from ..signals import Signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Probability model for BTC Up/Down markets
# ---------------------------------------------------------------------------
# A "BTC Up or Down" market resolves to "Up" if BTC closes above its
# price at market open, "Down" otherwise. This is equivalent to a
# digital (binary) option.
#
# Model: random walk with drift=0 (conservative), volatility σ per minute.
#   P(BTC stays above reference | currently X% above, T min left)
#     = Φ(X / (σ√T))
#   where Φ is the standard normal CDF.
#
# This gives us a MODEL-BASED true probability to compare against the
# market's implied probability (best ask). The edge is:
#   edge = model_prob - implied_prob
# We only trade when edge > fee + minimum threshold.

def _normal_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _model_up_probability(
    spot: float,
    reference: float,
    mins_left: float,
    vol_per_min: float,
) -> float:
    """Compute model probability of BTC finishing above reference.

    Args:
        spot: Current BTC price
        reference: BTC price at market window open
        mins_left: Minutes until market closes
        vol_per_min: Estimated BTC per-minute return volatility (e.g. 0.0004)

    Returns:
        Probability [0.01, 0.99] that BTC finishes above reference.
    """
    if mins_left <= 0.1:
        # Basically resolved — if above, it's up
        return 0.99 if spot > reference else 0.01
    if reference <= 0:
        return 0.50

    delta_pct = (spot - reference) / reference  # e.g. +0.003 = up 0.3%
    sigma_t = vol_per_min * math.sqrt(mins_left)

    if sigma_t < 1e-8:
        return 0.99 if delta_pct > 0 else 0.01

    z = delta_pct / sigma_t
    # Clamp to avoid extreme tails
    z = max(-4.0, min(4.0, z))
    return max(0.01, min(0.99, _normal_cdf(z)))


def _parse_market_start_time(question: str) -> Optional[datetime]:
    """Parse the START time of an Up/Down market from its question text.

    Example: "Bitcoin Up or Down - February 15, 12:15AM-12:30AM ET"
    Returns: datetime(2026, 2, 15, 5, 15, tzinfo=UTC)  (12:15 AM ET = 5:15 UTC)
    """
    m = re.search(
        r'(\w+ \d+),?\s*(\d{1,2}):(\d{2})(am|pm)\s*-\s*\d{1,2}:\d{2}(?:am|pm)\s*et',
        question.lower(),
    )
    if not m:
        return None

    date_str = m.group(1)  # "february 15"
    h, mi, p = int(m.group(2)), int(m.group(3)), m.group(4)

    if p == 'pm' and h != 12:
        h += 12
    if p == 'am' and h == 12:
        h = 0

    now = datetime.now(timezone.utc)
    try:
        start_dt = datetime.strptime(
            f"{date_str} {now.year} {h:02d}:{mi:02d}",
            "%B %d %Y %H:%M",
        ).replace(tzinfo=timezone.utc)
        # ET → UTC
        from ..execution import _et_to_utc_offset_hours
        start_dt = start_dt + timedelta(hours=_et_to_utc_offset_hours())
    except ValueError:
        return None

    return start_dt


# Default BTC per-minute return volatility.
# BTC 5-minute vol is typically 0.05-0.15%, so per-minute ≈ 0.03-0.07%.
# We use a moderate estimate; will be refined with live data.
_DEFAULT_VOL_PER_MIN = 0.0004  # 0.04% per minute


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
    # Locks are riskless — size them bigger than directional.
    "lock_size_usdc": 50.0,
    # Up/Down directional: lower threshold than event markets since
    # 5m/15m markets have momentum persistence and fast resolution.
    "updown_min_probability": 0.78,
    "updown_min_ask_depth": 30,
    # Momentum entry: if the midpoint moves this much over recent
    # scans, enter at a LOWER threshold. Books are thin and move fast;
    # catching a trend early beats waiting for 78%.
    "momentum_min_probability": 0.62,  # Enter trending side at 62%+
    "momentum_min_delta": 0.05,        # Need 5%+ midpoint shift
    "momentum_min_scans": 2,           # Over at least 2 scans
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
        # condition_id → {side, entry_price, other_token_id, other_side, ...}
        self._open_legs: dict[str, dict] = {}
        # Internal throttle: Gamma API scan is heavy (200+ markets), but
        # Up/Down scan is light (3-5 markets). Run Up/Down every cycle,
        # Gamma only every 5 min.
        self._last_gamma_scan: float = 0.0
        self._gamma_interval: float = 300.0  # 5 minutes
        # Scan-to-scan midpoint history for Up/Down markets.
        # condition_id → list of (timestamp, up_mid) — last N scans.
        # Used to detect momentum: if the mid moves 5%+ in one
        # direction over 2-3 scans, enter even at lower thresholds.
        self._mid_history: dict[str, list[tuple[float, float]]] = {}
        self._mid_history_max = 6  # Keep last ~60s of scans
        # Reference price tracker: condition_id → (start_time_utc, spot_at_start).
        # Records spot when we first see each market window.
        # Used by the probability model: P(up) = Φ(delta / (σ√T)).
        self._reference_prices: dict[str, tuple[datetime, float]] = {}
        # Per-asset volatility tracking.
        # asset → { "history": [(ts, price), ...], "vol": float }
        self._asset_vol: dict[str, dict] = {}
        # Which assets have Up/Down markets enabled
        self._updown_assets = [
            a for a, cfg in ASSETS.items() if cfg.get("updown_enabled")
        ]

    def scan(self, assets: Optional[list[str]] = None) -> list[Signal]:
        """
        Scan all active markets for high-probability opportunities.

        Two market sources, at different speeds:
        - FAST (every cycle): BTC Up/Down 5m/15m — lock + directional
        - SLOW (every 5 min): Gamma API event markets — directional only

        Up/Down runs every cycle because locks need fast reaction time.
        A 5m market resolves in 5 min — if we wait 5 min between scans,
        we miss the window to lock.
        """
        signals = []

        # FAST PATH: Up/Down markets — always run (light: 3-5 API calls)
        updown_signals = self._scan_updown_markets()
        signals.extend(updown_signals)

        # SLOW PATH: Gamma event markets — throttle internally
        now = time.time()
        if now - self._last_gamma_scan < self._gamma_interval:
            if signals:
                signals.sort(key=lambda s: s.edge, reverse=True)
            return signals
        self._last_gamma_scan = now

        logger.info(f"[GRINDER] Gamma scan for >{self.cfg['min_probability']*100:.0f}% events...")

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

    def _update_vol_estimate(self, asset: str, spot: float) -> None:
        """Update running per-minute volatility estimate for an asset."""
        now = time.time()
        if asset not in self._asset_vol:
            self._asset_vol[asset] = {"history": [], "vol": _DEFAULT_VOL_PER_MIN}

        av = self._asset_vol[asset]
        av["history"].append((now, spot))
        # Keep last 30 data points (~5 min at 10s intervals)
        av["history"] = av["history"][-30:]

        if len(av["history"]) < 3:
            return

        # Compute per-interval returns, then scale to per-minute
        returns = []
        for i in range(1, len(av["history"])):
            t0, p0 = av["history"][i - 1]
            t1, p1 = av["history"][i]
            dt_min = (t1 - t0) / 60.0
            if dt_min > 0 and p0 > 0:
                ret = (p1 - p0) / p0
                returns.append(ret / math.sqrt(max(dt_min, 0.1)))

        if len(returns) >= 2:
            mean_r = sum(returns) / len(returns)
            var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
            estimated = math.sqrt(var)
            av["vol"] = max(0.0001, 0.5 * estimated + 0.5 * _DEFAULT_VOL_PER_MIN)

    def _get_vol(self, asset: str) -> float:
        """Get current per-minute volatility estimate for an asset."""
        return self._asset_vol.get(asset, {}).get("vol", _DEFAULT_VOL_PER_MIN)

    def _scan_updown_markets(self) -> list[Signal]:
        """Scan crypto Up/Down 5m/15m markets for locks and model entries.

        Scans all updown-enabled assets (BTC, ETH, SOL).

        Priority order per market:
        1. IMMEDIATE LOCK: Up_ask + Down_ask < $0.98 → buy both
        2. DELAYED LOCK: We hold one side, other side now cheap → lock
        3. MODEL DIRECTIONAL: model_prob > market_prob + fees → buy

        The model: P(up) = Φ((spot - ref) / (σ√T))
        """
        from .market_maker import _parse_market_duration, _minutes_to_market_end

        signals = []
        durations = MARKET_MAKER.get("durations", ["5m", "15m"])
        min_minutes = MARKET_MAKER.get("min_minutes_to_expiry", 2)
        fee_rate = self.cfg["lock_fee_rate"]

        self._expire_stale_legs()

        # Fetch spot prices for all enabled assets
        asset_spots: dict[str, float] = {}
        for asset in self._updown_assets:
            composite = get_composite_price(asset)
            if composite and composite.price > 0:
                asset_spots[asset] = composite.price
                self._update_vol_estimate(asset, composite.price)

        # Discover markets across all enabled assets
        all_markets: list[tuple[str, "Market"]] = []  # (asset, market)
        for asset in self._updown_assets:
            markets = self.client.find_updown_markets(asset, durations)
            for m in markets:
                all_markets.append((asset, m))

        if not all_markets:
            return []

        scanned = 0
        locks_found = 0
        directional_found = 0

        for asset, market in all_markets:
            spot = asset_spots.get(asset, 0.0)
            vol_per_min = self._get_vol(asset)
            q = market.question
            duration = _parse_market_duration(q)
            if duration not in durations:
                continue

            mins_left = _minutes_to_market_end(q)
            if mins_left is None or mins_left < min_minutes:
                continue

            if len(market.token_ids) < 2:
                continue

            cond = market.condition_id

            # Record spot as reference price when we first see this market.
            # If the market just started (we catch it within the first scan),
            # this is very close to the true reference price.
            if cond not in self._reference_prices and spot > 0:
                start_time = _parse_market_start_time(q)
                self._reference_prices[cond] = (
                    start_time or datetime.now(timezone.utc),
                    spot,
                )

            up_token = market.token_ids[0]
            down_token = market.token_ids[1]

            up_book = self.client.get_order_book(up_token)
            down_book = self.client.get_order_book(down_token)
            if not up_book or not down_book:
                continue
            if not up_book.asks or not down_book.asks:
                continue
            scanned += 1

            up_ask = up_book.best_ask
            down_ask = down_book.best_ask
            up_mid = up_book.midpoint

            # Record midpoint for momentum tracking
            now_ts = time.time()
            if cond not in self._mid_history:
                self._mid_history[cond] = []
            self._mid_history[cond].append((now_ts, up_mid))
            # Trim old entries
            self._mid_history[cond] = self._mid_history[cond][-self._mid_history_max:]

            # Compute scan-to-scan momentum (delta from oldest to newest)
            history = self._mid_history[cond]
            mid_delta = 0.0
            mid_direction = "flat"
            if len(history) >= self.cfg["momentum_min_scans"]:
                mid_delta = history[-1][1] - history[0][1]
                if mid_delta > 0.01:
                    mid_direction = "up"
                elif mid_delta < -0.01:
                    mid_direction = "down"

            # --- MODE 1: IMMEDIATE LOCK ---
            # Both sides available now, total < threshold → riskless profit.
            total_cost = up_ask + down_ask
            # Fee: Polymarket charges on winnings of the winning side.
            # Worst case = fee on the cheaper side's full payout.
            imm_fee = fee_rate * max(1.0 - up_ask, 1.0 - down_ask)
            imm_profit = 1.0 - total_cost - imm_fee

            if (total_cost <= self.cfg["lock_max_total"]
                    and imm_profit >= self.cfg["lock_min_profit"]):
                lock_sig = self._make_lock_signal(
                    asset, market, duration, mins_left,
                    up_token, up_ask, up_book,
                    down_token, down_ask, down_book,
                    imm_profit, total_cost, "IMMEDIATE",
                )
                if lock_sig:
                    signals.append(lock_sig)
                    locks_found += 1
                    continue  # Lock takes priority

            # --- MODE 2: DELAYED LOCK ---
            # We hold one side from a previous cycle. Is the other side
            # cheap enough now to lock in profit?
            if cond in self._open_legs:
                leg = self._open_legs[cond]
                other_ask = down_ask if leg["side"] == "Up" else up_ask
                other_book = down_book if leg["side"] == "Up" else up_book

                lock_total = leg["entry_price"] + other_ask
                # Fee computed on actual lock prices, not current market
                lock_winning = max(1.0 - leg["entry_price"], 1.0 - other_ask)
                lock_fee = fee_rate * lock_winning
                lock_profit = 1.0 - lock_total - lock_fee

                if (lock_total <= self.cfg["lock_max_total"]
                        and lock_profit >= self.cfg["lock_min_profit"]):
                    other_depth = sum(s for _, s in other_book.asks[:3])
                    min_depth = self.cfg["updown_min_ask_depth"]

                    if other_depth >= min_depth:
                        # Match original leg size for full hedge
                        size = leg["size"]
                        # But cap at available depth
                        size = min(size, int(other_depth))
                        if size >= 1:
                            signal = Signal(
                                asset=asset,
                                market=market,
                                token_id=leg["other_token_id"],
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
                                    f"LOCK-DELAYED: held {leg['side']}"
                                    f"@{leg['entry_price']:.2f} + buy "
                                    f"{leg['other_side']}@{other_ask:.2f}"
                                    f"={lock_total:.2f} ({duration}, "
                                    f"{mins_left:.0f}m) | "
                                    f"locked={lock_profit:.3f}/sh "
                                    f"(${lock_profit * size:.2f})"
                                ),
                                strategy="high_prob_grinder",
                            )
                            signals.append(signal)
                            locks_found += 1
                            del self._open_legs[cond]
                            continue

            # --- MODE 3: MODEL-BASED DIRECTIONAL ---
            # Use the random walk model to compute true probability from
            # actual spot data, then compare to market price.
            #
            # P(up) = Φ((spot - ref) / (σ√T))
            #
            # Only trade when model_prob - implied_prob > fee + min_edge.
            if cond in self._seen_conditions:
                continue

            min_depth = self.cfg["updown_min_ask_depth"]

            # Get reference price for this market window
            ref_data = self._reference_prices.get(cond)
            ref_price = ref_data[1] if ref_data else 0.0

            # Compute model probability of "Up"
            if spot > 0 and ref_price > 0:
                model_up = _model_up_probability(
                    spot, ref_price, mins_left, vol_per_min,
                )
                model_down = 1.0 - model_up
            else:
                # No spot data — fall back to market midpoint
                model_up = up_mid
                model_down = 1.0 - up_mid

            delta_pct = ((spot - ref_price) / ref_price * 100
                         if ref_price > 0 else 0.0)

            # Check both sides for model edge
            sides = [
                (up_token, "Up", up_book, down_token, "Down", model_up),
                (down_token, "Down", down_book, up_token, "Up", model_down),
            ]
            # Check the model-favored side first
            sides.sort(key=lambda s: s[5], reverse=True)

            for token_id, token_side, book, other_tok, other_side, model_prob in sides:
                if book.spread > self.cfg["max_spread"]:
                    continue

                ask_depth = sum(size for _, size in book.asks[:3])
                if ask_depth < min_depth:
                    continue

                entry_price = book.best_ask
                implied_prob = entry_price

                # The edge: model says X%, market says Y%.
                # Fee is ~2% on the winning payout.
                effective_fee = fee_rate * (1.0 - entry_price)
                edge = model_prob - implied_prob - effective_fee

                if edge < 0.03:
                    # Need at least 3% model edge to overcome noise
                    continue

                profit_per_share = 1.0 - entry_price
                ev_per_share = (model_prob * profit_per_share -
                                (1 - model_prob) * entry_price)

                if ev_per_share <= 0:
                    continue

                # Edge-scaled sizing: bigger edge → bigger bet.
                # Base $20 at 3% edge, scales up to $100 at 15%+.
                base_size_usd = self.cfg["size_per_trade_usdc"]
                edge_multiple = min(5.0, edge / 0.03)  # 1x at 3%, 5x at 15%
                size_usd = base_size_usd * edge_multiple
                size_usd = min(size_usd, 100.0)  # Hard cap
                # Also cap at available depth
                size = max(1, int(size_usd / entry_price))
                size = min(size, int(ask_depth * 0.8))  # Don't eat entire book
                if size < 1:
                    continue

                # Confidence from model certainty and time
                if model_prob >= 0.90:
                    confidence = 0.90
                elif model_prob >= 0.80:
                    confidence = 0.80
                elif model_prob >= 0.70:
                    confidence = 0.70
                else:
                    confidence = 0.60

                # Time boost: less time = model is more reliable
                if mins_left <= 3:
                    confidence = min(1.0, confidence + 0.05)

                signal = Signal(
                    asset=asset,
                    market=market,
                    token_id=token_id,
                    side="BUY",
                    token_side=token_side,
                    spot_price=spot,
                    target_price=ref_price,
                    implied_prob=implied_prob,
                    fair_prob=model_prob,
                    edge=ev_per_share,
                    confidence=confidence,
                    momentum_score=mid_delta,
                    momentum_direction=mid_direction,
                    confirming_timeframes=0,
                    suggested_price=round(entry_price, 2),
                    suggested_size=size,
                    timestamp=time.time(),
                    reason=(
                        f"MODEL: {asset} {token_side}@{entry_price:.2f} "
                        f"({duration}, {mins_left:.0f}m left) | "
                        f"model={model_prob:.1%} mkt={implied_prob:.1%} "
                        f"edge={edge:+.1%} ${size_usd:.0f} | "
                        f"{asset} {delta_pct:+.3f}% vs ref "
                        f"σ={vol_per_min:.4f}/min"
                    ),
                    strategy="high_prob_grinder",
                )
                signals.append(signal)
                self._seen_conditions.add(cond)
                directional_found += 1

                # Track for delayed lock on next cycle
                self._open_legs[cond] = {
                    "side": token_side,
                    "entry_price": entry_price,
                    "other_token_id": other_tok,
                    "other_side": other_side,
                    "size": size,
                    "market": market,
                    "time": time.time(),
                }
                break

        assets_str = ",".join(self._updown_assets)
        logger.info(f"[GRINDER] Up/Down ({assets_str}): {scanned} mkts | "
                     f"{locks_found} locks + {directional_found} directional | "
                     f"{len(self._open_legs)} open legs")
        return signals

    def _make_lock_signal(
        self, asset, market, duration, mins_left,
        up_token, up_ask, up_book,
        down_token, down_ask, down_book,
        net_profit, total_cost, lock_type,
    ) -> Optional[Signal]:
        """Create a bilateral lock signal (buy both sides)."""
        up_depth = sum(s for _, s in up_book.asks[:3])
        down_depth = sum(s for _, s in down_book.asks[:3])
        min_depth = min(up_depth, down_depth)
        depth_threshold = self.cfg["updown_min_ask_depth"]

        if min_depth < depth_threshold:
            return None

        # Locks are riskless — size bigger than directional bets
        lock_usdc = self.cfg["lock_size_usdc"]
        max_price = max(up_ask, down_ask)
        size = max(1, int(lock_usdc / max_price))
        size = min(size, int(min_depth))

        return Signal(
            asset=asset,
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
                f"LOCK-{lock_type}: Up@{up_ask:.2f}+Down@{down_ask:.2f}"
                f"={total_cost:.2f} ({duration}, {mins_left:.0f}m) | "
                f"profit={net_profit:.3f}/sh (${net_profit * size:.2f})"
            ),
            strategy="high_prob_grinder",
            arb_legs=[
                {"token_id": up_token, "side": "BUY",
                 "price": up_ask, "label": "Up"},
                {"token_id": down_token, "side": "BUY",
                 "price": down_ask, "label": "Down"},
            ],
        )

    def _expire_stale_legs(self):
        """Remove open legs, stale history, and old references (>20 min)."""
        cutoff = time.time() - 20 * 60
        expired = [cid for cid, leg in self._open_legs.items()
                   if leg["time"] < cutoff]
        for cid in expired:
            del self._open_legs[cid]
            self._mid_history.pop(cid, None)
            self._reference_prices.pop(cid, None)
        if expired:
            logger.debug(f"[GRINDER] Expired {len(expired)} stale legs")
        # Also clean history/refs for markets we no longer see
        stale_hist = [cid for cid, hist in self._mid_history.items()
                      if hist and hist[-1][0] < cutoff]
        for cid in stale_hist:
            del self._mid_history[cid]
            self._reference_prices.pop(cid, None)

    def reset_seen(self):
        """Clear the seen-conditions cache (e.g., new trading session)."""
        self._seen_conditions.clear()
