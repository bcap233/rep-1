"""
Polymarket Arbitrage Bot Configuration.

Credentials are loaded from environment variables or a .env file.
Run `python -m polymarket --setup` to generate credentials.
"""

import os

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# ============================================================
# POLYMARKET API
# ============================================================
# CLOB API for order placement and market data.
# Credentials come from .env file (created by --setup).
# The bot uses the CLOB (Central Limit Order Book) API.

POLYMARKET = {
    # CLOB API endpoint
    "clob_url": _env("POLYMARKET_CLOB_URL", "https://clob.polymarket.com"),
    # Gamma API for market discovery
    "gamma_url": _env("POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com"),
    # API key (derived from wallet by --setup)
    "api_key": _env("POLYMARKET_API_KEY"),
    # API secret
    "api_secret": _env("POLYMARKET_API_SECRET"),
    # API passphrase
    "api_passphrase": _env("POLYMARKET_API_PASSPHRASE"),
    # Polygon chain ID (137 = mainnet, 80001 = mumbai testnet)
    "chain_id": int(_env("POLYMARKET_CHAIN_ID", "137")),
    # Wallet private key for signing (hex, without 0x prefix)
    "private_key": _env("POLYMARKET_PRIVATE_KEY"),
    # Funder address (if using a separate funding wallet)
    "funder": _env("POLYMARKET_FUNDER"),
}

# ============================================================
# EXCHANGE DATA SOURCES
# ============================================================
# Spot price feeds. These use public endpoints (no auth needed).

EXCHANGES = {
    "binance": {
        "enabled": True,
        "base_url": "https://api.binance.com",
        "weight": 0.50,  # Weighting in composite price
    },
    "coinbase": {
        "enabled": True,
        "base_url": "https://api.coinbase.com",
        "weight": 0.30,
    },
    "kraken": {
        "enabled": True,
        "base_url": "https://api.kraken.com",
        "weight": 0.20,
    },
}

# ============================================================
# EVENT CATEGORIES
# ============================================================
# Broad categories for market discovery. The grinder and bilateral
# arb strategies use these to search beyond crypto price markets.

EVENT_CATEGORIES = {
    "politics": {
        "queries": ["trump", "democrat", "republican", "president", "congress",
                     "senate", "governor", "election", "vote"],
        "label": "POL",
    },
    "sports": {
        "queries": ["nba", "nfl", "mlb", "nhl", "ufc", "soccer",
                     "finals", "championship", "super bowl", "world series"],
        "label": "SPORT",
    },
    "economics": {
        "queries": ["fed", "inflation", "gdp", "interest rate", "tariff",
                     "recession", "unemployment", "cpi", "revenue", "budget",
                     "deficit", "treasury"],
        "label": "ECON",
    },
    "tech": {
        "queries": ["ai", "openai", "google", "apple", "meta", "nvidia",
                     "tesla", "spacex", "tiktok"],
        "label": "TECH",
    },
    "world": {
        "queries": ["ukraine", "russia", "china", "israel", "gaza", "war",
                     "nato", "ceasefire"],
        "label": "WORLD",
    },
    "culture": {
        "queries": ["elon", "oscar", "grammy", "netflix", "gta", "viral"],
        "label": "POP",
    },
    "crypto": {
        "queries": ["bitcoin", "btc", "ethereum", "eth", "solana", "crypto"],
        "label": "CRYPTO",
    },
}

# ============================================================
# TRACKED ASSETS
# ============================================================
# Map of assets to their exchange symbols and Polymarket search terms.
# Used by spot_divergence strategy for crypto price plays.

ASSETS = {
    "BTC": {
        "binance": "BTCUSDT",
        "coinbase": "BTC-USD",
        "kraken": "XXBTZUSD",
        "polymarket_tags": ["bitcoin", "btc"],
        "updown_enabled": True,
    },
    "ETH": {
        "binance": "ETHUSDT",
        "coinbase": "ETH-USD",
        "kraken": "XETHZUSD",
        "polymarket_tags": ["ethereum", "eth"],
    },
    "SOL": {
        "binance": "SOLUSDT",
        "coinbase": "SOL-USD",
        "kraken": "SOLUSD",
        "polymarket_tags": ["solana", "sol"],
    },
}

# ============================================================
# CHART TIMEFRAMES
# ============================================================
# Analysis windows in minutes. The bot builds candles at each
# interval and computes momentum indicators.

TIMEFRAMES = [5, 10, 15]  # minutes

# ============================================================
# BTC UP/DOWN SHORT-DURATION MARKETS
# ============================================================
# These markets follow a deterministic slug pattern on Polymarket:
#   btc-updown-{duration}-{aligned_unix_timestamp}
# They are not discoverable via the Gamma search API.

BTC_UPDOWN_DURATIONS = {
    "5m": {
        "interval_seconds": 300,
        "offset_seconds": 0,
        "label": "5-Minute",
    },
    "15m": {
        "interval_seconds": 900,
        "offset_seconds": 0,
        "label": "15-Minute",
    },
    "4h": {
        "interval_seconds": 14400,
        "offset_seconds": 3600,    # 4h blocks start at :00 ET (01:00 UTC)
        "label": "4-Hour",
    },
}

# ============================================================
# STRATEGY PARAMETERS
# ============================================================

STRATEGY = {
    # Minimum edge (probability points) to trigger a trade.
    # E.g., 0.05 = 5 cents on a $1 contract.
    "min_edge": 0.05,

    # Momentum thresholds (rate of change over the timeframe).
    # If spot momentum exceeds this and Polymarket hasn't caught up, signal fires.
    "momentum_threshold_pct": 0.3,  # 0.3% move in the timeframe

    # VWAP deviation threshold (% from VWAP).
    # Strong moves away from VWAP increase conviction.
    "vwap_deviation_pct": 0.2,

    # RSI thresholds (14-period on the timeframe candles).
    "rsi_overbought": 70,
    "rsi_oversold": 30,

    # Minimum number of timeframes confirming the signal.
    # If 2 out of 3 timeframes (5/10/15) agree, the signal is stronger.
    "min_confirming_timeframes": 2,

    # Stale market filter: skip if Polymarket spread > this.
    "max_spread": 0.08,

    # Confidence weighting: how much each indicator contributes.
    "weights": {
        "momentum": 0.35,
        "vwap_dev": 0.25,
        "rsi": 0.20,
        "volume": 0.10,
        "multi_tf": 0.10,
    },
}

# ============================================================
# RISK MANAGEMENT
# ============================================================

RISK = {
    # Max position size per trade (in USDC)
    "max_position_usdc": 250.0,

    # Max total exposure across all open positions.
    # When running multiple strategies (grinder=2k, bilateral=2k, spot=500),
    # set this high enough to accommodate them all.
    "max_total_exposure_usdc": 5_000.0,

    # Max number of concurrent open positions.
    # Grinder alone can hold 50, bilateral 20, so set to accommodate all.
    "max_open_positions": 75,

    # Stop loss: exit if contract moves against us by this much
    "stop_loss": 0.18,  # 18 cents — wider to survive short-duration swings

    # Take profit: exit if contract moves in our favor by this much
    "take_profit": 0.20,  # 20 cents

    # Max holding time (minutes). Close position after this regardless.
    "max_hold_minutes": 120,

    # Cooldown after a loss (seconds). Don't trade the same market.
    "loss_cooldown_seconds": 30,

    # Daily loss limit (USDC). Stop trading for the day if hit.
    "daily_loss_limit_usdc": 200.0,
}

# ============================================================
# EXECUTION
# ============================================================

EXECUTION = {
    # "paper" = log signals only, no real trades
    # "live" = place real orders on Polymarket
    "mode": "paper",

    # Order type: "limit" or "market" (market = aggressive limit at best bid/ask)
    "order_type": "limit",

    # For limit orders: offset from fair value (in probability points).
    # Positive = more conservative (less likely to fill, better price).
    "limit_offset": 0.01,

    # Poll interval (seconds) between each cycle
    "poll_interval": 10,

    # How often to refresh Polymarket market list (seconds)
    "market_refresh_interval": 300,
}

# ============================================================
# STRATEGY: HIGH PROBABILITY GRINDER
# ============================================================
# Buy high-probability tokens (YES or NO) at scale.
# Profit per win is small (5-15c) but win rate is very high.

HIGH_PROB_GRINDER = {
    # Watch-only mode: scan and log signals, but don't execute trades.
    # Signal count/quality feeds into dynamic MM capital scaling.
    "watch_only": True,
    # Buy tokens priced in this range (checks both YES and NO sides)
    "min_probability": 0.85,
    "max_probability": 0.97,
    # Sweet spot: best risk/reward zone
    "sweet_spot_low": 0.88,
    "sweet_spot_high": 0.95,
    # Market quality filters
    "min_liquidity": 5_000,
    "min_volume": 10_000,
    "max_spread": 0.04,
    "min_ask_depth": 50,
    # Fixed size per trade (USDC). Many small bets.
    "size_per_trade_usdc": 20.0,
    # Strategy-level position limits
    "max_positions": 50,
    "max_exposure_usdc": 2_000.0,
    # Time-to-expiry window — keep capital turning over fast
    "min_hours_to_expiry": 1,  # Skip markets resolving in < 1 hour (price already locked)
    "max_days_to_expiry": 7,  # Only short-dated markets (resolves within a week)
    # Number of markets to scan per cycle (per query)
    "scan_limit": 200,
    # Run category-specific searches alongside the broad scan
    "category_scan": True,
    # Minimum positive EV per share
    "min_edge": 0.01,
}

# ============================================================
# STRATEGY: BILATERAL ARBITRAGE
# ============================================================
# Buy both sides (YES + NO, or all outcomes) when the total
# cost is less than $1.00, locking in risk-free profit.

BILATERAL_ARB = {
    # Watch-only mode: scan and log signals, but don't execute trades.
    # Signal count/quality feeds into dynamic MM capital scaling.
    "watch_only": True,
    # Minimum gap after fees to execute each arb type
    "min_gap_intra": 0.02,   # Same-market YES+NO arb
    "min_gap_cross": 0.03,   # Cross-market complementary arb
    "min_gap_multi": 0.04,   # Multi-outcome bracket arb
    # Polymarket fee rate on winnings
    "fee_rate": 0.02,
    # Per-side quality requirements
    "min_liquidity_per_side": 2_000,
    "min_depth_per_side": 20,
    "max_spread_per_side": 0.05,
    # Position limits
    "max_positions": 20,
    "size_per_leg_usdc": 50.0,
    "max_exposure_usdc": 2_000.0,
    # Scan size
    "scan_limit": 200,
    # Max time gap between complementary market expirations
    "max_expiry_gap_hours": 24,
}

# ============================================================
# STRATEGY: MARKET MAKER
# ============================================================
# Post bids and asks on both sides of BTC Up/Down markets.
# Profit comes from the spread, not from predicting direction.
# Inventory risk is managed by skewing quotes toward neutral.

MARKET_MAKER = {
    # Which durations to make markets on
    "durations": ["5m", "15m", "4h"],  # Include 5m for more opportunities
    # Spread we want to capture (half-spread on each side)
    "half_spread": 0.02,         # 2c on each side = 4c total spread
    # Min spread in the book before we quote
    "min_book_spread": 0.01,
    # Size per side (USDC)
    "size_per_side_usdc": 100.0,
    # Max inventory imbalance before we stop quoting one side
    # (net shares long or short on Up vs Down)
    "max_inventory_shares": 250,
    # Inventory skew: shift quotes to offload inventory
    # At max inventory, shift by this many cents away from the heavy side
    "inventory_skew_max": 0.03,
    # Use spot momentum to bias fair value (lean into trends)
    "momentum_bias_weight": 0.05,
    # Only quote if we're at least this far from market expiry
    "min_minutes_to_expiry": 2,
    # Max exposure across all MM positions
    "max_exposure_usdc": 2_000.0,
    # Max concurrent MM positions (each market = 1 position tracking both sides)
    "max_positions": 30,
}

# ============================================================
# NOTIFICATIONS
# ============================================================

NOTIFICATIONS = {
    "webhook_url": "",
    # Discord example: "https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN"
}

# ============================================================
# DATA / STATE
# ============================================================

DATA = {
    "state_file": "polymarket/state.json",
    "trade_log": "polymarket/trades.jsonl",
}
