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
# TRACKED ASSETS
# ============================================================
# Map of assets to their exchange symbols and Polymarket search terms.

ASSETS = {
    "BTC": {
        "binance": "BTCUSDT",
        "coinbase": "BTC-USD",
        "kraken": "XXBTZUSD",
        "polymarket_tags": ["bitcoin", "btc"],
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
    "max_position_usdc": 100.0,

    # Max total exposure across all open positions.
    # When running multiple strategies (grinder=2k, bilateral=2k, spot=500),
    # set this high enough to accommodate them all.
    "max_total_exposure_usdc": 5_000.0,

    # Max number of concurrent open positions.
    # Grinder alone can hold 50, bilateral 20, so set to accommodate all.
    "max_open_positions": 75,

    # Stop loss: exit if contract moves against us by this much
    "stop_loss": 0.10,  # 10 cents

    # Take profit: exit if contract moves in our favor by this much
    "take_profit": 0.15,  # 15 cents

    # Max holding time (minutes). Close position after this regardless.
    "max_hold_minutes": 60,

    # Cooldown after a loss (seconds). Don't trade the same market.
    "loss_cooldown_seconds": 300,

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
# Buy YES tokens on events priced >90% at scale.
# Profit per win is small (5-10c) but win rate is very high.

HIGH_PROB_GRINDER = {
    # Only buy YES tokens priced in this range
    "min_probability": 0.90,
    "max_probability": 0.97,
    # Sweet spot: best risk/reward zone
    "sweet_spot_low": 0.91,
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
    # Time-to-expiry window
    "min_hours_to_expiry": 2,
    "max_days_to_expiry": 30,
    # Number of markets to scan per cycle
    "scan_limit": 200,
    # Minimum positive EV per share
    "min_edge": 0.01,
}

# ============================================================
# STRATEGY: BILATERAL ARBITRAGE
# ============================================================
# Buy both sides (YES + NO, or all outcomes) when the total
# cost is less than $1.00, locking in risk-free profit.

BILATERAL_ARB = {
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
