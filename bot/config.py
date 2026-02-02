"""
V7 Rotation Bot Configuration.

Defines pairs, strategy parameters, and integration settings.
Copy this to bot_config.py and fill in your credentials.
"""

# ============================================================
# STRATEGY PARAMETERS (V7)
# ============================================================

STRATEGY = {
    "sma_period": 50,
    "band_pct": 5.0,
    "debounce_days": 3,
    "rv_window": 20,
    "income_weeks": 4,
    "min_income_rv_ratio": 1.5,
}

# Minimum trading days of history needed for SMA + RV to be valid
MIN_HISTORY_DAYS = 120

# ============================================================
# TRADING PAIRS
# ============================================================
# Each pair: bull ETF (income in uptrend) / bear ETF (income in downtrend)
# The signal is generated from the bull ETF's price vs its SMA.

PAIRS = [
    {
        "name": "MSTR",
        "bull": "MSTY",
        "bear": "WNTR",
        "enabled": True,
    },
    {
        "name": "NVIDIA",
        "bull": "NVDY",
        "bear": "DIPS",
        "enabled": True,
    },
    {
        "name": "Coinbase",
        "bull": "CONY",
        "bear": "FIAT",
        "enabled": True,
    },
    {
        "name": "Tesla",
        "bull": "TSLY",
        "bear": "CRSH",
        "enabled": True,
    },
]

# ============================================================
# BROKERAGE (Alpaca)
# ============================================================
# Paper trading: https://app.alpaca.markets/paper/dashboard/overview
# Live trading: https://app.alpaca.markets/

ALPACA = {
    "api_key": "",
    "secret_key": "",
    "base_url": "https://paper-api.alpaca.markets",  # Paper trading
    # "base_url": "https://api.alpaca.markets",       # Live trading
}

# ============================================================
# NOTIFICATIONS
# ============================================================
# Webhook URL for alerts (Discord, Slack, or any webhook endpoint).
# Leave empty to disable.

NOTIFICATIONS = {
    "webhook_url": "",
    # Discord example: "https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN"
    # Slack example:   "https://hooks.slack.com/services/YOUR/HOOK/URL"
}

# ============================================================
# DATA
# ============================================================

DATA = {
    # How far back to fetch price history (trading days)
    "lookback_days": 200,
    # Where to cache state between runs
    "state_file": "bot/state.json",
}

# ============================================================
# EXECUTION
# ============================================================

EXECUTION = {
    # "signal_only" = just compute signals and log/notify (no trades)
    # "live" = actually place orders via brokerage
    "mode": "signal_only",

    # For live mode: what fraction of the pair's allocation to trade
    # 1.0 = 100% of allocated capital per pair
    "position_size": 1.0,
}
