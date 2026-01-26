"""
Configuration for MSTY collapse detection analysis.
"""

# Primary instrument
TICKER = "MSTY"
UNDERLYING = "MSTR"

# SMA configuration
SMA_PERIOD = 50

# Analysis period (1 year)
LOOKBACK_DAYS = 365

# Collapse detection thresholds
# These are the key parameters we're trying to calibrate:
# At what % below the 50 SMA does a "true" collapse begin?
DISTANCE_THRESHOLDS = {
    "warning": -5,      # 5% below SMA - early warning
    "caution": -10,     # 10% below SMA - elevated risk
    "danger": -15,      # 15% below SMA - likely in collapse
    "collapse": -20,    # 20% below SMA - confirmed collapse
}

# Velocity thresholds (rate of change in SMA distance)
# How fast is price moving away from SMA?
VELOCITY_THRESHOLDS = {
    "accelerating": -2,    # Losing 2% per day relative to SMA
    "rapid": -5,           # Losing 5% per day relative to SMA
}

# Confirmation signals for "true" collapse
COLLAPSE_CONFIRMATION = {
    "min_consecutive_days_below": 3,   # Days price must stay below threshold
    "min_drawdown_from_peak": -10,     # Minimum drawdown to confirm
    "underlying_correlation_min": 0.5,  # MSTY should track MSTR in collapse
}

# Output
OUTPUT_DIR = "output"
