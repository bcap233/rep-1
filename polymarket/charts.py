"""
Price Chart Analysis Engine.

Builds 5/10/15 minute charts from exchange candle data and computes
technical indicators for momentum detection:
  - Rate of Change (ROC)
  - RSI (Relative Strength Index)
  - VWAP (Volume-Weighted Average Price)
  - Volume profile
  - Bollinger Band position

These indicators are used by the signal generator to detect when
spot price momentum diverges from Polymarket implied probabilities.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from .exchanges import Candle

logger = logging.getLogger(__name__)


@dataclass
class ChartAnalysis:
    """Technical analysis results for one timeframe."""
    asset: str
    timeframe_minutes: int
    num_candles: int

    # Current price info
    last_price: float
    last_close: float
    last_volume: float

    # Rate of change
    roc_1: float       # 1-candle ROC (%)
    roc_3: float       # 3-candle ROC (%)
    roc_5: float       # 5-candle ROC (%)

    # RSI
    rsi: float         # 14-period RSI

    # VWAP
    vwap: float        # Session VWAP
    vwap_deviation: float  # (price - vwap) / vwap * 100

    # Bollinger Bands (20-period, 2 std dev)
    bb_upper: float
    bb_lower: float
    bb_middle: float
    bb_position: float  # 0 = at lower, 1 = at upper

    # Volume analysis
    volume_sma: float       # Average volume (20-period)
    volume_ratio: float     # Current volume / average
    volume_trend: str       # "rising", "falling", "flat"

    # Momentum summary
    momentum_score: float   # -1.0 (bearish) to +1.0 (bullish)
    momentum_direction: str  # "bullish", "bearish", "neutral"

    # Raw candles for further analysis
    candles: list[Candle] = field(default_factory=list)


def compute_roc(prices: list[float], period: int) -> float:
    """Rate of change over N periods (%)."""
    if len(prices) < period + 1:
        return 0.0
    old = prices[-(period + 1)]
    if old == 0:
        return 0.0
    return (prices[-1] - old) / old * 100


def compute_rsi(prices: list[float], period: int = 14) -> float:
    """RSI using exponential moving average of gains/losses."""
    if len(prices) < period + 1:
        return 50.0  # Neutral default

    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

    # Use only the last N+1 periods for calculation
    relevant = changes[-(period * 3):]  # Extra history for EMA warmup

    gains = [max(0, c) for c in relevant]
    losses = [max(0, -c) for c in relevant]

    if len(gains) < period:
        return 50.0

    # Initial SMA
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # EMA smoothing
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_vwap(candles: list[Candle]) -> float:
    """Volume-Weighted Average Price."""
    if not candles:
        return 0.0

    total_pv = 0.0
    total_v = 0.0

    for c in candles:
        typical_price = (c.high + c.low + c.close) / 3
        total_pv += typical_price * c.volume
        total_v += c.volume

    if total_v == 0:
        return candles[-1].close
    return total_pv / total_v


def compute_bollinger(prices: list[float], period: int = 20,
                      num_std: float = 2.0) -> tuple[float, float, float]:
    """Bollinger Bands: (upper, middle, lower)."""
    if len(prices) < period:
        p = prices[-1] if prices else 0
        return p, p, p

    window = prices[-period:]
    middle = sum(window) / len(window)
    variance = sum((x - middle) ** 2 for x in window) / len(window)
    std = math.sqrt(variance)

    return middle + num_std * std, middle, middle - num_std * std


def compute_volume_trend(volumes: list[float], period: int = 10) -> str:
    """Determine if volume is trending up, down, or flat."""
    if len(volumes) < period:
        return "flat"

    recent = volumes[-period:]
    first_half = sum(recent[:period // 2]) / (period // 2)
    second_half = sum(recent[period // 2:]) / (period // 2)

    if first_half == 0:
        return "flat"

    change = (second_half - first_half) / first_half
    if change > 0.20:
        return "rising"
    elif change < -0.20:
        return "falling"
    return "flat"


def analyze_timeframe(asset: str, candles: list[Candle],
                      timeframe_minutes: int) -> Optional[ChartAnalysis]:
    """
    Run full technical analysis on a set of candles for one timeframe.

    Args:
        asset: Asset name (e.g., "BTC")
        candles: OHLCV candles, sorted by time ascending
        timeframe_minutes: The timeframe these candles represent

    Returns:
        ChartAnalysis with all indicators computed
    """
    if len(candles) < 5:
        logger.warning(f"{asset} {timeframe_minutes}m: Not enough candles ({len(candles)})")
        return None

    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]

    # Rate of change
    roc_1 = compute_roc(closes, 1)
    roc_3 = compute_roc(closes, 3)
    roc_5 = compute_roc(closes, 5)

    # RSI
    rsi = compute_rsi(closes, 14)

    # VWAP
    vwap = compute_vwap(candles)
    vwap_dev = (closes[-1] - vwap) / vwap * 100 if vwap != 0 else 0

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = compute_bollinger(closes, 20)
    bb_range = bb_upper - bb_lower
    bb_position = (closes[-1] - bb_lower) / bb_range if bb_range > 0 else 0.5

    # Volume analysis
    vol_sma = sum(volumes[-20:]) / min(len(volumes), 20) if volumes else 0
    vol_ratio = volumes[-1] / vol_sma if vol_sma > 0 else 1.0
    vol_trend = compute_volume_trend(volumes)

    # Momentum score: composite of all indicators
    # Scale each to [-1, +1] and weight
    momentum_components = []

    # ROC component (strongest recent momentum)
    roc_norm = max(-1, min(1, roc_3 / 1.0))  # ±1% = full signal
    momentum_components.append(("roc", roc_norm, 0.35))

    # VWAP deviation
    vwap_norm = max(-1, min(1, vwap_dev / 0.5))  # ±0.5% = full signal
    momentum_components.append(("vwap", vwap_norm, 0.25))

    # RSI component
    rsi_norm = (rsi - 50) / 50  # 0→-1, 50→0, 100→+1
    rsi_norm = max(-1, min(1, rsi_norm))
    momentum_components.append(("rsi", rsi_norm, 0.20))

    # Volume confirmation (high volume on directional moves = stronger signal)
    vol_confirm = 0.0
    if vol_ratio > 1.5:
        vol_confirm = 1.0 if roc_1 > 0 else -1.0
    elif vol_ratio > 1.0:
        vol_confirm = 0.5 if roc_1 > 0 else -0.5
    momentum_components.append(("volume", vol_confirm, 0.10))

    # BB position
    bb_norm = (bb_position - 0.5) * 2  # 0→-1, 0.5→0, 1→+1
    bb_norm = max(-1, min(1, bb_norm))
    momentum_components.append(("bb", bb_norm, 0.10))

    # Weighted score
    score = sum(val * weight for _, val, weight in momentum_components)
    score = max(-1, min(1, score))

    if score > 0.2:
        direction = "bullish"
    elif score < -0.2:
        direction = "bearish"
    else:
        direction = "neutral"

    return ChartAnalysis(
        asset=asset,
        timeframe_minutes=timeframe_minutes,
        num_candles=len(candles),
        last_price=closes[-1],
        last_close=closes[-1],
        last_volume=volumes[-1],
        roc_1=roc_1,
        roc_3=roc_3,
        roc_5=roc_5,
        rsi=rsi,
        vwap=vwap,
        vwap_deviation=vwap_dev,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        bb_middle=bb_middle,
        bb_position=bb_position,
        volume_sma=vol_sma,
        volume_ratio=vol_ratio,
        volume_trend=vol_trend,
        momentum_score=score,
        momentum_direction=direction,
        candles=candles,
    )


def analyze_all_timeframes(asset: str,
                           candles_by_tf: dict[int, list[Candle]]) -> list[ChartAnalysis]:
    """
    Analyze all timeframes and return results.

    Args:
        asset: Asset name
        candles_by_tf: {timeframe_minutes: [candles]}

    Returns:
        List of ChartAnalysis, one per timeframe
    """
    results = []
    for tf, candles in sorted(candles_by_tf.items()):
        analysis = analyze_timeframe(asset, candles, tf)
        if analysis:
            results.append(analysis)
    return results


def multi_timeframe_consensus(analyses: list[ChartAnalysis]) -> tuple[float, str]:
    """
    Compute consensus across multiple timeframes.

    Returns (consensus_score, direction).
    Consensus is stronger when all timeframes agree.
    """
    if not analyses:
        return 0.0, "neutral"

    scores = [a.momentum_score for a in analyses]
    avg_score = sum(scores) / len(scores)

    # Check agreement
    all_bullish = all(s > 0 for s in scores)
    all_bearish = all(s < 0 for s in scores)

    if all_bullish or all_bearish:
        # Amplify consensus: all timeframes agree
        avg_score *= 1.25
        avg_score = max(-1, min(1, avg_score))

    if avg_score > 0.15:
        direction = "bullish"
    elif avg_score < -0.15:
        direction = "bearish"
    else:
        direction = "neutral"

    return avg_score, direction
