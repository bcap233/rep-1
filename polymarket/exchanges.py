"""
Exchange spot price feeds.

Fetches real-time prices and OHLCV candles from Binance, Coinbase, and Kraken
using their public REST APIs (no authentication needed).

Each exchange adapter returns a normalized format so the rest of the bot
doesn't need to care which exchange the data came from.
"""

import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

from .config import EXCHANGES, ASSETS

logger = logging.getLogger(__name__)


@dataclass
class Ticker:
    """Real-time spot price from an exchange."""
    exchange: str
    symbol: str
    price: float
    bid: float
    ask: float
    volume_24h: float
    timestamp: float  # Unix epoch


@dataclass
class Candle:
    """Single OHLCV candle."""
    timestamp: float  # Open time (unix epoch)
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class CompositePrice:
    """Volume-weighted composite price across exchanges."""
    asset: str
    price: float
    bid: float
    ask: float
    sources: list[Ticker] = field(default_factory=list)
    timestamp: float = 0.0


def _http_get(url: str, timeout: int = 10) -> Optional[dict | list]:
    """Simple HTTP GET returning parsed JSON."""
    req = urllib.request.Request(url, headers={"User-Agent": "polymarket-arb-bot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        logger.warning(f"HTTP {e.code} from {url}")
        return None
    except Exception as e:
        logger.warning(f"Request failed for {url}: {e}")
        return None


# ============================================================
# BINANCE
# ============================================================

def binance_ticker(symbol: str) -> Optional[Ticker]:
    """Fetch current ticker from Binance."""
    base = EXCHANGES["binance"]["base_url"]
    data = _http_get(f"{base}/api/v3/ticker/bookTicker?symbol={symbol}")
    if not data:
        return None

    price_data = _http_get(f"{base}/api/v3/ticker/24hr?symbol={symbol}")
    volume = float(price_data.get("quoteVolume", 0)) if price_data else 0

    return Ticker(
        exchange="binance",
        symbol=symbol,
        price=(float(data["bidPrice"]) + float(data["askPrice"])) / 2,
        bid=float(data["bidPrice"]),
        ask=float(data["askPrice"]),
        volume_24h=volume,
        timestamp=time.time(),
    )


def binance_candles(symbol: str, interval_minutes: int, limit: int = 100) -> list[Candle]:
    """
    Fetch klines (candles) from Binance.

    interval_minutes maps to Binance intervals: 1, 3, 5, 15, 30, 60, etc.
    """
    interval_map = {1: "1m", 3: "3m", 5: "5m", 10: "5m", 15: "15m", 30: "30m", 60: "1h"}
    interval = interval_map.get(interval_minutes, "5m")

    # For 10-minute candles, fetch 5m and aggregate
    fetch_limit = limit * 2 if interval_minutes == 10 else limit

    base = EXCHANGES["binance"]["base_url"]
    data = _http_get(f"{base}/api/v3/klines?symbol={symbol}&interval={interval}&limit={fetch_limit}")
    if not data:
        return []

    candles = []
    for k in data:
        candles.append(Candle(
            timestamp=k[0] / 1000.0,
            open=float(k[1]),
            high=float(k[2]),
            low=float(k[3]),
            close=float(k[4]),
            volume=float(k[5]),
        ))

    if interval_minutes == 10:
        candles = _aggregate_candles(candles, 2)

    return candles


# ============================================================
# COINBASE
# ============================================================

def coinbase_ticker(symbol: str) -> Optional[Ticker]:
    """Fetch current ticker from Coinbase."""
    base = EXCHANGES["coinbase"]["base_url"]

    # Coinbase uses /v2/prices/{pair}/spot for price
    data = _http_get(f"{base}/v2/prices/{symbol}/spot")
    if not data or "data" not in data:
        return None

    price = float(data["data"]["amount"])

    # Get bid/ask from order book
    book = _http_get(f"{base}/v2/prices/{symbol}/buy")
    buy_price = float(book["data"]["amount"]) if book and "data" in book else price
    sell = _http_get(f"{base}/v2/prices/{symbol}/sell")
    sell_price = float(sell["data"]["amount"]) if sell and "data" in sell else price

    return Ticker(
        exchange="coinbase",
        symbol=symbol,
        price=price,
        bid=sell_price,  # What you'd get selling (bid)
        ask=buy_price,   # What you'd pay buying (ask)
        volume_24h=0,    # Coinbase v2 doesn't expose this easily
        timestamp=time.time(),
    )


def coinbase_candles(symbol: str, interval_minutes: int, limit: int = 100) -> list[Candle]:
    """
    Fetch candles from Coinbase Exchange API.

    Uses the Coinbase Exchange (Advanced Trade) candle endpoint.
    Granularity: 60, 300, 900, 3600, 21600, 86400
    """
    granularity_map = {1: 60, 5: 300, 10: 300, 15: 900, 30: 1800, 60: 3600}
    granularity = granularity_map.get(interval_minutes, 300)

    fetch_limit = limit * 2 if interval_minutes == 10 else limit
    end = int(time.time())
    start = end - (fetch_limit * granularity)

    # Coinbase Exchange API (public, no auth needed for candles)
    url = (f"https://api.exchange.coinbase.com/products/{symbol}/candles"
           f"?granularity={granularity}&start={start}&end={end}")
    data = _http_get(url)
    if not data:
        return []

    candles = []
    for row in sorted(data, key=lambda x: x[0]):
        # Coinbase format: [time, low, high, open, close, volume]
        candles.append(Candle(
            timestamp=float(row[0]),
            open=float(row[3]),
            high=float(row[2]),
            low=float(row[1]),
            close=float(row[4]),
            volume=float(row[5]),
        ))

    if interval_minutes == 10:
        candles = _aggregate_candles(candles, 2)

    return candles


# ============================================================
# KRAKEN
# ============================================================

def kraken_ticker(symbol: str) -> Optional[Ticker]:
    """Fetch current ticker from Kraken."""
    base = EXCHANGES["kraken"]["base_url"]
    data = _http_get(f"{base}/0/public/Ticker?pair={symbol}")
    if not data or data.get("error"):
        return None

    result = data.get("result", {})
    if not result:
        return None

    # Kraken uses internal pair names in the response
    key = list(result.keys())[0]
    info = result[key]

    return Ticker(
        exchange="kraken",
        symbol=symbol,
        price=float(info["c"][0]),  # Last trade price
        bid=float(info["b"][0]),
        ask=float(info["a"][0]),
        volume_24h=float(info["v"][1]) * float(info["c"][0]),  # Volume * price
        timestamp=time.time(),
    )


def kraken_candles(symbol: str, interval_minutes: int, limit: int = 100) -> list[Candle]:
    """
    Fetch OHLC data from Kraken.

    Intervals: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600
    """
    interval_map = {1: 1, 5: 5, 10: 5, 15: 15, 30: 30, 60: 60}
    interval = interval_map.get(interval_minutes, 5)

    base = EXCHANGES["kraken"]["base_url"]
    data = _http_get(f"{base}/0/public/OHLC?pair={symbol}&interval={interval}")
    if not data or data.get("error"):
        return []

    result = data.get("result", {})
    key = [k for k in result.keys() if k != "last"]
    if not key:
        return []

    rows = result[key[0]]
    candles = []
    for row in rows[-limit * (2 if interval_minutes == 10 else 1):]:
        # Kraken format: [time, open, high, low, close, vwap, volume, count]
        candles.append(Candle(
            timestamp=float(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[6]),
        ))

    if interval_minutes == 10:
        candles = _aggregate_candles(candles, 2)

    return candles


# ============================================================
# AGGREGATION
# ============================================================

def _aggregate_candles(candles: list[Candle], factor: int) -> list[Candle]:
    """
    Aggregate smaller candles into larger ones.

    E.g., factor=2 on 5-minute candles produces 10-minute candles.
    """
    aggregated = []
    for i in range(0, len(candles) - factor + 1, factor):
        group = candles[i:i + factor]
        aggregated.append(Candle(
            timestamp=group[0].timestamp,
            open=group[0].open,
            high=max(c.high for c in group),
            low=min(c.low for c in group),
            close=group[-1].close,
            volume=sum(c.volume for c in group),
        ))
    return aggregated


def get_composite_price(asset: str) -> Optional[CompositePrice]:
    """
    Get a volume-weighted composite price for an asset across all enabled exchanges.

    Falls back to equal weighting if volume data is unavailable.
    """
    if asset not in ASSETS:
        logger.error(f"Unknown asset: {asset}")
        return None

    asset_cfg = ASSETS[asset]
    tickers = []

    # Fetch from each enabled exchange
    fetchers = {
        "binance": binance_ticker,
        "coinbase": coinbase_ticker,
        "kraken": kraken_ticker,
    }

    for exchange, cfg in EXCHANGES.items():
        if not cfg.get("enabled"):
            continue
        symbol = asset_cfg.get(exchange)
        if not symbol:
            continue

        fetcher = fetchers.get(exchange)
        if not fetcher:
            continue

        ticker = fetcher(symbol)
        if ticker:
            tickers.append(ticker)

    if not tickers:
        logger.error(f"No price data for {asset} from any exchange")
        return None

    # Compute weighted price
    total_weight = 0
    weighted_price = 0
    weighted_bid = 0
    weighted_ask = 0

    for t in tickers:
        w = EXCHANGES[t.exchange]["weight"]
        weighted_price += t.price * w
        weighted_bid += t.bid * w
        weighted_ask += t.ask * w
        total_weight += w

    if total_weight > 0:
        weighted_price /= total_weight
        weighted_bid /= total_weight
        weighted_ask /= total_weight

    return CompositePrice(
        asset=asset,
        price=weighted_price,
        bid=weighted_bid,
        ask=weighted_ask,
        sources=tickers,
        timestamp=time.time(),
    )


def get_candles(asset: str, exchange: str, interval_minutes: int,
                limit: int = 100) -> list[Candle]:
    """
    Get candles for an asset from a specific exchange.

    Args:
        asset: Asset key (e.g., "BTC")
        exchange: Exchange name (e.g., "binance")
        interval_minutes: Candle interval (5, 10, 15)
        limit: Number of candles to fetch
    """
    if asset not in ASSETS:
        return []

    symbol = ASSETS[asset].get(exchange)
    if not symbol:
        return []

    fetchers = {
        "binance": binance_candles,
        "coinbase": coinbase_candles,
        "kraken": kraken_candles,
    }

    fetcher = fetchers.get(exchange)
    if not fetcher:
        return []

    return fetcher(symbol, interval_minutes, limit)


def get_best_candles(asset: str, interval_minutes: int, limit: int = 100) -> list[Candle]:
    """
    Get candles from the highest-priority enabled exchange.

    Tries Binance first (most liquid), then Coinbase, then Kraken.
    """
    priority = ["binance", "coinbase", "kraken"]

    for exchange in priority:
        if not EXCHANGES.get(exchange, {}).get("enabled"):
            continue
        candles = get_candles(asset, exchange, interval_minutes, limit)
        if candles:
            return candles

    return []
