"""
Polymarket CLOB API Client.

Handles market discovery (via Gamma API) and order placement (via CLOB API).
Uses plain HTTP requests for portability — no SDK dependency.

Key concepts:
  - Condition: A prediction market question (e.g., "Will BTC be above $100k?")
  - Token: A YES or NO outcome token for a condition
  - CLOB: Central Limit Order Book where tokens are traded
  - Price: 0.00 to 1.00 representing probability (and $/share for a $1 payout)
"""

import hashlib
import hmac
import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

from .config import POLYMARKET

logger = logging.getLogger(__name__)


@dataclass
class Market:
    """A Polymarket prediction market."""
    condition_id: str
    question: str
    description: str
    outcomes: list[str]          # e.g., ["Yes", "No"]
    token_ids: list[str]         # Corresponding token IDs
    end_date: str                # ISO date when market resolves
    active: bool
    closed: bool
    volume: float                # Total volume traded (USDC)
    liquidity: float             # Current liquidity (USDC)
    tags: list[str] = field(default_factory=list)

    @property
    def yes_token_id(self) -> str:
        return self.token_ids[0] if self.token_ids else ""

    @property
    def no_token_id(self) -> str:
        return self.token_ids[1] if len(self.token_ids) > 1 else ""


@dataclass
class OrderBook:
    """Order book snapshot for a token."""
    token_id: str
    bids: list[tuple[float, float]]  # [(price, size), ...]
    asks: list[tuple[float, float]]  # [(price, size), ...]
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    midpoint: float = 0.0


@dataclass
class Order:
    """An order on the CLOB."""
    order_id: str
    token_id: str
    side: str           # "BUY" or "SELL"
    price: float
    size: float
    status: str          # "LIVE", "FILLED", "CANCELLED"
    filled_size: float = 0.0
    timestamp: float = 0.0


def _http_request(url: str, method: str = "GET",
                  body: Optional[dict] = None,
                  headers: Optional[dict] = None,
                  timeout: int = 10) -> Optional[dict | list]:
    """HTTP request with JSON parsing."""
    default_headers = {
        "Content-Type": "application/json",
        "User-Agent": "polymarket-arb-bot/1.0",
    }
    if headers:
        default_headers.update(headers)

    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, headers=default_headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        logger.warning(f"HTTP {e.code} on {method} {url}: {body_text[:200]}")
        return None
    except Exception as e:
        logger.warning(f"Request failed: {e}")
        return None


class PolymarketClient:
    """
    Client for Polymarket's CLOB and Gamma APIs.

    Usage:
        client = PolymarketClient()
        markets = client.search_markets("bitcoin price")
        book = client.get_order_book(token_id)
        order = client.place_order(token_id, "BUY", 0.45, 50)
    """

    def __init__(self):
        self.clob_url = POLYMARKET["clob_url"].rstrip("/")
        self.gamma_url = POLYMARKET["gamma_url"].rstrip("/")
        self.api_key = POLYMARKET["api_key"]
        self.api_secret = POLYMARKET["api_secret"]
        self.api_passphrase = POLYMARKET["api_passphrase"]

    def _auth_headers(self, method: str, path: str,
                      body: str = "") -> dict:
        """Generate HMAC authentication headers for CLOB API."""
        if not self.api_key or not self.api_secret:
            return {}

        timestamp = str(int(time.time()))
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return {
            "POLY-API-KEY": self.api_key,
            "POLY-SIGNATURE": signature,
            "POLY-TIMESTAMP": timestamp,
            "POLY-PASSPHRASE": self.api_passphrase,
        }

    # --------------------------------------------------------
    # Market Discovery (Gamma API — public, no auth)
    # --------------------------------------------------------

    def search_markets(self, query: str, limit: int = 20,
                       active_only: bool = True) -> list[Market]:
        """
        Search for markets matching a query string.

        Uses the Gamma API which indexes all Polymarket markets.
        """
        params = f"?limit={limit}"
        if active_only:
            params += "&active=true&closed=false"

        url = f"{self.gamma_url}/markets{params}"
        data = _http_request(url)
        if not data:
            return []

        markets = []
        query_lower = query.lower()

        for item in data:
            question = item.get("question", "")
            description = item.get("description", "")
            tags = [t.get("label", "").lower() for t in item.get("tags", [])]

            # Filter by query relevance
            if (query_lower not in question.lower() and
                    query_lower not in description.lower() and
                    not any(query_lower in tag for tag in tags)):
                continue

            # Extract token IDs from CLOB token IDs field
            clob_token_ids = item.get("clobTokenIds", [])
            outcomes = item.get("outcomes", ["Yes", "No"])

            markets.append(Market(
                condition_id=item.get("conditionId", item.get("id", "")),
                question=question,
                description=description,
                outcomes=outcomes if isinstance(outcomes, list) else json.loads(outcomes),
                token_ids=clob_token_ids if isinstance(clob_token_ids, list) else json.loads(clob_token_ids),
                end_date=item.get("endDate", ""),
                active=item.get("active", False),
                closed=item.get("closed", False),
                volume=float(item.get("volume", 0)),
                liquidity=float(item.get("liquidity", 0)),
                tags=tags,
            ))

        return markets

    def get_market(self, condition_id: str) -> Optional[Market]:
        """Fetch a specific market by condition ID."""
        url = f"{self.gamma_url}/markets/{condition_id}"
        item = _http_request(url)
        if not item:
            return None

        clob_token_ids = item.get("clobTokenIds", [])
        outcomes = item.get("outcomes", ["Yes", "No"])
        tags = [t.get("label", "").lower() for t in item.get("tags", [])]

        return Market(
            condition_id=item.get("conditionId", item.get("id", "")),
            question=item.get("question", ""),
            description=item.get("description", ""),
            outcomes=outcomes if isinstance(outcomes, list) else json.loads(outcomes),
            token_ids=clob_token_ids if isinstance(clob_token_ids, list) else json.loads(clob_token_ids),
            end_date=item.get("endDate", ""),
            active=item.get("active", False),
            closed=item.get("closed", False),
            volume=float(item.get("volume", 0)),
            liquidity=float(item.get("liquidity", 0)),
            tags=tags,
        )

    def find_crypto_price_markets(self, asset: str) -> list[Market]:
        """
        Find active Polymarket markets related to a crypto asset's price.

        Searches for markets like "Will BTC be above $X?" or "Bitcoin price on [date]".
        """
        from .config import ASSETS
        tags = ASSETS.get(asset, {}).get("polymarket_tags", [asset.lower()])

        all_markets = []
        for tag in tags:
            markets = self.search_markets(tag, limit=50)
            all_markets.extend(markets)

        # Deduplicate by condition_id
        seen = set()
        unique = []
        for m in all_markets:
            if m.condition_id not in seen:
                seen.add(m.condition_id)
                unique.append(m)

        # Filter for price-related markets
        price_keywords = ["price", "above", "below", "reach", "hit", "over", "under", "higher", "lower"]
        price_markets = []
        for m in unique:
            q = m.question.lower()
            if any(kw in q for kw in price_keywords):
                price_markets.append(m)

        return price_markets

    # --------------------------------------------------------
    # Order Book (CLOB API — public for reads)
    # --------------------------------------------------------

    def get_order_book(self, token_id: str) -> Optional[OrderBook]:
        """Fetch the current order book for a token."""
        url = f"{self.clob_url}/book?token_id={token_id}"
        data = _http_request(url)
        if not data:
            return None

        bids = [(float(o["price"]), float(o["size"]))
                for o in data.get("bids", [])]
        asks = [(float(o["price"]), float(o["size"]))
                for o in data.get("asks", [])]

        # Sort: bids descending, asks ascending
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])

        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 1.0
        spread = best_ask - best_bid
        midpoint = (best_bid + best_ask) / 2

        return OrderBook(
            token_id=token_id,
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            midpoint=midpoint,
        )

    def get_price(self, token_id: str) -> Optional[float]:
        """Get the midpoint price for a token."""
        book = self.get_order_book(token_id)
        if book:
            return book.midpoint
        return None

    def get_prices(self, token_ids: list[str]) -> dict[str, float]:
        """Get prices for multiple tokens."""
        prices = {}
        for tid in token_ids:
            price = self.get_price(tid)
            if price is not None:
                prices[tid] = price
        return prices

    # --------------------------------------------------------
    # Trading (CLOB API — requires auth)
    # --------------------------------------------------------

    def place_order(self, token_id: str, side: str,
                    price: float, size: float) -> Optional[Order]:
        """
        Place a limit order on the CLOB.

        Args:
            token_id: The token to trade (YES or NO token)
            side: "BUY" or "SELL"
            price: Limit price (0.01 to 0.99)
            size: Number of shares
        """
        if not self.api_key:
            logger.error("Cannot place order: API key not configured")
            return None

        path = "/order"
        body = {
            "tokenID": token_id,
            "side": side.upper(),
            "price": str(round(price, 2)),
            "size": str(round(size, 2)),
            "type": "GTC",  # Good Till Cancelled
        }

        body_str = json.dumps(body)
        headers = self._auth_headers("POST", path, body_str)
        url = f"{self.clob_url}{path}"

        data = _http_request(url, method="POST", body=body, headers=headers)
        if not data:
            return None

        return Order(
            order_id=data.get("orderID", data.get("id", "")),
            token_id=token_id,
            side=side.upper(),
            price=price,
            size=size,
            status=data.get("status", "LIVE"),
            timestamp=time.time(),
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if not self.api_key:
            return False

        path = f"/order/{order_id}"
        headers = self._auth_headers("DELETE", path)
        url = f"{self.clob_url}{path}"

        data = _http_request(url, method="DELETE", headers=headers)
        return data is not None

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        if not self.api_key:
            return False

        path = "/orders/cancel-all"
        headers = self._auth_headers("DELETE", path)
        url = f"{self.clob_url}{path}"

        data = _http_request(url, method="DELETE", headers=headers)
        return data is not None

    def get_open_orders(self) -> list[Order]:
        """Get all open orders."""
        if not self.api_key:
            return []

        path = "/orders?status=LIVE"
        headers = self._auth_headers("GET", path)
        url = f"{self.clob_url}{path}"

        data = _http_request(url, headers=headers)
        if not data:
            return []

        orders = []
        for item in data:
            orders.append(Order(
                order_id=item.get("orderID", item.get("id", "")),
                token_id=item.get("tokenID", ""),
                side=item.get("side", ""),
                price=float(item.get("price", 0)),
                size=float(item.get("size", 0)),
                status=item.get("status", ""),
                filled_size=float(item.get("filledSize", 0)),
                timestamp=float(item.get("timestamp", 0)),
            ))
        return orders

    def get_trades(self, token_id: str, limit: int = 50) -> list[dict]:
        """Get recent trades for a token (public)."""
        url = f"{self.clob_url}/trades?token_id={token_id}&limit={limit}"
        data = _http_request(url)
        return data if data else []

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get the midpoint price for a token."""
        url = f"{self.clob_url}/midpoint?token_id={token_id}"
        data = _http_request(url)
        if data and "mid" in data:
            return float(data["mid"])
        return None
