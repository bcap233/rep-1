"""
Polymarket CLOB API Client.

Handles market discovery (via Gamma API) and order placement (via CLOB API).

Authentication:
  Polymarket's CLOB uses API keys derived from your Polygon wallet.
  You generate a key pair by signing a message with your private key.
  Orders are then signed with EIP-712 typed data signatures.

  This module uses the official `py-clob-client` SDK for authenticated
  operations (order placement, cancellation) and raw HTTP for public
  endpoints (order books, market data) to minimize latency.

Setup:
  1. pip install py-clob-client
  2. Get a Polygon wallet with USDC
  3. Generate API creds: python -m polymarket --setup
  4. Fund your account on polymarket.com
"""

import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

from .config import POLYMARKET

logger = logging.getLogger(__name__)

# SDK imports — graceful fallback if not installed
_HAS_SDK = False
_ClobClient = None
_ApiCreds = None

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
    _ClobClient = ClobClient
    _ApiCreds = ApiCreds
    _HAS_SDK = True
except ImportError:
    logger.debug("py-clob-client not installed — live trading disabled, "
                  "paper mode and market data still work")


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


def _http_get(url: str, timeout: int = 10) -> Optional[dict | list]:
    """Simple HTTP GET returning parsed JSON. Used for public endpoints."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "polymarket-arb-bot/1.0",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        logger.warning(f"HTTP {e.code} from {url}: {body_text[:200]}")
        return None
    except Exception as e:
        logger.warning(f"Request failed for {url}: {e}")
        return None


class PolymarketClient:
    """
    Client for Polymarket's CLOB and Gamma APIs.

    Two modes:
      - Public (no SDK needed): market discovery, order books, midpoints
      - Authenticated (requires py-clob-client): order placement, cancellation

    Usage:
        client = PolymarketClient()

        # Public — always works
        markets = client.search_markets("bitcoin price")
        book = client.get_order_book(token_id)

        # Authenticated — requires SDK + credentials
        order = client.place_order(token_id, "BUY", 0.45, 50)
    """

    def __init__(self):
        self.clob_url = POLYMARKET["clob_url"].rstrip("/")
        self.gamma_url = POLYMARKET["gamma_url"].rstrip("/")
        self._sdk_client = None
        self._authenticated = False

        # Initialize SDK client if credentials are present
        private_key = POLYMARKET.get("private_key", "")
        if private_key and _HAS_SDK:
            self._init_sdk(private_key)

    def _init_sdk(self, private_key: str):
        """
        Initialize the py-clob-client SDK for authenticated operations.

        The SDK handles:
          - Deriving API credentials from your Polygon wallet
          - EIP-712 typed data signing for orders
          - CLOB API authentication headers
        """
        try:
            chain_id = POLYMARKET.get("chain_id", 137)

            # If we have saved API creds, use them directly
            api_key = POLYMARKET.get("api_key", "")
            api_secret = POLYMARKET.get("api_secret", "")
            api_passphrase = POLYMARKET.get("api_passphrase", "")

            if api_key and api_secret and api_passphrase:
                creds = _ApiCreds(
                    api_key=api_key,
                    api_secret=api_secret,
                    api_passphrase=api_passphrase,
                )
                self._sdk_client = _ClobClient(
                    self.clob_url,
                    key=private_key,
                    chain_id=chain_id,
                    creds=creds,
                    funder=POLYMARKET.get("funder", "") or None,
                )
            else:
                # Create client and derive creds from wallet
                self._sdk_client = _ClobClient(
                    self.clob_url,
                    key=private_key,
                    chain_id=chain_id,
                    funder=POLYMARKET.get("funder", "") or None,
                )
                # Derive and save API credentials
                self._sdk_client.set_api_creds(self._sdk_client.create_or_derive_api_creds())

            self._authenticated = True
            logger.info("Polymarket SDK initialized — live trading enabled")

        except Exception as e:
            logger.error(f"Failed to initialize Polymarket SDK: {e}")
            logger.info("Falling back to public-only mode (no live trading)")
            self._sdk_client = None
            self._authenticated = False

    @property
    def can_trade(self) -> bool:
        """Whether authenticated trading is available."""
        return self._authenticated and self._sdk_client is not None

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
        data = _http_get(url)
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
        item = _http_get(url)
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
        price_keywords = ["price", "above", "below", "reach", "hit",
                          "over", "under", "higher", "lower"]
        price_markets = []
        for m in unique:
            q = m.question.lower()
            if any(kw in q for kw in price_keywords):
                price_markets.append(m)

        return price_markets

    def search_all_categories(self, limit_per_query: int = 100) -> list[Market]:
        """
        Search across all event categories for broad market coverage.

        Runs the default broad query plus targeted queries from
        EVENT_CATEGORIES, deduplicating results.
        """
        from .config import EVENT_CATEGORIES

        seen: set[str] = set()
        all_markets: list[Market] = []

        def _add(markets: list[Market]):
            for m in markets:
                if m.condition_id not in seen:
                    seen.add(m.condition_id)
                    all_markets.append(m)

        # Broad scan first (gets whatever Gamma returns by default)
        _add(self.search_markets("", limit=200, active_only=True))

        # Category-specific queries to find markets the broad scan misses
        for cat_name, cat_cfg in EVENT_CATEGORIES.items():
            for query in cat_cfg["queries"]:
                results = self.search_markets(query, limit=limit_per_query,
                                              active_only=True)
                _add(results)

        logger.info(f"Category scan: {len(all_markets)} unique markets "
                     f"across {len(EVENT_CATEGORIES)} categories")
        return all_markets

    # --------------------------------------------------------
    # Order Book (CLOB API — public for reads)
    # --------------------------------------------------------

    def get_order_book(self, token_id: str) -> Optional[OrderBook]:
        """Fetch the current order book for a token."""
        url = f"{self.clob_url}/book?token_id={token_id}"
        data = _http_get(url)
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

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get the midpoint price for a token."""
        url = f"{self.clob_url}/midpoint?token_id={token_id}"
        data = _http_get(url)
        if data and "mid" in data:
            return float(data["mid"])
        return None

    def get_trades(self, token_id: str, limit: int = 50) -> list[dict]:
        """Get recent trades for a token (public)."""
        url = f"{self.clob_url}/trades?token_id={token_id}&limit={limit}"
        data = _http_get(url)
        return data if data else []

    # --------------------------------------------------------
    # Trading (CLOB API — requires py-clob-client SDK)
    # --------------------------------------------------------

    def place_order(self, token_id: str, side: str,
                    price: float, size: float) -> Optional[Order]:
        """
        Place a limit order on the CLOB.

        Requires py-clob-client SDK and valid credentials.
        The SDK handles EIP-712 signing automatically.

        Args:
            token_id: The token to trade (YES or NO token)
            side: "BUY" or "SELL"
            price: Limit price (0.01 to 0.99)
            size: Number of shares
        """
        if not self.can_trade:
            if not _HAS_SDK:
                logger.error("Cannot place order: py-clob-client not installed. "
                             "Run: pip install py-clob-client")
            else:
                logger.error("Cannot place order: not authenticated. "
                             "Set private_key in config.py")
            return None

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType

            # Build order via SDK — it handles signing
            order_args = OrderArgs(
                price=round(price, 2),
                size=round(size, 2),
                side=side.upper(),
                token_id=token_id,
            )

            # Create the signed order
            signed_order = self._sdk_client.create_order(order_args)

            # Post it to the CLOB
            response = self._sdk_client.post_order(signed_order, OrderType.GTC)

            if not response:
                logger.error("Order post returned empty response")
                return None

            order_id = ""
            status = "LIVE"
            if isinstance(response, dict):
                order_id = response.get("orderID", response.get("id", ""))
                status = response.get("status", "LIVE")
            else:
                order_id = str(response)

            logger.info(f"Order placed: {order_id} | {side} {size} @ {price}")

            return Order(
                order_id=order_id,
                token_id=token_id,
                side=side.upper(),
                price=price,
                size=size,
                status=status,
                timestamp=time.time(),
            )

        except Exception as e:
            logger.error(f"Order placement failed: {e}", exc_info=True)
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if not self.can_trade:
            return False

        try:
            self._sdk_client.cancel(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Cancel failed for {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        if not self.can_trade:
            return False

        try:
            self._sdk_client.cancel_all()
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.error(f"Cancel all failed: {e}")
            return False

    def get_open_orders(self) -> list[Order]:
        """Get all open orders."""
        if not self.can_trade:
            return []

        try:
            raw_orders = self._sdk_client.get_orders()
            if not raw_orders:
                return []

            orders = []
            for item in raw_orders:
                if isinstance(item, dict):
                    orders.append(Order(
                        order_id=item.get("orderID", item.get("id", "")),
                        token_id=item.get("asset_id", item.get("tokenID", "")),
                        side=item.get("side", ""),
                        price=float(item.get("price", 0)),
                        size=float(item.get("original_size", item.get("size", 0))),
                        status=item.get("status", ""),
                        filled_size=float(item.get("size_matched", 0)),
                        timestamp=time.time(),
                    ))
            return orders

        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            return []

    # --------------------------------------------------------
    # Account info
    # --------------------------------------------------------

    def get_balance(self) -> Optional[float]:
        """Get USDC balance available for trading."""
        if not self.can_trade:
            return None

        try:
            # The SDK doesn't expose balance directly — use the API
            url = f"{self.clob_url}/balance"
            # This requires auth, so use the SDK's internal request
            balance = self._sdk_client.get_balance_allowance()
            if balance and isinstance(balance, dict):
                return float(balance.get("balance", 0))
        except Exception as e:
            logger.debug(f"Balance check failed: {e}")

        return None


def derive_api_creds(private_key: str, chain_id: int = 137) -> Optional[dict]:
    """
    Derive Polymarket API credentials from a private key.

    This is used during initial setup. The derived creds should be
    saved to config.py so they don't need to be re-derived each time.

    Returns {"api_key": ..., "api_secret": ..., "api_passphrase": ...}
    """
    if not _HAS_SDK:
        logger.error("py-clob-client required. Run: pip install py-clob-client")
        return None

    try:
        clob_url = POLYMARKET["clob_url"]
        client = _ClobClient(clob_url, key=private_key, chain_id=chain_id)
        creds = client.create_or_derive_api_creds()

        if hasattr(creds, 'api_key'):
            return {
                "api_key": creds.api_key,
                "api_secret": creds.api_secret,
                "api_passphrase": creds.api_passphrase,
            }
        elif isinstance(creds, dict):
            return creds

    except Exception as e:
        logger.error(f"Credential derivation failed: {e}", exc_info=True)

    return None
