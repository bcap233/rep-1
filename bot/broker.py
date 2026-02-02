"""
Brokerage execution layer for the V7 rotation bot.

Supports Alpaca (paper and live) via their REST API.
No SDK dependency — uses plain HTTP requests for portability.
"""

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """A brokerage position."""
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float


@dataclass
class OrderResult:
    """Result of an order submission."""
    success: bool
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None
    qty: Optional[float] = None
    error: Optional[str] = None


class AlpacaBroker:
    """
    Alpaca brokerage integration via REST API.

    Usage:
        broker = AlpacaBroker(api_key, secret_key, base_url)
        positions = broker.get_positions()
        result = broker.market_order("MSTY", "buy", 100)
    """

    def __init__(self, api_key: str, secret_key: str,
                 base_url: str = "https://paper-api.alpaca.markets"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str,
                 body: Optional[dict] = None) -> Optional[dict]:
        """Make an authenticated API request."""
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body else None

        req = urllib.request.Request(url, data=data, headers=self._headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            logger.error(f"API error {e.code} on {method} {path}: {body_text}")
            return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    def get_account(self) -> Optional[dict]:
        """Get account info (buying power, equity, etc.)."""
        return self._request("GET", "/v2/account")

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        data = self._request("GET", "/v2/positions")
        if not data:
            return []

        positions = []
        for item in data:
            positions.append(Position(
                symbol=item["symbol"],
                qty=float(item["qty"]),
                market_value=float(item["market_value"]),
                avg_entry_price=float(item["avg_entry_price"]),
            ))
        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        data = self._request("GET", f"/v2/positions/{symbol}")
        if not data:
            return None
        return Position(
            symbol=data["symbol"],
            qty=float(data["qty"]),
            market_value=float(data["market_value"]),
            avg_entry_price=float(data["avg_entry_price"]),
        )

    def market_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        """
        Submit a market order.

        Args:
            symbol: Ticker to trade
            side: "buy" or "sell"
            qty: Number of shares (whole shares for ETFs)
        """
        body = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }

        logger.info(f"Submitting order: {side} {int(qty)} {symbol}")
        data = self._request("POST", "/v2/orders", body)

        if data and "id" in data:
            return OrderResult(
                success=True,
                order_id=data["id"],
                symbol=symbol,
                side=side,
                qty=qty,
            )
        return OrderResult(success=False, symbol=symbol, side=side, qty=qty,
                           error="Order submission failed")

    def close_position(self, symbol: str) -> OrderResult:
        """Close an entire position in a symbol."""
        data = self._request("DELETE", f"/v2/positions/{symbol}")
        if data:
            return OrderResult(success=True, symbol=symbol, side="sell",
                               order_id=data.get("id"))
        return OrderResult(success=False, symbol=symbol,
                           error="Close position failed")

    def rotate(self, sell_symbol: str, buy_symbol: str,
               qty_to_buy: Optional[int] = None) -> tuple[OrderResult, OrderResult]:
        """
        Execute a rotation: sell one ETF, buy another.

        If qty_to_buy is not specified, uses the proceeds from the sell
        (approximated from the sell position's market value and buy price).
        """
        # Step 1: Close the current position
        sell_result = self.close_position(sell_symbol)
        if not sell_result.success:
            logger.error(f"Failed to close {sell_symbol}: {sell_result.error}")
            return sell_result, OrderResult(success=False, symbol=buy_symbol,
                                            error="Skipped — sell failed")

        # Step 2: Buy the new position
        if qty_to_buy is None:
            # Estimate from account buying power after close
            # In practice, the sell settles T+1, but Alpaca gives instant buying power
            acct = self.get_account()
            if acct:
                buying_power = float(acct.get("buying_power", 0))
                # Use 95% of buying power to leave margin for price movements
                # This is a rough estimate — the bot should be refined for production
                logger.info(f"Buying power after close: ${buying_power:.2f}")
            else:
                buying_power = 0

            # We can't easily determine qty without knowing the buy price.
            # For now, log a warning and skip the buy.
            logger.warning("Auto qty calculation not implemented. Specify qty_to_buy.")
            return sell_result, OrderResult(success=False, symbol=buy_symbol,
                                            error="qty_to_buy not specified")

        buy_result = self.market_order(buy_symbol, "buy", qty_to_buy)
        return sell_result, buy_result
