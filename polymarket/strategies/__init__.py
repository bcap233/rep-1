"""
Strategy Framework.

All strategies implement the BaseStrategy interface and produce
Signal objects that the execution engine consumes uniformly.

Available strategies:
  - spot_divergence:   Spot momentum vs Polymarket implied probability lag
  - high_prob_grinder: Buy 90%+ events at scale for small recurring profits
  - bilateral_arb:     Buy both YES+NO when total ask < $1.00 for risk-free profit
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from ..polymarket_client import PolymarketClient

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Base class for all Polymarket trading strategies.

    Each strategy scans for opportunities and produces Signal objects.
    The execution engine handles order placement, position tracking,
    and risk management identically regardless of which strategy
    produced the signal.
    """

    name: str = "base"
    description: str = ""

    def __init__(self, client: PolymarketClient):
        self.client = client

    @abstractmethod
    def scan(self, assets: Optional[list[str]] = None) -> list:
        """
        Scan for trading opportunities.

        Args:
            assets: Optional list of assets to focus on (e.g., ["BTC", "ETH"]).
                    Some strategies ignore this (e.g., high_prob_grinder
                    scans all markets regardless of asset).

        Returns:
            List of Signal objects, sorted by quality (best first).
        """
        ...

    def describe(self) -> str:
        """Human-readable description for the dashboard."""
        return f"{self.name}: {self.description}"


# Strategy registry — populated by imports
STRATEGIES: dict[str, type[BaseStrategy]] = {}


def register_strategy(cls: type[BaseStrategy]) -> type[BaseStrategy]:
    """Decorator to register a strategy class."""
    STRATEGIES[cls.name] = cls
    return cls


def get_strategy(name: str, client: PolymarketClient) -> BaseStrategy:
    """Instantiate a strategy by name."""
    if name not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return STRATEGIES[name](client)


def list_strategies() -> list[tuple[str, str]]:
    """Return [(name, description), ...] for all registered strategies."""
    return [(name, cls.description) for name, cls in STRATEGIES.items()]


# Import strategies to trigger registration
from . import spot_divergence  # noqa: E402, F401
from . import high_prob_grinder  # noqa: E402, F401
from . import bilateral_arb  # noqa: E402, F401
from . import market_maker  # noqa: E402, F401
