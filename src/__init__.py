"""
Covered Call / Yield Max Backtest Analysis Package
"""

from .data_fetcher import DataFetcher
from .indicators import TechnicalIndicators
from .analysis import DeclineAnalyzer, SMARelationshipAnalyzer
from .visualization import BacktestVisualizer
from .collapse_detector import CollapseDetector, CollapseEvent, CollapseSignal
from .returns_analysis import (
    ReturnsRegimeAnalyzer,
    ReturnsVisualizer,
    RegimeStats,
    RollingReturnsAnalyzer,
    RollingReturnsVisualizer,
)
from .counterparty_analysis import (
    TotalReturnCalculator,
    CounterpartyAnalyzer,
    CounterpartyVisualizer,
)
from .dividend_analysis import (
    DividendAnalyzer,
    DividendVisualizer,
    DividendPayment,
)

__all__ = [
    "DataFetcher",
    "TechnicalIndicators",
    "DeclineAnalyzer",
    "SMARelationshipAnalyzer",
    "BacktestVisualizer",
    "CollapseDetector",
    "CollapseEvent",
    "CollapseSignal",
    "ReturnsRegimeAnalyzer",
    "ReturnsVisualizer",
    "RegimeStats",
    "RollingReturnsAnalyzer",
    "RollingReturnsVisualizer",
    "TotalReturnCalculator",
    "CounterpartyAnalyzer",
    "CounterpartyVisualizer",
    "DividendAnalyzer",
    "DividendVisualizer",
    "DividendPayment",
]
