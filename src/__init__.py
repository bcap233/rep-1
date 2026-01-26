"""
Covered Call / Yield Max Backtest Analysis Package
"""

from .data_fetcher import DataFetcher
from .indicators import TechnicalIndicators
from .analysis import DeclineAnalyzer, SMARelationshipAnalyzer
from .visualization import BacktestVisualizer
from .collapse_detector import CollapseDetector, CollapseEvent, CollapseSignal
from .returns_analysis import ReturnsRegimeAnalyzer, ReturnsVisualizer, RegimeStats

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
]
