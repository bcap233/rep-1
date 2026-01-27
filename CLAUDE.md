# CLAUDE.md - AI Assistant Guide

## Project Overview

This is a **Covered Call ETF and Options Income Strategy Analysis System** focused on detecting collapse events in yield-focused ETFs. The primary subject of analysis is **MSTY** (YieldMax MSTR Option Income Strategy).

### Core Problem Being Solved

Covered call ETFs like MSTY exhibit asymmetric behavior:
- Upside is capped (premium from sold calls)
- Full downside exposure during declines
- "Reflexive collapses" can occur when the strategy's mechanics accelerate losses

The system uses technical analysis (primarily 50-day SMA relationships) to:
1. Detect when a "true collapse" begins (vs. normal volatility)
2. Analyze returns in different market regimes (above/below SMA)
3. Find optimal thresholds for collapse prediction
4. Compare covered call ETF performance to underlying assets

## Directory Structure

```
/home/user/rep-1/
├── src/                          # Core analysis modules
│   ├── __init__.py              # Package exports
│   ├── data_fetcher.py          # Yahoo Finance data retrieval
│   ├── indicators.py            # SMA, EMA technical calculations
│   ├── analysis.py              # Decline and SMA relationship analysis
│   ├── collapse_detector.py     # Core collapse detection engine
│   ├── returns_analysis.py      # Returns regime analysis
│   └── visualization.py         # Chart generation
├── analyze_msty.py              # Main MSTY analysis (live API data)
├── analyze_real_msty.py         # MSTY analysis using CSV data
├── run_backtest.py              # Multi-ticker covered call analysis
├── demo_analysis.py             # Demo with synthetic data
├── config.py                    # General configuration
├── msty_config.py               # MSTY-specific config
├── msty_data.csv                # Historical MSTY price data
├── requirements.txt             # Python dependencies
└── .gitignore                   # Git ignore rules
```

## Key Concepts

### 1. SMA Distance
The percentage distance between current price and the 50-day Simple Moving Average. This is the primary indicator for collapse detection.

```python
sma_distance_pct = ((price - sma_50) / sma_50) * 100
```

### 2. Collapse Thresholds (from msty_config.py)
- **Warning**: -5% below SMA
- **Caution**: -10% below SMA
- **Danger**: -15% below SMA
- **Collapse**: -20% below SMA

### 3. Velocity
The rate of change in SMA distance (how fast price is moving away from the SMA).

### 4. Collapse Probability
A multi-factor score (0-100%) combining:
- Distance from SMA (0-30 pts)
- Velocity (0-25 pts)
- Consecutive days below threshold (0-20 pts)
- Drawdown magnitude (0-15 pts)
- Underlying correlation (0-10 pts)

### 5. Regime Analysis
Returns are analyzed separately for different market regimes:
- **Above SMA**: Price is above the 50-day moving average
- **Below SMA**: Price is below but not far
- **Far Below SMA**: Price is significantly below (danger zone)

## Main Scripts

### `analyze_msty.py` - Primary Analysis Tool
```bash
python analyze_msty.py                  # Full analysis with live data
python analyze_msty.py --days 365       # Specify lookback period
python analyze_msty.py --current        # Show current status only
```

### `analyze_real_msty.py` - CSV Data Analysis
```bash
python analyze_real_msty.py             # Analyze using msty_data.csv
```

### `run_backtest.py` - Multi-Ticker Analysis
```bash
python run_backtest.py                           # Analyze all configured tickers
python run_backtest.py --tickers QYLD,XYLD      # Specific tickers only
python run_backtest.py --start 2020-01-01       # Custom date range
```

### `demo_analysis.py` - Synthetic Data Demo
```bash
python demo_analysis.py                 # Demo without API calls
```

## Core Classes

### `src/collapse_detector.py`
- **CollapseEvent**: Data class representing a detected collapse
- **CollapseSignal**: Real-time signal with probability and recommended action
- **CollapseDetector**: Main detection algorithm with methods:
  - `prepare_data()`: Calculate all metrics from OHLCV data
  - `calculate_collapse_probability()`: Score likelihood of true collapse
  - `get_current_signal()`: Get latest collapse signal
  - `find_optimal_threshold()`: Calibrate best threshold distance
  - `detect_collapse_events()`: Find historical collapses

### `src/returns_analysis.py`
- **RegimeStats**: Statistics for a particular SMA regime
- **ReturnsRegimeAnalyzer**: Analyze returns by market regime
- **RollingReturnsAnalyzer**: Calculate rolling returns over time windows
- **ReturnsVisualizer** / **RollingReturnsVisualizer**: Chart generators

### `src/indicators.py`
- **TechnicalIndicators**: Static methods for SMA, EMA, price ratios, drawdown

### `src/data_fetcher.py`
- **DataFetcher**: Yahoo Finance data retrieval with caching

### `src/visualization.py`
- **BacktestVisualizer**: Generate analysis charts (saved to `output/charts/`)

## Configuration Files

### `config.py`
General settings for covered call ETF analysis:
- `YIELD_MAX_TICKERS`: Dict of tickers to analyze (QYLD, XYLD, JEPI, etc.)
- `UNDERLYING_TICKERS`: Mapping to underlying indices
- `SMA_PERIODS`: [10, 20, 50, 100, 200]
- `DECLINE_THRESHOLDS`: Dict of severity levels (-5% to -30%)
- `OUTPUT_DIR`, `CHARTS_DIR`, `REPORTS_DIR`: Output paths

### `msty_config.py`
MSTY-specific settings:
- `TICKER = "MSTY"`, `UNDERLYING = "MSTR"`
- `SMA_PERIOD = 50`
- `DISTANCE_THRESHOLDS`, `VELOCITY_THRESHOLDS`
- `COLLAPSE_CONFIRMATION`: Minimum days, drawdown, correlation rules

## Code Conventions

### Type Hints
All functions use Python type hints:
```python
def calculate_sma(
    df: pd.DataFrame,
    periods: Union[int, List[int]],
    column: str = "close",
) -> pd.DataFrame:
```

### Docstrings
Google-style docstrings with Args, Returns sections:
```python
def method(self, param: type) -> ReturnType:
    """
    Brief description.

    Args:
        param: Description of parameter

    Returns:
        Description of return value
    """
```

### Data Classes
Use `@dataclass` for structured data:
```python
@dataclass
class CollapseEvent:
    start_date: datetime
    end_date: Optional[datetime]
    # ... fields with types
```

### DataFrame Conventions
- OHLCV columns are lowercase: `open`, `high`, `low`, `close`, `volume`
- Index is always `DatetimeIndex`
- Calculated columns use descriptive names: `sma_50`, `sma_distance_pct`, `drawdown_pct`

### Output Formatting
Use `tabulate` for console output:
```python
from tabulate import tabulate
print(tabulate(data, headers=headers, tablefmt="grid"))
```

### Visualization
- Charts saved to `output/charts/` at 150 DPI
- Use matplotlib with consistent styling
- Multi-panel figures for related metrics

## Dependencies

```
yfinance>=0.2.36       # Yahoo Finance API
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
matplotlib>=3.7.0      # Charting
seaborn>=0.12.0        # Statistical visualization
scipy>=1.10.0          # Scientific computing
tabulate>=0.9.0        # Pretty-print tables
```

Install with:
```bash
pip install -r requirements.txt
```

## Output

Generated files go to `output/` directory:
- `output/charts/`: PNG visualization files
- `output/reports/`: Analysis reports (if generated)

The `output/` directory and all `*.png`, `*.csv` files are gitignored.

## Key Insights for Analysis

1. **50-day SMA is the key indicator** - Most analysis revolves around price relationship to this moving average

2. **Velocity matters as much as distance** - How fast price moves away from SMA is critical

3. **Optimal threshold is usually -15% to -20%** - This distinguishes true collapses from normal pullbacks

4. **3+ consecutive days confirms signals** - Single-day breaks are often false signals

5. **Underlying correlation increases during collapses** - MSTY tracks MSTR more closely during declines

6. **Rolling returns before crashes show patterns** - Predictive analysis looks at what metrics showed before historical selloffs

## Adding New Analysis

To add a new analysis module:

1. Create new file in `src/` following existing patterns
2. Export classes in `src/__init__.py`
3. Create a top-level script if needed (like `analyze_msty.py`)
4. Use existing classes for data fetching and indicators
5. Follow type hint and docstring conventions

## Testing

No formal test suite exists. To verify functionality:
```bash
python demo_analysis.py    # Test with synthetic data (no API needed)
python analyze_real_msty.py  # Test with CSV data (no API needed)
```

## Common Tasks

### Get current MSTY collapse status
```bash
python analyze_msty.py --current
```

### Run full analysis on specific date range
```bash
python run_backtest.py --start 2023-01-01 --end 2024-01-01
```

### Analyze returns by market regime
```python
from src import ReturnsRegimeAnalyzer
analyzer = ReturnsRegimeAnalyzer(sma_period=50)
stats = analyzer.calculate_regime_stats(df)
```

### Detect historical collapse events
```python
from src import CollapseDetector, DataFetcher
fetcher = DataFetcher()
detector = CollapseDetector(sma_period=50)
# ... fetch data and run detect_collapse_events()
```
