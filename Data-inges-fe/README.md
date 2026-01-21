# Financial Data Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An end-to-end financial data pipeline for fetching, validating, and engineering features from equity market data.

## Overview

This pipeline provides a robust, reproducible system for processing financial market data with strict controls to prevent data leakage and ensure data quality.

### Data Source
- **Yahoo Finance** (via yfinance) - ONLY data source
  - Historical equity OHLCV data
  - Historical index OHLCV data
  - Both raw and adjusted prices (accounting for splits and dividends)

### Supported Timeframes
- 1 minute (1m)
- 5 minutes (5m)
- 1 hour (1h)
- 1 day (1d)

## Architecture

```
External APIs (Yahoo Finance)
    ↓
Data Ingestion
    ↓
Data Cleaning & Validation
    ↓
Feature Engineering
```

## Directory Structure

```
src/
├── config/
│   └── settings.py           # Configuration parameters
│
├── ingestion/
│   ├── equity_ohlcv.py       # Yahoo Finance data fetcher
│   └── runner.py             # Ingestion pipeline runner
│
├── validation/
│   ├── ohlcv_checks.py       # Data validation checks
│   └── validation_runner.py  # Validation pipeline runner
│
└── features/
    ├── technical_indicators.py  # Feature computation
    └── feature_runner.py        # Feature pipeline runner

main.py                       # Main pipeline orchestrator
requirements.txt              # Python dependencies
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Full Pipeline

Execute all stages (ingestion → validation → features):

```bash
python main.py
```

### Run Individual Stages

**Ingestion only:**
```bash
python src/ingestion/runner.py
```

**Validation only:**
```bash
python src/validation/validation_runner.py
```

**Feature Engineering only:**
```bash
python src/features/feature_runner.py
```

### Customize Execution

Edit `src/config/settings.py` to customize:
- Symbols to fetch
- Timeframes
- Lookback periods
- Validation thresholds
- Feature parameters

## Pipeline Stages

### 1. Data Ingestion

**Module:** `src/ingestion/equity_ohlcv.py`

- Fetches historical OHLCV data from Yahoo Finance
- Retrieves both raw and adjusted prices
- Normalizes all timestamps to UTC
- Stores raw data (immutable)

**Output:** `data/raw/{timeframe}/{symbol}_{timeframe}_raw.csv`

### 2. Data Validation

**Module:** `src/validation/ohlcv_checks.py`

Performs the following checks:

- **Price Relationships:** Validates high ≥ low, high ≥ close, etc.
- **Volume:** Ensures volume ≥ 0
- **Outliers:** Flags price changes > 20% (configurable)
- **Duplicates:** Detects duplicate timestamps
- **Missing Data:** Identifies gaps in expected timestamp sequence

**Outputs:**
- `data/validated/{timeframe}/{symbol}_{timeframe}_validated.csv` (with validation flags)
- `data/validated/{timeframe}/clean/{symbol}_{timeframe}_clean.csv` (clean data only)
- `data/validated/validation_log.csv` (issues log)

### 3. Feature Engineering

**Module:** `src/features/technical_indicators.py`

Computes technical indicators across four categories:

#### Trend Indicators
- Simple Moving Average (SMA): 10, 20, 50, 200 periods
- Exponential Moving Average (EMA): 12, 26 periods
- MACD (12, 26, 9)
- ADX (14)

#### Momentum Indicators
- RSI (14)
- Rate of Change (ROC): 5, 10, 20 periods
- Stochastic Oscillator (14, 3)

#### Volatility Indicators
- Average True Range (ATR): 14 periods
- Bollinger Bands (20, 2)
- Historical Volatility: 10, 20, 60 periods

#### Volume Indicators
- Volume Weighted Average Price (VWAP)
- On-Balance Volume (OBV)
- Volume Rate of Change

**Output:** `data/features/{timeframe}/{symbol}_{timeframe}_features.csv`

## Key Constraints

### Data Integrity
- ✅ Raw data remains **immutable** throughout pipeline
- ✅ All timestamps normalized to **UTC**
- ✅ No look-ahead bias - features use only past data
- ✅ Reproducible pipeline execution

### Scope Limitations
- ❌ No ML models or strategies
- ❌ No backtesting systems
- ❌ No database or external storage
- ❌ No data sources other than Yahoo Finance

## Output Data Structure

### Raw Data
```
timestamp, symbol, open, high, low, close, volume, 
adj_open, adj_high, adj_low, adj_close, adj_volume, dividends, stock splits
```

### Validated Data
```
Same as raw + validation flags:
valid_price_relationship, valid_volume, is_outlier, is_duplicate, has_gap, is_valid
```

### Feature Data
```
Same as validated + ~40 technical indicators
(SMA, EMA, MACD, RSI, ROC, ATR, Bollinger Bands, VWAP, OBV, etc.)
```

## Logging

All stages produce detailed logs including:
- Data fetch status
- Validation issues with timestamps and symbols
- Feature computation progress
- Error handling and warnings

Logs are output to console with timestamps and severity levels.

## Configuration

Key settings in `src/config/settings.py`:

```python
# Symbols
DEFAULT_EQUITY_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
DEFAULT_INDEX_SYMBOLS = ["^GSPC", "^DJI", "^IXIC"]

# Timeframes
TIMEFRAMES = ["1m", "5m", "1h", "1d"]

# Validation
VALIDATION_CONFIG = {
    "price_outlier_threshold": 0.20,  # 20%
    "max_gap_multiplier": 2.0,
}

# Features (configurable periods for all indicators)
FEATURE_CONFIG = { ... }
```

## Tech Stack

- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical computations
- **yfinance** - Yahoo Finance API client

## Error Handling

- Failed symbol fetches are logged and skipped
- Validation failures are logged but don't stop pipeline
- Each stage can run independently
- Failures in one stage don't corrupt downstream outputs

## Best Practices

1. Run ingestion during market hours for most recent data
2. Review validation logs before using data
3. Verify feature alignment before analysis
4. Keep raw data for re-processing if needed
5. Monitor for API rate limits from Yahoo Finance

## License

This project is for educational and research purposes.

## Data Disclaimer

Market data is provided by Yahoo Finance. This pipeline is not responsible for data accuracy or completeness. Always verify critical data from primary sources.
