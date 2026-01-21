# Liquid Instruments Trading Pipeline - Configuration Guide

## Overview
This pipeline is now configured for **professional trading strategy development** with focus on:
- **Liquid instruments** with high trading volume and tight spreads
- **10 years of historical data** (daily) to capture multiple market regimes
- **36 default instruments** across indices and diversified sectors
- **30+ technical indicators** for comprehensive analysis

---

## 🎯 Core Assets - Major Indices (6)

| Symbol | Name | Market | Description |
|--------|------|--------|-------------|
| ^GSPC | S&P 500 | US | Large-cap benchmark, 500 companies |
| ^IXIC | NASDAQ Composite | US | Technology-heavy index |
| ^DJI | Dow Jones Industrial | US | 30 blue-chip companies |
| ^NSEI | NIFTY 50 | India | Top 50 companies on NSE |
| ^NSEBANK | BANKNIFTY | India | Banking sector index |
| ^VIX | Volatility Index | US | Market fear gauge |

---

## 📈 Liquid Individual Stocks (30)

### Technology Sector (7 stocks)
- **AAPL** - Apple Inc. (Consumer Electronics)
- **MSFT** - Microsoft (Software/Cloud)
- **GOOGL** - Alphabet (Search/Advertising)
- **AMZN** - Amazon (E-commerce/Cloud)
- **NVDA** - NVIDIA (Semiconductors/AI)
- **META** - Meta Platforms (Social Media)
- **ORCL** - Oracle (Enterprise Software)

### Financial Services (5 stocks)
- **JPM** - JPMorgan Chase (Banking)
- **BAC** - Bank of America (Banking)
- **GS** - Goldman Sachs (Investment Banking)
- **V** - Visa (Payments)
- **MA** - Mastercard (Payments)

### Healthcare (4 stocks)
- **JNJ** - Johnson & Johnson (Pharmaceuticals)
- **UNH** - UnitedHealth (Health Insurance)
- **PFE** - Pfizer (Pharmaceuticals)
- **ABBV** - AbbVie (Biotechnology)

### Consumer (4 stocks)
- **WMT** - Walmart (Retail)
- **PG** - Procter & Gamble (Consumer Goods)
- **KO** - Coca-Cola (Beverages)
- **MCD** - McDonald's (Fast Food)

### Energy (3 stocks)
- **XOM** - Exxon Mobil (Oil & Gas)
- **CVX** - Chevron (Oil & Gas)
- **COP** - ConocoPhillips (Oil & Gas)

### Industrials (3 stocks)
- **BA** - Boeing (Aerospace)
- **CAT** - Caterpillar (Heavy Machinery)
- **GE** - General Electric (Conglomerate)

### Communication Services (2 stocks)
- **DIS** - Disney (Media/Entertainment)
- **NFLX** - Netflix (Streaming)

### Automotive/EVs (2 stocks)
- **TSLA** - Tesla (Electric Vehicles)
- **F** - Ford (Automotive)

---

## ⏰ Historical Data Coverage

### Timeframe Specifications

| Timeframe | History | Use Case | Records (approx) |
|-----------|---------|----------|------------------|
| **1d** | 10 years | Backtesting, long-term analysis | ~2,500 bars |
| **1h** | 2 years | Swing trading, medium-term | ~3,500 bars |
| **5m** | 60 days | Intraday strategies | ~4,300 bars |
| **1m** | 7 days | High-frequency patterns | ~2,700 bars |

### Market Regimes Captured (10-year daily data)
- ✅ **2014-2015**: Post-crisis bull market
- ✅ **2016**: Brexit volatility
- ✅ **2018**: December correction (-20%)
- ✅ **2020**: COVID-19 crash and recovery (-35% to +100%)
- ✅ **2022**: Inflation/rate hike bear market
- ✅ **2023-2024**: Recovery and consolidation
- ✅ Multiple sideways/ranging periods

---

## 🔧 Technical Indicators (33 Features)

### Trend Indicators (10)
- SMA: 10, 20, 50, 200 periods
- EMA: 12, 26 periods
- MACD: Line, Signal, Histogram
- ADX: 14 periods

### Momentum Indicators (8)
- RSI: 14 periods
- ROC: 5, 10, 20 periods
- Stochastic: %K and %D
- Volume ROC: 5, 10, 20 periods

### Volatility Indicators (7)
- ATR: 14 periods
- Bollinger Bands: Middle, Upper, Lower, Width
- Historical Volatility: 10, 20, 60 periods

### Volume Indicators (2)
- VWAP (Volume Weighted Average Price)
- OBV (On-Balance Volume)

### Base Data (6)
- Open, High, Low, Close, Volume, Timestamp

**Total: 39 columns per record**

---

## 🚀 Quick Start

### Run with All Default Instruments (36 symbols, 10 years daily data)
```bash
python main.py
# Press Enter when prompted for symbols
# Enter "1d" for timeframe (recommended for initial run)
```

### Run Specific Instruments
```bash
# Interactive mode
python main.py
# Then enter: AAPL, MSFT, ^GSPC, ^NSEI

# CLI mode
python main.py --symbols=AAPL,MSFT,^GSPC --timeframes=1d
```

### Programmatic Usage
```python
from main import run_full_pipeline

# All 36 default instruments, daily data, 10 years
results = run_full_pipeline(
    symbols=None,  # None = all defaults
    timeframes=['1d'],
    save_to_file=True
)

# Specific instruments
results = run_full_pipeline(
    symbols=['AAPL', 'MSFT', '^GSPC', '^NSEI'],
    timeframes=['1h', '1d'],
    save_to_file=True
)
```

---

## 📊 Expected Output

### File Structure
```
data/
├── raw/
│   └── 1d/
│       ├── AAPL_1d_raw.csv       (~2,500 rows)
│       ├── ^GSPC_1d_raw.csv      (~2,500 rows)
│       └── ... (36 files)
├── validated/
│   └── 1d/
│       └── clean/
│           ├── AAPL_1d_clean.csv
│           └── ... (36 files)
└── features/
    └── 1d/
        ├── AAPL_1d_features.csv  (39 columns, ~2,500 rows)
        ├── ^GSPC_1d_features.csv
        └── ... (36 files)
```

### Data Volume Estimates
- **Daily (1d)**: 36 symbols × 2,500 rows = 90,000 total records
- **Storage**: ~150-200 MB total for all timeframes
- **Processing time**: 5-15 minutes for all 36 symbols (all timeframes)

---

## 💡 Strategy Development Tips

### 1. Start with Daily Data
```python
# Load 10 years of S&P 500 with indicators
import pandas as pd
df = pd.read_csv('data/features/1d/^GSPC_1d_features.csv')

# You now have:
# - 2,500+ days of data
# - Multiple market regimes
# - 33 technical indicators
# - Ready for backtesting
```

### 2. Sector Diversification
Test strategies across different sectors to avoid overfitting:
- Tech: AAPL, MSFT, NVDA
- Finance: JPM, V, MA
- Consumer: WMT, KO, PG
- Energy: XOM, CVX

### 3. Index Correlation Analysis
Compare individual stocks against indices:
- Stock vs ^GSPC (broad market)
- Tech stocks vs ^IXIC (sector)
- Check ^VIX for volatility regime

### 4. Multi-Timeframe Analysis
```python
# Daily for trend
daily_df = pd.read_csv('data/features/1d/AAPL_1d_features.csv')

# Hourly for entry/exit
hourly_df = pd.read_csv('data/features/1h/AAPL_1h_features.csv')
```

---

## 🎓 Why These Instruments?

### Liquidity Criteria
✅ High average daily volume (millions of shares)  
✅ Tight bid-ask spreads (< 0.1%)  
✅ Market cap > $50B (for stocks)  
✅ Options available (for hedging)  
✅ Minimal slippage in live trading  

### Sector Diversification
✅ 8 different sectors  
✅ Uncorrelated during certain periods  
✅ Different risk/return profiles  
✅ Balanced exposure to economic cycles  

### Historical Depth
✅ 10 years = ~2,500 trading days  
✅ Captures 3+ complete market cycles  
✅ Multiple crashes and recoveries  
✅ Sufficient data for ML/AI training  
✅ Statistical significance for backtesting  

---

## 📝 Notes

- **Indian Indices**: ^NSEI and ^NSEBANK may have limited data availability on Yahoo Finance. Verify data quality after first run.
- **Intraday Limits**: Yahoo Finance restricts 1m data to 7 days, 5m to 60 days
- **Data Gaps**: Weekends, holidays, and market closures create natural gaps
- **Validation**: Pipeline automatically validates and cleans data, logs issues to `data/validated/validation_log.csv`

---

## 🔄 Next Steps

1. **Initial Run**: Start with daily data for all 36 instruments
   ```bash
   python main.py
   # Press Enter for all symbols
   # Enter "1d" for timeframe
   ```

2. **Verify Data Quality**: Check validation log
   ```bash
   cat data/validated/validation_log.csv
   ```

3. **Explore Features**: Load and analyze
   ```python
   import pandas as pd
   df = pd.read_csv('data/features/1d/AAPL_1d_features.csv')
   print(df.info())
   print(df.describe())
   ```

4. **Build Strategies**: Use the 10 years of data with 33 indicators for your backtesting framework

---

**Ready to build robust trading strategies with professional-grade data!** 🚀
