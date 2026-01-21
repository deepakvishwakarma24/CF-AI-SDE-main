# Technical Indicators Guide - 60+ Features

## Overview
This pipeline now generates **approximately 60+ technical indicators** spanning four key categories that capture comprehensive market dynamics for professional trading strategy development.

---

## 📊 Feature Categories Summary

| Category | Count | Purpose |
|----------|-------|---------|
| **Trend Indicators** | ~20 | Direction and strength of price movement |
| **Momentum Indicators** | ~15 | Speed and strength of price changes |
| **Volatility Indicators** | ~12 | Price variability and risk measurement |
| **Volume Indicators** | ~8 | Volume flow and institutional activity |
| **Pattern Recognition** | ~7 | Candlestick patterns and formations |
| **Support/Resistance** | ~3 | Key price levels |
| **Fibonacci Levels** | ~4 | Retracement and reversal zones |
| **Price Features** | ~7 | Derived price characteristics |
| **OHLCV Base Data** | 6 | Raw market data |

**Total: 76+ features per record**

---

## 1️⃣ TREND INDICATORS (~20 features)

### Simple Moving Averages (SMA) - 5 features
Smooth price action and reveal support/resistance levels.

- **SMA 10**: Short-term trend (2 weeks)
- **SMA 20**: Primary swing trading MA
- **SMA 50**: Medium-term trend (2.5 months)
- **SMA 100**: Intermediate trend (5 months)
- **SMA 200**: Long-term trend (1 year) - Major S/R level

**Usage**: Price above SMA = bullish, below = bearish. Crossovers signal trend changes.

### Exponential Moving Averages (EMA) - 3 features
More responsive to recent price changes than SMA.

- **EMA 12**: Fast line for MACD
- **EMA 26**: Slow line for MACD
- **EMA 50**: Medium-term responsive trend

**Usage**: Weights recent prices more heavily, faster signals than SMA.

### MACD (Moving Average Convergence Divergence) - 3 features
Detects trend changes and momentum shifts.

- **MACD Line**: EMA(12) - EMA(26)
- **Signal Line**: EMA(9) of MACD Line
- **Histogram**: MACD Line - Signal Line

**Usage**: 
- MACD above signal = bullish momentum
- Histogram expanding = strengthening trend
- Centerline crossovers = trend changes

### Average Directional Index (ADX) - 1 feature
Quantifies trend strength (0-100 scale).

- **ADX 14**: Trend strength indicator

**Usage**: 
- ADX > 25 = Strong trend (trend-following strategies)
- ADX < 20 = Weak trend/ranging (mean reversion strategies)
- **Critical for regime detection**

### Moving Average Crossovers - 8 features
Relationships between price and moving averages.

- **price_vs_sma20**: % deviation from SMA 20
- **price_vs_sma50**: % deviation from SMA 50
- **price_vs_sma200**: % deviation from SMA 200
- **sma20_vs_sma50**: Short/medium MA relationship
- **sma50_vs_sma200**: Medium/long MA relationship
- **golden_cross**: SMA 50 crosses above SMA 200 (bullish)
- **death_cross**: SMA 50 crosses below SMA 200 (bearish)
- Plus trend angle indicators

**Usage**: Identify trend strength and multi-timeframe alignment.

---

## 2️⃣ MOMENTUM INDICATORS (~15 features)

### Relative Strength Index (RSI) - 4 features
Measures overbought/oversold conditions (0-100 scale).

- **RSI 7**: Short-term momentum
- **RSI 14**: Standard period (most common)
- **RSI 21**: Longer-term momentum

**Usage**: 
- RSI > 70 = Overbought (potential reversal)
- RSI < 30 = Oversold (potential bounce)
- **Ideal for mean reversion strategies**

### Rate of Change (ROC) - 3 features
Shows momentum strength at multiple periods.

- **ROC 5**: 1-week momentum
- **ROC 10**: 2-week momentum
- **ROC 20**: 1-month momentum

**Usage**: Measures % price change. Divergences signal reversals.

### Stochastic Oscillator - 2 features
Compares closing price to price range (0-100 scale).

- **%K**: Fast stochastic line
- **%D**: Slow stochastic (signal line)

**Usage**: 
- > 80 = Overbought
- < 20 = Oversold
- %K crossing %D = entry/exit signals

### Williams %R - 1 feature
Momentum oscillator measuring overbought/oversold (-100 to 0).

- **Williams R 14**: Inverse of stochastic

**Usage**: 
- > -20 = Overbought
- < -80 = Oversold

### Commodity Channel Index (CCI) - 1 feature
Identifies cyclical trends and extremes.

- **CCI 20**: Deviation from average price

**Usage**: 
- CCI > +100 = Overbought
- CCI < -100 = Oversold
- Cycles around zero

### Money Flow Index (MFI) - 1 feature
Volume-weighted RSI combining price and volume.

- **MFI 14**: "Volume RSI"

**Usage**: 
- MFI > 80 = Overbought with volume confirmation
- MFI < 20 = Oversold with volume confirmation
- More reliable than RSI alone

### Ultimate Oscillator - 1 feature
Multi-timeframe momentum combining 7, 14, and 28 periods.

- **Ultimate Oscillator**: Weighted average of 3 timeframes

**Usage**: 
- > 70 = Overbought
- < 30 = Oversold
- Reduces false signals by using multiple periods

---

## 3️⃣ VOLATILITY INDICATORS (~12 features)

### Bollinger Bands - 4 features
2 standard deviation envelope containing ~95% of price action.

- **BB Middle**: SMA 20 (middle band)
- **BB Upper**: SMA 20 + (2 × std dev)
- **BB Lower**: SMA 20 - (2 × std dev)
- **BB Width**: Upper - Lower (volatility measure)

**Usage**: 
- Wide bands = High volatility
- Narrow bands = Low volatility (squeeze)
- Price touching bands = potential reversal
- **Critical for volatility regime detection**

### Average True Range (ATR) - 4 features
Measures average price range including gaps.

- **ATR 7**: Short-term volatility
- **ATR 14**: Standard period (most common)
- **ATR 21**: Longer-term volatility

**Usage**: 
- **Indispensable for position sizing**
- **Essential for stop-loss placement**
- Higher ATR = wider stops needed
- ATR × 2 = common stop distance

### Historical Volatility - 3 features
Rolling standard deviation of log returns.

- **Hist Vol 10**: 2-week realized volatility
- **Hist Vol 20**: 1-month realized volatility
- **Hist Vol 60**: 3-month realized volatility

**Usage**: Shows realized volatility at different timescales. Compare to implied vol.

### Keltner Channels - 3 features
ATR-based price envelope (alternative to Bollinger Bands).

- **Keltner Middle**: EMA 20
- **Keltner Upper**: EMA 20 + (2 × ATR)
- **Keltner Lower**: EMA 20 - (2 × ATR)

**Usage**: 
- ATR-based (vs std dev for Bollinger)
- Price outside channel = strong trend
- Inside channel = range-bound

### Donchian Channels - 3 features
Highest high and lowest low over period.

- **Donchian Upper**: Highest high (20 periods)
- **Donchian Lower**: Lowest low (20 periods)
- **Donchian Middle**: Average of upper/lower

**Usage**: 
- Breakout trading (Turtle Trader strategy)
- Upper/lower define trading range
- Breakout = potential new trend

---

## 4️⃣ VOLUME INDICATORS (~8 features)

### VWAP (Volume Weighted Average Price) - 1 feature
Average price weighted by volume.

- **VWAP**: Institutional execution benchmark

**Usage**: 
- **Institutions aim to execute near VWAP**
- Price above VWAP = bullish
- Price below VWAP = bearish
- Intraday reset (daily calculation)

### On-Balance Volume (OBV) - 1 feature
Cumulative volume indicator confirming trends.

- **OBV**: Adds volume on up days, subtracts on down days

**Usage**: 
- Rising OBV with rising price = confirmed uptrend
- Falling OBV with rising price = divergence (warning)
- **Confirms price movements or signals divergences**

### Volume Rate of Change - 3 features
Compares current volume to historical volume.

- **Volume ROC 5**: 1-week volume change
- **Volume ROC 10**: 2-week volume change
- **Volume ROC 20**: 1-month volume change

**Usage**: **Identifies unusual activity** - potential breakouts or reversals.

### Accumulation/Distribution Line - 1 feature
Volume flow indicator based on close position in range.

- **A/D Line**: Cumulative volume flow

**Usage**: 
- Rising A/D = Accumulation (buying pressure)
- Falling A/D = Distribution (selling pressure)
- Divergence from price = warning signal

### Chaikin Money Flow (CMF) - 1 feature
Volume-weighted price momentum (20-period).

- **CMF 20**: Short-term money flow

**Usage**: 
- CMF > 0 = Buying pressure
- CMF < 0 = Selling pressure
- Confirms trend strength with volume

### Volume Weighted Moving Average (VWMA) - 1 feature
Moving average weighted by volume.

- **VWMA 20**: Volume-weighted MA

**Usage**: Gives more weight to high-volume periods.

---

## 5️⃣ PATTERN RECOGNITION (~7 features)

### Candlestick Patterns - 7 binary features
Encode common reversal and continuation patterns.

- **Doji**: Indecision (small body, equal shadows)
- **Hammer**: Bullish reversal (small body at top, long lower shadow)
- **Shooting Star**: Bearish reversal (small body at bottom, long upper shadow)
- **Engulfing Bullish**: Strong bullish reversal (engulfs previous red candle)
- **Engulfing Bearish**: Strong bearish reversal (engulfs previous green candle)
- **Morning Star**: 3-candle bullish reversal pattern
- **Evening Star**: 3-candle bearish reversal pattern

**Usage**: Binary (0/1) features indicating pattern presence. Combine with other indicators.

---

## 6️⃣ SUPPORT/RESISTANCE (~3 features)

### S/R Levels - 3 features
Identify price levels where reactions are likely.

- **distance_to_resistance**: Distance to nearest resistance
- **distance_to_support**: Distance to nearest support
- **support_resistance_ratio**: Relative position between S/R

**Usage**: 
- Calculated from local maxima/minima (20-period lookback)
- Price approaching S/R = potential reversal
- **Critical for entry/exit planning**

---

## 7️⃣ FIBONACCI RETRACEMENT (~4 features)

### Fibonacci Levels - 4 features
Predict potential reversal zones from swing highs/lows.

- **fib_236**: Distance to 23.6% retracement
- **fib_382**: Distance to 38.2% retracement (shallow)
- **fib_500**: Distance to 50% retracement (midpoint)
- **fib_618**: Distance to 61.8% retracement (golden ratio - strongest)

**Usage**: 
- Price near Fib level = potential reversal zone
- 61.8% most important (golden ratio)
- Combine with S/R for confluence

---

## 8️⃣ PRICE-BASED FEATURES (~7 features)

### Returns at Multiple Periods - 4 features
Log returns capturing price momentum.

- **return_1d**: Daily return
- **return_5d**: 1-week return
- **return_10d**: 2-week return
- **return_20d**: 1-month return

### Range and Position - 3 features

- **high_low_ratio**: Daily range relative to close
- **close_position**: Close position in daily range (0=low, 1=high)
- **gap**: Gap from previous close (overnight/weekend gap)

**Usage**: 
- Returns for momentum strategies
- Range for volatility assessment
- Close position for intraday bias
- Gaps for overnight risk

---

## 9️⃣ BASE OHLCV DATA (6 features)

- **timestamp**: Date/time of bar
- **open**: Opening price
- **high**: Highest price
- **low**: Lowest price
- **close**: Closing price
- **volume**: Trading volume

---

## 🎯 Total Feature Count by Category

```
Trend Indicators:        ~20 features
Momentum Indicators:     ~15 features
Volatility Indicators:   ~12 features
Volume Indicators:        ~8 features
Pattern Recognition:      ~7 features
Support/Resistance:       ~3 features
Fibonacci Levels:         ~4 features
Price Features:           ~7 features
Base OHLCV:               6 features
─────────────────────────────────────
TOTAL:                   ~76+ features
```

---

## 💡 Strategy Development Guidelines

### 1. **Regime Detection**
Use ADX, Bollinger Band Width, and Historical Volatility to detect market regime:

```python
# Trending vs Ranging
trending = df['adx_14'] > 25
ranging = df['adx_14'] < 20

# High vs Low Volatility
high_vol = df['bb_width'] > df['bb_width'].rolling(50).mean()
low_vol = df['bb_width'] < df['bb_width'].rolling(50).mean()
```

### 2. **Mean Reversion Strategies**
Use RSI, Stochastic, Bollinger Bands in ranging markets (ADX < 20):

```python
# Oversold bounce setup
oversold_signal = (
    (df['rsi_14'] < 30) & 
    (df['close'] < df['bb_lower']) &
    (df['adx_14'] < 20)
)
```

### 3. **Trend Following Strategies**
Use MACD, MA crossovers, ADX in trending markets (ADX > 25):

```python
# Strong uptrend entry
uptrend_signal = (
    (df['macd'] > df['macd_signal']) &
    (df['golden_cross'] == 1) &
    (df['adx_14'] > 25)
)
```

### 4. **Breakout Strategies**
Use Donchian Channels, volume indicators, ATR:

```python
# Breakout with volume confirmation
breakout_signal = (
    (df['close'] > df['donchian_upper']) &
    (df['volume_roc_5'] > 50) &
    (df['obv'] > df['obv'].shift(1))
)
```

### 5. **Volume Confirmation**
Always confirm price signals with volume:

```python
# Price up, volume up = healthy move
confirmed_move = (
    (df['close'] > df['close'].shift(1)) &
    (df['obv'] > df['obv'].shift(1)) &
    (df['cmf_20'] > 0)
)
```

### 6. **Multi-Timeframe Analysis**
Combine daily for trend, hourly for entry:

```python
# Daily uptrend, hourly pullback
daily_uptrend = daily_df['sma_50'] > daily_df['sma_200']
hourly_pullback = hourly_df['rsi_14'] < 40
```

### 7. **Risk Management with ATR**
Use ATR for position sizing and stops:

```python
# Stop loss: 2× ATR below entry
stop_distance = 2 * df['atr_14']
position_size = risk_per_trade / stop_distance

# Profit target: 3× ATR above entry
profit_target = 3 * df['atr_14']
```

---

## 📊 Feature Importance for ML Models

### High-Importance Features (Start Here)
1. **Returns** (return_1d, return_5d, return_20d)
2. **RSI** (rsi_14, rsi_7)
3. **MACD** (macd, macd_histogram)
4. **Volume** (obv, volume_roc_5, cmf_20)
5. **Volatility** (atr_14, bb_width)
6. **Trend** (adx_14, sma_50_vs_sma_200)

### Medium-Importance Features
7. Moving averages (sma_20, sma_50, ema_12, ema_26)
8. Stochastic (%K, %D)
9. Bollinger Bands (bb_upper, bb_lower)
10. Support/Resistance distances

### Experimental Features
11. Candlestick patterns
12. Fibonacci levels
13. Price position features

---

## 🚀 Getting Started

### Load Features
```python
import pandas as pd

# Load 10 years of daily data with all 76+ features
df = pd.read_csv('data/features/1d/AAPL_1d_features.csv')

print(f"Total features: {len(df.columns)}")
print(f"Total records: {len(df)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Check for any features
print("\nTrend indicators:", [c for c in df.columns if 'sma' in c or 'ema' in c])
print("Momentum indicators:", [c for c in df.columns if 'rsi' in c or 'macd' in c])
print("Volatility indicators:", [c for c in df.columns if 'atr' in c or 'bb' in c])
print("Volume indicators:", [c for c in df.columns if 'obv' in c or 'volume' in c])
```

### Quick Analysis
```python
# Latest indicator values
latest = df.iloc[-1]

print(f"\n📈 {latest['symbol']} - Latest Indicators")
print(f"Price: ${latest['close']:.2f}")
print(f"RSI: {latest['rsi_14']:.1f}")
print(f"MACD: {latest['macd']:.4f}")
print(f"ADX: {latest['adx_14']:.1f} ({'Trending' if latest['adx_14'] > 25 else 'Ranging'})")
print(f"ATR: {latest['atr_14']:.2f}")
```

---

## 📚 References & Further Reading

- **Trend**: "Technical Analysis of the Financial Markets" - John Murphy
- **Momentum**: "New Concepts in Technical Trading Systems" - J. Welles Wilder
- **Volatility**: "Bollinger on Bollinger Bands" - John Bollinger
- **Volume**: "Secrets of Volume Trading" - Buff Dormeier
- **Patterns**: "Japanese Candlestick Charting Techniques" - Steve Nison

---

**You now have a professional-grade technical analysis toolkit with 76+ features ready for serious trading strategy development!** 🎯📊
