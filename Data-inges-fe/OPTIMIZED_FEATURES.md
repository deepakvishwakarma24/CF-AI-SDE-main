# Optimized Feature Set (60-80 Features)

## Summary
**Total Raw Features: ~70**
**Total with Normalization: ~140 columns (70 raw + 70 normalized)**

This optimized configuration reduces redundancy while keeping the most impactful features for ML models.

---

## Feature Breakdown

### 1. **OHLCV Base Data (5 columns)**
- `open`, `high`, `low`, `close`, `volume`

### 2. **Trend Indicators (10 features)**
- **SMA (3):** `sma_20`, `sma_50`, `sma_200` *(removed: sma_10, sma_100)*
- **EMA (2):** `ema_12`, `ema_26` *(removed: ema_50)*
- **MACD (3):** `macd`, `macd_signal`, `macd_histogram`
- **ADX (1):** `adx_14`
- **MA Crossovers (3):** `price_vs_sma20`, `golden_cross`, `death_cross` *(removed: price_vs_sma50, price_vs_sma200, sma20_vs_sma50, sma50_vs_sma200)*

**Reduction:** 20 → 10 features (50% reduction)

---

### 3. **Momentum Indicators (8 features)**
- **RSI (1):** `rsi_14` *(removed: rsi_7, rsi_21 - redundant multi-period)*
- **Stochastic (2):** `stoch_k`, `stoch_d`
- **ROC (2):** `roc_5`, `roc_20` *(removed: roc_10)*
- **CCI (1):** `cci_20`
- **MFI (1):** `mfi_14`
- **Williams %R (1):** `williams_r_14`
- **Ultimate Oscillator (0):** REMOVED *(redundant with other momentum indicators)*

**Reduction:** 15 → 8 features (47% reduction)

---

### 4. **Volatility Indicators (7 features)**
- **Bollinger Bands (4):** `bb_middle`, `bb_upper`, `bb_lower`, `bb_width`
- **ATR (1):** `atr_14` *(removed: atr_7, atr_21 - multi-period)*
- **Historical Volatility (2):** `hist_vol_20`, `hist_vol_60` *(removed: hist_vol_10)*
- **Keltner Channels (0):** REMOVED *(redundant with Bollinger Bands)*
- **Donchian Channels (0):** REMOVED *(redundant with Bollinger Bands)*

**Reduction:** 15 → 7 features (53% reduction)

---

### 5. **Volume Indicators (5 features)**
- `vwap` - Volume Weighted Average Price
- `obv` - On-Balance Volume
- **Volume ROC (1):** `volume_roc_10` *(removed: volume_roc_5, volume_roc_20)*
- `ad_line` - Accumulation/Distribution Line
- `cmf_20` - Chaikin Money Flow
- **VWMA (0):** REMOVED *(redundant with VWAP)*

**Reduction:** 8 → 5 features (38% reduction)

---

### 6. **Candlestick Patterns (5 features)**
- `doji` - Indecision
- `hammer` - Bullish reversal
- `engulfing_bullish`, `engulfing_bearish` - Strong reversals
- `morning_star` - Bullish reversal
- `evening_star` - Bearish reversal
- **Removed:** `shooting_star` *(less reliable)*

**Reduction:** 7 → 5 features (29% reduction)

---

### 7. **Support/Resistance (3 features)**
- `distance_to_resistance`
- `distance_to_support`
- `support_resistance_ratio`

**No change:** 3 features

---

### 8. **Fibonacci Levels (0 features)**
- **DISABLED** - Less critical for ML models
- *(Removed: fib_236, fib_382, fib_500, fib_618)*

**Reduction:** 4 → 0 features (100% reduction)

---

### 9. **Price Features (5 features)**
- **Returns (3):** `return_1d`, `return_5d`, `return_20d` *(removed: return_10d)*
- `high_low_ratio` - Daily range
- `close_position` - Position in daily range
- `gap` - Gap from previous close

**Reduction:** 7 → 5 features (29% reduction)

---

### 10. **Indicator Slopes (16 features)**
*4 indicators × 2 periods × 2 types (slope + rising) = 16*

**Indicators (4):** `rsi_14`, `macd`, `stoch_k`, `cci_20`
- *(Removed: adx_14, stoch_d)*

**Periods (2):** 5, 10
- *(Removed: period 3)*

**Features:**
- `rsi_14_slope_5`, `rsi_14_slope_10`, `rsi_14_rising_5`, `rsi_14_rising_10`
- `macd_slope_5`, `macd_slope_10`, `macd_rising_5`, `macd_rising_10`
- `stoch_k_slope_5`, `stoch_k_slope_10`, `stoch_k_rising_5`, `stoch_k_rising_10`
- `cci_20_slope_5`, `cci_20_slope_10`, `cci_20_rising_5`, `cci_20_rising_10`

**Reduction:** 36 → 16 features (56% reduction)

---

### 11. **Crossover Signals (8 features)**
- **MACD (2):** `macd_cross_above`, `macd_cross_below` *(removed: zero line crosses)*
- **Stochastic (2):** `stoch_cross_above`, `stoch_cross_below`
- **RSI (2):** `rsi_cross_above_30`, `rsi_cross_below_70` *(removed: 50-level crosses)*
- **Bollinger Bands (2):** `price_cross_bb_upper`, `price_cross_bb_lower`
- **EMA (0):** REMOVED *(redundant with MACD crossovers)*

**Reduction:** 14 → 8 features (43% reduction)

---

### 12. **Normalized Features (70 features)**
*All 70 raw features duplicated with `_norm` suffix*
- Z-score normalization: `(x - mean) / std`
- Examples: `rsi_14_norm`, `macd_norm`, `sma_20_norm`, etc.

---

## Total Feature Count Summary

| Category | Original | Optimized | Reduction |
|----------|----------|-----------|-----------|
| OHLCV Base | 5 | 5 | 0% |
| Trend Indicators | 20 | 10 | 50% |
| Momentum Indicators | 15 | 8 | 47% |
| Volatility Indicators | 15 | 7 | 53% |
| Volume Indicators | 8 | 5 | 38% |
| Candlestick Patterns | 7 | 5 | 29% |
| Support/Resistance | 3 | 3 | 0% |
| Fibonacci Levels | 4 | 0 | 100% |
| Price Features | 7 | 5 | 29% |
| Indicator Slopes | 36 | 16 | 56% |
| Crossover Signals | 14 | 8 | 43% |
| **Subtotal (Raw)** | **~128** | **~70** | **45%** |
| Normalized Features | 128 | 70 | 45% |
| **TOTAL COLUMNS** | **~269** | **~145** | **46%** |

---

## Key Optimization Decisions

### ✅ What We Kept (High Impact)
1. **Core trend indicators**: SMA 20/50/200, EMA 12/26, MACD, ADX
2. **Best momentum**: RSI-14, Stochastic, MFI, CCI
3. **Essential volatility**: Bollinger Bands, ATR-14, Historical Vol
4. **Critical volume**: VWAP, OBV, A/D Line, CMF
5. **Key crossovers**: MACD, Stochastic, RSI 30/70, BB touches
6. **Important slopes**: RSI, MACD, Stochastic, CCI (momentum direction)
7. **Multi-horizon returns**: 1d, 5d, 20d (short/medium/long term)

### ❌ What We Removed (Redundant/Low Impact)
1. **Multi-period redundancy**: Multiple ATR/RSI/ROC periods
2. **Redundant channels**: Keltner, Donchian (BB is sufficient)
3. **Redundant averages**: VWMA (VWAP covers this)
4. **Extra MAs**: SMA 10/100, EMA 50
5. **Fibonacci levels**: Less predictive for ML models
6. **Ultimate Oscillator**: Covered by other momentum indicators
7. **Extra crossovers**: RSI-50, MACD zero-line, EMA crosses (less critical)
8. **Extra slopes**: ADX, Stochastic-D (redundant with kept features)

---

## Benefits of Optimization

1. **Faster Training**: 46% fewer features = faster model training
2. **Less Overfitting**: Fewer redundant features reduce multicollinearity
3. **Better Generalization**: Focus on most predictive features
4. **Easier Interpretation**: Smaller feature set is easier to analyze
5. **Lower Memory**: ~46% reduction in storage requirements
6. **Maintained Coverage**: Still covers all critical market aspects:
   - ✅ Trend (MAs, MACD, ADX)
   - ✅ Momentum (RSI, Stochastic, CCI, MFI)
   - ✅ Volatility (BB, ATR, Hist Vol)
   - ✅ Volume (VWAP, OBV, CMF)
   - ✅ Patterns (Candlesticks, S/R)
   - ✅ Derivatives (Slopes, Crossovers)

---

## Next Steps

1. **Run the pipeline**: Test with optimized features
   ```bash
   python3 main.py
   ```

2. **Verify feature count**: Should see ~70 raw + 70 normalized = ~145 total columns

3. **Compare performance**: 
   - Train ML model with original 128 features
   - Train ML model with optimized 70 features
   - Compare accuracy, training time, overfitting

4. **Fine-tune if needed**: Add back specific features if performance drops significantly

---

## Configuration Files Modified

1. **`src/config/settings.py`**:
   - Reduced SMA periods: [10,20,50,100,200] → [20,50,200]
   - Reduced EMA periods: [12,26,50] → [12,26]
   - Disabled multi-period RSI/ATR
   - Removed Keltner, Donchian, Fibonacci
   - Reduced slope periods: [3,5,10] → [5,10]
   - Reduced return periods: [1,5,10,20] → [1,5,20]

2. **`src/features/technical_indicators.py`**:
   - Updated `ma_crossovers()`: 7 → 3 features
   - Updated `calculate_indicator_slopes()`: 6 → 4 indicators
   - Updated `calculate_additional_crossovers()`: 14 → 8 features

---

**Date:** 2026-01-20  
**Status:** ✅ OPTIMIZED - Ready for testing
