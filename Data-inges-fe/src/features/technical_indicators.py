"""
Feature engineering module for technical indicators.
Computes trend, momentum, volatility, and volume indicators.
All features are computed using only past data to prevent look-ahead bias.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Computes technical indicators for financial time series data.
    All calculations respect causality - only past data is used.
    """
    
    def __init__(self):
        """Initialize the technical indicators calculator."""
        pass
    
    # ==================== TREND INDICATORS ====================
    
    def sma(self, df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """
        Simple Moving Average.
        
        Args:
            df: DataFrame with price data
            period: Number of periods for SMA
            column: Column to calculate SMA on
            
        Returns:
            Series with SMA values
        """
        return df[column].rolling(window=period, min_periods=period).mean()
    
    def ema(self, df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """
        Exponential Moving Average.
        
        Args:
            df: DataFrame with price data
            period: Number of periods for EMA
            column: Column to calculate EMA on
            
        Returns:
            Series with EMA values
        """
        return df[column].ewm(span=period, adjust=False, min_periods=period).mean()
    
    def macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, pd.Series]:
        """
        Moving Average Convergence Divergence (MACD).
        
        Args:
            df: DataFrame with price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        ema_fast = self.ema(df, fast)
        ema_slow = self.ema(df, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }
    
    def adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average Directional Index (ADX).
        Measures trend strength.
        
        Args:
            df: DataFrame with OHLC data
            period: Number of periods
            
        Returns:
            Series with ADX values
        """
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth the True Range and Directional Movements
        atr = tr.rolling(window=period, min_periods=period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).mean() / atr
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period, min_periods=period).mean()
        
        return adx
    
    # ==================== MOMENTUM INDICATORS ====================
    
    def rsi(self, df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """
        Relative Strength Index (RSI).
        
        Args:
            df: DataFrame with price data
            period: Number of periods for RSI
            column: Column to calculate RSI on
            
        Returns:
            Series with RSI values
        """
        delta = df[column].diff()
        
        gain = delta.where(delta > 0, 0)  # type: ignore
        loss = -delta.where(delta < 0, 0)  # type: ignore
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def roc(self, df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """
        Rate of Change (ROC).
        
        Args:
            df: DataFrame with price data
            period: Number of periods for ROC
            column: Column to calculate ROC on
            
        Returns:
            Series with ROC values
        """
        roc = ((df[column] - df[column].shift(period)) / df[column].shift(period)) * 100
        return roc
    
    def stochastic_oscillator(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """
        Stochastic Oscillator.
        
        Args:
            df: DataFrame with OHLC data
            k_period: Period for %K line
            d_period: Period for %D line (signal)
            
        Returns:
            Dictionary with %K and %D values
        """
        low_min = df['low'].rolling(window=k_period, min_periods=k_period).min()
        high_max = df['high'].rolling(window=k_period, min_periods=k_period).max()
        
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period, min_periods=d_period).mean()
        
        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }
    
    def williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Williams %R - Momentum oscillator measuring overbought/oversold levels.
        
        Args:
            df: DataFrame with OHLC data
            period: Number of periods
            
        Returns:
            Series with Williams %R values (-100 to 0)
        """
        highest_high = df['high'].rolling(window=period, min_periods=period).max()
        lowest_low = df['low'].rolling(window=period, min_periods=period).min()
        
        williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        
        return williams_r
    
    def cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index (CCI) - Overbought/oversold indicator.
        
        Args:
            df: DataFrame with OHLC data
            period: Number of periods
            
        Returns:
            Series with CCI values
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period, min_periods=period).mean()
        mean_deviation = typical_price.rolling(window=period, min_periods=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    def mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Money Flow Index (MFI) - Volume-weighted RSI.
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods
            
        Returns:
            Series with MFI values (0-100)
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        # Positive and negative money flow
        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)
        
        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = money_flow.iloc[i]
        
        positive_mf = positive_flow.rolling(window=period, min_periods=period).sum()
        negative_mf = negative_flow.rolling(window=period, min_periods=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        
        return mfi
    
    def ultimate_oscillator(
        self,
        df: pd.DataFrame,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28
    ) -> pd.Series:
        """
        Ultimate Oscillator - Multi-timeframe momentum indicator.
        
        Args:
            df: DataFrame with OHLC data
            period1: Short period
            period2: Medium period
            period3: Long period
            
        Returns:
            Series with Ultimate Oscillator values (0-100)
        """
        # Calculate buying pressure
        bp = df['close'] - pd.concat([df['low'], df['close'].shift(1)], axis=1).min(axis=1)
        
        # Calculate true range
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate averages for each period
        avg1 = bp.rolling(window=period1, min_periods=period1).sum() / tr.rolling(window=period1, min_periods=period1).sum()
        avg2 = bp.rolling(window=period2, min_periods=period2).sum() / tr.rolling(window=period2, min_periods=period2).sum()
        avg3 = bp.rolling(window=period3, min_periods=period3).sum() / tr.rolling(window=period3, min_periods=period3).sum()
        
        # Weighted average
        uo = 100 * ((4 * avg1 + 2 * avg2 + avg3) / 7)
        
        return uo
    
    # ==================== VOLATILITY INDICATORS ====================
    
    def atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR).
        
        Args:
            df: DataFrame with OHLC data
            period: Number of periods
            
        Returns:
            Series with ATR values
        """
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        
        return atr
    
    def bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = 'close'
    ) -> Dict[str, pd.Series]:
        """
        Bollinger Bands.
        
        Args:
            df: DataFrame with price data
            period: Number of periods for moving average
            std_dev: Number of standard deviations
            column: Column to calculate bands on
            
        Returns:
            Dictionary with middle band, upper band, and lower band
        """
        middle_band = self.sma(df, period, column)
        std = df[column].rolling(window=period, min_periods=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return {
            'bb_middle': middle_band,
            'bb_upper': upper_band,
            'bb_lower': lower_band,
            'bb_width': upper_band - lower_band
        }
    
    def historical_volatility(
        self,
        df: pd.DataFrame,
        period: int,
        column: str = 'close'
    ) -> pd.Series:
        """
        Historical Volatility (standard deviation of returns).
        
        Args:
            df: DataFrame with price data
            period: Number of periods
            column: Column to calculate volatility on
            
        Returns:
            Series with historical volatility values
        """
        returns = df[column].pct_change()
        volatility = returns.rolling(window=period, min_periods=period).std() * np.sqrt(252)
        
        return volatility
    
    def keltner_channels(
        self,
        df: pd.DataFrame,
        period: int = 20,
        atr_period: int = 14,
        multiplier: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        Keltner Channels - ATR-based price envelope.
        
        Args:
            df: DataFrame with OHLC data
            period: EMA period for middle line
            atr_period: ATR calculation period
            multiplier: ATR multiplier for bands
            
        Returns:
            Dictionary with middle, upper, and lower channels
        """
        middle = self.ema(df, period)
        atr_values = self.atr(df, atr_period)
        
        upper = middle + (multiplier * atr_values)
        lower = middle - (multiplier * atr_values)
        
        return {
            'keltner_middle': middle,
            'keltner_upper': upper,
            'keltner_lower': lower
        }
    
    def donchian_channels(self, df: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
        """
        Donchian Channels - Highest high and lowest low over period.
        
        Args:
            df: DataFrame with OHLC data
            period: Lookback period
            
        Returns:
            Dictionary with upper and lower channels
        """
        upper = df['high'].rolling(window=period, min_periods=period).max()
        lower = df['low'].rolling(window=period, min_periods=period).min()
        middle = (upper + lower) / 2
        
        return {
            'donchian_upper': upper,
            'donchian_lower': lower,
            'donchian_middle': middle
        }
    
    # ==================== VOLUME INDICATORS ====================
    
    def vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Volume Weighted Average Price (VWAP).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with VWAP values
        """
        q = df['volume'] * (df['close'] - df['close'].shift(1))
        v = df['volume'].shift(1)
        vwap = q.cumsum() / v.cumsum()
        
        return vwap
    
    def obv(self, df: pd.DataFrame) -> pd.Series:
        """
        On-Balance Volume (OBV).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with OBV values
        """
        obv = df['volume'].where(df['close'] > df['close'].shift(1), -df['volume']).cumsum()
        return obv
    
    def volume_roc(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Volume Rate of Change (ROC).
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods
            
        Returns:
            Series with volume ROC values
        """
        return ((df['volume'] - df['volume'].shift(period)) / df['volume'].shift(period)) * 100
    
    def ad_line(self, df: pd.DataFrame) -> pd.Series:
        """
        Accumulation/Distribution Line - Volume flow indicator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with A/D Line values
        """
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)  # Handle division by zero when high == low
        ad = (clv * df['volume']).cumsum()
        
        return ad
    
    def cmf(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Chaikin Money Flow - Volume-weighted price momentum.
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods
            
        Returns:
            Series with CMF values
        """
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.fillna(0)
        mfv = mfm * df['volume']
        
        cmf = mfv.rolling(window=period, min_periods=period).sum() / df['volume'].rolling(window=period, min_periods=period).sum()
        
        return cmf
    
    def vwma(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Volume Weighted Moving Average.
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods
            
        Returns:
            Series with VWMA values
        """
        vwma = (df['close'] * df['volume']).rolling(window=period, min_periods=period).sum() / \
               df['volume'].rolling(window=period, min_periods=period).sum()
        
        return vwma
    
    # ==================== PATTERN RECOGNITION ====================
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Detect common candlestick patterns.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dictionary with binary indicators for each pattern
        """
        patterns = {}
        
        # Calculate body and shadows
        body = abs(df['close'] - df['open'])
        body_avg = body.rolling(window=10, min_periods=10).mean()
        range_hl = df['high'] - df['low']
        
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        
        # Doji - Small body relative to shadows (indecision)
        patterns['doji'] = (body <= body_avg * 0.1).astype(int)
        
        # Hammer - Small body at top, long lower shadow (bullish reversal)
        patterns['hammer'] = (
            (lower_shadow > body * 2) & 
            (upper_shadow < body * 0.3) & 
            (df['close'] > df['open'])
        ).astype(int)
        
        # Shooting Star - Small body at bottom, long upper shadow (bearish reversal)
        patterns['shooting_star'] = (
            (upper_shadow > body * 2) & 
            (lower_shadow < body * 0.3) & 
            (df['close'] < df['open'])
        ).astype(int)
        
        # Bullish Engulfing - Large bullish candle engulfs previous bearish
        patterns['engulfing_bullish'] = (
            (df['close'] > df['open']) &
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1))
        ).astype(int)
        
        # Bearish Engulfing
        patterns['engulfing_bearish'] = (
            (df['close'] < df['open']) &
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1))
        ).astype(int)
        
        # Morning Star - 3-candle bullish reversal pattern
        patterns['morning_star'] = (
            (df['close'].shift(2) < df['open'].shift(2)) &  # First: bearish
            (body.shift(1) < body_avg.shift(1) * 0.3) &     # Second: small body (star)
            (df['close'] > df['open']) &                     # Third: bullish
            (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2)  # Closes above midpoint
        ).astype(int)
        
        # Evening Star - 3-candle bearish reversal pattern
        patterns['evening_star'] = (
            (df['close'].shift(2) > df['open'].shift(2)) &  # First: bullish
            (body.shift(1) < body_avg.shift(1) * 0.3) &     # Second: small body (star)
            (df['close'] < df['open']) &                     # Third: bearish
            (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2)  # Closes below midpoint
        ).astype(int)
        
        return patterns
    
    # ==================== SUPPORT/RESISTANCE ====================
    
    def support_resistance_levels(
        self,
        df: pd.DataFrame,
        lookback: int = 20,
        num_levels: int = 3
    ) -> Dict[str, pd.Series]:
        """
        Identify support and resistance levels from local extrema.
        
        Args:
            df: DataFrame with OHLC data
            lookback: Lookback period for finding extrema
            num_levels: Number of S/R levels to track
            
        Returns:
            Dictionary with distances to nearest support/resistance
        """
        # Find local maxima (resistance) and minima (support)
        high_rolling = df['high'].rolling(window=lookback, center=True, min_periods=lookback)
        low_rolling = df['low'].rolling(window=lookback, center=True, min_periods=lookback)
        
        resistance = df['close'] - df['high'].rolling(window=lookback, min_periods=lookback).max()
        support = df['close'] - df['low'].rolling(window=lookback, min_periods=lookback).min()
        
        return {
            'distance_to_resistance': resistance,
            'distance_to_support': support,
            'support_resistance_ratio': resistance / (support.abs() + 1e-10)
        }
    
    # ==================== FIBONACCI RETRACEMENT ====================
    
    def fibonacci_levels(
        self,
        df: pd.DataFrame,
        lookback: int = 50,
        levels: list = [0.236, 0.382, 0.500, 0.618]
    ) -> Dict[str, pd.Series]:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            df: DataFrame with OHLC data
            lookback: Period to find swing high/low
            levels: Fibonacci levels to calculate
            
        Returns:
            Dictionary with distances to Fibonacci levels
        """
        # Find swing high and low
        swing_high = df['high'].rolling(window=lookback, min_periods=lookback).max()
        swing_low = df['low'].rolling(window=lookback, min_periods=lookback).min()
        diff = swing_high - swing_low
        
        fib_features = {}
        
        for level in levels:
            fib_level = swing_high - (diff * level)
            fib_features[f'fib_{int(level*1000)}'] = df['close'] - fib_level
        
        return fib_features
    
    # ==================== PRICE-BASED FEATURES ====================
    
    def price_features(self, df: pd.DataFrame, return_periods: list = [1, 5, 10, 20]) -> Dict[str, pd.Series]:
        """
        Calculate price-based derived features.
        
        Args:
            df: DataFrame with OHLC data
            return_periods: Periods for calculating returns
            
        Returns:
            Dictionary with price features
        """
        features = {}
        
        # Log returns at multiple periods
        for period in return_periods:
            features[f'return_{period}d'] = np.log(df['close'] / df['close'].shift(period))
        
        # High-Low ratio (daily range)
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        
        # Close position in daily range (0 = at low, 1 = at high)
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Gap from previous close
        features['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        return features
    
    # ==================== TREND FEATURES ====================
    
    def ma_crossovers(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate moving average crossover features (OPTIMIZED).
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with MA crossover features (reduced to 3 most important)
        """
        features = {}
        
        # Price vs MAs - Keep only SMA20 (most important short-term)
        sma_20 = self.sma(df, 20)
        sma_50 = self.sma(df, 50)
        sma_200 = self.sma(df, 200)
        
        features['price_vs_sma20'] = (df['close'] - sma_20) / sma_20
        
        # Golden/Death cross signal (binary) - Most important crossover signals
        features['golden_cross'] = ((sma_50 > sma_200) & (sma_50.shift(1) <= sma_200.shift(1))).astype(int)
        features['death_cross'] = ((sma_50 < sma_200) & (sma_50.shift(1) >= sma_200.shift(1))).astype(int)
        
        return features
    
    # ==================== DERIVATIVE FEATURES (ML-READY) ====================
    
    def calculate_indicator_slopes(
        self,
        df: pd.DataFrame,
        indicators: List[str],
        periods: List[int] = [3, 5, 10]
    ) -> Dict[str, pd.Series]:
        """
        Calculate slopes (rate of change) for indicators to detect rising/falling trends.
        This helps ML models understand momentum direction of indicators.
        
        Args:
            df: DataFrame with indicator columns
            indicators: List of indicator column names to calculate slopes for
            periods: Lookback periods for slope calculation
            
        Returns:
            Dictionary with slope features (percentage change and binary rising/falling)
        """
        slopes = {}
        
        for indicator in indicators:
            if indicator in df.columns:
                for period in periods:
                    # Calculate slope as percentage change
                    slope_name = f'{indicator}_slope_{period}'
                    slopes[slope_name] = df[indicator].pct_change(period) * 100
                    
                    # Binary: is indicator rising (1) or falling (0)?
                    rising_name = f'{indicator}_rising_{period}'
                    slopes[rising_name] = (df[indicator] > df[indicator].shift(period)).astype(int)
        
        return slopes
    
    def calculate_additional_crossovers(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate additional crossover signals for various indicators (OPTIMIZED).
        Crossovers are critical events that often signal trend changes.
        Reduced to ~8 most important crossovers.
        
        Args:
            df: DataFrame with indicator columns
            
        Returns:
            Dictionary with binary crossover features
        """
        crossovers = {}
        
        # MACD Crossovers - Keep 2 most important
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            # MACD crosses above signal line (bullish)
            crossovers['macd_cross_above'] = (
                (df['macd'] > df['macd_signal']) & 
                (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            ).astype(int)
            
            # MACD crosses below signal line (bearish)
            crossovers['macd_cross_below'] = (
                (df['macd'] < df['macd_signal']) & 
                (df['macd'].shift(1) >= df['macd_signal'].shift(1))
            ).astype(int)
        
        # Stochastic Crossovers - Keep both
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            # %K crosses above %D (bullish)
            crossovers['stoch_cross_above'] = (
                (df['stoch_k'] > df['stoch_d']) & 
                (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
            ).astype(int)
            
            # %K crosses below %D (bearish)
            crossovers['stoch_cross_below'] = (
                (df['stoch_k'] < df['stoch_d']) & 
                (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
            ).astype(int)
        
        # RSI Level Crossovers - Keep 2 most important (30 and 70 levels)
        if 'rsi_14' in df.columns:
            # RSI crosses above 30 (oversold exit)
            crossovers['rsi_cross_above_30'] = (
                (df['rsi_14'] > 30) & (df['rsi_14'].shift(1) <= 30)
            ).astype(int)
            
            # RSI crosses below 70 (overbought exit)
            crossovers['rsi_cross_below_70'] = (
                (df['rsi_14'] < 70) & (df['rsi_14'].shift(1) >= 70)
            ).astype(int)
        
        # Price crosses Bollinger Bands - Keep both
        if all(col in df.columns for col in ['close', 'bb_upper', 'bb_lower']):
            # Price crosses above upper band
            crossovers['price_cross_bb_upper'] = (
                (df['close'] > df['bb_upper']) & 
                (df['close'].shift(1) <= df['bb_upper'].shift(1))
            ).astype(int)
            
            # Price crosses below lower band
            crossovers['price_cross_bb_lower'] = (
                (df['close'] < df['bb_lower']) & 
                (df['close'].shift(1) >= df['bb_lower'].shift(1))
            ).astype(int)
        
        return crossovers
    
    # ==================== FEATURE GENERATION ====================
    
    def generate_all_features(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> pd.DataFrame:
        """
        Generate all technical indicators based on configuration.
        
        Args:
            df: DataFrame with OHLCV data
            config: Configuration dictionary with indicator parameters
            
        Returns:
            DataFrame with all features added
        """
        df = df.copy()
        
        logger.info(f"Generating features for {len(df)} records")
        
        # Trend Indicators
        logger.info("Computing trend indicators...")
        for period in config.get('sma_periods', []):
            df[f'sma_{period}'] = self.sma(df, period)
        
        for period in config.get('ema_periods', []):
            df[f'ema_{period}'] = self.ema(df, period)
        
        macd_params = config.get('macd_params', {})
        if macd_params:
            macd_result = self.macd(
                df,
                fast=macd_params.get('fast', 12),
                slow=macd_params.get('slow', 26),
                signal=macd_params.get('signal', 9)
            )
            for key, value in macd_result.items():
                df[key] = value
        
        adx_period = config.get('adx_period', 14)
        if adx_period:
            df[f'adx_{adx_period}'] = self.adx(df, adx_period)
        
        # Momentum Indicators
        logger.info("Computing momentum indicators...")
        rsi_period = config.get('rsi_period', 14)
        if rsi_period:
            df[f'rsi_{rsi_period}'] = self.rsi(df, rsi_period)
        
        for period in config.get('roc_periods', []):
            df[f'roc_{period}'] = self.roc(df, period)
        
        stoch_params = config.get('stoch_params', {})
        if stoch_params:
            stoch_result = self.stochastic_oscillator(
                df,
                k_period=stoch_params.get('k_period', 14),
                d_period=stoch_params.get('d_period', 3)
            )
            for key, value in stoch_result.items():
                df[key] = value
        
        # Multi-period RSI
        rsi_multi = config.get('rsi_periods_multi', [])
        for period in rsi_multi:
            df[f'rsi_{period}'] = self.rsi(df, period)
        
        # Additional momentum indicators
        williams_r_period = config.get('williams_r_period', 14)
        if williams_r_period:
            df[f'williams_r_{williams_r_period}'] = self.williams_r(df, williams_r_period)
        
        cci_period = config.get('cci_period', 20)
        if cci_period:
            df[f'cci_{cci_period}'] = self.cci(df, cci_period)
        
        mfi_period = config.get('mfi_period', 14)
        if mfi_period:
            df[f'mfi_{mfi_period}'] = self.mfi(df, mfi_period)
        
        uo_periods = config.get('ultimate_osc_periods', [7, 14, 28])
        if uo_periods and len(uo_periods) >= 3:
            df['ultimate_oscillator'] = self.ultimate_oscillator(
                df,
                period1=uo_periods[0],
                period2=uo_periods[1],
                period3=uo_periods[2]
            )
        
        # Volatility Indicators
        logger.info("Computing volatility indicators...")
        atr_period = config.get('atr_period', 14)
        if atr_period:
            df[f'atr_{atr_period}'] = self.atr(df, atr_period)
        
        # Multi-period ATR
        atr_multi = config.get('atr_periods_multi', [])
        for period in atr_multi:
            if period != atr_period:  # Avoid duplicate
                df[f'atr_{period}'] = self.atr(df, period)
        
        bb_params = config.get('bollinger_params', {})
        if bb_params:
            bb_result = self.bollinger_bands(
                df,
                period=bb_params.get('period', 20),
                std_dev=bb_params.get('std_dev', 2)
            )
            for key, value in bb_result.items():
                df[key] = value
        
        for period in config.get('hist_vol_periods', []):
            df[f'hist_vol_{period}'] = self.historical_volatility(df, period)
        
        # Keltner Channels
        keltner_params = config.get('keltner_params', {})
        if keltner_params:
            keltner_result = self.keltner_channels(
                df,
                period=keltner_params.get('period', 20),
                atr_period=keltner_params.get('atr_period', 14),
                multiplier=keltner_params.get('multiplier', 2)
            )
            for key, value in keltner_result.items():
                df[key] = value
        
        # Donchian Channels
        donchian_period = config.get('donchian_period', 20)
        if donchian_period:
            donchian_result = self.donchian_channels(df, donchian_period)
            for key, value in donchian_result.items():
                df[key] = value
        
        # Volume Indicators
        logger.info("Computing volume indicators...")
        if config.get('vwap', True):
            df['vwap'] = self.vwap(df)
        
        if config.get('obv', True):
            df['obv'] = self.obv(df)
        
        vol_roc_periods = config.get('volume_roc_periods', [])
        for period in vol_roc_periods:
            df[f'volume_roc_{period}'] = self.volume_roc(df, period)
        
        # Additional volume indicators
        if config.get('ad_line', True):
            df['ad_line'] = self.ad_line(df)
        
        cmf_period = config.get('cmf_period', 20)
        if cmf_period:
            df[f'cmf_{cmf_period}'] = self.cmf(df, cmf_period)
        
        vwma_period = config.get('vwma_period', 20)
        if vwma_period:
            df[f'vwma_{vwma_period}'] = self.vwma(df, vwma_period)
        
        # MA Crossovers and Trend Features
        if config.get('ma_crossovers', False):
            logger.info("Computing MA crossover features...")
            ma_cross_features = self.ma_crossovers(df)
            for key, value in ma_cross_features.items():
                df[key] = value
        
        # Pattern Recognition
        candlestick_config = config.get('candlestick_patterns', {})
        if candlestick_config.get('enabled', False):
            logger.info("Detecting candlestick patterns...")
            pattern_features = self.detect_candlestick_patterns(df)
            for key, value in pattern_features.items():
                df[key] = value
        
        # Support/Resistance Levels
        sr_config = config.get('support_resistance', {})
        if sr_config.get('enabled', False):
            logger.info("Calculating support/resistance levels...")
            sr_features = self.support_resistance_levels(
                df,
                lookback=sr_config.get('lookback', 20),
                num_levels=sr_config.get('num_levels', 3)
            )
            for key, value in sr_features.items():
                df[key] = value
        
        # Fibonacci Retracement Levels
        fib_config = config.get('fibonacci', {})
        if fib_config.get('enabled', False):
            logger.info("Calculating Fibonacci retracement levels...")
            fib_features = self.fibonacci_levels(
                df,
                lookback=fib_config.get('lookback', 50),
                levels=fib_config.get('levels', [0.236, 0.382, 0.500, 0.618])
            )
            for key, value in fib_features.items():
                df[key] = value
        
        # Price-based Features
        price_config = config.get('price_features', {})
        if price_config:
            logger.info("Calculating price-based features...")
            price_feat = self.price_features(
                df,
                return_periods=price_config.get('returns', [1, 5, 10, 20])
            )
            for key, value in price_feat.items():
                df[key] = value
        
        vwma_period = config.get('vwma_period', 20)
        if vwma_period:
            df[f'vwma_{vwma_period}'] = self.vwma(df, vwma_period)
        
        # Pattern Recognition
        logger.info("Detecting candlestick patterns...")
        pattern_results = self.detect_candlestick_patterns(df)
        for key, value in pattern_results.items():
            df[key] = value
        
        # Support and Resistance
        logger.info("Calculating support and resistance levels...")
        sr_results = self.support_resistance_levels(df, lookback=20, num_levels=3)
        for key, value in sr_results.items():
            df[key] = value
        
        # Fibonacci Retracement
        logger.info("Calculating Fibonacci retracement levels...")
        fib_results = self.fibonacci_levels(df, lookback=50)
        for key, value in fib_results.items():
            df[key] = value
        
        # Price-based features
        logger.info("Calculating price-based features...")
        price_results = self.price_features(df, return_periods=[1, 5, 10, 20])
        for key, value in price_results.items():
            df[key] = value
        
        # Trend features
        logger.info("Calculating trend features...")
        trend_results = self.ma_crossovers(df)
        for key, value in trend_results.items():
            df[key] = value
        
        # Indicator slopes (derivative features for ML) - OPTIMIZED: 4 indicators, 2 periods
        logger.info("Calculating indicator slopes...")
        slope_indicators = ['rsi_14', 'macd', 'stoch_k', 'cci_20']  # Reduced from 6 to 4 indicators
        slope_periods = config.get('slope_periods', [5, 10])  # Reduced from 3 to 2 periods
        slope_results = self.calculate_indicator_slopes(df, slope_indicators, slope_periods)
        for key, value in slope_results.items():
            df[key] = value
        
        # Additional crossover signals (critical events for ML)
        logger.info("Calculating additional crossover signals...")
        crossover_results = self.calculate_additional_crossovers(df)
        for key, value in crossover_results.items():
            df[key] = value
        
        logger.info(f"Feature generation complete. Total columns: {len(df.columns)}")
        
        return df
