"""
Data validation module for OHLCV data.
Performs quality checks and identifies data issues without modifying raw data.
"""

import logging
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from datetime import timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OHLCVValidator:
    """
    Validates OHLCV data quality and detects anomalies.
    Raw data remains immutable; validation creates separate cleaned datasets.
    """
    
    def __init__(self, price_outlier_threshold: float = 0.20):
        """
        Initialize validator with configuration.
        
        Args:
            price_outlier_threshold: Threshold for detecting price outliers (default: 20%)
        """
        self.price_outlier_threshold = price_outlier_threshold
        self.validation_log: List[Dict] = []
    
    def validate_price_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that high >= low and other price relationships.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with validation flags
        """
        df = df.copy()
        
        # Check high >= low
        invalid_hl = df['high'] < df['low']
        if invalid_hl.any():
            for idx in df[invalid_hl].index:
                self.validation_log.append({
                    'timestamp': df.loc[idx, 'timestamp'],
                    'symbol': df.loc[idx, 'symbol'],
                    'issue': 'high < low',
                    'high': df.loc[idx, 'high'],
                    'low': df.loc[idx, 'low']
                })
            logger.warning(f"Found {invalid_hl.sum()} records where high < low")
        
        # Check high >= close
        invalid_hc = df['high'] < df['close']
        if invalid_hc.any():
            for idx in df[invalid_hc].index:
                self.validation_log.append({
                    'timestamp': df.loc[idx, 'timestamp'],
                    'symbol': df.loc[idx, 'symbol'],
                    'issue': 'high < close',
                    'high': df.loc[idx, 'high'],
                    'close': df.loc[idx, 'close']
                })
            logger.warning(f"Found {invalid_hc.sum()} records where high < close")
        
        # Check high >= open
        invalid_ho = df['high'] < df['open']
        if invalid_ho.any():
            for idx in df[invalid_ho].index:
                self.validation_log.append({
                    'timestamp': df.loc[idx, 'timestamp'],
                    'symbol': df.loc[idx, 'symbol'],
                    'issue': 'high < open',
                    'high': df.loc[idx, 'high'],
                    'open': df.loc[idx, 'open']
                })
            logger.warning(f"Found {invalid_ho.sum()} records where high < open")
        
        # Check low <= close
        invalid_lc = df['low'] > df['close']
        if invalid_lc.any():
            for idx in df[invalid_lc].index:
                self.validation_log.append({
                    'timestamp': df.loc[idx, 'timestamp'],
                    'symbol': df.loc[idx, 'symbol'],
                    'issue': 'low > close',
                    'low': df.loc[idx, 'low'],
                    'close': df.loc[idx, 'close']
                })
            logger.warning(f"Found {invalid_lc.sum()} records where low > close")
        
        # Check low <= open
        invalid_lo = df['low'] > df['open']
        if invalid_lo.any():
            for idx in df[invalid_lo].index:
                self.validation_log.append({
                    'timestamp': df.loc[idx, 'timestamp'],
                    'symbol': df.loc[idx, 'symbol'],
                    'issue': 'low > open',
                    'low': df.loc[idx, 'low'],
                    'open': df.loc[idx, 'open']
                })
            logger.warning(f"Found {invalid_lo.sum()} records where low > open")
        
        # Mark invalid records
        df['valid_price_relationship'] = ~(invalid_hl | invalid_hc | invalid_ho | invalid_lc | invalid_lo)
        
        return df
    
    def validate_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that volume >= 0.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with validation flags
        """
        df = df.copy()
        
        invalid_volume = df['volume'] < 0
        if invalid_volume.any():
            for idx in df[invalid_volume].index:
                self.validation_log.append({
                    'timestamp': df.loc[idx, 'timestamp'],
                    'symbol': df.loc[idx, 'symbol'],
                    'issue': 'negative volume',
                    'volume': df.loc[idx, 'volume']
                })
            logger.warning(f"Found {invalid_volume.sum()} records with negative volume")
        
        df['valid_volume'] = ~invalid_volume
        
        return df
    
    def detect_price_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect price outliers based on percentage change threshold.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with outlier flags
        """
        df = df.copy()
        
        # Calculate price change percentage
        df['price_change_pct'] = abs((df['close'] - df['open']) / df['open'])
        
        # Flag outliers
        outliers = df['price_change_pct'] > self.price_outlier_threshold
        if outliers.any():
            for idx in df[outliers].index:
                self.validation_log.append({
                    'timestamp': df.loc[idx, 'timestamp'],
                    'symbol': df.loc[idx, 'symbol'],
                    'issue': f'price outlier (>{self.price_outlier_threshold*100}%)',
                    'open': df.loc[idx, 'open'],
                    'close': df.loc[idx, 'close'],
                    'change_pct': df.loc[idx, 'price_change_pct']
                })
            logger.warning(f"Found {outliers.sum()} price outliers")
        
        df['is_outlier'] = outliers
        
        return df
    
    def detect_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and flag duplicate timestamps.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with duplicate flags
        """
        df = df.copy()
        
        # Check for duplicate timestamps
        duplicates = df.duplicated(subset=['timestamp'], keep='first')
        if duplicates.any():
            for idx in df[duplicates].index:
                self.validation_log.append({
                    'timestamp': df.loc[idx, 'timestamp'],
                    'symbol': df.loc[idx, 'symbol'],
                    'issue': 'duplicate timestamp'
                })
            logger.warning(f"Found {duplicates.sum()} duplicate timestamps")
        
        df['is_duplicate'] = duplicates
        
        return df
    
    def detect_missing_timestamps(
        self,
        df: pd.DataFrame,
        interval: str
    ) -> pd.DataFrame:
        """
        Detect missing timestamps based on expected frequency.
        
        Args:
            df: DataFrame with OHLCV data
            interval: Timeframe (1m, 5m, 1h, 1d)
            
        Returns:
            DataFrame with missing timestamp information
        """
        df = df.copy()
        
        # Define expected frequency
        freq_map = {
            '1m': '1min',
            '5m': '5min',
            '1h': '1H',
            '1d': '1D'
        }
        
        if interval not in freq_map:
            logger.warning(f"Unknown interval: {interval}")
            return df
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Expected frequency
        expected_freq = freq_map[interval]
        
        # Calculate time differences
        df['time_diff'] = df['timestamp'].diff()
        
        # Expected time delta
        if interval == '1m':
            expected_delta = timedelta(minutes=1)
        elif interval == '5m':
            expected_delta = timedelta(minutes=5)
        elif interval == '1h':
            expected_delta = timedelta(hours=1)
        elif interval == '1d':
            expected_delta = timedelta(days=1)
        else:
            expected_delta = timedelta(days=1)
        
        # Flag gaps (allowing for weekends/holidays with 2x multiplier)
        max_expected = expected_delta * 2
        has_gap = df['time_diff'] > max_expected
        
        if has_gap.any():
            gap_count = has_gap.sum()
            for idx in df[has_gap].index:
                if idx > 0:
                    self.validation_log.append({
                        'timestamp': df.loc[idx, 'timestamp'],
                        'symbol': df.loc[idx, 'symbol'],
                        'issue': 'missing timestamps detected',
                        'gap_duration': str(df.loc[idx, 'time_diff'])
                    })
            logger.warning(f"Found {gap_count} timestamp gaps")
        
        df['has_gap'] = has_gap
        
        return df
    
    def validate_dataset(
        self,
        df: pd.DataFrame,
        interval: str
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Run all validation checks on a dataset.
        
        Args:
            df: DataFrame with OHLCV data
            interval: Timeframe identifier
            
        Returns:
            Tuple of (validated DataFrame, validation log)
        """
        logger.info(f"Validating dataset with {len(df)} records")
        
        # Reset validation log
        self.validation_log = []
        
        # Run all validation checks
        df = self.validate_price_relationships(df)
        df = self.validate_volume(df)
        df = self.detect_price_outliers(df)
        df = self.detect_duplicates(df)
        df = self.detect_missing_timestamps(df, interval)
        
        # Overall validity flag
        df['is_valid'] = (
            df['valid_price_relationship'] &
            df['valid_volume'] &
            ~df['is_outlier'] &
            ~df['is_duplicate']
        )
        
        valid_count = df['is_valid'].sum()
        invalid_count = len(df) - valid_count
        
        logger.info(f"Validation complete: {valid_count} valid, {invalid_count} invalid records")
        
        return df, self.validation_log
    
    def get_clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return only valid records for downstream processing.
        
        Args:
            df: Validated DataFrame
            
        Returns:
            DataFrame containing only valid records
        """
        if 'is_valid' not in df.columns:
            logger.warning("Dataset not validated yet")
            return df
        
        clean_df = df[df['is_valid']].copy()
        
        # Remove duplicate timestamps, keeping first occurrence
        clean_df = clean_df[~clean_df['is_duplicate']].copy()
        
        # Drop validation columns
        validation_cols = [
            'valid_price_relationship', 'valid_volume', 'is_outlier',
            'is_duplicate', 'has_gap', 'is_valid', 'price_change_pct', 'time_diff'
        ]
        clean_df = clean_df.drop(columns=[col for col in validation_cols if col in clean_df.columns])
        
        logger.info(f"Clean dataset contains {len(clean_df)} records")
        
        return clean_df
