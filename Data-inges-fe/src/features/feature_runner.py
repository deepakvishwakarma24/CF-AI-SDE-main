"""
Runner script for feature engineering pipeline.
Computes technical indicators on validated data.
"""

import logging
from pathlib import Path
import sys
import pandas as pd
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.settings import (
    DEFAULT_EQUITY_SYMBOLS,
    DEFAULT_INDEX_SYMBOLS,
    TIMEFRAMES,
    DATA_PATHS,
    FEATURE_CONFIG
)
from src.features.technical_indicators import TechnicalIndicators
from src.features.normalization import FeatureNormalizer
from src.ingestion.equity_ohlcv import YahooFinanceIngestion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_feature_engineering(
    clean_data: Optional[dict] = None,
    timeframes: Optional[list] = None,
    load_from_file: bool = False,
    save_to_file: bool = True,
    apply_normalization: bool = True
) -> dict:
    """
    Run the complete feature engineering pipeline with ML-ready normalization.
    
    Args:
        clean_data: Dictionary of clean data by timeframe and symbol
        timeframes: List of timeframes to process
        load_from_file: Whether to load data from CSV files
        save_to_file: Whether to save feature data to CSV files
        apply_normalization: Whether to apply feature normalization (stores both raw and normalized)
        
    Returns:
        Dictionary containing datasets with computed features
    """
    # Initialize feature calculator and normalizer
    feature_calculator = TechnicalIndicators()
    normalizer = FeatureNormalizer()
    
    # Storage for feature data
    feature_data = {}
    
    logger.info("="*80)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE (ML-READY)")
    logger.info("="*80)
    
    # Load data if needed
    if load_from_file:
        # Default parameters
        if timeframes is None:
            timeframes = TIMEFRAMES
        logger.info("Loading clean data from files...")
        ingestion = YahooFinanceIngestion()
        clean_data = {}
        
        for interval in timeframes:
            input_dir = f"{DATA_PATHS['validated']}/{interval}/clean"
            symbols = DEFAULT_EQUITY_SYMBOLS + DEFAULT_INDEX_SYMBOLS
            
            # Load clean data
            loaded_data = {}
            input_path = Path(input_dir)
            
            if not input_path.exists():
                logger.warning(f"Directory not found: {input_dir}")
                continue
            
            for symbol in symbols:
                filename = f"{symbol}_{interval}_clean.csv"
                filepath = input_path / filename
                
                if not filepath.exists():
                    continue
                
                try:
                    df = pd.read_csv(filepath)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    loaded_data[symbol] = df
                    logger.info(f"Loaded {symbol} clean data from {filepath}")
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {str(e)}")
            
            clean_data[interval] = loaded_data
    
    if not clean_data:
        logger.error("No clean data to process. Run validation first.")
        return {}
    
    # If timeframes not specified, use the timeframes that have data
    if timeframes is None:
        timeframes = list(clean_data.keys())
    
    # Generate features for each timeframe
    for interval in timeframes:
        if interval not in clean_data:
            logger.warning(f"No clean data found for timeframe: {interval}")
            continue
        
        logger.info(f"\nProcessing timeframe: {interval}")
        logger.info("-"*80)
        
        feature_data[interval] = {}
        
        # Process each symbol
        for symbol, df in clean_data[interval].items():
            logger.info(f"Generating features for {symbol}...")
            
            # Sort by timestamp to ensure chronological order
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Generate all features
            df_with_features = feature_calculator.generate_all_features(
                df,
                FEATURE_CONFIG
            )
            
            # Apply normalization if requested (ML-ready: stores both raw and normalized)
            if apply_normalization:
                logger.info(f"Normalizing features for {symbol}...")
                # Exclude OHLCV and metadata from normalization
                exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                               'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume',
                               'dividends', 'stock_splits']
                
                # Apply z-score normalization (best for neural networks)
                df_normalized, norm_params = normalizer.normalize_features(
                    df_with_features,
                    method='zscore',
                    exclude_columns=exclude_cols,
                    fit=True
                )
                df_with_features = df_normalized
            
            feature_data[interval][symbol] = df_with_features
            
            # Log summary
            original_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                           'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume',
                           'dividends', 'stock_splits']
            num_raw_features = len([col for col in df_with_features.columns 
                                   if col not in original_cols and not col.endswith('_norm')])
            num_norm_features = len([col for col in df_with_features.columns if col.endswith('_norm')])
            logger.info(f"{symbol}: {num_raw_features} raw features + {num_norm_features} normalized features")
        
        # Save normalization parameters if normalization was applied
        if apply_normalization and save_to_file:
            output_dir = f"{DATA_PATHS['features']}/{interval}"
            norm_params_file = Path(output_dir) / "normalization_params.json"
            normalizer.save_normalization_params(normalizer.normalization_params, norm_params_file)
            logger.info(f"Saved normalization parameters to {norm_params_file}")
        
        # Save to file if requested
        if save_to_file:
            output_dir = f"{DATA_PATHS['features']}/{interval}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            for symbol, df in feature_data[interval].items():
                filepath = Path(output_dir) / f"{symbol}_{interval}_features.csv"
                df.to_csv(filepath, index=False)
                logger.info(f"Saved feature data to {filepath}")
    
    logger.info("\n" + "="*80)
    logger.info("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    
    # Summary statistics
    for interval in timeframes:
        if interval in feature_data:
            total_records = sum(len(df) for df in feature_data[interval].values())
            if feature_data[interval]:
                sample_df = list(feature_data[interval].values())[0]
                num_features = len([col for col in sample_df.columns 
                                   if col not in ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                                                 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume',
                                                 'dividends', 'stock splits']])
                logger.info(f"{interval}: {len(feature_data[interval])} symbols, {total_records} records, ~{num_features} features")
    
    return feature_data


if __name__ == "__main__":
    # Run feature engineering on previously validated data
    results = run_feature_engineering(load_from_file=True)
    
    # Display sample feature data
    if results:
        for interval, data in results.items():
            if data:
                symbol = list(data.keys())[0]
                logger.info(f"\nSample feature data for {symbol} ({interval}):")
                logger.info(f"\n{data[symbol].head()}")
                logger.info(f"\nColumns: {list(data[symbol].columns)}")
