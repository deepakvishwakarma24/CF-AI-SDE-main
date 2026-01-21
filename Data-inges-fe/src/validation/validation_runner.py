"""
Runner script for data validation pipeline.
Validates ingested OHLCV data and produces clean datasets.
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
    VALIDATION_CONFIG
)
from src.validation.ohlcv_checks import OHLCVValidator
from src.ingestion.equity_ohlcv import YahooFinanceIngestion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_validation(
    raw_data: Optional[dict] = None,
    timeframes: Optional[list] = None,
    load_from_file: bool = False,
    save_to_file: bool = True
) -> dict:
    """
    Run the complete data validation pipeline.
    
    Args:
        raw_data: Dictionary of raw data by timeframe and symbol
        timeframes: List of timeframes to validate
        load_from_file: Whether to load data from CSV files
        save_to_file: Whether to save validated data to CSV files
        
    Returns:
        Dictionary containing validated and clean datasets
    """
    # Default parameters
    if timeframes is None:
        timeframes = TIMEFRAMES
    
    # Initialize validator
    validator = OHLCVValidator(
        price_outlier_threshold=VALIDATION_CONFIG['price_outlier_threshold']
    )
    
    # Storage for validated data
    validated_data = {}
    clean_data = {}
    all_validation_logs = []
    
    logger.info("="*80)
    logger.info("STARTING DATA VALIDATION PIPELINE")
    logger.info("="*80)
    
    # Load data if needed
    if load_from_file:
        logger.info("Loading data from files...")
        ingestion = YahooFinanceIngestion()
        raw_data = {}
        
        for interval in timeframes:
            input_dir = f"{DATA_PATHS['raw']}/{interval}"
            symbols = DEFAULT_EQUITY_SYMBOLS + DEFAULT_INDEX_SYMBOLS
            raw_data[interval] = ingestion.load_data(input_dir, symbols, interval)
    
    if not raw_data:
        logger.error("No data to validate. Run ingestion first.")
        return {}
    
    # Validate data for each timeframe
    for interval in timeframes:
        if interval not in raw_data:
            logger.warning(f"No data found for timeframe: {interval}")
            continue
        
        logger.info(f"\nValidating timeframe: {interval}")
        logger.info("-"*80)
        
        validated_data[interval] = {}
        clean_data[interval] = {}
        
        # Validate each symbol
        for symbol, df in raw_data[interval].items():
            logger.info(f"Validating {symbol}...")
            
            # Run validation
            validated_df, validation_log = validator.validate_dataset(df, interval)
            validated_data[interval][symbol] = validated_df
            
            # Get clean dataset
            clean_df = validator.get_clean_dataset(validated_df)
            clean_data[interval][symbol] = clean_df
            
            # Store validation log
            all_validation_logs.extend(validation_log)
            
            # Log summary
            total = len(df)
            valid = len(clean_df)
            invalid = total - valid
            logger.info(f"{symbol}: {valid}/{total} valid records ({invalid} removed)")
        
        # Save to file if requested
        if save_to_file:
            # Save validated data with flags
            validated_dir = f"{DATA_PATHS['validated']}/{interval}"
            Path(validated_dir).mkdir(parents=True, exist_ok=True)
            
            for symbol, df in validated_data[interval].items():
                filepath = Path(validated_dir) / f"{symbol}_{interval}_validated.csv"
                df.to_csv(filepath, index=False)
                logger.info(f"Saved validated data to {filepath}")
            
            # Save clean data
            clean_dir = f"{DATA_PATHS['validated']}/{interval}/clean"
            Path(clean_dir).mkdir(parents=True, exist_ok=True)
            
            for symbol, df in clean_data[interval].items():
                filepath = Path(clean_dir) / f"{symbol}_{interval}_clean.csv"
                df.to_csv(filepath, index=False)
                logger.info(f"Saved clean data to {filepath}")
    
    # Save validation log
    if save_to_file and all_validation_logs:
        log_df = pd.DataFrame(all_validation_logs)
        log_path = Path(DATA_PATHS['validated']) / "validation_log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(log_path, index=False)
        logger.info(f"\nSaved validation log to {log_path}")
        logger.info(f"Total validation issues logged: {len(all_validation_logs)}")
    
    logger.info("\n" + "="*80)
    logger.info("DATA VALIDATION COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    
    # Summary statistics
    for interval in timeframes:
        if interval in clean_data:
            total_records = sum(len(df) for df in clean_data[interval].values())
            logger.info(f"{interval}: {len(clean_data[interval])} symbols, {total_records} clean records")
    
    return {
        'validated': validated_data,
        'clean': clean_data,
        'validation_log': all_validation_logs
    }


if __name__ == "__main__":
    # Run validation on previously ingested data
    results = run_validation(load_from_file=True)
    
    # Display sample clean data
    if results and 'clean' in results:
        for interval, data in results['clean'].items():
            if data:
                symbol = list(data.keys())[0]
                logger.info(f"\nSample clean data for {symbol} ({interval}):")
                logger.info(f"\n{data[symbol].head()}")
