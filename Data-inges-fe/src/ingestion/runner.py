"""
Runner script for data ingestion pipeline.
Orchestrates fetching of OHLCV data from Yahoo Finance.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.settings import (
    DEFAULT_EQUITY_SYMBOLS,
    DEFAULT_INDEX_SYMBOLS,
    DEFAULT_LOOKBACK_DAYS,
    TIMEFRAMES,
    DATA_PATHS
)
from src.ingestion.equity_ohlcv import YahooFinanceIngestion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_ingestion(
    symbols: Optional[list] = None,
    timeframes: Optional[list] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    save_to_file: bool = True
) -> dict:
    """
    Run the complete data ingestion pipeline.
    
    Args:
        symbols: List of symbols to fetch (default: equities + indices)
        timeframes: List of timeframes to fetch (default: all supported)
        start_date: Start date for data (default: based on timeframe)
        end_date: End date for data (default: now)
        save_to_file: Whether to save data to CSV files
        
    Returns:
        Dictionary containing ingested data organized by timeframe and symbol
    """
    # Default parameters
    if symbols is None:
        symbols = DEFAULT_EQUITY_SYMBOLS + DEFAULT_INDEX_SYMBOLS
    
    if timeframes is None:
        timeframes = TIMEFRAMES
    
    if end_date is None:
        end_date = datetime.now()
    
    # Initialize ingestion handler
    ingestion = YahooFinanceIngestion()
    
    # Storage for all ingested data
    all_data = {}
    
    logger.info("="*80)
    logger.info("STARTING DATA INGESTION PIPELINE")
    logger.info("="*80)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"End Date: {end_date}")
    logger.info("="*80)
    
    # Fetch data for each timeframe
    for interval in timeframes:
        logger.info(f"\nProcessing timeframe: {interval}")
        logger.info("-"*80)
        
        # Calculate start date based on timeframe
        if start_date is None:
            lookback = DEFAULT_LOOKBACK_DAYS.get(interval, 365)
            calc_start_date = end_date - timedelta(days=lookback)
        else:
            calc_start_date = start_date
        
        logger.info(f"Date range: {calc_start_date} to {end_date}")
        
        # Fetch raw data
        raw_data = ingestion.fetch_ohlcv(
            symbols=symbols,
            start_date=calc_start_date,
            end_date=end_date,
            interval=interval
        )
        
        # Fetch adjusted data
        adjusted_data = ingestion.fetch_adjusted_prices(
            symbols=symbols,
            start_date=calc_start_date,
            end_date=end_date,
            interval=interval
        )
        
        # Merge raw and adjusted data
        merged_data = ingestion.merge_raw_and_adjusted(raw_data, adjusted_data)
        
        # Store in results
        all_data[interval] = merged_data
        
        # Save to file if requested
        if save_to_file:
            output_dir = f"{DATA_PATHS['raw']}/{interval}"
            ingestion.save_data(merged_data, output_dir, interval)
        
        # Log summary
        total_records = sum(len(df) for df in merged_data.values())
        logger.info(f"Ingested {len(merged_data)} symbols with {total_records} total records")
    
    logger.info("\n" + "="*80)
    logger.info("DATA INGESTION COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    
    # Summary statistics
    for interval, data in all_data.items():
        total_records = sum(len(df) for df in data.values())
        logger.info(f"{interval}: {len(data)} symbols, {total_records} records")
    
    return all_data


if __name__ == "__main__":
    # Run ingestion for all configured symbols and timeframes
    ingested_data = run_ingestion()
    
    # Display sample data
    for interval, data in ingested_data.items():
        if data:
            symbol = list(data.keys())[0]
            logger.info(f"\nSample data for {symbol} ({interval}):")
            logger.info(f"\n{data[symbol].head()}")
