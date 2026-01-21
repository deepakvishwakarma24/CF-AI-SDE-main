"""
Data ingestion module for equity and index OHLCV data from Yahoo Finance.
Fetches historical price data with proper timezone normalization to UTC.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YahooFinanceIngestion:
    """
    Handles data ingestion from Yahoo Finance API.
    Fetches OHLCV data with both raw and adjusted prices.
    """
    
    def __init__(self):
        """Initialize the ingestion handler."""
        self.raw_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        
    def fetch_ohlcv(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data from Yahoo Finance for given symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Timeframe (1m, 5m, 1h, 1d)
            
        Returns:
            Dictionary mapping symbol to DataFrame with OHLCV data
        """
        data = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching {interval} data for {symbol}")
                
                # Download data from Yahoo Finance
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=False,  # Keep both raw and adjusted prices
                    actions=True  # Include dividends and splits
                )
                
                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    continue
                
                # Normalize timezone to UTC
                # Type ignore for pandas DatetimeIndex timezone attributes
                if df.index.tz is not None:  # type: ignore
                    df.index = df.index.tz_convert('UTC')  # type: ignore
                else:
                    df.index = df.index.tz_localize('UTC')  # type: ignore
                
                # Rename columns to lowercase for consistency
                df.columns = [col.lower() for col in df.columns]
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Reset index to make datetime a column
                df = df.reset_index()
                df.rename(columns={'index': 'timestamp', 'Date': 'timestamp', 'Datetime': 'timestamp'}, inplace=True)
                
                # Ensure timestamp column exists
                if 'timestamp' not in df.columns:
                    logger.error(f"Timestamp column missing for {symbol}")
                    continue
                
                # Store raw data (immutable)
                data[symbol] = df.copy()
                
                logger.info(f"Successfully fetched {len(df)} records for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        return data
    
    def fetch_adjusted_prices(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch adjusted OHLCV data accounting for splits and dividends.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Timeframe (1m, 5m, 1h, 1d)
            
        Returns:
            Dictionary mapping symbol to DataFrame with adjusted OHLCV data
        """
        data = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching adjusted {interval} data for {symbol}")
                
                # Download adjusted data from Yahoo Finance
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,  # Use adjusted prices
                    actions=False
                )
                
                if df.empty:
                    logger.warning(f"No adjusted data returned for {symbol}")
                    continue
                
                # Normalize timezone to UTC
                # Type ignore for pandas DatetimeIndex timezone attributes
                if df.index.tz is not None:  # type: ignore
                    df.index = df.index.tz_convert('UTC')  # type: ignore
                else:
                    df.index = df.index.tz_localize('UTC')  # type: ignore
                
                # Rename columns to lowercase for consistency
                df.columns = [col.lower() for col in df.columns]
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Reset index to make datetime a column
                df = df.reset_index()
                df.rename(columns={'index': 'timestamp', 'Date': 'timestamp', 'Datetime': 'timestamp'}, inplace=True)
                
                # Rename adjusted columns
                df.rename(columns={
                    'open': 'adj_open',
                    'high': 'adj_high',
                    'low': 'adj_low',
                    'close': 'adj_close',
                    'volume': 'adj_volume'
                }, inplace=True)
                
                data[symbol] = df.copy()
                
                logger.info(f"Successfully fetched {len(df)} adjusted records for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching adjusted data for {symbol}: {str(e)}")
                continue
        
        return data
    
    def merge_raw_and_adjusted(
        self,
        raw_data: Dict[str, pd.DataFrame],
        adjusted_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Merge raw and adjusted price data into single DataFrames.
        
        Args:
            raw_data: Dictionary of raw OHLCV data
            adjusted_data: Dictionary of adjusted OHLCV data
            
        Returns:
            Dictionary of merged DataFrames
        """
        merged_data = {}
        
        for symbol in raw_data.keys():
            if symbol not in adjusted_data:
                logger.warning(f"No adjusted data for {symbol}, using raw only")
                merged_data[symbol] = raw_data[symbol].copy()
                continue
            
            try:
                # Merge on timestamp
                raw_df = raw_data[symbol].copy()
                adj_df = adjusted_data[symbol].copy()
                
                merged = pd.merge(
                    raw_df,
                    adj_df[['timestamp', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']],
                    on='timestamp',
                    how='left'
                )
                
                merged_data[symbol] = merged
                
                logger.info(f"Merged raw and adjusted data for {symbol}")
                
            except Exception as e:
                logger.error(f"Error merging data for {symbol}: {str(e)}")
                merged_data[symbol] = raw_data[symbol].copy()
        
        return merged_data
    
    def save_data(
        self,
        data: Dict[str, pd.DataFrame],
        output_dir: str,
        interval: str
    ) -> None:
        """
        Save ingested data to CSV files.
        
        Args:
            data: Dictionary of DataFrames to save
            output_dir: Output directory path
            interval: Timeframe identifier
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for symbol, df in data.items():
            filename = f"{symbol}_{interval}_raw.csv"
            filepath = output_path / filename
            
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {symbol} data to {filepath}")
    
    def load_data(
        self,
        input_dir: str,
        symbols: List[str],
        interval: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Load previously ingested data from CSV files.
        
        Args:
            input_dir: Input directory path
            symbols: List of symbols to load
            interval: Timeframe identifier
            
        Returns:
            Dictionary of loaded DataFrames
        """
        data = {}
        input_path = Path(input_dir)
        
        for symbol in symbols:
            filename = f"{symbol}_{interval}_raw.csv"
            filepath = input_path / filename
            
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                continue
            
            try:
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                data[symbol] = df
                logger.info(f"Loaded {symbol} data from {filepath}")
            except Exception as e:
                logger.error(f"Error loading {filepath}: {str(e)}")
        
        return data
