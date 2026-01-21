#!/usr/bin/env python3
"""
Main pipeline runner for end-to-end financial data processing.
Executes: Ingestion → Validation → Feature Engineering
"""

import logging
from pathlib import Path
import sys
from datetime import datetime
from typing import Optional
import pandas as pd

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from src.ingestion.runner import run_ingestion
from src.validation.validation_runner import run_validation
from src.features.feature_runner import run_feature_engineering
from src.config.settings import TIMEFRAMES, DEFAULT_EQUITY_SYMBOLS, DEFAULT_INDEX_SYMBOLS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def display_indicator_summary(symbol: str, timeframe: str, features_path: Path):
    """
    Display a summary of technical indicators for a given symbol and timeframe.
    
    Args:
        symbol: Stock symbol
        timeframe: Data timeframe (1m, 5m, 1h, 1d)
        features_path: Path to the features CSV file
    """
    try:
        if not features_path.exists():
            return
        
        df = pd.read_csv(features_path)
        if len(df) == 0:
            return
        
        latest = df.iloc[-1]
        
        print(f"\n📈 TECHNICAL INDICATORS - {symbol} ({timeframe})")
        print("-"*80)
        
        # Price indicators
        print(f"{'Close Price:':<25} ${latest['close']:.2f}")
        
        # Moving Averages
        if 'sma_20' in df.columns and pd.notna(latest['sma_20']):
            print(f"{'SMA 20:':<25} ${latest['sma_20']:.2f}")
        if 'sma_50' in df.columns and pd.notna(latest['sma_50']):
            print(f"{'SMA 50:':<25} ${latest['sma_50']:.2f}")
        if 'sma_200' in df.columns and pd.notna(latest['sma_200']):
            print(f"{'SMA 200:':<25} ${latest['sma_200']:.2f}")
        if 'ema_12' in df.columns and pd.notna(latest['ema_12']):
            print(f"{'EMA 12:':<25} ${latest['ema_12']:.2f}")
        if 'ema_26' in df.columns and pd.notna(latest['ema_26']):
            print(f"{'EMA 26:':<25} ${latest['ema_26']:.2f}")
        
        # Momentum Indicators
        if 'macd' in df.columns and pd.notna(latest['macd']):
            print(f"{'MACD:':<25} {latest['macd']:.4f}")
        if 'rsi_14' in df.columns and pd.notna(latest['rsi_14']):
            rsi_status = "Overbought" if latest['rsi_14'] > 70 else "Oversold" if latest['rsi_14'] < 30 else "Neutral"
            print(f"{'RSI (14):':<25} {latest['rsi_14']:.2f} ({rsi_status})")
        if 'stoch_k' in df.columns and pd.notna(latest['stoch_k']):
            print(f"{'Stochastic %K:':<25} {latest['stoch_k']:.2f}")
        
        # Volatility Indicators
        if 'bb_upper' in df.columns and pd.notna(latest['bb_upper']):
            print(f"{'Bollinger Upper:':<25} ${latest['bb_upper']:.2f}")
        if 'bb_lower' in df.columns and pd.notna(latest['bb_lower']):
            print(f"{'Bollinger Lower:':<25} ${latest['bb_lower']:.2f}")
        if 'atr_14' in df.columns and pd.notna(latest['atr_14']):
            print(f"{'ATR (14):':<25} {latest['atr_14']:.2f}")
        
        # Trend Indicators
        if 'adx_14' in df.columns and pd.notna(latest['adx_14']):
            trend_strength = "Strong" if latest['adx_14'] > 25 else "Weak"
            print(f"{'ADX (14):':<25} {latest['adx_14']:.2f} ({trend_strength} trend)")
        
        # Volume Indicators
        if 'vwap' in df.columns and pd.notna(latest['vwap']):
            print(f"{'VWAP:':<25} ${latest['vwap']:.2f}")
        if 'obv' in df.columns and pd.notna(latest['obv']):
            print(f"{'OBV:':<25} {latest['obv']:,.0f}")
        
        print(f"\n{'Total Records:':<25} {len(df):,}")
        print(f"{'Date Range:':<25} {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
        
    except Exception as e:
        logger.debug(f"Could not display indicators for {symbol} ({timeframe}): {e}")


def get_user_input():
    """
    Get stock symbols and timeframes from user input with rich formatting.
    
    Returns:
        Tuple of (symbols list, timeframes list)
    """
    print("\n" + "="*80)
    print(" "*15 + "LIQUID INSTRUMENTS TRADING PIPELINE")
    print("="*80)
    print("\nThis pipeline will:")
    print("  1. Fetch OHLCV data from Yahoo Finance")
    print("  2. Validate and clean the data")
    print("  3. Generate 30+ technical indicators")
    print("  4. Provide 10 years of historical data (1d) for backtesting")
    print("\n" + "="*80)
    
    # Get stock symbols
    print("\n📊 ENTER STOCK SYMBOLS")
    print("-"*80)
    print("Enter one or more stock symbols (comma-separated)")
    print("\nExamples:")
    print("  • Single stock:    AAPL")
    print("  • Multiple stocks: AAPL, MSFT, JPM, NVDA")
    print("  • With indices:    AAPL, ^GSPC, ^NSEI")
    print("  • Press Enter for all 36 default liquid instruments")
    print("\nMajor Indices:")
    print("  ^GSPC (S&P 500), ^IXIC (NASDAQ), ^DJI (Dow Jones)")
    print("  ^NSEI (NIFTY 50), ^NSEBANK (BANKNIFTY), ^VIX (Volatility)")
    print("\nLiquid Stocks (30 across sectors):")
    print("  Tech: AAPL, MSFT, GOOGL, AMZN, NVDA, META, ORCL")
    print("  Finance: JPM, BAC, GS, V, MA")
    print("  Healthcare: JNJ, UNH, PFE, ABBV")
    print("  Consumer: WMT, PG, KO, MCD")
    print("  Energy: XOM, CVX, COP")
    print("  Industrials: BA, CAT, GE")
    print("  Media: DIS, NFLX")
    print("  Auto: TSLA, F")
    
    while True:
        user_symbols = input("\n👉 Enter symbols (or press Enter for all 36 defaults): ").strip()
        
        if not user_symbols:
            # Use all default symbols
            symbols = DEFAULT_EQUITY_SYMBOLS + DEFAULT_INDEX_SYMBOLS
            print(f"✅ Using all {len(symbols)} default liquid instruments")
            break
        
        # Parse symbols
        symbols = [s.strip().upper() for s in user_symbols.split(',') if s.strip()]
        
        if symbols:
            break
        else:
            print("❌ Invalid input. Please try again.")
    
    # Get timeframes
    print("\n" + "="*80)
    print("\n⏰ SELECT TIMEFRAMES")
    print("-"*80)
    print("Available timeframes:")
    print("  1m  - 1 minute  (last 7 days) - High-frequency patterns")
    print("  5m  - 5 minutes (last 60 days) - Intraday strategies")
    print("  1h  - 1 hour    (last 2 years) - Swing trading")
    print("  1d  - 1 day     (last 10 years) - Long-term analysis & backtesting")
    print("\n💡 Recommended: 1d for initial backtesting (captures multiple market regimes)")
    print("\nEnter timeframes (comma-separated) or press Enter for all")
    print("Examples: 1d  or  1h,1d  or  1m,5m,1h,1d")
    
    user_timeframes = input("\n👉 Timeframes (or Enter for all): ").strip()
    
    if user_timeframes:
        timeframes = [tf.strip() for tf in user_timeframes.split(',') if tf.strip() in TIMEFRAMES]
        if not timeframes:
            print("⚠️  Invalid timeframes. Using all timeframes.")
            timeframes = None
    else:
        timeframes = None  # Use all timeframes
        
    # Confirmation
    print("\n" + "="*80)
    print("\n📋 SUMMARY")
    print("-"*80)
    print(f"Symbols:    {', '.join(symbols)}")
    print(f"Timeframes: {', '.join(timeframes) if timeframes else 'All (' + ', '.join(TIMEFRAMES) + ')'}")
    print("\n" + "="*80)
    
    confirm = input("\n👉 Proceed? (Y/n): ").strip().lower()
    if confirm and confirm not in ['y', 'yes']:
        print("\n❌ Aborted by user.")
        sys.exit(0)
    
    return symbols, timeframes


def run_full_pipeline(
    symbols: Optional[list] = None,
    timeframes: Optional[list] = None,
    save_to_file: bool = True,
    interactive: bool = False
):
    """
    Run the complete end-to-end pipeline.
    
    Args:
        symbols: List of stock symbols. Defaults to all configured symbols
        timeframes: List of timeframes. Defaults to all configured timeframes
        save_to_file: Whether to save data to CSV files
        interactive: Whether to display interactive success messages
    
    Returns:
        dict: Results from each pipeline stage
    """
    start_time = datetime.now()
    
    logger.info("="*80)
    logger.info("STARTING END-TO-END FINANCIAL DATA PIPELINE")
    logger.info("="*80)
    
    # Stage 1: Ingestion
    logger.info("\n" + "="*80)
    logger.info("STAGE 1: DATA INGESTION")
    logger.info("="*80)
    ingestion_results = run_ingestion(
        symbols=symbols,
        timeframes=timeframes,
        save_to_file=save_to_file
    )
    
    # Stage 2: Validation
    logger.info("\n" + "="*80)
    logger.info("STAGE 2: DATA VALIDATION")
    logger.info("="*80)
    validation_results = run_validation(
        raw_data=ingestion_results,
        save_to_file=save_to_file
    )
    
    # Stage 3: Feature Engineering
    logger.info("\n" + "="*80)
    logger.info("STAGE 3: FEATURE ENGINEERING")
    logger.info("="*80)
    feature_results = run_feature_engineering(
        clean_data=validation_results.get('clean', validation_results),
        save_to_file=save_to_file
    )
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    results = {
        'ingestion': ingestion_results,
        'validation': validation_results,
        'features': feature_results,
        'elapsed_seconds': elapsed
    }
    
    logger.info("\n" + "="*80)
    logger.info(f"PIPELINE COMPLETED IN {elapsed:.2f} SECONDS")
    logger.info("="*80)
    
    # Log summary stats
    if 'features' in results:
        for interval, data in results['features'].items():
            feature_count = sum(len(df) for df in data.values())
            if data:
                sample_df = list(data.values())[0]
                num_features = len([col for col in sample_df.columns 
                                   if col not in ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']])
                logger.info(f"Features ({interval}): {len(data)} symbols, {feature_count} records, ~{num_features} features/record")
    
    logger.info("="*80 + "\n")
    
    # Display user-friendly success message if interactive
    if interactive and results.get('features'):
        print("\n" + "="*80)
        print(" "*30 + "✅ SUCCESS!")
        print("="*80)
        
        print("\n📁 Generated Files:")
        print("-"*80)
        
        for tf in (timeframes or TIMEFRAMES):
            for sym in (symbols or []):
                if tf in results.get('features', {}) and sym in results['features'][tf]:
                    print(f"\n{sym} ({tf}):")
                    print(f"  • data/raw/{tf}/{sym}_{tf}_raw.csv")
                    print(f"  • data/validated/{tf}/clean/{sym}_{tf}_clean.csv")
                    features_file = f"data/features/{tf}/{sym}_{tf}_features.csv"
                    print(f"  • {features_file}")
                    
                    # Display technical indicators
                    display_indicator_summary(sym, tf, Path(features_file))
        
        print("\n" + "="*80)
        print("\n💡 Next Steps:")
        print("  1. Check validation log: data/validated/validation_log.csv")
        print("  2. Load features: import pandas as pd; df = pd.read_csv('data/features/...')")
        print("  3. Analyze your data with the technical indicators!")
        print("\n" + "="*80 + "\n")
    
    return results


def run_stage_independently(stage: str, **kwargs):
    """
    Run a specific pipeline stage independently.
    
    Args:
        stage: Pipeline stage ('ingestion', 'validation', 'features')
        **kwargs: Additional arguments for the stage
    """
    if stage == 'ingestion':
        return run_ingestion(**kwargs)
    elif stage == 'validation':
        return run_validation(load_from_file=True, **kwargs)
    elif stage == 'features':
        return run_feature_engineering(load_from_file=True, **kwargs)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def main():
    """Main entry point for the pipeline."""
    
    # Check for command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print_help()
            return
        
        # Parse CLI arguments
        symbols = None
        timeframes = None
        
        for arg in sys.argv[1:]:
            if arg.startswith('--symbols='):
                symbols = arg.split('=')[1].split(',')
            elif arg.startswith('--timeframes='):
                timeframes = arg.split('=')[1].split(',')
        
        # Run with CLI args
        run_full_pipeline(symbols=symbols, timeframes=timeframes, save_to_file=True)
    else:
        # Interactive mode (default)
        symbols, timeframes = get_user_input()
        run_full_pipeline(
            symbols=symbols,
            timeframes=timeframes,
            save_to_file=True,
            interactive=True
        )


def print_help():
    """Print help message."""
    print("""
Financial Data Pipeline - Help
================================

USAGE:
    python main.py                                  # Interactive mode (default)
    python main.py --symbols=AAPL,MSFT              # CLI mode with specific symbols
    python main.py --timeframes=1h,1d               # CLI mode with specific timeframes
    python main.py --symbols=AAPL --timeframes=1d   # CLI mode with both

OPTIONS:
    --symbols=SYM1,SYM2     Comma-separated list of stock symbols
    --timeframes=TF1,TF2    Comma-separated list of timeframes (1m, 5m, 1h, 1d)
    -h, --help              Show this help message

INTERACTIVE MODE:
    Simply run 'python main.py' and follow the prompts.

PROGRAMMATIC USAGE:
    from main import run_full_pipeline
    
    results = run_full_pipeline(
        symbols=['AAPL', 'MSFT'],
        timeframes=['1h', '1d'],
        save_to_file=True
    )

OUTPUT:
    - Raw data: data/raw/{timeframe}/{symbol}_{timeframe}_raw.csv
    - Validated: data/validated/{timeframe}/clean/{symbol}_{timeframe}_clean.csv
    - Features: data/features/{timeframe}/{symbol}_{timeframe}_features.csv
    """)


if __name__ == "__main__":
    main()
