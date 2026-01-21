#!/usr/bin/env python3
"""
Quick Stock Data Pipeline - Command Line Interface
Usage: python3 quick_run.py SYMBOL1 SYMBOL2 ... [OPTIONS]

Examples:
  python3 quick_run.py AAPL
  python3 quick_run.py AAPL MSFT GOOGL
  python3 quick_run.py AAPL --timeframes 1h,1d
  python3 quick_run.py NVDA META --timeframes 1d
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from main import run_full_pipeline
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Quick Stock Data Pipeline - Fetch, validate, and engineer features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL
  %(prog)s AAPL MSFT GOOGL
  %(prog)s AAPL --timeframes 1h,1d
  %(prog)s NVDA META --timeframes 1d
  %(prog)s "^GSPC" "^DJI" --timeframes 1d
        """
    )
    
    parser.add_argument(
        'symbols',
        nargs='+',
        help='Stock symbols to process (e.g., AAPL MSFT GOOGL)'
    )
    
    parser.add_argument(
        '-t', '--timeframes',
        default='1h,1d',
        help='Comma-separated timeframes: 1m,5m,1h,1d (default: 1h,1d)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to CSV files'
    )
    
    return parser.parse_args()


def main():
    """Run pipeline from command line."""
    args = parse_args()
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols]
    
    # Parse timeframes
    timeframes = [tf.strip().lower() for tf in args.timeframes.split(',') if tf.strip()]
    
    # Validate timeframes
    valid_tf = ['1m', '5m', '1h', '1d']
    timeframes = [tf for tf in timeframes if tf in valid_tf]
    
    if not timeframes:
        logger.error("No valid timeframes specified. Use: 1m, 5m, 1h, or 1d")
        sys.exit(1)
    
    # Display configuration
    print("\n" + "="*80)
    print(f"Processing {len(symbols)} symbol(s) across {len(timeframes)} timeframe(s)")
    print("="*80)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print("="*80 + "\n")
    
    # Run pipeline
    try:
        results = run_full_pipeline(
            symbols=symbols,
            timeframes=timeframes,
            save_to_file=not args.no_save
        )
        
        if results:
            print("\n" + "="*80)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            
            if not args.no_save:
                print("\n📁 Data saved to:")
                for tf in timeframes:
                    print(f"  • data/features/{tf}/")
                print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted.\n")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
