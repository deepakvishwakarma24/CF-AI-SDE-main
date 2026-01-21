#!/usr/bin/env python3
"""
MongoDB-Integrated Pipeline Runner
Wraps the existing pipeline with parallel MongoDB writes.
Does NOT modify original pipeline behavior - all CSV outputs preserved.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import original pipeline functions (unchanged)
from main import (
    run_full_pipeline,
    run_stage_independently,
    get_user_input,
    display_indicator_summary,
)
from src.config.settings import TIMEFRAMES

# Import MongoDB integration layer
from src.integration.mongodb_writer import (
    IngestionDBWriter,
    ValidationDBWriter,
    FeatureDBWriter,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline_with_mongodb(
    symbols: Optional[list] = None,
    timeframes: Optional[list] = None,
    save_to_file: bool = True,
    interactive: bool = False,
    enable_mongodb: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete pipeline with parallel MongoDB writes.

    This function wraps the original pipeline and adds MongoDB persistence
    without modifying any existing behavior. All CSV outputs are preserved.

    Args:
        symbols: List of stock symbols
        timeframes: List of timeframes
        save_to_file: Whether to save to CSV (always True for backward compatibility)
        interactive: Whether to display interactive messages
        enable_mongodb: Whether to write to MongoDB in parallel

    Returns:
        dict: Results from pipeline stages plus MongoDB write summaries
    """
    start_time = datetime.now()

    logger.info("=" * 80)
    logger.info("STARTING MONGODB-INTEGRATED PIPELINE")
    logger.info("=" * 80)

    # Run original pipeline (unchanged behavior)
    results = run_full_pipeline(
        symbols=symbols,
        timeframes=timeframes,
        save_to_file=save_to_file,
        interactive=interactive,
    )

    # Add MongoDB writes in parallel (non-blocking on failure)
    mongodb_results = {}

    if enable_mongodb:
        logger.info("\n" + "=" * 80)
        logger.info("MONGODB PARALLEL WRITES")
        logger.info("=" * 80)

        try:
            # Write ingestion data to MongoDB
            if "ingestion" in results and results["ingestion"]:
                logger.info("Writing ingestion data to MongoDB...")
                mongodb_results["ingestion"] = IngestionDBWriter.write_ingested_data(
                    data=results["ingestion"], source="yahoo_finance"
                )

            # Write validation data to MongoDB
            if "validation" in results and results["validation"]:
                logger.info("Writing validation data to MongoDB...")
                validation_data = results["validation"]

                # Handle both dict formats
                if isinstance(validation_data, dict):
                    validated = validation_data.get("validated", {})
                    clean = validation_data.get("clean", {})
                    logs = validation_data.get("validation_log", [])

                    mongodb_results["validation"] = (
                        ValidationDBWriter.write_validation_results(
                            validated_data=validated,
                            clean_data=clean,
                            validation_logs=logs,
                        )
                    )

            # Write feature data to MongoDB
            if "features" in results and results["features"]:
                logger.info("Writing feature data to MongoDB...")
                mongodb_results["features"] = FeatureDBWriter.write_feature_data(
                    feature_data=results["features"], version=1
                )

        except Exception as e:
            logger.error(f"MongoDB write error (non-fatal): {e}")
            mongodb_results["error"] = str(e)

    # Add MongoDB results to overall results
    results["mongodb"] = mongodb_results

    elapsed = (datetime.now() - start_time).total_seconds()
    results["total_elapsed_seconds"] = elapsed

    logger.info("\n" + "=" * 80)
    logger.info(f"MONGODB-INTEGRATED PIPELINE COMPLETED IN {elapsed:.2f} SECONDS")
    logger.info("=" * 80)

    # Log MongoDB summary
    if mongodb_results:
        for stage, summary in mongodb_results.items():
            if isinstance(summary, dict):
                status = summary.get("status", "unknown")
                if stage == "ingestion":
                    records = summary.get("total_records", 0)
                    logger.info(f"MongoDB {stage}: {status} ({records} records)")
                elif stage == "validation":
                    validated = summary.get("validated_records", 0)
                    clean = summary.get("clean_records", 0)
                    logger.info(
                        f"MongoDB {stage}: {status} ({validated} validated, {clean} clean)"
                    )
                elif stage == "features":
                    records = summary.get("feature_records", 0)
                    logger.info(f"MongoDB {stage}: {status} ({records} records)")

    return results


def run_ingestion_with_mongodb(
    symbols: Optional[list] = None, timeframes: Optional[list] = None
) -> Dict[str, Any]:
    """Run ingestion stage only with MongoDB writes."""
    from src.ingestion.runner import run_ingestion

    results = run_ingestion(symbols=symbols, timeframes=timeframes, save_to_file=True)

    mongodb_result = IngestionDBWriter.write_ingested_data(
        data=results, source="yahoo_finance"
    )

    return {"data": results, "mongodb": mongodb_result}


def run_validation_with_mongodb(
    raw_data: Optional[dict] = None, load_from_file: bool = False
) -> Dict[str, Any]:
    """Run validation stage only with MongoDB writes."""
    from src.validation.validation_runner import run_validation

    results = run_validation(
        raw_data=raw_data, load_from_file=load_from_file, save_to_file=True
    )

    mongodb_result = ValidationDBWriter.write_validation_results(
        validated_data=results.get("validated", {}),
        clean_data=results.get("clean", {}),
        validation_logs=results.get("validation_log", []),
    )

    return {"data": results, "mongodb": mongodb_result}


def run_features_with_mongodb(
    clean_data: Optional[dict] = None, load_from_file: bool = False
) -> Dict[str, Any]:
    """Run feature engineering stage only with MongoDB writes."""
    from src.features.feature_runner import run_feature_engineering

    results = run_feature_engineering(
        clean_data=clean_data, load_from_file=load_from_file, save_to_file=True
    )

    mongodb_result = FeatureDBWriter.write_feature_data(feature_data=results, version=1)

    return {"data": results, "mongodb": mongodb_result}


def main():
    """Main entry point for MongoDB-integrated pipeline."""

    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help"]:
            print_help()
            return

        # Parse CLI arguments
        symbols = None
        timeframes = None
        enable_mongodb = True

        for arg in sys.argv[1:]:
            if arg.startswith("--symbols="):
                symbols = arg.split("=")[1].split(",")
            elif arg.startswith("--timeframes="):
                timeframes = arg.split("=")[1].split(",")
            elif arg == "--no-mongodb":
                enable_mongodb = False

        run_pipeline_with_mongodb(
            symbols=symbols,
            timeframes=timeframes,
            save_to_file=True,
            enable_mongodb=enable_mongodb,
        )
    else:
        # Interactive mode
        symbols, timeframes = get_user_input()
        run_pipeline_with_mongodb(
            symbols=symbols,
            timeframes=timeframes,
            save_to_file=True,
            interactive=True,
            enable_mongodb=True,
        )


def print_help():
    """Print help message."""
    print(
        """
MongoDB-Integrated Financial Data Pipeline
============================================

USAGE:
    python pipeline_with_mongodb.py                           # Interactive mode
    python pipeline_with_mongodb.py --symbols=AAPL,MSFT       # CLI with symbols
    python pipeline_with_mongodb.py --timeframes=1h,1d        # CLI with timeframes
    python pipeline_with_mongodb.py --no-mongodb              # Disable MongoDB writes

ENVIRONMENT VARIABLES:
    MONGODB_URI       MongoDB connection string (default: mongodb://localhost:27017)
    MONGODB_DATABASE  Database name (default: cf_ai_sde)

This pipeline wraps the original pipeline with parallel MongoDB writes.
All CSV outputs are preserved exactly as before.
    """
    )


if __name__ == "__main__":
    main()
