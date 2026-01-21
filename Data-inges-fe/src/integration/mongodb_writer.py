"""
MongoDB Integration for Data-inges-fe Pipeline
Provides parallel MongoDB writes alongside existing CSV outputs.
Does NOT modify existing pipeline behavior.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from db.writers import MarketDataWriter, ValidationWriter, FeatureWriter
from db.connection import is_mongodb_available

logger = logging.getLogger(__name__)


class IngestionDBWriter:
    """
    MongoDB writer for ingestion pipeline.
    Writes raw data to market_data_raw collection in parallel with CSV outputs.
    """

    @staticmethod
    def write_ingested_data(
        data: Dict[str, Dict[str, pd.DataFrame]], source: str = "yahoo_finance"
    ) -> Dict[str, Any]:
        """
        Write ingested OHLCV data to MongoDB.

        Args:
            data: Dictionary of {timeframe: {symbol: DataFrame}}
            source: Data source identifier

        Returns:
            Summary of write operations
        """
        if not is_mongodb_available():
            logger.info("MongoDB not available, skipping raw data persistence")
            return {"status": "skipped", "reason": "mongodb_unavailable"}

        summary = {
            "status": "success",
            "timeframes": {},
            "total_records": 0,
            "total_symbols": 0,
        }

        for timeframe, symbols_data in data.items():
            timeframe_summary = {
                "symbols_written": 0,
                "records_written": 0,
                "errors": [],
            }

            for symbol, df in symbols_data.items():
                try:
                    success = MarketDataWriter.write_raw(
                        df=df, symbol=symbol, timeframe=timeframe, source=source
                    )

                    if success:
                        timeframe_summary["symbols_written"] += 1
                        timeframe_summary["records_written"] += len(df)
                    else:
                        timeframe_summary["errors"].append(f"{symbol}: write failed")

                except Exception as e:
                    logger.error(f"Error writing {symbol}/{timeframe} to MongoDB: {e}")
                    timeframe_summary["errors"].append(f"{symbol}: {str(e)}")

            summary["timeframes"][timeframe] = timeframe_summary
            summary["total_records"] += timeframe_summary["records_written"]
            summary["total_symbols"] += timeframe_summary["symbols_written"]

        logger.info(
            f"MongoDB ingestion write complete: {summary['total_symbols']} symbols, "
            f"{summary['total_records']} records"
        )

        return summary


class ValidationDBWriter:
    """
    MongoDB writer for validation pipeline.
    Writes validated, clean data and logs to MongoDB collections.
    """

    @staticmethod
    def write_validation_results(
        validated_data: Dict[str, Dict[str, pd.DataFrame]],
        clean_data: Dict[str, Dict[str, pd.DataFrame]],
        validation_logs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Write validation results to MongoDB.

        Args:
            validated_data: {timeframe: {symbol: DataFrame}} with validation flags
            clean_data: {timeframe: {symbol: DataFrame}} clean records only
            validation_logs: List of validation issue logs

        Returns:
            Summary of write operations
        """
        if not is_mongodb_available():
            logger.info("MongoDB not available, skipping validation data persistence")
            return {"status": "skipped", "reason": "mongodb_unavailable"}

        summary = {
            "status": "success",
            "validated_records": 0,
            "clean_records": 0,
            "log_entries": 0,
            "errors": [],
        }

        # Write validated data
        for timeframe, symbols_data in validated_data.items():
            for symbol, df in symbols_data.items():
                try:
                    success = MarketDataWriter.write_validated(
                        df=df, symbol=symbol, timeframe=timeframe
                    )
                    if success:
                        summary["validated_records"] += len(df)
                except Exception as e:
                    logger.error(f"Error writing validated {symbol}/{timeframe}: {e}")
                    summary["errors"].append(
                        f"validated_{symbol}_{timeframe}: {str(e)}"
                    )

        # Write clean data
        for timeframe, symbols_data in clean_data.items():
            for symbol, df in symbols_data.items():
                try:
                    success = MarketDataWriter.write_clean(
                        df=df, symbol=symbol, timeframe=timeframe
                    )
                    if success:
                        summary["clean_records"] += len(df)
                except Exception as e:
                    logger.error(f"Error writing clean {symbol}/{timeframe}: {e}")
                    summary["errors"].append(f"clean_{symbol}_{timeframe}: {str(e)}")

        # Write validation logs grouped by symbol/timeframe
        log_groups = {}
        for log_entry in validation_logs:
            symbol = log_entry.get("symbol", "unknown")
            timeframe = log_entry.get("timeframe", "unknown")
            key = f"{symbol}_{timeframe}"

            if key not in log_groups:
                log_groups[key] = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "issues": [],
                }
            log_groups[key]["issues"].append(log_entry)

        for key, group in log_groups.items():
            try:
                # Calculate stats
                symbol = group["symbol"]
                timeframe = group["timeframe"]
                issues = group["issues"]

                # Get row counts from data if available
                rows_before = 0
                rows_after = 0

                if timeframe in validated_data and symbol in validated_data[timeframe]:
                    rows_before = len(validated_data[timeframe][symbol])
                if timeframe in clean_data and symbol in clean_data[timeframe]:
                    rows_after = len(clean_data[timeframe][symbol])

                success = ValidationWriter.write_log(
                    symbol=symbol,
                    timeframe=timeframe,
                    validation_results={"total_issues": len(issues)},
                    issues_found=issues,
                    rows_before=rows_before,
                    rows_after=rows_after,
                )

                if success:
                    summary["log_entries"] += 1

            except Exception as e:
                logger.error(f"Error writing validation log for {key}: {e}")
                summary["errors"].append(f"log_{key}: {str(e)}")

        logger.info(
            f"MongoDB validation write complete: {summary['validated_records']} validated, "
            f"{summary['clean_records']} clean, {summary['log_entries']} logs"
        )

        return summary


class FeatureDBWriter:
    """
    MongoDB writer for feature engineering pipeline.
    Writes feature data and normalization params to MongoDB collections.
    """

    @staticmethod
    def write_feature_data(
        feature_data: Dict[str, Dict[str, pd.DataFrame]],
        normalization_params: Optional[Dict[str, Any]] = None,
        version: int = 1,
    ) -> Dict[str, Any]:
        """
        Write feature data to MongoDB.
        Preserves all indicators exactly as generated (including duplicates).

        Args:
            feature_data: {timeframe: {symbol: DataFrame}} with computed features
            normalization_params: Optional normalization parameters to store
            version: Version number for feature set

        Returns:
            Summary of write operations
        """
        if not is_mongodb_available():
            logger.info("MongoDB not available, skipping feature data persistence")
            return {"status": "skipped", "reason": "mongodb_unavailable"}

        summary = {
            "status": "success",
            "feature_records": 0,
            "normalization_params_written": 0,
            "errors": [],
        }

        # Write feature data
        for timeframe, symbols_data in feature_data.items():
            for symbol, df in symbols_data.items():
                try:
                    success = FeatureWriter.write_features(
                        df=df, symbol=symbol, timeframe=timeframe, version=version
                    )
                    if success:
                        summary["feature_records"] += len(df)
                except Exception as e:
                    logger.error(f"Error writing features {symbol}/{timeframe}: {e}")
                    summary["errors"].append(f"features_{symbol}_{timeframe}: {str(e)}")

        # Write normalization params if provided
        if normalization_params:
            for timeframe, symbols_data in feature_data.items():
                for symbol in symbols_data.keys():
                    try:
                        success = FeatureWriter.write_normalization_params(
                            symbol=symbol,
                            timeframe=timeframe,
                            params=normalization_params,
                            version=version,
                        )
                        if success:
                            summary["normalization_params_written"] += 1
                    except Exception as e:
                        logger.error(
                            f"Error writing norm params {symbol}/{timeframe}: {e}"
                        )
                        summary["errors"].append(f"norm_{symbol}_{timeframe}: {str(e)}")

        logger.info(
            f"MongoDB feature write complete: {summary['feature_records']} records, "
            f"{summary['normalization_params_written']} norm params"
        )

        return summary


def integrate_mongodb_with_pipeline():
    """
    Hook function to integrate MongoDB writes with existing pipeline.
    Called after pipeline stages complete successfully.
    """
    logger.info("MongoDB integration layer initialized")
    return {
        "ingestion": IngestionDBWriter,
        "validation": ValidationDBWriter,
        "features": FeatureDBWriter,
    }
