"""
MongoDB Writers Module
Handles writing data to MongoDB collections with append-only semantics.
All timestamps are in UTC.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import pandas as pd
from pymongo.errors import BulkWriteError, PyMongoError

from .connection import get_collection, is_mongodb_available

logger = logging.getLogger(__name__)


def _get_utc_now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def _dataframe_to_records(
    df: pd.DataFrame, symbol: str, timeframe: str
) -> List[Dict[str, Any]]:
    """Convert DataFrame to list of MongoDB documents."""
    records = []
    for idx, row in df.iterrows():
        record = row.to_dict()
        record["symbol"] = symbol
        record["timeframe"] = timeframe
        # Handle timestamp/index
        if isinstance(idx, pd.Timestamp):
            record["timestamp"] = idx.to_pydatetime().replace(tzinfo=timezone.utc)
        elif "Date" in record:
            if isinstance(record["Date"], pd.Timestamp):
                record["timestamp"] = (
                    record["Date"].to_pydatetime().replace(tzinfo=timezone.utc)
                )
            else:
                record["timestamp"] = record["Date"]
        elif "Datetime" in record:
            if isinstance(record["Datetime"], pd.Timestamp):
                record["timestamp"] = (
                    record["Datetime"].to_pydatetime().replace(tzinfo=timezone.utc)
                )
            else:
                record["timestamp"] = record["Datetime"]
        records.append(record)
    return records


class MarketDataWriter:
    """Writer for market data collections (raw, validated, clean)."""

    @staticmethod
    def write_raw(
        df: pd.DataFrame, symbol: str, timeframe: str, source: str = "yahoo_finance"
    ) -> bool:
        """
        Write raw OHLCV data to market_data_raw collection.
        Append-only - does not update existing records.
        """
        if not is_mongodb_available():
            logger.debug("MongoDB not available, skipping raw data write")
            return False

        collection = get_collection("market_data_raw")
        if collection is None:
            return False

        try:
            records = _dataframe_to_records(df, symbol, timeframe)
            ingested_at = _get_utc_now()

            for record in records:
                record["source"] = source
                record["ingested_at"] = ingested_at

            if records:
                collection.insert_many(records, ordered=False)
                logger.info(
                    f"Wrote {len(records)} raw records for {symbol}/{timeframe}"
                )
            return True

        except BulkWriteError as e:
            # Some documents may have been inserted
            logger.warning(
                f"Bulk write partial failure for raw data: {e.details.get('nInserted', 0)} inserted"
            )
            return True
        except PyMongoError as e:
            logger.error(f"Failed to write raw data: {e}")
            return False

    @staticmethod
    def write_validated(df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """Write validated OHLCV data to market_data_validated collection."""
        if not is_mongodb_available():
            logger.debug("MongoDB not available, skipping validated data write")
            return False

        collection = get_collection("market_data_validated")
        if collection is None:
            return False

        try:
            records = _dataframe_to_records(df, symbol, timeframe)
            validated_at = _get_utc_now()

            for record in records:
                record["validated_at"] = validated_at

            if records:
                collection.insert_many(records, ordered=False)
                logger.info(
                    f"Wrote {len(records)} validated records for {symbol}/{timeframe}"
                )
            return True

        except BulkWriteError as e:
            logger.warning(
                f"Bulk write partial failure for validated data: {e.details.get('nInserted', 0)} inserted"
            )
            return True
        except PyMongoError as e:
            logger.error(f"Failed to write validated data: {e}")
            return False

    @staticmethod
    def write_clean(df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """Write clean OHLCV data to market_data_clean collection."""
        if not is_mongodb_available():
            logger.debug("MongoDB not available, skipping clean data write")
            return False

        collection = get_collection("market_data_clean")
        if collection is None:
            return False

        try:
            records = _dataframe_to_records(df, symbol, timeframe)
            cleaned_at = _get_utc_now()

            for record in records:
                record["cleaned_at"] = cleaned_at

            if records:
                collection.insert_many(records, ordered=False)
                logger.info(
                    f"Wrote {len(records)} clean records for {symbol}/{timeframe}"
                )
            return True

        except BulkWriteError as e:
            logger.warning(
                f"Bulk write partial failure for clean data: {e.details.get('nInserted', 0)} inserted"
            )
            return True
        except PyMongoError as e:
            logger.error(f"Failed to write clean data: {e}")
            return False


class ValidationWriter:
    """Writer for validation logs."""

    @staticmethod
    def write_log(
        symbol: str,
        timeframe: str,
        validation_results: Dict[str, Any],
        issues_found: List[Dict[str, Any]],
        rows_before: int,
        rows_after: int,
    ) -> bool:
        """Write validation log entry to validation_log collection."""
        if not is_mongodb_available():
            logger.debug("MongoDB not available, skipping validation log write")
            return False

        collection = get_collection("validation_log")
        if collection is None:
            return False

        try:
            log_entry = {
                "symbol": symbol,
                "timeframe": timeframe,
                "validated_at": _get_utc_now(),
                "validation_results": validation_results,
                "issues_found": issues_found,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "rows_removed": rows_before - rows_after,
                "pass_rate": rows_after / rows_before if rows_before > 0 else 0.0,
            }

            collection.insert_one(log_entry)
            logger.info(f"Wrote validation log for {symbol}/{timeframe}")
            return True

        except PyMongoError as e:
            logger.error(f"Failed to write validation log: {e}")
            return False


class FeatureWriter:
    """Writer for feature data and normalization parameters."""

    @staticmethod
    def write_features(
        df: pd.DataFrame, symbol: str, timeframe: str, version: int = 1
    ) -> bool:
        """
        Write feature data to market_features collection.
        Includes all generated indicators exactly as produced.
        """
        if not is_mongodb_available():
            logger.debug("MongoDB not available, skipping features write")
            return False

        collection = get_collection("market_features")
        if collection is None:
            return False

        try:
            records = _dataframe_to_records(df, symbol, timeframe)
            generated_at = _get_utc_now()

            for record in records:
                record["version"] = version
                record["generated_at"] = generated_at
                # Store all columns including duplicated indicators
                record["feature_count"] = len(
                    [
                        k
                        for k in record.keys()
                        if k
                        not in [
                            "symbol",
                            "timeframe",
                            "timestamp",
                            "version",
                            "generated_at",
                        ]
                    ]
                )

            if records:
                collection.insert_many(records, ordered=False)
                logger.info(
                    f"Wrote {len(records)} feature records for {symbol}/{timeframe} (v{version})"
                )
            return True

        except BulkWriteError as e:
            logger.warning(
                f"Bulk write partial failure for features: {e.details.get('nInserted', 0)} inserted"
            )
            return True
        except PyMongoError as e:
            logger.error(f"Failed to write features: {e}")
            return False

    @staticmethod
    def write_normalization_params(
        symbol: str, timeframe: str, params: Dict[str, Any], version: int = 1
    ) -> bool:
        """Write normalization parameters to normalization_params collection."""
        if not is_mongodb_available():
            logger.debug("MongoDB not available, skipping normalization params write")
            return False

        collection = get_collection("normalization_params")
        if collection is None:
            return False

        try:
            doc = {
                "symbol": symbol,
                "timeframe": timeframe,
                "version": version,
                "created_at": _get_utc_now(),
                "parameters": params,
            }

            collection.insert_one(doc)
            logger.info(
                f"Wrote normalization params for {symbol}/{timeframe} (v{version})"
            )
            return True

        except PyMongoError as e:
            logger.error(f"Failed to write normalization params: {e}")
            return False


class AgentOutputWriter:
    """Writer for agent outputs (AgentResponse persistence)."""

    @staticmethod
    def write_output(
        agent_name: str,
        response_data: Dict[str, Any],
        run_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Write agent output to agent_outputs collection.
        Preserves exact AgentResponse structure.
        """
        if not is_mongodb_available():
            logger.debug("MongoDB not available, skipping agent output write")
            return False

        collection = get_collection("agent_outputs")
        if collection is None:
            return False

        try:
            doc = {
                "agent_name": agent_name,
                "run_id": run_id,
                "created_at": _get_utc_now(),
                "timestamp": response_data.get("timestamp", _get_utc_now()),
                "response": response_data,
                "context": context or {},
            }

            collection.insert_one(doc)
            logger.info(f"Wrote output for agent: {agent_name} (run: {run_id})")
            return True

        except PyMongoError as e:
            logger.error(f"Failed to write agent output: {e}")
            return False

    @staticmethod
    def write_batch(outputs: List[Dict[str, Any]], run_id: str) -> bool:
        """Write multiple agent outputs in batch."""
        if not is_mongodb_available():
            logger.debug("MongoDB not available, skipping batch agent output write")
            return False

        collection = get_collection("agent_outputs")
        if collection is None:
            return False

        try:
            created_at = _get_utc_now()
            docs = []

            for output in outputs:
                doc = {
                    "agent_name": output.get("agent_name"),
                    "run_id": run_id,
                    "created_at": created_at,
                    "timestamp": output.get("timestamp", created_at),
                    "response": output.get("response", {}),
                    "context": output.get("context", {}),
                }
                docs.append(doc)

            if docs:
                collection.insert_many(docs, ordered=False)
                logger.info(f"Wrote {len(docs)} agent outputs (run: {run_id})")
            return True

        except BulkWriteError as e:
            logger.warning(
                f"Bulk write partial failure for agent outputs: {e.details.get('nInserted', 0)} inserted"
            )
            return True
        except PyMongoError as e:
            logger.error(f"Failed to write agent outputs batch: {e}")
            return False


class AgentMemoryWriter:
    """Writer for agent memory logs."""

    @staticmethod
    def write_memory(
        agent_name: str, memory_entries: List[Dict[str, Any]], session_id: str
    ) -> bool:
        """Write agent memory entries to agent_memory collection."""
        if not is_mongodb_available():
            logger.debug("MongoDB not available, skipping memory write")
            return False

        collection = get_collection("agent_memory")
        if collection is None:
            return False

        try:
            created_at = _get_utc_now()
            docs = []

            for entry in memory_entries:
                doc = {
                    "agent_name": agent_name,
                    "session_id": session_id,
                    "created_at": created_at,
                    "timestamp": entry.get("timestamp", created_at),
                    "memory_type": entry.get("type", "general"),
                    "content": entry.get("content", entry),
                    "metadata": entry.get("metadata", {}),
                }
                docs.append(doc)

            if docs:
                collection.insert_many(docs, ordered=False)
                logger.info(f"Wrote {len(docs)} memory entries for agent: {agent_name}")
            return True

        except BulkWriteError as e:
            logger.warning(
                f"Bulk write partial failure for agent memory: {e.details.get('nInserted', 0)} inserted"
            )
            return True
        except PyMongoError as e:
            logger.error(f"Failed to write agent memory: {e}")
            return False


class PositionsRiskWriter:
    """Writer for positions and risk data (optional collection)."""

    @staticmethod
    def write_position(
        symbol: str, position_data: Dict[str, Any], risk_metrics: Dict[str, Any]
    ) -> bool:
        """Write position and risk data to positions_and_risk collection."""
        if not is_mongodb_available():
            logger.debug("MongoDB not available, skipping position write")
            return False

        collection = get_collection("positions_and_risk")
        if collection is None:
            return False

        try:
            doc = {
                "symbol": symbol,
                "timestamp": _get_utc_now(),
                "position": position_data,
                "risk_metrics": risk_metrics,
            }

            collection.insert_one(doc)
            logger.info(f"Wrote position/risk data for {symbol}")
            return True

        except PyMongoError as e:
            logger.error(f"Failed to write position data: {e}")
            return False
