"""
MongoDB Readers Module
Handles reading data from MongoDB collections.
Provides fallback behavior when MongoDB is unavailable.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
from pymongo import DESCENDING
from pymongo.errors import PyMongoError

from .connection import get_collection, is_mongodb_available

logger = logging.getLogger(__name__)


class MarketDataReader:
    """Reader for market data collections."""

    @staticmethod
    def read_raw(
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Read raw OHLCV data from market_data_raw collection.
        Returns None if MongoDB unavailable or no data found.
        """
        if not is_mongodb_available():
            logger.debug("MongoDB not available, returning None for raw data read")
            return None

        collection = get_collection("market_data_raw")
        if collection is None:
            return None

        try:
            query = {"symbol": symbol, "timeframe": timeframe}

            if start_date or end_date:
                query["timestamp"] = {}
                if start_date:
                    query["timestamp"]["$gte"] = start_date
                if end_date:
                    query["timestamp"]["$lte"] = end_date

            cursor = collection.find(query).sort("timestamp", DESCENDING)

            if limit:
                cursor = cursor.limit(limit)

            records = list(cursor)

            if not records:
                logger.debug(f"No raw data found for {symbol}/{timeframe}")
                return None

            df = pd.DataFrame(records)

            # Clean up MongoDB-specific fields
            if "_id" in df.columns:
                df = df.drop("_id", axis=1)

            # Set timestamp as index if present
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()

            logger.info(f"Read {len(df)} raw records for {symbol}/{timeframe}")
            return df

        except PyMongoError as e:
            logger.error(f"Failed to read raw data: {e}")
            return None

    @staticmethod
    def read_clean(
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """Read clean OHLCV data from market_data_clean collection."""
        if not is_mongodb_available():
            logger.debug("MongoDB not available, returning None for clean data read")
            return None

        collection = get_collection("market_data_clean")
        if collection is None:
            return None

        try:
            query = {"symbol": symbol, "timeframe": timeframe}

            if start_date or end_date:
                query["timestamp"] = {}
                if start_date:
                    query["timestamp"]["$gte"] = start_date
                if end_date:
                    query["timestamp"]["$lte"] = end_date

            cursor = collection.find(query).sort("timestamp", DESCENDING)

            if limit:
                cursor = cursor.limit(limit)

            records = list(cursor)

            if not records:
                logger.debug(f"No clean data found for {symbol}/{timeframe}")
                return None

            df = pd.DataFrame(records)

            if "_id" in df.columns:
                df = df.drop("_id", axis=1)

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()

            logger.info(f"Read {len(df)} clean records for {symbol}/{timeframe}")
            return df

        except PyMongoError as e:
            logger.error(f"Failed to read clean data: {e}")
            return None

    @staticmethod
    def get_latest(symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get the latest raw data record for a symbol/timeframe."""
        if not is_mongodb_available():
            return None

        collection = get_collection("market_data_raw")
        if collection is None:
            return None

        try:
            record = collection.find_one(
                {"symbol": symbol, "timeframe": timeframe},
                sort=[("timestamp", DESCENDING)],
            )

            if record and "_id" in record:
                del record["_id"]

            return record

        except PyMongoError as e:
            logger.error(f"Failed to get latest data: {e}")
            return None


class FeatureReader:
    """Reader for feature data and normalization parameters."""

    @staticmethod
    def read_features(
        symbol: str,
        timeframe: str,
        version: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Read feature data from market_features collection.
        If version not specified, reads latest version.
        """
        if not is_mongodb_available():
            logger.debug("MongoDB not available, returning None for features read")
            return None

        collection = get_collection("market_features")
        if collection is None:
            return None

        try:
            query = {"symbol": symbol, "timeframe": timeframe}

            if version:
                query["version"] = version

            if start_date or end_date:
                query["timestamp"] = {}
                if start_date:
                    query["timestamp"]["$gte"] = start_date
                if end_date:
                    query["timestamp"]["$lte"] = end_date

            cursor = collection.find(query).sort(
                [("version", DESCENDING), ("timestamp", DESCENDING)]
            )

            if limit:
                cursor = cursor.limit(limit)

            records = list(cursor)

            if not records:
                logger.debug(f"No features found for {symbol}/{timeframe}")
                return None

            df = pd.DataFrame(records)

            if "_id" in df.columns:
                df = df.drop("_id", axis=1)

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()

            logger.info(f"Read {len(df)} feature records for {symbol}/{timeframe}")
            return df

        except PyMongoError as e:
            logger.error(f"Failed to read features: {e}")
            return None

    @staticmethod
    def get_normalization_params(
        symbol: str, timeframe: str, version: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Get normalization parameters for a symbol/timeframe."""
        if not is_mongodb_available():
            return None

        collection = get_collection("normalization_params")
        if collection is None:
            return None

        try:
            query = {"symbol": symbol, "timeframe": timeframe}
            if version:
                query["version"] = version

            record = collection.find_one(query, sort=[("version", DESCENDING)])

            if record:
                if "_id" in record:
                    del record["_id"]
                return record.get("parameters", record)

            return None

        except PyMongoError as e:
            logger.error(f"Failed to get normalization params: {e}")
            return None

    @staticmethod
    def get_returns_and_indicators(
        symbol: str, timeframe: str, lookback_days: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Get returns and key indicators for agent context assembly.
        Returns dict with returns, volatility metrics, and momentum indicators.
        """
        if not is_mongodb_available():
            return None

        collection = get_collection("market_features")
        if collection is None:
            return None

        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=lookback_days)

            records = list(
                collection.find(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "timestamp": {"$gte": start_date, "$lte": end_date},
                    },
                    sort=[("timestamp", DESCENDING)],
                ).limit(lookback_days * 2)
            )  # Extra buffer for weekends/holidays

            if not records:
                return None

            # Extract key metrics for agent context
            latest = records[0]

            context = {
                "symbol": symbol,
                "timeframe": timeframe,
                "latest_timestamp": latest.get("timestamp"),
                "returns": {},
                "volatility": {},
                "momentum": {},
                "trend": {},
            }

            # Returns
            for key in ["Returns", "Log_Returns", "returns", "log_returns"]:
                if key in latest:
                    context["returns"][key.lower()] = latest[key]

            # Volatility
            for key in ["ATR", "ATR_14", "Volatility", "BB_Width", "atr", "volatility"]:
                if key in latest:
                    context["volatility"][key.lower()] = latest[key]

            # Momentum
            for key in [
                "RSI",
                "RSI_14",
                "MACD",
                "MACD_Signal",
                "Stoch_K",
                "Stoch_D",
                "rsi",
                "macd",
                "stoch_k",
                "stoch_d",
            ]:
                if key in latest:
                    context["momentum"][key.lower()] = latest[key]

            # Trend
            for key in [
                "SMA_20",
                "SMA_50",
                "EMA_12",
                "EMA_26",
                "ADX",
                "Trend",
                "sma_20",
                "sma_50",
                "ema_12",
                "ema_26",
                "adx",
            ]:
                if key in latest:
                    context["trend"][key.lower()] = latest[key]

            return context

        except PyMongoError as e:
            logger.error(f"Failed to get returns and indicators: {e}")
            return None


class AgentOutputReader:
    """Reader for agent outputs."""

    @staticmethod
    def get_latest_output(agent_name: str) -> Optional[Dict[str, Any]]:
        """Get the latest output for a specific agent."""
        if not is_mongodb_available():
            logger.debug("MongoDB not available, returning None for agent output read")
            return None

        collection = get_collection("agent_outputs")
        if collection is None:
            return None

        try:
            record = collection.find_one(
                {"agent_name": agent_name}, sort=[("created_at", DESCENDING)]
            )

            if record and "_id" in record:
                del record["_id"]

            return record

        except PyMongoError as e:
            logger.error(f"Failed to get latest agent output: {e}")
            return None

    @staticmethod
    def get_latest_run_outputs(run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all agent outputs from the latest run.
        If run_id not specified, gets outputs from most recent run.
        """
        if not is_mongodb_available():
            logger.debug("MongoDB not available, returning empty list for run outputs")
            return []

        collection = get_collection("agent_outputs")
        if collection is None:
            return []

        try:
            if run_id is None:
                # Get the latest run_id
                latest = collection.find_one(sort=[("created_at", DESCENDING)])
                if latest is None:
                    return []
                run_id = latest.get("run_id")

            records = list(collection.find({"run_id": run_id}))

            for record in records:
                if "_id" in record:
                    del record["_id"]

            logger.info(f"Read {len(records)} agent outputs for run: {run_id}")
            return records

        except PyMongoError as e:
            logger.error(f"Failed to get run outputs: {e}")
            return []

    @staticmethod
    def get_risk_regime_output() -> Optional[Dict[str, Any]]:
        """
        Get the latest risk regime output for UI consumption.
        Returns data formatted for /api/risk-regime endpoint.
        """
        if not is_mongodb_available():
            logger.debug("MongoDB not available, returning None for risk regime")
            return None

        collection = get_collection("agent_outputs")
        if collection is None:
            return None

        try:
            # Try RegimeDetectionAgent first
            record = collection.find_one(
                {"agent_name": "RegimeDetectionAgent"},
                sort=[("created_at", DESCENDING)],
            )

            if record is None:
                # Fallback to RiskMonitoringAgent
                record = collection.find_one(
                    {"agent_name": "RiskMonitoringAgent"},
                    sort=[("created_at", DESCENDING)],
                )

            if record is None:
                return None

            if "_id" in record:
                del record["_id"]

            # Extract response data
            response = record.get("response", {})

            return {
                "agent_name": record.get("agent_name"),
                "timestamp": record.get("timestamp"),
                "created_at": record.get("created_at"),
                "run_id": record.get("run_id"),
                "signal": response.get("signal"),
                "confidence": response.get("confidence"),
                "reasoning": response.get("reasoning"),
                "data": response.get("data", {}),
                "metadata": response.get("metadata", {}),
            }

        except PyMongoError as e:
            logger.error(f"Failed to get risk regime output: {e}")
            return None

    @staticmethod
    def get_aggregated_signals() -> Optional[Dict[str, Any]]:
        """Get the latest aggregated signals from SignalAggregatorAgent."""
        if not is_mongodb_available():
            return None

        collection = get_collection("agent_outputs")
        if collection is None:
            return None

        try:
            record = collection.find_one(
                {"agent_name": "SignalAggregatorAgent"},
                sort=[("created_at", DESCENDING)],
            )

            if record:
                if "_id" in record:
                    del record["_id"]
                return record

            return None

        except PyMongoError as e:
            logger.error(f"Failed to get aggregated signals: {e}")
            return None


class ValidationLogReader:
    """Reader for validation logs."""

    @staticmethod
    def get_latest_log(symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get the latest validation log for a symbol/timeframe."""
        if not is_mongodb_available():
            return None

        collection = get_collection("validation_log")
        if collection is None:
            return None

        try:
            record = collection.find_one(
                {"symbol": symbol, "timeframe": timeframe},
                sort=[("validated_at", DESCENDING)],
            )

            if record and "_id" in record:
                del record["_id"]

            return record

        except PyMongoError as e:
            logger.error(f"Failed to get validation log: {e}")
            return None

    @staticmethod
    def get_validation_stats(
        symbol: str, timeframe: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent validation statistics for a symbol/timeframe."""
        if not is_mongodb_available():
            return []

        collection = get_collection("validation_log")
        if collection is None:
            return []

        try:
            records = list(
                collection.find(
                    {"symbol": symbol, "timeframe": timeframe},
                    sort=[("validated_at", DESCENDING)],
                ).limit(limit)
            )

            for record in records:
                if "_id" in record:
                    del record["_id"]

            return records

        except PyMongoError as e:
            logger.error(f"Failed to get validation stats: {e}")
            return []
