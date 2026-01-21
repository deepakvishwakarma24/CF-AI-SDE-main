"""
MongoDB Connection Module
Handles database connections with environment variable support and connection pooling.
"""

import os
import logging
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MONGODB_URI = "mongodb://localhost:27017"
DEFAULT_DATABASE_NAME = "cf_ai_sde"
CONNECTION_TIMEOUT_MS = 5000
SERVER_SELECTION_TIMEOUT_MS = 5000


class MongoDBConnection:
    """
    Singleton MongoDB connection manager.
    Uses environment variables for configuration.
    """

    _instance: Optional["MongoDBConnection"] = None
    _client: Optional[MongoClient] = None
    _database: Optional[Database] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._connect()

    def _connect(self) -> None:
        """Establish MongoDB connection using environment variables."""
        mongodb_uri = os.environ.get("MONGODB_URI", DEFAULT_MONGODB_URI)
        database_name = os.environ.get("MONGODB_DATABASE", DEFAULT_DATABASE_NAME)

        try:
            self._client = MongoClient(
                mongodb_uri,
                connectTimeoutMS=CONNECTION_TIMEOUT_MS,
                serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
                maxPoolSize=50,
                minPoolSize=5,
            )
            # Test connection
            self._client.admin.command("ping")
            self._database = self._client[database_name]
            logger.info(f"Successfully connected to MongoDB: {database_name}")

            # Initialize indexes
            self._create_indexes()

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.warning(
                f"MongoDB connection failed: {e}. Operations will use fallback mode."
            )
            self._client = None
            self._database = None

    def _create_indexes(self) -> None:
        """Create indexes for all collections as per integration plan."""
        if self._database is None:
            return

        try:
            # market_data_raw indexes
            self._database.market_data_raw.create_index(
                [("symbol", 1), ("timestamp", -1), ("timeframe", 1)]
            )
            self._database.market_data_raw.create_index([("ingested_at", -1)])

            # market_data_validated indexes
            self._database.market_data_validated.create_index(
                [("symbol", 1), ("timestamp", -1), ("timeframe", 1)]
            )
            self._database.market_data_validated.create_index([("validated_at", -1)])

            # market_data_clean indexes
            self._database.market_data_clean.create_index(
                [("symbol", 1), ("timestamp", -1), ("timeframe", 1)]
            )

            # market_features indexes
            self._database.market_features.create_index(
                [("symbol", 1), ("timestamp", -1), ("timeframe", 1)]
            )
            self._database.market_features.create_index([("generated_at", -1)])
            self._database.market_features.create_index([("version", -1)])

            # normalization_params indexes
            self._database.normalization_params.create_index(
                [("symbol", 1), ("timeframe", 1), ("version", -1)]
            )

            # validation_log indexes
            self._database.validation_log.create_index(
                [("symbol", 1), ("timeframe", 1), ("validated_at", -1)]
            )

            # agent_outputs indexes
            self._database.agent_outputs.create_index(
                [("agent_name", 1), ("timestamp", -1)]
            )
            self._database.agent_outputs.create_index([("run_id", 1)])
            self._database.agent_outputs.create_index([("created_at", -1)])

            # agent_memory indexes
            self._database.agent_memory.create_index(
                [("agent_name", 1), ("timestamp", -1)]
            )
            self._database.agent_memory.create_index([("session_id", 1)])

            # positions_and_risk indexes (optional collection)
            self._database.positions_and_risk.create_index(
                [("symbol", 1), ("timestamp", -1)]
            )

            logger.info("MongoDB indexes created successfully")

        except Exception as e:
            logger.warning(f"Failed to create some indexes: {e}")

    @property
    def client(self) -> Optional[MongoClient]:
        """Get the MongoDB client."""
        return self._client

    @property
    def database(self) -> Optional[Database]:
        """Get the database instance."""
        return self._database

    def get_collection(self, collection_name: str) -> Optional[Collection]:
        """Get a specific collection."""
        if self._database is None:
            return None
        return self._database[collection_name]

    def is_connected(self) -> bool:
        """Check if MongoDB is connected."""
        if self._client is None:
            return False
        try:
            self._client.admin.command("ping")
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._database = None
            logger.info("MongoDB connection closed")


# Module-level convenience functions
_connection: Optional[MongoDBConnection] = None


def get_connection() -> MongoDBConnection:
    """Get the singleton MongoDB connection instance."""
    global _connection
    if _connection is None:
        _connection = MongoDBConnection()
    return _connection


def get_database() -> Optional[Database]:
    """Get the MongoDB database instance."""
    return get_connection().database


def get_collection(collection_name: str) -> Optional[Collection]:
    """Get a specific MongoDB collection."""
    return get_connection().get_collection(collection_name)


def is_mongodb_available() -> bool:
    """Check if MongoDB is available and connected."""
    return get_connection().is_connected()
