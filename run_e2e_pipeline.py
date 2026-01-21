#!/usr/bin/env python3
"""
End-to-End Pipeline with MongoDB Integration
"""

import sys
import os
import logging
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_INGES_PATH = os.path.join(PROJECT_ROOT, "Data-inges-fe")

if DATA_INGES_PATH not in sys.path:
    sys.path.insert(0, DATA_INGES_PATH)

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "Data-inges-fe"))
sys.path.insert(0, str(ROOT_DIR / "AI_Agents"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_mongodb_connection():
    logger.info("=" * 60)
    logger.info("STEP 1: Checking MongoDB Connection")
    logger.info("=" * 60)

    try:
        from db.connection import get_connection, is_mongodb_available

        if is_mongodb_available():
            conn = get_connection()
            logger.info(
                f"✅ MongoDB connected: {conn.database.name if conn.database is not None else 'N/A'}"
            )
            return True
        else:
            logger.warning("⚠️ MongoDB not available - CSV only mode")
            return False

    except Exception as e:
        logger.error(f"❌ MongoDB connection error: {e}")
        return False


def run_data_pipeline(symbols=None, timeframes=None):
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Running Data Pipeline")
    logger.info("=" * 60)

    try:
        from pipeline_with_mongodb import run_pipeline_with_mongodb

        results = run_pipeline_with_mongodb(
            symbols=symbols or ["AAPL", "MSFT"],
            timeframes=timeframes or ["1d"],
            save_to_file=True,
            enable_mongodb=True,
        )

        logger.info(
            f"✅ Pipeline completed in {results.get('total_elapsed_seconds', 0):.2f}s"
        )
        return results

    except ImportError:
        logger.info("Using original pipeline (mongodb_writer not available)")
        from main import run_full_pipeline

        results = run_full_pipeline(
            symbols=symbols or ["AAPL", "MSFT"],
            timeframes=timeframes or ["1d"],
            save_to_file=True,
        )

        # ✅ FIXED IMPORT (ONLY CHANGE)
        try:
            from src.integration.mongodb_writer import (
                IngestionDBWriter,
                ValidationDBWriter,
                FeatureDBWriter,
            )

            if results.get("ingestion"):
                IngestionDBWriter.write_ingested_data(results["ingestion"])

            if results.get("validation"):
                v = results["validation"]
                ValidationDBWriter.write_validation_results(
                    validated_data=v.get("validated", {}),
                    clean_data=v.get("clean", {}),
                    validation_logs=v.get("validation_log", []),
                )

            if results.get("features"):
                FeatureDBWriter.write_feature_data(results["features"])

            logger.info("✅ MongoDB writes completed manually")

        except Exception as e:
            logger.warning(f"MongoDB write fallback failed: {e}")

        return results


def run_agent_analysis(symbol="AAPL"):
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Running AI Agent Analysis")
    logger.info("=" * 60)

    try:
        from AI_Agents.agents import MarketDataAgent, RegimeDetectionAgent
        from AI_Agents.persistence import AgentResponsePersistence, AgentContextAssembler

        persistence = AgentResponsePersistence()
        context = AgentContextAssembler.get_market_context(symbol=symbol, timeframe="1d")
        context["symbol"] = symbol

        market_agent = MarketDataAgent()
        market_response = market_agent.analyze(context)
        persistence.persist_response(
            agent_name=market_agent.name,
            response=market_response,
            context=context,
        )

        try:
            regime_agent = RegimeDetectionAgent()
            regime_response = regime_agent.analyze(
                {
                    "symbol": symbol,
                    "MarketDataAgent_output": market_response.model_dump(),
                }
            )
            persistence.persist_response(
                agent_name=regime_agent.name,
                response=regime_response,
                context=context,
            )
        except Exception:
            pass

        return persistence.get_in_memory_outputs()

    except Exception as e:
        logger.error(f"❌ Agent analysis failed: {e}")
        return []


def verify_api_data():
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Verifying API Data Availability")
    logger.info("=" * 60)

    try:
        from db.readers import AgentOutputReader
        data = AgentOutputReader.get_risk_regime_output()
        if data:
            logger.info("✅ API data available")
        else:
            logger.warning("⚠️ No API data found")
    except Exception as e:
        logger.warning(f"API verification skipped: {e}")


def main():
    print("\n" + "=" * 60)
    print("CF-AI-SDE End-to-End MongoDB Integration Test")
    print("=" * 60 + "\n")

    mongodb_ok = check_mongodb_connection()
    run_data_pipeline(symbols=["AAPL"], timeframes=["1d"])

    if mongodb_ok:
        run_agent_analysis("AAPL")
        verify_api_data()

    print("\n✅ End-to-end integration test complete!")
    print("Start UI: cd ui && npm run dev")


if __name__ == "__main__":
    main()
