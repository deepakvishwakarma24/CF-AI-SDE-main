"""
Agent Persistence Layer

Provides MongoDB persistence for AI Agent outputs and memory.
Does NOT modify agent logic - wraps execution to persist results.
"""

import os
import sys
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from base_agent import BaseAgent, AgentResponse, MemoryEntry

# Import DB writers
try:
    from db.writers import AgentOutputWriter, AgentMemoryWriter
    from db.readers import AgentOutputReader, FeatureReader
    from db.connection import is_mongodb_available

    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

logger = logging.getLogger(__name__)


def _generate_run_id() -> str:
    """Generate a unique run ID for this execution session."""
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _generate_session_id() -> str:
    """Generate a unique session ID for memory persistence."""
    return f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class AgentResponsePersistence:
    """
    Wraps agent execution to persist AgentResponse outputs to MongoDB.
    Does NOT modify agent behavior - only adds persistence layer.
    """

    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize persistence layer.

        Args:
            run_id: Optional run ID. If not provided, generates a new one.
        """
        self.run_id = run_id or _generate_run_id()
        self.outputs: List[Dict[str, Any]] = []

    def persist_response(
        self,
        agent_name: str,
        response: AgentResponse,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Persist an AgentResponse to MongoDB.

        Args:
            agent_name: Name of the agent
            response: AgentResponse object to persist
            context: Optional context data that was passed to the agent

        Returns:
            True if persisted successfully, False otherwise
        """
        if not DB_AVAILABLE or not is_mongodb_available():
            logger.debug("MongoDB not available, skipping response persistence")
            # Still store in memory
            self.outputs.append(
                {
                    "agent_name": agent_name,
                    "response": (
                        response.model_dump()
                        if hasattr(response, "model_dump")
                        else response.dict()
                    ),
                    "context": context,
                }
            )
            return False

        try:
            response_data = (
                response.model_dump()
                if hasattr(response, "model_dump")
                else response.dict()
            )

            success = AgentOutputWriter.write_output(
                agent_name=agent_name,
                response_data=response_data,
                run_id=self.run_id,
                context=context,
            )

            if success:
                logger.info(f"Persisted response for {agent_name} (run: {self.run_id})")

            # Also store in memory
            self.outputs.append(
                {
                    "agent_name": agent_name,
                    "response": response_data,
                    "context": context,
                }
            )

            return success

        except Exception as e:
            logger.error(f"Failed to persist response for {agent_name}: {e}")
            return False

    def persist_batch(self, responses: Dict[str, AgentResponse]) -> bool:
        """
        Persist multiple agent responses in batch.

        Args:
            responses: Dictionary mapping agent names to AgentResponse objects

        Returns:
            True if all persisted successfully
        """
        if not DB_AVAILABLE or not is_mongodb_available():
            logger.debug("MongoDB not available, skipping batch persistence")
            return False

        try:
            outputs = []
            for agent_name, response in responses.items():
                response_data = (
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else response.dict()
                )
                outputs.append(
                    {
                        "agent_name": agent_name,
                        "response": response_data,
                        "timestamp": (
                            response.timestamp
                            if hasattr(response, "timestamp")
                            else datetime.now(timezone.utc)
                        ),
                    }
                )

            success = AgentOutputWriter.write_batch(outputs=outputs, run_id=self.run_id)

            if success:
                logger.info(
                    f"Persisted {len(outputs)} agent responses (run: {self.run_id})"
                )

            return success

        except Exception as e:
            logger.error(f"Failed to persist batch responses: {e}")
            return False

    def get_run_id(self) -> str:
        """Get the current run ID."""
        return self.run_id

    def get_in_memory_outputs(self) -> List[Dict[str, Any]]:
        """Get outputs stored in memory (fallback when DB unavailable)."""
        return self.outputs


class AgentMemoryPersistence:
    """
    Persists agent memory logs to MongoDB.
    Does NOT modify agent learning logic - only adds persistence.
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize memory persistence.

        Args:
            session_id: Optional session ID. If not provided, generates a new one.
        """
        self.session_id = session_id or _generate_session_id()

    def persist_memory(self, agent: BaseAgent) -> bool:
        """
        Persist an agent's memory log to MongoDB.

        Args:
            agent: BaseAgent instance with memory_log attribute

        Returns:
            True if persisted successfully
        """
        if not DB_AVAILABLE or not is_mongodb_available():
            logger.debug("MongoDB not available, skipping memory persistence")
            return False

        if not hasattr(agent, "memory_log") or len(agent.memory_log) == 0:
            logger.debug(f"No memory to persist for {agent.name}")
            return True

        try:
            # Convert MemoryEntry objects to dicts
            memory_entries = []
            for entry in agent.memory_log:
                if hasattr(entry, "model_dump"):
                    entry_dict = entry.model_dump()
                elif hasattr(entry, "dict"):
                    entry_dict = entry.dict()
                else:
                    entry_dict = {"content": str(entry)}

                memory_entries.append(entry_dict)

            success = AgentMemoryWriter.write_memory(
                agent_name=agent.name,
                memory_entries=memory_entries,
                session_id=self.session_id,
            )

            if success:
                logger.info(
                    f"Persisted {len(memory_entries)} memory entries for {agent.name}"
                )

            return success

        except Exception as e:
            logger.error(f"Failed to persist memory for {agent.name}: {e}")
            return False

    def persist_all_agents(self, agents: List[BaseAgent]) -> Dict[str, bool]:
        """
        Persist memory for all agents.

        Args:
            agents: List of BaseAgent instances

        Returns:
            Dictionary mapping agent names to persistence success status
        """
        results = {}
        for agent in agents:
            results[agent.name] = self.persist_memory(agent)
        return results


class AgentContextAssembler:
    """
    Assembles context for agents from MongoDB data.
    Provides returns and indicators from market_features collection.
    Does NOT modify agent function signatures.
    """

    @staticmethod
    def get_market_context(
        symbol: str = "NIFTY", timeframe: str = "1d", lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Assemble market context from MongoDB for agent consumption.

        Args:
            symbol: Stock symbol
            timeframe: Data timeframe
            lookback_days: Days of history to include

        Returns:
            Context dictionary with returns, volatility, momentum, trend data
        """
        if not DB_AVAILABLE or not is_mongodb_available():
            logger.debug("MongoDB not available for context assembly")
            return {"source": "none", "symbol": symbol, "timeframe": timeframe}

        try:
            context = FeatureReader.get_returns_and_indicators(
                symbol=symbol, timeframe=timeframe, lookback_days=lookback_days
            )

            if context:
                context["source"] = "mongodb"
                return context
            else:
                return {"source": "none", "symbol": symbol, "timeframe": timeframe}

        except Exception as e:
            logger.error(f"Failed to assemble market context: {e}")
            return {
                "source": "error",
                "error": str(e),
                "symbol": symbol,
                "timeframe": timeframe,
            }

    @staticmethod
    def enrich_context(
        base_context: Dict[str, Any],
        symbol: Optional[str] = None,
        timeframe: str = "1d",
    ) -> Dict[str, Any]:
        """
        Enrich existing context with MongoDB data if available.
        Preserves all original context data.

        Args:
            base_context: Original context dictionary
            symbol: Stock symbol (uses context['symbol'] if not provided)
            timeframe: Data timeframe

        Returns:
            Enriched context dictionary
        """
        # Determine symbol
        if symbol is None:
            symbol = base_context.get("symbol", "NIFTY")

        # Get market context from MongoDB
        market_context = AgentContextAssembler.get_market_context(
            symbol=symbol, timeframe=timeframe
        )

        # Merge - original context takes precedence
        enriched = {**market_context, **base_context}
        enriched["mongodb_enriched"] = market_context.get("source") == "mongodb"

        return enriched


class PersistentAgentOrchestrator:
    """
    Wraps agent orchestration with MongoDB persistence.
    Does NOT modify original orchestrator behavior.
    """

    def __init__(self):
        self.response_persistence = AgentResponsePersistence()
        self.memory_persistence = AgentMemoryPersistence()

    def execute_with_persistence(
        self,
        orchestrator,  # AgentOrchestrator instance
        initial_context: Dict[str, Any],
        persist_responses: bool = True,
        persist_memory: bool = True,
        enrich_context: bool = True,
    ) -> Dict[str, AgentResponse]:
        """
        Execute orchestrator pipeline with persistence.

        Args:
            orchestrator: AgentOrchestrator instance
            initial_context: Initial context for agents
            persist_responses: Whether to persist agent responses
            persist_memory: Whether to persist agent memory
            enrich_context: Whether to enrich context with MongoDB data

        Returns:
            Dictionary mapping agent names to responses
        """
        # Optionally enrich context
        if enrich_context:
            initial_context = AgentContextAssembler.enrich_context(initial_context)

        # Execute pipeline (original behavior)
        results = orchestrator.execute_pipeline(initial_context)

        # Persist responses
        if persist_responses:
            self.response_persistence.persist_batch(results)

        # Persist memory
        if persist_memory:
            self.memory_persistence.persist_all_agents(orchestrator.agents)

        return results

    def execute_aggregation_with_persistence(
        self,
        orchestrator,  # AgentOrchestrator instance
        initial_context: Dict[str, Any],
        aggregator_name: str = "SignalAggregatorAgent",
    ) -> AgentResponse:
        """
        Execute with aggregation and persist all outputs.

        Args:
            orchestrator: AgentOrchestrator instance
            initial_context: Initial context for agents
            aggregator_name: Name of aggregator agent

        Returns:
            Final aggregated response
        """
        # Enrich context
        enriched_context = AgentContextAssembler.enrich_context(initial_context)

        # Execute with aggregation
        final_response = orchestrator.execute_with_aggregation(
            enriched_context, aggregator_name=aggregator_name
        )

        # Persist final aggregated response
        self.response_persistence.persist_response(
            agent_name=aggregator_name,
            response=final_response,
            context=enriched_context,
        )

        return final_response

    def get_run_id(self) -> str:
        """Get the current run ID."""
        return self.response_persistence.get_run_id()

    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self.memory_persistence.session_id


# Convenience functions for direct use
def persist_agent_response(
    agent_name: str,
    response: AgentResponse,
    run_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Convenience function to persist a single agent response.

    Args:
        agent_name: Name of the agent
        response: AgentResponse to persist
        run_id: Optional run ID
        context: Optional context data

    Returns:
        True if successful
    """
    persistence = AgentResponsePersistence(run_id=run_id)
    return persistence.persist_response(agent_name, response, context)


def get_latest_agent_output(agent_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the latest output for a specific agent.

    Args:
        agent_name: Name of the agent

    Returns:
        Latest output or None
    """
    if not DB_AVAILABLE or not is_mongodb_available():
        return None

    return AgentOutputReader.get_latest_output(agent_name)


def get_risk_regime_for_ui() -> Optional[Dict[str, Any]]:
    """
    Get risk regime data formatted for UI consumption.

    Returns:
        Risk regime data or None
    """
    if not DB_AVAILABLE or not is_mongodb_available():
        return None

    return AgentOutputReader.get_risk_regime_output()
