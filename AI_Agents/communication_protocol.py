"""
Communication Protocol Module

Defines the message passing interface and coordination logic for agent communication.
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import json
from dataclasses import dataclass, asdict

from base_agent import BaseAgent, AgentResponse


class MessageType(Enum):
    """Types of messages that can be passed between agents"""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    ERROR = "error"


class MessagePriority(Enum):
    """Priority levels for message routing"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentMessage:
    """
    Standard message format for inter-agent communication.
    """
    sender: str
    receiver: Optional[str]  # None for broadcast messages
    message_type: MessageType
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.MEDIUM
    timestamp: datetime = None
    message_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.message_id is None:
            self.message_id = f"{self.sender}_{self.timestamp.timestamp()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts message to dictionary"""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id
        }
    
    def to_json(self) -> str:
        """Converts message to JSON string"""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Creates message from dictionary"""
        return cls(
            sender=data["sender"],
            receiver=data.get("receiver"),
            message_type=MessageType(data["message_type"]),
            payload=data["payload"],
            priority=MessagePriority(data["priority"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_id=data["message_id"]
        )


class MessageRouter:
    """
    Routes messages between agents based on priority and receiver.
    """
    
    def __init__(self):
        self.message_queue: List[AgentMessage] = []
        self.agents: Dict[str, BaseAgent] = {}
        self.message_history: List[AgentMessage] = []
    
    def register_agent(self, agent: BaseAgent) -> None:
        """
        Registers an agent with the router.
        
        Args:
            agent: BaseAgent instance to register
        """
        self.agents[agent.name] = agent
        print(f"Agent '{agent.name}' registered with router")
    
    def send_message(self, message: AgentMessage) -> None:
        """
        Adds message to queue for processing.
        
        Args:
            message: AgentMessage to send
        """
        self.message_queue.append(message)
        self.message_history.append(message)
    
    def route_message(self, message: AgentMessage) -> Optional[Any]:
        """
        Routes message to appropriate receiver.
        
        Args:
            message: AgentMessage to route
            
        Returns:
            Response from receiver agent, if any
        """
        if message.receiver is None:
            # Broadcast message
            return self._broadcast(message)
        
        receiver_agent = self.agents.get(message.receiver)
        if receiver_agent is None:
            print(f"Warning: Receiver '{message.receiver}' not found")
            return None
        
        # Process message based on type
        if message.message_type == MessageType.REQUEST:
            # Execute the receiver's analyze method with the payload
            response = receiver_agent.analyze(message.payload)
            return response
        
        return None
    
    def _broadcast(self, message: AgentMessage) -> List[Any]:
        """
        Broadcasts message to all registered agents.
        
        Args:
            message: AgentMessage to broadcast
            
        Returns:
            List of responses from all agents
        """
        responses = []
        for agent_name, agent in self.agents.items():
            if agent_name != message.sender:
                response = agent.analyze(message.payload)
                responses.append(response)
        
        return responses
    
    def process_queue(self) -> List[Any]:
        """
        Processes all messages in queue, sorted by priority.
        
        Returns:
            List of responses from message processing
        """
        # Sort by priority (highest first)
        self.message_queue.sort(key=lambda m: m.priority.value, reverse=True)
        
        responses = []
        while len(self.message_queue) > 0:
            message = self.message_queue.pop(0)
            response = self.route_message(message)
            if response is not None:
                responses.append(response)
        
        return responses


class AgentOrchestrator:
    """
    Orchestrates the execution of multiple agents in a coordinated workflow.
    """
    
    def __init__(self):
        self.router = MessageRouter()
        self.agents: List[BaseAgent] = []
        self.execution_order: List[str] = []
    
    def add_agent(self, agent: BaseAgent) -> None:
        """
        Adds agent to orchestrator and registers with router.
        
        Args:
            agent: BaseAgent instance to add
        """
        self.agents.append(agent)
        self.router.register_agent(agent)
    
    def set_execution_order(self, agent_names: List[str]) -> None:
        """
        Defines the order in which agents should be executed.
        
        Args:
            agent_names: List of agent names in execution order
        """
        self.execution_order = agent_names
    
    def execute_pipeline(self, initial_context: Dict[str, Any]) -> Dict[str, AgentResponse]:
        """
        Executes all agents in the defined order.
        
        Args:
            initial_context: Initial data to pass to first agent
            
        Returns:
            Dictionary mapping agent names to their responses
        """
        results = {}
        context = initial_context.copy()
        
        for agent_name in self.execution_order:
            agent = self.router.agents.get(agent_name)
            if agent is None:
                print(f"Warning: Agent '{agent_name}' not found in execution order")
                continue
            
            print(f"Executing {agent_name}...")
            response = agent.analyze(context)
            results[agent_name] = response
            
            # Add response to context for next agent
            context[f"{agent_name}_output"] = response.model_dump()
        
        return results
    
    def execute_with_aggregation(
        self, 
        initial_context: Dict[str, Any],
        aggregator_name: str = "SignalAggregatorAgent"
    ) -> AgentResponse:
        """
        Executes all agents and then runs aggregator for final decision.
        
        Args:
            initial_context: Initial data for agents
            aggregator_name: Name of the aggregator agent
            
        Returns:
            Final aggregated response
        """
        # Execute all non-aggregator agents
        agent_responses = []
        
        for agent in self.agents:
            if agent.name == aggregator_name:
                continue
            
            print(f"Executing {agent.name}...")
            response = agent.analyze(initial_context)
            agent_responses.append(response)
        
        # Run aggregator
        aggregator = self.router.agents.get(aggregator_name)
        if aggregator is None:
            raise ValueError(f"Aggregator '{aggregator_name}' not found")
        
        print(f"\nAggregating results from {len(agent_responses)} agents...")
        final_response = aggregator.analyze({
            "agent_outputs": agent_responses
        })
        
        return final_response


class DataSerializer:
    """
    Helper class for serializing and deserializing complex data types.
    """
    
    @staticmethod
    def serialize_agent_response(response: AgentResponse) -> Dict[str, Any]:
        """
        Converts AgentResponse to JSON-compatible dictionary.
        
        Args:
            response: AgentResponse to serialize
            
        Returns:
            Dictionary representation
        """
        return response.model_dump()
    
    @staticmethod
    def deserialize_agent_response(data: Dict[str, Any]) -> AgentResponse:
        """
        Creates AgentResponse from dictionary.
        
        Args:
            data: Dictionary with response data
            
        Returns:
            AgentResponse instance
        """
        return AgentResponse(**data)
    
    @staticmethod
    def serialize_context(context: Dict[str, Any]) -> str:
        """
        Serializes context dictionary to JSON string.
        
        Args:
            context: Context dictionary
            
        Returns:
            JSON string
        """
        return json.dumps(context, default=str)
    
    @staticmethod
    def deserialize_context(json_str: str) -> Dict[str, Any]:
        """
        Deserializes JSON string to context dictionary.
        
        Args:
            json_str: JSON string
            
        Returns:
            Context dictionary
        """
        return json.loads(json_str)


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    """
    Example usage of the communication protocol.
    """
    from agents import MarketDataAgent, RiskMonitoringAgent, SignalAggregatorAgent
    
    # 1. Create orchestrator
    orchestrator = AgentOrchestrator()
    
    # 2. Create and add agents
    market_agent = MarketDataAgent()
    risk_agent = RiskMonitoringAgent()
    aggregator = SignalAggregatorAgent()
    
    orchestrator.add_agent(market_agent)
    orchestrator.add_agent(risk_agent)
    orchestrator.add_agent(aggregator)
    
    # 3. Execute with aggregation
    initial_context = {
        "symbol": "NIFTY",
        "positions": []  # Empty for demo
    }
    
    final_decision = orchestrator.execute_with_aggregation(initial_context)
    
    print("\n" + "="*50)
    print("FINAL DECISION")
    print("="*50)
    print(f"Summary: {final_decision.summary}")
    print(f"Recommendation: {final_decision.recommendation}")
    print(f"Confidence: {final_decision.confidence_score:.2%}")
