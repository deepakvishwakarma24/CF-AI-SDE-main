"""
Base Agent Module

This module provides the foundation for all trading agents in the system.
It defines the abstract BaseAgent class and the AgentResponse data contract.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator
import statistics


class AgentResponse(BaseModel):
    """
    Standardized response format for all agents.
    This enforces a consistent data contract across the agent ecosystem.
    """
    agent_name: str = Field(..., description="Name of the agent that generated this response")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this response was generated")
    summary: str = Field(..., description="Natural language summary of the analysis")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Agent's confidence in this output (0.0-1.0)")
    structured_data: Dict[str, Any] = Field(default_factory=dict, description="JSON-compatible structured data")
    recommendation: str = Field(..., description="Actionable recommendation (e.g., 'BUY', 'SELL', 'HOLD', 'ALERT')")
    
    @field_validator('confidence_score')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence score is between 0 and 1"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryEntry(BaseModel):
    """
    Single entry in an agent's memory log.
    Stores what was predicted vs what actually happened.
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    input_context: Dict[str, Any] = Field(..., description="The input data that was analyzed")
    prediction: AgentResponse = Field(..., description="What the agent predicted")
    actual_outcome: Optional[Dict[str, Any]] = Field(default=None, description="What actually happened")
    accuracy_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="How accurate was the prediction")


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.
    
    Enforces:
    1. Standardized output format (AgentResponse)
    2. Memory/Learning capabilities (store_memory, update_confidence)
    3. Self-evaluation (performance_weight)
    
    All concrete agents must inherit from this class.
    """
    
    def __init__(self, name: str, initial_weight: float = 1.0):
        """
        Initialize the base agent.
        
        Args:
            name: Unique identifier for this agent
            initial_weight: Starting performance weight (default: 1.0)
        """
        self.name = name
        self.memory_log: List[MemoryEntry] = []
        self.performance_weight: float = initial_weight
        
    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Every child agent MUST implement this method.
        
        Args:
            context: Dictionary containing all necessary input data
            
        Returns:
            AgentResponse object with the agent's analysis
        """
        pass
    
    def store_memory(
        self, 
        input_context: Dict[str, Any], 
        prediction: AgentResponse, 
        actual_outcome: Optional[Dict[str, Any]] = None,
        accuracy_score: Optional[float] = None
    ) -> None:
        """
        Stores a prediction in memory for future learning.
        
        This is the "Innovation Opportunity" - agents learn from their mistakes.
        
        Args:
            input_context: The data that was analyzed
            prediction: What the agent predicted
            actual_outcome: What actually happened (can be added later)
            accuracy_score: How accurate the prediction was (0.0-1.0)
        """
        entry = MemoryEntry(
            input_context=input_context,
            prediction=prediction,
            actual_outcome=actual_outcome,
            accuracy_score=accuracy_score
        )
        self.memory_log.append(entry)
        
        # If we have an accuracy score, immediately trigger weight update
        if accuracy_score is not None:
            self.update_confidence()
    
    def update_confidence(self, lookback_window: int = 20) -> None:
        """
        Meta-learning logic: Adjusts performance_weight based on recent accuracy.
        
        This allows the SignalAggregatorAgent to dynamically weight agents
        based on their recent track record.
        
        Args:
            lookback_window: Number of recent predictions to consider
        """
        # Filter memory entries that have accuracy scores
        scored_entries = [
            entry for entry in self.memory_log 
            if entry.accuracy_score is not None
        ]
        
        if len(scored_entries) == 0:
            return  # No data to update with
        
        # Take only the most recent N entries
        recent_entries = scored_entries[-lookback_window:]
        
        # Calculate average accuracy
        recent_scores = [entry.accuracy_score for entry in recent_entries]
        avg_accuracy = statistics.mean(recent_scores)
        
        # Update performance weight
        # Weight formula: Weighted average of current weight and recent accuracy
        # This prevents drastic swings while still responding to performance changes
        alpha = 0.3  # Learning rate
        self.performance_weight = (1 - alpha) * self.performance_weight + alpha * avg_accuracy
        
        # Clamp to reasonable bounds
        self.performance_weight = max(0.1, min(2.0, self.performance_weight))
    
    def get_recent_performance(self, window: int = 10) -> Dict[str, float]:
        """
        Returns statistics about recent performance.
        
        Args:
            window: Number of recent predictions to analyze
            
        Returns:
            Dictionary with performance metrics
        """
        scored_entries = [
            entry for entry in self.memory_log 
            if entry.accuracy_score is not None
        ]
        
        if len(scored_entries) == 0:
            return {
                "avg_accuracy": 0.0,
                "current_weight": self.performance_weight,
                "predictions_evaluated": 0
            }
        
        recent = scored_entries[-window:]
        scores = [e.accuracy_score for e in recent]
        
        return {
            "avg_accuracy": statistics.mean(scores),
            "std_deviation": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "current_weight": self.performance_weight,
            "predictions_evaluated": len(recent),
            "total_memories": len(self.memory_log)
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', weight={self.performance_weight:.3f}, memories={len(self.memory_log)})"
