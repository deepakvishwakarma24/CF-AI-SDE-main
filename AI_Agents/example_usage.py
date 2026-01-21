"""
Example Usage Script

Demonstrates how to use the multi-agent trading system.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(__file__))

from base_agent import AgentResponse
from agents import (
    MarketDataAgent,
    RiskMonitoringAgent,
    MacroAgent,
    SentimentAgent,
    VolatilityAgent,
    RegimeDetectionAgent,
    SignalAggregatorAgent
)
from communication_protocol import AgentOrchestrator


def create_mock_data():
    """Creates mock data for testing"""
    # Mock returns data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
    
    # Mock indicators
    indicators = pd.DataFrame({
        'returns': returns,
        'volume': np.random.uniform(1e6, 5e6, len(dates)),
        'volatility': returns.rolling(20).std()
    })
    
    # Mock positions
    positions = pd.DataFrame({
        'symbol': ['NIFTY', 'BANKNIFTY'],
        'quantity': [100, 50],
        'returns': [0.01, -0.005],
        'timestamp': [datetime.now(), datetime.now()]
    })
    
    return returns, indicators, positions


def example_1_single_agent():
    """Example 1: Running a single agent"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Agent Execution")
    print("="*60)
    
    # Create market data agent
    agent = MarketDataAgent()
    
    # Analyze
    context = {"symbol": "NIFTY"}
    response = agent.analyze(context)
    
    print(f"\nAgent: {response.agent_name}")
    print(f"Summary: {response.summary}")
    print(f"Recommendation: {response.recommendation}")
    print(f"Confidence: {response.confidence_score:.2%}")
    print(f"\nStructured Data:")
    for key, value in response.structured_data.items():
        print(f"  {key}: {value}")


def example_2_multiple_agents():
    """Example 2: Running multiple agents independently"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multiple Independent Agents")
    print("="*60)
    
    # Mock data
    returns, indicators, positions = create_mock_data()
    
    # Create agents
    volatility_agent = VolatilityAgent()
    regime_agent = RegimeDetectionAgent()
    
    # Analyze volatility
    vol_response = volatility_agent.analyze({"returns": returns})
    print(f"\n{vol_response.agent_name}:")
    print(f"  {vol_response.summary}")
    
    # Analyze regime
    regime_response = regime_agent.analyze({"indicators": indicators})
    print(f"\n{regime_response.agent_name}:")
    print(f"  {regime_response.summary}")


def example_3_orchestrated_execution():
    """Example 3: Full orchestrated execution with aggregation"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Orchestrated Execution with Aggregation")
    print("="*60)
    
    # Mock data
    returns, indicators, positions = create_mock_data()
    
    # Create orchestrator
    orchestrator = AgentOrchestrator()
    
    # Add all agents
    print("\nRegistering agents...")
    orchestrator.add_agent(MarketDataAgent())
    orchestrator.add_agent(RiskMonitoringAgent())
    orchestrator.add_agent(VolatilityAgent())
    orchestrator.add_agent(RegimeDetectionAgent())
    orchestrator.add_agent(SignalAggregatorAgent())
    
    # Prepare context with all necessary data
    context = {
        "symbol": "NIFTY",
        "returns": returns,
        "indicators": indicators,
        "positions": positions,
        "current_drawdown": 0.03  # 3% drawdown
    }
    
    # Execute with aggregation
    print("\nExecuting agent pipeline...\n")
    final_decision = orchestrator.execute_with_aggregation(context)
    
    # Display results
    print("\n" + "="*60)
    print("FINAL AGGREGATED DECISION")
    print("="*60)
    print(f"\nSummary: {final_decision.summary}")
    print(f"Recommendation: {final_decision.recommendation}")
    print(f"Confidence: {final_decision.confidence_score:.2%}")
    print(f"\nDetailed Analysis:")
    for key, value in final_decision.structured_data.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


def example_4_memory_and_learning():
    """Example 4: Using memory and self-evaluation"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Memory and Learning")
    print("="*60)
    
    agent = VolatilityAgent()
    returns, _, _ = create_mock_data()
    
    # Make prediction
    context = {"returns": returns}
    prediction = agent.analyze(context)
    
    print(f"\nInitial prediction:")
    print(f"  {prediction.summary}")
    print(f"  Performance weight: {agent.performance_weight:.3f}")
    
    # Store memory with accuracy score
    agent.store_memory(
        input_context=context,
        prediction=prediction,
        actual_outcome={"actual_volatility": 0.018},
        accuracy_score=0.85  # 85% accurate
    )
    
    print(f"\nAfter storing memory:")
    print(f"  Performance weight: {agent.performance_weight:.3f}")
    print(f"  Memories stored: {len(agent.memory_log)}")
    
    # Simulate multiple predictions
    for i in range(5):
        prediction = agent.analyze(context)
        # Random accuracy scores
        accuracy = np.random.uniform(0.6, 0.9)
        agent.store_memory(context, prediction, {}, accuracy)
    
    # Get performance statistics
    performance = agent.get_recent_performance(window=5)
    
    print(f"\nPerformance Statistics (last 5 predictions):")
    print(f"  Average Accuracy: {performance['avg_accuracy']:.2%}")
    print(f"  Std Deviation: {performance['std_deviation']:.2%}")
    print(f"  Current Weight: {performance['current_weight']:.3f}")
    print(f"  Total Memories: {performance['total_memories']}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("MULTI-AGENT TRADING SYSTEM - USAGE EXAMPLES")
    print("="*60)
    
    try:
        # Note: Some examples may fail if APIs are not configured
        example_1_single_agent()
    except Exception as e:
        print(f"Example 1 failed: {e}")
        print("(This is expected if MongoDB is not configured)")
    
    example_2_multiple_agents()
    
    try:
        example_3_orchestrated_execution()
    except Exception as e:
        print(f"Example 3 failed: {e}")
        print("(This is expected if APIs are not fully configured)")
    
    example_4_memory_and_learning()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
