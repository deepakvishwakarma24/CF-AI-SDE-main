# AI Agents - Multi-Agent Trading System

A sophisticated OOP-based multi-agent system for financial trading analysis, featuring 7 specialized agents with self-learning capabilities and LLM-powered decision synthesis.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Module Documentation](#module-documentation)
4. [Quick Start Guide](#quick-start-guide)
5. [Usage Examples](#usage-examples)
6. [Agent Descriptions](#agent-descriptions)
7. [Integration Guide](#integration-guide)
8. [Advanced Features](#advanced-features)

---

## Overview

### What This System Does

The AI Agents system provides a **production-ready framework** for multi-faceted financial market analysis. It coordinates 7 specialized agents that analyze different aspects of market conditions:

- **Market Data**: Detects anomalies in price and volume
- **Risk Monitoring**: Calculates VaR and checks portfolio limits
- **Macro Analysis**: Tracks economic events and historical impacts
- **Sentiment**: Analyzes news using FinBERT
- **Volatility**: Forecasts volatility using GARCH/LSTM models
- **Regime Detection**: Classifies market states (Bull/Bear/Range/Crisis)
- **Signal Aggregator**: Synthesizes all signals using LLM reasoning

### Key Features

✅ **Self-Learning**: Agents track their accuracy and auto-adjust influence weights  
✅ **ML Integration**: Direct imports from your `ML_Models` directory  
✅ **MongoDB Native**: Time-series data storage with `pymongo`  
✅ **LLM Synthesis**: Gemini for complex reasoning, Groq for lightweight tasks  
✅ **Production-Ready**: All APIs implemented, just add keys to `.env`  
✅ **Conflict Resolution**: Built-in logic for contradictory signals  

---

## System Architecture

```
┌────────────────────────────────────────────────────────┐
│              AgentOrchestrator                          │
│         (communication_protocol.py)                     │
└──────────────────┬─────────────────────────────────────┘
                   │
       ┌───────────┼───────────┐
       ↓           ↓           ↓
  Deterministic  Probabilistic  Synthesis
     Agents        Agents        Agent
       │             │             │
   ┌───┴───┐    ┌───┴───┐    ┌────┴────┐
   │Market │    │Sentiment│   │Aggregator│
   │ Data  │    │        │   │  (LLM)  │
   ├───────┤    ├────────┤   └──────────┘
   │ Risk  │    │Volatility│
   ├───────┤    ├────────┤
   │ Macro │    │ Regime │
   └───────┘    └────────┘
       │            │
       ↓            ↓
   MongoDB    ML_Models/
```

### Data Flow

```
User Input (symbol, dates)
    ↓
AgentOrchestrator.execute_with_aggregation()
    ↓
[All Agents Run in Parallel]
    ↓
SignalAggregatorAgent receives all responses
    ↓
Conflict resolution + LLM synthesis
    ↓
Final Decision (AgentResponse)
```

---

## Module Documentation

### 1. `base_agent.py` - Foundation

**Purpose**: Defines the core abstractions that all agents inherit from.

#### Key Classes

##### `AgentResponse` (Pydantic Model)
Standardized output format enforced across all agents.

```python
class AgentResponse(BaseModel):
    agent_name: str              # Who generated this
    timestamp: datetime          # When it was generated
    summary: str                 # Natural language summary
    confidence_score: float      # 0.0 to 1.0
    structured_data: Dict        # JSON-compatible data
    recommendation: str          # BUY/SELL/HOLD/ALERT/etc.
```

**Usage**:
```python
response = agent.analyze(context)
print(response.summary)
print(response.recommendation)
print(response.confidence_score)
```

##### `BaseAgent` (Abstract Base Class)
All concrete agents inherit from this class.

**Attributes**:
- `name`: Agent identifier
- `memory_log`: List of past predictions vs outcomes
- `performance_weight`: Self-evaluated confidence (0.0-2.0)

**Key Methods**:

| Method                                           | Description                                           | When to Use                     |
| ------------------------------------------------ | ----------------------------------------------------- | ------------------------------- |
| `analyze(context)`                               | **Abstract** - Must be implemented by children        | Override in every agent         |
| `store_memory(input, output, outcome, accuracy)` | Logs prediction for learning                          | After receiving actual outcome  |
| `update_confidence()`                            | Adjusts `performance_weight` based on recent accuracy | Auto-called by `store_memory()` |
| `get_recent_performance(window=10)`              | Returns accuracy statistics                           | Debugging/monitoring            |

**Example**:
```python
# Create agent
agent = VolatilityAgent()

# Make prediction
prediction = agent.analyze({"returns": returns_series})

# Later, when actual outcome is known
agent.store_memory(
    input_context={"returns": returns_series},
    prediction=prediction,
    actual_outcome={"actual_vol": 0.018},
    accuracy_score=0.85  # 85% accurate
)

# Check performance
stats = agent.get_recent_performance(window=10)
print(f"Avg Accuracy: {stats['avg_accuracy']:.2%}")
print(f"Current Weight: {stats['current_weight']:.3f}")
```

---

### 2. `agents.py` - Concrete Agents

**Purpose**: Contains all 7 specialized agent implementations.

#### Deterministic Agents

##### `MarketDataAgent`

**What it does**: Fetches OHLCV data from MongoDB and detects anomalies.

**Required Context**:
```python
context = {
    "symbol": "NIFTY"  # Stock symbol
}
```

**What it detects**:
- Gap opens >2% from previous close
- Volume spikes >2x 20-day average
- Volatility outliers >2 standard deviations

**Example**:
```python
from agents import MarketDataAgent

agent = MarketDataAgent()
response = agent.analyze({"symbol": "NIFTY"})

if response.recommendation == "ALERT":
    print(f"⚠️ {response.summary}")
    print(f"Anomalies: {response.structured_data['anomalies']}")
```

**Configuration**:
- Set `MONGODB_URI` in `.env`
- Ensure MongoDB has collection with schema (see CONFIGURATION.md)

---

##### `RiskMonitoringAgent`

**What it does**: Calculates Value at Risk (VaR) and checks against limits.

**Required Context**:
```python
context = {
    "positions": pd.DataFrame({
        'symbol': ['NIFTY', 'BANKNIFTY'],
        'quantity': [100, 50],
        'returns': [0.01, -0.005],
        'timestamp': [datetime.now(), datetime.now()]
    }),
    "current_drawdown": 0.03  # 3% drawdown
}
```

**What it checks**:
- VaR breach (default limit: 5%)
- Drawdown breach (default limit: 10%)

**Example**:
```python
agent = RiskMonitoringAgent()
response = agent.analyze(context)

if response.recommendation == "REDUCE_EXPOSURE":
    print("🚨 Risk limits breached!")
    breaches = response.structured_data['breaches']
    print(f"VaR: {breaches['details']['var']}")
```

**Configuration**:
- Set `VAR_LIMIT` in `.env` (default: 0.05)
- Set `DRAWDOWN_LIMIT` in `.env` (default: 0.10)

---

##### `MacroAgent`

**What it does**: Fetches upcoming economic events and historical impacts.

**Required Context**:
```python
context = {}  # No specific requirements
```

**What it returns**:
- List of high-impact events in next 7 days
- Historical market reactions to similar events

**Example**:
```python
agent = MacroAgent()
response = agent.analyze({})

events = response.structured_data['events']
for event in events:
    print(f"📅 {event['Event']} on {event['Date']}")
```

**Configuration**:
- Set `ECONOMIC_CALENDAR_API_KEY` in `.env`
- Optional: Set `ECONOMIC_CALENDAR_API_URL` (defaults to TradingEconomics)

---

#### Probabilistic Agents

##### `SentimentAgent`

**What it does**: Analyzes news sentiment using FinBERT.

**Required Context**:
```python
context = {}  # Optional parameters (defaults work)
```

**What it analyzes**:
- Last 24 hours of news headlines
- Sentiment scores (positive/negative/neutral)
- Sentiment momentum (improving/stable/deteriorating)

**Example**:
```python
agent = SentimentAgent()
response = agent.analyze({})

print(f"Sentiment: {response.recommendation}")  # BULLISH/BEARISH/NEUTRAL
print(f"Score: {response.structured_data['sentiment_score']}")
print(f"News analyzed: {response.structured_data['news_count']}")
```

**Configuration**:
- Set `NEWS_API_KEY` in `.env`
- FinBERT downloads automatically on first use

---

##### `VolatilityAgent`

**What it does**: Forecasts volatility using GARCH/EGARCH models from `ML_Models`.

**Required Context**:
```python
context = {
    "returns": pd.Series([0.01, -0.005, 0.003, ...])  # Log returns
}
```

**What it forecasts**:
- Next-day volatility using GARCH
- Next-day volatility using EGARCH
- Regime detection (expanding/compressing/stable)

**Example**:
```python
agent = VolatilityAgent()
response = agent.analyze({"returns": returns_series})

forecast = response.structured_data['forecast_details']
print(f"GARCH forecast: {forecast['garch_volatility']:.2%}")
print(f"EGARCH forecast: {forecast['egarch_volatility']:.2%}")
print(f"Regime: {response.structured_data['regime']}")
```

**ML Integration**:
```python
# The agent internally does:
from Volatility_Forecasting import Volatility_Models
vol_models = Volatility_Models(returns=returns)
forecast = vol_models.predict_volatility(horizon=1)
```

**Data Requirements**:
- Minimum 30 days of log returns
- Returns should be pandas Series

---

##### `RegimeDetectionAgent`

**What it does**: Classifies market regime and suggests strategy.

**Required Context**:
```python
context = {
    "indicators": pd.DataFrame({
        'returns': [...],
        'volume': [...],
        'volatility': [...]
    })
}
```

**What it classifies**:
- Regime: Range/Trending Up/Trending Down/Crisis
- Strategy suggestion: Mean Reversion/Momentum Long/Defensive/Risk Off
- Transition probabilities

**Example**:
```python
agent = RegimeDetectionAgent()
response = agent.analyze({"indicators": indicators_df})

print(f"Regime: {response.structured_data['regime']}")
print(f"Strategy: {response.structured_data['strategy']}")
print(f"Transitions: {response.structured_data['transition_probabilities']}")
```

**ML Integration**:
```python
# To use trained models:
from Regime_Classificaiton import Regime_Classifier
# Replace heuristic logic in predict_regime() with:
# regime = self.regime_model.predict_regime(current_sequence)
```

**Data Requirements**:
- Minimum 60 days of indicator data
- DataFrame with at least 'returns' column

---

#### Synthesis Agent

##### `SignalAggregatorAgent`

**What it does**: Combines all agent outputs and generates final decision using LLM.

**Required Context**:
```python
context = {
    "agent_outputs": [response1, response2, ...]  # List of AgentResponse objects
}
```

**What it does**:
1. **Conflict Resolution**: Detects contradictions (e.g., "Bullish sentiment but high risk")
2. **LLM Synthesis**: Sends all signals to Gemini/Groq for reasoning
3. **Dynamic Weighting**: Uses `performance_weight` from each agent

**Example**:
```python
# Collect responses from all agents
responses = [
    market_agent.analyze({"symbol": "NIFTY"}),
    sentiment_agent.analyze({}),
    volatility_agent.analyze({"returns": returns}),
    # ... etc
]

# Aggregate
aggregator = SignalAggregatorAgent()
final_decision = aggregator.analyze({"agent_outputs": responses})

print(f"Final Recommendation: {final_decision.recommendation}")
print(f"LLM Reasoning: {final_decision.structured_data['llm_synthesis']}")
```

**Configuration**:
- Set `GEMINI_API_KEY` in `.env`
- Optional: Set `GROQ_API_KEY` for lightweight tasks
- Set `USE_LIGHTWEIGHT_LLM=true` to use Groq for simple aggregations

---

### 3. `communication_protocol.py` - Orchestration

**Purpose**: Coordinates agent execution and message passing.

#### Key Classes

##### `AgentOrchestrator`

**What it does**: Manages the execution workflow of multiple agents.

**Usage Pattern 1: Sequential Execution**
```python
from communication_protocol import AgentOrchestrator
from agents import *

orchestrator = AgentOrchestrator()

# Add agents
orchestrator.add_agent(MarketDataAgent())
orchestrator.add_agent(VolatilityAgent())

# Define execution order
orchestrator.set_execution_order([
    "MarketDataAgent",
    "VolatilityAgent"
])

# Execute pipeline
context = {"symbol": "NIFTY", "returns": returns_series}
results = orchestrator.execute_pipeline(context)

# Access individual results
market_result = results["MarketDataAgent"]
vol_result = results["VolatilityAgent"]
```

**Usage Pattern 2: Parallel with Aggregation**
```python
orchestrator = AgentOrchestrator()

# Add all agents
orchestrator.add_agent(MarketDataAgent())
orchestrator.add_agent(RiskMonitoringAgent())
orchestrator.add_agent(SentimentAgent())
orchestrator.add_agent(VolatilityAgent())
orchestrator.add_agent(SignalAggregatorAgent())

# Execute all in parallel, then aggregate
context = {
    "symbol": "NIFTY",
    "returns": returns_series,
    "positions": positions_df,
    "current_drawdown": 0.03
}

final_decision = orchestrator.execute_with_aggregation(context)
print(final_decision.summary)
```

##### `MessageRouter`

**What it does**: Routes messages between agents.

**When to use**: Advanced use cases where agents need to communicate mid-execution.

**Example**:
```python
from communication_protocol import MessageRouter, AgentMessage, MessageType

router = MessageRouter()
router.register_agent(agent1)
router.register_agent(agent2)

# Send message
msg = AgentMessage(
    sender="agent1",
    receiver="agent2",
    message_type=MessageType.REQUEST,
    payload={"symbol": "NIFTY"}
)

response = router.route_message(msg)
```

---

## Quick Start Guide

### Step 1: Install Dependencies

```bash
cd CF-AI-SDE
pip install -r requirements.txt
```

### Step 2: Configure Environment

Edit `AI_Agents/.env`:

```bash
# Required
MONGODB_URI=mongodb://localhost:27017/
NEWS_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
ECONOMIC_CALENDAR_API_KEY=your_key_here

# Optional
GROQ_API_KEY=your_key_here
USE_LIGHTWEIGHT_LLM=false
```

### Step 3: Setup MongoDB

```javascript
// In MongoDB shell
use trading_db

// Create collection
db.createCollection("market_data")

// Create index
db.market_data.createIndex({ "symbol": 1, "timestamp": -1 })

// Insert sample data
db.market_data.insertOne({
    "symbol": "NIFTY",
    "timestamp": new Date(),
    "open": 18500,
    "high": 18550,
    "low": 18480,
    "close": 18520,
    "volume": 1500000
})
```

### Step 4: Run Examples

```bash
cd AI_Agents
python example_usage.py
```

---

## Usage Examples

### Example 1: Single Agent Analysis

```python
from agents import MarketDataAgent

# Create agent
agent = MarketDataAgent()

# Analyze
response = agent.analyze({"symbol": "NIFTY"})

# Use results
print(response.summary)
if response.recommendation == "ALERT":
    anomalies = response.structured_data['anomalies']
    if anomalies['gap_open']:
        print(f"Gap: {anomalies['details']['gap_percentage']}%")
```

### Example 2: Multiple Agents with Manual Aggregation

```python
from agents import *
import pandas as pd

# Prepare data
returns = pd.Series([...])  # Your returns data
indicators = pd.DataFrame({...})  # Your indicators

# Run agents
sentiment = SentimentAgent().analyze({})
volatility = VolatilityAgent().analyze({"returns": returns})
regime = RegimeDetectionAgent().analyze({"indicators": indicators})

# Manual decision logic
if (sentiment.recommendation == "BULLISH" and 
    volatility.structured_data['regime'] == "stable" and
    regime.structured_data['regime'] != "Crisis"):
    decision = "BUY"
else:
    decision = "HOLD"
```

### Example 3: Full Orchestrated System

```python
from communication_protocol import AgentOrchestrator
from agents import *
import pandas as pd

# Create orchestrator
orchestrator = AgentOrchestrator()

# Add all agents
orchestrator.add_agent(MarketDataAgent())
orchestrator.add_agent(RiskMonitoringAgent())
orchestrator.add_agent(MacroAgent())
orchestrator.add_agent(SentimentAgent())
orchestrator.add_agent(VolatilityAgent())
orchestrator.add_agent(RegimeDetectionAgent())
orchestrator.add_agent(SignalAggregatorAgent())

# Prepare context
context = {
    "symbol": "NIFTY",
    "returns": pd.Series([...]),
    "indicators": pd.DataFrame({...}),
    "positions": pd.DataFrame({...}),
    "current_drawdown": 0.03
}

# Execute
final_decision = orchestrator.execute_with_aggregation(context)

# Use results
print(f"Decision: {final_decision.recommendation}")
print(f"Confidence: {final_decision.confidence_score:.2%}")
print(f"Reasoning: {final_decision.structured_data['llm_synthesis']}")
```

### Example 4: Using Memory/Learning

```python
from agents import VolatilityAgent

agent = VolatilityAgent()

# Day 1: Make prediction
prediction = agent.analyze({"returns": today_returns})
print(f"Predicted vol: {prediction.structured_data['forecasted_volatility']}")

# Day 2: Record actual outcome
agent.store_memory(
    input_context={"returns": today_returns},
    prediction=prediction,
    actual_outcome={"actual_vol": 0.015},
    accuracy_score=0.90  # 90% accurate
)

# After 10 predictions, check performance
stats = agent.get_recent_performance(window=10)
print(f"Agent accuracy: {stats['avg_accuracy']:.2%}")
print(f"Agent weight: {stats['current_weight']:.3f}")
```

---

## Agent Descriptions

| Agent                     | Type          | Data Source           | Output                | Use Case                         |
| ------------------------- | ------------- | --------------------- | --------------------- | -------------------------------- |
| **MarketDataAgent**       | Deterministic | MongoDB (OHLCV)       | Anomaly flags         | Detect unusual market moves      |
| **RiskMonitoringAgent**   | Deterministic | Portfolio positions   | VaR & limit breaches  | Risk management                  |
| **MacroAgent**            | Deterministic | Economic Calendar API | Upcoming events       | Event-driven trading             |
| **SentimentAgent**        | Probabilistic | News API + FinBERT    | Sentiment score       | Market psychology                |
| **VolatilityAgent**       | Probabilistic | GARCH/LSTM models     | Volatility forecast   | Options pricing, position sizing |
| **RegimeDetectionAgent**  | Probabilistic | HMM/Clustering        | Regime classification | Strategy selection               |
| **SignalAggregatorAgent** | Synthesis     | All agent outputs     | Final decision        | Trading decision                 |

---

## Integration Guide

### Integrating with Your Data Pipeline

```python
# Your existing data pipeline
def get_market_data(symbol):
    # Your code to fetch data
    return df

def prepare_features(df):
    # Your feature engineering
    return features_df, returns_series

# Integration with AI Agents
from communication_protocol import AgentOrchestrator
from agents import *

# Setup (do once)
orchestrator = AgentOrchestrator()
orchestrator.add_agent(MarketDataAgent())
orchestrator.add_agent(VolatilityAgent())
orchestrator.add_agent(RegimeDetectionAgent())
orchestrator.add_agent(SignalAggregatorAgent())

# In your trading loop
symbol = "NIFTY"
df = get_market_data(symbol)
features, returns = prepare_features(df)

# Get AI decision
context = {
    "symbol": symbol,
    "returns": returns,
    "indicators": features
}
decision = orchestrator.execute_with_aggregation(context)

# Use decision in your strategy
if decision.recommendation == "BUY" and decision.confidence_score > 0.75:
    place_order(symbol, "BUY")
```

### Integrating ML Models

**Volatility Models**:
```python
# Already integrated in VolatilityAgent
# The agent imports from ML_Models/Volatility_Forecasting.py
# Just ensure your models are trained and the file exists
```

**Regime Models**:
```python
# To enable ML-based regime detection:
# 1. Train your Regime_Classifier model
# 2. In RegimeDetectionAgent.__init__(), add:

from Regime_Classificaiton import Regime_Classifier

self.regime_model = Regime_Classifier(
    features=historical_features,
    regime_labels=historical_labels
)

# 3. In predict_regime(), replace heuristic with:
regime = self.regime_model.predict_regime(current_sequence)
```

---

## Advanced Features

### 1. Custom Agents

Create your own agent by inheriting from `BaseAgent`:

```python
from base_agent import BaseAgent, AgentResponse
from typing import Dict, Any

class MyCustomAgent(BaseAgent):
    def __init__(self, name: str = "MyCustomAgent"):
        super().__init__(name)
        # Your initialization
    
    def analyze(self, context: Dict[str, Any]) -> AgentResponse:
        # Your analysis logic
        result = self.my_analysis(context)
        
        return AgentResponse(
            agent_name=self.name,
            summary=f"Analysis result: {result}",
            confidence_score=0.8,
            structured_data={"result": result},
            recommendation="BUY" if result > 0 else "SELL"
        )
```

### 2. Dynamic Weight Adjustment

The system automatically adjusts agent weights based on accuracy:

```python
# Agents start with weight = 1.0
# After each prediction with known outcome:
agent.store_memory(input, prediction, outcome, accuracy=0.9)

# Weight updates using formula:
# new_weight = (1 - α) * old_weight + α * avg_recent_accuracy
# Where α = 0.3 (learning rate)

# Check current weight
print(agent.performance_weight)  # e.g., 1.12 (12% boost)
```

### 3. Conflict Resolution

Built-in logic in `SignalAggregatorAgent.resolve_conflicts()`:

```python
# Example conflicts:
# - Sentiment=BULLISH but Risk=HIGH → Resolution: HOLD
# - Volatility=EXPANDING but Regime=RANGE → Resolution: REDUCE_SIZE

# Customize in agents.py:
def resolve_conflicts(self, agent_outputs):
    conflicts = []
    
    # Add your custom rules
    if sentiment == "BULLISH" and macro_risk == "HIGH":
        conflicts.append({
            "type": "sentiment_vs_macro",
            "resolution": "WAIT"
        })
    
    return {"conflicts": conflicts}
```

---

## Troubleshooting

### Common Issues

**Issue**: "MongoDB connection refused"
```bash
# Solution: Ensure MongoDB is running
sudo service mongodb start  # Linux
net start MongoDB           # Windows
```

**Issue**: "FinBERT model not found"
```bash
# Solution: Install transformers and download model
pip install transformers torch
# Model auto-downloads on first use
```

**Issue**: "Gemini API rate limit"
```bash
# Solution: Enable lightweight LLM fallback
# In .env:
USE_LIGHTWEIGHT_LLM=true
GROQ_API_KEY=your_groq_key
```

**Issue**: "Agent returns low confidence"
```bash
# Solution: Check data quality
# - Ensure sufficient historical data (>60 days)
# - Verify data format (pandas Series/DataFrame)
# - Check for NaN values
```

---

## Next Steps

1. **Populate MongoDB**: Add historical OHLCV data (see CONFIGURATION.md)
2. **Train Models**: Ensure ML models in `ML_Models/` are trained
3. **Test Agents**: Run `example_usage.py` to verify setup
4. **Monitor Performance**: Use `get_recent_performance()` to track accuracy
5. **Customize**: Add your own agents or modify conflict resolution logic

---

## Support & Contributing

- **Configuration Help**: See `CONFIGURATION.md`
- **Setup Issues**: Check troubleshooting section above
- **Custom Agents**: Follow the "Custom Agents" pattern in Advanced Features

For questions about specific components, refer to the inline documentation in each module.
