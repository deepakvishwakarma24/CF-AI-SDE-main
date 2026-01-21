# AI Agents Configuration Guide

This document lists all placeholders and setup requirements for the multi-agent trading system.

## Table of Contents
1. [Required API Keys](#required-api-keys)
2. [MongoDB Setup](#mongodb-setup)
3. [ML Model Integration](#ml-model-integration)
4. [Environment Variables](#environment-variables)
5. [Step-by-Step Setup](#step-by-step-setup)

---

## Required API Keys

### 1. News API
- **Purpose**: Fetch financial news headlines for sentiment analysis
- **Get it from**: [https://newsapi.org/](https://newsapi.org/)
- **Free tier**: 100 requests/day
- **Environment variable**: `NEWS_API_KEY`

### 2. Economic Calendar API
- **Purpose**: Fetch upcoming high-impact macroeconomic events
- **Recommended provider**: [TradingEconomics](https://tradingeconomics.com/)
- **Alternative**: [Forex Factory API](https://www.forexfactory.com/)
- **Environment variables**: 
  - `ECONOMIC_CALENDAR_API_KEY`
  - `ECONOMIC_CALENDAR_API_URL`

### 3. Google Gemini API
- **Purpose**: Complex reasoning for signal aggregation
- **Get it from**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Free tier**: 60 requests/minute
- **Environment variable**: `GEMINI_API_KEY`

### 4. Groq API (Optional)
- **Purpose**: Fast lightweight LLM for simple tasks
- **Get it from**: [Groq Console](https://console.groq.com/)
- **Free tier**: Available
- **Environment variable**: `GROQ_API_KEY`

### 5. Huggingface Token (Optional)
- **Purpose**: Access to FinBERT and other models
- **Get it from**: [Huggingface Settings](https://huggingface.co/settings/tokens)
- **Environment variable**: `HUGGINGFACE_TOKEN`

---

## MongoDB Setup

### Database Schema

The `MarketDataAgent` expects the following document structure:

```json
{
  "symbol": "NIFTY",
  "timestamp": "2025-01-21T10:00:00",
  "open": 18500.0,
  "high": 18550.0,
  "low": 18480.0,
  "close": 18520.0,
  "volume": 1500000
}
```

### Collections Required
- **Collection name**: `market_data` (configurable via `MONGODB_COLLECTION`)
- **Database name**: `trading_db` (configurable via `MONGODB_DATABASE`)

### Indexes (Recommended)
```javascript
db.market_data.createIndex({ "symbol": 1, "timestamp": -1 })
```

### Connection String Format
```
mongodb://username:password@host:port/database
```

For local MongoDB:
```
mongodb://localhost:27017/
```

---

## ML Model Integration

### 1. Volatility Models

**File**: `CF-AI-SDE/ML_Models/Volatility_Forecasting.py`

**Required by**: `VolatilityAgent`

**Integration**: The agent imports and uses:
```python
from Volatility_Forecasting import Volatility_Models, LSTM_Volatility_Model
```

**Data flow**:
- Agent passes `pd.Series` of log returns to `Volatility_Models`
- Model returns dictionary: `{"garch_volatility": float, "egarch_volatility": float}`

### 2. Regime Classification Models

**File**: `CF-AI-SDE/ML_Models/Regime_Classificaiton.py`

**Required by**: `RegimeDetectionAgent`

**Integration**: Currently using heuristic logic. To enable ML model:
```python
from Regime_Classificaiton import Regime_Classifier

# In __init__:
self.regime_model = Regime_Classifier(features=df_X, regime_labels=df['Regime'])

# In predict_regime:
regime = self.regime_model.predict_regime(current_sequence)
```

**Data flow**:
- Agent passes `pd.DataFrame` of technical indicators
- Model returns regime label (e.g., "Range", "Trending Up")

### 3. FinBERT Sentiment Model

**Model**: `ProsusAI/finbert` from Huggingface

**Required by**: `SentimentAgent`

**Auto-downloaded**: Model is downloaded automatically on first use

**Requirements**:
```bash
pip install transformers torch
```

---

## Environment Variables

### Complete List

| Variable                    | Required | Default                                     | Description                         |
| --------------------------- | -------- | ------------------------------------------- | ----------------------------------- |
| `MONGODB_URI`               | Yes      | `mongodb://localhost:27017/`                | MongoDB connection string           |
| `MONGODB_DATABASE`          | Yes      | `trading_db`                                | Database name                       |
| `MONGODB_COLLECTION`        | Yes      | `market_data`                               | Collection name for OHLCV data      |
| `NEWS_API_KEY`              | Yes      | -                                           | News API authentication key         |
| `NEWS_API_URL`              | No       | `https://newsapi.org/v2/everything`         | News API endpoint                   |
| `ECONOMIC_CALENDAR_API_KEY` | Yes      | -                                           | Economic calendar API key           |
| `ECONOMIC_CALENDAR_API_URL` | No       | `https://api.tradingeconomics.com/calendar` | Calendar API endpoint               |
| `GEMINI_API_KEY`            | Yes      | -                                           | Google Gemini API key               |
| `GROQ_API_KEY`              | No       | -                                           | Groq API key (optional)             |
| `HUGGINGFACE_TOKEN`         | No       | -                                           | Huggingface access token (optional) |
| `USE_LIGHTWEIGHT_LLM`       | No       | `false`                                     | Use Groq for simple tasks           |
| `VAR_LIMIT`                 | No       | `0.05`                                      | Maximum VaR threshold (5%)          |
| `DRAWDOWN_LIMIT`            | No       | `0.10`                                      | Maximum drawdown threshold (10%)    |

---

## Step-by-Step Setup

### Step 1: Install Dependencies

```bash
cd CF-AI-SDE/AI_Agents
pip install -r requirements.txt
```

**Required packages**:
```
pydantic
pymongo
python-dotenv
requests
transformers
torch
numpy
pandas
google-generativeai
groq
```

### Step 2: Configure Environment Variables

1. Copy the `.env` template file
2. Fill in your API keys:
   - News API key
   - Economic Calendar API key
   - Gemini API key
   - (Optional) Groq API key

### Step 3: Setup MongoDB

1. Install MongoDB locally or use MongoDB Atlas (cloud)
2. Create database: `trading_db`
3. Create collection: `market_data`
4. Populate with historical OHLCV data
5. Update `MONGODB_URI` in `.env`

### Step 4: Test Individual Agents

```python
from agents import MarketDataAgent

# Test market data fetching
agent = MarketDataAgent()
response = agent.analyze({"symbol": "NIFTY"})
print(response.summary)
```

### Step 5: Run Full Pipeline

```python
from communication_protocol import AgentOrchestrator
from agents import *

orchestrator = AgentOrchestrator()

# Add all agents
orchestrator.add_agent(MarketDataAgent())
orchestrator.add_agent(RiskMonitoringAgent())
orchestrator.add_agent(MacroAgent())
orchestrator.add_agent(SentimentAgent())
orchestrator.add_agent(VolatilityAgent())
orchestrator.add_agent(RegimeDetectionAgent())
orchestrator.add_agent(SignalAggregatorAgent())

# Execute
context = {"symbol": "NIFTY", "positions": [], "indicators": pd.DataFrame()}
final_decision = orchestrator.execute_with_aggregation(context)
print(final_decision.summary)
```

---

## Troubleshooting

### Issue: "No module named 'transformers'"
**Solution**: Install Huggingface transformers:
```bash
pip install transformers torch
```

### Issue: "FinBERT model download failed"
**Solution**: Set Huggingface token in `.env` and retry:
```bash
export HUGGINGFACE_TOKEN=your_token_here
```

### Issue: "MongoDB connection refused"
**Solution**: Ensure MongoDB is running:
```bash
# On Linux/Mac
sudo service mongodb start

# On Windows
net start MongoDB
```

### Issue: "Gemini API rate limit exceeded"
**Solution**: Enable lightweight LLM fallback:
```bash
USE_LIGHTWEIGHT_LLM=true
```

---

## Next Steps

1. **Populate MongoDB**: Add historical market data to the database
2. **Train Models**: Ensure ML models in `ML_Models` directory are trained with recent data
3. **Test Agents**: Run each agent individually to verify functionality
4. **Monitor Performance**: Use `BaseAgent.get_recent_performance()` to track agent accuracy
5. **Adjust Weights**: The `performance_weight` will auto-adjust based on prediction accuracy

---

## Support

For issues with specific components:
- **MongoDB**: [MongoDB Documentation](https://docs.mongodb.com/)
- **FinBERT**: [Huggingface Model Page](https://huggingface.co/ProsusAI/finbert)
- **Gemini API**: [Google AI Documentation](https://ai.google.dev/docs)
- **Groq API**: [Groq Documentation](https://console.groq.com/docs)
