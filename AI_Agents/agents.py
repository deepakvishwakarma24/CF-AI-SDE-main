"""
Concrete Trading Agents

This module contains all 7 specialized agents that inherit from BaseAgent.
Each agent encapsulates specific domain logic for financial analysis.
"""

import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import requests

# Import ML models from the ML_Models directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ML_Models'))
from Volatility_Forecasting import Volatility_Models, LSTM_Volatility_Model
from Regime_Classificaiton import Regime_Classifier

# Import base classes
from base_agent import BaseAgent, AgentResponse

# Load environment variables
load_dotenv()


# ==================== DETERMINISTIC AGENTS ====================

class MarketDataAgent(BaseAgent):
    """
    Fetches and analyzes raw market data from MongoDB.
    Detects anomalies like gap opens, volume spikes, and volatility outliers.
    """
    
    def __init__(self, name: str = "MarketDataAgent"):
        super().__init__(name)
        
        # MongoDB connection
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.client = MongoClient(mongo_uri)
        self.db = self.client[os.getenv("MONGODB_DATABASE", "trading_db")]
        self.collection = self.db[os.getenv("MONGODB_COLLECTION", "market_data")]
    
    def _get_ohlcv(self, symbol: str, period: int = 100) -> pd.DataFrame:
        """
        Private helper to fetch OHLCV data from MongoDB.
        
        Args:
            symbol: Stock symbol (e.g., "NIFTY")
            period: Number of days to fetch
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Query MongoDB for recent data
        query = {"symbol": symbol}
        cursor = self.collection.find(query).sort("timestamp", -1).limit(period)
        
        # Convert to DataFrame
        data = list(cursor)
        if len(data) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    def detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detects statistical anomalies in market data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            Dictionary of detected anomalies
        """
        if len(data) < 20:
            return {"error": "Insufficient data for anomaly detection"}
        
        anomalies = {
            "gap_open": False,
            "volume_spike": False,
            "volatility_outlier": False,
            "details": {}
        }
        
        # 1. Gap Open Detection (>2% gap from previous close)
        data['prev_close'] = data['close'].shift(1)
        data['gap_pct'] = ((data['open'] - data['prev_close']) / data['prev_close']) * 100
        latest_gap = data['gap_pct'].iloc[-1]
        
        if abs(latest_gap) > 2.0:
            anomalies["gap_open"] = True
            anomalies["details"]["gap_percentage"] = round(latest_gap, 2)
        
        # 2. Volume Spike Detection (>2x 20-day average)
        data['avg_volume'] = data['volume'].rolling(20).mean()
        latest_volume = data['volume'].iloc[-1]
        avg_volume = data['avg_volume'].iloc[-1]
        
        if latest_volume > 2 * avg_volume:
            anomalies["volume_spike"] = True
            anomalies["details"]["volume_ratio"] = round(latest_volume / avg_volume, 2)
        
        # 3. Volatility Outlier (>2 std devs from mean)
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        
        mean_vol = data['volatility'].mean()
        std_vol = data['volatility'].std()
        latest_vol = data['volatility'].iloc[-1]
        
        if latest_vol > mean_vol + 2 * std_vol:
            anomalies["volatility_outlier"] = True
            anomalies["details"]["volatility_zscore"] = round((latest_vol - mean_vol) / std_vol, 2)
        
        return anomalies
    
    def analyze(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Analyzes market data and returns structured response.
        
        Args:
            context: Must contain 'symbol' key
            
        Returns:
            AgentResponse with anomaly detection results
        """
        symbol = context.get("symbol", "NIFTY")
        
        # Fetch data
        data = self._get_ohlcv(symbol, period=100)
        
        if len(data) == 0:
            return AgentResponse(
                agent_name=self.name,
                summary=f"No data found for {symbol}",
                confidence_score=0.0,
                structured_data={"error": "No data available"},
                recommendation="HOLD"
            )
        
        # Detect anomalies
        anomalies = self.detect_anomalies(data)
        
        # Generate summary
        alert_flags = []
        if anomalies.get("gap_open"):
            alert_flags.append(f"Gap open: {anomalies['details']['gap_percentage']}%")
        if anomalies.get("volume_spike"):
            alert_flags.append(f"Volume spike: {anomalies['details']['volume_ratio']}x average")
        if anomalies.get("volatility_outlier"):
            alert_flags.append(f"Volatility outlier: {anomalies['details']['volatility_zscore']} std devs")
        
        if len(alert_flags) > 0:
            summary = f"Market anomalies detected for {symbol}: {', '.join(alert_flags)}"
            recommendation = "ALERT"
            confidence = 0.9
        else:
            summary = f"Normal market conditions for {symbol}. No anomalies detected."
            recommendation = "NORMAL"
            confidence = 0.8
        
        return AgentResponse(
            agent_name=self.name,
            summary=summary,
            confidence_score=confidence,
            structured_data={
                "symbol": symbol,
                "latest_close": float(data['close'].iloc[-1]),
                "anomalies": anomalies
            },
            recommendation=recommendation
        )


class RiskMonitoringAgent(BaseAgent):
    """
    Calculates portfolio risk metrics (VaR, Drawdown) and checks against limits.
    """
    
    def __init__(self, name: str = "RiskMonitoringAgent"):
        super().__init__(name)
        
        # Risk limits (can be configured)
        self.var_limit = float(os.getenv("VAR_LIMIT", "0.05"))  # 5% max VaR
        self.drawdown_limit = float(os.getenv("DRAWDOWN_LIMIT", "0.10"))  # 10% max drawdown
    
    def calculate_var(
        self, 
        positions: pd.DataFrame, 
        confidence: float = 0.95
    ) -> float:
        """
        Calculates Value at Risk using historical simulation.
        
        Args:
            positions: DataFrame with columns ['symbol', 'quantity', 'returns']
            confidence: Confidence level (default 95%)
            
        Returns:
            VaR as a positive percentage (e.g., 0.05 = 5% VaR)
        """
        if len(positions) == 0:
            return 0.0
        
        # Calculate portfolio returns
        positions['position_return'] = positions['quantity'] * positions['returns']
        portfolio_return = positions.groupby('timestamp')['position_return'].sum()
        
        # Calculate VaR at specified confidence level
        var_percentile = 1 - confidence
        var = abs(np.percentile(portfolio_return, var_percentile * 100))
        
        return var
    
    def check_limits(self, current_var: float, current_drawdown: float) -> Dict[str, Any]:
        """
        Checks if risk metrics breach configured limits.
        
        Args:
            current_var: Current VaR value
            current_drawdown: Current drawdown value
            
        Returns:
            Dictionary with breach status and details
        """
        breaches = {
            "var_breach": current_var > self.var_limit,
            "drawdown_breach": current_drawdown > self.drawdown_limit,
            "details": {
                "var": {"current": current_var, "limit": self.var_limit},
                "drawdown": {"current": current_drawdown, "limit": self.drawdown_limit}
            }
        }
        
        return breaches
    
    def analyze(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Analyzes portfolio risk and returns alerts if limits are breached.
        
        Args:
            context: Must contain 'positions' (DataFrame) and 'current_drawdown' (float)
            
        Returns:
            AgentResponse with risk assessment
        """
        positions = context.get("positions", pd.DataFrame())
        current_drawdown = context.get("current_drawdown", 0.0)
        
        # Calculate VaR
        current_var = self.calculate_var(positions)
        
        # Check limits
        breaches = self.check_limits(current_var, current_drawdown)
        
        # Generate response
        if breaches["var_breach"] or breaches["drawdown_breach"]:
            alert_msgs = []
            if breaches["var_breach"]:
                alert_msgs.append(f"VaR breach: {current_var:.2%} > {self.var_limit:.2%}")
            if breaches["drawdown_breach"]:
                alert_msgs.append(f"Drawdown breach: {current_drawdown:.2%} > {self.drawdown_limit:.2%}")
            
            summary = f"RISK ALERT: {', '.join(alert_msgs)}"
            recommendation = "REDUCE_EXPOSURE"
            confidence = 0.95
        else:
            summary = f"Risk metrics within limits. VaR: {current_var:.2%}, Drawdown: {current_drawdown:.2%}"
            recommendation = "NORMAL"
            confidence = 0.85
        
        return AgentResponse(
            agent_name=self.name,
            summary=summary,
            confidence_score=confidence,
            structured_data={
                "var": current_var,
                "drawdown": current_drawdown,
                "breaches": breaches
            },
            recommendation=recommendation
        )


class MacroAgent(BaseAgent):
    """
    Fetches upcoming macroeconomic events and analyzes historical impact.
    """
    
    def __init__(self, name: str = "MacroAgent"):
        super().__init__(name)
        
        # API configuration
        self.calendar_api_key = os.getenv("ECONOMIC_CALENDAR_API_KEY")
        self.calendar_api_url = os.getenv(
            "ECONOMIC_CALENDAR_API_URL", 
            "https://api.tradingeconomics.com/calendar"
        )
    
    def get_upcoming_events(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """
        Fetches high-impact economic events from API.
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of event dictionaries
        """
        if not self.calendar_api_key:
            return []
        
        try:
            params = {
                "c": self.calendar_api_key,
                "country": "india",
                "importance": 3  # High impact only
            }
            
            response = requests.get(self.calendar_api_url, params=params, timeout=10)
            response.raise_for_status()
            
            events = response.json()
            
            # Filter for upcoming events within the window
            cutoff = datetime.now() + timedelta(days=days_ahead)
            upcoming = [
                e for e in events 
                if datetime.fromisoformat(e['Date']) <= cutoff
            ]
            
            return upcoming
        
        except Exception as e:
            print(f"Error fetching economic calendar: {e}")
            return []
    
    def retrieve_historical_impact(self, event_type: str) -> Dict[str, Any]:
        """
        Looks up historical market reactions to this event type.
        
        Args:
            event_type: Type of event (e.g., "Fed Meeting", "CPI")
            
        Returns:
            Dictionary with historical statistics
        """
        # This would query a historical database
        # For now, return mock data
        historical_impacts = {
            "Fed Meeting": {"avg_move": 1.5, "direction": "volatile"},
            "CPI": {"avg_move": 0.8, "direction": "up_if_high"},
            "GDP": {"avg_move": 0.5, "direction": "up_if_strong"}
        }
        
        return historical_impacts.get(event_type, {"avg_move": 0.0, "direction": "unknown"})
    
    def analyze(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Analyzes upcoming macro events and their potential impact.
        
        Args:
            context: Optional filters for event analysis
            
        Returns:
            AgentResponse with macro outlook
        """
        events = self.get_upcoming_events()
        
        if len(events) == 0:
            return AgentResponse(
                agent_name=self.name,
                summary="No high-impact macro events scheduled in the next 7 days.",
                confidence_score=0.7,
                structured_data={"events": []},
                recommendation="NORMAL"
            )
        
        # Analyze events
        high_impact_count = len(events)
        event_summaries = [f"{e.get('Event', 'Unknown')} on {e.get('Date', 'TBD')}" for e in events[:3]]
        
        summary = f"{high_impact_count} high-impact macro event(s) upcoming: {', '.join(event_summaries)}"
        
        if high_impact_count >= 3:
            recommendation = "CAUTION"
            confidence = 0.8
        else:
            recommendation = "MONITOR"
            confidence = 0.75
        
        return AgentResponse(
            agent_name=self.name,
            summary=summary,
            confidence_score=confidence,
            structured_data={"events": events, "event_count": high_impact_count},
            recommendation=recommendation
        )


# ==================== PROBABILISTIC AGENTS ====================

class SentimentAgent(BaseAgent):
    """
    Analyzes news sentiment using FinBERT transformer model.
    """
    
    def __init__(self, name: str = "SentimentAgent"):
        super().__init__(name)
        
        # Initialize FinBERT pipeline
        try:
            from transformers import pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model="ProsusAI/finbert",
                device=-1  # CPU
            )
        except Exception as e:
            print(f"Warning: FinBERT not loaded: {e}")
            self.sentiment_pipeline = None
        
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.news_api_url = os.getenv("NEWS_API_URL", "https://newsapi.org/v2/everything")
    
    def _fetch_news(self, time_window: int = 24) -> List[str]:
        """
        Fetches recent news headlines from News API.
        
        Args:
            time_window: Hours to look back
            
        Returns:
            List of headline strings
        """
        if not self.news_api_key:
            return []
        
        try:
            from_time = datetime.now() - timedelta(hours=time_window)
            
            params = {
                "apiKey": self.news_api_key,
                "q": "NIFTY OR india stock market",
                "language": "en",
                "from": from_time.isoformat(),
                "sortBy": "publishedAt"
            }
            
            response = requests.get(self.news_api_url, params=params, timeout=10)
            response.raise_for_status()
            
            articles = response.json().get("articles", [])
            headlines = [a["title"] for a in articles]
            
            return headlines[:50]  # Limit to 50 headlines
        
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
    
    def predict_sentiment(self, text_batch: List[str]) -> List[Dict[str, Any]]:
        """
        Runs FinBERT inference on a batch of text.
        
        Args:
            text_batch: List of text strings to analyze
            
        Returns:
            List of sentiment dictionaries
        """
        if not self.sentiment_pipeline or len(text_batch) == 0:
            return []
        
        try:
            # Truncate long texts to 512 tokens
            truncated_batch = [text[:512] for text in text_batch]
            results = self.sentiment_pipeline(truncated_batch)
            return results
        except Exception as e:
            print(f"Error in sentiment prediction: {e}")
            return []
    
    def calculate_momentum(self, current_score: float, past_scores: List[float]) -> str:
        """
        Determines if sentiment is improving or deteriorating.
        
        Args:
            current_score: Latest aggregated sentiment score
            past_scores: Historical sentiment scores
            
        Returns:
            "improving", "stable", or "deteriorating"
        """
        if len(past_scores) < 2:
            return "stable"
        
        avg_past = np.mean(past_scores)
        
        if current_score > avg_past + 0.1:
            return "improving"
        elif current_score < avg_past - 0.1:
            return "deteriorating"
        else:
            return "stable"
    
    def analyze(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Analyzes news sentiment and returns aggregate score.
        
        Args:
            context: Optional parameters for sentiment analysis
            
        Returns:
            AgentResponse with sentiment analysis
        """
        # Fetch news
        headlines = self._fetch_news()
        
        if len(headlines) == 0:
            return AgentResponse(
                agent_name=self.name,
                summary="No news data available for sentiment analysis.",
                confidence_score=0.5,
                structured_data={"sentiment_score": 0.0},
                recommendation="HOLD"
            )
        
        # Run sentiment analysis
        sentiments = self.predict_sentiment(headlines)
        
        # Aggregate scores (FinBERT outputs: positive, negative, neutral)
        sentiment_values = []
        for s in sentiments:
            label = s.get("label", "neutral")
            confidence = s.get("score", 0.0)
            
            if label == "positive":
                sentiment_values.append(confidence)
            elif label == "negative":
                sentiment_values.append(-confidence)
            else:
                sentiment_values.append(0.0)
        
        avg_sentiment = np.mean(sentiment_values) if len(sentiment_values) > 0 else 0.0
        
        # Determine recommendation
        if avg_sentiment > 0.2:
            summary = "Positive market sentiment detected from news analysis."
            recommendation = "BULLISH"
            confidence = 0.75
        elif avg_sentiment < -0.2:
            summary = "Negative market sentiment detected from news analysis."
            recommendation = "BEARISH"
            confidence = 0.75
        else:
            summary = "Neutral market sentiment from news analysis."
            recommendation = "NEUTRAL"
            confidence = 0.65
        
        return AgentResponse(
            agent_name=self.name,
            summary=summary,
            confidence_score=confidence,
            structured_data={
                "sentiment_score": avg_sentiment,
                "news_count": len(headlines),
                "positive_count": sum(1 for v in sentiment_values if v > 0),
                "negative_count": sum(1 for v in sentiment_values if v < 0)
            },
            recommendation=recommendation
        )


class VolatilityAgent(BaseAgent):
    """
    Forecasts volatility using GARCH/LSTM models from ML_Models directory.
    """
    
    def __init__(self, name: str = "VolatilityAgent"):
        super().__init__(name)
        self.volatility_model = None
        self.lstm_model = None
    
    def forecast_volatility(self, returns: pd.Series) -> Dict[str, float]:
        """
        Runs GARCH/EGARCH volatility forecast.
        
        Args:
            returns: Series of log returns
            
        Returns:
            Dictionary with GARCH and EGARCH forecasts
        """
        try:
            # Initialize GARCH models
            vol_models = Volatility_Models(returns=returns)
            forecast = vol_models.predict_volatility(horizon=1)
            return forecast
        except Exception as e:
            print(f"Error in volatility forecast: {e}")
            return {"garch_volatility": 0.0, "egarch_volatility": 0.0}
    
    def detect_regime_change(self, current_vol: float, forecasted_vol: float) -> str:
        """
        Detects if volatility is expanding or compressing.
        
        Args:
            current_vol: Current realized volatility
            forecasted_vol: Forecasted volatility
            
        Returns:
            "expanding", "compressing", or "stable"
        """
        ratio = forecasted_vol / current_vol if current_vol > 0 else 1.0
        
        if ratio > 1.2:
            return "expanding"
        elif ratio < 0.8:
            return "compressing"
        else:
            return "stable"
    
    def compare_implied_realized(self, option_chain: Optional[Dict] = None) -> Dict[str, float]:
        """
        Compares implied volatility (from options) vs realized volatility.
        
        Args:
            option_chain: Optional dictionary with IV data
            
        Returns:
            Dictionary with IV/RV comparison
        """
        # Placeholder - would fetch from options API
        return {"iv_rv_spread": 0.0, "signal": "neutral"}
    
    def analyze(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Analyzes volatility and forecasts future levels.
        
        Args:
            context: Must contain 'returns' (pd.Series)
            
        Returns:
            AgentResponse with volatility forecast
        """
        returns = context.get("returns")
        
        if returns is None or len(returns) < 30:
            return AgentResponse(
                agent_name=self.name,
                summary="Insufficient data for volatility forecasting.",
                confidence_score=0.3,
                structured_data={},
                recommendation="HOLD"
            )
        
        # Forecast volatility
        forecast = self.forecast_volatility(returns)
        
        # Calculate current realized vol
        current_vol = returns.std()
        forecasted_vol = forecast.get("garch_volatility", current_vol)
        
        # Detect regime change
        regime = self.detect_regime_change(current_vol, forecasted_vol)
        
        # Generate response
        if regime == "expanding":
            summary = f"Volatility expanding. Current: {current_vol:.2%}, Forecast: {forecasted_vol:.2%}"
            recommendation = "CAUTION"
            confidence = 0.8
        elif regime == "compressing":
            summary = f"Volatility compressing. Current: {current_vol:.2%}, Forecast: {forecasted_vol:.2%}"
            recommendation = "OPPORTUNITY"
            confidence = 0.75
        else:
            summary = f"Stable volatility. Current: {current_vol:.2%}, Forecast: {forecasted_vol:.2%}"
            recommendation = "NORMAL"
            confidence = 0.7
        
        return AgentResponse(
            agent_name=self.name,
            summary=summary,
            confidence_score=confidence,
            structured_data={
                "current_volatility": current_vol,
                "forecasted_volatility": forecasted_vol,
                "regime": regime,
                "forecast_details": forecast
            },
            recommendation=recommendation
        )


class RegimeDetectionAgent(BaseAgent):
    """
    Classifies market regime using HMM/Clustering from ML_Models directory.
    """
    
    def __init__(self, name: str = "RegimeDetectionAgent"):
        super().__init__(name)
        self.regime_model = None
        
        # Strategy mappings
        self.regime_strategies = {
            "Range": "Mean Reversion",
            "Trending Up": "Momentum Long",
            "Trending Down": "Defensive/Short",
            "Crisis": "Risk Off"
        }
    
    def predict_regime(self, indicators: pd.DataFrame) -> str:
        """
        Predicts current market regime.
        
        Args:
            indicators: DataFrame of technical indicators
            
        Returns:
            Regime label (e.g., "Range", "Trending Up")
        """
        # Would use trained Regime_Classifier here
        # For now, simple heuristic based on volatility and trend
        
        if len(indicators) < 60:
            return "Unknown"
        
        # Mock logic - replace with actual model
        returns = indicators.get("returns", pd.Series([0]))
        volatility = returns.std()
        
        if volatility > 0.02:
            return "Crisis"
        elif returns.mean() > 0.001:
            return "Trending Up"
        elif returns.mean() < -0.001:
            return "Trending Down"
        else:
            return "Range"
    
    def get_transition_probability(self) -> Dict[str, Dict[str, float]]:
        """
        Returns the probability matrix of regime transitions.
        
        Returns:
            Nested dictionary of transition probabilities
        """
        # Mock transition matrix
        return {
            "Range": {"Range": 0.7, "Trending Up": 0.15, "Trending Down": 0.1, "Crisis": 0.05},
            "Trending Up": {"Range": 0.2, "Trending Up": 0.6, "Trending Down": 0.1, "Crisis": 0.1},
            "Trending Down": {"Range": 0.2, "Trending Up": 0.1, "Trending Down": 0.6, "Crisis": 0.1},
            "Crisis": {"Range": 0.3, "Trending Up": 0.1, "Trending Down": 0.2, "Crisis": 0.4}
        }
    
    def suggest_strategy(self, regime_id: str) -> str:
        """
        Maps regime to recommended trading strategy.
        
        Args:
            regime_id: Regime label
            
        Returns:
            Strategy name
        """
        return self.regime_strategies.get(regime_id, "Undefined")
    
    def analyze(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Detects market regime and suggests strategy.
        
        Args:
            context: Must contain 'indicators' (pd.DataFrame)
            
        Returns:
            AgentResponse with regime classification
        """
        indicators = context.get("indicators", pd.DataFrame())
        
        if len(indicators) < 60:
            return AgentResponse(
                agent_name=self.name,
                summary="Insufficient data for regime detection.",
                confidence_score=0.4,
                structured_data={},
                recommendation="HOLD"
            )
        
        # Predict regime
        regime = self.predict_regime(indicators)
        strategy = self.suggest_strategy(regime)
        
        # Get transition probabilities
        transitions = self.get_transition_probability().get(regime, {})
        
        summary = f"Market regime: {regime}. Recommended strategy: {strategy}"
        
        return AgentResponse(
            agent_name=self.name,
            summary=summary,
            confidence_score=0.75,
            structured_data={
                "regime": regime,
                "strategy": strategy,
                "transition_probabilities": transitions
            },
            recommendation=strategy.upper().replace(" ", "_")
        )


# ==================== SYNTHESIS AGENT ====================

class SignalAggregatorAgent(BaseAgent):
    """
    Aggregates outputs from all agents and generates final decision using LLM.
    """
    
    def __init__(self, name: str = "SignalAggregatorAgent"):
        super().__init__(name)
        
        # LLM configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.use_lightweight_llm = os.getenv("USE_LIGHTWEIGHT_LLM", "false").lower() == "true"
    
    def resolve_conflicts(self, agent_outputs: List[AgentResponse]) -> Dict[str, Any]:
        """
        Implements conflict resolution logic.
        
        Example: "If Sentiment=BUY but Risk=HIGH, then HOLD"
        
        Args:
            agent_outputs: List of responses from all agents
            
        Returns:
            Dictionary with conflict analysis
        """
        conflicts = []
        
        # Extract key signals
        sentiment_signal = None
        risk_signal = None
        
        for output in agent_outputs:
            if output.agent_name == "SentimentAgent":
                sentiment_signal = output.recommendation
            elif output.agent_name == "RiskMonitoringAgent":
                risk_signal = output.recommendation
        
        # Check for conflicts
        if sentiment_signal in ["BULLISH"] and risk_signal == "REDUCE_EXPOSURE":
            conflicts.append({
                "type": "sentiment_vs_risk",
                "description": "Sentiment is bullish but risk limits breached",
                "resolution": "HOLD - Wait for risk to normalize"
            })
        
        return {"conflicts": conflicts, "has_conflicts": len(conflicts) > 0}
    
    def _construct_prompt(self, agent_outputs: List[AgentResponse]) -> str:
        """
        Formats agent outputs into an LLM prompt.
        
        Args:
            agent_outputs: List of responses from all agents
            
        Returns:
            Formatted prompt string
        """
        prompt = "You are a senior trading strategist. Analyze the following agent reports and provide a final trading decision.\n\n"
        
        for output in agent_outputs:
            prompt += f"**{output.agent_name}** (Weight: {self.performance_weight:.2f}):\n"
            prompt += f"- Summary: {output.summary}\n"
            prompt += f"- Recommendation: {output.recommendation}\n"
            prompt += f"- Confidence: {output.confidence_score:.2%}\n\n"
        
        prompt += "\nProvide:\n1. Overall market assessment\n2. Final recommendation (BUY/SELL/HOLD)\n3. Confidence level (0-100%)\n4. Key reasoning"
        
        return prompt
    
    def synthesize(self, agent_outputs: List[AgentResponse]) -> str:
        """
        Calls LLM to generate final reasoning.
        
        Args:
            agent_outputs: List of responses from all agents
            
        Returns:
            LLM-generated synthesis
        """
        prompt = self._construct_prompt(agent_outputs)
        
        # Select LLM based on complexity
        if self.use_lightweight_llm and self.groq_api_key:
            # Use Groq for simple tasks
            return self._call_groq(prompt)
        elif self.gemini_api_key:
            # Use Gemini for complex reasoning
            return self._call_gemini(prompt)
        else:
            # Fallback: Simple rule-based aggregation
            return self._simple_aggregation(agent_outputs)
    
    def _call_gemini(self, prompt: str) -> str:
        """Calls Google Gemini API"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return "LLM synthesis failed"
    
    def _call_groq(self, prompt: str) -> str:
        """Calls Groq API"""
        try:
            from groq import Groq
            client = Groq(api_key=self.groq_api_key)
            completion = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": prompt}]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling Groq: {e}")
            return "LLM synthesis failed"
    
    def _simple_aggregation(self, agent_outputs: List[AgentResponse]) -> str:
        """Fallback: Simple weighted voting"""
        recommendations = {}
        
        for output in agent_outputs:
            rec = output.recommendation
            weight = output.confidence_score
            recommendations[rec] = recommendations.get(rec, 0) + weight
        
        final_rec = max(recommendations, key=recommendations.get)
        return f"Simple aggregation result: {final_rec} (total weight: {recommendations[final_rec]:.2f})"
    
    def adjust_weights_dynamic(self, agent_outputs: List[AgentResponse]) -> None:
        """
        Adjusts agent weights based on their performance_weight attribute.
        
        This is called before synthesis to give more influence to recently accurate agents.
        
        Args:
            agent_outputs: List of responses from all agents
        """
        # This would retrieve the actual agent instances and use their performance_weight
        # For now, this is a placeholder indicating where dynamic weighting occurs
        pass
    
    def analyze(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Aggregates all agent outputs and generates final decision.
        
        Args:
            context: Must contain 'agent_outputs' (List[AgentResponse])
            
        Returns:
            AgentResponse with final aggregated decision
        """
        agent_outputs = context.get("agent_outputs", [])
        
        if len(agent_outputs) == 0:
            return AgentResponse(
                agent_name=self.name,
                summary="No agent outputs to aggregate.",
                confidence_score=0.0,
                structured_data={},
                recommendation="HOLD"
            )
        
        # Resolve conflicts
        conflict_analysis = self.resolve_conflicts(agent_outputs)
        
        # Synthesize using LLM
        llm_synthesis = self.synthesize(agent_outputs)
        
        # Generate final response
        summary = f"Aggregated analysis from {len(agent_outputs)} agents. {llm_synthesis}"
        
        return AgentResponse(
            agent_name=self.name,
            summary=summary,
            confidence_score=0.85,
            structured_data={
                "conflict_analysis": conflict_analysis,
                "llm_synthesis": llm_synthesis,
                "agent_count": len(agent_outputs)
            },
            recommendation="SYNTHESIZED"
        )
