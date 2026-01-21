from typing import Any, List, Tuple, Dict, Optional, Union, Callable, Iterable, Sequence, TypeVar
import numpy as np
import pandas as pd
import re
import json

from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor, _tree, export_text
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from arch import arch_model
from arch.univariate.base import ARCHModelResult

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

'''Linear regression provides a baseline volatility forecast using features like historical volatility, ATR, Bollinger 
bandwidth, and recent return magnitudes'''
class Baseline_Pred():
    def __init__(self,features: pd.DataFrame, y_target: pd.Series):
        self.features = features
        self.y_target = y_target
        print("Training the baseline linear regression ")
        self.lin_reg = self.linear_reg_train()

    def linear_reg_train(self) -> LinearRegression: 
        '''Baseline model'''
        lin_reg = LinearRegression()
        lin_reg.fit(self.features,self.y_target)
        return lin_reg

'''  Random Forest and XGBoost regressors extend this to capture non-linear relationships between these features and future realized volatility. '''

class Non_linear_Models():
    def __init__(self,features: pd.DataFrame, y_target: pd.Series):
        self.features = features
        self.y_target = y_target
        self.forest = self.random_forest_train()
        self.xgb = self.xgb_train()

    def random_forest_train(self) -> RandomForestRegressor: 
        '''Random Forest model'''
        forest = RandomForestRegressor()
        forest.fit(self.features,self.y_target)
        return forest
    def xgb_train(self) -> XGBRegressor: 
        '''XGBoost model'''
        xgb = XGBRegressor()
        xgb.fit(self.features,self.y_target)
        return xgb


class Volatility_Models:
    def __init__(self, returns: pd.Series):
        """
        Initializes and immediately trains both GARCH and EGARCH models.
        """
        self.returns = returns
        # Scale returns by 100 for optimizer stability (Standard practice for GARCH)
        self.scaled_returns = self.returns * 100
        
        # Trigger training immediately, matching your flow
        print("Training GARCH(1,1)...")
        self.garch = self.garch_train()
        
        print("Training EGARCH(1,1)...")
        self.egarch = self.egarch_train()

    def garch_train(self) -> ARCHModelResult:
        '''
        Standard GARCH Model: Captures volatility clustering.
        '''
        model = arch_model(
            self.scaled_returns, 
            vol='Garch', 
            p=1, q=1, 
            mean='Zero', 
            dist='Normal'
        )
        # fit(disp='off') prevents printing the summary log to console during training
        garch_result = model.fit(disp='off')
        return garch_result

    def egarch_train(self) -> ARCHModelResult:
        '''
        EGARCH Model: Captures asymmetric reaction to market crashes.
        '''
        model = arch_model(
            self.scaled_returns, 
            vol='EGARCH', 
            p=1, q=1, 
            mean='Zero', 
            dist='Normal'
        )
        egarch_result = model.fit(disp='off')
        return egarch_result

    def predict_volatility(self, horizon: int = 1) -> Dict[str, float]:
        '''
        Forecasts future volatility (Standard Deviation) for the next timestep.
        Returns a dictionary with both GARCH and EGARCH predictions.
        '''
        # 1. FORECAST VARIANCE (The model predicts Variance, not Std Dev)
        # We take the last row (.iloc[-1]) because we want the forecast from the most recent data point
        garch_var = self.garch.forecast(horizon=horizon).variance.iloc[-1].values[0]
        egarch_var = self.egarch.forecast(horizon=horizon).variance.iloc[-1].values[0]
        
        # 2. CONVERT TO STD DEV & RESCALE
        # Formula: sqrt(variance) / 100
        # We divide by 100 to reverse the scaling we did in __init__
        garch_vol = np.sqrt(garch_var) / 100
        egarch_vol = np.sqrt(egarch_var) / 100
        
        return {
            "garch_volatility": garch_vol,
            "egarch_volatility": egarch_vol
        }

#LSTM
class LSTM_Volatility_Model:
    def __init__(self, features: pd.DataFrame, returns: pd.Series, window_size: int = 60, forecast_horizon: int = 5):
        """
        Initializes and trains the LSTM Volatility Regressor.
        
        Args:
            features (pd.DataFrame): Input indicators (RSI, MACD, etc.).
            returns (pd.Series): Raw log-returns (used to calculate the volatility target).
            window_size (int): Lookback period for sequence (default 60).
            forecast_horizon (int): How many days ahead to calculate realized volatility (default 5).
        """
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        
        # 1. Prepare Target: Forward-Looking Realized Volatility
        print("Calculating forward-looking volatility targets...")
        self.y_target = self._calculate_realized_volatility(returns)
        
        # 2. Scale Features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.features_scaled = self.scaler.fit_transform(features)
        
        # 3. Create Sequences
        self.X_seq, self.y_seq = self._create_sequences(self.features_scaled, self.y_target.values)
        
        # 4. Train Model Immediately
        print("Training LSTM Volatility Model...")
        self.model = self.train_lstm()

    def _calculate_realized_volatility(self, returns: pd.Series) -> pd.Series:
        """
        Computes the target: Realized Volatility over the NEXT 'n' days.
        Logic: Rolling Std Dev, shifted backwards to align T with T+1...T+5.
        """
        # Calculate rolling standard deviation (volatility)
        # We shift(-horizon) because at time T, we want to predict the volatility of T+1 to T+horizon
        future_vol = returns.rolling(window=self.forecast_horizon).std().shift(-self.forecast_horizon)
        
        # Drop NaNs created by the shift at the end of the dataframe
        return future_vol.dropna()

    def _create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts 2D data into 3D sequences (Samples, Time Steps, Features).
        """
        X, y = [], []
        # Ensure we don't go out of bounds (Target length is shorter due to shift)
        length = min(len(features), len(target))
        
        for i in range(self.window_size, length):
            X.append(features[i-self.window_size:i]) # Past 60 days
            y.append(target[i])                      # Target Volatility at this step
            
        return np.array(X), np.array(y)

    def train_lstm(self) -> Model:
        """
        Builds and fits the Regression LSTM.
        """
        # Define Architecture
        # Input shape: (60 timesteps, N features)
        input_shape = (self.X_seq.shape[1], self.X_seq.shape[2])
        
        model = Sequential([
            Input(shape=input_shape),
            # Layer 1: Captures patterns
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            
            # Layer 2: Compresses to 'Market State' vector
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            
            # Layer 3: Regression processing
            Dense(32, activation='relu'),
            
            # Output: Single continuous value (Predicted Volatility)
            # Linear activation is standard for regression, or 'relu' to enforce positive output
            Dense(1, activation='linear') 
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Fit Model
        model.fit(self.X_seq, self.y_seq, epochs=20, batch_size=32, verbose=1, validation_split=0.1)
        
        return model

    def predict(self, current_sequence: np.ndarray) -> float:
        """
        Predicts volatility for the next horizon.
        """
        # Ensure input is 3D: (1, 60, features)
        if current_sequence.ndim == 2:
            current_sequence = np.expand_dims(current_sequence, axis=0)
            
        predicted_vol = self.model.predict(current_sequence, verbose=0)
        return float(predicted_vol[0][0])

# --- USAGE EXAMPLE ---
# vol_lstm = LSTM_Volatility_Model(features=df_features, returns=df_returns)
# pred = vol_lstm.predict(last_60_days_data)
# print(f"Predicted Volatility Spike: {pred}")

