# ML Models Documentation

This directory contains various machine learning models for financial forecasting and analysis. The models are categorized into Volatility Forecasting, Direction Prediction, and Regime Classification.

## 1. Volatility Forecasting
**File:** `Volatility_Forecasting.py`

This module focuses on predicting the future volatility of financial assets. It employs a mix of classical statistical models and modern machine learning techniques.

### Classes

#### `Baseline_Pred`
- **Purpose**: Establishes a baseline for volatility prediction using linear regression.
- **Algorithm**: `LinearRegression` (Scikit-Learn).
- **Inputs**: DataFrame of features and target volatility series.

#### `Non_linear_Models`
- **Purpose**: Captures non-linear relationships in volatility data.
- **Algorithms**:
    - `RandomForestRegressor`: Ensemble learning method.
    - `XGBRegressor`: Gradient boosting framework.

#### `Volatility_Models`
- **Purpose**: Implements classical time-series volatility models.
- **Methods**:
    - `garch_train()`: Trains a standard GARCH(1,1) model to capture volatility clustering.
    - `egarch_train()`: Trains an EGARCH(1,1) model to handle asymmetric volatility shocks (leverage effect).
    - `predict_volatility(horizon)`: Returns the standard deviation forecast for the next `horizon` steps.

#### `LSTM_Volatility_Model`
- **Purpose**: Deep learning approach for volatility regression.
- **Algorithm**: Long Short-Term Memory (LSTM) Neural Network.
- **Architecture**:
    - Input Layer -> LSTM (128 units) -> Dropout -> LSTM (64 units) -> Dense (32) -> Output.
- **Key Features**:
    - Uses a lookback window (default 60 days).
    - Predicts realized volatility for a future horizon (default 5 days).

### Usage Guide

**Data Requirements:**
- **`features`**: Pandas DataFrame of technical indicators (RSI, MACD, etc.).
- **`returns`**: Pandas Series of asset log-returns.

**Example Code:**
```python
from Volatility_Forecasting import Volatility_Models, LSTM_Volatility_Model

# --- 1. Statistical Models (GARCH/EGARCH) ---
# Requires only returns series
vol_stats = Volatility_Models(returns=df['log_returns'])
# Get forecast for next day
forecast = vol_stats.predict_volatility(horizon=1) 
print(forecast) # {'garch_volatility': 0.012, 'egarch_volatility': 0.015}

# --- 2. LSTM Regression ---
# Requires features and returns for training target generation
vol_lstm = LSTM_Volatility_Model(features=df_X, returns=df['log_returns'])

# Predict using the last 60 days of data
# Input shape: (1, 60, n_features)
import numpy as np
last_sequence = np.array([df_X.iloc[-60:].values]) 
prediction = vol_lstm.predict(last_sequence)
```

---

## 2. Direction Prediction
**File:** `direction_pred.py`

This module aims to predict the market direction (Up/Down) based on input features.

### Classes

#### `Baseline_Pred`
- **Purpose**: Baseline classification model.
- **Algorithm**: `LogisticRegression`.

#### `Extract_Rules`
- **Purpose**: Interpretable model to extract decision rules.
- **Algorithm**: `DecisionTreeClassifier`.
- **Key Method**: `get_rules()` returns JSON-formatted decision rules extracted from the tree structure.

#### `Feature_importance`
- **Purpose**: Identifies the most significant features driving market direction.
- **Algorithm**: `RandomForestClassifier`.
- **Key Method**: `rank_features(n_top)` returns the top N most important features.

#### `XGBoost_Pred`
- **Purpose**: High-performance gradient boosting for classification.
- **Algorithm**: `XGBClassifier`.

#### `FF_NN` (Feed Forward Neural Network)
- **Purpose**: Captures non-linear complex patterns.
- **Architecture**: Dense (128) -> Dense (64) -> Dense (32) -> Sigmoid Output.

#### `LSTM_Pred`
- **Purpose**: Captures temporal dependencies in sequential data.
- **Architecture**: LSTM (128) -> LSTM (64) -> Dense (1, Sigmoid).

### Usage Guide

**Data Requirements:**
- **`features`**: Pandas DataFrame of input variables.
- **`y_direction`**: Pandas Series of binary labels (0 = Down, 1 = Up).

**Example Code:**
```python
from direction_pred import XGBoost_Pred, Extract_Rules

# --- 1. Train Classifiers ---
xgb_model = XGBoost_Pred(features=df_X, y_direction=df['Target_Binary'])

# --- 2. Extract Explainable Rules ---
# Useful for understanding WHY the market is predicted to move
rule_miner = Extract_Rules(features=df_X, y_direction=df['Target_Binary'])
rules_json = rule_miner.get_rules()
print(rules_json) # Outputs decision tree path logic
```

---

## 3. Regime Classification
**File:** `Regime_Classificaiton.py`

This module classifies the market into different states or "regimes" (e.g., Range, Trend Up, Crisis).

### Classes

#### `Regime_Classifier`
- **Purpose**: Dual-model approach to detect market regimes.
- **Components**:
    1.  **Random Forest Classifier (`train_rf`)**:
        -   Analyzes the current day's snapshot to classify the regime.
        -   Uses `class_weight='balanced'` to handle rare events like crises.
    2.  **LSTM Classifier (`train_lstm`)**:
        -   Analyzes a sequence of past behavior (default 60 days) to classify based on transitions.
        -   Outputs probabilities for each regime class.
- **Key Method**: `predict_regime(current_sequence)` uses the LSTM model to predict the current market status labels.

### Usage Guide

**Data Requirements:**
- **`features`**: Pandas DataFrame of indicators.
- **`regime_labels`**: Pandas Series containing class labels (0, 1, 2) or names ("Range", "Bull", "Bear").

**Example Code:**
```python
from Regime_Classificaiton import Regime_Classifier

# Initialize Classifiers (Trains both Random Forest and LSTM)
classifier = Regime_Classifier(
    features=df_X, 
    regime_labels=df['Regime'], 
    window_size=60
)

# Predict Current Regime
# Requires a sequence of the last 60 days
# Input shape: (n_features, 60) -> Handled internally if 2D provided? 
# Note: Ensure input is (1, 60, n_features) for best safety
current_state = classifier.predict_regime(last_60_days_data)
print(f"Current Market Regime: {current_state}")
```

---

## 4. Generative Models
**File:** `GAN.py`

This module implements a Conditional Generative Adversarial Network (cGAN) to synthesize financial market data sequences conditioned on specific market regimes. This helps in data augmentation and stress testing strategies under different market conditions.

### Classes

#### `GANConfig`
- **Purpose**: Centralized configuration for the GAN hyperparameters.
- **Key Parameters**: `seq_len` (30), `z_dim` (64), `hidden_dim` (128), `num_features` (5), `num_regimes` (3).

#### `MarketGAN`
- **Purpose**: The main GAN model class inheriting from `tf.keras.Model`, encapsulating both the Generator and Discriminator.
- **Components**:
    - **Generator**: 
        - Takes random noise and a regime label as input.
        - Uses LSTM layers (Configurable hidden dim, default 128) to generate synthetic market sequences.
        - Output shape: `(seq_len, num_features)`.
    - **Discriminator**:
        - Takes a market sequence and a regime label as input.
        - Uses LSTM layers to distinguish between real and fake sequences.
        - Output: Probability of the sequence being real.
- **Training**: Custom `train_step` implementing the Min-Max adversarial loss. 
- **Key Methods**:
    - `generate(regime_label, num_samples)`: Public method to generate `num_samples` of synthetic data for a given `regime_label` (e.g., Generate 1000 samples of "Bear Market" data).

### Usage Guide

**Data Requirements:**
- **`features`**: 3D Numpy Array `(num_samples, seq_len, num_features)`
- **`labels`**: 1D Numpy Array `(num_samples,)`

**Example Code:**
```python
from GAN import GANConfig, MarketGAN, get_dataset
import tensorflow as tf

# 1. Setup Configuration
config = GANConfig()

# 2. Initialize and Compile
gan = MarketGAN(config)
gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(config.lr),
    g_optimizer=tf.keras.optimizers.Adam(config.lr),
    loss_fn=tf.keras.losses.BinaryCrossentropy()
)

# 3. Train
# X_train shape: (N, 30, 5), y_train shape: (N,)
gan.fit(X_train, y_train, batch_size=64, epochs=100)

# 4. Generate Synthetic Data
# Create 100 samples of 'Crisis' data (assuming label 2 = Crisis)
synthetic_data = gan.generate(regime_label=2, num_samples=100)
print(synthetic_data.shape) # (100, 30, 5)
```
