import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Any, List

class Regime_Classifier:
    def __init__(self, features: pd.DataFrame, regime_labels: pd.Series, window_size: int = 60):
        """
        Initializes and trains both Regime Classifiers.
        
        Args:
            features (pd.DataFrame): Technical indicators (ADX, SMA_Diff, etc.).
            regime_labels (pd.Series): The target labels (e.g., 0='Range', 1='Trend Up', 2='Crisis').
            window_size (int): Lookback period for the LSTM sequences.
        """
        self.features = features
        self.raw_labels = regime_labels
        self.window_size = window_size
        
        # 1. Encode Labels (Convert 'Trending Up' -> 1, 'Crisis' -> 4)
        self.encoder = LabelEncoder()
        self.encoded_labels = self.encoder.fit_transform(self.raw_labels)
        self.num_classes = len(self.encoder.classes_)
        
        print(f"Detected {self.num_classes} Regimes: {self.encoder.classes_}")
        
        # 2. Train Random Forest (Instant State Classifier)
        print("Training Random Forest Classifier...")
        self.rf_model = self.train_rf()
        
        # 3. Train LSTM (Sequence/Transition Classifier)
        print("Training LSTM Classifier...")
        # Prepare 3D sequences and One-Hot targets for LSTM
        self.X_seq, self.y_seq_onehot = self._prepare_lstm_data()
        self.lstm_model = self.train_lstm()

    def train_rf(self) -> RandomForestClassifier:
        """
        Random Forest: Classifies regime based on the current day's snapshot.
        """
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,        # Slightly deeper to capture complex regime boundaries
            min_samples_leaf=5,
            class_weight='balanced', # Crucial: Crisis events are rare, this prevents ignoring them
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.features, self.encoded_labels)
        return rf

    def train_lstm(self) -> Model:
        """
        LSTM: Classifies regime based on the sequence of past behavior (Transitions).
        """
        # Architecture for Multi-Class Classification
        model = Sequential([
            Input(shape=(self.window_size, self.features.shape[1])),
            
            # Layer 1: Sequence processing
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            
            # Layer 2: Regime transition encoding
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            
            # Output Layer: Softmax for Multi-Class Probability
            # (e.g., [10% Range, 80% Uptrend, 10% Crisis])
            Dense(self.num_classes, activation='softmax') 
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy', # Required for multi-class
            metrics=['accuracy']
        )
        
        model.fit(self.X_seq, self.y_seq_onehot, epochs=20, batch_size=32, verbose=1)
        return model

    def _prepare_lstm_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper: Converts DataFrames to 3D sequences and One-Hot encodes targets.
        """
        X_vals = self.features.values
        y_vals = self.encoded_labels
        
        X, y = [], []
        # Create sequences
        for i in range(self.window_size, len(X_vals)):
            X.append(X_vals[i-self.window_size:i])
            y.append(y_vals[i]) # The regime at the END of the sequence
            
        # One-Hot Encode the target for Softmax output (e.g., 2 -> [0, 0, 1, 0])
        y_onehot = to_categorical(np.array(y), num_classes=self.num_classes)
        
        return np.array(X), y_onehot

    def predict_regime(self, current_sequence: np.ndarray) -> str:
        """
        Predicts the current market regime using the LSTM (more robust to noise).
        Returns the human-readable label (e.g., 'Crisis').
        """
        # Ensure 3D shape (1, 60, features)
        if current_sequence.ndim == 2:
            current_sequence = np.expand_dims(current_sequence, axis=0)
            
        # Get probabilities
        probs = self.lstm_model.predict(current_sequence, verbose=0)[0]
        
        # Get highest probability index
        predicted_index = np.argmax(probs)
        
        # Decode back to string
        return self.encoder.inverse_transform([predicted_index])[0]

    def save_models(self, directory: str = "./models") -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        joblib.dump(self.rf_model, os.path.join(directory, "regime_rf.pkl"))
        self.lstm_model.save(os.path.join(directory, "regime_lstm.keras"))
        
        # Important: Save the encoder to decode predictions later
        joblib.dump(self.encoder, os.path.join(directory, "regime_encoder.pkl"))

# --- USAGE ---
# regime_engine = Regime_Classifier(features=df_X, regime_labels=df['Regime'])
# current_state = regime_engine.predict_regime(last_60_days)
# print(f"Market Status: {current_state}")