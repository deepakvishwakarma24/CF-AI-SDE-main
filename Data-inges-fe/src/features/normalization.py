"""
Feature normalization module for ML model preparation.
Provides z-score and min-max normalization to ensure features are on comparable scales.
Stores both raw and normalized versions for different model types.
"""

import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureNormalizer:
    """
    Normalizes features for machine learning models.
    Supports z-score normalization (for neural networks) and min-max scaling.
    Stores normalization parameters for consistent transform on new data.
    """
    
    def __init__(self):
        """Initialize the feature normalizer."""
        self.normalization_params = {}
    
    def zscore_normalize(
        self,
        df: pd.DataFrame,
        columns: List[str],
        fit: bool = True,
        params: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply z-score normalization: (x - mean) / std
        
        Args:
            df: DataFrame with features
            columns: List of column names to normalize
            fit: If True, compute mean/std from data. If False, use provided params
            params: Pre-computed normalization parameters (for transform on new data)
            
        Returns:
            Tuple of (DataFrame with normalized columns added, normalization parameters)
        """
        df_norm = df.copy()
        norm_params = params if params is not None else {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if fit:
                # Fit: compute mean and std from current data
                col_mean = df[col].mean()
                col_std = df[col].std()
                
                # Store parameters
                norm_params[col] = {
                    'mean': float(col_mean),
                    'std': float(col_std),
                    'method': 'zscore'
                }
                
                # Normalize
                if col_std > 0:
                    df_norm[f'{col}_norm'] = (df[col] - col_mean) / col_std
                else:
                    df_norm[f'{col}_norm'] = 0
            else:
                # Transform: use pre-computed parameters
                if col in norm_params:
                    col_mean = norm_params[col]['mean']
                    col_std = norm_params[col]['std']
                    
                    if col_std > 0:
                        df_norm[f'{col}_norm'] = (df[col] - col_mean) / col_std
                    else:
                        df_norm[f'{col}_norm'] = 0
                else:
                    logger.warning(f"No normalization params found for {col}")
        
        return df_norm, norm_params
    
    def minmax_normalize(
        self,
        df: pd.DataFrame,
        columns: List[str],
        fit: bool = True,
        params: Optional[Dict] = None,
        feature_range: Tuple[float, float] = (0, 1)
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply min-max normalization: (x - min) / (max - min)
        
        Args:
            df: DataFrame with features
            columns: List of column names to normalize
            fit: If True, compute min/max from data. If False, use provided params
            params: Pre-computed normalization parameters
            feature_range: Desired range of transformed data (default: 0 to 1)
            
        Returns:
            Tuple of (DataFrame with normalized columns added, normalization parameters)
        """
        df_norm = df.copy()
        norm_params = params if params is not None else {}
        min_val, max_val = feature_range
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if fit:
                # Fit: compute min and max from current data
                col_min = df[col].min()
                col_max = df[col].max()
                
                # Store parameters
                norm_params[col] = {
                    'min': float(col_min),
                    'max': float(col_max),
                    'method': 'minmax',
                    'range': feature_range
                }
                
                # Normalize
                if col_max > col_min:
                    normalized = (df[col] - col_min) / (col_max - col_min)
                    df_norm[f'{col}_norm'] = normalized * (max_val - min_val) + min_val
                else:
                    df_norm[f'{col}_norm'] = min_val
            else:
                # Transform: use pre-computed parameters
                if col in norm_params:
                    col_min = norm_params[col]['min']
                    col_max = norm_params[col]['max']
                    
                    if col_max > col_min:
                        normalized = (df[col] - col_min) / (col_max - col_min)
                        df_norm[f'{col}_norm'] = normalized * (max_val - min_val) + min_val
                    else:
                        df_norm[f'{col}_norm'] = min_val
                else:
                    logger.warning(f"No normalization params found for {col}")
        
        return df_norm, norm_params
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        method: str = 'zscore',
        exclude_columns: Optional[List[str]] = None,
        fit: bool = True,
        params: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize all numeric features in DataFrame.
        
        Args:
            df: DataFrame with features
            method: Normalization method ('zscore' or 'minmax')
            exclude_columns: Columns to exclude from normalization (e.g., OHLCV, timestamp)
            fit: Whether to fit parameters from data
            params: Pre-computed parameters if fit=False
            
        Returns:
            Tuple of (DataFrame with both raw and normalized features, parameters)
        """
        if exclude_columns is None:
            exclude_columns = ['timestamp', 'symbol']
        
        # Identify numeric columns to normalize
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_columns]
        
        logger.info(f"Normalizing {len(cols_to_normalize)} features using {method} method")
        
        if method == 'zscore':
            df_norm, norm_params = self.zscore_normalize(df, cols_to_normalize, fit, params)
        elif method == 'minmax':
            df_norm, norm_params = self.minmax_normalize(df, cols_to_normalize, fit, params)
        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'zscore' or 'minmax'")
        
        # Store parameters in instance
        if fit:
            self.normalization_params = norm_params
        
        return df_norm, norm_params
    
    def save_normalization_params(self, params: Dict, filepath: Path) -> None:
        """
        Save normalization parameters to JSON file for reproducibility.
        
        Args:
            params: Normalization parameters dictionary
            filepath: Path to save parameters
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_params = {}
        for key, value in params.items():
            if isinstance(value, dict):
                serializable_params[key] = {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in value.items()
                }
            else:
                serializable_params[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_params, f, indent=2)
        
        logger.info(f"Saved normalization parameters to {filepath}")
    
    def load_normalization_params(self, filepath: Path) -> Dict:
        """
        Load normalization parameters from JSON file.
        
        Args:
            filepath: Path to parameters file
            
        Returns:
            Normalization parameters dictionary
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Normalization parameters file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        logger.info(f"Loaded normalization parameters from {filepath}")
        self.normalization_params = params
        
        return params
    
    def prepare_ml_dataset(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        use_normalized: bool = True,
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare dataset for ML models by separating features from target.
        
        Args:
            df: DataFrame with features and optional target
            target_column: Name of target column (if present)
            use_normalized: If True, use normalized features (*_norm columns)
            exclude_columns: Additional columns to exclude from features
            
        Returns:
            Tuple of (feature DataFrame, target Series or None)
        """
        if exclude_columns is None:
            exclude_columns = []
        
        # Default columns to exclude
        default_exclude = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                          'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume',
                          'dividends', 'stock_splits']
        exclude_columns.extend(default_exclude)
        
        # Add target to exclude list
        if target_column:
            exclude_columns.append(target_column)
        
        # Select features
        if use_normalized:
            # Use only normalized features (columns ending with _norm)
            feature_cols = [col for col in df.columns 
                          if col.endswith('_norm') and col not in exclude_columns]
        else:
            # Use raw features (exclude normalized columns)
            feature_cols = [col for col in df.columns 
                          if col not in exclude_columns and not col.endswith('_norm')]
        
        X = df[feature_cols]
        y = df[target_column] if target_column and target_column in df.columns else None
        
        logger.info(f"Prepared ML dataset: {len(feature_cols)} features, "
                   f"{'normalized' if use_normalized else 'raw'} values")
        
        return X, y
    
    def get_feature_vector(
        self,
        df: pd.DataFrame,
        timestamp: pd.Timestamp,
        use_normalized: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract feature vector for a specific timestamp (for real-time inference).
        
        Args:
            df: DataFrame with features
            timestamp: Timestamp to extract features for
            use_normalized: Whether to use normalized features
            
        Returns:
            Tuple of (feature vector as numpy array, feature names list)
        """
        # Filter to specific timestamp
        row = df[df['timestamp'] == timestamp]
        
        if len(row) == 0:
            raise ValueError(f"No data found for timestamp: {timestamp}")
        
        # Get features using prepare_ml_dataset
        X, _ = self.prepare_ml_dataset(row, use_normalized=use_normalized)
        
        feature_vector = X.values.flatten()
        feature_names = X.columns.tolist()
        
        return feature_vector, feature_names
