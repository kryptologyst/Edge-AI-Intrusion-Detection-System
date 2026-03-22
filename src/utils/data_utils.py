"""Data generation and preprocessing utilities for intrusion detection."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig


class NetworkTrafficGenerator:
    """Generate synthetic network traffic data for intrusion detection."""
    
    def __init__(self, config: DictConfig):
        """Initialize the traffic generator.
        
        Args:
            config: Configuration object containing data parameters.
        """
        self.config = config
        self.n_samples = config.data.n_samples
        self.random_state = config.data.random_state
        self.feature_names = config.data.features
        
    def generate_features(self) -> np.ndarray:
        """Generate synthetic network traffic features.
        
        Returns:
            Array of shape (n_samples, n_features) containing traffic features.
        """
        np.random.seed(self.random_state)
        
        # Duration: exponential distribution (most connections are short)
        duration = np.random.exponential(scale=2.0, size=self.n_samples)
        
        # Bytes sent: normal distribution with outliers
        bytes_sent = np.random.normal(1000, 300, self.n_samples)
        bytes_sent = np.clip(bytes_sent, 0, None)  # Ensure non-negative
        
        # Bytes received: normal distribution with outliers
        bytes_received = np.random.normal(1200, 400, self.n_samples)
        bytes_received = np.clip(bytes_received, 0, None)  # Ensure non-negative
        
        # Failed logins: Poisson distribution (rare events)
        failed_logins = np.random.poisson(0.2, self.n_samples)
        
        # Suspicious flags: Poisson distribution (rare events)
        suspicious_flags = np.random.poisson(0.3, self.n_samples)
        
        # Stack features
        features = np.stack([
            duration, bytes_sent, bytes_received, 
            failed_logins, suspicious_flags
        ], axis=1)
        
        return features
    
    def generate_labels(self, features: np.ndarray) -> np.ndarray:
        """Generate intrusion labels based on feature patterns.
        
        Args:
            features: Feature array of shape (n_samples, n_features).
            
        Returns:
            Binary labels (0=normal, 1=intrusion).
        """
        duration = features[:, 0]
        bytes_sent = features[:, 1]
        bytes_received = features[:, 2]
        failed_logins = features[:, 3]
        suspicious_flags = features[:, 4]
        
        # Intrusion detection rules
        intrusion = (
            (failed_logins > 1) |  # Multiple failed login attempts
            (suspicious_flags > 1) |  # Multiple suspicious flags
            ((duration > 5) & (bytes_sent > 2000)) |  # Long duration + large transfer
            ((duration > 3) & (bytes_received > 3000))  # Medium duration + large received
        ).astype(int)
        
        return intrusion
    
    def generate_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete dataset with features and labels.
        
        Returns:
            Tuple of (features, labels).
        """
        features = self.generate_features()
        labels = self.generate_labels(features)
        
        return features, labels


class DataPreprocessor:
    """Preprocess network traffic data for model training."""
    
    def __init__(self, scaler_type: str = 'standard'):
        """Initialize the preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax').
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.is_fitted = False
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data.
        
        Args:
            X: Input features.
            
        Returns:
            Scaled features.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler.
        
        Args:
            X: Input features.
            
        Returns:
            Scaled features.
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        return self.scaler.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data.
        
        Args:
            X: Scaled features.
            
        Returns:
            Original scale features.
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        return self.scaler.inverse_transform(X)


def create_data_splits(
    X: np.ndarray, 
    y: np.ndarray, 
    config: DictConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create train/test splits for the dataset.
    
    Args:
        X: Feature array.
        y: Label array.
        config: Configuration object.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    return train_test_split(
        X, y,
        test_size=config.training.validation_split,
        random_state=config.data.random_state,
        stratify=y  # Ensure balanced splits
    )


def get_feature_statistics(X: np.ndarray, feature_names: list) -> Dict[str, Dict[str, float]]:
    """Calculate statistics for each feature.
    
    Args:
        X: Feature array.
        feature_names: List of feature names.
        
    Returns:
        Dictionary containing statistics for each feature.
    """
    stats = {}
    for i, name in enumerate(feature_names):
        feature_data = X[:, i]
        stats[name] = {
            'mean': np.mean(feature_data),
            'std': np.std(feature_data),
            'min': np.min(feature_data),
            'max': np.max(feature_data),
            'median': np.median(feature_data),
            'q25': np.percentile(feature_data, 25),
            'q75': np.percentile(feature_data, 75)
        }
    return stats
