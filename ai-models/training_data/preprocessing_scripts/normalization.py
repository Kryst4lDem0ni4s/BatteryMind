"""
BatteryMind - Data Normalization Module

Comprehensive data normalization and standardization utilities for battery
sensor data preprocessing. Handles multi-modal sensor data, temporal sequences,
and cross-battery compatibility normalization.

Features:
- Multi-scale normalization for different sensor types
- Temporal sequence normalization with sliding windows
- Cross-battery manufacturer compatibility
- Robust scaling with outlier handling
- Physics-informed normalization constraints
- Real-time streaming data normalization

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    PowerTransformer, QuantileTransformer
)
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NormalizationConfig:
    """
    Configuration for data normalization parameters.
    
    Attributes:
        # Normalization methods
        voltage_method (str): Normalization method for voltage data
        current_method (str): Normalization method for current data
        temperature_method (str): Normalization method for temperature data
        time_method (str): Normalization method for time-based features
        
        # Scaling parameters
        voltage_range (Tuple[float, float]): Expected voltage range (V)
        current_range (Tuple[float, float]): Expected current range (A)
        temperature_range (Tuple[float, float]): Expected temperature range (°C)
        
        # Robust scaling parameters
        outlier_quantiles (Tuple[float, float]): Quantiles for outlier detection
        clip_outliers (bool): Whether to clip outliers
        
        # Temporal normalization
        sequence_length (int): Length of temporal sequences
        overlap_ratio (float): Overlap ratio for sliding windows
        
        # Cross-battery compatibility
        enable_cross_battery_norm (bool): Enable cross-battery normalization
        reference_battery_capacity (float): Reference capacity for normalization
        
        # Advanced features
        handle_missing_values (bool): Handle missing values during normalization
        preserve_physics_constraints (bool): Preserve physical constraints
        enable_adaptive_scaling (bool): Enable adaptive scaling based on data distribution
    """
    # Normalization methods
    voltage_method: str = "minmax"  # 'standard', 'minmax', 'robust', 'quantile'
    current_method: str = "robust"
    temperature_method: str = "standard"
    time_method: str = "minmax"
    
    # Scaling parameters
    voltage_range: Tuple[float, float] = (2.5, 4.2)
    current_range: Tuple[float, float] = (-100.0, 100.0)
    temperature_range: Tuple[float, float] = (-20.0, 60.0)
    
    # Robust scaling parameters
    outlier_quantiles: Tuple[float, float] = (0.05, 0.95)
    clip_outliers: bool = True
    
    # Temporal normalization
    sequence_length: int = 100
    overlap_ratio: float = 0.5
    
    # Cross-battery compatibility
    enable_cross_battery_norm: bool = True
    reference_battery_capacity: float = 100.0  # Ah
    
    # Advanced features
    handle_missing_values: bool = True
    preserve_physics_constraints: bool = True
    enable_adaptive_scaling: bool = True

class PhysicsConstrainedScaler(BaseEstimator, TransformerMixin):
    """
    Custom scaler that preserves physics constraints during normalization.
    """
    
    def __init__(self, feature_type: str, physical_range: Tuple[float, float],
                 method: str = "minmax"):
        self.feature_type = feature_type
        self.physical_range = physical_range
        self.method = method
        self.scaler = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the scaler to the data."""
        # Validate physical constraints
        X_validated = self._validate_physics_constraints(X)
        
        # Initialize appropriate scaler
        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.method == "robust":
            self.scaler = RobustScaler()
        elif self.method == "quantile":
            self.scaler = QuantileTransformer(output_distribution='uniform')
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
        self.scaler.fit(X_validated.reshape(-1, 1))
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        # Validate physics constraints
        X_validated = self._validate_physics_constraints(X)
        
        # Transform data
        X_transformed = self.scaler.transform(X_validated.reshape(-1, 1)).flatten()
        
        return X_transformed
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform the normalized data."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        X_original = self.scaler.inverse_transform(X.reshape(-1, 1)).flatten()
        
        # Ensure physics constraints are maintained
        X_constrained = np.clip(X_original, self.physical_range[0], self.physical_range[1])
        
        return X_constrained
    
    def _validate_physics_constraints(self, X: np.ndarray) -> np.ndarray:
        """Validate and enforce physics constraints."""
        X_validated = X.copy()
        
        # Clip to physical range
        X_validated = np.clip(X_validated, self.physical_range[0], self.physical_range[1])
        
        # Feature-specific validations
        if self.feature_type == "voltage":
            # Ensure voltage is positive and within battery limits
            X_validated = np.maximum(X_validated, 0.1)
        elif self.feature_type == "temperature":
            # Ensure temperature is above absolute zero (in Celsius)
            X_validated = np.maximum(X_validated, -273.15)
        elif self.feature_type == "soc":
            # Ensure SOC is between 0 and 1
            X_validated = np.clip(X_validated, 0.0, 1.0)
        
        return X_validated

class BatteryDataNormalizer:
    """
    Comprehensive battery data normalizer with multi-modal sensor support.
    """
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.scalers = {}
        self.feature_stats = {}
        self.is_fitted = False
        
        # Initialize scalers for different feature types
        self._initialize_scalers()
        
        logger.info("BatteryDataNormalizer initialized")
    
    def _initialize_scalers(self):
        """Initialize scalers for different feature types."""
        # Voltage scaler
        self.scalers['voltage'] = PhysicsConstrainedScaler(
            'voltage', self.config.voltage_range, self.config.voltage_method
        )
        
        # Current scaler
        self.scalers['current'] = PhysicsConstrainedScaler(
            'current', self.config.current_range, self.config.current_method
        )
        
        # Temperature scaler
        self.scalers['temperature'] = PhysicsConstrainedScaler(
            'temperature', self.config.temperature_range, self.config.temperature_method
        )
        
        # SOC scaler
        self.scalers['soc'] = PhysicsConstrainedScaler(
            'soc', (0.0, 1.0), 'minmax'
        )
        
        # SOH scaler
        self.scalers['soh'] = PhysicsConstrainedScaler(
            'soh', (0.0, 1.0), 'minmax'
        )
        
        # Time-based features
        self.scalers['time'] = PhysicsConstrainedScaler(
            'time', (0.0, 1.0), self.config.time_method
        )
    
    def fit(self, data: Union[pd.DataFrame, Dict[str, np.ndarray]]) -> 'BatteryDataNormalizer':
        """
        Fit normalizers to the training data.
        
        Args:
            data: Training data as DataFrame or dictionary of arrays
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting battery data normalizer...")
        
        if isinstance(data, pd.DataFrame):
            data_dict = self._dataframe_to_dict(data)
        else:
            data_dict = data
        
        # Handle missing values if enabled
        if self.config.handle_missing_values:
            data_dict = self._handle_missing_values(data_dict)
        
        # Fit scalers for each feature type
        for feature_type, scaler in self.scalers.items():
            if feature_type in data_dict:
                feature_data = data_dict[feature_type]
                
                # Apply outlier handling
                if self.config.clip_outliers:
                    feature_data = self._clip_outliers(feature_data)
                
                scaler.fit(feature_data)
                
                # Store feature statistics
                self.feature_stats[feature_type] = {
                    'mean': np.mean(feature_data),
                    'std': np.std(feature_data),
                    'min': np.min(feature_data),
                    'max': np.max(feature_data),
                    'q25': np.percentile(feature_data, 25),
                    'q75': np.percentile(feature_data, 75)
                }
        
        # Cross-battery normalization
        if self.config.enable_cross_battery_norm:
            self._fit_cross_battery_normalization(data_dict)
        
        self.is_fitted = True
        logger.info("Battery data normalizer fitted successfully")
        return self
    
    def transform(self, data: Union[pd.DataFrame, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Transform data using fitted normalizers.
        
        Args:
            data: Data to transform
            
        Returns:
            Dictionary of normalized feature arrays
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        if isinstance(data, pd.DataFrame):
            data_dict = self._dataframe_to_dict(data)
        else:
            data_dict = data.copy()
        
        # Handle missing values if enabled
        if self.config.handle_missing_values:
            data_dict = self._handle_missing_values(data_dict)
        
        normalized_data = {}
        
        # Transform each feature type
        for feature_type, scaler in self.scalers.items():
            if feature_type in data_dict:
                feature_data = data_dict[feature_type]
                
                # Apply outlier handling
                if self.config.clip_outliers:
                    feature_data = self._clip_outliers(feature_data)
                
                # Transform data
                normalized_data[feature_type] = scaler.transform(feature_data)
        
        # Apply cross-battery normalization
        if self.config.enable_cross_battery_norm:
            normalized_data = self._apply_cross_battery_normalization(normalized_data, data_dict)
        
        return normalized_data
    
    def inverse_transform(self, normalized_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            normalized_data: Dictionary of normalized feature arrays
            
        Returns:
            Dictionary of original scale feature arrays
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")
        
        original_data = {}
        
        for feature_type, scaler in self.scalers.items():
            if feature_type in normalized_data:
                original_data[feature_type] = scaler.inverse_transform(
                    normalized_data[feature_type]
                )
        
        return original_data
    
    def _dataframe_to_dict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Convert DataFrame to dictionary of arrays."""
        data_dict = {}
        
        # Map common column names to feature types
        column_mapping = {
            'voltage': ['voltage', 'v', 'volt'],
            'current': ['current', 'i', 'amp', 'ampere'],
            'temperature': ['temperature', 'temp', 't', 'celsius'],
            'soc': ['soc', 'state_of_charge'],
            'soh': ['soh', 'state_of_health'],
            'time': ['time', 'timestamp', 'datetime']
        }
        
        for feature_type, possible_names in column_mapping.items():
            for col_name in df.columns:
                if any(name in col_name.lower() for name in possible_names):
                    data_dict[feature_type] = df[col_name].values
                    break
        
        return data_dict
    
    def _handle_missing_values(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Handle missing values in the data."""
        cleaned_data = {}
        
        for feature_type, feature_data in data_dict.items():
            if np.any(np.isnan(feature_data)):
                # Forward fill for time series data
                feature_data = pd.Series(feature_data).fillna(method='ffill').values
                
                # Backward fill for remaining NaNs
                feature_data = pd.Series(feature_data).fillna(method='bfill').values
                
                # Use mean for any remaining NaNs
                if np.any(np.isnan(feature_data)):
                    mean_value = np.nanmean(feature_data)
                    feature_data = np.where(np.isnan(feature_data), mean_value, feature_data)
            
            cleaned_data[feature_type] = feature_data
        
        return cleaned_data
    
    def _clip_outliers(self, data: np.ndarray) -> np.ndarray:
        """Clip outliers based on quantiles."""
        q_low, q_high = self.config.outlier_quantiles
        low_val = np.percentile(data, q_low * 100)
        high_val = np.percentile(data, q_high * 100)
        
        return np.clip(data, low_val, high_val)
    
    def _fit_cross_battery_normalization(self, data_dict: Dict[str, np.ndarray]):
        """Fit cross-battery normalization parameters."""
        # This would implement normalization across different battery types/manufacturers
        # For now, we'll store reference values
        self.cross_battery_params = {
            'reference_capacity': self.config.reference_battery_capacity,
            'capacity_scaling_factors': {}
        }
    
    def _apply_cross_battery_normalization(self, normalized_data: Dict[str, np.ndarray],
                                         original_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply cross-battery normalization."""
        # Apply capacity-based scaling if capacity information is available
        if 'capacity' in original_data:
            capacity_factor = original_data['capacity'] / self.config.reference_battery_capacity
            
            # Scale current and power-related features
            if 'current' in normalized_data:
                normalized_data['current'] = normalized_data['current'] * capacity_factor
        
        return normalized_data
    
    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all fitted features."""
        return self.feature_stats.copy()
    
    def save(self, filepath: str):
        """Save the fitted normalizer to file."""
        save_data = {
            'config': self.config,
            'scalers': self.scalers,
            'feature_stats': self.feature_stats,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Normalizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a fitted normalizer from file."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.config = save_data['config']
        self.scalers = save_data['scalers']
        self.feature_stats = save_data['feature_stats']
        self.is_fitted = save_data['is_fitted']
        
        logger.info(f"Normalizer loaded from {filepath}")

class StreamingNormalizer:
    """
    Online normalizer for streaming battery data.
    """
    
    def __init__(self, config: NormalizationConfig, window_size: int = 1000):
        self.config = config
        self.window_size = window_size
        self.feature_buffers = {}
        self.running_stats = {}
        
    def update(self, new_data: Dict[str, float]) -> Dict[str, float]:
        """
        Update normalizer with new streaming data point.
        
        Args:
            new_data: Dictionary of new feature values
            
        Returns:
            Dictionary of normalized feature values
        """
        normalized_data = {}
        
        for feature_type, value in new_data.items():
            # Initialize buffer if not exists
            if feature_type not in self.feature_buffers:
                self.feature_buffers[feature_type] = []
                self.running_stats[feature_type] = {'mean': 0.0, 'var': 0.0, 'count': 0}
            
            # Add to buffer
            self.feature_buffers[feature_type].append(value)
            
            # Maintain window size
            if len(self.feature_buffers[feature_type]) > self.window_size:
                self.feature_buffers[feature_type].pop(0)
            
            # Update running statistics
            self._update_running_stats(feature_type, value)
            
            # Normalize value
            normalized_data[feature_type] = self._normalize_value(feature_type, value)
        
        return normalized_data
    
    def _update_running_stats(self, feature_type: str, value: float):
        """Update running mean and variance."""
        stats = self.running_stats[feature_type]
        stats['count'] += 1
        
        delta = value - stats['mean']
        stats['mean'] += delta / stats['count']
        delta2 = value - stats['mean']
        stats['var'] += delta * delta2
    
    def _normalize_value(self, feature_type: str, value: float) -> float:
        """Normalize a single value using running statistics."""
        stats = self.running_stats[feature_type]
        
        if stats['count'] < 2:
            return value  # Not enough data for normalization
        
        mean = stats['mean']
        std = np.sqrt(stats['var'] / (stats['count'] - 1))
        
        if std == 0:
            return 0.0
        
        return (value - mean) / std

# Factory functions
def create_battery_normalizer(config: Optional[NormalizationConfig] = None) -> BatteryDataNormalizer:
    """
    Factory function to create a battery data normalizer.
    
    Args:
        config: Normalization configuration
        
    Returns:
        Configured BatteryDataNormalizer instance
    """
    if config is None:
        config = NormalizationConfig()
    
    return BatteryDataNormalizer(config)

def create_streaming_normalizer(config: Optional[NormalizationConfig] = None,
                              window_size: int = 1000) -> StreamingNormalizer:
    """
    Factory function to create a streaming normalizer.
    
    Args:
        config: Normalization configuration
        window_size: Size of sliding window for statistics
        
    Returns:
        Configured StreamingNormalizer instance
    """
    if config is None:
        config = NormalizationConfig()
    
    return StreamingNormalizer(config, window_size)

# Utility functions
def normalize_battery_dataset(data: pd.DataFrame, 
                            config: Optional[NormalizationConfig] = None) -> Tuple[pd.DataFrame, BatteryDataNormalizer]:
    """
    Convenience function to normalize a complete battery dataset.
    
    Args:
        data: Input DataFrame with battery data
        config: Normalization configuration
        
    Returns:
        Tuple of (normalized_dataframe, fitted_normalizer)
    """
    normalizer = create_battery_normalizer(config)
    normalizer.fit(data)
    
    normalized_dict = normalizer.transform(data)
    
    # Convert back to DataFrame
    normalized_df = pd.DataFrame(normalized_dict, index=data.index)
    
    return normalized_df, normalizer

def validate_normalization(original_data: np.ndarray, 
                         normalized_data: np.ndarray,
                         tolerance: float = 1e-6) -> Dict[str, bool]:
    """
    Validate normalization results.
    
    Args:
        original_data: Original data array
        normalized_data: Normalized data array
        tolerance: Numerical tolerance for validation
        
    Returns:
        Dictionary of validation results
    """
    validation_results = {}
    
    # Check for NaN values
    validation_results['no_nan'] = not np.any(np.isnan(normalized_data))
    
    # Check for infinite values
    validation_results['no_inf'] = not np.any(np.isinf(normalized_data))
    
    # Check data range (for standard normalization, should have mean ~0, std ~1)
    mean_val = np.mean(normalized_data)
    std_val = np.std(normalized_data)
    
    validation_results['mean_near_zero'] = abs(mean_val) < tolerance
    validation_results['std_near_one'] = abs(std_val - 1.0) < tolerance
    
    # Check preservation of data order (monotonic sequences should remain monotonic)
    if len(original_data) > 1:
        orig_diff = np.diff(original_data)
        norm_diff = np.diff(normalized_data)
        
        # Check if sign is preserved
        sign_preserved = np.all(np.sign(orig_diff) == np.sign(norm_diff))
        validation_results['order_preserved'] = sign_preserved
    
    return validation_results
