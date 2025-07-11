"""
BatteryMind - Battery Health Preprocessing

Advanced preprocessing pipeline for battery health prediction with comprehensive
feature engineering, normalization, and data augmentation capabilities.

Features:
- Multi-modal sensor data preprocessing
- Advanced feature engineering for time-series data
- Physics-informed feature extraction
- Data normalization and standardization
- Comprehensive data augmentation techniques
- Real-time preprocessing for streaming data
- Integration with transformer model requirements

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import warnings
from pathlib import Path
import pickle
import json

# Scientific computing imports
from scipy import signal, stats
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    PowerTransformer, QuantileTransformer
)
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_regression
import pywt  # PyWavelets for wavelet transforms

# Time series processing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import tsfel  # Time Series Feature Extraction Library

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BatteryPreprocessingConfig:
    """
    Configuration for battery data preprocessing.
    
    Attributes:
        # Normalization configuration
        normalization_method (str): Normalization method ('standard', 'minmax', 'robust', 'quantile')
        feature_scaling_range (Tuple[float, float]): Range for MinMax scaling
        
        # Feature engineering configuration
        enable_feature_engineering (bool): Enable advanced feature engineering
        window_sizes (List[int]): Window sizes for rolling statistics
        lag_features (List[int]): Lag values for lag features
        
        # Frequency domain features
        enable_frequency_features (bool): Enable frequency domain analysis
        fft_components (int): Number of FFT components to extract
        wavelet_type (str): Wavelet type for wavelet transform
        
        # Physics-informed features
        enable_physics_features (bool): Enable physics-based feature extraction
        temperature_reference (float): Reference temperature for Arrhenius features
        
        # Data augmentation
        enable_augmentation (bool): Enable data augmentation
        noise_level (float): Gaussian noise level for augmentation
        time_warping_sigma (float): Time warping parameter
        
        # Outlier handling
        outlier_method (str): Outlier detection method ('zscore', 'iqr', 'isolation')
        outlier_threshold (float): Threshold for outlier detection
        
        # Missing value handling
        interpolation_method (str): Interpolation method ('linear', 'cubic', 'nearest')
        max_gap_size (int): Maximum gap size for interpolation
        
        # Sequence processing
        sequence_padding (str): Padding method ('zero', 'replicate', 'reflect')
        sequence_truncation (str): Truncation method ('random', 'beginning', 'end')
    """
    # Normalization configuration
    normalization_method: str = "standard"
    feature_scaling_range: Tuple[float, float] = (0.0, 1.0)
    
    # Feature engineering configuration
    enable_feature_engineering: bool = True
    window_sizes: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    lag_features: List[int] = field(default_factory=lambda: [1, 5, 10, 24])
    
    # Frequency domain features
    enable_frequency_features: bool = True
    fft_components: int = 10
    wavelet_type: str = "db4"
    
    # Physics-informed features
    enable_physics_features: bool = True
    temperature_reference: float = 25.0  # °C
    
    # Data augmentation
    enable_augmentation: bool = False
    noise_level: float = 0.01
    time_warping_sigma: float = 0.2
    
    # Outlier handling
    outlier_method: str = "zscore"
    outlier_threshold: float = 3.0
    
    # Missing value handling
    interpolation_method: str = "linear"
    max_gap_size: int = 10
    
    # Sequence processing
    sequence_padding: str = "zero"
    sequence_truncation: str = "random"

class BatteryFeatureExtractor:
    """
    Advanced feature extraction for battery time-series data.
    """
    
    def __init__(self, config: BatteryPreprocessingConfig):
        self.config = config
        
    def extract_statistical_features(self, data: np.ndarray, window_sizes: List[int]) -> Dict[str, np.ndarray]:
        """
        Extract statistical features using rolling windows.
        
        Args:
            data (np.ndarray): Input time series data
            window_sizes (List[int]): List of window sizes
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of extracted features
        """
        features = {}
        
        for window_size in window_sizes:
            # Rolling statistics
            features[f'rolling_mean_{window_size}'] = self._rolling_stat(data, window_size, np.mean)
            features[f'rolling_std_{window_size}'] = self._rolling_stat(data, window_size, np.std)
            features[f'rolling_min_{window_size}'] = self._rolling_stat(data, window_size, np.min)
            features[f'rolling_max_{window_size}'] = self._rolling_stat(data, window_size, np.max)
            features[f'rolling_median_{window_size}'] = self._rolling_stat(data, window_size, np.median)
            
            # Rolling percentiles
            features[f'rolling_q25_{window_size}'] = self._rolling_stat(data, window_size, lambda x: np.percentile(x, 25))
            features[f'rolling_q75_{window_size}'] = self._rolling_stat(data, window_size, lambda x: np.percentile(x, 75))
            
            # Rolling range and IQR
            features[f'rolling_range_{window_size}'] = features[f'rolling_max_{window_size}'] - features[f'rolling_min_{window_size}']
            features[f'rolling_iqr_{window_size}'] = features[f'rolling_q75_{window_size}'] - features[f'rolling_q25_{window_size}']
            
            # Rolling skewness and kurtosis
            features[f'rolling_skew_{window_size}'] = self._rolling_stat(data, window_size, stats.skew)
            features[f'rolling_kurtosis_{window_size}'] = self._rolling_stat(data, window_size, stats.kurtosis)
        
        return features
    
    def extract_lag_features(self, data: np.ndarray, lag_values: List[int]) -> Dict[str, np.ndarray]:
        """
        Extract lag features from time series data.
        
        Args:
            data (np.ndarray): Input time series data
            lag_values (List[int]): List of lag values
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of lag features
        """
        features = {}
        
        for lag in lag_values:
            # Simple lag
            features[f'lag_{lag}'] = np.roll(data, lag)
            features[f'lag_{lag}'][:lag] = data[0]  # Fill initial values
            
            # Lag difference
            features[f'lag_diff_{lag}'] = data - features[f'lag_{lag}']
            
            # Lag ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                features[f'lag_ratio_{lag}'] = np.where(features[f'lag_{lag}'] != 0, 
                                                       data / features[f'lag_{lag}'], 1.0)
        
        return features
    
    def extract_frequency_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract frequency domain features using FFT and wavelets.
        
        Args:
            data (np.ndarray): Input time series data
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of frequency features
        """
        features = {}
        
        # FFT features
        if self.config.enable_frequency_features:
            fft_values = fft(data)
            fft_magnitude = np.abs(fft_values)
            fft_phase = np.angle(fft_values)
            
            # Extract top FFT components
            n_components = min(self.config.fft_components, len(fft_magnitude) // 2)
            for i in range(n_components):
                features[f'fft_magnitude_{i}'] = np.full(len(data), fft_magnitude[i])
                features[f'fft_phase_{i}'] = np.full(len(data), fft_phase[i])
            
            # Spectral features
            features['spectral_centroid'] = np.full(len(data), self._spectral_centroid(fft_magnitude))
            features['spectral_bandwidth'] = np.full(len(data), self._spectral_bandwidth(fft_magnitude))
            features['spectral_rolloff'] = np.full(len(data), self._spectral_rolloff(fft_magnitude))
            
            # Wavelet features
            try:
                coeffs = pywt.wavedec(data, self.config.wavelet_type, level=4)
                for i, coeff in enumerate(coeffs):
                    features[f'wavelet_energy_{i}'] = np.full(len(data), np.sum(coeff**2))
                    features[f'wavelet_std_{i}'] = np.full(len(data), np.std(coeff))
            except Exception as e:
                logger.warning(f"Wavelet feature extraction failed: {e}")
        
        return features
    
    def extract_physics_features(self, voltage: np.ndarray, current: np.ndarray, 
                                temperature: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract physics-informed features specific to battery behavior.
        
        Args:
            voltage (np.ndarray): Voltage measurements
            current (np.ndarray): Current measurements
            temperature (np.ndarray): Temperature measurements
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of physics-based features
        """
        features = {}
        
        if self.config.enable_physics_features:
            # Power calculation
            features['power'] = voltage * current
            
            # Energy calculation (cumulative)
            features['energy_cumulative'] = np.cumsum(features['power'])
            
            # Resistance estimation (Ohm's law approximation)
            with np.errstate(divide='ignore', invalid='ignore'):
                features['resistance_estimate'] = np.where(current != 0, voltage / current, np.inf)
                features['resistance_estimate'] = np.clip(features['resistance_estimate'], 0, 10)  # Reasonable bounds
            
            # Temperature-based features (Arrhenius relationship)
            temp_kelvin = temperature + 273.15
            temp_ref_kelvin = self.config.temperature_reference + 273.15
            features['arrhenius_factor'] = np.exp(-1 / temp_kelvin + 1 / temp_ref_kelvin)
            
            # Voltage derivatives
            features['voltage_derivative'] = np.gradient(voltage)
            features['voltage_second_derivative'] = np.gradient(features['voltage_derivative'])
            
            # Current derivatives
            features['current_derivative'] = np.gradient(current)
            features['current_second_derivative'] = np.gradient(features['current_derivative'])
            
            # Temperature derivatives
            features['temperature_derivative'] = np.gradient(temperature)
            
            # Coulomb counting approximation
            features['coulomb_count'] = np.cumsum(current)
            
            # Voltage efficiency
            features['voltage_efficiency'] = voltage / np.max(voltage)
            
            # Power efficiency
            max_power = np.max(np.abs(features['power']))
            if max_power > 0:
                features['power_efficiency'] = features['power'] / max_power
            else:
                features['power_efficiency'] = np.zeros_like(features['power'])
        
        return features
    
    def _rolling_stat(self, data: np.ndarray, window_size: int, stat_func: Callable) -> np.ndarray:
        """Apply rolling statistical function."""
        result = np.zeros_like(data)
        
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            window_data = data[start_idx:end_idx]
            
            try:
                result[i] = stat_func(window_data)
            except:
                result[i] = data[i]  # Fallback to current value
        
        return result
    
    def _spectral_centroid(self, magnitude_spectrum: np.ndarray) -> float:
        """Calculate spectral centroid."""
        freqs = np.arange(len(magnitude_spectrum))
        return np.sum(freqs * magnitude_spectrum) / np.sum(magnitude_spectrum)
    
    def _spectral_bandwidth(self, magnitude_spectrum: np.ndarray) -> float:
        """Calculate spectral bandwidth."""
        freqs = np.arange(len(magnitude_spectrum))
        centroid = self._spectral_centroid(magnitude_spectrum)
        return np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude_spectrum) / np.sum(magnitude_spectrum))
    
    def _spectral_rolloff(self, magnitude_spectrum: np.ndarray, rolloff_threshold: float = 0.85) -> float:
        """Calculate spectral rolloff."""
        cumsum = np.cumsum(magnitude_spectrum)
        total_energy = cumsum[-1]
        rolloff_energy = rolloff_threshold * total_energy
        
        rolloff_idx = np.where(cumsum >= rolloff_energy)[0]
        return rolloff_idx[0] if len(rolloff_idx) > 0 else len(magnitude_spectrum) - 1

class BatteryDataAugmentation:
    """
    Data augmentation techniques for battery time-series data.
    """
    
    def __init__(self, config: BatteryPreprocessingConfig):
        self.config = config
        self.rng = np.random.RandomState(42)
    
    def add_gaussian_noise(self, data: np.ndarray, noise_level: Optional[float] = None) -> np.ndarray:
        """
        Add Gaussian noise to data.
        
        Args:
            data (np.ndarray): Input data
            noise_level (float, optional): Noise level (std of noise relative to data std)
            
        Returns:
            np.ndarray: Augmented data
        """
        if noise_level is None:
            noise_level = self.config.noise_level
        
        noise_std = noise_level * np.std(data)
        noise = self.rng.normal(0, noise_std, data.shape)
        
        return data + noise
    
    def time_warping(self, data: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """
        Apply time warping augmentation.
        
        Args:
            data (np.ndarray): Input time series data
            sigma (float, optional): Warping parameter
            
        Returns:
            np.ndarray: Time-warped data
        """
        if sigma is None:
            sigma = self.config.time_warping_sigma
        
        # Generate random warping curve
        n_points = len(data)
        warp_steps = self.rng.normal(1.0, sigma, n_points)
        warp_steps = np.cumsum(warp_steps)
        warp_steps = (warp_steps - warp_steps[0]) / (warp_steps[-1] - warp_steps[0]) * (n_points - 1)
        
        # Interpolate to get warped data
        interpolator = interp1d(np.arange(n_points), data, kind='linear', 
                               bounds_error=False, fill_value='extrapolate')
        
        return interpolator(warp_steps)
    
    def magnitude_warping(self, data: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """
        Apply magnitude warping augmentation.
        
        Args:
            data (np.ndarray): Input data
            sigma (float): Magnitude warping parameter
            
        Returns:
            np.ndarray: Magnitude-warped data
        """
        # Generate smooth random curve for magnitude warping
        n_points = len(data)
        knot_points = max(4, n_points // 20)
        
        # Create knot points
        knots = self.rng.normal(1.0, sigma, knot_points)
        knot_indices = np.linspace(0, n_points - 1, knot_points)
        
        # Interpolate to get smooth warping curve
        interpolator = interp1d(knot_indices, knots, kind='cubic', 
                               bounds_error=False, fill_value='extrapolate')
        warp_curve = interpolator(np.arange(n_points))
        
        return data * warp_curve

class BatteryPreprocessor:
    """
    Comprehensive preprocessing pipeline for battery health prediction.
    """
    
    def __init__(self, config: Optional[BatteryPreprocessingConfig] = None):
        self.config = config or BatteryPreprocessingConfig()
        self.feature_extractor = BatteryFeatureExtractor(self.config)
        self.data_augmentation = BatteryDataAugmentation(self.config)
        
        # Scalers for different normalization methods
        self.scalers = {}
        self.is_fitted = False
        
        logger.info(f"BatteryPreprocessor initialized with {self.config.normalization_method} normalization")
    
    def fit(self, data: pd.DataFrame) -> 'BatteryPreprocessor':
        """
        Fit preprocessing pipeline on training data.
        
        Args:
            data (pd.DataFrame): Training data
            
        Returns:
            BatteryPreprocessor: Fitted preprocessor
        """
        logger.info("Fitting preprocessing pipeline...")
        
        # Extract numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Fit scalers for each column
        for column in numeric_columns:
            if column in data.columns:
                column_data = data[column].values.reshape(-1, 1)
                
                # Create appropriate scaler
                if self.config.normalization_method == 'standard':
                    scaler = StandardScaler()
                elif self.config.normalization_method == 'minmax':
                    scaler = MinMaxScaler(feature_range=self.config.feature_scaling_range)
                elif self.config.normalization_method == 'robust':
                    scaler = RobustScaler()
                elif self.config.normalization_method == 'quantile':
                    scaler = QuantileTransformer(output_distribution='normal')
                else:
                    raise ValueError(f"Unknown normalization method: {self.config.normalization_method}")
                
                # Fit scaler
                scaler.fit(column_data)
                self.scalers[column] = scaler
        
        self.is_fitted = True
        logger.info(f"Preprocessing pipeline fitted on {len(numeric_columns)} features")
        
        return self
    
    def transform(self, data: pd.DataFrame, augment: bool = False) -> pd.DataFrame:
        """
        Transform data using fitted preprocessing pipeline.
        
        Args:
            data (pd.DataFrame): Input data
            augment (bool): Whether to apply data augmentation
            
        Returns:
            pd.DataFrame: Transformed data
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # Create copy of data
        transformed_data = data.copy()
        
        # Handle missing values
        transformed_data = self._handle_missing_values(transformed_data)
        
        # Remove outliers
        transformed_data = self._handle_outliers(transformed_data)
        
        # Feature engineering
        if self.config.enable_feature_engineering:
            transformed_data = self._engineer_features(transformed_data)
        
        # Normalize features
        transformed_data = self._normalize_features(transformed_data)
        
        # Data augmentation
        if augment and self.config.enable_augmentation:
            transformed_data = self._augment_data(transformed_data)
        
        return transformed_data
    
    def fit_transform(self, data: pd.DataFrame, augment: bool = False) -> pd.DataFrame:
        """
        Fit and transform data in one step.
        
        Args:
            data (pd.DataFrame): Input data
            augment (bool): Whether to apply data augmentation
            
        Returns:
            pd.DataFrame: Transformed data
        """
        return self.fit(data).transform(data, augment)
    
    def preprocess_single(self, data: Union[Dict, pd.DataFrame, np.ndarray], 
                         metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Preprocess single battery data sample for inference.
        
        Args:
            data: Single battery data sample
            metadata: Optional metadata
            
        Returns:
            Dict[str, Any]: Preprocessed data with features and metadata
        """
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, np.ndarray):
            # Assume standard feature order
            feature_names = ['voltage', 'current', 'temperature', 'state_of_charge',
                           'internal_resistance', 'capacity', 'cycle_count', 'age_days']
            df = pd.DataFrame(data.reshape(1, -1), columns=feature_names[:data.shape[-1]])
        else:
            df = data.copy()
        
        # Transform data
        if self.is_fitted:
            processed_df = self.transform(df, augment=False)
        else:
            logger.warning("Preprocessor not fitted. Using raw data.")
            processed_df = df
        
        # Extract features
        features = processed_df.select_dtypes(include=[np.number]).values
        
        return {
            'features': features,
            'metadata': metadata or {}
        }
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using configured method."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if data[column].isnull().any():
                if self.config.interpolation_method == 'linear':
                    data[column] = data[column].interpolate(method='linear')
                elif self.config.interpolation_method == 'cubic':
                    data[column] = data[column].interpolate(method='cubic')
                elif self.config.interpolation_method == 'nearest':
                    data[column] = data[column].fillna(method='ffill').fillna(method='bfill')
                
                # Fill remaining NaN values with median
                data[column] = data[column].fillna(data[column].median())
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using configured method."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if self.config.outlier_method == 'zscore':
                z_scores = np.abs(stats.zscore(data[column]))
                outlier_mask = z_scores > self.config.outlier_threshold
            elif self.config.outlier_method == 'iqr':
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))
            else:
                continue  # Skip outlier handling
            
            # Replace outliers with median
            if outlier_mask.any():
                data.loc[outlier_mask, column] = data[column].median()
        
        return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features."""
        # Basic feature columns
        voltage_col = 'voltage' if 'voltage' in data.columns else None
        current_col = 'current' if 'current' in data.columns else None
        temp_col = 'temperature' if 'temperature' in data.columns else None
        
        # Extract features for each numeric column
        for column in data.select_dtypes(include=[np.number]).columns:
            column_data = data[column].values
            
            # Statistical features
            if self.config.window_sizes:
                stat_features = self.feature_extractor.extract_statistical_features(
                    column_data, self.config.window_sizes
                )
                for feature_name, feature_values in stat_features.items():
                    data[f'{column}_{feature_name}'] = feature_values
            
            # Lag features
            if self.config.lag_features:
                lag_features = self.feature_extractor.extract_lag_features(
                    column_data, self.config.lag_features
                )
                for feature_name, feature_values in lag_features.items():
                    data[f'{column}_{feature_name}'] = feature_values
            
            # Frequency features
            if self.config.enable_frequency_features:
                freq_features = self.feature_extractor.extract_frequency_features(column_data)
                for feature_name, feature_values in freq_features.items():
                    data[f'{column}_{feature_name}'] = feature_values
        
        # Physics-informed features
        if (self.config.enable_physics_features and 
            voltage_col and current_col and temp_col):
            physics_features = self.feature_extractor.extract_physics_features(
                data[voltage_col].values,
                data[current_col].values,
                data[temp_col].values
            )
            for feature_name, feature_values in physics_features.items():
                data[feature_name] = feature_values
        
        return data
    
    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using fitted scalers."""
        for column in data.select_dtypes(include=[np.number]).columns:
            if column in self.scalers:
                column_data = data[column].values.reshape(-1, 1)
                normalized_data = self.scalers[column].transform(column_data)
                data[column] = normalized_data.flatten()
        
        return data
    
    def _augment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply data augmentation."""
        augmented_data = data.copy()
        
        for column in data.select_dtypes(include=[np.number]).columns:
            column_data = data[column].values
            
            # Apply random augmentation
            if np.random.random() < 0.5:  # 50% chance of augmentation
                if np.random.random() < 0.5:
                    # Gaussian noise
                    augmented_data[column] = self.data_augmentation.add_gaussian_noise(column_data)
                else:
                    # Magnitude warping
                    augmented_data[column] = self.data_augmentation.magnitude_warping(column_data)
        
        return augmented_data
    
    def save(self, file_path: str) -> None:
        """Save fitted preprocessor to file."""
        save_data = {
            'config': self.config,
            'scalers': self.scalers,
            'is_fitted': self.is_fitted
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Preprocessor saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'BatteryPreprocessor':
        """Load fitted preprocessor from file."""
        with open(file_path, 'rb') as f:
            save_data = pickle.load(f)
        
        preprocessor = cls(save_data['config'])
        preprocessor.scalers = save_data['scalers']
        preprocessor.is_fitted = save_data['is_fitted']
        
        logger.info(f"Preprocessor loaded from {file_path}")
        return preprocessor

# Factory functions
def create_battery_preprocessor(config: Optional[BatteryPreprocessingConfig] = None) -> BatteryPreprocessor:
    """
    Factory function to create a BatteryPreprocessor.
    
    Args:
        config (BatteryPreprocessingConfig, optional): Preprocessing configuration
        
    Returns:
        BatteryPreprocessor: Configured preprocessor instance
    """
    return BatteryPreprocessor(config)

def preprocess_battery_data(data: pd.DataFrame, 
                          config: Optional[BatteryPreprocessingConfig] = None,
                          fit_preprocessor: bool = True) -> Tuple[pd.DataFrame, BatteryPreprocessor]:
    """
    Convenience function to preprocess battery data.
    
    Args:
        data (pd.DataFrame): Input data
        config (BatteryPreprocessingConfig, optional): Preprocessing configuration
        fit_preprocessor (bool): Whether to fit the preprocessor
        
    Returns:
        Tuple[pd.DataFrame, BatteryPreprocessor]: Processed data and fitted preprocessor
    """
    preprocessor = BatteryPreprocessor(config)
    
    if fit_preprocessor:
        processed_data = preprocessor.fit_transform(data)
    else:
        processed_data = preprocessor.transform(data)
    
    return processed_data, preprocessor
