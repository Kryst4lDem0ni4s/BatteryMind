"""
BatteryMind - Feature Extractor

Advanced feature extraction pipeline for battery sensor data with temporal
feature engineering, statistical analysis, and domain-specific transformations.

Features:
- Time-series feature extraction with sliding windows
- Statistical and frequency domain features
- Battery-specific domain features
- Multi-modal sensor fusion features
- Automated feature selection and ranking
- Real-time feature extraction capabilities

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import warnings
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
import pywt
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FeatureExtractionConfig:
    """
    Configuration for feature extraction pipeline.
    
    Attributes:
        # Window parameters
        window_size (int): Size of sliding window for temporal features
        overlap_ratio (float): Overlap ratio between consecutive windows
        min_window_size (int): Minimum window size for feature extraction
        
        # Feature categories
        enable_statistical_features (bool): Extract statistical features
        enable_frequency_features (bool): Extract frequency domain features
        enable_temporal_features (bool): Extract temporal features
        enable_domain_features (bool): Extract battery-specific domain features
        enable_wavelet_features (bool): Extract wavelet transform features
        
        # Statistical features
        statistical_features (List[str]): List of statistical features to extract
        percentiles (List[float]): Percentiles to calculate
        
        # Frequency features
        max_frequency (float): Maximum frequency for FFT analysis
        frequency_bands (Dict[str, Tuple[float, float]]): Frequency bands for analysis
        
        # Temporal features
        lag_features (List[int]): Lag values for temporal features
        difference_orders (List[int]): Orders for difference features
        
        # Domain features
        battery_chemistry (str): Battery chemistry type
        nominal_capacity (float): Nominal battery capacity
        nominal_voltage (float): Nominal battery voltage
        
        # Feature selection
        enable_feature_selection (bool): Enable automatic feature selection
        max_features (int): Maximum number of features to select
        selection_method (str): Feature selection method
        
        # Preprocessing
        normalize_features (bool): Normalize extracted features
        handle_missing_values (bool): Handle missing values in features
        missing_value_strategy (str): Strategy for handling missing values
    """
    # Window parameters
    window_size: int = 100
    overlap_ratio: float = 0.5
    min_window_size: int = 10
    
    # Feature categories
    enable_statistical_features: bool = True
    enable_frequency_features: bool = True
    enable_temporal_features: bool = True
    enable_domain_features: bool = True
    enable_wavelet_features: bool = True
    
    # Statistical features
    statistical_features: List[str] = field(default_factory=lambda: [
        'mean', 'std', 'var', 'min', 'max', 'median', 'skew', 'kurtosis',
        'range', 'iqr', 'mad', 'rms', 'peak_to_peak'
    ])
    percentiles: List[float] = field(default_factory=lambda: [10, 25, 75, 90])
    
    # Frequency features
    max_frequency: float = 10.0  # Hz
    frequency_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'low': (0.0, 1.0),
        'medium': (1.0, 5.0),
        'high': (5.0, 10.0)
    })
    
    # Temporal features
    lag_features: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    difference_orders: List[int] = field(default_factory=lambda: [1, 2])
    
    # Domain features
    battery_chemistry: str = "LiFePO4"
    nominal_capacity: float = 100.0  # Ah
    nominal_voltage: float = 3.7     # V
    
    # Feature selection
    enable_feature_selection: bool = True
    max_features: int = 100
    selection_method: str = "mutual_info"  # "f_score", "mutual_info", "pca"
    
    # Preprocessing
    normalize_features: bool = True
    handle_missing_values: bool = True
    missing_value_strategy: str = "interpolate"  # "interpolate", "forward_fill", "drop"

class BaseFeatureExtractor(ABC):
    """Base class for feature extractors."""
    
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
        self.feature_names = []
        
    @abstractmethod
    def extract(self, data: np.ndarray) -> np.ndarray:
        """Extract features from input data."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        pass

class StatisticalFeatureExtractor(BaseFeatureExtractor):
    """Extract statistical features from time series data."""
    
    def __init__(self, config: FeatureExtractionConfig):
        super().__init__(config)
        self._setup_feature_names()
    
    def _setup_feature_names(self):
        """Setup feature names for statistical features."""
        self.feature_names = []
        
        # Basic statistical features
        for feature in self.config.statistical_features:
            self.feature_names.append(f"stat_{feature}")
        
        # Percentile features
        for percentile in self.config.percentiles:
            self.feature_names.append(f"percentile_{percentile}")
    
    def extract(self, data: np.ndarray) -> np.ndarray:
        """Extract statistical features from data."""
        if len(data) == 0:
            return np.full(len(self.feature_names), np.nan)
        
        features = []
        
        # Basic statistical features
        for feature_name in self.config.statistical_features:
            if feature_name == 'mean':
                features.append(np.mean(data))
            elif feature_name == 'std':
                features.append(np.std(data))
            elif feature_name == 'var':
                features.append(np.var(data))
            elif feature_name == 'min':
                features.append(np.min(data))
            elif feature_name == 'max':
                features.append(np.max(data))
            elif feature_name == 'median':
                features.append(np.median(data))
            elif feature_name == 'skew':
                features.append(stats.skew(data))
            elif feature_name == 'kurtosis':
                features.append(stats.kurtosis(data))
            elif feature_name == 'range':
                features.append(np.ptp(data))
            elif feature_name == 'iqr':
                features.append(np.percentile(data, 75) - np.percentile(data, 25))
            elif feature_name == 'mad':
                features.append(np.median(np.abs(data - np.median(data))))
            elif feature_name == 'rms':
                features.append(np.sqrt(np.mean(data**2)))
            elif feature_name == 'peak_to_peak':
                features.append(np.ptp(data))
        
        # Percentile features
        for percentile in self.config.percentiles:
            features.append(np.percentile(data, percentile))
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Get statistical feature names."""
        return self.feature_names.copy()

class FrequencyFeatureExtractor(BaseFeatureExtractor):
    """Extract frequency domain features using FFT and spectral analysis."""
    
    def __init__(self, config: FeatureExtractionConfig, sampling_rate: float = 1.0):
        super().__init__(config)
        self.sampling_rate = sampling_rate
        self._setup_feature_names()
    
    def _setup_feature_names(self):
        """Setup feature names for frequency features."""
        self.feature_names = []
        
        # Spectral features
        spectral_features = [
            'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
            'spectral_flatness', 'spectral_entropy', 'dominant_frequency',
            'peak_frequency_power'
        ]
        
        for feature in spectral_features:
            self.feature_names.append(f"freq_{feature}")
        
        # Band power features
        for band_name in self.config.frequency_bands.keys():
            self.feature_names.append(f"band_power_{band_name}")
            self.feature_names.append(f"band_power_ratio_{band_name}")
    
    def extract(self, data: np.ndarray) -> np.ndarray:
        """Extract frequency domain features."""
        if len(data) < 4:  # Need minimum data for FFT
            return np.full(len(self.feature_names), np.nan)
        
        features = []
        
        # Compute FFT
        fft_data = fft(data)
        freqs = fftfreq(len(data), 1/self.sampling_rate)
        
        # Only use positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        magnitude_spectrum = np.abs(fft_data[:len(fft_data)//2])
        power_spectrum = magnitude_spectrum**2
        
        # Normalize power spectrum
        total_power = np.sum(power_spectrum)
        if total_power > 0:
            normalized_power = power_spectrum / total_power
        else:
            normalized_power = power_spectrum
        
        # Spectral centroid
        if total_power > 0:
            spectral_centroid = np.sum(positive_freqs * normalized_power)
        else:
            spectral_centroid = 0
        features.append(spectral_centroid)
        
        # Spectral bandwidth
        if total_power > 0:
            spectral_bandwidth = np.sqrt(np.sum(((positive_freqs - spectral_centroid)**2) * normalized_power))
        else:
            spectral_bandwidth = 0
        features.append(spectral_bandwidth)
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumulative_power = np.cumsum(normalized_power)
        rolloff_idx = np.where(cumulative_power >= 0.85)[0]
        if len(rolloff_idx) > 0:
            spectral_rolloff = positive_freqs[rolloff_idx[0]]
        else:
            spectral_rolloff = positive_freqs[-1] if len(positive_freqs) > 0 else 0
        features.append(spectral_rolloff)
        
        # Spectral flatness (geometric mean / arithmetic mean)
        if np.all(magnitude_spectrum > 0):
            geometric_mean = stats.gmean(magnitude_spectrum)
            arithmetic_mean = np.mean(magnitude_spectrum)
            spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        else:
            spectral_flatness = 0
        features.append(spectral_flatness)
        
        # Spectral entropy
        if total_power > 0:
            spectral_entropy = -np.sum(normalized_power * np.log2(normalized_power + 1e-10))
        else:
            spectral_entropy = 0
        features.append(spectral_entropy)
        
        # Dominant frequency
        if len(magnitude_spectrum) > 0:
            dominant_freq_idx = np.argmax(magnitude_spectrum)
            dominant_frequency = positive_freqs[dominant_freq_idx]
            peak_frequency_power = magnitude_spectrum[dominant_freq_idx]
        else:
            dominant_frequency = 0
            peak_frequency_power = 0
        features.append(dominant_frequency)
        features.append(peak_frequency_power)
        
        # Band power features
        for band_name, (low_freq, high_freq) in self.config.frequency_bands.items():
            band_mask = (positive_freqs >= low_freq) & (positive_freqs <= high_freq)
            band_power = np.sum(power_spectrum[band_mask])
            band_power_ratio = band_power / total_power if total_power > 0 else 0
            
            features.append(band_power)
            features.append(band_power_ratio)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Get frequency feature names."""
        return self.feature_names.copy()

class TemporalFeatureExtractor(BaseFeatureExtractor):
    """Extract temporal features including lags, differences, and trends."""
    
    def __init__(self, config: FeatureExtractionConfig):
        super().__init__(config)
        self._setup_feature_names()
    
    def _setup_feature_names(self):
        """Setup feature names for temporal features."""
        self.feature_names = []
        
        # Lag features
        for lag in self.config.lag_features:
            self.feature_names.append(f"lag_{lag}")
        
        # Difference features
        for order in self.config.difference_orders:
            self.feature_names.extend([
                f"diff_{order}_mean", f"diff_{order}_std", f"diff_{order}_max"
            ])
        
        # Trend features
        trend_features = [
            'linear_trend_slope', 'linear_trend_intercept', 'linear_trend_r2',
            'monotonic_trend', 'turning_points', 'autocorrelation_lag1'
        ]
        
        for feature in trend_features:
            self.feature_names.append(f"temporal_{feature}")
    
    def extract(self, data: np.ndarray) -> np.ndarray:
        """Extract temporal features."""
        if len(data) == 0:
            return np.full(len(self.feature_names), np.nan)
        
        features = []
        
        # Lag features
        for lag in self.config.lag_features:
            if lag < len(data):
                lag_value = data[-lag-1] if lag < len(data) else np.nan
            else:
                lag_value = np.nan
            features.append(lag_value)
        
        # Difference features
        for order in self.config.difference_orders:
            if len(data) > order:
                diff_data = np.diff(data, n=order)
                features.extend([
                    np.mean(diff_data),
                    np.std(diff_data),
                    np.max(np.abs(diff_data))
                ])
            else:
                features.extend([np.nan, np.nan, np.nan])
        
        # Trend features
        if len(data) >= 3:
            # Linear trend
            x = np.arange(len(data))
            slope, intercept, r_value, _, _ = stats.linregress(x, data)
            features.extend([slope, intercept, r_value**2])
            
            # Monotonic trend
            monotonic_increasing = np.all(np.diff(data) >= 0)
            monotonic_decreasing = np.all(np.diff(data) <= 0)
            monotonic_trend = 1 if monotonic_increasing else (-1 if monotonic_decreasing else 0)
            features.append(monotonic_trend)
            
            # Turning points
            diff_data = np.diff(data)
            sign_changes = np.diff(np.sign(diff_data))
            turning_points = np.sum(np.abs(sign_changes) > 0)
            features.append(turning_points)
            
            # Autocorrelation at lag 1
            if len(data) > 1:
                autocorr = np.corrcoef(data[:-1], data[1:])[0, 1]
                autocorr = autocorr if not np.isnan(autocorr) else 0
            else:
                autocorr = 0
            features.append(autocorr)
        else:
            features.extend([np.nan] * 6)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Get temporal feature names."""
        return self.feature_names.copy()

class DomainFeatureExtractor(BaseFeatureExtractor):
    """Extract battery-specific domain features."""
    
    def __init__(self, config: FeatureExtractionConfig):
        super().__init__(config)
        self._setup_feature_names()
    
    def _setup_feature_names(self):
        """Setup feature names for domain features."""
        self.feature_names = [
            'soc_change_rate', 'voltage_efficiency', 'current_consistency',
            'temperature_stability', 'power_factor', 'energy_throughput',
            'charge_discharge_ratio', 'voltage_current_correlation',
            'thermal_gradient', 'capacity_utilization'
        ]
    
    def extract(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract domain-specific features from multi-modal battery data.
        
        Args:
            data_dict: Dictionary containing different sensor data
                      Expected keys: 'voltage', 'current', 'temperature', 'soc'
        """
        features = []
        
        # Get data arrays
        voltage = data_dict.get('voltage', np.array([]))
        current = data_dict.get('current', np.array([]))
        temperature = data_dict.get('temperature', np.array([]))
        soc = data_dict.get('soc', np.array([]))
        
        # SoC change rate
        if len(soc) > 1:
            soc_change_rate = np.mean(np.abs(np.diff(soc)))
        else:
            soc_change_rate = 0
        features.append(soc_change_rate)
        
        # Voltage efficiency (voltage stability under load)
        if len(voltage) > 0 and len(current) > 0:
            # Calculate voltage drop under different current levels
            if np.std(current) > 0:
                voltage_efficiency = 1 - (np.std(voltage) / np.mean(voltage))
            else:
                voltage_efficiency = 1
        else:
            voltage_efficiency = 0
        features.append(voltage_efficiency)
        
        # Current consistency
        if len(current) > 0:
            current_consistency = 1 - (np.std(current) / (np.mean(np.abs(current)) + 1e-6))
        else:
            current_consistency = 0
        features.append(current_consistency)
        
        # Temperature stability
        if len(temperature) > 0:
            temp_stability = 1 / (1 + np.std(temperature))
        else:
            temp_stability = 0
        features.append(temp_stability)
        
        # Power factor
        if len(voltage) > 0 and len(current) > 0:
            power = voltage * current
            power_factor = np.mean(power) / (np.mean(voltage) * np.mean(np.abs(current)) + 1e-6)
        else:
            power_factor = 0
        features.append(power_factor)
        
        # Energy throughput
        if len(voltage) > 0 and len(current) > 0:
            energy_throughput = np.sum(np.abs(voltage * current))
        else:
            energy_throughput = 0
        features.append(energy_throughput)
        
        # Charge/discharge ratio
        if len(current) > 0:
            charge_current = current[current > 0]
            discharge_current = current[current < 0]
            
            if len(charge_current) > 0 and len(discharge_current) > 0:
                charge_discharge_ratio = np.mean(charge_current) / np.mean(np.abs(discharge_current))
            else:
                charge_discharge_ratio = 1
        else:
            charge_discharge_ratio = 1
        features.append(charge_discharge_ratio)
        
        # Voltage-current correlation
        if len(voltage) > 1 and len(current) > 1:
            correlation = np.corrcoef(voltage, current)[0, 1]
            correlation = correlation if not np.isnan(correlation) else 0
        else:
            correlation = 0
        features.append(correlation)
        
        # Thermal gradient
        if len(temperature) > 1:
            thermal_gradient = np.mean(np.abs(np.diff(temperature)))
        else:
            thermal_gradient = 0
        features.append(thermal_gradient)
        
        # Capacity utilization
        if len(soc) > 0:
            capacity_utilization = (np.max(soc) - np.min(soc))
        else:
            capacity_utilization = 0
        features.append(capacity_utilization)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Get domain feature names."""
        return self.feature_names.copy()

class WaveletFeatureExtractor(BaseFeatureExtractor):
    """Extract wavelet transform features for multi-resolution analysis."""
    
    def __init__(self, config: FeatureExtractionConfig, wavelet: str = 'db4', levels: int = 4):
        super().__init__(config)
        self.wavelet = wavelet
        self.levels = levels
        self._setup_feature_names()
    
    def _setup_feature_names(self):
        """Setup feature names for wavelet features."""
        self.feature_names = []
        
        # Approximation coefficients features
        for level in range(1, self.levels + 1):
            self.feature_names.extend([
                f"wavelet_approx_L{level}_energy",
                f"wavelet_approx_L{level}_mean",
                f"wavelet_approx_L{level}_std"
            ])
        
        # Detail coefficients features
        for level in range(1, self.levels + 1):
            self.feature_names.extend([
                f"wavelet_detail_L{level}_energy",
                f"wavelet_detail_L{level}_mean",
                f"wavelet_detail_L{level}_std"
            ])
    
    def extract(self, data: np.ndarray) -> np.ndarray:
        """Extract wavelet features."""
        if len(data) < 2**self.levels:
            return np.full(len(self.feature_names), np.nan)
        
        features = []
        
        try:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(data, self.wavelet, level=self.levels)
            
            # Extract features from approximation coefficients
            for level in range(1, self.levels + 1):
                if level <= len(coeffs) - 1:
                    approx_coeffs = coeffs[-(level + 1)]
                    
                    # Energy
                    energy = np.sum(approx_coeffs**2)
                    features.append(energy)
                    
                    # Mean and std
                    features.append(np.mean(approx_coeffs))
                    features.append(np.std(approx_coeffs))
                else:
                    features.extend([0, 0, 0])
            
            # Extract features from detail coefficients
            for level in range(1, self.levels + 1):
                if level <= len(coeffs) - 1:
                    detail_coeffs = coeffs[-level]
                    
                    # Energy
                    energy = np.sum(detail_coeffs**2)
                    features.append(energy)
                    
                    # Mean and std
                    features.append(np.mean(detail_coeffs))
                    features.append(np.std(detail_coeffs))
                else:
                    features.extend([0, 0, 0])
                    
        except Exception as e:
            logger.warning(f"Wavelet decomposition failed: {e}")
            features = [np.nan] * len(self.feature_names)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Get wavelet feature names."""
        return self.feature_names.copy()

class BatteryFeatureExtractor:
    """
    Main feature extraction pipeline for battery sensor data.
    """
    
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
        self.extractors = {}
        self.feature_names = []
        self.scaler = None
        self.feature_selector = None
        
        # Initialize feature extractors
        self._initialize_extractors()
        
        logger.info("BatteryFeatureExtractor initialized")
    
    def _initialize_extractors(self):
        """Initialize feature extractors based on configuration."""
        if self.config.enable_statistical_features:
            self.extractors['statistical'] = StatisticalFeatureExtractor(self.config)
        
        if self.config.enable_frequency_features:
            self.extractors['frequency'] = FrequencyFeatureExtractor(self.config)
        
        if self.config.enable_temporal_features:
            self.extractors['temporal'] = TemporalFeatureExtractor(self.config)
        
        if self.config.enable_domain_features:
            self.extractors['domain'] = DomainFeatureExtractor(self.config)
        
        if self.config.enable_wavelet_features:
            self.extractors['wavelet'] = WaveletFeatureExtractor(self.config)
        
        # Build feature names
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Build comprehensive feature names list."""
        self.feature_names = []
        
        for extractor_name, extractor in self.extractors.items():
            if extractor_name != 'domain':  # Domain extractor has different interface
                names = extractor.get_feature_names()
                self.feature_names.extend(names)
        
        # Add domain feature names if enabled
        if 'domain' in self.extractors:
            domain_names = self.extractors['domain'].get_feature_names()
            self.feature_names.extend(domain_names)
    
    def extract_features(self, data: Union[np.ndarray, Dict[str, np.ndarray]], 
                        use_sliding_window: bool = True) -> np.ndarray:
        """
        Extract features from battery sensor data.
        
        Args:
            data: Input data (single array or dict of arrays for multi-modal)
            use_sliding_window: Whether to use sliding window approach
            
        Returns:
            np.ndarray: Extracted features
        """
        if isinstance(data, dict):
            return self._extract_multimodal_features(data, use_sliding_window)
        else:
            return self._extract_unimodal_features(data, use_sliding_window)
    
    def _extract_unimodal_features(self, data: np.ndarray, 
                                  use_sliding_window: bool = True) -> np.ndarray:
        """Extract features from single time series."""
        if use_sliding_window and len(data) >= self.config.window_size:
            return self._extract_windowed_features(data)
        else:
            return self._extract_single_window_features(data)
    
    def _extract_multimodal_features(self, data_dict: Dict[str, np.ndarray],
                                   use_sliding_window: bool = True) -> np.ndarray:
        """Extract features from multi-modal sensor data."""
        all_features = []
        
        # Extract features from each sensor modality
        for sensor_name, sensor_data in data_dict.items():
            if len(sensor_data) > 0:
                if use_sliding_window and len(sensor_data) >= self.config.window_size:
                    sensor_features = self._extract_windowed_features(sensor_data)
                else:
                    sensor_features = self._extract_single_window_features(sensor_data)
                
                all_features.append(sensor_features)
        
        # Extract domain features
        if 'domain' in self.extractors:
            domain_features = self.extractors['domain'].extract(data_dict)
            all_features.append(domain_features)
        
        # Concatenate all features
        if all_features:
            return np.concatenate(all_features)
        else:
            return np.array([])
    
    def _extract_windowed_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features using sliding window approach."""
        window_size = self.config.window_size
        step_size = int(window_size * (1 - self.config.overlap_ratio))
        
        window_features = []
        
        for start_idx in range(0, len(data) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_data = data[start_idx:end_idx]
            
            features = self._extract_single_window_features(window_data)
            window_features.append(features)
        
        if window_features:
            # Aggregate features across windows (mean)
            return np.mean(window_features, axis=0)
        else:
            return self._extract_single_window_features(data)
    
    def _extract_single_window_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features from a single window of data."""
        features = []
        
        # Extract features using each extractor
        for extractor_name, extractor in self.extractors.items():
            if extractor_name != 'domain':  # Domain extractor handled separately
                try:
                    extractor_features = extractor.extract(data)
                    features.append(extractor_features)
                except Exception as e:
                    logger.warning(f"Feature extraction failed for {extractor_name}: {e}")
                    # Add NaN features for failed extractor
                    nan_features = np.full(len(extractor.get_feature_names()), np.nan)
                    features.append(nan_features)
        
        # Concatenate all features
        if features:
            return np.concatenate(features)
        else:
            return np.array([])
    
    def fit_preprocessing(self, features: np.ndarray, target: Optional[np.ndarray] = None):
        """
        Fit preprocessing components (scaler, feature selector).
        
        Args:
            features: Training features
            target: Target values for supervised feature selection
        """
        # Handle missing values
        if self.config.handle_missing_values:
            features = self._handle_missing_values(features)
        
        # Fit scaler
        if self.config.normalize_features:
            self.scaler = StandardScaler()
            self.scaler.fit(features)
        
        # Fit feature selector
        if self.config.enable_feature_selection and target is not None:
            self._fit_feature_selector(features, target)
        
        logger.info("Preprocessing components fitted successfully")
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted preprocessing components.
        
        Args:
            features: Features to transform
            
        Returns:
            np.ndarray: Transformed features
        """
        # Handle missing values
        if self.config.handle_missing_values:
            features = self._handle_missing_values(features)
        
        # Apply scaling
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Apply feature selection
        if self.feature_selector is not None:
            features = self.feature_selector.transform(features)
        
        return features
    
    def _handle_missing_values(self, features: np.ndarray) -> np.ndarray:
        """Handle missing values in features."""
        if self.config.missing_value_strategy == "interpolate":
            # Linear interpolation for missing values
            for i in range(features.shape[1]):
                column = features[:, i]
                if np.any(np.isnan(column)):
                    valid_indices = ~np.isnan(column)
                    if np.any(valid_indices):
                        features[:, i] = np.interp(
                            np.arange(len(column)),
                            np.arange(len(column))[valid_indices],
                            column[valid_indices]
                        )
                    else:
                        features[:, i] = 0  # Fill with zeros if all NaN
        
        elif self.config.missing_value_strategy == "forward_fill":
            # Forward fill missing values
            for i in range(features.shape[1]):
                column = features[:, i]
                mask = np.isnan(column)
                if np.any(mask):
                    # Forward fill
                    features[:, i] = pd.Series(column).fillna(method='ffill').fillna(0).values
        
        elif self.config.missing_value_strategy == "drop":
            # Remove rows with any missing values
            features = features[~np.any(np.isnan(features), axis=1)]
        
        return features
    
    def _fit_feature_selector(self, features: np.ndarray, target: np.ndarray):
        """Fit feature selector based on configuration."""
        if self.config.selection_method == "f_score":
            self.feature_selector = SelectKBest(
                score_func=f_regression,
                k=min(self.config.max_features, features.shape[1])
            )
        elif self.config.selection_method == "mutual_info":
            self.feature_selector = SelectKBest(
                score_func=mutual_info_regression,
                k=min(self.config.max_features, features.shape[1])
            )
        elif self.config.selection_method == "pca":
            self.feature_selector = PCA(
                n_components=min(self.config.max_features, features.shape[1])
            )
        
        if self.feature_selector is not None:
            self.feature_selector.fit(features, target)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features."""
        if self.feature_selector is not None and hasattr(self.feature_selector, 'get_support'):
            # Return selected feature names
            selected_mask = self.feature_selector.get_support()
            return [name for name, selected in zip(self.feature_names, selected_mask) if selected]
        else:
            return self.feature_names.copy()
    
    def save_extractor(self, filepath: str):
        """Save the feature extractor to file."""
        extractor_data = {
            'config': self.config,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector
        }
        
        joblib.dump(extractor_data, filepath)
        logger.info(f"Feature extractor saved to {filepath}")
    
    @classmethod
    def load_extractor(cls, filepath: str) -> 'BatteryFeatureExtractor':
        """Load feature extractor from file."""
        extractor_data = joblib.load(filepath)
        
        extractor = cls(extractor_data['config'])
        extractor.feature_names = extractor_data['feature_names']
        extractor.scaler = extractor_data['scaler']
        extractor.feature_selector = extractor_data['feature_selector']
        
        logger.info(f"Feature extractor loaded from {filepath}")
        return extractor

# Factory functions
def create_feature_extractor(config: Optional[FeatureExtractionConfig] = None) -> BatteryFeatureExtractor:
    """
    Factory function to create a battery feature extractor.
    
    Args:
        config: Feature extraction configuration
        
    Returns:
        BatteryFeatureExtractor: Configured feature extractor
    """
    if config is None:
        config = FeatureExtractionConfig()
    
    return BatteryFeatureExtractor(config)

def extract_features_from_dataframe(df: pd.DataFrame, 
                                   config: Optional[FeatureExtractionConfig] = None,
                                   target_column: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract features from a pandas DataFrame.
    
    Args:
        df: Input DataFrame with battery sensor data
        config: Feature extraction configuration
        target_column: Name of target column (if any)
        
    Returns:
        Tuple of (features, target, feature_names)
    """
    if config is None:
        config = FeatureExtractionConfig()
    
    extractor = BatteryFeatureExtractor(config)
    
    # Separate features and target
    if target_column and target_column in df.columns:
        target = df[target_column].values
        feature_df = df.drop(columns=[target_column])
    else:
        target = None
        feature_df = df
    
    # Extract features
    all_features = []
    for idx, row in feature_df.iterrows():
        # Convert row to dictionary
        data_dict = row.to_dict()
        
        # Remove non-numeric columns
        data_dict = {k: v for k, v in data_dict.items() if isinstance(v, (int, float, np.number))}
        
        # Convert to arrays (assuming single values for now)
        data_arrays = {k: np.array([v]) for k, v in data_dict.items()}
        
        features = extractor.extract_features(data_arrays, use_sliding_window=False)
        all_features.append(features)
    
    features_array = np.array(all_features)
    
    # Fit preprocessing if target is available
    if target is not None:
        extractor.fit_preprocessing(features_array, target)
        features_array = extractor.transform_features(features_array)
    
    feature_names = extractor.get_feature_names()
    
    return features_array, target, feature_names
