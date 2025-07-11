"""
BatteryMind - Time Series Utilities

Comprehensive time-series processing utilities for degradation forecasting
with advanced seasonality detection, trend analysis, and change point detection.

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SeasonalityResult:
    """Result of seasonality detection."""
    is_seasonal: bool
    dominant_period: Optional[int]
    seasonal_strength: float
    detected_periods: List[int]
    seasonal_components: Dict[int, np.ndarray]
    confidence_score: float

@dataclass
class TrendResult:
    """Result of trend analysis."""
    trend_type: str  # 'increasing', 'decreasing', 'stable', 'non_linear'
    trend_strength: float
    trend_component: np.ndarray
    slope: float
    r_squared: float
    change_points: List[int]

@dataclass
class StationarityResult:
    """Result of stationarity testing."""
    is_stationary: bool
    adf_statistic: float
    adf_p_value: float
    kpss_statistic: float
    kpss_p_value: float
    differencing_order: int

class SeasonalityDetector:
    """
    Advanced seasonality detection using multiple methods.
    """
    
    def __init__(self, min_period: int = 2, max_period: int = None):
        self.min_period = min_period
        self.max_period = max_period
    
    def detect_seasonality(self, data: np.ndarray, 
                          timestamps: Optional[pd.DatetimeIndex] = None) -> SeasonalityResult:
        """
        Detect seasonality in time series data using multiple methods.
        
        Args:
            data (np.ndarray): Time series data
            timestamps (pd.DatetimeIndex, optional): Timestamps for the data
            
        Returns:
            SeasonalityResult: Seasonality detection results
        """
        if self.max_period is None:
            self.max_period = min(len(data) // 3, 365)
        
        # Method 1: Autocorrelation-based detection
        autocorr_periods = self._autocorrelation_seasonality(data)
        
        # Method 2: FFT-based detection
        fft_periods = self._fft_seasonality(data)
        
        # Method 3: Known seasonal patterns (if timestamps available)
        known_periods = []
        if timestamps is not None:
            known_periods = self._detect_known_patterns(data, timestamps)
        
        # Combine results
        all_periods = autocorr_periods + fft_periods + known_periods
        period_counts = {}
        for period in all_periods:
            period_counts[period] = period_counts.get(period, 0) + 1
        
        # Find dominant period
        if period_counts:
            dominant_period = max(period_counts.keys(), key=period_counts.get)
            confidence_score = period_counts[dominant_period] / 3.0  # Max 3 methods
        else:
            dominant_period = None
            confidence_score = 0.0
        
        # Extract seasonal components
        seasonal_components = {}
        detected_periods = list(period_counts.keys())
        
        for period in detected_periods:
            seasonal_comp = self._extract_seasonal_component(data, period)
            seasonal_components[period] = seasonal_comp
        
        # Calculate seasonal strength
        seasonal_strength = self._calculate_seasonal_strength(data, dominant_period)
        
        # Determine if seasonal
        is_seasonal = (seasonal_strength > 0.3 and 
                      confidence_score > 0.5 and 
                      dominant_period is not None)
        
        return SeasonalityResult(
            is_seasonal=is_seasonal,
            dominant_period=dominant_period,
            seasonal_strength=seasonal_strength,
            detected_periods=detected_periods,
            seasonal_components=seasonal_components,
            confidence_score=confidence_score
        )
    
    def _autocorrelation_seasonality(self, data: np.ndarray) -> List[int]:
        """Detect seasonality using autocorrelation."""
        # Calculate autocorrelation
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find peaks in autocorrelation
        peaks, _ = signal.find_peaks(autocorr[1:], height=0.3, distance=self.min_period)
        peaks += 1  # Adjust for offset
        
        # Filter by period constraints
        valid_periods = [p for p in peaks if self.min_period <= p <= self.max_period]
        
        return valid_periods[:3]  # Return top 3
    
    def _fft_seasonality(self, data: np.ndarray) -> List[int]:
        """Detect seasonality using FFT."""
        # Compute FFT
        fft_values = fft(data - np.mean(data))
        frequencies = fftfreq(len(data))
        
        # Get magnitude spectrum
        magnitude = np.abs(fft_values)
        
        # Find peaks in frequency domain
        peaks, _ = signal.find_peaks(magnitude[1:len(magnitude)//2], height=np.max(magnitude)*0.1)
        peaks += 1  # Adjust for offset
        
        # Convert frequencies to periods
        periods = []
        for peak in peaks:
            if frequencies[peak] > 0:
                period = int(1 / frequencies[peak])
                if self.min_period <= period <= self.max_period:
                    periods.append(period)
        
        return periods[:3]  # Return top 3
    
    def _detect_known_patterns(self, data: np.ndarray, 
                              timestamps: pd.DatetimeIndex) -> List[int]:
        """Detect known seasonal patterns from timestamps."""
        patterns = []
        
        # Daily pattern (24 hours)
        if len(data) >= 48:  # At least 2 days
            daily_strength = self._test_period_strength(data, timestamps, 24)
            if daily_strength > 0.3:
                patterns.append(24)
        
        # Weekly pattern (7 days)
        if len(data) >= 14:  # At least 2 weeks
            weekly_strength = self._test_period_strength(data, timestamps, 168)  # 7*24
            if weekly_strength > 0.3:
                patterns.append(168)
        
        # Monthly pattern (30 days)
        if len(data) >= 60:  # At least 2 months
            monthly_strength = self._test_period_strength(data, timestamps, 720)  # 30*24
            if monthly_strength > 0.3:
                patterns.append(720)
        
        return patterns
    
    def _test_period_strength(self, data: np.ndarray, 
                             timestamps: pd.DatetimeIndex, period: int) -> float:
        """Test strength of a specific period."""
        if len(data) < period * 2:
            return 0.0
        
        # Group data by period
        period_groups = []
        for i in range(period):
            indices = list(range(i, len(data), period))
            if len(indices) >= 2:
                period_data = data[indices]
                period_groups.append(period_data)
        
        if len(period_groups) < 2:
            return 0.0
        
        # Calculate correlation between periods
        min_length = min(len(group) for group in period_groups)
        truncated_groups = [group[:min_length] for group in period_groups]
        
        correlations = []
        for i in range(len(truncated_groups)):
            for j in range(i + 1, len(truncated_groups)):
                corr, _ = stats.pearsonr(truncated_groups[i], truncated_groups[j])
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _extract_seasonal_component(self, data: np.ndarray, period: int) -> np.ndarray:
        """Extract seasonal component for a given period."""
        if period >= len(data):
            return np.zeros_like(data)
        
        # Create seasonal pattern by averaging over periods
        seasonal_pattern = np.zeros(period)
        counts = np.zeros(period)
        
        for i, value in enumerate(data):
            seasonal_idx = i % period
            seasonal_pattern[seasonal_idx] += value
            counts[seasonal_idx] += 1
        
        # Avoid division by zero
        counts[counts == 0] = 1
        seasonal_pattern /= counts
        
        # Repeat pattern to match data length
        seasonal_component = np.tile(seasonal_pattern, len(data) // period + 1)[:len(data)]
        
        return seasonal_component
    
    def _calculate_seasonal_strength(self, data: np.ndarray, period: Optional[int]) -> float:
        """Calculate seasonal strength."""
        if period is None or period >= len(data):
            return 0.0
        
        seasonal_component = self._extract_seasonal_component(data, period)
        residual = data - seasonal_component
        
        # Seasonal strength as ratio of variances
        seasonal_var = np.var(seasonal_component)
        residual_var = np.var(residual)
        
        if seasonal_var + residual_var == 0:
            return 0.0
        
        return seasonal_var / (seasonal_var + residual_var)

class TrendExtractor:
    """
    Advanced trend extraction and analysis.
    """
    
    def extract_trend(self, data: np.ndarray, method: str = 'linear') -> TrendResult:
        """
        Extract trend from time series data.
        
        Args:
            data (np.ndarray): Time series data
            method (str): Trend extraction method
            
        Returns:
            TrendResult: Trend analysis results
        """
        x = np.arange(len(data))
        
        if method == 'linear':
            trend_component, slope, r_squared = self._linear_trend(x, data)
            trend_type = self._classify_linear_trend(slope, r_squared)
        elif method == 'polynomial':
            trend_component, slope, r_squared = self._polynomial_trend(x, data)
            trend_type = self._classify_polynomial_trend(trend_component)
        else:
            raise ValueError(f"Unknown trend method: {method}")
        
        # Detect change points
        change_points = self._detect_trend_changes(data)
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(data, trend_component)
        
        return TrendResult(
            trend_type=trend_type,
            trend_strength=trend_strength,
            trend_component=trend_component,
            slope=slope,
            r_squared=r_squared,
            change_points=change_points
        )
    
    def _linear_trend(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Extract linear trend."""
        reg = LinearRegression()
        reg.fit(x.reshape(-1, 1), y)
        
        trend = reg.predict(x.reshape(-1, 1))
        slope = reg.coef_[0]
        r_squared = reg.score(x.reshape(-1, 1), y)
        
        return trend, slope, r_squared
    
    def _polynomial_trend(self, x: np.ndarray, y: np.ndarray, 
                         degree: int = 2) -> Tuple[np.ndarray, float, float]:
        """Extract polynomial trend."""
        coeffs = np.polyfit(x, y, degree)
        trend = np.polyval(coeffs, x)
        
        # Calculate slope at midpoint
        derivative_coeffs = np.polyder(coeffs)
        midpoint = len(x) // 2
        slope = np.polyval(derivative_coeffs, midpoint)
        
        # Calculate R-squared
        ss_res = np.sum((y - trend) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return trend, slope, r_squared
    
    def _classify_linear_trend(self, slope: float, r_squared: float) -> str:
        """Classify linear trend type."""
        if r_squared < 0.1:
            return 'stable'
        elif slope > 0.001:
            return 'increasing'
        elif slope < -0.001:
            return 'decreasing'
        else:
            return 'stable'
    
    def _classify_polynomial_trend(self, trend: np.ndarray) -> str:
        """Classify polynomial trend type."""
        # Calculate first and second derivatives
        first_diff = np.diff(trend)
        second_diff = np.diff(first_diff)
        
        # Check for monotonicity
        if np.all(first_diff >= 0):
            return 'increasing'
        elif np.all(first_diff <= 0):
            return 'decreasing'
        else:
            return 'non_linear'
    
    def _detect_trend_changes(self, data: np.ndarray) -> List[int]:
        """Detect trend change points."""
        if len(data) < 10:
            return []
        
        # Use PELT (Pruned Exact Linear Time) algorithm approximation
        change_points = []
        window_size = max(5, len(data) // 20)
        
        for i in range(window_size, len(data) - window_size):
            # Calculate trend before and after point
            before_data = data[max(0, i - window_size):i]
            after_data = data[i:min(len(data), i + window_size)]
            
            if len(before_data) >= 3 and len(after_data) >= 3:
                before_trend = np.polyfit(range(len(before_data)), before_data, 1)[0]
                after_trend = np.polyfit(range(len(after_data)), after_data, 1)[0]
                
                # Significant change in trend
                if abs(before_trend - after_trend) > np.std(data) * 0.1:
                    change_points.append(i)
        
        return change_points
    
    def _calculate_trend_strength(self, data: np.ndarray, trend: np.ndarray) -> float:
        """Calculate trend strength."""
        detrended = data - trend
        
        trend_var = np.var(trend)
        residual_var = np.var(detrended)
        
        if trend_var + residual_var == 0:
            return 0.0
        
        return trend_var / (trend_var + residual_var)

class ChangePointDetector:
    """
    Change point detection for time series.
    """
    
    def detect_change_points(self, data: np.ndarray, method: str = 'cusum') -> List[int]:
        """
        Detect change points in time series.
        
        Args:
            data (np.ndarray): Time series data
            method (str): Detection method
            
        Returns:
            List[int]: Indices of detected change points
        """
        if method == 'cusum':
            return self._cusum_detection(data)
        elif method == 'variance':
            return self._variance_change_detection(data)
        else:
            raise ValueError(f"Unknown change point method: {method}")
    
    def _cusum_detection(self, data: np.ndarray, threshold: float = None) -> List[int]:
        """CUSUM-based change point detection."""
        if threshold is None:
            threshold = 3 * np.std(data)
        
        # Calculate CUSUM
        mean_data = np.mean(data)
        cusum_pos = np.zeros(len(data))
        cusum_neg = np.zeros(len(data))
        
        for i in range(1, len(data)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + data[i] - mean_data - threshold/2)
            cusum_neg[i] = min(0, cusum_neg[i-1] + data[i] - mean_data + threshold/2)
        
        # Find change points
        change_points = []
        for i in range(1, len(data)):
            if cusum_pos[i] > threshold or cusum_neg[i] < -threshold:
                change_points.append(i)
                # Reset CUSUM after detection
                cusum_pos[i:] = 0
                cusum_neg[i:] = 0
        
        return change_points
    
    def _variance_change_detection(self, data: np.ndarray, 
                                  window_size: int = None) -> List[int]:
        """Variance-based change point detection."""
        if window_size is None:
            window_size = max(5, len(data) // 20)
        
        change_points = []
        
        for i in range(window_size, len(data) - window_size):
            # Calculate variance before and after
            before_var = np.var(data[i-window_size:i])
            after_var = np.var(data[i:i+window_size])
            
            # Test for significant variance change
            if before_var > 0 and after_var > 0:
                f_stat = max(before_var, after_var) / min(before_var, after_var)
                if f_stat > 2.0:  # Threshold for significant change
                    change_points.append(i)
        
        return change_points

class ForecastValidator:
    """
    Validation utilities for forecasting models.
    """
    
    def validate_forecasts(self, forecasts: np.ndarray, actuals: np.ndarray,
                          confidence_intervals: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Validate forecasting performance.
        
        Args:
            forecasts (np.ndarray): Forecast values
            actuals (np.ndarray): Actual values
            confidence_intervals (np.ndarray, optional): Confidence intervals
            
        Returns:
            Dict[str, float]: Validation metrics
        """
        metrics = {}
        
        # Basic accuracy metrics
        metrics['mse'] = np.mean((forecasts - actuals) ** 2)
        metrics['mae'] = np.mean(np.abs(forecasts - actuals))
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Percentage errors
        non_zero_mask = actuals != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((actuals[non_zero_mask] - forecasts[non_zero_mask]) / 
                                actuals[non_zero_mask])) * 100
            metrics['mape'] = mape
        
        # Symmetric MAPE
        denominator = (np.abs(actuals) + np.abs(forecasts)) / 2
        non_zero_denom = denominator != 0
        if np.any(non_zero_denom):
            smape = np.mean(np.abs(actuals[non_zero_denom] - forecasts[non_zero_denom]) / 
                          denominator[non_zero_denom]) * 100
            metrics['smape'] = smape
        
        # Directional accuracy
        if len(forecasts) > 1:
            actual_direction = np.sign(np.diff(actuals))
            forecast_direction = np.sign(np.diff(forecasts))
            metrics['directional_accuracy'] = np.mean(actual_direction == forecast_direction)
        
        # Confidence interval metrics
        if confidence_intervals is not None:
            lower_bounds = confidence_intervals[:, 0]
            upper_bounds = confidence_intervals[:, 1]
            
            # Coverage probability
            coverage = np.mean((actuals >= lower_bounds) & (actuals <= upper_bounds))
            metrics['coverage_probability'] = coverage
            
            # Average interval width
            metrics['avg_interval_width'] = np.mean(upper_bounds - lower_bounds)
        
        return metrics

class TimeSeriesAugmentation:
    """
    Data augmentation techniques for time series.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def jitter(self, data: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add random jitter to time series."""
        noise = np.random.normal(0, noise_level * np.std(data), data.shape)
        return data + noise
    
    def scaling(self, data: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Scale time series by random factor."""
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return data * scale_factor
    
    def time_warping(self, data: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Apply time warping to time series."""
        n_points = len(data)
        warp_steps = np.random.normal(1.0, sigma, n_points)
        warp_steps = np.cumsum(warp_steps)
        warp_steps = (warp_steps - warp_steps[0]) / (warp_steps[-1] - warp_steps[0]) * (n_points - 1)
        
        # Interpolate
        return np.interp(np.arange(n_points), warp_steps, data)
    
    def window_slicing(self, data: np.ndarray, slice_ratio: float = 0.8) -> np.ndarray:
        """Extract random window from time series."""
        slice_length = int(len(data) * slice_ratio)
        start_idx = np.random.randint(0, len(data) - slice_length + 1)
        return data[start_idx:start_idx + slice_length]

class TimeSeriesProcessor:
    """
    Main time-series processing class combining all utilities.
    """
    
    def __init__(self):
        self.seasonality_detector = SeasonalityDetector()
        self.trend_extractor = TrendExtractor()
        self.change_point_detector = ChangePointDetector()
        self.forecast_validator = ForecastValidator()
        self.augmentation = TimeSeriesAugmentation()
    
    def analyze_series(self, data: np.ndarray, 
                      timestamps: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of time-series data.
        
        Args:
            data (np.ndarray): Time-series data
            timestamps (pd.DatetimeIndex, optional): Timestamps
            
        Returns:
            Dict[str, Any]: Complete analysis results
        """
        results = {
            'data_length': len(data),
            'data_range': (np.min(data), np.max(data)),
            'data_mean': np.mean(data),
            'data_std': np.std(data)
        }
        
        # Seasonality analysis
        seasonality_result = self.seasonality_detector.detect_seasonality(data, timestamps)
        results['seasonality'] = seasonality_result.__dict__
        
        # Trend analysis
        trend_result = self.trend_extractor.extract_trend(data)
        results['trend'] = trend_result.__dict__
        
        # Change point detection
        change_points = self.change_point_detector.detect_change_points(data)
        results['change_points'] = change_points
        
        # Stationarity testing
        stationarity_result = self._test_stationarity(data)
        results['stationarity'] = stationarity_result.__dict__
        
        return results
    
    def _test_stationarity(self, data: np.ndarray) -> StationarityResult:
        """Test stationarity using ADF and KPSS tests."""
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
            
            # Augmented Dickey-Fuller test
            adf_result = adfuller(data)
            adf_statistic = adf_result[0]
            adf_p_value = adf_result[1]
            
            # KPSS test
            kpss_result = kpss(data)
            kpss_statistic = kpss_result[0]
            kpss_p_value = kpss_result[1]
            
            # Determine stationarity
            is_stationary = (adf_p_value < 0.05) and (kpss_p_value > 0.05)
            
            # Determine differencing order
            differencing_order = 0
            test_data = data.copy()
            
            while not is_stationary and differencing_order < 3:
                test_data = np.diff(test_data)
                if len(test_data) > 10:
                    adf_result = adfuller(test_data)
                    kpss_result = kpss(test_data)
                    is_stationary = (adf_result[1] < 0.05) and (kpss_result[1] > 0.05)
                    differencing_order += 1
                else:
                    break
            
        except ImportError:
            logger.warning("statsmodels not available, using simplified stationarity test")
            # Simplified test using variance of differences
            diff_data = np.diff(data)
            adf_statistic = -np.var(diff_data) / np.var(data)
            adf_p_value = 0.05 if abs(adf_statistic) > 1 else 0.5
            kpss_statistic = np.var(data)
            kpss_p_value = 0.1
            is_stationary = adf_p_value < 0.05
            differencing_order = 1 if not is_stationary else 0
        
        return StationarityResult(
            is_stationary=is_stationary,
            adf_statistic=adf_statistic,
            adf_p_value=adf_p_value,
            kpss_statistic=kpss_statistic,
            kpss_p_value=kpss_p_value,
            differencing_order=differencing_order
        )
    
    def prepare_forecasting_data(self, data: np.ndarray, 
                                sequence_length: int,
                                forecast_horizon: int,
                                overlap_ratio: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Prepare data for forecasting model training.
        
        Args:
            data (np.ndarray): Time series data
            sequence_length (int): Length of input sequences
            forecast_horizon (int): Number of steps to forecast
            overlap_ratio (float): Overlap between sequences
            
        Returns:
            Dict[str, np.ndarray]: Prepared sequences and targets
        """
        step_size = int(sequence_length * (1 - overlap_ratio))
        
        sequences = []
        targets = []
        
        for i in range(0, len(data) - sequence_length - forecast_horizon + 1, step_size):
            seq = data[i:i + sequence_length]
            target = data[i + sequence_length:i + sequence_length + forecast_horizon]
            
            sequences.append(seq)
            targets.append(target)
        
        return {
            'sequences': np.array(sequences),
            'targets': np.array(targets)
        }
    
    def validate_data(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Validate time series data quality.
        
        Args:
            data (np.ndarray): Time series data
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            'length': len(data),
            'missing_values': np.sum(np.isnan(data)),
            'infinite_values': np.sum(np.isinf(data)),
            'zero_values': np.sum(data == 0),
            'negative_values': np.sum(data < 0),
            'outliers': 0,
            'quality_score': 1.0
        }
        
        # Detect outliers using IQR method
        if len(data) > 4:
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = np.sum((data < lower_bound) | (data > upper_bound))
            validation_results['outliers'] = outliers
        
        # Calculate quality score
        quality_penalties = [
            validation_results['missing_values'] / len(data),
            validation_results['infinite_values'] / len(data),
            validation_results['outliers'] / len(data)
        ]
        
        quality_score = 1.0 - sum(quality_penalties)
        validation_results['quality_score'] = max(0.0, quality_score)
        
        return validation_results

# Factory functions
def create_time_series_processor() -> TimeSeriesProcessor:
    """Create a TimeSeriesProcessor instance."""
    return TimeSeriesProcessor()

def analyze_battery_degradation_series(data: np.ndarray, 
                                     timestamps: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
    """
    Specialized analysis for battery degradation time series.
    
    Args:
        data (np.ndarray): Battery degradation data
        timestamps (pd.DatetimeIndex, optional): Timestamps
        
    Returns:
        Dict[str, Any]: Analysis results specific to battery degradation
    """
    processor = TimeSeriesProcessor()
    
    # General analysis
    results = processor.analyze_series(data, timestamps)
    
    # Battery-specific analysis
    results['degradation_rate'] = np.mean(np.diff(data)) if len(data) > 1 else 0.0
    results['acceleration'] = np.mean(np.diff(np.diff(data))) if len(data) > 2 else 0.0
    
    # Estimate remaining useful life (simplified)
    if results['degradation_rate'] > 0:
        current_health = data[-1] if len(data) > 0 else 1.0
        end_of_life_threshold = 0.7
        remaining_life = (current_health - end_of_life_threshold) / results['degradation_rate']
        results['estimated_rul'] = max(0, remaining_life)
    else:
        results['estimated_rul'] = float('inf')
    
    return results
