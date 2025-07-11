"""
BatteryMind - Battery Degradation Forecaster

Production-ready inference engine for battery degradation forecasting with
comprehensive uncertainty quantification, seasonal analysis, and integration
with battery health prediction systems.

Features:
- Multi-horizon degradation forecasting with uncertainty bounds
- Seasonal decomposition and trend analysis
- Real-time forecasting with streaming data support
- Integration with battery health predictor for comprehensive analysis
- Advanced forecasting metrics and validation
- Production-optimized inference with caching

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import time
import json
import pickle
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque
import hashlib

# Scientific computing imports
from scipy import stats, signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# AWS and production imports
import boto3
import redis
from prometheus_client import Counter, Histogram, Gauge

# Time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Local imports
from .model import DegradationForecaster, DegradationConfig
from .time_series_utils import TimeSeriesProcessor, ForecastValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
FORECAST_COUNTER = Counter('degradation_forecasts_total', 'Total degradation forecasts made')
FORECAST_LATENCY = Histogram('degradation_forecast_duration_seconds', 'Time spent on forecasting')
FORECAST_ACCURACY = Gauge('degradation_forecast_accuracy', 'Current forecasting accuracy')

@dataclass
class ForecastResult:
    """
    Comprehensive result structure for degradation forecasting.
    
    Attributes:
        battery_id (str): Unique battery identifier
        timestamp (float): Forecast timestamp
        forecast_horizon_hours (int): Forecast horizon in hours
        forecasts (Dict[str, np.ndarray]): Forecasted degradation metrics
        uncertainty_bounds (Dict[str, Dict]): Uncertainty bounds for forecasts
        seasonal_components (Dict[str, np.ndarray]): Seasonal decomposition results
        trend_analysis (Dict[str, Any]): Trend analysis results
        confidence_intervals (Dict[str, Dict]): Confidence intervals
        forecast_quality_score (float): Overall forecast quality assessment
        recommendations (List[str]): Actionable recommendations based on forecasts
        metadata (Dict[str, Any]): Additional forecast metadata
    """
    battery_id: str
    timestamp: float
    forecast_horizon_hours: int
    forecasts: Dict[str, np.ndarray] = field(default_factory=dict)
    uncertainty_bounds: Dict[str, Dict] = field(default_factory=dict)
    seasonal_components: Dict[str, np.ndarray] = field(default_factory=dict)
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    confidence_intervals: Dict[str, Dict] = field(default_factory=dict)
    forecast_quality_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'battery_id': self.battery_id,
            'timestamp': self.timestamp,
            'forecast_horizon_hours': self.forecast_horizon_hours,
            'forecasts': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                         for k, v in self.forecasts.items()},
            'uncertainty_bounds': self.uncertainty_bounds,
            'seasonal_components': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                  for k, v in self.seasonal_components.items()},
            'trend_analysis': self.trend_analysis,
            'confidence_intervals': self.confidence_intervals,
            'forecast_quality_score': self.forecast_quality_score,
            'recommendations': self.recommendations,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class ForecastMetrics:
    """
    Comprehensive metrics for forecast evaluation.
    """
    mse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    smape: float = 0.0
    mase: float = 0.0
    coverage_probability: float = 0.0
    prediction_interval_width: float = 0.0
    directional_accuracy: float = 0.0
    trend_accuracy: float = 0.0
    seasonal_accuracy: float = 0.0

@dataclass
class ForecastingConfig:
    """
    Configuration for degradation forecasting inference.
    
    Attributes:
        model_path (str): Path to trained forecasting model
        device (str): Inference device
        batch_size (int): Batch size for inference
        forecast_horizons (List[int]): Available forecast horizons in hours
        confidence_levels (List[float]): Confidence levels for prediction intervals
        enable_uncertainty (bool): Enable uncertainty quantification
        enable_seasonal_analysis (bool): Enable seasonal decomposition
        enable_trend_analysis (bool): Enable trend analysis
        cache_forecasts (bool): Enable forecast caching
        cache_ttl (int): Cache time-to-live in seconds
        max_sequence_length (int): Maximum input sequence length
        min_history_length (int): Minimum history required for forecasting
    """
    model_path: str = "./model_artifacts/best_forecasting_model.ckpt"
    device: str = "auto"
    batch_size: int = 16
    forecast_horizons: List[int] = field(default_factory=lambda: [24, 168, 720])  # 1 day, 1 week, 1 month
    confidence_levels: List[float] = field(default_factory=lambda: [0.8, 0.9, 0.95])
    enable_uncertainty: bool = True
    enable_seasonal_analysis: bool = True
    enable_trend_analysis: bool = True
    cache_forecasts: bool = True
    cache_ttl: int = 3600
    max_sequence_length: int = 1024
    min_history_length: int = 168  # 1 week minimum

class BatteryDegradationForecaster:
    """
    Production-ready battery degradation forecasting engine.
    
    Features:
    - Multi-horizon forecasting with uncertainty quantification
    - Seasonal decomposition and trend analysis
    - Real-time and batch forecasting modes
    - Comprehensive forecast validation and quality assessment
    - Integration with battery health prediction systems
    """
    
    def __init__(self, config: ForecastingConfig):
        self.config = config
        self.device = self._setup_device()
        self.time_series_processor = TimeSeriesProcessor()
        self.forecast_validator = ForecastValidator()
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Setup caching
        self.cache = self._setup_cache() if config.cache_forecasts else None
        
        # Performance optimization
        self._optimize_model()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Forecast history for validation
        self.forecast_history = deque(maxlen=1000)
        
        logger.info(f"BatteryDegradationForecaster initialized on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup inference device."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.config.device)
    
    def _load_model(self) -> DegradationForecaster:
        """Load trained forecasting model."""
        checkpoint = torch.load(self.config.model_path, map_location=self.device)
        
        # Extract model configuration
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            if hasattr(model_config, 'model'):
                model_config = DegradationConfig(**model_config.model)
            else:
                model_config = DegradationConfig()
        else:
            model_config = DegradationConfig()
        
        # Create and load model
        model = DegradationForecaster(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def _setup_cache(self) -> Optional[redis.Redis]:
        """Setup Redis cache for forecasts."""
        try:
            cache = redis.Redis(host='localhost', port=6379, db=1)
            cache.ping()
            return cache
        except Exception as e:
            logger.warning(f"Failed to setup cache: {e}")
            return None
    
    def _optimize_model(self) -> None:
        """Optimize model for inference."""
        # Enable inference mode optimizations
        torch.backends.cudnn.benchmark = True
        
        # Compile model if using PyTorch 2.0+
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled for optimized inference")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
    
    def forecast_degradation(self, battery_data: Union[pd.DataFrame, np.ndarray],
                           forecast_horizon: int = 168,
                           battery_metadata: Optional[Dict] = None,
                           include_uncertainty: bool = True) -> ForecastResult:
        """
        Generate degradation forecast for a single battery.
        
        Args:
            battery_data: Historical battery data
            forecast_horizon: Forecast horizon in hours
            battery_metadata: Additional battery metadata
            include_uncertainty: Whether to include uncertainty estimates
            
        Returns:
            ForecastResult: Comprehensive forecasting result
        """
        start_time = time.time()
        
        # Validate input data
        if not self._validate_input_data(battery_data):
            raise ValueError("Invalid input data for forecasting")
        
        # Generate cache key
        cache_key = self._generate_cache_key(battery_data, forecast_horizon, battery_metadata)
        
        # Check cache
        if self.cache:
            cached_result = self._get_cached_forecast(cache_key)
            if cached_result:
                return cached_result
        
        # Preprocess data
        processed_data = self._preprocess_data(battery_data, battery_metadata)
        
        # Generate time features
        time_features = self._extract_time_features(processed_data)
        
        # Make forecast
        with torch.no_grad():
            inputs = torch.tensor(processed_data['features'], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Adjust model forecast horizon if needed
            original_horizon = self.model.config.forecast_horizon
            if forecast_horizon != original_horizon:
                # For different horizons, we'll interpolate or extrapolate
                model_output = self.model(inputs, time_features, return_components=True)
                forecasts = self._adjust_forecast_horizon(
                    model_output['forecasts'], original_horizon, forecast_horizon
                )
            else:
                model_output = self.model(inputs, time_features, return_components=True)
                forecasts = model_output['forecasts']
        
        # Process model output
        result = self._process_forecast_output(
            forecasts, model_output, battery_data, battery_metadata,
            forecast_horizon, include_uncertainty
        )
        
        # Cache result
        if self.cache:
            self._cache_forecast(cache_key, result)
        
        # Update metrics
        forecast_time = time.time() - start_time
        FORECAST_COUNTER.inc()
        FORECAST_LATENCY.observe(forecast_time)
        
        # Store in history for validation
        self.forecast_history.append({
            'timestamp': time.time(),
            'battery_id': result.battery_id,
            'forecast_horizon': forecast_horizon,
            'result': result
        })
        
        return result
    
    def forecast_batch(self, battery_data_list: List[Union[pd.DataFrame, np.ndarray]],
                      forecast_horizon: int = 168,
                      battery_metadata_list: Optional[List[Dict]] = None) -> List[ForecastResult]:
        """
        Generate batch forecasts for multiple batteries.
        
        Args:
            battery_data_list: List of battery data
            forecast_horizon: Forecast horizon in hours
            battery_metadata_list: List of battery metadata
            
        Returns:
            List[ForecastResult]: List of forecasting results
        """
        if battery_metadata_list is None:
            battery_metadata_list = [None] * len(battery_data_list)
        
        results = []
        for i in range(0, len(battery_data_list), self.config.batch_size):
            batch_data = battery_data_list[i:i + self.config.batch_size]
            batch_metadata = battery_metadata_list[i:i + self.config.batch_size]
            
            batch_results = self._forecast_batch_internal(
                batch_data, forecast_horizon, batch_metadata
            )
            results.extend(batch_results)
        
        return results
    
    def _forecast_batch_internal(self, batch_data: List, forecast_horizon: int,
                               batch_metadata: List) -> List[ForecastResult]:
        """Internal batch forecasting implementation."""
        # Process batch data
        processed_batch = []
        for data, metadata in zip(batch_data, batch_metadata):
            if self._validate_input_data(data):
                processed = self._preprocess_data(data, metadata)
                processed_batch.append(processed)
        
        if not processed_batch:
            return []
        
        # Stack features
        features = torch.stack([
            torch.tensor(item['features'], dtype=torch.float32)
            for item in processed_batch
        ]).to(self.device)
        
        # Generate batch forecasts
        with torch.no_grad():
            # Create time features for batch
            batch_time_features = {}
            for key in processed_batch[0].get('time_features', {}):
                batch_time_features[key] = torch.stack([
                    torch.tensor(item['time_features'][key])
                    for item in processed_batch
                ])
            
            model_output = self.model(features, batch_time_features, return_components=True)
            forecasts = model_output['forecasts']
        
        # Process results
        results = []
        for i, (data, metadata) in enumerate(zip(batch_data, batch_metadata)):
            if i < len(forecasts):
                individual_output = {
                    'forecasts': forecasts[i:i+1],
                    'decomposition': model_output.get('decomposition')
                }
                
                result = self._process_forecast_output(
                    individual_output['forecasts'], individual_output,
                    data, metadata, forecast_horizon, True
                )
                results.append(result)
        
        return results
    
    def _validate_input_data(self, data: Union[pd.DataFrame, np.ndarray]) -> bool:
        """Validate input data for forecasting."""
        if isinstance(data, pd.DataFrame):
            if len(data) < self.config.min_history_length:
                logger.warning(f"Insufficient data length: {len(data)} < {self.config.min_history_length}")
                return False
            return True
        elif isinstance(data, np.ndarray):
            if data.shape[0] < self.config.min_history_length:
                logger.warning(f"Insufficient data length: {data.shape[0]} < {self.config.min_history_length}")
                return False
            return True
        else:
            logger.error(f"Unsupported data type: {type(data)}")
            return False
    
    def _preprocess_data(self, data: Union[pd.DataFrame, np.ndarray],
                        metadata: Optional[Dict]) -> Dict[str, Any]:
        """Preprocess data for forecasting."""
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            # Assume standard feature order
            feature_names = [
                'voltage', 'current', 'temperature', 'state_of_charge',
                'internal_resistance', 'capacity', 'cycle_count', 'age_days'
            ]
            df = pd.DataFrame(data, columns=feature_names[:data.shape[1]])
        else:
            df = data.copy()
        
        # Ensure minimum sequence length
        if len(df) > self.config.max_sequence_length:
            df = df.tail(self.config.max_sequence_length)
        
        # Extract features
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        features = df[numeric_columns].values.astype(np.float32)
        
        # Handle missing values
        if np.isnan(features).any():
            features = self.time_series_processor.interpolate_missing_values(features)
        
        return {
            'features': features,
            'metadata': metadata or {},
            'original_data': df
        }
    
    def _extract_time_features(self, processed_data: Dict) -> Dict[str, torch.Tensor]:
        """Extract time-based features for seasonal modeling."""
        time_features = {}
        
        # If original data has timestamp column
        df = processed_data['original_data']
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            
            # Extract seasonal features
            time_features['hour_of_day'] = torch.tensor(timestamps.dt.hour.values, dtype=torch.float32)
            time_features['day_of_week'] = torch.tensor(timestamps.dt.dayofweek.values, dtype=torch.float32)
            time_features['day_of_month'] = torch.tensor(timestamps.dt.day.values, dtype=torch.float32)
            time_features['month_of_year'] = torch.tensor(timestamps.dt.month.values, dtype=torch.float32)
            
            # Seasonal encodings for different periods
            for period in [24, 168, 720]:  # Daily, weekly, monthly
                if len(timestamps) >= period:
                    seasonal_values = np.arange(len(timestamps)) % period
                    time_features[f'seasonal_{period}'] = torch.tensor(seasonal_values, dtype=torch.float32)
        
        return time_features
    
    def _adjust_forecast_horizon(self, forecasts: torch.Tensor, 
                               original_horizon: int, target_horizon: int) -> torch.Tensor:
        """Adjust forecast horizon through interpolation or extrapolation."""
        if original_horizon == target_horizon:
            return forecasts
        
        batch_size, _, feature_dim = forecasts.shape
        
        # Create time indices
        original_times = np.linspace(0, 1, original_horizon)
        target_times = np.linspace(0, 1, target_horizon)
        
        # Interpolate/extrapolate for each batch and feature
        adjusted_forecasts = np.zeros((batch_size, target_horizon, feature_dim))
        
        for b in range(batch_size):
            for f in range(feature_dim):
                # Use linear interpolation/extrapolation
                interp_func = interp1d(
                    original_times, forecasts[b, :, f].cpu().numpy(),
                    kind='linear', fill_value='extrapolate'
                )
                adjusted_forecasts[b, :, f] = interp_func(target_times)
        
        return torch.tensor(adjusted_forecasts, dtype=torch.float32)
    
    def _process_forecast_output(self, forecasts: torch.Tensor, model_output: Dict,
                               original_data: Union[pd.DataFrame, np.ndarray],
                               metadata: Optional[Dict], forecast_horizon: int,
                               include_uncertainty: bool) -> ForecastResult:
        """Process model output into comprehensive forecast result."""
        forecasts_np = forecasts.squeeze().cpu().numpy()
        
        # Extract battery ID
        battery_id = metadata.get('battery_id', 'unknown') if metadata else 'unknown'
        
        # Process forecasts by degradation type
        degradation_types = [
            'capacity_fade_rate', 'resistance_increase_rate', 'thermal_degradation',
            'cycle_efficiency_decline', 'calendar_aging_rate', 'overall_health_decline'
        ]
        
        forecast_dict = {}
        for i, deg_type in enumerate(degradation_types):
            if i < forecasts_np.shape[-1]:
                forecast_dict[deg_type] = forecasts_np[:, i] if forecasts_np.ndim > 1 else forecasts_np[i:i+1]
        
        # Calculate uncertainty bounds if available
        uncertainty_bounds = {}
        confidence_intervals = {}
        
        if include_uncertainty and 'std' in model_output:
            std_np = model_output['std'].squeeze().cpu().numpy()
            
            for i, deg_type in enumerate(degradation_types):
                if i < std_np.shape[-1]:
                    std_values = std_np[:, i] if std_np.ndim > 1 else std_np[i:i+1]
                    
                    uncertainty_bounds[deg_type] = {
                        'std': std_values,
                        'variance': std_values ** 2
                    }
                    
                    # Calculate confidence intervals
                    forecast_values = forecast_dict[deg_type]
                    confidence_intervals[deg_type] = {}
                    
                    for conf_level in self.config.confidence_levels:
                        z_score = stats.norm.ppf((1 + conf_level) / 2)
                        lower = forecast_values - z_score * std_values
                        upper = forecast_values + z_score * std_values
                        
                        confidence_intervals[deg_type][f'{int(conf_level*100)}%'] = {
                            'lower': lower,
                            'upper': upper
                        }
        
        # Seasonal decomposition if available
        seasonal_components = {}
        if 'decomposition' in model_output and model_output['decomposition']:
            decomp = model_output['decomposition']
            for component in ['trend', 'seasonal', 'residual']:
                if component in decomp:
                    seasonal_components[component] = decomp[component].squeeze().cpu().numpy()
        
        # Trend analysis
        trend_analysis = self._analyze_trends(forecast_dict)
        
        # Calculate forecast quality score
        quality_score = self._calculate_forecast_quality(forecasts_np, model_output)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(forecast_dict, trend_analysis)
        
        return ForecastResult(
            battery_id=battery_id,
            timestamp=time.time(),
            forecast_horizon_hours=forecast_horizon,
            forecasts=forecast_dict,
            uncertainty_bounds=uncertainty_bounds,
            seasonal_components=seasonal_components,
            trend_analysis=trend_analysis,
            confidence_intervals=confidence_intervals,
            forecast_quality_score=quality_score,
            recommendations=recommendations,
            metadata={
                'model_version': '1.0.0',
                'forecast_method': 'transformer',
                'data_points_used': len(original_data) if hasattr(original_data, '__len__') else 0
            }
        )
    
    def _analyze_trends(self, forecasts: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze trends in forecasted degradation patterns."""
        trend_analysis = {}
        
        for deg_type, values in forecasts.items():
            if len(values) > 1:
                # Calculate trend slope
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                # Determine trend direction
                if slope > 0.001:
                    trend_direction = "increasing"
                elif slope < -0.001:
                    trend_direction = "decreasing"
                else:
                    trend_direction = "stable"
                
                # Calculate acceleration (second derivative)
                if len(values) > 2:
                    acceleration = np.mean(np.diff(values, n=2))
                else:
                    acceleration = 0.0
                
                trend_analysis[deg_type] = {
                    'slope': slope,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'trend_direction': trend_direction,
                    'acceleration': acceleration,
                    'trend_strength': abs(slope) * r_value ** 2
                }
        
        return trend_analysis
    
    def _calculate_forecast_quality(self, forecasts: np.ndarray, 
                                  model_output: Dict) -> float:
        """Calculate overall forecast quality score."""
        quality_score = 0.8  # Base score
        
        # Adjust based on uncertainty if available
        if 'std' in model_output:
            std_values = model_output['std'].squeeze().cpu().numpy()
            avg_uncertainty = np.mean(std_values)
            
            # Lower uncertainty = higher quality
            uncertainty_penalty = min(0.3, avg_uncertainty * 10)
            quality_score -= uncertainty_penalty
        
        # Adjust based on forecast smoothness
        if forecasts.ndim > 1 and forecasts.shape[0] > 1:
            # Calculate smoothness (penalize erratic forecasts)
            smoothness = 1.0 / (1.0 + np.mean(np.abs(np.diff(forecasts, axis=0))))
            quality_score *= smoothness
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, quality_score))
    
    def _generate_recommendations(self, forecasts: Dict[str, np.ndarray],
                                trend_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on forecasts."""
        recommendations = []
        
        # Check for high degradation rates
        for deg_type, values in forecasts.items():
            avg_rate = np.mean(values)
            
            if deg_type == 'capacity_fade_rate' and avg_rate > 0.001:
                recommendations.append(
                    f"High capacity fade rate detected ({avg_rate:.4f}/hour). "
                    "Consider optimizing charging protocols."
                )
            
            elif deg_type == 'thermal_degradation' and avg_rate > 0.0005:
                recommendations.append(
                    f"Elevated thermal degradation ({avg_rate:.4f}/hour). "
                    "Improve thermal management systems."
                )
            
            elif deg_type == 'resistance_increase_rate' and avg_rate > 0.0008:
                recommendations.append(
                    f"High resistance increase rate ({avg_rate:.4f}/hour). "
                    "Monitor for internal battery issues."
                )
        
        # Check trend directions
        for deg_type, analysis in trend_analysis.items():
            if analysis['trend_direction'] == 'increasing' and analysis['trend_strength'] > 0.5:
                recommendations.append(
                    f"Accelerating {deg_type.replace('_', ' ')} trend detected. "
                    "Consider preventive maintenance."
                )
        
        # General recommendations
        if not recommendations:
            recommendations.append("Battery degradation patterns are within normal ranges.")
        
        return recommendations
    
    def _generate_cache_key(self, data: Any, horizon: int, metadata: Optional[Dict]) -> str:
        """Generate cache key for forecast."""
        data_str = str(data) + str(horizon) + str(metadata)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_cached_forecast(self, cache_key: str) -> Optional[ForecastResult]:
        """Get cached forecast result."""
        try:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                result_dict = json.loads(cached_data)
                return ForecastResult(**result_dict)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None
    
    def _cache_forecast(self, cache_key: str, result: ForecastResult) -> None:
        """Cache forecast result."""
        try:
            self.cache.setex(cache_key, self.config.cache_ttl, result.to_json())
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    async def forecast_async(self, battery_data: Union[pd.DataFrame, np.ndarray],
                           forecast_horizon: int = 168,
                           battery_metadata: Optional[Dict] = None) -> ForecastResult:
        """Asynchronous forecasting for high-throughput scenarios."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.forecast_degradation,
            battery_data,
            forecast_horizon,
            battery_metadata
        )
    
    def validate_forecast_accuracy(self, historical_forecasts: List[Dict],
                                 actual_data: List[Dict]) -> ForecastMetrics:
        """Validate forecast accuracy against actual data."""
        return self.forecast_validator.evaluate_forecasts(historical_forecasts, actual_data)
    
    def get_forecaster_info(self) -> Dict[str, Any]:
        """Get comprehensive forecaster information."""
        return {
            'model_type': 'DegradationForecaster',
            'version': '1.0.0',
            'device': str(self.device),
            'available_horizons': self.config.forecast_horizons,
            'confidence_levels': self.config.confidence_levels,
            'features_enabled': {
                'uncertainty_quantification': self.config.enable_uncertainty,
                'seasonal_analysis': self.config.enable_seasonal_analysis,
                'trend_analysis': self.config.enable_trend_analysis,
                'caching': self.config.cache_forecasts
            },
            'performance_stats': {
                'total_forecasts': len(self.forecast_history),
                'cache_enabled': self.cache is not None
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the forecasting system."""
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'model_loaded': self.model is not None,
            'device_available': torch.cuda.is_available() if 'cuda' in str(self.device) else True,
            'cache_available': self.cache is not None and self.cache.ping() if self.cache else False
        }
        
        # Check model inference
        try:
            dummy_input = torch.randn(1, 168, 20).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            health_status['model_inference'] = True
        except Exception as e:
            health_status['model_inference'] = False
            health_status['inference_error'] = str(e)
            health_status['status'] = 'unhealthy'
        
        return health_status

# Factory functions
def create_battery_degradation_forecaster(config: Optional[ForecastingConfig] = None) -> BatteryDegradationForecaster:
    """
    Factory function to create a BatteryDegradationForecaster.
    
    Args:
        config (ForecastingConfig, optional): Forecasting configuration
        
    Returns:
        BatteryDegradationForecaster: Configured forecaster instance
    """
    if config is None:
        config = ForecastingConfig()
    
    return BatteryDegradationForecaster(config)
