"""
BatteryMind - Inference Predictors Module

Production-ready inference predictors for battery health monitoring, degradation
forecasting, and optimization recommendations. Provides unified interfaces for
all BatteryMind AI models with real-time and batch prediction capabilities.

Key Components:
- BatteryHealthPredictor: State of Health and degradation predictions
- DegradationPredictor: Long-term degradation forecasting
- OptimizationPredictor: Charging and maintenance optimization
- EnsemblePredictor: Combined predictions from multiple models

Features:
- Real-time inference with sub-100ms latency
- Batch processing for fleet-scale predictions
- Model ensemble orchestration
- Uncertainty quantification
- Physics-informed constraints
- Edge deployment optimization

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import warnings
from abc import ABC, abstractmethod

# BatteryMind imports
from ..preprocessing_scripts.data_cleaner import BatteryDataCleaner
from ..preprocessing_scripts.feature_extractor import BatteryFeatureExtractor
from ..preprocessing_scripts.normalization import BatteryDataNormalizer
from ..model_artifacts import ModelArtifactManager
from ..transformers.battery_health_predictor.predictor import BatteryHealthPredictor as TransformerPredictor
from ..transformers.degradation_forecaster.forecaster import DegradationForecaster
from ..transformers.optimization_recommender.recommender import OptimizationRecommender
from ..transformers.ensemble_model.ensemble import EnsembleModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module metadata
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module exports
__all__ = [
    # Base classes
    "BasePredictor",
    "PredictionRequest",
    "PredictionResponse",
    
    # Specific predictors
    "BatteryHealthPredictor",
    "DegradationPredictor", 
    "OptimizationPredictor",
    "EnsemblePredictor",
    
    # Utility functions
    "create_predictor",
    "load_predictor",
    "validate_prediction_input",
    "format_prediction_output",
    
    # Configuration
    "PredictorConfig",
    "InferenceConfig"
]

@dataclass
class PredictionRequest:
    """
    Standardized prediction request structure.
    
    Attributes:
        data (Dict[str, Any]): Input sensor data and metadata
        model_type (str): Type of model to use for prediction
        prediction_horizon (int): Number of time steps to predict
        confidence_level (float): Confidence level for uncertainty quantification
        batch_size (int): Batch size for processing
        real_time (bool): Whether this is a real-time prediction request
        metadata (Dict[str, Any]): Additional metadata
    """
    data: Dict[str, Any]
    model_type: str = "transformer"
    prediction_horizon: int = 1
    confidence_level: float = 0.95
    batch_size: int = 1
    real_time: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictionResponse:
    """
    Standardized prediction response structure.
    
    Attributes:
        predictions (Dict[str, Any]): Model predictions
        confidence_intervals (Dict[str, Tuple[float, float]]): Confidence intervals
        uncertainty_metrics (Dict[str, float]): Uncertainty quantification metrics
        feature_importance (Dict[str, float]): Feature importance scores
        processing_time_ms (float): Time taken for prediction
        model_version (str): Version of model used
        warnings (List[str]): Any warnings or issues
        metadata (Dict[str, Any]): Additional response metadata
    """
    predictions: Dict[str, Any]
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    uncertainty_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    model_version: str = "1.0.0"
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictorConfig:
    """
    Configuration for predictor components.
    
    Attributes:
        model_path (str): Path to trained model artifacts
        preprocessing_config (Dict[str, Any]): Preprocessing configuration
        inference_config (Dict[str, Any]): Inference configuration
        deployment_mode (str): Deployment mode (cloud, edge, mobile)
        max_batch_size (int): Maximum batch size for processing
        timeout_seconds (float): Timeout for predictions
        enable_caching (bool): Enable prediction caching
        log_predictions (bool): Log prediction requests and responses
    """
    model_path: str = "./model-artifacts/trained_models/"
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    inference_config: Dict[str, Any] = field(default_factory=dict)
    deployment_mode: str = "cloud"
    max_batch_size: int = 32
    timeout_seconds: float = 30.0
    enable_caching: bool = True
    log_predictions: bool = True

class BasePredictor(ABC):
    """
    Abstract base class for all BatteryMind predictors.
    """
    
    def __init__(self, config: PredictorConfig):
        self.config = config
        self.model = None
        self.preprocessor = None
        self.normalizer = None
        self.is_loaded = False
        self.prediction_cache = {}
        self.performance_metrics = {
            'total_predictions': 0,
            'average_latency_ms': 0.0,
            'error_count': 0,
            'cache_hit_rate': 0.0
        }
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the trained model and preprocessing components."""
        pass
    
    @abstractmethod
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make predictions on input data."""
        pass
    
    @abstractmethod
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data format and constraints."""
        pass
    
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply preprocessing to input data."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Clean data
        if self.preprocessor:
            cleaned_data = self.preprocessor.clean(data)
        else:
            cleaned_data = data
        
        # Normalize data
        if self.normalizer:
            normalized_data = self.normalizer.transform(cleaned_data)
        else:
            normalized_data = cleaned_data
        
        return normalized_data
    
    def postprocess_predictions(self, predictions: Any) -> Dict[str, Any]:
        """Apply postprocessing to model predictions."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        
        # Convert to standard format
        if isinstance(predictions, np.ndarray):
            if predictions.ndim == 1:
                result = {"prediction": predictions.tolist()}
            else:
                result = {"predictions": predictions.tolist()}
        else:
            result = {"predictions": predictions}
        
        return result
    
    def get_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key for input data."""
        import hashlib
        
        # Create deterministic hash of input data
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def update_performance_metrics(self, latency_ms: float, cache_hit: bool = False):
        """Update performance metrics."""
        self.performance_metrics['total_predictions'] += 1
        
        # Update average latency
        total_preds = self.performance_metrics['total_predictions']
        current_avg = self.performance_metrics['average_latency_ms']
        self.performance_metrics['average_latency_ms'] = (
            (current_avg * (total_preds - 1) + latency_ms) / total_preds
        )
        
        # Update cache hit rate
        if cache_hit:
            cache_hits = self.performance_metrics.get('cache_hits', 0) + 1
            self.performance_metrics['cache_hits'] = cache_hits
            self.performance_metrics['cache_hit_rate'] = cache_hits / total_preds
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def clear_cache(self):
        """Clear prediction cache."""
        self.prediction_cache.clear()
        self.logger.info("Prediction cache cleared")

# Utility functions
def create_predictor(predictor_type: str, config: Optional[PredictorConfig] = None) -> BasePredictor:
    """
    Factory function to create predictor instances.
    
    Args:
        predictor_type (str): Type of predictor to create
        config (PredictorConfig, optional): Predictor configuration
        
    Returns:
        BasePredictor: Created predictor instance
    """
    if config is None:
        config = PredictorConfig()
    
    predictor_map = {
        "battery_health": BatteryHealthPredictor,
        "degradation": DegradationPredictor,
        "optimization": OptimizationPredictor,
        "ensemble": EnsemblePredictor
    }
    
    if predictor_type not in predictor_map:
        available_types = list(predictor_map.keys())
        raise ValueError(f"Unknown predictor type: {predictor_type}. Available types: {available_types}")
    
    predictor_class = predictor_map[predictor_type]
    return predictor_class(config)

def load_predictor(predictor_path: str) -> BasePredictor:
    """
    Load a predictor from saved artifacts.
    
    Args:
        predictor_path (str): Path to predictor artifacts
        
    Returns:
        BasePredictor: Loaded predictor instance
    """
    # Load predictor configuration
    config_path = Path(predictor_path) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        predictor_type = config_dict.get('predictor_type', 'battery_health')
        config = PredictorConfig(**config_dict.get('config', {}))
        config.model_path = predictor_path
        
        # Create and load predictor
        predictor = create_predictor(predictor_type, config)
        predictor.load_model()
        
        return predictor
    else:
        raise FileNotFoundError(f"Predictor configuration not found at {config_path}")

def validate_prediction_input(data: Dict[str, Any], 
                            required_fields: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate prediction input data.
    
    Args:
        data (Dict[str, Any]): Input data to validate
        required_fields (List[str], optional): List of required fields
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []
    
    # Check if data is provided
    if not data:
        errors.append("No input data provided")
        return False, errors
    
    # Check required fields
    if required_fields:
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
    
    # Check data types and ranges for common battery parameters
    validation_rules = {
        'voltage': {'type': (int, float), 'range': (0, 10)},
        'current': {'type': (int, float), 'range': (-1000, 1000)},
        'temperature': {'type': (int, float), 'range': (-50, 100)},
        'soc': {'type': (int, float), 'range': (0, 1)},
        'soh': {'type': (int, float), 'range': (0, 1)}
    }
    
    for field, rules in validation_rules.items():
        if field in data:
            value = data[field]
            
            # Check type
            if not isinstance(value, rules['type']):
                errors.append(f"Field '{field}' must be of type {rules['type']}")
            
            # Check range
            elif 'range' in rules:
                min_val, max_val = rules['range']
                if not (min_val <= value <= max_val):
                    errors.append(f"Field '{field}' must be between {min_val} and {max_val}")
    
    return len(errors) == 0, errors

def format_prediction_output(predictions: Any, 
                           format_type: str = "json") -> Union[Dict[str, Any], str]:
    """
    Format prediction output for different consumption formats.
    
    Args:
        predictions (Any): Raw predictions from model
        format_type (str): Output format type
        
    Returns:
        Union[Dict[str, Any], str]: Formatted predictions
    """
    if isinstance(predictions, PredictionResponse):
        data = {
            'predictions': predictions.predictions,
            'confidence_intervals': predictions.confidence_intervals,
            'uncertainty_metrics': predictions.uncertainty_metrics,
            'processing_time_ms': predictions.processing_time_ms,
            'model_version': predictions.model_version,
            'warnings': predictions.warnings
        }
    else:
        data = {'predictions': predictions}
    
    if format_type == "json":
        return json.dumps(data, indent=2, default=str)
    elif format_type == "dict":
        return data
    else:
        raise ValueError(f"Unsupported format type: {format_type}")

# Performance monitoring
class PredictionMonitor:
    """Monitor prediction performance and system health."""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_latency_ms': 0.0,
            'throughput_per_second': 0.0,
            'error_rate': 0.0
        }
        self.start_time = time.time()
    
    def record_prediction(self, success: bool, latency_ms: float):
        """Record a prediction event."""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_predictions'] += 1
        else:
            self.metrics['failed_predictions'] += 1
        
        # Update average latency
        total_requests = self.metrics['total_requests']
        current_avg = self.metrics['average_latency_ms']
        self.metrics['average_latency_ms'] = (
            (current_avg * (total_requests - 1) + latency_ms) / total_requests
        )
        
        # Calculate throughput and error rate
        elapsed_time = time.time() - self.start_time
        self.metrics['throughput_per_second'] = self.metrics['total_requests'] / elapsed_time
        self.metrics['error_rate'] = self.metrics['failed_predictions'] / self.metrics['total_requests']
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = {
            'total_requests': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_latency_ms': 0.0,
            'throughput_per_second': 0.0,
            'error_rate': 0.0
        }
        self.start_time = time.time()

# Global monitor instance
prediction_monitor = PredictionMonitor()

def get_prediction_monitor() -> PredictionMonitor:
    """Get the global prediction monitor instance."""
    return prediction_monitor

# Error handling
class PredictionError(Exception):
    """Base exception for prediction errors."""
    pass

class ModelNotLoadedError(PredictionError):
    """Raised when model is not loaded."""
    pass

class InvalidInputError(PredictionError):
    """Raised when input data is invalid."""
    pass

class PredictionTimeoutError(PredictionError):
    """Raised when prediction times out."""
    pass

# Model response caching
class PredictionCache:
    """Simple in-memory cache for prediction responses."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[PredictionResponse]:
        """Get cached prediction response."""
        if key in self.cache:
            response, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return response
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, response: PredictionResponse):
        """Cache prediction response."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (response, time.time())
    
    def clear(self):
        """Clear all cached responses."""
        self.cache.clear()

# Global cache instance
prediction_cache = PredictionCache()

def get_prediction_cache() -> PredictionCache:
    """Get the global prediction cache instance."""
    return prediction_cache

# Module initialization
logger.info(f"BatteryMind Inference Predictors v{__version__} initialized")
logger.info(f"Available predictor types: {list(create_predictor.__annotations__.keys())}")
