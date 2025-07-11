"""
BatteryMind - Degradation Forecaster Module

Advanced time-series forecasting module for battery degradation prediction
using transformer architectures optimized for long-term temporal dependencies.

This module provides comprehensive forecasting capabilities for battery
degradation patterns, enabling proactive maintenance and lifecycle optimization.

Key Components:
- DegradationForecaster: Main transformer model for degradation forecasting
- DegradationTrainer: Training pipeline with time-series specific optimizations
- TimeSeriesUtils: Utilities for time-series data processing and analysis
- DegradationConfig: Configuration management for forecasting models

Features:
- Long-term degradation pattern prediction (weeks to years)
- Multi-horizon forecasting with uncertainty quantification
- Seasonal decomposition and trend analysis
- Physics-informed constraints for realistic predictions
- Integration with battery health prediction models
- Support for multiple battery chemistries and usage patterns

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

from .model import (
    DegradationForecaster,
    DegradationConfig,
    TemporalAttention,
    SeasonalDecomposition,
    TrendAnalysis,
    UncertaintyQuantification
)

from .trainer import (
    DegradationTrainer,
    DegradationTrainingConfig,
    DegradationTrainingMetrics,
    TimeSeriesLoss,
    ForecastingOptimizer
)

from .forecaster import (
    BatteryDegradationForecaster,
    ForecastResult,
    ForecastMetrics,
    ForecastingConfig
)

from .time_series_utils import (
    TimeSeriesProcessor,
    SeasonalityDetector,
    TrendExtractor,
    ChangePointDetector,
    ForecastValidator,
    TimeSeriesAugmentation
)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module metadata
__all__ = [
    # Model components
    "DegradationForecaster",
    "DegradationConfig",
    "TemporalAttention",
    "SeasonalDecomposition",
    "TrendAnalysis",
    "UncertaintyQuantification",
    
    # Training components
    "DegradationTrainer",
    "DegradationTrainingConfig",
    "DegradationTrainingMetrics",
    "TimeSeriesLoss",
    "ForecastingOptimizer",
    
    # Forecasting components
    "BatteryDegradationForecaster",
    "ForecastResult",
    "ForecastMetrics",
    "ForecastingConfig",
    
    # Time series utilities
    "TimeSeriesProcessor",
    "SeasonalityDetector",
    "TrendExtractor",
    "ChangePointDetector",
    "ForecastValidator",
    "TimeSeriesAugmentation"
]

# Module configuration
DEFAULT_FORECASTING_CONFIG = {
    "model": {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 8,  # Deeper for long-term dependencies
        "d_ff": 2048,
        "dropout": 0.1,
        "max_sequence_length": 2048,  # Longer sequences for forecasting
        "forecast_horizon": 168,  # 1 week in hours
        "uncertainty_quantiles": [0.1, 0.25, 0.75, 0.9]
    },
    "training": {
        "batch_size": 16,  # Smaller batch for longer sequences
        "learning_rate": 5e-5,  # Lower learning rate for stability
        "num_epochs": 200,
        "warmup_steps": 8000,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0
    },
    "data": {
        "sequence_length": 1024,  # Longer input sequences
        "forecast_horizon": 168,  # Forecast horizon in time steps
        "overlap_ratio": 0.8,  # Higher overlap for forecasting
        "feature_dim": 20,  # More features for forecasting
        "target_dim": 6  # Multiple degradation metrics
    },
    "time_series": {
        "enable_seasonality_detection": True,
        "enable_trend_analysis": True,
        "enable_change_point_detection": True,
        "seasonal_periods": [24, 168, 720],  # Daily, weekly, monthly
        "detrending_method": "linear",
        "seasonality_method": "additive"
    }
}

def get_default_forecasting_config():
    """
    Get default configuration for degradation forecasting.
    
    Returns:
        dict: Default configuration dictionary
    """
    return DEFAULT_FORECASTING_CONFIG.copy()

def create_degradation_forecaster(config=None):
    """
    Factory function to create a degradation forecasting model.
    
    Args:
        config (dict, optional): Model configuration. If None, uses default config.
        
    Returns:
        DegradationForecaster: Configured forecasting model instance
    """
    if config is None:
        config = get_default_forecasting_config()
    
    model_config = DegradationConfig(**config["model"])
    return DegradationForecaster(model_config)

def create_degradation_trainer(model, config=None):
    """
    Factory function to create a degradation forecasting trainer.
    
    Args:
        model (DegradationForecaster): Model to train
        config (dict, optional): Training configuration. If None, uses default config.
        
    Returns:
        DegradationTrainer: Configured trainer instance
    """
    if config is None:
        config = get_default_forecasting_config()
    
    training_config = DegradationTrainingConfig(**config["training"])
    return DegradationTrainer(model, training_config)

def create_battery_degradation_forecaster(model_path, config=None):
    """
    Factory function to create a battery degradation forecaster for inference.
    
    Args:
        model_path (str): Path to trained model
        config (dict, optional): Forecasting configuration. If None, uses default config.
        
    Returns:
        BatteryDegradationForecaster: Configured forecaster instance
    """
    if config is None:
        config = get_default_forecasting_config()
    
    forecasting_config = ForecastingConfig(**config.get("forecasting", {}))
    return BatteryDegradationForecaster(model_path, forecasting_config)

# Forecasting-specific constants
FORECASTING_CONSTANTS = {
    # Time horizons (in hours)
    "SHORT_TERM_HORIZON": 24,      # 1 day
    "MEDIUM_TERM_HORIZON": 168,    # 1 week
    "LONG_TERM_HORIZON": 720,      # 1 month
    "EXTENDED_HORIZON": 8760,      # 1 year
    
    # Degradation metrics
    "PRIMARY_METRICS": [
        "capacity_fade_rate",
        "resistance_increase_rate",
        "thermal_degradation_rate",
        "cycle_efficiency_decline",
        "calendar_aging_rate",
        "overall_health_decline"
    ],
    
    # Forecasting accuracy thresholds
    "ACCURACY_THRESHOLDS": {
        "short_term_mape": 5.0,     # 5% MAPE for short-term
        "medium_term_mape": 10.0,   # 10% MAPE for medium-term
        "long_term_mape": 15.0,     # 15% MAPE for long-term
        "uncertainty_coverage": 0.9  # 90% prediction interval coverage
    },
    
    # Physics constraints for degradation
    "PHYSICS_CONSTRAINTS": {
        "max_daily_capacity_fade": 0.001,    # 0.1% per day maximum
        "max_temperature_impact": 2.0,       # 2x degradation at 60°C vs 25°C
        "min_cycle_efficiency": 0.8,         # 80% minimum cycle efficiency
        "max_resistance_increase": 0.01      # 1% per month maximum
    }
}

def get_forecasting_constants():
    """
    Get forecasting-specific constants.
    
    Returns:
        dict: Dictionary of forecasting constants
    """
    return FORECASTING_CONSTANTS.copy()

# Integration with battery health predictor
def create_integrated_battery_model(health_model_path, forecasting_model_path, config=None):
    """
    Create an integrated model combining health prediction and degradation forecasting.
    
    Args:
        health_model_path (str): Path to trained health prediction model
        forecasting_model_path (str): Path to trained forecasting model
        config (dict, optional): Integration configuration
        
    Returns:
        dict: Dictionary containing both models and integration utilities
    """
    from ..battery_health_predictor import create_battery_predictor, BatteryInferenceConfig
    
    # Create health predictor
    health_config = BatteryInferenceConfig(model_path=health_model_path)
    health_predictor = create_battery_predictor(health_config)
    
    # Create degradation forecaster
    forecasting_config = ForecastingConfig(model_path=forecasting_model_path)
    degradation_forecaster = create_battery_degradation_forecaster(
        forecasting_model_path, {"forecasting": forecasting_config.__dict__}
    )
    
    return {
        "health_predictor": health_predictor,
        "degradation_forecaster": degradation_forecaster,
        "integration_config": config or {},
        "version": __version__
    }

# Validation utilities
def validate_forecasting_data(data, config=None):
    """
    Validate data for degradation forecasting.
    
    Args:
        data: Input data for validation
        config (dict, optional): Validation configuration
        
    Returns:
        dict: Validation results
    """
    from .time_series_utils import TimeSeriesProcessor
    
    processor = TimeSeriesProcessor(config or get_default_forecasting_config())
    return processor.validate_data(data)

def estimate_forecasting_performance(data, model_config=None):
    """
    Estimate expected forecasting performance based on data characteristics.
    
    Args:
        data: Historical data for analysis
        model_config (dict, optional): Model configuration
        
    Returns:
        dict: Performance estimates
    """
    from .time_series_utils import ForecastValidator
    
    validator = ForecastValidator(model_config or get_default_forecasting_config())
    return validator.estimate_performance(data)

# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BatteryMind Degradation Forecaster v{__version__} initialized")

# Compatibility checks
def check_compatibility():
    """
    Check compatibility with other BatteryMind modules.
    
    Returns:
        dict: Compatibility status
    """
    compatibility_status = {
        "battery_health_predictor": True,
        "ensemble_model": True,
        "federated_learning": True,
        "reinforcement_learning": False,  # Not directly compatible
        "version": __version__
    }
    
    try:
        from ..battery_health_predictor import __version__ as health_version
        compatibility_status["health_predictor_version"] = health_version
    except ImportError:
        compatibility_status["battery_health_predictor"] = False
        logger.warning("Battery health predictor module not found")
    
    return compatibility_status

# Performance optimization hints
OPTIMIZATION_HINTS = {
    "memory_optimization": {
        "use_gradient_checkpointing": True,
        "reduce_sequence_length": "if memory constrained",
        "use_mixed_precision": True,
        "batch_size_adjustment": "reduce for longer sequences"
    },
    "training_optimization": {
        "learning_rate_scheduling": "cosine annealing with warm restarts",
        "gradient_accumulation": "use for effective larger batch sizes",
        "early_stopping": "monitor validation loss with patience=20",
        "model_checkpointing": "save best model based on forecasting accuracy"
    },
    "inference_optimization": {
        "model_quantization": "for edge deployment",
        "sequence_caching": "cache intermediate representations",
        "batch_inference": "process multiple forecasts together",
        "uncertainty_estimation": "use dropout at inference for uncertainty"
    }
}

def get_optimization_hints():
    """
    Get optimization hints for degradation forecasting.
    
    Returns:
        dict: Optimization recommendations
    """
    return OPTIMIZATION_HINTS.copy()

# Export configuration for easy access
def export_config_template(file_path="degradation_forecaster_config.yaml"):
    """
    Export a configuration template for degradation forecasting.
    
    Args:
        file_path (str): Path to save the configuration template
    """
    import yaml
    
    config_template = get_default_forecasting_config()
    config_template["_metadata"] = {
        "version": __version__,
        "description": "BatteryMind Degradation Forecaster Configuration Template",
        "author": __author__,
        "created": "2025-07-11"
    }
    
    with open(file_path, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration template exported to {file_path}")

# Module health check
def health_check():
    """
    Perform a health check of the degradation forecaster module.
    
    Returns:
        dict: Health check results
    """
    health_status = {
        "module_loaded": True,
        "version": __version__,
        "dependencies_available": True,
        "configuration_valid": True,
        "compatibility_status": check_compatibility()
    }
    
    try:
        # Test basic functionality
        config = get_default_forecasting_config()
        model = create_degradation_forecaster(config)
        health_status["model_creation"] = True
    except Exception as e:
        health_status["model_creation"] = False
        health_status["error"] = str(e)
        logger.error(f"Health check failed: {e}")
    
    return health_status
