"""
BatteryMind - Data Preprocessing Scripts

Comprehensive data preprocessing pipeline for battery sensor data, including
cleaning, feature extraction, normalization, and time-series processing.

This module provides robust preprocessing capabilities for:
- Multi-modal sensor data (voltage, current, temperature, etc.)
- Time-series data with varying sampling rates
- Missing data imputation and outlier detection
- Feature engineering and extraction
- Data augmentation for improved model training
- Cross-battery normalization and standardization

Key Components:
- DataCleaner: Comprehensive data cleaning and validation
- FeatureExtractor: Advanced feature engineering for battery data
- DataAugmentation: Synthetic data generation and augmentation
- Normalization: Multi-battery normalization strategies
- TimeSeriesSplitter: Temporal data splitting with leak prevention

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

from .data_cleaner import (
    BatteryDataCleaner,
    DataQualityMetrics,
    CleaningConfiguration,
    OutlierDetector,
    MissingDataHandler,
    DataValidator,
    clean_battery_dataset,
    validate_sensor_data,
    detect_anomalies,
    repair_missing_data
)

from .feature_extractor import (
    BatteryFeatureExtractor,
    FeatureConfig,
    TemporalFeatures,
    StatisticalFeatures,
    DomainFeatures,
    extract_battery_features,
    create_feature_pipeline,
    compute_degradation_indicators,
    extract_usage_patterns
)

from .data_augmentation import (
    BatteryDataAugmentor,
    AugmentationConfig,
    NoiseAugmentation,
    TemporalAugmentation,
    PhysicsAugmentation,
    augment_battery_data,
    generate_synthetic_variations,
    create_augmentation_pipeline
)

from .normalization import (
    BatteryDataNormalizer,
    NormalizationConfig,
    ScalingStrategy,
    CrossBatteryNormalizer,
    TemporalNormalizer,
    normalize_battery_data,
    create_normalization_pipeline,
    fit_normalizer,
    transform_data
)

from .time_series_splitter import (
    TimeSeriesSplitter,
    SplittingConfig,
    TemporalSplit,
    CrossValidationSplitter,
    split_time_series_data,
    create_temporal_splits,
    validate_temporal_consistency,
    prevent_data_leakage
)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module metadata
__all__ = [
    # Data cleaning
    "BatteryDataCleaner",
    "DataQualityMetrics",
    "CleaningConfiguration", 
    "OutlierDetector",
    "MissingDataHandler",
    "DataValidator",
    "clean_battery_dataset",
    "validate_sensor_data",
    "detect_anomalies",
    "repair_missing_data",
    
    # Feature extraction
    "BatteryFeatureExtractor",
    "FeatureConfig",
    "TemporalFeatures",
    "StatisticalFeatures", 
    "DomainFeatures",
    "extract_battery_features",
    "create_feature_pipeline",
    "compute_degradation_indicators",
    "extract_usage_patterns",
    
    # Data augmentation
    "BatteryDataAugmentor",
    "AugmentationConfig",
    "NoiseAugmentation",
    "TemporalAugmentation",
    "PhysicsAugmentation", 
    "augment_battery_data",
    "generate_synthetic_variations",
    "create_augmentation_pipeline",
    
    # Normalization
    "BatteryDataNormalizer",
    "NormalizationConfig",
    "ScalingStrategy",
    "CrossBatteryNormalizer",
    "TemporalNormalizer",
    "normalize_battery_data",
    "create_normalization_pipeline",
    "fit_normalizer",
    "transform_data",
    
    # Time series splitting
    "TimeSeriesSplitter",
    "SplittingConfig", 
    "TemporalSplit",
    "CrossValidationSplitter",
    "split_time_series_data",
    "create_temporal_splits",
    "validate_temporal_consistency",
    "prevent_data_leakage"
]

# Default preprocessing configuration
DEFAULT_PREPROCESSING_CONFIG = {
    "cleaning": {
        "remove_outliers": True,
        "outlier_method": "isolation_forest",
        "outlier_threshold": 0.1,
        "handle_missing": True,
        "missing_strategy": "interpolation",
        "max_missing_ratio": 0.05,
        "validate_ranges": True,
        "voltage_range": [2.5, 4.2],
        "current_range": [-200, 200],
        "temperature_range": [-40, 80]
    },
    "features": {
        "temporal_features": True,
        "statistical_features": True,
        "domain_features": True,
        "window_sizes": [10, 50, 100],
        "overlap_ratio": 0.5,
        "extract_trends": True,
        "extract_cycles": True
    },
    "augmentation": {
        "enable_augmentation": True,
        "noise_level": 0.01,
        "temporal_jitter": 0.02,
        "physics_variations": True,
        "augmentation_factor": 2.0
    },
    "normalization": {
        "method": "robust_scaler",
        "per_battery": False,
        "temporal_normalization": True,
        "feature_wise": True
    },
    "splitting": {
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "temporal_split": True,
        "prevent_leakage": True,
        "cv_folds": 5
    }
}

def get_default_config():
    """
    Get default preprocessing configuration.
    
    Returns:
        dict: Default preprocessing configuration
    """
    return DEFAULT_PREPROCESSING_CONFIG.copy()

def create_preprocessing_pipeline(config=None):
    """
    Create a complete preprocessing pipeline.
    
    Args:
        config (dict, optional): Preprocessing configuration
        
    Returns:
        dict: Preprocessing pipeline components
    """
    if config is None:
        config = get_default_config()
    
    # Create individual components
    cleaner = BatteryDataCleaner(CleaningConfiguration(**config["cleaning"]))
    feature_extractor = BatteryFeatureExtractor(FeatureConfig(**config["features"]))
    augmentor = BatteryDataAugmentor(AugmentationConfig(**config["augmentation"]))
    normalizer = BatteryDataNormalizer(NormalizationConfig(**config["normalization"]))
    splitter = TimeSeriesSplitter(SplittingConfig(**config["splitting"]))
    
    return {
        "cleaner": cleaner,
        "feature_extractor": feature_extractor,
        "augmentor": augmentor,
        "normalizer": normalizer,
        "splitter": splitter,
        "config": config
    }

def preprocess_battery_dataset(data, pipeline=None, config=None):
    """
    Complete preprocessing of battery dataset.
    
    Args:
        data: Raw battery data
        pipeline (dict, optional): Preprocessing pipeline
        config (dict, optional): Preprocessing configuration
        
    Returns:
        dict: Preprocessed data and metadata
    """
    if pipeline is None:
        pipeline = create_preprocessing_pipeline(config)
    
    # Step 1: Data cleaning
    cleaned_data = pipeline["cleaner"].clean(data)
    
    # Step 2: Feature extraction
    features = pipeline["feature_extractor"].extract(cleaned_data)
    
    # Step 3: Data augmentation (if enabled)
    if pipeline["config"]["augmentation"]["enable_augmentation"]:
        augmented_data = pipeline["augmentor"].augment(features)
    else:
        augmented_data = features
    
    # Step 4: Normalization
    normalized_data = pipeline["normalizer"].normalize(augmented_data)
    
    # Step 5: Time series splitting
    splits = pipeline["splitter"].split(normalized_data)
    
    return {
        "processed_data": normalized_data,
        "splits": splits,
        "pipeline": pipeline,
        "metadata": {
            "original_samples": len(data),
            "cleaned_samples": len(cleaned_data),
            "augmented_samples": len(augmented_data),
            "features_extracted": features.shape[1] if hasattr(features, 'shape') else len(features),
            "preprocessing_config": pipeline["config"]
        }
    }

# Data quality assessment utilities
def assess_data_quality(data):
    """
    Assess the quality of battery data.
    
    Args:
        data: Battery dataset to assess
        
    Returns:
        dict: Data quality assessment results
    """
    import pandas as pd
    import numpy as np
    
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)
    
    quality_metrics = {
        "total_samples": len(df),
        "missing_data": {
            "total_missing": df.isnull().sum().sum(),
            "missing_percentage": (df.isnull().sum().sum() / df.size) * 100,
            "columns_with_missing": df.columns[df.isnull().any()].tolist()
        },
        "data_types": df.dtypes.to_dict(),
        "numeric_summary": df.describe().to_dict(),
        "duplicates": df.duplicated().sum(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    # Detect potential outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outliers[col] = {
            "count": outlier_count,
            "percentage": (outlier_count / len(df)) * 100
        }
    
    quality_metrics["outliers"] = outliers
    
    return quality_metrics

def validate_preprocessing_config(config):
    """
    Validate preprocessing configuration.
    
    Args:
        config (dict): Configuration to validate
        
    Returns:
        dict: Validation results
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check required sections
    required_sections = ["cleaning", "features", "normalization", "splitting"]
    for section in required_sections:
        if section not in config:
            validation_results["errors"].append(f"Missing required section: {section}")
    
    # Validate splitting ratios
    if "splitting" in config:
        split_config = config["splitting"]
        total_ratio = (split_config.get("train_ratio", 0) + 
                      split_config.get("val_ratio", 0) + 
                      split_config.get("test_ratio", 0))
        if abs(total_ratio - 1.0) > 0.01:
            validation_results["errors"].append("Train/val/test ratios must sum to 1.0")
    
    # Validate feature extraction parameters
    if "features" in config:
        feature_config = config["features"]
        window_sizes = feature_config.get("window_sizes", [])
        if not window_sizes or any(w <= 0 for w in window_sizes):
            validation_results["warnings"].append("Invalid window sizes for feature extraction")
    
    # Set validation status
    validation_results["valid"] = len(validation_results["errors"]) == 0
    
    return validation_results

# Performance monitoring
class PreprocessingMonitor:
    """Monitor preprocessing performance and statistics."""
    
    def __init__(self):
        self.processing_times = {}
        self.data_statistics = {}
        self.error_counts = {}
    
    def record_processing_time(self, step_name, duration):
        """Record processing time for a step."""
        if step_name not in self.processing_times:
            self.processing_times[step_name] = []
        self.processing_times[step_name].append(duration)
    
    def record_data_statistics(self, step_name, stats):
        """Record data statistics for a step."""
        self.data_statistics[step_name] = stats
    
    def record_error(self, step_name, error_type):
        """Record an error during preprocessing."""
        key = f"{step_name}_{error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
    
    def get_performance_summary(self):
        """Get performance summary."""
        import numpy as np
        
        summary = {
            "processing_times": {},
            "data_flow": self.data_statistics,
            "error_summary": self.error_counts,
            "total_errors": sum(self.error_counts.values())
        }
        
        for step, times in self.processing_times.items():
            summary["processing_times"][step] = {
                "mean_seconds": np.mean(times),
                "std_seconds": np.std(times),
                "min_seconds": np.min(times),
                "max_seconds": np.max(times),
                "total_calls": len(times)
            }
        
        return summary

# Global preprocessing monitor
preprocessing_monitor = PreprocessingMonitor()

def get_preprocessing_monitor():
    """Get the global preprocessing monitor instance."""
    return preprocessing_monitor

# Logging setup
import logging
logger = logging.getLogger(__name__)
logger.info(f"BatteryMind Preprocessing Scripts v{__version__} initialized")
