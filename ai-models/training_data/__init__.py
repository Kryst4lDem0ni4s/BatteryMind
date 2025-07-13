"""
BatteryMind - Training Data Module

Comprehensive training data management system for battery AI/ML models.
Provides synthetic data generation, real-world data integration, preprocessing
pipelines, and validation frameworks for robust model training.

Key Components:
- Synthetic Datasets: Physics-based synthetic battery data generation
- Real World Samples: Integration with real battery telemetry and lab data
- Preprocessing Scripts: Advanced data cleaning and feature engineering
- Validation Sets: Comprehensive validation and testing datasets
- Generators: Automated data generation and scenario building

Features:
- Physics-based battery simulation for 10,000+ virtual batteries
- Multi-modal sensor data synthesis (electrical, thermal, acoustic)
- Realistic degradation patterns across diverse operating conditions
- Fleet-scale usage pattern simulation for federated learning
- Automated data quality monitoring and validation
- Cross-chemistry compatibility and standardization

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

from .synthetic_datasets import (
    BatteryTelemetryGenerator,
    DegradationCurveGenerator,
    FleetPatternGenerator,
    EnvironmentalDataGenerator,
    UsageProfileGenerator,
    SyntheticDataConfig
)

from .real_world_samples import (
    TataEVDataLoader,
    LabTestDataLoader,
    FieldStudyDataLoader,
    BenchmarkDataLoader,
    RealDataConfig,
    DataQualityMetrics
)

from .preprocessing_scripts import (
    DataCleaner,
    FeatureExtractor,
    DataAugmentation,
    Normalization,
    TimeSeriesSplitter,
    PreprocessingConfig,
    PreprocessingPipeline
)

from .validation_sets import (
    TestScenarioGenerator,
    HoldoutDataManager,
    CrossValidationSplitter,
    PerformanceBenchmarks,
    ValidationConfig,
    ValidationMetrics
)

from .generators import (
    SyntheticGenerator,
    PhysicsSimulator,
    NoiseGenerator,
    ScenarioBuilder,
    GeneratorConfig,
    GenerationMetrics
)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module metadata
__all__ = [
    # Synthetic Datasets
    "BatteryTelemetryGenerator",
    "DegradationCurveGenerator",
    "FleetPatternGenerator",
    "EnvironmentalDataGenerator",
    "UsageProfileGenerator",
    "SyntheticDataConfig",
    
    # Real World Samples
    "TataEVDataLoader",
    "LabTestDataLoader",
    "FieldStudyDataLoader",
    "BenchmarkDataLoader",
    "RealDataConfig",
    "DataQualityMetrics",
    
    # Preprocessing Scripts
    "DataCleaner",
    "FeatureExtractor",
    "DataAugmentation",
    "Normalization",
    "TimeSeriesSplitter",
    "PreprocessingConfig",
    "PreprocessingPipeline",
    
    # Validation Sets
    "TestScenarioGenerator",
    "HoldoutDataManager",
    "CrossValidationSplitter",
    "PerformanceBenchmarks",
    "ValidationConfig",
    "ValidationMetrics",
    
    # Generators
    "SyntheticGenerator",
    "PhysicsSimulator",
    "NoiseGenerator",
    "ScenarioBuilder",
    "GeneratorConfig",
    "GenerationMetrics"
]

# Default data configuration
DEFAULT_DATA_CONFIG = {
    "synthetic_generation": {
        "num_batteries": 10000,
        "simulation_duration_hours": 8760,  # 1 year
        "sampling_frequency_seconds": 60,   # 1 minute
        "battery_chemistries": ["LiIon", "LiFePO4", "NiMH"],
        "capacity_range_kwh": [10, 100],
        "voltage_range_v": [3.0, 4.2],
        "temperature_range_c": [-20, 60],
        "degradation_models": ["calendar", "cyclic", "combined"],
        "noise_levels": [0.01, 0.05, 0.1],  # 1%, 5%, 10%
        "physics_fidelity": "high"
    },
    "real_data_integration": {
        "tata_ev_data": True,
        "lab_test_data": True,
        "field_study_data": True,
        "benchmark_datasets": True,
        "data_quality_threshold": 0.95,
        "missing_data_tolerance": 0.05,
        "outlier_detection": True,
        "temporal_alignment": True
    },
    "preprocessing": {
        "normalization_method": "z_score",
        "feature_scaling": "min_max",
        "outlier_removal": "iqr",
        "missing_value_strategy": "interpolation",
        "noise_reduction": "savgol_filter",
        "feature_selection": "correlation_threshold",
        "dimensionality_reduction": "pca",
        "time_series_length": 1440  # 24 hours at 1-minute intervals
    },
    "validation": {
        "train_split": 0.7,
        "validation_split": 0.15,
        "test_split": 0.15,
        "cross_validation_folds": 5,
        "temporal_split": True,
        "stratified_sampling": True,
        "holdout_percentage": 0.1,
        "performance_metrics": ["mse", "mae", "r2", "mape"]
    },
    "data_quality": {
        "completeness_threshold": 0.95,
        "consistency_threshold": 0.9,
        "accuracy_threshold": 0.95,
        "timeliness_threshold": 0.9,
        "validity_threshold": 0.95,
        "uniqueness_threshold": 0.99,
        "automated_quality_checks": True,
        "quality_reporting": True
    }
}

def get_default_data_config():
    """
    Get default configuration for training data management.
    
    Returns:
        dict: Default data configuration
    """
    return DEFAULT_DATA_CONFIG.copy()

def create_synthetic_data_generator(config=None):
    """
    Factory function to create a synthetic data generator.
    
    Args:
        config (dict, optional): Generation configuration
        
    Returns:
        SyntheticGenerator: Configured synthetic data generator
    """
    if config is None:
        config = get_default_data_config()
    
    gen_config = GeneratorConfig(**config["synthetic_generation"])
    return SyntheticGenerator(gen_config)

def create_preprocessing_pipeline(config=None):
    """
    Factory function to create a preprocessing pipeline.
    
    Args:
        config (dict, optional): Preprocessing configuration
        
    Returns:
        PreprocessingPipeline: Configured preprocessing pipeline
    """
    if config is None:
        config = get_default_data_config()
    
    prep_config = PreprocessingConfig(**config["preprocessing"])
    return PreprocessingPipeline(prep_config)

def create_validation_framework(config=None):
    """
    Factory function to create a validation framework.
    
    Args:
        config (dict, optional): Validation configuration
        
    Returns:
        dict: Validation framework components
    """
    if config is None:
        config = get_default_data_config()
    
    val_config = ValidationConfig(**config["validation"])
    
    return {
        "cross_validator": CrossValidationSplitter(val_config),
        "holdout_manager": HoldoutDataManager(val_config),
        "benchmark_generator": PerformanceBenchmarks(val_config),
        "scenario_generator": TestScenarioGenerator(val_config)
    }

# Data generation scenarios
DATA_SCENARIOS = {
    "standard_ev_fleet": {
        "description": "Standard electric vehicle fleet simulation",
        "num_vehicles": 1000,
        "battery_capacity_range": [40, 100],  # kWh
        "usage_patterns": ["daily_commute", "long_distance", "urban"],
        "charging_patterns": ["home", "workplace", "public"],
        "environmental_conditions": "moderate",
        "degradation_rate": "normal"
    },
    "commercial_fleet": {
        "description": "Commercial vehicle fleet with heavy usage",
        "num_vehicles": 500,
        "battery_capacity_range": [80, 200],  # kWh
        "usage_patterns": ["delivery", "logistics", "construction"],
        "charging_patterns": ["depot", "fast_charging"],
        "environmental_conditions": "harsh",
        "degradation_rate": "accelerated"
    },
    "energy_storage_grid": {
        "description": "Grid-scale energy storage systems",
        "num_systems": 100,
        "battery_capacity_range": [1000, 10000],  # kWh
        "usage_patterns": ["peak_shaving", "load_balancing", "backup"],
        "charging_patterns": ["grid_tied", "renewable_integration"],
        "environmental_conditions": "controlled",
        "degradation_rate": "slow"
    },
    "consumer_electronics": {
        "description": "Consumer electronics battery simulation",
        "num_devices": 10000,
        "battery_capacity_range": [0.01, 0.1],  # kWh
        "usage_patterns": ["continuous", "intermittent", "standby"],
        "charging_patterns": ["fast_charge", "trickle_charge"],
        "environmental_conditions": "indoor",
        "degradation_rate": "moderate"
    },
    "extreme_conditions": {
        "description": "Extreme operating conditions testing",
        "num_batteries": 500,
        "battery_capacity_range": [10, 100],  # kWh
        "usage_patterns": ["extreme_cold", "extreme_heat", "high_vibration"],
        "charging_patterns": ["emergency", "rapid"],
        "environmental_conditions": "extreme",
        "degradation_rate": "rapid"
    }
}

def get_data_scenario(scenario_name):
    """
    Get predefined data generation scenario.
    
    Args:
        scenario_name (str): Name of the data scenario
        
    Returns:
        dict: Scenario configuration
    """
    return DATA_SCENARIOS.get(scenario_name, {})

def list_data_scenarios():
    """
    List available data generation scenarios.
    
    Returns:
        list: List of available scenario names
    """
    return list(DATA_SCENARIOS.keys())

# Battery chemistry specifications
BATTERY_CHEMISTRIES = {
    "lithium_ion": {
        "nominal_voltage": 3.7,
        "voltage_range": [3.0, 4.2],
        "energy_density": 250,  # Wh/kg
        "cycle_life": 1000,
        "calendar_life": 10,  # years
        "temperature_sensitivity": "medium",
        "degradation_mechanisms": ["SEI_growth", "lithium_plating", "active_material_loss"]
    },
    "lifepo4": {
        "nominal_voltage": 3.2,
        "voltage_range": [2.5, 3.65],
        "energy_density": 160,  # Wh/kg
        "cycle_life": 3000,
        "calendar_life": 15,  # years
        "temperature_sensitivity": "low",
        "degradation_mechanisms": ["active_material_loss", "electrolyte_decomposition"]
    },
    "nimh": {
        "nominal_voltage": 1.2,
        "voltage_range": [1.0, 1.4],
        "energy_density": 80,  # Wh/kg
        "cycle_life": 500,
        "calendar_life": 5,  # years
        "temperature_sensitivity": "high",
        "degradation_mechanisms": ["memory_effect", "corrosion", "hydrogen_evolution"]
    },
    "solid_state": {
        "nominal_voltage": 3.8,
        "voltage_range": [3.0, 4.5],
        "energy_density": 400,  # Wh/kg
        "cycle_life": 5000,
        "calendar_life": 20,  # years
        "temperature_sensitivity": "very_low",
        "degradation_mechanisms": ["interface_resistance", "dendrite_formation"]
    }
}

def get_battery_chemistry_specs(chemistry_name):
    """
    Get specifications for a battery chemistry.
    
    Args:
        chemistry_name (str): Name of the battery chemistry
        
    Returns:
        dict: Chemistry specifications
    """
    return BATTERY_CHEMISTRIES.get(chemistry_name, {})

def list_battery_chemistries():
    """
    List available battery chemistries.
    
    Returns:
        list: List of available chemistry names
    """
    return list(BATTERY_CHEMISTRIES.keys())

# Data quality metrics
DATA_QUALITY_METRICS = {
    "completeness": {
        "description": "Percentage of non-missing values",
        "calculation": "non_null_count / total_count",
        "threshold": 0.95,
        "critical": True
    },
    "consistency": {
        "description": "Consistency across related data points",
        "calculation": "consistent_relationships / total_relationships",
        "threshold": 0.9,
        "critical": True
    },
    "accuracy": {
        "description": "Correctness of data values",
        "calculation": "accurate_values / total_values",
        "threshold": 0.95,
        "critical": True
    },
    "timeliness": {
        "description": "Data freshness and temporal alignment",
        "calculation": "timely_data_points / total_data_points",
        "threshold": 0.9,
        "critical": False
    },
    "validity": {
        "description": "Conformance to data format and constraints",
        "calculation": "valid_values / total_values",
        "threshold": 0.95,
        "critical": True
    },
    "uniqueness": {
        "description": "Absence of duplicate records",
        "calculation": "unique_records / total_records",
        "threshold": 0.99,
        "critical": False
    }
}

def get_data_quality_metrics():
    """
    Get data quality metrics definitions.
    
    Returns:
        dict: Data quality metrics
    """
    return DATA_QUALITY_METRICS.copy()

# Data preprocessing techniques
PREPROCESSING_TECHNIQUES = {
    "normalization": {
        "z_score": "Standardize to zero mean and unit variance",
        "min_max": "Scale to [0, 1] range",
        "robust": "Use median and IQR for outlier resistance",
        "quantile": "Transform to uniform distribution"
    },
    "outlier_detection": {
        "iqr": "Interquartile range method",
        "isolation_forest": "Isolation forest algorithm",
        "local_outlier_factor": "Local outlier factor",
        "one_class_svm": "One-class SVM"
    },
    "missing_value_handling": {
        "interpolation": "Linear or spline interpolation",
        "forward_fill": "Forward fill missing values",
        "backward_fill": "Backward fill missing values",
        "mean_imputation": "Replace with mean value",
        "median_imputation": "Replace with median value",
        "knn_imputation": "K-nearest neighbors imputation"
    },
    "noise_reduction": {
        "savgol_filter": "Savitzky-Golay filter",
        "moving_average": "Moving average filter",
        "gaussian_filter": "Gaussian smoothing",
        "butterworth_filter": "Butterworth filter",
        "kalman_filter": "Kalman filter"
    }
}

def get_preprocessing_techniques():
    """
    Get available preprocessing techniques.
    
    Returns:
        dict: Preprocessing techniques
    """
    return PREPROCESSING_TECHNIQUES.copy()

# Integration utilities
def create_integrated_data_pipeline(transformer_config=None, rl_config=None, federated_config=None):
    """
    Create integrated data pipeline for multiple AI/ML models.
    
    Args:
        transformer_config (dict, optional): Transformer model data requirements
        rl_config (dict, optional): RL model data requirements
        federated_config (dict, optional): Federated learning data requirements
        
    Returns:
        dict: Integrated data pipeline configuration
    """
    base_config = get_default_data_config()
    
    # Enhanced configuration for multi-model integration
    base_config["integration"] = {
        "multi_model_support": True,
        "shared_preprocessing": True,
        "cross_model_validation": True,
        "unified_data_format": True
    }
    
    if transformer_config:
        base_config["transformer_data"] = {
            "sequence_length": transformer_config.get("sequence_length", 1440),
            "feature_engineering": True,
            "attention_data_prep": True,
            "temporal_features": True
        }
    
    if rl_config:
        base_config["rl_data"] = {
            "state_action_pairs": True,
            "reward_calculation": True,
            "environment_simulation": True,
            "episode_segmentation": True
        }
    
    if federated_config:
        base_config["federated_data"] = {
            "client_data_distribution": True,
            "privacy_preserving_splits": True,
            "non_iid_simulation": True,
            "differential_privacy": True
        }
    
    return base_config

# Validation utilities
def validate_data_config(config):
    """
    Validate data configuration parameters.
    
    Args:
        config (dict): Data configuration
        
    Returns:
        dict: Validation results
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check required sections
    required_sections = ["synthetic_generation", "preprocessing", "validation"]
    for section in required_sections:
        if section not in config:
            validation_results["errors"].append(f"Missing required section: {section}")
            validation_results["valid"] = False
    
    # Validate synthetic generation parameters
    if "synthetic_generation" in config:
        syn_gen = config["synthetic_generation"]
        if syn_gen.get("num_batteries", 0) <= 0:
            validation_results["errors"].append("Number of batteries must be positive")
            validation_results["valid"] = False
        
        if syn_gen.get("simulation_duration_hours", 0) <= 0:
            validation_results["errors"].append("Simulation duration must be positive")
            validation_results["valid"] = False
    
    # Validate split ratios
    if "validation" in config:
        val_config = config["validation"]
        splits = [val_config.get("train_split", 0), 
                 val_config.get("validation_split", 0),
                 val_config.get("test_split", 0)]
        
        if abs(sum(splits) - 1.0) > 0.01:
            validation_results["errors"].append("Train/validation/test splits must sum to 1.0")
            validation_results["valid"] = False
    
    return validation_results

def estimate_data_requirements(config):
    """
    Estimate storage and computational requirements for data generation.
    
    Args:
        config (dict): Data configuration
        
    Returns:
        dict: Resource estimates
    """
    num_batteries = config.get("synthetic_generation", {}).get("num_batteries", 10000)
    duration_hours = config.get("synthetic_generation", {}).get("simulation_duration_hours", 8760)
    sampling_freq = config.get("synthetic_generation", {}).get("sampling_frequency_seconds", 60)
    
    # Calculate data points
    data_points_per_battery = duration_hours * 3600 / sampling_freq
    total_data_points = num_batteries * data_points_per_battery
    
    # Estimate storage (assuming 20 features per data point, 4 bytes per float)
    storage_gb = total_data_points * 20 * 4 / (1024**3)
    
    estimated_requirements = {
        "total_data_points": int(total_data_points),
        "storage_requirement_gb": round(storage_gb, 2),
        "generation_time_hours": num_batteries * 0.01,  # Rough estimate
        "memory_requirement_gb": max(8, num_batteries * 0.001),
        "recommended_cpu_cores": min(16, max(4, num_batteries // 1000))
    }
    
    return estimated_requirements

# Module health check
def health_check():
    """
    Perform a health check of the training data module.
    
    Returns:
        dict: Health check results
    """
    health_status = {
        "module_loaded": True,
        "version": __version__,
        "components_available": {
            "synthetic_datasets": True,
            "real_world_samples": True,
            "preprocessing_scripts": True,
            "validation_sets": True,
            "generators": True
        },
        "dependencies_satisfied": True
    }
    
    try:
        # Test basic functionality
        config = get_default_data_config()
        validation_results = validate_data_config(config)
        health_status["config_validation"] = validation_results["valid"]
    except Exception as e:
        health_status["config_validation"] = False
        health_status["error"] = str(e)
    
    return health_status

# Export configuration template
def export_data_config_template(file_path="data_config.yaml"):
    """
    Export a data configuration template.
    
    Args:
        file_path (str): Path to save the configuration template
    """
    import yaml
    
    config_template = get_default_data_config()
    config_template["_metadata"] = {
        "version": __version__,
        "description": "BatteryMind Training Data Configuration Template",
        "author": __author__,
        "created": "2025-07-11"
    }
    
    with open(file_path, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False, indent=2)

# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BatteryMind Training Data Module v{__version__} initialized")
