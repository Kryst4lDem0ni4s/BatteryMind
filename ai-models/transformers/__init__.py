"""
BatteryMind - Transformers Module

Comprehensive transformer-based AI models for battery intelligence and optimization.
This module provides state-of-the-art transformer architectures specifically designed
for battery health prediction, degradation forecasting, and optimization recommendations.

Key Components:
- Battery Health Predictor: Real-time battery health assessment and SoH prediction
- Degradation Forecaster: Long-term degradation pattern forecasting with uncertainty
- Optimization Recommender: AI-driven battery optimization and maintenance recommendations
- Ensemble Models: Combined model approaches for enhanced accuracy and robustness
- Common Utilities: Shared components for attention mechanisms, encoding, and utilities

Features:
- Multi-modal sensor data processing (voltage, current, temperature, usage patterns)
- Physics-informed constraints for realistic predictions
- Uncertainty quantification with prediction intervals
- Real-time inference optimization for production deployment
- Integration with federated learning and reinforcement learning systems
- Comprehensive monitoring and evaluation frameworks

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

# Core transformer models
import time
import numpy as np
import torch

from .battery_health_predictor import (
    BatteryHealthTransformer,
    BatteryHealthConfig,
    BatteryHealthTrainer,
    BatteryHealthPredictor,
    BatteryDataLoader,
    BatteryPreprocessor,
    create_battery_health_model,
    create_battery_trainer,
    create_battery_predictor
)

from .degradation_forecaster import (
    DegradationForecaster,
    DegradationConfig,
    DegradationTrainer,
    BatteryDegradationForecaster,
    TimeSeriesProcessor,
    create_degradation_forecaster,
    create_degradation_trainer,
    create_battery_degradation_forecaster
)

from .optimization_recommender import (
    OptimizationRecommender,
    OptimizationConfig,
    OptimizationTrainer,
    BatteryOptimizationRecommender,
    OptimizationUtils,
    create_optimization_recommender,
    create_optimization_trainer,
    create_battery_optimization_recommender
)

from .ensemble_model import (
    BatteryEnsemble,
    VotingClassifier,
    StackingRegressor,
    ModelFusion,
    create_battery_ensemble,
    create_voting_ensemble,
    create_stacking_ensemble
)

# Common utilities and base classes
from .common import (
    BaseTransformerModel,
    MultiHeadAttention,
    PositionalEncoding,
    TransformerUtils,
    AttentionVisualization,
    ModelRegistry,
    ConfigManager
)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module metadata
__all__ = [
    # Battery Health Predictor
    "BatteryHealthTransformer",
    "BatteryHealthConfig",
    "BatteryHealthTrainer",
    "BatteryHealthPredictor",
    "BatteryDataLoader",
    "BatteryPreprocessor",
    "create_battery_health_model",
    "create_battery_trainer",
    "create_battery_predictor",
    
    # Degradation Forecaster
    "DegradationForecaster",
    "DegradationConfig",
    "DegradationTrainer",
    "BatteryDegradationForecaster",
    "TimeSeriesProcessor",
    "create_degradation_forecaster",
    "create_degradation_trainer",
    "create_battery_degradation_forecaster",
    
    # Optimization Recommender
    "OptimizationRecommender",
    "OptimizationConfig",
    "OptimizationTrainer",
    "BatteryOptimizationRecommender",
    "OptimizationUtils",
    "create_optimization_recommender",
    "create_optimization_trainer",
    "create_battery_optimization_recommender",
    
    # Ensemble Models
    "BatteryEnsemble",
    "VotingClassifier",
    "StackingRegressor",
    "ModelFusion",
    "create_battery_ensemble",
    "create_voting_ensemble",
    "create_stacking_ensemble",
    
    # Common Utilities
    "BaseTransformerModel",
    "MultiHeadAttention",
    "PositionalEncoding",
    "TransformerUtils",
    "AttentionVisualization",
    "ModelRegistry",
    "ConfigManager"
]

# Global configuration for all transformer models
GLOBAL_TRANSFORMER_CONFIG = {
    "default_device": "auto",  # Auto-detect CUDA/CPU
    "mixed_precision": True,
    "gradient_checkpointing": False,
    "attention_dropout": 0.1,
    "hidden_dropout": 0.1,
    "layer_norm_eps": 1e-6,
    "initializer_range": 0.02,
    "max_position_embeddings": 2048,
    "type_vocab_size": 2,
    "use_cache": True,
    "torch_dtype": "float32",
    "transformers_version": "4.21.0"
}

# Model registry for tracking and managing different transformer models
MODEL_REGISTRY = {
    "battery_health_predictor": {
        "model_class": "BatteryHealthTransformer",
        "config_class": "BatteryHealthConfig",
        "trainer_class": "BatteryHealthTrainer",
        "predictor_class": "BatteryHealthPredictor",
        "description": "Real-time battery health assessment and State of Health prediction",
        "input_modalities": ["voltage", "current", "temperature", "usage_patterns"],
        "output_types": ["state_of_health", "degradation_patterns"],
        "use_cases": ["real_time_monitoring", "predictive_maintenance", "warranty_analysis"]
    },
    "degradation_forecaster": {
        "model_class": "DegradationForecaster",
        "config_class": "DegradationConfig",
        "trainer_class": "DegradationTrainer",
        "predictor_class": "BatteryDegradationForecaster",
        "description": "Long-term degradation pattern forecasting with uncertainty quantification",
        "input_modalities": ["time_series_data", "environmental_factors", "usage_history"],
        "output_types": ["degradation_forecasts", "uncertainty_intervals", "seasonal_patterns"],
        "use_cases": ["long_term_planning", "lifecycle_optimization", "replacement_scheduling"]
    },
    "optimization_recommender": {
        "model_class": "OptimizationRecommender",
        "config_class": "OptimizationConfig",
        "trainer_class": "OptimizationTrainer",
        "predictor_class": "BatteryOptimizationRecommender",
        "description": "AI-driven battery optimization and maintenance recommendations",
        "input_modalities": ["current_state", "usage_patterns", "environmental_conditions"],
        "output_types": ["optimization_actions", "maintenance_recommendations", "efficiency_improvements"],
        "use_cases": ["performance_optimization", "energy_efficiency", "operational_guidance"]
    },
    "ensemble_model": {
        "model_class": "BatteryEnsemble",
        "config_class": "EnsembleConfig",
        "trainer_class": "EnsembleTrainer",
        "predictor_class": "BatteryEnsemblePredictor",
        "description": "Combined model approaches for enhanced accuracy and robustness",
        "input_modalities": ["multi_model_inputs"],
        "output_types": ["ensemble_predictions", "confidence_scores", "model_agreements"],
        "use_cases": ["high_accuracy_prediction", "robust_decision_making", "uncertainty_reduction"]
    }
}

def get_global_config():
    """
    Get global configuration for transformer models.
    
    Returns:
        dict: Global configuration dictionary
    """
    return GLOBAL_TRANSFORMER_CONFIG.copy()

def get_model_registry():
    """
    Get model registry with information about available transformer models.
    
    Returns:
        dict: Model registry dictionary
    """
    return MODEL_REGISTRY.copy()

def create_integrated_battery_system(config=None):
    """
    Create an integrated battery intelligence system combining all transformer models.
    
    Args:
        config (dict, optional): System configuration
        
    Returns:
        dict: Integrated system with all models and utilities
    """
    if config is None:
        config = get_global_config()
    
    # Create individual models
    health_model = create_battery_health_model()
    forecaster_model = create_degradation_forecaster()
    optimizer_model = create_optimization_recommender()
    ensemble_model = create_battery_ensemble([health_model, forecaster_model, optimizer_model])
    
    # Create integrated system
    integrated_system = {
        "health_predictor": health_model,
        "degradation_forecaster": forecaster_model,
        "optimization_recommender": optimizer_model,
        "ensemble_model": ensemble_model,
        "config": config,
        "version": __version__,
        "capabilities": {
            "real_time_health_monitoring": True,
            "long_term_forecasting": True,
            "optimization_recommendations": True,
            "ensemble_predictions": True,
            "uncertainty_quantification": True,
            "physics_informed_constraints": True
        }
    }
    
    return integrated_system

def validate_model_compatibility(model1, model2):
    """
    Validate compatibility between different transformer models.
    
    Args:
        model1: First model to compare
        model2: Second model to compare
        
    Returns:
        dict: Compatibility analysis results
    """
    compatibility = {
        "compatible": True,
        "issues": [],
        "recommendations": []
    }
    
    # Check input/output dimensions
    if hasattr(model1, 'config') and hasattr(model2, 'config'):
        if model1.config.feature_dim != model2.config.feature_dim:
            compatibility["issues"].append("Feature dimension mismatch")
            compatibility["recommendations"].append("Use feature projection layers")
        
        if model1.config.d_model != model2.config.d_model:
            compatibility["issues"].append("Model dimension mismatch")
            compatibility["recommendations"].append("Add dimension adaptation layers")
    
    # Check device compatibility
    device1 = next(model1.parameters()).device if hasattr(model1, 'parameters') else None
    device2 = next(model2.parameters()).device if hasattr(model2, 'parameters') else None
    
    if device1 and device2 and device1 != device2:
        compatibility["issues"].append("Device mismatch")
        compatibility["recommendations"].append("Move models to same device")
    
    compatibility["compatible"] = len(compatibility["issues"]) == 0
    
    return compatibility

def optimize_transformer_performance(model, optimization_config=None):
    """
    Apply performance optimizations to transformer models.
    
    Args:
        model: Transformer model to optimize
        optimization_config (dict, optional): Optimization configuration
        
    Returns:
        torch.nn.Module: Optimized model
    """
    import torch
    
    if optimization_config is None:
        optimization_config = {
            "enable_torch_compile": True,
            "enable_gradient_checkpointing": False,
            "enable_mixed_precision": True,
            "optimize_for_inference": True
        }
    
    # Apply torch.compile if available (PyTorch 2.0+)
    if optimization_config.get("enable_torch_compile", False) and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="default")
            logger.info("Applied torch.compile optimization")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    
    # Enable gradient checkpointing for memory efficiency
    if optimization_config.get("enable_gradient_checkpointing", False):
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
    
    # Set model to evaluation mode for inference optimization
    if optimization_config.get("optimize_for_inference", False):
        model.eval()
        torch.backends.cudnn.benchmark = True
        logger.info("Optimized model for inference")
    
    return model

def get_model_summary(model_name=None):
    """
    Get comprehensive summary of transformer models.
    
    Args:
        model_name (str, optional): Specific model name to summarize
        
    Returns:
        dict: Model summary information
    """
    if model_name:
        if model_name in MODEL_REGISTRY:
            return MODEL_REGISTRY[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found in registry")
    
    # Return summary of all models
    summary = {
        "total_models": len(MODEL_REGISTRY),
        "models": MODEL_REGISTRY,
        "global_config": GLOBAL_TRANSFORMER_CONFIG,
        "version": __version__,
        "capabilities": {
            "supported_modalities": ["voltage", "current", "temperature", "usage_patterns", "environmental_factors"],
            "prediction_types": ["health_assessment", "degradation_forecasting", "optimization_recommendations"],
            "uncertainty_quantification": True,
            "physics_informed_modeling": True,
            "real_time_inference": True,
            "batch_processing": True,
            "ensemble_methods": True
        }
    }
    
    return summary

def health_check():
    """
    Perform comprehensive health check of the transformers module.
    
    Returns:
        dict: Health check results
    """
    health_status = {
        "module_status": "healthy",
        "version": __version__,
        "models_available": True,
        "dependencies_satisfied": True,
        "gpu_available": torch.cuda.is_available() if 'torch' in globals() else False,
        "model_registry_valid": True,
        "issues": []
    }
    
    try:
        # Test model creation
        test_health_model = create_battery_health_model()
        health_status["health_model_creation"] = True
    except Exception as e:
        health_status["health_model_creation"] = False
        health_status["issues"].append(f"Health model creation failed: {str(e)}")
    
    try:
        # Test forecaster creation
        test_forecaster_model = create_degradation_forecaster()
        health_status["forecaster_model_creation"] = True
    except Exception as e:
        health_status["forecaster_model_creation"] = False
        health_status["issues"].append(f"Forecaster model creation failed: {str(e)}")
    
    try:
        # Test optimizer creation
        test_optimizer_model = create_optimization_recommender()
        health_status["optimizer_model_creation"] = True
    except Exception as e:
        health_status["optimizer_model_creation"] = False
        health_status["issues"].append(f"Optimizer model creation failed: {str(e)}")
    
    # Determine overall status
    if health_status["issues"]:
        health_status["module_status"] = "degraded" if len(health_status["issues"]) < 3 else "unhealthy"
    
    return health_status

# Module initialization and compatibility checks
def _check_dependencies():
    """Check if all required dependencies are available."""
    required_packages = [
        'torch', 'numpy', 'pandas', 'sklearn', 'scipy',
        'transformers', 'datasets', 'tokenizers'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {missing_packages}")
        return False
    
    return True

# Initialize module
import logging
logger = logging.getLogger(__name__)

# Check dependencies on import
_dependencies_available = _check_dependencies()
if not _dependencies_available:
    logger.warning("Some dependencies are missing. Some functionality may be limited.")

# Log successful initialization
logger.info(f"BatteryMind Transformers Module v{__version__} initialized successfully")
logger.info(f"Available models: {list(MODEL_REGISTRY.keys())}")

# Export configuration templates
def export_all_configs(output_dir="./configs"):
    """
    Export configuration templates for all transformer models.
    
    Args:
        output_dir (str): Directory to save configuration files
    """
    from pathlib import Path
    import yaml
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export global config
    with open(output_path / "global_transformer_config.yaml", 'w') as f:
        yaml.dump(GLOBAL_TRANSFORMER_CONFIG, f, default_flow_style=False, indent=2)
    
    # Export model registry
    with open(output_path / "model_registry.yaml", 'w') as f:
        yaml.dump(MODEL_REGISTRY, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration templates exported to {output_dir}")

# Performance monitoring
class TransformerPerformanceMonitor:
    """Monitor performance across all transformer models."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def log_inference_time(self, model_name, inference_time):
        """Log inference time for a model."""
        if model_name not in self.metrics:
            self.metrics[model_name] = {"inference_times": []}
        self.metrics[model_name]["inference_times"].append(inference_time)
    
    def get_performance_summary(self):
        """Get performance summary across all models."""
        summary = {}
        for model_name, metrics in self.metrics.items():
            if "inference_times" in metrics:
                times = metrics["inference_times"]
                summary[model_name] = {
                    "avg_inference_time": np.mean(times),
                    "min_inference_time": np.min(times),
                    "max_inference_time": np.max(times),
                    "total_inferences": len(times)
                }
        return summary

# Global performance monitor instance
performance_monitor = TransformerPerformanceMonitor()

def get_performance_monitor():
    """Get the global performance monitor instance."""
    return performance_monitor
