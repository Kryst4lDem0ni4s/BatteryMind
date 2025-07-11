"""
BatteryMind - Ensemble Model Module

Advanced ensemble modeling framework that combines multiple AI models
(transformer health predictor, degradation forecaster, optimization recommender)
to provide robust, accurate, and comprehensive battery management decisions.

This module implements sophisticated ensemble techniques including voting,
stacking, and model fusion approaches specifically designed for battery
management applications with uncertainty quantification and decision fusion.

Key Components:
- BatteryEnsemble: Main ensemble model combining multiple specialized models
- VotingClassifier: Voting-based ensemble for classification tasks
- StackingRegressor: Stacking ensemble for regression predictions
- ModelFusion: Advanced model fusion with attention mechanisms
- EnsembleOptimizer: Optimization for ensemble weights and architecture

Features:
- Multi-model integration with heterogeneous architectures
- Uncertainty quantification and confidence estimation
- Adaptive ensemble weights based on model performance
- Real-time ensemble inference with load balancing
- Comprehensive ensemble evaluation and monitoring
- Integration with battery health, forecasting, and optimization models

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

from .ensemble import (
    BatteryEnsemble,
    EnsembleConfig,
    EnsembleResult,
    EnsembleMetrics,
    AdaptiveWeighting,
    UncertaintyAggregation
)

from .voting_classifier import (
    BatteryVotingClassifier,
    VotingConfig,
    VotingStrategy,
    WeightedVoting,
    MajorityVoting,
    SoftVoting
)

from .stacking_regressor import (
    BatteryStackingRegressor,
    StackingConfig,
    MetaLearner,
    BaseModelManager,
    StackingOptimizer
)

from .model_fusion import (
    BatteryModelFusion,
    FusionConfig,
    AttentionFusion,
    FeatureFusion,
    DecisionFusion,
    MultiModalFusion
)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module metadata
__all__ = [
    # Main ensemble components
    "BatteryEnsemble",
    "EnsembleConfig",
    "EnsembleResult",
    "EnsembleMetrics",
    "AdaptiveWeighting",
    "UncertaintyAggregation",
    
    # Voting ensemble components
    "BatteryVotingClassifier",
    "VotingConfig",
    "VotingStrategy",
    "WeightedVoting",
    "MajorityVoting",
    "SoftVoting",
    
    # Stacking ensemble components
    "BatteryStackingRegressor",
    "StackingConfig",
    "MetaLearner",
    "BaseModelManager",
    "StackingOptimizer",
    
    # Model fusion components
    "BatteryModelFusion",
    "FusionConfig",
    "AttentionFusion",
    "FeatureFusion",
    "DecisionFusion",
    "MultiModalFusion"
]

# Module configuration
DEFAULT_ENSEMBLE_CONFIG = {
    "ensemble": {
        "ensemble_type": "adaptive_weighted",
        "base_models": [
            "battery_health_predictor",
            "degradation_forecaster", 
            "optimization_recommender"
        ],
        "voting_strategy": "soft",
        "uncertainty_aggregation": "bayesian",
        "adaptive_weights": True,
        "confidence_threshold": 0.8
    },
    "voting": {
        "voting_type": "weighted",
        "weight_optimization": "performance_based",
        "diversity_bonus": 0.1,
        "confidence_weighting": True
    },
    "stacking": {
        "meta_learner": "gradient_boosting",
        "cross_validation_folds": 5,
        "feature_selection": True,
        "regularization": 0.01
    },
    "fusion": {
        "fusion_method": "attention_based",
        "attention_heads": 4,
        "feature_fusion": True,
        "decision_fusion": True,
        "uncertainty_fusion": True
    },
    "performance": {
        "parallel_inference": True,
        "model_caching": True,
        "load_balancing": True,
        "timeout_seconds": 30
    }
}

def get_default_ensemble_config():
    """
    Get default configuration for ensemble models.
    
    Returns:
        dict: Default configuration dictionary
    """
    return DEFAULT_ENSEMBLE_CONFIG.copy()

def create_battery_ensemble(base_models, config=None):
    """
    Factory function to create a battery ensemble model.
    
    Args:
        base_models (List): List of base model instances or paths
        config (dict, optional): Ensemble configuration. If None, uses default config.
        
    Returns:
        BatteryEnsemble: Configured ensemble model instance
    """
    if config is None:
        config = get_default_ensemble_config()
    
    ensemble_config = EnsembleConfig(**config["ensemble"])
    return BatteryEnsemble(base_models, ensemble_config)

def create_voting_classifier(base_models, config=None):
    """
    Factory function to create a voting classifier ensemble.
    
    Args:
        base_models (List): List of base model instances
        config (dict, optional): Voting configuration. If None, uses default config.
        
    Returns:
        BatteryVotingClassifier: Configured voting classifier instance
    """
    if config is None:
        config = get_default_ensemble_config()
    
    voting_config = VotingConfig(**config["voting"])
    return BatteryVotingClassifier(base_models, voting_config)

def create_stacking_regressor(base_models, meta_learner=None, config=None):
    """
    Factory function to create a stacking regressor ensemble.
    
    Args:
        base_models (List): List of base model instances
        meta_learner: Meta-learner model instance
        config (dict, optional): Stacking configuration. If None, uses default config.
        
    Returns:
        BatteryStackingRegressor: Configured stacking regressor instance
    """
    if config is None:
        config = get_default_ensemble_config()
    
    stacking_config = StackingConfig(**config["stacking"])
    return BatteryStackingRegressor(base_models, meta_learner, stacking_config)

def create_model_fusion(base_models, config=None):
    """
    Factory function to create a model fusion ensemble.
    
    Args:
        base_models (List): List of base model instances
        config (dict, optional): Fusion configuration. If None, uses default config.
        
    Returns:
        BatteryModelFusion: Configured model fusion instance
    """
    if config is None:
        config = get_default_ensemble_config()
    
    fusion_config = FusionConfig(**config["fusion"])
    return BatteryModelFusion(base_models, fusion_config)

# Ensemble strategies and utilities
ENSEMBLE_STRATEGIES = {
    "voting": {
        "majority": "Simple majority voting for classification",
        "weighted": "Weighted voting based on model performance",
        "soft": "Soft voting using prediction probabilities",
        "adaptive": "Adaptive voting with dynamic weight adjustment"
    },
    "stacking": {
        "linear": "Linear meta-learner for stacking",
        "gradient_boosting": "Gradient boosting meta-learner",
        "neural_network": "Neural network meta-learner",
        "random_forest": "Random forest meta-learner"
    },
    "fusion": {
        "attention": "Attention-based model fusion",
        "feature_level": "Feature-level fusion before prediction",
        "decision_level": "Decision-level fusion after prediction",
        "hierarchical": "Hierarchical fusion with multiple levels"
    }
}

def get_ensemble_strategies():
    """
    Get available ensemble strategies.
    
    Returns:
        dict: Dictionary of ensemble strategies and descriptions
    """
    return ENSEMBLE_STRATEGIES.copy()

# Model compatibility matrix
MODEL_COMPATIBILITY = {
    "battery_health_predictor": {
        "output_type": "regression",
        "output_shape": (4,),  # SoH + 3 degradation metrics
        "uncertainty": True,
        "real_time": True
    },
    "degradation_forecaster": {
        "output_type": "time_series",
        "output_shape": (168, 6),  # 168 hours, 6 degradation metrics
        "uncertainty": True,
        "real_time": False
    },
    "optimization_recommender": {
        "output_type": "multi_class",
        "output_shape": (12,),  # 12 recommendation categories
        "uncertainty": True,
        "real_time": True
    }
}

def check_model_compatibility(models):
    """
    Check compatibility between models for ensemble creation.
    
    Args:
        models (List): List of model names or instances
        
    Returns:
        dict: Compatibility analysis results
    """
    compatibility_results = {
        "compatible": True,
        "warnings": [],
        "recommendations": []
    }
    
    model_types = []
    for model in models:
        model_name = model if isinstance(model, str) else model.__class__.__name__.lower()
        
        if model_name in MODEL_COMPATIBILITY:
            model_info = MODEL_COMPATIBILITY[model_name]
            model_types.append(model_info["output_type"])
            
            if not model_info["uncertainty"]:
                compatibility_results["warnings"].append(
                    f"Model {model_name} does not support uncertainty quantification"
                )
            
            if not model_info["real_time"]:
                compatibility_results["warnings"].append(
                    f"Model {model_name} may not be suitable for real-time inference"
                )
    
    # Check for mixed output types
    unique_types = set(model_types)
    if len(unique_types) > 1:
        compatibility_results["warnings"].append(
            f"Mixed output types detected: {unique_types}. Consider using model fusion."
        )
        compatibility_results["recommendations"].append(
            "Use BatteryModelFusion for heterogeneous model integration"
        )
    
    return compatibility_results

# Performance optimization utilities
def optimize_ensemble_weights(ensemble, validation_data, metric="accuracy"):
    """
    Optimize ensemble weights based on validation performance.
    
    Args:
        ensemble: Ensemble model instance
        validation_data: Validation dataset
        metric (str): Optimization metric
        
    Returns:
        dict: Optimized weights and performance metrics
    """
    from scipy.optimize import minimize
    import numpy as np
    
    def objective(weights):
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Set ensemble weights
        ensemble.set_weights(weights)
        
        # Evaluate performance
        predictions = ensemble.predict(validation_data)
        performance = ensemble.evaluate(predictions, validation_data.targets, metric)
        
        # Return negative performance for minimization
        return -performance
    
    # Initial weights (equal weighting)
    n_models = len(ensemble.base_models)
    initial_weights = np.ones(n_models) / n_models
    
    # Constraints: weights sum to 1 and are non-negative
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_models)]
    
    # Optimize weights
    result = minimize(
        objective, 
        initial_weights, 
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return {
        'optimal_weights': result.x,
        'performance': -result.fun,
        'optimization_success': result.success
    }

# Integration utilities
def create_integrated_battery_system(health_model_path, forecasting_model_path, 
                                   optimization_model_path, config=None):
    """
    Create an integrated battery management system with all models.
    
    Args:
        health_model_path (str): Path to health prediction model
        forecasting_model_path (str): Path to forecasting model
        optimization_model_path (str): Path to optimization model
        config (dict, optional): Integration configuration
        
    Returns:
        dict: Integrated system with ensemble and individual models
    """
    from ..battery_health_predictor import create_battery_predictor
    from ..degradation_forecaster import create_battery_degradation_forecaster
    from ..optimization_recommender import create_optimization_recommender
    
    # Load individual models
    health_predictor = create_battery_predictor(health_model_path)
    degradation_forecaster = create_battery_degradation_forecaster(forecasting_model_path)
    optimization_recommender = create_optimization_recommender(optimization_model_path)
    
    # Create ensemble
    base_models = [health_predictor, degradation_forecaster, optimization_recommender]
    ensemble = create_battery_ensemble(base_models, config)
    
    return {
        'ensemble': ensemble,
        'health_predictor': health_predictor,
        'degradation_forecaster': degradation_forecaster,
        'optimization_recommender': optimization_recommender,
        'version': __version__
    }

# Evaluation and benchmarking
def benchmark_ensemble_performance(ensemble, test_datasets, metrics=None):
    """
    Comprehensive benchmarking of ensemble performance.
    
    Args:
        ensemble: Ensemble model instance
        test_datasets: Dictionary of test datasets
        metrics (List, optional): List of metrics to evaluate
        
    Returns:
        dict: Comprehensive performance benchmarks
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    benchmark_results = {
        'ensemble_performance': {},
        'individual_model_performance': {},
        'ensemble_vs_individual': {},
        'computational_metrics': {}
    }
    
    import time
    
    for dataset_name, dataset in test_datasets.items():
        # Ensemble performance
        start_time = time.time()
        ensemble_predictions = ensemble.predict(dataset)
        ensemble_time = time.time() - start_time
        
        ensemble_scores = {}
        for metric in metrics:
            score = ensemble.evaluate(ensemble_predictions, dataset.targets, metric)
            ensemble_scores[metric] = score
        
        benchmark_results['ensemble_performance'][dataset_name] = ensemble_scores
        benchmark_results['computational_metrics'][dataset_name] = {
            'inference_time': ensemble_time,
            'predictions_per_second': len(dataset) / ensemble_time
        }
        
        # Individual model performance
        individual_scores = {}
        for i, model in enumerate(ensemble.base_models):
            model_predictions = model.predict(dataset)
            model_scores = {}
            for metric in metrics:
                score = model.evaluate(model_predictions, dataset.targets, metric)
                model_scores[metric] = score
            individual_scores[f'model_{i}'] = model_scores
        
        benchmark_results['individual_model_performance'][dataset_name] = individual_scores
        
        # Ensemble vs individual comparison
        ensemble_vs_individual = {}
        for metric in metrics:
            ensemble_score = ensemble_scores[metric]
            individual_scores_metric = [scores[metric] for scores in individual_scores.values()]
            best_individual = max(individual_scores_metric)
            improvement = (ensemble_score - best_individual) / best_individual * 100
            ensemble_vs_individual[metric] = {
                'ensemble_score': ensemble_score,
                'best_individual_score': best_individual,
                'improvement_percent': improvement
            }
        
        benchmark_results['ensemble_vs_individual'][dataset_name] = ensemble_vs_individual
    
    return benchmark_results

# Module health check and diagnostics
def health_check():
    """
    Perform health check of the ensemble model module.
    
    Returns:
        dict: Health check results
    """
    health_status = {
        'module_loaded': True,
        'version': __version__,
        'dependencies_available': True,
        'configuration_valid': True,
        'factory_functions_working': True
    }
    
    try:
        # Test configuration loading
        config = get_default_ensemble_config()
        health_status['default_config_loaded'] = True
        
        # Test factory functions
        from unittest.mock import Mock
        mock_models = [Mock(), Mock(), Mock()]
        
        ensemble = create_battery_ensemble(mock_models, config)
        health_status['ensemble_creation'] = True
        
        voting_classifier = create_voting_classifier(mock_models, config)
        health_status['voting_classifier_creation'] = True
        
        stacking_regressor = create_stacking_regressor(mock_models, Mock(), config)
        health_status['stacking_regressor_creation'] = True
        
        model_fusion = create_model_fusion(mock_models, config)
        health_status['model_fusion_creation'] = True
        
    except Exception as e:
        health_status['factory_functions_working'] = False
        health_status['error'] = str(e)
        logger.error(f"Ensemble module health check failed: {e}")
    
    return health_status

# Export utilities
def export_ensemble_config(ensemble, file_path="ensemble_config.yaml"):
    """
    Export ensemble configuration to YAML file.
    
    Args:
        ensemble: Ensemble model instance
        file_path (str): Path to save configuration
    """
    import yaml
    
    config = {
        'ensemble_type': ensemble.__class__.__name__,
        'base_models': [model.__class__.__name__ for model in ensemble.base_models],
        'configuration': ensemble.config.__dict__,
        'version': __version__,
        'export_timestamp': time.time()
    }
    
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Ensemble configuration exported to {file_path}")

# Module initialization logging
import logging
import time
logger = logging.getLogger(__name__)
logger.info(f"BatteryMind Ensemble Model v{__version__} initialized")

# Performance monitoring
PERFORMANCE_METRICS = {
    'module_load_time': 0.0,
    'factory_calls': 0,
    'ensemble_creations': 0,
    'health_checks': 0
}

def get_performance_metrics():
    """
    Get module performance metrics.
    
    Returns:
        dict: Performance metrics
    """
    return PERFORMANCE_METRICS.copy()

# Module constants
ENSEMBLE_CONSTANTS = {
    'MAX_BASE_MODELS': 10,
    'MIN_BASE_MODELS': 2,
    'DEFAULT_CONFIDENCE_THRESHOLD': 0.8,
    'MAX_ENSEMBLE_DEPTH': 3,
    'SUPPORTED_OUTPUT_TYPES': ['regression', 'classification', 'time_series', 'multi_class'],
    'SUPPORTED_FUSION_METHODS': ['voting', 'stacking', 'attention', 'feature_level', 'decision_level']
}

def get_ensemble_constants():
    """
    Get ensemble-specific constants.
    
    Returns:
        dict: Dictionary of ensemble constants
    """
    return ENSEMBLE_CONSTANTS.copy()
