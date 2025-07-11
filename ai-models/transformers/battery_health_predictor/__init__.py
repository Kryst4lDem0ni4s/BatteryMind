"""
BatteryMind - Battery Health Predictor Module

This module provides transformer-based battery health prediction capabilities
for the BatteryMind autonomous intelligence platform. It includes model
architecture, training pipeline, and inference components for predicting
battery State of Health (SoH) and degradation patterns.

Key Components:
- BatteryHealthTransformer: Main transformer model for health prediction
- BatteryHealthTrainer: Training pipeline with advanced optimization
- BatteryHealthPredictor: Production inference engine
- BatteryDataLoader: Specialized data loading for battery telemetry
- BatteryPreprocessor: Data preprocessing and feature engineering

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

from .model import (
    BatteryHealthTransformer,
    BatteryHealthConfig,
    MultiHeadBatteryAttention,
    BatteryPositionalEncoding,
    BatteryTransformerBlock
)

from .trainer import (
    BatteryHealthTrainer,
    BatteryTrainingConfig,
    BatteryTrainingMetrics,
    BatteryLossFunction,
    BatteryOptimizer
)

from .predictor import (
    BatteryHealthPredictor,
    BatteryPredictionResult,
    BatteryHealthMetrics,
    BatteryInferenceConfig
)

from .data_loader import (
    BatteryDataLoader,
    BatteryDataset,
    BatterySequenceCollator,
    BatteryDataConfig
)

from .preprocessing import (
    BatteryPreprocessor,
    BatteryFeatureExtractor,
    BatteryNormalizer,
    BatterySequenceProcessor
)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module metadata
__all__ = [
    # Model components
    "BatteryHealthTransformer",
    "BatteryHealthConfig",
    "MultiHeadBatteryAttention",
    "BatteryPositionalEncoding",
    "BatteryTransformerBlock",
    
    # Training components
    "BatteryHealthTrainer",
    "BatteryTrainingConfig",
    "BatteryTrainingMetrics",
    "BatteryLossFunction",
    "BatteryOptimizer",
    
    # Inference components
    "BatteryHealthPredictor",
    "BatteryPredictionResult",
    "BatteryHealthMetrics",
    "BatteryInferenceConfig",
    
    # Data handling components
    "BatteryDataLoader",
    "BatteryDataset",
    "BatterySequenceCollator",
    "BatteryDataConfig",
    
    # Preprocessing components
    "BatteryPreprocessor",
    "BatteryFeatureExtractor",
    "BatteryNormalizer",
    "BatterySequenceProcessor"
]

# Module configuration
DEFAULT_CONFIG = {
    "model": {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "d_ff": 2048,
        "dropout": 0.1,
        "max_sequence_length": 1024,
        "vocab_size": 10000
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "warmup_steps": 4000,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0
    },
    "data": {
        "sequence_length": 512,
        "prediction_horizon": 24,
        "feature_dim": 16,
        "target_dim": 4
    }
}

def get_default_config():
    """
    Get default configuration for battery health predictor.
    
    Returns:
        dict: Default configuration dictionary
    """
    return DEFAULT_CONFIG.copy()

def create_battery_health_model(config=None):
    """
    Factory function to create a battery health transformer model.
    
    Args:
        config (dict, optional): Model configuration. If None, uses default config.
        
    Returns:
        BatteryHealthTransformer: Configured model instance
    """
    if config is None:
        config = get_default_config()
    
    model_config = BatteryHealthConfig(**config["model"])
    return BatteryHealthTransformer(model_config)

def create_battery_trainer(model, config=None):
    """
    Factory function to create a battery health trainer.
    
    Args:
        model (BatteryHealthTransformer): Model to train
        config (dict, optional): Training configuration. If None, uses default config.
        
    Returns:
        BatteryHealthTrainer: Configured trainer instance
    """
    if config is None:
        config = get_default_config()
    
    training_config = BatteryTrainingConfig(**config["training"])
    return BatteryHealthTrainer(model, training_config)

def create_battery_predictor(model_path, config=None):
    """
    Factory function to create a battery health predictor.
    
    Args:
        model_path (str): Path to trained model
        config (dict, optional): Inference configuration. If None, uses default config.
        
    Returns:
        BatteryHealthPredictor: Configured predictor instance
    """
    if config is None:
        config = get_default_config()
    
    inference_config = BatteryInferenceConfig(**config.get("inference", {}))
    return BatteryHealthPredictor(model_path, inference_config)

# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BatteryMind Battery Health Predictor v{__version__} initialized")
