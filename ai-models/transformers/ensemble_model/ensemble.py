"""
BatteryMind - Ensemble Model Core

Advanced ensemble modeling framework that combines multiple AI models for
enhanced battery health prediction and degradation forecasting with
sophisticated fusion techniques and uncertainty quantification.

Features:
- Multi-model ensemble with weighted voting and stacking
- Dynamic weight adjustment based on model performance
- Uncertainty quantification through ensemble diversity
- Physics-informed ensemble constraints
- Real-time model selection and adaptation
- Integration with transformer, RL, and federated learning models

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import pickle
import json
from pathlib import Path
import time

# Scientific computing imports
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Local imports
from ..battery_health_predictor import BatteryHealthPredictor, BatteryInferenceConfig
from ..degradation_forecaster import BatteryDegradationForecaster, ForecastingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnsembleConfig:
    """
    Configuration for ensemble model operations.
    
    Attributes:
        # Ensemble composition
        model_types (List[str]): Types of models to include in ensemble
        model_weights (Dict[str, float]): Initial weights for each model type
        
        # Fusion methods
        fusion_method (str): Method for combining predictions
        meta_learner_type (str): Type of meta-learner for stacking
        
        # Dynamic adaptation
        enable_dynamic_weights (bool): Enable dynamic weight adjustment
        adaptation_window (int): Window size for performance tracking
        weight_update_frequency (int): Frequency of weight updates
        
        # Uncertainty quantification
        enable_uncertainty (bool): Enable ensemble uncertainty estimation
        uncertainty_method (str): Method for uncertainty calculation
        confidence_threshold (float): Minimum confidence for predictions
        
        # Performance optimization
        parallel_inference (bool): Enable parallel model inference
        cache_predictions (bool): Cache individual model predictions
        
        # Physics constraints
        enable_physics_validation (bool): Validate predictions against physics
        consistency_threshold (float): Threshold for prediction consistency
        
        # Model selection
        enable_model_selection (bool): Enable dynamic model selection
        selection_criteria (str): Criteria for model selection
        min_models (int): Minimum number of models to use
        max_models (int): Maximum number of models to use
    """
    # Ensemble composition
    model_types: List[str] = field(default_factory=lambda: [
        'battery_health_predictor', 'degradation_forecaster'
    ])
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        'battery_health_predictor': 0.6,
        'degradation_forecaster': 0.4
    })
    
    # Fusion methods
    fusion_method: str = "weighted_average"  # 'weighted_average', 'stacking', 'voting'
    meta_learner_type: str = "ridge"  # 'ridge', 'random_forest', 'neural_network'
    
    # Dynamic adaptation
    enable_dynamic_weights: bool = True
    adaptation_window: int = 100
    weight_update_frequency: int = 10
    
    # Uncertainty quantification
    enable_uncertainty: bool = True
    uncertainty_method: str = "ensemble_variance"  # 'ensemble_variance', 'prediction_intervals'
    confidence_threshold: float = 0.7
    
    # Performance optimization
    parallel_inference: bool = True
    cache_predictions: bool = True
    
    # Physics constraints
    enable_physics_validation: bool = True
    consistency_threshold: float = 0.1
    
    # Model selection
    enable_model_selection: bool = True
    selection_criteria: str = "performance_weighted"  # 'performance_weighted', 'diversity_based'
    min_models: int = 2
    max_models: int = 5

@dataclass
class EnsemblePredictionResult:
    """
    Comprehensive result structure for ensemble predictions.
    """
    # Core predictions
    ensemble_prediction: np.ndarray
    individual_predictions: Dict[str, np.ndarray]
    model_weights: Dict[str, float]
    
    # Uncertainty estimates
    prediction_variance: Optional[np.ndarray] = None
    confidence_intervals: Optional[Dict[str, np.ndarray]] = None
    ensemble_confidence: Optional[float] = None
    
    # Model performance
    model_performances: Optional[Dict[str, float]] = None
    prediction_consistency: Optional[float] = None
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    models_used: List[str] = field(default_factory=list)
    fusion_method: str = "weighted_average"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'ensemble_prediction': self.ensemble_prediction.tolist() if isinstance(self.ensemble_prediction, np.ndarray) else self.ensemble_prediction,
            'individual_predictions': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                     for k, v in self.individual_predictions.items()},
            'model_weights': self.model_weights,
            'prediction_variance': self.prediction_variance.tolist() if self.prediction_variance is not None else None,
            'confidence_intervals': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                   for k, v in self.confidence_intervals.items()} if self.confidence_intervals else None,
            'ensemble_confidence': self.ensemble_confidence,
            'model_performances': self.model_performances,
            'prediction_consistency': self.prediction_consistency,
            'timestamp': self.timestamp,
            'models_used': self.models_used,
            'fusion_method': self.fusion_method
        }

class BaseEnsembleModel(ABC):
    """
    Abstract base class for ensemble models.
    """
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models = {}
        self.model_weights = config.model_weights.copy()
        self.performance_history = {model_type: [] for model_type in config.model_types}
        
    @abstractmethod
    def add_model(self, model_type: str, model: Any, weight: Optional[float] = None) -> None:
        """Add a model to the ensemble."""
        pass
    
    @abstractmethod
    def predict(self, inputs: Any, **kwargs) -> EnsemblePredictionResult:
        """Make ensemble prediction."""
        pass
    
    @abstractmethod
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """Update model weights based on performance."""
        pass

class WeightedEnsemble(BaseEnsembleModel):
    """
    Weighted ensemble that combines predictions using learned or fixed weights.
    """
    
    def __init__(self, config: EnsembleConfig):
        super().__init__(config)
        self.prediction_cache = {} if config.cache_predictions else None
        
    def add_model(self, model_type: str, model: Any, weight: Optional[float] = None) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model_type (str): Type identifier for the model
            model (Any): The model instance
            weight (float, optional): Weight for this model
        """
        self.models[model_type] = model
        if weight is not None:
            self.model_weights[model_type] = weight
        elif model_type not in self.model_weights:
            self.model_weights[model_type] = 1.0 / len(self.config.model_types)
        
        logger.info(f"Added {model_type} to ensemble with weight {self.model_weights[model_type]:.3f}")
    
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for model_type in self.model_weights:
                self.model_weights[model_type] /= total_weight
    
    def _get_individual_predictions(self, inputs: Any, **kwargs) -> Dict[str, np.ndarray]:
        """Get predictions from individual models."""
        predictions = {}
        
        for model_type, model in self.models.items():
            try:
                if model_type == 'battery_health_predictor':
                    result = model.predict(inputs, **kwargs)
                    if hasattr(result, 'state_of_health'):
                        # Extract relevant metrics for ensemble
                        pred_array = np.array([
                            result.state_of_health,
                            result.degradation_patterns.get('capacity_fade_rate', 0.0),
                            result.degradation_patterns.get('resistance_increase_rate', 0.0),
                            result.degradation_patterns.get('thermal_degradation', 0.0)
                        ])
                    else:
                        pred_array = np.array(result)
                    
                elif model_type == 'degradation_forecaster':
                    result = model.predict_degradation(inputs, **kwargs)
                    if 'forecasts' in result:
                        # Use mean forecast values
                        forecasts = result['forecasts']
                        if isinstance(forecasts, torch.Tensor):
                            forecasts = forecasts.cpu().numpy()
                        pred_array = np.mean(forecasts, axis=1) if len(forecasts.shape) > 1 else forecasts
                    else:
                        pred_array = np.array(result)
                
                else:
                    # Generic prediction interface
                    result = model.predict(inputs, **kwargs)
                    pred_array = np.array(result)
                
                predictions[model_type] = pred_array
                
            except Exception as e:
                logger.warning(f"Failed to get prediction from {model_type}: {e}")
                # Use zero prediction as fallback
                predictions[model_type] = np.zeros(4)  # Assuming 4-dimensional output
        
        return predictions
    
    def _calculate_ensemble_prediction(self, individual_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate weighted ensemble prediction."""
        if not individual_predictions:
            raise ValueError("No individual predictions available")
        
        # Ensure all predictions have the same shape
        pred_shapes = [pred.shape for pred in individual_predictions.values()]
        if len(set(pred_shapes)) > 1:
            # Pad or truncate to match the first prediction shape
            target_shape = pred_shapes[0]
            for model_type, pred in individual_predictions.items():
                if pred.shape != target_shape:
                    if len(pred) < len(target_shape):
                        # Pad with zeros
                        padded_pred = np.zeros(target_shape)
                        padded_pred[:len(pred)] = pred
                        individual_predictions[model_type] = padded_pred
                    else:
                        # Truncate
                        individual_predictions[model_type] = pred[:len(target_shape)]
        
        # Calculate weighted average
        ensemble_pred = np.zeros_like(list(individual_predictions.values())[0])
        total_weight = 0.0
        
        for model_type, prediction in individual_predictions.items():
            weight = self.model_weights.get(model_type, 0.0)
            ensemble_pred += weight * prediction
            total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred
    
    def _calculate_uncertainty(self, individual_predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate ensemble uncertainty metrics."""
        if len(individual_predictions) < 2:
            return {'variance': None, 'confidence': 1.0}
        
        predictions_array = np.array(list(individual_predictions.values()))
        
        # Calculate variance across models
        prediction_variance = np.var(predictions_array, axis=0)
        
        # Calculate ensemble confidence (inverse of normalized variance)
        mean_variance = np.mean(prediction_variance)
        confidence = 1.0 / (1.0 + mean_variance)
        
        # Calculate prediction intervals (assuming normal distribution)
        mean_pred = np.mean(predictions_array, axis=0)
        std_pred = np.std(predictions_array, axis=0)
        
        confidence_intervals = {
            '95%': {
                'lower': mean_pred - 1.96 * std_pred,
                'upper': mean_pred + 1.96 * std_pred
            },
            '90%': {
                'lower': mean_pred - 1.645 * std_pred,
                'upper': mean_pred + 1.645 * std_pred
            }
        }
        
        return {
            'variance': prediction_variance,
            'confidence': confidence,
            'confidence_intervals': confidence_intervals
        }
    
    def _validate_physics_constraints(self, prediction: np.ndarray) -> Tuple[bool, float]:
        """Validate predictions against physics constraints."""
        if not self.config.enable_physics_validation:
            return True, 1.0
        
        violations = 0
        total_checks = 0
        
        # Check State of Health bounds (0-1)
        if len(prediction) > 0:
            soh = prediction[0]
            if not (0.0 <= soh <= 1.0):
                violations += 1
            total_checks += 1
        
        # Check degradation rates (should be non-negative and reasonable)
        if len(prediction) > 1:
            for i in range(1, len(prediction)):
                if prediction[i] < 0 or prediction[i] > 0.01:  # Max 1% degradation per time unit
                    violations += 1
                total_checks += 1
        
        consistency_score = 1.0 - (violations / total_checks) if total_checks > 0 else 1.0
        is_valid = consistency_score >= self.config.consistency_threshold
        
        return is_valid, consistency_score
    
    def predict(self, inputs: Any, **kwargs) -> EnsemblePredictionResult:
        """
        Make ensemble prediction combining multiple models.
        
        Args:
            inputs: Input data for prediction
            **kwargs: Additional arguments for individual models
            
        Returns:
            EnsemblePredictionResult: Comprehensive ensemble prediction result
        """
        if not self.models:
            raise ValueError("No models available in ensemble")
        
        # Get individual predictions
        individual_predictions = self._get_individual_predictions(inputs, **kwargs)
        
        if not individual_predictions:
            raise ValueError("No valid predictions obtained from ensemble models")
        
        # Calculate ensemble prediction
        ensemble_prediction = self._calculate_ensemble_prediction(individual_predictions)
        
        # Calculate uncertainty metrics
        uncertainty_metrics = self._calculate_uncertainty(individual_predictions)
        
        # Validate physics constraints
        is_valid, consistency_score = self._validate_physics_constraints(ensemble_prediction)
        
        if not is_valid:
            logger.warning(f"Ensemble prediction violates physics constraints (consistency: {consistency_score:.3f})")
        
        # Create result
        result = EnsemblePredictionResult(
            ensemble_prediction=ensemble_prediction,
            individual_predictions=individual_predictions,
            model_weights=self.model_weights.copy(),
            prediction_variance=uncertainty_metrics['variance'],
            confidence_intervals=uncertainty_metrics.get('confidence_intervals'),
            ensemble_confidence=uncertainty_metrics['confidence'],
            prediction_consistency=consistency_score,
            models_used=list(individual_predictions.keys()),
            fusion_method=self.config.fusion_method
        )
        
        return result
    
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """
        Update model weights based on recent performance.
        
        Args:
            performance_metrics (Dict[str, float]): Performance metrics for each model
        """
        if not self.config.enable_dynamic_weights:
            return
        
        # Update performance history
        for model_type, performance in performance_metrics.items():
            if model_type in self.performance_history:
                self.performance_history[model_type].append(performance)
                # Keep only recent history
                if len(self.performance_history[model_type]) > self.config.adaptation_window:
                    self.performance_history[model_type].pop(0)
        
        # Calculate new weights based on recent performance
        new_weights = {}
        for model_type in self.model_weights:
            if model_type in self.performance_history and self.performance_history[model_type]:
                # Use exponentially weighted moving average of recent performance
                recent_performance = self.performance_history[model_type][-10:]  # Last 10 evaluations
                weights = np.exp(np.arange(len(recent_performance)))  # Exponential weights
                avg_performance = np.average(recent_performance, weights=weights)
                new_weights[model_type] = max(0.01, avg_performance)  # Minimum weight of 0.01
            else:
                new_weights[model_type] = self.model_weights[model_type]
        
        # Normalize weights
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for model_type in new_weights:
                new_weights[model_type] /= total_weight
        
        # Update weights with smoothing
        alpha = 0.1  # Smoothing factor
        for model_type in self.model_weights:
            if model_type in new_weights:
                self.model_weights[model_type] = (
                    alpha * new_weights[model_type] + 
                    (1 - alpha) * self.model_weights[model_type]
                )
        
        logger.info(f"Updated ensemble weights: {self.model_weights}")
    
    def get_model_importance(self) -> Dict[str, float]:
        """Get relative importance of each model in the ensemble."""
        return self.model_weights.copy()
    
    def save_ensemble(self, file_path: str) -> None:
        """Save ensemble configuration and weights."""
        ensemble_data = {
            'config': self.config.__dict__,
            'model_weights': self.model_weights,
            'performance_history': self.performance_history,
            'model_types': list(self.models.keys())
        }
        
        with open(file_path, 'w') as f:
            json.dump(ensemble_data, f, indent=2)
        
        logger.info(f"Ensemble saved to {file_path}")
    
    @classmethod
    def load_ensemble(cls, file_path: str) -> 'WeightedEnsemble':
        """Load ensemble configuration and weights."""
        with open(file_path, 'r') as f:
            ensemble_data = json.load(f)
        
        config = EnsembleConfig(**ensemble_data['config'])
        ensemble = cls(config)
        ensemble.model_weights = ensemble_data['model_weights']
        ensemble.performance_history = ensemble_data['performance_history']
        
        logger.info(f"Ensemble loaded from {file_path}")
        return ensemble

class StackingEnsemble(BaseEnsembleModel):
    """
    Stacking ensemble that uses a meta-learner to combine predictions.
    """
    
    def __init__(self, config: EnsembleConfig):
        super().__init__(config)
        self.meta_learner = self._create_meta_learner()
        self.is_fitted = False
        
    def _create_meta_learner(self):
        """Create meta-learner based on configuration."""
        if self.config.meta_learner_type == "ridge":
            return Ridge(alpha=1.0)
        elif self.config.meta_learner_type == "random_forest":
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.config.meta_learner_type == "neural_network":
            return self._create_neural_meta_learner()
        else:
            raise ValueError(f"Unknown meta-learner type: {self.config.meta_learner_type}")
    
    def _create_neural_meta_learner(self):
        """Create neural network meta-learner."""
        class NeuralMetaLearner(nn.Module):
            def __init__(self, input_dim: int, output_dim: int):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, output_dim)
                )
            
            def forward(self, x):
                return self.network(x)
        
        # Will be initialized when we know the dimensions
        return None
    
    def add_model(self, model_type: str, model: Any, weight: Optional[float] = None) -> None:
        """Add a model to the stacking ensemble."""
        self.models[model_type] = model
        logger.info(f"Added {model_type} to stacking ensemble")
    
    def fit_meta_learner(self, training_data: List[Tuple[Any, np.ndarray]]) -> None:
        """
        Fit the meta-learner using training data.
        
        Args:
            training_data: List of (input, target) pairs for training
        """
        if not self.models:
            raise ValueError("No base models available")
        
        # Collect base model predictions and targets
        base_predictions = []
        targets = []
        
        for inputs, target in training_data:
            # Get predictions from all base models
            individual_preds = self._get_individual_predictions(inputs)
            
            # Flatten and concatenate predictions
            pred_vector = np.concatenate([pred.flatten() for pred in individual_preds.values()])
            base_predictions.append(pred_vector)
            targets.append(target.flatten() if hasattr(target, 'flatten') else target)
        
        X = np.array(base_predictions)
        y = np.array(targets)
        
        # Fit meta-learner
        if self.config.meta_learner_type == "neural_network":
            # Initialize neural network with correct dimensions
            input_dim = X.shape[1]
            output_dim = y.shape[1] if len(y.shape) > 1 else 1
            self.meta_learner = self._create_neural_meta_learner()(input_dim, output_dim)
            
            # Train neural network (simplified training)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=0.001)
            
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = self.meta_learner(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
        else:
            # Fit sklearn-based meta-learner
            self.meta_learner.fit(X, y)
        
        self.is_fitted = True
        logger.info("Meta-learner fitted successfully")
    
    def _get_individual_predictions(self, inputs: Any) -> Dict[str, np.ndarray]:
        """Get predictions from base models (same as WeightedEnsemble)."""
        predictions = {}
        
        for model_type, model in self.models.items():
            try:
                if model_type == 'battery_health_predictor':
                    result = model.predict(inputs)
                    pred_array = np.array([
                        result.state_of_health,
                        result.degradation_patterns.get('capacity_fade_rate', 0.0),
                        result.degradation_patterns.get('resistance_increase_rate', 0.0),
                        result.degradation_patterns.get('thermal_degradation', 0.0)
                    ])
                elif model_type == 'degradation_forecaster':
                    result = model.predict_degradation(inputs)
                    forecasts = result['forecasts']
                    if isinstance(forecasts, torch.Tensor):
                        forecasts = forecasts.cpu().numpy()
                    pred_array = np.mean(forecasts, axis=1) if len(forecasts.shape) > 1 else forecasts
                else:
                    result = model.predict(inputs)
                    pred_array = np.array(result)
                
                predictions[model_type] = pred_array
                
            except Exception as e:
                logger.warning(f"Failed to get prediction from {model_type}: {e}")
                predictions[model_type] = np.zeros(4)
        
        return predictions
    
    def predict(self, inputs: Any, **kwargs) -> EnsemblePredictionResult:
        """Make stacking ensemble prediction."""
        if not self.is_fitted:
            raise ValueError("Meta-learner not fitted. Call fit_meta_learner first.")
        
        # Get base model predictions
        individual_predictions = self._get_individual_predictions(inputs)
        
        # Prepare input for meta-learner
        pred_vector = np.concatenate([pred.flatten() for pred in individual_predictions.values()])
        
        # Get meta-learner prediction
        if self.config.meta_learner_type == "neural_network":
            with torch.no_grad():
                pred_tensor = torch.FloatTensor(pred_vector.reshape(1, -1))
                ensemble_prediction = self.meta_learner(pred_tensor).numpy().flatten()
        else:
            ensemble_prediction = self.meta_learner.predict(pred_vector.reshape(1, -1)).flatten()
        
        # Calculate uncertainty (simplified for stacking)
        prediction_variance = np.var(list(individual_predictions.values()), axis=0)
        ensemble_confidence = 1.0 / (1.0 + np.mean(prediction_variance))
        
        result = EnsemblePredictionResult(
            ensemble_prediction=ensemble_prediction,
            individual_predictions=individual_predictions,
            model_weights={'meta_learner': 1.0},  # Meta-learner determines weights
            prediction_variance=prediction_variance,
            ensemble_confidence=ensemble_confidence,
            models_used=list(individual_predictions.keys()),
            fusion_method="stacking"
        )
        
        return result
    
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """Update not applicable for stacking - meta-learner handles weighting."""
        pass

# Factory functions
def create_ensemble_model(config: EnsembleConfig) -> BaseEnsembleModel:
    """
    Factory function to create an ensemble model.
    
    Args:
        config (EnsembleConfig): Ensemble configuration
        
    Returns:
        BaseEnsembleModel: Configured ensemble model
    """
    if config.fusion_method in ["weighted_average", "voting"]:
        return WeightedEnsemble(config)
    elif config.fusion_method == "stacking":
        return StackingEnsemble(config)
    else:
        raise ValueError(f"Unknown fusion method: {config.fusion_method}")

def create_battery_ensemble(health_predictor_path: str, 
                          forecaster_path: str,
                          config: Optional[EnsembleConfig] = None) -> BaseEnsembleModel:
    """
    Create a complete battery ensemble with health predictor and forecaster.
    
    Args:
        health_predictor_path (str): Path to health predictor model
        forecaster_path (str): Path to forecaster model
        config (EnsembleConfig, optional): Ensemble configuration
        
    Returns:
        BaseEnsembleModel: Configured ensemble with loaded models
    """
    if config is None:
        config = EnsembleConfig()
    
    ensemble = create_ensemble_model(config)
    
    # Load and add health predictor
    health_config = BatteryInferenceConfig(model_path=health_predictor_path)
    health_predictor = BatteryHealthPredictor(health_config)
    ensemble.add_model('battery_health_predictor', health_predictor)
    
    # Load and add degradation forecaster
    forecast_config = ForecastingConfig(model_path=forecaster_path)
    forecaster = BatteryDegradationForecaster(forecast_config)
    ensemble.add_model('degradation_forecaster', forecaster)
    
    logger.info("Battery ensemble created with health predictor and forecaster")
    return ensemble
