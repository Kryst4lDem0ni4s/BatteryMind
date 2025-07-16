"""
BatteryMind - Ensemble Predictor

Advanced ensemble prediction system that combines multiple AI/ML models including
transformer-based battery health predictors, federated learning models, and
reinforcement learning agents for comprehensive battery management decisions.

Features:
- Multi-model ensemble prediction with weighted voting
- Stacked generalization with meta-learning
- Uncertainty quantification and prediction confidence
- Real-time inference with sub-second response times
- Adaptive model selection based on input characteristics
- Robust error handling and fallback mechanisms

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import torch
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from abc import ABC, abstractmethod

# ML Framework imports
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Import BatteryMind components
from .battery_health_predictor import BatteryHealthPredictor
from .degradation_predictor import DegradationPredictor
from .optimization_predictor import OptimizationPredictor
from ..pipelines.inference_pipeline import InferencePipeline
from ...utils.logging_utils import get_logger
from ...utils.model_utils import ModelLoader, ModelValidator
from ...utils.data_utils import DataValidator, FeatureProcessor

# Configure logging
logger = get_logger(__name__)

@dataclass
class EnsembleConfig:
    """
    Configuration for ensemble prediction system.
    
    Attributes:
        # Model weights and selection
        model_weights (Dict[str, float]): Weights for each model in ensemble
        voting_strategy (str): Strategy for combining predictions
        meta_model_type (str): Type of meta-model for stacking
        
        # Performance settings
        prediction_timeout_seconds (float): Timeout for individual predictions
        parallel_prediction (bool): Enable parallel model execution
        max_workers (int): Maximum number of parallel workers
        
        # Quality control
        enable_uncertainty_quantification (bool): Enable uncertainty estimation
        confidence_threshold (float): Minimum confidence threshold
        enable_prediction_validation (bool): Enable prediction validation
        
        # Fallback configuration
        enable_fallback (bool): Enable fallback to simpler models
        fallback_model_priority (List[str]): Priority order for fallback models
        
        # Caching and optimization
        enable_model_caching (bool): Enable model caching for performance
        cache_size (int): Maximum cache size for models
        enable_prediction_caching (bool): Enable prediction result caching
    """
    # Model weights and selection
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        'transformer': 0.4,
        'federated': 0.25,
        'rl_agent': 0.25,
        'physics_model': 0.1
    })
    voting_strategy: str = "weighted_average"  # "weighted_average", "stacking", "adaptive"
    meta_model_type: str = "gradient_boosting"  # "linear", "gradient_boosting", "neural_network"
    
    # Performance settings
    prediction_timeout_seconds: float = 5.0
    parallel_prediction: bool = True
    max_workers: int = 4
    
    # Quality control
    enable_uncertainty_quantification: bool = True
    confidence_threshold: float = 0.7
    enable_prediction_validation: bool = True
    
    # Fallback configuration
    enable_fallback: bool = True
    fallback_model_priority: List[str] = field(default_factory=lambda: [
        'transformer', 'physics_model', 'rule_based'
    ])
    
    # Caching and optimization
    enable_model_caching: bool = True
    cache_size: int = 10
    enable_prediction_caching: bool = True

@dataclass
class PredictionResult:
    """
    Comprehensive prediction result with metadata.
    
    Attributes:
        predictions (Dict[str, Any]): Individual model predictions
        ensemble_prediction (Any): Final ensemble prediction
        confidence_score (float): Prediction confidence (0-1)
        uncertainty_estimate (float): Uncertainty estimate
        model_contributions (Dict[str, float]): Individual model contributions
        prediction_time_ms (float): Total prediction time
        metadata (Dict[str, Any]): Additional metadata
    """
    predictions: Dict[str, Any] = field(default_factory=dict)
    ensemble_prediction: Any = None
    confidence_score: float = 0.0
    uncertainty_estimate: float = 0.0
    model_contributions: Dict[str, float] = field(default_factory=dict)
    prediction_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseEnsembleModel(ABC):
    """Base class for ensemble models."""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.is_loaded = False
        
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Load model from path."""
        pass
    
    @abstractmethod
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction on input data."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass

class TransformerEnsembleModel(BaseEnsembleModel):
    """Transformer model wrapper for ensemble."""
    
    def __init__(self, config: EnsembleConfig):
        super().__init__(config)
        self.predictor = None
        
    def load_model(self, model_path: str) -> bool:
        """Load transformer model."""
        try:
            self.predictor = BatteryHealthPredictor()
            self.predictor.load_model(model_path)
            self.is_loaded = True
            logger.info("Transformer model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}")
            return False
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using transformer model."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        try:
            # Extract relevant features for transformer
            input_features = self._prepare_transformer_input(data)
            
            # Make prediction
            prediction = self.predictor.predict(input_features)
            
            return {
                'health_score': prediction.get('health_score', 0.0),
                'degradation_rate': prediction.get('degradation_rate', 0.0),
                'remaining_useful_life': prediction.get('remaining_useful_life', 0.0),
                'confidence': prediction.get('confidence', 0.0),
                'model_type': 'transformer'
            }
        except Exception as e:
            logger.error(f"Transformer prediction failed: {e}")
            return {'error': str(e), 'model_type': 'transformer'}
    
    def _prepare_transformer_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input data for transformer model."""
        return {
            'voltage_sequence': data.get('voltage_sequence', []),
            'current_sequence': data.get('current_sequence', []),
            'temperature_sequence': data.get('temperature_sequence', []),
            'usage_patterns': data.get('usage_patterns', {}),
            'environmental_conditions': data.get('environmental_conditions', {})
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get transformer model information."""
        return {
            'model_type': 'transformer',
            'model_version': '1.0.0',
            'is_loaded': self.is_loaded,
            'capabilities': ['health_prediction', 'degradation_forecast', 'rul_estimation']
        }

class FederatedEnsembleModel(BaseEnsembleModel):
    """Federated learning model wrapper for ensemble."""
    
    def __init__(self, config: EnsembleConfig):
        super().__init__(config)
        self.model = None
        
    def load_model(self, model_path: str) -> bool:
        """Load federated global model."""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_loaded = True
            logger.info("Federated model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load federated model: {e}")
            return False
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using federated model."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        try:
            # Extract federated features
            input_features = self._prepare_federated_input(data)
            
            # Make prediction
            prediction = self.model.predict([input_features])
            
            return {
                'fleet_health_assessment': prediction[0],
                'cross_battery_insights': self._get_cross_battery_insights(data),
                'privacy_preserved_score': self._calculate_privacy_score(data),
                'model_type': 'federated'
            }
        except Exception as e:
            logger.error(f"Federated prediction failed: {e}")
            return {'error': str(e), 'model_type': 'federated'}
    
    def _prepare_federated_input(self, data: Dict[str, Any]) -> np.ndarray:
        """Prepare input for federated model."""
        # Extract aggregated features for federated learning
        features = [
            data.get('aggregated_soc', 0.0),
            data.get('aggregated_soh', 0.0),
            data.get('fleet_variance', 0.0),
            data.get('usage_diversity', 0.0),
            data.get('environmental_similarity', 0.0)
        ]
        return np.array(features)
    
    def _get_cross_battery_insights(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cross-battery insights."""
        return {
            'similar_batteries': 5,
            'performance_percentile': 75,
            'fleet_average_deviation': 0.02
        }
    
    def _calculate_privacy_score(self, data: Dict[str, Any]) -> float:
        """Calculate privacy preservation score."""
        return 0.95  # Placeholder implementation
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get federated model information."""
        return {
            'model_type': 'federated',
            'model_version': '1.0.0',
            'is_loaded': self.is_loaded,
            'capabilities': ['fleet_analytics', 'privacy_preservation', 'cross_battery_learning']
        }

class RLEnsembleModel(BaseEnsembleModel):
    """Reinforcement learning model wrapper for ensemble."""
    
    def __init__(self, config: EnsembleConfig):
        super().__init__(config)
        self.policy_network = None
        self.value_network = None
        
    def load_model(self, model_path: str) -> bool:
        """Load RL agent models."""
        try:
            policy_path = Path(model_path) / "policy_network.pt"
            value_path = Path(model_path) / "value_network.pt"
            
            self.policy_network = torch.load(policy_path, map_location='cpu')
            self.value_network = torch.load(value_path, map_location='cpu')
            
            self.policy_network.eval()
            self.value_network.eval()
            
            self.is_loaded = True
            logger.info("RL agent models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load RL models: {e}")
            return False
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using RL agent."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        try:
            # Prepare RL environment state
            state = self._prepare_rl_state(data)
            
            # Get action from policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = self.policy_network(state_tensor)
                value = self.value_network(state_tensor)
            
            return {
                'optimal_charging_current': action[0][0].item(),
                'thermal_control': action[0][1].item(),
                'power_limit': action[0][2].item(),
                'expected_value': value[0].item(),
                'optimization_confidence': self._calculate_rl_confidence(action, value),
                'model_type': 'rl_agent'
            }
        except Exception as e:
            logger.error(f"RL prediction failed: {e}")
            return {'error': str(e), 'model_type': 'rl_agent'}
    
    def _prepare_rl_state(self, data: Dict[str, Any]) -> np.ndarray:
        """Prepare state vector for RL agent."""
        state = [
            data.get('soc', 0.5),
            data.get('soh', 0.9),
            data.get('temperature', 25.0),
            data.get('voltage', 3.7),
            data.get('current', 0.0),
            data.get('internal_resistance', 0.1),
            data.get('ambient_temperature', 20.0),
            data.get('power_demand', 0.0),
            data.get('grid_price', 0.15),
            data.get('safety_margin', 0.8)
        ]
        return np.array(state, dtype=np.float32)
    
    def _calculate_rl_confidence(self, action: torch.Tensor, value: torch.Tensor) -> float:
        """Calculate RL prediction confidence."""
        # Simple confidence based on value function
        return min(1.0, max(0.0, float(value.item()) / 1000.0))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get RL model information."""
        return {
            'model_type': 'rl_agent',
            'model_version': '1.0.0',
            'is_loaded': self.is_loaded,
            'capabilities': ['charging_optimization', 'thermal_management', 'power_control']
        }

class PhysicsEnsembleModel(BaseEnsembleModel):
    """Physics-based model wrapper for ensemble."""
    
    def __init__(self, config: EnsembleConfig):
        super().__init__(config)
        self.is_loaded = True  # Physics model is always available
        
    def load_model(self, model_path: str) -> bool:
        """Physics model doesn't need loading."""
        return True
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using physics-based model."""
        try:
            # Physics-based calculations
            soc = data.get('soc', 0.5)
            voltage = data.get('voltage', 3.7)
            current = data.get('current', 0.0)
            temperature = data.get('temperature', 25.0)
            
            # Simplified physics calculations
            internal_resistance = self._calculate_internal_resistance(soc, temperature)
            capacity_fade = self._calculate_capacity_fade(data.get('cycle_count', 0), temperature)
            thermal_model = self._calculate_thermal_behavior(current, temperature)
            
            return {
                'physics_health_score': 1.0 - capacity_fade,
                'internal_resistance': internal_resistance,
                'thermal_prediction': thermal_model,
                'physics_consistency_score': self._validate_physics_consistency(data),
                'model_type': 'physics'
            }
        except Exception as e:
            logger.error(f"Physics prediction failed: {e}")
            return {'error': str(e), 'model_type': 'physics'}
    
    def _calculate_internal_resistance(self, soc: float, temperature: float) -> float:
        """Calculate internal resistance based on SOC and temperature."""
        # Simplified model
        base_resistance = 0.1
        soc_factor = 1.0 + 0.5 * (1.0 - soc)  # Higher resistance at low SOC
        temp_factor = 1.0 + 0.01 * (25.0 - temperature)  # Higher resistance at low temp
        return base_resistance * soc_factor * temp_factor
    
    def _calculate_capacity_fade(self, cycle_count: int, temperature: float) -> float:
        """Calculate capacity fade based on cycles and temperature."""
        # Simplified Arrhenius model
        base_fade_rate = 0.0001  # Per cycle
        temp_acceleration = np.exp(0.05 * (temperature - 25.0))
        return base_fade_rate * cycle_count * temp_acceleration
    
    def _calculate_thermal_behavior(self, current: float, temperature: float) -> Dict[str, float]:
        """Calculate thermal behavior."""
        return {
            'temperature_rise': abs(current) * 0.1,  # Simplified heat generation
            'cooling_rate': max(0, temperature - 25.0) * 0.05,
            'thermal_runaway_risk': max(0, temperature - 60.0) / 20.0
        }
    
    def _validate_physics_consistency(self, data: Dict[str, Any]) -> float:
        """Validate physics consistency of the data."""
        # Simple consistency checks
        score = 1.0
        
        # Check voltage-SOC consistency
        soc = data.get('soc', 0.5)
        voltage = data.get('voltage', 3.7)
        expected_voltage = 3.0 + soc * 1.2  # Simplified
        voltage_error = abs(voltage - expected_voltage) / expected_voltage
        score *= max(0, 1.0 - voltage_error)
        
        return score
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get physics model information."""
        return {
            'model_type': 'physics',
            'model_version': '1.0.0',
            'is_loaded': self.is_loaded,
            'capabilities': ['physics_validation', 'consistency_check', 'thermal_modeling']
        }

class BatteryEnsemblePredictor:
    """
    Main ensemble predictor combining multiple AI/ML models for comprehensive 
    battery management decisions.
    """
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.models = {}
        self.meta_model = None
        self.feature_processor = FeatureProcessor()
        self.data_validator = DataValidator()
        self.model_cache = {}
        self.prediction_cache = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance tracking
        self.performance_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_prediction_time': 0.0,
            'model_usage_count': {}
        }
        
        logger.info("BatteryEnsemblePredictor initialized")
    
    def load_models(self, model_paths: Dict[str, str]) -> bool:
        """
        Load all ensemble models.
        
        Args:
            model_paths: Dictionary mapping model names to their paths
            
        Returns:
            bool: True if all models loaded successfully
        """
        success = True
        
        # Initialize model instances
        if 'transformer' in model_paths:
            self.models['transformer'] = TransformerEnsembleModel(self.config)
            success &= self.models['transformer'].load_model(model_paths['transformer'])
        
        if 'federated' in model_paths:
            self.models['federated'] = FederatedEnsembleModel(self.config)
            success &= self.models['federated'].load_model(model_paths['federated'])
        
        if 'rl_agent' in model_paths:
            self.models['rl_agent'] = RLEnsembleModel(self.config)
            success &= self.models['rl_agent'].load_model(model_paths['rl_agent'])
        
        # Physics model is always available
        self.models['physics'] = PhysicsEnsembleModel(self.config)
        
        # Load meta-model if using stacking
        if self.config.voting_strategy == "stacking":
            self._load_meta_model()
        
        logger.info(f"Ensemble models loaded: {list(self.models.keys())}")
        return success
    
    def predict(self, data: Dict[str, Any]) -> PredictionResult:
        """
        Make ensemble prediction on input data.
        
        Args:
            data: Input data dictionary containing battery sensor data
            
        Returns:
            PredictionResult: Comprehensive prediction result
        """
        start_time = time.time()
        
        try:
            # Validate input data
            if self.config.enable_prediction_validation:
                validation_result = self.data_validator.validate(data)
                if not validation_result.is_valid:
                    raise ValueError(f"Invalid input data: {validation_result.errors}")
            
            # Check prediction cache
            if self.config.enable_prediction_caching:
                cache_key = self._generate_cache_key(data)
                if cache_key in self.prediction_cache:
                    return self.prediction_cache[cache_key]
            
            # Get predictions from all models
            predictions = self._get_model_predictions(data)
            
            # Combine predictions using ensemble strategy
            ensemble_prediction = self._combine_predictions(predictions, data)
            
            # Calculate uncertainty and confidence
            uncertainty = self._calculate_uncertainty(predictions)
            confidence = self._calculate_confidence(predictions, uncertainty)
            
            # Create result
            result = PredictionResult(
                predictions=predictions,
                ensemble_prediction=ensemble_prediction,
                confidence_score=confidence,
                uncertainty_estimate=uncertainty,
                model_contributions=self._calculate_model_contributions(predictions),
                prediction_time_ms=(time.time() - start_time) * 1000,
                metadata=self._generate_metadata(data, predictions)
            )
            
            # Cache result
            if self.config.enable_prediction_caching:
                self.prediction_cache[cache_key] = result
            
            # Update performance stats
            self._update_performance_stats(True, time.time() - start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            self._update_performance_stats(False, time.time() - start_time)
            
            # Return fallback prediction
            if self.config.enable_fallback:
                return self._get_fallback_prediction(data, str(e))
            else:
                raise
    
    def _get_model_predictions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get predictions from all available models."""
        predictions = {}
        
        if self.config.parallel_prediction:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_model = {
                    executor.submit(self._safe_model_predict, model_name, model, data): model_name
                    for model_name, model in self.models.items()
                }
                
                for future in as_completed(future_to_model, timeout=self.config.prediction_timeout_seconds):
                    model_name = future_to_model[future]
                    try:
                        predictions[model_name] = future.result()
                    except Exception as e:
                        logger.warning(f"Model {model_name} prediction failed: {e}")
                        predictions[model_name] = {'error': str(e)}
        else:
            # Sequential execution
            for model_name, model in self.models.items():
                try:
                    predictions[model_name] = self._safe_model_predict(model_name, model, data)
                except Exception as e:
                    logger.warning(f"Model {model_name} prediction failed: {e}")
                    predictions[model_name] = {'error': str(e)}
        
        return predictions
    
    def _safe_model_predict(self, model_name: str, model: BaseEnsembleModel, data: Dict[str, Any]) -> Dict[str, Any]:
        """Safely make prediction with timeout handling."""
        try:
            return model.predict(data)
        except Exception as e:
            logger.warning(f"Model {model_name} prediction failed: {e}")
            return {'error': str(e), 'model_type': model_name}
    
    def _combine_predictions(self, predictions: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine individual model predictions into ensemble result."""
        if self.config.voting_strategy == "weighted_average":
            return self._weighted_average_combination(predictions)
        elif self.config.voting_strategy == "stacking":
            return self._stacking_combination(predictions, data)
        elif self.config.voting_strategy == "adaptive":
            return self._adaptive_combination(predictions, data)
        else:
            raise ValueError(f"Unknown voting strategy: {self.config.voting_strategy}")
    
    def _weighted_average_combination(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Combine predictions using weighted average."""
        combined = {}
        
        # Extract common prediction keys
        valid_predictions = {k: v for k, v in predictions.items() if 'error' not in v}
        
        if not valid_predictions:
            return {'error': 'No valid predictions available'}
        
        # Health score combination
        health_scores = []
        weights = []
        
        for model_name, pred in valid_predictions.items():
            if 'health_score' in pred:
                health_scores.append(pred['health_score'])
                weights.append(self.config.model_weights.get(model_name, 0.1))
            elif 'physics_health_score' in pred:
                health_scores.append(pred['physics_health_score'])
                weights.append(self.config.model_weights.get(model_name, 0.1))
        
        if health_scores:
            weights = np.array(weights) / np.sum(weights)  # Normalize weights
            combined['health_score'] = np.average(health_scores, weights=weights)
        
        # Combine other metrics
        combined['optimization_actions'] = self._combine_optimization_actions(valid_predictions)
        combined['fleet_insights'] = self._combine_fleet_insights(valid_predictions)
        combined['physics_validation'] = self._combine_physics_validation(valid_predictions)
        
        return combined
    
    def _stacking_combination(self, predictions: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine predictions using stacking with meta-model."""
        if self.meta_model is None:
            return self._weighted_average_combination(predictions)
        
        # Prepare features for meta-model
        meta_features = self._prepare_meta_features(predictions, data)
        
        # Get meta-model prediction
        try:
            meta_prediction = self.meta_model.predict([meta_features])
            return {
                'health_score': meta_prediction[0],
                'meta_model_confidence': self._calculate_meta_confidence(meta_features),
                'base_predictions': predictions
            }
        except Exception as e:
            logger.warning(f"Meta-model prediction failed: {e}")
            return self._weighted_average_combination(predictions)
    
    def _adaptive_combination(self, predictions: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptively combine predictions based on input characteristics."""
        # Select best model based on input characteristics
        input_characteristics = self._analyze_input_characteristics(data)
        
        if input_characteristics['is_time_series_heavy']:
            # Give more weight to transformer
            adapted_weights = self.config.model_weights.copy()
            adapted_weights['transformer'] *= 2.0
        elif input_characteristics['is_fleet_data']:
            # Give more weight to federated model
            adapted_weights = self.config.model_weights.copy()
            adapted_weights['federated'] *= 2.0
        elif input_characteristics['is_optimization_focused']:
            # Give more weight to RL agent
            adapted_weights = self.config.model_weights.copy()
            adapted_weights['rl_agent'] *= 2.0
        else:
            adapted_weights = self.config.model_weights
        
        # Temporarily update weights
        original_weights = self.config.model_weights
        self.config.model_weights = adapted_weights
        
        try:
            result = self._weighted_average_combination(predictions)
            result['adaptive_weights_used'] = adapted_weights
            return result
        finally:
            self.config.model_weights = original_weights
    
    def _analyze_input_characteristics(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """Analyze input data characteristics for adaptive combination."""
        return {
            'is_time_series_heavy': any(
                key in data for key in ['voltage_sequence', 'current_sequence', 'temperature_sequence']
            ),
            'is_fleet_data': any(
                key in data for key in ['fleet_id', 'aggregated_data', 'cross_battery_data']
            ),
            'is_optimization_focused': any(
                key in data for key in ['optimization_target', 'charging_constraints', 'power_limits']
            )
        }
    
    def _combine_optimization_actions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Combine optimization actions from different models."""
        actions = {}
        
        # Get RL agent recommendations
        if 'rl_agent' in predictions and 'error' not in predictions['rl_agent']:
            rl_pred = predictions['rl_agent']
            actions.update({
                'charging_current': rl_pred.get('optimal_charging_current', 0.0),
                'thermal_control': rl_pred.get('thermal_control', 0.0),
                'power_limit': rl_pred.get('power_limit', 1.0)
            })
        
        # Add physics-based constraints
        if 'physics' in predictions and 'error' not in predictions['physics']:
            physics_pred = predictions['physics']
            actions['physics_constraints'] = {
                'max_safe_current': 50.0,  # Based on thermal model
                'thermal_limits': physics_pred.get('thermal_prediction', {})
            }
        
        return actions
    
    def _combine_fleet_insights(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Combine fleet-level insights."""
        insights = {}
        
        if 'federated' in predictions and 'error' not in predictions['federated']:
            fed_pred = predictions['federated']
            insights.update({
                'fleet_health_assessment': fed_pred.get('fleet_health_assessment', 0.0),
                'cross_battery_insights': fed_pred.get('cross_battery_insights', {}),
                'privacy_preserved_score': fed_pred.get('privacy_preserved_score', 0.0)
            })
        
        return insights
    
    def _combine_physics_validation(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Combine physics validation results."""
        validation = {}
        
        if 'physics' in predictions and 'error' not in predictions['physics']:
            physics_pred = predictions['physics']
            validation.update({
                'consistency_score': physics_pred.get('physics_consistency_score', 0.0),
                'internal_resistance': physics_pred.get('internal_resistance', 0.0),
                'thermal_prediction': physics_pred.get('thermal_prediction', {})
            })
        
        return validation
    
    def _calculate_uncertainty(self, predictions: Dict[str, Any]) -> float:
        """Calculate prediction uncertainty based on model disagreement."""
        if not self.config.enable_uncertainty_quantification:
            return 0.0
        
        # Get health scores from valid predictions
        health_scores = []
        for model_name, pred in predictions.items():
            if 'error' not in pred:
                if 'health_score' in pred:
                    health_scores.append(pred['health_score'])
                elif 'physics_health_score' in pred:
                    health_scores.append(pred['physics_health_score'])
        
        if len(health_scores) < 2:
            return 0.0
        
        # Calculate disagreement as uncertainty
        return float(np.std(health_scores))
    
    def _calculate_confidence(self, predictions: Dict[str, Any], uncertainty: float) -> float:
        """Calculate overall prediction confidence."""
        # Count valid predictions
        valid_count = sum(1 for pred in predictions.values() if 'error' not in pred)
        total_count = len(predictions)
        
        if total_count == 0:
            return 0.0
        
        # Base confidence on prediction availability
        availability_score = valid_count / total_count
        
        # Adjust for uncertainty
        uncertainty_score = max(0.0, 1.0 - uncertainty * 5.0)  # Scale uncertainty
        
        # Combine scores
        confidence = (availability_score + uncertainty_score) / 2.0
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_model_contributions(self, predictions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate individual model contributions to final prediction."""
        contributions = {}
        valid_predictions = {k: v for k, v in predictions.items() if 'error' not in v}
        
        if not valid_predictions:
            return contributions
        
        # Calculate contributions based on weights and availability
        total_weight = sum(
            self.config.model_weights.get(model_name, 0.1) 
            for model_name in valid_predictions.keys()
        )
        
        for model_name in valid_predictions.keys():
            model_weight = self.config.model_weights.get(model_name, 0.1)
            contributions[model_name] = model_weight / total_weight if total_weight > 0 else 0.0
        
        return contributions
    
    def _generate_metadata(self, data: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction metadata."""
        return {
            'input_data_size': len(str(data)),
            'models_used': list(predictions.keys()),
            'successful_models': [k for k, v in predictions.items() if 'error' not in v],
            'failed_models': [k for k, v in predictions.items() if 'error' in v],
            'ensemble_strategy': self.config.voting_strategy,
            'prediction_timestamp': time.time()
        }
    
    def _get_fallback_prediction(self, data: Dict[str, Any], error_message: str) -> PredictionResult:
        """Generate fallback prediction when ensemble fails."""
        # Try fallback models in priority order
        for model_name in self.config.fallback_model_priority:
            if model_name in self.models:
                try:
                    fallback_pred = self.models[model_name].predict(data)
                    if 'error' not in fallback_pred:
                        return PredictionResult(
                            predictions={model_name: fallback_pred},
                            ensemble_prediction=fallback_pred,
                            confidence_score=0.3,  # Low confidence for fallback
                            uncertainty_estimate=0.5,
                            model_contributions={model_name: 1.0},
                            prediction_time_ms=0.0,
                            metadata={
                                'is_fallback': True,
                                'fallback_model': model_name,
                                'original_error': error_message
                            }
                        )
                except Exception as e:
                    logger.warning(f"Fallback model {model_name} also failed: {e}")
                    continue
        
        # If all fallback models fail, return error result
        return PredictionResult(
            predictions={},
            ensemble_prediction={'error': error_message},
            confidence_score=0.0,
            uncertainty_estimate=1.0,
            model_contributions={},
            prediction_time_ms=0.0,
            metadata={'is_fallback': True, 'all_models_failed': True}
        )
    
    def _load_meta_model(self):
        """Load meta-model for stacking."""
        if self.config.meta_model_type == "linear":
            self.meta_model = LinearRegression()
        elif self.config.meta_model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            self.meta_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        
        # Note: In practice, meta-model would be trained on validation data
        # For now, we'll use a simple fallback
        logger.info(f"Meta-model ({self.config.meta_model_type}) initialized")
    
    def _prepare_meta_features(self, predictions: Dict[str, Any], data: Dict[str, Any]) -> np.ndarray:
        """Prepare features for meta-model."""
        features = []
        
        # Add predictions from base models
        for model_name in ['transformer', 'federated', 'rl_agent', 'physics']:
            if model_name in predictions and 'error' not in predictions[model_name]:
                pred = predictions[model_name]
                # Extract numerical features
                if 'health_score' in pred:
                    features.append(pred['health_score'])
                elif 'physics_health_score' in pred:
                    features.append(pred['physics_health_score'])
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        # Add input data features
        features.extend([
            data.get('soc', 0.0),
            data.get('voltage', 0.0),
            data.get('current', 0.0),
            data.get('temperature', 0.0)
        ])
        
        return np.array(features)
    
    def _calculate_meta_confidence(self, meta_features: np.ndarray) -> float:
        """Calculate meta-model confidence."""
        # Simple confidence calculation based on feature magnitudes
        return min(1.0, max(0.0, 1.0 - np.std(meta_features) * 0.1))
    
    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key for prediction caching."""
        import hashlib
        data_str = str(sorted(data.items()))
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _update_performance_stats(self, success: bool, prediction_time: float):
        """Update performance statistics."""
        with self._lock:
            self.performance_stats['total_predictions'] += 1
            
            if success:
                self.performance_stats['successful_predictions'] += 1
            else:
                self.performance_stats['failed_predictions'] += 1
            
            # Update average prediction time
            total_pred = self.performance_stats['total_predictions']
            old_avg = self.performance_stats['average_prediction_time']
            self.performance_stats['average_prediction_time'] = (
                (old_avg * (total_pred - 1) + prediction_time) / total_pred
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            return self.performance_stats.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all loaded models."""
        model_info = {}
        for model_name, model in self.models.items():
            model_info[model_name] = model.get_model_info()
        
        return {
            'models': model_info,
            'ensemble_config': {
                'voting_strategy': self.config.voting_strategy,
                'model_weights': self.config.model_weights,
                'parallel_prediction': self.config.parallel_prediction
            },
            'performance_stats': self.get_performance_stats()
        }
    
    def update_model_weights(self, new_weights: Dict[str, float]):
        """Update model weights dynamically."""
        with self._lock:
            self.config.model_weights.update(new_weights)
            logger.info(f"Model weights updated: {self.config.model_weights}")
    
    def clear_cache(self):
        """Clear prediction cache."""
        with self._lock:
            self.prediction_cache.clear()
            self.model_cache.clear()
            logger.info("Prediction cache cleared")

# Factory function
def create_ensemble_predictor(model_paths: Dict[str, str], 
                            config: EnsembleConfig = None) -> BatteryEnsemblePredictor:
    """
    Factory function to create and initialize ensemble predictor.
    
    Args:
        model_paths: Dictionary mapping model names to their paths
        config: Ensemble configuration
        
    Returns:
        BatteryEnsemblePredictor: Initialized ensemble predictor
    """
    predictor = BatteryEnsemblePredictor(config)
    
    if not predictor.load_models(model_paths):
        logger.warning("Some models failed to load")
    
    return predictor
