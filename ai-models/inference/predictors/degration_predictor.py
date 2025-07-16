"""
BatteryMind - Degradation Predictor

Advanced degradation forecasting system for battery health prediction using
transformer models, physics-based constraints, and uncertainty quantification.
Provides real-time degradation predictions for proactive maintenance and
optimization decisions.

Features:
- Multi-horizon degradation forecasting
- Physics-informed predictions with uncertainty quantification
- Real-time streaming inference capabilities
- Configurable prediction horizons and confidence intervals
- Integration with battery management systems
- Automated model fallback and ensemble predictions

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json
import pickle
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import from BatteryMind modules
from ..transformers.degradation_forecaster.model import DegradationTransformer
from ..transformers.degradation_forecaster.predictor import DegradationPredictor as BasePredictor
from ..training_data.preprocessing_scripts.normalization import BatteryDataNormalizer
from ..training_data.preprocessing_scripts.feature_extractor import BatteryFeatureExtractor
from ..utils.model_utils import ModelLoader, ModelValidator
from ..utils.data_utils import DataProcessor, TimeSeriesProcessor
from ..utils.visualization import PredictionVisualizer
from ..utils.logging_utils import setup_logger

# Configure logging
logger = setup_logger(__name__)

@dataclass
class DegradationPredictionConfig:
    """
    Configuration for degradation prediction parameters.
    
    Attributes:
        # Model configuration
        model_path (str): Path to trained degradation model
        model_type (str): Type of degradation model
        ensemble_models (List[str]): List of ensemble model paths
        
        # Prediction parameters
        prediction_horizons (List[int]): Prediction horizons in days
        sequence_length (int): Input sequence length
        confidence_intervals (List[float]): Confidence interval levels
        
        # Data processing
        normalization_config (Dict): Normalization configuration
        feature_extraction_config (Dict): Feature extraction configuration
        
        # Performance settings
        batch_size (int): Batch size for inference
        device (str): Computing device ('cpu', 'cuda', 'auto')
        max_parallel_predictions (int): Maximum parallel predictions
        
        # Quality control
        prediction_validation (bool): Enable prediction validation
        uncertainty_quantification (bool): Enable uncertainty quantification
        physics_constraints (bool): Apply physics-based constraints
        
        # Monitoring
        enable_monitoring (bool): Enable prediction monitoring
        log_predictions (bool): Log prediction results
        alert_thresholds (Dict[str, float]): Alert thresholds
    """
    # Model configuration
    model_path: str = "./model-artifacts/trained_models/transformer_v1.0/"
    model_type: str = "transformer"
    ensemble_models: List[str] = field(default_factory=list)
    
    # Prediction parameters
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 7, 30, 90, 365])
    sequence_length: int = 100
    confidence_intervals: List[float] = field(default_factory=lambda: [0.68, 0.95])
    
    # Data processing
    normalization_config: Dict = field(default_factory=dict)
    feature_extraction_config: Dict = field(default_factory=dict)
    
    # Performance settings
    batch_size: int = 32
    device: str = "auto"
    max_parallel_predictions: int = 100
    
    # Quality control
    prediction_validation: bool = True
    uncertainty_quantification: bool = True
    physics_constraints: bool = True
    
    # Monitoring
    enable_monitoring: bool = True
    log_predictions: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'degradation_rate': 0.05,
        'capacity_loss': 0.2,
        'prediction_confidence': 0.8
    })

class DegradationPredictor:
    """
    Production-ready degradation predictor with advanced forecasting capabilities.
    """
    
    def __init__(self, config: DegradationPredictionConfig):
        self.config = config
        self.model = None
        self.ensemble_models = []
        self.normalizer = None
        self.feature_extractor = None
        self.device = self._setup_device()
        self.prediction_cache = {}
        self.monitoring_stats = {}
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"DegradationPredictor initialized with device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.config.device)
    
    def _initialize_components(self):
        """Initialize all predictor components."""
        # Load primary model
        self._load_primary_model()
        
        # Load ensemble models if configured
        if self.config.ensemble_models:
            self._load_ensemble_models()
        
        # Initialize data processors
        self._initialize_data_processors()
        
        # Setup monitoring
        if self.config.enable_monitoring:
            self._setup_monitoring()
    
    def _load_primary_model(self):
        """Load the primary degradation model."""
        try:
            model_loader = ModelLoader(self.config.model_path)
            self.model = model_loader.load_model(self.config.model_type)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Primary model loaded from {self.config.model_path}")
        except Exception as e:
            logger.error(f"Failed to load primary model: {e}")
            raise
    
    def _load_ensemble_models(self):
        """Load ensemble models for improved predictions."""
        for model_path in self.config.ensemble_models:
            try:
                model_loader = ModelLoader(model_path)
                ensemble_model = model_loader.load_model(self.config.model_type)
                ensemble_model.to(self.device)
                ensemble_model.eval()
                self.ensemble_models.append(ensemble_model)
                
                logger.info(f"Ensemble model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load ensemble model {model_path}: {e}")
    
    def _initialize_data_processors(self):
        """Initialize data processing components."""
        # Initialize normalizer
        self.normalizer = BatteryDataNormalizer(self.config.normalization_config)
        
        # Initialize feature extractor
        self.feature_extractor = BatteryFeatureExtractor(self.config.feature_extraction_config)
        
        # Load fitted processors if available
        self._load_fitted_processors()
    
    def _load_fitted_processors(self):
        """Load fitted data processors."""
        try:
            normalizer_path = Path(self.config.model_path) / "normalizer.pkl"
            if normalizer_path.exists():
                with open(normalizer_path, 'rb') as f:
                    self.normalizer = pickle.load(f)
                logger.info("Fitted normalizer loaded")
        except Exception as e:
            logger.warning(f"Failed to load fitted normalizer: {e}")
    
    def _setup_monitoring(self):
        """Setup prediction monitoring."""
        self.monitoring_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_prediction_time': 0.0,
            'prediction_accuracy_history': [],
            'uncertainty_scores': [],
            'alert_history': []
        }
    
    def predict_degradation(self, 
                          battery_data: Union[pd.DataFrame, Dict[str, np.ndarray]],
                          prediction_horizons: Optional[List[int]] = None,
                          return_uncertainty: bool = True) -> Dict[str, Any]:
        """
        Predict battery degradation for specified horizons.
        
        Args:
            battery_data: Historical battery data
            prediction_horizons: Prediction horizons in days
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary containing predictions and metadata
        """
        start_time = datetime.now()
        
        try:
            # Validate input data
            self._validate_input_data(battery_data)
            
            # Use default horizons if not provided
            if prediction_horizons is None:
                prediction_horizons = self.config.prediction_horizons
            
            # Preprocess data
            processed_data = self._preprocess_data(battery_data)
            
            # Generate predictions
            predictions = self._generate_predictions(processed_data, prediction_horizons)
            
            # Add uncertainty quantification if enabled
            if return_uncertainty and self.config.uncertainty_quantification:
                predictions = self._add_uncertainty_estimates(predictions, processed_data)
            
            # Apply physics constraints if enabled
            if self.config.physics_constraints:
                predictions = self._apply_physics_constraints(predictions)
            
            # Validate predictions
            if self.config.prediction_validation:
                self._validate_predictions(predictions)
            
            # Update monitoring stats
            self._update_monitoring_stats(start_time, True)
            
            # Log predictions if enabled
            if self.config.log_predictions:
                self._log_predictions(predictions)
            
            return {
                'predictions': predictions,
                'metadata': {
                    'prediction_timestamp': datetime.now().isoformat(),
                    'model_version': '1.0.0',
                    'prediction_horizons': prediction_horizons,
                    'confidence_intervals': self.config.confidence_intervals,
                    'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            self._update_monitoring_stats(start_time, False)
            raise
    
    def _validate_input_data(self, data: Union[pd.DataFrame, Dict[str, np.ndarray]]):
        """Validate input data format and quality."""
        if isinstance(data, pd.DataFrame):
            # Check required columns
            required_columns = ['timestamp', 'voltage', 'current', 'temperature', 'soc']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Check data length
            if len(data) < self.config.sequence_length:
                raise ValueError(f"Insufficient data length: {len(data)} < {self.config.sequence_length}")
            
            # Check for missing values
            if data.isnull().any().any():
                logger.warning("Input data contains missing values")
        
        elif isinstance(data, dict):
            # Validate dictionary format
            for key, values in data.items():
                if not isinstance(values, np.ndarray):
                    raise ValueError(f"Data values must be numpy arrays, got {type(values)} for {key}")
                
                if len(values) < self.config.sequence_length:
                    raise ValueError(f"Insufficient data length for {key}: {len(values)} < {self.config.sequence_length}")
    
    def _preprocess_data(self, data: Union[pd.DataFrame, Dict[str, np.ndarray]]) -> torch.Tensor:
        """Preprocess input data for model inference."""
        # Convert to DataFrame if necessary
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        # Extract features
        features = self.feature_extractor.extract_features(data)
        
        # Normalize features
        normalized_features = self.normalizer.transform(features)
        
        # Create sequences
        sequences = self._create_sequences(normalized_features)
        
        # Convert to tensor
        tensor_data = torch.tensor(sequences, dtype=torch.float32).to(self.device)
        
        return tensor_data
    
    def _create_sequences(self, features: np.ndarray) -> np.ndarray:
        """Create sequences from feature data."""
        sequences = []
        
        for i in range(len(features) - self.config.sequence_length + 1):
            sequence = features[i:i + self.config.sequence_length]
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def _generate_predictions(self, 
                            processed_data: torch.Tensor,
                            prediction_horizons: List[int]) -> Dict[str, Any]:
        """Generate degradation predictions for specified horizons."""
        predictions = {}
        
        with torch.no_grad():
            # Primary model predictions
            primary_predictions = self._predict_with_model(self.model, processed_data, prediction_horizons)
            predictions['primary'] = primary_predictions
            
            # Ensemble predictions if available
            if self.ensemble_models:
                ensemble_predictions = []
                for ensemble_model in self.ensemble_models:
                    ensemble_pred = self._predict_with_model(ensemble_model, processed_data, prediction_horizons)
                    ensemble_predictions.append(ensemble_pred)
                
                # Combine ensemble predictions
                predictions['ensemble'] = self._combine_ensemble_predictions(ensemble_predictions)
                
                # Use ensemble as final prediction
                predictions['final'] = predictions['ensemble']
            else:
                predictions['final'] = predictions['primary']
        
        return predictions
    
    def _predict_with_model(self, 
                          model: nn.Module, 
                          data: torch.Tensor,
                          horizons: List[int]) -> Dict[str, np.ndarray]:
        """Generate predictions with a single model."""
        model_predictions = {}
        
        for horizon in horizons:
            # Adjust model for specific horizon if needed
            horizon_predictions = []
            
            # Process in batches
            for i in range(0, len(data), self.config.batch_size):
                batch_data = data[i:i + self.config.batch_size]
                batch_predictions = model(batch_data, horizon=horizon)
                horizon_predictions.append(batch_predictions.cpu().numpy())
            
            # Combine batch predictions
            model_predictions[f'horizon_{horizon}'] = np.concatenate(horizon_predictions, axis=0)
        
        return model_predictions
    
    def _combine_ensemble_predictions(self, 
                                    ensemble_predictions: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Combine predictions from ensemble models."""
        combined_predictions = {}
        
        # Get all horizon keys
        horizon_keys = ensemble_predictions[0].keys()
        
        for horizon_key in horizon_keys:
            # Collect predictions for this horizon
            horizon_preds = [pred[horizon_key] for pred in ensemble_predictions]
            
            # Combine using weighted average (equal weights for now)
            combined_pred = np.mean(horizon_preds, axis=0)
            combined_predictions[horizon_key] = combined_pred
        
        return combined_predictions
    
    def _add_uncertainty_estimates(self, 
                                 predictions: Dict[str, Any],
                                 processed_data: torch.Tensor) -> Dict[str, Any]:
        """Add uncertainty quantification to predictions."""
        uncertainty_estimates = {}
        
        # Use ensemble variance for uncertainty if available
        if 'ensemble' in predictions:
            uncertainty_estimates = self._calculate_ensemble_uncertainty(predictions)
        else:
            # Use Monte Carlo dropout for uncertainty
            uncertainty_estimates = self._calculate_mc_dropout_uncertainty(processed_data)
        
        # Add confidence intervals
        for horizon_key, pred_values in predictions['final'].items():
            uncertainty_estimates[horizon_key] = self._calculate_confidence_intervals(
                pred_values, uncertainty_estimates.get(horizon_key, np.ones_like(pred_values) * 0.1)
            )
        
        predictions['uncertainty'] = uncertainty_estimates
        return predictions
    
    def _calculate_ensemble_uncertainty(self, predictions: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Calculate uncertainty from ensemble variance."""
        uncertainty = {}
        
        # This would require storing individual ensemble predictions
        # For now, return placeholder uncertainty
        for horizon_key in predictions['final'].keys():
            pred_values = predictions['final'][horizon_key]
            uncertainty[horizon_key] = np.ones_like(pred_values) * 0.05  # 5% uncertainty
        
        return uncertainty
    
    def _calculate_mc_dropout_uncertainty(self, processed_data: torch.Tensor) -> Dict[str, np.ndarray]:
        """Calculate uncertainty using Monte Carlo dropout."""
        uncertainty = {}
        
        # Enable dropout for uncertainty estimation
        self.model.train()
        
        n_samples = 50  # Number of MC samples
        mc_predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                sample_pred = self.model(processed_data)
                mc_predictions.append(sample_pred.cpu().numpy())
        
        # Calculate variance across samples
        mc_predictions = np.array(mc_predictions)
        uncertainty_values = np.var(mc_predictions, axis=0)
        
        # Return to eval mode
        self.model.eval()
        
        return {'mc_uncertainty': uncertainty_values}
    
    def _calculate_confidence_intervals(self, 
                                     predictions: np.ndarray,
                                     uncertainty: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals for predictions."""
        confidence_intervals = {}
        
        for confidence_level in self.config.confidence_intervals:
            # Calculate z-score for confidence level
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            
            # Calculate interval bounds
            lower_bound = predictions - z_score * uncertainty
            upper_bound = predictions + z_score * uncertainty
            
            confidence_intervals[f'ci_{int(confidence_level*100)}'] = {
                'lower': lower_bound,
                'upper': upper_bound
            }
        
        return confidence_intervals
    
    def _apply_physics_constraints(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Apply physics-based constraints to predictions."""
        for horizon_key, pred_values in predictions['final'].items():
            # Ensure predictions are within physical bounds
            
            # State of Health constraints (0.5 to 1.0)
            if 'soh' in horizon_key.lower():
                pred_values = np.clip(pred_values, 0.5, 1.0)
            
            # Capacity constraints (positive values)
            if 'capacity' in horizon_key.lower():
                pred_values = np.maximum(pred_values, 0.0)
            
            # Degradation rate constraints (positive values)
            if 'degradation' in horizon_key.lower():
                pred_values = np.maximum(pred_values, 0.0)
            
            predictions['final'][horizon_key] = pred_values
        
        return predictions
    
    def _validate_predictions(self, predictions: Dict[str, Any]):
        """Validate prediction quality and raise alerts if needed."""
        for horizon_key, pred_values in predictions['final'].items():
            # Check for NaN or infinite values
            if np.any(np.isnan(pred_values)) or np.any(np.isinf(pred_values)):
                raise ValueError(f"Invalid prediction values in {horizon_key}")
            
            # Check prediction ranges
            if 'soh' in horizon_key.lower():
                if np.any(pred_values < 0.0) or np.any(pred_values > 1.0):
                    logger.warning(f"SoH predictions out of range in {horizon_key}")
            
            # Check for extreme degradation predictions
            if 'degradation' in horizon_key.lower():
                if np.any(pred_values > self.config.alert_thresholds['degradation_rate']):
                    self._trigger_alert('high_degradation_rate', horizon_key, pred_values)
    
    def _trigger_alert(self, alert_type: str, horizon_key: str, values: np.ndarray):
        """Trigger monitoring alert."""
        alert_info = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': alert_type,
            'horizon': horizon_key,
            'max_value': np.max(values),
            'mean_value': np.mean(values)
        }
        
        self.monitoring_stats['alert_history'].append(alert_info)
        logger.warning(f"Alert triggered: {alert_type} in {horizon_key}")
    
    def _update_monitoring_stats(self, start_time: datetime, success: bool):
        """Update monitoring statistics."""
        self.monitoring_stats['total_predictions'] += 1
        
        if success:
            self.monitoring_stats['successful_predictions'] += 1
        else:
            self.monitoring_stats['failed_predictions'] += 1
        
        # Update average prediction time
        prediction_time = (datetime.now() - start_time).total_seconds()
        current_avg = self.monitoring_stats['average_prediction_time']
        total_predictions = self.monitoring_stats['total_predictions']
        
        self.monitoring_stats['average_prediction_time'] = (
            (current_avg * (total_predictions - 1) + prediction_time) / total_predictions
        )
    
    def _log_predictions(self, predictions: Dict[str, Any]):
        """Log prediction results."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction_summary': {
                horizon: {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
                for horizon, values in predictions['final'].items()
            }
        }
        
        logger.info(f"Prediction logged: {log_entry}")
    
    def predict_single_battery(self, 
                             battery_id: str,
                             battery_data: pd.DataFrame,
                             prediction_horizons: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Predict degradation for a single battery.
        
        Args:
            battery_id: Unique battery identifier
            battery_data: Historical battery data
            prediction_horizons: Prediction horizons in days
            
        Returns:
            Dictionary containing predictions and metadata
        """
        result = self.predict_degradation(battery_data, prediction_horizons)
        result['battery_id'] = battery_id
        result['metadata']['battery_id'] = battery_id
        
        return result
    
    def predict_batch(self, 
                     batch_data: Dict[str, pd.DataFrame],
                     prediction_horizons: Optional[List[int]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Predict degradation for multiple batteries.
        
        Args:
            batch_data: Dictionary mapping battery IDs to data
            prediction_horizons: Prediction horizons in days
            
        Returns:
            Dictionary mapping battery IDs to predictions
        """
        batch_results = {}
        
        for battery_id, battery_data in batch_data.items():
            try:
                result = self.predict_single_battery(battery_id, battery_data, prediction_horizons)
                batch_results[battery_id] = result
            except Exception as e:
                logger.error(f"Failed to predict for battery {battery_id}: {e}")
                batch_results[battery_id] = {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        return batch_results
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        return self.monitoring_stats.copy()
    
    def reset_monitoring_stats(self):
        """Reset monitoring statistics."""
        self.monitoring_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_prediction_time': 0.0,
            'prediction_accuracy_history': [],
            'uncertainty_scores': [],
            'alert_history': []
        }
    
    def validate_model_performance(self, 
                                 validation_data: pd.DataFrame,
                                 ground_truth: pd.DataFrame) -> Dict[str, float]:
        """
        Validate model performance on validation data.
        
        Args:
            validation_data: Validation input data
            ground_truth: Ground truth degradation values
            
        Returns:
            Dictionary of performance metrics
        """
        predictions = self.predict_degradation(validation_data, return_uncertainty=False)
        
        # Calculate performance metrics
        metrics = {}
        
        for horizon_key, pred_values in predictions['predictions']['final'].items():
            if horizon_key in ground_truth.columns:
                true_values = ground_truth[horizon_key].values
                
                # Ensure same length
                min_len = min(len(pred_values), len(true_values))
                pred_values = pred_values[:min_len]
                true_values = true_values[:min_len]
                
                # Calculate metrics
                metrics[f'{horizon_key}_mae'] = mean_absolute_error(true_values, pred_values)
                metrics[f'{horizon_key}_mse'] = mean_squared_error(true_values, pred_values)
                metrics[f'{horizon_key}_rmse'] = np.sqrt(metrics[f'{horizon_key}_mse'])
                metrics[f'{horizon_key}_r2'] = r2_score(true_values, pred_values)
        
        return metrics
    
    def save_predictions(self, predictions: Dict[str, Any], filepath: str):
        """Save predictions to file."""
        with open(filepath, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        logger.info(f"Predictions saved to {filepath}")
    
    def load_predictions(self, filepath: str) -> Dict[str, Any]:
        """Load predictions from file."""
        with open(filepath, 'r') as f:
            predictions = json.load(f)
        
        logger.info(f"Predictions loaded from {filepath}")
        return predictions

# Factory function
def create_degradation_predictor(config: Optional[DegradationPredictionConfig] = None) -> DegradationPredictor:
    """
    Factory function to create a degradation predictor.
    
    Args:
        config: Predictor configuration
        
    Returns:
        Configured DegradationPredictor instance
    """
    if config is None:
        config = DegradationPredictionConfig()
    
    return DegradationPredictor(config)
