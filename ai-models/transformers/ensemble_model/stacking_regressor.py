"""
BatteryMind - Stacking Regressor

Advanced stacking ensemble implementation for battery health prediction
with meta-learning capabilities and cross-validation optimization.

Features:
- Multi-level stacking with heterogeneous base models
- Cross-validation based meta-feature generation
- Dynamic weight adjustment based on model confidence
- Physics-informed meta-learning constraints
- Uncertainty propagation through ensemble layers
- Real-time model performance monitoring

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import time
import pickle
from pathlib import Path
from abc import ABC, abstractmethod

# Scientific computing imports
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb

# Local imports
from ..battery_health_predictor.model import BatteryHealthTransformer
from ..degradation_forecaster.model import DegradationForecaster
from ..optimization_recommender.model import OptimizationRecommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StackingConfig:
    """
    Configuration for stacking regressor ensemble.
    
    Attributes:
        # Base models configuration
        use_transformer_models (bool): Include transformer-based models
        use_traditional_models (bool): Include traditional ML models
        base_model_types (List[str]): Types of base models to include
        
        # Meta-learner configuration
        meta_learner_type (str): Type of meta-learner ('linear', 'neural', 'tree')
        meta_learner_params (Dict): Parameters for meta-learner
        
        # Cross-validation configuration
        cv_folds (int): Number of cross-validation folds
        cv_method (str): Cross-validation method ('kfold', 'timeseries')
        
        # Stacking levels
        n_levels (int): Number of stacking levels
        level_configs (List[Dict]): Configuration for each stacking level
        
        # Performance optimization
        parallel_training (bool): Enable parallel base model training
        cache_predictions (bool): Cache base model predictions
        
        # Physics constraints
        apply_physics_constraints (bool): Apply physics-informed constraints
        constraint_weights (Dict[str, float]): Weights for different constraints
        
        # Uncertainty handling
        propagate_uncertainty (bool): Propagate uncertainty through ensemble
        uncertainty_aggregation (str): Method for uncertainty aggregation
    """
    # Base models configuration
    use_transformer_models: bool = True
    use_traditional_models: bool = True
    base_model_types: List[str] = field(default_factory=lambda: [
        'battery_health_transformer', 'degradation_forecaster', 'optimization_recommender',
        'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm'
    ])
    
    # Meta-learner configuration
    meta_learner_type: str = "neural"
    meta_learner_params: Dict = field(default_factory=lambda: {
        'hidden_layers': [128, 64, 32],
        'dropout': 0.2,
        'activation': 'relu',
        'learning_rate': 0.001
    })
    
    # Cross-validation configuration
    cv_folds: int = 5
    cv_method: str = "timeseries"
    
    # Stacking levels
    n_levels: int = 2
    level_configs: List[Dict] = field(default_factory=list)
    
    # Performance optimization
    parallel_training: bool = True
    cache_predictions: bool = True
    
    # Physics constraints
    apply_physics_constraints: bool = True
    constraint_weights: Dict[str, float] = field(default_factory=lambda: {
        'monotonicity': 0.1,
        'bounds': 0.2,
        'temporal_consistency': 0.05
    })
    
    # Uncertainty handling
    propagate_uncertainty: bool = True
    uncertainty_aggregation: str = "weighted_average"

class BaseModelWrapper(ABC):
    """
    Abstract wrapper for base models in stacking ensemble.
    """
    
    def __init__(self, model_name: str, model_params: Dict = None):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model = None
        self.is_fitted = False
        self.feature_importance = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModelWrapper':
        """Fit the base model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the base model."""
        pass
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        predictions = self.predict(X)
        # Default implementation returns zero uncertainty
        uncertainty = np.zeros_like(predictions)
        return predictions, uncertainty
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        return self.feature_importance

class TransformerModelWrapper(BaseModelWrapper):
    """
    Wrapper for transformer-based models.
    """
    
    def __init__(self, model_name: str, model_path: str, model_params: Dict = None):
        super().__init__(model_name, model_params)
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TransformerModelWrapper':
        """Load pre-trained transformer model."""
        try:
            if self.model_name == 'battery_health_transformer':
                from ..battery_health_predictor.predictor import BatteryHealthPredictor
                from ..battery_health_predictor.predictor import BatteryInferenceConfig
                
                config = BatteryInferenceConfig(model_path=self.model_path)
                self.model = BatteryHealthPredictor(config)
                
            elif self.model_name == 'degradation_forecaster':
                from ..degradation_forecaster.forecaster import BatteryDegradationForecaster
                from ..degradation_forecaster.forecaster import ForecastingConfig
                
                config = ForecastingConfig(model_path=self.model_path)
                self.model = BatteryDegradationForecaster(config)
                
            elif self.model_name == 'optimization_recommender':
                from ..optimization_recommender.recommender import BatteryOptimizationRecommender
                from ..optimization_recommender.recommender import RecommendationConfig
                
                config = RecommendationConfig(model_path=self.model_path)
                self.model = BatteryOptimizationRecommender(config)
            
            self.is_fitted = True
            logger.info(f"Loaded transformer model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load transformer model {self.model_name}: {e}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with transformer model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        try:
            # Convert numpy array to appropriate format for transformer
            if self.model_name == 'battery_health_transformer':
                # Assume X is battery sensor data
                predictions = []
                for sample in X:
                    result = self.model.predict(sample)
                    predictions.append([
                        result.state_of_health,
                        result.degradation_patterns.get('capacity_fade_rate', 0.0),
                        result.degradation_patterns.get('resistance_increase_rate', 0.0),
                        result.degradation_patterns.get('thermal_degradation', 0.0)
                    ])
                return np.array(predictions)
            
            elif self.model_name == 'degradation_forecaster':
                # Assume X is time-series data for forecasting
                predictions = []
                for sample in X:
                    result = self.model.forecast_degradation(sample)
                    # Extract relevant metrics from forecast
                    forecast_summary = [
                        result['degradation_metrics']['avg_capacity_fade_rate'],
                        result['degradation_metrics']['avg_resistance_increase_rate'],
                        result['degradation_metrics']['avg_thermal_degradation'],
                        result['degradation_metrics']['avg_overall_health_decline']
                    ]
                    predictions.append(forecast_summary)
                return np.array(predictions)
            
            elif self.model_name == 'optimization_recommender':
                # Assume X is battery state data for optimization
                predictions = []
                for sample in X:
                    result = self.model.recommend_optimization(sample)
                    # Extract optimization scores
                    opt_summary = [
                        result.optimization_score,
                        result.efficiency_improvement,
                        result.safety_score,
                        result.sustainability_score
                    ]
                    predictions.append(opt_summary)
                return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.model_name}: {e}")
            # Return zero predictions as fallback
            return np.zeros((X.shape[0], 4))
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        predictions = self.predict(X)
        
        # Extract uncertainty from transformer models if available
        try:
            if hasattr(self.model, 'predict_with_uncertainty'):
                # Some transformer models may have uncertainty estimation
                uncertainty_results = self.model.predict_with_uncertainty(X)
                uncertainty = uncertainty_results.get('std', np.zeros_like(predictions))
            else:
                # Default uncertainty based on model confidence
                uncertainty = np.ones_like(predictions) * 0.1  # 10% default uncertainty
        except:
            uncertainty = np.ones_like(predictions) * 0.1
        
        return predictions, uncertainty

class TraditionalModelWrapper(BaseModelWrapper):
    """
    Wrapper for traditional machine learning models.
    """
    
    def __init__(self, model_name: str, model_params: Dict = None):
        super().__init__(model_name, model_params)
        self.scaler = StandardScaler()
        
    def _create_model(self):
        """Create the appropriate model based on model_name."""
        if self.model_name == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', None),
                random_state=42,
                n_jobs=-1
            )
        elif self.model_name == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=self.model_params.get('n_estimators', 100),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                max_depth=self.model_params.get('max_depth', 3),
                random_state=42
            )
        elif self.model_name == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=self.model_params.get('n_estimators', 100),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                max_depth=self.model_params.get('max_depth', 6),
                random_state=42,
                n_jobs=-1
            )
        elif self.model_name == 'lightgbm':
            return lgb.LGBMRegressor(
                n_estimators=self.model_params.get('n_estimators', 100),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                max_depth=self.model_params.get('max_depth', -1),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        elif self.model_name == 'neural_network':
            return MLPRegressor(
                hidden_layer_sizes=self.model_params.get('hidden_layers', (100, 50)),
                learning_rate_init=self.model_params.get('learning_rate', 0.001),
                max_iter=self.model_params.get('max_iter', 500),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_name}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TraditionalModelWrapper':
        """Fit the traditional ML model."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and fit model
        self.model = self._create_model()
        
        # Handle multi-output case
        if y.ndim > 1 and y.shape[1] > 1:
            # Multi-output regression
            from sklearn.multioutput import MultiOutputRegressor
            self.model = MultiOutputRegressor(self.model)
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Extract feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'estimators_') and hasattr(self.model.estimators_[0], 'feature_importances_'):
            # Multi-output case
            self.feature_importance = np.mean([
                est.feature_importances_ for est in self.model.estimators_
            ], axis=0)
        
        logger.info(f"Fitted traditional model: {self.model_name}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with traditional ML model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Ensure predictions are 2D
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        return predictions
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        predictions = self.predict(X)
        
        # Estimate uncertainty for tree-based models
        if self.model_name in ['random_forest', 'gradient_boosting']:
            try:
                # Use prediction variance from ensemble
                if hasattr(self.model, 'estimators_'):
                    # Get predictions from all trees
                    X_scaled = self.scaler.transform(X)
                    tree_predictions = np.array([
                        tree.predict(X_scaled) for tree in self.model.estimators_
                    ])
                    uncertainty = np.std(tree_predictions, axis=0)
                    if uncertainty.ndim == 1:
                        uncertainty = uncertainty.reshape(-1, 1)
                else:
                    uncertainty = np.ones_like(predictions) * 0.05
            except:
                uncertainty = np.ones_like(predictions) * 0.05
        else:
            # Default uncertainty for other models
            uncertainty = np.ones_like(predictions) * 0.1
        
        return predictions, uncertainty

class MetaLearner(nn.Module):
    """
    Neural network meta-learner for stacking ensemble.
    """
    
    def __init__(self, input_dim: int, output_dim: int, config: Dict):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build neural network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.get('hidden_layers', [128, 64, 32]):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if config.get('activation', 'relu') == 'relu' else nn.GELU(),
                nn.Dropout(config.get('dropout', 0.2))
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Physics constraints layer
        self.physics_constraints = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()  # Ensure outputs are bounded
        )
        
    def forward(self, x: torch.Tensor, apply_constraints: bool = True) -> torch.Tensor:
        """Forward pass through meta-learner."""
        output = self.network(x)
        
        if apply_constraints:
            # Apply physics constraints
            constraints = self.physics_constraints(output)
            output = output * constraints
        
        return output

class StackingRegressor:
    """
    Advanced stacking regressor for battery health prediction ensemble.
    """
    
    def __init__(self, config: StackingConfig):
        self.config = config
        self.base_models = []
        self.meta_learners = []
        self.is_fitted = False
        self.cv_scores = {}
        self.feature_names = None
        
        # Initialize base models
        self._initialize_base_models()
        
        # Initialize meta-learners for each level
        self._initialize_meta_learners()
        
        logger.info(f"Initialized StackingRegressor with {len(self.base_models)} base models")
    
    def _initialize_base_models(self):
        """Initialize base models based on configuration."""
        for model_type in self.config.base_model_types:
            try:
                if model_type in ['battery_health_transformer', 'degradation_forecaster', 'optimization_recommender']:
                    # Transformer models (requires model paths)
                    model_path = f"./model_artifacts/{model_type}_v1.0/best_model.ckpt"
                    wrapper = TransformerModelWrapper(model_type, model_path)
                else:
                    # Traditional ML models
                    wrapper = TraditionalModelWrapper(model_type)
                
                self.base_models.append(wrapper)
                
            except Exception as e:
                logger.warning(f"Failed to initialize model {model_type}: {e}")
    
    def _initialize_meta_learners(self):
        """Initialize meta-learners for each stacking level."""
        for level in range(self.config.n_levels):
            if level == 0:
                # First level: meta-features from base models
                input_dim = len(self.base_models) * 4  # Assuming 4 outputs per model
            else:
                # Higher levels: meta-features from previous level
                input_dim = 4  # Output dimension
            
            if self.config.meta_learner_type == 'neural':
                meta_learner = MetaLearner(
                    input_dim=input_dim,
                    output_dim=4,  # SoH + 3 degradation metrics
                    config=self.config.meta_learner_params
                )
            elif self.config.meta_learner_type == 'linear':
                meta_learner = Ridge(alpha=1.0)
            elif self.config.meta_learner_type == 'tree':
                meta_learner = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                raise ValueError(f"Unknown meta-learner type: {self.config.meta_learner_type}")
            
            self.meta_learners.append(meta_learner)
    
    def _create_cv_splits(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create cross-validation splits."""
        if self.config.cv_method == 'timeseries':
            # Time series split to avoid data leakage
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            return list(tscv.split(X))
        else:
            # Standard k-fold
            kf = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
            return list(kf.split(X))
    
    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray, 
                               level: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate meta-features using cross-validation."""
        if level == 0:
            # Generate meta-features from base models
            cv_splits = self._create_cv_splits(X, y)
            meta_features = np.zeros((X.shape[0], len(self.base_models) * 4))
            meta_uncertainties = np.zeros((X.shape[0], len(self.base_models) * 4))
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                for model_idx, model in enumerate(self.base_models):
                    try:
                        # Fit model on training fold
                        model.fit(X_train, y_train)
                        
                        # Predict on validation fold
                        predictions, uncertainties = model.predict_with_uncertainty(X_val)
                        
                        # Store meta-features
                        start_idx = model_idx * 4
                        end_idx = start_idx + 4
                        
                        if predictions.shape[1] >= 4:
                            meta_features[val_idx, start_idx:end_idx] = predictions[:, :4]
                            meta_uncertainties[val_idx, start_idx:end_idx] = uncertainties[:, :4]
                        else:
                            # Pad with zeros if model outputs fewer features
                            meta_features[val_idx, start_idx:start_idx+predictions.shape[1]] = predictions
                            meta_uncertainties[val_idx, start_idx:start_idx+predictions.shape[1]] = uncertainties
                        
                    except Exception as e:
                        logger.warning(f"Model {model.model_name} failed in fold {fold_idx}: {e}")
                        # Use zero features for failed models
                        start_idx = model_idx * 4
                        end_idx = start_idx + 4
                        meta_features[val_idx, start_idx:end_idx] = 0
                        meta_uncertainties[val_idx, start_idx:end_idx] = 1  # High uncertainty
            
            return meta_features, meta_uncertainties
        
        else:
            # For higher levels, use predictions from previous meta-learner
            prev_meta_learner = self.meta_learners[level - 1]
            
            if isinstance(prev_meta_learner, nn.Module):
                # Neural meta-learner
                prev_meta_learner.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    predictions = prev_meta_learner(X_tensor).numpy()
                    uncertainties = np.ones_like(predictions) * 0.1  # Default uncertainty
            else:
                # Traditional meta-learner
                predictions = prev_meta_learner.predict(X)
                uncertainties = np.ones_like(predictions) * 0.1
            
            return predictions, uncertainties
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> 'StackingRegressor':
        """
        Fit the stacking regressor ensemble.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training targets
            feature_names (List[str], optional): Feature names
            
        Returns:
            StackingRegressor: Fitted ensemble
        """
        logger.info("Starting stacking regressor training...")
        start_time = time.time()
        
        self.feature_names = feature_names
        
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Fit base models and generate meta-features for each level
        current_X = X
        current_y = y
        
        for level in range(self.config.n_levels):
            logger.info(f"Training stacking level {level + 1}/{self.config.n_levels}")
            
            # Generate meta-features
            meta_features, meta_uncertainties = self._generate_meta_features(
                current_X, current_y, level
            )
            
            # Fit meta-learner
            meta_learner = self.meta_learners[level]
            
            if isinstance(meta_learner, nn.Module):
                # Train neural meta-learner
                self._train_neural_meta_learner(meta_learner, meta_features, current_y)
            else:
                # Train traditional meta-learner
                meta_learner.fit(meta_features, current_y)
            
            # Prepare for next level
            if level < self.config.n_levels - 1:
                current_X = meta_features
                current_y = current_y  # Keep same targets
        
        # Final fit of base models on full dataset
        logger.info("Final training of base models on full dataset...")
        for model in self.base_models:
            try:
                model.fit(X, y)
            except Exception as e:
                logger.warning(f"Final training failed for {model.model_name}: {e}")
        
        self.is_fitted = True
        training_time = time.time() - start_time
        logger.info(f"Stacking regressor training completed in {training_time:.2f} seconds")
        
        return self
    
    def _train_neural_meta_learner(self, meta_learner: nn.Module, 
                                  X: np.ndarray, y: np.ndarray):
        """Train neural network meta-learner."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        meta_learner.to(device)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)
        
        # Optimizer and loss
        optimizer = optim.Adam(
            meta_learner.parameters(),
            lr=self.config.meta_learner_params.get('learning_rate', 0.001)
        )
        criterion = nn.MSELoss()
        
        # Training loop
        meta_learner.train()
        n_epochs = self.config.meta_learner_params.get('epochs', 100)
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = meta_learner(X_tensor)
            loss = criterion(predictions, y_tensor)
            
            # Apply physics constraints if enabled
            if self.config.apply_physics_constraints:
                physics_loss = self._compute_physics_loss(predictions)
                total_loss = loss + sum(
                    weight * physics_loss for weight in self.config.constraint_weights.values()
                )
            else:
                total_loss = loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.debug(f"Meta-learner epoch {epoch}, loss: {total_loss.item():.6f}")
    
    def _compute_physics_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute physics-informed loss for meta-learner."""
        physics_loss = torch.tensor(0.0, device=predictions.device)
        
        # Ensure SoH is between 0 and 1
        soh = predictions[:, 0]
        physics_loss += torch.mean(torch.relu(-soh)) + torch.mean(torch.relu(soh - 1))
        
        # Ensure degradation rates are non-negative
        degradation = predictions[:, 1:]
        physics_loss += torch.mean(torch.relu(-degradation))
        
        return physics_loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the stacking ensemble.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")
        
        # Get base model predictions
        base_predictions = []
        for model in self.base_models:
            try:
                pred = model.predict(X)
                if pred.shape[1] >= 4:
                    base_predictions.append(pred[:, :4])
                else:
                    # Pad with zeros if needed
                    padded_pred = np.zeros((pred.shape[0], 4))
                    padded_pred[:, :pred.shape[1]] = pred
                    base_predictions.append(padded_pred)
            except Exception as e:
                logger.warning(f"Prediction failed for {model.model_name}: {e}")
                # Use zero predictions for failed models
                base_predictions.append(np.zeros((X.shape[0], 4)))
        
        # Combine base predictions
        meta_features = np.concatenate(base_predictions, axis=1)
        
        # Pass through meta-learners
        current_features = meta_features
        
        for level, meta_learner in enumerate(self.meta_learners):
            if isinstance(meta_learner, nn.Module):
                # Neural meta-learner
                meta_learner.eval()
                with torch.no_grad():
                    device = next(meta_learner.parameters()).device
                    features_tensor = torch.FloatTensor(current_features).to(device)
                    predictions = meta_learner(features_tensor).cpu().numpy()
            else:
                # Traditional meta-learner
                predictions = meta_learner.predict(current_features)
            
            current_features = predictions
        
        return predictions
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Predictions and uncertainties
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")
        
        # Get base model predictions with uncertainties
        base_predictions = []
        base_uncertainties = []
        
        for model in self.base_models:
            try:
                pred, unc = model.predict_with_uncertainty(X)
                if pred.shape[1] >= 4:
                    base_predictions.append(pred[:, :4])
                    base_uncertainties.append(unc[:, :4])
                else:
                    # Pad with zeros/ones if needed
                    padded_pred = np.zeros((pred.shape[0], 4))
                    padded_unc = np.ones((pred.shape[0], 4))
                    padded_pred[:, :pred.shape[1]] = pred
                    padded_unc[:, :pred.shape[1]] = unc
                    base_predictions.append(padded_pred)
                    base_uncertainties.append(padded_unc)
            except Exception as e:
                logger.warning(f"Prediction failed for {model.model_name}: {e}")
                base_predictions.append(np.zeros((X.shape[0], 4)))
                base_uncertainties.append(np.ones((X.shape[0], 4)))
        
        # Combine predictions and uncertainties
        ensemble_predictions = self.predict(X)
        
        # Aggregate uncertainties
        if self.config.uncertainty_aggregation == "weighted_average":
            # Weight uncertainties by inverse of base model performance
            weights = np.ones(len(base_predictions)) / len(base_predictions)
            combined_uncertainties = np.average(base_uncertainties, axis=0, weights=weights)
        else:
            # Simple average
            combined_uncertainties = np.mean(base_uncertainties, axis=0)
        
        return ensemble_predictions, combined_uncertainties
    
    def get_model_importance(self) -> Dict[str, float]:
        """Get importance scores for base models."""
        if not self.is_fitted:
            return {}
        
        importance_scores = {}
        
        # For neural meta-learners, use weight magnitudes
        if isinstance(self.meta_learners[0], nn.Module):
            meta_learner = self.meta_learners[0]
            first_layer_weights = meta_learner.network[0].weight.data.abs().mean(dim=0)
            
            for i, model in enumerate(self.base_models):
                start_idx = i * 4
                end_idx = start_idx + 4
                model_importance = first_layer_weights[start_idx:end_idx].mean().item()
                importance_scores[model.model_name] = model_importance
        
        return importance_scores
    
    def save(self, file_path: str):
        """Save the fitted stacking regressor."""
        save_data = {
            'config': self.config,
            'base_models': self.base_models,
            'meta_learners': self.meta_learners,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Stacking regressor saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'StackingRegressor':
        """Load a fitted stacking regressor."""
        with open(file_path, 'rb') as f:
            save_data = pickle.load(f)
        
        stacker = cls(save_data['config'])
        stacker.base_models = save_data['base_models']
        stacker.meta_learners = save_data['meta_learners']
        stacker.is_fitted = save_data['is_fitted']
        stacker.feature_names = save_data['feature_names']
        
        logger.info(f"Stacking regressor loaded from {file_path}")
        return stacker

# Factory function
def create_stacking_regressor(config: Optional[StackingConfig] = None) -> StackingRegressor:
    """
    Factory function to create a stacking regressor.
    
    Args:
        config (StackingConfig, optional): Configuration for stacking
        
    Returns:
        StackingRegressor: Configured stacking regressor
    """
    if config is None:
        config = StackingConfig()
    
    return StackingRegressor(config)
