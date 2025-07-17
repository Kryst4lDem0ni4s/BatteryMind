"""
BatteryMind - Baseline Models for Benchmarking

Comprehensive baseline model implementations for battery health prediction,
degradation forecasting, and optimization benchmarking. Provides industry-standard
baseline models including statistical methods, classical ML algorithms, and
physics-based models for comparative evaluation.

Features:
- Classical ML baseline models (Linear Regression, Random Forest, SVM)
- Statistical time series models (ARIMA, ETS, Prophet)
- Physics-based analytical models for battery degradation
- Heuristic optimization algorithms for charging strategies
- Performance benchmarking and comparison utilities
- Automated baseline model evaluation pipelines

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import time
import warnings
from abc import ABC, abstractmethod
import pickle
import joblib

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Time series analysis imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("Statsmodels not available - some time series models will be disabled")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available - Prophet models will be disabled")

# BatteryMind imports
from ...utils.logging_utils import get_logger
from ...utils.data_utils import DataProcessor, TimeSeriesProcessor
from ...training_data.preprocessing_scripts.feature_extractor import BatteryFeatureExtractor
from ...training_data.generators.physics_simulator import BatteryPhysicsSimulator

# Configure logging
logger = get_logger(__name__)

@dataclass
class BaselineModelConfig:
    """
    Configuration for baseline model evaluation.
    
    Attributes:
        model_types (List[str]): Types of baseline models to evaluate
        cross_validation_folds (int): Number of CV folds
        test_size (float): Size of test set
        time_series_cv (bool): Use time series cross-validation
        random_state (int): Random seed for reproducibility
        performance_metrics (List[str]): Metrics to evaluate
        save_models (bool): Whether to save trained models
        output_dir (str): Directory to save results
    """
    model_types: List[str] = field(default_factory=lambda: [
        'linear_regression', 'random_forest', 'gradient_boosting',
        'svr', 'neural_network', 'arima', 'prophet', 'physics_based'
    ])
    cross_validation_folds: int = 5
    test_size: float = 0.2
    time_series_cv: bool = True
    random_state: int = 42
    performance_metrics: List[str] = field(default_factory=lambda: [
        'mse', 'mae', 'rmse', 'r2', 'mape'
    ])
    save_models: bool = True
    output_dir: str = "./baseline_results"

@dataclass
class BaselineResult:
    """
    Result from baseline model evaluation.
    
    Attributes:
        model_name (str): Name of the baseline model
        model_type (str): Type of model (ml, statistical, physics)
        performance_metrics (Dict[str, float]): Performance metrics
        training_time (float): Training time in seconds
        prediction_time (float): Average prediction time
        model_parameters (Dict[str, Any]): Model hyperparameters
        cross_validation_scores (List[float]): CV scores
        feature_importance (Dict[str, float]): Feature importance if available
    """
    model_name: str
    model_type: str
    performance_metrics: Dict[str, float]
    training_time: float
    prediction_time: float
    model_parameters: Dict[str, Any]
    cross_validation_scores: List[float] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)

class BaselineModel(ABC):
    """Abstract base class for baseline models."""
    
    def __init__(self, model_name: str, random_state: int = 42):
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and parameters."""
        pass
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available."""
        return {}

class LinearRegressionBaseline(BaselineModel):
    """Linear Regression baseline model."""
    
    def __init__(self, random_state: int = 42):
        super().__init__("Linear Regression", random_state)
        self.model = LinearRegression()
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit linear regression model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'linear_regression',
            'parameters': {},
            'features_used': getattr(self.model, 'n_features_in_', 0)
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature coefficients as importance."""
        if not self.is_fitted:
            return {}
        
        coefficients = np.abs(self.model.coef_)
        feature_names = [f'feature_{i}' for i in range(len(coefficients))]
        return dict(zip(feature_names, coefficients))

class RandomForestBaseline(BaselineModel):
    """Random Forest baseline model."""
    
    def __init__(self, random_state: int = 42):
        super().__init__("Random Forest", random_state)
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit random forest model."""
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'random_forest',
            'parameters': {
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'random_state': self.model.random_state
            },
            'features_used': getattr(self.model, 'n_features_in_', 0)
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_fitted:
            return {}
        
        importances = self.model.feature_importances_
        feature_names = [f'feature_{i}' for i in range(len(importances))]
        return dict(zip(feature_names, importances))

class GradientBoostingBaseline(BaselineModel):
    """Gradient Boosting baseline model."""
    
    def __init__(self, random_state: int = 42):
        super().__init__("Gradient Boosting", random_state)
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit gradient boosting model."""
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'gradient_boosting',
            'parameters': {
                'n_estimators': self.model.n_estimators,
                'learning_rate': self.model.learning_rate,
                'max_depth': self.model.max_depth
            },
            'features_used': getattr(self.model, 'n_features_in_', 0)
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_fitted:
            return {}
        
        importances = self.model.feature_importances_
        feature_names = [f'feature_{i}' for i in range(len(importances))]
        return dict(zip(feature_names, importances))

class SVRBaseline(BaselineModel):
    """Support Vector Regression baseline model."""
    
    def __init__(self, random_state: int = 42):
        super().__init__("Support Vector Regression", random_state)
        self.model = SVR(kernel='rbf', C=1.0, gamma='scale')
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit SVR model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'svr',
            'parameters': {
                'kernel': self.model.kernel,
                'C': self.model.C,
                'gamma': self.model.gamma
            },
            'features_used': getattr(self.model, 'n_support_', [0])[0] if hasattr(self.model, 'n_support_') else 0
        }

class NeuralNetworkBaseline(BaselineModel):
    """Neural Network baseline model."""
    
    def __init__(self, random_state: int = 42):
        super().__init__("Neural Network", random_state)
        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit neural network model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'neural_network',
            'parameters': {
                'hidden_layer_sizes': self.model.hidden_layer_sizes,
                'max_iter': self.model.max_iter,
                'n_iter_': getattr(self.model, 'n_iter_', 0)
            },
            'features_used': getattr(self.model, 'n_features_in_', 0)
        }

class ARIMABaseline(BaselineModel):
    """ARIMA time series baseline model."""
    
    def __init__(self, random_state: int = 42):
        super().__init__("ARIMA", random_state)
        self.order = (1, 1, 1)  # Default ARIMA order
        self.model = None
        self.fitted_model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit ARIMA model."""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels required for ARIMA model")
        
        # ARIMA uses only the target variable
        time_series = pd.Series(y)
        
        try:
            self.model = ARIMA(time_series, order=self.order)
            self.fitted_model = self.model.fit()
            self.is_fitted = True
        except Exception as e:
            logger.warning(f"ARIMA fitting failed: {e}")
            # Fallback to simpler model
            self.order = (1, 0, 0)
            self.model = ARIMA(time_series, order=self.order)
            self.fitted_model = self.model.fit()
            self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        n_periods = len(X)
        forecast = self.fitted_model.forecast(steps=n_periods)
        return np.array(forecast)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'arima',
            'parameters': {
                'order': self.order,
                'aic': getattr(self.fitted_model, 'aic', None) if self.fitted_model else None
            },
            'features_used': 1  # Time series models use time dimension
        }

class ProphetBaseline(BaselineModel):
    """Prophet time series baseline model."""
    
    def __init__(self, random_state: int = 42):
        super().__init__("Prophet", random_state)
        self.model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Prophet model."""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet required for Prophet model")
        
        # Create Prophet dataframe
        dates = pd.date_range(start='2020-01-01', periods=len(y), freq='D')
        df = pd.DataFrame({
            'ds': dates,
            'y': y
        })
        
        self.model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(df)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        n_periods = len(X)
        future = self.model.make_future_dataframe(periods=n_periods)
        forecast = self.model.predict(future)
        
        # Return only the forecasted periods
        return forecast['yhat'].tail(n_periods).values
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'prophet',
            'parameters': {
                'changepoint_prior_scale': getattr(self.model, 'changepoint_prior_scale', None),
                'seasonality_prior_scale': getattr(self.model, 'seasonality_prior_scale', None)
            },
            'features_used': 1  # Time series models use time dimension
        }

class PhysicsBasedBaseline(BaselineModel):
    """Physics-based analytical baseline model."""
    
    def __init__(self, random_state: int = 42):
        super().__init__("Physics-Based", random_state)
        self.physics_simulator = BatteryPhysicsSimulator()
        self.baseline_params = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit physics-based model by calibrating parameters."""
        # Extract relevant features for physics model
        # Assuming X contains [voltage, current, temperature, soc, ...]
        
        # Simple parameter calibration using least squares
        from scipy.optimize import minimize
        
        def objective(params):
            predictions = self._physics_predict(X, params)
            return np.mean((predictions - y) ** 2)
        
        # Initial parameter guess
        initial_params = {
            'capacity_fade_rate': 0.001,
            'resistance_increase_rate': 0.0001,
            'temperature_factor': 0.01
        }
        
        # Optimize parameters
        try:
            result = minimize(
                objective,
                list(initial_params.values()),
                method='Nelder-Mead',
                options={'maxiter': 100}
            )
            
            self.baseline_params = {
                'capacity_fade_rate': result.x[0],
                'resistance_increase_rate': result.x[1],
                'temperature_factor': result.x[2]
            }
        except Exception as e:
            logger.warning(f"Physics model calibration failed: {e}")
            self.baseline_params = initial_params
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using physics model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self._physics_predict(X, self.baseline_params)
    
    def _physics_predict(self, X: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Physics-based prediction function."""
        predictions = []
        
        for i, sample in enumerate(X):
            # Extract features (assuming standard order)
            voltage = sample[0] if len(sample) > 0 else 3.7
            current = sample[1] if len(sample) > 1 else 0.0
            temperature = sample[2] if len(sample) > 2 else 25.0
            soc = sample[3] if len(sample) > 3 else 0.5
            
            # Simple physics-based SOH calculation
            # Capacity fade due to cycling
            cycle_stress = abs(current) / 100.0  # Normalized current stress
            capacity_fade = params['capacity_fade_rate'] * cycle_stress
            
            # Temperature effects
            temp_stress = abs(temperature - 25.0) / 50.0
            temp_fade = params['temperature_factor'] * temp_stress
            
            # SOH prediction (simplified)
            base_soh = 1.0 - (capacity_fade + temp_fade)
            predicted_soh = max(0.5, min(1.0, base_soh))  # Clamp to reasonable range
            
            predictions.append(predicted_soh)
        
        return np.array(predictions)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'physics_based',
            'parameters': self.baseline_params,
            'features_used': 4  # voltage, current, temperature, soc
        }

class BaselineModelFactory:
    """Factory for creating baseline models."""
    
    @staticmethod
    def create_model(model_type: str, random_state: int = 42) -> BaselineModel:
        """Create baseline model by type."""
        model_map = {
            'linear_regression': LinearRegressionBaseline,
            'random_forest': RandomForestBaseline,
            'gradient_boosting': GradientBoostingBaseline,
            'svr': SVRBaseline,
            'neural_network': NeuralNetworkBaseline,
            'arima': ARIMABaseline,
            'prophet': ProphetBaseline,
            'physics_based': PhysicsBasedBaseline
        }
        
        if model_type not in model_map:
            available_types = list(model_map.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available_types}")
        
        return model_map[model_type](random_state)

class BaselineEvaluator:
    """Comprehensive baseline model evaluator."""
    
    def __init__(self, config: BaselineModelConfig = None):
        self.config = config or BaselineModelConfig()
        self.results = {}
        self.models = {}
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("BaselineEvaluator initialized")
    
    def evaluate_all_baselines(self, X: np.ndarray, y: np.ndarray, 
                             feature_names: List[str] = None) -> Dict[str, BaselineResult]:
        """
        Evaluate all baseline models.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: Feature names
            
        Returns:
            Dictionary of baseline results
        """
        logger.info(f"Evaluating {len(self.config.model_types)} baseline models")
        
        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=self.config.random_state,
            shuffle=not self.config.time_series_cv
        )
        
        results = {}
        
        for model_type in self.config.model_types:
            try:
                logger.info(f"Evaluating {model_type} baseline")
                result = self._evaluate_single_model(
                    model_type, X_train, X_test, y_train, y_test, feature_names
                )
                results[model_type] = result
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_type}: {e}")
                continue
        
        self.results = results
        
        # Save results
        self._save_results()
        
        return results
    
    def _evaluate_single_model(self, model_type: str, X_train: np.ndarray,
                             X_test: np.ndarray, y_train: np.ndarray,
                             y_test: np.ndarray, feature_names: List[str] = None) -> BaselineResult:
        """Evaluate a single baseline model."""
        
        # Create model
        model = BaselineModelFactory.create_model(model_type, self.config.random_state)
        
        # Training
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Prediction
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = (time.time() - start_time) / len(X_test)  # Per sample
        
        # Performance metrics
        performance_metrics = self._calculate_metrics(y_test, y_pred)
        
        # Cross-validation
        cv_scores = self._cross_validate_model(model_type, X_train, y_train)
        
        # Feature importance
        feature_importance = model.get_feature_importance()
        if feature_names and len(feature_importance) == len(feature_names):
            feature_importance = dict(zip(feature_names, feature_importance.values()))
        
        # Store model
        if self.config.save_models:
            self.models[model_type] = model
        
        return BaselineResult(
            model_name=model.model_name,
            model_type=model_type,
            performance_metrics=performance_metrics,
            training_time=training_time,
            prediction_time=prediction_time,
            model_parameters=model.get_model_info(),
            cross_validation_scores=cv_scores,
            feature_importance=feature_importance
        )
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}
        
        if 'mse' in self.config.performance_metrics:
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        
        if 'mae' in self.config.performance_metrics:
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        
        if 'rmse' in self.config.performance_metrics:
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        
        if 'r2' in self.config.performance_metrics:
            metrics['r2'] = float(r2_score(y_true, y_pred))
        
        if 'mape' in self.config.performance_metrics:
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            metrics['mape'] = float(mape)
        
        return metrics
    
    def _cross_validate_model(self, model_type: str, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Perform cross-validation."""
        try:
            model = BaselineModelFactory.create_model(model_type, self.config.random_state)
            
            if self.config.time_series_cv:
                cv = TimeSeriesSplit(n_splits=self.config.cross_validation_folds)
            else:
                cv = self.config.cross_validation_folds
            
            # For time series models, we need different CV approach
            if model_type in ['arima', 'prophet']:
                # Simple train-test splits for time series
                scores = []
                split_size = len(X) // self.config.cross_validation_folds
                
                for i in range(self.config.cross_validation_folds):
                    start_idx = i * split_size
                    end_idx = start_idx + split_size * 2  # Train on double the data
                    
                    if end_idx > len(X):
                        break
                    
                    X_fold_train = X[start_idx:start_idx + split_size]
                    y_fold_train = y[start_idx:start_idx + split_size]
                    X_fold_test = X[start_idx + split_size:end_idx]
                    y_fold_test = y[start_idx + split_size:end_idx]
                    
                    fold_model = BaselineModelFactory.create_model(model_type, self.config.random_state)
                    fold_model.fit(X_fold_train, y_fold_train)
                    y_pred = fold_model.predict(X_fold_test)
                    
                    score = r2_score(y_fold_test, y_pred)
                    scores.append(score)
                
                return scores
            else:
                # Standard cross-validation for ML models
                scores = cross_val_score(
                    model.model, X, y, cv=cv, scoring='r2', n_jobs=-1
                )
                return scores.tolist()
                
        except Exception as e:
            logger.warning(f"Cross-validation failed for {model_type}: {e}")
            return []
    
    def _save_results(self):
        """Save evaluation results."""
        # Save detailed results
        results_file = Path(self.config.output_dir) / "baseline_results.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for model_type, result in self.results.items():
            serializable_results[model_type] = {
                'model_name': result.model_name,
                'model_type': result.model_type,
                'performance_metrics': result.performance_metrics,
                'training_time': result.training_time,
                'prediction_time': result.prediction_time,
                'model_parameters': result.model_parameters,
                'cross_validation_scores': result.cross_validation_scores,
                'feature_importance': result.feature_importance
            }
        
        import json
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save models
        if self.config.save_models:
            models_file = Path(self.config.output_dir) / "baseline_models.pkl"
            with open(models_file, 'wb') as f:
                pickle.dump(self.models, f)
        
        logger.info(f"Results saved to {self.config.output_dir}")
    
    def get_best_baseline(self, metric: str = 'r2') -> Tuple[str, BaselineResult]:
        """Get the best performing baseline model."""
        if not self.results:
            raise ValueError("No results available. Run evaluation first.")
        
        best_model = None
        best_score = float('-inf') if metric != 'mse' and metric != 'mae' else float('inf')
        
        for model_type, result in self.results.items():
            if metric in result.performance_metrics:
                score = result.performance_metrics[metric]
                
                if metric in ['mse', 'mae']:  # Lower is better
                    if score < best_score:
                        best_score = score
                        best_model = model_type
                else:  # Higher is better
                    if score > best_score:
                        best_score = score
                        best_model = model_type
        
        if best_model is None:
            raise ValueError(f"No results found for metric: {metric}")
        
        return best_model, self.results[best_model]
    
    def compare_models(self, metric: str = 'r2') -> pd.DataFrame:
        """Compare all baseline models."""
        if not self.results:
            raise ValueError("No results available. Run evaluation first.")
        
        comparison_data = []
        for model_type, result in self.results.items():
            row = {
                'Model': result.model_name,
                'Type': result.model_type,
                'Training Time (s)': result.training_time,
                'Prediction Time (s)': result.prediction_time
            }
            
            # Add performance metrics
            for metric_name, metric_value in result.performance_metrics.items():
                row[metric_name.upper()] = metric_value
            
            # Add CV statistics
            if result.cross_validation_scores:
                row['CV Mean'] = np.mean(result.cross_validation_scores)
                row['CV Std'] = np.std(result.cross_validation_scores)
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by specified metric
        if metric.upper() in df.columns:
            ascending = metric.lower() in ['mse', 'mae']  # Lower is better
            df = df.sort_values(metric.upper(), ascending=ascending)
        
        return df
    
    def plot_results(self, metric: str = 'r2'):
        """Plot baseline comparison results."""
        try:
            import matplotlib.pyplot as plt
            
            df = self.compare_models(metric)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Performance comparison
            ax1.bar(df['Model'], df[metric.upper()])
            ax1.set_title(f'Baseline Model Comparison - {metric.upper()}')
            ax1.set_ylabel(metric.upper())
            ax1.tick_params(axis='x', rotation=45)
            
            # Training time comparison
            ax2.bar(df['Model'], df['Training Time (s)'])
            ax2.set_title('Training Time Comparison')
            ax2.set_ylabel('Training Time (seconds)')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(Path(self.config.output_dir) / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")

# Factory function
def create_baseline_evaluator(config: BaselineModelConfig = None) -> BaselineEvaluator:
    """Create baseline evaluator with configuration."""
    return BaselineEvaluator(config)

def run_baseline_evaluation(X: np.ndarray, y: np.ndarray, 
                          feature_names: List[str] = None,
                          config: BaselineModelConfig = None) -> Dict[str, BaselineResult]:
    """
    Convenience function to run complete baseline evaluation.
    
    Args:
        X: Feature matrix
        y: Target values
        feature_names: Feature names
        config: Evaluation configuration
        
    Returns:
        Baseline evaluation results
    """
    evaluator = create_baseline_evaluator(config)
    return evaluator.evaluate_all_baselines(X, y, feature_names)
