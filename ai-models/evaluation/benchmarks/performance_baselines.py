"""
BatteryMind - Performance Baselines

Comprehensive performance baseline system for battery health prediction and
optimization models. Provides standardized benchmarks, reference implementations,
and performance comparison frameworks for model evaluation.

Features:
- Multiple baseline model implementations
- Standardized performance metrics and evaluation protocols
- Cross-validation and statistical significance testing
- Performance regression detection
- Automated baseline updates and maintenance

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
import lightgbm as lgb

# Time series and statistical libraries
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Internal imports
from ..metrics.accuracy_metrics import AccuracyMetricsCalculator
from ..metrics.performance_metrics import PerformanceMetricsCalculator
from ..metrics.efficiency_metrics import EfficiencyMetricsCalculator
from ..metrics.business_metrics import BusinessMetricsCalculator
from ...utils.logging_utils import setup_logging
from ...utils.data_utils import TimeSeriesValidator

# Configure logging
logger = setup_logging(__name__)

@dataclass
class BaselineConfig:
    """Configuration for baseline model evaluation."""
    
    # Model selection
    models_to_evaluate: List[str] = field(default_factory=lambda: [
        'linear_regression', 'ridge_regression', 'random_forest', 
        'gradient_boosting', 'xgboost', 'svr', 'mlp', 'arima'
    ])
    
    # Evaluation settings
    cross_validation_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Performance thresholds
    minimum_r2_threshold: float = 0.7
    maximum_mae_threshold: float = 0.1
    maximum_rmse_threshold: float = 0.15
    
    # Statistical testing
    significance_level: float = 0.05
    confidence_interval: float = 0.95
    
    # Baseline update settings
    enable_automatic_updates: bool = True
    update_frequency_days: int = 30
    performance_degradation_threshold: float = 0.05

@dataclass
class BaselineResult:
    """Result from baseline model evaluation."""
    
    model_name: str
    model_type: str
    
    # Performance metrics
    mse: float
    mae: float
    rmse: float
    r2_score: float
    mape: float
    
    # Cross-validation results
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    
    # Statistical significance
    confidence_interval_lower: float
    confidence_interval_upper: float
    p_value: Optional[float] = None
    
    # Training information
    training_time_seconds: float
    prediction_time_ms: float
    model_size_mb: float
    
    # Feature importance (if available)
    feature_importance: Optional[Dict[str, float]] = None
    
    # Model parameters
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Validation information
    validation_date: datetime = field(default_factory=datetime.now)
    data_version: str = "1.0.0"

class PhysicsBasedBaseline:
    """Physics-based baseline model for battery health prediction."""
    
    def __init__(self):
        self.model_name = "physics_based"
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PhysicsBasedBaseline':
        """Fit physics-based model (no actual training needed)."""
        self.is_fitted = True
        logger.info("Physics-based baseline model fitted")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using physics-based equations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        for i, features in enumerate(X):
            # Extract relevant features (assuming standardized feature order)
            voltage = features[0] if len(features) > 0 else 3.7
            current = features[1] if len(features) > 1 else 0.0
            temperature = features[2] if len(features) > 2 else 25.0
            soc = features[3] if len(features) > 3 else 0.5
            cycle_count = features[4] if len(features) > 4 else 0
            
            # Physics-based SoH calculation
            # Voltage-based health indicator
            voltage_health = min(1.0, voltage / 4.2)
            
            # Temperature impact on degradation
            temp_factor = 1.0 - 0.001 * abs(temperature - 25.0)
            
            # Cycle-based degradation (simplified)
            cycle_degradation = max(0.0, 1.0 - cycle_count / 5000.0)
            
            # Combined health estimation
            soh = voltage_health * temp_factor * cycle_degradation
            soh = max(0.5, min(1.0, soh))  # Constrain to reasonable range
            
            predictions.append(soh)
        
        return np.array(predictions)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'model_type': 'physics_based',
            'uses_voltage_health': True,
            'uses_temperature_factor': True,
            'uses_cycle_degradation': True
        }

class HeuristicBaseline:
    """Heuristic baseline model using simple rules."""
    
    def __init__(self):
        self.model_name = "heuristic"
        self.is_fitted = False
        self.rules = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HeuristicBaseline':
        """Fit heuristic rules based on training data statistics."""
        # Calculate feature statistics for rule generation
        self.rules = {
            'voltage_mean': np.mean(X[:, 0]) if X.shape[1] > 0 else 3.7,
            'voltage_std': np.std(X[:, 0]) if X.shape[1] > 0 else 0.2,
            'current_mean': np.mean(X[:, 1]) if X.shape[1] > 1 else 0.0,
            'temperature_mean': np.mean(X[:, 2]) if X.shape[1] > 2 else 25.0,
            'soc_mean': np.mean(X[:, 3]) if X.shape[1] > 3 else 0.5,
            'target_mean': np.mean(y),
            'target_std': np.std(y)
        }
        
        self.is_fitted = True
        logger.info("Heuristic baseline model fitted")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using heuristic rules."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        for features in X:
            voltage = features[0] if len(features) > 0 else self.rules['voltage_mean']
            current = features[1] if len(features) > 1 else self.rules['current_mean']
            temperature = features[2] if len(features) > 2 else self.rules['temperature_mean']
            
            # Simple heuristic rules
            base_health = self.rules['target_mean']
            
            # Voltage-based adjustment
            voltage_deviation = abs(voltage - self.rules['voltage_mean']) / self.rules['voltage_std']
            voltage_penalty = min(0.2, voltage_deviation * 0.05)
            
            # Temperature-based adjustment
            temp_penalty = max(0.0, abs(temperature - 25.0) / 100.0)
            
            # Current-based adjustment
            current_penalty = min(0.1, abs(current) / 100.0)
            
            # Final prediction
            prediction = base_health - voltage_penalty - temp_penalty - current_penalty
            prediction = max(0.0, min(1.0, prediction))
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'model_type': 'heuristic',
            'rules': self.rules
        }

class PerformanceBaselinesEvaluator:
    """
    Comprehensive performance baseline evaluator for battery models.
    """
    
    def __init__(self, config: BaselineConfig = None):
        self.config = config or BaselineConfig()
        self.baseline_models = {}
        self.baseline_results = {}
        self.evaluation_history = []
        
        # Initialize metrics calculators
        self.accuracy_calculator = AccuracyMetricsCalculator()
        self.performance_calculator = PerformanceMetricsCalculator()
        self.efficiency_calculator = EfficiencyMetricsCalculator()
        self.business_calculator = BusinessMetricsCalculator()
        
        # Initialize baseline models
        self._initialize_baseline_models()
        
        logger.info("PerformanceBaselinesEvaluator initialized")
    
    def _initialize_baseline_models(self):
        """Initialize all baseline models."""
        # Traditional ML models
        if 'linear_regression' in self.config.models_to_evaluate:
            self.baseline_models['linear_regression'] = LinearRegression()
        
        if 'ridge_regression' in self.config.models_to_evaluate:
            self.baseline_models['ridge_regression'] = Ridge(alpha=1.0, random_state=self.config.random_state)
        
        if 'lasso_regression' in self.config.models_to_evaluate:
            self.baseline_models['lasso_regression'] = Lasso(alpha=1.0, random_state=self.config.random_state)
        
        if 'random_forest' in self.config.models_to_evaluate:
            self.baseline_models['random_forest'] = RandomForestRegressor(
                n_estimators=100, 
                random_state=self.config.random_state,
                n_jobs=-1
            )
        
        if 'gradient_boosting' in self.config.models_to_evaluate:
            self.baseline_models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=self.config.random_state
            )
        
        if 'xgboost' in self.config.models_to_evaluate:
            self.baseline_models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=self.config.random_state
            )
        
        if 'lightgbm' in self.config.models_to_evaluate:
            self.baseline_models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=self.config.random_state,
                verbose=-1
            )
        
        if 'svr' in self.config.models_to_evaluate:
            self.baseline_models['svr'] = SVR(kernel='rbf', C=1.0, gamma='scale')
        
        if 'knn' in self.config.models_to_evaluate:
            self.baseline_models['knn'] = KNeighborsRegressor(n_neighbors=5)
        
        if 'decision_tree' in self.config.models_to_evaluate:
            self.baseline_models['decision_tree'] = DecisionTreeRegressor(
                random_state=self.config.random_state,
                max_depth=10
            )
        
        if 'mlp' in self.config.models_to_evaluate:
            self.baseline_models['mlp'] = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.config.random_state
            )
        
        # Custom baseline models
        if 'physics_based' in self.config.models_to_evaluate:
            self.baseline_models['physics_based'] = PhysicsBasedBaseline()
        
        if 'heuristic' in self.config.models_to_evaluate:
            self.baseline_models['heuristic'] = HeuristicBaseline()
        
        logger.info(f"Initialized {len(self.baseline_models)} baseline models")
    
    def evaluate_baselines(self, X: np.ndarray, y: np.ndarray, 
                          feature_names: List[str] = None) -> Dict[str, BaselineResult]:
        """
        Evaluate all baseline models on the provided dataset.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: Optional feature names
            
        Returns:
            Dict mapping model names to baseline results
        """
        logger.info(f"Starting baseline evaluation on dataset with shape {X.shape}")
        
        results = {}
        
        # Prepare data
        X_scaled = self._prepare_features(X)
        
        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        # Evaluate each baseline model
        for model_name, model in self.baseline_models.items():
            try:
                logger.info(f"Evaluating baseline model: {model_name}")
                result = self._evaluate_single_model(
                    model, model_name, X_train, X_test, y_train, y_test, feature_names
                )
                results[model_name] = result
                
                # Check performance thresholds
                self._check_performance_thresholds(result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate baseline model {model_name}: {e}")
                continue
        
        # Store results
        self.baseline_results.update(results)
        
        # Update evaluation history
        self.evaluation_history.append({
            'timestamp': datetime.now(),
            'data_shape': X.shape,
            'models_evaluated': list(results.keys()),
            'best_model': self._find_best_model(results),
            'evaluation_summary': self._create_evaluation_summary(results)
        })
        
        logger.info(f"Baseline evaluation completed. {len(results)} models evaluated.")
        
        return results
    
    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Prepare features for baseline model evaluation."""
        # Handle missing values
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Scale features for models that require it
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        return X_scaled
    
    def _evaluate_single_model(self, model: Any, model_name: str,
                             X_train: np.ndarray, X_test: np.ndarray,
                             y_train: np.ndarray, y_test: np.ndarray,
                             feature_names: List[str] = None) -> BaselineResult:
        """Evaluate a single baseline model."""
        
        # Measure training time
        train_start = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - train_start).total_seconds()
        
        # Measure prediction time
        pred_start = datetime.now()
        y_pred = model.predict(X_test)
        prediction_time = (datetime.now() - pred_start).total_seconds() * 1000  # Convert to ms
        
        # Calculate performance metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Cross-validation
        cv_scores = self._perform_cross_validation(model, X_train, y_train)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Confidence interval for performance
        confidence_interval = stats.t.interval(
            self.config.confidence_interval,
            len(cv_scores) - 1,
            loc=cv_mean,
            scale=stats.sem(cv_scores)
        )
        
        # Feature importance (if available)
        feature_importance = self._extract_feature_importance(model, feature_names)
        
        # Model size estimation
        model_size_mb = self._estimate_model_size(model)
        
        # Model parameters
        model_parameters = self._extract_model_parameters(model)
        
        return BaselineResult(
            model_name=model_name,
            model_type=type(model).__name__,
            mse=mse,
            mae=mae,
            rmse=rmse,
            r2_score=r2,
            mape=mape,
            cv_scores=cv_scores.tolist(),
            cv_mean=cv_mean,
            cv_std=cv_std,
            confidence_interval_lower=confidence_interval[0],
            confidence_interval_upper=confidence_interval[1],
            training_time_seconds=training_time,
            prediction_time_ms=prediction_time,
            model_size_mb=model_size_mb,
            feature_importance=feature_importance,
            model_parameters=model_parameters
        )
    
    def _perform_cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Perform cross-validation for the model."""
        # Use TimeSeriesSplit for time series data
        cv_splitter = TimeSeriesSplit(n_splits=self.config.cross_validation_folds)
        
        try:
            cv_scores = cross_val_score(
                model, X, y, 
                cv=cv_splitter, 
                scoring='r2',
                n_jobs=-1
            )
        except Exception as e:
            logger.warning(f"Cross-validation failed, using simple split: {e}")
            # Fallback to simple train-test split
            from sklearn.model_selection import train_test_split
            X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(
                X, y, test_size=0.2, random_state=self.config.random_state
            )
            
            # Clone and fit model
            from sklearn.base import clone
            model_cv = clone(model)
            model_cv.fit(X_train_cv, y_train_cv)
            y_pred_cv = model_cv.predict(X_val_cv)
            cv_scores = np.array([r2_score(y_val_cv, y_pred_cv)])
        
        return cv_scores
    
    def _extract_feature_importance(self, model: Any, feature_names: List[str] = None) -> Optional[Dict[str, float]]:
        """Extract feature importance from model if available."""
        importance_dict = None
        
        # Check for feature_importances_ attribute (tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if feature_names and len(feature_names) == len(importances):
                importance_dict = dict(zip(feature_names, importances.tolist()))
            else:
                importance_dict = {f'feature_{i}': imp for i, imp in enumerate(importances)}
        
        # Check for coef_ attribute (linear models)
        elif hasattr(model, 'coef_'):
            coefficients = np.abs(model.coef_)
            if feature_names and len(feature_names) == len(coefficients):
                importance_dict = dict(zip(feature_names, coefficients.tolist()))
            else:
                importance_dict = {f'feature_{i}': coef for i, coef in enumerate(coefficients)}
        
        return importance_dict
    
    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in MB."""
        try:
            # Serialize model to estimate size
            import pickle
            model_bytes = pickle.dumps(model)
            size_mb = len(model_bytes) / (1024 * 1024)
            return round(size_mb, 2)
        except:
            return 0.0
    
    def _extract_model_parameters(self, model: Any) -> Dict[str, Any]:
        """Extract model parameters."""
        try:
            if hasattr(model, 'get_params'):
                return model.get_params()
            else:
                return {'model_type': type(model).__name__}
        except:
            return {}
    
    def _check_performance_thresholds(self, result: BaselineResult):
        """Check if model meets performance thresholds."""
        warnings = []
        
        if result.r2_score < self.config.minimum_r2_threshold:
            warnings.append(f"R² score {result.r2_score:.3f} below threshold {self.config.minimum_r2_threshold}")
        
        if result.mae > self.config.maximum_mae_threshold:
            warnings.append(f"MAE {result.mae:.3f} above threshold {self.config.maximum_mae_threshold}")
        
        if result.rmse > self.config.maximum_rmse_threshold:
            warnings.append(f"RMSE {result.rmse:.3f} above threshold {self.config.maximum_rmse_threshold}")
        
        if warnings:
            logger.warning(f"Performance warnings for {result.model_name}: {warnings}")
    
    def _find_best_model(self, results: Dict[str, BaselineResult]) -> str:
        """Find the best performing baseline model."""
        if not results:
            return ""
        
        # Rank by R² score
        best_model = max(results.keys(), key=lambda k: results[k].r2_score)
        return best_model
    
    def _create_evaluation_summary(self, results: Dict[str, BaselineResult]) -> Dict[str, Any]:
        """Create summary of evaluation results."""
        if not results:
            return {}
        
        r2_scores = [r.r2_score for r in results.values()]
        mae_scores = [r.mae for r in results.values()]
        rmse_scores = [r.rmse for r in results.values()]
        
        return {
            'num_models': len(results),
            'best_r2': max(r2_scores),
            'worst_r2': min(r2_scores),
            'mean_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores),
            'best_mae': min(mae_scores),
            'worst_mae': max(mae_scores),
            'mean_mae': np.mean(mae_scores),
            'best_rmse': min(rmse_scores),
            'worst_rmse': max(rmse_scores),
            'mean_rmse': np.mean(rmse_scores)
        }
    
    def compare_model_to_baselines(self, model_results: Dict[str, float],
                                 model_name: str = "test_model") -> Dict[str, Any]:
        """
        Compare a model's performance to baseline models.
        
        Args:
            model_results: Dictionary with model performance metrics
            model_name: Name of the model being compared
            
        Returns:
            Comparison results
        """
        if not self.baseline_results:
            logger.warning("No baseline results available for comparison")
            return {}
        
        comparison = {
            'model_name': model_name,
            'model_performance': model_results,
            'baseline_comparisons': {},
            'ranking': {},
            'statistical_tests': {}
        }
        
        # Compare against each baseline
        for baseline_name, baseline_result in self.baseline_results.items():
            model_r2 = model_results.get('r2_score', 0.0)
            baseline_r2 = baseline_result.r2_score
            
            comparison['baseline_comparisons'][baseline_name] = {
                'r2_improvement': model_r2 - baseline_r2,
                'r2_improvement_percent': ((model_r2 - baseline_r2) / baseline_r2) * 100 if baseline_r2 > 0 else 0,
                'is_better': model_r2 > baseline_r2,
                'baseline_r2': baseline_r2,
                'model_r2': model_r2
            }
        
        # Calculate ranking
        all_r2_scores = [(name, result.r2_score) for name, result in self.baseline_results.items()]
        all_r2_scores.append((model_name, model_results.get('r2_score', 0.0)))
        all_r2_scores.sort(key=lambda x: x[1], reverse=True)
        
        model_rank = next(i for i, (name, _) in enumerate(all_r2_scores, 1) if name == model_name)
        comparison['ranking'] = {
            'rank': model_rank,
            'total_models': len(all_r2_scores),
            'percentile': (len(all_r2_scores) - model_rank + 1) / len(all_r2_scores) * 100,
            'ranked_models': all_r2_scores
        }
        
        # Statistical significance testing (simplified)
        model_r2 = model_results.get('r2_score', 0.0)
        baseline_r2_values = [result.r2_score for result in self.baseline_results.values()]
        
        if baseline_r2_values:
            t_stat, p_value = stats.ttest_1samp(baseline_r2_values, model_r2)
            comparison['statistical_tests'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significantly_better': p_value < self.config.significance_level and model_r2 > np.mean(baseline_r2_values),
                'confidence_level': self.config.confidence_interval
            }
        
        return comparison
    
    def generate_baseline_report(self) -> Dict[str, Any]:
        """Generate comprehensive baseline evaluation report."""
        if not self.baseline_results:
            return {'error': 'No baseline results available'}
        
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'summary': {
                'total_models_evaluated': len(self.baseline_results),
                'best_performing_model': self._find_best_model(self.baseline_results),
                'evaluation_summary': self._create_evaluation_summary(self.baseline_results)
            },
            'detailed_results': {},
            'performance_rankings': [],
            'recommendations': []
        }
        
        # Detailed results for each model
        for model_name, result in self.baseline_results.items():
            report['detailed_results'][model_name] = {
                'performance_metrics': {
                    'r2_score': result.r2_score,
                    'mse': result.mse,
                    'mae': result.mae,
                    'rmse': result.rmse,
                    'mape': result.mape
                },
                'cross_validation': {
                    'cv_mean': result.cv_mean,
                    'cv_std': result.cv_std,
                    'confidence_interval': [
                        result.confidence_interval_lower,
                        result.confidence_interval_upper
                    ]
                },
                'efficiency_metrics': {
                    'training_time_seconds': result.training_time_seconds,
                    'prediction_time_ms': result.prediction_time_ms,
                    'model_size_mb': result.model_size_mb
                },
                'feature_importance': result.feature_importance
            }
        
        # Performance rankings
        ranked_models = sorted(
            self.baseline_results.items(),
            key=lambda x: x[1].r2_score,
            reverse=True
        )
        
        for rank, (model_name, result) in enumerate(ranked_models, 1):
            report['performance_rankings'].append({
                'rank': rank,
                'model_name': model_name,
                'r2_score': result.r2_score,
                'mae': result.mae,
                'training_time_seconds': result.training_time_seconds
            })
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on baseline evaluation results."""
        recommendations = []
        
        if not self.baseline_results:
            return ["No baseline results available for recommendations"]
        
        # Find best models
        best_model = self._find_best_model(self.baseline_results)
        best_result = self.baseline_results[best_model]
        
        # Performance recommendations
        if best_result.r2_score < 0.8:
            recommendations.append(
                "Consider more sophisticated models or feature engineering - "
                f"best baseline R² is only {best_result.r2_score:.3f}"
            )
        
        # Efficiency recommendations
        fast_models = [
            name for name, result in self.baseline_results.items()
            if result.training_time_seconds < 10.0
        ]
        
        if fast_models:
            recommendations.append(
                f"For fast training, consider: {', '.join(fast_models[:3])}"
            )
        
        # Complexity recommendations
        simple_models = ['linear_regression', 'ridge_regression', 'heuristic']
        complex_models = ['xgboost', 'lightgbm', 'random_forest']
        
        simple_performance = np.mean([
            self.baseline_results[name].r2_score 
            for name in simple_models 
            if name in self.baseline_results
        ])
        
        complex_performance = np.mean([
            self.baseline_results[name].r2_score 
            for name in complex_models 
            if name in self.baseline_results
        ])
        
        if complex_performance - simple_performance < 0.05:
            recommendations.append(
                "Complex models show minimal improvement over simple models - "
                "consider interpretability vs. performance trade-offs"
            )
        
        return recommendations
    
    def save_baseline_results(self, filepath: str):
        """Save baseline results to file."""
        results_data = {
            'baseline_results': {
                name: {
                    'model_name': result.model_name,
                    'model_type': result.model_type,
                    'mse': result.mse,
                    'mae': result.mae,
                    'rmse': result.rmse,
                    'r2_score': result.r2_score,
                    'mape': result.mape,
                    'cv_scores': result.cv_scores,
                    'cv_mean': result.cv_mean,
                    'cv_std': result.cv_std,
                    'training_time_seconds': result.training_time_seconds,
                    'prediction_time_ms': result.prediction_time_ms,
                    'model_size_mb': result.model_size_mb,
                    'feature_importance': result.feature_importance,
                    'validation_date': result.validation_date.isoformat()
                }
                for name, result in self.baseline_results.items()
            },
            'evaluation_history': self.evaluation_history,
            'config': self.config.__dict__
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Baseline results saved to {filepath}")
    
    def load_baseline_results(self, filepath: str):
        """Load baseline results from file."""
        with open(filepath, 'r') as f:
            results_data = json.load(f)
        
        # Reconstruct baseline results
        for name, result_dict in results_data['baseline_results'].items():
            self.baseline_results[name] = BaselineResult(
                model_name=result_dict['model_name'],
                model_type=result_dict['model_type'],
                mse=result_dict['mse'],
                mae=result_dict['mae'],
                rmse=result_dict['rmse'],
                r2_score=result_dict['r2_score'],
                mape=result_dict['mape'],
                cv_scores=result_dict['cv_scores'],
                cv_mean=result_dict['cv_mean'],
                cv_std=result_dict['cv_std'],
                confidence_interval_lower=result_dict.get('confidence_interval_lower', 0.0),
                confidence_interval_upper=result_dict.get('confidence_interval_upper', 1.0),
                training_time_seconds=result_dict['training_time_seconds'],
                prediction_time_ms=result_dict['prediction_time_ms'],
                model_size_mb=result_dict['model_size_mb'],
                feature_importance=result_dict.get('feature_importance'),
                validation_date=datetime.fromisoformat(result_dict['validation_date'])
            )
        
        self.evaluation_history = results_data.get('evaluation_history', [])
        
        logger.info(f"Baseline results loaded from {filepath}")

# Factory function
def create_baseline_evaluator(config: BaselineConfig = None) -> PerformanceBaselinesEvaluator:
    """Create and initialize baseline evaluator."""
    return PerformanceBaselinesEvaluator(config)
