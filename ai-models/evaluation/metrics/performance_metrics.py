"""
BatteryMind - Performance Metrics Module

Comprehensive performance evaluation metrics for battery management AI models.
This module provides detailed performance analysis including accuracy, speed,
resource utilization, and model-specific battery metrics.

Features:
- Model evaluation metrics for all AI architectures
- Real-time performance monitoring
- Comparative analysis across different models
- Battery-specific performance indicators
- Statistical significance testing
- Performance degradation tracking

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
import psutil
import logging
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceResult:
    """Data class for storing performance evaluation results."""
    model_name: str
    metrics: Dict[str, float]
    inference_time_ms: float
    memory_usage_mb: float
    throughput_samples_per_sec: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_significance: Dict[str, float]
    timestamp: str
    metadata: Dict[str, Any]

class BasePerformanceMetrics(ABC):
    """Abstract base class for performance metrics."""
    
    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        """Evaluate performance metrics."""
        pass
    
    @abstractmethod
    def get_metric_names(self) -> List[str]:
        """Get list of metric names."""
        pass

class RegressionMetrics(BasePerformanceMetrics):
    """Performance metrics for regression tasks."""
    
    def __init__(self, include_confidence_intervals: bool = True):
        self.include_confidence_intervals = include_confidence_intervals
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Evaluate regression performance metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of performance metrics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        # Basic regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = self._mean_absolute_percentage_error(y_true, y_pred)
        smape = self._symmetric_mean_absolute_percentage_error(y_true, y_pred)
        
        # Residual analysis
        residuals = y_true - y_pred
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)
        
        # Statistical tests
        normality_p_value = self._test_residual_normality(residuals)
        autocorr = self._calculate_autocorrelation(residuals)
        
        # Prediction intervals
        prediction_interval_coverage = self._calculate_prediction_interval_coverage(
            y_true, y_pred, residual_std
        )
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'smape': smape,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'normality_p_value': normality_p_value,
            'autocorrelation': autocorr,
            'prediction_interval_coverage': prediction_interval_coverage,
            'explained_variance': self._explained_variance_score(y_true, y_pred),
            'median_absolute_error': np.median(np.abs(residuals)),
            'max_error': np.max(np.abs(residuals))
        }
        
        return metrics
    
    def get_metric_names(self) -> List[str]:
        """Get list of regression metric names."""
        return [
            'mse', 'rmse', 'mae', 'r2_score', 'mape', 'smape',
            'residual_mean', 'residual_std', 'normality_p_value',
            'autocorrelation', 'prediction_interval_coverage',
            'explained_variance', 'median_absolute_error', 'max_error'
        ]
    
    def _mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _symmetric_mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if not np.any(mask):
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / denominator[mask])) * 100
    
    def _explained_variance_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate explained variance score."""
        numerator = np.var(y_true - y_pred)
        denominator = np.var(y_true)
        return 1 - (numerator / denominator) if denominator != 0 else 0.0
    
    def _test_residual_normality(self, residuals: np.ndarray) -> float:
        """Test residual normality using Shapiro-Wilk test."""
        try:
            if len(residuals) > 5000:
                # Use Kolmogorov-Smirnov test for large samples
                from scipy.stats import kstest
                _, p_value = kstest(residuals, 'norm')
            else:
                from scipy.stats import shapiro
                _, p_value = shapiro(residuals)
            return p_value
        except:
            return np.nan
    
    def _calculate_autocorrelation(self, residuals: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation of residuals."""
        if len(residuals) <= lag:
            return 0.0
        
        residuals_shifted = residuals[:-lag]
        residuals_lagged = residuals[lag:]
        
        correlation = np.corrcoef(residuals_shifted, residuals_lagged)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _calculate_prediction_interval_coverage(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                              residual_std: float, confidence: float = 0.95) -> float:
        """Calculate prediction interval coverage."""
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * residual_std
        
        lower_bound = y_pred - margin
        upper_bound = y_pred + margin
        
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        return coverage

class ClassificationMetrics(BasePerformanceMetrics):
    """Performance metrics for classification tasks."""
    
    def __init__(self, average: str = 'weighted', include_per_class: bool = False):
        self.average = average
        self.include_per_class = include_per_class
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_pred_proba: Optional[np.ndarray] = None, **kwargs) -> Dict[str, float]:
        """
        Evaluate classification performance metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of performance metrics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        # Basic classification metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            balanced_accuracy_score, matthews_corrcoef
        )
        
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=self.average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=self.average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=self.average, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'matthews_corrcoef': mcc
        }
        
        # Add AUC-ROC and AUC-PR if probabilities are provided
        if y_pred_proba is not None:
            try:
                if y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 2:
                    # Binary classification
                    if y_pred_proba.ndim == 2:
                        y_pred_proba = y_pred_proba[:, 1]
                    
                    auc_roc = roc_auc_score(y_true, y_pred_proba)
                    auc_pr = average_precision_score(y_true, y_pred_proba)
                    
                    metrics.update({
                        'auc_roc': auc_roc,
                        'auc_pr': auc_pr
                    })
                else:
                    # Multi-class classification
                    auc_roc = roc_auc_score(y_true, y_pred_proba, 
                                          average=self.average, multi_class='ovr')
                    metrics['auc_roc'] = auc_roc
                    
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")
        
        # Confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            # Binary classification specific metrics
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            metrics.update({
                'specificity': specificity,
                'sensitivity': sensitivity,
                'true_negative_rate': specificity,
                'true_positive_rate': sensitivity,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0.0
            })
        
        return metrics
    
    def get_metric_names(self) -> List[str]:
        """Get list of classification metric names."""
        base_metrics = [
            'accuracy', 'balanced_accuracy', 'precision', 'recall', 
            'f1_score', 'matthews_corrcoef'
        ]
        
        additional_metrics = [
            'auc_roc', 'auc_pr', 'specificity', 'sensitivity',
            'true_negative_rate', 'true_positive_rate',
            'false_positive_rate', 'false_negative_rate'
        ]
        
        return base_metrics + additional_metrics

class BatterySpecificMetrics(BasePerformanceMetrics):
    """Performance metrics specific to battery management tasks."""
    
    def __init__(self, tolerance_levels: Dict[str, float] = None):
        self.tolerance_levels = tolerance_levels or {
            'soh': 0.05,  # 5% tolerance for State of Health
            'rul': 0.1,   # 10% tolerance for Remaining Useful Life
            'soc': 0.02,  # 2% tolerance for State of Charge
            'temperature': 2.0,  # 2°C tolerance
            'voltage': 0.1,      # 0.1V tolerance
            'current': 1.0       # 1A tolerance
        }
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                metric_type: str = 'soh', **kwargs) -> Dict[str, float]:
        """
        Evaluate battery-specific performance metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            metric_type: Type of battery metric ('soh', 'rul', 'soc', etc.)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of battery-specific performance metrics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        # Basic regression metrics
        base_metrics = RegressionMetrics().evaluate(y_true, y_pred)
        
        # Battery-specific metrics
        tolerance = self.tolerance_levels.get(metric_type, 0.05)
        
        # Accuracy within tolerance
        within_tolerance = np.mean(np.abs(y_true - y_pred) <= tolerance)
        
        # Safety constraint violations (if applicable)
        safety_violations = self._calculate_safety_violations(y_true, y_pred, metric_type)
        
        # Degradation tracking accuracy (for time-series predictions)
        degradation_accuracy = self._calculate_degradation_accuracy(y_true, y_pred)
        
        # Prediction confidence intervals
        confidence_bounds = self._calculate_confidence_bounds(y_true, y_pred)
        
        # Battery-specific performance indicators
        battery_metrics = {
            f'{metric_type}_accuracy_within_tolerance': within_tolerance,
            f'{metric_type}_safety_violation_rate': safety_violations,
            f'{metric_type}_degradation_tracking_accuracy': degradation_accuracy,
            f'{metric_type}_confidence_interval_width': confidence_bounds['width'],
            f'{metric_type}_confidence_coverage': confidence_bounds['coverage'],
            f'{metric_type}_prediction_bias': np.mean(y_pred - y_true),
            f'{metric_type}_prediction_variance': np.var(y_pred - y_true)
        }
        
        # Combine with base regression metrics
        combined_metrics = {**base_metrics, **battery_metrics}
        
        return combined_metrics
    
    def get_metric_names(self) -> List[str]:
        """Get list of battery-specific metric names."""
        base_names = RegressionMetrics().get_metric_names()
        battery_names = [
            'accuracy_within_tolerance', 'safety_violation_rate',
            'degradation_tracking_accuracy', 'confidence_interval_width',
            'confidence_coverage', 'prediction_bias', 'prediction_variance'
        ]
        return base_names + battery_names
    
    def _calculate_safety_violations(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   metric_type: str) -> float:
        """Calculate safety constraint violations."""
        safety_thresholds = {
            'soh': {'min': 0.0, 'max': 1.0},
            'soc': {'min': 0.0, 'max': 1.0},
            'temperature': {'min': -40, 'max': 80},
            'voltage': {'min': 0.0, 'max': 5.0},
            'current': {'min': -1000, 'max': 1000}
        }
        
        if metric_type not in safety_thresholds:
            return 0.0
        
        thresholds = safety_thresholds[metric_type]
        violations = np.sum((y_pred < thresholds['min']) | (y_pred > thresholds['max']))
        
        return violations / len(y_pred)
    
    def _calculate_degradation_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy of degradation trend prediction."""
        if len(y_true) < 3:
            return np.nan
        
        # Calculate actual and predicted trends
        true_trends = np.diff(y_true)
        pred_trends = np.diff(y_pred)
        
        # Check if trends are in the same direction
        same_direction = np.sign(true_trends) == np.sign(pred_trends)
        
        return np.mean(same_direction)
    
    def _calculate_confidence_bounds(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate prediction confidence bounds."""
        residuals = y_true - y_pred
        residual_std = np.std(residuals)
        
        # 95% confidence interval
        confidence_level = 0.95
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin = z_score * residual_std
        
        # Calculate coverage and width
        lower_bound = y_pred - margin
        upper_bound = y_pred + margin
        
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        width = np.mean(upper_bound - lower_bound)
        
        return {'coverage': coverage, 'width': width}

class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self):
        self.metrics_calculators = {
            'regression': RegressionMetrics(),
            'classification': ClassificationMetrics(),
            'battery': BatterySpecificMetrics()
        }
        self.performance_history = []
        
    def evaluate_model_performance(self, model: Any, test_data: Tuple[np.ndarray, np.ndarray],
                                 task_type: str = 'regression', 
                                 metric_type: str = 'soh') -> PerformanceResult:
        """
        Comprehensive model performance evaluation.
        
        Args:
            model: Trained model instance
            test_data: Tuple of (X_test, y_test)
            task_type: Type of ML task ('regression', 'classification', 'battery')
            metric_type: Specific metric type for battery tasks
            
        Returns:
            PerformanceResult object with comprehensive evaluation
        """
        X_test, y_test = test_data
        
        # Measure inference performance
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Measure memory usage
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple predictions for throughput measurement
        throughput_samples = min(1000, len(X_test))
        start_time = time.time()
        for _ in range(10):
            _ = model.predict(X_test[:throughput_samples])
        throughput_time = time.time() - start_time
        throughput = (10 * throughput_samples) / throughput_time
        
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_usage = memory_after - memory_before
        
        # Calculate metrics based on task type
        if task_type == 'battery':
            metrics = self.metrics_calculators['battery'].evaluate(y_test, y_pred, metric_type)
        elif task_type == 'classification':
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except:
                    pass
            metrics = self.metrics_calculators['classification'].evaluate(y_test, y_pred, y_pred_proba)
        else:
            metrics = self.metrics_calculators['regression'].evaluate(y_test, y_pred)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(y_test, y_pred, metrics)
        
        # Statistical significance testing
        statistical_significance = self._test_statistical_significance(y_test, y_pred)
        
        # Create performance result
        result = PerformanceResult(
            model_name=getattr(model, '__class__', type(model)).__name__,
            metrics=metrics,
            inference_time_ms=inference_time / len(X_test),
            memory_usage_mb=memory_usage,
            throughput_samples_per_sec=throughput,
            confidence_intervals=confidence_intervals,
            statistical_significance=statistical_significance,
            timestamp=pd.Timestamp.now().isoformat(),
            metadata={
                'task_type': task_type,
                'metric_type': metric_type,
                'test_samples': len(X_test),
                'feature_dimension': X_test.shape[1] if len(X_test.shape) > 1 else 1
            }
        )
        
        self.performance_history.append(result)
        return result
    
    def _calculate_confidence_intervals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     metrics: Dict[str, float], 
                                     confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for performance metrics."""
        n = len(y_true)
        confidence_intervals = {}
        
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        bootstrap_metrics = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n, size=n, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Calculate metrics
            try:
                boot_metrics = RegressionMetrics().evaluate(y_true_boot, y_pred_boot)
                bootstrap_metrics.append(boot_metrics)
            except:
                continue
        
        if bootstrap_metrics:
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            for metric_name in ['mse', 'mae', 'r2_score']:
                if metric_name in metrics:
                    values = [m[metric_name] for m in bootstrap_metrics if metric_name in m]
                    if values:
                        lower = np.percentile(values, lower_percentile)
                        upper = np.percentile(values, upper_percentile)
                        confidence_intervals[metric_name] = (lower, upper)
        
        return confidence_intervals
    
    def _test_statistical_significance(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Test statistical significance of predictions."""
        significance_tests = {}
        
        # Test if predictions are significantly different from random
        residuals = y_true - y_pred
        
        # One-sample t-test for residual mean
        t_stat, p_value = stats.ttest_1samp(residuals, 0)
        significance_tests['residual_mean_p_value'] = p_value
        
        # Test for correlation between predictions and true values
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            corr, p_value = stats.pearsonr(y_true, y_pred)
            significance_tests['correlation_p_value'] = p_value
            significance_tests['correlation_coefficient'] = corr
        
        return significance_tests
    
    def compare_models(self, results: List[PerformanceResult]) -> pd.DataFrame:
        """
        Compare performance across multiple models.
        
        Args:
            results: List of PerformanceResult objects
            
        Returns:
            DataFrame with comparative performance metrics
        """
        if not results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for result in results:
            row = {
                'model_name': result.model_name,
                'inference_time_ms': result.inference_time_ms,
                'memory_usage_mb': result.memory_usage_mb,
                'throughput_samples_per_sec': result.throughput_samples_per_sec,
                'timestamp': result.timestamp
            }
            
            # Add all metrics
            row.update(result.metrics)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Add ranking columns
        if len(comparison_df) > 1:
            # Rank by key metrics (lower is better for error metrics)
            if 'mse' in comparison_df.columns:
                comparison_df['mse_rank'] = comparison_df['mse'].rank()
            if 'mae' in comparison_df.columns:
                comparison_df['mae_rank'] = comparison_df['mae'].rank()
            if 'r2_score' in comparison_df.columns:
                comparison_df['r2_rank'] = comparison_df['r2_score'].rank(ascending=False)
            
            # Overall performance score
            rank_columns = [col for col in comparison_df.columns if col.endswith('_rank')]
            if rank_columns:
                comparison_df['overall_rank'] = comparison_df[rank_columns].mean(axis=1)
        
        return comparison_df.sort_values('overall_rank' if 'overall_rank' in comparison_df.columns else 'model_name')
    
    def generate_performance_report(self, results: List[PerformanceResult]) -> str:
        """
        Generate comprehensive performance report.
        
        Args:
            results: List of PerformanceResult objects
            
        Returns:
            Formatted performance report string
        """
        if not results:
            return "No performance results available."
        
        comparison_df = self.compare_models(results)
        
        report_lines = []
        report_lines.append("🔋 BatteryMind Performance Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Models Evaluated: {len(results)}")
        report_lines.append("")
        
        # Executive summary
        report_lines.append("📊 Executive Summary")
        report_lines.append("-" * 30)
        
        if 'overall_rank' in comparison_df.columns:
            best_model = comparison_df.iloc[0]['model_name']
            report_lines.append(f"Best Overall Model: {best_model}")
        
        if 'mse' in comparison_df.columns:
            best_accuracy = comparison_df.loc[comparison_df['mse'].idxmin(), 'model_name']
            report_lines.append(f"Most Accurate Model: {best_accuracy}")
        
        if 'inference_time_ms' in comparison_df.columns:
            fastest_model = comparison_df.loc[comparison_df['inference_time_ms'].idxmin(), 'model_name']
            report_lines.append(f"Fastest Model: {fastest_model}")
        
        report_lines.append("")
        
        # Detailed comparison
        report_lines.append("📈 Detailed Performance Comparison")
        report_lines.append("-" * 40)
        
        key_metrics = ['mse', 'mae', 'r2_score', 'inference_time_ms', 'memory_usage_mb']
        display_metrics = [m for m in key_metrics if m in comparison_df.columns]
        
        if display_metrics:
            # Create formatted table
            headers = ['Model'] + display_metrics
            col_widths = [max(len(str(comparison_df[col].max())), len(col)) + 2 
                         for col in ['model_name'] + display_metrics]
            
            # Header
            header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
            report_lines.append(header_line)
            report_lines.append("-" * len(header_line))
            
            # Data rows
            for _, row in comparison_df.iterrows():
                row_data = [str(row['model_name'])]
                for metric in display_metrics:
                    if metric in row:
                        if isinstance(row[metric], float):
                            row_data.append(f"{row[metric]:.4f}")
                        else:
                            row_data.append(str(row[metric]))
                    else:
                        row_data.append("N/A")
                
                data_line = " | ".join(d.ljust(w) for d, w in zip(row_data, col_widths))
                report_lines.append(data_line)
        
        report_lines.append("")
        
        # Statistical significance
        report_lines.append("🧮 Statistical Analysis")
        report_lines.append("-" * 30)
        
        for result in results:
            if result.statistical_significance:
                report_lines.append(f"{result.model_name}:")
                for test, p_value in result.statistical_significance.items():
                    significance = "Significant" if p_value < 0.05 else "Not significant"
                    report_lines.append(f"  {test}: p={p_value:.4f} ({significance})")
                report_lines.append("")
        
        # Recommendations
        report_lines.append("💡 Recommendations")
        report_lines.append("-" * 25)
        
        if len(results) > 1:
            # Performance vs efficiency trade-off
            if 'mse' in comparison_df.columns and 'inference_time_ms' in comparison_df.columns:
                best_accuracy_idx = comparison_df['mse'].idxmin()
                fastest_idx = comparison_df['inference_time_ms'].idxmin()
                
                if best_accuracy_idx != fastest_idx:
                    best_acc_model = comparison_df.iloc[best_accuracy_idx]['model_name']
                    fastest_model = comparison_df.iloc[fastest_idx]['model_name']
                    
                    report_lines.append(f"• For highest accuracy: Use {best_acc_model}")
                    report_lines.append(f"• For fastest inference: Use {fastest_model}")
                else:
                    optimal_model = comparison_df.iloc[best_accuracy_idx]['model_name']
                    report_lines.append(f"• Optimal model (accuracy + speed): {optimal_model}")
            
            # Memory efficiency
            if 'memory_usage_mb' in comparison_df.columns:
                most_efficient = comparison_df.loc[comparison_df['memory_usage_mb'].idxmin(), 'model_name']
                report_lines.append(f"• For memory-constrained environments: Use {most_efficient}")
        
        report_lines.append("• Monitor model performance regularly for degradation")
        report_lines.append("• Consider ensemble methods for critical applications")
        report_lines.append("• Validate results on real-world battery data")
        
        return "\n".join(report_lines)
    
    def visualize_performance_comparison(self, results: List[PerformanceResult]) -> None:
        """Create visualization comparing model performance."""
        if not results or len(results) < 2:
            print("Need at least 2 models for comparison visualization")
            return
        
        comparison_df = self.compare_models(results)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🔋 BatteryMind Model Performance Comparison', fontsize=16)
        
        # Plot 1: Accuracy comparison
        if 'r2_score' in comparison_df.columns:
            axes[0, 0].bar(comparison_df['model_name'], comparison_df['r2_score'])
            axes[0, 0].set_title('Model Accuracy (R² Score)')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Inference time comparison
        if 'inference_time_ms' in comparison_df.columns:
            axes[0, 1].bar(comparison_df['model_name'], comparison_df['inference_time_ms'])
            axes[0, 1].set_title('Inference Time')
            axes[0, 1].set_ylabel('Time (ms)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Memory usage comparison
        if 'memory_usage_mb' in comparison_df.columns:
            axes[1, 0].bar(comparison_df['model_name'], comparison_df['memory_usage_mb'])
            axes[1, 0].set_title('Memory Usage')
            axes[1, 0].set_ylabel('Memory (MB)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Performance vs efficiency scatter
        if 'r2_score' in comparison_df.columns and 'inference_time_ms' in comparison_df.columns:
            axes[1, 1].scatter(comparison_df['inference_time_ms'], comparison_df['r2_score'])
            
            # Add labels for each point
            for _, row in comparison_df.iterrows():
                axes[1, 1].annotate(row['model_name'], 
                                  (row['inference_time_ms'], row['r2_score']),
                                  xytext=(5, 5), textcoords='offset points')
            
            axes[1, 1].set_xlabel('Inference Time (ms)')
            axes[1, 1].set_ylabel('R² Score')
            axes[1, 1].set_title('Performance vs Speed Trade-off')
        
        plt.tight_layout()
        plt.show()
    
    def get_performance_history(self) -> List[PerformanceResult]:
        """Get historical performance results."""
        return self.performance_history.copy()
    
    def clear_performance_history(self) -> None:
        """Clear performance history."""
        self.performance_history.clear()
