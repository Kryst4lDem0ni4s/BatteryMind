"""
BatteryMind - Accuracy Metrics

Comprehensive accuracy evaluation metrics for battery management AI models.
This module provides specialized metrics for battery health prediction, degradation 
forecasting, remaining useful life (RUL) estimation, and anomaly detection.

Features:
- Battery-specific accuracy metrics
- State of Health (SoH) prediction accuracy
- Remaining Useful Life (RUL) estimation accuracy
- Degradation pattern recognition metrics
- Anomaly detection performance metrics
- Time series forecasting accuracy
- Multi-modal sensor fusion accuracy

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_absolute_percentage_error
)
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BaseAccuracyMetrics:
    """Base class for accuracy metrics with common functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base accuracy metrics.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.results = {}
        self.calculated_metrics = {}
        
    def reset(self):
        """Reset calculated metrics."""
        self.results = {}
        self.calculated_metrics = {}
    
    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Validate input arrays."""
        if len(y_true) != len(y_pred):
            raise ValueError("Input arrays must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        # Check for NaN values
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            raise ValueError("Input arrays cannot contain NaN values")
    
    def _calculate_confidence_interval(self, data: np.ndarray, confidence: float = 0.95):
        """Calculate confidence interval for metric."""
        if len(data) < 2:
            return None, None
        
        alpha = 1 - confidence
        mean = np.mean(data)
        std_err = stats.sem(data)
        interval = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=std_err)
        
        return interval[0], interval[1]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of calculated metrics."""
        return {
            'metrics': self.calculated_metrics,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

class SoHAccuracyMetrics(BaseAccuracyMetrics):
    """
    State of Health (SoH) prediction accuracy metrics.
    
    Specialized metrics for evaluating battery SoH prediction models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.soh_threshold = self.config.get('soh_threshold', 0.05)  # 5% threshold
        
    def calculate_soh_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive SoH accuracy metrics.
        
        Args:
            y_true: True SoH values (0-1 range)
            y_pred: Predicted SoH values (0-1 range)
            
        Returns:
            Dictionary of SoH accuracy metrics
        """
        self._validate_inputs(y_true, y_pred)
        
        # Ensure values are in valid SoH range
        y_true = np.clip(y_true, 0, 1)
        y_pred = np.clip(y_pred, 0, 1)
        
        # Basic regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # SoH-specific metrics
        absolute_errors = np.abs(y_true - y_pred)
        
        # Accuracy within threshold
        accuracy_within_threshold = np.mean(absolute_errors <= self.soh_threshold)
        
        # Accuracy at different thresholds
        accuracy_1_percent = np.mean(absolute_errors <= 0.01)
        accuracy_2_percent = np.mean(absolute_errors <= 0.02)
        accuracy_5_percent = np.mean(absolute_errors <= 0.05)
        accuracy_10_percent = np.mean(absolute_errors <= 0.10)
        
        # Health category accuracy (Good: >0.8, Fair: 0.6-0.8, Poor: <0.6)
        true_categories = self._categorize_soh(y_true)
        pred_categories = self._categorize_soh(y_pred)
        category_accuracy = accuracy_score(true_categories, pred_categories)
        
        # Correlation metrics
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
        
        # Directional accuracy (for trends)
        directional_accuracy = self._calculate_directional_accuracy(y_true, y_pred)
        
        # Bias analysis
        bias = np.mean(y_pred - y_true)
        
        # Outlier analysis
        outlier_percentage = self._calculate_outlier_percentage(y_true, y_pred)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'accuracy_within_threshold': accuracy_within_threshold,
            'accuracy_1_percent': accuracy_1_percent,
            'accuracy_2_percent': accuracy_2_percent,
            'accuracy_5_percent': accuracy_5_percent,
            'accuracy_10_percent': accuracy_10_percent,
            'category_accuracy': category_accuracy,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'directional_accuracy': directional_accuracy,
            'bias': bias,
            'outlier_percentage': outlier_percentage
        }
        
        self.calculated_metrics.update(metrics)
        return metrics
    
    def _categorize_soh(self, soh_values: np.ndarray) -> np.ndarray:
        """Categorize SoH values into health categories."""
        categories = np.zeros(len(soh_values), dtype=int)
        categories[soh_values >= 0.8] = 2  # Good
        categories[(soh_values >= 0.6) & (soh_values < 0.8)] = 1  # Fair
        categories[soh_values < 0.6] = 0  # Poor
        return categories
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy for trend prediction."""
        if len(y_true) < 2:
            return 0.0
        
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        
        # Same direction (both positive, both negative, or both zero)
        same_direction = np.sign(true_diff) == np.sign(pred_diff)
        return np.mean(same_direction)
    
    def _calculate_outlier_percentage(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate percentage of outlier predictions."""
        errors = np.abs(y_true - y_pred)
        q75, q25 = np.percentile(errors, [75, 25])
        iqr = q75 - q25
        outlier_threshold = q75 + 1.5 * iqr
        
        return np.mean(errors > outlier_threshold) * 100
    
    def evaluate_soh_prediction_quality(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate overall SoH prediction quality.
        
        Args:
            y_true: True SoH values
            y_pred: Predicted SoH values
            
        Returns:
            Dictionary with quality assessment
        """
        metrics = self.calculate_soh_accuracy(y_true, y_pred)
        
        # Quality scoring
        quality_score = 0
        weights = {
            'r2': 0.25,
            'accuracy_5_percent': 0.25,
            'category_accuracy': 0.20,
            'pearson_correlation': 0.15,
            'directional_accuracy': 0.15
        }
        
        for metric, weight in weights.items():
            if metric in metrics:
                quality_score += metrics[metric] * weight
        
        # Quality classification
        if quality_score >= 0.9:
            quality_class = 'Excellent'
        elif quality_score >= 0.8:
            quality_class = 'Good'
        elif quality_score >= 0.7:
            quality_class = 'Fair'
        else:
            quality_class = 'Poor'
        
        return {
            'quality_score': quality_score,
            'quality_class': quality_class,
            'metrics': metrics,
            'recommendations': self._generate_recommendations(metrics)
        }
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations based on metrics."""
        recommendations = []
        
        if metrics['r2'] < 0.8:
            recommendations.append("Consider improving model architecture or adding more features")
        
        if metrics['accuracy_5_percent'] < 0.8:
            recommendations.append("Focus on reducing prediction errors - consider ensemble methods")
        
        if metrics['category_accuracy'] < 0.85:
            recommendations.append("Improve categorical prediction accuracy with better thresholds")
        
        if abs(metrics['bias']) > 0.02:
            recommendations.append("Address prediction bias through calibration or data balancing")
        
        if metrics['outlier_percentage'] > 10:
            recommendations.append("Investigate and handle outlier predictions")
        
        return recommendations

class RULAccuracyMetrics(BaseAccuracyMetrics):
    """
    Remaining Useful Life (RUL) prediction accuracy metrics.
    
    Specialized metrics for evaluating battery RUL prediction models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.rul_threshold_days = self.config.get('rul_threshold_days', 30)  # 30 days threshold
        
    def calculate_rul_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              time_unit: str = 'days') -> Dict[str, float]:
        """
        Calculate RUL prediction accuracy metrics.
        
        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            time_unit: Time unit for RUL ('days', 'cycles', 'hours')
            
        Returns:
            Dictionary of RUL accuracy metrics
        """
        self._validate_inputs(y_true, y_pred)
        
        # Basic regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred + 1e-8)  # Add small value to avoid division by zero
        
        # RUL-specific metrics
        absolute_errors = np.abs(y_true - y_pred)
        relative_errors = absolute_errors / (y_true + 1e-8)
        
        # Accuracy within threshold
        accuracy_within_threshold = np.mean(absolute_errors <= self.rul_threshold_days)
        
        # Accuracy at different thresholds
        accuracy_10_percent = np.mean(relative_errors <= 0.10)
        accuracy_20_percent = np.mean(relative_errors <= 0.20)
        accuracy_30_percent = np.mean(relative_errors <= 0.30)
        
        # Early/late prediction analysis
        early_predictions = np.mean(y_pred > y_true)
        late_predictions = np.mean(y_pred < y_true)
        
        # Prediction confidence intervals
        prediction_intervals = self._calculate_prediction_intervals(y_true, y_pred)
        
        # Survival analysis metrics
        survival_metrics = self._calculate_survival_metrics(y_true, y_pred)
        
        # Correlation analysis
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'accuracy_within_threshold': accuracy_within_threshold,
            'accuracy_10_percent': accuracy_10_percent,
            'accuracy_20_percent': accuracy_20_percent,
            'accuracy_30_percent': accuracy_30_percent,
            'early_predictions_ratio': early_predictions,
            'late_predictions_ratio': late_predictions,
            'prediction_interval_coverage': prediction_intervals['coverage'],
            'prediction_interval_width': prediction_intervals['width'],
            'survival_concordance': survival_metrics['concordance'],
            'survival_roc_auc': survival_metrics['roc_auc'],
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'time_unit': time_unit
        }
        
        self.calculated_metrics.update(metrics)
        return metrics
    
    def _calculate_prediction_intervals(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate prediction interval metrics."""
        errors = y_true - y_pred
        
        # 95% prediction interval
        interval_lower = np.percentile(errors, 2.5)
        interval_upper = np.percentile(errors, 97.5)
        
        # Coverage (percentage of true values within interval)
        coverage = np.mean((errors >= interval_lower) & (errors <= interval_upper))
        
        # Average interval width
        width = interval_upper - interval_lower
        
        return {
            'coverage': coverage,
            'width': width,
            'lower_bound': interval_lower,
            'upper_bound': interval_upper
        }
    
    def _calculate_survival_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate survival analysis metrics for RUL prediction."""
        # Concordance index (C-index)
        concordance = self._calculate_concordance_index(y_true, y_pred)
        
        # ROC AUC for binary classification (failed vs not failed at specific time)
        threshold = np.median(y_true)
        true_binary = (y_true <= threshold).astype(int)
        pred_binary = (y_pred <= threshold).astype(int)
        
        try:
            roc_auc = roc_auc_score(true_binary, pred_binary)
        except:
            roc_auc = 0.5
        
        return {
            'concordance': concordance,
            'roc_auc': roc_auc
        }
    
    def _calculate_concordance_index(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate concordance index for survival prediction."""
        n = len(y_true)
        concordant = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if y_true[i] != y_true[j]:
                    total_pairs += 1
                    if ((y_true[i] < y_true[j]) and (y_pred[i] < y_pred[j])) or \
                       ((y_true[i] > y_true[j]) and (y_pred[i] > y_pred[j])):
                        concordant += 1
        
        return concordant / total_pairs if total_pairs > 0 else 0.5

class DegradationMetrics(BaseAccuracyMetrics):
    """
    Battery degradation prediction accuracy metrics.
    
    Specialized metrics for evaluating battery degradation forecasting models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.degradation_threshold = self.config.get('degradation_threshold', 0.02)  # 2% threshold
        
    def calculate_degradation_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate degradation prediction accuracy metrics.
        
        Args:
            y_true: True degradation values
            y_pred: Predicted degradation values
            
        Returns:
            Dictionary of degradation accuracy metrics
        """
        self._validate_inputs(y_true, y_pred)
        
        # Basic regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Degradation-specific metrics
        absolute_errors = np.abs(y_true - y_pred)
        
        # Accuracy within threshold
        accuracy_within_threshold = np.mean(absolute_errors <= self.degradation_threshold)
        
        # Trend accuracy
        trend_accuracy = self._calculate_trend_accuracy(y_true, y_pred)
        
        # Rate of degradation accuracy
        rate_accuracy = self._calculate_rate_accuracy(y_true, y_pred)
        
        # Monotonicity check (degradation should be monotonic)
        monotonicity_score = self._calculate_monotonicity_score(y_true, y_pred)
        
        # Seasonality detection accuracy
        seasonality_score = self._calculate_seasonality_accuracy(y_true, y_pred)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'accuracy_within_threshold': accuracy_within_threshold,
            'trend_accuracy': trend_accuracy,
            'rate_accuracy': rate_accuracy,
            'monotonicity_score': monotonicity_score,
            'seasonality_score': seasonality_score
        }
        
        self.calculated_metrics.update(metrics)
        return metrics
    
    def _calculate_trend_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate trend prediction accuracy."""
        if len(y_true) < 3:
            return 0.0
        
        # Calculate trends using moving averages
        window = min(5, len(y_true) // 3)
        true_trend = np.convolve(y_true, np.ones(window)/window, mode='valid')
        pred_trend = np.convolve(y_pred, np.ones(window)/window, mode='valid')
        
        # Calculate trend direction accuracy
        true_direction = np.sign(np.diff(true_trend))
        pred_direction = np.sign(np.diff(pred_trend))
        
        return np.mean(true_direction == pred_direction)
    
    def _calculate_rate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate degradation rate accuracy."""
        if len(y_true) < 2:
            return 0.0
        
        true_rate = np.mean(np.diff(y_true))
        pred_rate = np.mean(np.diff(y_pred))
        
        if true_rate == 0:
            return 1.0 if pred_rate == 0 else 0.0
        
        return 1.0 - abs(pred_rate - true_rate) / abs(true_rate)
    
    def _calculate_monotonicity_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate monotonicity score for degradation prediction."""
        if len(y_true) < 2:
            return 1.0
        
        # Check if degradation is monotonic (non-decreasing)
        true_monotonic = np.all(np.diff(y_true) >= 0)
        pred_monotonic = np.all(np.diff(y_pred) >= 0)
        
        # Calculate monotonicity violation rate
        true_violations = np.sum(np.diff(y_true) < 0) / (len(y_true) - 1)
        pred_violations = np.sum(np.diff(y_pred) < 0) / (len(y_pred) - 1)
        
        return 1.0 - abs(true_violations - pred_violations)
    
    def _calculate_seasonality_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate seasonality detection accuracy."""
        if len(y_true) < 12:  # Need at least 12 points for seasonality
            return 1.0
        
        # Simple seasonality detection using autocorrelation
        true_autocorr = np.corrcoef(y_true[:-1], y_true[1:])[0, 1]
        pred_autocorr = np.corrcoef(y_pred[:-1], y_pred[1:])[0, 1]
        
        return 1.0 - abs(true_autocorr - pred_autocorr)

class AnomalyDetectionMetrics(BaseAccuracyMetrics):
    """
    Anomaly detection accuracy metrics for battery systems.
    
    Specialized metrics for evaluating battery anomaly detection models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.anomaly_threshold = self.config.get('anomaly_f1_threshold', 0.8)
        
    def calculate_anomaly_detection_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray,
                                           y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate anomaly detection accuracy metrics.
        
        Args:
            y_true: True anomaly labels (0=normal, 1=anomaly)
            y_pred: Predicted anomaly labels
            y_scores: Optional anomaly scores for ROC/PR curves
            
        Returns:
            Dictionary of anomaly detection metrics
        """
        self._validate_inputs(y_true, y_pred)
        
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # False positive rate and false negative rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Balanced accuracy
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # Matthews correlation coefficient
        mcc = self._calculate_mcc(tp, tn, fp, fn)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'balanced_accuracy': balanced_accuracy,
            'matthews_correlation': mcc,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        # Add ROC AUC if scores available
        if y_scores is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_scores)
                metrics['roc_auc'] = roc_auc
            except:
                metrics['roc_auc'] = 0.5
        
        self.calculated_metrics.update(metrics)
        return metrics
    
    def _calculate_mcc(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """Calculate Matthews Correlation Coefficient."""
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator == 0:
            return 0.0
        return (tp * tn - fp * fn) / denominator

class TimeSeriesMetrics(BaseAccuracyMetrics):
    """
    Time series prediction accuracy metrics for battery data.
    
    Specialized metrics for evaluating time series forecasting models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.mape_threshold = self.config.get('time_series_mape_threshold', 0.1)
        
    def calculate_time_series_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     timestamps: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate time series prediction accuracy metrics.
        
        Args:
            y_true: True time series values
            y_pred: Predicted time series values
            timestamps: Optional timestamps for temporal analysis
            
        Returns:
            Dictionary of time series accuracy metrics
        """
        self._validate_inputs(y_true, y_pred)
        
        # Basic regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred + 1e-8)
        
        # Time series specific metrics
        # Symmetric Mean Absolute Percentage Error (SMAPE)
        smape = self._calculate_smape(y_true, y_pred)
        
        # Mean Absolute Scaled Error (MASE)
        mase = self._calculate_mase(y_true, y_pred)
        
        # Directional accuracy
        directional_accuracy = self._calculate_directional_accuracy(y_true, y_pred)
        
        # Forecast bias
        forecast_bias = np.mean(y_pred - y_true)
        
        # Tracking signal
        tracking_signal = self._calculate_tracking_signal(y_true, y_pred)
        
        # Temporal consistency
        temporal_consistency = self._calculate_temporal_consistency(y_true, y_pred)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'smape': smape,
            'mase': mase,
            'directional_accuracy': directional_accuracy,
            'forecast_bias': forecast_bias,
            'tracking_signal': tracking_signal,
            'temporal_consistency': temporal_consistency
        }
        
        # Add temporal analysis if timestamps available
        if timestamps is not None:
            temporal_metrics = self._calculate_temporal_metrics(y_true, y_pred, timestamps)
            metrics.update(temporal_metrics)
        
        self.calculated_metrics.update(metrics)
        return metrics
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        return np.mean(np.abs(y_true - y_pred) / (denominator + 1e-8)) * 100
    
    def _calculate_mase(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Scaled Error."""
        if len(y_true) < 2:
            return 0.0
        
        # Naive forecast error (using seasonal naive with period=1)
        naive_error = np.mean(np.abs(y_true[1:] - y_true[:-1]))
        
        if naive_error == 0:
            return 0.0
        
        mae = mean_absolute_error(y_true, y_pred)
        return mae / naive_error
    
    def _calculate_tracking_signal(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate tracking signal for forecast bias detection."""
        errors = y_true - y_pred
        cumulative_error = np.cumsum(errors)
        mae = np.mean(np.abs(errors))
        
        if mae == 0:
            return 0.0
        
        return cumulative_error[-1] / (mae * len(errors))
    
    def _calculate_temporal_consistency(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate temporal consistency of predictions."""
        if len(y_true) < 2:
            return 1.0
        
        true_changes = np.diff(y_true)
        pred_changes = np.diff(y_pred)
        
        # Correlation of changes
        if np.std(true_changes) == 0 or np.std(pred_changes) == 0:
            return 1.0
        
        consistency = np.corrcoef(true_changes, pred_changes)[0, 1]
        return consistency if not np.isnan(consistency) else 0.0
    
    def _calculate_temporal_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   timestamps: np.ndarray) -> Dict[str, float]:
        """Calculate temporal-specific metrics."""
        # Convert timestamps to datetime if needed
        if isinstance(timestamps[0], (int, float)):
            timestamps = pd.to_datetime(timestamps, unit='s')
        
        # Create time series
        ts_true = pd.Series(y_true, index=timestamps)
        ts_pred = pd.Series(y_pred, index=timestamps)
        
        # Seasonal decomposition accuracy (if enough data)
        temporal_metrics = {}
        
        if len(ts_true) >= 24:  # At least 24 points for seasonal analysis
            try:
                # Simple seasonal strength calculation
                seasonal_strength = self._calculate_seasonal_strength(ts_true, ts_pred)
                temporal_metrics['seasonal_accuracy'] = seasonal_strength
            except:
                temporal_metrics['seasonal_accuracy'] = 0.0
        
        # Trend accuracy
        trend_accuracy = self._calculate_trend_accuracy(ts_true.values, ts_pred.values)
        temporal_metrics['trend_accuracy'] = trend_accuracy
        
        return temporal_metrics
    
    def _calculate_seasonal_strength(self, ts_true: pd.Series, ts_pred: pd.Series) -> float:
        """Calculate seasonal prediction strength."""
        # Simple seasonal strength using autocorrelation
        true_seasonal = ts_true.autocorr(lag=7) if len(ts_true) > 7 else 0
        pred_seasonal = ts_pred.autocorr(lag=7) if len(ts_pred) > 7 else 0
        
        return 1.0 - abs(true_seasonal - pred_seasonal)

class MultiModalMetrics(BaseAccuracyMetrics):
    """
    Multi-modal sensor fusion accuracy metrics.
    
    Specialized metrics for evaluating multi-modal sensor fusion models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.modal_weights = self.config.get('modal_weights', {})
        
    def calculate_multimodal_accuracy(self, predictions: Dict[str, np.ndarray],
                                     ground_truth: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate multi-modal fusion accuracy metrics.
        
        Args:
            predictions: Dictionary of predictions by modality
            ground_truth: Dictionary of ground truth by modality
            
        Returns:
            Dictionary of multi-modal accuracy metrics
        """
        metrics = {}
        
        # Individual modality performance
        for modality in predictions.keys():
            if modality in ground_truth:
                y_true = ground_truth[modality]
                y_pred = predictions[modality]
                
                self._validate_inputs(y_true, y_pred)
                
                # Calculate basic metrics for each modality
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                metrics[f'{modality}_mae'] = mae
                metrics[f'{modality}_r2'] = r2
        
        # Fusion effectiveness
        if len(predictions) > 1:
            fusion_metrics = self._calculate_fusion_effectiveness(predictions, ground_truth)
            metrics.update(fusion_metrics)
        
        # Cross-modal consistency
        consistency_metrics = self._calculate_cross_modal_consistency(predictions)
        metrics.update(consistency_metrics)
        
        self.calculated_metrics.update(metrics)
        return metrics
    
    def _calculate_fusion_effectiveness(self, predictions: Dict[str, np.ndarray],
                                       ground_truth: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate fusion effectiveness metrics."""
        # Simple fusion (average of all modalities)
        modalities = list(predictions.keys())
        
        if len(modalities) < 2:
            return {}
        
        # Calculate weighted average
        weights = [self.modal_weights.get(mod, 1.0) for mod in modalities]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        fused_predictions = np.zeros_like(predictions[modalities[0]])
        for i, modality in enumerate(modalities):
            fused_predictions += weights[i] * predictions[modality]
        
        # Compare fused prediction to individual modalities
        metrics = {}
        
        # Assume we have an overall ground truth (average of all modalities)
        overall_truth = np.mean([ground_truth[mod] for mod in modalities if mod in ground_truth], axis=0)
        
        # Fused performance
        fused_mae = mean_absolute_error(overall_truth, fused_predictions)
        fused_r2 = r2_score(overall_truth, fused_predictions)
        
        metrics['fused_mae'] = fused_mae
        metrics['fused_r2'] = fused_r2
        
        # Best individual modality performance
        best_individual_mae = float('inf')
        best_individual_r2 = float('-inf')
        
        for modality in modalities:
            if modality in ground_truth:
                mae = mean_absolute_error(ground_truth[modality], predictions[modality])
                r2 = r2_score(ground_truth[modality], predictions[modality])
                
                if mae < best_individual_mae:
                    best_individual_mae = mae
                if r2 > best_individual_r2:
                    best_individual_r2 = r2
        
        # Fusion improvement
        metrics['fusion_mae_improvement'] = (best_individual_mae - fused_mae) / best_individual_mae
        metrics['fusion_r2_improvement'] = (fused_r2 - best_individual_r2) / max(best_individual_r2, 1e-8)
        
        return metrics
    
    def _calculate_cross_modal_consistency(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate cross-modal consistency metrics."""
        modalities = list(predictions.keys())
        
        if len(modalities) < 2:
            return {}
        
        # Pairwise correlations
        correlations = []
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                if len(predictions[mod1]) == len(predictions[mod2]):
                    corr, _ = pearsonr(predictions[mod1], predictions[mod2])
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        metrics = {}
        if correlations:
            metrics['cross_modal_correlation_mean'] = np.mean(correlations)
            metrics['cross_modal_correlation_std'] = np.std(correlations)
            metrics['cross_modal_consistency'] = np.mean(correlations)
        
        return metrics

class BatteryEvaluationSuite:
    """
    Comprehensive evaluation suite for battery AI models.
    
    Combines all battery-specific metrics into a unified evaluation framework.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the battery evaluation suite.
        
        Args:
            config: Configuration dictionary with thresholds and settings
        """
        self.config = config or {}
        
        # Initialize individual metric calculators
        self.soh_metrics = SoHAccuracyMetrics(config)
        self.rul_metrics = RULAccuracyMetrics(config)
        self.degradation_metrics = DegradationMetrics(config)
        self.anomaly_metrics = AnomalyDetectionMetrics(config)
        self.timeseries_metrics = TimeSeriesMetrics(config)
        self.multimodal_metrics = MultiModalMetrics(config)
        
        # Results storage
        self.evaluation_results = {}
        
    def evaluate_model(self, model_type: str, y_true: np.ndarray, y_pred: np.ndarray,
                      **kwargs) -> Dict[str, Any]:
        """
        Evaluate a model based on its type.
        
        Args:
            model_type: Type of model ('soh', 'rul', 'degradation', 'anomaly', 'timeseries', 'multimodal')
            y_true: True values
            y_pred: Predicted values
            **kwargs: Additional arguments for specific metrics
            
        Returns:
            Dictionary of evaluation results
        """
        if model_type == 'soh':
            results = self.soh_metrics.calculate_soh_accuracy(y_true, y_pred)
        elif model_type == 'rul':
            results = self.rul_metrics.calculate_rul_accuracy(y_true, y_pred, **kwargs)
        elif model_type == 'degradation':
            results = self.degradation_metrics.calculate_degradation_accuracy(y_true, y_pred)
        elif model_type == 'anomaly':
            results = self.anomaly_metrics.calculate_anomaly_detection_accuracy(y_true, y_pred, **kwargs)
        elif model_type == 'timeseries':
            results = self.timeseries_metrics.calculate_time_series_accuracy(y_true, y_pred, **kwargs)
        elif model_type == 'multimodal':
            results = self.multimodal_metrics.calculate_multimodal_accuracy(y_true, y_pred, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.evaluation_results[model_type] = results
        return results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Returns:
            Dictionary containing complete evaluation report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results': self.evaluation_results,
            'summary': self._generate_summary(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics across all evaluations."""
        summary = {
            'total_evaluations': len(self.evaluation_results),
            'model_types': list(self.evaluation_results.keys()),
            'overall_performance': {}
        }
        
        # Calculate overall performance metrics
        all_r2_scores = []
        all_mae_scores = []
        
        for model_type, results in self.evaluation_results.items():
            if 'r2' in results:
                all_r2_scores.append(results['r2'])
            if 'mae' in results:
                all_mae_scores.append(results['mae'])
        
        if all_r2_scores:
            summary['overall_performance']['mean_r2'] = np.mean(all_r2_scores)
            summary['overall_performance']['std_r2'] = np.std(all_r2_scores)
        
        if all_mae_scores:
            summary['overall_performance']['mean_mae'] = np.mean(all_mae_scores)
            summary['overall_performance']['std_mae'] = np.std(all_mae_scores)
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        for model_type, results in self.evaluation_results.items():
            if model_type == 'soh':
                if results.get('r2', 0) < 0.8:
                    recommendations.append(f"SoH model: Consider improving model architecture (R² = {results.get('r2', 0):.3f})")
                if results.get('accuracy_5_percent', 0) < 0.8:
                    recommendations.append(f"SoH model: Focus on reducing prediction errors within 5% threshold")
            
            elif model_type == 'rul':
                if results.get('mape', 1) > 0.2:
                    recommendations.append(f"RUL model: High MAPE ({results.get('mape', 0):.1%}) - consider better features")
                if results.get('accuracy_20_percent', 0) < 0.7:
                    recommendations.append(f"RUL model: Improve accuracy within 20% threshold")
            
            elif model_type == 'anomaly':
                if results.get('f1_score', 0) < 0.8:
                    recommendations.append(f"Anomaly model: Low F1 score ({results.get('f1_score', 0):.3f}) - address class imbalance")
                if results.get('false_positive_rate', 1) > 0.1:
                    recommendations.append(f"Anomaly model: High false positive rate - fine-tune decision threshold")
        
        return recommendations

# Export all classes
__all__ = [
    'BaseAccuracyMetrics',
    'SoHAccuracyMetrics',
    'RULAccuracyMetrics',
    'DegradationMetrics',
    'AnomalyDetectionMetrics',
    'TimeSeriesMetrics',
    'MultiModalMetrics',
    'BatteryEvaluationSuite'
]
