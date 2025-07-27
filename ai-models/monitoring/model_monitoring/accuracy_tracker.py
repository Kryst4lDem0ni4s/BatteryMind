"""
BatteryMind - Accuracy Tracker

Advanced accuracy monitoring system for battery AI models, providing real-time
accuracy tracking, degradation detection, and automated alerting for model
performance issues across all model types in the BatteryMind ecosystem.

Features:
- Real-time accuracy monitoring and tracking
- Historical accuracy trend analysis
- Automated accuracy degradation detection
- Multi-model accuracy comparison
- Statistical significance testing for accuracy changes
- Automated alerting and notification system
- Accuracy baseline management and calibration

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json
import pickle
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import threading
import time

# BatteryMind imports
from ...utils.logging_utils import setup_logger
from ...utils.data_utils import StatisticalAnalyzer
from ...utils.visualization import AccuracyVisualizer
from ...evaluation.metrics.accuracy_metrics import AccuracyMetrics

# Configure logging
logger = setup_logger(__name__)

@dataclass
class AccuracyThresholds:
    """
    Accuracy thresholds configuration for different model types and metrics.
    
    Attributes:
        # General accuracy thresholds
        min_accuracy (float): Minimum acceptable accuracy
        min_precision (float): Minimum acceptable precision
        min_recall (float): Minimum acceptable recall
        min_f1_score (float): Minimum acceptable F1 score
        min_r2_score (float): Minimum acceptable R² score
        
        # Error thresholds
        max_mse (float): Maximum acceptable MSE
        max_mae (float): Maximum acceptable MAE
        max_mape (float): Maximum acceptable MAPE
        
        # Degradation thresholds
        max_accuracy_drop_percent (float): Maximum acceptable accuracy drop percentage
        max_performance_variance (float): Maximum acceptable performance variance
        min_samples_for_alert (int): Minimum samples before triggering alerts
        
        # Model-specific thresholds
        model_specific_thresholds (Dict[str, Dict[str, float]]): Model-specific thresholds
        
        # Statistical thresholds
        significance_level (float): Statistical significance level for tests
        confidence_interval (float): Confidence interval for accuracy estimates
    """
    # General accuracy thresholds
    min_accuracy: float = 0.85
    min_precision: float = 0.80
    min_recall: float = 0.80
    min_f1_score: float = 0.80
    min_r2_score: float = 0.85
    
    # Error thresholds
    max_mse: float = 0.01
    max_mae: float = 0.05
    max_mape: float = 0.10
    
    # Degradation thresholds
    max_accuracy_drop_percent: float = 5.0
    max_performance_variance: float = 0.02
    min_samples_for_alert: int = 100
    
    # Model-specific thresholds
    model_specific_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'transformer': {
            'min_accuracy': 0.90,
            'min_r2_score': 0.88,
            'max_mae': 0.03
        },
        'federated': {
            'min_accuracy': 0.85,
            'min_r2_score': 0.82,
            'max_mae': 0.05
        },
        'rl_agent': {
            'min_success_rate': 0.85,
            'min_reward_threshold': 0.80
        },
        'ensemble': {
            'min_accuracy': 0.92,
            'min_r2_score': 0.90,
            'max_mae': 0.02
        }
    })
    
    # Statistical thresholds
    significance_level: float = 0.05
    confidence_interval: float = 0.95

@dataclass
class AccuracyTrackerConfig:
    """
    Configuration for accuracy tracking system.
    
    Attributes:
        # Tracking settings
        tracking_interval_seconds (int): Tracking interval in seconds
        history_retention_days (int): History retention period in days
        enable_real_time_tracking (bool): Enable real-time tracking
        
        # Analysis settings
        analysis_window_size (int): Window size for trend analysis
        baseline_calculation_method (str): Method for baseline calculation
        enable_statistical_testing (bool): Enable statistical significance testing
        
        # Alerting settings
        enable_alerting (bool): Enable accuracy alerting
        alert_cooldown_minutes (int): Cooldown period between alerts
        alert_recipients (List[str]): Alert recipients
        
        # Storage settings
        save_detailed_metrics (bool): Save detailed accuracy metrics
        metrics_storage_path (str): Path for storing metrics
        enable_backup (bool): Enable metrics backup
        
        # Visualization settings
        enable_visualization (bool): Enable accuracy visualization
        plot_update_interval_minutes (int): Plot update interval
        max_plot_points (int): Maximum points to show in plots
    """
    # Tracking settings
    tracking_interval_seconds: int = 60
    history_retention_days: int = 30
    enable_real_time_tracking: bool = True
    
    # Analysis settings
    analysis_window_size: int = 100
    baseline_calculation_method: str = "rolling_average"
    enable_statistical_testing: bool = True
    
    # Alerting settings
    enable_alerting: bool = True
    alert_cooldown_minutes: int = 15
    alert_recipients: List[str] = field(default_factory=list)
    
    # Storage settings
    save_detailed_metrics: bool = True
    metrics_storage_path: str = "./accuracy_metrics"
    enable_backup: bool = True
    
    # Visualization settings
    enable_visualization: bool = True
    plot_update_interval_minutes: int = 5
    max_plot_points: int = 1000

@dataclass
class AccuracyRecord:
    """Individual accuracy record."""
    timestamp: datetime
    model_id: str
    model_type: str
    accuracy_metrics: Dict[str, float]
    sample_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class AccuracyTracker:
    """
    Comprehensive accuracy tracking system for battery AI models.
    """
    
    def __init__(self, 
                 thresholds: AccuracyThresholds,
                 config: AccuracyTrackerConfig):
        """
        Initialize accuracy tracker.
        
        Args:
            thresholds: Accuracy thresholds configuration
            config: Tracker configuration
        """
        self.thresholds = thresholds
        self.config = config
        
        # Initialize tracking data structures
        self.accuracy_history = {}  # model_id -> deque of AccuracyRecord
        self.baseline_metrics = {}  # model_id -> baseline metrics
        self.alert_history = []
        self.last_alert_time = {}  # model_id -> datetime
        
        # Initialize components
        self.accuracy_metrics = AccuracyMetrics()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = AccuracyVisualizer() if config.enable_visualization else None
        
        # Threading and monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Create storage directory
        Path(self.config.metrics_storage_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("AccuracyTracker initialized with comprehensive monitoring")
    
    def start_monitoring(self):
        """Start real-time accuracy monitoring."""
        if self.config.enable_real_time_tracking and not self.monitoring_active:
            self.monitoring_active = True
            self.stop_event.clear()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Real-time accuracy monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time accuracy monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            self.stop_event.set()
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            logger.info("Real-time accuracy monitoring stopped")
    
    def track_accuracy(self, 
                      model_id: str,
                      model_type: str,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Track accuracy for a specific model prediction.
        
        Args:
            model_id: Unique model identifier
            model_type: Type of model
            y_true: True values
            y_pred: Predicted values
            metadata: Additional metadata
            
        Returns:
            Dictionary containing tracking results and alerts
        """
        try:
            # Calculate accuracy metrics
            accuracy_metrics = self._calculate_comprehensive_metrics(
                y_true, y_pred, model_type
            )
            
            # Create accuracy record
            record = AccuracyRecord(
                timestamp=datetime.now(),
                model_id=model_id,
                model_type=model_type,
                accuracy_metrics=accuracy_metrics,
                sample_count=len(y_true),
                metadata=metadata or {}
            )
            
            # Store record
            self._store_accuracy_record(record)
            
            # Update baseline if needed
            self._update_baseline_metrics(model_id, accuracy_metrics)
            
            # Check for accuracy issues
            alerts = self._check_accuracy_alerts(model_id, accuracy_metrics)
            
            # Perform trend analysis
            trend_analysis = self._analyze_accuracy_trends(model_id)
            
            # Update visualizations
            if self.visualizer:
                self._update_visualizations(model_id)
            
            # Save metrics if configured
            if self.config.save_detailed_metrics:
                self._save_metrics(record)
            
            tracking_results = {
                'timestamp': record.timestamp.isoformat(),
                'model_id': model_id,
                'model_type': model_type,
                'accuracy_metrics': accuracy_metrics,
                'alerts': alerts,
                'trend_analysis': trend_analysis,
                'baseline_comparison': self._compare_with_baseline(model_id, accuracy_metrics),
                'sample_count': len(y_true)
            }
            
            logger.debug(f"Accuracy tracked for model {model_id}: {accuracy_metrics}")
            return tracking_results
            
        except Exception as e:
            logger.error(f"Error tracking accuracy for model {model_id}: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'model_id': model_id
            }
    
    def _calculate_comprehensive_metrics(self, 
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       model_type: str) -> Dict[str, float]:
        """Calculate comprehensive accuracy metrics."""
        metrics = {}
        
        # Determine if this is a classification or regression task
        is_classification = self._is_classification_task(y_true, y_pred)
        
        if is_classification:
            # Classification metrics
            metrics.update(self._calculate_classification_metrics(y_true, y_pred))
        else:
            # Regression metrics
            metrics.update(self._calculate_regression_metrics(y_true, y_pred))
        
        # Model-specific metrics
        if model_type == 'rl_agent':
            metrics.update(self._calculate_rl_specific_metrics(y_true, y_pred))
        elif model_type == 'ensemble':
            metrics.update(self._calculate_ensemble_specific_metrics(y_true, y_pred))
        
        return metrics
    
    def _is_classification_task(self, y_true: np.ndarray, y_pred: np.ndarray) -> bool:
        """Determine if this is a classification task."""
        # Check if all values are integers and within a reasonable range for classification
        return (np.all(np.equal(np.mod(y_true, 1), 0)) and 
                np.all(np.equal(np.mod(y_pred, 1), 0)) and
                len(np.unique(y_true)) <= 20)
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate classification accuracy metrics."""
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Additional metrics for binary classification
            if len(np.unique(y_true)) == 2:
                metrics['sensitivity'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
                metrics['specificity'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
                
                # ROC AUC if probabilities are available
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
                except:
                    pass
            
        except Exception as e:
            logger.warning(f"Error calculating classification metrics: {e}")
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression accuracy metrics."""
        metrics = {}
        
        try:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2_score'] = r2_score(y_true, y_pred)
            
            # MAPE (Mean Absolute Percentage Error)
            mask = y_true != 0
            if np.any(mask):
                metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            
            # Additional regression metrics
            residuals = y_true - y_pred
            metrics['residual_std'] = np.std(residuals)
            metrics['residual_mean'] = np.mean(residuals)
            
            # Explained variance
            if np.var(y_true) > 0:
                metrics['explained_variance'] = 1 - np.var(residuals) / np.var(y_true)
            
        except Exception as e:
            logger.warning(f"Error calculating regression metrics: {e}")
        
        return metrics
    
    def _calculate_rl_specific_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate RL-specific metrics."""
        metrics = {}
        
        try:
            # Assume y_true are rewards and y_pred are predicted values
            if len(y_true) > 0:
                metrics['success_rate'] = np.mean(y_true > 0)
                metrics['average_reward'] = np.mean(y_true)
                metrics['reward_std'] = np.std(y_true)
                metrics['max_reward'] = np.max(y_true)
                metrics['min_reward'] = np.min(y_true)
                
                # Value function accuracy
                if len(y_pred) == len(y_true):
                    metrics['value_function_accuracy'] = 1 - np.mean(np.abs(y_true - y_pred)) / (np.std(y_true) + 1e-8)
        
        except Exception as e:
            logger.warning(f"Error calculating RL metrics: {e}")
        
        return metrics
    
    def _calculate_ensemble_specific_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate ensemble-specific metrics."""
        metrics = {}
        
        try:
            # Ensemble consistency (variance of predictions)
            if hasattr(y_pred, 'shape') and len(y_pred.shape) > 1:
                # If predictions from multiple models
                metrics['ensemble_variance'] = np.mean(np.var(y_pred, axis=1))
                metrics['ensemble_agreement'] = 1 / (1 + metrics['ensemble_variance'])
            
            # Ensemble confidence (based on prediction spread)
            pred_std = np.std(y_pred)
            metrics['prediction_confidence'] = 1 / (1 + pred_std)
            
        except Exception as e:
            logger.warning(f"Error calculating ensemble metrics: {e}")
        
        return metrics
    
    def _store_accuracy_record(self, record: AccuracyRecord):
        """Store accuracy record in history."""
        model_id = record.model_id
        
        if model_id not in self.accuracy_history:
            self.accuracy_history[model_id] = deque(maxlen=10000)  # Limit memory usage
        
        self.accuracy_history[model_id].append(record)
        
        # Clean old records
        cutoff_time = datetime.now() - timedelta(days=self.config.history_retention_days)
        while (self.accuracy_history[model_id] and 
               self.accuracy_history[model_id][0].timestamp < cutoff_time):
            self.accuracy_history[model_id].popleft()
    
    def _update_baseline_metrics(self, model_id: str, current_metrics: Dict[str, float]):
        """Update baseline metrics for the model."""
        if model_id not in self.baseline_metrics:
            # Initialize baseline with current metrics
            self.baseline_metrics[model_id] = current_metrics.copy()
        else:
            # Update baseline using rolling average
            if self.config.baseline_calculation_method == "rolling_average":
                alpha = 0.1  # Learning rate for exponential moving average
                for metric_name, current_value in current_metrics.items():
                    if metric_name in self.baseline_metrics[model_id]:
                        baseline_value = self.baseline_metrics[model_id][metric_name]
                        updated_value = alpha * current_value + (1 - alpha) * baseline_value
                        self.baseline_metrics[model_id][metric_name] = updated_value
                    else:
                        self.baseline_metrics[model_id][metric_name] = current_value
    
    def _check_accuracy_alerts(self, model_id: str, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for accuracy-related alerts."""
        alerts = []
        
        # Check cooldown period
        if (model_id in self.last_alert_time and 
            datetime.now() - self.last_alert_time[model_id] < timedelta(minutes=self.config.alert_cooldown_minutes)):
            return alerts
        
        # Get applicable thresholds
        model_type = self._get_model_type(model_id)
        thresholds = self._get_applicable_thresholds(model_type)
        
        # Check threshold violations
        for metric_name, threshold_value in thresholds.items():
            if metric_name.startswith('min_') and metric_name.replace('min_', '') in current_metrics:
                current_value = current_metrics[metric_name.replace('min_', '')]
                if current_value < threshold_value:
                    alerts.append({
                        'type': 'threshold_violation',
                        'severity': 'high',
                        'metric': metric_name,
                        'current_value': current_value,
                        'threshold': threshold_value,
                        'message': f"Metric {metric_name} ({current_value:.4f}) below threshold ({threshold_value:.4f})"
                    })
            
            elif metric_name.startswith('max_') and metric_name.replace('max_', '') in current_metrics:
                current_value = current_metrics[metric_name.replace('max_', '')]
                if current_value > threshold_value:
                    alerts.append({
                        'type': 'threshold_violation',
                        'severity': 'high',
                        'metric': metric_name,
                        'current_value': current_value,
                        'threshold': threshold_value,
                        'message': f"Metric {metric_name} ({current_value:.4f}) above threshold ({threshold_value:.4f})"
                    })
        
        # Check for accuracy degradation
        degradation_alerts = self._check_accuracy_degradation(model_id, current_metrics)
        alerts.extend(degradation_alerts)
        
        # Record alert time if alerts were generated
        if alerts:
            self.last_alert_time[model_id] = datetime.now()
            self.alert_history.extend(alerts)
        
        return alerts
    
    def _check_accuracy_degradation(self, model_id: str, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for accuracy degradation."""
        alerts = []
        
        if model_id not in self.accuracy_history:
            return alerts
        
        history = list(self.accuracy_history[model_id])
        
        if len(history) < self.thresholds.min_samples_for_alert:
            return alerts
        
        # Get recent history for comparison
        recent_window = min(50, len(history))
        recent_records = history[-recent_window:]
        baseline_records = history[-100:-recent_window] if len(history) >= 100 else history[:-recent_window]
        
        if not baseline_records:
            return alerts
        
        # Compare recent performance with baseline
        for metric_name in current_metrics.keys():
            recent_values = [r.accuracy_metrics.get(metric_name, 0) for r in recent_records]
            baseline_values = [r.accuracy_metrics.get(metric_name, 0) for r in baseline_records]
            
            if len(recent_values) < 10 or len(baseline_values) < 10:
                continue
            
            recent_mean = np.mean(recent_values)
            baseline_mean = np.mean(baseline_values)
            
            # Check for significant degradation
            if metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score']:
                # Higher is better metrics
                degradation_percent = ((baseline_mean - recent_mean) / baseline_mean) * 100
                if degradation_percent > self.thresholds.max_accuracy_drop_percent:
                    alerts.append({
                        'type': 'accuracy_degradation',
                        'severity': 'critical',
                        'metric': metric_name,
                        'degradation_percent': degradation_percent,
                        'recent_mean': recent_mean,
                        'baseline_mean': baseline_mean,
                        'message': f"Accuracy degradation detected in {metric_name}: {degradation_percent:.1f}% drop"
                    })
            
            elif metric_name in ['mse', 'mae', 'rmse', 'mape']:
                # Lower is better metrics
                degradation_percent = ((recent_mean - baseline_mean) / baseline_mean) * 100
                if degradation_percent > self.thresholds.max_accuracy_drop_percent:
                    alerts.append({
                        'type': 'accuracy_degradation',
                        'severity': 'critical',
                        'metric': metric_name,
                        'degradation_percent': degradation_percent,
                        'recent_mean': recent_mean,
                        'baseline_mean': baseline_mean,
                        'message': f"Error increase detected in {metric_name}: {degradation_percent:.1f}% increase"
                    })
            
            # Statistical significance test
            if self.config.enable_statistical_testing:
                try:
                    t_stat, p_value = stats.ttest_ind(recent_values, baseline_values)
                    if p_value < self.thresholds.significance_level:
                        alerts.append({
                            'type': 'statistical_significance',
                            'severity': 'medium',
                            'metric': metric_name,
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'message': f"Statistically significant change in {metric_name} (p={p_value:.4f})"
                        })
                except:
                    pass
        
        return alerts
    
    def _analyze_accuracy_trends(self, model_id: str) -> Dict[str, Any]:
        """Analyze accuracy trends for the model."""
        if model_id not in self.accuracy_history:
            return {'status': 'no_data'}
        
        history = list(self.accuracy_history[model_id])
        
        if len(history) < 10:
            return {'status': 'insufficient_data'}
        
        trends = {}
        
        # Analyze trends for key metrics
        for metric_name in ['accuracy', 'r2_score', 'mse', 'mae']:
            values = [r.accuracy_metrics.get(metric_name, 0) for r in history[-self.config.analysis_window_size:]]
            
            if len(values) < 10:
                continue
            
            # Calculate trend
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Determine trend direction and significance
            if metric_name in ['accuracy', 'r2_score']:
                trend_direction = 'improving' if slope > 0 else 'degrading'
            else:  # For error metrics
                trend_direction = 'improving' if slope < 0 else 'degrading'
            
            trends[metric_name] = {
                'slope': slope,
                'r_value': r_value,
                'p_value': p_value,
                'trend_direction': trend_direction,
                'is_significant': p_value < 0.05,
                'recent_mean': np.mean(values[-10:]),
                'overall_mean': np.mean(values)
            }
        
        return {
            'status': 'analyzed',
            'trends': trends,
            'analysis_window_size': min(len(history), self.config.analysis_window_size),
            'timestamp': datetime.now().isoformat()
        }
    
    def _compare_with_baseline(self, model_id: str, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare current metrics with baseline."""
        if model_id not in self.baseline_metrics:
            return {'status': 'no_baseline'}
        
        baseline = self.baseline_metrics[model_id]
        comparison = {}
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in baseline:
                baseline_value = baseline[metric_name]
                
                if baseline_value != 0:
                    percent_change = ((current_value - baseline_value) / baseline_value) * 100
                else:
                    percent_change = 0
                
                comparison[metric_name] = {
                    'current_value': current_value,
                    'baseline_value': baseline_value,
                    'percent_change': percent_change,
                    'absolute_change': current_value - baseline_value
                }
        
        return {
            'status': 'compared',
            'comparison': comparison,
            'baseline_timestamp': datetime.now().isoformat()
        }
    
    def _get_applicable_thresholds(self, model_type: str) -> Dict[str, float]:
        """Get applicable thresholds for model type."""
        thresholds = {
            'min_accuracy': self.thresholds.min_accuracy,
            'min_precision': self.thresholds.min_precision,
            'min_recall': self.thresholds.min_recall,
            'min_f1_score': self.thresholds.min_f1_score,
            'min_r2_score': self.thresholds.min_r2_score,
            'max_mse': self.thresholds.max_mse,
            'max_mae': self.thresholds.max_mae,
            'max_mape': self.thresholds.max_mape
        }
        
        # Override with model-specific thresholds
        if model_type in self.thresholds.model_specific_thresholds:
            thresholds.update(self.thresholds.model_specific_thresholds[model_type])
        
        return thresholds
    
    def _get_model_type(self, model_id: str) -> str:
        """Get model type from model ID or history."""
        if model_id in self.accuracy_history and self.accuracy_history[model_id]:
            return self.accuracy_history[model_id][-1].model_type
        return 'unknown'
    
    def _save_metrics(self, record: AccuracyRecord):
        """Save metrics to file."""
        try:
            filename = f"{record.model_id}_{record.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = Path(self.config.metrics_storage_path) / filename
            
            record_dict = {
                'timestamp': record.timestamp.isoformat(),
                'model_id': record.model_id,
                'model_type': record.model_type,
                'accuracy_metrics': record.accuracy_metrics,
                'sample_count': record.sample_count,
                'metadata': record.metadata
            }
            
            with open(filepath, 'w') as f:
                json.dump(record_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def _update_visualizations(self, model_id: str):
        """Update accuracy visualizations."""
        if not self.visualizer or model_id not in self.accuracy_history:
            return
        
        try:
            history = list(self.accuracy_history[model_id])
            
            # Prepare data for visualization
            timestamps = [r.timestamp for r in history]
            metrics_data = {}
            
            for metric_name in ['accuracy', 'r2_score', 'mse', 'mae']:
                values = [r.accuracy_metrics.get(metric_name, 0) for r in history]
                if any(v != 0 for v in values):
                    metrics_data[metric_name] = values
            
            # Update plots
            self.visualizer.update_accuracy_plot(model_id, timestamps, metrics_data)
            
        except Exception as e:
            logger.error(f"Error updating visualizations: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop for real-time tracking."""
        while not self.stop_event.is_set():
            try:
                # Perform periodic analysis
                self._periodic_analysis()
                
                # Update visualizations
                if self.visualizer:
                    self._update_all_visualizations()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Wait for next interval
                self.stop_event.wait(self.config.tracking_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Wait before retrying
    
    def _periodic_analysis(self):
        """Perform periodic analysis of all tracked models."""
        for model_id in self.accuracy_history.keys():
            try:
                # Check for trends
                self._analyze_accuracy_trends(model_id)
                
                # Check for alerts
                if self.accuracy_history[model_id]:
                    latest_record = self.accuracy_history[model_id][-1]
                    self._check_accuracy_alerts(model_id, latest_record.accuracy_metrics)
                
            except Exception as e:
                logger.error(f"Error in periodic analysis for {model_id}: {e}")
    
    def _update_all_visualizations(self):
        """Update visualizations for all models."""
        if not self.visualizer:
            return
        
        for model_id in self.accuracy_history.keys():
            try:
                self._update_visualizations(model_id)
            except Exception as e:
                logger.error(f"Error updating visualization for {model_id}: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old data based on retention policy."""
        cutoff_time = datetime.now() - timedelta(days=self.config.history_retention_days)
        
        for model_id in self.accuracy_history.keys():
            while (self.accuracy_history[model_id] and 
                   self.accuracy_history[model_id][0].timestamp < cutoff_time):
                self.accuracy_history[model_id].popleft()
    
    def get_accuracy_summary(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get accuracy summary for model(s)."""
        if model_id:
            return self._get_model_accuracy_summary(model_id)
        else:
            return self._get_all_models_summary()
    
    def _get_model_accuracy_summary(self, model_id: str) -> Dict[str, Any]:
        """Get accuracy summary for specific model."""
        if model_id not in self.accuracy_history:
            return {'status': 'no_data', 'model_id': model_id}
        
        history = list(self.accuracy_history[model_id])
        
        if not history:
            return {'status': 'no_data', 'model_id': model_id}
        
        latest_record = history[-1]
        
        # Calculate summary statistics
        all_metrics = {}
        for record in history:
            for metric_name, value in record.accuracy_metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        summary_stats = {}
        for metric_name, values in all_metrics.items():
            summary_stats[metric_name] = {
                'current': latest_record.accuracy_metrics.get(metric_name, 0),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return {
            'status': 'success',
            'model_id': model_id,
            'model_type': latest_record.model_type,
            'total_records': len(history),
            'latest_timestamp': latest_record.timestamp.isoformat(),
            'summary_statistics': summary_stats,
            'baseline_metrics': self.baseline_metrics.get(model_id, {}),
            'recent_alerts': [a for a in self.alert_history if a.get('model_id') == model_id][-5:]
        }
    
    def _get_all_models_summary(self) -> Dict[str, Any]:
        """Get summary for all tracked models."""
        summary = {
            'total_models': len(self.accuracy_history),
            'active_models': len([mid for mid, hist in self.accuracy_history.items() if hist]),
            'total_alerts': len(self.alert_history),
            'models': {}
        }
        
        for model_id in self.accuracy_history.keys():
            summary['models'][model_id] = self._get_model_accuracy_summary(model_id)
        
        return summary
    
    def export_accuracy_data(self, model_id: str, filepath: str):
        """Export accuracy data for a model."""
        if model_id not in self.accuracy_history:
            raise ValueError(f"No data found for model {model_id}")
        
        history = list(self.accuracy_history[model_id])
        
        # Convert to exportable format
        export_data = []
        for record in history:
            export_record = {
                'timestamp': record.timestamp.isoformat(),
                'model_id': record.model_id,
                'model_type': record.model_type,
                'sample_count': record.sample_count,
                **record.accuracy_metrics,
                **record.metadata
            }
            export_data.append(export_record)
        
        # Save to file
        df = pd.DataFrame(export_data)
        
        if filepath.endswith('.csv'):
            df.to_csv(filepath, index=False)
        elif filepath.endswith('.json'):
            df.to_json(filepath, orient='records', indent=2)
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")
        
        logger.info(f"Accuracy data exported to {filepath}")

# Factory function
def create_accuracy_tracker(config: Optional[AccuracyTrackerConfig] = None) -> AccuracyTracker:
    """
    Factory function to create an accuracy tracker.
    
    Args:
        config: Tracker configuration
        
    Returns:
        Configured AccuracyTracker instance
    """
    if config is None:
        config = AccuracyTrackerConfig()
    
    thresholds = AccuracyThresholds()
    return AccuracyTracker(thresholds, config)
