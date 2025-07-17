"""
BatteryMind - Performance Validator

Advanced performance validation system for battery AI models, providing comprehensive
performance monitoring, threshold validation, and automated performance degradation
detection across all model types in the BatteryMind ecosystem.

Features:
- Real-time performance monitoring and validation
- Multi-metric performance assessment
- Automated threshold management and alerting
- Cross-model performance comparison
- Time-series performance tracking
- Automated performance regression detection
- Statistical significance testing for performance changes

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
    confusion_matrix, classification_report
)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# BatteryMind imports
from . import BaseValidator
from ..metrics.accuracy_metrics import AccuracyMetrics
from ..metrics.performance_metrics import PerformanceMetrics
from ..metrics.efficiency_metrics import EfficiencyMetrics
from ..metrics.business_metrics import BusinessMetrics
from ...utils.logging_utils import setup_logger
from ...utils.data_utils import StatisticalAnalyzer
from ...utils.visualization import PerformanceVisualizer

# Configure logging
logger = setup_logger(__name__)

@dataclass
class PerformanceThresholds:
    """
    Performance thresholds configuration for different metrics and model types.
    
    Attributes:
        # Accuracy thresholds
        min_accuracy (float): Minimum acceptable accuracy
        min_precision (float): Minimum acceptable precision
        min_recall (float): Minimum acceptable recall
        min_f1_score (float): Minimum acceptable F1 score
        
        # Error thresholds
        max_mse (float): Maximum acceptable MSE
        max_mae (float): Maximum acceptable MAE
        min_r2_score (float): Minimum acceptable R² score
        
        # Performance thresholds
        max_inference_time_ms (float): Maximum acceptable inference time
        max_memory_usage_mb (float): Maximum acceptable memory usage
        min_throughput_per_second (float): Minimum acceptable throughput
        
        # Degradation thresholds
        max_accuracy_degradation (float): Maximum acceptable accuracy degradation
        max_latency_degradation (float): Maximum acceptable latency degradation
        
        # Business metrics thresholds
        min_efficiency_score (float): Minimum acceptable efficiency score
        max_cost_per_prediction (float): Maximum acceptable cost per prediction
        
        # Model-specific thresholds
        model_specific_thresholds (Dict[str, Dict[str, float]]): Model-specific thresholds
    """
    # Accuracy thresholds
    min_accuracy: float = 0.85
    min_precision: float = 0.80
    min_recall: float = 0.80
    min_f1_score: float = 0.80
    
    # Error thresholds
    max_mse: float = 0.01
    max_mae: float = 0.05
    min_r2_score: float = 0.85
    
    # Performance thresholds
    max_inference_time_ms: float = 100.0
    max_memory_usage_mb: float = 512.0
    min_throughput_per_second: float = 100.0
    
    # Degradation thresholds
    max_accuracy_degradation: float = 0.05
    max_latency_degradation: float = 0.20
    
    # Business metrics thresholds
    min_efficiency_score: float = 0.80
    max_cost_per_prediction: float = 0.01
    
    # Model-specific thresholds
    model_specific_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'transformer': {
            'min_accuracy': 0.90,
            'max_inference_time_ms': 50.0,
            'max_memory_usage_mb': 1024.0
        },
        'federated': {
            'min_accuracy': 0.85,
            'max_inference_time_ms': 200.0,
            'max_memory_usage_mb': 256.0
        },
        'rl_agent': {
            'min_success_rate': 0.85,
            'max_inference_time_ms': 20.0,
            'max_memory_usage_mb': 128.0
        },
        'ensemble': {
            'min_accuracy': 0.92,
            'max_inference_time_ms': 150.0,
            'max_memory_usage_mb': 2048.0
        }
    })

@dataclass
class PerformanceValidationConfig:
    """
    Configuration for performance validation system.
    
    Attributes:
        # Validation settings
        validation_interval_minutes (int): Validation interval in minutes
        baseline_window_days (int): Baseline performance window in days
        performance_history_days (int): Performance history retention in days
        
        # Alerting settings
        enable_alerting (bool): Enable performance alerting
        alert_email_recipients (List[str]): Email recipients for alerts
        alert_slack_webhook (str): Slack webhook for alerts
        
        # Statistical settings
        confidence_level (float): Confidence level for statistical tests
        significance_threshold (float): Significance threshold for changes
        bootstrap_samples (int): Number of bootstrap samples for confidence intervals
        
        # Monitoring settings
        enable_continuous_monitoring (bool): Enable continuous performance monitoring
        monitoring_frequency_seconds (int): Monitoring frequency in seconds
        store_detailed_metrics (bool): Store detailed performance metrics
        
        # Regression detection
        enable_regression_detection (bool): Enable automated regression detection
        regression_detection_window_hours (int): Window for regression detection
        regression_significance_threshold (float): Significance threshold for regression
    """
    # Validation settings
    validation_interval_minutes: int = 15
    baseline_window_days: int = 7
    performance_history_days: int = 30
    
    # Alerting settings
    enable_alerting: bool = True
    alert_email_recipients: List[str] = field(default_factory=list)
    alert_slack_webhook: str = ""
    
    # Statistical settings
    confidence_level: float = 0.95
    significance_threshold: float = 0.05
    bootstrap_samples: int = 1000
    
    # Monitoring settings
    enable_continuous_monitoring: bool = True
    monitoring_frequency_seconds: int = 60
    store_detailed_metrics: bool = True
    
    # Regression detection
    enable_regression_detection: bool = True
    regression_detection_window_hours: int = 24
    regression_significance_threshold: float = 0.01

class PerformanceValidator(BaseValidator):
    """
    Advanced performance validator with comprehensive monitoring and alerting capabilities.
    """
    
    def __init__(self, 
                 thresholds: PerformanceThresholds,
                 config: PerformanceValidationConfig):
        super().__init__()
        self.thresholds = thresholds
        self.config = config
        self.performance_history = []
        self.baseline_metrics = {}
        self.alert_history = []
        
        # Initialize metrics calculators
        self.accuracy_metrics = AccuracyMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.efficiency_metrics = EfficiencyMetrics()
        self.business_metrics = BusinessMetrics()
        
        # Initialize analyzers
        self.statistical_analyzer = StatisticalAnalyzer()
        self.performance_visualizer = PerformanceVisualizer()
        
        # Performance tracking
        self.current_metrics = {}
        self.performance_trends = {}
        
        logger.info("PerformanceValidator initialized with comprehensive monitoring")
    
    def validate_model_performance(self, 
                                 model_predictions: Dict[str, np.ndarray],
                                 ground_truth: Dict[str, np.ndarray],
                                 model_type: str,
                                 model_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate model performance against established thresholds.
        
        Args:
            model_predictions: Dictionary of model predictions
            ground_truth: Dictionary of ground truth values
            model_type: Type of model being validated
            model_metadata: Additional model metadata
            
        Returns:
            Dictionary containing validation results
        """
        validation_start_time = datetime.now()
        
        try:
            # Calculate comprehensive performance metrics
            metrics = self._calculate_comprehensive_metrics(
                model_predictions, ground_truth, model_type
            )
            
            # Get applicable thresholds
            applicable_thresholds = self._get_applicable_thresholds(model_type)
            
            # Validate against thresholds
            validation_results = self._validate_against_thresholds(metrics, applicable_thresholds)
            
            # Perform statistical significance testing
            statistical_results = self._perform_statistical_testing(metrics, model_type)
            
            # Check for performance regression
            regression_results = self._check_performance_regression(metrics, model_type)
            
            # Update performance history
            self._update_performance_history(metrics, model_type, validation_start_time)
            
            # Generate alerts if necessary
            alerts = self._generate_performance_alerts(validation_results, regression_results)
            
            # Compile final results
            final_results = {
                'validation_timestamp': validation_start_time.isoformat(),
                'model_type': model_type,
                'performance_metrics': metrics,
                'threshold_validation': validation_results,
                'statistical_analysis': statistical_results,
                'regression_analysis': regression_results,
                'alerts': alerts,
                'overall_status': self._determine_overall_status(validation_results, regression_results),
                'recommendations': self._generate_recommendations(validation_results, regression_results),
                'metadata': model_metadata or {}
            }
            
            # Log validation results
            self._log_validation_results(final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            return {
                'validation_timestamp': validation_start_time.isoformat(),
                'model_type': model_type,
                'error': str(e),
                'overall_status': 'ERROR'
            }
    
    def _calculate_comprehensive_metrics(self, 
                                       predictions: Dict[str, np.ndarray],
                                       ground_truth: Dict[str, np.ndarray],
                                       model_type: str) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        # Accuracy metrics
        try:
            accuracy_results = self.accuracy_metrics.calculate_metrics(predictions, ground_truth)
            metrics['accuracy'] = accuracy_results
        except Exception as e:
            logger.warning(f"Failed to calculate accuracy metrics: {e}")
            metrics['accuracy'] = {}
        
        # Performance metrics
        try:
            performance_results = self.performance_metrics.calculate_metrics(predictions, ground_truth)
            metrics['performance'] = performance_results
        except Exception as e:
            logger.warning(f"Failed to calculate performance metrics: {e}")
            metrics['performance'] = {}
        
        # Efficiency metrics
        try:
            efficiency_results = self.efficiency_metrics.calculate_metrics(predictions, ground_truth)
            metrics['efficiency'] = efficiency_results
        except Exception as e:
            logger.warning(f"Failed to calculate efficiency metrics: {e}")
            metrics['efficiency'] = {}
        
        # Business metrics
        try:
            business_results = self.business_metrics.calculate_metrics(predictions, ground_truth)
            metrics['business'] = business_results
        except Exception as e:
            logger.warning(f"Failed to calculate business metrics: {e}")
            metrics['business'] = {}
        
        # Model-specific metrics
        if model_type == 'transformer':
            metrics['transformer_specific'] = self._calculate_transformer_metrics(predictions, ground_truth)
        elif model_type == 'federated':
            metrics['federated_specific'] = self._calculate_federated_metrics(predictions, ground_truth)
        elif model_type == 'rl_agent':
            metrics['rl_specific'] = self._calculate_rl_metrics(predictions, ground_truth)
        elif model_type == 'ensemble':
            metrics['ensemble_specific'] = self._calculate_ensemble_metrics(predictions, ground_truth)
        
        return metrics
    
    def _calculate_transformer_metrics(self, 
                                     predictions: Dict[str, np.ndarray],
                                     ground_truth: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate transformer-specific metrics."""
        metrics = {}
        
        # Attention quality metrics
        if 'attention_weights' in predictions:
            metrics['attention_entropy'] = self._calculate_attention_entropy(predictions['attention_weights'])
            metrics['attention_diversity'] = self._calculate_attention_diversity(predictions['attention_weights'])
        
        # Prediction consistency
        if 'sequence_predictions' in predictions:
            metrics['prediction_consistency'] = self._calculate_prediction_consistency(
                predictions['sequence_predictions']
            )
        
        # Confidence calibration
        if 'confidence_scores' in predictions:
            metrics['confidence_calibration'] = self._calculate_confidence_calibration(
                predictions['confidence_scores'], ground_truth.get('labels', np.array([]))
            )
        
        return metrics
    
    def _calculate_federated_metrics(self, 
                                   predictions: Dict[str, np.ndarray],
                                   ground_truth: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate federated learning-specific metrics."""
        metrics = {}
        
        # Client diversity metrics
        if 'client_predictions' in predictions:
            metrics['client_diversity'] = self._calculate_client_diversity(predictions['client_predictions'])
            metrics['client_agreement'] = self._calculate_client_agreement(predictions['client_predictions'])
        
        # Privacy preservation metrics
        if 'privacy_metrics' in predictions:
            metrics['privacy_budget_usage'] = predictions['privacy_metrics'].get('budget_usage', 0.0)
            metrics['noise_impact'] = predictions['privacy_metrics'].get('noise_impact', 0.0)
        
        return metrics
    
    def _calculate_rl_metrics(self, 
                            predictions: Dict[str, np.ndarray],
                            ground_truth: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate reinforcement learning-specific metrics."""
        metrics = {}
        
        # Reward-based metrics
        if 'episode_rewards' in predictions:
            rewards = predictions['episode_rewards']
            metrics['average_reward'] = np.mean(rewards)
            metrics['reward_stability'] = 1.0 / (1.0 + np.std(rewards))
            metrics['success_rate'] = np.mean(rewards > 0)
        
        # Action quality metrics
        if 'action_values' in predictions:
            metrics['action_value_consistency'] = self._calculate_action_value_consistency(
                predictions['action_values']
            )
        
        # Exploration vs exploitation
        if 'exploration_ratio' in predictions:
            metrics['exploration_ratio'] = predictions['exploration_ratio']
        
        return metrics
    
    def _calculate_ensemble_metrics(self, 
                                  predictions: Dict[str, np.ndarray],
                                  ground_truth: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate ensemble-specific metrics."""
        metrics = {}
        
        # Diversity metrics
        if 'base_model_predictions' in predictions:
            base_predictions = predictions['base_model_predictions']
            metrics['ensemble_diversity'] = self._calculate_ensemble_diversity(base_predictions)
            metrics['prediction_agreement'] = self._calculate_prediction_agreement(base_predictions)
        
        # Ensemble confidence
        if 'ensemble_confidence' in predictions:
            metrics['mean_confidence'] = np.mean(predictions['ensemble_confidence'])
            metrics['confidence_spread'] = np.std(predictions['ensemble_confidence'])
        
        return metrics
    
    def _get_applicable_thresholds(self, model_type: str) -> Dict[str, float]:
        """Get applicable thresholds for the given model type."""
        # Start with general thresholds
        thresholds = {
            'min_accuracy': self.thresholds.min_accuracy,
            'min_precision': self.thresholds.min_precision,
            'min_recall': self.thresholds.min_recall,
            'min_f1_score': self.thresholds.min_f1_score,
            'max_mse': self.thresholds.max_mse,
            'max_mae': self.thresholds.max_mae,
            'min_r2_score': self.thresholds.min_r2_score,
            'max_inference_time_ms': self.thresholds.max_inference_time_ms,
            'max_memory_usage_mb': self.thresholds.max_memory_usage_mb,
            'min_throughput_per_second': self.thresholds.min_throughput_per_second
        }
        
        # Override with model-specific thresholds
        if model_type in self.thresholds.model_specific_thresholds:
            thresholds.update(self.thresholds.model_specific_thresholds[model_type])
        
        return thresholds
    
    def _validate_against_thresholds(self, 
                                   metrics: Dict[str, Any],
                                   thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Validate metrics against thresholds."""
        validation_results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
        
        for threshold_name, threshold_value in thresholds.items():
            # Extract metric value from nested structure
            metric_value = self._extract_metric_value(metrics, threshold_name)
            
            if metric_value is None:
                validation_results['warnings'].append(f"Metric {threshold_name} not found in results")
                continue
            
            # Check threshold
            if threshold_name.startswith('min_'):
                if metric_value >= threshold_value:
                    validation_results['passed'].append({
                        'metric': threshold_name,
                        'value': metric_value,
                        'threshold': threshold_value,
                        'status': 'PASS'
                    })
                else:
                    validation_results['failed'].append({
                        'metric': threshold_name,
                        'value': metric_value,
                        'threshold': threshold_value,
                        'status': 'FAIL',
                        'deviation': threshold_value - metric_value
                    })
            elif threshold_name.startswith('max_'):
                if metric_value <= threshold_value:
                    validation_results['passed'].append({
                        'metric': threshold_name,
                        'value': metric_value,
                        'threshold': threshold_value,
                        'status': 'PASS'
                    })
                else:
                    validation_results['failed'].append({
                        'metric': threshold_name,
                        'value': metric_value,
                        'threshold': threshold_value,
                        'status': 'FAIL',
                        'deviation': metric_value - threshold_value
                    })
        
        return validation_results
    
    def _extract_metric_value(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract metric value from nested metrics structure."""
        # Define metric paths
        metric_paths = {
            'min_accuracy': ['accuracy', 'accuracy'],
            'min_precision': ['accuracy', 'precision'],
            'min_recall': ['accuracy', 'recall'],
            'min_f1_score': ['accuracy', 'f1_score'],
            'max_mse': ['accuracy', 'mse'],
            'max_mae': ['accuracy', 'mae'],
            'min_r2_score': ['accuracy', 'r2_score'],
            'max_inference_time_ms': ['performance', 'inference_time_ms'],
            'max_memory_usage_mb': ['performance', 'memory_usage_mb'],
            'min_throughput_per_second': ['performance', 'throughput_per_second']
        }
        
        if metric_name in metric_paths:
            path = metric_paths[metric_name]
            try:
                value = metrics
                for key in path:
                    value = value[key]
                return float(value)
            except (KeyError, TypeError, ValueError):
                return None
        
        return None
    
    def _perform_statistical_testing(self, 
                                   metrics: Dict[str, Any],
                                   model_type: str) -> Dict[str, Any]:
        """Perform statistical significance testing."""
        if not self.performance_history:
            return {'status': 'insufficient_data', 'message': 'No historical data for comparison'}
        
        # Get historical metrics for comparison
        historical_metrics = [entry['metrics'] for entry in self.performance_history 
                            if entry['model_type'] == model_type]
        
        if len(historical_metrics) < 10:
            return {'status': 'insufficient_data', 'message': 'Insufficient historical data'}
        
        statistical_results = {}
        
        # Compare current metrics with historical baseline
        for metric_category, metric_values in metrics.items():
            if isinstance(metric_values, dict):
                for metric_name, current_value in metric_values.items():
                    if isinstance(current_value, (int, float)):
                        historical_values = [
                            hist_metrics.get(metric_category, {}).get(metric_name, 0)
                            for hist_metrics in historical_metrics
                        ]
                        
                        if len(historical_values) >= 10:
                            # Perform t-test
                            t_stat, p_value = stats.ttest_1samp(historical_values, current_value)
                            
                            statistical_results[f"{metric_category}_{metric_name}"] = {
                                'current_value': current_value,
                                'historical_mean': np.mean(historical_values),
                                'historical_std': np.std(historical_values),
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'is_significant': p_value < self.config.significance_threshold,
                                'change_direction': 'improvement' if current_value > np.mean(historical_values) else 'degradation'
                            }
        
        return statistical_results
    
    def _check_performance_regression(self, 
                                    metrics: Dict[str, Any],
                                    model_type: str) -> Dict[str, Any]:
        """Check for performance regression."""
        if not self.config.enable_regression_detection:
            return {'status': 'disabled'}
        
        # Get recent performance data
        cutoff_time = datetime.now() - timedelta(hours=self.config.regression_detection_window_hours)
        recent_history = [
            entry for entry in self.performance_history
            if entry['model_type'] == model_type and entry['timestamp'] >= cutoff_time
        ]
        
        if len(recent_history) < 5:
            return {'status': 'insufficient_data'}
        
        regression_results = {}
        
        # Check for performance trends
        for metric_category, metric_values in metrics.items():
            if isinstance(metric_values, dict):
                for metric_name, current_value in metric_values.items():
                    if isinstance(current_value, (int, float)):
                        recent_values = [
                            entry['metrics'].get(metric_category, {}).get(metric_name, 0)
                            for entry in recent_history
                        ]
                        
                        if len(recent_values) >= 5:
                            # Calculate trend
                            x = np.arange(len(recent_values))
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_values)
                            
                            # Determine if there's a significant regression
                            is_regression = (
                                p_value < self.config.regression_significance_threshold and
                                ((metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score'] and slope < 0) or
                                 (metric_name in ['mse', 'mae', 'inference_time_ms', 'memory_usage_mb'] and slope > 0))
                            )
                            
                            regression_results[f"{metric_category}_{metric_name}"] = {
                                'slope': slope,
                                'r_value': r_value,
                                'p_value': p_value,
                                'is_regression': is_regression,
                                'trend_direction': 'decreasing' if slope < 0 else 'increasing',
                                'recent_values': recent_values
                            }
        
        return regression_results
    
    def _update_performance_history(self, 
                                  metrics: Dict[str, Any],
                                  model_type: str,
                                  timestamp: datetime):
        """Update performance history."""
        history_entry = {
            'timestamp': timestamp,
            'model_type': model_type,
            'metrics': metrics
        }
        
        self.performance_history.append(history_entry)
        
        # Cleanup old entries
        cutoff_time = datetime.now() - timedelta(days=self.config.performance_history_days)
        self.performance_history = [
            entry for entry in self.performance_history
            if entry['timestamp'] >= cutoff_time
        ]
    
    def _generate_performance_alerts(self, 
                                   validation_results: Dict[str, Any],
                                   regression_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate performance alerts."""
        alerts = []
        
        # Threshold violation alerts
        for failed_metric in validation_results['failed']:
            alert = {
                'type': 'threshold_violation',
                'severity': 'high' if failed_metric['deviation'] > 0.1 else 'medium',
                'metric': failed_metric['metric'],
                'current_value': failed_metric['value'],
                'threshold': failed_metric['threshold'],
                'deviation': failed_metric['deviation'],
                'timestamp': datetime.now().isoformat(),
                'message': f"Performance metric {failed_metric['metric']} ({failed_metric['value']:.4f}) "
                          f"violated threshold ({failed_metric['threshold']:.4f})"
            }
            alerts.append(alert)
        
        # Regression alerts
        for metric_name, regression_data in regression_results.items():
            if regression_data.get('is_regression', False):
                alert = {
                    'type': 'performance_regression',
                    'severity': 'high',
                    'metric': metric_name,
                    'slope': regression_data['slope'],
                    'p_value': regression_data['p_value'],
                    'timestamp': datetime.now().isoformat(),
                    'message': f"Performance regression detected in {metric_name} "
                              f"(slope: {regression_data['slope']:.6f}, p-value: {regression_data['p_value']:.6f})"
                }
                alerts.append(alert)
        
        # Store alerts in history
        self.alert_history.extend(alerts)
        
        return alerts
    
    def _determine_overall_status(self, 
                                validation_results: Dict[str, Any],
                                regression_results: Dict[str, Any]) -> str:
        """Determine overall validation status."""
        if validation_results['failed']:
            return 'FAILED'
        
        # Check for regressions
        regressions = [result for result in regression_results.values() 
                      if isinstance(result, dict) and result.get('is_regression', False)]
        
        if regressions:
            return 'DEGRADED'
        
        if validation_results['warnings']:
            return 'WARNING'
        
        return 'PASSED'
    
    def _generate_recommendations(self, 
                                validation_results: Dict[str, Any],
                                regression_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Threshold violation recommendations
        for failed_metric in validation_results['failed']:
            metric_name = failed_metric['metric']
            
            if 'accuracy' in metric_name:
                recommendations.append(f"Retrain model to improve {metric_name}")
            elif 'inference_time' in metric_name:
                recommendations.append(f"Optimize model or infrastructure to reduce {metric_name}")
            elif 'memory_usage' in metric_name:
                recommendations.append(f"Implement model compression to reduce {metric_name}")
        
        # Regression recommendations
        regression_count = len([r for r in regression_results.values() 
                              if isinstance(r, dict) and r.get('is_regression', False)])
        
        if regression_count > 0:
            recommendations.append("Investigate recent changes that may have caused performance regression")
            recommendations.append("Consider rolling back to previous model version")
            recommendations.append("Implement more frequent performance monitoring")
        
        return recommendations
    
    def _log_validation_results(self, results: Dict[str, Any]):
        """Log validation results."""
        status = results['overall_status']
        model_type = results['model_type']
        
        if status == 'FAILED':
            logger.error(f"Performance validation FAILED for {model_type}")
        elif status == 'DEGRADED':
            logger.warning(f"Performance degradation detected for {model_type}")
        elif status == 'WARNING':
            logger.warning(f"Performance validation warnings for {model_type}")
        else:
            logger.info(f"Performance validation PASSED for {model_type}")
    
    def get_performance_summary(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary."""
        if model_type:
            history = [entry for entry in self.performance_history 
                      if entry['model_type'] == model_type]
        else:
            history = self.performance_history
        
        if not history:
            return {'status': 'no_data'}
        
        # Calculate summary statistics
        recent_entry = history[-1] if history else None
        
        return {
            'total_validations': len(history),
            'most_recent_validation': recent_entry['timestamp'] if recent_entry else None,
            'recent_status': self._determine_overall_status(
                {'failed': [], 'warnings': []}, {}
            ) if recent_entry else 'unknown',
            'alert_count': len(self.alert_history),
            'performance_trends': self._calculate_performance_trends(history)
        }
    
    def _calculate_performance_trends(self, history: List[Dict[str, Any]]) -> Dict[str, str]:
        """Calculate performance trends."""
        if len(history) < 2:
            return {}
        
        trends = {}
        
        # Get first and last entries
        first_entry = history[0]
        last_entry = history[-1]
        
        # Compare key metrics
        key_metrics = ['accuracy', 'mse', 'inference_time_ms']
        
        for metric in key_metrics:
            first_value = self._extract_metric_value(first_entry['metrics'], f"min_{metric}")
            last_value = self._extract_metric_value(last_entry['metrics'], f"min_{metric}")
            
            if first_value is not None and last_value is not None:
                if metric in ['accuracy', 'r2_score']:
                    trend = 'improving' if last_value > first_value else 'degrading'
                else:
                    trend = 'improving' if last_value < first_value else 'degrading'
                
                trends[metric] = trend
        
        return trends
    
    def export_performance_report(self, filepath: str):
        """Export comprehensive performance report."""
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'performance_history': self.performance_history,
            'alert_history': self.alert_history,
            'current_thresholds': self.thresholds.__dict__,
            'configuration': self.config.__dict__,
            'summary': self.get_performance_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report exported to {filepath}")
