"""
BatteryMind - Performance Monitor

Advanced real-time performance monitoring system for battery AI models.
Provides comprehensive monitoring, alerting, and automated response capabilities
for production battery management systems.

Features:
- Real-time model performance tracking
- Multi-dimensional performance metrics
- Automated threshold management
- Performance degradation detection
- Resource utilization monitoring
- Automated alerting and escalation
- Historical performance analysis
- Predictive performance forecasting

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import threading
import time
import queue
import json
import warnings
from pathlib import Path
from collections import deque
from abc import ABC, abstractmethod
import psutil
import gc
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# BatteryMind imports
from ...utils.logging_utils import setup_logger
from ...utils.data_utils import StatisticalAnalyzer, TimeSeriesAnalyzer
from ...utils.config_parser import ConfigParser
from ..alerts.alert_manager import AlertManager
from ..alerts.notification_service import NotificationService

# Configure logging
logger = setup_logger(__name__)

@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    metric_name: str
    threshold_value: float
    comparison_operator: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    severity: str  # 'critical', 'high', 'medium', 'low'
    window_size: int = 10  # Number of consecutive violations before alert
    recovery_threshold: Optional[float] = None

@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    timestamp: datetime
    metric_name: str
    metric_value: float
    model_id: str
    model_type: str
    battery_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Performance alert data structure."""
    alert_id: str
    timestamp: datetime
    severity: str
    metric_name: str
    current_value: float
    threshold_value: float
    model_id: str
    message: str
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None

class PerformanceMonitorConfig:
    """Configuration for performance monitoring."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = ConfigParser.load_config(config_path) if config_path else {}
        
        # Monitoring settings
        self.monitoring_interval_seconds = self.config.get('monitoring_interval_seconds', 30)
        self.metric_retention_hours = self.config.get('metric_retention_hours', 168)  # 1 week
        self.batch_size = self.config.get('batch_size', 100)
        
        # Performance thresholds
        self.performance_thresholds = self._load_performance_thresholds()
        
        # Alert settings
        self.enable_alerting = self.config.get('enable_alerting', True)
        self.alert_cooldown_minutes = self.config.get('alert_cooldown_minutes', 15)
        
        # Resource monitoring
        self.monitor_system_resources = self.config.get('monitor_system_resources', True)
        self.cpu_threshold_percent = self.config.get('cpu_threshold_percent', 85.0)
        self.memory_threshold_percent = self.config.get('memory_threshold_percent', 90.0)
        
        # Prediction settings
        self.enable_performance_prediction = self.config.get('enable_performance_prediction', True)
        self.prediction_horizon_hours = self.config.get('prediction_horizon_hours', 24)
        
    def _load_performance_thresholds(self) -> List[PerformanceThreshold]:
        """Load performance thresholds from configuration."""
        threshold_configs = self.config.get('performance_thresholds', [])
        thresholds = []
        
        # Default thresholds
        default_thresholds = [
            {'metric_name': 'accuracy', 'threshold_value': 0.85, 'comparison_operator': 'lt', 'severity': 'high'},
            {'metric_name': 'mse', 'threshold_value': 0.01, 'comparison_operator': 'gt', 'severity': 'high'},
            {'metric_name': 'inference_time_ms', 'threshold_value': 100.0, 'comparison_operator': 'gt', 'severity': 'medium'},
            {'metric_name': 'memory_usage_mb', 'threshold_value': 1000.0, 'comparison_operator': 'gt', 'severity': 'medium'},
            {'metric_name': 'throughput_qps', 'threshold_value': 50.0, 'comparison_operator': 'lt', 'severity': 'medium'},
        ]
        
        # Use configured thresholds or defaults
        for threshold_config in (threshold_configs or default_thresholds):
            thresholds.append(PerformanceThreshold(**threshold_config))
        
        return thresholds

class MetricCollector(ABC):
    """Abstract base class for metric collectors."""
    
    @abstractmethod
    def collect_metrics(self, model_id: str, model_type: str, **kwargs) -> List[PerformanceMetric]:
        """Collect performance metrics."""
        pass

class AccuracyMetricCollector(MetricCollector):
    """Collector for accuracy-based metrics."""
    
    def collect_metrics(self, model_id: str, model_type: str, 
                       predictions: np.ndarray, ground_truth: np.ndarray,
                       **kwargs) -> List[PerformanceMetric]:
        """Collect accuracy metrics."""
        timestamp = datetime.now()
        metrics = []
        
        try:
            # Calculate accuracy metrics
            mse = mean_squared_error(ground_truth, predictions)
            mae = mean_absolute_error(ground_truth, predictions)
            r2 = r2_score(ground_truth, predictions)
            
            # Create metric objects
            metrics.extend([
                PerformanceMetric(timestamp, 'mse', mse, model_id, model_type),
                PerformanceMetric(timestamp, 'mae', mae, model_id, model_type),
                PerformanceMetric(timestamp, 'r2_score', r2, model_id, model_type)
            ])
            
            # Battery-specific accuracy
            if 'battery_id' in kwargs:
                battery_accuracy = np.mean(np.abs(predictions - ground_truth) < 0.05)
                metrics.append(PerformanceMetric(
                    timestamp, 'battery_accuracy', battery_accuracy, 
                    model_id, model_type, kwargs['battery_id']
                ))
            
        except Exception as e:
            logger.error(f"Error collecting accuracy metrics: {e}")
        
        return metrics

class LatencyMetricCollector(MetricCollector):
    """Collector for latency and performance metrics."""
    
    def collect_metrics(self, model_id: str, model_type: str,
                       inference_times: List[float], **kwargs) -> List[PerformanceMetric]:
        """Collect latency metrics."""
        timestamp = datetime.now()
        metrics = []
        
        try:
            if inference_times:
                avg_latency = np.mean(inference_times)
                p95_latency = np.percentile(inference_times, 95)
                p99_latency = np.percentile(inference_times, 99)
                
                metrics.extend([
                    PerformanceMetric(timestamp, 'avg_inference_time_ms', avg_latency, model_id, model_type),
                    PerformanceMetric(timestamp, 'p95_inference_time_ms', p95_latency, model_id, model_type),
                    PerformanceMetric(timestamp, 'p99_inference_time_ms', p99_latency, model_id, model_type)
                ])
                
                # Throughput calculation
                if 'batch_size' in kwargs and avg_latency > 0:
                    throughput = (kwargs['batch_size'] * 1000) / avg_latency  # QPS
                    metrics.append(PerformanceMetric(
                        timestamp, 'throughput_qps', throughput, model_id, model_type
                    ))
                    
        except Exception as e:
            logger.error(f"Error collecting latency metrics: {e}")
        
        return metrics

class ResourceMetricCollector(MetricCollector):
    """Collector for system resource metrics."""
    
    def collect_metrics(self, model_id: str, model_type: str, **kwargs) -> List[PerformanceMetric]:
        """Collect system resource metrics."""
        timestamp = datetime.now()
        metrics = []
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_mb = (memory.total - memory.available) / 1024 / 1024
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # GPU usage (if available)
            gpu_usage = self._get_gpu_usage()
            
            metrics.extend([
                PerformanceMetric(timestamp, 'cpu_usage_percent', cpu_percent, model_id, model_type),
                PerformanceMetric(timestamp, 'memory_usage_percent', memory_percent, model_id, model_type),
                PerformanceMetric(timestamp, 'memory_usage_mb', memory_mb, model_id, model_type),
                PerformanceMetric(timestamp, 'disk_usage_percent', disk_percent, model_id, model_type)
            ])
            
            if gpu_usage is not None:
                metrics.append(PerformanceMetric(
                    timestamp, 'gpu_usage_percent', gpu_usage, model_id, model_type
                ))
                
        except Exception as e:
            logger.error(f"Error collecting resource metrics: {e}")
        
        return metrics
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage if available."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except ImportError:
            pass
        return None

class PerformanceMonitor:
    """
    Advanced performance monitoring system for battery AI models.
    """
    
    def __init__(self, config: PerformanceMonitorConfig):
        self.config = config
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metric_queue = queue.Queue()
        self.metrics_storage = deque(maxlen=10000)
        self.alerts_history = deque(maxlen=1000)
        self.alert_manager = AlertManager()
        self.notification_service = NotificationService()
        
        # Metric collectors
        self.collectors = {
            'accuracy': AccuracyMetricCollector(),
            'latency': LatencyMetricCollector(),
            'resource': ResourceMetricCollector()
        }
        
        # Performance tracking
        self.performance_baselines = {}
        self.violation_counters = {}
        self.last_alert_times = {}
        
        # Statistics analyzers
        self.stats_analyzer = StatisticalAnalyzer()
        self.ts_analyzer = TimeSeriesAnalyzer()
        
        logger.info("PerformanceMonitor initialized with comprehensive monitoring")
    
    def start_monitoring(self):
        """Start the performance monitoring system."""
        if self.is_monitoring:
            logger.warning("Performance monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the performance monitoring system."""
        self.is_monitoring = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Performance monitoring stopped")
    
    def collect_model_metrics(self, model_id: str, model_type: str,
                            predictions: Optional[np.ndarray] = None,
                            ground_truth: Optional[np.ndarray] = None,
                            inference_times: Optional[List[float]] = None,
                            **kwargs):
        """Collect metrics for a specific model."""
        metrics = []
        
        # Collect accuracy metrics
        if predictions is not None and ground_truth is not None:
            accuracy_metrics = self.collectors['accuracy'].collect_metrics(
                model_id, model_type, predictions, ground_truth, **kwargs
            )
            metrics.extend(accuracy_metrics)
        
        # Collect latency metrics
        if inference_times is not None:
            latency_metrics = self.collectors['latency'].collect_metrics(
                model_id, model_type, inference_times, **kwargs
            )
            metrics.extend(latency_metrics)
        
        # Collect resource metrics
        if self.config.monitor_system_resources:
            resource_metrics = self.collectors['resource'].collect_metrics(
                model_id, model_type, **kwargs
            )
            metrics.extend(resource_metrics)
        
        # Add metrics to queue for processing
        for metric in metrics:
            self.metric_queue.put(metric)
        
        return metrics
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Process queued metrics
                self._process_metric_queue()
                
                # Check thresholds and generate alerts
                self._check_thresholds()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Performance prediction
                if self.config.enable_performance_prediction:
                    self._update_performance_predictions()
                
                # Wait for next iteration
                time.sleep(self.config.monitoring_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _process_metric_queue(self):
        """Process metrics from the queue."""
        processed_count = 0
        
        while not self.metric_queue.empty() and processed_count < self.config.batch_size:
            try:
                metric = self.metric_queue.get_nowait()
                self._store_metric(metric)
                processed_count += 1
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing metric: {e}")
    
    def _store_metric(self, metric: PerformanceMetric):
        """Store metric in local storage."""
        self.metrics_storage.append(metric)
        
        # Update baselines
        self._update_baseline(metric)
    
    def _update_baseline(self, metric: PerformanceMetric):
        """Update performance baselines."""
        key = f"{metric.model_id}_{metric.metric_name}"
        
        if key not in self.performance_baselines:
            self.performance_baselines[key] = {
                'values': deque(maxlen=100),
                'mean': 0.0,
                'std': 0.0,
                'last_updated': metric.timestamp
            }
        
        baseline = self.performance_baselines[key]
        baseline['values'].append(metric.metric_value)
        baseline['last_updated'] = metric.timestamp
        
        # Update statistics
        if len(baseline['values']) >= 10:
            baseline['mean'] = np.mean(baseline['values'])
            baseline['std'] = np.std(baseline['values'])
    
    def _check_thresholds(self):
        """Check performance thresholds and generate alerts."""
        current_time = datetime.now()
        
        for threshold in self.config.performance_thresholds:
            # Get recent metrics for this threshold
            recent_metrics = self._get_recent_metrics(threshold.metric_name, threshold.window_size)
            
            if len(recent_metrics) < threshold.window_size:
                continue
            
            # Check if all recent metrics violate the threshold
            violations = []
            for metric in recent_metrics:
                is_violation = self._check_single_threshold(metric.metric_value, threshold)
                violations.append(is_violation)
            
            # Generate alert if all recent metrics violate threshold
            if all(violations):
                self._generate_threshold_alert(threshold, recent_metrics[-1], current_time)
    
    def _check_single_threshold(self, value: float, threshold: PerformanceThreshold) -> bool:
        """Check if a single value violates the threshold."""
        if threshold.comparison_operator == 'gt':
            return value > threshold.threshold_value
        elif threshold.comparison_operator == 'lt':
            return value < threshold.threshold_value
        elif threshold.comparison_operator == 'gte':
            return value >= threshold.threshold_value
        elif threshold.comparison_operator == 'lte':
            return value <= threshold.threshold_value
        elif threshold.comparison_operator == 'eq':
            return abs(value - threshold.threshold_value) < 1e-6
        else:
            return False
    
    def _get_recent_metrics(self, metric_name: str, count: int) -> List[PerformanceMetric]:
        """Get recent metrics for a specific metric name."""
        relevant_metrics = [
            metric for metric in self.metrics_storage
            if metric.metric_name == metric_name
        ]
        
        # Sort by timestamp and return most recent
        relevant_metrics.sort(key=lambda x: x.timestamp, reverse=True)
        return relevant_metrics[:count]
    
    def _generate_threshold_alert(self, threshold: PerformanceThreshold, 
                                 metric: PerformanceMetric, timestamp: datetime):
        """Generate an alert for threshold violation."""
        alert_key = f"{metric.model_id}_{threshold.metric_name}"
        
        # Check cooldown period
        if alert_key in self.last_alert_times:
            time_since_last = timestamp - self.last_alert_times[alert_key]
            if time_since_last.total_seconds() < self.config.alert_cooldown_minutes * 60:
                return
        
        # Create alert
        alert = PerformanceAlert(
            alert_id=f"perf_{alert_key}_{int(timestamp.timestamp())}",
            timestamp=timestamp,
            severity=threshold.severity,
            metric_name=threshold.metric_name,
            current_value=metric.metric_value,
            threshold_value=threshold.threshold_value,
            model_id=metric.model_id,
            message=f"Performance metric {threshold.metric_name} ({metric.metric_value:.4f}) "
                   f"violated threshold ({threshold.threshold_value:.4f}) for model {metric.model_id}"
        )
        
        # Store alert
        self.alerts_history.append(alert)
        self.last_alert_times[alert_key] = timestamp
        
        # Send notifications
        if self.config.enable_alerting:
            self._send_alert_notification(alert)
        
        logger.warning(f"Performance alert generated: {alert.message}")
    
    def _send_alert_notification(self, alert: PerformanceAlert):
        """Send alert notification."""
        try:
            # Send to alert manager
            self.alert_manager.send_alert(
                alert_type='performance_threshold',
                severity=alert.severity,
                message=alert.message,
                metadata={
                    'model_id': alert.model_id,
                    'metric_name': alert.metric_name,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value
                }
            )
            
            # Send notification
            self.notification_service.send_notification(
                subject=f"Performance Alert - {alert.severity.upper()}",
                message=alert.message,
                severity=alert.severity
            )
            
        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old metrics and alerts."""
        cutoff_time = datetime.now() - timedelta(hours=self.config.metric_retention_hours)
        
        # Clean up metrics
        original_count = len(self.metrics_storage)
        self.metrics_storage = deque(
            [metric for metric in self.metrics_storage if metric.timestamp >= cutoff_time],
            maxlen=self.metrics_storage.maxlen
        )
        
        cleaned_count = original_count - len(self.metrics_storage)
        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} old metrics")
    
    def _update_performance_predictions(self):
        """Update performance predictions."""
        try:
            # Predict performance trends for each metric
            for metric_name in ['accuracy', 'mse', 'inference_time_ms']:
                recent_metrics = self._get_recent_metrics(metric_name, 50)
                
                if len(recent_metrics) >= 20:
                    values = [m.metric_value for m in recent_metrics]
                    timestamps = [m.timestamp for m in recent_metrics]
                    
                    # Predict future performance
                    prediction = self.ts_analyzer.predict_trend(
                        values, timestamps, 
                        horizon_hours=self.config.prediction_horizon_hours
                    )
                    
                    # Store prediction for later use
                    self._store_performance_prediction(metric_name, prediction)
                    
        except Exception as e:
            logger.error(f"Error updating performance predictions: {e}")
    
    def _store_performance_prediction(self, metric_name: str, prediction: Dict[str, Any]):
        """Store performance prediction."""
        # This would typically be stored in a database or cache
        # For now, we'll log the prediction
        logger.info(f"Performance prediction for {metric_name}: {prediction}")
    
    def get_performance_summary(self, model_id: Optional[str] = None, 
                              hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the specified period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter metrics
        if model_id:
            filtered_metrics = [
                m for m in self.metrics_storage 
                if m.model_id == model_id and m.timestamp >= cutoff_time
            ]
        else:
            filtered_metrics = [
                m for m in self.metrics_storage 
                if m.timestamp >= cutoff_time
            ]
        
        if not filtered_metrics:
            return {'status': 'no_data'}
        
        # Group by metric name
        metrics_by_name = {}
        for metric in filtered_metrics:
            if metric.metric_name not in metrics_by_name:
                metrics_by_name[metric.metric_name] = []
            metrics_by_name[metric.metric_name].append(metric.metric_value)
        
        # Calculate summary statistics
        summary = {
            'period_hours': hours,
            'total_metrics': len(filtered_metrics),
            'unique_metrics': len(metrics_by_name),
            'model_id': model_id,
            'generated_at': datetime.now().isoformat(),
            'metric_summaries': {}
        }
        
        for metric_name, values in metrics_by_name.items():
            summary['metric_summaries'][metric_name] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }
        
        # Add alert summary
        recent_alerts = [
            alert for alert in self.alerts_history
            if alert.timestamp >= cutoff_time and (not model_id or alert.model_id == model_id)
        ]
        
        summary['alerts'] = {
            'total_alerts': len(recent_alerts),
            'critical_alerts': len([a for a in recent_alerts if a.severity == 'critical']),
            'high_alerts': len([a for a in recent_alerts if a.severity == 'high']),
            'medium_alerts': len([a for a in recent_alerts if a.severity == 'medium']),
            'low_alerts': len([a for a in recent_alerts if a.severity == 'low'])
        }
        
        return summary
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export performance metrics to file."""
        try:
            if format == 'json':
                metrics_data = [
                    {
                        'timestamp': metric.timestamp.isoformat(),
                        'metric_name': metric.metric_name,
                        'metric_value': metric.metric_value,
                        'model_id': metric.model_id,
                        'model_type': metric.model_type,
                        'battery_id': metric.battery_id,
                        'metadata': metric.metadata
                    }
                    for metric in self.metrics_storage
                ]
                
                with open(filepath, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
                    
            elif format == 'csv':
                df = pd.DataFrame([
                    {
                        'timestamp': metric.timestamp,
                        'metric_name': metric.metric_name,
                        'metric_value': metric.metric_value,
                        'model_id': metric.model_id,
                        'model_type': metric.model_type,
                        'battery_id': metric.battery_id
                    }
                    for metric in self.metrics_storage
                ])
                df.to_csv(filepath, index=False)
            
            logger.info(f"Performance metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def get_health_status(self) -> Dict[str, str]:
        """Get health status of the monitoring system."""
        return {
            'monitoring_active': 'healthy' if self.is_monitoring else 'inactive',
            'metrics_count': str(len(self.metrics_storage)),
            'alerts_count': str(len(self.alerts_history)),
            'last_metric_time': (
                self.metrics_storage[-1].timestamp.isoformat() 
                if self.metrics_storage else 'none'
            ),
            'queue_size': str(self.metric_queue.qsize()),
            'collectors_active': str(len(self.collectors))
        }

# Factory function
def create_performance_monitor(config_path: Optional[str] = None) -> PerformanceMonitor:
    """Create a performance monitor instance."""
    config = PerformanceMonitorConfig(config_path)
    return PerformanceMonitor(config)

# Usage example
if __name__ == "__main__":
    # Create and start performance monitor
    monitor = create_performance_monitor()
    monitor.start_monitoring()
    
    # Simulate some metrics collection
    predictions = np.random.normal(0.8, 0.1, 100)
    ground_truth = np.random.normal(0.8, 0.05, 100)
    inference_times = np.random.normal(50, 10, 100).tolist()
    
    # Collect metrics
    monitor.collect_model_metrics(
        model_id="transformer_v1",
        model_type="transformer",
        predictions=predictions,
        ground_truth=ground_truth,
        inference_times=inference_times
    )
    
    # Wait and get summary
    time.sleep(2)
    summary = monitor.get_performance_summary()
    print("Performance Summary:", json.dumps(summary, indent=2))
    
    # Stop monitoring
    monitor.stop_monitoring()
