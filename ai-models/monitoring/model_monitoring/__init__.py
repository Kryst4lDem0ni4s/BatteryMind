"""
BatteryMind - Model Monitoring Module

Comprehensive model monitoring system for battery AI/ML models providing
real-time performance tracking, drift detection, accuracy monitoring,
and resource utilization analysis.

This module implements:
- Real-time model performance monitoring
- Model drift detection using statistical tests
- Accuracy tracking across different model types
- Resource utilization monitoring (CPU, memory, GPU)
- Automated alerting for performance degradation
- Model version comparison and A/B testing support
- Predictive performance analytics

Key Components:
- PerformanceMonitor: Track inference speed, throughput, and response times
- DriftDetector: Detect data and model drift using statistical methods
- AccuracyTracker: Monitor prediction accuracy and quality metrics
- ResourceMonitor: Track computational resource usage

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
import time
import warnings
from pathlib import Path

# Statistical and ML libraries
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import psutil
import torch

# BatteryMind imports
from ..metrics.accuracy_metrics import AccuracyMetrics
from ..metrics.performance_metrics import PerformanceMetrics
from ...utils.logging_utils import setup_logger

logger = setup_logger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"

class DriftType(Enum):
    """Types of drift detection."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"

class ModelState(Enum):
    """Model operational states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DRIFTED = "drifted"
    FAILED = "failed"
    UNKNOWN = "unknown"

@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    model_id: str
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time_ms: float
    memory_usage_mb: float
    throughput_qps: float
    error_rate: float
    drift_score: float
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'model_id': self.model_id,
            'timestamp': self.timestamp.isoformat(),
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'inference_time_ms': self.inference_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'throughput_qps': self.throughput_qps,
            'error_rate': self.error_rate,
            'drift_score': self.drift_score,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata
        }

@dataclass
class DriftAlert:
    """Drift detection alert."""
    model_id: str
    drift_type: DriftType
    severity: str
    score: float
    threshold: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'model_id': self.model_id,
            'drift_type': self.drift_type.value,
            'severity': self.severity,
            'score': self.score,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details
        }

# Import monitoring components with graceful fallback
try:
    from .performance_monitor import PerformanceMonitor
    logger.info("✓ PerformanceMonitor imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import PerformanceMonitor: {e}")
    class PerformanceMonitor:
        def __init__(self, *args, **kwargs):
            logger.warning("PerformanceMonitor placeholder initialized")
            self.config = kwargs.get('config', {})
            self.metrics_history = []
        
        def monitor_model(self, model_id: str, predictions: np.ndarray, 
                         actual: np.ndarray = None, **kwargs) -> ModelMetrics:
            """Placeholder monitoring method."""
            return ModelMetrics(
                model_id=model_id,
                timestamp=datetime.now(),
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                inference_time_ms=0.0,
                memory_usage_mb=0.0,
                throughput_qps=0.0,
                error_rate=0.0,
                drift_score=0.0,
                confidence_score=0.0
            )

try:
    from .drift_detector import DriftDetector
    logger.info("✓ DriftDetector imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import DriftDetector: {e}")
    class DriftDetector:
        def __init__(self, *args, **kwargs):
            logger.warning("DriftDetector placeholder initialized")
            self.config = kwargs.get('config', {})
            self.reference_data = None
        
        def detect_drift(self, current_data: np.ndarray, 
                        reference_data: np.ndarray = None) -> Dict[str, Any]:
            """Placeholder drift detection method."""
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'drift_type': None,
                'p_value': 1.0,
                'threshold': 0.05
            }

try:
    from .accuracy_tracker import AccuracyTracker
    logger.info("✓ AccuracyTracker imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import AccuracyTracker: {e}")
    class AccuracyTracker:
        def __init__(self, *args, **kwargs):
            logger.warning("AccuracyTracker placeholder initialized")
            self.config = kwargs.get('config', {})
            self.accuracy_history = []
        
        def track_accuracy(self, model_id: str, predictions: np.ndarray,
                          actual: np.ndarray, **kwargs) -> Dict[str, float]:
            """Placeholder accuracy tracking method."""
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }

try:
    from .resource_monitor import ResourceMonitor
    logger.info("✓ ResourceMonitor imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import ResourceMonitor: {e}")
    class ResourceMonitor:
        def __init__(self, *args, **kwargs):
            logger.warning("ResourceMonitor placeholder initialized")
            self.config = kwargs.get('config', {})
            self.resource_history = []
        
        def monitor_resources(self, model_id: str = None) -> Dict[str, float]:
            """Placeholder resource monitoring method."""
            return {
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'gpu_memory_mb': 0.0,
                'inference_time_ms': 0.0
            }

class ModelMonitoringConfig:
    """Configuration for model monitoring."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize monitoring configuration."""
        self.config = config or self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            'performance': {
                'accuracy_threshold': 0.85,
                'latency_threshold_ms': 100,
                'memory_threshold_mb': 1000,
                'error_rate_threshold': 0.01,
                'throughput_threshold_qps': 100
            },
            'drift_detection': {
                'enabled': True,
                'method': 'ks_test',
                'threshold': 0.05,
                'window_size': 1000,
                'detection_interval': 300  # seconds
            },
            'accuracy_tracking': {
                'enabled': True,
                'track_confidence': True,
                'track_feature_importance': True,
                'evaluation_interval': 60  # seconds
            },
            'resource_monitoring': {
                'enabled': True,
                'track_cpu': True,
                'track_memory': True,
                'track_gpu': True,
                'monitoring_interval': 30  # seconds
            },
            'alerting': {
                'enabled': True,
                'accuracy_alert_threshold': 0.8,
                'drift_alert_threshold': 0.1,
                'latency_alert_threshold': 200,
                'error_rate_alert_threshold': 0.05
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default

class ModelMonitoringManager:
    """Central manager for all model monitoring activities."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize model monitoring manager."""
        self.config = ModelMonitoringConfig(config)
        self.logger = setup_logger(self.__class__.__name__)
        
        # Initialize monitoring components
        self.performance_monitor = PerformanceMonitor(config=self.config.config)
        self.drift_detector = DriftDetector(config=self.config.config)
        self.accuracy_tracker = AccuracyTracker(config=self.config.config)
        self.resource_monitor = ResourceMonitor(config=self.config.config)
        
        # Monitoring state
        self.active_models = {}
        self.monitoring_threads = {}
        self.monitoring_enabled = True
        self.metrics_store = {}
        self.alert_callbacks = []
        
        self.logger.info("ModelMonitoringManager initialized")
    
    def register_model(self, model_id: str, model: Any, 
                      reference_data: np.ndarray = None):
        """Register a model for monitoring."""
        self.active_models[model_id] = {
            'model': model,
            'reference_data': reference_data,
            'registration_time': datetime.now(),
            'last_monitoring': None,
            'state': ModelState.HEALTHY
        }
        
        # Initialize metrics store for model
        self.metrics_store[model_id] = []
        
        # Start monitoring thread if enabled
        if self.monitoring_enabled:
            self._start_model_monitoring(model_id)
        
        self.logger.info(f"Registered model for monitoring: {model_id}")
    
    def unregister_model(self, model_id: str):
        """Unregister a model from monitoring."""
        if model_id in self.active_models:
            # Stop monitoring thread
            self._stop_model_monitoring(model_id)
            
            # Remove from active models
            del self.active_models[model_id]
            
            self.logger.info(f"Unregistered model from monitoring: {model_id}")
    
    def monitor_prediction(self, model_id: str, input_data: np.ndarray,
                          predictions: np.ndarray, actual: np.ndarray = None,
                          **kwargs) -> ModelMetrics:
        """Monitor a single prediction or batch of predictions."""
        if model_id not in self.active_models:
            raise ValueError(f"Model {model_id} not registered for monitoring")
        
        # Collect performance metrics
        perf_metrics = self.performance_monitor.monitor_model(
            model_id, predictions, actual, **kwargs
        )
        
        # Track accuracy if ground truth available
        accuracy_metrics = {}
        if actual is not None:
            accuracy_metrics = self.accuracy_tracker.track_accuracy(
                model_id, predictions, actual, **kwargs
            )
        
        # Check for drift
        drift_results = {}
        if self.config.get('drift_detection.enabled', True):
            reference_data = self.active_models[model_id].get('reference_data')
            if reference_data is not None:
                drift_results = self.drift_detector.detect_drift(
                    input_data, reference_data
                )
        
        # Monitor resources
        resource_metrics = self.resource_monitor.monitor_resources(model_id)
        
        # Create comprehensive metrics
        metrics = ModelMetrics(
            model_id=model_id,
            timestamp=datetime.now(),
            accuracy=accuracy_metrics.get('accuracy', 0.0),
            precision=accuracy_metrics.get('precision', 0.0),
            recall=accuracy_metrics.get('recall', 0.0),
            f1_score=accuracy_metrics.get('f1_score', 0.0),
            inference_time_ms=resource_metrics.get('inference_time_ms', 0.0),
            memory_usage_mb=resource_metrics.get('memory_mb', 0.0),
            throughput_qps=perf_metrics.throughput_qps if hasattr(perf_metrics, 'throughput_qps') else 0.0,
            error_rate=perf_metrics.error_rate if hasattr(perf_metrics, 'error_rate') else 0.0,
            drift_score=drift_results.get('drift_score', 0.0),
            confidence_score=kwargs.get('confidence', 0.0),
            metadata={
                'drift_detected': drift_results.get('drift_detected', False),
                'resource_metrics': resource_metrics,
                **kwargs
            }
        )
        
        # Store metrics
        self.metrics_store[model_id].append(metrics)
        
        # Limit history size
        max_history = 10000
        if len(self.metrics_store[model_id]) > max_history:
            self.metrics_store[model_id] = self.metrics_store[model_id][-max_history:]
        
        # Check for alerts
        self._check_alerts(metrics)
        
        # Update model state
        self._update_model_state(model_id, metrics)
        
        return metrics
    
    def get_model_metrics(self, model_id: str, 
                         start_time: datetime = None,
                         end_time: datetime = None) -> List[ModelMetrics]:
        """Get metrics for a specific model."""
        if model_id not in self.metrics_store:
            return []
        
        metrics = self.metrics_store[model_id]
        
        if start_time or end_time:
            filtered_metrics = []
            for metric in metrics:
                if start_time and metric.timestamp < start_time:
                    continue
                if end_time and metric.timestamp > end_time:
                    continue
                filtered_metrics.append(metric)
            return filtered_metrics
        
        return metrics
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get current status of a monitored model."""
        if model_id not in self.active_models:
            return {'status': 'not_registered'}
        
        model_info = self.active_models[model_id]
        recent_metrics = self.get_model_metrics(
            model_id, 
            start_time=datetime.now() - timedelta(hours=1)
        )
        
        latest_metrics = recent_metrics[-1] if recent_metrics else None
        
        return {
            'model_id': model_id,
            'state': model_info['state'].value,
            'registration_time': model_info['registration_time'].isoformat(),
            'last_monitoring': model_info['last_monitoring'].isoformat() if model_info['last_monitoring'] else None,
            'latest_metrics': latest_metrics.to_dict() if latest_metrics else None,
            'total_predictions': len(self.metrics_store.get(model_id, [])),
            'recent_predictions': len(recent_metrics),
            'monitoring_enabled': model_id in self.monitoring_threads
        }
    
    def get_all_models_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all monitored models."""
        return {
            model_id: self.get_model_status(model_id)
            for model_id in self.active_models.keys()
        }
    
    def add_alert_callback(self, callback):
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def _start_model_monitoring(self, model_id: str):
        """Start background monitoring for a model."""
        def monitoring_loop():
            while (self.monitoring_enabled and 
                   model_id in self.active_models):
                try:
                    # Periodic monitoring tasks
                    self._periodic_monitoring(model_id)
                    time.sleep(self.config.get('resource_monitoring.monitoring_interval', 30))
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop for {model_id}: {e}")
                    time.sleep(60)  # Wait before retrying
        
        thread = threading.Thread(target=monitoring_loop, daemon=True)
        thread.start()
        self.monitoring_threads[model_id] = thread
    
    def _stop_model_monitoring(self, model_id: str):
        """Stop background monitoring for a model."""
        if model_id in self.monitoring_threads:
            # Thread will stop when model is removed from active_models
            del self.monitoring_threads[model_id]
    
    def _periodic_monitoring(self, model_id: str):
        """Perform periodic monitoring tasks."""
        # Update resource usage
        resource_metrics = self.resource_monitor.monitor_resources(model_id)
        
        # Check for drift if enough data accumulated
        if self.config.get('drift_detection.enabled', True):
            self._check_periodic_drift(model_id)
        
        # Update last monitoring time
        self.active_models[model_id]['last_monitoring'] = datetime.now()
    
    def _check_periodic_drift(self, model_id: str):
        """Check for drift in accumulated data."""
        window_size = self.config.get('drift_detection.window_size', 1000)
        recent_metrics = self.get_model_metrics(
            model_id,
            start_time=datetime.now() - timedelta(hours=24)
        )
        
        if len(recent_metrics) >= window_size:
            # Extract features from recent predictions for drift analysis
            # This would need to be implemented based on specific model types
            pass
    
    def _check_alerts(self, metrics: ModelMetrics):
        """Check if metrics trigger any alerts."""
        alerts = []
        
        # Accuracy alerts
        accuracy_threshold = self.config.get('alerting.accuracy_alert_threshold', 0.8)
        if metrics.accuracy < accuracy_threshold:
            alerts.append({
                'type': 'accuracy_degradation',
                'severity': 'warning' if metrics.accuracy > 0.7 else 'critical',
                'message': f"Model accuracy ({metrics.accuracy:.3f}) below threshold ({accuracy_threshold})",
                'metrics': metrics
            })
        
        # Drift alerts
        drift_threshold = self.config.get('alerting.drift_alert_threshold', 0.1)
        if metrics.drift_score > drift_threshold:
            alerts.append({
                'type': 'drift_detected',
                'severity': 'warning' if metrics.drift_score < 0.2 else 'critical',
                'message': f"Model drift detected (score: {metrics.drift_score:.3f})",
                'metrics': metrics
            })
        
        # Latency alerts
        latency_threshold = self.config.get('alerting.latency_alert_threshold', 200)
        if metrics.inference_time_ms > latency_threshold:
            alerts.append({
                'type': 'high_latency',
                'severity': 'warning',
                'message': f"High inference latency ({metrics.inference_time_ms:.1f}ms)",
                'metrics': metrics
            })
        
        # Error rate alerts
        error_threshold = self.config.get('alerting.error_rate_alert_threshold', 0.05)
        if metrics.error_rate > error_threshold:
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'critical',
                'message': f"High error rate ({metrics.error_rate:.3f})",
                'metrics': metrics
            })
        
        # Send alerts to callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
    
    def _update_model_state(self, model_id: str, metrics: ModelMetrics):
        """Update model state based on metrics."""
        current_state = self.active_models[model_id]['state']
        
        # Determine new state based on metrics
        if (metrics.accuracy < 0.7 or 
            metrics.error_rate > 0.1 or 
            metrics.inference_time_ms > 1000):
            new_state = ModelState.FAILED
        elif (metrics.drift_score > 0.2 or 
              metrics.accuracy < 0.8):
            new_state = ModelState.DRIFTED
        elif (metrics.accuracy < 0.85 or 
              metrics.inference_time_ms > 200):
            new_state = ModelState.DEGRADED
        else:
            new_state = ModelState.HEALTHY
        
        # Update state if changed
        if new_state != current_state:
            self.active_models[model_id]['state'] = new_state
            self.logger.info(f"Model {model_id} state changed: {current_state.value} -> {new_state.value}")
    
    def start_monitoring(self):
        """Start monitoring for all registered models."""
        self.monitoring_enabled = True
        
        for model_id in self.active_models.keys():
            if model_id not in self.monitoring_threads:
                self._start_model_monitoring(model_id)
        
        self.logger.info("Model monitoring started for all registered models")
    
    def stop_monitoring(self):
        """Stop monitoring for all models."""
        self.monitoring_enabled = False
        
        # Wait for threads to stop
        for thread in self.monitoring_threads.values():
            if thread.is_alive():
                thread.join(timeout=5)
        
        self.monitoring_threads.clear()
        self.logger.info("Model monitoring stopped")
    
    def export_metrics(self, model_id: str = None, 
                      format: str = 'json') -> Union[str, Dict[str, Any]]:
        """Export metrics for analysis."""
        if model_id:
            metrics = self.get_model_metrics(model_id)
            data = [m.to_dict() for m in metrics]
        else:
            data = {}
            for mid in self.metrics_store.keys():
                metrics = self.get_model_metrics(mid)
                data[mid] = [m.to_dict() for m in metrics]
        
        if format == 'json':
            return json.dumps(data, indent=2, default=str)
        elif format == 'dataframe':
            if model_id:
                return pd.DataFrame(data)
            else:
                return {mid: pd.DataFrame(metrics) for mid, metrics in data.items()}
        else:
            return data

# Global monitoring manager instance
_model_monitoring_manager = None

def get_model_monitoring_manager(config: Dict[str, Any] = None) -> ModelMonitoringManager:
    """Get or create global model monitoring manager."""
    global _model_monitoring_manager
    if _model_monitoring_manager is None:
        _model_monitoring_manager = ModelMonitoringManager(config)
    return _model_monitoring_manager

def register_model_for_monitoring(model_id: str, model: Any, 
                                 reference_data: np.ndarray = None):
    """Register a model for monitoring."""
    manager = get_model_monitoring_manager()
    manager.register_model(model_id, model, reference_data)

def monitor_model_prediction(model_id: str, input_data: np.ndarray,
                           predictions: np.ndarray, actual: np.ndarray = None,
                           **kwargs) -> ModelMetrics:
    """Monitor a model prediction."""
    manager = get_model_monitoring_manager()
    return manager.monitor_prediction(model_id, input_data, predictions, actual, **kwargs)

def get_model_monitoring_status(model_id: str = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Get model monitoring status."""
    manager = get_model_monitoring_manager()
    if model_id:
        return manager.get_model_status(model_id)
    else:
        return manager.get_all_models_status()

# Export all public components
__all__ = [
    # Enums
    'DriftType',
    'ModelState',
    
    # Data classes
    'ModelMetrics',
    'DriftAlert',
    
    # Core classes
    'ModelMonitoringConfig',
    'ModelMonitoringManager',
    
    # Component classes
    'PerformanceMonitor',
    'DriftDetector',
    'AccuracyTracker',
    'ResourceMonitor',
    
    # Utility functions
    'get_model_monitoring_manager',
    'register_model_for_monitoring',
    'monitor_model_prediction',
    'get_model_monitoring_status',
    
    # Version info
    '__version__',
    '__author__'
]

# Module initialization
logger.info(f"BatteryMind Model Monitoring module initialized (v{__version__})")
logger.info("Available components: PerformanceMonitor, DriftDetector, AccuracyTracker, ResourceMonitor")
