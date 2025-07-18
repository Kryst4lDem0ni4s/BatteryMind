"""
BatteryMind - Drift Detector

Advanced data and model drift detection system for battery AI models.
Monitors for statistical changes in data distributions and model behavior
to ensure continued model reliability in production environments.

Features:
- Real-time statistical drift detection
- Multi-dimensional distribution analysis
- Model behavior drift monitoring
- Automated drift alerts and responses
- Historical drift pattern analysis
- Predictive drift forecasting
- Adaptive threshold management

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
import json
import warnings
from pathlib import Path
from collections import deque
from abc import ABC, abstractmethod
from scipy import stats
from scipy.spatial.distance import wasserstein_distance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# BatteryMind imports
from ...utils.logging_utils import setup_logger
from ...utils.data_utils import StatisticalAnalyzer, DistributionAnalyzer
from ...utils.config_parser import ConfigParser
from ..alerts.alert_manager import AlertManager

# Configure logging
logger = setup_logger(__name__)

@dataclass
class DriftThreshold:
    """Drift detection threshold configuration."""
    metric_name: str
    threshold_value: float
    detection_method: str  # 'statistical', 'distance', 'performance'
    severity: str  # 'critical', 'high', 'medium', 'low'
    window_size: int = 100
    significance_level: float = 0.05

@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    timestamp: datetime
    feature_name: str
    drift_detected: bool
    drift_score: float
    drift_method: str
    p_value: Optional[float] = None
    distance_metric: Optional[float] = None
    severity: str = 'low'
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DriftAlert:
    """Drift alert data structure."""
    alert_id: str
    timestamp: datetime
    drift_type: str  # 'data_drift', 'concept_drift', 'model_drift'
    severity: str
    feature_name: str
    drift_score: float
    model_id: str
    message: str
    recommended_actions: List[str] = field(default_factory=list)

class DriftDetectorConfig:
    """Configuration for drift detection system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = ConfigParser.load_config(config_path) if config_path else {}
        
        # Detection settings
        self.detection_interval_minutes = self.config.get('detection_interval_minutes', 30)
        self.reference_window_size = self.config.get('reference_window_size', 1000)
        self.current_window_size = self.config.get('current_window_size', 100)
        
        # Statistical settings
        self.significance_level = self.config.get('significance_level', 0.05)
        self.min_samples_required = self.config.get('min_samples_required', 50)
        
        # Drift thresholds
        self.drift_thresholds = self._load_drift_thresholds()
        
        # Alert settings
        self.enable_drift_alerts = self.config.get('enable_drift_alerts', True)
        self.alert_cooldown_hours = self.config.get('alert_cooldown_hours', 6)
        
        # Storage settings
        self.drift_history_days = self.config.get('drift_history_days', 30)
        
    def _load_drift_thresholds(self) -> List[DriftThreshold]:
        """Load drift detection thresholds."""
        threshold_configs = self.config.get('drift_thresholds', [])
        thresholds = []
        
        # Default thresholds
        default_thresholds = [
            {'metric_name': 'ks_test', 'threshold_value': 0.05, 'detection_method': 'statistical', 'severity': 'high'},
            {'metric_name': 'wasserstein_distance', 'threshold_value': 0.1, 'detection_method': 'distance', 'severity': 'medium'},
            {'metric_name': 'accuracy_drop', 'threshold_value': 0.05, 'detection_method': 'performance', 'severity': 'critical'},
            {'metric_name': 'population_stability_index', 'threshold_value': 0.2, 'detection_method': 'statistical', 'severity': 'medium'},
        ]
        
        for threshold_config in (threshold_configs or default_thresholds):
            thresholds.append(DriftThreshold(**threshold_config))
        
        return thresholds

class DriftDetectionMethod(ABC):
    """Abstract base class for drift detection methods."""
    
    @abstractmethod
    def detect_drift(self, reference_data: np.ndarray, 
                    current_data: np.ndarray, **kwargs) -> DriftDetectionResult:
        """Detect drift between reference and current data."""
        pass

class KolmogorovSmirnovTest(DriftDetectionMethod):
    """Kolmogorov-Smirnov test for drift detection."""
    
    def detect_drift(self, reference_data: np.ndarray, 
                    current_data: np.ndarray, **kwargs) -> DriftDetectionResult:
        """Perform KS test for drift detection."""
        timestamp = datetime.now()
        feature_name = kwargs.get('feature_name', 'unknown')
        threshold = kwargs.get('threshold', 0.05)
        
        try:
            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(reference_data, current_data)
            
            # Determine if drift is detected
            drift_detected = p_value < threshold
            severity = 'high' if drift_detected else 'low'
            
            return DriftDetectionResult(
                timestamp=timestamp,
                feature_name=feature_name,
                drift_detected=drift_detected,
                drift_score=ks_statistic,
                drift_method='kolmogorov_smirnov',
                p_value=p_value,
                severity=severity,
                details={
                    'ks_statistic': ks_statistic,
                    'p_value': p_value,
                    'threshold': threshold,
                    'reference_size': len(reference_data),
                    'current_size': len(current_data)
                }
            )
            
        except Exception as e:
            logger.error(f"KS test failed: {e}")
            return DriftDetectionResult(
                timestamp=timestamp,
                feature_name=feature_name,
                drift_detected=False,
                drift_score=0.0,
                drift_method='kolmogorov_smirnov',
                details={'error': str(e)}
            )

class WassersteinDistance(DriftDetectionMethod):
    """Wasserstein distance for drift detection."""
    
    def detect_drift(self, reference_data: np.ndarray, 
                    current_data: np.ndarray, **kwargs) -> DriftDetectionResult:
        """Calculate Wasserstein distance for drift detection."""
        timestamp = datetime.now()
        feature_name = kwargs.get('feature_name', 'unknown')
        threshold = kwargs.get('threshold', 0.1)
        
        try:
            # Calculate Wasserstein distance
            distance = wasserstein_distance(reference_data, current_data)
            
            # Normalize distance
            reference_range = np.max(reference_data) - np.min(reference_data)
            normalized_distance = distance / reference_range if reference_range > 0 else 0
            
            # Determine if drift is detected
            drift_detected = normalized_distance > threshold
            severity = 'medium' if drift_detected else 'low'
            
            return DriftDetectionResult(
                timestamp=timestamp,
                feature_name=feature_name,
                drift_detected=drift_detected,
                drift_score=normalized_distance,
                drift_method='wasserstein_distance',
                distance_metric=distance,
                severity=severity,
                details={
                    'raw_distance': distance,
                    'normalized_distance': normalized_distance,
                    'threshold': threshold,
                    'reference_range': reference_range
                }
            )
            
        except Exception as e:
            logger.error(f"Wasserstein distance calculation failed: {e}")
            return DriftDetectionResult(
                timestamp=timestamp,
                feature_name=feature_name,
                drift_detected=False,
                drift_score=0.0,
                drift_method='wasserstein_distance',
                details={'error': str(e)}
            )

class PopulationStabilityIndex(DriftDetectionMethod):
    """Population Stability Index for drift detection."""
    
    def detect_drift(self, reference_data: np.ndarray, 
                    current_data: np.ndarray, **kwargs) -> DriftDetectionResult:
        """Calculate PSI for drift detection."""
        timestamp = datetime.now()
        feature_name = kwargs.get('feature_name', 'unknown')
        threshold = kwargs.get('threshold', 0.2)
        n_bins = kwargs.get('n_bins', 10)
        
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference_data, bins=n_bins)
            
            # Calculate histograms
            ref_hist, _ = np.histogram(reference_data, bins=bin_edges, density=True)
            cur_hist, _ = np.histogram(current_data, bins=bin_edges, density=True)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            ref_hist = ref_hist + epsilon
            cur_hist = cur_hist + epsilon
            
            # Calculate PSI
            psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
            
            # Determine drift
            drift_detected = psi > threshold
            severity = self._get_psi_severity(psi)
            
            return DriftDetectionResult(
                timestamp=timestamp,
                feature_name=feature_name,
                drift_detected=drift_detected,
                drift_score=psi,
                drift_method='population_stability_index',
                severity=severity,
                details={
                    'psi_value': psi,
                    'threshold': threshold,
                    'n_bins': n_bins,
                    'reference_hist': ref_hist.tolist(),
                    'current_hist': cur_hist.tolist()
                }
            )
            
        except Exception as e:
            logger.error(f"PSI calculation failed: {e}")
            return DriftDetectionResult(
                timestamp=timestamp,
                feature_name=feature_name,
                drift_detected=False,
                drift_score=0.0,
                drift_method='population_stability_index',
                details={'error': str(e)}
            )
    
    def _get_psi_severity(self, psi_value: float) -> str:
        """Determine severity based on PSI value."""
        if psi_value > 0.25:
            return 'critical'
        elif psi_value > 0.1:
            return 'high'
        elif psi_value > 0.05:
            return 'medium'
        else:
            return 'low'

class ModelPerformanceDrift(DriftDetectionMethod):
    """Model performance drift detection."""
    
    def detect_drift(self, reference_performance: np.ndarray, 
                    current_performance: np.ndarray, **kwargs) -> DriftDetectionResult:
        """Detect performance drift."""
        timestamp = datetime.now()
        feature_name = kwargs.get('feature_name', 'model_performance')
        threshold = kwargs.get('threshold', 0.05)
        
        try:
            # Calculate performance metrics
            ref_mean = np.mean(reference_performance)
            cur_mean = np.mean(current_performance)
            
            # Calculate performance drop
            performance_drop = ref_mean - cur_mean
            relative_drop = performance_drop / ref_mean if ref_mean > 0 else 0
            
            # Statistical test for significance
            t_stat, p_value = stats.ttest_ind(reference_performance, current_performance)
            
            # Determine drift
            drift_detected = (relative_drop > threshold) and (p_value < 0.05)
            severity = 'critical' if drift_detected else 'low'
            
            return DriftDetectionResult(
                timestamp=timestamp,
                feature_name=feature_name,
                drift_detected=drift_detected,
                drift_score=relative_drop,
                drift_method='performance_drift',
                p_value=p_value,
                severity=severity,
                details={
                    'reference_mean': ref_mean,
                    'current_mean': cur_mean,
                    'absolute_drop': performance_drop,
                    'relative_drop': relative_drop,
                    'threshold': threshold,
                    't_statistic': t_stat
                }
            )
            
        except Exception as e:
            logger.error(f"Performance drift detection failed: {e}")
            return DriftDetectionResult(
                timestamp=timestamp,
                feature_name=feature_name,
                drift_detected=False,
                drift_score=0.0,
                drift_method='performance_drift',
                details={'error': str(e)}
            )

class DriftDetector:
    """
    Advanced drift detection system for battery AI models.
    """
    
    def __init__(self, config: DriftDetectorConfig):
        self.config = config
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Data storage
        self.reference_data = {}
        self.current_data = {}
        self.drift_history = deque(maxlen=10000)
        self.alert_history = deque(maxlen=1000)
        
        # Detection methods
        self.detection_methods = {
            'ks_test': KolmogorovSmirnovTest(),
            'wasserstein_distance': WassersteinDistance(),
            'psi': PopulationStabilityIndex(),
            'performance_drift': ModelPerformanceDrift()
        }
        
        # Alert management
        self.alert_manager = AlertManager()
        self.last_alert_times = {}
        
        # Statistics
        self.stats_analyzer = StatisticalAnalyzer()
        
        logger.info("DriftDetector initialized with comprehensive drift monitoring")
    
    def set_reference_data(self, feature_name: str, data: np.ndarray, model_id: str = "default"):
        """Set reference data for drift detection."""
        key = f"{model_id}_{feature_name}"
        self.reference_data[key] = {
            'data': data.copy(),
            'timestamp': datetime.now(),
            'statistics': {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'size': len(data)
            }
        }
        
        logger.info(f"Reference data set for {feature_name} (model: {model_id}): {len(data)} samples")
    
    def add_current_data(self, feature_name: str, data: np.ndarray, model_id: str = "default"):
        """Add current data for drift monitoring."""
        key = f"{model_id}_{feature_name}"
        
        if key not in self.current_data:
            self.current_data[key] = deque(maxlen=self.config.current_window_size)
        
        # Add data points
        if data.ndim == 0:
            self.current_data[key].append(data)
        else:
            self.current_data[key].extend(data)
    
    def detect_drift(self, feature_name: str, model_id: str = "default", 
                    methods: Optional[List[str]] = None) -> List[DriftDetectionResult]:
        """Detect drift for a specific feature."""
        key = f"{model_id}_{feature_name}"
        
        # Check if we have reference and current data
        if key not in self.reference_data:
            logger.warning(f"No reference data for {feature_name} (model: {model_id})")
            return []
        
        if key not in self.current_data or len(self.current_data[key]) < self.config.min_samples_required:
            logger.warning(f"Insufficient current data for {feature_name} (model: {model_id})")
            return []
        
        reference_data = self.reference_data[key]['data']
        current_data = np.array(list(self.current_data[key]))
        
        # Use specified methods or all available methods
        if methods is None:
            methods = list(self.detection_methods.keys())
        
        results = []
        
        for method_name in methods:
            if method_name not in self.detection_methods:
                logger.warning(f"Unknown drift detection method: {method_name}")
                continue
            
            try:
                # Get threshold for this method
                threshold = self._get_threshold_for_method(method_name)
                
                # Perform drift detection
                result = self.detection_methods[method_name].detect_drift(
                    reference_data, current_data,
                    feature_name=feature_name,
                    threshold=threshold
                )
                
                results.append(result)
                
                # Store in history
                self.drift_history.append(result)
                
                # Generate alert if drift detected
                if result.drift_detected:
                    self._generate_drift_alert(result, model_id)
                
            except Exception as e:
                logger.error(f"Drift detection failed for method {method_name}: {e}")
        
        return results
    
    def _get_threshold_for_method(self, method_name: str) -> float:
        """Get threshold value for a specific detection method."""
        for threshold in self.config.drift_thresholds:
            if threshold.metric_name == method_name:
                return threshold.threshold_value
        
        # Default thresholds
        defaults = {
            'ks_test': 0.05,
            'wasserstein_distance': 0.1,
            'psi': 0.2,
            'performance_drift': 0.05
        }
        
        return defaults.get(method_name, 0.05)
    
    def _generate_drift_alert(self, result: DriftDetectionResult, model_id: str):
        """Generate drift alert."""
        alert_key = f"{model_id}_{result.feature_name}_{result.drift_method}"
        current_time = datetime.now()
        
        # Check cooldown period
        if alert_key in self.last_alert_times:
            time_since_last = current_time - self.last_alert_times[alert_key]
            if time_since_last.total_seconds() < self.config.alert_cooldown_hours * 3600:
                return
        
        # Determine drift type
        drift_type = self._determine_drift_type(result.drift_method)
        
        # Generate recommended actions
        recommended_actions = self._generate_recommended_actions(result, drift_type)
        
        # Create alert
        alert = DriftAlert(
            alert_id=f"drift_{alert_key}_{int(current_time.timestamp())}",
            timestamp=current_time,
            drift_type=drift_type,
            severity=result.severity,
            feature_name=result.feature_name,
            drift_score=result.drift_score,
            model_id=model_id,
            message=f"Drift detected in {result.feature_name} using {result.drift_method} "
                   f"(score: {result.drift_score:.4f}, severity: {result.severity})",
            recommended_actions=recommended_actions
        )
        
        # Store alert
        self.alert_history.append(alert)
        self.last_alert_times[alert_key] = current_time
        
        # Send notification
        if self.config.enable_drift_alerts:
            self._send_drift_notification(alert)
        
        logger.warning(f"Drift alert generated: {alert.message}")
    
    def _determine_drift_type(self, method_name: str) -> str:
        """Determine the type of drift based on detection method."""
        if method_name in ['ks_test', 'wasserstein_distance', 'psi']:
            return 'data_drift'
        elif method_name == 'performance_drift':
            return 'concept_drift'
        else:
            return 'model_drift'
    
    def _generate_recommended_actions(self, result: DriftDetectionResult, drift_type: str) -> List[str]:
        """Generate recommended actions based on drift type and severity."""
        actions = []
        
        if drift_type == 'data_drift':
            actions.extend([
                "Investigate data collection process for changes",
                "Check sensor calibration and data preprocessing",
                "Consider retraining model with recent data"
            ])
        elif drift_type == 'concept_drift':
            actions.extend([
                "Retrain model with recent labeled data",
                "Update model parameters and thresholds",
                "Implement adaptive learning mechanisms"
            ])
        elif drift_type == 'model_drift':
            actions.extend([
                "Check model deployment and inference pipeline",
                "Validate model artifacts and configurations",
                "Consider model refresh or replacement"
            ])
        
        if result.severity == 'critical':
            actions.insert(0, "Immediate attention required - consider emergency model rollback")
        
        return actions
    
    def _send_drift_notification(self, alert: DriftAlert):
        """Send drift alert notification."""
        try:
            self.alert_manager.send_alert(
                alert_type='data_drift',
                severity=alert.severity,
                message=alert.message,
                metadata={
                    'model_id': alert.model_id,
                    'feature_name': alert.feature_name,
                    'drift_type': alert.drift_type,
                    'drift_score': alert.drift_score,
                    'recommended_actions': alert.recommended_actions
                }
            )
        except Exception as e:
            logger.error(f"Failed to send drift notification: {e}")
    
    def start_monitoring(self):
        """Start continuous drift monitoring."""
        if self.is_monitoring:
            logger.warning("Drift monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Drift monitoring started")
    
    def stop_monitoring(self):
        """Stop drift monitoring."""
        self.is_monitoring = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Drift monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Check drift for all features with sufficient data
                for key in list(self.current_data.keys()):
                    if len(self.current_data[key]) >= self.config.min_samples_required:
                        model_id, feature_name = key.split('_', 1)
                        self.detect_drift(feature_name, model_id)
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Wait for next iteration
                time.sleep(self.config.detection_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in drift monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _cleanup_old_data(self):
        """Clean up old drift history and alerts."""
        cutoff_time = datetime.now() - timedelta(days=self.config.drift_history_days)
        
        # Clean up drift history
        original_count = len(self.drift_history)
        self.drift_history = deque(
            [result for result in self.drift_history if result.timestamp >= cutoff_time],
            maxlen=self.drift_history.maxlen
        )
        
        cleaned_count = original_count - len(self.drift_history)
        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} old drift records")
    
    def get_drift_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get drift detection summary."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent drift results
        recent_results = [
            result for result in self.drift_history
            if result.timestamp >= cutoff_time
        ]
        
        # Filter recent alerts
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
        
        # Calculate statistics
        total_checks = len(recent_results)
        drift_detected = len([r for r in recent_results if r.drift_detected])
        
        # Group by feature and method
        feature_stats = {}
        method_stats = {}
        
        for result in recent_results:
            # Feature statistics
            if result.feature_name not in feature_stats:
                feature_stats[result.feature_name] = {'checks': 0, 'drifts': 0}
            feature_stats[result.feature_name]['checks'] += 1
            if result.drift_detected:
                feature_stats[result.feature_name]['drifts'] += 1
            
            # Method statistics
            if result.drift_method not in method_stats:
                method_stats[result.drift_method] = {'checks': 0, 'drifts': 0}
            method_stats[result.drift_method]['checks'] += 1
            if result.drift_detected:
                method_stats[result.drift_method]['drifts'] += 1
        
        return {
            'period_hours': hours,
            'total_drift_checks': total_checks,
            'drift_detected_count': drift_detected,
            'drift_rate': drift_detected / total_checks if total_checks > 0 else 0,
            'total_alerts': len(recent_alerts),
            'feature_statistics': feature_stats,
            'method_statistics': method_stats,
            'alert_breakdown': {
                'critical': len([a for a in recent_alerts if a.severity == 'critical']),
                'high': len([a for a in recent_alerts if a.severity == 'high']),
                'medium': len([a for a in recent_alerts if a.severity == 'medium']),
                'low': len([a for a in recent_alerts if a.severity == 'low'])
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def export_drift_report(self, filepath: str):
        """Export comprehensive drift report."""
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'configuration': {
                'detection_interval_minutes': self.config.detection_interval_minutes,
                'reference_window_size': self.config.reference_window_size,
                'current_window_size': self.config.current_window_size,
                'significance_level': self.config.significance_level
            },
            'drift_thresholds': [
                {
                    'metric_name': t.metric_name,
                    'threshold_value': t.threshold_value,
                    'detection_method': t.detection_method,
                    'severity': t.severity
                }
                for t in self.config.drift_thresholds
            ],
            'drift_history': [
                {
                    'timestamp': result.timestamp.isoformat(),
                    'feature_name': result.feature_name,
                    'drift_detected': result.drift_detected,
                    'drift_score': result.drift_score,
                    'drift_method': result.drift_method,
                    'severity': result.severity,
                    'p_value': result.p_value,
                    'distance_metric': result.distance_metric
                }
                for result in self.drift_history
            ],
            'alert_history': [
                {
                    'alert_id': alert.alert_id,
                    'timestamp': alert.timestamp.isoformat(),
                    'drift_type': alert.drift_type,
                    'severity': alert.severity,
                    'feature_name': alert.feature_name,
                    'drift_score': alert.drift_score,
                    'model_id': alert.model_id,
                    'message': alert.message,
                    'recommended_actions': alert.recommended_actions
                }
                for alert in self.alert_history
            ],
            'summary': self.get_drift_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Drift report exported to {filepath}")

# Factory function
def create_drift_detector(config_path: Optional[str] = None) -> DriftDetector:
    """Create a drift detector instance."""
    config = DriftDetectorConfig(config_path)
    return DriftDetector(config)

# Usage example
if __name__ == "__main__":
    # Create drift detector
    detector = create_drift_detector()
    
    # Set reference data
    reference_data = np.random.normal(0.8, 0.1, 1000)
    detector.set_reference_data('battery_soh', reference_data)
    
    # Add current data (simulating drift)
    current_data = np.random.normal(0.7, 0.15, 100)  # Shifted distribution
    detector.add_current_data('battery_soh', current_data)
    
    # Detect drift
    results = detector.detect_drift('battery_soh')
    
    for result in results:
        print(f"Method: {result.drift_method}")
        print(f"Drift detected: {result.drift_detected}")
        print(f"Drift score: {result.drift_score:.4f}")
        print(f"Severity: {result.severity}")
        print("---")
    
    # Get summary
    summary = detector.get_drift_summary()
    print("Drift Summary:", json.dumps(summary, indent=2))
