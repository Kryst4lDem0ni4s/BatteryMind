"""
BatteryMind - Anomaly Detector

Advanced anomaly detection system for battery data streams using statistical,
machine learning, and domain-specific techniques. Provides real-time anomaly
detection, adaptive thresholds, and comprehensive anomaly analysis for battery
telemetry, sensor data, and operational metrics.

Features:
- Multi-algorithm anomaly detection (statistical, ML, ensemble)
- Real-time streaming anomaly detection
- Adaptive threshold management
- Battery-specific anomaly patterns
- Severity classification and prioritization
- Anomaly explanation and root cause analysis
- Integration with alerting systems

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import warnings
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from enum import Enum
from collections import deque
import pickle

# Statistical and ML libraries
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope

# Time series analysis
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# BatteryMind imports
from ...utils.logging_utils import get_logger
from ...utils.data_utils import DataProcessor, TimeSeriesProcessor
from ...training_data.generators.physics_simulator import BatteryPhysicsSimulator

# Configure logging
logger = get_logger(__name__)

class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    POINT_ANOMALY = "point_anomaly"           # Single data point anomaly
    COLLECTIVE_ANOMALY = "collective_anomaly" # Sequence of points
    CONTEXTUAL_ANOMALY = "contextual_anomaly" # Context-dependent anomaly
    TREND_ANOMALY = "trend_anomaly"           # Unusual trend
    SEASONAL_ANOMALY = "seasonal_anomaly"     # Seasonal pattern deviation
    DRIFT_ANOMALY = "drift_anomaly"           # Gradual drift
    SPIKE_ANOMALY = "spike_anomaly"           # Sudden spike
    DROP_ANOMALY = "drop_anomaly"             # Sudden drop

class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""
    CRITICAL = "critical"     # Immediate attention required
    HIGH = "high"            # High priority
    MEDIUM = "medium"        # Medium priority
    LOW = "low"              # Low priority
    INFO = "info"            # Informational

class DetectorType(Enum):
    """Types of anomaly detectors."""
    STATISTICAL = "statistical"
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    AUTOENCODER = "autoencoder"
    ENSEMBLE = "ensemble"
    PHYSICS_BASED = "physics_based"
    THRESHOLD_BASED = "threshold_based"

@dataclass
class AnomalyEvent:
    """
    Detected anomaly event.
    
    Attributes:
        anomaly_id (str): Unique identifier
        timestamp (datetime): When anomaly occurred
        data_point (Dict): Anomalous data point
        anomaly_type (AnomalyType): Type of anomaly
        severity (AnomalySeverity): Severity level
        confidence_score (float): Detection confidence (0-1)
        affected_features (List[str]): Features showing anomalous behavior
        detector_name (str): Name of detector that found anomaly
        explanation (str): Human-readable explanation
        context (Dict): Additional context information
        suggested_actions (List[str]): Suggested remedial actions
    """
    anomaly_id: str
    timestamp: datetime
    data_point: Dict[str, Any]
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence_score: float
    affected_features: List[str] = field(default_factory=list)
    detector_name: str = ""
    explanation: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_actions: List[str] = field(default_factory=list)

@dataclass
class DetectorConfig:
    """Configuration for anomaly detectors."""
    detector_type: DetectorType
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    sensitivity: float = 0.5  # 0=low sensitivity, 1=high sensitivity
    adaptation_rate: float = 0.1  # How quickly to adapt thresholds

class BaseAnomalyDetector(ABC):
    """Abstract base class for anomaly detectors."""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.is_fitted = False
        self.detection_history = deque(maxlen=1000)
        
    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Fit the detector on training data."""
        pass
    
    @abstractmethod
    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in data.
        
        Returns:
            Tuple of (anomaly_labels, anomaly_scores)
        """
        pass
    
    @abstractmethod
    def get_detector_info(self) -> Dict[str, Any]:
        """Get detector information."""
        pass

class StatisticalAnomalyDetector(BaseAnomalyDetector):
    """Statistical anomaly detection using z-score and IQR methods."""
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.feature_stats = {}
        self.threshold_multiplier = config.parameters.get("threshold_multiplier", 3.0)
        
    def fit(self, X: np.ndarray) -> None:
        """Fit statistical parameters."""
        self.feature_stats = {}
        
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            self.feature_stats[i] = {
                "mean": np.mean(feature_data),
                "std": np.std(feature_data),
                "median": np.median(feature_data),
                "q1": np.percentile(feature_data, 25),
                "q3": np.percentile(feature_data, 75),
                "iqr": np.percentile(feature_data, 75) - np.percentile(feature_data, 25)
            }
        
        self.is_fitted = True
    
    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using statistical methods."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")
        
        anomaly_labels = np.zeros(X.shape[0], dtype=bool)
        anomaly_scores = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            max_score = 0
            is_anomaly = False
            
            for j in range(X.shape[1]):
                value = X[i, j]
                stats = self.feature_stats[j]
                
                # Z-score method
                z_score = abs((value - stats["mean"]) / (stats["std"] + 1e-8))
                
                # IQR method
                iqr_score = 0
                if value < stats["q1"] - 1.5 * stats["iqr"]:
                    iqr_score = (stats["q1"] - 1.5 * stats["iqr"] - value) / stats["iqr"]
                elif value > stats["q3"] + 1.5 * stats["iqr"]:
                    iqr_score = (value - stats["q3"] - 1.5 * stats["iqr"]) / stats["iqr"]
                
                # Combined score
                feature_score = max(z_score / self.threshold_multiplier, iqr_score)
                max_score = max(max_score, feature_score)
                
                # Check if anomalous
                if z_score > self.threshold_multiplier or iqr_score > 1.0:
                    is_anomaly = True
            
            anomaly_labels[i] = is_anomaly
            anomaly_scores[i] = min(max_score, 1.0)  # Cap at 1.0
        
        return anomaly_labels, anomaly_scores
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get detector information."""
        return {
            "detector_type": "statistical",
            "method": "z_score_iqr",
            "threshold_multiplier": self.threshold_multiplier,
            "features_fitted": len(self.feature_stats),
            "is_fitted": self.is_fitted
        }

class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest anomaly detection."""
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.model = IsolationForest(
            contamination=config.parameters.get("contamination", 0.1),
            n_estimators=config.parameters.get("n_estimators", 100),
            random_state=config.parameters.get("random_state", 42)
        )
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray) -> None:
        """Fit Isolation Forest model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
    
    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using Isolation Forest."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")
        
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly labels (-1 for anomaly, 1 for normal)
        labels = self.model.predict(X_scaled)
        anomaly_labels = labels == -1
        
        # Get anomaly scores (more negative = more anomalous)
        scores = self.model.decision_function(X_scaled)
        # Convert to 0-1 scale (higher = more anomalous)
        anomaly_scores = 1 / (1 + np.exp(scores))  # Sigmoid transformation
        
        return anomaly_labels, anomaly_scores
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get detector information."""
        return {
            "detector_type": "isolation_forest",
            "contamination": self.model.contamination,
            "n_estimators": self.model.n_estimators,
            "is_fitted": self.is_fitted
        }

class OneClassSVMDetector(BaseAnomalyDetector):
    """One-Class SVM anomaly detection."""
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.model = OneClassSVM(
            kernel=config.parameters.get("kernel", "rbf"),
            gamma=config.parameters.get("gamma", "scale"),
            nu=config.parameters.get("nu", 0.1)
        )
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray) -> None:
        """Fit One-Class SVM model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
    
    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using One-Class SVM."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")
        
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly labels
        labels = self.model.predict(X_scaled)
        anomaly_labels = labels == -1
        
        # Get decision function scores
        scores = self.model.decision_function(X_scaled)
        # Convert to 0-1 scale
        anomaly_scores = 1 / (1 + np.exp(scores))
        
        return anomaly_labels, anomaly_scores
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get detector information."""
        return {
            "detector_type": "one_class_svm",
            "kernel": self.model.kernel,
            "gamma": self.model.gamma,
            "nu": self.model.nu,
            "is_fitted": self.is_fitted
        }

class PhysicsBasedDetector(BaseAnomalyDetector):
    """Physics-based anomaly detection for battery data."""
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.physics_simulator = BatteryPhysicsSimulator()
        self.tolerance = config.parameters.get("tolerance", 0.1)
        self.physics_bounds = {}
        
    def fit(self, X: np.ndarray) -> None:
        """Fit physics-based bounds."""
        # Assume X has columns: [voltage, current, temperature, soc, ...]
        self.physics_bounds = {
            "voltage_min": 2.0, "voltage_max": 4.5,
            "current_min": -1000, "current_max": 1000,
            "temperature_min": -40, "temperature_max": 80,
            "soc_min": 0.0, "soc_max": 1.0,
            "power_max": 1000.0  # kW
        }
        self.is_fitted = True
    
    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect physics violations."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")
        
        anomaly_labels = np.zeros(X.shape[0], dtype=bool)
        anomaly_scores = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            violations = []
            
            # Assume column order: voltage, current, temperature, soc
            if X.shape[1] >= 4:
                voltage = X[i, 0]
                current = X[i, 1]
                temperature = X[i, 2]
                soc = X[i, 3]
                
                # Check voltage bounds
                if voltage < self.physics_bounds["voltage_min"]:
                    violations.append(("voltage_low", abs(voltage - self.physics_bounds["voltage_min"])))
                elif voltage > self.physics_bounds["voltage_max"]:
                    violations.append(("voltage_high", abs(voltage - self.physics_bounds["voltage_max"])))
                
                # Check current bounds
                if abs(current) > self.physics_bounds["current_max"]:
                    violations.append(("current_high", abs(current) - self.physics_bounds["current_max"]))
                
                # Check temperature bounds
                if temperature < self.physics_bounds["temperature_min"]:
                    violations.append(("temperature_low", abs(temperature - self.physics_bounds["temperature_min"])))
                elif temperature > self.physics_bounds["temperature_max"]:
                    violations.append(("temperature_high", abs(temperature - self.physics_bounds["temperature_max"])))
                
                # Check SOC bounds
                if soc < self.physics_bounds["soc_min"] or soc > self.physics_bounds["soc_max"]:
                    violations.append(("soc_bounds", max(abs(soc - self.physics_bounds["soc_min"]), 
                                                        abs(soc - self.physics_bounds["soc_max"]))))
                
                # Check power consistency (P = V * I)
                power = abs(voltage * current)
                if power > self.physics_bounds["power_max"]:
                    violations.append(("power_high", power - self.physics_bounds["power_max"]))
            
            # Determine if anomalous
            if violations:
                anomaly_labels[i] = True
                # Score based on worst violation
                max_violation = max(violation[1] for violation in violations)
                anomaly_scores[i] = min(max_violation / 10.0, 1.0)  # Normalize
            
        return anomaly_labels, anomaly_scores
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get detector information."""
        return {
            "detector_type": "physics_based",
            "physics_bounds": self.physics_bounds,
            "tolerance": self.tolerance,
            "is_fitted": self.is_fitted
        }

class EnsembleAnomalyDetector(BaseAnomalyDetector):
    """Ensemble of multiple anomaly detectors."""
    
    def __init__(self, configs: List[DetectorConfig]):
        super().__init__(DetectorConfig(DetectorType.ENSEMBLE))
        self.detectors = []
        self.detector_weights = []
        
        # Create individual detectors
        for config in configs:
            if config.detector_type == DetectorType.STATISTICAL:
                detector = StatisticalAnomalyDetector(config)
            elif config.detector_type == DetectorType.ISOLATION_FOREST:
                detector = IsolationForestDetector(config)
            elif config.detector_type == DetectorType.ONE_CLASS_SVM:
                detector = OneClassSVMDetector(config)
            elif config.detector_type == DetectorType.PHYSICS_BASED:
                detector = PhysicsBasedDetector(config)
            else:
                continue
            
            self.detectors.append(detector)
            self.detector_weights.append(1.0)  # Equal weights initially
        
        # Normalize weights
        total_weight = sum(self.detector_weights)
        self.detector_weights = [w / total_weight for w in self.detector_weights]
    
    def fit(self, X: np.ndarray) -> None:
        """Fit all detectors in ensemble."""
        for detector in self.detectors:
            try:
                detector.fit(X)
            except Exception as e:
                logger.warning(f"Failed to fit detector {detector.__class__.__name__}: {e}")
        
        self.is_fitted = True
    
    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using ensemble voting."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")
        
        all_labels = []
        all_scores = []
        
        # Get predictions from all detectors
        for detector in self.detectors:
            try:
                labels, scores = detector.detect(X)
                all_labels.append(labels)
                all_scores.append(scores)
            except Exception as e:
                logger.warning(f"Detection failed for {detector.__class__.__name__}: {e}")
                # Use zeros as fallback
                all_labels.append(np.zeros(X.shape[0], dtype=bool))
                all_scores.append(np.zeros(X.shape[0]))
        
        if not all_labels:
            return np.zeros(X.shape[0], dtype=bool), np.zeros(X.shape[0])
        
        # Ensemble voting for labels (majority vote)
        label_matrix = np.array(all_labels).T
        ensemble_labels = np.sum(label_matrix, axis=1) > (len(self.detectors) / 2)
        
        # Weighted average for scores
        score_matrix = np.array(all_scores).T
        ensemble_scores = np.average(score_matrix, axis=1, weights=self.detector_weights)
        
        return ensemble_labels, ensemble_scores
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get ensemble detector information."""
        return {
            "detector_type": "ensemble",
            "num_detectors": len(self.detectors),
            "detector_types": [d.__class__.__name__ for d in self.detectors],
            "detector_weights": self.detector_weights,
            "is_fitted": self.is_fitted
        }

class BatteryAnomalyDetector:
    """
    Main anomaly detection system for battery data.
    """
    
    def __init__(self, detector_configs: List[DetectorConfig] = None):
        self.detectors = {}
        self.anomaly_history = deque(maxlen=10000)
        self.physics_simulator = BatteryPhysicsSimulator()
        
        # Initialize default detectors if none provided
        if detector_configs is None:
            detector_configs = self._get_default_detector_configs()
        
        # Create detectors
        for config in detector_configs:
            if config.enabled:
                detector = self._create_detector(config)
                if detector:
                    self.detectors[config.detector_type.value] = detector
        
        logger.info(f"BatteryAnomalyDetector initialized with {len(self.detectors)} detectors")
    
    def _get_default_detector_configs(self) -> List[DetectorConfig]:
        """Get default detector configurations."""
        return [
            DetectorConfig(
                detector_type=DetectorType.STATISTICAL,
                parameters={"threshold_multiplier": 3.0}
            ),
            DetectorConfig(
                detector_type=DetectorType.ISOLATION_FOREST,
                parameters={"contamination": 0.1, "n_estimators": 100}
            ),
            DetectorConfig(
                detector_type=DetectorType.PHYSICS_BASED,
                parameters={"tolerance": 0.1}
            )
        ]
    
    def _create_detector(self, config: DetectorConfig) -> Optional[BaseAnomalyDetector]:
        """Create detector instance from configuration."""
        try:
            if config.detector_type == DetectorType.STATISTICAL:
                return StatisticalAnomalyDetector(config)
            elif config.detector_type == DetectorType.ISOLATION_FOREST:
                return IsolationForestDetector(config)
            elif config.detector_type == DetectorType.ONE_CLASS_SVM:
                return OneClassSVMDetector(config)
            elif config.detector_type == DetectorType.PHYSICS_BASED:
                return PhysicsBasedDetector(config)
            else:
                logger.warning(f"Unknown detector type: {config.detector_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to create detector {config.detector_type}: {e}")
            return None
    
    def train_detectors(self, training_data: np.ndarray):
        """Train all detectors on training data."""
        logger.info(f"Training {len(self.detectors)} detectors on {training_data.shape[0]} samples")
        
        for name, detector in self.detectors.items():
            try:
                detector.fit(training_data)
                logger.info(f"Successfully trained {name} detector")
            except Exception as e:
                logger.error(f"Failed to train {name} detector: {e}")
    
    def detect_anomalies(self, data: np.ndarray, 
                        feature_names: List[str] = None) -> List[AnomalyEvent]:
        """
        Detect anomalies in battery data.
        
        Args:
            data: Data to analyze (samples x features)
            feature_names: Names of features
            
        Returns:
            List of detected anomaly events
        """
        anomaly_events = []
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data.shape[1])]
        
        # Run each detector
        all_results = {}
        for name, detector in self.detectors.items():
            try:
                labels, scores = detector.detect(data)
                all_results[name] = (labels, scores)
            except Exception as e:
                logger.error(f"Detection failed for {name}: {e}")
                continue
        
        # Process results and create anomaly events
        for i in range(data.shape[0]):
            anomaly_detected = False
            max_score = 0
            detecting_methods = []
            
            # Check if any detector found an anomaly
            for detector_name, (labels, scores) in all_results.items():
                if labels[i]:
                    anomaly_detected = True
                    detecting_methods.append(detector_name)
                    max_score = max(max_score, scores[i])
            
            if anomaly_detected:
                # Create anomaly event
                event = self._create_anomaly_event(
                    data[i], feature_names, max_score, detecting_methods, i
                )
                anomaly_events.append(event)
                
                # Add to history
                self.anomaly_history.append(event)
        
        logger.info(f"Detected {len(anomaly_events)} anomalies")
        return anomaly_events
    
    def _create_anomaly_event(self, data_point: np.ndarray, 
                             feature_names: List[str],
                             confidence_score: float,
                             detecting_methods: List[str],
                             index: int) -> AnomalyEvent:
        """Create an anomaly event from detection results."""
        
        # Convert data point to dictionary
        data_dict = {name: float(value) for name, value in zip(feature_names, data_point)}
        
        # Determine anomaly type and affected features
        anomaly_type, affected_features = self._analyze_anomaly_type(data_dict)
        
        # Determine severity
        severity = self._determine_severity(confidence_score, affected_features)
        
        # Generate explanation
        explanation = self._generate_explanation(data_dict, affected_features, detecting_methods)
        
        # Generate suggested actions
        suggested_actions = self._generate_suggested_actions(anomaly_type, affected_features)
        
        return AnomalyEvent(
            anomaly_id=f"ANOM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index:04d}",
            timestamp=datetime.now(),
            data_point=data_dict,
            anomaly_type=anomaly_type,
            severity=severity,
            confidence_score=confidence_score,
            affected_features=affected_features,
            detector_name=", ".join(detecting_methods),
            explanation=explanation,
            suggested_actions=suggested_actions
        )
    
    def _analyze_anomaly_type(self, data_dict: Dict[str, float]) -> Tuple[AnomalyType, List[str]]:
        """Analyze the type of anomaly and affected features."""
        affected_features = []
        
        # Check for extreme values in different features
        if "voltage" in data_dict:
            voltage = data_dict["voltage"]
            if voltage < 2.5 or voltage > 4.3:
                affected_features.append("voltage")
        
        if "current" in data_dict:
            current = abs(data_dict["current"])
            if current > 500:  # High current
                affected_features.append("current")
        
        if "temperature" in data_dict:
            temperature = data_dict["temperature"]
            if temperature > 60 or temperature < -20:
                affected_features.append("temperature")
        
        if "soc" in data_dict:
            soc = data_dict["soc"]
            if soc < 0 or soc > 1:
                affected_features.append("soc")
        
        # Determine anomaly type based on affected features
        if len(affected_features) == 1:
            if "voltage" in affected_features:
                return AnomalyType.SPIKE_ANOMALY if data_dict["voltage"] > 4.0 else AnomalyType.DROP_ANOMALY, affected_features
            elif "temperature" in affected_features:
                return AnomalyType.SPIKE_ANOMALY, affected_features
            else:
                return AnomalyType.POINT_ANOMALY, affected_features
        elif len(affected_features) > 1:
            return AnomalyType.COLLECTIVE_ANOMALY, affected_features
        else:
            return AnomalyType.CONTEXTUAL_ANOMALY, ["unknown"]
    
    def _determine_severity(self, confidence_score: float, affected_features: List[str]) -> AnomalySeverity:
        """Determine severity based on confidence and affected features."""
        
        # Critical features that affect safety
        critical_features = ["voltage", "temperature", "current"]
        
        has_critical_feature = any(feature in critical_features for feature in affected_features)
        
        if confidence_score > 0.9 and has_critical_feature:
            return AnomalySeverity.CRITICAL
        elif confidence_score > 0.8 or has_critical_feature:
            return AnomalySeverity.HIGH
        elif confidence_score > 0.6:
            return AnomalySeverity.MEDIUM
        elif confidence_score > 0.4:
            return AnomalySeverity.LOW
        else:
            return AnomalySeverity.INFO
    
    def _generate_explanation(self, data_dict: Dict[str, float], 
                            affected_features: List[str],
                            detecting_methods: List[str]) -> str:
        """Generate human-readable explanation for the anomaly."""
        explanations = []
        
        for feature in affected_features:
            if feature in data_dict:
                value = data_dict[feature]
                
                if feature == "voltage":
                    if value > 4.2:
                        explanations.append(f"Voltage ({value:.2f}V) exceeds safe charging limit")
                    elif value < 2.5:
                        explanations.append(f"Voltage ({value:.2f}V) below critical discharge limit")
                
                elif feature == "temperature":
                    if value > 60:
                        explanations.append(f"Temperature ({value:.1f}°C) indicates overheating")
                    elif value < -20:
                        explanations.append(f"Temperature ({value:.1f}°C) below operating range")
                
                elif feature == "current":
                    if abs(value) > 500:
                        explanations.append(f"Current ({value:.1f}A) exceeds safe operating limits")
                
                elif feature == "soc":
                    if value > 1.0:
                        explanations.append(f"State of Charge ({value:.1%}) above physical maximum")
                    elif value < 0.0:
                        explanations.append(f"State of Charge ({value:.1%}) below physical minimum")
        
        if not explanations:
            explanations.append("Unusual pattern detected in battery data")
        
        explanation = ". ".join(explanations)
        explanation += f". Detected by: {', '.join(detecting_methods)}"
        
        return explanation
    
    def _generate_suggested_actions(self, anomaly_type: AnomalyType, 
                                  affected_features: List[str]) -> List[str]:
        """Generate suggested remedial actions."""
        actions = []
        
        if "voltage" in affected_features:
            actions.extend([
                "Check battery management system voltage measurements",
                "Verify charging/discharging circuits",
                "Inspect battery cell connections"
            ])
        
        if "temperature" in affected_features:
            actions.extend([
                "Check thermal management system",
                "Verify temperature sensor calibration",
                "Inspect cooling system operation"
            ])
        
        if "current" in affected_features:
            actions.extend([
                "Check current sensor calibration",
                "Verify load management system",
                "Inspect electrical connections"
            ])
        
        if "soc" in affected_features:
            actions.extend([
                "Recalibrate state of charge estimation",
                "Check coulomb counting accuracy",
                "Verify battery capacity measurements"
            ])
        
        # General actions
        actions.extend([
            "Log incident for trend analysis",
            "Consider preventive maintenance",
            "Monitor closely for pattern recurrence"
        ])
        
        return actions[:5]  # Limit to 5 most relevant actions
    
    def get_anomaly_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent anomalies."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        recent_anomalies = [
            event for event in self.anomaly_history 
            if event.timestamp >= cutoff_time
        ]
        
        if not recent_anomalies:
            return {
                "total_anomalies": 0,
                "time_window_hours": time_window_hours,
                "summary": "No anomalies detected in the specified time window"
            }
        
        # Count by severity
        severity_counts = {}
        for severity in AnomalySeverity:
            severity_counts[severity.value] = sum(
                1 for event in recent_anomalies 
                if event.severity == severity
            )
        
        # Count by type
        type_counts = {}
        for anomaly_type in AnomalyType:
            type_counts[anomaly_type.value] = sum(
                1 for event in recent_anomalies 
                if event.anomaly_type == anomaly_type
            )
        
        # Most affected features
        feature_counts = {}
        for event in recent_anomalies:
            for feature in event.affected_features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        return {
            "total_anomalies": len(recent_anomalies),
            "time_window_hours": time_window_hours,
            "severity_breakdown": severity_counts,
            "type_breakdown": type_counts,
            "most_affected_features": dict(sorted(feature_counts.items(), 
                                                key=lambda x: x[1], reverse=True)[:5]),
            "average_confidence": np.mean([event.confidence_score for event in recent_anomalies]),
            "critical_count": severity_counts.get("critical", 0),
            "high_count": severity_counts.get("high", 0)
        }
    
    def save_model(self, filepath: str):
        """Save trained detectors to file."""
        model_data = {
            "detectors": {},
            "detector_info": {},
            "save_timestamp": datetime.now().isoformat()
        }
        
        for name, detector in self.detectors.items():
            if detector.is_fitted:
                model_data["detectors"][name] = detector
                model_data["detector_info"][name] = detector.get_detector_info()
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved {len(model_data['detectors'])} trained detectors to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained detectors from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.detectors = model_data["detectors"]
        
        logger.info(f"Loaded {len(self.detectors)} trained detectors from {filepath}")

# Factory functions
def create_battery_anomaly_detector(detector_configs: List[DetectorConfig] = None) -> BatteryAnomalyDetector:
    """Create a battery anomaly detector instance."""
    return BatteryAnomalyDetector(detector_configs)

def detect_battery_anomalies(data: np.ndarray, 
                            training_data: np.ndarray = None,
                            feature_names: List[str] = None) -> List[AnomalyEvent]:
    """
    Convenience function to detect anomalies in battery data.
    
    Args:
        data: Data to analyze
        training_data: Training data for detectors (optional)
        feature_names: Feature names (optional)
        
    Returns:
        List of detected anomaly events
    """
    detector = create_battery_anomaly_detector()
    
    if training_data is not None:
        detector.train_detectors(training_data)
    else:
        # Use first 80% of data for training
        split_idx = int(0.8 * len(data))
        detector.train_detectors(data[:split_idx])
        data = data[split_idx:]  # Test on remaining data
    
    return detector.detect_anomalies(data, feature_names)
