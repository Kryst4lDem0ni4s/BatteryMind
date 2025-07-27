"""
BatteryMind - Data Monitoring Bias Detector
Advanced bias detection system for battery prediction models that identifies and 
quantifies various types of bias in data, model predictions, and system outcomes.

Features:
- Statistical bias detection across demographic and operational dimensions
- Fairness metrics calculation (demographic parity, equalized odds, etc.)
- Temporal bias detection for time-series battery data
- Geographic and environmental bias analysis
- Model prediction bias assessment
- Automated bias alerting and reporting
- Bias mitigation recommendations

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import threading
import time

# BatteryMind imports
from ..model_monitoring.drift_detector import DataDriftDetector
from ..alerts.alert_manager import AlertManager
from ...utils.logging_utils import setup_logger
from ...utils.statistical_utils import calculate_statistical_distance
from ...config.monitoring_config import MonitoringConfig

# Configure logging
logger = setup_logger(__name__)

class BiasType(Enum):
    """Types of bias that can be detected."""
    DEMOGRAPHIC = "demographic"
    TEMPORAL = "temporal"
    GEOGRAPHIC = "geographic"
    OPERATIONAL = "operational"
    PREDICTION = "prediction"
    SELECTION = "selection"
    PERFORMANCE = "performance"
    REPRESENTATION = "representation"

class FairnessMetric(Enum):
    """Fairness metrics for bias assessment."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUALITY_OF_OPPORTUNITY = "equality_of_opportunity"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"

@dataclass
class BiasThresholds:
    """Configurable thresholds for bias detection."""
    
    # Statistical significance thresholds
    p_value_threshold: float = 0.05
    effect_size_threshold: float = 0.2
    
    # Fairness metric thresholds
    demographic_parity_threshold: float = 0.1  # Maximum acceptable difference
    equalized_odds_threshold: float = 0.1
    calibration_threshold: float = 0.05
    
    # Distribution difference thresholds
    ks_statistic_threshold: float = 0.3
    chi_square_threshold: float = 0.05
    
    # Temporal bias thresholds
    trend_significance_threshold: float = 0.05
    seasonal_bias_threshold: float = 0.15
    
    # Performance bias thresholds
    accuracy_difference_threshold: float = 0.05
    precision_difference_threshold: float = 0.05
    recall_difference_threshold: float = 0.05

@dataclass
class BiasDetectionResult:
    """Result of bias detection analysis."""
    
    bias_type: BiasType
    metric_name: str
    bias_detected: bool
    bias_score: float
    p_value: float
    effect_size: float
    affected_groups: List[str]
    confidence_level: float
    
    # Detailed analysis
    group_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    distribution_analysis: Dict[str, Any] = field(default_factory=dict)
    fairness_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Temporal analysis (if applicable)
    temporal_trends: Optional[Dict[str, List[float]]] = None
    seasonal_patterns: Optional[Dict[str, Any]] = None
    
    # Recommendations
    severity: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    recommendations: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    
    # Metadata
    detection_timestamp: datetime = field(default_factory=datetime.now)
    data_size: int = 0
    feature_analyzed: Optional[str] = None

@dataclass
class BiasMonitoringConfig:
    """Configuration for bias monitoring system."""
    
    # Detection settings
    thresholds: BiasThresholds = field(default_factory=BiasThresholds)
    enabled_bias_types: List[BiasType] = field(default_factory=lambda: [
        BiasType.DEMOGRAPHIC, BiasType.TEMPORAL, BiasType.PREDICTION
    ])
    
    # Protected attributes for demographic bias
    protected_attributes: List[str] = field(default_factory=lambda: [
        'vehicle_type', 'geographic_region', 'user_segment', 'battery_chemistry'
    ])
    
    # Monitoring frequency
    monitoring_interval_hours: int = 24
    continuous_monitoring: bool = True
    
    # Reporting settings
    generate_reports: bool = True
    report_format: str = "json"  # json, html, pdf
    alert_on_bias: bool = True
    
    # Data requirements
    minimum_sample_size: int = 100
    minimum_group_size: int = 30
    confidence_level: float = 0.95

class BiasDetector:
    """
    Advanced bias detection system for battery prediction models.
    """
    
    def __init__(self, config: BiasMonitoringConfig):
        self.config = config
        self.alert_manager = AlertManager()
        self.drift_detector = DataDriftDetector()
        
        # Bias detection state
        self.detection_history: List[BiasDetectionResult] = []
        self.active_biases: Dict[str, BiasDetectionResult] = {}
        self.baseline_distributions: Dict[str, Dict[str, Any]] = {}
        
        # Threading for continuous monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        self.monitoring_lock = threading.Lock()
        
        logger.info("BiasDetector initialized")
    
    def detect_bias(self, 
                   data: pd.DataFrame,
                   predictions: Optional[np.ndarray] = None,
                   ground_truth: Optional[np.ndarray] = None,
                   protected_attribute: Optional[str] = None) -> List[BiasDetectionResult]:
        """
        Comprehensive bias detection across multiple dimensions.
        
        Args:
            data: Input data for analysis
            predictions: Model predictions (if available)
            ground_truth: True labels (if available)
            protected_attribute: Specific attribute to analyze for bias
            
        Returns:
            List of bias detection results
        """
        logger.info("Starting comprehensive bias detection")
        
        results = []
        
        # Validate input data
        if not self._validate_input_data(data):
            logger.error("Invalid input data for bias detection")
            return results
        
        # Demographic bias detection
        if BiasType.DEMOGRAPHIC in self.config.enabled_bias_types:
            demographic_results = self._detect_demographic_bias(
                data, predictions, ground_truth, protected_attribute
            )
            results.extend(demographic_results)
        
        # Temporal bias detection
        if BiasType.TEMPORAL in self.config.enabled_bias_types:
            temporal_results = self._detect_temporal_bias(data, predictions)
            results.extend(temporal_results)
        
        # Geographic bias detection
        if BiasType.GEOGRAPHIC in self.config.enabled_bias_types:
            geographic_results = self._detect_geographic_bias(data, predictions)
            results.extend(geographic_results)
        
        # Prediction bias detection
        if BiasType.PREDICTION in self.config.enabled_bias_types and predictions is not None:
            prediction_results = self._detect_prediction_bias(
                data, predictions, ground_truth
            )
            results.extend(prediction_results)
        
        # Performance bias detection
        if BiasType.PERFORMANCE in self.config.enabled_bias_types and ground_truth is not None:
            performance_results = self._detect_performance_bias(
                data, predictions, ground_truth
            )
            results.extend(performance_results)
        
        # Store results and generate alerts
        self._process_detection_results(results)
        
        logger.info(f"Bias detection completed. Found {len(results)} potential biases")
        
        return results
    
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for bias detection."""
        
        if data is None or data.empty:
            logger.error("Empty or None data provided")
            return False
        
        if len(data) < self.config.minimum_sample_size:
            logger.warning(f"Sample size {len(data)} below minimum {self.config.minimum_sample_size}")
            return False
        
        # Check for required columns
        required_columns = ['timestamp'] if 'timestamp' in self.config.protected_attributes else []
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
        
        return True
    
    def _detect_demographic_bias(self, 
                                data: pd.DataFrame,
                                predictions: Optional[np.ndarray] = None,
                                ground_truth: Optional[np.ndarray] = None,
                                protected_attribute: Optional[str] = None) -> List[BiasDetectionResult]:
        """Detect demographic bias across protected attributes."""
        
        results = []
        
        # Determine which attributes to analyze
        attrs_to_analyze = [protected_attribute] if protected_attribute else self.config.protected_attributes
        attrs_to_analyze = [attr for attr in attrs_to_analyze if attr in data.columns]
        
        for attr in attrs_to_analyze:
            logger.info(f"Analyzing demographic bias for attribute: {attr}")
            
            # Group data by protected attribute
            groups = data.groupby(attr)
            
            # Skip if insufficient group sizes
            if any(len(group) < self.config.minimum_group_size for _, group in groups):
                logger.warning(f"Insufficient group sizes for {attr}")
                continue
            
            # Statistical bias detection
            stat_result = self._analyze_statistical_bias(groups, attr)
            if stat_result:
                results.append(stat_result)
            
            # Fairness metrics (if predictions available)
            if predictions is not None:
                fairness_result = self._analyze_fairness_metrics(
                    groups, predictions, ground_truth, attr
                )
                if fairness_result:
                    results.append(fairness_result)
        
        return results
    
    def _analyze_statistical_bias(self, groups, attribute: str) -> Optional[BiasDetectionResult]:
        """Analyze statistical bias between groups."""
        
        try:
            group_names = list(groups.groups.keys())
            group_data = [group for _, group in groups]
            
            # Select numerical columns for analysis
            numerical_cols = groups.obj.select_dtypes(include=[np.number]).columns
            
            bias_scores = []
            p_values = []
            group_stats = {}
            
            for col in numerical_cols:
                if col == attribute:  # Skip the grouping attribute itself
                    continue
                
                # Extract column data for each group
                column_groups = [group[col].dropna() for group in group_data]
                
                # Skip if any group is too small
                if any(len(grp) < 10 for grp in column_groups):
                    continue
                
                # Perform Kolmogorov-Smirnov test between first two groups
                if len(column_groups) >= 2:
                    ks_stat, p_val = ks_2samp(column_groups[0], column_groups[1])
                    bias_scores.append(ks_stat)
                    p_values.append(p_val)
                    
                    # Calculate group statistics
                    for i, (group_name, group_vals) in enumerate(zip(group_names, column_groups)):
                        if group_name not in group_stats:
                            group_stats[group_name] = {}
                        
                        group_stats[group_name][col] = {
                            'mean': float(group_vals.mean()),
                            'std': float(group_vals.std()),
                            'median': float(group_vals.median()),
                            'size': len(group_vals)
                        }
            
            if not bias_scores:
                return None
            
            # Aggregate bias metrics
            avg_bias_score = np.mean(bias_scores)
            min_p_value = np.min(p_values)
            
            # Determine if bias exists
            bias_detected = (avg_bias_score > self.config.thresholds.ks_statistic_threshold or 
                           min_p_value < self.config.thresholds.p_value_threshold)
            
            # Calculate effect size (Cohen's d)
            effect_size = self._calculate_effect_size(group_data[0], group_data[1])
            
            return BiasDetectionResult(
                bias_type=BiasType.DEMOGRAPHIC,
                metric_name=f"demographic_bias_{attribute}",
                bias_detected=bias_detected,
                bias_score=avg_bias_score,
                p_value=min_p_value,
                effect_size=effect_size,
                affected_groups=group_names,
                confidence_level=self.config.confidence_level,
                group_statistics=group_stats,
                severity=self._determine_severity(avg_bias_score, min_p_value),
                recommendations=self._generate_demographic_recommendations(attribute, group_stats),
                data_size=sum(len(grp) for grp in group_data),
                feature_analyzed=attribute
            )
            
        except Exception as e:
            logger.error(f"Error analyzing statistical bias for {attribute}: {e}")
            return None
    
    def _calculate_effect_size(self, group1: pd.DataFrame, group2: pd.DataFrame) -> float:
        """Calculate Cohen's d effect size between groups."""
        
        try:
            # Use first numerical column for effect size calculation
            num_cols = group1.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                return 0.0
            
            col = num_cols[0]
            
            vals1 = group1[col].dropna()
            vals2 = group2[col].dropna()
            
            if len(vals1) == 0 or len(vals2) == 0:
                return 0.0
            
            # Cohen's d calculation
            pooled_std = np.sqrt(((len(vals1) - 1) * vals1.var() + 
                                (len(vals2) - 1) * vals2.var()) / 
                               (len(vals1) + len(vals2) - 2))
            
            if pooled_std == 0:
                return 0.0
            
            effect_size = (vals1.mean() - vals2.mean()) / pooled_std
            return abs(effect_size)
            
        except Exception as e:
            logger.warning(f"Error calculating effect size: {e}")
            return 0.0
    
    def _analyze_fairness_metrics(self, 
                                 groups,
                                 predictions: np.ndarray,
                                 ground_truth: Optional[np.ndarray],
                                 attribute: str) -> Optional[BiasDetectionResult]:
        """Analyze fairness metrics for predictions."""
        
        try:
            group_names = list(groups.groups.keys())
            fairness_metrics = {}
            
            # Demographic Parity
            demo_parity = self._calculate_demographic_parity(groups, predictions)
            fairness_metrics['demographic_parity'] = demo_parity
            
            # Equalized Odds (if ground truth available)
            if ground_truth is not None:
                eq_odds = self._calculate_equalized_odds(groups, predictions, ground_truth)
                fairness_metrics['equalized_odds'] = eq_odds
                
                # Equality of Opportunity
                eq_opp = self._calculate_equality_of_opportunity(groups, predictions, ground_truth)
                fairness_metrics['equality_of_opportunity'] = eq_opp
            
            # Determine bias based on fairness metrics
            bias_detected = (
                demo_parity > self.config.thresholds.demographic_parity_threshold or
                fairness_metrics.get('equalized_odds', 0) > self.config.thresholds.equalized_odds_threshold
            )
            
            bias_score = max(demo_parity, fairness_metrics.get('equalized_odds', 0))
            
            return BiasDetectionResult(
                bias_type=BiasType.DEMOGRAPHIC,
                metric_name=f"fairness_bias_{attribute}",
                bias_detected=bias_detected,
                bias_score=bias_score,
                p_value=0.0,  # Not applicable for fairness metrics
                effect_size=bias_score,
                affected_groups=group_names,
                confidence_level=self.config.confidence_level,
                fairness_metrics=fairness_metrics,
                severity=self._determine_fairness_severity(fairness_metrics),
                recommendations=self._generate_fairness_recommendations(attribute, fairness_metrics),
                data_size=len(predictions),
                feature_analyzed=attribute
            )
            
        except Exception as e:
            logger.error(f"Error analyzing fairness metrics for {attribute}: {e}")
            return None
    
    def _calculate_demographic_parity(self, groups, predictions: np.ndarray) -> float:
        """Calculate demographic parity difference."""
        
        group_rates = []
        
        for name, group in groups:
            group_indices = group.index
            group_predictions = predictions[group_indices]
            
            # Assuming binary classification (threshold at 0.5)
            positive_rate = np.mean(group_predictions > 0.5)
            group_rates.append(positive_rate)
        
        # Return maximum difference between groups
        return max(group_rates) - min(group_rates) if group_rates else 0.0
    
    def _calculate_equalized_odds(self, groups, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate equalized odds difference."""
        
        tpr_differences = []
        fpr_differences = []
        
        group_tprs = []
        group_fprs = []
        
        for name, group in groups:
            group_indices = group.index
            group_preds = predictions[group_indices] > 0.5
            group_truth = ground_truth[group_indices]
            
            # Calculate TPR and FPR
            tn, fp, fn, tp = confusion_matrix(group_truth, group_preds).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            group_tprs.append(tpr)
            group_fprs.append(fpr)
        
        # Return maximum difference in TPR and FPR
        tpr_diff = max(group_tprs) - min(group_tprs) if group_tprs else 0
        fpr_diff = max(group_fprs) - min(group_fprs) if group_fprs else 0
        
        return max(tpr_diff, fpr_diff)
    
    def _calculate_equality_of_opportunity(self, groups, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate equality of opportunity difference."""
        
        group_tprs = []
        
        for name, group in groups:
            group_indices = group.index
            group_preds = predictions[group_indices] > 0.5
            group_truth = ground_truth[group_indices]
            
            # Calculate TPR for positive class only
            positive_mask = group_truth == 1
            if np.sum(positive_mask) > 0:
                tpr = np.mean(group_preds[positive_mask])
                group_tprs.append(tpr)
        
        # Return maximum difference in TPR
        return max(group_tprs) - min(group_tprs) if len(group_tprs) > 1 else 0.0
    
    def _detect_temporal_bias(self, 
                             data: pd.DataFrame,
                             predictions: Optional[np.ndarray] = None) -> List[BiasDetectionResult]:
        """Detect temporal bias in data and predictions."""
        
        results = []
        
        if 'timestamp' not in data.columns:
            logger.warning("No timestamp column found for temporal bias detection")
            return results
        
        try:
            # Convert timestamp if needed
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Sort by timestamp
            data_sorted = data.sort_values('timestamp')
            
            # Detect trend bias
            trend_result = self._detect_trend_bias(data_sorted, predictions)
            if trend_result:
                results.append(trend_result)
            
            # Detect seasonal bias
            seasonal_result = self._detect_seasonal_bias(data_sorted, predictions)
            if seasonal_result:
                results.append(seasonal_result)
            
            # Detect concept drift bias
            drift_result = self._detect_concept_drift_bias(data_sorted, predictions)
            if drift_result:
                results.append(drift_result)
            
        except Exception as e:
            logger.error(f"Error in temporal bias detection: {e}")
        
        return results
    
    def _detect_trend_bias(self, data: pd.DataFrame, predictions: Optional[np.ndarray]) -> Optional[BiasDetectionResult]:
        """Detect systematic trends that indicate bias."""
        
        try:
            # Select numerical columns for trend analysis
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            
            trend_statistics = {}
            significant_trends = []
            
            for col in numerical_cols:
                if col == 'timestamp':
                    continue
                
                # Calculate trend using linear regression
                x = np.arange(len(data))
                y = data[col].fillna(data[col].mean())
                
                # Pearson correlation for trend
                correlation, p_value = stats.pearsonr(x, y)
                
                trend_statistics[col] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'slope': correlation * (y.std() / len(y))
                }
                
                # Check if trend is significant
                if abs(correlation) > 0.3 and p_value < self.config.thresholds.trend_significance_threshold:
                    significant_trends.append(col)
            
            if not significant_trends:
                return None
            
            # Calculate overall trend bias score
            bias_score = np.mean([abs(trend_statistics[col]['correlation']) 
                                for col in significant_trends])
            
            min_p_value = np.min([trend_statistics[col]['p_value'] 
                                for col in significant_trends])
            
            return BiasDetectionResult(
                bias_type=BiasType.TEMPORAL,
                metric_name="temporal_trend_bias",
                bias_detected=True,
                bias_score=bias_score,
                p_value=min_p_value,
                effect_size=bias_score,
                affected_groups=significant_trends,
                confidence_level=self.config.confidence_level,
                temporal_trends=trend_statistics,
                severity=self._determine_severity(bias_score, min_p_value),
                recommendations=self._generate_temporal_recommendations(significant_trends),
                data_size=len(data)
            )
            
        except Exception as e:
            logger.error(f"Error detecting trend bias: {e}")
            return None
    
    def _detect_seasonal_bias(self, data: pd.DataFrame, predictions: Optional[np.ndarray]) -> Optional[BiasDetectionResult]:
        """Detect seasonal bias patterns."""
        
        try:
            # Extract time features
            data = data.copy()
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['month'] = data['timestamp'].dt.month
            
            seasonal_bias_detected = False
            seasonal_analysis = {}
            
            # Analyze hourly patterns
            hourly_groups = data.groupby('hour')
            hourly_bias = self._analyze_seasonal_groups(hourly_groups, 'hour')
            seasonal_analysis['hourly'] = hourly_bias
            
            if hourly_bias['bias_score'] > self.config.thresholds.seasonal_bias_threshold:
                seasonal_bias_detected = True
            
            # Analyze weekly patterns
            weekly_groups = data.groupby('day_of_week')
            weekly_bias = self._analyze_seasonal_groups(weekly_groups, 'day_of_week')
            seasonal_analysis['weekly'] = weekly_bias
            
            if weekly_bias['bias_score'] > self.config.thresholds.seasonal_bias_threshold:
                seasonal_bias_detected = True
            
            # Analyze monthly patterns
            monthly_groups = data.groupby('month')
            monthly_bias = self._analyze_seasonal_groups(monthly_groups, 'month')
            seasonal_analysis['monthly'] = monthly_bias
            
            if monthly_bias['bias_score'] > self.config.thresholds.seasonal_bias_threshold:
                seasonal_bias_detected = True
            
            if not seasonal_bias_detected:
                return None
            
            # Calculate overall seasonal bias score
            bias_score = max(
                hourly_bias['bias_score'],
                weekly_bias['bias_score'],
                monthly_bias['bias_score']
            )
            
            return BiasDetectionResult(
                bias_type=BiasType.TEMPORAL,
                metric_name="seasonal_bias",
                bias_detected=seasonal_bias_detected,
                bias_score=bias_score,
                p_value=0.05,  # Conservative estimate
                effect_size=bias_score,
                affected_groups=['time_based_patterns'],
                confidence_level=self.config.confidence_level,
                seasonal_patterns=seasonal_analysis,
                severity=self._determine_severity(bias_score, 0.05),
                recommendations=self._generate_seasonal_recommendations(seasonal_analysis),
                data_size=len(data)
            )
            
        except Exception as e:
            logger.error(f"Error detecting seasonal bias: {e}")
            return None
    
    def _analyze_seasonal_groups(self, groups, time_unit: str) -> Dict[str, Any]:
        """Analyze bias within seasonal groups."""
        
        group_stats = {}
        bias_scores = []
        
        # Select first numerical column for analysis
        numerical_cols = groups.obj.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) == 0:
            return {'bias_score': 0.0, 'group_stats': {}}
        
        analysis_col = numerical_cols[0]
        
        for name, group in groups:
            values = group[analysis_col].dropna()
            if len(values) > 5:  # Minimum group size
                group_stats[str(name)] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'size': len(values)
                }
        
        # Calculate coefficient of variation as bias metric
        if len(group_stats) > 1:
            means = [stats['mean'] for stats in group_stats.values()]
            cv = np.std(means) / np.mean(means) if np.mean(means) != 0 else 0
            bias_score = cv
        else:
            bias_score = 0.0
        
        return {
            'bias_score': bias_score,
            'group_stats': group_stats,
            'analysis_column': analysis_col
        }
    
    def _detect_concept_drift_bias(self, data: pd.DataFrame, predictions: Optional[np.ndarray]) -> Optional[BiasDetectionResult]:
        """Detect concept drift that introduces bias."""
        
        try:
            # Split data into time windows
            window_size = len(data) // 4  # 4 time windows
            if window_size < self.config.minimum_sample_size:
                return None
            
            windows = [
                data.iloc[i*window_size:(i+1)*window_size] 
                for i in range(4)
            ]
            
            # Use drift detector
            drift_results = []
            
            for i in range(len(windows) - 1):
                drift_score = self.drift_detector.detect_drift(
                    windows[0],  # Reference window
                    windows[i+1]  # Comparison window
                )
                drift_results.append(drift_score)
            
            # Check if significant drift exists
            max_drift = max(drift_results) if drift_results else 0
            
            if max_drift > 0.3:  # Significant drift threshold
                return BiasDetectionResult(
                    bias_type=BiasType.TEMPORAL,
                    metric_name="concept_drift_bias",
                    bias_detected=True,
                    bias_score=max_drift,
                    p_value=0.01,  # Assume significant
                    effect_size=max_drift,
                    affected_groups=['temporal_segments'],
                    confidence_level=self.config.confidence_level,
                    distribution_analysis={'drift_scores': drift_results},
                    severity=self._determine_severity(max_drift, 0.01),
                    recommendations=self._generate_drift_recommendations(),
                    data_size=len(data)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting concept drift bias: {e}")
            return None
    
    def _detect_geographic_bias(self, data: pd.DataFrame, predictions: Optional[np.ndarray]) -> List[BiasDetectionResult]:
        """Detect geographic bias in data distribution and predictions."""
        
        results = []
        
        # Look for geographic indicators
        geo_columns = [col for col in data.columns 
                      if any(geo_term in col.lower() for geo_term in 
                           ['region', 'location', 'geographic', 'city', 'state', 'country', 'zone'])]
        
        if not geo_columns:
            logger.info("No geographic columns found for bias detection")
            return results
        
        for geo_col in geo_columns:
            try:
                # Group by geographic attribute
                geo_groups = data.groupby(geo_col)
                
                # Skip if insufficient groups
                if len(geo_groups) < 2:
                    continue
                
                # Analyze geographic distribution bias
                geo_result = self._analyze_geographic_distribution(geo_groups, geo_col, predictions)
                if geo_result:
                    results.append(geo_result)
                    
            except Exception as e:
                logger.error(f"Error analyzing geographic bias for {geo_col}: {e}")
        
        return results
    
    def _analyze_geographic_distribution(self, groups, geo_column: str, predictions: Optional[np.ndarray]) -> Optional[BiasDetectionResult]:
        """Analyze geographic distribution bias."""
        
        try:
            group_stats = {}
            bias_indicators = []
            
            # Analyze data distribution across geographic groups
            for name, group in groups:
                if len(group) < self.config.minimum_group_size:
                    continue
                
                # Calculate group statistics
                numerical_cols = group.select_dtypes(include=[np.number]).columns
                
                group_stats[str(name)] = {}
                for col in numerical_cols:
                    values = group[col].dropna()
                    if len(values) > 0:
                        group_stats[str(name)][col] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'size': len(values)
                        }
            
            # Calculate geographic bias score
            if len(group_stats) < 2:
                return None
            
            # Use coefficient of variation across regions as bias metric
            bias_scores = []
            
            for col in numerical_cols:
                col_means = []
                for region_stats in group_stats.values():
                    if col in region_stats:
                        col_means.append(region_stats[col]['mean'])
                
                if len(col_means) > 1:
                    cv = np.std(col_means) / np.mean(col_means) if np.mean(col_means) != 0 else 0
                    bias_scores.append(cv)
            
            if not bias_scores:
                return None
            
            avg_bias_score = np.mean(bias_scores)
            
            # Determine if bias is significant
            bias_detected = avg_bias_score > 0.2  # 20% coefficient of variation threshold
            
            return BiasDetectionResult(
                bias_type=BiasType.GEOGRAPHIC,
                metric_name=f"geographic_bias_{geo_column}",
                bias_detected=bias_detected,
                bias_score=avg_bias_score,
                p_value=0.05,  # Conservative estimate
                effect_size=avg_bias_score,
                affected_groups=list(group_stats.keys()),
                confidence_level=self.config.confidence_level,
                group_statistics=group_stats,
                severity=self._determine_severity(avg_bias_score, 0.05),
                recommendations=self._generate_geographic_recommendations(geo_column),
                data_size=sum(stats.get('size', 0) for region_stats in group_stats.values() 
                             for stats in region_stats.values()),
                feature_analyzed=geo_column
            )
            
        except Exception as e:
            logger.error(f"Error analyzing geographic distribution bias: {e}")
            return None
    
    def _detect_prediction_bias(self, 
                               data: pd.DataFrame,
                               predictions: np.ndarray,
                               ground_truth: Optional[np.ndarray] = None) -> List[BiasDetectionResult]:
        """Detect bias in model predictions."""
        
        results = []
        
        try:
            # Analyze prediction distribution bias
            pred_dist_result = self._analyze_prediction_distribution(predictions)
            if pred_dist_result:
                results.append(pred_dist_result)
            
            # Analyze calibration bias (if ground truth available)
            if ground_truth is not None:
                calibration_result = self._analyze_calibration_bias(predictions, ground_truth)
                if calibration_result:
                    results.append(calibration_result)
            
            # Analyze prediction variance bias
            variance_result = self._analyze_prediction_variance(predictions, data)
            if variance_result:
                results.append(variance_result)
                
        except Exception as e:
            logger.error(f"Error detecting prediction bias: {e}")
        
        return results
    
    def _analyze_prediction_distribution(self, predictions: np.ndarray) -> Optional[BiasDetectionResult]:
        """Analyze bias in prediction distributions."""
        
        try:
            # Test for normality and uniformity
            if len(predictions) < 20:
                return None
            
            # Shapiro-Wilk test for normality
            shapiro_stat, shapiro_p = stats.shapiro(predictions[:5000])  # Limit for performance
            
            # Anderson-Darling test
            anderson_result = stats.anderson(predictions)
            
            # Check for extreme skewness
            skewness = stats.skew(predictions)
            kurtosis = stats.kurtosis(predictions)
            
            # Determine bias based on distribution properties
            bias_detected = (
                abs(skewness) > 2.0 or  # High skewness
                abs(kurtosis) > 7.0 or  # High kurtosis
                shapiro_p < 0.01  # Non-normal distribution
            )
            
            bias_score = abs(skewness) + abs(kurtosis) / 10  # Combined metric
            
            if not bias_detected:
                return None
            
            return BiasDetectionResult(
                bias_type=BiasType.PREDICTION,
                metric_name="prediction_distribution_bias",
                bias_detected=bias_detected,
                bias_score=bias_score,
                p_value=shapiro_p,
                effect_size=abs(skewness),
                affected_groups=['prediction_distribution'],
                confidence_level=self.config.confidence_level,
                distribution_analysis={
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'shapiro_statistic': shapiro_stat,
                    'anderson_statistic': anderson_result.statistic
                },
                severity=self._determine_severity(bias_score, shapiro_p),
                recommendations=self._generate_prediction_recommendations(skewness, kurtosis),
                data_size=len(predictions)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing prediction distribution: {e}")
            return None
    
    def _analyze_calibration_bias(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Optional[BiasDetectionResult]:
        """Analyze calibration bias in predictions."""
        
        try:
            # Create calibration bins
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_errors = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find predictions in this bin
                in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    # Calculate accuracy in this bin
                    accuracy_in_bin = ground_truth[in_bin].mean()
                    avg_confidence_in_bin = predictions[in_bin].mean()
                    
                    # Calibration error
                    calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                    calibration_errors.append(calibration_error)
            
            # Expected Calibration Error (ECE)
            ece = np.mean(calibration_errors) if calibration_errors else 0
            
            # Determine if calibration bias exists
            bias_detected = ece > self.config.thresholds.calibration_threshold
            
            if not bias_detected:
                return None
            
            return BiasDetectionResult(
                bias_type=BiasType.PREDICTION,
                metric_name="calibration_bias",
                bias_detected=bias_detected,
                bias_score=ece,
                p_value=0.01,  # Assume significant
                effect_size=ece,
                affected_groups=['model_calibration'],
                confidence_level=self.config.confidence_level,
                fairness_metrics={'expected_calibration_error': ece},
                severity=self._determine_severity(ece, 0.01),
                recommendations=self._generate_calibration_recommendations(ece),
                data_size=len(predictions)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing calibration bias: {e}")
            return None
    
    def _analyze_prediction_variance(self, predictions: np.ndarray, data: pd.DataFrame) -> Optional[BiasDetectionResult]:
        """Analyze variance bias in predictions across different groups."""
        
        try:
            # Use protected attributes to analyze variance
            variance_results = {}
            
            for attr in self.config.protected_attributes:
                if attr not in data.columns:
                    continue
                
                groups = data.groupby(attr)
                group_variances = []
                
                for name, group in groups:
                    if len(group) < 10:  # Minimum group size
                        continue
                    
                    group_indices = group.index
                    group_predictions = predictions[group_indices]
                    variance = np.var(group_predictions)
                    group_variances.append(variance)
                
                if len(group_variances) > 1:
                    # Calculate coefficient of variation of variances
                    variance_cv = np.std(group_variances) / np.mean(group_variances) if np.mean(group_variances) > 0 else 0
                    variance_results[attr] = variance_cv
            
            if not variance_results:
                return None
            
            # Find maximum variance bias
            max_variance_bias = max(variance_results.values())
            
            # Determine if significant
            bias_detected = max_variance_bias > 0.5  # 50% threshold
            
            if not bias_detected:
                return None
            
            most_biased_attr = max(variance_results, key=variance_results.get)
            
            return BiasDetectionResult(
                bias_type=BiasType.PREDICTION,
                metric_name="prediction_variance_bias",
                bias_detected=bias_detected,
                bias_score=max_variance_bias,
                p_value=0.05,  # Conservative estimate
                effect_size=max_variance_bias,
                affected_groups=[most_biased_attr],
                confidence_level=self.config.confidence_level,
                distribution_analysis=variance_results,
                severity=self._determine_severity(max_variance_bias, 0.05),
                recommendations=self._generate_variance_recommendations(most_biased_attr),
                data_size=len(predictions),
                feature_analyzed=most_biased_attr
            )
            
        except Exception as e:
            logger.error(f"Error analyzing prediction variance bias: {e}")
            return None
    
    def _detect_performance_bias(self, 
                                data: pd.DataFrame,
                                predictions: np.ndarray,
                                ground_truth: np.ndarray) -> List[BiasDetectionResult]:
        """Detect performance bias across different groups."""
        
        results = []
        
        for attr in self.config.protected_attributes:
            if attr not in data.columns:
                continue
            
            try:
                # Group by protected attribute
                groups = data.groupby(attr)
                
                # Calculate performance metrics for each group
                group_performance = {}
                
                for name, group in groups:
                    if len(group) < self.config.minimum_group_size:
                        continue
                    
                    group_indices = group.index
                    group_preds = predictions[group_indices]
                    group_truth = ground_truth[group_indices]
                    
                    # Calculate metrics
                    accuracy = np.mean(group_preds.round() == group_truth)
                    
                    # For continuous predictions, use MAE and RMSE
                    mae = np.mean(np.abs(group_preds - group_truth))
                    rmse = np.sqrt(np.mean((group_preds - group_truth) ** 2))
                    
                    group_performance[str(name)] = {
                        'accuracy': accuracy,
                        'mae': mae,
                        'rmse': rmse,
                        'size': len(group)
                    }
                
                if len(group_performance) < 2:
                    continue
                
                # Analyze performance differences
                performance_result = self._analyze_performance_differences(
                    group_performance, attr
                )
                
                if performance_result:
                    results.append(performance_result)
                    
            except Exception as e:
                logger.error(f"Error detecting performance bias for {attr}: {e}")
        
        return results
    
    def _analyze_performance_differences(self, group_performance: Dict[str, Dict[str, float]], attribute: str) -> Optional[BiasDetectionResult]:
        """Analyze performance differences between groups."""
        
        try:
            # Extract performance metrics
            accuracies = [perf['accuracy'] for perf in group_performance.values()]
            maes = [perf['mae'] for perf in group_performance.values()]
            rmses = [perf['rmse'] for perf in group_performance.values()]
            
            # Calculate differences
            accuracy_diff = max(accuracies) - min(accuracies)
            mae_diff = max(maes) - min(maes)
            rmse_diff = max(rmses) - min(rmses)
            
            # Determine if bias exists
            bias_detected = (
                accuracy_diff > self.config.thresholds.accuracy_difference_threshold or
                mae_diff > 0.1 or  # 10% MAE difference threshold
                rmse_diff > 0.15   # 15% RMSE difference threshold
            )
            
            if not bias_detected:
                return None
            
            # Calculate overall bias score
            bias_score = max(accuracy_diff, mae_diff / 2, rmse_diff / 3)
            
            return BiasDetectionResult(
                bias_type=BiasType.PERFORMANCE,
                metric_name=f"performance_bias_{attribute}",
                bias_detected=bias_detected,
                bias_score=bias_score,
                p_value=0.05,  # Conservative estimate
                effect_size=bias_score,
                affected_groups=list(group_performance.keys()),
                confidence_level=self.config.confidence_level,
                group_statistics=group_performance,
                fairness_metrics={
                    'accuracy_difference': accuracy_diff,
                    'mae_difference': mae_diff,
                    'rmse_difference': rmse_diff
                },
                severity=self._determine_severity(bias_score, 0.05),
                recommendations=self._generate_performance_recommendations(attribute, group_performance),
                data_size=sum(perf['size'] for perf in group_performance.values()),
                feature_analyzed=attribute
            )
            
        except Exception as e:
            logger.error(f"Error analyzing performance differences: {e}")
            return None
    
    def _determine_severity(self, bias_score: float, p_value: float) -> str:
        """Determine bias severity level."""
        
        if bias_score > 0.8 or p_value < 0.001:
            return "CRITICAL"
        elif bias_score > 0.5 or p_value < 0.01:
            return "HIGH"
        elif bias_score > 0.3 or p_value < 0.05:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _determine_fairness_severity(self, fairness_metrics: Dict[str, float]) -> str:
        """Determine severity based on fairness metrics."""
        
        max_unfairness = max(fairness_metrics.values()) if fairness_metrics else 0
        
        if max_unfairness > 0.5:
            return "CRITICAL"
        elif max_unfairness > 0.3:
            return "HIGH"
        elif max_unfairness > 0.15:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_demographic_recommendations(self, attribute: str, group_stats: Dict[str, Dict]) -> List[str]:
        """Generate recommendations for addressing demographic bias."""
        
        recommendations = [
            f"Investigate data collection process for {attribute} to ensure representative sampling",
            f"Consider stratified sampling to balance representation across {attribute} groups",
            "Implement fairness constraints during model training",
            "Use bias mitigation techniques such as reweighting or adversarial debiasing",
            f"Monitor {attribute} distribution in production data continuously"
        ]
        
        # Add specific recommendations based on group statistics
        if group_stats:
            group_sizes = [stats.get('size', 0) for group_stats_dict in group_stats.values() 
                          for stats in group_stats_dict.values()]
            
            if len(group_sizes) > 1 and max(group_sizes) / min(group_sizes) > 3:
                recommendations.append("Address class imbalance through oversampling minority groups")
        
        return recommendations
    
    def _generate_fairness_recommendations(self, attribute: str, fairness_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving fairness."""
        
        recommendations = [
            "Implement post-processing calibration to improve fairness metrics",
            "Consider using fairness-aware algorithms during model development",
            "Apply threshold optimization separately for each protected group",
            "Implement fairness constraints in the optimization objective"
        ]
        
        # Specific recommendations based on metrics
        if fairness_metrics.get('demographic_parity', 0) > 0.2:
            recommendations.append("Apply demographic parity constraint during training")
        
        if fairness_metrics.get('equalized_odds', 0) > 0.2:
            recommendations.append("Implement equalized odds post-processing")
        
        return recommendations
    
    def _generate_temporal_recommendations(self, affected_features: List[str]) -> List[str]:
        """Generate recommendations for temporal bias."""
        
        return [
            "Implement temporal cross-validation to account for time-based patterns",
            "Consider using time-aware features or temporal embeddings",
            "Apply trend debiasing techniques to the affected features",
            "Monitor model performance across different time periods",
            "Implement model retraining schedule to adapt to temporal changes",
            f"Investigate root causes of trends in features: {', '.join(affected_features)}"
        ]
    
    def _generate_seasonal_recommendations(self, seasonal_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for seasonal bias."""
        
        recommendations = [
            "Implement seasonal normalization for time-sensitive features",
            "Consider using cyclical encoding for temporal features",
            "Apply seasonal decomposition to remove bias patterns",
            "Ensure training data covers all seasonal periods adequately"
        ]
        
        # Add specific recommendations based on analysis
        for time_unit, analysis in seasonal_analysis.items():
            if analysis.get('bias_score', 0) > 0.3:
                recommendations.append(f"Address significant {time_unit} bias patterns")
        
        return recommendations
    
    def _generate_drift_recommendations(self) -> List[str]:
        """Generate recommendations for concept drift bias."""
        
        return [
            "Implement online learning or incremental model updates",
            "Set up automated model retraining pipeline",
            "Monitor for concept drift continuously",
            "Use adaptive algorithms that can handle distribution shifts",
            "Implement windowed training to focus on recent data",
            "Consider ensemble methods that adapt to drift"
        ]
    
    def _generate_geographic_recommendations(self, geo_column: str) -> List[str]:
        """Generate recommendations for geographic bias."""
        
        return [
            f"Ensure balanced geographic representation in training data for {geo_column}",
            "Consider geographic stratification during data collection",
            "Implement location-aware features or embeddings",
            "Apply geographic normalization techniques",
            "Monitor model performance across different geographic regions",
            "Consider separate models for different geographic segments if warranted"
        ]
    
    def _generate_prediction_recommendations(self, skewness: float, kurtosis: float) -> List[str]:
        """Generate recommendations for prediction bias."""
        
        recommendations = []
        
        if abs(skewness) > 2:
            recommendations.append("Apply output transformation to reduce prediction skewness")
            recommendations.append("Consider using different loss functions that handle skewed distributions")
        
        if abs(kurtosis) > 7:
            recommendations.append("Investigate outliers in predictions and ground truth")
            recommendations.append("Consider robust training techniques")
        
        recommendations.extend([
            "Implement prediction calibration techniques",
            "Monitor prediction distribution regularly",
            "Use ensemble methods to improve prediction reliability"
        ])
        
        return recommendations
    
    def _generate_calibration_recommendations(self, ece: float) -> List[str]:
        """Generate recommendations for calibration bias."""
        
        return [
            "Apply Platt scaling or isotonic regression for calibration",
            "Implement temperature scaling for neural network outputs",
            "Use calibration-aware training objectives",
            "Monitor calibration metrics regularly",
            f"Current ECE of {ece:.3f} indicates poor calibration - prioritize calibration improvement",
            "Consider using calibrated classifiers from scikit-learn"
        ]
    
    def _generate_variance_recommendations(self, attribute: str) -> List[str]:
        """Generate recommendations for prediction variance bias."""
        
        return [
            f"Investigate prediction variance differences across {attribute} groups",
            "Consider using heteroscedastic models that account for varying uncertainty",
            "Implement group-specific uncertainty quantification",
            "Apply variance regularization techniques during training",
            "Monitor prediction confidence across different groups",
            "Consider ensemble methods to stabilize predictions"
        ]
    
    def _generate_performance_recommendations(self, attribute: str, group_performance: Dict[str, Dict]) -> List[str]:
        """Generate recommendations for performance bias."""
        
        recommendations = [
            f"Address performance disparities across {attribute} groups",
            "Implement group-specific model tuning or separate models",
            "Apply fairness-aware training techniques",
            "Consider cost-sensitive learning to balance performance",
            "Implement group-specific evaluation metrics"
        ]
        
        # Identify worst-performing group
        worst_group = min(group_performance.keys(), 
                         key=lambda x: group_performance[x]['accuracy'])
        
        recommendations.append(f"Focus improvement efforts on {worst_group} group with lowest performance")
        
        return recommendations
    
    def _process_detection_results(self, results: List[BiasDetectionResult]) -> None:
        """Process bias detection results and generate alerts."""
        
        # Store results in history
        self.detection_history.extend(results)
        
        # Update active biases
        for result in results:
            if result.bias_detected:
                key = f"{result.bias_type.value}_{result.metric_name}"
                self.active_biases[key] = result
                
                # Generate alert for significant biases
                if result.severity in ['HIGH', 'CRITICAL'] and self.config.alert_on_bias:
                    self._generate_bias_alert(result)
    
    def _generate_bias_alert(self, result: BiasDetectionResult) -> None:
        """Generate alert for detected bias."""
        
        try:
            alert_data = {
                'alert_type': 'BIAS_DETECTED',
                'severity': result.severity,
                'bias_type': result.bias_type.value,
                'metric_name': result.metric_name,
                'bias_score': result.bias_score,
                'affected_groups': result.affected_groups,
                'recommendations': result.recommendations[:3],  # Top 3 recommendations
                'timestamp': result.detection_timestamp.isoformat(),
                'data_size': result.data_size
            }
            
            self.alert_manager.create_alert(
                title=f"Bias Detected: {result.metric_name}",
                message=f"{result.bias_type.value.title()} bias detected with score {result.bias_score:.3f}",
                severity=result.severity,
                metadata=alert_data
            )
            
            logger.warning(f"Bias alert generated: {result.metric_name} (Severity: {result.severity})")
            
        except Exception as e:
            logger.error(f"Error generating bias alert: {e}")
    
    def start_continuous_monitoring(self) -> None:
        """Start continuous bias monitoring."""
        
        if self.is_monitoring:
            logger.warning("Bias monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Continuous bias monitoring started")
    
    def stop_continuous_monitoring(self) -> None:
        """Stop continuous bias monitoring."""
        
        self.is_monitoring = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Continuous bias monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for continuous bias detection."""
        
        while self.is_monitoring:
            try:
                # This would be implemented to fetch recent data and run bias detection
                # For now, just sleep
                time.sleep(self.config.monitoring_interval_hours * 3600)
                
                logger.info("Continuous bias monitoring check completed")
                
            except Exception as e:
                logger.error(f"Error in bias monitoring loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def get_bias_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of bias detection results."""
        
        # Filter recent results
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_results = [
            result for result in self.detection_history
            if result.detection_timestamp > cutoff_date
        ]
        
        # Calculate summary statistics
        total_checks = len(recent_results)
        biases_detected = len([r for r in recent_results if r.bias_detected])
        
        # Group by bias type
        bias_by_type = {}
        for result in recent_results:
            bias_type = result.bias_type.value
            if bias_type not in bias_by_type:
                bias_by_type[bias_type] = {'total': 0, 'detected': 0}
            
            bias_by_type[bias_type]['total'] += 1
            if result.bias_detected:
                bias_by_type[bias_type]['detected'] += 1
        
        # Severity distribution
        severity_dist = {}
        for result in recent_results:
            if result.bias_detected:
                severity = result.severity
                severity_dist[severity] = severity_dist.get(severity, 0) + 1
        
        return {
            'summary_period_days': days,
            'total_bias_checks': total_checks,
            'biases_detected': biases_detected,
            'bias_detection_rate': biases_detected / max(total_checks, 1),
            'bias_by_type': bias_by_type,
            'severity_distribution': severity_dist,
            'active_biases': len(self.active_biases),
            'most_common_bias_type': max(bias_by_type.keys(), 
                                       key=lambda x: bias_by_type[x]['detected']) if bias_by_type else None,
            'last_check': max([r.detection_timestamp for r in recent_results]).isoformat() if recent_results else None
        }
    
    def generate_bias_report(self, output_format: str = "json") -> str:
        """Generate comprehensive bias report."""
        
        summary = self.get_bias_summary()
        
        report_data = {
            'report_timestamp': datetime.now().isoformat(),
            'configuration': {
                'thresholds': self.config.thresholds.__dict__,
                'enabled_bias_types': [bt.value for bt in self.config.enabled_bias_types],
                'protected_attributes': self.config.protected_attributes
            },
            'summary': summary,
            'active_biases': {
                key: {
                    'bias_type': result.bias_type.value,
                    'metric_name': result.metric_name,
                    'bias_score': result.bias_score,
                    'severity': result.severity,
                    'affected_groups': result.affected_groups,
                    'detection_timestamp': result.detection_timestamp.isoformat()
                }
                for key, result in self.active_biases.items()
            },
            'recent_detections': [
                {
                    'bias_type': result.bias_type.value,
                    'metric_name': result.metric_name,
                    'bias_detected': result.bias_detected,
                    'bias_score': result.bias_score,
                    'severity': result.severity,
                    'timestamp': result.detection_timestamp.isoformat()
                }
                for result in self.detection_history[-50:]  # Last 50 results
            ]
        }
        
        if output_format.lower() == "json":
            return json.dumps(report_data, indent=2)
        else:
            # Could implement HTML/PDF generation here
            return json.dumps(report_data, indent=2)
    
    def clear_detection_history(self, days: Optional[int] = None) -> None:
        """Clear bias detection history."""
        
        if days is None:
            # Clear all history
            self.detection_history.clear()
            self.active_biases.clear()
            logger.info("All bias detection history cleared")
        else:
            # Clear history older than specified days
            cutoff_date = datetime.now() - timedelta(days=days)
            
            self.detection_history = [
                result for result in self.detection_history
                if result.detection_timestamp > cutoff_date
            ]
            
            # Clear old active biases
            old_keys = [
                key for key, result in self.active_biases.items()
                if result.detection_timestamp <= cutoff_date
            ]
            
            for key in old_keys:
                del self.active_biases[key]
            
            logger.info(f"Bias detection history older than {days} days cleared")

# Factory function for easy instantiation
def create_bias_detector(config: Optional[BiasMonitoringConfig] = None) -> BiasDetector:
    """Create bias detector with default or custom configuration."""
    
    if config is None:
        config = BiasMonitoringConfig()
    
    return BiasDetector(config)
