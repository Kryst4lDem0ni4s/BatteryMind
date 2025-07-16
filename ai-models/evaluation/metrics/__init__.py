"""
BatteryMind - Evaluation Metrics Module

This module provides comprehensive evaluation metrics for all BatteryMind AI/ML models
including accuracy metrics, performance metrics, efficiency metrics, and business metrics.

The module supports evaluation of:
- Transformer-based battery health prediction models
- Federated learning aggregation models
- Reinforcement learning agents
- Ensemble models
- Time series forecasting models

Features:
- Domain-specific battery evaluation metrics
- Multi-objective evaluation framework
- Statistical significance testing
- Performance benchmarking
- Business impact assessment

Author: BatteryMind Development Team
Version: 1.0.0
"""

from .accuracy_metrics import (
    BatteryHealthMetrics,
    DegradationMetrics,
    SoHAccuracyMetrics,
    RULAccuracyMetrics,
    AnomalyDetectionMetrics,
    TimeSeriesMetrics,
    MultiModalMetrics,
    BatteryEvaluationSuite
)

from .performance_metrics import (
    InferenceSpeedMetrics,
    MemoryUsageMetrics,
    ThroughputMetrics,
    LatencyMetrics,
    ScalabilityMetrics,
    ResourceUtilizationMetrics,
    PerformanceProfiler
)

from .efficiency_metrics import (
    EnergyEfficiencyMetrics,
    BatteryLifeExtensionMetrics,
    ChargingEfficiencyMetrics,
    ThermalEfficiencyMetrics,
    CostEfficiencyMetrics,
    EfficiencyAnalyzer
)

from .business_metrics import (
    ROIMetrics,
    CostSavingsMetrics,
    MaintenanceReductionMetrics,
    BatteryLifetimeMetrics,
    FleetOptimizationMetrics,
    BusinessImpactCalculator
)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"

# Supported metric types
METRIC_TYPES = {
    'accuracy': [
        'soh_accuracy',
        'rul_accuracy',
        'degradation_accuracy',
        'anomaly_detection_accuracy',
        'time_series_accuracy'
    ],
    'performance': [
        'inference_speed',
        'memory_usage',
        'throughput',
        'latency',
        'scalability'
    ],
    'efficiency': [
        'energy_efficiency',
        'battery_life_extension',
        'charging_efficiency',
        'thermal_efficiency',
        'cost_efficiency'
    ],
    'business': [
        'roi_metrics',
        'cost_savings',
        'maintenance_reduction',
        'battery_lifetime',
        'fleet_optimization'
    ]
}

# Evaluation configurations
DEFAULT_EVALUATION_CONFIG = {
    'accuracy_thresholds': {
        'soh_threshold': 0.05,  # 5% SoH prediction error
        'rul_threshold_days': 30,  # 30 days RUL prediction error
        'degradation_threshold': 0.02,  # 2% degradation prediction error
        'anomaly_f1_threshold': 0.8,  # 80% F1 score for anomaly detection
        'time_series_mape_threshold': 0.1  # 10% MAPE for time series
    },
    'performance_thresholds': {
        'max_inference_time_ms': 100,  # 100ms max inference time
        'max_memory_usage_mb': 1000,  # 1GB max memory usage
        'min_throughput_samples_per_sec': 100,  # 100 samples/sec min throughput
        'max_latency_ms': 50,  # 50ms max latency
        'min_scalability_factor': 0.8  # 80% performance retention when scaled
    },
    'efficiency_thresholds': {
        'min_energy_efficiency': 0.85,  # 85% energy efficiency
        'min_battery_life_extension': 0.2,  # 20% battery life extension
        'min_charging_efficiency': 0.9,  # 90% charging efficiency
        'max_thermal_cost': 0.1,  # 10% thermal efficiency cost
        'min_cost_efficiency': 0.7  # 70% cost efficiency
    },
    'business_thresholds': {
        'min_roi_percentage': 0.15,  # 15% minimum ROI
        'min_cost_savings_percentage': 0.1,  # 10% cost savings
        'min_maintenance_reduction': 0.25,  # 25% maintenance reduction
        'min_battery_lifetime_extension': 0.3,  # 30% lifetime extension
        'min_fleet_optimization_gain': 0.2  # 20% fleet optimization gain
    }
}

# Utility functions
def create_evaluation_suite(config=None):
    """
    Create a comprehensive evaluation suite for BatteryMind models.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        BatteryEvaluationSuite: Configured evaluation suite
    """
    if config is None:
        config = DEFAULT_EVALUATION_CONFIG
    
    return BatteryEvaluationSuite(config)

def get_metric_by_name(metric_name: str, config=None):
    """
    Get a specific metric class by name.
    
    Args:
        metric_name: Name of the metric
        config: Optional configuration
        
    Returns:
        Metric class instance
    """
    metric_mapping = {
        'soh_accuracy': SoHAccuracyMetrics,
        'rul_accuracy': RULAccuracyMetrics,
        'degradation_accuracy': DegradationMetrics,
        'anomaly_detection': AnomalyDetectionMetrics,
        'time_series': TimeSeriesMetrics,
        'inference_speed': InferenceSpeedMetrics,
        'memory_usage': MemoryUsageMetrics,
        'throughput': ThroughputMetrics,
        'latency': LatencyMetrics,
        'energy_efficiency': EnergyEfficiencyMetrics,
        'battery_life_extension': BatteryLifeExtensionMetrics,
        'roi_metrics': ROIMetrics,
        'cost_savings': CostSavingsMetrics
    }
    
    if metric_name not in metric_mapping:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    return metric_mapping[metric_name](config)

def validate_evaluation_results(results: dict, thresholds: dict = None):
    """
    Validate evaluation results against predefined thresholds.
    
    Args:
        results: Evaluation results dictionary
        thresholds: Optional threshold configuration
        
    Returns:
        Dict with validation results
    """
    if thresholds is None:
        thresholds = DEFAULT_EVALUATION_CONFIG
    
    validation_results = {
        'passed': True,
        'failures': [],
        'warnings': [],
        'summary': {}
    }
    
    # Check accuracy thresholds
    accuracy_thresholds = thresholds.get('accuracy_thresholds', {})
    for metric, threshold in accuracy_thresholds.items():
        if metric in results:
            if results[metric] < threshold:
                validation_results['failures'].append(
                    f"{metric}: {results[metric]} < {threshold}"
                )
                validation_results['passed'] = False
    
    # Check performance thresholds
    performance_thresholds = thresholds.get('performance_thresholds', {})
    for metric, threshold in performance_thresholds.items():
        if metric in results:
            if 'max_' in metric and results[metric] > threshold:
                validation_results['failures'].append(
                    f"{metric}: {results[metric]} > {threshold}"
                )
                validation_results['passed'] = False
            elif 'min_' in metric and results[metric] < threshold:
                validation_results['failures'].append(
                    f"{metric}: {results[metric]} < {threshold}"
                )
                validation_results['passed'] = False
    
    validation_results['summary'] = {
        'total_metrics': len(results),
        'passed_metrics': len(results) - len(validation_results['failures']),
        'failed_metrics': len(validation_results['failures']),
        'pass_rate': (len(results) - len(validation_results['failures'])) / len(results) if results else 0
    }
    
    return validation_results

# Export all
__all__ = [
    # Accuracy metrics
    'BatteryHealthMetrics',
    'DegradationMetrics',
    'SoHAccuracyMetrics',
    'RULAccuracyMetrics',
    'AnomalyDetectionMetrics',
    'TimeSeriesMetrics',
    'MultiModalMetrics',
    
    # Performance metrics
    'InferenceSpeedMetrics',
    'MemoryUsageMetrics',
    'ThroughputMetrics',
    'LatencyMetrics',
    'ScalabilityMetrics',
    'ResourceUtilizationMetrics',
    
    # Efficiency metrics
    'EnergyEfficiencyMetrics',
    'BatteryLifeExtensionMetrics',
    'ChargingEfficiencyMetrics',
    'ThermalEfficiencyMetrics',
    'CostEfficiencyMetrics',
    
    # Business metrics
    'ROIMetrics',
    'CostSavingsMetrics',
    'MaintenanceReductionMetrics',
    'BatteryLifetimeMetrics',
    'FleetOptimizationMetrics',
    
    # Comprehensive evaluation
    'BatteryEvaluationSuite',
    'PerformanceProfiler',
    'EfficiencyAnalyzer',
    'BusinessImpactCalculator',
    
    # Utility functions
    'create_evaluation_suite',
    'get_metric_by_name',
    'validate_evaluation_results',
    
    # Constants
    'METRIC_TYPES',
    'DEFAULT_EVALUATION_CONFIG'
]
