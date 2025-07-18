"""
BatteryMind - AWS SageMaker Monitoring Module

Comprehensive monitoring solution for SageMaker endpoints that tracks model performance,
data drift, system health, and business metrics specific to battery prediction workloads.

Features:
- Real-time model performance monitoring
- Data drift detection using statistical methods
- Custom battery-specific metrics tracking
- Integration with CloudWatch, X-Ray, and custom dashboards
- Automated alerting and notification system
- Model quality degradation detection
- Cost and resource utilization monitoring

Author: BatteryMind Development Team
Version: 1.0.0
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# BatteryMind imports
from .endpoint_config import SageMakerEndpointConfig
from ...monitoring.model_monitoring.drift_detector import DataDriftDetector
from ...monitoring.alerts.alert_manager import AlertManager
from ...utils.logging_utils import setup_logger
from ...utils.aws_helpers import AWSHelper

# Configure logging
logger = setup_logger(__name__)

@dataclass
class MonitoringConfig:
    """Configuration for SageMaker endpoint monitoring."""
    
    # Data drift monitoring
    drift_detection_enabled: bool = True
    drift_threshold: float = 0.1
    drift_check_interval_minutes: int = 60
    
    # Model performance monitoring
    accuracy_threshold: float = 0.95
    latency_threshold_ms: float = 1000.0
    error_rate_threshold: float = 0.05
    
    # Battery-specific thresholds
    soh_prediction_accuracy_threshold: float = 0.98
    degradation_forecast_accuracy_threshold: float = 0.95
    anomaly_detection_accuracy_threshold: float = 0.90
    
    # System monitoring
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    disk_threshold: float = 90.0
    
    # Cost monitoring
    cost_alert_threshold: float = 100.0  # USD per day
    cost_spike_threshold: float = 2.0    # 2x normal cost
    
    # Alerting
    enable_email_alerts: bool = True
    enable_slack_alerts: bool = True
    alert_severity_levels: List[str] = field(default_factory=lambda: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])

@dataclass
class ModelMetrics:
    """Model performance metrics."""
    
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mae: float = 0.0  # Mean Absolute Error
    rmse: float = 0.0  # Root Mean Square Error
    
    # Battery-specific metrics
    soh_accuracy: float = 0.0
    degradation_accuracy: float = 0.0
    anomaly_detection_accuracy: float = 0.0
    
    # Confidence metrics
    prediction_confidence: float = 0.0
    uncertainty_score: float = 0.0

@dataclass
class SystemMetrics:
    """System performance metrics."""
    
    invocations_per_minute: float = 0.0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_rate: float = 0.0
    
    # Resource utilization
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    disk_utilization: float = 0.0
    network_in_mb: float = 0.0
    network_out_mb: float = 0.0
    
    # Cost metrics
    hourly_cost: float = 0.0
    daily_cost: float = 0.0

class SageMakerMonitor:
    """
    Comprehensive monitoring system for SageMaker endpoints.
    """
    
    def __init__(self, config: MonitoringConfig, aws_region: str = 'us-west-2'):
        self.config = config
        self.aws_region = aws_region
        
        # Initialize AWS clients
        self.sagemaker_client = boto3.client('sagemaker', region_name=aws_region)
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=aws_region)
        self.logs_client = boto3.client('logs', region_name=aws_region)
        self.xray_client = boto3.client('xray', region_name=aws_region)
        
        # Initialize components
        self.aws_helper = AWSHelper()
        self.drift_detector = DataDriftDetector()
        self.alert_manager = AlertManager()
        
        # Monitoring state
        self.baseline_metrics = {}
        self.monitoring_history = []
        self.drift_alerts = []
        
        logger.info("SageMaker Monitor initialized")
    
    def setup_monitoring(self, endpoint_name: str, 
                        variant_name: str = 'AllTraffic') -> bool:
        """
        Set up comprehensive monitoring for a SageMaker endpoint.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            variant_name: Name of the production variant
            
        Returns:
            Success status
        """
        try:
            # Create CloudWatch dashboard
            self._create_dashboard(endpoint_name)
            
            # Set up custom metrics
            self._setup_custom_metrics(endpoint_name, variant_name)
            
            # Configure data capture for drift detection
            self._setup_data_capture(endpoint_name)
            
            # Create monitoring alarms
            self._create_monitoring_alarms(endpoint_name, variant_name)
            
            logger.info(f"Monitoring setup completed for endpoint: {endpoint_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup monitoring for {endpoint_name}: {e}")
            return False
    
    def _create_dashboard(self, endpoint_name: str):
        """Create CloudWatch dashboard for the endpoint."""
        
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "Invocations", "EndpointName", endpoint_name],
                            [".", "ModelLatency", ".", "."],
                            [".", "Model4XXErrors", ".", "."],
                            [".", "Model5XXErrors", ".", "."]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.aws_region,
                        "title": "Endpoint Performance",
                        "period": 300
                    }
                },
                {
                    "type": "metric",
                    "x": 12, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["BatteryMind/Models", "SOH_Accuracy", "EndpointName", endpoint_name],
                            [".", "Degradation_Accuracy", ".", "."],
                            [".", "Anomaly_Detection_Accuracy", ".", "."],
                            [".", "Prediction_Confidence", ".", "."]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.aws_region,
                        "title": "Model Quality Metrics",
                        "period": 300
                    }
                },
                {
                    "type": "metric",
                    "x": 0, "y": 6, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["BatteryMind/Monitoring", "Data_Drift_Score", "EndpointName", endpoint_name],
                            [".", "Feature_Drift_Count", ".", "."],
                            [".", "Distribution_Shift", ".", "."]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.aws_region,
                        "title": "Data Drift Detection",
                        "period": 300
                    }
                },
                {
                    "type": "metric",
                    "x": 12, "y": 6, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["BatteryMind/Cost", "Hourly_Cost", "EndpointName", endpoint_name],
                            [".", "Daily_Cost", ".", "."],
                            [".", "Cost_Per_Invocation", ".", "."]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.aws_region,
                        "title": "Cost Monitoring",
                        "period": 3600
                    }
                }
            ]
        }
        
        try:
            self.cloudwatch_client.put_dashboard(
                DashboardName=f'BatteryMind-{endpoint_name}',
                DashboardBody=json.dumps(dashboard_body)
            )
            
            logger.info(f"Created CloudWatch dashboard for {endpoint_name}")
            
        except ClientError as e:
            logger.error(f"Failed to create dashboard: {e}")
    
    def _setup_custom_metrics(self, endpoint_name: str, variant_name: str):
        """Set up custom metrics specific to battery predictions."""
        
        # This would be called by the prediction endpoint to publish custom metrics
        custom_metrics = [
            'SOH_Accuracy',
            'Degradation_Accuracy', 
            'Anomaly_Detection_Accuracy',
            'Prediction_Confidence',
            'Data_Drift_Score',
            'Feature_Drift_Count',
            'Distribution_Shift',
            'Hourly_Cost',
            'Daily_Cost',
            'Cost_Per_Invocation'
        ]
        
        logger.info(f"Custom metrics configured: {custom_metrics}")
    
    def _setup_data_capture(self, endpoint_name: str):
        """Configure data capture for the endpoint."""
        
        try:
            # Enable data capture on the endpoint
            data_capture_config = {
                'EnableCapture': True,
                'InitialSamplingPercentage': 100,
                'DestinationS3Uri': f's3://batterymind-data-capture/{endpoint_name}/',
                'CaptureOptions': [
                    {'CaptureMode': 'Input'},
                    {'CaptureMode': 'Output'}
                ]
            }
            
            # This would be applied during endpoint creation/update
            logger.info(f"Data capture configured for {endpoint_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup data capture: {e}")
    
    def _create_monitoring_alarms(self, endpoint_name: str, variant_name: str):
        """Create comprehensive monitoring alarms."""
        
        alarms = [
            # Model performance alarms
            {
                'AlarmName': f'{endpoint_name}-low-soh-accuracy',
                'ComparisonOperator': 'LessThanThreshold',
                'EvaluationPeriods': 2,
                'MetricName': 'SOH_Accuracy',
                'Namespace': 'BatteryMind/Models',
                'Period': 600,
                'Statistic': 'Average',
                'Threshold': self.config.soh_prediction_accuracy_threshold,
                'ActionsEnabled': True,
                'AlarmDescription': 'SOH prediction accuracy below threshold',
                'Dimensions': [{'Name': 'EndpointName', 'Value': endpoint_name}]
            },
            
            # Data drift alarms
            {
                'AlarmName': f'{endpoint_name}-data-drift-detected',
                'ComparisonOperator': 'GreaterThanThreshold',
                'EvaluationPeriods': 1,
                'MetricName': 'Data_Drift_Score',
                'Namespace': 'BatteryMind/Monitoring',
                'Period': 3600,
                'Statistic': 'Maximum',
                'Threshold': self.config.drift_threshold,
                'ActionsEnabled': True,
                'AlarmDescription': 'Data drift detected in input features',
                'Dimensions': [{'Name': 'EndpointName', 'Value': endpoint_name}]
            },
            
            # Cost alarms
            {
                'AlarmName': f'{endpoint_name}-high-cost',
                'ComparisonOperator': 'GreaterThanThreshold',
                'EvaluationPeriods': 1,
                'MetricName': 'Daily_Cost',
                'Namespace': 'BatteryMind/Cost',
                'Period': 3600,
                'Statistic': 'Maximum',
                'Threshold': self.config.cost_alert_threshold,
                'ActionsEnabled': True,
                'AlarmDescription': 'Daily cost exceeds threshold',
                'Dimensions': [{'Name': 'EndpointName', 'Value': endpoint_name}]
            },
            
            # System performance alarms
            {
                'AlarmName': f'{endpoint_name}-high-latency',
                'ComparisonOperator': 'GreaterThanThreshold',
                'EvaluationPeriods': 3,
                'MetricName': 'ModelLatency',
                'Namespace': 'AWS/SageMaker',
                'Period': 300,
                'Statistic': 'Average',
                'Threshold': self.config.latency_threshold_ms,
                'ActionsEnabled': True,
                'AlarmDescription': 'Model latency exceeds threshold',
                'Dimensions': [
                    {'Name': 'EndpointName', 'Value': endpoint_name},
                    {'Name': 'VariantName', 'Value': variant_name}
                ]
            }
        ]
        
        for alarm in alarms:
            try:
                self.cloudwatch_client.put_metric_alarm(**alarm)
                logger.info(f"Created alarm: {alarm['AlarmName']}")
                
            except ClientError as e:
                logger.error(f"Failed to create alarm {alarm['AlarmName']}: {e}")
    
    def collect_metrics(self, endpoint_name: str, 
                       variant_name: str = 'AllTraffic') -> Dict[str, Any]:
        """
        Collect comprehensive metrics for the endpoint.
        
        Returns:
            Dictionary containing model and system metrics
        """
        try:
            # Collect system metrics
            system_metrics = self._collect_system_metrics(endpoint_name, variant_name)
            
            # Collect model metrics
            model_metrics = self._collect_model_metrics(endpoint_name, variant_name)
            
            # Check for data drift
            drift_metrics = self._check_data_drift(endpoint_name)
            
            # Calculate cost metrics
            cost_metrics = self._calculate_cost_metrics(endpoint_name)
            
            metrics = {
                'timestamp': datetime.utcnow().isoformat(),
                'endpoint_name': endpoint_name,
                'variant_name': variant_name,
                'system_metrics': system_metrics,
                'model_metrics': model_metrics,
                'drift_metrics': drift_metrics,
                'cost_metrics': cost_metrics
            }
            
            # Store metrics history
            self.monitoring_history.append(metrics)
            
            # Trim history to last 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.monitoring_history = [
                m for m in self.monitoring_history 
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics for {endpoint_name}: {e}")
            return {}
    
    def _collect_system_metrics(self, endpoint_name: str, variant_name: str) -> SystemMetrics:
        """Collect system performance metrics."""
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)
        
        metrics = SystemMetrics()
        
        # CloudWatch metrics to collect
        cloudwatch_metrics = [
            ('Invocations', 'Sum', 'invocations_per_minute'),
            ('ModelLatency', 'Average', 'average_latency_ms'),
            ('Model4XXErrors', 'Sum', 'error_4xx'),
            ('Model5XXErrors', 'Sum', 'error_5xx')
        ]
        
        for metric_name, statistic, attr_name in cloudwatch_metrics:
            try:
                response = self.cloudwatch_client.get_metric_statistics(
                    Namespace='AWS/SageMaker',
                    MetricName=metric_name,
                    Dimensions=[
                        {'Name': 'EndpointName', 'Value': endpoint_name},
                        {'Name': 'VariantName', 'Value': variant_name}
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=300,
                    Statistics=[statistic]
                )
                
                if response['Datapoints']:
                    latest_point = max(response['Datapoints'], key=lambda x: x['Timestamp'])
                    setattr(metrics, attr_name, latest_point[statistic])
                    
            except Exception as e:
                logger.warning(f"Failed to get metric {metric_name}: {e}")
        
        # Calculate error rate
        total_errors = getattr(metrics, 'error_4xx', 0) + getattr(metrics, 'error_5xx', 0)
        total_invocations = max(metrics.invocations_per_minute, 1)
        metrics.error_rate = total_errors / total_invocations
        
        return metrics
    
    def _collect_model_metrics(self, endpoint_name: str, variant_name: str) -> ModelMetrics:
        """Collect model performance metrics."""
        
        metrics = ModelMetrics()
        
        # Get custom metrics from CloudWatch
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=15)
        
        custom_metrics = [
            'SOH_Accuracy',
            'Degradation_Accuracy',
            'Anomaly_Detection_Accuracy',
            'Prediction_Confidence'
        ]
        
        for metric_name in custom_metrics:
            try:
                response = self.cloudwatch_client.get_metric_statistics(
                    Namespace='BatteryMind/Models',
                    MetricName=metric_name,
                    Dimensions=[{'Name': 'EndpointName', 'Value': endpoint_name}],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=900,
                    Statistics=['Average']
                )
                
                if response['Datapoints']:
                    latest_point = max(response['Datapoints'], key=lambda x: x['Timestamp'])
                    attr_name = metric_name.lower().replace('_', '_')
                    setattr(metrics, attr_name, latest_point['Average'])
                    
            except Exception as e:
                logger.warning(f"Failed to get custom metric {metric_name}: {e}")
        
        return metrics
    
    def _check_data_drift(self, endpoint_name: str) -> Dict[str, Any]:
        """Check for data drift in the input features."""
        
        drift_results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'drifted_features': [],
            'drift_severity': 'LOW'
        }
        
        try:
            # Get recent prediction data from S3 (captured data)
            recent_data = self._get_recent_prediction_data(endpoint_name)
            
            if recent_data is not None and len(recent_data) > 100:
                # Compare with baseline data
                baseline_data = self.baseline_metrics.get(endpoint_name, {}).get('baseline_data')
                
                if baseline_data is not None:
                    # Use statistical tests to detect drift
                    drift_score, drifted_features = self.drift_detector.detect_drift(
                        baseline_data, recent_data
                    )
                    
                    drift_results.update({
                        'drift_detected': drift_score > self.config.drift_threshold,
                        'drift_score': drift_score,
                        'drifted_features': drifted_features,
                        'drift_severity': self._calculate_drift_severity(drift_score)
                    })
                    
                    # Publish drift metrics to CloudWatch
                    self._publish_drift_metrics(endpoint_name, drift_results)
                    
        except Exception as e:
            logger.error(f"Failed to check data drift: {e}")
        
        return drift_results
    
    def _get_recent_prediction_data(self, endpoint_name: str) -> Optional[pd.DataFrame]:
        """Get recent prediction data from S3 data capture."""
        
        # This would download and parse data from S3
        # For now, return None to indicate no data available
        return None
    
    def _calculate_drift_severity(self, drift_score: float) -> str:
        """Calculate drift severity based on score."""
        
        if drift_score > 0.5:
            return 'CRITICAL'
        elif drift_score > 0.3:
            return 'HIGH'
        elif drift_score > 0.15:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _publish_drift_metrics(self, endpoint_name: str, drift_results: Dict[str, Any]):
        """Publish drift detection metrics to CloudWatch."""
        
        try:
            self.cloudwatch_client.put_metric_data(
                Namespace='BatteryMind/Monitoring',
                MetricData=[
                    {
                        'MetricName': 'Data_Drift_Score',
                        'Dimensions': [{'Name': 'EndpointName', 'Value': endpoint_name}],
                        'Value': drift_results['drift_score'],
                        'Unit': 'None'
                    },
                    {
                        'MetricName': 'Feature_Drift_Count',
                        'Dimensions': [{'Name': 'EndpointName', 'Value': endpoint_name}],
                        'Value': len(drift_results['drifted_features']),
                        'Unit': 'Count'
                    }
                ]
            )
            
        except Exception as e:
            logger.error(f"Failed to publish drift metrics: {e}")
    
    def _calculate_cost_metrics(self, endpoint_name: str) -> Dict[str, float]:
        """Calculate cost metrics for the endpoint."""
        
        cost_metrics = {
            'hourly_cost': 0.0,
            'daily_cost': 0.0,
            'cost_per_invocation': 0.0,
            'cost_trend': 'stable'
        }
        
        try:
            # Get endpoint configuration for cost calculation
            endpoint_config = self.sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            
            total_hourly_cost = 0.0
            
            for variant in endpoint_config['ProductionVariants']:
                instance_type = variant['InstanceType']
                instance_count = variant['CurrentInstanceCount']
                
                # Get hourly cost per instance (simplified)
                hourly_cost_per_instance = self._get_instance_hourly_cost(instance_type)
                variant_cost = hourly_cost_per_instance * instance_count
                total_hourly_cost += variant_cost
            
            cost_metrics['hourly_cost'] = total_hourly_cost
            cost_metrics['daily_cost'] = total_hourly_cost * 24
            
            # Calculate cost per invocation
            recent_metrics = self._collect_system_metrics(endpoint_name, 'AllTraffic')
            if recent_metrics.invocations_per_minute > 0:
                invocations_per_hour = recent_metrics.invocations_per_minute * 60
                cost_metrics['cost_per_invocation'] = total_hourly_cost / invocations_per_hour
            
            # Determine cost trend
            cost_metrics['cost_trend'] = self._analyze_cost_trend(endpoint_name)
            
            # Publish cost metrics to CloudWatch
            self._publish_cost_metrics(endpoint_name, cost_metrics)
            
        except Exception as e:
            logger.error(f"Failed to calculate cost metrics: {e}")
        
        return cost_metrics
    
    def _get_instance_hourly_cost(self, instance_type: str) -> float:
        """Get hourly cost for instance type."""
        # Simplified cost mapping - should integrate with AWS Pricing API
        cost_map = {
            'ml.t2.medium': 0.056,
            'ml.m5.large': 0.115,
            'ml.m5.xlarge': 0.230,
            'ml.c5.large': 0.102,
            'ml.c5.xlarge': 0.204,
            'ml.c5.2xlarge': 0.408,
            'ml.g4dn.xlarge': 0.736
        }
        return cost_map.get(instance_type, 0.115)
    
    def _analyze_cost_trend(self, endpoint_name: str) -> str:
        """Analyze cost trend over time."""
        
        if len(self.monitoring_history) < 3:
            return 'stable'
        
        # Get recent cost data
        recent_costs = []
        for record in self.monitoring_history[-6:]:  # Last 6 records
            if record['endpoint_name'] == endpoint_name:
                cost = record.get('cost_metrics', {}).get('hourly_cost', 0.0)
                recent_costs.append(cost)
        
        if len(recent_costs) < 3:
            return 'stable'
        
        # Simple trend analysis
        recent_avg = np.mean(recent_costs[-3:])
        earlier_avg = np.mean(recent_costs[:-3])
        
        if recent_avg > earlier_avg * 1.2:
            return 'increasing'
        elif recent_avg < earlier_avg * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _publish_cost_metrics(self, endpoint_name: str, cost_metrics: Dict[str, float]):
        """Publish cost metrics to CloudWatch."""
        
        try:
            metric_data = []
            
            for metric_name, value in cost_metrics.items():
                if isinstance(value, (int, float)):
                    metric_data.append({
                        'MetricName': metric_name.replace('_', '_').title(),
                        'Dimensions': [{'Name': 'EndpointName', 'Value': endpoint_name}],
                        'Value': value,
                        'Unit': 'None'
                    })
            
            if metric_data:
                self.cloudwatch_client.put_metric_data(
                    Namespace='BatteryMind/Cost',
                    MetricData=metric_data
                )
                
        except Exception as e:
            logger.error(f"Failed to publish cost metrics: {e}")
    
    def generate_health_report(self, endpoint_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive health report for the endpoint.
        
        Returns:
            Health report with status and recommendations
        """
        try:
            # Collect current metrics
            current_metrics = self.collect_metrics(endpoint_name)
            
            health_report = {
                'endpoint_name': endpoint_name,
                'timestamp': datetime.utcnow().isoformat(),
                'overall_health': 'HEALTHY',
                'health_score': 100,
                'issues': [],
                'recommendations': [],
                'metrics_summary': current_metrics
            }
            
            # Analyze health indicators
            health_issues = []
            health_score = 100
            
            # Check system performance
            system_metrics = current_metrics.get('system_metrics', SystemMetrics())
            
            if system_metrics.error_rate > self.config.error_rate_threshold:
                health_issues.append({
                    'severity': 'HIGH',
                    'category': 'SYSTEM',
                    'description': f'High error rate: {system_metrics.error_rate:.2%}',
                    'recommendation': 'Investigate error logs and consider scaling up'
                })
                health_score -= 20
            
            if system_metrics.average_latency_ms > self.config.latency_threshold_ms:
                health_issues.append({
                    'severity': 'MEDIUM',
                    'category': 'PERFORMANCE',
                    'description': f'High latency: {system_metrics.average_latency_ms:.1f}ms',
                    'recommendation': 'Consider instance scaling or optimization'
                })
                health_score -= 15
            
            # Check model performance
            model_metrics = current_metrics.get('model_metrics', ModelMetrics())
            
            if model_metrics.soh_accuracy < self.config.soh_prediction_accuracy_threshold:
                health_issues.append({
                    'severity': 'HIGH',
                    'category': 'MODEL',
                    'description': f'Low SOH accuracy: {model_metrics.soh_accuracy:.2%}',
                    'recommendation': 'Model retraining may be required'
                })
                health_score -= 25
            
            # Check data drift
            drift_metrics = current_metrics.get('drift_metrics', {})
            if drift_metrics.get('drift_detected', False):
                severity = 'HIGH' if drift_metrics.get('drift_severity') == 'CRITICAL' else 'MEDIUM'
                health_issues.append({
                    'severity': severity,
                    'category': 'DATA',
                    'description': f'Data drift detected: {drift_metrics.get("drift_score", 0):.3f}',
                    'recommendation': 'Review data pipeline and consider model retraining'
                })
                health_score -= 20 if severity == 'HIGH' else 10
            
            # Check costs
            cost_metrics = current_metrics.get('cost_metrics', {})
            if cost_metrics.get('daily_cost', 0) > self.config.cost_alert_threshold:
                health_issues.append({
                    'severity': 'MEDIUM',
                    'category': 'COST',
                    'description': f'High daily cost: ${cost_metrics.get("daily_cost", 0):.2f}',
                    'recommendation': 'Review instance types and scaling policies'
                })
                health_score -= 10
            
            # Determine overall health
            health_report['health_score'] = max(0, health_score)
            health_report['issues'] = health_issues
            
            if health_score >= 90:
                health_report['overall_health'] = 'HEALTHY'
            elif health_score >= 70:
                health_report['overall_health'] = 'WARNING'
            elif health_score >= 50:
                health_report['overall_health'] = 'DEGRADED'
            else:
                health_report['overall_health'] = 'CRITICAL'
            
            # Generate recommendations
            recommendations = self._generate_recommendations(health_issues, current_metrics)
            health_report['recommendations'] = recommendations
            
            return health_report
            
        except Exception as e:
            logger.error(f"Failed to generate health report for {endpoint_name}: {e}")
            return {
                'endpoint_name': endpoint_name,
                'overall_health': 'UNKNOWN',
                'error': str(e)
            }
    
    def _generate_recommendations(self, health_issues: List[Dict[str, Any]], 
                                 metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on health issues."""
        
        recommendations = []
        
        # Performance recommendations
        system_metrics = metrics.get('system_metrics', SystemMetrics())
        
        if system_metrics.average_latency_ms > self.config.latency_threshold_ms:
            recommendations.append("Consider upgrading instance types or scaling out")
        
        if system_metrics.error_rate > self.config.error_rate_threshold:
            recommendations.append("Investigate application logs and error patterns")
        
        # Model recommendations
        model_metrics = metrics.get('model_metrics', ModelMetrics())
        
        if model_metrics.soh_accuracy < self.config.soh_prediction_accuracy_threshold:
            recommendations.append("Schedule model retraining with recent data")
        
        # Data drift recommendations
        drift_metrics = metrics.get('drift_metrics', {})
        if drift_metrics.get('drift_detected', False):
            recommendations.append("Review data pipeline and feature engineering")
            recommendations.append("Consider model adaptation or retraining")
        
        # Cost optimization recommendations
        cost_metrics = metrics.get('cost_metrics', {})
        if cost_metrics.get('daily_cost', 0) > self.config.cost_alert_threshold:
            recommendations.append("Review instance utilization and scaling policies")
            recommendations.append("Consider Spot instances for non-critical workloads")
        
        return recommendations
    
    def get_monitoring_history(self, endpoint_name: Optional[str] = None, 
                              hours: int = 24) -> List[Dict[str, Any]]:
        """Get monitoring history for analysis."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        filtered_history = [
            record for record in self.monitoring_history
            if datetime.fromisoformat(record['timestamp']) > cutoff_time
        ]
        
        if endpoint_name:
            filtered_history = [
                record for record in filtered_history
                if record.get('endpoint_name') == endpoint_name
            ]
        
        return filtered_history
    
    def set_baseline_metrics(self, endpoint_name: str, baseline_data: pd.DataFrame):
        """Set baseline metrics for drift detection."""
        
        if endpoint_name not in self.baseline_metrics:
            self.baseline_metrics[endpoint_name] = {}
        
        self.baseline_metrics[endpoint_name]['baseline_data'] = baseline_data
        self.baseline_metrics[endpoint_name]['baseline_timestamp'] = datetime.utcnow()
        
        logger.info(f"Baseline metrics set for {endpoint_name}")
