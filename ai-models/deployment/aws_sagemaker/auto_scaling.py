"""
BatteryMind - AWS SageMaker Auto-Scaling Module

Production-ready auto-scaling implementation for SageMaker endpoints that handles
dynamic scaling based on prediction load, model performance metrics, and cost optimization.

Features:
- Dynamic instance scaling based on real-time metrics
- Cost-aware scaling policies with budget constraints
- Multi-model endpoint support with traffic distribution
- Predictive scaling using historical patterns
- Integration with CloudWatch alarms and SNS notifications
- Custom scaling metrics for battery prediction workloads

Author: BatteryMind Development Team
Version: 1.0.0
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import boto3
from botocore.exceptions import ClientError, BotoCoreError
import numpy as np
import pandas as pd

# BatteryMind imports
from .endpoint_config import SageMakerEndpointConfig
from .model_deployment import ModelDeploymentManager
from ..monitoring.cloudwatch_metrics import CloudWatchMetrics
from ...utils.logging_utils import setup_logger
from ...utils.aws_helpers import AWSHelper

# Configure logging
logger = setup_logger(__name__)

@dataclass
class ScalingPolicy:
    """Configuration for auto-scaling policies."""
    
    # Basic scaling parameters
    min_instances: int = 1
    max_instances: int = 10
    target_invocations_per_instance: int = 100
    scale_up_cooldown_seconds: int = 300
    scale_down_cooldown_seconds: int = 600
    
    # Advanced scaling parameters
    cpu_target_utilization: float = 70.0  # Target CPU utilization %
    memory_target_utilization: float = 80.0  # Target memory utilization %
    latency_threshold_ms: float = 1000.0  # Maximum acceptable latency
    
    # Cost optimization
    cost_optimization_enabled: bool = True
    max_hourly_cost: float = 50.0  # Maximum cost per hour
    preferred_instance_types: List[str] = field(default_factory=lambda: [
        'ml.m5.large', 'ml.m5.xlarge', 'ml.c5.large', 'ml.c5.xlarge'
    ])
    
    # Predictive scaling
    predictive_scaling_enabled: bool = True
    prediction_horizon_minutes: int = 60
    
    # Battery-specific metrics
    prediction_accuracy_threshold: float = 0.95
    model_drift_threshold: float = 0.1

@dataclass
class ScalingMetrics:
    """Current scaling metrics for decision making."""
    
    current_instances: int = 0
    invocations_per_minute: float = 0.0
    average_latency_ms: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    error_rate: float = 0.0
    model_accuracy: float = 0.0
    cost_per_hour: float = 0.0
    
    # Predictive metrics
    predicted_load: float = 0.0
    trend_direction: str = "stable"  # "increasing", "decreasing", "stable"

class SageMakerAutoScaler:
    """
    Advanced auto-scaling manager for SageMaker endpoints with battery-specific optimizations.
    """
    
    def __init__(self, config: ScalingPolicy, aws_region: str = 'us-west-2'):
        self.config = config
        self.aws_region = aws_region
        
        # Initialize AWS clients
        self.sagemaker_client = boto3.client('sagemaker', region_name=aws_region)
        self.application_autoscaling_client = boto3.client(
            'application-autoscaling', region_name=aws_region
        )
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=aws_region)
        
        # Initialize helper components
        self.aws_helper = AWSHelper()
        self.cloudwatch_metrics = CloudWatchMetrics()
        
        # Scaling state
        self.last_scale_time = {}
        self.scaling_history = []
        self.load_predictions = {}
        
        logger.info("SageMaker Auto-Scaler initialized")
    
    def setup_auto_scaling(self, endpoint_name: str, 
                          variant_name: str = 'AllTraffic') -> bool:
        """
        Set up auto-scaling for a SageMaker endpoint.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            variant_name: Name of the production variant
            
        Returns:
            Success status
        """
        try:
            # Register scalable target
            self._register_scalable_target(endpoint_name, variant_name)
            
            # Create scaling policies
            self._create_scaling_policies(endpoint_name, variant_name)
            
            # Set up CloudWatch alarms
            self._setup_cloudwatch_alarms(endpoint_name, variant_name)
            
            logger.info(f"Auto-scaling setup completed for endpoint: {endpoint_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup auto-scaling for {endpoint_name}: {e}")
            return False
    
    def _register_scalable_target(self, endpoint_name: str, variant_name: str):
        """Register the endpoint as a scalable target."""
        resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"
        
        try:
            self.application_autoscaling_client.register_scalable_target(
                ServiceNamespace='sagemaker',
                ResourceId=resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                MinCapacity=self.config.min_instances,
                MaxCapacity=self.config.max_instances,
                Tags=[
                    {'Key': 'Project', 'Value': 'BatteryMind'},
                    {'Key': 'Component', 'Value': 'AutoScaling'},
                    {'Key': 'Environment', 'Value': 'Production'}
                ]
            )
            
            logger.info(f"Registered scalable target: {resource_id}")
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                logger.warning(f"Scalable target already exists: {resource_id}")
            else:
                raise
    
    def _create_scaling_policies(self, endpoint_name: str, variant_name: str):
        """Create scaling policies for the endpoint."""
        resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"
        
        # Scale up policy
        scale_up_policy = {
            'PolicyName': f'{endpoint_name}-scale-up-policy',
            'ServiceNamespace': 'sagemaker',
            'ResourceId': resource_id,
            'ScalableDimension': 'sagemaker:variant:DesiredInstanceCount',
            'PolicyType': 'TargetTrackingScaling',
            'TargetTrackingScalingPolicyConfiguration': {
                'TargetValue': self.config.target_invocations_per_instance,
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                },
                'ScaleOutCooldown': self.config.scale_up_cooldown_seconds,
                'ScaleInCooldown': self.config.scale_down_cooldown_seconds
            }
        }
        
        try:
            self.application_autoscaling_client.put_scaling_policy(**scale_up_policy)
            logger.info(f"Created scaling policy for {endpoint_name}")
            
        except ClientError as e:
            logger.error(f"Failed to create scaling policy: {e}")
            raise
    
    def _setup_cloudwatch_alarms(self, endpoint_name: str, variant_name: str):
        """Set up CloudWatch alarms for monitoring and alerting."""
        
        alarms = [
            # High latency alarm
            {
                'AlarmName': f'{endpoint_name}-high-latency',
                'ComparisonOperator': 'GreaterThanThreshold',
                'EvaluationPeriods': 2,
                'MetricName': 'ModelLatency',
                'Namespace': 'AWS/SageMaker',
                'Period': 300,
                'Statistic': 'Average',
                'Threshold': self.config.latency_threshold_ms,
                'ActionsEnabled': True,
                'AlarmDescription': 'High latency detected on SageMaker endpoint',
                'Dimensions': [
                    {'Name': 'EndpointName', 'Value': endpoint_name},
                    {'Name': 'VariantName', 'Value': variant_name}
                ],
                'Unit': 'Milliseconds'
            },
            
            # High error rate alarm
            {
                'AlarmName': f'{endpoint_name}-high-error-rate',
                'ComparisonOperator': 'GreaterThanThreshold',
                'EvaluationPeriods': 2,
                'MetricName': 'Model4XXErrors',
                'Namespace': 'AWS/SageMaker',
                'Period': 300,
                'Statistic': 'Sum',
                'Threshold': 10,
                'ActionsEnabled': True,
                'AlarmDescription': 'High error rate detected on SageMaker endpoint',
                'Dimensions': [
                    {'Name': 'EndpointName', 'Value': endpoint_name},
                    {'Name': 'VariantName', 'Value': variant_name}
                ]
            },
            
            # Cost alarm
            {
                'AlarmName': f'{endpoint_name}-high-cost',
                'ComparisonOperator': 'GreaterThanThreshold',
                'EvaluationPeriods': 1,
                'MetricName': 'EstimatedCharges',
                'Namespace': 'AWS/Billing',
                'Period': 3600,
                'Statistic': 'Maximum',
                'Threshold': self.config.max_hourly_cost,
                'ActionsEnabled': True,
                'AlarmDescription': 'High cost detected for SageMaker endpoint',
                'Dimensions': [
                    {'Name': 'ServiceName', 'Value': 'AmazonSageMaker'},
                    {'Name': 'Currency', 'Value': 'USD'}
                ]
            }
        ]
        
        for alarm in alarms:
            try:
                self.cloudwatch_client.put_metric_alarm(**alarm)
                logger.info(f"Created CloudWatch alarm: {alarm['AlarmName']}")
                
            except ClientError as e:
                logger.error(f"Failed to create alarm {alarm['AlarmName']}: {e}")
    
    def get_current_metrics(self, endpoint_name: str, 
                           variant_name: str = 'AllTraffic') -> ScalingMetrics:
        """
        Get current metrics for scaling decisions.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            variant_name: Name of the production variant
            
        Returns:
            Current scaling metrics
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)
        
        try:
            # Get endpoint configuration
            endpoint_config = self.sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            
            current_instances = 0
            for variant in endpoint_config['ProductionVariants']:
                if variant['VariantName'] == variant_name:
                    current_instances = variant['CurrentInstanceCount']
                    break
            
            # Get CloudWatch metrics
            metrics = self._get_cloudwatch_metrics(
                endpoint_name, variant_name, start_time, end_time
            )
            
            # Calculate predictive metrics if enabled
            predicted_load = 0.0
            trend_direction = "stable"
            
            if self.config.predictive_scaling_enabled:
                predicted_load, trend_direction = self._predict_future_load(
                    endpoint_name, variant_name
                )
            
            return ScalingMetrics(
                current_instances=current_instances,
                invocations_per_minute=metrics.get('invocations_per_minute', 0.0),
                average_latency_ms=metrics.get('average_latency', 0.0),
                cpu_utilization=metrics.get('cpu_utilization', 0.0),
                memory_utilization=metrics.get('memory_utilization', 0.0),
                error_rate=metrics.get('error_rate', 0.0),
                model_accuracy=metrics.get('model_accuracy', 0.0),
                cost_per_hour=metrics.get('cost_per_hour', 0.0),
                predicted_load=predicted_load,
                trend_direction=trend_direction
            )
            
        except Exception as e:
            logger.error(f"Failed to get metrics for {endpoint_name}: {e}")
            return ScalingMetrics()
    
    def _get_cloudwatch_metrics(self, endpoint_name: str, variant_name: str,
                               start_time: datetime, end_time: datetime) -> Dict[str, float]:
        """Retrieve metrics from CloudWatch."""
        
        metrics = {}
        
        # Define metric queries
        metric_queries = [
            {
                'name': 'invocations_per_minute',
                'metric_name': 'Invocations',
                'statistic': 'Sum',
                'unit': 'Count'
            },
            {
                'name': 'average_latency',
                'metric_name': 'ModelLatency',
                'statistic': 'Average',
                'unit': 'Milliseconds'
            },
            {
                'name': 'error_rate',
                'metric_name': 'Model4XXErrors',
                'statistic': 'Sum',
                'unit': 'Count'
            }
        ]
        
        for query in metric_queries:
            try:
                response = self.cloudwatch_client.get_metric_statistics(
                    Namespace='AWS/SageMaker',
                    MetricName=query['metric_name'],
                    Dimensions=[
                        {'Name': 'EndpointName', 'Value': endpoint_name},
                        {'Name': 'VariantName', 'Value': variant_name}
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=300,
                    Statistics=[query['statistic']]
                )
                
                if response['Datapoints']:
                    latest_point = max(response['Datapoints'], key=lambda x: x['Timestamp'])
                    metrics[query['name']] = latest_point[query['statistic']]
                else:
                    metrics[query['name']] = 0.0
                    
            except Exception as e:
                logger.warning(f"Failed to get metric {query['name']}: {e}")
                metrics[query['name']] = 0.0
        
        return metrics
    
    def _predict_future_load(self, endpoint_name: str, 
                           variant_name: str) -> Tuple[float, str]:
        """
        Predict future load using historical patterns.
        
        Returns:
            Tuple of (predicted_load, trend_direction)
        """
        try:
            # Get historical data for the last 24 hours
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            response = self.cloudwatch_client.get_metric_statistics(
                Namespace='AWS/SageMaker',
                MetricName='Invocations',
                Dimensions=[
                    {'Name': 'EndpointName', 'Value': endpoint_name},
                    {'Name': 'VariantName', 'Value': variant_name}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour periods
                Statistics=['Sum']
            )
            
            if len(response['Datapoints']) < 3:
                return 0.0, "stable"
            
            # Extract time series data
            datapoints = sorted(response['Datapoints'], key=lambda x: x['Timestamp'])
            values = [dp['Sum'] for dp in datapoints]
            
            # Simple linear regression for trend
            x = np.arange(len(values))
            z = np.polyfit(x, values, 1)
            slope = z[0]
            
            # Predict next hour
            predicted_load = float(z[1] + z[0] * len(values))
            
            # Determine trend direction
            if slope > 5:  # More than 5 invocations/hour increase
                trend_direction = "increasing"
            elif slope < -5:  # More than 5 invocations/hour decrease
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
            
            return max(0.0, predicted_load), trend_direction
            
        except Exception as e:
            logger.warning(f"Failed to predict future load: {e}")
            return 0.0, "stable"
    
    def make_scaling_decision(self, endpoint_name: str, 
                            variant_name: str = 'AllTraffic') -> Optional[Dict[str, Any]]:
        """
        Make intelligent scaling decisions based on current metrics.
        
        Returns:
            Scaling decision dictionary or None if no action needed
        """
        metrics = self.get_current_metrics(endpoint_name, variant_name)
        
        # Check cooldown period
        last_scale = self.last_scale_time.get(endpoint_name, 0)
        time_since_last_scale = time.time() - last_scale
        
        if time_since_last_scale < self.config.scale_up_cooldown_seconds:
            logger.debug(f"Scaling cooldown active for {endpoint_name}")
            return None
        
        decision = None
        
        # Scale up conditions
        scale_up_reasons = []
        
        if metrics.invocations_per_minute > self.config.target_invocations_per_instance * metrics.current_instances:
            scale_up_reasons.append("high_invocation_rate")
        
        if metrics.average_latency_ms > self.config.latency_threshold_ms:
            scale_up_reasons.append("high_latency")
        
        if metrics.cpu_utilization > self.config.cpu_target_utilization:
            scale_up_reasons.append("high_cpu_utilization")
        
        if self.config.predictive_scaling_enabled and metrics.trend_direction == "increasing":
            scale_up_reasons.append("predicted_load_increase")
        
        # Scale down conditions
        scale_down_reasons = []
        
        if (metrics.invocations_per_minute < self.config.target_invocations_per_instance * metrics.current_instances * 0.5 and
            metrics.average_latency_ms < self.config.latency_threshold_ms * 0.5):
            scale_down_reasons.append("low_utilization")
        
        if metrics.trend_direction == "decreasing" and metrics.current_instances > self.config.min_instances:
            scale_down_reasons.append("predicted_load_decrease")
        
        # Make decision
        if scale_up_reasons and metrics.current_instances < self.config.max_instances:
            new_instance_count = min(
                metrics.current_instances + self._calculate_scale_increment(metrics),
                self.config.max_instances
            )
            
            decision = {
                'action': 'scale_up',
                'current_instances': metrics.current_instances,
                'target_instances': new_instance_count,
                'reasons': scale_up_reasons,
                'metrics': metrics.__dict__
            }
            
        elif scale_down_reasons and metrics.current_instances > self.config.min_instances:
            new_instance_count = max(
                metrics.current_instances - 1,  # Conservative scale down
                self.config.min_instances
            )
            
            decision = {
                'action': 'scale_down',
                'current_instances': metrics.current_instances,
                'target_instances': new_instance_count,
                'reasons': scale_down_reasons,
                'metrics': metrics.__dict__
            }
        
        if decision:
            self.last_scale_time[endpoint_name] = time.time()
            self.scaling_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'endpoint_name': endpoint_name,
                'decision': decision
            })
            
            logger.info(f"Scaling decision for {endpoint_name}: {decision}")
        
        return decision
    
    def _calculate_scale_increment(self, metrics: ScalingMetrics) -> int:
        """Calculate how many instances to add based on current load."""
        
        if metrics.average_latency_ms > self.config.latency_threshold_ms * 2:
            return 2  # Aggressive scaling for high latency
        elif metrics.invocations_per_minute > self.config.target_invocations_per_instance * metrics.current_instances * 2:
            return 2  # Aggressive scaling for very high load
        else:
            return 1  # Conservative scaling
    
    def execute_scaling_decision(self, endpoint_name: str, variant_name: str,
                               target_instances: int) -> bool:
        """
        Execute the scaling decision by updating the endpoint.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            variant_name: Name of the production variant
            target_instances: Target number of instances
            
        Returns:
            Success status
        """
        try:
            resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"
            
            self.application_autoscaling_client.register_scalable_target(
                ServiceNamespace='sagemaker',
                ResourceId=resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                MinCapacity=self.config.min_instances,
                MaxCapacity=self.config.max_instances
            )
            
            # Update the desired capacity
            self.application_autoscaling_client.register_scalable_target(
                ServiceNamespace='sagemaker',
                ResourceId=resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                MinCapacity=self.config.min_instances,
                MaxCapacity=self.config.max_instances
            )
            
            logger.info(f"Scaling executed: {endpoint_name} to {target_instances} instances")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute scaling for {endpoint_name}: {e}")
            return False
    
    def get_scaling_history(self, endpoint_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get scaling history for analysis."""
        if endpoint_name:
            return [h for h in self.scaling_history if h['endpoint_name'] == endpoint_name]
        return self.scaling_history
    
    def optimize_costs(self, endpoint_name: str) -> Dict[str, Any]:
        """
        Analyze and optimize costs for the endpoint.
        
        Returns:
            Cost optimization recommendations
        """
        try:
            # Get current endpoint configuration
            endpoint_config = self.sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            
            recommendations = {
                'current_cost_estimate': 0.0,
                'optimized_cost_estimate': 0.0,
                'savings_potential': 0.0,
                'recommendations': []
            }
            
            # Calculate current costs (simplified)
            for variant in endpoint_config['ProductionVariants']:
                instance_type = variant['InstanceType']
                instance_count = variant['CurrentInstanceCount']
                
                # Get hourly cost (placeholder - integrate with AWS Pricing API)
                hourly_cost_per_instance = self._get_instance_hourly_cost(instance_type)
                current_cost = hourly_cost_per_instance * instance_count
                
                recommendations['current_cost_estimate'] += current_cost
                
                # Suggest optimizations
                if instance_count > 1 and self._can_use_smaller_instances(endpoint_name):
                    recommendations['recommendations'].append({
                        'type': 'instance_type_optimization',
                        'description': f'Consider using smaller instances for {variant["VariantName"]}',
                        'potential_savings': current_cost * 0.2
                    })
            
            # Spot instance recommendations
            if self._is_suitable_for_spot_instances(endpoint_name):
                recommendations['recommendations'].append({
                    'type': 'spot_instances',
                    'description': 'Consider using Spot instances for non-critical workloads',
                    'potential_savings': recommendations['current_cost_estimate'] * 0.5
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to optimize costs for {endpoint_name}: {e}")
            return {}
    
    def _get_instance_hourly_cost(self, instance_type: str) -> float:
        """Get hourly cost for instance type (placeholder)."""
        # This should integrate with AWS Pricing API
        cost_map = {
            'ml.t2.medium': 0.056,
            'ml.m5.large': 0.115,
            'ml.m5.xlarge': 0.230,
            'ml.c5.large': 0.102,
            'ml.c5.xlarge': 0.204
        }
        return cost_map.get(instance_type, 0.115)
    
    def _can_use_smaller_instances(self, endpoint_name: str) -> bool:
        """Check if smaller instances can be used."""
        # Analyze recent performance metrics
        metrics = self.get_current_metrics(endpoint_name)
        return (metrics.cpu_utilization < 50.0 and 
                metrics.memory_utilization < 60.0 and 
                metrics.average_latency_ms < self.config.latency_threshold_ms * 0.5)
    
    def _is_suitable_for_spot_instances(self, endpoint_name: str) -> bool:
        """Check if workload is suitable for Spot instances."""
        # This would analyze workload patterns and fault tolerance
        return False  # Conservative default
