"""
BatteryMind - CloudWatch Metrics Integration
Advanced CloudWatch metrics integration for battery management AI models with
automated metric collection, custom dashboards, and intelligent alerting.

Features:
- Automated metric collection and publishing
- Custom CloudWatch dashboards with battery-specific widgets
- Intelligent alarm management with dynamic thresholds
- Cost-optimized metric aggregation and retention
- Multi-dimensional metrics with custom dimensions
- Real-time metric streaming and alerting
- Integration with AWS services ecosystem

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import boto3
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import defaultdict, deque
from botocore.exceptions import ClientError, BotoCoreError

# BatteryMind imports
from ...utils.logging_utils import setup_logger
from ...config.monitoring_config import MonitoringConfig

# Configure logging
logger = setup_logger(__name__)

class MetricType(Enum):
    """CloudWatch metric types."""
    BATTERY_HEALTH = "BatteryHealth"
    BATTERY_PERFORMANCE = "BatteryPerformance"
    AI_MODEL_METRICS = "AIModelMetrics"
    FLEET_METRICS = "FleetMetrics"
    SYSTEM_METRICS = "SystemMetrics"
    BUSINESS_METRICS = "BusinessMetrics"
    SECURITY_METRICS = "SecurityMetrics"
    COST_METRICS = "CostMetrics"

class StatisticType(Enum):
    """CloudWatch statistic types."""
    AVERAGE = "Average"
    SUM = "Sum"
    MAXIMUM = "Maximum"
    MINIMUM = "Minimum"
    SAMPLE_COUNT = "SampleCount"

class ComparisonOperator(Enum):
    """CloudWatch alarm comparison operators."""
    GREATER_THAN_THRESHOLD = "GreaterThanThreshold"
    GREATER_THAN_OR_EQUAL_TO_THRESHOLD = "GreaterThanOrEqualToThreshold"
    LESS_THAN_THRESHOLD = "LessThanThreshold"
    LESS_THAN_OR_EQUAL_TO_THRESHOLD = "LessThanOrEqualToThreshold"
    LESS_THAN_LOWER_OR_GREATER_THAN_UPPER_THRESHOLD = "LessThanLowerOrGreaterThanUpperThreshold"
    LESS_THAN_LOWER_THRESHOLD = "LessThanLowerThreshold"
    GREATER_THAN_UPPER_THRESHOLD = "GreaterThanUpperThreshold"

class TreatMissingData(Enum):
    """How to treat missing data in alarms."""
    BREACHING = "breaching"
    NOT_BREACHING = "notBreaching"
    IGNORE = "ignore"
    MISSING = "missing"

@dataclass
class MetricDimension:
    """CloudWatch metric dimension."""
    
    name: str
    value: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to CloudWatch API format."""
        return {
            'Name': self.name,
            'Value': self.value
        }

@dataclass
class MetricData:
    """CloudWatch metric data point."""
    
    metric_name: str
    value: Union[float, int]
    unit: str = "None"
    dimensions: List[MetricDimension] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to CloudWatch API format."""
        metric_data = {
            'MetricName': self.metric_name,
            'Value': float(self.value),
            'Unit': self.unit
        }
        
        if self.dimensions:
            metric_data['Dimensions'] = [dim.to_dict() for dim in self.dimensions]
        
        if self.timestamp:
            metric_data['Timestamp'] = self.timestamp
        
        return metric_data

@dataclass
class CloudWatchWidget:
    """CloudWatch dashboard widget configuration."""
    
    type: str  # metric, log, number, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    width: int = 12
    height: int = 6
    x: int = 0
    y: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to CloudWatch dashboard format."""
        return {
            'type': self.type,
            'width': self.width,
            'height': self.height,
            'x': self.x,
            'y': self.y,
            'properties': self.properties
        }

@dataclass
class CloudWatchAlarm:
    """CloudWatch alarm configuration."""
    
    alarm_name: str
    alarm_description: str
    metric_name: str
    namespace: str
    statistic: StatisticType
    threshold: float
    comparison_operator: ComparisonOperator
    evaluation_periods: int = 2
    datapoints_to_alarm: int = 2
    period: int = 300  # 5 minutes
    dimensions: List[MetricDimension] = field(default_factory=list)
    treat_missing_data: TreatMissingData = TreatMissingData.NOT_BREACHING
    alarm_actions: List[str] = field(default_factory=list)
    ok_actions: List[str] = field(default_factory=list)
    insufficient_data_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to CloudWatch API format."""
        alarm_config = {
            'AlarmName': self.alarm_name,
            'AlarmDescription': self.alarm_description,
            'MetricName': self.metric_name,
            'Namespace': self.namespace,
            'Statistic': self.statistic.value,
            'Threshold': self.threshold,
            'ComparisonOperator': self.comparison_operator.value,
            'EvaluationPeriods': self.evaluation_periods,
            'DatapointsToAlarm': self.datapoints_to_alarm,
            'Period': self.period,
            'TreatMissingData': self.treat_missing_data.value
        }
        
        if self.dimensions:
            alarm_config['Dimensions'] = [dim.to_dict() for dim in self.dimensions]
        
        if self.alarm_actions:
            alarm_config['AlarmActions'] = self.alarm_actions
        
        if self.ok_actions:
            alarm_config['OKActions'] = self.ok_actions
        
        if self.insufficient_data_actions:
            alarm_config['InsufficientDataActions'] = self.insufficient_data_actions
        
        return alarm_config

@dataclass
class CloudWatchDashboard:
    """CloudWatch dashboard configuration."""
    
    dashboard_name: str
    widgets: List[CloudWatchWidget] = field(default_factory=list)
    
    def add_widget(self, widget: CloudWatchWidget) -> None:
        """Add a widget to the dashboard."""
        self.widgets.append(widget)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to CloudWatch dashboard format."""
        return {
            'DashboardName': self.dashboard_name,
            'DashboardBody': json.dumps({
                'widgets': [widget.to_dict() for widget in self.widgets]
            })
        }

class CloudWatchMetrics:
    """
    CloudWatch metrics manager for BatteryMind monitoring.
    """
    
    def __init__(self, 
                 region: str = "us-west-2",
                 namespace: str = "BatteryMind",
                 profile_name: Optional[str] = None):
        
        self.region = region
        self.namespace = namespace
        
        # Initialize AWS session and client
        try:
            if profile_name:
                session = boto3.Session(profile_name=profile_name)
            else:
                session = boto3.Session()
            
            self.cloudwatch_client = session.client('cloudwatch', region_name=region)
            self.logs_client = session.client('logs', region_name=region)
            
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            raise
        
        # Metric batching for efficient API calls
        self.metric_buffer: List[MetricData] = []
        self.buffer_lock = threading.Lock()
        self.max_batch_size = 20  # CloudWatch limit
        self.buffer_timeout = 60  # seconds
        
        # Statistics tracking
        self.stats = {
            'metrics_published': 0,
            'alarms_created': 0,
            'dashboards_created': 0,
            'api_calls': 0,
            'api_errors': 0
        }
        
        # Background processing
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        logger.info(f"CloudWatch Metrics initialized for namespace: {namespace}")
    
    def start(self):
        """Start background metric processing."""
        if self.is_running:
            logger.warning("CloudWatch metrics already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name="CloudWatchMetrics-Processor",
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info("CloudWatch metrics processing started")
    
    def stop(self, timeout: int = 30):
        """Stop background processing and flush remaining metrics."""
        if not self.is_running:
            return
        
        logger.info("Stopping CloudWatch metrics processing...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Flush remaining metrics
        self._flush_metrics()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=timeout)
        
        logger.info("CloudWatch metrics processing stopped")
    
    def put_metric(self, 
                   metric_name: str,
                   value: Union[float, int],
                   unit: str = "None",
                   dimensions: Optional[List[MetricDimension]] = None,
                   timestamp: Optional[datetime] = None) -> bool:
        """
        Put a single metric data point.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            dimensions: Metric dimensions
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            Success status
        """
        try:
            metric_data = MetricData(
                metric_name=metric_name,
                value=value,
                unit=unit,
                dimensions=dimensions or [],
                timestamp=timestamp
            )
            
            with self.buffer_lock:
                self.metric_buffer.append(metric_data)
                
                # Flush if buffer is full
                if len(self.metric_buffer) >= self.max_batch_size:
                    self._flush_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Error buffering metric {metric_name}: {e}")
            return False
    
    def put_metrics_batch(self, metrics: List[MetricData]) -> bool:
        """
        Put multiple metrics in batch.
        
        Args:
            metrics: List of metric data points
            
        Returns:
            Success status
        """
        try:
            with self.buffer_lock:
                self.metric_buffer.extend(metrics)
                
                # Flush if buffer is getting large
                if len(self.metric_buffer) >= self.max_batch_size:
                    self._flush_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Error buffering metric batch: {e}")
            return False
    
    def _flush_metrics(self):
        """Flush buffered metrics to CloudWatch."""
        if not self.metric_buffer:
            return
        
        try:
            with self.buffer_lock:
                metrics_to_send = self.metric_buffer[:self.max_batch_size]
                self.metric_buffer = self.metric_buffer[self.max_batch_size:]
            
            # Convert to CloudWatch format
            metric_data = [metric.to_dict() for metric in metrics_to_send]
            
            # Send to CloudWatch
            self.stats['api_calls'] += 1
            response = self.cloudwatch_client.put_metric_data(
                Namespace=self.namespace,
                MetricData=metric_data
            )
            
            self.stats['metrics_published'] += len(metrics_to_send)
            logger.debug(f"Published {len(metrics_to_send)} metrics to CloudWatch")
            
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to publish metrics to CloudWatch: {e}")
            self.stats['api_errors'] += 1
            
            # Return metrics to buffer for retry
            with self.buffer_lock:
                self.metric_buffer = metrics_to_send + self.metric_buffer
        
        except Exception as e:
            logger.error(f"Unexpected error flushing metrics: {e}")
            self.stats['api_errors'] += 1
    
    def _processing_loop(self):
        """Background processing loop for metric flushing."""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Flush metrics periodically
                if self.metric_buffer:
                    self._flush_metrics()
                
                # Wait for next flush interval
                self.shutdown_event.wait(self.buffer_timeout)
                
            except Exception as e:
                logger.error(f"Error in CloudWatch processing loop: {e}")
                time.sleep(10)
    
    def create_alarm(self, alarm: CloudWatchAlarm) -> bool:
        """
        Create a CloudWatch alarm.
        
        Args:
            alarm: Alarm configuration
            
        Returns:
            Success status
        """
        try:
            self.stats['api_calls'] += 1
            self.cloudwatch_client.put_metric_alarm(**alarm.to_dict())
            
            self.stats['alarms_created'] += 1
            logger.info(f"Created CloudWatch alarm: {alarm.alarm_name}")
            
            return True
            
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to create alarm {alarm.alarm_name}: {e}")
            self.stats['api_errors'] += 1
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error creating alarm: {e}")
            return False
    
    def delete_alarm(self, alarm_name: str) -> bool:
        """
        Delete a CloudWatch alarm.
        
        Args:
            alarm_name: Name of the alarm to delete
            
        Returns:
            Success status
        """
        try:
            self.stats['api_calls'] += 1
            self.cloudwatch_client.delete_alarms(AlarmNames=[alarm_name])
            
            logger.info(f"Deleted CloudWatch alarm: {alarm_name}")
            return True
            
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to delete alarm {alarm_name}: {e}")
            self.stats['api_errors'] += 1
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error deleting alarm: {e}")
            return False
    
    def create_dashboard(self, dashboard: CloudWatchDashboard) -> bool:
        """
        Create a CloudWatch dashboard.
        
        Args:
            dashboard: Dashboard configuration
            
        Returns:
            Success status
        """
        try:
            dashboard_config = dashboard.to_dict()
            
            self.stats['api_calls'] += 1
            self.cloudwatch_client.put_dashboard(**dashboard_config)
            
            self.stats['dashboards_created'] += 1
            logger.info(f"Created CloudWatch dashboard: {dashboard.dashboard_name}")
            
            return True
            
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to create dashboard {dashboard.dashboard_name}: {e}")
            self.stats['api_errors'] += 1
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error creating dashboard: {e}")
            return False
    
    def get_metric_statistics(self,
                             metric_name: str,
                             start_time: datetime,
                             end_time: datetime,
                             period: int,
                             statistics: List[str],
                             dimensions: Optional[List[MetricDimension]] = None) -> Optional[Dict[str, Any]]:
        """
        Get metric statistics from CloudWatch.
        
        Args:
            metric_name: Name of the metric
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            period: Period in seconds
            statistics: List of statistics to retrieve
            dimensions: Optional metric dimensions
            
        Returns:
            Metric statistics data or None if failed
        """
        try:
            params = {
                'Namespace': self.namespace,
                'MetricName': metric_name,
                'StartTime': start_time,
                'EndTime': end_time,
                'Period': period,
                'Statistics': statistics
            }
            
            if dimensions:
                params['Dimensions'] = [dim.to_dict() for dim in dimensions]
            
            self.stats['api_calls'] += 1
            response = self.cloudwatch_client.get_metric_statistics(**params)
            
            return response
            
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to get metric statistics for {metric_name}: {e}")
            self.stats['api_errors'] += 1
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error getting metric statistics: {e}")
            return None
    
    def create_battery_health_dashboard(self) -> CloudWatchDashboard:
        """Create a comprehensive battery health monitoring dashboard."""
        
        dashboard = CloudWatchDashboard("BatteryMind-BatteryHealth")
        
        # Battery SoH Widget
        soh_widget = CloudWatchWidget(
            type="metric",
            width=6,
            height=6,
            x=0,
            y=0,
            properties={
                "metrics": [
                    [self.namespace, "BatteryStateOfHealth", "BatteryId", "ALL"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Battery State of Health",
                "period": 300,
                "yAxis": {
                    "left": {
                        "min": 0,
                        "max": 100
                    }
                }
            }
        )
        dashboard.add_widget(soh_widget)
        
        # Battery Temperature Widget
        temp_widget = CloudWatchWidget(
            type="metric",
            width=6,
            height=6,
            x=6,
            y=0,
            properties={
                "metrics": [
                    [self.namespace, "BatteryTemperature", "BatteryId", "ALL"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Battery Temperature",
                "period": 300,
                "yAxis": {
                    "left": {
                        "min": -20,
                        "max": 80
                    }
                }
            }
        )
        dashboard.add_widget(temp_widget)
        
        # Battery Voltage Widget
        voltage_widget = CloudWatchWidget(
            type="metric",
            width=6,
            height=6,
            x=0,
            y=6,
            properties={
                "metrics": [
                    [self.namespace, "BatteryVoltage", "BatteryId", "ALL"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Battery Voltage",
                "period": 300
            }
        )
        dashboard.add_widget(voltage_widget)
        
        # Battery Current Widget
        current_widget = CloudWatchWidget(
            type="metric",
            width=6,
            height=6,
            x=6,
            y=6,
            properties={
                "metrics": [
                    [self.namespace, "BatteryCurrent", "BatteryId", "ALL"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Battery Current",
                "period": 300
            }
        )
        dashboard.add_widget(current_widget)
        
        # AI Model Accuracy Widget
        accuracy_widget = CloudWatchWidget(
            type="metric",
            width=12,
            height=6,
            x=0,
            y=12,
            properties={
                "metrics": [
                    [self.namespace, "ModelAccuracy", "ModelName", "BatteryHealthPredictor"],
                    [".", "ModelLatency", ".", "."]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "AI Model Performance",
                "period": 300
            }
        )
        dashboard.add_widget(accuracy_widget)
        
        return dashboard
    
    def create_fleet_dashboard(self) -> CloudWatchDashboard:
        """Create a fleet management monitoring dashboard."""
        
        dashboard = CloudWatchDashboard("BatteryMind-FleetManagement")
        
        # Fleet Overview Widget
        overview_widget = CloudWatchWidget(
            type="metric",
            width=12,
            height=6,
            x=0,
            y=0,
            properties={
                "metrics": [
                    [self.namespace, "ActiveVehicles", "FleetId", "ALL"],
                    [".", "ChargingVehicles", ".", "."],
                    [".", "MaintenanceVehicles", ".", "."]
                ],
                "view": "timeSeries",
                "stacked": True,
                "region": self.region,
                "title": "Fleet Overview",
                "period": 300
            }
        )
        dashboard.add_widget(overview_widget)
        
        # Energy Consumption Widget
        energy_widget = CloudWatchWidget(
            type="metric",
            width=6,
            height=6,
            x=0,
            y=6,
            properties={
                "metrics": [
                    [self.namespace, "EnergyConsumption", "FleetId", "ALL"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Energy Consumption",
                "period": 300
            }
        )
        dashboard.add_widget(energy_widget)
        
        # Efficiency Metrics Widget
        efficiency_widget = CloudWatchWidget(
            type="metric",
            width=6,
            height=6,
            x=6,
            y=6,
            properties={
                "metrics": [
                    [self.namespace, "FleetEfficiency", "FleetId", "ALL"],
                    [".", "UtilizationRate", ".", "."]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Fleet Efficiency",
                "period": 300
            }
        )
        dashboard.add_widget(efficiency_widget)
        
        return dashboard
    
    def create_battery_health_alarms(self) -> List[CloudWatchAlarm]:
        """Create standard battery health alarms."""
        
        alarms = []
        
        # Critical SoH Alarm
        critical_soh_alarm = CloudWatchAlarm(
            alarm_name="BatteryMind-CriticalSoH",
            alarm_description="Battery State of Health below critical threshold",
            metric_name="BatteryStateOfHealth",
            namespace=self.namespace,
            statistic=StatisticType.AVERAGE,
            threshold=70.0,
            comparison_operator=ComparisonOperator.LESS_THAN_THRESHOLD,
            evaluation_periods=2,
            datapoints_to_alarm=2,
            period=300
        )
        alarms.append(critical_soh_alarm)
        
        # High Temperature Alarm
        high_temp_alarm = CloudWatchAlarm(
            alarm_name="BatteryMind-HighTemperature",
            alarm_description="Battery temperature above safe threshold",
            metric_name="BatteryTemperature",
            namespace=self.namespace,
            statistic=StatisticType.MAXIMUM,
            threshold=60.0,
            comparison_operator=ComparisonOperator.GREATER_THAN_THRESHOLD,
            evaluation_periods=1,
            datapoints_to_alarm=1,
            period=300
        )
        alarms.append(high_temp_alarm)
        
        # Model Accuracy Alarm
        accuracy_alarm = CloudWatchAlarm(
            alarm_name="BatteryMind-ModelAccuracy",
            alarm_description="AI model accuracy below acceptable threshold",
            metric_name="ModelAccuracy",
            namespace=self.namespace,
            statistic=StatisticType.AVERAGE,
            threshold=95.0,
            comparison_operator=ComparisonOperator.LESS_THAN_THRESHOLD,
            evaluation_periods=3,
            datapoints_to_alarm=2,
            period=900,  # 15 minutes
            dimensions=[MetricDimension("ModelName", "BatteryHealthPredictor")]
        )
        alarms.append(accuracy_alarm)
        
        return alarms
    
    def publish_battery_metrics(self, 
                               battery_id: str,
                               soh: float,
                               soc: float,
                               voltage: float,
                               current: float,
                               temperature: float) -> bool:
        """
        Publish battery-specific metrics to CloudWatch.
        
        Args:
            battery_id: Unique battery identifier
            soh: State of Health (%)
            soc: State of Charge (%)
            voltage: Battery voltage (V)
            current: Battery current (A)
            temperature: Battery temperature (°C)
            
        Returns:
            Success status
        """
        try:
            dimensions = [MetricDimension("BatteryId", battery_id)]
            
            metrics = [
                MetricData("BatteryStateOfHealth", soh, "Percent", dimensions),
                MetricData("BatteryStateOfCharge", soc, "Percent", dimensions),
                MetricData("BatteryVoltage", voltage, "None", dimensions),
                MetricData("BatteryCurrent", current, "None", dimensions),
                MetricData("BatteryTemperature", temperature, "None", dimensions)
            ]
            
            return self.put_metrics_batch(metrics)
            
        except Exception as e:
            logger.error(f"Error publishing battery metrics for {battery_id}: {e}")
            return False
    
    def publish_ai_model_metrics(self,
                                model_name: str,
                                accuracy: float,
                                latency: float,
                                throughput: float,
                                memory_usage: float) -> bool:
        """
        Publish AI model performance metrics to CloudWatch.
        
        Args:
            model_name: Name of the AI model
            accuracy: Model accuracy (%)
            latency: Inference latency (ms)
            throughput: Requests per second
            memory_usage: Memory usage (MB)
            
        Returns:
            Success status
        """
        try:
            dimensions = [MetricDimension("ModelName", model_name)]
            
            metrics = [
                MetricData("ModelAccuracy", accuracy, "Percent", dimensions),
                MetricData("ModelLatency", latency, "Milliseconds", dimensions),
                MetricData("ModelThroughput", throughput, "Count/Second", dimensions),
                MetricData("ModelMemoryUsage", memory_usage, "Megabytes", dimensions)
            ]
            
            return self.put_metrics_batch(metrics)
            
        except Exception as e:
            logger.error(f"Error publishing AI model metrics for {model_name}: {e}")
            return False
    
    def publish_fleet_metrics(self,
                             fleet_id: str,
                             active_vehicles: int,
                             charging_vehicles: int,
                             energy_consumption: float,
                             efficiency: float) -> bool:
        """
        Publish fleet management metrics to CloudWatch.
        
        Args:
            fleet_id: Fleet identifier
            active_vehicles: Number of active vehicles
            charging_vehicles: Number of charging vehicles
            energy_consumption: Total energy consumption (kWh)
            efficiency: Fleet efficiency (%)
            
        Returns:
            Success status
        """
        try:
            dimensions = [MetricDimension("FleetId", fleet_id)]
            
            metrics = [
                MetricData("ActiveVehicles", active_vehicles, "Count", dimensions),
                MetricData("ChargingVehicles", charging_vehicles, "Count", dimensions),
                MetricData("EnergyConsumption", energy_consumption, "None", dimensions),
                MetricData("FleetEfficiency", efficiency, "Percent", dimensions)
            ]
            
            return self.put_metrics_batch(metrics)
            
        except Exception as e:
            logger.error(f"Error publishing fleet metrics for {fleet_id}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get CloudWatch metrics manager statistics."""
        with self.buffer_lock:
            buffered_metrics = len(self.metric_buffer)
        
        return {
            **self.stats,
            'buffered_metrics': buffered_metrics,
            'namespace': self.namespace,
            'region': self.region,
            'is_running': self.is_running
        }

# Factory functions
def create_cloudwatch_manager(region: str = "us-west-2", 
                             namespace: str = "BatteryMind") -> CloudWatchMetrics:
    """Create a CloudWatch metrics manager."""
    return CloudWatchMetrics(region=region, namespace=namespace)

def create_battery_monitoring_setup(cloudwatch: CloudWatchMetrics) -> Dict[str, Any]:
    """Create complete battery monitoring setup in CloudWatch."""
    
    setup_results = {
        'dashboards': [],
        'alarms': [],
        'success': True,
        'errors': []
    }
    
    try:
        # Create dashboards
        battery_dashboard = cloudwatch.create_battery_health_dashboard()
        fleet_dashboard = cloudwatch.create_fleet_dashboard()
        
        dashboard_success = cloudwatch.create_dashboard(battery_dashboard)
        if dashboard_success:
            setup_results['dashboards'].append(battery_dashboard.dashboard_name)
        else:
            setup_results['errors'].append(f"Failed to create {battery_dashboard.dashboard_name}")
        
        fleet_success = cloudwatch.create_dashboard(fleet_dashboard)
        if fleet_success:
            setup_results['dashboards'].append(fleet_dashboard.dashboard_name)
        else:
            setup_results['errors'].append(f"Failed to create {fleet_dashboard.dashboard_name}")
        
        # Create alarms
        alarms = cloudwatch.create_battery_health_alarms()
        for alarm in alarms:
            alarm_success = cloudwatch.create_alarm(alarm)
            if alarm_success:
                setup_results['alarms'].append(alarm.alarm_name)
            else:
                setup_results['errors'].append(f"Failed to create {alarm.alarm_name}")
        
        setup_results['success'] = len(setup_results['errors']) == 0
        
        logger.info(f"CloudWatch monitoring setup completed")
        logger.info(f"Created {len(setup_results['dashboards'])} dashboards")
        logger.info(f"Created {len(setup_results['alarms'])} alarms")
        
        return setup_results
        
    except Exception as e:
        logger.error(f"Error setting up CloudWatch monitoring: {e}")
        setup_results['success'] = False
        setup_results['errors'].append(str(e))
        return setup_results
