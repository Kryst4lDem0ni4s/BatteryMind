"""
BatteryMind - Monitoring Module

Comprehensive monitoring system for battery AI/ML models and data pipelines.
This module provides real-time monitoring, alerting, and dashboard capabilities
for ensuring system reliability, performance, and safety in production environments.

The monitoring system includes:
- Model performance monitoring and drift detection
- Data quality monitoring and validation
- Real-time alerting and notification systems
- Interactive dashboards and visualization
- Business KPI tracking and reporting
- Resource utilization monitoring

Key Components:
- Model Monitoring: Track model performance, accuracy, and drift
- Data Monitoring: Monitor data quality, schema, and anomalies
- Alerts: Automated alerting and notification management
- Dashboards: Real-time visualization and reporting

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

# Core monitoring components
import numpy as np
import pandas as pd
from datetime import datetime
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"
__license__ = "Proprietary"

# Monitoring status enumeration
class MonitoringStatus(Enum):
    """Monitoring system status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    OFFLINE = "offline"

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MonitoringMetric:
    """Standard monitoring metric structure."""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    
    def is_healthy(self) -> bool:
        """Check if metric is within healthy thresholds."""
        if self.threshold_min is not None and self.value < self.threshold_min:
            return False
        if self.threshold_max is not None and self.value > self.threshold_max:
            return False
        return True
    
    def get_status(self) -> MonitoringStatus:
        """Get status based on threshold violations."""
        if self.is_healthy():
            return MonitoringStatus.HEALTHY
        else:
            # Determine severity based on how far from threshold
            if self.threshold_min is not None and self.value < self.threshold_min:
                deviation = abs(self.value - self.threshold_min) / self.threshold_min
            elif self.threshold_max is not None and self.value > self.threshold_max:
                deviation = abs(self.value - self.threshold_max) / self.threshold_max
            else:
                deviation = 0
            
            if deviation > 0.5:  # 50% deviation
                return MonitoringStatus.CRITICAL
            elif deviation > 0.2:  # 20% deviation
                return MonitoringStatus.WARNING
            else:
                return MonitoringStatus.DEGRADED

@dataclass
class MonitoringAlert:
    """Standard monitoring alert structure."""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    source: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
            'metadata': self.metadata
        }

# Import monitoring components with graceful fallback
try:
    from .model_monitoring import (
        PerformanceMonitor,
        DriftDetector,
        AccuracyTracker,
        ResourceMonitor
    )
    logger.info("✓ Model monitoring components imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import model monitoring components: {e}")
    # Create placeholder classes
    class PerformanceMonitor:
        def __init__(self, *args, **kwargs):
            logger.warning("PerformanceMonitor placeholder initialized")
    
    class DriftDetector:
        def __init__(self, *args, **kwargs):
            logger.warning("DriftDetector placeholder initialized")
    
    class AccuracyTracker:
        def __init__(self, *args, **kwargs):
            logger.warning("AccuracyTracker placeholder initialized")
    
    class ResourceMonitor:
        def __init__(self, *args, **kwargs):
            logger.warning("ResourceMonitor placeholder initialized")

try:
    from .data_monitoring import (
        DataQualityMonitor,
        SchemaValidator,
        AnomalyDetector,
        BiasDetector
    )
    logger.info("✓ Data monitoring components imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import data monitoring components: {e}")
    # Create placeholder classes
    class DataQualityMonitor:
        def __init__(self, *args, **kwargs):
            logger.warning("DataQualityMonitor placeholder initialized")
    
    class SchemaValidator:
        def __init__(self, *args, **kwargs):
            logger.warning("SchemaValidator placeholder initialized")
    
    class AnomalyDetector:
        def __init__(self, *args, **kwargs):
            logger.warning("AnomalyDetector placeholder initialized")
    
    class BiasDetector:
        def __init__(self, *args, **kwargs):
            logger.warning("BiasDetector placeholder initialized")

try:
    from .alerts import (
        AlertManager,
        NotificationService,
        EscalationPolicy,
        AlertRules
    )
    logger.info("✓ Alert components imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import alert components: {e}")
    # Create placeholder classes
    class AlertManager:
        def __init__(self, *args, **kwargs):
            logger.warning("AlertManager placeholder initialized")
    
    class NotificationService:
        def __init__(self, *args, **kwargs):
            logger.warning("NotificationService placeholder initialized")
    
    class EscalationPolicy:
        def __init__(self, *args, **kwargs):
            logger.warning("EscalationPolicy placeholder initialized")
    
    class AlertRules:
        def __init__(self, *args, **kwargs):
            logger.warning("AlertRules placeholder initialized")

try:
    from .dashboards import (
        GrafanaDashboard,
        CloudWatchMetrics,
        CustomMetrics,
        BusinessKPIs
    )
    logger.info("✓ Dashboard components imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import dashboard components: {e}")
    # Create placeholder classes
    class GrafanaDashboard:
        def __init__(self, *args, **kwargs):
            logger.warning("GrafanaDashboard placeholder initialized")
    
    class CloudWatchMetrics:
        def __init__(self, *args, **kwargs):
            logger.warning("CloudWatchMetrics placeholder initialized")
    
    class CustomMetrics:
        def __init__(self, *args, **kwargs):
            logger.warning("CustomMetrics placeholder initialized")
    
    class BusinessKPIs:
        def __init__(self, *args, **kwargs):
            logger.warning("BusinessKPIs placeholder initialized")

class MonitoringConfig:
    """Configuration management for monitoring system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize monitoring configuration."""
        self.config_path = config_path or "../../config/deployment_config.yaml"
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            'monitoring': {
                'enabled': True,
                'collection_interval': 60,  # seconds
                'retention_days': 30,
                'alert_cooldown': 300,  # seconds
                'max_alerts_per_hour': 10,
                'health_check_interval': 30  # seconds
            },
            'model_monitoring': {
                'accuracy_threshold': 0.85,
                'drift_threshold': 0.1,
                'latency_threshold': 100,  # ms
                'memory_threshold': 1000,  # MB
                'error_rate_threshold': 0.01
            },
            'data_monitoring': {
                'completeness_threshold': 0.95,
                'quality_threshold': 0.9,
                'schema_validation': True,
                'anomaly_detection': True,
                'bias_detection': True
            },
            'alerts': {
                'email_enabled': True,
                'slack_enabled': False,
                'webhook_enabled': False,
                'escalation_enabled': True,
                'escalation_timeout': 1800  # 30 minutes
            },
            'dashboards': {
                'grafana_enabled': True,
                'cloudwatch_enabled': True,
                'custom_dashboard': True,
                'business_kpis': True,
                'refresh_interval': 30  # seconds
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value

class MetricsCollector:
    """Centralized metrics collection system."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_store = {}
        self.active_collectors = {}
        self.collection_enabled = True
        self.collection_interval = config.get('monitoring.collection_interval', 60)
        
    def register_collector(self, name: str, collector_func, interval: Optional[int] = None):
        """Register a metrics collector function."""
        self.active_collectors[name] = {
            'function': collector_func,
            'interval': interval or self.collection_interval,
            'last_collection': None,
            'enabled': True
        }
        logger.info(f"Registered metrics collector: {name}")
    
    def collect_metrics(self) -> Dict[str, List[MonitoringMetric]]:
        """Collect metrics from all registered collectors."""
        if not self.collection_enabled:
            return {}
        
        collected_metrics = {}
        current_time = datetime.now()
        
        for name, collector_info in self.active_collectors.items():
            if not collector_info['enabled']:
                continue
                
            # Check if it's time to collect
            last_collection = collector_info['last_collection']
            interval = collector_info['interval']
            
            if last_collection is None or (current_time - last_collection).total_seconds() >= interval:
                try:
                    metrics = collector_info['function']()
                    if isinstance(metrics, list):
                        collected_metrics[name] = metrics
                    elif isinstance(metrics, MonitoringMetric):
                        collected_metrics[name] = [metrics]
                    else:
                        logger.warning(f"Invalid metrics format from collector {name}")
                    
                    collector_info['last_collection'] = current_time
                    
                except Exception as e:
                    logger.error(f"Error collecting metrics from {name}: {e}")
        
        # Store metrics
        self._store_metrics(collected_metrics)
        return collected_metrics
    
    def _store_metrics(self, metrics: Dict[str, List[MonitoringMetric]]):
        """Store collected metrics."""
        for collector_name, metric_list in metrics.items():
            if collector_name not in self.metrics_store:
                self.metrics_store[collector_name] = []
            
            self.metrics_store[collector_name].extend(metric_list)
            
            # Maintain size limits
            max_metrics = 1000  # Keep last 1000 metrics per collector
            if len(self.metrics_store[collector_name]) > max_metrics:
                self.metrics_store[collector_name] = self.metrics_store[collector_name][-max_metrics:]
    
    def get_metrics(self, collector_name: str, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[MonitoringMetric]:
        """Get stored metrics for a collector."""
        metrics = self.metrics_store.get(collector_name, [])
        
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
    
    def get_latest_metric(self, collector_name: str, metric_name: str) -> Optional[MonitoringMetric]:
        """Get the latest metric value."""
        metrics = self.metrics_store.get(collector_name, [])
        
        for metric in reversed(metrics):
            if metric.name == metric_name:
                return metric
        
        return None
    
    def enable_collector(self, name: str):
        """Enable a specific collector."""
        if name in self.active_collectors:
            self.active_collectors[name]['enabled'] = True
            logger.info(f"Enabled collector: {name}")
    
    def disable_collector(self, name: str):
        """Disable a specific collector."""
        if name in self.active_collectors:
            self.active_collectors[name]['enabled'] = False
            logger.info(f"Disabled collector: {name}")

class HealthChecker:
    """System health monitoring."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.health_checks = {}
        self.last_health_check = None
        self.health_status = MonitoringStatus.HEALTHY
        
    def register_health_check(self, name: str, check_func):
        """Register a health check function."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def run_health_checks(self) -> Dict[str, MonitoringStatus]:
        """Run all registered health checks."""
        results = {}
        overall_status = MonitoringStatus.HEALTHY
        
        for name, check_func in self.health_checks.items():
            try:
                status = check_func()
                results[name] = status
                
                # Update overall status
                if status == MonitoringStatus.CRITICAL:
                    overall_status = MonitoringStatus.CRITICAL
                elif status == MonitoringStatus.WARNING and overall_status == MonitoringStatus.HEALTHY:
                    overall_status = MonitoringStatus.WARNING
                elif status == MonitoringStatus.DEGRADED and overall_status == MonitoringStatus.HEALTHY:
                    overall_status = MonitoringStatus.DEGRADED
                    
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results[name] = MonitoringStatus.OFFLINE
                if overall_status != MonitoringStatus.CRITICAL:
                    overall_status = MonitoringStatus.CRITICAL
        
        self.health_status = overall_status
        self.last_health_check = datetime.now()
        
        return results
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        return {
            'overall_status': self.health_status.value,
            'last_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'individual_checks': {name: status.value for name, status in self.run_health_checks().items()},
            'uptime_seconds': self._get_uptime(),
            'system_resources': self._get_system_resources()
        }
    
    def _get_uptime(self) -> float:
        """Get system uptime in seconds."""
        try:
            return time.time() - psutil.boot_time()
        except:
            return 0.0
    
    def _get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            }
        except:
            return {'cpu_percent': 0, 'memory_percent': 0, 'disk_percent': 0, 'load_average': 0}

class MonitoringManager:
    """Main monitoring system manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize monitoring manager."""
        self.config = MonitoringConfig(config_path)
        self.metrics_collector = MetricsCollector(self.config)
        self.health_checker = HealthChecker(self.config)
        self.alert_manager = AlertManager(self.config) if AlertManager else None
        
        # Component managers
        self.model_monitors = {}
        self.data_monitors = {}
        self.dashboard_managers = {}
        
        # Background tasks
        self.monitoring_enabled = True
        self.background_tasks = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("MonitoringManager initialized")
    
    def start_monitoring(self):
        """Start all monitoring services."""
        if not self.monitoring_enabled:
            return
        
        logger.info("Starting monitoring services...")
        
        # Start metrics collection
        self._start_metrics_collection()
        
        # Start health checks
        self._start_health_checks()
        
        # Start alert processing
        if self.alert_manager:
            self._start_alert_processing()
        
        logger.info("Monitoring services started")
    
    def stop_monitoring(self):
        """Stop all monitoring services."""
        logger.info("Stopping monitoring services...")
        
        self.monitoring_enabled = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Monitoring services stopped")
    
    def _start_metrics_collection(self):
        """Start background metrics collection."""
        def collect_loop():
            while self.monitoring_enabled:
                try:
                    self.metrics_collector.collect_metrics()
                    time.sleep(self.config.get('monitoring.collection_interval', 60))
                except Exception as e:
                    logger.error(f"Error in metrics collection loop: {e}")
                    time.sleep(30)  # Wait before retrying
        
        future = self.executor.submit(collect_loop)
        self.background_tasks.append(future)
    
    def _start_health_checks(self):
        """Start background health checks."""
        def health_check_loop():
            while self.monitoring_enabled:
                try:
                    self.health_checker.run_health_checks()
                    time.sleep(self.config.get('monitoring.health_check_interval', 30))
                except Exception as e:
                    logger.error(f"Error in health check loop: {e}")
                    time.sleep(30)  # Wait before retrying
        
        future = self.executor.submit(health_check_loop)
        self.background_tasks.append(future)
    
    def _start_alert_processing(self):
        """Start background alert processing."""
        if not self.alert_manager:
            return
            
        def alert_processing_loop():
            while self.monitoring_enabled:
                try:
                    # Process pending alerts
                    self.alert_manager.process_alerts()
                    time.sleep(10)  # Check every 10 seconds
                except Exception as e:
                    logger.error(f"Error in alert processing loop: {e}")
                    time.sleep(30)  # Wait before retrying
        
        future = self.executor.submit(alert_processing_loop)
        self.background_tasks.append(future)
    
    def register_model_monitor(self, model_id: str, monitor: Any):
        """Register a model monitor."""
        self.model_monitors[model_id] = monitor
        logger.info(f"Registered model monitor for: {model_id}")
    
    def register_data_monitor(self, data_source: str, monitor: Any):
        """Register a data monitor."""
        self.data_monitors[data_source] = monitor
        logger.info(f"Registered data monitor for: {data_source}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        return {
            'system_health': self.health_checker.get_system_health(),
            'monitoring_enabled': self.monitoring_enabled,
            'active_collectors': len(self.metrics_collector.active_collectors),
            'model_monitors': len(self.model_monitors),
            'data_monitors': len(self.data_monitors),
            'last_metrics_collection': self.metrics_collector.metrics_store.get('system', [{}])[-1].timestamp.isoformat() if self.metrics_collector.metrics_store.get('system') else None
        }

# Global monitoring manager instance
_monitoring_manager = None

def get_monitoring_manager(config_path: Optional[str] = None) -> MonitoringManager:
    """Get or create global monitoring manager instance."""
    global _monitoring_manager
    if _monitoring_manager is None:
        _monitoring_manager = MonitoringManager(config_path)
    return _monitoring_manager

def start_monitoring(config_path: Optional[str] = None):
    """Start global monitoring services."""
    manager = get_monitoring_manager(config_path)
    manager.start_monitoring()

def stop_monitoring():
    """Stop global monitoring services."""
    global _monitoring_manager
    if _monitoring_manager:
        _monitoring_manager.stop_monitoring()

def get_system_health() -> Dict[str, Any]:
    """Get current system health status."""
    manager = get_monitoring_manager()
    return manager.get_monitoring_status()

# Utility functions for creating standard metrics
def create_accuracy_metric(value: float, model_id: str, threshold: float = 0.85) -> MonitoringMetric:
    """Create a standard accuracy metric."""
    return MonitoringMetric(
        name="model_accuracy",
        value=value,
        unit="ratio",
        labels={"model_id": model_id},
        threshold_min=threshold
    )

def create_latency_metric(value: float, endpoint: str, threshold: float = 100.0) -> MonitoringMetric:
    """Create a standard latency metric."""
    return MonitoringMetric(
        name="inference_latency",
        value=value,
        unit="milliseconds",
        labels={"endpoint": endpoint},
        threshold_max=threshold
    )

def create_error_rate_metric(value: float, service: str, threshold: float = 0.01) -> MonitoringMetric:
    """Create a standard error rate metric."""
    return MonitoringMetric(
        name="error_rate",
        value=value,
        unit="ratio",
        labels={"service": service},
        threshold_max=threshold
    )

def create_resource_metric(metric_name: str, value: float, unit: str, threshold: float) -> MonitoringMetric:
    """Create a standard resource utilization metric."""
    return MonitoringMetric(
        name=metric_name,
        value=value,
        unit=unit,
        threshold_max=threshold
    )

# Export all public components
__all__ = [
    # Enums
    'MonitoringStatus',
    'AlertSeverity',
    
    # Data classes
    'MonitoringMetric',
    'MonitoringAlert',
    
    # Core classes
    'MonitoringConfig',
    'MetricsCollector',
    'HealthChecker',
    'MonitoringManager',
    
    # Model monitoring
    'PerformanceMonitor',
    'DriftDetector',
    'AccuracyTracker',
    'ResourceMonitor',
    
    # Data monitoring
    'DataQualityMonitor',
    'SchemaValidator',
    'AnomalyDetector',
    'BiasDetector',
    
    # Alerts
    'AlertManager',
    'NotificationService',
    'EscalationPolicy',
    'AlertRules',
    
    # Dashboards
    'GrafanaDashboard',
    'CloudWatchMetrics',
    'CustomMetrics',
    'BusinessKPIs',
    
    # Utility functions
    'get_monitoring_manager',
    'start_monitoring',
    'stop_monitoring',
    'get_system_health',
    'create_accuracy_metric',
    'create_latency_metric',
    'create_error_rate_metric',
    'create_resource_metric',
    
    # Version info
    '__version__',
    '__author__'
]

# Module initialization
logger.info(f"BatteryMind Monitoring module initialized (v{__version__})")
logger.info("Available monitoring components: model_monitoring, data_monitoring, alerts, dashboards")

# Health check for module initialization
def module_health_check() -> MonitoringStatus:
    """Health check for monitoring module."""
    try:
        # Check if core components are available
        if all([MonitoringConfig, MetricsCollector, HealthChecker, MonitoringManager]):
            return MonitoringStatus.HEALTHY
        else:
            return MonitoringStatus.DEGRADED
    except Exception:
        return MonitoringStatus.CRITICAL

# Register module health check
manager = get_monitoring_manager()
manager.health_checker.register_health_check("monitoring_module", module_health_check)
