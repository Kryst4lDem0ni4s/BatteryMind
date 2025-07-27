"""
BatteryMind - Monitoring Dashboards Module
Comprehensive monitoring dashboard system for battery management AI models with
real-time visualization, custom metrics, and business intelligence integration.

Features:
- Multi-platform dashboard integration (Grafana, CloudWatch, Custom)
- Real-time metrics visualization and alerting
- Business KPI tracking and reporting
- Custom dashboard creation and management
- Performance monitoring and analytics
- Interactive data exploration and filtering

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

# Import all dashboard components
from .grafana_dashboard import (
    GrafanaDashboard,
    GrafanaPanel,
    GrafanaQuery,
    GrafanaAlert,
    GrafanaDashboardManager,
    PanelType,
    DataSourceType,
    VisualizationType
)

from .cloudwatch_metrics import (
    CloudWatchMetrics,
    CloudWatchDashboard,
    CloudWatchWidget,
    CloudWatchAlarm,
    MetricType,
    StatisticType,
    ComparisonOperator
)

from .custom_metrics import (
    CustomMetrics,
    MetricCollector,
    MetricProcessor,
    MetricAggregator,
    CustomDashboard,
    MetricDefinition,
    AggregationType
)

from .business_kpis import (
    BusinessKPIs,
    KPIDefinition,
    KPICalculator,
    KPITracker,
    PerformanceIndicator,
    ROICalculator,
    SustainabilityMetrics
)

# Configure logging
logger = logging.getLogger(__name__)

# Module metadata
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"

# Export all public classes and functions
__all__ = [
    # Grafana Components
    "GrafanaDashboard",
    "GrafanaPanel", 
    "GrafanaQuery",
    "GrafanaAlert",
    "GrafanaDashboardManager",
    "PanelType",
    "DataSourceType",
    "VisualizationType",
    
    # CloudWatch Components
    "CloudWatchMetrics",
    "CloudWatchDashboard",
    "CloudWatchWidget",
    "CloudWatchAlarm",
    "MetricType",
    "StatisticType",
    "ComparisonOperator",
    
    # Custom Metrics Components
    "CustomMetrics",
    "MetricCollector",
    "MetricProcessor", 
    "MetricAggregator",
    "CustomDashboard",
    "MetricDefinition",
    "AggregationType",
    
    # Business KPI Components
    "BusinessKPIs",
    "KPIDefinition",
    "KPICalculator",
    "KPITracker",
    "PerformanceIndicator",
    "ROICalculator",
    "SustainabilityMetrics",
    
    # Utility Functions
    "create_monitoring_dashboard",
    "create_battery_dashboard",
    "create_fleet_dashboard",
    "create_ai_monitoring_dashboard",
    "create_business_intelligence_dashboard",
    
    # Factory Functions
    "create_grafana_manager",
    "create_cloudwatch_manager",
    "create_custom_metrics_collector",
    "create_kpi_tracker",
    
    # Configuration Classes
    "DashboardConfig",
    "MonitoringConfig",
    "VisualizationConfig",
    "AlertingConfig"
]

class DashboardType(Enum):
    """Types of monitoring dashboards."""
    OPERATIONAL = "operational"
    BUSINESS = "business"
    TECHNICAL = "technical"
    EXECUTIVE = "executive"
    REAL_TIME = "real_time"
    HISTORICAL = "historical"

class RefreshInterval(Enum):
    """Dashboard refresh intervals."""
    REAL_TIME = "5s"
    FAST = "10s"
    NORMAL = "30s"
    SLOW = "1m"
    BATCH = "5m"
    HOURLY = "1h"

@dataclass
class DashboardConfig:
    """Configuration for monitoring dashboards."""
    
    # Basic settings
    name: str = "BatteryMind Dashboard"
    description: str = "Comprehensive battery monitoring dashboard"
    dashboard_type: DashboardType = DashboardType.OPERATIONAL
    refresh_interval: RefreshInterval = RefreshInterval.NORMAL
    
    # Display settings
    theme: str = "dark"  # dark, light
    timezone: str = "UTC"
    time_range: str = "1h"  # 1h, 6h, 24h, 7d, 30d
    
    # Panel configuration
    panel_height: int = 300
    panel_width: int = 12  # Grid units (1-12)
    show_legends: bool = True
    show_tooltips: bool = True
    
    # Data settings
    data_source: str = "prometheus"
    metric_prefix: str = "batterymind"
    aggregation_interval: str = "1m"
    
    # Alerting
    enable_alerts: bool = True
    alert_threshold_warning: float = 80.0
    alert_threshold_critical: float = 95.0
    
    # Access control
    public_dashboard: bool = False
    allowed_users: List[str] = field(default_factory=list)
    allowed_roles: List[str] = field(default_factory=lambda: ["admin", "operator"])

@dataclass
class MonitoringConfig:
    """Configuration for monitoring system."""
    
    # Data collection
    collection_interval_seconds: int = 30
    retention_days: int = 90
    batch_size: int = 1000
    
    # Performance settings
    max_concurrent_queries: int = 10
    query_timeout_seconds: int = 30
    cache_duration_minutes: int = 5
    
    # Storage settings
    storage_backend: str = "prometheus"  # prometheus, influxdb, cloudwatch
    compression_enabled: bool = True
    archival_enabled: bool = True
    
    # Networking
    api_endpoint: str = "http://localhost:9090"
    authentication_enabled: bool = True
    ssl_verify: bool = True

@dataclass
class VisualizationConfig:
    """Configuration for data visualization."""
    
    # Chart settings
    default_chart_type: str = "line"
    color_palette: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ])
    
    # Animation settings
    enable_animations: bool = True
    animation_duration: int = 300
    
    # Interaction settings
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_selection: bool = True
    
    # Performance settings
    max_data_points: int = 10000
    decimation_threshold: int = 1000
    lazy_loading: bool = True

@dataclass
class AlertingConfig:
    """Configuration for dashboard alerting."""
    
    # Alert settings
    enable_dashboard_alerts: bool = True
    alert_evaluation_interval: int = 60  # seconds
    max_alerts_per_dashboard: int = 50
    
    # Notification settings
    notification_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    notification_templates: Dict[str, str] = field(default_factory=dict)
    
    # Escalation settings
    enable_escalation: bool = True
    escalation_delay_minutes: int = 15
    max_escalation_levels: int = 3

# Utility Functions
def create_monitoring_dashboard(dashboard_config: DashboardConfig,
                              monitoring_config: MonitoringConfig) -> Dict[str, Any]:
    """
    Create a comprehensive monitoring dashboard configuration.
    
    Args:
        dashboard_config: Dashboard configuration
        monitoring_config: Monitoring system configuration
        
    Returns:
        Dashboard configuration dictionary
    """
    dashboard = {
        "id": f"batterymind_{dashboard_config.dashboard_type.value}",
        "title": dashboard_config.name,
        "description": dashboard_config.description,
        "tags": ["batterymind", "monitoring", dashboard_config.dashboard_type.value],
        "timezone": dashboard_config.timezone,
        "refresh": dashboard_config.refresh_interval.value,
        "time": {
            "from": f"now-{dashboard_config.time_range}",
            "to": "now"
        },
        "panels": [],
        "templating": {
            "list": []
        },
        "annotations": {
            "list": []
        },
        "variables": [
            {
                "name": "instance",
                "type": "query",
                "query": f'{dashboard_config.metric_prefix}_info',
                "refresh": 1,
                "includeAll": True,
                "multi": True
            }
        ]
    }
    
    return dashboard

def create_battery_dashboard() -> Dict[str, Any]:
    """Create a battery-specific monitoring dashboard."""
    
    config = DashboardConfig(
        name="Battery Health Monitoring",
        description="Real-time battery health and performance monitoring",
        dashboard_type=DashboardType.OPERATIONAL,
        refresh_interval=RefreshInterval.NORMAL
    )
    
    monitoring_config = MonitoringConfig()
    dashboard = create_monitoring_dashboard(config, monitoring_config)
    
    # Add battery-specific panels
    battery_panels = [
        {
            "id": 1,
            "title": "Battery State of Health (SoH)",
            "type": "stat",
            "targets": [
                {
                    "expr": "batterymind_battery_soh",
                    "legendFormat": "{{battery_id}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": 0},
                            {"color": "yellow", "value": 70},
                            {"color": "green", "value": 85}
                        ]
                    }
                }
            },
            "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
        },
        {
            "id": 2,
            "title": "Battery State of Charge (SoC)",
            "type": "gauge",
            "targets": [
                {
                    "expr": "batterymind_battery_soc",
                    "legendFormat": "{{battery_id}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": 0},
                            {"color": "yellow", "value": 20},
                            {"color": "green", "value": 50}
                        ]
                    }
                }
            },
            "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0}
        },
        {
            "id": 3,
            "title": "Battery Temperature",
            "type": "timeseries",
            "targets": [
                {
                    "expr": "batterymind_battery_temperature",
                    "legendFormat": "{{battery_id}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "celsius",
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth",
                        "fillOpacity": 0.1
                    }
                }
            },
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
        },
        {
            "id": 4,
            "title": "Charging/Discharging Current",
            "type": "timeseries",
            "targets": [
                {
                    "expr": "batterymind_battery_current",
                    "legendFormat": "{{battery_id}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "amp",
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "linear"
                    }
                }
            },
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16}
        }
    ]
    
    dashboard["panels"] = battery_panels
    return dashboard

def create_fleet_dashboard() -> Dict[str, Any]:
    """Create a fleet management monitoring dashboard."""
    
    config = DashboardConfig(
        name="Fleet Management Dashboard",
        description="Fleet-wide battery and vehicle monitoring",
        dashboard_type=DashboardType.OPERATIONAL,
        refresh_interval=RefreshInterval.FAST
    )
    
    monitoring_config = MonitoringConfig()
    dashboard = create_monitoring_dashboard(config, monitoring_config)
    
    # Add fleet-specific panels
    fleet_panels = [
        {
            "id": 1,
            "title": "Fleet Overview",
            "type": "stat",
            "targets": [
                {
                    "expr": "count(batterymind_vehicle_status)",
                    "legendFormat": "Total Vehicles"
                },
                {
                    "expr": "count(batterymind_vehicle_status == 1)",
                    "legendFormat": "Active Vehicles"
                },
                {
                    "expr": "count(batterymind_vehicle_charging)",
                    "legendFormat": "Charging"
                }
            ],
            "gridPos": {"h": 4, "w": 12, "x": 0, "y": 0}
        },
        {
            "id": 2,
            "title": "Fleet Energy Consumption",
            "type": "timeseries",
            "targets": [
                {
                    "expr": "sum(rate(batterymind_energy_consumed[5m]))",
                    "legendFormat": "Total Energy Consumption"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "kwh",
                    "custom": {
                        "drawStyle": "line",
                        "fillOpacity": 0.2
                    }
                }
            },
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4}
        },
        {
            "id": 3,
            "title": "Vehicle Locations",
            "type": "geomap",
            "targets": [
                {
                    "expr": "batterymind_vehicle_location",
                    "legendFormat": "{{vehicle_id}}"
                }
            ],
            "gridPos": {"h": 10, "w": 12, "x": 0, "y": 12}
        }
    ]
    
    dashboard["panels"] = fleet_panels
    return dashboard

def create_ai_monitoring_dashboard() -> Dict[str, Any]:
    """Create an AI model monitoring dashboard."""
    
    config = DashboardConfig(
        name="AI Model Performance Monitoring",
        description="AI model accuracy, latency, and resource monitoring",
        dashboard_type=DashboardType.TECHNICAL,
        refresh_interval=RefreshInterval.NORMAL
    )
    
    monitoring_config = MonitoringConfig()
    dashboard = create_monitoring_dashboard(config, monitoring_config)
    
    # Add AI-specific panels
    ai_panels = [
        {
            "id": 1,
            "title": "Model Accuracy",
            "type": "stat",
            "targets": [
                {
                    "expr": "batterymind_model_accuracy",
                    "legendFormat": "{{model_name}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": 0},
                            {"color": "yellow", "value": 90},
                            {"color": "green", "value": 95}
                        ]
                    }
                }
            },
            "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
        },
        {
            "id": 2,
            "title": "Inference Latency",
            "type": "timeseries",
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, batterymind_inference_duration_seconds_bucket)",
                    "legendFormat": "95th percentile"
                },
                {
                    "expr": "histogram_quantile(0.50, batterymind_inference_duration_seconds_bucket)",
                    "legendFormat": "Median"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "s"
                }
            },
            "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0}
        },
        {
            "id": 3,
            "title": "Model Resource Usage",
            "type": "timeseries",
            "targets": [
                {
                    "expr": "batterymind_model_memory_usage_bytes",
                    "legendFormat": "Memory Usage"
                },
                {
                    "expr": "rate(batterymind_model_cpu_usage_seconds[5m])",
                    "legendFormat": "CPU Usage"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
        }
    ]
    
    dashboard["panels"] = ai_panels
    return dashboard

def create_business_intelligence_dashboard() -> Dict[str, Any]:
    """Create a business intelligence dashboard."""
    
    config = DashboardConfig(
        name="Business Intelligence Dashboard",
        description="Business KPIs, ROI, and sustainability metrics",
        dashboard_type=DashboardType.BUSINESS,
        refresh_interval=RefreshInterval.SLOW
    )
    
    monitoring_config = MonitoringConfig()
    dashboard = create_monitoring_dashboard(config, monitoring_config)
    
    # Add business intelligence panels
    bi_panels = [
        {
            "id": 1,
            "title": "Cost Savings",
            "type": "stat",
            "targets": [
                {
                    "expr": "batterymind_cost_savings_total",
                    "legendFormat": "Total Savings"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "currencyUSD",
                    "custom": {
                        "displayMode": "basic"
                    }
                }
            },
            "gridPos": {"h": 4, "w": 3, "x": 0, "y": 0}
        },
        {
            "id": 2,
            "title": "Battery Life Extension",
            "type": "stat",
            "targets": [
                {
                    "expr": "avg(batterymind_battery_life_extension_percent)",
                    "legendFormat": "Average Extension"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent"
                }
            },
            "gridPos": {"h": 4, "w": 3, "x": 3, "y": 0}
        },
        {
            "id": 3,
            "title": "Carbon Footprint Reduction",
            "type": "stat",
            "targets": [
                {
                    "expr": "batterymind_carbon_savings_kg_co2",
                    "legendFormat": "CO2 Saved"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "kg"
                }
            },
            "gridPos": {"h": 4, "w": 3, "x": 6, "y": 0}
        },
        {
            "id": 4,
            "title": "ROI Trend",
            "type": "timeseries",
            "targets": [
                {
                    "expr": "batterymind_roi_percent",
                    "legendFormat": "ROI %"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent"
                }
            },
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4}
        }
    ]
    
    dashboard["panels"] = bi_panels
    return dashboard

# Factory Functions
def create_grafana_manager(grafana_url: str, api_key: str) -> GrafanaDashboardManager:
    """Create a Grafana dashboard manager."""
    return GrafanaDashboardManager(
        base_url=grafana_url,
        api_key=api_key,
        timeout=30
    )

def create_cloudwatch_manager(region: str = "us-west-2") -> CloudWatchMetrics:
    """Create a CloudWatch metrics manager."""
    return CloudWatchMetrics(
        region=region,
        namespace="BatteryMind"
    )

def create_custom_metrics_collector(collection_interval: int = 30) -> MetricCollector:
    """Create a custom metrics collector."""
    return MetricCollector(
        collection_interval=collection_interval,
        batch_size=1000,
        enable_caching=True
    )

def create_kpi_tracker() -> KPITracker:
    """Create a business KPI tracker."""
    return KPITracker(
        calculation_interval_hours=1,
        retention_days=365,
        enable_forecasting=True
    )

# Dashboard Templates
DASHBOARD_TEMPLATES = {
    "battery_health": create_battery_dashboard,
    "fleet_management": create_fleet_dashboard,
    "ai_monitoring": create_ai_monitoring_dashboard,
    "business_intelligence": create_business_intelligence_dashboard
}

def get_dashboard_template(template_name: str) -> Dict[str, Any]:
    """
    Get a predefined dashboard template.
    
    Args:
        template_name: Name of the template
        
    Returns:
        Dashboard configuration dictionary
    """
    if template_name not in DASHBOARD_TEMPLATES:
        raise ValueError(f"Unknown dashboard template: {template_name}")
    
    return DASHBOARD_TEMPLATES[template_name]()

def list_available_templates() -> List[str]:
    """List all available dashboard templates."""
    return list(DASHBOARD_TEMPLATES.keys())

# Module initialization
def initialize_dashboard_system(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Initialize the complete dashboard monitoring system.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Initialized dashboard system components
    """
    try:
        # Load configuration if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Create configuration objects from loaded data
            dashboard_config = DashboardConfig(**config_data.get('dashboard', {}))
            monitoring_config = MonitoringConfig(**config_data.get('monitoring', {}))
            visualization_config = VisualizationConfig(**config_data.get('visualization', {}))
            alerting_config = AlertingConfig(**config_data.get('alerting', {}))
        else:
            # Use default configurations
            dashboard_config = DashboardConfig()
            monitoring_config = MonitoringConfig()
            visualization_config = VisualizationConfig()
            alerting_config = AlertingConfig()
        
        # Initialize components
        components = {
            'dashboard_config': dashboard_config,
            'monitoring_config': monitoring_config,
            'visualization_config': visualization_config,
            'alerting_config': alerting_config,
            'templates': DASHBOARD_TEMPLATES
        }
        
        logger.info(f"BatteryMind Dashboard System initialized successfully")
        logger.info(f"Available templates: {list(DASHBOARD_TEMPLATES.keys())}")
        
        return components
        
    except Exception as e:
        logger.error(f"Failed to initialize dashboard system: {e}")
        raise

# Default configurations for different environments
DEVELOPMENT_CONFIG = DashboardConfig(
    name="BatteryMind Development Dashboard",
    refresh_interval=RefreshInterval.FAST,
    time_range="1h",
    enable_alerts=False
)

STAGING_CONFIG = DashboardConfig(
    name="BatteryMind Staging Dashboard",
    refresh_interval=RefreshInterval.NORMAL,
    time_range="6h",
    enable_alerts=True
)

PRODUCTION_CONFIG = DashboardConfig(
    name="BatteryMind Production Dashboard",
    refresh_interval=RefreshInterval.NORMAL,
    time_range="24h",
    enable_alerts=True,
    public_dashboard=False
)

# Export environment configs
__all__.extend([
    "initialize_dashboard_system",
    "get_dashboard_template",
    "list_available_templates",
    "DEVELOPMENT_CONFIG",
    "STAGING_CONFIG", 
    "PRODUCTION_CONFIG"
])

# Log module initialization
logger.info(f"BatteryMind Dashboard Module v{__version__} loaded")
logger.info("Available dashboard types: Grafana, CloudWatch, Custom, Business KPIs")
