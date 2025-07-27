"""
BatteryMind - Monitoring Alerts Module
Comprehensive alerting system for battery management AI models with intelligent
routing, escalation, and multi-channel notification capabilities.

This module provides:
- Multi-level alert classification and routing
- Intelligent alert correlation and deduplication  
- Escalation policies with time-based triggers
- Multi-channel notifications (email, SMS, Slack, webhook)
- Alert suppression and snoozing capabilities
- Performance and business impact alerting
- Integration with monitoring and bias detection systems

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import json
from pathlib import Path

# Import all alert components
from .alert_manager import (
    AlertManager,
    Alert,
    AlertSeverity,
    AlertStatus,
    AlertChannel,
    AlertRule,
    AlertConfiguration
)

from .notification_service import (
    NotificationService,
    NotificationChannel,
    EmailNotifier,
    SlackNotifier,
    SMSNotifier,
    WebhookNotifier,
    NotificationTemplate,
    NotificationResult
)

from .escalation_policy import (
    EscalationPolicy,
    EscalationLevel,
    EscalationRule,
    EscalationManager,
    OnCallSchedule,
    EscalationAction
)

from .alert_rules import (
    AlertRulesEngine,
    AlertCondition,
    AlertThreshold,
    CompositeRule,
    MetricRule,
    AnomalyRule,
    PerformanceRule,
    BusinessRule
)

# Configure logging
logger = logging.getLogger(__name__)

# Module metadata
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"

# Export all public classes and functions
__all__ = [
    # Core Alert Management
    "AlertManager",
    "Alert", 
    "AlertSeverity",
    "AlertStatus",
    "AlertChannel",
    "AlertRule",
    "AlertConfiguration",
    
    # Notification System
    "NotificationService",
    "NotificationChannel",
    "EmailNotifier",
    "SlackNotifier", 
    "SMSNotifier",
    "WebhookNotifier",
    "NotificationTemplate",
    "NotificationResult",
    
    # Escalation Management
    "EscalationPolicy",
    "EscalationLevel",
    "EscalationRule",
    "EscalationManager",
    "OnCallSchedule", 
    "EscalationAction",
    
    # Alert Rules Engine
    "AlertRulesEngine",
    "AlertCondition",
    "AlertThreshold",
    "CompositeRule",
    "MetricRule",
    "AnomalyRule",
    "PerformanceRule",
    "BusinessRule",
    
    # Utility Functions
    "create_alert_system",
    "create_default_alert_config",
    "setup_battery_alerts",
    "setup_model_performance_alerts",
    "setup_business_alerts",
    
    # Factory Functions
    "create_email_notifier",
    "create_slack_notifier", 
    "create_escalation_policy",
    "create_alert_rules_engine",
    
    # Configuration Classes
    "AlertSystemConfig",
    "NotificationConfig",
    "EscalationConfig",
    "RulesConfig"
]

# Configuration Classes
@dataclass
class AlertSystemConfig:
    """Main configuration for the alerting system."""
    
    # System settings
    system_name: str = "BatteryMind"
    environment: str = "production"  # dev, staging, production
    timezone: str = "UTC"
    
    # Alert processing
    max_alerts_per_minute: int = 100
    alert_retention_days: int = 90
    enable_deduplication: bool = True
    deduplication_window_minutes: int = 5
    
    # Performance settings
    processing_timeout_seconds: int = 30
    batch_processing: bool = True
    batch_size: int = 50
    
    # Storage settings
    persist_alerts: bool = True
    alert_storage_path: str = "./alerts_data"
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_collection_interval: int = 60
    
    # Security settings
    encrypt_notifications: bool = True
    require_authentication: bool = True
    
    # Feature flags
    enable_smart_routing: bool = True
    enable_ml_correlation: bool = True
    enable_auto_resolution: bool = True

@dataclass  
class NotificationConfig:
    """Configuration for notification channels."""
    
    # Email settings
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_from: str = ""
    
    # Slack settings
    slack_webhook_url: str = ""
    slack_token: str = ""
    slack_channel: str = "#alerts"
    
    # SMS settings  
    sms_provider: str = "twilio"  # twilio, aws_sns
    sms_api_key: str = ""
    sms_api_secret: str = ""
    sms_from_number: str = ""
    
    # Webhook settings
    webhook_urls: List[str] = field(default_factory=list)
    webhook_timeout: int = 10
    webhook_retry_count: int = 3
    
    # Template settings
    template_directory: str = "./templates"
    default_language: str = "en"
    
    # Rate limiting
    email_rate_limit: int = 60  # per hour
    sms_rate_limit: int = 10    # per hour
    slack_rate_limit: int = 300  # per hour

@dataclass
class EscalationConfig:
    """Configuration for escalation policies."""
    
    # Default escalation settings
    default_escalation_delay: int = 30  # minutes  
    max_escalation_levels: int = 5
    auto_escalate_critical: bool = True
    
    # Business hours
    business_hours_start: str = "09:00"
    business_hours_end: str = "17:00"
    business_days: List[str] = field(default_factory=lambda: 
        ["monday", "tuesday", "wednesday", "thursday", "friday"])
    
    # On-call settings
    enable_on_call: bool = True
    on_call_rotation_days: int = 7
    
    # Escalation behavior
    escalate_on_no_response: bool = True
    response_timeout_minutes: int = 15
    auto_resolve_after_hours: int = 24

@dataclass
class RulesConfig:
    """Configuration for alert rules engine."""
    
    # Rule processing
    evaluation_interval_seconds: int = 60
    enable_rule_chaining: bool = True
    max_rule_complexity: int = 10
    
    # Thresholds
    default_cpu_threshold: float = 80.0
    default_memory_threshold: float = 85.0
    default_latency_threshold: float = 1000.0  # ms
    default_error_rate_threshold: float = 5.0  # percent
    
    # Business metrics
    battery_health_threshold: float = 80.0  # percent
    prediction_accuracy_threshold: float = 95.0  # percent
    model_drift_threshold: float = 0.1
    
    # Battery-specific thresholds
    voltage_min_threshold: float = 2.5  # volts
    voltage_max_threshold: float = 4.2  # volts  
    temperature_min_threshold: float = -10.0  # celsius
    temperature_max_threshold: float = 60.0   # celsius
    charge_rate_threshold: float = 2.0  # C-rate
    
    # Rule categories
    enable_performance_rules: bool = True
    enable_safety_rules: bool = True
    enable_business_rules: bool = True
    enable_ml_rules: bool = True

# Utility Functions
def create_alert_system(config: Optional[AlertSystemConfig] = None) -> AlertManager:
    """
    Create a complete alert system with default configuration.
    
    Args:
        config: Optional alert system configuration
        
    Returns:
        Configured AlertManager instance
    """
    if config is None:
        config = AlertSystemConfig()
    
    # Create notification service
    notification_config = NotificationConfig()
    notification_service = NotificationService(notification_config)
    
    # Create escalation manager
    escalation_config = EscalationConfig()
    escalation_manager = EscalationManager(escalation_config)
    
    # Create rules engine
    rules_config = RulesConfig()
    rules_engine = AlertRulesEngine(rules_config)
    
    # Create alert configuration
    alert_config = AlertConfiguration(
        deduplication_enabled=config.enable_deduplication,
        deduplication_window_minutes=config.deduplication_window_minutes,
        max_alerts_per_minute=config.max_alerts_per_minute,
        alert_retention_days=config.alert_retention_days
    )
    
    # Create and configure alert manager
    alert_manager = AlertManager(
        config=alert_config,
        notification_service=notification_service,
        escalation_manager=escalation_manager,
        rules_engine=rules_engine
    )
    
    logger.info("Alert system created successfully")
    return alert_manager

def create_default_alert_config() -> AlertConfiguration:
    """Create default alert configuration optimized for battery systems."""
    
    return AlertConfiguration(
        # Core settings
        deduplication_enabled=True,
        deduplication_window_minutes=5,
        max_alerts_per_minute=50,
        alert_retention_days=90,
        
        # Severity levels
        severity_levels={
            AlertSeverity.CRITICAL: {
                'color': '#FF0000',
                'icon': '🚨',
                'notification_channels': ['email', 'sms', 'slack'],
                'escalate_immediately': True
            },
            AlertSeverity.HIGH: {
                'color': '#FF8C00', 
                'icon': '⚠️',
                'notification_channels': ['email', 'slack'],
                'escalate_after_minutes': 15
            },
            AlertSeverity.MEDIUM: {
                'color': '#FFD700',
                'icon': '⚡',
                'notification_channels': ['slack'],
                'escalate_after_minutes': 60
            },
            AlertSeverity.LOW: {
                'color': '#90EE90',
                'icon': 'ℹ️', 
                'notification_channels': ['slack'],
                'escalate_after_minutes': 240
            }
        },
        
        # Channel configurations
        notification_channels={
            'email': {
                'enabled': True,
                'rate_limit_per_hour': 60,
                'template': 'battery_alert_email.html'
            },
            'slack': {
                'enabled': True,
                'rate_limit_per_hour': 300,
                'template': 'battery_alert_slack.json'
            },
            'sms': {
                'enabled': True,
                'rate_limit_per_hour': 10,
                'template': 'battery_alert_sms.txt'
            }
        }
    )

def setup_battery_alerts(alert_manager: AlertManager) -> None:
    """
    Set up battery-specific alert rules.
    
    Args:
        alert_manager: AlertManager instance to configure
    """
    rules_engine = alert_manager.rules_engine
    
    # Battery Health Alerts
    battery_health_rule = MetricRule(
        name="battery_health_degradation",
        description="Alert when battery health drops below threshold",
        metric_name="battery_soh",
        condition=AlertCondition.LESS_THAN,
        threshold=AlertThreshold(value=80.0, unit="percent"),
        severity=AlertSeverity.HIGH,
        evaluation_window_minutes=5
    )
    rules_engine.add_rule(battery_health_rule)
    
    # Voltage Safety Alerts
    voltage_low_rule = MetricRule(
        name="battery_voltage_low",
        description="Critical low voltage alert",
        metric_name="battery_voltage", 
        condition=AlertCondition.LESS_THAN,
        threshold=AlertThreshold(value=2.5, unit="volts"),
        severity=AlertSeverity.CRITICAL,
        evaluation_window_minutes=1
    )
    rules_engine.add_rule(voltage_low_rule)
    
    voltage_high_rule = MetricRule(
        name="battery_voltage_high",
        description="Critical high voltage alert",
        metric_name="battery_voltage",
        condition=AlertCondition.GREATER_THAN, 
        threshold=AlertThreshold(value=4.2, unit="volts"),
        severity=AlertSeverity.CRITICAL,
        evaluation_window_minutes=1
    )
    rules_engine.add_rule(voltage_high_rule)
    
    # Temperature Safety Alerts
    temp_high_rule = MetricRule(
        name="battery_temperature_high",
        description="High temperature safety alert",
        metric_name="battery_temperature",
        condition=AlertCondition.GREATER_THAN,
        threshold=AlertThreshold(value=60.0, unit="celsius"),
        severity=AlertSeverity.CRITICAL,
        evaluation_window_minutes=2
    )
    rules_engine.add_rule(temp_high_rule)
    
    # Charging Rate Alert
    charge_rate_rule = MetricRule(
        name="high_charge_rate",
        description="High charging rate detected",
        metric_name="charge_rate",
        condition=AlertCondition.GREATER_THAN,
        threshold=AlertThreshold(value=2.0, unit="c_rate"),
        severity=AlertSeverity.MEDIUM,
        evaluation_window_minutes=3
    )
    rules_engine.add_rule(charge_rate_rule)
    
    # Capacity Fade Alert
    capacity_fade_rule = AnomalyRule(
        name="capacity_fade_anomaly",
        description="Unusual capacity fade pattern detected",
        metric_name="battery_capacity",
        anomaly_threshold=2.0,  # Standard deviations
        severity=AlertSeverity.HIGH,
        evaluation_window_minutes=60
    )
    rules_engine.add_rule(capacity_fade_rule)
    
    logger.info("Battery-specific alert rules configured")

def setup_model_performance_alerts(alert_manager: AlertManager) -> None:
    """
    Set up ML model performance alert rules.
    
    Args:
        alert_manager: AlertManager instance to configure
    """
    rules_engine = alert_manager.rules_engine
    
    # Model Accuracy Alert
    accuracy_rule = MetricRule(
        name="model_accuracy_degradation",
        description="Model prediction accuracy below threshold",
        metric_name="model_accuracy", 
        condition=AlertCondition.LESS_THAN,
        threshold=AlertThreshold(value=95.0, unit="percent"),
        severity=AlertSeverity.HIGH,
        evaluation_window_minutes=15
    )
    rules_engine.add_rule(accuracy_rule)
    
    # Model Drift Alert
    drift_rule = MetricRule(
        name="model_data_drift",
        description="Significant data drift detected",
        metric_name="drift_score",
        condition=AlertCondition.GREATER_THAN,
        threshold=AlertThreshold(value=0.1, unit="score"),
        severity=AlertSeverity.MEDIUM,
        evaluation_window_minutes=60
    )
    rules_engine.add_rule(drift_rule)
    
    # Inference Latency Alert
    latency_rule = MetricRule(
        name="inference_latency_high",
        description="Model inference latency too high",
        metric_name="inference_latency",
        condition=AlertCondition.GREATER_THAN,
        threshold=AlertThreshold(value=1000.0, unit="milliseconds"),
        severity=AlertSeverity.MEDIUM,
        evaluation_window_minutes=5
    )
    rules_engine.add_rule(latency_rule)
    
    # Error Rate Alert
    error_rate_rule = MetricRule(
        name="model_error_rate_high",
        description="High model prediction error rate",
        metric_name="prediction_error_rate",
        condition=AlertCondition.GREATER_THAN,
        threshold=AlertThreshold(value=5.0, unit="percent"),
        severity=AlertSeverity.HIGH,
        evaluation_window_minutes=10
    )
    rules_engine.add_rule(error_rate_rule)
    
    # Resource Utilization Alert
    memory_rule = MetricRule(
        name="model_memory_usage_high",
        description="High memory usage by ML models",
        metric_name="model_memory_usage",
        condition=AlertCondition.GREATER_THAN,
        threshold=AlertThreshold(value=85.0, unit="percent"),
        severity=AlertSeverity.MEDIUM,
        evaluation_window_minutes=5
    )
    rules_engine.add_rule(memory_rule)
    
    logger.info("Model performance alert rules configured")

def setup_business_alerts(alert_manager: AlertManager) -> None:
    """
    Set up business impact alert rules.
    
    Args:
        alert_manager: AlertManager instance to configure  
    """
    rules_engine = alert_manager.rules_engine
    
    # Fleet Availability Alert
    fleet_availability_rule = MetricRule(
        name="fleet_availability_low",
        description="Fleet availability below acceptable level",
        metric_name="fleet_availability",
        condition=AlertCondition.LESS_THAN,
        threshold=AlertThreshold(value=95.0, unit="percent"),
        severity=AlertSeverity.HIGH,
        evaluation_window_minutes=10
    )
    rules_engine.add_rule(fleet_availability_rule)
    
    # Cost Threshold Alert
    cost_rule = MetricRule(
        name="daily_cost_threshold",
        description="Daily operational cost exceeds budget",
        metric_name="daily_cost",
        condition=AlertCondition.GREATER_THAN,
        threshold=AlertThreshold(value=1000.0, unit="usd"),
        severity=AlertSeverity.MEDIUM,
        evaluation_window_minutes=60
    )
    rules_engine.add_rule(cost_rule)
    
    # Energy Efficiency Alert  
    efficiency_rule = MetricRule(
        name="energy_efficiency_decline",
        description="Energy efficiency below optimal level",
        metric_name="energy_efficiency",
        condition=AlertCondition.LESS_THAN,
        threshold=AlertThreshold(value=90.0, unit="percent"),
        severity=AlertSeverity.MEDIUM,
        evaluation_window_minutes=30
    )
    rules_engine.add_rule(efficiency_rule)
    
    # Customer Satisfaction Alert
    satisfaction_rule = MetricRule(
        name="customer_satisfaction_low",
        description="Customer satisfaction score declining",
        metric_name="customer_satisfaction",
        condition=AlertCondition.LESS_THAN,
        threshold=AlertThreshold(value=4.0, unit="score_out_of_5"),
        severity=AlertSeverity.HIGH,
        evaluation_window_minutes=240  # 4 hours
    )
    rules_engine.add_rule(satisfaction_rule)
    
    logger.info("Business impact alert rules configured")

# Factory Functions
def create_email_notifier(config: NotificationConfig) -> EmailNotifier:
    """Create configured email notifier."""
    return EmailNotifier(
        smtp_server=config.smtp_server,
        smtp_port=config.smtp_port,
        username=config.smtp_username, 
        password=config.smtp_password,
        from_email=config.email_from
    )

def create_slack_notifier(config: NotificationConfig) -> SlackNotifier:
    """Create configured Slack notifier."""
    return SlackNotifier(
        webhook_url=config.slack_webhook_url,
        token=config.slack_token,
        default_channel=config.slack_channel
    )

def create_escalation_policy(config: EscalationConfig) -> EscalationPolicy:
    """Create default escalation policy."""
    
    # Create escalation levels
    levels = [
        EscalationLevel(
            level=1,
            name="Team Lead",
            delay_minutes=15,
            channels=[AlertChannel.SLACK],
            recipients=["team-lead@batterymind.com"]
        ),
        EscalationLevel(
            level=2, 
            name="Engineering Manager",
            delay_minutes=30,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            recipients=["eng-manager@batterymind.com"]
        ),
        EscalationLevel(
            level=3,
            name="On-Call Engineer", 
            delay_minutes=45,
            channels=[AlertChannel.EMAIL, AlertChannel.SMS, AlertChannel.SLACK],
            recipients=["oncall@batterymind.com"]
        ),
        EscalationLevel(
            level=4,
            name="Director of Engineering",
            delay_minutes=60,
            channels=[AlertChannel.EMAIL, AlertChannel.SMS],
            recipients=["director@batterymind.com"]
        )
    ]
    
    return EscalationPolicy(
        name="BatteryMind Default Escalation",
        description="Default escalation policy for BatteryMind alerts",
        levels=levels,
        business_hours_only=False,
        max_escalations=4
    )

def create_alert_rules_engine(config: RulesConfig) -> AlertRulesEngine:
    """Create configured alert rules engine."""
    return AlertRulesEngine(
        evaluation_interval_seconds=config.evaluation_interval_seconds,
        enable_rule_chaining=config.enable_rule_chaining,
        max_rule_complexity=config.max_rule_complexity
    )

# Module initialization
def initialize_alert_system(config_path: Optional[str] = None) -> AlertManager:
    """
    Initialize the complete BatteryMind alert system.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Configured AlertManager instance
    """
    try:
        # Load configuration if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Create configuration objects from loaded data
            system_config = AlertSystemConfig(**config_data.get('system', {}))
            notification_config = NotificationConfig(**config_data.get('notifications', {}))
            escalation_config = EscalationConfig(**config_data.get('escalation', {}))
            rules_config = RulesConfig(**config_data.get('rules', {}))
        else:
            # Use default configurations
            system_config = AlertSystemConfig()
            notification_config = NotificationConfig()
            escalation_config = EscalationConfig()
            rules_config = RulesConfig()
        
        # Create alert system
        alert_manager = create_alert_system(system_config)
        
        # Set up battery-specific rules
        setup_battery_alerts(alert_manager)
        setup_model_performance_alerts(alert_manager)
        setup_business_alerts(alert_manager)
        
        logger.info(f"BatteryMind alert system initialized successfully")
        logger.info(f"System: {system_config.system_name} ({system_config.environment})")
        logger.info(f"Alert retention: {system_config.alert_retention_days} days")
        logger.info(f"Max alerts per minute: {system_config.max_alerts_per_minute}")
        
        return alert_manager
        
    except Exception as e:
        logger.error(f"Failed to initialize alert system: {e}")
        raise

# Version and system information
def get_alert_system_info() -> Dict[str, Any]:
    """Get information about the alert system."""
    return {
        'version': __version__,
        'author': __author__,
        'components': {
            'alert_manager': 'Core alert management and routing',
            'notification_service': 'Multi-channel notification delivery',
            'escalation_manager': 'Alert escalation and on-call management',
            'rules_engine': 'Configurable alert rules and conditions'
        },
        'supported_channels': [
            'email', 'slack', 'sms', 'webhook', 'pagerduty'
        ],
        'supported_conditions': [
            'greater_than', 'less_than', 'equals', 'not_equals',
            'contains', 'anomaly', 'threshold_breach'
        ],
        'alert_severities': [
            'low', 'medium', 'high', 'critical'
        ]
    }

# Default configurations for different environments
DEVELOPMENT_CONFIG = AlertSystemConfig(
    environment="development",
    max_alerts_per_minute=20,
    alert_retention_days=7,
    enable_deduplication=False
)

STAGING_CONFIG = AlertSystemConfig(
    environment="staging", 
    max_alerts_per_minute=50,
    alert_retention_days=30,
    enable_deduplication=True
)

PRODUCTION_CONFIG = AlertSystemConfig(
    environment="production",
    max_alerts_per_minute=100,
    alert_retention_days=90,
    enable_deduplication=True,
    enable_smart_routing=True,
    enable_ml_correlation=True
)

# Export environment configs
__all__.extend([
    "initialize_alert_system",
    "get_alert_system_info", 
    "DEVELOPMENT_CONFIG",
    "STAGING_CONFIG",
    "PRODUCTION_CONFIG"
])

# Log module initialization
logger.info(f"BatteryMind Alerts Module v{__version__} loaded")
logger.info("Available components: AlertManager, NotificationService, EscalationManager, AlertRulesEngine")
