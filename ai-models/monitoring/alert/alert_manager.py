"""
BatteryMind - Alert Manager
Centralized alert management system for battery management AI models with intelligent
correlation, deduplication, escalation, and multi-channel notification capabilities.

Features:
- Intelligent alert correlation and grouping
- Priority-based alert routing and escalation
- Alert lifecycle management (creation, acknowledgment, resolution)
- Integration with notification services and external systems
- Alert suppression and snoozing capabilities
- SLA tracking and performance monitoring
- Configurable alert rules and thresholds
- Historical alert analysis and reporting

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import threading
import time
import json
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
from collections import defaultdict, deque
import asyncio
import sqlite3
from pathlib import Path

# BatteryMind imports
from .notification_service import NotificationService, NotificationChannel, NotificationResult
from .escalation_policy import EscalationPolicy, EscalationManager
from .alert_rules import AlertRulesEngine, AlertCondition
from ...utils.logging_utils import setup_logger
from ...config.monitoring_config import MonitoringConfig

# Configure logging
logger = setup_logger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels with numeric priorities."""
    LOW = (1, "low", "#90EE90")
    MEDIUM = (2, "medium", "#FFD700") 
    HIGH = (3, "high", "#FF8C00")
    CRITICAL = (4, "critical", "#FF0000")
    
    def __init__(self, priority: int, name: str, color: str):
        self.priority = priority
        self.display_name = name
        self.color = color

class AlertStatus(Enum):
    """Alert lifecycle status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged" 
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"
    ESCALATED = "escalated"

class AlertChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    TEAMS = "teams"

@dataclass
class Alert:
    """Core alert data structure."""
    
    # Core identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fingerprint: str = ""  # For deduplication
    
    # Alert metadata  
    title: str = ""
    description: str = ""
    severity: AlertSeverity = AlertSeverity.MEDIUM
    status: AlertStatus = AlertStatus.ACTIVE
    
    # Source information
    source: str = ""  # Component that generated alert
    source_type: str = ""  # battery, ai_model, blockchain, etc.
    metric_name: str = ""
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    
    # Time tracking
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Context and metadata
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Tracking information
    owner: Optional[str] = None
    assignee: Optional[str] = None
    escalation_level: int = 0
    notification_count: int = 0
    
    # State management
    suppressed_until: Optional[datetime] = None
    snooze_count: int = 0
    correlation_id: Optional[str] = None
    parent_alert_id: Optional[str] = None
    
    def generate_fingerprint(self) -> str:
        """Generate unique fingerprint for deduplication."""
        components = [
            self.source,
            self.source_type, 
            self.metric_name,
            str(self.labels.get('instance', 'default')),
            str(self.labels.get('job', 'default'))
        ]
        
        fingerprint_data = "|".join(components)
        self.fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
        return self.fingerprint
    
    def is_active(self) -> bool:
        """Check if alert is in active state."""
        if self.status != AlertStatus.ACTIVE:
            return False
        
        if self.expires_at and datetime.now() > self.expires_at:
            return False
            
        if self.suppressed_until and datetime.now() < self.suppressed_until:
            return False
            
        return True
    
    def get_age_seconds(self) -> int:
        """Get alert age in seconds."""
        return int((datetime.now() - self.created_at).total_seconds())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary format."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
            elif isinstance(value, (AlertSeverity, AlertStatus)):
                data[key] = value.value
        return data

@dataclass
class AlertConfiguration:
    """Alert manager configuration."""
    
    # Core settings
    deduplication_enabled: bool = True
    deduplication_window_minutes: int = 5
    max_alerts_per_minute: int = 100
    alert_retention_days: int = 90
    
    # Database settings
    database_path: str = "./alerts.db"
    enable_persistence: bool = True
    
    # Processing settings
    correlation_enabled: bool = True
    auto_resolution_enabled: bool = True
    auto_resolution_timeout_hours: int = 24
    
    # Notification settings
    default_channels: List[AlertChannel] = field(default_factory=lambda: [
        AlertChannel.EMAIL, AlertChannel.SLACK
    ])
    notification_retry_count: int = 3
    notification_timeout_seconds: int = 30
    
    # Escalation settings
    enable_escalation: bool = True
    escalation_timeout_minutes: int = 30
    max_escalation_levels: int = 3
    
    # Performance settings
    worker_thread_count: int = 4
    batch_processing_size: int = 50
    processing_interval_seconds: int = 10

class AlertManager:
    """
    Centralized alert management system with intelligent correlation,
    deduplication, and multi-channel notifications.
    """
    
    def __init__(self, 
                 config: AlertConfiguration,
                 notification_service: Optional[NotificationService] = None,
                 escalation_manager: Optional[EscalationManager] = None,
                 rules_engine: Optional[AlertRulesEngine] = None):
        
        self.config = config
        self.notification_service = notification_service or NotificationService()
        self.escalation_manager = escalation_manager or EscalationManager()
        self.rules_engine = rules_engine or AlertRulesEngine()
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}  # alert_id -> Alert
        self.alert_fingerprints: Dict[str, str] = {}  # fingerprint -> alert_id
        self.alert_correlations: Dict[str, List[str]] = {}  # correlation_id -> [alert_ids]
        
        # Processing queues
        self.processing_queue: deque = deque()
        self.notification_queue: deque = deque()
        self.escalation_queue: deque = deque()
        
        # Statistics and metrics
        self.stats = {
            'total_alerts': 0,
            'active_alerts': 0,
            'resolved_alerts': 0,
            'suppressed_alerts': 0,
            'escalated_alerts': 0,
            'notifications_sent': 0,
            'processing_errors': 0
        }
        
        # Threading and state
        self.is_running = False
        self.worker_threads = []
        self.shutdown_event = threading.Event()
        self.processing_lock = threading.RLock()
        
        # Rate limiting
        self.alert_rate_limiter = deque(maxlen=self.config.max_alerts_per_minute)
        
        # Database connection
        self.db_connection = None
        if self.config.enable_persistence:
            self._initialize_database()
        
        # Callback handlers
        self.alert_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        logger.info("Alert Manager initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database for alert persistence."""
        try:
            db_path = Path(self.config.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.db_connection = sqlite3.connect(
                str(db_path), 
                check_same_thread=False,
                timeout=30.0
            )
            
            # Create tables
            cursor = self.db_connection.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    fingerprint TEXT,
                    title TEXT,
                    description TEXT,
                    severity TEXT,
                    status TEXT,
                    source TEXT,
                    source_type TEXT,
                    metric_name TEXT,
                    current_value REAL,
                    threshold_value REAL,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    acknowledged_at TIMESTAMP,
                    resolved_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    labels TEXT,
                    annotations TEXT,
                    context TEXT,
                    owner TEXT,
                    assignee TEXT,
                    escalation_level INTEGER,
                    notification_count INTEGER,
                    correlation_id TEXT,
                    parent_alert_id TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_alerts_fingerprint ON alerts(fingerprint)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)
            ''')
            
            self.db_connection.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.config.enable_persistence = False
    
    def start(self):
        """Start the alert manager and worker threads."""
        if self.is_running:
            logger.warning("Alert manager is already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start worker threads
        for i in range(self.config.worker_thread_count):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"AlertManager-Worker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
        
        # Start processing thread
        processing_thread = threading.Thread(
            target=self._processing_loop,
            name="AlertManager-Processor",
            daemon=True
        )
        processing_thread.start()
        self.worker_threads.append(processing_thread)
        
        logger.info(f"Alert manager started with {self.config.worker_thread_count} workers")
    
    def stop(self, timeout: int = 30):
        """Stop the alert manager gracefully."""
        if not self.is_running:
            return
        
        logger.info("Shutting down alert manager...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Wait for threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=timeout)
        
        # Close database connection
        if self.db_connection:
            self.db_connection.close()
        
        logger.info("Alert manager stopped")
    
    def create_alert(self, 
                    title: str,
                    description: str = "",
                    severity: AlertSeverity = AlertSeverity.MEDIUM,
                    source: str = "",
                    source_type: str = "",
                    metric_name: str = "",
                    current_value: Optional[float] = None,
                    threshold_value: Optional[float] = None,
                    labels: Optional[Dict[str, str]] = None,
                    annotations: Optional[Dict[str, str]] = None,
                    context: Optional[Dict[str, Any]] = None,
                    expires_at: Optional[datetime] = None) -> Alert:
        """
        Create a new alert.
        
        Args:
            title: Alert title
            description: Detailed description
            severity: Alert severity level
            source: Source component
            source_type: Type of source (battery, ai_model, etc.)
            metric_name: Name of the metric that triggered the alert
            current_value: Current value of the metric
            threshold_value: Threshold that was breached
            labels: Key-value labels for grouping and filtering
            annotations: Additional annotations
            context: Additional context data
            expires_at: Optional expiration time
            
        Returns:
            Created Alert object
        """
        # Check rate limiting
        current_time = time.time()
        self.alert_rate_limiter.append(current_time)
        
        # Count alerts in the last minute
        minute_ago = current_time - 60
        recent_alerts = sum(1 for t in self.alert_rate_limiter if t > minute_ago)
        
        if recent_alerts > self.config.max_alerts_per_minute:
            logger.warning(f"Alert rate limit exceeded: {recent_alerts}/min")
            # Could implement more sophisticated rate limiting here
        
        # Create alert
        alert = Alert(
            title=title,
            description=description,
            severity=severity,
            source=source,
            source_type=source_type,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            labels=labels or {},
            annotations=annotations or {},
            context=context or {},
            expires_at=expires_at
        )
        
        # Generate fingerprint for deduplication
        alert.generate_fingerprint()
        
        # Check for deduplication
        if self.config.deduplication_enabled:
            existing_alert_id = self.alert_fingerprints.get(alert.fingerprint)
            if existing_alert_id and existing_alert_id in self.active_alerts:
                existing_alert = self.active_alerts[existing_alert_id]
                
                # Update existing alert
                existing_alert.updated_at = datetime.now()
                existing_alert.current_value = current_value
                existing_alert.notification_count += 1
                
                # Check if severity increased
                if severity.priority > existing_alert.severity.priority:
                    existing_alert.severity = severity
                    logger.info(f"Alert severity escalated: {existing_alert.id}")
                
                self._persist_alert(existing_alert)
                logger.debug(f"Deduplicated alert: {alert.fingerprint}")
                return existing_alert
        
        # Add to processing queue
        with self.processing_lock:
            self.active_alerts[alert.id] = alert
            self.alert_fingerprints[alert.fingerprint] = alert.id
            self.processing_queue.append(('create', alert))
        
        # Update statistics
        self.stats['total_alerts'] += 1
        self.stats['active_alerts'] += 1
        
        logger.info(f"Alert created: {alert.id} - {alert.title}")
        
        # Trigger alert handlers
        self._trigger_handlers('alert_created', alert)
        
        return alert
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            user: User who acknowledged the alert
            
        Returns:
            Success status
        """
        with self.processing_lock:
            if alert_id not in self.active_alerts:
                logger.warning(f"Alert not found for acknowledgment: {alert_id}")
                return False
            
            alert = self.active_alerts[alert_id]
            
            if alert.status != AlertStatus.ACTIVE:
                logger.warning(f"Alert {alert_id} is not in active state")
                return False
            
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.owner = user
            alert.updated_at = datetime.now()
            
            self._persist_alert(alert)
            
            logger.info(f"Alert acknowledged: {alert_id} by {user}")
            
            # Trigger handlers
            self._trigger_handlers('alert_acknowledged', alert)
            
            return True
    
    def resolve_alert(self, alert_id: str, user: str = "system", 
                     resolution_note: str = "") -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: ID of the alert to resolve
            user: User who resolved the alert
            resolution_note: Optional resolution note
            
        Returns:
            Success status
        """
        with self.processing_lock:
            if alert_id not in self.active_alerts:
                logger.warning(f"Alert not found for resolution: {alert_id}")
                return False
            
            alert = self.active_alerts[alert_id]
            
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            alert.owner = user
            alert.updated_at = datetime.now()
            
            if resolution_note:
                alert.annotations['resolution_note'] = resolution_note
            
            # Remove from active tracking
            if alert.fingerprint in self.alert_fingerprints:
                del self.alert_fingerprints[alert.fingerprint]
            
            self._persist_alert(alert)
            
            # Update statistics
            self.stats['active_alerts'] -= 1
            self.stats['resolved_alerts'] += 1
            
            logger.info(f"Alert resolved: {alert_id} by {user}")
            
            # Trigger handlers
            self._trigger_handlers('alert_resolved', alert)
            
            # Remove from active alerts after persistence
            del self.active_alerts[alert_id]
            
            return True
    
    def suppress_alert(self, alert_id: str, duration_minutes: int, 
                      user: str = "system") -> bool:
        """
        Suppress an alert for a specified duration.
        
        Args:
            alert_id: ID of the alert to suppress
            duration_minutes: Duration to suppress in minutes
            user: User who suppressed the alert
            
        Returns:
            Success status
        """
        with self.processing_lock:
            if alert_id not in self.active_alerts:
                logger.warning(f"Alert not found for suppression: {alert_id}")
                return False
            
            alert = self.active_alerts[alert_id]
            
            alert.status = AlertStatus.SUPPRESSED
            alert.suppressed_until = datetime.now() + timedelta(minutes=duration_minutes)
            alert.owner = user
            alert.updated_at = datetime.now()
            
            self._persist_alert(alert)
            
            # Update statistics
            self.stats['suppressed_alerts'] += 1
            
            logger.info(f"Alert suppressed: {alert_id} for {duration_minutes} minutes by {user}")
            
            # Trigger handlers
            self._trigger_handlers('alert_suppressed', alert)
            
            return True
    
    def snooze_alert(self, alert_id: str, duration_minutes: int) -> bool:
        """
        Snooze an alert (temporary suppression).
        
        Args:
            alert_id: ID of the alert to snooze
            duration_minutes: Duration to snooze in minutes
            
        Returns:
            Success status
        """
        with self.processing_lock:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.suppressed_until = datetime.now() + timedelta(minutes=duration_minutes)
            alert.snooze_count += 1
            alert.updated_at = datetime.now()
            
            self._persist_alert(alert)
            
            logger.info(f"Alert snoozed: {alert_id} for {duration_minutes} minutes")
            return True
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        return self.active_alerts.get(alert_id)
    
    def list_alerts(self, 
                   status: Optional[AlertStatus] = None,
                   severity: Optional[AlertSeverity] = None,
                   source_type: Optional[str] = None,
                   limit: int = 100,
                   offset: int = 0) -> List[Alert]:
        """
        List alerts with optional filtering.
        
        Args:
            status: Filter by status
            severity: Filter by severity
            source_type: Filter by source type
            limit: Maximum number of alerts to return
            offset: Offset for pagination
            
        Returns:            List of matching alerts
        """
        alerts = list(self.active_alerts.values())
        
        # Apply filters
        if status:
            alerts = [a for a in alerts if a.status == status]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if source_type:
            alerts = [a for a in alerts if a.source_type == source_type]
        
        # Sort by creation time (newest first)
        alerts.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        return alerts[offset:offset + limit]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics and metrics."""
        stats = self.stats.copy()
        
        # Add current counts
        stats['current_active'] = len([
            a for a in self.active_alerts.values() 
            if a.status == AlertStatus.ACTIVE
        ])
        
        stats['current_acknowledged'] = len([
            a for a in self.active_alerts.values()
            if a.status == AlertStatus.ACKNOWLEDGED
        ])
        
        # Severity breakdown
        severity_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            if alert.is_active():
                severity_counts[alert.severity.value] += 1
        
        stats['severity_breakdown'] = dict(severity_counts)
        
        # Source type breakdown  
        source_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            if alert.is_active():
                source_counts[alert.source_type] += 1
        
        stats['source_type_breakdown'] = dict(source_counts)
        
        return stats
    
    def _worker_loop(self):
        """Main worker loop for processing alerts."""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Process alerts from queue
                if self.processing_queue:
                    with self.processing_lock:
                        if self.processing_queue:
                            action, alert = self.processing_queue.popleft()
                            self._process_alert_action(action, alert)
                
                # Process notifications
                if self.notification_queue:
                    with self.processing_lock:
                        if self.notification_queue:
                            notification_task = self.notification_queue.popleft()
                            self._process_notification(notification_task)
                
                # Sleep briefly to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                self.stats['processing_errors'] += 1
                time.sleep(1)
    
    def _processing_loop(self):
        """Main processing loop for periodic tasks."""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Auto-resolve expired alerts
                if self.config.auto_resolution_enabled:
                    self._auto_resolve_expired_alerts()
                
                # Process escalations
                if self.config.enable_escalation:
                    self._process_escalations()
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                # Update statistics
                self._update_statistics()
                
                # Sleep for processing interval
                self.shutdown_event.wait(self.config.processing_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(10)
    
    def _process_alert_action(self, action: str, alert: Alert):
        """Process alert actions (create, update, etc.)."""
        try:
            if action == 'create':
                # Apply alert rules
                if self.rules_engine:
                    rule_result = self.rules_engine.evaluate_alert(alert)
                    if rule_result:
                        alert.labels.update(rule_result.get('labels', {}))
                        alert.annotations.update(rule_result.get('annotations', {}))
                
                # Correlate with existing alerts
                if self.config.correlation_enabled:
                    self._correlate_alert(alert)
                
                # Schedule notifications
                self._schedule_notifications(alert)
                
                # Persist alert
                self._persist_alert(alert)
                
            elif action == 'update':
                self._persist_alert(alert)
                
        except Exception as e:
            logger.error(f"Error processing alert action {action}: {e}")
    
    def _correlate_alert(self, alert: Alert):
        """Correlate alert with existing alerts."""
        try:
            # Simple correlation based on source and metric
            correlation_key = f"{alert.source}:{alert.metric_name}"
            
            # Find related alerts
            related_alerts = []
            for existing_alert in self.active_alerts.values():
                if (existing_alert.source == alert.source and 
                    existing_alert.metric_name == alert.metric_name and
                    existing_alert.id != alert.id):
                    related_alerts.append(existing_alert.id)
            
            if related_alerts:
                correlation_id = f"corr_{uuid.uuid4().hex[:8]}"
                alert.correlation_id = correlation_id
                self.alert_correlations[correlation_id] = related_alerts + [alert.id]
                
                logger.debug(f"Alert correlated: {alert.id} with {len(related_alerts)} others")
                
        except Exception as e:
            logger.error(f"Error correlating alert: {e}")
    
    def _schedule_notifications(self, alert: Alert):
        """Schedule notifications for an alert."""
        try:
            # Determine notification channels based on severity
            channels = self._get_notification_channels(alert)
            
            for channel in channels:
                notification_task = {
                    'alert': alert,
                    'channel': channel,
                    'attempt': 1,
                    'scheduled_at': datetime.now()
                }
                
                with self.processing_lock:
                    self.notification_queue.append(notification_task)
                
        except Exception as e:
            logger.error(f"Error scheduling notifications: {e}")
    
    def _get_notification_channels(self, alert: Alert) -> List[AlertChannel]:
        """Get appropriate notification channels for alert severity."""
        if alert.severity == AlertSeverity.CRITICAL:
            return [AlertChannel.EMAIL, AlertChannel.SMS, AlertChannel.SLACK, AlertChannel.PAGERDUTY]
        elif alert.severity == AlertSeverity.HIGH:
            return [AlertChannel.EMAIL, AlertChannel.SLACK]
        elif alert.severity == AlertSeverity.MEDIUM:
            return [AlertChannel.SLACK]
        else:
            return [AlertChannel.SLACK]
    
    def _process_notification(self, notification_task: Dict[str, Any]):
        """Process a notification task."""
        try:
            alert = notification_task['alert']
            channel = notification_task['channel']
            attempt = notification_task['attempt']
            
            # Send notification
            result = self.notification_service.send_notification(
                channel=NotificationChannel(channel.value),
                title=alert.title,
                message=alert.description,
                severity=alert.severity.value,
                metadata={
                    'alert_id': alert.id,
                    'source': alert.source,
                    'metric_name': alert.metric_name,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value
                }
            )
            
            if result.success:
                alert.notification_count += 1
                self.stats['notifications_sent'] += 1
                logger.debug(f"Notification sent: {alert.id} via {channel.value}")
            else:
                logger.warning(f"Notification failed: {alert.id} via {channel.value} - {result.error}")
                
                # Retry if within limits
                if attempt < self.config.notification_retry_count:
                    retry_task = notification_task.copy()
                    retry_task['attempt'] = attempt + 1
                    retry_task['scheduled_at'] = datetime.now() + timedelta(minutes=attempt * 2)
                    
                    with self.processing_lock:
                        self.notification_queue.append(retry_task)
                
        except Exception as e:
            logger.error(f"Error processing notification: {e}")
    
    def _auto_resolve_expired_alerts(self):
        """Automatically resolve expired alerts."""
        try:
            current_time = datetime.now()
            expired_alerts = []
            
            for alert in self.active_alerts.values():
                # Check explicit expiration
                if alert.expires_at and current_time > alert.expires_at:
                    expired_alerts.append(alert.id)
                    continue
                
                # Check auto-resolution timeout
                age_hours = (current_time - alert.created_at).total_seconds() / 3600
                if age_hours > self.config.auto_resolution_timeout_hours:
                    expired_alerts.append(alert.id)
            
            # Resolve expired alerts
            for alert_id in expired_alerts:
                self.resolve_alert(alert_id, user="auto-resolver", 
                                 resolution_note="Expired or timeout")
                
            if expired_alerts:
                logger.info(f"Auto-resolved {len(expired_alerts)} expired alerts")
                
        except Exception as e:
            logger.error(f"Error auto-resolving expired alerts: {e}")
    
    def _process_escalations(self):
        """Process alert escalations."""
        try:
            current_time = datetime.now()
            
            for alert in self.active_alerts.values():
                if (alert.status == AlertStatus.ACTIVE and 
                    alert.escalation_level < self.config.max_escalation_levels):
                    
                    # Check if escalation is due
                    age_minutes = (current_time - alert.created_at).total_seconds() / 60
                    escalation_threshold = (alert.escalation_level + 1) * self.config.escalation_timeout_minutes
                    
                    if age_minutes > escalation_threshold:
                        self._escalate_alert(alert)
                        
        except Exception as e:
            logger.error(f"Error processing escalations: {e}")
    
    def _escalate_alert(self, alert: Alert):
        """Escalate an alert to the next level."""
        try:
            alert.escalation_level += 1
            alert.status = AlertStatus.ESCALATED
            alert.updated_at = datetime.now()
            
            # Use escalation manager if available
            if self.escalation_manager:
                escalation_result = self.escalation_manager.escalate_alert(alert)
                if escalation_result:
                    alert.assignee = escalation_result.get('assignee')
            
            self.stats['escalated_alerts'] += 1
            
            logger.info(f"Alert escalated: {alert.id} to level {alert.escalation_level}")
            
            # Trigger handlers
            self._trigger_handlers('alert_escalated', alert)
            
            # Schedule additional notifications
            self._schedule_notifications(alert)
            
        except Exception as e:
            logger.error(f"Error escalating alert: {e}")
    
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts from memory."""
        try:
            cutoff_date = datetime.now() - timedelta(days=1)  # Keep resolved alerts for 1 day
            
            alerts_to_remove = []
            for alert_id, alert in self.active_alerts.items():
                if (alert.status == AlertStatus.RESOLVED and 
                    alert.resolved_at and alert.resolved_at < cutoff_date):
                    alerts_to_remove.append(alert_id)
            
            for alert_id in alerts_to_remove:
                del self.active_alerts[alert_id]
                
            if alerts_to_remove:
                logger.debug(f"Cleaned up {len(alerts_to_remove)} old alerts")
                
        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")
    
    def _update_statistics(self):
        """Update internal statistics."""
        try:
            active_count = len([
                a for a in self.active_alerts.values() 
                if a.status == AlertStatus.ACTIVE
            ])
            
            self.stats['active_alerts'] = active_count
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    def _persist_alert(self, alert: Alert):
        """Persist alert to database."""
        if not self.config.enable_persistence or not self.db_connection:
            return
        
        try:
            cursor = self.db_connection.cursor()
            
            # Convert complex fields to JSON
            labels_json = json.dumps(alert.labels)
            annotations_json = json.dumps(alert.annotations)
            context_json = json.dumps(alert.context)
            
            cursor.execute('''
                INSERT OR REPLACE INTO alerts (
                    id, fingerprint, title, description, severity, status,
                    source, source_type, metric_name, current_value, threshold_value,
                    created_at, updated_at, acknowledged_at, resolved_at, expires_at,
                    labels, annotations, context, owner, assignee,
                    escalation_level, notification_count, correlation_id, parent_alert_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id, alert.fingerprint, alert.title, alert.description,
                alert.severity.value, alert.status.value,
                alert.source, alert.source_type, alert.metric_name,
                alert.current_value, alert.threshold_value,
                alert.created_at, alert.updated_at, alert.acknowledged_at,
                alert.resolved_at, alert.expires_at,
                labels_json, annotations_json, context_json,
                alert.owner, alert.assignee, alert.escalation_level,
                alert.notification_count, alert.correlation_id, alert.parent_alert_id
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error persisting alert: {e}")
    
    def _trigger_handlers(self, event: str, alert: Alert):
        """Trigger registered event handlers."""
        try:
            handlers = self.alert_handlers.get(event, [])
            for handler in handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler {handler.__name__}: {e}")
                    
        except Exception as e:
            logger.error(f"Error triggering handlers: {e}")
    
    def register_handler(self, event: str, handler: Callable[[Alert], None]):
        """
        Register an event handler.
        
        Args:
            event: Event name (alert_created, alert_acknowledged, etc.)
            handler: Handler function that takes an Alert parameter
        """
        self.alert_handlers[event].append(handler)
        logger.info(f"Registered handler for event: {event}")
    
    def unregister_handler(self, event: str, handler: Callable[[Alert], None]):
        """Unregister an event handler."""
        if event in self.alert_handlers and handler in self.alert_handlers[event]:
            self.alert_handlers[event].remove(handler)
    
    def export_alerts(self, format: str = "json", 
                     status: Optional[AlertStatus] = None,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> str:
        """
        Export alerts in specified format.
        
        Args:
            format: Export format (json, csv)
            status: Filter by status
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Exported data as string
        """
        try:
            alerts = list(self.active_alerts.values())
            
            # Apply filters
            if status:
                alerts = [a for a in alerts if a.status == status]
            
            if start_date:
                alerts = [a for a in alerts if a.created_at >= start_date]
            
            if end_date:
                alerts = [a for a in alerts if a.created_at <= end_date]
            
            if format.lower() == "json":
                return json.dumps([alert.to_dict() for alert in alerts], indent=2)
            elif format.lower() == "csv":
                # Implement CSV export
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow([
                    'id', 'title', 'severity', 'status', 'source', 'source_type',
                    'created_at', 'resolved_at', 'owner'
                ])
                
                # Write data
                for alert in alerts:
                    writer.writerow([
                        alert.id, alert.title, alert.severity.value, alert.status.value,
                        alert.source, alert.source_type, alert.created_at,
                        alert.resolved_at, alert.owner
                    ])
                
                return output.getvalue()
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting alerts: {e}")
            return ""
