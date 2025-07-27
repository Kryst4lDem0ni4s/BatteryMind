"""
BatteryMind - Escalation Policy Management
Advanced escalation policy management system for battery management AI models
with intelligent routing, on-call scheduling, and automated escalation workflows.

Features:
- Multi-level escalation policies with time-based triggers
- On-call schedule management with rotation and availability
- Intelligent escalation routing based on alert context
- SLA tracking and escalation performance monitoring
- Integration with external paging systems
- Automated escalation decision making
- Escalation analytics and optimization

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import threading
import time
import json
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dt_time
from enum import Enum
import uuid
from collections import defaultdict, deque
import pytz
from croniter import croniter

# BatteryMind imports
from .notification_service import NotificationService, NotificationChannel, NotificationResult
from ...utils.logging_utils import setup_logger
from ...config.monitoring_config import MonitoringConfig

# Configure logging
logger = setup_logger(__name__)

class EscalationAction(Enum):
    """Types of escalation actions."""
    NOTIFY = "notify"
    ASSIGN = "assign"
    ESCALATE = "escalate"
    ACKNOWLEDGE = "acknowledge"
    RESOLVE = "resolve"
    SNOOZE = "snooze"
    PAGE = "page"
    CALL = "call"

class EscalationTrigger(Enum):
    """Escalation trigger conditions."""
    TIME_BASED = "time_based"
    NO_RESPONSE = "no_response"
    SEVERITY_INCREASE = "severity_increase"
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    THRESHOLD_BREACH = "threshold_breach"
    PATTERN_MATCH = "pattern_match"

class ScheduleType(Enum):
    """On-call schedule types."""
    ROUND_ROBIN = "round_robin"
    FIXED = "fixed"
    WEIGHTED = "weighted"
    SKILL_BASED = "skill_based"
    LOAD_BALANCED = "load_balanced"
    TIME_ZONE_AWARE = "time_zone_aware"

@dataclass
class EscalationLevel:
    """Individual escalation level configuration."""
    
    level: int
    name: str
    delay_minutes: int
    channels: List[NotificationChannel] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)
    groups: List[str] = field(default_factory=list)
    
    # Advanced settings
    max_attempts: int = 3
    retry_interval_minutes: int = 5
    timeout_minutes: int = 30
    require_acknowledgment: bool = True
    
    # Business hours handling
    business_hours_only: bool = False
    after_hours_escalation: bool = True
    weekend_escalation: bool = True
    
    # Custom conditions
    conditions: Dict[str, Any] = field(default_factory=dict)
    custom_logic: Optional[str] = None
    
    def is_active_time(self, current_time: datetime, timezone: str = "UTC") -> bool:
        """Check if current time is within active hours for this level."""
        if not self.business_hours_only:
            return True
        
        tz = pytz.timezone(timezone)
        local_time = current_time.astimezone(tz)
        
        # Check business hours (assuming 9 AM to 5 PM)
        is_business_hours = (
            9 <= local_time.hour < 17 and
            local_time.weekday() < 5  # Monday=0, Sunday=6
        )
        
        if is_business_hours:
            return True
        
        # Check after hours and weekend settings
        if not is_business_hours and self.after_hours_escalation:
            return True
        
        if local_time.weekday() >= 5 and self.weekend_escalation:
            return True
        
        return False

@dataclass
class OnCallPerson:
    """On-call person configuration."""
    
    id: str
    name: str
    email: str
    phone: str
    timezone: str = "UTC"
    
    # Availability settings
    preferred_channels: List[NotificationChannel] = field(default_factory=list)
    escalation_delay_minutes: int = 10
    max_alerts_per_hour: int = 20
    
    # Skills and specializations
    skills: List[str] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    experience_level: str = "intermediate"  # junior, intermediate, senior, expert
    
    # Availability schedule
    available_hours: Dict[str, Dict[str, str]] = field(default_factory=dict)
    blackout_periods: List[Dict[str, datetime]] = field(default_factory=list)
    
    # Performance metrics
    response_time_minutes: float = 15.0
    resolution_rate: float = 0.85
    satisfaction_score: float = 4.2
    
    def is_available(self, current_time: datetime) -> bool:
        """Check if person is available at current time."""
        # Check blackout periods
        for period in self.blackout_periods:
            if period['start'] <= current_time <= period['end']:
                return False
        
        # Check availability hours
        tz = pytz.timezone(self.timezone)
        local_time = current_time.astimezone(tz)
        weekday = local_time.strftime('%A').lower()
        
        if weekday in self.available_hours:
            hours = self.available_hours[weekday]
            start_time = dt_time.fromisoformat(hours['start'])
            end_time = dt_time.fromisoformat(hours['end'])
            current_time_only = local_time.time()
            
            return start_time <= current_time_only <= end_time
        
        return True  # Available by default if no schedule specified

@dataclass
class OnCallSchedule:
    """On-call schedule management."""
    
    id: str
    name: str
    schedule_type: ScheduleType
    people: List[OnCallPerson] = field(default_factory=list)
    
    # Schedule settings
    rotation_weeks: int = 1
    handover_time: str = "09:00"  # HH:MM format
    timezone: str = "UTC"
    
    # Advanced scheduling
    skill_requirements: List[str] = field(default_factory=list)
    minimum_experience: str = "intermediate"
    coverage_requirements: Dict[str, int] = field(default_factory=dict)
    
    # Schedule state
    current_primary: Optional[str] = None
    current_secondary: Optional[str] = None
    last_rotation: Optional[datetime] = None
    
    def get_current_on_call(self, current_time: datetime) -> List[OnCallPerson]:
        """Get currently on-call people based on schedule."""
        available_people = [
            person for person in self.people
            if person.is_available(current_time)
        ]
        
        if not available_people:
            return []
        
        if self.schedule_type == ScheduleType.ROUND_ROBIN:
            return self._round_robin_schedule(available_people, current_time)
        elif self.schedule_type == ScheduleType.SKILL_BASED:
            return self._skill_based_schedule(available_people, current_time)
        elif self.schedule_type == ScheduleType.LOAD_BALANCED:
            return self._load_balanced_schedule(available_people, current_time)
        else:
            return available_people[:2]  # Primary and secondary
    
    def _round_robin_schedule(self, people: List[OnCallPerson], current_time: datetime) -> List[OnCallPerson]:
        """Implement round-robin scheduling."""
        if not self.last_rotation:
            self.last_rotation = current_time
            return people[:2]
        
        # Check if rotation is due
        weeks_since_rotation = (current_time - self.last_rotation).days // 7
        if weeks_since_rotation >= self.rotation_weeks:
            # Rotate schedule
            self.people = self.people[1:] + [self.people[0]]
            self.last_rotation = current_time
        
        return people[:2]
    
    def _skill_based_schedule(self, people: List[OnCallPerson], current_time: datetime) -> List[OnCallPerson]:
        """Implement skill-based scheduling."""
        # Filter by required skills
        skilled_people = []
        for person in people:
            if all(skill in person.skills for skill in self.skill_requirements):
                skilled_people.append(person)
        
        # Sort by experience level and performance
        experience_order = {"expert": 4, "senior": 3, "intermediate": 2, "junior": 1}
        skilled_people.sort(
            key=lambda p: (
                experience_order.get(p.experience_level, 0),
                p.resolution_rate,
                -p.response_time_minutes
            ),
            reverse=True
        )
        
        return skilled_people[:2]
    
    def _load_balanced_schedule(self, people: List[OnCallPerson], current_time: datetime) -> List[OnCallPerson]:
        """Implement load-balanced scheduling based on recent activity."""
        # This would typically use historical data to balance load
        # For now, sort by response time and satisfaction
        people.sort(
            key=lambda p: (p.response_time_minutes, -p.satisfaction_score)
        )
        return people[:2]

@dataclass
class EscalationRule:
    """Escalation rule definition."""
    
    id: str
    name: str
    description: str
    trigger: EscalationTrigger
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Rule logic
    severity_threshold: Optional[str] = None
    time_threshold_minutes: Optional[int] = None
    pattern_match: Optional[str] = None
    custom_logic: Optional[str] = None
    
    # Actions
    action: EscalationAction = EscalationAction.ESCALATE
    target_level: Optional[int] = None
    notification_template: Optional[str] = None
    
    # Rule state
    enabled: bool = True
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    
    def evaluate(self, alert_context: Dict[str, Any]) -> bool:
        """Evaluate if rule conditions are met."""
        try:
            # Check severity threshold
            if self.severity_threshold:
                alert_severity = alert_context.get('severity', 'LOW')
                severity_order = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
                threshold_order = severity_order.get(self.severity_threshold, 0)
                current_order = severity_order.get(alert_severity, 0)
                
                if current_order < threshold_order:
                    return False
            
            # Check time threshold
            if self.time_threshold_minutes:
                created_at = alert_context.get('created_at')
                if created_at:
                    age_minutes = (datetime.now() - created_at).total_seconds() / 60
                    if age_minutes < self.time_threshold_minutes:
                        return False
            
            # Check pattern match
            if self.pattern_match:
                alert_title = alert_context.get('title', '')
                alert_description = alert_context.get('description', '')
                text_to_match = f"{alert_title} {alert_description}".lower()
                
                if self.pattern_match.lower() not in text_to_match:
                    return False
            
            # Check custom conditions
            for key, expected_value in self.conditions.items():
                actual_value = alert_context.get(key)
                if actual_value != expected_value:
                    return False
            
            # Execute custom logic if provided
            if self.custom_logic:
                # This would typically be executed in a safe sandbox
                # For now, we'll use simple eval (not recommended for production)
                try:
                    context = {'alert': alert_context, 'datetime': datetime}
                    result = eval(self.custom_logic, {"__builtins__": {}}, context)
                    return bool(result)
                except Exception as e:
                    logger.error(f"Error evaluating custom logic: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating escalation rule {self.id}: {e}")
            return False

@dataclass
class EscalationPolicy:
    """Complete escalation policy definition."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Escalation configuration
    levels: List[EscalationLevel] = field(default_factory=list)
    rules: List[EscalationRule] = field(default_factory=list)
    schedules: List[OnCallSchedule] = field(default_factory=list)
    
    # Policy settings
    max_escalations: int = 3
    auto_resolve_after_hours: int = 24
    suppress_during_maintenance: bool = True
    business_hours_only: bool = False
    
    # Advanced features
    intelligent_routing: bool = True
    context_aware_escalation: bool = True
    ml_optimization: bool = False
    
    # Policy state
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def get_escalation_path(self, alert_context: Dict[str, Any]) -> List[EscalationLevel]:
        """Get escalation path based on alert context and rules."""
        applicable_levels = []
        
        for level in self.levels:
            # Check if level is applicable based on context
            if self._is_level_applicable(level, alert_context):
                applicable_levels.append(level)
        
        # Sort by level number
        applicable_levels.sort(key=lambda x: x.level)
        
        return applicable_levels[:self.max_escalations]
    
    def _is_level_applicable(self, level: EscalationLevel, alert_context: Dict[str, Any]) -> bool:
        """Check if escalation level is applicable for the alert."""
        # Check time constraints
        if not level.is_active_time(datetime.now()):
            return False
        
        # Check custom conditions
        for key, expected_value in level.conditions.items():
            actual_value = alert_context.get(key)
            if actual_value != expected_value:
                return False
        
        return True

@dataclass
class EscalationConfig:
    """Configuration for escalation management."""
    
    # Default settings
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
    
    # Performance settings
    response_timeout_minutes: int = 15
    escalation_timeout_minutes: int = 30
    max_retries: int = 3
    
    # Integration settings
    pagerduty_enabled: bool = False
    opsgenie_enabled: bool = False
    victorops_enabled: bool = False

class EscalationManager:
    """
    Advanced escalation policy management system.
    """
    
    def __init__(self, config: Optional[EscalationConfig] = None,
                 notification_service: Optional[NotificationService] = None):
        
        self.config = config or EscalationConfig()
        self.notification_service = notification_service or NotificationService()
        
        # Policy storage
        self.policies: Dict[str, EscalationPolicy] = {}
        self.active_escalations: Dict[str, Dict[str, Any]] = {}
        
        # Statistics and metrics
        self.escalation_stats = {
            'total_escalations': 0,
            'successful_escalations': 0,
            'failed_escalations': 0,
            'average_response_time': 0.0,
            'resolution_rate': 0.0
        }
        
        # Threading and state
        self.is_running = False
        self.escalation_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Callback handlers
        self.escalation_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        logger.info("Escalation Manager initialized")
    
    def add_policy(self, policy: EscalationPolicy) -> bool:
        """Add an escalation policy."""
        try:
            self.policies[policy.id] = policy
            logger.info(f"Added escalation policy: {policy.name}")
            return True
        except Exception as e:
            logger.error(f"Error adding escalation policy: {e}")
            return False
    
    def remove_policy(self, policy_id: str) -> bool:
        """Remove an escalation policy."""
        try:
            if policy_id in self.policies:
                del self.policies[policy_id]
                logger.info(f"Removed escalation policy: {policy_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing escalation policy: {e}")
            return False
    
    def start(self):
        """Start the escalation manager."""
        if self.is_running:
            logger.warning("Escalation manager is already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start escalation processing thread
        self.escalation_thread = threading.Thread(
            target=self._escalation_loop,
            name="EscalationManager",
            daemon=True
        )
        self.escalation_thread.start()
        
        logger.info("Escalation manager started")
    
    def stop(self, timeout: int = 30):
        """Stop the escalation manager."""
        if not self.is_running:
            return
        
        logger.info("Stopping escalation manager...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        if self.escalation_thread and self.escalation_thread.is_alive():
            self.escalation_thread.join(timeout=timeout)
        
        logger.info("Escalation manager stopped")
    
    def escalate_alert(self, alert: Any, policy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Escalate an alert using the specified policy.
        
        Args:
            alert: Alert object to escalate
            policy_id: ID of escalation policy to use
            
        Returns:
            Escalation result with status and details
        """
        try:
            # Get escalation policy
            if policy_id and policy_id in self.policies:
                policy = self.policies[policy_id]
            else:
                # Use default policy or first available
                if self.policies:
                    policy = next(iter(self.policies.values()))
                else:
                    return {
                        'success': False,
                        'error': 'No escalation policy available',
                        'escalation_id': None
                    }
            
            # Create alert context
            alert_context = {
                'id': getattr(alert, 'id', 'unknown'),
                'title': getattr(alert, 'title', ''),
                'description': getattr(alert, 'description', ''),
                'severity': getattr(alert, 'severity', 'MEDIUM'),
                'source': getattr(alert, 'source', ''),
                'created_at': getattr(alert, 'created_at', datetime.now()),
                'status': getattr(alert, 'status', 'ACTIVE')
            }
            
            # Get escalation path
            escalation_path = policy.get_escalation_path(alert_context)
            
            if not escalation_path:
                return {
                    'success': False,
                    'error': 'No applicable escalation levels found',
                    'escalation_id': None
                }
            
            # Create escalation record
            escalation_id = str(uuid.uuid4())
            escalation_record = {
                'id': escalation_id,
                'alert_id': alert_context['id'],
                'policy_id': policy.id,
                'current_level': 0,
                'escalation_path': escalation_path,
                'started_at': datetime.now(),
                'status': 'ACTIVE',
                'attempts': 0,
                'responses': []
            }
            
            self.active_escalations[escalation_id] = escalation_record
            
            # Start escalation process
            self._process_escalation(escalation_record, alert_context)
            
            # Update statistics
            self.escalation_stats['total_escalations'] += 1
            
            # Trigger escalation handlers
            self._trigger_handlers('escalation_started', escalation_record)
            
            logger.info(f"Started escalation for alert {alert_context['id']}")
            
            return {
                'success': True,
                'escalation_id': escalation_id,
                'policy_name': policy.name,
                'levels_count': len(escalation_path)
            }
            
        except Exception as e:
            logger.error(f"Error escalating alert: {e}")
            return {
                'success': False,
                'error': str(e),
                'escalation_id': None
            }
    
    def acknowledge_escalation(self, escalation_id: str, user: str = "system") -> bool:
        """Acknowledge an escalation."""
        try:
            if escalation_id not in self.active_escalations:
                return False
            
            escalation = self.active_escalations[escalation_id]
            escalation['status'] = 'ACKNOWLEDGED'
            escalation['acknowledged_by'] = user
            escalation['acknowledged_at'] = datetime.now()
            
            # Trigger handlers
            self._trigger_handlers('escalation_acknowledged', escalation)
            
            logger.info(f"Escalation {escalation_id} acknowledged by {user}")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging escalation: {e}")
            return False
    
    def resolve_escalation(self, escalation_id: str, user: str = "system") -> bool:
        """Resolve an escalation."""
        try:
            if escalation_id not in self.active_escalations:
                return False
            
            escalation = self.active_escalations[escalation_id]
            escalation['status'] = 'RESOLVED'
            escalation['resolved_by'] = user
            escalation['resolved_at'] = datetime.now()
            
            # Update statistics
            self.escalation_stats['successful_escalations'] += 1
            
            # Trigger handlers
            self._trigger_handlers('escalation_resolved', escalation)
            
            logger.info(f"Escalation {escalation_id} resolved by {user}")
            
            # Remove from active escalations
            del self.active_escalations[escalation_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Error resolving escalation: {e}")
            return False
    
    def _escalation_loop(self):
        """Main escalation processing loop."""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()
                
                # Process active escalations
                escalations_to_process = list(self.active_escalations.items())
                
                for escalation_id, escalation in escalations_to_process:
                    if escalation['status'] != 'ACTIVE':
                        continue
                    
                    # Check if escalation should proceed to next level
                    if self._should_escalate_to_next_level(escalation, current_time):
                        self._escalate_to_next_level(escalation_id, escalation)
                
                # Sleep for processing interval
                self.shutdown_event.wait(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in escalation loop: {e}")
                time.sleep(60)
    
    def _process_escalation(self, escalation: Dict[str, Any], alert_context: Dict[str, Any]):
        """Process the first level of escalation."""
        try:
            if not escalation['escalation_path']:
                return
            
            first_level = escalation['escalation_path'][0]
            self._notify_escalation_level(escalation, first_level, alert_context)
            
        except Exception as e:
            logger.error(f"Error processing escalation: {e}")
    
    def _should_escalate_to_next_level(self, escalation: Dict[str, Any], current_time: datetime) -> bool:
        """Check if escalation should proceed to next level."""
        try:
            current_level_index = escalation['current_level']
            escalation_path = escalation['escalation_path']
            
            if current_level_index >= len(escalation_path):
                return False
            
            current_level = escalation_path[current_level_index]
            level_start_time = escalation.get('level_start_time', escalation['started_at'])
            
            # Check if delay time has passed
            delay_seconds = current_level.delay_minutes * 60
            time_elapsed = (current_time - level_start_time).total_seconds()
            
            if time_elapsed >= delay_seconds:
                # Check if maximum attempts reached
                if escalation['attempts'] >= current_level.max_attempts:
                    return True
                
                # Check if timeout reached
                timeout_seconds = current_level.timeout_minutes * 60
                if time_elapsed >= timeout_seconds:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking escalation condition: {e}")
            return False
    
    def _escalate_to_next_level(self, escalation_id: str, escalation: Dict[str, Any]):
        """Escalate to the next level."""
        try:
            escalation['current_level'] += 1
            escalation['level_start_time'] = datetime.now()
            escalation['attempts'] = 0
            
            current_level_index = escalation['current_level']
            escalation_path = escalation['escalation_path']
            
            if current_level_index >= len(escalation_path):
                # Escalation completed
                escalation['status'] = 'COMPLETED'
                self.escalation_stats['failed_escalations'] += 1
                logger.warning(f"Escalation {escalation_id} completed without resolution")
                return
            
            # Get next level and process
            next_level = escalation_path[current_level_index]
            
            # Create alert context (this would typically come from stored data)
            alert_context = {
                'id': escalation['alert_id'],
                'title': f"Escalated Alert (Level {current_level_index + 1})",
                'description': f"Alert escalated to level {current_level_index + 1}",
                'severity': 'HIGH',  # Escalated alerts typically have higher severity
                'escalation_level': current_level_index + 1
            }
            
            self._notify_escalation_level(escalation, next_level, alert_context)
            
            # Trigger handlers
            self._trigger_handlers('escalation_level_changed', escalation)
            
            logger.info(f"Escalated {escalation_id} to level {current_level_index + 1}")
            
        except Exception as e:
            logger.error(f"Error escalating to next level: {e}")
    
    def _notify_escalation_level(self, escalation: Dict[str, Any], 
                                level: EscalationLevel, 
                                alert_context: Dict[str, Any]):
        """Send notifications for an escalation level."""
        try:
            escalation['attempts'] += 1
            
            # Prepare notification content
            title = f"[ESCALATION] {alert_context.get('title', 'Alert')}"
            message = self._build_escalation_message(escalation, level, alert_context)
            
            # Send notifications to all recipients
            for recipient in level.recipients:
                for channel in level.channels:
                    try:
                        result = self.notification_service.send_notification_sync(
                            channel=channel,
                            recipient=recipient,
                            title=title,
                            message=message,
                            severity="high",
                            metadata={
                                'escalation_id': escalation['id'],
                                'escalation_level': level.level,
                                'alert_id': escalation['alert_id']
                            }
                        )
                        
                        # Record response
                        escalation['responses'].append({
                            'timestamp': datetime.now(),
                            'recipient': recipient,
                            'channel': channel.value,
                            'success': result.success,
                            'message_id': result.message_id,
                            'error': result.error
                        })
                        
                    except Exception as e:
                        logger.error(f"Error sending escalation notification: {e}")
            
        except Exception as e:
            logger.error(f"Error notifying escalation level: {e}")
    
    def _build_escalation_message(self, escalation: Dict[str, Any], 
                                 level: EscalationLevel, 
                                 alert_context: Dict[str, Any]) -> str:
        """Build escalation notification message."""
        message = f"""
Escalation Alert - Level {level.level}

Alert Details:
- Alert ID: {alert_context.get('id', 'N/A')}
- Title: {alert_context.get('title', 'N/A')}
- Severity: {alert_context.get('severity', 'N/A')}
- Source: {alert_context.get('source', 'N/A')}

Escalation Details:
- Escalation ID: {escalation['id']}
- Level: {level.level} ({level.name})
- Started: {escalation['started_at'].strftime('%Y-%m-%d %H:%M:%S')}
- Attempts: {escalation['attempts']}/{level.max_attempts}

Description:
{alert_context.get('description', 'No description available')}

Please acknowledge this escalation and take appropriate action.
        """.strip()
        
        return message
    
    def _trigger_handlers(self, event: str, escalation: Dict[str, Any]):
        """Trigger registered event handlers."""
        try:
            handlers = self.escalation_handlers.get(event, [])
            for handler in handlers:
                try:
                    handler(escalation)
                except Exception as e:
                    logger.error(f"Error in escalation handler {handler.__name__}: {e}")
        except Exception as e:
            logger.error(f"Error triggering handlers: {e}")
    
    def register_handler(self, event: str, handler: Callable):
        """Register an event handler."""
        self.escalation_handlers[event].append(handler)
        logger.info(f"Registered escalation handler for event: {event}")
    
    def get_escalation_statistics(self) -> Dict[str, Any]:
        """Get escalation statistics."""
        stats = self.escalation_stats.copy()
        
        # Add current state information
        stats['active_escalations'] = len(self.active_escalations)
        stats['total_policies'] = len(self.policies)
        
        # Calculate derived metrics
        if stats['total_escalations'] > 0:
            stats['success_rate'] = stats['successful_escalations'] / stats['total_escalations']
            stats['failure_rate'] = stats['failed_escalations'] / stats['total_escalations']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        return stats
    
    def get_active_escalations(self) -> List[Dict[str, Any]]:
        """Get list of active escalations."""
        return list(self.active_escalations.values())
    
    def get_policy(self, policy_id: str) -> Optional[EscalationPolicy]:
        """Get escalation policy by ID."""
        return self.policies.get(policy_id)
    
    def list_policies(self) -> List[EscalationPolicy]:
        """Get list of all escalation policies."""
        return list(self.policies.values())

# Factory functions
def create_default_escalation_policy() -> EscalationPolicy:
    """Create a default escalation policy for BatteryMind."""
    
    # Create escalation levels
    levels = [
        EscalationLevel(
            level=1,
            name="Team Lead",
            delay_minutes=15,
            channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL],
            recipients=["team-lead@batterymind.com"],
            max_attempts=2,
            timeout_minutes=30
        ),
        EscalationLevel(
            level=2,
            name="Engineering Manager",
            delay_minutes=30,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SMS],
            recipients=["eng-manager@batterymind.com"],
            max_attempts=2,
            timeout_minutes=45
        ),
        EscalationLevel(
            level=3,
            name="On-Call Engineer",
            delay_minutes=45,
            channels=[NotificationChannel.PAGERDUTY, NotificationChannel.SMS],
            recipients=["oncall@batterymind.com"],
            max_attempts=3,
            timeout_minutes=60
        )
    ]
    
    # Create escalation rules
    rules = [
        EscalationRule(
            id=str(uuid.uuid4()),
            name="Critical Battery Alert",
            description="Escalate critical battery alerts immediately",
            trigger=EscalationTrigger.SEVERITY_INCREASE,
            severity_threshold="CRITICAL",
            action=EscalationAction.ESCALATE,
            target_level=1
        ),
        EscalationRule(
            id=str(uuid.uuid4()),
            name="No Response Escalation",
            description="Escalate if no response within 30 minutes",
            trigger=EscalationTrigger.NO_RESPONSE,
            time_threshold_minutes=30,
            action=EscalationAction.ESCALATE
        )
    ]
    
    return EscalationPolicy(
        name="BatteryMind Default Escalation",
        description="Default escalation policy for BatteryMind alerts",
        levels=levels,
        rules=rules,
        max_escalations=3,
        auto_resolve_after_hours=24
    )

def create_on_call_schedule() -> OnCallSchedule:
    """Create a default on-call schedule."""
    
    # Create on-call people
    people = [
        OnCallPerson(
            id="john_doe",
            name="John Doe",
            email="john.doe@batterymind.com",
            phone="+1-555-0101",
            timezone="America/New_York",
            skills=["battery_systems", "ai_models", "debugging"],
            experience_level="senior"
        ),
        OnCallPerson(
            id="jane_smith",
            name="Jane Smith",
            email="jane.smith@batterymind.com",
            phone="+1-555-0102",
            timezone="America/Los_Angeles",
            skills=["blockchain", "smart_contracts", "security"],
            experience_level="expert"
        )
    ]
    
    return OnCallSchedule(
        id="default_schedule",
        name="BatteryMind On-Call",
        schedule_type=ScheduleType.ROUND_ROBIN,
        people=people,
        rotation_weeks=1,
        timezone="UTC"
    )
