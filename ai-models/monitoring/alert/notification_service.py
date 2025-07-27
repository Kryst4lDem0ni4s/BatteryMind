"""
BatteryMind - Notification Service
Multi-channel notification service for alert delivery with template support,
rate limiting, delivery tracking, and failure handling.

Features:
- Multi-channel notifications (Email, SMS, Slack, Webhook, Teams, PagerDuty)
- Template-based message formatting with customization
- Rate limiting and throttling per channel
- Delivery tracking and confirmation
- Retry mechanisms with exponential backoff
- Notification batching and aggregation
- Escalation integration
- Analytics and reporting

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import smtplib
import json
import requests
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import ssl
import threading
from collections import deque, defaultdict
import asyncio
import aiohttp
from jinja2 import Template, Environment, FileSystemLoader
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

# BatteryMind imports
from ...utils.logging_utils import setup_logger
from ...utils.config_parser import ConfigParser
from ...config.monitoring_config import MonitoringConfig

# Configure logging
logger = setup_logger(__name__)

class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"
    DISCORD = "discord"
    TELEGRAM = "telegram"

class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class NotificationTemplate:
    """Template for notification formatting."""
    
    name: str
    channel: NotificationChannel
    subject_template: str = ""
    body_template: str = ""
    format_type: str = "text"  # text, html, markdown
    variables: Dict[str, Any] = field(default_factory=dict)
    
    def render(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Render template with context variables."""
        try:
            env = Environment()
            
            # Render subject
            subject = ""
            if self.subject_template:
                subject_tmpl = env.from_string(self.subject_template)
                subject = subject_tmpl.render(**context, **self.variables)
            
            # Render body
            body = ""
            if self.body_template:
                body_tmpl = env.from_string(self.body_template)
                body = body_tmpl.render(**context, **self.variables)
            
            return {
                "subject": subject,
                "body": body,
                "format": self.format_type
            }
            
        except Exception as e:
            logger.error(f"Error rendering template {self.name}: {e}")
            return {"subject": "Alert", "body": str(context), "format": "text"}

@dataclass
class NotificationResult:
    """Result of a notification attempt."""
    
    success: bool = False
    message_id: Optional[str] = None
    error: Optional[str] = None
    channel: Optional[NotificationChannel] = None
    recipient: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    response_time_ms: float = 0.0
    retry_count: int = 0

@dataclass
class NotificationConfig:
    """Configuration for notification service."""
    
    # Rate limiting (per hour)
    email_rate_limit: int = 100
    sms_rate_limit: int = 20
    slack_rate_limit: int = 500
    webhook_rate_limit: int = 1000
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: int = 60
    exponential_backoff: bool = True
    
    # Timeout configuration
    request_timeout_seconds: int = 30
    email_timeout_seconds: int = 60
    
    # Template configuration
    template_directory: str = "./templates"
    default_templates: bool = True
    
    # Batching configuration
    enable_batching: bool = True
    batch_size: int = 10
    batch_timeout_seconds: int = 300

class BaseNotifier:
    """Base class for notification channels."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rate_limiter = deque(maxlen=1000)
        self.delivery_stats = {
            'sent': 0,
            'failed': 0,
            'rate_limited': 0
        }
    
    def check_rate_limit(self, limit_per_hour: int) -> bool:
        """Check if rate limit is exceeded."""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        # Remove old entries
        while self.rate_limiter and self.rate_limiter[0] < hour_ago:
            self.rate_limiter.popleft()
        
        return len(self.rate_limiter) < limit_per_hour
    
    def record_attempt(self):
        """Record a notification attempt."""
        self.rate_limiter.append(time.time())
    
    async def send(self, recipient: str, subject: str, body: str, 
                  priority: NotificationPriority = NotificationPriority.NORMAL,
                  metadata: Optional[Dict[str, Any]] = None) -> NotificationResult:
        """Send notification (to be implemented by subclasses)."""
        raise NotImplementedError

class EmailNotifier(BaseNotifier):
    """Email notification handler using SMTP."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username', '')
        self.password = config.get('password', '')
        self.from_email = config.get('from_email', 'alerts@batterymind.com')
        self.use_tls = config.get('use_tls', True)
    
    async def send(self, recipient: str, subject: str, body: str,
                  priority: NotificationPriority = NotificationPriority.NORMAL,
                  metadata: Optional[Dict[str, Any]] = None) -> NotificationResult:
        """Send email notification."""
        start_time = time.time()
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = recipient
            
            # Set priority headers
            if priority == NotificationPriority.URGENT:
                msg['X-Priority'] = '1'
                msg['Importance'] = 'high'
            elif priority == NotificationPriority.HIGH:
                msg['X-Priority'] = '2'
                msg['Importance'] = 'high'
            
            # Add body (support both text and HTML)
            if metadata and metadata.get('format') == 'html':
                html_part = MIMEText(body, 'html')
                msg.attach(html_part)
            else:
                text_part = MIMEText(body, 'plain')
                msg.attach(text_part)
            
            # Send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls(context=context)
                
                if self.username and self.password:
                    server.login(self.username, self.password)
                
                text = msg.as_string()
                server.sendmail(self.from_email, recipient, text)
            
            response_time = (time.time() - start_time) * 1000
            self.delivery_stats['sent'] += 1
            
            return NotificationResult(
                success=True,
                channel=NotificationChannel.EMAIL,
                recipient=recipient,
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.delivery_stats['failed'] += 1
            
            return NotificationResult(
                success=False,
                error=str(e),
                channel=NotificationChannel.EMAIL,
                recipient=recipient,
                response_time_ms=response_time
            )

class SlackNotifier(BaseNotifier):
    """Slack notification handler using webhooks."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get('webhook_url', '')
        self.token = config.get('token', '')
        self.default_channel = config.get('default_channel', '#alerts')
        self.username = config.get('username', 'BatteryMind')
        self.icon_emoji = config.get('icon_emoji', ':battery:')
    
    async def send(self, recipient: str, subject: str, body: str,
                  priority: NotificationPriority = NotificationPriority.NORMAL,
                  metadata: Optional[Dict[str, Any]] = None) -> NotificationResult:
        """Send Slack notification."""
        start_time = time.time()
        
        try:
            # Build Slack message payload
            payload = {
                "channel": recipient if recipient.startswith('#') else self.default_channel,
                "username": self.username,
                "icon_emoji": self.icon_emoji,
                "attachments": [{
                    "color": self._get_color_for_priority(priority),
                    "title": subject,
                    "text": body,
                    "fields": [],
                    "footer": "BatteryMind Alert System",
                    "ts": int(time.time())
                }]
            }
            
            # Add metadata fields
            if metadata:
                for key, value in metadata.items():
                    if key not in ['format']:
                        payload["attachments"][0]["fields"].append({
                            "title": key.replace('_', ' ').title(),
                            "value": str(value),
                            "short": True
                        })
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        response_time = (time.time() - start_time) * 1000
                        self.delivery_stats['sent'] += 1
                        
                        return NotificationResult(
                            success=True,
                            channel=NotificationChannel.SLACK,
                            recipient=recipient,
                            response_time_ms=response_time
                        )
                    else:
                        error_text = await response.text()
                        raise Exception(f"Slack API error: {response.status} - {error_text}")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.delivery_stats['failed'] += 1
            
            return NotificationResult(
                success=False,
                error=str(e),
                channel=NotificationChannel.SLACK,
                recipient=recipient,
                response_time_ms=response_time
            )
    
    def _get_color_for_priority(self, priority: NotificationPriority) -> str:
        """Get color code for priority level."""
        color_map = {
            NotificationPriority.LOW: "#36a64f",      # Green
            NotificationPriority.NORMAL: "#ffcc00",   # Yellow
            NotificationPriority.HIGH: "#ff8c00",     # Orange
            NotificationPriority.URGENT: "#ff0000"    # Red
        }
        return color_map.get(priority, "#ffcc00")

class SMSNotifier(BaseNotifier):
    """SMS notification handler using AWS SNS."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.aws_region = config.get('aws_region', 'us-west-2')
        self.access_key = config.get('aws_access_key')
        self.secret_key = config.get('aws_secret_key')
        
        # Initialize SNS client
        session = boto3.Session(
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.aws_region
        )
        self.sns_client = session.client('sns')
    
    async def send(self, recipient: str, subject: str, body: str,
                  priority: NotificationPriority = NotificationPriority.NORMAL,
                  metadata: Optional[Dict[str, Any]] = None) -> NotificationResult:
        """Send SMS notification."""
        start_time = time.time()
        
        try:
            # Format message (SMS has character limits)
            message = f"{subject}\n\n{body}"
            if len(message) > 160:
                message = f"{subject}\n\n{body[:140]}..."
            
            # Send SMS via SNS
            response = self.sns_client.publish(
                PhoneNumber=recipient,
                Message=message,
                MessageAttributes={
                    'AWS.SNS.SMS.SenderID': {
                        'DataType': 'String',
                        'StringValue': 'BatteryMind'
                    }
                }
            )
            
            response_time = (time.time() - start_time) * 1000
            self.delivery_stats['sent'] += 1
            
            return NotificationResult(
                success=True,
                message_id=response.get('MessageId'),
                channel=NotificationChannel.SMS,
                recipient=recipient,
                response_time_ms=response_time
            )
            
        except ClientError as e:
            response_time = (time.time() - start_time) * 1000
            self.delivery_stats['failed'] += 1
            
            return NotificationResult(
                success=False,
                error=str(e),
                channel=NotificationChannel.SMS,
                recipient=recipient,
                response_time_ms=response_time
            )

class WebhookNotifier(BaseNotifier):
    """Generic webhook notification handler."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_urls = config.get('webhook_urls', [])
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.authentication = config.get('authentication', {})
    
    async def send(self, recipient: str, subject: str, body: str,
                  priority: NotificationPriority = NotificationPriority.NORMAL,
                  metadata: Optional[Dict[str, Any]] = None) -> NotificationResult:
        """Send webhook notification."""
        start_time = time.time()
        
        try:
            # Build webhook payload
            payload = {
                "timestamp": datetime.now().isoformat(),
                "priority": priority.name,
                "subject": subject,
                "body": body,
                "recipient": recipient,
                "metadata": metadata or {}
            }
            
            # Add authentication if configured
            headers = self.headers.copy()
            if self.authentication.get('type') == 'bearer':
                headers['Authorization'] = f"Bearer {self.authentication.get('token')}"
            elif self.authentication.get('type') == 'api_key':
                headers[self.authentication.get('header', 'X-API-Key')] = self.authentication.get('key')
            
            # Send to webhook endpoint
            webhook_url = recipient if recipient.startswith('http') else self.webhook_urls[0]
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if 200 <= response.status < 300:
                        response_time = (time.time() - start_time) * 1000
                        self.delivery_stats['sent'] += 1
                        
                        return NotificationResult(
                            success=True,
                            channel=NotificationChannel.WEBHOOK,
                            recipient=recipient,
                            response_time_ms=response_time
                        )
                    else:
                        error_text = await response.text()
                        raise Exception(f"Webhook error: {response.status} - {error_text}")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.delivery_stats['failed'] += 1
            
            return NotificationResult(
                success=False,
                error=str(e),
                channel=NotificationChannel.WEBHOOK,
                recipient=recipient,
                response_time_ms=response_time
            )

class TeamsNotifier(BaseNotifier):
    """Microsoft Teams notification handler."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get('webhook_url', '')
    
    async def send(self, recipient: str, subject: str, body: str,
                  priority: NotificationPriority = NotificationPriority.NORMAL,
                  metadata: Optional[Dict[str, Any]] = None) -> NotificationResult:
        """Send Teams notification."""
        start_time = time.time()
        
        try:
            # Build Teams message card
            payload = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": subject,
                "themeColor": self._get_color_for_priority(priority),
                "sections": [{
                    "activityTitle": subject,
                    "activitySubtitle": "BatteryMind Alert",
                    "activityImage": "https://batterymind.com/icon.png",
                    "text": body,
                    "facts": []
                }]
            }
            
            # Add metadata as facts
            if metadata:
                for key, value in metadata.items():
                    if key not in ['format']:
                        payload["sections"][0]["facts"].append({
                            "name": key.replace('_', ' ').title(),
                            "value": str(value)
                        })
            
            # Send to Teams
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        response_time = (time.time() - start_time) * 1000
                        self.delivery_stats['sent'] += 1
                        
                        return NotificationResult(
                            success=True,
                            channel=NotificationChannel.TEAMS,
                            recipient=recipient,
                            response_time_ms=response_time
                        )
                    else:
                        error_text = await response.text()
                        raise Exception(f"Teams API error: {response.status} - {error_text}")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.delivery_stats['failed'] += 1
            
            return NotificationResult(
                success=False,
                error=str(e),
                channel=NotificationChannel.TEAMS,
                recipient=recipient,
                response_time_ms=response_time
            )
    
    def _get_color_for_priority(self, priority: NotificationPriority) -> str:
        """Get color code for priority level."""
        color_map = {
            NotificationPriority.LOW: "00FF00",      # Green
            NotificationPriority.NORMAL: "FFFF00",   # Yellow
            NotificationPriority.HIGH: "FF8C00",     # Orange
            NotificationPriority.URGENT: "FF0000"    # Red
        }
        return color_map.get(priority, "FFFF00")

class PagerDutyNotifier(BaseNotifier):
    """PagerDuty notification handler."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.integration_key = config.get('integration_key', '')
        self.api_url = "https://events.pagerduty.com/v2/enqueue"
    
    async def send(self, recipient: str, subject: str, body: str,
                  priority: NotificationPriority = NotificationPriority.NORMAL,
                  metadata: Optional[Dict[str, Any]] = None) -> NotificationResult:
        """Send PagerDuty alert."""
        start_time = time.time()
        
        try:
            # Build PagerDuty event payload
            payload = {
                "routing_key": self.integration_key,
                "event_action": "trigger",
                "dedup_key": metadata.get('alert_id', subject) if metadata else subject,
                "payload": {
                    "summary": subject,
                    "source": "BatteryMind",
                    "severity": self._get_severity_for_priority(priority),
                    "component": metadata.get('source', 'BatteryMind') if metadata else 'BatteryMind',
                    "group": metadata.get('source_type', 'alert') if metadata else 'alert',
                    "class": "battery_management",
                    "custom_details": {
                        "description": body,
                        **(metadata or {})
                    }
                }
            }
            
            # Send to PagerDuty
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 202:
                        response_data = await response.json()
                        response_time = (time.time() - start_time) * 1000
                        self.delivery_stats['sent'] += 1
                        
                        return NotificationResult(
                            success=True,
                            message_id=response_data.get('dedup_key'),
                            channel=NotificationChannel.PAGERDUTY,
                            recipient=recipient,
                            response_time_ms=response_time
                        )
                    else:
                        error_text = await response.text()
                        raise Exception(f"PagerDuty API error: {response.status} - {error_text}")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.delivery_stats['failed'] += 1
            
            return NotificationResult(
                success=False,
                error=str(e),
                channel=NotificationChannel.PAGERDUTY,
                recipient=recipient,
                response_time_ms=response_time
            )
    
    def _get_severity_for_priority(self, priority: NotificationPriority) -> str:
        """Map priority to PagerDuty severity."""
        severity_map = {
            NotificationPriority.LOW: "info",
            NotificationPriority.NORMAL: "warning",
            NotificationPriority.HIGH: "error",
            NotificationPriority.URGENT: "critical"
        }
        return severity_map.get(priority, "warning")

class NotificationService:
    """
    Comprehensive notification service supporting multiple channels.
    """
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig()
        
        # Initialize notifiers
        self.notifiers: Dict[NotificationChannel, BaseNotifier] = {}
        self.templates: Dict[str, NotificationTemplate] = {}
        
        # Processing queues
        self.notification_queue = deque()
        self.retry_queue = deque()
        
        # Statistics
        self.stats = {
            'total_sent': 0,
            'total_failed': 0,
            'total_retried': 0,
            'channel_stats': defaultdict(lambda: {'sent': 0, 'failed': 0})
        }
        
        # Threading
        self.is_running = False
        self.worker_threads = []
        self.shutdown_event = threading.Event()
        
        # Load default templates
        if self.config.default_templates:
            self._load_default_templates()
        
        logger.info("Notification service initialized")
    
    def configure_channel(self, channel: NotificationChannel, config: Dict[str, Any]):
        """Configure a notification channel."""
        try:
            if channel == NotificationChannel.EMAIL:
                self.notifiers[channel] = EmailNotifier(config)
            elif channel == NotificationChannel.SLACK:
                self.notifiers[channel] = SlackNotifier(config)
            elif channel == NotificationChannel.SMS:
                self.notifiers[channel] = SMSNotifier(config)
            elif channel == NotificationChannel.WEBHOOK:
                self.notifiers[channel] = WebhookNotifier(config)
            elif channel == NotificationChannel.TEAMS:
                self.notifiers[channel] = TeamsNotifier(config)
            elif channel == NotificationChannel.PAGERDUTY:
                self.notifiers[channel] = PagerDutyNotifier(config)
            else:
                logger.warning(f"Unsupported notification channel: {channel}")
                return False
            
            logger.info(f"Configured notification channel: {channel.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure channel {channel.value}: {e}")
            return False
    
    def _load_default_templates(self):
        """Load default notification templates."""
        # Battery Alert Template
        battery_alert_template = NotificationTemplate(
            name="battery_alert",
            channel=NotificationChannel.EMAIL,
            subject_template="🔋 Battery Alert: {{ title }}",
            body_template="""
Battery Alert Details:
===================

Alert: {{ title }}
Severity: {{ severity }}
Source: {{ source }}
Metric: {{ metric_name }}

Description:
{{ description }}

Current Value: {{ current_value }}
Threshold: {{ threshold_value }}

Time: {{ timestamp }}

---
BatteryMind Alert System
            """.strip(),
            format_type="text"
        )
        self.templates["battery_alert"] = battery_alert_template
        
        # Slack Battery Alert Template
        slack_battery_template = NotificationTemplate(
            name="battery_alert_slack",
            channel=NotificationChannel.SLACK,
            subject_template="🔋 {{ title }}",
            body_template="{{ description }}\n\n*Source:* {{ source }}\n*Metric:* {{ metric_name }}\n*Value:* {{ current_value }}\n*Threshold:* {{ threshold_value }}",
            format_type="markdown"
        )
        self.templates["battery_alert_slack"] = slack_battery_template
        
        # AI Model Alert Template
        ai_model_template = NotificationTemplate(
            name="ai_model_alert",
            channel=NotificationChannel.EMAIL,
            subject_template="🤖 AI Model Alert: {{ title }}",
            body_template="""
AI Model Alert:
==============

Alert: {{ title }}
Model: {{ model_name | default('Unknown') }}
Severity: {{ severity }}

Description:
{{ description }}

Accuracy: {{ accuracy | default('N/A') }}
Latency: {{ latency | default('N/A') }}ms
Error Rate: {{ error_rate | default('N/A') }}%

Time: {{ timestamp }}

---
BatteryMind AI Monitoring
            """.strip(),
            format_type="text"
        )
        self.templates["ai_model_alert"] = ai_model_template
    
    def add_template(self, template: NotificationTemplate):
        """Add a custom notification template."""
        self.templates[template.name] = template
        logger.info(f"Added notification template: {template.name}")
    
    def start(self):
        """Start the notification service."""
        if self.is_running:
            logger.warning("Notification service is already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start worker threads
        for i in range(2):  # 2 worker threads
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"NotificationService-Worker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
        
        logger.info("Notification service started")
    
    def stop(self, timeout: int = 30):
        """Stop the notification service."""
        if not self.is_running:
            return
        
        logger.info("Stopping notification service...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Wait for threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=timeout)
        
        logger.info("Notification service stopped")
    
    async def send_notification(self,
                               channel: NotificationChannel,
                               recipient: str,
                               title: str,
                               message: str,
                               severity: str = "medium",
                               priority: NotificationPriority = NotificationPriority.NORMAL,
                               template_name: Optional[str] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> NotificationResult:
        """
        Send a notification through the specified channel.
        
        Args:
            channel: Notification channel to use
            recipient: Recipient address/identifier
            title: Notification title
            message: Notification body/message
            severity: Alert severity level
            priority: Notification priority
            template_name: Optional template to use
            metadata: Additional metadata for the notification
            
        Returns:
            NotificationResult with delivery status
        """
        try:
            # Check if channel is configured
            if channel not in self.notifiers:
                return NotificationResult(
                    success=False,
                    error=f"Channel {channel.value} not configured",
                    channel=channel,
                    recipient=recipient
                )
            
            notifier = self.notifiers[channel]
            
            # Check rate limiting
            rate_limit_map = {
                NotificationChannel.EMAIL: self.config.email_rate_limit,
                NotificationChannel.SMS: self.config.sms_rate_limit,
                NotificationChannel.SLACK: self.config.slack_rate_limit,
                NotificationChannel.WEBHOOK: self.config.webhook_rate_limit
            }
            
            rate_limit = rate_limit_map.get(channel, 100)
            if not notifier.check_rate_limit(rate_limit):
                notifier.delivery_stats['rate_limited'] += 1
                return NotificationResult(
                    success=False,
                    error="Rate limit exceeded",
                    channel=channel,
                    recipient=recipient
                )
            
            # Apply template if specified
            if template_name and template_name in self.templates:
                template = self.templates[template_name]
                context = {
                    'title': title,
                    'description': message,
                    'severity': severity,
                    'timestamp': datetime.now().isoformat(),
                    **(metadata or {})
                }
                
                rendered = template.render(context)
                title = rendered.get('subject', title)
                message = rendered.get('body', message)
                
                if metadata:
                    metadata['format'] = rendered.get('format', 'text')
            
            # Record attempt
            notifier.record_attempt()
            
            # Send notification
            result = await notifier.send(
                recipient=recipient,
                subject=title,
                body=message,
                priority=priority,
                metadata=metadata
            )
            
            # Update statistics
            if result.success:
                self.stats['total_sent'] += 1
                self.stats['channel_stats'][channel.value]['sent'] += 1
            else:
                self.stats['total_failed'] += 1
                self.stats['channel_stats'][channel.value]['failed'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error sending notification via {channel.value}: {e}")
            return NotificationResult(
                success=False,
                error=str(e),
                channel=channel,
                recipient=recipient
            )
    
    def send_notification_sync(self,
                              channel: NotificationChannel,
                              recipient: str,
                              title: str,
                              message: str,
                              severity: str = "medium",
                              priority: NotificationPriority = NotificationPriority.NORMAL,
                              template_name: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> NotificationResult:
        """Synchronous wrapper for send_notification."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.send_notification(
                    channel, recipient, title, message, severity,
                    priority, template_name, metadata
                )
            )
        finally:
            loop.close()
    
    def send_batch_notifications(self,
                                notifications: List[Dict[str, Any]]) -> List[NotificationResult]:
        """Send multiple notifications in batch."""
        results = []
        
        for notification in notifications:
            result = self.send_notification_sync(**notification)
            results.append(result)
        
        return results
    
    def _worker_loop(self):
        """Worker loop for processing queued notifications."""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Process queued notifications
                if self.notification_queue:
                    notification_task = self.notification_queue.popleft()
                    result = self.send_notification_sync(**notification_task)
                    
                    # Handle retries for failed notifications
                    if not result.success and result.retry_count < self.config.max_retries:
                        notification_task['retry_count'] = result.retry_count + 1
                        self.retry_queue.append(notification_task)
                
                # Process retry queue
                if self.retry_queue:
                    retry_task = self.retry_queue.popleft()
                    
                    # Calculate retry delay
                    delay = self.config.retry_delay_seconds
                    if self.config.exponential_backoff:
                        delay *= (2 ** retry_task.get('retry_count', 0))
                    
                    time.sleep(min(delay, 300))  # Max 5 minute delay
                    
                    result = self.send_notification_sync(**retry_task)
                    self.stats['total_retried'] += 1
                    
                    # Retry again if still failing
                    if not result.success and retry_task.get('retry_count', 0) < self.config.max_retries:
                        self.retry_queue.append(retry_task)
                
                # Sleep briefly if no work
                if not self.notification_queue and not self.retry_queue:
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in notification worker loop: {e}")
                time.sleep(5)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification service statistics."""
        return {
            'total_sent': self.stats['total_sent'],
            'total_failed': self.stats['total_failed'],
            'total_retried': self.stats['total_retried'],
            'channel_statistics': dict(self.stats['channel_stats']),
            'configured_channels': [ch.value for ch in self.notifiers.keys()],
            'available_templates': list(self.templates.keys()),
            'queue_sizes': {
                'notification_queue': len(self.notification_queue),
                'retry_queue': len(self.retry_queue)
            }
        }
    
    def test_channel(self, channel: NotificationChannel, 
                    recipient: str) -> NotificationResult:
        """Test a notification channel with a sample message."""
        return self.send_notification_sync(
            channel=channel,
            recipient=recipient,
            title="BatteryMind Test Notification",
            message="This is a test notification from BatteryMind alert system. If you receive this, the channel is working correctly.",
            severity="low",
            priority=NotificationPriority.LOW,
            metadata={'test': True}
        )
