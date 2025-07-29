"""
BatteryMind - Logging Utilities
Advanced logging, monitoring, and audit utilities for the BatteryMind
autonomous battery management system with comprehensive tracking capabilities.

Features:
- Structured logging with contextual information
- Performance monitoring and metrics collection
- Audit trail for compliance and security
- Real-time log streaming and alerting
- Log aggregation and analysis
- Custom log formatters for different components
- Integration with monitoring systems

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import logging.handlers
import json
import time
import threading
import queue
import sys
import os
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import traceback
import uuid
import socket
import psutil
from functools import wraps
import contextvars

# Context variables for request tracking
request_id_context = contextvars.ContextVar('request_id', default=None)
user_id_context = contextvars.ContextVar('user_id', default=None)
session_id_context = contextvars.ContextVar('session_id', default=None)

class LogLevel(Enum):
    """Extended log levels for BatteryMind."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    AUDIT = 35  # Custom level for audit logs
    PERFORMANCE = 25  # Custom level for performance logs
    SECURITY = 45  # Custom level for security logs

class LogCategory(Enum):
    """Log categories for better organization."""
    SYSTEM = "system"
    BATTERY = "battery"
    FLEET = "fleet"
    AI_MODEL = "ai_model"
    BLOCKCHAIN = "blockchain"
    AUTONOMOUS = "autonomous"
    FEDERATED = "federated"
    CIRCULAR_ECONOMY = "circular_economy"
    IOT = "iot"
    SECURITY = "security"
    PERFORMANCE = "performance"
    AUDIT = "audit"
    USER_ACTION = "user_action"
    API = "api"

@dataclass
class LogEntry:
    """Structured log entry for BatteryMind."""
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    level: str = "INFO"
    category: LogCategory = LogCategory.SYSTEM
    message: str = ""
    
    # Context information
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Technical details
    logger_name: str = ""
    module: str = ""
    function: str = ""
    line_number: int = 0
    
    # Additional data
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Error information
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # System information
    hostname: str = field(default_factory=socket.gethostname)
    process_id: int = field(default_factory=os.getpid)
    thread_id: str = field(default_factory=lambda: str(threading.current_thread().ident))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary."""
        data = asdict(self)
        # Convert datetime to ISO string
        data['timestamp'] = self.timestamp.isoformat()
        # Convert enum to string
        data['category'] = self.category.value
        return data
    
    def to_json(self) -> str:
        """Convert log entry to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        try:
            # Extract context information
            request_id = request_id_context.get()
            user_id = user_id_context.get()
            session_id = session_id_context.get()
            
            # Create log entry
            log_entry = LogEntry(
                timestamp=datetime.utcnow(),
                level=record.levelname,
                category=getattr(record, 'category', LogCategory.SYSTEM),
                message=record.getMessage(),
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
                logger_name=record.name,
                module=record.module if hasattr(record, 'module') else '',
                function=record.funcName,
                line_number=record.lineno,
                extra_data=getattr(record, 'extra_data', {}),
                execution_time_ms=getattr(record, 'execution_time_ms', None),
                memory_usage_mb=getattr(record, 'memory_usage_mb', None)
            )
            
            # Add exception information if present
            if record.exc_info:
                log_entry.exception_type = record.exc_info[0].__name__
                log_entry.exception_message = str(record.exc_info[1])
                log_entry.stack_trace = self.formatException(record.exc_info)
            
            return log_entry.to_json()
            
        except Exception as e:
            # Fallback to simple formatting if structured formatting fails
            return f"LOGGING_ERROR: {e} | Original: {record.getMessage()}"

class PerformanceLogger:
    """Logger for performance monitoring."""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
        self.start_times: Dict[str, float] = {}
        self.performance_data: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
    
    def start_timing(self, operation_id: str):
        """Start timing an operation."""
        with self.lock:
            self.start_times[operation_id] = time.time()
    
    def end_timing(self, operation_id: str, 
                   operation_name: str = "", 
                   extra_data: Dict[str, Any] = None):
        """End timing and log performance data."""
        try:
            with self.lock:
                if operation_id not in self.start_times:
                    self.logger.warning(f"No start time found for operation: {operation_id}")
                    return
                
                end_time = time.time()
                execution_time = (end_time - self.start_times[operation_id]) * 1000  # Convert to ms
                
                # Get memory usage
                process = psutil.Process()
                memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
                
                # Create performance record
                perf_data = {
                    'operation_id': operation_id,
                    'operation_name': operation_name or operation_id,
                    'execution_time_ms': execution_time,
                    'memory_usage_mb': memory_usage,
                    'timestamp': datetime.utcnow().isoformat(),
                    'extra_data': extra_data or {}
                }
                
                self.performance_data.append(perf_data)
                
                # Log performance data
                self.logger.info(
                    f"Performance: {operation_name or operation_id}",
                    extra={
                        'category': LogCategory.PERFORMANCE,
                        'execution_time_ms': execution_time,
                        'memory_usage_mb': memory_usage,
                        'extra_data': perf_data
                    }
                )
                
                # Clean up
                del self.start_times[operation_id]
                
        except Exception as e:
            self.logger.error(f"Error in end_timing: {e}")
    
    def get_performance_summary(self, operation_name: str = None) -> Dict[str, Any]:
        """Get performance summary statistics."""
        try:
            with self.lock:
                # Filter data if operation_name specified
                data = self.performance_data
                if operation_name:
                    data = [d for d in data if d['operation_name'] == operation_name]
                
                if not data:
                    return {}
                
                execution_times = [d['execution_time_ms'] for d in data]
                memory_usages = [d['memory_usage_mb'] for d in data]
                
                import numpy as np
                
                summary = {
                    'operation_name': operation_name or 'all_operations',
                    'sample_count': len(data),
                    'execution_time': {
                        'mean_ms': np.mean(execution_times),
                        'median_ms': np.median(execution_times),
                        'min_ms': np.min(execution_times),
                        'max_ms': np.max(execution_times),
                        'std_ms': np.std(execution_times),
                        'p95_ms': np.percentile(execution_times, 95),
                        'p99_ms': np.percentile(execution_times, 99)
                    },
                    'memory_usage': {
                        'mean_mb': np.mean(memory_usages),
                        'median_mb': np.median(memory_usages),
                        'min_mb': np.min(memory_usages),
                        'max_mb': np.max(memory_usages),
                        'std_mb': np.std(memory_usages)
                    }
                }
                
                return summary
                
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}

class AuditLogger:
    """Logger for audit trails and compliance."""
    
    def __init__(self, logger_name: str = "audit"):
        self.logger = logging.getLogger(logger_name)
        
        # Add custom audit level
        logging.addLevelName(LogLevel.AUDIT.value, "AUDIT")
    
    def log_user_action(self, 
                       action: str,
                       resource: str,
                       user_id: str,
                       result: str = "success",
                       details: Dict[str, Any] = None):
        """Log user actions for audit trail."""
        try:
            audit_data = {
                'action': action,
                'resource': resource,
                'user_id': user_id,
                'result': result,
                'details': details or {},
                'ip_address': self._get_client_ip(),
                'user_agent': self._get_user_agent(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.log(
                LogLevel.AUDIT.value,
                f"User Action: {user_id} {action} {resource} - {result}",
                extra={
                    'category': LogCategory.AUDIT,
                    'extra_data': audit_data
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error logging user action: {e}")
    
    def log_system_event(self,
                        event_type: str,
                        component: str,
                        description: str,
                        severity: str = "info",
                        details: Dict[str, Any] = None):
        """Log system events for audit trail."""
        try:
            audit_data = {
                'event_type': event_type,
                'component': component,
                'description': description,
                'severity': severity,
                'details': details or {},
                'hostname': socket.gethostname(),
                'process_id': os.getpid(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            level = getattr(LogLevel, severity.upper(), LogLevel.INFO)
            
            self.logger.log(
                level.value,
                f"System Event: {component} - {description}",
                extra={
                    'category': LogCategory.AUDIT,
                    'extra_data': audit_data
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error logging system event: {e}")
    
    def log_security_event(self,
                          event_type: str,
                          source_ip: str,
                          description: str,
                          severity: str = "warning",
                          details: Dict[str, Any] = None):
        """Log security events."""
        try:
            security_data = {
                'event_type': event_type,
                'source_ip': source_ip,
                'description': description,
                'severity': severity,
                'details': details or {},
                'hostname': socket.gethostname(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.log(
                LogLevel.SECURITY.value,
                f"Security Event: {event_type} from {source_ip} - {description}",
                extra={
                    'category': LogCategory.SECURITY,
                    'extra_data': security_data
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error logging security event: {e}")
    
    def _get_client_ip(self) -> str:
        """Get client IP address from context."""
        # This would be set by middleware in web applications
        return getattr(threading.current_thread(), 'client_ip', 'unknown')
    
    def _get_user_agent(self) -> str:
        """Get user agent from context."""
        # This would be set by middleware in web applications
        return getattr(threading.current_thread(), 'user_agent', 'unknown')

class LogAggregator:
    """Aggregates logs from multiple sources."""
    
    def __init__(self, 
                 buffer_size: int = 1000,
                 flush_interval: int = 60):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Log buffer
        self.log_buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.aggregated_logs: List[Dict[str, Any]] = []
        
        # Background processing
        self.is_running = False
        self.aggregation_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.stats = {
            'logs_processed': 0,
            'logs_dropped': 0,
            'aggregation_errors': 0,
            'last_flush_time': None
        }
        
        logger = logging.getLogger(__name__)
        logger.info("Log Aggregator initialized")
    
    def start(self):
        """Start log aggregation."""
        if self.is_running:
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        self.aggregation_thread = threading.Thread(
            target=self._aggregation_loop,
            name="LogAggregator",
            daemon=True
        )
        self.aggregation_thread.start()
    
    def stop(self, timeout: int = 30):
        """Stop log aggregation."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.shutdown_event.set()
        
        if self.aggregation_thread and self.aggregation_thread.is_alive():
            self.aggregation_thread.join(timeout=timeout)
        
        # Flush remaining logs
        self._flush_logs()
    
    def add_log(self, log_entry: Dict[str, Any]):
        """Add log entry to buffer."""
        try:
            self.log_buffer.put_nowait(log_entry)
            self.stats['logs_processed'] += 1
        except queue.Full:
            self.stats['logs_dropped'] += 1
    
    def _aggregation_loop(self):
        """Main aggregation loop."""
        while self.is_running:
            try:
                # Process logs in buffer
                logs_to_process = []
                
                # Collect logs from buffer
                while not self.log_buffer.empty() and len(logs_to_process) < 100:
                    try:
                        log_entry = self.log_buffer.get_nowait()
                        logs_to_process.append(log_entry)
                    except queue.Empty:
                        break
                
                # Process collected logs
                if logs_to_process:
                    self._process_log_batch(logs_to_process)
                
                # Wait for next iteration or shutdown
                self.shutdown_event.wait(self.flush_interval)
                
            except Exception as e:
                self.stats['aggregation_errors'] += 1
                # Continue processing even if there's an error
                time.sleep(1)
    
    def _process_log_batch(self, logs: List[Dict[str, Any]]):
        """Process a batch of logs."""
        try:
            # Add logs to aggregated collection
            self.aggregated_logs.extend(logs)
            
            # Perform aggregation analysis
            self._analyze_log_patterns(logs)
            
            # Flush if needed
            if len(self.aggregated_logs) >= self.buffer_size:
                self._flush_logs()
                
        except Exception as e:
            self.stats['aggregation_errors'] += 1
    
    def _analyze_log_patterns(self, logs: List[Dict[str, Any]]):
        """Analyze log patterns for insights."""
        try:
            # Count by level
            level_counts = {}
            category_counts = {}
            
            for log in logs:
                level = log.get('level', 'UNKNOWN')
                category = log.get('category', 'unknown')
                
                level_counts[level] = level_counts.get(level, 0) + 1
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Log analysis results
            logger = logging.getLogger(__name__)
            logger.debug(f"Log analysis - Levels: {level_counts}, Categories: {category_counts}")
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error analyzing log patterns: {e}")
    
    def _flush_logs(self):
        """Flush aggregated logs."""
        try:
            if not self.aggregated_logs:
                return
            
            # Here you would send logs to external systems
            # For now, we'll just clear the buffer
            log_count = len(self.aggregated_logs)
            self.aggregated_logs.clear()
            
            self.stats['last_flush_time'] = datetime.utcnow().isoformat()
            
            logger = logging.getLogger(__name__)
            logger.debug(f"Flushed {log_count} logs")
            
        except Exception as e:
            self.stats['aggregation_errors'] += 1

class BatteryMindLogHandler(logging.Handler):
    """Custom log handler for BatteryMind system."""
    
    def __init__(self, 
                 aggregator: Optional[LogAggregator] = None,
                 enable_alerts: bool = True):
        super().__init__()
        self.aggregator = aggregator
        self.enable_alerts = enable_alerts
        
        # Alert thresholds
        self.error_threshold = 10  # errors per minute
        self.warning_threshold = 50  # warnings per minute
        
        # Rate limiting
        self.error_count = 0
        self.warning_count = 0
        self.last_reset = time.time()
    
    def emit(self, record: logging.LogRecord):
        """Emit log record."""
        try:
            # Format record
            log_entry = self.format_record(record)
            
            # Send to aggregator if available
            if self.aggregator:
                self.aggregator.add_log(log_entry)
            
            # Check for alerts
            if self.enable_alerts:
                self._check_alerts(record)
                
        except Exception as e:
            self.handleError(record)
    
    def format_record(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Format log record as dictionary."""
        try:
            # Get context information
            request_id = request_id_context.get()
            user_id = user_id_context.get()
            session_id = session_id_context.get()
            
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'category': getattr(record, 'category', LogCategory.SYSTEM).value,
                'message': record.getMessage(),
                'logger_name': record.name,
                'module': getattr(record, 'module', ''),
                'function': record.funcName,
                'line_number': record.lineno,
                'request_id': request_id,
                'user_id': user_id,
                'session_id': session_id,
                'hostname': socket.gethostname(),
                'process_id': os.getpid(),
                'thread_id': str(threading.current_thread().ident),
                'extra_data': getattr(record, 'extra_data', {})
            }
            
            # Add exception information if present
            if record.exc_info:
                log_entry['exception_type'] = record.exc_info[0].__name__
                log_entry['exception_message'] = str(record.exc_info[1])
                log_entry['stack_trace'] = self.formatter.formatException(record.exc_info) if self.formatter else ''
            
            return log_entry
            
        except Exception as e:
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'level': 'ERROR',
                'message': f'Error formatting log record: {e}',
                'original_message': getattr(record, 'message', str(record))
            }
    
    def _check_alerts(self, record: logging.LogRecord):
        """Check if alerts should be triggered."""
        try:
            current_time = time.time()
            
            # Reset counters every minute
            if current_time - self.last_reset > 60:
                self.error_count = 0
                self.warning_count = 0
                self.last_reset = current_time
            
            # Count errors and warnings
            if record.levelno >= logging.ERROR:
                self.error_count += 1
                if self.error_count >= self.error_threshold:
                    self._trigger_alert('ERROR_THRESHOLD_EXCEEDED', 
                                      f'Error count exceeded threshold: {self.error_count}')
            
            elif record.levelno >= logging.WARNING:
                self.warning_count += 1
                if self.warning_count >= self.warning_threshold:
                    self._trigger_alert('WARNING_THRESHOLD_EXCEEDED',
                                      f'Warning count exceeded threshold: {self.warning_count}')
                    
        except Exception as e:
            # Don't let alert checking interfere with logging
            pass
    
    def _trigger_alert(self, alert_type: str, message: str):
        """Trigger an alert."""
        try:
            # Here you would integrate with alerting systems
            # For now, just log the alert
            alert_logger = logging.getLogger('alerts')
            alert_logger.critical(f"ALERT: {alert_type} - {message}")
            
        except Exception as e:
            pass

# Decorators for logging
def log_performance(operation_name: str = "", 
                   logger_name: str = "performance",
                   log_args: bool = False,
                   log_result: bool = False):
    """Decorator to log function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            perf_logger = PerformanceLogger(logger_name)
            operation_id = str(uuid.uuid4())
            actual_operation_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Prepare extra data
            extra_data = {
                'function_name': func.__name__,
                'module': func.__module__
            }
            
            if log_args:
                extra_data['args'] = str(args)
                extra_data['kwargs'] = str(kwargs)
            
            try:
                perf_logger.start_timing(operation_id)
                result = func(*args, **kwargs)
                
                if log_result:
                    extra_data['result'] = str(result)
                
                perf_logger.end_timing(operation_id, actual_operation_name, extra_data)
                return result
                
            except Exception as e:
                extra_data['exception'] = str(e)
                perf_logger.end_timing(operation_id, actual_operation_name, extra_data)
                raise
        
        return wrapper
    return decorator

def log_errors(logger_name: str = None, 
               reraise: bool = True,
               log_args: bool = False):
    """Decorator to log function errors."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name or func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                extra_data = {
                    'function_name': func.__name__,
                    'module': func.__module__,
                    'exception_type': type(e).__name__,
                    'exception_message': str(e)
                }
                
                if log_args:
                    extra_data['args'] = str(args)
                    extra_data['kwargs'] = str(kwargs)
                
                logger.error(
                    f"Error in {func.__name__}: {e}",
                    extra={
                        'category': LogCategory.SYSTEM,
                        'extra_data': extra_data
                    },
                    exc_info=True
                )
                
                if reraise:
                    raise
                
        return wrapper
    return decorator

# Utility functions
def setup_logger(name: str,
                level: Union[int, str] = logging.INFO,
                log_file: Optional[str] = None,
                max_bytes: int = 10 * 1024 * 1024,  # 10MB
                backup_count: int = 5,
                structured: bool = True,
                enable_console: bool = True) -> logging.Logger:
    """
    Set up a logger with standard BatteryMind configuration.
    
    Args:
        name: Logger name
        level: Log level
        log_file: Optional log file path
        max_bytes: Maximum log file size
        backup_count: Number of backup files
        structured: Whether to use structured JSON logging
        enable_console: Whether to enable console output
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set up formatter
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with standard configuration."""
    return logging.getLogger(name)

def set_context(request_id: str = None, 
               user_id: str = None, 
               session_id: str = None):
    """Set logging context variables."""
    if request_id:
        request_id_context.set(request_id)
    if user_id:
        user_id_context.set(user_id)
    if session_id:
        session_id_context.set(session_id)

def clear_context():
    """Clear logging context variables."""
    request_id_context.set(None)
    user_id_context.set(None)
    session_id_context.set(None)

# Factory functions
def create_performance_logger(name: str = "performance") -> PerformanceLogger:
    """Create a performance logger instance."""
    return PerformanceLogger(name)

def create_audit_logger(name: str = "audit") -> AuditLogger:
    """Create an audit logger instance."""
    return AuditLogger(name)

def create_log_aggregator(buffer_size: int = 1000, 
                         flush_interval: int = 60) -> LogAggregator:
    """Create a log aggregator instance."""
    return LogAggregator(buffer_size, flush_interval)

# Initialize custom log levels
logging.addLevelName(LogLevel.TRACE.value, "TRACE")
logging.addLevelName(LogLevel.AUDIT.value, "AUDIT")
logging.addLevelName(LogLevel.PERFORMANCE.value, "PERFORMANCE")
logging.addLevelName(LogLevel.SECURITY.value, "SECURITY")

# Module logger
logger = logging.getLogger(__name__)
logger.info("BatteryMind Logging Utils Module v1.0.0 loaded successfully")
