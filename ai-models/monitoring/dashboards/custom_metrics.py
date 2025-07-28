"""
BatteryMind - Custom Metrics System
Advanced custom metrics collection and processing system for battery management
AI models with flexible metric definitions, real-time aggregation, and custom
visualization capabilities.

Features:
- Flexible custom metric definitions and collection
- Real-time metric aggregation and processing
- Time-series data storage with configurable retention
- Custom dashboard creation and management
- Advanced metric correlation and analysis
- Alerting integration with custom thresholds
- Performance optimization and caching

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import threading
import time
import json
import sqlite3
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from pathlib import Path
import asyncio
import queue

# BatteryMind imports
from ...utils.logging_utils import setup_logger
from ...config.monitoring_config import MonitoringConfig

# Configure logging
logger = setup_logger(__name__)

class AggregationType(Enum):
    """Types of metric aggregation."""
    SUM = "sum"
    AVERAGE = "average"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    COUNT = "count"
    PERCENTILE = "percentile"
    STANDARD_DEVIATION = "std_dev"
    MEDIAN = "median"
    RATE = "rate"
    DELTA = "delta"

class MetricScope(Enum):
    """Scope of metric collection."""
    BATTERY = "battery"
    FLEET = "fleet"
    MODEL = "model"
    SYSTEM = "system"
    BUSINESS = "business"
    GLOBAL = "global"

class DataRetention(Enum):
    """Data retention policies."""
    REAL_TIME = timedelta(hours=1)
    SHORT_TERM = timedelta(days=7)
    MEDIUM_TERM = timedelta(days=30)
    LONG_TERM = timedelta(days=365)
    PERMANENT = None

@dataclass
class MetricDefinition:
    """Definition of a custom metric."""
    
    # Basic properties
    name: str
    description: str = ""
    unit: str = "None"
    scope: MetricScope = MetricScope.SYSTEM
    
    # Data properties
    data_type: type = float
    aggregation_type: AggregationType = AggregationType.AVERAGE
    retention_policy: DataRetention = DataRetention.MEDIUM_TERM
    
    # Collection properties
    collection_interval: int = 60  # seconds
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Validation
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    valid_values: Optional[List[Any]] = None
    
    # Processing
    processor_function: Optional[Callable] = None
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    version: int = 1
    
    def validate_value(self, value: Any) -> bool:
        """Validate a metric value against definition constraints."""
        try:
            # Type validation
            if not isinstance(value, self.data_type):
                try:
                    value = self.data_type(value)
                except (ValueError, TypeError):
                    return False
            
            # Range validation
            if self.min_value is not None and value < self.min_value:
                return False
            
            if self.max_value is not None and value > self.max_value:
                return False
            
            # Valid values validation
            if self.valid_values is not None and value not in self.valid_values:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating metric value for {self.name}: {e}")
            return False

@dataclass
class MetricDataPoint:
    """A single metric data point."""
    
    metric_name: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    dimensions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'metric_name': self.metric_name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'dimensions': self.dimensions,
            'metadata': self.metadata
        }

@dataclass
class AggregatedMetric:
    """Aggregated metric result."""
    
    metric_name: str
    aggregation_type: AggregationType
    value: float
    sample_count: int
    start_time: datetime
    end_time: datetime
    dimensions: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'metric_name': self.metric_name,
            'aggregation_type': self.aggregation_type.value,
            'value': self.value,
            'sample_count': self.sample_count,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'dimensions': self.dimensions
        }

class MetricCollector:
    """
    Collects metrics from various sources.
    """
    
    def __init__(self, 
                 collection_interval: int = 30,
                 batch_size: int = 1000,
                 enable_caching: bool = True):
        
        self.collection_interval = collection_interval
        self.batch_size = batch_size
        self.enable_caching = enable_caching
        
        # Metric definitions registry
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # Collection state
        self.is_collecting = False
        self.collection_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Data collection queue
        self.data_queue: queue.Queue = queue.Queue(maxsize=10000)
        
        # Collector functions
        self.collectors: Dict[str, Callable] = {}
        
        # Statistics
        self.stats = {
            'metrics_collected': 0,
            'collection_errors': 0,
            'data_points_processed': 0,
            'last_collection_time': None
        }
        
        logger.info("Metric Collector initialized")
    
    def register_metric(self, definition: MetricDefinition) -> bool:
        """
        Register a metric definition.
        
        Args:
            definition: Metric definition
            
        Returns:
            Success status
        """
        try:
            self.metric_definitions[definition.name] = definition
            logger.info(f"Registered metric: {definition.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering metric {definition.name}: {e}")
            return False
    
    def register_collector(self, metric_name: str, collector_func: Callable) -> bool:
        """
        Register a collector function for a metric.
        
        Args:
            metric_name: Name of the metric
            collector_func: Function that collects the metric value
            
        Returns:
            Success status
        """
        try:
            self.collectors[metric_name] = collector_func
            logger.info(f"Registered collector for metric: {metric_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering collector for {metric_name}: {e}")
            return False
    
    def start_collection(self):
        """Start metric collection."""
        if self.is_collecting:
            logger.warning("Metric collection already running")
            return
        
        self.is_collecting = True
        self.shutdown_event.clear()
        
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            name="MetricCollector",
            daemon=True
        )
        self.collection_thread.start()
        
        logger.info("Metric collection started")
    
    def stop_collection(self, timeout: int = 30):
        """Stop metric collection."""
        if not self.is_collecting:
            return
        
        logger.info("Stopping metric collection...")
        
        self.is_collecting = False
        self.shutdown_event.set()
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=timeout)
        
        logger.info("Metric collection stopped")
    
    def _collection_loop(self):
        """Main collection loop."""
        while self.is_collecting and not self.shutdown_event.is_set():
            try:
                collection_start = time.time()
                
                # Collect all registered metrics
                for metric_name, definition in self.metric_definitions.items():
                    if not definition.enabled:
                        continue
                    
                    try:
                        self._collect_metric(metric_name, definition)
                    except Exception as e:
                        logger.error(f"Error collecting metric {metric_name}: {e}")
                        self.stats['collection_errors'] += 1
                
                self.stats['last_collection_time'] = datetime.now()
                collection_time = time.time() - collection_start
                
                # Sleep for remaining interval time
                sleep_time = max(0, self.collection_interval - collection_time)
                self.shutdown_event.wait(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(10)
    
    def _collect_metric(self, metric_name: str, definition: MetricDefinition):
        """Collect a single metric."""
        try:
            # Get collector function
            collector = self.collectors.get(metric_name)
            if not collector:
                logger.debug(f"No collector registered for metric: {metric_name}")
                return
            
            # Collect metric value
            value = collector()
            
            # Validate value
            if not definition.validate_value(value):
                logger.warning(f"Invalid value for metric {metric_name}: {value}")
                return
            
            # Process value if processor function exists
            if definition.processor_function:
                value = definition.processor_function(value)
            
            # Create data point
            data_point = MetricDataPoint(
                metric_name=metric_name,
                value=value,
                dimensions=definition.tags
            )
            
            # Queue data point for processing
            try:
                self.data_queue.put_nowait(data_point)
                self.stats['metrics_collected'] += 1
            except queue.Full:
                logger.warning(f"Data queue full, dropping metric: {metric_name}")
            
        except Exception as e:
            logger.error(f"Error collecting metric {metric_name}: {e}")
            raise
    
    def get_queued_data(self, max_items: Optional[int] = None) -> List[MetricDataPoint]:
        """Get queued metric data points."""
        data_points = []
        max_items = max_items or self.batch_size
        
        for _ in range(min(max_items, self.data_queue.qsize())):
            try:
                data_point = self.data_queue.get_nowait()
                data_points.append(data_point)
                self.stats['data_points_processed'] += 1
            except queue.Empty:
                break
        
        return data_points

class MetricProcessor:
    """
    Processes and aggregates collected metrics.
    """
    
    def __init__(self, 
                 processing_interval: int = 60,
                 aggregation_windows: List[int] = None):
        
        self.processing_interval = processing_interval
        self.aggregation_windows = aggregation_windows or [60, 300, 900, 3600]  # 1m, 5m, 15m, 1h
        
        # Processing state
        self.is_processing = False
        self.processing_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Data storage
        self.raw_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregated_data: Dict[str, Dict[int, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )
        
        # Processing lock
        self.data_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'data_points_processed': 0,
            'aggregations_computed': 0,
            'processing_errors': 0,
            'last_processing_time': None
        }
        
        logger.info("Metric Processor initialized")
    
    def start_processing(self):
        """Start metric processing."""
        if self.is_processing:
            logger.warning("Metric processing already running")
            return
        
        self.is_processing = True
        self.shutdown_event.clear()
        
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name="MetricProcessor",
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info("Metric processing started")
    
    def stop_processing(self, timeout: int = 30):
        """Stop metric processing."""
        if not self.is_processing:
            return
        
        logger.info("Stopping metric processing...")
        
        self.is_processing = False
        self.shutdown_event.set()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=timeout)
        
        logger.info("Metric processing stopped")
    
    def add_data_points(self, data_points: List[MetricDataPoint]):
        """Add data points for processing."""
        with self.data_lock:
            for data_point in data_points:
                self.raw_data[data_point.metric_name].append(data_point)
                self.stats['data_points_processed'] += 1
    
    def _processing_loop(self):
        """Main processing loop."""
        while self.is_processing and not self.shutdown_event.is_set():
            try:
                processing_start = time.time()
                
                # Process aggregations for all windows
                for window_seconds in self.aggregation_windows:
                    self._process_aggregations(window_seconds)
                
                self.stats['last_processing_time'] = datetime.now()
                processing_time = time.time() - processing_start
                
                # Sleep for remaining interval time
                sleep_time = max(0, self.processing_interval - processing_time)
                self.shutdown_event.wait(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self.stats['processing_errors'] += 1
                time.sleep(10)
    
    def _process_aggregations(self, window_seconds: int):
        """Process aggregations for a specific time window."""
        try:
            current_time = datetime.now()
            window_start = current_time - timedelta(seconds=window_seconds)
            
            with self.data_lock:
                for metric_name, raw_data in self.raw_data.items():
                    # Filter data points within window
                    window_data = [
                        dp for dp in raw_data 
                        if dp.timestamp >= window_start
                    ]
                    
                    if not window_data:
                        continue
                    
                    # Calculate aggregations
                    aggregations = self._calculate_aggregations(window_data)
                    
                    # Store aggregated results
                    for agg_type, value in aggregations.items():
                        aggregated_metric = AggregatedMetric(
                            metric_name=metric_name,
                            aggregation_type=agg_type,
                            value=value,
                            sample_count=len(window_data),
                            start_time=window_start,
                            end_time=current_time
                        )
                        
                        self.aggregated_data[metric_name][window_seconds].append(aggregated_metric)
                        self.stats['aggregations_computed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing aggregations for window {window_seconds}s: {e}")
            raise
    
    def _calculate_aggregations(self, data_points: List[MetricDataPoint]) -> Dict[AggregationType, float]:
        """Calculate various aggregations for data points."""
        try:
            values = [dp.value for dp in data_points if isinstance(dp.value, (int, float))]
            
            if not values:
                return {}
            
            aggregations = {}
            
            # Basic aggregations
            aggregations[AggregationType.SUM] = sum(values)
            aggregations[AggregationType.AVERAGE] = np.mean(values)
            aggregations[AggregationType.MINIMUM] = min(values)
            aggregations[AggregationType.MAXIMUM] = max(values)
            aggregations[AggregationType.COUNT] = len(values)
            aggregations[AggregationType.MEDIAN] = np.median(values)
            aggregations[AggregationType.STANDARD_DEVIATION] = np.std(values)
            
            # Percentiles
            aggregations[AggregationType.PERCENTILE] = np.percentile(values, 95)
            
            # Rate calculation (if we have time series data)
            if len(data_points) > 1:
                time_diff = (data_points[-1].timestamp - data_points[0].timestamp).total_seconds()
                if time_diff > 0:
                    aggregations[AggregationType.RATE] = len(values) / time_diff
            
            return aggregations
            
        except Exception as e:
            logger.error(f"Error calculating aggregations: {e}")
            return {}
    
    def get_aggregated_metrics(self, 
                              metric_name: str,
                              window_seconds: int,
                              limit: int = 100) -> List[AggregatedMetric]:
        """Get aggregated metrics for a specific metric and window."""
        with self.data_lock:
            if metric_name in self.aggregated_data and window_seconds in self.aggregated_data[metric_name]:
                data = list(self.aggregated_data[metric_name][window_seconds])
                return data[-limit:] if limit else data
            return []

class MetricAggregator:
    """
    High-level metric aggregation and analysis.
    """
    
    def __init__(self):
        self.processor = None
        
        # Analysis cache
        self.analysis_cache: Dict[str, Any] = {}
        self.cache_lock = threading.RLock()
        
        logger.info("Metric Aggregator initialized")
    
    def set_processor(self, processor: MetricProcessor):
        """Set the metric processor."""
        self.processor = processor
    
    def get_metric_summary(self, 
                          metric_name: str,
                          time_range: timedelta = timedelta(hours=1)) -> Optional[Dict[str, Any]]:
        """Get a comprehensive summary of a metric."""
        if not self.processor:
            return None
        
        try:
            summary = {
                'metric_name': metric_name,
                'time_range': time_range,
                'timestamp': datetime.now(),
                'windows': {}
            }
            
            # Get data for different windows
            for window_seconds in self.processor.aggregation_windows:
                aggregated_metrics = self.processor.get_aggregated_metrics(
                    metric_name, window_seconds, limit=100
                )
                
                if aggregated_metrics:
                    # Get latest values for each aggregation type
                    latest_by_type = {}
                    for metric in aggregated_metrics:
                        latest_by_type[metric.aggregation_type] = metric.value
                    
                    summary['windows'][f'{window_seconds}s'] = latest_by_type
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting metric summary for {metric_name}: {e}")
            return None
    
    def analyze_metric_trends(self, 
                             metric_name: str,
                             window_seconds: int = 3600,
                             lookback_periods: int = 24) -> Optional[Dict[str, Any]]:
        """Analyze trends in metric data."""
        if not self.processor:
            return None
        
        try:
            aggregated_metrics = self.processor.get_aggregated_metrics(
                metric_name, window_seconds, limit=lookback_periods
            )
            
            if len(aggregated_metrics) < 2:
                return None
            
            # Extract average values for trend analysis
            avg_values = [
                m.value for m in aggregated_metrics 
                if m.aggregation_type == AggregationType.AVERAGE
            ]
            
            if len(avg_values) < 2:
                return None
            
            # Calculate trend metrics
            trend_analysis = {
                'metric_name': metric_name,
                'window_seconds': window_seconds,
                'data_points': len(avg_values),
                'latest_value': avg_values[-1],
                'trend_direction': self._calculate_trend_direction(avg_values),
                'volatility': np.std(avg_values) if len(avg_values) > 1 else 0,
                'change_rate': self._calculate_change_rate(avg_values),
                'anomalies': self._detect_anomalies(avg_values)
            }
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trends for {metric_name}: {e}")
            return None
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate overall trend direction."""
        if len(values) < 2:
            return "unknown"
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_change_rate(self, values: List[float]) -> float:
        """Calculate rate of change."""
        if len(values) < 2:
            return 0.0
        
        return (values[-1] - values[0]) / len(values)
    
    def _detect_anomalies(self, values: List[float]) -> List[int]:
        """Detect anomalies in metric values."""
        if len(values) < 5:
            return []
        
        # Simple z-score based anomaly detection
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return []
        
        z_scores = [(v - mean_val) / std_val for v in values]
        anomalies = [i for i, z in enumerate(z_scores) if abs(z) > 2.5]
        
        return anomalies

@dataclass
class CustomDashboard:
    """Custom dashboard configuration."""
    
    dashboard_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    metrics: List[str] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)
    refresh_interval: int = 30
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)

class CustomMetrics:
    """
    Main custom metrics management system.
    """
    
    def __init__(self, 
                 storage_path: str = "./custom_metrics.db",
                 collection_interval: int = 30,
                 processing_interval: int = 60):
        
        self.storage_path = storage_path
        
        # Initialize components
        self.collector = MetricCollector(collection_interval=collection_interval)
        self.processor = MetricProcessor(processing_interval=processing_interval)
        self.aggregator = MetricAggregator()
        
        # Link components
        self.aggregator.set_processor(self.processor)
        
        # Database for persistence
        self.db_connection = None
        self._initialize_database()
        
        # Processing coordination
        self.coordination_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Dashboard management
        self.dashboards: Dict[str, CustomDashboard] = {}
        
        logger.info("Custom Metrics system initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database for metrics storage."""
        try:
            db_path = Path(self.storage_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
            
            # Create tables
            cursor = self.db_connection.cursor()
            
            # Metric definitions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metric_definitions (
                    name TEXT PRIMARY KEY,
                    definition TEXT,
                    created_at TIMESTAMP
                )
            ''')
            
            # Raw data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS raw_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    value REAL,
                    timestamp TIMESTAMP,
                    dimensions TEXT,
                    metadata TEXT
                )
            ''')
            
            # Aggregated data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS aggregated_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    aggregation_type TEXT,
                    value REAL,
                    sample_count INTEGER,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    window_seconds INTEGER,
                    dimensions TEXT
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_timestamp ON raw_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_metric_name ON raw_metrics(metric_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_agg_metric_name ON aggregated_metrics(metric_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_agg_window ON aggregated_metrics(window_seconds)')
            
            self.db_connection.commit()
            logger.info("Custom metrics database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def start(self):
        """Start the custom metrics system."""
        if self.is_running:
            logger.warning("Custom metrics system already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start components
        self.collector.start_collection()
        self.processor.start_processing()
        
        # Start coordination thread
        self.coordination_thread = threading.Thread(
            target=self._coordination_loop,
            name="CustomMetrics-Coordinator",
            daemon=True
        )
        self.coordination_thread.start()
        
        logger.info("Custom metrics system started")
    
    def stop(self, timeout: int = 30):
        """Stop the custom metrics system."""
        if not self.is_running:
            return
        
        logger.info("Stopping custom metrics system...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop components
        self.collector.stop_collection(timeout)
        self.processor.stop_processing(timeout)
        
        if self.coordination_thread and self.coordination_thread.is_alive():
            self.coordination_thread.join(timeout=timeout)
        
        # Close database
        if self.db_connection:
            self.db_connection.close()
        
        logger.info("Custom metrics system stopped")
    
    def _coordination_loop(self):
        """Coordinate data flow between collector and processor."""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Get collected data
                data_points = self.collector.get_queued_data(max_items=1000)
                
                if data_points:
                    # Send to processor
                    self.processor.add_data_points(data_points)
                    
                    # Persist raw data
                    self._persist_raw_data(data_points)
                
                # Brief sleep
                self.shutdown_event.wait(1)
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                time.sleep(5)
    
    def _persist_raw_data(self, data_points: List[MetricDataPoint]):
        """Persist raw metric data to database."""
        try:
            cursor = self.db_connection.cursor()
            
            for dp in data_points:
                cursor.execute('''
                    INSERT INTO raw_metrics (metric_name, value, timestamp, dimensions, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    dp.metric_name,
                    dp.value,
                    dp.timestamp,
                    json.dumps(dp.dimensions),
                    json.dumps(dp.metadata)
                ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error persisting raw data: {e}")
    
    def register_metric(self, definition: MetricDefinition) -> bool:
        """Register a custom metric definition."""
        success = self.collector.register_metric(definition)
        
        if success:
            # Persist definition
            try:
                cursor = self.db_connection.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO metric_definitions (name, definition, created_at)
                    VALUES (?, ?, ?)
                ''', (
                    definition.name,
                    json.dumps(asdict(definition), default=str),
                    definition.created_at
                ))
                self.db_connection.commit()
                
            except Exception as e:
                logger.error(f"Error persisting metric definition: {e}")
        
        return success
    
    def register_collector(self, metric_name: str, collector_func: Callable) -> bool:
        """Register a collector function for a metric."""
        return self.collector.register_collector(metric_name, collector_func)
    
    def get_metric_summary(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive metric summary."""
        return self.aggregator.get_metric_summary(metric_name)
    
    def analyze_trends(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Analyze metric trends."""
        return self.aggregator.analyze_metric_trends(metric_name)
    
    def create_dashboard(self, dashboard: CustomDashboard) -> bool:
        """Create a custom dashboard."""
        try:
            self.dashboards[dashboard.dashboard_id] = dashboard
            logger.info(f"Created custom dashboard: {dashboard.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        return {
            'collector_stats': self.collector.stats,
            'processor_stats': self.processor.stats,
            'total_dashboards': len(self.dashboards),
            'is_running': self.is_running
        }

# Factory functions and default metrics
def create_battery_health_metrics() -> List[MetricDefinition]:
    """Create standard battery health metrics."""
    
    metrics = [
        MetricDefinition(
            name="battery_soh",
            description="Battery State of Health",
            unit="percent",
            scope=MetricScope.BATTERY,
            data_type=float,
            min_value=0.0,
            max_value=100.0,
            alert_thresholds={"critical": 70.0, "warning": 80.0}
        ),
        MetricDefinition(
            name="battery_temperature",
            description="Battery Temperature",
            unit="celsius",
            scope=MetricScope.BATTERY,
            data_type=float,
            min_value=-40.0,
            max_value=100.0,
            alert_thresholds={"critical": 60.0, "warning": 50.0}
        ),
        MetricDefinition(
            name="battery_voltage",
            description="Battery Voltage",
            unit="volts",
            scope=MetricScope.BATTERY,
            data_type=float,
            min_value=0.0,
            max_value=1000.0,
            alert_thresholds={"critical_low": 2.5, "critical_high": 4.5}
        ),
        MetricDefinition(
            name="model_accuracy",
            description="AI Model Prediction Accuracy",
            unit="percent",
            scope=MetricScope.MODEL,
            data_type=float,
            min_value=0.0,
            max_value=100.0,
            alert_thresholds={"critical": 90.0, "warning": 95.0}
        )
    ]
    
    return metrics

def create_custom_metrics_system(storage_path: str = "./custom_metrics.db") -> CustomMetrics:
    """Create a fully configured custom metrics system."""
    
    system = CustomMetrics(storage_path=storage_path)
    
    # Register default metrics
    default_metrics = create_battery_health_metrics()
    for metric_def in default_metrics:
        system.register_metric(metric_def)
    
    return system
