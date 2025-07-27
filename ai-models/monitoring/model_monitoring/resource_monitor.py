"""
BatteryMind - Resource Monitor

Comprehensive system resource monitoring for battery AI models, providing
real-time tracking of CPU, memory, GPU, disk, and network utilization.
Includes automated bottleneck detection, performance optimization recommendations,
and resource usage forecasting.

Features:
- Real-time system resource monitoring
- Multi-process and multi-GPU tracking
- Resource bottleneck detection and alerting
- Performance optimization recommendations
- Resource usage forecasting and capacity planning
- Automated resource scaling recommendations
- Integration with container orchestration systems

Author: BatteryMind Development Team
Version: 1.0.0
"""

import psutil
import GPUtil
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json
import threading
import time
import subprocess
import platform
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns

# BatteryMind imports
from ...utils.logging_utils import setup_logger
from ...utils.data_utils import TimeSeriesAnalyzer
from ...utils.visualization import ResourceVisualizer

# Configure logging
logger = setup_logger(__name__)

@dataclass
class ResourceThresholds:
    """
    Resource utilization thresholds for alerting and optimization.
    
    Attributes:
        # CPU thresholds
        cpu_usage_warning_percent (float): CPU warning threshold
        cpu_usage_critical_percent (float): CPU critical threshold
        cpu_load_warning (float): Load average warning threshold
        
        # Memory thresholds
        memory_usage_warning_percent (float): Memory warning threshold
        memory_usage_critical_percent (float): Memory critical threshold
        swap_usage_warning_percent (float): Swap warning threshold
        
        # Disk thresholds
        disk_usage_warning_percent (float): Disk warning threshold
        disk_usage_critical_percent (float): Disk critical threshold
        disk_io_warning_mbps (float): Disk I/O warning threshold
        
        # Network thresholds
        network_usage_warning_mbps (float): Network warning threshold
        network_usage_critical_mbps (float): Network critical threshold
        
        # GPU thresholds
        gpu_usage_warning_percent (float): GPU warning threshold
        gpu_usage_critical_percent (float): GPU critical threshold
        gpu_memory_warning_percent (float): GPU memory warning threshold
        gpu_memory_critical_percent (float): GPU memory critical threshold
        gpu_temperature_warning_c (float): GPU temperature warning
        gpu_temperature_critical_c (float): GPU temperature critical
        
        # Process thresholds
        process_memory_warning_mb (float): Process memory warning
        process_cpu_warning_percent (float): Process CPU warning
        
        # System health thresholds
        temperature_warning_c (float): System temperature warning
        temperature_critical_c (float): System temperature critical
        
        # Performance thresholds
        response_time_warning_ms (float): Response time warning
        throughput_warning_ops_per_sec (float): Throughput warning
    """
    # CPU thresholds
    cpu_usage_warning_percent: float = 80.0
    cpu_usage_critical_percent: float = 95.0
    cpu_load_warning: float = 0.8
    
    # Memory thresholds
    memory_usage_warning_percent: float = 80.0
    memory_usage_critical_percent: float = 95.0
    swap_usage_warning_percent: float = 50.0
    
    # Disk thresholds
    disk_usage_warning_percent: float = 80.0
    disk_usage_critical_percent: float = 95.0
    disk_io_warning_mbps: float = 100.0
    
    # Network thresholds
    network_usage_warning_mbps: float = 100.0
    network_usage_critical_mbps: float = 500.0
    
    # GPU thresholds
    gpu_usage_warning_percent: float = 85.0
    gpu_usage_critical_percent: float = 95.0
    gpu_memory_warning_percent: float = 80.0
    gpu_memory_critical_percent: float = 95.0
    gpu_temperature_warning_c: float = 80.0
    gpu_temperature_critical_c: float = 90.0
    
    # Process thresholds
    process_memory_warning_mb: float = 1000.0
    process_cpu_warning_percent: float = 50.0
    
    # System health thresholds
    temperature_warning_c: float = 70.0
    temperature_critical_c: float = 80.0
    
    # Performance thresholds
    response_time_warning_ms: float = 100.0
    throughput_warning_ops_per_sec: float = 100.0

@dataclass
class ResourceMonitorConfig:
    """
    Configuration for resource monitoring system.
    
    Attributes:
        # Monitoring settings
        monitoring_interval_seconds (int): Monitoring interval
        detailed_monitoring (bool): Enable detailed monitoring
        monitor_processes (bool): Monitor individual processes
        monitor_gpu (bool): Monitor GPU resources
        monitor_network (bool): Monitor network resources
        monitor_disk (bool): Monitor disk resources
        
        # History settings
        history_retention_hours (int): History retention period
        max_history_points (int): Maximum history points to keep
        enable_forecasting (bool): Enable resource forecasting
        
        # Alerting settings
        enable_alerting (bool): Enable resource alerting
        alert_cooldown_minutes (int): Alert cooldown period
        alert_recipients (List[str]): Alert recipients
        
        # Optimization settings
        enable_auto_optimization (bool): Enable automatic optimization
        optimization_recommendations (bool): Provide optimization recommendations
        
        # Storage settings
        save_metrics (bool): Save resource metrics
        metrics_storage_path (str): Metrics storage path
        compression_enabled (bool): Enable metrics compression
        
        # Visualization settings
        enable_plots (bool): Enable resource plots
        plot_update_interval_seconds (int): Plot update interval
        plot_retention_hours (int): Plot data retention
        
        # Advanced settings
        process_whitelist (List[str]): Processes to monitor specifically
        process_blacklist (List[str]): Processes to ignore
        custom_metrics (Dict[str, str]): Custom metrics definitions
    """
    # Monitoring settings
    monitoring_interval_seconds: int = 5
    detailed_monitoring: bool = True
    monitor_processes: bool = True
    monitor_gpu: bool = True
    monitor_network: bool = True
    monitor_disk: bool = True
    
    # History settings
    history_retention_hours: int = 24
    max_history_points: int = 17280  # 24 hours at 5-second intervals
    enable_forecasting: bool = True
    
    # Alerting settings
    enable_alerting: bool = True
    alert_cooldown_minutes: int = 5
    alert_recipients: List[str] = field(default_factory=list)
    
    # Optimization settings
    enable_auto_optimization: bool = False
    optimization_recommendations: bool = True
    
    # Storage settings
    save_metrics: bool = True
    metrics_storage_path: str = "./resource_metrics"
    compression_enabled: bool = True
    
    # Visualization settings
    enable_plots: bool = True
    plot_update_interval_seconds: int = 30
    plot_retention_hours: int = 6
    
    # Advanced settings
    process_whitelist: List[str] = field(default_factory=lambda: ['python', 'nvidia-smi'])
    process_blacklist: List[str] = field(default_factory=lambda: ['idle', 'System Idle Process'])
    custom_metrics: Dict[str, str] = field(default_factory=dict)

@dataclass
class ResourceSnapshot:
    """Individual resource utilization snapshot."""
    timestamp: datetime
    cpu_percent: float
    cpu_count: int
    cpu_freq: Optional[float]
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    swap_percent: float
    disk_usage_percent: float
    disk_io_read_mbps: float
    disk_io_write_mbps: float
    network_sent_mbps: float
    network_recv_mbps: float
    gpu_metrics: List[Dict[str, Any]] = field(default_factory=list)
    process_metrics: List[Dict[str, Any]] = field(default_factory=list)
    system_load: Optional[float] = None
    temperatures: Dict[str, float] = field(default_factory=dict)

class ResourceMonitor:
    """
    Comprehensive system resource monitoring for battery AI applications.
    """
    
    def __init__(self, 
                 thresholds: ResourceThresholds,
                 config: ResourceMonitorConfig):
        """
        Initialize resource monitor.
        
        Args:
            thresholds: Resource thresholds configuration
            config: Monitor configuration
        """
        self.thresholds = thresholds
        self.config = config
        
        # Initialize monitoring data structures
        self.resource_history = deque(maxlen=config.max_history_points)
        self.alert_history = []
        self.last_alert_time = {}
        
        # Initialize components
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.visualizer = ResourceVisualizer() if config.enable_plots else None
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Cache for network and disk I/O calculations
        self.last_network_stats = None
        self.last_disk_stats = None
        self.last_measurement_time = None
        
        # Create storage directory
        if self.config.save_metrics:
            Path(self.config.metrics_storage_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize GPU monitoring
        self.gpu_available = self._check_gpu_availability()
        
        logger.info("ResourceMonitor initialized with comprehensive monitoring")
    
    def start_monitoring(self):
        """Start real-time resource monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.stop_event.clear()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            self.stop_event.set()
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=10)
            logger.info("Resource monitoring stopped")
    
    def capture_snapshot(self) -> ResourceSnapshot:
        """Capture current resource utilization snapshot."""
        try:
            current_time = datetime.now()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            cpu_freq = None
            try:
                cpu_freq_info = psutil.cpu_freq()
                cpu_freq = cpu_freq_info.current if cpu_freq_info else None
            except:
                pass
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io_read_mbps, disk_io_write_mbps = self._calculate_disk_io()
            
            # Network metrics
            network_sent_mbps, network_recv_mbps = self._calculate_network_io()
            
            # System load
            system_load = None
            try:
                if hasattr(psutil, 'getloadavg'):
                    system_load = psutil.getloadavg()[0] / cpu_count
            except:
                pass
            
            # GPU metrics
            gpu_metrics = []
            if self.config.monitor_gpu and self.gpu_available:
                gpu_metrics = self._get_gpu_metrics()
            
            # Process metrics
            process_metrics = []
            if self.config.monitor_processes:
                process_metrics = self._get_process_metrics()
            
            # Temperature metrics
            temperatures = self._get_temperature_metrics()
            
            snapshot = ResourceSnapshot(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                cpu_freq=cpu_freq,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                swap_percent=swap.percent,
                disk_usage_percent=disk_usage.percent,
                disk_io_read_mbps=disk_io_read_mbps,
                disk_io_write_mbps=disk_io_write_mbps,
                network_sent_mbps=network_sent_mbps,
                network_recv_mbps=network_recv_mbps,
                gpu_metrics=gpu_metrics,
                process_metrics=process_metrics,
                system_load=system_load,
                temperatures=temperatures
            )
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error capturing resource snapshot: {e}")
            # Return minimal snapshot
            return ResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0,
                cpu_count=1,
                cpu_freq=None,
                memory_percent=0,
                memory_used_gb=0,
                memory_total_gb=1,
                swap_percent=0,
                disk_usage_percent=0,
                disk_io_read_mbps=0,
                disk_io_write_mbps=0,
                network_sent_mbps=0,
                network_recv_mbps=0
            )
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            GPUtil.getGPUs()
            return True
        except:
            logger.info("GPU monitoring not available")
            return False
    
    def _calculate_disk_io(self) -> Tuple[float, float]:
        """Calculate disk I/O rates in MB/s."""
        try:
            current_stats = psutil.disk_io_counters()
            current_time = time.time()
            
            if self.last_disk_stats is None or self.last_measurement_time is None:
                self.last_disk_stats = current_stats
                self.last_measurement_time = current_time
                return 0.0, 0.0
            
            time_delta = current_time - self.last_measurement_time
            
            if time_delta <= 0:
                return 0.0, 0.0
            
            read_bytes_delta = current_stats.read_bytes - self.last_disk_stats.read_bytes
            write_bytes_delta = current_stats.write_bytes - self.last_disk_stats.write_bytes
            
            read_mbps = (read_bytes_delta / time_delta) / (1024**2)
            write_mbps = (write_bytes_delta / time_delta) / (1024**2)
            
            self.last_disk_stats = current_stats
            self.last_measurement_time = current_time
            
            return read_mbps, write_mbps
            
        except Exception as e:
            logger.warning(f"Error calculating disk I/O: {e}")
            return 0.0, 0.0
    
    def _calculate_network_io(self) -> Tuple[float, float]:
        """Calculate network I/O rates in MB/s."""
        try:
            current_stats = psutil.net_io_counters()
            current_time = time.time()
            
            if self.last_network_stats is None or self.last_measurement_time is None:
                self.last_network_stats = current_stats
                return 0.0, 0.0
            
            time_delta = current_time - self.last_measurement_time
            
            if time_delta <= 0:
                return 0.0, 0.0
            
            sent_bytes_delta = current_stats.bytes_sent - self.last_network_stats.bytes_sent
            recv_bytes_delta = current_stats.bytes_recv - self.last_network_stats.bytes_recv
            
            sent_mbps = (sent_bytes_delta / time_delta) / (1024**2)
            recv_mbps = (recv_bytes_delta / time_delta) / (1024**2)
            
            self.last_network_stats = current_stats
            
            return sent_mbps, recv_mbps
            
        except Exception as e:
            logger.warning(f"Error calculating network I/O: {e}")
            return 0.0, 0.0
    
    def _get_gpu_metrics(self) -> List[Dict[str, Any]]:
        """Get GPU utilization metrics."""
        gpu_metrics = []
        
        try:
            gpus = GPUtil.getGPUs()
            
            for gpu in gpus:
                gpu_info = {
                    'id': gpu.id,
                    'name': gpu.name,
                    'load_percent': gpu.load * 100,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature_c': gpu.temperature,
                    'power_draw_w': getattr(gpu, 'powerDraw', None),
                    'power_limit_w': getattr(gpu, 'powerLimit', None)
                }
                gpu_metrics.append(gpu_info)
                
        except Exception as e:
            logger.warning(f"Error getting GPU metrics: {e}")
        
        return gpu_metrics
    
    def _get_process_metrics(self) -> List[Dict[str, Any]]:
        """Get process-specific metrics."""
        process_metrics = []
        
        try:
            # Get all processes
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'create_time']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Filter and sort processes
            filtered_processes = []
            for proc in processes:
                proc_name = proc.get('name', '').lower()
                
                # Apply whitelist/blacklist filters
                if self.config.process_whitelist:
                    if not any(wl in proc_name for wl in self.config.process_whitelist):
                        continue
                
                if self.config.process_blacklist:
                    if any(bl in proc_name for bl in self.config.process_blacklist):
                        continue
                
                # Filter by resource usage
                cpu_percent = proc.get('cpu_percent', 0) or 0
                memory_info = proc.get('memory_info')
                memory_mb = memory_info.rss / (1024**2) if memory_info else 0
                
                if (cpu_percent > 1.0 or memory_mb > 50.0):  # Only include processes using significant resources
                    proc_metrics = {
                        'pid': proc.get('pid'),
                        'name': proc.get('name'),
                        'cpu_percent': cpu_percent,
                        'memory_mb': memory_mb,
                        'create_time': proc.get('create_time')
                    }
                    filtered_processes.append(proc_metrics)
            
            # Sort by CPU usage and take top processes
            filtered_processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            process_metrics = filtered_processes[:20]  # Top 20 processes
            
        except Exception as e:
            logger.warning(f"Error getting process metrics: {e}")
        
        return process_metrics
    
    def _get_temperature_metrics(self) -> Dict[str, float]:
        """Get system temperature metrics."""
        temperatures = {}
        
        try:
            # Try to get CPU temperature
            if hasattr(psutil, 'sensors_temperatures'):
                temp_info = psutil.sensors_temperatures()
                
                for name, entries in temp_info.items():
                    for entry in entries:
                        if entry.current:
                            temp_key = f"{name}_{entry.label}" if entry.label else name
                            temperatures[temp_key] = entry.current
            
        except Exception as e:
            logger.debug(f"Temperature monitoring not available: {e}")
        
        return temperatures
    
    def analyze_resource_usage(self, snapshot: ResourceSnapshot) -> Dict[str, Any]:
        """Analyze current resource usage and detect issues."""
        analysis = {
            'timestamp': snapshot.timestamp.isoformat(),
            'alerts': [],
            'bottlenecks': [],
            'recommendations': [],
            'health_score': 100.0
        }
        
        # Analyze CPU usage
        cpu_analysis = self._analyze_cpu_usage(snapshot)
        analysis['alerts'].extend(cpu_analysis['alerts'])
        analysis['bottlenecks'].extend(cpu_analysis['bottlenecks'])
        analysis['recommendations'].extend(cpu_analysis['recommendations'])
        analysis['health_score'] *= cpu_analysis['health_factor']
        
        # Analyze memory usage
        memory_analysis = self._analyze_memory_usage(snapshot)
        analysis['alerts'].extend(memory_analysis['alerts'])
        analysis['bottlenecks'].extend(memory_analysis['bottlenecks'])
        analysis['recommendations'].extend(memory_analysis['recommendations'])
        analysis['health_score'] *= memory_analysis['health_factor']
        
        # Analyze disk usage
        disk_analysis = self._analyze_disk_usage(snapshot)
        analysis['alerts'].extend(disk_analysis['alerts'])
        analysis['bottlenecks'].extend(disk_analysis['bottlenecks'])
        analysis['recommendations'].extend(disk_analysis['recommendations'])
        analysis['health_score'] *= disk_analysis['health_factor']
        
        # Analyze network usage
        network_analysis = self._analyze_network_usage(snapshot)
        analysis['alerts'].extend(network_analysis['alerts'])
        analysis['bottlenecks'].extend(network_analysis['bottlenecks'])
        analysis['recommendations'].extend(network_analysis['recommendations'])
        analysis['health_score'] *= network_analysis['health_factor']
        
        # Analyze GPU usage
        if snapshot.gpu_metrics:
            gpu_analysis = self._analyze_gpu_usage(snapshot)
            analysis['alerts'].extend(gpu_analysis['alerts'])
            analysis['bottlenecks'].extend(gpu_analysis['bottlenecks'])
            analysis['recommendations'].extend(gpu_analysis['recommendations'])
            analysis['health_score'] *= gpu_analysis['health_factor']
        
        # Overall health assessment
        if analysis['health_score'] > 90:
            analysis['overall_status'] = 'healthy'
        elif analysis['health_score'] > 70:
            analysis['overall_status'] = 'warning'
        else:
            analysis['overall_status'] = 'critical'
        
        return analysis
    
    def _analyze_cpu_usage(self, snapshot: ResourceSnapshot) -> Dict[str, Any]:
        """Analyze CPU usage patterns."""
        analysis = {
            'alerts': [],
            'bottlenecks': [],
            'recommendations': [],
            'health_factor': 1.0
        }
        
        cpu_percent = snapshot.cpu_percent
        
        # Check thresholds
        if cpu_percent >= self.thresholds.cpu_usage_critical_percent:
            analysis['alerts'].append({
                'type': 'cpu_critical',
                'severity': 'critical',
                'message': f"CPU usage critical: {cpu_percent:.1f}%",
                'value': cpu_percent,
                'threshold': self.thresholds.cpu_usage_critical_percent
            })
            analysis['bottlenecks'].append('cpu_overload')
            analysis['health_factor'] = 0.3
            
        elif cpu_percent >= self.thresholds.cpu_usage_warning_percent:
            analysis['alerts'].append({
                'type': 'cpu_warning',
                'severity': 'warning',
                'message': f"CPU usage high: {cpu_percent:.1f}%",
                'value': cpu_percent,
                'threshold': self.thresholds.cpu_usage_warning_percent
            })
            analysis['health_factor'] = 0.7
        
        # Check system load
        if snapshot.system_load and snapshot.system_load > self.thresholds.cpu_load_warning:
            analysis['alerts'].append({
                'type': 'load_warning',
                'severity': 'warning',
                'message': f"System load high: {snapshot.system_load:.2f}",
                'value': snapshot.system_load,
                'threshold': self.thresholds.cpu_load_warning
            })
        
        # Generate recommendations
        if cpu_percent > 80:
            analysis['recommendations'].append("Consider scaling up CPU resources or optimizing CPU-intensive processes")
        
        if len(snapshot.process_metrics) > 0:
            top_cpu_process = max(snapshot.process_metrics, key=lambda x: x.get('cpu_percent', 0))
            if top_cpu_process['cpu_percent'] > 50:
                analysis['recommendations'].append(f"Process '{top_cpu_process['name']}' consuming high CPU: {top_cpu_process['cpu_percent']:.1f}%")
        
        return analysis
    
    def _analyze_memory_usage(self, snapshot: ResourceSnapshot) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        analysis = {
            'alerts': [],
            'bottlenecks': [],
            'recommendations': [],
            'health_factor': 1.0
        }
        
        memory_percent = snapshot.memory_percent
        
        # Check memory thresholds
        if memory_percent >= self.thresholds.memory_usage_critical_percent:
            analysis['alerts'].append({
                'type': 'memory_critical',
                'severity': 'critical',
                'message': f"Memory usage critical: {memory_percent:.1f}%",
                'value': memory_percent,
                'threshold': self.thresholds.memory_usage_critical_percent
            })
            analysis['bottlenecks'].append('memory_pressure')
            analysis['health_factor'] = 0.2
            
        elif memory_percent >= self.thresholds.memory_usage_warning_percent:
            analysis['alerts'].append({
                'type': 'memory_warning',
                'severity': 'warning',
                'message': f"Memory usage high: {memory_percent:.1f}%",
                'value': memory_percent,
                'threshold': self.thresholds.memory_usage_warning_percent
            })
            analysis['health_factor'] = 0.6
        
        # Check swap usage
        if snapshot.swap_percent >= self.thresholds.swap_usage_warning_percent:
            analysis['alerts'].append({
                'type': 'swap_warning',
                'severity': 'warning',
                'message': f"Swap usage high: {snapshot.swap_percent:.1f}%",
                'value': snapshot.swap_percent,
                'threshold': self.thresholds.swap_usage_warning_percent
            })
            analysis['bottlenecks'].append('memory_swapping')
        
        # Generate recommendations
        if memory_percent > 85:
            analysis['recommendations'].append("Consider increasing available memory or optimizing memory usage")
        
        if snapshot.swap_percent > 10:
            analysis['recommendations'].append("High swap usage detected - consider adding more RAM")
        
        # Check for memory-intensive processes
        if len(snapshot.process_metrics) > 0:
            top_memory_process = max(snapshot.process_metrics, key=lambda x: x.get('memory_mb', 0))
            if top_memory_process['memory_mb'] > self.thresholds.process_memory_warning_mb:
                analysis['recommendations'].append(f"Process '{top_memory_process['name']}' using high memory: {top_memory_process['memory_mb']:.1f}MB")
        
        return analysis
    
    def _analyze_disk_usage(self, snapshot: ResourceSnapshot) -> Dict[str, Any]:
        """Analyze disk usage patterns."""
        analysis = {
            'alerts': [],
            'bottlenecks': [],
            'recommendations': [],
            'health_factor': 1.0
        }
        
        disk_percent = snapshot.disk_usage_percent
        
        # Check disk space thresholds
        if disk_percent >= self.thresholds.disk_usage_critical_percent:
            analysis['alerts'].append({
                'type': 'disk_space_critical',
                'severity': 'critical',
                'message': f"Disk usage critical: {disk_percent:.1f}%",
                'value': disk_percent,
                'threshold': self.thresholds.disk_usage_critical_percent
            })
            analysis['bottlenecks'].append('disk_space')
            analysis['health_factor'] = 0.3
            
        elif disk_percent >= self.thresholds.disk_usage_warning_percent:
            analysis['alerts'].append({
                'type': 'disk_space_warning',
                'severity': 'warning',
                'message': f"Disk usage high: {disk_percent:.1f}%",
                'value': disk_percent,
                'threshold': self.thresholds.disk_usage_warning_percent
            })
            analysis['health_factor'] = 0.7
        
        # Check disk I/O
        total_disk_io = snapshot.disk_io_read_mbps + snapshot.disk_io_write_mbps
        if total_disk_io >= self.thresholds.disk_io_warning_mbps:
            analysis['alerts'].append({
                'type': 'disk_io_warning',
                'severity': 'warning',
                'message': f"High disk I/O: {total_disk_io:.1f} MB/s",
                'value': total_disk_io,
                'threshold': self.thresholds.disk_io_warning_mbps
            })
            analysis['bottlenecks'].append('disk_io')
        
        # Generate recommendations
        if disk_percent > 90:
            analysis['recommendations'].append("Clean up disk space or add additional storage")
        
        if total_disk_io > 50:
            analysis['recommendations'].append("High disk I/O detected - consider using faster storage or optimizing disk access patterns")
        
        return analysis
    
    def _analyze_network_usage(self, snapshot: ResourceSnapshot) -> Dict[str, Any]:
        """Analyze network usage patterns."""
        analysis = {
            'alerts': [],
            'bottlenecks': [],
            'recommendations': [],
            'health_factor': 1.0
        }
        
        total_network = snapshot.network_sent_mbps + snapshot.network_recv_mbps
        
        # Check network thresholds
        if total_network >= self.thresholds.network_usage_critical_mbps:
            analysis['alerts'].append({
                'type': 'network_critical',
                'severity': 'critical',
                'message': f"Network usage critical: {total_network:.1f} MB/s",
                'value': total_network,
                'threshold': self.thresholds.network_usage_critical_mbps
            })
            analysis['bottlenecks'].append('network_bandwidth')
            analysis['health_factor'] = 0.4
            
        elif total_network >= self.thresholds.network_usage_warning_mbps:
            analysis['alerts'].append({
                'type': 'network_warning',
                'severity': 'warning',
                'message': f"Network usage high: {total_network:.1f} MB/s",
                'value': total_network,
                'threshold': self.thresholds.network_usage_warning_mbps
            })
            analysis['health_factor'] = 0.8
        
        # Generate recommendations
        if total_network > 50:
            analysis['recommendations'].append("High network usage detected - consider optimizing data transfer or upgrading network capacity")
        
        return analysis
    
    def _analyze_gpu_usage(self, snapshot: ResourceSnapshot) -> Dict[str, Any]:
        """Analyze GPU usage patterns."""
        analysis = {
            'alerts': [],
            'bottlenecks': [],
            'recommendations': [],
            'health_factor': 1.0
        }
        
        for gpu in snapshot.gpu_metrics:
            gpu_id = gpu.get('id', 'unknown')
            
            # Check GPU utilization
            gpu_load = gpu.get('load_percent', 0)
            if gpu_load >= self.thresholds.gpu_usage_critical_percent:
                analysis['alerts'].append({
                    'type': 'gpu_critical',
                    'severity': 'critical',
                    'message': f"GPU {gpu_id} usage critical: {gpu_load:.1f}%",
                    'value': gpu_load,
                    'threshold': self.thresholds.gpu_usage_critical_percent
                })
                analysis['bottlenecks'].append(f'gpu_{gpu_id}_compute')
                analysis['health_factor'] = min(analysis['health_factor'], 0.3)
            
            # Check GPU memory
            gpu_memory = gpu.get('memory_percent', 0)
            if gpu_memory >= self.thresholds.gpu_memory_critical_percent:
                analysis['alerts'].append({
                    'type': 'gpu_memory_critical',
                    'severity': 'critical',
                    'message': f"GPU {gpu_id} memory critical: {gpu_memory:.1f}%",
                    'value': gpu_memory,
                    'threshold': self.thresholds.gpu_memory_critical_percent
                })
                analysis['bottlenecks'].append(f'gpu_{gpu_id}_memory')
                analysis['health_factor'] = min(analysis['health_factor'], 0.4)
            
            # Check GPU temperature
            gpu_temp = gpu.get('temperature_c', 0)
            if gpu_temp >= self.thresholds.gpu_temperature_critical_c:
                analysis['alerts'].append({
                    'type': 'gpu_temperature_critical',
                    'severity': 'critical',
                    'message': f"GPU {gpu_id} temperature critical: {gpu_temp:.1f}°C",
                    'value': gpu_temp,
                    'threshold': self.thresholds.gpu_temperature_critical_c
                })
                analysis['bottlenecks'].append(f'gpu_{gpu_id}_thermal')
            
            # Generate recommendations
            if gpu_load > 90:
                analysis['recommendations'].append(f"GPU {gpu_id} highly utilized - consider load balancing or additional GPU resources")
            
            if gpu_memory > 85:
                analysis['recommendations'].append(f"GPU {gpu_id} memory high - optimize memory usage or use GPU with more memory")
            
            if gpu_temp > 80:
                analysis['recommendations'].append(f"GPU {gpu_id} temperature high - check cooling and ventilation")
        
        return analysis
    
    def get_resource_forecast(self, hours_ahead: int = 1) -> Dict[str, Any]:
        """Forecast resource usage for the specified time ahead."""
        if not self.config.enable_forecasting or len(self.resource_history) < 50:
            return {'status': 'insufficient_data'}
        
        try:
            # Prepare time series data
            timestamps = [r.timestamp for r in self.resource_history]
            cpu_data = [r.cpu_percent for r in self.resource_history]
            memory_data = [r.memory_percent for r in self.resource_history]
            
            # Create forecasts
            forecast_results = {}
            
            # CPU forecast
            cpu_forecast = self.time_series_analyzer.forecast(
                timestamps, cpu_data, hours_ahead
            )
            forecast_results['cpu'] = cpu_forecast
            
            # Memory forecast
            memory_forecast = self.time_series_analyzer.forecast(
                timestamps, memory_data, hours_ahead
            )
            forecast_results['memory'] = memory_forecast
            
            # Generate capacity planning recommendations
            recommendations = []
            
            if cpu_forecast.get('predicted_value', 0) > 80:
                recommendations.append("CPU usage expected to be high - consider capacity planning")
            
            if memory_forecast.get('predicted_value', 0) > 80:
                recommendations.append("Memory usage expected to be high - consider adding more RAM")
            
            return {
                'status': 'success',
                'forecast_horizon_hours': hours_ahead,
                'forecasts': forecast_results,
                'recommendations': recommendations,
                'confidence_level': 0.8,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating resource forecast: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                # Capture resource snapshot
                snapshot = self.capture_snapshot()
                
                # Store in history
                self.resource_history.append(snapshot)
                
                # Analyze current usage
                analysis = self.analyze_resource_usage(snapshot)
                
                # Handle alerts
                if self.config.enable_alerting and analysis['alerts']:
                    self._handle_alerts(analysis['alerts'])
                
                # Save metrics if configured
                if self.config.save_metrics:
                    self._save_snapshot(snapshot)
                
                # Update visualizations
                if self.visualizer:
                    self._update_visualizations()
                
                # Auto-optimization if enabled
                if self.config.enable_auto_optimization:
                    self._apply_auto_optimizations(analysis)
                
                # Wait for next interval
                self.stop_event.wait(self.config.monitoring_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _handle_alerts(self, alerts: List[Dict[str, Any]]):
        """Handle resource alerts."""
        for alert in alerts:
            # Check cooldown
            alert_key = f"{alert['type']}_{alert.get('severity', 'unknown')}"
            
            if (alert_key in self.last_alert_time and 
                datetime.now() - self.last_alert_time[alert_key] < timedelta(minutes=self.config.alert_cooldown_minutes)):
                continue
            
            # Add timestamp to alert
            alert['timestamp'] = datetime.now().isoformat()
            
            # Store in history
            self.alert_history.append(alert)
            
            # Update last alert time
            self.last_alert_time[alert_key] = datetime.now()
            
            # Log alert
            severity = alert.get('severity', 'unknown')
            message = alert.get('message', 'Unknown alert')
            
            if severity == 'critical':
                logger.critical(f"RESOURCE ALERT: {message}")
            elif severity == 'warning':
                logger.warning(f"RESOURCE ALERT: {message}")
            else:
                logger.info(f"RESOURCE ALERT: {message}")
    
    def _save_snapshot(self, snapshot: ResourceSnapshot):
        """Save resource snapshot to file."""
        try:
            timestamp_str = snapshot.timestamp.strftime('%Y%m%d_%H%M%S')
            filename = f"resource_snapshot_{timestamp_str}.json"
            filepath = Path(self.config.metrics_storage_path) / filename
            
            # Convert snapshot to dictionary
            snapshot_dict = {
                'timestamp': snapshot.timestamp.isoformat(),
                'cpu_percent': snapshot.cpu_percent,
                'cpu_count': snapshot.cpu_count,
                'cpu_freq': snapshot.cpu_freq,
                'memory_percent': snapshot.memory_percent,
                'memory_used_gb': snapshot.memory_used_gb,
                'memory_total_gb': snapshot.memory_total_gb,
                'swap_percent': snapshot.swap_percent,
                'disk_usage_percent': snapshot.disk_usage_percent,
                'disk_io_read_mbps': snapshot.disk_io_read_mbps,
                'disk_io_write_mbps': snapshot.disk_io_write_mbps,
                'network_sent_mbps': snapshot.network_sent_mbps,
                'network_recv_mbps': snapshot.network_recv_mbps,
                'gpu_metrics': snapshot.gpu_metrics,
                'process_metrics': snapshot.process_metrics,
                'system_load': snapshot.system_load,
                'temperatures': snapshot.temperatures
            }
            
            with open(filepath, 'w') as f:
                json.dump(snapshot_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving resource snapshot: {e}")
    
    def _update_visualizations(self):
        """Update resource visualizations."""
        if not self.visualizer or len(self.resource_history) < 2:
            return
        
        try:
            # Prepare data for visualization
            recent_history = list(self.resource_history)[-self.config.plot_retention_hours * 720:]  # Assuming 5-second intervals
            
            timestamps = [r.timestamp for r in recent_history]
            cpu_data = [r.cpu_percent for r in recent_history]
            memory_data = [r.memory_percent for r in recent_history]
            
            # Update plots
            self.visualizer.update_resource_plots(timestamps, {
                'cpu_percent': cpu_data,
                'memory_percent': memory_data
            })
            
        except Exception as e:
            logger.error(f"Error updating visualizations: {e}")
    
    def _apply_auto_optimizations(self, analysis: Dict[str, Any]):
        """Apply automatic optimizations based on analysis."""
        # This is a placeholder for auto-optimization logic
        # In a real implementation, this would include:
        # - Process priority adjustments
        # - Resource rebalancing
        # - Scaling recommendations
        # - Alert escalation
        pass
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource usage summary."""
        if not self.resource_history:
            return {'status': 'no_data'}
        
        latest_snapshot = self.resource_history[-1]
        
        # Calculate averages over different time windows
        windows = {
            'last_hour': 720,    # 1 hour at 5-second intervals
            'last_6_hours': 4320, # 6 hours
            'last_24_hours': 17280 # 24 hours
        }
        
        averages = {}
        for window_name, window_size in windows.items():
            window_data = list(self.resource_history)[-window_size:]
            if window_data:
                averages[window_name] = {
                    'cpu_percent': np.mean([r.cpu_percent for r in window_data]),
                    'memory_percent': np.mean([r.memory_percent for r in window_data]),
                    'disk_usage_percent': np.mean([r.disk_usage_percent for r in window_data])
                }
        
        summary = {
            'current': {
                'timestamp': latest_snapshot.timestamp.isoformat(),
                'cpu_percent': latest_snapshot.cpu_percent,
                'memory_percent': latest_snapshot.memory_percent,
                'memory_used_gb': latest_snapshot.memory_used_gb,
                'memory_total_gb': latest_snapshot.memory_total_gb,
                'disk_usage_percent': latest_snapshot.disk_usage_percent,
                'gpu_count': len(latest_snapshot.gpu_metrics),
                'active_processes': len(latest_snapshot.process_metrics)
            },
            'averages': averages,
            'alerts': {
                'total_alerts': len(self.alert_history),
                'recent_alerts': len([a for a in self.alert_history 
                                    if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1)]),
                'critical_alerts': len([a for a in self.alert_history if a.get('severity') == 'critical'])
            },
            'monitoring_status': {
                'active': self.monitoring_active,
                'data_points': len(self.resource_history),
                'monitoring_duration_hours': (datetime.now() - self.resource_history[0].timestamp).total_seconds() / 3600 if self.resource_history else 0
            }
        }
        
        return summary

# Factory function
def create_resource_monitor(config: Optional[ResourceMonitorConfig] = None) -> ResourceMonitor:
    """
    Factory function to create a resource monitor.
    
    Args:
        config: Monitor configuration
        
    Returns:
        Configured ResourceMonitor instance
    """
    if config is None:
        config = ResourceMonitorConfig()
    
    thresholds = ResourceThresholds()
    return ResourceMonitor(thresholds, config)
