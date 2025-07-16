"""
BatteryMind - Inference Pipelines Module

Comprehensive inference pipeline system for battery management AI/ML models
providing real-time, batch, and edge inference capabilities with robust
error handling, monitoring, and optimization features.

This module provides:
- Unified inference pipeline interface
- Real-time streaming inference
- Batch processing capabilities
- Edge deployment optimization
- Performance monitoring and metrics
- Error handling and fallback mechanisms
- Caching and optimization features

Features:
- Multi-model pipeline orchestration
- Adaptive resource management
- Automatic scaling and load balancing
- Comprehensive logging and monitoring
- Security and privacy controls
- Integration with model versioning

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
from contextlib import contextmanager

# Import pipeline components
from .inference_pipeline import (
    InferencePipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    create_inference_pipeline
)

from .batch_inference import (
    BatchInferencePipeline,
    BatchConfig,
    BatchResult,
    BatchProcessor,
    create_batch_processor
)

from .real_time_inference import (
    RealTimeInferencePipeline,
    RealTimeConfig,
    StreamingProcessor,
    create_real_time_processor
)

from .edge_inference import (
    EdgeInferencePipeline,
    EdgeConfig,
    EdgeOptimizer,
    create_edge_processor
)

# Utility imports
from ...utils.logging_utils import get_logger
from ...utils.config_parser import load_config
from ...utils.model_utils import ModelManager
from ...monitoring.model_monitoring.performance_monitor import PerformanceMonitor

# Configure logging
logger = get_logger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module exports
__all__ = [
    # Pipeline classes
    "InferencePipeline",
    "BatchInferencePipeline", 
    "RealTimeInferencePipeline",
    "EdgeInferencePipeline",
    
    # Configuration classes
    "PipelineConfig",
    "BatchConfig",
    "RealTimeConfig",
    "EdgeConfig",
    
    # Result classes
    "PipelineResult",
    "BatchResult",
    
    # Processor classes
    "BatchProcessor",
    "StreamingProcessor",
    "EdgeOptimizer",
    
    # Pipeline stages
    "PipelineStage",
    
    # Factory functions
    "create_inference_pipeline",
    "create_batch_processor",
    "create_real_time_processor",
    "create_edge_processor",
    
    # Pipeline manager
    "PipelineManager",
    "PipelineRegistry",
    
    # Utility functions
    "get_pipeline_factory",
    "validate_pipeline_config",
    "optimize_pipeline_performance",
    "monitor_pipeline_health"
]

@dataclass
class PipelineMetrics:
    """
    Comprehensive metrics for pipeline performance monitoring.
    
    Attributes:
        total_requests (int): Total number of inference requests
        successful_requests (int): Number of successful requests
        failed_requests (int): Number of failed requests
        average_latency_ms (float): Average response latency
        throughput_rps (float): Requests per second throughput
        error_rate (float): Error rate (0-1)
        resource_utilization (Dict[str, float]): Resource utilization metrics
        model_performance (Dict[str, float]): Per-model performance metrics
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    model_performance: Dict[str, float] = field(default_factory=dict)

class PipelineRegistry:
    """
    Registry for managing multiple inference pipelines.
    """
    
    def __init__(self):
        self._pipelines: Dict[str, InferencePipeline] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._metrics: Dict[str, PipelineMetrics] = {}
        self._performance_monitor = PerformanceMonitor()
        self._lock = threading.Lock()
        
        logger.info("PipelineRegistry initialized")
    
    def register_pipeline(self, name: str, pipeline: InferencePipeline, 
                         config: Dict[str, Any] = None):
        """
        Register a pipeline with the registry.
        
        Args:
            name: Pipeline name
            pipeline: Pipeline instance
            config: Pipeline configuration
        """
        with self._lock:
            self._pipelines[name] = pipeline
            self._configs[name] = config or {}
            self._metrics[name] = PipelineMetrics()
            
            logger.info(f"Pipeline '{name}' registered")
    
    def get_pipeline(self, name: str) -> Optional[InferencePipeline]:
        """Get pipeline by name."""
        return self._pipelines.get(name)
    
    def list_pipelines(self) -> List[str]:
        """List all registered pipeline names."""
        return list(self._pipelines.keys())
    
    def get_pipeline_metrics(self, name: str) -> Optional[PipelineMetrics]:
        """Get metrics for a specific pipeline."""
        return self._metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, PipelineMetrics]:
        """Get metrics for all pipelines."""
        return self._metrics.copy()
    
    def remove_pipeline(self, name: str) -> bool:
        """Remove pipeline from registry."""
        with self._lock:
            if name in self._pipelines:
                del self._pipelines[name]
                del self._configs[name]
                del self._metrics[name]
                logger.info(f"Pipeline '{name}' removed")
                return True
            return False
    
    def update_metrics(self, name: str, latency: float, success: bool):
        """Update pipeline metrics."""
        if name not in self._metrics:
            return
        
        with self._lock:
            metrics = self._metrics[name]
            metrics.total_requests += 1
            
            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
            
            # Update average latency
            total_req = metrics.total_requests
            old_avg = metrics.average_latency_ms
            metrics.average_latency_ms = (old_avg * (total_req - 1) + latency) / total_req
            
            # Update error rate
            metrics.error_rate = metrics.failed_requests / metrics.total_requests
            
            # Update throughput (simplified)
            metrics.throughput_rps = min(1000.0, 1000.0 / max(metrics.average_latency_ms, 1.0))

class PipelineManager:
    """
    Central manager for all inference pipelines.
    """
    
    def __init__(self):
        self.registry = PipelineRegistry()
        self.model_manager = ModelManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Pipeline factories
        self._pipeline_factories = {
            'inference': create_inference_pipeline,
            'batch': create_batch_processor,
            'real_time': create_real_time_processor,
            'edge': create_edge_processor
        }
        
        # Default configurations
        self._default_configs = {
            'inference': {
                'enable_caching': True,
                'max_batch_size': 32,
                'timeout_seconds': 30.0
            },
            'batch': {
                'batch_size': 1000,
                'max_workers': 4,
                'chunk_size': 100
            },
            'real_time': {
                'buffer_size': 10000,
                'processing_interval_ms': 100,
                'max_latency_ms': 500
            },
            'edge': {
                'model_compression': True,
                'quantization_enabled': True,
                'memory_limit_mb': 100
            }
        }
        
        logger.info("PipelineManager initialized")
    
    def create_pipeline(self, pipeline_type: str, name: str, 
                       config: Dict[str, Any] = None) -> InferencePipeline:
        """
        Create and register a new pipeline.
        
        Args:
            pipeline_type: Type of pipeline ('inference', 'batch', 'real_time', 'edge')
            name: Pipeline name
            config: Pipeline configuration
            
        Returns:
            InferencePipeline: Created pipeline instance
        """
        if pipeline_type not in self._pipeline_factories:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        # Merge with default config
        final_config = self._default_configs.get(pipeline_type, {}).copy()
        if config:
            final_config.update(config)
        
        # Create pipeline
        factory = self._pipeline_factories[pipeline_type]
        pipeline = factory(final_config)
        
        # Register pipeline
        self.registry.register_pipeline(name, pipeline, final_config)
        
        logger.info(f"Pipeline '{name}' of type '{pipeline_type}' created")
        return pipeline
    
    def get_pipeline(self, name: str) -> Optional[InferencePipeline]:
        """Get pipeline by name."""
        return self.registry.get_pipeline(name)
    
    def execute_pipeline(self, name: str, data: Any) -> Any:
        """
        Execute pipeline with performance monitoring.
        
        Args:
            name: Pipeline name
            data: Input data
            
        Returns:
            Pipeline execution result
        """
        pipeline = self.registry.get_pipeline(name)
        if not pipeline:
            raise ValueError(f"Pipeline '{name}' not found")
        
        start_time = time.time()
        success = False
        
        try:
            result = pipeline.process(data)
            success = True
            return result
        except Exception as e:
            logger.error(f"Pipeline '{name}' execution failed: {e}")
            raise
        finally:
            # Update metrics
            latency = (time.time() - start_time) * 1000  # Convert to ms
            self.registry.update_metrics(name, latency, success)
    
    def get_pipeline_status(self, name: str) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        pipeline = self.registry.get_pipeline(name)
        if not pipeline:
            return {"error": f"Pipeline '{name}' not found"}
        
        metrics = self.registry.get_pipeline_metrics(name)
        
        return {
            "name": name,
            "status": "healthy" if pipeline else "error",
            "metrics": metrics.__dict__ if metrics else {},
            "configuration": self.registry._configs.get(name, {}),
            "last_updated": time.time()
        }
    
    def get_all_pipeline_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all pipelines."""
        status = {}
        for name in self.registry.list_pipelines():
            status[name] = self.get_pipeline_status(name)
        return status
    
    def optimize_pipeline_performance(self, name: str) -> Dict[str, Any]:
        """
        Optimize pipeline performance based on metrics.
        
        Args:
            name: Pipeline name
            
        Returns:
            Dict containing optimization results
        """
        metrics = self.registry.get_pipeline_metrics(name)
        if not metrics:
            return {"error": f"No metrics found for pipeline '{name}'"}
        
        optimization_results = {
            "pipeline_name": name,
            "optimization_applied": [],
            "performance_improvement": {},
            "recommendations": []
        }
        
        # Analyze performance and suggest optimizations
        if metrics.average_latency_ms > 1000:
            optimization_results["recommendations"].append(
                "Consider enabling caching or model compression"
            )
        
        if metrics.error_rate > 0.05:
            optimization_results["recommendations"].append(
                "High error rate detected - review input validation and error handling"
            )
        
        if metrics.throughput_rps < 100:
            optimization_results["recommendations"].append(
                "Low throughput - consider batch processing or parallel execution"
            )
        
        return optimization_results
    
    def monitor_pipeline_health(self) -> Dict[str, Any]:
        """
        Monitor health of all pipelines.
        
        Returns:
            Dict containing health status and recommendations
        """
        health_report = {
            "timestamp": time.time(),
            "overall_health": "healthy",
            "pipeline_count": len(self.registry.list_pipelines()),
            "pipeline_health": {},
            "system_recommendations": []
        }
        
        unhealthy_pipelines = []
        
        for name in self.registry.list_pipelines():
            metrics = self.registry.get_pipeline_metrics(name)
            if metrics:
                pipeline_health = "healthy"
                
                # Check health criteria
                if metrics.error_rate > 0.1:
                    pipeline_health = "unhealthy"
                    unhealthy_pipelines.append(name)
                elif metrics.average_latency_ms > 5000:
                    pipeline_health = "degraded"
                elif metrics.error_rate > 0.05:
                    pipeline_health = "warning"
                
                health_report["pipeline_health"][name] = {
                    "status": pipeline_health,
                    "error_rate": metrics.error_rate,
                    "avg_latency_ms": metrics.average_latency_ms,
                    "throughput_rps": metrics.throughput_rps
                }
        
        # Update overall health
        if unhealthy_pipelines:
            health_report["overall_health"] = "degraded"
            health_report["system_recommendations"].append(
                f"Investigate unhealthy pipelines: {', '.join(unhealthy_pipelines)}"
            )
        
        return health_report
    
    def shutdown_pipeline(self, name: str) -> bool:
        """Shutdown and remove pipeline."""
        pipeline = self.registry.get_pipeline(name)
        if pipeline and hasattr(pipeline, 'shutdown'):
            pipeline.shutdown()
        
        return self.registry.remove_pipeline(name)
    
    def shutdown_all_pipelines(self):
        """Shutdown all pipelines."""
        for name in self.registry.list_pipelines():
            self.shutdown_pipeline(name)
        
        logger.info("All pipelines shut down")

# Global pipeline manager instance
_pipeline_manager = None

def get_pipeline_manager() -> PipelineManager:
    """Get the global pipeline manager instance."""
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager()
    return _pipeline_manager

def get_pipeline_factory(pipeline_type: str) -> Optional[Callable]:
    """
    Get pipeline factory function by type.
    
    Args:
        pipeline_type: Type of pipeline
        
    Returns:
        Factory function or None if not found
    """
    manager = get_pipeline_manager()
    return manager._pipeline_factories.get(pipeline_type)

def validate_pipeline_config(config: Dict[str, Any], pipeline_type: str) -> Dict[str, Any]:
    """
    Validate pipeline configuration.
    
    Args:
        config: Configuration to validate
        pipeline_type: Type of pipeline
        
    Returns:
        Validation results
    """
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "recommendations": []
    }
    
    # Required fields by pipeline type
    required_fields = {
        'inference': ['model_path'],
        'batch': ['batch_size', 'max_workers'],
        'real_time': ['buffer_size', 'processing_interval_ms'],
        'edge': ['memory_limit_mb']
    }
    
    # Check required fields
    if pipeline_type in required_fields:
        for field in required_fields[pipeline_type]:
            if field not in config:
                validation_results["errors"].append(f"Missing required field: {field}")
                validation_results["is_valid"] = False
    
    # Performance recommendations
    if pipeline_type == 'batch' and config.get('batch_size', 0) > 10000:
        validation_results["warnings"].append(
            "Large batch size may cause memory issues"
        )
    
    if pipeline_type == 'real_time' and config.get('max_latency_ms', 0) > 1000:
        validation_results["warnings"].append(
            "High latency limit may impact real-time performance"
        )
    
    return validation_results

def optimize_pipeline_performance(pipeline_name: str) -> Dict[str, Any]:
    """
    Optimize specific pipeline performance.
    
    Args:
        pipeline_name: Name of pipeline to optimize
        
    Returns:
        Optimization results
    """
    manager = get_pipeline_manager()
    return manager.optimize_pipeline_performance(pipeline_name)

def monitor_pipeline_health() -> Dict[str, Any]:
    """
    Monitor health of all pipelines.
    
    Returns:
        Health monitoring report
    """
    manager = get_pipeline_manager()
    return manager.monitor_pipeline_health()

@contextmanager
def pipeline_context(pipeline_name: str, pipeline_type: str, config: Dict[str, Any] = None):
    """
    Context manager for pipeline lifecycle management.
    
    Args:
        pipeline_name: Name of pipeline
        pipeline_type: Type of pipeline
        config: Pipeline configuration
    """
    manager = get_pipeline_manager()
    
    # Create pipeline
    pipeline = manager.create_pipeline(pipeline_type, pipeline_name, config)
    
    try:
        yield pipeline
    finally:
        # Cleanup
        manager.shutdown_pipeline(pipeline_name)

# Configuration validation schemas
PIPELINE_CONFIG_SCHEMAS = {
    'inference': {
        'required': ['model_path'],
        'optional': ['enable_caching', 'max_batch_size', 'timeout_seconds'],
        'defaults': {
            'enable_caching': True,
            'max_batch_size': 32,
            'timeout_seconds': 30.0
        }
    },
    'batch': {
        'required': ['batch_size'],
        'optional': ['max_workers', 'chunk_size', 'output_format'],
        'defaults': {
            'max_workers': 4,
            'chunk_size': 100,
            'output_format': 'json'
        }
    },
    'real_time': {
        'required': ['buffer_size'],
        'optional': ['processing_interval_ms', 'max_latency_ms', 'stream_source'],
        'defaults': {
            'processing_interval_ms': 100,
            'max_latency_ms': 500,
            'stream_source': 'kafka'
        }
    },
    'edge': {
        'required': ['memory_limit_mb'],
        'optional': ['model_compression', 'quantization_enabled', 'target_device'],
        'defaults': {
            'model_compression': 'quantization',
            'quantization_enabled': True,
            'target_device': 'cpu'
        }
    }
}

def validate_pipeline_config(pipeline_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize pipeline configuration.
    
    Args:
        pipeline_type: Type of pipeline to validate
        config: Configuration dictionary to validate
        
    Returns:
        Validated and normalized configuration
        
    Raises:
        ValueError: If configuration is invalid
    """
    if pipeline_type not in PIPELINE_CONFIG_SCHEMAS:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    schema = PIPELINE_CONFIG_SCHEMAS[pipeline_type]
    validated_config = config.copy()
    
    # Check required fields
    for field in schema['required']:
        if field not in validated_config:
            raise ValueError(f"Missing required field '{field}' for {pipeline_type} pipeline")
    
    # Apply defaults for optional fields
    for field, default_value in schema['defaults'].items():
        if field not in validated_config:
            validated_config[field] = default_value
    
    # Validate field types and values
    _validate_config_values(pipeline_type, validated_config)
    
    return validated_config

def _validate_config_values(pipeline_type: str, config: Dict[str, Any]) -> None:
    """Validate specific configuration values."""
    if pipeline_type == 'inference':
        if 'timeout_seconds' in config and config['timeout_seconds'] <= 0:
            raise ValueError("timeout_seconds must be positive")
        if 'max_batch_size' in config and config['max_batch_size'] <= 0:
            raise ValueError("max_batch_size must be positive")
    
    elif pipeline_type == 'batch':
        if 'batch_size' in config and config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")
        if 'max_workers' in config and config['max_workers'] <= 0:
            raise ValueError("max_workers must be positive")
    
    elif pipeline_type == 'real_time':
        if 'buffer_size' in config and config['buffer_size'] <= 0:
            raise ValueError("buffer_size must be positive")
        if 'processing_interval_ms' in config and config['processing_interval_ms'] <= 0:
            raise ValueError("processing_interval_ms must be positive")
    
    elif pipeline_type == 'edge':
        if 'memory_limit_mb' in config and config['memory_limit_mb'] <= 0:
            raise ValueError("memory_limit_mb must be positive")

def get_pipeline_status(pipeline_name: str) -> Dict[str, Any]:
    """
    Get the status of a pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
        
    Returns:
        Pipeline status information
    """
    manager = get_pipeline_manager()
    return manager.get_pipeline_status(pipeline_name)

def list_active_pipelines() -> List[str]:
    """
    List all active pipelines.
    
    Returns:
        List of active pipeline names
    """
    manager = get_pipeline_manager()
    return manager.list_active_pipelines()

def shutdown_all_pipelines() -> None:
    """Shutdown all active pipelines."""
    manager = get_pipeline_manager()
    manager.shutdown_all_pipelines()

def get_pipeline_metrics(pipeline_name: str) -> Dict[str, Any]:
    """
    Get performance metrics for a pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
        
    Returns:
        Pipeline performance metrics
    """
    manager = get_pipeline_manager()
    return manager.get_pipeline_metrics(pipeline_name)

def register_pipeline_callback(pipeline_name: str, event_type: str, callback: callable) -> None:
    """
    Register a callback for pipeline events.
    
    Args:
        pipeline_name: Name of the pipeline
        event_type: Type of event to listen for
        callback: Callback function to execute
    """
    manager = get_pipeline_manager()
    manager.register_callback(pipeline_name, event_type, callback)

# Exception classes
class PipelineError(Exception):
    """Base exception for pipeline operations."""
    pass

class PipelineConfigurationError(PipelineError):
    """Exception raised for pipeline configuration errors."""
    pass

class PipelineExecutionError(PipelineError):
    """Exception raised for pipeline execution errors."""
    pass

class PipelineTimeoutError(PipelineError):
    """Exception raised when pipeline operations timeout."""
    pass

# Pipeline health check utilities
def health_check_pipeline(pipeline_name: str) -> Dict[str, Any]:
    """
    Perform health check on a pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
        
    Returns:
        Health check results
    """
    try:
        status = get_pipeline_status(pipeline_name)
        metrics = get_pipeline_metrics(pipeline_name)
        
        health_status = {
            'pipeline_name': pipeline_name,
            'status': status.get('status', 'unknown'),
            'healthy': status.get('status') == 'running',
            'uptime_seconds': status.get('uptime_seconds', 0),
            'last_activity': status.get('last_activity'),
            'error_count': metrics.get('error_count', 0),
            'success_rate': metrics.get('success_rate', 0.0),
            'average_latency_ms': metrics.get('average_latency_ms', 0.0),
            'memory_usage_mb': metrics.get('memory_usage_mb', 0.0),
            'cpu_usage_percent': metrics.get('cpu_usage_percent', 0.0)
        }
        
        # Determine health based on metrics
        if health_status['error_count'] > 100:
            health_status['healthy'] = False
            health_status['health_issues'] = health_status.get('health_issues', []) + ['high_error_count']
        
        if health_status['success_rate'] < 0.95:
            health_status['healthy'] = False
            health_status['health_issues'] = health_status.get('health_issues', []) + ['low_success_rate']
        
        if health_status['average_latency_ms'] > 5000:
            health_status['healthy'] = False
            health_status['health_issues'] = health_status.get('health_issues', []) + ['high_latency']
        
        return health_status
        
    except Exception as e:
        return {
            'pipeline_name': pipeline_name,
            'status': 'error',
            'healthy': False,
            'error': str(e),
            'health_issues': ['health_check_failed']
        }

def health_check_all_pipelines() -> Dict[str, Dict[str, Any]]:
    """
    Perform health check on all active pipelines.
    
    Returns:
        Health check results for all pipelines
    """
    active_pipelines = list_active_pipelines()
    health_results = {}
    
    for pipeline_name in active_pipelines:
        health_results[pipeline_name] = health_check_pipeline(pipeline_name)
    
    return health_results

# Pipeline optimization utilities
def optimize_pipeline_config(pipeline_type: str, current_config: Dict[str, Any],
                           performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest optimizations for pipeline configuration based on performance metrics.
    
    Args:
        pipeline_type: Type of pipeline
        current_config: Current pipeline configuration
        performance_metrics: Current performance metrics
        
    Returns:
        Optimized configuration suggestions
    """
    optimized_config = current_config.copy()
    optimization_suggestions = []
    
    if pipeline_type == 'batch':
        # Optimize batch processing
        if performance_metrics.get('cpu_usage_percent', 0) < 50:
            # CPU underutilized, increase batch size or workers
            if 'batch_size' in optimized_config:
                new_batch_size = min(optimized_config['batch_size'] * 2, 128)
                optimized_config['batch_size'] = new_batch_size
                optimization_suggestions.append(f"Increased batch_size to {new_batch_size}")
        
        if performance_metrics.get('memory_usage_mb', 0) < 1000:
            # Memory underutilized, increase workers
            if 'max_workers' in optimized_config:
                new_workers = min(optimized_config['max_workers'] + 2, 16)
                optimized_config['max_workers'] = new_workers
                optimization_suggestions.append(f"Increased max_workers to {new_workers}")
    
    elif pipeline_type == 'real_time':
        # Optimize real-time processing
        if performance_metrics.get('average_latency_ms', 0) > 1000:
            # High latency, reduce buffer size or processing interval
            if 'buffer_size' in optimized_config:
                new_buffer_size = max(optimized_config['buffer_size'] // 2, 10)
                optimized_config['buffer_size'] = new_buffer_size
                optimization_suggestions.append(f"Reduced buffer_size to {new_buffer_size}")
        
        if performance_metrics.get('throughput_per_second', 0) < 100:
            # Low throughput, increase processing interval
            if 'processing_interval_ms' in optimized_config:
                new_interval = max(optimized_config['processing_interval_ms'] // 2, 10)
                optimized_config['processing_interval_ms'] = new_interval
                optimization_suggestions.append(f"Reduced processing_interval_ms to {new_interval}")
    
    optimized_config['optimization_suggestions'] = optimization_suggestions
    return optimized_config

# Pipeline monitoring and alerting
def setup_pipeline_monitoring(pipeline_name: str, alert_config: Dict[str, Any]) -> None:
    """
    Set up monitoring and alerting for a pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
        alert_config: Alert configuration
    """
    manager = get_pipeline_manager()
    
    # Register monitoring callbacks
    if alert_config.get('enable_latency_alerts', False):
        def latency_alert_callback(metrics):
            if metrics.get('average_latency_ms', 0) > alert_config.get('latency_threshold_ms', 5000):
                logger.warning(f"High latency detected in pipeline {pipeline_name}: {metrics['average_latency_ms']}ms")
        
        register_pipeline_callback(pipeline_name, 'metrics_updated', latency_alert_callback)
    
    if alert_config.get('enable_error_alerts', False):
        def error_alert_callback(metrics):
            if metrics.get('error_rate', 0) > alert_config.get('error_rate_threshold', 0.1):
                logger.error(f"High error rate detected in pipeline {pipeline_name}: {metrics['error_rate']}")
        
        register_pipeline_callback(pipeline_name, 'metrics_updated', error_alert_callback)
    
    if alert_config.get('enable_resource_alerts', False):
        def resource_alert_callback(metrics):
            if metrics.get('memory_usage_mb', 0) > alert_config.get('memory_threshold_mb', 2000):
                logger.warning(f"High memory usage in pipeline {pipeline_name}: {metrics['memory_usage_mb']}MB")
        
        register_pipeline_callback(pipeline_name, 'metrics_updated', resource_alert_callback)

# Advanced pipeline utilities
def create_pipeline_from_template(template_name: str, pipeline_name: str, 
                                template_params: Dict[str, Any]) -> str:
    """
    Create a pipeline from a predefined template.
    
    Args:
        template_name: Name of the template
        pipeline_name: Name for the new pipeline
        template_params: Parameters to customize the template
        
    Returns:
        Created pipeline name
    """
    # Template configurations
    templates = {
        'basic_inference': {
            'pipeline_type': 'inference',
            'config': {
                'model_path': template_params.get('model_path', ''),
                'enable_caching': True,
                'max_batch_size': 32,
                'timeout_seconds': 30.0
            }
        },
        'high_throughput_batch': {
            'pipeline_type': 'batch',
            'config': {
                'batch_size': 128,
                'max_workers': 8,
                'chunk_size': 1000,
                'output_format': 'json'
            }
        },
        'low_latency_realtime': {
            'pipeline_type': 'real_time',
            'config': {
                'buffer_size': 50,
                'processing_interval_ms': 50,
                'max_latency_ms': 200,
                'stream_source': 'kafka'
            }
        },
        'edge_optimized': {
            'pipeline_type': 'edge',
            'config': {
                'memory_limit_mb': 256,
                'model_compression': 'quantization',
                'quantization_enabled': True,
                'target_device': 'cpu'
            }
        }
    }
    
    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}")
    
    template = templates[template_name]
    
    # Merge template params with template config
    config = template['config'].copy()
    config.update(template_params)
    
    # Create pipeline
    return create_pipeline(template['pipeline_type'], pipeline_name, config)

def clone_pipeline(source_pipeline_name: str, target_pipeline_name: str,
                  config_overrides: Optional[Dict[str, Any]] = None) -> str:
    """
    Clone an existing pipeline with optional configuration overrides.
    
    Args:
        source_pipeline_name: Name of the source pipeline
        target_pipeline_name: Name for the cloned pipeline
        config_overrides: Optional configuration overrides
        
    Returns:
        Cloned pipeline name
    """
    manager = get_pipeline_manager()
    
    # Get source pipeline configuration
    source_config = manager.get_pipeline_config(source_pipeline_name)
    source_type = manager.get_pipeline_type(source_pipeline_name)
    
    # Apply overrides if provided
    if config_overrides:
        source_config.update(config_overrides)
    
    # Create cloned pipeline
    return create_pipeline(source_type, target_pipeline_name, source_config)

# Pipeline versioning and rollback
def create_pipeline_snapshot(pipeline_name: str, snapshot_name: str) -> str:
    """
    Create a snapshot of a pipeline configuration.
    
    Args:
        pipeline_name: Name of the pipeline
        snapshot_name: Name for the snapshot
        
    Returns:
        Snapshot identifier
    """
    manager = get_pipeline_manager()
    return manager.create_pipeline_snapshot(pipeline_name, snapshot_name)

def rollback_pipeline(pipeline_name: str, snapshot_name: str) -> None:
    """
    Rollback a pipeline to a previous snapshot.
    
    Args:
        pipeline_name: Name of the pipeline
        snapshot_name: Name of the snapshot to rollback to
    """
    manager = get_pipeline_manager()
    manager.rollback_pipeline(pipeline_name, snapshot_name)

def list_pipeline_snapshots(pipeline_name: str) -> List[Dict[str, Any]]:
    """
    List all snapshots for a pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
        
    Returns:
        List of snapshot information
    """
    manager = get_pipeline_manager()
    return manager.list_pipeline_snapshots(pipeline_name)

# Module cleanup
def cleanup_module() -> None:
    """Cleanup module resources."""
    global _pipeline_manager
    if _pipeline_manager:
        _pipeline_manager.shutdown_all_pipelines()
        _pipeline_manager = None

# Register cleanup handler
import atexit
atexit.register(cleanup_module)

# Module initialization
logger.info(f"BatteryMind Inference Pipelines module v{__version__} initialized")
logger.info(f"Available pipeline types: {list(PIPELINE_CONFIG_SCHEMAS.keys())}")

