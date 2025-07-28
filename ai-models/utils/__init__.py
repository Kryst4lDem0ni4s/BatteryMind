"""
BatteryMind - Utilities Module
Comprehensive utility functions and helpers for the BatteryMind AI-powered
autonomous battery management system.

Features:
- Data processing and transformation utilities
- Model management and deployment helpers
- Visualization and plotting utilities
- Logging and monitoring helpers
- Configuration management utilities
- File and I/O operations
- AWS cloud service integrations
- Security and encryption utilities

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Import all utility modules for easy access
from .data_utils import *
from .model_utils import *
from .visualization import *
from .logging_utils import *
from .config_parser import *
from .file_handlers import *
from .aws_helpers import *

# Version and metadata
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"

# Module-level configuration
UTILS_CONFIG = {
    'default_encoding': 'utf-8',
    'default_timezone': 'UTC',
    'numeric_precision': 6,
    'memory_threshold_mb': 1024,
    'batch_size_default': 1000,
    'timeout_default': 30,
    'retry_attempts': 3,
    'cache_ttl_seconds': 3600
}

# Export all utility classes and functions
__all__ = [
    # Data utilities
    'DataProcessor',
    'DataValidator',
    'DataTransformer',
    'TimeSeriesProcessor',
    'FeatureEngineer',
    'DataAggregator',
    'BatteryDataProcessor',
    'FleetDataProcessor',
    
    # Model utilities
    'ModelManager',
    'ModelValidator',
    'ModelDeployer',
    'PerformanceEvaluator',
    'HyperparameterTuner',
    'ModelRegistry',
    'InferenceEngine',
    'ModelMonitor',
    
    # Visualization utilities
    'BatteryVisualizer',
    'FleetVisualizer',
    'AIInsightsVisualizer',
    'CircularEconomyVisualizer',
    'RealTimeChartManager',
    'DashboardGenerator',
    
    # Logging utilities
    'setup_logger',
    'get_logger',
    'log_performance',
    'log_error',
    'AuditLogger',
    'MetricsLogger',
    
    # Configuration utilities
    'ConfigManager',
    'EnvironmentManager',
    'SecretManager',
    'FeatureFlagManager',
    'load_config',
    'validate_config',
    
    # File handling utilities
    'FileManager',
    'S3FileHandler',
    'LocalFileHandler',
    'DataExporter',
    'DataImporter',
    'BackupManager',
    
    # AWS helpers
    'AWSManager',
    'IoTCoreManager',
    'SageMakerManager',
    'DynamoDBManager',
    'KinesisManager',
    'CloudWatchManager',
    
    # Core utility functions
    'sanitize_input',
    'validate_data',
    'format_timestamp',
    'calculate_metrics',
    'optimize_memory',
    'retry_on_failure',
    'cache_result',
    'async_executor',
    
    # Constants and configurations
    'UTILS_CONFIG',
    'BATTERY_CONSTANTS',
    'FLEET_CONSTANTS',
    'AI_MODEL_CONSTANTS',
    'BLOCKCHAIN_CONSTANTS'
]

# Battery-specific constants
BATTERY_CONSTANTS = {
    'soh_range': (0, 100),
    'soc_range': (0, 100),
    'voltage_range': (2.5, 4.5),
    'current_range': (-200, 200),
    'temperature_range': (-40, 85),
    'cycle_count_max': 10000,
    'degradation_threshold': 80,
    'critical_temperature': 60,
    'nominal_capacity': 100,  # kWh
    'chemistry_types': ['NMC', 'LFP', 'NCA', 'LTO'],
    'form_factors': ['cylindrical', 'prismatic', 'pouch']
}

# Fleet management constants
FLEET_CONSTANTS = {
    'max_fleet_size': 10000,
    'vehicle_types': ['passenger', 'commercial', 'bus', 'truck'],
    'charging_types': ['slow', 'fast', 'rapid', 'ultra_rapid'],
    'efficiency_range': (0.15, 0.35),  # kWh/km
    'utilization_target': 85,  # percentage
    'maintenance_intervals': {
        'basic': 30,  # days
        'intermediate': 90,
        'comprehensive': 365
    }
}

# AI model constants
AI_MODEL_CONSTANTS = {
    'model_types': ['transformer', 'lstm', 'cnn', 'ensemble'],
    'inference_timeout': 5,  # seconds
    'batch_sizes': {
        'small': 32,
        'medium': 128,
        'large': 512
    },
    'accuracy_threshold': 0.95,
    'drift_threshold': 0.1,
    'model_formats': ['pkl', 'h5', 'onnx', 'tflite'],
    'deployment_modes': ['real_time', 'batch', 'edge']
}

# Blockchain constants
BLOCKCHAIN_CONSTANTS = {
    'networks': ['ethereum', 'polygon', 'binance'],
    'gas_limits': {
        'simple': 21000,
        'contract': 200000,
        'complex': 500000
    },
    'confirmation_blocks': 12,
    'retry_attempts': 3,
    'timeout_seconds': 30
}

# Initialize logging for the utils module
logger = logging.getLogger(__name__)

def initialize_utils_module():
    """Initialize the utilities module with default configurations."""
    logger.info("Initializing BatteryMind Utils Module v%s", __version__)
    logger.info("Available utility modules: %s", len(__all__))
    logger.info("Configuration loaded: %s", UTILS_CONFIG)

def get_module_info() -> Dict[str, Any]:
    """Get information about the utils module."""
    return {
        'version': __version__,
        'author': __author__,
        'config': UTILS_CONFIG,
        'available_utilities': len(__all__),
        'battery_constants': BATTERY_CONSTANTS,
        'fleet_constants': FLEET_CONSTANTS,
        'ai_constants': AI_MODEL_CONSTANTS,
        'blockchain_constants': BLOCKCHAIN_CONSTANTS
    }

# Core utility functions
def sanitize_input(data: Any, input_type: str = 'string') -> Any:
    """
    Sanitize input data based on type.
    
    Args:
        data: Input data to sanitize
        input_type: Type of data (string, numeric, boolean)
        
    Returns:
        Sanitized data
    """
    try:
        if input_type == 'string':
            if isinstance(data, str):
                # Remove potentially harmful characters
                import re
                sanitized = re.sub(r'[<>"\'\&\;]', '', data)
                return sanitized.strip()
            return str(data)
        
        elif input_type == 'numeric':
            if isinstance(data, (int, float)):
                return data
            try:
                return float(data)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert {data} to numeric")
                return 0.0
        
        elif input_type == 'boolean':
            if isinstance(data, bool):
                return data
            if isinstance(data, str):
                return data.lower() in ['true', '1', 'yes', 'on']
            return bool(data)
        
        else:
            return data
            
    except Exception as e:
        logger.error(f"Error sanitizing input: {e}")
        return None

def validate_data(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate data against a schema.
    
    Args:
        data: Data to validate
        schema: Validation schema
        
    Returns:
        Validation result with errors and warnings
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'sanitized_data': {}
    }
    
    try:
        for field, rules in schema.items():
            if field not in data:
                if rules.get('required', False):
                    result['errors'].append(f"Required field '{field}' is missing")
                    result['valid'] = False
                continue
            
            value = data[field]
            field_type = rules.get('type', 'string')
            
            # Sanitize the value
            sanitized_value = sanitize_input(value, field_type)
            result['sanitized_data'][field] = sanitized_value
            
            # Type validation
            if field_type == 'numeric':
                min_val = rules.get('min')
                max_val = rules.get('max')
                
                if min_val is not None and sanitized_value < min_val:
                    result['errors'].append(f"Field '{field}' below minimum: {min_val}")
                    result['valid'] = False
                
                if max_val is not None and sanitized_value > max_val:
                    result['errors'].append(f"Field '{field}' above maximum: {max_val}")
                    result['valid'] = False
            
            elif field_type == 'string':
                min_len = rules.get('min_length', 0)
                max_len = rules.get('max_length', 1000)
                
                if len(sanitized_value) < min_len:
                    result['errors'].append(f"Field '{field}' too short: minimum {min_len}")
                    result['valid'] = False
                
                if len(sanitized_value) > max_len:
                    result['errors'].append(f"Field '{field}' too long: maximum {max_len}")
                    result['valid'] = False
        
        return result
        
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        return {
            'valid': False,
            'errors': [f"Validation error: {str(e)}"],
            'warnings': [],
            'sanitized_data': {}
        }

def format_timestamp(timestamp: Any, format_type: str = 'iso') -> str:
    """
    Format timestamp to specified format.
    
    Args:
        timestamp: Timestamp to format
        format_type: Format type (iso, unix, human)
        
    Returns:
        Formatted timestamp string
    """
    try:
        from datetime import datetime
        import pandas as pd
        
        # Convert to datetime if needed
        if isinstance(timestamp, str):
            dt = pd.to_datetime(timestamp)
        elif isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, datetime):
            dt = timestamp
        else:
            dt = datetime.now()
        
        if format_type == 'iso':
            return dt.isoformat()
        elif format_type == 'unix':
            return str(int(dt.timestamp()))
        elif format_type == 'human':
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return str(dt)
            
    except Exception as e:
        logger.error(f"Error formatting timestamp: {e}")
        return str(timestamp)

def calculate_metrics(data: List[float], metrics: List[str] = None) -> Dict[str, float]:
    """
    Calculate statistical metrics for data.
    
    Args:
        data: List of numeric values
        metrics: List of metrics to calculate
        
    Returns:
        Dictionary of calculated metrics
    """
    try:
        import numpy as np
        
        if not data:
            return {}
        
        if metrics is None:
            metrics = ['mean', 'std', 'min', 'max', 'median']
        
        result = {}
        
        for metric in metrics:
            if metric == 'mean':
                result['mean'] = np.mean(data)
            elif metric == 'std':
                result['std'] = np.std(data)
            elif metric == 'min':
                result['min'] = np.min(data)
            elif metric == 'max':
                result['max'] = np.max(data)
            elif metric == 'median':
                result['median'] = np.median(data)
            elif metric == 'percentile_95':
                result['percentile_95'] = np.percentile(data, 95)
            elif metric == 'percentile_99':
                result['percentile_99'] = np.percentile(data, 99)
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {}

def optimize_memory():
    """Optimize memory usage by cleaning up unused objects."""
    try:
        import gc
        import psutil
        import os
        
        # Get memory usage before cleanup
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Force garbage collection
        gc.collect()
        
        # Get memory usage after cleanup
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_freed = memory_before - memory_after
        logger.info(f"Memory optimization: freed {memory_freed:.2f} MB")
        
        return {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_freed_mb': memory_freed
        }
        
    except Exception as e:
        logger.error(f"Error optimizing memory: {e}")
        return {}

def retry_on_failure(func, max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry function on failure.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        Decorated function
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                    raise
                else:
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}): {e}")
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
        
    return wrapper

def cache_result(ttl_seconds: int = 3600):
    """
    Decorator to cache function results.
    
    Args:
        ttl_seconds: Time to live for cached results
        
    Returns:
        Decorated function
    """
    import time
    from functools import wraps
    
    cache = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Check if result is cached and not expired
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl_seconds:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            
            return result
        
        return wrapper
    return decorator

async def async_executor(func, *args, **kwargs):
    """
    Execute function asynchronously.
    
    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    try:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, func, *args, **kwargs)
            return result
            
    except Exception as e:
        logger.error(f"Error in async execution: {e}")
        raise

# Initialize the module when imported
initialize_utils_module()

# Module information
logger.info(f"BatteryMind Utils Module v{__version__} loaded successfully")
logger.info(f"Available utilities: {', '.join(__all__[:10])}...")  # Show first 10
