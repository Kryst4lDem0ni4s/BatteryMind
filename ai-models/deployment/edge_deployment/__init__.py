"""
BatteryMind Edge Deployment Module

This module provides comprehensive edge deployment capabilities for battery management
systems, including model optimization, quantization, pruning, and edge runtime
environments. The module is designed to deploy AI models on resource-constrained
edge devices with strict latency and memory requirements.

Key Components:
- Model Optimization: Advanced optimization techniques for edge deployment
- Quantization: Model quantization for reduced memory footprint
- Pruning: Neural network pruning for faster inference
- Edge Runtime: Lightweight runtime environment for edge devices

Features:
- Memory footprint optimization (<100MB)
- Real-time inference (<10ms latency)
- Cross-platform compatibility (ARM, x86, GPU)
- Offline operation capabilities
- Safety-critical deployment support

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import platform
import psutil

# Core imports
import numpy as np
import torch
import onnx
import onnxruntime as ort

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"
__license__ = "Proprietary"

# Edge deployment enums
class DeviceType(Enum):
    """Supported edge device types."""
    RASPBERRY_PI = "raspberry_pi"
    JETSON_NANO = "jetson_nano"
    JETSON_XAVIER = "jetson_xavier"
    INTEL_NUC = "intel_nuc"
    ARM_CORTEX = "arm_cortex"
    GENERIC_X86 = "generic_x86"
    CUSTOM = "custom"

class OptimizationLevel(Enum):
    """Model optimization levels."""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

class PrecisionType(Enum):
    """Precision types for quantization."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    MIXED = "mixed"

# Module metadata
__all__ = [
    # Enums
    'DeviceType',
    'OptimizationLevel',
    'PrecisionType',
    
    # Main classes
    'ModelOptimizer',
    'Quantizer',
    'Pruner',
    'EdgeRuntime',
    
    # Configuration
    'EdgeDeploymentConfig',
    'DeviceProfile',
    'OptimizationProfile',
    
    # Utilities
    'EdgeModelRegistry',
    'PerformanceProfiler',
    'CompatibilityChecker',
    'DeviceDetector',
    
    # Factory functions
    'create_edge_optimizer',
    'create_edge_runtime',
    'detect_device_capabilities',
    'optimize_for_edge'
]

# Import main classes with error handling
try:
    from .model_optimization import ModelOptimizer
    logger.info("Successfully imported ModelOptimizer")
except ImportError as e:
    logger.error(f"Failed to import ModelOptimizer: {e}")
    class ModelOptimizer:
        def __init__(self):
            logger.warning("ModelOptimizer placeholder initialized")

try:
    from .quantization import Quantizer
    logger.info("Successfully imported Quantizer")
except ImportError as e:
    logger.error(f"Failed to import Quantizer: {e}")
    class Quantizer:
        def __init__(self):
            logger.warning("Quantizer placeholder initialized")

try:
    from .pruning import Pruner
    logger.info("Successfully imported Pruner")
except ImportError as e:
    logger.error(f"Failed to import Pruner: {e}")
    class Pruner:
        def __init__(self):
            logger.warning("Pruner placeholder initialized")

try:
    from .edge_runtime import EdgeRuntime
    logger.info("Successfully imported EdgeRuntime")
except ImportError as e:
    logger.error(f"Failed to import EdgeRuntime: {e}")
    class EdgeRuntime:
        def __init__(self):
            logger.warning("EdgeRuntime placeholder initialized")

# Configuration classes
@dataclass
class DeviceProfile:
    """Profile for edge device capabilities."""
    
    device_type: DeviceType
    cpu_cores: int
    memory_mb: int
    storage_gb: float
    has_gpu: bool = False
    gpu_memory_mb: int = 0
    supports_fp16: bool = False
    supports_int8: bool = True
    max_power_watts: float = 25.0
    operating_temp_range: Tuple[float, float] = (-10.0, 70.0)
    
    @classmethod
    def auto_detect(cls) -> 'DeviceProfile':
        """Auto-detect device capabilities."""
        # Get system information
        cpu_cores = psutil.cpu_count(logical=False)
        memory_mb = int(psutil.virtual_memory().total / 1024 / 1024)
        storage_gb = psutil.disk_usage('/').total / 1024 / 1024 / 1024
        
        # Detect device type based on platform
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if 'arm' in machine:
            device_type = DeviceType.ARM_CORTEX
        elif system == 'linux' and 'tegra' in platform.platform().lower():
            device_type = DeviceType.JETSON_NANO  # Default Jetson
        else:
            device_type = DeviceType.GENERIC_X86
        
        # Check for GPU
        has_gpu = torch.cuda.is_available()
        gpu_memory_mb = 0
        if has_gpu:
            gpu_memory_mb = int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
        
        return cls(
            device_type=device_type,
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            storage_gb=storage_gb,
            has_gpu=has_gpu,
            gpu_memory_mb=gpu_memory_mb,
            supports_fp16=has_gpu,  # Assume FP16 support with GPU
            supports_int8=True
        )

@dataclass
class OptimizationProfile:
    """Profile for optimization settings."""
    
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    target_precision: PrecisionType = PrecisionType.FP32
    max_memory_mb: int = 100
    max_latency_ms: float = 10.0
    target_accuracy_retention: float = 0.95
    enable_pruning: bool = True
    pruning_ratio: float = 0.3
    enable_quantization: bool = True
    enable_fusion: bool = True
    batch_size: int = 1
    sequence_length: int = 100

@dataclass
class EdgeDeploymentConfig:
    """Configuration for edge deployment."""
    
    device_profile: DeviceProfile = field(default_factory=DeviceProfile.auto_detect)
    optimization_profile: OptimizationProfile = field(default_factory=OptimizationProfile)
    
    # Model paths
    source_model_path: str = ""
    output_model_path: str = ""
    calibration_data_path: str = ""
    
    # Deployment settings
    runtime_engine: str = "onnxruntime"  # onnxruntime, tensorrt, openvino
    enable_monitoring: bool = True
    safety_checks: bool = True
    offline_mode: bool = True
    
    # Performance targets
    target_fps: float = 10.0
    target_power_watts: float = 5.0
    target_temperature_celsius: float = 60.0
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EdgeDeploymentConfig':
        """Create config from dictionary."""
        if 'device_profile' in config_dict and isinstance(config_dict['device_profile'], dict):
            config_dict['device_profile'] = DeviceProfile(**config_dict['device_profile'])
        
        if 'optimization_profile' in config_dict and isinstance(config_dict['optimization_profile'], dict):
            config_dict['optimization_profile'] = OptimizationProfile(**config_dict['optimization_profile'])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result

# Edge model registry
class EdgeModelRegistry:
    """Registry for edge-optimized models."""
    
    def __init__(self):
        """Initialize edge model registry."""
        self.models = {}
        self.metadata = {}
        self.performance_stats = {}
        
    def register_model(self, 
                      name: str, 
                      model_path: str,
                      device_profile: DeviceProfile,
                      optimization_profile: OptimizationProfile,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register an edge-optimized model."""
        
        self.models[name] = {
            'model_path': model_path,
            'device_profile': device_profile,
            'optimization_profile': optimization_profile,
            'registered_at': datetime.now()
        }
        
        self.metadata[name] = metadata or {}
        
        logger.info(f"Registered edge model: {name}")
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get model information."""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.models.keys())
    
    def find_compatible_models(self, device_profile: DeviceProfile) -> List[str]:
        """Find models compatible with device profile."""
        compatible = []
        
        for name, info in self.models.items():
            model_device = info['device_profile']
            
            # Check basic compatibility
            if (model_device.memory_mb <= device_profile.memory_mb and
                model_device.cpu_cores <= device_profile.cpu_cores):
                compatible.append(name)
        
        return compatible

# Performance profiler
class PerformanceProfiler:
    """Profiler for edge model performance."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.metrics = {}
        
    def profile_model(self, 
                     model_path: str,
                     input_shape: Tuple[int, ...],
                     device: str = "cpu",
                     num_iterations: int = 100) -> Dict[str, float]:
        """Profile model performance."""
        
        try:
            # Load model based on format
            if model_path.endswith('.onnx'):
                session = ort.InferenceSession(model_path)
                input_name = session.get_inputs()[0].name
                
                # Create random input
                dummy_input = np.random.randn(*input_shape).astype(np.float32)
                
                # Warm up
                for _ in range(10):
                    _ = session.run(None, {input_name: dummy_input})
                
                # Measure performance
                import time
                start_time = time.time()
                
                for _ in range(num_iterations):
                    _ = session.run(None, {input_name: dummy_input})
                
                end_time = time.time()
                
                avg_latency = (end_time - start_time) / num_iterations * 1000  # ms
                throughput = 1000.0 / avg_latency  # FPS
                
                # Memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                metrics = {
                    'avg_latency_ms': avg_latency,
                    'throughput_fps': throughput,
                    'memory_usage_mb': memory_mb,
                    'model_size_mb': Path(model_path).stat().st_size / 1024 / 1024
                }
                
                self.metrics[model_path] = metrics
                return metrics
                
            else:
                logger.error(f"Unsupported model format: {model_path}")
                return {}
                
        except Exception as e:
            logger.error(f"Error profiling model {model_path}: {e}")
            return {}
    
    def get_metrics(self, model_path: str) -> Dict[str, float]:
        """Get cached metrics for model."""
        return self.metrics.get(model_path, {})

# Compatibility checker
class CompatibilityChecker:
    """Checker for device and model compatibility."""
    
    @staticmethod
    def check_model_compatibility(model_path: str, 
                                 device_profile: DeviceProfile) -> Dict[str, Any]:
        """Check if model is compatible with device."""
        
        compatibility = {
            'compatible': True,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check model format
            if model_path.endswith('.onnx'):
                model = onnx.load(model_path)
                
                # Estimate memory requirements
                model_size_mb = Path(model_path).stat().st_size / 1024 / 1024
                
                # Check memory constraints
                if model_size_mb > device_profile.memory_mb * 0.8:  # 80% of available memory
                    compatibility['compatible'] = False
                    compatibility['issues'].append(f"Model size ({model_size_mb:.1f}MB) exceeds device memory")
                    compatibility['recommendations'].append("Consider model quantization or pruning")
                
                # Check precision support
                graph = model.graph
                for node in graph.node:
                    if any(attr.name == 'dtype' for attr in node.attribute):
                        # Check if device supports required precision
                        pass
                
            else:
                compatibility['issues'].append(f"Unsupported model format: {Path(model_path).suffix}")
            
        except Exception as e:
            compatibility['compatible'] = False
            compatibility['issues'].append(f"Error checking compatibility: {e}")
        
        return compatibility
    
    @staticmethod
    def check_runtime_compatibility(runtime_engine: str,
                                   device_profile: DeviceProfile) -> bool:
        """Check if runtime engine is compatible with device."""
        
        if runtime_engine == "onnxruntime":
            return True  # ONNX Runtime supports most devices
        
        elif runtime_engine == "tensorrt":
            return device_profile.has_gpu and "nvidia" in platform.processor().lower()
        
        elif runtime_engine == "openvino":
            return device_profile.device_type in [DeviceType.INTEL_NUC, DeviceType.GENERIC_X86]
        
        return False

# Device detector
class DeviceDetector:
    """Detector for edge device capabilities."""
    
    @staticmethod
    def detect_device_type() -> DeviceType:
        """Detect device type."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Check for specific devices
        if 'arm' in machine:
            # Check for Raspberry Pi
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    if 'raspberry pi' in cpuinfo.lower():
                        return DeviceType.RASPBERRY_PI
            except:
                pass
            
            # Check for Jetson
            if 'tegra' in platform.platform().lower():
                return DeviceType.JETSON_NANO
            
            return DeviceType.ARM_CORTEX
        
        elif system == 'linux':
            # Check for Jetson devices
            if 'tegra' in platform.platform().lower():
                return DeviceType.JETSON_NANO
        
        return DeviceType.GENERIC_X86
    
    @staticmethod
    def get_hardware_info() -> Dict[str, Any]:
        """Get detailed hardware information."""
        return {
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_total': psutil.disk_usage('/').total,
            'disk_free': psutil.disk_usage('/').free,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'machine': platform.machine(),
            'system': platform.system()
        }

# Factory functions
def create_edge_optimizer(config: Optional[EdgeDeploymentConfig] = None) -> ModelOptimizer:
    """Create edge model optimizer."""
    if config is None:
        config = EdgeDeploymentConfig()
    
    return ModelOptimizer(config)

def create_edge_runtime(config: Optional[EdgeDeploymentConfig] = None) -> EdgeRuntime:
    """Create edge runtime."""
    if config is None:
        config = EdgeDeploymentConfig()
    
    return EdgeRuntime(config)

def detect_device_capabilities() -> DeviceProfile:
    """Detect current device capabilities."""
    return DeviceProfile.auto_detect()

def optimize_for_edge(model_path: str,
                     output_path: str,
                     device_profile: Optional[DeviceProfile] = None,
                     optimization_level: OptimizationLevel = OptimizationLevel.STANDARD) -> str:
    """
    Optimize model for edge deployment.
    
    Args:
        model_path: Path to source model
        output_path: Path for optimized model
        device_profile: Target device profile
        optimization_level: Level of optimization
        
    Returns:
        Path to optimized model
    """
    if device_profile is None:
        device_profile = DeviceProfile.auto_detect()
    
    # Create optimization profile
    opt_profile = OptimizationProfile(optimization_level=optimization_level)
    
    # Create configuration
    config = EdgeDeploymentConfig(
        device_profile=device_profile,
        optimization_profile=opt_profile,
        source_model_path=model_path,
        output_model_path=output_path
    )
    
    # Create optimizer and optimize
    optimizer = ModelOptimizer(config)
    optimized_path = optimizer.optimize()
    
    return optimized_path

# Module initialization
def initialize_edge_deployment_module(config_path: Optional[str] = None) -> None:
    """Initialize edge deployment module."""
    
    # Detect device capabilities
    device_profile = DeviceProfile.auto_detect()
    
    logger.info("BatteryMind Edge Deployment Module initialized")
    logger.info(f"Device Type: {device_profile.device_type.value}")
    logger.info(f"CPU Cores: {device_profile.cpu_cores}")
    logger.info(f"Memory: {device_profile.memory_mb}MB")
    logger.info(f"GPU Available: {device_profile.has_gpu}")
    
    if device_profile.has_gpu:
        logger.info(f"GPU Memory: {device_profile.gpu_memory_mb}MB")

# Global objects
_global_registry = EdgeModelRegistry()
_global_profiler = PerformanceProfiler()

def get_global_registry() -> EdgeModelRegistry:
    """Get global edge model registry."""
    return _global_registry

def get_global_profiler() -> PerformanceProfiler:
    """Get global performance profiler."""
    return _global_profiler

# Health check
def health_check() -> Dict[str, Any]:
    """Perform health check on edge deployment module."""
    try:
        device_profile = DeviceProfile.auto_detect()
        hardware_info = DeviceDetector.get_hardware_info()
        
        return {
            'status': 'healthy',
            'device_profile': device_profile.__dict__,
            'hardware_info': hardware_info,
            'registered_models': len(_global_registry.list_models()),
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# Initialize module
try:
    initialize_edge_deployment_module()
except Exception as e:
    logger.error(f"Failed to initialize edge deployment module: {e}")
    warnings.warn(f"Edge deployment module initialization failed: {e}")
