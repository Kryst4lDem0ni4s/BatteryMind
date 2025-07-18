"""
BatteryMind Edge Runtime

High-performance edge computing runtime for deploying optimized battery management
AI models on resource-constrained devices. This runtime provides efficient model
execution, resource management, and real-time inference capabilities.

Features:
- Multi-model inference engine
- Dynamic resource allocation
- Hardware acceleration (GPU/TPU when available)
- Model quantization and optimization
- Real-time telemetry processing
- Edge-specific caching and batching
- Fault tolerance and recovery

Author: BatteryMind Development Team
Version: 1.0.0
"""

import os
import sys
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
import queue
from concurrent.futures import ThreadPoolExecutor, Future
import psutil
import gc
from abc import ABC, abstractmethod

# Core ML libraries
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# BatteryMind imports
from .model_optimization import ModelOptimizer
from .quantization import ModelQuantizer
from .pruning import ModelPruner
from ..aws_sagemaker.monitoring import ModelMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EdgeConfig:
    """Configuration for edge runtime deployment."""
    max_models: int = 5
    max_memory_mb: int = 512
    max_cpu_percent: float = 80.0
    inference_timeout_ms: int = 100
    batch_size: int = 8
    cache_size: int = 1000
    enable_gpu: bool = True
    enable_quantization: bool = True
    model_warm_up: bool = True
    telemetry_buffer_size: int = 100
    heartbeat_interval_sec: int = 30
    log_level: str = "INFO"

@dataclass
class ModelMetadata:
    """Metadata for deployed models."""
    model_id: str
    model_type: str
    version: str
    input_shape: List[int]
    output_shape: List[int]
    model_size_mb: float
    optimization_level: str
    hardware_requirements: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_updated: str = ""

@dataclass
class InferenceRequest:
    """Request for model inference."""
    request_id: str
    model_id: str
    input_data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass
class InferenceResult:
    """Result from model inference."""
    request_id: str
    model_id: str
    prediction: np.ndarray
    confidence: float
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None

class ModelRuntime(ABC):
    """Abstract base class for model runtimes."""
    
    @abstractmethod
    def load_model(self, model_path: str, metadata: ModelMetadata) -> bool:
        """Load model from path."""
        pass
    
    @abstractmethod
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data."""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload model from memory."""
        pass

class ONNXRuntime(ModelRuntime):
    """ONNX Runtime implementation."""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.session = None
        self.metadata = None
        
    def load_model(self, model_path: str, metadata: ModelMetadata) -> bool:
        """Load ONNX model."""
        try:
            # Configure providers based on hardware availability
            providers = ['CPUExecutionProvider']
            if self.config.enable_gpu:
                providers.insert(0, 'CUDAExecutionProvider')
            
            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = min(4, psutil.cpu_count())
            sess_options.inter_op_num_threads = min(2, psutil.cpu_count())
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                model_path,
                sess_options,
                providers=providers
            )
            self.metadata = metadata
            
            logger.info(f"Loaded ONNX model: {metadata.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return False
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        # Get input name from session
        input_name = self.session.get_inputs()[0].name
        
        # Run inference
        result = self.session.run(None, {input_name: input_data})
        return result[0]
    
    def get_memory_usage(self) -> float:
        """Get approximate memory usage."""
        if self.metadata:
            return self.metadata.model_size_mb
        return 0.0
    
    def unload_model(self) -> None:
        """Unload ONNX model."""
        if self.session:
            del self.session
            self.session = None
            self.metadata = None
            gc.collect()

class TensorFlowLiteRuntime(ModelRuntime):
    """TensorFlow Lite runtime implementation."""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.interpreter = None
        self.metadata = None
        
    def load_model(self, model_path: str, metadata: ModelMetadata) -> bool:
        """Load TensorFlow Lite model."""
        try:
            self.interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=min(4, psutil.cpu_count())
            )
            self.interpreter.allocate_tensors()
            self.metadata = metadata
            
            logger.info(f"Loaded TFLite model: {metadata.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            return False
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run TensorFlow Lite inference."""
        if self.interpreter is None:
            raise RuntimeError("Model not loaded")
        
        # Get input/output details
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        # Set input tensor
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(output_details[0]['index'])
        return output
    
    def get_memory_usage(self) -> float:
        """Get approximate memory usage."""
        if self.metadata:
            return self.metadata.model_size_mb
        return 0.0
    
    def unload_model(self) -> None:
        """Unload TensorFlow Lite model."""
        if self.interpreter:
            del self.interpreter
            self.interpreter = None
            self.metadata = None
            gc.collect()

class PyTorchRuntime(ModelRuntime):
    """PyTorch runtime implementation."""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.model = None
        self.metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.enable_gpu else 'cpu')
        
    def load_model(self, model_path: str, metadata: ModelMetadata) -> bool:
        """Load PyTorch model."""
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            self.metadata = metadata
            
            logger.info(f"Loaded PyTorch model: {metadata.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            return False
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run PyTorch inference."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_data).to(self.device)
            output = self.model(input_tensor)
            return output.cpu().numpy()
    
    def get_memory_usage(self) -> float:
        """Get approximate memory usage."""
        if self.metadata:
            return self.metadata.model_size_mb
        return 0.0
    
    def unload_model(self) -> None:
        """Unload PyTorch model."""
        if self.model:
            del self.model
            self.model = None
            self.metadata = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

class ResourceManager:
    """Manages hardware resources for edge deployment."""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.current_memory_mb = 0.0
        self.current_cpu_percent = 0.0
        self.lock = threading.Lock()
        
    def check_resources(self, required_memory_mb: float) -> bool:
        """Check if resources are available for new model."""
        with self.lock:
            # Check memory
            total_memory = self.current_memory_mb + required_memory_mb
            if total_memory > self.config.max_memory_mb:
                return False
            
            # Check CPU
            if self.current_cpu_percent > self.config.max_cpu_percent:
                return False
            
            return True
    
    def allocate_resources(self, memory_mb: float) -> bool:
        """Allocate resources for model."""
        with self.lock:
            if not self.check_resources(memory_mb):
                return False
            
            self.current_memory_mb += memory_mb
            return True
    
    def release_resources(self, memory_mb: float) -> None:
        """Release allocated resources."""
        with self.lock:
            self.current_memory_mb = max(0, self.current_memory_mb - memory_mb)
    
    def update_cpu_usage(self) -> None:
        """Update current CPU usage."""
        self.current_cpu_percent = psutil.cpu_percent(interval=0.1)
    
    def get_resource_status(self) -> Dict[str, float]:
        """Get current resource utilization."""
        return {
            'memory_used_mb': self.current_memory_mb,
            'memory_available_mb': self.config.max_memory_mb - self.current_memory_mb,
            'memory_utilization_percent': (self.current_memory_mb / self.config.max_memory_mb) * 100,
            'cpu_utilization_percent': self.current_cpu_percent,
            'system_memory_mb': psutil.virtual_memory().used / 1024 / 1024,
            'system_memory_percent': psutil.virtual_memory().percent
        }

class ModelCache:
    """Caching system for inference results."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.lock = threading.Lock()
    
    def _generate_key(self, model_id: str, input_data: np.ndarray) -> str:
        """Generate cache key from model ID and input."""
        input_hash = hash(input_data.tobytes())
        return f"{model_id}_{input_hash}"
    
    def get(self, model_id: str, input_data: np.ndarray) -> Optional[np.ndarray]:
        """Get cached result if available."""
        key = self._generate_key(model_id, input_data)
        
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
        
        return None
    
    def put(self, model_id: str, input_data: np.ndarray, result: np.ndarray) -> None:
        """Cache inference result."""
        key = self._generate_key(model_id, input_data)
        
        with self.lock:
            # Remove least recently used if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            # Add/update cache
            if key in self.cache:
                self.access_order.remove(key)
            
            self.cache[key] = result.copy()
            self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear entire cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization_percent': int((len(self.cache) / self.max_size) * 100)
            }

class EdgeRuntime:
    """Main edge runtime for BatteryMind models."""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.resource_manager = ResourceManager(config)
        self.cache = ModelCache(config.cache_size)
        self.models: Dict[str, ModelRuntime] = {}
        self.model_metadata: Dict[str, ModelMetadata] = {}
        
        # Threading components
        self.request_queue = queue.PriorityQueue()
        self.result_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        
        # Performance tracking
        self.inference_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        
        # Telemetry buffer
        self.telemetry_buffer = []
        self.telemetry_lock = threading.Lock()
        
        logger.info("EdgeRuntime initialized")
    
    def start(self) -> None:
        """Start the edge runtime."""
        self.running = True
        
        # Start background threads
        threading.Thread(target=self._process_requests, daemon=True).start()
        threading.Thread(target=self._monitor_resources, daemon=True).start()
        threading.Thread(target=self._heartbeat, daemon=True).start()
        
        logger.info("EdgeRuntime started")
    
    def stop(self) -> None:
        """Stop the edge runtime."""
        self.running = False
        
        # Unload all models
        for model_id in list(self.models.keys()):
            self.unload_model(model_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("EdgeRuntime stopped")
    
    def load_model(self, model_path: str, metadata: ModelMetadata) -> bool:
        """Load a model into the runtime."""
        if metadata.model_id in self.models:
            logger.warning(f"Model {metadata.model_id} already loaded")
            return True
        
        if len(self.models) >= self.config.max_models:
            logger.error("Maximum number of models reached")
            return False
        
        # Check resources
        if not self.resource_manager.check_resources(metadata.model_size_mb):
            logger.error(f"Insufficient resources for model {metadata.model_id}")
            return False
        
        # Determine runtime type based on model file extension
        model_path_obj = Path(model_path)
        
        if model_path_obj.suffix == '.onnx' and ONNX_AVAILABLE:
            runtime = ONNXRuntime(self.config)
        elif model_path_obj.suffix == '.tflite' and TF_AVAILABLE:
            runtime = TensorFlowLiteRuntime(self.config)
        elif model_path_obj.suffix in ['.pt', '.pth'] and TORCH_AVAILABLE:
            runtime = PyTorchRuntime(self.config)
        else:
            logger.error(f"Unsupported model format: {model_path_obj.suffix}")
            return False
        
        # Load model
        if runtime.load_model(model_path, metadata):
            # Allocate resources
            self.resource_manager.allocate_resources(metadata.model_size_mb)
            
            # Store model and metadata
            self.models[metadata.model_id] = runtime
            self.model_metadata[metadata.model_id] = metadata
            
            # Warm up model if enabled
            if self.config.model_warm_up:
                self._warm_up_model(metadata.model_id)
            
            logger.info(f"Successfully loaded model: {metadata.model_id}")
            return True
        else:
            logger.error(f"Failed to load model: {metadata.model_id}")
            return False
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from the runtime."""
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not found")
            return False
        
        # Get model and metadata
        runtime = self.models[model_id]
        metadata = self.model_metadata[model_id]
        
        # Unload model
        runtime.unload_model()
        
        # Release resources
        self.resource_manager.release_resources(metadata.model_size_mb)
        
        # Remove from tracking
        del self.models[model_id]
        del self.model_metadata[model_id]
        
        logger.info(f"Unloaded model: {model_id}")
        return True
    
    def predict(self, request: InferenceRequest) -> Future[InferenceResult]:
        """Submit prediction request."""
        # Add to priority queue
        priority_item = (request.priority, time.time(), request)
        self.request_queue.put(priority_item)
        
        # Submit to executor and return future
        future = self.executor.submit(self._wait_for_result, request.request_id)
        return future
    
    def predict_sync(self, request: InferenceRequest) -> InferenceResult:
        """Synchronous prediction."""
        return self._process_single_request(request)
    
    def _process_single_request(self, request: InferenceRequest) -> InferenceResult:
        """Process a single inference request."""
        start_time = time.time()
        
        try:
            # Validate model exists
            if request.model_id not in self.models:
                return InferenceResult(
                    request_id=request.request_id,
                    model_id=request.model_id,
                    prediction=np.array([]),
                    confidence=0.0,
                    latency_ms=0.0,
                    error=f"Model {request.model_id} not found"
                )
            
            # Check cache first
            cached_result = self.cache.get(request.model_id, request.input_data)
            if cached_result is not None:
                latency_ms = (time.time() - start_time) * 1000
                return InferenceResult(
                    request_id=request.request_id,
                    model_id=request.model_id,
                    prediction=cached_result,
                    confidence=1.0,  # Cached results have high confidence
                    latency_ms=latency_ms,
                    metadata={'cached': True}
                )
            
            # Get model runtime
            runtime = self.models[request.model_id]
            
            # Run inference
            prediction = runtime.predict(request.input_data)
            
            # Calculate confidence (simplified)
            confidence = float(np.max(prediction)) if prediction.size > 0 else 0.0
            
            # Cache result
            self.cache.put(request.model_id, request.input_data, prediction)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Update performance metrics
            self.inference_count += 1
            self.total_latency += latency_ms
            
            # Create result
            result = InferenceResult(
                request_id=request.request_id,
                model_id=request.model_id,
                prediction=prediction,
                confidence=confidence,
                latency_ms=latency_ms,
                metadata={'cached': False}
            )
            
            # Add to telemetry
            self._add_telemetry(request, result)
            
            return result
            
        except Exception as e:
            self.error_count += 1
            latency_ms = (time.time() - start_time) * 1000
            
            logger.error(f"Inference error for request {request.request_id}: {e}")
            
            return InferenceResult(
                request_id=request.request_id,
                model_id=request.model_id,
                prediction=np.array([]),
                confidence=0.0,
                latency_ms=latency_ms,
                error=str(e)
            )
    
    def _process_requests(self) -> None:
        """Background thread to process inference requests."""
        while self.running:
            try:
                # Get request from queue with timeout
                priority, timestamp, request = self.request_queue.get(timeout=1.0)
                
                # Process request
                result = self._process_single_request(request)
                
                # Store result
                self.result_cache[request.request_id] = result
                
                # Clean up old results
                self._cleanup_old_results()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing request: {e}")
    
    def _wait_for_result(self, request_id: str) -> InferenceResult:
        """Wait for inference result."""
        timeout = self.config.inference_timeout_ms / 1000.0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request_id in self.result_cache:
                result = self.result_cache.pop(request_id)
                return result
            time.sleep(0.001)  # 1ms sleep
        
        # Timeout
        return InferenceResult(
            request_id=request_id,
            model_id="unknown",
            prediction=np.array([]),
            confidence=0.0,
            latency_ms=timeout * 1000,
            error="Request timeout"
        )
    
    def _cleanup_old_results(self) -> None:
        """Clean up old cached results."""
        current_time = time.time()
        timeout = 60.0  # 60 seconds
        
        expired_keys = [
            key for key, result in self.result_cache.items()
            if current_time - result.timestamp > timeout
        ]
        
        for key in expired_keys:
            del self.result_cache[key]
    
    def _warm_up_model(self, model_id: str) -> None:
        """Warm up model with dummy data."""
        try:
            metadata = self.model_metadata[model_id]
            dummy_input = np.random.randn(*metadata.input_shape).astype(np.float32)
            
            # Run a few warm-up predictions
            for _ in range(3):
                self.models[model_id].predict(dummy_input)
            
            logger.info(f"Model {model_id} warmed up")
            
        except Exception as e:
            logger.warning(f"Failed to warm up model {model_id}: {e}")
    
    def _monitor_resources(self) -> None:
        """Background thread to monitor system resources."""
        while self.running:
            try:
                self.resource_manager.update_cpu_usage()
                
                # Log resource status periodically
                if self.inference_count % 100 == 0:
                    status = self.resource_manager.get_resource_status()
                    logger.debug(f"Resource status: {status}")
                
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
    
    def _heartbeat(self) -> None:
        """Send periodic heartbeat."""
        while self.running:
            try:
                self._send_heartbeat()
                time.sleep(self.config.heartbeat_interval_sec)
                
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
    
    def _send_heartbeat(self) -> None:
        """Send heartbeat with system status."""
        status = {
            'timestamp': time.time(),
            'models_loaded': len(self.models),
            'inference_count': self.inference_count,
            'error_count': self.error_count,
            'avg_latency_ms': self.total_latency / max(1, self.inference_count),
            'resource_status': self.resource_manager.get_resource_status(),
            'cache_stats': self.cache.get_stats()
        }
        
        logger.info(f"Heartbeat: {status}")
    
    def _add_telemetry(self, request: InferenceRequest, result: InferenceResult) -> None:
        """Add telemetry data."""
        with self.telemetry_lock:
            telemetry = {
                'timestamp': result.timestamp,
                'model_id': request.model_id,
                'latency_ms': result.latency_ms,
                'confidence': result.confidence,
                'input_shape': list(request.input_data.shape),
                'output_shape': list(result.prediction.shape) if result.prediction.size > 0 else [],
                'cached': result.metadata.get('cached', False),
                'error': result.error is not None
            }
            
            self.telemetry_buffer.append(telemetry)
            
            # Keep buffer size limited
            if len(self.telemetry_buffer) > self.config.telemetry_buffer_size:
                self.telemetry_buffer.pop(0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive runtime status."""
        return {
            'runtime_status': 'running' if self.running else 'stopped',
            'models_loaded': len(self.models),
            'model_list': list(self.models.keys()),
            'performance_metrics': {
                'total_inferences': self.inference_count,
                'total_errors': self.error_count,
                'avg_latency_ms': self.total_latency / max(1, self.inference_count),
                'error_rate': self.error_count / max(1, self.inference_count)
            },
            'resource_status': self.resource_manager.get_resource_status(),
            'cache_stats': self.cache.get_stats(),
            'queue_size': self.request_queue.qsize(),
            'config': self.config.__dict__
        }
    
    def get_telemetry(self) -> List[Dict[str, Any]]:
        """Get telemetry data."""
        with self.telemetry_lock:
            return self.telemetry_buffer.copy()
    
    def clear_telemetry(self) -> None:
        """Clear telemetry buffer."""
        with self.telemetry_lock:
            self.telemetry_buffer.clear()

# Factory function for easy instantiation
def create_edge_runtime(config_path: Optional[str] = None) -> EdgeRuntime:
    """Create edge runtime with configuration."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = EdgeConfig(**config_data)
    else:
        config = EdgeConfig()
    
    return EdgeRuntime(config)

# Main execution for standalone deployment
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BatteryMind Edge Runtime")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--models-dir", type=str, default="./models", help="Models directory")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port")
    
    args = parser.parse_args()
    
    # Create runtime
    runtime = create_edge_runtime(args.config)
    
    try:
        runtime.start()
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        runtime.stop()
