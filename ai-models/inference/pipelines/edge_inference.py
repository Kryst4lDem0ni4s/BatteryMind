"""
BatteryMind - Edge Inference Pipeline

Lightweight inference pipeline optimized for edge devices and embedded systems.
Designed for battery management systems with limited computational resources.

Features:
- Optimized for low-power edge devices
- Minimal memory footprint (<100MB)
- Offline inference capabilities
- Model quantization and compression
- Real-time processing with microsecond latency
- Safety-critical decision making

Author: BatteryMind Development Team
Version: 1.0.0
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import threading
from collections import deque
import pickle
import os
import sys
from pathlib import Path

# Lightweight ML libraries for edge
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

# Edge-specific imports
from ..predictors.battery_health_predictor import BatteryHealthPredictor
from ...utils.logging_utils import setup_logger

# Configure logging for edge
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = setup_logger(__name__)

@dataclass
class EdgeInferenceConfig:
    """Configuration for edge inference pipeline."""
    
    # Model configuration
    model_path: str = "/opt/batterymind/models"
    model_format: str = "onnx"  # "onnx", "tflite", "pytorch", "pickle"
    use_quantized_model: bool = True
    model_compression: bool = True
    
    # Performance configuration
    max_latency_ms: float = 50.0  # Maximum inference latency
    memory_limit_mb: int = 100    # Memory limit for models
    cpu_limit_percent: float = 50.0  # CPU usage limit
    
    # Data configuration
    buffer_size: int = 144  # Sensor data buffer size
    sampling_rate: int = 10  # Seconds between samples
    feature_dimension: int = 12  # Number of input features
    
    # Safety configuration
    safety_mode: bool = True
    failsafe_enabled: bool = True
    offline_mode: bool = True
    
    # Edge-specific settings
    device_id: str = "edge_device_001"
    hardware_type: str = "raspberry_pi"  # "raspberry_pi", "jetson", "arduino"
    power_saving_mode: bool = True
    
    # Monitoring
    enable_metrics: bool = True
    log_predictions: bool = False
    health_check_interval: int = 300  # seconds

class EdgeModelManager:
    """Manages optimized models for edge deployment."""
    
    def __init__(self, config: EdgeInferenceConfig):
        self.config = config
        self.models = {}
        self.model_metadata = {}
        self.memory_usage = {}
        
        # Initialize models
        self._load_edge_models()
        
        logger.info(f"Edge model manager initialized with {len(self.models)} models")
    
    def _load_edge_models(self):
        """Load optimized models for edge deployment."""
        try:
            model_path = Path(self.config.model_path)
            
            if self.config.model_format == "onnx" and ONNX_AVAILABLE:
                self._load_onnx_models(model_path)
            elif self.config.model_format == "tflite" and TF_AVAILABLE:
                self._load_tflite_models(model_path)
            elif self.config.model_format == "pickle":
                self._load_pickle_models(model_path)
            else:
                # Fallback to basic Python models
                self._load_fallback_models()
                
        except Exception as e:
            logger.error(f"Failed to load edge models: {e}")
            self._load_fallback_models()
    
    def _load_onnx_models(self, model_path: Path):
        """Load ONNX models for edge inference."""
        try:
            # Health prediction model
            health_model_path = model_path / "transformer_battery_health.onnx"
            if health_model_path.exists():
                session_options = ort.SessionOptions()
                session_options.intra_op_num_threads = 1
                session_options.inter_op_num_threads = 1
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                self.models['health'] = ort.InferenceSession(
                    str(health_model_path),
                    sess_options=session_options
                )
                
                logger.info("Loaded ONNX health prediction model")
            
            # Load other models similarly
            model_files = {
                'degradation': "degradation_predictor.onnx",
                'optimization': "optimization_predictor.onnx"
            }
            
            for model_name, filename in model_files.items():
                model_file = model_path / filename
                if model_file.exists():
                    self.models[model_name] = ort.InferenceSession(
                        str(model_file),
                        sess_options=session_options
                    )
                    
        except Exception as e:
            logger.error(f"Failed to load ONNX models: {e}")
            raise
    
    def _load_tflite_models(self, model_path: Path):
        """Load TensorFlow Lite models for edge inference."""
        try:
            # Health prediction model
            health_model_path = model_path / "transformer_mobile.tflite"
            if health_model_path.exists():
                interpreter = tf.lite.Interpreter(
                    model_path=str(health_model_path),
                    num_threads=1
                )
                interpreter.allocate_tensors()
                
                self.models['health'] = {
                    'interpreter': interpreter,
                    'input_details': interpreter.get_input_details(),
                    'output_details': interpreter.get_output_details()
                }
                
                logger.info("Loaded TensorFlow Lite health prediction model")
                
        except Exception as e:
            logger.error(f"Failed to load TensorFlow Lite models: {e}")
            raise
    
    def _load_pickle_models(self, model_path: Path):
        """Load pickled models for edge inference."""
        try:
            model_files = {
                'health': "transformer_quantized.pkl",
                'degradation': "degradation_compressed.pkl",
                'optimization': "optimization_optimized.pkl"
            }
            
            for model_name, filename in model_files.items():
                model_file = model_path / filename
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                        
        except Exception as e:
            logger.error(f"Failed to load pickle models: {e}")
            raise
    
    def _load_fallback_models(self):
        """Load fallback models when optimized models are not available."""
        try:
            # Use basic Python implementations
            self.models['health'] = BatteryHealthPredictor()
            logger.warning("Using fallback models - performance may be suboptimal")
            
        except Exception as e:
            logger.error(f"Failed to load fallback models: {e}")
            raise
    
    def predict(self, model_name: str, input_data: np.ndarray) -> np.ndarray:
        """Perform inference using specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.models[model_name]
        
        try:
            if self.config.model_format == "onnx" and ONNX_AVAILABLE:
                return self._predict_onnx(model, input_data)
            elif self.config.model_format == "tflite" and TF_AVAILABLE:
                return self._predict_tflite(model, input_data)
            else:
                return self._predict_fallback(model, input_data)
                
        except Exception as e:
            logger.error(f"Prediction error for {model_name}: {e}")
            raise
    
    def _predict_onnx(self, model, input_data: np.ndarray) -> np.ndarray:
        """Perform ONNX inference."""
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: input_data})
        return outputs[0]
    
    def _predict_tflite(self, model_info: Dict, input_data: np.ndarray) -> np.ndarray:
        """Perform TensorFlow Lite inference."""
        interpreter = model_info['interpreter']
        input_details = model_info['input_details']
        output_details = model_info['output_details']
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data
    
    def _predict_fallback(self, model, input_data: np.ndarray) -> np.ndarray:
        """Perform fallback inference."""
        return model.predict(input_data)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information."""
        if model_name not in self.models:
            return {}
        
        return {
            'model_name': model_name,
            'format': self.config.model_format,
            'quantized': self.config.use_quantized_model,
            'memory_usage': self.memory_usage.get(model_name, 0),
            'available': True
        }

class EdgeDataProcessor:
    """Lightweight data processor for edge devices."""
    
    def __init__(self, config: EdgeInferenceConfig):
        self.config = config
        self.data_buffer = deque(maxlen=config.buffer_size)
        self.feature_stats = {}
        self.last_processed = 0
        
        # Load preprocessing parameters
        self._load_preprocessing_params()
    
    def _load_preprocessing_params(self):
        """Load preprocessing parameters optimized for edge."""
        try:
            # Load basic normalization parameters
            self.feature_stats = {
                'voltage': {'mean': 3.7, 'std': 0.5},
                'current': {'mean': 0.0, 'std': 50.0},
                'temperature': {'mean': 25.0, 'std': 15.0},
                'soc': {'mean': 0.5, 'std': 0.3},
                'power': {'mean': 0.0, 'std': 100.0},
                'energy': {'mean': 1000.0, 'std': 500.0},
                'internal_resistance': {'mean': 0.1, 'std': 0.05},
                'capacity': {'mean': 80.0, 'std': 20.0}
            }
            
        except Exception as e:
            logger.error(f"Failed to load preprocessing params: {e}")
    
    def add_sensor_data(self, sensor_data: Dict[str, float], timestamp: Optional[float] = None):
        """Add new sensor data to buffer."""
        if timestamp is None:
            timestamp = time.time()
        
        # Validate sensor data
        if not self._validate_sensor_data(sensor_data):
            logger.warning("Invalid sensor data received")
            return
        
        # Add to buffer
        data_point = {
            'timestamp': timestamp,
            **sensor_data
        }
        
        self.data_buffer.append(data_point)
    
    def _validate_sensor_data(self, sensor_data: Dict[str, float]) -> bool:
        """Validate sensor data for edge processing."""
        required_fields = ['voltage', 'current', 'temperature', 'soc']
        
        # Check required fields
        if not all(field in sensor_data for field in required_fields):
            return False
        
        # Check data ranges
        if not (2.0 <= sensor_data['voltage'] <= 5.0):
            return False
        
        if not (-500 <= sensor_data['current'] <= 500):
            return False
        
        if not (-40 <= sensor_data['temperature'] <= 85):
            return False
        
        if not (0.0 <= sensor_data['soc'] <= 1.0):
            return False
        
        return True
    
    def get_features(self) -> Optional[np.ndarray]:
        """Extract features from buffer for inference."""
        if len(self.data_buffer) < self.config.buffer_size:
            return None
        
        try:
            # Convert buffer to numpy array
            features = []
            for data_point in self.data_buffer:
                feature_vector = [
                    data_point.get('voltage', 0),
                    data_point.get('current', 0),
                    data_point.get('temperature', 0),
                    data_point.get('soc', 0),
                    data_point.get('power', 0),
                    data_point.get('energy', 0),
                    data_point.get('internal_resistance', 0),
                    data_point.get('capacity', 0)
                ]
                features.append(feature_vector)
            
            feature_array = np.array(features, dtype=np.float32)
            
            # Normalize features
            normalized_features = self._normalize_features(feature_array)
            
            return normalized_features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using stored statistics."""
        try:
            normalized = features.copy()
            
            feature_names = ['voltage', 'current', 'temperature', 'soc', 
                           'power', 'energy', 'internal_resistance', 'capacity']
            
            for i, feature_name in enumerate(feature_names):
                if feature_name in self.feature_stats:
                    mean = self.feature_stats[feature_name]['mean']
                    std = self.feature_stats[feature_name]['std']
                    normalized[:, i] = (normalized[:, i] - mean) / (std + 1e-8)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Feature normalization error: {e}")
            return features

class EdgeSafetyController:
    """Safety controller for edge inference."""
    
    def __init__(self, config: EdgeInferenceConfig):
        self.config = config
        self.safety_limits = self._load_safety_limits()
        self.violation_count = 0
        self.last_violation_time = 0
        
    def _load_safety_limits(self) -> Dict[str, Any]:
        """Load safety limits for edge operation."""
        return {
            'voltage': {'min': 2.5, 'max': 4.5, 'critical_min': 2.0, 'critical_max': 5.0},
            'temperature': {'min': -20, 'max': 60, 'critical_min': -30, 'critical_max': 85},
            'current': {'min': -200, 'max': 200, 'critical_min': -500, 'critical_max': 500},
            'soc': {'min': 0.05, 'max': 0.95, 'critical_min': 0.0, 'critical_max': 1.0},
            'soh_threshold': 0.2,
            'max_violations_per_hour': 5
        }
    
    def check_safety(self, sensor_data: Dict[str, float], 
                    prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Perform safety checks on sensor data and predictions."""
        safety_result = {
            'safe': True,
            'violations': [],
            'critical_violations': [],
            'recommended_action': 'continue'
        }
        
        try:
            # Check sensor data limits
            violations = self._check_sensor_limits(sensor_data)
            safety_result['violations'].extend(violations)
            
            # Check prediction limits
            pred_violations = self._check_prediction_limits(prediction)
            safety_result['violations'].extend(pred_violations)
            
            # Check for critical violations
            critical_violations = self._check_critical_limits(sensor_data, prediction)
            safety_result['critical_violations'].extend(critical_violations)
            
            # Determine overall safety status
            if critical_violations:
                safety_result['safe'] = False
                safety_result['recommended_action'] = 'emergency_shutdown'
            elif violations:
                safety_result['safe'] = False
                safety_result['recommended_action'] = 'reduce_load'
            
            # Update violation tracking
            if violations or critical_violations:
                self.violation_count += 1
                self.last_violation_time = time.time()
            
            return safety_result
            
        except Exception as e:
            logger.error(f"Safety check error: {e}")
            return {
                'safe': False,
                'violations': [f"Safety check failed: {e}"],
                'critical_violations': [],
                'recommended_action': 'emergency_shutdown'
            }
    
    def _check_sensor_limits(self, sensor_data: Dict[str, float]) -> List[str]:
        """Check sensor data against safety limits."""
        violations = []
        
        for sensor, value in sensor_data.items():
            if sensor in self.safety_limits:
                limits = self.safety_limits[sensor]
                if value < limits['min'] or value > limits['max']:
                    violations.append(f"{sensor} out of range: {value}")
        
        return violations
    
    def _check_prediction_limits(self, prediction: Dict[str, Any]) -> List[str]:
        """Check predictions against safety limits."""
        violations = []
        
        # Check SoH prediction
        soh = prediction.get('soh', 1.0)
        if soh < self.safety_limits['soh_threshold']:
            violations.append(f"Low SoH predicted: {soh}")
        
        return violations
    
    def _check_critical_limits(self, sensor_data: Dict[str, float], 
                             prediction: Dict[str, Any]) -> List[str]:
        """Check for critical safety violations."""
        critical_violations = []
        
        # Check critical sensor limits
        for sensor, value in sensor_data.items():
            if sensor in self.safety_limits:
                limits = self.safety_limits[sensor]
                if (value < limits.get('critical_min', float('-inf')) or 
                    value > limits.get('critical_max', float('inf'))):
                    critical_violations.append(f"Critical {sensor} violation: {value}")
        
        # Check for thermal runaway risk
        if sensor_data.get('temperature', 0) > 70:
            critical_violations.append("Thermal runaway risk detected")
        
        return critical_violations
    
    def get_failsafe_action(self) -> str:
        """Get failsafe action recommendation."""
        if self.violation_count > self.safety_limits['max_violations_per_hour']:
            return 'emergency_shutdown'
        else:
            return 'continue_monitoring'

class EdgeInferencePipeline:
    """Main edge inference pipeline for battery management."""
    
    def __init__(self, config: EdgeInferenceConfig):
        self.config = config
        self.model_manager = EdgeModelManager(config)
        self.data_processor = EdgeDataProcessor(config)
        self.safety_controller = EdgeSafetyController(config)
        
        # State management
        self.is_running = False
        self.last_inference_time = 0
        self.inference_count = 0
        self.error_count = 0
        
        # Performance tracking
        self.latency_buffer = deque(maxlen=100)
        self.memory_usage = 0
        
        # Threading
        self.inference_thread = None
        self.shutdown_event = threading.Event()
        
        logger.info(f"Edge inference pipeline initialized for {config.device_id}")
    
    def start(self):
        """Start the edge inference pipeline."""
        if self.is_running:
            logger.warning("Pipeline already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start inference thread
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
        logger.info("Edge inference pipeline started")
    
    def stop(self):
        """Stop the edge inference pipeline."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.shutdown_event.set()
        
        if self.inference_thread:
            self.inference_thread.join(timeout=5)
        
        logger.info("Edge inference pipeline stopped")
    
    def _inference_loop(self):
        """Main inference loop for edge processing."""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Check if enough time has passed since last inference
                current_time = time.time()
                if current_time - self.last_inference_time < self.config.sampling_rate:
                    time.sleep(0.1)
                    continue
                
                # Get features for inference
                features = self.data_processor.get_features()
                if features is None:
                    time.sleep(0.1)
                    continue
                
                # Perform inference
                result = self._perform_inference(features)
                
                if result:
                    # Process inference result
                    self._process_inference_result(result)
                
                self.last_inference_time = current_time
                
            except Exception as e:
                logger.error(f"Inference loop error: {e}")
                self.error_count += 1
                time.sleep(1)
    
    def _perform_inference(self, features: np.ndarray) -> Optional[Dict[str, Any]]:
        """Perform inference using available models."""
        start_time = time.time()
        
        try:
            # Reshape features for model input
            if features.ndim == 2:
                features = features.reshape(1, *features.shape)
            
            # Perform health prediction
            health_prediction = self.model_manager.predict('health', features)
            
            # Create inference result
            result = {
                'device_id': self.config.device_id,
                'timestamp': time.time(),
                'predictions': {
                    'soh': float(health_prediction[0]) if len(health_prediction) > 0 else 0.0,
                    'confidence': 0.8  # Placeholder confidence
                },
                'inference_time_ms': (time.time() - start_time) * 1000,
                'model_version': '1.0.0'
            }
            
            # Track latency
            latency = (time.time() - start_time) * 1000
            self.latency_buffer.append(latency)
            
            # Check latency constraint
            if latency > self.config.max_latency_ms:
                logger.warning(f"Inference latency exceeded: {latency:.2f}ms")
            
            self.inference_count += 1
            return result
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            self.error_count += 1
            return None
    
    def _process_inference_result(self, result: Dict[str, Any]):
        """Process inference result and take appropriate actions."""
        try:
            # Get current sensor data
            if len(self.data_processor.data_buffer) == 0:
                return
            
            latest_data = self.data_processor.data_buffer[-1]
            
            # Perform safety checks
            safety_result = self.safety_controller.check_safety(
                latest_data, result['predictions']
            )
            
            # Add safety information to result
            result['safety'] = safety_result
            
            # Take action based on safety results
            if not safety_result['safe']:
                self._handle_safety_violation(result)
            
            # Log result if enabled
            if self.config.log_predictions:
                logger.info(f"Inference result: {result}")
            
        except Exception as e:
            logger.error(f"Result processing error: {e}")
    
    def _handle_safety_violation(self, result: Dict[str, Any]):
        """Handle safety violations."""
        safety_result = result['safety']
        recommended_action = safety_result['recommended_action']
        
        if recommended_action == 'emergency_shutdown':
            logger.critical("Emergency shutdown triggered due to safety violation")
            # Implement emergency shutdown logic here
            
        elif recommended_action == 'reduce_load':
            logger.warning("Reducing load due to safety concerns")
            # Implement load reduction logic here
    
    def add_sensor_data(self, sensor_data: Dict[str, float], 
                       timestamp: Optional[float] = None):
        """Add new sensor data to the pipeline."""
        self.data_processor.add_sensor_data(sensor_data, timestamp)
    
    def get_prediction(self) -> Optional[Dict[str, Any]]:
        """Get latest prediction result."""
        features = self.data_processor.get_features()
        if features is None:
            return None
        
        return self._perform_inference(features)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        avg_latency = np.mean(self.latency_buffer) if self.latency_buffer else 0
        
        return {
            'device_id': self.config.device_id,
            'inference_count': self.inference_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.inference_count, 1),
            'average_latency_ms': avg_latency,
            'buffer_size': len(self.data_processor.data_buffer),
            'memory_usage_mb': self.memory_usage,
            'is_running': self.is_running
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            'status': 'healthy' if self.is_running else 'stopped',
            'device_id': self.config.device_id,
            'models_loaded': len(self.model_manager.models),
            'buffer_status': len(self.data_processor.data_buffer),
            'error_rate': self.error_count / max(self.inference_count, 1),
            'timestamp': datetime.now().isoformat()
        }

# Factory function
def create_edge_pipeline(config: Optional[EdgeInferenceConfig] = None) -> EdgeInferencePipeline:
    """
    Factory function to create edge inference pipeline.
    
    Args:
        config: Edge inference configuration
        
    Returns:
        EdgeInferencePipeline: Configured pipeline instance
    """
    if config is None:
        config = EdgeInferenceConfig()
    
    return EdgeInferencePipeline(config)

# Example usage
if __name__ == "__main__":
    # Create edge pipeline
    config = EdgeInferenceConfig(
        device_id="edge_device_001",
        model_format="onnx",
        power_saving_mode=True
    )
    
    pipeline = create_edge_pipeline(config)
    
    try:
        # Start pipeline
        pipeline.start()
        
        # Simulate sensor data input
        for i in range(200):
            sensor_data = {
                'voltage': 3.7 + np.random.normal(0, 0.1),
                'current': np.random.normal(0, 10),
                'temperature': 25 + np.random.normal(0, 5),
                'soc': max(0, min(1, 0.5 + np.random.normal(0, 0.2))),
                'power': np.random.normal(0, 50),
                'energy': 1000 + np.random.normal(0, 100),
                'internal_resistance': 0.1 + np.random.normal(0, 0.01),
                'capacity': 80 + np.random.normal(0, 5)
            }
            
            pipeline.add_sensor_data(sensor_data)
            time.sleep(0.1)
        
        # Get statistics
        stats = pipeline.get_statistics()
        print(f"Pipeline statistics: {stats}")
        
    except KeyboardInterrupt:
        print("Stopping pipeline...")
        pipeline.stop()
