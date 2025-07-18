"""
BatteryMind Edge Deployment - Model Quantization

Advanced quantization techniques for deploying battery AI models on edge devices
with limited computational resources. This module provides comprehensive 
quantization strategies including dynamic quantization, static quantization,
and QAT (Quantization Aware Training).

Features:
- Dynamic quantization for immediate size reduction
- Static quantization with calibration datasets
- Quantization Aware Training (QAT) for minimal accuracy loss
- Mixed precision quantization strategies
- Hardware-specific optimizations (ARM, x86, specialized chips)
- Quantization validation and accuracy assessment
- Performance benchmarking tools

Author: BatteryMind Development Team
Version: 1.0.0
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core ML frameworks
try:
    import torch
    import torch.nn as nn
    import torch.quantization as quantization
    from torch.quantization import QConfig, default_qconfig
    from torch.quantization.quantize_fx import prepare_fx, convert_fx
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - some quantization features disabled")

try:
    import tensorflow as tf
    from tensorflow.lite.python import lite
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available - some quantization features disabled")

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, CalibrationDataReader
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX not available - some quantization features disabled")

# BatteryMind imports
from .model_optimization import ModelOptimizer, OptimizationConfig
from ...utils.logging_utils import setup_logger
from ...evaluation.metrics.performance_metrics import PerformanceMetrics

# Configure logging
logger = setup_logger(__name__)

class QuantizationConfig:
    """Configuration for model quantization operations."""
    
    def __init__(self):
        # Quantization strategy
        self.quantization_type = 'dynamic'  # 'dynamic', 'static', 'qat'
        self.target_dtype = 'int8'  # 'int8', 'int16', 'float16'
        
        # Hardware optimization
        self.target_hardware = 'cpu'  # 'cpu', 'arm', 'gpu', 'edge_tpu'
        self.use_hardware_acceleration = True
        
        # Accuracy preservation
        self.accuracy_threshold = 0.02  # Max 2% accuracy loss
        self.preserve_critical_layers = True
        self.mixed_precision = False
        
        # Calibration settings
        self.calibration_dataset_size = 1000
        self.calibration_method = 'entropy'  # 'entropy', 'minmax', 'percentile'
        
        # Output settings
        self.output_format = 'onnx'  # 'onnx', 'tflite', 'pytorch'
        self.validate_quantized_model = True
        
        # Performance settings
        self.target_latency_ms = 50.0
        self.target_memory_mb = 100.0
        self.target_power_mw = 500.0

class BatteryDataCalibrator(CalibrationDataReader if ONNX_AVAILABLE else object):
    """Custom calibration data reader for battery models."""
    
    def __init__(self, calibration_data: np.ndarray, batch_size: int = 32):
        """
        Initialize calibration data reader.
        
        Args:
            calibration_data: Calibration dataset
            batch_size: Batch size for calibration
        """
        self.calibration_data = calibration_data
        self.batch_size = batch_size
        self.current_index = 0
        self.total_samples = len(calibration_data)
        
    def get_next(self) -> Dict[str, np.ndarray]:
        """Get next batch of calibration data."""
        if self.current_index >= self.total_samples:
            return None
        
        end_index = min(self.current_index + self.batch_size, self.total_samples)
        batch = self.calibration_data[self.current_index:end_index]
        self.current_index = end_index
        
        return {'input': batch.astype(np.float32)}

class ModelQuantizer:
    """Main quantization engine for battery AI models."""
    
    def __init__(self, config: QuantizationConfig):
        """
        Initialize model quantizer.
        
        Args:
            config: Quantization configuration
        """
        self.config = config
        self.performance_metrics = PerformanceMetrics()
        self.quantization_results = {}
        
        logger.info("ModelQuantizer initialized with config: %s", config.quantization_type)
    
    def quantize_model(self, 
                      model_path: str, 
                      calibration_data: Optional[np.ndarray] = None,
                      output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Quantize a model based on the configuration.
        
        Args:
            model_path: Path to the original model
            calibration_data: Optional calibration dataset
            output_path: Output path for quantized model
            
        Returns:
            Dictionary with quantization results
        """
        start_time = datetime.now()
        
        try:
            # Determine model format
            model_format = self._detect_model_format(model_path)
            
            # Perform quantization based on format
            if model_format == 'pytorch':
                result = self._quantize_pytorch_model(model_path, calibration_data, output_path)
            elif model_format == 'tensorflow':
                result = self._quantize_tensorflow_model(model_path, calibration_data, output_path)
            elif model_format == 'onnx':
                result = self._quantize_onnx_model(model_path, calibration_data, output_path)
            else:
                raise ValueError(f"Unsupported model format: {model_format}")
            
            # Calculate optimization metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            result['execution_time_seconds'] = execution_time
            
            # Validate quantized model if requested
            if self.config.validate_quantized_model:
                validation_result = self._validate_quantized_model(
                    result['quantized_model_path'], 
                    model_path,
                    calibration_data
                )
                result.update(validation_result)
            
            self.quantization_results[model_path] = result
            logger.info("Model quantization completed in %.2f seconds", execution_time)
            
            return result
            
        except Exception as e:
            logger.error("Model quantization failed: %s", str(e))
            raise
    
    def _detect_model_format(self, model_path: str) -> str:
        """Detect model format from file extension."""
        path_obj = Path(model_path)
        extension = path_obj.suffix.lower()
        
        if extension in ['.pt', '.pth']:
            return 'pytorch'
        elif extension in ['.pb', '.h5', '.keras']:
            return 'tensorflow'
        elif extension == '.onnx':
            return 'onnx'
        else:
            raise ValueError(f"Unknown model format: {extension}")
    
    def _quantize_pytorch_model(self, 
                               model_path: str, 
                               calibration_data: Optional[np.ndarray],
                               output_path: Optional[str]) -> Dict[str, Any]:
        """Quantize PyTorch model."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for quantization")
        
        # Load model
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        
        # Prepare for quantization
        if self.config.quantization_type == 'dynamic':
            quantized_model = self._pytorch_dynamic_quantization(model)
        elif self.config.quantization_type == 'static':
            quantized_model = self._pytorch_static_quantization(model, calibration_data)
        elif self.config.quantization_type == 'qat':
            quantized_model = self._pytorch_qat_quantization(model, calibration_data)
        else:
            raise ValueError(f"Unknown quantization type: {self.config.quantization_type}")
        
        # Save quantized model
        if output_path is None:
            output_path = model_path.replace('.pt', '_quantized.pt')
        
        torch.save(quantized_model, output_path)
        
        # Calculate compression metrics
        original_size = os.path.getsize(model_path)
        quantized_size = os.path.getsize(output_path)
        compression_ratio = original_size / quantized_size
        
        return {
            'quantized_model_path': output_path,
            'original_size_mb': original_size / (1024 * 1024),
            'quantized_size_mb': quantized_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'quantization_type': self.config.quantization_type,
            'target_dtype': self.config.target_dtype,
            'framework': 'pytorch'
        }
    
    def _pytorch_dynamic_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply dynamic quantization to PyTorch model."""
        # Define layers to quantize
        layers_to_quantize = [torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d]
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            layers_to_quantize,
            dtype=torch.qint8 if self.config.target_dtype == 'int8' else torch.qfloat16
        )
        
        return quantized_model
    
    def _pytorch_static_quantization(self, 
                                   model: torch.nn.Module, 
                                   calibration_data: np.ndarray) -> torch.nn.Module:
        """Apply static quantization to PyTorch model."""
        # Set quantization configuration
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model for static quantization
        model_prepared = torch.quantization.prepare(model)
        
        # Calibrate with sample data
        if calibration_data is not None:
            model_prepared.eval()
            with torch.no_grad():
                for i in range(min(100, len(calibration_data))):  # Use subset for calibration
                    sample = torch.from_numpy(calibration_data[i:i+1]).float()
                    model_prepared(sample)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)
        
        return quantized_model
    
    def _pytorch_qat_quantization(self, 
                                model: torch.nn.Module, 
                                calibration_data: np.ndarray) -> torch.nn.Module:
        """Apply Quantization Aware Training to PyTorch model."""
        # This is a simplified QAT implementation
        # In practice, you would need to retrain the model with quantization-aware training
        
        # Set QAT configuration
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare model for QAT
        model_prepared = torch.quantization.prepare_qat(model)
        
        # In real implementation, you would train the model here
        # For now, we'll just calibrate
        model_prepared.eval()
        if calibration_data is not None:
            with torch.no_grad():
                for i in range(min(100, len(calibration_data))):
                    sample = torch.from_numpy(calibration_data[i:i+1]).float()
                    model_prepared(sample)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)
        
        return quantized_model
    
    def _quantize_tensorflow_model(self, 
                                 model_path: str, 
                                 calibration_data: Optional[np.ndarray],
                                 output_path: Optional[str]) -> Dict[str, Any]:
        """Quantize TensorFlow model."""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available for quantization")
        
        # Load model
        if model_path.endswith('.h5'):
            model = tf.keras.models.load_model(model_path)
        else:
            model = tf.saved_model.load(model_path)
        
        # Convert to TensorFlow Lite with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if self.config.quantization_type == 'dynamic':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif self.config.quantization_type == 'static':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self._create_tf_calibration_dataset(calibration_data)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        # Convert model
        quantized_tflite_model = converter.convert()
        
        # Save quantized model
        if output_path is None:
            output_path = model_path.replace('.h5', '_quantized.tflite')
        
        with open(output_path, 'wb') as f:
            f.write(quantized_tflite_model)
        
        # Calculate compression metrics
        original_size = os.path.getsize(model_path)
        quantized_size = len(quantized_tflite_model)
        compression_ratio = original_size / quantized_size
        
        return {
            'quantized_model_path': output_path,
            'original_size_mb': original_size / (1024 * 1024),
            'quantized_size_mb': quantized_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'quantization_type': self.config.quantization_type,
            'target_dtype': self.config.target_dtype,
            'framework': 'tensorflow'
        }
    
    def _create_tf_calibration_dataset(self, calibration_data: np.ndarray):
        """Create TensorFlow calibration dataset generator."""
        def representative_data_gen():
            for i in range(min(100, len(calibration_data))):
                yield [calibration_data[i:i+1].astype(np.float32)]
        
        return representative_data_gen
    
    def _quantize_onnx_model(self, 
                           model_path: str, 
                           calibration_data: Optional[np.ndarray],
                           output_path: Optional[str]) -> Dict[str, Any]:
        """Quantize ONNX model."""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX not available for quantization")
        
        if output_path is None:
            output_path = model_path.replace('.onnx', '_quantized.onnx')
        
        if self.config.quantization_type == 'dynamic':
            # Dynamic quantization
            quantize_dynamic(
                model_input=model_path,
                model_output=output_path,
                weight_type=QuantType.QInt8 if self.config.target_dtype == 'int8' else QuantType.QUInt8
            )
        
        elif self.config.quantization_type == 'static':
            # Static quantization with calibration
            if calibration_data is None:
                raise ValueError("Calibration data required for static quantization")
            
            calibrator = BatteryDataCalibrator(calibration_data)
            
            quantize_static(
                model_input=model_path,
                model_output=output_path,
                calibration_data_reader=calibrator,
                quant_format=QuantType.QInt8 if self.config.target_dtype == 'int8' else QuantType.QUInt8
            )
        
        # Calculate compression metrics
        original_size = os.path.getsize(model_path)
        quantized_size = os.path.getsize(output_path)
        compression_ratio = original_size / quantized_size
        
        return {
            'quantized_model_path': output_path,
            'original_size_mb': original_size / (1024 * 1024),
            'quantized_size_mb': quantized_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'quantization_type': self.config.quantization_type,
            'target_dtype': self.config.target_dtype,
            'framework': 'onnx'
        }
    
    def _validate_quantized_model(self, 
                                quantized_model_path: str,
                                original_model_path: str,
                                validation_data: Optional[np.ndarray]) -> Dict[str, Any]:
        """Validate quantized model performance."""
        if validation_data is None:
            return {'validation_performed': False}
        
        try:
            # Load models and compare outputs
            original_outputs = self._run_inference(original_model_path, validation_data[:100])
            quantized_outputs = self._run_inference(quantized_model_path, validation_data[:100])
            
            # Calculate accuracy metrics
            mse = np.mean((original_outputs - quantized_outputs) ** 2)
            mae = np.mean(np.abs(original_outputs - quantized_outputs))
            max_error = np.max(np.abs(original_outputs - quantized_outputs))
            
            # Calculate relative error
            relative_error = mae / (np.mean(np.abs(original_outputs)) + 1e-8)
            
            # Performance comparison
            original_latency = self._measure_inference_latency(original_model_path, validation_data[:10])
            quantized_latency = self._measure_inference_latency(quantized_model_path, validation_data[:10])
            
            speedup = original_latency / quantized_latency if quantized_latency > 0 else 0
            
            return {
                'validation_performed': True,
                'mse': float(mse),
                'mae': float(mae),
                'max_error': float(max_error),
                'relative_error': float(relative_error),
                'accuracy_preserved': relative_error < self.config.accuracy_threshold,
                'original_latency_ms': original_latency * 1000,
                'quantized_latency_ms': quantized_latency * 1000,
                'speedup_factor': speedup
            }
            
        except Exception as e:
            logger.error("Validation failed: %s", str(e))
            return {'validation_performed': False, 'error': str(e)}
    
    def _run_inference(self, model_path: str, input_data: np.ndarray) -> np.ndarray:
        """Run inference on model."""
        model_format = self._detect_model_format(model_path)
        
        if model_format == 'onnx':
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_data.astype(np.float32)})
            return outputs[0]
        
        elif model_format == 'pytorch':
            model = torch.load(model_path, map_location='cpu')
            model.eval()
            with torch.no_grad():
                outputs = model(torch.from_numpy(input_data).float())
            return outputs.numpy()
        
        elif model_format == 'tensorflow':
            if model_path.endswith('.tflite'):
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                outputs = []
                for sample in input_data:
                    interpreter.set_tensor(input_details[0]['index'], sample.reshape(1, -1))
                    interpreter.invoke()
                    output = interpreter.get_tensor(output_details[0]['index'])
                    outputs.append(output[0])
                
                return np.array(outputs)
            else:
                model = tf.keras.models.load_model(model_path)
                return model.predict(input_data)
        
        else:
            raise ValueError(f"Unsupported model format for inference: {model_format}")
    
    def _measure_inference_latency(self, model_path: str, input_data: np.ndarray) -> float:
        """Measure inference latency."""
        import time
        
        # Warm up
        for _ in range(3):
            self._run_inference(model_path, input_data[:1])
        
        # Measure latency
        start_time = time.time()
        for sample in input_data:
            self._run_inference(model_path, sample.reshape(1, -1))
        end_time = time.time()
        
        return (end_time - start_time) / len(input_data)
    
    def quantize_battery_model_suite(self, 
                                   model_paths: Dict[str, str],
                                   calibration_data: Dict[str, np.ndarray],
                                   output_dir: str) -> Dict[str, Any]:
        """
        Quantize a complete suite of battery models.
        
        Args:
            model_paths: Dictionary of model names to paths
            calibration_data: Dictionary of model names to calibration data
            output_dir: Output directory for quantized models
            
        Returns:
            Dictionary with quantization results for all models
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        for model_name, model_path in model_paths.items():
            logger.info("Quantizing model: %s", model_name)
            
            try:
                # Prepare output path
                output_path = os.path.join(output_dir, f"{model_name}_quantized")
                
                # Get calibration data for this model
                calib_data = calibration_data.get(model_name)
                
                # Quantize model
                result = self.quantize_model(model_path, calib_data, output_path)
                results[model_name] = result
                
                logger.info("Successfully quantized %s: %.1fx compression", 
                          model_name, result['compression_ratio'])
                
            except Exception as e:
                logger.error("Failed to quantize %s: %s", model_name, str(e))
                results[model_name] = {'error': str(e)}
        
        # Generate summary report
        summary = self._generate_quantization_summary(results)
        results['summary'] = summary
        
        # Save results
        results_path = os.path.join(output_dir, 'quantization_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def _generate_quantization_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of quantization results."""
        successful_quantizations = [r for r in results.values() if 'error' not in r]
        
        if not successful_quantizations:
            return {'total_models': len(results), 'successful': 0, 'failed': len(results)}
        
        # Calculate aggregate metrics
        avg_compression = np.mean([r['compression_ratio'] for r in successful_quantizations])
        total_size_reduction_mb = sum([
            r['original_size_mb'] - r['quantized_size_mb'] 
            for r in successful_quantizations
        ])
        
        # Performance metrics (if available)
        performance_results = [r for r in successful_quantizations if r.get('validation_performed')]
        avg_speedup = np.mean([r['speedup_factor'] for r in performance_results]) if performance_results else 0
        avg_accuracy_loss = np.mean([r['relative_error'] for r in performance_results]) if performance_results else 0
        
        return {
            'total_models': len(results),
            'successful_quantizations': len(successful_quantizations),
            'failed_quantizations': len(results) - len(successful_quantizations),
            'average_compression_ratio': float(avg_compression),
            'total_size_reduction_mb': float(total_size_reduction_mb),
            'average_speedup_factor': float(avg_speedup),
            'average_accuracy_loss': float(avg_accuracy_loss),
            'quantization_type': self.config.quantization_type,
            'target_dtype': self.config.target_dtype
        }

class AdvancedQuantizationTechniques:
    """Advanced quantization techniques for specialized scenarios."""
    
    @staticmethod
    def mixed_precision_quantization(model_path: str, 
                                   sensitive_layers: List[str],
                                   output_path: str) -> Dict[str, Any]:
        """
        Apply mixed precision quantization, keeping sensitive layers in higher precision.
        
        Args:
            model_path: Path to original model
            sensitive_layers: List of layer names to keep in higher precision
            output_path: Output path for quantized model
            
        Returns:
            Quantization results
        """
        # This is a placeholder for mixed precision quantization
        # Implementation would depend on the specific framework and model architecture
        
        logger.info("Applying mixed precision quantization")
        
        # In practice, you would:
        # 1. Analyze model sensitivity by layer
        # 2. Apply different quantization schemes to different layers
        # 3. Optimize the mixed precision configuration
        
        return {
            'technique': 'mixed_precision',
            'sensitive_layers': sensitive_layers,
            'output_path': output_path,
            'status': 'placeholder_implementation'
        }
    
    @staticmethod
    def knowledge_distillation_quantization(teacher_model_path: str,
                                          student_model_path: str,
                                          training_data: np.ndarray,
                                          output_path: str) -> Dict[str, Any]:
        """
        Apply quantization with knowledge distillation to minimize accuracy loss.
        
        Args:
            teacher_model_path: Path to full precision teacher model
            student_model_path: Path to quantized student model
            training_data: Training data for distillation
            output_path: Output path for final model
            
        Returns:
            Quantization results with distillation metrics
        """
        # This is a placeholder for knowledge distillation quantization
        logger.info("Applying knowledge distillation quantization")
        
        return {
            'technique': 'knowledge_distillation',
            'teacher_model': teacher_model_path,
            'student_model': student_model_path,
            'output_path': output_path,
            'status': 'placeholder_implementation'
        }

def create_battery_quantization_pipeline(model_paths: Dict[str, str],
                                       calibration_data: Dict[str, np.ndarray],
                                       config: Optional[QuantizationConfig] = None) -> ModelQuantizer:
    """
    Create a complete quantization pipeline for battery models.
    
    Args:
        model_paths: Dictionary of model names to paths
        calibration_data: Dictionary of calibration datasets
        config: Optional quantization configuration
        
    Returns:
        Configured ModelQuantizer instance
    """
    if config is None:
        config = QuantizationConfig()
    
    quantizer = ModelQuantizer(config)
    
    logger.info("Created battery quantization pipeline with %d models", len(model_paths))
    
    return quantizer

# Export main classes
__all__ = [
    'QuantizationConfig',
    'ModelQuantizer',
    'BatteryDataCalibrator',
    'AdvancedQuantizationTechniques',
    'create_battery_quantization_pipeline'
]
