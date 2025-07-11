"""
BatteryMind - Base Model Architecture

Abstract base classes and common utilities for all transformer models in the
BatteryMind system. Provides standardized interfaces, common functionality,
and shared architectural components.

Features:
- Abstract base class for all BatteryMind models
- Common model utilities and helper functions
- Standardized model interfaces and protocols
- Shared configuration management
- Model versioning and metadata handling
- Integration hooks for monitoring and logging

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import json
import time
from pathlib import Path
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """
    Metadata container for BatteryMind models.
    
    Attributes:
        model_name (str): Name of the model
        version (str): Model version
        created_at (str): Creation timestamp
        author (str): Model author
        description (str): Model description
        input_shape (Tuple): Expected input shape
        output_shape (Tuple): Expected output shape
        parameters_count (int): Total number of parameters
        model_size_mb (float): Model size in megabytes
        training_dataset (str): Training dataset identifier
        performance_metrics (Dict): Key performance metrics
        tags (List[str]): Model tags for categorization
    """
    model_name: str
    version: str = "1.0.0"
    created_at: str = ""
    author: str = "BatteryMind Development Team"
    description: str = ""
    input_shape: Tuple = ()
    output_shape: Tuple = ()
    parameters_count: int = 0
    model_size_mb: float = 0.0
    training_dataset: str = ""
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")

class ModelProtocol(Protocol):
    """Protocol defining the interface for BatteryMind models."""
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        ...
    
    def predict(self, x: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """High-level prediction interface."""
        ...
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        ...

class BatteryMindBaseModel(nn.Module, ABC):
    """
    Abstract base class for all BatteryMind transformer models.
    
    Provides common functionality, standardized interfaces, and shared utilities
    for battery health prediction, degradation forecasting, and optimization models.
    """
    
    def __init__(self, config: Any, metadata: Optional[ModelMetadata] = None):
        """
        Initialize base model.
        
        Args:
            config: Model configuration object
            metadata: Model metadata information
        """
        super().__init__()
        self.config = config
        self.metadata = metadata or ModelMetadata(
            model_name=self.__class__.__name__,
            description=f"BatteryMind {self.__class__.__name__} model"
        )
        
        # Model state tracking
        self._is_compiled = False
        self._is_quantized = False
        self._device = torch.device("cpu")
        
        # Performance tracking
        self._inference_times = []
        self._memory_usage = []
        
        # Hooks for monitoring
        self._forward_hooks = []
        self._backward_hooks = []
        
        logger.info(f"Initialized {self.__class__.__name__} model")
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Abstract forward pass method.
        
        Args:
            x (torch.Tensor): Input tensor
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, torch.Tensor]: Model outputs
        """
        pass
    
    @abstractmethod
    def predict(self, x: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        Abstract prediction method for high-level inference.
        
        Args:
            x (torch.Tensor): Input tensor
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dict[str, Any]: Model information dictionary
        """
        # Calculate model statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        # Update metadata
        self.metadata.parameters_count = total_params
        self.metadata.model_size_mb = model_size_mb
        
        return {
            "metadata": self.metadata.__dict__,
            "architecture": {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": model_size_mb,
                "device": str(self._device),
                "is_compiled": self._is_compiled,
                "is_quantized": self._is_quantized
            },
            "performance": {
                "avg_inference_time_ms": np.mean(self._inference_times) if self._inference_times else 0,
                "avg_memory_usage_mb": np.mean(self._memory_usage) if self._memory_usage else 0,
                "inference_count": len(self._inference_times)
            },
            "config": self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
        }
    
    def compile_model(self, **kwargs) -> 'BatteryMindBaseModel':
        """
        Compile model for optimized inference (PyTorch 2.0+).
        
        Args:
            **kwargs: Compilation arguments
            
        Returns:
            BatteryMindBaseModel: Self for method chaining
        """
        if hasattr(torch, 'compile'):
            try:
                compiled_model = torch.compile(self, **kwargs)
                self._is_compiled = True
                logger.info(f"Model {self.__class__.__name__} compiled successfully")
                return compiled_model
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
                return self
        else:
            logger.warning("torch.compile not available. Skipping compilation.")
            return self
    
    def quantize_model(self, quantization_type: str = "dynamic") -> 'BatteryMindBaseModel':
        """
        Quantize model for reduced memory usage and faster inference.
        
        Args:
            quantization_type (str): Type of quantization ('dynamic', 'static', 'qat')
            
        Returns:
            BatteryMindBaseModel: Quantized model
        """
        try:
            if quantization_type == "dynamic":
                quantized_model = torch.quantization.quantize_dynamic(
                    self, {nn.Linear}, dtype=torch.qint8
                )
            elif quantization_type == "static":
                # Static quantization requires calibration data
                logger.warning("Static quantization requires calibration data. Using dynamic instead.")
                quantized_model = torch.quantization.quantize_dynamic(
                    self, {nn.Linear}, dtype=torch.qint8
                )
            else:
                raise ValueError(f"Unsupported quantization type: {quantization_type}")
            
            quantized_model._is_quantized = True
            logger.info(f"Model quantized using {quantization_type} quantization")
            return quantized_model
            
        except Exception as e:
            logger.error(f"Model quantization failed: {e}")
            return self
    
    def to_device(self, device: Union[str, torch.device]) -> 'BatteryMindBaseModel':
        """
        Move model to specified device with proper tracking.
        
        Args:
            device: Target device
            
        Returns:
            BatteryMindBaseModel: Self for method chaining
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        self._device = device
        super().to(device)
        logger.info(f"Model moved to device: {device}")
        return self
    
    def save_model(self, file_path: str, include_optimizer: bool = False,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """
        Save model with comprehensive metadata.
        
        Args:
            file_path (str): Path to save the model
            include_optimizer (bool): Whether to include optimizer state
            optimizer: Optimizer to save (if include_optimizer is True)
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'metadata': self.metadata.__dict__,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config),
            'model_info': self.get_model_info(),
            'pytorch_version': torch.__version__
        }
        
        if include_optimizer and optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(save_dict, file_path)
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path: str, config: Any, 
                   device: Optional[Union[str, torch.device]] = None) -> 'BatteryMindBaseModel':
        """
        Load model from file with metadata restoration.
        
        Args:
            file_path (str): Path to the saved model
            config: Model configuration
            device: Device to load the model on
            
        Returns:
            BatteryMindBaseModel: Loaded model instance
        """
        checkpoint = torch.load(file_path, map_location='cpu')
        
        # Restore metadata
        metadata_dict = checkpoint.get('metadata', {})
        metadata = ModelMetadata(**metadata_dict)
        
        # Create model instance
        model = cls(config, metadata)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device if specified
        if device is not None:
            model.to_device(device)
        
        logger.info(f"Model loaded from {file_path}")
        return model
    
    def add_forward_hook(self, hook_fn: callable) -> None:
        """Add forward hook for monitoring."""
        handle = self.register_forward_hook(hook_fn)
        self._forward_hooks.append(handle)
    
    def add_backward_hook(self, hook_fn: callable) -> None:
        """Add backward hook for monitoring."""
        handle = self.register_full_backward_hook(hook_fn)
        self._backward_hooks.append(handle)
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._forward_hooks + self._backward_hooks:
            handle.remove()
        self._forward_hooks.clear()
        self._backward_hooks.clear()
    
    def benchmark_inference(self, input_tensor: torch.Tensor, 
                          num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark model inference performance.
        
        Args:
            input_tensor (torch.Tensor): Input tensor for benchmarking
            num_runs (int): Number of benchmark runs
            warmup_runs (int): Number of warmup runs
            
        Returns:
            Dict[str, float]: Benchmark results
        """
        self.eval()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self(input_tensor)
        
        # Benchmark runs
        inference_times = []
        memory_usage = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                _ = self(input_tensor)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                inference_times.append((end_time - start_time) * 1000)  # Convert to ms
                
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.memory_allocated() / 1024 / 1024)  # MB
        
        # Update internal tracking
        self._inference_times.extend(inference_times)
        self._memory_usage.extend(memory_usage)
        
        results = {
            'avg_inference_time_ms': np.mean(inference_times),
            'std_inference_time_ms': np.std(inference_times),
            'min_inference_time_ms': np.min(inference_times),
            'max_inference_time_ms': np.max(inference_times),
            'throughput_samples_per_second': 1000 / np.mean(inference_times)
        }
        
        if memory_usage:
            results.update({
                'avg_memory_usage_mb': np.mean(memory_usage),
                'max_memory_usage_mb': np.max(memory_usage)
            })
        
        return results
    
    def validate_input(self, x: torch.Tensor) -> bool:
        """
        Validate input tensor shape and properties.
        
        Args:
            x (torch.Tensor): Input tensor to validate
            
        Returns:
            bool: True if input is valid
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D tensor, got {x.dim()}D")
        
        if torch.isnan(x).any():
            raise ValueError("Input tensor contains NaN values")
        
        if torch.isinf(x).any():
            raise ValueError("Input tensor contains infinite values")
        
        return True
    
    def get_attention_weights(self, x: torch.Tensor, layer_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights from transformer layers.
        
        Args:
            x (torch.Tensor): Input tensor
            layer_idx (int, optional): Specific layer index to extract from
            
        Returns:
            Dict[str, torch.Tensor]: Attention weights by layer
        """
        attention_weights = {}
        
        def attention_hook(module, input, output):
            if hasattr(output, 'attention_weights'):
                layer_name = f"layer_{len(attention_weights)}"
                attention_weights[layer_name] = output.attention_weights.detach()
        
        # Register hooks
        hooks = []
        for name, module in self.named_modules():
            if 'attention' in name.lower():
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def explain_prediction(self, x: torch.Tensor, method: str = "gradient") -> Dict[str, torch.Tensor]:
        """
        Generate explanation for model predictions.
        
        Args:
            x (torch.Tensor): Input tensor
            method (str): Explanation method ('gradient', 'integrated_gradient')
            
        Returns:
            Dict[str, torch.Tensor]: Explanation results
        """
        if method == "gradient":
            return self._gradient_explanation(x)
        elif method == "integrated_gradient":
            return self._integrated_gradient_explanation(x)
        else:
            raise ValueError(f"Unsupported explanation method: {method}")
    
    def _gradient_explanation(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate gradient-based explanations."""
        x.requires_grad_(True)
        
        output = self(x)
        
        # Assume the main output is in 'predictions' or first tensor value
        if isinstance(output, dict):
            main_output = output.get('predictions', list(output.values())[0])
        else:
            main_output = output
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=main_output.sum(),
            inputs=x,
            create_graph=False,
            retain_graph=False
        )[0]
        
        return {
            'input_gradients': gradients,
            'gradient_magnitude': torch.abs(gradients),
            'gradient_norm': torch.norm(gradients, dim=-1)
        }
    
    def _integrated_gradient_explanation(self, x: torch.Tensor, steps: int = 50) -> Dict[str, torch.Tensor]:
        """Generate integrated gradient explanations."""
        baseline = torch.zeros_like(x)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps, device=x.device)
        interpolated_inputs = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (x - baseline)
            interpolated_inputs.append(interpolated)
        
        # Calculate gradients for each interpolated input
        gradients = []
        for interpolated in interpolated_inputs:
            interpolated.requires_grad_(True)
            output = self(interpolated)
            
            if isinstance(output, dict):
                main_output = output.get('predictions', list(output.values())[0])
            else:
                main_output = output
            
            grad = torch.autograd.grad(
                outputs=main_output.sum(),
                inputs=interpolated,
                create_graph=False,
                retain_graph=False
            )[0]
            gradients.append(grad)
        
        # Integrate gradients
        integrated_gradients = torch.stack(gradients).mean(dim=0) * (x - baseline)
        
        return {
            'integrated_gradients': integrated_gradients,
            'attribution_magnitude': torch.abs(integrated_gradients),
            'attribution_norm': torch.norm(integrated_gradients, dim=-1)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of the model.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        health_status = {
            'model_loaded': True,
            'device': str(self._device),
            'parameters_finite': True,
            'gradients_finite': True,
            'memory_usage_mb': 0.0,
            'inference_test': False
        }
        
        try:
            # Check parameter finiteness
            for param in self.parameters():
                if not torch.isfinite(param).all():
                    health_status['parameters_finite'] = False
                    break
            
            # Check gradient finiteness (if available)
            for param in self.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    health_status['gradients_finite'] = False
                    break
            
            # Memory usage
            if torch.cuda.is_available() and self._device.type == 'cuda':
                health_status['memory_usage_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            
            # Simple inference test
            dummy_input = torch.randn(1, 10, getattr(self.config, 'feature_dim', 16))
            dummy_input = dummy_input.to(self._device)
            
            with torch.no_grad():
                _ = self(dummy_input)
            health_status['inference_test'] = True
            
        except Exception as e:
            health_status['error'] = str(e)
            health_status['inference_test'] = False
        
        return health_status

class ModelRegistry:
    """
    Registry for managing multiple BatteryMind models.
    """
    
    def __init__(self):
        self._models = {}
        self._metadata = {}
    
    def register_model(self, name: str, model: BatteryMindBaseModel) -> None:
        """Register a model in the registry."""
        self._models[name] = model
        self._metadata[name] = model.metadata
        logger.info(f"Registered model: {name}")
    
    def get_model(self, name: str) -> Optional[BatteryMindBaseModel]:
        """Get a model from the registry."""
        return self._models.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self._models.keys())
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered model."""
        model = self._models.get(name)
        return model.get_model_info() if model else None
    
    def remove_model(self, name: str) -> bool:
        """Remove a model from the registry."""
        if name in self._models:
            del self._models[name]
            del self._metadata[name]
            logger.info(f"Removed model: {name}")
            return True
        return False

# Global model registry instance
model_registry = ModelRegistry()

# Utility functions
def create_model_from_config(model_class: type, config: Any, 
                           metadata: Optional[ModelMetadata] = None) -> BatteryMindBaseModel:
    """
    Factory function to create models from configuration.
    
    Args:
        model_class: Model class to instantiate
        config: Model configuration
        metadata: Optional metadata
        
    Returns:
        BatteryMindBaseModel: Created model instance
    """
    if not issubclass(model_class, BatteryMindBaseModel):
        raise TypeError(f"Model class must inherit from BatteryMindBaseModel")
    
    return model_class(config, metadata)

def compare_models(models: List[BatteryMindBaseModel], 
                  input_tensor: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """
    Compare performance of multiple models.
    
    Args:
        models: List of models to compare
        input_tensor: Input tensor for benchmarking
        
    Returns:
        Dict[str, Dict[str, float]]: Comparison results
    """
    results = {}
    
    for i, model in enumerate(models):
        model_name = f"model_{i}_{model.__class__.__name__}"
        benchmark_results = model.benchmark_inference(input_tensor)
        model_info = model.get_model_info()
        
        results[model_name] = {
            **benchmark_results,
            'parameters': model_info['architecture']['total_parameters'],
            'model_size_mb': model_info['architecture']['model_size_mb']
        }
    
    return results

def validate_model_compatibility(model1: BatteryMindBaseModel, 
                                model2: BatteryMindBaseModel) -> Dict[str, bool]:
    """
    Check compatibility between two models for ensemble or comparison.
    
    Args:
        model1: First model
        model2: Second model
        
    Returns:
        Dict[str, bool]: Compatibility check results
    """
    compatibility = {
        'same_input_shape': False,
        'same_output_shape': False,
        'same_device': False,
        'compatible_configs': False
    }
    
    # Check input/output shapes
    if hasattr(model1.config, 'feature_dim') and hasattr(model2.config, 'feature_dim'):
        compatibility['same_input_shape'] = model1.config.feature_dim == model2.config.feature_dim
    
    if hasattr(model1.config, 'target_dim') and hasattr(model2.config, 'target_dim'):
        compatibility['same_output_shape'] = model1.config.target_dim == model2.config.target_dim
    
    # Check device compatibility
    compatibility['same_device'] = model1._device == model2._device
    
    # Check config compatibility (basic check)
    config1_dict = model1.config.__dict__ if hasattr(model1.config, '__dict__') else {}
    config2_dict = model2.config.__dict__ if hasattr(model2.config, '__dict__') else {}
    
    common_keys = set(config1_dict.keys()) & set(config2_dict.keys())
    if common_keys:
        compatible_values = sum(1 for key in common_keys if config1_dict[key] == config2_dict[key])
        compatibility['compatible_configs'] = compatible_values / len(common_keys) > 0.8
    
    return compatibility
