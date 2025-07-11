"""
BatteryMind - Common Transformer Utilities

Shared components and utilities for all transformer models in the BatteryMind system.
Provides base classes, attention mechanisms, positional encodings, and utility functions
that ensure consistency and reusability across different transformer architectures.

Key Components:
- BaseTransformerModel: Abstract base class for all transformer models
- MultiHeadAttention: Advanced attention mechanisms with battery-specific optimizations
- PositionalEncoding: Various positional encoding strategies for time-series data
- TransformerUtils: Utility functions for model operations and optimizations
- AttentionVisualization: Tools for visualizing and interpreting attention patterns
- ModelRegistry: Centralized model management and versioning system
- ConfigManager: Configuration management and validation utilities

Features:
- Standardized interfaces across all transformer models
- Physics-informed attention mechanisms for battery applications
- Advanced positional encodings for temporal data
- Comprehensive model utilities and helper functions
- Attention pattern visualization and interpretation
- Centralized configuration management
- Model versioning and registry system

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

from .base_model import (
    BaseTransformerModel,
    BaseTransformerConfig,
    ModelInterface,
    TransformerMixin,
    PhysicsInformedMixin
)

from .attention_layers import (
    MultiHeadAttention,
    BatterySpecificAttention,
    TemporalAttention,
    SpatialAttention,
    CrossModalAttention,
    SelfAttention,
    AttentionMask,
    AttentionWeights
)

from .positional_encoding import (
    PositionalEncoding,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    RotaryPositionalEncoding,
    RelativePositionalEncoding,
    TemporalPositionalEncoding,
    BatteryPositionalEncoding
)

from .transformer_utils import (
    TransformerUtils,
    ModelOptimizer,
    GradientClipping,
    LearningRateScheduler,
    ModelCheckpointing,
    EarlyStopping,
    ModelCompression,
    InferenceOptimizer
)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module metadata
__all__ = [
    # Base model components
    "BaseTransformerModel",
    "BaseTransformerConfig",
    "ModelInterface",
    "TransformerMixin",
    "PhysicsInformedMixin",
    
    # Attention mechanisms
    "MultiHeadAttention",
    "BatterySpecificAttention",
    "TemporalAttention",
    "SpatialAttention",
    "CrossModalAttention",
    "SelfAttention",
    "AttentionMask",
    "AttentionWeights",
    
    # Positional encodings
    "PositionalEncoding",
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding",
    "RotaryPositionalEncoding",
    "RelativePositionalEncoding",
    "TemporalPositionalEncoding",
    "BatteryPositionalEncoding",
    
    # Utility functions
    "TransformerUtils",
    "ModelOptimizer",
    "GradientClipping",
    "LearningRateScheduler",
    "ModelCheckpointing",
    "EarlyStopping",
    "ModelCompression",
    "InferenceOptimizer"
]

# Common configuration for all transformer utilities
COMMON_CONFIG = {
    "attention": {
        "dropout": 0.1,
        "temperature": 1.0,
        "use_bias": True,
        "scale_attention": True,
        "attention_type": "scaled_dot_product"
    },
    "positional_encoding": {
        "max_length": 2048,
        "encoding_type": "sinusoidal",
        "learnable": False,
        "dropout": 0.1
    },
    "optimization": {
        "gradient_clip_norm": 1.0,
        "weight_decay": 0.01,
        "warmup_steps": 4000,
        "lr_scheduler": "cosine_with_warmup"
    },
    "physics_constraints": {
        "enable_constraints": True,
        "temperature_bounds": [-20.0, 60.0],
        "voltage_bounds": [2.5, 4.2],
        "current_bounds": [-100.0, 100.0],
        "degradation_bounds": [0.0, 0.01]
    }
}

def get_common_config():
    """
    Get common configuration for transformer utilities.
    
    Returns:
        dict: Common configuration dictionary
    """
    return COMMON_CONFIG.copy()

def create_attention_layer(attention_type="multi_head", **kwargs):
    """
    Factory function to create attention layers.
    
    Args:
        attention_type (str): Type of attention mechanism
        **kwargs: Additional arguments for attention layer
        
    Returns:
        nn.Module: Attention layer instance
    """
    if attention_type == "multi_head":
        return MultiHeadAttention(**kwargs)
    elif attention_type == "battery_specific":
        return BatterySpecificAttention(**kwargs)
    elif attention_type == "temporal":
        return TemporalAttention(**kwargs)
    elif attention_type == "spatial":
        return SpatialAttention(**kwargs)
    elif attention_type == "cross_modal":
        return CrossModalAttention(**kwargs)
    elif attention_type == "self":
        return SelfAttention(**kwargs)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")

def create_positional_encoding(encoding_type="sinusoidal", **kwargs):
    """
    Factory function to create positional encoding layers.
    
    Args:
        encoding_type (str): Type of positional encoding
        **kwargs: Additional arguments for positional encoding
        
    Returns:
        nn.Module: Positional encoding layer instance
    """
    if encoding_type == "sinusoidal":
        return SinusoidalPositionalEncoding(**kwargs)
    elif encoding_type == "learned":
        return LearnedPositionalEncoding(**kwargs)
    elif encoding_type == "rotary":
        return RotaryPositionalEncoding(**kwargs)
    elif encoding_type == "relative":
        return RelativePositionalEncoding(**kwargs)
    elif encoding_type == "temporal":
        return TemporalPositionalEncoding(**kwargs)
    elif encoding_type == "battery":
        return BatteryPositionalEncoding(**kwargs)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")

def validate_transformer_config(config):
    """
    Validate transformer configuration for consistency and correctness.
    
    Args:
        config (dict): Configuration to validate
        
    Returns:
        dict: Validation results with any issues found
    """
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": []
    }
    
    # Check required fields
    required_fields = ["d_model", "n_heads", "n_layers"]
    for field in required_fields:
        if field not in config:
            validation_results["errors"].append(f"Missing required field: {field}")
    
    # Check value constraints
    if "d_model" in config and config["d_model"] % config.get("n_heads", 1) != 0:
        validation_results["errors"].append("d_model must be divisible by n_heads")
    
    if "dropout" in config and not (0.0 <= config["dropout"] <= 1.0):
        validation_results["errors"].append("dropout must be between 0.0 and 1.0")
    
    # Check physics constraints if enabled
    if config.get("use_physics_constraints", False):
        physics_config = config.get("physics_constraints", {})
        if "temperature_bounds" in physics_config:
            temp_bounds = physics_config["temperature_bounds"]
            if temp_bounds[0] >= temp_bounds[1]:
                validation_results["errors"].append("Invalid temperature bounds")
    
    # Set validation status
    validation_results["valid"] = len(validation_results["errors"]) == 0
    
    return validation_results

def optimize_attention_computation(attention_scores, optimization_type="memory"):
    """
    Optimize attention computation for memory or speed.
    
    Args:
        attention_scores (torch.Tensor): Attention scores to optimize
        optimization_type (str): Type of optimization ("memory" or "speed")
        
    Returns:
        torch.Tensor: Optimized attention scores
    """
    import torch
    
    if optimization_type == "memory":
        # Use gradient checkpointing for memory efficiency
        if hasattr(torch.utils.checkpoint, 'checkpoint'):
            return torch.utils.checkpoint.checkpoint(
                lambda x: torch.softmax(x, dim=-1), attention_scores
            )
    elif optimization_type == "speed":
        # Use optimized attention implementations if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized attention (PyTorch 2.0+)
            return torch.nn.functional.scaled_dot_product_attention(
                attention_scores, attention_scores, attention_scores
            )
    
    # Fallback to standard computation
    return torch.softmax(attention_scores, dim=-1)

def get_model_complexity(model):
    """
    Calculate model complexity metrics.
    
    Args:
        model (torch.nn.Module): Model to analyze
        
    Returns:
        dict: Complexity metrics
    """
    import torch
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate FLOPs (simplified calculation)
    flops = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            flops += module.in_features * module.out_features
        elif isinstance(module, torch.nn.Conv1d):
            flops += (module.in_channels * module.out_channels * 
                     module.kernel_size[0] * module.stride[0])
    
    complexity_metrics = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        "estimated_flops": flops,
        "memory_footprint_mb": total_params * 4 / (1024 * 1024) * 2,  # Rough estimate
        "parameter_efficiency": trainable_params / total_params if total_params > 0 else 0
    }
    
    return complexity_metrics

def benchmark_attention_performance(attention_layer, input_size, batch_size=32, num_iterations=100):
    """
    Benchmark attention layer performance.
    
    Args:
        attention_layer (torch.nn.Module): Attention layer to benchmark
        input_size (tuple): Input tensor size (seq_len, d_model)
        batch_size (int): Batch size for benchmarking
        num_iterations (int): Number of iterations for averaging
        
    Returns:
        dict: Performance benchmarking results
    """
    import torch
    import time
    
    device = next(attention_layer.parameters()).device
    seq_len, d_model = input_size
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = attention_layer(dummy_input, dummy_input, dummy_input)
    
    # Benchmark forward pass
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = attention_layer(dummy_input, dummy_input, dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_forward_time = (end_time - start_time) / num_iterations
    
    # Benchmark backward pass
    attention_layer.train()
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    for _ in range(num_iterations):
        dummy_input.requires_grad_(True)
        output = attention_layer(dummy_input, dummy_input, dummy_input)
        loss = output[0].sum() if isinstance(output, tuple) else output.sum()
        loss.backward()
        dummy_input.grad = None
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_backward_time = (end_time - start_time) / num_iterations
    
    return {
        "avg_forward_time_ms": avg_forward_time * 1000,
        "avg_backward_time_ms": avg_backward_time * 1000,
        "total_time_ms": (avg_forward_time + avg_backward_time) * 1000,
        "throughput_samples_per_sec": batch_size / (avg_forward_time + avg_backward_time),
        "input_size": input_size,
        "batch_size": batch_size,
        "device": str(device)
    }

# Utility functions for common transformer operations
def apply_attention_mask(attention_scores, mask):
    """Apply attention mask to attention scores."""
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
    return attention_scores

def compute_attention_weights(query, key, scale=None, mask=None):
    """Compute attention weights from query and key tensors."""
    import torch
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Apply scaling
    if scale is not None:
        scores = scores * scale
    
    # Apply mask
    scores = apply_attention_mask(scores, mask)
    
    # Apply softmax
    attention_weights = torch.softmax(scores, dim=-1)
    
    return attention_weights

def create_causal_mask(seq_len, device=None):
    """Create causal attention mask for autoregressive models."""
    import torch
    
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask == 0

def create_padding_mask(lengths, max_len=None, device=None):
    """Create padding mask from sequence lengths."""
    import torch
    
    if max_len is None:
        max_len = max(lengths)
    
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.bool)
    
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    return mask

# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BatteryMind Common Transformer Utilities v{__version__} initialized")

# Health check for common utilities
def health_check():
    """
    Perform health check of common utilities.
    
    Returns:
        dict: Health check results
    """
    health_status = {
        "status": "healthy",
        "version": __version__,
        "utilities_available": True,
        "torch_available": False,
        "issues": []
    }
    
    try:
        import torch
        health_status["torch_available"] = True
        
        # Test basic tensor operations
        test_tensor = torch.randn(2, 4, 8)
        attention_weights = compute_attention_weights(test_tensor, test_tensor)
        health_status["attention_computation"] = True
        
    except Exception as e:
        health_status["attention_computation"] = False
        health_status["issues"].append(f"Attention computation test failed: {str(e)}")
    
    # Determine overall status
    if health_status["issues"]:
        health_status["status"] = "degraded"
    
    return health_status
