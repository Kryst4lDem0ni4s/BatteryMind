"""
BatteryMind - Common Transformer Utilities

Comprehensive utility functions and helper classes for transformer models
with specialized support for battery health prediction, degradation forecasting,
and optimization tasks.

Features:
- Attention pattern analysis and visualization
- Model initialization and weight management
- Sequence processing and batching utilities
- Performance optimization helpers
- Model interpretability tools
- Integration utilities for multi-model systems

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Scientific computing
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionAnalyzer:
    """
    Utility class for analyzing and visualizing attention patterns in transformer models.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attention_hooks = []
        self.attention_maps = defaultdict(list)
    
    def register_attention_hooks(self) -> None:
        """Register hooks to capture attention weights during forward pass."""
        def attention_hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attention_weights = output[1]  # Attention weights
                layer_name = self._get_layer_name(module)
                self.attention_maps[layer_name].append(attention_weights.detach().cpu())
        
        # Register hooks for all attention modules
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(attention_hook)
                self.attention_hooks.append(hook)
                logger.info(f"Registered attention hook for layer: {name}")
    
    def remove_attention_hooks(self) -> None:
        """Remove all registered attention hooks."""
        for hook in self.attention_hooks:
            hook.remove()
        self.attention_hooks.clear()
        self.attention_maps.clear()
    
    def _get_layer_name(self, module: nn.Module) -> str:
        """Get a descriptive name for the module."""
        for name, mod in self.model.named_modules():
            if mod is module:
                return name
        return "unknown_layer"
    
    def analyze_attention_patterns(self, input_data: torch.Tensor, 
                                 input_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze attention patterns for given input data.
        
        Args:
            input_data (torch.Tensor): Input data for analysis
            input_labels (List[str], optional): Labels for input positions
            
        Returns:
            Dict[str, Any]: Analysis results including statistics and patterns
        """
        self.model.eval()
        self.attention_maps.clear()
        
        with torch.no_grad():
            # Forward pass to capture attention
            _ = self.model(input_data.to(self.device))
        
        analysis_results = {}
        
        for layer_name, attention_list in self.attention_maps.items():
            if not attention_list:
                continue
            
            # Combine attention maps from all batches
            attention_tensor = torch.cat(attention_list, dim=0)
            
            # Analyze attention patterns
            layer_analysis = self._analyze_layer_attention(attention_tensor, input_labels)
            analysis_results[layer_name] = layer_analysis
        
        return analysis_results
    
    def _analyze_layer_attention(self, attention_tensor: torch.Tensor, 
                                input_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze attention patterns for a specific layer."""
        batch_size, num_heads, seq_len, _ = attention_tensor.shape
        
        analysis = {
            'shape': attention_tensor.shape,
            'mean_attention': attention_tensor.mean().item(),
            'std_attention': attention_tensor.std().item(),
            'max_attention': attention_tensor.max().item(),
            'min_attention': attention_tensor.min().item()
        }
        
        # Attention entropy (measure of focus)
        attention_entropy = -torch.sum(attention_tensor * torch.log(attention_tensor + 1e-8), dim=-1)
        analysis['mean_entropy'] = attention_entropy.mean().item()
        analysis['std_entropy'] = attention_entropy.std().item()
        
        # Attention sparsity (percentage of low attention weights)
        sparsity_threshold = 0.1
        sparse_mask = attention_tensor < sparsity_threshold
        analysis['sparsity_ratio'] = sparse_mask.float().mean().item()
        
        # Head diversity (how different attention heads are)
        head_similarities = []
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                similarity = F.cosine_similarity(
                    attention_tensor[:, i, :, :].flatten(),
                    attention_tensor[:, j, :, :].flatten(),
                    dim=0
                )
                head_similarities.append(similarity.item())
        
        analysis['head_diversity'] = 1.0 - np.mean(head_similarities) if head_similarities else 0.0
        
        # Positional attention bias
        position_attention = attention_tensor.mean(dim=(0, 1))  # Average across batch and heads
        diagonal_attention = torch.diag(position_attention).mean().item()
        analysis['diagonal_bias'] = diagonal_attention
        
        return analysis
    
    def visualize_attention(self, layer_name: str, head_idx: int = 0, 
                          sample_idx: int = 0, save_path: Optional[str] = None) -> None:
        """
        Visualize attention patterns for a specific layer and head.
        
        Args:
            layer_name (str): Name of the layer to visualize
            head_idx (int): Index of the attention head
            sample_idx (int): Index of the sample in batch
            save_path (str, optional): Path to save the visualization
        """
        if layer_name not in self.attention_maps:
            logger.warning(f"No attention data found for layer: {layer_name}")
            return
        
        attention_data = self.attention_maps[layer_name][0]  # First batch
        
        if sample_idx >= attention_data.shape[0] or head_idx >= attention_data.shape[1]:
            logger.warning("Invalid sample or head index")
            return
        
        # Extract attention matrix for visualization
        attention_matrix = attention_data[sample_idx, head_idx].numpy()
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_matrix, cmap='Blues', cbar=True, square=True)
        plt.title(f'Attention Pattern - {layer_name} (Head {head_idx})')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

class ModelInitializer:
    """
    Utility class for advanced model initialization strategies.
    """
    
    @staticmethod
    def xavier_uniform_init(module: nn.Module, gain: float = 1.0) -> None:
        """Apply Xavier uniform initialization."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    @staticmethod
    def xavier_normal_init(module: nn.Module, gain: float = 1.0) -> None:
        """Apply Xavier normal initialization."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_normal_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    @staticmethod
    def kaiming_uniform_init(module: nn.Module, mode: str = 'fan_in', 
                           nonlinearity: str = 'relu') -> None:
        """Apply Kaiming uniform initialization."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=nonlinearity)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    @staticmethod
    def transformer_init(module: nn.Module, d_model: int) -> None:
        """Apply transformer-specific initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    @staticmethod
    def battery_specific_init(module: nn.Module) -> None:
        """Apply battery-specific initialization for better convergence."""
        if isinstance(module, nn.Linear):
            # Smaller initialization for battery prediction stability
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

class SequenceProcessor:
    """
    Utility class for processing sequences in transformer models.
    """
    
    @staticmethod
    def create_padding_mask(sequences: torch.Tensor, padding_value: float = 0.0) -> torch.Tensor:
        """
        Create padding mask for variable-length sequences.
        
        Args:
            sequences (torch.Tensor): Input sequences
            padding_value (float): Value used for padding
            
        Returns:
            torch.Tensor: Padding mask (True for valid positions)
        """
        return (sequences != padding_value).any(dim=-1)
    
    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
        """
        Create causal (lower triangular) mask for autoregressive models.
        
        Args:
            seq_len (int): Sequence length
            device (torch.device, optional): Device for the mask
            
        Returns:
            torch.Tensor: Causal mask
        """
        device = device or torch.device('cpu')
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0
    
    @staticmethod
    def create_look_ahead_mask(size: int, device: torch.device = None) -> torch.Tensor:
        """
        Create look-ahead mask for preventing future information leakage.
        
        Args:
            size (int): Size of the mask
            device (torch.device, optional): Device for the mask
            
        Returns:
            torch.Tensor: Look-ahead mask
        """
        device = device or torch.device('cpu')
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask == 0
    
    @staticmethod
    def pad_sequences(sequences: List[torch.Tensor], padding_value: float = 0.0,
                     max_length: Optional[int] = None) -> torch.Tensor:
        """
        Pad sequences to the same length.
        
        Args:
            sequences (List[torch.Tensor]): List of sequences to pad
            padding_value (float): Value to use for padding
            max_length (int, optional): Maximum length (uses longest sequence if None)
            
        Returns:
            torch.Tensor: Padded sequences
        """
        if not sequences:
            return torch.empty(0)
        
        # Determine maximum length
        if max_length is None:
            max_length = max(seq.size(0) for seq in sequences)
        
        # Pad sequences
        padded_sequences = []
        for seq in sequences:
            seq_len = seq.size(0)
            if seq_len < max_length:
                padding = torch.full((max_length - seq_len,) + seq.shape[1:], 
                                   padding_value, dtype=seq.dtype, device=seq.device)
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq[:max_length]
            padded_sequences.append(padded_seq)
        
        return torch.stack(padded_sequences)
    
    @staticmethod
    def create_sliding_windows(sequence: torch.Tensor, window_size: int, 
                             stride: int = 1) -> torch.Tensor:
        """
        Create sliding windows from a sequence.
        
        Args:
            sequence (torch.Tensor): Input sequence
            window_size (int): Size of each window
            stride (int): Stride between windows
            
        Returns:
            torch.Tensor: Sliding windows
        """
        seq_len = sequence.size(0)
        num_windows = (seq_len - window_size) // stride + 1
        
        windows = []
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            windows.append(sequence[start_idx:end_idx])
        
        return torch.stack(windows) if windows else torch.empty(0, window_size, *sequence.shape[1:])

class PerformanceOptimizer:
    """
    Utility class for optimizing transformer model performance.
    """
    
    @staticmethod
    def enable_gradient_checkpointing(model: nn.Module) -> None:
        """Enable gradient checkpointing to reduce memory usage."""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        else:
            logger.warning("Model does not support gradient checkpointing")
    
    @staticmethod
    def optimize_for_inference(model: nn.Module) -> nn.Module:
        """Optimize model for inference by fusing operations and enabling optimizations."""
        model.eval()
        
        # Fuse batch normalization and convolution layers
        try:
            model = torch.jit.optimize_for_inference(torch.jit.script(model))
        except Exception as e:
            logger.warning(f"Could not apply JIT optimization: {e}")
        
        return model
    
    @staticmethod
    def calculate_model_size(model: nn.Module) -> Dict[str, float]:
        """
        Calculate model size and parameter statistics.
        
        Args:
            model (nn.Module): Model to analyze
            
        Returns:
            Dict[str, float]: Model size statistics
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate memory usage (assuming float32)
        param_size_mb = total_params * 4 / (1024 * 1024)
        
        # Estimate activation memory (rough approximation)
        sample_input = torch.randn(1, 100, 512)  # Typical input size
        try:
            with torch.no_grad():
                _ = model(sample_input)
            # This is a very rough estimate
            activation_size_mb = sample_input.numel() * 4 / (1024 * 1024) * 10
        except:
            activation_size_mb = 0.0
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_size_mb': param_size_mb,
            'estimated_activation_size_mb': activation_size_mb,
            'estimated_total_size_mb': param_size_mb + activation_size_mb
        }
    
    @staticmethod
    def profile_model_speed(model: nn.Module, input_tensor: torch.Tensor, 
                          num_runs: int = 100) -> Dict[str, float]:
        """
        Profile model inference speed.
        
        Args:
            model (nn.Module): Model to profile
            input_tensor (torch.Tensor): Sample input
            num_runs (int): Number of inference runs
            
        Returns:
            Dict[str, float]: Speed profiling results
        """
        model.eval()
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Synchronize GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Time inference
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = 1.0 / avg_time
        
        return {
            'total_time_seconds': total_time,
            'average_time_seconds': avg_time,
            'throughput_samples_per_second': throughput,
            'num_runs': num_runs
        }

class ModelInterpretability:
    """
    Utility class for model interpretability and explainability.
    """
    
    @staticmethod
    def compute_gradient_attribution(model: nn.Module, input_tensor: torch.Tensor,
                                   target_class: Optional[int] = None) -> torch.Tensor:
        """
        Compute gradient-based attribution scores.
        
        Args:
            model (nn.Module): Model to analyze
            input_tensor (torch.Tensor): Input tensor
            target_class (int, optional): Target class for attribution
            
        Returns:
            torch.Tensor: Attribution scores
        """
        input_tensor.requires_grad_(True)
        model.eval()
        
        output = model(input_tensor)
        
        if target_class is None:
            # For regression or when target class is not specified
            if output.dim() > 1 and output.size(-1) > 1:
                target_class = output.argmax(dim=-1)
                score = output.gather(-1, target_class.unsqueeze(-1)).squeeze(-1)
            else:
                score = output.squeeze()
        else:
            score = output[..., target_class]
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=score.sum(),
            inputs=input_tensor,
            create_graph=False,
            retain_graph=False
        )[0]
        
        return gradients
    
    @staticmethod
    def compute_integrated_gradients(model: nn.Module, input_tensor: torch.Tensor,
                                   baseline: Optional[torch.Tensor] = None,
                                   steps: int = 50) -> torch.Tensor:
        """
        Compute integrated gradients for attribution.
        
        Args:
            model (nn.Module): Model to analyze
            input_tensor (torch.Tensor): Input tensor
            baseline (torch.Tensor, optional): Baseline for integration
            steps (int): Number of integration steps
            
        Returns:
            torch.Tensor: Integrated gradients
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps, device=input_tensor.device)
        interpolated_inputs = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated_inputs.append(interpolated)
        
        # Compute gradients for all interpolated inputs
        gradients = []
        for interpolated in interpolated_inputs:
            grad = ModelInterpretability.compute_gradient_attribution(model, interpolated)
            gradients.append(grad)
        
        # Average gradients and multiply by input difference
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = (input_tensor - baseline) * avg_gradients
        
        return integrated_gradients
    
    @staticmethod
    def analyze_feature_importance(model: nn.Module, input_tensor: torch.Tensor,
                                 feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Analyze feature importance using gradient-based methods.
        
        Args:
            model (nn.Module): Model to analyze
            input_tensor (torch.Tensor): Input tensor
            feature_names (List[str], optional): Names of features
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        gradients = ModelInterpretability.compute_gradient_attribution(model, input_tensor)
        
        # Compute importance as absolute gradient magnitude
        importance_scores = torch.abs(gradients).mean(dim=0)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(importance_scores.size(-1))]
        
        # Create importance dictionary
        importance_dict = {}
        for i, name in enumerate(feature_names):
            if i < importance_scores.size(-1):
                importance_dict[name] = importance_scores[i].item()
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        return importance_dict

class ModelEnsembleUtils:
    """
    Utility functions for working with ensemble models.
    """
    
    @staticmethod
    def compute_prediction_diversity(predictions: List[torch.Tensor]) -> float:
        """
        Compute diversity among ensemble predictions.
        
        Args:
            predictions (List[torch.Tensor]): List of predictions from different models
            
        Returns:
            float: Diversity score (higher is more diverse)
        """
        if len(predictions) < 2:
            return 0.0
        
        # Stack predictions
        stacked_predictions = torch.stack(predictions)
        
        # Compute pairwise correlations
        correlations = []
        num_models = len(predictions)
        
        for i in range(num_models):
            for j in range(i + 1, num_models):
                pred_i = stacked_predictions[i].flatten()
                pred_j = stacked_predictions[j].flatten()
                
                correlation = F.cosine_similarity(pred_i, pred_j, dim=0)
                correlations.append(correlation.item())
        
        # Diversity is 1 - average correlation
        avg_correlation = np.mean(correlations) if correlations else 0.0
        diversity = 1.0 - avg_correlation
        
        return diversity
    
    @staticmethod
    def weighted_ensemble_prediction(predictions: List[torch.Tensor], 
                                   weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        Compute weighted ensemble prediction.
        
        Args:
            predictions (List[torch.Tensor]): List of predictions
            weights (List[float], optional): Weights for each prediction
            
        Returns:
            torch.Tensor: Weighted ensemble prediction
        """
        if not predictions:
            raise ValueError("No predictions provided")
        
        if weights is None:
            weights = [1.0 / len(predictions)] * len(predictions)
        
        if len(weights) != len(predictions):
            raise ValueError("Number of weights must match number of predictions")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Compute weighted average
        weighted_pred = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            weighted_pred += weight * pred
        
        return weighted_pred

# Utility functions for common operations
def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model (nn.Module): Model to analyze
        
    Returns:
        Dict[str, int]: Parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

def freeze_model_parameters(model: nn.Module, freeze_embeddings: bool = True,
                          freeze_encoder: bool = False, freeze_decoder: bool = False) -> None:
    """
    Freeze specific parts of a model.
    
    Args:
        model (nn.Module): Model to modify
        freeze_embeddings (bool): Whether to freeze embedding layers
        freeze_encoder (bool): Whether to freeze encoder layers
        freeze_decoder (bool): Whether to freeze decoder layers
    """
    for name, param in model.named_parameters():
        if freeze_embeddings and ('embedding' in name.lower() or 'embed' in name.lower()):
            param.requires_grad = False
        elif freeze_encoder and 'encoder' in name.lower():
            param.requires_grad = False
        elif freeze_decoder and 'decoder' in name.lower():
            param.requires_grad = False

def get_model_device(model: nn.Module) -> torch.device:
    """
    Get the device of a model.
    
    Args:
        model (nn.Module): Model to check
        
    Returns:
        torch.device: Device of the model
    """
    return next(model.parameters()).device

def move_to_device(data: Union[torch.Tensor, Dict, List], device: torch.device) -> Union[torch.Tensor, Dict, List]:
    """
    Move data to specified device.
    
    Args:
        data: Data to move (tensor, dict, or list)
        device: Target device
        
    Returns:
        Data moved to the specified device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    else:
        return data

def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, 
                epsilon: float = 1e-8) -> torch.Tensor:
    """
    Perform safe division with epsilon to avoid division by zero.
    
    Args:
        numerator (torch.Tensor): Numerator tensor
        denominator (torch.Tensor): Denominator tensor
        epsilon (float): Small value to add to denominator
        
    Returns:
        torch.Tensor: Result of safe division
    """
    return numerator / (denominator + epsilon)

def apply_temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply temperature scaling to logits for calibration.
    
    Args:
        logits (torch.Tensor): Input logits
        temperature (float): Temperature parameter
        
    Returns:
        torch.Tensor: Temperature-scaled logits
    """
    return logits / temperature

def compute_model_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """
    Estimate FLOPs for a model (rough approximation).
    
    Args:
        model (nn.Module): Model to analyze
        input_shape (Tuple[int, ...]): Shape of input tensor
        
    Returns:
        int: Estimated FLOPs
    """
    total_flops = 0
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # FLOPs for linear layer: input_features * output_features * batch_size
            total_flops += module.in_features * module.out_features
        elif isinstance(module, nn.Conv1d):
            # Simplified FLOP calculation for 1D convolution
            kernel_flops = module.kernel_size[0] * module.in_channels * module.out_channels
            total_flops += kernel_flops
        elif isinstance(module, nn.MultiheadAttention):
            # Simplified attention FLOP calculation
            embed_dim = module.embed_dim
            total_flops += embed_dim * embed_dim * 3  # Q, K, V projections
    
    return total_flops

# Export all utility classes and functions
__all__ = [
    'AttentionAnalyzer',
    'ModelInitializer', 
    'SequenceProcessor',
    'PerformanceOptimizer',
    'ModelInterpretability',
    'ModelEnsembleUtils',
    'count_parameters',
    'freeze_model_parameters',
    'get_model_device',
    'move_to_device',
    'safe_divide',
    'apply_temperature_scaling',
    'compute_model_flops'
]
