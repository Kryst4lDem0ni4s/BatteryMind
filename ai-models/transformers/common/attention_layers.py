"""
BatteryMind - Advanced Attention Layers

Comprehensive collection of attention mechanisms optimized for battery-related
time-series data and multi-modal sensor fusion. Provides specialized attention
variants for different aspects of battery health monitoring and prediction.

Features:
- Multi-head attention with battery-specific optimizations
- Temporal attention for time-series data
- Cross-modal attention for sensor fusion
- Sparse attention for long sequences
- Relative position encoding
- Attention visualization utilities

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AttentionConfig:
    """
    Configuration for attention mechanisms.
    
    Attributes:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        dropout (float): Dropout probability
        use_bias (bool): Whether to use bias in linear layers
        attention_dropout (float): Attention-specific dropout
        scale_factor (Optional[float]): Custom scaling factor
        max_relative_position (int): Maximum relative position for relative encoding
        use_relative_position (bool): Whether to use relative position encoding
        temperature (float): Temperature for attention softmax
    """
    d_model: int = 512
    n_heads: int = 8
    dropout: float = 0.1
    use_bias: bool = True
    attention_dropout: float = 0.1
    scale_factor: Optional[float] = None
    max_relative_position: int = 128
    use_relative_position: bool = False
    temperature: float = 1.0

class MultiHeadAttention(nn.Module):
    """
    Enhanced multi-head attention with battery-specific optimizations.
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        
        self.config = config
        self.d_k = config.d_model // config.n_heads
        self.scale = config.scale_factor or (1.0 / math.sqrt(self.d_k))
        
        # Linear projections
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        
        # Dropout layers
        self.dropout = nn.Dropout(config.dropout)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
        # Relative position encoding
        if config.use_relative_position:
            self.relative_position_k = nn.Parameter(
                torch.randn(2 * config.max_relative_position + 1, self.d_k)
            )
            self.relative_position_v = nn.Parameter(
                torch.randn(2 * config.max_relative_position + 1, self.d_k)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """Generate relative position indices."""
        positions = torch.arange(seq_len, dtype=torch.long)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = torch.clamp(
            relative_positions,
            -self.config.max_relative_position,
            self.config.max_relative_position
        ) + self.config.max_relative_position
        return relative_positions
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply multi-head attention.
        
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model)
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model)
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Attention mask
            return_attention (bool): Whether to return attention weights
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Output tensor and optionally attention weights
        """
        batch_size, seq_len, d_model = query.size()
        
        # Linear projections and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.config.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.config.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.config.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        attention_output, attention_weights = self._compute_attention(Q, K, V, mask)
        
        # Reshape and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.w_o(attention_output)
        output = self.dropout(output)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def _compute_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention scores and apply to values."""
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Add relative position bias if enabled
        if self.config.use_relative_position:
            seq_len = Q.size(-2)
            relative_positions = self._get_relative_positions(seq_len).to(Q.device)
            
            # Relative position bias for keys
            relative_position_scores_k = torch.einsum(
                'bhld,lrd->bhlr', Q, self.relative_position_k[relative_positions]
            )
            scores = scores + relative_position_scores_k
        
        # Apply temperature scaling
        scores = scores / self.config.temperature
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Add relative position bias for values if enabled
        if self.config.use_relative_position:
            seq_len = Q.size(-2)
            relative_positions = self._get_relative_positions(seq_len).to(Q.device)
            relative_position_values = torch.einsum(
                'bhlr,lrd->bhld', attention_weights, self.relative_position_v[relative_positions]
            )
            context = context + relative_position_values
        
        return context, attention_weights

class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism optimized for time-series data.
    """
    
    def __init__(self, config: AttentionConfig, temporal_decay: bool = True):
        super().__init__()
        self.base_attention = MultiHeadAttention(config)
        self.temporal_decay = temporal_decay
        
        if temporal_decay:
            self.decay_factors = nn.Parameter(torch.ones(config.n_heads))
            self.time_bias = nn.Parameter(torch.zeros(1, config.n_heads, 1, 1))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                time_distances: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply temporal attention with optional time decay.
        
        Args:
            x (torch.Tensor): Input tensor
            mask (torch.Tensor, optional): Attention mask
            time_distances (torch.Tensor, optional): Time distances between positions
            return_attention (bool): Whether to return attention weights
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Output and optionally attention weights
        """
        if self.temporal_decay and time_distances is not None:
            # Apply temporal decay to attention scores
            temporal_mask = self._compute_temporal_decay(time_distances)
            if mask is not None:
                mask = mask * temporal_mask
            else:
                mask = temporal_mask
        
        return self.base_attention(x, x, x, mask, return_attention)
    
    def _compute_temporal_decay(self, time_distances: torch.Tensor) -> torch.Tensor:
        """Compute temporal decay factors."""
        # Apply exponential decay based on time distances
        decay = torch.exp(-self.decay_factors.view(1, -1, 1, 1) * time_distances.unsqueeze(1))
        return decay

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing different sensor modalities.
    """
    
    def __init__(self, config: AttentionConfig, modality_dims: Dict[str, int]):
        super().__init__()
        self.config = config
        self.modality_dims = modality_dims
        
        # Modality-specific projections
        self.modality_projections = nn.ModuleDict({
            modality: nn.Linear(dim, config.d_model)
            for modality, dim in modality_dims.items()
        })
        
        # Cross-attention layers
        self.cross_attentions = nn.ModuleDict({
            f"{mod1}_to_{mod2}": MultiHeadAttention(config)
            for mod1 in modality_dims.keys()
            for mod2 in modality_dims.keys()
            if mod1 != mod2
        })
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.d_model * len(modality_dims), config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model)
        )
    
    def forward(self, modality_inputs: Dict[str, torch.Tensor],
                masks: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Apply cross-modal attention fusion.
        
        Args:
            modality_inputs (Dict[str, torch.Tensor]): Input tensors for each modality
            masks (Dict[str, torch.Tensor], optional): Attention masks for each modality
            
        Returns:
            torch.Tensor: Fused representation
        """
        # Project each modality to common dimension
        projected_modalities = {}
        for modality, input_tensor in modality_inputs.items():
            projected_modalities[modality] = self.modality_projections[modality](input_tensor)
        
        # Apply cross-modal attention
        attended_modalities = {}
        for modality in projected_modalities.keys():
            attended_features = []
            
            for other_modality in projected_modalities.keys():
                if modality != other_modality:
                    attention_key = f"{modality}_to_{other_modality}"
                    if attention_key in self.cross_attentions:
                        mask = masks.get(other_modality) if masks else None
                        attended = self.cross_attentions[attention_key](
                            projected_modalities[modality],
                            projected_modalities[other_modality],
                            projected_modalities[other_modality],
                            mask
                        )
                        attended_features.append(attended)
            
            # Combine attended features for this modality
            if attended_features:
                attended_modalities[modality] = torch.stack(attended_features).mean(dim=0)
            else:
                attended_modalities[modality] = projected_modalities[modality]
        
        # Fuse all modalities
        fused_features = torch.cat(list(attended_modalities.values()), dim=-1)
        output = self.fusion_layer(fused_features)
        
        return output

class SparseAttention(nn.Module):
    """
    Sparse attention mechanism for handling long sequences efficiently.
    """
    
    def __init__(self, config: AttentionConfig, sparsity_pattern: str = "local"):
        super().__init__()
        self.config = config
        self.sparsity_pattern = sparsity_pattern
        self.base_attention = MultiHeadAttention(config)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply sparse attention.
        
        Args:
            x (torch.Tensor): Input tensor
            mask (torch.Tensor, optional): Attention mask
            return_attention (bool): Whether to return attention weights
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Output and optionally attention weights
        """
        seq_len = x.size(1)
        sparse_mask = self._create_sparse_mask(seq_len, x.device)
        
        if mask is not None:
            sparse_mask = sparse_mask * mask
        
        return self.base_attention(x, x, x, sparse_mask, return_attention)
    
    def _create_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sparse attention mask based on pattern."""
        if self.sparsity_pattern == "local":
            # Local attention with fixed window
            window_size = min(64, seq_len // 4)
            mask = torch.zeros(seq_len, seq_len, device=device)
            
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2 + 1)
                mask[i, start:end] = 1
                
        elif self.sparsity_pattern == "strided":
            # Strided attention
            stride = max(1, seq_len // 64)
            mask = torch.zeros(seq_len, seq_len, device=device)
            
            for i in range(seq_len):
                # Attend to positions at regular intervals
                positions = list(range(0, seq_len, stride))
                positions.append(i)  # Always attend to self
                mask[i, positions] = 1
                
        elif self.sparsity_pattern == "random":
            # Random sparse attention
            sparsity_ratio = 0.1
            mask = torch.rand(seq_len, seq_len, device=device) < sparsity_ratio
            # Ensure diagonal is always attended
            mask.fill_diagonal_(True)
            mask = mask.float()
            
        else:
            raise ValueError(f"Unknown sparsity pattern: {self.sparsity_pattern}")
        
        return mask

class BatterySpecificAttention(nn.Module):
    """
    Battery-specific attention mechanism that incorporates domain knowledge.
    """
    
    def __init__(self, config: AttentionConfig, sensor_types: List[str]):
        super().__init__()
        self.config = config
        self.sensor_types = sensor_types
        
        # Base attention
        self.base_attention = MultiHeadAttention(config)
        
        # Sensor-specific attention weights
        self.sensor_importance = nn.Parameter(torch.ones(len(sensor_types)))
        
        # Physics-informed attention bias
        self.physics_bias = nn.Parameter(torch.zeros(config.n_heads, len(sensor_types), len(sensor_types)))
        
        self._init_physics_bias()
    
    def _init_physics_bias(self):
        """Initialize physics-informed bias based on sensor relationships."""
        # Define known sensor correlations
        sensor_correlations = {
            ('voltage', 'current'): 0.8,
            ('voltage', 'temperature'): -0.3,
            ('current', 'temperature'): 0.5,
            ('temperature', 'resistance'): 0.7,
            ('voltage', 'soc'): 0.9,
            ('current', 'power'): 0.95
        }
        
        with torch.no_grad():
            for i, sensor1 in enumerate(self.sensor_types):
                for j, sensor2 in enumerate(self.sensor_types):
                    correlation = sensor_correlations.get((sensor1, sensor2), 0.0)
                    correlation = correlation or sensor_correlations.get((sensor2, sensor1), 0.0)
                    self.physics_bias[:, i, j] = correlation
    
    def forward(self, x: torch.Tensor, sensor_weights: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply battery-specific attention.
        
        Args:
            x (torch.Tensor): Input tensor
            sensor_weights (torch.Tensor, optional): Sensor importance weights
            mask (torch.Tensor, optional): Attention mask
            return_attention (bool): Whether to return attention weights
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Output and optionally attention weights
        """
        # Apply sensor importance weighting
        if sensor_weights is not None:
            importance_weights = self.sensor_importance * sensor_weights
        else:
            importance_weights = self.sensor_importance
        
        # Scale input by sensor importance
        x_weighted = x * importance_weights.view(1, 1, -1)
        
        # Get base attention output
        if return_attention:
            output, attention_weights = self.base_attention(x_weighted, x_weighted, x_weighted, mask, True)
            
            # Apply physics bias to attention weights
            batch_size, n_heads, seq_len, _ = attention_weights.shape
            physics_bias_expanded = self.physics_bias.unsqueeze(0).expand(batch_size, -1, -1, -1)
            
            # Broadcast physics bias to sequence length
            if seq_len > len(self.sensor_types):
                # Repeat pattern for longer sequences
                repeat_factor = seq_len // len(self.sensor_types)
                remainder = seq_len % len(self.sensor_types)
                
                physics_bias_tiled = physics_bias_expanded.repeat(1, 1, repeat_factor, repeat_factor)
                if remainder > 0:
                    partial_bias = physics_bias_expanded[:, :, :remainder, :remainder]
                    physics_bias_tiled = torch.cat([physics_bias_tiled, partial_bias], dim=2)
                    physics_bias_tiled = torch.cat([physics_bias_tiled, partial_bias.transpose(-2, -1)], dim=3)
                
                physics_bias_final = physics_bias_tiled[:, :, :seq_len, :seq_len]
            else:
                physics_bias_final = physics_bias_expanded[:, :, :seq_len, :seq_len]
            
            # Apply physics bias
            attention_weights = attention_weights + 0.1 * physics_bias_final
            attention_weights = F.softmax(attention_weights, dim=-1)
            
            return output, attention_weights
        else:
            return self.base_attention(x_weighted, x_weighted, x_weighted, mask, False)

class AttentionVisualizer:
    """
    Utility class for visualizing attention patterns.
    """
    
    @staticmethod
    def extract_attention_patterns(model: nn.Module, input_tensor: torch.Tensor,
                                 layer_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Extract attention patterns from a model.
        
        Args:
            model (nn.Module): Model containing attention layers
            input_tensor (torch.Tensor): Input tensor
            layer_names (List[str], optional): Specific layer names to extract from
            
        Returns:
            Dict[str, torch.Tensor]: Attention patterns by layer
        """
        attention_patterns = {}
        
        def attention_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    attention_patterns[name] = output[1].detach()
                elif hasattr(output, 'attention_weights'):
                    attention_patterns[name] = output.attention_weights.detach()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if any(attention_type in name.lower() for attention_type in ['attention', 'attn']):
                if layer_names is None or name in layer_names:
                    hook = module.register_forward_hook(attention_hook(name))
                    hooks.append(hook)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_patterns
    
    @staticmethod
    def compute_attention_statistics(attention_weights: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics for attention weights.
        
        Args:
            attention_weights (torch.Tensor): Attention weights tensor
            
        Returns:
            Dict[str, float]: Attention statistics
        """
        # Flatten attention weights
        flat_weights = attention_weights.view(-1)
        
        return {
            'mean': float(flat_weights.mean()),
            'std': float(flat_weights.std()),
            'min': float(flat_weights.min()),
            'max': float(flat_weights.max()),
            'entropy': float(-torch.sum(flat_weights * torch.log(flat_weights + 1e-8))),
            'sparsity': float((flat_weights < 0.01).float().mean()),
            'concentration': float((flat_weights > 0.1).float().mean())
        }

# Factory functions
def create_attention_layer(attention_type: str, config: AttentionConfig, **kwargs) -> nn.Module:
    """
    Factory function to create attention layers.
    
    Args:
        attention_type (str): Type of attention layer
        config (AttentionConfig): Attention configuration
        **kwargs: Additional arguments
        
    Returns:
        nn.Module: Created attention layer
    """
    if attention_type == "multi_head":
        return MultiHeadAttention(config)
    elif attention_type == "temporal":
        return TemporalAttention(config, **kwargs)
    elif attention_type == "cross_modal":
        return CrossModalAttention(config, **kwargs)
    elif attention_type == "sparse":
        return SparseAttention(config, **kwargs)
    elif attention_type == "battery_specific":
        return BatterySpecificAttention(config, **kwargs)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")

def benchmark_attention_layers(attention_layers: List[nn.Module], 
                             input_tensor: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """
    Benchmark multiple attention layers.
    
    Args:
        attention_layers (List[nn.Module]): List of attention layers
        input_tensor (torch.Tensor): Input tensor for benchmarking
        
    Returns:
        Dict[str, Dict[str, float]]: Benchmark results
    """
    results = {}
    
    for i, layer in enumerate(attention_layers):
        layer_name = f"layer_{i}_{layer.__class__.__name__}"
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = layer(input_tensor, input_tensor, input_tensor)
        
        # Benchmark
        times = []
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                _ = layer(input_tensor, input_tensor, input_tensor)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        results[layer_name] = {
            'avg_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'parameters': sum(p.numel() for p in layer.parameters())
        }
    
    return results
