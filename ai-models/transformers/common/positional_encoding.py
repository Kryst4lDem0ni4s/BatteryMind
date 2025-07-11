"""
BatteryMind - Common Positional Encoding Utilities

Advanced positional encoding implementations for transformer models with
specialized support for time-series data, battery-specific features, and
multi-modal sensor fusion.

Features:
- Standard sinusoidal positional encoding
- Learnable positional embeddings
- Relative positional encoding
- Battery-specific temporal encoding
- Seasonal and cyclic pattern encoding
- Multi-modal sensor position encoding
- Adaptive positional encoding for variable sequences

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasePositionalEncoding(nn.Module, ABC):
    """
    Abstract base class for all positional encoding implementations.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply positional encoding to input tensor."""
        pass

class SinusoidalPositionalEncoding(BasePositionalEncoding):
    """
    Standard sinusoidal positional encoding as described in 'Attention Is All You Need'.
    
    This implementation provides the classic transformer positional encoding
    with optional temperature scaling and frequency modulation.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1,
                 temperature: float = 10000.0, learnable_scale: bool = False):
        super().__init__(d_model, max_len, dropout)
        self.temperature = temperature
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate division term for frequency scaling
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(temperature) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        # Register as buffer (not a parameter, but part of state)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
        # Optional learnable scaling factor
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('scale', torch.ones(1))
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply sinusoidal positional encoding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            positions (torch.Tensor, optional): Custom position indices
            
        Returns:
            torch.Tensor: Input with positional encoding added
        """
        if positions is not None:
            # Use custom positions
            pe = self.pe[positions.long()]
            if pe.dim() == 3:  # (seq_len, batch_size, d_model)
                pe = pe.transpose(0, 1)  # (batch_size, seq_len, d_model)
        else:
            # Use sequential positions
            seq_len = x.size(1)
            pe = self.pe[:seq_len, :].transpose(0, 1)  # (batch_size, seq_len, d_model)
            pe = pe.expand(x.size(0), -1, -1)
        
        x = x + self.scale * pe
        return self.dropout(x)

class LearnablePositionalEncoding(BasePositionalEncoding):
    """
    Learnable positional embeddings that are trained along with the model.
    
    This approach allows the model to learn optimal positional representations
    specific to the task and data distribution.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1,
                 init_std: float = 0.02):
        super().__init__(d_model, max_len, dropout)
        
        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
        # Initialize with small random values
        nn.init.normal_(self.position_embeddings.weight, std=init_std)
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply learnable positional encoding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            positions (torch.Tensor, optional): Custom position indices
            
        Returns:
            torch.Tensor: Input with positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        if positions is not None:
            # Use custom positions
            pos_embeddings = self.position_embeddings(positions.long())
        else:
            # Use sequential positions
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            pos_embeddings = self.position_embeddings(positions)
        
        x = x + pos_embeddings
        return self.dropout(x)

class RelativePositionalEncoding(BasePositionalEncoding):
    """
    Relative positional encoding that focuses on relative distances between positions
    rather than absolute positions. Particularly useful for time-series data.
    """
    
    def __init__(self, d_model: int, max_relative_position: int = 128, 
                 dropout: float = 0.1, bidirectional: bool = True):
        super().__init__(d_model, max_relative_position * 2 + 1, dropout)
        self.max_relative_position = max_relative_position
        self.bidirectional = bidirectional
        
        # Relative position embeddings
        vocab_size = max_relative_position * 2 + 1 if bidirectional else max_relative_position + 1
        self.relative_position_embeddings = nn.Embedding(vocab_size, d_model)
        
        # Initialize embeddings
        nn.init.normal_(self.relative_position_embeddings.weight, std=0.02)
    
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """Generate relative position matrix."""
        positions = torch.arange(seq_len, dtype=torch.long)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        if self.bidirectional:
            # Clamp to valid range and shift to positive indices
            relative_positions = torch.clamp(
                relative_positions, 
                -self.max_relative_position, 
                self.max_relative_position
            ) + self.max_relative_position
        else:
            # Only forward relative positions
            relative_positions = torch.clamp(relative_positions, 0, self.max_relative_position)
        
        return relative_positions
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply relative positional encoding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Input with relative positional encoding
        """
        batch_size, seq_len, d_model = x.shape
        
        # Get relative positions
        relative_positions = self._get_relative_positions(seq_len).to(x.device)
        
        # Get relative position embeddings
        relative_embeddings = self.relative_position_embeddings(relative_positions)
        
        # Average relative embeddings for each position
        position_embeddings = relative_embeddings.mean(dim=1)  # (seq_len, d_model)
        position_embeddings = position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        x = x + position_embeddings
        return self.dropout(x)

class BatteryTemporalEncoding(BasePositionalEncoding):
    """
    Battery-specific temporal encoding that incorporates battery lifecycle information,
    charging cycles, and temporal patterns specific to battery behavior.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1,
                 max_cycles: int = 10000, max_age_days: int = 3650):
        super().__init__(d_model, max_len, dropout)
        self.max_cycles = max_cycles
        self.max_age_days = max_age_days
        
        # Different encoding dimensions
        pos_dim = d_model // 4
        cycle_dim = d_model // 4
        age_dim = d_model // 4
        temp_dim = d_model - pos_dim - cycle_dim - age_dim
        
        # Standard positional encoding
        self.position_encoding = SinusoidalPositionalEncoding(
            pos_dim, max_len, dropout=0.0
        )
        
        # Cycle count encoding
        self.cycle_encoding = nn.Sequential(
            nn.Linear(1, cycle_dim),
            nn.ReLU(),
            nn.Linear(cycle_dim, cycle_dim)
        )
        
        # Battery age encoding
        self.age_encoding = nn.Sequential(
            nn.Linear(1, age_dim),
            nn.ReLU(),
            nn.Linear(age_dim, age_dim)
        )
        
        # Temperature-based encoding
        self.temperature_encoding = nn.Sequential(
            nn.Linear(1, temp_dim),
            nn.ReLU(),
            nn.Linear(temp_dim, temp_dim)
        )
    
    def forward(self, x: torch.Tensor, battery_metadata: Optional[Dict] = None) -> torch.Tensor:
        """
        Apply battery-specific temporal encoding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            battery_metadata (Dict, optional): Battery metadata including cycles, age, temperature
            
        Returns:
            torch.Tensor: Input with battery temporal encoding
        """
        batch_size, seq_len, d_model = x.shape
        
        # Extract position encoding
        pos_x = x[:, :, :self.d_model//4]
        pos_encoding = self.position_encoding(pos_x)
        
        # Default metadata if not provided
        if battery_metadata is None:
            battery_metadata = {
                'charge_cycles': torch.zeros(batch_size, 1, device=x.device),
                'battery_age_days': torch.zeros(batch_size, 1, device=x.device),
                'avg_temperature': torch.full((batch_size, 1), 25.0, device=x.device)
            }
        
        # Encode battery characteristics
        cycles = battery_metadata.get('charge_cycles', torch.zeros(batch_size, 1, device=x.device))
        age = battery_metadata.get('battery_age_days', torch.zeros(batch_size, 1, device=x.device))
        temp = battery_metadata.get('avg_temperature', torch.full((batch_size, 1), 25.0, device=x.device))
        
        # Normalize inputs
        cycles_norm = cycles.float() / self.max_cycles
        age_norm = age.float() / self.max_age_days
        temp_norm = (temp.float() - 25.0) / 50.0  # Normalize around 25°C
        
        # Generate encodings
        cycle_enc = self.cycle_encoding(cycles_norm)
        age_enc = self.age_encoding(age_norm)
        temp_enc = self.temperature_encoding(temp_norm)
        
        # Expand to sequence length
        cycle_enc = cycle_enc.unsqueeze(1).expand(-1, seq_len, -1)
        age_enc = age_enc.unsqueeze(1).expand(-1, seq_len, -1)
        temp_enc = temp_enc.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine all encodings
        combined_encoding = torch.cat([pos_encoding, cycle_enc, age_enc, temp_enc], dim=-1)
        
        x = x + combined_encoding
        return self.dropout(x)

class SeasonalPositionalEncoding(BasePositionalEncoding):
    """
    Seasonal positional encoding for capturing periodic patterns in time-series data.
    Useful for battery data that may have daily, weekly, or monthly patterns.
    """
    
    def __init__(self, d_model: int, seasonal_periods: List[int] = None,
                 dropout: float = 0.1, learnable: bool = True):
        super().__init__(d_model, max(seasonal_periods) if seasonal_periods else 8760, dropout)
        
        self.seasonal_periods = seasonal_periods or [24, 168, 720, 8760]  # hour, week, month, year
        self.learnable = learnable
        
        if learnable:
            # Learnable seasonal embeddings
            self.seasonal_embeddings = nn.ModuleList([
                nn.Embedding(period, d_model // len(self.seasonal_periods))
                for period in self.seasonal_periods
            ])
        else:
            # Fixed sinusoidal seasonal encodings
            self.register_buffer('seasonal_encodings', self._create_seasonal_encodings())
    
    def _create_seasonal_encodings(self) -> torch.Tensor:
        """Create fixed sinusoidal seasonal encodings."""
        max_period = max(self.seasonal_periods)
        encoding_dim = self.d_model // len(self.seasonal_periods)
        
        encodings = []
        for period in self.seasonal_periods:
            pe = torch.zeros(max_period, encoding_dim)
            position = torch.arange(0, max_period, dtype=torch.float).unsqueeze(1)
            
            # Create seasonal frequencies
            div_term = torch.exp(torch.arange(0, encoding_dim, 2).float() * 
                               (-math.log(period) / encoding_dim))
            
            pe[:, 0::2] = torch.sin(2 * math.pi * position / period * div_term)
            if encoding_dim % 2 == 0:
                pe[:, 1::2] = torch.cos(2 * math.pi * position / period * div_term)
            else:
                pe[:, 1::2] = torch.cos(2 * math.pi * position / period * div_term[:-1])
            
            encodings.append(pe)
        
        return torch.cat(encodings, dim=-1)
    
    def forward(self, x: torch.Tensor, time_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply seasonal positional encoding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            time_indices (torch.Tensor, optional): Time indices for seasonal patterns
            
        Returns:
            torch.Tensor: Input with seasonal positional encoding
        """
        batch_size, seq_len, d_model = x.shape
        
        if time_indices is None:
            # Use sequential time indices
            time_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        if self.learnable:
            # Use learnable embeddings
            seasonal_encodings = []
            for i, (period, embedding) in enumerate(zip(self.seasonal_periods, self.seasonal_embeddings)):
                seasonal_idx = time_indices % period
                seasonal_enc = embedding(seasonal_idx.long())
                seasonal_encodings.append(seasonal_enc)
            
            seasonal_encoding = torch.cat(seasonal_encodings, dim=-1)
        else:
            # Use fixed encodings
            seasonal_encoding = self.seasonal_encodings[time_indices.long()]
        
        # Pad or truncate to match d_model
        if seasonal_encoding.size(-1) < d_model:
            padding = torch.zeros(batch_size, seq_len, d_model - seasonal_encoding.size(-1), 
                                device=x.device)
            seasonal_encoding = torch.cat([seasonal_encoding, padding], dim=-1)
        elif seasonal_encoding.size(-1) > d_model:
            seasonal_encoding = seasonal_encoding[:, :, :d_model]
        
        x = x + seasonal_encoding
        return self.dropout(x)

class AdaptivePositionalEncoding(BasePositionalEncoding):
    """
    Adaptive positional encoding that adjusts based on sequence characteristics.
    Uses attention mechanisms to weight different positional encoding strategies.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1,
                 num_strategies: int = 3):
        super().__init__(d_model, max_len, dropout)
        self.num_strategies = num_strategies
        
        # Multiple encoding strategies
        self.sinusoidal = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
        self.learnable = LearnablePositionalEncoding(d_model, max_len, dropout=0.0)
        self.relative = RelativePositionalEncoding(d_model, max_len//10, dropout=0.0)
        
        # Strategy selection network
        self.strategy_selector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_strategies),
            nn.Softmax(dim=-1)
        )
        
        # Strategy combination weights
        self.combination_weights = nn.Parameter(torch.ones(num_strategies) / num_strategies)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply adaptive positional encoding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Input with adaptive positional encoding
        """
        # Apply different encoding strategies
        sin_encoded = self.sinusoidal(x.clone(), **kwargs)
        learnable_encoded = self.learnable(x.clone(), **kwargs)
        relative_encoded = self.relative(x.clone(), **kwargs)
        
        # Stack encoded versions
        encoded_stack = torch.stack([sin_encoded, learnable_encoded, relative_encoded], dim=-1)
        
        # Calculate adaptive weights based on input characteristics
        input_stats = torch.mean(x, dim=1)  # (batch_size, d_model)
        strategy_weights = self.strategy_selector(input_stats)  # (batch_size, num_strategies)
        
        # Expand weights for broadcasting
        strategy_weights = strategy_weights.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, num_strategies)
        
        # Combine strategies using learned weights
        combined_weights = strategy_weights * self.combination_weights.view(1, 1, 1, -1)
        adaptive_encoded = torch.sum(encoded_stack * combined_weights, dim=-1)
        
        return self.dropout(adaptive_encoded)

class MultiModalPositionalEncoding(BasePositionalEncoding):
    """
    Multi-modal positional encoding for handling different types of sensor data
    with distinct positional characteristics.
    """
    
    def __init__(self, d_model: int, modality_dims: Dict[str, int], 
                 max_len: int = 5000, dropout: float = 0.1):
        super().__init__(d_model, max_len, dropout)
        self.modality_dims = modality_dims
        self.total_modalities = len(modality_dims)
        
        # Separate encodings for each modality
        self.modality_encodings = nn.ModuleDict({
            modality: SinusoidalPositionalEncoding(dim, max_len, dropout=0.0)
            for modality, dim in modality_dims.items()
        })
        
        # Modality fusion layer
        total_dim = sum(modality_dims.values())
        if total_dim != d_model:
            self.fusion_layer = nn.Linear(total_dim, d_model)
        else:
            self.fusion_layer = nn.Identity()
        
        # Modality attention for dynamic weighting
        self.modality_attention = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
    
    def forward(self, x: torch.Tensor, modality_data: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Apply multi-modal positional encoding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            modality_data (Dict[str, torch.Tensor], optional): Data for each modality
            
        Returns:
            torch.Tensor: Input with multi-modal positional encoding
        """
        if modality_data is None:
            # Split input tensor across modalities
            modality_data = {}
            start_idx = 0
            for modality, dim in self.modality_dims.items():
                end_idx = start_idx + dim
                modality_data[modality] = x[:, :, start_idx:end_idx]
                start_idx = end_idx
        
        # Apply positional encoding to each modality
        encoded_modalities = []
        for modality, data in modality_data.items():
            if modality in self.modality_encodings:
                encoded = self.modality_encodings[modality](data)
                encoded_modalities.append(encoded)
        
        # Concatenate encoded modalities
        if encoded_modalities:
            combined_encoding = torch.cat(encoded_modalities, dim=-1)
            combined_encoding = self.fusion_layer(combined_encoding)
        else:
            combined_encoding = torch.zeros_like(x)
        
        # Apply cross-modal attention
        attended_encoding, _ = self.modality_attention(
            combined_encoding, combined_encoding, combined_encoding
        )
        
        x = x + attended_encoding
        return self.dropout(x)

# Factory functions for easy instantiation
def create_positional_encoding(encoding_type: str, d_model: int, **kwargs) -> BasePositionalEncoding:
    """
    Factory function to create positional encoding based on type.
    
    Args:
        encoding_type (str): Type of positional encoding
        d_model (int): Model dimension
        **kwargs: Additional arguments for specific encoding types
        
    Returns:
        BasePositionalEncoding: Configured positional encoding instance
    """
    encoding_map = {
        'sinusoidal': SinusoidalPositionalEncoding,
        'learnable': LearnablePositionalEncoding,
        'relative': RelativePositionalEncoding,
        'battery_temporal': BatteryTemporalEncoding,
        'seasonal': SeasonalPositionalEncoding,
        'adaptive': AdaptivePositionalEncoding,
        'multimodal': MultiModalPositionalEncoding
    }
    
    if encoding_type not in encoding_map:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    return encoding_map[encoding_type](d_model, **kwargs)

def get_encoding_recommendations(data_type: str, sequence_length: int) -> Dict[str, str]:
    """
    Get recommended positional encoding based on data characteristics.
    
    Args:
        data_type (str): Type of data ('battery', 'timeseries', 'multimodal')
        sequence_length (int): Length of sequences
        
    Returns:
        Dict[str, str]: Recommendations for positional encoding
    """
    recommendations = {}
    
    if data_type == 'battery':
        if sequence_length > 1000:
            recommendations['primary'] = 'battery_temporal'
            recommendations['secondary'] = 'seasonal'
        else:
            recommendations['primary'] = 'sinusoidal'
            recommendations['secondary'] = 'battery_temporal'
    
    elif data_type == 'timeseries':
        if sequence_length > 500:
            recommendations['primary'] = 'seasonal'
            recommendations['secondary'] = 'relative'
        else:
            recommendations['primary'] = 'sinusoidal'
            recommendations['secondary'] = 'learnable'
    
    elif data_type == 'multimodal':
        recommendations['primary'] = 'multimodal'
        recommendations['secondary'] = 'adaptive'
    
    else:
        recommendations['primary'] = 'sinusoidal'
        recommendations['secondary'] = 'learnable'
    
    return recommendations
