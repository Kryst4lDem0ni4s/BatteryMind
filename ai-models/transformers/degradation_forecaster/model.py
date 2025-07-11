"""
BatteryMind - Degradation Forecaster Model

Advanced transformer architecture specifically designed for long-term battery
degradation forecasting with temporal attention mechanisms and uncertainty
quantification capabilities.

Features:
- Multi-horizon forecasting with configurable time scales
- Temporal attention mechanisms for long-term dependencies
- Seasonal decomposition and trend analysis
- Uncertainty quantification with prediction intervals
- Physics-informed constraints for realistic predictions
- Integration with battery health prediction models

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
from abc import ABC, abstractmethod

# Time series processing imports
from scipy import signal
from scipy.stats import norm
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DegradationConfig:
    """
    Configuration class for Degradation Forecaster model.
    
    Attributes:
        # Model architecture
        d_model (int): Model dimension (embedding size)
        n_heads (int): Number of attention heads
        n_layers (int): Number of transformer layers
        d_ff (int): Feed-forward network dimension
        dropout (float): Dropout probability
        max_sequence_length (int): Maximum input sequence length
        
        # Forecasting specific
        forecast_horizon (int): Number of future time steps to predict
        uncertainty_quantiles (List[float]): Quantiles for uncertainty estimation
        enable_seasonal_decomposition (bool): Enable seasonal pattern modeling
        seasonal_periods (List[int]): Known seasonal periods (daily, weekly, monthly)
        
        # Input/Output dimensions
        feature_dim (int): Input feature dimension
        target_dim (int): Output target dimension
        
        # Advanced features
        use_temporal_attention (bool): Use specialized temporal attention
        use_trend_analysis (bool): Enable trend decomposition
        use_change_point_detection (bool): Enable change point modeling
        
        # Physics constraints
        use_physics_constraints (bool): Enable physics-informed constraints
        degradation_bounds (Tuple[float, float]): Valid degradation rate bounds
        temperature_sensitivity (float): Temperature impact factor
        
        # Uncertainty modeling
        enable_uncertainty (bool): Enable uncertainty quantification
        uncertainty_method (str): Method for uncertainty estimation
        monte_carlo_samples (int): Number of MC samples for uncertainty
    """
    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8  # Deeper for long-term dependencies
    d_ff: int = 2048
    dropout: float = 0.1
    max_sequence_length: int = 2048  # Longer sequences for forecasting
    
    # Forecasting specific
    forecast_horizon: int = 168  # 1 week in hours
    uncertainty_quantiles: List[float] = None
    enable_seasonal_decomposition: bool = True
    seasonal_periods: List[int] = None
    
    # Input/Output dimensions
    feature_dim: int = 20
    target_dim: int = 6  # Multiple degradation metrics
    
    # Advanced features
    use_temporal_attention: bool = True
    use_trend_analysis: bool = True
    use_change_point_detection: bool = True
    
    # Physics constraints
    use_physics_constraints: bool = True
    degradation_bounds: Tuple[float, float] = (0.0, 0.01)  # 0-1% per time step
    temperature_sensitivity: float = 2.0
    
    # Uncertainty modeling
    enable_uncertainty: bool = True
    uncertainty_method: str = "monte_carlo"
    monte_carlo_samples: int = 100
    
    def __post_init__(self):
        if self.uncertainty_quantiles is None:
            self.uncertainty_quantiles = [0.1, 0.25, 0.75, 0.9]
        if self.seasonal_periods is None:
            self.seasonal_periods = [24, 168, 720]  # Daily, weekly, monthly

class TemporalPositionalEncoding(nn.Module):
    """
    Advanced positional encoding for time-series forecasting with seasonal patterns.
    """
    
    def __init__(self, d_model: int, max_len: int = 10000, 
                 seasonal_periods: List[int] = None):
        super().__init__()
        self.d_model = d_model
        self.seasonal_periods = seasonal_periods or [24, 168, 720]
        
        # Standard sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Seasonal encodings
        self.seasonal_encodings = nn.ModuleList([
            nn.Linear(2, d_model // len(self.seasonal_periods))
            for _ in self.seasonal_periods
        ])
        
        # Trend encoding
        self.trend_encoding = nn.Linear(1, d_model // 4)
        
    def forward(self, x: torch.Tensor, time_features: Optional[Dict] = None) -> torch.Tensor:
        """
        Apply temporal positional encoding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            time_features (Dict, optional): Time-based features
            
        Returns:
            torch.Tensor: Positionally encoded tensor
        """
        batch_size, seq_len, _ = x.shape
        
        # Standard positional encoding
        pos_encoding = self.pe[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Seasonal encodings if time features provided
        if time_features is not None:
            seasonal_encodings = []
            
            for i, period in enumerate(self.seasonal_periods):
                if f'seasonal_{period}' in time_features:
                    seasonal_data = time_features[f'seasonal_{period}']
                    # Create sin/cos encoding for seasonality
                    seasonal_sin = torch.sin(2 * math.pi * seasonal_data / period)
                    seasonal_cos = torch.cos(2 * math.pi * seasonal_data / period)
                    seasonal_input = torch.stack([seasonal_sin, seasonal_cos], dim=-1)
                    seasonal_enc = self.seasonal_encodings[i](seasonal_input)
                    seasonal_encodings.append(seasonal_enc)
            
            if seasonal_encodings:
                seasonal_encoding = torch.cat(seasonal_encodings, dim=-1)
                # Pad to match d_model if necessary
                if seasonal_encoding.size(-1) < self.d_model:
                    padding_size = self.d_model - seasonal_encoding.size(-1)
                    padding = torch.zeros(batch_size, seq_len, padding_size, 
                                        device=seasonal_encoding.device)
                    seasonal_encoding = torch.cat([seasonal_encoding, padding], dim=-1)
                
                pos_encoding = pos_encoding + seasonal_encoding
        
        return x + pos_encoding

class TemporalAttention(nn.Module):
    """
    Specialized attention mechanism for time-series forecasting with temporal bias.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 max_relative_position: int = 128):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_relative_position = max_relative_position
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Relative position embeddings
        self.relative_position_k = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, self.d_k)
        )
        self.relative_position_v = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, self.d_k)
        )
        
        # Temporal decay factor
        self.temporal_decay = nn.Parameter(torch.ones(n_heads))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """Generate relative position matrix."""
        positions = torch.arange(seq_len, dtype=torch.long)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_relative_position, 
            self.max_relative_position
        ) + self.max_relative_position
        return relative_positions
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention mechanism.
        
        Args:
            query (torch.Tensor): Query tensor
            key (torch.Tensor): Key tensor
            value (torch.Tensor): Value tensor
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights
        """
        batch_size, seq_len, d_model = query.size()
        
        # Linear projections and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Add relative position bias
        relative_positions = self._get_relative_positions(seq_len).to(query.device)
        relative_position_scores = torch.einsum(
            'bhld,lrd->bhlr', Q, self.relative_position_k[relative_positions]
        )
        scores = scores + relative_position_scores
        
        # Apply temporal decay
        time_distances = torch.abs(torch.arange(seq_len).unsqueeze(0) - 
                                 torch.arange(seq_len).unsqueeze(1)).float().to(query.device)
        temporal_bias = -self.temporal_decay.view(1, -1, 1, 1) * time_distances.unsqueeze(0).unsqueeze(0)
        scores = scores + temporal_bias
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values with relative position bias
        context = torch.matmul(attention_weights, V)
        relative_position_values = torch.einsum(
            'bhlr,lrd->bhld', attention_weights, self.relative_position_v[relative_positions]
        )
        context = context + relative_position_values
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        output = self.w_o(context)
        
        return output, attention_weights

class SeasonalDecomposition(nn.Module):
    """
    Neural seasonal decomposition for time-series forecasting.
    """
    
    def __init__(self, d_model: int, seasonal_periods: List[int]):
        super().__init__()
        self.seasonal_periods = seasonal_periods
        
        # Seasonal extractors
        self.seasonal_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=period, 
                         padding=period//2, groups=d_model),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, kernel_size=1)
            ) for period in seasonal_periods
        ])
        
        # Trend extractor
        self.trend_extractor = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=25, padding=12),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1)
        )
        
        # Residual processor
        self.residual_processor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Dict[str, torch.Tensor]: Decomposed components
        """
        batch_size, seq_len, d_model = x.shape
        
        # Transpose for conv1d (batch_size, d_model, seq_len)
        x_conv = x.transpose(1, 2)
        
        # Extract trend
        trend = self.trend_extractor(x_conv).transpose(1, 2)
        
        # Extract seasonal components
        seasonal_components = []
        for extractor in self.seasonal_extractors:
            seasonal = extractor(x_conv).transpose(1, 2)
            seasonal_components.append(seasonal)
        
        # Combine seasonal components
        seasonal = torch.stack(seasonal_components, dim=0).sum(dim=0)
        
        # Calculate residual
        residual = x - trend - seasonal
        residual = self.residual_processor(residual)
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'reconstructed': trend + seasonal + residual
        }

class UncertaintyQuantification(nn.Module):
    """
    Uncertainty quantification for forecasting predictions.
    """
    
    def __init__(self, d_model: int, target_dim: int, quantiles: List[float],
                 method: str = "monte_carlo"):
        super().__init__()
        self.quantiles = quantiles
        self.method = method
        self.target_dim = target_dim
        
        if method == "quantile_regression":
            # Separate heads for each quantile
            self.quantile_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model // 2, target_dim)
                ) for _ in quantiles
            ])
        elif method == "monte_carlo":
            # Single head with dropout for MC sampling
            self.prediction_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, target_dim)
            )
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
    
    def forward(self, x: torch.Tensor, training: bool = True, 
                n_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        Generate predictions with uncertainty estimates.
        
        Args:
            x (torch.Tensor): Input features
            training (bool): Whether in training mode
            n_samples (int): Number of MC samples
            
        Returns:
            Dict[str, torch.Tensor]: Predictions with uncertainty estimates
        """
        if self.method == "quantile_regression":
            # Generate predictions for each quantile
            quantile_predictions = []
            for head in self.quantile_heads:
                pred = head(x)
                quantile_predictions.append(pred)
            
            predictions = torch.stack(quantile_predictions, dim=-1)
            
            # Calculate mean and std from quantiles
            mean_pred = predictions.mean(dim=-1)
            std_pred = predictions.std(dim=-1)
            
            return {
                'mean': mean_pred,
                'std': std_pred,
                'quantiles': predictions,
                'quantile_values': self.quantiles
            }
        
        elif self.method == "monte_carlo":
            if training:
                # Single forward pass during training
                prediction = self.prediction_head(x)
                return {
                    'mean': prediction,
                    'std': torch.zeros_like(prediction),
                    'samples': prediction.unsqueeze(-1)
                }
            else:
                # Multiple forward passes for uncertainty estimation
                self.train()  # Enable dropout
                samples = []
                for _ in range(n_samples):
                    sample = self.prediction_head(x)
                    samples.append(sample)
                
                samples = torch.stack(samples, dim=-1)
                mean_pred = samples.mean(dim=-1)
                std_pred = samples.std(dim=-1)
                
                return {
                    'mean': mean_pred,
                    'std': std_pred,
                    'samples': samples
                }

class DegradationTransformerBlock(nn.Module):
    """
    Transformer block optimized for degradation forecasting.
    """
    
    def __init__(self, config: DegradationConfig):
        super().__init__()
        self.config = config
        
        # Temporal attention
        if config.use_temporal_attention:
            self.attention = TemporalAttention(
                config.d_model, config.n_heads, config.dropout
            )
        else:
            self.attention = nn.MultiheadAttention(
                config.d_model, config.n_heads, config.dropout, batch_first=True
            )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer block."""
        # Self-attention with residual connection
        if isinstance(self.attention, TemporalAttention):
            attn_output, _ = self.attention(x, x, x, mask)
        else:
            attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DegradationForecaster(nn.Module):
    """
    Main transformer model for battery degradation forecasting.
    
    This model specializes in long-term degradation pattern prediction with
    uncertainty quantification and seasonal decomposition capabilities.
    """
    
    def __init__(self, config: DegradationConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.feature_dim, config.d_model)
        
        # Positional encoding
        self.positional_encoding = TemporalPositionalEncoding(
            config.d_model, config.max_sequence_length, config.seasonal_periods
        )
        
        # Seasonal decomposition
        if config.enable_seasonal_decomposition:
            self.seasonal_decomposition = SeasonalDecomposition(
                config.d_model, config.seasonal_periods
            )
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            DegradationTransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Forecasting head
        self.forecasting_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.target_dim * config.forecast_horizon)
        )
        
        # Uncertainty quantification
        if config.enable_uncertainty:
            self.uncertainty_quantification = UncertaintyQuantification(
                config.d_model, config.target_dim * config.forecast_horizon,
                config.uncertainty_quantiles, config.uncertainty_method
            )
        
        # Physics constraints layer
        if config.use_physics_constraints:
            self.physics_constraints = DegradationPhysicsConstraints(config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
    
    def create_forecasting_mask(self, seq_len: int, forecast_len: int) -> torch.Tensor:
        """Create causal mask for forecasting."""
        total_len = seq_len + forecast_len
        mask = torch.triu(torch.ones(total_len, total_len), diagonal=1)
        return mask == 0
    
    def forward(self, x: torch.Tensor, time_features: Optional[Dict] = None,
                return_components: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the degradation forecaster.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim)
            time_features (Dict, optional): Time-based features for seasonal modeling
            return_components (bool): Whether to return decomposed components
            
        Returns:
            Dict[str, torch.Tensor]: Forecasting results with uncertainty estimates
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # Input validation
        if feature_dim != self.config.feature_dim:
            raise ValueError(f"Expected feature_dim {self.config.feature_dim}, got {feature_dim}")
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x, time_features)
        
        # Seasonal decomposition if enabled
        decomposition_results = None
        if self.config.enable_seasonal_decomposition:
            decomposition_results = self.seasonal_decomposition(x)
            x = decomposition_results['reconstructed']
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Use last token for forecasting
        forecast_input = x[:, -1, :]  # (batch_size, d_model)
        
        # Generate forecasts
        if self.config.enable_uncertainty:
            uncertainty_results = self.uncertainty_quantification(
                forecast_input, self.training
            )
            forecasts = uncertainty_results['mean']
        else:
            forecasts = self.forecasting_head(forecast_input)
        
        # Reshape forecasts
        forecasts = forecasts.view(batch_size, self.config.forecast_horizon, self.config.target_dim)
        
        # Apply physics constraints if enabled
        if self.config.use_physics_constraints:
            forecasts = self.physics_constraints(forecasts, x, time_features)
        
        # Prepare output
        output = {
            'forecasts': forecasts,
            'hidden_states': x
        }
        
        # Add uncertainty estimates if available
        if self.config.enable_uncertainty:
            uncertainty_results['mean'] = uncertainty_results['mean'].view(
                batch_size, self.config.forecast_horizon, self.config.target_dim
            )
            uncertainty_results['std'] = uncertainty_results['std'].view(
                batch_size, self.config.forecast_horizon, self.config.target_dim
            )
            output.update(uncertainty_results)
        
        # Add decomposition results if requested
        if return_components and decomposition_results:
            output['decomposition'] = decomposition_results
        
        return output
    
    def predict_degradation(self, x: torch.Tensor, 
                          time_features: Optional[Dict] = None,
                          confidence_level: float = 0.95) -> Dict[str, Union[torch.Tensor, float]]:
        """
        High-level interface for degradation forecasting.
        
        Args:
            x (torch.Tensor): Input sensor data
            time_features (Dict, optional): Time features
            confidence_level (float): Confidence level for prediction intervals
            
        Returns:
            Dict[str, Union[torch.Tensor, float]]: Forecasting results
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x, time_features, return_components=True)
            
            forecasts = output['forecasts']
            
            # Calculate prediction intervals if uncertainty is available
            prediction_intervals = None
            if 'std' in output:
                std = output['std']
                z_score = norm.ppf((1 + confidence_level) / 2)
                lower_bound = forecasts - z_score * std
                upper_bound = forecasts + z_score * std
                prediction_intervals = {
                    'lower': lower_bound,
                    'upper': upper_bound,
                    'confidence_level': confidence_level
                }
            
            # Calculate degradation metrics
            degradation_metrics = self._calculate_degradation_metrics(forecasts)
            
            return {
                'forecasts': forecasts,
                'prediction_intervals': prediction_intervals,
                'degradation_metrics': degradation_metrics,
                'forecast_horizon_hours': self.config.forecast_horizon,
                'decomposition': output.get('decomposition')
            }
    
    def _calculate_degradation_metrics(self, forecasts: torch.Tensor) -> Dict[str, float]:
        """Calculate degradation-specific metrics from forecasts."""
        # Assuming target dimensions are:
        # 0: capacity_fade_rate, 1: resistance_increase_rate, 2: thermal_degradation
        # 3: cycle_efficiency_decline, 4: calendar_aging_rate, 5: overall_health_decline
        
        metrics = {}
        
        if forecasts.size(-1) >= 6:
            # Calculate average degradation rates
            metrics['avg_capacity_fade_rate'] = forecasts[:, :, 0].mean().item()
            metrics['avg_resistance_increase_rate'] = forecasts[:, :, 1].mean().item()
            metrics['avg_thermal_degradation'] = forecasts[:, :, 2].mean().item()
            metrics['avg_cycle_efficiency_decline'] = forecasts[:, :, 3].mean().item()
            metrics['avg_calendar_aging_rate'] = forecasts[:, :, 4].mean().item()
            metrics['avg_overall_health_decline'] = forecasts[:, :, 5].mean().item()
            
            # Calculate total degradation over forecast horizon
            total_degradation = forecasts.sum(dim=1)  # Sum over time
            metrics['total_capacity_loss'] = total_degradation[:, 0].mean().item()
            metrics['total_resistance_increase'] = total_degradation[:, 1].mean().item()
            
            # Estimate remaining useful life (simplified)
            health_decline_rate = forecasts[:, :, 5].mean()
            if health_decline_rate > 0:
                # Assume end-of-life at 70% health
                current_health = 0.9  # Assume starting at 90% health
                rul_hours = (current_health - 0.7) / health_decline_rate
                metrics['estimated_rul_hours'] = rul_hours.item()
                metrics['estimated_rul_days'] = rul_hours.item() / 24
        
        return metrics

class DegradationPhysicsConstraints(nn.Module):
    """
    Physics-informed constraints for degradation forecasting.
    """
    
    def __init__(self, config: DegradationConfig):
        super().__init__()
        self.config = config
        
    def forward(self, forecasts: torch.Tensor, hidden_states: torch.Tensor,
                time_features: Optional[Dict] = None) -> torch.Tensor:
        """
        Apply physics constraints to degradation forecasts.
        
        Args:
            forecasts (torch.Tensor): Raw forecasting predictions
            hidden_states (torch.Tensor): Hidden states from transformer
            time_features (Dict, optional): Time features
            
        Returns:
            torch.Tensor: Constrained forecasts
        """
        # Ensure degradation rates are non-negative
        forecasts = torch.clamp(forecasts, min=0.0)
        
        # Apply degradation bounds
        min_deg, max_deg = self.config.degradation_bounds
        forecasts = torch.clamp(forecasts, min=min_deg, max=max_deg)
        
        # Apply temperature sensitivity if temperature features available
        if time_features and 'temperature' in time_features:
            temp = time_features['temperature']
            temp_factor = torch.exp((temp - 25) / 25)  # Arrhenius-like relationship
            temp_factor = torch.clamp(temp_factor, 0.5, self.config.temperature_sensitivity)
            
            # Apply temperature factor to thermal degradation components
            if forecasts.size(-1) >= 3:
                forecasts[:, :, 2] = forecasts[:, :, 2] * temp_factor.unsqueeze(-1)
        
        # Ensure monotonic degradation (degradation should not reverse)
        for i in range(1, forecasts.size(1)):
            forecasts[:, i, :] = torch.maximum(forecasts[:, i, :], forecasts[:, i-1, :])
        
        return forecasts

# Factory function for easy model creation
def create_degradation_forecaster(config: Optional[DegradationConfig] = None) -> DegradationForecaster:
    """
    Factory function to create a DegradationForecaster model.
    
    Args:
        config (DegradationConfig, optional): Model configuration
        
    Returns:
        DegradationForecaster: Configured model instance
    """
    if config is None:
        config = DegradationConfig()
    
    model = DegradationForecaster(config)
    logger.info(f"Created DegradationForecaster with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model

# Model summary function
def get_forecaster_summary(model: DegradationForecaster) -> Dict[str, Union[int, str]]:
    """
    Get comprehensive model summary.
    
    Args:
        model (DegradationForecaster): Model instance
        
    Returns:
        Dict[str, Union[int, str]]: Model summary information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'model_name': 'DegradationForecaster',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'forecast_horizon': model.config.forecast_horizon,
        'uncertainty_enabled': model.config.enable_uncertainty,
        'seasonal_decomposition': model.config.enable_seasonal_decomposition,
        'config': model.config.__dict__
    }
