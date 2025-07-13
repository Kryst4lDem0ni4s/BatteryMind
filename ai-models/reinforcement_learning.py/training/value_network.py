"""
BatteryMind - Value Network for Reinforcement Learning

Advanced value network architectures for reinforcement learning agents
in battery management systems. Supports state-value and action-value
functions with battery-specific optimizations.

Features:
- State-value and action-value network architectures
- Dueling network architectures for improved learning
- Distributional value networks for uncertainty quantification
- Multi-objective value estimation
- Temporal difference learning optimizations
- Battery physics-aware value estimation
- Ensemble value networks for robust predictions

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValueNetworkConfig:
    """
    Configuration for value networks.
    
    Attributes:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space (for Q-networks)
        hidden_dims (List[int]): Hidden layer dimensions
        activation (str): Activation function type
        output_activation (str): Output activation function
        use_layer_norm (bool): Whether to use layer normalization
        use_batch_norm (bool): Whether to use batch normalization
        dropout_rate (float): Dropout rate for regularization
        dueling_architecture (bool): Use dueling network architecture
        distributional (bool): Use distributional value networks
        num_atoms (int): Number of atoms for distributional networks
        v_min (float): Minimum value for distributional networks
        v_max (float): Maximum value for distributional networks
        use_attention (bool): Whether to use attention mechanisms
        attention_heads (int): Number of attention heads
        use_recurrent (bool): Whether to use recurrent layers
        rnn_hidden_size (int): Hidden size for RNN layers
        ensemble_size (int): Number of networks in ensemble
        uncertainty_estimation (bool): Enable uncertainty estimation
        bootstrap_heads (int): Number of bootstrap heads for uncertainty
    """
    state_dim: int = 64
    action_dim: int = 4
    hidden_dims: List[int] = None
    activation: str = "relu"
    output_activation: str = "linear"
    use_layer_norm: bool = True
    use_batch_norm: bool = False
    dropout_rate: float = 0.1
    dueling_architecture: bool = True
    distributional: bool = False
    num_atoms: int = 51
    v_min: float = -100.0
    v_max: float = 100.0
    use_attention: bool = False
    attention_heads: int = 4
    use_recurrent: bool = False
    rnn_hidden_size: int = 128
    ensemble_size: int = 1
    uncertainty_estimation: bool = False
    bootstrap_heads: int = 5
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]

class AttentionLayer(nn.Module):
    """
    Multi-head attention layer for value networks.
    """
    
    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.output = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, input_dim = x.shape
        
        # Compute Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, input_dim
        )
        
        output = self.output(context)
        return output

class BaseValueNetwork(nn.Module, ABC):
    """
    Abstract base class for value networks.
    """
    
    def __init__(self, config: ValueNetworkConfig):
        super().__init__()
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
    
    @abstractmethod
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the value network."""
        pass

class StateValueNetwork(BaseValueNetwork):
    """
    State-value network (V-function) for policy evaluation.
    """
    
    def __init__(self, config: ValueNetworkConfig):
        super().__init__(config)
        
        # Build feature extractor
        self.feature_extractor = self._build_feature_extractor()
        
        # Attention layer
        if config.use_attention:
            self.attention = AttentionLayer(
                config.hidden_dims[-1], config.attention_heads, config.dropout_rate
            )
        
        # Recurrent layer
        if config.use_recurrent:
            self.rnn = nn.LSTM(
                config.hidden_dims[-1], config.rnn_hidden_size, batch_first=True
            )
            feature_dim = config.rnn_hidden_size
        else:
            feature_dim = config.hidden_dims[-1]
        
        # Value head
        if config.distributional:
            self.value_head = nn.Linear(feature_dim, config.num_atoms)
            self.register_buffer('support', torch.linspace(config.v_min, config.v_max, config.num_atoms))
        else:
            self.value_head = nn.Linear(feature_dim, 1)
        
        # Uncertainty estimation heads
        if config.uncertainty_estimation:
            self.bootstrap_heads = nn.ModuleList([
                nn.Linear(feature_dim, 1) for _ in range(config.bootstrap_heads)
            ])
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _build_feature_extractor(self) -> nn.Module:
        """Build feature extraction layers."""
        layers = []
        input_dim = self.config.state_dim
        
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif self.config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(self._get_activation(self.config.activation))
            
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
            
            input_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'swish': nn.SiLU(),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def _initialize_parameters(self):
        """Initialize network parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the state-value network.
        
        Args:
            state (torch.Tensor): Input state
            action (torch.Tensor, optional): Not used for state-value networks
            
        Returns:
            torch.Tensor: State value or value distribution
        """
        # Feature extraction
        features = self.feature_extractor(state)
        
        # Attention mechanism
        if self.config.use_attention:
            features = features.unsqueeze(1)
            features = self.attention(features)
            features = features.squeeze(1)
        
        # Recurrent processing
        if self.config.use_recurrent:
            features = features.unsqueeze(1)
            features, _ = self.rnn(features)
            features = features.squeeze(1)
        
        # Value estimation
        if self.config.distributional:
            # Distributional value network
            logits = self.value_head(features)
            probabilities = F.softmax(logits, dim=-1)
            return probabilities
        else:
            # Standard value network
            value = self.value_head(features)
            return value
    
    def get_value_distribution(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get value distribution for distributional networks.
        
        Args:
            state (torch.Tensor): Input state
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (support, probabilities)
        """
        if not self.config.distributional:
            raise ValueError("Network is not distributional")
        
        probabilities = self.forward(state)
        return self.support, probabilities
    
    def get_uncertainty(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get value uncertainty estimation.
        
        Args:
            state (torch.Tensor): Input state
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (mean_value, uncertainty)
        """
        if not self.config.uncertainty_estimation:
            raise ValueError("Uncertainty estimation not enabled")
        
        # Get features
        features = self.feature_extractor(state)
        
        # Get predictions from bootstrap heads
        predictions = []
        for head in self.bootstrap_heads:
            pred = head(features)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_heads, batch_size, 1]
        
        mean_value = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_value, uncertainty

class ActionValueNetwork(BaseValueNetwork):
    """
    Action-value network (Q-function) for action evaluation.
    """
    
    def __init__(self, config: ValueNetworkConfig):
        super().__init__(config)
        
        # Build feature extractor
        self.feature_extractor = self._build_feature_extractor()
        
        # Attention layer
        if config.use_attention:
            self.attention = AttentionLayer(
                config.hidden_dims[-1], config.attention_heads, config.dropout_rate
            )
        
        # Recurrent layer
        if config.use_recurrent:
            self.rnn = nn.LSTM(
                config.hidden_dims[-1], config.rnn_hidden_size, batch_first=True
            )
            feature_dim = config.rnn_hidden_size
        else:
            feature_dim = config.hidden_dims[-1]
        
        # Dueling architecture
        if config.dueling_architecture:
            # Advantage stream
            self.advantage_head = nn.Linear(feature_dim, config.action_dim)
            # Value stream
            self.value_head = nn.Linear(feature_dim, 1)
        else:
            # Standard Q-network
            if config.distributional:
                self.q_head = nn.Linear(feature_dim, config.action_dim * config.num_atoms)
                self.register_buffer('support', torch.linspace(config.v_min, config.v_max, config.num_atoms))
            else:
                self.q_head = nn.Linear(feature_dim, config.action_dim)
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _build_feature_extractor(self) -> nn.Module:
        """Build feature extraction layers."""
        layers = []
        input_dim = self.config.state_dim
        
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif self.config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(self._get_activation(self.config.activation))
            
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
            
            input_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'swish': nn.SiLU(),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def _initialize_parameters(self):
        """Initialize network parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the action-value network.
        
        Args:
            state (torch.Tensor): Input state
            action (torch.Tensor, optional): Specific action for continuous control
            
        Returns:
            torch.Tensor: Action values or Q-value distribution
        """
        # Feature extraction
        features = self.feature_extractor(state)
        
        # Attention mechanism
        if self.config.use_attention:
            features = features.unsqueeze(1)
            features = self.attention(features)
            features = features.squeeze(1)
        
        # Recurrent processing
        if self.config.use_recurrent:
            features = features.unsqueeze(1)
            features, _ = self.rnn(features)
            features = features.squeeze(1)
        
        # Q-value estimation
        if self.config.dueling_architecture:
            # Dueling network
            advantage = self.advantage_head(features)
            value = self.value_head(features)
            
            # Combine value and advantage
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
            return q_values
        else:
            # Standard Q-network
            if self.config.distributional:
                # Distributional Q-network
                logits = self.q_head(features)
                logits = logits.view(-1, self.config.action_dim, self.config.num_atoms)
                probabilities = F.softmax(logits, dim=-1)
                return probabilities
            else:
                q_values = self.q_head(features)
                return q_values
    
    def get_q_distribution(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get Q-value distribution for distributional networks.
        
        Args:
            state (torch.Tensor): Input state
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (support, probabilities)
        """
        if not self.config.distributional:
            raise ValueError("Network is not distributional")
        
        probabilities = self.forward(state)
        return self.support, probabilities

class ContinuousActionValueNetwork(BaseValueNetwork):
    """
    Action-value network for continuous action spaces (Critic in Actor-Critic).
    """
    
    def __init__(self, config: ValueNetworkConfig):
        super().__init__(config)
        
        # State feature extractor
        self.state_features = self._build_state_features()
        
        # Action feature extractor
        self.action_features = self._build_action_features()
        
        # Combined feature processing
        combined_dim = config.hidden_dims[-1] + config.action_dim
        self.combined_features = self._build_combined_features(combined_dim)
        
        # Value head
        self.value_head = nn.Linear(config.hidden_dims[-1], 1)
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _build_state_features(self) -> nn.Module:
        """Build state feature extraction layers."""
        layers = []
        input_dim = self.config.state_dim
        
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if self.config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(self._get_activation(self.config.activation))
            input_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_action_features(self) -> nn.Module:
        """Build action feature extraction layers."""
        return nn.Linear(self.config.action_dim, self.config.action_dim)
    
    def _build_combined_features(self, input_dim: int) -> nn.Module:
        """Build combined feature processing layers."""
        layers = []
        
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if self.config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(self._get_activation(self.config.activation))
            
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
            
            input_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'swish': nn.SiLU(),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def _initialize_parameters(self):
        """Initialize network parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the continuous action-value network.
        
        Args:
            state (torch.Tensor): Input state
            action (torch.Tensor): Input action
            
        Returns:
            torch.Tensor: Q-value for state-action pair
        """
        # Extract state features
        state_features = self.state_features(state)
        
        # Extract action features
        action_features = self.action_features(action)
        
        # Combine state and action features
        combined = torch.cat([state_features, action_features], dim=-1)
        
        # Process combined features
        features = self.combined_features(combined)
        
        # Compute Q-value
        q_value = self.value_head(features)
        
        return q_value

class EnsembleValueNetwork(nn.Module):
    """
    Ensemble of value networks for improved robustness and uncertainty estimation.
    """
    
    def __init__(self, config: ValueNetworkConfig, network_type: str = "state_value"):
        super().__init__()
        self.config = config
        self.network_type = network_type
        self.ensemble_size = config.ensemble_size
        
        # Create ensemble of networks
        self.networks = nn.ModuleList()
        for _ in range(self.ensemble_size):
            if network_type == "state_value":
                network = StateValueNetwork(config)
            elif network_type == "action_value":
                network = ActionValueNetwork(config)
            elif network_type == "continuous_action_value":
                network = ContinuousActionValueNetwork(config)
            else:
                raise ValueError(f"Unknown network type: {network_type}")
            
            self.networks.append(network)
    
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through ensemble networks.
        
        Args:
            state (torch.Tensor): Input state
            action (torch.Tensor, optional): Input action
            
        Returns:
            torch.Tensor: Ensemble predictions
        """
        predictions = []
        
        for network in self.networks:
            if self.network_type == "continuous_action_value":
                pred = network(state, action)
            else:
                pred = network(state, action)
            predictions.append(pred)
        
        # Stack predictions
        ensemble_predictions = torch.stack(predictions, dim=0)  # [ensemble_size, batch_size, ...]
        
        return ensemble_predictions
    
    def get_mean_prediction(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get mean prediction from ensemble."""
        predictions = self.forward(state, action)
        return predictions.mean(dim=0)
    
    def get_uncertainty(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prediction uncertainty from ensemble."""
        predictions = self.forward(state, action)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty

# Factory functions
def create_value_network(config: ValueNetworkConfig, network_type: str = "state_value") -> BaseValueNetwork:
    """
    Factory function to create appropriate value network.
    
    Args:
        config (ValueNetworkConfig): Value network configuration
        network_type (str): Type of value network
        
    Returns:
        BaseValueNetwork: Configured value network
    """
    if network_type == "state_value":
        return StateValueNetwork(config)
    elif network_type == "action_value":
        return ActionValueNetwork(config)
    elif network_type == "continuous_action_value":
        return ContinuousActionValueNetwork(config)
    elif network_type == "ensemble":
        return EnsembleValueNetwork(config)
    else:
        raise ValueError(f"Unknown network type: {network_type}")

def create_battery_value_network(state_dim: int, action_dim: int = None, 
                                network_type: str = "state_value") -> BaseValueNetwork:
    """
    Create a battery-specific value network with optimized configuration.
    
    Args:
        state_dim (int): State space dimension
        action_dim (int, optional): Action space dimension
        network_type (str): Type of value network
        
    Returns:
        BaseValueNetwork: Battery-optimized value network
    """
    config = ValueNetworkConfig(
        state_dim=state_dim,
        action_dim=action_dim or 4,
        hidden_dims=[512, 256, 128],
        activation="relu",
        use_layer_norm=True,
        dropout_rate=0.1,
        dueling_architecture=True,
        distributional=False,
        use_attention=True,
        attention_heads=4,
        uncertainty_estimation=True,
        bootstrap_heads=5
    )
    
    return create_value_network(config, network_type)
