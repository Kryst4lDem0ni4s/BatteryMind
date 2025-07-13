"""
BatteryMind - Policy Network for Reinforcement Learning

Advanced policy network architectures for reinforcement learning agents
in battery management systems. Supports multiple RL algorithms including
PPO, SAC, DDPG, and custom battery-specific policy designs.

Features:
- Multi-layer policy networks with customizable architectures
- Continuous and discrete action space support
- Battery-specific policy constraints and safety mechanisms
- Attention mechanisms for temporal dependencies
- Hierarchical policy structures for multi-objective optimization
- Adaptive exploration strategies
- Integration with battery physics constraints

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MultivariateNormal
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
class PolicyNetworkConfig:
    """
    Configuration for policy networks.
    
    Attributes:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        hidden_dims (List[int]): Hidden layer dimensions
        activation (str): Activation function type
        output_activation (str): Output activation function
        action_space_type (str): Type of action space ('continuous', 'discrete', 'mixed')
        use_layer_norm (bool): Whether to use layer normalization
        use_batch_norm (bool): Whether to use batch normalization
        dropout_rate (float): Dropout rate for regularization
        init_std (float): Initial standard deviation for continuous actions
        min_std (float): Minimum standard deviation for exploration
        max_std (float): Maximum standard deviation for exploration
        use_attention (bool): Whether to use attention mechanisms
        attention_heads (int): Number of attention heads
        use_recurrent (bool): Whether to use recurrent layers
        rnn_hidden_size (int): Hidden size for RNN layers
        safety_constraints (bool): Enable battery safety constraints
        constraint_penalty (float): Penalty factor for constraint violations
    """
    state_dim: int = 64
    action_dim: int = 4
    hidden_dims: List[int] = None
    activation: str = "relu"
    output_activation: str = "tanh"
    action_space_type: str = "continuous"
    use_layer_norm: bool = True
    use_batch_norm: bool = False
    dropout_rate: float = 0.1
    init_std: float = 0.3
    min_std: float = 0.01
    max_std: float = 2.0
    use_attention: bool = False
    attention_heads: int = 4
    use_recurrent: bool = False
    rnn_hidden_size: int = 128
    safety_constraints: bool = True
    constraint_penalty: float = 10.0
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]

class AttentionLayer(nn.Module):
    """
    Multi-head attention layer for policy networks.
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

class BatterySafetyConstraints(nn.Module):
    """
    Battery safety constraint layer for policy networks.
    """
    
    def __init__(self, action_dim: int, constraint_penalty: float = 10.0):
        super().__init__()
        self.action_dim = action_dim
        self.constraint_penalty = constraint_penalty
        
        # Define safety bounds for different battery parameters
        self.safety_bounds = {
            'voltage': (2.5, 4.2),      # Voltage limits (V)
            'current': (-50.0, 50.0),   # Current limits (A)
            'temperature': (0.0, 60.0), # Temperature limits (°C)
            'power': (0.0, 100.0)       # Power limits (kW)
        }
    
    def forward(self, actions: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply safety constraints to actions and compute penalty.
        
        Args:
            actions (torch.Tensor): Raw policy actions
            state (torch.Tensor): Current battery state
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (constrained_actions, penalty)
        """
        constrained_actions = actions.clone()
        penalty = torch.zeros(actions.shape[0], device=actions.device)
        
        # Apply voltage constraints
        if self.action_dim >= 1:
            voltage_min, voltage_max = self.safety_bounds['voltage']
            voltage_violations = torch.clamp(actions[:, 0], voltage_min, voltage_max) - actions[:, 0]
            constrained_actions[:, 0] = torch.clamp(actions[:, 0], voltage_min, voltage_max)
            penalty += self.constraint_penalty * torch.abs(voltage_violations)
        
        # Apply current constraints
        if self.action_dim >= 2:
            current_min, current_max = self.safety_bounds['current']
            current_violations = torch.clamp(actions[:, 1], current_min, current_max) - actions[:, 1]
            constrained_actions[:, 1] = torch.clamp(actions[:, 1], current_min, current_max)
            penalty += self.constraint_penalty * torch.abs(current_violations)
        
        # Apply temperature constraints
        if self.action_dim >= 3:
            temp_min, temp_max = self.safety_bounds['temperature']
            temp_violations = torch.clamp(actions[:, 2], temp_min, temp_max) - actions[:, 2]
            constrained_actions[:, 2] = torch.clamp(actions[:, 2], temp_min, temp_max)
            penalty += self.constraint_penalty * torch.abs(temp_violations)
        
        # Apply power constraints
        if self.action_dim >= 4:
            power_min, power_max = self.safety_bounds['power']
            power_violations = torch.clamp(actions[:, 3], power_min, power_max) - actions[:, 3]
            constrained_actions[:, 3] = torch.clamp(actions[:, 3], power_min, power_max)
            penalty += self.constraint_penalty * torch.abs(power_violations)
        
        return constrained_actions, penalty

class BasePolicyNetwork(nn.Module, ABC):
    """
    Abstract base class for policy networks.
    """
    
    def __init__(self, config: PolicyNetworkConfig):
        super().__init__()
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        
        # Safety constraints
        if config.safety_constraints:
            self.safety_layer = BatterySafetyConstraints(
                config.action_dim, config.constraint_penalty
            )
        else:
            self.safety_layer = None
    
    @abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy network."""
        pass
    
    @abstractmethod
    def sample_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from the policy."""
        pass
    
    @abstractmethod
    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of action given state."""
        pass

class ContinuousPolicyNetwork(BasePolicyNetwork):
    """
    Policy network for continuous action spaces.
    """
    
    def __init__(self, config: PolicyNetworkConfig):
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
        
        # Policy head (mean and log_std)
        self.mean_head = nn.Linear(feature_dim, config.action_dim)
        self.log_std_head = nn.Linear(feature_dim, config.action_dim)
        
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
        
        # Initialize policy head with smaller weights
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
        nn.init.constant_(self.log_std_head.weight, 0.0)
        nn.init.constant_(self.log_std_head.bias, math.log(self.config.init_std))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            state (torch.Tensor): Input state
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (mean, log_std)
        """
        # Feature extraction
        features = self.feature_extractor(state)
        
        # Attention mechanism
        if self.config.use_attention:
            # Reshape for attention (add sequence dimension)
            features = features.unsqueeze(1)
            features = self.attention(features)
            features = features.squeeze(1)
        
        # Recurrent processing
        if self.config.use_recurrent:
            features = features.unsqueeze(1)  # Add sequence dimension
            features, _ = self.rnn(features)
            features = features.squeeze(1)  # Remove sequence dimension
        
        # Policy outputs
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Apply output activation to mean
        if self.config.output_activation == 'tanh':
            mean = torch.tanh(mean)
        elif self.config.output_activation == 'sigmoid':
            mean = torch.sigmoid(mean)
        
        # Clamp log_std to reasonable bounds
        log_std = torch.clamp(
            log_std, 
            math.log(self.config.min_std), 
            math.log(self.config.max_std)
        )
        
        return mean, log_std
    
    def sample_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy.
        
        Args:
            state (torch.Tensor): Input state
            deterministic (bool): Whether to use deterministic policy
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (action, log_prob)
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros_like(mean).sum(dim=-1, keepdim=True)
        else:
            # Sample from normal distribution
            normal = Normal(mean, std)
            action = normal.rsample()  # Reparameterization trick
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Apply safety constraints
        if self.safety_layer is not None:
            action, penalty = self.safety_layer(action, state)
            log_prob = log_prob - penalty.unsqueeze(-1)
        
        return action, log_prob
    
    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of action given state.
        
        Args:
            state (torch.Tensor): Input state
            action (torch.Tensor): Action to evaluate
            
        Returns:
            torch.Tensor: Log probability
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        normal = Normal(mean, std)
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        
        return log_prob
    
    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute policy entropy.
        
        Args:
            state (torch.Tensor): Input state
            
        Returns:
            torch.Tensor: Policy entropy
        """
        _, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Entropy of multivariate normal distribution
        entropy = 0.5 * torch.log(2 * math.pi * math.e * std.pow(2)).sum(dim=-1, keepdim=True)
        
        return entropy

class DiscretePolicyNetwork(BasePolicyNetwork):
    """
    Policy network for discrete action spaces.
    """
    
    def __init__(self, config: PolicyNetworkConfig):
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
        
        # Policy head
        self.policy_head = nn.Linear(feature_dim, config.action_dim)
        
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
        
        # Initialize policy head with smaller weights
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy network.
        
        Args:
            state (torch.Tensor): Input state
            
        Returns:
            torch.Tensor: Action logits
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
        
        # Policy logits
        logits = self.policy_head(features)
        
        return logits
    
    def sample_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy.
        
        Args:
            state (torch.Tensor): Input state
            deterministic (bool): Whether to use deterministic policy
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (action, log_prob)
        """
        logits = self.forward(state)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1, keepdim=True)
            log_prob = F.log_softmax(logits, dim=-1).gather(-1, action)
        else:
            # Sample from categorical distribution
            categorical = Categorical(logits=logits)
            action = categorical.sample().unsqueeze(-1)
            log_prob = categorical.log_prob(action.squeeze(-1)).unsqueeze(-1)
        
        return action, log_prob
    
    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of action given state.
        
        Args:
            state (torch.Tensor): Input state
            action (torch.Tensor): Action to evaluate
            
        Returns:
            torch.Tensor: Log probability
        """
        logits = self.forward(state)
        log_prob = F.log_softmax(logits, dim=-1).gather(-1, action.long())
        
        return log_prob
    
    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute policy entropy.
        
        Args:
            state (torch.Tensor): Input state
            
        Returns:
            torch.Tensor: Policy entropy
        """
        logits = self.forward(state)
        categorical = Categorical(logits=logits)
        entropy = categorical.entropy().unsqueeze(-1)
        
        return entropy

class HierarchicalPolicyNetwork(nn.Module):
    """
    Hierarchical policy network for multi-objective battery optimization.
    """
    
    def __init__(self, config: PolicyNetworkConfig, num_objectives: int = 3):
        super().__init__()
        self.config = config
        self.num_objectives = num_objectives
        
        # Shared feature extractor
        self.shared_features = self._build_shared_features()
        
        # Objective-specific policy heads
        self.objective_policies = nn.ModuleList([
            ContinuousPolicyNetwork(config) for _ in range(num_objectives)
        ])
        
        # Objective selector (meta-policy)
        self.objective_selector = nn.Linear(config.hidden_dims[-1], num_objectives)
        
    def _build_shared_features(self) -> nn.Module:
        """Build shared feature extraction layers."""
        layers = []
        input_dim = self.config.state_dim
        
        for hidden_dim in self.config.hidden_dims[:-1]:  # Exclude last layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through hierarchical policy.
        
        Args:
            state (torch.Tensor): Input state
            
        Returns:
            Tuple containing objective weights and list of (mean, log_std) for each objective
        """
        # Shared features
        shared_features = self.shared_features(state)
        
        # Objective selection weights
        objective_weights = F.softmax(self.objective_selector(shared_features), dim=-1)
        
        # Objective-specific policies
        objective_outputs = []
        for policy in self.objective_policies:
            mean, log_std = policy.forward(state)
            objective_outputs.append((mean, log_std))
        
        return objective_weights, objective_outputs
    
    def sample_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from hierarchical policy."""
        objective_weights, objective_outputs = self.forward(state)
        
        # Sample actions from each objective policy
        actions = []
        log_probs = []
        
        for i, (mean, log_std) in enumerate(objective_outputs):
            std = torch.exp(log_std)
            
            if deterministic:
                action = mean
                log_prob = torch.zeros_like(mean).sum(dim=-1, keepdim=True)
            else:
                normal = Normal(mean, std)
                action = normal.rsample()
                log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
            
            actions.append(action)
            log_probs.append(log_prob)
        
        # Weighted combination of actions
        actions_tensor = torch.stack(actions, dim=1)  # [batch, num_objectives, action_dim]
        weights = objective_weights.unsqueeze(-1)  # [batch, num_objectives, 1]
        
        combined_action = (actions_tensor * weights).sum(dim=1)  # [batch, action_dim]
        
        # Weighted combination of log probabilities
        log_probs_tensor = torch.stack(log_probs, dim=1)  # [batch, num_objectives, 1]
        combined_log_prob = (log_probs_tensor * weights).sum(dim=1)  # [batch, 1]
        
        return combined_action, combined_log_prob

# Factory functions
def create_policy_network(config: PolicyNetworkConfig) -> BasePolicyNetwork:
    """
    Factory function to create appropriate policy network.
    
    Args:
        config (PolicyNetworkConfig): Policy network configuration
        
    Returns:
        BasePolicyNetwork: Configured policy network
    """
    if config.action_space_type == "continuous":
        return ContinuousPolicyNetwork(config)
    elif config.action_space_type == "discrete":
        return DiscretePolicyNetwork(config)
    else:
        raise ValueError(f"Unsupported action space type: {config.action_space_type}")

def create_battery_policy_network(state_dim: int, action_dim: int, 
                                 action_space_type: str = "continuous") -> BasePolicyNetwork:
    """
    Create a battery-specific policy network with optimized configuration.
    
    Args:
        state_dim (int): State space dimension
        action_dim (int): Action space dimension
        action_space_type (str): Type of action space
        
    Returns:
        BasePolicyNetwork: Battery-optimized policy network
    """
    config = PolicyNetworkConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[512, 256, 128],
        activation="relu",
        output_activation="tanh",
        action_space_type=action_space_type,
        use_layer_norm=True,
        dropout_rate=0.1,
        init_std=0.3,
        use_attention=True,
        attention_heads=4,
        safety_constraints=True,
        constraint_penalty=10.0
    )
    
    return create_policy_network(config)
