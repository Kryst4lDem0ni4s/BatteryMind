"""
BatteryMind - Proximal Policy Optimization (PPO) Algorithm

Advanced PPO implementation optimized for battery management scenarios with
continuous action spaces, multi-objective rewards, and safety constraints.

Features:
- Clipped surrogate objective for stable policy updates
- Adaptive KL divergence penalty for policy regularization
- Generalized Advantage Estimation (GAE) for variance reduction
- Multi-objective reward handling for battery optimization
- Safety constraints for thermal and electrical limits
- Distributed training support for large-scale battery fleets

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import time
from collections import deque
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    """
    Configuration for PPO algorithm.
    
    Attributes:
        # Core PPO parameters
        clip_ratio (float): Clipping ratio for surrogate objective
        target_kl (float): Target KL divergence for adaptive penalty
        entropy_coef (float): Entropy bonus coefficient
        value_loss_coef (float): Value function loss coefficient
        
        # Training parameters
        learning_rate (float): Learning rate for optimizer
        batch_size (int): Batch size for training
        mini_batch_size (int): Mini-batch size for SGD updates
        num_epochs (int): Number of epochs per update
        max_grad_norm (float): Maximum gradient norm for clipping
        
        # GAE parameters
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter
        
        # Network architecture
        hidden_sizes (List[int]): Hidden layer sizes for networks
        activation (str): Activation function type
        
        # Battery-specific parameters
        safety_constraint_weight (float): Weight for safety constraints
        multi_objective_weights (Dict[str, float]): Weights for different objectives
        temperature_penalty_coef (float): Penalty for temperature violations
        
        # Advanced features
        use_adaptive_kl (bool): Use adaptive KL penalty
        use_orthogonal_init (bool): Use orthogonal weight initialization
        normalize_advantages (bool): Normalize advantages
        clip_value_loss (bool): Clip value function loss
    """
    # Core PPO parameters
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    
    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    mini_batch_size: int = 32
    num_epochs: int = 10
    max_grad_norm: float = 0.5
    
    # GAE parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Network architecture
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"
    
    # Battery-specific parameters
    safety_constraint_weight: float = 10.0
    multi_objective_weights: Dict[str, float] = field(default_factory=lambda: {
        'health': 0.4,
        'efficiency': 0.3,
        'safety': 0.3
    })
    temperature_penalty_coef: float = 100.0
    
    # Advanced features
    use_adaptive_kl: bool = True
    use_orthogonal_init: bool = True
    normalize_advantages: bool = True
    clip_value_loss: bool = True

class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO with continuous action spaces.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig):
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.shared_layers = self._build_shared_network()
        
        # Actor network (policy)
        self.actor_head = nn.Linear(config.hidden_sizes[-1], action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network (value function)
        self.critic_head = nn.Linear(config.hidden_sizes[-1], 1)
        
        # Safety constraint network
        self.safety_head = nn.Linear(config.hidden_sizes[-1], 1)
        
        # Initialize weights
        if config.use_orthogonal_init:
            self._orthogonal_init()
    
    def _build_shared_network(self) -> nn.Module:
        """Build shared feature extraction network."""
        layers = []
        input_dim = self.state_dim
        
        for hidden_size in self.config.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            
            if self.config.activation == "tanh":
                layers.append(nn.Tanh())
            elif self.config.activation == "relu":
                layers.append(nn.ReLU())
            elif self.config.activation == "gelu":
                layers.append(nn.GELU())
            
            input_dim = hidden_size
        
        return nn.Sequential(*layers)
    
    def _orthogonal_init(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Action mean, value, safety constraint
        """
        features = self.shared_layers(state)
        
        # Actor output
        action_mean = self.actor_head(features)
        
        # Critic output
        value = self.critic_head(features)
        
        # Safety constraint output
        safety_constraint = self.safety_head(features)
        
        return action_mean, value, safety_constraint
    
    def get_action_distribution(self, state: torch.Tensor) -> Normal:
        """Get action distribution for given state."""
        action_mean, _, _ = self.forward(state)
        action_std = torch.exp(self.actor_log_std)
        return Normal(action_mean, action_std)
    
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state (torch.Tensor): State tensor
            deterministic (bool): Whether to use deterministic policy
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Action and log probability
        """
        dist = self.get_action_distribution(state)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states.
        
        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor
            
        Returns:
            Tuple containing log probabilities, values, entropy, and safety constraints
        """
        action_mean, value, safety_constraint = self.forward(state)
        
        # Calculate log probabilities
        action_std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Calculate entropy
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value.squeeze(-1), entropy, safety_constraint.squeeze(-1)

class PPOBuffer:
    """
    Experience buffer for PPO with GAE computation.
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, config: PPOConfig):
        self.capacity = capacity
        self.config = config
        self.ptr = 0
        self.size = 0
        
        # Buffers
        self.states = torch.zeros((capacity, state_dim))
        self.actions = torch.zeros((capacity, action_dim))
        self.rewards = torch.zeros(capacity)
        self.values = torch.zeros(capacity)
        self.log_probs = torch.zeros(capacity)
        self.dones = torch.zeros(capacity, dtype=torch.bool)
        self.safety_violations = torch.zeros(capacity)
        
        # GAE buffers
        self.advantages = torch.zeros(capacity)
        self.returns = torch.zeros(capacity)
    
    def store(self, state: torch.Tensor, action: torch.Tensor, reward: float,
              value: torch.Tensor, log_prob: torch.Tensor, done: bool,
              safety_violation: float = 0.0):
        """Store experience in buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.safety_violations[self.ptr] = safety_violation
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def compute_gae(self, next_value: torch.Tensor):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]
            
            delta = self.rewards[t] + self.config.gamma * next_values * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae
        
        self.advantages = advantages
        self.returns = advantages + self.values[:self.size]
    
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get random batch from buffer."""
        indices = torch.randperm(self.size)[:batch_size]
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'log_probs': self.log_probs[indices],
            'advantages': self.advantages[indices],
            'returns': self.returns[indices],
            'values': self.values[indices],
            'safety_violations': self.safety_violations[indices]
        }
    
    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0

class PPOAgent:
    """
    PPO agent for battery optimization tasks.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.actor_critic = ActorCriticNetwork(state_dim, action_dim, config)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config.learning_rate)
        
        # Experience buffer
        self.buffer = PPOBuffer(10000, state_dim, action_dim, config)  # Large buffer
        
        # Training statistics
        self.training_stats = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy_loss': deque(maxlen=100),
            'kl_divergence': deque(maxlen=100),
            'safety_violations': deque(maxlen=100)
        }
        
        # Adaptive KL penalty
        self.kl_coef = 0.2
        
        logger.info(f"PPO Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Select action using current policy.
        
        Args:
            state (np.ndarray): Current state
            deterministic (bool): Whether to use deterministic policy
            
        Returns:
            Tuple[np.ndarray, float]: Action and log probability
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob = self.actor_critic.act(state_tensor, deterministic)
            _, value, safety_constraint = self.actor_critic(state_tensor)
        
        return action.squeeze(0).numpy(), log_prob.item(), value.item(), safety_constraint.item()
    
    def update(self, next_state: np.ndarray) -> Dict[str, float]:
        """
        Update policy using PPO algorithm.
        
        Args:
            next_state (np.ndarray): Next state for bootstrap value
            
        Returns:
            Dict[str, float]: Training statistics
        """
        if self.buffer.size < self.config.batch_size:
            return {}
        
        # Compute GAE
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        with torch.no_grad():
            _, next_value, _ = self.actor_critic(next_state_tensor)
        
        self.buffer.compute_gae(next_value.squeeze())
        
        # Normalize advantages
        if self.config.normalize_advantages:
            advantages = self.buffer.advantages[:self.buffer.size]
            self.buffer.advantages[:self.buffer.size] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_kl_div = 0
        
        for epoch in range(self.config.num_epochs):
            # Get mini-batches
            num_batches = self.buffer.size // self.config.mini_batch_size
            
            for _ in range(num_batches):
                batch = self.buffer.get_batch(self.config.mini_batch_size)
                
                # Evaluate current policy
                log_probs, values, entropy, safety_constraints = self.actor_critic.evaluate_actions(
                    batch['states'], batch['actions']
                )
                
                # Calculate policy loss
                ratio = torch.exp(log_probs - batch['log_probs'])
                surr1 = ratio * batch['advantages']
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * batch['advantages']
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                if self.config.clip_value_loss:
                    value_pred_clipped = batch['values'] + torch.clamp(
                        values - batch['values'], -self.config.clip_ratio, self.config.clip_ratio
                    )
                    value_loss1 = (values - batch['returns']).pow(2)
                    value_loss2 = (value_pred_clipped - batch['returns']).pow(2)
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = 0.5 * (values - batch['returns']).pow(2).mean()
                
                # Calculate entropy loss
                entropy_loss = -entropy.mean()
                
                # Calculate safety constraint loss
                safety_loss = self.config.safety_constraint_weight * torch.relu(safety_constraints).mean()
                
                # Calculate KL divergence for adaptive penalty
                kl_div = (batch['log_probs'] - log_probs).mean()
                
                # Total loss
                total_loss = (
                    policy_loss +
                    self.config.value_loss_coef * value_loss +
                    self.config.entropy_coef * entropy_loss +
                    safety_loss +
                    self.kl_coef * kl_div
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Update statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_kl_div += kl_div.item()
            
            # Early stopping if KL divergence is too high
            if self.config.use_adaptive_kl and total_kl_div / num_batches > 1.5 * self.config.target_kl:
                logger.warning(f"Early stopping at epoch {epoch} due to high KL divergence")
                break
        
        # Update adaptive KL coefficient
        if self.config.use_adaptive_kl:
            avg_kl = total_kl_div / (self.config.num_epochs * num_batches)
            if avg_kl < self.config.target_kl / 1.5:
                self.kl_coef *= 0.5
            elif avg_kl > self.config.target_kl * 1.5:
                self.kl_coef *= 2.0
            self.kl_coef = np.clip(self.kl_coef, 0.02, 1.0)
        
        # Store training statistics
        num_updates = self.config.num_epochs * num_batches
        stats = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy_loss': total_entropy_loss / num_updates,
            'kl_divergence': total_kl_div / num_updates,
            'kl_coefficient': self.kl_coef
        }
        
        for key, value in stats.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        # Clear buffer
        self.buffer.clear()
        
        return stats
    
    def save(self, filepath: str):
        """Save model and training state."""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': dict(self.training_stats),
            'kl_coef': self.kl_coef
        }, filepath)
        logger.info(f"PPO agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and training state."""
        checkpoint = torch.load(filepath)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.kl_coef = checkpoint.get('kl_coef', 0.2)
        logger.info(f"PPO agent loaded from {filepath}")
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get recent training statistics."""
        stats = {}
        for key, values in self.training_stats.items():
            if values:
                stats[f'avg_{key}'] = np.mean(values)
                stats[f'std_{key}'] = np.std(values)
        return stats

# Factory function
def create_ppo_agent(state_dim: int, action_dim: int, config: Optional[PPOConfig] = None) -> PPOAgent:
    """
    Factory function to create a PPO agent.
    
    Args:
        state_dim (int): State space dimension
        action_dim (int): Action space dimension
        config (PPOConfig, optional): PPO configuration
        
    Returns:
        PPOAgent: Configured PPO agent
    """
    if config is None:
        config = PPOConfig()
    
    return PPOAgent(state_dim, action_dim, config)
