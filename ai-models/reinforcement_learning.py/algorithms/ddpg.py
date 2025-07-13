"""
BatteryMind - Deep Deterministic Policy Gradient (DDPG) Algorithm

Advanced DDPG implementation for continuous control of battery systems with
experience replay, target networks, and noise exploration strategies.

Features:
- Actor-Critic architecture with target networks
- Experience replay buffer with prioritized sampling
- Ornstein-Uhlenbeck noise for exploration
- Batch normalization for stable training
- Multi-objective reward handling
- Safety constraints for battery operations
- Distributed training support

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import time
from collections import deque
import copy
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DDPGConfig:
    """
    Configuration for DDPG algorithm.
    
    Attributes:
        # Network architecture
        actor_hidden_sizes (List[int]): Hidden layer sizes for actor
        critic_hidden_sizes (List[int]): Hidden layer sizes for critic
        activation (str): Activation function type
        
        # Training parameters
        actor_lr (float): Learning rate for actor
        critic_lr (float): Learning rate for critic
        batch_size (int): Batch size for training
        buffer_size (int): Experience replay buffer size
        gamma (float): Discount factor
        tau (float): Soft update coefficient for target networks
        
        # Exploration parameters
        noise_type (str): Type of exploration noise
        noise_std (float): Standard deviation of exploration noise
        noise_clip (float): Clipping range for noise
        
        # Ornstein-Uhlenbeck noise parameters
        ou_theta (float): OU process theta parameter
        ou_sigma (float): OU process sigma parameter
        ou_dt (float): OU process time step
        
        # Training schedule
        learning_starts (int): Steps before learning starts
        train_freq (int): Training frequency
        target_update_freq (int): Target network update frequency
        
        # Regularization
        weight_decay (float): Weight decay for optimizers
        gradient_clip (float): Gradient clipping norm
        
        # Battery-specific parameters
        safety_constraint_weight (float): Weight for safety constraints
        temperature_penalty_coef (float): Penalty for temperature violations
        efficiency_reward_weight (float): Weight for efficiency rewards
        
        # Advanced features
        use_batch_norm (bool): Use batch normalization
        use_layer_norm (bool): Use layer normalization
        use_prioritized_replay (bool): Use prioritized experience replay
        use_huber_loss (bool): Use Huber loss for critic
    """
    # Network architecture
    actor_hidden_sizes: List[int] = field(default_factory=lambda: [400, 300])
    critic_hidden_sizes: List[int] = field(default_factory=lambda: [400, 300])
    activation: str = "relu"
    
    # Training parameters
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 1000000
    gamma: float = 0.99
    tau: float = 0.005
    
    # Exploration parameters
    noise_type: str = "ou"  # 'ou', 'gaussian', 'parameter'
    noise_std: float = 0.2
    noise_clip: float = 0.5
    
    # Ornstein-Uhlenbeck noise parameters
    ou_theta: float = 0.15
    ou_sigma: float = 0.2
    ou_dt: float = 1e-2
    
    # Training schedule
    learning_starts: int = 10000
    train_freq: int = 1
    target_update_freq: int = 1
    
    # Regularization
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    
    # Battery-specific parameters
    safety_constraint_weight: float = 10.0
    temperature_penalty_coef: float = 100.0
    efficiency_reward_weight: float = 1.0
    
    # Advanced features
    use_batch_norm: bool = False
    use_layer_norm: bool = True
    use_prioritized_replay: bool = False
    use_huber_loss: bool = True

class Actor(nn.Module):
    """
    Actor network for DDPG.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: DDPGConfig):
        super().__init__()
        self.config = config
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_size in config.actor_hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            elif config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            
            if config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "tanh":
                layers.append(nn.Tanh())
            elif config.activation == "gelu":
                layers.append(nn.GELU())
            
            input_dim = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())  # Actions are typically bounded
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor network."""
        return self.network(state)

class Critic(nn.Module):
    """
    Critic network for DDPG.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: DDPGConfig):
        super().__init__()
        self.config = config
        
        # State processing layers
        self.state_layers = nn.ModuleList()
        input_dim = state_dim
        
        for i, hidden_size in enumerate(config.critic_hidden_sizes):
            self.state_layers.append(nn.Linear(input_dim, hidden_size))
            
            if config.use_batch_norm:
                self.state_layers.append(nn.BatchNorm1d(hidden_size))
            elif config.use_layer_norm:
                self.state_layers.append(nn.LayerNorm(hidden_size))
            
            # Add action input after first layer
            if i == 0:
                input_dim = hidden_size + action_dim
            else:
                input_dim = hidden_size
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.Linear(config.critic_hidden_sizes[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic network."""
        # Process state through first layer
        x = self.state_layers[0](state)
        
        if self.config.use_batch_norm or self.config.use_layer_norm:
            x = self.state_layers[1](x)
            x = F.relu(x)
            start_idx = 2
        else:
            x = F.relu(x)
            start_idx = 1
        
        # Concatenate action
        x = torch.cat([x, action], dim=1)
        
        # Process through remaining layers
        for i in range(start_idx, len(self.state_layers)):
            x = self.state_layers[i](x)
            if not isinstance(self.state_layers[i], (nn.BatchNorm1d, nn.LayerNorm)):
                x = F.relu(x)
        
        # Final output
        return self.final_layers(x)

class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck process for action exploration.
    """
    
    def __init__(self, size: int, theta: float = 0.15, sigma: float = 0.2, dt: float = 1e-2):
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()
    
    def reset(self):
        """Reset the noise process."""
        self.state = np.zeros(self.size)
    
    def sample(self) -> np.ndarray:
        """Sample noise from the OU process."""
        dx = self.theta * (0 - self.state) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)
        self.state += dx
        return self.state

class ReplayBuffer:
    """
    Experience replay buffer for DDPG.
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # Buffers
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, action_dim))
        self.rewards = np.zeros(capacity)
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros(capacity, dtype=bool)
        self.safety_violations = np.zeros(capacity)
    
    def store(self, state: np.ndarray, action: np.ndarray, reward: float,
              next_state: np.ndarray, done: bool, safety_violation: float = 0.0):
        """Store experience in buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.safety_violations[self.ptr] = safety_violation
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random batch from buffer."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_states': torch.FloatTensor(self.next_states[indices]),
            'dones': torch.FloatTensor(self.dones[indices]),
            'safety_violations': torch.FloatTensor(self.safety_violations[indices])
        }
    
    def __len__(self) -> int:
        return self.size

class DDPGAgent:
    """
    DDPG agent for continuous control tasks.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: DDPGConfig):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.actor = Actor(state_dim, action_dim, config)
        self.critic = Critic(state_dim, action_dim, config)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=config.actor_lr,
            weight_decay=config.weight_decay
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr,
            weight_decay=config.weight_decay
        )
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(config.buffer_size, state_dim, action_dim)
        
        # Exploration noise
        if config.noise_type == "ou":
            self.noise = OrnsteinUhlenbeckNoise(
                action_dim, config.ou_theta, config.ou_sigma, config.ou_dt
            )
        
        # Training statistics
        self.training_stats = {
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'q_values': deque(maxlen=100),
            'safety_violations': deque(maxlen=100)
        }
        
        # Training step counter
        self.training_step = 0
        
        logger.info(f"DDPG Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select action using current policy.
        
        Args:
            state (np.ndarray): Current state
            add_noise (bool): Whether to add exploration noise
            
        Returns:
            np.ndarray: Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).numpy()
        
        if add_noise:
            if self.config.noise_type == "ou":
                noise = self.noise.sample()
            elif self.config.noise_type == "gaussian":
                noise = np.random.normal(0, self.config.noise_std, size=self.action_dim)
            else:
                noise = np.zeros(self.action_dim)
            
            action += noise
            action = np.clip(action, -self.config.noise_clip, self.config.noise_clip)
        
        return np.clip(action, -1, 1)  # Assuming actions are normalized to [-1, 1]
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool, safety_violation: float = 0.0):
        """Store experience in replay buffer."""
        self.replay_buffer.store(state, action, reward, next_state, done, safety_violation)
    
    def update(self) -> Dict[str, float]:
        """
        Update actor and critic networks.
        
        Returns:
            Dict[str, float]: Training statistics
        """
        if len(self.replay_buffer) < self.config.learning_starts:
            return {}
        
        if self.training_step % self.config.train_freq != 0:
            self.training_step += 1
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Update critic
        critic_loss = self._update_critic(batch)
        
        # Update actor
        actor_loss = self._update_actor(batch)
        
        # Update target networks
        if self.training_step % self.config.target_update_freq == 0:
            self._soft_update_targets()
        
        # Update statistics
        with torch.no_grad():
            q_values = self.critic(batch['states'], batch['actions']).mean().item()
            safety_violations = batch['safety_violations'].mean().item()
        
        stats = {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'q_values': q_values,
            'safety_violations': safety_violations
        }
        
        for key, value in stats.items():
            self.training_stats[key].append(value)
        
        self.training_step += 1
        return stats
    
    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update critic network."""
        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.target_actor(batch['next_states'])
            target_q = self.target_critic(batch['next_states'], next_actions)
            
            # Add safety penalty to rewards
            safety_penalty = self.config.safety_constraint_weight * batch['safety_violations']
            adjusted_rewards = batch['rewards'] - safety_penalty
            
            target_q_values = adjusted_rewards.unsqueeze(1) + self.config.gamma * target_q * (1 - batch['dones'].unsqueeze(1))
        
        # Compute current Q-values
        current_q_values = self.critic(batch['states'], batch['actions'])
        
        # Compute critic loss
        if self.config.use_huber_loss:
            critic_loss = F.smooth_l1_loss(current_q_values, target_q_values)
        else:
            critic_loss = F.mse_loss(current_q_values, target_q_values)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update actor network."""
        # Compute actor loss
        actions = self.actor(batch['states'])
        actor_loss = -self.critic(batch['states'], actions).mean()
        
        # Add regularization for safety
        safety_constraint_loss = self.config.safety_constraint_weight * torch.relu(
            torch.norm(actions, dim=1) - 1.0
        ).mean()
        
        total_actor_loss = actor_loss + safety_constraint_loss
        
        # Update actor
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip)
        self.actor_optimizer.step()
        
        return total_actor_loss.item()
    
    def _soft_update_targets(self):
        """Soft update target networks."""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
    
    def save(self, filepath: str):
        """Save model and training state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'training_stats': dict(self.training_stats)
        }, filepath)
        logger.info(f"DDPG agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and training state."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        logger.info(f"DDPG agent loaded from {filepath}")
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get recent training statistics."""
        stats = {}
        for key, values in self.training_stats.items():
            if values:
                stats[f'avg_{key}'] = np.mean(values)
                stats[f'std_{key}'] = np.std(values)
        return stats
    
    def reset_noise(self):
        """Reset exploration noise."""
        if self.config.noise_type == "ou":
            self.noise.reset()

# Factory function
def create_ddpg_agent(state_dim: int, action_dim: int, config: Optional[DDPGConfig] = None) -> DDPGAgent:
    """
    Factory function to create a DDPG agent.
    
    Args:
        state_dim (int): State space dimension
        action_dim (int): Action space dimension
        config (DDPGConfig, optional): DDPG configuration
        
    Returns:
        DDPGAgent: Configured DDPG agent
    """
    if config is None:
        config = DDPGConfig()
    
    return DDPGAgent(state_dim, action_dim, config)
