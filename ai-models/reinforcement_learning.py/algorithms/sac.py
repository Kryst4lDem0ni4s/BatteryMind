"""
BatteryMind - Soft Actor-Critic (SAC) Algorithm

Advanced off-policy reinforcement learning algorithm for continuous control
in battery management systems. SAC combines the sample efficiency of off-policy
methods with the stability of policy gradient methods through entropy regularization.

Features:
- Continuous action spaces for precise charging control
- Automatic entropy temperature tuning
- Twin critic networks for reduced overestimation bias
- Replay buffer integration for sample efficiency
- Battery-specific reward shaping and constraints
- Multi-objective optimization for safety and efficiency

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import copy
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SACConfig:
    """
    Configuration for Soft Actor-Critic algorithm.
    
    Attributes:
        # Network architecture
        actor_hidden_dims (List[int]): Hidden layer dimensions for actor network
        critic_hidden_dims (List[int]): Hidden layer dimensions for critic networks
        
        # Training parameters
        learning_rate_actor (float): Learning rate for actor network
        learning_rate_critic (float): Learning rate for critic networks
        learning_rate_alpha (float): Learning rate for entropy temperature
        
        # SAC specific parameters
        gamma (float): Discount factor
        tau (float): Soft update coefficient for target networks
        alpha (float): Initial entropy regularization coefficient
        automatic_entropy_tuning (bool): Enable automatic entropy tuning
        target_entropy (Optional[float]): Target entropy for automatic tuning
        
        # Training configuration
        batch_size (int): Batch size for training
        replay_buffer_size (int): Size of replay buffer
        warmup_steps (int): Number of random exploration steps
        update_frequency (int): Frequency of network updates
        target_update_frequency (int): Frequency of target network updates
        
        # Network regularization
        weight_decay (float): L2 regularization coefficient
        gradient_clip_norm (float): Gradient clipping norm
        
        # Battery-specific parameters
        safety_constraint_weight (float): Weight for safety constraints
        efficiency_reward_weight (float): Weight for efficiency rewards
        temperature_penalty_weight (float): Weight for temperature penalties
    """
    # Network architecture
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    
    # Training parameters
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 3e-4
    learning_rate_alpha: float = 3e-4
    
    # SAC specific parameters
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    automatic_entropy_tuning: bool = True
    target_entropy: Optional[float] = None
    
    # Training configuration
    batch_size: int = 256
    replay_buffer_size: int = 1000000
    warmup_steps: int = 10000
    update_frequency: int = 1
    target_update_frequency: int = 1
    
    # Network regularization
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    
    # Battery-specific parameters
    safety_constraint_weight: float = 10.0
    efficiency_reward_weight: float = 1.0
    temperature_penalty_weight: float = 5.0

class GaussianPolicy(nn.Module):
    """
    Gaussian policy network for continuous action spaces.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int],
                 action_scale: float = 1.0, action_bias: float = 0.0):
        super().__init__()
        self.action_scale = action_scale
        self.action_bias = action_bias
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            state (torch.Tensor): Input state
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log_std of action distribution
        """
        x = self.backbone(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy.
        
        Args:
            state (torch.Tensor): Input state
            deterministic (bool): Whether to use deterministic (mean) actions
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled actions and log probabilities
        """
        mean, log_std = self.forward(state)
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            
            # Reparameterization trick
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            
            # Calculate log probability with change of variables
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Scale and shift actions
        action = action * self.action_scale + self.action_bias
        
        return action, log_prob

class QNetwork(nn.Module):
    """
    Q-value network for state-action value estimation.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        # Build network layers
        layers = []
        prev_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Q-network.
        
        Args:
            state (torch.Tensor): Input state
            action (torch.Tensor): Input action
            
        Returns:
            torch.Tensor: Q-value estimate
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class SoftActorCritic:
    """
    Soft Actor-Critic algorithm implementation for battery management.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: SACConfig,
                 device: torch.device = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.actor = GaussianPolicy(
            state_dim, action_dim, config.actor_hidden_dims
        ).to(self.device)
        
        self.critic1 = QNetwork(
            state_dim, action_dim, config.critic_hidden_dims
        ).to(self.device)
        
        self.critic2 = QNetwork(
            state_dim, action_dim, config.critic_hidden_dims
        ).to(self.device)
        
        # Target networks
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        # Freeze target networks
        for param in self.target_critic1.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=config.learning_rate_actor,
            weight_decay=config.weight_decay
        )
        
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(),
            lr=config.learning_rate_critic,
            weight_decay=config.weight_decay
        )
        
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(),
            lr=config.learning_rate_critic,
            weight_decay=config.weight_decay
        )
        
        # Automatic entropy tuning
        if config.automatic_entropy_tuning:
            self.target_entropy = config.target_entropy or -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.learning_rate_alpha)
        else:
            self.alpha = config.alpha
        
        # Training statistics
        self.training_step = 0
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
        
        logger.info(f"SAC agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    @property
    def alpha(self) -> float:
        """Get current entropy regularization coefficient."""
        if self.config.automatic_entropy_tuning:
            return self.log_alpha.exp().item()
        else:
            return self.config.alpha
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action using the current policy.
        
        Args:
            state (np.ndarray): Current state
            deterministic (bool): Whether to use deterministic actions
            
        Returns:
            np.ndarray: Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _ = self.actor.sample(state_tensor, deterministic=deterministic)
        
        return action.cpu().numpy().flatten()
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update SAC networks using a batch of experiences.
        
        Args:
            batch (Dict[str, torch.Tensor]): Batch of experiences
            
        Returns:
            Dict[str, float]: Training metrics
        """
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Update critics
        critic_loss = self._update_critics(states, actions, rewards, next_states, dones)
        
        # Update actor
        actor_loss = self._update_actor(states)
        
        # Update entropy coefficient
        alpha_loss = 0.0
        if self.config.automatic_entropy_tuning:
            alpha_loss = self._update_alpha(states)
        
        # Update target networks
        if self.training_step % self.config.target_update_frequency == 0:
            self._soft_update_targets()
        
        self.training_step += 1
        
        # Store losses for monitoring
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.alpha_losses.append(alpha_loss)
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha
        }
    
    def _update_critics(self, states: torch.Tensor, actions: torch.Tensor,
                       rewards: torch.Tensor, next_states: torch.Tensor,
                       dones: torch.Tensor) -> float:
        """Update critic networks."""
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # Compute target Q-values
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            
            # Apply battery-specific reward shaping
            shaped_rewards = self._shape_rewards(rewards, states, actions)
            
            target_q = shaped_rewards + (1 - dones) * self.config.gamma * target_q
        
        # Current Q-values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        total_critic_loss = critic1_loss + critic2_loss
        
        # Update critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.config.gradient_clip_norm)
        self.critic1_optimizer.step()
        
        # Update critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.config.gradient_clip_norm)
        self.critic2_optimizer.step()
        
        return total_critic_loss.item()
    
    def _update_actor(self, states: torch.Tensor) -> float:
        """Update actor network."""
        # Sample actions from current policy
        actions, log_probs = self.actor.sample(states)
        
        # Compute Q-values
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q = torch.min(q1, q2)
        
        # Actor loss (maximize Q-value minus entropy)
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # Apply safety constraints
        safety_penalty = self._compute_safety_penalty(states, actions)
        actor_loss += self.config.safety_constraint_weight * safety_penalty
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip_norm)
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _update_alpha(self, states: torch.Tensor) -> float:
        """Update entropy regularization coefficient."""
        with torch.no_grad():
            _, log_probs = self.actor.sample(states)
        
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return alpha_loss.item()
    
    def _soft_update_targets(self):
        """Soft update target networks."""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
    
    def _shape_rewards(self, rewards: torch.Tensor, states: torch.Tensor,
                      actions: torch.Tensor) -> torch.Tensor:
        """Apply battery-specific reward shaping."""
        # Extract battery state information (assuming specific state structure)
        # This would be customized based on the actual battery environment
        
        # Efficiency bonus
        efficiency_bonus = self._compute_efficiency_bonus(states, actions)
        
        # Temperature penalty
        temperature_penalty = self._compute_temperature_penalty(states)
        
        # Combine rewards
        shaped_rewards = (
            rewards + 
            self.config.efficiency_reward_weight * efficiency_bonus -
            self.config.temperature_penalty_weight * temperature_penalty
        )
        
        return shaped_rewards
    
    def _compute_efficiency_bonus(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute efficiency bonus based on charging behavior."""
        # Placeholder implementation - would be customized for specific battery metrics
        # Example: bonus for maintaining optimal charging rates
        return torch.zeros(states.size(0), 1, device=self.device)
    
    def _compute_temperature_penalty(self, states: torch.Tensor) -> torch.Tensor:
        """Compute penalty for excessive battery temperature."""
        # Placeholder implementation - would extract temperature from state
        # Example: penalty for temperatures outside safe range
        return torch.zeros(states.size(0), 1, device=self.device)
    
    def _compute_safety_penalty(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute safety penalty for dangerous actions."""
        # Placeholder implementation - would check for safety violations
        # Example: penalty for actions that could damage the battery
        return torch.zeros(1, device=self.device)
    
    def save(self, filepath: str):
        """Save the SAC agent."""
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step
        }
        
        if self.config.automatic_entropy_tuning:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        
        torch.save(save_dict, filepath)
        logger.info(f"SAC agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the SAC agent."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
        if self.config.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.training_step = checkpoint.get('training_step', 0)
        
        logger.info(f"SAC agent loaded from {filepath}")
    
    def get_training_metrics(self) -> Dict[str, float]:
        """Get training metrics for monitoring."""
        if not self.actor_losses:
            return {}
        
        return {
            'avg_actor_loss': np.mean(self.actor_losses[-100:]),
            'avg_critic_loss': np.mean(self.critic_losses[-100:]),
            'avg_alpha_loss': np.mean(self.alpha_losses[-100:]) if self.alpha_losses else 0.0,
            'current_alpha': self.alpha,
            'training_steps': self.training_step
        }

# Factory function
def create_sac_agent(state_dim: int, action_dim: int, 
                    config: Optional[SACConfig] = None) -> SoftActorCritic:
    """
    Factory function to create a SAC agent.
    
    Args:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        config (SACConfig, optional): SAC configuration
        
    Returns:
        SoftActorCritic: Configured SAC agent
    """
    if config is None:
        config = SACConfig()
    
    return SoftActorCritic(state_dim, action_dim, config)
