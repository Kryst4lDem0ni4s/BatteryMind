"""
BatteryMind - Deep Q-Network (DQN) Algorithm

Deep Q-Network implementation with experience replay and target networks
for discrete action spaces in battery management systems. Includes Double DQN
and Dueling DQN variants for improved performance.

Features:
- Experience replay buffer for sample efficiency
- Target network for stable training
- Double DQN to reduce overestimation bias
- Dueling DQN architecture for better value estimation
- Prioritized experience replay (optional)
- Battery-specific action discretization and reward shaping
- Multi-step returns for improved learning

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import deque, namedtuple
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class DQNConfig:
    """
    Configuration for Deep Q-Network algorithm.
    
    Attributes:
        # Network architecture
        hidden_dims (List[int]): Hidden layer dimensions
        dueling (bool): Use dueling DQN architecture
        
        # Training parameters
        learning_rate (float): Learning rate for optimizer
        gamma (float): Discount factor
        epsilon_start (float): Initial exploration rate
        epsilon_end (float): Final exploration rate
        epsilon_decay (int): Steps for epsilon decay
        
        # Experience replay
        replay_buffer_size (int): Size of replay buffer
        batch_size (int): Batch size for training
        min_replay_size (int): Minimum replay buffer size before training
        
        # Target network
        target_update_frequency (int): Frequency of target network updates
        soft_update (bool): Use soft updates for target network
        tau (float): Soft update coefficient
        
        # DQN variants
        double_dqn (bool): Use Double DQN
        multi_step (int): Number of steps for multi-step returns
        
        # Prioritized replay
        prioritized_replay (bool): Use prioritized experience replay
        alpha (float): Prioritization exponent
        beta_start (float): Initial importance sampling weight
        beta_end (float): Final importance sampling weight
        beta_decay_steps (int): Steps for beta annealing
        
        # Network regularization
        weight_decay (float): L2 regularization coefficient
        gradient_clip_norm (float): Gradient clipping norm
        
        # Battery-specific parameters
        action_discretization (int): Number of discrete actions
        safety_constraint_weight (float): Weight for safety constraints
        efficiency_reward_weight (float): Weight for efficiency rewards
    """
    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dueling: bool = True
    
    # Training parameters
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 100000
    
    # Experience replay
    replay_buffer_size: int = 1000000
    batch_size: int = 32
    min_replay_size: int = 10000
    
    # Target network
    target_update_frequency: int = 1000
    soft_update: bool = False
    tau: float = 0.005
    
    # DQN variants
    double_dqn: bool = True
    multi_step: int = 1
    
    # Prioritized replay
    prioritized_replay: bool = False
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    beta_decay_steps: int = 100000
    
    # Network regularization
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    
    # Battery-specific parameters
    action_discretization: int = 11  # 11 discrete charging rates
    safety_constraint_weight: float = 10.0
    efficiency_reward_weight: float = 1.0

class DuelingDQN(nn.Module):
    """
    Dueling DQN network architecture.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int],
                 dueling: bool = True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling
        
        # Shared feature layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        if dueling:
            # Dueling architecture: separate value and advantage streams
            self.value_stream = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], 1)
            )
            
            self.advantage_stream = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], action_dim)
            )
        else:
            # Standard DQN architecture
            self.q_network = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], action_dim)
            )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DQN network.
        
        Args:
            state (torch.Tensor): Input state
            
        Returns:
            torch.Tensor: Q-values for all actions
        """
        features = self.feature_layers(state)
        
        if self.dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            
            # Combine value and advantage
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = self.q_network(features)
        
        return q_values

class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def add(self, experience: Experience):
        """Add experience to buffer with maximum priority."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with prioritized sampling."""
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough experiences in buffer")
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class ReplayBuffer:
    """
    Standard experience replay buffer.
    """
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch from buffer."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DeepQNetwork:
    """
    Deep Q-Network algorithm implementation for battery management.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig,
                 device: torch.device = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        
        # Initialize networks
        self.q_network = DuelingDQN(
            state_dim, action_dim, config.hidden_dims, config.dueling
        ).to(self.device)
        
        self.target_network = DuelingDQN(
            state_dim, action_dim, config.hidden_dims, config.dueling
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Experience replay buffer
        if config.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                config.replay_buffer_size, config.alpha
            )
        else:
            self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        
        # Training state
        self.training_step = 0
        self.epsilon = config.epsilon_start
        self.beta = config.beta_start
        
        # Multi-step return buffer
        if config.multi_step > 1:
            self.multi_step_buffer = deque(maxlen=config.multi_step)
        
        # Training statistics
        self.losses = []
        self.q_values = []
        
        logger.info(f"DQN agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state (np.ndarray): Current state
            deterministic (bool): Whether to use deterministic actions
            
        Returns:
            int: Selected action
        """
        if not deterministic and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        experience = Experience(state, action, reward, next_state, done)
        
        if self.config.multi_step > 1:
            self.multi_step_buffer.append(experience)
            
            if len(self.multi_step_buffer) == self.config.multi_step:
                # Calculate multi-step return
                multi_step_experience = self._calculate_multi_step_return()
                self.replay_buffer.add(multi_step_experience)
        else:
            self.replay_buffer.add(experience)
    
    def _calculate_multi_step_return(self) -> Experience:
        """Calculate multi-step return for experience."""
        first_exp = self.multi_step_buffer[0]
        last_exp = self.multi_step_buffer[-1]
        
        # Calculate discounted return
        multi_step_return = 0
        gamma_power = 1
        
        for exp in self.multi_step_buffer:
            multi_step_return += gamma_power * exp.reward
            gamma_power *= self.config.gamma
            
            if exp.done:
                break
        
        return Experience(
            first_exp.state,
            first_exp.action,
            multi_step_return,
            last_exp.next_state,
            last_exp.done
        )
    
    def update(self) -> Optional[Dict[str, float]]:
        """
        Update DQN networks using experience replay.
        
        Returns:
            Optional[Dict[str, float]]: Training metrics
        """
        if len(self.replay_buffer) < self.config.min_replay_size:
            return None
        
        # Sample batch
        if self.config.prioritized_replay:
            experiences, indices, weights = self.replay_buffer.sample(
                self.config.batch_size, self.beta
            )
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            experiences = self.replay_buffer.sample(self.config.batch_size)
            weights = torch.ones(self.config.batch_size).to(self.device)
            indices = None
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            # Apply battery-specific reward shaping
            shaped_rewards = self._shape_rewards(rewards, states, actions)
            
            target_q_values = shaped_rewards.unsqueeze(1) + (
                ~dones.unsqueeze(1) * (self.config.gamma ** self.config.multi_step) * next_q_values
            )
        
        # Compute loss
        td_errors = current_q_values - target_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        
        # Update priorities
        if self.config.prioritized_replay:
            priorities = td_errors.abs().detach().cpu().numpy().flatten()
            self.replay_buffer.update_priorities(indices, priorities + 1e-6)
        
        # Update target network
        if self.training_step % self.config.target_update_frequency == 0:
            if self.config.soft_update:
                self._soft_update_target()
            else:
                self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update exploration and importance sampling
        self._update_epsilon()
        self._update_beta()
        
        self.training_step += 1
        
        # Store metrics
        self.losses.append(loss.item())
        self.q_values.append(current_q_values.mean().item())
        
        return {
            'loss': loss.item(),
            'avg_q_value': current_q_values.mean().item(),
            'epsilon': self.epsilon,
            'beta': self.beta if self.config.prioritized_replay else 0.0
        }
    
    def _soft_update_target(self):
        """Soft update target network."""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
    
    def _update_epsilon(self):
        """Update exploration rate."""
        if self.training_step < self.config.epsilon_decay:
            self.epsilon = (
                self.config.epsilon_start - 
                (self.config.epsilon_start - self.config.epsilon_end) * 
                (self.training_step / self.config.epsilon_decay)
            )
        else:
            self.epsilon = self.config.epsilon_end
    
    def _update_beta(self):
        """Update importance sampling weight."""
        if self.config.prioritized_replay and self.training_step < self.config.beta_decay_steps:
            self.beta = (
                self.config.beta_start + 
                (self.config.beta_end - self.config.beta_start) * 
                (self.training_step / self.config.beta_decay_steps)
            )
        elif self.config.prioritized_replay:
            self.beta = self.config.beta_end
    
    def _shape_rewards(self, rewards: torch.Tensor, states: torch.Tensor,
                      actions: torch.Tensor) -> torch.Tensor:
        """Apply battery-specific reward shaping."""
        # Efficiency bonus for optimal charging rates
        efficiency_bonus = self._compute_efficiency_bonus(states, actions)
        
        # Safety penalty for dangerous actions
        safety_penalty = self._compute_safety_penalty(states, actions)
        
        # Combine rewards
        shaped_rewards = (
            rewards + 
            self.config.efficiency_reward_weight * efficiency_bonus -
            self.config.safety_constraint_weight * safety_penalty
        )
        
        return shaped_rewards
    
    def _compute_efficiency_bonus(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute efficiency bonus based on charging behavior."""
        # Placeholder implementation - would be customized for specific battery metrics
        return torch.zeros(states.size(0), device=self.device)
    
    def _compute_safety_penalty(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute safety penalty for dangerous actions."""
        # Placeholder implementation - would check for safety violations
        return torch.zeros(states.size(0), device=self.device)
    
    def save(self, filepath: str):
        """Save the DQN agent."""
        save_dict = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'beta': self.beta
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"DQN agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the DQN agent."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.training_step = checkpoint.get('training_step', 0)
        self.epsilon = checkpoint.get('epsilon', self.config.epsilon_start)
        self.beta = checkpoint.get('beta', self.config.beta_start)
        
        logger.info(f"DQN agent loaded from {filepath}")
    
    def get_training_metrics(self) -> Dict[str, float]:
        """Get training metrics for monitoring."""
        if not self.losses:
            return {}
        
        return {
            'avg_loss': np.mean(self.losses[-100:]),
            'avg_q_value': np.mean(self.q_values[-100:]),
            'epsilon': self.epsilon,
            'beta': self.beta,
            'training_steps': self.training_step,
            'replay_buffer_size': len(self.replay_buffer)
        }

# Factory function
def create_dqn_agent(state_dim: int, action_dim: int, 
                    config: Optional[DQNConfig] = None) -> DeepQNetwork:
    """
    Factory function to create a DQN agent.
    
    Args:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        config (DQNConfig, optional): DQN configuration
        
    Returns:
        DeepQNetwork: Configured DQN agent
    """
    if config is None:
        config = DQNConfig()
    
    return DeepQNetwork(state_dim, action_dim, config)
