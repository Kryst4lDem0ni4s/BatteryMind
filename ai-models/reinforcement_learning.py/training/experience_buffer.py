"""
BatteryMind - Experience Buffer for Reinforcement Learning

Advanced experience replay buffer implementation with prioritized sampling,
multi-step returns, and efficient storage for battery management RL training.

Features:
- Prioritized experience replay with importance sampling
- Multi-step returns for improved sample efficiency
- Efficient circular buffer implementation
- Support for different data types and shapes
- Memory-efficient storage with compression
- Thread-safe operations for parallel training

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
from dataclasses import dataclass
import logging
import threading
import pickle
import gzip
from collections import deque
import random
import heapq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Experience(NamedTuple):
    """Single experience tuple for RL training."""
    state: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = {}

@dataclass
class BufferConfig:
    """
    Configuration for experience buffer.
    
    Attributes:
        capacity (int): Maximum buffer capacity
        prioritized (bool): Use prioritized experience replay
        alpha (float): Prioritization exponent
        beta (float): Importance sampling exponent
        beta_increment (float): Beta increment per sampling
        epsilon (float): Small constant for numerical stability
        multi_step (int): Number of steps for multi-step returns
        gamma (float): Discount factor for multi-step returns
        compress_states (bool): Compress states to save memory
        device (str): Device for tensor operations
    """
    capacity: int = 100000
    prioritized: bool = True
    alpha: float = 0.6
    beta: float = 0.4
    beta_increment: float = 0.001
    epsilon: float = 1e-6
    multi_step: int = 1
    gamma: float = 0.99
    compress_states: bool = False
    device: str = "cpu"

class SumTree:
    """
    Sum tree data structure for efficient prioritized sampling.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_ptr = 0
        self.size = 0
    
    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve sample index based on priority."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Get total priority sum."""
        return self.tree[0]
    
    def add(self, priority: float, data: Any) -> None:
        """Add new experience with priority."""
        idx = self.write_ptr + self.capacity - 1
        
        self.data[self.write_ptr] = data
        self.update(idx, priority)
        
        self.write_ptr += 1
        if self.write_ptr >= self.capacity:
            self.write_ptr = 0
        
        if self.size < self.capacity:
            self.size += 1
    
    def update(self, idx: int, priority: float) -> None:
        """Update priority of experience."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Any]:
        """Get experience based on priority value."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class ExperienceBuffer:
    """
    Advanced experience replay buffer with prioritized sampling.
    """
    
    def __init__(self, config: Optional[BufferConfig] = None):
        self.config = config or BufferConfig()
        
        # Initialize storage
        if self.config.prioritized:
            self.tree = SumTree(self.config.capacity)
            self.max_priority = 1.0
        else:
            self.buffer = deque(maxlen=self.config.capacity)
        
        # Multi-step buffer for n-step returns
        if self.config.multi_step > 1:
            self.multi_step_buffer = deque(maxlen=self.config.multi_step)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.total_added = 0
        self.total_sampled = 0
        
        # Beta schedule for importance sampling
        self.beta = self.config.beta
        
        logger.info(f"ExperienceBuffer initialized with capacity {self.config.capacity}")
    
    def add(self, state: np.ndarray, action: Union[int, np.ndarray], 
            reward: float, next_state: np.ndarray, done: bool,
            info: Optional[Dict] = None) -> None:
        """
        Add experience to buffer.
        
        Args:
            state (np.ndarray): Current state
            action (Union[int, np.ndarray]): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Episode termination flag
            info (Dict, optional): Additional information
        """
        with self.lock:
            experience = Experience(
                state=self._compress_state(state) if self.config.compress_states else state,
                action=action,
                reward=reward,
                next_state=self._compress_state(next_state) if self.config.compress_states else next_state,
                done=done,
                info=info or {}
            )
            
            if self.config.multi_step > 1:
                self._add_multi_step(experience)
            else:
                self._add_single_step(experience)
            
            self.total_added += 1
    
    def _add_single_step(self, experience: Experience) -> None:
        """Add single-step experience."""
        if self.config.prioritized:
            # Add with maximum priority for new experiences
            self.tree.add(self.max_priority ** self.config.alpha, experience)
        else:
            self.buffer.append(experience)
    
    def _add_multi_step(self, experience: Experience) -> None:
        """Add experience with multi-step returns."""
        self.multi_step_buffer.append(experience)
        
        if len(self.multi_step_buffer) == self.config.multi_step:
            # Calculate multi-step return
            multi_step_exp = self._calculate_multi_step_return()
            self._add_single_step(multi_step_exp)
    
    def _calculate_multi_step_return(self) -> Experience:
        """Calculate multi-step return from buffer."""
        first_exp = self.multi_step_buffer[0]
        last_exp = self.multi_step_buffer[-1]
        
        # Calculate discounted return
        multi_step_return = 0.0
        gamma_power = 1.0
        
        for exp in self.multi_step_buffer:
            multi_step_return += gamma_power * exp.reward
            gamma_power *= self.config.gamma
            if exp.done:
                break
        
        # Create multi-step experience
        return Experience(
            state=first_exp.state,
            action=first_exp.action,
            reward=multi_step_return,
            next_state=last_exp.next_state,
            done=last_exp.done,
            info=first_exp.info
        )
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], 
                                             Optional[torch.Tensor], 
                                             Optional[np.ndarray]]:
        """
        Sample batch of experiences.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            Tuple containing:
            - Dict[str, torch.Tensor]: Batch of experiences
            - Optional[torch.Tensor]: Importance sampling weights (if prioritized)
            - Optional[np.ndarray]: Tree indices (if prioritized)
        """
        with self.lock:
            if len(self) < batch_size:
                raise ValueError(f"Not enough experiences: {len(self)} < {batch_size}")
            
            if self.config.prioritized:
                return self._sample_prioritized(batch_size)
            else:
                return self._sample_uniform(batch_size)
    
    def _sample_uniform(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], None, None]:
        """Sample uniformly from buffer."""
        experiences = random.sample(list(self.buffer), batch_size)
        batch = self._experiences_to_batch(experiences)
        self.total_sampled += batch_size
        return batch, None, None
    
    def _sample_prioritized(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], 
                                                          torch.Tensor, np.ndarray]:
        """Sample with prioritized experience replay."""
        indices = []
        priorities = []
        experiences = []
        
        # Sample from tree
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, experience = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            experiences.append(experience)
        
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = (len(self) * sampling_probabilities) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Convert to tensors
        batch = self._experiences_to_batch(experiences)
        weights_tensor = torch.FloatTensor(weights).to(self.config.device)
        
        # Update beta
        self.beta = min(1.0, self.beta + self.config.beta_increment)
        
        self.total_sampled += batch_size
        return batch, weights_tensor, np.array(indices)
    
    def _experiences_to_batch(self, experiences: List[Experience]) -> Dict[str, torch.Tensor]:
        """Convert list of experiences to batch tensors."""
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for exp in experiences:
            state = self._decompress_state(exp.state) if self.config.compress_states else exp.state
            next_state = self._decompress_state(exp.next_state) if self.config.compress_states else exp.next_state
            
            states.append(state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(next_state)
            dones.append(exp.done)
        
        # Convert to tensors
        device = self.config.device
        
        batch = {
            'states': torch.FloatTensor(np.array(states)).to(device),
            'actions': torch.LongTensor(actions).to(device) if isinstance(actions[0], int) 
                      else torch.FloatTensor(np.array(actions)).to(device),
            'rewards': torch.FloatTensor(rewards).to(device),
            'next_states': torch.FloatTensor(np.array(next_states)).to(device),
            'dones': torch.BoolTensor(dones).to(device)
        }
        
        return batch
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for prioritized replay.
        
        Args:
            indices (np.ndarray): Tree indices to update
            priorities (np.ndarray): New priority values
        """
        if not self.config.prioritized:
            return
        
        with self.lock:
            for idx, priority in zip(indices, priorities):
                # Ensure priority is positive
                priority = abs(priority) + self.config.epsilon
                self.tree.update(idx, priority ** self.config.alpha)
                self.max_priority = max(self.max_priority, priority)
    
    def _compress_state(self, state: np.ndarray) -> bytes:
        """Compress state to save memory."""
        return gzip.compress(pickle.dumps(state))
    
    def _decompress_state(self, compressed_state: bytes) -> np.ndarray:
        """Decompress state."""
        return pickle.loads(gzip.decompress(compressed_state))
    
    def __len__(self) -> int:
        """Get buffer size."""
        if self.config.prioritized:
            return self.tree.size
        else:
            return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences for training."""
        return len(self) >= min_size
    
    def clear(self) -> None:
        """Clear all experiences from buffer."""
        with self.lock:
            if self.config.prioritized:
                self.tree = SumTree(self.config.capacity)
                self.max_priority = 1.0
            else:
                self.buffer.clear()
            
            if hasattr(self, 'multi_step_buffer'):
                self.multi_step_buffer.clear()
            
            self.total_added = 0
            self.total_sampled = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'size': len(self),
            'capacity': self.config.capacity,
            'total_added': self.total_added,
            'total_sampled': self.total_sampled,
            'utilization': len(self) / self.config.capacity,
            'prioritized': self.config.prioritized,
            'multi_step': self.config.multi_step,
            'beta': self.beta if self.config.prioritized else None
        }
    
    def save(self, filepath: str) -> None:
        """Save buffer to file."""
        with self.lock:
            data = {
                'config': self.config,
                'experiences': list(self.buffer) if not self.config.prioritized else None,
                'tree_data': self.tree.data if self.config.prioritized else None,
                'tree_tree': self.tree.tree if self.config.prioritized else None,
                'statistics': self.get_statistics()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Buffer saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load buffer from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        with self.lock:
            self.config = data['config']
            
            if self.config.prioritized:
                self.tree = SumTree(self.config.capacity)
                if data['tree_data'] is not None:
                    self.tree.data = data['tree_data']
                    self.tree.tree = data['tree_tree']
            else:
                self.buffer = deque(data['experiences'], maxlen=self.config.capacity)
            
            stats = data['statistics']
            self.total_added = stats['total_added']
            self.total_sampled = stats['total_sampled']
            
            logger.info(f"Buffer loaded from {filepath}")

class MultiAgentExperienceBuffer:
    """
    Experience buffer for multi-agent reinforcement learning.
    """
    
    def __init__(self, agent_ids: List[str], config: Optional[BufferConfig] = None):
        self.agent_ids = agent_ids
        self.config = config or BufferConfig()
        
        # Create separate buffer for each agent
        self.agent_buffers = {
            agent_id: ExperienceBuffer(config)
            for agent_id in agent_ids
        }
        
        # Shared buffer for joint experiences
        self.shared_buffer = ExperienceBuffer(config)
        
        logger.info(f"MultiAgentExperienceBuffer initialized for agents: {agent_ids}")
    
    def add_agent_experience(self, agent_id: str, state: np.ndarray, 
                           action: Union[int, np.ndarray], reward: float,
                           next_state: np.ndarray, done: bool,
                           info: Optional[Dict] = None) -> None:
        """Add experience for specific agent."""
        if agent_id not in self.agent_buffers:
            raise ValueError(f"Unknown agent ID: {agent_id}")
        
        self.agent_buffers[agent_id].add(state, action, reward, next_state, done, info)
    
    def add_joint_experience(self, joint_state: np.ndarray, joint_action: np.ndarray,
                           joint_reward: float, joint_next_state: np.ndarray,
                           done: bool, info: Optional[Dict] = None) -> None:
        """Add joint experience for all agents."""
        self.shared_buffer.add(joint_state, joint_action, joint_reward, 
                             joint_next_state, done, info)
    
    def sample_agent_batch(self, agent_id: str, batch_size: int):
        """Sample batch for specific agent."""
        if agent_id not in self.agent_buffers:
            raise ValueError(f"Unknown agent ID: {agent_id}")
        
        return self.agent_buffers[agent_id].sample(batch_size)
    
    def sample_joint_batch(self, batch_size: int):
        """Sample joint batch for all agents."""
        return self.shared_buffer.sample(batch_size)
    
    def is_ready(self, min_size: int) -> Dict[str, bool]:
        """Check readiness for all agent buffers."""
        readiness = {}
        for agent_id in self.agent_ids:
            readiness[agent_id] = self.agent_buffers[agent_id].is_ready(min_size)
        readiness['shared'] = self.shared_buffer.is_ready(min_size)
        return readiness
    
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all buffers."""
        stats = {}
        for agent_id in self.agent_ids:
            stats[agent_id] = self.agent_buffers[agent_id].get_statistics()
        stats['shared'] = self.shared_buffer.get_statistics()
        return stats

# Factory functions
def create_experience_buffer(capacity: int = 100000, prioritized: bool = True,
                           **kwargs) -> ExperienceBuffer:
    """
    Factory function to create experience buffer.
    
    Args:
        capacity (int): Buffer capacity
        prioritized (bool): Use prioritized replay
        **kwargs: Additional configuration parameters
        
    Returns:
        ExperienceBuffer: Configured buffer instance
    """
    config = BufferConfig(capacity=capacity, prioritized=prioritized, **kwargs)
    return ExperienceBuffer(config)

def create_multi_agent_buffer(agent_ids: List[str], capacity: int = 100000,
                            **kwargs) -> MultiAgentExperienceBuffer:
    """
    Factory function to create multi-agent experience buffer.
    
    Args:
        agent_ids (List[str]): List of agent identifiers
        capacity (int): Buffer capacity per agent
        **kwargs: Additional configuration parameters
        
    Returns:
        MultiAgentExperienceBuffer: Configured multi-agent buffer
    """
    config = BufferConfig(capacity=capacity, **kwargs)
    return MultiAgentExperienceBuffer(agent_ids, config)
