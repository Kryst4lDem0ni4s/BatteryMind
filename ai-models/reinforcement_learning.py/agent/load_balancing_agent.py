"""
BatteryMind - Load Balancing Agent

Advanced reinforcement learning agent for intelligent battery load balancing
across multiple battery units, cells, or systems to optimize performance,
longevity, and safety through dynamic load distribution strategies.

Features:
- Multi-battery load balancing with real-time optimization
- Dynamic load redistribution based on battery health and capacity
- Predictive load balancing using battery degradation forecasts
- Safety-constrained load balancing with thermal and electrical limits
- Integration with battery health monitoring and forecasting systems
- Hierarchical load balancing for fleet-level optimization

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import time
from collections import deque
import gym
from gym import spaces

# Scientific computing imports
from scipy.optimize import minimize
from scipy.stats import norm
import cvxpy as cp

# Local imports
from ..environments.battery_env import BatteryEnvironment
from ..rewards.composite_reward import CompositeReward
from ..algorithms.ppo import PPOAlgorithm
from ..algorithms.ddpg import DDPGAlgorithm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LoadBalancingConfig:
    """
    Configuration for load balancing agent.
    
    Attributes:
        # Agent configuration
        agent_type (str): Type of RL algorithm ('PPO', 'DDPG', 'SAC')
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        
        # Load balancing parameters
        num_batteries (int): Number of batteries to balance
        max_load_per_battery (float): Maximum load per battery (A)
        min_load_per_battery (float): Minimum load per battery (A)
        balancing_frequency (float): Load balancing frequency (Hz)
        
        # Optimization objectives
        balance_weight (float): Weight for load balance objective
        efficiency_weight (float): Weight for efficiency objective
        health_weight (float): Weight for battery health objective
        safety_weight (float): Weight for safety objective
        
        # Safety constraints
        max_temperature (float): Maximum allowed temperature (°C)
        max_voltage (float): Maximum allowed voltage (V)
        min_voltage (float): Minimum allowed voltage (V)
        max_current (float): Maximum allowed current (A)
        
        # Learning parameters
        learning_rate (float): Learning rate for RL algorithm
        batch_size (int): Training batch size
        memory_size (int): Experience replay buffer size
        exploration_noise (float): Exploration noise for action selection
        
        # Prediction integration
        use_health_prediction (bool): Use battery health predictions
        use_degradation_forecast (bool): Use degradation forecasting
        prediction_horizon (int): Prediction horizon (time steps)
    """
    # Agent configuration
    agent_type: str = "PPO"
    state_dim: int = 32
    action_dim: int = 8
    
    # Load balancing parameters
    num_batteries: int = 8
    max_load_per_battery: float = 100.0
    min_load_per_battery: float = 0.0
    balancing_frequency: float = 10.0
    
    # Optimization objectives
    balance_weight: float = 0.3
    efficiency_weight: float = 0.25
    health_weight: float = 0.25
    safety_weight: float = 0.2
    
    # Safety constraints
    max_temperature: float = 45.0
    max_voltage: float = 4.2
    min_voltage: float = 2.8
    max_current: float = 50.0
    
    # Learning parameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    memory_size: int = 100000
    exploration_noise: float = 0.1
    
    # Prediction integration
    use_health_prediction: bool = True
    use_degradation_forecast: bool = True
    prediction_horizon: int = 100

class LoadBalancingEnvironment(gym.Env):
    """
    Custom environment for load balancing reinforcement learning.
    """
    
    def __init__(self, config: LoadBalancingConfig):
        super().__init__()
        self.config = config
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(config.num_batteries,),
            dtype=np.float32
        )
        
        # State includes: battery states, loads, temperatures, voltages, health metrics
        state_size = config.num_batteries * 6  # 6 features per battery
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_size,),
            dtype=np.float32
        )
        
        # Initialize battery states
        self.battery_states = self._initialize_battery_states()
        self.total_load_demand = 0.0
        self.time_step = 0
        
        # Performance tracking
        self.load_history = deque(maxlen=1000)
        self.efficiency_history = deque(maxlen=1000)
        self.temperature_history = deque(maxlen=1000)
        
    def _initialize_battery_states(self) -> Dict[int, Dict[str, float]]:
        """Initialize battery states with realistic values."""
        battery_states = {}
        
        for i in range(self.config.num_batteries):
            battery_states[i] = {
                'voltage': np.random.uniform(3.2, 4.0),
                'current': 0.0,
                'temperature': np.random.uniform(20.0, 30.0),
                'soc': np.random.uniform(0.2, 0.9),
                'soh': np.random.uniform(0.8, 1.0),
                'internal_resistance': np.random.uniform(0.05, 0.15),
                'capacity': np.random.uniform(80.0, 100.0),
                'load': 0.0
            }
        
        return battery_states
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.battery_states = self._initialize_battery_states()
        self.total_load_demand = np.random.uniform(50.0, 400.0)  # Total system load
        self.time_step = 0
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step within the environment.
        
        Args:
            action (np.ndarray): Load distribution ratios for each battery
            
        Returns:
            Tuple[np.ndarray, float, bool, Dict]: observation, reward, done, info
        """
        # Normalize action to ensure sum equals 1
        action = action / (np.sum(action) + 1e-8)
        
        # Distribute load based on action
        self._distribute_load(action)
        
        # Update battery states
        self._update_battery_states()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        done = self._check_done()
        
        # Update time step
        self.time_step += 1
        
        # Collect info
        info = self._get_info()
        
        return self._get_observation(), reward, done, info
    
    def _distribute_load(self, action: np.ndarray):
        """Distribute total load among batteries based on action."""
        for i, ratio in enumerate(action):
            load = ratio * self.total_load_demand
            
            # Apply safety constraints
            max_safe_load = self._calculate_max_safe_load(i)
            load = min(load, max_safe_load)
            
            self.battery_states[i]['load'] = load
            self.battery_states[i]['current'] = load / self.battery_states[i]['voltage']
    
    def _calculate_max_safe_load(self, battery_idx: int) -> float:
        """Calculate maximum safe load for a battery."""
        battery = self.battery_states[battery_idx]
        
        # Current limit
        max_current_load = self.config.max_current * battery['voltage']
        
        # Temperature limit (reduce load if temperature is high)
        temp_factor = max(0.1, 1.0 - (battery['temperature'] - 25.0) / 20.0)
        max_temp_load = self.config.max_load_per_battery * temp_factor
        
        # Health limit (reduce load for degraded batteries)
        health_factor = battery['soh']
        max_health_load = self.config.max_load_per_battery * health_factor
        
        return min(max_current_load, max_temp_load, max_health_load)
    
    def _update_battery_states(self):
        """Update battery states based on current loads."""
        for i, battery in self.battery_states.items():
            # Update temperature (simplified thermal model)
            heat_generation = battery['current'] ** 2 * battery['internal_resistance']
            cooling_rate = 0.1 * (battery['temperature'] - 25.0)
            battery['temperature'] += (heat_generation - cooling_rate) * 0.01
            
            # Update voltage (simplified electrical model)
            voltage_drop = battery['current'] * battery['internal_resistance']
            battery['voltage'] = battery['soc'] * 4.2 - voltage_drop
            
            # Update SoC (simplified)
            discharge_rate = battery['current'] / battery['capacity']
            battery['soc'] -= discharge_rate * 0.01  # Assuming 1% time step
            battery['soc'] = max(0.0, min(1.0, battery['soc']))
            
            # Update SoH (very simplified degradation)
            stress_factor = (battery['temperature'] - 25.0) / 20.0 + battery['current'] / 50.0
            degradation = stress_factor * 1e-6
            battery['soh'] -= degradation
            battery['soh'] = max(0.5, battery['soh'])
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward based on load balancing performance."""
        # Load balance reward (prefer even distribution)
        loads = np.array([self.battery_states[i]['load'] for i in range(self.config.num_batteries)])
        load_std = np.std(loads)
        load_balance_reward = -load_std / 100.0  # Negative because we want low std
        
        # Efficiency reward (prefer high efficiency batteries)
        efficiencies = []
        for i in range(self.config.num_batteries):
            battery = self.battery_states[i]
            efficiency = battery['voltage'] * battery['current'] / (battery['voltage'] * battery['current'] + 
                                                                  battery['current'] ** 2 * battery['internal_resistance'])
            efficiencies.append(efficiency)
        
        avg_efficiency = np.mean(efficiencies)
        efficiency_reward = avg_efficiency - 0.9  # Baseline efficiency
        
        # Health reward (prefer preserving battery health)
        health_values = [self.battery_states[i]['soh'] for i in range(self.config.num_batteries)]
        avg_health = np.mean(health_values)
        health_reward = avg_health - 0.9  # Baseline health
        
        # Safety reward (penalize constraint violations)
        safety_penalty = 0.0
        for i in range(self.config.num_batteries):
            battery = self.battery_states[i]
            
            if battery['temperature'] > self.config.max_temperature:
                safety_penalty += (battery['temperature'] - self.config.max_temperature) * 0.1
            
            if battery['voltage'] > self.config.max_voltage or battery['voltage'] < self.config.min_voltage:
                safety_penalty += 0.5
            
            if battery['current'] > self.config.max_current:
                safety_penalty += (battery['current'] - self.config.max_current) * 0.1
        
        safety_reward = -safety_penalty
        
        # Combine rewards
        total_reward = (
            self.config.balance_weight * load_balance_reward +
            self.config.efficiency_weight * efficiency_reward +
            self.config.health_weight * health_reward +
            self.config.safety_weight * safety_reward
        )
        
        return total_reward
    
    def _check_done(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if any safety constraint is severely violated
        for i in range(self.config.num_batteries):
            battery = self.battery_states[i]
            
            if battery['temperature'] > self.config.max_temperature + 10.0:
                return True
            
            if battery['soc'] < 0.05:  # Battery critically low
                return True
        
        # Terminate after maximum episode length
        if self.time_step >= 1000:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        observation = []
        
        for i in range(self.config.num_batteries):
            battery = self.battery_states[i]
            observation.extend([
                battery['voltage'] / 4.2,  # Normalized voltage
                battery['current'] / self.config.max_current,  # Normalized current
                battery['temperature'] / 60.0,  # Normalized temperature
                battery['soc'],  # State of charge
                battery['soh'],  # State of health
                battery['load'] / self.config.max_load_per_battery  # Normalized load
            ])
        
        # Add global information
        total_load_normalized = self.total_load_demand / (self.config.num_batteries * self.config.max_load_per_battery)
        observation.extend([total_load_normalized])
        
        # Pad to expected state dimension
        while len(observation) < self.config.state_dim:
            observation.append(0.0)
        
        return np.array(observation[:self.config.state_dim], dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state."""
        loads = [self.battery_states[i]['load'] for i in range(self.config.num_batteries)]
        temperatures = [self.battery_states[i]['temperature'] for i in range(self.config.num_batteries)]
        health_values = [self.battery_states[i]['soh'] for i in range(self.config.num_batteries)]
        
        return {
            'load_distribution': loads,
            'load_balance_std': np.std(loads),
            'avg_temperature': np.mean(temperatures),
            'max_temperature': np.max(temperatures),
            'avg_health': np.mean(health_values),
            'min_health': np.min(health_values),
            'total_load': self.total_load_demand,
            'time_step': self.time_step
        }

class LoadBalancingAgent:
    """
    Reinforcement learning agent for intelligent battery load balancing.
    """
    
    def __init__(self, config: LoadBalancingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create environment
        self.env = LoadBalancingEnvironment(config)
        
        # Initialize RL algorithm
        self.algorithm = self._create_algorithm()
        
        # Performance tracking
        self.training_history = []
        self.evaluation_metrics = {}
        
        logger.info(f"LoadBalancingAgent initialized with {config.agent_type} algorithm")
    
    def _create_algorithm(self):
        """Create RL algorithm based on configuration."""
        if self.config.agent_type == "PPO":
            return PPOAlgorithm(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                learning_rate=self.config.learning_rate,
                device=self.device
            )
        elif self.config.agent_type == "DDPG":
            return DDPGAlgorithm(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                learning_rate=self.config.learning_rate,
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported agent type: {self.config.agent_type}")
    
    def train(self, num_episodes: int = 1000, eval_frequency: int = 100) -> Dict[str, List[float]]:
        """
        Train the load balancing agent.
        
        Args:
            num_episodes (int): Number of training episodes
            eval_frequency (int): Frequency of evaluation episodes
            
        Returns:
            Dict[str, List[float]]: Training history and metrics
        """
        logger.info(f"Starting training for {num_episodes} episodes")
        
        episode_rewards = []
        episode_lengths = []
        evaluation_scores = []
        
        for episode in range(num_episodes):
            # Training episode
            state = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            while True:
                # Select action
                action = self.algorithm.select_action(state, training=True)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.algorithm.store_experience(state, action, reward, next_state, done)
                
                # Update metrics
                episode_reward += reward
                episode_length += 1
                
                # Move to next state
                state = next_state
                
                if done:
                    break
            
            # Train algorithm
            if len(self.algorithm.memory) > self.config.batch_size:
                self.algorithm.train()
            
            # Record metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Evaluation
            if episode % eval_frequency == 0:
                eval_score = self.evaluate(num_episodes=5)
                evaluation_scores.append(eval_score)
                
                logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, "
                           f"Length={episode_length}, Eval Score={eval_score:.2f}")
        
        # Store training history
        self.training_history = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'evaluation_scores': evaluation_scores
        }
        
        logger.info("Training completed")
        return self.training_history
    
    def evaluate(self, num_episodes: int = 10) -> float:
        """
        Evaluate the agent's performance.
        
        Args:
            num_episodes (int): Number of evaluation episodes
            
        Returns:
            float: Average evaluation score
        """
        total_reward = 0.0
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            
            while True:
                action = self.algorithm.select_action(state, training=False)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def predict_load_distribution(self, battery_states: Dict[int, Dict[str, float]], 
                                 total_load: float) -> np.ndarray:
        """
        Predict optimal load distribution for given battery states and total load.
        
        Args:
            battery_states (Dict[int, Dict[str, float]]): Current battery states
            total_load (float): Total load to distribute
            
        Returns:
            np.ndarray: Optimal load distribution
        """
        # Create observation from battery states
        observation = []
        for i in range(self.config.num_batteries):
            if i in battery_states:
                battery = battery_states[i]
                observation.extend([
                    battery.get('voltage', 3.7) / 4.2,
                    battery.get('current', 0.0) / self.config.max_current,
                    battery.get('temperature', 25.0) / 60.0,
                    battery.get('soc', 0.5),
                    battery.get('soh', 1.0),
                    battery.get('load', 0.0) / self.config.max_load_per_battery
                ])
            else:
                observation.extend([0.5, 0.0, 0.4, 0.5, 1.0, 0.0])  # Default values
        
        # Add total load information
        total_load_normalized = total_load / (self.config.num_batteries * self.config.max_load_per_battery)
        observation.append(total_load_normalized)
        
        # Pad to expected dimension
        while len(observation) < self.config.state_dim:
            observation.append(0.0)
        
        observation = np.array(observation[:self.config.state_dim], dtype=np.float32)
        
        # Get action from trained agent
        action = self.algorithm.select_action(observation, training=False)
        
        # Convert action to actual load distribution
        action = action / (np.sum(action) + 1e-8)  # Normalize
        load_distribution = action * total_load
        
        return load_distribution
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self.training_history:
            return {}
        
        episode_rewards = self.training_history['episode_rewards']
        
        return {
            'avg_reward': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'reward_std': np.std(episode_rewards),
            'total_episodes': len(episode_rewards),
            'convergence_episode': self._find_convergence_episode(),
            'final_evaluation_score': self.training_history['evaluation_scores'][-1] if self.training_history['evaluation_scores'] else None
        }
    
    def _find_convergence_episode(self) -> Optional[int]:
        """Find episode where training converged."""
        if len(self.training_history['episode_rewards']) < 100:
            return None
        
        rewards = self.training_history['episode_rewards']
        window_size = 50
        
        for i in range(window_size, len(rewards) - window_size):
            before_window = rewards[i-window_size:i]
            after_window = rewards[i:i+window_size]
            
            # Check if improvement has plateaued
            before_avg = np.mean(before_window)
            after_avg = np.mean(after_window)
            
            if abs(after_avg - before_avg) < 0.01:  # Convergence threshold
                return i
        
        return None
    
    def save_agent(self, filepath: str):
        """Save trained agent."""
        save_data = {
            'config': self.config,
            'algorithm_state': self.algorithm.get_state(),
            'training_history': self.training_history,
            'performance_metrics': self.get_performance_metrics()
        }
        
        torch.save(save_data, filepath)
        logger.info(f"Agent saved to {filepath}")
    
    def load_agent(self, filepath: str):
        """Load trained agent."""
        save_data = torch.load(filepath, map_location=self.device)
        
        self.config = save_data['config']
        self.algorithm.load_state(save_data['algorithm_state'])
        self.training_history = save_data.get('training_history', [])
        
        logger.info(f"Agent loaded from {filepath}")

# Factory function
def create_load_balancing_agent(config: Optional[LoadBalancingConfig] = None) -> LoadBalancingAgent:
    """
    Factory function to create a load balancing agent.
    
    Args:
        config (LoadBalancingConfig, optional): Agent configuration
        
    Returns:
        LoadBalancingAgent: Configured load balancing agent
    """
    if config is None:
        config = LoadBalancingConfig()
    
    return LoadBalancingAgent(config)
