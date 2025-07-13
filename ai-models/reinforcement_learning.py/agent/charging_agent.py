"""
BatteryMind - Charging Agent

Advanced reinforcement learning agent for optimizing battery charging protocols
to maximize battery lifespan, efficiency, and safety while meeting user requirements.

Features:
- Multi-objective optimization (lifespan, efficiency, safety, user satisfaction)
- Physics-informed reward functions with battery degradation models
- Adaptive charging strategies based on battery state and environmental conditions
- Safety constraints and emergency protocols
- Integration with battery health prediction and degradation forecasting
- Real-time learning and adaptation capabilities

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
import json
from abc import ABC, abstractmethod
from collections import deque
import random

# RL-specific imports
import gym
from gym import spaces
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

# Scientific computing imports
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChargingAgentConfig:
    """
    Configuration for the charging optimization agent.
    
    Attributes:
        # Agent parameters
        algorithm (str): RL algorithm to use ('PPO', 'SAC', 'TD3')
        learning_rate (float): Learning rate for the agent
        batch_size (int): Batch size for training
        buffer_size (int): Size of experience replay buffer
        
        # Environment parameters
        max_episode_steps (int): Maximum steps per episode
        observation_space_dim (int): Dimension of observation space
        action_space_dim (int): Dimension of action space
        
        # Charging parameters
        max_charging_rate (float): Maximum charging rate (C-rate)
        min_charging_rate (float): Minimum charging rate (C-rate)
        voltage_range (Tuple[float, float]): Voltage range (V)
        temperature_range (Tuple[float, float]): Temperature range (°C)
        
        # Reward weights
        lifespan_weight (float): Weight for battery lifespan objective
        efficiency_weight (float): Weight for charging efficiency objective
        safety_weight (float): Weight for safety objective
        time_weight (float): Weight for charging time objective
        
        # Safety constraints
        max_temperature (float): Maximum allowed temperature (°C)
        min_voltage (float): Minimum voltage threshold (V)
        max_voltage (float): Maximum voltage threshold (V)
        emergency_stop_conditions (Dict): Emergency stop conditions
        
        # Learning parameters
        exploration_noise (float): Exploration noise for action selection
        target_update_freq (int): Frequency of target network updates
        gradient_steps (int): Number of gradient steps per update
    """
    # Agent parameters
    algorithm: str = "PPO"
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100000
    
    # Environment parameters
    max_episode_steps: int = 1000
    observation_space_dim: int = 15
    action_space_dim: int = 3
    
    # Charging parameters
    max_charging_rate: float = 2.0  # C-rate
    min_charging_rate: float = 0.1  # C-rate
    voltage_range: Tuple[float, float] = (3.0, 4.2)
    temperature_range: Tuple[float, float] = (0.0, 45.0)
    
    # Reward weights
    lifespan_weight: float = 0.4
    efficiency_weight: float = 0.3
    safety_weight: float = 0.2
    time_weight: float = 0.1
    
    # Safety constraints
    max_temperature: float = 50.0
    min_voltage: float = 2.8
    max_voltage: float = 4.3
    emergency_stop_conditions: Dict = field(default_factory=lambda: {
        'temperature_critical': 55.0,
        'voltage_critical_low': 2.5,
        'voltage_critical_high': 4.4,
        'current_critical': 5.0  # C-rate
    })
    
    # Learning parameters
    exploration_noise: float = 0.1
    target_update_freq: int = 100
    gradient_steps: int = 1

class BatteryState:
    """
    Comprehensive battery state representation for the charging agent.
    """
    
    def __init__(self):
        # Core battery parameters
        self.voltage = 3.7  # V
        self.current = 0.0  # A
        self.temperature = 25.0  # °C
        self.state_of_charge = 0.5  # 0-1
        self.state_of_health = 1.0  # 0-1
        self.internal_resistance = 0.1  # Ω
        self.capacity = 100.0  # Ah
        
        # Derived parameters
        self.power = 0.0  # W
        self.energy = 0.0  # Wh
        self.cycle_count = 0
        self.age_days = 0
        
        # Environmental conditions
        self.ambient_temperature = 25.0  # °C
        self.humidity = 50.0  # %
        self.pressure = 1013.25  # hPa
        
        # Charging history
        self.charging_history = deque(maxlen=100)
        self.degradation_history = deque(maxlen=1000)
        
        # Safety flags
        self.safety_violations = []
        self.emergency_stop = False
    
    def update(self, voltage: float, current: float, temperature: float):
        """Update battery state with new measurements."""
        self.voltage = voltage
        self.current = current
        self.temperature = temperature
        self.power = voltage * current
        
        # Update derived parameters
        self._update_soc()
        self._update_soh()
        self._check_safety()
    
    def _update_soc(self):
        """Update State of Charge based on current integration."""
        # Simplified SoC calculation (in practice, use more sophisticated methods)
        if self.current > 0:  # Charging
            delta_soc = (self.current * 1/3600) / self.capacity  # Assuming 1-second timestep
            self.state_of_charge = min(1.0, self.state_of_charge + delta_soc)
        elif self.current < 0:  # Discharging
            delta_soc = abs(self.current * 1/3600) / self.capacity
            self.state_of_charge = max(0.0, self.state_of_charge - delta_soc)
    
    def _update_soh(self):
        """Update State of Health based on degradation models."""
        # Simplified SoH degradation (temperature and cycle-based)
        temp_degradation = max(0, (self.temperature - 25) / 100) * 0.0001
        cycle_degradation = self.cycle_count * 0.00001
        
        self.state_of_health = max(0.5, 1.0 - temp_degradation - cycle_degradation)
    
    def _check_safety(self):
        """Check safety conditions and update flags."""
        self.safety_violations = []
        
        if self.temperature > 50.0:
            self.safety_violations.append("high_temperature")
        if self.voltage < 2.8 or self.voltage > 4.3:
            self.safety_violations.append("voltage_out_of_range")
        if abs(self.current) > self.capacity * 3.0:  # 3C limit
            self.safety_violations.append("high_current")
        
        # Emergency stop conditions
        if (self.temperature > 55.0 or 
            self.voltage < 2.5 or self.voltage > 4.4 or
            abs(self.current) > self.capacity * 5.0):
            self.emergency_stop = True
    
    def to_observation(self) -> np.ndarray:
        """Convert battery state to observation vector for RL agent."""
        observation = np.array([
            self.voltage,
            self.current,
            self.temperature,
            self.state_of_charge,
            self.state_of_health,
            self.internal_resistance,
            self.capacity / 100.0,  # Normalized
            self.power / 1000.0,    # Normalized
            self.ambient_temperature / 50.0,  # Normalized
            self.humidity / 100.0,
            len(self.safety_violations) / 5.0,  # Normalized
            float(self.emergency_stop),
            # Time-based features
            np.sin(2 * np.pi * (time.time() % 86400) / 86400),  # Time of day
            np.cos(2 * np.pi * (time.time() % 86400) / 86400),
            self.cycle_count / 1000.0  # Normalized
        ], dtype=np.float32)
        
        return observation

class ChargingEnvironment(gym.Env):
    """
    Gym environment for battery charging optimization.
    """
    
    def __init__(self, config: ChargingAgentConfig):
        super().__init__()
        self.config = config
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([config.min_charging_rate, config.voltage_range[0], -1.0]),
            high=np.array([config.max_charging_rate, config.voltage_range[1], 1.0]),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config.observation_space_dim,),
            dtype=np.float32
        )
        
        # Initialize battery state
        self.battery_state = BatteryState()
        self.initial_soh = self.battery_state.state_of_health
        
        # Episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0
        self.charging_start_time = time.time()
        
        # Reward components tracking
        self.reward_components = {
            'lifespan': 0.0,
            'efficiency': 0.0,
            'safety': 0.0,
            'time': 0.0
        }
        
        # Physics simulator for battery dynamics
        self.physics_simulator = BatteryPhysicsSimulator()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action (np.ndarray): Action [charging_rate, target_voltage, thermal_control]
            
        Returns:
            Tuple: (observation, reward, done, info)
        """
        self.episode_step += 1
        
        # Parse action
        charging_rate = np.clip(action[0], self.config.min_charging_rate, self.config.max_charging_rate)
        target_voltage = np.clip(action[1], self.config.voltage_range[0], self.config.voltage_range[1])
        thermal_control = np.clip(action[2], -1.0, 1.0)
        
        # Apply action through physics simulator
        new_state = self.physics_simulator.simulate_step(
            self.battery_state, charging_rate, target_voltage, thermal_control
        )
        
        # Update battery state
        self.battery_state = new_state
        
        # Calculate reward
        reward = self._calculate_reward(action)
        self.episode_reward += reward
        
        # Check termination conditions
        done = self._check_done()
        
        # Get observation
        observation = self.battery_state.to_observation()
        
        # Prepare info dictionary
        info = {
            'episode_step': self.episode_step,
            'battery_state': {
                'soc': self.battery_state.state_of_charge,
                'soh': self.battery_state.state_of_health,
                'temperature': self.battery_state.temperature,
                'voltage': self.battery_state.voltage
            },
            'reward_components': self.reward_components.copy(),
            'safety_violations': self.battery_state.safety_violations.copy(),
            'emergency_stop': self.battery_state.emergency_stop
        }
        
        return observation, reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        # Reset battery state with some randomization
        self.battery_state = BatteryState()
        self.battery_state.state_of_charge = np.random.uniform(0.1, 0.9)
        self.battery_state.temperature = np.random.uniform(15.0, 35.0)
        self.battery_state.state_of_health = np.random.uniform(0.8, 1.0)
        
        # Reset episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0
        self.charging_start_time = time.time()
        self.initial_soh = self.battery_state.state_of_health
        
        # Reset reward components
        self.reward_components = {
            'lifespan': 0.0,
            'efficiency': 0.0,
            'safety': 0.0,
            'time': 0.0
        }
        
        return self.battery_state.to_observation()
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate multi-objective reward function."""
        # Lifespan reward (minimize degradation)
        soh_change = self.battery_state.state_of_health - self.initial_soh
        lifespan_reward = -abs(soh_change) * 1000  # Penalize SoH loss
        
        # Efficiency reward
        charging_efficiency = self._calculate_charging_efficiency()
        efficiency_reward = charging_efficiency * 10
        
        # Safety reward
        safety_reward = self._calculate_safety_reward()
        
        # Time reward (encourage faster charging when safe)
        time_reward = self._calculate_time_reward()
        
        # Combine rewards with weights
        total_reward = (
            self.config.lifespan_weight * lifespan_reward +
            self.config.efficiency_weight * efficiency_reward +
            self.config.safety_weight * safety_reward +
            self.config.time_weight * time_reward
        )
        
        # Update reward components for tracking
        self.reward_components['lifespan'] = lifespan_reward
        self.reward_components['efficiency'] = efficiency_reward
        self.reward_components['safety'] = safety_reward
        self.reward_components['time'] = time_reward
        
        return total_reward
    
    def _calculate_charging_efficiency(self) -> float:
        """Calculate charging efficiency."""
        if self.battery_state.power <= 0:
            return 0.0
        
        # Simplified efficiency calculation
        voltage_efficiency = 1.0 - abs(self.battery_state.voltage - 3.7) / 3.7 * 0.1
        temperature_efficiency = 1.0 - abs(self.battery_state.temperature - 25) / 25 * 0.1
        current_efficiency = 1.0 - abs(self.battery_state.current) / (self.battery_state.capacity * 2) * 0.1
        
        return voltage_efficiency * temperature_efficiency * current_efficiency
    
    def _calculate_safety_reward(self) -> float:
        """Calculate safety-based reward."""
        safety_reward = 10.0  # Base safety reward
        
        # Penalize safety violations
        for violation in self.battery_state.safety_violations:
            if violation == "high_temperature":
                safety_reward -= 5.0
            elif violation == "voltage_out_of_range":
                safety_reward -= 3.0
            elif violation == "high_current":
                safety_reward -= 2.0
        
        # Severe penalty for emergency stop
        if self.battery_state.emergency_stop:
            safety_reward -= 50.0
        
        return safety_reward
    
    def _calculate_time_reward(self) -> float:
        """Calculate time-based reward."""
        # Reward faster charging when safe and efficient
        if (self.battery_state.state_of_charge < 0.8 and 
            len(self.battery_state.safety_violations) == 0):
            charging_rate = self.battery_state.current / self.battery_state.capacity
            return charging_rate * 2.0
        else:
            return 0.0
    
    def _check_done(self) -> bool:
        """Check if episode should terminate."""
        # Emergency stop
        if self.battery_state.emergency_stop:
            return True
        
        # Fully charged
        if self.battery_state.state_of_charge >= 0.95:
            return True
        
        # Maximum episode steps
        if self.episode_step >= self.config.max_episode_steps:
            return True
        
        # Severe degradation
        if self.battery_state.state_of_health < 0.7:
            return True
        
        return False

class BatteryPhysicsSimulator:
    """
    Physics-based battery simulator for realistic dynamics.
    """
    
    def __init__(self):
        self.dt = 1.0  # Time step in seconds
        
    def simulate_step(self, current_state: BatteryState, charging_rate: float,
                     target_voltage: float, thermal_control: float) -> BatteryState:
        """
        Simulate one time step of battery dynamics.
        
        Args:
            current_state (BatteryState): Current battery state
            charging_rate (float): Charging rate (C-rate)
            target_voltage (float): Target voltage
            thermal_control (float): Thermal control input (-1 to 1)
            
        Returns:
            BatteryState: Updated battery state
        """
        new_state = BatteryState()
        
        # Copy current state
        new_state.__dict__.update(current_state.__dict__)
        
        # Calculate charging current
        charging_current = charging_rate * current_state.capacity
        
        # Voltage dynamics (simplified)
        voltage_error = target_voltage - current_state.voltage
        new_state.voltage = current_state.voltage + voltage_error * 0.1 * self.dt
        new_state.voltage = np.clip(new_state.voltage, 2.5, 4.4)
        
        # Current dynamics
        new_state.current = charging_current
        
        # Temperature dynamics
        heat_generation = self._calculate_heat_generation(new_state)
        heat_dissipation = self._calculate_heat_dissipation(new_state, thermal_control)
        
        temp_change = (heat_generation - heat_dissipation) * self.dt / 1000  # Simplified thermal mass
        new_state.temperature = current_state.temperature + temp_change
        
        # Update other parameters
        new_state.update(new_state.voltage, new_state.current, new_state.temperature)
        
        # Update degradation
        self._update_degradation(new_state, current_state)
        
        return new_state
    
    def _calculate_heat_generation(self, state: BatteryState) -> float:
        """Calculate heat generation from battery operation."""
        # Joule heating
        joule_heat = state.current**2 * state.internal_resistance
        
        # Reaction heat (simplified)
        reaction_heat = abs(state.current) * 0.1
        
        return joule_heat + reaction_heat
    
    def _calculate_heat_dissipation(self, state: BatteryState, thermal_control: float) -> float:
        """Calculate heat dissipation."""
        # Natural convection
        temp_diff = state.temperature - state.ambient_temperature
        natural_dissipation = temp_diff * 0.5
        
        # Active thermal management
        active_dissipation = thermal_control * 2.0 if thermal_control > 0 else 0
        
        return natural_dissipation + active_dissipation
    
    def _update_degradation(self, new_state: BatteryState, old_state: BatteryState):
        """Update battery degradation based on stress factors."""
        # Temperature-based degradation
        temp_stress = max(0, (new_state.temperature - 25) / 25)
        temp_degradation = temp_stress * 0.00001 * self.dt
        
        # Current-based degradation
        current_stress = abs(new_state.current) / new_state.capacity
        current_degradation = current_stress * 0.00001 * self.dt
        
        # Voltage stress degradation
        voltage_stress = max(0, (new_state.voltage - 4.0) / 0.2)
        voltage_degradation = voltage_stress * 0.00001 * self.dt
        
        # Apply degradation
        total_degradation = temp_degradation + current_degradation + voltage_degradation
        new_state.state_of_health = max(0.5, old_state.state_of_health - total_degradation)

class ChargingAgent:
    """
    Main charging optimization agent using reinforcement learning.
    """
    
    def __init__(self, config: ChargingAgentConfig):
        self.config = config
        self.env = ChargingEnvironment(config)
        
        # Create RL model
        self.model = self._create_model()
        
        # Training metrics
        self.training_metrics = {
            'episodes': 0,
            'total_steps': 0,
            'average_reward': 0.0,
            'best_reward': -np.inf,
            'safety_violations': 0,
            'emergency_stops': 0
        }
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        logger.info(f"Charging agent initialized with {config.algorithm} algorithm")
    
    def _create_model(self):
        """Create the RL model based on configuration."""
        if self.config.algorithm == "PPO":
            return PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                verbose=1,
                device="auto"
            )
        elif self.config.algorithm == "SAC":
            return SAC(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                buffer_size=self.config.buffer_size,
                verbose=1,
                device="auto"
            )
        elif self.config.algorithm == "TD3":
            # Add action noise for TD3
            action_noise = NormalActionNoise(
                mean=np.zeros(self.config.action_space_dim),
                sigma=self.config.exploration_noise * np.ones(self.config.action_space_dim)
            )
            
            return TD3(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                buffer_size=self.config.buffer_size,
                action_noise=action_noise,
                verbose=1,
                device="auto"
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
    
    def train(self, total_timesteps: int, callback=None) -> Dict[str, Any]:
        """
        Train the charging agent.
        
        Args:
            total_timesteps (int): Total training timesteps
            callback: Optional training callback
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info(f"Starting training for {total_timesteps} timesteps")
        start_time = time.time()
        
        # Create training callback
        training_callback = ChargingTrainingCallback(self)
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[training_callback, callback] if callback else training_callback
        )
        
        training_time = time.time() - start_time
        
        # Update training metrics
        self.training_metrics['total_steps'] += total_timesteps
        self.training_metrics['training_time'] = training_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'training_metrics': self.training_metrics,
            'final_model': self.model,
            'training_time': training_time
        }
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Predict optimal charging action.
        
        Args:
            observation (np.ndarray): Current observation
            deterministic (bool): Whether to use deterministic policy
            
        Returns:
            Tuple[np.ndarray, Dict]: Action and additional info
        """
        action, _states = self.model.predict(observation, deterministic=deterministic)
        
        # Add safety checks
        action = self._apply_safety_constraints(action, observation)
        
        # Prepare action info
        action_info = {
            'charging_rate': action[0],
            'target_voltage': action[1],
            'thermal_control': action[2],
            'safety_constrained': np.any(action != self.model.predict(observation, deterministic=deterministic)[0])
        }
        
        return action, action_info
    
    def _apply_safety_constraints(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """Apply safety constraints to actions."""
        constrained_action = action.copy()
        
        # Extract relevant state information
        voltage = observation[0]
        temperature = observation[2]
        soc = observation[3]
        emergency_stop = observation[11] > 0.5
        
        # Emergency stop override
        if emergency_stop:
            constrained_action[0] = 0.0  # Stop charging
            return constrained_action
        
        # Temperature-based constraints
        if temperature > 45.0:
            constrained_action[0] = min(constrained_action[0], 0.5)  # Reduce charging rate
        
        # Voltage-based constraints
        if voltage > 4.1:
            constrained_action[0] = min(constrained_action[0], 0.3)  # Reduce charging rate
        
        # SoC-based constraints
        if soc > 0.9:
            constrained_action[0] = min(constrained_action[0], 0.2)  # Trickle charge
        
        return constrained_action
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate the trained agent.
        
        Args:
            n_episodes (int): Number of evaluation episodes
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        logger.info(f"Evaluating agent for {n_episodes} episodes")
        
        episode_rewards = []
        episode_lengths = []
        safety_violations = 0
        emergency_stops = 0
        
        for episode in range(n_episodes):
            obs = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Track safety metrics
                if info['safety_violations']:
                    safety_violations += len(info['safety_violations'])
                if info['emergency_stop']:
                    emergency_stops += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        evaluation_results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'safety_violations_per_episode': safety_violations / n_episodes,
            'emergency_stop_rate': emergency_stops / n_episodes,
            'episode_rewards': episode_rewards
        }
        
        logger.info(f"Evaluation completed. Mean reward: {evaluation_results['mean_reward']:.2f}")
        
        return evaluation_results
    
    def save(self, path: str):
        """Save the trained model."""
        self.model.save(path)
        
        # Save additional metadata
        metadata = {
            'config': self.config.__dict__,
            'training_metrics': self.training_metrics,
            'model_type': self.config.algorithm
        }
        
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, config: Optional[ChargingAgentConfig] = None):
        """Load a trained model."""
        # Load metadata
        try:
            with open(f"{path}_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            if config is None:
                config = ChargingAgentConfig(**metadata['config'])
        except FileNotFoundError:
            if config is None:
                config = ChargingAgentConfig()
            logger.warning("Metadata file not found, using provided or default config")
        
        # Create agent
        agent = cls(config)
        
        # Load model
        if config.algorithm == "PPO":
            agent.model = PPO.load(path, env=agent.env)
        elif config.algorithm == "SAC":
            agent.model = SAC.load(path, env=agent.env)
        elif config.algorithm == "TD3":
            agent.model = TD3.load(path, env=agent.env)
        
        logger.info(f"Model loaded from {path}")
        return agent

class ChargingTrainingCallback(BaseCallback):
    """
    Custom callback for monitoring charging agent training.
    """
    
    def __init__(self, agent: ChargingAgent, verbose: int = 1):
        super().__init__(verbose)
        self.agent = agent
        
    def _on_step(self) -> bool:
        """Called at each training step."""
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        # Update training metrics
        if hasattr(self.locals, 'infos'):
            infos = self.locals['infos']
            for info in infos:
                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    
                    self.agent.episode_rewards.append(episode_reward)
                    self.agent.episode_lengths.append(episode_length)
                    
                    # Update best reward
                    if episode_reward > self.agent.training_metrics['best_reward']:
                        self.agent.training_metrics['best_reward'] = episode_reward
                    
                    # Update average reward
                    self.agent.training_metrics['average_reward'] = np.mean(self.agent.episode_rewards)
                    self.agent.training_metrics['episodes'] += 1

# Factory functions
def create_charging_agent(config: Optional[ChargingAgentConfig] = None) -> ChargingAgent:
    """
    Factory function to create a charging agent.
    
    Args:
        config (ChargingAgentConfig, optional): Agent configuration
        
    Returns:
        ChargingAgent: Configured charging agent
    """
    if config is None:
        config = ChargingAgentConfig()
    
    return ChargingAgent(config)

def create_charging_environment(config: Optional[ChargingAgentConfig] = None) -> ChargingEnvironment:
    """
    Factory function to create a charging environment.
    
    Args:
        config (ChargingAgentConfig, optional): Environment configuration
        
    Returns:
        ChargingEnvironment: Configured charging environment
    """
    if config is None:
        config = ChargingAgentConfig()
    
    return ChargingEnvironment(config)
