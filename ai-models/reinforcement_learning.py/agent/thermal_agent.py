"""
BatteryMind - Thermal Agent

Advanced reinforcement learning agent for optimizing battery thermal management
to maintain optimal operating temperatures while maximizing efficiency and safety.

Features:
- Multi-zone thermal control with predictive capabilities
- Physics-informed thermal dynamics modeling
- Safety-critical temperature management with emergency protocols
- Energy-efficient cooling and heating strategies
- Integration with charging optimization and health prediction
- Adaptive control based on environmental conditions and usage patterns

Author: BatteryMind Development Team
Version: 1.0.0
"""

import os
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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

# Scientific computing imports
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ThermalAgentConfig:
    """
    Configuration for the thermal management agent.
    
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
        
        # Thermal parameters
        target_temperature (float): Target operating temperature (°C)
        temperature_tolerance (float): Acceptable temperature deviation (°C)
        max_cooling_power (float): Maximum cooling power (W)
        max_heating_power (float): Maximum heating power (W)
        thermal_zones (int): Number of thermal zones to control
        
        # Safety constraints
        critical_temp_high (float): Critical high temperature (°C)
        critical_temp_low (float): Critical low temperature (°C)
        emergency_cooling_threshold (float): Emergency cooling threshold (°C)
        thermal_runaway_threshold (float): Thermal runaway threshold (°C)
        
        # Reward weights
        temperature_control_weight (float): Weight for temperature control objective
        energy_efficiency_weight (float): Weight for energy efficiency objective
        safety_weight (float): Weight for safety objective
        stability_weight (float): Weight for temperature stability objective
        
        # Physical parameters
        thermal_mass (float): Thermal mass of battery (J/K)
        ambient_temperature (float): Ambient temperature (°C)
        convection_coefficient (float): Heat transfer coefficient (W/m²K)
        surface_area (float): Battery surface area (m²)
        
        # Learning parameters
        exploration_noise (float): Exploration noise for action selection
        prediction_horizon (int): Prediction horizon for thermal dynamics
    """
    # Agent parameters
    algorithm: str = "PPO"
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100000
    
    # Environment parameters
    max_episode_steps: int = 1000
    observation_space_dim: int = 20
    action_space_dim: int = 4
    
    # Thermal parameters
    target_temperature: float = 25.0  # °C
    temperature_tolerance: float = 5.0  # °C
    max_cooling_power: float = 100.0  # W
    max_heating_power: float = 50.0   # W
    thermal_zones: int = 4
    
    # Safety constraints
    critical_temp_high: float = 55.0  # °C
    critical_temp_low: float = -10.0  # °C
    emergency_cooling_threshold: float = 50.0  # °C
    thermal_runaway_threshold: float = 60.0  # °C
    
    # Reward weights
    temperature_control_weight: float = 0.4
    energy_efficiency_weight: float = 0.3
    safety_weight: float = 0.2
    stability_weight: float = 0.1
    
    # Physical parameters
    thermal_mass: float = 1000.0  # J/K
    ambient_temperature: float = 25.0  # °C
    convection_coefficient: float = 10.0  # W/m²K
    surface_area: float = 0.1  # m²
    
    # Learning parameters
    exploration_noise: float = 0.1
    prediction_horizon: int = 10

class ThermalState:
    """
    Comprehensive thermal state representation for the thermal agent.
    """
    
    def __init__(self, n_zones: int = 4):
        self.n_zones = n_zones
        
        # Core thermal parameters
        self.temperatures = np.full(n_zones, 25.0)  # °C for each zone
        self.temperature_gradients = np.zeros(n_zones)  # °C/s
        self.heat_generation_rates = np.zeros(n_zones)  # W
        self.cooling_powers = np.zeros(n_zones)  # W
        self.heating_powers = np.zeros(n_zones)  # W
        
        # Environmental conditions
        self.ambient_temperature = 25.0  # °C
        self.ambient_humidity = 50.0  # %
        self.air_flow_rate = 0.0  # m/s
        self.external_heat_sources = 0.0  # W
        
        # Battery operation state
        self.charging_current = 0.0  # A
        self.internal_resistance = 0.1  # Ω
        self.state_of_charge = 0.5  # 0-1
        self.power_dissipation = 0.0  # W
        
        # Thermal management system state
        self.fan_speeds = np.zeros(n_zones)  # 0-1
        self.coolant_flow_rates = np.zeros(n_zones)  # L/min
        self.heater_states = np.zeros(n_zones)  # 0-1
        self.thermal_interface_conductance = np.full(n_zones, 1.0)  # W/K
        
        # Safety and monitoring
        self.temperature_alarms = np.zeros(n_zones, dtype=bool)
        self.thermal_runaway_risk = np.zeros(n_zones)  # 0-1
        self.emergency_cooling_active = False
        
        # Historical data
        self.temperature_history = deque(maxlen=100)
        self.control_history = deque(maxlen=100)
        
        # Performance metrics
        self.energy_consumption = 0.0  # J
        self.temperature_stability = 1.0  # 0-1
        self.control_efficiency = 1.0  # 0-1
    
    def update(self, new_temperatures: np.ndarray, control_actions: np.ndarray):
        """Update thermal state with new measurements and control actions."""
        # Update temperatures
        old_temperatures = self.temperatures.copy()
        self.temperatures = new_temperatures
        
        # Calculate temperature gradients
        dt = 1.0  # Assuming 1-second timestep
        self.temperature_gradients = (new_temperatures - old_temperatures) / dt
        
        # Update control actions
        self.cooling_powers = control_actions[:self.n_zones]
        self.heating_powers = control_actions[self.n_zones:2*self.n_zones]
        self.fan_speeds = control_actions[2*self.n_zones:3*self.n_zones]
        
        # Update safety flags
        self._update_safety_flags()
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Store history
        self.temperature_history.append(self.temperatures.copy())
        self.control_history.append(control_actions.copy())
    
    def _update_safety_flags(self):
        """Update safety flags based on current temperatures."""
        # Temperature alarms
        self.temperature_alarms = (self.temperatures > 50.0) | (self.temperatures < 0.0)
        
        # Thermal runaway risk assessment
        for i in range(self.n_zones):
            if self.temperatures[i] > 45.0:
                # Risk increases with temperature and temperature gradient
                temp_risk = (self.temperatures[i] - 45.0) / 15.0  # Normalized
                gradient_risk = max(0, self.temperature_gradients[i]) / 5.0  # Normalized
                self.thermal_runaway_risk[i] = min(1.0, temp_risk + gradient_risk)
            else:
                self.thermal_runaway_risk[i] = 0.0
        
        # Emergency cooling activation
        self.emergency_cooling_active = np.any(self.temperatures > 55.0)
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        # Energy consumption (simplified)
        cooling_energy = np.sum(self.cooling_powers) * 1.0  # 1 second
        heating_energy = np.sum(self.heating_powers) * 1.0
        fan_energy = np.sum(self.fan_speeds) * 10.0  # 10W per fan at full speed
        self.energy_consumption += cooling_energy + heating_energy + fan_energy
        
        # Temperature stability
        if len(self.temperature_history) > 1:
            temp_variance = np.var([np.mean(temps) for temps in self.temperature_history])
            self.temperature_stability = 1.0 / (1.0 + temp_variance)
        
        # Control efficiency
        total_control_power = np.sum(self.cooling_powers) + np.sum(self.heating_powers)
        if total_control_power > 0:
            temp_error = np.mean(np.abs(self.temperatures - 25.0))
            self.control_efficiency = 1.0 / (1.0 + temp_error / total_control_power * 100)
    
    def to_observation(self) -> np.ndarray:
        """Convert thermal state to observation vector for RL agent."""
        observation = np.concatenate([
            self.temperatures,  # 4 values
            self.temperature_gradients,  # 4 values
            self.heat_generation_rates,  # 4 values
            [self.ambient_temperature, self.ambient_humidity],  # 2 values
            [self.charging_current, self.power_dissipation],  # 2 values
            self.thermal_runaway_risk,  # 4 values
            [float(self.emergency_cooling_active)],  # 1 value
            [self.energy_consumption / 1000.0],  # 1 value (normalized)
            [self.temperature_stability, self.control_efficiency],  # 2 values
            # Time-based features
            [np.sin(2 * np.pi * (time.time() % 86400) / 86400)],  # Time of day
            [np.cos(2 * np.pi * (time.time() % 86400) / 86400)]   # Time of day
        ])
        
        return observation.astype(np.float32)

class ThermalEnvironment(gym.Env):
    """
    Gym environment for battery thermal management optimization.
    """
    
    def __init__(self, config: ThermalAgentConfig):
        super().__init__()
        self.config = config
        
        # Define action and observation spaces
        # Actions: [cooling_power_zone1, cooling_power_zone2, ..., heating_power_zone1, ..., fan_speed_zone1, ...]
        self.action_space = spaces.Box(
            low=np.concatenate([
                np.zeros(config.thermal_zones),  # Cooling powers
                np.zeros(config.thermal_zones),  # Heating powers
                np.zeros(config.thermal_zones),  # Fan speeds
                [-1.0]  # Emergency override
            ]),
            high=np.concatenate([
                np.full(config.thermal_zones, config.max_cooling_power),
                np.full(config.thermal_zones, config.max_heating_power),
                np.ones(config.thermal_zones),
                [1.0]
            ]),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config.observation_space_dim,),
            dtype=np.float32
        )
        
        # Initialize thermal state
        self.thermal_state = ThermalState(config.thermal_zones)
        
        # Episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0
        
        # Reward components tracking
        self.reward_components = {
            'temperature_control': 0.0,
            'energy_efficiency': 0.0,
            'safety': 0.0,
            'stability': 0.0
        }
        
        # Physics simulator for thermal dynamics
        self.thermal_simulator = ThermalPhysicsSimulator(config)
        
        # External disturbances
        self.disturbance_generator = ThermalDisturbanceGenerator()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the thermal environment.
        
        Args:
            action (np.ndarray): Control action for thermal management
            
        Returns:
            Tuple: (observation, reward, done, info)
        """
        self.episode_step += 1
        
        # Parse and constrain actions
        n_zones = self.config.thermal_zones
        cooling_powers = np.clip(action[:n_zones], 0, self.config.max_cooling_power)
        heating_powers = np.clip(action[n_zones:2*n_zones], 0, self.config.max_heating_power)
        fan_speeds = np.clip(action[2*n_zones:3*n_zones], 0, 1)
        emergency_override = action[-1] if len(action) > 3*n_zones else 0.0
        
        # Apply emergency override if needed
        if self.thermal_state.emergency_cooling_active or emergency_override > 0.5:
            cooling_powers = np.full(n_zones, self.config.max_cooling_power)
            heating_powers = np.zeros(n_zones)
            fan_speeds = np.ones(n_zones)
        
        # Generate external disturbances
        disturbances = self.disturbance_generator.generate_disturbances(self.episode_step)
        
        # Simulate thermal dynamics
        control_vector = np.concatenate([cooling_powers, heating_powers, fan_speeds])
        new_temperatures = self.thermal_simulator.simulate_step(
            self.thermal_state, control_vector, disturbances
        )
        
        # Update thermal state
        self.thermal_state.update(new_temperatures, control_vector)
        
        # Calculate reward
        reward = self._calculate_reward(action)
        self.episode_reward += reward
        
        # Check termination conditions
        done = self._check_done()
        
        # Get observation
        observation = self.thermal_state.to_observation()
        
        # Prepare info dictionary
        info = {
            'episode_step': self.episode_step,
            'thermal_state': {
                'temperatures': self.thermal_state.temperatures.tolist(),
                'temperature_gradients': self.thermal_state.temperature_gradients.tolist(),
                'thermal_runaway_risk': self.thermal_state.thermal_runaway_risk.tolist(),
                'emergency_cooling_active': self.thermal_state.emergency_cooling_active
            },
            'reward_components': self.reward_components.copy(),
            'energy_consumption': self.thermal_state.energy_consumption,
            'control_actions': {
                'cooling_powers': cooling_powers.tolist(),
                'heating_powers': heating_powers.tolist(),
                'fan_speeds': fan_speeds.tolist()
            }
        }
        
        return observation, reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        # Reset thermal state with some randomization
        self.thermal_state = ThermalState(self.config.thermal_zones)
        
        # Randomize initial conditions
        self.thermal_state.temperatures = np.random.uniform(20.0, 35.0, self.config.thermal_zones)
        self.thermal_state.ambient_temperature = np.random.uniform(15.0, 40.0)
        self.thermal_state.charging_current = np.random.uniform(0.0, 50.0)
        
        # Reset episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0
        
        # Reset reward components
        self.reward_components = {
            'temperature_control': 0.0,
            'energy_efficiency': 0.0,
            'safety': 0.0,
            'stability': 0.0
        }
        
        # Reset simulator
        self.thermal_simulator.reset()
        
        return self.thermal_state.to_observation()
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate multi-objective reward function."""
        # Temperature control reward
        temp_errors = np.abs(self.thermal_state.temperatures - self.config.target_temperature)
        temp_control_reward = -np.mean(temp_errors) / self.config.temperature_tolerance
        
        # Energy efficiency reward
        total_energy = (np.sum(action[:self.config.thermal_zones]) +  # Cooling
                       np.sum(action[self.config.thermal_zones:2*self.config.thermal_zones]) +  # Heating
                       np.sum(action[2*self.config.thermal_zones:3*self.config.thermal_zones]) * 10)  # Fans
        efficiency_reward = -total_energy / 1000.0  # Normalize
        
        # Safety reward
        safety_reward = self._calculate_safety_reward()
        
        # Stability reward
        stability_reward = self._calculate_stability_reward()
        
        # Combine rewards with weights
        total_reward = (
            self.config.temperature_control_weight * temp_control_reward +
            self.config.energy_efficiency_weight * efficiency_reward +
            self.config.safety_weight * safety_reward +
            self.config.stability_weight * stability_reward
        )
        
        # Update reward components for tracking
        self.reward_components['temperature_control'] = temp_control_reward
        self.reward_components['energy_efficiency'] = efficiency_reward
        self.reward_components['safety'] = safety_reward
        self.reward_components['stability'] = stability_reward
        
        return total_reward
    
    def _calculate_safety_reward(self) -> float:
        """Calculate safety-based reward."""
        safety_reward = 10.0  # Base safety reward
        
        # Penalize high temperatures
        for temp in self.thermal_state.temperatures:
            if temp > self.config.critical_temp_high:
                safety_reward -= 20.0
            elif temp > self.config.emergency_cooling_threshold:
                safety_reward -= 10.0
        
        # Penalize low temperatures
        for temp in self.thermal_state.temperatures:
            if temp < self.config.critical_temp_low:
                safety_reward -= 15.0
        
        # Penalize thermal runaway risk
        max_runaway_risk = np.max(self.thermal_state.thermal_runaway_risk)
        safety_reward -= max_runaway_risk * 30.0
        
        # Severe penalty for emergency conditions
        if self.thermal_state.emergency_cooling_active:
            safety_reward -= 50.0
        
        return safety_reward
    
    def _calculate_stability_reward(self) -> float:
        """Calculate temperature stability reward."""
        if len(self.thermal_state.temperature_history) < 2:
            return 0.0
        
        # Calculate temperature variance over recent history
        recent_temps = list(self.thermal_state.temperature_history)[-10:]
        temp_variance = np.var([np.mean(temps) for temps in recent_temps])
        
        # Reward low variance (stable temperatures)
        stability_reward = 5.0 / (1.0 + temp_variance)
        
        return stability_reward
    
    def _check_done(self) -> bool:
        """Check if episode should terminate."""
        # Thermal runaway condition
        if np.any(self.thermal_state.temperatures > self.config.thermal_runaway_threshold):
            return True
        
        # Extreme cold condition
        if np.any(self.thermal_state.temperatures < self.config.critical_temp_low):
            return True
        
        # Maximum episode steps
        if self.episode_step >= self.config.max_episode_steps:
            return True
        
        return False

class ThermalPhysicsSimulator:
    """
    Physics-based thermal simulator for realistic thermal dynamics.
    """
    
    def __init__(self, config: ThermalAgentConfig):
        self.config = config
        self.dt = 1.0  # Time step in seconds
        
        # Thermal properties
        self.thermal_mass = config.thermal_mass / config.thermal_zones  # J/K per zone
        self.convection_coeff = config.convection_coefficient  # W/m²K
        self.surface_area = config.surface_area / config.thermal_zones  # m² per zone
        
        # Inter-zone thermal conductance
        self.inter_zone_conductance = 5.0  # W/K
        
    def simulate_step(self, current_state: ThermalState, 
                     control_actions: np.ndarray,
                     disturbances: Dict[str, float]) -> np.ndarray:
        """
        Simulate one time step of thermal dynamics.
        
        Args:
            current_state (ThermalState): Current thermal state
            control_actions (np.ndarray): Control actions
            disturbances (Dict[str, float]): External disturbances
            
        Returns:
            np.ndarray: Updated temperatures
        """
        n_zones = self.config.thermal_zones
        
        # Parse control actions
        cooling_powers = control_actions[:n_zones]
        heating_powers = control_actions[n_zones:2*n_zones]
        fan_speeds = control_actions[2*n_zones:3*n_zones]
        
        # Calculate heat generation for each zone
        heat_generation = self._calculate_heat_generation(current_state)
        
        # Calculate heat transfer
        convection_heat_loss = self._calculate_convection_heat_loss(
            current_state, fan_speeds, disturbances
        )
        
        # Calculate inter-zone heat transfer
        inter_zone_heat_transfer = self._calculate_inter_zone_heat_transfer(current_state)
        
        # Calculate net heat flow for each zone
        net_heat_flow = (
            heat_generation +
            heating_powers -
            cooling_powers -
            convection_heat_loss +
            inter_zone_heat_transfer
        )
        
        # Calculate temperature change
        temperature_change = net_heat_flow * self.dt / self.thermal_mass
        
        # Update temperatures
        new_temperatures = current_state.temperatures + temperature_change
        
        # Apply physical constraints
        new_temperatures = np.clip(new_temperatures, -20.0, 80.0)
        
        return new_temperatures
    
    def _calculate_heat_generation(self, state: ThermalState) -> np.ndarray:
        """Calculate heat generation in each zone."""
        # Joule heating from battery operation
        joule_heat = (state.charging_current ** 2 * state.internal_resistance) / self.config.thermal_zones
        
        # Additional heat from chemical reactions (simplified)
        reaction_heat = abs(state.charging_current) * 0.1 / self.config.thermal_zones
        
        # Distribute heat generation across zones
        heat_generation = np.full(self.config.thermal_zones, joule_heat + reaction_heat)
        
        return heat_generation
    
    def _calculate_convection_heat_loss(self, state: ThermalState, 
                                      fan_speeds: np.ndarray,
                                      disturbances: Dict[str, float]) -> np.ndarray:
        """Calculate convective heat loss for each zone."""
        # Enhanced convection coefficient with fan effect
        enhanced_coeff = self.convection_coeff * (1 + fan_speeds * 2)
        
        # Air flow effect from disturbances
        air_flow_factor = 1 + disturbances.get('air_flow', 0.0) * 0.5
        enhanced_coeff *= air_flow_factor
        
        # Temperature difference
        temp_diff = state.temperatures - (state.ambient_temperature + disturbances.get('ambient_temp_change', 0.0))
        
        # Convective heat loss
        convection_loss = enhanced_coeff * self.surface_area * temp_diff
        
        return convection_loss
    
    def _calculate_inter_zone_heat_transfer(self, state: ThermalState) -> np.ndarray:
        """Calculate heat transfer between adjacent zones."""
        n_zones = self.config.thermal_zones
        heat_transfer = np.zeros(n_zones)
        
        for i in range(n_zones):
            # Heat transfer with adjacent zones
            if i > 0:  # Heat from previous zone
                heat_transfer[i] += self.inter_zone_conductance * (state.temperatures[i-1] - state.temperatures[i])
            if i < n_zones - 1:  # Heat from next zone
                heat_transfer[i] += self.inter_zone_conductance * (state.temperatures[i+1] - state.temperatures[i])
        
        return heat_transfer
    
    def reset(self):
        """Reset simulator state."""
        pass

class ThermalDisturbanceGenerator:
    """
    Generator for external thermal disturbances.
    """
    
    def __init__(self):
        self.disturbance_history = []
        
    def generate_disturbances(self, step: int) -> Dict[str, float]:
        """Generate external disturbances for thermal simulation."""
        disturbances = {}
        
        # Ambient temperature variation
        daily_cycle = np.sin(2 * np.pi * step / 1440) * 5.0  # Daily temperature cycle
        random_variation = np.random.normal(0, 1.0)
        disturbances['ambient_temp_change'] = daily_cycle + random_variation
        
        # Air flow variation
        disturbances['air_flow'] = np.random.uniform(0.0, 1.0)
        
        # External heat sources (e.g., solar heating)
        solar_factor = max(0, np.sin(2 * np.pi * step / 1440))  # Day/night cycle
        disturbances['external_heat'] = solar_factor * np.random.uniform(0, 20.0)
        
        # Humidity effects
        disturbances['humidity_factor'] = np.random.uniform(0.8, 1.2)
        
        self.disturbance_history.append(disturbances)
        return disturbances

class ThermalAgent:
    """
    Main thermal management agent using reinforcement learning.
    """
    
    def __init__(self, config: ThermalAgentConfig):
        self.config = config
        self.env = ThermalEnvironment(config)
        
        # Create RL model
        self.model = self._create_model()
        
        # Training metrics
        self.training_metrics = {
            'episodes': 0,
            'total_steps': 0,
            'average_reward': 0.0,
            'best_reward': -np.inf,
            'temperature_violations': 0,
            'emergency_activations': 0
        }
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        logger.info(f"Thermal agent initialized with {config.algorithm} algorithm")
    
    def _create_model(self):
        """Create the RL model based on configuration."""
        if self.config.algorithm == "PPO":
            return PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                verbose=1,
                device=self.config.device,
                n_steps=2048,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5
            )
        elif self.config.algorithm == "SAC":
            return SAC(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                verbose=1,
                device=self.config.device,
                buffer_size=100000,
                learning_starts=1000,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1
            )
        elif self.config.algorithm == "DDPG":
            return DDPG(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                verbose=1,
                device=self.config.device,
                buffer_size=100000,
                learning_starts=1000,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, "episode"),
                gradient_steps=-1
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
    
    def train(self, total_timesteps: int, callback=None) -> Dict[str, Any]:
        """
        Train the thermal management agent.
        
        Args:
            total_timesteps (int): Total number of training timesteps
            callback: Optional callback for training monitoring
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        logger.info(f"Starting thermal agent training for {total_timesteps} timesteps")
        
        # Create training callback
        training_callback = self._create_training_callback(callback)
        
        # Train the model
        start_time = time.time()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=training_callback,
            progress_bar=True
        )
        training_time = time.time() - start_time
        
        # Update training metrics
        self.training_metrics['total_steps'] += total_timesteps
        self.training_metrics['episodes'] = len(self.episode_rewards)
        
        if self.episode_rewards:
            self.training_metrics['average_reward'] = np.mean(self.episode_rewards)
            self.training_metrics['best_reward'] = max(self.episode_rewards)
        
        # Compile training results
        training_results = {
            'training_time': training_time,
            'total_timesteps': total_timesteps,
            'final_metrics': self.training_metrics.copy(),
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'model_path': self._save_model()
        }
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Average reward: {self.training_metrics['average_reward']:.4f}")
        
        return training_results
    
    def _create_training_callback(self, external_callback=None):
        """Create comprehensive training callback."""
        callbacks = []
        
        # Episode tracking callback
        class EpisodeTracker(BaseCallback):
            def __init__(self, agent):
                super().__init__()
                self.agent = agent
                self.episode_count = 0
                self.episode_reward = 0
                self.episode_length = 0
            
            def _on_step(self) -> bool:
                self.episode_reward += self.locals['rewards'][0]
                self.episode_length += 1
                
                # Check for episode end
                if self.locals['dones'][0]:
                    self.agent.episode_rewards.append(self.episode_reward)
                    self.agent.episode_lengths.append(self.episode_length)
                    
                    # Log episode metrics
                    if self.episode_count % 10 == 0:
                        logger.info(f"Episode {self.episode_count}: "
                                  f"Reward={self.episode_reward:.4f}, "
                                  f"Length={self.episode_length}")
                    
                    # Reset for next episode
                    self.episode_count += 1
                    self.episode_reward = 0
                    self.episode_length = 0
                
                return True
        
        callbacks.append(EpisodeTracker(self))
        
        # Performance monitoring callback
        class PerformanceMonitor(BaseCallback):
            def __init__(self, agent):
                super().__init__()
                self.agent = agent
                self.check_freq = 1000
            
            def _on_step(self) -> bool:
                if self.n_calls % self.check_freq == 0:
                    # Check for temperature violations
                    current_temp = self.training_env.get_attr('current_temperature')[0]
                    if current_temp > self.agent.config.max_temperature:
                        self.agent.training_metrics['temperature_violations'] += 1
                    
                    # Check for emergency activations
                    if hasattr(self.training_env.get_attr('thermal_system')[0], 'emergency_active'):
                        if self.training_env.get_attr('thermal_system')[0].emergency_active:
                            self.agent.training_metrics['emergency_activations'] += 1
                
                return True
        
        callbacks.append(PerformanceMonitor(self))
        
        # Add external callback if provided
        if external_callback:
            callbacks.append(external_callback)
        
        return callbacks
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict thermal management action for given observation.
        
        Args:
            observation (np.ndarray): Current system observation
            deterministic (bool): Whether to use deterministic policy
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Action and additional info
        """
        # Get action from model
        action, _states = self.model.predict(observation, deterministic=deterministic)
        
        # Process action through thermal system constraints
        processed_action = self._process_action(action, observation)
        
        # Generate additional information
        action_info = {
            'raw_action': action,
            'processed_action': processed_action,
            'action_type': self._classify_action(processed_action),
            'confidence': self._calculate_action_confidence(observation, action),
            'safety_check': self._safety_check(observation, processed_action)
        }
        
        return processed_action, action_info
    
    def _process_action(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """Process raw action through thermal system constraints."""
        processed_action = action.copy()
        
        # Extract current state from observation
        current_temp = observation[0]
        target_temp = observation[1]
        
        # Apply safety constraints
        if current_temp > self.config.emergency_temperature:
            # Emergency cooling - override action
            processed_action[0] = 1.0  # Maximum cooling
            processed_action[1] = 0.0  # No heating
            logger.warning(f"Emergency cooling activated: temp={current_temp:.2f}°C")
        
        # Ensure action bounds
        processed_action = np.clip(processed_action, 
                                 self.env.action_space.low, 
                                 self.env.action_space.high)
        
        # Apply rate limiting
        if hasattr(self, 'last_action'):
            max_change = 0.1  # Maximum 10% change per step
            action_diff = processed_action - self.last_action
            action_diff = np.clip(action_diff, -max_change, max_change)
            processed_action = self.last_action + action_diff
        
        self.last_action = processed_action.copy()
        
        return processed_action
    
    def _classify_action(self, action: np.ndarray) -> str:
        """Classify the type of thermal action."""
        cooling_power = action[0]
        heating_power = action[1] if len(action) > 1 else 0.0
        
        if cooling_power > 0.7:
            return "aggressive_cooling"
        elif cooling_power > 0.3:
            return "moderate_cooling"
        elif heating_power > 0.3:
            return "heating"
        else:
            return "passive"
    
    def _calculate_action_confidence(self, observation: np.ndarray, action: np.ndarray) -> float:
        """Calculate confidence score for the action."""
        # Simple confidence based on action magnitude and state
        current_temp = observation[0]
        target_temp = observation[1]
        temp_error = abs(current_temp - target_temp)
        
        # Higher confidence for stronger actions when temperature error is large
        action_magnitude = np.linalg.norm(action)
        confidence = min(1.0, action_magnitude * (1.0 + temp_error / 10.0))
        
        return confidence
    
    def _safety_check(self, observation: np.ndarray, action: np.ndarray) -> Dict[str, bool]:
        """Perform safety checks on the proposed action."""
        current_temp = observation[0]
        cooling_power = action[0]
        
        safety_status = {
            'temperature_safe': current_temp < self.config.max_temperature,
            'action_magnitude_safe': np.linalg.norm(action) <= 1.0,
            'cooling_appropriate': cooling_power > 0 if current_temp > 25.0 else True,
            'emergency_override': current_temp > self.config.emergency_temperature
        }
        
        safety_status['overall_safe'] = all([
            safety_status['temperature_safe'],
            safety_status['action_magnitude_safe'],
            safety_status['cooling_appropriate']
        ])
        
        return safety_status
    
    def evaluate_performance(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate agent performance over multiple episodes.
        
        Args:
            num_episodes (int): Number of episodes to evaluate
            
        Returns:
            Dict[str, Any]: Performance evaluation results
        """
        logger.info(f"Evaluating thermal agent performance over {num_episodes} episodes")
        
        evaluation_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'temperature_violations': 0,
            'emergency_activations': 0,
            'average_temperature_error': [],
            'cooling_efficiency': [],
            'energy_consumption': []
        }
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            temp_errors = []
            energy_used = 0
            
            done = False
            while not done:
                action, action_info = self.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Track temperature error
                current_temp = obs[0]
                target_temp = obs[1]
                temp_errors.append(abs(current_temp - target_temp))
                
                # Track energy consumption
                energy_used += np.sum(np.abs(action))
                
                # Check for violations
                if current_temp > self.config.max_temperature:
                    evaluation_metrics['temperature_violations'] += 1
                
                if action_info['safety_check']['emergency_override']:
                    evaluation_metrics['emergency_activations'] += 1
            
            # Store episode metrics
            evaluation_metrics['episode_rewards'].append(episode_reward)
            evaluation_metrics['episode_lengths'].append(episode_length)
            evaluation_metrics['average_temperature_error'].append(np.mean(temp_errors))
            evaluation_metrics['energy_consumption'].append(energy_used)
            
            # Calculate cooling efficiency
            if energy_used > 0:
                efficiency = episode_reward / energy_used
                evaluation_metrics['cooling_efficiency'].append(efficiency)
        
        # Calculate summary statistics
        summary_stats = {
            'mean_reward': np.mean(evaluation_metrics['episode_rewards']),
            'std_reward': np.std(evaluation_metrics['episode_rewards']),
            'mean_episode_length': np.mean(evaluation_metrics['episode_lengths']),
            'mean_temperature_error': np.mean(evaluation_metrics['average_temperature_error']),
            'total_violations': evaluation_metrics['temperature_violations'],
            'total_emergencies': evaluation_metrics['emergency_activations'],
            'mean_energy_consumption': np.mean(evaluation_metrics['energy_consumption']),
            'mean_cooling_efficiency': np.mean(evaluation_metrics['cooling_efficiency']) if evaluation_metrics['cooling_efficiency'] else 0.0
        }
        
        logger.info(f"Evaluation completed - Mean reward: {summary_stats['mean_reward']:.4f}")
        
        return {
            'detailed_metrics': evaluation_metrics,
            'summary_statistics': summary_stats,
            'evaluation_episodes': num_episodes
        }
    
    def optimize_hyperparameters(self, param_space: Dict[str, Any], n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            param_space (Dict[str, Any]): Parameter space for optimization
            n_trials (int): Number of optimization trials
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        import optuna
        
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 
                                              param_space.get('learning_rate', [1e-5, 1e-2])[0],
                                              param_space.get('learning_rate', [1e-5, 1e-2])[1],
                                              log=True)
            
            batch_size = trial.suggest_categorical('batch_size',
                                                 param_space.get('batch_size', [32, 64, 128, 256]))
            
            # Update config
            temp_config = self.config
            temp_config.learning_rate = learning_rate
            temp_config.batch_size = batch_size
            
            # Create temporary agent
            temp_agent = ThermalAgent(temp_config)
            
            # Train for limited timesteps
            training_timesteps = param_space.get('training_timesteps', 10000)
            temp_agent.train(training_timesteps)
            
            # Evaluate performance
            eval_results = temp_agent.evaluate_performance(num_episodes=5)
            
            # Return objective value (negative because Optuna minimizes)
            return -eval_results['summary_statistics']['mean_reward']
        
        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Update agent with best parameters
        best_params = study.best_params
        self.config.learning_rate = best_params['learning_rate']
        self.config.batch_size = best_params['batch_size']
        
        # Recreate model with optimized parameters
        self.model = self._create_model()
        
        return {
            'best_parameters': best_params,
            'best_value': -study.best_value,
            'optimization_history': study.trials_dataframe().to_dict('records')
        }
    
    def _save_model(self, path: str = None) -> str:
        """Save the trained model."""
        if path is None:
            timestamp = int(time.time())
            path = f"./model_artifacts/thermal_agent_{timestamp}.zip"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        self.model.save(path)
        
        # Save additional metadata
        metadata = {
            'config': self.config.__dict__,
            'training_metrics': self.training_metrics,
            'timestamp': time.time()
        }
        
        metadata_path = path.replace('.zip', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
        return path
    
    def load_model(self, path: str) -> None:
        """Load a trained model."""
        self.model = self.model.load(path)
        
        # Load metadata if available
        metadata_path = path.replace('.zip', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.training_metrics = metadata.get('training_metrics', {})
        
        logger.info(f"Model loaded from {path}")
    
    def get_thermal_insights(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Generate thermal management insights for current state.
        
        Args:
            observation (np.ndarray): Current system observation
            
        Returns:
            Dict[str, Any]: Thermal insights and recommendations
        """
        current_temp = observation[0]
        target_temp = observation[1]
        temp_rate = observation[2] if len(observation) > 2 else 0.0
        
        # Predict action and get confidence
        action, action_info = self.predict(observation, deterministic=True)
        
        # Generate insights
        insights = {
            'current_state': {
                'temperature': current_temp,
                'target_temperature': target_temp,
                'temperature_error': current_temp - target_temp,
                'temperature_rate': temp_rate
            },
            'recommended_action': {
                'cooling_power': action[0],
                'heating_power': action[1] if len(action) > 1 else 0.0,
                'action_type': action_info['action_type'],
                'confidence': action_info['confidence']
            },
            'thermal_analysis': {
                'thermal_stress_level': self._calculate_thermal_stress(current_temp),
                'cooling_urgency': self._calculate_cooling_urgency(current_temp, target_temp),
                'energy_efficiency_score': self._calculate_efficiency_score(action),
                'safety_status': action_info['safety_check']
            },
            'predictions': {
                'temperature_trend': 'increasing' if temp_rate > 0.1 else 'decreasing' if temp_rate < -0.1 else 'stable',
                'estimated_time_to_target': self._estimate_time_to_target(current_temp, target_temp, action),
                'risk_assessment': self._assess_thermal_risk(current_temp, temp_rate)
            }
        }
        
        return insights
    
    def _calculate_thermal_stress(self, temperature: float) -> str:
        """Calculate thermal stress level."""
        if temperature > 50:
            return "high"
        elif temperature > 40:
            return "medium"
        else:
            return "low"
    
    def _calculate_cooling_urgency(self, current_temp: float, target_temp: float) -> str:
        """Calculate cooling urgency level."""
        temp_diff = current_temp - target_temp
        if temp_diff > 10:
            return "urgent"
        elif temp_diff > 5:
            return "moderate"
        else:
            return "low"
    
    def _calculate_efficiency_score(self, action: np.ndarray) -> float:
        """Calculate energy efficiency score for action."""
        # Simple efficiency metric based on action magnitude
        action_magnitude = np.linalg.norm(action)
        efficiency = max(0.0, 1.0 - action_magnitude)
        return efficiency
    
    def _estimate_time_to_target(self, current_temp: float, target_temp: float, action: np.ndarray) -> float:
        """Estimate time to reach target temperature."""
        temp_diff = abs(current_temp - target_temp)
        cooling_power = action[0]
        
        if temp_diff < 1.0:
            return 0.0
        
        # Simple estimation based on cooling power
        if cooling_power > 0.5:
            return temp_diff / (cooling_power * 2.0)  # Fast cooling
        elif cooling_power > 0.1:
            return temp_diff / cooling_power  # Normal cooling
        else:
            return temp_diff / 0.1  # Passive cooling
    
    def _assess_thermal_risk(self, temperature: float, temp_rate: float) -> str:
        """Assess thermal risk level."""
        if temperature > 55 or (temperature > 45 and temp_rate > 1.0):
            return "critical"
        elif temperature > 45 or (temperature > 35 and temp_rate > 2.0):
            return "high"
        elif temperature > 35:
            return "medium"
        else:
            return "low"

# Factory function for easy agent creation
def create_thermal_agent(config: Optional[ThermalAgentConfig] = None) -> ThermalAgent:
    """
    Factory function to create a thermal management agent.
    
    Args:
        config (ThermalAgentConfig, optional): Agent configuration
        
    Returns:
        ThermalAgent: Configured thermal agent
    """
    if config is None:
        config = ThermalAgentConfig()
    
    return ThermalAgent(config)

# Utility functions for thermal management
def optimize_thermal_policy(agent: ThermalAgent, optimization_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize thermal management policy using advanced techniques.
    
    Args:
        agent (ThermalAgent): Thermal agent to optimize
        optimization_config (Dict[str, Any]): Optimization configuration
        
    Returns:
        Dict[str, Any]: Optimization results
    """
    # Extract configuration
    method = optimization_config.get('method', 'hyperparameter_tuning')
    
    if method == 'hyperparameter_tuning':
        param_space = optimization_config.get('param_space', {
            'learning_rate': [1e-5, 1e-2],
            'batch_size': [32, 64, 128, 256],
            'training_timesteps': 20000
        })
        n_trials = optimization_config.get('n_trials', 50)
        
        return agent.optimize_hyperparameters(param_space, n_trials)
    
    else:
        raise ValueError(f"Unknown optimization method: {method}")

def evaluate_thermal_strategies(agents: List[ThermalAgent], evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare multiple thermal management strategies.
    
    Args:
        agents (List[ThermalAgent]): List of thermal agents to compare
        evaluation_config (Dict[str, Any]): Evaluation configuration
        
    Returns:
        Dict[str, Any]: Comparison results
    """
    num_episodes = evaluation_config.get('num_episodes', 10)
    comparison_results = {}
    
    for i, agent in enumerate(agents):
        agent_name = f"agent_{i}"
        evaluation_results = agent.evaluate_performance(num_episodes)
        comparison_results[agent_name] = evaluation_results
    
    # Calculate relative performance
    mean_rewards = [results['summary_statistics']['mean_reward'] 
                   for results in comparison_results.values()]
    best_reward = max(mean_rewards)
    
    for agent_name, results in comparison_results.items():
        relative_performance = results['summary_statistics']['mean_reward'] / best_reward
        results['relative_performance'] = relative_performance
    
    return {
        'individual_results': comparison_results,
        'best_agent': max(comparison_results.keys(), 
                         key=lambda x: comparison_results[x]['summary_statistics']['mean_reward']),
        'performance_ranking': sorted(comparison_results.keys(),
                                    key=lambda x: comparison_results[x]['summary_statistics']['mean_reward'],
                                    reverse=True)
    }
