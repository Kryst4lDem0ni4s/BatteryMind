"""
BatteryMind - Charging Environment

Advanced reinforcement learning environment for battery charging optimization
with realistic physics simulation, safety constraints, and multi-objective rewards.

Features:
- Physics-based battery charging simulation
- Temperature-dependent charging dynamics
- Safety constraint enforcement
- Multi-objective reward functions
- Degradation modeling during charging
- Real-time charging protocol optimization

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import math
import warnings
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BatteryParameters:
    """
    Physical and chemical parameters of the battery.
    
    Attributes:
        nominal_capacity (float): Nominal capacity in Ah
        nominal_voltage (float): Nominal voltage in V
        max_voltage (float): Maximum charging voltage in V
        min_voltage (float): Minimum discharge voltage in V
        internal_resistance (float): Internal resistance in Ohms
        thermal_capacity (float): Thermal capacity in J/K
        thermal_resistance (float): Thermal resistance in K/W
        max_charge_current (float): Maximum charging current in A
        max_discharge_current (float): Maximum discharge current in A
        initial_soh (float): Initial state of health (0-1)
        chemistry_type (str): Battery chemistry type
        cycle_life (int): Expected cycle life
        calendar_life (int): Expected calendar life in days
    """
    nominal_capacity: float = 100.0  # Ah
    nominal_voltage: float = 3.7     # V
    max_voltage: float = 4.2         # V
    min_voltage: float = 2.8         # V
    internal_resistance: float = 0.1  # Ohms
    thermal_capacity: float = 1000.0 # J/K
    thermal_resistance: float = 0.1   # K/W
    max_charge_current: float = 50.0  # A
    max_discharge_current: float = 100.0 # A
    initial_soh: float = 1.0         # 0-1
    chemistry_type: str = "LiFePO4"
    cycle_life: int = 3000
    calendar_life: int = 3650

@dataclass
class EnvironmentConfig:
    """
    Configuration for the charging environment.
    
    Attributes:
        max_episode_steps (int): Maximum steps per episode
        dt (float): Time step in seconds
        ambient_temperature (float): Ambient temperature in Celsius
        safety_temperature_limit (float): Maximum safe temperature in Celsius
        target_soc (float): Target state of charge (0-1)
        charging_efficiency (float): Charging efficiency (0-1)
        reward_weights (Dict): Weights for different reward components
        enable_degradation (bool): Whether to model battery degradation
        enable_thermal_dynamics (bool): Whether to model thermal dynamics
        observation_noise_std (float): Standard deviation of observation noise
        action_noise_std (float): Standard deviation of action noise
    """
    max_episode_steps: int = 1000
    dt: float = 60.0  # 1 minute time steps
    ambient_temperature: float = 25.0  # Celsius
    safety_temperature_limit: float = 60.0  # Celsius
    target_soc: float = 0.8  # Target 80% charge
    charging_efficiency: float = 0.95
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'charging_speed': 0.3,
        'energy_efficiency': 0.25,
        'temperature_penalty': 0.2,
        'degradation_penalty': 0.15,
        'safety_penalty': 0.1
    })
    enable_degradation: bool = True
    enable_thermal_dynamics: bool = True
    observation_noise_std: float = 0.01
    action_noise_std: float = 0.005

class BatteryPhysicsModel:
    """
    Physics-based battery model for realistic charging simulation.
    """
    
    def __init__(self, battery_params: BatteryParameters, config: EnvironmentConfig):
        self.params = battery_params
        self.config = config
        
        # State variables
        self.soc = 0.2  # Initial 20% charge
        self.soh = battery_params.initial_soh
        self.temperature = config.ambient_temperature
        self.voltage = self._calculate_ocv(self.soc)
        self.current = 0.0
        self.cycle_count = 0
        self.total_energy_charged = 0.0
        self.total_energy_discharged = 0.0
        
        # Degradation tracking
        self.capacity_fade = 0.0
        self.resistance_increase = 0.0
        self.calendar_aging_factor = 1.0
        
        # Temperature coefficients
        self.temp_coeff_capacity = -0.005  # %/°C
        self.temp_coeff_resistance = 0.01   # %/°C
        
    def _calculate_ocv(self, soc: float) -> float:
        """Calculate open circuit voltage based on SoC."""
        # Simplified OCV curve for lithium-ion battery
        if soc <= 0:
            return self.params.min_voltage
        elif soc >= 1:
            return self.params.max_voltage
        else:
            # Polynomial approximation of OCV curve
            voltage_range = self.params.max_voltage - self.params.min_voltage
            # S-curve approximation
            normalized_voltage = 1 / (1 + np.exp(-10 * (soc - 0.5)))
            return self.params.min_voltage + voltage_range * normalized_voltage
    
    def _calculate_internal_resistance(self) -> float:
        """Calculate current internal resistance including degradation and temperature effects."""
        base_resistance = self.params.internal_resistance
        
        # Temperature effect
        temp_factor = 1 + self.temp_coeff_resistance * (self.temperature - 25.0)
        
        # SoC effect (higher resistance at low SoC)
        soc_factor = 1 + 0.5 * np.exp(-5 * self.soc)
        
        # Degradation effect
        degradation_factor = 1 + self.resistance_increase
        
        return base_resistance * temp_factor * soc_factor * degradation_factor
    
    def _calculate_capacity(self) -> float:
        """Calculate current capacity including degradation and temperature effects."""
        base_capacity = self.params.nominal_capacity
        
        # Temperature effect
        temp_factor = 1 + self.temp_coeff_capacity * (self.temperature - 25.0)
        
        # Degradation effect
        degradation_factor = 1 - self.capacity_fade
        
        return base_capacity * temp_factor * degradation_factor * self.soh
    
    def _update_thermal_dynamics(self, current: float, dt: float) -> None:
        """Update battery temperature based on heat generation and dissipation."""
        if not self.config.enable_thermal_dynamics:
            return
        
        # Heat generation from I²R losses
        resistance = self._calculate_internal_resistance()
        heat_generation = current ** 2 * resistance  # Watts
        
        # Heat dissipation to ambient
        heat_dissipation = (self.temperature - self.config.ambient_temperature) / self.params.thermal_resistance
        
        # Net heat flow
        net_heat = heat_generation - heat_dissipation
        
        # Temperature change
        dT_dt = net_heat / self.params.thermal_capacity
        self.temperature += dT_dt * dt
        
        # Ensure temperature doesn't go below ambient
        self.temperature = max(self.temperature, self.config.ambient_temperature)
    
    def _update_degradation(self, current: float, dt: float) -> None:
        """Update battery degradation based on usage patterns."""
        if not self.config.enable_degradation:
            return
        
        # Cycle aging (capacity fade due to charge/discharge cycles)
        cycle_stress = abs(current) / self.params.max_charge_current
        temperature_stress = np.exp((self.temperature - 25) / 10)
        soc_stress = 1 + 0.5 * abs(self.soc - 0.5)  # Higher stress at extreme SoCs
        
        cycle_degradation_rate = 1e-6 * cycle_stress * temperature_stress * soc_stress
        self.capacity_fade += cycle_degradation_rate * dt / 3600  # Convert to hours
        
        # Resistance increase
        resistance_degradation_rate = 5e-7 * cycle_stress * temperature_stress
        self.resistance_increase += resistance_degradation_rate * dt / 3600
        
        # Calendar aging (time-based degradation)
        calendar_degradation_rate = 1e-8 * temperature_stress
        self.calendar_aging_factor *= (1 - calendar_degradation_rate * dt / 3600)
        
        # Update SoH
        self.soh = (1 - self.capacity_fade) * self.calendar_aging_factor
        self.soh = max(0.5, self.soh)  # Minimum 50% SoH
    
    def step(self, current: float, dt: float) -> Dict[str, float]:
        """
        Simulate one time step of battery dynamics.
        
        Args:
            current (float): Applied current in Amperes (positive for charging)
            dt (float): Time step in seconds
            
        Returns:
            Dict[str, float]: Updated battery state
        """
        # Clamp current to safe limits
        if current > 0:  # Charging
            current = min(current, self.params.max_charge_current)
        else:  # Discharging
            current = max(current, -self.params.max_discharge_current)
        
        # Calculate terminal voltage
        resistance = self._calculate_internal_resistance()
        ocv = self._calculate_ocv(self.soc)
        terminal_voltage = ocv - current * resistance
        
        # Check voltage limits
        if terminal_voltage > self.params.max_voltage and current > 0:
            # Voltage limit reached, reduce current
            current = (self.params.max_voltage - ocv) / resistance
        elif terminal_voltage < self.params.min_voltage and current < 0:
            # Voltage limit reached, reduce discharge current
            current = (self.params.min_voltage - ocv) / resistance
        
        # Update SoC
        capacity = self._calculate_capacity()
        if capacity > 0:
            dsoc_dt = current / (capacity * 3600)  # Convert Ah to As
            self.soc += dsoc_dt * dt
            self.soc = np.clip(self.soc, 0.0, 1.0)
        
        # Update energy counters
        energy_delta = abs(current * terminal_voltage * dt / 3600)  # Wh
        if current > 0:
            self.total_energy_charged += energy_delta
        else:
            self.total_energy_discharged += energy_delta
        
        # Update cycle count (simplified)
        if current > 0 and self.current <= 0:  # Started charging
            self.cycle_count += 0.5
        elif current < 0 and self.current >= 0:  # Started discharging
            self.cycle_count += 0.5
        
        # Update thermal dynamics
        self._update_thermal_dynamics(current, dt)
        
        # Update degradation
        self._update_degradation(current, dt)
        
        # Update stored current and voltage
        self.current = current
        self.voltage = terminal_voltage
        
        return {
            'soc': self.soc,
            'soh': self.soh,
            'voltage': self.voltage,
            'current': self.current,
            'temperature': self.temperature,
            'internal_resistance': resistance,
            'capacity': capacity,
            'cycle_count': self.cycle_count,
            'capacity_fade': self.capacity_fade,
            'resistance_increase': self.resistance_increase
        }

class ChargingEnvironment(gym.Env):
    """
    Gymnasium environment for battery charging optimization using reinforcement learning.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, battery_params: Optional[BatteryParameters] = None,
                 config: Optional[EnvironmentConfig] = None):
        super().__init__()
        
        self.battery_params = battery_params or BatteryParameters()
        self.config = config or EnvironmentConfig()
        
        # Initialize battery model
        self.battery = BatteryPhysicsModel(self.battery_params, self.config)
        
        # Environment state
        self.step_count = 0
        self.episode_reward = 0.0
        self.charging_start_time = None
        self.charging_complete = False
        
        # Define action space: [charging_current_ratio]
        # Action is normalized charging current (0 to 1, where 1 = max_charge_current)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Define observation space
        # [soc, soh, voltage, current, temperature, time_remaining, target_soc_diff]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.5, self.battery_params.min_voltage, 
                         -self.battery_params.max_discharge_current,
                         0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, self.battery_params.max_voltage,
                          self.battery_params.max_charge_current,
                          100.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Rendering
        self.render_mode = None
        self.screen = None
        self.clock = None
        
        logger.info("ChargingEnvironment initialized")
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        time_remaining = 1.0 - (self.step_count / self.config.max_episode_steps)
        target_soc_diff = self.config.target_soc - self.battery.soc
        
        obs = np.array([
            self.battery.soc,
            self.battery.soh,
            self.battery.voltage / self.battery_params.max_voltage,  # Normalized
            self.battery.current / self.battery_params.max_charge_current,  # Normalized
            self.battery.temperature / 100.0,  # Normalized to 0-100°C range
            time_remaining,
            target_soc_diff
        ], dtype=np.float32)
        
        # Add observation noise if configured
        if self.config.observation_noise_std > 0:
            noise = np.random.normal(0, self.config.observation_noise_std, obs.shape)
            obs += noise
            obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        
        return obs
    
    def _calculate_reward(self, action: np.ndarray, battery_state: Dict[str, float]) -> float:
        """Calculate reward based on charging performance and constraints."""
        reward = 0.0
        weights = self.config.reward_weights
        
        # Charging speed reward (progress toward target SoC)
        soc_progress = min(battery_state['soc'], self.config.target_soc) - \
                      min(self.battery.soc, self.config.target_soc)
        charging_speed_reward = soc_progress * weights['charging_speed'] * 100
        
        # Energy efficiency reward
        if self.battery.current > 0:  # Only during charging
            theoretical_energy = self.battery.current * self.battery_params.nominal_voltage * self.config.dt / 3600
            actual_energy = self.battery.current * battery_state['voltage'] * self.config.dt / 3600
            efficiency = actual_energy / (theoretical_energy + 1e-6)
            efficiency_reward = efficiency * weights['energy_efficiency']
        else:
            efficiency_reward = 0.0
        
        # Temperature penalty (exponential penalty for high temperatures)
        temp_penalty = 0.0
        if battery_state['temperature'] > 40.0:
            temp_excess = battery_state['temperature'] - 40.0
            temp_penalty = -np.exp(temp_excess / 10.0) * weights['temperature_penalty']
        
        # Degradation penalty
        degradation_penalty = -(battery_state['capacity_fade'] + 
                               battery_state['resistance_increase']) * weights['degradation_penalty']
        
        # Safety penalty (severe penalty for exceeding safety limits)
        safety_penalty = 0.0
        if battery_state['temperature'] > self.config.safety_temperature_limit:
            safety_penalty = -10.0 * weights['safety_penalty']
        if battery_state['voltage'] > self.battery_params.max_voltage + 0.1:
            safety_penalty -= 10.0 * weights['safety_penalty']
        
        # Completion bonus
        completion_bonus = 0.0
        if battery_state['soc'] >= self.config.target_soc and not self.charging_complete:
            completion_bonus = 5.0
            self.charging_complete = True
        
        # Combine all reward components
        reward = (charging_speed_reward + efficiency_reward + temp_penalty + 
                 degradation_penalty + safety_penalty + completion_bonus)
        
        return reward
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if target SoC reached
        if self.battery.soc >= self.config.target_soc:
            return True
        
        # Terminate if safety limits exceeded
        if self.battery.temperature > self.config.safety_temperature_limit:
            return True
        
        # Terminate if voltage limits exceeded
        if (self.battery.voltage > self.battery_params.max_voltage + 0.1 or
            self.battery.voltage < self.battery_params.min_voltage - 0.1):
            return True
        
        return False
    
    def _is_truncated(self) -> bool:
        """Check if episode should be truncated."""
        return self.step_count >= self.config.max_episode_steps
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset environment state
        self.step_count = 0
        self.episode_reward = 0.0
        self.charging_start_time = None
        self.charging_complete = False
        
        # Reset battery to initial state with some randomization
        if seed is not None:
            np.random.seed(seed)
        
        # Randomize initial conditions slightly
        initial_soc = 0.2 + np.random.uniform(-0.05, 0.05)  # 15-25% initial charge
        initial_temp = self.config.ambient_temperature + np.random.uniform(-2, 2)
        
        self.battery = BatteryPhysicsModel(self.battery_params, self.config)
        self.battery.soc = np.clip(initial_soc, 0.1, 0.3)
        self.battery.temperature = initial_temp
        self.battery.voltage = self.battery._calculate_ocv(self.battery.soc)
        
        observation = self._get_observation()
        info = {
            'battery_state': {
                'soc': self.battery.soc,
                'soh': self.battery.soh,
                'temperature': self.battery.temperature,
                'voltage': self.battery.voltage
            }
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Add action noise if configured
        if self.config.action_noise_std > 0:
            noise = np.random.normal(0, self.config.action_noise_std, action.shape)
            action = action + noise
        
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Convert normalized action to actual charging current
        charging_current = action[0] * self.battery_params.max_charge_current
        
        # Step the battery physics model
        battery_state = self.battery.step(charging_current, self.config.dt)
        
        # Calculate reward
        reward = self._calculate_reward(action, battery_state)
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        # Get new observation
        observation = self._get_observation()
        
        # Update step count
        self.step_count += 1
        
        # Prepare info dictionary
        info = {
            'battery_state': battery_state,
            'episode_reward': self.episode_reward,
            'step_count': self.step_count,
            'charging_complete': self.charging_complete,
            'safety_violation': battery_state['temperature'] > self.config.safety_temperature_limit
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = 'human'):
        """Render the environment."""
        if mode == 'human':
            print(f"Step: {self.step_count}")
            print(f"SoC: {self.battery.soc:.3f}")
            print(f"SoH: {self.battery.soh:.3f}")
            print(f"Voltage: {self.battery.voltage:.2f}V")
            print(f"Current: {self.battery.current:.2f}A")
            print(f"Temperature: {self.battery.temperature:.1f}°C")
            print(f"Episode Reward: {self.episode_reward:.2f}")
            print("-" * 40)
    
    def close(self):
        """Clean up environment resources."""
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()

# Factory functions
def create_charging_environment(battery_type: str = "default",
                              difficulty: str = "medium") -> ChargingEnvironment:
    """
    Factory function to create charging environments with predefined configurations.
    
    Args:
        battery_type (str): Type of battery ("default", "high_capacity", "fast_charge")
        difficulty (str): Difficulty level ("easy", "medium", "hard")
        
    Returns:
        ChargingEnvironment: Configured charging environment
    """
    # Battery configurations
    battery_configs = {
        "default": BatteryParameters(),
        "high_capacity": BatteryParameters(
            nominal_capacity=200.0,
            max_charge_current=100.0,
            cycle_life=5000
        ),
        "fast_charge": BatteryParameters(
            nominal_capacity=80.0,
            max_charge_current=80.0,
            thermal_capacity=800.0
        )
    }
    
    # Difficulty configurations
    difficulty_configs = {
        "easy": EnvironmentConfig(
            safety_temperature_limit=70.0,
            observation_noise_std=0.005,
            action_noise_std=0.002
        ),
        "medium": EnvironmentConfig(),
        "hard": EnvironmentConfig(
            safety_temperature_limit=50.0,
            observation_noise_std=0.02,
            action_noise_std=0.01,
            ambient_temperature=35.0
        )
    }
    
    battery_params = battery_configs.get(battery_type, BatteryParameters())
    env_config = difficulty_configs.get(difficulty, EnvironmentConfig())
    
    return ChargingEnvironment(battery_params, env_config)
