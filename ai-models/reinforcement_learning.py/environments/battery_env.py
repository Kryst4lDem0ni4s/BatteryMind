"""
BatteryMind - Core Battery Environment

Comprehensive battery simulation environment for reinforcement learning
with realistic physics modeling, safety constraints, and multi-objective
reward systems. Designed for training RL agents in battery management
scenarios including health optimization and charging control.

Features:
- Realistic electrochemical and thermal modeling
- Physics-informed state transitions
- Multi-objective reward functions
- Safety constraint enforcement
- Degradation modeling and lifecycle simulation
- Integration with OpenAI Gym interface
- Comprehensive observation and action spaces

Author: BatteryMind Development Team
Version: 1.0.0
"""

import gym
from gym import spaces
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import time
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PhysicsConfig:
    """
    Configuration for battery physics simulation.
    
    Attributes:
        # Electrochemical parameters
        nominal_capacity (float): Nominal battery capacity in Ah
        nominal_voltage (float): Nominal voltage in V
        internal_resistance (float): Internal resistance in Ohms
        open_circuit_voltage_curve (List[Tuple[float, float]]): SoC vs OCV curve
        
        # Thermal parameters
        thermal_mass (float): Thermal mass in J/K
        thermal_resistance (float): Thermal resistance in K/W
        ambient_temperature (float): Ambient temperature in Celsius
        
        # Degradation parameters
        calendar_aging_rate (float): Calendar aging rate per day
        cycle_aging_rate (float): Cycle aging rate per equivalent full cycle
        temperature_aging_factor (float): Temperature acceleration factor
        
        # Simulation parameters
        timestep (float): Simulation timestep in seconds
        max_temperature (float): Maximum safe temperature in Celsius
        min_temperature (float): Minimum operating temperature in Celsius
        max_voltage (float): Maximum safe voltage in V
        min_voltage (float): Minimum safe voltage in V
        max_current (float): Maximum current magnitude in A
    """
    # Electrochemical parameters
    nominal_capacity: float = 100.0  # Ah
    nominal_voltage: float = 3.7     # V
    internal_resistance: float = 0.1  # Ohms
    open_circuit_voltage_curve: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 3.0), (0.1, 3.3), (0.2, 3.5), (0.3, 3.6), (0.4, 3.65),
        (0.5, 3.7), (0.6, 3.75), (0.7, 3.8), (0.8, 3.9), (0.9, 4.0), (1.0, 4.1)
    ])
    
    # Thermal parameters
    thermal_mass: float = 1000.0      # J/K
    thermal_resistance: float = 0.1   # K/W
    ambient_temperature: float = 25.0 # Celsius
    
    # Degradation parameters
    calendar_aging_rate: float = 1e-5  # per day
    cycle_aging_rate: float = 1e-4     # per equivalent full cycle
    temperature_aging_factor: float = 2.0
    
    # Simulation parameters
    timestep: float = 1.0              # seconds
    max_temperature: float = 60.0      # Celsius
    min_temperature: float = -20.0     # Celsius
    max_voltage: float = 4.2           # V
    min_voltage: float = 2.5           # V
    max_current: float = 100.0         # A

@dataclass
class SafetyConstraints:
    """Safety constraints for battery operation."""
    enable_thermal_protection: bool = True
    enable_voltage_protection: bool = True
    enable_current_protection: bool = True
    enable_soc_protection: bool = True
    
    thermal_shutdown_temp: float = 65.0    # Celsius
    voltage_cutoff_high: float = 4.25      # V
    voltage_cutoff_low: float = 2.3        # V
    current_limit_charge: float = 50.0     # A
    current_limit_discharge: float = -80.0 # A
    soc_limit_high: float = 0.95          # 95%
    soc_limit_low: float = 0.05           # 5%
    
    emergency_shutdown: bool = True
    safety_margin: float = 0.1

@dataclass
class BatteryState:
    """
    Comprehensive battery state representation.
    
    Attributes:
        # Core electrical state
        state_of_charge (float): State of charge (0-1)
        voltage (float): Terminal voltage in V
        current (float): Current in A (positive = charging)
        power (float): Power in W
        
        # Thermal state
        temperature (float): Battery temperature in Celsius
        heat_generation (float): Heat generation rate in W
        
        # Health and aging
        state_of_health (float): State of health (0-1)
        capacity_fade (float): Capacity fade factor (0-1)
        resistance_increase (float): Resistance increase factor (>1)
        cycle_count (float): Equivalent full cycle count
        calendar_age (float): Calendar age in days
        
        # Physics state
        internal_resistance (float): Current internal resistance in Ohms
        open_circuit_voltage (float): Open circuit voltage in V
        
        # Environmental
        ambient_temperature (float): Ambient temperature in Celsius
        
        # Safety and constraints
        safety_status (Dict[str, bool]): Safety constraint status
        constraint_violations (List[str]): Active constraint violations
        
        # Metadata
        timestamp (float): Current simulation time
        episode_step (int): Current episode step
    """
    # Core electrical state
    state_of_charge: float = 0.5
    voltage: float = 3.7
    current: float = 0.0
    power: float = 0.0
    
    # Thermal state
    temperature: float = 25.0
    heat_generation: float = 0.0
    
    # Health and aging
    state_of_health: float = 1.0
    capacity_fade: float = 0.0
    resistance_increase: float = 1.0
    cycle_count: float = 0.0
    calendar_age: float = 0.0
    
    # Physics state
    internal_resistance: float = 0.1
    open_circuit_voltage: float = 3.7
    
    # Environmental
    ambient_temperature: float = 25.0
    
    # Safety and constraints
    safety_status: Dict[str, bool] = field(default_factory=lambda: {
        "thermal_safe": True,
        "voltage_safe": True, 
        "current_safe": True,
        "soc_safe": True
    })
    constraint_violations: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: float = 0.0
    episode_step: int = 0
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for RL observation."""
        return np.array([
            self.state_of_charge,
            self.voltage,
            self.current,
            self.power,
            self.temperature,
            self.state_of_health,
            self.internal_resistance,
            self.open_circuit_voltage,
            self.ambient_temperature,
            float(all(self.safety_status.values())),
            self.cycle_count,
            self.calendar_age
        ], dtype=np.float32)
    
    def is_safe(self) -> bool:
        """Check if battery is in safe operating state."""
        return all(self.safety_status.values()) and len(self.constraint_violations) == 0

@dataclass
class BatteryAction:
    """
    Battery action representation for RL agents.
    
    Attributes:
        current_setpoint (float): Desired current in A
        thermal_control (float): Thermal management control (-1 to 1)
        safety_override (bool): Emergency safety override
    """
    current_setpoint: float = 0.0
    thermal_control: float = 0.0
    safety_override: bool = False
    
    def to_array(self) -> np.ndarray:
        """Convert action to numpy array."""
        return np.array([
            self.current_setpoint,
            self.thermal_control,
            float(self.safety_override)
        ], dtype=np.float32)
    
    @classmethod
    def from_array(cls, action_array: np.ndarray) -> 'BatteryAction':
        """Create action from numpy array."""
        return cls(
            current_setpoint=float(action_array[0]),
            thermal_control=float(action_array[1]),
            safety_override=bool(action_array[2] > 0.5)
        )

@dataclass
class BatteryReward:
    """
    Multi-objective reward structure for battery optimization.
    
    Attributes:
        # Primary objectives
        health_preservation (float): Reward for maintaining battery health
        energy_efficiency (float): Reward for energy efficiency
        safety_compliance (float): Reward for safety compliance
        
        # Secondary objectives
        temperature_control (float): Reward for temperature management
        power_delivery (float): Reward for meeting power demands
        longevity (float): Reward for extending battery life
        
        # Penalties
        constraint_violation_penalty (float): Penalty for constraint violations
        safety_violation_penalty (float): Penalty for safety violations
        
        # Composite reward
        total_reward (float): Weighted sum of all components
        
        # Weights for multi-objective optimization
        weights (Dict[str, float]): Weights for different reward components
    """
    # Primary objectives
    health_preservation: float = 0.0
    energy_efficiency: float = 0.0
    safety_compliance: float = 0.0
    
    # Secondary objectives
    temperature_control: float = 0.0
    power_delivery: float = 0.0
    longevity: float = 0.0
    
    # Penalties
    constraint_violation_penalty: float = 0.0
    safety_violation_penalty: float = 0.0
    
    # Composite reward
    total_reward: float = 0.0
    
    # Weights for multi-objective optimization
    weights: Dict[str, float] = field(default_factory=lambda: {
        "health_preservation": 0.3,
        "energy_efficiency": 0.25,
        "safety_compliance": 0.25,
        "temperature_control": 0.1,
        "power_delivery": 0.05,
        "longevity": 0.05
    })
    
    def calculate_total_reward(self) -> float:
        """Calculate weighted total reward."""
        self.total_reward = (
            self.weights["health_preservation"] * self.health_preservation +
            self.weights["energy_efficiency"] * self.energy_efficiency +
            self.weights["safety_compliance"] * self.safety_compliance +
            self.weights["temperature_control"] * self.temperature_control +
            self.weights["power_delivery"] * self.power_delivery +
            self.weights["longevity"] * self.longevity -
            self.constraint_violation_penalty -
            self.safety_violation_penalty
        )
        return self.total_reward

class BatteryPhysicsEngine:
    """
    Advanced battery physics simulation engine.
    """
    
    def __init__(self, config: PhysicsConfig):
        self.config = config
        self._setup_physics_models()
    
    def _setup_physics_models(self):
        """Initialize physics models."""
        # Create OCV interpolation function
        soc_points, ocv_points = zip(*self.config.open_circuit_voltage_curve)
        self.soc_points = np.array(soc_points)
        self.ocv_points = np.array(ocv_points)
        
        # Initialize degradation tracking
        self.degradation_state = {
            "capacity_fade": 0.0,
            "resistance_increase": 1.0,
            "cycle_count": 0.0,
            "calendar_age": 0.0
        }
    
    def get_open_circuit_voltage(self, soc: float) -> float:
        """Calculate open circuit voltage from state of charge."""
        return np.interp(soc, self.soc_points, self.ocv_points)
    
    def calculate_terminal_voltage(self, soc: float, current: float, 
                                 internal_resistance: float) -> float:
        """Calculate terminal voltage including resistive losses."""
        ocv = self.get_open_circuit_voltage(soc)
        return ocv - (current * internal_resistance)
    
    def update_thermal_state(self, current_temp: float, heat_generation: float, 
                           ambient_temp: float, timestep: float) -> float:
        """Update battery temperature using thermal model."""
        # Simple thermal model: dT/dt = (P_gen - (T-T_amb)/R_th) / C_th
        thermal_power_out = (current_temp - ambient_temp) / self.config.thermal_resistance
        net_thermal_power = heat_generation - thermal_power_out
        
        temp_change = (net_thermal_power / self.config.thermal_mass) * timestep
        new_temperature = current_temp + temp_change
        
        return new_temperature
    
    def calculate_heat_generation(self, current: float, internal_resistance: float) -> float:
        """Calculate heat generation from resistive losses."""
        return current ** 2 * internal_resistance
    
    def update_degradation(self, state: BatteryState, timestep: float) -> Dict[str, float]:
        """Update battery degradation based on stress factors."""
        # Calendar aging
        calendar_aging = self.config.calendar_aging_rate * (timestep / 86400)  # per day
        
        # Temperature acceleration factor
        temp_factor = self.config.temperature_aging_factor ** ((state.temperature - 25) / 10)
        calendar_aging *= temp_factor
        
        # Cycle aging (based on current throughput)
        current_throughput = abs(state.current) * timestep / 3600  # Ah
        cycle_aging = (current_throughput / self.config.nominal_capacity) * self.config.cycle_aging_rate
        
        # Update degradation state
        self.degradation_state["calendar_age"] += timestep / 86400
        self.degradation_state["cycle_count"] += current_throughput / self.config.nominal_capacity
        self.degradation_state["capacity_fade"] += calendar_aging + cycle_aging
        self.degradation_state["resistance_increase"] += (calendar_aging + cycle_aging) * 0.5
        
        return self.degradation_state.copy()
    
    def step_physics(self, state: BatteryState, action: BatteryAction, 
                    timestep: float) -> BatteryState:
        """
        Perform one physics simulation step.
        
        Args:
            state (BatteryState): Current battery state
            action (BatteryAction): Applied action
            timestep (float): Simulation timestep
            
        Returns:
            BatteryState: Updated battery state
        """
        new_state = BatteryState(**state.__dict__)
        
        # Update electrical state
        current = action.current_setpoint
        
        # Calculate new SoC based on current integration
        charge_change = (current * timestep) / (3600 * self.config.nominal_capacity)
        new_soc = np.clip(state.state_of_charge + charge_change, 0.0, 1.0)
        
        # Update degradation
        degradation = self.update_degradation(state, timestep)
        
        # Calculate effective capacity and resistance
        effective_capacity = self.config.nominal_capacity * (1 - degradation["capacity_fade"])
        effective_resistance = self.config.internal_resistance * degradation["resistance_increase"]
        
        # Calculate terminal voltage
        terminal_voltage = self.calculate_terminal_voltage(new_soc, current, effective_resistance)
        
        # Calculate heat generation
        heat_gen = self.calculate_heat_generation(current, effective_resistance)
        
        # Update thermal state with thermal control
        thermal_control_power = action.thermal_control * 100  # Max 100W cooling/heating
        total_heat = heat_gen + thermal_control_power
        
        new_temperature = self.update_thermal_state(
            state.temperature, total_heat, state.ambient_temperature, timestep
        )
        
        # Update state
        new_state.state_of_charge = new_soc
        new_state.voltage = terminal_voltage
        new_state.current = current
        new_state.power = terminal_voltage * current
        new_state.temperature = new_temperature
        new_state.heat_generation = heat_gen
        new_state.state_of_health = 1.0 - degradation["capacity_fade"]
        new_state.capacity_fade = degradation["capacity_fade"]
        new_state.resistance_increase = degradation["resistance_increase"]
        new_state.cycle_count = degradation["cycle_count"]
        new_state.calendar_age = degradation["calendar_age"]
        new_state.internal_resistance = effective_resistance
        new_state.open_circuit_voltage = self.get_open_circuit_voltage(new_soc)
        new_state.timestamp = state.timestamp + timestep
        new_state.episode_step = state.episode_step + 1
        
        return new_state

class BatteryEnvironment(gym.Env):
    """
    OpenAI Gym compatible battery simulation environment.
    """
    
    def __init__(self, physics_config: PhysicsConfig = None, 
                 safety_constraints: SafetyConstraints = None,
                 reward_weights: Dict[str, float] = None,
                 max_episode_steps: int = 1000,
                 random_seed: int = None):
        """
        Initialize battery environment.
        
        Args:
            physics_config (PhysicsConfig): Physics simulation configuration
            safety_constraints (SafetyConstraints): Safety constraints
            reward_weights (Dict[str, float]): Reward function weights
            max_episode_steps (int): Maximum steps per episode
            random_seed (int): Random seed for reproducibility
        """
        super().__init__()
        
        # Configuration
        self.physics_config = physics_config or PhysicsConfig()
        self.safety_constraints = safety_constraints or SafetyConstraints()
        self.max_episode_steps = max_episode_steps
        
        # Initialize physics engine
        self.physics_engine = BatteryPhysicsEngine(self.physics_config)
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
            self.np_random = np.random.RandomState(random_seed)
        else:
            self.np_random = np.random.RandomState()
        
        # Define observation space (12 continuous values)
        obs_low = np.array([
            0.0,    # SoC
            self.physics_config.min_voltage,  # voltage
            -self.physics_config.max_current, # current
            -1000.0, # power
            self.physics_config.min_temperature, # temperature
            0.0,    # SoH
            0.0,    # internal resistance
            self.physics_config.min_voltage,  # OCV
            self.physics_config.min_temperature, # ambient temp
            0.0,    # safety status
            0.0,    # cycle count
            0.0     # calendar age
        ], dtype=np.float32)
        
        obs_high = np.array([
            1.0,    # SoC
            self.physics_config.max_voltage,  # voltage
            self.physics_config.max_current,  # current
            1000.0, # power
            self.physics_config.max_temperature, # temperature
            1.0,    # SoH
            1.0,    # internal resistance
            self.physics_config.max_voltage,  # OCV
            self.physics_config.max_temperature, # ambient temp
            1.0,    # safety status
            10000.0, # cycle count
            10000.0  # calendar age
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Define action space (3 continuous values)
        action_low = np.array([
            -self.physics_config.max_current,  # current setpoint
            -1.0,  # thermal control
            0.0    # safety override
        ], dtype=np.float32)
        
        action_high = np.array([
            self.physics_config.max_current,   # current setpoint
            1.0,   # thermal control
            1.0    # safety override
        ], dtype=np.float32)
        
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        
        # Initialize state
        self.state = None
        self.episode_step_count = 0
        self.episode_reward = 0.0
        
        # Reward configuration
        self.reward_weights = reward_weights or {
            "health_preservation": 0.3,
            "energy_efficiency": 0.25,
            "safety_compliance": 0.25,
            "temperature_control": 0.1,
            "power_delivery": 0.05,
            "longevity": 0.05
        }
        
        # Episode tracking
        self.episode_history = []
        self.current_episode_data = []
        
        logger.info("BatteryEnvironment initialized")
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            np.ndarray: Initial observation
        """
        # Save previous episode data
        if self.current_episode_data:
            self.episode_history.append(self.current_episode_data.copy())
            self.current_episode_data.clear()
        
        # Initialize random state
        initial_soc = self.np_random.uniform(0.2, 0.8)
        initial_temp = self.np_random.uniform(20.0, 30.0)
        ambient_temp = self.np_random.uniform(15.0, 35.0)
        
        self.state = BatteryState(
            state_of_charge=initial_soc,
            voltage=self.physics_engine.get_open_circuit_voltage(initial_soc),
            current=0.0,
            power=0.0,
            temperature=initial_temp,
            heat_generation=0.0,
            state_of_health=1.0,
            capacity_fade=0.0,
            resistance_increase=1.0,
            cycle_count=0.0,
            calendar_age=0.0,
            internal_resistance=self.physics_config.internal_resistance,
            open_circuit_voltage=self.physics_engine.get_open_circuit_voltage(initial_soc),
            ambient_temperature=ambient_temp,
            timestamp=0.0,
            episode_step=0
        )
        
        self.episode_step_count = 0
        self.episode_reward = 0.0
        
        # Reset physics engine degradation state
        self.physics_engine.degradation_state = {
            "capacity_fade": 0.0,
            "resistance_increase": 1.0,
            "cycle_count": 0.0,
            "calendar_age": 0.0
        }
        
        return self.state.to_array()
    
    def step(self, action: Union[np.ndarray, List[float]]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.state is None:
            raise RuntimeError("Environment must be reset before stepping")
        
        # Convert action to BatteryAction
        action_array = np.array(action, dtype=np.float32)
        battery_action = BatteryAction.from_array(action_array)
        
        # Apply safety constraints to action
        battery_action = self._apply_safety_constraints(battery_action)
        
        # Step physics simulation
        new_state = self.physics_engine.step_physics(
            self.state, battery_action, self.physics_config.timestep
        )
        
        # Update safety status
        new_state = self._update_safety_status(new_state)
        
        # Calculate reward
        reward_info = self._calculate_reward(self.state, new_state, battery_action)
        reward = reward_info.total_reward
        
        # Check termination conditions
        done = self._check_termination(new_state)
        
        # Update state
        self.state = new_state
        self.episode_step_count += 1
        self.episode_reward += reward
        
        # Store episode data
        step_data = {
            "step": self.episode_step_count,
            "state": self.state.__dict__.copy(),
            "action": battery_action.__dict__.copy(),
            "reward": reward,
            "reward_components": reward_info.__dict__.copy()
        }
        self.current_episode_data.append(step_data)
        
        # Create info dictionary
        info = {
            "state": self.state.__dict__.copy(),
            "reward_components": reward_info.__dict__.copy(),
            "safety_violations": self.state.constraint_violations.copy(),
            "episode_step": self.episode_step_count,
            "episode_reward": self.episode_reward
        }
        
        return self.state.to_array(), reward, done, info
    
    def _apply_safety_constraints(self, action: BatteryAction) -> BatteryAction:
        """Apply safety constraints to action."""
        constrained_action = BatteryAction(**action.__dict__)
        
        # Current limits
        if self.safety_constraints.enable_current_protection:
            constrained_action.current_setpoint = np.clip(
                action.current_setpoint,
                self.safety_constraints.current_limit_discharge,
                self.safety_constraints.current_limit_charge
            )
        
        # Thermal control limits
        constrained_action.thermal_control = np.clip(action.thermal_control, -1.0, 1.0)
        
        return constrained_action
    
    def _update_safety_status(self, state: BatteryState) -> BatteryState:
        """Update safety status and constraint violations."""
        updated_state = BatteryState(**state.__dict__)
        updated_state.constraint_violations.clear()
        
        # Temperature safety
        if self.safety_constraints.enable_thermal_protection:
            temp_safe = (self.physics_config.min_temperature <= state.temperature <= 
                        self.physics_config.max_temperature)
            updated_state.safety_status["thermal_safe"] = temp_safe
            
            if not temp_safe:
                updated_state.constraint_violations.append("temperature_violation")
            
            # Emergency thermal shutdown
            if state.temperature >= self.safety_constraints.thermal_shutdown_temp:
                updated_state.constraint_violations.append("thermal_emergency")
        
        # Voltage safety
        if self.safety_constraints.enable_voltage_protection:
            voltage_safe = (self.safety_constraints.voltage_cutoff_low <= state.voltage <= 
                           self.safety_constraints.voltage_cutoff_high)
            updated_state.safety_status["voltage_safe"] = voltage_safe
            
            if not voltage_safe:
                updated_state.constraint_violations.append("voltage_violation")
        
        # Current safety
        if self.safety_constraints.enable_current_protection:
            current_safe = (self.safety_constraints.current_limit_discharge <= state.current <= 
                           self.safety_constraints.current_limit_charge)
            updated_state.safety_status["current_safe"] = current_safe
            
            if not current_safe:
                updated_state.constraint_violations.append("current_violation")
        
        # SoC safety
        if self.safety_constraints.enable_soc_protection:
            soc_safe = (self.safety_constraints.soc_limit_low <= state.state_of_charge <= 
                       self.safety_constraints.soc_limit_high)
            updated_state.safety_status["soc_safe"] = soc_safe
            
            if not soc_safe:
                updated_state.constraint_violations.append("soc_violation")
        
        return updated_state
    
    def _calculate_reward(self, prev_state: BatteryState, current_state: BatteryState, 
                         action: BatteryAction) -> BatteryReward:
        """Calculate multi-objective reward."""
        reward = BatteryReward(weights=self.reward_weights)
        
        # Health preservation reward
        health_change = current_state.state_of_health - prev_state.state_of_health
        reward.health_preservation = -health_change * 1000  # Penalize health loss
        
        # Energy efficiency reward
        if abs(current_state.current) > 0.1:
            efficiency = abs(current_state.power) / (abs(current_state.current) * current_state.voltage)
            reward.energy_efficiency = efficiency - 0.5  # Reward high efficiency
        
        # Safety compliance reward
        if current_state.is_safe():
            reward.safety_compliance = 1.0
        else:
            reward.safety_compliance = -1.0
        
        # Temperature control reward
        optimal_temp = 25.0
        temp_deviation = abs(current_state.temperature - optimal_temp)
        reward.temperature_control = max(0, 1.0 - temp_deviation / 20.0)
        
        # Power delivery reward (if there's a power demand)
        # This would be enhanced with actual power demand signals
        reward.power_delivery = 0.5
        
        # Longevity reward
        cycle_penalty = current_state.cycle_count * 0.001
        calendar_penalty = current_state.calendar_age * 0.0001
        reward.longevity = -(cycle_penalty + calendar_penalty)
        
        # Constraint violation penalties
        reward.constraint_violation_penalty = len(current_state.constraint_violations) * 0.5
        
        # Safety violation penalties
        if not current_state.is_safe():
            reward.safety_violation_penalty = 2.0
        
        # Calculate total reward
        reward.calculate_total_reward()
        
        return reward
    
    def _check_termination(self, state: BatteryState) -> bool:
        """Check if episode should terminate."""
        # Maximum episode steps
        if self.episode_step_count >= self.max_episode_steps:
            return True
        
        # Emergency shutdown conditions
        if self.safety_constraints.emergency_shutdown:
            # Thermal emergency
            if state.temperature >= self.safety_constraints.thermal_shutdown_temp:
                return True
            
            # Voltage emergency
            if (state.voltage <= self.safety_constraints.voltage_cutoff_low or 
                state.voltage >= self.safety_constraints.voltage_cutoff_high):
                return True
            
            # Severe health degradation
            if state.state_of_health <= 0.5:
                return True
        
        return False
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render environment state."""
        if mode == 'human':
            print(f"Step: {self.episode_step_count}")
            print(f"SoC: {self.state.state_of_charge:.3f}")
            print(f"Voltage: {self.state.voltage:.3f}V")
            print(f"Current: {self.state.current:.3f}A")
            print(f"Temperature: {self.state.temperature:.1f}°C")
            print(f"SoH: {self.state.state_of_health:.3f}")
            print(f"Safety: {self.state.is_safe()}")
            print(f"Violations: {self.state.constraint_violations}")
            print("-" * 40)
        
        return None
    
    def close(self):
        """Clean up environment resources."""
        pass
    
    def get_episode_data(self) -> List[Dict]:
        """Get data from current episode."""
        return self.current_episode_data.copy()
    
    def get_episode_history(self) -> List[List[Dict]]:
        """Get data from all completed episodes."""
        return self.episode_history.copy()

# Factory function
def create_battery_environment(config: Dict = None, **kwargs) -> BatteryEnvironment:
    """
    Factory function to create battery environment.
    
    Args:
        config (Dict): Environment configuration
        **kwargs: Additional environment parameters
        
    Returns:
        BatteryEnvironment: Configured environment instance
    """
    if config is None:
        config = {}
    
    # Extract configurations
    physics_config = PhysicsConfig(**config.get("physics", {}))
    safety_constraints = SafetyConstraints(**config.get("safety", {}))
    
    # Create environment
    env = BatteryEnvironment(
        physics_config=physics_config,
        safety_constraints=safety_constraints,
        reward_weights=config.get("reward_weights"),
        max_episode_steps=config.get("simulation", {}).get("max_episode_steps", 1000),
        random_seed=config.get("simulation", {}).get("random_seed")
    )
    
    return env
