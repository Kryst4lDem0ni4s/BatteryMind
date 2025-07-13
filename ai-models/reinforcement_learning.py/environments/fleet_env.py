"""
BatteryMind - Fleet Environment

Multi-agent reinforcement learning environment for battery fleet management
with coordinated charging, load balancing, and resource optimization.

Features:
- Multi-battery fleet simulation
- Coordinated charging strategies
- Load balancing optimization
- Energy grid integration
- Fleet-wide performance metrics
- Scalable multi-agent architecture

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
from collections import defaultdict
import random

# Import charging environment components
from .charging_env import BatteryParameters, BatteryPhysicsModel, EnvironmentConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FleetConfig:
    """
    Configuration for fleet management environment.
    
    Attributes:
        num_batteries (int): Number of batteries in the fleet
        max_episode_steps (int): Maximum steps per episode
        dt (float): Time step in seconds
        total_power_limit (float): Total power limit for the fleet in kW
        energy_price_schedule (List[float]): Hourly energy prices
        demand_schedule (List[float]): Hourly energy demand schedule
        coordination_strategy (str): Fleet coordination strategy
        enable_vehicle_mobility (bool): Whether vehicles can move between locations
        grid_connection_capacity (float): Maximum grid connection capacity in kW
        renewable_energy_fraction (float): Fraction of renewable energy available
        peak_demand_penalty (float): Penalty for exceeding peak demand
        load_balancing_weight (float): Weight for load balancing objective
    """
    num_batteries: int = 10
    max_episode_steps: int = 1440  # 24 hours in minutes
    dt: float = 60.0  # 1 minute time steps
    total_power_limit: float = 500.0  # kW
    energy_price_schedule: List[float] = field(default_factory=lambda: [0.12] * 24)  # $/kWh
    demand_schedule: List[float] = field(default_factory=lambda: [50.0] * 24)  # kW
    coordination_strategy: str = "centralized"  # "centralized", "distributed", "hierarchical"
    enable_vehicle_mobility: bool = False
    grid_connection_capacity: float = 1000.0  # kW
    renewable_energy_fraction: float = 0.3
    peak_demand_penalty: float = 0.5  # $/kW
    load_balancing_weight: float = 0.2

@dataclass
class VehicleProfile:
    """
    Profile for individual vehicles in the fleet.
    
    Attributes:
        vehicle_id (str): Unique vehicle identifier
        battery_params (BatteryParameters): Battery specifications
        arrival_time (float): Arrival time in hours
        departure_time (float): Departure time in hours
        arrival_soc (float): State of charge upon arrival
        target_soc (float): Desired state of charge at departure
        priority (int): Charging priority (1-5, 5 being highest)
        location (str): Current location/charging station
        mobility_pattern (str): Mobility pattern type
    """
    vehicle_id: str
    battery_params: BatteryParameters
    arrival_time: float = 0.0
    departure_time: float = 24.0
    arrival_soc: float = 0.2
    target_soc: float = 0.8
    priority: int = 3
    location: str = "station_1"
    mobility_pattern: str = "stationary"

class FleetEnergyManager:
    """
    Energy management system for the fleet.
    """
    
    def __init__(self, config: FleetConfig):
        self.config = config
        self.current_time = 0.0  # Hours
        self.total_energy_consumed = 0.0
        self.total_energy_cost = 0.0
        self.peak_demand = 0.0
        self.renewable_energy_used = 0.0
        
        # Grid state
        self.grid_load = 0.0
        self.available_renewable = 0.0
        
    def get_current_energy_price(self) -> float:
        """Get current energy price based on time of day."""
        hour = int(self.current_time) % 24
        return self.config.energy_price_schedule[hour]
    
    def get_current_demand(self) -> float:
        """Get current base energy demand."""
        hour = int(self.current_time) % 24
        return self.config.demand_schedule[hour]
    
    def get_available_renewable_energy(self) -> float:
        """Calculate available renewable energy."""
        # Simplified solar generation model
        hour = self.current_time % 24
        if 6 <= hour <= 18:  # Daylight hours
            solar_factor = np.sin(np.pi * (hour - 6) / 12)
            renewable_capacity = self.config.grid_connection_capacity * self.config.renewable_energy_fraction
            return renewable_capacity * solar_factor
        return 0.0
    
    def update_grid_state(self, fleet_power_demand: float, dt: float) -> Dict[str, float]:
        """Update grid state based on fleet power demand."""
        # Calculate total grid load
        base_demand = self.get_current_demand()
        self.grid_load = base_demand + fleet_power_demand
        
        # Calculate available renewable energy
        self.available_renewable = self.get_available_renewable_energy()
        
        # Determine energy mix
        renewable_used = min(fleet_power_demand, self.available_renewable)
        grid_energy_used = fleet_power_demand - renewable_used
        
        # Update counters
        energy_consumed = fleet_power_demand * dt / 3600  # Convert to kWh
        self.total_energy_consumed += energy_consumed
        self.renewable_energy_used += renewable_used * dt / 3600
        
        # Calculate cost
        energy_price = self.get_current_energy_price()
        energy_cost = grid_energy_used * dt / 3600 * energy_price
        self.total_energy_cost += energy_cost
        
        # Update peak demand
        self.peak_demand = max(self.peak_demand, self.grid_load)
        
        # Update time
        self.current_time += dt / 3600  # Convert to hours
        
        return {
            'grid_load': self.grid_load,
            'renewable_used': renewable_used,
            'grid_energy_used': grid_energy_used,
            'energy_cost': energy_cost,
            'energy_price': energy_price,
            'available_renewable': self.available_renewable
        }

class FleetEnvironment(gym.Env):
    """
    Multi-agent environment for battery fleet management.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, fleet_config: Optional[FleetConfig] = None,
                 vehicle_profiles: Optional[List[VehicleProfile]] = None):
        super().__init__()
        
        self.fleet_config = fleet_config or FleetConfig()
        self.vehicle_profiles = vehicle_profiles or self._generate_default_fleet()
        
        # Initialize fleet components
        self.batteries = []
        self.energy_manager = FleetEnergyManager(self.fleet_config)
        
        # Environment state
        self.step_count = 0
        self.episode_reward = 0.0
        self.fleet_metrics = defaultdict(float)
        
        # Create battery models for each vehicle
        for profile in self.vehicle_profiles:
            battery_config = EnvironmentConfig(
                dt=self.fleet_config.dt,
                max_episode_steps=self.fleet_config.max_episode_steps
            )
            battery = BatteryPhysicsModel(profile.battery_params, battery_config)
            battery.soc = profile.arrival_soc
            battery.vehicle_profile = profile
            self.batteries.append(battery)
        
        # Define action space: charging power for each battery (normalized 0-1)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.batteries),),
            dtype=np.float32
        )
        
        # Define observation space
        # For each battery: [soc, soh, temperature, time_to_departure, target_soc_diff, priority]
        # Plus global state: [current_time, energy_price, available_renewable, grid_load]
        battery_obs_dim = 6
        global_obs_dim = 4
        total_obs_dim = len(self.batteries) * battery_obs_dim + global_obs_dim
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_obs_dim,),
            dtype=np.float32
        )
        
        logger.info(f"FleetEnvironment initialized with {len(self.batteries)} batteries")
    
    def _generate_default_fleet(self) -> List[VehicleProfile]:
        """Generate default vehicle profiles for the fleet."""
        profiles = []
        
        for i in range(self.fleet_config.num_batteries):
            # Randomize vehicle characteristics
            arrival_time = np.random.uniform(6, 10)  # Arrive between 6-10 AM
            departure_time = np.random.uniform(16, 20)  # Depart between 4-8 PM
            arrival_soc = np.random.uniform(0.15, 0.35)  # Arrive with 15-35% charge
            target_soc = np.random.uniform(0.75, 0.95)  # Target 75-95% charge
            priority = np.random.randint(1, 6)  # Random priority 1-5
            
            # Vary battery parameters
            if i < 3:  # Some high-capacity vehicles
                battery_params = BatteryParameters(
                    nominal_capacity=150.0,
                    max_charge_current=75.0
                )
            elif i < 6:  # Some fast-charging vehicles
                battery_params = BatteryParameters(
                    nominal_capacity=80.0,
                    max_charge_current=80.0
                )
            else:  # Standard vehicles
                battery_params = BatteryParameters()
            
            profile = VehicleProfile(
                vehicle_id=f"vehicle_{i:03d}",
                battery_params=battery_params,
                arrival_time=arrival_time,
                departure_time=departure_time,
                arrival_soc=arrival_soc,
                target_soc=target_soc,
                priority=priority,
                location=f"station_{i % 5 + 1}"  # Distribute across 5 stations
            )
            profiles.append(profile)
        
        return profiles
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector for the entire fleet."""
        obs = []
        
        # Battery-specific observations
        for battery in self.batteries:
            profile = battery.vehicle_profile
            current_time_hours = self.step_count * self.fleet_config.dt / 3600
            
            # Time to departure (normalized)
            time_to_departure = max(0, profile.departure_time - current_time_hours) / 24
            
            # Target SoC difference
            target_soc_diff = profile.target_soc - battery.soc
            
            # Priority (normalized)
            priority_norm = profile.priority / 5.0
            
            battery_obs = [
                battery.soc,
                battery.soh,
                battery.temperature / 100.0,  # Normalized temperature
                time_to_departure,
                target_soc_diff,
                priority_norm
            ]
            obs.extend(battery_obs)
        
        # Global state observations
        current_time_norm = (self.step_count * self.fleet_config.dt / 3600) % 24 / 24
        energy_price_norm = self.energy_manager.get_current_energy_price() / 0.5  # Normalize to typical max price
        available_renewable_norm = self.energy_manager.get_available_renewable_energy() / self.fleet_config.grid_connection_capacity
        grid_load_norm = self.energy_manager.grid_load / self.fleet_config.grid_connection_capacity
        
        global_obs = [
            current_time_norm,
            energy_price_norm,
            available_renewable_norm,
            grid_load_norm
        ]
        obs.extend(global_obs)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_fleet_reward(self, actions: np.ndarray, 
                               battery_states: List[Dict], 
                               grid_state: Dict) -> float:
        """Calculate reward for the entire fleet."""
        reward = 0.0
        
        # Individual battery rewards
        for i, (battery, state) in enumerate(zip(self.batteries, battery_states)):
            profile = battery.vehicle_profile
            
            # Charging progress reward
            soc_progress = state['soc'] - battery.soc
            progress_reward = soc_progress * profile.priority * 10
            
            # Target achievement reward
            if state['soc'] >= profile.target_soc:
                target_reward = 5.0 * profile.priority
            else:
                target_reward = 0.0
            
            # Temperature penalty
            temp_penalty = 0.0
            if state['temperature'] > 50.0:
                temp_penalty = -0.1 * (state['temperature'] - 50.0)
            
            # Degradation penalty
            degradation_penalty = -state['capacity_fade'] * 10
            
            battery_reward = progress_reward + target_reward + temp_penalty + degradation_penalty
            reward += battery_reward
        
        # Fleet-level rewards
        
        # Energy cost penalty
        energy_cost_penalty = -grid_state['energy_cost'] * 100  # Scale cost impact
        
        # Renewable energy bonus
        renewable_bonus = grid_state['renewable_used'] * 0.1
        
        # Load balancing reward
        total_power = sum(actions) * sum(b.vehicle_profile.battery_params.max_charge_current for b in self.batteries)
        power_variance = np.var([actions[i] * self.batteries[i].vehicle_profile.battery_params.max_charge_current 
                               for i in range(len(self.batteries))])
        load_balance_reward = -power_variance * self.fleet_config.load_balancing_weight
        
        # Grid stability reward (penalize exceeding grid capacity)
        if grid_state['grid_load'] > self.fleet_config.grid_connection_capacity:
            grid_penalty = -10.0 * (grid_state['grid_load'] - self.fleet_config.grid_connection_capacity)
        else:
            grid_penalty = 0.0
        
        # Peak demand penalty
        if grid_state['grid_load'] > self.energy_manager.peak_demand * 0.95:
            peak_penalty = -self.fleet_config.peak_demand_penalty * grid_state['grid_load']
        else:
            peak_penalty = 0.0
        
        # Combine all rewards
        fleet_reward = (energy_cost_penalty + renewable_bonus + load_balance_reward + 
                       grid_penalty + peak_penalty)
        
        total_reward = reward + fleet_reward
        
        # Update fleet metrics
        self.fleet_metrics['total_energy_cost'] = self.energy_manager.total_energy_cost
        self.fleet_metrics['renewable_fraction'] = (self.energy_manager.renewable_energy_used / 
                                                   max(self.energy_manager.total_energy_consumed, 1e-6))
        self.fleet_metrics['peak_demand'] = self.energy_manager.peak_demand
        self.fleet_metrics['average_soc'] = np.mean([b.soc for b in self.batteries])
        
        return total_reward
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Check if all vehicles have reached their target SoC
        all_charged = all(battery.soc >= battery.vehicle_profile.target_soc 
                         for battery in self.batteries)
        
        # Check for safety violations
        safety_violation = any(battery.temperature > 70.0 for battery in self.batteries)
        
        return all_charged or safety_violation
    
    def _is_truncated(self) -> bool:
        """Check if episode should be truncated."""
        return self.step_count >= self.fleet_config.max_episode_steps
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the fleet environment."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset environment state
        self.step_count = 0
        self.episode_reward = 0.0
        self.fleet_metrics = defaultdict(float)
        
        # Reset energy manager
        self.energy_manager = FleetEnergyManager(self.fleet_config)
        
        # Reset all batteries
        for i, battery in enumerate(self.batteries):
            profile = battery.vehicle_profile
            
            # Reset battery state with some randomization
            initial_soc = profile.arrival_soc + np.random.uniform(-0.05, 0.05)
            initial_temp = 25.0 + np.random.uniform(-3, 3)
            
            battery_config = EnvironmentConfig(
                dt=self.fleet_config.dt,
                max_episode_steps=self.fleet_config.max_episode_steps
            )
            self.batteries[i] = BatteryPhysicsModel(profile.battery_params, battery_config)
            self.batteries[i].soc = np.clip(initial_soc, 0.1, 0.4)
            self.batteries[i].temperature = initial_temp
            self.batteries[i].vehicle_profile = profile
        
        observation = self._get_observation()
        info = {
            'fleet_size': len(self.batteries),
            'total_capacity': sum(b.vehicle_profile.battery_params.nominal_capacity for b in self.batteries),
            'average_soc': np.mean([b.soc for b in self.batteries]),
            'energy_manager_state': {
                'current_time': self.energy_manager.current_time,
                'energy_price': self.energy_manager.get_current_energy_price()
            }
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step for the entire fleet."""
        # Clip actions to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Convert normalized actions to actual charging currents
        charging_currents = []
        total_power_demand = 0.0
        
        for i, battery in enumerate(self.batteries):
            max_current = battery.vehicle_profile.battery_params.max_charge_current
            current = action[i] * max_current
            charging_currents.append(current)
            
            # Calculate power demand (simplified)
            power = current * battery.voltage / 1000  # Convert to kW
            total_power_demand += power
        
        # Check total power limit
        if total_power_demand > self.fleet_config.total_power_limit:
            # Scale down all currents proportionally
            scale_factor = self.fleet_config.total_power_limit / total_power_demand
            charging_currents = [current * scale_factor for current in charging_currents]
            total_power_demand = self.fleet_config.total_power_limit
        
        # Update grid state
        grid_state = self.energy_manager.update_grid_state(total_power_demand, self.fleet_config.dt)
        
        # Step each battery
        battery_states = []
        for i, battery in enumerate(self.batteries):
            state = battery.step(charging_currents[i], self.fleet_config.dt)
            battery_states.append(state)
        
        # Calculate reward
        reward = self._calculate_fleet_reward(action, battery_states, grid_state)
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
            'battery_states': battery_states,
            'grid_state': grid_state,
            'fleet_metrics': dict(self.fleet_metrics),
            'episode_reward': self.episode_reward,
            'step_count': self.step_count,
            'total_power_demand': total_power_demand,
            'charging_currents': charging_currents
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = 'human'):
        """Render the fleet environment."""
        if mode == 'human':
            print(f"\n=== Fleet Status - Step {self.step_count} ===")
            print(f"Time: {self.energy_manager.current_time:.2f} hours")
            print(f"Energy Price: ${self.energy_manager.get_current_energy_price():.3f}/kWh")
            print(f"Grid Load: {self.energy_manager.grid_load:.1f} kW")
            print(f"Available Renewable: {self.energy_manager.available_renewable:.1f} kW")
            
            print("\nBattery Status:")
            for i, battery in enumerate(self.batteries):
                profile = battery.vehicle_profile
                print(f"  {profile.vehicle_id}: SoC={battery.soc:.2f}, "
                      f"Target={profile.target_soc:.2f}, "
                      f"Temp={battery.temperature:.1f}°C, "
                      f"Priority={profile.priority}")
            
            print(f"\nFleet Metrics:")
            for key, value in self.fleet_metrics.items():
                print(f"  {key}: {value:.3f}")
            
            print(f"Episode Reward: {self.episode_reward:.2f}")
            print("=" * 50)
    
    def close(self):
        """Clean up environment resources."""
        pass

# Factory functions
def create_fleet_environment(fleet_size: int = 10, 
                           scenario: str = "mixed_fleet") -> FleetEnvironment:
    """
    Factory function to create fleet environments with predefined scenarios.
    
    Args:
        fleet_size (int): Number of vehicles in the fleet
        scenario (str): Fleet scenario ("mixed_fleet", "delivery_fleet", "passenger_fleet")
        
    Returns:
        FleetEnvironment: Configured fleet environment
    """
    fleet_config = FleetConfig(num_batteries=fleet_size)
    
    if scenario == "delivery_fleet":
        # Commercial delivery vehicles
        fleet_config.energy_price_schedule = [0.08] * 6 + [0.15] * 12 + [0.08] * 6  # Peak pricing
        fleet_config.total_power_limit = 200.0  # Lower power limit
        
        vehicle_profiles = []
        for i in range(fleet_size):
            profile = VehicleProfile(
                vehicle_id=f"delivery_{i:03d}",
                battery_params=BatteryParameters(nominal_capacity=120.0, max_charge_current=60.0),
                arrival_time=np.random.uniform(18, 22),  # Evening return
                departure_time=np.random.uniform(6, 8),   # Morning departure
                arrival_soc=np.random.uniform(0.1, 0.3),
                target_soc=0.95,  # Full charge needed
                priority=4,  # High priority for commercial use
                mobility_pattern="delivery"
            )
            vehicle_profiles.append(profile)
    
    elif scenario == "passenger_fleet":
        # Personal passenger vehicles
        fleet_config.renewable_energy_fraction = 0.4  # Higher renewable integration
        
        vehicle_profiles = []
        for i in range(fleet_size):
            profile = VehicleProfile(
                vehicle_id=f"passenger_{i:03d}",
                battery_params=BatteryParameters(nominal_capacity=75.0, max_charge_current=40.0),
                arrival_time=np.random.uniform(17, 20),  # Evening arrival
                departure_time=np.random.uniform(7, 9),   # Morning departure
                arrival_soc=np.random.uniform(0.2, 0.4),
                target_soc=np.random.uniform(0.7, 0.9),
                priority=np.random.randint(2, 4),
                mobility_pattern="commuter"
            )
            vehicle_profiles.append(profile)
    
    else:  # mixed_fleet
        vehicle_profiles = None  # Use default generation
    
    return FleetEnvironment(fleet_config, vehicle_profiles)

def create_multi_location_fleet(num_locations: int = 3, 
                              vehicles_per_location: int = 5) -> FleetEnvironment:
    """
    Create a fleet environment with multiple charging locations.
    
    Args:
        num_locations (int): Number of charging locations
        vehicles_per_location (int): Average vehicles per location
        
    Returns:
        FleetEnvironment: Multi-location fleet environment
    """
    total_vehicles = num_locations * vehicles_per_location
    fleet_config = FleetConfig(
        num_batteries=total_vehicles,
        total_power_limit=100.0 * num_locations,  # Scale power limit
        enable_vehicle_mobility=True
    )
    
    vehicle_profiles = []
    for loc in range(num_locations):
        for v in range(vehicles_per_location):
            profile = VehicleProfile(
                vehicle_id=f"loc{loc}_veh{v:02d}",
                battery_params=BatteryParameters(),
                arrival_time=np.random.uniform(6, 10),
                departure_time=np.random.uniform(16, 20),
                arrival_soc=np.random.uniform(0.15, 0.35),
                target_soc=np.random.uniform(0.75, 0.95),
                priority=np.random.randint(1, 6),
                location=f"location_{loc}",
                mobility_pattern="mobile" if np.random.random() < 0.3 else "stationary"
            )
            vehicle_profiles.append(profile)
    
    return FleetEnvironment(fleet_config, vehicle_profiles)
