"""
BatteryMind - Efficiency Reward System

Advanced efficiency reward mechanisms for reinforcement learning agents
optimizing battery performance, energy utilization, and operational efficiency.

Features:
- Multi-objective efficiency optimization
- Energy consumption minimization
- Charging efficiency maximization
- Thermal efficiency optimization
- Lifecycle cost optimization
- Performance-efficiency trade-offs

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EfficiencyMetrics:
    """
    Container for efficiency-related metrics.
    
    Attributes:
        energy_efficiency (float): Energy utilization efficiency (0-1)
        charging_efficiency (float): Charging process efficiency (0-1)
        thermal_efficiency (float): Thermal management efficiency (0-1)
        power_efficiency (float): Power delivery efficiency (0-1)
        cycle_efficiency (float): Battery cycle efficiency (0-1)
        cost_efficiency (float): Cost per unit performance (normalized)
        time_efficiency (float): Time utilization efficiency (0-1)
        overall_efficiency (float): Weighted overall efficiency score
    """
    energy_efficiency: float = 0.0
    charging_efficiency: float = 0.0
    thermal_efficiency: float = 0.0
    power_efficiency: float = 0.0
    cycle_efficiency: float = 0.0
    cost_efficiency: float = 0.0
    time_efficiency: float = 0.0
    overall_efficiency: float = 0.0

@dataclass
class EfficiencyConfig:
    """
    Configuration for efficiency reward calculation.
    
    Attributes:
        energy_weight (float): Weight for energy efficiency component
        charging_weight (float): Weight for charging efficiency component
        thermal_weight (float): Weight for thermal efficiency component
        power_weight (float): Weight for power efficiency component
        cycle_weight (float): Weight for cycle efficiency component
        cost_weight (float): Weight for cost efficiency component
        time_weight (float): Weight for time efficiency component
        efficiency_threshold (float): Minimum efficiency threshold for rewards
        penalty_factor (float): Penalty multiplier for inefficient operations
        bonus_factor (float): Bonus multiplier for highly efficient operations
    """
    energy_weight: float = 0.25
    charging_weight: float = 0.20
    thermal_weight: float = 0.15
    power_weight: float = 0.15
    cycle_weight: float = 0.10
    cost_weight: float = 0.10
    time_weight: float = 0.05
    efficiency_threshold: float = 0.7
    penalty_factor: float = 2.0
    bonus_factor: float = 1.5

class BaseEfficiencyReward(ABC):
    """
    Abstract base class for efficiency reward mechanisms.
    """
    
    def __init__(self, config: EfficiencyConfig):
        self.config = config
        self.efficiency_history = []
        
    @abstractmethod
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any],
                        next_state: Dict[str, Any]) -> float:
        """Calculate efficiency reward for state-action transition."""
        pass
    
    @abstractmethod
    def get_efficiency_metrics(self, state: Dict[str, Any]) -> EfficiencyMetrics:
        """Extract efficiency metrics from state."""
        pass

class EnergyEfficiencyReward(BaseEfficiencyReward):
    """
    Energy efficiency reward focusing on energy utilization optimization.
    """
    
    def __init__(self, config: EfficiencyConfig):
        super().__init__(config)
        self.baseline_consumption = 1.0  # Baseline energy consumption
        
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any],
                        next_state: Dict[str, Any]) -> float:
        """
        Calculate energy efficiency reward.
        
        Args:
            state (Dict): Current state
            action (Dict): Action taken
            next_state (Dict): Resulting state
            
        Returns:
            float: Energy efficiency reward
        """
        # Extract energy-related metrics
        current_power = state.get('power_consumption', 1.0)
        next_power = next_state.get('power_consumption', 1.0)
        energy_delivered = next_state.get('energy_delivered', 0.0)
        energy_consumed = next_state.get('energy_consumed', 1.0)
        
        # Calculate energy efficiency
        energy_efficiency = self._calculate_energy_efficiency(
            energy_delivered, energy_consumed
        )
        
        # Calculate power efficiency improvement
        power_improvement = self._calculate_power_improvement(
            current_power, next_power
        )
        
        # Calculate load matching efficiency
        load_efficiency = self._calculate_load_matching_efficiency(next_state)
        
        # Combine efficiency components
        total_efficiency = (
            0.5 * energy_efficiency +
            0.3 * power_improvement +
            0.2 * load_efficiency
        )
        
        # Apply reward scaling
        reward = self._scale_efficiency_reward(total_efficiency)
        
        return reward
    
    def _calculate_energy_efficiency(self, energy_delivered: float, 
                                   energy_consumed: float) -> float:
        """Calculate energy conversion efficiency."""
        if energy_consumed <= 0:
            return 0.0
        
        efficiency = energy_delivered / energy_consumed
        return min(1.0, efficiency)
    
    def _calculate_power_improvement(self, current_power: float, 
                                   next_power: float) -> float:
        """Calculate power consumption improvement."""
        if current_power <= 0:
            return 0.0
        
        improvement = (current_power - next_power) / current_power
        return max(0.0, min(1.0, improvement))
    
    def _calculate_load_matching_efficiency(self, state: Dict[str, Any]) -> float:
        """Calculate how well power delivery matches load requirements."""
        required_power = state.get('required_power', 1.0)
        delivered_power = state.get('delivered_power', 0.0)
        
        if required_power <= 0:
            return 1.0
        
        # Penalize both under-delivery and over-delivery
        ratio = delivered_power / required_power
        if ratio <= 1.0:
            return ratio
        else:
            return 1.0 / ratio
    
    def _scale_efficiency_reward(self, efficiency: float) -> float:
        """Scale efficiency to reward range."""
        if efficiency >= self.config.efficiency_threshold:
            # Bonus for high efficiency
            bonus = (efficiency - self.config.efficiency_threshold) * self.config.bonus_factor
            return efficiency + bonus
        else:
            # Penalty for low efficiency
            penalty = (self.config.efficiency_threshold - efficiency) * self.config.penalty_factor
            return efficiency - penalty
    
    def get_efficiency_metrics(self, state: Dict[str, Any]) -> EfficiencyMetrics:
        """Extract energy efficiency metrics from state."""
        energy_delivered = state.get('energy_delivered', 0.0)
        energy_consumed = state.get('energy_consumed', 1.0)
        
        energy_efficiency = self._calculate_energy_efficiency(
            energy_delivered, energy_consumed
        )
        
        return EfficiencyMetrics(
            energy_efficiency=energy_efficiency,
            overall_efficiency=energy_efficiency
        )

class ChargingEfficiencyReward(BaseEfficiencyReward):
    """
    Charging efficiency reward focusing on battery charging optimization.
    """
    
    def __init__(self, config: EfficiencyConfig):
        super().__init__(config)
        self.optimal_charging_rate = 0.5  # C-rate for optimal efficiency
        
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any],
                        next_state: Dict[str, Any]) -> float:
        """
        Calculate charging efficiency reward.
        
        Args:
            state (Dict): Current state
            action (Dict): Action taken
            next_state (Dict): Resulting state
            
        Returns:
            float: Charging efficiency reward
        """
        # Extract charging-related metrics
        charging_current = action.get('charging_current', 0.0)
        battery_capacity = state.get('battery_capacity', 1.0)
        soc_change = next_state.get('soc', 0.0) - state.get('soc', 0.0)
        charging_losses = next_state.get('charging_losses', 0.0)
        temperature = next_state.get('temperature', 25.0)
        
        # Calculate C-rate
        c_rate = charging_current / battery_capacity if battery_capacity > 0 else 0.0
        
        # Calculate charging efficiency components
        rate_efficiency = self._calculate_rate_efficiency(c_rate)
        thermal_efficiency = self._calculate_thermal_charging_efficiency(temperature)
        loss_efficiency = self._calculate_loss_efficiency(soc_change, charging_losses)
        
        # Combine efficiency components
        total_efficiency = (
            0.4 * rate_efficiency +
            0.3 * thermal_efficiency +
            0.3 * loss_efficiency
        )
        
        # Apply reward scaling
        reward = self._scale_efficiency_reward(total_efficiency)
        
        return reward
    
    def _calculate_rate_efficiency(self, c_rate: float) -> float:
        """Calculate efficiency based on charging rate."""
        # Optimal efficiency around 0.5C, decreasing for higher rates
        if c_rate <= 0:
            return 0.0
        
        optimal_rate = self.optimal_charging_rate
        if c_rate <= optimal_rate:
            return c_rate / optimal_rate
        else:
            # Exponential decay for high C-rates
            excess = c_rate - optimal_rate
            return optimal_rate * math.exp(-2 * excess)
    
    def _calculate_thermal_charging_efficiency(self, temperature: float) -> float:
        """Calculate efficiency based on charging temperature."""
        # Optimal temperature range: 15-35°C
        optimal_min, optimal_max = 15.0, 35.0
        
        if optimal_min <= temperature <= optimal_max:
            return 1.0
        elif temperature < optimal_min:
            # Reduced efficiency at low temperatures
            return max(0.1, 1.0 - 0.05 * (optimal_min - temperature))
        else:
            # Reduced efficiency at high temperatures
            return max(0.1, 1.0 - 0.03 * (temperature - optimal_max))
    
    def _calculate_loss_efficiency(self, soc_change: float, 
                                 charging_losses: float) -> float:
        """Calculate efficiency based on charging losses."""
        if soc_change <= 0:
            return 0.0
        
        total_energy = soc_change + charging_losses
        if total_energy <= 0:
            return 0.0
        
        efficiency = soc_change / total_energy
        return min(1.0, efficiency)
    
    def get_efficiency_metrics(self, state: Dict[str, Any]) -> EfficiencyMetrics:
        """Extract charging efficiency metrics from state."""
        charging_current = state.get('charging_current', 0.0)
        battery_capacity = state.get('battery_capacity', 1.0)
        temperature = state.get('temperature', 25.0)
        
        c_rate = charging_current / battery_capacity if battery_capacity > 0 else 0.0
        rate_efficiency = self._calculate_rate_efficiency(c_rate)
        thermal_efficiency = self._calculate_thermal_charging_efficiency(temperature)
        
        charging_efficiency = 0.6 * rate_efficiency + 0.4 * thermal_efficiency
        
        return EfficiencyMetrics(
            charging_efficiency=charging_efficiency,
            thermal_efficiency=thermal_efficiency,
            overall_efficiency=charging_efficiency
        )

class ThermalEfficiencyReward(BaseEfficiencyReward):
    """
    Thermal efficiency reward focusing on temperature management optimization.
    """
    
    def __init__(self, config: EfficiencyConfig):
        super().__init__(config)
        self.optimal_temperature_range = (20.0, 30.0)  # Optimal operating range
        
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any],
                        next_state: Dict[str, Any]) -> float:
        """
        Calculate thermal efficiency reward.
        
        Args:
            state (Dict): Current state
            action (Dict): Action taken
            next_state (Dict): Resulting state
            
        Returns:
            float: Thermal efficiency reward
        """
        # Extract thermal-related metrics
        current_temp = state.get('temperature', 25.0)
        next_temp = next_state.get('temperature', 25.0)
        cooling_power = action.get('cooling_power', 0.0)
        heating_power = action.get('heating_power', 0.0)
        ambient_temp = state.get('ambient_temperature', 25.0)
        
        # Calculate thermal efficiency components
        temperature_efficiency = self._calculate_temperature_efficiency(next_temp)
        control_efficiency = self._calculate_control_efficiency(
            current_temp, next_temp, cooling_power, heating_power
        )
        gradient_efficiency = self._calculate_gradient_efficiency(next_state)
        
        # Combine efficiency components
        total_efficiency = (
            0.5 * temperature_efficiency +
            0.3 * control_efficiency +
            0.2 * gradient_efficiency
        )
        
        # Apply reward scaling
        reward = self._scale_efficiency_reward(total_efficiency)
        
        return reward
    
    def _calculate_temperature_efficiency(self, temperature: float) -> float:
        """Calculate efficiency based on operating temperature."""
        optimal_min, optimal_max = self.optimal_temperature_range
        
        if optimal_min <= temperature <= optimal_max:
            return 1.0
        elif temperature < optimal_min:
            # Exponential decay below optimal range
            deviation = optimal_min - temperature
            return math.exp(-0.1 * deviation)
        else:
            # Exponential decay above optimal range
            deviation = temperature - optimal_max
            return math.exp(-0.15 * deviation)  # Higher penalty for overheating
    
    def _calculate_control_efficiency(self, current_temp: float, next_temp: float,
                                    cooling_power: float, heating_power: float) -> float:
        """Calculate thermal control efficiency."""
        temp_change = next_temp - current_temp
        total_power = cooling_power + heating_power
        
        if total_power <= 0:
            return 1.0 if abs(temp_change) < 1.0 else 0.5
        
        # Efficiency based on temperature change per unit power
        efficiency = abs(temp_change) / total_power
        return min(1.0, efficiency * 10)  # Scale factor
    
    def _calculate_gradient_efficiency(self, state: Dict[str, Any]) -> float:
        """Calculate thermal gradient efficiency."""
        # Assume uniform temperature distribution is most efficient
        temp_gradient = state.get('temperature_gradient', 0.0)
        max_gradient = 10.0  # Maximum acceptable gradient
        
        if temp_gradient <= max_gradient:
            return 1.0 - (temp_gradient / max_gradient)
        else:
            return 0.0
    
    def get_efficiency_metrics(self, state: Dict[str, Any]) -> EfficiencyMetrics:
        """Extract thermal efficiency metrics from state."""
        temperature = state.get('temperature', 25.0)
        temp_efficiency = self._calculate_temperature_efficiency(temperature)
        gradient_efficiency = self._calculate_gradient_efficiency(state)
        
        thermal_efficiency = 0.7 * temp_efficiency + 0.3 * gradient_efficiency
        
        return EfficiencyMetrics(
            thermal_efficiency=thermal_efficiency,
            overall_efficiency=thermal_efficiency
        )

class CompositeEfficiencyReward(BaseEfficiencyReward):
    """
    Composite efficiency reward combining multiple efficiency aspects.
    """
    
    def __init__(self, config: EfficiencyConfig):
        super().__init__(config)
        
        # Initialize component rewards
        self.energy_reward = EnergyEfficiencyReward(config)
        self.charging_reward = ChargingEfficiencyReward(config)
        self.thermal_reward = ThermalEfficiencyReward(config)
        
        # Performance tracking
        self.efficiency_history = []
        
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any],
                        next_state: Dict[str, Any]) -> float:
        """
        Calculate composite efficiency reward.
        
        Args:
            state (Dict): Current state
            action (Dict): Action taken
            next_state (Dict): Resulting state
            
        Returns:
            float: Composite efficiency reward
        """
        # Calculate individual efficiency rewards
        energy_reward = self.energy_reward.calculate_reward(state, action, next_state)
        charging_reward = self.charging_reward.calculate_reward(state, action, next_state)
        thermal_reward = self.thermal_reward.calculate_reward(state, action, next_state)
        
        # Calculate additional efficiency components
        cycle_efficiency = self._calculate_cycle_efficiency(state, next_state)
        cost_efficiency = self._calculate_cost_efficiency(state, action, next_state)
        time_efficiency = self._calculate_time_efficiency(state, action, next_state)
        
        # Weighted combination
        composite_reward = (
            self.config.energy_weight * energy_reward +
            self.config.charging_weight * charging_reward +
            self.config.thermal_weight * thermal_reward +
            self.config.cycle_weight * cycle_efficiency +
            self.config.cost_weight * cost_efficiency +
            self.config.time_weight * time_efficiency
        )
        
        # Apply efficiency bonus/penalty
        composite_reward = self._apply_efficiency_bonus(composite_reward)
        
        # Track efficiency history
        self._update_efficiency_history(composite_reward)
        
        return composite_reward
    
    def _calculate_cycle_efficiency(self, state: Dict[str, Any], 
                                  next_state: Dict[str, Any]) -> float:
        """Calculate battery cycle efficiency."""
        cycle_count = next_state.get('cycle_count', 0)
        capacity_retention = next_state.get('capacity_retention', 1.0)
        
        if cycle_count <= 0:
            return 1.0
        
        # Efficiency based on capacity retention per cycle
        efficiency = capacity_retention ** (1.0 / cycle_count)
        return min(1.0, efficiency)
    
    def _calculate_cost_efficiency(self, state: Dict[str, Any], action: Dict[str, Any],
                                 next_state: Dict[str, Any]) -> float:
        """Calculate cost efficiency."""
        energy_cost = action.get('energy_cost', 0.0)
        maintenance_cost = next_state.get('maintenance_cost', 0.0)
        performance_gain = next_state.get('performance_gain', 0.0)
        
        total_cost = energy_cost + maintenance_cost
        
        if total_cost <= 0:
            return 1.0 if performance_gain > 0 else 0.5
        
        # Cost efficiency as performance per unit cost
        efficiency = performance_gain / total_cost
        return min(1.0, efficiency)
    
    def _calculate_time_efficiency(self, state: Dict[str, Any], action: Dict[str, Any],
                                 next_state: Dict[str, Any]) -> float:
        """Calculate time efficiency."""
        time_taken = action.get('time_duration', 1.0)
        task_completion = next_state.get('task_completion', 0.0)
        
        if time_taken <= 0:
            return 0.0
        
        # Time efficiency as task completion per unit time
        efficiency = task_completion / time_taken
        return min(1.0, efficiency)
    
    def _apply_efficiency_bonus(self, base_reward: float) -> float:
        """Apply efficiency bonus/penalty based on performance."""
        if base_reward >= self.config.efficiency_threshold:
            # Apply bonus for high efficiency
            bonus = (base_reward - self.config.efficiency_threshold) * self.config.bonus_factor
            return base_reward + bonus
        else:
            # Apply penalty for low efficiency
            penalty = (self.config.efficiency_threshold - base_reward) * self.config.penalty_factor
            return base_reward - penalty
    
    def _update_efficiency_history(self, efficiency_reward: float) -> None:
        """Update efficiency history for trend analysis."""
        self.efficiency_history.append(efficiency_reward)
        
        # Keep only recent history
        if len(self.efficiency_history) > 1000:
            self.efficiency_history = self.efficiency_history[-1000:]
    
    def get_efficiency_metrics(self, state: Dict[str, Any]) -> EfficiencyMetrics:
        """Extract comprehensive efficiency metrics from state."""
        # Get individual efficiency metrics
        energy_metrics = self.energy_reward.get_efficiency_metrics(state)
        charging_metrics = self.charging_reward.get_efficiency_metrics(state)
        thermal_metrics = self.thermal_reward.get_efficiency_metrics(state)
        
        # Calculate additional metrics
        cycle_efficiency = self._calculate_cycle_efficiency(state, state)
        
        # Combine into comprehensive metrics
        overall_efficiency = (
            self.config.energy_weight * energy_metrics.energy_efficiency +
            self.config.charging_weight * charging_metrics.charging_efficiency +
            self.config.thermal_weight * thermal_metrics.thermal_efficiency +
            self.config.cycle_weight * cycle_efficiency
        )
        
        return EfficiencyMetrics(
            energy_efficiency=energy_metrics.energy_efficiency,
            charging_efficiency=charging_metrics.charging_efficiency,
            thermal_efficiency=thermal_metrics.thermal_efficiency,
            cycle_efficiency=cycle_efficiency,
            overall_efficiency=overall_efficiency
        )
    
    def get_efficiency_trend(self) -> Dict[str, float]:
        """Get efficiency trend analysis."""
        if len(self.efficiency_history) < 10:
            return {"trend": "insufficient_data", "average": 0.0}
        
        recent_avg = np.mean(self.efficiency_history[-10:])
        historical_avg = np.mean(self.efficiency_history[:-10])
        
        trend_direction = "improving" if recent_avg > historical_avg else "declining"
        
        return {
            "trend": trend_direction,
            "recent_average": recent_avg,
            "historical_average": historical_avg,
            "improvement": recent_avg - historical_avg
        }

# Factory function
def create_efficiency_reward(reward_type: str = "composite", 
                           config: Optional[EfficiencyConfig] = None) -> BaseEfficiencyReward:
    """
    Factory function to create efficiency reward instances.
    
    Args:
        reward_type (str): Type of efficiency reward
        config (EfficiencyConfig, optional): Configuration for the reward
        
    Returns:
        BaseEfficiencyReward: Configured efficiency reward instance
    """
    if config is None:
        config = EfficiencyConfig()
    
    if reward_type == "energy":
        return EnergyEfficiencyReward(config)
    elif reward_type == "charging":
        return ChargingEfficiencyReward(config)
    elif reward_type == "thermal":
        return ThermalEfficiencyReward(config)
    elif reward_type == "composite":
        return CompositeEfficiencyReward(config)
    else:
        raise ValueError(f"Unknown efficiency reward type: {reward_type}")
