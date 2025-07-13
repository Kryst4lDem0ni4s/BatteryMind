"""
BatteryMind - Battery Health Reward Functions

Sophisticated reward functions for optimizing battery health and longevity
in reinforcement learning applications. These rewards are based on battery
electrochemistry principles and real-world degradation mechanisms.

Features:
- State of Health (SoH) optimization rewards
- Degradation penalty mechanisms
- Capacity retention incentives
- Thermal health management rewards
- Cycle life extension rewards
- Physics-informed reward shaping

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import math
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HealthRewardConfig:
    """
    Configuration for battery health reward functions.
    
    Attributes:
        # State of Health parameters
        soh_target (float): Target State of Health (0-1)
        soh_weight (float): Weight for SoH in overall health reward
        soh_tolerance (float): Tolerance for SoH target achievement
        
        # Degradation parameters
        degradation_penalty_weight (float): Weight for degradation penalties
        max_degradation_rate (float): Maximum acceptable degradation rate
        degradation_threshold (float): Threshold for severe degradation penalty
        
        # Capacity parameters
        capacity_retention_weight (float): Weight for capacity retention
        initial_capacity (float): Initial battery capacity (Ah)
        min_acceptable_capacity (float): Minimum acceptable capacity ratio
        
        # Thermal parameters
        thermal_weight (float): Weight for thermal health
        optimal_temp_range (Tuple[float, float]): Optimal temperature range (°C)
        thermal_penalty_factor (float): Factor for thermal penalty calculation
        
        # Cycle life parameters
        cycle_life_weight (float): Weight for cycle life optimization
        target_cycle_life (int): Target number of cycles
        cycle_degradation_factor (float): Degradation factor per cycle
        
        # Reward shaping parameters
        reward_shaping (bool): Enable reward shaping
        temporal_discount (float): Temporal discount factor for future health
        smoothing_factor (float): Smoothing factor for reward signals
    """
    # State of Health parameters
    soh_target: float = 0.8
    soh_weight: float = 0.4
    soh_tolerance: float = 0.05
    
    # Degradation parameters
    degradation_penalty_weight: float = 0.3
    max_degradation_rate: float = 0.001  # per cycle
    degradation_threshold: float = 0.005
    
    # Capacity parameters
    capacity_retention_weight: float = 0.2
    initial_capacity: float = 100.0  # Ah
    min_acceptable_capacity: float = 0.7
    
    # Thermal parameters
    thermal_weight: float = 0.1
    optimal_temp_range: Tuple[float, float] = (15.0, 35.0)
    thermal_penalty_factor: float = 0.01
    
    # Cycle life parameters
    cycle_life_weight: float = 0.0
    target_cycle_life: int = 3000
    cycle_degradation_factor: float = 0.0001
    
    # Reward shaping parameters
    reward_shaping: bool = True
    temporal_discount: float = 0.95
    smoothing_factor: float = 0.1

class BaseHealthReward(ABC):
    """
    Abstract base class for battery health reward functions.
    """
    
    def __init__(self, config: HealthRewardConfig):
        self.config = config
        self.previous_reward = 0.0
        
    @abstractmethod
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any],
                        next_state: Dict[str, Any]) -> float:
        """Calculate reward based on battery state and action."""
        pass
    
    def normalize_reward(self, reward: float) -> float:
        """Normalize reward to [-1, 1] range."""
        return np.tanh(reward)
    
    def apply_smoothing(self, reward: float) -> float:
        """Apply temporal smoothing to reward signal."""
        if self.config.reward_shaping:
            smoothed = ((1 - self.config.smoothing_factor) * self.previous_reward + 
                       self.config.smoothing_factor * reward)
            self.previous_reward = smoothed
            return smoothed
        return reward

class StateOfHealthReward(BaseHealthReward):
    """
    Reward function based on State of Health (SoH) optimization.
    """
    
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any],
                        next_state: Dict[str, Any]) -> float:
        """
        Calculate reward based on SoH improvement or maintenance.
        
        Args:
            state (Dict): Current battery state
            action (Dict): Action taken
            next_state (Dict): Resulting battery state
            
        Returns:
            float: SoH-based reward
        """
        current_soh = state.get('state_of_health', 1.0)
        next_soh = next_state.get('state_of_health', 1.0)
        
        # Calculate SoH change
        soh_change = next_soh - current_soh
        
        # Reward for maintaining or improving SoH
        if next_soh >= self.config.soh_target:
            # Bonus for achieving target SoH
            target_bonus = 1.0
            # Additional bonus for improvement
            improvement_bonus = max(0, soh_change * 10)
            reward = target_bonus + improvement_bonus
        else:
            # Penalty proportional to distance from target
            distance_penalty = (self.config.soh_target - next_soh) / self.config.soh_target
            # Additional penalty for degradation
            degradation_penalty = max(0, -soh_change * 20)
            reward = -distance_penalty - degradation_penalty
        
        # Apply tolerance
        if abs(next_soh - self.config.soh_target) <= self.config.soh_tolerance:
            reward += 0.5  # Tolerance bonus
        
        return self.apply_smoothing(self.normalize_reward(reward))

class DegradationPenaltyReward(BaseHealthReward):
    """
    Reward function that penalizes battery degradation.
    """
    
    def __init__(self, config: HealthRewardConfig):
        super().__init__(config)
        self.degradation_history = []
        
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any],
                        next_state: Dict[str, Any]) -> float:
        """
        Calculate reward based on degradation penalties.
        
        Args:
            state (Dict): Current battery state
            action (Dict): Action taken
            next_state (Dict): Resulting battery state
            
        Returns:
            float: Degradation-based reward
        """
        # Calculate various degradation metrics
        capacity_degradation = self._calculate_capacity_degradation(state, next_state)
        resistance_degradation = self._calculate_resistance_degradation(state, next_state)
        thermal_degradation = self._calculate_thermal_degradation(state, next_state)
        
        # Combine degradation metrics
        total_degradation = (capacity_degradation + resistance_degradation + thermal_degradation) / 3
        
        # Store in history for trend analysis
        self.degradation_history.append(total_degradation)
        if len(self.degradation_history) > 100:
            self.degradation_history.pop(0)
        
        # Calculate reward
        if total_degradation <= self.config.max_degradation_rate:
            # Reward for low degradation
            reward = 1.0 - (total_degradation / self.config.max_degradation_rate)
        else:
            # Penalty for high degradation
            penalty_factor = total_degradation / self.config.max_degradation_rate
            if total_degradation > self.config.degradation_threshold:
                penalty_factor *= 2  # Severe penalty for extreme degradation
            reward = -penalty_factor
        
        # Add trend-based adjustment
        if len(self.degradation_history) >= 10:
            recent_trend = np.mean(self.degradation_history[-5:]) - np.mean(self.degradation_history[-10:-5])
            if recent_trend > 0:  # Increasing degradation
                reward -= 0.2
            else:  # Decreasing degradation
                reward += 0.1
        
        return self.apply_smoothing(self.normalize_reward(reward))
    
    def _calculate_capacity_degradation(self, state: Dict, next_state: Dict) -> float:
        """Calculate capacity degradation rate."""
        current_capacity = state.get('capacity', self.config.initial_capacity)
        next_capacity = next_state.get('capacity', self.config.initial_capacity)
        
        capacity_loss = max(0, current_capacity - next_capacity)
        return capacity_loss / self.config.initial_capacity
    
    def _calculate_resistance_degradation(self, state: Dict, next_state: Dict) -> float:
        """Calculate internal resistance degradation rate."""
        current_resistance = state.get('internal_resistance', 0.01)
        next_resistance = next_state.get('internal_resistance', 0.01)
        
        resistance_increase = max(0, next_resistance - current_resistance)
        return resistance_increase / current_resistance if current_resistance > 0 else 0
    
    def _calculate_thermal_degradation(self, state: Dict, next_state: Dict) -> float:
        """Calculate thermal-induced degradation."""
        temperature = next_state.get('temperature', 25.0)
        
        # Arrhenius-based thermal degradation
        if temperature > 25.0:
            thermal_factor = np.exp((temperature - 25.0) / 10.0) - 1
        else:
            thermal_factor = 0
        
        return thermal_factor * 0.001  # Scale factor

class CapacityRetentionReward(BaseHealthReward):
    """
    Reward function for maintaining battery capacity.
    """
    
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any],
                        next_state: Dict[str, Any]) -> float:
        """
        Calculate reward based on capacity retention.
        
        Args:
            state (Dict): Current battery state
            action (Dict): Action taken
            next_state (Dict): Resulting battery state
            
        Returns:
            float: Capacity retention reward
        """
        current_capacity = next_state.get('capacity', self.config.initial_capacity)
        capacity_ratio = current_capacity / self.config.initial_capacity
        
        # Reward based on capacity retention
        if capacity_ratio >= self.config.min_acceptable_capacity:
            # Exponential reward for high capacity retention
            reward = np.exp(2 * (capacity_ratio - self.config.min_acceptable_capacity))
        else:
            # Severe penalty for low capacity
            penalty = (self.config.min_acceptable_capacity - capacity_ratio) * 10
            reward = -penalty
        
        # Bonus for capacity above initial (if possible through optimization)
        if capacity_ratio > 1.0:
            reward += (capacity_ratio - 1.0) * 5
        
        return self.apply_smoothing(self.normalize_reward(reward))

class ThermalHealthReward(BaseHealthReward):
    """
    Reward function for thermal health management.
    """
    
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any],
                        next_state: Dict[str, Any]) -> float:
        """
        Calculate reward based on thermal health.
        
        Args:
            state (Dict): Current battery state
            action (Dict): Action taken
            next_state (Dict): Resulting battery state
            
        Returns:
            float: Thermal health reward
        """
        temperature = next_state.get('temperature', 25.0)
        temp_gradient = next_state.get('temperature_gradient', 0.0)
        
        # Optimal temperature range reward
        temp_min, temp_max = self.config.optimal_temp_range
        
        if temp_min <= temperature <= temp_max:
            # Reward for being in optimal range
            temp_reward = 1.0
        else:
            # Penalty for being outside optimal range
            if temperature < temp_min:
                temp_penalty = (temp_min - temperature) * self.config.thermal_penalty_factor
            else:
                temp_penalty = (temperature - temp_max) * self.config.thermal_penalty_factor
            temp_reward = -temp_penalty
        
        # Penalty for high temperature gradients (thermal stress)
        gradient_penalty = abs(temp_gradient) * 0.1
        
        # Severe penalty for dangerous temperatures
        if temperature > 60.0 or temperature < -20.0:
            danger_penalty = 5.0
        else:
            danger_penalty = 0.0
        
        total_reward = temp_reward - gradient_penalty - danger_penalty
        
        return self.apply_smoothing(self.normalize_reward(total_reward))

class CycleLifeReward(BaseHealthReward):
    """
    Reward function for optimizing battery cycle life.
    """
    
    def __init__(self, config: HealthRewardConfig):
        super().__init__(config)
        self.cycle_count = 0
        self.depth_of_discharge_history = []
        
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any],
                        next_state: Dict[str, Any]) -> float:
        """
        Calculate reward based on cycle life optimization.
        
        Args:
            state (Dict): Current battery state
            action (Dict): Action taken
            next_state (Dict): Resulting battery state
            
        Returns:
            float: Cycle life reward
        """
        # Track cycle completion
        current_soc = state.get('state_of_charge', 0.5)
        next_soc = next_state.get('state_of_charge', 0.5)
        
        # Detect cycle completion (simplified)
        if current_soc < 0.2 and next_soc > 0.8:
            self.cycle_count += 1
            depth_of_discharge = 1.0 - min(current_soc, next_soc)
            self.depth_of_discharge_history.append(depth_of_discharge)
        
        # Calculate cycle life impact
        if self.cycle_count > 0:
            # Reward for reaching target cycle life
            cycle_progress = min(1.0, self.cycle_count / self.config.target_cycle_life)
            cycle_reward = cycle_progress
            
            # Penalty for deep discharges (reduces cycle life)
            if self.depth_of_discharge_history:
                avg_dod = np.mean(self.depth_of_discharge_history[-10:])
                if avg_dod > 0.8:  # Deep discharge penalty
                    cycle_reward -= (avg_dod - 0.8) * 2
            
            # Bonus for shallow cycling
            if len(self.depth_of_discharge_history) >= 5:
                recent_dod = np.mean(self.depth_of_discharge_history[-5:])
                if recent_dod < 0.5:
                    cycle_reward += 0.2
        else:
            cycle_reward = 0.0
        
        return self.apply_smoothing(self.normalize_reward(cycle_reward))

class BatteryHealthReward:
    """
    Comprehensive battery health reward combining multiple health aspects.
    """
    
    def __init__(self, config: HealthRewardConfig):
        self.config = config
        
        # Initialize component rewards
        self.soh_reward = StateOfHealthReward(config)
        self.degradation_reward = DegradationPenaltyReward(config)
        self.capacity_reward = CapacityRetentionReward(config)
        self.thermal_reward = ThermalHealthReward(config)
        self.cycle_reward = CycleLifeReward(config)
        
        # Reward history for analysis
        self.reward_history = []
        self.component_history = {
            'soh': [],
            'degradation': [],
            'capacity': [],
            'thermal': [],
            'cycle': []
        }
        
        logger.info("BatteryHealthReward initialized with comprehensive health metrics")
    
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any],
                        next_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comprehensive battery health reward.
        
        Args:
            state (Dict): Current battery state
            action (Dict): Action taken
            next_state (Dict): Resulting battery state
            
        Returns:
            Dict[str, float]: Component rewards and total reward
        """
        # Calculate component rewards
        soh_reward = self.soh_reward.calculate_reward(state, action, next_state)
        degradation_reward = self.degradation_reward.calculate_reward(state, action, next_state)
        capacity_reward = self.capacity_reward.calculate_reward(state, action, next_state)
        thermal_reward = self.thermal_reward.calculate_reward(state, action, next_state)
        cycle_reward = self.cycle_reward.calculate_reward(state, action, next_state)
        
        # Weighted combination
        total_reward = (
            self.config.soh_weight * soh_reward +
            self.config.degradation_penalty_weight * degradation_reward +
            self.config.capacity_retention_weight * capacity_reward +
            self.config.thermal_weight * thermal_reward +
            self.config.cycle_life_weight * cycle_reward
        )
        
        # Store in history
        component_rewards = {
            'soh': soh_reward,
            'degradation': degradation_reward,
            'capacity': capacity_reward,
            'thermal': thermal_reward,
            'cycle': cycle_reward,
            'total': total_reward
        }
        
        self.reward_history.append(total_reward)
        for component, reward in component_rewards.items():
            if component != 'total':
                self.component_history[component].append(reward)
        
        # Limit history size
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)
            for component in self.component_history:
                if len(self.component_history[component]) > 1000:
                    self.component_history[component].pop(0)
        
        return component_rewards
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get statistics about reward performance."""
        if not self.reward_history:
            return {}
        
        stats = {
            'total_reward': {
                'mean': np.mean(self.reward_history),
                'std': np.std(self.reward_history),
                'min': np.min(self.reward_history),
                'max': np.max(self.reward_history),
                'trend': self._calculate_trend(self.reward_history)
            }
        }
        
        # Component statistics
        for component, history in self.component_history.items():
            if history:
                stats[f'{component}_reward'] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': np.min(history),
                    'max': np.max(history),
                    'trend': self._calculate_trend(history)
                }
        
        return stats
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for reward values."""
        if len(values) < 10:
            return "insufficient_data"
        
        # Simple trend calculation using linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"
    
    def reset(self):
        """Reset reward components for new episode."""
        self.soh_reward.previous_reward = 0.0
        self.degradation_reward.previous_reward = 0.0
        self.degradation_reward.degradation_history.clear()
        self.capacity_reward.previous_reward = 0.0
        self.thermal_reward.previous_reward = 0.0
        self.cycle_reward.previous_reward = 0.0
        self.cycle_reward.cycle_count = 0
        self.cycle_reward.depth_of_discharge_history.clear()

# Factory function
def create_battery_health_reward(config: Optional[HealthRewardConfig] = None) -> BatteryHealthReward:
    """
    Factory function to create a BatteryHealthReward.
    
    Args:
        config (HealthRewardConfig, optional): Reward configuration
        
    Returns:
        BatteryHealthReward: Configured reward instance
    """
    if config is None:
        config = HealthRewardConfig()
    
    return BatteryHealthReward(config)
