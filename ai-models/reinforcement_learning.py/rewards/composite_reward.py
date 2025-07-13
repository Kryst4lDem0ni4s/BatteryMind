"""
BatteryMind - Composite Reward System for Reinforcement Learning

Advanced composite reward system that combines multiple reward components for
comprehensive battery optimization. Enables sophisticated multi-objective
optimization for battery health, efficiency, and safety.

Features:
- Multi-objective reward composition with configurable weights
- Dynamic reward scaling and normalization
- Temporal reward shaping for long-term optimization
- Adaptive weight adjustment based on performance
- Safety constraint integration
- Performance monitoring and analytics

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import time
from abc import ABC, abstractmethod
from collections import deque, defaultdict
import json
import warnings

# Local imports
from .battery_health_reward import BatteryHealthReward
from .efficiency_reward import EfficiencyReward
from .safety_reward import SafetyReward

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RewardComponent:
    """
    Configuration for individual reward components.
    
    Attributes:
        name (str): Name of the reward component
        weight (float): Weight for this component in composite reward
        enabled (bool): Whether this component is active
        min_value (float): Minimum expected reward value
        max_value (float): Maximum expected reward value
        normalization_method (str): Normalization method ('minmax', 'zscore', 'none')
        temporal_discount (float): Temporal discount factor for this component
    """
    name: str
    weight: float = 1.0
    enabled: bool = True
    min_value: float = -1.0
    max_value: float = 1.0
    normalization_method: str = "minmax"
    temporal_discount: float = 1.0

@dataclass
class CompositeRewardConfig:
    """
    Configuration for composite reward system.
    
    Attributes:
        components (Dict[str, RewardComponent]): Individual reward components
        global_scaling_factor (float): Global scaling factor for final reward
        adaptive_weights (bool): Enable adaptive weight adjustment
        weight_adaptation_rate (float): Learning rate for weight adaptation
        safety_override (bool): Enable safety override for critical violations
        temporal_shaping (bool): Enable temporal reward shaping
        performance_tracking (bool): Enable performance tracking and analytics
        normalization_window (int): Window size for running normalization
    """
    components: Dict[str, RewardComponent] = field(default_factory=dict)
    global_scaling_factor: float = 1.0
    adaptive_weights: bool = True
    weight_adaptation_rate: float = 0.01
    safety_override: bool = True
    temporal_shaping: bool = True
    performance_tracking: bool = True
    normalization_window: int = 1000

class RewardNormalizer:
    """
    Handles normalization of reward components for stable training.
    """
    
    def __init__(self, method: str = "minmax", window_size: int = 1000):
        self.method = method
        self.window_size = window_size
        self.value_history = deque(maxlen=window_size)
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0
    
    def update(self, value: float) -> None:
        """Update normalizer with new value."""
        self.value_history.append(value)
        
        # Update running statistics
        self.count += 1
        delta = value - self.running_mean
        self.running_mean += delta / self.count
        delta2 = value - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.count
    
    def normalize(self, value: float, min_val: float = None, max_val: float = None) -> float:
        """
        Normalize value using specified method.
        
        Args:
            value (float): Value to normalize
            min_val (float, optional): Minimum value for minmax normalization
            max_val (float, optional): Maximum value for minmax normalization
            
        Returns:
            float: Normalized value
        """
        if self.method == "minmax":
            if min_val is not None and max_val is not None:
                if max_val > min_val:
                    return (value - min_val) / (max_val - min_val)
                else:
                    return 0.0
            elif len(self.value_history) > 1:
                hist_min = min(self.value_history)
                hist_max = max(self.value_history)
                if hist_max > hist_min:
                    return (value - hist_min) / (hist_max - hist_min)
                else:
                    return 0.0
            else:
                return value
        
        elif self.method == "zscore":
            if self.count > 1 and self.running_var > 0:
                std_dev = np.sqrt(self.running_var)
                return (value - self.running_mean) / std_dev
            else:
                return value
        
        elif self.method == "none":
            return value
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

class AdaptiveWeightManager:
    """
    Manages adaptive weight adjustment for reward components.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.component_performance = defaultdict(list)
        self.weight_history = defaultdict(list)
        self.adaptation_count = 0
    
    def update_performance(self, component_name: str, performance_score: float) -> None:
        """Update performance score for a component."""
        self.component_performance[component_name].append(performance_score)
        
        # Keep only recent history
        if len(self.component_performance[component_name]) > 100:
            self.component_performance[component_name] = \
                self.component_performance[component_name][-100:]
    
    def adapt_weights(self, current_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Adapt weights based on component performance.
        
        Args:
            current_weights (Dict[str, float]): Current component weights
            
        Returns:
            Dict[str, float]: Adapted weights
        """
        if self.adaptation_count < 10:  # Wait for sufficient data
            self.adaptation_count += 1
            return current_weights
        
        adapted_weights = current_weights.copy()
        
        # Calculate performance-based adjustments
        for component_name in current_weights:
            if component_name in self.component_performance:
                recent_performance = self.component_performance[component_name][-10:]
                if len(recent_performance) >= 5:
                    avg_performance = np.mean(recent_performance)
                    performance_trend = np.mean(np.diff(recent_performance))
                    
                    # Increase weight for improving components
                    if performance_trend > 0:
                        weight_adjustment = self.learning_rate * performance_trend
                        adapted_weights[component_name] += weight_adjustment
                    
                    # Decrease weight for declining components
                    elif performance_trend < 0:
                        weight_adjustment = self.learning_rate * abs(performance_trend)
                        adapted_weights[component_name] = max(
                            0.1, adapted_weights[component_name] - weight_adjustment
                        )
        
        # Normalize weights to sum to original total
        original_sum = sum(current_weights.values())
        current_sum = sum(adapted_weights.values())
        if current_sum > 0:
            normalization_factor = original_sum / current_sum
            adapted_weights = {k: v * normalization_factor for k, v in adapted_weights.items()}
        
        # Store weight history
        for component_name, weight in adapted_weights.items():
            self.weight_history[component_name].append(weight)
        
        return adapted_weights

class TemporalRewardShaper:
    """
    Implements temporal reward shaping for long-term optimization.
    """
    
    def __init__(self, discount_factor: float = 0.99, shaping_horizon: int = 10):
        self.discount_factor = discount_factor
        self.shaping_horizon = shaping_horizon
        self.reward_history = deque(maxlen=shaping_horizon)
        self.shaped_rewards = deque(maxlen=shaping_horizon)
    
    def shape_reward(self, current_reward: float, future_potential: float = 0.0) -> float:
        """
        Apply temporal shaping to current reward.
        
        Args:
            current_reward (float): Current step reward
            future_potential (float): Estimated future potential
            
        Returns:
            float: Shaped reward
        """
        # Store current reward
        self.reward_history.append(current_reward)
        
        # Calculate shaped reward using potential-based shaping
        if len(self.reward_history) > 1:
            # Potential-based reward shaping: F(s,a,s') = γΦ(s') - Φ(s)
            # where Φ is the potential function
            current_potential = self._calculate_potential()
            previous_potential = self._calculate_potential(offset=1)
            
            shaping_bonus = self.discount_factor * current_potential - previous_potential
            shaped_reward = current_reward + shaping_bonus + future_potential
        else:
            shaped_reward = current_reward + future_potential
        
        self.shaped_rewards.append(shaped_reward)
        return shaped_reward
    
    def _calculate_potential(self, offset: int = 0) -> float:
        """Calculate potential function value."""
        if len(self.reward_history) <= offset:
            return 0.0
        
        # Simple potential based on recent reward trend
        recent_rewards = list(self.reward_history)[:-offset] if offset > 0 else list(self.reward_history)
        if len(recent_rewards) < 2:
            return 0.0
        
        # Potential based on reward improvement trend
        trend = np.mean(np.diff(recent_rewards[-5:]))  # Last 5 steps
        return trend * len(recent_rewards)

class CompositeReward:
    """
    Main composite reward system that combines multiple reward components.
    """
    
    def __init__(self, config: CompositeRewardConfig):
        self.config = config
        
        # Initialize reward components
        self.reward_components = {
            'battery_health': BatteryHealthReward(),
            'efficiency': EfficiencyReward(),
            'safety': SafetyReward()
        }
        
        # Initialize normalizers for each component
        self.normalizers = {}
        for name, component_config in config.components.items():
            if component_config.enabled:
                self.normalizers[name] = RewardNormalizer(
                    method=component_config.normalization_method,
                    window_size=config.normalization_window
                )
        
        # Initialize adaptive weight manager
        if config.adaptive_weights:
            self.weight_manager = AdaptiveWeightManager(config.weight_adaptation_rate)
        else:
            self.weight_manager = None
        
        # Initialize temporal shaper
        if config.temporal_shaping:
            self.temporal_shaper = TemporalRewardShaper()
        else:
            self.temporal_shaper = None
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.reward_history = deque(maxlen=10000)
        self.component_contributions = defaultdict(list)
        
        logger.info("Composite reward system initialized with components: " + 
                   ", ".join([name for name, comp in config.components.items() if comp.enabled]))
    
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any], 
                        next_state: Dict[str, Any], info: Dict[str, Any] = None) -> float:
        """
        Calculate composite reward from multiple components.
        
        Args:
            state (Dict[str, Any]): Current state
            action (Dict[str, Any]): Action taken
            next_state (Dict[str, Any]): Next state
            info (Dict[str, Any], optional): Additional information
            
        Returns:
            float: Composite reward value
        """
        if info is None:
            info = {}
        
        component_rewards = {}
        normalized_rewards = {}
        
        # Calculate individual component rewards
        for component_name, component_config in self.config.components.items():
            if not component_config.enabled:
                continue
            
            try:
                if component_name in self.reward_components:
                    # Calculate raw reward
                    raw_reward = self.reward_components[component_name].calculate_reward(
                        state, action, next_state, info
                    )
                    component_rewards[component_name] = raw_reward
                    
                    # Update normalizer
                    if component_name in self.normalizers:
                        self.normalizers[component_name].update(raw_reward)
                        
                        # Normalize reward
                        normalized_reward = self.normalizers[component_name].normalize(
                            raw_reward, 
                            component_config.min_value, 
                            component_config.max_value
                        )
                        normalized_rewards[component_name] = normalized_reward
                    else:
                        normalized_rewards[component_name] = raw_reward
                    
                    # Track component contribution
                    self.component_contributions[component_name].append(normalized_reward)
                
            except Exception as e:
                logger.warning(f"Error calculating {component_name} reward: {e}")
                normalized_rewards[component_name] = 0.0
        
        # Apply safety override if enabled
        if self.config.safety_override and 'safety' in normalized_rewards:
            safety_reward = normalized_rewards['safety']
            if safety_reward < -0.8:  # Critical safety violation
                logger.warning("Critical safety violation detected - applying safety override")
                return safety_reward * 10.0  # Heavily penalize unsafe actions
        
        # Get current weights
        current_weights = {name: comp.weight for name, comp in self.config.components.items() 
                          if comp.enabled}
        
        # Adapt weights if enabled
        if self.weight_manager:
            # Update performance scores (simplified)
            for component_name in normalized_rewards:
                performance_score = normalized_rewards[component_name]
                self.weight_manager.update_performance(component_name, performance_score)
            
            # Adapt weights
            current_weights = self.weight_manager.adapt_weights(current_weights)
        
        # Calculate weighted composite reward
        composite_reward = 0.0
        total_weight = 0.0
        
        for component_name, reward_value in normalized_rewards.items():
            if component_name in current_weights:
                weight = current_weights[component_name]
                component_config = self.config.components[component_name]
                
                # Apply temporal discount if specified
                discounted_reward = reward_value * component_config.temporal_discount
                
                # Add weighted contribution
                composite_reward += weight * discounted_reward
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            composite_reward /= total_weight
        
        # Apply global scaling
        composite_reward *= self.config.global_scaling_factor
        
        # Apply temporal shaping if enabled
        if self.temporal_shaper:
            # Estimate future potential (simplified)
            future_potential = self._estimate_future_potential(next_state, info)
            composite_reward = self.temporal_shaper.shape_reward(composite_reward, future_potential)
        
        # Store reward history
        self.reward_history.append(composite_reward)
        
        # Update performance tracking
        if self.config.performance_tracking:
            self._update_performance_tracking(component_rewards, normalized_rewards, 
                                            current_weights, composite_reward)
        
        return composite_reward
    
    def _estimate_future_potential(self, next_state: Dict[str, Any], 
                                 info: Dict[str, Any]) -> float:
        """Estimate future reward potential for temporal shaping."""
        # Simplified future potential estimation
        potential = 0.0
        
        # Battery health potential
        if 'battery_soh' in next_state:
            soh = next_state['battery_soh']
            if soh > 0.8:
                potential += 0.1  # Good health has positive potential
            elif soh < 0.6:
                potential -= 0.1  # Poor health has negative potential
        
        # Efficiency potential
        if 'energy_efficiency' in next_state:
            efficiency = next_state['energy_efficiency']
            if efficiency > 0.9:
                potential += 0.05
            elif efficiency < 0.7:
                potential -= 0.05
        
        # Safety potential
        if 'temperature' in next_state:
            temp = next_state['temperature']
            if temp > 60 or temp < -10:  # Extreme temperatures
                potential -= 0.2
        
        return potential
    
    def _update_performance_tracking(self, component_rewards: Dict[str, float],
                                   normalized_rewards: Dict[str, float],
                                   weights: Dict[str, float],
                                   composite_reward: float) -> None:
        """Update performance tracking metrics."""
        timestamp = time.time()
        
        performance_entry = {
            'timestamp': timestamp,
            'component_rewards': component_rewards.copy(),
            'normalized_rewards': normalized_rewards.copy(),
            'weights': weights.copy(),
            'composite_reward': composite_reward
        }
        
        self.performance_history['entries'].append(performance_entry)
        
        # Keep only recent history
        if len(self.performance_history['entries']) > 1000:
            self.performance_history['entries'] = self.performance_history['entries'][-1000:]
    
    def get_reward_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown of reward components."""
        if not self.reward_history:
            return {}
        
        breakdown = {
            'composite_reward_stats': {
                'mean': np.mean(self.reward_history),
                'std': np.std(self.reward_history),
                'min': np.min(self.reward_history),
                'max': np.max(self.reward_history),
                'count': len(self.reward_history)
            },
            'component_contributions': {}
        }
        
        # Component contribution analysis
        for component_name, contributions in self.component_contributions.items():
            if contributions:
                breakdown['component_contributions'][component_name] = {
                    'mean': np.mean(contributions[-100:]),  # Recent contributions
                    'std': np.std(contributions[-100:]),
                    'trend': np.mean(np.diff(contributions[-10:])) if len(contributions) > 10 else 0.0
                }
        
        # Weight analysis if adaptive weights enabled
        if self.weight_manager:
            breakdown['adaptive_weights'] = {}
            for component_name, weight_history in self.weight_manager.weight_history.items():
                if weight_history:
                    breakdown['adaptive_weights'][component_name] = {
                        'current': weight_history[-1],
                        'initial': weight_history[0],
                        'trend': np.mean(np.diff(weight_history[-10:])) if len(weight_history) > 10 else 0.0
                    }
        
        return breakdown
    
    def reset(self) -> None:
        """Reset the composite reward system for new episode."""
        # Reset temporal shaper
        if self.temporal_shaper:
            self.temporal_shaper.reward_history.clear()
            self.temporal_shaper.shaped_rewards.clear()
        
        # Reset individual reward components
        for component in self.reward_components.values():
            if hasattr(component, 'reset'):
                component.reset()
    
    def update_config(self, new_config: CompositeRewardConfig) -> None:
        """Update configuration during training."""
        self.config = new_config
        
        # Update normalizers if needed
        for name, component_config in new_config.components.items():
            if component_config.enabled and name not in self.normalizers:
                self.normalizers[name] = RewardNormalizer(
                    method=component_config.normalization_method,
                    window_size=new_config.normalization_window
                )
            elif not component_config.enabled and name in self.normalizers:
                del self.normalizers[name]
        
        logger.info("Composite reward configuration updated")
    
    def save_performance_data(self, filepath: str) -> None:
        """Save performance tracking data to file."""
        if self.config.performance_tracking:
            with open(filepath, 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)
            logger.info(f"Performance data saved to {filepath}")
    
    def load_performance_data(self, filepath: str) -> None:
        """Load performance tracking data from file."""
        try:
            with open(filepath, 'r') as f:
                self.performance_history = json.load(f)
            logger.info(f"Performance data loaded from {filepath}")
        except Exception as e:
            logger.warning(f"Failed to load performance data: {e}")

# Factory functions
def create_default_composite_reward() -> CompositeReward:
    """Create composite reward with default configuration."""
    config = CompositeRewardConfig(
        components={
            'battery_health': RewardComponent(
                name='battery_health',
                weight=0.4,
                enabled=True,
                min_value=-1.0,
                max_value=1.0,
                normalization_method='minmax'
            ),
            'efficiency': RewardComponent(
                name='efficiency',
                weight=0.3,
                enabled=True,
                min_value=-1.0,
                max_value=1.0,
                normalization_method='minmax'
            ),
            'safety': RewardComponent(
                name='safety',
                weight=0.3,
                enabled=True,
                min_value=-1.0,
                max_value=1.0,
                normalization_method='minmax'
            )
        },
        adaptive_weights=True,
        temporal_shaping=True,
        performance_tracking=True
    )
    
    return CompositeReward(config)

def create_battery_optimization_reward() -> CompositeReward:
    """Create composite reward optimized for battery health and longevity."""
    config = CompositeRewardConfig(
        components={
            'battery_health': RewardComponent(
                name='battery_health',
                weight=0.6,  # Higher weight for battery health
                enabled=True,
                min_value=-1.0,
                max_value=1.0,
                normalization_method='minmax',
                temporal_discount=0.99
            ),
            'efficiency': RewardComponent(
                name='efficiency',
                weight=0.2,
                enabled=True,
                min_value=-1.0,
                max_value=1.0,
                normalization_method='minmax',
                temporal_discount=0.95
            ),
            'safety': RewardComponent(
                name='safety',
                weight=0.2,
                enabled=True,
                min_value=-1.0,
                max_value=1.0,
                normalization_method='minmax',
                temporal_discount=1.0  # No discount for safety
            )
        },
        global_scaling_factor=1.0,
        adaptive_weights=True,
        weight_adaptation_rate=0.005,  # Slower adaptation
        safety_override=True,
        temporal_shaping=True,
        performance_tracking=True
    )
    
    return CompositeReward(config)

def create_efficiency_focused_reward() -> CompositeReward:
    """Create composite reward focused on energy efficiency."""
    config = CompositeRewardConfig(
        components={
            'battery_health': RewardComponent(
                name='battery_health',
                weight=0.2,
                enabled=True,
                min_value=-1.0,
                max_value=1.0,
                normalization_method='minmax'
            ),
            'efficiency': RewardComponent(
                name='efficiency',
                weight=0.6,  # Higher weight for efficiency
                enabled=True,
                min_value=-1.0,
                max_value=1.0,
                normalization_method='minmax'
            ),
            'safety': RewardComponent(
                name='safety',
                weight=0.2,
                enabled=True,
                min_value=-1.0,
                max_value=1.0,
                normalization_method='minmax'
            )
        },
        adaptive_weights=False,  # Fixed weights for efficiency focus
        temporal_shaping=False,
        performance_tracking=True
    )
    
    return CompositeReward(config)
