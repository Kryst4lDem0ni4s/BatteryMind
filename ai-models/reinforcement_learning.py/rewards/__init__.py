"""
BatteryMind - Reinforcement Learning Rewards Package

Comprehensive reward function implementations for battery optimization and management
using reinforcement learning. This package provides sophisticated reward mechanisms
that balance multiple objectives including battery health, efficiency, safety, and
sustainability.

Key Components:
- BatteryHealthReward: Rewards for maintaining and improving battery health
- EfficiencyReward: Rewards for energy efficiency and performance optimization
- SafetyReward: Rewards for maintaining safe operating conditions
- CompositeReward: Multi-objective reward combining various reward signals
- SustainabilityReward: Rewards for environmental and sustainability considerations

Features:
- Multi-objective reward optimization with configurable weights
- Physics-informed reward functions based on battery electrochemistry
- Adaptive reward scaling based on battery state and conditions
- Safety-constrained rewards with hard constraint enforcement
- Temporal reward shaping for long-term optimization
- Uncertainty-aware reward estimation

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

from .battery_health_reward import (
    BatteryHealthReward,
    HealthRewardConfig,
    StateOfHealthReward,
    DegradationPenaltyReward,
    CapacityRetentionReward,
    ThermalHealthReward
)

from .efficiency_reward import (
    EfficiencyReward,
    EfficiencyRewardConfig,
    EnergyEfficiencyReward,
    ChargingEfficiencyReward,
    PowerDeliveryReward,
    RoundTripEfficiencyReward
)

from .safety_reward import (
    SafetyReward,
    SafetyRewardConfig,
    ThermalSafetyReward,
    VoltageSafetyReward,
    CurrentSafetyReward,
    OverchargeProtectionReward
)

from .composite_reward import (
    CompositeReward,
    CompositeRewardConfig,
    MultiObjectiveReward,
    WeightedRewardCombiner,
    AdaptiveRewardWeighting,
    ParetoOptimalReward
)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module metadata
__all__ = [
    # Battery health rewards
    "BatteryHealthReward",
    "HealthRewardConfig", 
    "StateOfHealthReward",
    "DegradationPenaltyReward",
    "CapacityRetentionReward",
    "ThermalHealthReward",
    
    # Efficiency rewards
    "EfficiencyReward",
    "EfficiencyRewardConfig",
    "EnergyEfficiencyReward", 
    "ChargingEfficiencyReward",
    "PowerDeliveryReward",
    "RoundTripEfficiencyReward",
    
    # Safety rewards
    "SafetyReward",
    "SafetyRewardConfig",
    "ThermalSafetyReward",
    "VoltageSafetyReward", 
    "CurrentSafetyReward",
    "OverchargeProtectionReward",
    
    # Composite rewards
    "CompositeReward",
    "CompositeRewardConfig",
    "MultiObjectiveReward",
    "WeightedRewardCombiner",
    "AdaptiveRewardWeighting",
    "ParetoOptimalReward"
]

# Default reward configuration
DEFAULT_REWARD_CONFIG = {
    "battery_health": {
        "weight": 0.4,
        "soh_target": 0.8,
        "degradation_penalty": 1.0,
        "thermal_penalty": 0.5
    },
    "efficiency": {
        "weight": 0.3,
        "energy_efficiency_target": 0.95,
        "charging_efficiency_target": 0.92,
        "power_delivery_target": 0.90
    },
    "safety": {
        "weight": 0.3,
        "thermal_safety_margin": 10.0,  # °C
        "voltage_safety_margin": 0.1,   # V
        "current_safety_margin": 0.05   # A
    }
}

def get_default_reward_config():
    """
    Get default configuration for reward functions.
    
    Returns:
        dict: Default reward configuration
    """
    return DEFAULT_REWARD_CONFIG.copy()

def create_battery_health_reward(config=None):
    """
    Factory function to create a battery health reward.
    
    Args:
        config (dict, optional): Reward configuration
        
    Returns:
        BatteryHealthReward: Configured reward instance
    """
    if config is None:
        config = DEFAULT_REWARD_CONFIG["battery_health"]
    
    reward_config = HealthRewardConfig(**config)
    return BatteryHealthReward(reward_config)

def create_efficiency_reward(config=None):
    """
    Factory function to create an efficiency reward.
    
    Args:
        config (dict, optional): Reward configuration
        
    Returns:
        EfficiencyReward: Configured reward instance
    """
    if config is None:
        config = DEFAULT_REWARD_CONFIG["efficiency"]
    
    reward_config = EfficiencyRewardConfig(**config)
    return EfficiencyReward(reward_config)

def create_safety_reward(config=None):
    """
    Factory function to create a safety reward.
    
    Args:
        config (dict, optional): Reward configuration
        
    Returns:
        SafetyReward: Configured reward instance
    """
    if config is None:
        config = DEFAULT_REWARD_CONFIG["safety"]
    
    reward_config = SafetyRewardConfig(**config)
    return SafetyReward(reward_config)

def create_composite_reward(health_weight=0.4, efficiency_weight=0.3, safety_weight=0.3):
    """
    Factory function to create a composite reward combining multiple objectives.
    
    Args:
        health_weight (float): Weight for battery health reward
        efficiency_weight (float): Weight for efficiency reward
        safety_weight (float): Weight for safety reward
        
    Returns:
        CompositeReward: Configured composite reward
    """
    # Create individual rewards
    health_reward = create_battery_health_reward()
    efficiency_reward = create_efficiency_reward()
    safety_reward = create_safety_reward()
    
    # Create composite reward
    rewards = {
        'health': health_reward,
        'efficiency': efficiency_reward,
        'safety': safety_reward
    }
    
    weights = {
        'health': health_weight,
        'efficiency': efficiency_weight,
        'safety': safety_weight
    }
    
    config = CompositeRewardConfig(
        reward_weights=weights,
        normalization_method='min_max',
        adaptive_weighting=True
    )
    
    return CompositeReward(rewards, config)

# Reward utility functions
def normalize_reward(reward, min_val=-1.0, max_val=1.0):
    """
    Normalize reward to specified range.
    
    Args:
        reward (float): Raw reward value
        min_val (float): Minimum normalized value
        max_val (float): Maximum normalized value
        
    Returns:
        float: Normalized reward
    """
    return max(min_val, min(max_val, reward))

def clip_reward(reward, threshold=10.0):
    """
    Clip reward to prevent extreme values.
    
    Args:
        reward (float): Raw reward value
        threshold (float): Clipping threshold
        
    Returns:
        float: Clipped reward
    """
    return max(-threshold, min(threshold, reward))

def smooth_reward(reward, previous_reward, smoothing_factor=0.1):
    """
    Apply temporal smoothing to reward signal.
    
    Args:
        reward (float): Current reward
        previous_reward (float): Previous reward
        smoothing_factor (float): Smoothing factor (0-1)
        
    Returns:
        float: Smoothed reward
    """
    return (1 - smoothing_factor) * previous_reward + smoothing_factor * reward

# Reward validation utilities
def validate_reward_config(config):
    """
    Validate reward configuration for consistency.
    
    Args:
        config (dict): Reward configuration
        
    Returns:
        dict: Validation results
    """
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": []
    }
    
    # Check weight normalization
    total_weight = sum([
        config.get("battery_health", {}).get("weight", 0),
        config.get("efficiency", {}).get("weight", 0),
        config.get("safety", {}).get("weight", 0)
    ])
    
    if abs(total_weight - 1.0) > 0.01:
        validation_results["warnings"].append(
            f"Reward weights sum to {total_weight:.3f}, not 1.0"
        )
    
    # Check for negative weights
    for reward_type in ["battery_health", "efficiency", "safety"]:
        if reward_type in config:
            weight = config[reward_type].get("weight", 0)
            if weight < 0:
                validation_results["errors"].append(
                    f"Negative weight for {reward_type}: {weight}"
                )
                validation_results["valid"] = False
    
    return validation_results

# Performance monitoring for rewards
class RewardMonitor:
    """
    Monitor reward function performance and statistics.
    """
    
    def __init__(self):
        self.reward_history = []
        self.component_history = {}
        
    def log_reward(self, total_reward, component_rewards=None):
        """Log reward values for monitoring."""
        self.reward_history.append(total_reward)
        
        if component_rewards:
            for component, reward in component_rewards.items():
                if component not in self.component_history:
                    self.component_history[component] = []
                self.component_history[component].append(reward)
    
    def get_statistics(self):
        """Get reward statistics."""
        if not self.reward_history:
            return {}
        
        import numpy as np
        
        stats = {
            "total_rewards": {
                "mean": np.mean(self.reward_history),
                "std": np.std(self.reward_history),
                "min": np.min(self.reward_history),
                "max": np.max(self.reward_history),
                "count": len(self.reward_history)
            }
        }
        
        # Component statistics
        for component, history in self.component_history.items():
            stats[f"{component}_rewards"] = {
                "mean": np.mean(history),
                "std": np.std(history),
                "min": np.min(history),
                "max": np.max(history)
            }
        
        return stats

# Global reward monitor instance
reward_monitor = RewardMonitor()

def get_reward_monitor():
    """Get the global reward monitor instance."""
    return reward_monitor

# Module health check
def health_check():
    """
    Perform health check of reward functions.
    
    Returns:
        dict: Health check results
    """
    health_status = {
        "module_loaded": True,
        "version": __version__,
        "reward_functions_available": True,
        "configuration_valid": True
    }
    
    try:
        # Test reward creation
        health_reward = create_battery_health_reward()
        efficiency_reward = create_efficiency_reward()
        safety_reward = create_safety_reward()
        composite_reward = create_composite_reward()
        
        health_status["reward_creation"] = True
        
        # Test configuration validation
        config = get_default_reward_config()
        validation = validate_reward_config(config)
        health_status["configuration_valid"] = validation["valid"]
        
    except Exception as e:
        health_status["error"] = str(e)
        health_status["reward_functions_available"] = False
    
    return health_status

# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BatteryMind Reinforcement Learning Rewards v{__version__} initialized")
