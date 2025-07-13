"""
BatteryMind - Reinforcement Learning Agents

Advanced reinforcement learning agents for autonomous battery management
and optimization with multi-agent coordination capabilities.

This module implements state-of-the-art RL algorithms specifically designed
for battery management scenarios including charging optimization, thermal
control, load balancing, and fleet coordination.

Key Components:
- ChargingAgent: Autonomous charging protocol optimization
- ThermalAgent: Intelligent thermal management and control
- LoadBalancingAgent: Dynamic load distribution optimization
- MultiAgentSystem: Coordinated multi-agent battery management

Features:
- PPO, DDPG, SAC, and DQN algorithm implementations
- Physics-informed reward functions for battery safety
- Hierarchical reinforcement learning for complex scenarios
- Multi-agent coordination with communication protocols
- Real-time adaptation to changing battery conditions
- Safety-constrained optimization with hard limits

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

from .charging_agent import (
    ChargingAgent,
    ChargingAgentConfig,
    ChargingPolicy,
    ChargingReward,
    ChargingEnvironment
)

from .thermal_agent import (
    ThermalAgent,
    ThermalAgentConfig,
    ThermalPolicy,
    ThermalReward,
    ThermalEnvironment
)

from .load_balancing_agent import (
    LoadBalancingAgent,
    LoadBalancingConfig,
    LoadBalancingPolicy,
    LoadBalancingReward,
    LoadBalancingEnvironment
)

from .multi_agent_system import (
    MultiAgentSystem,
    MultiAgentConfig,
    AgentCoordinator,
    CommunicationProtocol,
    ConsensusAlgorithm
)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module metadata
__all__ = [
    # Charging agent components
    "ChargingAgent",
    "ChargingAgentConfig",
    "ChargingPolicy",
    "ChargingReward",
    "ChargingEnvironment",
    
    # Thermal agent components
    "ThermalAgent",
    "ThermalAgentConfig", 
    "ThermalPolicy",
    "ThermalReward",
    "ThermalEnvironment",
    
    # Load balancing agent components
    "LoadBalancingAgent",
    "LoadBalancingConfig",
    "LoadBalancingPolicy", 
    "LoadBalancingReward",
    "LoadBalancingEnvironment",
    
    # Multi-agent system components
    "MultiAgentSystem",
    "MultiAgentConfig",
    "AgentCoordinator",
    "CommunicationProtocol",
    "ConsensusAlgorithm"
]

# Default RL agent configurations
DEFAULT_AGENT_CONFIGS = {
    "charging_agent": {
        "algorithm": "ppo",
        "learning_rate": 3e-4,
        "batch_size": 64,
        "buffer_size": 100000,
        "gamma": 0.99,
        "tau": 0.005,
        "policy_frequency": 2,
        "noise_clip": 0.5,
        "policy_noise": 0.2,
        "exploration_noise": 0.1
    },
    "thermal_agent": {
        "algorithm": "sac",
        "learning_rate": 3e-4,
        "batch_size": 256,
        "buffer_size": 1000000,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.2,
        "automatic_entropy_tuning": True,
        "target_update_interval": 1
    },
    "load_balancing_agent": {
        "algorithm": "ddpg",
        "learning_rate": 1e-4,
        "batch_size": 128,
        "buffer_size": 1000000,
        "gamma": 0.99,
        "tau": 0.001,
        "noise_type": "ou_noise",
        "noise_std": 0.2
    },
    "multi_agent_system": {
        "coordination_algorithm": "consensus",
        "communication_frequency": 10,
        "consensus_threshold": 0.8,
        "max_agents": 50,
        "synchronization_method": "async"
    }
}

def get_default_agent_config(agent_type):
    """
    Get default configuration for a specific agent type.
    
    Args:
        agent_type (str): Type of agent ('charging', 'thermal', 'load_balancing', 'multi_agent')
        
    Returns:
        dict: Default configuration for the agent type
    """
    config_key = f"{agent_type}_agent" if agent_type != "multi_agent" else "multi_agent_system"
    return DEFAULT_AGENT_CONFIGS.get(config_key, {}).copy()

def create_charging_agent(config=None):
    """
    Factory function to create a charging optimization agent.
    
    Args:
        config (dict, optional): Agent configuration. If None, uses default.
        
    Returns:
        ChargingAgent: Configured charging agent instance
    """
    if config is None:
        config = get_default_agent_config("charging")
    
    agent_config = ChargingAgentConfig(**config)
    return ChargingAgent(agent_config)

def create_thermal_agent(config=None):
    """
    Factory function to create a thermal management agent.
    
    Args:
        config (dict, optional): Agent configuration. If None, uses default.
        
    Returns:
        ThermalAgent: Configured thermal agent instance
    """
    if config is None:
        config = get_default_agent_config("thermal")
    
    agent_config = ThermalAgentConfig(**config)
    return ThermalAgent(agent_config)

def create_load_balancing_agent(config=None):
    """
    Factory function to create a load balancing agent.
    
    Args:
        config (dict, optional): Agent configuration. If None, uses default.
        
    Returns:
        LoadBalancingAgent: Configured load balancing agent instance
    """
    if config is None:
        config = get_default_agent_config("load_balancing")
    
    agent_config = LoadBalancingConfig(**config)
    return LoadBalancingAgent(agent_config)

def create_multi_agent_system(agents=None, config=None):
    """
    Factory function to create a multi-agent system.
    
    Args:
        agents (list, optional): List of agents to include in the system
        config (dict, optional): System configuration. If None, uses default.
        
    Returns:
        MultiAgentSystem: Configured multi-agent system instance
    """
    if config is None:
        config = get_default_agent_config("multi_agent")
    
    system_config = MultiAgentConfig(**config)
    return MultiAgentSystem(agents or [], system_config)

# RL algorithms supported
RL_ALGORITHMS = {
    "ppo": {
        "name": "Proximal Policy Optimization",
        "type": "policy_gradient",
        "continuous_actions": True,
        "discrete_actions": True,
        "sample_efficiency": "medium",
        "stability": "high"
    },
    "sac": {
        "name": "Soft Actor-Critic", 
        "type": "actor_critic",
        "continuous_actions": True,
        "discrete_actions": False,
        "sample_efficiency": "high",
        "stability": "high"
    },
    "ddpg": {
        "name": "Deep Deterministic Policy Gradient",
        "type": "actor_critic",
        "continuous_actions": True,
        "discrete_actions": False,
        "sample_efficiency": "medium",
        "stability": "medium"
    },
    "dqn": {
        "name": "Deep Q-Network",
        "type": "value_based",
        "continuous_actions": False,
        "discrete_actions": True,
        "sample_efficiency": "low",
        "stability": "medium"
    },
    "td3": {
        "name": "Twin Delayed Deep Deterministic Policy Gradient",
        "type": "actor_critic",
        "continuous_actions": True,
        "discrete_actions": False,
        "sample_efficiency": "high",
        "stability": "high"
    }
}

def get_rl_algorithms():
    """
    Get available reinforcement learning algorithms.
    
    Returns:
        dict: Dictionary of available RL algorithms and their properties
    """
    return RL_ALGORITHMS.copy()

# Battery-specific RL scenarios
BATTERY_RL_SCENARIOS = {
    "fast_charging_optimization": {
        "description": "Optimize fast charging protocols for battery longevity",
        "agent_type": "charging_agent",
        "algorithm": "ppo",
        "action_space": "continuous",
        "state_space_dim": 15,
        "action_space_dim": 3,
        "reward_components": ["charging_speed", "battery_health", "safety"]
    },
    "thermal_management": {
        "description": "Intelligent thermal control for battery safety",
        "agent_type": "thermal_agent", 
        "algorithm": "sac",
        "action_space": "continuous",
        "state_space_dim": 12,
        "action_space_dim": 4,
        "reward_components": ["temperature_control", "energy_efficiency", "safety"]
    },
    "fleet_load_balancing": {
        "description": "Dynamic load balancing across battery fleet",
        "agent_type": "load_balancing_agent",
        "algorithm": "ddpg",
        "action_space": "continuous", 
        "state_space_dim": 20,
        "action_space_dim": 10,
        "reward_components": ["load_distribution", "efficiency", "fairness"]
    },
    "autonomous_charging_station": {
        "description": "Multi-agent coordination for charging station management",
        "agent_type": "multi_agent_system",
        "algorithm": "ppo",
        "action_space": "mixed",
        "num_agents": 5,
        "coordination_type": "centralized_training_decentralized_execution"
    },
    "battery_lifecycle_optimization": {
        "description": "Long-term battery lifecycle optimization",
        "agent_type": "charging_agent",
        "algorithm": "sac",
        "action_space": "continuous",
        "state_space_dim": 25,
        "action_space_dim": 5,
        "reward_components": ["longevity", "performance", "cost", "sustainability"]
    }
}

def get_battery_rl_scenarios():
    """
    Get predefined battery-specific RL scenarios.
    
    Returns:
        dict: Dictionary of battery RL scenarios
    """
    return BATTERY_RL_SCENARIOS.copy()

def create_battery_rl_system(scenario_name):
    """
    Create a complete RL system for battery applications.
    
    Args:
        scenario_name (str): Name of the battery scenario
        
    Returns:
        dict: Complete RL system with agent and environment
    """
    if scenario_name not in BATTERY_RL_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    scenario = BATTERY_RL_SCENARIOS[scenario_name]
    agent_type = scenario["agent_type"]
    
    # Create agent configuration
    agent_config = get_default_agent_config(agent_type.replace("_agent", ""))
    agent_config["algorithm"] = scenario["algorithm"]
    
    # Create appropriate agent
    if agent_type == "charging_agent":
        agent = create_charging_agent(agent_config)
    elif agent_type == "thermal_agent":
        agent = create_thermal_agent(agent_config)
    elif agent_type == "load_balancing_agent":
        agent = create_load_balancing_agent(agent_config)
    elif agent_type == "multi_agent_system":
        # Create multiple agents for multi-agent scenario
        agents = []
        for i in range(scenario.get("num_agents", 3)):
            agent = create_charging_agent(agent_config)
            agents.append(agent)
        agent = create_multi_agent_system(agents, agent_config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return {
        "agent": agent,
        "scenario": scenario,
        "config": agent_config
    }

# Reward function components
REWARD_COMPONENTS = {
    "battery_health": {
        "description": "Reward for maintaining/improving battery health",
        "weight_range": (0.2, 0.5),
        "optimization_goal": "maximize"
    },
    "energy_efficiency": {
        "description": "Reward for energy-efficient operations",
        "weight_range": (0.1, 0.3),
        "optimization_goal": "maximize"
    },
    "safety": {
        "description": "Penalty for unsafe operations",
        "weight_range": (0.3, 0.6),
        "optimization_goal": "maximize"
    },
    "charging_speed": {
        "description": "Reward for faster charging when safe",
        "weight_range": (0.1, 0.2),
        "optimization_goal": "maximize"
    },
    "temperature_control": {
        "description": "Reward for maintaining optimal temperature",
        "weight_range": (0.2, 0.4),
        "optimization_goal": "maximize"
    },
    "load_distribution": {
        "description": "Reward for balanced load distribution",
        "weight_range": (0.2, 0.4),
        "optimization_goal": "maximize"
    },
    "cost_efficiency": {
        "description": "Reward for cost-effective operations",
        "weight_range": (0.1, 0.2),
        "optimization_goal": "maximize"
    }
}

def get_reward_components():
    """
    Get available reward function components.
    
    Returns:
        dict: Dictionary of reward components
    """
    return REWARD_COMPONENTS.copy()

# Agent performance metrics
AGENT_METRICS = {
    "learning_metrics": [
        "episode_reward",
        "episode_length", 
        "learning_rate",
        "policy_loss",
        "value_loss",
        "entropy"
    ],
    "battery_metrics": [
        "battery_health_improvement",
        "energy_efficiency_gain",
        "safety_violations",
        "temperature_stability",
        "charging_time_reduction"
    ],
    "system_metrics": [
        "inference_time",
        "memory_usage",
        "cpu_utilization",
        "convergence_time",
        "sample_efficiency"
    ],
    "multi_agent_metrics": [
        "coordination_efficiency",
        "communication_overhead",
        "consensus_time",
        "fairness_index",
        "scalability_factor"
    ]
}

def get_agent_metrics():
    """
    Get comprehensive agent performance metrics.
    
    Returns:
        dict: Dictionary of agent metrics
    """
    return AGENT_METRICS.copy()

# Integration with other BatteryMind modules
def integrate_with_battery_models(rl_system, battery_model_paths):
    """
    Integrate RL system with existing battery models.
    
    Args:
        rl_system (dict): RL system
        battery_model_paths (dict): Paths to battery model artifacts
        
    Returns:
        dict: Integrated system with battery models
    """
    from ...transformers.battery_health_predictor import create_battery_predictor
    from ...transformers.degradation_forecaster import create_battery_degradation_forecaster
    
    integrated_system = rl_system.copy()
    
    # Load battery models for RL environment
    if "health_predictor" in battery_model_paths:
        health_model = create_battery_predictor(battery_model_paths["health_predictor"])
        integrated_system["environment_models"] = integrated_system.get("environment_models", {})
        integrated_system["environment_models"]["health_predictor"] = health_model
    
    if "degradation_forecaster" in battery_model_paths:
        forecaster_model = create_battery_degradation_forecaster(battery_model_paths["degradation_forecaster"])
        integrated_system["environment_models"] = integrated_system.get("environment_models", {})
        integrated_system["environment_models"]["degradation_forecaster"] = forecaster_model
    
    return integrated_system

# Module health check
def health_check():
    """
    Perform health check of the RL agents module.
    
    Returns:
        dict: Health check results
    """
    health_status = {
        "module_loaded": True,
        "version": __version__,
        "agents_available": True,
        "algorithms_supported": True
    }
    
    try:
        # Test agent imports
        from .charging_agent import ChargingAgent
        from .thermal_agent import ThermalAgent
        from .load_balancing_agent import LoadBalancingAgent
        from .multi_agent_system import MultiAgentSystem
        
        health_status["charging_agent_available"] = True
        health_status["thermal_agent_available"] = True
        health_status["load_balancing_agent_available"] = True
        health_status["multi_agent_system_available"] = True
        
        # Test factory functions
        test_charging_agent = create_charging_agent()
        test_thermal_agent = create_thermal_agent()
        test_load_balancing_agent = create_load_balancing_agent()
        
        health_status["factory_functions"] = True
        
    except Exception as e:
        health_status["agents_available"] = False
        health_status["error"] = str(e)
    
    return health_status

# Export configuration template
def export_agent_config(agent_type, file_path=None):
    """
    Export agent configuration template.
    
    Args:
        agent_type (str): Type of agent
        file_path (str, optional): Path to save configuration
    """
    import yaml
    
    if file_path is None:
        file_path = f"{agent_type}_agent_config.yaml"
    
    config_template = get_default_agent_config(agent_type)
    config_template["_metadata"] = {
        "version": __version__,
        "description": f"BatteryMind {agent_type.title()} Agent Configuration",
        "author": __author__
    }
    
    with open(file_path, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False, indent=2)

# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BatteryMind RL Agents v{__version__} initialized")

# Constants for RL agents
RL_CONSTANTS = {
    "MAX_EPISODE_LENGTH": 1000,
    "MIN_BUFFER_SIZE": 1000,
    "MAX_BUFFER_SIZE": 1000000,
    "DEFAULT_GAMMA": 0.99,
    "DEFAULT_TAU": 0.005,
    "MIN_LEARNING_RATE": 1e-6,
    "MAX_LEARNING_RATE": 1e-2,
    "SAFETY_CONSTRAINT_PENALTY": -1000,
    "MAX_TEMPERATURE": 60.0,  # Celsius
    "MIN_TEMPERATURE": -20.0,  # Celsius
    "MAX_VOLTAGE": 4.2,  # Volts
    "MIN_VOLTAGE": 2.5   # Volts
}

def get_rl_constants():
    """
    Get RL agent constants.
    
    Returns:
        dict: Dictionary of RL constants
    """
    return RL_CONSTANTS.copy()
