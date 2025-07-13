"""
BatteryMind - Reinforcement Learning Module

Advanced reinforcement learning framework for battery management optimization.
Provides intelligent agents for charging optimization, thermal management,
load balancing, and multi-agent fleet coordination.

Key Components:
- Agents: Specialized RL agents for different battery management tasks
- Environments: Realistic battery simulation environments with physics constraints
- Algorithms: State-of-the-art RL algorithms (PPO, DDPG, SAC, DQN)
- Rewards: Comprehensive reward functions for battery optimization
- Training: Advanced training infrastructure with experience replay

Features:
- Multi-agent reinforcement learning for fleet coordination
- Physics-based battery environment simulation
- Safety-constrained optimization with thermal and electrical limits
- Hierarchical reinforcement learning for complex decision making
- Transfer learning across different battery chemistries
- Real-time adaptation to changing operating conditions

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

from .agent import (
    ChargingAgent,
    ThermalAgent,
    LoadBalancingAgent,
    MultiAgentSystem,
    AgentConfig,
    AgentMetrics
)

from .environments import (
    BatteryEnvironment,
    ChargingEnvironment,
    FleetEnvironment,
    PhysicsSimulator,
    EnvironmentConfig,
    BatteryState,
    ChargingAction,
    EnvironmentMetrics
)

from .algorithms import (
    PPOAlgorithm,
    DDPGAlgorithm,
    SACAlgorithm,
    DQNAlgorithm,
    AlgorithmConfig,
    TrainingMetrics
)

from .rewards import (
    BatteryHealthReward,
    EfficiencyReward,
    SafetyReward,
    CompositeReward,
    RewardConfig,
    RewardMetrics
)

from .training import (
    RLTrainer,
    ExperienceBuffer,
    PolicyNetwork,
    ValueNetwork,
    TrainingConfig,
    TrainingProgress
)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module metadata
__all__ = [
    # Agents
    "ChargingAgent",
    "ThermalAgent", 
    "LoadBalancingAgent",
    "MultiAgentSystem",
    "AgentConfig",
    "AgentMetrics",
    
    # Environments
    "BatteryEnvironment",
    "ChargingEnvironment",
    "FleetEnvironment",
    "PhysicsSimulator",
    "EnvironmentConfig",
    "BatteryState",
    "ChargingAction",
    "EnvironmentMetrics",
    
    # Algorithms
    "PPOAlgorithm",
    "DDPGAlgorithm",
    "SACAlgorithm",
    "DQNAlgorithm",
    "AlgorithmConfig",
    "TrainingMetrics",
    
    # Rewards
    "BatteryHealthReward",
    "EfficiencyReward",
    "SafetyReward",
    "CompositeReward",
    "RewardConfig",
    "RewardMetrics",
    
    # Training
    "RLTrainer",
    "ExperienceBuffer",
    "PolicyNetwork",
    "ValueNetwork",
    "TrainingConfig",
    "TrainingProgress"
]

# Default configuration constants
DEFAULT_RL_CONFIG = {
    "training": {
        "algorithm": "PPO",
        "learning_rate": 3e-4,
        "batch_size": 64,
        "buffer_size": 100000,
        "gamma": 0.99,
        "tau": 0.005,
        "update_frequency": 1000,
        "max_episodes": 10000,
        "max_steps_per_episode": 1000
    },
    "environment": {
        "battery_capacity": 100.0,  # kWh
        "voltage_range": [3.0, 4.2],  # V
        "current_range": [-200, 200],  # A
        "temperature_range": [-20, 60],  # °C
        "soc_range": [0.1, 0.9],  # 10% to 90%
        "physics_timestep": 1.0,  # seconds
        "safety_constraints": True
    },
    "agent": {
        "observation_space_dim": 20,
        "action_space_dim": 5,
        "hidden_layers": [256, 256],
        "activation": "relu",
        "exploration_noise": 0.1,
        "target_update_frequency": 100
    },
    "rewards": {
        "battery_health_weight": 0.4,
        "efficiency_weight": 0.3,
        "safety_weight": 0.3,
        "penalty_multiplier": 10.0,
        "reward_scaling": 1.0
    }
}

def get_default_rl_config():
    """
    Get default configuration for reinforcement learning.
    
    Returns:
        dict: Default RL configuration
    """
    return DEFAULT_RL_CONFIG.copy()

def create_charging_agent(config=None):
    """
    Factory function to create a charging optimization agent.
    
    Args:
        config (dict, optional): Agent configuration
        
    Returns:
        ChargingAgent: Configured charging agent
    """
    if config is None:
        config = get_default_rl_config()
    
    agent_config = AgentConfig(**config["agent"])
    return ChargingAgent(agent_config)

def create_thermal_agent(config=None):
    """
    Factory function to create a thermal management agent.
    
    Args:
        config (dict, optional): Agent configuration
        
    Returns:
        ThermalAgent: Configured thermal agent
    """
    if config is None:
        config = get_default_rl_config()
    
    agent_config = AgentConfig(**config["agent"])
    return ThermalAgent(agent_config)

def create_multi_agent_system(num_agents=3, config=None):
    """
    Factory function to create a multi-agent system.
    
    Args:
        num_agents (int): Number of agents in the system
        config (dict, optional): System configuration
        
    Returns:
        MultiAgentSystem: Configured multi-agent system
    """
    if config is None:
        config = get_default_rl_config()
    
    return MultiAgentSystem(num_agents, config)

def create_battery_environment(config=None):
    """
    Factory function to create a battery environment.
    
    Args:
        config (dict, optional): Environment configuration
        
    Returns:
        BatteryEnvironment: Configured battery environment
    """
    if config is None:
        config = get_default_rl_config()
    
    env_config = EnvironmentConfig(**config["environment"])
    return BatteryEnvironment(env_config)

def create_rl_trainer(agent, environment, config=None):
    """
    Factory function to create an RL trainer.
    
    Args:
        agent: RL agent to train
        environment: Training environment
        config (dict, optional): Training configuration
        
    Returns:
        RLTrainer: Configured RL trainer
    """
    if config is None:
        config = get_default_rl_config()
    
    training_config = TrainingConfig(**config["training"])
    return RLTrainer(agent, environment, training_config)

# Reinforcement learning scenarios
RL_SCENARIOS = {
    "single_battery_charging": {
        "description": "Single battery charging optimization",
        "agent_type": "charging",
        "environment_type": "battery",
        "objective": "maximize_battery_life",
        "constraints": ["thermal_limits", "voltage_limits", "current_limits"]
    },
    "fleet_coordination": {
        "description": "Multi-battery fleet coordination",
        "agent_type": "multi_agent",
        "environment_type": "fleet",
        "objective": "optimize_fleet_performance",
        "constraints": ["load_balancing", "thermal_management", "safety"]
    },
    "thermal_management": {
        "description": "Battery thermal management optimization",
        "agent_type": "thermal",
        "environment_type": "battery",
        "objective": "maintain_optimal_temperature",
        "constraints": ["cooling_power_limits", "ambient_conditions"]
    },
    "load_balancing": {
        "description": "Dynamic load balancing across battery systems",
        "agent_type": "load_balancing",
        "environment_type": "fleet",
        "objective": "balance_load_distribution",
        "constraints": ["capacity_limits", "efficiency_requirements"]
    },
    "emergency_response": {
        "description": "Emergency response and safety management",
        "agent_type": "multi_agent",
        "environment_type": "battery",
        "objective": "ensure_safety",
        "constraints": ["critical_safety_limits", "emergency_protocols"]
    }
}

def get_rl_scenario(scenario_name):
    """
    Get predefined RL scenario configuration.
    
    Args:
        scenario_name (str): Name of the RL scenario
        
    Returns:
        dict: Scenario configuration
    """
    return RL_SCENARIOS.get(scenario_name, {})

def list_rl_scenarios():
    """
    List available RL scenarios.
    
    Returns:
        list: List of available scenario names
    """
    return list(RL_SCENARIOS.keys())

# Battery optimization objectives
OPTIMIZATION_OBJECTIVES = {
    "battery_life_extension": {
        "primary_metric": "cycle_life",
        "reward_components": ["health_preservation", "degradation_minimization"],
        "optimization_horizon": "long_term",
        "safety_priority": "high"
    },
    "energy_efficiency": {
        "primary_metric": "round_trip_efficiency",
        "reward_components": ["charging_efficiency", "discharging_efficiency"],
        "optimization_horizon": "medium_term",
        "safety_priority": "medium"
    },
    "fast_charging": {
        "primary_metric": "charging_time",
        "reward_components": ["charging_speed", "temperature_control"],
        "optimization_horizon": "short_term",
        "safety_priority": "high"
    },
    "cost_optimization": {
        "primary_metric": "operational_cost",
        "reward_components": ["energy_cost", "maintenance_cost", "replacement_cost"],
        "optimization_horizon": "long_term",
        "safety_priority": "medium"
    },
    "performance_maximization": {
        "primary_metric": "power_output",
        "reward_components": ["peak_power", "sustained_power", "response_time"],
        "optimization_horizon": "short_term",
        "safety_priority": "high"
    }
}

def get_optimization_objective(objective_name):
    """
    Get optimization objective configuration.
    
    Args:
        objective_name (str): Name of the optimization objective
        
    Returns:
        dict: Objective configuration
    """
    return OPTIMIZATION_OBJECTIVES.get(objective_name, {})

def list_optimization_objectives():
    """
    List available optimization objectives.
    
    Returns:
        list: List of available objective names
    """
    return list(OPTIMIZATION_OBJECTIVES.keys())

# RL algorithm configurations
ALGORITHM_CONFIGS = {
    "PPO": {
        "type": "policy_gradient",
        "on_policy": True,
        "continuous_actions": True,
        "discrete_actions": True,
        "advantages": ["stable_training", "sample_efficient", "robust"],
        "best_for": ["continuous_control", "multi_agent", "safety_critical"]
    },
    "DDPG": {
        "type": "actor_critic",
        "on_policy": False,
        "continuous_actions": True,
        "discrete_actions": False,
        "advantages": ["deterministic_policy", "off_policy", "sample_efficient"],
        "best_for": ["continuous_control", "robotics", "precise_control"]
    },
    "SAC": {
        "type": "actor_critic",
        "on_policy": False,
        "continuous_actions": True,
        "discrete_actions": False,
        "advantages": ["maximum_entropy", "exploration", "robust"],
        "best_for": ["exploration_heavy", "stochastic_environments", "robustness"]
    },
    "DQN": {
        "type": "value_based",
        "on_policy": False,
        "continuous_actions": False,
        "discrete_actions": True,
        "advantages": ["experience_replay", "target_network", "stable"],
        "best_for": ["discrete_actions", "atari_games", "simple_environments"]
    }
}

def get_algorithm_info(algorithm_name):
    """
    Get information about an RL algorithm.
    
    Args:
        algorithm_name (str): Name of the algorithm
        
    Returns:
        dict: Algorithm information
    """
    return ALGORITHM_CONFIGS.get(algorithm_name, {})

def list_rl_algorithms():
    """
    List available RL algorithms.
    
    Returns:
        list: List of available algorithm names
    """
    return list(ALGORITHM_CONFIGS.keys())

# Integration with other BatteryMind modules
def create_integrated_rl_system(transformer_predictor=None, federated_client=None):
    """
    Create integrated RL system with other BatteryMind components.
    
    Args:
        transformer_predictor: Battery health predictor from transformers module
        federated_client: Federated learning client
        
    Returns:
        dict: Integrated system configuration
    """
    config = get_default_rl_config()
    
    # Enhanced configuration for integration
    config["integration"] = {
        "transformer_integration": transformer_predictor is not None,
        "federated_learning": federated_client is not None,
        "multi_modal_input": True,
        "cross_module_optimization": True
    }
    
    if transformer_predictor:
        config["integration"]["health_prediction"] = {
            "enabled": True,
            "prediction_horizon": 24,  # hours
            "update_frequency": 3600   # seconds
        }
    
    if federated_client:
        config["integration"]["federated_learning"] = {
            "enabled": True,
            "model_sharing": True,
            "privacy_preserving": True,
            "aggregation_frequency": 86400  # daily
        }
    
    return config

# Validation and testing utilities
def validate_rl_config(config):
    """
    Validate RL configuration parameters.
    
    Args:
        config (dict): RL configuration
        
    Returns:
        dict: Validation results
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check required sections
    required_sections = ["training", "environment", "agent", "rewards"]
    for section in required_sections:
        if section not in config:
            validation_results["errors"].append(f"Missing required section: {section}")
            validation_results["valid"] = False
    
    # Validate training parameters
    if "training" in config:
        training = config["training"]
        if training.get("learning_rate", 0) <= 0:
            validation_results["errors"].append("Learning rate must be positive")
            validation_results["valid"] = False
        
        if training.get("batch_size", 0) <= 0:
            validation_results["errors"].append("Batch size must be positive")
            validation_results["valid"] = False
    
    # Validate environment parameters
    if "environment" in config:
        env = config["environment"]
        if env.get("battery_capacity", 0) <= 0:
            validation_results["errors"].append("Battery capacity must be positive")
            validation_results["valid"] = False
    
    return validation_results

def estimate_training_resources(config):
    """
    Estimate computational resources required for RL training.
    
    Args:
        config (dict): RL configuration
        
    Returns:
        dict: Resource estimates
    """
    max_episodes = config.get("training", {}).get("max_episodes", 10000)
    max_steps = config.get("training", {}).get("max_steps_per_episode", 1000)
    batch_size = config.get("training", {}).get("batch_size", 64)
    
    # Rough estimates based on typical RL workloads
    estimated_resources = {
        "total_training_time_hours": (max_episodes * max_steps * 0.001) / 3600,
        "memory_requirement_gb": batch_size * 0.01,
        "storage_requirement_gb": max_episodes * 0.1,
        "gpu_utilization": "recommended",
        "computational_complexity": "O(episodes * steps * batch_size)"
    }
    
    return estimated_resources

# Module health check
def health_check():
    """
    Perform a health check of the reinforcement learning module.
    
    Returns:
        dict: Health check results
    """
    health_status = {
        "module_loaded": True,
        "version": __version__,
        "components_available": {
            "agents": True,
            "environments": True,
            "algorithms": True,
            "rewards": True,
            "training": True
        },
        "dependencies_satisfied": True
    }
    
    try:
        # Test basic functionality
        config = get_default_rl_config()
        validation_results = validate_rl_config(config)
        health_status["config_validation"] = validation_results["valid"]
    except Exception as e:
        health_status["config_validation"] = False
        health_status["error"] = str(e)
    
    return health_status

# Export configuration template
def export_rl_config_template(file_path="rl_config.yaml"):
    """
    Export an RL configuration template.
    
    Args:
        file_path (str): Path to save the configuration template
    """
    import yaml
    
    config_template = get_default_rl_config()
    config_template["_metadata"] = {
        "version": __version__,
        "description": "BatteryMind Reinforcement Learning Configuration Template",
        "author": __author__,
        "created": "2025-07-11"
    }
    
    with open(file_path, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False, indent=2)

# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BatteryMind Reinforcement Learning Module v{__version__} initialized")
