"""
BatteryMind - Reinforcement Learning Environments

Comprehensive collection of battery simulation environments for training
reinforcement learning agents. Provides realistic physics-based simulations
for various battery management scenarios including charging optimization,
thermal management, and fleet coordination.

Key Components:
- BatteryEnvironment: Core battery physics simulation environment
- ChargingEnvironment: Specialized environment for charging protocol optimization
- FleetEnvironment: Multi-battery fleet management environment
- PhysicsSimulator: Advanced physics-based battery modeling
- Environment utilities and validation tools

Features:
- Realistic battery physics modeling with electrochemical constraints
- Multi-objective reward systems for complex optimization scenarios
- Scalable environments supporting single battery to fleet-level simulation
- Integration with OpenAI Gym interface for compatibility
- Comprehensive observation and action space definitions
- Safety constraints and failure mode simulation
- Real-time performance monitoring and logging

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

from .battery_env import (
    BatteryEnvironment,
    BatteryState,
    BatteryAction,
    BatteryObservation,
    BatteryReward,
    EnvironmentConfig,
    PhysicsConfig,
    SafetyConstraints,
    create_battery_environment
)

from .charging_env import (
    ChargingEnvironment,
    ChargingProtocol,
    ChargingState,
    ChargingAction,
    ChargingReward,
    create_charging_environment
)

from .fleet_env import (
    FleetEnvironment,
    FleetState,
    FleetAction,
    FleetReward,
    FleetCoordination,
    create_fleet_environment
)

from .physics_simulator import (
    PhysicsSimulator,
    ElectrochemicalModel,
    ThermalModel,
    DegradationModel,
    BatteryPhysics,
    create_physics_simulator
)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module metadata
__all__ = [
    # Core battery environment
    "BatteryEnvironment",
    "BatteryState", 
    "BatteryAction",
    "BatteryObservation",
    "BatteryReward",
    "EnvironmentConfig",
    "PhysicsConfig",
    "SafetyConstraints",
    "create_battery_environment",
    
    # Charging environment
    "ChargingEnvironment",
    "ChargingProtocol",
    "ChargingState",
    "ChargingAction", 
    "ChargingReward",
    "create_charging_environment",
    
    # Fleet environment
    "FleetEnvironment",
    "FleetState",
    "FleetAction",
    "FleetReward",
    "FleetCoordination",
    "create_fleet_environment",
    
    # Physics simulation
    "PhysicsSimulator",
    "ElectrochemicalModel",
    "ThermalModel", 
    "DegradationModel",
    "BatteryPhysics",
    "create_physics_simulator"
]

# Environment registry for managing different environment types
ENVIRONMENT_REGISTRY = {
    "battery": {
        "class": "BatteryEnvironment",
        "description": "Core battery physics simulation environment",
        "observation_space": "continuous",
        "action_space": "continuous", 
        "reward_type": "multi_objective",
        "physics_enabled": True,
        "safety_constraints": True,
        "use_cases": ["health_optimization", "charging_control", "thermal_management"]
    },
    "charging": {
        "class": "ChargingEnvironment", 
        "description": "Specialized charging protocol optimization environment",
        "observation_space": "continuous",
        "action_space": "discrete_continuous",
        "reward_type": "composite",
        "physics_enabled": True,
        "safety_constraints": True,
        "use_cases": ["charging_optimization", "fast_charging", "battery_longevity"]
    },
    "fleet": {
        "class": "FleetEnvironment",
        "description": "Multi-battery fleet management environment", 
        "observation_space": "multi_agent",
        "action_space": "multi_agent",
        "reward_type": "cooperative",
        "physics_enabled": True,
        "safety_constraints": True,
        "use_cases": ["fleet_optimization", "load_balancing", "coordinated_charging"]
    }
}

# Default configuration for all environments
DEFAULT_ENV_CONFIG = {
    "physics": {
        "enable_electrochemical_model": True,
        "enable_thermal_model": True,
        "enable_degradation_model": True,
        "simulation_timestep": 1.0,  # seconds
        "physics_accuracy": "high"
    },
    "safety": {
        "enable_safety_constraints": True,
        "temperature_limits": [-20.0, 60.0],  # Celsius
        "voltage_limits": [2.5, 4.2],  # Volts
        "current_limits": [-100.0, 100.0],  # Amperes
        "emergency_shutdown": True
    },
    "simulation": {
        "max_episode_steps": 1000,
        "random_seed": 42,
        "deterministic": False,
        "real_time_factor": 1.0
    },
    "logging": {
        "enable_logging": True,
        "log_level": "INFO",
        "log_frequency": 100,
        "save_trajectories": True
    }
}

def get_environment_registry():
    """
    Get registry of available environments.
    
    Returns:
        dict: Environment registry with metadata
    """
    return ENVIRONMENT_REGISTRY.copy()

def get_default_config():
    """
    Get default configuration for environments.
    
    Returns:
        dict: Default configuration dictionary
    """
    return DEFAULT_ENV_CONFIG.copy()

def create_environment(env_type: str, config: dict = None, **kwargs):
    """
    Factory function to create environments.
    
    Args:
        env_type (str): Type of environment to create
        config (dict, optional): Environment configuration
        **kwargs: Additional environment-specific arguments
        
    Returns:
        Environment instance
    """
    if config is None:
        config = get_default_config()
    
    if env_type == "battery":
        return create_battery_environment(config, **kwargs)
    elif env_type == "charging":
        return create_charging_environment(config, **kwargs)
    elif env_type == "fleet":
        return create_fleet_environment(config, **kwargs)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

def validate_environment_config(config: dict) -> dict:
    """
    Validate environment configuration.
    
    Args:
        config (dict): Configuration to validate
        
    Returns:
        dict: Validation results
    """
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": []
    }
    
    # Check required sections
    required_sections = ["physics", "safety", "simulation"]
    for section in required_sections:
        if section not in config:
            validation_results["errors"].append(f"Missing required section: {section}")
    
    # Validate physics configuration
    if "physics" in config:
        physics_config = config["physics"]
        if "simulation_timestep" in physics_config:
            timestep = physics_config["simulation_timestep"]
            if timestep <= 0 or timestep > 60:
                validation_results["warnings"].append("Simulation timestep should be between 0 and 60 seconds")
    
    # Validate safety constraints
    if "safety" in config:
        safety_config = config["safety"]
        if "temperature_limits" in safety_config:
            temp_limits = safety_config["temperature_limits"]
            if temp_limits[0] >= temp_limits[1]:
                validation_results["errors"].append("Invalid temperature limits")
        
        if "voltage_limits" in safety_config:
            voltage_limits = safety_config["voltage_limits"]
            if voltage_limits[0] >= voltage_limits[1]:
                validation_results["errors"].append("Invalid voltage limits")
    
    # Set validation status
    validation_results["valid"] = len(validation_results["errors"]) == 0
    
    return validation_results

def benchmark_environment_performance(env, num_episodes: int = 10) -> dict:
    """
    Benchmark environment performance.
    
    Args:
        env: Environment instance to benchmark
        num_episodes (int): Number of episodes to run
        
    Returns:
        dict: Performance benchmarking results
    """
    import time
    import numpy as np
    
    episode_times = []
    step_times = []
    total_steps = 0
    
    for episode in range(num_episodes):
        episode_start = time.time()
        obs = env.reset()
        done = False
        episode_step_count = 0
        
        while not done:
            step_start = time.time()
            
            # Take random action for benchmarking
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            step_end = time.time()
            step_times.append(step_end - step_start)
            episode_step_count += 1
            total_steps += 1
        
        episode_end = time.time()
        episode_times.append(episode_end - episode_start)
    
    return {
        "avg_episode_time": np.mean(episode_times),
        "avg_step_time": np.mean(step_times),
        "steps_per_second": 1.0 / np.mean(step_times),
        "total_steps": total_steps,
        "total_episodes": num_episodes,
        "environment_type": env.__class__.__name__
    }

def get_environment_info(env_type: str = None) -> dict:
    """
    Get comprehensive information about environments.
    
    Args:
        env_type (str, optional): Specific environment type
        
    Returns:
        dict: Environment information
    """
    if env_type:
        if env_type in ENVIRONMENT_REGISTRY:
            return ENVIRONMENT_REGISTRY[env_type]
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
    
    return {
        "available_environments": list(ENVIRONMENT_REGISTRY.keys()),
        "environment_details": ENVIRONMENT_REGISTRY,
        "default_config": DEFAULT_ENV_CONFIG,
        "version": __version__
    }

# Environment validation utilities
def validate_observation_space(obs_space) -> bool:
    """Validate observation space definition."""
    try:
        # Check if observation space has required attributes
        if not hasattr(obs_space, 'shape'):
            return False
        if not hasattr(obs_space, 'dtype'):
            return False
        return True
    except:
        return False

def validate_action_space(action_space) -> bool:
    """Validate action space definition."""
    try:
        # Check if action space has required attributes
        if not hasattr(action_space, 'shape'):
            return False
        if not hasattr(action_space, 'sample'):
            return False
        return True
    except:
        return False

# Module health check
def health_check() -> dict:
    """
    Perform health check of the environments module.
    
    Returns:
        dict: Health check results
    """
    health_status = {
        "module_loaded": True,
        "version": __version__,
        "environments_available": True,
        "dependencies_satisfied": True,
        "issues": []
    }
    
    try:
        # Test environment creation
        test_config = get_default_config()
        test_env = create_environment("battery", test_config)
        health_status["battery_env_creation"] = True
        
        # Test basic environment operations
        obs = test_env.reset()
        action = test_env.action_space.sample()
        obs, reward, done, info = test_env.step(action)
        health_status["basic_operations"] = True
        
    except Exception as e:
        health_status["environments_available"] = False
        health_status["issues"].append(f"Environment creation failed: {str(e)}")
    
    # Check dependencies
    try:
        import gym
        import numpy as np
        health_status["gym_available"] = True
    except ImportError as e:
        health_status["dependencies_satisfied"] = False
        health_status["issues"].append(f"Missing dependency: {str(e)}")
    
    return health_status

# Performance monitoring
class EnvironmentMonitor:
    """Monitor environment performance and statistics."""
    
    def __init__(self):
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        self.episode_lengths = []
    
    def on_episode_start(self):
        """Called at the start of each episode."""
        self.episode_count += 1
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
    
    def on_step(self, reward: float):
        """Called after each environment step."""
        self.step_count += 1
        self.current_episode_steps += 1
        self.current_episode_reward += reward
        self.total_reward += reward
    
    def on_episode_end(self):
        """Called at the end of each episode."""
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_steps)
    
    def get_statistics(self) -> dict:
        """Get comprehensive statistics."""
        import numpy as np
        
        return {
            "total_episodes": self.episode_count,
            "total_steps": self.step_count,
            "total_reward": self.total_reward,
            "avg_episode_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "avg_episode_length": np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            "max_episode_reward": np.max(self.episode_rewards) if self.episode_rewards else 0.0,
            "min_episode_reward": np.min(self.episode_rewards) if self.episode_rewards else 0.0
        }

# Global environment monitor instance
environment_monitor = EnvironmentMonitor()

def get_environment_monitor():
    """Get the global environment monitor instance."""
    return environment_monitor

# Logging setup
import logging
logger = logging.getLogger(__name__)
logger.info(f"BatteryMind RL Environments v{__version__} initialized")
logger.info(f"Available environments: {list(ENVIRONMENT_REGISTRY.keys())}")
