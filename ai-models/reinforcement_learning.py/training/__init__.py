"""
BatteryMind - Reinforcement Learning Training Module

Comprehensive training infrastructure for reinforcement learning agents in
battery management systems. Provides unified interfaces, training utilities,
and integration components for all RL algorithms and environments.

Key Components:
- RLTrainer: Main training orchestrator for RL agents
- ExperienceBuffer: Experience replay and data management
- PolicyNetwork: Policy network architectures and utilities
- ValueNetwork: Value function networks and critics
- TrainingMetrics: Performance tracking and analytics
- HyperparameterOptimizer: Automated hyperparameter tuning

Features:
- Multi-algorithm support (PPO, DDPG, SAC, DQN)
- Distributed training capabilities
- Advanced experience replay mechanisms
- Curriculum learning for complex battery scenarios
- Integration with battery environments and reward systems
- Comprehensive monitoring and visualization

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

from .rl_trainer import (
    RLTrainer,
    TrainingConfig,
    TrainingMetrics,
    TrainingCallback,
    MultiAgentTrainer,
    DistributedTrainer
)

from .experience_buffer import (
    ExperienceBuffer,
    PrioritizedExperienceBuffer,
    HindsightExperienceBuffer,
    DistributedExperienceBuffer,
    Experience,
    BufferConfig
)

from .policy_network import (
    PolicyNetwork,
    ActorNetwork,
    ContinuousPolicyNetwork,
    DiscretePolicyNetwork,
    MultiModalPolicyNetwork,
    PolicyConfig
)

from .value_network import (
    ValueNetwork,
    CriticNetwork,
    QNetwork,
    VNetwork,
    AdvantageNetwork,
    ValueConfig
)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module metadata
__all__ = [
    # Main training components
    "RLTrainer",
    "TrainingConfig", 
    "TrainingMetrics",
    "TrainingCallback",
    "MultiAgentTrainer",
    "DistributedTrainer",
    
    # Experience management
    "ExperienceBuffer",
    "PrioritizedExperienceBuffer", 
    "HindsightExperienceBuffer",
    "DistributedExperienceBuffer",
    "Experience",
    "BufferConfig",
    
    # Policy networks
    "PolicyNetwork",
    "ActorNetwork",
    "ContinuousPolicyNetwork",
    "DiscretePolicyNetwork", 
    "MultiModalPolicyNetwork",
    "PolicyConfig",
    
    # Value networks
    "ValueNetwork",
    "CriticNetwork",
    "QNetwork",
    "VNetwork", 
    "AdvantageNetwork",
    "ValueConfig",
    
    # Utility functions
    "create_trainer",
    "create_experience_buffer",
    "create_policy_network",
    "create_value_network",
    "get_training_config_template",
    "validate_training_setup"
]

# Training configuration constants
TRAINING_CONSTANTS = {
    # Default hyperparameters
    "DEFAULT_LEARNING_RATE": 3e-4,
    "DEFAULT_BATCH_SIZE": 64,
    "DEFAULT_BUFFER_SIZE": 1000000,
    "DEFAULT_GAMMA": 0.99,
    "DEFAULT_TAU": 0.005,
    
    # Training parameters
    "MIN_EPISODES_BEFORE_TRAINING": 10,
    "DEFAULT_UPDATE_FREQUENCY": 4,
    "DEFAULT_TARGET_UPDATE_FREQUENCY": 100,
    "DEFAULT_GRADIENT_STEPS": 1,
    
    # Network architecture
    "DEFAULT_HIDDEN_SIZE": 256,
    "DEFAULT_NUM_LAYERS": 2,
    "DEFAULT_ACTIVATION": "relu",
    "DEFAULT_DROPOUT": 0.0,
    
    # Experience replay
    "DEFAULT_PRIORITY_ALPHA": 0.6,
    "DEFAULT_PRIORITY_BETA": 0.4,
    "DEFAULT_PRIORITY_EPS": 1e-6,
    
    # Distributed training
    "DEFAULT_NUM_WORKERS": 4,
    "DEFAULT_SYNC_FREQUENCY": 100,
    "DEFAULT_COMMUNICATION_BACKEND": "nccl"
}

def get_training_constants():
    """
    Get training constants dictionary.
    
    Returns:
        dict: Dictionary of training constants
    """
    return TRAINING_CONSTANTS.copy()

def create_trainer(algorithm: str, env_config: dict, training_config: dict = None):
    """
    Factory function to create RL trainer for specified algorithm.
    
    Args:
        algorithm (str): RL algorithm ('ppo', 'ddpg', 'sac', 'dqn')
        env_config (dict): Environment configuration
        training_config (dict, optional): Training configuration
        
    Returns:
        RLTrainer: Configured trainer instance
    """
    from ..algorithms import create_algorithm
    
    # Create algorithm instance
    algo_instance = create_algorithm(algorithm, env_config)
    
    # Create training configuration
    if training_config is None:
        training_config = get_default_training_config(algorithm)
    
    config = TrainingConfig(**training_config)
    
    # Create trainer
    return RLTrainer(
        algorithm=algo_instance,
        config=config
    )

def create_experience_buffer(buffer_type: str = "standard", **kwargs):
    """
    Factory function to create experience buffer.
    
    Args:
        buffer_type (str): Type of buffer ('standard', 'prioritized', 'hindsight')
        **kwargs: Buffer configuration parameters
        
    Returns:
        ExperienceBuffer: Configured buffer instance
    """
    buffer_config = BufferConfig(**kwargs)
    
    if buffer_type == "standard":
        return ExperienceBuffer(buffer_config)
    elif buffer_type == "prioritized":
        return PrioritizedExperienceBuffer(buffer_config)
    elif buffer_type == "hindsight":
        return HindsightExperienceBuffer(buffer_config)
    elif buffer_type == "distributed":
        return DistributedExperienceBuffer(buffer_config)
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")

def create_policy_network(network_type: str, input_dim: int, output_dim: int, **kwargs):
    """
    Factory function to create policy network.
    
    Args:
        network_type (str): Type of network ('continuous', 'discrete', 'multimodal')
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        **kwargs: Network configuration parameters
        
    Returns:
        PolicyNetwork: Configured policy network
    """
    config = PolicyConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        **kwargs
    )
    
    if network_type == "continuous":
        return ContinuousPolicyNetwork(config)
    elif network_type == "discrete":
        return DiscretePolicyNetwork(config)
    elif network_type == "multimodal":
        return MultiModalPolicyNetwork(config)
    else:
        return PolicyNetwork(config)

def create_value_network(network_type: str, input_dim: int, **kwargs):
    """
    Factory function to create value network.
    
    Args:
        network_type (str): Type of network ('q', 'v', 'critic', 'advantage')
        input_dim (int): Input dimension
        **kwargs: Network configuration parameters
        
    Returns:
        ValueNetwork: Configured value network
    """
    config = ValueConfig(
        input_dim=input_dim,
        **kwargs
    )
    
    if network_type == "q":
        return QNetwork(config)
    elif network_type == "v":
        return VNetwork(config)
    elif network_type == "critic":
        return CriticNetwork(config)
    elif network_type == "advantage":
        return AdvantageNetwork(config)
    else:
        return ValueNetwork(config)

def get_default_training_config(algorithm: str) -> dict:
    """
    Get default training configuration for specified algorithm.
    
    Args:
        algorithm (str): RL algorithm name
        
    Returns:
        dict: Default training configuration
    """
    base_config = {
        "learning_rate": TRAINING_CONSTANTS["DEFAULT_LEARNING_RATE"],
        "batch_size": TRAINING_CONSTANTS["DEFAULT_BATCH_SIZE"],
        "buffer_size": TRAINING_CONSTANTS["DEFAULT_BUFFER_SIZE"],
        "gamma": TRAINING_CONSTANTS["DEFAULT_GAMMA"],
        "tau": TRAINING_CONSTANTS["DEFAULT_TAU"],
        "update_frequency": TRAINING_CONSTANTS["DEFAULT_UPDATE_FREQUENCY"],
        "gradient_steps": TRAINING_CONSTANTS["DEFAULT_GRADIENT_STEPS"]
    }
    
    # Algorithm-specific configurations
    if algorithm.lower() == "ppo":
        base_config.update({
            "clip_ratio": 0.2,
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
            "num_epochs": 4,
            "gae_lambda": 0.95
        })
    
    elif algorithm.lower() == "ddpg":
        base_config.update({
            "noise_std": 0.1,
            "noise_clip": 0.5,
            "policy_delay": 2,
            "target_noise": 0.2
        })
    
    elif algorithm.lower() == "sac":
        base_config.update({
            "alpha": 0.2,
            "automatic_entropy_tuning": True,
            "target_entropy": None
        })
    
    elif algorithm.lower() == "dqn":
        base_config.update({
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
            "double_dqn": True,
            "dueling_dqn": True
        })
    
    return base_config

def get_training_config_template() -> dict:
    """
    Get comprehensive training configuration template.
    
    Returns:
        dict: Training configuration template
    """
    return {
        "training": {
            "algorithm": "ppo",
            "total_timesteps": 1000000,
            "learning_rate": TRAINING_CONSTANTS["DEFAULT_LEARNING_RATE"],
            "batch_size": TRAINING_CONSTANTS["DEFAULT_BATCH_SIZE"],
            "buffer_size": TRAINING_CONSTANTS["DEFAULT_BUFFER_SIZE"],
            "gamma": TRAINING_CONSTANTS["DEFAULT_GAMMA"],
            "tau": TRAINING_CONSTANTS["DEFAULT_TAU"],
            "update_frequency": TRAINING_CONSTANTS["DEFAULT_UPDATE_FREQUENCY"],
            "gradient_steps": TRAINING_CONSTANTS["DEFAULT_GRADIENT_STEPS"],
            "max_grad_norm": 0.5,
            "seed": 42
        },
        "network": {
            "policy_hidden_size": TRAINING_CONSTANTS["DEFAULT_HIDDEN_SIZE"],
            "value_hidden_size": TRAINING_CONSTANTS["DEFAULT_HIDDEN_SIZE"],
            "num_layers": TRAINING_CONSTANTS["DEFAULT_NUM_LAYERS"],
            "activation": TRAINING_CONSTANTS["DEFAULT_ACTIVATION"],
            "dropout": TRAINING_CONSTANTS["DEFAULT_DROPOUT"]
        },
        "experience_buffer": {
            "buffer_type": "standard",
            "buffer_size": TRAINING_CONSTANTS["DEFAULT_BUFFER_SIZE"],
            "prioritized": False,
            "priority_alpha": TRAINING_CONSTANTS["DEFAULT_PRIORITY_ALPHA"],
            "priority_beta": TRAINING_CONSTANTS["DEFAULT_PRIORITY_BETA"]
        },
        "evaluation": {
            "eval_frequency": 10000,
            "eval_episodes": 10,
            "save_best_model": True,
            "early_stopping": False,
            "patience": 50
        },
        "logging": {
            "log_frequency": 1000,
            "tensorboard": True,
            "wandb": False,
            "save_frequency": 50000
        },
        "distributed": {
            "enabled": False,
            "num_workers": TRAINING_CONSTANTS["DEFAULT_NUM_WORKERS"],
            "sync_frequency": TRAINING_CONSTANTS["DEFAULT_SYNC_FREQUENCY"],
            "backend": TRAINING_CONSTANTS["DEFAULT_COMMUNICATION_BACKEND"]
        }
    }

def validate_training_setup(config: dict, environment: any) -> dict:
    """
    Validate training setup and configuration.
    
    Args:
        config (dict): Training configuration
        environment: Training environment
        
    Returns:
        dict: Validation results
    """
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "recommendations": []
    }
    
    # Validate basic configuration
    required_keys = ["algorithm", "learning_rate", "batch_size"]
    for key in required_keys:
        if key not in config.get("training", {}):
            validation_results["errors"].append(f"Missing required training parameter: {key}")
            validation_results["valid"] = False
    
    # Validate learning rate
    lr = config.get("training", {}).get("learning_rate", 0)
    if lr <= 0 or lr > 1:
        validation_results["errors"].append("Learning rate must be between 0 and 1")
        validation_results["valid"] = False
    elif lr > 0.01:
        validation_results["warnings"].append("Learning rate seems high, consider reducing")
    
    # Validate batch size
    batch_size = config.get("training", {}).get("batch_size", 0)
    if batch_size <= 0:
        validation_results["errors"].append("Batch size must be positive")
        validation_results["valid"] = False
    elif batch_size < 16:
        validation_results["warnings"].append("Small batch size may lead to unstable training")
    
    # Validate environment compatibility
    if hasattr(environment, 'observation_space') and hasattr(environment, 'action_space'):
        obs_space = environment.observation_space
        action_space = environment.action_space
        
        # Check if algorithm is compatible with action space
        algorithm = config.get("training", {}).get("algorithm", "").lower()
        
        if algorithm in ["ddpg", "sac"] and not hasattr(action_space, 'high'):
            validation_results["errors"].append(
                f"{algorithm.upper()} requires continuous action space"
            )
            validation_results["valid"] = False
        
        if algorithm == "dqn" and hasattr(action_space, 'high'):
            validation_results["warnings"].append(
                "DQN typically works better with discrete action spaces"
            )
    
    # Performance recommendations
    total_timesteps = config.get("training", {}).get("total_timesteps", 0)
    if total_timesteps < 100000:
        validation_results["recommendations"].append(
            "Consider increasing total_timesteps for better convergence"
        )
    
    buffer_size = config.get("experience_buffer", {}).get("buffer_size", 0)
    if buffer_size < total_timesteps / 10:
        validation_results["recommendations"].append(
            "Consider increasing buffer size relative to total timesteps"
        )
    
    return validation_results

def setup_training_environment(config: dict) -> dict:
    """
    Setup training environment based on configuration.
    
    Args:
        config (dict): Training configuration
        
    Returns:
        dict: Environment setup information
    """
    import torch
    import numpy as np
    import random
    
    # Set random seeds for reproducibility
    seed = config.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup logging
    import logging
    log_level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(level=getattr(logging, log_level))
    
    setup_info = {
        "seed": seed,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "pytorch_version": torch.__version__
    }
    
    return setup_info

# Integration utilities for battery-specific training
def create_battery_training_config(battery_type: str = "li_ion", 
                                 optimization_goal: str = "longevity") -> dict:
    """
    Create training configuration optimized for battery applications.
    
    Args:
        battery_type (str): Type of battery ('li_ion', 'lifepo4', 'lfp')
        optimization_goal (str): Optimization goal ('longevity', 'efficiency', 'safety')
        
    Returns:
        dict: Battery-optimized training configuration
    """
    base_config = get_training_config_template()
    
    # Battery-specific optimizations
    if optimization_goal == "longevity":
        base_config["training"].update({
            "gamma": 0.995,  # Higher discount for long-term rewards
            "learning_rate": 1e-4,  # More conservative learning
            "total_timesteps": 2000000  # Longer training for complex optimization
        })
    
    elif optimization_goal == "efficiency":
        base_config["training"].update({
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "total_timesteps": 1000000
        })
    
    elif optimization_goal == "safety":
        base_config["training"].update({
            "gamma": 0.999,  # Very high discount for safety
            "learning_rate": 5e-5,  # Very conservative for safety-critical applications
            "total_timesteps": 3000000  # Extensive training for safety
        })
    
    # Battery type specific adjustments
    if battery_type == "li_ion":
        base_config["training"]["max_grad_norm"] = 0.3  # More stable gradients
    elif battery_type == "lifepo4":
        base_config["training"]["tau"] = 0.001  # Slower target updates
    
    return base_config

# Module health check
def training_module_health_check() -> dict:
    """
    Perform health check of the training module.
    
    Returns:
        dict: Health check results
    """
    health_status = {
        "module_loaded": True,
        "version": __version__,
        "components_available": {
            "rl_trainer": True,
            "experience_buffer": True, 
            "policy_network": True,
            "value_network": True
        },
        "dependencies_satisfied": True,
        "gpu_available": False
    }
    
    try:
        import torch
        health_status["gpu_available"] = torch.cuda.is_available()
        health_status["pytorch_version"] = torch.__version__
        
        # Test basic functionality
        config = get_default_training_config("ppo")
        health_status["config_generation"] = True
        
    except Exception as e:
        health_status["error"] = str(e)
        health_status["dependencies_satisfied"] = False
    
    return health_status

# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BatteryMind RL Training Module v{__version__} initialized")
