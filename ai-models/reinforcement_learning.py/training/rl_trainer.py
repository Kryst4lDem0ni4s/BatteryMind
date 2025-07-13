"""
BatteryMind - Reinforcement Learning Trainer

Comprehensive training framework for reinforcement learning agents in battery
management systems. Supports multiple RL algorithms with advanced training
features including experience replay, curriculum learning, and multi-agent training.

Features:
- Support for PPO, SAC, DDPG, and DQN algorithms
- Experience replay with prioritized sampling
- Curriculum learning for complex battery scenarios
- Multi-agent training coordination
- Advanced logging and monitoring
- Integration with battery environments

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import time
import json
import pickle
from pathlib import Path
import threading
from collections import defaultdict, deque
import copy

# Visualization and monitoring
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter

# RL algorithm imports
from ..algorithms.ppo import PPOAlgorithm
from ..algorithms.sac import SACAlgorithm
from ..algorithms.ddpg import DDPGAlgorithm
from ..algorithms.dqn import DQNAlgorithm

# Environment imports
from ..environments.battery_env import BatteryEnvironment
from ..environments.charging_env import ChargingEnvironment
from ..environments.fleet_env import FleetEnvironment

# Reward imports
from ..rewards.composite_reward import CompositeReward

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """
    Configuration for RL training.
    
    Attributes:
        # Algorithm settings
        algorithm (str): RL algorithm to use ('ppo', 'sac', 'ddpg', 'dqn')
        total_timesteps (int): Total training timesteps
        learning_rate (float): Learning rate for optimizer
        batch_size (int): Batch size for training
        
        # Training parameters
        n_epochs (int): Number of training epochs per update
        n_steps (int): Number of steps per environment per update
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter
        clip_range (float): PPO clipping range
        
        # Experience replay
        buffer_size (int): Size of experience replay buffer
        min_buffer_size (int): Minimum buffer size before training
        prioritized_replay (bool): Use prioritized experience replay
        alpha (float): Prioritization exponent
        beta (float): Importance sampling exponent
        
        # Exploration
        exploration_fraction (float): Fraction of training for exploration
        exploration_initial_eps (float): Initial exploration rate
        exploration_final_eps (float): Final exploration rate
        
        # Network architecture
        policy_network_arch (List[int]): Policy network architecture
        value_network_arch (List[int]): Value network architecture
        activation_fn (str): Activation function ('relu', 'tanh', 'elu')
        
        # Training schedule
        learning_starts (int): Timesteps before learning starts
        train_freq (int): Training frequency (in timesteps)
        target_update_freq (int): Target network update frequency
        
        # Evaluation and logging
        eval_freq (int): Evaluation frequency (in timesteps)
        eval_episodes (int): Number of episodes for evaluation
        log_freq (int): Logging frequency (in timesteps)
        save_freq (int): Model saving frequency (in timesteps)
        
        # Multi-agent settings
        multi_agent (bool): Enable multi-agent training
        agent_types (List[str]): Types of agents for multi-agent training
        coordination_mechanism (str): Agent coordination mechanism
        
        # Curriculum learning
        curriculum_learning (bool): Enable curriculum learning
        curriculum_stages (List[Dict]): Curriculum learning stages
        
        # Advanced features
        use_lstm (bool): Use LSTM in policy/value networks
        normalize_observations (bool): Normalize observations
        normalize_rewards (bool): Normalize rewards
        clip_rewards (bool): Clip rewards to [-1, 1]
        
        # Device and performance
        device (str): Training device ('cpu', 'cuda')
        num_envs (int): Number of parallel environments
        num_workers (int): Number of worker processes
        
        # Checkpointing
        checkpoint_dir (str): Directory for saving checkpoints
        resume_from_checkpoint (Optional[str]): Path to checkpoint to resume from
        save_best_model (bool): Save best performing model
        
        # Early stopping
        early_stopping (bool): Enable early stopping
        patience (int): Early stopping patience
        min_improvement (float): Minimum improvement for early stopping
    """
    # Algorithm settings
    algorithm: str = "ppo"
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    batch_size: int = 64
    
    # Training parameters
    n_epochs: int = 10
    n_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    
    # Experience replay
    buffer_size: int = 100000
    min_buffer_size: int = 1000
    prioritized_replay: bool = True
    alpha: float = 0.6
    beta: float = 0.4
    
    # Exploration
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    
    # Network architecture
    policy_network_arch: List[int] = field(default_factory=lambda: [256, 256])
    value_network_arch: List[int] = field(default_factory=lambda: [256, 256])
    activation_fn: str = "relu"
    
    # Training schedule
    learning_starts: int = 10000
    train_freq: int = 4
    target_update_freq: int = 10000
    
    # Evaluation and logging
    eval_freq: int = 10000
    eval_episodes: int = 10
    log_freq: int = 1000
    save_freq: int = 50000
    
    # Multi-agent settings
    multi_agent: bool = False
    agent_types: List[str] = field(default_factory=lambda: ["charging", "thermal"])
    coordination_mechanism: str = "centralized"
    
    # Curriculum learning
    curriculum_learning: bool = False
    curriculum_stages: List[Dict] = field(default_factory=list)
    
    # Advanced features
    use_lstm: bool = False
    normalize_observations: bool = True
    normalize_rewards: bool = False
    clip_rewards: bool = False
    
    # Device and performance
    device: str = "auto"
    num_envs: int = 1
    num_workers: int = 4
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None
    save_best_model: bool = True
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_improvement: float = 0.01

@dataclass
class TrainingMetrics:
    """Training metrics and statistics."""
    episode: int = 0
    timestep: int = 0
    episode_reward: float = 0.0
    episode_length: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    explained_variance: float = 0.0
    learning_rate: float = 0.0
    exploration_rate: float = 0.0
    battery_health_improvement: float = 0.0
    energy_efficiency: float = 0.0
    safety_violations: int = 0

class CurriculumManager:
    """
    Manages curriculum learning for progressive training difficulty.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.current_stage = 0
        self.stage_progress = 0
        self.stages = config.curriculum_stages or self._default_curriculum()
        
    def _default_curriculum(self) -> List[Dict]:
        """Create default curriculum for battery training."""
        return [
            {
                "name": "basic_charging",
                "duration": 100000,
                "env_params": {
                    "max_charging_rate": 0.5,
                    "temperature_range": (20, 30),
                    "noise_level": 0.0
                }
            },
            {
                "name": "variable_conditions",
                "duration": 200000,
                "env_params": {
                    "max_charging_rate": 0.8,
                    "temperature_range": (10, 40),
                    "noise_level": 0.1
                }
            },
            {
                "name": "extreme_conditions",
                "duration": 300000,
                "env_params": {
                    "max_charging_rate": 1.0,
                    "temperature_range": (-10, 50),
                    "noise_level": 0.2
                }
            }
        ]
    
    def get_current_stage(self) -> Dict:
        """Get current curriculum stage."""
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return self.stages[-1]  # Stay at final stage
    
    def update_progress(self, timesteps: int) -> bool:
        """Update curriculum progress and return True if stage changed."""
        self.stage_progress += timesteps
        current_stage_info = self.get_current_stage()
        
        if (self.stage_progress >= current_stage_info["duration"] and 
            self.current_stage < len(self.stages) - 1):
            self.current_stage += 1
            self.stage_progress = 0
            logger.info(f"Advanced to curriculum stage {self.current_stage}: {self.get_current_stage()['name']}")
            return True
        
        return False
    
    def get_env_params(self) -> Dict:
        """Get environment parameters for current stage."""
        return self.get_current_stage().get("env_params", {})

class RLTrainer:
    """
    Main reinforcement learning trainer for battery management systems.
    """
    
    def __init__(self, config: TrainingConfig, environment_factory: Callable,
                 model_save_path: Optional[str] = None):
        self.config = config
        self.environment_factory = environment_factory
        self.model_save_path = model_save_path or "./models"
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize components
        self.env = None
        self.algorithm = None
        self.experience_buffer = None
        self.curriculum_manager = None
        
        # Training state
        self.current_timestep = 0
        self.current_episode = 0
        self.best_reward = float('-inf')
        self.training_metrics = []
        self.evaluation_metrics = []
        
        # Early stopping
        self.early_stopping_counter = 0
        self.best_eval_reward = float('-inf')
        
        # Logging and monitoring
        self.writer = None
        self.setup_logging()
        
        # Multi-threading
        self.training_lock = threading.Lock()
        
        logger.info(f"RLTrainer initialized with {config.algorithm} algorithm")
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def setup_logging(self) -> None:
        """Setup TensorBoard logging."""
        log_dir = Path(self.config.checkpoint_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_dir))
    
    def initialize_training(self) -> None:
        """Initialize training components."""
        # Create environment
        self.env = self.environment_factory()
        
        # Initialize curriculum manager
        if self.config.curriculum_learning:
            self.curriculum_manager = CurriculumManager(self.config)
            # Apply initial curriculum parameters
            env_params = self.curriculum_manager.get_env_params()
            if hasattr(self.env, 'update_parameters'):
                self.env.update_parameters(env_params)
        
        # Initialize algorithm
        self.algorithm = self._create_algorithm()
        
        # Initialize experience buffer
        from .experience_buffer import ExperienceBuffer
        self.experience_buffer = ExperienceBuffer(
            capacity=self.config.buffer_size,
            prioritized=self.config.prioritized_replay,
            alpha=self.config.alpha,
            beta=self.config.beta
        )
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
        
        logger.info("Training initialization completed")
    
    def _create_algorithm(self):
        """Create RL algorithm based on configuration."""
        obs_space = self.env.observation_space
        action_space = self.env.action_space
        
        algorithm_params = {
            'observation_space': obs_space,
            'action_space': action_space,
            'learning_rate': self.config.learning_rate,
            'device': self.device,
            'policy_network_arch': self.config.policy_network_arch,
            'value_network_arch': self.config.value_network_arch
        }
        
        if self.config.algorithm == "ppo":
            algorithm_params.update({
                'n_steps': self.config.n_steps,
                'batch_size': self.config.batch_size,
                'n_epochs': self.config.n_epochs,
                'gamma': self.config.gamma,
                'gae_lambda': self.config.gae_lambda,
                'clip_range': self.config.clip_range
            })
            return PPOAlgorithm(**algorithm_params)
        
        elif self.config.algorithm == "sac":
            algorithm_params.update({
                'buffer_size': self.config.buffer_size,
                'batch_size': self.config.batch_size,
                'gamma': self.config.gamma,
                'tau': 0.005,  # Soft update coefficient
                'alpha': 0.2   # Entropy coefficient
            })
            return SACAlgorithm(**algorithm_params)
        
        elif self.config.algorithm == "ddpg":
            algorithm_params.update({
                'buffer_size': self.config.buffer_size,
                'batch_size': self.config.batch_size,
                'gamma': self.config.gamma,
                'tau': 0.005,
                'noise_std': 0.1
            })
            return DDPGAlgorithm(**algorithm_params)
        
        elif self.config.algorithm == "dqn":
            algorithm_params.update({
                'buffer_size': self.config.buffer_size,
                'batch_size': self.config.batch_size,
                'gamma': self.config.gamma,
                'target_update_freq': self.config.target_update_freq,
                'exploration_fraction': self.config.exploration_fraction,
                'exploration_initial_eps': self.config.exploration_initial_eps,
                'exploration_final_eps': self.config.exploration_final_eps
            })
            return DQNAlgorithm(**algorithm_params)
        
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Dict[str, Any]: Training results and statistics
        """
        logger.info("Starting RL training...")
        start_time = time.time()
        
        # Initialize training
        self.initialize_training()
        
        # Training loop
        obs = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_start_time = time.time()
        
        while self.current_timestep < self.config.total_timesteps:
            # Select action
            action = self.algorithm.predict(obs, deterministic=False)
            
            # Take step in environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Store experience
            if hasattr(self.algorithm, 'store_transition'):
                self.algorithm.store_transition(obs, action, reward, next_obs, done)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            self.current_timestep += 1
            
            # Update curriculum if enabled
            if self.curriculum_manager:
                stage_changed = self.curriculum_manager.update_progress(1)
                if stage_changed:
                    env_params = self.curriculum_manager.get_env_params()
                    if hasattr(self.env, 'update_parameters'):
                        self.env.update_parameters(env_params)
            
            # Training step
            if (self.current_timestep >= self.config.learning_starts and
                self.current_timestep % self.config.train_freq == 0):
                
                training_info = self.algorithm.train()
                
                # Log training metrics
                if training_info and self.current_timestep % self.config.log_freq == 0:
                    self._log_training_metrics(training_info)
            
            # Episode end
            if done:
                # Record episode metrics
                episode_time = time.time() - episode_start_time
                metrics = TrainingMetrics(
                    episode=self.current_episode,
                    timestep=self.current_timestep,
                    episode_reward=episode_reward,
                    episode_length=episode_length,
                    battery_health_improvement=info.get('battery_health_improvement', 0.0),
                    energy_efficiency=info.get('energy_efficiency', 0.0),
                    safety_violations=info.get('safety_violations', 0)
                )
                
                self.training_metrics.append(metrics)
                
                # Log episode metrics
                if self.writer:
                    self.writer.add_scalar('Episode/Reward', episode_reward, self.current_episode)
                    self.writer.add_scalar('Episode/Length', episode_length, self.current_episode)
                    self.writer.add_scalar('Episode/Time', episode_time, self.current_episode)
                
                # Reset for next episode
                obs = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                episode_start_time = time.time()
                self.current_episode += 1
                
                logger.info(f"Episode {self.current_episode}: Reward={episode_reward:.2f}, "
                           f"Length={episode_length}, Timestep={self.current_timestep}")
            else:
                obs = next_obs
            
            # Evaluation
            if self.current_timestep % self.config.eval_freq == 0:
                eval_results = self.evaluate()
                self._log_evaluation_metrics(eval_results)
                
                # Early stopping check
                if self.config.early_stopping:
                    if self._check_early_stopping(eval_results['mean_reward']):
                        logger.info("Early stopping triggered")
                        break
            
            # Save checkpoint
            if self.current_timestep % self.config.save_freq == 0:
                self.save_checkpoint()
        
        # Final evaluation and save
        final_eval = self.evaluate()
        self.save_model()
        
        training_time = time.time() - start_time
        
        # Compile results
        results = {
            'total_timesteps': self.current_timestep,
            'total_episodes': self.current_episode,
            'training_time': training_time,
            'final_evaluation': final_eval,
            'best_reward': self.best_reward,
            'training_metrics': self.training_metrics,
            'evaluation_metrics': self.evaluation_metrics
        }
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        return results
    
    def evaluate(self, num_episodes: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate the current policy.
        
        Args:
            num_episodes (int, optional): Number of episodes to evaluate
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        num_episodes = num_episodes or self.config.eval_episodes
        
        episode_rewards = []
        episode_lengths = []
        battery_improvements = []
        energy_efficiencies = []
        safety_violations = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action = self.algorithm.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            battery_improvements.append(info.get('battery_health_improvement', 0.0))
            energy_efficiencies.append(info.get('energy_efficiency', 0.0))
            safety_violations.append(info.get('safety_violations', 0))
        
        eval_metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_battery_improvement': np.mean(battery_improvements),
            'mean_energy_efficiency': np.mean(energy_efficiencies),
            'total_safety_violations': np.sum(safety_violations)
        }
        
        # Update best reward
        if eval_metrics['mean_reward'] > self.best_reward:
            self.best_reward = eval_metrics['mean_reward']
            if self.config.save_best_model:
                self.save_model(suffix='_best')
        
        self.evaluation_metrics.append({
            'timestep': self.current_timestep,
            'metrics': eval_metrics
        })
        
        return eval_metrics
    
    def _log_training_metrics(self, training_info: Dict) -> None:
        """Log training metrics to TensorBoard."""
        if not self.writer:
            return
        
        for key, value in training_info.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Training/{key}', value, self.current_timestep)
    
    def _log_evaluation_metrics(self, eval_metrics: Dict) -> None:
        """Log evaluation metrics to TensorBoard."""
        if not self.writer:
            return
        
        for key, value in eval_metrics.items():
            self.writer.add_scalar(f'Evaluation/{key}', value, self.current_timestep)
    
    def _check_early_stopping(self, eval_reward: float) -> bool:
        """Check if early stopping criteria are met."""
        if eval_reward > self.best_eval_reward + self.config.min_improvement:
            self.best_eval_reward = eval_reward
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= self.config.patience
    
    def save_checkpoint(self) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'timestep': self.current_timestep,
            'episode': self.current_episode,
            'algorithm_state': self.algorithm.get_state(),
            'best_reward': self.best_reward,
            'config': self.config,
            'training_metrics': self.training_metrics,
            'evaluation_metrics': self.evaluation_metrics
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_{self.current_timestep}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.current_timestep = checkpoint['timestep']
        self.current_episode = checkpoint['episode']
        self.best_reward = checkpoint['best_reward']
        self.training_metrics = checkpoint['training_metrics']
        self.evaluation_metrics = checkpoint['evaluation_metrics']
        
        if self.algorithm:
            self.algorithm.load_state(checkpoint['algorithm_state'])
        
        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
    
    def save_model(self, suffix: str = "") -> None:
        """Save trained model."""
        model_dir = Path(self.model_save_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{self.config.algorithm}_model{suffix}.pkl"
        self.algorithm.save(str(model_path))
        
        # Save configuration
        config_path = model_dir / f"{self.config.algorithm}_config{suffix}.json"
        with open(config_path, 'w') as f:
            # Convert dataclass to dict for JSON serialization
            config_dict = {k: v for k, v in self.config.__dict__.items() 
                          if not callable(v) and not k.startswith('_')}
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved: {model_path}")
    
    def close(self) -> None:
        """Clean up resources."""
        if self.writer:
            self.writer.close()
        
        if self.env:
            self.env.close()

# Factory functions
def create_rl_trainer(algorithm: str, environment_factory: Callable,
                     **kwargs) -> RLTrainer:
    """
    Factory function to create RL trainer.
    
    Args:
        algorithm (str): RL algorithm name
        environment_factory (Callable): Function to create environment
        **kwargs: Additional configuration parameters
        
    Returns:
        RLTrainer: Configured trainer instance
    """
    config = TrainingConfig(algorithm=algorithm, **kwargs)
    return RLTrainer(config, environment_factory)

def train_battery_charging_agent(total_timesteps: int = 1000000,
                                algorithm: str = "ppo") -> Dict[str, Any]:
    """
    Train a battery charging optimization agent.
    
    Args:
        total_timesteps (int): Total training timesteps
        algorithm (str): RL algorithm to use
        
    Returns:
        Dict[str, Any]: Training results
    """
    def env_factory():
        return ChargingEnvironment()
    
    trainer = create_rl_trainer(
        algorithm=algorithm,
        environment_factory=env_factory,
        total_timesteps=total_timesteps,
        curriculum_learning=True
    )
    
    return trainer.train()

def train_multi_agent_system(total_timesteps: int = 1000000) -> Dict[str, Any]:
    """
    Train multi-agent system for battery fleet management.
    
    Args:
        total_timesteps (int): Total training timesteps
        
    Returns:
        Dict[str, Any]: Training results
    """
    def env_factory():
        return FleetEnvironment(num_batteries=10)
    
    trainer = create_rl_trainer(
        algorithm="ppo",
        environment_factory=env_factory,
        total_timesteps=total_timesteps,
        multi_agent=True,
        agent_types=["charging", "thermal", "load_balancing"]
    )
    
    return trainer.train()
