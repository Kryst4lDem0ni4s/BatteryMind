"""
BatteryMind - Battery Health Trainer

Advanced training pipeline for battery health prediction transformer models.
Implements state-of-the-art training techniques including distributed training,
mixed precision, advanced optimizers, and comprehensive monitoring.

Features:
- Distributed training with AWS SageMaker integration
- Mixed precision training for improved performance
- Advanced learning rate scheduling and optimization
- Comprehensive metrics tracking and visualization
- Early stopping and model checkpointing
- Physics-informed loss functions
- Data augmentation and regularization
- Integration with MLflow for experiment tracking

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import time
import os
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# AWS and MLOps imports
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import mlflow
import mlflow.pytorch

# Local imports
from .model import BatteryHealthTransformer, BatteryHealthConfig
from .data_loader import BatteryDataLoader, BatteryDataset
from .preprocessing import BatteryPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BatteryTrainingConfig:
    """
    Comprehensive configuration for battery health model training.
    
    Attributes:
        # Training parameters
        batch_size (int): Training batch size
        learning_rate (float): Initial learning rate
        num_epochs (int): Number of training epochs
        warmup_steps (int): Number of warmup steps for learning rate
        weight_decay (float): Weight decay for regularization
        gradient_clip_norm (float): Gradient clipping norm
        
        # Optimization parameters
        optimizer_type (str): Optimizer type ('adam', 'adamw', 'sgd')
        scheduler_type (str): Learning rate scheduler type
        scheduler_params (Dict): Scheduler-specific parameters
        
        # Training techniques
        mixed_precision (bool): Enable mixed precision training
        gradient_accumulation_steps (int): Gradient accumulation steps
        early_stopping_patience (int): Early stopping patience
        
        # Data parameters
        train_split (float): Training data split ratio
        val_split (float): Validation data split ratio
        test_split (float): Test data split ratio
        
        # Checkpointing and logging
        checkpoint_dir (str): Directory for saving checkpoints
        log_interval (int): Logging interval in steps
        eval_interval (int): Evaluation interval in steps
        save_interval (int): Model saving interval in steps
        
        # Distributed training
        distributed (bool): Enable distributed training
        world_size (int): Number of processes for distributed training
        
        # Physics-informed training
        physics_loss_weight (float): Weight for physics-informed loss
        consistency_loss_weight (float): Weight for consistency loss
        
        # Regularization
        dropout_rate (float): Dropout rate
        label_smoothing (float): Label smoothing factor
        
        # AWS SageMaker integration
        use_sagemaker (bool): Use AWS SageMaker for training
        instance_type (str): SageMaker instance type
        instance_count (int): Number of SageMaker instances
    """
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 4000
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Optimization parameters
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine_with_warmup"
    scheduler_params: Dict = field(default_factory=dict)
    
    # Training techniques
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 10
    
    # Data parameters
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Checkpointing and logging
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    
    # Physics-informed training
    physics_loss_weight: float = 0.1
    consistency_loss_weight: float = 0.05
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.0
    
    # AWS SageMaker integration
    use_sagemaker: bool = False
    instance_type: str = "ml.p3.2xlarge"
    instance_count: int = 1

@dataclass
class BatteryTrainingMetrics:
    """
    Comprehensive metrics tracking for battery health training.
    """
    epoch: int = 0
    step: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.0
    
    # Battery-specific metrics
    soh_mae: float = 0.0
    soh_rmse: float = 0.0
    soh_r2: float = 0.0
    degradation_mae: float = 0.0
    degradation_rmse: float = 0.0
    
    # Physics-informed metrics
    physics_loss: float = 0.0
    consistency_loss: float = 0.0
    
    # Training efficiency metrics
    training_time: float = 0.0
    memory_usage: float = 0.0
    gpu_utilization: float = 0.0

class BatteryLossFunction(nn.Module):
    """
    Advanced loss function for battery health prediction with physics constraints.
    """
    
    def __init__(self, config: BatteryTrainingConfig):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.SmoothL1Loss()
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                battery_metadata: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss for battery health prediction.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth targets
            battery_metadata (Dict, optional): Battery metadata for physics constraints
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of loss components
        """
        # Primary prediction loss
        soh_pred = predictions[:, 0]
        soh_target = targets[:, 0]
        degradation_pred = predictions[:, 1:]
        degradation_target = targets[:, 1:]
        
        # SoH loss (more important, higher weight)
        soh_loss = self.huber_loss(soh_pred, soh_target)
        
        # Degradation pattern loss
        degradation_loss = self.mse_loss(degradation_pred, degradation_target)
        
        # Physics-informed constraints
        physics_loss = self._compute_physics_loss(predictions, battery_metadata)
        
        # Consistency loss (temporal consistency)
        consistency_loss = self._compute_consistency_loss(predictions)
        
        # Combined loss
        total_loss = (soh_loss + 
                     degradation_loss + 
                     self.config.physics_loss_weight * physics_loss +
                     self.config.consistency_loss_weight * consistency_loss)
        
        return {
            'total_loss': total_loss,
            'soh_loss': soh_loss,
            'degradation_loss': degradation_loss,
            'physics_loss': physics_loss,
            'consistency_loss': consistency_loss
        }
    
    def _compute_physics_loss(self, predictions: torch.Tensor,
                            battery_metadata: Optional[Dict] = None) -> torch.Tensor:
        """Compute physics-informed loss constraints."""
        physics_loss = torch.tensor(0.0, device=predictions.device)
        
        # SoH should be between 0 and 1
        soh = predictions[:, 0]
        physics_loss += torch.mean(torch.relu(-soh)) + torch.mean(torch.relu(soh - 1))
        
        # Degradation rates should be non-negative
        degradation = predictions[:, 1:]
        physics_loss += torch.mean(torch.relu(-degradation))
        
        # Temperature-dependent degradation (if metadata available)
        if battery_metadata and 'temperature' in battery_metadata:
            temp = battery_metadata['temperature']
            # Higher temperatures should correlate with higher degradation
            temp_factor = torch.exp((temp - 25) / 10)
            expected_degradation = degradation * temp_factor.unsqueeze(-1)
            physics_loss += self.mse_loss(degradation, expected_degradation)
        
        return physics_loss
    
    def _compute_consistency_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency loss."""
        if predictions.size(0) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # SoH should be monotonically decreasing or stable
        soh = predictions[:, 0]
        soh_diff = soh[1:] - soh[:-1]
        consistency_loss = torch.mean(torch.relu(soh_diff))  # Penalize increases
        
        return consistency_loss

class BatteryOptimizer:
    """
    Advanced optimizer with learning rate scheduling for battery health training.
    """
    
    def __init__(self, model: nn.Module, config: BatteryTrainingConfig):
        self.model = model
        self.config = config
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if config.mixed_precision else None
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.config.optimizer_type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        if self.config.scheduler_type == "cosine_with_warmup":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.warmup_steps,
                **self.config.scheduler_params
            )
        elif self.config.scheduler_type == "linear_warmup":
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )
        elif self.config.scheduler_type == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
    
    def step(self, loss: torch.Tensor) -> None:
        """Perform optimization step with gradient scaling if enabled."""
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()

class BatteryHealthTrainer:
    """
    Comprehensive training pipeline for battery health prediction models.
    
    Features:
    - Distributed training support
    - Mixed precision training
    - Advanced optimization and scheduling
    - Comprehensive metrics tracking
    - Early stopping and checkpointing
    - Integration with MLflow and Weights & Biases
    - AWS SageMaker integration
    """
    
    def __init__(self, model: BatteryHealthTransformer, config: BatteryTrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize training components
        self.loss_function = BatteryLossFunction(config)
        self.optimizer = BatteryOptimizer(model, config)
        
        # Metrics tracking
        self.metrics_history = []
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Setup directories
        self._setup_directories()
        
        # Initialize distributed training if enabled
        if config.distributed:
            self._setup_distributed_training()
        
        # Initialize experiment tracking
        self._setup_experiment_tracking()
        
        # Move model to device
        self.model.to(self.device)
        
        logger.info(f"BatteryHealthTrainer initialized on device: {self.device}")
    
    def _setup_directories(self) -> None:
        """Setup necessary directories for training."""
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.config.checkpoint_dir}/logs").mkdir(parents=True, exist_ok=True)
        Path(f"{self.config.checkpoint_dir}/metrics").mkdir(parents=True, exist_ok=True)
    
    def _setup_distributed_training(self) -> None:
        """Setup distributed training if enabled."""
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}")
        
        self.model = DDP(self.model, device_ids=[local_rank])
        logger.info(f"Distributed training setup complete. Local rank: {local_rank}")
    
    def _setup_experiment_tracking(self) -> None:
        """Setup experiment tracking with MLflow and Weights & Biases."""
        # MLflow setup
        mlflow.set_experiment("battery_health_prediction")
        mlflow.start_run()
        mlflow.log_params(self.config.__dict__)
        
        # Weights & Biases setup (optional)
        try:
            wandb.init(
                project="batterymind",
                name=f"battery_health_training_{int(time.time())}",
                config=self.config.__dict__
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Weights & Biases: {e}")
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
              test_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Main training loop with comprehensive monitoring and checkpointing.
        
        Args:
            train_dataloader (DataLoader): Training data loader
            val_dataloader (DataLoader): Validation data loader
            test_dataloader (DataLoader, optional): Test data loader
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        logger.info("Starting battery health model training...")
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(train_dataloader, epoch)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_dataloader, epoch)
            
            # Combine metrics
            epoch_metrics = BatteryTrainingMetrics(
                epoch=epoch,
                step=epoch * len(train_dataloader),
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                train_accuracy=train_metrics.get('accuracy', 0.0),
                val_accuracy=val_metrics.get('accuracy', 0.0),
                learning_rate=self.optimizer.scheduler.get_last_lr()[0],
                soh_mae=val_metrics.get('soh_mae', 0.0),
                soh_rmse=val_metrics.get('soh_rmse', 0.0),
                soh_r2=val_metrics.get('soh_r2', 0.0),
                training_time=time.time() - epoch_start_time
            )
            
            # Log metrics
            self._log_metrics(epoch_metrics)
            self.metrics_history.append(epoch_metrics)
            
            # Save checkpoint
            if epoch % (self.config.save_interval // len(train_dataloader)) == 0:
                self._save_checkpoint(epoch, val_metrics['loss'])
            
            # Early stopping check
            if self._check_early_stopping(val_metrics['loss']):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Log epoch completion
            logger.info(f"Epoch {epoch}/{self.config.num_epochs} completed. "
                       f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}")
        
        # Final evaluation
        final_results = self._final_evaluation(test_dataloader)
        
        # Cleanup
        self._cleanup_training()
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        return {
            'metrics_history': self.metrics_history,
            'final_results': final_results,
            'training_time': total_time,
            'best_val_loss': self.best_val_loss
        }
    
    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            inputs = batch['inputs'].to(self.device)
            targets = batch['targets'].to(self.device)
            metadata = batch.get('metadata', {})
            
            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(inputs, metadata)
                    loss_dict = self.loss_function(outputs['predictions'], targets, metadata)
                    loss = loss_dict['total_loss']
            else:
                outputs = self.model(inputs, metadata)
                loss_dict = self.loss_function(outputs['predictions'], targets, metadata)
                loss = loss_dict['total_loss']
            
            # Backward pass
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps
            
            self.optimizer.step(loss)
            
            # Update metrics
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / total_samples,
                'lr': self.optimizer.scheduler.get_last_lr()[0]
            })
            
            # Log batch metrics
            if batch_idx % self.config.log_interval == 0:
                self._log_batch_metrics(epoch, batch_idx, loss_dict)
        
        return {'loss': total_loss / total_samples}
    
    def _validate_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Validation Epoch {epoch}"):
                inputs = batch['inputs'].to(self.device)
                targets = batch['targets'].to(self.device)
                metadata = batch.get('metadata', {})
                
                # Forward pass
                outputs = self.model(inputs, metadata)
                loss_dict = self.loss_function(outputs['predictions'], targets, metadata)
                
                # Accumulate metrics
                total_loss += loss_dict['total_loss'].item() * inputs.size(0)
                total_samples += inputs.size(0)
                
                # Store predictions for detailed metrics
                all_predictions.append(outputs['predictions'].cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Compute detailed metrics
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # SoH metrics
        soh_pred = predictions[:, 0]
        soh_target = targets[:, 0]
        soh_mae = mean_absolute_error(soh_target, soh_pred)
        soh_rmse = np.sqrt(mean_squared_error(soh_target, soh_pred))
        soh_r2 = r2_score(soh_target, soh_pred)
        
        return {
            'loss': total_loss / total_samples,
            'soh_mae': soh_mae,
            'soh_rmse': soh_rmse,
            'soh_r2': soh_r2
        }
    
    def _log_metrics(self, metrics: BatteryTrainingMetrics) -> None:
        """Log metrics to various tracking systems."""
        # MLflow logging
        mlflow.log_metrics({
            'train_loss': metrics.train_loss,
            'val_loss': metrics.val_loss,
            'learning_rate': metrics.learning_rate,
            'soh_mae': metrics.soh_mae,
            'soh_rmse': metrics.soh_rmse,
            'soh_r2': metrics.soh_r2
        }, step=metrics.epoch)
        
        # Weights & Biases logging
        try:
            wandb.log({
                'epoch': metrics.epoch,
                'train_loss': metrics.train_loss,
                'val_loss': metrics.val_loss,
                'learning_rate': metrics.learning_rate,
                'soh_mae': metrics.soh_mae,
                'soh_rmse': metrics.soh_rmse,
                'soh_r2': metrics.soh_r2
            })
        except:
            pass
    
    def _log_batch_metrics(self, epoch: int, batch_idx: int, loss_dict: Dict[str, torch.Tensor]) -> None:
        """Log batch-level metrics."""
        step = epoch * 1000 + batch_idx  # Approximate step calculation
        
        mlflow.log_metrics({
            'batch_total_loss': loss_dict['total_loss'].item(),
            'batch_soh_loss': loss_dict['soh_loss'].item(),
            'batch_physics_loss': loss_dict['physics_loss'].item()
        }, step=step)
    
    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'metrics_history': self.metrics_history
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, f"{self.config.checkpoint_dir}/latest_checkpoint.ckpt")
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, f"{self.config.checkpoint_dir}/best_model.ckpt")
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # Save epoch checkpoint
        torch.save(checkpoint, f"{self.config.checkpoint_dir}/epoch_{epoch:03d}.ckpt")
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check early stopping criteria."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        return self.early_stopping_counter >= self.config.early_stopping_patience
    
    def _final_evaluation(self, test_dataloader: Optional[DataLoader]) -> Dict[str, Any]:
        """Perform final evaluation on test set."""
        if test_dataloader is None:
            return {}
        
        logger.info("Performing final evaluation on test set...")
        test_metrics = self._validate_epoch(test_dataloader, -1)
        
        # Log final test metrics
        mlflow.log_metrics({
            'test_loss': test_metrics['loss'],
            'test_soh_mae': test_metrics['soh_mae'],
            'test_soh_rmse': test_metrics['soh_rmse'],
            'test_soh_r2': test_metrics['soh_r2']
        })
        
        return test_metrics
    
    def _cleanup_training(self) -> None:
        """Cleanup training resources."""
        mlflow.end_run()
        try:
            wandb.finish()
        except:
            pass
        
        if self.config.distributed:
            dist.destroy_process_group()
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.optimizer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.metrics_history = checkpoint.get('metrics_history', [])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def hyperparameter_optimization(self, train_dataloader: DataLoader,
                                  val_dataloader: DataLoader,
                                  n_trials: int = 50) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization using Optuna.
        
        Args:
            train_dataloader (DataLoader): Training data
            val_dataloader (DataLoader): Validation data
            n_trials (int): Number of optimization trials
            
        Returns:
            Dict[str, Any]: Best hyperparameters and results
        """
        def objective(trial):
            # Suggest hyperparameters
            lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
            
            # Update config
            self.config.learning_rate = lr
            self.config.batch_size = batch_size
            self.config.weight_decay = weight_decay
            
            # Recreate optimizer with new hyperparameters
            self.optimizer = BatteryOptimizer(self.model, self.config)
            
            # Train for a few epochs
            self.config.num_epochs = 5
            results = self.train(train_dataloader, val_dataloader)
            
            return results['best_val_loss']
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }

# Factory functions for easy instantiation
def create_battery_trainer(model: BatteryHealthTransformer,
                          config: Optional[BatteryTrainingConfig] = None) -> BatteryHealthTrainer:
    """
    Factory function to create a BatteryHealthTrainer.
    
    Args:
        model (BatteryHealthTransformer): Model to train
        config (BatteryTrainingConfig, optional): Training configuration
        
    Returns:
        BatteryHealthTrainer: Configured trainer instance
    """
    if config is None:
        config = BatteryTrainingConfig()
    
    return BatteryHealthTrainer(model, config)

def train_battery_model_on_sagemaker(model_config: BatteryHealthConfig,
                                   training_config: BatteryTrainingConfig,
                                   data_path: str) -> str:
    """
    Train battery health model on AWS SageMaker.
    
    Args:
        model_config (BatteryHealthConfig): Model configuration
        training_config (BatteryTrainingConfig): Training configuration
        data_path (str): S3 path to training data
        
    Returns:
        str: SageMaker training job name
    """
    # Create SageMaker PyTorch estimator
    estimator = PyTorch(
        entry_point='train_sagemaker.py',
        source_dir='.',
        role=sagemaker.get_execution_role(),
        instance_type=training_config.instance_type,
        instance_count=training_config.instance_count,
        framework_version='1.12',
        py_version='py38',
        hyperparameters={
            'model_config': model_config.__dict__,
            'training_config': training_config.__dict__
        }
    )
    
    # Start training
    estimator.fit({'training': data_path})
    
    return estimator.latest_training_job.name
