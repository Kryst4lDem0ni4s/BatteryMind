"""
BatteryMind - Degradation Forecaster Trainer

Advanced training pipeline for degradation forecasting models with time-series
specific optimizations, uncertainty quantification, and comprehensive monitoring.

Features:
- Time-series aware training with temporal validation
- Multi-horizon loss functions with uncertainty penalties
- Seasonal pattern learning and validation
- Physics-informed loss constraints
- Advanced learning rate scheduling for forecasting
- Comprehensive forecasting metrics and evaluation

Author: BatteryMind Development Team
Version: 1.0.0
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb
import optuna

# Time series specific imports
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy import stats
from scipy.stats import pearsonr
import warnings

# AWS and MLOps imports
import boto3
import mlflow
import mlflow.pytorch

# Local imports
from .model import DegradationForecaster, DegradationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DegradationTrainingConfig:
    """
    Comprehensive configuration for degradation forecasting model training.
    
    Attributes:
        # Training parameters
        batch_size (int): Training batch size
        learning_rate (float): Initial learning rate
        num_epochs (int): Number of training epochs
        warmup_steps (int): Number of warmup steps
        weight_decay (float): Weight decay for regularization
        gradient_clip_norm (float): Gradient clipping norm
        
        # Time-series specific parameters
        temporal_validation (bool): Use temporal validation splits
        validation_window_size (int): Size of validation window in time steps
        forecast_loss_weights (List[float]): Weights for different forecast horizons
        
        # Uncertainty training
        uncertainty_loss_weight (float): Weight for uncertainty loss
        quantile_loss_alpha (float): Alpha parameter for quantile loss
        
        # Physics-informed training
        physics_loss_weight (float): Weight for physics constraints
        temporal_consistency_weight (float): Weight for temporal consistency
        seasonal_consistency_weight (float): Weight for seasonal consistency
        
        # Advanced optimization
        optimizer_type (str): Optimizer type
        scheduler_type (str): Learning rate scheduler type
        scheduler_params (Dict): Scheduler parameters
        
        # Training techniques
        mixed_precision (bool): Enable mixed precision training
        gradient_accumulation_steps (int): Gradient accumulation steps
        early_stopping_patience (int): Early stopping patience
        
        # Monitoring and logging
        log_interval (int): Logging interval in steps
        eval_interval (int): Evaluation interval in steps
        save_interval (int): Model saving interval in steps
        
        # Forecasting evaluation
        forecast_metrics (List[str]): Metrics to evaluate forecasting performance
        confidence_levels (List[float]): Confidence levels for prediction intervals
    """
    # Training parameters
    batch_size: int = 16  # Smaller batch for longer sequences
    learning_rate: float = 5e-5  # Lower learning rate for stability
    num_epochs: int = 200
    warmup_steps: int = 8000
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Time-series specific parameters
    temporal_validation: bool = True
    validation_window_size: int = 168  # 1 week
    forecast_loss_weights: List[float] = field(default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2])
    
    # Uncertainty training
    uncertainty_loss_weight: float = 0.1
    quantile_loss_alpha: float = 0.1
    
    # Physics-informed training
    physics_loss_weight: float = 0.05
    temporal_consistency_weight: float = 0.02
    seasonal_consistency_weight: float = 0.01
    
    # Advanced optimization
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine_with_restarts"
    scheduler_params: Dict = field(default_factory=dict)
    
    # Training techniques
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 2
    early_stopping_patience: int = 20
    
    # Monitoring and logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    
    # Forecasting evaluation
    forecast_metrics: List[str] = field(default_factory=lambda: [
        'mse', 'mae', 'mape', 'smape', 'mase', 'coverage_probability'
    ])
    confidence_levels: List[float] = field(default_factory=lambda: [0.8, 0.9, 0.95])

@dataclass
class DegradationTrainingMetrics:
    """
    Comprehensive metrics for degradation forecasting training.
    """
    epoch: int = 0
    step: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    
    # Forecasting specific metrics
    forecast_mse: float = 0.0
    forecast_mae: float = 0.0
    forecast_mape: float = 0.0
    forecast_smape: float = 0.0
    forecast_mase: float = 0.0
    
    # Uncertainty metrics
    coverage_probability: float = 0.0
    prediction_interval_width: float = 0.0
    uncertainty_calibration: float = 0.0
    
    # Physics-informed metrics
    physics_violation_rate: float = 0.0
    temporal_consistency_score: float = 0.0
    seasonal_consistency_score: float = 0.0
    
    # Training efficiency
    learning_rate: float = 0.0
    training_time: float = 0.0
    memory_usage: float = 0.0

class TimeSeriesLoss(nn.Module):
    """
    Advanced loss function for time-series forecasting with uncertainty.
    """
    
    def __init__(self, config: DegradationTrainingConfig):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
    def quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                     quantiles: torch.Tensor) -> torch.Tensor:
        """Compute quantile loss for uncertainty estimation."""
        errors = targets.unsqueeze(-1) - predictions
        loss = torch.maximum(
            quantiles * errors,
            (quantiles - 1) * errors
        )
        return loss.mean()
    
    def temporal_consistency_loss(self, forecasts: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency loss to ensure smooth forecasts."""
        if forecasts.size(1) < 2:
            return torch.tensor(0.0, device=forecasts.device)
        
        # Calculate differences between consecutive time steps
        temporal_diff = forecasts[:, 1:, :] - forecasts[:, :-1, :]
        
        # Penalize large jumps in degradation rates
        consistency_loss = torch.mean(torch.abs(temporal_diff))
        
        return consistency_loss
    
    def seasonal_consistency_loss(self, forecasts: torch.Tensor, 
                                seasonal_period: int = 24) -> torch.Tensor:
        """Compute seasonal consistency loss."""
        if forecasts.size(1) < seasonal_period * 2:
            return torch.tensor(0.0, device=forecasts.device)
        
        # Compare forecasts at seasonal intervals
        seasonal_diff = (forecasts[:, seasonal_period:, :] - 
                        forecasts[:, :-seasonal_period, :])
        
        # Penalize large seasonal variations in degradation patterns
        seasonal_loss = torch.mean(torch.abs(seasonal_diff))
        
        return seasonal_loss
    
    def physics_constraint_loss(self, forecasts: torch.Tensor) -> torch.Tensor:
        """Compute physics constraint violation loss."""
        physics_loss = torch.tensor(0.0, device=forecasts.device)
        
        # Degradation rates should be non-negative
        negative_penalty = torch.mean(torch.relu(-forecasts))
        physics_loss += negative_penalty
        
        # Degradation should be monotonic (non-decreasing cumulative)
        cumulative_degradation = torch.cumsum(forecasts, dim=1)
        for i in range(1, cumulative_degradation.size(1)):
            monotonic_penalty = torch.mean(
                torch.relu(cumulative_degradation[:, i-1, :] - cumulative_degradation[:, i, :])
            )
            physics_loss += monotonic_penalty
        
        return physics_loss
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss for degradation forecasting.
        
        Args:
            predictions (Dict[str, torch.Tensor]): Model predictions
            targets (torch.Tensor): Ground truth targets
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of loss components
        """
        forecasts = predictions['forecasts']
        
        # Primary forecasting loss
        forecast_loss = self.mse_loss(forecasts, targets)
        
        # Multi-horizon weighted loss
        if len(self.config.forecast_loss_weights) > 1:
            horizon_losses = []
            for i, weight in enumerate(self.config.forecast_loss_weights):
                if i < forecasts.size(1):
                    horizon_loss = self.mse_loss(forecasts[:, i, :], targets[:, i, :])
                    horizon_losses.append(weight * horizon_loss)
            
            if horizon_losses:
                forecast_loss = torch.stack(horizon_losses).sum()
        
        # Uncertainty loss
        uncertainty_loss = torch.tensor(0.0, device=forecasts.device)
        if 'quantiles' in predictions:
            quantiles = predictions['quantiles']
            quantile_values = torch.tensor(
                predictions.get('quantile_values', [0.1, 0.25, 0.75, 0.9]),
                device=forecasts.device
            )
            uncertainty_loss = self.quantile_loss(quantiles, targets, quantile_values)
        
        # Physics constraint loss
        physics_loss = self.physics_constraint_loss(forecasts)
        
        # Temporal consistency loss
        temporal_loss = self.temporal_consistency_loss(forecasts)
        
        # Seasonal consistency loss
        seasonal_loss = self.seasonal_consistency_loss(forecasts)
        
        # Combined loss
        total_loss = (forecast_loss + 
                     self.config.uncertainty_loss_weight * uncertainty_loss +
                     self.config.physics_loss_weight * physics_loss +
                     self.config.temporal_consistency_weight * temporal_loss +
                     self.config.seasonal_consistency_weight * seasonal_loss)
        
        return {
            'total_loss': total_loss,
            'forecast_loss': forecast_loss,
            'uncertainty_loss': uncertainty_loss,
            'physics_loss': physics_loss,
            'temporal_consistency_loss': temporal_loss,
            'seasonal_consistency_loss': seasonal_loss
        }

class ForecastingOptimizer:
    """
    Advanced optimizer for degradation forecasting with specialized scheduling.
    """
    
    def __init__(self, model: nn.Module, config: DegradationTrainingConfig):
        self.model = model
        self.config = config
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if config.mixed_precision else None
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.config.optimizer_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.95)  # Adjusted for forecasting
            )
        elif self.config.optimizer_type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler for forecasting."""
        if self.config.scheduler_type == "cosine_with_restarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.warmup_steps,
                T_mult=2,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler_type == "linear_warmup":
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
    
    def step(self, loss: torch.Tensor) -> None:
        """Perform optimization step with gradient scaling."""
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

class DegradationTrainer:
    """
    Comprehensive training pipeline for degradation forecasting models.
    """
    
    def __init__(self, model: DegradationForecaster, config: DegradationTrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize training components
        self.loss_function = TimeSeriesLoss(config)
        self.optimizer = ForecastingOptimizer(model, config)
        
        # Metrics tracking
        self.metrics_history = []
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Setup directories
        self._setup_directories()
        
        # Initialize experiment tracking
        self._setup_experiment_tracking()
        
        # Move model to device
        self.model.to(self.device)
        
        logger.info(f"DegradationTrainer initialized on device: {self.device}")
    
    def _setup_directories(self) -> None:
        """Setup necessary directories for training."""
        Path("./checkpoints").mkdir(parents=True, exist_ok=True)
        Path("./checkpoints/logs").mkdir(parents=True, exist_ok=True)
        Path("./checkpoints/metrics").mkdir(parents=True, exist_ok=True)
    
    def _setup_experiment_tracking(self) -> None:
        """Setup experiment tracking."""
        # MLflow setup
        mlflow.set_experiment("degradation_forecasting")
        mlflow.start_run()
        mlflow.log_params(self.config.__dict__)
        
        # Weights & Biases setup
        try:
            wandb.init(
                project="batterymind-forecasting",
                name=f"degradation_forecasting_{int(time.time())}",
                config=self.config.__dict__
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Weights & Biases: {e}")
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
              test_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Main training loop for degradation forecasting.
        
        Args:
            train_dataloader (DataLoader): Training data loader
            val_dataloader (DataLoader): Validation data loader
            test_dataloader (DataLoader, optional): Test data loader
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        logger.info("Starting degradation forecasting model training...")
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(train_dataloader, epoch)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_dataloader, epoch)
            
            # Combine metrics
            epoch_metrics = DegradationTrainingMetrics(
                epoch=epoch,
                step=epoch * len(train_dataloader),
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                forecast_mse=val_metrics.get('forecast_mse', 0.0),
                forecast_mae=val_metrics.get('forecast_mae', 0.0),
                forecast_mape=val_metrics.get('forecast_mape', 0.0),
                coverage_probability=val_metrics.get('coverage_probability', 0.0),
                learning_rate=self.optimizer.scheduler.get_last_lr()[0],
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
            inputs = batch['inputs'].to(self.device)
            targets = batch['targets'].to(self.device)
            time_features = batch.get('time_features', {})
            
            # Forward pass
            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(inputs, time_features)
                    loss_dict = self.loss_function(outputs, targets)
                    loss = loss_dict['total_loss']
            else:
                outputs = self.model(inputs, time_features)
                loss_dict = self.loss_function(outputs, targets)
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
        
        return {'loss': total_loss / total_samples}
    
    def _validate_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch with forecasting metrics."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        all_forecasts = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Validation Epoch {epoch}"):
                inputs = batch['inputs'].to(self.device)
                targets = batch['targets'].to(self.device)
                time_features = batch.get('time_features', {})
                
                outputs = self.model(inputs, time_features)
                loss_dict = self.loss_function(outputs, targets)
                
                total_loss += loss_dict['total_loss'].item() * inputs.size(0)
                total_samples += inputs.size(0)
                
                # Store predictions for detailed metrics
                all_forecasts.append(outputs['forecasts'].cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                if 'std' in outputs:
                    all_uncertainties.append(outputs['std'].cpu().numpy())
        
        # Compute forecasting metrics
        forecasts = np.concatenate(all_forecasts, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        metrics = self._compute_forecasting_metrics(forecasts, targets)
        
        # Add uncertainty metrics if available
        if all_uncertainties:
            uncertainties = np.concatenate(all_uncertainties, axis=0)
            uncertainty_metrics = self._compute_uncertainty_metrics(
                forecasts, targets, uncertainties
            )
            metrics.update(uncertainty_metrics)
        
        metrics['loss'] = total_loss / total_samples
        return metrics
    
    def _compute_forecasting_metrics(self, forecasts: np.ndarray, 
                                   targets: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive forecasting metrics."""
        metrics = {}
        
        # Flatten for metric computation
        forecasts_flat = forecasts.reshape(-1, forecasts.shape[-1])
        targets_flat = targets.reshape(-1, targets.shape[-1])
        
        # Basic metrics
        metrics['forecast_mse'] = mean_squared_error(targets_flat, forecasts_flat)
        metrics['forecast_mae'] = mean_absolute_error(targets_flat, forecasts_flat)
        
        # MAPE (handling zero values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mape_values = []
            for i in range(targets_flat.shape[1]):
                non_zero_mask = targets_flat[:, i] != 0
                if non_zero_mask.sum() > 0:
                    mape = mean_absolute_percentage_error(
                        targets_flat[non_zero_mask, i], 
                        forecasts_flat[non_zero_mask, i]
                    )
                    mape_values.append(mape)
            
            metrics['forecast_mape'] = np.mean(mape_values) if mape_values else 0.0
        
        # SMAPE (Symmetric MAPE)
        smape_values = []
        for i in range(targets_flat.shape[1]):
            numerator = np.abs(forecasts_flat[:, i] - targets_flat[:, i])
            denominator = (np.abs(forecasts_flat[:, i]) + np.abs(targets_flat[:, i])) / 2
            smape = np.mean(numerator / (denominator + 1e-8)) * 100
            smape_values.append(smape)
        
        metrics['forecast_smape'] = np.mean(smape_values)
        
        return metrics
    
    def _compute_uncertainty_metrics(self, forecasts: np.ndarray, targets: np.ndarray,
                                   uncertainties: np.ndarray) -> Dict[str, float]:
        """Compute uncertainty quantification metrics."""
        metrics = {}
        
        # Coverage probability for different confidence levels
        for confidence_level in self.config.confidence_levels:
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            lower_bound = forecasts - z_score * uncertainties
            upper_bound = forecasts + z_score * uncertainties
            
            coverage = np.mean((targets >= lower_bound) & (targets <= upper_bound))
            metrics[f'coverage_{int(confidence_level*100)}'] = coverage
        
        # Average prediction interval width
        z_score_95 = stats.norm.ppf(0.975)
        interval_width = 2 * z_score_95 * uncertainties
        metrics['prediction_interval_width'] = np.mean(interval_width)
        
        # Overall coverage probability (95% confidence)
        metrics['coverage_probability'] = metrics.get('coverage_95', 0.0)
        
        return metrics
    
    def _log_metrics(self, metrics: DegradationTrainingMetrics) -> None:
        """Log metrics to tracking systems."""
        # MLflow logging
        mlflow.log_metrics({
            'train_loss': metrics.train_loss,
            'val_loss': metrics.val_loss,
            'forecast_mse': metrics.forecast_mse,
            'forecast_mae': metrics.forecast_mae,
            'forecast_mape': metrics.forecast_mape,
            'coverage_probability': metrics.coverage_probability,
            'learning_rate': metrics.learning_rate
        }, step=metrics.epoch)
        
        # Weights & Biases logging
        try:
            wandb.log({
                'epoch': metrics.epoch,
                'train_loss': metrics.train_loss,
                'val_loss': metrics.val_loss,
                'forecast_mse': metrics.forecast_mse,
                'forecast_mae': metrics.forecast_mae,
                'forecast_mape': metrics.forecast_mape,
                'coverage_probability': metrics.coverage_probability,
                'learning_rate': metrics.learning_rate
            })
        except:
            pass
    
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
        
        # Additional forecasting-specific evaluation
        forecasting_metrics = self._evaluate_forecasting_performance(test_dataloader)
        test_metrics.update(forecasting_metrics)
        
        # Log final test metrics
        mlflow.log_metrics({
            'test_loss': test_metrics['loss'],
            'test_mape': test_metrics.get('mape', 0.0),
            'test_forecast_accuracy': test_metrics.get('forecast_accuracy', 0.0),
            'test_uncertainty_coverage': test_metrics.get('uncertainty_coverage', 0.0)
        })
        
        return test_metrics
    
    def _evaluate_forecasting_performance(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate forecasting-specific performance metrics."""
        self.model.eval()
        
        all_forecasts = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating forecasting performance"):
                inputs = batch['inputs'].to(self.device)
                targets = batch['targets'].to(self.device)
                time_features = batch.get('time_features', {})
                
                # Forward pass
                outputs = self.model(inputs, time_features)
                forecasts = outputs['forecasts']
                
                # Store predictions and targets
                all_forecasts.append(forecasts.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                # Store uncertainties if available
                if 'std' in outputs:
                    all_uncertainties.append(outputs['std'].cpu().numpy())
        
        # Concatenate all predictions
        forecasts = np.concatenate(all_forecasts, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Calculate forecasting metrics
        metrics = {}
        
        # Mean Absolute Percentage Error
        mape = mean_absolute_percentage_error(
            targets.reshape(-1), forecasts.reshape(-1)
        )
        metrics['mape'] = mape
        
        # Forecast accuracy (1 - normalized RMSE)
        rmse = np.sqrt(mean_squared_error(targets.reshape(-1), forecasts.reshape(-1)))
        target_range = targets.max() - targets.min()
        normalized_rmse = rmse / target_range if target_range > 0 else rmse
        metrics['forecast_accuracy'] = max(0, 1 - normalized_rmse)
        
        # Directional accuracy
        target_direction = np.diff(targets, axis=1) > 0
        forecast_direction = np.diff(forecasts, axis=1) > 0
        directional_accuracy = np.mean(target_direction == forecast_direction)
        metrics['directional_accuracy'] = directional_accuracy
        
        # Uncertainty coverage if available
        if all_uncertainties:
            uncertainties = np.concatenate(all_uncertainties, axis=0)
            # Calculate 95% prediction interval coverage
            lower_bound = forecasts - 1.96 * uncertainties
            upper_bound = forecasts + 1.96 * uncertainties
            coverage = np.mean((targets >= lower_bound) & (targets <= upper_bound))
            metrics['uncertainty_coverage'] = coverage
        
        # Horizon-specific metrics
        for h in range(min(24, forecasts.shape[1])):  # First 24 hours
            horizon_mape = mean_absolute_percentage_error(
                targets[:, h, :].reshape(-1), forecasts[:, h, :].reshape(-1)
            )
            metrics[f'mape_horizon_{h+1}h'] = horizon_mape
        
        return metrics
    
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
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
            weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
            n_layers = trial.suggest_int('n_layers', 4, 12)
            n_heads = trial.suggest_categorical('n_heads', [4, 8, 16])
            
            # Update config
            self.config.learning_rate = lr
            self.config.batch_size = batch_size
            self.config.weight_decay = weight_decay
            
            # Update model config
            model_config = self.model.config
            model_config.n_layers = n_layers
            model_config.n_heads = n_heads
            
            # Recreate model and optimizer
            self.model = DegradationForecaster(model_config).to(self.device)
            self.optimizer = ForecastingOptimizer(self.model, self.config)
            
            # Train for a few epochs
            original_epochs = self.config.num_epochs
            self.config.num_epochs = 10
            
            try:
                results = self.train(train_dataloader, val_dataloader)
                return results['best_val_loss']
            finally:
                self.config.num_epochs = original_epochs
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }
    
    def generate_forecasting_report(self, test_dataloader: DataLoader) -> Dict[str, Any]:
        """
        Generate comprehensive forecasting performance report.
        
        Args:
            test_dataloader (DataLoader): Test data loader
            
        Returns:
            Dict[str, Any]: Comprehensive performance report
        """
        logger.info("Generating forecasting performance report...")
        
        # Evaluate model performance
        test_metrics = self._evaluate_forecasting_performance(test_dataloader)
        
        # Generate sample forecasts for visualization
        sample_forecasts = self._generate_sample_forecasts(test_dataloader, n_samples=5)
        
        # Analyze forecasting patterns
        pattern_analysis = self._analyze_forecasting_patterns(test_dataloader)
        
        # Create performance summary
        report = {
            'model_info': {
                'model_type': 'DegradationForecaster',
                'forecast_horizon': self.model.config.forecast_horizon,
                'uncertainty_enabled': self.model.config.enable_uncertainty,
                'seasonal_decomposition': self.model.config.enable_seasonal_decomposition
            },
            'performance_metrics': test_metrics,
            'sample_forecasts': sample_forecasts,
            'pattern_analysis': pattern_analysis,
            'training_summary': {
                'total_epochs': len(self.metrics_history),
                'best_validation_loss': self.best_val_loss,
                'final_learning_rate': self.optimizer.scheduler.get_last_lr()[0]
            }
        }
        
        # Save report
        report_path = f"{self.config.checkpoint_dir}/forecasting_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Forecasting report saved to {report_path}")
        return report
    
    def _generate_sample_forecasts(self, dataloader: DataLoader, n_samples: int = 5) -> List[Dict]:
        """Generate sample forecasts for visualization."""
        self.model.eval()
        sample_forecasts = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= n_samples:
                    break
                
                inputs = batch['inputs'].to(self.device)
                targets = batch['targets'].to(self.device)
                time_features = batch.get('time_features', {})
                
                # Generate forecast
                outputs = self.model(inputs, time_features, return_components=True)
                
                # Store sample
                sample = {
                    'input_sequence': inputs[0].cpu().numpy(),
                    'target_sequence': targets[0].cpu().numpy(),
                    'forecast': outputs['forecasts'][0].cpu().numpy(),
                    'uncertainty': outputs.get('std', torch.zeros_like(outputs['forecasts']))[0].cpu().numpy(),
                    'decomposition': {k: v[0].cpu().numpy() for k, v in outputs.get('decomposition', {}).items()}
                }
                sample_forecasts.append(sample)
        
        return sample_forecasts
    
    def _analyze_forecasting_patterns(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Analyze forecasting patterns and model behavior."""
        self.model.eval()
        
        forecast_errors = []
        seasonal_performance = {}
        horizon_performance = {}
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['inputs'].to(self.device)
                targets = batch['targets'].to(self.device)
                time_features = batch.get('time_features', {})
                
                outputs = self.model(inputs, time_features)
                forecasts = outputs['forecasts']
                
                # Calculate errors
                errors = (forecasts - targets).cpu().numpy()
                forecast_errors.append(errors)
                
                # Analyze seasonal patterns if time features available
                if time_features:
                    for period in [24, 168]:  # Daily, weekly
                        if f'seasonal_{period}' in time_features:
                            seasonal_data = time_features[f'seasonal_{period}'].cpu().numpy()
                            seasonal_bins = np.digitize(seasonal_data, np.linspace(0, period, 10))
                            
                            for bin_idx in range(1, 11):
                                mask = seasonal_bins == bin_idx
                                if mask.any():
                                    bin_errors = errors[mask]
                                    if f'seasonal_{period}' not in seasonal_performance:
                                        seasonal_performance[f'seasonal_{period}'] = {}
                                    seasonal_performance[f'seasonal_{period}'][f'bin_{bin_idx}'] = {
                                        'mae': np.mean(np.abs(bin_errors)),
                                        'mse': np.mean(bin_errors**2)
                                    }
        
        # Aggregate errors
        all_errors = np.concatenate(forecast_errors, axis=0)
        
        # Analyze horizon-specific performance
        for h in range(min(24, all_errors.shape[1])):
            horizon_errors = all_errors[:, h, :]
            horizon_performance[f'horizon_{h+1}h'] = {
                'mae': np.mean(np.abs(horizon_errors)),
                'mse': np.mean(horizon_errors**2),
                'bias': np.mean(horizon_errors)
            }
        
        return {
            'overall_statistics': {
                'mean_absolute_error': np.mean(np.abs(all_errors)),
                'mean_squared_error': np.mean(all_errors**2),
                'bias': np.mean(all_errors),
                'error_std': np.std(all_errors)
            },
            'seasonal_performance': seasonal_performance,
            'horizon_performance': horizon_performance
        }

# Specialized loss functions for forecasting
class TimeSeriesLoss(nn.Module):
    """
    Advanced loss function for time-series forecasting with multiple components.
    """
    
    def __init__(self, config: DegradationTrainingConfig):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.SmoothL1Loss()
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor,
                time_features: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss for forecasting.
        
        Args:
            outputs (Dict[str, torch.Tensor]): Model outputs
            targets (torch.Tensor): Ground truth targets
            time_features (Dict, optional): Time features
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of loss components
        """
        forecasts = outputs['forecasts']
        
        # Primary forecasting loss
        forecast_loss = self.huber_loss(forecasts, targets)
        
        # Horizon-weighted loss
        horizon_loss = self._compute_horizon_weighted_loss(forecasts, targets)
        
        # Uncertainty loss if available
        uncertainty_loss = torch.tensor(0.0, device=forecasts.device)
        if 'std' in outputs:
            uncertainty_loss = self._compute_uncertainty_loss(outputs, targets)
        
        # Physics constraints loss
        physics_loss = self._compute_physics_loss(forecasts, time_features)
        
        # Temporal consistency loss
        temporal_loss = self._compute_temporal_consistency_loss(forecasts)
        
        # Seasonal consistency loss
        seasonal_loss = self._compute_seasonal_consistency_loss(forecasts, time_features)
        
        # Combined loss
        total_loss = (forecast_loss + 
                     horizon_loss +
                     self.config.uncertainty_loss_weight * uncertainty_loss +
                     self.config.physics_loss_weight * physics_loss +
                     self.config.temporal_consistency_weight * temporal_loss +
                     self.config.seasonal_consistency_weight * seasonal_loss)
        
        return {
            'total_loss': total_loss,
            'forecast_loss': forecast_loss,
            'horizon_loss': horizon_loss,
            'uncertainty_loss': uncertainty_loss,
            'physics_loss': physics_loss,
            'temporal_loss': temporal_loss,
            'seasonal_loss': seasonal_loss
        }
    
    def _compute_horizon_weighted_loss(self, forecasts: torch.Tensor, 
                                     targets: torch.Tensor) -> torch.Tensor:
        """Compute horizon-weighted forecasting loss."""
        horizon_losses = []
        weights = self.config.forecast_loss_weights
        
        for h in range(forecasts.size(1)):
            weight = weights[h] if h < len(weights) else weights[-1]
            horizon_loss = self.mse_loss(forecasts[:, h, :], targets[:, h, :])
            horizon_losses.append(weight * horizon_loss)
        
        return torch.stack(horizon_losses).sum()
    
    def _compute_uncertainty_loss(self, outputs: Dict[str, torch.Tensor],
                                targets: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty quantification loss."""
        if 'quantiles' in outputs:
            # Quantile regression loss
            quantiles = outputs['quantiles']
            quantile_values = outputs['quantile_values']
            
            quantile_losses = []
            for i, q in enumerate(quantile_values):
                errors = targets.unsqueeze(-1) - quantiles[:, :, :, i]
                quantile_loss = torch.maximum(
                    q * errors, (q - 1) * errors
                ).mean()
                quantile_losses.append(quantile_loss)
            
            return torch.stack(quantile_losses).mean()
        
        elif 'std' in outputs:
            # Negative log-likelihood for Gaussian uncertainty
            mean = outputs['mean'] if 'mean' in outputs else outputs['forecasts']
            std = outputs['std']
            
            # Ensure std is positive
            std = torch.clamp(std, min=1e-6)
            
            # Negative log-likelihood
            nll = 0.5 * torch.log(2 * math.pi * std**2) + 0.5 * ((targets - mean) / std)**2
            return nll.mean()
        
        return torch.tensor(0.0, device=targets.device)
    
    def _compute_physics_loss(self, forecasts: torch.Tensor,
                            time_features: Optional[Dict] = None) -> torch.Tensor:
        """Compute physics-informed loss."""
        physics_loss = torch.tensor(0.0, device=forecasts.device)
        
        # Ensure non-negative degradation rates
        physics_loss += torch.mean(torch.relu(-forecasts))
        
        # Monotonic degradation constraint
        for i in range(1, forecasts.size(1)):
            violation = torch.relu(forecasts[:, i-1, :] - forecasts[:, i, :])
            physics_loss += torch.mean(violation)
        
        # Temperature dependency if available
        if time_features and 'temperature' in time_features:
            temp = time_features['temperature']
            # Higher temperatures should correlate with higher degradation
            temp_normalized = (temp - 25) / 25  # Normalize around 25°C
            expected_factor = torch.exp(temp_normalized * 0.1)
            
            # Apply to thermal degradation component if available
            if forecasts.size(-1) >= 3:
                thermal_degradation = forecasts[:, :, 2]
                expected_degradation = thermal_degradation * expected_factor.unsqueeze(-1)
                physics_loss += self.mse_loss(thermal_degradation, expected_degradation)
        
        return physics_loss
    
    def _compute_temporal_consistency_loss(self, forecasts: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency loss."""
        if forecasts.size(1) < 2:
            return torch.tensor(0.0, device=forecasts.device)
        
        # Penalize large jumps in degradation rates
        temporal_diff = torch.diff(forecasts, dim=1)
        consistency_loss = torch.mean(torch.abs(temporal_diff))
        
        return consistency_loss
    
    def _compute_seasonal_consistency_loss(self, forecasts: torch.Tensor,
                                         time_features: Optional[Dict] = None) -> torch.Tensor:
        """Compute seasonal consistency loss."""
        if time_features is None:
            return torch.tensor(0.0, device=forecasts.device)
        
        seasonal_loss = torch.tensor(0.0, device=forecasts.device)
        
        # Check for seasonal patterns in forecasts
        for period in [24, 168]:  # Daily, weekly
            if f'seasonal_{period}' in time_features:
                seasonal_data = time_features[f'seasonal_{period}']
                
                # Group forecasts by seasonal phase
                seasonal_phases = torch.floor(seasonal_data % period).long()
                unique_phases = torch.unique(seasonal_phases)
                
                for phase in unique_phases:
                    phase_mask = seasonal_phases == phase
                    if phase_mask.sum() > 1:
                        phase_forecasts = forecasts[phase_mask]
                        # Penalize high variance within same seasonal phase
                        phase_variance = torch.var(phase_forecasts, dim=0)
                        seasonal_loss += torch.mean(phase_variance)
        
        return seasonal_loss

class ForecastingOptimizer:
    """
    Advanced optimizer for forecasting models with specialized scheduling.
    """
    
    def __init__(self, model: nn.Module, config: DegradationTrainingConfig):
        self.model = model
        self.config = config
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if config.mixed_precision else None
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.config.optimizer_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.95)  # Better for forecasting
            )
        elif self.config.optimizer_type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        if self.config.scheduler_type == "cosine_with_warmup":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.scheduler_params.get('T_0', 4000),
                T_mult=self.config.scheduler_params.get('T_mult', 2),
                eta_min=self.config.scheduler_params.get('eta_min', 1e-7)
            )
        elif self.config.scheduler_type == "linear_warmup":
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
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

# Factory functions for easy instantiation
def create_degradation_trainer(model: DegradationForecaster,
                             config: Optional[DegradationTrainingConfig] = None) -> DegradationTrainer:
    """
    Factory function to create a DegradationTrainer.
    
    Args:
        model (DegradationForecaster): Model to train
        config (DegradationTrainingConfig, optional): Training configuration
        
    Returns:
        DegradationTrainer: Configured trainer instance
    """
    if config is None:
        config = DegradationTrainingConfig()
    
    return DegradationTrainer(model, config)

