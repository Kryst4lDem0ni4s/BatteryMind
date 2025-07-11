"""
BatteryMind - Optimization Recommender Trainer

Advanced training pipeline for optimization recommendation models with
multi-objective learning, recommendation ranking, and explainable AI training.

Features:
- Multi-objective optimization training
- Recommendation quality assessment and ranking
- Explainable AI training with justification loss
- Physics-informed constraint learning
- Advanced evaluation metrics for recommendation systems
- Integration with reinforcement learning for recommendation validation

Author: BatteryMind Development Team
Version: 1.0.0
"""

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

# Recommendation system specific imports
from sklearn.metrics import ndcg_score, average_precision_score
from scipy.stats import kendalltau, spearmanr
import warnings

# AWS and MLOps imports
import boto3
import mlflow
import mlflow.pytorch

# Local imports
from .model import OptimizationRecommender, OptimizationConfig, OptimizationType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationTrainingConfig:
    """
    Comprehensive configuration for optimization recommender training.
    
    Attributes:
        # Training parameters
        batch_size (int): Training batch size
        learning_rate (float): Initial learning rate
        num_epochs (int): Number of training epochs
        warmup_steps (int): Number of warmup steps
        weight_decay (float): Weight decay for regularization
        gradient_clip_norm (float): Gradient clipping norm
        
        # Recommendation-specific training
        recommendation_loss_weight (float): Weight for recommendation quality loss
        ranking_loss_weight (float): Weight for ranking loss
        diversity_loss_weight (float): Weight for diversity loss
        explanation_loss_weight (float): Weight for explanation quality loss
        
        # Multi-objective training
        objective_balance_strategy (str): Strategy for balancing multiple objectives
        objective_weights_schedule (str): Schedule for objective weight updates
        
        # Physics-informed training
        physics_constraint_weight (float): Weight for physics constraint violations
        feasibility_threshold (float): Minimum feasibility score for valid recommendations
        
        # Advanced optimization
        optimizer_type (str): Optimizer type
        scheduler_type (str): Learning rate scheduler type
        scheduler_params (Dict): Scheduler parameters
        
        # Training techniques
        mixed_precision (bool): Enable mixed precision training
        gradient_accumulation_steps (int): Gradient accumulation steps
        early_stopping_patience (int): Early stopping patience
        
        # Evaluation and validation
        validation_metrics (List[str]): Metrics for validation
        recommendation_evaluation_k (List[int]): K values for top-k evaluation
        
        # Monitoring and logging
        log_interval (int): Logging interval in steps
        eval_interval (int): Evaluation interval in steps
        save_interval (int): Model saving interval in steps
    """
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 4000
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Recommendation-specific training
    recommendation_loss_weight: float = 1.0
    ranking_loss_weight: float = 0.3
    diversity_loss_weight: float = 0.2
    explanation_loss_weight: float = 0.1
    
    # Multi-objective training
    objective_balance_strategy: str = "adaptive"
    objective_weights_schedule: str = "cosine"
    
    # Physics-informed training
    physics_constraint_weight: float = 0.1
    feasibility_threshold: float = 0.7
    
    # Advanced optimization
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine_with_warmup"
    scheduler_params: Dict = field(default_factory=dict)
    
    # Training techniques
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 15
    
    # Evaluation and validation
    validation_metrics: List[str] = field(default_factory=lambda: [
        'ndcg', 'map', 'precision', 'recall', 'diversity', 'novelty'
    ])
    recommendation_evaluation_k: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    
    # Monitoring and logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000

@dataclass
class OptimizationTrainingMetrics:
    """
    Comprehensive metrics for optimization recommender training.
    """
    epoch: int = 0
    step: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    
    # Recommendation quality metrics
    recommendation_accuracy: float = 0.0
    ranking_correlation: float = 0.0
    diversity_score: float = 0.0
    novelty_score: float = 0.0
    
    # Top-k metrics
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    map_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    
    # Physics and feasibility metrics
    feasibility_rate: float = 0.0
    physics_violation_rate: float = 0.0
    
    # Explanation quality metrics
    explanation_coherence: float = 0.0
    explanation_accuracy: float = 0.0
    
    # Training efficiency
    learning_rate: float = 0.0
    training_time: float = 0.0
    memory_usage: float = 0.0

class RecommendationLoss(nn.Module):
    """
    Advanced loss function for optimization recommendation training.
    """
    
    def __init__(self, config: OptimizationTrainingConfig):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ranking_loss = nn.MarginRankingLoss(margin=0.1)
        
    def recommendation_quality_loss(self, predictions: Dict[str, torch.Tensor],
                                  targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute recommendation quality loss."""
        # Score prediction loss
        if 'recommendation_scores' in predictions and 'target_scores' in targets:
            score_loss = self.mse_loss(
                predictions['recommendation_scores'],
                targets['target_scores']
            )
        else:
            score_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        # Feasibility prediction loss
        if 'feasibility_scores' in predictions and 'target_feasibility' in targets:
            feasibility_loss = self.bce_loss(
                predictions['feasibility_scores'],
                targets['target_feasibility']
            )
        else:
            feasibility_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        return score_loss + feasibility_loss
    
    def ranking_loss_fn(self, predictions: Dict[str, torch.Tensor],
                       targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute ranking loss for recommendation ordering."""
        if 'ranking_scores' not in predictions or 'target_rankings' not in targets:
            return torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        pred_scores = predictions['ranking_scores']
        target_rankings = targets['target_rankings']
        
        # Pairwise ranking loss
        batch_size, num_types, max_per_type = pred_scores.shape
        ranking_loss = torch.tensor(0.0, device=pred_scores.device)
        
        for i in range(max_per_type):
            for j in range(i + 1, max_per_type):
                # Get scores for pairs
                score_i = pred_scores[:, :, i]
                score_j = pred_scores[:, :, j]
                
                # Get target preferences
                target_i = target_rankings[:, :, i]
                target_j = target_rankings[:, :, j]
                
                # Compute preference (1 if i should be ranked higher, -1 otherwise)
                preference = torch.sign(target_i - target_j)
                
                # Apply margin ranking loss
                pair_loss = self.ranking_loss(score_i.flatten(), score_j.flatten(), preference.flatten())
                ranking_loss += pair_loss
        
        return ranking_loss / (max_per_type * (max_per_type - 1) / 2)
    
    def diversity_loss_fn(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute diversity loss to encourage varied recommendations."""
        if 'recommendations' not in predictions:
            return torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        recommendations = predictions['recommendations']
        batch_size, num_types, max_per_type, rec_dim = recommendations.shape
        
        diversity_loss = torch.tensor(0.0, device=recommendations.device)
        
        # Calculate pairwise similarities within each type
        for type_idx in range(num_types):
            type_recs = recommendations[:, type_idx, :, :]  # (batch_size, max_per_type, rec_dim)
            
            for i in range(max_per_type):
                for j in range(i + 1, max_per_type):
                    # Cosine similarity between recommendations
                    rec_i = type_recs[:, i, :]
                    rec_j = type_recs[:, j, :]
                    
                    similarity = F.cosine_similarity(rec_i, rec_j, dim=-1)
                    
                    # Penalize high similarity (encourage diversity)
                    diversity_loss += similarity.mean()
        
        return diversity_loss / (num_types * max_per_type * (max_per_type - 1) / 2)
    
    def explanation_loss_fn(self, predictions: Dict[str, torch.Tensor],
                          targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute explanation quality loss."""
        if 'justifications' not in predictions or 'target_explanations' not in targets:
            return torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        pred_explanations = predictions['justifications']
        target_explanations = targets['target_explanations']
        
        # Cosine similarity loss for explanation embeddings
        explanation_loss = 1 - F.cosine_similarity(
            pred_explanations.flatten(0, -2),
            target_explanations.flatten(0, -2),
            dim=-1
        ).mean()
        
        return explanation_loss
    
    def physics_constraint_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute physics constraint violation loss."""
        if 'feasibility_scores' not in predictions:
            return torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        feasibility = predictions['feasibility_scores']
        
        # Penalize infeasible recommendations
        infeasible_penalty = torch.mean(
            torch.relu(self.config.feasibility_threshold - feasibility)
        )
        
        return infeasible_penalty
    
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss for optimization recommendation training.
        
        Args:
            predictions (Dict[str, torch.Tensor]): Model predictions
            targets (Dict[str, torch.Tensor]): Ground truth targets
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of loss components
        """
        # Primary recommendation loss
        rec_loss = self.recommendation_quality_loss(predictions, targets)
        
        # Ranking loss
        rank_loss = self.ranking_loss_fn(predictions, targets)
        
        # Diversity loss
        div_loss = self.diversity_loss_fn(predictions)
        
        # Explanation loss
        exp_loss = self.explanation_loss_fn(predictions, targets)
        
        # Physics constraint loss
        physics_loss = self.physics_constraint_loss(predictions)
        
        # Combined loss
        total_loss = (
            self.config.recommendation_loss_weight * rec_loss +
            self.config.ranking_loss_weight * rank_loss +
            self.config.diversity_loss_weight * div_loss +
            self.config.explanation_loss_weight * exp_loss +
            self.config.physics_constraint_weight * physics_loss
        )
        
        return {
            'total_loss': total_loss,
            'recommendation_loss': rec_loss,
            'ranking_loss': rank_loss,
            'diversity_loss': div_loss,
            'explanation_loss': exp_loss,
            'physics_loss': physics_loss
        }

class RecommendationOptimizer:
    """
    Advanced optimizer for optimization recommender training.
    """
    
    def __init__(self, model: nn.Module, config: OptimizationTrainingConfig):
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
                betas=(0.9, 0.999)
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

class OptimizationTrainer:
    """
    Comprehensive training pipeline for optimization recommendation models.
    """
    
    def __init__(self, model: OptimizationRecommender, config: OptimizationTrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize training components
        self.loss_function = RecommendationLoss(config)
        self.optimizer = RecommendationOptimizer(model, config)
        
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
        
        logger.info(f"OptimizationTrainer initialized on device: {self.device}")
    
    def _setup_directories(self) -> None:
        """Setup necessary directories for training."""
        Path("./checkpoints").mkdir(parents=True, exist_ok=True)
        Path("./checkpoints/logs").mkdir(parents=True, exist_ok=True)
        Path("./checkpoints/metrics").mkdir(parents=True, exist_ok=True)
    
    def _setup_experiment_tracking(self) -> None:
        """Setup experiment tracking."""
        # MLflow setup
        mlflow.set_experiment("optimization_recommendation")
        mlflow.start_run()
        mlflow.log_params(self.config.__dict__)
        
        # Weights & Biases setup
        try:
            wandb.init(
                project="batterymind-optimization",
                name=f"optimization_recommender_{int(time.time())}",
                config=self.config.__dict__
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Weights & Biases: {e}")
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
              test_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Main training loop for optimization recommender.
        
        Args:
            train_dataloader (DataLoader): Training data loader
            val_dataloader (DataLoader): Validation data loader
            test_dataloader (DataLoader, optional): Test data loader
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        logger.info("Starting optimization recommender training...")
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(train_dataloader, epoch)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_dataloader, epoch)
            
            # Combine metrics
            epoch_metrics = OptimizationTrainingMetrics(
                epoch=epoch,
                step=epoch * len(train_dataloader),
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                recommendation_accuracy=val_metrics.get('recommendation_accuracy', 0.0),
                ranking_correlation=val_metrics.get('ranking_correlation', 0.0),
                diversity_score=val_metrics.get('diversity_score', 0.0),
                feasibility_rate=val_metrics.get('feasibility_rate', 0.0),
                learning_rate=self.optimizer.scheduler.get_last_lr()[0],
                training_time=time.time() - epoch_start_time
            )
            
            # Add top-k metrics
            for k in self.config.recommendation_evaluation_k:
                if f'ndcg_at_{k}' in val_metrics:
                    epoch_metrics.ndcg_at_k[k] = val_metrics[f'ndcg_at_{k}']
                if f'map_at_{k}' in val_metrics:
                    epoch_metrics.map_at_k[k] = val_metrics[f'map_at_{k}']
            
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
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in batch['targets'].items()}
            context = batch.get('context', None)
            if context is not None:
                context = context.to(self.device)
            
            # Forward pass
            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(inputs, context, return_explanations=True)
                    loss_dict = self.loss_function(outputs, targets)
                    loss = loss_dict['total_loss']
            else:
                outputs = self.model(inputs, context, return_explanations=True)
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
        """Validate for one epoch with recommendation metrics."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Validation Epoch {epoch}"):
                inputs = batch['inputs'].to(self.device)
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch['targets'].items()}
                context = batch.get('context', None)
                if context is not None:
                    context = context.to(self.device)
                
                outputs = self.model(inputs, context, return_explanations=True)
                loss_dict = self.loss_function(outputs, targets)
                
                total_loss += loss_dict['total_loss'].item() * inputs.size(0)
                total_samples += inputs.size(0)
                
                # Store predictions for detailed metrics
                all_predictions.append({
                    'recommendations': outputs['recommendations'].cpu().numpy(),
                    'scores': outputs['recommendation_scores'].cpu().numpy(),
                    'feasibility': outputs['feasibility_scores'].cpu().numpy()
                })
                all_targets.append({
                    'target_scores': targets.get('target_scores', torch.zeros_like(outputs['recommendation_scores'])).cpu().numpy(),
                    'target_feasibility': targets.get('target_feasibility', torch.zeros_like(outputs['feasibility_scores'])).cpu().numpy()
                })
        
        # Compute recommendation metrics
        metrics = self._compute_recommendation_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / total_samples
        
        return metrics
    
    def _compute_recommendation_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """Compute comprehensive recommendation metrics."""
        metrics = {}
        
        # Aggregate predictions and targets
        all_pred_scores = np.concatenate([p['scores'] for p in predictions], axis=0)
        all_target_scores = np.concatenate([t['target_scores'] for t in targets], axis=0)
        all_feasibility = np.concatenate([p['feasibility'] for p in predictions], axis=0)
        
        # Recommendation accuracy (correlation between predicted and target scores)
        if all_target_scores.size > 0:
            correlation, _ = spearmanr(all_pred_scores.flatten(), all_target_scores.flatten())
            metrics['recommendation_accuracy'] = correlation if not np.isnan(correlation) else 0.0
        
        # Feasibility rate
        metrics['feasibility_rate'] = np.mean(all_feasibility >= self.config.feasibility_threshold)
        
        # Diversity score (average pairwise distance within recommendations)
        diversity_scores = []
        for pred in predictions:
            recs = pred['recommendations']
            batch_size, num_types, max_per_type, rec_dim = recs.shape
            
            for b in range(batch_size):
                for t in range(num_types):
                    type_recs = recs[b, t, :, :]
                    pairwise_distances = []
                    
                    for i in range(max_per_type):
                        for j in range(i + 1, max_per_type):
                            dist = np.linalg.norm(type_recs[i] - type_recs[j])
                            pairwise_distances.append(dist)
                    
                    if pairwise_distances:
                        diversity_scores.append(np.mean(pairwise_distances))
        
        metrics['diversity_score'] = np.mean(diversity_scores) if diversity_scores else 0.0
        
        # Top-k metrics
        for k in self.config.recommendation_evaluation_k:
            if k <= all_pred_scores.shape[-1]:
                # NDCG@k
                try:
                    ndcg_scores = []
                    for i in range(len(all_pred_scores)):
                        pred_scores = all_pred_scores[i].flatten()
                        target_scores = all_target_scores[i].flatten()
                        
                        if len(pred_scores) >= k and len(target_scores) >= k:
                            ndcg = ndcg_score([target_scores], [pred_scores], k=k)
                            ndcg_scores.append(ndcg)
                    
                    metrics[f'ndcg_at_{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
                except:
                    metrics[f'ndcg_at_{k}'] = 0.0
                
                # MAP@k (simplified)
                try:
                    map_scores = []
                    for i in range(len(all_pred_scores)):
                        pred_scores = all_pred_scores[i].flatten()
                        target_scores = all_target_scores[i].flatten()
                        
                        if len(pred_scores) >= k and len(target_scores) >= k:
                            # Get top-k predictions
                            top_k_indices = np.argsort(pred_scores)[-k:]
                            relevant = target_scores[top_k_indices] > 0.5  # Binary relevance
                            
                            if np.sum(relevant) > 0:
                                precision_at_i = []
                                for j in range(k):
                                    if relevant[j]:
                                        precision_at_i.append(np.sum(relevant[:j+1]) / (j+1))
                                
                                map_scores.append(np.mean(precision_at_i))
                    
                                metrics[f'map_at_{k}'] = np.mean(map_scores) if map_scores else 0.0
                except:
                    metrics[f'map_at_{k}'] = 0.0
        
        # Business impact metrics
        if len(predictions) > 0:
            # Energy efficiency improvement
            energy_improvements = []
            for pred in predictions:
                if 'energy_efficiency' in pred:
                    energy_improvements.append(pred['energy_efficiency'])
            
            if energy_improvements:
                metrics['avg_energy_improvement'] = np.mean(energy_improvements)
                metrics['max_energy_improvement'] = np.max(energy_improvements)
        
            # Battery life extension
            life_extensions = []
            for pred in predictions:
                if 'life_extension' in pred:
                    life_extensions.append(pred['life_extension'])
            
            if life_extensions:
                metrics['avg_life_extension'] = np.mean(life_extensions)
                metrics['max_life_extension'] = np.max(life_extensions)
        
        # Safety compliance rate
        safety_compliance = []
        for pred in predictions:
            if 'safety_score' in pred:
                safety_compliance.append(pred['safety_score'] >= self.config.safety_threshold)
        
        if safety_compliance:
            metrics['safety_compliance_rate'] = np.mean(safety_compliance)
        
        return metrics
    
    def _log_metrics(self, metrics: OptimizationTrainingMetrics) -> None:
        """Log metrics to various tracking systems."""
        # MLflow logging
        mlflow.log_metrics({
            'train_loss': metrics.train_loss,
            'val_loss': metrics.val_loss,
            'learning_rate': metrics.learning_rate,
            'recommendation_accuracy': metrics.recommendation_accuracy,
            'feasibility_rate': metrics.feasibility_rate,
            'diversity_score': metrics.diversity_score,
            'safety_compliance_rate': metrics.safety_compliance_rate,
            'energy_improvement': metrics.energy_improvement,
            'life_extension': metrics.life_extension
        }, step=metrics.epoch)
        
        # Weights & Biases logging
        try:
            wandb.log({
                'epoch': metrics.epoch,
                'train_loss': metrics.train_loss,
                'val_loss': metrics.val_loss,
                'learning_rate': metrics.learning_rate,
                'recommendation_accuracy': metrics.recommendation_accuracy,
                'feasibility_rate': metrics.feasibility_rate,
                'diversity_score': metrics.diversity_score,
                'safety_compliance_rate': metrics.safety_compliance_rate,
                'energy_improvement': metrics.energy_improvement,
                'life_extension': metrics.life_extension
            })
        except:
            pass
    
    def _log_batch_metrics(self, epoch: int, batch_idx: int, loss_dict: Dict[str, torch.Tensor]) -> None:
        """Log batch-level metrics."""
        step = epoch * 1000 + batch_idx
        
        mlflow.log_metrics({
            'batch_total_loss': loss_dict['total_loss'].item(),
            'batch_recommendation_loss': loss_dict['recommendation_loss'].item(),
            'batch_feasibility_loss': loss_dict['feasibility_loss'].item(),
            'batch_diversity_loss': loss_dict['diversity_loss'].item(),
            'batch_safety_loss': loss_dict['safety_loss'].item()
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
        torch.save(checkpoint, "./checkpoints/latest_checkpoint.ckpt")
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, "./checkpoints/best_model.ckpt")
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # Save epoch checkpoint
        torch.save(checkpoint, f"./checkpoints/epoch_{epoch:03d}.ckpt")
    
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
        
        # Additional comprehensive evaluation
        comprehensive_metrics = self._comprehensive_evaluation(test_dataloader)
        test_metrics.update(comprehensive_metrics)
        
        # Log final test metrics
        mlflow.log_metrics({
            'test_loss': test_metrics['loss'],
            'test_recommendation_accuracy': test_metrics.get('recommendation_accuracy', 0.0),
            'test_feasibility_rate': test_metrics.get('feasibility_rate', 0.0),
            'test_diversity_score': test_metrics.get('diversity_score', 0.0),
            'test_safety_compliance_rate': test_metrics.get('safety_compliance_rate', 0.0),
            'test_energy_improvement': test_metrics.get('avg_energy_improvement', 0.0),
            'test_life_extension': test_metrics.get('avg_life_extension', 0.0)
        })
        
        return test_metrics
    
    def _comprehensive_evaluation(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Perform comprehensive evaluation with detailed analysis."""
        self.model.eval()
        
        all_recommendations = []
        all_targets = []
        all_metadata = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Comprehensive Evaluation"):
                inputs = batch['inputs'].to(self.device)
                targets = batch['targets'].to(self.device)
                metadata = batch.get('metadata', {})
                
                # Generate recommendations
                outputs = self.model(inputs, metadata)
                
                # Store for analysis
                all_recommendations.append(outputs)
                all_targets.append(targets)
                all_metadata.append(metadata)
        
        # Analyze recommendation patterns
        pattern_analysis = self._analyze_recommendation_patterns(all_recommendations, all_metadata)
        
        # Evaluate business impact
        business_impact = self._evaluate_business_impact(all_recommendations, all_targets)
        
        # Safety analysis
        safety_analysis = self._analyze_safety_compliance(all_recommendations)
        
        # Diversity analysis
        diversity_analysis = self._analyze_recommendation_diversity(all_recommendations)
        
        return {
            'pattern_analysis': pattern_analysis,
            'business_impact': business_impact,
            'safety_analysis': safety_analysis,
            'diversity_analysis': diversity_analysis
        }
    
    def _analyze_recommendation_patterns(self, recommendations: List[Dict], 
                                       metadata: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in recommendations."""
        analysis = {
            'most_common_recommendations': {},
            'recommendation_frequency': {},
            'context_sensitivity': {}
        }
        
        # Aggregate recommendation types
        rec_types = {}
        for rec_batch in recommendations:
            if 'recommendation_types' in rec_batch:
                types = rec_batch['recommendation_types'].cpu().numpy()
                for type_id in types.flatten():
                    rec_types[type_id] = rec_types.get(type_id, 0) + 1
        
        analysis['recommendation_frequency'] = rec_types
        
        # Find most common recommendations
        if rec_types:
            sorted_types = sorted(rec_types.items(), key=lambda x: x[1], reverse=True)
            analysis['most_common_recommendations'] = dict(sorted_types[:10])
        
        # Analyze context sensitivity (how recommendations change with context)
        context_variations = {}
        for i, (rec_batch, meta_batch) in enumerate(zip(recommendations, metadata)):
            if isinstance(meta_batch, dict) and 'battery_type' in meta_batch:
                battery_type = meta_batch['battery_type']
                if battery_type not in context_variations:
                    context_variations[battery_type] = []
                
                if 'recommendation_types' in rec_batch:
                    types = rec_batch['recommendation_types'].cpu().numpy()
                    context_variations[battery_type].extend(types.flatten().tolist())
        
        # Calculate context sensitivity score
        if len(context_variations) > 1:
            type_distributions = {}
            for context, types in context_variations.items():
                type_dist = {}
                for t in types:
                    type_dist[t] = type_dist.get(t, 0) + 1
                type_distributions[context] = type_dist
            
            # Calculate Jensen-Shannon divergence between distributions
            analysis['context_sensitivity']['distributions'] = type_distributions
        
        return analysis
    
    def _evaluate_business_impact(self, recommendations: List[Dict], 
                                targets: List[Dict]) -> Dict[str, Any]:
        """Evaluate business impact of recommendations."""
        impact = {
            'energy_savings': [],
            'cost_reductions': [],
            'life_extensions': [],
            'efficiency_improvements': []
        }
        
        for rec_batch in recommendations:
            if 'business_impact' in rec_batch:
                business_data = rec_batch['business_impact']
                
                if 'energy_savings' in business_data:
                    impact['energy_savings'].extend(business_data['energy_savings'].cpu().numpy().flatten())
                
                if 'cost_reductions' in business_data:
                    impact['cost_reductions'].extend(business_data['cost_reductions'].cpu().numpy().flatten())
                
                if 'life_extensions' in business_data:
                    impact['life_extensions'].extend(business_data['life_extensions'].cpu().numpy().flatten())
                
                if 'efficiency_improvements' in business_data:
                    impact['efficiency_improvements'].extend(business_data['efficiency_improvements'].cpu().numpy().flatten())
        
        # Calculate statistics
        impact_stats = {}
        for metric, values in impact.items():
            if values:
                impact_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'percentile_95': np.percentile(values, 95)
                }
        
        return impact_stats
    
    def _analyze_safety_compliance(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """Analyze safety compliance of recommendations."""
        safety_analysis = {
            'overall_compliance_rate': 0.0,
            'violation_types': {},
            'risk_distribution': {}
        }
        
        all_safety_scores = []
        violation_counts = {}
        
        for rec_batch in recommendations:
            if 'safety_scores' in rec_batch:
                safety_scores = rec_batch['safety_scores'].cpu().numpy()
                all_safety_scores.extend(safety_scores.flatten())
                
                # Count violations
                violations = safety_scores < self.config.safety_threshold
                violation_counts['thermal_violations'] = violation_counts.get('thermal_violations', 0) + np.sum(violations)
            
            if 'safety_violations' in rec_batch:
                violations = rec_batch['safety_violations']
                for violation_type, count in violations.items():
                    violation_counts[violation_type] = violation_counts.get(violation_type, 0) + count
        
        # Calculate overall compliance rate
        if all_safety_scores:
            compliant = np.array(all_safety_scores) >= self.config.safety_threshold
            safety_analysis['overall_compliance_rate'] = np.mean(compliant)
            
            # Risk distribution
            safety_analysis['risk_distribution'] = {
                'low_risk': np.mean(np.array(all_safety_scores) >= 0.8),
                'medium_risk': np.mean((np.array(all_safety_scores) >= 0.6) & (np.array(all_safety_scores) < 0.8)),
                'high_risk': np.mean(np.array(all_safety_scores) < 0.6)
            }
        
        safety_analysis['violation_types'] = violation_counts
        
        return safety_analysis
    
    def _analyze_recommendation_diversity(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """Analyze diversity of recommendations."""
        diversity_analysis = {
            'intra_batch_diversity': [],
            'inter_batch_diversity': 0.0,
            'coverage_metrics': {}
        }
        
        all_recommendations = []
        
        for rec_batch in recommendations:
            if 'recommendations' in rec_batch:
                recs = rec_batch['recommendations'].cpu().numpy()
                batch_size, num_types, max_per_type, rec_dim = recs.shape
                
                # Calculate intra-batch diversity
                batch_diversity = []
                for b in range(batch_size):
                    batch_recs = recs[b].reshape(-1, rec_dim)
                    
                    # Calculate pairwise distances
                    distances = []
                    for i in range(len(batch_recs)):
                        for j in range(i + 1, len(batch_recs)):
                            dist = np.linalg.norm(batch_recs[i] - batch_recs[j])
                            distances.append(dist)
                    
                    if distances:
                        batch_diversity.append(np.mean(distances))
                
                if batch_diversity:
                    diversity_analysis['intra_batch_diversity'].extend(batch_diversity)
                
                # Store for inter-batch analysis
                all_recommendations.append(recs)
        
        # Calculate inter-batch diversity
        if len(all_recommendations) > 1:
            # Sample recommendations from different batches
            sample_size = min(100, len(all_recommendations))
            sampled_recs = []
            
            for i in range(sample_size):
                batch_idx = i % len(all_recommendations)
                rec_batch = all_recommendations[batch_idx]
                
                # Sample random recommendation from batch
                b_idx = np.random.randint(0, rec_batch.shape[0])
                t_idx = np.random.randint(0, rec_batch.shape[1])
                r_idx = np.random.randint(0, rec_batch.shape[2])
                
                sampled_recs.append(rec_batch[b_idx, t_idx, r_idx])
            
            # Calculate inter-batch diversity
            if len(sampled_recs) > 1:
                inter_distances = []
                for i in range(len(sampled_recs)):
                    for j in range(i + 1, len(sampled_recs)):
                        dist = np.linalg.norm(sampled_recs[i] - sampled_recs[j])
                        inter_distances.append(dist)
                
                diversity_analysis['inter_batch_diversity'] = np.mean(inter_distances)
        
        # Calculate coverage metrics
        if all_recommendations:
            # Estimate coverage of recommendation space
            all_flat_recs = np.concatenate([r.reshape(-1, r.shape[-1]) for r in all_recommendations], axis=0)
            
            # Calculate range coverage for each dimension
            coverage_per_dim = []
            for dim in range(all_flat_recs.shape[1]):
                dim_range = np.max(all_flat_recs[:, dim]) - np.min(all_flat_recs[:, dim])
                coverage_per_dim.append(dim_range)
            
            diversity_analysis['coverage_metrics'] = {
                'avg_dimension_coverage': np.mean(coverage_per_dim),
                'min_dimension_coverage': np.min(coverage_per_dim),
                'max_dimension_coverage': np.max(coverage_per_dim)
            }
        
        return diversity_analysis
    
    def _cleanup_training(self) -> None:
        """Cleanup training resources."""
        mlflow.end_run()
        try:
            wandb.finish()
        except:
            pass
    
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
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            
            # Update config
            self.config.learning_rate = lr
            self.config.batch_size = batch_size
            self.config.weight_decay = weight_decay
            
            # Update model dropout
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout
            
            # Recreate optimizer with new hyperparameters
            self.optimizer = OptimizationOptimizer(self.model, self.config)
            
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
    
    def generate_recommendation_report(self, test_dataloader: DataLoader) -> Dict[str, Any]:
        """
        Generate comprehensive recommendation performance report.
        
        Args:
            test_dataloader (DataLoader): Test data for evaluation
            
        Returns:
            Dict[str, Any]: Comprehensive performance report
        """
        logger.info("Generating comprehensive recommendation report...")
        
        # Perform evaluation
        test_metrics = self._final_evaluation(test_dataloader)
        
        # Generate visualizations
        visualizations = self._generate_visualizations(test_dataloader)
        
        # Create summary report
        report = {
            'model_info': {
                'model_type': 'OptimizationRecommender',
                'version': '1.0.0',
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'training_config': self.config.__dict__
            },
            'performance_metrics': test_metrics,
            'visualizations': visualizations,
            'recommendations': {
                'strengths': self._identify_model_strengths(test_metrics),
                'weaknesses': self._identify_model_weaknesses(test_metrics),
                'improvement_suggestions': self._suggest_improvements(test_metrics)
            }
        }
        
        # Save report
        report_path = f"./checkpoints/recommendation_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_report = self._convert_numpy_types(report)
            json.dump(json_report, f, indent=2)
        
        logger.info(f"Recommendation report saved to {report_path}")
        
        return report
    
    def _generate_visualizations(self, dataloader: DataLoader) -> Dict[str, str]:
        """Generate visualization plots for the report."""
        visualizations = {}
        
        try:
            # Training curves
            if self.metrics_history:
                plt.figure(figsize=(12, 8))
                
                epochs = [m.epoch for m in self.metrics_history]
                train_losses = [m.train_loss for m in self.metrics_history]
                val_losses = [m.val_loss for m in self.metrics_history]
                
                plt.subplot(2, 2, 1)
                plt.plot(epochs, train_losses, label='Train Loss')
                plt.plot(epochs, val_losses, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Curves')
                plt.legend()
                
                # Recommendation accuracy
                rec_accuracies = [m.recommendation_accuracy for m in self.metrics_history]
                plt.subplot(2, 2, 2)
                plt.plot(epochs, rec_accuracies)
                plt.xlabel('Epoch')
                plt.ylabel('Recommendation Accuracy')
                plt.title('Recommendation Accuracy Over Time')
                
                # Feasibility rate
                feasibility_rates = [m.feasibility_rate for m in self.metrics_history]
                plt.subplot(2, 2, 3)
                plt.plot(epochs, feasibility_rates)
                plt.xlabel('Epoch')
                plt.ylabel('Feasibility Rate')
                plt.title('Feasibility Rate Over Time')
                
                # Diversity score
                diversity_scores = [m.diversity_score for m in self.metrics_history]
                plt.subplot(2, 2, 4)
                plt.plot(epochs, diversity_scores)
                plt.xlabel('Epoch')
                plt.ylabel('Diversity Score')
                plt.title('Recommendation Diversity Over Time')
                
                plt.tight_layout()
                
                viz_path = "./checkpoints/training_curves.png"
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                visualizations['training_curves'] = viz_path
        
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")
        
        return visualizations
    
    def _identify_model_strengths(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify model strengths based on metrics."""
        strengths = []
        
        if metrics.get('recommendation_accuracy', 0) > 0.8:
            strengths.append("High recommendation accuracy")
        
        if metrics.get('feasibility_rate', 0) > 0.9:
            strengths.append("Excellent feasibility rate")
        
        if metrics.get('safety_compliance_rate', 0) > 0.95:
            strengths.append("Strong safety compliance")
        
        if metrics.get('diversity_score', 0) > 0.7:
            strengths.append("Good recommendation diversity")
        
        if metrics.get('avg_energy_improvement', 0) > 0.15:
            strengths.append("Significant energy efficiency improvements")
        
        return strengths
    
    def _identify_model_weaknesses(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify model weaknesses based on metrics."""
        weaknesses = []
        
        if metrics.get('recommendation_accuracy', 0) < 0.6:
            weaknesses.append("Low recommendation accuracy")
        
        if metrics.get('feasibility_rate', 0) < 0.7:
            weaknesses.append("Poor feasibility rate")
        
        if metrics.get('safety_compliance_rate', 0) < 0.9:
            weaknesses.append("Safety compliance concerns")
        
        if metrics.get('diversity_score', 0) < 0.4:
            weaknesses.append("Limited recommendation diversity")
        
        return weaknesses
    
    def _suggest_improvements(self, metrics: Dict[str, Any]) -> List[str]:
        """Suggest improvements based on metrics."""
        suggestions = []
        
        if metrics.get('recommendation_accuracy', 0) < 0.7:
            suggestions.append("Consider increasing model capacity or training data")
        
        if metrics.get('feasibility_rate', 0) < 0.8:
            suggestions.append("Strengthen feasibility constraints in loss function")
        
        if metrics.get('diversity_score', 0) < 0.5:
            suggestions.append("Increase diversity loss weight or implement diversity regularization")
        
        if metrics.get('safety_compliance_rate', 0) < 0.95:
            suggestions.append("Enhance safety constraints and validation")
        
        suggestions.append("Consider ensemble methods for improved robustness")
        suggestions.append("Implement active learning for continuous improvement")
        
        return suggestions
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

# Factory functions for easy instantiation
def create_optimization_trainer(model, config: Optional[OptimizationTrainingConfig] = None) -> OptimizationTrainer:
    """
    Factory function to create an OptimizationTrainer.
    
    Args:
        model: OptimizationRecommender model to train
        config (OptimizationTrainingConfig, optional): Training configuration
        
    Returns:
        OptimizationTrainer: Configured trainer instance
    """
    if config is None:
        config = OptimizationTrainingConfig()
    
    return OptimizationTrainer(model, config)

def train_optimization_model_with_validation(model, train_data: DataLoader, 
                                           val_data: DataLoader,
                                           config: Optional[OptimizationTrainingConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to train optimization model with comprehensive validation.
    
    Args:
        model: OptimizationRecommender model
        train_data (DataLoader): Training data
        val_data (DataLoader): Validation data
        config (OptimizationTrainingConfig, optional): Training configuration
        
    Returns:
        Dict[str, Any]: Training results and comprehensive evaluation
    """
    trainer = create_optimization_trainer(model, config)
    
    # Train the model
    training_results = trainer.train(train_data, val_data)
    
    # Generate comprehensive report
    report = trainer.generate_recommendation_report(val_data)
    
    return {
        'training_results': training_results,
        'evaluation_report': report,
        'trainer': trainer
    }

