"""
BatteryMind - Model Fusion

Advanced model fusion techniques for combining heterogeneous models
with adaptive weighting, attention mechanisms, and uncertainty handling.

Features:
- Attention-based model fusion with learned weights
- Dynamic model selection based on input characteristics
- Uncertainty-aware fusion with confidence weighting
- Multi-modal fusion for different data types
- Real-time adaptation to model performance
- Physics-informed fusion constraints

Author: BatteryMind Development Team
Version: 1.0.0
"""

from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import time
import pickle
from pathlib import Path
from abc import ABC, abstractmethod

# Scientific computing imports
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from scipy.spatial.distance import cosine
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FusionConfig:
    """
    Configuration for model fusion ensemble.
    
    Attributes:
        # Fusion method
        fusion_method (str): Type of fusion ('attention', 'weighted', 'gating', 'adaptive')
        
        # Attention mechanism
        attention_dim (int): Dimension for attention mechanism
        attention_heads (int): Number of attention heads
        
        # Dynamic weighting
        use_dynamic_weights (bool): Enable dynamic weight adjustment
        weight_update_frequency (int): Frequency of weight updates
        
        # Uncertainty handling
        uncertainty_weighting (bool): Weight models by uncertainty
        confidence_threshold (float): Minimum confidence threshold
        
        # Multi-modal fusion
        modal_fusion_strategy (str): Strategy for multi-modal fusion
        modal_attention (bool): Use attention across modalities
        
        # Performance adaptation
        adaptive_learning_rate (float): Learning rate for adaptation
        performance_window_size (int): Window size for performance tracking
        
        # Physics constraints
        enforce_physics_constraints (bool): Enforce physics-based constraints
        constraint_violation_penalty (float): Penalty for constraint violations
    """
    # Fusion method
    fusion_method: str = "attention"
    
    # Attention mechanism
    attention_dim: int = 128
    attention_heads: int = 4
    
    # Dynamic weighting
    use_dynamic_weights: bool = True
    weight_update_frequency: int = 100
    
    # Uncertainty handling
    uncertainty_weighting: bool = True
    confidence_threshold: float = 0.5
    
    # Multi-modal fusion
    modal_fusion_strategy: str = "hierarchical"
    modal_attention: bool = True
    
    # Performance adaptation
    adaptive_learning_rate: float = 0.001
    performance_window_size: int = 1000
    
    # Physics constraints
    enforce_physics_constraints: bool = True
    constraint_violation_penalty: float = 0.1

class AttentionFusion(nn.Module):
    """
    Attention-based fusion mechanism for combining model predictions.
    """
    
    def __init__(self, n_models: int, feature_dim: int, config: FusionConfig):
        super().__init__()
        self.n_models = n_models
        self.feature_dim = feature_dim
        self.config = config
        
        # Multi-head attention for model fusion
        self.model_attention = nn.MultiheadAttention(
            embed_dim=config.attention_dim,
            num_heads=config.attention_heads,
            batch_first=True
        )
        
        # Projection layers for each model
        self.model_projections = nn.ModuleList([
            nn.Linear(feature_dim, config.attention_dim)
            for _ in range(n_models)
        ])
        
        # Context encoder for input features
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim, config.attention_dim),
            nn.ReLU(),
            nn.Linear(config.attention_dim, config.attention_dim)
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.attention_dim, feature_dim)
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.attention_dim, config.attention_dim // 2),
            nn.ReLU(),
            nn.Linear(config.attention_dim // 2, feature_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
    def forward(self, model_predictions: List[torch.Tensor], 
                input_features: torch.Tensor,
                model_uncertainties: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Fuse model predictions using attention mechanism.
        
        Args:
            model_predictions (List[torch.Tensor]): Predictions from each model
            input_features (torch.Tensor): Original input features
            model_uncertainties (List[torch.Tensor], optional): Uncertainty estimates
            
        Returns:
            Dict[str, torch.Tensor]: Fused predictions and attention weights
        """
        batch_size = model_predictions[0].shape[0]
        
        # Project model predictions to attention dimension
        projected_predictions = []
        for i, pred in enumerate(model_predictions):
            proj_pred = self.model_projections[i](pred)
            projected_predictions.append(proj_pred)
        
        # Stack predictions for attention
        model_stack = torch.stack(projected_predictions, dim=1)  # (batch, n_models, attention_dim)
        
        # Encode input context
        context = self.context_encoder(input_features).unsqueeze(1)  # (batch, 1, attention_dim)
        
        # Apply attention
        fused_features, attention_weights = self.model_attention(
            query=context,
            key=model_stack,
            value=model_stack
        )
        
        # Generate final prediction
        fused_prediction = self.output_projection(fused_features.squeeze(1))
        
        # Estimate uncertainty
        fused_uncertainty = self.uncertainty_head(fused_features.squeeze(1))
        
        # Incorporate model uncertainties if provided
        if model_uncertainties is not None and self.config.uncertainty_weighting:
            # Weight by inverse uncertainty
            uncertainty_weights = []
            for unc in model_uncertainties:
                inv_unc = 1.0 / (unc + 1e-6)  # Add small epsilon to avoid division by zero
                uncertainty_weights.append(inv_unc)
            
            uncertainty_weights = torch.stack(uncertainty_weights, dim=1)
            uncertainty_weights = F.softmax(uncertainty_weights, dim=1)
            
            # Combine with attention weights
            combined_weights = attention_weights.squeeze(1) * uncertainty_weights
            combined_weights = F.softmax(combined_weights, dim=1)
            
            # Re-weight predictions
            weighted_predictions = []
            for i, pred in enumerate(model_predictions):
                weighted_pred = pred * combined_weights[:, i:i+1]
                weighted_predictions.append(weighted_pred)
            
            fused_prediction = torch.sum(torch.stack(weighted_predictions, dim=1), dim=1)
        
        return {
            'prediction': fused_prediction,
            'uncertainty': fused_uncertainty,
            'attention_weights': attention_weights.squeeze(1),
            'model_contributions': attention_weights.squeeze(1)
        }

class GatingNetwork(nn.Module):
    """
    Gating network for dynamic model selection based on input characteristics.
    """
    
    def __init__(self, input_dim: int, n_models: int, config: FusionConfig):
        super().__init__()
        self.n_models = n_models
        self.config = config
        
        # Feature extractor for gating decisions
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, config.attention_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.attention_dim, config.attention_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Gating heads for each model
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.attention_dim // 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in range(n_models)
        ])
        
        # Sparsity regularization
        self.sparsity_weight = config.constraint_violation_penalty
        
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Compute gating weights for each model.
        
        Args:
            input_features (torch.Tensor): Input features for gating decision
            
        Returns:
            torch.Tensor: Gating weights for each model
        """
        # Extract features for gating
        features = self.feature_extractor(input_features)
        
        # Compute gates for each model
        gates = []
        for gate in self.gates:
            gate_weight = gate(features)
            gates.append(gate_weight)
        
        # Stack and normalize gates
        gate_weights = torch.cat(gates, dim=1)  # (batch, n_models)
        
        # Apply softmax for probability distribution
        gate_weights = F.softmax(gate_weights, dim=1)
        
        return gate_weights
    
    def compute_sparsity_loss(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """Compute sparsity regularization loss."""
        # Encourage sparse gating (few active models)
        entropy_loss = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=1)
        sparsity_loss = torch.mean(entropy_loss)
        return self.sparsity_weight * sparsity_loss

class AdaptiveWeighting(nn.Module):
    """
    Adaptive weighting mechanism that adjusts based on recent performance.
    """
    
    def __init__(self, n_models: int, config: FusionConfig):
        super().__init__()
        self.n_models = n_models
        self.config = config
        
        # Initialize equal weights
        self.register_buffer('model_weights', torch.ones(n_models) / n_models)
        self.register_buffer('performance_history', torch.zeros(n_models, config.performance_window_size))
        self.register_buffer('update_counter', torch.tensor(0))
        
        # Learning rate for weight adaptation
        self.lr = config.adaptive_learning_rate
        
    def update_performance(self, model_errors: torch.Tensor):
        """
        Update performance history with recent model errors.
        
        Args:
            model_errors (torch.Tensor): Recent errors for each model
        """
        # Shift history and add new errors
        self.performance_history = torch.roll(self.performance_history, 1, dims=1)
        self.performance_history[:, 0] = model_errors
        
        # Update counter
        self.update_counter += 1
        
        # Update weights if enough samples collected
        if self.update_counter % self.config.weight_update_frequency == 0:
            self._update_weights()
    
    def _update_weights(self):
        """Update model weights based on performance history."""
        # Calculate average performance (lower error = better performance)
        avg_errors = torch.mean(self.performance_history, dim=1)
        
        # Convert errors to weights (inverse relationship)
        inv_errors = 1.0 / (avg_errors + 1e-6)
        new_weights = inv_errors / torch.sum(inv_errors)
        
        # Exponential moving average update
        alpha = self.lr
        self.model_weights = (1 - alpha) * self.model_weights + alpha * new_weights
    
    def forward(self) -> torch.Tensor:
        """Get current model weights."""
        return self.model_weights

class ModelFusion:
    """
    Advanced model fusion system for battery health prediction ensemble.
    """
    
    def __init__(self, models: List[Any], config: FusionConfig):
        self.models = models
        self.config = config
        self.n_models = len(models)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize fusion components
        self._initialize_fusion_components()
        
        # Performance tracking
        self.performance_tracker = {}
        self.prediction_history = []
        
        logger.info(f"Initialized ModelFusion with {self.n_models} models using {config.fusion_method} method")
    
    def _initialize_fusion_components(self):
        """Initialize fusion mechanism components."""
        # Determine feature dimension from first model
        sample_input = torch.randn(1, 16)  # Assuming 16 input features
        sample_output = torch.randn(1, 4)   # Assuming 4 output features
        
        feature_dim = sample_output.shape[1]
        input_dim = sample_input.shape[1]
        
        if self.config.fusion_method == "attention":
            self.fusion_network = AttentionFusion(
                n_models=self.n_models,
                feature_dim=feature_dim,
                config=self.config
            ).to(self.device)
            
        elif self.config.fusion_method == "gating":
            self.gating_network = GatingNetwork(
                input_dim=input_dim,
                n_models=self.n_models,
                config=self.config
            ).to(self.device)
            
        elif self.config.fusion_method == "adaptive":
            self.adaptive_weights = AdaptiveWeighting(
                n_models=self.n_models,
                config=self.config
            ).to(self.device)
        
        # Physics constraints module
        if self.config.enforce_physics_constraints:
            self.physics_constraints = PhysicsConstraintsModule(
                feature_dim=feature_dim,
                config=self.config
            ).to(self.device)
    
    def predict(self, X: np.ndarray, return_details: bool = False) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Make fused predictions from all models.
        
        Args:
            X (np.ndarray): Input features
            return_details (bool): Whether to return detailed fusion information
            
        Returns:
            Union[np.ndarray, Dict[str, Any]]: Fused predictions or detailed results
        """
        # Get predictions from all models
        model_predictions = []
        model_uncertainties = []
        
        for model in self.models:
            try:
                if hasattr(model, 'predict_with_uncertainty'):
                    pred, unc = model.predict_with_uncertainty(X)
                else:
                    pred = model.predict(X)
                    unc = np.ones_like(pred) * 0.1  # Default uncertainty
                
                model_predictions.append(pred)
                model_uncertainties.append(unc)
                
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                # Use zero predictions for failed models
                model_predictions.append(np.zeros((X.shape[0], 4)))
                model_uncertainties.append(np.ones((X.shape[0], 4)))
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        pred_tensors = [torch.FloatTensor(pred).to(self.device) for pred in model_predictions]
        unc_tensors = [torch.FloatTensor(unc).to(self.device) for unc in model_uncertainties]
        
        # Apply fusion method
        if self.config.fusion_method == "attention":
            fusion_results = self.fusion_network(pred_tensors, X_tensor, unc_tensors)
            fused_prediction = fusion_results['prediction']
            fusion_details = fusion_results
            
        elif self.config.fusion_method == "gating":
            gate_weights = self.gating_network(X_tensor)
            
            # Apply gating weights
            weighted_predictions = []
            for i, pred in enumerate(pred_tensors):
                weighted_pred = pred * gate_weights[:, i:i+1]
                weighted_predictions.append(weighted_pred)
            
            fused_prediction = torch.sum(torch.stack(weighted_predictions, dim=1), dim=1)
            fusion_details = {
                'prediction': fused_prediction,
                'gate_weights': gate_weights,
                'model_contributions': gate_weights
            }
            
        elif self.config.fusion_method == "adaptive":
            model_weights = self.adaptive_weights()
            
            # Apply adaptive weights
            weighted_predictions = []
            for i, pred in enumerate(pred_tensors):
                weighted_pred = pred * model_weights[i]
                weighted_predictions.append(weighted_pred)
            
            fused_prediction = torch.sum(torch.stack(weighted_predictions, dim=1), dim=1)
            fusion_details = {
                'prediction': fused_prediction,
                'adaptive_weights': model_weights,
                'model_contributions': model_weights.unsqueeze(0).expand(X.shape[0], -1)
            }
            
        elif self.config.fusion_method == "weighted":
            # Simple weighted average
            weights = torch.ones(self.n_models, device=self.device) / self.n_models
            
            weighted_predictions = []
            for i, pred in enumerate(pred_tensors):
                weighted_pred = pred * weights[i]
                weighted_predictions.append(weighted_pred)
            
            fused_prediction = torch.sum(torch.stack(weighted_predictions, dim=1), dim=1)
            fusion_details = {
                'prediction': fused_prediction,
                'weights': weights,
                'model_contributions': weights.unsqueeze(0).expand(X.shape[0], -1)
            }
        
        # Apply physics constraints if enabled
        if self.config.enforce_physics_constraints:
            fused_prediction = self.physics_constraints(fused_prediction, X_tensor)
        
        # Convert back to numpy
        fused_prediction_np = fused_prediction.detach().cpu().numpy()
        
        if return_details:
            # Convert tensor details to numpy
            details = {}
            for key, value in fusion_details.items():
                if isinstance(value, torch.Tensor):
                    details[key] = value.detach().cpu().numpy()
                else:
                    details[key] = value
            
            details['prediction'] = fused_prediction_np
            details['individual_predictions'] = model_predictions
            details['individual_uncertainties'] = model_uncertainties
            
            return details
        else:
            return fused_prediction_np
    
    def update_performance(self, X: np.ndarray, y_true: np.ndarray):
        """
        Update model performance tracking for adaptive weighting.
        
        Args:
            X (np.ndarray): Input features
            y_true (np.ndarray): True targets
        """
        if self.config.fusion_method == "adaptive":
            # Get individual model predictions
            model_errors = []
            
            for model in self.models:
                try:
                    pred = model.predict(X)
                    error = mean_squared_error(y_true, pred)
                    model_errors.append(error)
                except:
                    model_errors.append(1.0)  # High error for failed models
            
            # Update adaptive weights
            error_tensor = torch.FloatTensor(model_errors).to(self.device)
            self.adaptive_weights.update_performance(error_tensor)
    
    def get_model_contributions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get contribution scores for each model.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            Dict[str, np.ndarray]: Model contribution scores
        """
        details = self.predict(X, return_details=True)
        
        contributions = {}
        if 'model_contributions' in details:
            model_contributions = details['model_contributions']
            
            for i, model in enumerate(self.models):
                model_name = getattr(model, 'model_name', f'model_{i}')
                contributions[model_name] = model_contributions[:, i]
        
        return contributions
    
    def analyze_fusion_quality(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Analyze the quality of model fusion.
        
        Args:
            X (np.ndarray): Input features
            y_true (np.ndarray): True targets
            
        Returns:
            Dict[str, float]: Fusion quality metrics
        """
        # Get detailed predictions
        details = self.predict(X, return_details=True)
        fused_pred = details['prediction']
        individual_preds = details['individual_predictions']
        
        # Calculate metrics
        fused_mse = mean_squared_error(y_true, fused_pred)
        fused_mae = mean_absolute_error(y_true, fused_pred)
        
        # Individual model performance
        individual_mses = []
        for pred in individual_preds:
            mse = mean_squared_error(y_true, pred)
            individual_mses.append(mse)
        
        best_individual_mse = min(individual_mses)
        avg_individual_mse = np.mean(individual_mses)
        
        # Fusion improvement
        improvement_over_best = (best_individual_mse - fused_mse) / best_individual_mse
        improvement_over_avg = (avg_individual_mse - fused_mse) / avg_individual_mse
        
        # Diversity metrics
        if 'model_contributions' in details:
            contributions = details['model_contributions']
            diversity = np.std(contributions, axis=1).mean()
            entropy_score = np.mean([entropy(contrib + 1e-8) for contrib in contributions])
        else:
            diversity = 0.0
            entropy_score = 0.0
        
        return {
            'fused_mse': fused_mse,
            'fused_mae': fused_mae,
            'best_individual_mse': best_individual_mse,
            'avg_individual_mse': avg_individual_mse,
            'improvement_over_best': improvement_over_best,
            'improvement_over_avg': improvement_over_avg,
            'model_diversity': diversity,
            'contribution_entropy': entropy_score
        }
    
    def save(self, file_path: str):
        """Save the fusion model."""
        save_data = {
            'config': self.config,
            'n_models': self.n_models,
            'fusion_components': {}
        }
        
        # Save fusion components
        if hasattr(self, 'fusion_network'):
            save_data['fusion_components']['fusion_network'] = self.fusion_network.state_dict()
        if hasattr(self, 'gating_network'):
            save_data['fusion_components']['gating_network'] = self.gating_network.state_dict()
        if hasattr(self, 'adaptive_weights'):
            save_data['fusion_components']['adaptive_weights'] = self.adaptive_weights.state_dict()
        
        torch.save(save_data, file_path)
        logger.info(f"Model fusion saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str, models: List[Any]) -> 'ModelFusion':
        """Load a saved fusion model."""
        save_data = torch.load(file_path)
        config = save_data['config']
        
        fusion = cls(models, config)
        
        # Load fusion components
        if 'fusion_network' in save_data['fusion_components']:
            fusion.fusion_network.load_state_dict(save_data['fusion_components']['fusion_network'])
        if 'gating_network' in save_data['fusion_components']:
            fusion.gating_network.load_state_dict(save_data['fusion_components']['gating_network'])
        if 'adaptive_weights' in save_data['fusion_components']:
            fusion.adaptive_weights.load_state_dict(save_data['fusion_components']['adaptive_weights'])
        
        logger.info(f"Model fusion loaded from {file_path}")
        return fusion

class PhysicsConstraintsModule(nn.Module):
    """
    Physics constraints module for ensuring realistic predictions.
    """
    
    def __init__(self, feature_dim: int, config: FusionConfig):
        super().__init__()
        self.feature_dim = feature_dim
        self.config = config
        
        # Constraint enforcement network
        self.constraint_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()  # Ensure outputs are bounded
        )
    
    def forward(self, predictions: torch.Tensor, input_features: torch.Tensor) -> torch.Tensor:
        """
        Apply physics constraints to predictions.
        
        Args:
            predictions (torch.Tensor): Raw predictions
            input_features (torch.Tensor): Input features for context
            
        Returns:
            torch.Tensor: Constrained predictions
        """
        # Apply learned constraints
        constraint_factors = self.constraint_network(predictions)
        constrained_predictions = predictions * constraint_factors
        
        # Apply hard constraints
        # Ensure SoH is between 0 and 1
        constrained_predictions[:, 0] = torch.clamp(constrained_predictions[:, 0], 0.0, 1.0)
        
        # Ensure degradation rates are non-negative
        if constrained_predictions.size(1) > 1:
            constrained_predictions[:, 1:] = torch.clamp(constrained_predictions[:, 1:], 0.0, float('inf'))
        
        # Apply temperature-dependent constraints if temperature data available
        if input_features.size(1) >= 3:  # Assuming temperature is the 3rd feature
            temperature = input_features[:, 2]
            temp_factor = torch.exp((temperature - 25) / 10)  # Arrhenius-like relationship
            temp_factor = torch.clamp(temp_factor, 0.5, 3.0)  # Reasonable bounds
            
            # Apply temperature factor to degradation rates
            if constrained_predictions.size(1) > 1:
                constrained_predictions[:, 1:] = constrained_predictions[:, 1:] * temp_factor.unsqueeze(-1)
        
        # Ensure monotonic degradation (SoH should not increase)
        if hasattr(self, 'previous_soh'):
            constrained_predictions[:, 0] = torch.minimum(
                constrained_predictions[:, 0], 
                self.previous_soh
            )
        self.previous_soh = constrained_predictions[:, 0].clone()
        
        return constrained_predictions

class AdaptiveFusionModule(nn.Module):
    """
    Adaptive fusion module that learns optimal combination weights based on input characteristics.
    """
    
    def __init__(self, num_models: int, feature_dim: int, config: FusionConfig):
        super().__init__()
        self.num_models = num_models
        self.feature_dim = feature_dim
        self.config = config
        
        # Context analysis network
        self.context_analyzer = nn.Sequential(
            nn.Linear(feature_dim, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_dim // 2, num_models),
            nn.Softmax(dim=-1)
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(feature_dim + num_models, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_dim, num_models),
            nn.Sigmoid()
        )
        
        # Uncertainty quantification
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim + num_models, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_dim, num_models),
            nn.Softplus()  # Ensure positive uncertainty values
        )
        
    def forward(self, predictions: List[torch.Tensor], input_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Adaptively fuse predictions based on input context.
        
        Args:
            predictions (List[torch.Tensor]): List of predictions from different models
            input_features (torch.Tensor): Input features for context analysis
            
        Returns:
            Dict[str, torch.Tensor]: Fused predictions with uncertainty estimates
        """
        batch_size = input_features.size(0)
        
        # Stack predictions
        stacked_predictions = torch.stack(predictions, dim=1)  # (batch, num_models, output_dim)
        
        # Analyze context to determine fusion weights
        fusion_weights = self.context_analyzer(input_features)  # (batch, num_models)
        
        # Estimate model confidences
        confidence_input = torch.cat([input_features, fusion_weights], dim=-1)
        model_confidences = self.confidence_estimator(confidence_input)  # (batch, num_models)
        
        # Estimate uncertainties
        uncertainty_estimates = self.uncertainty_head(confidence_input)  # (batch, num_models)
        
        # Combine weights with confidences
        combined_weights = fusion_weights * model_confidences
        combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Weighted fusion
        fused_predictions = torch.sum(
            stacked_predictions * combined_weights.unsqueeze(-1), 
            dim=1
        )
        
        # Aggregate uncertainty
        weighted_uncertainties = torch.sum(
            uncertainty_estimates * combined_weights, 
            dim=1, 
            keepdim=True
        )
        
        # Calculate prediction variance
        prediction_variance = torch.sum(
            combined_weights.unsqueeze(-1) * (stacked_predictions - fused_predictions.unsqueeze(1))**2,
            dim=1
        )
        
        return {
            'predictions': fused_predictions,
            'fusion_weights': combined_weights,
            'model_confidences': model_confidences,
            'uncertainties': weighted_uncertainties,
            'prediction_variance': prediction_variance,
            'individual_predictions': stacked_predictions
        }

class ModelFusion(nn.Module):
    """
    Advanced model fusion system for combining multiple battery prediction models.
    
    This class implements sophisticated fusion strategies including adaptive weighting,
    uncertainty quantification, and physics-informed constraints.
    """
    
    def __init__(self, models: List[nn.Module], config: FusionConfig):
        super().__init__()
        self.config = config
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        # Determine feature dimension from first model
        self.feature_dim = self._infer_feature_dim()
        
        # Initialize fusion components
        if config.fusion_strategy == "adaptive":
            self.fusion_module = AdaptiveFusionModule(
                self.num_models, self.feature_dim, config
            )
        elif config.fusion_strategy == "attention":
            self.fusion_module = AttentionFusionModule(
                self.num_models, self.feature_dim, config
            )
        else:
            self.fusion_module = WeightedFusionModule(
                self.num_models, config
            )
        
        # Physics constraints
        if config.apply_physics_constraints:
            self.physics_constraints = PhysicsConstraintsModule(
                self.feature_dim, config
            )
        
        # Uncertainty calibration
        if config.enable_uncertainty_calibration:
            self.uncertainty_calibrator = UncertaintyCalibrationModule(config)
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
    def _infer_feature_dim(self) -> int:
        """Infer feature dimension from model architecture."""
        # Try to get feature dimension from model configuration
        for model in self.models:
            if hasattr(model, 'config') and hasattr(model.config, 'feature_dim'):
                return model.config.feature_dim
            elif hasattr(model, 'feature_dim'):
                return model.feature_dim
        
        # Default fallback
        return 16
    
    def forward(self, x: torch.Tensor, return_individual: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the fusion system.
        
        Args:
            x (torch.Tensor): Input features
            return_individual (bool): Whether to return individual model predictions
            
        Returns:
            Dict[str, torch.Tensor]: Fused predictions and metadata
        """
        # Get predictions from all models
        individual_predictions = []
        individual_uncertainties = []
        
        for model in self.models:
            with torch.no_grad() if not self.training else torch.enable_grad():
                if hasattr(model, 'predict_health'):
                    # Battery health predictor
                    pred_dict = model.predict_health(x)
                    pred = torch.tensor([[pred_dict['state_of_health']]], device=x.device)
                elif hasattr(model, 'predict_degradation'):
                    # Degradation forecaster
                    pred_dict = model.predict_degradation(x)
                    pred = pred_dict['forecasts'].mean(dim=1)  # Average over time horizon
                else:
                    # Generic model
                    pred = model(x)
                    if isinstance(pred, dict):
                        pred = pred.get('predictions', pred.get('output', list(pred.values())[0]))
                
                individual_predictions.append(pred)
                
                # Extract uncertainty if available
                if isinstance(pred, dict) and 'uncertainty' in pred:
                    individual_uncertainties.append(pred['uncertainty'])
                else:
                    # Default uncertainty estimate
                    individual_uncertainties.append(torch.ones_like(pred) * 0.1)
        
        # Ensure all predictions have the same shape
        individual_predictions = self._align_predictions(individual_predictions)
        
        # Fuse predictions
        fusion_result = self.fusion_module(individual_predictions, x)
        
        # Apply physics constraints
        if self.config.apply_physics_constraints:
            fusion_result['predictions'] = self.physics_constraints(
                fusion_result['predictions'], x
            )
        
        # Calibrate uncertainty
        if self.config.enable_uncertainty_calibration:
            fusion_result = self.uncertainty_calibrator(fusion_result, x)
        
        # Monitor performance
        self.performance_monitor.update(fusion_result, individual_predictions)
        
        # Prepare output
        output = {
            'predictions': fusion_result['predictions'],
            'fusion_weights': fusion_result.get('fusion_weights'),
            'uncertainties': fusion_result.get('uncertainties'),
            'prediction_variance': fusion_result.get('prediction_variance'),
            'performance_metrics': self.performance_monitor.get_metrics()
        }
        
        if return_individual:
            output['individual_predictions'] = fusion_result.get('individual_predictions')
        
        return output
    
    def _align_predictions(self, predictions: List[torch.Tensor]) -> List[torch.Tensor]:
        """Align predictions to have consistent shapes."""
        if not predictions:
            return predictions
        
        # Find target shape (use the most common shape or largest)
        shapes = [pred.shape for pred in predictions]
        target_shape = max(shapes, key=lambda s: np.prod(s))
        
        aligned_predictions = []
        for pred in predictions:
            if pred.shape != target_shape:
                # Reshape or pad as needed
                if len(pred.shape) < len(target_shape):
                    # Add dimensions
                    while len(pred.shape) < len(target_shape):
                        pred = pred.unsqueeze(-1)
                
                # Interpolate or repeat to match size
                if pred.shape[-1] < target_shape[-1]:
                    repeat_factor = target_shape[-1] // pred.shape[-1]
                    remainder = target_shape[-1] % pred.shape[-1]
                    pred = pred.repeat(1, repeat_factor)
                    if remainder > 0:
                        pred = torch.cat([pred, pred[:, :remainder]], dim=-1)
                elif pred.shape[-1] > target_shape[-1]:
                    pred = pred[:, :target_shape[-1]]
            
            aligned_predictions.append(pred)
        
        return aligned_predictions
    
    def evaluate_fusion_quality(self, x: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the quality of fusion compared to individual models.
        
        Args:
            x (torch.Tensor): Input features
            targets (torch.Tensor): Ground truth targets
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        with torch.no_grad():
            # Get fusion results
            fusion_output = self.forward(x, return_individual=True)
            fused_pred = fusion_output['predictions']
            individual_preds = fusion_output['individual_predictions']
            
            # Calculate metrics
            fusion_mse = F.mse_loss(fused_pred, targets).item()
            fusion_mae = F.l1_loss(fused_pred, targets).item()
            
            # Individual model metrics
            individual_mses = []
            individual_maes = []
            
            for pred in individual_preds.unbind(1):
                mse = F.mse_loss(pred, targets).item()
                mae = F.l1_loss(pred, targets).item()
                individual_mses.append(mse)
                individual_maes.append(mae)
            
            # Calculate improvement
            best_individual_mse = min(individual_mses)
            best_individual_mae = min(individual_maes)
            
            mse_improvement = (best_individual_mse - fusion_mse) / best_individual_mse * 100
            mae_improvement = (best_individual_mae - fusion_mae) / best_individual_mae * 100
            
            return {
                'fusion_mse': fusion_mse,
                'fusion_mae': fusion_mae,
                'best_individual_mse': best_individual_mse,
                'best_individual_mae': best_individual_mae,
                'mse_improvement_percent': mse_improvement,
                'mae_improvement_percent': mae_improvement,
                'individual_mses': individual_mses,
                'individual_maes': individual_maes
            }

class UncertaintyCalibrationModule(nn.Module):
    """
    Module for calibrating uncertainty estimates from fused predictions.
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Platt scaling parameters
        self.platt_a = nn.Parameter(torch.ones(1))
        self.platt_b = nn.Parameter(torch.zeros(1))
        
    def forward(self, fusion_result: Dict[str, torch.Tensor], 
                input_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calibrate uncertainty estimates.
        
        Args:
            fusion_result (Dict[str, torch.Tensor]): Fusion results
            input_features (torch.Tensor): Input features
            
        Returns:
            Dict[str, torch.Tensor]: Calibrated fusion results
        """
        if 'uncertainties' in fusion_result:
            # Apply temperature scaling
            calibrated_uncertainties = fusion_result['uncertainties'] / self.temperature
            
            # Apply Platt scaling
            calibrated_uncertainties = torch.sigmoid(
                self.platt_a * calibrated_uncertainties + self.platt_b
            )
            
            fusion_result['calibrated_uncertainties'] = calibrated_uncertainties
        
        return fusion_result

class PerformanceMonitor:
    """
    Monitor fusion performance and model contributions.
    """
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.model_contributions = {}
        
    def update(self, fusion_result: Dict[str, torch.Tensor], 
               individual_predictions: List[torch.Tensor]):
        """Update performance metrics."""
        if 'fusion_weights' in fusion_result:
            weights = fusion_result['fusion_weights'].mean(dim=0).cpu().numpy()
            
            # Update model contributions
            for i, weight in enumerate(weights):
                model_key = f'model_{i}'
                if model_key not in self.model_contributions:
                    self.model_contributions[model_key] = deque(maxlen=100)
                self.model_contributions[model_key].append(weight)
        
        # Store metrics
        metrics = {
            'timestamp': time.time(),
            'fusion_weights': fusion_result.get('fusion_weights'),
            'prediction_variance': fusion_result.get('prediction_variance'),
            'uncertainties': fusion_result.get('uncertainties')
        }
        self.metrics_history.append(metrics)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.model_contributions:
            return {}
        
        # Calculate average contributions
        avg_contributions = {}
        for model, contributions in self.model_contributions.items():
            avg_contributions[model] = np.mean(list(contributions))
        
        return {
            'average_model_contributions': avg_contributions,
            'total_predictions': len(self.metrics_history),
            'contribution_stability': self._calculate_stability()
        }
    
    def _calculate_stability(self) -> float:
        """Calculate stability of model contributions."""
        if not self.model_contributions:
            return 0.0
        
        stabilities = []
        for contributions in self.model_contributions.values():
            if len(contributions) > 1:
                stability = 1.0 - np.std(list(contributions))
                stabilities.append(max(0.0, stability))
        
        return np.mean(stabilities) if stabilities else 0.0

# Factory functions for easy instantiation
def create_model_fusion(models: List[nn.Module], 
                       config: Optional[FusionConfig] = None) -> ModelFusion:
    """
    Factory function to create a ModelFusion instance.
    
    Args:
        models (List[nn.Module]): List of models to fuse
        config (FusionConfig, optional): Fusion configuration
        
    Returns:
        ModelFusion: Configured fusion system
    """
    if config is None:
        config = FusionConfig()
    
    return ModelFusion(models, config)

def create_battery_ensemble(health_predictor_path: str,
                          forecaster_path: str,
                          config: Optional[FusionConfig] = None) -> ModelFusion:
    """
    Create a battery ensemble combining health prediction and forecasting.
    
    Args:
        health_predictor_path (str): Path to health predictor model
        forecaster_path (str): Path to forecaster model
        config (FusionConfig, optional): Fusion configuration
        
    Returns:
        ModelFusion: Battery ensemble system
    """
    from ..battery_health_predictor import create_battery_predictor, BatteryInferenceConfig
    from ..degradation_forecaster import create_battery_degradation_forecaster, ForecastingConfig
    
    # Load models
    health_config = BatteryInferenceConfig(model_path=health_predictor_path)
    health_predictor = create_battery_predictor(health_config)
    
    forecasting_config = ForecastingConfig(model_path=forecaster_path)
    forecaster = create_battery_degradation_forecaster(forecaster_path, {"forecasting": forecasting_config.__dict__})
    
    # Create fusion system
    models = [health_predictor, forecaster]
    return create_model_fusion(models, config)

# Evaluation utilities
def evaluate_fusion_performance(fusion_model: ModelFusion,
                               test_data: torch.Tensor,
                               test_targets: torch.Tensor) -> Dict[str, Any]:
    """
    Comprehensive evaluation of fusion model performance.
    
    Args:
        fusion_model (ModelFusion): Fusion model to evaluate
        test_data (torch.Tensor): Test input data
        test_targets (torch.Tensor): Test targets
        
    Returns:
        Dict[str, Any]: Comprehensive evaluation results
    """
    fusion_model.eval()
    
    with torch.no_grad():
        # Get predictions
        results = fusion_model(test_data, return_individual=True)
        
        # Evaluate fusion quality
        quality_metrics = fusion_model.evaluate_fusion_quality(test_data, test_targets)
        
        # Calculate additional metrics
        predictions = results['predictions']
        uncertainties = results.get('uncertainties')
        
        # Prediction accuracy
        mse = F.mse_loss(predictions, test_targets).item()
        mae = F.l1_loss(predictions, test_targets).item()
        
        # Uncertainty calibration (if available)
        calibration_metrics = {}
        if uncertainties is not None:
            # Calculate prediction intervals
            lower_bound = predictions - 1.96 * uncertainties
            upper_bound = predictions + 1.96 * uncertainties
            
            # Coverage probability
            coverage = torch.mean(
                ((test_targets >= lower_bound) & (test_targets <= upper_bound)).float()
            ).item()
            
            calibration_metrics = {
                'coverage_probability': coverage,
                'average_uncertainty': uncertainties.mean().item(),
                'uncertainty_std': uncertainties.std().item()
            }
        
        return {
            'fusion_quality': quality_metrics,
            'mse': mse,
            'mae': mae,
            'calibration_metrics': calibration_metrics,
            'predictions': predictions.cpu().numpy(),
            'uncertainties': uncertainties.cpu().numpy() if uncertainties is not None else None,
            'individual_predictions': results.get('individual_predictions')
        }

