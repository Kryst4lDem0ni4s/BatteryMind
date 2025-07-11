"""
BatteryMind - Optimization Recommender Model

Advanced transformer architecture for generating intelligent optimization
recommendations for battery management systems. Provides actionable insights
for charging protocols, thermal management, and operational efficiency.

Features:
- Multi-objective optimization recommendation generation
- Context-aware recommendation system with attention mechanisms
- Physics-informed optimization constraints
- Real-time recommendation scoring and ranking
- Integration with battery health and degradation models
- Explainable AI for recommendation justification

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    """Types of optimization recommendations."""
    CHARGING_PROTOCOL = "charging_protocol"
    THERMAL_MANAGEMENT = "thermal_management"
    LOAD_BALANCING = "load_balancing"
    MAINTENANCE_SCHEDULING = "maintenance_scheduling"
    ENERGY_EFFICIENCY = "energy_efficiency"
    SAFETY_PROTOCOL = "safety_protocol"
    LIFECYCLE_EXTENSION = "lifecycle_extension"

@dataclass
class OptimizationConfig:
    """
    Configuration class for Optimization Recommender model.
    
    Attributes:
        # Model architecture
        d_model (int): Model dimension (embedding size)
        n_heads (int): Number of attention heads
        n_layers (int): Number of transformer layers
        d_ff (int): Feed-forward network dimension
        dropout (float): Dropout probability
        max_sequence_length (int): Maximum input sequence length
        
        # Recommendation specific
        num_recommendation_types (int): Number of different recommendation types
        max_recommendations_per_type (int): Maximum recommendations per category
        recommendation_embedding_dim (int): Dimension for recommendation embeddings
        
        # Context modeling
        context_window_size (int): Size of context window for recommendations
        enable_multi_objective (bool): Enable multi-objective optimization
        objective_weights (Dict[str, float]): Weights for different objectives
        
        # Input/Output dimensions
        feature_dim (int): Input feature dimension
        recommendation_dim (int): Output recommendation dimension
        
        # Advanced features
        use_context_attention (bool): Use context-aware attention
        use_recommendation_ranking (bool): Enable recommendation ranking
        use_explainable_ai (bool): Enable explainable recommendations
        
        # Physics constraints
        use_physics_constraints (bool): Enable physics-informed constraints
        safety_constraints (Dict[str, Tuple[float, float]]): Safety constraint bounds
        
        # Optimization parameters
        temperature_scaling (float): Temperature for recommendation scoring
        diversity_penalty (float): Penalty for similar recommendations
        feasibility_threshold (float): Minimum feasibility score
    """
    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_sequence_length: int = 1024
    
    # Recommendation specific
    num_recommendation_types: int = 7  # Number of OptimizationType values
    max_recommendations_per_type: int = 5
    recommendation_embedding_dim: int = 256
    
    # Context modeling
    context_window_size: int = 168  # 1 week in hours
    enable_multi_objective: bool = True
    objective_weights: Dict[str, float] = None
    
    # Input/Output dimensions
    feature_dim: int = 20
    recommendation_dim: int = 128
    
    # Advanced features
    use_context_attention: bool = True
    use_recommendation_ranking: bool = True
    use_explainable_ai: bool = True
    
    # Physics constraints
    use_physics_constraints: bool = True
    safety_constraints: Dict[str, Tuple[float, float]] = None
    
    # Optimization parameters
    temperature_scaling: float = 1.0
    diversity_penalty: float = 0.1
    feasibility_threshold: float = 0.7
    
    def __post_init__(self):
        if self.objective_weights is None:
            self.objective_weights = {
                'efficiency': 0.3,
                'safety': 0.25,
                'longevity': 0.25,
                'performance': 0.2
            }
        if self.safety_constraints is None:
            self.safety_constraints = {
                'temperature': (0.0, 60.0),  # °C
                'voltage': (2.5, 4.2),       # V
                'current': (-100.0, 100.0),  # A
                'power': (0.0, 1000.0)       # W
            }

class ContextualAttention(nn.Module):
    """
    Context-aware attention mechanism for optimization recommendations.
    """
    
    def __init__(self, d_model: int, n_heads: int, context_dim: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.context_dim = context_dim
        
        # Standard attention projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Context integration
        self.context_projection = nn.Linear(context_dim, d_model)
        self.context_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # Optimization-specific attention bias
        self.optimization_bias = nn.Parameter(torch.zeros(n_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply contextual attention mechanism.
        
        Args:
            query (torch.Tensor): Query tensor
            key (torch.Tensor): Key tensor
            value (torch.Tensor): Value tensor
            context (torch.Tensor, optional): Context information
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights
        """
        batch_size, seq_len, d_model = query.size()
        
        # Linear projections and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Add optimization bias
        scores = scores + self.optimization_bias
        
        # Integrate context if provided
        if context is not None:
            context_proj = self.context_projection(context)
            context_expanded = context_proj.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Compute context-aware attention modification
            combined = torch.cat([query, context_expanded], dim=-1)
            context_gate_weights = self.context_gate(combined)
            
            # Apply context gating to attention scores
            context_influence = torch.matmul(
                Q, context_proj.unsqueeze(-1).expand(-1, -1, self.d_k).view(
                    batch_size, 1, self.n_heads, self.d_k
                ).transpose(1, 2).transpose(-2, -1)
            ) / self.scale
            
            scores = scores + context_influence * context_gate_weights.unsqueeze(1).unsqueeze(-1)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context_output = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context_output = context_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        output = self.w_o(context_output)
        
        return output, attention_weights

class RecommendationGenerator(nn.Module):
    """
    Neural module for generating optimization recommendations.
    """
    
    def __init__(self, d_model: int, recommendation_dim: int, 
                 num_types: int, max_per_type: int):
        super().__init__()
        self.d_model = d_model
        self.recommendation_dim = recommendation_dim
        self.num_types = num_types
        self.max_per_type = max_per_type
        
        # Recommendation type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_types)
        )
        
        # Recommendation generators for each type
        self.recommendation_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, recommendation_dim * max_per_type)
            ) for _ in range(num_types)
        ])
        
        # Recommendation scoring
        self.recommendation_scorer = nn.Sequential(
            nn.Linear(recommendation_dim, recommendation_dim // 2),
            nn.ReLU(),
            nn.Linear(recommendation_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Feasibility checker
        self.feasibility_checker = nn.Sequential(
            nn.Linear(recommendation_dim + d_model, recommendation_dim),
            nn.ReLU(),
            nn.Linear(recommendation_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states: torch.Tensor, 
                battery_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Generate optimization recommendations.
        
        Args:
            hidden_states (torch.Tensor): Hidden states from transformer
            battery_context (torch.Tensor, optional): Battery context information
            
        Returns:
            Dict[str, torch.Tensor]: Generated recommendations and scores
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Use last hidden state for recommendation generation
        last_hidden = hidden_states[:, -1, :]  # (batch_size, d_model)
        
        # Classify recommendation types
        type_logits = self.type_classifier(last_hidden)
        type_probs = F.softmax(type_logits, dim=-1)
        
        # Generate recommendations for each type
        all_recommendations = []
        all_scores = []
        all_feasibility = []
        
        for i, generator in enumerate(self.recommendation_generators):
            # Generate recommendations for this type
            recommendations = generator(last_hidden)  # (batch_size, recommendation_dim * max_per_type)
            recommendations = recommendations.view(
                batch_size, self.max_per_type, self.recommendation_dim
            )
            
            # Score recommendations
            scores = self.recommendation_scorer(recommendations).squeeze(-1)  # (batch_size, max_per_type)
            
            # Check feasibility
            if battery_context is not None:
                context_expanded = battery_context.unsqueeze(1).expand(-1, self.max_per_type, -1)
                feasibility_input = torch.cat([recommendations, context_expanded], dim=-1)
            else:
                context_expanded = last_hidden.unsqueeze(1).expand(-1, self.max_per_type, -1)
                feasibility_input = torch.cat([recommendations, context_expanded], dim=-1)
            
            feasibility = self.feasibility_checker(feasibility_input).squeeze(-1)
            
            all_recommendations.append(recommendations)
            all_scores.append(scores)
            all_feasibility.append(feasibility)
        
        # Stack all recommendations
        recommendations_tensor = torch.stack(all_recommendations, dim=1)  # (batch_size, num_types, max_per_type, recommendation_dim)
        scores_tensor = torch.stack(all_scores, dim=1)  # (batch_size, num_types, max_per_type)
        feasibility_tensor = torch.stack(all_feasibility, dim=1)  # (batch_size, num_types, max_per_type)
        
        return {
            'recommendations': recommendations_tensor,
            'type_probabilities': type_probs,
            'recommendation_scores': scores_tensor,
            'feasibility_scores': feasibility_tensor,
            'hidden_states': hidden_states
        }

class ExplainableAI(nn.Module):
    """
    Explainable AI module for recommendation justification.
    """
    
    def __init__(self, d_model: int, recommendation_dim: int):
        super().__init__()
        self.d_model = d_model
        self.recommendation_dim = recommendation_dim
        
        # Feature importance analyzer
        self.feature_importance = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Softmax(dim=-1)
        )
        
        # Recommendation justification generator
        self.justification_generator = nn.Sequential(
            nn.Linear(d_model + recommendation_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 512),  # Text embedding dimension
            nn.Tanh()
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(d_model + recommendation_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states: torch.Tensor, 
                recommendations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate explanations for recommendations.
        
        Args:
            hidden_states (torch.Tensor): Hidden states from transformer
            recommendations (torch.Tensor): Generated recommendations
            
        Returns:
            Dict[str, torch.Tensor]: Explanation components
        """
        batch_size = hidden_states.size(0)
        last_hidden = hidden_states[:, -1, :]
        
        # Calculate feature importance
        feature_importance = self.feature_importance(last_hidden)
        
        # Generate justifications for each recommendation
        explanations = []
        confidences = []
        
        for i in range(recommendations.size(1)):  # For each recommendation type
            for j in range(recommendations.size(2)):  # For each recommendation
                rec = recommendations[:, i, j, :]
                combined = torch.cat([last_hidden, rec], dim=-1)
                
                justification = self.justification_generator(combined)
                confidence = self.confidence_estimator(combined)
                
                explanations.append(justification)
                confidences.append(confidence)
        
        explanations_tensor = torch.stack(explanations, dim=1)
        confidences_tensor = torch.stack(confidences, dim=1)
        
        return {
            'feature_importance': feature_importance,
            'justifications': explanations_tensor,
            'confidence_scores': confidences_tensor
        }

class OptimizationTransformerBlock(nn.Module):
    """
    Transformer block optimized for optimization recommendation generation.
    """
    
    def __init__(self, config: OptimizationConfig):
        super().__init__()
        self.config = config
        
        # Context-aware attention
        if config.use_context_attention:
            self.attention = ContextualAttention(
                config.d_model, config.n_heads, config.feature_dim, config.dropout
            )
        else:
            self.attention = nn.MultiheadAttention(
                config.d_model, config.n_heads, config.dropout, batch_first=True
            )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through optimization transformer block."""
        # Self-attention with residual connection
        if isinstance(self.attention, ContextualAttention):
            attn_output, _ = self.attention(x, x, x, context, mask)
        else:
            attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class OptimizationRecommender(nn.Module):
    """
    Main transformer model for battery optimization recommendations.
    
    This model analyzes battery state and operational context to generate
    intelligent optimization recommendations with explanations and feasibility scores.
    """
    
    def __init__(self, config: OptimizationConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.feature_dim, config.d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, config.max_sequence_length, config.d_model) * 0.02
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.d_model)
        )
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            OptimizationTransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Recommendation generator
        self.recommendation_generator = RecommendationGenerator(
            config.d_model, config.recommendation_dim,
            config.num_recommendation_types, config.max_recommendations_per_type
        )
        
        # Explainable AI module
        if config.use_explainable_ai:
            self.explainable_ai = ExplainableAI(config.d_model, config.recommendation_dim)
        
        # Physics constraints layer
        if config.use_physics_constraints:
            self.physics_constraints = OptimizationPhysicsConstraints(config)
        
        # Recommendation ranking
        if config.use_recommendation_ranking:
            self.recommendation_ranker = RecommendationRanker(config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                return_explanations: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the optimization recommender.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim)
            context (torch.Tensor, optional): Additional context information
            return_explanations (bool): Whether to return explanations
            
        Returns:
            Dict[str, torch.Tensor]: Recommendations with scores and explanations
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # Input validation
        if feature_dim != self.config.feature_dim:
            raise ValueError(f"Expected feature_dim {self.config.feature_dim}, got {feature_dim}")
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:, :seq_len, :]
        x = x + pos_encoding
        
        # Encode context if provided
        encoded_context = None
        if context is not None:
            encoded_context = self.context_encoder(context)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, encoded_context)
        
        # Generate recommendations
        recommendation_output = self.recommendation_generator(x, encoded_context)
        
        # Apply physics constraints if enabled
        if self.config.use_physics_constraints:
            recommendation_output = self.physics_constraints(recommendation_output, x, context)
        
        # Rank recommendations if enabled
        if self.config.use_recommendation_ranking:
            recommendation_output = self.recommendation_ranker(recommendation_output)
        
        # Generate explanations if requested
        if return_explanations and self.config.use_explainable_ai:
            explanations = self.explainable_ai(x, recommendation_output['recommendations'])
            recommendation_output.update(explanations)
        
        return recommendation_output
    
    def generate_recommendations(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                               top_k: int = 3) -> Dict[str, Union[torch.Tensor, List]]:
        """
        High-level interface for generating optimization recommendations.
        
        Args:
            x (torch.Tensor): Input sensor data
            context (torch.Tensor, optional): Additional context
            top_k (int): Number of top recommendations to return per type
            
        Returns:
            Dict[str, Union[torch.Tensor, List]]: Formatted recommendations
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x, context, return_explanations=True)
            
            recommendations = output['recommendations']
            scores = output['recommendation_scores']
            feasibility = output['feasibility_scores']
            type_probs = output['type_probabilities']
            
            # Format recommendations by type
            formatted_recommendations = {}
            
            for i, opt_type in enumerate(OptimizationType):
                type_recommendations = []
                
                # Get top-k recommendations for this type
                type_scores = scores[:, i, :]
                type_feasibility = feasibility[:, i, :]
                type_recs = recommendations[:, i, :, :]
                
                # Combine score and feasibility
                combined_scores = type_scores * type_feasibility
                
                # Get top-k indices
                _, top_indices = torch.topk(combined_scores, min(top_k, combined_scores.size(-1)), dim=-1)
                
                for batch_idx in range(recommendations.size(0)):
                    batch_recommendations = []
                    for k in range(min(top_k, top_indices.size(-1))):
                        idx = top_indices[batch_idx, k]
                        rec_data = {
                            'type': opt_type.value,
                            'recommendation_vector': type_recs[batch_idx, idx, :],
                            'score': combined_scores[batch_idx, idx].item(),
                            'feasibility': type_feasibility[batch_idx, idx].item(),
                            'type_probability': type_probs[batch_idx, i].item()
                        }
                        batch_recommendations.append(rec_data)
                    type_recommendations.append(batch_recommendations)
                
                formatted_recommendations[opt_type.value] = type_recommendations
            
            return {
                'recommendations_by_type': formatted_recommendations,
                'raw_output': output
            }

class OptimizationPhysicsConstraints(nn.Module):
    """
    Physics-informed constraints for optimization recommendations.
    """
    
    def __init__(self, config: OptimizationConfig):
        super().__init__()
        self.config = config
        
    def forward(self, recommendation_output: Dict[str, torch.Tensor],
                hidden_states: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Apply physics constraints to recommendations.
        
        Args:
            recommendation_output (Dict[str, torch.Tensor]): Raw recommendations
            hidden_states (torch.Tensor): Hidden states from transformer
            context (torch.Tensor, optional): Context information
            
        Returns:
            Dict[str, torch.Tensor]: Constrained recommendations
        """
        # Apply safety constraints to feasibility scores
        feasibility = recommendation_output['feasibility_scores']
        
        # Penalize recommendations that violate safety constraints
        if context is not None:
            for constraint_name, (min_val, max_val) in self.config.safety_constraints.items():
                # This is a simplified constraint application
                # In practice, you would decode the recommendation vectors
                # and check specific parameter constraints
                constraint_penalty = torch.ones_like(feasibility)
                feasibility = feasibility * constraint_penalty
        
        # Ensure feasibility scores are within valid range
        feasibility = torch.clamp(feasibility, 0.0, 1.0)
        
        recommendation_output['feasibility_scores'] = feasibility
        return recommendation_output

class RecommendationRanker(nn.Module):
    """
    Neural ranking module for optimization recommendations.
    """
    
    def __init__(self, config: OptimizationConfig):
        super().__init__()
        self.config = config
        
        # Multi-objective scorer
        self.objective_scorers = nn.ModuleDict({
            objective: nn.Sequential(
                nn.Linear(config.recommendation_dim, config.recommendation_dim // 2),
                nn.ReLU(),
                nn.Linear(config.recommendation_dim // 2, 1),
                nn.Sigmoid()
            ) for objective in config.objective_weights.keys()
        })
        
        # Diversity calculator
        self.diversity_scorer = nn.Sequential(
            nn.Linear(config.recommendation_dim * 2, config.recommendation_dim),
            nn.ReLU(),
            nn.Linear(config.recommendation_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, recommendation_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Rank recommendations based on multiple objectives.
        
        Args:
            recommendation_output (Dict[str, torch.Tensor]): Input recommendations
            
        Returns:
            Dict[str, torch.Tensor]: Ranked recommendations
        """
        recommendations = recommendation_output['recommendations']
        batch_size, num_types, max_per_type, rec_dim = recommendations.shape
        
        # Calculate objective scores
        objective_scores = {}
        for objective, scorer in self.objective_scorers.items():
            scores = scorer(recommendations.view(-1, rec_dim))
            objective_scores[objective] = scores.view(batch_size, num_types, max_per_type)
        
        # Combine objective scores with weights
        combined_scores = torch.zeros(batch_size, num_types, max_per_type, device=recommendations.device)
        for objective, weight in self.config.objective_weights.items():
            if objective in objective_scores:
                combined_scores += weight * objective_scores[objective]
        
        # Apply diversity penalty
        diversity_penalties = torch.zeros_like(combined_scores)
        for i in range(max_per_type):
            for j in range(i + 1, max_per_type):
                rec_i = recommendations[:, :, i, :]
                rec_j = recommendations[:, :, j, :]
                combined_recs = torch.cat([rec_i, rec_j], dim=-1)
                
                diversity = self.diversity_scorer(combined_recs.view(-1, rec_dim * 2))
                diversity = diversity.view(batch_size, num_types)
                
                # Apply penalty to both recommendations
                diversity_penalties[:, :, i] += (1 - diversity) * self.config.diversity_penalty
                diversity_penalties[:, :, j] += (1 - diversity) * self.config.diversity_penalty
        
        # Final ranking scores
        final_scores = combined_scores - diversity_penalties
        
        # Update recommendation scores
        recommendation_output['ranking_scores'] = final_scores
        recommendation_output['objective_scores'] = objective_scores
        
        return recommendation_output

# Factory function for easy model creation
def create_optimization_recommender(config: Optional[OptimizationConfig] = None) -> OptimizationRecommender:
    """
    Factory function to create an OptimizationRecommender model.
    
    Args:
        config (OptimizationConfig, optional): Model configuration
        
    Returns:
        OptimizationRecommender: Configured model instance
    """
    if config is None:
        config = OptimizationConfig()
    
    model = OptimizationRecommender(config)
    logger.info(f"Created OptimizationRecommender with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model

# Model summary function
def get_recommender_summary(model: OptimizationRecommender) -> Dict[str, Union[int, str]]:
    """
    Get comprehensive model summary.
    
    Args:
        model (OptimizationRecommender): Model instance
        
    Returns:
        Dict[str, Union[int, str]]: Model summary information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'model_name': 'OptimizationRecommender',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'num_recommendation_types': model.config.num_recommendation_types,
        'max_recommendations_per_type': model.config.max_recommendations_per_type,
        'explainable_ai_enabled': model.config.use_explainable_ai,
        'physics_constraints_enabled': model.config.use_physics_constraints,
        'config': model.config.__dict__
    }
