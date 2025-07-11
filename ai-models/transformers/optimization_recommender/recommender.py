"""
BatteryMind - Optimization Recommender

Production-ready inference engine for battery optimization recommendations.
Provides intelligent suggestions for charging protocols, thermal management,
and operational strategies to maximize battery performance and lifespan.

Features:
- Multi-objective optimization recommendations
- Real-time optimization strategy generation
- Physics-informed constraint handling
- Integration with battery health and degradation models
- Comprehensive recommendation explanations
- Performance impact quantification

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import time
import json
from pathlib import Path
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Optimization imports
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import cvxpy as cp

# Local imports
from .model import OptimizationTransformer, OptimizationConfig
from .optimization_utils import (
    OptimizationObjective, ConstraintHandler, PerformanceEstimator,
    RecommendationValidator, OptimizationMetrics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationPriority(Enum):
    """Optimization priority levels."""
    SAFETY = "safety"
    LONGEVITY = "longevity"
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    COST = "cost"

class RecommendationType(Enum):
    """Types of optimization recommendations."""
    CHARGING_PROTOCOL = "charging_protocol"
    THERMAL_MANAGEMENT = "thermal_management"
    LOAD_BALANCING = "load_balancing"
    MAINTENANCE_SCHEDULE = "maintenance_schedule"
    OPERATIONAL_STRATEGY = "operational_strategy"
    REPLACEMENT_PLANNING = "replacement_planning"

@dataclass
class OptimizationRecommendation:
    """
    Comprehensive optimization recommendation structure.
    
    Attributes:
        recommendation_id (str): Unique recommendation identifier
        recommendation_type (RecommendationType): Type of recommendation
        priority (OptimizationPriority): Priority level
        confidence_score (float): Confidence in recommendation (0-1)
        expected_impact (Dict): Expected performance improvements
        implementation_steps (List[str]): Step-by-step implementation guide
        parameters (Dict): Specific parameter recommendations
        constraints (Dict): Operational constraints to consider
        risks (List[str]): Potential risks and mitigation strategies
        timeline (Dict): Implementation timeline and milestones
        cost_benefit (Dict): Cost-benefit analysis
        monitoring_metrics (List[str]): Metrics to monitor post-implementation
        explanation (str): Detailed explanation of recommendation
        alternatives (List[Dict]): Alternative optimization strategies
    """
    recommendation_id: str
    recommendation_type: RecommendationType
    priority: OptimizationPriority
    confidence_score: float
    expected_impact: Dict[str, float] = field(default_factory=dict)
    implementation_steps: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    risks: List[str] = field(default_factory=list)
    timeline: Dict[str, Any] = field(default_factory=dict)
    cost_benefit: Dict[str, float] = field(default_factory=dict)
    monitoring_metrics: List[str] = field(default_factory=list)
    explanation: str = ""
    alternatives: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary."""
        return {
            'recommendation_id': self.recommendation_id,
            'recommendation_type': self.recommendation_type.value,
            'priority': self.priority.value,
            'confidence_score': self.confidence_score,
            'expected_impact': self.expected_impact,
            'implementation_steps': self.implementation_steps,
            'parameters': self.parameters,
            'constraints': self.constraints,
            'risks': self.risks,
            'timeline': self.timeline,
            'cost_benefit': self.cost_benefit,
            'monitoring_metrics': self.monitoring_metrics,
            'explanation': self.explanation,
            'alternatives': self.alternatives
        }

@dataclass
class OptimizationContext:
    """
    Context information for optimization recommendations.
    
    Attributes:
        battery_id (str): Battery identifier
        current_state (Dict): Current battery state
        historical_data (pd.DataFrame): Historical performance data
        operational_constraints (Dict): Operational limitations
        business_objectives (List[str]): Business optimization goals
        environmental_conditions (Dict): Current environmental conditions
        usage_patterns (Dict): Typical usage patterns
        maintenance_history (List[Dict]): Historical maintenance records
        cost_parameters (Dict): Cost-related parameters
        safety_requirements (Dict): Safety constraints and requirements
    """
    battery_id: str
    current_state: Dict[str, float]
    historical_data: Optional[pd.DataFrame] = None
    operational_constraints: Dict[str, Any] = field(default_factory=dict)
    business_objectives: List[str] = field(default_factory=list)
    environmental_conditions: Dict[str, float] = field(default_factory=dict)
    usage_patterns: Dict[str, Any] = field(default_factory=dict)
    maintenance_history: List[Dict] = field(default_factory=list)
    cost_parameters: Dict[str, float] = field(default_factory=dict)
    safety_requirements: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecommenderConfig:
    """
    Configuration for optimization recommender.
    
    Attributes:
        model_path (str): Path to trained optimization model
        device (str): Inference device
        confidence_threshold (float): Minimum confidence for recommendations
        max_recommendations (int): Maximum number of recommendations
        enable_explanations (bool): Generate detailed explanations
        enable_alternatives (bool): Generate alternative strategies
        optimization_horizon (int): Optimization time horizon (hours)
        safety_margin (float): Safety margin for recommendations
        cost_weight (float): Weight for cost considerations
        performance_weight (float): Weight for performance considerations
    """
    model_path: str = "./model_artifacts/optimization_model.ckpt"
    device: str = "auto"
    confidence_threshold: float = 0.7
    max_recommendations: int = 5
    enable_explanations: bool = True
    enable_alternatives: bool = True
    optimization_horizon: int = 168  # 1 week
    safety_margin: float = 0.1
    cost_weight: float = 0.3
    performance_weight: float = 0.7

class BatteryOptimizationRecommender:
    """
    Advanced optimization recommender for battery management systems.
    
    Provides intelligent recommendations for various optimization objectives
    including charging protocols, thermal management, and operational strategies.
    """
    
    def __init__(self, config: RecommenderConfig):
        self.config = config
        self.device = self._setup_device()
        
        # Load optimization model
        self.model = self._load_model()
        self.model.eval()
        
        # Initialize optimization utilities
        self.objective_handler = OptimizationObjective()
        self.constraint_handler = ConstraintHandler()
        self.performance_estimator = PerformanceEstimator()
        self.validator = RecommendationValidator()
        self.metrics_calculator = OptimizationMetrics()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"BatteryOptimizationRecommender initialized on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup inference device."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.config.device)
    
    def _load_model(self) -> OptimizationTransformer:
        """Load trained optimization model."""
        checkpoint = torch.load(self.config.model_path, map_location=self.device)
        
        # Extract model configuration
        model_config = checkpoint.get('config')
        if hasattr(model_config, 'model'):
            model_config = OptimizationConfig(**model_config.model)
        else:
            model_config = OptimizationConfig()
        
        # Create and load model
        model = OptimizationTransformer(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def generate_recommendations(self, context: OptimizationContext,
                                priorities: List[OptimizationPriority] = None) -> List[OptimizationRecommendation]:
        """
        Generate comprehensive optimization recommendations.
        
        Args:
            context (OptimizationContext): Optimization context and constraints
            priorities (List[OptimizationPriority], optional): Optimization priorities
            
        Returns:
            List[OptimizationRecommendation]: Generated recommendations
        """
        if priorities is None:
            priorities = [OptimizationPriority.SAFETY, OptimizationPriority.LONGEVITY, 
                         OptimizationPriority.PERFORMANCE]
        
        recommendations = []
        
        # Analyze current state and generate recommendations for each priority
        for priority in priorities:
            priority_recommendations = self._generate_priority_recommendations(context, priority)
            recommendations.extend(priority_recommendations)
        
        # Filter and rank recommendations
        recommendations = self._filter_and_rank_recommendations(recommendations, context)
        
        # Validate recommendations
        validated_recommendations = []
        for rec in recommendations[:self.config.max_recommendations]:
            if self.validator.validate_recommendation(rec, context):
                validated_recommendations.append(rec)
        
        return validated_recommendations
    
    def _generate_priority_recommendations(self, context: OptimizationContext,
                                         priority: OptimizationPriority) -> List[OptimizationRecommendation]:
        """Generate recommendations for specific priority."""
        recommendations = []
        
        # Prepare input for model
        model_input = self._prepare_model_input(context)
        
        with torch.no_grad():
            # Get model predictions
            outputs = self.model(model_input, return_attention=True)
            optimization_scores = outputs['optimization_scores']
            attention_weights = outputs.get('attention_weights', [])
        
        # Generate recommendations based on priority
        if priority == OptimizationPriority.SAFETY:
            recommendations.extend(self._generate_safety_recommendations(context, optimization_scores))
        elif priority == OptimizationPriority.LONGEVITY:
            recommendations.extend(self._generate_longevity_recommendations(context, optimization_scores))
        elif priority == OptimizationPriority.PERFORMANCE:
            recommendations.extend(self._generate_performance_recommendations(context, optimization_scores))
        elif priority == OptimizationPriority.EFFICIENCY:
            recommendations.extend(self._generate_efficiency_recommendations(context, optimization_scores))
        elif priority == OptimizationPriority.COST:
            recommendations.extend(self._generate_cost_recommendations(context, optimization_scores))
        
        return recommendations
    
    def _generate_safety_recommendations(self, context: OptimizationContext,
                                       scores: torch.Tensor) -> List[OptimizationRecommendation]:
        """Generate safety-focused recommendations."""
        recommendations = []
        
        # Analyze safety risks
        safety_risks = self._analyze_safety_risks(context)
        
        # Temperature management recommendation
        if context.current_state.get('temperature', 25) > 40:
            rec = OptimizationRecommendation(
                recommendation_id=f"safety_thermal_{int(time.time())}",
                recommendation_type=RecommendationType.THERMAL_MANAGEMENT,
                priority=OptimizationPriority.SAFETY,
                confidence_score=0.95,
                expected_impact={
                    'temperature_reduction': 10.0,
                    'safety_improvement': 25.0,
                    'lifespan_extension': 15.0
                },
                implementation_steps=[
                    "Implement active cooling system",
                    "Reduce charging current by 20%",
                    "Monitor temperature continuously",
                    "Set temperature alerts at 45°C"
                ],
                parameters={
                    'max_charging_current': context.current_state.get('current', 50) * 0.8,
                    'cooling_activation_temp': 35.0,
                    'emergency_shutdown_temp': 55.0
                },
                constraints={
                    'max_temperature': 45.0,
                    'min_cooling_efficiency': 0.8
                },
                risks=[
                    "Reduced charging speed",
                    "Increased energy consumption for cooling"
                ],
                timeline={
                    'immediate': "Reduce charging current",
                    'short_term': "Install cooling system",
                    'monitoring': "Continuous temperature monitoring"
                },
                explanation="High temperature detected. Immediate thermal management required to prevent safety hazards and extend battery life."
            )
            recommendations.append(rec)
        
        # Voltage safety recommendation
        voltage = context.current_state.get('voltage', 3.7)
        if voltage > 4.1 or voltage < 2.8:
            rec = OptimizationRecommendation(
                recommendation_id=f"safety_voltage_{int(time.time())}",
                recommendation_type=RecommendationType.CHARGING_PROTOCOL,
                priority=OptimizationPriority.SAFETY,
                confidence_score=0.9,
                expected_impact={
                    'voltage_stability': 20.0,
                    'safety_improvement': 30.0,
                    'cycle_life_extension': 25.0
                },
                implementation_steps=[
                    "Adjust charging voltage limits",
                    "Implement voltage monitoring",
                    "Set protective cutoff voltages",
                    "Calibrate voltage sensors"
                ],
                parameters={
                    'max_charge_voltage': 4.05,
                    'min_discharge_voltage': 3.0,
                    'voltage_tolerance': 0.05
                },
                explanation="Voltage levels outside safe operating range detected. Immediate adjustment required."
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_longevity_recommendations(self, context: OptimizationContext,
                                          scores: torch.Tensor) -> List[OptimizationRecommendation]:
        """Generate longevity-focused recommendations."""
        recommendations = []
        
        # Analyze degradation patterns
        degradation_analysis = self._analyze_degradation_patterns(context)
        
        # Charging optimization for longevity
        if context.current_state.get('state_of_charge', 0.5) > 0.9:
            rec = OptimizationRecommendation(
                recommendation_id=f"longevity_charging_{int(time.time())}",
                recommendation_type=RecommendationType.CHARGING_PROTOCOL,
                priority=OptimizationPriority.LONGEVITY,
                confidence_score=0.85,
                expected_impact={
                    'cycle_life_extension': 40.0,
                    'capacity_retention': 15.0,
                    'degradation_reduction': 30.0
                },
                implementation_steps=[
                    "Limit maximum State of Charge to 85%",
                    "Implement smart charging schedule",
                    "Use lower charging rates for top-up",
                    "Avoid deep discharge cycles"
                ],
                parameters={
                    'max_soc': 0.85,
                    'min_soc': 0.15,
                    'optimal_soc_range': (0.2, 0.8),
                    'trickle_charge_threshold': 0.8
                },
                constraints={
                    'charging_rate_limit': 0.5,  # C-rate
                    'temperature_limit': 35.0
                },
                expected_impact={
                    'cycle_life_extension': 40.0,
                    'capacity_retention': 15.0
                },
                explanation="Optimized charging protocol to minimize degradation and maximize battery lifespan."
            )
            recommendations.append(rec)
        
        # Maintenance scheduling recommendation
        cycle_count = context.current_state.get('cycle_count', 0)
        if cycle_count > 1000 and cycle_count % 500 == 0:
            rec = OptimizationRecommendation(
                recommendation_id=f"longevity_maintenance_{int(time.time())}",
                recommendation_type=RecommendationType.MAINTENANCE_SCHEDULE,
                priority=OptimizationPriority.LONGEVITY,
                confidence_score=0.8,
                expected_impact={
                    'performance_restoration': 10.0,
                    'degradation_prevention': 20.0,
                    'reliability_improvement': 25.0
                },
                implementation_steps=[
                    "Schedule comprehensive battery inspection",
                    "Perform capacity calibration",
                    "Check connection integrity",
                    "Update battery management firmware"
                ],
                timeline={
                    'inspection': "Within 1 week",
                    'calibration': "Within 2 weeks",
                    'firmware_update': "Within 1 month"
                },
                explanation=f"Battery has completed {cycle_count} cycles. Preventive maintenance recommended."
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_performance_recommendations(self, context: OptimizationContext,
                                            scores: torch.Tensor) -> List[OptimizationRecommendation]:
        """Generate performance-focused recommendations."""
        recommendations = []
        
        # Power delivery optimization
        current_power = context.current_state.get('power', 0)
        max_power = context.operational_constraints.get('max_power', 1000)
        
        if current_power < max_power * 0.8:
            rec = OptimizationRecommendation(
                recommendation_id=f"performance_power_{int(time.time())}",
                recommendation_type=RecommendationType.OPERATIONAL_STRATEGY,
                priority=OptimizationPriority.PERFORMANCE,
                confidence_score=0.8,
                expected_impact={
                    'power_increase': 20.0,
                    'efficiency_improvement': 10.0,
                    'response_time_reduction': 15.0
                },
                implementation_steps=[
                    "Optimize power delivery algorithms",
                    "Adjust current limits based on temperature",
                    "Implement dynamic power management",
                    "Monitor power quality metrics"
                ],
                parameters={
                    'target_power': max_power * 0.9,
                    'power_ramp_rate': 100,  # W/s
                    'efficiency_target': 0.95
                },
                explanation="Power delivery can be optimized for better performance while maintaining safety."
            )
            recommendations.append(rec)
        
        # Load balancing recommendation
        if len(context.usage_patterns.get('load_distribution', [])) > 1:
            rec = OptimizationRecommendation(
                recommendation_id=f"performance_balancing_{int(time.time())}",
                recommendation_type=RecommendationType.LOAD_BALANCING,
                priority=OptimizationPriority.PERFORMANCE,
                confidence_score=0.75,
                expected_impact={
                    'load_distribution_improvement': 25.0,
                    'efficiency_gain': 12.0,
                    'thermal_optimization': 18.0
                },
                implementation_steps=[
                    "Implement dynamic load balancing",
                    "Monitor individual cell performance",
                    "Adjust load distribution algorithms",
                    "Set up automated balancing triggers"
                ],
                parameters={
                    'balancing_threshold': 0.05,  # 5% imbalance threshold
                    'balancing_current': 0.1,     # C/10 balancing rate
                    'balancing_frequency': 24      # hours
                },
                explanation="Load balancing optimization to improve overall system performance and efficiency."
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_efficiency_recommendations(self, context: OptimizationContext,
                                           scores: torch.Tensor) -> List[OptimizationRecommendation]:
        """Generate efficiency-focused recommendations."""
        recommendations = []
        
        # Energy efficiency optimization
        efficiency = context.current_state.get('efficiency', 0.9)
        if efficiency < 0.92:
            rec = OptimizationRecommendation(
                recommendation_id=f"efficiency_energy_{int(time.time())}",
                recommendation_type=RecommendationType.OPERATIONAL_STRATEGY,
                priority=OptimizationPriority.EFFICIENCY,
                confidence_score=0.8,
                expected_impact={
                    'efficiency_improvement': (0.95 - efficiency) * 100,
                    'energy_savings': 5.0,
                    'cost_reduction': 8.0
                },
                implementation_steps=[
                    "Optimize charging/discharging curves",
                    "Implement smart energy management",
                    "Reduce parasitic losses",
                    "Monitor efficiency metrics continuously"
                ],
                parameters={
                    'target_efficiency': 0.95,
                    'efficiency_monitoring_interval': 1,  # hours
                    'efficiency_threshold': 0.92
                },
                explanation="Energy efficiency below optimal levels. Implementation of smart energy management recommended."
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_cost_recommendations(self, context: OptimizationContext,
                                     scores: torch.Tensor) -> List[OptimizationRecommendation]:
        """Generate cost-focused recommendations."""
        recommendations = []
        
        # Replacement planning recommendation
        soh = context.current_state.get('state_of_health', 1.0)
        if soh < 0.8:
            replacement_cost = context.cost_parameters.get('replacement_cost', 10000)
            maintenance_cost = context.cost_parameters.get('maintenance_cost', 500)
            
            rec = OptimizationRecommendation(
                recommendation_id=f"cost_replacement_{int(time.time())}",
                recommendation_type=RecommendationType.REPLACEMENT_PLANNING,
                priority=OptimizationPriority.COST,
                confidence_score=0.85,
                expected_impact={
                    'cost_optimization': 15.0,
                    'performance_restoration': 100.0,
                    'reliability_improvement': 50.0
                },
                implementation_steps=[
                    "Evaluate replacement vs. refurbishment options",
                    "Plan replacement timeline",
                    "Optimize procurement strategy",
                    "Implement gradual transition plan"
                ],
                cost_benefit={
                    'replacement_cost': replacement_cost,
                    'maintenance_savings': maintenance_cost * 12,
                    'performance_value': replacement_cost * 0.2,
                    'roi_months': 18
                },
                timeline={
                    'evaluation': "2 weeks",
                    'procurement': "1-2 months",
                    'replacement': "3 months"
                },
                explanation=f"Battery SoH at {soh:.1%}. Cost-benefit analysis suggests replacement planning."
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _prepare_model_input(self, context: OptimizationContext) -> torch.Tensor:
        """Prepare input tensor for optimization model."""
        # Extract features from context
        features = []
        
        # Current state features
        state_features = [
            context.current_state.get('voltage', 3.7),
            context.current_state.get('current', 0.0),
            context.current_state.get('temperature', 25.0),
            context.current_state.get('state_of_charge', 0.5),
            context.current_state.get('state_of_health', 1.0),
            context.current_state.get('internal_resistance', 0.1),
            context.current_state.get('capacity', 100.0),
            context.current_state.get('cycle_count', 0),
            context.current_state.get('age_days', 0),
            context.current_state.get('power', 0.0),
            context.current_state.get('efficiency', 0.9)
        ]
        features.extend(state_features)
        
        # Environmental features
        env_features = [
            context.environmental_conditions.get('ambient_temperature', 25.0),
            context.environmental_conditions.get('humidity', 50.0),
            context.environmental_conditions.get('pressure', 1013.25)
        ]
        features.extend(env_features)
        
        # Operational constraint features
        constraint_features = [
            context.operational_constraints.get('max_current', 100.0),
            context.operational_constraints.get('max_power', 1000.0),
            context.operational_constraints.get('max_temperature', 45.0)
        ]
        features.extend(constraint_features)
        
        # Convert to tensor
        feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return feature_tensor.to(self.device)
    
    def _analyze_safety_risks(self, context: OptimizationContext) -> Dict[str, float]:
        """Analyze safety risks based on current state."""
        risks = {}
        
        # Temperature risk
        temp = context.current_state.get('temperature', 25)
        if temp > 45:
            risks['thermal_runaway'] = min(1.0, (temp - 45) / 15)
        
        # Voltage risk
        voltage = context.current_state.get('voltage', 3.7)
        if voltage > 4.2:
            risks['overvoltage'] = min(1.0, (voltage - 4.2) / 0.3)
        elif voltage < 2.5:
            risks['undervoltage'] = min(1.0, (2.5 - voltage) / 0.5)
        
        # Current risk
        current = abs(context.current_state.get('current', 0))
        max_current = context.operational_constraints.get('max_current', 100)
        if current > max_current:
            risks['overcurrent'] = min(1.0, (current - max_current) / max_current)
        
        return risks
    
    def _analyze_degradation_patterns(self, context: OptimizationContext) -> Dict[str, Any]:
        """Analyze degradation patterns from historical data."""
        analysis = {
            'capacity_fade_rate': 0.0,
            'resistance_increase_rate': 0.0,
            'cycle_aging': 0.0,
            'calendar_aging': 0.0
        }
        
        if context.historical_data is not None and len(context.historical_data) > 1:
            # Calculate capacity fade rate
            if 'capacity' in context.historical_data.columns:
                capacity_data = context.historical_data['capacity'].values
                if len(capacity_data) > 1:
                    fade_rate = (capacity_data[0] - capacity_data[-1]) / len(capacity_data)
                    analysis['capacity_fade_rate'] = fade_rate
            
            # Calculate resistance increase rate
            if 'internal_resistance' in context.historical_data.columns:
                resistance_data = context.historical_data['internal_resistance'].values
                if len(resistance_data) > 1:
                    increase_rate = (resistance_data[-1] - resistance_data[0]) / len(resistance_data)
                    analysis['resistance_increase_rate'] = increase_rate
        
        return analysis
    
    def _filter_and_rank_recommendations(self, recommendations: List[OptimizationRecommendation],
                                       context: OptimizationContext) -> List[OptimizationRecommendation]:
        """Filter and rank recommendations by relevance and impact."""
        # Filter by confidence threshold
        filtered_recs = [rec for rec in recommendations 
                        if rec.confidence_score >= self.config.confidence_threshold]
        
        # Calculate ranking scores
        for rec in filtered_recs:
            score = self._calculate_recommendation_score(rec, context)
            rec.ranking_score = score
        
        # Sort by ranking score
        filtered_recs.sort(key=lambda x: getattr(x, 'ranking_score', 0), reverse=True)
        
        return filtered_recs
    
    def _calculate_recommendation_score(self, recommendation: OptimizationRecommendation,
                                      context: OptimizationContext) -> float:
        """Calculate ranking score for recommendation."""
        score = 0.0
        
        # Base score from confidence
        score += recommendation.confidence_score * 0.3
        
        # Impact score
        impact_score = sum(recommendation.expected_impact.values()) / len(recommendation.expected_impact) if recommendation.expected_impact else 0
        score += min(impact_score / 100, 1.0) * 0.4
        
        # Priority weight
        priority_weights = {
            OptimizationPriority.SAFETY: 1.0,
            OptimizationPriority.LONGEVITY: 0.8,
            OptimizationPriority.PERFORMANCE: 0.7,
            OptimizationPriority.EFFICIENCY: 0.6,
            OptimizationPriority.COST: 0.5
        }
        score += priority_weights.get(recommendation.priority, 0.5) * 0.3
        
        return score
    
    async def generate_recommendations_async(self, context: OptimizationContext,
                                           priorities: List[OptimizationPriority] = None) -> List[OptimizationRecommendation]:
        """Asynchronous recommendation generation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.generate_recommendations,
            context,
            priorities
        )
    
    def explain_recommendation(self, recommendation: OptimizationRecommendation,
                             context: OptimizationContext) -> str:
        """Generate detailed explanation for recommendation."""
        explanation = f"Recommendation: {recommendation.recommendation_type.value}\n"
        explanation += f"Priority: {recommendation.priority.value}\n"
        explanation += f"Confidence: {recommendation.confidence_score:.2%}\n\n"
        
        explanation += "Analysis:\n"
        explanation += recommendation.explanation + "\n\n"
        
        explanation += "Expected Impact:\n"
        for metric, value in recommendation.expected_impact.items():
            explanation += f"- {metric}: {value:+.1f}%\n"
        
        explanation += "\nImplementation Steps:\n"
        for i, step in enumerate(recommendation.implementation_steps, 1):
            explanation += f"{i}. {step}\n"
        
        if recommendation.risks:
            explanation += "\nRisks and Considerations:\n"
            for risk in recommendation.risks:
                explanation += f"- {risk}\n"
        
        return explanation
    
    def get_recommender_status(self) -> Dict[str, Any]:
        """Get comprehensive recommender status."""
        return {
            'model_loaded': self.model is not None,
            'device': str(self.device),
            'config': self.config.__dict__,
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'version': '1.0.0'
        }

# Factory functions
def create_optimization_recommender(config: Optional[RecommenderConfig] = None) -> BatteryOptimizationRecommender:
    """
    Factory function to create an optimization recommender.
    
    Args:
        config (RecommenderConfig, optional): Recommender configuration
        
    Returns:
        BatteryOptimizationRecommender: Configured recommender instance
    """
    if config is None:
        config = RecommenderConfig()
    
    return BatteryOptimizationRecommender(config)

def create_optimization_context(battery_id: str, current_state: Dict[str, float],
                              **kwargs) -> OptimizationContext:
    """
    Factory function to create optimization context.
    
    Args:
        battery_id (str): Battery identifier
        current_state (Dict[str, float]): Current battery state
        **kwargs: Additional context parameters
        
    Returns:
        OptimizationContext: Configured context instance
    """
    return OptimizationContext(
        battery_id=battery_id,
        current_state=current_state,
        **kwargs
    )
