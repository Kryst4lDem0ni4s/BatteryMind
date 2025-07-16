"""
BatteryMind Thermal Optimizer

Advanced thermal management optimization for battery systems using AI/ML models
to maintain optimal operating temperatures and prevent thermal runaway.

This module provides:
- Multi-objective thermal optimization
- Predictive thermal modeling
- Adaptive cooling/heating strategies
- Safety constraint enforcement
- Real-time thermal monitoring
- Fleet-level thermal coordination

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod
import time
from concurrent.futures import ThreadPoolExecutor
import json

# BatteryMind imports
from ..predictors.battery_health_predictor import BatteryHealthPredictor
from ..predictors.degradation_predictor import DegradationPredictor
from ...transformers.optimization_recommender import OptimizationRecommender
from ...reinforcement_learning.agents.thermal_agent import ThermalAgent
from ...training_data.generators.physics_simulator import BatteryPhysicsSimulator
from ...evaluation.metrics.efficiency_metrics import ThermalEfficiencyMetrics
from ...utils.logging_utils import setup_logger
from ...utils.config_parser import ConfigParser

# Configure logging
logger = setup_logger(__name__)

class ThermalStrategy(Enum):
    """Thermal management strategies."""
    PASSIVE = "passive"
    ACTIVE_COOLING = "active_cooling"
    ACTIVE_HEATING = "active_heating"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"

class ThermalPriority(Enum):
    """Thermal optimization priorities."""
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    LONGEVITY = "longevity"
    PERFORMANCE = "performance"

@dataclass
class ThermalConstraints:
    """Thermal constraints for optimization."""
    min_temperature: float = -20.0  # °C
    max_temperature: float = 60.0   # °C
    optimal_min: float = 15.0       # °C
    optimal_max: float = 35.0       # °C
    critical_max: float = 70.0      # °C
    gradient_max: float = 5.0       # °C/min
    thermal_runaway_threshold: float = 80.0  # °C

@dataclass
class ThermalState:
    """Current thermal state of battery system."""
    temperature: float
    gradient: float
    ambient_temperature: float
    heat_generation_rate: float
    cooling_capacity: float
    heating_capacity: float
    thermal_mass: float
    thermal_resistance: float
    timestamp: float

@dataclass
class ThermalAction:
    """Thermal management actions."""
    cooling_power: float = 0.0      # W
    heating_power: float = 0.0      # W
    fan_speed: float = 0.0          # %
    coolant_flow: float = 0.0       # L/min
    thermal_interface: float = 0.0   # W/K
    strategy: ThermalStrategy = ThermalStrategy.PASSIVE

@dataclass
class ThermalOptimizationResult:
    """Result of thermal optimization."""
    action: ThermalAction
    predicted_temperature: float
    safety_score: float
    efficiency_score: float
    longevity_impact: float
    confidence: float
    execution_time: float
    metadata: Dict[str, Any]

class ThermalModel:
    """Physics-based thermal model for battery systems."""
    
    def __init__(self, thermal_params: Dict[str, float]):
        self.thermal_mass = thermal_params.get('thermal_mass', 1000.0)  # J/K
        self.thermal_resistance = thermal_params.get('thermal_resistance', 0.1)  # K/W
        self.ambient_resistance = thermal_params.get('ambient_resistance', 0.2)  # K/W
        self.heat_capacity = thermal_params.get('heat_capacity', 500.0)  # J/kg/K
        self.mass = thermal_params.get('mass', 2.0)  # kg
        
    def predict_temperature(self, current_temp: float, heat_generation: float, 
                          cooling_power: float, ambient_temp: float, 
                          time_step: float = 1.0) -> float:
        """Predict temperature change over time step."""
        
        # Net heat flow
        net_heat_flow = heat_generation - cooling_power
        
        # Heat transfer to ambient
        ambient_heat_transfer = (current_temp - ambient_temp) / self.ambient_resistance
        
        # Total heat change
        total_heat_change = net_heat_flow - ambient_heat_transfer
        
        # Temperature change
        temp_change = (total_heat_change * time_step) / self.thermal_mass
        
        return current_temp + temp_change
    
    def calculate_heat_generation(self, current: float, voltage: float, 
                                resistance: float) -> float:
        """Calculate heat generation from electrical losses."""
        return (current ** 2) * resistance

class ThermalOptimizer:
    """Main thermal optimization engine."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.constraints = ThermalConstraints(**config.get('constraints', {}))
        self.thermal_model = ThermalModel(config.get('thermal_params', {}))
        self.physics_simulator = BatteryPhysicsSimulator()
        self.efficiency_metrics = ThermalEfficiencyMetrics()
        
        # Initialize AI/ML models
        self._initialize_models()
        
        # Optimization parameters
        self.optimization_horizon = config.get('optimization_horizon', 60)  # minutes
        self.time_step = config.get('time_step', 1.0)  # seconds
        self.max_iterations = config.get('max_iterations', 100)
        self.convergence_threshold = config.get('convergence_threshold', 0.01)
        
        # Safety parameters
        self.safety_factor = config.get('safety_factor', 1.2)
        self.emergency_cooling_threshold = config.get('emergency_cooling_threshold', 65.0)
        
        # Performance tracking
        self.optimization_history = []
        self.safety_violations = []
        
        logger.info("ThermalOptimizer initialized successfully")
    
    def _initialize_models(self):
        """Initialize AI/ML models for thermal optimization."""
        try:
            # Load trained models
            self.health_predictor = BatteryHealthPredictor.load(
                self.config.get('health_predictor_path', 
                               '../../model-artifacts/trained_models/transformer_v1.0/model.pkl')
            )
            
            self.degradation_predictor = DegradationPredictor.load(
                self.config.get('degradation_predictor_path',
                               '../../model-artifacts/trained_models/transformer_v1.0/model.pkl')
            )
            
            # Initialize RL thermal agent
            self.thermal_agent = ThermalAgent(
                observation_space_size=8,
                action_space_size=4,
                **self.config.get('rl_agent_params', {})
            )
            
            # Load trained RL agent if available
            rl_model_path = self.config.get('rl_agent_path')
            if rl_model_path:
                self.thermal_agent.load_policy(rl_model_path)
            
            logger.info("AI/ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def optimize_thermal_management(self, thermal_state: ThermalState,
                                   battery_state: Dict[str, float],
                                   constraints: Optional[ThermalConstraints] = None,
                                   priority: ThermalPriority = ThermalPriority.SAFETY) -> ThermalOptimizationResult:
        """
        Optimize thermal management for given state and constraints.
        
        Args:
            thermal_state: Current thermal state
            battery_state: Current battery state (SoC, SoH, etc.)
            constraints: Optional thermal constraints
            priority: Optimization priority
            
        Returns:
            ThermalOptimizationResult: Optimization result
        """
        start_time = time.time()
        
        try:
            # Use provided constraints or default
            constraints = constraints or self.constraints
            
            # Safety check
            if thermal_state.temperature > constraints.critical_max:
                logger.warning(f"Critical temperature detected: {thermal_state.temperature}°C")
                return self._emergency_cooling(thermal_state, battery_state)
            
            # Choose optimization strategy based on priority
            if priority == ThermalPriority.SAFETY:
                result = self._safety_first_optimization(thermal_state, battery_state, constraints)
            elif priority == ThermalPriority.EFFICIENCY:
                result = self._efficiency_optimization(thermal_state, battery_state, constraints)
            elif priority == ThermalPriority.LONGEVITY:
                result = self._longevity_optimization(thermal_state, battery_state, constraints)
            elif priority == ThermalPriority.PERFORMANCE:
                result = self._performance_optimization(thermal_state, battery_state, constraints)
            else:
                result = self._multi_objective_optimization(thermal_state, battery_state, constraints)
            
            # Post-process and validate result
            result = self._validate_and_adjust_result(result, constraints)
            
            # Update history
            self.optimization_history.append({
                'timestamp': time.time(),
                'thermal_state': thermal_state,
                'battery_state': battery_state,
                'result': result,
                'priority': priority
            })
            
            result.execution_time = time.time() - start_time
            logger.info(f"Thermal optimization completed in {result.execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Thermal optimization failed: {e}")
            raise
    
    def _safety_first_optimization(self, thermal_state: ThermalState,
                                  battery_state: Dict[str, float],
                                  constraints: ThermalConstraints) -> ThermalOptimizationResult:
        """Safety-first thermal optimization."""
        
        # Predict temperature evolution
        predicted_temp = self._predict_temperature_evolution(
            thermal_state, battery_state, time_horizon=300  # 5 minutes
        )
        
        # Calculate required cooling to maintain safe temperature
        if predicted_temp > constraints.max_temperature:
            required_cooling = self._calculate_required_cooling(
                predicted_temp, constraints.max_temperature, thermal_state
            )
        else:
            required_cooling = 0.0
        
        # Calculate heating if temperature too low
        if predicted_temp < constraints.min_temperature:
            required_heating = self._calculate_required_heating(
                predicted_temp, constraints.min_temperature, thermal_state
            )
        else:
            required_heating = 0.0
        
        # Create thermal action
        action = ThermalAction(
            cooling_power=min(required_cooling, thermal_state.cooling_capacity),
            heating_power=min(required_heating, thermal_state.heating_capacity),
            fan_speed=min(required_cooling / thermal_state.cooling_capacity * 100, 100),
            strategy=ThermalStrategy.ADAPTIVE
        )
        
        # Predict result
        final_temp = self.thermal_model.predict_temperature(
            thermal_state.temperature,
            self._estimate_heat_generation(battery_state),
            action.cooling_power - action.heating_power,
            thermal_state.ambient_temperature
        )
        
        # Calculate scores
        safety_score = self._calculate_safety_score(final_temp, constraints)
        efficiency_score = self._calculate_efficiency_score(action, thermal_state)
        longevity_impact = self._calculate_longevity_impact(final_temp, battery_state)
        
        return ThermalOptimizationResult(
            action=action,
            predicted_temperature=final_temp,
            safety_score=safety_score,
            efficiency_score=efficiency_score,
            longevity_impact=longevity_impact,
            confidence=0.95,  # High confidence for safety-first approach
            execution_time=0.0,
            metadata={'strategy': 'safety_first', 'required_cooling': required_cooling}
        )
    
    def _efficiency_optimization(self, thermal_state: ThermalState,
                               battery_state: Dict[str, float],
                               constraints: ThermalConstraints) -> ThermalOptimizationResult:
        """Efficiency-focused thermal optimization."""
        
        # Use reinforcement learning agent for efficiency optimization
        state_vector = self._create_state_vector(thermal_state, battery_state)
        rl_action = self.thermal_agent.act(state_vector)
        
        # Convert RL action to thermal action
        action = self._convert_rl_action(rl_action, thermal_state)
        
        # Validate action against constraints
        action = self._enforce_constraints(action, constraints, thermal_state)
        
        # Predict result
        final_temp = self.thermal_model.predict_temperature(
            thermal_state.temperature,
            self._estimate_heat_generation(battery_state),
            action.cooling_power - action.heating_power,
            thermal_state.ambient_temperature
        )
        
        # Calculate scores
        safety_score = self._calculate_safety_score(final_temp, constraints)
        efficiency_score = self._calculate_efficiency_score(action, thermal_state)
        longevity_impact = self._calculate_longevity_impact(final_temp, battery_state)
        
        return ThermalOptimizationResult(
            action=action,
            predicted_temperature=final_temp,
            safety_score=safety_score,
            efficiency_score=efficiency_score,
            longevity_impact=longevity_impact,
            confidence=0.85,  # Lower confidence for RL-based approach
            execution_time=0.0,
            metadata={'strategy': 'efficiency', 'rl_action': rl_action.tolist()}
        )
    
    def _longevity_optimization(self, thermal_state: ThermalState,
                              battery_state: Dict[str, float],
                              constraints: ThermalConstraints) -> ThermalOptimizationResult:
        """Longevity-focused thermal optimization."""
        
        # Use degradation predictor to optimize for longevity
        current_degradation = self.degradation_predictor.predict_degradation(
            temperature=thermal_state.temperature,
            soc=battery_state.get('soc', 0.5),
            cycle_count=battery_state.get('cycle_count', 0)
        )
        
        # Find optimal temperature for minimal degradation
        optimal_temp = self._find_optimal_temperature_for_longevity(
            thermal_state, battery_state, constraints
        )
        
        # Calculate required thermal action
        temp_diff = optimal_temp - thermal_state.temperature
        
        if temp_diff > 0:  # Need heating
            required_heating = abs(temp_diff) * self.thermal_model.thermal_mass / 60  # W
            action = ThermalAction(
                heating_power=min(required_heating, thermal_state.heating_capacity),
                strategy=ThermalStrategy.PREDICTIVE
            )
        else:  # Need cooling
            required_cooling = abs(temp_diff) * self.thermal_model.thermal_mass / 60  # W
            action = ThermalAction(
                cooling_power=min(required_cooling, thermal_state.cooling_capacity),
                fan_speed=min(required_cooling / thermal_state.cooling_capacity * 100, 100),
                strategy=ThermalStrategy.PREDICTIVE
            )
        
        # Predict result
        final_temp = self.thermal_model.predict_temperature(
            thermal_state.temperature,
            self._estimate_heat_generation(battery_state),
            action.cooling_power - action.heating_power,
            thermal_state.ambient_temperature
        )
        
        # Calculate scores
        safety_score = self._calculate_safety_score(final_temp, constraints)
        efficiency_score = self._calculate_efficiency_score(action, thermal_state)
        longevity_impact = self._calculate_longevity_impact(final_temp, battery_state)
        
        return ThermalOptimizationResult(
            action=action,
            predicted_temperature=final_temp,
            safety_score=safety_score,
            efficiency_score=efficiency_score,
            longevity_impact=longevity_impact,
            confidence=0.88,
            execution_time=0.0,
            metadata={'strategy': 'longevity', 'optimal_temp': optimal_temp}
        )
    
    def _performance_optimization(self, thermal_state: ThermalState,
                                battery_state: Dict[str, float],
                                constraints: ThermalConstraints) -> ThermalOptimizationResult:
        """Performance-focused thermal optimization."""
        
        # Find optimal temperature for maximum performance
        performance_temp = self._find_optimal_temperature_for_performance(
            thermal_state, battery_state, constraints
        )
        
        # Calculate required thermal action
        temp_diff = performance_temp - thermal_state.temperature
        
        if temp_diff > 0:  # Need heating
            required_heating = abs(temp_diff) * self.thermal_model.thermal_mass / 30  # W (faster response)
            action = ThermalAction(
                heating_power=min(required_heating, thermal_state.heating_capacity),
                strategy=ThermalStrategy.ACTIVE_HEATING
            )
        else:  # Need cooling
            required_cooling = abs(temp_diff) * self.thermal_model.thermal_mass / 30  # W
            action = ThermalAction(
                cooling_power=min(required_cooling, thermal_state.cooling_capacity),
                fan_speed=min(required_cooling / thermal_state.cooling_capacity * 100, 100),
                strategy=ThermalStrategy.ACTIVE_COOLING
            )
        
        # Predict result
        final_temp = self.thermal_model.predict_temperature(
            thermal_state.temperature,
            self._estimate_heat_generation(battery_state),
            action.cooling_power - action.heating_power,
            thermal_state.ambient_temperature
        )
        
        # Calculate scores
        safety_score = self._calculate_safety_score(final_temp, constraints)
        efficiency_score = self._calculate_efficiency_score(action, thermal_state)
        longevity_impact = self._calculate_longevity_impact(final_temp, battery_state)
        
        return ThermalOptimizationResult(
            action=action,
            predicted_temperature=final_temp,
            safety_score=safety_score,
            efficiency_score=efficiency_score,
            longevity_impact=longevity_impact,
            confidence=0.80,
            execution_time=0.0,
            metadata={'strategy': 'performance', 'performance_temp': performance_temp}
        )
    
    def _multi_objective_optimization(self, thermal_state: ThermalState,
                                    battery_state: Dict[str, float],
                                    constraints: ThermalConstraints) -> ThermalOptimizationResult:
        """Multi-objective thermal optimization using weighted objectives."""
        
        # Define objective weights
        weights = {
            'safety': 0.4,
            'efficiency': 0.3,
            'longevity': 0.2,
            'performance': 0.1
        }
        
        # Get results from each optimization strategy
        safety_result = self._safety_first_optimization(thermal_state, battery_state, constraints)
        efficiency_result = self._efficiency_optimization(thermal_state, battery_state, constraints)
        longevity_result = self._longevity_optimization(thermal_state, battery_state, constraints)
        performance_result = self._performance_optimization(thermal_state, battery_state, constraints)
        
        # Calculate weighted scores for each action
        results = [safety_result, efficiency_result, longevity_result, performance_result]
        strategies = ['safety', 'efficiency', 'longevity', 'performance']
        
        best_score = -1
        best_result = safety_result  # Default to safety
        
        for result, strategy in zip(results, strategies):
            # Calculate composite score
            composite_score = (
                weights['safety'] * result.safety_score +
                weights['efficiency'] * result.efficiency_score +
                weights['longevity'] * result.longevity_impact +
                weights['performance'] * (1.0 - abs(result.predicted_temperature - 25.0) / 25.0)
            )
            
            if composite_score > best_score:
                best_score = composite_score
                best_result = result
        
        # Update metadata
        best_result.metadata['strategy'] = 'multi_objective'
        best_result.metadata['composite_score'] = best_score
        best_result.metadata['weights'] = weights
        
        return best_result
    
    def _emergency_cooling(self, thermal_state: ThermalState,
                          battery_state: Dict[str, float]) -> ThermalOptimizationResult:
        """Emergency cooling for critical temperatures."""
        
        logger.warning("Emergency cooling activated")
        
        # Maximum cooling action
        action = ThermalAction(
            cooling_power=thermal_state.cooling_capacity,
            fan_speed=100.0,
            coolant_flow=100.0,
            strategy=ThermalStrategy.ACTIVE_COOLING
        )
        
        # Predict emergency cooling result
        final_temp = self.thermal_model.predict_temperature(
            thermal_state.temperature,
            self._estimate_heat_generation(battery_state),
            action.cooling_power,
            thermal_state.ambient_temperature
        )
        
        # Record safety violation
        self.safety_violations.append({
            'timestamp': time.time(),
            'temperature': thermal_state.temperature,
            'action': 'emergency_cooling'
        })
        
        return ThermalOptimizationResult(
            action=action,
            predicted_temperature=final_temp,
            safety_score=0.0,  # Safety compromised
            efficiency_score=0.0,  # Efficiency not priority
            longevity_impact=-0.5,  # Negative impact
            confidence=1.0,  # High confidence in emergency action
            execution_time=0.0,
            metadata={'strategy': 'emergency', 'original_temp': thermal_state.temperature}
        )
    
    def _predict_temperature_evolution(self, thermal_state: ThermalState,
                                     battery_state: Dict[str, float],
                                     time_horizon: int = 300) -> float:
        """Predict temperature evolution over time horizon."""
        
        current_temp = thermal_state.temperature
        heat_gen = self._estimate_heat_generation(battery_state)
        
        # Simulate temperature evolution
        for _ in range(time_horizon):
            current_temp = self.thermal_model.predict_temperature(
                current_temp, heat_gen, 0.0, thermal_state.ambient_temperature, 1.0
            )
        
        return current_temp
    
    def _estimate_heat_generation(self, battery_state: Dict[str, float]) -> float:
        """Estimate heat generation based on battery state."""
        
        current = battery_state.get('current', 0.0)
        voltage = battery_state.get('voltage', 3.7)
        resistance = battery_state.get('internal_resistance', 0.1)
        
        # Joule heating
        joule_heat = (current ** 2) * resistance
        
        # Additional heat sources
        chemical_heat = abs(current) * 0.05  # Simplified chemical heat
        
        return joule_heat + chemical_heat
    
    def _calculate_required_cooling(self, predicted_temp: float,
                                  target_temp: float,
                                  thermal_state: ThermalState) -> float:
        """Calculate required cooling power."""
        
        temp_diff = predicted_temp - target_temp
        if temp_diff <= 0:
            return 0.0
        
        # Required cooling power
        cooling_power = temp_diff * self.thermal_model.thermal_mass / 60  # W
        
        return min(cooling_power, thermal_state.cooling_capacity)
    
    def _calculate_required_heating(self, predicted_temp: float,
                                  target_temp: float,
                                  thermal_state: ThermalState) -> float:
        """Calculate required heating power."""
        
        temp_diff = target_temp - predicted_temp
        if temp_diff <= 0:
            return 0.0
        
        # Required heating power
        heating_power = temp_diff * self.thermal_model.thermal_mass / 60  # W
        
        return min(heating_power, thermal_state.heating_capacity)
    
    def _create_state_vector(self, thermal_state: ThermalState,
                           battery_state: Dict[str, float]) -> np.ndarray:
        """Create state vector for RL agent."""
        
        return np.array([
            thermal_state.temperature / 100.0,  # Normalized temperature
            thermal_state.gradient / 10.0,      # Normalized gradient
            thermal_state.ambient_temperature / 50.0,  # Normalized ambient
            battery_state.get('soc', 0.5),      # State of charge
            battery_state.get('soh', 1.0),      # State of health
            battery_state.get('current', 0.0) / 100.0,  # Normalized current
            thermal_state.cooling_capacity / 1000.0,    # Normalized cooling capacity
            thermal_state.heating_capacity / 1000.0     # Normalized heating capacity
        ])
    
    def _convert_rl_action(self, rl_action: np.ndarray,
                          thermal_state: ThermalState) -> ThermalAction:
        """Convert RL action to thermal action."""
        
        # RL action: [cooling_power, heating_power, fan_speed, coolant_flow]
        return ThermalAction(
            cooling_power=rl_action[0] * thermal_state.cooling_capacity,
            heating_power=rl_action[1] * thermal_state.heating_capacity,
            fan_speed=rl_action[2] * 100.0,
            coolant_flow=rl_action[3] * 100.0,
            strategy=ThermalStrategy.ADAPTIVE
        )
    
    def _enforce_constraints(self, action: ThermalAction,
                           constraints: ThermalConstraints,
                           thermal_state: ThermalState) -> ThermalAction:
        """Enforce thermal constraints on action."""
        
        # Ensure cooling and heating are not active simultaneously
        if action.cooling_power > 0 and action.heating_power > 0:
            if action.cooling_power > action.heating_power:
                action.heating_power = 0.0
            else:
                action.cooling_power = 0.0
        
        # Enforce capacity limits
        action.cooling_power = min(action.cooling_power, thermal_state.cooling_capacity)
        action.heating_power = min(action.heating_power, thermal_state.heating_capacity)
        
        # Enforce range limits
        action.fan_speed = max(0.0, min(action.fan_speed, 100.0))
        action.coolant_flow = max(0.0, min(action.coolant_flow, 100.0))
        
        return action
    
    def _calculate_safety_score(self, temperature: float,
                              constraints: ThermalConstraints) -> float:
        """Calculate safety score based on temperature."""
        
        if temperature > constraints.critical_max:
            return 0.0
        elif temperature > constraints.max_temperature:
            return 0.5
        elif temperature < constraints.min_temperature:
            return 0.7
        elif constraints.optimal_min <= temperature <= constraints.optimal_max:
            return 1.0
        else:
            return 0.8
    
    def _calculate_efficiency_score(self, action: ThermalAction,
                                  thermal_state: ThermalState) -> float:
        """Calculate efficiency score based on action."""
        
        # Power consumption
        power_consumption = action.cooling_power + action.heating_power
        
        # Efficiency score (lower power consumption = higher efficiency)
        max_power = thermal_state.cooling_capacity + thermal_state.heating_capacity
        if max_power > 0:
            efficiency = 1.0 - (power_consumption / max_power)
        else:
            efficiency = 1.0
        
        return max(0.0, efficiency)
    
    def _calculate_longevity_impact(self, temperature: float,
                                  battery_state: Dict[str, float]) -> float:
        """Calculate longevity impact based on temperature."""
        
        # Temperature impact on longevity (simplified model)
        optimal_temp = 25.0  # °C
        temp_deviation = abs(temperature - optimal_temp)
        
        # Longevity impact (0 = negative, 1 = positive)
        longevity_impact = max(0.0, 1.0 - (temp_deviation / 30.0))
        
        return longevity_impact
    
    def _find_optimal_temperature_for_longevity(self, thermal_state: ThermalState,
                                              battery_state: Dict[str, float],
                                              constraints: ThermalConstraints) -> float:
        """Find optimal temperature for battery longevity."""
        
        # For longevity, typically want moderate temperatures
        optimal_temp = 25.0  # °C
        
        # Constrain within limits
        optimal_temp = max(constraints.min_temperature, 
                          min(optimal_temp, constraints.max_temperature))
        
        return optimal_temp
    
    def _find_optimal_temperature_for_performance(self, thermal_state: ThermalState,
                                                battery_state: Dict[str, float],
                                                constraints: ThermalConstraints) -> float:
        """Find optimal temperature for battery performance."""
        
        # For performance, want higher temperatures (within limits)
        optimal_temp = 35.0  # °C
        
        # Constrain within limits
        optimal_temp = max(constraints.min_temperature,
                          min(optimal_temp, constraints.max_temperature))
        
        return optimal_temp
    
    def _validate_and_adjust_result(self, result: ThermalOptimizationResult,
                                   constraints: ThermalConstraints) -> ThermalOptimizationResult:
        """Validate and adjust optimization result."""
        
        # Check predicted temperature against constraints
        if result.predicted_temperature > constraints.max_temperature:
            logger.warning(f"Predicted temperature {result.predicted_temperature}°C exceeds maximum")
            # Increase cooling
            result.action.cooling_power = min(
                result.action.cooling_power * 1.2,
                constraints.max_temperature * 10  # Simplified adjustment
            )
        
        if result.predicted_temperature < constraints.min_temperature:
            logger.warning(f"Predicted temperature {result.predicted_temperature}°C below minimum")
            # Increase heating
            result.action.heating_power = min(
                result.action.heating_power * 1.2,
                abs(constraints.min_temperature) * 10  # Simplified adjustment
            )
        
        return result
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self.optimization_history
    
    def get_safety_violations(self) -> List[Dict[str, Any]]:
        """Get safety violations."""
        return self.safety_violations
    
    def reset_history(self):
        """Reset optimization history."""
        self.optimization_history = []
        self.safety_violations = []
    
    def save_configuration(self, filepath: str):
        """Save thermal optimizer configuration."""
        config_data = {
            'constraints': self.constraints.__dict__,
            'optimization_params': {
                'optimization_horizon': self.optimization_horizon,
                'time_step': self.time_step,
                'max_iterations': self.max_iterations,
                'convergence_threshold': self.convergence_threshold
            },
            'thermal_model': {
                'thermal_mass': self.thermal_model.thermal_mass,
                'thermal_resistance': self.thermal_model.thermal_resistance,
                'ambient_resistance': self.thermal_model.ambient_resistance
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
    
    def load_configuration(self, filepath: str):
        """Load thermal optimizer configuration."""
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        self.constraints = ThermalConstraints(**config_data['constraints'])
        self.optimization_horizon = config_data['optimization_params']['optimization_horizon']
        self.time_step = config_data['optimization_params']['time_step']
        self.max_iterations = config_data['optimization_params']['max_iterations']
        self.convergence_threshold = config_data['optimization_params']['convergence_threshold']
        
        # Update thermal model
        self.thermal_model = ThermalModel(config_data['thermal_model'])
        
        logger.info(f"Configuration loaded from {filepath}")

class FleetThermalOptimizer:
    """Fleet-level thermal optimization coordinator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.individual_optimizers = {}
        self.fleet_constraints = ThermalConstraints(**config.get('fleet_constraints', {}))
        
        # Fleet coordination parameters
        self.coordination_enabled = config.get('coordination_enabled', True)
        self.thermal_sharing_enabled = config.get('thermal_sharing_enabled', True)
        self.max_concurrent_optimizations = config.get('max_concurrent_optimizations', 10)
        
        logger.info("FleetThermalOptimizer initialized")
    
    def add_battery(self, battery_id: str, battery_config: Dict[str, Any]):
        """Add a battery to the fleet."""
        self.individual_optimizers[battery_id] = ThermalOptimizer(battery_config)
        logger.info(f"Battery {battery_id} added to fleet")
    
    def optimize_fleet_thermal(self, fleet_states: Dict[str, Tuple[ThermalState, Dict[str, float]]],
                              priority: ThermalPriority = ThermalPriority.SAFETY) -> Dict[str, ThermalOptimizationResult]:
        """Optimize thermal management for entire fleet."""
        
        results = {}
        
        if self.coordination_enabled:
            # Coordinated optimization
            results = self._coordinated_optimization(fleet_states, priority)
        else:
            # Independent optimization
            with ThreadPoolExecutor(max_workers=self.max_concurrent_optimizations) as executor:
                futures = {}
                
                for battery_id, (thermal_state, battery_state) in fleet_states.items():
                    if battery_id in self.individual_optimizers:
                        future = executor.submit(
                            self.individual_optimizers[battery_id].optimize_thermal_management,
                            thermal_state, battery_state, self.fleet_constraints, priority
                        )
                        futures[battery_id] = future
                
                # Collect results
                for battery_id, future in futures.items():
                    try:
                        results[battery_id] = future.result()
                    except Exception as e:
                        logger.error(f"Optimization failed for battery {battery_id}: {e}")
        
        return results
    
    def _coordinated_optimization(self, fleet_states: Dict[str, Tuple[ThermalState, Dict[str, float]]],
                                priority: ThermalPriority) -> Dict[str, ThermalOptimizationResult]:
        """Perform coordinated thermal optimization across fleet."""
        
        # Calculate fleet-wide thermal load
        total_heat_generation = sum(
            optimizer._estimate_heat_generation(battery_state)
            for battery_id, (thermal_state, battery_state) in fleet_states.items()
            if battery_id in self.individual_optimizers
            for optimizer in [self.individual_optimizers[battery_id]]
        )
        
        # Identify critical batteries
        critical_batteries = []
        for battery_id, (thermal_state, battery_state) in fleet_states.items():
            if thermal_state.temperature > self.fleet_constraints.max_temperature:
                critical_batteries.append(battery_id)
        
        # Optimize with coordination
        results = {}
        
        # First, handle critical batteries
        for battery_id in critical_batteries:
            if battery_id in self.individual_optimizers:
                thermal_state, battery_state = fleet_states[battery_id]
                results[battery_id] = self.individual_optimizers[battery_id].optimize_thermal_management(
                    thermal_state, battery_state, self.fleet_constraints, ThermalPriority.SAFETY
                )
        
        # Then optimize remaining batteries
        for battery_id, (thermal_state, battery_state) in fleet_states.items():
            if battery_id not in critical_batteries and battery_id in self.individual_optimizers:
                results[battery_id] = self.individual_optimizers[battery_id].optimize_thermal_management(
                    thermal_state, battery_state, self.fleet_constraints, priority
                )
        
        return results
    
    def get_fleet_thermal_summary(self) -> Dict[str, Any]:
        """Get fleet-wide thermal summary."""
        
        total_optimizers = len(self.individual_optimizers)
        total_history = sum(len(opt.optimization_history) for opt in self.individual_optimizers.values())
        total_violations = sum(len(opt.safety_violations) for opt in self.individual_optimizers.values())
        
        return {
            'total_batteries': total_optimizers,
            'total_optimizations': total_history,
            'total_safety_violations': total_violations,
            'coordination_enabled': self.coordination_enabled,
            'thermal_sharing_enabled': self.thermal_sharing_enabled
        }
