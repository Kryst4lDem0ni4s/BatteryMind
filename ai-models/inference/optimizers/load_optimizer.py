"""
BatteryMind Load Optimizer

Advanced load balancing and power management optimization for battery systems
using AI/ML models to optimize load distribution and power allocation.

This module provides:
- Multi-battery load balancing
- Dynamic power allocation
- Peak shaving optimization
- Demand response coordination
- Grid integration optimization
- Fleet-level load management

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod
import time
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime, timedelta
import asyncio

# BatteryMind imports
from ..predictors.battery_health_predictor import BatteryHealthPredictor
from ..predictors.optimization_predictor import OptimizationPredictor
from ...transformers.optimization_recommender import OptimizationRecommender
from ...reinforcement_learning.agents.load_balancing_agent import LoadBalancingAgent
from ...training_data.generators.physics_simulator import BatteryPhysicsSimulator
from ...evaluation.metrics.efficiency_metrics import LoadBalancingMetrics
from ...utils.logging_utils import setup_logger
from ...utils.config_parser import ConfigParser

# Configure logging
logger = setup_logger(__name__)

class LoadStrategy(Enum):
    """Load balancing strategies."""
    EQUAL_DISTRIBUTION = "equal_distribution"
    CAPACITY_WEIGHTED = "capacity_weighted"
    HEALTH_WEIGHTED = "health_weighted"
    EFFICIENCY_OPTIMIZED = "efficiency_optimized"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"

class LoadPriority(Enum):
    """Load optimization priorities."""
    EFFICIENCY = "efficiency"
    LONGEVITY = "longevity"
    PERFORMANCE = "performance"
    COST = "cost"
    RELIABILITY = "reliability"

class LoadPhase(Enum):
    """Load phases for three-phase systems."""
    PHASE_A = "phase_a"
    PHASE_B = "phase_b"
    PHASE_C = "phase_c"
    BALANCED = "balanced"

@dataclass
class LoadConstraints:
    """Load balancing constraints."""
    min_power_per_battery: float = 0.0      # kW
    max_power_per_battery: float = 100.0    # kW
    total_power_limit: float = 1000.0       # kW
    max_imbalance_percent: float = 10.0     # %
    min_efficiency: float = 0.8             # 80%
    max_temperature_rise: float = 10.0      # °C
    safety_margin: float = 0.1              # 10%

@dataclass
class BatteryLoadState:
    """Current load state of a battery."""
    battery_id: str
    current_power: float        # kW
    max_power_capacity: float   # kW
    current_efficiency: float   # %
    temperature: float          # °C
    soc: float                  # %
    soh: float                  # %
    internal_resistance: float  # Ohms
    load_factor: float          # %
    phase: LoadPhase
    priority: int               # Priority level (1-10)
    timestamp: float

@dataclass
class LoadAllocation:
    """Load allocation for a battery."""
    battery_id: str
    allocated_power: float      # kW
    target_efficiency: float    # %
    expected_temperature: float # °C
    load_factor: float          # %
    phase: LoadPhase
    ramp_rate: float           # kW/s
    duration: float            # seconds
    confidence: float          # 0-1

@dataclass
class LoadOptimizationResult:
    """Result of load optimization."""
    allocations: List[LoadAllocation]
    total_power: float
    system_efficiency: float
    load_balance_score: float
    safety_score: float
    cost_score: float
    execution_time: float
    strategy_used: LoadStrategy
    metadata: Dict[str, Any]

class LoadPredictor:
    """Load demand prediction model."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.historical_data = []
        self.prediction_horizon = config.get('prediction_horizon', 3600)  # 1 hour
        
    def predict_load_demand(self, current_time: datetime,
                           historical_data: List[Dict[str, Any]]) -> List[float]:
        """Predict future load demand."""
        
        # Simple prediction based on historical patterns
        if not historical_data:
            return [100.0] * (self.prediction_horizon // 60)  # Default 100kW per minute
        
        # Extract load patterns
        loads = [d['total_power'] for d in historical_data[-60:]]  # Last hour
        
        # Simple moving average prediction
        if len(loads) > 0:
            avg_load = np.mean(loads)
            trend = np.mean(np.diff(loads)) if len(loads) > 1 else 0
            
            predictions = []
            for i in range(self.prediction_horizon // 60):
                predicted_load = avg_load + trend * i
                predictions.append(max(0, predicted_load))
            
            return predictions
        
        return [100.0] * (self.prediction_horizon // 60)
    
    def predict_peak_demand(self, current_time: datetime) -> Tuple[float, datetime]:
        """Predict next peak demand and timing."""
        
        # Simplified peak prediction
        hour = current_time.hour
        
        # Typical peak hours
        if 6 <= hour <= 8:  # Morning peak
            peak_time = current_time.replace(hour=7, minute=30, second=0, microsecond=0)
            peak_demand = 300.0  # kW
        elif 17 <= hour <= 19:  # Evening peak
            peak_time = current_time.replace(hour=18, minute=0, second=0, microsecond=0)
            peak_demand = 400.0  # kW
        else:
            # Next morning peak
            next_day = current_time + timedelta(days=1)
            peak_time = next_day.replace(hour=7, minute=30, second=0, microsecond=0)
            peak_demand = 300.0  # kW
        
        return peak_demand, peak_time

class LoadOptimizer:
    """Main load optimization engine."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.constraints = LoadConstraints(**config.get('constraints', {}))
        self.load_predictor = LoadPredictor(config.get('predictor_config', {}))
        self.physics_simulator = BatteryPhysicsSimulator()
        self.load_metrics = LoadBalancingMetrics()
        
        # Initialize AI/ML models
        self._initialize_models()
        
        # Optimization parameters
        self.optimization_window = config.get('optimization_window', 300)  # 5 minutes
        self.rebalance_threshold = config.get('rebalance_threshold', 0.1)  # 10%
        self.max_optimization_time = config.get('max_optimization_time', 5.0)  # seconds
        
        # Load balancing parameters
        self.imbalance_penalty = config.get('imbalance_penalty', 0.1)
        self.efficiency_weight = config.get('efficiency_weight', 0.4)
        self.longevity_weight = config.get('longevity_weight', 0.3)
        self.cost_weight = config.get('cost_weight', 0.3)
        
        # Performance tracking
        self.optimization_history = []
        self.load_history = []
        self.rebalance_events = []
        
        logger.info("LoadOptimizer initialized successfully")
    
    def _initialize_models(self):
        """Initialize AI/ML models for load optimization."""
        try:
            # Load trained models
            self.health_predictor = BatteryHealthPredictor.load(
                self.config.get('health_predictor_path',
                               '../../model-artifacts/trained_models/transformer_v1.0/model.pkl')
            )
            
            self.optimization_predictor = OptimizationPredictor.load(
                self.config.get('optimization_predictor_path',
                               '../../model-artifacts/trained_models/transformer_v1.0/model.pkl')
            )
            
            # Initialize RL load balancing agent
            self.load_balancing_agent = LoadBalancingAgent(
                observation_space_size=12,
                action_space_size=8,
                **self.config.get('rl_agent_params', {})
            )
            
            # Load trained RL agent if available
            rl_model_path = self.config.get('rl_agent_path')
            if rl_model_path:
                self.load_balancing_agent.load_policy(rl_model_path)
            
            logger.info("AI/ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def optimize_load_distribution(self, battery_states: List[BatteryLoadState],
                                  target_total_power: float,
                                  constraints: Optional[LoadConstraints] = None,
                                  strategy: LoadStrategy = LoadStrategy.ADAPTIVE,
                                  priority: LoadPriority = LoadPriority.EFFICIENCY) -> LoadOptimizationResult:
        """
        Optimize load distribution across batteries.
        
        Args:
            battery_states: List of current battery states
            target_total_power: Target total power output (kW)
            constraints: Optional load constraints
            strategy: Load balancing strategy
            priority: Optimization priority
            
        Returns:
            LoadOptimizationResult: Optimization result
        """
        start_time = time.time()
        
        try:
            # Use provided constraints or default
            constraints = constraints or self.constraints
            
            # Validate inputs
            if not battery_states:
                raise ValueError("No battery states provided")
            
            if target_total_power <= 0:
                raise ValueError("Target power must be positive")
            
            # Check if total power is achievable
            max_total_power = sum(battery.max_power_capacity for battery in battery_states)
            if target_total_power > max_total_power:
                logger.warning(f"Target power {target_total_power}kW exceeds maximum capacity {max_total_power}kW")
                target_total_power = max_total_power * 0.9  # 90% of max capacity
            
            # Choose optimization strategy
            if strategy == LoadStrategy.EQUAL_DISTRIBUTION:
                result = self._equal_distribution_optimization(battery_states, target_total_power, constraints)
            elif strategy == LoadStrategy.CAPACITY_WEIGHTED:
                result = self._capacity_weighted_optimization(battery_states, target_total_power, constraints)
            elif strategy == LoadStrategy.HEALTH_WEIGHTED:
                result = self._health_weighted_optimization(battery_states, target_total_power, constraints)
            elif strategy == LoadStrategy.EFFICIENCY_OPTIMIZED:
                result = self._efficiency_optimization(battery_states, target_total_power, constraints)
            elif strategy == LoadStrategy.PREDICTIVE:
                result = self._predictive_optimization(battery_states, target_total_power, constraints, priority)
            elif strategy == LoadStrategy.ADAPTIVE:
                result = self._adaptive_optimization(battery_states, target_total_power, constraints, priority)
            else:
                result = self._adaptive_optimization(battery_states, target_total_power, constraints, priority)
            
            # Post-process and validate result
            result = self._validate_and_adjust_result(result, constraints)
            
            # Update history
            self.optimization_history.append({
                'timestamp': time.time(),
                'battery_states': battery_states,
                'target_power': target_total_power,
                'result': result,
                'strategy': strategy,
                'priority': priority
            })
            
            result.execution_time = time.time() - start_time
            logger.info(f"Load optimization completed in {result.execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Load optimization failed: {e}")
            raise
    
    def _equal_distribution_optimization(self, battery_states: List[BatteryLoadState],
                                       target_total_power: float,
                                       constraints: LoadConstraints) -> LoadOptimizationResult:
        """Equal distribution load optimization."""
        
        active_batteries = [b for b in battery_states if b.soh > 0.7]  # Only healthy batteries
        
        if not active_batteries:
            raise ValueError("No healthy batteries available")
        
        # Equal power distribution
        power_per_battery = target_total_power / len(active_batteries)
        
        allocations = []
        for battery in active_batteries:
            # Respect individual battery limits
            allocated_power = min(power_per_battery, battery.max_power_capacity)
            allocated_power = max(allocated_power, constraints.min_power_per_battery)
            
            # Calculate expected efficiency and temperature
            load_factor = allocated_power / battery.max_power_capacity
            expected_efficiency = self._estimate_efficiency(battery, load_factor)
            expected_temperature = self._estimate_temperature(battery, load_factor)
            
            allocation = LoadAllocation(
                battery_id=battery.battery_id,
                allocated_power=allocated_power,
                target_efficiency=expected_efficiency,
                expected_temperature=expected_temperature,
                load_factor=load_factor,
                phase=battery.phase,
                ramp_rate=10.0,  # kW/s
                duration=300.0,  # 5 minutes
                confidence=0.9
            )
            allocations.append(allocation)
        
        # Calculate system metrics
        total_power = sum(a.allocated_power for a in allocations)
        system_efficiency = np.mean([a.target_efficiency for a in allocations])
        load_balance_score = self._calculate_load_balance_score(allocations)
        safety_score = self._calculate_safety_score(allocations)
        cost_score = self._calculate_cost_score(allocations)
        
        return LoadOptimizationResult(
            allocations=allocations,
            total_power=total_power,
            system_efficiency=system_efficiency,
            load_balance_score=load_balance_score,
            safety_score=safety_score,
            cost_score=cost_score,
            execution_time=0.0,
            strategy_used=LoadStrategy.EQUAL_DISTRIBUTION,
            metadata={'strategy': 'equal_distribution', 'active_batteries': len(active_batteries)}
        )
    
    def _capacity_weighted_optimization(self, battery_states: List[BatteryLoadState],
                                      target_total_power: float,
                                      constraints: LoadConstraints) -> LoadOptimizationResult:
        """Capacity-weighted load optimization."""
        
        active_batteries = [b for b in battery_states if b.soh > 0.7]
        
        if not active_batteries:
            raise ValueError("No healthy batteries available")
        
        # Calculate total capacity
        total_capacity = sum(b.max_power_capacity for b in active_batteries)
        
        allocations = []
        for battery in active_batteries:
            # Capacity-weighted allocation
            capacity_ratio = battery.max_power_capacity / total_capacity
            allocated_power = target_total_power * capacity_ratio
            
            # Respect constraints
            allocated_power = min(allocated_power, battery.max_power_capacity)
            allocated_power = max(allocated_power, constraints.min_power_per_battery)
            
            # Calculate metrics
            load_factor = allocated_power / battery.max_power_capacity
            expected_efficiency = self._estimate_efficiency(battery, load_factor)
            expected_temperature = self._estimate_temperature(battery, load_factor)
            
            allocation = LoadAllocation(
                battery_id=battery.battery_id,
                allocated_power=allocated_power,
                target_efficiency=expected_efficiency,
                expected_temperature=expected_temperature,
                load_factor=load_factor,
                phase=battery.phase,
                ramp_rate=8.0,  # kW/s
                duration=300.0,
                confidence=0.85
            )
            allocations.append(allocation)
        
        # Calculate system metrics
        total_power = sum(a.allocated_power for a in allocations)
        system_efficiency = np.mean([a.target_efficiency for a in allocations])
        load_balance_score = self._calculate_load_balance_score(allocations)
        safety_score = self._calculate_safety_score(allocations)
        cost_score = self._calculate_cost_score(allocations)
        
        return LoadOptimizationResult(
            allocations=allocations,
            total_power=total_power,
            system_efficiency=system_efficiency,
            load_balance_score=load_balance_score,
            safety_score=safety_score,
            cost_score=cost_score,
            execution_time=0.0,
            strategy_used=LoadStrategy.CAPACITY_WEIGHTED,
            metadata={'strategy': 'capacity_weighted', 'total_capacity': total_capacity}
        )
    
    def _health_weighted_optimization(self, battery_states: List[BatteryLoadState],
                                    target_total_power: float,
                                    constraints: LoadConstraints) -> LoadOptimizationResult:
        """Health-weighted load optimization."""
        
        active_batteries = [b for b in battery_states if b.soh > 0.7]
        
        if not active_batteries:
            raise ValueError("No healthy batteries available")
        
        # Calculate health weights (higher SoH = higher weight)
        health_weights = [b.soh for b in active_batteries]
        total_health_weight = sum(health_weights)
        
        allocations = []
        for i, battery in enumerate(active_batteries):
            # Health-weighted allocation
            health_ratio = health_weights[i] / total_health_weight
            allocated_power = target_total_power * health_ratio
            
            # Respect constraints
            allocated_power = min(allocated_power, battery.max_power_capacity)
            allocated_power = max(allocated_power, constraints.min_power_per_battery)
            
            # Calculate metrics
            load_factor = allocated_power / battery.max_power_capacity
            expected_efficiency = self._estimate_efficiency(battery, load_factor)
            expected_temperature = self._estimate_temperature(battery, load_factor)
            
            allocation = LoadAllocation(
                battery_id=battery.battery_id,
                allocated_power=allocated_power,
                target_efficiency=expected_efficiency,
                expected_temperature=expected_temperature,
                load_factor=load_factor,
                phase=battery.phase,
                ramp_rate=6.0,  # kW/s (slower for health preservation)
                duration=300.0,
                confidence=0.88
            )
            allocations.append(allocation)
        
        # Calculate system metrics
        total_power = sum(a.allocated_power for a in allocations)
        system_efficiency = np.mean([a.target_efficiency for a in allocations])
        load_balance_score = self._calculate_load_balance_score(allocations)
        safety_score = self._calculate_safety_score(allocations)
        cost_score = self._calculate_cost_score(allocations)
        
        return LoadOptimizationResult(
            allocations=allocations,
            total_power=total_power,
            system_efficiency=system_efficiency,
            load_balance_score=load_balance_score,
            safety_score=safety_score,
            cost_score=cost_score,
            execution_time=0.0,
            strategy_used=LoadStrategy.HEALTH_WEIGHTED,
            metadata={'strategy': 'health_weighted', 'avg_health': np.mean(health_weights)}
        )
    
    def _efficiency_optimization(self, battery_states: List[BatteryLoadState],
                               target_total_power: float,
                               constraints: LoadConstraints) -> LoadOptimizationResult:
        """Efficiency-optimized load distribution."""
        
        active_batteries = [b for b in battery_states if b.soh > 0.7]
        
        if not active_batteries:
            raise ValueError("No healthy batteries available")
        
        # Use optimization algorithm to find efficiency-optimal allocation
        best_allocation = self._optimize_for_efficiency(active_batteries, target_total_power, constraints)
        
        allocations = []
        for i, battery in enumerate(active_batteries):
            allocated_power = best_allocation[i]
            
            # Calculate metrics
            load_factor = allocated_power / battery.max_power_capacity
            expected_efficiency = self._estimate_efficiency(battery, load_factor)
            expected_temperature = self._estimate_temperature(battery, load_factor)
            
            allocation = LoadAllocation(
                battery_id=battery.battery_id,
                allocated_power=allocated_power,
                target_efficiency=expected_efficiency,
                expected_temperature=expected_temperature,
                load_factor=load_factor,
                phase=battery.phase,
                ramp_rate=12.0,  # kW/s
                duration=300.0,
                confidence=0.92
            )
            allocations.append(allocation)
        
        # Calculate system metrics
        total_power = sum(a.allocated_power for a in allocations)
        system_efficiency = np.mean([a.target_efficiency for a in allocations])
        load_balance_score = self._calculate_load_balance_score(allocations)
        safety_score = self._calculate_safety_score(allocations)
        cost_score = self._calculate_cost_score(allocations)
        
        return LoadOptimizationResult(
            allocations=allocations,
            total_power=total_power,
            system_efficiency=system_efficiency,
            load_balance_score=load_balance_score,
            safety_score=safety_score,
            cost_score=cost_score,
            execution_time=0.0,
                    strategy_used=LoadStrategy.EFFICIENCY_OPTIMIZED,
        metadata={'strategy': 'efficiency_optimized', 'optimization_algorithm': 'scipy_minimize'}
    )

    def _predictive_optimization(self, battery_states: List[BatteryLoadState],
                            target_total_power: float,
                            constraints: LoadConstraints,
                            priority: LoadPriority) -> LoadOptimizationResult:
        """Predictive load optimization using AI/ML models."""
        
        # Create state vector for RL agent
        state_vector = self._create_state_vector(battery_states, target_total_power)
        
        # Get action from RL agent
        rl_action = self.load_balancing_agent.act(state_vector)
        
        # Convert RL action to load allocations
        allocations = self._convert_rl_action_to_allocations(rl_action, battery_states, target_total_power)
        
        # Validate and adjust allocations
        allocations = self._validate_allocations(allocations, constraints)
        
        # Calculate system metrics
        total_power = sum(a.allocated_power for a in allocations)
        system_efficiency = np.mean([a.target_efficiency for a in allocations])
        load_balance_score = self._calculate_load_balance_score(allocations)
        safety_score = self._calculate_safety_score(allocations)
        cost_score = self._calculate_cost_score(allocations)
        
        # Calculate composite score
        composite_score = (
            0.4 * system_efficiency +
            0.25 * load_balance_score +
            0.25 * safety_score +
            0.1 * cost_score
        )
        
        # Predict future performance
        future_performance = self._predict_future_performance(allocations, battery_states)
        
        return LoadOptimizationResult(
            allocations=allocations,
            total_power=total_power,
            system_efficiency=system_efficiency,
            load_balance_score=load_balance_score,
            safety_score=safety_score,
            cost_score=cost_score,
            composite_score=composite_score,
            convergence_iterations=1,
            optimization_time=time.time() - start_time,
            strategy_used=LoadStrategy.PREDICTIVE_AI,
            metadata={
                'strategy': 'predictive_ai',
                'rl_agent': 'load_balancing_agent',
                'future_performance': future_performance
            }
        )

    def _create_state_vector(self, battery_states: List[BatteryLoadState], 
                            target_total_power: float) -> np.ndarray:
        """Create state vector for RL agent."""
        state_features = []
        
        # Global features
        state_features.extend([
            target_total_power,
            len(battery_states),
            np.mean([bs.current_soc for bs in battery_states]),
            np.mean([bs.temperature for bs in battery_states]),
            np.mean([bs.voltage for bs in battery_states])
        ])
        
        # Battery-specific features (normalized)
        for battery_state in battery_states:
            state_features.extend([
                battery_state.current_soc,
                battery_state.soh,
                battery_state.temperature,
                battery_state.voltage,
                battery_state.max_power_capacity,
                battery_state.available_capacity,
                battery_state.internal_resistance,
                battery_state.charge_cycles
            ])
        
        # Pad or truncate to fixed size
        max_batteries = 20
        expected_size = 5 + max_batteries * 8  # 5 global + 8 per battery
        
        if len(state_features) > expected_size:
            state_features = state_features[:expected_size]
        elif len(state_features) < expected_size:
            state_features.extend([0.0] * (expected_size - len(state_features)))
        
        return np.array(state_features, dtype=np.float32)

    def _convert_rl_action_to_allocations(self, rl_action: np.ndarray, 
                                        battery_states: List[BatteryLoadState],
                                        target_total_power: float) -> List[BatteryLoadAllocation]:
        """Convert RL agent action to load allocations."""
        allocations = []
        
        # RL action represents normalized power distribution
        normalized_actions = np.clip(rl_action[:len(battery_states)], 0, 1)
        
        # Normalize to sum to 1
        if np.sum(normalized_actions) > 0:
            normalized_actions = normalized_actions / np.sum(normalized_actions)
        else:
            normalized_actions = np.ones(len(battery_states)) / len(battery_states)
        
        # Convert to actual power allocations
        for i, (battery_state, norm_action) in enumerate(zip(battery_states, normalized_actions)):
            allocated_power = norm_action * target_total_power
            
            # Ensure allocation is within battery limits
            allocated_power = min(allocated_power, battery_state.max_power_capacity)
            
            # Calculate expected efficiency
            efficiency = self._calculate_expected_efficiency(battery_state, allocated_power)
            
            allocation = BatteryLoadAllocation(
                battery_id=battery_state.battery_id,
                allocated_power=allocated_power,
                target_efficiency=efficiency,
                expected_temperature_rise=self._calculate_temperature_rise(battery_state, allocated_power),
                safety_margin=self._calculate_safety_margin(battery_state, allocated_power),
                cost_per_kwh=self._calculate_cost_per_kwh(battery_state, allocated_power)
            )
            
            allocations.append(allocation)
        
        return allocations

    def _validate_allocations(self, allocations: List[BatteryLoadAllocation], 
                            constraints: LoadConstraints) -> List[BatteryLoadAllocation]:
        """Validate and adjust load allocations to meet constraints."""
        validated_allocations = []
        
        for allocation in allocations:
            # Check power limits
            if allocation.allocated_power > constraints.max_power_per_battery:
                allocation.allocated_power = constraints.max_power_per_battery
            
            # Check temperature constraints
            if allocation.expected_temperature_rise > constraints.max_temperature_rise:
                # Reduce power to meet temperature constraint
                power_reduction = 0.8  # Reduce by 20%
                allocation.allocated_power *= power_reduction
                allocation.expected_temperature_rise = self._calculate_temperature_rise(
                    None, allocation.allocated_power
                )
            
            # Check safety margin
            if allocation.safety_margin < constraints.min_safety_margin:
                # Reduce power to increase safety margin
                power_reduction = 0.9  # Reduce by 10%
                allocation.allocated_power *= power_reduction
                allocation.safety_margin = self._calculate_safety_margin(
                    None, allocation.allocated_power
                )
            
            # Recalculate efficiency with adjusted power
            allocation.target_efficiency = self._calculate_expected_efficiency(
                None, allocation.allocated_power
            )
            
            validated_allocations.append(allocation)
        
        return validated_allocations

    def _calculate_load_balance_score(self, allocations: List[BatteryLoadAllocation]) -> float:
        """Calculate load balance score (higher is better)."""
        if not allocations:
            return 0.0
        
        powers = [a.allocated_power for a in allocations]
        mean_power = np.mean(powers)
        
        if mean_power == 0:
            return 1.0
        
        # Calculate coefficient of variation
        cv = np.std(powers) / mean_power
        
        # Convert to score (lower CV = higher score)
        return 1.0 / (1.0 + cv)

    def _calculate_safety_score(self, allocations: List[BatteryLoadAllocation]) -> float:
        """Calculate system safety score (higher is better)."""
        if not allocations:
            return 0.0
        
        safety_margins = [a.safety_margin for a in allocations]
        return np.mean(safety_margins)

    def _calculate_cost_score(self, allocations: List[BatteryLoadAllocation]) -> float:
        """Calculate cost efficiency score (higher is better)."""
        if not allocations:
            return 0.0
        
        costs = [a.cost_per_kwh for a in allocations]
        max_cost = max(costs) if costs else 1.0
        
        # Normalize costs (lower cost = higher score)
        normalized_costs = [1.0 - (cost / max_cost) for cost in costs]
        return np.mean(normalized_costs)

    def _calculate_expected_efficiency(self, battery_state: Optional[BatteryLoadState], 
                                    allocated_power: float) -> float:
        """Calculate expected efficiency for given power allocation."""
        if battery_state is None:
            return 0.85  # Default efficiency
        
        # Efficiency model based on power level and battery state
        optimal_power = battery_state.max_power_capacity * 0.7  # 70% of max capacity
        power_ratio = allocated_power / optimal_power if optimal_power > 0 else 0
        
        # Gaussian efficiency curve
        base_efficiency = 0.95
        efficiency_drop = 0.15 * np.exp(-0.5 * ((power_ratio - 1) / 0.3) ** 2)
        
        # Temperature effect
        temp_effect = 1.0 - max(0, (battery_state.temperature - 25) / 100)
        
        # SoH effect
        soh_effect = battery_state.soh
        
        final_efficiency = base_efficiency * efficiency_drop * temp_effect * soh_effect
        return np.clip(final_efficiency, 0.5, 0.98)

    def _calculate_temperature_rise(self, battery_state: Optional[BatteryLoadState], 
                                allocated_power: float) -> float:
        """Calculate expected temperature rise for given power allocation."""
        if battery_state is None:
            return allocated_power * 0.01  # Default: 1°C per 100W
        
        # Temperature rise model
        thermal_resistance = 0.1  # °C/W
        internal_resistance = battery_state.internal_resistance
        
        # Power loss calculation
        current = allocated_power / battery_state.voltage if battery_state.voltage > 0 else 0
        power_loss = current ** 2 * internal_resistance
        
        # Temperature rise
        temp_rise = power_loss * thermal_resistance
        
        return min(temp_rise, 50.0)  # Cap at 50°C rise

    def _calculate_safety_margin(self, battery_state: Optional[BatteryLoadState], 
                                allocated_power: float) -> float:
        """Calculate safety margin for given power allocation."""
        if battery_state is None:
            return 0.8  # Default safety margin
        
        # Safety margin based on power utilization
        power_utilization = allocated_power / battery_state.max_power_capacity
        
        # Higher utilization = lower safety margin
        base_safety = 1.0 - power_utilization
        
        # Adjust for temperature
        temp_factor = 1.0 - max(0, (battery_state.temperature - 40) / 60)
        
        # Adjust for SoH
        soh_factor = battery_state.soh
        
        safety_margin = base_safety * temp_factor * soh_factor
        return np.clip(safety_margin, 0.0, 1.0)

    def _calculate_cost_per_kwh(self, battery_state: Optional[BatteryLoadState], 
                            allocated_power: float) -> float:
        """Calculate cost per kWh for given power allocation."""
        if battery_state is None:
            return 0.15  # Default cost in $/kWh
        
        # Base cost
        base_cost = 0.10  # $/kWh
        
        # Efficiency penalty
        efficiency = self._calculate_expected_efficiency(battery_state, allocated_power)
        efficiency_penalty = (1.0 - efficiency) * 0.05
        
        # Degradation penalty
        degradation_penalty = (1.0 - battery_state.soh) * 0.03
        
        # Temperature penalty
        temp_penalty = max(0, (battery_state.temperature - 35) / 100) * 0.02
        
        total_cost = base_cost + efficiency_penalty + degradation_penalty + temp_penalty
        return min(total_cost, 0.30)  # Cap at $0.30/kWh

    def _predict_future_performance(self, allocations: List[BatteryLoadAllocation], 
                                battery_states: List[BatteryLoadState]) -> Dict[str, float]:
        """Predict future system performance with current allocations."""
        future_metrics = {}
        
        # Predict efficiency degradation over time
        current_efficiency = np.mean([a.target_efficiency for a in allocations])
        future_metrics['efficiency_24h'] = current_efficiency * 0.98  # 2% degradation
        future_metrics['efficiency_7d'] = current_efficiency * 0.95   # 5% degradation
        
        # Predict temperature evolution
        current_temp_rise = np.mean([a.expected_temperature_rise for a in allocations])
        future_metrics['temperature_rise_24h'] = current_temp_rise * 1.1  # 10% increase
        
        # Predict cost evolution
        current_cost = np.mean([a.cost_per_kwh for a in allocations])
        future_metrics['cost_24h'] = current_cost * 1.05  # 5% increase
        
        # Predict safety margin evolution
        current_safety = np.mean([a.safety_margin for a in allocations])
        future_metrics['safety_margin_24h'] = current_safety * 0.95  # 5% decrease
        
        return future_metrics

    def get_optimization_history(self) -> List[Dict]:
        """Get optimization history for analysis."""
        return self.optimization_history

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.optimization_history:
            return {}
        
        latest_result = self.optimization_history[-1]
        return {
            'system_efficiency': latest_result['result'].system_efficiency,
            'load_balance_score': latest_result['result'].load_balance_score,
            'safety_score': latest_result['result'].safety_score,
            'cost_score': latest_result['result'].cost_score,
            'composite_score': latest_result['result'].composite_score,
            'optimization_time': latest_result['result'].optimization_time
        }

    def reset_optimization_history(self):
        """Reset optimization history."""
        self.optimization_history = []
        self.logger.info("Optimization history reset")

    def validate_configuration(self) -> Dict[str, bool]:
        """Validate optimizer configuration."""
        validation_results = {}
        
        # Check if models are loaded
        validation_results['transformer_model_loaded'] = self.transformer_model is not None
        validation_results['ensemble_model_loaded'] = self.ensemble_model is not None
        validation_results['load_balancing_agent_loaded'] = self.load_balancing_agent is not None
        
        # Check configuration values
        validation_results['valid_max_iterations'] = self.max_iterations > 0
        validation_results['valid_convergence_tolerance'] = self.convergence_tolerance > 0
        validation_results['valid_default_strategy'] = self.default_strategy in LoadStrategy
        
        return validation_results

    def __str__(self) -> str:
        """String representation of the optimizer."""
        return f"LoadOptimizer(strategy={self.default_strategy}, models_loaded={self.transformer_model is not None})"

    def __repr__(self) -> str:
        """Detailed representation of the optimizer."""
        return (f"LoadOptimizer(default_strategy={self.default_strategy}, "
                f"max_iterations={self.max_iterations}, "
                f"convergence_tolerance={self.convergence_tolerance}, "
                f"transformer_model={'loaded' if self.transformer_model else 'not_loaded'}, "
                f"ensemble_model={'loaded' if self.ensemble_model else 'not_loaded'}, "
                f"load_balancing_agent={'loaded' if self.load_balancing_agent else 'not_loaded'})")
