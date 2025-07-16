"""
BatteryMind - Optimization Predictor

Advanced optimization recommendation system for battery charging and thermal
management using reinforcement learning agents, physics-based constraints,
and multi-objective optimization techniques.

Features:
- Real-time charging optimization recommendations
- Thermal management optimization
- Multi-objective optimization (health, efficiency, safety)
- Physics-informed constraint handling
- Uncertainty-aware optimization decisions
- Integration with RL agents and transformer predictions

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json
import pickle
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import cvxpy as cp
from sklearn.preprocessing import StandardScaler

# Import from BatteryMind modules
from ..reinforcement_learning.agents.charging_agent import ChargingAgent
from ..reinforcement_learning.agents.thermal_agent import ThermalAgent
from ..reinforcement_learning.environments.battery_env import BatteryEnvironment
from ..transformers.optimization_recommender.model import OptimizationTransformer
from ..transformers.optimization_recommender.recommender import OptimizationRecommender as BaseRecommender
from ..utils.model_utils import ModelLoader, ModelValidator
from ..utils.data_utils import DataProcessor, OptimizationProcessor
from ..utils.logging_utils import setup_logger

# Configure logging
logger = setup_logger(__name__)

@dataclass
class OptimizationPredictionConfig:
    """
    Configuration for optimization prediction parameters.
    
    Attributes:
        # Model configuration
        rl_agent_path (str): Path to trained RL agent
        transformer_path (str): Path to optimization transformer
        ensemble_mode (bool): Use ensemble of optimization methods
        
        # Optimization parameters
        objectives (List[str]): Optimization objectives
        constraints (Dict[str, Any]): Optimization constraints
        optimization_horizon (int): Optimization horizon in hours
        
        # RL agent settings
        agent_type (str): Type of RL agent ('charging', 'thermal', 'multi')
        action_space_size (int): Action space dimensionality
        state_space_size (int): State space dimensionality
        
        # Physics constraints
        voltage_limits (Tuple[float, float]): Voltage limits (V)
        current_limits (Tuple[float, float]): Current limits (A)
        temperature_limits (Tuple[float, float]): Temperature limits (°C)
        soc_limits (Tuple[float, float]): SoC limits (0-1)
        
        # Performance settings
        optimization_method (str): Optimization method
        max_iterations (int): Maximum optimization iterations
        convergence_threshold (float): Convergence threshold
        
        # Safety settings
        safety_margin (float): Safety margin factor
        emergency_protocols (bool): Enable emergency protocols
        fail_safe_mode (bool): Enable fail-safe mode
        
        # Monitoring
        enable_monitoring (bool): Enable optimization monitoring
        log_decisions (bool): Log optimization decisions
        performance_tracking (bool): Track optimization performance
    """
    # Model configuration
    rl_agent_path: str = "./model-artifacts/trained_models/rl_agent_v1.0/"
    transformer_path: str = "./model-artifacts/trained_models/transformer_v1.0/"
    ensemble_mode: bool = True
    
    # Optimization parameters
    objectives: List[str] = field(default_factory=lambda: [
        'battery_health', 'energy_efficiency', 'charging_speed', 'safety'
    ])
    constraints: Dict[str, Any] = field(default_factory=dict)
    optimization_horizon: int = 24  # hours
    
    # RL agent settings
    agent_type: str = "charging"
    action_space_size: int = 3
    state_space_size: int = 16
    
    # Physics constraints
    voltage_limits: Tuple[float, float] = (2.5, 4.2)
    current_limits: Tuple[float, float] = (-100.0, 100.0)
    temperature_limits: Tuple[float, float] = (-20.0, 60.0)
    soc_limits: Tuple[float, float] = (0.05, 0.95)
    
    # Performance settings
    optimization_method: str = "multi_objective"
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    
    # Safety settings
    safety_margin: float = 0.1
    emergency_protocols: bool = True
    fail_safe_mode: bool = True
    
    # Monitoring
    enable_monitoring: bool = True
    log_decisions: bool = True
    performance_tracking: bool = True

class OptimizationPredictor:
    """
    Production-ready optimization predictor with multi-objective recommendations.
    """
    
    def __init__(self, config: OptimizationPredictionConfig):
        self.config = config
        self.rl_agent = None
        self.transformer_model = None
        self.battery_env = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimization_history = []
        self.monitoring_stats = {}
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"OptimizationPredictor initialized with device: {self.device}")
    
    def _initialize_components(self):
        """Initialize all optimization components."""
        # Load RL agent
        self._load_rl_agent()
        
        # Load transformer model
        self._load_transformer_model()
        
        # Initialize battery environment
        self._initialize_battery_environment()
        
        # Setup monitoring
        if self.config.enable_monitoring:
            self._setup_monitoring()
    
    def _load_rl_agent(self):
        """Load the trained RL agent."""
        try:
            agent_path = Path(self.config.rl_agent_path)
            
            if self.config.agent_type == "charging":
                self.rl_agent = ChargingAgent.load(agent_path)
            elif self.config.agent_type == "thermal":
                self.rl_agent = ThermalAgent.load(agent_path)
            else:
                # Load multi-agent system
                self.rl_agent = self._load_multi_agent_system(agent_path)
            
            logger.info(f"RL agent loaded from {agent_path}")
        except Exception as e:
            logger.error(f"Failed to load RL agent: {e}")
            raise
    
    def _load_transformer_model(self):
        """Load the optimization transformer model."""
        try:
            model_loader = ModelLoader(self.config.transformer_path)
            self.transformer_model = model_loader.load_model("optimization_transformer")
            self.transformer_model.to(self.device)
            self.transformer_model.eval()
            
            logger.info(f"Transformer model loaded from {self.config.transformer_path}")
        except Exception as e:
            logger.warning(f"Failed to load transformer model: {e}")
            # Continue without transformer if RL agent is available
    
    def _load_multi_agent_system(self, agent_path: Path):
        """Load multi-agent system."""
        # This would load a multi-agent system with coordination
        # For now, return a placeholder
        return None
    
    def _initialize_battery_environment(self):
        """Initialize battery environment for optimization."""
        self.battery_env = BatteryEnvironment(
            state_space_size=self.config.state_space_size,
            action_space_size=self.config.action_space_size,
            voltage_limits=self.config.voltage_limits,
            current_limits=self.config.current_limits,
            temperature_limits=self.config.temperature_limits
        )
    
    def _setup_monitoring(self):
        """Setup optimization monitoring."""
        self.monitoring_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'average_optimization_time': 0.0,
            'objective_improvements': [],
            'constraint_violations': [],
            'safety_interventions': 0
        }
    
    def predict_optimization(self, 
                           battery_state: Dict[str, Any],
                           optimization_goals: Optional[Dict[str, float]] = None,
                           constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict optimal charging/thermal management actions.
        
        Args:
            battery_state: Current battery state
            optimization_goals: Optimization goals and weights
            constraints: Additional constraints
            
        Returns:
            Dictionary containing optimization recommendations
        """
        start_time = datetime.now()
        
        try:
            # Validate input
            self._validate_battery_state(battery_state)
            
            # Set default goals if not provided
            if optimization_goals is None:
                optimization_goals = self._get_default_optimization_goals()
            
            # Combine constraints
            combined_constraints = self._combine_constraints(constraints)
            
            # Generate optimization recommendations
            recommendations = self._generate_optimization_recommendations(
                battery_state, optimization_goals, combined_constraints
            )
            
            # Validate recommendations
            self._validate_recommendations(recommendations, battery_state)
            
            # Apply safety checks
            recommendations = self._apply_safety_checks(recommendations, battery_state)
            
            # Update monitoring
            self._update_monitoring_stats(start_time, True)
            
            # Log decisions if enabled
            if self.config.log_decisions:
                self._log_optimization_decision(battery_state, recommendations)
            
            return {
                'recommendations': recommendations,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'optimization_method': self.config.optimization_method,
                    'objectives': list(optimization_goals.keys()),
                    'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                    'safety_checks_passed': True
                }
            }
            
        except Exception as e:
            logger.error(f"Optimization prediction failed: {e}")
            self._update_monitoring_stats(start_time, False)
            
            # Return fail-safe recommendations
            if self.config.fail_safe_mode:
                return self._generate_fail_safe_recommendations(battery_state)
            else:
                raise
    
    def _validate_battery_state(self, battery_state: Dict[str, Any]):
        """Validate battery state input."""
        required_keys = ['voltage', 'current', 'temperature', 'soc', 'soh']
        missing_keys = [key for key in required_keys if key not in battery_state]
        
        if missing_keys:
            raise ValueError(f"Missing required battery state keys: {missing_keys}")
        
        # Validate value ranges
        if not (self.config.voltage_limits[0] <= battery_state['voltage'] <= self.config.voltage_limits[1]):
            raise ValueError(f"Voltage out of range: {battery_state['voltage']}")
        
        if not (self.config.soc_limits[0] <= battery_state['soc'] <= self.config.soc_limits[1]):
            raise ValueError(f"SoC out of range: {battery_state['soc']}")
        
        if not (self.config.temperature_limits[0] <= battery_state['temperature'] <= self.config.temperature_limits[1]):
            raise ValueError(f"Temperature out of range: {battery_state['temperature']}")
    
    def _get_default_optimization_goals(self) -> Dict[str, float]:
        """Get default optimization goals and weights."""
        return {
            'battery_health': 0.4,
            'energy_efficiency': 0.3,
            'charging_speed': 0.2,
            'safety': 0.1
        }
    
    def _combine_constraints(self, additional_constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine default and additional constraints."""
        constraints = {
            'voltage_min': self.config.voltage_limits[0],
            'voltage_max': self.config.voltage_limits[1],
            'current_min': self.config.current_limits[0],
            'current_max': self.config.current_limits[1],
            'temperature_min': self.config.temperature_limits[0],
            'temperature_max': self.config.temperature_limits[1],
            'soc_min': self.config.soc_limits[0],
            'soc_max': self.config.soc_limits[1]
        }
        
        if additional_constraints:
            constraints.update(additional_constraints)
        
        return constraints
    
    def _generate_optimization_recommendations(self, 
                                            battery_state: Dict[str, Any],
                                            optimization_goals: Dict[str, float],
                                            constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization recommendations using available methods."""
        recommendations = {}
        
        # RL-based recommendations
        if self.rl_agent:
            rl_recommendations = self._get_rl_recommendations(battery_state)
            recommendations['rl_agent'] = rl_recommendations
        
        # Transformer-based recommendations
        if self.transformer_model:
            transformer_recommendations = self._get_transformer_recommendations(battery_state)
            recommendations['transformer'] = transformer_recommendations
        
        # Physics-based optimization
        physics_recommendations = self._get_physics_based_recommendations(
            battery_state, optimization_goals, constraints
        )
        recommendations['physics_based'] = physics_recommendations
        
        # Ensemble recommendations
        if self.config.ensemble_mode and len(recommendations) > 1:
            ensemble_recommendations = self._combine_recommendations(recommendations)
            recommendations['ensemble'] = ensemble_recommendations
            recommendations['final'] = ensemble_recommendations
        else:
            # Use best available method
            if 'rl_agent' in recommendations:
                recommendations['final'] = recommendations['rl_agent']
            elif 'transformer' in recommendations:
                recommendations['final'] = recommendations['transformer']
            else:
                recommendations['final'] = recommendations['physics_based']
        
        return recommendations
    
    def _get_rl_recommendations(self, battery_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations from RL agent."""
        # Convert battery state to environment state
        env_state = self._convert_to_env_state(battery_state)
        
        # Get action from RL agent
        with torch.no_grad():
            action = self.rl_agent.predict(env_state)
        
        # Convert action to recommendations
        recommendations = self._convert_action_to_recommendations(action)
        
        return recommendations
    
    def _get_transformer_recommendations(self, battery_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations from transformer model."""
        # Prepare input for transformer
        transformer_input = self._prepare_transformer_input(battery_state)
        
        # Get recommendations
        with torch.no_grad():
            recommendations = self.transformer_model(transformer_input)
        
        return self._process_transformer_output(recommendations)
    
    def _get_physics_based_recommendations(self, 
                                        battery_state: Dict[str, Any],
                                        optimization_goals: Dict[str, float],
                                        constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations using physics-based optimization."""
        
        # Define optimization variables
        charging_current = cp.Variable()
        thermal_control = cp.Variable()
        
        # Define objective function
        objective = self._define_physics_objective(
            charging_current, thermal_control, battery_state, optimization_goals
        )
        
        # Define constraints
        physics_constraints = self._define_physics_constraints(
            charging_current, thermal_control, battery_state, constraints
        )
        
        # Solve optimization problem
        problem = cp.Problem(cp.Maximize(objective), physics_constraints)
        problem.solve()
        
        if problem.status == cp.OPTIMAL:
            recommendations = {
                'charging_current': charging_current.value,
                'thermal_control': thermal_control.value,
                'optimization_status': 'optimal',
                'objective_value': problem.value
            }
        else:
            # Fallback to heuristic solution
            recommendations = self._get_heuristic_recommendations(battery_state)
            recommendations['optimization_status'] = 'heuristic'
        
        return recommendations
    
    def _define_physics_objective(self, 
                                charging_current: cp.Variable,
                                thermal_control: cp.Variable,
                                battery_state: Dict[str, Any],
                                optimization_goals: Dict[str, float]) -> cp.Expression:
        """Define physics-based objective function."""
        
        # Battery health objective (minimize degradation)
        health_objective = -cp.square(charging_current) * 0.01  # Penalize high current
        
        # Energy efficiency objective
        efficiency_objective = charging_current * 0.9  # Reward efficient charging
        
        # Safety objective (maintain safe temperature)
        safety_objective = -cp.square(thermal_control) * 0.05  # Penalize extreme thermal control
        
        # Combine objectives with weights
        total_objective = (
            optimization_goals['battery_health'] * health_objective +
            optimization_goals['energy_efficiency'] * efficiency_objective +
            optimization_goals['safety'] * safety_objective
        )
        
        return total_objective
    
    def _define_physics_constraints(self, 
                                  charging_current: cp.Variable,
                                  thermal_control: cp.Variable,
                                  battery_state: Dict[str, Any],
                                  constraints: Dict[str, Any]) -> List[cp.Constraint]:
        """Define physics-based constraints."""
        
        physics_constraints = [
            # Current limits
            charging_current >= constraints['current_min'],
            charging_current <= constraints['current_max'],
            
            # Thermal control limits
            thermal_control >= -1.0,
            thermal_control <= 1.0,
            
            # Voltage constraints (simplified)
            battery_state['voltage'] + charging_current * 0.1 >= constraints['voltage_min'],
            battery_state['voltage'] + charging_current * 0.1 <= constraints['voltage_max'],
            
            # Temperature constraints (simplified)
            battery_state['temperature'] + thermal_control * 5.0 >= constraints['temperature_min'],
            battery_state['temperature'] + thermal_control * 5.0 <= constraints['temperature_max']
        ]
        
        return physics_constraints
    
    def _get_heuristic_recommendations(self, battery_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get heuristic recommendations as fallback."""
        soc = battery_state['soc']
        temperature = battery_state['temperature']
        
        # Simple heuristic rules
        if soc < 0.2:
            # Low SoC - charge with moderate current
            charging_current = 20.0
        elif soc > 0.8:
            # High SoC - reduce charging current
            charging_current = 5.0
        else:
            # Normal charging
            charging_current = 15.0
        
        # Thermal control based on temperature
        if temperature > 40.0:
            thermal_control = -0.5  # Cooling
        elif temperature < 15.0:
            thermal_control = 0.3   # Heating
        else:
            thermal_control = 0.0   # No thermal control
        
        return {
            'charging_current': charging_current,
            'thermal_control': thermal_control,
            'power_limit': 1.0,
            'optimization_method': 'heuristic'
        }
    
    def _combine_recommendations(self, recommendations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine recommendations from different methods."""
        combined = {}
        
        # Extract numerical recommendations
        methods = [key for key in recommendations.keys() if key not in ['ensemble', 'final']]
        
        for param in ['charging_current', 'thermal_control', 'power_limit']:
            values = []
            weights = []
            
            for method in methods:
                if param in recommendations[method]:
                    values.append(recommendations[method][param])
                    # Weight based on method reliability
                    if method == 'rl_agent':
                        weights.append(0.5)
                    elif method == 'transformer':
                        weights.append(0.3)
                    else:  # physics_based
                        weights.append(0.2)
            
            if values:
                # Weighted average
                combined[param] = np.average(values, weights=weights)
        
        combined['optimization_method'] = 'ensemble'
        
        return combined
    
    def _convert_to_env_state(self, battery_state: Dict[str, Any]) -> np.ndarray:
        """Convert battery state to environment state."""
        # This would convert the battery state dictionary to the format expected by the RL agent
        env_state = np.array([
            battery_state['voltage'],
            battery_state['current'],
            battery_state['temperature'],
            battery_state['soc'],
            battery_state.get('soh', 1.0),
            battery_state.get('internal_resistance', 0.1),
            battery_state.get('capacity', 100.0),
            battery_state.get('cycle_count', 0),
            battery_state.get('ambient_temperature', 25.0),
            battery_state.get('time_of_day', 12.0),
            battery_state.get('power_demand', 0.0),
            battery_state.get('grid_price', 0.15),
            battery_state.get('renewable_availability', 0.5),
            battery_state.get('safety_margin', 1.0),
            battery_state.get('efficiency_score', 0.9),
            battery_state.get('degradation_rate', 0.001)
        ])
        
        return env_state
    
    def _convert_action_to_recommendations(self, action: np.ndarray) -> Dict[str, Any]:
        """Convert RL action to recommendations."""
        return {
            'charging_current': float(action[0]),
            'thermal_control': float(action[1]),
            'power_limit': float(action[2]),
            'optimization_method': 'rl_agent'
        }
    
    def _prepare_transformer_input(self, battery_state: Dict[str, Any]) -> torch.Tensor:
        """Prepare input for transformer model."""
        # This would prepare the input format expected by the transformer
        input_features = [
            battery_state['voltage'],
            battery_state['current'],
            battery_state['temperature'],
            battery_state['soc']
        ]
        
        return torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def _process_transformer_output(self, output: torch.Tensor) -> Dict[str, Any]:
        """Process transformer output to recommendations."""
        output_np = output.cpu().numpy().flatten()
        
        return {
            'charging_current': float(output_np[0]),
            'thermal_control': float(output_np[1]),
            'power_limit': float(output_np[2]),
            'optimization_method': 'transformer'
        }
    
    def _validate_recommendations(self, recommendations: Dict[str, Any], battery_state: Dict[str, Any]):
        """Validate optimization recommendations."""
        final_rec = recommendations['final']
        
        # Check for NaN or infinite values
        for key, value in final_rec.items():
            if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                raise ValueError(f"Invalid recommendation value for {key}: {value}")
        
        # Check constraint violations
        if 'charging_current' in final_rec:
            current = final_rec['charging_current']
            if not (self.config.current_limits[0] <= current <= self.config.current_limits[1]):
                logger.warning(f"Charging current out of limits: {current}")
        
        if 'thermal_control' in final_rec:
            thermal = final_rec['thermal_control']
            if not (-1.0 <= thermal <= 1.0):
                logger.warning(f"Thermal control out of range: {thermal}")
    
    def _apply_safety_checks(self, recommendations: Dict[str, Any], battery_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply safety checks to recommendations."""
        final_rec = recommendations['final'].copy()
        
        # Temperature safety check
        if battery_state['temperature'] > 50.0:
            final_rec['charging_current'] = min(final_rec.get('charging_current', 0), 10.0)
            final_rec['thermal_control'] = -0.8  # Aggressive cooling
            self.monitoring_stats['safety_interventions'] += 1
        
        # SoC safety check
        if battery_state['soc'] < 0.05:
            final_rec['charging_current'] = max(final_rec.get('charging_current', 0), 5.0)
        elif battery_state['soc'] > 0.95:
            final_rec['charging_current'] = min(final_rec.get('charging_current', 0), 2.0)
        
        # Voltage safety check
        if battery_state['voltage'] < 2.8:
            final_rec['charging_current'] = max(final_rec.get('charging_current', 0), 1.0)
        elif battery_state['voltage'] > 4.1:
            final_rec['charging_current'] = min(final_rec.get('charging_current', 0), 1.0)
        
        recommendations['final'] = final_rec
        return recommendations
    
    def _generate_fail_safe_recommendations(self, battery_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fail-safe recommendations."""
        return {
            'recommendations': {
                'final': {
                    'charging_current': 0.0,  # Stop charging
                    'thermal_control': 0.0,   # No thermal control
                    'power_limit': 0.0,       # No power
                    'optimization_method': 'fail_safe'
                }
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'safety_mode': 'fail_safe',
                'error_recovery': True
            }
        }
    
    def _update_monitoring_stats(self, start_time: datetime, success: bool):
        """Update monitoring statistics."""
        self.monitoring_stats['total_optimizations'] += 1
        
        if success:
            self.monitoring_stats['successful_optimizations'] += 1
        else:
            self.monitoring_stats['failed_optimizations'] += 1
        
        # Update timing
        optimization_time = (datetime.now() - start_time).total_seconds()
        current_avg = self.monitoring_stats['average_optimization_time']
        total_opts = self.monitoring_stats['total_optimizations']
        
        self.monitoring_stats['average_optimization_time'] = (
            (current_avg * (total_opts - 1) + optimization_time) / total_opts
        )
    
    def _log_optimization_decision(self, battery_state: Dict[str, Any], recommendations: Dict[str, Any]):
        """Log optimization decision."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'battery_state': battery_state,
            'recommendations': recommendations['final'],
            'optimization_method': recommendations['final'].get('optimization_method', 'unknown')
        }
        
        self.optimization_history.append(log_entry)
        logger.info(f"Optimization decision logged: {log_entry}")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        return self.monitoring_stats.copy()
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization decision history."""
        return self.optimization_history.copy()
    
    def predict_charging_optimization(self, battery_state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal charging strategy."""
        optimization_goals = {
            'battery_health': 0.5,
            'energy_efficiency': 0.3,
            'charging_speed': 0.2
        }
        
        return self.predict_optimization(battery_state, optimization_goals)
    
    def predict_thermal_optimization(self, battery_state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal thermal management strategy."""
        optimization_goals = {
            'battery_health': 0.4,
            'safety': 0.4,
            'energy_efficiency': 0.2
        }
        
        return self.predict_optimization(battery_state, optimization_goals)
    
    def predict_multi_objective_optimization(self, 
                                           battery_state: Dict[str, Any],
                                           custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Predict multi-objective optimization strategy."""
        if custom_weights is None:
            custom_weights = {
                'battery_health': 0.3,
                'energy_efficiency': 0.25,
                'charging_speed': 0.2,
                'safety': 0.25
            }
        
        return self.predict_optimization(battery_state, custom_weights)

# Factory function
def create_optimization_predictor(config: Optional[OptimizationPredictionConfig] = None) -> OptimizationPredictor:
    """
    Factory function to create an optimization predictor.
    
    Args:
        config: Predictor configuration
        
    Returns:
        Configured OptimizationPredictor instance
    """
    if config is None:
        config = OptimizationPredictionConfig()
    
    return OptimizationPredictor(config)
