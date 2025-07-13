"""
BatteryMind - Safety Reward System

Comprehensive safety reward mechanisms for reinforcement learning agents
ensuring safe battery operation, preventing hazardous conditions, and
maintaining operational safety standards.

Features:
- Multi-layered safety constraint enforcement
- Thermal runaway prevention
- Overcharge/overdischarge protection
- Voltage and current safety limits
- Temperature safety monitoring
- Mechanical stress protection
- Emergency response protocols

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SafetyLimits:
    """
    Safety limits for battery operation.
    
    Attributes:
        min_voltage (float): Minimum safe voltage (V)
        max_voltage (float): Maximum safe voltage (V)
        min_current (float): Minimum safe current (A)
        max_current (float): Maximum safe current (A)
        min_temperature (float): Minimum safe temperature (°C)
        max_temperature (float): Maximum safe temperature (°C)
        min_soc (float): Minimum safe state of charge (0-1)
        max_soc (float): Maximum safe state of charge (0-1)
        max_pressure (float): Maximum safe pressure (Pa)
        max_stress (float): Maximum safe mechanical stress (Pa)
    """
    min_voltage: float = 2.5
    max_voltage: float = 4.2
    min_current: float = -100.0
    max_current: float = 100.0
    min_temperature: float = -20.0
    max_temperature: float = 60.0
    min_soc: float = 0.05
    max_soc: float = 0.95
    max_pressure: float = 101325.0  # 1 atm
    max_stress: float = 1e6  # 1 MPa

@dataclass
class SafetyMetrics:
    """
    Container for safety-related metrics.
    
    Attributes:
        voltage_safety (float): Voltage safety score (0-1)
        current_safety (float): Current safety score (0-1)
        temperature_safety (float): Temperature safety score (0-1)
        soc_safety (float): State of charge safety score (0-1)
        thermal_runaway_risk (float): Thermal runaway risk (0-1)
        overcharge_risk (float): Overcharge risk (0-1)
        overdischarge_risk (float): Overdischarge risk (0-1)
        mechanical_safety (float): Mechanical safety score (0-1)
        overall_safety (float): Overall safety score (0-1)
        safety_violations (int): Number of safety violations
    """
    voltage_safety: float = 1.0
    current_safety: float = 1.0
    temperature_safety: float = 1.0
    soc_safety: float = 1.0
    thermal_runaway_risk: float = 0.0
    overcharge_risk: float = 0.0
    overdischarge_risk: float = 0.0
    mechanical_safety: float = 1.0
    overall_safety: float = 1.0
    safety_violations: int = 0

@dataclass
class SafetyConfig:
    """
    Configuration for safety reward calculation.
    
    Attributes:
        voltage_weight (float): Weight for voltage safety component
        current_weight (float): Weight for current safety component
        temperature_weight (float): Weight for temperature safety component
        soc_weight (float): Weight for SOC safety component
        thermal_weight (float): Weight for thermal safety component
        mechanical_weight (float): Weight for mechanical safety component
        violation_penalty (float): Penalty for safety violations
        risk_threshold (float): Risk threshold for safety warnings
        emergency_penalty (float): Penalty for emergency conditions
        safety_bonus (float): Bonus for maintaining high safety
    """
    voltage_weight: float = 0.20
    current_weight: float = 0.20
    temperature_weight: float = 0.25
    soc_weight: float = 0.15
    thermal_weight: float = 0.10
    mechanical_weight: float = 0.10
    violation_penalty: float = 10.0
    risk_threshold: float = 0.8
    emergency_penalty: float = 100.0
    safety_bonus: float = 1.2

class BaseSafetyReward(ABC):
    """
    Abstract base class for safety reward mechanisms.
    """
    
    def __init__(self, safety_limits: SafetyLimits, config: SafetyConfig):
        self.safety_limits = safety_limits
        self.config = config
        self.violation_history = []
        
    @abstractmethod
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any],
                        next_state: Dict[str, Any]) -> float:
        """Calculate safety reward for state-action transition."""
        pass
    
    @abstractmethod
    def get_safety_metrics(self, state: Dict[str, Any]) -> SafetyMetrics:
        """Extract safety metrics from state."""
        pass
    
    def is_safe_state(self, state: Dict[str, Any]) -> bool:
        """Check if state is within safety limits."""
        metrics = self.get_safety_metrics(state)
        return metrics.overall_safety > self.config.risk_threshold

class VoltageSafetyReward(BaseSafetyReward):
    """
    Voltage safety reward focusing on voltage limit enforcement.
    """
    
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any],
                        next_state: Dict[str, Any]) -> float:
        """
        Calculate voltage safety reward.
        
        Args:
            state (Dict): Current state
            action (Dict): Action taken
            next_state (Dict): Resulting state
            
        Returns:
            float: Voltage safety reward
        """
        voltage = next_state.get('voltage', 3.7)
        
        # Calculate voltage safety score
        safety_score = self._calculate_voltage_safety(voltage)
        
        # Check for violations
        violation_penalty = self._calculate_voltage_violations(voltage)
        
        # Calculate rate of change penalty
        prev_voltage = state.get('voltage', 3.7)
        rate_penalty = self._calculate_voltage_rate_penalty(prev_voltage, voltage)
        
        # Combine components
        total_reward = safety_score - violation_penalty - rate_penalty
        
        return total_reward
    
    def _calculate_voltage_safety(self, voltage: float) -> float:
        """Calculate voltage safety score."""
        min_v, max_v = self.safety_limits.min_voltage, self.safety_limits.max_voltage
        
        if min_v <= voltage <= max_v:
            # Within safe range - calculate distance from limits
            center = (min_v + max_v) / 2
            range_half = (max_v - min_v) / 2
            distance_from_center = abs(voltage - center)
            safety_score = 1.0 - (distance_from_center / range_half) * 0.2
            return safety_score
        else:
            # Outside safe range - exponential penalty
            if voltage < min_v:
                violation = min_v - voltage
                return max(0.0, 1.0 - violation * 2.0)
            else:
                violation = voltage - max_v
                return max(0.0, 1.0 - violation * 2.0)
    
    def _calculate_voltage_violations(self, voltage: float) -> float:
        """Calculate penalty for voltage violations."""
        min_v, max_v = self.safety_limits.min_voltage, self.safety_limits.max_voltage
        
        if voltage < min_v:
            violation = min_v - voltage
            return self.config.violation_penalty * violation
        elif voltage > max_v:
            violation = voltage - max_v
            return self.config.violation_penalty * violation
        else:
            return 0.0
    
    def _calculate_voltage_rate_penalty(self, prev_voltage: float, 
                                      current_voltage: float) -> float:
        """Calculate penalty for rapid voltage changes."""
        rate_of_change = abs(current_voltage - prev_voltage)
        max_safe_rate = 0.1  # Maximum safe voltage change per step
        
        if rate_of_change > max_safe_rate:
            excess_rate = rate_of_change - max_safe_rate
            return excess_rate * 5.0  # Penalty factor
        else:
            return 0.0
    
    def get_safety_metrics(self, state: Dict[str, Any]) -> SafetyMetrics:
        """Extract voltage safety metrics from state."""
        voltage = state.get('voltage', 3.7)
        voltage_safety = self._calculate_voltage_safety(voltage)
        
        return SafetyMetrics(
            voltage_safety=voltage_safety,
            overall_safety=voltage_safety
        )

class TemperatureSafetyReward(BaseSafetyReward):
    """
    Temperature safety reward focusing on thermal management and safety.
    """
    
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any],
                        next_state: Dict[str, Any]) -> float:
        """
        Calculate temperature safety reward.
        
        Args:
            state (Dict): Current state
            action (Dict): Action taken
            next_state (Dict): Resulting state
            
        Returns:
            float: Temperature safety reward
        """
        temperature = next_state.get('temperature', 25.0)
        
        # Calculate temperature safety score
        safety_score = self._calculate_temperature_safety(temperature)
        
        # Calculate thermal runaway risk
        runaway_risk = self._calculate_thermal_runaway_risk(next_state)
        
        # Calculate temperature gradient safety
        gradient_safety = self._calculate_gradient_safety(next_state)
        
        # Check for violations
        violation_penalty = self._calculate_temperature_violations(temperature)
        
        # Combine components
        total_reward = (
            0.5 * safety_score +
            0.3 * (1.0 - runaway_risk) +
            0.2 * gradient_safety -
            violation_penalty
        )
        
        return total_reward
    
    def _calculate_temperature_safety(self, temperature: float) -> float:
        """Calculate temperature safety score."""
        min_t, max_t = self.safety_limits.min_temperature, self.safety_limits.max_temperature
        
        if min_t <= temperature <= max_t:
            # Within safe range - optimal around 25°C
            optimal_temp = 25.0
            deviation = abs(temperature - optimal_temp)
            max_deviation = max(optimal_temp - min_t, max_t - optimal_temp)
            safety_score = 1.0 - (deviation / max_deviation) * 0.3
            return safety_score
        else:
            # Outside safe range
            if temperature < min_t:
                violation = min_t - temperature
                return max(0.0, 1.0 - violation * 0.05)
            else:
                violation = temperature - max_t
                return max(0.0, 1.0 - violation * 0.1)  # Higher penalty for overheating
    
    def _calculate_thermal_runaway_risk(self, state: Dict[str, Any]) -> float:
        """Calculate thermal runaway risk."""
        temperature = state.get('temperature', 25.0)
        voltage = state.get('voltage', 3.7)
        current = state.get('current', 0.0)
        
        # Risk factors
        temp_risk = max(0.0, (temperature - 50.0) / 50.0)  # Risk increases above 50°C
        voltage_risk = max(0.0, (voltage - 4.0) / 0.5)     # Risk increases above 4.0V
        current_risk = max(0.0, (abs(current) - 50.0) / 50.0)  # Risk with high current
        
        # Combined risk with exponential scaling
        combined_risk = temp_risk + 0.5 * voltage_risk + 0.3 * current_risk
        thermal_runaway_risk = 1.0 - math.exp(-combined_risk)
        
        return min(1.0, thermal_runaway_risk)
    
    def _calculate_gradient_safety(self, state: Dict[str, Any]) -> float:
        """Calculate temperature gradient safety."""
        temp_gradient = state.get('temperature_gradient', 0.0)
        max_safe_gradient = 5.0  # °C/cm
        
        if temp_gradient <= max_safe_gradient:
            return 1.0 - (temp_gradient / max_safe_gradient) * 0.2
        else:
            excess_gradient = temp_gradient - max_safe_gradient
            return max(0.0, 1.0 - excess_gradient * 0.1)
    
    def _calculate_temperature_violations(self, temperature: float) -> float:
        """Calculate penalty for temperature violations."""
        min_t, max_t = self.safety_limits.min_temperature, self.safety_limits.max_temperature
        
        if temperature < min_t:
            violation = min_t - temperature
            return self.config.violation_penalty * violation * 0.1
        elif temperature > max_t:
            violation = temperature - max_t
            return self.config.violation_penalty * violation * 0.2  # Higher penalty for overheating
        else:
            return 0.0
    
    def get_safety_metrics(self, state: Dict[str, Any]) -> SafetyMetrics:
        """Extract temperature safety metrics from state."""
        temperature = state.get('temperature', 25.0)
        temperature_safety = self._calculate_temperature_safety(temperature)
        thermal_runaway_risk = self._calculate_thermal_runaway_risk(state)
        
        return SafetyMetrics(
            temperature_safety=temperature_safety,
            thermal_runaway_risk=thermal_runaway_risk,
            overall_safety=temperature_safety * (1.0 - thermal_runaway_risk)
        )

class ComprehensiveSafetyReward(BaseSafetyReward):
    """
    Comprehensive safety reward combining all safety aspects.
    """
    
    def __init__(self, safety_limits: SafetyLimits, config: SafetyConfig):
        super().__init__(safety_limits, config)
        
        # Initialize component safety rewards
        self.voltage_safety = VoltageSafetyReward(safety_limits, config)
        self.temperature_safety = TemperatureSafetyReward(safety_limits, config)
        
        # Safety violation tracking
        self.violation_count = 0
        self.emergency_count = 0
        
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any],
                        next_state: Dict[str, Any]) -> float:
        """
        Calculate comprehensive safety reward.
        
        Args:
            state (Dict): Current state
            action (Dict): Action taken
            next_state (Dict): Resulting state
            
        Returns:
            float: Comprehensive safety reward
        """
        # Calculate individual safety components
        voltage_reward = self.voltage_safety.calculate_reward(state, action, next_state)
        temperature_reward = self.temperature_safety.calculate_reward(state, action, next_state)
        
        # Calculate additional safety components
        current_safety = self._calculate_current_safety(next_state)
        soc_safety = self._calculate_soc_safety(next_state)
        mechanical_safety = self._calculate_mechanical_safety(next_state)
        
        # Check for emergency conditions
        emergency_penalty = self._calculate_emergency_penalty(next_state)
        
        # Calculate overcharge/overdischarge risks
        overcharge_risk = self._calculate_overcharge_risk(next_state)
        overdischarge_risk = self._calculate_overdischarge_risk(next_state)
        
        # Weighted combination
        safety_score = (
            self.config.voltage_weight * voltage_reward +
            self.config.temperature_weight * temperature_reward +
            self.config.current_weight * current_safety +
            self.config.soc_weight * soc_safety +
            self.config.mechanical_weight * mechanical_safety
        )
        
        # Apply risk penalties
        risk_penalty = (overcharge_risk + overdischarge_risk) * self.config.violation_penalty
        
        # Apply emergency penalty
        total_reward = safety_score - risk_penalty - emergency_penalty
        
        # Apply safety bonus for consistently safe operation
        if safety_score > self.config.risk_threshold:
            total_reward *= self.config.safety_bonus
        
        # Track violations
        self._update_violation_tracking(next_state)
        
        return total_reward
    
    def _calculate_current_safety(self, state: Dict[str, Any]) -> float:
        """Calculate current safety score."""
        current = state.get('current', 0.0)
        min_i, max_i = self.safety_limits.min_current, self.safety_limits.max_current
        
        if min_i <= current <= max_i:
            # Within safe range
            center = (min_i + max_i) / 2
            range_half = (max_i - min_i) / 2
            distance_from_center = abs(current - center)
            safety_score = 1.0 - (distance_from_center / range_half) * 0.1
            return safety_score
        else:
            # Outside safe range
            if current < min_i:
                violation = min_i - current
                return max(0.0, 1.0 - violation * 0.01)
            else:
                violation = current - max_i
                return max(0.0, 1.0 - violation * 0.01)
    
    def _calculate_soc_safety(self, state: Dict[str, Any]) -> float:
        """Calculate state of charge safety score."""
        soc = state.get('soc', 0.5)
        min_soc, max_soc = self.safety_limits.min_soc, self.safety_limits.max_soc
        
        if min_soc <= soc <= max_soc:
            # Within safe range
            return 1.0
        else:
            # Outside safe range
            if soc < min_soc:
                violation = min_soc - soc
                return max(0.0, 1.0 - violation * 10.0)
            else:
                violation = soc - max_soc
                return max(0.0, 1.0 - violation * 10.0)
    
    def _calculate_mechanical_safety(self, state: Dict[str, Any]) -> float:
        """Calculate mechanical safety score."""
        pressure = state.get('pressure', 101325.0)
        stress = state.get('mechanical_stress', 0.0)
        
        # Pressure safety
        pressure_safety = 1.0 if pressure <= self.safety_limits.max_pressure else \
                         max(0.0, 1.0 - (pressure - self.safety_limits.max_pressure) / self.safety_limits.max_pressure)
        
        # Stress safety
        stress_safety = 1.0 if stress <= self.safety_limits.max_stress else \
                       max(0.0, 1.0 - (stress - self.safety_limits.max_stress) / self.safety_limits.max_stress)
        
        return 0.5 * pressure_safety + 0.5 * stress_safety
    
    def _calculate_overcharge_risk(self, state: Dict[str, Any]) -> float:
        """Calculate overcharge risk."""
        soc = state.get('soc', 0.5)
        voltage = state.get('voltage', 3.7)
        charging_current = state.get('charging_current', 0.0)
        
        # Risk factors
        soc_risk = max(0.0, (soc - 0.9) / 0.1)
        voltage_risk = max(0.0, (voltage - 4.1) / 0.2)
        current_risk = max(0.0, charging_current / 50.0) if charging_current > 0 else 0.0
        
        # Combined risk
        overcharge_risk = min(1.0, soc_risk + voltage_risk + current_risk)
        return overcharge_risk
    
    def _calculate_overdischarge_risk(self, state: Dict[str, Any]) -> float:
        """Calculate overdischarge risk."""
        soc = state.get('soc', 0.5)
        voltage = state.get('voltage', 3.7)
        discharge_current = state.get('discharge_current', 0.0)
        
        # Risk factors
        soc_risk = max(0.0, (0.1 - soc) / 0.1)
        voltage_risk = max(0.0, (2.8 - voltage) / 0.3)
        current_risk = max(0.0, abs(discharge_current) / 50.0) if discharge_current < 0 else 0.0
        
        # Combined risk
        overdischarge_risk = min(1.0, soc_risk + voltage_risk + current_risk)
        return overdischarge_risk
    
    def _calculate_emergency_penalty(self, state: Dict[str, Any]) -> float:
        """Calculate penalty for emergency conditions."""
        temperature = state.get('temperature', 25.0)
        voltage = state.get('voltage', 3.7)
        current = state.get('current', 0.0)
        
        emergency_conditions = 0
        
        # Critical temperature
        if temperature > 70.0 or temperature < -30.0:
            emergency_conditions += 1
        
        # Critical voltage
        if voltage > 4.5 or voltage < 2.0:
            emergency_conditions += 1
        
        # Critical current
        if abs(current) > 150.0:
            emergency_conditions += 1
        
        if emergency_conditions > 0:
            self.emergency_count += 1
            return emergency_conditions * self.config.emergency_penalty
        else:
            return 0.0
    
    def _update_violation_tracking(self, state: Dict[str, Any]) -> None:
        """Update safety violation tracking."""
        metrics = self.get_safety_metrics(state)
        
        if metrics.safety_violations > 0:
            self.violation_count += metrics.safety_violations
            self.violation_history.append({
                'timestamp': len(self.violation_history),
                'violations': metrics.safety_violations,
                'overall_safety': metrics.overall_safety
            })
    
    def get_safety_metrics(self, state: Dict[str, Any]) -> SafetyMetrics:
        """Extract comprehensive safety metrics from state."""
        # Get individual safety metrics
        voltage_metrics = self.voltage_safety.get_safety_metrics(state)
        temperature_metrics = self.temperature_safety.get_safety_metrics(state)
        
        # Calculate additional metrics
        current_safety = self._calculate_current_safety(state)
        soc_safety = self._calculate_soc_safety(state)
        mechanical_safety = self._calculate_mechanical_safety(state)
        overcharge_risk = self._calculate_overcharge_risk(state)
        overdischarge_risk = self._calculate_overdischarge_risk(state)
        
        # Count violations
        violations = 0
        if voltage_metrics.voltage_safety < 0.8:
            violations += 1
        if temperature_metrics.temperature_safety < 0.8:
            violations += 1
        if current_safety < 0.8:
            violations += 1
        if soc_safety < 0.8:
            violations += 1
        
        # Calculate overall safety
        overall_safety = (
            self.config.voltage_weight * voltage_metrics.voltage_safety +
            self.config.temperature_weight * temperature_metrics.temperature_safety +
            self.config.current_weight * current_safety +
            self.config.soc_weight * soc_safety +
            self.config.mechanical_weight * mechanical_safety
        )
        
        return SafetyMetrics(
            voltage_safety=voltage_metrics.voltage_safety,
            current_safety=current_safety,
            temperature_safety=temperature_metrics.temperature_safety,
            soc_safety=soc_safety,
            thermal_runaway_risk=temperature_metrics.thermal_runaway_risk,
            overcharge_risk=overcharge_risk,
            overdischarge_risk=overdischarge_risk,
            mechanical_safety=mechanical_safety,
            overall_safety=overall_safety,
            safety_violations=violations
        )
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety report."""
        return {
            "total_violations": self.violation_count,
            "emergency_incidents": self.emergency_count,
            "violation_history": self.violation_history[-10:],  # Last 10 violations
            "safety_limits": {
                "voltage_range": [self.safety_limits.min_voltage, self.safety_limits.max_voltage],
                "current_range": [self.safety_limits.min_current, self.safety_limits.max_current],
                "temperature_range": [self.safety_limits.min_temperature, self.safety_limits.max_temperature],
                "soc_range": [self.safety_limits.min_soc, self.safety_limits.max_soc]
            }
        }

# Factory function
def create_safety_reward(reward_type: str = "comprehensive",
                        safety_limits: Optional[SafetyLimits] = None,
                        config: Optional[SafetyConfig] = None) -> BaseSafetyReward:
    """
    Factory function to create safety reward instances.
    
    Args:
        reward_type (str): Type of safety reward
        safety_limits (SafetyLimits, optional): Safety limits configuration
        config (SafetyConfig, optional): Safety reward configuration
        
    Returns:
        BaseSafetyReward: Configured safety reward instance
    """
    if safety_limits is None:
        safety_limits = SafetyLimits()
    
    if config is None:
        config = SafetyConfig()
    
    if reward_type == "voltage":
        return VoltageSafetyReward(safety_limits, config)
    elif reward_type == "temperature":
        return TemperatureSafetyReward(safety_limits, config)
    elif reward_type == "comprehensive":
        return ComprehensiveSafetyReward(safety_limits, config)
    else:
        raise ValueError(f"Unknown safety reward type: {reward_type}")
