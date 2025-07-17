"""
BatteryMind - Safety Validator

Critical safety validation system for battery AI models, ensuring all AI decisions
comply with safety constraints and regulations for battery management systems.
Provides comprehensive safety monitoring, constraint validation, and emergency
response capabilities.

Features:
- Real-time safety constraint validation
- Physics-based safety boundary enforcement
- Thermal and electrical safety monitoring
- Emergency response and fail-safe mechanisms
- Regulatory compliance validation
- Safety incident tracking and reporting
- Automated safety alert generation

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json
from enum import Enum
import math

# BatteryMind imports
from . import BaseValidator
from ...utils.logging_utils import setup_logger
from ...utils.data_utils import PhysicsValidator
from ...training_data.generators.physics_simulator import BatteryPhysicsSimulator

# Configure logging
logger = setup_logger(__name__)

class SafetyLevel(Enum):
    """Safety levels for different types of violations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class SafetyViolationType(Enum):
    """Types of safety violations."""
    THERMAL_RUNAWAY = "thermal_runaway"
    OVERVOLTAGE = "overvoltage"
    UNDERVOLTAGE = "undervoltage"
    OVERCURRENT = "overcurrent"
    OVERTEMPERATURE = "overtemperature"
    UNDERTEMPERATURE = "undertemperature"
    CAPACITY_ANOMALY = "capacity_anomaly"
    RESISTANCE_ANOMALY = "resistance_anomaly"
    PRESSURE_ANOMALY = "pressure_anomaly"
    CHEMICAL_HAZARD = "chemical_hazard"
    MECHANICAL_STRESS = "mechanical_stress"
    ELECTRICAL_FAULT = "electrical_fault"
    REGULATORY_VIOLATION = "regulatory_violation"

@dataclass
class SafetyConstraints:
    """
    Comprehensive safety constraints for battery systems.
    
    Attributes:
        # Electrical constraints
        voltage_min_v (float): Minimum safe voltage (V)
        voltage_max_v (float): Maximum safe voltage (V)
        current_min_a (float): Minimum safe current (A)
        current_max_a (float): Maximum safe current (A)
        power_max_w (float): Maximum safe power (W)
        
        # Thermal constraints
        temperature_min_c (float): Minimum safe temperature (°C)
        temperature_max_c (float): Maximum safe temperature (°C)
        temperature_gradient_max_c_per_min (float): Maximum temperature gradient (°C/min)
        
        # Chemical constraints
        soc_min (float): Minimum safe state of charge (0-1)
        soc_max (float): Maximum safe state of charge (0-1)
        soh_min (float): Minimum safe state of health (0-1)
        
        # Mechanical constraints
        pressure_max_pa (float): Maximum safe pressure (Pa)
        vibration_max_g (float): Maximum safe vibration (g)
        
        # Time-based constraints
        charge_time_max_hours (float): Maximum charging time (hours)
        discharge_time_min_hours (float): Minimum discharge time (hours)
        
        # Environmental constraints
        humidity_max_percent (float): Maximum safe humidity (%)
        altitude_max_m (float): Maximum safe altitude (m)
        
        # Regulatory constraints
        safety_margin_factor (float): Safety margin factor
        emergency_shutdown_threshold (float): Emergency shutdown threshold
        
        # Battery-specific constraints
        battery_specific_constraints (Dict[str, Dict[str, float]]): Battery-specific constraints
    """
    # Electrical constraints
    voltage_min_v: float = 2.5
    voltage_max_v: float = 4.2
    current_min_a: float = -100.0
    current_max_a: float = 100.0
    power_max_w: float = 1000.0
    
    # Thermal constraints
    temperature_min_c: float = -20.0
    temperature_max_c: float = 60.0
    temperature_gradient_max_c_per_min: float = 5.0
    
    # Chemical constraints
    soc_min: float = 0.05
    soc_max: float = 0.95
    soh_min: float = 0.70
    
    # Mechanical constraints
    pressure_max_pa: float = 101325.0  # 1 atm
    vibration_max_g: float = 10.0
    
    # Time-based constraints
    charge_time_max_hours: float = 24.0
    discharge_time_min_hours: float = 0.5
    
    # Environmental constraints
    humidity_max_percent: float = 95.0
    altitude_max_m: float = 3000.0
    
    # Regulatory constraints
    safety_margin_factor: float = 0.1
    emergency_shutdown_threshold: float = 0.05
    
    # Battery-specific constraints
    battery_specific_constraints: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'lithium_ion': {
            'voltage_min_v': 2.8,
            'voltage_max_v': 4.2,
            'temperature_max_c': 45.0,
            'soc_max': 0.90
        },
        'lifepo4': {
            'voltage_min_v': 2.0,
            'voltage_max_v': 3.6,
            'temperature_max_c': 60.0,
            'soc_max': 0.95
        },
        'nickel_metal_hydride': {
            'voltage_min_v': 0.9,
            'voltage_max_v': 1.4,
            'temperature_max_c': 50.0,
            'soc_max': 0.90
        }
    })

@dataclass
class SafetyValidationConfig:
    """
    Configuration for safety validation system.
    
    Attributes:
        # Validation settings
        continuous_monitoring (bool): Enable continuous safety monitoring
        validation_frequency_seconds (int): Validation frequency in seconds
        emergency_response_enabled (bool): Enable emergency response
        
        # Alert settings
        enable_immediate_alerts (bool): Enable immediate safety alerts
        alert_retention_days (int): Alert retention period in days
        emergency_contact_list (List[str]): Emergency contact list
        
        # Physics validation
        enable_physics_validation (bool): Enable physics-based validation
        physics_tolerance (float): Physics validation tolerance
        
        # Regulatory compliance
        regulatory_standards (List[str]): Applicable regulatory standards
        compliance_reporting_enabled (bool): Enable compliance reporting
        
        # Fail-safe settings
        fail_safe_mode_enabled (bool): Enable fail-safe mode
        automatic_shutdown_enabled (bool): Enable automatic shutdown
        backup_system_enabled (bool): Enable backup system
        
        # Logging and reporting
        detailed_logging_enabled (bool): Enable detailed safety logging
        incident_reporting_enabled (bool): Enable incident reporting
        audit_trail_enabled (bool): Enable audit trail
    """
    # Validation settings
    continuous_monitoring: bool = True
    validation_frequency_seconds: int = 1
    emergency_response_enabled: bool = True
    
    # Alert settings
    enable_immediate_alerts: bool = True
    alert_retention_days: int = 90
    emergency_contact_list: List[str] = field(default_factory=list)
    
    # Physics validation
    enable_physics_validation: bool = True
    physics_tolerance: float = 0.05
    
    # Regulatory compliance
    regulatory_standards: List[str] = field(default_factory=lambda: [
        'UL2054', 'IEC62133', 'UN38.3', 'SAE J2464'
    ])
    compliance_reporting_enabled: bool = True
    
    # Fail-safe settings
    fail_safe_mode_enabled: bool = True
    automatic_shutdown_enabled: bool = True
    backup_system_enabled: bool = True
    
    # Logging and reporting
    detailed_logging_enabled: bool = True
    incident_reporting_enabled: bool = True
    audit_trail_enabled: bool = True

class SafetyValidator(BaseValidator):
    """
    Critical safety validator for battery AI systems.
    """
    
    def __init__(self, 
                 safety_constraints: SafetyConstraints,
                 config: SafetyValidationConfig):
        super().__init__()
        self.safety_constraints = safety_constraints
        self.config = config
        self.safety_incidents = []
        self.emergency_states = []
        self.compliance_history = []
        
        # Initialize physics validator
        self.physics_validator = PhysicsValidator()
        self.physics_simulator = BatteryPhysicsSimulator()
        
        # Safety monitoring state
        self.monitoring_active = False
        self.last_validation_time = None
        self.emergency_mode = False
        
        logger.info("SafetyValidator initialized with critical safety monitoring")
    
    def validate_battery_safety(self, 
                               battery_data: Dict[str, Any],
                               model_predictions: Dict[str, Any],
                               battery_type: str = "lithium_ion",
                               battery_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate battery safety based on current state and AI predictions.
        
        Args:
            battery_data: Current battery state data
            model_predictions: AI model predictions
            battery_type: Type of battery chemistry
            battery_id: Unique battery identifier
            
        Returns:
            Dictionary containing safety validation results
        """
        validation_start_time = datetime.now()
        
        try:
            # Get applicable safety constraints
            applicable_constraints = self._get_applicable_constraints(battery_type)
            
            # Validate current battery state
            state_validation = self._validate_battery_state(battery_data, applicable_constraints)
            
            # Validate AI predictions
            prediction_validation = self._validate_ai_predictions(
                model_predictions, applicable_constraints
            )
            
            # Perform physics-based validation
            physics_validation = self._validate_physics_constraints(
                battery_data, model_predictions, applicable_constraints
            )
            
            # Check for emergency conditions
            emergency_check = self._check_emergency_conditions(
                battery_data, model_predictions, applicable_constraints
            )
            
            # Regulatory compliance check
            compliance_check = self._check_regulatory_compliance(
                battery_data, model_predictions, battery_type
            )
            
            # Generate safety recommendations
            recommendations = self._generate_safety_recommendations(
                state_validation, prediction_validation, physics_validation, emergency_check
            )
            
            # Determine overall safety status
            overall_status = self._determine_safety_status(
                state_validation, prediction_validation, physics_validation, emergency_check
            )
            
            # Handle emergency situations
            if emergency_check['emergency_detected']:
                self._handle_emergency_situation(emergency_check, battery_data, battery_id)
            
            # Record safety validation
            validation_results = {
                'validation_timestamp': validation_start_time.isoformat(),
                'battery_id': battery_id,
                'battery_type': battery_type,
                'overall_status': overall_status,
                'state_validation': state_validation,
                'prediction_validation': prediction_validation,
                'physics_validation': physics_validation,
                'emergency_check': emergency_check,
                'compliance_check': compliance_check,
                'recommendations': recommendations,
                'safety_score': self._calculate_safety_score(
                    state_validation, prediction_validation, physics_validation
                )
            }
            
            # Log validation results
            self._log_safety_validation(validation_results)
            
            # Update safety history
            self._update_safety_history(validation_results)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Safety validation failed: {e}")
            
            # Return emergency status on validation failure
            return {
                'validation_timestamp': validation_start_time.isoformat(),
                'battery_id': battery_id,
                'battery_type': battery_type,
                'overall_status': 'EMERGENCY',
                'error': str(e),
                'emergency_actions': ['immediate_shutdown', 'manual_inspection']
            }
    
    def _get_applicable_constraints(self, battery_type: str) -> Dict[str, float]:
        """Get applicable safety constraints for battery type."""
        # Start with general constraints
        constraints = {
            'voltage_min_v': self.safety_constraints.voltage_min_v,
            'voltage_max_v': self.safety_constraints.voltage_max_v,
            'current_min_a': self.safety_constraints.current_min_a,
            'current_max_a': self.safety_constraints.current_max_a,
            'power_max_w': self.safety_constraints.power_max_w,
            'temperature_min_c': self.safety_constraints.temperature_min_c,
            'temperature_max_c': self.safety_constraints.temperature_max_c,
            'temperature_gradient_max_c_per_min': self.safety_constraints.temperature_gradient_max_c_per_min,
            'soc_min': self.safety_constraints.soc_min,
            'soc_max': self.safety_constraints.soc_max,
            'soh_min': self.safety_constraints.soh_min,
            'pressure_max_pa': self.safety_constraints.pressure_max_pa,
            'vibration_max_g': self.safety_constraints.vibration_max_g
        }
        
        # Override with battery-specific constraints
        if battery_type in self.safety_constraints.battery_specific_constraints:
            constraints.update(self.safety_constraints.battery_specific_constraints[battery_type])
        
        return constraints
    
    def _validate_battery_state(self, 
                               battery_data: Dict[str, Any],
                               constraints: Dict[str, float]) -> Dict[str, Any]:
        """Validate current battery state against safety constraints."""
        violations = []
        warnings = []
        
        # Voltage validation
        voltage = battery_data.get('voltage', 0)
        if voltage < constraints['voltage_min_v']:
            violations.append({
                'type': SafetyViolationType.UNDERVOLTAGE.value,
                'severity': SafetyLevel.HIGH.value,
                'current_value': voltage,
                'constraint': constraints['voltage_min_v'],
                'message': f"Voltage {voltage}V below minimum safe level {constraints['voltage_min_v']}V"
            })
        elif voltage > constraints['voltage_max_v']:
            violations.append({
                'type': SafetyViolationType.OVERVOLTAGE.value,
                'severity': SafetyLevel.CRITICAL.value,
                'current_value': voltage,
                'constraint': constraints['voltage_max_v'],
                'message': f"Voltage {voltage}V exceeds maximum safe level {constraints['voltage_max_v']}V"
            })
        
        # Current validation
        current = battery_data.get('current', 0)
        if abs(current) > constraints['current_max_a']:
            violations.append({
                'type': SafetyViolationType.OVERCURRENT.value,
                'severity': SafetyLevel.HIGH.value,
                'current_value': current,
                'constraint': constraints['current_max_a'],
                'message': f"Current {current}A exceeds maximum safe level {constraints['current_max_a']}A"
            })
        
        # Temperature validation
        temperature = battery_data.get('temperature', 25)
        if temperature < constraints['temperature_min_c']:
            violations.append({
                'type': SafetyViolationType.UNDERTEMPERATURE.value,
                'severity': SafetyLevel.MEDIUM.value,
                'current_value': temperature,
                'constraint': constraints['temperature_min_c'],
                'message': f"Temperature {temperature}°C below minimum safe level {constraints['temperature_min_c']}°C"
            })
        elif temperature > constraints['temperature_max_c']:
            violations.append({
                'type': SafetyViolationType.OVERTEMPERATURE.value,
                'severity': SafetyLevel.CRITICAL.value,
                'current_value': temperature,
                'constraint': constraints['temperature_max_c'],
                'message': f"Temperature {temperature}°C exceeds maximum safe level {constraints['temperature_max_c']}°C"
            })
        
        # SoC validation
        soc = battery_data.get('soc', 0.5)
        if soc < constraints['soc_min']:
            warnings.append({
                'type': 'low_soc',
                'severity': SafetyLevel.MEDIUM.value,
                'current_value': soc,
                'constraint': constraints['soc_min'],
                'message': f"SoC {soc:.1%} below recommended minimum {constraints['soc_min']:.1%}"
            })
        elif soc > constraints['soc_max']:
            violations.append({
                'type': 'high_soc',
                'severity': SafetyLevel.HIGH.value,
                'current_value': soc,
                'constraint': constraints['soc_max'],
                'message': f"SoC {soc:.1%} exceeds maximum safe level {constraints['soc_max']:.1%}"
            })
        
        # SoH validation
        soh = battery_data.get('soh', 1.0)
        if soh < constraints['soh_min']:
            warnings.append({
                'type': 'low_soh',
                'severity': SafetyLevel.HIGH.value,
                'current_value': soh,
                'constraint': constraints['soh_min'],
                'message': f"SoH {soh:.1%} below safe operating level {constraints['soh_min']:.1%}"
            })
        
        # Power validation
        power = voltage * current
        if abs(power) > constraints['power_max_w']:
            violations.append({
                'type': 'overpower',
                'severity': SafetyLevel.HIGH.value,
                'current_value': power,
                'constraint': constraints['power_max_w'],
                'message': f"Power {power}W exceeds maximum safe level {constraints['power_max_w']}W"
            })
        
        return {
            'violations': violations,
            'warnings': warnings,
            'validation_passed': len(violations) == 0,
            'critical_violations': len([v for v in violations if v['severity'] == SafetyLevel.CRITICAL.value])
        }
    
    def _validate_ai_predictions(self, 
                                model_predictions: Dict[str, Any],
                                constraints: Dict[str, float]) -> Dict[str, Any]:
        """Validate AI model predictions for safety compliance."""
        violations = []
        warnings = []
        
        # Validate charging current predictions
        if 'charging_current' in model_predictions:
            predicted_current = model_predictions['charging_current']
            if abs(predicted_current) > constraints['current_max_a']:
                violations.append({
                    'type': SafetyViolationType.OVERCURRENT.value,
                    'severity': SafetyLevel.HIGH.value,
                    'predicted_value': predicted_current,
                    'constraint': constraints['current_max_a'],
                    'message': f"AI predicted charging current {predicted_current}A exceeds safety limit"
                })
        
        # Validate voltage predictions
        if 'target_voltage' in model_predictions:
            predicted_voltage = model_predictions['target_voltage']
            if predicted_voltage > constraints['voltage_max_v']:
                violations.append({
                    'type': SafetyViolationType.OVERVOLTAGE.value,
                    'severity': SafetyLevel.CRITICAL.value,
                    'predicted_value': predicted_voltage,
                    'constraint': constraints['voltage_max_v'],
                    'message': f"AI predicted voltage {predicted_voltage}V exceeds safety limit"
                })
        
        # Validate thermal control predictions
        if 'thermal_control' in model_predictions:
            thermal_action = model_predictions['thermal_control']
            # Check if thermal control would lead to unsafe temperatures
            if abs(thermal_action) > 1.0:  # Normalized thermal control should be [-1, 1]
                warnings.append({
                    'type': 'thermal_control_extreme',
                    'severity': SafetyLevel.MEDIUM.value,
                    'predicted_value': thermal_action,
                    'message': f"AI predicted extreme thermal control action {thermal_action}"
                })
        
        # Validate power predictions
        if 'power_limit' in model_predictions:
            predicted_power = model_predictions['power_limit']
            if predicted_power > constraints['power_max_w']:
                violations.append({
                    'type': 'overpower',
                    'severity': SafetyLevel.HIGH.value,
                    'predicted_value': predicted_power,
                    'constraint': constraints['power_max_w'],
                    'message': f"AI predicted power {predicted_power}W exceeds safety limit"
                })
        
        return {
            'violations': violations,
            'warnings': warnings,
            'validation_passed': len(violations) == 0,
            'ai_safety_compliant': len(violations) == 0
        }
    
    def _validate_physics_constraints(self, 
                                    battery_data: Dict[str, Any],
                                    model_predictions: Dict[str, Any],
                                    constraints: Dict[str, float]) -> Dict[str, Any]:
        """Validate physics-based constraints."""
        if not self.config.enable_physics_validation:
            return {'validation_enabled': False}
        
        violations = []
        
        # Validate Ohm's law compliance
        voltage = battery_data.get('voltage', 0)
        current = battery_data.get('current', 0)
        internal_resistance = battery_data.get('internal_resistance', 0.1)
        
        if voltage > 0 and current != 0:
            expected_voltage = current * internal_resistance
            voltage_error = abs(voltage - expected_voltage) / voltage
            
            if voltage_error > self.config.physics_tolerance:
                violations.append({
                    'type': SafetyViolationType.ELECTRICAL_FAULT.value,
                    'severity': SafetyLevel.MEDIUM.value,
                    'physics_law': 'ohms_law',
                    'expected_value': expected_voltage,
                    'actual_value': voltage,
                    'error_percentage': voltage_error * 100,
                    'message': f"Voltage measurement violates Ohm's law (error: {voltage_error:.1%})"
                })
        
        # Validate energy conservation
        power_in = model_predictions.get('charging_power', 0)
        power_out = model_predictions.get('discharging_power', 0)
        
        if power_in > 0 and power_out > 0:
            efficiency = power_out / power_in
            if efficiency > 1.0:
                violations.append({
                    'type': SafetyViolationType.ELECTRICAL_FAULT.value,
                    'severity': SafetyLevel.HIGH.value,
                    'physics_law': 'energy_conservation',
                    'efficiency': efficiency,
                    'message': f"Energy efficiency {efficiency:.1%} violates conservation law"
                })
        
        # Validate thermal physics
        temperature = battery_data.get('temperature', 25)
        ambient_temperature = battery_data.get('ambient_temperature', 25)
        
        if abs(temperature - ambient_temperature) > 50:  # Unrealistic temperature difference
            violations.append({
                'type': SafetyViolationType.THERMAL_RUNAWAY.value,
                'severity': SafetyLevel.CRITICAL.value,
                'physics_law': 'thermal_physics',
                'temperature_difference': temperature - ambient_temperature,
                'message': f"Temperature difference {temperature - ambient_temperature}°C indicates thermal issue"
            })
        
        return {
            'violations': violations,
            'validation_passed': len(violations) == 0,
            'physics_compliance': len(violations) == 0
        }
    
    def _check_emergency_conditions(self, 
                                  battery_data: Dict[str, Any],
                                  model_predictions: Dict[str, Any],
                                  constraints: Dict[str, float]) -> Dict[str, Any]:
        """Check for emergency conditions requiring immediate action."""
        emergency_conditions = []
        
        # Thermal runaway detection
        temperature = battery_data.get('temperature', 25)
        if temperature > constraints['temperature_max_c'] + 10:  # 10°C above max
            emergency_conditions.append({
                'type': SafetyViolationType.THERMAL_RUNAWAY.value,
                'severity': SafetyLevel.CRITICAL.value,
                'condition': 'thermal_runaway_risk',
                'value': temperature,
                'threshold': constraints['temperature_max_c'] + 10,
                'immediate_actions': ['emergency_shutdown', 'cooling_activation', 'fire_suppression_standby']
            })
        
        # Rapid voltage drop
        voltage = battery_data.get('voltage', 0)
        if voltage < constraints['voltage_min_v'] - 0.5:  # 0.5V below minimum
            emergency_conditions.append({
                'type': SafetyViolationType.ELECTRICAL_FAULT.value,
                'severity': SafetyLevel.CRITICAL.value,
                'condition': 'rapid_voltage_drop',
                'value': voltage,
                'threshold': constraints['voltage_min_v'] - 0.5,
                'immediate_actions': ['disconnect_load', 'stop_charging', 'diagnostic_check']
            })
        
        # Overcurrent emergency
        current = battery_data.get('current', 0)
        if abs(current) > constraints['current_max_a'] * 2:  # 2x max current
            emergency_conditions.append({
                'type': SafetyViolationType.OVERCURRENT.value,
                'severity': SafetyLevel.CRITICAL.value,
                'condition': 'severe_overcurrent',
                'value': current,
                'threshold': constraints['current_max_a'] * 2,
                'immediate_actions': ['emergency_disconnect', 'fuse_check', 'circuit_protection']
            })
        
        # Battery swelling detection
        pressure = battery_data.get('pressure', 101325)
        if pressure > constraints['pressure_max_pa'] * 1.5:
            emergency_conditions.append({
                'type': SafetyViolationType.PRESSURE_ANOMALY.value,
                'severity': SafetyLevel.HIGH.value,
                'condition': 'battery_swelling',
                'value': pressure,
                'threshold': constraints['pressure_max_pa'] * 1.5,
                'immediate_actions': ['pressure_relief', 'physical_inspection', 'isolation']
            })
        
        return {
            'emergency_detected': len(emergency_conditions) > 0,
            'emergency_conditions': emergency_conditions,
            'critical_count': len([c for c in emergency_conditions if c['severity'] == SafetyLevel.CRITICAL.value]),
            'recommended_actions': list(set([
                action for condition in emergency_conditions 
                for action in condition['immediate_actions']
                     ]))
        }
    
    def _check_regulatory_compliance(self, 
                                   battery_data: Dict[str, Any],
                                   model_predictions: Dict[str, Any],
                                   battery_type: str) -> Dict[str, Any]:
        """Check compliance with regulatory standards."""
        if not self.config.compliance_reporting_enabled:
            return {'compliance_checking_enabled': False}
        
        compliance_results = {}
        
        for standard in self.config.regulatory_standards:
            if standard == 'UL2054':
                compliance_results[standard] = self._check_ul2054_compliance(battery_data, model_predictions)
            elif standard == 'IEC62133':
                compliance_results[standard] = self._check_iec62133_compliance(battery_data, model_predictions)
            elif standard == 'UN38.3':
                compliance_results[standard] = self._check_un383_compliance(battery_data, model_predictions)
            elif standard == 'SAE J2464':
                compliance_results[standard] = self._check_sae_j2464_compliance(battery_data, model_predictions)
        
        overall_compliance = all(result['compliant'] for result in compliance_results.values())
        
        return {
            'overall_compliance': overall_compliance,
            'standards_checked': list(compliance_results.keys()),
            'compliance_details': compliance_results,
            'non_compliant_standards': [
                standard for standard, result in compliance_results.items() 
                if not result['compliant']
            ]
        }
    
    def _check_ul2054_compliance(self, battery_data: Dict[str, Any], model_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Check UL2054 compliance for household and commercial batteries."""
        compliance_issues = []
        
        # UL2054 specific checks
        temperature = battery_data.get('temperature', 25.0)
        voltage = battery_data.get('voltage', 3.7)
        current = battery_data.get('current', 0.0)
        
        # Temperature limits (UL2054 Section 19.2)
        if temperature > 60.0:
            compliance_issues.append({
                'issue': 'excessive_temperature',
                'description': 'Temperature exceeds UL2054 limit of 60°C',
                'current_value': temperature,
                'standard_limit': 60.0,
                'severity': 'high'
            })
        
        # Voltage limits (UL2054 Section 18.1)
        if voltage > 4.5:
            compliance_issues.append({
                'issue': 'excessive_voltage',
                'description': 'Voltage exceeds UL2054 safety limit',
                'current_value': voltage,
                'standard_limit': 4.5,
                'severity': 'high'
            })
        
        # Current density limits
        capacity = battery_data.get('capacity', 100.0)
        current_density = abs(current) / capacity if capacity > 0 else 0
        if current_density > 2.0:  # C-rate
            compliance_issues.append({
                'issue': 'excessive_current_density',
                'description': 'Current density exceeds UL2054 recommendations',
                'current_value': current_density,
                'standard_limit': 2.0,
                'severity': 'medium'
            })
        
        # Overcharge protection (UL2054 Section 20.1)
        soc = battery_data.get('soc', 0.5)
        if soc > 1.0:
            compliance_issues.append({
                'issue': 'overcharge_condition',
                'description': 'State of charge exceeds 100% - overcharge protection required',
                'current_value': soc,
                'standard_limit': 1.0,
                'severity': 'critical'
            })
        
        # Predicted failure modes check
        if 'failure_modes' in model_predictions:
            for failure_mode in model_predictions['failure_modes']:
                if failure_mode['probability'] > 0.1:  # 10% threshold
                    compliance_issues.append({
                        'issue': 'predicted_failure_mode',
                        'description': f'High probability of {failure_mode["type"]} failure',
                        'current_value': failure_mode['probability'],
                        'standard_limit': 0.1,
                        'severity': 'high'
                    })
        
        return {
            'compliant': len(compliance_issues) == 0,
            'standard': 'UL2054',
            'issues': compliance_issues,
            'total_issues': len(compliance_issues),
            'critical_issues': len([i for i in compliance_issues if i['severity'] == 'critical']),
            'compliance_score': max(0, 1 - len(compliance_issues) / 10)
        }
    
    def _check_iec62133_compliance(self, battery_data: Dict[str, Any], model_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Check IEC62133 compliance for secondary cells and batteries."""
        compliance_issues = []
        
        # IEC62133 specific checks
        temperature = battery_data.get('temperature', 25.0)
        voltage = battery_data.get('voltage', 3.7)
        internal_resistance = battery_data.get('internal_resistance', 0.1)
        
        # Temperature cycling limits (IEC62133-2 Section 7.3.6)
        if temperature > 65.0:
            compliance_issues.append({
                'issue': 'temperature_cycling_limit',
                'description': 'Temperature exceeds IEC62133 cycling limit of 65°C',
                'current_value': temperature,
                'standard_limit': 65.0,
                'severity': 'high'
            })
        
        # Voltage stress test (IEC62133-2 Section 7.3.14)
        if voltage > 4.35:
            compliance_issues.append({
                'issue': 'voltage_stress_limit',
                'description': 'Voltage exceeds IEC62133 stress test limit',
                'current_value': voltage,
                'standard_limit': 4.35,
                'severity': 'high'
            })
        
        # Internal short circuit protection (IEC62133-2 Section 7.3.15)
        if internal_resistance < 0.01:
            compliance_issues.append({
                'issue': 'internal_short_risk',
                'description': 'Internal resistance too low, potential short circuit risk',
                'current_value': internal_resistance,
                'standard_limit': 0.01,
                'severity': 'critical'
            })
        
        # Thermal runaway protection (IEC62133-2 Section 7.3.16)
        if 'thermal_runaway_risk' in model_predictions:
            thermal_risk = model_predictions['thermal_runaway_risk']
            if thermal_risk > 0.05:  # 5% threshold
                compliance_issues.append({
                    'issue': 'thermal_runaway_risk',
                    'description': 'Predicted thermal runaway risk exceeds IEC62133 safety threshold',
                    'current_value': thermal_risk,
                    'standard_limit': 0.05,
                    'severity': 'critical'
                })
        
        # Battery pack safety (IEC62133-2 Section 8)
        if 'battery_pack_safety' in model_predictions:
            pack_safety = model_predictions['battery_pack_safety']
            if pack_safety['cell_imbalance'] > 0.1:  # 100mV
                compliance_issues.append({
                    'issue': 'cell_imbalance',
                    'description': 'Cell voltage imbalance exceeds IEC62133 limits',
                    'current_value': pack_safety['cell_imbalance'],
                    'standard_limit': 0.1,
                    'severity': 'medium'
                })
        
        return {
            'compliant': len(compliance_issues) == 0,
            'standard': 'IEC62133',
            'issues': compliance_issues,
            'total_issues': len(compliance_issues),
            'critical_issues': len([i for i in compliance_issues if i['severity'] == 'critical']),
            'compliance_score': max(0, 1 - len(compliance_issues) / 8)
        }
    
    def _check_un383_compliance(self, battery_data: Dict[str, Any], model_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Check UN38.3 compliance for lithium battery transport."""
        compliance_issues = []
        
        # UN38.3 specific checks for transport safety
        temperature = battery_data.get('temperature', 25.0)
        voltage = battery_data.get('voltage', 3.7)
        soc = battery_data.get('soc', 0.5)
        
        # Temperature test (UN38.3 Test T.1)
        if temperature > 72.0:
            compliance_issues.append({
                'issue': 'temperature_test_failure',
                'description': 'Temperature exceeds UN38.3 test limit of 72°C',
                'current_value': temperature,
                'standard_limit': 72.0,
                'severity': 'high'
            })
        
        # Altitude simulation (UN38.3 Test T.2)
        pressure = battery_data.get('ambient_pressure', 101325)  # Pa
        if pressure < 11600:  # 15km altitude equivalent
            compliance_issues.append({
                'issue': 'altitude_simulation_risk',
                'description': 'Low pressure conditions may affect battery safety',
                'current_value': pressure,
                'standard_limit': 11600,
                'severity': 'medium'
            })
        
        # Thermal test (UN38.3 Test T.3)
        if temperature < -40.0:
            compliance_issues.append({
                'issue': 'thermal_test_low_temp',
                'description': 'Temperature below UN38.3 thermal test limit',
                'current_value': temperature,
                'standard_limit': -40.0,
                'severity': 'medium'
            })
        
        # Vibration test compliance (UN38.3 Test T.4)
        if 'vibration_exposure' in battery_data:
            vibration = battery_data['vibration_exposure']
            if vibration > 8.0:  # g-force
                compliance_issues.append({
                    'issue': 'vibration_test_failure',
                    'description': 'Vibration exposure exceeds UN38.3 test limits',
                    'current_value': vibration,
                    'standard_limit': 8.0,
                    'severity': 'high'
                })
        
        # Shock test (UN38.3 Test T.5)
        if 'shock_exposure' in battery_data:
            shock = battery_data['shock_exposure']
            if shock > 150.0:  # g-force
                compliance_issues.append({
                    'issue': 'shock_test_failure',
                    'description': 'Shock exposure exceeds UN38.3 test limits',
                    'current_value': shock,
                    'standard_limit': 150.0,
                    'severity': 'high'
                })
        
        # External short circuit (UN38.3 Test T.6)
        if 'short_circuit_risk' in model_predictions:
            short_risk = model_predictions['short_circuit_risk']
            if short_risk > 0.01:  # 1% threshold
                compliance_issues.append({
                    'issue': 'external_short_circuit_risk',
                    'description': 'External short circuit risk exceeds UN38.3 safety threshold',
                    'current_value': short_risk,
                    'standard_limit': 0.01,
                    'severity': 'critical'
                })
        
        # Impact test (UN38.3 Test T.7)
        if 'impact_damage' in model_predictions:
            impact_risk = model_predictions['impact_damage']
            if impact_risk > 0.05:  # 5% threshold
                compliance_issues.append({
                    'issue': 'impact_test_failure',
                    'description': 'Impact damage risk exceeds UN38.3 limits',
                    'current_value': impact_risk,
                    'standard_limit': 0.05,
                    'severity': 'high'
                })
        
        # Overcharge test (UN38.3 Test T.8)
        if soc > 1.0:
            compliance_issues.append({
                'issue': 'overcharge_test_failure',
                'description': 'Overcharge condition detected - UN38.3 test failure',
                'current_value': soc,
                'standard_limit': 1.0,
                'severity': 'critical'
            })
        
        return {
            'compliant': len(compliance_issues) == 0,
            'standard': 'UN38.3',
            'issues': compliance_issues,
            'total_issues': len(compliance_issues),
            'critical_issues': len([i for i in compliance_issues if i['severity'] == 'critical']),
            'compliance_score': max(0, 1 - len(compliance_issues) / 12)
        }
    
    def _check_sae_j2464_compliance(self, battery_data: Dict[str, Any], model_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Check SAE J2464 compliance for electric vehicle batteries."""
        compliance_issues = []
        
        # SAE J2464 specific checks for EV batteries
        temperature = battery_data.get('temperature', 25.0)
        voltage = battery_data.get('voltage', 3.7)
        current = battery_data.get('current', 0.0)
        soc = battery_data.get('soc', 0.5)
        
        # Operating temperature range (SAE J2464 Section 4.1)
        if temperature > 60.0 or temperature < -30.0:
            compliance_issues.append({
                'issue': 'operating_temperature_range',
                'description': 'Temperature outside SAE J2464 operating range (-30°C to 60°C)',
                'current_value': temperature,
                'standard_limit': '[-30, 60]',
                'severity': 'high'
            })
        
        # Voltage stability (SAE J2464 Section 4.2)
        if 'voltage_stability' in model_predictions:
            voltage_stability = model_predictions['voltage_stability']
            if voltage_stability < 0.95:  # 95% stability
                compliance_issues.append({
                    'issue': 'voltage_stability',
                    'description': 'Voltage stability below SAE J2464 requirements',
                    'current_value': voltage_stability,
                    'standard_limit': 0.95,
                    'severity': 'medium'
                })
        
        # Power capability (SAE J2464 Section 4.3)
        capacity = battery_data.get('capacity', 100.0)
        power_capability = abs(current * voltage)
        specific_power = power_capability / capacity if capacity > 0 else 0
        
        if specific_power > 1000:  # W/kg limit
            compliance_issues.append({
                'issue': 'power_capability_limit',
                'description': 'Specific power exceeds SAE J2464 safety limits',
                'current_value': specific_power,
                'standard_limit': 1000,
                'severity': 'high'
            })
        
        # Energy efficiency (SAE J2464 Section 4.4)
        if 'energy_efficiency' in model_predictions:
            efficiency = model_predictions['energy_efficiency']
            if efficiency < 0.85:  # 85% minimum efficiency
                compliance_issues.append({
                    'issue': 'energy_efficiency',
                    'description': 'Energy efficiency below SAE J2464 minimum requirements',
                    'current_value': efficiency,
                    'standard_limit': 0.85,
                    'severity': 'medium'
                })
        
        # Cycle life (SAE J2464 Section 4.5)
        if 'cycle_life' in model_predictions:
            cycle_life = model_predictions['cycle_life']
            if cycle_life < 1000:  # Minimum cycle life
                compliance_issues.append({
                    'issue': 'cycle_life_requirement',
                    'description': 'Predicted cycle life below SAE J2464 minimum',
                    'current_value': cycle_life,
                    'standard_limit': 1000,
                    'severity': 'medium'
                })
        
        # Thermal management (SAE J2464 Section 4.6)
        if 'thermal_management' in model_predictions:
            thermal_mgmt = model_predictions['thermal_management']
            if thermal_mgmt['cooling_effectiveness'] < 0.8:
                compliance_issues.append({
                    'issue': 'thermal_management_effectiveness',
                    'description': 'Thermal management effectiveness below SAE J2464 requirements',
                    'current_value': thermal_mgmt['cooling_effectiveness'],
                    'standard_limit': 0.8,
                    'severity': 'medium'
                })
        
        # Safety systems (SAE J2464 Section 4.7)
        if 'safety_systems' in model_predictions:
            safety_systems = model_predictions['safety_systems']
            if not safety_systems.get('emergency_shutdown', False):
                compliance_issues.append({
                    'issue': 'emergency_shutdown_system',
                    'description': 'Emergency shutdown system not available or non-functional',
                    'current_value': False,
                    'standard_limit': True,
                    'severity': 'critical'
                })
        
        return {
            'compliant': len(compliance_issues) == 0,
            'standard': 'SAE J2464',
            'issues': compliance_issues,
            'total_issues': len(compliance_issues),
            'critical_issues': len([i for i in compliance_issues if i['severity'] == 'critical']),
            'compliance_score': max(0, 1 - len(compliance_issues) / 10)
        }
    
    def _assess_operational_safety(self, 
                                 battery_data: Dict[str, Any],
                                 model_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Assess operational safety conditions."""
        safety_assessment = {
            'overall_safety_score': 0.0,
            'safety_factors': {},
            'recommendations': [],
            'immediate_actions': []
        }
        
        # Temperature safety
        temp_safety = self._assess_temperature_safety(battery_data)
        safety_assessment['safety_factors']['temperature'] = temp_safety
        
        # Electrical safety
        electrical_safety = self._assess_electrical_safety(battery_data)
        safety_assessment['safety_factors']['electrical'] = electrical_safety
        
        # Mechanical safety
        mechanical_safety = self._assess_mechanical_safety(battery_data)
        safety_assessment['safety_factors']['mechanical'] = mechanical_safety
        
        # Chemical safety
        chemical_safety = self._assess_chemical_safety(battery_data, model_predictions)
        safety_assessment['safety_factors']['chemical'] = chemical_safety
        
        # Calculate overall safety score
        safety_scores = [
            temp_safety['score'],
            electrical_safety['score'],
            mechanical_safety['score'],
            chemical_safety['score']
        ]
        safety_assessment['overall_safety_score'] = np.mean(safety_scores)
        
        # Generate recommendations
        safety_assessment['recommendations'] = self._generate_safety_recommendations(
            safety_assessment['safety_factors']
        )
        
        # Identify immediate actions
        safety_assessment['immediate_actions'] = self._identify_immediate_actions(
            safety_assessment['safety_factors']
        )
        
        return safety_assessment
    
    def _assess_temperature_safety(self, battery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess temperature-related safety factors."""
        temperature = battery_data.get('temperature', 25.0)
        temp_gradient = battery_data.get('temperature_gradient', 0.0)
        
        safety_score = 1.0
        issues = []
        
        # Critical temperature thresholds
        if temperature > 60.0:
            safety_score *= 0.3  # Severe penalty
            issues.append('critical_high_temperature')
        elif temperature > 50.0:
            safety_score *= 0.7  # Moderate penalty
            issues.append('high_temperature_warning')
        elif temperature < -20.0:
            safety_score *= 0.5  # Cold temperature penalty
            issues.append('low_temperature_warning')
        
        # Temperature gradient assessment
        if abs(temp_gradient) > 5.0:  # °C/min
            safety_score *= 0.8
            issues.append('rapid_temperature_change')
        
        return {
            'score': safety_score,
            'issues': issues,
            'temperature': temperature,
            'temperature_gradient': temp_gradient,
            'status': 'critical' if safety_score < 0.5 else 'warning' if safety_score < 0.8 else 'normal'
        }
    
    def _assess_electrical_safety(self, battery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess electrical safety factors."""
        voltage = battery_data.get('voltage', 3.7)
        current = battery_data.get('current', 0.0)
        internal_resistance = battery_data.get('internal_resistance', 0.1)
        
        safety_score = 1.0
        issues = []
        
        # Voltage safety assessment
        if voltage > 4.3:
            safety_score *= 0.4  # Overvoltage penalty
            issues.append('overvoltage_condition')
        elif voltage < 2.5:
            safety_score *= 0.6  # Undervoltage penalty
            issues.append('undervoltage_condition')
        
        # Current safety assessment
        if abs(current) > 100.0:
            safety_score *= 0.5  # High current penalty
            issues.append('excessive_current')
        
        # Internal resistance assessment
        if internal_resistance < 0.01:
            safety_score *= 0.3  # Short circuit risk
            issues.append('low_internal_resistance')
        elif internal_resistance > 1.0:
            safety_score *= 0.8  # High resistance warning
            issues.append('high_internal_resistance')
        
        return {
            'score': safety_score,
            'issues': issues,
            'voltage': voltage,
            'current': current,
            'internal_resistance': internal_resistance,
            'status': 'critical' if safety_score < 0.5 else 'warning' if safety_score < 0.8 else 'normal'
        }
    
    def _assess_mechanical_safety(self, battery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess mechanical safety factors."""
        vibration = battery_data.get('vibration_level', 0.0)
        shock = battery_data.get('shock_level', 0.0)
        pressure = battery_data.get('internal_pressure', 0.0)
        
        safety_score = 1.0
        issues = []
        
        # Vibration assessment
        if vibration > 5.0:  # g-force
            safety_score *= 0.7
            issues.append('excessive_vibration')
        
        # Shock assessment
        if shock > 50.0:  # g-force
            safety_score *= 0.6
            issues.append('excessive_shock')
        
        # Internal pressure assessment
        if pressure > 0.5:  # bar
            safety_score *= 0.4
            issues.append('excessive_internal_pressure')
        
        return {
            'score': safety_score,
            'issues': issues,
            'vibration_level': vibration,
            'shock_level': shock,
            'internal_pressure': pressure,
            'status': 'critical' if safety_score < 0.5 else 'warning' if safety_score < 0.8 else 'normal'
        }
    
    def _assess_chemical_safety(self, battery_data: Dict[str, Any], model_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Assess chemical safety factors."""
        electrolyte_level = battery_data.get('electrolyte_level', 1.0)
        gas_evolution = battery_data.get('gas_evolution_rate', 0.0)
        
        safety_score = 1.0
        issues = []
        
        # Electrolyte level assessment
        if electrolyte_level < 0.3:
            safety_score *= 0.5
            issues.append('low_electrolyte_level')
        
        # Gas evolution assessment
        if gas_evolution > 0.1:  # ml/min
            safety_score *= 0.6
            issues.append('excessive_gas_evolution')
        
        # Thermal runaway risk from predictions
        if 'thermal_runaway_risk' in model_predictions:
            thermal_risk = model_predictions['thermal_runaway_risk']
            if thermal_risk > 0.1:
                safety_score *= 0.3
                issues.append('thermal_runaway_risk')
        
        return {
            'score': safety_score,
            'issues': issues,
            'electrolyte_level': electrolyte_level,
            'gas_evolution_rate': gas_evolution,
            'status': 'critical' if safety_score < 0.5 else 'warning' if safety_score < 0.8 else 'normal'
        }
    
    def _generate_safety_recommendations(self, safety_factors: Dict[str, Any]) -> List[str]:
        """Generate safety recommendations based on assessment."""
        recommendations = []
        
        for factor_name, factor_data in safety_factors.items():
            if factor_data['status'] == 'critical':
                recommendations.append(f"CRITICAL: Immediate attention required for {factor_name} safety")
            elif factor_data['status'] == 'warning':
                recommendations.append(f"WARNING: Monitor {factor_name} safety closely")
            
            # Specific recommendations based on issues
            for issue in factor_data['issues']:
                if issue == 'critical_high_temperature':
                    recommendations.append("Implement emergency cooling measures immediately")
                elif issue == 'overvoltage_condition':
                    recommendations.append("Activate overvoltage protection systems")
                elif issue == 'low_internal_resistance':
                    recommendations.append("Check for internal short circuit, isolate battery")
                elif issue == 'thermal_runaway_risk':
                    recommendations.append("Initiate thermal runaway prevention protocols")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _identify_immediate_actions(self, safety_factors: Dict[str, Any]) -> List[str]:
        """Identify immediate actions required."""
        immediate_actions = []
        
        # Check for critical conditions requiring immediate action
        for factor_name, factor_data in safety_factors.items():
            if factor_data['score'] < 0.5:  # Critical threshold
                immediate_actions.append(f"EMERGENCY: Shut down {factor_name} systems")
            
            # Specific immediate actions
            for issue in factor_data['issues']:
                if issue in ['critical_high_temperature', 'overvoltage_condition', 'thermal_runaway_risk']:
                    immediate_actions.append("IMMEDIATE: Activate emergency shutdown procedures")
                elif issue == 'low_internal_resistance':
                    immediate_actions.append("IMMEDIATE: Isolate battery from all connections")
                elif issue == 'excessive_internal_pressure':
                    immediate_actions.append("IMMEDIATE: Evacuate area and contact emergency services")
        
        return list(set(immediate_actions))  # Remove duplicates
    
    def generate_safety_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate comprehensive safety validation report."""
        report = []
        report.append("=" * 60)
        report.append("BATTERYMIND SAFETY VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Validation ID: {validation_results['validation_id']}")
        report.append()
        
        # Overall safety status
        overall_score = validation_results['overall_safety_score']
        status = "SAFE" if overall_score >= 0.8 else "WARNING" if overall_score >= 0.6 else "CRITICAL"
        report.append(f"OVERALL SAFETY STATUS: {status}")
        report.append(f"Overall Safety Score: {overall_score:.2f}/1.00")
        report.append()
        
        # Safety constraint violations
        if validation_results['constraint_violations']:
            report.append("SAFETY CONSTRAINT VIOLATIONS:")
            report.append("-" * 40)
            for violation in validation_results['constraint_violations']:
                report.append(f"• {violation['constraint']}: {violation['description']}")
                report.append(f"  Current: {violation['current_value']}, Limit: {violation['limit']}")
                report.append(f"  Severity: {violation['severity']}")
            report.append()
        
        # Regulatory compliance
        if 'regulatory_compliance' in validation_results:
            compliance = validation_results['regulatory_compliance']
            report.append("REGULATORY COMPLIANCE:")
            report.append("-" * 40)
            report.append(f"Overall Compliance: {'PASS' if compliance['overall_compliance'] else 'FAIL'}")
            
            for standard, result in compliance['compliance_details'].items():
                status = "PASS" if result['compliant'] else "FAIL"
                report.append(f"{standard}: {status} (Score: {result['compliance_score']:.2f})")
                if result['issues']:
                    for issue in result['issues']:
                        report.append(f"  - {issue['description']} ({issue['severity']})")
            report.append()
        
        # Safety recommendations
        if validation_results['safety_recommendations']:
            report.append("SAFETY RECOMMENDATIONS:")
            report.append("-" * 40)
            for rec in validation_results['safety_recommendations']:
                report.append(f"• {rec}")
            report.append()
        
        # Immediate actions
        if validation_results['immediate_actions']:
            report.append("IMMEDIATE ACTIONS REQUIRED:")
            report.append("-" * 40)
            for action in validation_results['immediate_actions']:
                report.append(f"• {action}")
            report.append()
        
        # Validation statistics
        stats = validation_results['validation_statistics']
        report.append("VALIDATION STATISTICS:")
        report.append("-" * 40)
        report.append(f"Total Constraints Checked: {stats['total_constraints_checked']}")
        report.append(f"Constraints Passed: {stats['constraints_passed']}")
        report.append(f"Constraints Failed: {stats['constraints_failed']}")
        report.append(f"Validation Duration: {stats['validation_duration']:.2f}s")
        report.append()
        
        report.append("=" * 60)
        report.append("END OF SAFETY VALIDATION REPORT")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def export_validation_results(self, results: Dict[str, Any], filepath: str):
        """Export validation results to file."""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Validation results exported to {filepath}")
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get validation history."""
        return self.validation_history.copy()
    
    def clear_validation_history(self):
        """Clear validation history."""
        self.validation_history.clear()
        logger.info("Validation history cleared")

# Factory function
def create_safety_validator(config: Optional[SafetyValidationConfig] = None) -> SafetyValidator:
    """
    Factory function to create a safety validator.
    
    Args:
        config: Validation configuration
        
    Returns:
        Configured SafetyValidator instance
    """
    if config is None:
        config = SafetyValidationConfig()
    
    return SafetyValidator(config)

# Example usage
if __name__ == "__main__":
    # Example battery data
    battery_data = {
        'voltage': 3.8,
        'current': 15.0,
        'temperature': 35.0,
        'soc': 0.7,
        'soh': 0.85,
        'internal_resistance': 0.08,
        'capacity': 100.0,
        'cycle_count': 500
    }
    
    # Example model predictions
    model_predictions = {
        'degradation_rate': 0.001,
        'failure_modes': [
            {'type': 'capacity_fade', 'probability': 0.05},
            {'type': 'internal_short', 'probability': 0.01}
        ],
        'thermal_runaway_risk': 0.02,
        'remaining_useful_life': 2000
    }
    
    # Create validator
    validator = create_safety_validator()
    
    # Validate safety
    results = validator.validate_safety(battery_data, model_predictions, 'lithium_ion')
    
    # Generate report
    report = validator.generate_safety_report(results)
    print(report)
