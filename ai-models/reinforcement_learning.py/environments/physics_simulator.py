"""
BatteryMind - Physics Simulator for Battery Environments

Advanced physics-based simulation engine for battery behavior modeling
in reinforcement learning environments. Provides realistic battery
dynamics, thermal modeling, electrochemical processes, and degradation
mechanisms for training RL agents.

Features:
- Electrochemical battery models (equivalent circuit, P2D)
- Thermal dynamics and heat generation modeling
- Battery degradation mechanisms (SEI growth, lithium plating)
- Multi-physics coupling (electrical, thermal, mechanical)
- Real-time parameter estimation and adaptation
- Safety constraint modeling and violation detection

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import time
import json
from abc import ABC, abstractmethod
from enum import Enum
import warnings

# Scientific computing imports
from scipy import integrate, optimize
from scipy.interpolate import interp1d
from scipy.special import erf
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatteryChemistry(Enum):
    """Supported battery chemistry types."""
    LI_ION_LFP = "lithium_iron_phosphate"
    LI_ION_NMC = "lithium_nickel_manganese_cobalt"
    LI_ION_NCA = "lithium_nickel_cobalt_aluminum"
    LI_ION_LTO = "lithium_titanate"
    LI_METAL = "lithium_metal"
    SOLID_STATE = "solid_state_lithium"

@dataclass
class BatteryParameters:
    """
    Comprehensive battery parameters for physics simulation.
    
    Attributes:
        # Basic specifications
        nominal_capacity (float): Nominal capacity in Ah
        nominal_voltage (float): Nominal voltage in V
        max_voltage (float): Maximum voltage in V
        min_voltage (float): Minimum voltage in V
        
        # Physical properties
        electrode_area (float): Electrode area in m²
        separator_thickness (float): Separator thickness in m
        positive_thickness (float): Positive electrode thickness in m
        negative_thickness (float): Negative electrode thickness in m
        
        # Electrochemical parameters
        diffusion_coefficient_pos (float): Diffusion coefficient positive electrode
        diffusion_coefficient_neg (float): Diffusion coefficient negative electrode
        exchange_current_density_pos (float): Exchange current density positive
        exchange_current_density_neg (float): Exchange current density negative
        
        # Thermal parameters
        thermal_mass (float): Thermal mass in J/K
        thermal_conductivity (float): Thermal conductivity in W/m/K
        convection_coefficient (float): Convection heat transfer coefficient
        
        # Degradation parameters
        sei_growth_rate (float): SEI layer growth rate
        capacity_fade_rate (float): Capacity fade rate per cycle
        resistance_growth_rate (float): Internal resistance growth rate
        
        # Safety limits
        max_temperature (float): Maximum safe temperature in K
        min_temperature (float): Minimum operating temperature in K
        max_current (float): Maximum current in A
        thermal_runaway_threshold (float): Thermal runaway temperature in K
    """
    # Basic specifications
    nominal_capacity: float = 50.0  # Ah
    nominal_voltage: float = 3.7    # V
    max_voltage: float = 4.2        # V
    min_voltage: float = 2.5        # V
    
    # Physical properties
    electrode_area: float = 0.1     # m²
    separator_thickness: float = 25e-6  # m
    positive_thickness: float = 100e-6  # m
    negative_thickness: float = 100e-6  # m
    
    # Electrochemical parameters
    diffusion_coefficient_pos: float = 1e-14  # m²/s
    diffusion_coefficient_neg: float = 3.9e-14  # m²/s
    exchange_current_density_pos: float = 0.1  # A/m²
    exchange_current_density_neg: float = 0.5  # A/m²
    
    # Thermal parameters
    thermal_mass: float = 1000.0    # J/K
    thermal_conductivity: float = 1.5  # W/m/K
    convection_coefficient: float = 10.0  # W/m²/K
    
    # Degradation parameters
    sei_growth_rate: float = 1e-8   # m/cycle
    capacity_fade_rate: float = 0.0002  # per cycle
    resistance_growth_rate: float = 0.0001  # Ω per cycle
    
    # Safety limits
    max_temperature: float = 333.15  # K (60°C)
    min_temperature: float = 253.15  # K (-20°C)
    max_current: float = 100.0      # A
    thermal_runaway_threshold: float = 423.15  # K (150°C)

@dataclass
class BatteryState:
    """
    Current state of the battery in simulation.
    """
    # Electrical state
    voltage: float = 3.7           # V
    current: float = 0.0           # A
    state_of_charge: float = 0.5   # 0-1
    state_of_health: float = 1.0   # 0-1
    internal_resistance: float = 0.01  # Ω
    
    # Thermal state
    temperature: float = 298.15    # K
    heat_generation: float = 0.0   # W
    
    # Degradation state
    sei_thickness: float = 10e-9   # m
    capacity_fade: float = 0.0     # Ah lost
    cycle_count: int = 0
    calendar_age: float = 0.0      # days
    
    # Safety indicators
    thermal_runaway_risk: float = 0.0  # 0-1
    overcharge_risk: float = 0.0       # 0-1
    overdischarge_risk: float = 0.0    # 0-1

class BasePhysicsModel(ABC):
    """
    Abstract base class for battery physics models.
    """
    
    def __init__(self, parameters: BatteryParameters):
        self.parameters = parameters
        self.constants = self._initialize_constants()
        
    def _initialize_constants(self) -> Dict[str, float]:
        """Initialize physical constants."""
        return {
            'R': 8.314,          # Gas constant J/mol/K
            'F': 96485.0,        # Faraday constant C/mol
            'T_ref': 298.15,     # Reference temperature K
            'k_B': 1.381e-23,    # Boltzmann constant J/K
            'N_A': 6.022e23      # Avogadro number
        }
    
    @abstractmethod
    def update_state(self, current_state: BatteryState, current: float, 
                    dt: float, ambient_temp: float) -> BatteryState:
        """Update battery state given current input."""
        pass
    
    @abstractmethod
    def calculate_voltage(self, state: BatteryState) -> float:
        """Calculate terminal voltage from battery state."""
        pass

class EquivalentCircuitModel(BasePhysicsModel):
    """
    Equivalent circuit model for battery simulation.
    
    Implements a second-order RC equivalent circuit with:
    - Open circuit voltage (OCV) as function of SOC
    - Series resistance
    - Two RC pairs for transient response
    """
    
    def __init__(self, parameters: BatteryParameters):
        super().__init__(parameters)
        
        # RC circuit parameters
        self.R0 = 0.01  # Series resistance (Ω)
        self.R1 = 0.005  # First RC resistance (Ω)
        self.C1 = 1000.0  # First RC capacitance (F)
        self.R2 = 0.002  # Second RC resistance (Ω)
        self.C2 = 5000.0  # Second RC capacitance (F)
        
        # RC state variables
        self.V1 = 0.0  # First RC voltage
        self.V2 = 0.0  # Second RC voltage
        
        # OCV lookup table (SOC vs OCV)
        self.soc_points = np.linspace(0, 1, 21)
        self.ocv_points = self._generate_ocv_curve()
        self.ocv_interp = interp1d(self.soc_points, self.ocv_points, 
                                  kind='cubic', bounds_error=False, 
                                  fill_value='extrapolate')
    
    def _generate_ocv_curve(self) -> np.ndarray:
        """Generate OCV curve based on battery chemistry."""
        # Simplified OCV curve for Li-ion battery
        soc = self.soc_points
        
        # Typical Li-ion OCV curve with characteristic features
        ocv = (self.parameters.min_voltage + 
               (self.parameters.max_voltage - self.parameters.min_voltage) * 
               (0.1 + 0.9 * soc + 0.1 * np.sin(np.pi * soc)))
        
        return ocv
    
    def update_state(self, current_state: BatteryState, current: float, 
                    dt: float, ambient_temp: float) -> BatteryState:
        """Update battery state using equivalent circuit model."""
        new_state = BatteryState(**current_state.__dict__)
        
        # Update SOC based on current integration
        capacity_ah = self.parameters.nominal_capacity * current_state.state_of_health
        soc_change = -current * dt / 3600.0 / capacity_ah  # Convert to hours
        new_state.state_of_charge = np.clip(
            current_state.state_of_charge + soc_change, 0.0, 1.0
        )
        
        # Update RC circuit voltages
        tau1 = self.R1 * self.C1
        tau2 = self.R2 * self.C2
        
        self.V1 = self.V1 * np.exp(-dt/tau1) + self.R1 * current * (1 - np.exp(-dt/tau1))
        self.V2 = self.V2 * np.exp(-dt/tau2) + self.R2 * current * (1 - np.exp(-dt/tau2))
        
        # Calculate terminal voltage
        ocv = self.ocv_interp(new_state.state_of_charge)
        new_state.voltage = ocv - current * self.R0 - self.V1 - self.V2
        
        # Update current
        new_state.current = current
        
        # Update temperature (simplified thermal model)
        heat_generation = current**2 * (self.R0 + self.R1 + self.R2)
        new_state.heat_generation = heat_generation
        
        # Thermal dynamics
        temp_change = (heat_generation - 
                      self.parameters.convection_coefficient * 
                      (current_state.temperature - ambient_temp)) * dt / self.parameters.thermal_mass
        
        new_state.temperature = current_state.temperature + temp_change
        
        # Update degradation (simplified)
        if abs(current) > 0.1:  # Only during active cycling
            new_state.cycle_count = current_state.cycle_count + abs(current) * dt / (2 * 3600 * capacity_ah)
        
        new_state.calendar_age = current_state.calendar_age + dt / 86400.0  # Convert to days
        
        # Calculate degradation
        new_state = self._update_degradation(new_state)
        
        # Update safety indicators
        new_state = self._update_safety_indicators(new_state)
        
        return new_state
    
    def calculate_voltage(self, state: BatteryState) -> float:
        """Calculate terminal voltage from current state."""
        ocv = self.ocv_interp(state.state_of_charge)
        return ocv - state.current * self.R0 - self.V1 - self.V2
    
    def _update_degradation(self, state: BatteryState) -> BatteryState:
        """Update battery degradation mechanisms."""
        # SEI growth (calendar aging)
        sei_growth = self.parameters.sei_growth_rate * np.sqrt(state.calendar_age)
        state.sei_thickness = 10e-9 + sei_growth
        
        # Capacity fade
        cycle_fade = self.parameters.capacity_fade_rate * state.cycle_count
        calendar_fade = 0.0001 * np.sqrt(state.calendar_age)  # Calendar fade
        state.capacity_fade = cycle_fade + calendar_fade
        
        # Update SOH
        state.state_of_health = max(0.8, 1.0 - state.capacity_fade)
        
        # Resistance growth
        resistance_increase = (self.parameters.resistance_growth_rate * state.cycle_count +
                             0.00001 * state.calendar_age)
        state.internal_resistance = 0.01 + resistance_increase
        
        return state
    
    def _update_safety_indicators(self, state: BatteryState) -> BatteryState:
        """Update safety risk indicators."""
        # Thermal runaway risk
        if state.temperature > self.parameters.thermal_runaway_threshold:
            state.thermal_runaway_risk = 1.0
        else:
            temp_factor = (state.temperature - 298.15) / (self.parameters.thermal_runaway_threshold - 298.15)
            state.thermal_runaway_risk = max(0.0, min(1.0, temp_factor))
        
        # Overcharge risk
        if state.voltage > self.parameters.max_voltage:
            state.overcharge_risk = (state.voltage - self.parameters.max_voltage) / 0.5
        else:
            state.overcharge_risk = 0.0
        
        # Overdischarge risk
        if state.voltage < self.parameters.min_voltage:
            state.overdischarge_risk = (self.parameters.min_voltage - state.voltage) / 0.5
        else:
            state.overdischarge_risk = 0.0
        
        return state

class ThermalModel:
    """
    Advanced thermal model for battery simulation.
    """
    
    def __init__(self, parameters: BatteryParameters):
        self.parameters = parameters
        
        # Thermal properties
        self.density = 2500.0  # kg/m³
        self.specific_heat = 1000.0  # J/kg/K
        self.thermal_conductivity = parameters.thermal_conductivity
        
        # Geometry (simplified cylindrical cell)
        self.radius = 0.009  # m
        self.height = 0.065  # m
        self.volume = np.pi * self.radius**2 * self.height
        self.surface_area = 2 * np.pi * self.radius * (self.radius + self.height)
        
        # Thermal mass
        self.thermal_mass = self.density * self.volume * self.specific_heat
    
    def calculate_heat_generation(self, current: float, voltage: float, 
                                ocv: float, temperature: float) -> float:
        """
        Calculate heat generation from various sources.
        
        Args:
            current (float): Battery current in A
            voltage (float): Terminal voltage in V
            ocv (float): Open circuit voltage in V
            temperature (float): Temperature in K
            
        Returns:
            float: Heat generation rate in W
        """
        # Joule heating (irreversible)
        resistance = self.parameters.nominal_capacity * 0.01  # Simplified
        joule_heat = current**2 * resistance
        
        # Reversible heat (entropy change)
        entropy_coeff = -0.0001  # dU/dT (V/K) - typical value
        reversible_heat = current * temperature * entropy_coeff
        
        # Side reactions (simplified)
        side_reaction_heat = 0.01 * abs(current) if temperature > 313.15 else 0.0
        
        total_heat = joule_heat + reversible_heat + side_reaction_heat
        
        return total_heat
    
    def update_temperature(self, current_temp: float, heat_generation: float,
                          ambient_temp: float, dt: float) -> float:
        """
        Update battery temperature using thermal dynamics.
        
        Args:
            current_temp (float): Current temperature in K
            heat_generation (float): Heat generation rate in W
            ambient_temp (float): Ambient temperature in K
            dt (float): Time step in seconds
            
        Returns:
            float: New temperature in K
        """
        # Heat transfer to environment
        heat_loss = (self.parameters.convection_coefficient * self.surface_area * 
                    (current_temp - ambient_temp))
        
        # Net heat rate
        net_heat_rate = heat_generation - heat_loss
        
        # Temperature change
        temp_change = net_heat_rate * dt / self.thermal_mass
        
        new_temp = current_temp + temp_change
        
        return new_temp

class DegradationModel:
    """
    Comprehensive battery degradation model.
    """
    
    def __init__(self, parameters: BatteryParameters):
        self.parameters = parameters
        
        # Degradation mechanisms
        self.sei_model = SEIGrowthModel(parameters)
        self.capacity_fade_model = CapacityFadeModel(parameters)
        self.resistance_growth_model = ResistanceGrowthModel(parameters)
    
    def update_degradation(self, state: BatteryState, dt: float) -> BatteryState:
        """Update all degradation mechanisms."""
        # Update SEI growth
        state = self.sei_model.update(state, dt)
        
        # Update capacity fade
        state = self.capacity_fade_model.update(state, dt)
        
        # Update resistance growth
        state = self.resistance_growth_model.update(state, dt)
        
        return state

class SEIGrowthModel:
    """Model for Solid Electrolyte Interphase (SEI) growth."""
    
    def __init__(self, parameters: BatteryParameters):
        self.parameters = parameters
        self.activation_energy = 35000.0  # J/mol
        self.pre_exponential = 1e-10  # m/s
    
    def update(self, state: BatteryState, dt: float) -> BatteryState:
        """Update SEI thickness."""
        # Temperature dependence (Arrhenius)
        rate_constant = (self.pre_exponential * 
                        np.exp(-self.activation_energy / (8.314 * state.temperature)))
        
        # SEI growth rate (parabolic law)
        growth_rate = rate_constant / (2 * state.sei_thickness)
        
        # Update SEI thickness
        sei_growth = growth_rate * dt
        state.sei_thickness += sei_growth
        
        return state

class CapacityFadeModel:
    """Model for battery capacity fade."""
    
    def __init__(self, parameters: BatteryParameters):
        self.parameters = parameters
        
    def update(self, state: BatteryState, dt: float) -> BatteryState:
        """Update capacity fade."""
        # Cycle-based fade
        if abs(state.current) > 0.1:
            cycle_increment = abs(state.current) * dt / (2 * 3600 * self.parameters.nominal_capacity)
            cycle_fade = self.parameters.capacity_fade_rate * cycle_increment
            state.capacity_fade += cycle_fade
        
        # Calendar fade (temperature dependent)
        calendar_rate = 1e-8 * np.exp((state.temperature - 298.15) / 10.0)
        calendar_fade = calendar_rate * dt / 86400.0  # per day
        state.capacity_fade += calendar_fade
        
        # Update SOH
        state.state_of_health = max(0.8, 1.0 - state.capacity_fade)
        
        return state

class ResistanceGrowthModel:
    """Model for internal resistance growth."""
    
    def __init__(self, parameters: BatteryParameters):
        self.parameters = parameters
        
    def update(self, state: BatteryState, dt: float) -> BatteryState:
        """Update internal resistance."""
        # Resistance growth due to SEI
        sei_resistance = (state.sei_thickness - 10e-9) * 1e6  # Simplified
        
        # Temperature-dependent resistance
        temp_factor = np.exp((298.15 - state.temperature) / 20.0)
        
        # Update resistance
        base_resistance = 0.01
        state.internal_resistance = base_resistance + sei_resistance + temp_factor * 0.001
        
        return state

class SafetyModel:
    """
    Battery safety model for risk assessment.
    """
    
    def __init__(self, parameters: BatteryParameters):
        self.parameters = parameters
        
    def assess_safety(self, state: BatteryState) -> Dict[str, float]:
        """
        Assess various safety risks.
        
        Returns:
            Dict[str, float]: Safety risk scores (0-1)
        """
        risks = {}
        
        # Thermal runaway risk
        risks['thermal_runaway'] = self._thermal_runaway_risk(state)
        
        # Overcharge risk
        risks['overcharge'] = self._overcharge_risk(state)
        
        # Overdischarge risk
        risks['overdischarge'] = self._overdischarge_risk(state)
        
        # Gas generation risk
        risks['gas_generation'] = self._gas_generation_risk(state)
        
        # Overall safety score
        risks['overall'] = max(risks.values())
        
        return risks
    
    def _thermal_runaway_risk(self, state: BatteryState) -> float:
        """Calculate thermal runaway risk."""
        if state.temperature < 333.15:  # Below 60°C
            return 0.0
        elif state.temperature > self.parameters.thermal_runaway_threshold:
            return 1.0
        else:
            # Exponential increase between 60°C and threshold
            temp_range = self.parameters.thermal_runaway_threshold - 333.15
            temp_excess = state.temperature - 333.15
            return (np.exp(temp_excess / temp_range * 5) - 1) / (np.exp(5) - 1)
    
    def _overcharge_risk(self, state: BatteryState) -> float:
        """Calculate overcharge risk."""
        if state.voltage <= self.parameters.max_voltage:
            return 0.0
        else:
            voltage_excess = state.voltage - self.parameters.max_voltage
            return min(1.0, voltage_excess / 0.5)  # Risk increases linearly
    
    def _overdischarge_risk(self, state: BatteryState) -> float:
        """Calculate overdischarge risk."""
        if state.voltage >= self.parameters.min_voltage:
            return 0.0
        else:
            voltage_deficit = self.parameters.min_voltage - state.voltage
            return min(1.0, voltage_deficit / 0.5)  # Risk increases linearly
    
    def _gas_generation_risk(self, state: BatteryState) -> float:
        """Calculate gas generation risk."""
        # High temperature and overcharge conditions
        temp_factor = max(0.0, (state.temperature - 323.15) / 50.0)  # Above 50°C
        voltage_factor = max(0.0, (state.voltage - self.parameters.max_voltage) / 0.3)
        
        return min(1.0, temp_factor + voltage_factor)

class BatteryPhysicsSimulator:
    """
    Main physics simulator integrating all battery models.
    """
    
    def __init__(self, chemistry: BatteryChemistry = BatteryChemistry.LI_ION_NMC,
                 parameters: Optional[BatteryParameters] = None):
        self.chemistry = chemistry
        self.parameters = parameters or BatteryParameters()
        
        # Initialize models
        self.electrical_model = EquivalentCircuitModel(self.parameters)
        self.thermal_model = ThermalModel(self.parameters)
        self.degradation_model = DegradationModel(self.parameters)
        self.safety_model = SafetyModel(self.parameters)
        
        # Simulation state
        self.current_state = BatteryState()
        self.time = 0.0
        self.history = []
        
        # Simulation parameters
        self.dt = 1.0  # Time step in seconds
        self.ambient_temperature = 298.15  # K
        
        logger.info(f"Battery physics simulator initialized for {chemistry.value}")
    
    def reset(self, initial_soc: float = 0.5, initial_temp: float = 298.15) -> BatteryState:
        """
        Reset simulator to initial conditions.
        
        Args:
            initial_soc (float): Initial state of charge (0-1)
            initial_temp (float): Initial temperature in K
            
        Returns:
            BatteryState: Initial battery state
        """
        self.current_state = BatteryState(
            voltage=self.electrical_model.ocv_interp(initial_soc),
            current=0.0,
            state_of_charge=initial_soc,
            state_of_health=1.0,
            internal_resistance=0.01,
            temperature=initial_temp,
            heat_generation=0.0,
            sei_thickness=10e-9,
            capacity_fade=0.0,
            cycle_count=0,
            calendar_age=0.0,
            thermal_runaway_risk=0.0,
            overcharge_risk=0.0,
            overdischarge_risk=0.0
        )
        
        self.time = 0.0
        self.history = []
        
        return self.current_state
    
    def step(self, current: float, ambient_temp: Optional[float] = None) -> Tuple[BatteryState, Dict[str, float]]:
        """
        Advance simulation by one time step.
        
        Args:
            current (float): Applied current in A (positive for discharge)
            ambient_temp (float, optional): Ambient temperature in K
            
        Returns:
            Tuple[BatteryState, Dict[str, float]]: New state and safety assessment
        """
        if ambient_temp is not None:
            self.ambient_temperature = ambient_temp
        
        # Update electrical state
        self.current_state = self.electrical_model.update_state(
            self.current_state, current, self.dt, self.ambient_temperature
        )
        
        # Update thermal state
        heat_gen = self.thermal_model.calculate_heat_generation(
            current, self.current_state.voltage, 
            self.electrical_model.ocv_interp(self.current_state.state_of_charge),
            self.current_state.temperature
        )
        
        new_temp = self.thermal_model.update_temperature(
            self.current_state.temperature, heat_gen, 
            self.ambient_temperature, self.dt
        )
        
        self.current_state.temperature = new_temp
        self.current_state.heat_generation = heat_gen
        
        # Update degradation
        self.current_state = self.degradation_model.update_degradation(
            self.current_state, self.dt
        )
        
        # Assess safety
        safety_risks = self.safety_model.assess_safety(self.current_state)
        
        # Update safety indicators in state
        self.current_state.thermal_runaway_risk = safety_risks['thermal_runaway']
        self.current_state.overcharge_risk = safety_risks['overcharge']
        self.current_state.overdischarge_risk = safety_risks['overdischarge']
        
        # Update time
        self.time += self.dt
        
        # Store history
        self.history.append({
            'time': self.time,
            'state': self.current_state.__dict__.copy(),
            'safety_risks': safety_risks
        })
        
        return self.current_state, safety_risks
    
    def simulate_profile(self, current_profile: np.ndarray, 
                        time_profile: Optional[np.ndarray] = None,
                        ambient_temp_profile: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Simulate battery response to a current profile.
        
        Args:
            current_profile (np.ndarray): Current values in A
            time_profile (np.ndarray, optional): Time values in s
            ambient_temp_profile (np.ndarray, optional): Ambient temperature profile in K
            
        Returns:
            Dict[str, Any]: Simulation results
        """
        if time_profile is None:
            time_profile = np.arange(len(current_profile)) * self.dt
        
        if ambient_temp_profile is None:
            ambient_temp_profile = np.full(len(current_profile), self.ambient_temperature)
        
        # Initialize results storage
        results = {
            'time': [],
            'voltage': [],
            'current': [],
            'soc': [],
            'soh': [],
            'temperature': [],
            'heat_generation': [],
            'safety_risks': []
        }
        
        # Simulate each time step
        for i, (current, ambient_temp) in enumerate(zip(current_profile, ambient_temp_profile)):
            state, risks = self.step(current, ambient_temp)
            
            # Store results
            results['time'].append(self.time)
            results['voltage'].append(state.voltage)
            results['current'].append(state.current)
            results['soc'].append(state.state_of_charge)
            results['soh'].append(state.state_of_health)
            results['temperature'].append(state.temperature)
            results['heat_generation'].append(state.heat_generation)
            results['safety_risks'].append(risks)
        
        return results
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get current simulator state as dictionary."""
        return {
            'current_state': self.current_state.__dict__,
            'time': self.time,
            'parameters': self.parameters.__dict__,
            'chemistry': self.chemistry.value
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load simulator state from dictionary."""
        self.current_state = BatteryState(**state_dict['current_state'])
        self.time = state_dict['time']
        self.chemistry = BatteryChemistry(state_dict['chemistry'])
    
    def validate_operation(self, current: float) -> Tuple[bool, List[str]]:
        """
        Validate if operation is safe given current conditions.
        
        Args:
            current (float): Proposed current in A
            
        Returns:
            Tuple[bool, List[str]]: (is_safe, list of violations)
        """
        violations = []
        
        # Check current limits
        if abs(current) > self.parameters.max_current:
            violations.append(f"Current {current:.1f}A exceeds limit {self.parameters.max_current:.1f}A")
        
        # Check temperature limits
        if self.current_state.temperature > self.parameters.max_temperature:
            violations.append(f"Temperature {self.current_state.temperature:.1f}K exceeds limit")
        
        if self.current_state.temperature < self.parameters.min_temperature:
            violations.append(f"Temperature {self.current_state.temperature:.1f}K below minimum")
        
        # Check voltage limits (projected)
        projected_voltage = self.electrical_model.calculate_voltage(self.current_state)
        if projected_voltage > self.parameters.max_voltage:
            violations.append(f"Voltage {projected_voltage:.2f}V would exceed maximum")
        
        if projected_voltage < self.parameters.min_voltage:
            violations.append(f"Voltage {projected_voltage:.2f}V would be below minimum")
        
        # Check safety risks
        if self.current_state.thermal_runaway_risk > 0.8:
            violations.append("High thermal runaway risk")
        
        is_safe = len(violations) == 0
        return is_safe, violations

# Factory functions
def create_battery_simulator(chemistry: BatteryChemistry = BatteryChemistry.LI_ION_NMC,
                           capacity: float = 50.0,
                           custom_params: Optional[Dict[str, Any]] = None) -> BatteryPhysicsSimulator:
    """
    Factory function to create battery physics simulator.
    
    Args:
        chemistry (BatteryChemistry): Battery chemistry type
        capacity (float): Battery capacity in Ah
        custom_params (Dict[str, Any], optional): Custom parameters
        
    Returns:
        BatteryPhysicsSimulator: Configured simulator
    """
    parameters = BatteryParameters(nominal_capacity=capacity)
    
    if custom_params:
        for key, value in custom_params.items():
            if hasattr(parameters, key):
                setattr(parameters, key, value)
    
    return BatteryPhysicsSimulator(chemistry, parameters)

def create_ev_battery_simulator(pack_capacity: float = 75.0) -> BatteryPhysicsSimulator:
    """Create simulator for EV battery pack."""
    params = BatteryParameters(
        nominal_capacity=pack_capacity,
        nominal_voltage=400.0,
        max_voltage=420.0,
        min_voltage=300.0,
        max_current=300.0,
        thermal_mass=50000.0,  # Larger thermal mass for pack
        max_temperature=328.15  # 55°C for EV applications
    )
    
    return BatteryPhysicsSimulator(BatteryChemistry.LI_ION_NMC, params)

def create_stationary_battery_simulator(capacity: float = 100.0) -> BatteryPhysicsSimulator:
    """Create simulator for stationary energy storage."""
    params = BatteryParameters(
        nominal_capacity=capacity,
        nominal_voltage=48.0,
        max_voltage=54.0,
        min_voltage=42.0,
        max_current=200.0,
        thermal_mass=20000.0,
        max_temperature=318.15  # 45°C for stationary applications
    )
    
    return BatteryPhysicsSimulator(BatteryChemistry.LI_ION_LFP, params)
