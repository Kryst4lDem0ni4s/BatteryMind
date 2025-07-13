"""
BatteryMind - Physics-Based Battery Simulator

Advanced physics-based battery simulation engine for generating realistic
training data for battery management systems. Implements electrochemical
models, thermal dynamics, and degradation mechanisms.

Features:
- Electrochemical impedance modeling
- Thermal dynamics simulation
- Calendar and cycle aging models
- Multi-chemistry battery support (Li-ion, LiFePO4, NiMH)
- Environmental condition effects
- State of Health (SoH) and State of Charge (SoC) modeling
- Realistic degradation patterns

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import time
import json
from enum import Enum
import math
from scipy import integrate, optimize
from scipy.interpolate import interp1d
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatteryChemistry(Enum):
    """Supported battery chemistries."""
    LITHIUM_ION = "lithium_ion"
    LIFEPO4 = "lifepo4"
    NIMH = "nimh"
    LEAD_ACID = "lead_acid"

class OperatingMode(Enum):
    """Battery operating modes."""
    DISCHARGE = "discharge"
    CHARGE = "charge"
    REST = "rest"
    PULSE = "pulse"

@dataclass
class BatteryParameters:
    """
    Physical and electrochemical parameters for battery simulation.
    
    Attributes:
        chemistry (BatteryChemistry): Battery chemistry type
        nominal_capacity_ah (float): Nominal capacity in Ah
        nominal_voltage (float): Nominal voltage in V
        max_voltage (float): Maximum voltage in V
        min_voltage (float): Minimum voltage in V
        internal_resistance (float): Internal resistance in Ohms
        thermal_mass (float): Thermal mass in J/K
        thermal_resistance (float): Thermal resistance in K/W
        diffusion_coefficient (float): Diffusion coefficient in m²/s
        electrode_area (float): Electrode area in m²
        separator_thickness (float): Separator thickness in m
        temperature_coefficient (float): Temperature coefficient in V/K
        self_discharge_rate (float): Self-discharge rate per day
        cycle_life (int): Expected cycle life
        calendar_life_years (float): Calendar life in years
    """
    chemistry: BatteryChemistry = BatteryChemistry.LITHIUM_ION
    nominal_capacity_ah: float = 50.0
    nominal_voltage: float = 3.7
    max_voltage: float = 4.2
    min_voltage: float = 2.5
    internal_resistance: float = 0.05
    thermal_mass: float = 1000.0
    thermal_resistance: float = 0.1
    diffusion_coefficient: float = 1e-14
    electrode_area: float = 0.1
    separator_thickness: float = 25e-6
    temperature_coefficient: float = -0.003
    self_discharge_rate: float = 0.05
    cycle_life: int = 3000
    calendar_life_years: float = 10.0

@dataclass
class EnvironmentalConditions:
    """
    Environmental conditions for battery simulation.
    
    Attributes:
        ambient_temperature (float): Ambient temperature in Celsius
        humidity (float): Relative humidity (0-1)
        pressure (float): Atmospheric pressure in Pa
        vibration_level (float): Vibration level (0-1)
        altitude (float): Altitude in meters
        thermal_gradient (float): Temperature gradient in K/m
    """
    ambient_temperature: float = 25.0
    humidity: float = 0.5
    pressure: float = 101325.0
    vibration_level: float = 0.0
    altitude: float = 0.0
    thermal_gradient: float = 0.0

@dataclass
class BatteryState:
    """
    Current state of the battery during simulation.
    
    Attributes:
        soc (float): State of Charge (0-1)
        soh (float): State of Health (0-1)
        voltage (float): Terminal voltage in V
        current (float): Current in A (positive for discharge)
        temperature (float): Battery temperature in Celsius
        internal_resistance (float): Current internal resistance in Ohms
        capacity_ah (float): Current capacity in Ah
        cycle_count (float): Cumulative cycle count
        age_days (float): Calendar age in days
        power (float): Power in W
        energy_throughput (float): Cumulative energy throughput in Wh
    """
    soc: float = 1.0
    soh: float = 1.0
    voltage: float = 4.2
    current: float = 0.0
    temperature: float = 25.0
    internal_resistance: float = 0.05
    capacity_ah: float = 50.0
    cycle_count: float = 0.0
    age_days: float = 0.0
    power: float = 0.0
    energy_throughput: float = 0.0

class ElectrochemicalModel:
    """
    Electrochemical model for battery voltage and current relationships.
    """
    
    def __init__(self, parameters: BatteryParameters):
        self.params = parameters
        self._initialize_ocv_curves()
        
    def _initialize_ocv_curves(self) -> None:
        """Initialize Open Circuit Voltage (OCV) curves for different chemistries."""
        if self.params.chemistry == BatteryChemistry.LITHIUM_ION:
            # Typical Li-ion OCV curve
            soc_points = np.linspace(0, 1, 21)
            ocv_points = np.array([
                2.5, 2.8, 3.0, 3.2, 3.4, 3.5, 3.6, 3.65, 3.7, 3.75,
                3.8, 3.85, 3.9, 3.95, 4.0, 4.05, 4.1, 4.15, 4.18, 4.2, 4.2
            ])
        elif self.params.chemistry == BatteryChemistry.LIFEPO4:
            # LiFePO4 OCV curve (flatter profile)
            soc_points = np.linspace(0, 1, 21)
            ocv_points = np.array([
                2.0, 2.5, 3.0, 3.1, 3.2, 3.25, 3.28, 3.3, 3.32, 3.34,
                3.36, 3.38, 3.4, 3.42, 3.44, 3.46, 3.48, 3.5, 3.55, 3.6, 3.65
            ])
        elif self.params.chemistry == BatteryChemistry.NIMH:
            # NiMH OCV curve
            soc_points = np.linspace(0, 1, 21)
            ocv_points = np.array([
                1.0, 1.1, 1.15, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3,
                1.32, 1.34, 1.36, 1.38, 1.4, 1.42, 1.44, 1.46, 1.48, 1.5, 1.5
            ])
        else:
            # Default to Li-ion
            soc_points = np.linspace(0, 1, 21)
            ocv_points = np.linspace(self.params.min_voltage, self.params.max_voltage, 21)
        
        self.ocv_interpolator = interp1d(soc_points, ocv_points, kind='cubic', 
                                        bounds_error=False, fill_value='extrapolate')
    
    def get_ocv(self, soc: float, temperature: float = 25.0) -> float:
        """
        Get Open Circuit Voltage for given SoC and temperature.
        
        Args:
            soc (float): State of Charge (0-1)
            temperature (float): Temperature in Celsius
            
        Returns:
            float: Open Circuit Voltage in V
        """
        base_ocv = float(self.ocv_interpolator(np.clip(soc, 0, 1)))
        
        # Temperature compensation
        temp_compensation = self.params.temperature_coefficient * (temperature - 25.0)
        
        return base_ocv + temp_compensation
    
    def get_internal_resistance(self, soc: float, temperature: float, 
                              soh: float, current: float) -> float:
        """
        Calculate internal resistance based on state and conditions.
        
        Args:
            soc (float): State of Charge
            temperature (float): Temperature in Celsius
            soh (float): State of Health
            current (float): Current in A
            
        Returns:
            float: Internal resistance in Ohms
        """
        base_resistance = self.params.internal_resistance
        
        # SoC dependency (higher resistance at low SoC)
        soc_factor = 1.0 + 0.5 * np.exp(-10 * soc)
        
        # Temperature dependency (Arrhenius relationship)
        temp_kelvin = temperature + 273.15
        temp_factor = np.exp(3000 * (1/temp_kelvin - 1/298.15))
        
        # SoH dependency (resistance increases with aging)
        soh_factor = 1.0 / soh
        
        # Current dependency (slight increase at high currents)
        current_factor = 1.0 + 0.01 * abs(current)
        
        return base_resistance * soc_factor * temp_factor * soh_factor * current_factor
    
    def calculate_voltage(self, state: BatteryState, current: float) -> float:
        """
        Calculate terminal voltage given current.
        
        Args:
            state (BatteryState): Current battery state
            current (float): Applied current in A
            
        Returns:
            float: Terminal voltage in V
        """
        ocv = self.get_ocv(state.soc, state.temperature)
        resistance = self.get_internal_resistance(
            state.soc, state.temperature, state.soh, current
        )
        
        # Terminal voltage with IR drop
        voltage = ocv - current * resistance
        
        # Add concentration overpotential for high currents
        if abs(current) > 0.1 * self.params.nominal_capacity_ah:
            concentration_overpotential = 0.01 * np.log(1 + abs(current) / 
                                                       self.params.nominal_capacity_ah)
            voltage -= np.sign(current) * concentration_overpotential
        
        return voltage

class ThermalModel:
    """
    Thermal model for battery temperature dynamics.
    """
    
    def __init__(self, parameters: BatteryParameters):
        self.params = parameters
        
    def calculate_heat_generation(self, current: float, voltage: float, 
                                 ocv: float) -> float:
        """
        Calculate heat generation rate.
        
        Args:
            current (float): Current in A
            voltage (float): Terminal voltage in V
            ocv (float): Open circuit voltage in V
            
        Returns:
            float: Heat generation rate in W
        """
        # Joule heating (I²R losses)
        joule_heat = current**2 * self.params.internal_resistance
        
        # Reversible heat (entropy change)
        reversible_heat = current * (ocv - voltage - current * self.params.internal_resistance)
        
        return joule_heat + reversible_heat
    
    def update_temperature(self, current_temp: float, heat_generation: float,
                          ambient_temp: float, dt: float) -> float:
        """
        Update battery temperature using thermal model.
        
        Args:
            current_temp (float): Current temperature in Celsius
            heat_generation (float): Heat generation rate in W
            ambient_temp (float): Ambient temperature in Celsius
            dt (float): Time step in seconds
            
        Returns:
            float: New temperature in Celsius
        """
        # Heat transfer to environment
        heat_loss = (current_temp - ambient_temp) / self.params.thermal_resistance
        
        # Net heat rate
        net_heat_rate = heat_generation - heat_loss
        
        # Temperature change
        temp_change = net_heat_rate * dt / self.params.thermal_mass
        
        return current_temp + temp_change

class DegradationModel:
    """
    Battery degradation model including calendar and cycle aging.
    """
    
    def __init__(self, parameters: BatteryParameters):
        self.params = parameters
        
    def calculate_calendar_aging(self, temperature: float, soc: float, 
                                dt_days: float) -> float:
        """
        Calculate capacity loss due to calendar aging.
        
        Args:
            temperature (float): Temperature in Celsius
            soc (float): State of Charge
            dt_days (float): Time step in days
            
        Returns:
            float: Capacity loss fraction
        """
        # Arrhenius temperature dependency
        temp_kelvin = temperature + 273.15
        arrhenius_factor = np.exp(-6000 * (1/temp_kelvin - 1/298.15))
        
        # SoC stress factor (higher SoC accelerates aging)
        soc_stress = 1.0 + 2.0 * soc
        
        # Calendar aging rate (fraction per day)
        aging_rate = (1.0 / (self.params.calendar_life_years * 365)) * arrhenius_factor * soc_stress
        
        return aging_rate * dt_days
    
    def calculate_cycle_aging(self, dod: float, temperature: float, 
                             c_rate: float) -> float:
        """
        Calculate capacity loss due to cycle aging.
        
        Args:
            dod (float): Depth of Discharge (0-1)
            temperature (float): Temperature in Celsius
            c_rate (float): C-rate (current/capacity)
            
        Returns:
            float: Capacity loss per cycle
        """
        # Base cycle aging (Wöhler curve)
        base_aging = 1.0 / self.params.cycle_life
        
        # DoD stress factor
        dod_stress = (dod / 0.8) ** 2  # Normalized to 80% DoD
        
        # Temperature stress
        temp_kelvin = temperature + 273.15
        temp_stress = np.exp(2000 * (1/temp_kelvin - 1/298.15))
        
        # C-rate stress
        crate_stress = 1.0 + 0.5 * abs(c_rate)
        
        return base_aging * dod_stress * temp_stress * crate_stress
    
    def update_soh(self, current_soh: float, calendar_loss: float, 
                   cycle_loss: float) -> float:
        """
        Update State of Health based on aging mechanisms.
        
        Args:
            current_soh (float): Current SoH
            calendar_loss (float): Calendar aging loss
            cycle_loss (float): Cycle aging loss
            
        Returns:
            float: Updated SoH
        """
        total_loss = calendar_loss + cycle_loss
        new_soh = current_soh - total_loss
        
        return max(0.5, new_soh)  # Minimum SoH of 50%

class BatteryPhysicsSimulator:
    """
    Main battery physics simulator integrating all models.
    """
    
    def __init__(self, parameters: BatteryParameters):
        self.params = parameters
        self.electrochemical_model = ElectrochemicalModel(parameters)
        self.thermal_model = ThermalModel(parameters)
        self.degradation_model = DegradationModel(parameters)
        
        # Initialize state
        self.state = BatteryState(
            capacity_ah=parameters.nominal_capacity_ah,
            voltage=parameters.nominal_voltage,
            internal_resistance=parameters.internal_resistance
        )
        
        # Simulation history
        self.history = []
        
    def reset(self, initial_soc: float = 1.0, initial_temperature: float = 25.0) -> None:
        """Reset simulator to initial conditions."""
        self.state = BatteryState(
            soc=initial_soc,
            soh=1.0,
            voltage=self.electrochemical_model.get_ocv(initial_soc, initial_temperature),
            temperature=initial_temperature,
            capacity_ah=self.params.nominal_capacity_ah,
            internal_resistance=self.params.internal_resistance
        )
        self.history = []
    
    def step(self, current: float, dt: float, 
             environmental: EnvironmentalConditions) -> BatteryState:
        """
        Perform one simulation step.
        
        Args:
            current (float): Applied current in A (positive for discharge)
            dt (float): Time step in seconds
            environmental (EnvironmentalConditions): Environmental conditions
            
        Returns:
            BatteryState: Updated battery state
        """
        # Calculate OCV
        ocv = self.electrochemical_model.get_ocv(self.state.soc, self.state.temperature)
        
        # Calculate terminal voltage
        voltage = self.electrochemical_model.calculate_voltage(self.state, current)
        
        # Update SoC (Coulomb counting)
        if self.state.capacity_ah > 0:
            dsoc = -current * dt / (3600 * self.state.capacity_ah)
            new_soc = np.clip(self.state.soc + dsoc, 0.0, 1.0)
        else:
            new_soc = self.state.soc
        
        # Calculate heat generation
        heat_gen = self.thermal_model.calculate_heat_generation(current, voltage, ocv)
        
        # Update temperature
        new_temperature = self.thermal_model.update_temperature(
            self.state.temperature, heat_gen, environmental.ambient_temperature, dt
        )
        
        # Calculate degradation
        dt_days = dt / (24 * 3600)
        calendar_loss = self.degradation_model.calculate_calendar_aging(
            new_temperature, new_soc, dt_days
        )
        
        # Cycle aging (if significant current)
        if abs(current) > 0.01 * self.params.nominal_capacity_ah:
            dod = abs(self.state.soc - new_soc)
            c_rate = abs(current) / self.state.capacity_ah
            cycle_loss = self.degradation_model.calculate_cycle_aging(
                dod, new_temperature, c_rate
            ) * dod  # Scale by actual DoD
        else:
            cycle_loss = 0.0
        
        # Update SoH
        new_soh = self.degradation_model.update_soh(
            self.state.soh, calendar_loss, cycle_loss
        )
        
        # Update capacity based on SoH
        new_capacity = self.params.nominal_capacity_ah * new_soh
        
        # Update internal resistance
        new_resistance = self.electrochemical_model.get_internal_resistance(
            new_soc, new_temperature, new_soh, current
        )
        
        # Calculate power and energy
        power = voltage * current
        energy_increment = abs(power) * dt / 3600  # Wh
        
        # Update cycle count (simplified)
        if abs(current) > 0.01 * self.params.nominal_capacity_ah:
            cycle_increment = abs(dsoc) / 2  # Half cycle per SoC change
        else:
            cycle_increment = 0.0
        
        # Update state
        self.state = BatteryState(
            soc=new_soc,
            soh=new_soh,
            voltage=voltage,
            current=current,
            temperature=new_temperature,
            internal_resistance=new_resistance,
            capacity_ah=new_capacity,
            cycle_count=self.state.cycle_count + cycle_increment,
            age_days=self.state.age_days + dt_days,
            power=power,
            energy_throughput=self.state.energy_throughput + energy_increment
        )
        
        # Store history
        self.history.append({
            'timestamp': len(self.history) * dt,
            'soc': self.state.soc,
            'soh': self.state.soh,
            'voltage': self.state.voltage,
            'current': self.state.current,
            'temperature': self.state.temperature,
            'internal_resistance': self.state.internal_resistance,
            'power': self.state.power,
            'cycle_count': self.state.cycle_count,
            'age_days': self.state.age_days
        })
        
        return self.state
    
    def simulate_profile(self, current_profile: np.ndarray, dt: float,
                        environmental: EnvironmentalConditions = None) -> pd.DataFrame:
        """
        Simulate battery response to a current profile.
        
        Args:
            current_profile (np.ndarray): Current profile in A
            dt (float): Time step in seconds
            environmental (EnvironmentalConditions, optional): Environmental conditions
            
        Returns:
            pd.DataFrame: Simulation results
        """
        if environmental is None:
            environmental = EnvironmentalConditions()
        
        # Reset history
        self.history = []
        
        # Simulate each time step
        for current in current_profile:
            self.step(current, dt, environmental)
        
        # Convert to DataFrame
        return pd.DataFrame(self.history)
    
    def generate_drive_cycle(self, cycle_type: str = "urban", 
                           duration_hours: float = 1.0,
                           dt: float = 1.0) -> np.ndarray:
        """
        Generate realistic drive cycle current profiles.
        
        Args:
            cycle_type (str): Type of drive cycle
            duration_hours (float): Duration in hours
            dt (float): Time step in seconds
            
        Returns:
            np.ndarray: Current profile in A
        """
        num_points = int(duration_hours * 3600 / dt)
        time_array = np.linspace(0, duration_hours * 3600, num_points)
        
        if cycle_type == "urban":
            # Urban drive cycle with frequent stops
            base_current = 0.2 * self.params.nominal_capacity_ah
            current_profile = base_current * (
                0.5 * np.sin(2 * np.pi * time_array / 300) +  # 5-minute cycle
                0.3 * np.sin(2 * np.pi * time_array / 60) +   # 1-minute variations
                0.2 * np.random.normal(0, 1, num_points)      # Random variations
            )
            
        elif cycle_type == "highway":
            # Highway drive cycle with steady load
            base_current = 0.3 * self.params.nominal_capacity_ah
            current_profile = base_current * (
                1.0 + 0.1 * np.sin(2 * np.pi * time_array / 1800) +  # 30-minute cycle
                0.05 * np.random.normal(0, 1, num_points)            # Small variations
            )
            
        elif cycle_type == "mixed":
            # Mixed drive cycle
            base_current = 0.25 * self.params.nominal_capacity_ah
            current_profile = base_current * (
                0.7 * np.sin(2 * np.pi * time_array / 600) +   # 10-minute cycle
                0.2 * np.sin(2 * np.pi * time_array / 120) +   # 2-minute cycle
                0.1 * np.random.normal(0, 1, num_points)       # Random variations
            )
            
        else:
            # Constant current
            current_profile = np.full(num_points, 0.1 * self.params.nominal_capacity_ah)
        
        # Ensure non-negative current (discharge only)
        current_profile = np.maximum(current_profile, 0)
        
        # Add charging periods (negative current)
        if cycle_type in ["urban", "mixed"]:
            # Add regenerative braking
            regen_mask = np.random.random(num_points) < 0.1  # 10% of time
            current_profile[regen_mask] *= -0.5
        
        return current_profile
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get current state as dictionary."""
        return {
            'soc': self.state.soc,
            'soh': self.state.soh,
            'voltage': self.state.voltage,
            'current': self.state.current,
            'temperature': self.state.temperature,
            'internal_resistance': self.state.internal_resistance,
            'capacity_ah': self.state.capacity_ah,
            'cycle_count': self.state.cycle_count,
            'age_days': self.state.age_days,
            'power': self.state.power,
            'energy_throughput': self.state.energy_throughput
        }

# Factory functions
def create_lithium_ion_battery(capacity_ah: float = 50.0) -> BatteryPhysicsSimulator:
    """Create a lithium-ion battery simulator."""
    params = BatteryParameters(
        chemistry=BatteryChemistry.LITHIUM_ION,
        nominal_capacity_ah=capacity_ah,
        nominal_voltage=3.7,
        max_voltage=4.2,
        min_voltage=2.5,
        internal_resistance=0.05,
        cycle_life=3000
    )
    return BatteryPhysicsSimulator(params)

def create_lifepo4_battery(capacity_ah: float = 100.0) -> BatteryPhysicsSimulator:
    """Create a LiFePO4 battery simulator."""
    params = BatteryParameters(
        chemistry=BatteryChemistry.LIFEPO4,
        nominal_capacity_ah=capacity_ah,
        nominal_voltage=3.3,
        max_voltage=3.65,
        min_voltage=2.0,
        internal_resistance=0.03,
        cycle_life=5000
    )
    return BatteryPhysicsSimulator(params)

def create_nimh_battery(capacity_ah: float = 20.0) -> BatteryPhysicsSimulator:
    """Create a NiMH battery simulator."""
    params = BatteryParameters(
        chemistry=BatteryChemistry.NIMH,
        nominal_capacity_ah=capacity_ah,
        nominal_voltage=1.2,
        max_voltage=1.5,
        min_voltage=1.0,
        internal_resistance=0.1,
        cycle_life=1000
    )
    return BatteryPhysicsSimulator(params)

def simulate_battery_fleet(num_batteries: int = 100, 
                          duration_hours: float = 24.0,
                          chemistry: str = "lithium_ion") -> pd.DataFrame:
    """
    Simulate a fleet of batteries with varied parameters.
    
    Args:
        num_batteries (int): Number of batteries to simulate
        duration_hours (float): Simulation duration in hours
        chemistry (str): Battery chemistry type
        
    Returns:
        pd.DataFrame: Fleet simulation results
    """
    results = []
    
    for i in range(num_batteries):
        # Create battery with random variations
        if chemistry == "lithium_ion":
            capacity = np.random.normal(50.0, 5.0)
            simulator = create_lithium_ion_battery(max(10.0, capacity))
        elif chemistry == "lifepo4":
            capacity = np.random.normal(100.0, 10.0)
            simulator = create_lifepo4_battery(max(20.0, capacity))
        else:
            capacity = np.random.normal(20.0, 2.0)
            simulator = create_nimh_battery(max(5.0, capacity))
        
        # Random initial conditions
        initial_soc = np.random.uniform(0.2, 1.0)
        initial_temp = np.random.normal(25.0, 5.0)
        simulator.reset(initial_soc, initial_temp)
        
        # Generate drive cycle
        cycle_type = np.random.choice(["urban", "highway", "mixed"])
        current_profile = simulator.generate_drive_cycle(cycle_type, duration_hours)
        
        # Simulate
        env_conditions = EnvironmentalConditions(
            ambient_temperature=np.random.normal(25.0, 10.0),
            humidity=np.random.uniform(0.3, 0.8)
        )
        
        df = simulator.simulate_profile(current_profile, 1.0, env_conditions)
        df['battery_id'] = i
        df['chemistry'] = chemistry
        df['cycle_type'] = cycle_type
        
        results.append(df)
    
    return pd.concat(results, ignore_index=True)
