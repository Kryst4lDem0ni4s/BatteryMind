"""
BatteryMind - Training Data Scenario Builder

Advanced scenario generation framework for creating realistic battery management
training scenarios with physics-based constraints and domain expertise.

This module generates comprehensive training scenarios that cover diverse
battery operating conditions, usage patterns, environmental factors, and
failure modes to ensure robust model training.

Features:
- Physics-based scenario generation with electrochemical constraints
- Multi-dimensional parameter space exploration
- Realistic degradation pattern simulation
- Environmental condition modeling
- Fleet usage pattern generation
- Anomaly and failure scenario creation
- Temporal sequence generation for time-series models

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import json
import random
from enum import Enum
from pathlib import Path
import itertools
from scipy import stats
from scipy.interpolate import interp1d
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatteryChemistry(Enum):
    """Supported battery chemistry types."""
    LITHIUM_ION = "lithium_ion"
    LITHIUM_IRON_PHOSPHATE = "lifepo4"
    NICKEL_METAL_HYDRIDE = "nimh"
    LITHIUM_POLYMER = "lipo"
    SOLID_STATE = "solid_state"

class ApplicationType(Enum):
    """Battery application types."""
    ELECTRIC_VEHICLE = "electric_vehicle"
    ENERGY_STORAGE = "energy_storage"
    CONSUMER_ELECTRONICS = "consumer_electronics"
    INDUSTRIAL = "industrial"
    AEROSPACE = "aerospace"

class OperatingCondition(Enum):
    """Operating condition categories."""
    NORMAL = "normal"
    EXTREME_COLD = "extreme_cold"
    EXTREME_HOT = "extreme_hot"
    HIGH_POWER = "high_power"
    DEEP_DISCHARGE = "deep_discharge"
    FAST_CHARGING = "fast_charging"

@dataclass
class ScenarioParameters:
    """
    Parameters defining a battery scenario.
    
    Attributes:
        chemistry (BatteryChemistry): Battery chemistry type
        application (ApplicationType): Application domain
        capacity_ah (float): Battery capacity in Ah
        voltage_nominal (float): Nominal voltage in V
        temperature_range (Tuple[float, float]): Operating temperature range in °C
        soc_range (Tuple[float, float]): State of charge range (0-1)
        c_rate_range (Tuple[float, float]): C-rate range for charging/discharging
        cycle_life_target (int): Target cycle life
        calendar_life_years (float): Calendar life in years
        operating_conditions (List[OperatingCondition]): Operating conditions
        degradation_mechanisms (List[str]): Active degradation mechanisms
        environmental_factors (Dict[str, Any]): Environmental factor ranges
        usage_patterns (List[str]): Usage pattern types
    """
    chemistry: BatteryChemistry = BatteryChemistry.LITHIUM_ION
    application: ApplicationType = ApplicationType.ELECTRIC_VEHICLE
    capacity_ah: float = 50.0
    voltage_nominal: float = 3.7
    temperature_range: Tuple[float, float] = (-20.0, 60.0)
    soc_range: Tuple[float, float] = (0.1, 0.9)
    c_rate_range: Tuple[float, float] = (0.1, 3.0)
    cycle_life_target: int = 3000
    calendar_life_years: float = 10.0
    operating_conditions: List[OperatingCondition] = field(default_factory=list)
    degradation_mechanisms: List[str] = field(default_factory=list)
    environmental_factors: Dict[str, Any] = field(default_factory=dict)
    usage_patterns: List[str] = field(default_factory=list)

class BatteryScenarioBuilder:
    """
    Builds comprehensive battery scenarios for training data generation.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Load battery chemistry properties
        self.chemistry_properties = self._load_chemistry_properties()
        
        # Load degradation mechanisms
        self.degradation_mechanisms = self._load_degradation_mechanisms()
        
        # Load usage patterns
        self.usage_patterns = self._load_usage_patterns()
        
        logger.info("BatteryScenarioBuilder initialized")
    
    def _load_chemistry_properties(self) -> Dict[str, Dict[str, Any]]:
        """Load battery chemistry properties and constraints."""
        return {
            BatteryChemistry.LITHIUM_ION.value: {
                "voltage_range": (2.5, 4.2),
                "energy_density_wh_kg": (150, 250),
                "power_density_w_kg": (300, 1500),
                "cycle_life_range": (500, 5000),
                "temperature_range": (-20, 60),
                "c_rate_max": 5.0,
                "self_discharge_rate": 0.02,  # %/month
                "degradation_factors": ["sei_growth", "li_plating", "particle_cracking"]
            },
            BatteryChemistry.LITHIUM_IRON_PHOSPHATE.value: {
                "voltage_range": (2.0, 3.65),
                "energy_density_wh_kg": (90, 160),
                "power_density_w_kg": (400, 2000),
                "cycle_life_range": (2000, 8000),
                "temperature_range": (-20, 70),
                "c_rate_max": 10.0,
                "self_discharge_rate": 0.01,
                "degradation_factors": ["iron_dissolution", "electrolyte_decomposition"]
            },
            BatteryChemistry.NICKEL_METAL_HYDRIDE.value: {
                "voltage_range": (0.9, 1.4),
                "energy_density_wh_kg": (60, 120),
                "power_density_w_kg": (250, 1000),
                "cycle_life_range": (300, 1500),
                "temperature_range": (-40, 60),
                "c_rate_max": 3.0,
                "self_discharge_rate": 0.15,
                "degradation_factors": ["memory_effect", "corrosion", "hydrogen_evolution"]
            }
        }
    
    def _load_degradation_mechanisms(self) -> Dict[str, Dict[str, Any]]:
        """Load battery degradation mechanisms and their characteristics."""
        return {
            "sei_growth": {
                "description": "Solid Electrolyte Interphase growth",
                "temperature_dependency": "arrhenius",
                "soc_dependency": "high_soc_accelerated",
                "time_dependency": "sqrt_time",
                "capacity_impact": 0.8,
                "resistance_impact": 1.2
            },
            "li_plating": {
                "description": "Lithium plating during charging",
                "temperature_dependency": "low_temp_accelerated",
                "c_rate_dependency": "high_rate_accelerated",
                "soc_dependency": "high_soc_accelerated",
                "capacity_impact": 0.9,
                "resistance_impact": 1.1
            },
            "particle_cracking": {
                "description": "Active material particle cracking",
                "cycle_dependency": "cycle_count",
                "temperature_dependency": "thermal_cycling",
                "capacity_impact": 0.85,
                "resistance_impact": 1.15
            },
            "electrolyte_decomposition": {
                "description": "Electrolyte decomposition",
                "temperature_dependency": "high_temp_accelerated",
                "time_dependency": "linear_time",
                "capacity_impact": 0.9,
                "resistance_impact": 1.1
            }
        }
    
    def _load_usage_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load typical usage patterns for different applications."""
        return {
            "daily_commute": {
                "description": "Daily commuting pattern",
                "cycles_per_day": 1,
                "depth_of_discharge": 0.3,
                "charging_frequency": "daily",
                "c_rate_charge": 0.5,
                "c_rate_discharge": 1.0,
                "temperature_variation": 20
            },
            "long_distance": {
                "description": "Long distance travel",
                "cycles_per_day": 0.2,
                "depth_of_discharge": 0.8,
                "charging_frequency": "fast_charge",
                "c_rate_charge": 2.0,
                "c_rate_discharge": 1.5,
                "temperature_variation": 30
            },
            "grid_storage": {
                "description": "Grid energy storage",
                "cycles_per_day": 1.5,
                "depth_of_discharge": 0.9,
                "charging_frequency": "continuous",
                "c_rate_charge": 0.3,
                "c_rate_discharge": 0.3,
                "temperature_variation": 10
            },
            "peak_shaving": {
                "description": "Peak shaving application",
                "cycles_per_day": 2,
                "depth_of_discharge": 0.4,
                "charging_frequency": "scheduled",
                "c_rate_charge": 1.0,
                "c_rate_discharge": 2.0,
                "temperature_variation": 15
            }
        }
    
    def generate_base_scenarios(self, num_scenarios: int = 100) -> List[ScenarioParameters]:
        """
        Generate base battery scenarios covering diverse conditions.
        
        Args:
            num_scenarios (int): Number of scenarios to generate
            
        Returns:
            List[ScenarioParameters]: List of generated scenarios
        """
        scenarios = []
        
        # Define parameter ranges for systematic exploration
        chemistry_types = list(BatteryChemistry)
        application_types = list(ApplicationType)
        
        for i in range(num_scenarios):
            # Select chemistry and application
            chemistry = np.random.choice(chemistry_types)
            application = np.random.choice(application_types)
            
            # Get chemistry properties
            chem_props = self.chemistry_properties[chemistry.value]
            
            # Generate scenario parameters
            scenario = ScenarioParameters(
                chemistry=chemistry,
                application=application,
                capacity_ah=self._generate_capacity(application),
                voltage_nominal=self._generate_voltage(chemistry),
                temperature_range=self._generate_temperature_range(application),
                soc_range=self._generate_soc_range(application),
                c_rate_range=self._generate_c_rate_range(chemistry, application),
                cycle_life_target=self._generate_cycle_life(chemistry, application),
                calendar_life_years=self._generate_calendar_life(application),
                operating_conditions=self._select_operating_conditions(application),
                degradation_mechanisms=self._select_degradation_mechanisms(chemistry),
                environmental_factors=self._generate_environmental_factors(application),
                usage_patterns=self._select_usage_patterns(application)
            )
            
            scenarios.append(scenario)
        
        logger.info(f"Generated {len(scenarios)} base scenarios")
        return scenarios
    
    def _generate_capacity(self, application: ApplicationType) -> float:
        """Generate battery capacity based on application."""
        capacity_ranges = {
            ApplicationType.ELECTRIC_VEHICLE: (40, 100),
            ApplicationType.ENERGY_STORAGE: (100, 1000),
            ApplicationType.CONSUMER_ELECTRONICS: (0.5, 5),
            ApplicationType.INDUSTRIAL: (20, 200),
            ApplicationType.AEROSPACE: (10, 50)
        }
        
        min_cap, max_cap = capacity_ranges[application]
        return np.random.uniform(min_cap, max_cap)
    
    def _generate_voltage(self, chemistry: BatteryChemistry) -> float:
        """Generate nominal voltage based on chemistry."""
        voltage_ranges = {
            BatteryChemistry.LITHIUM_ION: (3.6, 3.8),
            BatteryChemistry.LITHIUM_IRON_PHOSPHATE: (3.2, 3.3),
            BatteryChemistry.NICKEL_METAL_HYDRIDE: (1.2, 1.25),
            BatteryChemistry.LITHIUM_POLYMER: (3.7, 3.8),
            BatteryChemistry.SOLID_STATE: (3.8, 4.0)
        }
        
        min_v, max_v = voltage_ranges[chemistry]
        return np.random.uniform(min_v, max_v)
    
    def _generate_temperature_range(self, application: ApplicationType) -> Tuple[float, float]:
        """Generate operating temperature range based on application."""
        temp_ranges = {
            ApplicationType.ELECTRIC_VEHICLE: (-30, 60),
            ApplicationType.ENERGY_STORAGE: (-10, 50),
            ApplicationType.CONSUMER_ELECTRONICS: (0, 40),
            ApplicationType.INDUSTRIAL: (-20, 70),
            ApplicationType.AEROSPACE: (-40, 80)
        }
        
        base_min, base_max = temp_ranges[application]
        
        # Add some variation
        temp_min = base_min + np.random.uniform(-5, 5)
        temp_max = base_max + np.random.uniform(-5, 5)
        
        return (temp_min, temp_max)
    
    def _generate_soc_range(self, application: ApplicationType) -> Tuple[float, float]:
        """Generate SOC operating range based on application."""
        soc_ranges = {
            ApplicationType.ELECTRIC_VEHICLE: (0.1, 0.9),
            ApplicationType.ENERGY_STORAGE: (0.05, 0.95),
            ApplicationType.CONSUMER_ELECTRONICS: (0.15, 0.85),
            ApplicationType.INDUSTRIAL: (0.1, 0.9),
            ApplicationType.AEROSPACE: (0.2, 0.8)
        }
        
        base_min, base_max = soc_ranges[application]
        
        # Add some variation
        soc_min = max(0.05, base_min + np.random.uniform(-0.05, 0.05))
        soc_max = min(0.95, base_max + np.random.uniform(-0.05, 0.05))
        
        return (soc_min, soc_max)
    
    def _generate_c_rate_range(self, chemistry: BatteryChemistry, 
                             application: ApplicationType) -> Tuple[float, float]:
        """Generate C-rate range based on chemistry and application."""
        chem_props = self.chemistry_properties[chemistry.value]
        max_c_rate = chem_props["c_rate_max"]
        
        app_factors = {
            ApplicationType.ELECTRIC_VEHICLE: 0.8,
            ApplicationType.ENERGY_STORAGE: 0.3,
            ApplicationType.CONSUMER_ELECTRONICS: 0.5,
            ApplicationType.INDUSTRIAL: 0.6,
            ApplicationType.AEROSPACE: 0.4
        }
        
        app_max = max_c_rate * app_factors[application]
        c_rate_min = 0.1
        c_rate_max = app_max + np.random.uniform(-0.2, 0.2)
        
        return (c_rate_min, max(c_rate_min + 0.1, c_rate_max))
    
    def _generate_cycle_life(self, chemistry: BatteryChemistry, 
                           application: ApplicationType) -> int:
        """Generate target cycle life."""
        chem_props = self.chemistry_properties[chemistry.value]
        min_cycles, max_cycles = chem_props["cycle_life_range"]
        
        app_factors = {
            ApplicationType.ELECTRIC_VEHICLE: 0.8,
            ApplicationType.ENERGY_STORAGE: 1.2,
            ApplicationType.CONSUMER_ELECTRONICS: 0.6,
            ApplicationType.INDUSTRIAL: 1.0,
            ApplicationType.AEROSPACE: 0.7
        }
        
        target_cycles = int(np.random.uniform(min_cycles, max_cycles) * app_factors[application])
        return max(500, target_cycles)
    
    def _generate_calendar_life(self, application: ApplicationType) -> float:
        """Generate calendar life based on application."""
        calendar_ranges = {
            ApplicationType.ELECTRIC_VEHICLE: (8, 15),
            ApplicationType.ENERGY_STORAGE: (15, 25),
            ApplicationType.CONSUMER_ELECTRONICS: (3, 8),
            ApplicationType.INDUSTRIAL: (10, 20),
            ApplicationType.AEROSPACE: (5, 15)
        }
        
        min_years, max_years = calendar_ranges[application]
        return np.random.uniform(min_years, max_years)
    
    def _select_operating_conditions(self, application: ApplicationType) -> List[OperatingCondition]:
        """Select relevant operating conditions for application."""
        condition_probs = {
            ApplicationType.ELECTRIC_VEHICLE: {
                OperatingCondition.NORMAL: 0.6,
                OperatingCondition.EXTREME_COLD: 0.1,
                OperatingCondition.EXTREME_HOT: 0.1,
                OperatingCondition.HIGH_POWER: 0.1,
                OperatingCondition.FAST_CHARGING: 0.1
            },
            ApplicationType.ENERGY_STORAGE: {
                OperatingCondition.NORMAL: 0.7,
                OperatingCondition.EXTREME_HOT: 0.1,
                OperatingCondition.DEEP_DISCHARGE: 0.2
            },
            ApplicationType.CONSUMER_ELECTRONICS: {
                OperatingCondition.NORMAL: 0.8,
                OperatingCondition.FAST_CHARGING: 0.2
            }
        }
        
        probs = condition_probs.get(application, {OperatingCondition.NORMAL: 1.0})
        
        selected_conditions = []
        for condition, prob in probs.items():
            if np.random.random() < prob:
                selected_conditions.append(condition)
        
        return selected_conditions
    
    def _select_degradation_mechanisms(self, chemistry: BatteryChemistry) -> List[str]:
        """Select relevant degradation mechanisms for chemistry."""
        chem_props = self.chemistry_properties[chemistry.value]
        available_mechanisms = chem_props["degradation_factors"]
        
        # Select 2-4 mechanisms randomly
        num_mechanisms = np.random.randint(2, min(5, len(available_mechanisms) + 1))
        selected = np.random.choice(available_mechanisms, num_mechanisms, replace=False)
        
        return selected.tolist()
    
    def _generate_environmental_factors(self, application: ApplicationType) -> Dict[str, Any]:
        """Generate environmental factors based on application."""
        base_factors = {
            "humidity_range": (20, 80),  # %
            "pressure_range": (0.8, 1.2),  # atm
            "vibration_level": np.random.uniform(0.1, 2.0),  # g
            "thermal_cycling": np.random.choice([True, False]),
            "corrosive_environment": np.random.choice([True, False])
        }
        
        # Modify based on application
        if application == ApplicationType.AEROSPACE:
            base_factors["pressure_range"] = (0.1, 1.0)
            base_factors["vibration_level"] *= 2
        elif application == ApplicationType.INDUSTRIAL:
            base_factors["humidity_range"] = (10, 90)
            base_factors["corrosive_environment"] = True
        
        return base_factors
    
    def _select_usage_patterns(self, application: ApplicationType) -> List[str]:
        """Select usage patterns for application."""
        pattern_mapping = {
            ApplicationType.ELECTRIC_VEHICLE: ["daily_commute", "long_distance"],
            ApplicationType.ENERGY_STORAGE: ["grid_storage", "peak_shaving"],
            ApplicationType.CONSUMER_ELECTRONICS: ["daily_commute"],
            ApplicationType.INDUSTRIAL: ["grid_storage"],
            ApplicationType.AEROSPACE: ["daily_commute"]
        }
        
        available_patterns = pattern_mapping.get(application, ["daily_commute"])
        num_patterns = np.random.randint(1, len(available_patterns) + 1)
        
        return np.random.choice(available_patterns, num_patterns, replace=False).tolist()
    
    def generate_extreme_scenarios(self, base_scenarios: List[ScenarioParameters],
                                 num_extreme: int = 50) -> List[ScenarioParameters]:
        """
        Generate extreme scenarios for stress testing and edge case coverage.
        
        Args:
            base_scenarios (List[ScenarioParameters]): Base scenarios to modify
            num_extreme (int): Number of extreme scenarios to generate
            
        Returns:
            List[ScenarioParameters]: List of extreme scenarios
        """
        extreme_scenarios = []
        
        for i in range(num_extreme):
            # Select a base scenario to modify
            base_scenario = np.random.choice(base_scenarios)
            extreme_scenario = self._create_extreme_scenario(base_scenario)
            extreme_scenarios.append(extreme_scenario)
        
        logger.info(f"Generated {len(extreme_scenarios)} extreme scenarios")
        return extreme_scenarios
    
    def _create_extreme_scenario(self, base_scenario: ScenarioParameters) -> ScenarioParameters:
        """Create an extreme scenario by modifying a base scenario."""
        import copy
        extreme_scenario = copy.deepcopy(base_scenario)
        
        # Randomly select extreme modifications
        modifications = [
            self._apply_extreme_temperature,
            self._apply_extreme_c_rates,
            self._apply_extreme_soc_range,
            self._apply_extreme_degradation,
            self._apply_extreme_environment
        ]
        
        # Apply 1-3 random modifications
        num_mods = np.random.randint(1, 4)
        selected_mods = np.random.choice(modifications, num_mods, replace=False)
        
        for mod_func in selected_mods:
            extreme_scenario = mod_func(extreme_scenario)
        
        return extreme_scenario
    
    def _apply_extreme_temperature(self, scenario: ScenarioParameters) -> ScenarioParameters:
        """Apply extreme temperature conditions."""
        temp_min, temp_max = scenario.temperature_range
        
        # Extend temperature range to extremes
        extreme_min = temp_min - np.random.uniform(10, 30)
        extreme_max = temp_max + np.random.uniform(10, 30)
        
        scenario.temperature_range = (extreme_min, extreme_max)
        scenario.operating_conditions.extend([OperatingCondition.EXTREME_COLD, 
                                            OperatingCondition.EXTREME_HOT])
        
        return scenario
    
    def _apply_extreme_c_rates(self, scenario: ScenarioParameters) -> ScenarioParameters:
        """Apply extreme C-rates."""
        c_min, c_max = scenario.c_rate_range
        
        # Increase maximum C-rate significantly
        extreme_max = c_max * np.random.uniform(2, 5)
        scenario.c_rate_range = (c_min, extreme_max)
        scenario.operating_conditions.append(OperatingCondition.HIGH_POWER)
        
        return scenario
    
    def _apply_extreme_soc_range(self, scenario: ScenarioParameters) -> ScenarioParameters:
        """Apply extreme SOC ranges."""
        # Use full SOC range with deep discharge
        scenario.soc_range = (0.05, 0.95)
        scenario.operating_conditions.append(OperatingCondition.DEEP_DISCHARGE)
        
        return scenario
    
    def _apply_extreme_degradation(self, scenario: ScenarioParameters) -> ScenarioParameters:
        """Apply accelerated degradation conditions."""
        # Add all possible degradation mechanisms
        all_mechanisms = list(self.degradation_mechanisms.keys())
        scenario.degradation_mechanisms = all_mechanisms
        
        # Reduce cycle life target
        scenario.cycle_life_target = int(scenario.cycle_life_target * 0.5)
        
        return scenario
    
    def _apply_extreme_environment(self, scenario: ScenarioParameters) -> ScenarioParameters:
        """Apply extreme environmental conditions."""
        scenario.environmental_factors.update({
            "humidity_range": (5, 95),
            "pressure_range": (0.1, 2.0),
            "vibration_level": 5.0,
            "thermal_cycling": True,
            "corrosive_environment": True,
            "radiation_exposure": True,
            "salt_spray": True
        })
        
        return scenario
    
    def generate_failure_scenarios(self, base_scenarios: List[ScenarioParameters],
                                 num_failures: int = 30) -> List[ScenarioParameters]:
        """
        Generate failure scenarios for anomaly detection training.
        
        Args:
            base_scenarios (List[ScenarioParameters]): Base scenarios
            num_failures (int): Number of failure scenarios
            
        Returns:
            List[ScenarioParameters]: List of failure scenarios
        """
        failure_scenarios = []
        
        failure_types = [
            "thermal_runaway",
            "internal_short_circuit",
            "electrolyte_leakage",
            "separator_failure",
            "current_collector_corrosion",
            "gas_generation",
            "mechanical_damage"
        ]
        
        for i in range(num_failures):
            base_scenario = np.random.choice(base_scenarios)
            failure_type = np.random.choice(failure_types)
            
            failure_scenario = self._create_failure_scenario(base_scenario, failure_type)
            failure_scenarios.append(failure_scenario)
        
        logger.info(f"Generated {len(failure_scenarios)} failure scenarios")
        return failure_scenarios
    
    def _create_failure_scenario(self, base_scenario: ScenarioParameters, 
                               failure_type: str) -> ScenarioParameters:
        """Create a failure scenario with specific failure mode."""
        import copy
        failure_scenario = copy.deepcopy(base_scenario)
        
        # Add failure-specific conditions
        failure_scenario.environmental_factors["failure_mode"] = failure_type
        
        if failure_type == "thermal_runaway":
            failure_scenario.temperature_range = (60, 150)
            failure_scenario.operating_conditions.append(OperatingCondition.EXTREME_HOT)
        elif failure_type == "internal_short_circuit":
            failure_scenario.c_rate_range = (0, 10)
            failure_scenario.environmental_factors["internal_resistance_drop"] = True
        elif failure_type == "electrolyte_leakage":
            failure_scenario.environmental_factors["electrolyte_loss"] = True
            failure_scenario.cycle_life_target = int(failure_scenario.cycle_life_target * 0.1)
        
        return failure_scenario
    
    def create_scenario_combinations(self, scenarios: List[ScenarioParameters]) -> pd.DataFrame:
        """
        Create a comprehensive DataFrame of scenario combinations.
        
        Args:
            scenarios (List[ScenarioParameters]): List of scenarios
            
        Returns:
            pd.DataFrame: DataFrame with scenario parameters
        """
        scenario_data = []
        
        for i, scenario in enumerate(scenarios):
            scenario_dict = {
                "scenario_id": f"scenario_{i:04d}",
                "chemistry": scenario.chemistry.value,
                "application": scenario.application.value,
                "capacity_ah": scenario.capacity_ah,
                "voltage_nominal": scenario.voltage_nominal,
                "temp_min": scenario.temperature_range[0],
                "temp_max": scenario.temperature_range[1],
                "soc_min": scenario.soc_range[0],
                "soc_max": scenario.soc_range[1],
                "c_rate_min": scenario.c_rate_range[0],
                "c_rate_max": scenario.c_rate_range[1],
                "cycle_life_target": scenario.cycle_life_target,
                "calendar_life_years": scenario.calendar_life_years,
                "operating_conditions": ",".join([oc.value for oc in scenario.operating_conditions]),
                "degradation_mechanisms": ",".join(scenario.degradation_mechanisms),
                "usage_patterns": ",".join(scenario.usage_patterns)
            }
            
            # Add environmental factors
            for key, value in scenario.environmental_factors.items():
                scenario_dict[f"env_{key}"] = value
            
            scenario_data.append(scenario_dict)
        
        df = pd.DataFrame(scenario_data)
        logger.info(f"Created scenario DataFrame with {len(df)} scenarios and {len(df.columns)} features")
        
        return df
    
    def save_scenarios(self, scenarios: List[ScenarioParameters], 
                      filepath: str) -> None:
        """Save scenarios to file."""
        scenario_df = self.create_scenario_combinations(scenarios)
        
        if filepath.endswith('.csv'):
            scenario_df.to_csv(filepath, index=False)
        elif filepath.endswith('.json'):
            scenario_df.to_json(filepath, orient='records', indent=2)
        else:
            # Default to CSV
            scenario_df.to_csv(filepath + '.csv', index=False)
        
        logger.info(f"Saved {len(scenarios)} scenarios to {filepath}")
    
    def generate_comprehensive_scenarios(self, total_scenarios: int = 1000) -> List[ScenarioParameters]:
        """
        Generate a comprehensive set of scenarios covering all aspects.
        
        Args:
            total_scenarios (int): Total number of scenarios to generate
            
        Returns:
            List[ScenarioParameters]: Complete set of scenarios
        """
        # Distribute scenarios across types
        num_base = int(total_scenarios * 0.7)
        num_extreme = int(total_scenarios * 0.2)
        num_failure = int(total_scenarios * 0.1)
        
        # Generate base scenarios
        base_scenarios = self.generate_base_scenarios(num_base)
        
        # Generate extreme scenarios
        extreme_scenarios = self.generate_extreme_scenarios(base_scenarios, num_extreme)
        
        # Generate failure scenarios
        failure_scenarios = self.generate_failure_scenarios(base_scenarios, num_failure)
        
        # Combine all scenarios
        all_scenarios = base_scenarios + extreme_scenarios + failure_scenarios
        
        logger.info(f"Generated comprehensive scenario set: {len(all_scenarios)} total scenarios")
        logger.info(f"  - Base scenarios: {len(base_scenarios)}")
        logger.info(f"  - Extreme scenarios: {len(extreme_scenarios)}")
        logger.info(f"  - Failure scenarios: {len(failure_scenarios)}")
        
        return all_scenarios

# Factory functions
def create_scenario_builder(seed: int = 42) -> BatteryScenarioBuilder:
    """Create a battery scenario builder."""
    return BatteryScenarioBuilder(seed)

def generate_battery_scenarios(num_scenarios: int = 1000, 
                             output_path: Optional[str] = None) -> List[ScenarioParameters]:
    """
    Generate comprehensive battery scenarios.
    
    Args:
        num_scenarios (int): Number of scenarios to generate
        output_path (str, optional): Path to save scenarios
        
    Returns:
        List[ScenarioParameters]: Generated scenarios
    """
    builder = create_scenario_builder()
    scenarios = builder.generate_comprehensive_scenarios(num_scenarios)
    
    if output_path:
        builder.save_scenarios(scenarios, output_path)
    
    return scenarios

# Utility functions
def validate_scenario(scenario: ScenarioParameters) -> bool:
    """Validate a scenario for physical consistency."""
    # Check temperature range
    if scenario.temperature_range[0] >= scenario.temperature_range[1]:
        return False
    
    # Check SOC range
    if not (0 <= scenario.soc_range[0] < scenario.soc_range[1] <= 1):
        return False
    
    # Check C-rate range
    if scenario.c_rate_range[0] >= scenario.c_rate_range[1]:
        return False
    
    # Check capacity
    if scenario.capacity_ah <= 0:
        return False
    
    return True

def analyze_scenario_coverage(scenarios: List[ScenarioParameters]) -> Dict[str, Any]:
    """Analyze the coverage of generated scenarios."""
    analysis = {
        "total_scenarios": len(scenarios),
        "chemistry_distribution": {},
        "application_distribution": {},
        "capacity_range": (float('inf'), -float('inf')),
        "temperature_range": (float('inf'), -float('inf')),
        "operating_conditions": set(),
        "degradation_mechanisms": set()
    }
    
    for scenario in scenarios:
        # Chemistry distribution
        chem = scenario.chemistry.value
        analysis["chemistry_distribution"][chem] = analysis["chemistry_distribution"].get(chem, 0) + 1
        
        # Application distribution
        app = scenario.application.value
        analysis["application_distribution"][app] = analysis["application_distribution"].get(app, 0) + 1
        
        # Capacity range
        analysis["capacity_range"] = (
            min(analysis["capacity_range"][0], scenario.capacity_ah),
            max(analysis["capacity_range"][1], scenario.capacity_ah)
        )
        
        # Temperature range
        analysis["temperature_range"] = (
            min(analysis["temperature_range"][0], scenario.temperature_range[0]),
            max(analysis["temperature_range"][1], scenario.temperature_range[1])
        )
        
        # Operating conditions
        analysis["operating_conditions"].update([oc.value for oc in scenario.operating_conditions])
        
        # Degradation mechanisms
        analysis["degradation_mechanisms"].update(scenario.degradation_mechanisms)
    
    # Convert sets to lists for JSON serialization
    analysis["operating_conditions"] = list(analysis["operating_conditions"])
    analysis["degradation_mechanisms"] = list(analysis["degradation_mechanisms"])
    
    return analysis
