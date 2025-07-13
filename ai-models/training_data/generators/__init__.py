"""
BatteryMind - Training Data Generators Package

Comprehensive synthetic data generation framework for battery management systems.
Provides physics-based simulation, statistical modeling, and domain-expert
knowledge integration for creating realistic training datasets.

Key Components:
- SyntheticGenerator: Main data generation orchestrator
- PhysicsSimulator: Battery physics and electrochemical modeling
- NoiseGenerator: Realistic sensor noise and environmental variations
- ScenarioBuilder: Complex usage pattern and fleet behavior simulation

Features:
- Physics-based battery modeling with electrochemical accuracy
- Multi-modal sensor data synthesis (electrical, thermal, acoustic)
- Fleet-scale simulation with diverse usage patterns
- Environmental condition modeling across global climates
- Degradation pattern synthesis with realistic aging mechanisms
- Anomaly injection for robust model training
- Federated learning data distribution simulation

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

from .synthetic_generator import (
    SyntheticDataGenerator,
    BatteryDataConfig,
    FleetSimulationConfig,
    DataGenerationPipeline,
    MultiModalDataGenerator,
    FederatedDataDistributor
)

from .physics_simulator import (
    BatteryPhysicsSimulator,
    ElectrochemicalModel,
    ThermalModel,
    DegradationModel,
    PhysicsBasedGenerator
)

from .noise_generator import (
    SensorNoiseGenerator,
    EnvironmentalNoiseModel,
    MeasurementUncertaintyModel,
    RealisticNoiseInjector
)

from .scenario_builder import (
    UsageScenarioBuilder,
    FleetBehaviorSimulator,
    EnvironmentalConditionGenerator,
    AnomalyScenarioBuilder
)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module metadata
__all__ = [
    # Main generators
    "SyntheticDataGenerator",
    "BatteryDataConfig", 
    "FleetSimulationConfig",
    "DataGenerationPipeline",
    "MultiModalDataGenerator",
    "FederatedDataDistributor",
    
    # Physics simulation
    "BatteryPhysicsSimulator",
    "ElectrochemicalModel",
    "ThermalModel", 
    "DegradationModel",
    "PhysicsBasedGenerator",
    
    # Noise modeling
    "SensorNoiseGenerator",
    "EnvironmentalNoiseModel",
    "MeasurementUncertaintyModel",
    "RealisticNoiseInjector",
    
    # Scenario building
    "UsageScenarioBuilder",
    "FleetBehaviorSimulator", 
    "EnvironmentalConditionGenerator",
    "AnomalyScenarioBuilder"
]

# Default configuration templates
DEFAULT_BATTERY_CONFIG = {
    "chemistry": "lithium_ion",
    "capacity_ah": 75.0,
    "nominal_voltage": 3.7,
    "cell_count": 96,
    "pack_configuration": "96s1p",
    "thermal_mass": 50.0,
    "internal_resistance": 0.05
}

DEFAULT_FLEET_CONFIG = {
    "fleet_size": 1000,
    "vehicle_types": ["passenger", "commercial", "heavy_duty"],
    "geographic_distribution": "global",
    "usage_patterns": ["urban", "highway", "mixed"],
    "charging_infrastructure": ["home", "workplace", "public", "fast"]
}

DEFAULT_SIMULATION_CONFIG = {
    "simulation_duration_days": 365,
    "sampling_frequency_hz": 1.0,
    "environmental_conditions": True,
    "degradation_modeling": True,
    "anomaly_injection_rate": 0.05,
    "noise_modeling": True
}

# Data generation presets for different use cases
GENERATION_PRESETS = {
    "development": {
        "num_batteries": 100,
        "simulation_days": 30,
        "sampling_rate": 0.1,
        "complexity": "low"
    },
    "training": {
        "num_batteries": 10000,
        "simulation_days": 365,
        "sampling_rate": 1.0,
        "complexity": "high"
    },
    "validation": {
        "num_batteries": 1000,
        "simulation_days": 180,
        "sampling_rate": 1.0,
        "complexity": "medium"
    },
    "federated": {
        "num_batteries": 50000,
        "simulation_days": 730,
        "sampling_rate": 1.0,
        "complexity": "high",
        "distributed": True
    }
}

def get_generation_preset(preset_name: str) -> dict:
    """
    Get predefined generation configuration.
    
    Args:
        preset_name (str): Name of the preset configuration
        
    Returns:
        dict: Generation configuration
    """
    return GENERATION_PRESETS.get(preset_name, GENERATION_PRESETS["development"])

def create_battery_fleet_generator(preset: str = "training") -> 'SyntheticDataGenerator':
    """
    Factory function to create a battery fleet data generator.
    
    Args:
        preset (str): Generation preset to use
        
    Returns:
        SyntheticDataGenerator: Configured data generator
    """
    config = get_generation_preset(preset)
    return SyntheticDataGenerator(
        battery_config=BatteryDataConfig(**DEFAULT_BATTERY_CONFIG),
        fleet_config=FleetSimulationConfig(**DEFAULT_FLEET_CONFIG),
        **config
    )

def create_federated_data_distributor(num_clients: int = 100) -> 'FederatedDataDistributor':
    """
    Factory function to create a federated learning data distributor.
    
    Args:
        num_clients (int): Number of federated learning clients
        
    Returns:
        FederatedDataDistributor: Configured data distributor
    """
    return FederatedDataDistributor(
        num_clients=num_clients,
        distribution_strategy="non_iid",
        privacy_preserving=True
    )

def create_physics_based_generator() -> 'BatteryPhysicsSimulator':
    """
    Factory function to create a physics-based battery simulator.
    
    Returns:
        BatteryPhysicsSimulator: Configured physics simulator
    """
    return BatteryPhysicsSimulator(
        electrochemical_model="equivalent_circuit",
        thermal_model="lumped_parameter",
        degradation_model="empirical_aging"
    )

# Utility functions for data generation
def estimate_generation_time(config: dict) -> dict:
    """
    Estimate time required for data generation.
    
    Args:
        config (dict): Generation configuration
        
    Returns:
        dict: Time estimates
    """
    num_batteries = config.get("num_batteries", 1000)
    simulation_days = config.get("simulation_days", 365)
    sampling_rate = config.get("sampling_rate", 1.0)
    
    # Rough estimates based on typical generation performance
    base_time_per_battery = 0.1  # seconds
    complexity_factor = {
        "low": 1.0,
        "medium": 2.0,
        "high": 4.0
    }.get(config.get("complexity", "medium"), 2.0)
    
    estimated_seconds = (num_batteries * simulation_days * sampling_rate * 
                        base_time_per_battery * complexity_factor)
    
    return {
        "estimated_seconds": estimated_seconds,
        "estimated_minutes": estimated_seconds / 60,
        "estimated_hours": estimated_seconds / 3600,
        "data_points": num_batteries * simulation_days * 24 * 3600 * sampling_rate
    }

def validate_generation_config(config: dict) -> dict:
    """
    Validate data generation configuration.
    
    Args:
        config (dict): Configuration to validate
        
    Returns:
        dict: Validation results
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check required fields
    required_fields = ["num_batteries", "simulation_days"]
    for field in required_fields:
        if field not in config:
            validation_results["errors"].append(f"Missing required field: {field}")
            validation_results["valid"] = False
    
    # Check reasonable values
    if config.get("num_batteries", 0) <= 0:
        validation_results["errors"].append("num_batteries must be positive")
        validation_results["valid"] = False
    
    if config.get("simulation_days", 0) <= 0:
        validation_results["errors"].append("simulation_days must be positive")
        validation_results["valid"] = False
    
    # Check for potentially large generations
    if config.get("num_batteries", 0) > 100000:
        validation_results["warnings"].append("Large number of batteries may require significant time")
    
    if config.get("simulation_days", 0) > 1000:
        validation_results["warnings"].append("Long simulation duration may require significant storage")
    
    return validation_results

# Battery chemistry specifications
BATTERY_CHEMISTRIES = {
    "lithium_ion": {
        "nominal_voltage": 3.7,
        "energy_density_wh_kg": 250,
        "power_density_w_kg": 1500,
        "cycle_life": 3000,
        "temperature_range": (-20, 60),
        "degradation_rate": 0.02  # per 100 cycles
    },
    "lifepo4": {
        "nominal_voltage": 3.2,
        "energy_density_wh_kg": 160,
        "power_density_w_kg": 1000,
        "cycle_life": 6000,
        "temperature_range": (-30, 70),
        "degradation_rate": 0.01
    },
    "nimh": {
        "nominal_voltage": 1.2,
        "energy_density_wh_kg": 80,
        "power_density_w_kg": 500,
        "cycle_life": 1000,
        "temperature_range": (-20, 50),
        "degradation_rate": 0.05
    }
}

def get_battery_chemistry_specs(chemistry: str) -> dict:
    """
    Get specifications for a battery chemistry.
    
    Args:
        chemistry (str): Battery chemistry name
        
    Returns:
        dict: Chemistry specifications
    """
    return BATTERY_CHEMISTRIES.get(chemistry, BATTERY_CHEMISTRIES["lithium_ion"])

# Environmental condition templates
ENVIRONMENTAL_CONDITIONS = {
    "tropical": {
        "temperature_range": (20, 45),
        "humidity_range": (60, 95),
        "pressure_range": (1000, 1020),
        "seasonal_variation": 0.2
    },
    "temperate": {
        "temperature_range": (-10, 35),
        "humidity_range": (30, 80),
        "pressure_range": (980, 1030),
        "seasonal_variation": 0.8
    },
    "arctic": {
        "temperature_range": (-40, 10),
        "humidity_range": (20, 70),
        "pressure_range": (950, 1050),
        "seasonal_variation": 1.0
    },
    "desert": {
        "temperature_range": (5, 50),
        "humidity_range": (5, 30),
        "pressure_range": (990, 1010),
        "seasonal_variation": 0.6
    }
}

def get_environmental_conditions(climate: str) -> dict:
    """
    Get environmental conditions for a climate type.
    
    Args:
        climate (str): Climate type
        
    Returns:
        dict: Environmental conditions
    """
    return ENVIRONMENTAL_CONDITIONS.get(climate, ENVIRONMENTAL_CONDITIONS["temperate"])

# Module health check
def health_check() -> dict:
    """
    Perform health check of the data generators module.
    
    Returns:
        dict: Health check results
    """
    health_status = {
        "module_loaded": True,
        "version": __version__,
        "components_available": {
            "synthetic_generator": True,
            "physics_simulator": True,
            "noise_generator": True,
            "scenario_builder": True
        },
        "dependencies_satisfied": True
    }
    
    try:
        # Test basic functionality
        config = get_generation_preset("development")
        validation_results = validate_generation_config(config)
        health_status["config_validation"] = validation_results["valid"]
    except Exception as e:
        health_status["config_validation"] = False
        health_status["error"] = str(e)
    
    return health_status

# Export configuration template
def export_config_template(file_path: str = "data_generation_config.yaml") -> None:
    """
    Export a configuration template for data generation.
    
    Args:
        file_path (str): Path to save the configuration template
    """
    import yaml
    
    config_template = {
        "battery_config": DEFAULT_BATTERY_CONFIG,
        "fleet_config": DEFAULT_FLEET_CONFIG,
        "simulation_config": DEFAULT_SIMULATION_CONFIG,
        "generation_presets": GENERATION_PRESETS,
        "_metadata": {
            "version": __version__,
            "description": "BatteryMind Data Generation Configuration Template",
            "author": __author__
        }
    }
    
    with open(file_path, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False, indent=2)

# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BatteryMind Training Data Generators v{__version__} initialized")
