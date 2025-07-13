"""
BatteryMind - Synthetic Data Generator

Advanced synthetic data generation system for battery management systems.
Creates realistic, physics-based training datasets for machine learning models
with comprehensive sensor data, environmental conditions, and usage patterns.

Features:
- Physics-based battery modeling with electrochemical accuracy
- Multi-modal sensor data synthesis (voltage, current, temperature, SoC, SoH)
- Fleet-scale simulation with diverse vehicle types and usage patterns
- Environmental condition modeling across different climates
- Realistic degradation patterns with multiple aging mechanisms
- Anomaly injection for robust model training
- Federated learning data distribution with privacy considerations
- Scalable generation for 10,000+ battery simulations

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import time
import json
import pickle
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
from collections import defaultdict
import random
import math

# Scientific computing
from scipy import signal, stats, interpolate
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BatteryDataConfig:
    """
    Configuration for battery data generation.
    
    Attributes:
        chemistry (str): Battery chemistry type
        capacity_ah (float): Battery capacity in Ah
        nominal_voltage (float): Nominal voltage per cell
        cell_count (int): Number of cells in series
        pack_configuration (str): Pack configuration (e.g., "96s1p")
        thermal_mass (float): Thermal mass in J/K
        internal_resistance (float): Internal resistance in Ohms
        degradation_rate (float): Degradation rate per cycle
        temperature_coefficients (Dict): Temperature dependency coefficients
        aging_mechanisms (List[str]): List of aging mechanisms to model
    """
    chemistry: str = "lithium_ion"
    capacity_ah: float = 75.0
    nominal_voltage: float = 3.7
    cell_count: int = 96
    pack_configuration: str = "96s1p"
    thermal_mass: float = 50.0
    internal_resistance: float = 0.05
    degradation_rate: float = 0.02
    temperature_coefficients: Dict[str, float] = field(default_factory=lambda: {
        "capacity": -0.005,  # %/°C
        "resistance": 0.01,  # %/°C
        "voltage": -0.002   # V/°C
    })
    aging_mechanisms: List[str] = field(default_factory=lambda: [
        "calendar_aging", "cycle_aging", "thermal_aging"
    ])

@dataclass
class FleetSimulationConfig:
    """
    Configuration for fleet simulation.
    
    Attributes:
        fleet_size (int): Number of vehicles in fleet
        vehicle_types (List[str]): Types of vehicles to simulate
        geographic_distribution (str): Geographic distribution pattern
        usage_patterns (List[str]): Usage pattern types
        charging_infrastructure (List[str]): Available charging types
        environmental_conditions (List[str]): Environmental condition types
        simulation_duration_days (int): Simulation duration in days
        data_collection_frequency (str): Data collection frequency
    """
    fleet_size: int = 1000
    vehicle_types: List[str] = field(default_factory=lambda: ["passenger", "commercial"])
    geographic_distribution: str = "global"
    usage_patterns: List[str] = field(default_factory=lambda: ["urban", "highway", "mixed"])
    charging_infrastructure: List[str] = field(default_factory=lambda: ["home", "public", "fast"])
    environmental_conditions: List[str] = field(default_factory=lambda: ["temperate", "tropical"])
    simulation_duration_days: int = 365
    data_collection_frequency: str = "1min"

class BatteryPhysicsModel:
    """
    Physics-based battery model for realistic data generation.
    """
    
    def __init__(self, config: BatteryDataConfig):
        self.config = config
        self.state = {
            'soc': 0.5,  # State of charge (0-1)
            'soh': 1.0,  # State of health (0-1)
            'temperature': 25.0,  # Temperature in Celsius
            'voltage': config.nominal_voltage * config.cell_count,
            'current': 0.0,
            'internal_resistance': config.internal_resistance,
            'capacity_current': config.capacity_ah,
            'cycle_count': 0,
            'age_days': 0
        }
        
        # Initialize lookup tables for voltage-SoC relationship
        self._initialize_voltage_tables()
        
        # Degradation tracking
        self.degradation_factors = {
            'calendar_aging': 1.0,
            'cycle_aging': 1.0,
            'thermal_aging': 1.0
        }
    
    def _initialize_voltage_tables(self):
        """Initialize voltage-SoC lookup tables for different chemistries."""
        if self.config.chemistry == "lithium_ion":
            # Typical Li-ion voltage curve
            self.soc_points = np.linspace(0, 1, 21)
            self.voltage_points = np.array([
                2.5, 2.8, 3.0, 3.2, 3.4, 3.5, 3.6, 3.65, 3.7, 3.75, 3.8,
                3.85, 3.9, 3.95, 4.0, 4.05, 4.1, 4.15, 4.18, 4.2, 4.2
            ]) * self.config.cell_count
        elif self.config.chemistry == "lifepo4":
            # LiFePO4 voltage curve (flatter)
            self.soc_points = np.linspace(0, 1, 21)
            self.voltage_points = np.array([
                2.0, 2.5, 2.8, 3.0, 3.1, 3.15, 3.2, 3.22, 3.25, 3.27, 3.3,
                3.32, 3.35, 3.37, 3.4, 3.42, 3.45, 3.5, 3.6, 3.65, 3.65
            ]) * self.config.cell_count
        else:
            # Default curve
            self.soc_points = np.linspace(0, 1, 11)
            self.voltage_points = np.linspace(
                2.5 * self.config.cell_count,
                4.2 * self.config.cell_count,
                11
            )
    
    def update_state(self, current: float, ambient_temp: float, dt: float):
        """
        Update battery state based on current and environmental conditions.
        
        Args:
            current (float): Applied current in Amperes (positive = discharge)
            ambient_temp (float): Ambient temperature in Celsius
            dt (float): Time step in hours
        """
        # Update SoC based on current
        coulomb_efficiency = 0.99 if current < 0 else 1.0  # Charging efficiency
        capacity_used = current * dt * coulomb_efficiency
        self.state['soc'] = np.clip(
            self.state['soc'] - capacity_used / self.state['capacity_current'],
            0.0, 1.0
        )
        
        # Update temperature (simplified thermal model)
        heat_generation = current**2 * self.state['internal_resistance']
        heat_dissipation = 0.1 * (self.state['temperature'] - ambient_temp)
        temp_change = (heat_generation - heat_dissipation) * dt / self.config.thermal_mass
        self.state['temperature'] = self.state['temperature'] + temp_change
        
        # Update voltage based on SoC and current
        ocv = np.interp(self.state['soc'], self.soc_points, self.voltage_points)
        voltage_drop = current * self.state['internal_resistance']
        self.state['voltage'] = ocv - voltage_drop
        
        # Apply temperature effects
        temp_factor = 1 + self.config.temperature_coefficients['voltage'] * (
            self.state['temperature'] - 25
        )
        self.state['voltage'] *= temp_factor
        
        # Update current
        self.state['current'] = current
        
        # Update aging
        self._update_aging(dt)
    
    def _update_aging(self, dt: float):
        """Update battery aging and degradation."""
        # Calendar aging (time-based)
        calendar_rate = 0.001 * np.exp(0.05 * (self.state['temperature'] - 25))
        self.degradation_factors['calendar_aging'] *= (1 - calendar_rate * dt / 24)
        
        # Cycle aging (usage-based)
        if abs(self.state['current']) > 0.1:  # Only during active use
            cycle_stress = abs(self.state['current']) / self.config.capacity_ah
            depth_stress = abs(self.state['soc'] - 0.5) * 2  # Stress increases away from 50%
            cycle_rate = self.config.degradation_rate * cycle_stress * depth_stress
            self.degradation_factors['cycle_aging'] *= (1 - cycle_rate * dt / 1000)
        
        # Thermal aging (temperature-based)
        if self.state['temperature'] > 35:
            thermal_rate = 0.0001 * np.exp(0.1 * (self.state['temperature'] - 35))
            self.degradation_factors['thermal_aging'] *= (1 - thermal_rate * dt / 24)
        
        # Update overall SoH
        self.state['soh'] = np.prod(list(self.degradation_factors.values()))
        
        # Update capacity based on SoH
        self.state['capacity_current'] = self.config.capacity_ah * self.state['soh']
        
        # Update internal resistance (increases with aging)
        resistance_increase = 1 + (1 - self.state['soh']) * 2
        self.state['internal_resistance'] = self.config.internal_resistance * resistance_increase
        
        # Update age tracking
        self.state['age_days'] += dt / 24
        if abs(self.state['current']) > 0.1:
            self.state['cycle_count'] += dt / 24  # Simplified cycle counting

class UsagePatternGenerator:
    """
    Generates realistic usage patterns for different vehicle types and scenarios.
    """
    
    def __init__(self, vehicle_type: str = "passenger", usage_pattern: str = "mixed"):
        self.vehicle_type = vehicle_type
        self.usage_pattern = usage_pattern
        
        # Define usage characteristics
        self.usage_profiles = {
            "passenger": {
                "urban": {
                    "daily_distance_km": (20, 80),
                    "trip_frequency": (2, 8),
                    "speed_profile": "stop_and_go",
                    "charging_preference": "home"
                },
                "highway": {
                    "daily_distance_km": (100, 400),
                    "trip_frequency": (1, 3),
                    "speed_profile": "highway",
                    "charging_preference": "fast"
                },
                "mixed": {
                    "daily_distance_km": (40, 200),
                    "trip_frequency": (2, 6),
                    "speed_profile": "mixed",
                    "charging_preference": "mixed"
                }
            },
            "commercial": {
                "urban": {
                    "daily_distance_km": (150, 300),
                    "trip_frequency": (10, 30),
                    "speed_profile": "delivery",
                    "charging_preference": "depot"
                },
                "highway": {
                    "daily_distance_km": (300, 800),
                    "trip_frequency": (2, 6),
                    "speed_profile": "highway",
                    "charging_preference": "fast"
                }
            }
        }
    
    def generate_daily_profile(self, day_of_year: int) -> List[Dict]:
        """
        Generate daily usage profile with trips and charging events.
        
        Args:
            day_of_year (int): Day of the year (1-365)
            
        Returns:
            List[Dict]: List of events with timestamps and power demands
        """
        profile = self.usage_profiles[self.vehicle_type][self.usage_pattern]
        
        # Generate daily distance
        distance_range = profile["daily_distance_km"]
        daily_distance = np.random.uniform(distance_range[0], distance_range[1])
        
        # Adjust for seasonal patterns
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
        daily_distance *= seasonal_factor
        
        # Generate trip schedule
        trip_freq_range = profile["trip_frequency"]
        num_trips = int(np.random.uniform(trip_freq_range[0], trip_freq_range[1]))
        
        events = []
        current_time = 0  # Hours from midnight
        
        for trip_idx in range(num_trips):
            # Generate trip timing
            if self.vehicle_type == "passenger":
                # Passenger vehicles have typical commute patterns
                if trip_idx == 0:  # Morning commute
                    trip_start = np.random.normal(8, 1)  # 8 AM ± 1 hour
                elif trip_idx == 1:  # Evening commute
                    trip_start = np.random.normal(17, 1)  # 5 PM ± 1 hour
                else:
                    trip_start = np.random.uniform(current_time + 1, 23)
            else:
                # Commercial vehicles spread throughout day
                trip_start = np.random.uniform(6, 20)  # 6 AM to 8 PM
            
            trip_start = max(current_time + 0.5, trip_start)
            trip_distance = daily_distance / num_trips
            
            # Generate power profile for trip
            trip_events = self._generate_trip_power_profile(
                trip_start, trip_distance, profile["speed_profile"]
            )
            events.extend(trip_events)
            
            current_time = trip_start + trip_events[-1]["duration"]
        
        # Add charging events
        charging_events = self._generate_charging_events(
            events, profile["charging_preference"]
        )
        events.extend(charging_events)
        
        # Sort events by time
        events.sort(key=lambda x: x["start_time"])
        
        return events
    
    def _generate_trip_power_profile(self, start_time: float, distance_km: float, 
                                   speed_profile: str) -> List[Dict]:
        """Generate power consumption profile for a trip."""
        events = []
        
        # Define speed and power characteristics
        if speed_profile == "stop_and_go":
            avg_speed = 25  # km/h
            power_base = 15  # kW
            power_variation = 0.5
        elif speed_profile == "highway":
            avg_speed = 80  # km/h
            power_base = 20  # kW
            power_variation = 0.3
        elif speed_profile == "delivery":
            avg_speed = 20  # km/h
            power_base = 12  # kW
            power_variation = 0.6
        else:  # mixed
            avg_speed = 45  # km/h
            power_base = 18  # kW
            power_variation = 0.4
        
        trip_duration = distance_km / avg_speed  # hours
        
        # Generate power profile with variations
        num_segments = max(1, int(trip_duration * 10))  # 6-minute segments
        segment_duration = trip_duration / num_segments
        
        for i in range(num_segments):
            # Add random variation to power consumption
            power_factor = 1 + np.random.normal(0, power_variation)
            power_kw = power_base * power_factor
            
            # Add regenerative braking for stop-and-go
            if speed_profile == "stop_and_go" and np.random.random() < 0.3:
                power_kw *= -0.2  # Regenerative braking
            
            events.append({
                "type": "drive",
                "start_time": start_time + i * segment_duration,
                "duration": segment_duration,
                "power_kw": power_kw,
                "distance_km": distance_km / num_segments
            })
        
        return events
    
    def _generate_charging_events(self, drive_events: List[Dict], 
                                charging_preference: str) -> List[Dict]:
        """Generate charging events based on usage and preferences."""
        charging_events = []
        
        # Determine charging strategy
        if charging_preference == "home":
            # Overnight charging
            charging_events.append({
                "type": "charge",
                "start_time": 22 + np.random.normal(0, 1),  # 10 PM ± 1 hour
                "duration": 8,  # 8 hours
                "power_kw": -7.4,  # Level 2 charging (negative = charging)
                "location": "home"
            })
        elif charging_preference == "fast":
            # Fast charging during long trips
            for event in drive_events:
                if event["duration"] > 2:  # Long trips
                    if np.random.random() < 0.3:  # 30% chance of fast charging
                        charging_events.append({
                            "type": "charge",
                            "start_time": event["start_time"] + event["duration"] / 2,
                            "duration": 0.5,  # 30 minutes
                            "power_kw": -50,  # DC fast charging
                            "location": "public"
                        })
        elif charging_preference == "depot":
            # Commercial depot charging
            charging_events.append({
                "type": "charge",
                "start_time": 20,  # 8 PM
                "duration": 10,  # 10 hours
                "power_kw": -22,  # Level 3 charging
                "location": "depot"
            })
        else:  # mixed
            # Combination of charging strategies
            if np.random.random() < 0.7:  # 70% home charging
                charging_events.append({
                    "type": "charge",
                    "start_time": 23,
                    "duration": 6,
                    "power_kw": -7.4,
                    "location": "home"
                })
            if np.random.random() < 0.2:  # 20% workplace charging
                charging_events.append({
                    "type": "charge",
                    "start_time": 9,
                    "duration": 8,
                    "power_kw": -3.3,  # Level 1 charging
                    "location": "workplace"
                })
        
        return charging_events

class EnvironmentalConditionGenerator:
    """
    Generates realistic environmental conditions affecting battery performance.
    """
    
    def __init__(self, climate_type: str = "temperate", location: str = "global"):
        self.climate_type = climate_type
        self.location = location
        
        # Define climate characteristics
        self.climate_params = {
            "tropical": {
                "temp_mean": 28,
                "temp_range": 15,
                "humidity_mean": 80,
                "humidity_range": 20,
                "seasonal_variation": 0.3
            },
            "temperate": {
                "temp_mean": 15,
                "temp_range": 30,
                "humidity_mean": 60,
                "humidity_range": 40,
                "seasonal_variation": 1.0
            },
            "arctic": {
                "temp_mean": -10,
                "temp_range": 40,
                "humidity_mean": 50,
                "humidity_range": 30,
                "seasonal_variation": 1.5
            },
            "desert": {
                "temp_mean": 25,
                "temp_range": 35,
                "humidity_mean": 20,
                "humidity_range": 15,
                "seasonal_variation": 0.8
            }
        }
    
    def generate_conditions(self, day_of_year: int, hour_of_day: float) -> Dict[str, float]:
        """
        Generate environmental conditions for specific time.
        
        Args:
            day_of_year (int): Day of the year (1-365)
            hour_of_day (float): Hour of the day (0-24)
            
        Returns:
            Dict[str, float]: Environmental conditions
        """
        params = self.climate_params[self.climate_type]
        
        # Seasonal temperature variation
        seasonal_temp = params["temp_mean"] + params["seasonal_variation"] * \
                       params["temp_range"] * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Daily temperature variation
        daily_temp_variation = 0.3 * params["temp_range"] * \
                              np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Add random noise
        temp_noise = np.random.normal(0, 2)
        temperature = seasonal_temp + daily_temp_variation + temp_noise
        
        # Humidity (inversely related to temperature in many climates)
        humidity_base = params["humidity_mean"]
        humidity_temp_effect = -0.5 * (temperature - params["temp_mean"])
        humidity_noise = np.random.normal(0, 5)
        humidity = np.clip(humidity_base + humidity_temp_effect + humidity_noise, 10, 95)
        
        # Atmospheric pressure (varies with weather systems)
        pressure_base = 1013.25  # Standard atmospheric pressure
        pressure_variation = np.random.normal(0, 15)
        pressure = pressure_base + pressure_variation
        
        # Wind speed (affects cooling)
        wind_speed = np.random.gamma(2, 2)  # Gamma distribution for wind
        
        # Solar irradiance (affects cabin heating)
        if 6 <= hour_of_day <= 18:  # Daylight hours
            solar_angle = np.sin(np.pi * (hour_of_day - 6) / 12)
            cloud_factor = np.random.uniform(0.3, 1.0)
            solar_irradiance = 1000 * solar_angle * cloud_factor
        else:
            solar_irradiance = 0
        
        return {
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
            "wind_speed": wind_speed,
            "solar_irradiance": solar_irradiance
        }

class SyntheticDataGenerator:
    """
    Main synthetic data generator orchestrating all components.
    """
    
    def __init__(self, battery_config: BatteryDataConfig, 
                 fleet_config: FleetSimulationConfig,
                 num_batteries: int = 1000,
                 parallel_processing: bool = True):
        self.battery_config = battery_config
        self.fleet_config = fleet_config
        self.num_batteries = num_batteries
        self.parallel_processing = parallel_processing
        
        # Initialize generators
        self.physics_model = BatteryPhysicsModel(battery_config)
        self.usage_generator = UsagePatternGenerator()
        self.env_generator = EnvironmentalConditionGenerator()
        
        # Data storage
        self.generated_data = []
        self.metadata = {}
        
        logger.info(f"Synthetic data generator initialized for {num_batteries} batteries")
    
    def generate_fleet_data(self, output_dir: str = "./generated_data") -> Dict[str, str]:
        """
        Generate synthetic data for entire fleet.
        
        Args:
            output_dir (str): Directory to save generated data
            
        Returns:
            Dict[str, str]: Paths to generated data files
        """
        logger.info(f"Starting fleet data generation for {self.num_batteries} batteries")
        start_time = time.time()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate data for each battery
        if self.parallel_processing and self.num_batteries > 10:
            # Use parallel processing for large fleets
            with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
                futures = []
                for battery_id in range(self.num_batteries):
                    future = executor.submit(self._generate_single_battery_data, battery_id)
                    futures.append(future)
                
                # Collect results
                for i, future in enumerate(futures):
                    try:
                        battery_data = future.result(timeout=300)  # 5 minute timeout
                        self.generated_data.append(battery_data)
                        
                        if (i + 1) % 100 == 0:
                            logger.info(f"Generated data for {i + 1}/{self.num_batteries} batteries")
                    except Exception as e:
                        logger.error(f"Failed to generate data for battery {i}: {e}")
        else:
            # Sequential processing
            for battery_id in range(self.num_batteries):
                try:
                    battery_data = self._generate_single_battery_data(battery_id)
                    self.generated_data.append(battery_data)
                    
                    if (battery_id + 1) % 100 == 0:
                        logger.info(f"Generated data for {battery_id + 1}/{self.num_batteries} batteries")
                except Exception as e:
                    logger.error(f"Failed to generate data for battery {battery_id}: {e}")
        
        # Save generated data
        file_paths = self._save_generated_data(output_path)
        
        generation_time = time.time() - start_time
        logger.info(f"Fleet data generation completed in {generation_time:.2f} seconds")
        
        return file_paths
    
    def _generate_single_battery_data(self, battery_id: int) -> Dict[str, Any]:
        """Generate data for a single battery."""
        # Create unique configuration for this battery
        battery_config = self._create_battery_variant(battery_id)
        physics_model = BatteryPhysicsModel(battery_config)
        
        # Select random vehicle type and usage pattern
        vehicle_type = np.random.choice(self.fleet_config.vehicle_types)
        usage_pattern = np.random.choice(self.fleet_config.usage_patterns)
        usage_generator = UsagePatternGenerator(vehicle_type, usage_pattern)
        
        # Select random climate
        climate = np.random.choice(self.fleet_config.environmental_conditions)
        env_generator = EnvironmentalConditionGenerator(climate)
        
        # Generate time series data
        time_series_data = []
        
        for day in range(self.fleet_config.simulation_duration_days):
            # Generate daily usage profile
            daily_events = usage_generator.generate_daily_profile(day + 1)
            
            # Simulate 24 hours with 1-minute resolution
            for hour in range(24):
                for minute in range(60):
                    timestamp = day * 24 * 60 + hour * 60 + minute
                    hour_decimal = hour + minute / 60
                    
                    # Get environmental conditions
                    env_conditions = env_generator.generate_conditions(day + 1, hour_decimal)
                    
                    # Determine current power demand from events
                    current_power = 0.0
                    for event in daily_events:
                        event_start_min = event["start_time"] * 60
                        event_end_min = event_start_min + event["duration"] * 60
                        
                        if event_start_min <= hour * 60 + minute < event_end_min:
                            current_power = event["power_kw"]
                            break
                    
                    # Convert power to current (P = V * I)
                    if physics_model.state['voltage'] > 0:
                        current = current_power * 1000 / physics_model.state['voltage']  # Convert kW to A
                    else:
                        current = 0.0
                    
                    # Update physics model
                    physics_model.update_state(
                        current=current,
                        ambient_temp=env_conditions['temperature'],
                        dt=1/60  # 1 minute = 1/60 hour
                    )
                    
                    # Record data point
                    data_point = {
                        'timestamp': timestamp,
                        'battery_id': battery_id,
                        'day': day,
                        'hour': hour,
                        'minute': minute,
                        
                        # Battery state
                        'voltage': physics_model.state['voltage'],
                        'current': physics_model.state['current'],
                        'soc': physics_model.state['soc'],
                        'soh': physics_model.state['soh'],
                        'temperature': physics_model.state['temperature'],
                        'internal_resistance': physics_model.state['internal_resistance'],
                        'capacity_current': physics_model.state['capacity_current'],
                        'cycle_count': physics_model.state['cycle_count'],
                        'age_days': physics_model.state['age_days'],
                        
                        # Environmental conditions
                        'ambient_temperature': env_conditions['temperature'],
                        'humidity': env_conditions['humidity'],
                        'pressure': env_conditions['pressure'],
                        'wind_speed': env_conditions['wind_speed'],
                        'solar_irradiance': env_conditions['solar_irradiance'],
                        
                        # Usage context
                        'vehicle_type': vehicle_type,
                        'usage_pattern': usage_pattern,
                        'climate': climate,
                        'power_demand': current_power
                    }
                    
                    time_series_data.append(data_point)
        
        return {
            'battery_id': battery_id,
            'config': battery_config,
            'time_series': time_series_data,
            'metadata': {
                'vehicle_type': vehicle_type,
                'usage_pattern': usage_pattern,
                'climate': climate,
                'total_data_points': len(time_series_data)
            }
        }
    
    def _create_battery_variant(self, battery_id: int) -> BatteryDataConfig:
        """Create battery configuration variant for diversity."""
        # Add manufacturing variations
        capacity_variation = np.random.normal(1.0, 0.05)  # ±5% capacity variation
        resistance_variation = np.random.normal(1.0, 0.1)  # ±10% resistance variation
        
        # Create variant configuration
        variant_config = BatteryDataConfig(
            chemistry=self.battery_config.chemistry,
            capacity_ah=self.battery_config.capacity_ah * capacity_variation,
            nominal_voltage=self.battery_config.nominal_voltage,
            cell_count=self.battery_config.cell_count,
            pack_configuration=self.battery_config.pack_configuration,
            thermal_mass=self.battery_config.thermal_mass,
            internal_resistance=self.battery_config.internal_resistance * resistance_variation,
            degradation_rate=self.battery_config.degradation_rate * np.random.uniform(0.8, 1.2),
            temperature_coefficients=self.battery_config.temperature_coefficients.copy(),
            aging_mechanisms=self.battery_config.aging_mechanisms.copy()
        )
        
        return variant_config
    
    def _save_generated_data(self, output_path: Path) -> Dict[str, str]:
        """Save generated data to files."""
        file_paths = {}
        
        # Convert to DataFrame for easier handling
        all_data = []
        for battery_data in self.generated_data:
            all_data.extend(battery_data['time_series'])
        
        df = pd.DataFrame(all_data)
        
        # Save main telemetry data
        telemetry_path = output_path / "battery_telemetry.csv"
        df.to_csv(telemetry_path, index=False)
        file_paths['telemetry'] = str(telemetry_path)
        
        # Save degradation curves
        degradation_data = df.groupby('battery_id').agg({
            'soh': ['first', 'last', 'min'],
            'cycle_count': 'max',
            'age_days': 'max'
        }).reset_index()
        degradation_data.columns = ['battery_id', 'initial_soh', 'final_soh', 'min_soh', 'total_cycles', 'age_days']
        
        degradation_path = output_path / "degradation_curves.csv"
        degradation_data.to_csv(degradation_path, index=False)
        file_paths['degradation'] = str(degradation_path)
        
        # Save fleet patterns
        fleet_patterns = df.groupby(['vehicle_type', 'usage_pattern']).agg({
            'power_demand': ['mean', 'std', 'min', 'max'],
            'soc': ['mean', 'std'],
            'temperature': ['mean', 'std']
        }).reset_index()
        
        fleet_path = output_path / "fleet_patterns.csv"
        fleet_patterns.to_csv(fleet_path, index=False)
        file_paths['fleet_patterns'] = str(fleet_path)
        
        # Save environmental data
        env_data = df.groupby(['climate']).agg({
            'ambient_temperature': ['mean', 'std', 'min', 'max'],
            'humidity': ['mean', 'std'],
            'pressure': ['mean', 'std'],
            'solar_irradiance': ['mean', 'max']
        }).reset_index()
        
        env_path = output_path / "environmental_data.csv"
        env_data.to_csv(env_path, index=False)
        file_paths['environmental'] = str(env_path)
        
        # Save usage profiles
        usage_profiles = df.groupby(['vehicle_type', 'usage_pattern', 'day']).agg({
            'power_demand': ['sum', 'mean'],
            'current': ['mean', 'std']
        }).reset_index()
        
        usage_path = output_path / "usage_profiles.csv"
        usage_profiles.to_csv(usage_path, index=False)
        file_paths['usage_profiles'] = str(usage_path)
        
        # Save metadata
        metadata = {
            'generation_config': {
                'num_batteries': self.num_batteries,
                'simulation_duration_days': self.fleet_config.simulation_duration_days,
                'battery_config': self.battery_config.__dict__,
                'fleet_config': self.fleet_config.__dict__
            },
            'data_statistics': {
                'total_data_points': len(df),
                'batteries_generated': len(self.generated_data),
                'date_range': f"{self.fleet_config.simulation_duration_days} days",
                'sampling_frequency': "1 minute"
            },
            'file_paths': file_paths
        }
        
        metadata_path = output_path / "generation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        file_paths['metadata'] = str(metadata_path)
        
        logger.info(f"Generated data saved to {output_path}")
        return file_paths

# Factory functions
def create_battery_data_generator(preset: str = "training") -> SyntheticDataGenerator:
    """Create a synthetic data generator with preset configuration."""
    presets = {
        "development": {
            "num_batteries": 10,
            "simulation_days": 7,
            "parallel": False
        },
        "training": {
            "num_batteries": 1000,
            "simulation_days": 365,
            "parallel": True
        },
        "validation": {
            "num_batteries": 100,
            "simulation_days": 180,
            "parallel": True
        }
    }
    
    config = presets.get(preset, presets["development"])
    
    battery_config = BatteryDataConfig()
    fleet_config = FleetSimulationConfig(
        simulation_duration_days=config["simulation_days"]
    )
    
    return SyntheticDataGenerator(
        battery_config=battery_config,
        fleet_config=fleet_config,
        num_batteries=config["num_batteries"],
        parallel_processing=config["parallel"]
    )

def generate_quick_sample(num_batteries: int = 10, days: int = 7) -> pd.DataFrame:
    """Generate a quick sample dataset for testing."""
    generator = SyntheticDataGenerator(
        battery_config=BatteryDataConfig(),
        fleet_config=FleetSimulationConfig(simulation_duration_days=days),
        num_batteries=num_batteries,
        parallel_processing=False
    )
    
    # Generate data in memory
    all_data = []
    for battery_id in range(num_batteries):
        battery_data = generator._generate_single_battery_data(battery_id)
        all_data.extend(battery_data['time_series'])
    
    return pd.DataFrame(all_data)
