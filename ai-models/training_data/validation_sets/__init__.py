"""
BatteryMind - Validation Sets Data Module

Comprehensive validation dataset generation module that creates realistic
test scenarios, holdout datasets, cross-validation sets, and performance
benchmarks for battery management system validation and testing.

Key Components:
- TestScenariosGenerator: Creates diverse test scenarios for model validation
- HoldoutDataGenerator: Generates holdout datasets for unbiased evaluation
- CrossValidationGenerator: Creates time-series aware cross-validation sets
- BenchmarkGenerator: Generates performance benchmark datasets

Features:
- Time-series aware data splitting
- Scenario-based testing (normal, edge cases, fault conditions)
- Industry standard benchmark compliance
- Statistical validation of data distributions
- Temporal consistency preservation
- Domain-specific test case generation

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import time
import json
from pathlib import Path
import random
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module metadata
__all__ = [
    # Validation generators
    "TestScenariosGenerator",
    "HoldoutDataGenerator",
    "CrossValidationGenerator", 
    "BenchmarkGenerator",
    
    # Configuration classes
    "ValidationConfig",
    "TestScenarioConfig",
    "CrossValidationConfig",
    "BenchmarkConfig",
    
    # Utility functions
    "generate_test_scenarios",
    "generate_holdout_data",
    "generate_cross_validation_sets",
    "generate_performance_benchmarks",
    "validate_temporal_consistency",
    "calculate_validation_metrics"
]

@dataclass
class TestScenarioConfig:
    """
    Configuration for test scenario generation.
    
    Attributes:
        scenario_types (List[str]): Types of scenarios to generate
        num_scenarios_per_type (int): Number of scenarios per type
        duration_range (Tuple[int, int]): Duration range in days
        severity_levels (List[str]): Severity levels for scenarios
        include_edge_cases (bool): Include edge case scenarios
        include_fault_conditions (bool): Include fault condition scenarios
    """
    scenario_types: List[str] = field(default_factory=lambda: [
        'normal_operation', 'extreme_temperature', 'high_current', 
        'deep_discharge', 'overcharge', 'thermal_runaway', 'capacity_fade'
    ])
    num_scenarios_per_type: int = 100
    duration_range: Tuple[int, int] = (1, 30)  # days
    severity_levels: List[str] = field(default_factory=lambda: ['low', 'medium', 'high', 'critical'])
    include_edge_cases: bool = True
    include_fault_conditions: bool = True

@dataclass
class CrossValidationConfig:
    """
    Configuration for cross-validation set generation.
    
    Attributes:
        n_splits (int): Number of cross-validation splits
        test_size (float): Proportion of data for testing
        gap_size (int): Gap between train and test sets (time steps)
        max_train_size (Optional[int]): Maximum training set size
        shuffle (bool): Whether to shuffle data (not recommended for time series)
    """
    n_splits: int = 5
    test_size: float = 0.2
    gap_size: int = 0
    max_train_size: Optional[int] = None
    shuffle: bool = False

@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmark dataset generation.
    
    Attributes:
        benchmark_types (List[str]): Types of benchmarks to generate
        performance_metrics (List[str]): Performance metrics to include
        baseline_models (List[str]): Baseline models for comparison
        industry_standards (List[str]): Industry standards to comply with
        statistical_tests (List[str]): Statistical tests to include
    """
    benchmark_types: List[str] = field(default_factory=lambda: [
        'accuracy_benchmark', 'speed_benchmark', 'robustness_benchmark',
        'safety_benchmark', 'efficiency_benchmark'
    ])
    performance_metrics: List[str] = field(default_factory=lambda: [
        'mae', 'rmse', 'mape', 'r2_score', 'inference_time', 'memory_usage'
    ])
    baseline_models: List[str] = field(default_factory=lambda: [
        'linear_regression', 'random_forest', 'svm', 'lstm'
    ])
    industry_standards: List[str] = field(default_factory=lambda: [
        'iec_61960', 'iec_62660', 'sae_j1798', 'ieee_1725'
    ])
    statistical_tests: List[str] = field(default_factory=lambda: [
        'kolmogorov_smirnov', 'anderson_darling', 'shapiro_wilk'
    ])

@dataclass
class ValidationConfig:
    """
    Master configuration for validation dataset generation.
    
    Attributes:
        test_scenarios (TestScenarioConfig): Test scenario configuration
        cross_validation (CrossValidationConfig): Cross-validation configuration
        benchmarks (BenchmarkConfig): Benchmark configuration
        output_format (str): Output format for datasets
        preserve_temporal_order (bool): Preserve temporal ordering
        add_metadata (bool): Add metadata to datasets
        validation_split (float): Proportion for validation split
    """
    test_scenarios: TestScenarioConfig = field(default_factory=TestScenarioConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    benchmarks: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    output_format: str = "csv"
    preserve_temporal_order: bool = True
    add_metadata: bool = True
    validation_split: float = 0.15

class ValidationDataGenerator(ABC):
    """
    Abstract base class for validation data generators.
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.rng = np.random.RandomState(42)
        self.generated_sets = {}
        
    @abstractmethod
    def generate_validation_set(self) -> Dict[str, pd.DataFrame]:
        """Generate validation dataset."""
        pass
    
    def _ensure_temporal_consistency(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Ensure temporal consistency in the dataset."""
        if 'timestamp' in dataset.columns:
            dataset = dataset.sort_values('timestamp').reset_index(drop=True)
            
            # Check for temporal gaps
            if len(dataset) > 1:
                time_diffs = dataset['timestamp'].diff().dt.total_seconds()
                median_diff = time_diffs.median()
                
                # Flag unusual time gaps
                unusual_gaps = time_diffs > median_diff * 5
                if unusual_gaps.any():
                    logger.warning(f"Found {unusual_gaps.sum()} unusual temporal gaps")
        
        return dataset
    
    def _add_validation_metadata(self, dataset: pd.DataFrame, 
                                dataset_type: str) -> pd.DataFrame:
        """Add validation metadata to dataset."""
        if self.config.add_metadata:
            dataset['validation_set_type'] = dataset_type
            dataset['generation_timestamp'] = datetime.now().isoformat()
            dataset['validation_config'] = json.dumps(self.config.__dict__, default=str)
        
        return dataset

class TestScenariosGenerator(ValidationDataGenerator):
    """
    Generator for comprehensive test scenarios covering normal and edge cases.
    """
    
    def generate_validation_set(self) -> Dict[str, pd.DataFrame]:
        """Generate comprehensive test scenarios."""
        logger.info("Generating test scenarios for validation")
        
        test_scenarios = {}
        
        for scenario_type in self.config.test_scenarios.scenario_types:
            scenarios = self._generate_scenario_type(scenario_type)
            test_scenarios[scenario_type] = scenarios
        
        return test_scenarios
    
    def _generate_scenario_type(self, scenario_type: str) -> pd.DataFrame:
        """Generate specific type of test scenario."""
        scenarios_data = []
        
        for i in range(self.config.test_scenarios.num_scenarios_per_type):
            scenario_data = self._create_single_scenario(scenario_type, i)
            scenarios_data.append(scenario_data)
        
        combined_scenarios = pd.concat(scenarios_data, ignore_index=True)
        return self._add_validation_metadata(combined_scenarios, f"test_scenario_{scenario_type}")
    
    def _create_single_scenario(self, scenario_type: str, scenario_id: int) -> pd.DataFrame:
        """Create a single test scenario."""
        # Generate scenario duration
        duration_days = self.rng.randint(*self.config.test_scenarios.duration_range)
        n_points = duration_days * 24 * 60  # Minute-level data
        
        # Generate base timeline
        timestamps = pd.date_range(
            start='2023-01-01',
            periods=n_points,
            freq='1min'
        )
        
        # Generate scenario-specific data
        if scenario_type == 'normal_operation':
            scenario_data = self._generate_normal_operation_scenario(n_points)
        elif scenario_type == 'extreme_temperature':
            scenario_data = self._generate_extreme_temperature_scenario(n_points)
        elif scenario_type == 'high_current':
            scenario_data = self._generate_high_current_scenario(n_points)
        elif scenario_type == 'deep_discharge':
            scenario_data = self._generate_deep_discharge_scenario(n_points)
        elif scenario_type == 'overcharge':
            scenario_data = self._generate_overcharge_scenario(n_points)
        elif scenario_type == 'thermal_runaway':
            scenario_data = self._generate_thermal_runaway_scenario(n_points)
        elif scenario_type == 'capacity_fade':
            scenario_data = self._generate_capacity_fade_scenario(n_points)
        else:
            scenario_data = self._generate_default_scenario(n_points)
        
        # Create DataFrame
        scenario_df = pd.DataFrame({
            'scenario_id': f"{scenario_type}_{scenario_id:04d}",
            'scenario_type': scenario_type,
            'timestamp': timestamps,
            **scenario_data
        })
        
        return scenario_df
    
    def _generate_normal_operation_scenario(self, n_points: int) -> Dict[str, np.ndarray]:
        """Generate normal operation scenario."""
        # Normal battery operation parameters
        voltage = self.rng.normal(3.7, 0.1, n_points)
        voltage = np.clip(voltage, 3.0, 4.2)
        
        current = self.rng.normal(0, 10, n_points)  # Mixed charging/discharging
        
        temperature = self.rng.normal(25, 5, n_points)
        temperature = np.clip(temperature, 0, 50)
        
        soc = np.cumsum(current) / 1000 + 0.5  # Simplified SoC
        soc = np.clip(soc, 0.1, 0.9)
        
        soh = np.ones(n_points) * self.rng.uniform(0.85, 1.0)
        
        return {
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'soc': soc,
            'soh': soh,
            'severity_level': ['normal'] * n_points
        }
    
    def _generate_extreme_temperature_scenario(self, n_points: int) -> Dict[str, np.ndarray]:
        """Generate extreme temperature scenario."""
        # Extreme temperature conditions
        temp_extreme = self.rng.choice(['hot', 'cold'])
        
        if temp_extreme == 'hot':
            temperature = self.rng.normal(55, 5, n_points)
            temperature = np.clip(temperature, 45, 70)
        else:
            temperature = self.rng.normal(-15, 5, n_points)
            temperature = np.clip(temperature, -30, 0)
        
        # Temperature affects other parameters
        voltage = 3.7 + (temperature - 25) * 0.002  # Temperature coefficient
        current = self.rng.normal(0, 5, n_points)  # Reduced current in extreme temps
        
        soc = np.cumsum(current) / 1000 + 0.5
        soc = np.clip(soc, 0.1, 0.9)
        
        # SoH degrades faster in extreme temperatures
        base_soh = 0.9
        temp_degradation = np.abs(temperature - 25) / 1000
        soh = base_soh - np.cumsum(temp_degradation)
        soh = np.clip(soh, 0.5, 1.0)
        
        severity = ['critical' if abs(t) > 50 else 'high' for t in temperature]
        
        return {
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'soc': soc,
            'soh': soh,
            'severity_level': severity
        }
    
    def _generate_high_current_scenario(self, n_points: int) -> Dict[str, np.ndarray]:
        """Generate high current scenario."""
        # High current operation (fast charging/discharging)
        current_magnitude = self.rng.uniform(50, 200, n_points)
        current_direction = self.rng.choice([-1, 1], n_points)
        current = current_magnitude * current_direction
        
        # Voltage drops under high current
        voltage = 3.7 - np.abs(current) * 0.001  # IR drop
        voltage = np.clip(voltage, 2.8, 4.2)
        
        # Temperature rises with high current
        temperature = 25 + np.abs(current) * 0.1
        temperature = np.clip(temperature, 20, 60)
        
        soc = np.cumsum(current) / 1000 + 0.5
        soc = np.clip(soc, 0.05, 0.95)
        
        # SoH degrades with high current stress
        current_stress = np.abs(current) / 100
        soh = 0.95 - np.cumsum(current_stress) / 10000
        soh = np.clip(soh, 0.6, 1.0)
        
        severity = ['critical' if abs(c) > 150 else 'high' if abs(c) > 100 else 'medium' 
                   for c in current]
        
        return {
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'soc': soc,
            'soh': soh,
            'severity_level': severity
        }
    
    def _generate_deep_discharge_scenario(self, n_points: int) -> Dict[str, np.ndarray]:
        """Generate deep discharge scenario."""
        # Deep discharge scenario
        initial_soc = 0.8
        discharge_rate = self.rng.uniform(0.5, 2.0)  # C-rate
        
        soc = np.zeros(n_points)
        soc[0] = initial_soc
        
        for i in range(1, n_points):
            delta_soc = -discharge_rate / (60 * 100)  # Per minute
            soc[i] = max(0.0, soc[i-1] + delta_soc)
        
        # Voltage drops significantly at low SoC
        voltage = 2.8 + (4.2 - 2.8) * soc**0.5
        
        # Current is consistently negative (discharging)
        current = -self.rng.uniform(10, 50, n_points)
        
        temperature = self.rng.normal(25, 3, n_points)
        
        # SoH degrades with deep discharge
        deep_discharge_stress = np.where(soc < 0.2, 0.001, 0.0001)
        soh = 0.95 - np.cumsum(deep_discharge_stress)
        soh = np.clip(soh, 0.7, 1.0)
        
        severity = ['critical' if s < 0.1 else 'high' if s < 0.2 else 'medium' 
                   for s in soc]
        
        return {
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'soc': soc,
            'soh': soh,
            'severity_level': severity
        }
    
    def _generate_overcharge_scenario(self, n_points: int) -> Dict[str, np.ndarray]:
        """Generate overcharge scenario."""
        # Overcharge scenario
        initial_soc = 0.7
        charge_rate = self.rng.uniform(0.5, 1.5)  # C-rate
        
        soc = np.zeros(n_points)
        soc[0] = initial_soc
        
        for i in range(1, n_points):
            delta_soc = charge_rate / (60 * 100)  # Per minute
            soc[i] = min(1.1, soc[i-1] + delta_soc)  # Allow overcharge
        
        # Voltage rises with overcharge
        voltage = 3.0 + 1.5 * soc
        voltage = np.clip(voltage, 3.0, 4.5)  # Allow overvoltage
        
        # Current is consistently positive (charging)
        current = self.rng.uniform(10, 80, n_points)
        
        # Temperature rises with overcharge
        overcharge_heat = np.where(soc > 1.0, (soc - 1.0) * 50, 0)
        temperature = 25 + overcharge_heat + self.rng.normal(0, 2, n_points)
        
        # SoH degrades rapidly with overcharge
        overcharge_stress = np.where(soc > 1.0, 0.01, 0.0001)
        soh = 0.95 - np.cumsum(overcharge_stress)
        soh = np.clip(soh, 0.5, 1.0)
        
        severity = ['critical' if s > 1.05 else 'high' if s > 1.0 else 'medium' 
                   for s in soc]
        
        return {
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'soc': soc,
            'soh': soh,
            'severity_level': severity
        }
    
    def _generate_thermal_runaway_scenario(self, n_points: int) -> Dict[str, np.ndarray]:
        """Generate thermal runaway scenario."""
        # Thermal runaway scenario
        temperature = np.zeros(n_points)
        temperature[0] = 30
        
        # Simulate thermal runaway progression
        runaway_start = self.rng.randint(n_points // 4, n_points // 2)
        
        for i in range(1, n_points):
            if i < runaway_start:
                # Normal heating
                temperature[i] = temperature[i-1] + self.rng.normal(0.1, 0.5)
            else:
                # Thermal runaway - exponential heating
                heat_rate = 1.1 ** (i - runaway_start)
                temperature[i] = temperature[i-1] + heat_rate
            
            temperature[i] = max(20, temperature[i])
        
        # Other parameters respond to thermal runaway
        voltage = 4.2 - (temperature - 30) * 0.01  # Voltage drops with heat
        voltage = np.clip(voltage, 0, 4.2)
        
        current = self.rng.normal(0, 20, n_points)
        
        soc = 0.5 + self.rng.normal(0, 0.1, n_points)
        soc = np.clip(soc, 0, 1)
        
        # SoH drops rapidly during thermal runaway
        thermal_stress = np.where(temperature > 60, 0.1, 0.001)
        soh = 0.95 - np.cumsum(thermal_stress)
        soh = np.clip(soh, 0.1, 1.0)
        
        severity = ['critical' if t > 80 else 'high' if t > 60 else 'medium' if t > 40 else 'low' 
                   for t in temperature]
        
        return {
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'soc': soc,
            'soh': soh,
            'severity_level': severity
        }
    
    def _generate_capacity_fade_scenario(self, n_points: int) -> Dict[str, np.ndarray]:
        """Generate capacity fade scenario."""
        # Long-term capacity fade scenario
        initial_soh = 1.0
        fade_rate = self.rng.uniform(0.001, 0.01)  # Per time step
        
        soh = np.zeros(n_points)
        soh[0] = initial_soh
        
        for i in range(1, n_points):
            # Gradual capacity fade
            soh[i] = soh[i-1] - fade_rate * self.rng.uniform(0.5, 1.5)
            soh[i] = max(0.5, soh[i])
        
        # Other parameters affected by capacity fade
        voltage = 3.7 * soh  # Voltage scales with capacity
        
        current = self.rng.normal(0, 15, n_points)
        
        temperature = self.rng.normal(25, 5, n_points)
        
        # SoC becomes less accurate with capacity fade
        soc = 0.5 + self.rng.normal(0, 0.1 * (1 - soh), n_points)
        soc = np.clip(soc, 0, 1)
        
        severity = ['critical' if h < 0.6 else 'high' if h < 0.7 else 'medium' if h < 0.8 else 'low' 
                   for h in soh]
        
        return {
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'soc': soc,
            'soh': soh,
            'severity_level': severity
        }
    
    def _generate_default_scenario(self, n_points: int) -> Dict[str, np.ndarray]:
        """Generate default scenario for unknown types."""
        return self._generate_normal_operation_scenario(n_points)

class HoldoutDataGenerator(ValidationDataGenerator):
    """
    Generator for holdout datasets ensuring temporal consistency.
    """
    
    def generate_validation_set(self) -> Dict[str, pd.DataFrame]:
        """Generate holdout validation dataset."""
        logger.info("Generating holdout validation dataset")
        
        # This would typically split an existing dataset
        # For demonstration, we'll generate a representative holdout set
        holdout_data = self._generate_representative_holdout()
        
        return {
            'holdout_validation': self._add_validation_metadata(holdout_data, 'holdout')
        }
    
    def _generate_representative_holdout(self) -> pd.DataFrame:
        """Generate representative holdout dataset."""
        # Generate diverse battery scenarios for holdout testing
        n_batteries = 50
        days_per_battery = 30
        
        all_data = []
        
        for battery_id in range(n_batteries):
            battery_data = self._generate_battery_holdout_data(battery_id, days_per_battery)
            all_data.append(battery_data)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        return self._ensure_temporal_consistency(combined_data)
    
    def _generate_battery_holdout_data(self, battery_id: int, duration_days: int) -> pd.DataFrame:
        """Generate holdout data for a single battery."""
        n_points = duration_days * 24 * 6  # 10-minute intervals
        
        timestamps = pd.date_range(
            start='2023-06-01',  # Different time period than training
            periods=n_points,
            freq='10min'
        )
        
        # Generate realistic battery operation
        voltage = self.rng.normal(3.7, 0.15, n_points)
        voltage = np.clip(voltage, 3.0, 4.2)
        
        current = self.rng.normal(0, 20, n_points)
        
        temperature = 25 + 10 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 6))  # Daily cycle
        temperature += self.rng.normal(0, 3, n_points)
        
        soc = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 6))  # Daily cycle
        soc = np.clip(soc, 0.1, 0.9)
        
        soh = np.ones(n_points) * self.rng.uniform(0.8, 0.95)
        
        return pd.DataFrame({
            'battery_id': f'holdout_battery_{battery_id:03d}',
            'timestamp': timestamps,
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'soc': soc,
            'soh': soh
        })

class CrossValidationGenerator(ValidationDataGenerator):
    """
    Generator for time-series aware cross-validation sets.
    """
    
    def generate_validation_set(self) -> Dict[str, pd.DataFrame]:
        """Generate cross-validation datasets."""
        logger.info("Generating cross-validation datasets")
        
        # Generate base dataset for splitting
        base_dataset = self._generate_base_cv_dataset()
        
        # Create time-series splits
        cv_splits = self._create_time_series_splits(base_dataset)
        
        return cv_splits
    
    def _generate_base_cv_dataset(self) -> pd.DataFrame:
        """Generate base dataset for cross-validation splitting."""
        # Generate comprehensive dataset for CV
        n_points = 10000  # Large dataset for multiple splits
        
        timestamps = pd.date_range(
            start='2023-01-01',
            periods=n_points,
            freq='1H'
        )
        
        # Generate diverse patterns
        voltage = 3.7 + 0.3 * np.sin(2 * np.pi * np.arange(n_points) / 168)  # Weekly cycle
        voltage += self.rng.normal(0, 0.1, n_points)
        voltage = np.clip(voltage, 3.0, 4.2)
        
        current = 20 * np.sin(2 * np.pi * np.arange(n_points) / 24)  # Daily cycle
        current += self.rng.normal(0, 10, n_points)
        
        temperature = 25 + 15 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 365))  # Yearly cycle
        temperature += 5 * np.sin(2 * np.pi * np.arange(n_points) / 24)  # Daily cycle
        temperature += self.rng.normal(0, 2, n_points)
        
        soc = 0.5 + 0.4 * np.sin(2 * np.pi * np.arange(n_points) / 24)  # Daily cycle
        soc = np.clip(soc, 0.05, 0.95)
        
        # Gradual SoH degradation
        soh = 1.0 - np.arange(n_points) / n_points * 0.1
        soh += self.rng.normal(0, 0.01, n_points)
        soh = np.clip(soh, 0.7, 1.0)
        
        # Additional features for comprehensive validation
        internal_resistance = 0.1 + np.arange(n_points) / n_points * 0.05  # Gradual increase
        internal_resistance += self.rng.normal(0, 0.005, n_points)
        internal_resistance = np.clip(internal_resistance, 0.05, 0.2)
        
        capacity = 100.0 * soh  # Capacity correlates with SoH
        capacity += self.rng.normal(0, 1, n_points)
        capacity = np.clip(capacity, 70.0, 105.0)
        
        # Environmental factors
        humidity = 50 + 30 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 365))
        humidity += self.rng.normal(0, 5, n_points)
        humidity = np.clip(humidity, 20, 80)
        
        pressure = 1013.25 + self.rng.normal(0, 10, n_points)
        pressure = np.clip(pressure, 980, 1050)
        
        # Usage patterns
        charge_cycles = np.cumsum(self.rng.poisson(0.5, n_points))  # Cumulative charge cycles
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'soc': soc,
            'soh': soh,
            'internal_resistance': internal_resistance,
            'capacity': capacity,
            'humidity': humidity,
            'pressure': pressure,
            'charge_cycles': charge_cycles,
            'battery_id': self.rng.choice(['BAT_001', 'BAT_002', 'BAT_003', 'BAT_004'], n_points),
            'usage_type': self.rng.choice(['normal', 'aggressive', 'conservative'], n_points, p=[0.6, 0.2, 0.2])
        })
        
        return df
    
    def create_time_series_splits(self, data: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time-series aware cross-validation splits.
        
        Args:
            data (pd.DataFrame): Time series data
            n_splits (int): Number of CV splits
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: List of (train_idx, test_idx) tuples
        """
        splits = []
        n_samples = len(data)
        
        # Calculate split sizes
        test_size = n_samples // (n_splits + 1)
        
        for i in range(n_splits):
            # Progressive training set (expanding window)
            train_end = (i + 1) * test_size + i * test_size
            test_start = train_end
            test_end = test_start + test_size
            
            if test_end > n_samples:
                test_end = n_samples
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            if len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        
        return splits
    
    def create_stratified_splits(self, data: pd.DataFrame, target_col: str, 
                               n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create stratified cross-validation splits.
        
        Args:
            data (pd.DataFrame): Dataset
            target_col (str): Target column for stratification
            n_splits (int): Number of CV splits
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: List of (train_idx, test_idx) tuples
        """
        from sklearn.model_selection import StratifiedKFold
        
        # Discretize continuous targets for stratification
        if data[target_col].dtype in ['float64', 'float32']:
            target_discrete = pd.qcut(data[target_col], q=10, labels=False, duplicates='drop')
        else:
            target_discrete = data[target_col]
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        splits = list(skf.split(data, target_discrete))
        
        return splits
    
    def create_group_splits(self, data: pd.DataFrame, group_col: str,
                          n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create group-based cross-validation splits.
        
        Args:
            data (pd.DataFrame): Dataset
            group_col (str): Column defining groups
            n_splits (int): Number of CV splits
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: List of (train_idx, test_idx) tuples
        """
        from sklearn.model_selection import GroupKFold
        
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(data, groups=data[group_col]))
        
        return splits
    
    def validate_splits(self, data: pd.DataFrame, splits: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """
        Validate cross-validation splits for data leakage and balance.
        
        Args:
            data (pd.DataFrame): Original dataset
            splits (List[Tuple[np.ndarray, np.ndarray]]): CV splits
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            'n_splits': len(splits),
            'train_sizes': [],
            'test_sizes': [],
            'train_test_overlap': [],
            'temporal_leakage': False,
            'data_balance': {}
        }
        
        for i, (train_idx, test_idx) in enumerate(splits):
            # Check sizes
            validation_results['train_sizes'].append(len(train_idx))
            validation_results['test_sizes'].append(len(test_idx))
            
            # Check for overlap
            overlap = len(set(train_idx) & set(test_idx))
            validation_results['train_test_overlap'].append(overlap)
            
            # Check temporal leakage (if timestamp column exists)
            if 'timestamp' in data.columns:
                train_max_time = data.iloc[train_idx]['timestamp'].max()
                test_min_time = data.iloc[test_idx]['timestamp'].min()
                
                if train_max_time > test_min_time:
                    validation_results['temporal_leakage'] = True
        
        # Check data balance across splits
        if 'battery_id' in data.columns:
            battery_distribution = []
            for train_idx, test_idx in splits:
                train_batteries = set(data.iloc[train_idx]['battery_id'])
                test_batteries = set(data.iloc[test_idx]['battery_id'])
                battery_distribution.append({
                    'train_batteries': len(train_batteries),
                    'test_batteries': len(test_batteries),
                    'overlap_batteries': len(train_batteries & test_batteries)
                })
            validation_results['data_balance']['battery_distribution'] = battery_distribution
        
        return validation_results
    
    def save_validation_sets(self, output_dir: str = None):
        """Save all validation sets to files."""
        if output_dir is None:
            output_dir = self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each validation set
        for name, dataset in self.validation_sets.items():
            if isinstance(dataset, pd.DataFrame):
                filepath = os.path.join(output_dir, f"{name}.csv")
                dataset.to_csv(filepath, index=False)
                logger.info(f"Saved {name} to {filepath}")
            elif isinstance(dataset, dict):
                # Save complex validation sets as JSON
                filepath = os.path.join(output_dir, f"{name}.json")
                # Convert numpy arrays to lists for JSON serialization
                json_data = self._convert_for_json(dataset)
                with open(filepath, 'w') as f:
                    json.dump(json_data, f, indent=2)
                logger.info(f"Saved {name} to {filepath}")
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and other non-JSON types for serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return obj
    
    def load_validation_sets(self, input_dir: str = None) -> Dict[str, Any]:
        """
        Load validation sets from files.
        
        Args:
            input_dir (str, optional): Directory to load from
            
        Returns:
            Dict[str, Any]: Loaded validation sets
        """
        if input_dir is None:
            input_dir = self.output_dir
        
        validation_sets = {}
        
        # Load CSV files
        csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
        for filepath in csv_files:
            name = os.path.splitext(os.path.basename(filepath))[0]
            validation_sets[name] = pd.read_csv(filepath)
            logger.info(f"Loaded {name} from {filepath}")
        
        # Load JSON files
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        for filepath in json_files:
            name = os.path.splitext(os.path.basename(filepath))[0]
            with open(filepath, 'r') as f:
                validation_sets[name] = json.load(f)
            logger.info(f"Loaded {name} from {filepath}")
        
        return validation_sets
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation sets."""
        summary = {
            'total_sets': len(self.validation_sets),
            'set_details': {},
            'generation_time': time.time() - self.start_time if hasattr(self, 'start_time') else None
        }
        
        for name, dataset in self.validation_sets.items():
            if isinstance(dataset, pd.DataFrame):
                summary['set_details'][name] = {
                    'type': 'DataFrame',
                    'shape': dataset.shape,
                    'columns': list(dataset.columns),
                    'memory_usage': f"{dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                    'date_range': {
                        'start': dataset['timestamp'].min().isoformat() if 'timestamp' in dataset.columns else None,
                        'end': dataset['timestamp'].max().isoformat() if 'timestamp' in dataset.columns else None
                    } if 'timestamp' in dataset.columns else None
                }
            elif isinstance(dataset, dict):
                summary['set_details'][name] = {
                    'type': 'Dictionary',
                    'keys': list(dataset.keys()),
                    'structure': str(type(dataset))
                }
            else:
                summary['set_details'][name] = {
                    'type': str(type(dataset)),
                    'description': 'Custom validation set'
                }
        
        return summary

# Utility functions for validation set management
def create_validation_manager(config: Dict[str, Any] = None) -> ValidationSetManager:
    """
    Create validation set manager with configuration.
    
    Args:
        config (Dict[str, Any], optional): Configuration dictionary
        
    Returns:
        ValidationSetManager: Configured validation manager
    """
    if config is None:
        config = {}
    
    return ValidationSetManager(**config)

def generate_battery_validation_sets(output_dir: str = "./validation_sets",
                                   random_seed: int = 42) -> Dict[str, Any]:
    """
    Generate comprehensive battery validation sets.
    
    Args:
        output_dir (str): Output directory for validation sets
        random_seed (int): Random seed for reproducibility
        
    Returns:
        Dict[str, Any]: Generated validation sets
    """
    manager = ValidationSetManager(output_dir=output_dir, random_seed=random_seed)
    manager.generate_all_sets()
    manager.save_validation_sets()
    
    return manager.get_validation_summary()

def validate_temporal_consistency(data: pd.DataFrame, timestamp_col: str = 'timestamp') -> Dict[str, Any]:
    """
    Validate temporal consistency in time series data.
    
    Args:
        data (pd.DataFrame): Time series data
        timestamp_col (str): Name of timestamp column
        
    Returns:
        Dict[str, Any]: Validation results
    """
    results = {
        'is_sorted': data[timestamp_col].is_monotonic_increasing,
        'has_duplicates': data[timestamp_col].duplicated().any(),
        'missing_timestamps': data[timestamp_col].isnull().sum(),
        'time_gaps': []
    }
    
    # Check for time gaps
    if results['is_sorted']:
        time_diffs = data[timestamp_col].diff().dropna()
        median_diff = time_diffs.median()
        
        # Find gaps larger than 2x median
        large_gaps = time_diffs[time_diffs > 2 * median_diff]
        results['time_gaps'] = large_gaps.tolist()
        results['median_time_diff'] = str(median_diff)
        results['max_time_gap'] = str(time_diffs.max())
    
    return results

def create_holdout_split(data: pd.DataFrame, test_size: float = 0.2,
                        stratify_col: str = None, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create holdout train/test split.
    
    Args:
        data (pd.DataFrame): Dataset to split
        test_size (float): Proportion of test set
        stratify_col (str, optional): Column for stratification
        random_state (int): Random state for reproducibility
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_data, test_data)
    """
    from sklearn.model_selection import train_test_split
    
    if stratify_col and stratify_col in data.columns:
        # Stratified split
        if data[stratify_col].dtype in ['float64', 'float32']:
            stratify_values = pd.qcut(data[stratify_col], q=5, labels=False, duplicates='drop')
        else:
            stratify_values = data[stratify_col]
        
        train_data, test_data = train_test_split(
            data, test_size=test_size, stratify=stratify_values, random_state=random_state
        )
    else:
        # Random split
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state
        )
    
    return train_data, test_data

# Export validation set configurations
VALIDATION_CONFIGS = {
    'battery_health': {
        'target_columns': ['soh', 'capacity'],
        'feature_columns': ['voltage', 'current', 'temperature', 'soc', 'internal_resistance'],
        'cv_strategy': 'time_series',
        'n_splits': 5,
        'test_size': 0.2
    },
    'degradation_forecast': {
        'target_columns': ['soh'],
        'feature_columns': ['voltage', 'current', 'temperature', 'charge_cycles'],
        'cv_strategy': 'group',
        'group_column': 'battery_id',
        'n_splits': 5,
        'test_size': 0.2
    },
    'charging_optimization': {
        'target_columns': ['charging_efficiency', 'charging_time'],
        'feature_columns': ['voltage', 'current', 'temperature', 'soc'],
        'cv_strategy': 'stratified',
        'stratify_column': 'usage_type',
        'n_splits': 5,
        'test_size': 0.2
    }
}

def get_validation_config(model_type: str) -> Dict[str, Any]:
    """
    Get validation configuration for specific model type.
    
    Args:
        model_type (str): Type of model ('battery_health', 'degradation_forecast', etc.)
        
    Returns:
        Dict[str, Any]: Validation configuration
    """
    return VALIDATION_CONFIGS.get(model_type, VALIDATION_CONFIGS['battery_health'])

# Module initialization
logger.info("BatteryMind Training Data Validation Sets module initialized")

# Performance benchmarks for validation sets
VALIDATION_BENCHMARKS = {
    'min_samples_per_split': 1000,
    'max_temporal_gap_hours': 24,
    'min_battery_diversity': 3,
    'max_data_leakage_rate': 0.01,
    'target_cv_score_std': 0.05
}

def benchmark_validation_quality(validation_sets: Dict[str, Any]) -> Dict[str, bool]:
    """
    Benchmark validation set quality against standards.
    
    Args:
        validation_sets (Dict[str, Any]): Validation sets to benchmark
        
    Returns:
        Dict[str, bool]: Benchmark results
    """
    results = {}
    
    for name, dataset in validation_sets.items():
        if isinstance(dataset, pd.DataFrame):
            results[f"{name}_size_check"] = len(dataset) >= VALIDATION_BENCHMARKS['min_samples_per_split']
            
            if 'battery_id' in dataset.columns:
                battery_count = dataset['battery_id'].nunique()
                results[f"{name}_diversity_check"] = battery_count >= VALIDATION_BENCHMARKS['min_battery_diversity']
    
    return results
