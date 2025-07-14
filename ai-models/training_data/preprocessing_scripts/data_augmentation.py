"""
BatteryMind - Data Augmentation

Advanced data augmentation techniques for battery sensor data to improve
model robustness and generalization. Includes physics-aware augmentations
and domain-specific transformations.

Features:
- Time-series specific augmentation techniques
- Physics-aware noise injection
- Temporal transformations (scaling, warping)
- Multi-modal data augmentation
- Synthetic scenario generation
- Adversarial augmentation techniques

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import warnings
from scipy import signal, interpolate
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import random
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AugmentationConfig:
    """
    Configuration for data augmentation pipeline.
    
    Attributes:
        # General settings
        augmentation_factor (float): Factor by which to increase dataset size
        preserve_original (bool): Whether to keep original data in augmented dataset
        random_seed (int): Random seed for reproducibility
        
        # Noise augmentation
        enable_noise_injection (bool): Enable noise injection
        noise_types (List[str]): Types of noise to inject
        noise_intensity_range (Tuple[float, float]): Range of noise intensities
        
        # Temporal augmentation
        enable_temporal_augmentation (bool): Enable temporal transformations
        time_scaling_range (Tuple[float, float]): Range for time scaling
        time_warping_intensity (float): Intensity of time warping
        
        # Physics-aware augmentation
        enable_physics_augmentation (bool): Enable physics-aware augmentations
        temperature_drift_range (Tuple[float, float]): Temperature drift range
        voltage_fluctuation_range (Tuple[float, float]): Voltage fluctuation range
        current_variation_range (Tuple[float, float]): Current variation range
        
        # Synthetic scenario augmentation
        enable_scenario_augmentation (bool): Enable synthetic scenario generation
        scenario_types (List[str]): Types of scenarios to generate
        
        # Advanced augmentation
        enable_adversarial_augmentation (bool): Enable adversarial augmentation
        adversarial_intensity (float): Intensity of adversarial perturbations
        
        # Quality control
        max_augmentation_deviation (float): Maximum allowed deviation from original
        quality_check_enabled (bool): Enable quality checks for augmented data
    """
    # General settings
    augmentation_factor: float = 2.0
    preserve_original: bool = True
    random_seed: int = 42
    
    # Noise augmentation
    enable_noise_injection: bool = True
    noise_types: List[str] = field(default_factory=lambda: [
        'gaussian', 'uniform', 'salt_pepper', 'drift'
    ])
    noise_intensity_range: Tuple[float, float] = (0.01, 0.05)
    
    # Temporal augmentation
    enable_temporal_augmentation: bool = True
    time_scaling_range: Tuple[float, float] = (0.8, 1.2)
    time_warping_intensity: float = 0.1
    
    # Physics-aware augmentation
    enable_physics_augmentation: bool = True
    temperature_drift_range: Tuple[float, float] = (-2.0, 2.0)
    voltage_fluctuation_range: Tuple[float, float] = (-0.05, 0.05)
    current_variation_range: Tuple[float, float] = (-0.1, 0.1)
    
    # Synthetic scenario augmentation
    enable_scenario_augmentation: bool = True
    scenario_types: List[str] = field(default_factory=lambda: [
        'fast_charging', 'slow_charging', 'temperature_stress', 'aging_simulation'
    ])
    
    # Advanced augmentation
    enable_adversarial_augmentation: bool = False
    adversarial_intensity: float = 0.02
    
    # Quality control
    max_augmentation_deviation: float = 0.3
    quality_check_enabled: bool = True

class BaseAugmentation(ABC):
    """Base class for data augmentation techniques."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        
    @abstractmethod
    def augment(self, data: np.ndarray) -> np.ndarray:
        """Apply augmentation to input data."""
        pass
    
    def _validate_augmentation(self, original: np.ndarray, augmented: np.ndarray) -> bool:
        """Validate that augmentation is within acceptable bounds."""
        if not self.config.quality_check_enabled:
            return True
        
        # Check for NaN or infinite values
        if np.any(np.isnan(augmented)) or np.any(np.isinf(augmented)):
            return False
        
        # Check deviation from original
        if len(original) > 0 and len(augmented) > 0:
            relative_deviation = np.abs(augmented - original) / (np.abs(original) + 1e-6)
            max_deviation = np.max(relative_deviation)
            
            if max_deviation > self.config.max_augmentation_deviation:
                return False
        
        return True

class NoiseInjectionAugmentation(BaseAugmentation):
    """Inject various types of noise into sensor data."""
    
    def __init__(self, config: AugmentationConfig):
        super().__init__(config)
        
    def augment(self, data: np.ndarray) -> np.ndarray:
        """Apply noise injection augmentation."""
        augmented_data = data.copy()
        
        # Select random noise type
        noise_type = np.random.choice(self.config.noise_types)
        
        # Select random noise intensity
        noise_intensity = np.random.uniform(*self.config.noise_intensity_range)
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_intensity * np.std(data), data.shape)
            augmented_data += noise
            
        elif noise_type == 'uniform':
            noise_range = noise_intensity * (np.max(data) - np.min(data))
            noise = np.random.uniform(-noise_range/2, noise_range/2, data.shape)
            augmented_data += noise
            
        elif noise_type == 'salt_pepper':
            # Salt and pepper noise
            prob = noise_intensity
            mask = np.random.random(data.shape) < prob
            salt_mask = np.random.random(data.shape) < 0.5
            
            augmented_data[mask & salt_mask] = np.max(data)
            augmented_data[mask & ~salt_mask] = np.min(data)
            
        elif noise_type == 'drift':
            # Gradual drift
            drift = np.linspace(0, noise_intensity * np.std(data), len(data))
            if np.random.random() < 0.5:
                drift = -drift
            augmented_data += drift
        
        # Validate augmentation
        if self._validate_augmentation(data, augmented_data):
            return augmented_data
        else:
            return data
    
    def augment_multimodal(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply noise injection to multi-modal data."""
        augmented_dict = {}
        
        for sensor_name, sensor_data in data_dict.items():
            augmented_dict[sensor_name] = self.augment(sensor_data)
        
        return augmented_dict

class TemporalAugmentation(BaseAugmentation):
    """Apply temporal transformations to time series data."""
    
    def __init__(self, config: AugmentationConfig):
        super().__init__(config)
        
    def augment(self, data: np.ndarray) -> np.ndarray:
        """Apply temporal augmentation."""
        augmentation_type = np.random.choice(['time_scaling', 'time_warping', 'jittering'])
        
        if augmentation_type == 'time_scaling':
            return self._time_scaling(data)
        elif augmentation_type == 'time_warping':
            return self._time_warping(data)
        elif augmentation_type == 'jittering':
            return self._jittering(data)
        else:
            return data
    
    def _time_scaling(self, data: np.ndarray) -> np.ndarray:
        """Apply time scaling (speed up or slow down)."""
        scale_factor = np.random.uniform(*self.config.time_scaling_range)
        
        # Create new time indices
        original_indices = np.arange(len(data))
        new_length = int(len(data) / scale_factor)
        new_indices = np.linspace(0, len(data) - 1, new_length)
        
        # Interpolate data
        interpolator = interpolate.interp1d(original_indices, data, kind='linear', 
                                          bounds_error=False, fill_value='extrapolate')
        scaled_data = interpolator(new_indices)
        
        return scaled_data
    
    def _time_warping(self, data: np.ndarray) -> np.ndarray:
        """Apply time warping with smooth deformation."""
        # Generate smooth warping function
        warp_intensity = self.config.time_warping_intensity
        n_points = max(3, len(data) // 10)
        
        # Create control points for warping
        control_points = np.linspace(0, len(data) - 1, n_points)
        warp_offsets = np.random.uniform(-warp_intensity * len(data), 
                                       warp_intensity * len(data), n_points)
        
        # Ensure monotonic warping
        warp_offsets = np.cumsum(warp_offsets - np.mean(warp_offsets))
        warped_points = control_points + warp_offsets
        warped_points = np.clip(warped_points, 0, len(data) - 1)
        
        # Interpolate warping function
        original_indices = np.arange(len(data))
        interpolator = interpolate.interp1d(control_points, warped_points, 
                                          kind='cubic', bounds_error=False, 
                                          fill_value='extrapolate')
        warped_indices = interpolator(original_indices)
        warped_indices = np.clip(warped_indices, 0, len(data) - 1)
        
        # Apply warping
        data_interpolator = interpolate.interp1d(original_indices, data, 
                                               kind='linear', bounds_error=False, 
                                               fill_value='extrapolate')
        warped_data = data_interpolator(warped_indices)
        
        return warped_data
    
    def _jittering(self, data: np.ndarray) -> np.ndarray:
        """Apply random jittering to data points."""
        jitter_intensity = 0.01 * np.std(data)
        jitter = np.random.normal(0, jitter_intensity, data.shape)
        
        return data + jitter

class PhysicsAwareAugmentation(BaseAugmentation):
    """Apply physics-aware augmentations specific to battery data."""
    
    def __init__(self, config: AugmentationConfig):
        super().__init__(config)
        
    def augment_multimodal(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply physics-aware augmentation to multi-modal battery data."""
        augmented_dict = copy.deepcopy(data_dict)
        
        # Apply correlated changes to maintain physical consistency
        augmentation_type = np.random.choice([
            'temperature_effect', 'aging_effect', 'load_variation', 'environmental_change'
        ])
        
        if augmentation_type == 'temperature_effect':
            augmented_dict = self._simulate_temperature_effect(augmented_dict)
        elif augmentation_type == 'aging_effect':
            augmented_dict = self._simulate_aging_effect(augmented_dict)
        elif augmentation_type == 'load_variation':
            augmented_dict = self._simulate_load_variation(augmented_dict)
        elif augmentation_type == 'environmental_change':
            augmented_dict = self._simulate_environmental_change(augmented_dict)
        
        return augmented_dict
    
    def _simulate_temperature_effect(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Simulate temperature effects on battery parameters."""
        temp_change = np.random.uniform(*self.config.temperature_drift_range)
        
        # Temperature directly affects temperature sensor
        if 'temperature' in data_dict:
            data_dict['temperature'] += temp_change
        
        # Temperature affects voltage (negative temperature coefficient)
        if 'voltage' in data_dict:
            voltage_change = -0.003 * temp_change  # Typical -3mV/°C
            data_dict['voltage'] += voltage_change
        
        # Temperature affects internal resistance
        if 'current' in data_dict and 'voltage' in data_dict:
            # Simulate resistance change effect on voltage under load
            resistance_factor = 1 + 0.01 * temp_change  # 1% per °C
            current = data_dict['current']
            voltage_drop_change = current * 0.1 * (resistance_factor - 1)
            data_dict['voltage'] -= voltage_drop_change
        
        return data_dict
    
    def _simulate_aging_effect(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Simulate battery aging effects."""
        aging_factor = np.random.uniform(0.95, 1.0)  # 0-5% capacity loss
        
        # Aging affects capacity (reflected in SoC changes)
        if 'soc' in data_dict:
            # Simulate slightly faster SoC changes due to capacity loss
            soc_changes = np.diff(data_dict['soc'])
            soc_changes *= (1 / aging_factor)
            data_dict['soc'][1:] = data_dict['soc'][0] + np.cumsum(soc_changes)
            data_dict['soc'] = np.clip(data_dict['soc'], 0, 1)
        
        # Aging increases internal resistance
        if 'voltage' in data_dict and 'current' in data_dict:
            resistance_increase = (1 - aging_factor) * 0.5  # 50% of capacity loss
            voltage_drop = data_dict['current'] * 0.1 * resistance_increase
            data_dict['voltage'] -= voltage_drop
        
        return data_dict
    
    def _simulate_load_variation(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Simulate load variation effects."""
        load_factor = np.random.uniform(0.8, 1.2)
        
        # Scale current
        if 'current' in data_dict:
            data_dict['current'] *= load_factor
        
        # Adjust voltage based on load change
        if 'voltage' in data_dict and 'current' in data_dict:
            # Higher load -> lower voltage due to internal resistance
            voltage_change = -0.1 * (load_factor - 1) * np.abs(data_dict['current'])
            data_dict['voltage'] += voltage_change
        
        # Adjust temperature based on load
        if 'temperature' in data_dict:
            temp_change = 2.0 * (load_factor - 1)  # 2°C per 100% load change
            data_dict['temperature'] += temp_change
        
        return data_dict
    
    def _simulate_environmental_change(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Simulate environmental condition changes."""
        # Ambient temperature change
        ambient_change = np.random.uniform(-5.0, 5.0)
        
        if 'temperature' in data_dict:
            # Battery temperature follows ambient with some lag
            thermal_coupling = 0.3  # 30% coupling to ambient
            data_dict['temperature'] += ambient_change * thermal_coupling
        
        # Humidity effects (minor)
        if 'voltage' in data_dict:
            humidity_effect = np.random.uniform(-0.01, 0.01)
            data_dict['voltage'] += humidity_effect
        
        return data_dict

class ScenarioAugmentation(BaseAugmentation):
    """Generate synthetic scenarios for battery operation."""
    
    def __init__(self, config: AugmentationConfig):
        super().__init__(config)
        
    def generate_scenario(self, base_data: Dict[str, np.ndarray], 
                         scenario_type: str) -> Dict[str, np.ndarray]:
        """Generate a specific scenario based on base data."""
        if scenario_type == 'fast_charging':
            return self._generate_fast_charging_scenario(base_data)
        elif scenario_type == 'slow_charging':
            return self._generate_slow_charging_scenario(base_data)
        elif scenario_type == 'temperature_stress':
            return self._generate_temperature_stress_scenario(base_data)
        elif scenario_type == 'aging_simulation':
            return self._generate_aging_scenario(base_data)
        else:
            return base_data
    
    def _generate_fast_charging_scenario(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate fast charging scenario."""
        scenario_data = copy.deepcopy(data)
        
        if 'current' in scenario_data:
            # Increase charging current
            charging_mask = scenario_data['current'] > 0
            scenario_data['current'][charging_mask] *= np.random.uniform(1.5, 2.0)
        
        if 'temperature' in scenario_data:
            # Higher temperature due to fast charging
            temp_increase = np.random.uniform(5.0, 15.0)
            scenario_data['temperature'] += temp_increase
        
        if 'voltage' in scenario_data and 'current' in scenario_data:
            # Voltage drop due to higher current
            voltage_drop = scenario_data['current'] * 0.05  # Simplified
            scenario_data['voltage'] -= voltage_drop
        
        return scenario_data
    
    def _generate_slow_charging_scenario(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate slow charging scenario."""
        scenario_data = copy.deepcopy(data)
        
        if 'current' in scenario_data:
            # Reduce charging current
            charging_mask = scenario_data['current'] > 0
            scenario_data['current'][charging_mask] *= np.random.uniform(0.3, 0.6)
        
        if 'temperature' in scenario_data:
            # Lower temperature due to slow charging
            temp_decrease = np.random.uniform(2.0, 5.0)
            scenario_data['temperature'] -= temp_decrease
        
        return scenario_data
    
    def _generate_temperature_stress_scenario(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate temperature stress scenario."""
        scenario_data = copy.deepcopy(data)
        
        stress_type = np.random.choice(['hot', 'cold'])
        
        if 'temperature' in scenario_data:
            if stress_type == 'hot':
                temp_stress = np.random.uniform(15.0, 25.0)
                scenario_data['temperature'] += temp_stress
            else:
                temp_stress = np.random.uniform(-15.0, -5.0)
                scenario_data['temperature'] += temp_stress
        
        # Temperature affects other parameters
        if 'voltage' in scenario_data:
            temp_coeff = -0.003 if stress_type == 'hot' else 0.003
            voltage_change = temp_coeff * abs(temp_stress)
            scenario_data['voltage'] += voltage_change
        
        return scenario_data
    
    def _generate_aging_scenario(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate aging simulation scenario."""
        scenario_data = copy.deepcopy(data)
        
        # Simulate different aging levels
        aging_level = np.random.uniform(0.1, 0.3)  # 10-30% degradation
        
        if 'voltage' in scenario_data and 'current' in scenario_data:
            # Increased internal resistance
            resistance_increase = aging_level * 0.1  # Ohms
            voltage_drop = scenario_data['current'] * resistance_increase
            scenario_data['voltage'] -= voltage_drop
        
        if 'soc' in scenario_data:
            # Capacity fade simulation
            capacity_factor = 1 - aging_level
            # Adjust SoC changes to reflect capacity loss
            soc_changes = np.diff(scenario_data['soc'])
            soc_changes /= capacity_factor
            scenario_data['soc'][1:] = scenario_data['soc'][0] + np.cumsum(soc_changes)
            scenario_data['soc'] = np.clip(scenario_data['soc'], 0, 1)
        
        return scenario_data

class AdversarialAugmentation(BaseAugmentation):
    """Apply adversarial augmentation techniques."""
    
    def __init__(self, config: AugmentationConfig):
        super().__init__(config)
        
    def augment(self, data: np.ndarray, gradient: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply adversarial augmentation."""
        if gradient is not None:
            # Gradient-based adversarial perturbation
            perturbation = self.config.adversarial_intensity * np.sign(gradient)
        else:
            # Random adversarial perturbation
            perturbation = np.random.uniform(
                -self.config.adversarial_intensity,
                self.config.adversarial_intensity,
                data.shape
            )
        
        augmented_data = data + perturbation
        
        if self._validate_augmentation(data, augmented_data):
            return augmented_data
        else:
            return data

class BatteryDataAugmentor:
    """
    Main data augmentation pipeline for battery sensor data.
    """
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.augmentations = {}
        
        # Set random seed
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
        
        # Initialize augmentation techniques
        self._initialize_augmentations()
        
        # Statistics tracking
        self.augmentation_stats = {
            'total_samples_processed': 0,
            'successful_augmentations': 0,
            'failed_augmentations': 0,
            'augmentation_counts': defaultdict(int)
        }
        
        logger.info(f"BatteryDataAugmentor initialized with {len(self.augmentations)} techniques")
    
    def _initialize_augmentations(self):
        """Initialize all augmentation techniques based on configuration."""
        if self.config.enable_noise_injection:
            self.augmentations['noise'] = NoiseInjection(self.config)
        
        if self.config.enable_time_warping:
            self.augmentations['time_warp'] = TimeWarping(self.config)
        
        if self.config.enable_magnitude_scaling:
            self.augmentations['magnitude'] = MagnitudeScaling(self.config)
        
        if self.config.enable_time_shifting:
            self.augmentations['time_shift'] = TimeShifting(self.config)
        
        if self.config.enable_frequency_masking:
            self.augmentations['freq_mask'] = FrequencyMasking(self.config)
        
        if self.config.enable_mixup:
            self.augmentations['mixup'] = MixupAugmentation(self.config)
        
        if self.config.enable_cutmix:
            self.augmentations['cutmix'] = CutMixAugmentation(self.config)
        
        if self.config.enable_adversarial:
            self.augmentations['adversarial'] = AdversarialAugmentation(self.config)
    
    def augment_batch(self, data: np.ndarray, labels: Optional[np.ndarray] = None,
                     metadata: Optional[Dict] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        """
        Apply augmentation to a batch of data.
        
        Args:
            data (np.ndarray): Input data batch [batch_size, sequence_length, features]
            labels (np.ndarray, optional): Corresponding labels
            metadata (Dict, optional): Additional metadata
            
        Returns:
            Tuple containing augmented data, labels, and augmentation info
        """
        batch_size = data.shape[0]
        augmented_data = data.copy()
        augmented_labels = labels.copy() if labels is not None else None
        augmentation_info = {
            'applied_augmentations': [],
            'augmentation_parameters': {},
            'original_batch_size': batch_size
        }
        
        # Apply augmentations based on probability
        for aug_name, augmentation in self.augmentations.items():
            if np.random.random() < self.config.augmentation_probability:
                try:
                    # Apply augmentation to each sample in batch
                    for i in range(batch_size):
                        if aug_name in ['mixup', 'cutmix']:
                            # These require multiple samples
                            if i < batch_size - 1:
                                augmented_data[i], mix_info = self._apply_mixing_augmentation(
                                    augmented_data[i], augmented_data[i + 1], aug_name
                                )
                                if augmented_labels is not None:
                                    augmented_labels[i] = self._mix_labels(
                                        augmented_labels[i], augmented_labels[i + 1], mix_info
                                    )
                        else:
                            # Standard augmentations
                            augmented_data[i] = augmentation.augment(augmented_data[i])
                    
                    augmentation_info['applied_augmentations'].append(aug_name)
                    self.augmentation_stats['augmentation_counts'][aug_name] += 1
                    self.augmentation_stats['successful_augmentations'] += 1
                    
                except Exception as e:
                    logger.warning(f"Augmentation {aug_name} failed: {e}")
                    self.augmentation_stats['failed_augmentations'] += 1
        
        self.augmentation_stats['total_samples_processed'] += batch_size
        
        return augmented_data, augmented_labels, augmentation_info
    
    def _apply_mixing_augmentation(self, sample1: np.ndarray, sample2: np.ndarray, 
                                 aug_type: str) -> Tuple[np.ndarray, Dict]:
        """Apply mixing-based augmentations."""
        if aug_type == 'mixup':
            return self.augmentations['mixup'].augment(sample1, sample2)
        elif aug_type == 'cutmix':
            return self.augmentations['cutmix'].augment(sample1, sample2)
        else:
            return sample1, {}
    
    def _mix_labels(self, label1: np.ndarray, label2: np.ndarray, mix_info: Dict) -> np.ndarray:
        """Mix labels according to mixing parameters."""
        if 'lambda' in mix_info:
            # Mixup-style label mixing
            return mix_info['lambda'] * label1 + (1 - mix_info['lambda']) * label2
        elif 'mask' in mix_info:
            # CutMix-style label mixing
            mask_ratio = np.mean(mix_info['mask'])
            return mask_ratio * label1 + (1 - mask_ratio) * label2
        else:
            return label1
    
    def augment_dataset(self, dataset: Dict[str, np.ndarray], 
                       augmentation_factor: int = 2) -> Dict[str, np.ndarray]:
        """
        Augment entire dataset by specified factor.
        
        Args:
            dataset (Dict[str, np.ndarray]): Dataset with 'data' and optionally 'labels'
            augmentation_factor (int): Factor by which to increase dataset size
            
        Returns:
            Dict[str, np.ndarray]: Augmented dataset
        """
        original_data = dataset['data']
        original_labels = dataset.get('labels', None)
        original_size = original_data.shape[0]
        
        # Calculate number of augmented samples needed
        target_size = original_size * augmentation_factor
        augmented_samples_needed = target_size - original_size
        
        # Prepare augmented data storage
        augmented_data_list = [original_data]
        if original_labels is not None:
            augmented_labels_list = [original_labels]
        
        # Generate augmented samples
        samples_generated = 0
        while samples_generated < augmented_samples_needed:
            # Randomly select samples to augment
            batch_size = min(self.config.batch_size, augmented_samples_needed - samples_generated)
            indices = np.random.choice(original_size, batch_size, replace=True)
            
            batch_data = original_data[indices]
            batch_labels = original_labels[indices] if original_labels is not None else None
            
            # Apply augmentation
            aug_data, aug_labels, _ = self.augment_batch(batch_data, batch_labels)
            
            augmented_data_list.append(aug_data)
            if aug_labels is not None:
                augmented_labels_list.append(aug_labels)
            
            samples_generated += batch_size
        
        # Combine all data
        final_data = np.concatenate(augmented_data_list, axis=0)
        result = {'data': final_data}
        
        if original_labels is not None:
            final_labels = np.concatenate(augmented_labels_list, axis=0)
            result['labels'] = final_labels
        
        logger.info(f"Dataset augmented from {original_size} to {final_data.shape[0]} samples")
        
        return result
    
    def create_augmentation_pipeline(self, augmentation_sequence: List[str]) -> callable:
        """
        Create a custom augmentation pipeline with specified sequence.
        
        Args:
            augmentation_sequence (List[str]): Sequence of augmentation names
            
        Returns:
            callable: Augmentation pipeline function
        """
        def pipeline(data: np.ndarray) -> np.ndarray:
            augmented = data.copy()
            for aug_name in augmentation_sequence:
                if aug_name in self.augmentations:
                    augmented = self.augmentations[aug_name].augment(augmented)
                else:
                    logger.warning(f"Augmentation '{aug_name}' not available")
            return augmented
        
        return pipeline
    
    def validate_augmentation_quality(self, original_data: np.ndarray, 
                                    augmented_data: np.ndarray) -> Dict[str, float]:
        """
        Validate the quality of augmented data.
        
        Args:
            original_data (np.ndarray): Original data
            augmented_data (np.ndarray): Augmented data
            
        Returns:
            Dict[str, float]: Quality metrics
        """
        metrics = {}
        
        # Statistical similarity
        metrics['mean_difference'] = np.abs(np.mean(augmented_data) - np.mean(original_data))
        metrics['std_difference'] = np.abs(np.std(augmented_data) - np.std(original_data))
        
        # Correlation with original
        if original_data.ndim == 2:
            correlations = []
            for i in range(original_data.shape[1]):
                corr = np.corrcoef(original_data[:, i], augmented_data[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            metrics['average_correlation'] = np.mean(correlations) if correlations else 0.0
        
        # Signal-to-noise ratio
        signal_power = np.mean(original_data ** 2)
        noise_power = np.mean((augmented_data - original_data) ** 2)
        metrics['snr_db'] = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Diversity measure
        metrics['diversity_score'] = np.mean(np.std(augmented_data, axis=0))
        
        return metrics
    
    def get_augmentation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive augmentation statistics."""
        stats = self.augmentation_stats.copy()
        
        # Calculate success rate
        total_attempts = stats['successful_augmentations'] + stats['failed_augmentations']
        stats['success_rate'] = stats['successful_augmentations'] / max(total_attempts, 1)
        
        # Calculate augmentation distribution
        total_augmentations = sum(stats['augmentation_counts'].values())
        stats['augmentation_distribution'] = {
            name: count / max(total_augmentations, 1) 
            for name, count in stats['augmentation_counts'].items()
        }
        
        return stats
    
    def save_augmentation_config(self, filepath: str):
        """Save augmentation configuration and statistics."""
        save_data = {
            'config': self.config.__dict__,
            'statistics': self.get_augmentation_statistics(),
            'available_augmentations': list(self.augmentations.keys()),
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        logger.info(f"Augmentation configuration saved to {filepath}")
    
    def load_augmentation_config(self, filepath: str):
        """Load augmentation configuration."""
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        # Update configuration
        config_dict = save_data['config']
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Reinitialize augmentations
        self._initialize_augmentations()
        
        logger.info(f"Augmentation configuration loaded from {filepath}")

# Utility functions for battery-specific augmentation
def create_battery_augmentation_config(scenario: str = "default") -> AugmentationConfig:
    """
    Create battery-specific augmentation configuration.
    
    Args:
        scenario (str): Augmentation scenario ("conservative", "default", "aggressive")
        
    Returns:
        AugmentationConfig: Configured augmentation settings
    """
    if scenario == "conservative":
        return AugmentationConfig(
            noise_std=0.001,
            time_warp_sigma=0.1,
            magnitude_scale_range=(0.95, 1.05),
            time_shift_range=(-2, 2),
            augmentation_probability=0.3,
            preserve_battery_physics=True
        )
    elif scenario == "aggressive":
        return AugmentationConfig(
            noise_std=0.01,
            time_warp_sigma=0.3,
            magnitude_scale_range=(0.8, 1.2),
            time_shift_range=(-10, 10),
            augmentation_probability=0.7,
            preserve_battery_physics=False
        )
    else:  # default
        return AugmentationConfig()

def validate_battery_data_augmentation(original_data: np.ndarray, 
                                     augmented_data: np.ndarray,
                                     feature_names: List[str]) -> Dict[str, bool]:
    """
    Validate that augmented battery data maintains physical constraints.
    
    Args:
        original_data (np.ndarray): Original battery sensor data
        augmented_data (np.ndarray): Augmented battery sensor data
        feature_names (List[str]): Names of features (e.g., 'voltage', 'current', 'temperature')
        
    Returns:
        Dict[str, bool]: Validation results for each constraint
    """
    validation_results = {}
    
    for i, feature_name in enumerate(feature_names):
        if feature_name.lower() == 'voltage':
            # Voltage should be within reasonable bounds
            min_voltage, max_voltage = 2.0, 5.0  # Typical Li-ion range
            voltage_valid = np.all((augmented_data[:, i] >= min_voltage) & 
                                 (augmented_data[:, i] <= max_voltage))
            validation_results['voltage_bounds'] = voltage_valid
            
        elif feature_name.lower() == 'temperature':
            # Temperature should be within operational range
            min_temp, max_temp = -40, 80  # Celsius
            temp_valid = np.all((augmented_data[:, i] >= min_temp) & 
                              (augmented_data[:, i] <= max_temp))
            validation_results['temperature_bounds'] = temp_valid
            
        elif feature_name.lower() == 'current':
            # Current should not exceed physical limits
            max_current = 1000.0  # Amperes
            current_valid = np.all(np.abs(augmented_data[:, i]) <= max_current)
            validation_results['current_bounds'] = current_valid
    
    # Check for NaN or infinite values
    validation_results['no_invalid_values'] = np.all(np.isfinite(augmented_data))
    
    # Check temporal consistency (if time series)
    if augmented_data.shape[0] > 1:
        max_change_rate = np.max(np.abs(np.diff(augmented_data, axis=0)))
        validation_results['temporal_consistency'] = max_change_rate < 100.0  # Reasonable change rate
    
    return validation_results

def create_battery_specific_augmentor(battery_type: str = "li_ion") -> BatteryDataAugmentor:
    """
    Create augmentor specifically configured for battery type.
    
    Args:
        battery_type (str): Type of battery ("li_ion", "lead_acid", "ni_mh")
        
    Returns:
        BatteryDataAugmentor: Configured augmentor
    """
    if battery_type == "li_ion":
        config = AugmentationConfig(
            voltage_bounds=(2.5, 4.2),
            temperature_bounds=(-20, 60),
            current_bounds=(-200, 200),
            preserve_battery_physics=True,
            enable_frequency_masking=True,  # Good for Li-ion impedance data
            enable_mixup=True  # Helps with capacity estimation
        )
    elif battery_type == "lead_acid":
        config = AugmentationConfig(
            voltage_bounds=(1.8, 2.4),
            temperature_bounds=(-40, 50),
            current_bounds=(-100, 100),
            preserve_battery_physics=True,
            enable_time_warping=True,  # Good for lead-acid discharge curves
            enable_magnitude_scaling=True
        )
    else:  # Default to li_ion
        config = create_battery_augmentation_config("default")
    
    return BatteryDataAugmentor(config)

# Export main classes and functions
__all__ = [
    'AugmentationConfig',
    'BaseAugmentation',
    'NoiseInjection',
    'TimeWarping',
    'MagnitudeScaling',
    'TimeShifting',
    'FrequencyMasking',
    'MixupAugmentation',
    'CutMixAugmentation',
    'AdversarialAugmentation',
    'BatteryDataAugmentor',
    'create_battery_augmentation_config',
    'validate_battery_data_augmentation',
    'create_battery_specific_augmentor'
]

# Module initialization
logger.info("BatteryMind Data Augmentation module v1.0.0 initialized")

# Self-test function
def run_augmentation_self_test():
    """Run basic self-test of augmentation functionality."""
    try:
        # Create test data
        test_data = np.random.randn(100, 50, 5)  # 100 samples, 50 time steps, 5 features
        test_labels = np.random.randn(100, 3)    # 100 samples, 3 output features
        
        # Create augmentor
        config = create_battery_augmentation_config("default")
        augmentor = BatteryDataAugmentor(config)
        
        # Test batch augmentation
        aug_data, aug_labels, info = augmentor.augment_batch(test_data[:10], test_labels[:10])
        
        # Validate results
        assert aug_data.shape == test_data[:10].shape, "Augmented data shape mismatch"
        assert aug_labels.shape == test_labels[:10].shape, "Augmented labels shape mismatch"
        assert len(info['applied_augmentations']) >= 0, "Augmentation info missing"
        
        logger.info("Data augmentation self-test passed")
        return True
        
    except Exception as e:
        logger.error(f"Data augmentation self-test failed: {e}")
        return False

# Run self-test on import
if __name__ != "__main__":
    run_augmentation_self_test()
