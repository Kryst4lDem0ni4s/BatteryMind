"""
BatteryMind - Noise Mechanisms for Privacy-Preserving Federated Learning

Advanced noise mechanisms for differential privacy in federated learning
with specialized implementations for battery data privacy protection.

Features:
- Gaussian noise mechanisms with adaptive variance
- Laplace noise for sensitivity-based privacy
- Exponential mechanisms for categorical data
- Composition-aware privacy accounting
- Battery-specific noise calibration
- Utility-privacy trade-off optimization

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import math
from abc import ABC, abstractmethod
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PrivacyBudget:
    """
    Privacy budget tracking for differential privacy mechanisms.
    
    Attributes:
        epsilon (float): Privacy parameter epsilon
        delta (float): Privacy parameter delta
        composition_method (str): Method for privacy composition
        max_epsilon (float): Maximum allowed epsilon
        max_delta (float): Maximum allowed delta
        spent_epsilon (float): Already spent epsilon
        spent_delta (float): Already spent delta
    """
    epsilon: float = 1.0
    delta: float = 1e-5
    composition_method: str = "advanced"
    max_epsilon: float = 10.0
    max_delta: float = 1e-3
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    
    def can_spend(self, epsilon: float, delta: float) -> bool:
        """Check if privacy budget allows spending epsilon and delta."""
        return (self.spent_epsilon + epsilon <= self.max_epsilon and 
                self.spent_delta + delta <= self.max_delta)
    
    def spend(self, epsilon: float, delta: float) -> None:
        """Spend privacy budget."""
        if not self.can_spend(epsilon, delta):
            raise ValueError("Insufficient privacy budget")
        
        if self.composition_method == "basic":
            self.spent_epsilon += epsilon
            self.spent_delta += delta
        elif self.composition_method == "advanced":
            # Advanced composition theorem
            self.spent_epsilon = math.sqrt(2 * math.log(1.25 / delta)) * math.sqrt(
                self.spent_epsilon**2 + epsilon**2
            )
            self.spent_delta = self.spent_delta + delta
        else:
            raise ValueError(f"Unknown composition method: {self.composition_method}")
    
    def remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget."""
        return (self.max_epsilon - self.spent_epsilon, 
                self.max_delta - self.spent_delta)

@dataclass
class NoiseParameters:
    """
    Parameters for noise mechanisms.
    """
    mechanism_type: str = "gaussian"
    sensitivity: float = 1.0
    epsilon: float = 1.0
    delta: float = 1e-5
    clipping_bound: float = 1.0
    adaptive_clipping: bool = True
    noise_multiplier: float = 1.1
    temperature: float = 1.0  # For exponential mechanism

class NoiseInterface(ABC):
    """
    Abstract interface for noise mechanisms.
    """
    
    @abstractmethod
    def add_noise(self, data: torch.Tensor, 
                  sensitivity: float, 
                  privacy_budget: PrivacyBudget) -> torch.Tensor:
        """Add noise to data for differential privacy."""
        pass
    
    @abstractmethod
    def calibrate_noise(self, sensitivity: float, 
                       epsilon: float, 
                       delta: float) -> float:
        """Calibrate noise scale for given privacy parameters."""
        pass
    
    @abstractmethod
    def privacy_analysis(self, noise_scale: float, 
                        sensitivity: float) -> Tuple[float, float]:
        """Analyze privacy guarantees for given noise scale."""
        pass

class GaussianNoiseMechanism(NoiseInterface):
    """
    Gaussian noise mechanism for differential privacy.
    
    Provides (ε, δ)-differential privacy by adding Gaussian noise
    calibrated to the L2 sensitivity of the function.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.noise_history = []
    
    def add_noise(self, data: torch.Tensor, 
                  sensitivity: float, 
                  privacy_budget: PrivacyBudget) -> torch.Tensor:
        """
        Add Gaussian noise to data.
        
        Args:
            data (torch.Tensor): Input data tensor
            sensitivity (float): L2 sensitivity of the function
            privacy_budget (PrivacyBudget): Privacy budget parameters
            
        Returns:
            torch.Tensor: Noisy data tensor
        """
        # Calibrate noise scale
        noise_scale = self.calibrate_noise(
            sensitivity, privacy_budget.epsilon, privacy_budget.delta
        )
        
        # Generate Gaussian noise
        noise = torch.normal(
            mean=0.0, 
            std=noise_scale, 
            size=data.shape, 
            device=data.device
        )
        
        # Add noise to data
        noisy_data = data + noise
        
        # Track noise for analysis
        self.noise_history.append({
            'noise_scale': noise_scale,
            'sensitivity': sensitivity,
            'epsilon': privacy_budget.epsilon,
            'delta': privacy_budget.delta,
            'data_shape': data.shape
        })
        
        return noisy_data
    
    def calibrate_noise(self, sensitivity: float, 
                       epsilon: float, 
                       delta: float) -> float:
        """
        Calibrate Gaussian noise scale for (ε, δ)-DP.
        
        Uses the formula: σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        
        Args:
            sensitivity (float): L2 sensitivity
            epsilon (float): Privacy parameter ε
            delta (float): Privacy parameter δ
            
        Returns:
            float: Noise scale (standard deviation)
        """
        if epsilon <= 0 or delta <= 0 or delta >= 1:
            raise ValueError("Invalid privacy parameters")
        
        noise_scale = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        return noise_scale
    
    def privacy_analysis(self, noise_scale: float, 
                        sensitivity: float) -> Tuple[float, float]:
        """
        Analyze privacy guarantees for given noise scale.
        
        Args:
            noise_scale (float): Standard deviation of noise
            sensitivity (float): L2 sensitivity
            
        Returns:
            Tuple[float, float]: (epsilon, delta) privacy parameters
        """
        # For Gaussian mechanism: ε = sensitivity * sqrt(2 * ln(1.25/δ)) / σ
        # Solve for ε given δ = 1e-5
        delta = 1e-5
        epsilon = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / noise_scale
        
        return epsilon, delta
    
    def adaptive_clipping(self, gradients: List[torch.Tensor], 
                         target_quantile: float = 0.5) -> float:
        """
        Adaptive gradient clipping based on quantile estimation.
        
        Args:
            gradients (List[torch.Tensor]): List of gradient tensors
            target_quantile (float): Target quantile for clipping
            
        Returns:
            float: Adaptive clipping bound
        """
        # Compute L2 norms of gradients
        norms = []
        for grad in gradients:
            if grad is not None:
                norm = torch.norm(grad, p=2).item()
                norms.append(norm)
        
        if not norms:
            return 1.0
        
        # Estimate quantile
        clipping_bound = np.quantile(norms, target_quantile)
        
        return max(clipping_bound, 0.1)  # Minimum clipping bound

class LaplaceNoiseMechanism(NoiseInterface):
    """
    Laplace noise mechanism for differential privacy.
    
    Provides ε-differential privacy by adding Laplace noise
    calibrated to the L1 sensitivity of the function.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.noise_history = []
    
    def add_noise(self, data: torch.Tensor, 
                  sensitivity: float, 
                  privacy_budget: PrivacyBudget) -> torch.Tensor:
        """
        Add Laplace noise to data.
        
        Args:
            data (torch.Tensor): Input data tensor
            sensitivity (float): L1 sensitivity of the function
            privacy_budget (PrivacyBudget): Privacy budget parameters
            
        Returns:
            torch.Tensor: Noisy data tensor
        """
        # Calibrate noise scale
        noise_scale = self.calibrate_noise(
            sensitivity, privacy_budget.epsilon, 0.0  # Laplace doesn't use delta
        )
        
        # Generate Laplace noise using transformation method
        uniform_noise = torch.rand(data.shape, device=data.device) - 0.5
        laplace_noise = noise_scale * torch.sign(uniform_noise) * torch.log(
            1 - 2 * torch.abs(uniform_noise)
        )
        
        # Add noise to data
        noisy_data = data + laplace_noise
        
        # Track noise for analysis
        self.noise_history.append({
            'noise_scale': noise_scale,
            'sensitivity': sensitivity,
            'epsilon': privacy_budget.epsilon,
            'data_shape': data.shape
        })
        
        return noisy_data
    
    def calibrate_noise(self, sensitivity: float, 
                       epsilon: float, 
                       delta: float = 0.0) -> float:
        """
        Calibrate Laplace noise scale for ε-DP.
        
        Uses the formula: b = sensitivity / ε
        
        Args:
            sensitivity (float): L1 sensitivity
            epsilon (float): Privacy parameter ε
            delta (float): Ignored for Laplace mechanism
            
        Returns:
            float: Noise scale (scale parameter)
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        noise_scale = sensitivity / epsilon
        return noise_scale
    
    def privacy_analysis(self, noise_scale: float, 
                        sensitivity: float) -> Tuple[float, float]:
        """
        Analyze privacy guarantees for given noise scale.
        
        Args:
            noise_scale (float): Scale parameter of Laplace noise
            sensitivity (float): L1 sensitivity
            
        Returns:
            Tuple[float, float]: (epsilon, 0) privacy parameters
        """
        epsilon = sensitivity / noise_scale
        return epsilon, 0.0

class ExponentialMechanism(NoiseInterface):
    """
    Exponential mechanism for differential privacy on categorical data.
    
    Provides ε-differential privacy by sampling from an exponential
    distribution over the output space.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.selection_history = []
    
    def add_noise(self, data: torch.Tensor, 
                  sensitivity: float, 
                  privacy_budget: PrivacyBudget) -> torch.Tensor:
        """
        Apply exponential mechanism to select output.
        
        Note: This is a simplified implementation for demonstration.
        Real exponential mechanism requires a utility function.
        
        Args:
            data (torch.Tensor): Input data (utility scores)
            sensitivity (float): Sensitivity of utility function
            privacy_budget (PrivacyBudget): Privacy budget parameters
            
        Returns:
            torch.Tensor: Selected output based on exponential mechanism
        """
        # Calibrate noise scale (temperature parameter)
        temperature = self.calibrate_noise(
            sensitivity, privacy_budget.epsilon, 0.0
        )
        
        # Apply exponential mechanism (softmax with temperature)
        probabilities = torch.softmax(data / temperature, dim=-1)
        
        # Sample from the distribution
        if data.dim() == 1:
            selected_idx = torch.multinomial(probabilities, 1)
            result = torch.zeros_like(data)
            result[selected_idx] = 1.0
        else:
            # For batch processing
            selected_indices = torch.multinomial(probabilities, 1)
            result = torch.zeros_like(data)
            for i, idx in enumerate(selected_indices):
                result[i, idx] = 1.0
        
        # Track selection for analysis
        self.selection_history.append({
            'temperature': temperature,
            'sensitivity': sensitivity,
            'epsilon': privacy_budget.epsilon,
            'data_shape': data.shape
        })
        
        return result
    
    def calibrate_noise(self, sensitivity: float, 
                       epsilon: float, 
                       delta: float = 0.0) -> float:
        """
        Calibrate temperature for exponential mechanism.
        
        Args:
            sensitivity (float): Sensitivity of utility function
            epsilon (float): Privacy parameter ε
            delta (float): Ignored for exponential mechanism
            
        Returns:
            float: Temperature parameter
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        temperature = 2 * sensitivity / epsilon
        return temperature
    
    def privacy_analysis(self, temperature: float, 
                        sensitivity: float) -> Tuple[float, float]:
        """
        Analyze privacy guarantees for given temperature.
        
        Args:
            temperature (float): Temperature parameter
            sensitivity (float): Sensitivity of utility function
            
        Returns:
            Tuple[float, float]: (epsilon, 0) privacy parameters
        """
        epsilon = 2 * sensitivity / temperature
        return epsilon, 0.0

class AdaptiveNoiseMechanism:
    """
    Adaptive noise mechanism that adjusts noise based on data characteristics
    and privacy requirements for battery data.
    """
    
    def __init__(self, base_mechanism: NoiseInterface):
        self.base_mechanism = base_mechanism
        self.adaptation_history = []
        self.battery_specific_calibration = True
    
    def add_adaptive_noise(self, data: torch.Tensor,
                          base_sensitivity: float,
                          privacy_budget: PrivacyBudget,
                          data_characteristics: Optional[Dict] = None) -> torch.Tensor:
        """
        Add adaptive noise based on data characteristics.
        
        Args:
            data (torch.Tensor): Input data
            base_sensitivity (float): Base sensitivity estimate
            privacy_budget (PrivacyBudget): Privacy budget
            data_characteristics (Dict, optional): Data-specific characteristics
            
        Returns:
            torch.Tensor: Adaptively noised data
        """
        # Analyze data characteristics
        if data_characteristics is None:
            data_characteristics = self._analyze_data_characteristics(data)
        
        # Adapt sensitivity based on data characteristics
        adapted_sensitivity = self._adapt_sensitivity(
            base_sensitivity, data_characteristics
        )
        
        # Apply base noise mechanism with adapted sensitivity
        noisy_data = self.base_mechanism.add_noise(
            data, adapted_sensitivity, privacy_budget
        )
        
        # Track adaptation
        self.adaptation_history.append({
            'base_sensitivity': base_sensitivity,
            'adapted_sensitivity': adapted_sensitivity,
            'data_characteristics': data_characteristics,
            'adaptation_factor': adapted_sensitivity / base_sensitivity
        })
        
        return noisy_data
    
    def _analyze_data_characteristics(self, data: torch.Tensor) -> Dict[str, float]:
        """Analyze characteristics of battery data."""
        characteristics = {}
        
        # Statistical properties
        characteristics['mean'] = torch.mean(data).item()
        characteristics['std'] = torch.std(data).item()
        characteristics['min'] = torch.min(data).item()
        characteristics['max'] = torch.max(data).item()
        characteristics['range'] = characteristics['max'] - characteristics['min']
        
        # Data distribution properties
        characteristics['skewness'] = self._compute_skewness(data)
        characteristics['kurtosis'] = self._compute_kurtosis(data)
        
        # Gradient properties (if applicable)
        if data.requires_grad:
            characteristics['gradient_norm'] = torch.norm(data.grad).item() if data.grad is not None else 0.0
        
        return characteristics
    
    def _adapt_sensitivity(self, base_sensitivity: float, 
                          characteristics: Dict[str, float]) -> float:
        """Adapt sensitivity based on data characteristics."""
        adaptation_factor = 1.0
        
        # Adapt based on data range
        if characteristics['range'] > 10.0:  # Large range
            adaptation_factor *= 1.2
        elif characteristics['range'] < 0.1:  # Small range
            adaptation_factor *= 0.8
        
        # Adapt based on standard deviation
        if characteristics['std'] > 5.0:  # High variance
            adaptation_factor *= 1.1
        elif characteristics['std'] < 0.1:  # Low variance
            adaptation_factor *= 0.9
        
        # Battery-specific adaptations
        if self.battery_specific_calibration:
            # Voltage data typically has lower sensitivity
            if 2.0 <= characteristics['mean'] <= 5.0:  # Likely voltage data
                adaptation_factor *= 0.7
            
            # Temperature data adaptation
            elif -50.0 <= characteristics['mean'] <= 100.0:  # Likely temperature
                adaptation_factor *= 0.8
            
            # Current data adaptation
            elif abs(characteristics['mean']) > 10.0:  # Likely current data
                adaptation_factor *= 1.3
        
        return base_sensitivity * adaptation_factor
    
    def _compute_skewness(self, data: torch.Tensor) -> float:
        """Compute skewness of data."""
        mean = torch.mean(data)
        std = torch.std(data)
        if std == 0:
            return 0.0
        
        skewness = torch.mean(((data - mean) / std) ** 3)
        return skewness.item()
    
    def _compute_kurtosis(self, data: torch.Tensor) -> float:
        """Compute kurtosis of data."""
        mean = torch.mean(data)
        std = torch.std(data)
        if std == 0:
            return 0.0
        
        kurtosis = torch.mean(((data - mean) / std) ** 4) - 3
        return kurtosis.item()

class CompositionTracker:
    """
    Track privacy composition across multiple noise applications.
    """
    
    def __init__(self, composition_method: str = "advanced"):
        self.composition_method = composition_method
        self.privacy_history = []
        self.total_epsilon = 0.0
        self.total_delta = 0.0
    
    def add_mechanism(self, epsilon: float, delta: float, 
                     mechanism_type: str = "gaussian") -> None:
        """
        Add a privacy mechanism to the composition.
        
        Args:
            epsilon (float): Privacy parameter ε
            delta (float): Privacy parameter δ
            mechanism_type (str): Type of mechanism
        """
        self.privacy_history.append({
            'epsilon': epsilon,
            'delta': delta,
            'mechanism_type': mechanism_type,
            'timestamp': len(self.privacy_history)
        })
        
        # Update total privacy cost
        if self.composition_method == "basic":
            self.total_epsilon += epsilon
            self.total_delta += delta
        elif self.composition_method == "advanced":
            # Advanced composition for Gaussian mechanisms
            if mechanism_type == "gaussian":
                self.total_epsilon = math.sqrt(
                    sum(entry['epsilon']**2 for entry in self.privacy_history)
                ) * math.sqrt(2 * math.log(1.25 / delta))
            else:
                self.total_epsilon += epsilon
            
            self.total_delta += delta
        else:
            raise ValueError(f"Unknown composition method: {self.composition_method}")
    
    def get_total_privacy_cost(self) -> Tuple[float, float]:
        """Get total privacy cost."""
        return self.total_epsilon, self.total_delta
    
    def remaining_budget(self, max_epsilon: float, max_delta: float) -> Tuple[float, float]:
        """Get remaining privacy budget."""
        return (max_epsilon - self.total_epsilon, 
                max_delta - self.total_delta)

class BatterySpecificNoiseCalibration:
    """
    Battery-specific noise calibration for different types of sensor data.
    """
    
    def __init__(self):
        self.sensor_sensitivities = {
            'voltage': 0.1,      # Volts
            'current': 1.0,      # Amperes
            'temperature': 0.5,   # Celsius
            'soc': 0.01,         # State of Charge (%)
            'soh': 0.005,        # State of Health (%)
            'resistance': 0.001,  # Ohms
            'capacity': 0.1,     # Ah
            'power': 5.0,        # Watts
            'energy': 10.0       # Wh
        }
        
        self.privacy_requirements = {
            'voltage': {'epsilon': 2.0, 'delta': 1e-5},
            'current': {'epsilon': 1.5, 'delta': 1e-5},
            'temperature': {'epsilon': 3.0, 'delta': 1e-5},
            'soc': {'epsilon': 1.0, 'delta': 1e-6},
            'soh': {'epsilon': 0.5, 'delta': 1e-6},
            'resistance': {'epsilon': 1.0, 'delta': 1e-5},
            'capacity': {'epsilon': 1.0, 'delta': 1e-5},
            'power': {'epsilon': 2.0, 'delta': 1e-5},
            'energy': {'epsilon': 2.0, 'delta': 1e-5}
        }
    
    def get_sensor_sensitivity(self, sensor_type: str) -> float:
        """Get sensitivity for specific sensor type."""
        return self.sensor_sensitivities.get(sensor_type, 1.0)
    
    def get_privacy_requirements(self, sensor_type: str) -> Dict[str, float]:
        """Get privacy requirements for specific sensor type."""
        return self.privacy_requirements.get(
            sensor_type, 
            {'epsilon': 1.0, 'delta': 1e-5}
        )
    
    def calibrate_for_battery_data(self, data: torch.Tensor, 
                                  sensor_type: str,
                                  mechanism: NoiseInterface) -> torch.Tensor:
        """
        Calibrate noise specifically for battery sensor data.
        
        Args:
            data (torch.Tensor): Battery sensor data
            sensor_type (str): Type of sensor data
            mechanism (NoiseInterface): Noise mechanism to use
            
        Returns:
            torch.Tensor: Noised battery data
        """
        # Get sensor-specific parameters
        sensitivity = self.get_sensor_sensitivity(sensor_type)
        privacy_req = self.get_privacy_requirements(sensor_type)
        
        # Create privacy budget
        privacy_budget = PrivacyBudget(
            epsilon=privacy_req['epsilon'],
            delta=privacy_req['delta']
        )
        
        # Apply noise mechanism
        noisy_data = mechanism.add_noise(data, sensitivity, privacy_budget)
        
        return noisy_data

# Factory functions for easy instantiation
def create_gaussian_mechanism(device: str = "cpu") -> GaussianNoiseMechanism:
    """Create Gaussian noise mechanism."""
    return GaussianNoiseMechanism(device)

def create_laplace_mechanism(device: str = "cpu") -> LaplaceNoiseMechanism:
    """Create Laplace noise mechanism."""
    return LaplaceNoiseMechanism(device)

def create_exponential_mechanism(device: str = "cpu") -> ExponentialMechanism:
    """Create exponential mechanism."""
    return ExponentialMechanism(device)

def create_adaptive_mechanism(base_mechanism: NoiseInterface) -> AdaptiveNoiseMechanism:
    """Create adaptive noise mechanism."""
    return AdaptiveNoiseMechanism(base_mechanism)

# Utility functions
def estimate_gradient_sensitivity(gradients: List[torch.Tensor], 
                                clipping_bound: float) -> float:
    """
    Estimate gradient sensitivity for federated learning.
    
    Args:
        gradients (List[torch.Tensor]): List of gradient tensors
        clipping_bound (float): Gradient clipping bound
        
    Returns:
        float: Estimated L2 sensitivity
    """
    # For clipped gradients, sensitivity is 2 * clipping_bound
    return 2.0 * clipping_bound

def privacy_accounting_report(composition_tracker: CompositionTracker) -> Dict[str, Any]:
    """
    Generate privacy accounting report.
    
    Args:
        composition_tracker (CompositionTracker): Privacy composition tracker
        
    Returns:
        Dict[str, Any]: Privacy accounting report
    """
    total_epsilon, total_delta = composition_tracker.get_total_privacy_cost()
    
    report = {
        'total_epsilon': total_epsilon,
        'total_delta': total_delta,
        'num_mechanisms': len(composition_tracker.privacy_history),
        'composition_method': composition_tracker.composition_method,
        'mechanism_breakdown': composition_tracker.privacy_history
    }
    
    return report
