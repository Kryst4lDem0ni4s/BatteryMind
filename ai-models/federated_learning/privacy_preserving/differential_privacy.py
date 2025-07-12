"""
BatteryMind - Differential Privacy Implementation

Advanced differential privacy mechanisms for federated learning in battery
management systems. Provides mathematically proven privacy guarantees while
enabling collaborative learning across distributed battery fleets.

Features:
- Multiple noise mechanisms (Gaussian, Laplace, Exponential)
- Privacy budget management and accounting
- Gradient clipping and noise injection
- Composition analysis for multiple queries
- Integration with PyTorch optimizers
- Adaptive privacy parameters

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PrivacyBudget:
    """
    Privacy budget for differential privacy mechanisms.
    
    Attributes:
        epsilon (float): Privacy budget parameter (smaller = more private)
        delta (float): Failure probability (probability of privacy breach)
        composition_method (str): Method for privacy composition
        spent_epsilon (float): Amount of epsilon already spent
        spent_delta (float): Amount of delta already spent
    """
    epsilon: float = 1.0
    delta: float = 1e-5
    composition_method: str = "basic"
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    
    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if not 0 <= self.delta <= 1:
            raise ValueError("Delta must be between 0 and 1")
    
    def remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget."""
        return (self.epsilon - self.spent_epsilon, 
                self.delta - self.spent_delta)
    
    def spend_budget(self, epsilon_cost: float, delta_cost: float) -> None:
        """Spend privacy budget."""
        remaining_eps, remaining_delta = self.remaining_budget()
        
        if epsilon_cost > remaining_eps:
            raise ValueError(f"Insufficient epsilon budget: need {epsilon_cost}, have {remaining_eps}")
        if delta_cost > remaining_delta:
            raise ValueError(f"Insufficient delta budget: need {delta_cost}, have {remaining_delta}")
        
        self.spent_epsilon += epsilon_cost
        self.spent_delta += delta_cost
    
    def is_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        remaining_eps, remaining_delta = self.remaining_budget()
        return remaining_eps <= 0 or remaining_delta <= 0

class NoiseGenerator(ABC):
    """
    Abstract base class for noise generation mechanisms.
    """
    
    @abstractmethod
    def generate_noise(self, shape: torch.Size, device: torch.device = None) -> torch.Tensor:
        """Generate noise tensor with specified shape."""
        pass
    
    @abstractmethod
    def calibrate_noise(self, sensitivity: float, epsilon: float, delta: float = None) -> None:
        """Calibrate noise parameters for given privacy requirements."""
        pass

class GaussianNoise(NoiseGenerator):
    """
    Gaussian noise mechanism for differential privacy.
    """
    
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
        self.generator = torch.Generator()
    
    def generate_noise(self, shape: torch.Size, device: torch.device = None) -> torch.Tensor:
        """Generate Gaussian noise."""
        if device is None:
            device = torch.device('cpu')
        
        return torch.normal(
            mean=0.0, 
            std=self.sigma, 
            size=shape, 
            device=device,
            generator=self.generator
        )
    
    def calibrate_noise(self, sensitivity: float, epsilon: float, delta: float = 1e-5) -> None:
        """Calibrate Gaussian noise for (ε, δ)-differential privacy."""
        # For Gaussian mechanism: σ = sqrt(2 * ln(1.25/δ)) * Δ / ε
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1) for Gaussian mechanism")
        
        self.sigma = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon
        logger.info(f"Calibrated Gaussian noise: σ = {self.sigma:.4f}")

class LaplaceNoise(NoiseGenerator):
    """
    Laplace noise mechanism for differential privacy.
    """
    
    def __init__(self, scale: float = 1.0):
        self.scale = scale
    
    def generate_noise(self, shape: torch.Size, device: torch.device = None) -> torch.Tensor:
        """Generate Laplace noise."""
        if device is None:
            device = torch.device('cpu')
        
        # Generate Laplace noise using uniform random variables
        uniform = torch.rand(shape, device=device) - 0.5
        return -self.scale * torch.sign(uniform) * torch.log(1 - 2 * torch.abs(uniform))
    
    def calibrate_noise(self, sensitivity: float, epsilon: float, delta: float = None) -> None:
        """Calibrate Laplace noise for ε-differential privacy."""
        # For Laplace mechanism: b = Δ / ε
        self.scale = sensitivity / epsilon
        logger.info(f"Calibrated Laplace noise: b = {self.scale:.4f}")

class ExponentialMechanism(NoiseGenerator):
    """
    Exponential mechanism for differential privacy.
    """
    
    def __init__(self, sensitivity: float = 1.0, epsilon: float = 1.0):
        self.sensitivity = sensitivity
        self.epsilon = epsilon
    
    def generate_noise(self, shape: torch.Size, device: torch.device = None) -> torch.Tensor:
        """Generate noise using exponential mechanism."""
        if device is None:
            device = torch.device('cpu')
        
        # Simplified implementation for continuous case
        scale = 2 * self.sensitivity / self.epsilon
        return torch.exponential(torch.ones(shape, device=device) / scale)
    
    def calibrate_noise(self, sensitivity: float, epsilon: float, delta: float = None) -> None:
        """Calibrate exponential mechanism."""
        self.sensitivity = sensitivity
        self.epsilon = epsilon

class PrivacyAccountant:
    """
    Privacy accountant for tracking privacy budget consumption.
    """
    
    def __init__(self, initial_budget: PrivacyBudget):
        self.initial_budget = initial_budget
        self.privacy_history = []
        self.composition_tracker = defaultdict(list)
    
    def spend_privacy(self, mechanism: str, epsilon: float, delta: float = 0.0,
                     metadata: Dict = None) -> None:
        """
        Record privacy expenditure.
        
        Args:
            mechanism (str): Name of the privacy mechanism
            epsilon (float): Epsilon cost
            delta (float): Delta cost
            metadata (Dict, optional): Additional metadata
        """
        # Check if budget is sufficient
        remaining_eps, remaining_delta = self.initial_budget.remaining_budget()
        
        if epsilon > remaining_eps:
            raise ValueError(f"Insufficient epsilon budget: need {epsilon}, have {remaining_eps}")
        if delta > remaining_delta:
            raise ValueError(f"Insufficient delta budget: need {delta}, have {remaining_delta}")
        
        # Record expenditure
        expenditure = {
            'mechanism': mechanism,
            'epsilon': epsilon,
            'delta': delta,
            'timestamp': torch.tensor(0.0),  # Would use actual timestamp in production
            'metadata': metadata or {}
        }
        
        self.privacy_history.append(expenditure)
        self.composition_tracker[mechanism].append((epsilon, delta))
        
        # Update budget
        self.initial_budget.spend_budget(epsilon, delta)
        
        logger.info(f"Privacy spent - {mechanism}: ε={epsilon:.4f}, δ={delta:.6f}")
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get summary of privacy expenditure."""
        total_epsilon = sum(exp['epsilon'] for exp in self.privacy_history)
        total_delta = sum(exp['delta'] for exp in self.privacy_history)
        
        remaining_eps, remaining_delta = self.initial_budget.remaining_budget()
        
        return {
            'initial_budget': {
                'epsilon': self.initial_budget.epsilon,
                'delta': self.initial_budget.delta
            },
            'spent_budget': {
                'epsilon': total_epsilon,
                'delta': total_delta
            },
            'remaining_budget': {
                'epsilon': remaining_eps,
                'delta': remaining_delta
            },
            'expenditure_count': len(self.privacy_history),
            'mechanisms_used': list(self.composition_tracker.keys())
        }
    
    def analyze_composition(self) -> Dict[str, float]:
        """Analyze privacy composition across mechanisms."""
        if not self.privacy_history:
            return {'total_epsilon': 0.0, 'total_delta': 0.0}
        
        # Basic composition (worst case)
        total_epsilon = sum(exp['epsilon'] for exp in self.privacy_history)
        total_delta = sum(exp['delta'] for exp in self.privacy_history)
        
        # Advanced composition could be implemented here
        # using techniques like RDP (Rényi Differential Privacy)
        
        return {
            'total_epsilon': total_epsilon,
            'total_delta': total_delta,
            'composition_method': self.initial_budget.composition_method
        }

class GradientClipper:
    """
    Gradient clipping for differential privacy.
    """
    
    def __init__(self, clipping_threshold: float = 1.0, norm_type: str = 'l2'):
        self.clipping_threshold = clipping_threshold
        self.norm_type = norm_type
    
    def clip_gradients(self, parameters: List[torch.Tensor]) -> float:
        """
        Clip gradients to bound sensitivity.
        
        Args:
            parameters (List[torch.Tensor]): Model parameters
            
        Returns:
            float: Actual clipping norm
        """
        if self.norm_type == 'l2':
            return self._clip_l2_norm(parameters)
        elif self.norm_type == 'l1':
            return self._clip_l1_norm(parameters)
        else:
            raise ValueError(f"Unsupported norm type: {self.norm_type}")
    
    def _clip_l2_norm(self, parameters: List[torch.Tensor]) -> float:
        """Clip gradients using L2 norm."""
        total_norm = 0.0
        
        # Calculate total gradient norm
        for param in parameters:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        
        # Clip if necessary
        if total_norm > self.clipping_threshold:
            clip_factor = self.clipping_threshold / total_norm
            for param in parameters:
                if param.grad is not None:
                    param.grad.data.mul_(clip_factor)
        
        return min(total_norm, self.clipping_threshold)
    
    def _clip_l1_norm(self, parameters: List[torch.Tensor]) -> float:
        """Clip gradients using L1 norm."""
        total_norm = 0.0
        
        # Calculate total gradient norm
        for param in parameters:
            if param.grad is not None:
                param_norm = param.grad.data.norm(1)
                total_norm += param_norm.item()
        
        # Clip if necessary
        if total_norm > self.clipping_threshold:
            clip_factor = self.clipping_threshold / total_norm
            for param in parameters:
                if param.grad is not None:
                    param.grad.data.mul_(clip_factor)
        
        return min(total_norm, self.clipping_threshold)

class DPOptimizer:
    """
    Differentially private optimizer wrapper.
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 noise_generator: NoiseGenerator,
                 gradient_clipper: GradientClipper,
                 privacy_accountant: PrivacyAccountant):
        self.optimizer = optimizer
        self.noise_generator = noise_generator
        self.gradient_clipper = gradient_clipper
        self.privacy_accountant = privacy_accountant
        self.step_count = 0
    
    def step(self, closure: Callable = None) -> None:
        """
        Perform one optimization step with differential privacy.
        
        Args:
            closure (Callable, optional): Closure for computing loss
        """
        # Get model parameters
        parameters = []
        for group in self.optimizer.param_groups:
            parameters.extend(group['params'])
        
        # Clip gradients
        clipping_norm = self.gradient_clipper.clip_gradients(parameters)
        
        # Add noise to gradients
        self._add_noise_to_gradients(parameters)
        
        # Perform optimization step
        self.optimizer.step(closure)
        
        # Record privacy expenditure
        self._record_privacy_cost()
        
        self.step_count += 1
    
    def _add_noise_to_gradients(self, parameters: List[torch.Tensor]) -> None:
        """Add calibrated noise to gradients."""
        for param in parameters:
            if param.grad is not None:
                noise = self.noise_generator.generate_noise(
                    param.grad.shape, 
                    param.grad.device
                )
                param.grad.data.add_(noise)
    
    def _record_privacy_cost(self) -> None:
        """Record privacy cost for this optimization step."""
        # Calculate privacy cost based on noise mechanism
        if isinstance(self.noise_generator, GaussianNoise):
            # For Gaussian mechanism with clipping
            sensitivity = self.gradient_clipper.clipping_threshold
            sigma = self.noise_generator.sigma
            
            # Calculate epsilon for this step
            epsilon = sensitivity / sigma
            delta = 1e-5  # Typical delta for Gaussian mechanism
            
            self.privacy_accountant.spend_privacy(
                mechanism="gaussian_sgd",
                epsilon=epsilon,
                delta=delta,
                metadata={'step': self.step_count, 'sigma': sigma}
            )
        
        elif isinstance(self.noise_generator, LaplaceNoise):
            # For Laplace mechanism
            sensitivity = self.gradient_clipper.clipping_threshold
            scale = self.noise_generator.scale
            
            epsilon = sensitivity / scale
            
            self.privacy_accountant.spend_privacy(
                mechanism="laplace_sgd",
                epsilon=epsilon,
                delta=0.0,
                metadata={'step': self.step_count, 'scale': scale}
            )
    
    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def state_dict(self) -> Dict:
        """Get optimizer state."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load optimizer state."""
        self.optimizer.load_state_dict(state_dict)

class DifferentialPrivacy:
    """
    Main differential privacy engine for federated learning.
    """
    
    def __init__(self, privacy_budget: PrivacyBudget):
        self.privacy_budget = privacy_budget
        self.privacy_accountant = PrivacyAccountant(privacy_budget)
        self.noise_generators = {
            'gaussian': GaussianNoise(),
            'laplace': LaplaceNoise(),
            'exponential': ExponentialMechanism()
        }
        self.gradient_clipper = GradientClipper()
    
    def create_dp_optimizer(self, optimizer: torch.optim.Optimizer,
                           noise_mechanism: str = 'gaussian',
                           clipping_threshold: float = 1.0,
                           noise_multiplier: float = 1.1) -> DPOptimizer:
        """
        Create differentially private optimizer.
        
        Args:
            optimizer (torch.optim.Optimizer): Base optimizer
            noise_mechanism (str): Type of noise mechanism
            clipping_threshold (float): Gradient clipping threshold
            noise_multiplier (float): Noise scale multiplier
            
        Returns:
            DPOptimizer: Differentially private optimizer
        """
        # Configure noise generator
        noise_gen = self.noise_generators[noise_mechanism]
        
        # Calibrate noise based on privacy budget
        remaining_eps, remaining_delta = self.privacy_budget.remaining_budget()
        
        if noise_mechanism == 'gaussian':
            noise_gen.calibrate_noise(
                sensitivity=clipping_threshold,
                epsilon=remaining_eps / 100,  # Conservative allocation
                delta=remaining_delta / 100
            )
            noise_gen.sigma *= noise_multiplier
        elif noise_mechanism == 'laplace':
            noise_gen.calibrate_noise(
                sensitivity=clipping_threshold,
                epsilon=remaining_eps / 100
            )
            noise_gen.scale *= noise_multiplier
        
        # Configure gradient clipper
        clipper = GradientClipper(clipping_threshold=clipping_threshold)
        
        return DPOptimizer(
            optimizer=optimizer,
            noise_generator=noise_gen,
            gradient_clipper=clipper,
            privacy_accountant=self.privacy_accountant
        )
    
    def add_noise_to_tensor(self, tensor: torch.Tensor, 
                           sensitivity: float,
                           mechanism: str = 'gaussian') -> torch.Tensor:
        """
        Add calibrated noise to a tensor.
        
        Args:
            tensor (torch.Tensor): Input tensor
            sensitivity (float): Global sensitivity
            mechanism (str): Noise mechanism
            
        Returns:
            torch.Tensor: Noisy tensor
        """
        remaining_eps, remaining_delta = self.privacy_budget.remaining_budget()
        
        if remaining_eps <= 0:
            raise ValueError("Privacy budget exhausted")
        
        # Use a small fraction of remaining budget
        epsilon_cost = min(0.1, remaining_eps / 10)
        delta_cost = min(1e-6, remaining_delta / 10) if mechanism == 'gaussian' else 0.0
        
        # Generate noise
        noise_gen = self.noise_generators[mechanism]
        noise_gen.calibrate_noise(sensitivity, epsilon_cost, delta_cost)
        noise = noise_gen.generate_noise(tensor.shape, tensor.device)
        
        # Record privacy cost
        self.privacy_accountant.spend_privacy(
            mechanism=f"{mechanism}_tensor_noise",
            epsilon=epsilon_cost,
            delta=delta_cost
        )
        
        return tensor + noise
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get comprehensive privacy summary."""
        return self.privacy_accountant.get_privacy_summary()
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.privacy_budget.is_exhausted()

# Factory functions
def create_differential_privacy_engine(epsilon: float = 1.0, 
                                     delta: float = 1e-5) -> DifferentialPrivacy:
    """
    Factory function to create differential privacy engine.
    
    Args:
        epsilon (float): Privacy budget parameter
        delta (float): Failure probability parameter
        
    Returns:
        DifferentialPrivacy: Configured differential privacy engine
    """
    privacy_budget = PrivacyBudget(epsilon=epsilon, delta=delta)
    return DifferentialPrivacy(privacy_budget)

def create_dp_sgd_optimizer(model: nn.Module, lr: float = 0.01,
                           epsilon: float = 1.0, delta: float = 1e-5,
                           clipping_threshold: float = 1.0,
                           noise_multiplier: float = 1.1) -> DPOptimizer:
    """
    Create DP-SGD optimizer for a model.
    
    Args:
        model (nn.Module): PyTorch model
        lr (float): Learning rate
        epsilon (float): Privacy budget
        delta (float): Failure probability
        clipping_threshold (float): Gradient clipping threshold
        noise_multiplier (float): Noise scale multiplier
        
    Returns:
        DPOptimizer: Differentially private SGD optimizer
    """
    # Create base optimizer
    base_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Create DP engine
    dp_engine = create_differential_privacy_engine(epsilon, delta)
    
    # Create DP optimizer
    return dp_engine.create_dp_optimizer(
        optimizer=base_optimizer,
        noise_mechanism='gaussian',
        clipping_threshold=clipping_threshold,
        noise_multiplier=noise_multiplier
    )

# Utility functions
def estimate_privacy_cost(num_steps: int, noise_multiplier: float,
                         batch_size: int, dataset_size: int) -> Tuple[float, float]:
    """
    Estimate privacy cost for DP-SGD training.
    
    Args:
        num_steps (int): Number of training steps
        noise_multiplier (float): Noise multiplier
        batch_size (int): Batch size
        dataset_size (int): Dataset size
        
    Returns:
        Tuple[float, float]: Estimated (epsilon, delta) cost
    """
    # Simplified privacy analysis
    # In practice, would use more sophisticated tools like TensorFlow Privacy
    
    sampling_rate = batch_size / dataset_size
    
    # Basic composition bound
    epsilon_per_step = sampling_rate / noise_multiplier
    total_epsilon = epsilon_per_step * num_steps
    
    # Conservative delta estimate
    delta = 1 / dataset_size
    
    return total_epsilon, delta
