"""
BatteryMind - Privacy-Preserving Module

Advanced privacy protection mechanisms for federated learning in battery
management systems. Implements differential privacy, homomorphic encryption,
and secure aggregation protocols to ensure data privacy while enabling
collaborative learning across battery fleets.

Key Components:
- DifferentialPrivacy: Differential privacy mechanisms for federated learning
- HomomorphicEncryption: Homomorphic encryption for secure computations
- SecureAggregation: Secure aggregation protocols for model updates
- NoiseGenerator: Advanced noise generation mechanisms for privacy protection

Features:
- Mathematically proven privacy guarantees
- Configurable privacy budgets and noise levels
- Integration with federated learning protocols
- Support for multiple privacy-preserving techniques
- Compliance with data protection regulations

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

from .differential_privacy import (
    DifferentialPrivacy,
    PrivacyBudget,
    GaussianMechanism,
    LaplaceMechanism,
    ExponentialMechanism,
    PrivacyAccountant,
    DPOptimizer
)

from .homomorphic_encryption import (
    HomomorphicEncryption,
    EncryptionKey,
    EncryptedTensor,
    SecureComputation,
    KeyGenerator
)

from .secure_aggregation import (
    SecureAggregation,
    SecureAggregationProtocol,
    MaskedAggregation,
    ThresholdSecretSharing,
    SecureSum
)

from .noise_mechanisms import (
    NoiseGenerator,
    GaussianNoise,
    LaplaceNoise,
    DiscreteGaussianNoise,
    CompositionMechanism,
    AdaptiveNoise
)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module metadata
__all__ = [
    # Differential Privacy
    "DifferentialPrivacy",
    "PrivacyBudget",
    "GaussianMechanism",
    "LaplaceMechanism",
    "ExponentialMechanism",
    "PrivacyAccountant",
    "DPOptimizer",
    
    # Homomorphic Encryption
    "HomomorphicEncryption",
    "EncryptionKey",
    "EncryptedTensor",
    "SecureComputation",
    "KeyGenerator",
    
    # Secure Aggregation
    "SecureAggregation",
    "SecureAggregationProtocol",
    "MaskedAggregation",
    "ThresholdSecretSharing",
    "SecureSum",
    
    # Noise Mechanisms
    "NoiseGenerator",
    "GaussianNoise",
    "LaplaceNoise",
    "DiscreteGaussianNoise",
    "CompositionMechanism",
    "AdaptiveNoise"
]

# Privacy configuration constants
PRIVACY_CONSTANTS = {
    # Standard privacy parameters
    "DEFAULT_EPSILON": 1.0,  # Default privacy budget
    "DEFAULT_DELTA": 1e-5,   # Default failure probability
    "MIN_EPSILON": 0.1,      # Minimum privacy budget
    "MAX_EPSILON": 10.0,     # Maximum privacy budget
    
    # Noise calibration
    "GAUSSIAN_NOISE_MULTIPLIER": 1.1,
    "LAPLACE_SENSITIVITY": 1.0,
    "CLIPPING_THRESHOLD": 1.0,
    
    # Secure aggregation
    "MIN_PARTICIPANTS": 3,
    "DROPOUT_THRESHOLD": 0.5,
    "SECRET_SHARING_THRESHOLD": 2,
    
    # Homomorphic encryption
    "KEY_SIZE": 2048,
    "PLAINTEXT_MODULUS": 1024,
    "SECURITY_LEVEL": 128
}

def get_privacy_constants():
    """
    Get privacy-preserving constants.
    
    Returns:
        dict: Dictionary of privacy constants
    """
    return PRIVACY_CONSTANTS.copy()

def create_differential_privacy_engine(epsilon: float = 1.0, delta: float = 1e-5):
    """
    Factory function to create a differential privacy engine.
    
    Args:
        epsilon (float): Privacy budget parameter
        delta (float): Failure probability parameter
        
    Returns:
        DifferentialPrivacy: Configured differential privacy engine
    """
    privacy_budget = PrivacyBudget(epsilon=epsilon, delta=delta)
    return DifferentialPrivacy(privacy_budget)

def create_secure_aggregation_protocol(min_participants: int = 3, 
                                     threshold: int = 2):
    """
    Factory function to create a secure aggregation protocol.
    
    Args:
        min_participants (int): Minimum number of participants
        threshold (int): Threshold for secret sharing
        
    Returns:
        SecureAggregationProtocol: Configured secure aggregation protocol
    """
    return SecureAggregationProtocol(
        min_participants=min_participants,
        threshold=threshold
    )

def create_homomorphic_encryption_system(key_size: int = 2048):
    """
    Factory function to create a homomorphic encryption system.
    
    Args:
        key_size (int): Encryption key size
        
    Returns:
        HomomorphicEncryption: Configured homomorphic encryption system
    """
    return HomomorphicEncryption(key_size=key_size)

# Privacy validation utilities
def validate_privacy_parameters(epsilon: float, delta: float) -> bool:
    """
    Validate privacy parameters for differential privacy.
    
    Args:
        epsilon (float): Privacy budget
        delta (float): Failure probability
        
    Returns:
        bool: True if parameters are valid
    """
    return (PRIVACY_CONSTANTS["MIN_EPSILON"] <= epsilon <= PRIVACY_CONSTANTS["MAX_EPSILON"] and
            0 < delta < 1)

def calculate_composition_privacy(privacy_budgets: list) -> tuple:
    """
    Calculate composed privacy budget for multiple mechanisms.
    
    Args:
        privacy_budgets (list): List of (epsilon, delta) tuples
        
    Returns:
        tuple: Composed (epsilon, delta) privacy budget
    """
    total_epsilon = sum(eps for eps, _ in privacy_budgets)
    total_delta = sum(delta for _, delta in privacy_budgets)
    
    return total_epsilon, total_delta

def estimate_noise_scale(sensitivity: float, epsilon: float, 
                        mechanism: str = "gaussian") -> float:
    """
    Estimate noise scale for differential privacy mechanism.
    
    Args:
        sensitivity (float): Global sensitivity of the function
        epsilon (float): Privacy budget
        mechanism (str): Noise mechanism type
        
    Returns:
        float: Estimated noise scale
    """
    if mechanism == "gaussian":
        # For Gaussian mechanism: σ = sqrt(2 * ln(1.25/δ)) * Δ / ε
        delta = PRIVACY_CONSTANTS["DEFAULT_DELTA"]
        import math
        return math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon
    elif mechanism == "laplace":
        # For Laplace mechanism: b = Δ / ε
        return sensitivity / epsilon
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

# Integration utilities
def create_federated_privacy_config(num_clients: int, privacy_budget: float,
                                   aggregation_method: str = "secure") -> dict:
    """
    Create privacy configuration for federated learning.
    
    Args:
        num_clients (int): Number of federated learning clients
        privacy_budget (float): Total privacy budget
        aggregation_method (str): Aggregation method
        
    Returns:
        dict: Privacy configuration dictionary
    """
    # Distribute privacy budget across clients
    client_epsilon = privacy_budget / num_clients
    client_delta = PRIVACY_CONSTANTS["DEFAULT_DELTA"] / num_clients
    
    config = {
        "differential_privacy": {
            "enabled": True,
            "epsilon": client_epsilon,
            "delta": client_delta,
            "mechanism": "gaussian",
            "clipping_threshold": PRIVACY_CONSTANTS["CLIPPING_THRESHOLD"]
        },
        "secure_aggregation": {
            "enabled": aggregation_method == "secure",
            "min_participants": max(2, num_clients // 2),
            "threshold": max(2, num_clients // 3),
            "dropout_tolerance": PRIVACY_CONSTANTS["DROPOUT_THRESHOLD"]
        },
        "homomorphic_encryption": {
            "enabled": False,  # Optional for advanced scenarios
            "key_size": PRIVACY_CONSTANTS["KEY_SIZE"],
            "security_level": PRIVACY_CONSTANTS["SECURITY_LEVEL"]
        }
    }
    
    return config

# Compliance and auditing
def generate_privacy_report(privacy_mechanisms: list) -> dict:
    """
    Generate privacy compliance report.
    
    Args:
        privacy_mechanisms (list): List of privacy mechanisms used
        
    Returns:
        dict: Privacy compliance report
    """
    report = {
        "privacy_guarantees": [],
        "total_privacy_cost": {"epsilon": 0.0, "delta": 0.0},
        "compliance_status": "compliant",
        "recommendations": []
    }
    
    for mechanism in privacy_mechanisms:
        if hasattr(mechanism, 'privacy_budget'):
            budget = mechanism.privacy_budget
            report["privacy_guarantees"].append({
                "mechanism": mechanism.__class__.__name__,
                "epsilon": budget.epsilon,
                "delta": budget.delta
            })
            report["total_privacy_cost"]["epsilon"] += budget.epsilon
            report["total_privacy_cost"]["delta"] += budget.delta
    
    # Check compliance
    total_epsilon = report["total_privacy_cost"]["epsilon"]
    if total_epsilon > PRIVACY_CONSTANTS["MAX_EPSILON"]:
        report["compliance_status"] = "non_compliant"
        report["recommendations"].append(
            f"Total epsilon ({total_epsilon:.2f}) exceeds maximum allowed "
            f"({PRIVACY_CONSTANTS['MAX_EPSILON']})"
        )
    
    return report

# Module health check
def privacy_module_health_check() -> dict:
    """
    Perform health check of the privacy-preserving module.
    
    Returns:
        dict: Health check results
    """
    health_status = {
        "module_loaded": True,
        "version": __version__,
        "components_available": {
            "differential_privacy": True,
            "homomorphic_encryption": True,
            "secure_aggregation": True,
            "noise_mechanisms": True
        },
        "dependencies_satisfied": True
    }
    
    try:
        # Test basic functionality
        dp_engine = create_differential_privacy_engine()
        health_status["differential_privacy_test"] = True
    except Exception as e:
        health_status["differential_privacy_test"] = False
        health_status["error"] = str(e)
    
    return health_status

# Export configuration template
def export_privacy_config_template(file_path: str = "privacy_config.yaml"):
    """
    Export a privacy configuration template.
    
    Args:
        file_path (str): Path to save the configuration template
    """
    import yaml
    
    config_template = {
        "privacy_preserving": {
            "differential_privacy": {
                "epsilon": PRIVACY_CONSTANTS["DEFAULT_EPSILON"],
                "delta": PRIVACY_CONSTANTS["DEFAULT_DELTA"],
                "mechanism": "gaussian",
                "clipping_threshold": PRIVACY_CONSTANTS["CLIPPING_THRESHOLD"],
                "noise_multiplier": PRIVACY_CONSTANTS["GAUSSIAN_NOISE_MULTIPLIER"]
            },
            "secure_aggregation": {
                "enabled": True,
                "min_participants": PRIVACY_CONSTANTS["MIN_PARTICIPANTS"],
                "threshold": PRIVACY_CONSTANTS["SECRET_SHARING_THRESHOLD"],
                "dropout_threshold": PRIVACY_CONSTANTS["DROPOUT_THRESHOLD"]
            },
            "homomorphic_encryption": {
                "enabled": False,
                "key_size": PRIVACY_CONSTANTS["KEY_SIZE"],
                "security_level": PRIVACY_CONSTANTS["SECURITY_LEVEL"]
            }
        },
        "_metadata": {
            "version": __version__,
            "description": "BatteryMind Privacy-Preserving Configuration Template",
            "author": __author__
        }
    }
    
    with open(file_path, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False, indent=2)

# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BatteryMind Privacy-Preserving Module v{__version__} initialized")
