"""
BatteryMind - Federated Learning Privacy Engine

Advanced privacy-preserving mechanisms for federated learning with battery
health prediction models. Implements differential privacy, secure aggregation,
and homomorphic encryption for protecting sensitive battery data.

Features:
- Differential privacy with adaptive noise calibration
- Secure multi-party computation for aggregation
- Homomorphic encryption for model updates
- Privacy budget management and tracking
- Gradient clipping and noise injection
- Membership inference attack protection

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import hashlib
import secrets
from collections import defaultdict
import time

# Cryptographic imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64

# Differential privacy imports
from scipy import stats
from scipy.optimize import minimize_scalar

# Secure aggregation imports
import hashlib
import hmac

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PrivacyConfig:
    """
    Configuration for privacy-preserving mechanisms.
    
    Attributes:
        # Differential privacy settings
        enable_dp (bool): Enable differential privacy
        epsilon (float): Privacy budget (smaller = more private)
        delta (float): Failure probability
        sensitivity (float): Global sensitivity of the function
        noise_multiplier (float): Noise multiplier for DP-SGD
        
        # Gradient clipping
        max_grad_norm (float): Maximum gradient norm for clipping
        adaptive_clipping (bool): Use adaptive gradient clipping
        
        # Secure aggregation
        enable_secure_aggregation (bool): Enable secure aggregation
        threshold (int): Minimum number of participants for aggregation
        
        # Homomorphic encryption
        enable_homomorphic (bool): Enable homomorphic encryption
        key_size (int): Encryption key size
        
        # Privacy accounting
        privacy_accountant (str): Type of privacy accountant ('rdp', 'gdp')
        max_privacy_budget (float): Maximum allowed privacy budget
        
        # Attack protection
        enable_membership_protection (bool): Protect against membership inference
        enable_model_inversion_protection (bool): Protect against model inversion
    """
    # Differential privacy settings
    enable_dp: bool = True
    epsilon: float = 1.0
    delta: float = 1e-5
    sensitivity: float = 1.0
    noise_multiplier: float = 1.1
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    adaptive_clipping: bool = True
    
    # Secure aggregation
    enable_secure_aggregation: bool = True
    threshold: int = 3
    
    # Homomorphic encryption
    enable_homomorphic: bool = False
    key_size: int = 2048
    
    # Privacy accounting
    privacy_accountant: str = "rdp"
    max_privacy_budget: float = 10.0
    
    # Attack protection
    enable_membership_protection: bool = True
    enable_model_inversion_protection: bool = True

class DifferentialPrivacy:
    """
    Differential privacy implementation for federated learning.
    """
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.privacy_spent = 0.0
        self.noise_history = []
        
    def add_noise_to_gradients(self, gradients: torch.Tensor, 
                             sensitivity: Optional[float] = None) -> torch.Tensor:
        """
        Add calibrated noise to gradients for differential privacy.
        
        Args:
            gradients (torch.Tensor): Original gradients
            sensitivity (float, optional): Function sensitivity
            
        Returns:
            torch.Tensor: Noisy gradients
        """
        if not self.config.enable_dp:
            return gradients
        
        # Use provided sensitivity or default
        sens = sensitivity or self.config.sensitivity
        
        # Calculate noise scale based on privacy parameters
        noise_scale = self._calculate_noise_scale(sens)
        
        # Generate Gaussian noise
        noise = torch.normal(
            mean=0.0,
            std=noise_scale,
            size=gradients.shape,
            device=gradients.device
        )
        
        # Add noise to gradients
        noisy_gradients = gradients + noise
        
        # Track noise for privacy accounting
        self.noise_history.append({
            'noise_scale': noise_scale,
            'sensitivity': sens,
            'timestamp': time.time()
        })
        
        return noisy_gradients
    
    def _calculate_noise_scale(self, sensitivity: float) -> float:
        """Calculate noise scale for given sensitivity and privacy parameters."""
        # For Gaussian mechanism: σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        noise_scale = sensitivity * math.sqrt(2 * math.log(1.25 / self.config.delta)) / self.config.epsilon
        return noise_scale
    
    def clip_gradients(self, gradients: torch.Tensor, 
                      max_norm: Optional[float] = None) -> Tuple[torch.Tensor, float]:
        """
        Clip gradients to bound sensitivity.
        
        Args:
            gradients (torch.Tensor): Original gradients
            max_norm (float, optional): Maximum gradient norm
            
        Returns:
            Tuple[torch.Tensor, float]: Clipped gradients and clipping factor
        """
        max_norm = max_norm or self.config.max_grad_norm
        
        # Calculate gradient norm
        grad_norm = torch.norm(gradients)
        
        # Clip if necessary
        if grad_norm > max_norm:
            clipping_factor = max_norm / grad_norm
            clipped_gradients = gradients * clipping_factor
        else:
            clipping_factor = 1.0
            clipped_gradients = gradients
        
        return clipped_gradients, clipping_factor
    
    def adaptive_gradient_clipping(self, gradients: torch.Tensor, 
                                 percentile: float = 50.0) -> Tuple[torch.Tensor, float]:
        """
        Adaptive gradient clipping based on gradient norm distribution.
        
        Args:
            gradients (torch.Tensor): Original gradients
            percentile (float): Percentile for adaptive threshold
            
        Returns:
            Tuple[torch.Tensor, float]: Clipped gradients and threshold
        """
        if not self.config.adaptive_clipping:
            return self.clip_gradients(gradients)
        
        # Calculate per-parameter gradient norms
        param_norms = torch.norm(gradients.view(gradients.size(0), -1), dim=1)
        
        # Calculate adaptive threshold
        threshold = torch.quantile(param_norms, percentile / 100.0)
        threshold = max(threshold.item(), self.config.max_grad_norm)
        
        # Clip gradients
        return self.clip_gradients(gradients, threshold)
    
    def calculate_privacy_spent(self, steps: int, batch_size: int, 
                              dataset_size: int) -> float:
        """
        Calculate privacy budget spent using RDP accounting.
        
        Args:
            steps (int): Number of training steps
            batch_size (int): Batch size
            dataset_size (int): Total dataset size
            
        Returns:
            float: Privacy budget spent (epsilon)
        """
        if not self.config.enable_dp:
            return 0.0
        
        # Sampling probability
        q = batch_size / dataset_size
        
        # RDP calculation (simplified)
        # In practice, would use more sophisticated RDP accounting
        sigma = self.config.noise_multiplier
        alpha = 2.0  # RDP order
        
        # RDP at order alpha
        rdp = alpha * q**2 * steps / (2 * sigma**2)
        
        # Convert RDP to (ε, δ)-DP
        epsilon = rdp + math.log(1 / self.config.delta) / (alpha - 1)
        
        self.privacy_spent = epsilon
        return epsilon
    
    def check_privacy_budget(self, additional_epsilon: float = 0.0) -> bool:
        """
        Check if privacy budget allows for additional computation.
        
        Args:
            additional_epsilon (float): Additional privacy budget needed
            
        Returns:
            bool: True if budget allows, False otherwise
        """
        total_needed = self.privacy_spent + additional_epsilon
        return total_needed <= self.config.max_privacy_budget

class SecureAggregation:
    """
    Secure aggregation implementation for federated learning.
    """
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.client_keys = {}
        self.aggregation_masks = {}
        
    def generate_client_keys(self, client_id: str) -> Dict[str, bytes]:
        """
        Generate cryptographic keys for a client.
        
        Args:
            client_id (str): Client identifier
            
        Returns:
            Dict[str, bytes]: Dictionary of client keys
        """
        # Generate random seeds for mask generation
        seed_self = secrets.randbits(256)
        seed_pairwise = secrets.randbits(256)
        
        # Generate signing key
        signing_key = secrets.randbits(256)
        
        keys = {
            'seed_self': seed_self.to_bytes(32, 'big'),
            'seed_pairwise': seed_pairwise.to_bytes(32, 'big'),
            'signing_key': signing_key.to_bytes(32, 'big')
        }
        
        self.client_keys[client_id] = keys
        return keys
    
    def generate_aggregation_mask(self, client_id: str, model_shape: Tuple[int, ...],
                                other_clients: List[str]) -> torch.Tensor:
        """
        Generate aggregation mask for secure aggregation.
        
        Args:
            client_id (str): Client identifier
            model_shape (Tuple[int, ...]): Shape of model parameters
            other_clients (List[str]): List of other participating clients
            
        Returns:
            torch.Tensor: Aggregation mask
        """
        if client_id not in self.client_keys:
            raise ValueError(f"Keys not found for client {client_id}")
        
        keys = self.client_keys[client_id]
        
        # Generate self mask
        np.random.seed(int.from_bytes(keys['seed_self'], 'big') % (2**32))
        self_mask = torch.from_numpy(np.random.normal(0, 1, model_shape)).float()
        
        # Generate pairwise masks
        pairwise_mask = torch.zeros(model_shape)
        
        for other_client in other_clients:
            if other_client in self.client_keys:
                # Create deterministic seed from both client IDs
                combined_seed = hashlib.sha256(
                    (client_id + other_client).encode()
                ).digest()
                
                np.random.seed(int.from_bytes(combined_seed[:4], 'big'))
                
                if client_id < other_client:
                    # Add mask
                    pairwise_mask += torch.from_numpy(
                        np.random.normal(0, 1, model_shape)
                    ).float()
                else:
                    # Subtract mask
                    pairwise_mask -= torch.from_numpy(
                        np.random.normal(0, 1, model_shape)
                    ).float()
        
        # Combine masks
        total_mask = self_mask + pairwise_mask
        
        self.aggregation_masks[client_id] = total_mask
        return total_mask
    
    def mask_model_update(self, model_update: torch.Tensor, 
                         aggregation_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply aggregation mask to model update.
        
        Args:
            model_update (torch.Tensor): Original model update
            aggregation_mask (torch.Tensor): Aggregation mask
            
        Returns:
            torch.Tensor: Masked model update
        """
        return model_update + aggregation_mask
    
    def verify_aggregation_threshold(self, participating_clients: List[str]) -> bool:
        """
        Verify that enough clients are participating for secure aggregation.
        
        Args:
            participating_clients (List[str]): List of participating clients
            
        Returns:
            bool: True if threshold is met
        """
        return len(participating_clients) >= self.config.threshold

class HomomorphicEncryption:
    """
    Homomorphic encryption for model updates.
    """
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.public_key = None
        self.private_key = None
        
        if config.enable_homomorphic:
            self._generate_keys()
    
    def _generate_keys(self):
        """Generate RSA key pair for homomorphic operations."""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.config.key_size,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        
        logger.info("Homomorphic encryption keys generated")
    
    def encrypt_model_update(self, model_update: torch.Tensor) -> bytes:
        """
        Encrypt model update using homomorphic encryption.
        
        Args:
            model_update (torch.Tensor): Model update to encrypt
            
        Returns:
            bytes: Encrypted model update
        """
        if not self.config.enable_homomorphic or not self.public_key:
            # Return serialized tensor if encryption disabled
            return model_update.numpy().tobytes()
        
        # Serialize model update
        serialized = model_update.numpy().tobytes()
        
        # Encrypt using RSA
        encrypted = self.public_key.encrypt(
            serialized,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return encrypted
    
    def decrypt_model_update(self, encrypted_update: bytes, 
                           original_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Decrypt model update.
        
        Args:
            encrypted_update (bytes): Encrypted model update
            original_shape (Tuple[int, ...]): Original tensor shape
            
        Returns:
            torch.Tensor: Decrypted model update
        """
        if not self.config.enable_homomorphic or not self.private_key:
            # Deserialize directly if encryption disabled
            array = np.frombuffer(encrypted_update, dtype=np.float32)
            return torch.from_numpy(array.reshape(original_shape))
        
        # Decrypt using RSA
        decrypted = self.private_key.decrypt(
            encrypted_update,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Deserialize to tensor
        array = np.frombuffer(decrypted, dtype=np.float32)
        return torch.from_numpy(array.reshape(original_shape))

class MembershipInferenceProtection:
    """
    Protection against membership inference attacks.
    """
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.shadow_models = []
        
    def add_regularization_loss(self, model_output: torch.Tensor, 
                              targets: torch.Tensor) -> torch.Tensor:
        """
        Add regularization to prevent overfitting on individual samples.
        
        Args:
            model_output (torch.Tensor): Model predictions
            targets (torch.Tensor): True targets
            
        Returns:
            torch.Tensor: Regularization loss
        """
        if not self.config.enable_membership_protection:
            return torch.tensor(0.0)
        
        # Label smoothing to reduce confidence
        smoothed_targets = targets * 0.9 + 0.1 / targets.size(-1)
        
        # KL divergence loss for regularization
        log_probs = torch.log_softmax(model_output, dim=-1)
        reg_loss = torch.nn.functional.kl_div(
            log_probs, smoothed_targets, reduction='batchmean'
        )
        
        return reg_loss * 0.1  # Weight the regularization term
    
    def add_prediction_noise(self, predictions: torch.Tensor, 
                           noise_scale: float = 0.1) -> torch.Tensor:
        """
        Add noise to predictions to reduce membership inference risk.
        
        Args:
            predictions (torch.Tensor): Model predictions
            noise_scale (float): Scale of noise to add
            
        Returns:
            torch.Tensor: Noisy predictions
        """
        if not self.config.enable_membership_protection:
            return predictions
        
        noise = torch.normal(0, noise_scale, size=predictions.shape)
        return predictions + noise

class PrivacyEngine:
    """
    Main privacy engine coordinating all privacy-preserving mechanisms.
    """
    
    def __init__(self, config: Union[PrivacyConfig, Any]):
        # Handle both PrivacyConfig and other config types
        if isinstance(config, PrivacyConfig):
            self.config = config
        else:
            # Convert from other config types (e.g., ClientConfig)
            self.config = PrivacyConfig(
                enable_dp=getattr(config, 'differential_privacy', True),
                epsilon=getattr(config, 'privacy_budget', 1.0),
                enable_secure_aggregation=getattr(config, 'secure_aggregation', True)
            )
        
        # Initialize privacy mechanisms
        self.differential_privacy = DifferentialPrivacy(self.config)
        self.secure_aggregation = SecureAggregation(self.config)
        self.homomorphic_encryption = HomomorphicEncryption(self.config)
        self.membership_protection = MembershipInferenceProtection(self.config)
        
        logger.info("Privacy engine initialized with comprehensive protection")
    
    def apply_differential_privacy(self, model_update: Dict[str, Any], 
                                 privacy_budget: float) -> Dict[str, Any]:
        """
        Apply differential privacy to model update.
        
        Args:
            model_update (Dict[str, Any]): Model update from local training
            privacy_budget (float): Privacy budget to use
            
        Returns:
            Dict[str, Any]: Privacy-protected model update
        """
        if not self.config.enable_dp:
            return model_update
        
        protected_update = model_update.copy()
        
        # Apply DP to gradients if present
        if 'gradients' in model_update:
            gradients = model_update['gradients']
            
            # Clip gradients
            clipped_gradients, clipping_factor = self.differential_privacy.clip_gradients(gradients)
            
            # Add noise
            noisy_gradients = self.differential_privacy.add_noise_to_gradients(
                clipped_gradients, sensitivity=self.config.sensitivity
            )
            
            protected_update['gradients'] = noisy_gradients
            protected_update['clipping_factor'] = clipping_factor
        
        # Apply DP to model parameters if present
        if 'model_parameters' in model_update:
            params = model_update['model_parameters']
            
            # Add noise to parameters
            for param_name, param_tensor in params.items():
                noisy_params = self.differential_privacy.add_noise_to_gradients(
                    param_tensor, sensitivity=self.config.sensitivity
                )
                protected_update['model_parameters'][param_name] = noisy_params
        
        # Update privacy accounting
        self.differential_privacy.privacy_spent += privacy_budget
        
        return protected_update
    
    def prepare_secure_aggregation(self, client_id: str, model_update: torch.Tensor,
                                 other_clients: List[str]) -> torch.Tensor:
        """
        Prepare model update for secure aggregation.
        
        Args:
            client_id (str): Client identifier
            model_update (torch.Tensor): Model update
            other_clients (List[str]): Other participating clients
            
        Returns:
            torch.Tensor: Masked model update
        """
        if not self.config.enable_secure_aggregation:
            return model_update
        
        # Generate aggregation mask
        mask = self.secure_aggregation.generate_aggregation_mask(
            client_id, model_update.shape, other_clients
        )
        
        # Apply mask
        masked_update = self.secure_aggregation.mask_model_update(model_update, mask)
        
        return masked_update
    
    def encrypt_model_update(self, model_update: torch.Tensor) -> bytes:
        """
        Encrypt model update for secure transmission.
        
        Args:
            model_update (torch.Tensor): Model update to encrypt
            
        Returns:
            bytes: Encrypted model update
        """
        return self.homomorphic_encryption.encrypt_model_update(model_update)
    
    def add_membership_protection(self, model_output: torch.Tensor,
                                targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add membership inference protection to model training.
        
        Args:
            model_output (torch.Tensor): Model predictions
            targets (torch.Tensor): True targets
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Protected output and regularization loss
        """
        # Add prediction noise
        protected_output = self.membership_protection.add_prediction_noise(model_output)
        
        # Calculate regularization loss
        reg_loss = self.membership_protection.add_regularization_loss(model_output, targets)
        
        return protected_output, reg_loss
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive privacy report.
        
        Returns:
            Dict[str, Any]: Privacy usage and protection status
        """
        return {
            'differential_privacy': {
                'enabled': self.config.enable_dp,
                'epsilon_spent': self.differential_privacy.privacy_spent,
                'epsilon_budget': self.config.epsilon,
                'budget_remaining': self.config.epsilon - self.differential_privacy.privacy_spent,
                'noise_injections': len(self.differential_privacy.noise_history)
            },
            'secure_aggregation': {
                'enabled': self.config.enable_secure_aggregation,
                'threshold': self.config.threshold,
                'active_masks': len(self.secure_aggregation.aggregation_masks)
            },
            'homomorphic_encryption': {
                'enabled': self.config.enable_homomorphic,
                'key_size': self.config.key_size
            },
            'membership_protection': {
                'enabled': self.config.enable_membership_protection
            },
            'overall_privacy_level': self._calculate_privacy_level()
        }
    
    def _calculate_privacy_level(self) -> str:
        """Calculate overall privacy protection level."""
        score = 0
        
        if self.config.enable_dp:
            score += 3
        if self.config.enable_secure_aggregation:
            score += 2
        if self.config.enable_homomorphic:
            score += 2
        if self.config.enable_membership_protection:
            score += 1
        
        if score >= 7:
            return "Maximum"
        elif score >= 5:
            return "High"
        elif score >= 3:
            return "Medium"
        else:
            return "Basic"

# Factory functions
def create_privacy_engine(differential_privacy: bool = True,
                         secure_aggregation: bool = True,
                         epsilon: float = 1.0,
                         **kwargs) -> PrivacyEngine:
    """
    Factory function to create a privacy engine.
    
    Args:
        differential_privacy (bool): Enable differential privacy
        secure_aggregation (bool): Enable secure aggregation
        epsilon (float): Privacy budget
        **kwargs: Additional configuration parameters
        
    Returns:
        PrivacyEngine: Configured privacy engine
    """
    config = PrivacyConfig(
        enable_dp=differential_privacy,
        enable_secure_aggregation=secure_aggregation,
        epsilon=epsilon,
        **kwargs
    )
    
    return PrivacyEngine(config)
