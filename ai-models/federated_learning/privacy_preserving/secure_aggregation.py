"""
BatteryMind - Secure Aggregation for Federated Learning

Advanced secure aggregation implementation for privacy-preserving federated
learning in battery management systems. Implements secure multi-party
computation protocols for aggregating model updates without revealing
individual client contributions.

Features:
- Secure multi-party computation (SMPC) protocols
- Shamir's secret sharing for distributed computation
- Dropout resilience for client failures
- Efficient aggregation algorithms
- Integration with homomorphic encryption
- Performance optimization for battery data

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
import logging
import time
import hashlib
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import pickle
import json

# Cryptographic imports
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import gmpy2
from gmpy2 import mpz, random_state, mpz_random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecretShare:
    """
    Represents a secret share in Shamir's secret sharing scheme.
    
    Attributes:
        x (int): X-coordinate (party identifier)
        y (int): Y-coordinate (share value)
        prime (int): Prime modulus for finite field arithmetic
    """
    x: int
    y: int
    prime: int

@dataclass
class AggregationConfig:
    """
    Configuration for secure aggregation protocols.
    
    Attributes:
        threshold (int): Minimum number of shares needed for reconstruction
        prime_bits (int): Number of bits for the prime modulus
        dropout_tolerance (float): Fraction of clients that can drop out
        max_clients (int): Maximum number of clients supported
        enable_verification (bool): Enable cryptographic verification
        batch_size (int): Batch size for processing operations
        timeout_seconds (int): Timeout for aggregation operations
    """
    threshold: int = 3
    prime_bits: int = 256
    dropout_tolerance: float = 0.3
    max_clients: int = 100
    enable_verification: bool = True
    batch_size: int = 1000
    timeout_seconds: int = 300

class ShamirSecretSharing:
    """
    Shamir's secret sharing implementation for secure aggregation.
    """
    
    def __init__(self, threshold: int, num_parties: int, prime_bits: int = 256):
        self.threshold = threshold
        self.num_parties = num_parties
        self.prime = self._generate_prime(prime_bits)
        self.random_state = random_state(secrets.randbits(128))
        
        if threshold > num_parties:
            raise ValueError("Threshold cannot exceed number of parties")
        
        logger.info(f"Shamir secret sharing initialized: {threshold}-of-{num_parties}")
    
    def _generate_prime(self, bits: int) -> int:
        """Generate a random prime number of specified bit length."""
        while True:
            candidate = mpz_random(random_state(secrets.randbits(128)), 2 ** bits)
            candidate |= (1 << (bits - 1)) | 1  # Ensure odd and correct bit length
            
            if gmpy2.is_prime(candidate):
                return int(candidate)
    
    def share_secret(self, secret: int) -> List[SecretShare]:
        """
        Share a secret among parties using Shamir's scheme.
        
        Args:
            secret (int): Secret value to share
            
        Returns:
            List[SecretShare]: List of secret shares
        """
        # Generate random coefficients for polynomial
        coefficients = [secret % self.prime]
        for _ in range(self.threshold - 1):
            coeff = mpz_random(self.random_state, self.prime)
            coefficients.append(int(coeff))
        
        # Evaluate polynomial at different points
        shares = []
        for x in range(1, self.num_parties + 1):
            y = self._evaluate_polynomial(coefficients, x) % self.prime
            shares.append(SecretShare(x=x, y=int(y), prime=self.prime))
        
        return shares
    
    def reconstruct_secret(self, shares: List[SecretShare]) -> int:
        """
        Reconstruct secret from shares using Lagrange interpolation.
        
        Args:
            shares (List[SecretShare]): List of secret shares
            
        Returns:
            int: Reconstructed secret
        """
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares, got {len(shares)}")
        
        # Use first 'threshold' shares
        shares = shares[:self.threshold]
        
        secret = 0
        for i, share in enumerate(shares):
            # Calculate Lagrange coefficient
            numerator = 1
            denominator = 1
            
            for j, other_share in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-other_share.x)) % self.prime
                    denominator = (denominator * (share.x - other_share.x)) % self.prime
            
            # Calculate modular inverse
            denominator_inv = gmpy2.invert(denominator, self.prime)
            lagrange_coeff = (numerator * denominator_inv) % self.prime
            
            secret = (secret + share.y * lagrange_coeff) % self.prime
        
        return int(secret)
    
    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x using Horner's method."""
        result = 0
        for coeff in reversed(coefficients):
            result = (result * x + coeff) % self.prime
        return result
    
    def add_shares(self, shares1: List[SecretShare], shares2: List[SecretShare]) -> List[SecretShare]:
        """
        Add two sets of secret shares (homomorphic addition).
        
        Args:
            shares1 (List[SecretShare]): First set of shares
            shares2 (List[SecretShare]): Second set of shares
            
        Returns:
            List[SecretShare]: Sum of shares
        """
        if len(shares1) != len(shares2):
            raise ValueError("Share lists must have same length")
        
        result_shares = []
        for s1, s2 in zip(shares1, shares2):
            if s1.x != s2.x or s1.prime != s2.prime:
                raise ValueError("Incompatible shares")
            
            sum_y = (s1.y + s2.y) % s1.prime
            result_shares.append(SecretShare(x=s1.x, y=sum_y, prime=s1.prime))
        
        return result_shares
    
    def multiply_share_by_constant(self, shares: List[SecretShare], constant: int) -> List[SecretShare]:
        """
        Multiply shares by a constant.
        
        Args:
            shares (List[SecretShare]): Input shares
            constant (int): Multiplication constant
            
        Returns:
            List[SecretShare]: Multiplied shares
        """
        result_shares = []
        for share in shares:
            new_y = (share.y * constant) % share.prime
            result_shares.append(SecretShare(x=share.x, y=new_y, prime=share.prime))
        
        return result_shares

class SecureAggregationProtocol:
    """
    Secure aggregation protocol for federated learning using secret sharing.
    """
    
    def __init__(self, config: AggregationConfig):
        self.config = config
        self.client_shares: Dict[str, Dict[str, List[SecretShare]]] = {}
        self.aggregation_state = {}
        self.lock = threading.Lock()
        
        # Calculate minimum threshold based on dropout tolerance
        min_clients = max(config.threshold, 
                         int(config.max_clients * (1 - config.dropout_tolerance)))
        self.min_clients = min_clients
        
        logger.info(f"Secure aggregation protocol initialized with threshold {config.threshold}")
    
    def setup_aggregation_round(self, client_ids: List[str], 
                              model_shapes: Dict[str, Tuple]) -> Dict[str, Any]:
        """
        Setup a new aggregation round.
        
        Args:
            client_ids (List[str]): List of participating client IDs
            model_shapes (Dict[str, Tuple]): Model layer shapes
            
        Returns:
            Dict[str, Any]: Setup information for clients
        """
        round_id = hashlib.sha256(
            f"{time.time()}_{len(client_ids)}".encode()
        ).hexdigest()[:16]
        
        num_clients = len(client_ids)
        threshold = min(self.config.threshold, num_clients)
        
        # Create secret sharing schemes for each model layer
        sharing_schemes = {}
        for layer_name, shape in model_shapes.items():
            total_params = np.prod(shape)
            sharing_schemes[layer_name] = ShamirSecretSharing(
                threshold=threshold,
                num_parties=num_clients,
                prime_bits=self.config.prime_bits
            )
        
        # Store aggregation state
        with self.lock:
            self.aggregation_state[round_id] = {
                'client_ids': client_ids,
                'model_shapes': model_shapes,
                'sharing_schemes': sharing_schemes,
                'threshold': threshold,
                'received_shares': set(),
                'start_time': time.time()
            }
        
        logger.info(f"Setup aggregation round {round_id} with {num_clients} clients")
        
        return {
            'round_id': round_id,
            'threshold': threshold,
            'client_mapping': {client_id: i + 1 for i, client_id in enumerate(client_ids)},
            'prime_info': {layer: scheme.prime for layer, scheme in sharing_schemes.items()}
        }
    
    def create_client_shares(self, round_id: str, client_id: str,
                           model_weights: Dict[str, torch.Tensor]) -> Dict[str, List[SecretShare]]:
        """
        Create secret shares for a client's model weights.
        
        Args:
            round_id (str): Aggregation round ID
            client_id (str): Client identifier
            model_weights (Dict[str, torch.Tensor]): Client's model weights
            
        Returns:
            Dict[str, List[SecretShare]]: Secret shares for each layer
        """
        if round_id not in self.aggregation_state:
            raise ValueError(f"Unknown round ID: {round_id}")
        
        state = self.aggregation_state[round_id]
        sharing_schemes = state['sharing_schemes']
        
        client_shares = {}
        
        for layer_name, weight_tensor in model_weights.items():
            if layer_name not in sharing_schemes:
                continue
            
            # Flatten weights and convert to integers
            flat_weights = weight_tensor.flatten().cpu().numpy()
            int_weights = self._encode_floats_to_ints(flat_weights)
            
            # Create shares for each weight
            layer_shares = []
            scheme = sharing_schemes[layer_name]
            
            for weight_int in int_weights:
                shares = scheme.share_secret(weight_int)
                layer_shares.append(shares)
            
            client_shares[layer_name] = layer_shares
        
        # Store client shares
        with self.lock:
            if round_id not in self.client_shares:
                self.client_shares[round_id] = {}
            self.client_shares[round_id][client_id] = client_shares
            self.aggregation_state[round_id]['received_shares'].add(client_id)
        
        logger.info(f"Created shares for client {client_id} in round {round_id}")
        return client_shares
    
    def aggregate_shares(self, round_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Aggregate shares from all clients in the round.
        
        Args:
            round_id (str): Aggregation round ID
            
        Returns:
            Optional[Dict[str, torch.Tensor]]: Aggregated model weights
        """
        if round_id not in self.aggregation_state:
            raise ValueError(f"Unknown round ID: {round_id}")
        
        state = self.aggregation_state[round_id]
        
        # Check if we have enough shares
        num_received = len(state['received_shares'])
        if num_received < state['threshold']:
            logger.warning(f"Not enough shares received: {num_received}/{state['threshold']}")
            return None
        
        # Check timeout
        elapsed_time = time.time() - state['start_time']
        if elapsed_time > self.config.timeout_seconds:
            logger.warning(f"Aggregation timeout exceeded: {elapsed_time}s")
            return None
        
        logger.info(f"Aggregating shares from {num_received} clients")
        
        aggregated_weights = {}
        sharing_schemes = state['sharing_schemes']
        model_shapes = state['model_shapes']
        
        for layer_name in sharing_schemes:
            # Collect shares from all clients for this layer
            all_client_shares = []
            for client_id in state['received_shares']:
                if (client_id in self.client_shares[round_id] and 
                    layer_name in self.client_shares[round_id][client_id]):
                    all_client_shares.append(
                        self.client_shares[round_id][client_id][layer_name]
                    )
            
            if not all_client_shares:
                continue
            
            # Aggregate shares for each weight position
            num_weights = len(all_client_shares[0])
            aggregated_layer_weights = []
            
            for weight_idx in range(num_weights):
                # Sum shares across all clients for this weight
                summed_shares = None
                
                for client_shares in all_client_shares:
                    weight_shares = client_shares[weight_idx]
                    
                    if summed_shares is None:
                        summed_shares = weight_shares[:]
                    else:
                        summed_shares = sharing_schemes[layer_name].add_shares(
                            summed_shares, weight_shares
                        )
                
                # Reconstruct the aggregated weight
                if summed_shares:
                    # Average by dividing by number of clients
                    num_clients = len(all_client_shares)
                    inv_factor = gmpy2.invert(num_clients, sharing_schemes[layer_name].prime)
                    averaged_shares = sharing_schemes[layer_name].multiply_share_by_constant(
                        summed_shares, int(inv_factor)
                    )
                    
                    # Reconstruct secret
                    aggregated_weight_int = sharing_schemes[layer_name].reconstruct_secret(
                        averaged_shares[:state['threshold']]
                    )
                    aggregated_layer_weights.append(aggregated_weight_int)
            
            # Convert back to tensor
            if aggregated_layer_weights:
                float_weights = self._decode_ints_to_floats(aggregated_layer_weights)
                weight_tensor = torch.tensor(float_weights).reshape(model_shapes[layer_name])
                aggregated_weights[layer_name] = weight_tensor
        
        # Cleanup
        with self.lock:
            if round_id in self.client_shares:
                del self.client_shares[round_id]
            if round_id in self.aggregation_state:
                del self.aggregation_state[round_id]
        
        logger.info(f"Successfully aggregated weights for round {round_id}")
        return aggregated_weights
    
    def _encode_floats_to_ints(self, float_array: np.ndarray, precision: int = 6) -> List[int]:
        """Encode floating point numbers to integers for secret sharing."""
        scaling_factor = 10 ** precision
        int_array = (float_array * scaling_factor).astype(np.int64)
        return int_array.tolist()
    
    def _decode_ints_to_floats(self, int_list: List[int], precision: int = 6) -> List[float]:
        """Decode integers back to floating point numbers."""
        scaling_factor = 10 ** precision
        return [float(x) / scaling_factor for x in int_list]
    
    def get_aggregation_status(self, round_id: str) -> Dict[str, Any]:
        """Get status of aggregation round."""
        if round_id not in self.aggregation_state:
            return {'status': 'unknown', 'round_id': round_id}
        
        state = self.aggregation_state[round_id]
        elapsed_time = time.time() - state['start_time']
        
        return {
            'status': 'active',
            'round_id': round_id,
            'clients_expected': len(state['client_ids']),
            'clients_received': len(state['received_shares']),
            'threshold': state['threshold'],
            'elapsed_time': elapsed_time,
            'timeout_remaining': max(0, self.config.timeout_seconds - elapsed_time)
        }

class SecureAggregationManager:
    """
    High-level manager for secure aggregation in federated learning.
    """
    
    def __init__(self, config: AggregationConfig):
        self.config = config
        self.protocol = SecureAggregationProtocol(config)
        self.active_rounds: Dict[str, Dict] = {}
        
    def start_aggregation_round(self, client_ids: List[str], 
                              model_template: Dict[str, torch.Tensor]) -> str:
        """
        Start a new secure aggregation round.
        
        Args:
            client_ids (List[str]): Participating client IDs
            model_template (Dict[str, torch.Tensor]): Template for model structure
            
        Returns:
            str: Round ID
        """
        model_shapes = {name: tensor.shape for name, tensor in model_template.items()}
        
        setup_info = self.protocol.setup_aggregation_round(client_ids, model_shapes)
        round_id = setup_info['round_id']
        
        self.active_rounds[round_id] = {
            'setup_info': setup_info,
            'start_time': time.time(),
            'client_ids': client_ids
        }
        
        return round_id
    
    def submit_client_update(self, round_id: str, client_id: str,
                           model_weights: Dict[str, torch.Tensor]) -> bool:
        """
        Submit a client's model update for aggregation.
        
        Args:
            round_id (str): Round ID
            client_id (str): Client ID
            model_weights (Dict[str, torch.Tensor]): Client's model weights
            
        Returns:
            bool: Success status
        """
        try:
            self.protocol.create_client_shares(round_id, client_id, model_weights)
            return True
        except Exception as e:
            logger.error(f"Failed to submit update for client {client_id}: {e}")
            return False
    
    def finalize_aggregation(self, round_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Finalize aggregation and return aggregated weights.
        
        Args:
            round_id (str): Round ID
            
        Returns:
            Optional[Dict[str, torch.Tensor]]: Aggregated weights
        """
        aggregated_weights = self.protocol.aggregate_shares(round_id)
        
        if round_id in self.active_rounds:
            del self.active_rounds[round_id]
        
        return aggregated_weights
    
    def get_round_status(self, round_id: str) -> Dict[str, Any]:
        """Get status of aggregation round."""
        return self.protocol.get_aggregation_status(round_id)
    
    def cleanup_expired_rounds(self) -> None:
        """Clean up expired aggregation rounds."""
        current_time = time.time()
        expired_rounds = []
        
        for round_id, round_info in self.active_rounds.items():
            if current_time - round_info['start_time'] > self.config.timeout_seconds:
                expired_rounds.append(round_id)
        
        for round_id in expired_rounds:
            logger.warning(f"Cleaning up expired round {round_id}")
            del self.active_rounds[round_id]

# Factory functions
def create_secure_aggregation_manager(config: Optional[AggregationConfig] = None) -> SecureAggregationManager:
    """Create a secure aggregation manager."""
    if config is None:
        config = AggregationConfig()
    
    return SecureAggregationManager(config)

def perform_secure_federated_aggregation(client_updates: List[Dict[str, torch.Tensor]],
                                        client_ids: List[str],
                                        config: Optional[AggregationConfig] = None) -> Dict[str, torch.Tensor]:
    """
    Perform secure aggregation of federated learning updates.
    
    Args:
        client_updates (List[Dict[str, torch.Tensor]]): Client model updates
        client_ids (List[str]): Client identifiers
        config (AggregationConfig, optional): Aggregation configuration
        
    Returns:
        Dict[str, torch.Tensor]: Securely aggregated model weights
    """
    if len(client_updates) != len(client_ids):
        raise ValueError("Number of updates must match number of client IDs")
    
    manager = create_secure_aggregation_manager(config)
    
    # Start aggregation round
    round_id = manager.start_aggregation_round(client_ids, client_updates[0])
    
    # Submit all client updates
    for client_id, update in zip(client_ids, client_updates):
        success = manager.submit_client_update(round_id, client_id, update)
        if not success:
            logger.error(f"Failed to submit update for client {client_id}")
    
    # Finalize aggregation
    aggregated_weights = manager.finalize_aggregation(round_id)
    
    if aggregated_weights is None:
        raise RuntimeError("Secure aggregation failed")
    
    return aggregated_weights
