"""
BatteryMind - Homomorphic Encryption for Federated Learning

Advanced homomorphic encryption implementation for secure federated learning
in battery management systems. Enables computation on encrypted data while
preserving privacy across distributed battery clients.

Features:
- Paillier homomorphic encryption for secure aggregation
- Batch encryption/decryption for efficiency
- Key management and rotation
- Secure multi-party computation primitives
- Integration with federated learning workflows
- Performance optimization for battery data

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import time
import pickle
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import secrets

# Cryptographic imports
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import gmpy2
from gmpy2 import mpz, random_state, mpz_random, powmod, invert

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PaillierKeyPair:
    """
    Paillier cryptosystem key pair for homomorphic encryption.
    
    Attributes:
        n (int): Public key modulus
        g (int): Public key generator
        lambda_val (int): Private key lambda value
        mu (int): Private key mu value
        bit_length (int): Key bit length for security
    """
    n: int
    g: int
    lambda_val: int
    mu: int
    bit_length: int
    
    def get_public_key(self) -> Tuple[int, int]:
        """Get public key components."""
        return (self.n, self.g)
    
    def get_private_key(self) -> Tuple[int, int]:
        """Get private key components."""
        return (self.lambda_val, self.mu)

@dataclass
class EncryptionConfig:
    """
    Configuration for homomorphic encryption operations.
    
    Attributes:
        key_size (int): RSA key size in bits
        precision (int): Decimal precision for floating point encoding
        batch_size (int): Batch size for encryption operations
        enable_caching (bool): Enable encryption result caching
        max_cache_size (int): Maximum cache size in MB
        threading_enabled (bool): Enable multi-threading for operations
        max_workers (int): Maximum number of worker threads
    """
    key_size: int = 2048
    precision: int = 6
    batch_size: int = 1000
    enable_caching: bool = True
    max_cache_size: int = 512
    threading_enabled: bool = True
    max_workers: int = 4

class PaillierCryptosystem:
    """
    Paillier homomorphic encryption implementation optimized for federated learning.
    
    Features:
    - Additive homomorphic properties
    - Secure key generation and management
    - Efficient batch operations
    - Floating point number support
    - Thread-safe operations
    """
    
    def __init__(self, config: EncryptionConfig):
        self.config = config
        self.key_pair: Optional[PaillierKeyPair] = None
        self.random_state = random_state(secrets.randbits(128))
        
        # Performance optimization
        self.encryption_cache = {} if config.enable_caching else None
        self.cache_lock = threading.Lock()
        
        # Thread pool for parallel operations
        if config.threading_enabled:
            self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        else:
            self.executor = None
        
        logger.info(f"Paillier cryptosystem initialized with {config.key_size}-bit keys")
    
    def generate_keypair(self) -> PaillierKeyPair:
        """
        Generate a new Paillier key pair.
        
        Returns:
            PaillierKeyPair: Generated key pair
        """
        logger.info("Generating Paillier key pair...")
        start_time = time.time()
        
        # Generate two large primes p and q
        bit_length = self.config.key_size // 2
        
        while True:
            p = self._generate_prime(bit_length)
            q = self._generate_prime(bit_length)
            
            # Ensure p != q and gcd(pq, (p-1)(q-1)) = 1
            if p != q:
                n = p * q
                lambda_val = self._lcm(p - 1, q - 1)
                
                # Check if n and lambda are coprime
                if gmpy2.gcd(n, lambda_val) == 1:
                    break
        
        # Calculate g (generator)
        g = n + 1  # Simple choice that works well
        
        # Calculate mu = (L(g^lambda mod n^2))^(-1) mod n
        n_squared = n * n
        g_lambda = powmod(g, lambda_val, n_squared)
        l_result = self._l_function(g_lambda, n)
        mu = invert(l_result, n)
        
        self.key_pair = PaillierKeyPair(
            n=int(n),
            g=int(g),
            lambda_val=int(lambda_val),
            mu=int(mu),
            bit_length=self.config.key_size
        )
        
        generation_time = time.time() - start_time
        logger.info(f"Key pair generated in {generation_time:.2f} seconds")
        
        return self.key_pair
    
    def _generate_prime(self, bit_length: int) -> int:
        """Generate a random prime number of specified bit length."""
        while True:
            candidate = mpz_random(self.random_state, 2 ** bit_length)
            candidate |= (1 << (bit_length - 1)) | 1  # Ensure it's odd and has correct bit length
            
            if gmpy2.is_prime(candidate):
                return candidate
    
    def _lcm(self, a: int, b: int) -> int:
        """Calculate least common multiple."""
        return abs(a * b) // gmpy2.gcd(a, b)
    
    def _l_function(self, x: int, n: int) -> int:
        """L function: L(x) = (x - 1) / n."""
        return (x - 1) // n
    
    def encrypt(self, plaintext: Union[float, int, List[Union[float, int]]]) -> Union[int, List[int]]:
        """
        Encrypt plaintext using Paillier encryption.
        
        Args:
            plaintext: Value(s) to encrypt
            
        Returns:
            Encrypted value(s)
        """
        if self.key_pair is None:
            raise ValueError("Key pair not generated. Call generate_keypair() first.")
        
        if isinstance(plaintext, (list, np.ndarray)):
            return self._encrypt_batch(plaintext)
        else:
            return self._encrypt_single(plaintext)
    
    def _encrypt_single(self, plaintext: Union[float, int]) -> int:
        """Encrypt a single value."""
        # Encode floating point to integer
        encoded = self._encode_float(plaintext)
        
        # Check cache
        if self.encryption_cache is not None:
            cache_key = f"enc_{encoded}"
            with self.cache_lock:
                if cache_key in self.encryption_cache:
                    return self.encryption_cache[cache_key]
        
        # Paillier encryption: c = g^m * r^n mod n^2
        n = self.key_pair.n
        g = self.key_pair.g
        n_squared = n * n
        
        # Generate random r
        r = mpz_random(self.random_state, n)
        while gmpy2.gcd(r, n) != 1:
            r = mpz_random(self.random_state, n)
        
        # Encrypt
        g_m = powmod(g, encoded, n_squared)
        r_n = powmod(r, n, n_squared)
        ciphertext = (g_m * r_n) % n_squared
        
        result = int(ciphertext)
        
        # Cache result
        if self.encryption_cache is not None:
            with self.cache_lock:
                if len(self.encryption_cache) < self.config.max_cache_size:
                    self.encryption_cache[cache_key] = result
        
        return result
    
    def _encrypt_batch(self, plaintexts: List[Union[float, int]]) -> List[int]:
        """Encrypt multiple values efficiently."""
        if self.executor and len(plaintexts) > self.config.batch_size:
            # Parallel encryption for large batches
            chunks = [plaintexts[i:i + self.config.batch_size] 
                     for i in range(0, len(plaintexts), self.config.batch_size)]
            
            futures = [self.executor.submit(self._encrypt_chunk, chunk) for chunk in chunks]
            results = []
            for future in futures:
                results.extend(future.result())
            
            return results
        else:
            # Sequential encryption
            return [self._encrypt_single(pt) for pt in plaintexts]
    
    def _encrypt_chunk(self, chunk: List[Union[float, int]]) -> List[int]:
        """Encrypt a chunk of values."""
        return [self._encrypt_single(pt) for pt in chunk]
    
    def decrypt(self, ciphertext: Union[int, List[int]]) -> Union[float, List[float]]:
        """
        Decrypt ciphertext using Paillier decryption.
        
        Args:
            ciphertext: Encrypted value(s) to decrypt
            
        Returns:
            Decrypted value(s)
        """
        if self.key_pair is None:
            raise ValueError("Key pair not generated. Call generate_keypair() first.")
        
        if isinstance(ciphertext, list):
            return self._decrypt_batch(ciphertext)
        else:
            return self._decrypt_single(ciphertext)
    
    def _decrypt_single(self, ciphertext: int) -> float:
        """Decrypt a single value."""
        # Paillier decryption: m = L(c^lambda mod n^2) * mu mod n
        n = self.key_pair.n
        lambda_val = self.key_pair.lambda_val
        mu = self.key_pair.mu
        n_squared = n * n
        
        c_lambda = powmod(ciphertext, lambda_val, n_squared)
        l_result = self._l_function(c_lambda, n)
        plaintext = (l_result * mu) % n
        
        # Decode integer back to float
        return self._decode_float(int(plaintext))
    
    def _decrypt_batch(self, ciphertexts: List[int]) -> List[float]:
        """Decrypt multiple values efficiently."""
        if self.executor and len(ciphertexts) > self.config.batch_size:
            # Parallel decryption for large batches
            chunks = [ciphertexts[i:i + self.config.batch_size] 
                     for i in range(0, len(ciphertexts), self.config.batch_size)]
            
            futures = [self.executor.submit(self._decrypt_chunk, chunk) for chunk in chunks]
            results = []
            for future in futures:
                results.extend(future.result())
            
            return results
        else:
            # Sequential decryption
            return [self._decrypt_single(ct) for ct in ciphertexts]
    
    def _decrypt_chunk(self, chunk: List[int]) -> List[float]:
        """Decrypt a chunk of values."""
        return [self._decrypt_single(ct) for ct in chunk]
    
    def add_encrypted(self, ciphertext1: int, ciphertext2: int) -> int:
        """
        Add two encrypted values using homomorphic property.
        
        Args:
            ciphertext1: First encrypted value
            ciphertext2: Second encrypted value
            
        Returns:
            Encrypted sum
        """
        if self.key_pair is None:
            raise ValueError("Key pair not generated.")
        
        n_squared = self.key_pair.n * self.key_pair.n
        return (ciphertext1 * ciphertext2) % n_squared
    
    def multiply_encrypted_by_constant(self, ciphertext: int, constant: Union[float, int]) -> int:
        """
        Multiply encrypted value by a plaintext constant.
        
        Args:
            ciphertext: Encrypted value
            constant: Plaintext constant
            
        Returns:
            Encrypted result
        """
        if self.key_pair is None:
            raise ValueError("Key pair not generated.")
        
        encoded_constant = self._encode_float(constant)
        n_squared = self.key_pair.n * self.key_pair.n
        return powmod(ciphertext, encoded_constant, n_squared)
    
    def _encode_float(self, value: Union[float, int]) -> int:
        """Encode floating point number to integer for encryption."""
        scaling_factor = 10 ** self.config.precision
        return int(value * scaling_factor)
    
    def _decode_float(self, encoded_value: int) -> float:
        """Decode integer back to floating point number."""
        scaling_factor = 10 ** self.config.precision
        
        # Handle negative numbers (two's complement)
        n = self.key_pair.n
        if encoded_value > n // 2:
            encoded_value = encoded_value - n
        
        return float(encoded_value) / scaling_factor
    
    def save_keys(self, filepath: str) -> None:
        """Save key pair to file."""
        if self.key_pair is None:
            raise ValueError("No key pair to save.")
        
        key_data = {
            'n': str(self.key_pair.n),
            'g': str(self.key_pair.g),
            'lambda_val': str(self.key_pair.lambda_val),
            'mu': str(self.key_pair.mu),
            'bit_length': self.key_pair.bit_length
        }
        
        with open(filepath, 'w') as f:
            json.dump(key_data, f, indent=2)
        
        logger.info(f"Keys saved to {filepath}")
    
    def load_keys(self, filepath: str) -> None:
        """Load key pair from file."""
        with open(filepath, 'r') as f:
            key_data = json.load(f)
        
        self.key_pair = PaillierKeyPair(
            n=int(key_data['n']),
            g=int(key_data['g']),
            lambda_val=int(key_data['lambda_val']),
            mu=int(key_data['mu']),
            bit_length=key_data['bit_length']
        )
        
        logger.info(f"Keys loaded from {filepath}")
    
    def get_public_key_info(self) -> Dict[str, Any]:
        """Get public key information for sharing."""
        if self.key_pair is None:
            raise ValueError("No key pair generated.")
        
        return {
            'n': str(self.key_pair.n),
            'g': str(self.key_pair.g),
            'bit_length': self.key_pair.bit_length
        }

class HomomorphicEncryptionManager:
    """
    High-level manager for homomorphic encryption in federated learning.
    """
    
    def __init__(self, config: EncryptionConfig):
        self.config = config
        self.cryptosystem = PaillierCryptosystem(config)
        self.is_initialized = False
        
    def initialize(self, key_path: Optional[str] = None) -> None:
        """Initialize the encryption system."""
        if key_path and Path(key_path).exists():
            self.cryptosystem.load_keys(key_path)
        else:
            self.cryptosystem.generate_keypair()
            if key_path:
                self.cryptosystem.save_keys(key_path)
        
        self.is_initialized = True
        logger.info("Homomorphic encryption manager initialized")
    
    def encrypt_model_weights(self, weights: Dict[str, torch.Tensor]) -> Dict[str, List[int]]:
        """
        Encrypt model weights for secure aggregation.
        
        Args:
            weights: Model weights dictionary
            
        Returns:
            Encrypted weights dictionary
        """
        if not self.is_initialized:
            raise ValueError("Encryption manager not initialized")
        
        encrypted_weights = {}
        
        for layer_name, weight_tensor in weights.items():
            # Flatten tensor and convert to list
            flat_weights = weight_tensor.flatten().cpu().numpy().tolist()
            
            # Encrypt weights
            encrypted_flat = self.cryptosystem.encrypt(flat_weights)
            encrypted_weights[layer_name] = encrypted_flat
            
            logger.debug(f"Encrypted {len(flat_weights)} weights for layer {layer_name}")
        
        return encrypted_weights
    
    def decrypt_model_weights(self, encrypted_weights: Dict[str, List[int]], 
                            original_shapes: Dict[str, Tuple]) -> Dict[str, torch.Tensor]:
        """
        Decrypt model weights after aggregation.
        
        Args:
            encrypted_weights: Encrypted weights dictionary
            original_shapes: Original tensor shapes
            
        Returns:
            Decrypted weights as tensors
        """
        if not self.is_initialized:
            raise ValueError("Encryption manager not initialized")
        
        decrypted_weights = {}
        
        for layer_name, encrypted_flat in encrypted_weights.items():
            # Decrypt weights
            decrypted_flat = self.cryptosystem.decrypt(encrypted_flat)
            
            # Reshape back to original tensor shape
            original_shape = original_shapes[layer_name]
            decrypted_tensor = torch.tensor(decrypted_flat).reshape(original_shape)
            decrypted_weights[layer_name] = decrypted_tensor
            
            logger.debug(f"Decrypted {len(decrypted_flat)} weights for layer {layer_name}")
        
        return decrypted_weights
    
    def aggregate_encrypted_weights(self, encrypted_weights_list: List[Dict[str, List[int]]]) -> Dict[str, List[int]]:
        """
        Aggregate encrypted weights from multiple clients.
        
        Args:
            encrypted_weights_list: List of encrypted weights from clients
            
        Returns:
            Aggregated encrypted weights
        """
        if not encrypted_weights_list:
            raise ValueError("No encrypted weights to aggregate")
        
        # Initialize aggregated weights with first client's weights
        aggregated = encrypted_weights_list[0].copy()
        
        # Add remaining clients' weights
        for client_weights in encrypted_weights_list[1:]:
            for layer_name in aggregated:
                if layer_name in client_weights:
                    # Element-wise addition of encrypted values
                    for i in range(len(aggregated[layer_name])):
                        aggregated[layer_name][i] = self.cryptosystem.add_encrypted(
                            aggregated[layer_name][i],
                            client_weights[layer_name][i]
                        )
        
        # Divide by number of clients (multiply by 1/n)
        num_clients = len(encrypted_weights_list)
        weight_factor = 1.0 / num_clients
        
        for layer_name in aggregated:
            for i in range(len(aggregated[layer_name])):
                aggregated[layer_name][i] = self.cryptosystem.multiply_encrypted_by_constant(
                    aggregated[layer_name][i],
                    weight_factor
                )
        
        logger.info(f"Aggregated encrypted weights from {num_clients} clients")
        return aggregated
    
    def get_encryption_stats(self) -> Dict[str, Any]:
        """Get encryption system statistics."""
        return {
            'is_initialized': self.is_initialized,
            'key_size': self.config.key_size,
            'precision': self.config.precision,
            'cache_enabled': self.config.enable_caching,
            'threading_enabled': self.config.threading_enabled,
            'public_key_info': self.cryptosystem.get_public_key_info() if self.is_initialized else None
        }

# Factory functions
def create_homomorphic_encryption_manager(config: Optional[EncryptionConfig] = None) -> HomomorphicEncryptionManager:
    """Create a homomorphic encryption manager."""
    if config is None:
        config = EncryptionConfig()
    
    return HomomorphicEncryptionManager(config)

def encrypt_federated_model_update(model_weights: Dict[str, torch.Tensor],
                                 encryption_manager: HomomorphicEncryptionManager) -> Dict[str, Any]:
    """
    Encrypt model update for federated learning.
    
    Args:
        model_weights: Model weights to encrypt
        encryption_manager: Encryption manager instance
        
    Returns:
        Encrypted model update with metadata
    """
    # Store original shapes
    original_shapes = {name: tensor.shape for name, tensor in model_weights.items()}
    
    # Encrypt weights
    encrypted_weights = encryption_manager.encrypt_model_weights(model_weights)
    
    return {
        'encrypted_weights': encrypted_weights,
        'original_shapes': original_shapes,
        'encryption_info': encryption_manager.get_encryption_stats()
    }
