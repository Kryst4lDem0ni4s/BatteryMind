"""
BatteryMind - Federated Learning Security Utilities

Comprehensive security utilities for federated learning including encryption,
authentication, secure communication, and privacy-preserving mechanisms.

Features:
- End-to-end encryption for model parameters and gradients
- Client authentication and authorization
- Secure key exchange and management
- Digital signatures for data integrity
- Privacy-preserving aggregation protocols
- Secure multi-party computation primitives
- Attack detection and mitigation

Author: BatteryMind Development Team
Version: 1.0.0
"""

import os
import hashlib
import hmac
import secrets
import time
import base64
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import logging
from enum import Enum
from pathlib import Path

# Cryptographic libraries
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.exceptions import InvalidSignature
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logger.warning("Cryptography library not available. Security features will be limited.")

# JWT for token-based authentication
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("PyJWT library not available. Token authentication will be limited.")

import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    AES_256_GCM = "aes_256_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_OAEP = "rsa_oaep"

class HashAlgorithm(Enum):
    """Supported hash algorithms."""
    SHA256 = "sha256"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"

class SignatureAlgorithm(Enum):
    """Supported digital signature algorithms."""
    RSA_PSS = "rsa_pss"
    ECDSA = "ecdsa"

@dataclass
class SecurityConfig:
    """
    Security configuration for federated learning.
    
    Attributes:
        encryption_algorithm (EncryptionAlgorithm): Encryption algorithm to use
        hash_algorithm (HashAlgorithm): Hash algorithm for integrity checks
        signature_algorithm (SignatureAlgorithm): Digital signature algorithm
        key_size (int): Key size in bits
        token_expiry_hours (int): JWT token expiry time in hours
        max_failed_attempts (int): Maximum failed authentication attempts
        enable_perfect_forward_secrecy (bool): Enable perfect forward secrecy
        enable_client_verification (bool): Enable client certificate verification
        audit_logging (bool): Enable security audit logging
    """
    encryption_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    signature_algorithm: SignatureAlgorithm = SignatureAlgorithm.RSA_PSS
    key_size: int = 2048
    token_expiry_hours: int = 24
    max_failed_attempts: int = 3
    enable_perfect_forward_secrecy: bool = True
    enable_client_verification: bool = True
    audit_logging: bool = True

class CryptographicKeyManager:
    """
    Manages cryptographic keys for federated learning security.
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.keys = {}
        self.key_history = []
        
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library required for key management")
    
    def generate_symmetric_key(self, algorithm: EncryptionAlgorithm = None) -> bytes:
        """
        Generate a symmetric encryption key.
        
        Args:
            algorithm (EncryptionAlgorithm, optional): Encryption algorithm
            
        Returns:
            bytes: Generated symmetric key
        """
        if algorithm is None:
            algorithm = self.config.encryption_algorithm
        
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            return secrets.token_bytes(32)  # 256 bits
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return secrets.token_bytes(32)  # 256 bits
        else:
            raise ValueError(f"Unsupported symmetric algorithm: {algorithm}")
    
    def generate_asymmetric_keypair(self, algorithm: SignatureAlgorithm = None) -> Tuple[bytes, bytes]:
        """
        Generate an asymmetric key pair.
        
        Args:
            algorithm (SignatureAlgorithm, optional): Signature algorithm
            
        Returns:
            Tuple[bytes, bytes]: (private_key, public_key) in PEM format
        """
        if algorithm is None:
            algorithm = self.config.signature_algorithm
        
        if algorithm == SignatureAlgorithm.RSA_PSS:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.config.key_size
            )
            public_key = private_key.public_key()
            
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return private_pem, public_pem
            
        elif algorithm == SignatureAlgorithm.ECDSA:
            private_key = ec.generate_private_key(ec.SECP256R1())
            public_key = private_key.public_key()
            
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return private_pem, public_pem
        else:
            raise ValueError(f"Unsupported asymmetric algorithm: {algorithm}")
    
    def derive_key(self, password: str, salt: bytes = None) -> bytes:
        """
        Derive a key from a password using PBKDF2.
        
        Args:
            password (str): Password to derive key from
            salt (bytes, optional): Salt for key derivation
            
        Returns:
            bytes: Derived key
        """
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        return kdf.derive(password.encode('utf-8'))
    
    def store_key(self, key_id: str, key_data: bytes, key_type: str = "symmetric") -> None:
        """Store a key securely."""
        self.keys[key_id] = {
            'data': key_data,
            'type': key_type,
            'created_at': time.time(),
            'usage_count': 0
        }
        
        # Keep key history for rotation
        self.key_history.append({
            'key_id': key_id,
            'created_at': time.time(),
            'action': 'created'
        })
    
    def get_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve a stored key."""
        if key_id in self.keys:
            self.keys[key_id]['usage_count'] += 1
            return self.keys[key_id]['data']
        return None
    
    def rotate_key(self, key_id: str) -> bytes:
        """Rotate a symmetric key."""
        if key_id not in self.keys:
            raise ValueError(f"Key {key_id} not found")
        
        # Generate new key
        new_key = self.generate_symmetric_key()
        
        # Archive old key
        old_key = self.keys[key_id]
        archived_key_id = f"{key_id}_archived_{int(time.time())}"
        self.keys[archived_key_id] = old_key
        
        # Store new key
        self.store_key(key_id, new_key)
        
        # Log rotation
        self.key_history.append({
            'key_id': key_id,
            'created_at': time.time(),
            'action': 'rotated'
        })
        
        return new_key

class EncryptionManager:
    """
    Handles encryption and decryption operations for federated learning.
    """
    
    def __init__(self, config: SecurityConfig, key_manager: CryptographicKeyManager):
        self.config = config
        self.key_manager = key_manager
        
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library required for encryption")
    
    def encrypt_data(self, data: bytes, key: bytes, 
                    algorithm: EncryptionAlgorithm = None) -> Tuple[bytes, bytes]:
        """
        Encrypt data using specified algorithm.
        
        Args:
            data (bytes): Data to encrypt
            key (bytes): Encryption key
            algorithm (EncryptionAlgorithm, optional): Encryption algorithm
            
        Returns:
            Tuple[bytes, bytes]: (encrypted_data, nonce/iv)
        """
        if algorithm is None:
            algorithm = self.config.encryption_algorithm
        
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
            aesgcm = AESGCM(key)
            ciphertext = aesgcm.encrypt(nonce, data, None)
            return ciphertext, nonce
            
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            nonce = secrets.token_bytes(12)  # 96-bit nonce
            chacha = ChaCha20Poly1305(key)
            ciphertext = chacha.encrypt(nonce, data, None)
            return ciphertext, nonce
            
        else:
            raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
    
    def decrypt_data(self, ciphertext: bytes, key: bytes, nonce: bytes,
                    algorithm: EncryptionAlgorithm = None) -> bytes:
        """
        Decrypt data using specified algorithm.
        
        Args:
            ciphertext (bytes): Encrypted data
            key (bytes): Decryption key
            nonce (bytes): Nonce/IV used for encryption
            algorithm (EncryptionAlgorithm, optional): Encryption algorithm
            
        Returns:
            bytes: Decrypted data
        """
        if algorithm is None:
            algorithm = self.config.encryption_algorithm
        
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            aesgcm = AESGCM(key)
            return aesgcm.decrypt(nonce, ciphertext, None)
            
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            chacha = ChaCha20Poly1305(key)
            return chacha.decrypt(nonce, ciphertext, None)
            
        else:
            raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
    
    def encrypt_model_parameters(self, parameters: Dict[str, torch.Tensor],
                                key: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """
        Encrypt model parameters for secure transmission.
        
        Args:
            parameters (Dict[str, torch.Tensor]): Model parameters
            key (bytes): Encryption key
            
        Returns:
            Tuple[bytes, Dict[str, Any]]: (encrypted_data, encryption_metadata)
        """
        # Serialize parameters
        import pickle
        serialized_params = pickle.dumps(parameters, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Encrypt
        ciphertext, nonce = self.encrypt_data(serialized_params, key)
        
        # Create metadata
        metadata = {
            'algorithm': self.config.encryption_algorithm.value,
            'nonce': base64.b64encode(nonce).decode('utf-8'),
            'encrypted_at': time.time(),
            'data_hash': hashlib.sha256(serialized_params).hexdigest()
        }
        
        return ciphertext, metadata
    
    def decrypt_model_parameters(self, ciphertext: bytes, key: bytes,
                               metadata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Decrypt model parameters.
        
        Args:
            ciphertext (bytes): Encrypted parameters
            key (bytes): Decryption key
            metadata (Dict[str, Any]): Encryption metadata
            
        Returns:
            Dict[str, torch.Tensor]: Decrypted parameters
        """
        # Extract metadata
        algorithm = EncryptionAlgorithm(metadata['algorithm'])
        nonce = base64.b64decode(metadata['nonce'])
        
        # Decrypt
        decrypted_data = self.decrypt_data(ciphertext, key, nonce, algorithm)
        
        # Verify integrity
        if 'data_hash' in metadata:
            calculated_hash = hashlib.sha256(decrypted_data).hexdigest()
            if calculated_hash != metadata['data_hash']:
                raise ValueError("Data integrity check failed")
        
        # Deserialize
        import pickle
        return pickle.loads(decrypted_data)

class DigitalSignatureManager:
    """
    Manages digital signatures for data integrity and authentication.
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("Cryptography library required for digital signatures")
        
        # Initialize key pairs for signing
        self.private_key = None
        self.public_key = None
        self.peer_public_keys = {}
        
        # Generate or load key pair
        self._initialize_keys()
    
    def _initialize_keys(self) -> None:
        """Initialize or load cryptographic keys."""
        try:
            # Try to load existing keys
            if hasattr(self.config, 'private_key_path') and os.path.exists(self.config.private_key_path):
                self._load_private_key()
            else:
                # Generate new key pair
                self._generate_key_pair()
                
        except Exception as e:
            logger.error(f"Failed to initialize keys: {e}")
            # Generate new keys as fallback
            self._generate_key_pair()
    
    def _generate_key_pair(self) -> None:
        """Generate new RSA key pair for signing."""
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.primitives import serialization
            
            # Generate private key
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Get public key
            self.public_key = self.private_key.public_key()
            
            logger.info("Generated new RSA key pair for digital signatures")
            
        except ImportError:
            logger.warning("RSA key generation not available - using HMAC signatures")
            # Fallback to HMAC-based signatures
            self.private_key = secrets.token_bytes(32)
            self.public_key = None
    
    def _load_private_key(self) -> None:
        """Load private key from file."""
        try:
            from cryptography.hazmat.primitives import serialization
            
            with open(self.config.private_key_path, 'rb') as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
            
            self.public_key = self.private_key.public_key()
            logger.info("Loaded private key from file")
            
        except Exception as e:
            logger.error(f"Failed to load private key: {e}")
            raise
    
    def save_keys(self, private_key_path: str, public_key_path: str) -> None:
        """Save key pair to files."""
        try:
            from cryptography.hazmat.primitives import serialization
            
            if self.private_key and hasattr(self.private_key, 'private_bytes'):
                # Save private key
                with open(private_key_path, 'wb') as f:
                    f.write(self.private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    ))
                
                # Save public key
                with open(public_key_path, 'wb') as f:
                    f.write(self.public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    ))
                
                logger.info(f"Saved keys to {private_key_path} and {public_key_path}")
            
        except Exception as e:
            logger.error(f"Failed to save keys: {e}")
            raise
    
    def sign_data(self, data: bytes) -> bytes:
        """
        Create digital signature for data.
        
        Args:
            data (bytes): Data to sign
            
        Returns:
            bytes: Digital signature
        """
        try:
            if hasattr(self.private_key, 'sign'):
                # RSA signature
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import padding
                
                signature = self.private_key.sign(
                    data,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                return signature
            else:
                # HMAC signature (fallback)
                import hmac
                return hmac.new(self.private_key, data, hashlib.sha256).digest()
                
        except Exception as e:
            logger.error(f"Failed to sign data: {e}")
            raise
    
    def verify_signature(self, data: bytes, signature: bytes, 
                        public_key: Optional[bytes] = None) -> bool:
        """
        Verify digital signature.
        
        Args:
            data (bytes): Original data
            signature (bytes): Signature to verify
            public_key (bytes, optional): Public key for verification
            
        Returns:
            bool: True if signature is valid
        """
        try:
            if public_key and hasattr(self, '_load_public_key'):
                # Use provided public key
                key = self._load_public_key(public_key)
            elif self.public_key:
                # Use own public key
                key = self.public_key
            else:
                # HMAC verification
                import hmac
                expected = hmac.new(self.private_key, data, hashlib.sha256).digest()
                return hmac.compare_digest(signature, expected)
            
            # RSA signature verification
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            
            try:
                key.verify(
                    signature,
                    data,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                return True
            except:
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify signature: {e}")
            return False
    
    def _load_public_key(self, public_key_data: bytes):
        """Load public key from bytes."""
        try:
            from cryptography.hazmat.primitives import serialization
            
            return serialization.load_pem_public_key(public_key_data)
        except Exception as e:
            logger.error(f"Failed to load public key: {e}")
            raise
    
    def get_public_key_bytes(self) -> bytes:
        """Get public key as bytes."""
        try:
            if self.public_key:
                from cryptography.hazmat.primitives import serialization
                
                return self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
            else:
                # Return empty bytes for HMAC mode
                return b""
                
        except Exception as e:
            logger.error(f"Failed to get public key bytes: {e}")
            return b""
    
    def add_peer_public_key(self, peer_id: str, public_key_data: bytes) -> None:
        """Add public key for a peer."""
        try:
            public_key = self._load_public_key(public_key_data)
            self.peer_public_keys[peer_id] = public_key
            logger.info(f"Added public key for peer {peer_id}")
        except Exception as e:
            logger.error(f"Failed to add peer public key: {e}")
    
    def verify_peer_signature(self, peer_id: str, data: bytes, signature: bytes) -> bool:
        """Verify signature from a specific peer."""
        if peer_id not in self.peer_public_keys:
            logger.warning(f"No public key available for peer {peer_id}")
            return False
        
        try:
            public_key = self.peer_public_keys[peer_id]
            public_key_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            return self.verify_signature(data, signature, public_key_bytes)
        except Exception as e:
            logger.error(f"Failed to verify peer signature: {e}")
            return False

class SecureCommunicationManager:
    """
    Manages secure communication protocols for federated learning.
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.encryption_manager = EncryptionManager(config)
        self.signature_manager = DigitalSignatureManager(config)
        
        # Session management
        self.active_sessions = {}
        self.session_timeout = 3600  # 1 hour
        
        # Rate limiting
        self.rate_limiter = {}
        self.max_requests_per_minute = 100
    
    def create_secure_session(self, peer_id: str) -> Dict[str, Any]:
        """
        Create secure communication session with a peer.
        
        Args:
            peer_id (str): Peer identifier
            
        Returns:
            Dict[str, Any]: Session information
        """
        try:
            # Generate session key
            session_key = secrets.token_bytes(32)
            session_id = secrets.token_urlsafe(16)
            
            # Create session metadata
            session_info = {
                'session_id': session_id,
                'peer_id': peer_id,
                'session_key': session_key,
                'created_at': time.time(),
                'last_activity': time.time(),
                'message_count': 0
            }
            
            # Store session
            self.active_sessions[session_id] = session_info
            
            # Prepare session establishment message
            session_data = {
                'session_id': session_id,
                'public_key': self.signature_manager.get_public_key_bytes().decode('utf-8'),
                'timestamp': time.time()
            }
            
            logger.info(f"Created secure session {session_id} with peer {peer_id}")
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to create secure session: {e}")
            raise
    
    def encrypt_message(self, session_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypt message for secure transmission.
        
        Args:
            session_id (str): Session identifier
            message (Dict[str, Any]): Message to encrypt
            
        Returns:
            Dict[str, Any]: Encrypted message package
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")
        
        session = self.active_sessions[session_id]
        
        try:
            # Serialize message
            message_data = json.dumps(message, default=str).encode('utf-8')
            
            # Create message signature
            signature = self.signature_manager.sign_data(message_data)
            
            # Encrypt message
            encrypted_data, metadata = self.encryption_manager.encrypt_data(
                message_data, session['session_key']
            )
            
            # Create secure package
            secure_package = {
                'session_id': session_id,
                'encrypted_data': base64.b64encode(encrypted_data).decode('utf-8'),
                'signature': base64.b64encode(signature).decode('utf-8'),
                'encryption_metadata': metadata,
                'timestamp': time.time(),
                'message_id': secrets.token_urlsafe(8)
            }
            
            # Update session activity
            session['last_activity'] = time.time()
            session['message_count'] += 1
            
            return secure_package
            
        except Exception as e:
            logger.error(f"Failed to encrypt message: {e}")
            raise
    
    def decrypt_message(self, secure_package: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt received secure message.
        
        Args:
            secure_package (Dict[str, Any]): Encrypted message package
            
        Returns:
            Dict[str, Any]: Decrypted message
        """
        session_id = secure_package['session_id']
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")
        
        session = self.active_sessions[session_id]
        
        try:
            # Decode encrypted data
            encrypted_data = base64.b64decode(secure_package['encrypted_data'])
            signature = base64.b64decode(secure_package['signature'])
            
            # Decrypt message
            decrypted_data = self.encryption_manager.decrypt_data(
                encrypted_data,
                session['session_key'],
                base64.b64decode(secure_package['encryption_metadata']['nonce']),
                EncryptionAlgorithm(secure_package['encryption_metadata']['algorithm'])
            )
            
            # Verify signature
            if not self.signature_manager.verify_signature(decrypted_data, signature):
                raise ValueError("Message signature verification failed")
            
            # Deserialize message
            message = json.loads(decrypted_data.decode('utf-8'))
            
            # Update session activity
            session['last_activity'] = time.time()
            
            return message
            
        except Exception as e:
            logger.error(f"Failed to decrypt message: {e}")
            raise
    
    def validate_session(self, session_id: str) -> bool:
        """Validate if session is active and not expired."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        current_time = time.time()
        
        # Check if session expired
        if current_time - session['last_activity'] > self.session_timeout:
            self.close_session(session_id)
            return False
        
        return True
    
    def close_session(self, session_id: str) -> None:
        """Close and cleanup session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Closed session {session_id}")
    
    def cleanup_expired_sessions(self) -> None:
        """Remove expired sessions."""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if current_time - session['last_activity'] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.close_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def check_rate_limit(self, peer_id: str) -> bool:
        """Check if peer is within rate limits."""
        current_time = time.time()
        
        if peer_id not in self.rate_limiter:
            self.rate_limiter[peer_id] = []
        
        # Remove old requests (older than 1 minute)
        self.rate_limiter[peer_id] = [
            req_time for req_time in self.rate_limiter[peer_id]
            if current_time - req_time < 60
        ]
        
        # Check if under limit
        if len(self.rate_limiter[peer_id]) >= self.max_requests_per_minute:
            return False
        
        # Add current request
        self.rate_limiter[peer_id].append(current_time)
        return True

class SecurityAuditLogger:
    """
    Logs security events and maintains audit trail.
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.audit_log_path = getattr(config, 'audit_log_path', 'security_audit.log')
        
        # Setup audit logger
        self.audit_logger = logging.getLogger('security_audit')
        self.audit_logger.setLevel(logging.INFO)
        
        # File handler for audit logs
        handler = logging.FileHandler(self.audit_log_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.audit_logger.addHandler(handler)
    
    def log_authentication_event(self, peer_id: str, event_type: str, 
                                success: bool, details: Optional[str] = None) -> None:
        """Log authentication events."""
        event_data = {
            'event_type': 'authentication',
            'sub_type': event_type,
            'peer_id': peer_id,
            'success': success,
            'timestamp': time.time(),
            'details': details or ''
        }
        
        self.audit_logger.info(f"AUTH: {json.dumps(event_data)}")
    
    def log_encryption_event(self, operation: str, algorithm: str, 
                           success: bool, details: Optional[str] = None) -> None:
        """Log encryption/decryption events."""
        event_data = {
            'event_type': 'encryption',
            'operation': operation,
            'algorithm': algorithm,
            'success': success,
            'timestamp': time.time(),
            'details': details or ''
        }
        
        self.audit_logger.info(f"CRYPT: {json.dumps(event_data)}")
    
    def log_communication_event(self, peer_id: str, message_type: str, 
                               direction: str, success: bool) -> None:
        """Log communication events."""
        event_data = {
            'event_type': 'communication',
            'peer_id': peer_id,
            'message_type': message_type,
            'direction': direction,  # 'sent' or 'received'
            'success': success,
            'timestamp': time.time()
        }
        
        self.audit_logger.info(f"COMM: {json.dumps(event_data)}")
    
    def log_security_violation(self, violation_type: str, peer_id: str, 
                             severity: str, details: str) -> None:
        """Log security violations."""
        event_data = {
            'event_type': 'security_violation',
            'violation_type': violation_type,
            'peer_id': peer_id,
            'severity': severity,
            'timestamp': time.time(),
            'details': details
        }
        
        self.audit_logger.warning(f"VIOLATION: {json.dumps(event_data)}")
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit summary for the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        summary = {
            'authentication_events': 0,
            'encryption_events': 0,
            'communication_events': 0,
            'security_violations': 0,
            'unique_peers': set(),
            'time_range': f"Last {hours} hours"
        }
        
        try:
            with open(self.audit_log_path, 'r') as f:
                for line in f:
                    try:
                        # Extract JSON from log line
                        if ' - INFO - ' in line:
                            json_part = line.split(' - INFO - ')[1]
                            if json_part.startswith(('AUTH:', 'CRYPT:', 'COMM:')):
                                event_data = json.loads(json_part.split(': ', 1)[1])
                                
                                if event_data['timestamp'] >= cutoff_time:
                                    event_type = event_data['event_type']
                                    summary[f'{event_type}_events'] += 1
                                    
                                    if 'peer_id' in event_data:
                                        summary['unique_peers'].add(event_data['peer_id'])
                        
                        elif ' - WARNING - ' in line and 'VIOLATION:' in line:
                            json_part = line.split('VIOLATION: ')[1]
                            event_data = json.loads(json_part)
                            
                            if event_data['timestamp'] >= cutoff_time:
                                summary['security_violations'] += 1
                                summary['unique_peers'].add(event_data['peer_id'])
                    
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
            
            summary['unique_peers'] = len(summary['unique_peers'])
            
        except FileNotFoundError:
            logger.warning(f"Audit log file not found: {self.audit_log_path}")
        
        return summary

# Utility functions for security operations
def generate_secure_token(length: int = 32) -> str:
    """Generate cryptographically secure token."""
    return secrets.token_urlsafe(length)

def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    """Hash password with salt."""
    if salt is None:
        salt = secrets.token_bytes(32)
    
    # Use PBKDF2 for password hashing
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    
    return base64.b64encode(key).decode('utf-8'), base64.b64encode(salt).decode('utf-8')

def verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """Verify password against hash."""
    try:
        salt_bytes = base64.b64decode(salt)
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt_bytes, 100000)
        expected_hash = base64.b64encode(key).decode('utf-8')
        
        return secrets.compare_digest(hashed_password, expected_hash)
    except Exception:
        return False

def create_secure_config(encryption_algorithm: str = "AES_GCM",
                        enable_signatures: bool = True,
                        key_size: int = 256) -> SecurityConfig:
    """Create secure configuration with recommended settings."""
    return SecurityConfig(
        encryption_algorithm=EncryptionAlgorithm(encryption_algorithm),
        enable_digital_signatures=enable_signatures,
        key_size=key_size,
        enable_secure_communication=True,
        session_timeout=3600,
        max_failed_attempts=5,
        audit_logging=True
    )

# Factory function for creating security managers
def create_security_manager(config: Optional[SecurityConfig] = None) -> Dict[str, Any]:
    """
    Create complete security management system.
    
    Args:
        config (SecurityConfig, optional): Security configuration
        
    Returns:
        Dict[str, Any]: Security managers and utilities
    """
    if config is None:
        config = create_secure_config()
    
    return {
        'encryption_manager': EncryptionManager(config),
        'signature_manager': DigitalSignatureManager(config),
        'communication_manager': SecureCommunicationManager(config),
        'audit_logger': SecurityAuditLogger(config),
        'config': config
    }

# Security validation utilities
def validate_model_integrity(model_data: bytes, expected_hash: str) -> bool:
    """Validate model data integrity using hash."""
    calculated_hash = hashlib.sha256(model_data).hexdigest()
    return secrets.compare_digest(calculated_hash, expected_hash)

def sanitize_peer_id(peer_id: str) -> str:
    """Sanitize peer ID to prevent injection attacks."""
    # Remove potentially dangerous characters
    import re
    sanitized = re.sub(r'[^\w\-_.]', '', peer_id)
    
    # Limit length
    return sanitized[:64]

def validate_message_format(message: Dict[str, Any], required_fields: List[str]) -> bool:
    """Validate message format and required fields."""
    try:
        # Check required fields
        for field in required_fields:
            if field not in message:
                return False
        
        # Check for suspicious content
        message_str = json.dumps(message)
        if len(message_str) > 1024 * 1024:  # 1MB limit
            return False
        
        return True
    except Exception:
        return False

# Performance monitoring for security operations
class SecurityPerformanceMonitor:
    """Monitor performance of security operations."""
    
    def __init__(self):
        self.metrics = {
            'encryption_times': [],
            'decryption_times': [],
            'signature_times': [],
            'verification_times': []
        }
    
    def record_operation(self, operation: str, duration: float) -> None:
        """Record operation duration."""
        if f'{operation}_times' in self.metrics:
            self.metrics[f'{operation}_times'].append(duration)
            
            # Keep only last 1000 measurements
            if len(self.metrics[f'{operation}_times']) > 1000:
                self.metrics[f'{operation}_times'].pop(0)
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        stats = {}
        
        for operation, times in self.metrics.items():
            if times:
                stats[operation] = {
                    'avg': np.mean(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'std': np.std(times),
                    'count': len(times)
                }
            else:
                stats[operation] = {
                    'avg': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0, 'count': 0
                }
        
        return stats

# Global security performance monitor
security_monitor = SecurityPerformanceMonitor()

# Context manager for timing security operations
class SecurityOperationTimer:
    """Context manager for timing security operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            security_monitor.record_operation(self.operation_name, duration)

# Export main classes and functions
__all__ = [
    'SecurityConfig',
    'EncryptionAlgorithm',
    'EncryptionManager',
    'DigitalSignatureManager',
    'SecureCommunicationManager',
    'SecurityAuditLogger',
    'SecurityPerformanceMonitor',
    'create_security_manager',
    'create_secure_config',
    'generate_secure_token',
    'hash_password',
    'verify_password',
    'validate_model_integrity',
    'sanitize_peer_id',
    'validate_message_format',
    'SecurityOperationTimer',
    'security_monitor'
]

# Module initialization
logger.info("BatteryMind Federated Learning Security Utils v1.0.0 initialized")

# Security self-test
def run_security_self_test() -> bool:
    """Run basic security functionality self-test."""
    try:
        # Test encryption
        config = create_secure_config()
        encryption_manager = EncryptionManager(config)
        
        test_data = b"Hello, secure world!"
        key = secrets.token_bytes(32)
        
        encrypted, metadata = encryption_manager.encrypt_data(test_data, key)
        decrypted = encryption_manager.decrypt_data(
            encrypted, key, 
            base64.b64decode(metadata['nonce']),
            EncryptionAlgorithm(metadata['algorithm'])
        )
        
        if decrypted != test_data:
            return False
        
        # Test signatures
        signature_manager = DigitalSignatureManager(config)
        signature = signature_manager.sign_data(test_data)
        verified = signature_manager.verify_signature(test_data, signature)
        
        if not verified:
            return False
        
        logger.info("Security self-test passed")
        return True
        
    except Exception as e:
        logger.error(f"Security self-test failed: {e}")
        return False

# Run self-test on import
if __name__ != "__main__":
    run_security_self_test()

