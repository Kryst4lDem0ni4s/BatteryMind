"""
BatteryMind - Federated Learning Serialization Utilities

Comprehensive serialization and deserialization utilities for federated learning
components including model parameters, gradients, and metadata with support for
compression, encryption, and efficient network transmission.

Features:
- Secure model parameter serialization
- Gradient compression and quantization
- Metadata serialization with versioning
- Protocol buffer support for efficient transmission
- Compression algorithms for bandwidth optimization
- Encryption-aware serialization
- Cross-platform compatibility

Author: BatteryMind Development Team
Version: 1.0.0
"""

import pickle
import json
import gzip
import lz4.frame
import zstandard as zstd
import numpy as np
import torch
import io
import struct
import hashlib
import base64
from typing import Dict, List, Optional, Tuple, Union, Any, BinaryIO
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import time
from pathlib import Path

# Protocol buffers (if available)
try:
    import google.protobuf.message as pb_message
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False
    pb_message = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompressionType(Enum):
    """Supported compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"

class SerializationFormat(Enum):
    """Supported serialization formats."""
    PICKLE = "pickle"
    JSON = "json"
    PROTOBUF = "protobuf"
    CUSTOM_BINARY = "custom_binary"

@dataclass
class SerializationMetadata:
    """
    Metadata for serialized objects.
    
    Attributes:
        format (SerializationFormat): Serialization format used
        compression (CompressionType): Compression algorithm used
        version (str): Serialization version
        timestamp (float): Serialization timestamp
        checksum (str): Data integrity checksum
        original_size (int): Original data size in bytes
        compressed_size (int): Compressed data size in bytes
        encryption_info (Dict): Encryption metadata if applicable
    """
    format: SerializationFormat
    compression: CompressionType
    version: str = "1.0.0"
    timestamp: float = 0.0
    checksum: str = ""
    original_size: int = 0
    compressed_size: int = 0
    encryption_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.encryption_info is None:
            self.encryption_info = {}

class ModelParameterSerializer:
    """
    Specialized serializer for neural network model parameters.
    """
    
    def __init__(self, compression: CompressionType = CompressionType.ZSTD,
                 quantization_bits: Optional[int] = None):
        self.compression = compression
        self.quantization_bits = quantization_bits
        
    def serialize_parameters(self, parameters: Dict[str, torch.Tensor]) -> Tuple[bytes, SerializationMetadata]:
        """
        Serialize model parameters with optional quantization and compression.
        
        Args:
            parameters (Dict[str, torch.Tensor]): Model parameters to serialize
            
        Returns:
            Tuple[bytes, SerializationMetadata]: Serialized data and metadata
        """
        # Convert tensors to numpy arrays for serialization
        param_dict = {}
        for name, tensor in parameters.items():
            if isinstance(tensor, torch.Tensor):
                param_dict[name] = tensor.detach().cpu().numpy()
            else:
                param_dict[name] = tensor
        
        # Apply quantization if specified
        if self.quantization_bits is not None:
            param_dict = self._quantize_parameters(param_dict, self.quantization_bits)
        
        # Serialize using pickle
        serialized_data = pickle.dumps(param_dict, protocol=pickle.HIGHEST_PROTOCOL)
        original_size = len(serialized_data)
        
        # Apply compression
        compressed_data = self._compress_data(serialized_data, self.compression)
        compressed_size = len(compressed_data)
        
        # Calculate checksum
        checksum = hashlib.sha256(compressed_data).hexdigest()
        
        # Create metadata
        metadata = SerializationMetadata(
            format=SerializationFormat.PICKLE,
            compression=self.compression,
            checksum=checksum,
            original_size=original_size,
            compressed_size=compressed_size
        )
        
        return compressed_data, metadata
    
    def deserialize_parameters(self, data: bytes, 
                             metadata: SerializationMetadata) -> Dict[str, torch.Tensor]:
        """
        Deserialize model parameters with decompression and validation.
        
        Args:
            data (bytes): Serialized data
            metadata (SerializationMetadata): Serialization metadata
            
        Returns:
            Dict[str, torch.Tensor]: Deserialized model parameters
        """
        # Verify checksum
        if metadata.checksum:
            calculated_checksum = hashlib.sha256(data).hexdigest()
            if calculated_checksum != metadata.checksum:
                raise ValueError("Data integrity check failed: checksum mismatch")
        
        # Decompress data
        decompressed_data = self._decompress_data(data, metadata.compression)
        
        # Deserialize
        param_dict = pickle.loads(decompressed_data)
        
        # Dequantize if needed
        if self.quantization_bits is not None:
            param_dict = self._dequantize_parameters(param_dict, self.quantization_bits)
        
        # Convert back to tensors
        parameters = {}
        for name, array in param_dict.items():
            if isinstance(array, np.ndarray):
                parameters[name] = torch.from_numpy(array)
            else:
                parameters[name] = array
        
        return parameters
    
    def _quantize_parameters(self, parameters: Dict[str, np.ndarray], 
                           bits: int) -> Dict[str, Any]:
        """Apply quantization to reduce parameter precision."""
        quantized_params = {}
        
        for name, param in parameters.items():
            if isinstance(param, np.ndarray) and param.dtype in [np.float32, np.float64]:
                # Calculate quantization scale
                param_min = param.min()
                param_max = param.max()
                scale = (param_max - param_min) / (2**bits - 1)
                
                # Quantize
                quantized = np.round((param - param_min) / scale).astype(np.uint8 if bits <= 8 else np.uint16)
                
                quantized_params[name] = {
                    'data': quantized,
                    'scale': scale,
                    'min_val': param_min,
                    'original_shape': param.shape,
                    'original_dtype': param.dtype,
                    'quantized': True
                }
            else:
                quantized_params[name] = param
        
        return quantized_params
    
    def _dequantize_parameters(self, parameters: Dict[str, Any], 
                             bits: int) -> Dict[str, np.ndarray]:
        """Restore quantized parameters to original precision."""
        dequantized_params = {}
        
        for name, param in parameters.items():
            if isinstance(param, dict) and param.get('quantized', False):
                # Dequantize
                quantized_data = param['data']
                scale = param['scale']
                min_val = param['min_val']
                original_dtype = param['original_dtype']
                
                dequantized = (quantized_data.astype(original_dtype) * scale + min_val)
                dequantized_params[name] = dequantized
            else:
                dequantized_params[name] = param
        
        return dequantized_params
    
    def _compress_data(self, data: bytes, compression: CompressionType) -> bytes:
        """Apply compression to serialized data."""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.compress(data)
        elif compression == CompressionType.LZ4:
            return lz4.frame.compress(data)
        elif compression == CompressionType.ZSTD:
            cctx = zstd.ZstdCompressor(level=3)
            return cctx.compress(data)
        else:
            raise ValueError(f"Unsupported compression type: {compression}")
    
    def _decompress_data(self, data: bytes, compression: CompressionType) -> bytes:
        """Decompress serialized data."""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.decompress(data)
        elif compression == CompressionType.LZ4:
            return lz4.frame.decompress(data)
        elif compression == CompressionType.ZSTD:
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        else:
            raise ValueError(f"Unsupported compression type: {compression}")

class GradientSerializer:
    """
    Specialized serializer for gradient data with compression and sparsification.
    """
    
    def __init__(self, compression: CompressionType = CompressionType.ZSTD,
                 sparsification_threshold: float = 1e-6):
        self.compression = compression
        self.sparsification_threshold = sparsification_threshold
        
    def serialize_gradients(self, gradients: Dict[str, torch.Tensor],
                          apply_sparsification: bool = True) -> Tuple[bytes, SerializationMetadata]:
        """
        Serialize gradients with optional sparsification and compression.
        
        Args:
            gradients (Dict[str, torch.Tensor]): Gradients to serialize
            apply_sparsification (bool): Whether to apply gradient sparsification
            
        Returns:
            Tuple[bytes, SerializationMetadata]: Serialized data and metadata
        """
        # Convert to numpy and apply sparsification
        grad_dict = {}
        for name, grad in gradients.items():
            if isinstance(grad, torch.Tensor):
                grad_np = grad.detach().cpu().numpy()
                
                if apply_sparsification:
                    grad_dict[name] = self._sparsify_gradient(grad_np)
                else:
                    grad_dict[name] = grad_np
            else:
                grad_dict[name] = grad
        
        # Serialize
        serialized_data = pickle.dumps(grad_dict, protocol=pickle.HIGHEST_PROTOCOL)
        original_size = len(serialized_data)
        
        # Compress
        compressed_data = self._compress_data(serialized_data, self.compression)
        compressed_size = len(compressed_data)
        
        # Calculate checksum
        checksum = hashlib.sha256(compressed_data).hexdigest()
        
        metadata = SerializationMetadata(
            format=SerializationFormat.PICKLE,
            compression=self.compression,
            checksum=checksum,
            original_size=original_size,
            compressed_size=compressed_size
        )
        
        return compressed_data, metadata
    
    def deserialize_gradients(self, data: bytes,
                            metadata: SerializationMetadata) -> Dict[str, torch.Tensor]:
        """
        Deserialize gradients with decompression and desparsification.
        
        Args:
            data (bytes): Serialized gradient data
            metadata (SerializationMetadata): Serialization metadata
            
        Returns:
            Dict[str, torch.Tensor]: Deserialized gradients
        """
        # Verify checksum
        if metadata.checksum:
            calculated_checksum = hashlib.sha256(data).hexdigest()
            if calculated_checksum != metadata.checksum:
                raise ValueError("Gradient data integrity check failed")
        
        # Decompress
        decompressed_data = self._decompress_data(data, metadata.compression)
        
        # Deserialize
        grad_dict = pickle.loads(decompressed_data)
        
        # Convert back to tensors and desparsify
        gradients = {}
        for name, grad_data in grad_dict.items():
            if isinstance(grad_data, dict) and 'sparse' in grad_data:
                # Desparsify
                grad_array = self._desparsify_gradient(grad_data)
                gradients[name] = torch.from_numpy(grad_array)
            elif isinstance(grad_data, np.ndarray):
                gradients[name] = torch.from_numpy(grad_data)
            else:
                gradients[name] = grad_data
        
        return gradients
    
    def _sparsify_gradient(self, gradient: np.ndarray) -> Dict[str, Any]:
        """Apply sparsification to gradient array."""
        # Find significant values
        mask = np.abs(gradient) > self.sparsification_threshold
        
        if np.sum(mask) < gradient.size * 0.1:  # If less than 10% are significant
            # Store as sparse
            indices = np.where(mask)
            values = gradient[mask]
            
            return {
                'sparse': True,
                'indices': indices,
                'values': values,
                'shape': gradient.shape,
                'dtype': gradient.dtype,
                'sparsity_ratio': 1.0 - (len(values) / gradient.size)
            }
        else:
            # Store as dense
            return gradient
    
    def _desparsify_gradient(self, sparse_data: Dict[str, Any]) -> np.ndarray:
        """Restore sparse gradient to dense format."""
        gradient = np.zeros(sparse_data['shape'], dtype=sparse_data['dtype'])
        gradient[sparse_data['indices']] = sparse_data['values']
        return gradient
    
    def _compress_data(self, data: bytes, compression: CompressionType) -> bytes:
        """Apply compression to serialized data."""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.compress(data)
        elif compression == CompressionType.LZ4:
            return lz4.frame.compress(data)
        elif compression == CompressionType.ZSTD:
            cctx = zstd.ZstdCompressor(level=3)
            return cctx.compress(data)
        else:
            raise ValueError(f"Unsupported compression type: {compression}")
    
    def _decompress_data(self, data: bytes, compression: CompressionType) -> bytes:
        """Decompress serialized data."""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.decompress(data)
        elif compression == CompressionType.LZ4:
            return lz4.frame.decompress(data)
        elif compression == CompressionType.ZSTD:
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        else:
            raise ValueError(f"Unsupported compression type: {compression}")

class MetadataSerializer:
    """
    Serializer for federated learning metadata and configuration data.
    """
    
    def __init__(self, format: SerializationFormat = SerializationFormat.JSON):
        self.format = format
        
    def serialize_metadata(self, metadata: Dict[str, Any]) -> Tuple[bytes, SerializationMetadata]:
        """
        Serialize metadata dictionary.
        
        Args:
            metadata (Dict[str, Any]): Metadata to serialize
            
        Returns:
            Tuple[bytes, SerializationMetadata]: Serialized data and metadata
        """
        if self.format == SerializationFormat.JSON:
            # Convert numpy types to native Python types
            serializable_metadata = self._make_json_serializable(metadata)
            serialized_data = json.dumps(serializable_metadata, indent=2).encode('utf-8')
        elif self.format == SerializationFormat.PICKLE:
            serialized_data = pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"Unsupported metadata format: {self.format}")
        
        # Calculate checksum
        checksum = hashlib.sha256(serialized_data).hexdigest()
        
        meta = SerializationMetadata(
            format=self.format,
            compression=CompressionType.NONE,
            checksum=checksum,
            original_size=len(serialized_data),
            compressed_size=len(serialized_data)
        )
        
        return serialized_data, meta
    
    def deserialize_metadata(self, data: bytes, 
                           metadata: SerializationMetadata) -> Dict[str, Any]:
        """
        Deserialize metadata.
        
        Args:
            data (bytes): Serialized metadata
            metadata (SerializationMetadata): Serialization metadata
            
        Returns:
            Dict[str, Any]: Deserialized metadata
        """
        # Verify checksum
        if metadata.checksum:
            calculated_checksum = hashlib.sha256(data).hexdigest()
            if calculated_checksum != metadata.checksum:
                raise ValueError("Metadata integrity check failed")
        
        if metadata.format == SerializationFormat.JSON:
            return json.loads(data.decode('utf-8'))
        elif metadata.format == SerializationFormat.PICKLE:
            return pickle.loads(data)
        else:
            raise ValueError(f"Unsupported metadata format: {metadata.format}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif obj is None or isinstance(obj, (str, int, float)):
            return obj
        else:
            # For other types, convert to string representation
            return str(obj)

class SecureSerializer:
    """
    Serializer with built-in encryption support for sensitive data.
    """
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key
        self.param_serializer = ModelParameterSerializer()
        self.grad_serializer = GradientSerializer()
        self.meta_serializer = MetadataSerializer()
        
    def serialize_secure(self, data: Any, data_type: str = "parameters") -> Tuple[bytes, SerializationMetadata]:
        """
        Serialize data with encryption.
        
        Args:
            data: Data to serialize
            data_type (str): Type of data ("parameters", "gradients", "metadata")
            
        Returns:
            Tuple[bytes, SerializationMetadata]: Encrypted serialized data and metadata
        """
        # Choose appropriate serializer
        if data_type == "parameters":
            serialized_data, metadata = self.param_serializer.serialize_parameters(data)
        elif data_type == "gradients":
            serialized_data, metadata = self.grad_serializer.serialize_gradients(data)
        elif data_type == "metadata":
            serialized_data, metadata = self.meta_serializer.serialize_metadata(data)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Encrypt if key is provided
        if self.encryption_key:
            encrypted_data = self._encrypt_data(serialized_data)
            metadata.encryption_info = {
                'encrypted': True,
                'algorithm': 'AES-256-GCM',
                'key_hash': hashlib.sha256(self.encryption_key).hexdigest()[:16]
            }
            return encrypted_data, metadata
        else:
            return serialized_data, metadata
    
    def deserialize_secure(self, data: bytes, metadata: SerializationMetadata,
                         data_type: str = "parameters") -> Any:
        """
        Deserialize encrypted data.
        
        Args:
            data (bytes): Encrypted serialized data
            metadata (SerializationMetadata): Serialization metadata
            data_type (str): Type of data
            
        Returns:
            Any: Deserialized data
        """
        # Decrypt if needed
        if metadata.encryption_info.get('encrypted', False):
            if not self.encryption_key:
                raise ValueError("Encryption key required for decryption")
            
            decrypted_data = self._decrypt_data(data)
        else:
            decrypted_data = data
        
        # Deserialize
        if data_type == "parameters":
            return self.param_serializer.deserialize_parameters(decrypted_data, metadata)
        elif data_type == "gradients":
            return self.grad_serializer.deserialize_gradients(decrypted_data, metadata)
        elif data_type == "metadata":
            return self.meta_serializer.deserialize_metadata(decrypted_data, metadata)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using AES-256-GCM."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            # Generate random nonce
            nonce = os.urandom(12)
            
            # Create cipher
            aesgcm = AESGCM(self.encryption_key)
            
            # Encrypt
            ciphertext = aesgcm.encrypt(nonce, data, None)
            
            # Prepend nonce to ciphertext
            return nonce + ciphertext
            
        except ImportError:
            logger.warning("Cryptography library not available, using base64 encoding")
            return base64.b64encode(data)
    
    def _decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using AES-256-GCM."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            # Extract nonce and ciphertext
            nonce = encrypted_data[:12]
            ciphertext = encrypted_data[12:]
            
            # Create cipher
            aesgcm = AESGCM(self.encryption_key)
            
            # Decrypt
            return aesgcm.decrypt(nonce, ciphertext, None)
            
        except ImportError:
            logger.warning("Cryptography library not available, using base64 decoding")
            return base64.b64decode(encrypted_data)

class FederatedMessageSerializer:
    """
    High-level serializer for complete federated learning messages.
    """
    
    def __init__(self, compression: CompressionType = CompressionType.ZSTD,
                 encryption_key: Optional[bytes] = None):
        self.compression = compression
        self.encryption_key = encryption_key
        self.secure_serializer = SecureSerializer(encryption_key)
        
    def serialize_message(self, message: Dict[str, Any]) -> bytes:
        """
        Serialize a complete federated learning message.
        
        Args:
            message (Dict[str, Any]): Message containing parameters, gradients, metadata
            
        Returns:
            bytes: Serialized message
        """
        serialized_parts = {}
        metadata_parts = {}
        
        # Serialize each part of the message
        for key, value in message.items():
            if key in ['parameters', 'model_parameters']:
                data, meta = self.secure_serializer.serialize_secure(value, "parameters")
                serialized_parts[key] = data
                metadata_parts[key] = asdict(meta)
            elif key in ['gradients', 'model_gradients']:
                data, meta = self.secure_serializer.serialize_secure(value, "gradients")
                serialized_parts[key] = data
                metadata_parts[key] = asdict(meta)
            else:
                # Treat as metadata
                data, meta = self.secure_serializer.serialize_secure(value, "metadata")
                serialized_parts[key] = data
                metadata_parts[key] = asdict(meta)
        
        # Create complete message
        complete_message = {
            'data': serialized_parts,
            'metadata': metadata_parts,
            'message_id': hashlib.sha256(str(time.time()).encode()).hexdigest()[:16],
            'timestamp': time.time(),
            'version': '1.0.0'
        }
        
        # Serialize the complete message
        message_bytes = pickle.dumps(complete_message, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Apply compression
        if self.compression != CompressionType.NONE:
            if self.compression == CompressionType.GZIP:
                message_bytes = gzip.compress(message_bytes)
            elif self.compression == CompressionType.LZ4:
                message_bytes = lz4.frame.compress(message_bytes)
            elif self.compression == CompressionType.ZSTD:
                cctx = zstd.ZstdCompressor(level=3)
                message_bytes = cctx.compress(message_bytes)
        
        return message_bytes
    
    def deserialize_message(self, data: bytes) -> Dict[str, Any]:
        """
        Deserialize a complete federated learning message.
        
        Args:
            data (bytes): Serialized message
            
        Returns:
            Dict[str, Any]: Deserialized message
        """
        # Decompress if needed
        if self.compression != CompressionType.NONE:
            if self.compression == CompressionType.GZIP:
                data = gzip.decompress(data)
            elif self.compression == CompressionType.LZ4:
                data = lz4.frame.decompress(data)
            elif self.compression == CompressionType.ZSTD:
                dctx = zstd.ZstdDecompressor()
                data = dctx.decompress(data)
        
        # Deserialize message structure
        complete_message = pickle.loads(data)
        
        # Deserialize each part
        deserialized_message = {}
        
        for key, serialized_data in complete_message['data'].items():
            metadata_dict = complete_message['metadata'][key]
            metadata = SerializationMetadata(**metadata_dict)
            
            if key in ['parameters', 'model_parameters']:
                deserialized_message[key] = self.secure_serializer.deserialize_secure(
                    serialized_data, metadata, "parameters"
                )
            elif key in ['gradients', 'model_gradients']:
                deserialized_message[key] = self.secure_serializer.deserialize_secure(
                    serialized_data, metadata, "gradients"
                )
            else:
                deserialized_message[key] = self.secure_serializer.deserialize_secure(
                    serialized_data, metadata, "metadata"
                )
        
        # Add message metadata
        deserialized_message['_message_metadata'] = {
            'message_id': complete_message['message_id'],
            'timestamp': complete_message['timestamp'],
            'version': complete_message['version']
        }
        
        return deserialized_message

# Utility functions
def estimate_serialization_size(obj: Any, compression: CompressionType = CompressionType.NONE) -> Dict[str, int]:
    """
    Estimate the serialization size of an object.
    
    Args:
        obj: Object to estimate size for
        compression: Compression type to use
        
    Returns:
        Dict[str, int]: Size estimates
    """
    # Serialize without compression
    serialized = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    original_size = len(serialized)
    
    # Apply compression if specified
    if compression == CompressionType.GZIP:
        compressed = gzip.compress(serialized)
    elif compression == CompressionType.LZ4:
        compressed = lz4.frame.compress(serialized)
    elif compression == CompressionType.ZSTD:
        cctx = zstd.ZstdCompressor(level=3)
        compressed = cctx.compress(serialized)
    else:
        compressed = serialized
    
    compressed_size = len(compressed)
    
    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': original_size / compressed_size if compressed_size > 0 else 1.0
    }

def benchmark_compression_methods(data: bytes) -> Dict[str, Dict[str, Union[int, float]]]:
    """
    Benchmark different compression methods on given data.
    
    Args:
        data (bytes): Data to benchmark
        
    Returns:
        Dict[str, Dict[str, Union[int, float]]]: Benchmark results
    """
    results = {}
    original_size = len(data)
    
    # Test each compression method
    for compression in CompressionType:
        if compression == CompressionType.NONE:
            compressed_size = original_size
            compression_time = 0.0
            decompression_time = 0.0
        else:
            # Measure compression time
            start_time = time.time()
            if compression == CompressionType.GZIP:
                compressed = gzip.compress(data)
            elif compression == CompressionType.LZ4:
                compressed = lz4.frame.compress(data)
            elif compression == CompressionType.ZSTD:
                cctx = zstd.ZstdCompressor(level=3)
                compressed = cctx.compress(data)
            compression_time = time.time() - start_time
            
            compressed_size = len(compressed)
            
            # Measure decompression time
            start_time = time.time()
            if compression == CompressionType.GZIP:
                gzip.decompress(compressed)
            elif compression == CompressionType.LZ4:
                lz4.frame.decompress(compressed)
            elif compression == CompressionType.ZSTD:
                dctx = zstd.ZstdDecompressor()
                dctx.decompress(compressed)
            decompression_time = time.time() - start_time
        
        results[compression.value] = {
            'compressed_size': compressed_size,
            'compression_ratio': original_size / compressed_size if compressed_size > 0 else 1.0,
            'compression_time': compression_time,
            'decompression_time': decompression_time,
            'total_time': compression_time + decompression_time
        }
    
    return results

# Factory functions
def create_parameter_serializer(compression: str = "zstd", 
                               quantization_bits: Optional[int] = None) -> ModelParameterSerializer:
    """Create a model parameter serializer with specified options."""
    compression_type = CompressionType(compression.lower())
    return ModelParameterSerializer(compression_type, quantization_bits)

def create_gradient_serializer(compression: str = "zstd",
                             sparsification_threshold: float = 1e-6) -> GradientSerializer:
    """Create a gradient serializer with specified options."""
    compression_type = CompressionType(compression.lower())
    return GradientSerializer(compression_type, sparsification_threshold)

def create_secure_serializer(encryption_key: Optional[bytes] = None) -> SecureSerializer:
    """Create a secure serializer with encryption support."""
    return SecureSerializer(encryption_key)

def create_message_serializer(compression: str = "zstd",
                            encryption_key: Optional[bytes] = None) -> FederatedMessageSerializer:
    """Create a complete message serializer."""
    compression_type = CompressionType(compression.lower())
    return FederatedMessageSerializer(compression_type, encryption_key)
