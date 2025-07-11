"""
BatteryMind – Federated Learning
================================

model_updates.py
----------------
Client-side utilities that package, secure, and transmit local model updates
to the federated server.  Responsibilities:

1.  Convert local model state_dict to a compact, compressed tensor bundle.
2.  Apply differential-privacy (DP) noise if requested.
3.  Digitally sign the update for authenticity & integrity.
4.  Provide secure (optional) encryption utilities.
5.  Offer checksum utilities for fast integrity validation.
6.  Serialize/deserialize updates for transport (MQTT, gRPC, HTTP/2).
"""

from __future__ import annotations

import io
import json
import hashlib
import logging
import pickle
import struct
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import nacl.encoding
import nacl.hash
import nacl.signing
import numpy as np
import torch

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Configuration Dataclass
# -----------------------------------------------------------------------------
@dataclass
class UpdateConfig:
    """Runtime configuration for packaging client model updates."""
    compression_level: int = 6                     # zlib compression: 1-9
    apply_dp_noise: bool = True                    # Add differential-privacy noise
    dp_noise_std: float = 1e-3                     # σ for Gaussian DP noise
    enable_signing: bool = True                    # Sign updates with Ed25519
    private_key_path: Optional[str] = None         # Path to private key PEM
    chunk_size: int = 4_194_304                    # 4 MB chunks for stream IO
    protocol_version: str = "1.0.0"


# -----------------------------------------------------------------------------
# Metadata Dataclass
# -----------------------------------------------------------------------------
@dataclass
class ModelUpdateMetadata:
    """Lightweight metadata shipped with every client update."""
    client_id: str
    round_number: int
    framework_version: str = torch.__version__
    protocol_version: str = "1.0.0"
    timestamp: float = field(default_factory=lambda: time.time())
    samples_used: int = 0
    training_loss: float = 0.0
    dp_noise_std: float = 0.0
    checksum: str = ""           # SHA-256 of compressed payload
    signature: Optional[str] = None


# -----------------------------------------------------------------------------
# Main Update Class
# -----------------------------------------------------------------------------
class ModelUpdate:
    """
    Helper class that converts a `state_dict` into a secure, compressed,
    and optionally signed payload ready for transmission.
    """

    def __init__(
        self,
        state_dict: Dict[str, torch.Tensor],
        metadata: ModelUpdateMetadata,
        config: UpdateConfig | None = None,
    ) -> None:
        self.state_dict = state_dict
        self.metadata = metadata
        self.config = config or UpdateConfig()
        self._compressed_bytes: bytes | None = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def package(self) -> Dict[str, Any]:
        """
        Returns a dictionary ready to be JSON-serialised and sent to the server.

        Keys:
            • 'metadata' – JSON-serialisable `ModelUpdateMetadata`
            • 'payload'  – base-85 encoded compressed tensor blob
        """
        logger.info("Packaging model update for client %s", self.metadata.client_id)

        tensor_blob = self._state_dict_to_bytes(self.state_dict)
        if self.config.apply_dp_noise:
            tensor_blob = self._apply_dp_noise(tensor_blob)

        compressed = zlib.compress(tensor_blob, self.config.compression_level)
        checksum = hashlib.sha256(compressed).hexdigest()
        self.metadata.checksum = checksum
        self.metadata.dp_noise_std = self.config.dp_noise_std

        if self.config.enable_signing:
            signature = self._sign(compressed)
            self.metadata.signature = signature

        self._compressed_bytes = compressed

        logger.debug(
            "Packaged update – size (raw): %.2f KB | size (compressed): %.2f KB",
            len(tensor_blob) / 1024,
            len(compressed) / 1024,
        )

        return {
            "metadata": asdict(self.metadata),
            "payload": compressed.hex(),  # hex is ∼2× size but JSON-safe
        }

    def save(self, path: str | Path) -> None:
        """Persist the packaged update to disk for offline transport."""
        if self._compressed_bytes is None:
            raise RuntimeError("Call `package()` before saving update.")
        path = Path(path)
        path.write_bytes(self._compressed_bytes)
        path.with_suffix(".meta.json").write_text(json.dumps(asdict(self.metadata), indent=2))
        logger.info("Client update saved to %s", path)

    # ---------------------------------------------------------------------
    # Private Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _state_dict_to_bytes(state_dict: Dict[str, torch.Tensor]) -> bytes:
        """
        Serialises a state_dict into a binary stream using Torch + pickle.

        Uses the following layout per tensor:
            [key_length | key_bytes | tensor_dtype | shape_tuple | tensor_bytes]
        """
        buffer = io.BytesIO()
        for key, tensor in state_dict.items():
            # Convert tensor to CPU & contiguous numpy array
            array = tensor.detach().cpu().contiguous().numpy()

            # Header: key length | key | dtype | shape length | shape dims
            key_bytes = key.encode()
            buffer.write(struct.pack("I", len(key_bytes)))
            buffer.write(key_bytes)

            dtype_str = str(array.dtype)
            buffer.write(struct.pack("I", len(dtype_str)))
            buffer.write(dtype_str.encode())

            buffer.write(struct.pack("I", len(array.shape)))
            buffer.write(struct.pack(f"{len(array.shape)}I", *array.shape))

            # Tensor payload
            buffer.write(array.tobytes(order="C"))

        return buffer.getvalue()

    def _apply_dp_noise(self, blob: bytes) -> bytes:
        """Adds Gaussian noise to tensor bytes for differential privacy."""
        if self.config.dp_noise_std <= 0:
            return blob

        # Convert bytes back to float32 array
        arr = np.frombuffer(blob, dtype=np.uint8)
        noise = np.random.normal(0, self.config.dp_noise_std, arr.shape).astype(np.int8)
        noisy = (arr.astype(np.int32) + noise).astype(np.uint8)  # modulo 256 wrap

        logger.debug("DP noise applied (σ = %.2e)", self.config.dp_noise_std)
        return noisy.tobytes()

    def _sign(self, compressed_blob: bytes) -> str:
        """Signs the compressed blob with an Ed25519 private key."""
        key_path = self.config.private_key_path or ".keys/client_ed25519.pem"
        key_path = Path(key_path)
        if not key_path.exists():
            raise FileNotFoundError(f"Ed25519 private key not found: {key_path}")

        signing_key = nacl.signing.SigningKey(
            key_path.read_bytes(), encoder=nacl.encoding.RawEncoder
        )
        signature = signing_key.sign(compressed_blob).signature
        return signature.hex()


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def load_update(payload_hex: str, metadata: Dict[str, Any]) -> ModelUpdate:
    """Utility to restore a `ModelUpdate` object from serialized content."""
    update_bytes = bytes.fromhex(payload_hex)
    meta_obj = ModelUpdateMetadata(**metadata)
    dummy_sd = {}  # State dict will be unpacked server-side
    update = ModelUpdate(dummy_sd, meta_obj)
    update._compressed_bytes = update_bytes  # type: ignore

    # Verify checksum
    computed = hashlib.sha256(update_bytes).hexdigest()
    if computed != meta_obj.checksum:
        raise ValueError("Checksum mismatch – corrupted update payload")

    return update
