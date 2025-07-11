"""
BatteryMind – Federated Learning Server Package
==============================================

This package orchestrates the server-side components of the federated-learning
workflow, including:

* `federated_server.py`     – Entry-point gRPC/HTTP server.
* `aggregation_algorithms.py` – FedAvg, FedProx, FedOPT, etc.
* `model_aggregator.py`     – Model-versioning & weight-update logic.
* `global_model.py`         – Global model lifecycle management.

The package initialises logging, environment variables, and exposes
factory helpers for seamless integration with the rest of the platform.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Dict, Type

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("batterymind.federated_server")

# -----------------------------------------------------------------------------
# Dynamic Module Loading
# -----------------------------------------------------------------------------
_REQUIRED_MODULES = [
    "ai-models.federated-learning.server.federated_server",
    "ai-models.federated-learning.server.aggregation_algorithms",
    "ai-models.federated-learning.server.model_aggregator",
    "ai-models.federated-learning.server.global_model",
]

_MISSING: list[str] = []
for mod in _REQUIRED_MODULES:
    try:
        importlib.import_module(mod)
    except ImportError as exc:
        logger.warning("Deferred import – module %s not yet available: %s", mod, exc)
        _MISSING.append(mod)

if _MISSING:
    logger.info(
        "Federated-Server initialisation complete with %d module(s) pending implementation",
        len(_MISSING),
    )

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
__all__ = [
    "get_server_config",
    "ensure_runtime_dirs",
]


def get_server_config() -> Dict[str, str]:
    """Return environment-specific server configuration."""
    return {
        "AGGREGATION_ALGO": os.getenv("BMIND_AGG_ALGO", "FedAvg"),
        "MODEL_REG_DIR": os.getenv("BMIND_MODEL_REG_DIR", "./model_registry"),
        "CHECKPOINT_DIR": os.getenv("BMIND_CHECKPOINT_DIR", "./model-artifacts/checkpoints"),
        "LOG_LEVEL": logging.getLevelName(logger.getEffectiveLevel()),
    }


def ensure_runtime_dirs() -> None:
    """Create runtime directories if they do not yet exist."""
    cfg = get_server_config()
    Path(cfg["MODEL_REG_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["CHECKPOINT_DIR"]).mkdir(parents=True, exist_ok=True)
    logger.debug("Runtime directories verified/created.")
