"""
BatteryMind – Federated Aggregation Algorithms
==============================================

Collection of reusable, production–grade aggregation strategies for
privacy-preserving federated learning. The module is framework-agnostic
(Pure-PyTorch tensors) and can be extended with custom algorithms.

Implemented algorithms
----------------------
• FedAvg              – Classical weighted average of client weights  
• FedProx             – FedAvg + proximal term to tackle heterogeneous data  
• FedYogi             – Adaptive momentum‐based global optimisation  
• SecAgg (interface)  – Secure aggregation hook (stub for cryptographic add-on)

Author: BatteryMind Dev Team
License: Proprietary – Tata Technologies InnoVent 2025
Version: 1.0.0
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import logging

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Generic helper utilities                                                   #
# --------------------------------------------------------------------------- #
def _weighted_average(
    client_weights: List[Dict[str, torch.Tensor]],
    client_sizes: List[int]
) -> Dict[str, torch.Tensor]:
    """Compute size-weighted average of client model parameters."""
    total_samples = sum(client_sizes)
    averaged_weights: Dict[str, torch.Tensor] = {}

    # Iterate over every parameter key
    for key in client_weights[0]:
        # Stack parameter tensors from all clients
        stacked = torch.stack([cw[key] * (cs / total_samples)
                               for cw, cs in zip(client_weights, client_sizes)], dim=0)
        averaged_weights[key] = torch.sum(stacked, dim=0)
    return averaged_weights


# --------------------------------------------------------------------------- #
#  Aggregation algorithm implementations                                      #
# --------------------------------------------------------------------------- #
@dataclass
class AggregationResult:
    """Container for aggregation output and metadata."""
    global_weights: Dict[str, torch.Tensor]
    aux_stats: Dict[str, float]


class FedAvg:
    """Standard Federated Averaging algorithm."""

    @staticmethod
    def aggregate(
        client_weights: List[Dict[str, torch.Tensor]],
        client_sizes: List[int]
    ) -> AggregationResult:
        logger.debug("FedAvg aggregation started")
        global_weights = _weighted_average(client_weights, client_sizes)
        logger.debug("FedAvg aggregation completed")
        return AggregationResult(global_weights=global_weights,
                                 aux_stats={"algorithm": "FedAvg"})


class FedProx:
    """FedProx – Adds a proximal term to combat data heterogeneity."""

    def __init__(self, mu: float = 0.01):
        """
        Args
        ----
        mu : float  
            Proximal term coefficient (λ in original paper).
        """
        self.mu = mu

    def aggregate(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
        global_prev: Dict[str, torch.Tensor]
    ) -> AggregationResult:
        logger.debug("FedProx aggregation started")
        # First perform FedAvg
        global_weights = _weighted_average(client_weights, client_sizes)

        # Apply proximal update
        for k in global_weights:
            global_weights[k] = global_weights[k] - self.mu * (global_prev[k] - global_weights[k])

        logger.debug("FedProx aggregation completed")
        return AggregationResult(global_weights=global_weights,
                                 aux_stats={"algorithm": "FedProx", "mu": self.mu})


class FedYogi:
    """
    FedYogi – Adaptive federated optimiser (Yogi variant of Adam).

    Reference:
        R. Zhao et al., "Federated Learning with Yogi Optimizer", 2020.
    """

    def __init__(self, beta1: float = 0.9, beta2: float = 0.99, tau: float = 1e-3):
        self.beta1, self.beta2, self.tau = beta1, beta2, tau
        self.m: Dict[str, torch.Tensor] = {}
        self.v: Dict[str, torch.Tensor] = {}  # second moment

    def aggregate(
        self,
        client_weight_deltas: List[Dict[str, torch.Tensor]],
        client_sizes: List[int]
    ) -> Dict[str, torch.Tensor]:
        if not self.m:  # init moments
            for k in client_weight_deltas[0]:
                shape = client_weight_deltas[0][k].shape
                device = client_weight_deltas[0][k].device
                self.m[k] = torch.zeros(shape, device=device)
                self.v[k] = torch.zeros(shape, device=device)

        total_samples = sum(client_sizes)
        for k in self.m:
            grad = sum([delta[k] * (cs / total_samples)
                        for delta, cs in zip(client_weight_deltas, client_sizes)])
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grad
            self.v[k] = self.v[k] - (1 - self.beta2) * torch.sign(self.v[k] - grad ** 2) * grad ** 2
            update = self.m[k] / (torch.sqrt(self.v[k]) + self.tau)
            self.m[k] = update  # reuse m for actual update output

        return {k: self.m[k] for k in self.m}  # Return new global delta


# --------------------------------------------------------------------------- #
#  Secure Aggregation stub (interface only)                                   #
# --------------------------------------------------------------------------- #
class SecureAggregator:
    """
    Interface placeholder for secure aggregation (e.g., SecAgg, Homomorphic).

    Actual cryptographic routines are implemented in
    `federated-learning/privacy_preserving/*`.
    """

    def aggregate(self, *args, **kwargs):
        raise NotImplementedError("Secure aggregation not yet implemented")


# --------------------------------------------------------------------------- #
#  Factory helper                                                             #
# --------------------------------------------------------------------------- #
def get_aggregator(name: str, **kwargs):
    """Factory for getting aggregation algorithm by name."""
    name = name.lower()
    if name == "fedavg":
        return FedAvg()
    if name == "fedprox":
        return FedProx(**kwargs)
    if name == "fedyogi":
        return FedYogi(**kwargs)
    if name in {"secagg", "secureagg"}:
        return SecureAggregator()
    raise ValueError(f"Unsupported aggregation algorithm: {name}")
