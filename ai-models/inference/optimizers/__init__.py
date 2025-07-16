"""
BatteryMind – Inference Optimizers Package
=========================================

This namespace exposes optimizers that translate high-level model
predictions into *actionable* control set-points for Battery
Management Systems (BMS), energy-management back-ends and edge
controllers.

Current sub-modules
-------------------
- ``charging_optimizer`` : Adaptive optimiser that converts the
  platform’s RL-policy outputs (or a physics-informed rule fallback)
  into real-time charging commands while enforcing hard-safety limits.

Copyright
---------
© 2025 Tata Technologies – BatteryMind Project
"""

from .charging_optimizer import ChargingOptimizer, ChargingOptimizerConfig  # noqa: F401

__all__ = [
    "ChargingOptimizer",
    "ChargingOptimizerConfig",
]
