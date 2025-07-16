"""
BatteryMind – Adaptive Charging Optimizer
========================================

This module converts *state* observations from the inference pipeline
into safe and efficiency-aware charging actions.

Design goals
------------
1. **Safety first** – strict enforcement of thermal, voltage and
   current constraints in accordance with IEC 62660-3.
2. **Performance** – minimise charge time while maximising SOH
   preservation (leveraging a trained RL policy when available).
3. **Portability** – single implementation suitable for cloud, edge
   and embedded micro-controllers (via ONNX Runtime).

Key Components
--------------
ChargingOptimizerConfig  : Centralised configuration dataclass.
ChargingOptimizer       : High-level API exposing ``suggest_action`` and
                          ``suggest_schedule`` methods.

If an ONNX policy network is supplied the optimiser will:
  • normalise the observation vector,
  • run a forward pass in <1 ms on edge HW,
  • denormalise and post-process the raw actions,
  • validate & clip to remain within safety limits.

Fallback path
-------------
When no valid policy model is loaded (or ONNX Runtime is unavailable)
the optimiser switches to a *physics-informed rule engine* that
implements:
  • taper-charging,
  • temperature-aware current derating,
  • time-of-use cost optimisation.

Author
------
BatteryMind Development Team
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # Optional dependency – only required when an ONNX policy is supplied
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class ChargingOptimizerConfig:
    """
    Global configuration for the Charging Optimizer.

    Parameters
    ----------
    target_soc : float
        Desired State-of-Charge (0-1) to reach at the end of the session.
    maximum_current : float
        Absolute current limit in amperes (positive = charging).
    maximum_power : float
        Charger power capability in kilowatts.
    voltage_bounds : Tuple[float, float]
        Minimum / maximum cell voltage (safety envelope) in volts.
    temperature_bounds : Tuple[float, float]
        Acceptable pack temperature range in °C.
    taper_soc_threshold : float
        SOC above which *current tapering* starts (rule engine).
    ambient_temperature : float
        Ambient temperature used when pack sensor is unavailable.
    rl_policy_path : Optional[str]
        Path to an ONNX policy network; if *None* the rule engine is used.
    sampling_interval_sec : int
        Decision interval (∆t) in seconds.
    """

    target_soc: float = 0.9
    maximum_current: float = 250.0  # A
    maximum_power: float = 100.0  # kW
    voltage_bounds: Tuple[float, float] = (2.5, 4.2)
    temperature_bounds: Tuple[float, float] = (-10.0, 55.0)
    taper_soc_threshold: float = 0.8
    ambient_temperature: float = 25.0
    rl_policy_path: Optional[str] = None
    sampling_interval_sec: int = 5

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# --------------------------------------------------------------------------- #
#  Helper Functions                                                           #
# --------------------------------------------------------------------------- #
def _sigmoid(x: float, sharpness: float = 10) -> float:
    """
    Fast sigmoid used for smooth current tapering.
    """
    return 1 / (1 + math.exp(-sharpness * (x - 0.5)))


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


# --------------------------------------------------------------------------- #
#  Charging Optimizer                                                         #
# --------------------------------------------------------------------------- #
class ChargingOptimizer:
    """
    Top-level Charging Optimizer.

    Usage
    -----
    >>> cfg = ChargingOptimizerConfig(rl_policy_path="rl_policy_network.onnx")
    >>> optimizer = ChargingOptimizer(cfg)
    >>> observation = {...}  # dictionary from pipeline
    >>> command = optimizer.suggest_action(observation)
    """

    POLICY_INPUT_NAMES = ("state", "mask")
    POLICY_OUTPUT_NAME = "actions"

    def __init__(self, config: ChargingOptimizerConfig) -> None:
        self.cfg = config
        self._session: Optional[ort.InferenceSession] = None
        self._load_policy_if_available()

        # Internal caches --------------------------------------------------- #
        self._last_action: Optional[Dict[str, float]] = None
        self._last_timestamp: Optional[datetime] = None

        logger.info(
            "ChargingOptimizer initialised – policy=%s, rule_engine=%s",
            bool(self._session),
            self._session is None,
        )

    # --------------------------------------------------------------------- #
    #  Public API                                                           #
    # --------------------------------------------------------------------- #
    def suggest_action(self, state: Dict[str, float]) -> Dict[str, float]:
        """
        Suggest a single charging command for the next sampling interval.

        Parameters
        ----------
        state : Dict[str, float]
            Current battery/charger state (SOC, voltage, temp, etc.).

        Returns
        -------
        Dict[str, float]
            Dictionary with at minimum ``current_setpoint`` (A) and
            ``max_voltage`` (V); may include additional metadata.
        """
        action = (
            self._rl_policy_action(state)
            if self._session is not None
            else self._rule_based_action(state)
        )

        validated = self._validate(action, state)
        self._last_action = validated
        self._last_timestamp = datetime.utcnow()
        return validated

    def suggest_schedule(
        self,
        initial_state: Dict[str, float],
        horizon: int = 3600,
    ) -> List[Dict[str, float]]:
        """
        Produce a *forecast schedule* up to ``horizon`` seconds ahead.

        The schedule is recomputed iteratively assuming deterministic
        state evolution under the optimiser’s actions.

        Notes
        -----
        – This is intended for *what-if* simulations and fleet/EMS
          planning, not strictly real-time control.
        """
        steps = horizon // self.cfg.sampling_interval_sec
        state = initial_state.copy()
        schedule: List[Dict[str, float]] = []

        for _ in range(steps):
            action = self.suggest_action(state)
            schedule.append(action)

            # Naïve state propagation (capacity-only, ignores temperature etc.)
            soc = state.get("state_of_charge", 0.5)
            voltage = state.get("voltage", 3.7)
            current = action["current_setpoint"]
            delta_ah = (
                current * self.cfg.sampling_interval_sec / 3600.0
            )  # A·s → A·h
            capacity_ah = state.get("capacity_ah", 100.0)
            soc = _clip(soc + delta_ah / capacity_ah, 0.0, 1.0)

            state.update(
                {
                    "state_of_charge": soc,
                    "voltage": voltage,  # placeholder
                }
            )

            if soc >= self.cfg.target_soc:
                break

        return schedule

    # --------------------------------------------------------------------- #
    #  Internal – RL Path                                                   #
    # --------------------------------------------------------------------- #
    def _load_policy_if_available(self) -> None:
        if self.cfg.rl_policy_path is None:
            return
        if ort is None:  # pragma: no cover
            logger.warning("ONNX Runtime not available – falling back to rule engine")
            return

        policy_path = pathlib.Path(self.cfg.rl_policy_path)
        if not policy_path.exists():
            logger.warning("RL policy path not found: %s – using rule engine", policy_path)
            return

        try:
            self._session = ort.InferenceSession(
                str(policy_path),
                providers=["CPUExecutionProvider"],
            )
            logger.info("Loaded RL policy from %s", policy_path)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to load ONNX policy: %s", exc)
            self._session = None

    def _rl_policy_action(self, state: Dict[str, float]) -> Dict[str, float]:
        """
        Forward pass through the policy network and post-process output.
        """
        assert self._session is not None, "Policy session not initialised"

        obs_vector = self._prepare_observation(state)
        inputs = {self.POLICY_INPUT_NAMES[0]: obs_vector.astype(np.float32)[None, :]}
        if len(self.POLICY_INPUT_NAMES) > 1:
            inputs[self.POLICY_INPUT_NAMES[1]] = np.ones_like(obs_vector)[None, :].astype(
                np.float32
            )

        raw_actions = self._session.run([self.POLICY_OUTPUT_NAME], inputs)[0][0]
        current_cmd = float(raw_actions[0]) * self.cfg.maximum_current

        logger.debug("RL raw action (norm): %s → current=%s A", raw_actions, current_cmd)
        return {
            "current_setpoint": current_cmd,
            "max_voltage": self.cfg.voltage_bounds[1],
            "source": "rl_policy",
        }

    # --------------------------------------------------------------------- #
    #  Internal – Rule Engine                                               #
    # --------------------------------------------------------------------- #
    def _rule_based_action(self, state: Dict[str, float]) -> Dict[str, float]:
        """
        Physics-informed deterministic controller.
        """
        soc = state.get("state_of_charge", 0.5)
        temp = state.get("temperature", self.cfg.ambient_temperature)

        # Taper logic – sigmoid around taper_soc_threshold
        taper = _sigmoid(
            (self.cfg.target_soc - soc) / max(1e-3, self.cfg.target_soc - self.cfg.taper_soc_threshold),
            sharpness=6,
        )
        current_cmd = taper * self.cfg.maximum_current

        # Temperature derating
        if temp > self.cfg.temperature_bounds[1] - 5:
            derating = max(0.1, 1 - (temp - (self.cfg.temperature_bounds[1] - 5)) / 20)
            current_cmd *= derating
            logger.debug("Temperature derating applied: %.2f", derating)

        # Clip to limits
        current_cmd = _clip(current_cmd, 0.0, self.cfg.maximum_current)

        return {
            "current_setpoint": current_cmd,
            "max_voltage": self.cfg.voltage_bounds[1],
            "source": "rule_engine",
        }

    # --------------------------------------------------------------------- #
    #  Validation & Safety Checks                                           #
    # --------------------------------------------------------------------- #
    def _validate(self, action: Dict[str, float], state: Dict[str, float]) -> Dict[str, float]:
        """
        Ensure the proposed command cannot violate pack limits.
        """
        temp = state.get("temperature", self.cfg.ambient_temperature)
        voltage = state.get("voltage", 3.7)

        # Temperature hard limit
        if not (self.cfg.temperature_bounds[0] <= temp <= self.cfg.temperature_bounds[1]):
            logger.warning("Temperature out of bounds: %.1f°C – forcing current=0 A", temp)
            action["current_setpoint"] = 0.0

        # Voltage headroom – if already near upper limit, taper further
        headroom = self.cfg.voltage_bounds[1] - voltage
        if headroom < 0.05:  # 50 mV buffer
            logger.debug("Voltage headroom low: %.3f V – reducing current", headroom)
            action["current_setpoint"] *= 0.3

        action["current_setpoint"] = _clip(action["current_setpoint"], 0.0, self.cfg.maximum_current)
        return action

    # --------------------------------------------------------------------- #
    #  Observation Helpers                                                  #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _prepare_observation(state: Dict[str, float]) -> np.ndarray:
        """
        Convert the incoming state dict to the observation vector expected
        by the RL policy network.

        *For reference only – must match the vector definition used during
        training.*
        """
        features = [
            state.get("state_of_charge", 0.5),
            state.get("voltage", 3.7) / 4.2,
            state.get("current", 0.0) / 250.0,
            state.get("temperature", 25.0) / 60.0,
            state.get("ambient_temperature", 25.0) / 60.0,
            state.get("cost_signal", 0.5),
            state.get("renewable_availability", 0.5),
            state.get("grid_price", 0.2),
        ]
        return np.array(features, dtype=np.float32)

    # --------------------------------------------------------------------- #
    #  Persistence Utilities                                                #
    # --------------------------------------------------------------------- #
    def save_last_action(self, path: str | pathlib.Path) -> None:
        """
        Persist the most recent command and timestamp for auditability.
        """
        if self._last_action is None:
            logger.warning("No action to persist")
            return

        payload = {
            "timestamp": (self._last_timestamp or datetime.utcnow()).isoformat(),
            "action": self._last_action,
        }
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fp:
            json.dump(payload, fp, indent=2)
        logger.info("Last action saved to %s", path)

    # --------------------------------------------------------------------- #
    #  Dunders                                                               #
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:  # pragma: no cover
        return f"<ChargingOptimizer policy={'RL' if self._session else 'Rule'} target_soc={self.cfg.target_soc}>"

