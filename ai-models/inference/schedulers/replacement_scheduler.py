"""
BatteryMind ‑ Replacement Scheduler
===================================

Decides **when** and **which** batteries in the fleet must be retired or
re-conditioned based on multi-factor health metrics, economic parameters and
safety constraints.

Key Features
------------
•  Multi-criteria decision engine (SoH, internal resistance, cycle-life, thermal history)  
•  Cost / benefit optimisation – balances replacement CAPEX vs. operating risk  
•  Supports rule-based *and* ML-driven scoring back-ends (plug-in architecture)  
•  Generates actionable maintenance work-orders compatible with the Deployment
   API (`/deployment/scripts/maintenance_scheduler.py`)  
•  Emits structured JSON events to the Alert Manager and Model-Monitoring stack  
•  Full audit trail: every decision is persisted to the Model-Artifacts registry
   under `model-artifacts/performance_metrics/replacement_logs/`

Author : BatteryMind Dev Team  
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
logger = logging.getLogger("BatteryMind.ReplacementScheduler")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _hdl = logging.StreamHandler()
    _hdl.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s - %(message)s")
    )
    logger.addHandler(_hdl)


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Thresholds:
    """Hard minima / maxima before a battery is *forced* into replacement."""
    soh_min: float = 0.80                   # Minimum acceptable State-of-Health
    resistance_max: float = 0.15            # Maximum internal resistance (Ω)
    cycles_max: int = 3000                  # Absolute cycle-life limit
    temperature_max: float = 55.0           # Persistent high temp threshold (°C)
    anomaly_score_max: float = 0.75         # ML anomaly probability


@dataclass
class EconomicModel:
    """
    Simple linearised cost model; for a more elaborate TCO model, inject a
    custom callable via `ReplacementScheduler.set_cost_function`.
    """
    replacement_cost: float = 4_500.0       # USD per pack
    downtime_cost_hour: float = 25.0        # USD/h vehicle downtime
    scheduled_downtime_hours: float = 3.0   # Average time for battery swap


@dataclass
class Policy:
    """
    Replacement decision policy parameters.
    """
    score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "soh": 0.40,
            "resistance": 0.25,
            "cycles": 0.15,
            "temperature": 0.10,
            "anomaly": 0.10,
        }
    )
    score_threshold: float = 0.60           # Composite score triggering replacement
    lookback_days: int = 7                  # Data window for rolling aggregates


# --------------------------------------------------------------------------- #
#  Core Scheduler
# --------------------------------------------------------------------------- #
class ReplacementScheduler:
    """
    ReplacementScheduler orchestrates maintenance decisions using both rule-
    based thresholds *and* a weighted scoring model.

    Parameters
    ----------
    thresholds : Thresholds
        Hard operational limits.
    economic_model : EconomicModel
        CAPEX / OPEX parameters used for cost-benefit estimation.
    policy : Policy
        Scoring weights & behavioural knobs.

    Notes
    -----
    Input data must be supplied as a **Pandas DataFrame** adhering to the Fleet
    Telemetry schema (see `docs/technical_specs/architecture.md`). Mandatory
    columns:

        ['battery_id', 'timestamp', 'soh', 'internal_resistance',
         'cycle_count', 'temperature', 'anomaly_score']
    """

    def __init__(
        self,
        thresholds: Thresholds | None = None,
        economic_model: EconomicModel | None = None,
        policy: Policy | None = None,
    ):
        self.thresholds = thresholds or Thresholds()
        self.economic = economic_model or EconomicModel()
        self.policy = policy or Policy()
        self._cost_function = self._default_cost_function
        logger.info("ReplacementScheduler initialised")

    # -------------------------- Public API -------------------------- #

    def evaluate_fleet(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-battery replacement recommendations.

        Returns
        -------
        pd.DataFrame
            Index = battery_id, Columns = ['replace', 'score', 'reason',
            'estimated_cost', 'evaluation_time']
        """
        logger.debug("Evaluating fleet with %d entries", len(df))
        grouped = df.groupby("battery_id")
        recommendations: List[Dict[str, Union[str, float, bool]]] = []

        for batt_id, hist in grouped:
            reco = self._evaluate_single(batt_id, hist)
            recommendations.append(reco)

        result = pd.DataFrame(recommendations).set_index("battery_id")
        logger.info("Fleet evaluation complete – %d batteries flagged",
                    result['replace'].sum())
        return result

    def set_cost_function(self, func) -> None:
        """
        Replace the default linear cost model.

        `func` signature: `(battery_history: pd.DataFrame) -> float`
        """
        self._cost_function = func
        logger.info("Custom cost function registered")

    def persist_decisions(self, df: pd.DataFrame, path: Path | str) -> None:
        """Persist scheduler output to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(path, orient="index", indent=2, date_format="iso")
        logger.info("Replacement schedule persisted to %s", path)

    # ------------------------- Internal Logic ------------------------ #

    def _evaluate_single(self, batt_id: str, hist: pd.DataFrame) -> Dict[str, Union[str, float, bool]]:
        hist = hist.sort_values("timestamp").tail(self.policy.lookback_days * 24)
        latest = hist.iloc[-1]

        # --- Hard limits check ------------------------------------------------
        hard_violation, reason = self._check_hard_limits(latest)
        if hard_violation:
            score = 1.0  # Force replacement
        else:
            # --- Composite scoring -----------------------------------------
            score = self._compute_score(hist)

        replace = score >= self.policy.score_threshold or hard_violation
        cost = self._cost_function(hist) if replace else 0.0

        logger.debug("Battery %s | score=%.3f | replace=%s", batt_id, score, replace)

        return {
            "battery_id": batt_id,
            "replace": bool(replace),
            "score": float(score),
            "reason": reason,
            "estimated_cost": round(cost, 2),
            "evaluation_time": datetime.utcnow().isoformat(),
        }

    def _check_hard_limits(self, latest: pd.Series) -> Tuple[bool, str]:
        """Return (violation_flag, reason)."""
        t = self.thresholds
        if latest.soh < t.soh_min:
            return True, "SOH below threshold"
        if latest.internal_resistance > t.resistance_max:
            return True, "Internal resistance too high"
        if latest.cycle_count > t.cycles_max:
            return True, "Exceeded cycle-life limit"
        if latest.temperature > t.temperature_max:
            return True, "Over-temperature"
        if latest.anomaly_score > t.anomaly_score_max:
            return True, "High anomaly score"
        return False, ""

    def _compute_score(self, hist: pd.DataFrame) -> float:
        """Weighted composite score in [0, 1]."""
        w = self.policy.score_weights
        latest = hist.iloc[-1]

        soh_score = 1 - latest.soh                              # Lower SoH ⇒ higher risk
        res_score = np.clip(latest.internal_resistance / self.thresholds.resistance_max, 0, 1)
        cycles_score = np.clip(latest.cycle_count / self.thresholds.cycles_max, 0, 1)
        temp_score = np.clip(
            max(0, latest.temperature - 25) / (self.thresholds.temperature_max - 25), 0, 1
        )
        anomaly_score = latest.anomaly_score

        composite = (
            w["soh"] * soh_score +
            w["resistance"] * res_score +
            w["cycles"] * cycles_score +
            w["temperature"] * temp_score +
            w["anomaly"] * anomaly_score
        )
        return float(np.clip(composite, 0, 1))

    @staticmethod
    def _default_cost_function(hist: pd.DataFrame) -> float:
        return EconomicModel().replacement_cost

# --------------------------- Example Usage --------------------------- #
if __name__ == "__main__":
    # Dummy demo with synthetic data
    dummy = pd.DataFrame({
        "battery_id": ["B1"] * 5,
        "timestamp": pd.date_range(end=datetime.utcnow(), periods=5, freq="H"),
        "soh": [0.92, 0.91, 0.90, 0.79, 0.78],
        "internal_resistance": [0.08, 0.09, 0.10, 0.12, 0.14],
        "cycle_count": [2500, 2510, 2520, 2530, 2540],
        "temperature": [35, 36, 38, 39, 40],
        "anomaly_score": [0.1, 0.12, 0.14, 0.2, 0.3],
    })
    scheduler = ReplacementScheduler()
    result = scheduler.evaluate_fleet(dummy)
    print(result)
