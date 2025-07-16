"""
BatteryMind – Maintenance Scheduler
===================================

Coordinates preventive and corrective maintenance actions for battery fleets
based on real-time health inference, degradation forecasts and business rules.

Author  : BatteryMind Development Team
Version : 1.0.0
Created : 2025-01-10
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# BatteryMind internal imports
from inference.predictors.battery_health_predictor import BatteryHealthPredictor
from inference.predictors.degradation_predictor import DegradationPredictor
from monitoring.model_monitoring.performance_monitor import PerformanceMonitor
from utils.logging_utils import configure_logger

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
LOGGER = configure_logger(name="maintenance_scheduler", level=logging.INFO)

# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class MaintenancePolicy:
    """
    Domain rules that decide when to service or replace a battery pack.
    Values are examples – tune in `deployment_config.yaml`.
    """
    soh_threshold: float = 0.80          # Trigger when SoH < 80 %
    temperature_threshold: float = 50.0  # °C – sustained high temp triggers check
    internal_resistance_threshold: float = 0.15  # Ω
    max_cycles_before_check: int = 3_000
    max_calendar_age_months: int = 60
    degradation_rate_threshold: float = 0.015    # ΔSoH / month

    # Business parameters
    mandatory_check_months: int = 12
    critical_alert_soh: float = 0.65


@dataclass
class MaintenanceTask:
    """Represents a single scheduled task for a battery pack."""
    battery_id: str
    task_type: str  # e.g. PREVENTIVE, CORRECTIVE, CRITICAL
    scheduled_date: datetime
    reason: str
    predicted_soh: float
    additional_data: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Scheduler implementation
# --------------------------------------------------------------------------- #
class MaintenanceScheduler:
    """
    Analyses prediction streams and produces a maintenance work-order queue.
    """

    def __init__(
        self,
        health_predictor: BatteryHealthPredictor,
        degradation_predictor: Optional[DegradationPredictor] = None,
        policy: Optional[MaintenancePolicy] = None,
        work_queue_path: Path | str = "maintenance_work_queue.json",
    ) -> None:
        self.health_predictor = health_predictor
        self.degradation_predictor = degradation_predictor
        self.policy = policy or MaintenancePolicy()
        self.work_queue_path = Path(work_queue_path)
        self._initialize_storage()

        LOGGER.info("MaintenanceScheduler initialised – policy: %s", self.policy)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def evaluate_battery(
        self,
        battery_id: str,
        sensor_payload: Dict[str, float],
        metadata: Optional[Dict[str, str]] = None,
    ) -> Optional[MaintenanceTask]:
        """
        Called by the inference pipeline once fresh telemetry is available.

        Returns a MaintenanceTask if the battery qualifies for service.
        """
        predicted = self.health_predictor.predict(sensor_payload)
        predicted_soh: float = predicted["soh"]
        internal_resistance: float = predicted.get("internal_resistance", 0.0)
        temperature: float = sensor_payload.get("temperature", 25.0)
        cycle_count: int = int(sensor_payload.get("cycle_count", 0))
        age_months: int = int(sensor_payload.get("age_days", 0) // 30)

        degradation_rate = self._estimate_degradation_rate(
            battery_id=battery_id,
            current_soh=predicted_soh
        )

        LOGGER.debug(
            "Battery %s – SoH: %.3f, dRate: %.4f",
            battery_id, predicted_soh, degradation_rate
        )

        # Decision logic
        task_type, reason = self._classify(
            predicted_soh,
            internal_resistance,
            temperature,
            cycle_count,
            age_months,
            degradation_rate
        )

        if task_type is None:
            # No maintenance required
            return None

        task = MaintenanceTask(
            battery_id=battery_id,
            task_type=task_type,
            scheduled_date=datetime.utcnow(),
            reason=reason,
            predicted_soh=predicted_soh,
            additional_data={
                "temperature": temperature,
                "internal_resistance": internal_resistance,
                "cycle_count": cycle_count,
                "age_months": age_months,
                "degradation_rate": degradation_rate,
            }
        )
        self._enqueue(task)
        return task

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _classify(
        self,
        soh: float,
        r_int: float,
        temp: float,
        cycles: int,
        age: int,
        d_rate: float
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return (task_type, reason) or (None, None) if no action required."""
        p = self.policy

        # Critical
        if soh < p.critical_alert_soh:
            return "CRITICAL", "SoH below critical threshold"

        # Corrective
        if (
            soh < p.soh_threshold or
            r_int > p.internal_resistance_threshold or
            temp > p.temperature_threshold
        ):
            return "CORRECTIVE", "Parameter exceeded corrective threshold"

        # Preventive
        if (
            cycles > p.max_cycles_before_check or
            age > p.max_calendar_age_months or
            d_rate > p.degradation_rate_threshold
        ):
            return "PREVENTIVE", "Preventive maintenance due"

        return None, None

    def _estimate_degradation_rate(self, battery_id: str, current_soh: float) -> float:
        """
        Returns ΔSoH / month using predictor history or degradation model.
        """
        if self.degradation_predictor:
            forecast = self.degradation_predictor.forecast(battery_id)
            next_month_soh = forecast.get("soh_month+1", current_soh)
            return current_soh - next_month_soh

        # Fallback – naive constant rate
        return 0.0

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #
    def _initialize_storage(self) -> None:
        if not self.work_queue_path.exists():
            self.work_queue_path.write_text("[]")
            LOGGER.debug("Created new work-queue at %s", self.work_queue_path)

    def _enqueue(self, task: MaintenanceTask) -> None:
        data = json.loads(self.work_queue_path.read_text())
        data.append(self._task_to_dict(task))
        self.work_queue_path.write_text(json.dumps(data, indent=2, default=str))
        LOGGER.info(
            "Enqueued %s maintenance task for battery %s – reason: %s",
            task.task_type, task.battery_id, task.reason
        )

    @staticmethod
    def _task_to_dict(task: MaintenanceTask) -> Dict[str, str]:
        d = task.__dict__.copy()
        d["scheduled_date"] = d["scheduled_date"].isoformat()
        return d
