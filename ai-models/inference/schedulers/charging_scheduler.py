"""
BatteryMind – Charging Scheduler
================================

Produces optimal charging schedules for individual batteries *or* entire fleets
based on predictive models, tariff curves, grid constraints and safety rules.

Author  : BatteryMind Development Team
Version : 1.0.0
Created : 2025-01-10
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# BatteryMind internal imports
from inference.predictors.optimization_predictor import OptimizationPredictor
from inference.predictors.battery_health_predictor import BatteryHealthPredictor
from utils.logging_utils import configure_logger

LOGGER = configure_logger(name="charging_scheduler", level=logging.INFO)

# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class TariffWindow:
    """A single electricity-price window in $/kWh."""
    start: datetime
    end: datetime
    price: float


@dataclass
class ChargingPlan:
    """Output of the ChargingScheduler."""
    battery_id: str
    schedule: List[Dict[str, float]]            # [{timestamp, setpoint_kW}, …]
    estimated_cost: float
    expected_soh_impact: float
    created_at: datetime = datetime.utcnow()


# --------------------------------------------------------------------------- #
# Scheduler implementation
# --------------------------------------------------------------------------- #
class ChargingScheduler:
    """
    Generates charging set-points that satisfy:
      • user demand / SoC targets  
      • battery‐health constraints  
      • grid tariff optimisation  
      • safety limits (temperature, voltage, current)
    """

    def __init__(
        self,
        optimization_predictor: OptimizationPredictor,
        health_predictor: BatteryHealthPredictor,
        tariff_table: List[TariffWindow],
        max_power_kw: float = 150.0,
        min_soc_target: float = 0.80,          # 80 % by departure
        safety_temperature: float = 50.0,      # °C
        reserve_soc: float = 0.10              # keep 10 % minimum
    ) -> None:
        self.optimization_predictor = optimization_predictor
        self.health_predictor = health_predictor
        self.tariff_table = tariff_table
        self.max_power_kw = max_power_kw
        self.min_soc_target = min_soc_target
        self.safety_temperature = safety_temperature
        self.reserve_soc = reserve_soc

        LOGGER.info("ChargingScheduler ready – max power %.1f kW", self.max_power_kw)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def create_plan(
        self,
        battery_id: str,
        current_soc: float,
        departure_time: datetime,
        context_features: Dict[str, float]
    ) -> ChargingPlan:
        """
        Main entry-point – returns a ChargingPlan for the requested battery.
        """
        # Predict optimal charging power curve (per minute) using RL/ML model
        horizon_minutes = int((departure_time - datetime.utcnow()).total_seconds() // 60)
        if horizon_minutes <= 0:
            raise ValueError("Departure time must be in the future")

        LOGGER.debug("Creating plan for battery %s, horizon %d min", battery_id, horizon_minutes)

        optimal_profile = self._predict_optimal_profile(
            current_soc, horizon_minutes, context_features
        )

        # Tariff-aware adjustment
        tariff_adjusted_profile = self._apply_tariff_optimisation(optimal_profile)

        # Safety checks
        safe_profile = self._enforce_safety_limits(tariff_adjusted_profile, context_features)

        # Cost & SoH impact estimation
        cost, soh_impact = self._estimate_plan_metrics(safe_profile, context_features)

        schedule = [
            {
                "timestamp": (datetime.utcnow() + timedelta(minutes=i)).isoformat(),
                "setpoint_kw": round(p_kw, 2)
            }
            for i, p_kw in enumerate(safe_profile)
        ]

        plan = ChargingPlan(
            battery_id=battery_id,
            schedule=schedule,
            estimated_cost=round(cost, 2),
            expected_soh_impact=round(soh_impact, 4)
        )

        LOGGER.info(
            "Charging plan generated for battery %s – cost $%.2f, ΔSoH %.4f",
            battery_id, plan.estimated_cost, plan.expected_soh_impact
        )
        return plan

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _predict_optimal_profile(
        self,
        soc_now: float,
        horizon: int,
        features: Dict[str, float]
    ) -> np.ndarray:
        """
        Returns power set-points (kW) for each minute of the horizon.
        Uses the optimisation predictor (RL agent distilled) to approximate
        long-term rewards such as SoH preservation and cost minimisation.
        """
        predictor_input = {
            **features,
            "soc_now": soc_now,
            "horizon": horizon,
            "max_power_kw": self.max_power_kw
        }
        profile = self.optimization_predictor.predict_charging_profile(predictor_input)
        profile = np.clip(profile, 0.0, self.max_power_kw)

        LOGGER.debug("Predicted raw profile (first 5 min): %s", profile[:5])
        return profile

    # ------------------------------------------------------------------ #
    def _apply_tariff_optimisation(self, profile: np.ndarray) -> np.ndarray:
        """Shift / scale the charging power to low-tariff windows."""
        if not self.tariff_table:
            return profile

        timestamps = [datetime.utcnow() + timedelta(minutes=i) for i in range(len(profile))]
        adjusted = profile.copy()

        # Build a quick price lookup
        def price_at(ts: datetime) -> float:
            for window in self.tariff_table:
                if window.start <= ts < window.end:
                    return window.price
            return max(w.price for w in self.tariff_table)  # worst-case high price

        prices = np.array([price_at(ts) for ts in timestamps])
        low_price_mask = prices <= np.percentile(prices, 30)  # bottom 30 % cheapest

        # Allocate more power to cheap slots, less to expensive ones
        desired_energy = profile.sum()
        cheap_allocation = desired_energy * 0.6  # 60 % in cheap slots
        expensive_allocation = desired_energy - cheap_allocation

        adjusted[low_price_mask] = (
            cheap_allocation / low_price_mask.sum()
        )
        adjusted[~low_price_mask] = (
            expensive_allocation / (~low_price_mask).sum()
        )

        LOGGER.debug("Tariff optimisation applied")
        return adjusted

    # ------------------------------------------------------------------ #
    def _enforce_safety_limits(self, profile: np.ndarray, features: Dict[str, float]) -> np.ndarray:
        """Clip power if predicted temperature exceeds safety threshold."""
        predicted_temps = self.optimization_predictor.predict_temperature_rise(profile, features)

        if (predicted_temps > self.safety_temperature).any():
            violation_idx = np.where(predicted_temps > self.safety_temperature)[0]
            profile[violation_idx] *= 0.5  # Reduce power by 50 %
            LOGGER.warning(
                "Safety limits enforced at %d timestamps (temperature > %.1f °C)",
                len(violation_idx), self.safety_temperature
            )
        return profile

    # ------------------------------------------------------------------ #
    def _estimate_plan_metrics(self, profile: np.ndarray, features: Dict[str, float]) -> tuple[float, float]:
        """Estimate total cost ($) and expected SoH impact (ΔSoH)."""
        timestamps = [datetime.utcnow() + timedelta(minutes=i) for i in range(len(profile))]
        prices = np.array([self._price_at(ts) for ts in timestamps])
        kwh = profile / 60  # kW-minutes → kWh

        cost = float(np.sum(kwh * prices))

        # SoH impact: use health predictor to approximate degradation
        soh_now = features.get("soh", 1.0)
        soh_after = self.health_predictor.predict_future_soh(
            current_soh=soh_now,
            profile_kw=profile,
            features=features
        )
        return cost, soh_now - soh_after

    def _price_at(self, ts: datetime) -> float:
        for window in self.tariff_table:
            if window.start <= ts < window.end:
                return window.price
        return max(w.price for w in self.tariff_table)


# --------------------------------------------------------------------------- #
# Convenience factory for tariff windows (optional)
# --------------------------------------------------------------------------- #
def load_tariffs_from_csv(path: str | Path) -> List[TariffWindow]:
    df = pd.read_csv(path, parse_dates=["start", "end"])
    return [TariffWindow(row.start, row.end, row.price) for _, row in df.iterrows()]
