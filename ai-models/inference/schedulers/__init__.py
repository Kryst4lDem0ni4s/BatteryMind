"""
BatteryMind ‑ Fleet Optimizer
─────────────────────────────
Aggregates per-battery optimizers to generate fleet-level, constraint-aware
actions. Designed for real-time inference pipelines and batch simulations.

Author  : BatteryMind Development Team
Version : 1.0.0
License : MIT
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from .charging_optimizer import ChargingOptimizer
from .thermal_optimizer import ThermalOptimizer
from .load_optimizer import LoadOptimizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FleetOptimizer:
    """
    Coordinates individual optimizers to optimise an entire battery fleet.

    Workflow
    --------
    1.  Each battery’s latest state is passed to its dedicated sub-optimizer.
    2.  Fleet-level hard limits (site power, transformer cap, tariff windows)
        are enforced via a post-processing step.
    3.  Returned action set is compatible with the real-time inference pipeline.
    """

    # --------------------------------------------------------------------- #
    # Construction helpers
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        max_workers: int = 8,
        charging_cfg: Dict | None = None,
        thermal_cfg: Dict | None = None,
        load_cfg: Dict | None = None,
        fleet_constraints: Dict | None = None,
    ) -> None:
        self.charging_opt = ChargingOptimizer(**(charging_cfg or {}))
        self.thermal_opt = ThermalOptimizer(**(thermal_cfg or {}))
        self.load_opt = LoadOptimizer(**(load_cfg or {}))

        # Default fleet constraints
        self.constraints = {
            "site_power_limit_kw": 2_000,
            "peak_shaving_enabled": True,
            "tariff_windows": [],  # list of tuples: (start_ts, end_ts, max_kw)
            "min_soc_reserve": 0.2,
        }
        if fleet_constraints:
            self.constraints.update(fleet_constraints)

        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        logger.info("FleetOptimizer initialised with %d workers", max_workers)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def optimise_fleet(
        self,
        fleet_state: List[Dict],
        timestamp: int,
    ) -> List[Dict]:
        """
        Parameters
        ----------
        fleet_state : list(dict)
            Latest telemetry for every battery pack in the fleet.
        timestamp   : int (epoch seconds)
            Current epoch (needed for tariff windows & forecasting).

        Returns
        -------
        list(dict)
            Action dictionaries keyed by battery_id.
        """
        logger.debug("Received %d battery states for optimisation", len(fleet_state))
        futures = {
            self.pool.submit(self._optimise_single, state, timestamp): state["battery_id"]
            for state in fleet_state
        }

        actions = {}
        for future in as_completed(futures):
            bid = futures[future]
            try:
                actions[bid] = future.result()
            except Exception as exc:  # pragma: no cover
                logger.exception("Optimisation failed for %s: %s", bid, exc)
                actions[bid] = {"error": str(exc)}

        # Fleet-level reconciliation
        reconciled = self._enforce_fleet_constraints(actions, timestamp)
        logger.info("Fleet optimisation complete (%d packs)", len(reconciled))
        return reconciled

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _optimise_single(self, state: Dict, ts: int) -> Dict:
        """Optimise a single battery pack."""
        bid = state["battery_id"]
        charge_action = self.charging_opt.optimise(state, ts)
        thermal_action = self.thermal_opt.optimise(state, ts)
        load_action = self.load_opt.optimise(state, ts)

        combined = {
            "battery_id": bid,
            **charge_action,
            **thermal_action,
            **load_action,
        }
        logger.debug("Battery %s actions: %s", bid, combined)
        return combined

    def _enforce_fleet_constraints(
        self,
        actions: Dict[str, Dict],
        ts: int,
    ) -> List[Dict]:
        """
        Ensures aggregated actions respect site-level limits and tariff windows.
        Adjusts charging currents proportionally when needed.
        """
        site_limit = self._current_site_limit(ts)
        total_kw = sum(a.get("charging_power_kw", 0) - a.get("discharge_power_kw", 0)
                       for a in actions.values())

        if total_kw > site_limit:
            logger.warning("Site limit exceeded (%.1f > %.1f kW) – scaling actions",
                           total_kw, site_limit)
            scale = site_limit / total_kw
            for a in actions.values():
                if "charging_power_kw" in a:
                    a["charging_power_kw"] *= scale
                if "charging_current_a" in a:
                    a["charging_current_a"] *= scale

        # Minimum SoC reserve safeguard
        for a in actions.values():
            if a.get("predicted_soc", 1.0) < self.constraints["min_soc_reserve"]:
                a["charging_power_kw"] = max(
                    a.get("charging_power_kw", 0),
                    self.charging_opt.minimum_top_up_kw,
                )

        return list(actions.values())

    def _current_site_limit(self, ts: int) -> float:
        """Returns the active site power limit for the given timestamp."""
        for start, end, max_kw in self.constraints["tariff_windows"]:
            if start <= ts < end:
                return max_kw
        return self.constraints["site_power_limit_kw"]

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        """Gracefully terminate thread-pool workers."""
        self.pool.shutdown(wait=True)
        logger.info("FleetOptimizer shut down")

