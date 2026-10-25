"""Authoritative per-simulation worker loop, simulation mode, and bounded snapshot read interface."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import threading
import time
from typing import Any
import uuid

import numpy as np

from thermotwin.control.actions import generate_candidates
from thermotwin.control.objectives import score_rollout
from thermotwin.control.policies import make_policy
from thermotwin.experiments.metrics import summarize_snapshots
from thermotwin.forecasting.hybrid import ResidualCorrector
from thermotwin.physics.plant import DataCenterPlant
from thermotwin.physics.twin import DigitalTwin
from thermotwin.physics.workload import SCENARIOS, scenario_inputs
from thermotwin.types import Action, Snapshot


CONTROLLERS = ("fixed", "reactive", "physics_mpc", "hybrid_mpc")


@dataclass
class SimulationMode:
    """Branch state for the interactive simulation mode.

    When active, the live session's plant is preserved; the shadow_plant is a deep-copy
    snapshot that absorbs user perturbations and applied candidates without mutating the
    background run.
    """

    baseline_snapshot: Snapshot
    """Policy-visible state at the moment simulation mode was activated."""
    shadow_plant: DataCenterPlant
    """Isolated clone of the plant; absorbs perturbations and applied candidates."""
    perturbed_racks: list[int] = field(default_factory=list)
    """Indices of racks whose temperature or load were manually changed."""
    applied_actions: list[dict[str, Any]] = field(default_factory=list)
    """Candidate actions that have been committed to the shadow plant."""
    activated_at_s: float = 0.0
    """Shadow plant time when simulation mode was activated."""


@dataclass
class SimulationSession:
    """All mutable state owned by exactly one worker and serialized by one lock."""

    simulation_id: str
    config: dict[str, Any]
    scenario: str
    controller: str
    seed: int
    twin: DigitalTwin
    plant: DataCenterPlant = field(init=False)
    policy: Any = field(init=False)
    workload_rng: np.random.Generator = field(init=False)
    status: str = "initialized"
    speed: float = 1.0
    step_index: int = 0
    decisions: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    updated_monotonic: float = field(default_factory=time.monotonic)
    lock: threading.RLock = field(default_factory=threading.RLock)
    wake: threading.Event = field(default_factory=threading.Event)
    stop: threading.Event = field(default_factory=threading.Event)
    simulation_mode: SimulationMode | None = None
    """Active simulation mode branch, or None when in normal/preset operation."""

    def __post_init__(self) -> None:
        self._reset_objects()

    def _reset_objects(self) -> None:
        self.plant = DataCenterPlant(self.config, self.seed)
        self.policy = make_policy(self.controller, self.config, self.twin)
        self.workload_rng = np.random.default_rng(self.seed ^ 0xC0FFEE)
        self.step_index, self.decisions, self.error = 0, [], None

    def step_once(self) -> None:
        offered, ambient, humidity = scenario_inputs(self.step_index, self.plant.n_racks, self.scenario, self.workload_rng)
        action = Action.noop(self.plant.n_zones)
        interval = int(self.config["control"]["decision_interval_steps"])
        if self.step_index % interval == 0:
            decision = self.policy.decide(self.plant.snapshot(ambient, humidity))
            action = decision.chosen_action
            self.decisions.append(decision.as_dict())
            self.decisions = self.decisions[-250:]
        accepted, reason = self.plant.apply_action(action)
        if not accepted:
            self.decisions.append({"time_s": self.plant.time_s, "controller": self.controller, "fallback": True, "explanation": f"Live action rejected: {reason}", "candidates": []})
        degradation = 0.72 if self.scenario == "cooling_degradation" and self.step_index >= 60 else None
        self.plant.advance(offered, ambient, humidity, degradation=degradation)
        max_history = int(self.config["service"]["max_history"])
        if len(self.plant.history) > max_history:
            self.plant.history = self.plant.history[-max_history:]
            self.plant.truth_history = self.plant.truth_history[-max_history:]
        self.step_index += 1
        self.updated_monotonic = time.monotonic()


class SimulationService:
    """Create, command, inspect, and terminate simulations without GET side effects."""

    def __init__(self, config: dict[str, Any], bundle: dict[str, Any] | None = None):
        self.config, self.bundle = config, bundle
        residual = bundle["action_residual"] if bundle else ResidualCorrector()
        self.twin = DigitalTwin(config, residual)
        self.sessions: dict[str, SimulationSession] = {}
        self.lock = threading.RLock()

    @property
    def trained_models_available(self) -> bool:
        return self.bundle is not None

    def create(self, scenario: str, controller: str, speed: float = 1.0, seed: int | None = None) -> dict[str, Any]:
        if scenario not in SCENARIOS:
            raise ValueError("unknown scenario")
        if controller not in CONTROLLERS:
            raise ValueError("unknown controller")
        if controller == "hybrid_mpc" and not self.trained_models_available:
            raise ValueError("hybrid_mpc requires prepared models; run prepare --profile smoke")
        if not np.isfinite(speed) or not 0.25 <= speed <= 20:
            raise ValueError("speed must be finite and between 0.25 and 20")
        identifier = uuid.uuid4().hex[:12]
        session = SimulationSession(identifier, self.config, scenario, controller, int(seed or self.config["project"]["seed"]), self.twin, speed=float(speed))
        with self.lock:
            self.sessions[identifier] = session
        thread = threading.Thread(target=self._worker, args=(session,), name=f"thermotwin-sim-{identifier}", daemon=True)
        thread.start()
        return self.describe(identifier)

    def _worker(self, session: SimulationSession) -> None:
        while not session.stop.is_set():
            if session.status != "running":
                session.wake.wait(0.25)
                session.wake.clear()
                continue
            try:
                with session.lock:
                    if session.status == "running":
                        session.step_once()
            except Exception as error:
                with session.lock:
                    session.status, session.error = "failed", f"{type(error).__name__}: {error}"
            session.stop.wait(float(session.config["service"]["playback_interval_s"]) / session.speed)

    def get(self, simulation_id: str) -> SimulationSession:
        with self.lock:
            session = self.sessions.get(simulation_id)
        if session is None:
            raise KeyError(simulation_id)
        return session

    def control(self, simulation_id: str, command: str, speed: float | None = None) -> dict[str, Any]:
        session = self.get(simulation_id)
        if command not in {"play", "pause", "step", "reset", "speed"}:
            raise ValueError("unknown control command")
        with session.lock:
            if command == "play":
                if session.status == "failed":
                    raise ValueError("failed simulation must be reset")
                session.status = "running"
            elif command == "pause":
                session.status = "paused"
            elif command == "step":
                if session.status == "running":
                    raise ValueError("pause before single-step")
                session.step_once()
                session.status = "paused"
            elif command == "reset":
                session.status = "paused"
                session._reset_objects()
                session.updated_monotonic = time.monotonic()
            elif command == "speed":
                if speed is None or not np.isfinite(speed) or not 0.25 <= speed <= 20:
                    raise ValueError("speed must be finite and between 0.25 and 20")
                session.speed = float(speed)
            session.wake.set()
        return self.describe(simulation_id)

    def describe(self, simulation_id: str) -> dict[str, Any]:
        """Read state only; it never advances simulated time."""
        session = self.get(simulation_id)
        with session.lock:
            snapshot = session.plant.history[-1]
            metrics = summarize_snapshots(session.plant.history, session.config["thresholds"]["hotspot_c"], session.config["thresholds"]["hard_limit_c"])
            stale = session.status == "running" and time.monotonic() - session.updated_monotonic > 5
            return {
                "simulation_id": simulation_id, "status": "stale" if stale else session.status,
                "scenario": session.scenario, "controller": session.controller, "speed": session.speed,
                "step_index": session.step_index, "error": session.error, "snapshot": snapshot.as_dict(),
                "metrics": metrics, "models_available": self.trained_models_available,
            }

    def history(self, simulation_id: str, limit: int) -> list[dict[str, Any]]:
        session = self.get(simulation_id)
        maximum = int(session.config["service"]["max_history"])
        if not 1 <= limit <= maximum:
            raise ValueError(f"limit must be between 1 and {maximum}")
        with session.lock:
            return [item.as_dict() for item in session.plant.history[-limit:]]

    def decisions(self, simulation_id: str) -> list[dict[str, Any]]:
        session = self.get(simulation_id)
        with session.lock:
            return list(session.decisions)

    def forecast(self, simulation_id: str, rack_id: int, horizon_min: int) -> dict[str, Any]:
        session = self.get(simulation_id)
        if rack_id not in range(session.plant.n_racks):
            raise ValueError("rack_id out of range")
        maximum = int(session.config["service"]["max_horizon_min"])
        if not 1 <= horizon_min <= maximum:
            raise ValueError(f"horizon_min must be between 1 and {maximum}")
        with session.lock:
            snapshot = session.plant.history[-1]
            history = session.plant.history[-90:]
        steps = int(horizon_min * 60 / session.config["control"]["rollout_step_s"])
        configured_horizons = np.asarray(session.config["forecasting"]["horizons_min"], dtype=int)
        projection_minutes = max(horizon_min, int(configured_horizons.max())) if self.bundle else horizon_min
        projection_steps = int(projection_minutes * 60 / session.config["control"]["rollout_step_s"])
        inputs = {"utilization": np.repeat(snapshot.utilization[None, :], projection_steps, axis=0), "ambient_c": np.full(projection_steps, snapshot.ambient_c)}
        noop = Action.noop(session.plant.n_zones)
        physics = self.twin.simulate_action(snapshot, noop, projection_minutes * 60, inputs, False)
        hybrid = self.twin.simulate_action(snapshot, noop, projection_minutes * 60, inputs, bool(self.bundle))
        future_minutes = np.arange(1, steps + 1)
        if self.bundle:
            radii = np.interp(future_minutes, configured_horizons, self.bundle["interval"].radii, left=self.bundle["interval"].radii[0], right=self.bundle["interval"].radii[-1])
        else:
            radii = np.full(steps, np.nan)
        predicted = hybrid.temperatures_c[:steps, rack_id]
        probability_horizon = None
        if self.bundle:
            direct = np.asarray([[hybrid.temperatures_c[horizon - 1, rack_id] for horizon in configured_horizons]])
            direct_widths = np.asarray([2.0 * self.bundle["interval"].radii])
            calibrated = self.bundle["hotspot"].predict_proba(direct, direct_widths)[0]
            probability_index = int(np.searchsorted(configured_horizons, horizon_min, side="left"))
            probability_index = min(probability_index, len(configured_horizons) - 1)
            hotspot_probability = float(calibrated[probability_index])
            probability_horizon = int(configured_horizons[probability_index])
        else:
            hotspot_probability = None
        return {
            "rack_id": rack_id, "forecast_origin_s": snapshot.time_s, "horizon_min": horizon_min,
            "history": [{"time_s": item.time_s, "temperature_c": float(item.estimated_temperatures_c[rack_id])} for item in history],
            "future": [{
                "time_s": float(physics.times_s[index]), "physics_c": float(physics.temperatures_c[index, rack_id]),
                "hybrid_c": float(predicted[index]),
                "lower_c": float(predicted[index] - radii[index]) if np.isfinite(radii[index]) else None,
                "upper_c": float(predicted[index] + radii[index]) if np.isfinite(radii[index]) else None,
            } for index in range(steps)],
            "interval_level": self.bundle["interval"].level if self.bundle else None,
            "hotspot_probability": hotspot_probability,
            "hotspot_probability_horizon_min": probability_horizon,
            "model_status": "trained hybrid residual" if self.bundle else "causal nominal fallback; run preparation for ML",
        }

    def shutdown(self) -> None:
        with self.lock:
            sessions = list(self.sessions.values())
        for session in sessions:
            session.stop.set()
            session.wake.set()

    # ── Simulation Mode ──────────────────────────────────────────────────────

    def perturb(
        self,
        simulation_id: str,
        rack_temperatures: dict[int, float] | None = None,
        rack_loads: dict[int, float] | None = None,
        random_racks: int | None = None,
    ) -> dict[str, Any]:
        """Branch from the current snapshot and apply manual perturbations.

        Activates simulation mode if not already active. The background run continues
        normally; the shadow plant is a deep copy that receives the overrides.

        Args:
            rack_temperatures: {rack_index: target_temperature_c} overrides.
            rack_loads: {rack_index: offered_utilization [0,1]} overrides.
            random_racks: if provided, randomly pick this many racks and apply
                          a ±3 °C temperature perturbation to each.
        """
        session = self.get(simulation_id)
        with session.lock:
            # Branch from current live snapshot
            baseline = session.plant.history[-1]
            shadow = session.plant.clone()
            shadow.history = list(shadow.history)
            shadow.truth_history = list(shadow.truth_history)

            perturbed_racks: list[int] = []

            # Random rack perturbation
            if random_racks is not None:
                n = int(np.clip(random_racks, 1, shadow.n_racks))
                chosen = list(np.random.default_rng(int(time.monotonic() * 1e6) % (2**31)).choice(
                    shadow.n_racks, size=n, replace=False
                ))
                rack_temperatures = rack_temperatures or {}
                for rack in chosen:
                    rack_temperatures[int(rack)] = float(shadow.hidden.temperatures_c[rack]) + np.random.uniform(-3.0, 3.0)
                perturbed_racks.extend([int(r) for r in chosen])

            # Apply temperature overrides
            if rack_temperatures:
                for rack_idx, temp_c in rack_temperatures.items():
                    rack_idx = int(rack_idx)
                    if 0 <= rack_idx < shadow.n_racks:
                        shadow.hidden.temperatures_c[rack_idx] = float(temp_c)
                        shadow.estimated_c[rack_idx] = float(temp_c)
                        if rack_idx not in perturbed_racks:
                            perturbed_racks.append(rack_idx)

            # Apply load overrides
            if rack_loads:
                offered = shadow.offered_utilization.copy()
                for rack_idx, load in rack_loads.items():
                    rack_idx = int(rack_idx)
                    if 0 <= rack_idx < shadow.n_racks:
                        offered[rack_idx] = float(np.clip(load, 0.0, 1.18))
                        if rack_idx not in perturbed_racks:
                            perturbed_racks.append(rack_idx)
                shadow.offered_utilization = offered

            session.simulation_mode = SimulationMode(
                baseline_snapshot=baseline,
                shadow_plant=shadow,
                perturbed_racks=perturbed_racks,
                activated_at_s=float(shadow.time_s),
            )

            snap = session.simulation_mode.shadow_plant.snapshot(baseline.ambient_c, baseline.humidity_pct)
            return {
                "simulation_mode": True,
                "activated_at_s": session.simulation_mode.activated_at_s,
                "perturbed_racks": perturbed_racks,
                "snapshot": snap.as_dict(),
            }

    def get_candidates(self, simulation_id: str) -> dict[str, Any]:
        """Run the optimizer on the current shadow state and return ranked candidates.

        Returns candidates sorted by feasibility then objective score, with predicted
        temperature, energy_delta_kwh (vs noop baseline), and hotspot_risk_delta.
        """
        session = self.get(simulation_id)
        with session.lock:
            if session.simulation_mode is None:
                raise ValueError("not in simulation mode; call perturb first")
            shadow = session.simulation_mode.shadow_plant
            snap = shadow.snapshot(
                session.simulation_mode.baseline_snapshot.ambient_c,
                session.simulation_mode.baseline_snapshot.humidity_pct,
            )

        candidates = generate_candidates(snap, self.config)
        horizon_s = float(self.config["control"]["horizon_min"]) * 60
        steps = max(1, int(horizon_s / self.config["control"]["rollout_step_s"]))
        forecast_inputs = {
            "utilization": np.repeat(snap.utilization[None, :], steps, axis=0),
            "ambient_c": np.full(steps, snap.ambient_c),
        }
        use_residual = bool(self.bundle)

        # Compute noop baseline for delta calculations
        noop = Action.noop(shadow.n_zones)
        noop_result = self.twin.simulate_action(snap, noop, horizon_s, forecast_inputs, use_residual)
        noop_energy = noop_result.energy_kwh if noop_result.temperatures_c.size else 0.0
        hot_thresh = float(self.config["thresholds"]["hotspot_c"])
        noop_hotspot_risk = float(np.mean(np.maximum(noop_result.temperatures_c - hot_thresh, 0.0) ** 2)) if noop_result.temperatures_c.size else 0.0

        results = []
        for action in candidates:
            rollout = self.twin.simulate_action(snap, action, horizon_s, forecast_inputs, use_residual)
            if not rollout.temperatures_c.size:
                continue
            score, terms = score_rollout(rollout, action, self.config)
            hotspot_risk = float(np.mean(np.maximum(rollout.temperatures_c - hot_thresh, 0.0) ** 2))
            results.append({
                "action_id": action.action_id,
                "label": action.label,
                "feasible": rollout.feasible,
                "objective": round(score, 4),
                "objective_terms": {k: round(v, 5) for k, v in terms.items()},
                "predicted_peak_c": round(rollout.max_temperature_c, 2),
                "predicted_energy_kwh": round(rollout.energy_kwh, 4),
                "energy_delta_kwh": round(rollout.energy_kwh - noop_energy, 4),
                "hotspot_risk_delta": round(hotspot_risk - noop_hotspot_risk, 6),
                "supply_delta_c": list(action.supply_delta_c),
                "airflow_delta": list(action.airflow_delta),
                "migration": action.migration,
            })

        # Sort: feasible first, then by objective score
        results.sort(key=lambda r: (not r["feasible"], r["objective"]))
        return {
            "simulation_mode": True,
            "candidates": results,
            "noop_baseline": {
                "energy_kwh": round(noop_energy, 4),
                "hotspot_risk": round(noop_hotspot_risk, 6),
                "peak_c": round(noop_result.max_temperature_c, 2) if noop_result.temperatures_c.size else None,
            },
        }

    def apply_candidate(self, simulation_id: str, action_id: str) -> dict[str, Any]:
        """Commit a chosen candidate action to the shadow plant and advance one control step.

        The shadow plant plays forward using the current scenario's workload for one
        decision interval, so the user can watch the physics settle.
        """
        session = self.get(simulation_id)
        with session.lock:
            if session.simulation_mode is None:
                raise ValueError("not in simulation mode")
            shadow = session.simulation_mode.shadow_plant
            baseline = session.simulation_mode.baseline_snapshot
            snap = shadow.snapshot(baseline.ambient_c, baseline.humidity_pct)

            # Rebuild the action from the candidate list to avoid trusting user-supplied deltas
            candidates = generate_candidates(snap, self.config)
            action = next((a for a in candidates if a.action_id == action_id), None)
            if action is None:
                # Allow noop by name
                if action_id == "noop":
                    action = Action.noop(shadow.n_zones)
                else:
                    raise ValueError(f"action_id '{action_id}' not found in current candidate set")

            accepted, reason = shadow.apply_action(action)
            if not accepted:
                raise ValueError(f"action not accepted by shadow plant: {reason}")

            # Advance the shadow plant one observation interval using the live scenario inputs
            workload_rng = np.random.default_rng(int(time.monotonic() * 1e6) % (2**31))
            offered, ambient, humidity = scenario_inputs(
                session.step_index, shadow.n_racks, session.scenario, workload_rng
            )
            shadow.advance(offered, ambient, humidity)

            session.simulation_mode.applied_actions.append({
                "action_id": action.action_id,
                "label": action.label,
                "supply_delta_c": list(action.supply_delta_c),
                "airflow_delta": list(action.airflow_delta),
                "migration": action.migration,
                "applied_at_s": float(shadow.time_s),
            })

            new_snap = shadow.history[-1]
            return {
                "simulation_mode": True,
                "applied_action": action.action_id,
                "applied_action_label": action.label,
                "snapshot": new_snap.as_dict(),
                "applied_actions_count": len(session.simulation_mode.applied_actions),
            }

    def get_simulation_mode_state(self, simulation_id: str) -> dict[str, Any]:
        """Return the full simulation mode state: current shadow snapshot and all rack values."""
        session = self.get(simulation_id)
        with session.lock:
            if session.simulation_mode is None:
                return {"simulation_mode": False}
            mode = session.simulation_mode
            shadow = mode.shadow_plant
            snap = shadow.snapshot(mode.baseline_snapshot.ambient_c, mode.baseline_snapshot.humidity_pct)
            return {
                "simulation_mode": True,
                "activated_at_s": mode.activated_at_s,
                "perturbed_racks": mode.perturbed_racks,
                "applied_actions": list(mode.applied_actions),
                "snapshot": snap.as_dict(),
                "baseline_snapshot": mode.baseline_snapshot.as_dict(),
            }

    def end_simulation_mode(self, simulation_id: str) -> dict[str, Any]:
        """End simulation mode and return a Past/Present/Future recap.

        - Past: the baseline snapshot before perturbation.
        - Present: the current shadow snapshot after all applied actions.
        - Future: a 30-minute physics forward projection from the present state.
        Returns a recap dict that routes.py hands to the report generator.
        """
        session = self.get(simulation_id)
        with session.lock:
            if session.simulation_mode is None:
                raise ValueError("not in simulation mode")
            mode = session.simulation_mode
            shadow = mode.shadow_plant
            present_snap = shadow.snapshot(mode.baseline_snapshot.ambient_c, mode.baseline_snapshot.humidity_pct)

            # 30-minute forward projection
            horizon_s = 30 * 60
            steps = int(horizon_s / self.config["control"]["rollout_step_s"])
            forecast_inputs = {
                "utilization": np.repeat(present_snap.utilization[None, :], steps, axis=0),
                "ambient_c": np.full(steps, present_snap.ambient_c),
            }
            noop = Action.noop(shadow.n_zones)
            future_rollout = self.twin.simulate_action(present_snap, noop, horizon_s, forecast_inputs, bool(self.bundle))

            recap = {
                "simulation_id": simulation_id,
                "past": {
                    "label": "Before perturbation",
                    "snapshot": mode.baseline_snapshot.as_dict(),
                    "peak_c": float(np.max(mode.baseline_snapshot.estimated_temperatures_c)),
                },
                "present": {
                    "label": "After simulation actions",
                    "snapshot": present_snap.as_dict(),
                    "peak_c": float(np.max(present_snap.estimated_temperatures_c)),
                    "applied_actions": list(mode.applied_actions),
                    "perturbed_racks": mode.perturbed_racks,
                },
                "future": {
                    "label": "30-minute forward projection",
                    "horizon_min": 30,
                    "peak_c": round(future_rollout.max_temperature_c, 2) if future_rollout.temperatures_c.size else None,
                    "energy_kwh": round(future_rollout.energy_kwh, 4) if future_rollout.temperatures_c.size else None,
                    "hotspot_rack_minutes": round(future_rollout.hotspot_rack_minutes, 2) if future_rollout.temperatures_c.size else None,
                    "temperatures_c": future_rollout.temperatures_c.tolist() if future_rollout.temperatures_c.size else [],
                    "times_s": future_rollout.times_s.tolist() if future_rollout.times_s.size else [],
                },
            }

            # Clear simulation mode after recap
            session.simulation_mode = None

            return recap
