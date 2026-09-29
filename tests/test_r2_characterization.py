"""R2.0a — service-level characterization golden for AutonomousGraspService.pick (the R2 decomposition safety net).

The R2 keystone decomposes the execution composition root's god-methods (_execute_pick_body / _pick_with_decision
/ _pick_with_refinement, ~1000 lines). The R1.0/R1.2 goldens pin only the calculator + the orchestrator, NOT
service.pick / the AutonomousGraspReport — so this module snapshots the FULL report across the pick-path outcomes,
built fully off-box via AutonomousGraspService.from_components + self-contained test doubles (mirroring
tests/test_autonomous_grasp_service.py). Any drift in the report (outcome / profile / telemetry / pick_report /
decision / recovery) during R2.1 fails here with a readable diff.

This file (R2.0a) covers the LEGACY pick path (_execute_pick_body, now _run_legacy_attempt) + the mode-routing +
the report contract across EASY / AUTO / DENSE_CLUTTER. The WIRED decision scenarios (_pick_with_decision) + the
orchestrator-state side-effect harness land in R2.0b (the report-only golden is BLIND to orch._last_policy_report /
perception.calls — the R1.2 hazard recurring).

The two-scan refinement and the two modes that ran it left on 2026-09-29, and their four scenarios with them
(``dense_autonomous_unwired``, ``refine_two_scan``, ``refine_hold``, ``closed_loop_s4_gate``), as did the profile's
``refinement_enabled`` leaf. The profile's ``verification_enabled`` leaf left the same day with the post-grasp
verification stage, and ``next_viewpoint`` left the ``auto`` and ``dense_clutter`` allow-lists, merged into
``rescan`` (cleanup phase 4); ``dense_clutter`` keeps ``nudge_target``. ``next_target`` ("rescan, skipping the
failed part", same label only) joined both allow-lists the same day (owner's recovery decisions), which is the one
leaf that moved in each scenario's ``profile``.

Regenerate INTENTIONALLY (approved change only): WILLY_R2_UPDATE_GOLDEN=1 python -m pytest tests/test_r2_characterization.py
"""

from __future__ import annotations

import json
import os
import unittest
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import MotionCommand, MotionResult, RobotCapabilities
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspService,
    EffectiveGraspingConfig,
    EffectiveUncertaintyConfig,
    EffectiveWatchdogConfig,
    GraspMode,
)
from src.robot.execution.autonomous_grasp.watchdog import WatchdogSample
from src.robot.grasping.decision import DecisionEngine, DecisionPolicy
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import PerceptionFrame

_GOLDEN = Path(__file__).parent / "data" / "r2_characterization" / "service_pick_golden.json"

# Telemetry keys whose VALUE is non-deterministic (wall-clock / per-run ids); the golden pins their PRESENCE but
# normalizes the value so the safety net never goes flaky-red. Matched by exact name OR by substring suffix.
_VOLATILE_EXACT = frozenset({"attempt_id", "timestamp_utc", "timestamp_ns"})
_VOLATILE_SUBSTR = (
    "_ms", "_ns", "_time_s", "_seconds", "_sec", "latency", "runtime", "wall", "elapsed",
    "timestamp", "duration",
)


def _is_volatile(key: str) -> bool:
    k = str(key)
    return k in _VOLATILE_EXACT or any(s in k for s in _VOLATILE_SUBSTR)


def _jsonable(v: object) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return round(v, 5)
    if isinstance(v, (int, str)) or v is None:
        return v
    if isinstance(v, Enum):
        return v.value
    if is_dataclass(v) and not isinstance(v, type):
        return _jsonable(asdict(v))
    if isinstance(v, dict):
        return {
            str(k): ("<volatile>" if _is_volatile(k) else _jsonable(val))
            for k, val in sorted(v.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return f"<{type(v).__name__}>"


def _snap_report(report: object) -> dict:
    """Snapshot the observable AutonomousGraspReport surface (volatile values normalized)."""
    pr = getattr(report, "pick_report", None)
    profile = getattr(report, "profile", None)
    return {
        "outcome": getattr(getattr(report, "outcome", None), "value", None),
        "mode": getattr(getattr(report, "mode", None), "value", None),
        "profile": {
            "sampling_mode": getattr(getattr(profile, "sampling_mode", None), "value", None),
            "recovery_allowed_actions": list(getattr(profile, "recovery_allowed_actions", ()) or ()),
        },
        "telemetry": _jsonable(dict(getattr(report, "telemetry", {}) or {})),
        "pick_report": None if pr is None else {
            "outcome": getattr(getattr(pr, "outcome", None), "value", str(getattr(pr, "outcome", None))),
            "n_attempts": len(getattr(pr, "attempts", ()) or ()),
            "attempt_actions": [getattr(a, "action", None) for a in (getattr(pr, "attempts", ()) or ())],
        },
        "decision": _jsonable(getattr(report, "decision", None)),
        "uncertainty": _jsonable(getattr(report, "uncertainty", None)),
        "recovery_actions": _jsonable(getattr(report, "recovery_actions", ()) or ()),
        "carriers_set": {
            name: getattr(report, name, None) is not None
            for name in (
                "shadow_success_telemetry", "ranking_blend_telemetry",
                "uncertainty_rerank_telemetry", "shadow_router_telemetry",
            )
        },
    }


# --------------------------------------------------------------------------- test doubles (self-contained)
_REAL_UR_CAPS = RobotCapabilities(
    vendor="ur", model="ur5e", dof=6, supports_joint_move=True, supports_linear_move=True,
    supports_async_move=False, has_native_fk=True, has_native_ik=True, has_force_control=False,
    is_simulated=False,
)


class _TypedFakeArm:
    def __init__(self) -> None:
        self._tcp = Pose(position_mm=np.array([0.0, 0.0, 500.0]),
                         quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), frame=Frame.BASE, label="home")

    @property
    def capabilities(self) -> RobotCapabilities:
        return _REAL_UR_CAPS

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move(self, pose: Pose, **_: object) -> MotionResult:
        if pose.frame == Frame.BASE:
            self._tcp = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


class _FakePerception:
    def __init__(self, frames: list) -> None:
        self._frames, self.calls = frames, 0

    def acquire(self) -> object:
        f = self._frames[min(self.calls, len(self._frames) - 1)]
        self.calls += 1
        return f


class _ScriptedCalculator:
    def __init__(self, results: list) -> None:
        self._results, self.calls = results, 0

    def compute_result(self, *_a: object, **_k: object) -> GraspResult:
        r = self._results[min(self.calls, len(self._results) - 1)]
        self.calls += 1
        return r


def _segmentation() -> SimpleNamespace:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:22] = 1
    return SimpleNamespace(mask=mask)


def _frame(with_segs: bool = True) -> PerceptionFrame:
    return PerceptionFrame(
        depth_map=np.full((32, 32), 500.0, dtype=np.float64),
        intrinsics=np.array([[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        segmentations=(_segmentation(),) if with_segs else (),
    )


def _success_result() -> GraspResult:
    return GraspResult(
        candidates=(GraspPoint(position=np.array([100.0, 50.0, 400.0]), approach=np.array([0.0, 0.0, 1.0]),
                               axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=0.9,
                               frame=GraspFrame.BASE, label="test"),),
        reasons=(), top_score=0.9,
    )


def _rescan_result() -> GraspResult:
    return GraspResult(candidates=(), reasons=(GraspFailureReason.RESCAN_RECOMMENDED,), top_score=0.0)


def _svc(mode: GraspMode, *, results: list, frames: list) -> AutonomousGraspService:
    return AutonomousGraspService.from_components(
        arm=_TypedFakeArm(),  # type: ignore[arg-type]
        calculator=_ScriptedCalculator(results),  # type: ignore[arg-type]
        perception=_FakePerception(frames),
        mode=mode,
    )


# --------------------------------------------------------------------------- R2.0b: orch-state harness
def _orch_state(svc: AutonomousGraspService) -> dict:
    """The orchestrator-side side-effects the report-only golden is BLIND to (the R1.2 hazard):
    perception.calls, whether _last_policy_report was set, and watchdog_history differ per path. R2.1 must keep
    these byte-identical."""
    orch = svc.runtime.orchestrator
    return {
        "perception_calls": getattr(getattr(orch, "perception", None), "calls", None),
        "last_policy_report_set": getattr(orch, "_last_policy_report", None) is not None,
        "watchdog_history_len": len(getattr(svc, "watchdog_history", ()) or ()),
    }


def _run(svc: AutonomousGraspService, **pick_kw: object) -> dict:
    report = svc.pick(**pick_kw)
    return {**_snap_report(report), "orch_state": _orch_state(svc)}


# --------------------------------------------------------------------------- R2.0b: decision-path doubles
def _base_effective_config(**overrides: object) -> EffectiveGraspingConfig:
    return EffectiveGraspingConfig(
        default_mode=GraspMode.AUTO,
        max_attempts=1,
        **overrides,  # type: ignore[arg-type]
    )


def _decision_svc(
    *,
    effective_config: object = None,
    watchdog_history: object = None,
) -> AutonomousGraspService:
    """A from_components service routed to _pick_with_decision (decision_engine wired). An optional
    effective_config activates the watchdog/uncertainty stages (from_components leaves it None -> INERT)."""
    svc = AutonomousGraspService.from_components(
        arm=_TypedFakeArm(),  # type: ignore[arg-type]
        calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
        perception=_FakePerception([_frame()]),
        mode=GraspMode.AUTO,
    )
    engine = DecisionEngine(policy=DecisionPolicy())
    svc.decision_engine = engine
    svc.decision_policy = engine.policy
    if effective_config is not None:
        svc.effective_config = effective_config  # type: ignore[assignment]
    if watchdog_history is not None:
        svc.watchdog_history = watchdog_history  # type: ignore[assignment]
    return svc


# --------------------------------------------------------------------------- scenarios
def _scenarios() -> dict:
    out: dict[str, dict] = {}
    # Legacy _execute_pick_body path + the mode routing (R2.0a).
    out["easy_success"] = _run(_svc(GraspMode.EASY, results=[_success_result()], frames=[_frame()]), mode=GraspMode.EASY)
    out["easy_no_target"] = _run(_svc(GraspMode.EASY, results=[_success_result()], frames=[_frame(with_segs=False)]), mode=GraspMode.EASY)
    out["easy_rescan_exhausted"] = _run(_svc(GraspMode.EASY, results=[_rescan_result()], frames=[_frame()]), mode=GraspMode.EASY)
    out["auto_success"] = _run(_svc(GraspMode.AUTO, results=[_success_result()], frames=[_frame()]))
    out["dense_clutter_success"] = _run(_svc(GraspMode.DENSE_CLUTTER, results=[_success_result()], frames=[_frame()]))
    # R2.0b (pre-STEP-B): the wired _pick_with_decision path — pins _decide_once + _rank_grasp_for_decision +
    # _handle_watchdog_pretick (BLOCK_AUTO) + _resolve_uncertainty_runtime/_build_uncertainty_snapshot_for_tick.
    out["decision_grasp_now"] = _run(_decision_svc())
    out["decision_block_auto"] = _run(_decision_svc(
        effective_config=_base_effective_config(
            watchdog=EffectiveWatchdogConfig(mode="active", block_modes=("auto",)),
        ),
        watchdog_history=[WatchdogSample(predicted_observed_calibration_delta_mm=15.0, ood_score=0.05)],
    ))
    out["decision_uncertainty_active"] = _run(_decision_svc(
        effective_config=_base_effective_config(uncertainty=EffectiveUncertaintyConfig(enabled=True)),
    ))
    return out


class ServicePickCharacterizationTests(unittest.TestCase):
    """Pins AutonomousGraspReport across the legacy pick path + the mode routing; R2.1 must reproduce it."""

    def test_matches_golden(self) -> None:
        snaps = _scenarios()
        if os.environ.get("WILLY_R2_UPDATE_GOLDEN"):
            _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            _GOLDEN.write_text(json.dumps(snaps, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.skipTest("golden regenerated (WILLY_R2_UPDATE_GOLDEN)")
        self.assertTrue(_GOLDEN.exists(), f"missing golden {_GOLDEN}; regenerate with WILLY_R2_UPDATE_GOLDEN=1")
        golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
        for scene in snaps:
            self.assertEqual(snaps[scene], golden.get(scene), f"AutonomousGraspReport drifted in scene '{scene}'")


if __name__ == "__main__":
    unittest.main()
