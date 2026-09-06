"""R1.2a value-golden — pins the V3/V4/V5/V6 producer-DERIVED telemetry inputs that R1.2b-e move into the
`_ShadowTelemetryAggregator`.

The orchestrator's PickReport golden is BLIND to shadow telemetry (V5/V6 are carrier-only; V3/V4 land on the
AutonomousGraspReport). So this module drives `BinPickingOrchestrator.run()` with a RECORDING-STUB ShadowRouter
that captures the exact kwargs each producer/finalizer derives + passes to `run_*_shadow` (the orchestrator-side
derivation logic — precisely what moves to the aggregator), across three scenes:
  * success         — V3 ranking shadow fires; V4/V5 finalize; V6 skips the success path.
  * multi_fail      — V3 never fires (no success path); V4 derives per-failure sequencing states; V6 recovery
                      fires on the terminal failing attempt.
  * no_perception   — NO_PERCEPTION early-return; the 3 finalizers STILL fire via run()'s finally (guard-all).
plus the V3 direct-call path (`orch._maybe_run_ranking_shadow(...)`, mirroring tests/test_pick_loop.py:390,
the highest-coupling R1.2e seam). The recording stub returns deterministic sentinels (the router's real policy
math lives in rl/ and is NOT part of R1.2), so the golden pins the DERIVATION, not the policy artifacts.

Regenerate INTENTIONALLY (approved change only): WILLY_R1_UPDATE_GOLDEN=1 python -m pytest tests/test_r1_2_shadow_value_golden.py
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
from src.robot.grasping import GraspFrame, GraspPoint
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, PerceptionFrame

_GOLDEN = Path(__file__).parent / "data" / "r1_characterization" / "shadow_value_golden.json"

_GETATTR_SLOTS = (
    "_ranking_telemetry", "_candidate_log", "_behavior_candidate_id",
    "_sequencing_telemetry", "_perception_telemetry", "_recovery_telemetry",
)


# --------------------------------------------------------------------------- snapshot coercion
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
        return {str(k): _jsonable(val) for k, val in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return f"<{type(v).__name__}>"  # the recording-stub telemetry sentinels


# --------------------------------------------------------------------------- recording-stub router
_SENTINELS = {"ranking": "<ranking>", "sequencing": "<sequencing>", "perception": "<perception>", "recovery": "<recovery>"}


class _RecordingRouter:
    """A ShadowRouter stand-in: truthy policy slots (so the producer guards pass) + run_*_shadow methods that
    RECORD the producer-derived kwargs and return a deterministic sentinel."""

    ranking_policy = object()
    sequencing_policy = object()
    perception_policy = object()
    recovery_policy = object()

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run_ranking_shadow(self, **kw: object) -> str:
        self.calls.append(("ranking", dict(kw)))
        return _SENTINELS["ranking"]

    def run_sequencing_shadow(self, **kw: object) -> str:
        self.calls.append(("sequencing", dict(kw)))
        return _SENTINELS["sequencing"]

    def run_perception_shadow(self, **kw: object) -> str:
        self.calls.append(("perception", dict(kw)))
        return _SENTINELS["perception"]

    def run_recovery_shadow(self, **kw: object) -> str:
        self.calls.append(("recovery", dict(kw)))
        return _SENTINELS["recovery"]


# --------------------------------------------------------------------------- fixtures
def _intrinsics() -> np.ndarray:
    return np.array([[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _segmentation() -> SimpleNamespace:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:22] = 1
    return SimpleNamespace(mask=mask)


def _frame(with_segs: bool = True) -> PerceptionFrame:
    return PerceptionFrame(
        depth_map=np.full((32, 32), 500.0, dtype=np.float64),
        intrinsics=_intrinsics(),
        segmentations=(_segmentation(),) if with_segs else (),
    )


def _grasp_point_with_md() -> GraspPoint:
    # Overlapping metadata blocks exercise the V3 feature-precedence (rl_state_features > shadow > top-level
    # > 0.0): geometric_score appears in all three (rl wins), feasibility_score only in shadow, a top-level key.
    return GraspPoint(
        position=np.array([100.0, 50.0, 400.0]),
        approach=np.array([0.0, 0.0, 1.0]),
        axis=np.array([1.0, 0.0, 0.0]),
        grip_width_mm=40.0,
        score=0.9,
        frame=GraspFrame.BASE,
        label="test",
        metadata={
            "rl_state_features": {"geometric_score": 0.11},
            "shadow": {"geometric_score": 0.22, "feasibility_score": 0.33},
            "geometric_score": 0.44,
        },
    )


def _success_result() -> GraspResult:
    return GraspResult(candidates=(_grasp_point_with_md(),), reasons=(), top_score=0.9)


def _rescan_result() -> GraspResult:
    return GraspResult(candidates=(), reasons=(GraspFailureReason.RESCAN_RECOMMENDED,), top_score=0.0)


class _FakeArm:
    def __init__(self) -> None:
        self._tcp = Pose(position_mm=np.array([0.0, 0.0, 500.0]),
                         quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), frame=Frame.BASE, label="home")

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move_to(self, pose: Pose, **_: object) -> None:
        if pose.frame == Frame.BASE:
            self._tcp = pose


class _FakePerception:
    def __init__(self, frames: list) -> None:
        self._frames, self._i = frames, 0

    def acquire(self) -> object:
        f = self._frames[min(self._i, len(self._frames) - 1)]
        self._i += 1
        return f


class _ScriptedCalculator:
    def __init__(self, results: list) -> None:
        self._results, self._i = results, 0

    def compute_result(self, *_a: object, **_k: object) -> GraspResult:
        r = self._results[min(self._i, len(self._results) - 1)]
        self._i += 1
        return r


def _orch(router: _RecordingRouter, calc: _ScriptedCalculator, frames: list, max_attempts: int) -> BinPickingOrchestrator:
    return BinPickingOrchestrator(
        arm=_FakeArm(),  # type: ignore[arg-type]
        calculator=calc,  # type: ignore[arg-type]
        perception=_FakePerception(frames),  # type: ignore[arg-type]
        shadow_router=router,  # type: ignore[arg-type]
        max_attempts=max_attempts,
    )


def _snap_after_run(orch: BinPickingOrchestrator, router: _RecordingRouter, report: object) -> dict:
    return {
        "outcome": getattr(getattr(report, "outcome", None), "value", None),
        "router_calls": [{"phase": p, "kw": _jsonable(kw)} for p, kw in router.calls],
        "slots_set": {s: getattr(orch, s) is not None for s in _GETATTR_SLOTS},
        "candidate_log": _jsonable(orch._candidate_log),
        "behavior_candidate_id": orch._behavior_candidate_id,
        "v4_sequencing_states": _jsonable(orch._sequencing_states),
    }


# --------------------------------------------------------------------------- scenes
def _scene_success() -> dict:
    r = _RecordingRouter()
    orch = _orch(r, _ScriptedCalculator([_success_result()]), [_frame()], max_attempts=3)
    report = orch.run()
    return _snap_after_run(orch, r, report)


def _scene_multi_fail() -> dict:
    r = _RecordingRouter()
    orch = _orch(r, _ScriptedCalculator([_rescan_result()]), [_frame()], max_attempts=3)
    report = orch.run()
    return _snap_after_run(orch, r, report)


def _scene_no_perception() -> dict:
    r = _RecordingRouter()
    orch = _orch(r, _ScriptedCalculator([_success_result()]), [_frame(with_segs=False)], max_attempts=3)
    report = orch.run()
    return _snap_after_run(orch, r, report)


def _scene_v3_direct_call() -> dict:
    r = _RecordingRouter()
    orch = _orch(r, _ScriptedCalculator([_success_result()]), [_frame()], max_attempts=3)
    cands = (_grasp_point_with_md(),)
    orch._maybe_run_ranking_shadow(pre_blend_candidates=cands, post_blend_candidates=cands, attempt_index=0)
    return {
        "router_calls": [{"phase": p, "kw": _jsonable(kw)} for p, kw in r.calls],
        "candidate_log": _jsonable(orch._candidate_log),
        "behavior_candidate_id": orch._behavior_candidate_id,
        "v3_set": orch._ranking_telemetry is not None,
    }


def _all_snapshots() -> dict:
    return {
        "success": _scene_success(),
        "multi_fail": _scene_multi_fail(),
        "no_perception": _scene_no_perception(),
        "v3_direct_call": _scene_v3_direct_call(),
    }


class ShadowValueGoldenTests(unittest.TestCase):
    """Pins the producer-derived shadow telemetry inputs across the scenes; R1.2b-e must reproduce them."""

    def test_matches_golden(self) -> None:
        snaps = _all_snapshots()
        if os.environ.get("WILLY_R1_UPDATE_GOLDEN"):
            _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            _GOLDEN.write_text(json.dumps(snaps, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.skipTest("golden regenerated (WILLY_R1_UPDATE_GOLDEN)")
        self.assertTrue(_GOLDEN.exists(), f"missing golden {_GOLDEN}; regenerate with WILLY_R1_UPDATE_GOLDEN=1")
        golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
        for scene in ("success", "multi_fail", "no_perception", "v3_direct_call"):
            self.assertEqual(snaps[scene], golden[scene], f"shadow telemetry drifted in scene '{scene}'")


if __name__ == "__main__":
    unittest.main()
