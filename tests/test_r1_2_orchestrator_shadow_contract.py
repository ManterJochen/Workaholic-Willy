"""R1.2a (core) — the storage-location contract the BinPickingOrchestrator -> ShadowTelemetryAggregator
decomposition must hold byte-identically.

The R1.2 hazard is verified PURE storage-location coupling: the AutonomousGraspService shadow seam reads the
orchestrator's per-pick shadow telemetry via DEFENSIVE getattr WITH A NONE DEFAULT —
``getattr(orchestrator, "_ranking_telemetry", None)`` and likewise ``_candidate_log`` /
``_behavior_candidate_id`` / ``_sequencing_telemetry`` / ``_perception_telemetry`` /
``_recovery_telemetry`` (backend/src/robot/execution/autonomous_grasp/shadow.py ~L310-395). Two more slots
are HARD-read (no default): ``_last_policy_report`` (runtime_pick.py) raises AttributeError if it moves, and
``_pending_camera_to_base`` (run_dense_pick.py debug). If ANY of these slots stops being a declared field on the
orchestrator instance, getattr silently returns None and the shadow telemetry is dropped — a SILENT regression
the PickReport golden is BLIND to (V5/V6 are carrier-only; V3/V4 land on AutonomousGraspReport, not the
PickReport). These tests make that silent-drop STRUCTURALLY IMPOSSIBLE to ship: the slots must remain declared
dataclass fields, written textually inside the orchestrator. (The router-ON value golden that pins the producer-
derived telemetry values is a separate R1.2a follow-up.)
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.robot.grasping import GraspCalculator
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, PerceptionFrame

# The slots the external readers depend on. The six getattr-read (shadow.py) + the two hard-read + the
# cross-cutting loop-state slots the shadow finalizers consume LIVE. None may move off the instance in R1.2.
_GETATTR_READ_SLOTS = (
    "_ranking_telemetry",
    "_candidate_log",
    "_behavior_candidate_id",
    "_sequencing_telemetry",
    "_perception_telemetry",
    "_recovery_telemetry",
)
_HARD_READ_SLOTS = ("_last_policy_report", "_pending_camera_to_base")
_LOOP_STATE_SLOTS = (
    "_sequencing_current_attempts",
    "_sequencing_states",
    "_viewpoints_visited",
    "_commit_reobserve_count",
    "_resolved_sampling_mode",
)
_ALL_SLOTS = _GETATTR_READ_SLOTS + _HARD_READ_SLOTS + _LOOP_STATE_SLOTS


def _camera_matrix() -> np.ndarray:
    return np.array([[200.0, 0.0, 12.0], [0.0, 200.0, 12.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _box_seg(shape: tuple[int, int] = (32, 32)) -> SimpleNamespace:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[10:22, 8:24] = 1
    return SimpleNamespace(mask=mask, label="box")


class _FakeArm:
    def move_to(self, pose: object) -> bool:  # noqa: ANN001
        return True


class _FakePerception:
    def __init__(self, frames: list) -> None:
        self._frames, self._i = frames, 0

    def acquire(self) -> object:
        f = self._frames[min(self._i, len(self._frames) - 1)]
        self._i += 1
        return f


class OrchestratorShadowSlotContractTests(unittest.TestCase):
    """The per-pick shadow + loop-state slots must remain BinPickingOrchestrator dataclass fields."""

    def test_all_slots_remain_dataclass_fields(self) -> None:
        fields = set(BinPickingOrchestrator.__dataclass_fields__)
        missing = [s for s in _ALL_SLOTS if s not in fields]
        self.assertEqual(
            missing, [],
            "R1.2: these per-pick slots MUST stay BinPickingOrchestrator dataclass fields — shadow.py reads the "
            "six getattr slots via getattr(orch, slot, None) and runtime_pick/run_dense_pick hard-read two more; "
            f"moving any off the instance silently drops shadow telemetry. Missing: {missing}",
        )

    def test_slot_field_declarations_present_in_source(self) -> None:
        # grep-guard: belt-and-suspenders for the __dataclass_fields__ check (catches a slot demoted to a
        # runtime-set attribute that would still appear absent here but for a different reason).
        src = (Path(__file__).resolve().parents[1]
               / "src" / "robot" / "grasping" / "loop" / "pick_loop.py").read_text(encoding="utf-8")
        for slot in _GETATTR_READ_SLOTS + _HARD_READ_SLOTS:
            self.assertIn(
                f"{slot}:", src,
                f"R1.2: the field declaration `{slot}: ... = field(init=False)` must remain in pick_loop.py",
            )


class OrchestratorNoRouterNoopTests(unittest.TestCase):
    """The no-shadow-router path must leave all six getattr-read slots None after a real pick (byte-identical
    no-op the R1.2e V3 extraction must preserve)."""

    def test_no_router_pick_leaves_shadow_slots_none(self) -> None:
        seg = _box_seg()
        depth = np.full((32, 32), 1000.0, dtype=np.float64)
        frame = PerceptionFrame(depth_map=depth, intrinsics=_camera_matrix(), segmentations=(seg,))
        orch = BinPickingOrchestrator(
            arm=_FakeArm(),  # type: ignore[arg-type]
            calculator=GraspCalculator(min_grip_width_mm=1.0, max_grip_width_mm=200.0,
                                       max_candidates=3, camera_matrix=_camera_matrix()),
            perception=_FakePerception([frame]),  # type: ignore[arg-type]
        )
        report = orch.run()
        self.assertIsNotNone(report)
        for slot in _GETATTR_READ_SLOTS:
            self.assertIsNone(
                getattr(orch, slot), f"no-router path must leave {slot} None (byte-identical no-op)",
            )


if __name__ == "__main__":
    unittest.main()
