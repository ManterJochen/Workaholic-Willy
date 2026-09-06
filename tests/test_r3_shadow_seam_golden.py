"""R3.0 STEP-0 — the shadow-seam contract test (closes the R1.2 stub-router blind spot).

The R1.2 shadow value-golden (tests/test_r1_2_shadow_value_golden.py) drives BinPickingOrchestrator.run() with a
RECORDING-STUB router (deterministic sentinels) — so it pins the orchestrator-side DERIVATION but NOT the real rl/
policy outputs through the runtime getattr seam. This module closes that gap: it builds the REAL ShadowRouter from
the 5 committed artifacts (via maybe_build_shadow_router), runs the orchestrator with it (the real V3/V4/V5/V6
policies fire + populate the `_vX` slots), and drives attach_shadow_router (the real V2 candidate policy + the
sub-block fold) — asserting the seam STRUCTURE (the 5 policies load, the slots populate, the 4 carrier sub-blocks
fold non-None, the rl_* extras emit).

It is a STRUCTURAL contract test, NOT a value snapshot: the trained-artifact floats are platform-specific (the
committed-SHA tests are WILLY_DETERMINISM_NATIVE-locked to the canonical origin and skip/fail off-origin), so a
value golden would be Windows-locked + brittle. The R3 byte-identical proof for the artifacts is the separate
same-box WILLY_DETERMINISM_NATIVE=1 pre/post regeneration check. This net catches an R3 refactor that breaks the
real-policy-through-seam wiring (load drift, slot-name drift, fold drift, a dropped sub-block).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.geometry import Frame, Pose
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    AutonomousGraspReport,
    GraspMode,
)
from src.robot.execution.autonomous_grasp.config import _profile_for
from src.robot.execution.autonomous_grasp.shadow import (
    attach_shadow_router,
    maybe_build_shadow_router,
)
from src.robot.grasping import GraspFrame, GraspPoint
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, PerceptionFrame

_POLICY_SLOTS = (
    "candidate_policy",
    "ranking_policy",
    "sequencing_policy",
    "perception_policy",
    "recovery_policy",
)


def _real_router() -> object:
    """The REAL ShadowRouter built from the 5 committed baselines (rl.mode='rl_shadow', no path overrides)."""
    return maybe_build_shadow_router(SimpleNamespace(rl=SimpleNamespace(mode="rl_shadow")))


# --------------------------------------------------------------------------- minimal orchestrator harness
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


def _grasp_point() -> GraspPoint:
    return GraspPoint(
        position=np.array([100.0, 50.0, 400.0]), approach=np.array([0.0, 0.0, 1.0]),
        axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE, label="test",
        metadata={"rl_state_features": {"geometric_score": 0.11}, "shadow": {"feasibility_score": 0.33}},
    )


def _success_result() -> GraspResult:
    return GraspResult(candidates=(_grasp_point(),), reasons=(), top_score=0.9)


def _rescan_result() -> GraspResult:
    return GraspResult(candidates=(), reasons=(GraspFailureReason.RESCAN_RECOMMENDED,), top_score=0.0)


def _orch(router: object, results: list, frames: list) -> BinPickingOrchestrator:
    return BinPickingOrchestrator(
        arm=_FakeArm(),  # type: ignore[arg-type]
        calculator=_ScriptedCalculator(results),  # type: ignore[arg-type]
        perception=_FakePerception(frames),  # type: ignore[arg-type]
        shadow_router=router,  # type: ignore[arg-type]
        max_attempts=3,
    )


def _report_for(orch: BinPickingOrchestrator, outcome: AutonomousGraspOutcome) -> AutonomousGraspReport:
    """A minimal AutonomousGraspReport whose pick_report duck-types attach_shadow_router's `attempts` read."""
    return AutonomousGraspReport(
        outcome=outcome,
        mode=GraspMode.AUTO,
        profile=_profile_for(GraspMode.AUTO),
        pick_report=SimpleNamespace(attempts=[object()]),  # type: ignore[arg-type]
        telemetry={"attempt_wall_time_s": 0.0},
    )


class ShadowSeamContractTests(unittest.TestCase):
    """Drives the 5 REAL committed artifacts through the runtime shadow seam (the R1.2 blind spot)."""

    def test_real_router_loads_all_five_committed_policies(self) -> None:
        router = _real_router()
        self.assertIsNotNone(router, "maybe_build_shadow_router returned None for rl.mode='rl_shadow'")
        for slot in _POLICY_SLOTS:
            self.assertIsNotNone(
                getattr(router, slot, None), f"the real router did not load {slot} from its committed baseline"
            )

    def test_success_run_drives_real_v3_v4_v5_through_orchestrator(self) -> None:
        orch = _orch(_real_router(), [_success_result()], [_frame()])
        orch.run()
        # The real V3/V4/V5 policies fire on the success path + populate the getattr slots shadow.py reads.
        self.assertIsNotNone(orch._ranking_telemetry)
        self.assertIsNotNone(orch._sequencing_telemetry)
        self.assertIsNotNone(orch._perception_telemetry)

    def test_multi_fail_run_drives_real_v4_v6_through_orchestrator(self) -> None:
        orch = _orch(_real_router(), [_rescan_result()], [_frame()])
        orch.run()
        # The real V6 recovery policy fires on the terminal failing attempt (+ V4 sequencing per-failure).
        self.assertIsNotNone(orch._sequencing_telemetry)
        self.assertIsNotNone(orch._recovery_telemetry)

    def test_attach_folds_real_sub_blocks_and_drives_v2(self) -> None:
        # Success: attach drives the REAL V2 candidate policy (run_candidate_selection) + folds V3/V4/V5.
        orch = _orch(_real_router(), [_success_result()], [_frame()])
        orch.run()
        report = attach_shadow_router(
            shadow_router=orch.shadow_router,
            runtime=SimpleNamespace(orchestrator=orch),
            report=_report_for(orch, AutonomousGraspOutcome.SUCCEEDED),
        )
        carrier = report.shadow_router_telemetry
        self.assertIsNotNone(carrier, "attach_shadow_router did not produce a ShadowRouterTelemetry carrier")
        self.assertIsNotNone(carrier.ranking_telemetry, "V3 ranking sub-block was not folded")
        self.assertIsNotNone(carrier.sequencing_telemetry, "V4 sequencing sub-block was not folded")
        self.assertIsNotNone(carrier.perception_telemetry, "V5 perception sub-block was not folded")
        # The real V2 candidate policy ran (its rl_* extras landed on report.telemetry).
        rl_keys = [k for k in report.telemetry if str(k).startswith("rl_")]
        self.assertTrue(rl_keys, "no rl_* shadow extras emitted from the real V2 candidate policy")

    def test_attach_folds_real_recovery_sub_block(self) -> None:
        orch = _orch(_real_router(), [_rescan_result()], [_frame()])
        orch.run()
        report = attach_shadow_router(
            shadow_router=orch.shadow_router,
            runtime=SimpleNamespace(orchestrator=orch),
            report=_report_for(orch, AutonomousGraspOutcome.NO_VALID_GRASP),
        )
        self.assertIsNotNone(report.shadow_router_telemetry)
        self.assertIsNotNone(
            report.shadow_router_telemetry.recovery_telemetry, "V6 recovery sub-block was not folded"
        )


if __name__ == "__main__":
    unittest.main()
