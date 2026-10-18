"""Tests for the bin-picking orchestrator."""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping import GraspCalculator
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.loop.pick_loop import (
    BinPickingOrchestrator,
    PerceptionFrame,
    PickOutcome,
)
from tests._helpers import (
    _FakeArm,
    _FakePerception,
    _ScriptedCalculator,
    _perception_frame,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _grasp_point(score: float = 0.9) -> GraspPoint:
    return GraspPoint(
        position=np.array([100.0, 50.0, 400.0]),
        approach=np.array([0.0, 0.0, 1.0]),
        axis=np.array([1.0, 0.0, 0.0]),
        grip_width_mm=40.0,
        score=score,
        frame=GraspFrame.BASE,
        label="test",
    )


def _success_result() -> GraspResult:
    return GraspResult(
        candidates=(_grasp_point(),),
        reasons=(),
        top_score=0.9,
    )


def _rescan_result() -> GraspResult:
    return GraspResult(
        candidates=(),
        reasons=(GraspFailureReason.RESCAN_RECOMMENDED,),
        top_score=0.0,
    )


def _active_perception_result() -> GraspResult:
    return GraspResult(
        candidates=(),
        reasons=(GraspFailureReason.ACTIVE_PERCEPTION_RECOMMENDED,),
        top_score=0.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class BinPickingOrchestratorTests(unittest.TestCase):
    def test_execute_on_first_success(self) -> None:
        arm = _FakeArm()
        calc = _ScriptedCalculator([_success_result()])
        perception = _FakePerception([_perception_frame()])
        orch = BinPickingOrchestrator(
            arm=arm,  # type: ignore[arg-type]
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            max_attempts=3,
        )
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.EXECUTED)
        self.assertEqual(len(report.attempts), 1)
        self.assertEqual(report.attempts[0].action, "executed")
        # 4 approach waypoints + 1 retreat = 5 motion commands.
        self.assertEqual(len(arm.move_to_calls), 5)

    def test_rescan_then_success(self) -> None:
        arm = _FakeArm()
        calc = _ScriptedCalculator([_rescan_result(), _success_result()])
        perception = _FakePerception([_perception_frame()])
        orch = BinPickingOrchestrator(
            arm=arm,  # type: ignore[arg-type]
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            max_attempts=3,
        )
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.EXECUTED)
        self.assertEqual(len(report.attempts), 2)
        self.assertEqual(report.attempts[0].action, "rescan")
        self.assertEqual(report.attempts[1].action, "executed")
        self.assertEqual(perception.calls, 2)

    def test_active_perception_is_rescanned_without_moving_the_camera(self) -> None:
        # The relocate path left with the viewpoint planners (2026-09-29): a recommendation to look
        # from elsewhere takes a fresh frame from where the camera stands, like any rescan reason.
        arm = _FakeArm()
        calc = _ScriptedCalculator([_active_perception_result(), _success_result()])
        perception = _FakePerception([_perception_frame()])
        orch = BinPickingOrchestrator(
            arm=arm,  # type: ignore[arg-type]
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            max_attempts=3,
        )
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.EXECUTED)
        self.assertEqual(report.attempts[0].action, "rescan")
        self.assertEqual(perception.calls, 2)
        # 4 approach waypoints + 1 retreat: no camera move before them.
        self.assertEqual(len(arm.move_to_calls), 5)

    def test_active_perception_on_the_last_attempt_ends_rescanned_exhausted(self) -> None:
        arm = _FakeArm()
        calc = _ScriptedCalculator([_active_perception_result()])
        perception = _FakePerception([_perception_frame()])
        orch = BinPickingOrchestrator(
            arm=arm,  # type: ignore[arg-type]
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            max_attempts=2,
        )
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.RESCANNED_EXHAUSTED)
        self.assertEqual([a.action for a in report.attempts], ["rescan", "rescan"])
        self.assertEqual(len(arm.move_to_calls), 0)

    def test_no_segmentations_yields_no_perception(self) -> None:
        arm = _FakeArm()
        calc = _ScriptedCalculator([_success_result()])
        empty_frame = PerceptionFrame(
            depth_map=np.full((16, 16), 500.0, dtype=np.float64),
            intrinsics=np.eye(3),
            segmentations=(),
        )
        perception = _FakePerception([empty_frame])
        orch = BinPickingOrchestrator(
            arm=arm,  # type: ignore[arg-type]
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.NO_PERCEPTION)
        self.assertEqual(len(arm.move_to_calls), 0)

    def test_real_calculator_smoke(self) -> None:
        """A real GraspCalculator + the orchestrator must not raise."""
        arm = _FakeArm()
        perception = _FakePerception([_perception_frame()])
        orch = BinPickingOrchestrator(
            arm=arm,  # type: ignore[arg-type]
            calculator=GraspCalculator(
                min_grip_width_mm=1.0,
                max_grip_width_mm=200.0,
                max_candidates=3,
            ),
            perception=perception,
            max_attempts=2,
            )
        report = orch.run()
        # We don't care about the outcome here; we care that the
        # plumbing (calculator <-> orchestrator <-> arm) is wired.
        self.assertIn(
            report.outcome,
            {
                PickOutcome.EXECUTED,
                PickOutcome.RESCANNED_EXHAUSTED,
                PickOutcome.NO_PERCEPTION,
            },
        )


class _SpyPolicy:
    """Records which candidate was executed (so the test can assert the fall-back)."""

    def __init__(self) -> None:
        self.calls: list[GraspPoint] = []

    def execute(self, grasp: GraspPoint) -> PolicyReport:
        self.calls.append(grasp)
        return PolicyReport(outcome=PolicyOutcome.EXECUTED)


if __name__ == "__main__":
    unittest.main()
