"""A stopped cell is not a failed grasp, and the pick loop now knows the difference.

MEASURED against URSim on 2026-08-18. With a real protective stop active, commanded motions came
back as ``RobotMotionRejected: Driver reported moveJ() failure`` -- which the loop mapped to
``EXECUTION_FAILED``, a verdict indistinguishable from a bad grasp. The driver knew perfectly well:
``get_robot_status()`` reported ``protective_stopped=True`` and ``raise_if_stopped()`` raised. Nothing
in the pick path asked.

Three owner decisions shape what is tested here (2026-08-18):

* the criterion is ``not is_operational``, not ``is_stopped`` -- a controller powered off mid-session
  reported every stop flag as False, correctly, while being exactly as unable to move;
* the question is asked only AFTER a motion has failed, so a healthy pick pays nothing;
* the pick then ABORTS with a typed reason, and the recovery dispatcher maps that reason to no action
  at all. A stopped cell needs a human, not a retry, and least of all a motion.

The check is inert by construction for anything that is not a real UR: only ``URRobotArm`` implements
``SupportsRobotStatus``, so the sim, the dummy and every fixture below take the untouched path.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np

from src.robot.core import RobotMode, RobotStatus, SafetyMode
from src.robot.grasping.loop.pick_loop import (
    BinPickingOrchestrator,
    PickOutcome,
)
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.grasping.recovery.orchestrator import RecoveryDispatcher
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import PerceptionFrame


@dataclass
class _Seg:
    mask: np.ndarray


class _Perception:
    def acquire(self) -> PerceptionFrame:
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        return PerceptionFrame(
            depth_map=np.full((8, 8), 500.0),
            intrinsics=np.array([[500.0, 0, 4.0], [0, 500.0, 4.0], [0, 0, 1.0]]),
            segmentations=(_Seg(mask=mask),),
        )


class _Calculator:
    """Always returns one usable candidate, so the run always reaches the motion."""

    render_debug_images = False

    def compute_result(self, *_a, **_k) -> GraspResult:
        gp = GraspPoint(
            position=np.array([0.0, 0.0, 500.0]),
            approach=np.array([0.0, 0.0, 1.0]),
            axis=np.array([1.0, 0.0, 0.0]),
            grip_width_mm=40.0,
            score=0.9,
            frame=GraspFrame.CAMERA,
        )
        return GraspResult(candidates=(gp,), top_score=0.9)


class _Policy:
    """A policy whose motion always fails -- the state the check is asked about."""

    def execute(self, _grasp) -> PolicyReport:
        return PolicyReport(outcome=PolicyOutcome.MOTION_FAILED)


def _status(*, mode: RobotMode, safety: SafetyMode, protective: bool = False,
            emergency: bool = False, message: str = "") -> RobotStatus:
    return RobotStatus(
        robot_mode=mode, safety_mode=safety,
        protective_stopped=protective, emergency_stopped=emergency, message=message,
    )


class _StatusArm(MagicMock):
    """A MagicMock arm that also satisfies SupportsRobotStatus structurally."""

    def get_robot_status(self):  # noqa: ANN201
        return self._status

    def recover_from_protective_stop(self) -> bool:  # pragma: no cover - never called, by design
        raise AssertionError("the pick loop must never offer to clear a stop")


def _orchestrator(arm) -> BinPickingOrchestrator:
    return BinPickingOrchestrator(
        arm=arm, calculator=_Calculator(), perception=_Perception(),
        policy=_Policy(), max_attempts=1,
    )


def _run(arm) -> tuple[PickOutcome, tuple]:
    report = _orchestrator(arm).run()
    reasons = report.attempts[-1].reasons if report.attempts else ()
    return report.outcome, reasons


class ControllerStateTests(unittest.TestCase):
    def test_a_protective_stop_is_not_reported_as_a_failed_execution(self) -> None:
        arm = _StatusArm()
        arm._status = _status(
            mode=RobotMode.RUNNING, safety=SafetyMode.PROTECTIVE_STOP, protective=True,
            message="Safetystatus: PROTECTIVE_STOP",
        )
        outcome, reasons = _run(arm)
        self.assertIs(outcome, PickOutcome.CONTROLLER_NOT_OPERATIONAL)
        self.assertIn(GraspFailureReason.CONTROLLER_NOT_OPERATIONAL, reasons)

    def test_a_POWERED_OFF_controller_is_caught_too(self) -> None:
        # The case that motivated `not is_operational` over `is_stopped`: measured on URSim, a
        # powered-off controller reports every stop flag as False and is exactly as unable to move.
        arm = _StatusArm()
        arm._status = _status(mode=RobotMode.POWER_OFF, safety=SafetyMode.NORMAL)
        self.assertFalse(arm._status.is_stopped)      # every stop flag says "not stopped" ...
        self.assertFalse(arm._status.is_operational)  # ... and it still cannot move
        outcome, reasons = _run(arm)
        self.assertIs(outcome, PickOutcome.CONTROLLER_NOT_OPERATIONAL)
        self.assertIn(GraspFailureReason.CONTROLLER_NOT_OPERATIONAL, reasons)

    def test_a_healthy_controller_still_reports_a_failed_execution(self) -> None:
        # The motion failed and the cell is fine, so this IS a grasp failure and must stay one.
        arm = _StatusArm()
        arm._status = _status(mode=RobotMode.RUNNING, safety=SafetyMode.NORMAL)
        outcome, reasons = _run(arm)
        self.assertIs(outcome, PickOutcome.EXECUTION_FAILED)
        self.assertNotIn(GraspFailureReason.CONTROLLER_NOT_OPERATIONAL, reasons)

    def test_an_arm_without_the_capability_is_untouched(self) -> None:
        # Only URRobotArm implements SupportsRobotStatus. The sim, the dummy and every fixture must
        # behave exactly as before -- that is what makes this change byte-identical off real hardware.
        plain = MagicMock(spec=["get_tcp_pose", "move_to", "move_joint"])
        outcome, reasons = _run(plain)
        self.assertIs(outcome, PickOutcome.EXECUTION_FAILED)
        self.assertNotIn(GraspFailureReason.CONTROLLER_NOT_OPERATIONAL, reasons)

    def test_a_status_call_that_raises_never_masks_the_real_failure(self) -> None:
        # A diagnosis must not replace the fault it was trying to explain.
        arm = _StatusArm()
        arm.get_robot_status = MagicMock(side_effect=RuntimeError("link down"))
        outcome, _reasons = _run(arm)
        self.assertIs(outcome, PickOutcome.EXECUTION_FAILED)


class NoRecoveryForAStoppedCellTests(unittest.TestCase):
    """The dispatcher must offer nothing at all -- not a rescan, and certainly not a motion."""

    def test_the_reason_maps_to_no_recovery_action(self) -> None:
        actions = RecoveryDispatcher().actions_for(
            GraspFailureReason.CONTROLLER_NOT_OPERATIONAL
        )
        self.assertEqual(actions, ())

    def test_it_is_the_only_sensible_mapping(self) -> None:
        # Stated explicitly because widening this is a safety change: every action in the table either
        # re-perceives (pointless -- the scene is fine) or commands motion (refused, and the last
        # thing to ask of a stopped cell).
        from src.robot.grasping.recovery.policy import SceneRecoveryAction

        actions = RecoveryDispatcher().actions_for(
            GraspFailureReason.CONTROLLER_NOT_OPERATIONAL
        )
        for forbidden in SceneRecoveryAction:
            with self.subTest(action=forbidden):
                self.assertNotIn(forbidden, actions)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
