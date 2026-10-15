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

Since 2026-10-13 the reason is said as the ``ArmFault`` it is: its class, the controller's own text, and what a
person does about it. Willy ends the campaign on it through the report (``controller_stopped``), and a fault a
motion raised reaches the report as its ``fault``, CANCELLED.

The check is inert by construction for anything that is not a real UR: only ``URRobotArm`` implements
``SupportsRobotStatus``, so the sim, the dummy and every fixture below take the untouched path.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np

from src.robot.core import ArmFault, ArmFaultClass, RobotMode, RobotStatus, SafetyMode
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
            emergency: bool = False, message: str = "", program_running: bool = True) -> RobotStatus:
    return RobotStatus(
        robot_mode=mode, safety_mode=safety,
        protective_stopped=protective, emergency_stopped=emergency, message=message,
        program_running=program_running,
    )


class _StatusArm(MagicMock):
    """A MagicMock arm that also satisfies SupportsRobotStatus structurally.

    Both members are defined here on purpose, and from Python 3.12 on that carries weight: an ``isinstance`` against a
    runtime-checkable Protocol looks members up with ``inspect.getattr_static`` there, which does not ask
    ``__getattr__``, so a plain MagicMock satisfies no Protocol. Measured in aurora's Linux 3.12 container
    (2026-09-20): a double that kept the old name of the recovery verb stopped being a ``SupportsRobotStatus`` on 3.12
    alone, and its stopped controller read as an ordinary failed grasp, green on 3.11.
    """

    def get_robot_status(self):  # noqa: ANN201
        return self._status

    def acknowledge_fault(self, fault: object) -> bool:  # pragma: no cover - never called, by design
        raise AssertionError("the pick loop must never offer to clear a stop")


class TheDoubleIsWhatItClaimsTests(unittest.TestCase):
    """The fixture check, because a mock that answers every name cannot fail one.

    On Python 3.11 a MagicMock satisfies every runtime-checkable Protocol; from 3.12 on it satisfies none it has not
    really got. A double that drifts from the contract is green on one version and wrong on the other, so this asserts
    the members are defined, on both.
    """

    def test_the_status_double_defines_the_capability_it_stands_for(self) -> None:
        import inspect

        from src.robot.core import SupportsRobotStatus

        arm = _StatusArm()
        for member in ("get_robot_status", "acknowledge_fault"):
            with self.subTest(member=member):
                self.assertTrue(inspect.getattr_static(arm, member, None),
                                f"the double fabricates {member} instead of defining it")
        self.assertIsInstance(arm, SupportsRobotStatus)


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
        said = _orchestrator(arm)._controller_cannot_move()  # noqa: SLF001 (the diagnosis the pick logs and reports)
        fault = ArmFault.from_status(arm._status)
        assert fault is not None
        self.assertIs(ArmFaultClass.PROTECTIVE_STOP, fault.kind)
        self.assertTrue(fault.acknowledgeable)
        self.assertEqual(fault.render(), said, "the pick says the fault in the fault's own words")
        self.assertIn("Safetystatus: PROTECTIVE_STOP", said)
        self.assertIn("clear the stop where the arm is visible", said)

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
        fault = ArmFault.from_status(arm._status)
        assert fault is not None
        self.assertIs(ArmFaultClass.CONTROLLER_FAULT, fault.kind)
        self.assertFalse(fault.acknowledgeable, "a powered-off controller is not cleared from software")
        self.assertIn("power the controller back on", _orchestrator(arm)._controller_cannot_move() or "")  # noqa: SLF001

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


class TheCampaignStopsOnAFaultTests(unittest.TestCase):
    """A fault a motion raised reaches the report a campaign reads: CANCELLED, with the fault on it."""

    def _service(self) -> object:
        from src.robot.execution.autonomous_grasp.service import AutonomousGraspService

        service = AutonomousGraspService.__new__(AutonomousGraspService)
        service.effective_config = None  # type: ignore[assignment]
        service.default_mode = "easy"  # type: ignore[assignment]
        return service

    def test_the_service_reports_the_fault_and_calls_it_cancelled(self) -> None:
        from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome

        fault = ArmFault(ArmFaultClass.PROTECTIVE_STOP, "Safetystatus: PROTECTIVE_STOP")

        report = self._service()._fault_report(fault, mode="easy")  # type: ignore[attr-defined]  # noqa: SLF001

        self.assertIs(AutonomousGraspOutcome.CANCELLED, report.outcome,
                      "a stopped cell filed under failed grasps is a machine state in the grasp taxonomy")
        self.assertIs(fault, report.fault)
        self.assertTrue(report.controller_stopped, "the campaign stops on a stopped controller")

    def test_a_camera_fault_is_still_an_execution_failure(self) -> None:
        """The control: only a cell that stopped is cancelled; the other faults of the cell read as before."""
        from src.robot.core.errors import CameraWorldUnavailable
        from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome

        blind = CameraWorldUnavailable(camera="overhead", verdict="no frame", attempts=4, reason="silent")

        report = self._service()._fault_report(blind, mode="easy")  # type: ignore[attr-defined]  # noqa: SLF001

        self.assertIs(AutonomousGraspOutcome.EXECUTION_FAILED, report.outcome)
        self.assertIs(blind, report.fault)
        self.assertFalse(report.controller_stopped)


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
