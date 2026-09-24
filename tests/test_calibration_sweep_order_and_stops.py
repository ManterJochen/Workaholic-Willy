"""A sweep stops when the arm cannot be trusted to be where it was judged, and goes on where nothing moved.

Calibration chain, part 3 (owner, 2026-09-23), the robustness items of the analysis:

* any move that did not execute was followed by the next pose, a connection error and a protective stop midway
  through a ``moveJ`` included, so the sweep tried and refused every remaining pose and reported too few samples;
* a ``False`` from ``wait_until_steady`` was ignored and the frame taken anyway.

The sweep-order half of that item went with the generated sweep it ordered (the owner, 2026-09-24, deleted every
automatic station generator): stations run in the order somebody wrote them down or guided the arm to them. The
stop changes when the commanding ends, never how any one move is judged: every station still goes through the arm's
own gates.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.config.schema.robot import WorkspaceLimitsConfig
from src.geometry import Frame, Pose
from src.robot.core import RobotEmergencyStop
from src.robot.core.arm_capabilities import RobotMode, RobotStatus, SafetyMode
from src.robot.core.motion_result import NO_PLAN_FAIL_SAFE_MESSAGE, MotionCommand, MotionResult, MotionStatus
from src.robot.events import RobotCalibrationEvent
from src.robot.execution.calibration import CalibrationRoutine, SweepStopped
from src.robot.execution.hand_eye import CalibrationOutcome, CalibrationRunReport
from tests.test_robot_boundaries import _inverse, _synthetic_eye_to_hand_data

_WIDE = WorkspaceLimitsConfig(x_min=-2000.0, x_max=2000.0, y_min=-2000.0, y_max=2000.0, z_min=-2000.0, z_max=2000.0)


def _at(x: float, y: float, z: float = 300.0, label: str = "") -> Pose:
    return Pose.tool_down(x, y, z, label=label)


# ---------------------------------------------------------------------------------------------------------------------
# When the sweep stops, and when it goes on
# ---------------------------------------------------------------------------------------------------------------------


class _Arm:
    """Arrives where it is sent, except at ``refuse``'s labels, which answer with a status and a message. ``status``
    is what ``get_robot_status`` reports, or raises when it is an exception."""

    def __init__(self, refuse: dict[str, tuple[MotionStatus, str]], *, status: Any = None,
                 steady: list[Any] | None = None) -> None:
        self.refuse = refuse
        self.status = status
        self.steady = list(steady or [])
        self.at: Pose | None = None
        self.moves: list[str] = []
        self.waits: list[float] = []

    def move(self, pose: Pose, **_keywords: Any) -> MotionResult:
        self.moves.append(str(pose.label))
        if pose.label in self.refuse:
            status, message = self.refuse[pose.label]
            return MotionResult.failed(status, MotionCommand.MOVE_TO, target_pose=pose, message=message)
        self.at = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)

    def get_tcp_pose(self) -> Pose:
        assert self.at is not None
        return self.at

    def wait_until_steady(self, timeout_s: float) -> Any:
        self.waits.append(timeout_s)
        return self.steady.pop(0) if self.steady else True


class _StatusArm(_Arm):
    def get_robot_status(self) -> RobotStatus:
        if isinstance(self.status, Exception):
            raise self.status
        return self.status if self.status is not None else _running()


def _running() -> RobotStatus:
    return RobotStatus(RobotMode.RUNNING, SafetyMode.NORMAL, protective_stopped=False, emergency_stopped=False)


def _protective() -> RobotStatus:
    return RobotStatus(RobotMode.RUNNING, SafetyMode.PROTECTIVE_STOP, protective_stopped=True,
                       emergency_stopped=False, message="Protective stop: C153A1")


def _sweep(arm: _Arm, *, settle: float = 0.0, count: int = 7) -> tuple[CalibrationRoutine, list[Pose], list]:
    T_cam_to_base, T_tool_to_marker, tool = _synthetic_eye_to_hand_data()
    poses = [Pose.from_matrix(T, frame=Frame.BASE, label=f"p{i + 1}") for i, T in enumerate(tool[:count])]
    events: list[tuple[str, dict]] = []

    def source() -> np.ndarray:
        return _inverse(T_cam_to_base) @ arm.get_tcp_pose().to_matrix() @ T_tool_to_marker

    routine = CalibrationRoutine(
        arm=arm, marker_source=source, workspace_limits=_WIDE, calibration_mode="eye_to_hand",  # type: ignore[arg-type]
        settle_time_s=settle,
        eth_settings=SimpleNamespace(mode="eye_to_hand", min_samples=4, min_distance_mm=10.0, min_angle=2.0,
                                     min_angle_deg=2.0),  # type: ignore[arg-type]
        on_event=lambda kind, data: events.append((kind, dict(data))))
    return routine, poses, events


_SENT = ("Driver reported moveJ() failure. moveJ was sent and the controller did not complete it, so the arm may "
         "have moved part of the way. The controller reports robot mode running and safety mode protective_stop.")


class TheSweepStopsWhereTheArmCannotBeTrustedTests(unittest.TestCase):

    def _stopped(self, arm: _Arm, **keywords: Any) -> tuple[SweepStopped, CalibrationRoutine, list]:
        routine, poses, events = _sweep(arm)
        with self.assertRaises(SweepStopped) as caught:
            routine.run_with_poses(poses, **keywords)
        return caught.exception, routine, events

    def test_a_controller_refusal_after_the_move_was_sent_stops_the_sweep_there(self) -> None:
        """Red before: the sweep went on to p4..p7, each refused the same way, and reported too few samples."""
        arm = _Arm({"p3": (MotionStatus.CONTROLLER_REJECTED, _SENT)})
        stopped, routine, events = self._stopped(arm)
        self.assertEqual(arm.moves, ["p1", "p2", "p3"])
        self.assertEqual((stopped.index, stopped.total, stopped.label, stopped.status), (3, 7, "p3", "controller_rejected"))
        self.assertTrue(str(stopped).startswith("the sweep stopped at pose 3/7 'p3' with 2 counted, and commanded "
                                                "nothing after it: the controller refused the move after it was sent"),
                        str(stopped))
        last = routine.pose_log[-1]
        self.assertEqual((len(routine.pose_log), last.reason), (3, "move_rejected"))
        self.assertIn("; the sweep stops here: the controller refused the move after it was sent", last.detail)
        (rejected,) = [data for kind, data in events if kind == RobotCalibrationEvent.POSE_REJECTED]
        self.assertEqual(rejected["stops_sweep"], stopped.why)

    def test_a_connection_error_stops_the_sweep(self) -> None:
        arm = _Arm({"p2": (MotionStatus.CONNECTION_ERROR, "cuRobo move requires an open connection.")})
        stopped, _, _ = self._stopped(arm)
        self.assertEqual(arm.moves, ["p1", "p2"])
        self.assertIn("connection_error", stopped.why)

    def test_a_controller_refusal_that_says_nothing_stops_the_sweep(self) -> None:
        """The UR driver's ik path reports a moveJ that returned False as CONTROLLER_REJECTED with no message."""
        arm = _Arm({"p2": (MotionStatus.CONTROLLER_REJECTED, "")})
        stopped, _, _ = self._stopped(arm)
        self.assertEqual(arm.moves, ["p1", "p2"])
        self.assertIn("did not say whether it had been sent", stopped.why)

    def test_a_planner_refusal_after_which_the_controller_reports_a_stop_stops_the_sweep(self) -> None:
        arm = _StatusArm({"p2": (MotionStatus.TIMEOUT, NO_PLAN_FAIL_SAFE_MESSAGE)}, status=_protective())
        stopped, _, _ = self._stopped(arm)
        self.assertEqual(arm.moves, ["p1", "p2"])
        self.assertEqual(stopped.why, "the controller reports it is stopped (safety mode protective_stop: Protective "
                                      "stop: C153A1)")

    def test_an_arm_that_only_raises_when_stopped_is_read_too(self) -> None:
        arm = _Arm({"p2": (MotionStatus.SELF_COLLISION_REJECTED, "the planner refused this joint path")})

        def stopped_state() -> None:
            raise RobotEmergencyStop("UR controller in robot_emergency_stop (robot_mode=running)")

        arm.raise_if_stopped = stopped_state  # type: ignore[attr-defined]
        stopped, _, _ = self._stopped(arm)
        self.assertIn("robot_emergency_stop", stopped.why)

    def test_an_exception_after_which_the_controller_reports_a_stop_stops_the_sweep(self) -> None:
        arm = _StatusArm({}, status=_protective())
        routine, poses, _ = _sweep(arm)
        original = arm.move

        def raising(pose: Pose, **keywords: Any) -> MotionResult:
            if pose.label == "p3":
                arm.moves.append("p3")
                raise RuntimeError("RTDE control script stopped")
            return original(pose, **keywords)

        arm.move = raising  # type: ignore[method-assign]
        with self.assertRaises(SweepStopped) as caught:
            routine.run_with_poses(poses)
        self.assertEqual(arm.moves, ["p1", "p2", "p3"])
        self.assertEqual(routine.pose_log[-1].reason, "exception")
        self.assertIn("RTDE control script stopped", str(caught.exception))

    def test_the_dataset_of_a_stopped_sweep_is_kept(self) -> None:
        arm = _Arm({"p5": (MotionStatus.CONNECTION_ERROR, "link down")})
        routine, poses, _ = _sweep(arm)
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "dataset.json"
        with self.assertRaises(SweepStopped):
            routine.run_with_poses(poses, dataset_save_path=str(path))
        self.assertTrue(path.is_file())

    def test_a_stopped_sweep_reads_as_a_failed_sweep_with_its_poses(self) -> None:
        arm = _Arm({"p3": (MotionStatus.CONTROLLER_REJECTED, _SENT)})
        stopped, routine, _ = self._stopped(arm)
        from tests.test_calibration_sweep_events import _check

        report = CalibrationRunReport(check=_check(poses=7), outcome=CalibrationOutcome.SWEEP_FAILED,
                                      failure=f"{type(stopped).__name__}: {stopped}", pose_log=routine.pose_log)
        self.assertEqual(report.exit_code, 3)
        summary = report.summary()
        self.assertIn("[sweep] FAILED: SweepStopped: the sweep stopped at pose 3/7 'p3'", summary)
        self.assertIn("  per pose          2 of 3 counted", summary)


class TheSweepGoesOnWhereNothingMovedTests(unittest.TestCase):

    def test_planner_and_gate_refusals_skip_their_pose_and_the_sweep_goes_on(self) -> None:
        refusals = {
            "p2": (MotionStatus.TIMEOUT, NO_PLAN_FAIL_SAFE_MESSAGE),
            "p3": (MotionStatus.CONTROLLER_REJECTED, "cuRobo planner unavailable: the planner answered nothing in 30 s"),
            "p4": (MotionStatus.SELF_COLLISION_REJECTED, "the planner refused this joint path: world"),
            "p5": (MotionStatus.CONTROLLER_REJECTED, "the plan ends 812.0 mm and 179.0 deg from its goal, so nothing "
                                                     "moved"),
            "p6": (MotionStatus.INVALID_TARGET, "ur_rtde refused waypoint 1 of 41 before sending it (speed), so that "
                                                "moveJ was not sent and nothing had moved"),
        }
        arm = _StatusArm(refusals)
        routine, poses, _ = _sweep(arm)
        with self.assertRaises(Exception) as caught:  # two poses count, so the solve refuses; the sweep ran to the end
            routine.run_with_poses(poses)
        self.assertNotIsInstance(caught.exception, SweepStopped)
        self.assertEqual(arm.moves, [f"p{i}" for i in range(1, 8)])
        self.assertEqual([v.reason for v in routine.pose_log], ["", *["move_rejected"] * 5, ""])

    def test_a_status_that_cannot_be_read_is_no_stop(self) -> None:
        arm = _StatusArm({"p2": (MotionStatus.WORKSPACE_REJECTED, "outside the box")},
                         status=RuntimeError("dashboard not connected"))
        routine, poses, _ = _sweep(arm)
        result = routine.run_with_poses(poses)
        self.assertEqual(len(arm.moves), 7)
        self.assertEqual(result.num_samples, 6)

    def test_a_mock_arm_is_never_read_as_stopped(self) -> None:
        routine = object.__new__(CalibrationRoutine)
        routine.arm = MagicMock()  # type: ignore[attr-defined]
        routine.logger = MagicMock()  # type: ignore[attr-defined]
        self.assertIsNone(routine._stopped_state())  # noqa: SLF001


# ---------------------------------------------------------------------------------------------------------------------
# wait_until_steady
# ---------------------------------------------------------------------------------------------------------------------


class ASweepWaitsForTheArmToStandStillTests(unittest.TestCase):

    def test_a_first_false_is_waited_once_more_and_the_pose_counts(self) -> None:
        arm = _Arm({}, steady=[False, True])
        routine, poses, _ = _sweep(arm, settle=0.25, count=5)
        result = routine.run_with_poses(poses)
        self.assertEqual(arm.waits, [0.25] * 6)
        self.assertEqual(result.num_samples, 5)

    def test_two_falses_skip_the_capture_as_not_steady(self) -> None:
        """Red before: the False was dropped and the frame taken while the tool still moved."""
        arm = _Arm({}, steady=[True, False, False])
        routine, poses, events = _sweep(arm, settle=0.25, count=6)
        captured: list[str] = []
        source = routine._marker_source  # noqa: SLF001

        def counting() -> Any:
            captured.append(str(arm.at.label if arm.at is not None else ""))
            return source()  # type: ignore[misc]

        routine._marker_source = counting  # noqa: SLF001
        result = routine.run_with_poses(poses)
        self.assertEqual(captured, ["p1", "p3", "p4", "p5", "p6"])
        verdict = result.pose_log[1]
        self.assertEqual((verdict.label, verdict.reason), ("p2", "not_steady"))
        self.assertEqual(verdict.detail, "the arm did not report steady within 0.25 s, twice, after the move, so no "
                                         "frame was taken")
        self.assertIn("not_steady", [data["reason"] for kind, data in events
                                     if kind == RobotCalibrationEvent.POSE_REJECTED])

    def test_a_raise_after_a_false_is_not_steady_and_a_raise_alone_falls_back_to_sleep(self) -> None:
        routine = object.__new__(CalibrationRoutine)
        routine.settle_time_s = 0.001  # type: ignore[attr-defined]
        routine.logger = MagicMock()  # type: ignore[attr-defined]
        answers: list[Any] = [False, RuntimeError("rtde receive lost")]

        def wait(_timeout: float) -> Any:
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        routine.arm = SimpleNamespace(wait_until_steady=wait)  # type: ignore[attr-defined]
        self.assertFalse(routine._wait_for_settle())  # noqa: SLF001
        answers[:] = [RuntimeError("not connected")]
        self.assertTrue(routine._wait_for_settle())  # noqa: SLF001
        routine.arm = SimpleNamespace(wait_until_steady=lambda _timeout: None)  # type: ignore[attr-defined]
        self.assertTrue(routine._wait_for_settle())  # noqa: SLF001


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
