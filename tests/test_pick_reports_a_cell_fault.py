"""A refused pick is a report: a fault of the cell comes back on it, a programmer error still raises (Q11, owner 2026-09-18).

`service.pick()` promised an `AutonomousGraspReport`, and a camera that stopped delivering, a camera that
could not vouch for the cell or a wrist frame taken while the tool moved left it as an exception instead.
`PickRun` caught it; a Python caller writing one pick outside a campaign did not. The report now carries
the fault, and the campaign still stops on it (owner, Step 4 Q1).

Honesty bucket (2): the real service and pick loop, a fake arm and scripted perception. No hardware.

Why each test is red on the code before this step is said in its docstring; the two controls say that
they are green before and after.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import (
    CameraWorldUnavailable,
    MotionCommand,
    MotionResult,
    PerceptionFrameMoved,
    RobotCapabilities,
)
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    AutonomousGraspReport,
    AutonomousGraspService,
    GraspMode,
)
from src.robot.execution.autonomous_grasp.config import _profile_for
from src.robot.execution.pick_run import PickOutcome, PickRun, Recording
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import PerceptionFrame

_CAPS = RobotCapabilities(
    vendor="ur", model="ur5e", dof=6, supports_joint_move=True, supports_linear_move=True,
    supports_async_move=False, has_native_fk=True, has_native_ik=True, has_force_control=False,
    is_simulated=False,
)
_K = np.array([[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float64)
_NO_FRAME = "Frame didn't arrive within 5000"


class _Arm:
    def __init__(self) -> None:
        self._tcp = Pose(position_mm=np.array([0.0, 0.0, 500.0]),
                         quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), frame=Frame.BASE, label="home")

    @property
    def capabilities(self) -> RobotCapabilities:
        return _CAPS

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move(self, pose: Pose, **_: object) -> MotionResult:
        if pose.frame == Frame.BASE:
            self._tcp = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


class _BlindWorldArm(_Arm):
    """An arm whose first motion meets a camera that could not vouch for the cell."""

    def move(self, pose: Pose, **_: object) -> MotionResult:
        raise CameraWorldUnavailable(camera="overhead", verdict="no_frame", attempts=4, reason="no depth")


class _Frames:
    """One segmented box, every time."""

    def acquire(self) -> PerceptionFrame:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:22, 10:22] = 1
        return PerceptionFrame(depth_map=np.full((32, 32), 500.0), intrinsics=_K.copy(),
                               segmentations=(SimpleNamespace(mask=mask),))


class _DeadCamera:
    """A camera that raises ``error`` on every frame, and counts the frames it was asked for."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def acquire(self) -> PerceptionFrame:
        self.calls += 1
        raise self.error


class _Calculator:
    def compute_result(self, *_args: object, **_kwargs: object) -> GraspResult:
        return GraspResult(
            candidates=(GraspPoint(
                position=np.array([100.0, 50.0, 400.0]), approach=np.array([0.0, 0.0, 1.0]),
                axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE,
                label="test",
            ),),
            reasons=(), top_score=0.9,
        )


def _service(*, perception: Any = None, arm: Any = None) -> AutonomousGraspService:
    return AutonomousGraspService.from_components(
        arm=arm or _Arm(),
        calculator=_Calculator(),  # type: ignore[arg-type]
        perception=perception or _Frames(),
        mode=GraspMode.EASY,
    )


def _moved() -> PerceptionFrameMoved:
    return PerceptionFrameMoved(camera="wrist", attempts=3, moved_mm=4.2, turned_deg=0.5,
                                tolerance_mm=1.0, tolerance_deg=0.2)


class AFaultOfTheCellIsAReportTests(unittest.TestCase):

    def test_a_camera_that_stops_delivering_is_a_report(self) -> None:
        """Red before this step: the RuntimeError leaves `pick()`."""
        error = RuntimeError(_NO_FRAME)

        report = _service(perception=_DeadCamera(error)).pick()

        self.assertIs(report.outcome, AutonomousGraspOutcome.EXECUTION_FAILED)
        self.assertIs(report.fault, error)
        self.assertIsNone(report.pick_report, "nothing below the service returned, so there is no trail")
        self.assertIn(f"fault=RuntimeError: {_NO_FRAME}", report.failure_summary())

    def test_a_camera_that_could_not_vouch_is_a_report(self) -> None:
        """The Step 4 fault, raised out of the execution policy on purpose, now ends on the report.

        Red before this step: CameraWorldUnavailable leaves `pick()`.
        """
        report = _service(arm=_BlindWorldArm()).pick()

        self.assertIs(report.outcome, AutonomousGraspOutcome.EXECUTION_FAILED)
        self.assertIsInstance(report.fault, CameraWorldUnavailable)
        self.assertIn("fault=CameraWorldUnavailable", report.failure_summary())

    def test_a_wrist_frame_taken_while_the_tool_moved_is_a_report(self) -> None:
        """Red before this step: PerceptionFrameMoved leaves `pick()`."""
        report = _service(perception=_DeadCamera(_moved())).pick()

        self.assertIsInstance(report.fault, PerceptionFrameMoved)
        self.assertFalse(report.succeeded)

    def test_a_fault_writes_no_record(self) -> None:
        """The corpus counts grasps, and a camera that stopped delivering is not a grasp that missed: the
        raise wrote nothing, and the report writes nothing either.

        Red before this step: `pick()` raises before the assertion is reached.
        """
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "records.jsonl"
            service = _service(perception=_DeadCamera(RuntimeError(_NO_FRAME)))
            service.enable_record_logging(path)

            report = service.pick()

            self.assertIsNotNone(report.fault)
            self.assertFalse(path.exists(), "a fault of the cell was logged as a grasp attempt")

    def test_a_programmer_error_still_raises(self) -> None:
        """A control, green before and after: a TypeError from a wrong double and a NotImplementedError,
        the RuntimeError subclass Python raises for unfinished code, are not faults of a cell."""
        for error in (TypeError("acquire() takes no arguments"), NotImplementedError("later")):
            with self.subTest(type(error).__name__), self.assertRaises(type(error)):
                _service(perception=_DeadCamera(error)).pick()


class TheReportSaysTheFaultTests(unittest.TestCase):

    def _report(self, fault: Exception) -> AutonomousGraspReport:
        return AutonomousGraspReport(
            outcome=AutonomousGraspOutcome.EXECUTION_FAILED, mode=GraspMode.AUTO,
            profile=_profile_for(GraspMode.AUTO), fault=fault,
        )

    def test_both_halves_carry_it(self) -> None:
        """Red before this step: the report has no `fault` field (TypeError)."""
        report = self._report(RuntimeError(_NO_FRAME))

        payload = report.to_dict()

        self.assertEqual(payload["fault"], {"type": "RuntimeError", "message": _NO_FRAME})
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertIn(f"fault=RuntimeError: {_NO_FRAME}", report.render())
        self.assertIsNone(AutonomousGraspReport(
            outcome=AutonomousGraspOutcome.SUCCEEDED, mode=GraspMode.AUTO,
            profile=_profile_for(GraspMode.AUTO)).to_dict()["fault"])

    def test_render_stays_ascii_whatever_the_driver_said(self) -> None:
        """A driver's sentence may carry any character, and render prints to a cp1252 console.

        Red before this step: the report has no `fault` field.
        """
        text = self._report(RuntimeError("Greifer meldet " + chr(0xDC) + "berlast")).render()

        self.assertTrue(text.isascii(), text)
        self.assertFalse(text.endswith("\n"))


class ACampaignStillStopsTests(unittest.TestCase):

    def test_a_reported_fault_stops_the_campaign_in_the_words_of_a_raise(self) -> None:
        """Owner, Step 4 Q1: a campaign stops on a dead camera rather than retrying it as a missed grasp.
        The attempt reads as it did when the fault was raised, and the fault's report is the last one.

        Red before this step: the raise left `last` empty, so `report.last.fault` fails.
        """
        error = RuntimeError(_NO_FRAME)
        camera = _DeadCamera(error)

        report = PickRun.from_service(_service(perception=camera), runs=3, recording=Recording.off()).execute()

        self.assertEqual([a.outcome for a in report.attempts],
                         [PickOutcome.RAISED, PickOutcome.CANCELLED, PickOutcome.CANCELLED])
        self.assertEqual(report.attempts[0].detail, f"RuntimeError: {_NO_FRAME}")
        self.assertEqual(report.exit_code, 3)
        self.assertEqual(camera.calls, 1, "the campaign asked a dead camera again")
        self.assertIs(report.last.fault, error)

    def test_a_double_that_reports_a_fault_stops_the_campaign(self) -> None:
        """The campaign reads the report, not the service type.

        Red before this step: the campaign counted the report as an ordinary failure and went on.
        """
        fault = CameraWorldUnavailable(camera="overhead", verdict="stale", attempts=4, reason="old frames")
        faulted = SimpleNamespace(outcome="execution_failed", fault=fault, failure_summary=lambda: "")
        service = SimpleNamespace(pick=lambda: faulted, enable_record_logging=lambda *a, **k: None)

        report = PickRun.from_service(service, runs=3, recording=Recording.off()).execute()

        self.assertTrue(report.raised)
        self.assertEqual(report.cancelled, 2)
        self.assertTrue(report.attempts[0].detail.startswith("CameraWorldUnavailable: camera 'overhead'"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
