"""The camera-world stamp reaches what a person reads, and stops short of the record.

A stamp that stays on a motion result nobody prints is not visible. The owner chose where it goes:
``PolicyReport`` and ``PickSessionReport`` carry one stamp per typed motion and read the weakest one
first; the attempt and the progress event carry that weakest stamp's use and reason whenever it says
something; the console speaks only MISSING and DECLINED, at the event's own severity, because a dummy
or ik cell would otherwise repeat UNPLANNED on every event; and the pick report prints a ``camera`` line
on every attempt. ``GraspAttemptRecord`` does not carry the stamp in this step.
"""

from __future__ import annotations

import io
import json
import unittest
from collections.abc import Sequence
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np

from api.events import Severity
from api.runs import _payload, _sentence
from src.geometry import Frame, Pose
from src.robot.core import MotionCommand, MotionResult, MotionStatus, RobotCapabilities
from src.robot.core.camera_world import CameraWorldDecline, CameraWorldStamp
from src.robot.execution.autonomous_grasp.config import GraspMode, _profile_for
from src.robot.execution.autonomous_grasp.record_logging import to_attempt_record
from src.robot.execution.autonomous_grasp.report import (
    AutonomousGraspOutcome,
    AutonomousGraspReport,
)
from src.robot.execution.runtime_pick import PickSessionReport, RuntimePickService
from src.robot.grasping.loop import pick_loop
from src.robot.grasping.loop.pick_loop import PickAttempt, PickOutcome
from src.robot.grasping.loop.progress import PickProgress, PickStage
from src.robot.grasping.motion import execution_policy
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy, PolicyOutcome
from tests.test_grasp_execution_policy import _FakeArm, _grasp
from tests.test_runtime_pick import (
    _empty_perception_frame,
    _FakePerception,
    _LegacyFakeArm,
    _perception_frame,
    _rescan_only_result,
    _ScriptedCalculator,
    _success_result,
)

_PLANNED = tuple(
    CameraWorldStamp.planned(cameras=("realsense_d435",), captured_at_s=float(second))
    for second in range(1, 6)
)
_MISSING = CameraWorldStamp.missing(
    "robot.ur.motion_planner is 'curobo' and no live camera world is wired to this arm")
_DECLINED = CameraWorldStamp.declined(CameraWorldDecline("bench check, no cameras mounted"))
_UNPLANNED = CameraWorldStamp.unplanned("DummyRobotArm has no planner")

_CAPS = RobotCapabilities(vendor="dummy", model="double", is_simulated=True)


class _StampingArm:
    """A typed arm whose n-th motion carries the n-th stamp, or ``every`` stamp, and fails at ``fail_at``."""

    def __init__(self, stamps: Sequence[CameraWorldStamp] = (), *,
                 every: CameraWorldStamp | None = None, fail_at: int | None = None) -> None:
        self._stamps, self._every, self._fail_at = tuple(stamps), every, fail_at
        self.calls = 0
        self.capabilities = _CAPS

    def get_tcp_pose(self) -> Pose:
        return Pose(position_mm=np.array([0.0, 0.0, 500.0]),
                    quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), frame=Frame.BASE)

    def move(self, pose: Pose, **_: object) -> MotionResult:
        index, self.calls = self.calls, self.calls + 1
        stamp = self._every if self._every is not None else self._stamps[index]
        if index == self._fail_at:
            return MotionResult.failed(MotionStatus.WORKSPACE_REJECTED, MotionCommand.MOVE_TO,
                                       target_pose=pose, message="outside the box",
                                       camera_world=stamp)
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose, camera_world=stamp)


def _service(arm: object, *, frame: object, result: object) -> RuntimePickService:
    return RuntimePickService.from_components(
        arm=arm,  # type: ignore[arg-type]
        calculator=_ScriptedCalculator([result]),  # type: ignore[list-item, arg-type]
        perception=_FakePerception([frame]),  # type: ignore[list-item]
        max_attempts=2,
    )


# ---------------------------------------------------------------------------------------------------
# 1. The policy
# ---------------------------------------------------------------------------------------------------


class ThePolicyCarriesOneStampPerMotionTests(unittest.TestCase):

    def test_five_motions_carry_five_stamps_and_the_first_that_does_not_vouch_is_read(self) -> None:
        stamps = (_PLANNED[0], _PLANNED[1], _MISSING, _DECLINED, _PLANNED[4])
        report = GraspExecutionPolicy(arm=_StampingArm(stamps), approach_steps=4).execute(_grasp())
        self.assertIs(report.outcome, PolicyOutcome.EXECUTED)
        self.assertEqual(report.camera_worlds, stamps)
        self.assertEqual(report.camera_world, _MISSING)

    def test_when_every_motion_vouches_the_last_stamp_is_read(self) -> None:
        report = GraspExecutionPolicy(arm=_StampingArm(_PLANNED), approach_steps=4).execute(_grasp())
        self.assertEqual(report.camera_world, _PLANNED[4])

    def test_a_legacy_arm_carries_no_stamp(self) -> None:
        report = GraspExecutionPolicy(arm=_FakeArm(), approach_steps=4).execute(_grasp())
        self.assertEqual(report.camera_worlds, ())
        self.assertIsNone(report.camera_world)

    def test_a_failure_mid_approach_carries_the_stamp_of_the_motion_that_failed(self) -> None:
        stamps = (_UNPLANNED, _UNPLANNED, _DECLINED, _UNPLANNED, _UNPLANNED)
        arm = _StampingArm(stamps, fail_at=2)
        report = GraspExecutionPolicy(arm=arm, approach_steps=4).execute(_grasp())
        self.assertIs(report.outcome, PolicyOutcome.MOTION_FAILED)
        self.assertIs(report.motion_status, MotionStatus.WORKSPACE_REJECTED)
        self.assertEqual(report.camera_worlds, stamps[:3])
        self.assertEqual(report.camera_world, _UNPLANNED)

    def test_the_weakest_of_no_stamp_is_none(self) -> None:
        self.assertIsNone(execution_policy.weakest_camera_world(()))


# ---------------------------------------------------------------------------------------------------
# 2. The session report
# ---------------------------------------------------------------------------------------------------


class TheSessionReportCarriesThePolicyStampsTests(unittest.TestCase):

    def test_the_session_report_carries_what_the_policy_carried(self) -> None:
        arm = _StampingArm(every=_UNPLANNED)
        service = _service(arm, frame=_perception_frame(), result=_success_result())
        report = service.run_attempt()
        self.assertIs(report.outcome, PickOutcome.EXECUTED)
        self.assertEqual(report.camera_worlds, (_UNPLANNED,) * arm.calls)
        self.assertEqual(report.camera_worlds,
                         service.orchestrator._last_policy_report.camera_worlds)  # noqa: SLF001
        self.assertEqual(report.camera_world, _UNPLANNED)

    def test_a_legacy_arm_and_a_pick_that_never_moved_carry_no_stamp(self) -> None:
        legacy = _service(_LegacyFakeArm(), frame=_perception_frame(), result=_success_result())
        blind = _service(_StampingArm(()), frame=_empty_perception_frame(),
                         result=_rescan_only_result())
        for label, service in (("legacy", legacy), ("no perception", blind)):
            with self.subTest(label):
                report = service.run_attempt()
                self.assertEqual(report.camera_worlds, ())
                self.assertIsNone(report.camera_world)


# ---------------------------------------------------------------------------------------------------
# 3. The attempt and the progress event
# ---------------------------------------------------------------------------------------------------


class TheAttemptAndTheEventCarryTheWeakestStampTests(unittest.TestCase):

    @staticmethod
    def _run(stamp: CameraWorldStamp) -> tuple[PickAttempt, PickProgress]:
        events: list[PickProgress] = []
        service = _service(_StampingArm(every=stamp), frame=_perception_frame(),
                           result=_success_result())
        service.orchestrator.on_progress = events.append
        report = service.run_attempt()
        finished = [event for event in events if event.stage is PickStage.ATTEMPT_FINISHED]
        return report.attempts[-1], finished[-1]

    def test_an_unstated_stamp_adds_nothing(self) -> None:
        attempt, event = self._run(CameraWorldStamp.unstated())
        self.assertEqual((attempt.camera_world, attempt.camera_world_reason), (None, None))
        self.assertEqual((event.camera_world, event.camera_world_reason), (None, None))

    def test_a_stamp_that_says_something_reaches_both(self) -> None:
        attempt, event = self._run(_UNPLANNED)
        expected = ("unplanned", "DummyRobotArm has no planner")
        self.assertEqual((attempt.camera_world, attempt.camera_world_reason), expected)
        self.assertEqual((event.camera_world, event.camera_world_reason), expected)

    def test_the_fields_default_to_none(self) -> None:
        attempt = PickAttempt(attempt_index=0, target_index=None, reasons=(), score=0.0,
                              action="executed")
        event = PickProgress(stage=PickStage.ATTEMPT_FINISHED)
        for carrier in (attempt, event):
            with self.subTest(type(carrier).__name__):
                self.assertIsNone(carrier.camera_world)
                self.assertIsNone(carrier.camera_world_reason)
        self.assertEqual(pick_loop._motion_fields(None), {
            "motion_status": None, "motion_message": None, "motion_error": None,
            "camera_world": None, "camera_world_reason": None,
        })


# ---------------------------------------------------------------------------------------------------
# 4. The console
# ---------------------------------------------------------------------------------------------------


class TheConsoleTests(unittest.TestCase):

    def test_the_payload_omits_an_unsaid_world_and_carries_a_said_one(self) -> None:
        plain = _payload(PickProgress(stage=PickStage.ATTEMPT_FINISHED, attempt=0, action="executed"))
        self.assertNotIn("camera_world", plain)
        self.assertNotIn("camera_world_reason", plain)
        said = _payload(PickProgress(
            stage=PickStage.ATTEMPT_FINISHED, attempt=0, action="executed",
            camera_world="unplanned", camera_world_reason="DummyRobotArm has no planner",
        ))
        self.assertEqual((said["camera_world"], said["camera_world_reason"]),
                         ("unplanned", "DummyRobotArm has no planner"))

    def test_the_sentence_is_unchanged_when_the_world_is_unsaid_or_unplanned(self) -> None:
        for extra in ({}, {"camera_world": "unplanned",
                           "camera_world_reason": "DummyRobotArm has no planner"}):
            with self.subTest(**extra):
                self.assertEqual(
                    _sentence(PickProgress(stage=PickStage.ATTEMPT_FINISHED, attempt=0,
                                           action="executed", **extra)),
                    (Severity.INFO, "Attempt ended: executed."),
                )

    def test_missing_and_declined_are_spoken_at_the_event_severity(self) -> None:
        missing = _sentence(PickProgress(
            stage=PickStage.ATTEMPT_FINISHED, attempt=0, action="executed",
            camera_world="missing", camera_world_reason="no live camera world is wired",
        ))
        self.assertEqual(missing, (
            Severity.INFO,
            "Attempt ended: executed. Camera world: MISSING (no live camera world is wired).",
        ))
        declined = _sentence(PickProgress(
            stage=PickStage.ATTEMPT_FINISHED, attempt=0, action="execution_failed",
            motion_status="unsupported", motion_message="Refused before planning.",
            camera_world="declined", camera_world_reason="bench check",
        ))
        self.assertEqual(declined, (
            Severity.WARN,
            "Attempt ended: execution_failed. Nothing moved: Refused before planning. (unsupported) "
            "Camera world: DECLINED (bench check).",
        ))


# ---------------------------------------------------------------------------------------------------
# 5. The pick report
# ---------------------------------------------------------------------------------------------------


def _pick(camera_worlds: tuple[CameraWorldStamp, ...]) -> PickSessionReport:
    return PickSessionReport(outcome=PickOutcome.EXECUTED, robot_vendor="dummy", is_simulated=True,
                             gripper_present=True, camera_worlds=camera_worlds)


def _report(pick: object) -> AutonomousGraspReport:
    return AutonomousGraspReport(outcome=AutonomousGraspOutcome.SUCCEEDED, mode=GraspMode.AUTO,
                                 profile=_profile_for(GraspMode.AUTO),
                                 pick_report=pick)  # type: ignore[arg-type]


class ThePickReportTests(unittest.TestCase):

    def test_render_prints_a_camera_line_on_every_attempt(self) -> None:
        cases = (
            ("unplanned", _report(_pick((_UNPLANNED,) * 5)),
             "  camera     0 of 5 motion(s) vouched; camera world  UNPLANNED  DummyRobotArm has no "
             "planner"),
            ("mixed", _report(_pick((_PLANNED[0], _MISSING, _PLANNED[1]))),
             "  camera     2 of 3 motion(s) vouched; camera world  MISSING  robot.ur.motion_planner is "
             "'curobo' and no live camera world is wired to this arm"),
            ("no typed motion", _report(_pick(())), "  camera     no typed motion was commanded"),
            ("no pick", _report(None), "  camera     no typed motion was commanded"),
        )
        for label, report, line in cases:
            with self.subTest(label):
                text = report.render()
                self.assertIn(line, text.splitlines())
                self.assertTrue(text.isascii())
                self.assertFalse(text.endswith("\n"))

    def test_to_dict_carries_the_stamps_as_views_and_survives_json(self) -> None:
        stamps = (_PLANNED[0], _MISSING)
        payload = _report(_pick(stamps)).to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertEqual(payload["camera_worlds"], [stamp.to_dict() for stamp in stamps])
        self.assertEqual(payload["camera_world"], _MISSING.to_dict())
        nothing = _report(None).to_dict()
        self.assertEqual((nothing["camera_worlds"], nothing["camera_world"]), ([], None))

    def test_the_rehearsal_transcript_prints_the_camera_line(self) -> None:
        from src.robot.execution.real_cell.__main__ import main

        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["--rehearse", "--runs", "1", "--profile", "console_dummy"])
        self.assertEqual(code, 0)
        self.assertIn(
            "  camera     0 of 5 motion(s) vouched; camera world  UNPLANNED  DummyRobotArm has no "
            "planner",
            out.getvalue().splitlines(),
        )


# ---------------------------------------------------------------------------------------------------
# 6. The record
# ---------------------------------------------------------------------------------------------------


class TheRecordDoesNotCarryTheStampTests(unittest.TestCase):
    """Q4: no record change until the service first declines. A control, green before and after."""

    def test_a_pick_with_stamps_writes_the_record_a_pick_without_them_writes(self) -> None:
        common = dict(calculator_telemetry={}, executed_grasp=None, attempts=(),
                      motion_status_chain=(), motion_message="")
        stamped = SimpleNamespace(camera_worlds=(_MISSING, _DECLINED), camera_world=_MISSING, **common)
        plain = SimpleNamespace(**common)
        records = [to_attempt_record(_report(pick), attempt_id="a", timestamp=0.0).to_dict()
                   for pick in (stamped, plain)]
        self.assertEqual(records[0], records[1])
        self.assertEqual(list(records[0]), [
            "schema_version", "timestamp", "attempt_id", "mode", "final_outcome", "profile", "frame",
            "target", "initial_grasp", "initial_telemetry", "refined_grasp", "refinement",
            "selected_grasp", "execution", "verification", "recovery_actions", "extra",
        ])
        self.assertNotIn("camera_world", json.dumps(records[0]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
