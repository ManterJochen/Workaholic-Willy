"""A robot picks a part and places it: standoff, one line in, the hand verb, one line out.

Before any command a pick or a place refuses a robot that cannot hold, a closed link, a pose not in BASE, a camera world
the arm's own motion would be refused for, and an arm that keeps no line or does not say. Every motion carries the
caller's decline and waits for the arm's own steady gate where its tree asks for one. A refused motion ends the verb
with nothing commanded after it, a close that measures nothing opens and backs out, and a release the gripper does not
confirm stays where it stands. A pick holds its keep-out offer through every motion.

Every cuRobo double here states its camera world at the call site.
"""

from __future__ import annotations

import json
import time
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import MotionResult
from src.robot.core.camera_world import (
    DECLINE_ON_A_LIVE_WORLD_MESSAGE,
    NO_CAMERA_WORLD_MESSAGE,
    CameraWorldDecline,
    CameraWorldUse,
)
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.core.gripper import HoldEvidence
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.execution.robot import Robot
from tests.test_robot_hand_verbs import _Hand

_DECLINE = CameraWorldDecline("unit double: a pick on a bench with no cameras")


def _handling():  # noqa: ANN202
    from src.robot.execution import handling

    return handling


def _pose(x: float = 400.0, z: float = 100.0) -> Pose:
    """Tool pointing down: the pose's own +Z, the approach, is BASE -Z."""
    return Pose(position_mm=np.array([x, 0.0, z]), quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]), frame=Frame.BASE)


class _Log:
    def __init__(self) -> None:
        self.entries: list[tuple[Any, ...]] = []


class _RecordingArm(DummyRobotArm):
    """The dummy arm, writing each motion and its keywords onto a log the gripper shares; fails where told."""

    def __init__(self, log: _Log, *, refuse_linear: bool = False, raise_on: "int | None" = None,
                 world: Any = None) -> None:
        super().__init__()
        self.log = log
        self.refuse_linear = refuse_linear
        self.raise_on = raise_on
        self.detaches = 0
        if world is not None:
            self.live_planner_world = world
        self.held_at_motion: list[bool] = []

    def detach_payload(self) -> bool:
        self.detaches += 1
        self.log.entries.append(("detach",))
        return True

    def move(self, pose: Pose, **kwargs: Any) -> MotionResult:
        index = sum(1 for entry in self.log.entries if entry[0] == "move")
        world = getattr(self, "live_planner_world", None)
        self.held_at_motion.append(bool(getattr(world, "holding", False)))
        self.log.entries.append(("move", tuple(round(float(v), 3) for v in pose.position_mm),
                                 bool(kwargs.get("linear", False)), kwargs.get("camera_world")))
        if self.raise_on is not None and index == self.raise_on:
            raise CameraWorldUnavailable(camera="overhead", verdict="no frame", attempts=4,
                                         reason="the overhead camera stayed silent")
        if self.refuse_linear and kwargs.get("linear"):
            from src.robot.core import MotionCommand, MotionStatus

            return MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                                       message="lfinger|fixture:seen_00: mesh distance 0.4 mm")
        return super().move(pose, **kwargs)


class _LoggedHand(_Hand):
    def __init__(self, log: _Log, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.log = log

    def set_width_mm(self, width_mm: float, *, speed: float | None = None, force: float | None = None) -> None:
        self.log.entries.append(("set_width", float(width_mm)))
        super().set_width_mm(width_mm, speed=speed, force=force)


def _robot(log: _Log, *, hold: HoldEvidence = HoldEvidence.HELD, **arm_kwargs: Any) -> Robot:
    arm = _RecordingArm(log, **arm_kwargs)
    arm.connect()
    return Robot.from_parts(arm=arm, gripper=_LoggedHand(log, hold=hold, width_mm=39.0), lock_key=None)


def _kinds(log: _Log) -> list[Any]:
    out: list[Any] = []
    for entry in log.entries:
        if entry[0] == "move":
            out.append(("move", entry[1], entry[2]))
        else:
            out.append(entry)
    return out


class PickAndPlaceInOrderTests(unittest.TestCase):
    def test_a_pick_in_order(self) -> None:
        handling = _handling()
        log = _Log()
        robot = _robot(log)

        report = robot.pick(_pose(), 40.0)

        self.assertIs(handling.HandlingOutcome.EXECUTED, report.outcome, report.render())
        self.assertEqual([
            ("detach",),
            ("set_width", 85.0),
            ("move", (400.0, 0.0, 180.0), False),
            ("move", (400.0, 0.0, 100.0), True),
            ("set_width", 39.0),
            ("move", (400.0, 0.0, 180.0), True),
        ], _kinds(log))
        assert report.hand is not None
        self.assertIs(HoldEvidence.HELD, report.hand.hold)

    def test_a_place_in_order(self) -> None:
        handling = _handling()
        log = _Log()
        robot = _robot(log, hold=HoldEvidence.EMPTY)

        report = robot.place(_pose(), standoff_mm=50.0)

        self.assertIs(handling.HandlingOutcome.EXECUTED, report.outcome, report.render())
        self.assertEqual([
            ("move", (400.0, 0.0, 150.0), False),
            ("move", (400.0, 0.0, 100.0), True),
            ("set_width", 85.0),
            ("move", (400.0, 0.0, 150.0), True),
        ], _kinds(log))

    def test_every_motion_carries_the_callers_decline(self) -> None:
        log = _Log()
        robot = _robot(log)

        robot.pick(_pose(), 40.0, camera_world=_DECLINE)
        robot.gripper.hold = HoldEvidence.EMPTY  # type: ignore[union-attr]  # the part left the jaws, so the place backs out
        robot.place(_pose(), camera_world=_DECLINE)

        declines = [entry[3] for entry in log.entries if entry[0] == "move"]
        self.assertEqual(6, len(declines))
        self.assertTrue(all(decline is _DECLINE for decline in declines))

    def test_an_empty_grasp_opens_and_backs_out(self) -> None:
        handling = _handling()
        log = _Log()
        robot = _robot(log, hold=HoldEvidence.EMPTY)

        report = robot.pick(_pose(), 40.0)

        self.assertIs(handling.HandlingOutcome.NOTHING_HELD, report.outcome)
        self.assertEqual([("set_width", 39.0), ("set_width", 85.0), ("move", (400.0, 0.0, 180.0), True)],
                         _kinds(log)[-3:])

    def test_an_unconfirmed_release_stays_put(self) -> None:
        handling = _handling()
        log = _Log()
        robot = _robot(log, hold=HoldEvidence.HELD)

        report = robot.place(_pose())

        self.assertIs(handling.HandlingOutcome.RELEASE_NOT_CONFIRMED, report.outcome)
        self.assertEqual(2, sum(1 for entry in log.entries if entry[0] == "move"))


class TheCameraWorldIsRefusedBeforeAnyCommandTests(unittest.TestCase):
    def test_a_pick_on_a_curobo_arm_with_no_world_and_no_decline_commands_nothing(self) -> None:
        from tests.test_camera_world_on_every_driver import _sim_unconnected, _ur_curobo

        handling = _handling()
        for label, build in (("ur", _ur_curobo), ("sim, not a UR", _sim_unconnected)):
            with self.subTest(label):
                arm = build()
                hand = _Hand(hold=HoldEvidence.HELD)
                robot = Robot.from_parts(arm=arm, gripper=hand, lock_key=None)

                report = robot.pick(_pose(), 40.0)

                self.assertIs(handling.HandlingOutcome.REFUSED, report.outcome, report.render())
                self.assertEqual(NO_CAMERA_WORLD_MESSAGE, report.message)
                self.assertIs(CameraWorldUse.MISSING, report.camera_worlds[0].use)
                self.assertEqual([], hand.commands)
                self.assertEqual((), report.poses)

    def test_with_a_decline_the_motions_say_declined(self) -> None:
        """The control: the same sim double with a decline gets past the refusal and its first motion says DECLINED."""
        from tests.test_every_verb_meets_the_camera_world import _Client
        from tests.test_the_goal_keeps_its_jaws_clear import _sim_arm

        from src.robot.drivers.sim import arm as sim_arm_module

        client = _Client(joint_names=sim_arm_module._ARM_JOINT_NAMES)
        client.dt = 0.01  # type: ignore[attr-defined]
        arm = _sim_arm(client)
        arm.set_live_planner_world(None)
        arm._execute_curobo_trajectory = MagicMock()  # type: ignore[method-assign]
        arm._preflight.path_judge_refusal.return_value = None  # type: ignore[union-attr]
        robot = Robot.from_parts(arm=arm, gripper=_Hand(hold=HoldEvidence.HELD), lock_key=None)

        report = robot.pick(_pose(), 40.0, camera_world=_DECLINE)

        self.assertGreaterEqual(len(report.camera_worlds), 1, report.render())
        self.assertIs(CameraWorldUse.DECLINED, report.camera_worlds[0].use)

    def test_a_declined_pick_on_a_wired_world_commands_nothing(self) -> None:
        from tests.test_camera_world_on_every_driver import _ur_curobo

        handling = _handling()
        hand = _Hand(hold=HoldEvidence.HELD)
        robot = Robot.from_parts(arm=_ur_curobo(live_world=object()), gripper=hand, lock_key=None)

        report = robot.pick(_pose(), 40.0, camera_world=_DECLINE)

        self.assertIs(handling.HandlingOutcome.REFUSED, report.outcome)
        self.assertEqual(DECLINE_ON_A_LIVE_WORLD_MESSAGE, report.message)
        self.assertEqual([], hand.commands)


class AMotionThatFailsEndsTheVerbTests(unittest.TestCase):
    def test_a_refused_descent_closes_nothing(self) -> None:
        handling = _handling()
        log = _Log()
        robot = _robot(log, refuse_linear=True)

        report = robot.pick(_pose(), 40.0)

        self.assertIs(handling.HandlingOutcome.MOTION_REFUSED, report.outcome)
        self.assertIn("mesh distance 0.4 mm", report.message)
        widths = [entry[1] for entry in log.entries if entry[0] == "set_width"]
        self.assertEqual([85.0], widths, "the jaws closed after a refused descent")
        self.assertEqual(2, len(report.poses))

    def test_a_camera_that_cannot_vouch_ends_the_pick_as_its_own_outcome(self) -> None:
        """The verb promises a report, so the camera's fault is an outcome, and the motions before it stay in it."""
        from src.robot.core.keep_out import SegmentationOffer
        from tests.test_keep_out_path import _HoldingWorld

        handling = _handling()
        log = _Log()
        world = _HoldingWorld()
        robot = _robot(log, raise_on=1, world=world)
        offer = SegmentationOffer(captured_at_s=time.time(), target_points_base_mm=np.array([[400.0, 0.0, 90.0]]))

        report = robot.pick(_pose(), 40.0, keep_out=offer)

        self.assertIs(handling.HandlingOutcome.CAMERA_WORLD_UNAVAILABLE, report.outcome, report.render())
        self.assertFalse(report.ok)
        self.assertIn("'overhead'", report.message)
        self.assertEqual(2, len(report.poses), "the standoff and the descent the camera stopped")
        self.assertEqual(1, world.forgets, "the scope stayed open after the camera's fault")
        self.assertFalse(world.holding)
        widths = [entry[1] for entry in log.entries if entry[0] == "set_width"]
        self.assertEqual([85.0], widths, "a gripper command followed the camera's fault")

    def test_a_pick_holds_its_keep_out_for_its_motions(self) -> None:
        from src.robot.core.keep_out import SegmentationOffer
        from tests.test_keep_out_path import _HoldingWorld

        handling = _handling()
        log = _Log()
        world = _HoldingWorld()
        robot = _robot(log, world=world)
        offer = SegmentationOffer(captured_at_s=time.time(), target_points_base_mm=np.array([[400.0, 0.0, 90.0]]))

        report = robot.pick(_pose(), 40.0, keep_out=offer)

        self.assertIs(handling.HandlingOutcome.EXECUTED, report.outcome)
        self.assertTrue(report.keep_out_held)
        arm = robot.arm
        assert isinstance(arm, _RecordingArm)
        self.assertEqual([True, True, True], arm.held_at_motion)
        self.assertEqual(1, world.forgets)
        self.assertFalse(world.holding)


class LinesTheArmCannotKeepTests(unittest.TestCase):
    def test_the_ur_descent_is_a_checked_line(self) -> None:
        from src.robot.core import MotionCommand
        from tests.test_ur_checked_linear import _arm as _line_ur
        from tests.test_ur_checked_linear import _mesh_backend_available

        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")
        arm = _line_ur()
        arm._curobo_ur.detach_payload = lambda: True  # type: ignore[union-attr]
        arm.wait_until_steady = lambda timeout_s=5.0, poll_interval_s=0.02: True  # type: ignore[method-assign]
        approach = MagicMock(return_value=MotionResult.executed(MotionCommand.MOVE_TO, message="planned"))
        arm._drive_curobo = approach  # type: ignore[method-assign]
        planner = arm._curobo_ur
        checks: list[int] = []
        original = planner.check_joint_path  # type: ignore[union-attr]

        def counted(samples, *, refresh=True):  # noqa: ANN001, ANN202
            checks.append(len(list(samples)))
            return original(samples, refresh=refresh)

        planner.check_joint_path = counted  # type: ignore[union-attr, method-assign]
        robot = Robot.from_parts(arm=arm, gripper=_Hand(hold=HoldEvidence.HELD), lock_key=None)

        report = robot.pick(_pose(x=300.0, z=300.0), 40.0, camera_world=_DECLINE, standoff_mm=30.0)

        from src.robot.core.arm_capabilities import LineMotion

        assert report.line is not None
        self.assertIs(LineMotion.CHECKED, report.line.motion, report.line.reason)
        approach.assert_called_once()
        self.assertEqual(2, len(checks), report.render())

    def test_a_sim_on_ik_refuses_before_moving(self) -> None:
        from src.robot.drivers.sim.arm import IsaacRobotArm
        from src.robot.drivers.sim.config import SimRobotConfig
        from tests.test_camera_world_on_every_driver import _sim_mock

        handling = _handling()
        arm = IsaacRobotArm(SimRobotConfig(enabled=True, scene="placeholder.usd", robot_prim_path="/World/Robot",
                                           motion_planner="ik"))
        arm._connected = True
        hand = _Hand(hold=HoldEvidence.HELD)
        robot = Robot.from_parts(arm=arm, gripper=hand, lock_key=None)

        report = robot.pick(_pose(), 40.0)

        self.assertIs(handling.HandlingOutcome.REFUSED, report.outcome)
        self.assertIn("'ik'", report.message)
        self.assertEqual([], hand.commands)

        mock = _sim_mock()
        control = Robot.from_parts(arm=mock, gripper=_Hand(hold=HoldEvidence.HELD), lock_key=None).pick(_pose(), 40.0)
        self.assertIs(handling.HandlingOutcome.EXECUTED, control.outcome, control.render())
        self.assertIn("no controller", control.render())

    def test_a_ur_whose_path_cannot_be_judged_refuses_before_the_pre_open(self) -> None:
        from src.robot.safety.preflight import SafetyPreflight
        from tests.test_ur_checked_linear import _arm as _line_ur

        handling = _handling()
        hand = _Hand(hold=HoldEvidence.HELD)
        robot = Robot.from_parts(arm=_line_ur(), gripper=hand, lock_key=None)
        sentence = "no self_collision guard is wired, so nothing here can judge a path"

        with patch.object(SafetyPreflight, "path_judge_refusal", return_value=sentence):
            report = robot.pick(_pose(), 40.0, camera_world=_DECLINE)

        self.assertIs(handling.HandlingOutcome.REFUSED, report.outcome)
        self.assertIn(sentence, report.message)
        self.assertEqual([], hand.commands)

    def test_an_arm_that_reports_no_line_motion_is_refused(self) -> None:
        handling = _handling()

        class _Typed:
            is_connected = True

            def move(self, pose: Pose, **_: Any) -> MotionResult:
                raise AssertionError("moved")

        hand = _Hand(hold=HoldEvidence.HELD)
        robot = Robot.from_parts(arm=_Typed(), gripper=hand, lock_key=None)  # type: ignore[arg-type]

        report = robot.pick(_pose(), 40.0)

        self.assertIs(handling.HandlingOutcome.REFUSED, report.outcome)
        self.assertIn("_Typed", report.message)
        self.assertEqual([], hand.commands)


class TheSteadyGateTests(unittest.TestCase):
    def test_a_dwell_timeout_moves_nothing_further(self) -> None:
        from src.robot.core import MotionStatus
        from src.robot.safety.preflight import SafetyPreflight
        from tests.test_ur_checked_linear import _arm as _line_ur

        handling = _handling()
        arm = _line_ur()
        arm._curobo_ur.detach_payload = lambda: True  # type: ignore[union-attr]
        arm.wait_until_steady = lambda timeout_s=5.0, poll_interval_s=0.02: False  # type: ignore[method-assign]
        arm._drive_curobo = MagicMock()  # type: ignore[method-assign]
        hand = _Hand(hold=HoldEvidence.HELD)
        robot = Robot.from_parts(arm=arm, gripper=hand, lock_key=None)

        with patch.object(SafetyPreflight, "path_judge_refusal", return_value=None):
            report = robot.pick(_pose(), 40.0, camera_world=_DECLINE)

        self.assertIs(handling.HandlingOutcome.MOTION_REFUSED, report.outcome, report.render())
        self.assertEqual((MotionStatus.TIMEOUT,), report.statuses)
        arm._drive_curobo.assert_not_called()
        self.assertEqual([("set_width_mm", 85.0)], hand.commands)


class RefusalsBeforeAnyCommandTests(unittest.TestCase):
    def test_a_camera_frame_pose_is_refused_before_any_command(self) -> None:
        handling = _handling()
        log = _Log()
        robot = _robot(log)
        camera_pose = Pose(position_mm=np.zeros(3), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), frame=Frame.CAMERA)

        report = robot.pick(camera_pose, 40.0)

        self.assertIs(handling.HandlingOutcome.REFUSED, report.outcome)
        self.assertEqual([], log.entries)

    def test_a_pick_outside_connected_is_refused(self) -> None:
        handling = _handling()
        log = _Log()
        arm = _RecordingArm(log)
        robot = Robot.from_parts(arm=arm, gripper=_LoggedHand(log, hold=HoldEvidence.HELD), lock_key=None)

        for report in (robot.pick(_pose(), 40.0), robot.place(_pose())):
            self.assertIs(handling.HandlingOutcome.REFUSED, report.outcome)
            self.assertIn("robot.connected()", report.message)
        self.assertEqual([], log.entries)


class TheReportTests(unittest.TestCase):
    def test_render_and_to_dict(self) -> None:
        log = _Log()
        robot = _robot(log)
        for report in (robot.pick(_pose(), 40.0), robot.place(_pose()), _robot(_Log(), hold=HoldEvidence.EMPTY)
                       .pick(Pose(position_mm=np.zeros(3), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                                  frame=Frame.CAMERA), 40.0)):
            with self.subTest(report.outcome.value):
                text = report.render()
                self.assertTrue(text.isascii())
                self.assertFalse(text.endswith("\n"))
                data = json.loads(json.dumps(report.to_dict()))
                self.assertEqual(report.outcome.value, data["outcome"])
                self.assertEqual(len(report.poses), len(data["poses_mm"]))
        executed = robot.pick(_pose(), 40.0)
        self.assertIn("no controller", executed.render())
        self.assertIs(CameraWorldUse.UNPLANNED, executed.weakest_camera_world.use)  # type: ignore[union-attr]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
