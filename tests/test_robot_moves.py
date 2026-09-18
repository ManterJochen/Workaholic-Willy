"""A robot moves its arm as one verb and says what stood behind the motion.

``Robot.move``, ``Robot.move_joints`` and ``Robot.home`` each return one report, and everything that moves the arm goes
through cuRobo and the exact mesh guard (Coal) with the camera world. So each verb reads the arm's route before any
command: PLANNED on a cuRobo arm whose paths are judged, UNPLANNED on a desk arm (the dummy, the kinematic mock), and
REFUSED on an arm that plans nothing where a real arm would move (a UR on the ik planner, a KUKA), with nothing
commanded. ``decline="reason"`` is a keyword on every verb that moves, beside the block. A verb that promises a report
does not raise for a domain refusal; a programmer's error still raises.

Every cuRobo double here states its camera world at the call site, and the path judge is patched only where the test
says the arm's paths are judged.
"""

from __future__ import annotations

import inspect
import json
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from src.contracts import UNSET
from src.geometry import Frame
from src.robot.core import MotionCommand, MotionResult, MotionStatus, RobotCapabilities
from src.robot.core.camera_world import (
    DECLINE_ON_A_LIVE_WORLD_MESSAGE,
    NO_CAMERA_WORLD_MESSAGE,
    CameraWorldDecline,
    CameraWorldStamp,
    CameraWorldUse,
    active_decline,
)
from src.robot.core.errors import CameraWorldUnavailable, RobotConnectionError, RobotMotionRejected
from src.robot.core.gripper import HoldEvidence
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.drivers.sim.arm import IsaacRobotArm
from src.robot.drivers.sim.config import SimRobotConfig
from src.robot.execution.robot import Robot
from src.robot.safety.preflight import SafetyPreflight
from tests.test_camera_world_on_every_driver import _JOINTS, _dummy, _kuka, _sim_mock, _ur_curobo, _ur_ik
from tests.test_robot import _ROBOT_MUST_NOT_LOAD
from tests.test_robot_hand_verbs import _Hand
from tests.test_robot_parts import _loaded_after
from tests.test_robot_pick_and_place import _Log, _robot
from tests.test_robot_pick_and_place import _pose as _pick_pose
from tests.test_ur_arm import _pose

_BENCH = "bench check, no cameras mounted"


def _motion():  # noqa: ANN202
    """The module the verbs live in, imported by each test so a missing one fails that test alone."""
    from src.robot.execution import motion

    return motion


def _judged():  # noqa: ANN202
    """The arm's paths are judged: the exact mesh guard answers for every sample, as on a cell that wires it."""
    return patch.object(SafetyPreflight, "path_judge_refusal", return_value=None)


def _desk(arm: DummyRobotArm | None = None) -> Robot:
    arm = arm if arm is not None else DummyRobotArm()
    arm.connect()
    return Robot.from_parts(arm=arm, gripper=None, lock_key=None)


def _tripwires(arm: Any) -> dict[str, MagicMock]:
    """The arm's three motion verbs, each failing the test if it is reached."""
    wires = {name: MagicMock(side_effect=AssertionError(f"{name} was commanded"))
             for name in ("move", "move_to_joints", "move_home")}
    for name, wire in wires.items():
        setattr(arm, name, wire)
    return wires


def _every_verb(robot: Robot, **decline: str) -> list[Any]:
    return [robot.move(_pose(), **decline), robot.move_joints(_JOINTS, **decline), robot.home(**decline)]


class _Claims(DummyRobotArm):
    """A caller's arm that says it sets the pose, on a controller it does not say is simulated."""

    @property
    def capabilities(self) -> RobotCapabilities:
        return RobotCapabilities(vendor="ur", model="ur5e")


class _Keywords(DummyRobotArm):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def move(self, pose: Any, **kwargs: Any) -> MotionResult:
        self.calls.append(dict(kwargs))
        return super().move(pose, **kwargs)


class _SilentCamera(DummyRobotArm):
    def move(self, pose: Any, **kwargs: Any) -> MotionResult:
        raise CameraWorldUnavailable(camera="overhead", verdict="no frame", attempts=4,
                                     reason="the overhead camera stayed silent")


class _Raises(DummyRobotArm):
    def __init__(self, exc: BaseException) -> None:
        super().__init__()
        self.exc = exc

    def move_to_joints(self, joints: Any, **kwargs: Any) -> MotionResult:
        raise self.exc


class _RefusesHome(DummyRobotArm):
    def move_home(self) -> bool:
        return False


class _Untyped(DummyRobotArm):
    def move(self, pose: Any, **kwargs: Any) -> Any:
        return True


# ---------------------------------------------------------------------------------------------------
# The route, read before any command
# ---------------------------------------------------------------------------------------------------


class TheRouteTests(unittest.TestCase):

    def test_each_arm_reads_its_route(self) -> None:
        motion = _motion()
        sim_ik = IsaacRobotArm(SimRobotConfig(enabled=True, scene="placeholder.usd", robot_prim_path="/World/Robot",
                                              motion_planner="ik"))
        rows: list[tuple[str, object, Any, str]] = [
            ("dummy", _dummy(), motion.MotionRoute.UNPLANNED, "DummyRobotArm has no controller"),
            ("sim mock", _sim_mock(), motion.MotionRoute.UNPLANNED, "mock_mode"),
            ("ur ik", _ur_ik(), motion.MotionRoute.REFUSED, "'ik'"),
            ("kuka", _kuka(), motion.MotionRoute.REFUSED, "KukaRobotArm"),
            ("sim ik", sim_ik, motion.MotionRoute.REFUSED, "'ik'"),
            ("ur curobo, no path guard", _ur_curobo(), motion.MotionRoute.REFUSED, "no self_collision guard"),
            ("a caller's teleport on a real controller", _Claims(), motion.MotionRoute.REFUSED, "nothing plans"),
            ("an object", object(), motion.MotionRoute.REFUSED, "object does not say"),
        ]
        for label, arm, route, words in rows:
            with self.subTest(label):
                reading = motion.route_of(arm)
                self.assertIs(route, reading.route, reading.reason)
                self.assertIn(words, reading.reason)
        with _judged():
            judged = motion.route_of(_ur_curobo())
        self.assertIs(motion.MotionRoute.PLANNED, judged.route, judged.reason)
        self.assertIn("exact mesh guard", judged.reason)

    def test_the_robot_reads_its_route(self) -> None:
        motion = _motion()
        self.assertIs(motion.MotionRoute.UNPLANNED, _desk().route().route)


# ---------------------------------------------------------------------------------------------------
# The verbs on a desk arm
# ---------------------------------------------------------------------------------------------------


class ADeskArmMovesTests(unittest.TestCase):

    def test_a_desk_arm_moves_and_says_unplanned(self) -> None:
        motion = _motion()
        robot = _desk()
        target = _pose()

        report = robot.move(target)

        self.assertIs(motion.MotionOutcome.EXECUTED, report.outcome, report.render())
        self.assertTrue(report.ok)
        self.assertIs(motion.MotionRoute.UNPLANNED, report.route.route)
        self.assertIs(CameraWorldUse.UNPLANNED, report.camera_world.use)
        self.assertIs(target, robot.arm.get_tcp_pose())
        self.assertIsNone(report.line, "a motion that was not a line carries no line reading")

    def test_a_line_carries_what_the_arm_keeps_of_it(self) -> None:
        from src.robot.core.arm_capabilities import LineMotion

        report = _desk().move(_pose(), linear=True)

        assert report.line is not None
        self.assertIs(LineMotion.TELEPORT, report.line.motion)

    def test_vel_and_acc_reach_the_arm_only_when_chosen(self) -> None:
        arm = _Keywords()
        robot = _desk(arm)

        robot.move(_pose())
        robot.move(_pose(), vel=0.1, acc=0.2)

        self.assertEqual([{"linear": False}, {"linear": False, "vel": 0.1, "acc": 0.2}], arm.calls)

    def test_joints_and_home_on_a_desk_arm(self) -> None:
        motion = _motion()
        robot = _desk()

        joints = robot.move_joints([0.1] * 6)
        self.assertEqual([0.1] * 6, [float(v) for v in robot.arm.get_joint_positions().values])
        home = robot.home()

        self.assertIs(motion.MotionOutcome.EXECUTED, joints.outcome, joints.render())
        self.assertIs(motion.MotionOutcome.EXECUTED, home.outcome, home.render())
        self.assertIs(CameraWorldUse.UNPLANNED, home.camera_world.use, "a desk arm's home says UNPLANNED")
        self.assertIn("desk arm", home.camera_world.reason)


# ---------------------------------------------------------------------------------------------------
# An arm that plans nothing where a real arm would move
# ---------------------------------------------------------------------------------------------------


class AnArmThatPlansNothingIsRefusedTests(unittest.TestCase):
    """The arm verbs of a UR on ik and a KUKA drive unplanned, so the robot verbs refuse them before any command."""

    def test_a_ur_on_ik_commands_nothing(self) -> None:
        motion = _motion()
        arm = _ur_ik()
        wires = _tripwires(arm)
        robot = Robot.from_parts(arm=arm, gripper=None, lock_key=None)

        for report in _every_verb(robot):
            with self.subTest(report.verb.value):
                self.assertIs(motion.MotionOutcome.REFUSED, report.outcome, report.render())
                self.assertIs(MotionStatus.UNSUPPORTED, report.status)
                self.assertIs(motion.MotionRoute.REFUSED, report.route.route)
                self.assertIn("'ik'", report.message)
        for wire in wires.values():
            wire.assert_not_called()

    def test_a_kuka_commands_nothing(self) -> None:
        motion = _motion()
        arm = _kuka()
        wires = _tripwires(arm)
        robot = Robot.from_parts(arm=arm, gripper=None, lock_key=None)

        for report in _every_verb(robot, decline=_BENCH):
            with self.subTest(report.verb.value):
                self.assertIs(motion.MotionOutcome.REFUSED, report.outcome, report.render())
                self.assertIn("KukaRobotArm", report.message)
        for wire in wires.values():
            wire.assert_not_called()

    def test_a_curobo_arm_whose_paths_cannot_be_judged_commands_nothing(self) -> None:
        """Declined, so the camera world lets it through; the route stops it, because Coal would judge no path."""
        motion = _motion()
        arm = _ur_curobo()
        wires = _tripwires(arm)
        robot = Robot.from_parts(arm=arm, gripper=None, lock_key=None)

        for report in _every_verb(robot, decline=_BENCH):
            with self.subTest(report.verb.value):
                self.assertIs(motion.MotionOutcome.REFUSED, report.outcome, report.render())
                self.assertIn("no self_collision guard", report.message)
        for wire in wires.values():
            wire.assert_not_called()

    def test_a_ur_on_ik_pick_commands_nothing(self) -> None:
        """The same route stands before a pick, so a UR on ik is refused there too."""
        from src.robot.execution.handling import HandlingOutcome

        hand = _Hand(hold=HoldEvidence.HELD)
        robot = Robot.from_parts(arm=_ur_ik(), gripper=hand, lock_key=None)

        report = robot.pick(_pick_pose(), 40.0)

        self.assertIs(HandlingOutcome.REFUSED, report.outcome, report.render())
        self.assertIn("'ik'", report.message)
        self.assertEqual([], hand.commands)
        self.assertEqual((), report.poses)


# ---------------------------------------------------------------------------------------------------
# The camera world on a cuRobo arm
# ---------------------------------------------------------------------------------------------------


class TheCameraWorldStandsBehindEveryMotionTests(unittest.TestCase):

    def test_no_world_and_no_decline_commands_nothing(self) -> None:
        motion = _motion()
        arm = _ur_curobo()
        wires = _tripwires(arm)
        robot = Robot.from_parts(arm=arm, gripper=None, lock_key=None)

        with _judged():
            reports = _every_verb(robot)

        for report in reports:
            with self.subTest(report.verb.value):
                self.assertIs(motion.MotionOutcome.REFUSED, report.outcome, report.render())
                self.assertEqual(NO_CAMERA_WORLD_MESSAGE, report.message)
                self.assertIs(CameraWorldUse.MISSING, report.camera_world.use)
                self.assertIs(motion.MotionRoute.PLANNED, report.route.route)
        for wire in wires.values():
            wire.assert_not_called()

    def test_a_declined_motion_runs_and_says_declined(self) -> None:
        motion = _motion()
        robot = Robot.from_parts(arm=_ur_curobo(), gripper=None, lock_key=None)
        declined = CameraWorldStamp.declined(CameraWorldDecline(_BENCH))

        with _judged():
            reports = [robot.move(_pose(), decline=_BENCH), robot.move_joints(_JOINTS, decline=_BENCH)]

        for report in reports:
            with self.subTest(report.verb.value):
                self.assertIs(motion.MotionOutcome.EXECUTED, report.outcome, report.render())
                self.assertEqual(declined, report.camera_world)

    def test_a_decline_on_a_wired_world_commands_nothing(self) -> None:
        motion = _motion()
        arm = _ur_curobo(live_world=object())
        wires = _tripwires(arm)
        robot = Robot.from_parts(arm=arm, gripper=None, lock_key=None)

        with _judged():
            reports = _every_verb(robot, decline=_BENCH)

        for report in reports:
            with self.subTest(report.verb.value):
                self.assertIs(motion.MotionOutcome.REFUSED, report.outcome, report.render())
                self.assertEqual(DECLINE_ON_A_LIVE_WORLD_MESSAGE, report.message)
        for wire in wires.values():
            wire.assert_not_called()

    def test_home_declines_for_the_arm_while_its_move_home_runs(self) -> None:
        """``move_home`` takes no keyword, so the decline reaches it as a block around the call, and ends with it."""
        motion = _motion()
        arm = _ur_curobo()
        seen: list[Any] = []

        def home() -> bool:
            seen.append(active_decline(arm))
            return True

        arm.move_home = home  # type: ignore[method-assign]
        robot = Robot.from_parts(arm=arm, gripper=None, lock_key=None)

        with _judged():
            report = robot.home(decline=_BENCH)

        self.assertIs(motion.MotionOutcome.EXECUTED, report.outcome, report.render())
        self.assertEqual([CameraWorldDecline(_BENCH)], seen)
        self.assertIs(CameraWorldUse.DECLINED, report.camera_world.use)
        self.assertIsNone(active_decline(arm), "the decline outlived the home")

    def test_the_block_declines_the_verbs_inside_it(self) -> None:
        motion = _motion()
        robot = Robot.from_parts(arm=_ur_curobo(), gripper=None, lock_key=None)

        with _judged(), robot.without_camera_world(_BENCH):
            report = robot.move(_pose())

        self.assertIs(motion.MotionOutcome.EXECUTED, report.outcome, report.render())
        self.assertIs(CameraWorldUse.DECLINED, report.camera_world.use)


# ---------------------------------------------------------------------------------------------------
# decline="reason" on every verb that moves
# ---------------------------------------------------------------------------------------------------


class DeclineIsAReasonTests(unittest.TestCase):

    def test_every_verb_that_moves_takes_decline_as_a_keyword(self) -> None:
        for name in ("move", "move_joints", "home", "pick", "place"):
            with self.subTest(name):
                parameter = inspect.signature(getattr(Robot, name)).parameters["decline"]
                self.assertIs(inspect.Parameter.KEYWORD_ONLY, parameter.kind)
                self.assertIs(UNSET, parameter.default)

    def test_a_pick_carries_the_reason_to_every_motion(self) -> None:
        log = _Log()
        robot = _robot(log)

        robot.pick(_pick_pose(), 40.0, decline=_BENCH)
        robot.gripper.hold = HoldEvidence.EMPTY  # type: ignore[union-attr]  # the part left the jaws, so the place backs out
        robot.place(_pick_pose(), decline=_BENCH)

        declines = [entry[3] for entry in log.entries if entry[0] == "move"]
        self.assertEqual(6, len(declines))
        self.assertTrue(all(decline == CameraWorldDecline(_BENCH) for decline in declines), declines)

    def test_the_old_keyword_is_an_alias_and_both_at_once_raise(self) -> None:
        """Both at once raise for being given twice, not for an unexpected keyword."""
        from src.robot.execution.handling import HandlingOutcome

        log = _Log()
        robot = _robot(log)

        self.assertIs(HandlingOutcome.EXECUTED,
                      robot.pick(_pick_pose(), 40.0, camera_world=CameraWorldDecline(_BENCH)).outcome)
        with self.assertRaisesRegex(TypeError, "not both"):
            robot.pick(_pick_pose(), 40.0, decline=_BENCH, camera_world=CameraWorldDecline(_BENCH))

    def test_a_decline_that_is_no_reason_raises_before_anything_moves(self) -> None:
        """A programmer's error raises."""
        arm = DummyRobotArm()
        wires = _tripwires(arm)
        robot = _desk(arm)

        with self.assertRaises(ValueError):
            robot.move(_pose(), decline="   ")
        with self.assertRaisesRegex(TypeError, "reason as text"):
            robot.home(decline=CameraWorldDecline(_BENCH))  # type: ignore[arg-type]
        for wire in wires.values():
            wire.assert_not_called()


# ---------------------------------------------------------------------------------------------------
# A refusal is a report
# ---------------------------------------------------------------------------------------------------


class ARefusalIsAReportTests(unittest.TestCase):

    def test_outside_connected_is_refused(self) -> None:
        motion = _motion()
        robot = Robot.from_parts(arm=DummyRobotArm(), gripper=None, lock_key=None)

        for report in _every_verb(robot):
            with self.subTest(report.verb.value):
                self.assertIs(motion.MotionOutcome.REFUSED, report.outcome)
                self.assertIs(MotionStatus.CONNECTION_ERROR, report.status)
                self.assertIn("robot.connected()", report.message)

    def test_a_pose_not_in_base_is_refused(self) -> None:
        motion = _motion()
        arm = DummyRobotArm()
        wires = _tripwires(arm)

        report = _desk(arm).move(_pose(Frame.TOOL))

        self.assertIs(motion.MotionOutcome.REFUSED, report.outcome)
        self.assertIs(MotionStatus.INVALID_TARGET, report.status)
        wires["move"].assert_not_called()

    def test_a_camera_that_cannot_vouch_ends_a_move_as_its_own_outcome(self) -> None:
        motion = _motion()

        report = _desk(_SilentCamera()).move(_pose())

        self.assertIs(motion.MotionOutcome.CAMERA_WORLD_UNAVAILABLE, report.outcome, report.render())
        self.assertIn("'overhead'", report.message)
        self.assertIsInstance(report.result.exception, CameraWorldUnavailable)
        self.assertIn("look at the camera", report.render())

    def test_a_camera_that_cannot_vouch_ends_a_place_as_its_own_outcome(self) -> None:
        from src.robot.execution.handling import HandlingOutcome

        log = _Log()
        robot = _robot(log, hold=HoldEvidence.EMPTY, raise_on=0)

        report = robot.place(_pick_pose())

        self.assertIs(HandlingOutcome.CAMERA_WORLD_UNAVAILABLE, report.outcome, report.render())
        self.assertEqual([], [entry for entry in log.entries if entry[0] == "set_width"])

    def test_a_driver_fault_is_a_report(self) -> None:
        motion = _motion()
        refusal = MotionResult.failed(MotionStatus.JOINT_LIMIT_REJECTED, MotionCommand.MOVE_JOINTS,
                                      message="joint 3 is past its limit")

        dropped = _desk(_Raises(RobotConnectionError("rtde link down"))).move_joints(_JOINTS)
        rejected = _desk(_Raises(RobotMotionRejected("refused", result=refusal))).move_joints(_JOINTS)

        self.assertIs(motion.MotionOutcome.MOTION_REFUSED, dropped.outcome)
        self.assertIs(MotionStatus.CONNECTION_ERROR, dropped.status)
        self.assertEqual("RobotConnectionError: rtde link down", dropped.message)
        self.assertIs(motion.MotionOutcome.MOTION_REFUSED, rejected.outcome)
        self.assertIs(refusal, rejected.result, "the typed refusal the raise carried is the report's result")

    def test_a_home_the_arm_refuses_reads_unknown(self) -> None:
        motion = _motion()

        report = _desk(_RefusesHome()).home()

        self.assertIs(motion.MotionOutcome.MOTION_REFUSED, report.outcome)
        self.assertIs(MotionStatus.UNKNOWN, report.status)
        self.assertIn("move_home", report.message)

    def test_an_arm_that_breaks_the_contract_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, "not a MotionResult"):
            _desk(_Untyped()).move(_pose())


# ---------------------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------------------


class TheMotionReportTests(unittest.TestCase):

    def test_both_halves_on_every_outcome(self) -> None:
        reports = [
            _desk().move(_pose(), linear=True),
            Robot.from_parts(arm=DummyRobotArm(), gripper=None, lock_key=None).home(),
            _desk(_SilentCamera()).move(_pose()),
            _desk(_RefusesHome()).home(),
        ]
        self.assertEqual({"executed", "refused", "camera_world_unavailable", "motion_refused"},
                         {report.outcome.value for report in reports})
        for report in reports:
            with self.subTest(report.outcome.value):
                text = report.render()
                self.assertTrue(text.isascii())
                self.assertFalse(text.endswith("\n"))
                self.assertTrue(text.startswith(f"{report.verb.value}  {report.outcome.value.upper()}"))
                self.assertIn("route  ", text)
                data = json.loads(json.dumps(report.to_dict()))
                self.assertEqual(report.outcome.value, data["outcome"])
                self.assertEqual(report.ok, data["ok"])
                self.assertEqual(report.status.value, data["result"]["status"])
                self.assertEqual(report.camera_world.use.value, data["result"]["camera_world"]["use"])
                self.assertEqual(report.route.route.value, data["route"]["route"])
        self.assertIn("line  TELEPORT", reports[0].render())
        self.assertEqual("teleport", reports[0].to_dict()["line"]["motion"])


class MovingARobotLoadsNoPickServiceTests(unittest.TestCase):

    def test_moving_a_robot_loads_no_grasping_stack(self) -> None:
        script = "\n".join((
            "from src.config.schema.robot import RobotConfig",
            "from src.robot.execution.robot import Robot",
            "robot = Robot.from_config(RobotConfig(vendor='dummy', gripper={'vendor': 'dummy'}))",
            "with robot.connected():",
            "    assert robot.move(robot.arm.get_tcp_pose()).ok",
            "    assert robot.home().ok",
            "robot.render()",
        ))
        self.assertEqual(_loaded_after(script, _ROBOT_MUST_NOT_LOAD), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
