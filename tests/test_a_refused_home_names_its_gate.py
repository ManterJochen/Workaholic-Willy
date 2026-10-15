"""A home the arm refuses says which gate refused it, and a refusal from the planner or the controller keeps its kind.

``Robot.home()`` read ``move_home``'s bool, so every refusal on a UR became UNKNOWN with one fixed sentence, and the gate
that refused was only in ``ur_arm.log``. Reproduced on a UR10 profile before this change: a home straight up (the box),
a home written in degrees (joint limits), a ``moveJ`` the controller did not complete, and a planner refusal all read
UNKNOWN. The UR driver now goes home as a typed verb (``HomesTyped.move_to_home``), the facade passes its result
through, and ``move_home`` is that result's ``ok``.

Three refusals that were flattened on the way are typed here as well: a ``moveJ`` that failed reads
CONTROLLER_REJECTED (not UNKNOWN) and says it had been sent; the planner's own kind of refusal chooses the status
(a joint limit is not a self collision); and a planner that will not start from where the arm stands is not "no plan".

The judged configuration is the sent one: one ``JointPositions`` object reaches every gate and the drive.

The home is decided in ``PlannedArm``, so since P0g slice 13 these run on the kit's planned arm over the fake transport
(``tests/_arm_kit.planned_arm``), with the real guards: a joint move is a ``run_trajectory`` on the fake's tape, the
home's pose is read through ``controller_fk``, and a stopped controller is a status the fake reports.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.robot.core import (
    NO_PLAN_FAIL_SAFE_MESSAGE,
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotArm,
    RobotMode,
    RobotStatus,
    SafetyMode,
)
from src.robot.core.arm_capabilities import HomesTyped
from src.robot.core.camera_world import CameraWorldUse
from src.robot.core.errors import RobotMotionRejected
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.drivers.planned.arm import PlannedArm
from src.robot.execution.robot import Robot
from src.robot.safety._fcl_self_collision import mesh_backend_status
from src.robot.safety.planning import JointCheckVerdict, StateRefusal, StateRefusalKind, StateWhere
from tests._arm_kit import bare_arm, controller_pose, kit_pose, kit_tree, planned_arm
from tests._fake_transport import Raises, fake_transport

_BENCH = "bench check, no cameras mounted"

#: Clear, and where the fake controller says the arm is standing (tests/test_a_joint_move_is_judged_as_a_path.py).
_HERE = (0.0, -1.5, 1.5, 0.0, 0.0, 0.0)
#: Clear, a short joint move away. The home in most tests below.
_THERE = (0.2, -1.5, 1.5, 0.0, 0.0, 0.0)
#: A home copied off the pendant in degrees.
_DEGREES = (0.0, -90.0, 90.0, -90.0, -90.0, 0.0)
#: What the fake controller's FK answers for a home inside the shipped box: (400, 0, 300) mm, tool down.
_INSIDE = [0.4, 0.0, 0.3, 0.0, 3.14159, 0.0]
#: A flange 1.4 m up: the grasp centre of an arm standing straight up, outside the shipped z_max of 500 mm.
_ABOVE = [0.0, -0.4, 1.4, 0.0, 3.14159, 0.0]
#: The Isaac cell's tool frame, 132 mm along flange +Y: the frame the other UR fixtures carry.
_TOOL = {"source": "willy", "offset_mm": (0.0, 132.0, 0.0),
         "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)}


def _mesh_backend_available() -> bool:
    return mesh_backend_status("ur5e") == "ok"


class _Planner:
    """Answers ``check_joint_path`` with ``verdict``, a pass when it is ``None``, and records what it was asked.

    It plans nothing: a line it refuses is planned around, and there is no plan around it, so the line's refusal is
    the move's.
    """

    def __init__(self, verdict: JointCheckVerdict | None = None) -> None:
        self._verdict = verdict
        self.checked: list[list[float]] | None = None
        self.last_refusal: StateRefusal | None = None

    def check_joint_path(self, samples: Any, *, refresh: bool = True, clearance_mm: float = 0.0) -> JointCheckVerdict:
        self.checked = [list(s) for s in samples]
        if self._verdict is None:
            return JointCheckVerdict(valid=True, first_invalid=None, checked=len(self.checked), reason="clear")
        return self._verdict

    def plan_joint(self, goal: Any, *, refresh: bool = True, **_: Any) -> None:
        return None


def _arm(*, home: tuple[float, ...] | None = _THERE, planner: _Planner | None = None,
         motion_planner: str = "curobo") -> Any:
    """A planned arm with the real guards on a fake controller that stands at ``_HERE``, completes every moveJ and
    reports the home ``_INSIDE`` the box."""
    blocks: dict[str, Any] = {"safety": {"payload": {"enforce": False}}, "gripper": {"tool_frame": _TOOL}}
    if home is not None:
        blocks["home_joint_positions"] = home
    fake = fake_transport(joints=_HERE)
    fake.always("controller_fk", controller_pose(_INSIDE))
    arm: Any = planned_arm(kit_tree(motion_planner, **blocks), transport=fake)
    if motion_planner == "curobo":
        arm._curobo = planner if planner is not None else _Planner()
    return arm


def _fake(arm: PlannedArm) -> Any:
    return arm._transport


def _moves(arm: PlannedArm) -> list[Any]:
    """Every joint move the arm sent: one ``run_trajectory`` each, its one waypoint the goal."""
    return _fake(arm).calls("run_trajectory")  # type: ignore[no-any-return]


def _assert_nothing_sent(test: unittest.TestCase, arm: PlannedArm, *, anchor: str = "controller_fk") -> None:
    """No joint move, on a tape that holds what the refused home read first (the anchor)."""
    test.assertTrue(_fake(arm).calls(anchor), f"the fake recorded no {anchor}, so 'nothing was sent' reads nothing")
    test.assertEqual([], _moves(arm))


def _stopped(arm: PlannedArm) -> None:
    """The fake controller reports a protective stop, as the measured URSim did when moveJ came back False."""
    _fake(arm).always("read_status", RobotStatus(
        robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.PROTECTIVE_STOP, protective_stopped=True,
        emergency_stopped=False, message="Safetystatus: PROTECTIVE_STOP", program_running=True))


def _home(arm: Any) -> Any:
    return Robot.from_parts(arm=arm, gripper=None, lock_key=None).home(decline=_BENCH)


def _refusal(kind: StateRefusalKind, *, where: StateWhere = StateWhere.GOAL, **names: Any) -> StateRefusal:
    return StateRefusal(where=where, kind=kind, joints=_THERE, depth_mm=4.2, **names)


def _refused_by(refusal: StateRefusal | None) -> JointCheckVerdict:
    return JointCheckVerdict(valid=False, first_invalid=3, checked=27,
                             reason="the cuRobo check refuses sample 3 of 27, counted from 0", refusal=refusal)


# ---------------------------------------------------------------------------------------------------
# Which arm goes home as a typed verb
# ---------------------------------------------------------------------------------------------------


class WhichArmGoesHomeTypedTests(unittest.TestCase):

    def test_a_planned_arm_goes_home_typed_and_the_protocol_names_no_such_member(self) -> None:
        """A member on ``RobotArm`` would reach every driver that subclasses it as a stub that returns ``None``."""
        self.assertIsInstance(_arm(), HomesTyped)
        self.assertNotIsInstance(DummyRobotArm(), HomesTyped)
        self.assertFalse(hasattr(RobotArm, "move_to_home"))

    def test_an_arm_that_answers_only_with_a_bool_still_reads_unknown(self) -> None:
        class _Refuses(DummyRobotArm):
            def move_home(self) -> bool:
                return False

        arm = _Refuses()
        arm.connect()

        report = Robot.from_parts(arm=arm, gripper=None, lock_key=None).home()

        self.assertIs(MotionStatus.UNKNOWN, report.status, report.render())
        self.assertIn("move_home answers only with False", report.message)


# ---------------------------------------------------------------------------------------------------
# Robot.home says which gate refused
# ---------------------------------------------------------------------------------------------------


class RobotHomeNamesTheGateTests(unittest.TestCase):
    """Every refusal reproduced on the UR10 profile, through ``Robot.home`` and the real guards."""

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_a_clear_home_executes_and_the_home_judged_is_the_home_sent(self) -> None:
        planner = _Planner()
        arm = _arm(planner=planner)

        report = _home(arm)

        self.assertTrue(report.ok, report.render())
        self.assertIs(MotionCommand.MOVE_HOME, report.result.command)
        self.assertIs(CameraWorldUse.DECLINED, report.camera_world.use)
        (move,) = _moves(arm)
        assert planner.checked is not None
        np.testing.assert_allclose(move.args[0], [planner.checked[-1]], atol=1e-12)
        np.testing.assert_allclose(move.args[0], [_THERE], atol=1e-12)

    def test_a_home_outside_the_box_is_workspace_rejected_and_names_the_box(self) -> None:
        planner = _Planner()
        arm = _arm(planner=planner)
        _fake(arm).always("controller_fk", controller_pose(_ABOVE))

        report = _home(arm)

        self.assertIs(MotionStatus.WORKSPACE_REJECTED, report.status, report.render())
        self.assertIs(MotionCommand.MOVE_HOME, report.result.command)
        self.assertIn("the home grasp centre (", report.message)
        self.assertIn("workspace_limits (x -500.0 to 500.0, y -500.0 to 500.0, z 0.0 to 500.0 mm)", report.message)
        self.assertIn("plus the declared tool frame, and not the flange itself", report.message)
        self.assertIn("in radians", report.message)
        _assert_nothing_sent(self, arm)
        self.assertIsNone(planner.checked, "a home the box refused was judged as a path")

    def test_the_shipped_default_home_says_it_is_the_default(self) -> None:
        arm = _arm(home=None)
        _fake(arm).always("controller_fk", controller_pose(_ABOVE))

        report = _home(arm)

        self.assertIs(MotionStatus.WORKSPACE_REJECTED, report.status, report.render())
        self.assertIn("No robot.home_joint_positions is set", report.message)
        np.testing.assert_allclose(report.result.target_joints.values, (0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0))

    def test_a_home_in_degrees_is_joint_limit_rejected_and_says_degrees(self) -> None:
        arm = _arm(home=_DEGREES)

        report = _home(arm)

        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, report.status, report.render())
        self.assertIs(MotionCommand.MOVE_HOME, report.result.command, "the destination gate's label is relabelled")
        self.assertIn("[safety:joint_limit/joint_limit] axis 1", report.message)
        self.assertIn("reads as degrees", report.message)
        self.assertEqual([], _fake(arm).calls("controller_fk"), "a home in degrees was boxed")
        _assert_nothing_sent(self, arm, anchor="wait_until_steady")

    def test_a_home_the_controller_cannot_place_is_controller_rejected(self) -> None:
        arm = _arm()
        _fake(arm).always("controller_fk", RuntimeError("controller dropped"))

        report = _home(arm)

        self.assertIs(MotionStatus.CONTROLLER_REJECTED, report.status, report.render())
        self.assertIn("could not check the home pose", report.message)
        self.assertIn("RuntimeError: controller dropped", report.message)
        _assert_nothing_sent(self, arm)

    def test_a_joint_read_that_raises_is_a_connection_error_not_a_raise(self) -> None:
        """A transport's joint read may raise ``RuntimeError`` (the UR's does), which is not a robot error the facade
        catches."""
        arm = _arm()
        _fake(arm).always("read_joints", RuntimeError("Not connected: call connect() first."))

        report = _home(arm)

        self.assertIs(MotionStatus.CONNECTION_ERROR, report.status, report.render())
        self.assertIn("joint positions could not be read", report.message)
        self.assertIn("nothing was sent", report.message)
        _assert_nothing_sent(self, arm, anchor="read_joints")

    def test_a_movej_the_controller_did_not_complete_says_it_was_sent_and_why(self) -> None:
        arm = _arm()
        _fake(arm).always("run_trajectory", False)
        _stopped(arm)

        report = _home(arm)

        self.assertIs(MotionStatus.CONTROLLER_REJECTED, report.status, report.render())
        self.assertIn("moveJ was sent", report.message)
        self.assertIn("may have moved part of the way", report.message)
        self.assertIn("safety mode protective_stop, with a protective stop active", report.message)
        self.assertIn("Safetystatus: PROTECTIVE_STOP", report.message)
        self.assertEqual(1, len(_moves(arm)))

    def test_a_movej_that_raises_says_it_was_sent(self) -> None:
        arm = _arm()
        _fake(arm).always("run_trajectory", Raises(RuntimeError("RTDE control script is not running")))

        report = _home(arm)

        self.assertIs(MotionStatus.CONNECTION_ERROR, report.status, report.render())
        self.assertIn("moveJ was sent and raised", report.message)
        self.assertIn("RTDE control script is not running", report.message)

    def test_the_planner_s_kind_of_refusal_chooses_the_status(self) -> None:
        cases = (
            (_refusal(StateRefusalKind.JOINT_LIMIT), MotionStatus.JOINT_LIMIT_REJECTED,
             "a joint sits outside the limits the planner was built with"),
            (_refusal(StateRefusalKind.SELF_COLLISION, link_a="forearm_link", link_b="wrist_2_link"),
             MotionStatus.SELF_COLLISION_REJECTED, "forearm_link and wrist_2_link overlap by 4.2 mm"),
            (_refusal(StateRefusalKind.WORLD), MotionStatus.SELF_COLLISION_REJECTED,
             "into the world the planner holds"),
            (None, MotionStatus.SELF_COLLISION_REJECTED, "the planner refused this straight joint line"),
        )
        for refusal, status, said in cases:
            with self.subTest(kind=None if refusal is None else refusal.kind.value):
                arm = _arm(planner=_Planner(_refused_by(refusal)))

                report = _home(arm)

                self.assertIs(status, report.status, report.render())
                self.assertIn("sample 3 of 27", report.message)
                self.assertIn(said, report.message)
                _assert_nothing_sent(self, arm)

    def test_an_arm_standing_inside_a_guard_s_margin_is_refused_at_sample_1(self) -> None:
        """Pinned as it stands: the path gate judges where the arm is, so an arm inside a margin cannot be homed."""
        arm = _arm()
        _fake(arm).joints = [0.0, -1.5, 1.5, 0.0, 0.0, float(np.radians(357.0))]

        report = _home(arm)

        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, report.status, report.render())
        self.assertIn("sample 1 of", report.message)
        _assert_nothing_sent(self, arm)


# ---------------------------------------------------------------------------------------------------
# move_home is the typed verb's ok
# ---------------------------------------------------------------------------------------------------


class TheBoolIsTheTypedVerbsOkTests(unittest.TestCase):

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def _cases(self) -> dict[str, Any]:
        def box(arm: PlannedArm) -> None:
            _fake(arm).always("controller_fk", controller_pose(_ABOVE))

        def fk(arm: PlannedArm) -> None:
            _fake(arm).always("controller_fk", RuntimeError("controller dropped"))

        def movej(arm: PlannedArm) -> None:
            _fake(arm).always("run_trajectory", False)

        return {"box": (box, MotionStatus.WORKSPACE_REJECTED), "fk": (fk, MotionStatus.CONTROLLER_REJECTED),
                "moveJ": (movej, MotionStatus.CONTROLLER_REJECTED)}

    def test_each_refusal_is_false_on_both_verbs_with_one_status(self) -> None:
        for label, (arrange, status) in self._cases().items():
            with self.subTest(label):
                arms = [_arm(), _arm()]
                for arm in arms:
                    arrange(arm)
                    self.enterContext(arm.without_camera_world(_BENCH))
                typed = arms[0].move_to_home()
                self.assertIs(status, typed.status, typed.render())
                self.assertFalse(arms[1].move_home())

    def test_a_clear_home_is_true_on_both_verbs(self) -> None:
        arms = [_arm(), _arm()]
        for arm in arms:
            self.enterContext(arm.without_camera_world(_BENCH))
        self.assertTrue(arms[0].move_to_home().ok)
        self.assertTrue(arms[1].move_home())
        for arm in arms:
            self.assertEqual(1, len(_moves(arm)))

    def test_a_degrees_home_is_false_and_reads_joint_limit(self) -> None:
        arm = _arm(home=_DEGREES)
        self.enterContext(arm.without_camera_world(_BENCH))
        self.assertFalse(arm.move_home())
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, arm.move_to_home().status)


class OneHomeIsJudgedAndSentTests(unittest.TestCase):
    """The configuration the gates judge is the object the drive sends, not a second one built from the same list."""

    def test_every_gate_and_the_drive_see_the_same_object(self) -> None:
        arm = _arm()
        self.enterContext(arm.without_camera_world(_BENCH))
        seen: list[tuple[str, Any]] = []
        preflight = MagicMock()
        preflight.gate_joint_target.side_effect = lambda joints, **_: seen.append(("destination", joints))
        arm._preflight = preflight
        arm._judge_joint_move = (  # type: ignore[method-assign]
            lambda joints, *, command: (seen.append(("path", joints)), (joints, None, None))[1])
        arm._drive_joints = lambda joints, **_: seen.append(("drive", joints))  # type: ignore[method-assign]

        result = arm.move_to_home()

        self.assertTrue(result.ok, result.render())
        self.assertEqual(["destination", "path", "drive"], [name for name, _ in seen])
        first = seen[0][1]
        for name, joints in seen:
            self.assertIs(first, joints, f"{name} saw another JointPositions than the destination gate")
        self.assertIs(first, result.target_joints)
        np.testing.assert_allclose(_fake(arm).calls("controller_fk")[-1].args[0], first.values, atol=0.0)

    def test_a_destination_refusal_keeps_its_status_and_reads_move_home(self) -> None:
        arm = _arm(motion_planner="ik")
        arm._preflight = MagicMock()
        arm._preflight.gate_joint_target.return_value = MotionResult.failed(
            MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_JOINTS, message="test veto")

        result = arm.move_to_home()

        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status)
        self.assertIs(MotionCommand.MOVE_HOME, result.command)
        self.assertEqual("test veto", result.message)
        _assert_nothing_sent(self, arm, anchor="wait_until_steady")


# ---------------------------------------------------------------------------------------------------
# A joint move whose moveJ failed is typed too
# ---------------------------------------------------------------------------------------------------


class AFailedMoveJIsTypedTests(unittest.TestCase):

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_robot_move_joints_reads_controller_rejected_not_unknown(self) -> None:
        arm = _arm()
        _fake(arm).always("run_trajectory", False)

        report = Robot.from_parts(arm=arm, gripper=None, lock_key=None).move_joints(_THERE, decline=_BENCH)

        self.assertIs(MotionStatus.CONTROLLER_REJECTED, report.status, report.render())
        self.assertIs(MotionCommand.MOVE_JOINTS, report.result.command)
        self.assertIn("moveJ was sent", report.message)
        self.assertIs(CameraWorldUse.DECLINED, report.camera_world.use, "the typed result is stamped like any other")

    def test_the_raise_carries_the_same_typed_result(self) -> None:
        """What the raising ``move_joint`` carried until 2026-09-19, ``raise_unless_executed`` carries now."""
        arm = _arm()
        _fake(arm).always("run_trajectory", False)
        self.enterContext(arm.without_camera_world(_BENCH))

        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_to_joints(JointPositions(_THERE)).raise_unless_executed()

        result = caught.exception.result
        assert isinstance(result, MotionResult)
        self.assertIs(MotionStatus.CONTROLLER_REJECTED, result.status)
        self.assertIn("Driver reported moveJ() failure", str(caught.exception))

    def test_a_joint_read_that_raises_is_a_connection_error_on_move_joints(self) -> None:
        arm = _arm()
        _fake(arm).always("read_joints", OSError("socket closed"))

        report = Robot.from_parts(arm=arm, gripper=None, lock_key=None).move_joints(_THERE, decline=_BENCH)

        self.assertIs(MotionStatus.CONNECTION_ERROR, report.status, report.render())
        self.assertIn("OSError: socket closed", report.message)


class AWrongNumberOfJointsIsAnInvalidTargetTests(unittest.TestCase):

    def test_the_drive_refuses_it_before_movej_as_invalid_target(self) -> None:
        arm = _arm(motion_planner="ik")
        arm._preflight = None

        result = arm.move_to_joints(JointPositions((0.0, -1.5, 1.5, 0.0, 0.0)))

        self.assertIs(MotionStatus.INVALID_TARGET, result.status, result.render())
        self.assertIn("moveJ was not sent", result.message)
        _assert_nothing_sent(self, arm, anchor="wait_until_steady")


# ---------------------------------------------------------------------------------------------------
# The planner's kind on a line, and a start the planner will not leave
# ---------------------------------------------------------------------------------------------------


class ALineRefusalKeepsTheKindTests(unittest.TestCase):

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_a_joint_limit_on_the_line_reads_joint_limit(self) -> None:
        from tests._arm_kit import RecordingPlanner, line_arm
        from tests.test_a_checked_line_is_sampled_and_solved import _pose as _line_pose

        for refusal, status in ((_refusal(StateRefusalKind.JOINT_LIMIT), MotionStatus.JOINT_LIMIT_REJECTED),
                                (_refusal(StateRefusalKind.WORLD), MotionStatus.SELF_COLLISION_REJECTED)):
            with self.subTest(kind=refusal.kind.value):
                arm = line_arm(planner=RecordingPlanner(_refused_by(refusal)))
                self.enterContext(arm.without_camera_world(_BENCH))

                result = arm.move(_line_pose(x=450.0), linear=True)

                self.assertIs(status, result.status, result.render())
                self.assertIn("the planner refused this line", result.message)
                self.assertIn(refusal.render(), result.message)
                fake: Any = arm._transport
                self.assertTrue(fake.calls("read_joints"), "the line never read where the arm stands")
                self.assertEqual([], fake.calls("move_pose"))

    def test_a_controller_refusal_below_the_judge_keeps_its_cause(self) -> None:
        from tests._arm_kit import line_arm
        from tests._fake_transport import Rejects
        from tests.test_a_checked_line_is_sampled_and_solved import _pose as _line_pose

        cases = ((MotionStatus.IK_QUALITY_REJECTED, "before moveL was sent"),
                 (MotionStatus.CONTROLLER_REJECTED, "sent as moveL"),
                 (None, "gave no cause"))
        for status, said in cases:
            with self.subTest(status=status):
                arm = line_arm()
                self.enterContext(arm.without_camera_world(_BENCH))
                fake: Any = arm._transport
                fake.script("move_pose", Rejects(status=status))

                result = arm.move(_line_pose(x=450.0), linear=True)

                self.assertIs(status or MotionStatus.CONTROLLER_REJECTED, result.status, result.render())
                self.assertIn(said, result.message)
                self.assertEqual(1, len(fake.calls("move_pose")), "the refusal was not the controller's")


class _NoPlan:
    """A cuRobo glue double that finds no plan and says why, as ``CuroboPlanner.last_refusal`` does.

    It passes every goal it screens and refuses every straight line, so each move asks it for a plan.
    """

    def __init__(self, refusal: StateRefusal | None) -> None:
        self._refusal = refusal
        self.last_refusal: StateRefusal | None = None

    def check_joint_path(self, samples: Any, *, refresh: bool = True, clearance_mm: float = 0.0) -> JointCheckVerdict:
        count = len(list(samples))
        if clearance_mm > 0.0:
            return JointCheckVerdict(valid=False, first_invalid=0, checked=count, reason="the line grazes the tote")
        return JointCheckVerdict(valid=True, first_invalid=None, checked=count, reason="clear")

    def plan_joint(self, goal: Any, *, refresh: bool = True, **_: Any) -> list[list[float]]:
        self.last_refusal = self._refusal
        return []

    def execute(self, *args: Any, **kwargs: Any) -> MotionResult:
        raise AssertionError("an empty plan was executed")


class AStartThePlannerWillNotLeaveTests(unittest.TestCase):
    """``_drive_curobo`` read every empty plan as TIMEOUT "no plan", including a refused start."""

    def _move(self, refusal: StateRefusal | None) -> MotionResult:
        arm: Any = bare_arm("curobo")
        arm._transport.joints = list(_HERE)  # where the straight line to the goal starts
        arm._curobo = _NoPlan(refusal)
        self.enterContext(arm.without_camera_world(_BENCH))
        return arm.move(kit_pose())  # type: ignore[no-any-return]

    def test_a_refused_start_reads_its_kind_and_says_where_the_arm_stands(self) -> None:
        for kind, status in ((StateRefusalKind.JOINT_LIMIT, MotionStatus.JOINT_LIMIT_REJECTED),
                             (StateRefusalKind.SELF_COLLISION, MotionStatus.SELF_COLLISION_REJECTED),
                             (StateRefusalKind.WORLD, MotionStatus.SELF_COLLISION_REJECTED)):
            with self.subTest(kind=kind.value):
                refusal = _refusal(kind, where=StateWhere.START)

                result = self._move(refusal)

                self.assertIs(status, result.status, result.render())
                self.assertIn("the arm stands where the planner will not start from", result.message)
                self.assertIn(refusal.render(), result.message)
                self.assertNotEqual(NO_PLAN_FAIL_SAFE_MESSAGE, result.message)

    def test_a_goal_it_cannot_reach_is_still_no_plan(self) -> None:
        """The recovery reads TIMEOUT plus this exact sentence as a refusal it may look again at; that one is kept."""
        for refusal in (_refusal(StateRefusalKind.SELF_COLLISION, where=StateWhere.GOAL), None):
            with self.subTest(refusal=None if refusal is None else refusal.where.value):
                result = self._move(refusal)

                self.assertIs(MotionStatus.TIMEOUT, result.status)
                self.assertEqual(NO_PLAN_FAIL_SAFE_MESSAGE, result.message)

    def test_a_refused_start_is_not_a_plan_refusal_the_recovery_looks_again_at(self) -> None:
        """What the change costs the pick recovery: a start refusal no longer maps to MOTION_PLAN_REFUSED (a rescan).

        A rescan plans again from the same configuration, which the planner refuses again.
        """
        from src.robot.execution.autonomous_grasp.service import _is_motion_plan_refusal

        start = self._move(_refusal(StateRefusalKind.SELF_COLLISION, where=StateWhere.START))
        goal = self._move(_refusal(StateRefusalKind.SELF_COLLISION, where=StateWhere.GOAL))

        self.assertFalse(_is_motion_plan_refusal(SimpleNamespace(motion_status=start.status,
                                                                 motion_message=start.message)))
        self.assertTrue(_is_motion_plan_refusal(SimpleNamespace(motion_status=goal.status,
                                                                motion_message=goal.message)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
