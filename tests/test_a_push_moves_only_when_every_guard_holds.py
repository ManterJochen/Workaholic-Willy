"""A push moves the arm only when every guard holds, runs its contact legs as the owner set them, and stops where the
arm is when anything fails after it moved.

The owner's cell: a UR10 CB3 with no force sensor, cuRobo with a camera world, and a Robotiq Hand-E driven as
``jaw_io`` single_toggle on tool DO0, where EVERY change of DO0 moves the jaws once and the program keeps the count.
The push (``nudge_target``, owner, 2026-09-29) pushes a failed part with the outer face of the leading finger, jaws
open, so it needs zero DO0 changes. ``execute_push`` drives one planned push, and here it runs on

* a real :class:`JawIOGripper` single_toggle on a tool-DO0 double (``_ToolDO0``) that logs every write, connected
  once with a person saying "open", and nobody asked again (``builtins.input`` fails the test);
* an arm double (``_Arm``) that judges lines (``LineMotion.CHECKED``), reports its controller, and logs every call to
  every motion verb, with a live camera world double (``_World``) that logs every offer and forget on the same log.

What is pinned (the synthesis' "Tests that must pin it" 1 to 3):

* zero calls to any motion verb, no keep-out offered and the toggle's count untouched when any precondition fails:
  the planner refused, the count says closed, the count is unknown, DO0 was switched by hand, the hand is not
  connected (never, or disconnected, or disconnected and switched at the pendant), a jaw read raises, the controller
  is stopped, unreadable or answers no status, no judged lines or a line reading that raises, no typed move, no live
  world, no part points, an offer of another part, no gripper, a gripper that cannot say, a leg faster than the
  owner allows or legs out of order, a plan that is not the shape of a push (above 50 mm, under 10 mm, a push leg
  longer than its push, a leg off its line, an approach tilted more than 5 deg), axes that make no tool pose, an arm
  not at rest before P0. Each is a ``refused_*`` outcome the recovery loop falls through on, and so is a live world
  that refuses the offer;
* the whole push: P0 a plain ``move`` like a grasp approach, then the four contact legs, each ``linear=True`` with an
  explicit speed at or below its cap (down, back, up 50 mm/s; the push 25 mm/s; 0.1 m/s^2), every motion inside
  the keep-out scope that holds the part and its swept travel;
* the toggle across the push: ``commands_sent`` unchanged, DO0 never written, nobody asked;
* after something moved: a refusal or a failure on leg N, for every N, and nothing is called after it; a
  protective stop mid-push, or as the last leg ends, ends as ``controller_not_operational`` and nothing calls
  ``recover_from_protective_stop``; an arm not at rest before a contact leg stops there; a P0 refused before
  anything was sent is still a refusal the loop falls through on, and a P0 that may have been commanded is a stop;
  the down leg refused before it was sent, the arm in the air at P0, is a refusal the loop falls through on too
  (``refused_down_not_sent``, the lead's ruling of 2026-09-29, pending the owner), and one that may have been sent is
  a stop;
* the cell's steady gate (``safety.dwell``) before P0 and before every contact leg;
* the budget key: the same centre for ``check`` and ``record``, the landing the centre moved by the push.
"""

from __future__ import annotations

import builtins
import dataclasses
import json
import math
import unittest
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.quaternion import to_rotation_matrix
from src.robot.core import (
    NO_PLAN_FAIL_SAFE_MESSAGE,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotError,
    RobotMode,
    RobotStatus,
    SafetyMode,
)
from src.robot.core.arm_capabilities import DigitalIOPort, LineMotion, LineReading
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.core.keep_out import SegmentationOffer
from src.robot.grasping.recovery import push_planner
from src.robot.grasping.recovery.policy import refused_before_motion
from src.robot.grasping.recovery.push_budgets import BUDGET_ALLOWED, BUDGET_PART_SPENT, PushBudgets
from src.robot.grasping.recovery.push_motion import (
    APPROACH_LEG,
    APPROACH_RISE_MM,
    BACK_OFF_MM,
    CONTACT_GAP_MM,
    CONTACT_LEGS,
    LEG_ACCEL_CAP_M_S2,
    LEG_SPEED_CAPS_M_S,
    LONGEST_PUSH_MM,
    SHORTEST_PUSH_MM,
    PushOutcome,
    PushOutcomeCode as C,
    execute_push,
    push_budget_key,
    push_tool_quaternion,
)
from src.robot.grasping.recovery.push_planner import AxisBox, PushHand, PushPlan, PushRefusal, plan_push
from src.robot.grippers.jaw_io import JawIOGripper

# --- the scene: a 30 mm block on a level table, a post 15 mm off its +Y face, set back along -X -----------------

HAND_E = PushHand(finger_thickness_mm=10.80, finger_width_mm=29.24, fingertip_depth_mm=10.45,
                  finger_length_mm=36.53, palm_width_mm=75.0, open_width_mm=49.99, palm_thickness_mm=75.0)
WORKSPACE = AxisBox(min_mm=(-800.0, -800.0, -100.0), max_mm=(800.0, 800.0, 600.0))


def _block(centre_xy: tuple[float, float] = (0.0, 0.0), size_xy: tuple[float, float] = (30.0, 30.0),
           height: float = 30.0, step: float = 2.0) -> np.ndarray:
    cx, cy = centre_xy
    sx, sy = size_xy
    xs = np.arange(cx - sx / 2, cx + sx / 2 + 1e-9, step)
    ys = np.arange(cy - sy / 2, cy + sy / 2 + 1e-9, step)
    zs = np.arange(0.0, height + 1e-9, step)
    top = np.array([(x, y, height) for x in xs for y in ys])
    sides = [np.array([(x, cy - sy / 2, z) for x in xs for z in zs]),
             np.array([(x, cy + sy / 2, z) for x in xs for z in zs]),
             np.array([(cx - sx / 2, y, z) for y in ys for z in zs]),
             np.array([(cx + sx / 2, y, z) for y in ys for z in zs])]
    return np.vstack([top, *sides])


PART = _block()
POST = _block(centre_xy=(-20.0, 35.0), size_xy=(10.0, 10.0))
TABLE = np.array([(x, y, 0.0) for x in np.arange(-400.0, 400.0 + 1e-9, 5.0)
                  for y in np.arange(-400.0, 400.0 + 1e-9, 5.0)])


def _plan() -> PushPlan:
    plan = plan_push(target_points_mm=PART, neighbour_points_mm=POST, support_normal=(0.0, 0.0, 1.0),
                     support_offset_mm=0.0, workspace=WORKSPACE, hand=HAND_E, table_points_mm=TABLE,
                     push_axes_xy=((1.0, 0.0),))
    assert isinstance(plan, PushPlan), plan
    return plan


PLAN = _plan()
OFFER = SegmentationOffer(captured_at_s=1.0, target_points_base_mm=PART, target_label="part")

# --- the log and the doubles that write to it ------------------------------------------------------------------

#: Every verb that moves an arm, or stops it: none may be called after a push stopped, and none but ``move`` at all.
MOTION_VERBS = ("move", "move_to", "move_linear", "move_joint", "move_to_joints", "move_home", "move_to_home", "stop")

_RUNNING = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.NORMAL, protective_stopped=False,
                       emergency_stopped=False)
_PROTECTIVE = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.PROTECTIVE_STOP,
                          protective_stopped=True, emergency_stopped=False, message="Safetystatus: PROTECTIVE_STOP")


class _Log:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def write(self, kind: str, data: Any = None) -> None:
        self.events.append((kind, data))

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.events]

    def of(self, *kinds: str) -> list[Any]:
        return [data for kind, data in self.events if kind in kinds]


class _ToolDO0:
    """Tool DO0 of the owner's UR10, with the Hand-E on it: every write logged, every change moves the jaws."""

    def __init__(self, log: _Log) -> None:
        self.log = log
        self.level = False
        self.next_write_fails = False
        self.read_fails = False

    def set_digital_output(self, pin: int, value: bool, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> None:
        if (int(pin), port) != (0, DigitalIOPort.TOOL):
            raise AssertionError(f"the owner's hand is on tool DO0, and {port.value} output {pin} was written")
        self.log.write("DO", bool(value))
        if self.next_write_fails:
            self.next_write_fails = False
            raise RobotError("the controller did not take the write on tool output 0")
        self.level = bool(value)

    def get_digital_output(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        if self.read_fails:
            raise RobotError("the controller did not answer the read of tool output 0")
        return self.level

    def get_digital_input(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        raise AssertionError("a single toggle reads no input")

    def set_analog_output(self, pin: int, value: float, *, current: bool = False) -> None:
        raise AssertionError("nothing analog is wired")


class _Person:
    """Says "open" at the connect; after that, any question fails the test."""

    def __init__(self) -> None:
        self.answers = ["open"]
        self.asked_after_connect: list[str] = []

    def __call__(self, question: str) -> str:
        if not self.answers:
            self.asked_after_connect.append(question)
            raise AssertionError(f"a push asked a person: {question}")
        return self.answers.pop(0)


class _World:
    """The arm's live camera world, as the keep-out scope reaches it: every offer and forget on the log."""

    def __init__(self, log: _Log) -> None:
        self.log = log

    def offer_segmentation(self, **keywords: Any) -> None:
        self.log.write("offer", keywords)

    def forget_segmentation(self) -> None:
        self.log.write("forget")


class _Arm:
    """A cuRobo UR as the push sees it: it judges lines, has a live world, and logs every call to every motion verb.

    ``answers`` maps a ``move`` call's index to what that call does instead of arriving: a :class:`MotionResult`, an
    exception to raise, or ``"stop"`` (the controller protective-stops during it). ``stop_after`` is the index of
    the call after whose arrival the controller protective-stops.
    """

    def __init__(self, log: _Log, *, answers: Optional[dict[int, Any]] = None, stop_after: Optional[int] = None,
                 world: bool = True, lines: Optional[LineMotion] = LineMotion.CHECKED) -> None:
        self.log = log
        self.answers = dict(answers or {})
        self.stop_after = stop_after
        self.live_planner_world = _World(log) if world else None
        self._lines = lines
        self.status = _RUNNING
        self.moves = 0
        self.tcp = Pose.tool_down(0.0, 0.0, 400.0)

    def line_motion(self) -> LineReading:
        if self._lines is None:
            raise AssertionError("an arm that does not say is built without line_motion")
        return LineReading(self._lines, "cuRobo samples and judges every line")

    def get_tcp_pose(self) -> Pose:
        return self.tcp

    def move(self, pose: Pose, **keywords: Any) -> MotionResult:
        index, self.moves = self.moves, self.moves + 1
        self.log.write("move", (pose, dict(keywords)))
        answer = self.answers.get(index)
        if answer == "stop":
            self.status = _PROTECTIVE
            return MotionResult.failed(MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                                       message="Driver reported moveL() failure")
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, MotionResult):
            return answer
        self.tcp = pose
        if index == self.stop_after:
            self.status = _PROTECTIVE
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)

    # Every other motion verb: logged, never expected.
    def move_to(self, *args: Any, **kwargs: Any) -> bool:
        self.log.write("move_to", args)
        return False

    def move_linear(self, *args: Any, **kwargs: Any) -> None:
        self.log.write("move_linear", args)

    def move_joint(self, *args: Any, **kwargs: Any) -> None:
        self.log.write("move_joint", args)

    def move_to_joints(self, *args: Any, **kwargs: Any) -> MotionResult:
        self.log.write("move_to_joints", args)
        return MotionResult.failed(MotionStatus.UNSUPPORTED, MotionCommand.MOVE_JOINTS)

    def move_home(self) -> bool:
        self.log.write("move_home")
        return False

    def move_to_home(self) -> MotionResult:
        self.log.write("move_to_home")
        return MotionResult.failed(MotionStatus.UNSUPPORTED, MotionCommand.MOVE_HOME)

    def stop(self) -> None:
        self.log.write("stop")


class _UR(_Arm):
    """The same arm, reporting its controller as the UR driver does; clearing a stop fails the test."""

    def __init__(self, log: _Log, *, unreadable_after: Optional[int] = None, **keywords: Any) -> None:
        super().__init__(log, **keywords)
        self.unreadable_after = unreadable_after
        self.reads = 0

    def get_robot_status(self) -> RobotStatus:
        self.reads += 1
        self.log.write("status")
        if self.unreadable_after is not None and self.moves > self.unreadable_after:
            raise RobotError("the dashboard did not answer")
        return self.status

    def recover_from_protective_stop(self) -> bool:
        self.log.write("recover")
        raise AssertionError("nothing on a push may clear a stop: a person does")


class _Undeclared(_UR):
    """An arm that does not say what it keeps of a straight line."""

    line_motion = None  # type: ignore[assignment]


class _NoTypedMove(_UR):
    move = None  # type: ignore[assignment]


class _LinesUnreadable(_UR):
    """An arm whose answer to what it keeps of a straight line raises."""

    def line_motion(self) -> LineReading:
        raise RobotError("the planner did not answer")


class _StatusNone(_UR):
    """An arm whose controller read answers ``None`` instead of a status once ``moves`` passed ``none_after``."""

    def __init__(self, log: _Log, *, none_after: int, **keywords: Any) -> None:
        super().__init__(log, **keywords)
        self.none_after = none_after

    def get_robot_status(self) -> RobotStatus:
        status = super().get_robot_status()
        return None if self.moves > self.none_after else status  # type: ignore[return-value]


class _Steady(_UR):
    """An arm whose tree asks for the steady gate (``safety.dwell``), as the owner's robot.yaml does; every wait is
    on the log, and ``waits`` maps a wait's index to what it answers instead of True (False, or an exception)."""

    TIMEOUT_S = 3.0

    def __init__(self, log: _Log, *, waits: Optional[dict[int, Any]] = None, require: bool = True,
                 **keywords: Any) -> None:
        super().__init__(log, **keywords)
        dwell = SimpleNamespace(require_steady_before_motion=require, steady_timeout_s=self.TIMEOUT_S)
        self.config = SimpleNamespace(safety=SimpleNamespace(dwell=dwell))
        self.waits = dict(waits or {})
        self.waited = 0

    def wait_until_steady(self, timeout_s: float = 5.0, poll_interval_s: float = 0.02) -> bool:
        index, self.waited = self.waited, self.waited + 1
        self.log.write("steady", timeout_s)
        answer = self.waits.get(index, True)
        if isinstance(answer, BaseException):
            raise answer
        return bool(answer)


class _RefusingWorld(_World):
    """A live world that refuses the offer, as the real one refuses masks of a camera it does not know."""

    def offer_segmentation(self, **keywords: Any) -> None:
        raise ValueError("no camera 'wrist' in this world")


class _Measured:
    """A gripper that measures its width, as the Robotiq socket driver or the sim does."""

    def __init__(self, width_mm: float, max_width_mm: float = 49.99, *, connected: bool = True,
                 read_fails: bool = False) -> None:
        self.width_mm = width_mm
        self.max_width_mm = max_width_mm
        self.is_connected = connected
        self.read_fails = read_fails
        self.writes: list[Any] = []

    def width_is_measured(self) -> bool:
        return True

    def get_width_mm(self) -> float:
        if self.read_fails:
            raise RobotError("the gripper did not answer the width read")
        return self.width_mm

    def set_width_mm(self, *args: Any, **kwargs: Any) -> None:
        self.writes.append(args)
        raise AssertionError("a push never commands the jaws")


class _Cell:
    """The owner's cell for one push: the log, the tool output, the connected Hand-E, the arm."""

    def __init__(self, case: unittest.TestCase, arm_class: type = _UR, *, connected: bool = True,
                 **arm: Any) -> None:
        self.log = _Log()
        self.io = _ToolDO0(self.log)
        self.person = _Person()
        self.jaws = JawIOGripper(self.io, actuation="single_toggle", close_output_pin=0, io_port="tool",
                                 pulse_s=0.2, close_settle_s=2.0, min_width_mm=5.0, max_width_mm=49.99,
                                 ask=self.person, sleep=lambda seconds: None)
        if connected:
            self.jaws.connect()
        else:
            self.person.answers = []  # never connected: nobody said where the jaws stand, and nobody may be asked
        self.arm = arm_class(self.log, **arm)
        self.case = case
        case.enterContext(patch.object(builtins, "input", side_effect=AssertionError("input() was called")))

    def push(self, plan: Any = PLAN, *, gripper: Any = "jaws", offer: SegmentationOffer = OFFER) -> PushOutcome:
        self.commands_before = self.jaws.commands_sent
        self.writes_before = len(self.log.of("DO"))
        self.log.events.clear()
        return execute_push(self.arm, plan, gripper=self.jaws if gripper == "jaws" else gripper, offer=offer)

    def motions(self) -> list[str]:
        return [kind for kind in self.log.kinds() if kind in MOTION_VERBS]

    def moves(self) -> list[tuple[Pose, dict[str, Any]]]:
        return self.log.of("move")

    def assert_the_jaws_untouched(self) -> None:
        self.case.assertEqual(self.jaws.commands_sent, self.commands_before)
        self.case.assertEqual(self.log.of("DO"), [])
        self.case.assertEqual(self.person.asked_after_connect, [])

    def assert_nothing_moved(self, outcome: PushOutcome) -> None:
        self.case.assertTrue(outcome.code.startswith("refused_"), outcome)
        self.case.assertTrue(refused_before_motion(outcome.code))
        self.case.assertTrue(outcome.refused_before_motion)
        self.case.assertFalse(outcome.motion_started)
        self.case.assertFalse(outcome.executed)
        self.case.assertEqual(self.motions(), [])
        self.case.assertEqual(self.log.of("offer", "forget"), [])
        self.case.assertTrue(outcome.reason.endswith("."), outcome.reason)
        self.assert_the_jaws_untouched()


def _refused(status: MotionStatus, message: str = "refused") -> MotionResult:
    return MotionResult.failed(status, MotionCommand.MOVE_TO, message=message)


def _moved(point: Any, by: Any) -> tuple[float, float, float]:
    moved = np.asarray(point, dtype=np.float64) + np.asarray(by, dtype=np.float64)
    return (float(moved[0]), float(moved[1]), float(moved[2]))


def _push_leg_longer(plan: PushPlan, extra_mm: float) -> PushPlan:
    """``plan`` with its push leg driven ``extra_mm`` further and the back-off and lift moved with it, while its
    ``push_distance_mm`` still says what it said: a plan no planner made."""
    along = extra_mm * np.asarray(plan.direction, dtype=np.float64)
    return dataclasses.replace(plan, end_tcp_mm=_moved(plan.end_tcp_mm, along),
                               back_off_tcp_mm=_moved(plan.back_off_tcp_mm, along),
                               lift_tcp_mm=_moved(plan.lift_tcp_mm, along))


def _pushing(distance_mm: float) -> PushPlan:
    """PLAN pushing ``distance_mm`` instead, every station and the landing moved with it: a consistent plan no
    planner made."""
    plan = _push_leg_longer(PLAN, distance_mm - PLAN.push_distance_mm)
    along = (distance_mm - PLAN.push_distance_mm) * np.asarray(PLAN.direction, dtype=np.float64)
    return dataclasses.replace(plan, push_distance_mm=distance_mm,
                               predicted_landing_centre_mm=_moved(PLAN.predicted_landing_centre_mm, along))


class _Reordered(PushPlan):
    """A plan whose contact legs come in another order (back before push)."""

    @property
    def legs(self) -> tuple[push_planner.PushLeg, ...]:
        down, push, back, up = super().legs
        return (down, back, push, up)


# --- the tests -------------------------------------------------------------------------------------------------


class APushRunsAsTheOwnerSetIt(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = _Cell(self)
        self.outcome = self.cell.push()

    def test_it_pushes_and_lifts_clear(self) -> None:
        outcome = self.outcome
        self.assertIs(outcome.code, C.PUSHED)
        self.assertTrue(outcome.executed)
        self.assertTrue(outcome.motion_started)
        self.assertFalse(outcome.stopped)
        self.assertEqual(outcome.legs_done, (APPROACH_LEG, *CONTACT_LEGS))
        self.assertEqual(len(outcome.results), 5)
        self.assertIsNone(outcome.leg)
        assert outcome.tcp_after is not None
        np.testing.assert_allclose(outcome.tcp_after.position_mm, PLAN.lift_tcp_mm)

    def test_p0_is_a_plain_move_like_a_grasp_approach_and_every_leg_a_judged_line_at_its_speed(self) -> None:
        moves = self.cell.moves()
        self.assertEqual(self.cell.motions(), ["move"] * 5)
        p0, p0_keywords = moves[0]
        self.assertEqual(p0_keywords, {})
        np.testing.assert_allclose(p0.position_mm, PLAN.approach_tcp_mm)
        # The owner's numbers, written out: down, back and up at 50 mm/s, the push at 25 mm/s, 0.1 m/s^2.
        owners = {"down": 0.05, "push": 0.025, "back": 0.05, "up": 0.05}
        self.assertEqual(dict(LEG_SPEED_CAPS_M_S), owners)
        self.assertEqual(LEG_ACCEL_CAP_M_S2, 0.1)
        for leg, (pose, keywords) in zip(PLAN.legs, moves[1:]):
            with self.subTest(leg=leg.name):
                self.assertEqual(set(keywords), {"linear", "vel", "acc"})
                self.assertIs(keywords["linear"], True)
                self.assertLessEqual(keywords["vel"], owners[leg.name] + 1e-12)
                self.assertGreater(keywords["vel"], 0.0)
                self.assertLessEqual(keywords["acc"], 0.1 + 1e-12)
                self.assertEqual((keywords["vel"], keywords["acc"]), (leg.vel_m_s, leg.acc_m_s2))
                np.testing.assert_allclose(pose.position_mm, leg.end_tcp_mm)
                self.assertIs(pose.frame, Frame.BASE)

    def test_every_motion_runs_inside_the_keep_out_of_the_part_and_its_swept_travel(self) -> None:
        kinds = self.cell.log.kinds()
        self.assertEqual(kinds.count("offer"), 1)
        self.assertEqual(kinds.count("forget"), 1)
        first, last = kinds.index("offer"), kinds.index("forget")
        moves = [i for i, kind in enumerate(kinds) if kind == "move"]
        self.assertTrue(first < min(moves) and max(moves) < last, kinds)
        offered = self.cell.log.of("offer")[0]
        self.assertIs(offered["hold"], True)
        self.assertEqual(offered["target_label"], "part")
        held = np.asarray(offered["target_points_base_mm"])
        # The part where it stands and every place along its push, up to where it lands.
        self.assertAlmostEqual(float(held[:, 0].min()), float(PART[:, 0].min()))
        self.assertAlmostEqual(float(held[:, 0].max()), float(PART[:, 0].max()) + PLAN.push_distance_mm)
        self.assertGreater(len(held), len(PART))

    def test_the_tool_points_down_and_closes_along_the_push_on_every_pose(self) -> None:
        quaternions = {tuple(np.round(pose.quaternion_xyzw, 12)) for pose, _ in self.cell.moves()}
        self.assertEqual(len(quaternions), 1)
        rotation = to_rotation_matrix(self.cell.moves()[0][0].quaternion_xyzw)
        np.testing.assert_allclose(rotation[:, 2], PLAN.approach, atol=1e-9)
        self.assertAlmostEqual(abs(float(rotation[:, 0] @ np.asarray(PLAN.direction))), 1.0)

    def test_the_toggle_is_only_read(self) -> None:
        self.cell.assert_the_jaws_untouched()
        self.assertFalse(self.cell.jaws.jaws_closed)
        self.assertEqual(self.cell.jaws.why_jaws_unknown(), "")

    def test_the_outcome_is_plain_data(self) -> None:
        data = json.loads(json.dumps(self.outcome.to_dict()))
        self.assertEqual(data["code"], "pushed")
        self.assertEqual(data["legs_done"], ["approach", "down", "push", "back", "up"])
        self.assertTrue(data["motion_started"])


class NothingMovesWithoutEveryGuard(unittest.TestCase):
    def _cell(self, arm_class: type = _UR, **arm: Any) -> _Cell:
        return _Cell(self, arm_class, **arm)

    def test_the_planners_refusal_is_answered_with_a_refusal(self) -> None:
        cell = self._cell()
        refusal = PushRefusal(code="no_free_direction", sentence="None of the 8 push directions is free.")
        outcome = cell.push(refusal)
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_NO_PLAN)
        self.assertIs(outcome.refusal, refusal)
        self.assertIn("no_free_direction", outcome.reason)
        cell = self._cell()
        cell.assert_nothing_moved(cell.push(None))

    def test_the_count_says_closed(self) -> None:
        cell = self._cell()
        cell.jaws.set_closed(True)  # the one change a close sends, before the push
        outcome = cell.push()
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_JAWS_CLOSED)
        self.assertTrue(cell.jaws.jaws_closed)

    def test_the_count_is_unknown(self) -> None:
        cell = self._cell()
        cell.io.next_write_fails = True
        with self.assertRaises(RobotError):
            cell.jaws.set_closed(True)
        self.assertTrue(cell.jaws.edge_unknown)
        outcome = cell.push()
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_JAWS_UNKNOWN)

    def test_do0_was_switched_by_hand(self) -> None:
        cell = self._cell()
        cell.io.level = not cell.io.level  # the owner at the pendant's I/O tab
        said = cell.jaws.why_jaws_unknown()
        self.assertTrue(said)
        outcome = cell.push()
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_JAWS_UNKNOWN)
        self.assertIn(said, outcome.reason)

    def test_do0_cannot_be_read(self) -> None:
        # why_jaws_unknown() reads DO0 to see whether a person switched it; a read that raises refuses the push.
        cell = self._cell()
        cell.io.read_fails = True
        outcome = cell.push()
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_JAWS_UNKNOWN)
        self.assertIn("could not be read", outcome.reason)

    def test_the_controller_is_stopped_or_cannot_be_read(self) -> None:
        cell = self._cell()
        cell.arm.status = _PROTECTIVE
        outcome = cell.push()
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_CONTROLLER_STOPPED)
        self.assertIn("protective_stop=True", outcome.reason)
        self.assertNotIn("recover", cell.log.kinds())
        cell = self._cell(unreadable_after=-1)
        outcome = cell.push()
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_CONTROLLER_STOPPED)
        self.assertIn("could not be read", outcome.reason)

    def test_the_arm_does_not_judge_its_lines(self) -> None:
        for lines in (LineMotion.CONTROLLER_LINE, LineMotion.TELEPORT, LineMotion.NOT_KEPT):
            with self.subTest(lines=lines):
                cell = self._cell(lines=lines)
                outcome = cell.push()
                cell.assert_nothing_moved(outcome)
                self.assertIs(outcome.code, C.REFUSED_LINES_NOT_JUDGED)
        cell = self._cell(_Undeclared)
        outcome = cell.push()
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_LINES_NOT_JUDGED)

    def test_the_arm_has_no_typed_move(self) -> None:
        cell = self._cell(_NoTypedMove)
        outcome = cell.push()
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_NO_TYPED_MOVE)

    def test_the_arm_has_no_live_world(self) -> None:
        cell = self._cell(world=False)
        outcome = cell.push()
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_NO_LIVE_WORLD)

    def test_the_offer_carries_no_points_of_the_part(self) -> None:
        cell = self._cell()
        outcome = cell.push(offer=SegmentationOffer(captured_at_s=1.0, target_label="part"))
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_NO_TARGET_POINTS)

    def test_there_is_no_gripper_or_one_that_cannot_say(self) -> None:
        for gripper, code in ((None, C.REFUSED_NO_GRIPPER), (object(), C.REFUSED_JAWS_UNKNOWN)):
            with self.subTest(gripper=gripper):
                cell = self._cell()
                outcome = cell.push(gripper=gripper)
                cell.assert_nothing_moved(outcome)
                self.assertIs(outcome.code, code)

    def test_a_measured_gripper_is_read_and_never_commanded(self) -> None:
        cell = self._cell()
        narrow = _Measured(width_mm=20.0)
        outcome = cell.push(gripper=narrow)
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_JAWS_CLOSED)
        cell = self._cell()
        open_ = _Measured(width_mm=49.2)
        outcome = cell.push(gripper=open_)
        self.assertIs(outcome.code, C.PUSHED)
        self.assertEqual((narrow.writes, open_.writes), ([], []))

    def test_a_leg_faster_than_the_owner_allows(self) -> None:
        # A planner constant that drifts is refused, never driven: the caps are the owner's numbers, written apart.
        for constant, value in (("PUSH_SPEED_MM_S", 30.0), ("DESCENT_SPEED_MM_S", 60.0), ("LIFT_SPEED_MM_S", 51.0),
                                ("LEG_ACCEL_M_S2", 0.2)):
            with self.subTest(constant=constant), patch.object(push_planner, constant, value):
                cell = self._cell()
                outcome = cell.push()
                cell.assert_nothing_moved(outcome)
                self.assertIs(outcome.code, C.REFUSED_LEG_TOO_FAST)

    def test_axes_that_make_no_tool_pose(self) -> None:
        for name, changes in (("closing along the approach", {"direction": (0.0, 0.0, -1.0)}),
                              ("no approach", {"approach": (0.0, 0.0, 0.0)}),
                              ("a station off the map", {"approach_tcp_mm": (float("nan"), 0.0, 0.0)})):
            with self.subTest(case=name):
                cell = self._cell()
                outcome = cell.push(dataclasses.replace(PLAN, **changes))
                cell.assert_nothing_moved(outcome)
                self.assertIs(outcome.code, C.REFUSED_PLAN_MALFORMED)

    def test_the_arms_line_reading_raises(self) -> None:
        cell = self._cell(_LinesUnreadable)
        outcome = cell.push()
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_LINES_NOT_JUDGED)
        self.assertIn("could not be read", outcome.reason)

    def test_the_controller_answers_no_status(self) -> None:
        cell = self._cell(_StatusNone, none_after=-1)
        outcome = cell.push()
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_CONTROLLER_STOPPED)
        self.assertIn("could not be read", outcome.reason)

    def test_the_live_world_refuses_the_offer(self) -> None:
        # The real LivePlannerWorld raises for masks of a camera it does not know: nothing was held, nothing moves.
        cell = self._cell()
        cell.arm.live_planner_world = _RefusingWorld(cell.log)
        outcome = cell.push()
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_KEEP_OUT_NOT_TAKEN)
        self.assertIn("ValueError", outcome.reason)


class TheJawsAreReadOffAHandThatIsConnected(unittest.TestCase):
    """The count stands only while the hand is connected: a connect asks where the jaws stand, a disconnect ends
    the count, and a disconnected toggle is not compared with DO0 any more. So a hand that is not connected is a
    count nobody vouches for, whatever ``jaws_closed`` and ``why_jaws_unknown()`` answer."""

    def _assert_unknown(self, cell: _Cell, outcome: PushOutcome) -> None:
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_JAWS_UNKNOWN)

    def test_a_toggle_never_connected(self) -> None:
        cell = _Cell(self, connected=False)
        # What the read-only checks answer before anybody said where the jaws stand: they would pass.
        self.assertEqual((cell.jaws.jaws_closed, cell.jaws.edge_unknown, cell.jaws.why_jaws_unknown()),
                         (False, False, ""))
        outcome = cell.push()
        self._assert_unknown(cell, outcome)
        self.assertIn("not connected", outcome.reason)

    def test_a_toggle_disconnected_after_its_count_said_open(self) -> None:
        cell = _Cell(self)
        cell.jaws.disconnect()
        self._assert_unknown(cell, cell.push())

    def test_a_toggle_disconnected_and_then_switched_at_the_pendant(self) -> None:
        cell = _Cell(self)
        cell.jaws.disconnect()
        cell.io.level = not cell.io.level  # the jaws moved; a disconnected hand does not compare DO0 any more
        self.assertEqual(cell.jaws.why_jaws_unknown(), "")
        self._assert_unknown(cell, cell.push())

    def test_a_measured_gripper_not_connected(self) -> None:
        cell = _Cell(self)
        outcome = cell.push(gripper=_Measured(width_mm=49.99, connected=False))
        self._assert_unknown(cell, outcome)
        self.assertIn("not connected", outcome.reason)

    def test_a_jaw_read_that_raises(self) -> None:
        cell = _Cell(self)
        with patch.object(cell.jaws, "why_jaws_unknown", side_effect=RobotError("the dashboard did not answer")):
            outcome = cell.push()
        self._assert_unknown(cell, outcome)
        self.assertIn("could not be read", outcome.reason)
        cell = _Cell(self)
        outcome = cell.push(gripper=_Measured(width_mm=49.99, read_fails=True))
        self._assert_unknown(cell, outcome)
        self.assertIn("could not be read", outcome.reason)

    def test_a_measured_gripper_just_outside_its_open_tolerance(self) -> None:
        cell = _Cell(self)
        outcome = cell.push(gripper=_Measured(width_mm=47.4))  # 2.59 mm short of fully open, the tolerance is 2
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_JAWS_CLOSED)


class ThePlanIsDrivenOnlyInTheShapeTheOwnerSet(unittest.TestCase):
    """``execute_push`` checks the plan it is handed, whoever built it. ``plan_push`` never makes these; a plan
    built or replaced by a caller must not reach the arm either, so each is refused before anything moves."""

    def test_the_shape_is_the_owners_numbers_and_the_planners(self) -> None:
        self.assertEqual((SHORTEST_PUSH_MM, LONGEST_PUSH_MM, BACK_OFF_MM), (10.0, 50.0, 5.0))
        self.assertEqual((SHORTEST_PUSH_MM, LONGEST_PUSH_MM, BACK_OFF_MM, CONTACT_GAP_MM, APPROACH_RISE_MM),
                         (push_planner.MIN_CLEARANCE_GAIN_MM, push_planner.PUSH_DISTANCE_CAP_MM,
                          push_planner.BACK_OFF_MM, push_planner.CONTACT_GAP_MM, push_planner.PUSH_APPROACH_RISE_MM))

    def test_a_push_longer_than_fifty_mm(self) -> None:
        for distance in (50.5, 200.0):
            with self.subTest(distance=distance):
                cell = _Cell(self)
                outcome = cell.push(_pushing(distance))
                cell.assert_nothing_moved(outcome)
                self.assertIs(outcome.code, C.REFUSED_PUSH_TOO_LONG)
                self.assertIn("50 mm", outcome.reason)
        self.assertIs(_Cell(self).push(_pushing(50.0)).code, C.PUSHED)

    def test_a_push_shorter_than_ten_mm(self) -> None:
        for distance in (5.0, 9.5):
            with self.subTest(distance=distance):
                cell = _Cell(self)
                outcome = cell.push(_pushing(distance))
                cell.assert_nothing_moved(outcome)
                self.assertIs(outcome.code, C.REFUSED_PUSH_TOO_SHORT)
                self.assertIn("10 mm", outcome.reason)
        self.assertIs(_Cell(self).push(_pushing(10.0)).code, C.PUSHED)

    def test_a_push_leg_longer_than_the_push_it_names(self) -> None:
        # It says 30 mm and drives 300: the keep-out would be swept 30 mm and the report would say 30.
        cell = _Cell(self)
        outcome = cell.push(_push_leg_longer(PLAN, 270.0))
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_PLAN_MALFORMED)
        self.assertIn("push leg", outcome.reason)

    def test_legs_off_the_plans_own_lines(self) -> None:
        d, a = np.asarray(PLAN.direction), np.asarray(PLAN.approach)
        side = np.cross(a, d)
        cases: dict[str, dict[str, Any]] = {
            "the down leg runs sideways": {"contact_start_tcp_mm": _moved(PLAN.contact_start_tcp_mm, 300.0 * side)},
            "P0 stands beside the contact start": {"approach_tcp_mm": _moved(PLAN.approach_tcp_mm, 20.0 * d)},
            "P0 stands higher than the rise": {"approach_tcp_mm": _moved(PLAN.approach_tcp_mm, -50.0 * a)},
            "the back-off runs 50 mm": {"back_off_tcp_mm": _moved(PLAN.back_off_tcp_mm, -45.0 * d),
                                        "lift_tcp_mm": _moved(PLAN.lift_tcp_mm, -45.0 * d)},
            "the back-off runs forward": {"back_off_tcp_mm": _moved(PLAN.back_off_tcp_mm, 10.0 * d),
                                          "lift_tcp_mm": _moved(PLAN.lift_tcp_mm, 10.0 * d)},
            "the up leg goes into the table": {"lift_tcp_mm": (PLAN.lift_tcp_mm[0], PLAN.lift_tcp_mm[1], -150.0)},
        }
        for name, changes in cases.items():
            with self.subTest(case=name):
                cell = _Cell(self)
                outcome = cell.push(dataclasses.replace(PLAN, **changes))
                cell.assert_nothing_moved(outcome)
                self.assertIs(outcome.code, C.REFUSED_PLAN_MALFORMED)

    def test_an_approach_tilted_away_from_straight_down(self) -> None:
        for tilt_deg, code in ((10.0, C.REFUSED_PLAN_MALFORMED), (4.0, C.PUSHED)):
            with self.subTest(tilt_deg=tilt_deg):
                tilt = math.radians(tilt_deg)
                approach = (0.0, math.sin(tilt), -math.cos(tilt))  # tilted about BASE X, still square to the push
                a = np.asarray(approach)
                plan = dataclasses.replace(PLAN, approach=approach,
                                           approach_tcp_mm=_moved(PLAN.contact_start_tcp_mm, -80.0 * a),
                                           lift_tcp_mm=_moved(PLAN.back_off_tcp_mm, -80.0 * a))
                cell = _Cell(self)
                outcome = cell.push(plan)
                self.assertIs(outcome.code, code, outcome.reason)
                if code is not C.PUSHED:
                    cell.assert_nothing_moved(outcome)

    def test_legs_in_another_order(self) -> None:
        cell = _Cell(self)
        plan = _Reordered(**{f.name: getattr(PLAN, f.name) for f in dataclasses.fields(PLAN)})
        outcome = cell.push(plan)
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_LEG_TOO_FAST)

    def test_an_offer_of_another_part(self) -> None:
        elsewhere = SegmentationOffer(captured_at_s=1.0, target_points_base_mm=PART + np.array([200.0, 0.0, 0.0]),
                                      target_label="part")
        cell = _Cell(self)
        outcome = cell.push(offer=elsewhere)
        cell.assert_nothing_moved(outcome)
        self.assertIs(outcome.code, C.REFUSED_OFFER_NOT_THE_PART)

    def test_the_keep_out_is_swept_as_far_as_the_push_leg_drives(self) -> None:
        cell = _Cell(self)
        outcome = cell.push(_push_leg_longer(PLAN, 0.8))  # within the stations' tolerance
        self.assertIs(outcome.code, C.PUSHED)
        held = np.asarray(cell.log.of("offer")[0]["target_points_base_mm"])
        d = np.asarray(PLAN.direction)
        self.assertAlmostEqual(float((held @ d).max() - (PART @ d).max()), PLAN.push_distance_mm + 0.8, places=6)


class P0RefusedBeforeAnythingWasSentIsStillARefusal(unittest.TestCase):
    def test_a_guard_or_the_planner_refused_p0(self) -> None:
        for answer in (_refused(MotionStatus.WORKSPACE_REJECTED), _refused(MotionStatus.SELF_COLLISION_REJECTED),
                       _refused(MotionStatus.IK_FAILED), _refused(MotionStatus.TIMEOUT, NO_PLAN_FAIL_SAFE_MESSAGE)):
            with self.subTest(status=answer.status):
                cell = _Cell(self, answers={0: answer})
                outcome = cell.push()
                self.assertIs(outcome.code, C.REFUSED_APPROACH_NOT_SENT)
                self.assertTrue(outcome.refused_before_motion)
                self.assertFalse(outcome.motion_started)
                self.assertEqual((outcome.leg, outcome.legs_done), (APPROACH_LEG, ()))
                self.assertEqual(cell.motions(), ["move"])
                self.assertEqual(cell.log.kinds().count("forget"), 1)
                cell.assert_the_jaws_untouched()


class AFailureAfterSomethingMovedStopsWhereTheArmIs(unittest.TestCase):
    def _assert_stopped(self, cell: _Cell, outcome: PushOutcome, *, leg: str, calls: int,
                        code: C = C.UNSAFE_RECOVERY_REFUSED) -> None:
        self.assertIs(outcome.code, code, outcome.reason)
        self.assertTrue(outcome.stopped)
        self.assertTrue(outcome.motion_started)
        self.assertFalse(outcome.refused_before_motion)
        self.assertEqual(outcome.leg, leg)
        self.assertEqual(cell.motions(), ["move"] * calls)
        kinds = cell.log.kinds()
        # Nothing moves after the motion that failed: the scope closed right after it.
        last_move = max(i for i, kind in enumerate(kinds) if kind in MOTION_VERBS)
        self.assertEqual([k for k in kinds[last_move + 1:] if k in MOTION_VERBS], [])
        self.assertEqual(kinds.count("forget"), 1)
        self.assertNotIn("recover", kinds)
        self.assertIn("a person decides", outcome.reason)
        cell.assert_the_jaws_untouched()

    def test_a_p0_that_may_have_moved_stops_there(self) -> None:
        # The same set the pick loop's look motions take as "sent nothing"; the rest may have been commanded.
        for answer in (_refused(MotionStatus.TIMEOUT, "moveJ did not finish"),
                       _refused(MotionStatus.CONTROLLER_REJECTED, "Driver reported moveJ() failure"),
                       _refused(MotionStatus.UNKNOWN), _refused(MotionStatus.UNSUPPORTED),
                       _refused(MotionStatus.CONNECTION_ERROR), _refused(MotionStatus.INVALID_TARGET),
                       RobotError("the link dropped")):
            with self.subTest(answer=answer):
                cell = _Cell(self, answers={0: answer})
                outcome = cell.push()
                self._assert_stopped(cell, outcome, leg=APPROACH_LEG, calls=1)
                self.assertEqual(outcome.legs_done, ())

    def test_a_refusal_on_any_contact_leg_stops_every_motion_after_it(self) -> None:
        # The down leg refused before anything was sent is the one exception, a fall-through (the test below).
        for n, leg in enumerate(CONTACT_LEGS):
            for answer in (_refused(MotionStatus.WORKSPACE_REJECTED, "sample 7 leaves the workspace box"),
                           _refused(MotionStatus.SELF_COLLISION_REJECTED, "rfinger: mesh distance 0.4 mm"),
                           _refused(MotionStatus.TIMEOUT, NO_PLAN_FAIL_SAFE_MESSAGE),
                           RobotError("the line raised")):
                if n == 0 and not isinstance(answer, BaseException):
                    continue
                with self.subTest(leg=leg, answer=answer):
                    cell = _Cell(self, answers={n + 1: answer})
                    outcome = cell.push()
                    self._assert_stopped(cell, outcome, leg=leg, calls=n + 2)
                    self.assertEqual(outcome.legs_done, (APPROACH_LEG, *CONTACT_LEGS[:n]))
                    if isinstance(answer, BaseException):
                        self.assertIs(outcome.error, answer)
                    else:
                        self.assertIs(outcome.motion_status, answer.status)
                    assert outcome.tcp_after is not None  # where the arm stands is read, never moved to
                    self.assertEqual(outcome.controller, "")

    def test_a_down_leg_refused_before_it_was_sent_is_a_fall_through_with_the_arm_at_p0(self) -> None:
        """The lead's ruling of 2026-09-29, pending the owner: the arm is still in the air at P0 and nothing touched the
        part, so the push is a refusal the loop falls through on (``refused_down_not_sent``), not a stop; the caller
        moves the arm back to the look. A down leg that may have been sent (a timeout that was no planner refusal, a
        raise) still stops."""
        for answer in (_refused(MotionStatus.WORKSPACE_REJECTED, "sample 7 leaves the workspace box"),
                       _refused(MotionStatus.SELF_COLLISION_REJECTED, "rfinger: mesh distance 0.4 mm"),
                       _refused(MotionStatus.TIMEOUT, NO_PLAN_FAIL_SAFE_MESSAGE)):
            with self.subTest(answer=answer):
                cell = _Cell(self, answers={1: answer})
                outcome = cell.push()
                self.assertIs(outcome.code, C.REFUSED_DOWN_NOT_SENT, outcome.reason)
                self.assertTrue(outcome.refused_before_motion)
                self.assertFalse(outcome.motion_started)
                self.assertFalse(outcome.stopped)
                self.assertEqual(outcome.leg, "down")
                self.assertEqual(outcome.legs_done, (APPROACH_LEG,), "the caller reads the arm left the look here")
                self.assertEqual(cell.motions(), ["move"] * 2, "P0 and the refused down leg, nothing after")
                self.assertEqual(cell.log.kinds().count("forget"), 1)
                self.assertNotIn("recover", cell.log.kinds())
                self.assertIn("back to the look", outcome.reason)
                cell.assert_the_jaws_untouched()
        for answer in (_refused(MotionStatus.TIMEOUT, "moveL did not finish"),
                       _refused(MotionStatus.CONTROLLER_REJECTED, "Driver reported moveL() failure")):
            with self.subTest(answer=answer):
                cell = _Cell(self, answers={1: answer})
                outcome = cell.push()
                self._assert_stopped(cell, outcome, leg="down", calls=2)

    def test_a_protective_stop_mid_push_is_the_controllers_and_nobody_clears_it(self) -> None:
        cell = _Cell(self, answers={2: "stop"})  # the push leg meets the stop
        outcome = cell.push()
        self._assert_stopped(cell, outcome, leg="push", calls=3, code=C.CONTROLLER_NOT_OPERATIONAL)
        self.assertIn("protective_stop=True", outcome.controller)
        self.assertEqual(outcome.legs_done, (APPROACH_LEG, "down"))

    def test_a_stop_after_a_leg_arrived_is_seen_before_the_next_leg(self) -> None:
        cell = _Cell(self, stop_after=1)  # the controller stops right after the down leg arrived
        outcome = cell.push()
        self._assert_stopped(cell, outcome, leg="push", calls=2, code=C.CONTROLLER_NOT_OPERATIONAL)
        self.assertEqual(outcome.legs_done, (APPROACH_LEG, "down"))

    def test_a_controller_that_stops_answering_mid_push_stops_the_push(self) -> None:
        cell = _Cell(self, unreadable_after=2)  # readable until the push leg arrived
        outcome = cell.push()
        self._assert_stopped(cell, outcome, leg="back", calls=3)
        self.assertIn("could not be read", outcome.controller)

    def test_a_camera_world_that_cannot_vouch_for_the_cell_is_raised_with_nothing_more_commanded(self) -> None:
        silent = CameraWorldUnavailable(camera="wrist", verdict="no_frame", attempts=3,
                                        reason="the wrist camera stayed silent")
        cell = _Cell(self, answers={2: silent})
        with self.assertRaises(CameraWorldUnavailable):
            cell.push()
        self.assertEqual(cell.motions(), ["move"] * 3)
        self.assertEqual(cell.log.kinds()[-1], "forget")
        cell.assert_the_jaws_untouched()

    def test_an_arm_that_does_not_report_its_controller_still_stops_on_a_failed_leg(self) -> None:
        cell = _Cell(self, _Arm, answers={3: _refused(MotionStatus.CONTROLLER_REJECTED)})
        outcome = cell.push()
        self._assert_stopped(cell, outcome, leg="back", calls=4)

    def test_a_stop_as_the_last_leg_ends_is_no_finished_push(self) -> None:
        # The controller is read once more after the lift: a stop there is not a push the caller moves on from.
        cell = _Cell(self, stop_after=4)
        outcome = cell.push()
        self._assert_stopped(cell, outcome, leg="up", calls=5, code=C.CONTROLLER_NOT_OPERATIONAL)
        self.assertEqual(outcome.legs_done, (APPROACH_LEG, *CONTACT_LEGS))
        self.assertIn("protective_stop=True", outcome.controller)
        cell = _Cell(self, unreadable_after=4)
        outcome = cell.push()
        self._assert_stopped(cell, outcome, leg="up", calls=5)
        self.assertEqual(outcome.legs_done, (APPROACH_LEG, *CONTACT_LEGS))
        self.assertIn("could not be read", outcome.controller)

    def test_a_controller_that_answers_no_status_mid_push_stops_it_without_a_raise(self) -> None:
        cell = _Cell(self, _StatusNone, none_after=1)  # a status until the down leg arrived, then None
        outcome = cell.push()
        self._assert_stopped(cell, outcome, leg="push", calls=2)
        self.assertIn("could not be read", outcome.controller)

    def test_an_arm_not_at_rest_before_a_contact_leg_stops_there(self) -> None:
        for answer in (False, RobotError("the dashboard did not answer")):
            with self.subTest(answer=answer):
                cell = _Cell(self, _Steady, waits={2: answer})  # the wait before the push leg
                outcome = cell.push()
                self._assert_stopped(cell, outcome, leg="push", calls=2)
                self.assertEqual(outcome.legs_done, (APPROACH_LEG, "down"))
                self.assertIn("rest", outcome.reason)


class TheArmIsAtRestBeforeEveryMotion(unittest.TestCase):
    """The cell's steady gate (``safety.dwell.require_steady_before_motion``, on in the owner's robot.yaml) holds
    for the push as for every grasp and look motion: the arm is waited on before P0 and before every contact leg."""

    def test_every_motion_waits_for_the_arm_to_rest_first(self) -> None:
        cell = _Cell(self, _Steady)
        outcome = cell.push()
        self.assertIs(outcome.code, C.PUSHED)
        self.assertEqual([kind for kind in cell.log.kinds() if kind in ("steady", "move")], ["steady", "move"] * 5)
        self.assertEqual(cell.log.of("steady"), [_Steady.TIMEOUT_S] * 5)

    def test_no_wait_where_the_tree_asks_for_none(self) -> None:
        cell = _Cell(self, _Steady, require=False)
        self.assertIs(cell.push().code, C.PUSHED)
        self.assertNotIn("steady", cell.log.kinds())

    def test_an_arm_not_at_rest_before_p0_is_a_refusal(self) -> None:
        for answer in (False, RobotError("the dashboard did not answer")):
            with self.subTest(answer=answer):
                cell = _Cell(self, _Steady, waits={0: answer})
                outcome = cell.push()
                cell.assert_nothing_moved(outcome)
                self.assertIs(outcome.code, C.REFUSED_ARM_NOT_STEADY)
                self.assertIn("rest", outcome.reason)


class ThePushIsCountedByOneKey(unittest.TestCase):
    def test_the_centre_and_the_landing_the_budgets_count_by(self) -> None:
        centre, landing = push_budget_key(PLAN)
        self.assertEqual(centre, PLAN.target_centre_mm)
        expected = np.asarray(PLAN.target_centre_mm) + PLAN.push_distance_mm * np.asarray(PLAN.direction)
        np.testing.assert_allclose(landing, expected)
        np.testing.assert_allclose(landing, PLAN.predicted_landing_centre_mm)

    def test_check_and_record_with_that_key_spend_the_part(self) -> None:
        cell = _Cell(self)
        outcome = cell.push()
        budgets = PushBudgets()
        budgets.start_pick()
        assert outcome.budget_centre_mm is not None
        self.assertEqual(budgets.check(outcome.budget_centre_mm).code, BUDGET_ALLOWED)
        self.assertTrue(outcome.motion_started)
        budgets.record(centre_mm=outcome.budget_centre_mm, landing_mm=outcome.budget_landing_mm)
        budgets.start_pick()
        self.assertEqual(budgets.check(outcome.budget_centre_mm).code, BUDGET_PART_SPENT)
        assert outcome.budget_landing_mm is not None
        self.assertEqual(budgets.check(outcome.budget_landing_mm).code, BUDGET_PART_SPENT)

    def test_a_refusal_before_motion_spends_nothing_but_still_names_its_key(self) -> None:
        cell = _Cell(self)
        cell.arm.status = _PROTECTIVE
        outcome = cell.push()
        self.assertFalse(outcome.motion_started)
        self.assertEqual(outcome.budget_centre_mm, push_budget_key(PLAN)[0])


class TheWristTurnsTheShortWay(unittest.TestCase):
    def test_the_closing_axis_takes_the_sign_nearer_the_tool_where_it_stands(self) -> None:
        direction = np.asarray(PLAN.direction)
        for here, sign in (((1.0, 0.0, 0.0), 1.0), ((-1.0, 0.0, 0.0), -1.0), ((0.6, 0.8, 0.0), 1.0), (None, 1.0)):
            with self.subTest(here=here):
                quaternion = push_tool_quaternion(PLAN, current_tool_x=here)
                assert quaternion is not None
                rotation = to_rotation_matrix(quaternion)
                np.testing.assert_allclose(rotation[:, 0], sign * direction, atol=1e-9)
                np.testing.assert_allclose(rotation[:, 2], PLAN.approach, atol=1e-9)

    def test_the_arm_standing_turned_half_way_round_gets_the_other_finger_leading(self) -> None:
        cell = _Cell(self)
        cell.arm.tcp = Pose.tool_down(0.0, 0.0, 400.0, yaw_deg=180.0)
        self.assertIs(cell.push().code, C.PUSHED)
        rotation = to_rotation_matrix(cell.moves()[0][0].quaternion_xyzw)
        np.testing.assert_allclose(rotation[:, 0], -np.asarray(PLAN.direction), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
