"""A tilted pick lifts the part straight up, a failed pick says whether another candidate may follow, and says why.

Side grasps are on by the owner's decision (2026-10-01), so ``Robot.pick`` now meets grasps up to 90 degrees off
vertical. It backed out of every grasp along its approach, which for a side grasp drags the part sideways along the
support, through whatever the side approach just kept its room from. A grasp more than 10 degrees off vertical now
lifts straight up (BASE +Z) after the close, by the standoff and at least 60 mm, and W-M1's carried judgement judges
that same lift before the jaws close. An empty hand still backs out along the approach: after a close that measured
nothing, and when the carried lift is refused (``CARRIED_RETREAT_REFUSED``).

``HandlingReport.another_candidate_may_follow`` (the cell-fix plan's contract 4) is true only after a guard or planner
refusal of the pick's first motion, nothing sent and nothing commanded to the jaws, or after
``CARRIED_RETREAT_REFUSED`` with the empty hand backed out open. A program that tries its next candidate reads it (the
owner's "next grasps", example 13), and looks around again first.

And every failed verb leaves a WARNING with its message in the robot log (RC6 of the plan: P5's failed moveJ reached no
file).
"""

from __future__ import annotations

import logging
import unittest
from typing import Any

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import NO_PLAN_FAIL_SAFE_MESSAGE, MotionCommand, MotionResult, MotionStatus
from src.robot.core.arm_capabilities import LineMotion, LineReading, PayloadModel
from src.robot.core.gripper import HoldEvidence
from src.robot.grasping.geometry.grasp_frame import pose_from_grasp_axes
from tests.test_a_stopped_controller_moves_no_jaws import _PROTECTIVE, _RUNNING, _pulses, _toggle
from tests.test_robot_pick_and_place import _Log, _LoggedHand, _RecordingArm

#: The pick's part, BASE mm: 100 mm over the bench, as the other pick doubles have it.
_AT = (400.0, 0.0, 100.0)
#: What the arm answers about a lift it would refuse with the part in its jaws.
_HELD = MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO,
                            message="the planner refused this line at sample 0 of 12: a part is carried")


def _handling() -> Any:
    from src.robot.execution import handling

    return handling


def _tilted(deg: float) -> Pose:
    """A grasp at ``_AT`` tilted ``deg`` about BASE x: the tool travels toward +y and down, the hand leans to -y."""
    t = np.radians(deg)
    return pose_from_grasp_axes(np.array(_AT), approach=np.array([0.0, np.sin(t), -np.cos(t)]),
                                closing_axis=np.array([1.0, 0.0, 0.0]), frame=Frame.BASE)


def _mm(position: Any) -> tuple[float, ...]:
    return tuple(round(float(v), 2) for v in position)


class _Arm(_RecordingArm):
    """The desk arm, judging a lift as if its jaws held the part (``JudgesCarriedLines``) and refusing the motions it
    is told to, nothing sent. ``statuses`` is what its controller answers, the last one from then on."""

    def __init__(self, events: list[Any], *, carried: "MotionResult | None" = None,
                 refuse: "dict[int, MotionResult] | None" = None,
                 statuses: "tuple[Any, ...]" = (_RUNNING,)) -> None:
        log = _Log()
        log.entries = events
        super().__init__(log)
        self.carried = carried
        self.refuse = dict(refuse or {})
        self._statuses = list(statuses)

    def get_robot_status(self) -> Any:
        return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]

    def carried_line_refusal(self, pose: Pose, *, grip_width_mm: float) -> "MotionResult | None":
        self.log.entries.append(("judge", _mm(pose.position_mm), round(float(grip_width_mm), 2)))
        return self.carried

    def attach_payload(self, width_mm: float) -> bool:
        self.log.entries.append(("attach", round(float(width_mm), 2)))
        return True

    def payload_declined_reason(self) -> "str | None":
        return None

    def payload_model(self) -> PayloadModel:
        return PayloadModel.PLANNER_AND_FILTER

    def move(self, pose: Pose, **kwargs: Any) -> MotionResult:
        index = sum(1 for entry in self.log.entries if entry[0] == "move")
        if index in self.refuse:
            self.log.entries.append(("move", tuple(round(float(v), 3) for v in pose.position_mm),
                                     bool(kwargs.get("linear", False)), kwargs.get("camera_world")))
            return self.refuse[index]
        return super().move(pose, **kwargs)


def _robot(events: list[Any], *, toggle: bool = True, hold: HoldEvidence = HoldEvidence.HELD, **arm: Any) -> Any:
    """The owner's toggle Hand-E on a recording tool I/O (``toggle``), or a hand that measures and pre-opens."""
    from src.robot.execution.robot import Robot

    built = _Arm(events, **arm)
    built.connect()
    if toggle:
        hand: Any = _toggle(events)
    else:
        log = _Log()
        log.entries = events
        hand = _LoggedHand(log, hold=hold, width_mm=39.0)
    return Robot.from_parts(arm=built, gripper=hand, lock_key=None)


def _moves(events: list[Any]) -> list[tuple[tuple[float, ...], bool]]:
    return [(_mm(e[1]), e[2]) for e in events if isinstance(e, tuple) and e[0] == "move"]


def _refusal(status: MotionStatus, message: str = "lfinger|seen_00: mesh distance 3.4 mm") -> MotionResult:
    return MotionResult.failed(status, MotionCommand.MOVE_TO, message=message)


# --------------------------------------------------------------------------------------------------------------------
# The lift
# --------------------------------------------------------------------------------------------------------------------


class ATiltedPickLiftsStraightUpTests(unittest.TestCase):
    def test_a_30_degree_grasp_lifts_straight_up_judged_carrying(self) -> None:
        """⭐ Red before: the line out went back along the approach to the standoff, sideways through the room the side
        grasp kept. Now: in along the approach, closed, then straight up by the standoff, and that lift is the one the
        arm judged carrying before the jaws closed."""
        handling = _handling()
        events: list[Any] = []
        robot = _robot(events, carried=None)

        report = robot.pick(_tilted(30.0), 40.0)

        self.assertIs(handling.HandlingOutcome.EXECUTED, report.outcome, report.render())
        standoff = _mm(np.array(_AT) - 80.0 * np.array([0.0, 0.5, -np.cos(np.radians(30.0))]))
        lift = (400.0, 0.0, 180.0)
        self.assertEqual([(standoff, False), (_mm(_AT), True), (lift, True)], _moves(events))
        judged = [e for e in events if isinstance(e, tuple) and e[0] == "judge"]
        self.assertEqual([("judge", lift, 39.0)], judged)
        self.assertLess(events.index(judged[0]), next(i for i, e in enumerate(events) if e == ("attach", 39.0)))
        self.assertEqual(1, _pulses(events))
        self.assertEqual("lift", report.poses[-1].label)

    def test_the_lift_is_the_standoff_but_never_under_60_mm(self) -> None:
        """The owner's 10 mm standoff: the part still comes 60 mm straight up off the support."""
        events: list[Any] = []
        robot = _robot(events, carried=None)

        robot.pick(_tilted(30.0), 40.0, standoff_mm=10.0)

        self.assertEqual((400.0, 0.0, 160.0), _moves(events)[-1][0])
        self.assertIn(("judge", (400.0, 0.0, 160.0), 39.0), events)

    def test_a_vertical_grasp_is_unchanged(self) -> None:
        events: list[Any] = []
        robot = _robot(events, carried=None)
        vertical = Pose(position_mm=np.array(_AT), quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]), frame=Frame.BASE)

        robot.pick(vertical, 40.0)

        self.assertEqual([((400.0, 0.0, 180.0), False), (_mm(_AT), True), ((400.0, 0.0, 180.0), True)],
                         _moves(events))
        self.assertIn(("judge", (400.0, 0.0, 180.0), 39.0), events)

    def test_ten_degrees_off_vertical_still_backs_out_along_the_approach(self) -> None:
        """The bound: 10 degrees backs out along the approach as before; 12 lifts straight up."""
        for deg, straight_up in ((10.0, False), (12.0, True)):
            with self.subTest(deg=deg):
                events: list[Any] = []
                robot = _robot(events, carried=None)
                robot.pick(_tilted(deg), 40.0)
                moves = _moves(events)
                self.assertEqual(moves[-1][0] == (400.0, 0.0, 180.0), straight_up, moves)
                self.assertEqual(moves[-1][0] == moves[0][0], not straight_up, moves)

    def test_a_close_that_measured_nothing_backs_out_along_the_approach(self) -> None:
        """An empty hand goes back the way it came, open: nothing to lift."""
        handling = _handling()
        events: list[Any] = []
        robot = _robot(events, toggle=False, hold=HoldEvidence.EMPTY, carried=None)

        report = robot.pick(_tilted(30.0), 40.0)

        self.assertIs(handling.HandlingOutcome.NOTHING_HELD, report.outcome, report.render())
        moves = _moves(events)
        self.assertEqual(moves[0][0], moves[-1][0], "back to the standoff along the approach")

    def test_a_refused_carried_lift_backs_out_along_the_approach_with_the_jaws_open(self) -> None:
        handling = _handling()
        events: list[Any] = []
        robot = _robot(events, carried=_HELD)

        report = robot.pick(_tilted(30.0), 40.0)

        self.assertIs(handling.HandlingOutcome.CARRIED_RETREAT_REFUSED, report.outcome, report.render())
        moves = _moves(events)
        self.assertEqual(3, len(moves))
        self.assertEqual(moves[0][0], moves[-1][0], "back up the line it came down")
        self.assertIn(("judge", (400.0, 0.0, 180.0), 39.0), events, "the lift it would have run was judged")
        self.assertEqual(0, _pulses(events))


# --------------------------------------------------------------------------------------------------------------------
# another_candidate_may_follow
# --------------------------------------------------------------------------------------------------------------------


class AnotherCandidateMayFollowTests(unittest.TestCase):
    def _pick(self, **robot: Any) -> Any:
        events: list[Any] = []
        report = _robot(events, **robot).pick(_tilted(30.0), 40.0)
        self.events = events
        return report

    def test_true_after_a_guard_refused_the_first_motion_before_sending(self) -> None:
        """⭐ Red before: the report had no such field."""
        report = self._pick(refuse={0: _refusal(MotionStatus.SELF_COLLISION_REJECTED)})
        self.assertIs(True, getattr(report, "another_candidate_may_follow", None), report.render())
        self.assertEqual(0, _pulses(self.events))

    def test_true_after_the_planner_found_no_plan_for_the_first_motion(self) -> None:
        report = self._pick(refuse={0: _refusal(MotionStatus.TIMEOUT, NO_PLAN_FAIL_SAFE_MESSAGE)})
        self.assertIs(True, report.another_candidate_may_follow, report.render())

    def test_true_after_the_carried_lift_was_refused_and_the_empty_hand_backed_out(self) -> None:
        handling = _handling()
        report = self._pick(carried=_HELD)
        self.assertIs(handling.HandlingOutcome.CARRIED_RETREAT_REFUSED, report.outcome)
        self.assertIs(True, report.another_candidate_may_follow)

    def test_false_where_anything_was_sent_stopped_or_unknown(self) -> None:
        from tests.test_camera_world_on_every_driver import _ur_curobo

        cases: dict[str, Any] = {
            "a sent motion the controller rejected": lambda: self._pick(
                refuse={0: _refusal(MotionStatus.CONTROLLER_REJECTED, "Driver reported moveJ() failure")}),
            "a motion that ran out of time": lambda: self._pick(
                refuse={0: _refusal(MotionStatus.TIMEOUT, "the move did not finish within 30 s")}),
            "a refusal of the line in, after the standoff ran": lambda: self._pick(
                refuse={1: _refusal(MotionStatus.SELF_COLLISION_REJECTED)}),
            "a stopped controller": lambda: self._pick(statuses=(_PROTECTIVE,)),
            "the toggle question": lambda: _toggle_says_closed(),
            "a camera world refusal": lambda: _pick_on(_ur_curobo()),
            "a route refusal": lambda: _pick_on(_IkArm()),
            "a gripper fault": lambda: _pick_with_a_hand_that_raises(),
            "a refusal after the jaws were told to pre-open": lambda: self._pick(
                toggle=False, refuse={0: _refusal(MotionStatus.SELF_COLLISION_REJECTED)}),
            "a pick that ran": lambda: self._pick(carried=None),
        }
        for name, pick in cases.items():
            with self.subTest(name):
                report = pick()
                self.assertIs(False, report.another_candidate_may_follow, report.render())

    def test_a_place_never_says_another_candidate_may_follow(self) -> None:
        events: list[Any] = []
        robot = _robot(events, refuse={0: _refusal(MotionStatus.SELF_COLLISION_REJECTED)})
        self.assertIs(False, robot.place(_tilted(30.0)).another_candidate_may_follow)

    def test_the_report_carries_it_as_data_and_says_it(self) -> None:
        report = self._pick(refuse={0: _refusal(MotionStatus.SELF_COLLISION_REJECTED)})
        self.assertIs(True, report.to_dict()["another_candidate_may_follow"])
        self.assertIn("another candidate may follow", report.render())

    def test_the_refusals_before_sending_are_the_wrist_picks_own(self) -> None:
        """One rule for "refused before anything was sent": the set the generated view and the pick loop read."""
        from src.robot.execution import generated_view, handling

        self.assertEqual(generated_view.REFUSED_BEFORE_SENDING, handling._REFUSED_BEFORE_SENDING)


def _pick_on(arm: Any) -> Any:
    from src.robot.execution.robot import Robot

    events: list[Any] = []
    return Robot.from_parts(arm=arm, gripper=_toggle(events), lock_key=None).pick(_tilted(30.0), 40.0)


class _IkArm(_RecordingArm):
    """An arm whose controller draws the line and nothing plans it: the route is refused."""

    def __init__(self) -> None:
        super().__init__(_Log())
        self.connect()

    def line_motion(self) -> LineReading:
        return LineReading(LineMotion.CONTROLLER_LINE, "the controller draws a moveL")


def _toggle_says_closed() -> Any:
    """The toggle believes its jaws closed, and the person at the terminal only ever presses Enter: open at connect, no
    answer at a pick start, so the pick is refused before the arm moves."""
    from src.robot.execution.robot import Robot
    from src.robot.grippers.jaw_io import JawIOGripper
    from tests.test_a_stopped_controller_moves_no_jaws import _IO

    events: list[Any] = []
    jaws = JawIOGripper(_IO(events), actuation="single_toggle", close_output_pin=0, pulse_s=0.0, close_settle_s=0.0,
                        min_width_mm=5.0, max_width_mm=49.99, ask=lambda _q: "", sleep=lambda _s: None)
    jaws.connect()
    jaws.set_closed(True)
    arm = _Arm(events)
    arm.connect()
    return Robot.from_parts(arm=arm, gripper=jaws, lock_key=None).pick(_tilted(30.0), 40.0)


def _pick_with_a_hand_that_raises() -> Any:
    from src.robot.execution.robot import Robot

    class _Raising(_LoggedHand):
        def set_width_mm(self, width_mm: float, *, speed: Any = None, force: Any = None) -> None:
            raise RuntimeError("the gripper's socket closed")

    events: list[Any] = []
    log = _Log()
    log.entries = events
    arm = _Arm(events)
    arm.connect()
    return Robot.from_parts(arm=arm, gripper=_Raising(log, hold=HoldEvidence.HELD, width_mm=39.0),
                            lock_key=None).pick(_tilted(30.0), 40.0)


# --------------------------------------------------------------------------------------------------------------------
# The robot log
# --------------------------------------------------------------------------------------------------------------------


_LOGGER = "src.robot.execution.handling"


class AFailedVerbIsInTheRobotLogTests(unittest.TestCase):
    def test_a_refused_pick_leaves_a_warning_with_its_message(self) -> None:
        """⭐ Red before: a failed verb said nothing to any log; P5's failed motion is in no file (RC6)."""
        events: list[Any] = []
        robot = _robot(events, refuse={0: _refusal(MotionStatus.SELF_COLLISION_REJECTED, "shoulder|seen_54: -3.511 mm")})

        with self.assertLogs(_LOGGER, level="WARNING") as said:
            report = robot.pick(_tilted(30.0), 40.0)

        self.assertFalse(report.ok)
        self.assertTrue(any("shoulder|seen_54: -3.511 mm" in line and "motion_refused" in line for line in said.output),
                        said.output)

    def test_a_failed_place_and_a_failed_hand_verb_leave_one_too(self) -> None:
        events: list[Any] = []
        robot = _robot(events, toggle=False, hold=HoldEvidence.EMPTY)
        with self.assertLogs(_LOGGER, level="WARNING") as said:
            grasped = robot.grasp(40.0)
        self.assertFalse(grasped.ok)
        self.assertTrue(any("nothing_held" in line.lower() and "hold empty" in line.lower() for line in said.output),
                        said.output)

        robot = _robot([], refuse={0: _refusal(MotionStatus.SELF_COLLISION_REJECTED, "the bench")})
        with self.assertLogs(_LOGGER, level="WARNING") as said:
            placed = robot.place(_tilted(30.0))
        self.assertFalse(placed.ok)
        self.assertTrue(any("the bench" in line for line in said.output), said.output)

    def test_a_pick_inside_a_pick_is_said_once(self) -> None:
        """The pick's own grasp is part of the pick: one WARNING for a pick whose close measured nothing, not two."""
        events: list[Any] = []
        robot = _robot(events, toggle=False, hold=HoldEvidence.EMPTY, carried=None)
        with self.assertLogs(_LOGGER, level="WARNING") as said:
            robot.pick(_tilted(30.0), 40.0)
        self.assertEqual(1, len(said.output), said.output)

    def test_a_pick_that_ran_says_nothing_at_warning(self) -> None:
        robot = _robot([], carried=None)
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            self.assertTrue(robot.pick(_tilted(30.0), 40.0).ok)

    def test_the_handling_log_reaches_the_robot_log(self) -> None:
        from src.robot.execution import handling  # noqa: F401  (the logger is built at import)

        files = [getattr(h, "baseFilename", "") for h in logging.getLogger(_LOGGER).handlers]
        self.assertTrue(any(f.replace("\\", "/").endswith("/robot.log") for f in files), files)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
