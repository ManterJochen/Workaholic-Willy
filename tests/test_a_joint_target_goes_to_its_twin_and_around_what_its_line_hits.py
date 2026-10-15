"""A joint target on a cuRobo cell runs as its full turn inside the cable window, by its line or by a plan around it.

What a joint move on a cuRobo cell does since 85f082b, held here on the kit's planned arm with the route planner double
(``tests/_route_planner.py``) and the real destination guards. The local path gate has its own tests
(``tests/test_a_joint_move_is_judged_as_a_path.py``) and stands down here.

* A joint outside the window the planner and the joint-limit guard admit goes onto its full turn nearest the arm
  inside it: a station stored at -200 degrees on a wrist is the pose +160 is, and with the half turn about home the
  guard would refuse the number and not the pose. A joint already inside the window is left as the caller wrote it,
  and the log says what was turned. A value past the range the description rates its joint for is never turned: it is
  a target written in degrees, and turning it would make a plausible configuration out of a typing mistake.
* The destination guards judge the target before a line is sampled or the planner asked (``gate_joint_target``), on a
  cuRobo cell as on an ik cell. A value past the range the arm's description rates its joint for is refused there,
  JOINT_LIMIT_REJECTED, and the refusal says it reads as degrees: no joint of the arm reaches it in radians, and the
  pendant shows joints in degrees. The range is read per joint off the description (``limit_deg``), a full turn either
  way on every UR today, so a joint another vendor rates past a turn is held to its own range and not to a UR's (the
  owner, 2026-09-25; the bound was a full turn on every arm before).
* The straight joint line from where the arm stands is judged by both authorities, the planner at
  ``safety.planned_motion.line_clearance_mm`` from its world rather than at no contact: the line runs as one ``moveJ``
  nobody planned, and a line that grazes an obstacle by a millimetre passes a check that asks only for no contact.
  Where it passes it runs as that one ``moveJ``.
* Where the line collides, by either authority, cuRobo plans to the same target in joint space instead of the move
  being refused; the plan is held to ``safety.planned_motion.max_detour_deg``, its legs are judged, and exactly that
  list runs through the planner glue, at the speed ``robot.motion_limits`` allows. A line refused for anything but a
  collision, and a planner that cannot be reached for one, is not planned around.
"""

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest import mock

import numpy as np

from src.config.arms import load_arm
from src.config.schema.robot import RobotConfig
from src.robot.core import JointPositions, MotionCommand, MotionResult, MotionSent, MotionStatus
from src.robot.core.errors import RobotMotionRejected
from src.robot.safety.planning import CuroboUnavailableError, StateRefusal, StateRefusalKind, StateWhere
from tests._arm_kit import LOGGER_NAME, kit_tree, planned_arm
from tests._fake_transport import chain_of, fake_transport
from tests._plan_end import OPEN_WORKSPACE
from tests._route_planner import RoutePlanner, line

_DECLINED = "unit double: this file reads how a joint target is judged and run, and no camera world is wired"

#: Clear, and where the fake controller stands.
_HERE = [0.0, -1.5, 1.5, 0.0, 0.0, 0.0]
#: Clear, a short joint move away.
_THERE = [0.2, -1.5, 1.5, 0.0, 0.0, 0.0]
#: The shipped UR10 home copied off the pendant in degrees and handed over as radians.
_DEGREES = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]


def _arm(planner: RoutePlanner, *, safety: "dict[str, Any] | None" = None,
         home: "list[float] | None" = None) -> Any:
    config = kit_tree("curobo", safety={"payload": {"enforce": False}, **(safety or {})},
                      workspace_limits=OPEN_WORKSPACE)
    arm: Any = planned_arm(config, transport=fake_transport(joints=planner.here, kinematics=chain_of("ur5e")),
                           home_joints=home)
    arm._curobo = planner
    # The local path gate has its own tests; the destination guards stay real.
    arm._preflight.gate_planned_path = lambda waypoints, *, arm=None, command=None: None
    return arm


def _move(arm: Any, joints: "list[float]") -> Any:
    with arm.without_camera_world(_DECLINED):
        return arm.move_to_joints(JointPositions(joints))


def _rated(axis: int, limit_deg: "tuple[float, float]") -> Any:
    """The ur5e description with joint ``axis`` rated ``limit_deg``, as another vendor's wrist might be."""
    real = load_arm("ur5e")
    joints = list(real.joints)
    joints[axis] = joints[axis].model_copy(update={"limit_deg": limit_deg})
    return real.model_copy(update={"joints": type(real.joints)(joints)})


class TheTwinTests(unittest.TestCase):
    def test_a_joint_outside_the_cable_window_runs_as_its_twin_inside_and_the_log_says_so(self) -> None:
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner, safety={"joint_limits": {"within_half_turn_of_home": True}}, home=list(_HERE))
        station = [*_THERE[:5], math.radians(-200.0)]
        with self.assertLogs(LOGGER_NAME, level="INFO") as logs:
            result = _move(arm, station)
        self.assertTrue(result.ok, result.message)
        (sent,) = arm._transport.sent
        self.assertAlmostEqual(math.radians(160.0), sent[5], places=12)
        np.testing.assert_allclose(sent[:5], station[:5], atol=0.0)
        np.testing.assert_allclose(result.target_joints.values, sent, atol=0.0)  # the result names what ran
        (lined,) = planner.lines()
        self.assertAlmostEqual(math.radians(160.0), lined[1][-1][5], places=12, msg="the line judged is the one run")
        said = "\n".join(logs.output)
        self.assertIn("wrist_3_joint -200.0 -> 160.0 deg", said)
        self.assertIn("twin wrap", said)

    def test_the_same_station_is_refused_where_the_arm_does_not_turn_it(self) -> None:
        """The control, as the arm stood before the wrap: the guard refuses the number, which is not the pose."""
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner, safety={"joint_limits": {"within_half_turn_of_home": True}}, home=list(_HERE))
        arm._twin_near_the_arm = lambda joints, here: (joints, "")
        result = _move(arm, [*_THERE[:5], math.radians(-200.0)])
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status, result.message)
        self.assertEqual([], arm._transport.sent)

    def test_a_joint_inside_the_window_is_left_as_written(self) -> None:
        """Without the half turn the window is two turns wide; a turn inside it is the caller's to choose. Wrist 3 at
        -1.0 with the arm at 5.0: its other turn, 5.283, is inside the window too and nearer the arm, and not taken."""
        planner = RoutePlanner(here=[*_HERE[:5], 5.0])
        arm = _arm(planner)
        target = [*_THERE[:5], -1.0]
        self.assertEqual((JointPositions(target), ""), arm._twin_near_the_arm(JointPositions(target), planner.here))
        result = _move(arm, target)
        self.assertTrue(result.ok, result.message)
        self.assertEqual([target], arm._transport.sent)

    def test_the_raise_refuses_the_same_target_the_same_way(self) -> None:
        arm = _arm(RoutePlanner(here=_HERE))
        with self.assertRaises(RobotMotionRejected) as caught:
            _move(arm, _DEGREES).raise_unless_executed()
        result = caught.exception.result
        assert isinstance(result, MotionResult)
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status)
        self.assertEqual([], arm._transport.sent)

    def test_a_target_written_in_degrees_is_never_turned_onto_a_twin(self) -> None:
        arm = _arm(RoutePlanner(here=_HERE))
        self.assertEqual("", arm._twin_near_the_arm(JointPositions(_DEGREES), _HERE)[1],
                         "a value past the rated range was turned onto a twin")

    def test_no_joint_of_a_target_in_degrees_is_turned_and_the_refusal_names_what_was_written(self) -> None:
        """Wrist 3 at 5 rad is inside its rating and outside the half turn about home, so alone it would turn; beside
        four values in degrees it is left as written, and the refusal says what the caller sent."""
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner, safety={"joint_limits": {"within_half_turn_of_home": True}}, home=list(_HERE))
        written = [5.0, -90.0, 90.0, -90.0, -90.0, 0.0]
        with self.assertNoLogs(LOGGER_NAME, level="INFO"):
            turned, said = arm._twin_near_the_arm(JointPositions(written), _HERE)
        self.assertEqual((JointPositions(written), ""), (turned, said))
        result = _move(arm, written)
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status, result.message)
        np.testing.assert_allclose(result.target_joints.values, written, atol=0.0)
        self.assertEqual([], arm._transport.sent)

    def test_a_description_that_cannot_be_read_turns_nothing_and_raises_nothing(self) -> None:
        from src.config.loader import ConfigError

        arm = _arm(RoutePlanner(here=_HERE))
        station = JointPositions([*_THERE[:5], 7.0])
        with mock.patch("src.config.arms.load_arm", side_effect=ConfigError("no arm 'ur5e' here")):
            self.assertEqual((station, ""), arm._twin_near_the_arm(station, _HERE))

    def test_a_joint_rated_past_a_turn_is_turned_where_a_urs_is_not(self) -> None:
        """The owner's decision: the bound is the description's, per joint. 7 rad on a UR's wrist 3 is past its rating
        and never turned; on a wrist rated a turn and a quarter each way it is the pose 7 - 2 pi is, and turns onto it."""
        arm = _arm(RoutePlanner(here=_HERE))
        seven = JointPositions([*_HERE[:5], 7.0])
        self.assertEqual("", arm._twin_near_the_arm(seven, _HERE)[1])
        with mock.patch("src.config.arms.load_arm", return_value=_rated(5, (-450.0, 450.0))):
            turned, said = arm._twin_near_the_arm(seven, _HERE)
        self.assertAlmostEqual(7.0 - 2.0 * math.pi, float(turned.values[5]), places=12)
        self.assertIn("wrist_3_joint", said)


class ThePlanAroundTests(unittest.TestCase):
    def test_a_line_that_collides_is_planned_around_and_the_judged_list_runs(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        corner = [_HERE[0] + 0.1, _HERE[1] - 0.3, *_HERE[2:]]
        planner.answer = lambda goal: line(_HERE, corner, 11) + line(corner, goal, 11)[1:]
        arm = _arm(planner)
        with self.assertLogs(LOGGER_NAME, level="INFO") as logs:
            result = _move(arm, _THERE)
        self.assertTrue(result.ok, result.message)
        self.assertIs(MotionCommand.MOVE_JOINTS, result.command)
        self.assertEqual("move_to_joints", result.message)
        np.testing.assert_allclose(result.target_joints.values, _THERE, atol=0.0)
        (asked,) = planner.named("plan_joint")
        np.testing.assert_allclose(asked[1], _THERE, atol=0.0)
        self.assertIs(False, asked[2], "the plan refreshed the world the line was judged in")
        (execute,) = planner.named("execute")
        self.assertEqual(3, len(execute[1]), "the plan's corner was dropped, or the plan was not shortened")
        self.assertIs(MotionCommand.MOVE_JOINTS, execute[2])
        np.testing.assert_allclose(execute[3].values, _THERE, atol=0.0)
        self.assertEqual([], arm._transport.sent, "a waypoint went round the planner glue's execute")
        self.assertIn("move_to_joints: cuRobo plan around the straight line, which was refused (self_collision_rejected",
                      "\n".join(logs.output))

    def test_a_line_the_local_gate_refuses_for_a_collision_is_planned_around_too(self) -> None:
        """The robot, the fixtures and the planner's world all read SELF_COLLISION_REJECTED, and all are gone around.
        The gate refuses the straight line whenever it is shown it, so what runs is the plan's corner."""
        planner = RoutePlanner(here=_HERE)
        corner = [_HERE[0] + 0.1, _HERE[1] - 0.3, *_HERE[2:]]
        planner.answer = lambda goal: line(_HERE, corner, 11) + line(corner, goal, 11)[1:]
        arm = _arm(planner)
        straight = [list(_HERE), list(_THERE)]

        def gate(waypoints: Any, *, arm: Any = None, command: Any = None) -> "MotionResult | None":
            if [[float(v) for v in w] for w in waypoints] == straight:
                return MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, command,
                                           message="[safety:self_collision/fixture] the line grazes the tote")
            return None

        arm._preflight.gate_planned_path = gate
        result = _move(arm, _THERE)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(1, len(planner.named("plan_joint")))
        (execute,) = planner.named("execute")
        self.assertNotEqual(straight, [list(w) for w in execute[1]], "the plan ran as the line the gate refused")

    def test_a_plan_whose_legs_are_refused_runs_nothing(self) -> None:
        """Its legs meet both authorities as a Cartesian plan's do: the gate refusing every list refuses the move."""
        planner = RoutePlanner(here=_HERE)
        corner = [_HERE[0] + 0.1, _HERE[1] - 0.3, *_HERE[2:]]
        planner.answer = lambda goal: line(_HERE, corner, 11) + line(corner, goal, 11)[1:]
        arm = _arm(planner)
        arm._preflight.gate_planned_path = lambda waypoints, *, arm=None, command=None: MotionResult.failed(
            MotionStatus.SELF_COLLISION_REJECTED, command, message="[safety:self_collision/fixture] the tote")
        result = _move(arm, _THERE)
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status, result.message)
        self.assertEqual(1, len(planner.named("plan_joint")))
        self.assertEqual([], planner.named("execute"))
        self.assertIs(MotionSent.NOTHING, result.sent)
        self.assertEqual([], arm._transport.sent)

    def test_a_planned_route_the_controller_does_not_complete_says_so_in_its_own_words(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        refused = MotionResult.failed(MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_JOINTS,
                                      message="UR moveJ rejected waypoint 2 of 3", sent=MotionSent.SENT)
        planner.execute = lambda traj, pose=None, **_: refused  # type: ignore[method-assign]
        arm = _arm(planner)
        result = _move(arm, _THERE)
        # The verb stamps its camera world on the result; the rest is the glue's, not rewritten on its way out.
        self.assertEqual((refused.status, refused.message, refused.sent), (result.status, result.message, result.sent))

    def test_every_refusal_of_the_judge_says_nothing_was_sent(self) -> None:
        """A gate's refusal does not say whether a command went out; the judge sends none, so the verb says so."""
        for name, setup in (
            ("destination", lambda arm, planner: None),
            ("line and no plan", lambda arm, planner: (setattr(planner, "line_clear", lambda s, e: False),
                                                       setattr(planner, "answer", lambda goal: None))),
        ):
            with self.subTest(refusal=name):
                planner = RoutePlanner(here=_HERE)
                arm = _arm(planner)
                setup(arm, planner)
                result = _move(arm, _DEGREES if name == "destination" else _THERE)
                self.assertFalse(result.ok, result.message)
                self.assertIs(MotionSent.NOTHING, result.sent, result.message)

    def test_a_transport_lost_while_cuRobo_plans_is_a_typed_connection_error(self) -> None:
        """Found porting 85f082b to dev, 2026-09-25: ``plan_joint`` reads where the arm stands again, and a transport
        that fails there raised out of the verb instead of the typed result the first read gives."""
        for error in (RuntimeError("RTDE receive interface stopped"), OSError("connection reset")):
            with self.subTest(type(error).__name__):
                planner = RoutePlanner(here=_HERE)
                planner.line_clear = lambda start, end: False

                def lost(goal: Any, error: Exception = error) -> Any:
                    raise error

                planner.answer = lost
                arm = _arm(planner)
                result = _move(arm, _THERE)
                self.assertIs(MotionStatus.CONNECTION_ERROR, result.status, result.message)
                self.assertIn("nothing was sent", result.message or "")
                self.assertIs(MotionSent.NOTHING, result.sent)
                self.assertEqual([], planner.named("execute"))
                self.assertEqual([], arm._transport.sent)

    def test_a_planned_route_runs_and_raises_nothing(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        arm = _arm(planner)
        _move(arm, _THERE).raise_unless_executed()
        self.assertEqual(1, len(planner.named("execute")))

    def test_a_line_the_local_gate_refuses_for_anything_else_is_not_planned_around(self) -> None:
        """The control, green before the plan around too: a joint-limit sample on a straight line means an end is
        outside, and no plan fixes that."""
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)
        arm._preflight.gate_planned_path = lambda waypoints, *, arm=None, command=None: MotionResult.failed(
            MotionStatus.JOINT_LIMIT_REJECTED, command, message="[safety:joint_limit] sample 5 of 9 leaves the window")
        result = _move(arm, _THERE)
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status, result.message)
        self.assertEqual([], planner.named("plan_joint"))
        self.assertEqual([], arm._transport.sent)

    def test_a_plan_that_shortens_back_into_the_refused_line_runs_as_cuRobo_returned_it(self) -> None:
        """Found porting 85f082b to dev, 2026-09-25: cuRobo's plan barely bends, and shortened it is the line refused
        at the clearance. Judged at that clearance it is refused again, and the plan runs as cuRobo returned it, at no
        contact. Before, the shortened list was judged at no contact, passed, and the refused line ran."""
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        dense = line(_HERE, _THERE, 21)
        planner.answer = lambda goal: [list(w) for w in dense]
        arm = _arm(planner)
        result = _move(arm, _THERE)
        self.assertTrue(result.ok, result.message)
        (execute,) = planner.named("execute")
        self.assertEqual(21, len(execute[1]), "the plan ran shortened into the line the planner refused")
        self.assertEqual([3.0, 3.0, 0.0], [c[3] for c in planner.named("check") if len(c[1]) > 1])

    def test_home_is_planned_around_too_and_reads_move_home(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        arm = _arm(planner, home=list(_THERE))
        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_home()
        self.assertTrue(result.ok, result.message)
        self.assertIs(MotionCommand.MOVE_HOME, result.command)
        (execute,) = planner.named("execute")
        self.assertIs(MotionCommand.MOVE_HOME, execute[2])

    def test_a_plan_past_the_detour_bound_is_refused_naming_the_joint(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        swing = [*_HERE[:5], math.radians(80.0)]
        planner.answer = lambda goal: [list(_HERE), swing, list(goal)]
        arm = _arm(planner)
        result = _move(arm, _THERE)
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status, result.message)
        self.assertIn("swings wrist_3_joint 80.0 deg beyond the span", result.message or "")
        self.assertIn("max_detour_deg of 45", result.message or "")
        self.assertEqual([], planner.named("execute"))
        self.assertEqual([], arm._transport.sent)

    def test_no_plan_around_a_colliding_line_is_the_lines_refusal(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        planner.answer = lambda goal: None
        planner.refusal_on_a_line = StateRefusal(where=StateWhere.GOAL, kind=StateRefusalKind.WORLD,
                                                 joints=tuple(_THERE), depth_mm=3.0)
        arm = _arm(planner)
        result = _move(arm, _THERE)
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status, result.message)
        self.assertIn("the planner refused this straight joint line", result.message or "")
        self.assertIn("no plan goes around it", result.message or "")
        self.assertEqual([], arm._transport.sent)

    def test_a_start_the_planner_will_not_leave_is_typed(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        planner.answer = lambda goal: None
        planner.refusal_on_no_plan = StateRefusal(where=StateWhere.START, kind=StateRefusalKind.JOINT_LIMIT,
                                                  joints=tuple(_HERE))
        arm = _arm(planner)
        result = _move(arm, _THERE)
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status, result.message)
        self.assertIn("will not start from", result.message or "")
        self.assertEqual([], arm._transport.sent)

    def test_a_planner_that_cannot_be_reached_for_the_line_is_not_planned_around(self) -> None:
        """The control, green before the plan around too: an unreachable planner is no collision to plan around."""
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)

        def unavailable(*_: Any, **__: Any) -> Any:
            raise CuroboUnavailableError("no GPU env")

        planner.check_joint_path = unavailable  # type: ignore[method-assign]
        result = _move(arm, _THERE)
        self.assertIs(MotionStatus.CONTROLLER_REJECTED, result.status, result.message)
        self.assertEqual([], planner.named("plan_joint"))
        self.assertEqual([], arm._transport.sent)

    def test_a_planner_lost_while_it_plans_is_typed_and_sends_nothing(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False

        def lost(goal: Any) -> Any:
            raise CuroboUnavailableError("the sidecar exited")

        planner.answer = lost
        arm = _arm(planner)
        result = _move(arm, _THERE)
        self.assertIs(MotionStatus.CONTROLLER_REJECTED, result.status, result.message)
        self.assertIn("cuRobo planner unavailable", result.message or "")
        self.assertEqual([], arm._transport.sent)


class TheDestinationIsGatedFirstTests(unittest.TestCase):
    def test_a_target_written_in_degrees_is_refused_before_a_line_is_sampled_and_nothing_is_sent(self) -> None:
        """The safety gap this closes: a target in degrees met only the line's samples, which a double can pass."""
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)
        result = _move(arm, _DEGREES)
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status, result.message)
        self.assertIs(MotionCommand.MOVE_JOINTS, result.command)
        self.assertIn("reads as degrees", result.message or "")
        np.testing.assert_allclose(result.target_joints.values, _DEGREES, atol=0.0)
        self.assertEqual([], arm._transport.sent)
        self.assertEqual([], planner.named("check"), "a line to a target in degrees was sampled")
        self.assertEqual([], planner.named("plan_joint"))

    def test_the_degrees_sentence_names_each_joint_and_the_range_it_is_rated_for(self) -> None:
        result = _move(_arm(RoutePlanner(here=_HERE)), _DEGREES)
        said = result.message or ""
        for name, value in (("shoulder_lift_joint", -90), ("elbow_joint", 90), ("wrist_1_joint", -90),
                            ("wrist_2_joint", -90)):
            self.assertIn(f"{name} at {value} rad, rated -360 to 360 deg", said)
        self.assertNotIn("shoulder_pan_joint at", said, "a joint inside its range was named")

    def test_a_target_the_guard_refuses_inside_the_rated_range_is_refused_first_and_not_read_as_degrees(self) -> None:
        """Half a turn about home, less the guard's 5 degree margin: wrist 3 at pi from home stands in the band the margin
        keeps free either side of the seam, where no turn of it lies inside the window, so there is no twin to take and
        the guard refuses it; it is inside the full turn the UR is rated for, so it does not read as degrees."""
        target = [*_THERE[:5], math.pi]
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner, safety={"joint_limits": {"within_half_turn_of_home": True}}, home=list(_HERE))
        result = _move(arm, target)
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status, result.message)
        self.assertNotIn("reads as degrees", result.message or "")
        self.assertEqual([], planner.named("check"), "the line was sampled before the destination was judged")
        self.assertEqual([], arm._transport.sent)

    def test_the_same_target_runs_where_no_half_turn_is_kept(self) -> None:
        """The control: the refusal above is the guard's, not the target's."""
        target = [*_THERE[:5], math.pi]
        arm = _arm(RoutePlanner(here=_HERE))
        result = _move(arm, target)
        self.assertTrue(result.ok, result.message)
        self.assertEqual([target], arm._transport.sent)

    def test_the_world_is_refreshed_before_the_destination_is_judged_and_the_line_after_both(self) -> None:
        """The destination guards read the perceived obstacles the refresh hands them, so the refresh comes first."""
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)
        order: list[str] = []
        gate, refresh, check = arm._preflight.gate_joint_target, arm._refresh_before_the_path_gate, planner.check_joint_path
        arm._preflight.gate_joint_target = lambda joints, *, arm=None: (order.append("destination"),
                                                                        gate(joints, arm=arm))[1]
        arm._refresh_before_the_path_gate = lambda **asked: (order.append("refresh"), refresh(**asked))[1]
        planner.check_joint_path = lambda samples, **asked: (order.append("line"), check(samples, **asked))[1]
        result = _move(arm, _THERE)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(["refresh", "destination", "line"], order)

    def test_a_home_the_destination_gate_refuses_after_the_refresh_reads_move_home(self) -> None:
        """The home gate passed, and the destination gate after the refresh, which may hold perceived obstacles the
        home gate never saw, did not: the refusal is the home's."""
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner, home=list(_THERE))
        arm._home_refusal = lambda joints: None
        arm._preflight.gate_joint_target = lambda joints, *, arm=None: MotionResult.failed(
            MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_JOINTS, target_joints=joints,
            message="[safety:self_collision/perceived] the home stands in a perceived box")
        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_home()
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status, result.message)
        self.assertIs(MotionCommand.MOVE_HOME, result.command)
        self.assertEqual([], planner.named("check"))
        self.assertEqual([], arm._transport.sent)

    def test_a_home_written_in_degrees_names_each_joint_and_the_key_that_takes_degrees(self) -> None:
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner, home=list(_DEGREES))
        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_home()
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status, result.message)
        self.assertIs(MotionCommand.MOVE_HOME, result.command)
        self.assertIn("shoulder_lift_joint at -90 rad, rated -360 to 360 deg", result.message or "")
        self.assertIn("robot.home_joint_positions_deg takes the pendant's degrees", result.message or "")
        self.assertEqual([], arm._transport.sent)

    def test_the_rated_range_is_the_descriptions_per_joint(self) -> None:
        """A wrist rated a turn and a quarter each way reads 7 rad as radians; a UR's, rated one turn, as degrees."""
        arm = _arm(RoutePlanner(here=_HERE))
        seven = JointPositions([*_HERE[:5], 7.0])
        self.assertIn("wrist_3_joint at 7 rad, rated -360 to 360 deg", arm._read_as_degrees(seven))
        with mock.patch("src.config.arms.load_arm", return_value=_rated(5, (-450.0, 450.0))):
            self.assertEqual("", arm._read_as_degrees(seven))
        self.assertEqual("", arm._read_as_degrees(JointPositions(_THERE)))


class TheLineTests(unittest.TestCase):
    def test_a_clear_line_is_judged_at_the_configured_clearance_and_runs_as_one_movej(self) -> None:
        for clearance, safety in ((3.0, {}), (25.0, {"planned_motion": {"line_clearance_mm": 25.0}})):
            with self.subTest(clearance_mm=clearance):
                planner = RoutePlanner(here=_HERE)
                arm = _arm(planner, safety=safety)
                with self.assertLogs(LOGGER_NAME, level="INFO") as logs:
                    result = _move(arm, _THERE)
                self.assertTrue(result.ok, result.message)
                self.assertIn("move_to_joints: direct line; 2 waypoint(s) dense, 2 executed (1 leg(s))",
                              "\n".join(logs.output))
                (lined,) = planner.lines()
                self.assertEqual(clearance, lined[3])
                self.assertIs(False, lined[2], "the planner refreshed a world the path gate had already judged in")
                np.testing.assert_allclose(lined[1][0], _HERE, atol=0.0)
                np.testing.assert_allclose(lined[1][-1], _THERE, atol=0.0)
                self.assertEqual([_THERE], arm._transport.sent)
                self.assertEqual([], planner.named("plan_joint"))
                self.assertEqual([], planner.named("execute"))

    def test_home_is_judged_the_same_way_and_reads_move_home(self) -> None:
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner, home=list(_THERE))
        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_home()
        self.assertTrue(result.ok, result.message)
        self.assertIs(MotionCommand.MOVE_HOME, result.command)
        (lined,) = planner.lines()
        self.assertEqual(3.0, lined[3])
        self.assertEqual([_THERE], arm._transport.sent)


    def test_a_clearance_of_zero_judges_the_line_as_a_plan_is_judged(self) -> None:
        """The control, green before this change too: 0 asks the planner exactly as every joint line was asked."""
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False  # refuses only at a clearance, so 0 passes
        arm = _arm(planner, safety={"planned_motion": {"line_clearance_mm": 0.0}})
        result = _move(arm, _THERE)
        self.assertTrue(result.ok, result.message)
        self.assertEqual([], planner.lines())
        (checked,) = planner.named("check")
        self.assertEqual(0.0, checked[3])
        np.testing.assert_allclose(checked[1][-1], _THERE, atol=0.0)


class TheConfigTests(unittest.TestCase):
    def test_the_keys_default_to_the_planners_clearance_and_45_degrees(self) -> None:
        planned = RobotConfig.model_validate({"vendor": "ur"}).safety.planned_motion
        self.assertEqual((3.0, 45.0), (planned.line_clearance_mm, planned.max_detour_deg))

    def test_a_bound_of_nothing_and_a_clearance_the_planner_cannot_ask_are_refused(self) -> None:
        from pydantic import ValidationError

        for block in ({"max_detour_deg": 0.0}, {"max_detour_deg": -5.0}, {"line_clearance_mm": -1.0},
                      {"line_clearance_mm": 150.0}, {"surplus": 1}):
            with self.subTest(block=block), self.assertRaises(ValidationError):
                RobotConfig.model_validate({"vendor": "ur", "safety": {"planned_motion": block}})
        zero = RobotConfig.model_validate({"vendor": "ur", "safety": {"planned_motion": {"line_clearance_mm": 0.0}}})
        self.assertEqual(0.0, zero.safety.planned_motion.line_clearance_mm, "0 judges a line as a plan is judged")


if __name__ == "__main__":
    unittest.main()
