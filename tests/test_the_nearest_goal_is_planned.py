"""A Cartesian cuRobo move goes to the configuration the arm chooses, by the straight joint line where it is clear.

cuRobo, handed a pose, picks the goal configuration from random seeds over the whole joint range. Measured on a
UR10 on this project's GPU: a wrist sweep took 8 of 22 legs the long way, one turning the base 8.57 rad where 2.37
would do, and the same sweep planned twice took different ways. The UR driver now never hands cuRobo a pose. It solves
the flange goal in closed form, turns every solution onto the full turns nearest the arm inside the planner's and the
joint-limit guard's window, ranks them on the branch the arm holds first and by the least time a ``moveJ`` there takes
after it, and screens each with the planner and the endpoint gate. Then, in that order:

* the straight joint line to each admissible goal, judged by the local path gate and by the planner at
  ``safety.planned_motion.line_clearance_mm`` from its world; the first that passes runs as it is, one leg;
* only where no line passes, cuRobo plans in joint space to up to three of them, and a plan is taken where it ends on
  the configuration asked for and swings no joint more than ``safety.planned_motion.max_detour_deg`` beyond its span;
* a branch other than the arm's only once every goal on it failed, with a warning naming both;
* no line and no plan is the no-plan refusal, and cuRobo is never left to choose a goal.

What this file holds with a planner double: which configuration is asked for, in what order lines and plans are
tried, the clearance a line is judged at, the detour bound, the branch preference, and that what runs is what was
judged. The GPU half, with the real sidecar, is for the lead's bench run, not here.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.core import NO_PLAN_FAIL_SAFE_MESSAGE, MotionCommand, MotionResult, MotionStatus
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.drivers.ur.curobo_motion import PLANNER_JOINT_ENVELOPE_RAD, UR_ARM_JOINT_NAMES, CuroboUrPlanner
from src.robot.safety._ur_ik import nearest_goals, ur_branch, ur_flange_ik
from src.robot.safety._ur_kinematics import ur_link_transforms_mm
from src.robot.safety.path_samples import span_overshoot_rad
from src.robot.safety.planning import CuroboUnavailableError, StateRefusal, StateRefusalKind, StateWhere
from tests._plan_end import OPEN_WORKSPACE, pose_where_it_ends
from tests._route_planner import RoutePlanner, line

_DECLINED = "unit double: this file reads which goal a planner double is asked for, and no camera world is wired"
_ROOT = pathlib.Path(__file__).resolve().parents[1]

_HERE = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
_THERE = [0.3, -1.4, 1.4, -1.6, -1.5, 0.2]
#: The UR10 CB3 home: candle-straight, the elbow stretched and wrist 2 at 0, every branch point at once.
_CANDLE = [0.0, -math.pi / 2.0, 0.0, -math.pi / 2.0, 0.0, 0.0]


def _arm(planner: RoutePlanner, *, model: str = "ur5e", safety: "dict[str, Any] | None" = None) -> URRobotArm:
    arm = URRobotArm(RobotConfig.model_validate({
        "vendor": "ur", "ur": {"model": model, "motion_planner": "curobo"},
        "safety": {"payload": {"enforce": False}, **(safety or {})}, "gripper": {"model": "robotiq_2f85"},
        "workspace_limits": OPEN_WORKSPACE,
    }))
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.get_joint_positions.return_value = list(planner.here)
    arm._curobo_ur = planner  # type: ignore[assignment]
    # The local gate and the end have their own tests; this file reads which goal was asked for and how.
    arm._preflight.gate_planned_path = lambda waypoints, *, arm=None, command=None: None  # type: ignore[method-assign]
    arm._gate_planned_config = lambda pose, joints: None  # type: ignore[method-assign]
    return arm


def _move(arm: URRobotArm, end: "list[float]") -> MotionResult:
    with arm.without_camera_world(_DECLINED):
        return arm.move(pose_where_it_ends(arm, end))


def _ranked(arm: URRobotArm, end: "list[float]", here: "list[float]") -> list[Any]:
    """The order of the goals by time alone, from the pure function the arm calls."""
    frames = ur_link_transforms_mm(str(arm.config.ur.model), np.asarray(end, dtype=np.float64))
    assert frames is not None
    solutions = ur_flange_ik(str(arm.config.ur.model), frames[-1], q6_if_singular=here[-1])
    assert solutions is not None
    lower, upper = arm._goal_joint_window()
    return list(nearest_goals(solutions, current=here, lower=lower, upper=upper, velocity=1.0))


def _executed_goal(planner: RoutePlanner) -> "tuple[float, ...]":
    (execute,) = planner.named("execute")
    return execute[1][-1]


def _near(a: "Any", b: "Any", tol: float = 1e-6) -> bool:
    return max(abs(float(x) - float(y)) for x, y in zip(a, b)) < tol


class AClearLineRunsTests(unittest.TestCase):
    def test_a_clear_line_runs_to_the_nearest_configuration_as_one_leg_and_cuRobo_plans_nothing(self) -> None:
        planner = RoutePlanner(here=_HERE)
        result = _move(_arm(planner), _THERE)
        self.assertTrue(result.ok, result.message)
        self.assertEqual([], planner.named("plan_joint"), "cuRobo was asked to plan a line that was clear")
        (execute,) = planner.named("execute")
        self.assertEqual(2, len(execute[1]), "a straight line is its two ends and nothing else")
        np.testing.assert_allclose(execute[1][0], _HERE, atol=1e-12)
        np.testing.assert_allclose(execute[1][1], _THERE, atol=1e-6)

    def test_the_line_is_judged_at_the_configured_clearance_and_a_screen_at_none(self) -> None:
        for clearance, safety in ((10.0, {}), (25.0, {"planned_motion": {"line_clearance_mm": 25.0}})):
            with self.subTest(clearance_mm=clearance):
                planner = RoutePlanner(here=_HERE)
                _move(_arm(planner, safety=safety), _THERE)
                (lined,) = planner.lines()
                self.assertEqual(clearance, lined[3])
                np.testing.assert_allclose(lined[1][0], _HERE, atol=1e-12)
                np.testing.assert_allclose(lined[1][-1], _THERE, atol=1e-6)
                screens = [c for c in planner.named("check") if len(c[1]) == 1]
                self.assertTrue(screens and all(c[3] == 0.0 for c in screens))

    def test_a_joint_goes_on_its_full_turn_nearest_the_arm(self) -> None:
        """Wrist 3 stands at 5.0; the goal's wrist 3 angle is -0.983, and 5.3 is the same angle one turn on."""
        here = [*_HERE[:5], 5.0]
        end = [*_THERE[:5], 5.3]
        planner = RoutePlanner(here=here)
        _move(_arm(planner), end)
        self.assertAlmostEqual(5.3, _executed_goal(planner)[5], places=6)
        np.testing.assert_allclose(_executed_goal(planner), end, atol=1e-6)

    def test_a_nearer_turn_outside_the_planners_window_is_not_asked_for(self) -> None:
        """Wrist 1 at 6.1 rad: the goal's 0.4 + 2 pi is nearer, and past 2 pi - 0.1, so the planner never sees it."""
        here = [_HERE[0], _HERE[1], _HERE[2], 6.1, _HERE[4], _HERE[5]]
        end = [_THERE[0], _THERE[1], _THERE[2], 0.4, _THERE[4], _THERE[5]]
        planner = RoutePlanner(here=here)
        planner.line_clear = lambda start, end: False
        _move(_arm(planner), end)
        lower, upper = PLANNER_JOINT_ENVELOPE_RAD
        asked = ([c[1] for c in planner.named("plan_joint")] + [c[1][0] for c in planner.named("check") if len(c[1]) == 1]
                 + [c[1][-1] for c in planner.lines()])
        self.assertTrue(planner.named("plan_joint"), "no joint goal was asked for, so nothing here was tested")
        for goal in asked:
            for axis, value in enumerate(goal):
                self.assertTrue(lower[axis] <= value <= upper[axis], f"{UR_ARM_JOINT_NAMES[axis]} asked at {value}")

    def test_the_guards_window_narrows_the_planners(self) -> None:
        """A cell that limits wrist 3 to +-100 deg (less a 5 deg margin) is never asked to go one turn on."""
        limits = {"joint_limits": {"min_deg": [-360.0] * 5 + [-100.0], "max_deg": [360.0] * 5 + [100.0]}}
        here = [*_HERE[:5], 1.5]
        end = [*_THERE[:5], 1.2]
        planner = RoutePlanner(here=here)
        arm = _arm(planner, safety=limits)
        self.assertAlmostEqual(math.radians(95.0), arm._goal_joint_window()[1][5])
        _move(arm, end)
        self.assertAlmostEqual(1.2, _executed_goal(planner)[5], places=6)

    def test_the_elbow_is_never_asked_past_the_planners_half_turn(self) -> None:
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)
        for goal in _ranked(arm, _THERE, _HERE):
            self.assertLessEqual(abs(goal.joints[2]), math.pi - 0.1)


class WhatIsRefusedBeforeAnythingIsAskedTests(unittest.TestCase):
    def test_a_goal_no_turn_of_which_the_guard_admits_is_refused_and_cuRobo_is_never_asked(self) -> None:
        limits = {"joint_limits": {"min_deg": [-360.0] * 5 + [-30.0], "max_deg": [360.0] * 5 + [30.0]}}
        planner = RoutePlanner(here=[*_HERE[:5], 0.0])
        result = _move(_arm(planner, safety=limits), [*_THERE[:5], 2.0])
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status)
        self.assertIn("joint window", result.message or "")
        self.assertIn("never left to choose", result.message or "")
        self.assertEqual([], planner.named("plan_joint"))
        self.assertEqual([], planner.named("execute"))

    def test_a_pose_out_of_reach_is_an_ik_failure_and_nothing_is_asked(self) -> None:
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)
        far = pose_where_it_ends(arm, _THERE)
        far = type(far)(position_mm=np.asarray(far.position_mm) * 5.0, quaternion_xyzw=far.quaternion_xyzw,
                        frame=far.frame, label="five times out")
        with arm.without_camera_world(_DECLINED):
            result = arm.move(far)
        self.assertIs(MotionStatus.IK_FAILED, result.status)
        self.assertEqual([], planner.calls)

    def test_the_glue_has_no_cartesian_plan_so_cuRobo_can_never_choose_a_goal(self) -> None:
        self.assertFalse(hasattr(CuroboUrPlanner, "plan"))
        self.assertFalse(hasattr(CuroboUrPlanner, "plans_joint_goals"))


class ScreenedLinedAndPlannedTests(unittest.TestCase):
    def test_a_goal_the_planner_refuses_is_neither_lined_nor_planned_to(self) -> None:
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)
        ranked = _ranked(arm, _THERE, _HERE)
        first = ranked[0].joints
        planner.screen = lambda config: not _near(config, first, 1e-9)
        _move(arm, _THERE)
        for lined in planner.lines():
            self.assertFalse(_near(lined[1][-1], first, 1e-9), "a goal the planner refused was lined")
        screens = [c for c in planner.named("check") if len(c[1]) == 1]
        self.assertTrue(all(c[2] is False for c in screens), "a screen refreshed the world the line was judged in")

    def test_a_goal_the_endpoint_gate_refuses_is_neither_lined_nor_planned_to(self) -> None:
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)
        first = _ranked(arm, _THERE, _HERE)[0].joints
        refusal = MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO, message="the spy")
        arm._gate_planned_config = (  # type: ignore[method-assign]
            lambda pose, joints: refusal if _near(joints.tolist(), first, 1e-9) else None)
        result = _move(arm, _THERE)
        self.assertTrue(result.ok, result.message)
        self.assertFalse(_near(_executed_goal(planner), first, 1e-9))

    def test_every_configuration_the_endpoint_gate_refuses_is_that_gates_refusal(self) -> None:
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)
        refusal = MotionResult.failed(MotionStatus.WORKSPACE_REJECTED, MotionCommand.MOVE_TO, message="the box")
        arm._gate_planned_config = lambda pose, joints: refusal  # type: ignore[method-assign]
        result = _move(arm, _THERE)
        self.assertIs(MotionStatus.WORKSPACE_REJECTED, result.status)
        self.assertIn("passes the endpoint gate", result.message or "")
        self.assertEqual([], planner.named("plan_joint"))

    def test_a_line_that_comes_too_close_is_planned_around_to_the_same_goal(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        result = _move(_arm(planner), _THERE)
        self.assertTrue(result.ok, result.message)
        (asked,) = planner.named("plan_joint")
        np.testing.assert_allclose(asked[1], _THERE, atol=1e-6)
        self.assertIs(False, asked[2], "the plan refreshed the world the line and the screens were judged in")
        # The plan is a straight 21-waypoint line. Shortened to its two ends it is the line refused at the clearance,
        # and judged at that clearance it is refused again, so the plan runs as cuRobo returned it (found porting
        # 85f082b to dev, 2026-09-25; before, it ran as the refused line).
        self.assertEqual(21, len(planner.named("execute")[0][1]))

    def test_a_line_the_local_gate_refuses_is_planned_around(self) -> None:
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)
        refused = MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO, message="bench")
        judged: list[Any] = []

        def gate(waypoints: Any, *, arm: Any = None, command: Any = None) -> "MotionResult | None":
            judged.append([tuple(w) for w in waypoints])
            return refused if len(judged) == 1 else None

        arm._preflight.gate_planned_path = gate  # type: ignore[method-assign]
        result = _move(arm, _THERE)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(1, len(planner.named("plan_joint")))
        before = planner.calls[:planner.calls.index(planner.named("plan_joint")[0])]
        self.assertEqual([], [c for c in before if c[0] == "check" and c[3] > 0.0],
                         "a line the local gate refused was still sent to the planner")

    def test_every_line_is_tried_before_any_plan(self) -> None:
        """Out of a candle-straight home every goal is on the arm's branch; each is lined before any is planned."""
        planner = RoutePlanner(here=_CANDLE)
        planner.line_clear = lambda start, end: False
        planner.answer = lambda goal: None
        _move(_arm(planner), _THERE)
        kinds = [c[0] if c[0] != "check" else ("line" if c[3] > 0.0 else "screen") for c in planner.calls]
        self.assertGreater(kinds.count("line"), 1, "the test needs more than one goal on the branch")
        self.assertLess(max(i for i, k in enumerate(kinds) if k == "line"), kinds.index("plan_joint"))
        self.assertEqual(3, kinds.count("plan_joint"), "no more than three goals are planned to")

    def test_what_runs_is_the_planners_own_trajectory_shortened(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        corner = [_HERE[0] + 0.3, *_HERE[1:]]
        returned: list[list[list[float]]] = []

        def answer(goal: "list[float]") -> "list[list[float]]":
            trajectory = line(_HERE, corner, 11) + line(corner, goal, 11)[1:]
            returned.append(trajectory)
            return trajectory

        planner.answer = answer
        _move(_arm(planner), _THERE)
        (execute,) = planner.named("execute")
        dense = [tuple(w) for w in returned[0]]
        self.assertEqual(3, len(execute[1]), "the corner of a bent plan was dropped or the plan was not shortened")
        self.assertTrue(all(w in dense for w in execute[1]), "a waypoint ran that cuRobo never returned")

    def test_a_plan_that_ends_off_the_goal_it_was_asked_for_is_not_taken(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        planner.answer = lambda goal: line(_HERE, [goal[0] + 1e-3, *goal[1:]], 5)
        result = _move(_arm(planner), _THERE)
        self.assertIs(MotionStatus.TIMEOUT, result.status)
        self.assertEqual(NO_PLAN_FAIL_SAFE_MESSAGE, result.message)
        self.assertEqual([], planner.named("execute"))

    def test_a_plan_inside_the_tolerance_is_taken(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        planner.answer = lambda goal: line(_HERE, [goal[0] + 5e-5, *goal[1:]], 5)
        result = _move(_arm(planner), _THERE)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(1, len(planner.named("plan_joint")))

    def test_no_line_and_no_plan_is_the_no_plan_refusal_and_the_log_names_every_goal(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        planner.answer = lambda goal: None
        arm = _arm(planner)
        with self.assertLogs("URRobotArm", level="ERROR") as logs:
            result = _move(arm, _THERE)
        self.assertIs(MotionStatus.TIMEOUT, result.status)
        self.assertEqual(NO_PLAN_FAIL_SAFE_MESSAGE, result.message)
        self.assertLessEqual(len(planner.named("plan_joint")), 3)
        said = "\n".join(logs.output)
        self.assertIn("did not plan", said)
        self.assertIn("the straight line to goal 1 of", said)
        self.assertEqual([], planner.named("execute"))

    def test_a_transport_lost_while_cuRobo_plans_is_a_typed_connection_error(self) -> None:
        """Found porting 85f082b to dev, 2026-09-25: ``plan_joint`` reads where the arm stands again, and a transport
        that fails there raised out of ``move`` instead of the typed result the first read gives."""
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
                self.assertEqual([], planner.named("execute"))
                arm._conn.moveJ.assert_not_called()

    def test_a_planner_lost_while_it_plans_is_still_the_planner_unavailable(self) -> None:
        """⭐ THE CONTROL: ``CuroboUnavailableError`` is a ``RuntimeError``, and the transport catch does not take it."""
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False

        def gone(goal: Any) -> Any:
            raise CuroboUnavailableError("the sidecar exited")

        planner.answer = gone
        result = _move(_arm(planner), _THERE)
        self.assertIs(MotionStatus.CONTROLLER_REJECTED, result.status, result.message)
        self.assertIn("cuRobo planner unavailable", result.message or "")
        self.assertEqual([], planner.named("execute"))

    def test_a_start_the_planner_refuses_stops_the_search_and_is_reported_typed(self) -> None:
        refusal = StateRefusal(where=StateWhere.START, kind=StateRefusalKind.SELF_COLLISION, joints=tuple(_HERE))
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        planner.answer = lambda goal: None
        planner.refusal_on_no_plan = refusal
        result = _move(_arm(planner), _THERE)
        self.assertEqual(1, len(planner.named("plan_joint")))
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status)
        self.assertIn("will not start from", result.message or "")

    def test_the_world_is_refreshed_once_near_the_goal_and_everything_after_uses_it(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        arm = _arm(planner)
        arm._live_world = object()  # type: ignore[assignment]  # wired, so the refresh is asked; the double holds none
        pose = pose_where_it_ends(arm, _THERE)
        result = arm._drive_curobo(pose)
        self.assertTrue(result.ok, result.message)
        self.assertEqual("refresh", planner.calls[0][0])
        (refresh,) = planner.named("refresh")
        np.testing.assert_allclose(refresh[1], arm._pose_to_flange(pose).position_mm)
        self.assertTrue(all(call[2] is False for call in planner.named("check")))
        self.assertTrue(all(call[2] is False for call in planner.named("plan_joint")))

    def test_the_route_log_names_how_the_goal_was_reached(self) -> None:
        planner = RoutePlanner(here=_HERE)
        with self.assertLogs("URRobotArm", level="INFO") as logs:
            _move(_arm(planner), _THERE)
        said = "\n".join(logs.output)
        self.assertIn("direct line to goal 1 of", said)
        self.assertIn("2 waypoint(s) dense, 2 executed (1 leg(s))", said)


class TheDetourBoundTests(unittest.TestCase):
    @staticmethod
    def _swinging(goal: "list[float]", *, degrees: float) -> "list[list[float]]":
        """A plan to ``goal`` whose wrist 3 swings ``degrees`` past the goal's side of its span on the way."""
        out = line(_CANDLE, goal, 9)
        high = max(_CANDLE[5], goal[5]) + math.radians(degrees)
        out[4] = [*out[4][:5], high]
        return out

    def _planner(self, *, degrees_first: float, degrees_rest: "float | None") -> RoutePlanner:
        planner = RoutePlanner(here=_CANDLE)
        planner.line_clear = lambda start, end: False
        asked: list[int] = []

        def answer(goal: "list[float]") -> "list[list[float]] | None":
            asked.append(1)
            if len(asked) == 1:
                return self._swinging(goal, degrees=degrees_first)
            return None if degrees_rest is None else self._swinging(goal, degrees=degrees_rest)

        planner.answer = answer
        return planner

    def test_a_plan_past_the_bound_is_not_taken_and_the_next_goal_is_planned(self) -> None:
        planner = self._planner(degrees_first=60.0, degrees_rest=10.0)
        with self.assertLogs("URRobotArm", level="INFO") as logs:
            result = _move(_arm(planner), _THERE)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(2, len(planner.named("plan_joint")))
        self.assertIn("swings wrist_3_joint 60.0 deg beyond the span", "\n".join(logs.output))
        np.testing.assert_allclose(_executed_goal(planner), planner.named("plan_joint")[1][1], atol=1e-9)

    def test_when_every_plan_swings_too_far_the_move_is_refused_naming_the_joint(self) -> None:
        planner = self._planner(degrees_first=60.0, degrees_rest=90.0)
        result = _move(_arm(planner), _THERE)
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status)
        self.assertIn("wrist_3_joint 90.0 deg", result.message or "", "the refusal names the worst detour")
        self.assertIn("max_detour_deg", result.message or "")
        self.assertEqual([], planner.named("execute"))

    def test_the_bound_is_the_configured_one(self) -> None:
        planner = self._planner(degrees_first=60.0, degrees_rest=None)
        result = _move(_arm(planner, safety={"planned_motion": {"max_detour_deg": 75.0}}), _THERE)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(1, len(planner.named("plan_joint")))


class SpanOvershootTests(unittest.TestCase):
    """``span_overshoot_rad``: how far each joint of a path swings past the span between its two ends."""

    def test_a_straight_line_overshoots_nothing(self) -> None:
        self.assertEqual((0.0,) * 6, span_overshoot_rad(line(_HERE, _THERE, 11)))

    def test_a_joint_that_swings_out_and_back_overshoots_by_how_far_it_went(self) -> None:
        out = [*_HERE[:5], 1.0]
        self.assertEqual((0.0,) * 5 + (1.0,), span_overshoot_rad([_HERE, out, _HERE]))
        below = [*_HERE[:5], -0.4]
        path = [_HERE, below, [*_HERE[:5], 0.3]]
        self.assertAlmostEqual(0.4, span_overshoot_rad(path)[5], places=12, msg="past the low end counts as well")

    def test_a_subset_that_keeps_both_ends_overshoots_no_further(self) -> None:
        rng = np.random.default_rng(5)
        dense = [list(_HERE)]
        for _ in range(40):
            dense.append([q + float(d) for q, d in zip(dense[-1], rng.normal(0.0, 0.05, 6))])
        whole = span_overshoot_rad(dense)
        kept = span_overshoot_rad([dense[0], *dense[5:30:4], dense[-1]])
        for axis in range(6):
            self.assertLessEqual(kept[axis], whole[axis] + 1e-15)

    def test_what_cannot_be_read_is_refused_and_nothing_is_nothing(self) -> None:
        self.assertEqual((), span_overshoot_rad([]))
        for path in ([_HERE, _THERE[:5]], [_HERE, [float("nan")] * 6, _THERE]):
            with self.subTest(path=path), self.assertRaises(ValueError):
                span_overshoot_rad(path)


class TheBranchTheArmHoldsTests(unittest.TestCase):
    """Wrist 2 at -0.3 rad, and a goal whose quickest configuration puts it at +0.3: the other side of its branch point.

    On the arm's own branch the same pose is reached with wrist 1 and wrist 3 turned about half a turn instead. The
    time alone ranks the other branch first; the branch the arm holds goes first, and a change is a warning.
    """

    _FROM = [0.3, -1.4, 1.4, -1.6, -0.3, 0.2]
    _TO = [0.3, -1.4, 1.4, -1.6, 0.3, 0.2]

    def _on_the_branch(self, arm: URRobotArm) -> "tuple[float, ...]":
        held = ur_branch("ur5e", self._FROM)
        assert held is not None
        same = [g.joints for g in _ranked(arm, self._TO, self._FROM) if held.agrees_with(ur_branch("ur5e", g.joints))]
        self.assertEqual(1, len(same))
        return same[0]

    def test_the_quickest_goal_is_on_another_branch(self) -> None:
        """The control: without the branch rule the arm would cross wrist 2's branch point."""
        arm = _arm(RoutePlanner(here=self._FROM))
        quickest = _ranked(arm, self._TO, self._FROM)[0].joints
        np.testing.assert_allclose(quickest, self._TO, atol=1e-6)
        self.assertFalse(ur_branch("ur5e", self._FROM).agrees_with(ur_branch("ur5e", quickest)))  # type: ignore[union-attr]

    def test_the_branch_the_arm_holds_goes_first(self) -> None:
        planner = RoutePlanner(here=self._FROM)
        arm = _arm(planner)
        with self.assertNoLogs("URRobotArm", level="WARNING"):
            result = _move(arm, self._TO)
        self.assertTrue(result.ok, result.message)
        np.testing.assert_allclose(_executed_goal(planner), self._on_the_branch(arm), atol=1e-9)

    def test_another_branch_only_after_every_goal_on_it_failed_and_the_change_is_a_warning(self) -> None:
        planner = RoutePlanner(here=self._FROM)
        arm = _arm(planner)
        same = self._on_the_branch(arm)
        planner.line_clear = lambda start, end: not _near(end, same, 1e-9)
        planner.answer = lambda goal: None
        with self.assertLogs("URRobotArm", level="WARNING") as logs:
            result = _move(arm, self._TO)
        self.assertTrue(result.ok, result.message)
        (asked,) = planner.named("plan_joint")
        np.testing.assert_allclose(asked[1], same, atol=1e-9, err_msg="the arm's branch was not planned to first")
        np.testing.assert_allclose(_executed_goal(planner), self._TO, atol=1e-6)
        said = "\n".join(logs.output)
        self.assertIn("changes the arm's branch", said)
        self.assertIn("shoulder +, wrist 2 -, elbow +", said)
        self.assertIn("shoulder +, wrist 2 +, elbow +", said)

    def test_out_of_a_candle_home_every_goal_is_on_the_branch_and_the_quickest_runs(self) -> None:
        self.assertEqual((0, 0, 0), tuple(vars_of(ur_branch("ur10", _CANDLE))))
        planner = RoutePlanner(here=_CANDLE)
        arm = _arm(planner, model="ur10")
        with self.assertNoLogs("URRobotArm", level="WARNING"):
            result = _move(arm, _THERE)
        self.assertTrue(result.ok, result.message)
        np.testing.assert_allclose(_executed_goal(planner), _ranked(arm, _THERE, _CANDLE)[0].joints, atol=1e-9)


def vars_of(branch: Any) -> "tuple[int, int, int]":
    return (branch.shoulder, branch.wrist, branch.elbow)


class ThePlannerGlueTests(unittest.TestCase):
    """``CuroboUrPlanner.plan_joint``: the only plan the glue asks for."""

    class _Client:
        def __init__(self, joint_names: "list[str]", answer: "list[list[float]] | None") -> None:
            self.joint_names = list(joint_names)
            self.dt = 0.025
            self.answer = answer
            self.asked: "tuple[list[float], list[float]] | None" = None
            self.last_refusal = None

        def start(self) -> None:
            return None

        def plan_joint(self, start: "list[float]", goal: "list[float]") -> "list[list[float]] | None":
            self.asked = (list(start), list(goal))
            return self.answer

    def _glue(self, client: Any, here: "list[float]") -> CuroboUrPlanner:
        conn = MagicMock()
        conn.get_joint_positions.return_value = list(here)
        return CuroboUrPlanner(conn, client_factory=lambda: client)

    def test_the_start_and_the_goal_go_in_planner_order_and_the_plan_comes_back_in_ur_order(self) -> None:
        permuted = list(reversed(UR_ARM_JOINT_NAMES))
        client = self._Client(permuted, [[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
        glue = self._glue(client, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        trajectory = glue.plan_joint([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        assert client.asked is not None
        self.assertEqual([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], client.asked[0])
        self.assertEqual([15.0, 14.0, 13.0, 12.0, 11.0, 10.0], client.asked[1])
        self.assertEqual([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], trajectory)

    def test_the_world_is_refreshed_unless_the_caller_already_did(self) -> None:
        client = self._Client(list(UR_ARM_JOINT_NAMES), [[0.0] * 6])
        glue = self._glue(client, [0.0] * 6)
        refreshed: list[Any] = []
        glue._refresh_world = lambda **kwargs: refreshed.append(kwargs)  # type: ignore[method-assign]
        glue.plan_joint([0.1] * 6, near_point_mm=(1.0, 2.0, 3.0))
        self.assertEqual([[1.0, 2.0, 3.0]], [call["near_point_mm"] for call in refreshed])
        glue.plan_joint([0.1] * 6, refresh=False)
        self.assertEqual(1, len(refreshed))

    def test_no_plan_is_none_and_the_reason_is_logged(self) -> None:
        client = self._Client(list(UR_ARM_JOINT_NAMES), None)
        client.last_refusal = StateRefusal(where=StateWhere.GOAL, kind=StateRefusalKind.WORLD, joints=(0.0,) * 6)
        glue = self._glue(client, [0.0] * 6)
        with self.assertLogs("CuroboUrPlanner", level="WARNING") as logs:
            self.assertIsNone(glue.plan_joint([0.1] * 6))
        self.assertIn("joint goal", "\n".join(logs.output))


def test_the_planners_window_is_the_one_the_descriptor_builder_declares() -> None:
    """Restated here because the builder runs under the cuRobo interpreter; the two may not drift."""
    path = _ROOT / "scripts" / "curobo" / "_planner_limits.py"
    spec = importlib.util.spec_from_file_location("_planner_limits_for_the_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lower, upper = module.planner_envelope_rad()
    np.testing.assert_allclose(PLANNER_JOINT_ENVELOPE_RAD[0], lower, atol=1e-12)
    np.testing.assert_allclose(PLANNER_JOINT_ENVELOPE_RAD[1], upper, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
