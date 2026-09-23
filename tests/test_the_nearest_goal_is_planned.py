"""A Cartesian cuRobo move goes to the configuration nearest the arm, chosen before planning and never after.

cuRobo, handed a pose, picks the goal configuration from random seeds over the whole joint range. Measured on a
UR10 on this project's GPU: a wrist sweep took 8 of 22 legs the long way, one turning the base 8.57 rad where 2.37
would do, and the same sweep planned twice took different ways. The UR driver now solves the flange goal in closed
form, turns every solution onto the full turns nearest the arm inside the planner's and the joint-limit guard's
window, screens the quickest few with the planner and the endpoint gate, and plans to them in joint space. A plan is
taken only where it ends on the configuration asked for; where none is taken, cuRobo chooses as it always did.

What this file holds with a planner double: which configuration is asked for (the nearest turn, the nearest branch,
nothing outside the window), what is screened and how often cuRobo is asked, the fall back and the end tolerance,
and that the trajectory which runs is the planner's own. The GPU half, with the real sidecar, is in the report of
the change that brought this in, not here.
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
from src.robot.core import MotionCommand, MotionResult, MotionStatus
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.drivers.ur.curobo_motion import PLANNER_JOINT_ENVELOPE_RAD, UR_ARM_JOINT_NAMES, CuroboUrPlanner
from src.robot.safety._ur_ik import nearest_goals, ur_flange_ik
from src.robot.safety._ur_kinematics import ur_link_transforms_mm
from src.robot.safety.planning import JointCheckVerdict, StateRefusal, StateRefusalKind, StateWhere
from tests._plan_end import OPEN_WORKSPACE, pose_where_it_ends

_DECLINED = "unit double: this file reads which goal a planner double is asked for, and no camera world is wired"
_ROOT = pathlib.Path(__file__).resolve().parents[1]

_HERE = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
_THERE = [0.3, -1.4, 1.4, -1.6, -1.5, 0.2]


def _line(start: "list[float]", end: "list[float]", count: int) -> list[list[float]]:
    out = [list(start)]
    for index in range(1, count - 1):
        out.append([a + (b - a) * index / (count - 1) for a, b in zip(start, end)])
    out.append(list(end))
    return out


class _Planner:
    """A planner that plans joint goals: a straight line to whatever it is asked for, unless told otherwise."""

    def __init__(self, *, here: "list[float]", cartesian: "list[list[float]] | None" = None) -> None:
        self.here = list(here)
        self.cartesian = cartesian if cartesian is not None else []
        self.calls: list[tuple[Any, ...]] = []
        self.last_refusal: "StateRefusal | None" = None
        #: goal -> trajectory; None plans nothing. The default is a straight line of 21 waypoints.
        self.answer = lambda goal: _line(self.here, goal, 21)
        self.screen = lambda config: True
        self.refusal_on_no_plan: "StateRefusal | None" = None

    def plans_joint_goals(self) -> bool:
        return True

    def refresh_world(self, *, near_point_mm: Any = None, goal_keep_out: Any = None) -> None:
        self.calls.append(("refresh", near_point_mm))

    def plan(self, goal: Any, **_: Any) -> "list[list[float]]":
        self.calls.append(("plan",))
        return self.cartesian

    def plan_joint(self, goal: Any, *, refresh: bool = True, **_: Any) -> "list[list[float]] | None":
        self.calls.append(("plan_joint", [float(v) for v in goal], refresh))
        trajectory = self.answer([float(v) for v in goal])
        self.last_refusal = None if trajectory else self.refusal_on_no_plan
        return trajectory

    def check_joint_path(self, samples: Any, *, refresh: bool = True) -> JointCheckVerdict:
        configs = [tuple(float(v) for v in sample) for sample in samples]
        self.calls.append(("check", configs, refresh))
        if len(configs) == 1 and not self.screen(configs[0]):
            return JointCheckVerdict(valid=False, first_invalid=0, checked=1, reason="the double refuses this goal")
        return JointCheckVerdict(valid=True, first_invalid=None, checked=len(configs), reason="the double accepts")

    def execute(self, traj: Any, pose: Any, *, vel: Any = None, acc: Any = None) -> MotionResult:
        self.calls.append(("execute", traj))
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose, message="curobo")

    def named(self, name: str) -> list[tuple[Any, ...]]:
        return [call for call in self.calls if call[0] == name]


def _arm(planner: _Planner, *, model: str = "ur5e", safety: "dict[str, Any] | None" = None) -> URRobotArm:
    arm = URRobotArm(RobotConfig.model_validate({
        "vendor": "ur", "ur": {"model": model, "motion_planner": "curobo"},
        "safety": {"payload": {"enforce": False}, **(safety or {})}, "gripper": {"model": "robotiq_2f85"},
        "workspace_limits": OPEN_WORKSPACE,
    }))
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.get_joint_positions.return_value = list(planner.here)
    arm._curobo_ur = planner  # type: ignore[assignment]
    # The legs and the end have their own tests; this file reads which goal was asked for.
    arm._preflight.gate_planned_path = lambda waypoints, *, arm=None, command=None: None  # type: ignore[method-assign]
    arm._gate_planned_config = lambda pose, joints: None  # type: ignore[method-assign]
    return arm


def _move(arm: URRobotArm, end: "list[float]") -> MotionResult:
    with arm.without_camera_world(_DECLINED):
        return arm.move(pose_where_it_ends(arm, end))


def _ranked(arm: URRobotArm, end: "list[float]", here: "list[float]") -> list[Any]:
    """The order the arm ranks the goals in, from the pure function the arm calls."""
    frames = ur_link_transforms_mm(str(arm.config.ur.model), np.asarray(end, dtype=np.float64))
    assert frames is not None
    solutions = ur_flange_ik(str(arm.config.ur.model), frames[-1], q6_if_singular=here[-1])
    assert solutions is not None
    lower, upper = arm._goal_joint_window()
    return list(nearest_goals(solutions, current=here, lower=lower, upper=upper, velocity=1.0))


class WhichGoalIsAskedForTests(unittest.TestCase):
    def test_the_configuration_nearest_the_arm_is_planned_to_and_cuRobo_chooses_nothing(self) -> None:
        planner = _Planner(here=_HERE)
        result = _move(_arm(planner), _THERE)
        self.assertTrue(result.ok, result.message)
        (asked,) = planner.named("plan_joint")
        np.testing.assert_allclose(asked[1], _THERE, atol=1e-6)
        self.assertEqual([], planner.named("plan"), "cuRobo was asked for a Cartesian plan as well")

    def test_a_joint_goes_on_its_full_turn_nearest_the_arm(self) -> None:
        """Wrist 3 stands at 5.0; the goal's wrist 3 angle is -0.983, and 5.3 is the same angle one turn on."""
        here = [*_HERE[:5], 5.0]
        end = [*_THERE[:5], 5.3]
        planner = _Planner(here=here)
        _move(_arm(planner), end)
        (asked,) = planner.named("plan_joint")
        self.assertAlmostEqual(5.3, asked[1][5], places=6)
        np.testing.assert_allclose(asked[1], end, atol=1e-6)

    def test_a_nearer_turn_outside_the_planners_window_is_not_asked_for(self) -> None:
        """Wrist 1 at 6.1 rad: the goal's 0.4 + 2 pi is nearer, and past 2 pi - 0.1, so the planner never sees it."""
        here = [_HERE[0], _HERE[1], _HERE[2], 6.1, _HERE[4], _HERE[5]]
        end = [_THERE[0], _THERE[1], _THERE[2], 0.4, _THERE[4], _THERE[5]]
        planner = _Planner(here=here)
        _move(_arm(planner), end)
        lower, upper = PLANNER_JOINT_ENVELOPE_RAD
        self.assertTrue(planner.named("plan_joint"), "no joint goal was asked for, so nothing here was tested")
        for call in planner.named("plan_joint") + [c for c in planner.named("check") if len(c[1]) == 1]:
            goal = call[1] if call[0] == "plan_joint" else call[1][0]
            for axis, value in enumerate(goal):
                self.assertTrue(lower[axis] <= value <= upper[axis], f"{UR_ARM_JOINT_NAMES[axis]} asked at {value}")

    def test_the_guards_window_narrows_the_planners(self) -> None:
        """A cell that limits wrist 3 to +-100 deg (less a 5 deg margin) is never asked to go one turn on."""
        limits = {"joint_limits": {"min_deg": [-360.0] * 5 + [-100.0], "max_deg": [360.0] * 5 + [100.0]}}
        here = [*_HERE[:5], 1.5]
        end = [*_THERE[:5], 1.2]
        planner = _Planner(here=here)
        arm = _arm(planner, safety=limits)
        self.assertAlmostEqual(math.radians(95.0), arm._goal_joint_window()[1][5])
        _move(arm, end)
        (asked,) = planner.named("plan_joint")
        self.assertAlmostEqual(1.2, asked[1][5], places=6)

    def test_a_goal_no_turn_of_which_the_guard_admits_is_left_to_cuRobo(self) -> None:
        limits = {"joint_limits": {"min_deg": [-360.0] * 5 + [-30.0], "max_deg": [360.0] * 5 + [30.0]}}
        here = [*_HERE[:5], 0.0]
        end = [*_THERE[:5], 2.0]
        planner = _Planner(here=here, cartesian=[list(here), list(end)])
        arm = _arm(planner, safety=limits)
        with self.assertLogs("URRobotArm", level="INFO") as logs:
            _move(arm, end)
        self.assertEqual([], planner.named("plan_joint"))
        self.assertEqual(1, len(planner.named("plan")))
        self.assertIn("joint window", "\n".join(logs.output))

    def test_the_elbow_is_never_asked_past_the_planners_half_turn(self) -> None:
        planner = _Planner(here=_HERE)
        arm = _arm(planner)
        for goal in _ranked(arm, _THERE, _HERE):
            self.assertLessEqual(abs(goal.joints[2]), math.pi - 0.1)


class WhatIsScreenedAndPlannedTests(unittest.TestCase):
    def test_a_goal_the_planner_refuses_is_not_planned_to(self) -> None:
        planner = _Planner(here=_HERE)
        arm = _arm(planner)
        ranked = _ranked(arm, _THERE, _HERE)
        first = ranked[0].joints
        planner.screen = lambda config: max(abs(a - b) for a, b in zip(config, first)) > 1e-9
        _move(arm, _THERE)
        (asked,) = planner.named("plan_joint")
        np.testing.assert_allclose(asked[1], ranked[1].joints, atol=1e-9)
        screens = [c for c in planner.named("check") if len(c[1]) == 1]
        self.assertTrue(all(c[2] is False for c in screens), "a screen refreshed the world the plan was judged in")

    def test_a_goal_the_endpoint_gate_refuses_is_not_planned_to(self) -> None:
        planner = _Planner(here=_HERE)
        arm = _arm(planner)
        ranked = _ranked(arm, _THERE, _HERE)
        first = ranked[0].joints
        refusal = MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO, message="the spy")
        seen: list[tuple[float, ...]] = []

        def gate(pose: Any, joints: Any) -> "MotionResult | None":
            seen.append(tuple(joints.tolist()))
            return refusal if max(abs(a - b) for a, b in zip(joints.tolist(), first)) < 1e-9 else None

        arm._gate_planned_config = gate  # type: ignore[method-assign]
        result = _move(arm, _THERE)
        self.assertTrue(result.ok, result.message)
        (asked,) = planner.named("plan_joint")
        np.testing.assert_allclose(asked[1], ranked[1].joints, atol=1e-9)
        # And the plan's own end met the same gate again afterwards, unchanged.
        np.testing.assert_allclose(seen[-1], ranked[1].joints, atol=1e-9)

    def test_no_more_than_three_goals_are_planned_to_and_then_cuRobo_chooses(self) -> None:
        planner = _Planner(here=_HERE, cartesian=[list(_HERE), list(_THERE)])
        planner.answer = lambda goal: None
        arm = _arm(planner)
        self.assertGreaterEqual(len(_ranked(arm, _THERE, _HERE)), 4, "the test needs more than three goals")
        with self.assertLogs("URRobotArm", level="INFO") as logs:
            result = _move(arm, _THERE)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(3, len(planner.named("plan_joint")))
        self.assertEqual(1, len(planner.named("plan")))
        said = "\n".join(logs.output)
        self.assertIn("chooses the goal configuration", said)
        self.assertIn("did not plan", said)
        self.assertIn("cuRobo's own choice", said)

    def test_a_plan_that_ends_off_the_goal_it_was_asked_for_is_not_taken(self) -> None:
        planner = _Planner(here=_HERE, cartesian=[list(_HERE), list(_THERE)])
        planner.answer = lambda goal: _line(_HERE, [goal[0] + 1e-3, *goal[1:]], 5)
        _move(_arm(planner), _THERE)
        self.assertEqual(3, len(planner.named("plan_joint")))
        self.assertEqual(1, len(planner.named("plan")))

    def test_a_plan_inside_the_tolerance_is_taken(self) -> None:
        planner = _Planner(here=_HERE)
        planner.answer = lambda goal: _line(_HERE, [goal[0] + 5e-5, *goal[1:]], 5)
        result = _move(_arm(planner), _THERE)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(1, len(planner.named("plan_joint")))
        self.assertEqual([], planner.named("plan"))

    def test_a_start_the_planner_refuses_stops_the_search_and_is_reported_typed(self) -> None:
        refusal = StateRefusal(where=StateWhere.START, kind=StateRefusalKind.SELF_COLLISION, joints=tuple(_HERE))
        planner = _Planner(here=_HERE, cartesian=[])
        planner.answer = lambda goal: None
        planner.refusal_on_no_plan = refusal
        arm = _arm(planner)
        original_plan = planner.plan

        def plan(goal: Any, **kwargs: Any) -> "list[list[float]]":
            planner.last_refusal = refusal
            return original_plan(goal, **kwargs)

        planner.plan = plan  # type: ignore[method-assign]
        result = _move(arm, _THERE)
        self.assertEqual(1, len(planner.named("plan_joint")))
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status)
        self.assertIn("will not start from", result.message or "")

    def test_the_world_is_refreshed_once_near_the_goal_and_everything_after_uses_it(self) -> None:
        planner = _Planner(here=_HERE)
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

    def test_what_runs_is_the_planners_own_trajectory(self) -> None:
        planner = _Planner(here=_HERE)
        returned: list[list[list[float]]] = []

        def answer(goal: "list[float]") -> "list[list[float]]":
            trajectory = _line(_HERE, goal, 21)
            returned.append(trajectory)
            return trajectory

        planner.answer = answer
        _move(_arm(planner), _THERE)
        (execute,) = planner.named("execute")
        self.assertEqual([tuple(returned[0][0]), tuple(returned[0][-1])], [tuple(w) for w in execute[1]])

    def test_a_planner_that_plans_no_joint_goals_is_asked_as_before(self) -> None:
        planner = _Planner(here=_HERE, cartesian=[list(_HERE), list(_THERE)])
        planner.plans_joint_goals = lambda: False  # type: ignore[method-assign]
        result = _move(_arm(planner), _THERE)
        self.assertTrue(result.ok, result.message)
        self.assertEqual([], planner.named("plan_joint"))
        self.assertEqual(1, len(planner.named("plan")))

    def test_the_goal_log_names_where_it_came_from(self) -> None:
        planner = _Planner(here=_HERE)
        with self.assertLogs("URRobotArm", level="INFO") as logs:
            _move(_arm(planner), _THERE)
        self.assertIn("goal from the nearest goal 1 of", "\n".join(logs.output))


class ThePlannerGlueTests(unittest.TestCase):
    """``CuroboUrPlanner.plan_joint``: the joint-goal twin of ``plan``."""

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

    def test_it_says_whether_its_client_plans_joint_goals(self) -> None:
        self.assertTrue(self._glue(self._Client(list(UR_ARM_JOINT_NAMES), None), [0.0] * 6).plans_joint_goals())

        class _Cartesian:
            joint_names = list(UR_ARM_JOINT_NAMES)

            def start(self) -> None:
                return None

        self.assertFalse(self._glue(_Cartesian(), [0.0] * 6).plans_joint_goals())


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
