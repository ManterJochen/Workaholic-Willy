"""A cuRobo plan runs as the fewest of its own waypoints, and the list that runs is the list both authorities judged.

cuRobo samples a plan every 25 ms, 21 to 281 waypoints a move on a UR10, and the UR driver sent each one as its own blocking
``moveJ``, which starts from rest and stops at its target. The arm stopped at every one of them. Now the plan is
shortened to a subset of its own waypoints whose joint-space legs stay within the path gate's step of it, the local
path gate and the planner judge the legs of that shortened list, and exactly that list runs. Where either refuses,
the plan as cuRobo returned it is judged by both and runs instead; where that is refused too, nothing moves.

A plan is asked for only where the straight joint line to the goal is refused (tests/robot/drivers/planned/test_the_nearest_goal_is_planned.py
holds that choice). Here the local gate refuses every first list it is shown, which is that line, so each move below
reaches the plan, and the lists the tests read are the plan's.

The rule this file holds is the project's: the move that was judged is the move that runs. So the tests compare
objects, not lengths: the list handed to ``execute`` is the list ``gate_planned_path`` judged, and the samples the
planner judged are the samples of that list.

The arms are ``PlannedArm`` over the fake transport since P0g slice 13 (``tests/_arm_kit.planned_arm``), and the planner
glue runs over the fake too: a trajectory reaches the controller one ``run_trajectory`` of one waypoint at a time, which
the fake records, and how the UR transport turns each into a ``moveJ`` is held by
``tests/robot/drivers/ur/test_every_transport_keeps_the_transport_contract.py``.
"""

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config.arms import load_arm
from src.config.schema.robot import RobotConfig
from src.robot.core import JointPositions, MotionCommand, MotionResult, MotionStatus
from src.robot.drivers.planned.arm import PlannedArm
from src.robot.drivers.planned.curobo_motion import CuroboPlanner
from src.robot.safety._fcl_self_collision import mesh_backend_status
from src.robot.safety.path_samples import simplify_joint_path, waypoint_path_samples
from src.robot.safety.planning import JointCheckVerdict, StateRefusal, StateRefusalKind, StateWhere
from tests._arm_kit import LOGGER_NAME, kit_tree, planned_arm, with_the_whole_envelope
from tests._fake_transport import Raises, fake_transport
from tests._plan_end import OPEN_WORKSPACE, pose_where_it_ends

#: The controller's joint order as the ur5e description states it; the planner is handed it since P0e E4.
_UR_JOINT_NAMES = tuple(joint.name for joint in load_arm("ur5e").joints)

_DECLINED = "unit double: this file judges a planner double's path, and no camera world is wired to it"

#: A UR5e tool-down configuration in free space, and one a short way off it.
_HERE = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
_THERE = [0.6, -1.3, 1.3, -1.6, -1.5708, 0.4]
#: Per-joint radii and the step of the ur5e preflight with a 2F-85, measured 2026-09-23; the unit tests use their shape.
_RADII = (1473.0, 1310.5, 885.5, 493.3, 360.0, 160.7)


def _line(start: "list[float]", end: "list[float]", count: int) -> list[list[float]]:
    """``count`` waypoints evenly along the straight joint line, both ends exact: cuRobo's plan of a free move."""
    out = [list(start)]
    for index in range(1, count - 1):
        fraction = index / (count - 1)
        out.append([a + (b - a) * fraction for a, b in zip(start, end)])
    out.append(list(end))
    return out


def _brute_off_mm(point: "tuple[float, ...]", a: "tuple[float, ...]", b: "tuple[float, ...]") -> float:
    """The distance to the chord searched on a fine grid: an upper bound on the exact minimum."""
    best = math.inf
    for t in np.linspace(0.0, 1.0, 4001):
        best = min(best, sum(abs(p - (x + t * (y - x))) * r for p, x, y, r in zip(point, a, b, _RADII)))
    return best


# ------------------------------------------------------------------------------------------------------------------
# The pure function
# ------------------------------------------------------------------------------------------------------------------


class SimplifyJointPathTests(unittest.TestCase):
    def test_a_straight_plan_of_41_waypoints_is_two(self) -> None:
        dense = _line(_HERE, _THERE, 41)
        kept = simplify_joint_path(dense, reach_mm=_RADII, tolerance_mm=10.0)
        self.assertEqual((tuple(dense[0]), tuple(dense[-1])), kept)

    def test_the_ends_are_kept_bit_for_bit_and_nothing_is_invented(self) -> None:
        rng = np.random.default_rng(3)
        dense = [list(_HERE)]
        for _ in range(60):
            dense.append([q + float(d) for q, d in zip(dense[-1], rng.normal(0.0, 0.03, 6))])
        kept = simplify_joint_path(dense, reach_mm=_RADII, tolerance_mm=10.0)
        self.assertEqual(tuple(dense[0]), kept[0])
        self.assertEqual(tuple(dense[-1]), kept[-1])
        as_tuples = [tuple(w) for w in dense]
        positions = [as_tuples.index(w) for w in kept]
        self.assertEqual(sorted(positions), positions, "the kept waypoints are out of order")
        self.assertEqual(len(set(positions)), len(positions))

    def test_every_dropped_waypoint_is_within_the_tolerance_of_the_chord_that_replaced_it(self) -> None:
        rng = np.random.default_rng(11)
        dense = [list(_HERE)]
        for _ in range(80):
            dense.append([q + float(d) for q, d in zip(dense[-1], rng.normal(0.0, 0.02, 6))])
        for tolerance in (2.0, 10.0, 40.0):
            with self.subTest(tolerance_mm=tolerance):
                kept = simplify_joint_path(dense, reach_mm=_RADII, tolerance_mm=tolerance)
                as_tuples = [tuple(w) for w in dense]
                at = [as_tuples.index(w) for w in kept]
                for first, last in zip(at, at[1:]):
                    for index in range(first + 1, last):
                        self.assertLessEqual(
                            _brute_off_mm(as_tuples[index], as_tuples[first], as_tuples[last]), tolerance + 1e-6,
                        )

    def test_a_wrist_detour_is_weighed_by_the_wrist_and_a_shoulder_detour_by_the_shoulder(self) -> None:
        """The same 0.05 rad out and back: 8 mm on wrist 3 (dropped at 10 mm) and 74 mm on the shoulder (kept)."""
        mid = list(_HERE)
        wrist = [list(_HERE), [*mid[:5], mid[5] + 0.05], list(_HERE)]
        wrist[-1][0] += 0.3  # the chord is not a point
        shoulder = [list(_HERE), [mid[0] + 0.05, *mid[1:]], list(_HERE)]
        shoulder[-1][5] += 0.3
        self.assertEqual(2, len(simplify_joint_path(wrist, reach_mm=_RADII, tolerance_mm=10.0)))
        self.assertEqual(3, len(simplify_joint_path(shoulder, reach_mm=_RADII, tolerance_mm=10.0)))

    def test_a_curved_plan_keeps_its_corner(self) -> None:
        corner = [_HERE[0] + 0.5, *_HERE[1:]]
        end = [corner[0], corner[1] + 0.5, *corner[2:]]
        dense = _line(_HERE, corner, 21) + _line(corner, end, 21)[1:]
        kept = simplify_joint_path(dense, reach_mm=_RADII, tolerance_mm=10.0)
        self.assertEqual((tuple(_HERE), tuple(corner), tuple(end)), kept)

    def test_a_tolerance_of_zero_drops_only_what_lies_on_a_chord(self) -> None:
        bent = [list(_HERE), [_HERE[0] + 0.1, *_HERE[1:]], [_HERE[0] + 0.1, _HERE[1] + 0.1, *_HERE[2:]]]
        self.assertEqual(3, len(simplify_joint_path(bent, reach_mm=_RADII, tolerance_mm=0.0)))
        straight = _line(_HERE, _THERE, 5)
        self.assertLessEqual(len(simplify_joint_path(straight, reach_mm=_RADII, tolerance_mm=0.0)), 5)

    def test_short_paths_come_back_whole(self) -> None:
        self.assertEqual((), simplify_joint_path([], reach_mm=_RADII, tolerance_mm=10.0))
        self.assertEqual((tuple(_HERE),), simplify_joint_path([_HERE], reach_mm=_RADII, tolerance_mm=10.0))
        self.assertEqual(2, len(simplify_joint_path([_HERE, _THERE], reach_mm=_RADII, tolerance_mm=10.0)))

    def test_one_reach_for_every_joint_is_allowed(self) -> None:
        self.assertEqual(2, len(simplify_joint_path(_line(_HERE, _THERE, 9), reach_mm=1000.0, tolerance_mm=10.0)))

    def test_what_cannot_be_read_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            simplify_joint_path(_line(_HERE, _THERE, 3), reach_mm=_RADII, tolerance_mm=-1.0)
        with self.assertRaises(ValueError):
            simplify_joint_path(_line(_HERE, _THERE, 3), reach_mm=_RADII, tolerance_mm=float("nan"))
        with self.assertRaises(ValueError):
            simplify_joint_path([_HERE, [float("nan")] * 6, _THERE], reach_mm=_RADII, tolerance_mm=10.0)
        with self.assertRaises(ValueError):
            simplify_joint_path([_HERE, _THERE[:5], _THERE], reach_mm=_RADII, tolerance_mm=10.0)
        with self.assertRaises(ValueError):
            simplify_joint_path(_line(_HERE, _THERE, 3), reach_mm=_RADII[:5], tolerance_mm=10.0)


# ------------------------------------------------------------------------------------------------------------------
# The arm: judged is executed
# ------------------------------------------------------------------------------------------------------------------


class _Planner:
    """Hands back one trajectory to every joint goal and records, in order, everything the arm asked of the plan.

    The screen of a goal, one configuration, is answered clear and kept apart in ``screens``: this file reads the plan.
    """

    def __init__(self, trajectory: "list[list[float]]", *, verdicts: "list[JointCheckVerdict] | None" = None,
                 refuse_lines: bool = False) -> None:
        self.trajectory = trajectory
        self.calls: list[tuple[Any, ...]] = []
        self.screens: list[Any] = []
        self.lines: list[Any] = []
        self._verdicts = list(verdicts or [])
        self._refuse_lines = refuse_lines

    def plan_joint(self, goal: Any, *, refresh: bool = True, **_: Any) -> "list[list[float]]":
        self.calls.append(("plan",))
        return self.trajectory

    def check_joint_path(self, samples: Any, *, refresh: bool = True, clearance_mm: float = 0.0) -> JointCheckVerdict:
        configs = [tuple(float(v) for v in sample) for sample in samples]
        if len(configs) == 1:
            self.screens.append(configs[0])
            return JointCheckVerdict(valid=True, first_invalid=None, checked=1, reason="the double accepts")
        if clearance_mm > 0.0 and self._refuse_lines:
            self.lines.append(configs)
            return JointCheckVerdict(valid=False, first_invalid=1, checked=len(configs),
                                     reason="the double's straight line comes too close to the tote")
        self.calls.append(("check", configs, refresh, clearance_mm))
        if self._verdicts:
            return self._verdicts.pop(0)
        return JointCheckVerdict(valid=True, first_invalid=None, checked=len(configs), reason="the double accepts")

    def execute(self, traj: Any, pose: Any = None, *, vel: Any = None, acc: Any = None, **_: Any) -> MotionResult:
        self.calls.append(("execute", traj, vel, acc))
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose, message="curobo")

    def named(self, name: str) -> list[tuple[Any, ...]]:
        return [call for call in self.calls if call[0] == name]


def _refused(reason: str = "the double refuses") -> JointCheckVerdict:
    return JointCheckVerdict(valid=False, first_invalid=0, checked=1, reason=reason)


def _tree() -> RobotConfig:
    """A ur5e cell on cuRobo, its workspace box open so the box never decides a test about the path."""
    return kit_tree("curobo", safety={}, motion={"workspace": OPEN_WORKSPACE})


def _on_curobo(planner: _Planner) -> Any:
    """A planned arm standing at ``_HERE`` over the fake transport, ``planner`` as its cuRobo."""
    arm: Any = planned_arm(_tree(), transport=fake_transport(joints=_HERE))
    arm._curobo = planner
    return arm


def _arm(planner: _Planner) -> tuple[PlannedArm, list[Any]]:
    """A ur5e on cuRobo with a planner double; the local path gate is a spy that records what it judged.

    The first list the spy is shown is the straight line to the goal, which it refuses without recording, so that the
    move plans; everything it records after that is the plan's.
    """
    arm = _on_curobo(planner)
    # The endpoint gate has its own tests (tests/robot/safety/test_ungated_motion_surfaces.py); this file reads the legs.
    arm._gate_planned_config = lambda pose, joints: None  # type: ignore[method-assign]
    judged: list[Any] = []
    verdicts: list["MotionResult | None"] = []
    lined: list[Any] = []

    def gate(waypoints: Any, *, arm: Any = None, command: Any = None) -> "MotionResult | None":
        if not lined:
            lined.append(waypoints)
            return MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO,
                                       message="the spy: the straight line grazes the bench")
        judged.append(waypoints)
        return verdicts.pop(0) if verdicts else None

    # The spy keeps the verdicts it hands out on itself: a list written onto the arm would be a name the arm never
    # reads (tests/_injection_guard.py).
    gate.verdicts = verdicts  # type: ignore[attr-defined]
    arm._preflight.gate_planned_path = gate  # type: ignore[method-assign]
    return arm, judged


def _local_refusal() -> MotionResult:
    return MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO, message="the spy refuses")


class JudgedIsExecutedTests(unittest.TestCase):
    def _move(self, arm: PlannedArm, end: "list[float]", **kwargs: Any) -> MotionResult:
        self.enterContext(arm.without_camera_world(_DECLINED))
        return arm.move(pose_where_it_ends(arm, end), **kwargs)

    def test_a_straight_plan_runs_as_one_moveJ_and_that_list_is_the_one_judged(self) -> None:
        planner = _Planner(_line(_HERE, _THERE, 41))
        arm, judged = _arm(planner)
        result = self._move(arm, _THERE)
        self.assertTrue(result.ok, result.message)
        (execute,) = planner.named("execute")
        executed = execute[1]
        # Every cuRobo arm gates its own sends: the gate found the arm at the list's first waypoint, so one moveJ runs,
        # to its end (``PlannedArm._sent_waypoints``).
        self.assertEqual([tuple(_THERE)], [tuple(w) for w in executed])
        self.assertEqual(1, len(judged), "the shortened list was judged more than once, or not at all")
        self.assertEqual([tuple(_HERE), tuple(_THERE)], [tuple(w) for w in judged[0]])
        self.assertEqual([tuple(w) for w in judged[0][1:]], [tuple(w) for w in executed],
                         "the list the gate judged is not the list that ran")

    def test_the_planner_judges_the_legs_that_run_in_the_world_it_already_holds(self) -> None:
        planner = _Planner(_line(_HERE, _THERE, 41))
        arm, _ = _arm(planner)
        self._move(arm, _THERE)
        (check,) = planner.named("check")
        (execute,) = planner.named("execute")
        step = arm._preflight.path_step_mm
        reach = arm._preflight.joint_radii_mm(arm)
        assert step is not None and reach is not None
        # The legs that run start where the gate found the arm, the judged list's first waypoint.
        expected = waypoint_path_samples([list(_HERE), *execute[1]], reach_mm=reach, max_step_mm=step).configs
        self.assertEqual(list(expected), check[1])
        self.assertIs(False, check[2], "the planner refreshed its world after the gate had judged against it")
        self.assertEqual(["plan", "check", "execute"], [call[0] for call in planner.calls])
        self.assertEqual(3.0, check[3], "a shortened list's chords are lines nobody planned, judged at the line "
                                         "clearance (found porting 85f082b to dev, 2026-09-25)")

    def test_the_plan_as_cuRobo_returned_it_is_judged_at_no_clearance(self) -> None:
        """Where the shortened list is refused, the plan as returned is judged as cuRobo validated it: at no contact."""
        dense = _line(_HERE, _THERE, 41)
        planner = _Planner(dense, verdicts=[_refused("a chord grazes the tote")])
        arm, _ = _arm(planner)
        self.assertTrue(self._move(arm, _THERE).ok)
        self.assertEqual([3.0, 0.0], [check[3] for check in planner.named("check")])

    def test_a_plan_with_nothing_to_drop_is_judged_as_cuRobo_returned_it(self) -> None:
        planner = _Planner([list(_HERE), list(_THERE)])
        arm, _ = _arm(planner)
        self._move(arm, _THERE)
        self.assertEqual([0.0], [check[3] for check in planner.named("check")])

    def test_a_shortened_list_the_planner_refuses_falls_back_to_the_plan_as_returned(self) -> None:
        dense = _line(_HERE, _THERE, 41)
        planner = _Planner(dense, verdicts=[_refused("a chord grazes the tote")])
        arm, judged = _arm(planner)
        result = self._move(arm, _THERE)
        self.assertTrue(result.ok, result.message)
        (execute,) = planner.named("execute")
        self.assertEqual(dense[1:], execute[1], "the dense plan was not the one that ran")
        self.assertEqual(2, len(judged))
        self.assertIs(dense, judged[-1])
        checks = planner.named("check")
        self.assertEqual(2, len(checks), "the dense plan ran without the planner judging its legs")
        step, reach = arm._preflight.path_step_mm, arm._preflight.joint_radii_mm(arm)
        assert step is not None and reach is not None
        self.assertEqual(list(waypoint_path_samples(dense, reach_mm=reach, max_step_mm=step).configs), checks[-1][1])

    def test_a_shortened_list_the_local_gate_refuses_falls_back_to_the_plan_as_returned(self) -> None:
        dense = _line(_HERE, _THERE, 41)
        planner = _Planner(dense)
        arm, judged = _arm(planner)
        arm._preflight.gate_planned_path.verdicts.append(_local_refusal())  # type: ignore[attr-defined]
        result = self._move(arm, _THERE)
        self.assertTrue(result.ok, result.message)
        (execute,) = planner.named("execute")
        self.assertEqual(dense[1:], execute[1])
        self.assertEqual(2, len(judged[0]))
        self.assertIs(dense, judged[1])
        self.assertEqual(1, len(planner.named("check")), "a list the local gate refused was sent to the planner")

    def test_when_the_plan_as_returned_is_refused_too_nothing_moves(self) -> None:
        planner = _Planner(_line(_HERE, _THERE, 41), verdicts=[_refused(), _refused("the plan's own leg")])
        arm, _ = _arm(planner)
        result = self._move(arm, _THERE)
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status)
        self.assertIn("the plan's own leg", result.message or "")
        self.assertEqual([], planner.named("execute"))

    def test_a_joint_limit_on_the_legs_reads_as_one(self) -> None:
        refusal = StateRefusal(where=StateWhere.GOAL, kind=StateRefusalKind.JOINT_LIMIT, joints=(0.0,) * 6)
        verdict = JointCheckVerdict(valid=False, first_invalid=3, checked=9, reason="sample 3", refusal=refusal)
        planner = _Planner([list(_HERE), list(_THERE)], verdicts=[verdict])
        arm, _ = _arm(planner)
        result = self._move(arm, _THERE)
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status)
        self.assertEqual([], planner.named("execute"))

    def test_a_plan_with_nothing_to_drop_is_judged_once(self) -> None:
        planner = _Planner([list(_HERE), list(_THERE)])
        arm, judged = _arm(planner)
        self._move(arm, _THERE)
        self.assertEqual(1, len(judged))
        self.assertEqual(1, len(planner.named("check")))
        self.assertEqual(planner.trajectory[1:], planner.named("execute")[0][1])

    def test_a_bent_plan_keeps_its_corner(self) -> None:
        corner = [_HERE[0] + 0.5, *_HERE[1:]]
        end = [corner[0], corner[1] + 0.3, *corner[2:]]
        planner = _Planner(_line(_HERE, corner, 21) + _line(corner, end, 21)[1:])
        arm, _ = _arm(planner)
        self._move(arm, end)
        executed = [tuple(w) for w in planner.named("execute")[0][1]]
        self.assertEqual([tuple(corner), tuple(end)], executed)

    def test_no_step_means_no_planner_question_as_on_a_joint_move(self) -> None:
        """Where no step derives the local gate has refused already; the planner is not a second voice."""
        planner = _Planner(_line(_HERE, _THERE, 41))
        arm, judged = _arm(planner)
        with mock.patch.object(type(arm._preflight), "path_step_mm", new_callable=mock.PropertyMock,
                               return_value=None):
            self._move(arm, _THERE)
        self.assertEqual([], planner.named("check"))
        self.assertEqual(1, len(judged))
        self.assertIs(planner.trajectory, judged[0], "a list was shortened without the step it is shortened by")

    def test_the_speed_the_caller_asks_is_clamped_to_motion_limits(self) -> None:
        planner = _Planner(_line(_HERE, _THERE, 5))
        arm, _ = _arm(planner)
        self._move(arm, _THERE, vel=5.0, acc=40.0)
        _, _, vel, acc = planner.named("execute")[0]
        self.assertEqual((1.0, 0.5), (vel, acc))

    def test_a_speed_inside_the_limits_is_passed_as_asked(self) -> None:
        planner = _Planner(_line(_HERE, _THERE, 5))
        arm, _ = _arm(planner)
        self._move(arm, _THERE, vel=0.3, acc=0.2)
        self.assertEqual((0.3, 0.2), planner.named("execute")[0][2:])

    def test_every_plan_says_how_many_waypoints_ran_and_how_far_the_joints_turn(self) -> None:
        planner = _Planner(_line(_HERE, _THERE, 41))
        arm, _ = _arm(planner)
        with self.assertLogs(LOGGER_NAME, level="INFO") as logs:
            self._move(arm, _THERE)
        said = "\n".join(logs.output)
        self.assertIn("cuRobo plan to goal 1 of", said)
        self.assertIn("41 waypoint(s) dense, 2 executed (1 leg(s))", said)
        self.assertIn("total/largest leg", said)
        self.assertIn(f"shoulder_pan_joint {math.degrees(0.6):.1f}/{math.degrees(0.6):.1f}", said)
        self.assertNotIn("long way round", said)

    def test_a_joint_turning_the_long_way_round_is_a_warning(self) -> None:
        """A joint target a full turn and a bit away inside a window of two turns: the one move still left that can. The
        arm's whole envelope, as the Isaac arm runs it: a real arm keeps half a turn about home, and turns such a target
        onto its twin inside."""
        end = [*_HERE[:5], _HERE[5] + 2 * math.pi - 0.3]
        planner = _Planner(_line(_HERE, end, 41))
        arm, _ = _arm(planner)
        with_the_whole_envelope(arm._preflight)
        with arm.without_camera_world(_DECLINED), self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
            result = arm.move_to_joints(JointPositions(end))
        self.assertTrue(result.ok, result.message)
        said = "\n".join(logs.output)
        self.assertIn("wrist_3_joint", said)
        self.assertIn("long way round", said)


@pytest.mark.skipif(mesh_backend_status("ur5e") != "ok", reason="no exact mesh backend on this box")
def test_with_the_real_local_gate_a_plan_that_shortens_back_into_the_refused_line_runs_as_cuRobo_returned_it() -> None:
    """Found porting 85f082b to dev, 2026-09-25: the straight line refused at the clearance must not come back as a
    shortened plan.

    The planner double refuses every list judged at a clearance, so the move plans; cuRobo's plan is straight, and
    shortened it is the very line refused. Judged at the line clearance it is refused again, and the plan runs as
    cuRobo returned it, judged at no contact, with nothing stubbed on the local side.
    """
    dense = _line(_HERE, _THERE, 41)
    planner = _Planner(dense, refuse_lines=True)
    arm = _on_curobo(planner)
    with arm.without_camera_world(_DECLINED):
        result = arm.move(pose_where_it_ends(arm, _THERE))
    assert result.ok, result.message
    assert len(planner.lines) == 2, "the line and the shortened list were not both judged at the clearance"
    assert [len(configs) > 2 for configs in planner.lines] == [True, True]
    (execute,) = planner.named("execute")
    assert execute[1] == dense[1:], "the plan ran shortened into the line the planner refused"


@pytest.mark.skipif(mesh_backend_status("ur5e") != "ok", reason="no exact mesh backend on this box")
def test_with_the_real_local_gate_a_clear_line_runs_and_nothing_is_planned() -> None:
    """The control of the control: where both authorities pass the straight line, it runs and cuRobo plans nothing."""
    planner = _Planner(_line(_HERE, _THERE, 41))
    arm = _on_curobo(planner)
    with arm.without_camera_world(_DECLINED):
        result = arm.move(pose_where_it_ends(arm, _THERE))
    assert result.ok, result.message
    assert planner.named("plan") == []
    ((_, configs, _, clearance),) = planner.named("check")
    assert clearance == 3.0, "the line was not judged at the shipped clearance"
    (execute,) = planner.named("execute")
    assert len(execute[1]) == 1, "a clear line runs as one moveJ to its end"
    np.testing.assert_allclose(execute[1][-1], _THERE, atol=1e-6)
    np.testing.assert_allclose(configs[-1], execute[1][-1], atol=0.0)


@pytest.mark.skipif(mesh_backend_status("ur5e") != "ok", reason="no exact mesh backend on this box")
def test_a_waypoint_that_cannot_be_read_is_refused_by_the_gate_and_not_shortened_away() -> None:
    """A NaN in the middle of a plan: shortening would drop it without a word, so the plan meets the gate whole."""
    planner = _Planner([list(_HERE), [float("nan")] * 6, list(_THERE)], refuse_lines=True)
    arm = _on_curobo(planner)
    arm._gate_planned_config = lambda pose, joints: None  # type: ignore[method-assign]
    with arm.without_camera_world(_DECLINED):
        result = arm.move(pose_where_it_ends(arm, _THERE))
    assert not result.ok
    assert "not a finite number" in (result.message or ""), result.message
    assert planner.named("execute") == []
    assert planner.named("check") == [], "a plan the local gate refused was sent to the planner"


# ------------------------------------------------------------------------------------------------------------------
# Execution: what a failure says
# ------------------------------------------------------------------------------------------------------------------


def _refusing(at: int, *, error: "BaseException | None" = None) -> Any:
    """A fake that takes ``at`` one-waypoint trajectories and then refuses the next: raising ``error``, or answering
    ``False`` when there is none. A ``ValueError`` means the library refused that waypoint before sending it."""
    fake = fake_transport(joints=_HERE)
    fake.script("run_trajectory", *([True] * at), Raises(error) if error is not None else False)
    return fake


def _glue(fake: Any, *, state: "Any" = None) -> CuroboPlanner:
    """The planner glue over the fake: each waypoint reaches it as one ``run_trajectory`` of one waypoint."""
    return CuroboPlanner(fake, joint_names=_UR_JOINT_NAMES, controller_yaw_deg=180,
                         client_factory=MagicMock, vel=1.0, acc=0.5, controller_state=state)


class ExecuteSaysHowFarItGotTests(unittest.TestCase):
    _PLAN = [list(_HERE), [0.3, -1.4, 1.4, -1.6, -1.5708, 0.2], list(_THERE)]

    def test_a_range_error_from_ur_rtde_is_an_invalid_target_and_says_what_ran(self) -> None:
        fake = _refusing(1, error=ValueError("The value is not within [0;3.14]"))
        result = _glue(fake).execute(self._PLAN, MagicMock(), vel=5.0)
        self.assertIs(MotionStatus.INVALID_TARGET, result.status)
        self.assertIn("waypoint 2 of 3", result.message or "")
        self.assertIn("was not sent", result.message or "")
        self.assertIn("the 1 before it ran", result.message or "")
        self.assertEqual([self._PLAN[0]], fake.sent, "the waypoint before the refused one did not run, or it did too")

    def test_a_range_error_on_the_first_waypoint_says_nothing_moved(self) -> None:
        result = _glue(_refusing(0, error=ValueError("range"))).execute(self._PLAN, MagicMock())
        self.assertIs(MotionStatus.INVALID_TARGET, result.status)
        self.assertIn("nothing had moved", result.message or "")

    def test_a_refused_moveJ_carries_the_controller_state(self) -> None:
        fake = _refusing(2)
        result = _glue(fake, state=lambda: " The controller reports a protective stop.").execute(self._PLAN, MagicMock())
        self.assertIs(MotionStatus.CONTROLLER_REJECTED, result.status)
        self.assertIn("waypoint 3 of 3", result.message or "")
        self.assertIn("moveJ was sent", result.message or "")
        self.assertIn("protective stop", result.message or "")

    def test_a_moveJ_that_raised_carries_the_controller_state(self) -> None:
        fake = _refusing(0, error=RuntimeError("RTDE control script is not running"))
        result = _glue(fake, state=lambda: " The controller reports robot mode idle.").execute(self._PLAN, MagicMock())
        self.assertIs(MotionStatus.CONNECTION_ERROR, result.status)
        self.assertIn("robot mode idle", result.message or "")

    def test_a_state_that_cannot_be_read_is_not_a_second_fault(self) -> None:
        def broken() -> str:
            raise RuntimeError("the dashboard is gone too")

        result = _glue(_refusing(0), state=broken).execute(self._PLAN, MagicMock())
        self.assertIs(MotionStatus.CONTROLLER_REJECTED, result.status)
        self.assertNotIn("dashboard", result.message or "")

    def test_each_failure_says_in_its_field_whether_anything_was_sent(self) -> None:
        """P0g R2: a refusal at the first waypoint sent nothing; one at a later waypoint, a moveJ that raised and one
        that answered False all sent something, whatever status they carry."""
        from src.robot.core import MotionSent

        cases = (
            (0, ValueError("range"), MotionSent.NOTHING),
            (1, ValueError("range"), MotionSent.SENT),
            (0, RuntimeError("RTDE control script is not running"), MotionSent.SENT),
            (2, None, MotionSent.SENT),
        )
        for at, error, expected in cases:
            with self.subTest(expected=expected, refuse_at=at):
                self.assertIs(expected, _glue(_refusing(at, error=error)).execute(self._PLAN, MagicMock()).sent)

    def test_every_waypoint_of_the_list_is_one_moveJ_in_order_at_the_speed_given(self) -> None:
        fake = fake_transport(joints=_HERE)
        result = _glue(fake).execute(self._PLAN, MagicMock(), vel=0.4, acc=0.3)
        self.assertTrue(result.ok)
        sent = [(call.args[0], call.kwargs["vel"], call.kwargs["acc"]) for call in fake.calls("run_trajectory")]
        self.assertEqual([([w], 0.4, 0.3) for w in self._PLAN], sent)

    def test_the_arm_hands_its_planner_its_own_controller_state(self) -> None:
        arm: Any = planned_arm(kit_tree("curobo", safety={}),
                               curobo_client_factory=MagicMock)
        planner = arm._curobo_planner()
        self.assertEqual(arm._controller_state_text, planner._controller_state)

    def test_through_the_arm_a_range_error_is_a_result_and_not_a_raise(self) -> None:
        planner = _Planner([list(_HERE), list(_THERE)])
        arm, _ = _arm(planner)
        fake: Any = arm._transport
        fake.script("run_trajectory", Raises(ValueError("The value is not within [0;40]")))
        planner.execute = _glue(fake).execute  # type: ignore[method-assign]
        with arm.without_camera_world(_DECLINED):
            result = arm.move(pose_where_it_ends(arm, _THERE))
        self.assertIs(MotionStatus.INVALID_TARGET, result.status)
        self.assertEqual(1, len(fake.calls("run_trajectory")), "the refusal was not the controller library's")


def test_the_joint_names_the_log_uses_are_the_ur_order() -> None:
    """The log names joints by the arm's own order, which a driver reads from its description."""
    arm, _ = _arm(_Planner([list(_HERE), list(_THERE)]))
    assert arm._joint_names == _UR_JOINT_NAMES
    assert len(_UR_JOINT_NAMES) == 6 and _UR_JOINT_NAMES[5] == "wrist_3_joint"


if __name__ == "__main__":
    unittest.main()
