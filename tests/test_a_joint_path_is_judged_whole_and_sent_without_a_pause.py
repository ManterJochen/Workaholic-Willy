"""A UR arm runs a joint path through several waypoints as one motion, judged whole before anything is sent, and the
exact mesh guard alone judges a motion where it holds everything the planner would judge it against.

The owner, 2026-10-07: the wave at a greeting "muss eine konsistente Bewegung sein"; swing by swing, each judged on its
own, the hand stood five seconds at every turn. And "einfach mit den meshes prüfen und nicht mit kugeln": the planner's
spheres refused lines the meshes kept 18 mm clear, and every refusal cost a report over thousands of sphere pairs.

What this file pins:

* ``move_through_joints_on_the_line`` (``DrivesJointPaths``): every leg from where the arm stands through every
  waypoint judged in one check, then the waypoints sent as a judged plan is sent, one list to the executor, no blend; a
  leg that is not clear refuses the whole path, nothing sent and nothing planned around it; an arm whose paths nobody
  judges refuses it; the motion verb refuses an arm without the capability, by name;
* mesh first (``safety.planned_motion.mesh_first``): the exact guard alone where it holds what the planner holds, and the
  planner asked beside it where it holds more: the switch off, no exact guard, a mesh in the planner's world, a distance
  field of the camera's points, a carried part, a planner that cannot say whether it carries one, and the declared
  support plane wherever a part of the arm or the hand comes within the guard's distance of its top;
* mesh first beside the bin of ``test_a_bin_the_camera_saw_beside_the_base_is_the_exact_guards.py``, on the real exact
  guard: the line the planner's spheres refuse runs without the planner being asked; on a cell that models a carried
  part, a hand not known empty and open asks it, and its refusal stands (the owner, 2026-10-01);
* how low the arm reaches (``MeshSelfCollisionBackend.lowest_mm``): the lowest point of every part past the shoulder,
  exact, the shoulder left out.

Honesty bucket (2): the real UR driver over a mocked controller and the route planner double, the real mesh backend over
a recording engine adapter.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.core import JointPositions, MotionCommand, MotionStatus
from src.robot.core.arm_capabilities import DrivesJointPaths
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.execution import motion
from src.robot.safety._fcl_self_collision import MeshSelfCollisionBackend
from src.robot.safety.planning.band import ExactPairs
from tests._plan_end import OPEN_WORKSPACE
from tests._route_planner import RoutePlanner
from tests._task_fakes import TaskArm

_DECLINED = "unit double: this file reads which path a joint path runs on, and no camera world is wired"

#: Where the fake controller says the arm is standing, and two waypoints on from it along one joint.
_HERE = [0.0, -1.5, 1.5, 0.0, 0.0, 0.0]
_ONE = [0.1, -1.5, 1.5, 0.0, 0.0, 0.0]
_TWO = [0.2, -1.5, 1.5, 0.0, 0.0, 0.0]
#: A waypoint off that line, so the path bends.
_BENT = [0.2, -1.4, 1.5, 0.0, 0.0, 0.0]


class _NeverPlans(RoutePlanner):
    """The route planner double, failing the test the moment it is asked to plan."""

    def __init__(self, test: unittest.TestCase, *, here: list[float]) -> None:
        super().__init__(here=here)
        self.test = test

    def plan_joint(self, goal: Any, *, refresh: bool = True, **_: Any) -> Any:
        self.test.fail(f"cuRobo was asked to plan to {list(goal)}: a joint path is never planned around")


def _arm(planner: RoutePlanner, *, motion_planner: str = "curobo", world: "dict[str, Any] | None" = None,
         mesh_first: bool = True) -> URRobotArm:
    safety: dict[str, Any] = {"payload": {"enforce": False}, "planned_motion": {"mesh_first": mesh_first}}
    if world is not None:
        safety["planning_world"] = world
    arm = URRobotArm(RobotConfig.model_validate({
        "vendor": "ur", "ur": {"model": "ur5e", "motion_planner": motion_planner},
        "safety": safety, "gripper": {"model": "robotiq_2f85"},
        "workspace_limits": OPEN_WORKSPACE,
    }))
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.get_joint_positions.return_value = list(planner.here)
    arm._conn.moveJ.return_value = True
    arm._conn.fk.return_value = [0.4, 0.0, 0.3, 0.0, 3.14159, 0.0]
    arm._curobo_ur = planner  # type: ignore[assignment]
    # The local path gate has its own tests (tests/test_ur_checked_joint_verbs.py); the destination guards stay real.
    arm._preflight.gate_planned_path = lambda waypoints, *, arm=None, command=None: None  # type: ignore[method-assign]
    return arm


def _path(*waypoints: list[float]) -> list[JointPositions]:
    return [JointPositions(w) for w in waypoints]


class TheJointPathTests(unittest.TestCase):
    def test_the_whole_path_is_judged_in_one_check_and_sent_as_one_list(self) -> None:
        planner = _NeverPlans(self, here=_HERE)
        arm = _arm(planner)

        with arm.without_camera_world(_DECLINED):
            result = arm.move_through_joints_on_the_line(_path(_ONE, _BENT, _HERE))

        self.assertTrue(result.ok, result.message)
        self.assertEqual((MotionCommand.MOVE_JOINTS, "move_through_joints_on_the_line"), (result.command,
                                                                                            result.message))
        (checked,) = planner.lines()
        np.testing.assert_allclose(checked[1][0], _HERE, atol=0.0)
        np.testing.assert_allclose(checked[1][-1], _HERE, atol=0.0)
        self.assertTrue(any(np.allclose(c, _BENT) for c in checked[1]), "the bent leg was not judged")
        self.assertEqual(3.0, checked[3], "the legs were not judged at safety.planned_motion.line_clearance_mm")
        (executed,) = planner.named("execute")
        self.assertEqual([tuple(_ONE), tuple(_BENT), tuple(_HERE)], executed[1])
        self.assertIs(MotionCommand.MOVE_JOINTS, executed[2])
        arm._conn.moveJ.assert_not_called()

    def test_a_leg_that_is_not_clear_refuses_the_whole_path_and_nothing_is_sent(self) -> None:
        planner = _NeverPlans(self, here=_HERE)
        planner.line_clear = lambda start, end: False
        arm = _arm(planner)

        with arm.without_camera_world(_DECLINED):
            result = arm.move_through_joints_on_the_line(_path(_ONE, _TWO))

        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status, result.message)
        self.assertIn("a joint path only: nothing is planned around it, nothing was sent", result.message or "")
        self.assertEqual([], planner.named("execute"))
        arm._conn.moveJ.assert_not_called()

    def test_the_destination_guards_judge_every_waypoint_first(self) -> None:
        planner = _NeverPlans(self, here=_HERE)
        arm = _arm(planner)

        with arm.without_camera_world(_DECLINED):
            result = arm.move_through_joints_on_the_line(_path(_ONE, [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]))

        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status, result.message)
        self.assertEqual([], planner.lines())
        self.assertEqual([], planner.named("execute"))

    def test_an_arm_whose_paths_nobody_judges_refuses_the_path(self) -> None:
        planner = _NeverPlans(self, here=_HERE)
        arm = _arm(planner, motion_planner="ik")

        with arm.without_camera_world(_DECLINED):
            result = arm.move_through_joints_on_the_line(_path(_ONE, _TWO))

        self.assertIs(MotionStatus.UNSUPPORTED, result.status, result.message)
        self.assertIn("nobody judges", result.message or "")
        self.assertEqual([], planner.named("execute"))
        arm._conn.moveJ.assert_not_called()

    def test_an_empty_path_is_refused(self) -> None:
        planner = _NeverPlans(self, here=_HERE)
        arm = _arm(planner)

        with arm.without_camera_world(_DECLINED):
            result = arm.move_through_joints_on_the_line([])

        self.assertIs(MotionStatus.INVALID_TARGET, result.status, result.message)
        self.assertEqual([], planner.named("execute"))

    def test_only_an_arm_with_the_capability_runs_one_and_the_verb_says_so(self) -> None:
        self.assertIsInstance(_arm(RoutePlanner(here=_HERE)), DrivesJointPaths)
        arm = TaskArm([])
        self.assertNotIsInstance(arm, DrivesJointPaths)
        report = motion.move_through_joints_on_the_line(arm, _path(_ONE, _TWO))
        self.assertIs(motion.MotionOutcome.REFUSED, report.outcome)
        self.assertIs(MotionStatus.UNSUPPORTED, report.status)
        self.assertIn("DrivesJointPaths", report.message)


class _Glue(RoutePlanner):
    """The route planner double, saying whether it carries a part as the real glue does."""

    carries_part = False


_PLANE = {"height_mm": 0.0, "extent_mm": [900.0, 700.0], "thickness_mm": 50.0}


def _pairs(lowest: "float | None" = 400.0, *, says: bool = True) -> ExactPairs:
    return ExactPairs(checks=lambda a, b: True, frames={}, distance=lambda joints, a, b: None, min_distance_mm=3.0,
                      lowest=(lambda joints: lowest) if says else None)


def _mesh_first(*, world: "dict[str, Any] | None" = None, pairs: "ExactPairs | None" = None,
                mesh_first: bool = True, planner: "RoutePlanner | None" = None,
                arm_setup: Any = None) -> "str | None":
    glue = planner if planner is not None else _Glue(here=_HERE)
    arm = _arm(glue, world=world, mesh_first=mesh_first)
    arm._preflight.exact_pairs = lambda arm=None: pairs  # type: ignore[method-assign]
    if arm_setup is not None:
        arm_setup(arm)
    return arm._mesh_first_refusal(glue, [_HERE, _ONE])


class MeshFirstTests(unittest.TestCase):
    def test_the_exact_guard_alone_where_it_holds_what_the_planner_holds(self) -> None:
        self.assertIsNone(_mesh_first(world={"enabled": True, "support_plane": _PLANE}, pairs=_pairs()))
        self.assertIsNone(_mesh_first(pairs=_pairs(lowest=None, says=False)), "no planning world, no plane to keep")

    def test_the_switch_off_or_no_exact_guard_asks_the_planner(self) -> None:
        self.assertIn("mesh_first is off", _mesh_first(pairs=_pairs(), mesh_first=False) or "")
        self.assertIn("no exact mesh guard", _mesh_first(pairs=None) or "")

    def test_what_only_the_planner_holds_asks_it(self) -> None:
        field = {"enabled": True, "support_plane": _PLANE, "perceived": {"voxel_field_mm": 20.0}}
        self.assertIn("distance field", _mesh_first(world=field, pairs=_pairs()) or "")

        def carrying(arm: URRobotArm) -> None:
            arm._attached_payload = object()  # type: ignore[assignment]

        self.assertIn("a part is carried", _mesh_first(pairs=_pairs(), arm_setup=carrying) or "")
        loaded = _Glue(here=_HERE)
        loaded.carries_part = True
        self.assertIn("a part is carried", _mesh_first(pairs=_pairs(), planner=loaded) or "")
        self.assertIn("cannot say whether it carries", _mesh_first(pairs=_pairs(), planner=RoutePlanner(here=_HERE))
                      or "")

    def test_the_support_plane_asks_the_planner_where_the_arm_comes_near_it(self) -> None:
        world = {"enabled": True, "support_plane": _PLANE}
        said = _mesh_first(world=world, pairs=_pairs(lowest=2.0)) or ""
        self.assertIn("2.0 mm over the declared support plane", said)
        self.assertIn("within the guard's 3 mm", said)
        self.assertIsNone(_mesh_first(world=world, pairs=_pairs(lowest=3.5)))
        self.assertIn("cannot say how low", _mesh_first(world=world, pairs=_pairs(says=False)) or "")
        self.assertIn("cannot place the arm", _mesh_first(world=world, pairs=_pairs(lowest=None)) or "")

    def test_a_raised_plane_is_read_at_its_height(self) -> None:
        raised = {"enabled": True, "support_plane": {**_PLANE, "height_mm": 398.0}}
        self.assertIn("2.0 mm over", _mesh_first(world=raised, pairs=_pairs(lowest=400.0)) or "")

    def test_on_a_cell_that_models_a_carried_part_a_hand_not_known_open_asks_the_planner(self) -> None:
        def unknown(arm: URRobotArm) -> None:
            arm.set_hand(None)

        modelled = {"enabled": True, "support_plane": _PLANE, "payload": {"length_mm": 1.0}}
        self.assertIn("not known to be empty and open", _mesh_first(world=modelled, pairs=_pairs(), arm_setup=unknown)
                      or "")
        unmodelled = {"enabled": True, "support_plane": _PLANE, "payload": {"enabled": False}}
        self.assertIsNone(_mesh_first(world=unmodelled, pairs=_pairs(), arm_setup=unknown),
                          "a cell that models no carried part reads no hand (the owner, 2026-10-05)")


class MeshFirstBesideTheBinTests(unittest.TestCase):
    """The real exact guard on the UR10 and the Hand-E, the bin 30 mm from the arm's meshes that only the planner's
    spheres refuse (``test_a_bin_the_camera_saw_beside_the_base_is_the_exact_guards.py``), mesh first on."""

    def test_the_line_runs_and_the_planner_is_not_asked(self) -> None:
        from tests.test_a_bin_the_camera_saw_beside_the_base_is_the_exact_guards import OFF, OFF_ON, _cell, _from, _line

        arm, planner = _cell(mesh_first=True)
        _from(arm, planner, OFF)
        result = _line(arm, OFF_ON)
        self.assertTrue(result.ok, result.message)
        arm._conn.moveJ.assert_called_once()
        self.assertEqual([], [call for call in planner.named("check") if len(call[1]) > 1],
                         "the planner's spheres were asked about the line")
        self.assertEqual([], planner.set_aside())

    def test_a_hand_not_known_open_on_a_cell_that_models_a_part_leaves_it_to_the_planner(self) -> None:
        from tests.test_a_bin_the_camera_saw_beside_the_base_is_the_exact_guards import OFF, OFF_ON, _cell, _from, _line

        arm, planner = _cell(mesh_first=True, carried_part_mm=40.0)
        arm.set_hand(None)
        _from(arm, planner, OFF)
        result = _line(arm, OFF_ON)
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status, result.message)
        arm._conn.moveJ.assert_not_called()
        self.assertIn("the hand is not known to be empty and open", result.message or "")


class _Adapter:
    """An engine adapter that builds nothing: ``lowest_mm`` reads the vertices alone."""

    kind = "recording"

    def build_object(self, vertices: Any, faces: Any) -> object:
        return object()


def _cube(low_z: float, high_z: float) -> tuple[np.ndarray, np.ndarray]:
    corners = np.asarray([[x, y, z] for x in (-10.0, 10.0) for y in (-10.0, 10.0) for z in (low_z, high_z)])
    return corners, np.zeros((0, 3), dtype=np.int64)


class HowLowTheArmReachesTests(unittest.TestCase):
    def _backend(self) -> MeshSelfCollisionBackend:
        shoulder_v, shoulder_f = _cube(-500.0, 10.0)
        arm_v, arm_f = _cube(50.0, 60.0)
        finger_v, finger_f = _cube(5.0, 8.0)
        return MeshSelfCollisionBackend(_Adapter(), {  # type: ignore[arg-type]
            "shoulder": (shoulder_v, shoulder_f, 1), "upper_arm": (arm_v, arm_f, 2), "rfinger": (finger_v, finger_f, 6)})

    def test_the_lowest_point_of_every_part_past_the_shoulder(self) -> None:
        frames = [np.eye(4) for _ in range(7)]
        frames[6] = np.eye(4)
        frames[6][2, 3] = 100.0
        self.assertAlmostEqual(50.0, self._backend().lowest_mm(frames, 0.0))
        frames[6][2, 3] = 20.0
        self.assertAlmostEqual(25.0, self._backend().lowest_mm(frames, 0.0), msg="the finger, lifted 20 mm")

    def test_a_turned_part_reaches_as_turned(self) -> None:
        frames = [np.eye(4) for _ in range(7)]
        frames[6][2, 3] = 1000.0
        flipped = np.eye(4)
        flipped[:3, :3] = np.diag([1.0, -1.0, -1.0])  # half a turn about X
        frames[2] = flipped
        self.assertAlmostEqual(-60.0, self._backend().lowest_mm(frames, 37.0))

    def test_the_shoulder_is_left_out_and_from_frame_brings_it_in(self) -> None:
        frames = [np.eye(4) for _ in range(7)]
        self.assertAlmostEqual(-500.0, self._backend().lowest_mm(frames, 0.0, from_frame=1))


if __name__ == "__main__":
    unittest.main()
