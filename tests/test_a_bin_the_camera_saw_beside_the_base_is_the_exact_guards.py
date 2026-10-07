"""A bin the camera saw beside the base is the exact guard's to judge where only the camera's boxes refuse the planner.

The owner's Option 1, its own commit after the guard fixes of 2026-09-30. cuRobo judges its world with a sphere cover that
reaches 25 to 29 mm past the UR10's shoulder housing, and the world term stayed its own, so a bin the camera saw needed
about 50 mm for a plan and about 60 for a straight line, where the exact guard alone takes about 26. Now, where the
planner refuses samples on its world, the driver asks it once more with every box the camera saw set aside, and the
refused samples are admitted only where all of this holds:

* the second judgement clears the world and the bounds of every refused sample and finds the robot itself as the first
  did (its self pairs are decided as F1 decides them);
* no part is carried, for the arm or for the planner: the carried part is the planner's alone;
* the exact guard and the planner hold the same boxes the camera saw, and the exact guard accepts every refused sample
  right there with them, turned, at ``perceived_min_distance_mm``;
* the exact mesh engine runs.

The bench, the declared fixtures and meshes, and the carried part stay the planner's. Anything unexpected leaves the
refusal standing, and the sentence says which authority it stands on.

The exact guard here is the real one, on the committed UR10 and Hand-E bundles, and the camera world the real pipeline:
a bin ray cast straight down, the live world, its refresh. The planner is a double: its world is the boxes it holds, and
its sphere cover reaches :data:`REACH_MM` past the arm's meshes, the shoulder housing's measured 25 to 29 mm; its own
term is the band of ``test_a_pose_only_the_planners_spheres_refuse_is_the_exact_guards.py``. At the investigation's
joints (0, -60, 80, -110, -90, 0) deg a bin turned 30 degrees 30 mm from the arm's meshes is 9.1 mm from the boxes the
camera grew around it: the exact guard accepts it at 5, the planner's 27 mm meets it.

Those joints lie in the band as well, so the planner refuses them on the robot itself too and types its verdict a self
collision. The ordinary case at a cell is a pose beside a bin out of the band: the planner refuses it on its world alone
and types the verdict WORLD. :data:`OFF` is that pose, the wrist turned out of the band with the housing where it was,
and every decision is held there too (verifier W, 2026-10-01).
"""

from __future__ import annotations

import math
import unittest
from functools import lru_cache
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionCommand, MotionStatus
from src.robot.core.arm_capabilities import PayloadModel
from src.robot.drivers.ur.pose_adapter import pose_to_urpose
from src.robot.safety._capsule import AxisAlignedBox
from src.robot.safety._fcl_self_collision import mesh_backend_status
from src.robot.safety._ur_kinematics import ur_link_transforms_mm
from src.robot.safety.planning import CuroboUnavailableError, StateRefusalKind
from src.robot.safety.planning.band import PoseVerdict
from src.robot.safety.planning.curobo_client import PathJudgement, RefusedSample
from tests._plan_end import pose_where_it_ends
from tests._seen_scenes import Solid, open_bin
from tests.test_a_pose_only_the_planners_spheres_refuse_is_the_exact_guards import BandPlanner
from tests.test_the_guard_holds_the_boxes_the_planner_holds import _refresh

#: How far the planner's sphere cover reaches past the arm's meshes beside the shoulder housing, millimetres: the CPU
#: replica measured 25 to 29 (2026-09-30), and the GPU agreed at 47.7, 54.3 and 60.8 mm of real clearance.
REACH_MM = 27.0
#: The investigation's joints: the upper arm's shoulder housing hangs along base -Y, 52 mm over the base plate. wrist_1
#: at -110 degrees is in the band double's cushion band too, so the robot's own term refuses here as well.
Q_DEG = (0.0, -60.0, 80.0, -110.0, -90.0, 0.0)
Q = [math.radians(v) for v in Q_DEG]
#: The same joints with the wrist turned on: the housing does not move, the band holds, the bin stays 30 mm away.
Q_ON = [math.radians(v) for v in (0.0, -60.0, 80.0, -105.0, -90.0, 0.0)]
#: The investigation's joints with the wrist turned out of the band (wrist_1 -95, where the double's band ends at -101;
#: the GPU probe holds the kernel's self term clear there too): the planner finds nothing of the robot itself, the
#: housing is where it was at Q, and the bin stays 30 mm away. Beside it the planner refuses on its world alone and types
#: the verdict WORLD.
OFF_DEG = (0.0, -60.0, 80.0, -95.0, -90.0, 0.0)
OFF = [math.radians(v) for v in OFF_DEG]
#: The same with the wrist turned on further, still out of the band and beside the bin.
OFF_ON = [math.radians(v) for v in (0.0, -60.0, 80.0, -90.0, -90.0, 0.0)]
#: OFF with the shoulder turned 30 degrees away from the bin: the housing keeps 65 mm from the camera's boxes, past the
#: planner's reach and the line clearance, so a line from here meets the bin only on its way.
AWAY = [math.radians(v) for v in (-30.0, -60.0, 80.0, -95.0, -90.0, 0.0)]
_DECLINED = "unit double: the boxes the camera saw are handed to both authorities by hand"
_LINE_MM = 10.0


def _needs_the_engine() -> None:
    if mesh_backend_status("ur10", mesh_name="robotiq_hande") != "ok":
        raise unittest.SkipTest("no exact mesh backend on this box")


def _arm(fixtures: "list[dict[str, Any]] | None" = None, *, carried_part_mm: "float | None" = None,
         mesh_first: bool = False) -> Any:
    """The owner-like UR10 with the Hand-E standing at Q, built as ``owner_like_arm`` builds it, ``fixtures`` declared where given.

    It keeps the distances this file's scenes were measured at: 10 mm from the arm and a declared fixture, 5 mm from a
    seen box, a 10 mm line clearance. The owner ships 3, 3 and 3 since 2026-10-05; the mechanism is the same.
    ``carried_part_mm`` declares how far a carried part hangs past the fingertips: the cell then models a carried part
    (``safety.planning_world.payload.length_mm``); the owner's models none. These scenes pin Option 1, which mesh first
    (``safety.planned_motion.mesh_first``, on as shipped) never reaches where it applies, so it is off unless
    ``mesh_first``: its own scene beside this bin is ``test_a_joint_path_is_judged_whole_and_sent_without_a_pause.py``."""
    _needs_the_engine()
    from src.config.schema.robot import RobotConfig
    from src.robot.drivers.ur.arm import URRobotArm

    from tests._plan_end import OPEN_WORKSPACE

    arm: Any = URRobotArm(RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"model": "ur10", "motion_planner": "curobo"},
        "workspace_limits": OPEN_WORKSPACE,
        "gripper": {"model": "robotiq_hande", "coupling_plates": [{"name": "adapter", "thickness_mm": 20.0}],
                    "tool_frame": {"source": "willy", "offset_mm": [0.0, 0.0, 155.75],
                                   "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        "safety": {"payload": {"enforce": False},
                   "planned_motion": {"line_clearance_mm": _LINE_MM, "mesh_first": bool(mesh_first)},
                   "self_collision": {"backend": "fcl", "min_distance_mm": 10.0, "perceived_min_distance_mm": 5.0,
                                      "kinematics_model": "ur10", "planner_margin_mm": 4.0,
                                      "fixtures": list(fixtures or [])},
                   **({} if carried_part_mm is None else
                      {"planning_world": {"payload": {"length_mm": float(carried_part_mm)}}})},
    }))
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.get_joint_positions.return_value = list(Q)
    arm._conn.moveJ.return_value = True
    return arm


def _backend(arm: Any) -> Any:
    return arm._preflight._path_authority(arm)._exact_mesh_backend("ur10")


def _distance(backend: Any, joints: "list[float]", boxes: "tuple[AxisAlignedBox, ...]") -> float:
    """The exact meshes' distance at ``joints`` to the nearest of ``boxes``, millimetres, by bisection."""
    transforms = ur_link_transforms_mm("ur10", np.asarray(joints, dtype=np.float64))
    if backend.evaluate(transforms, 0.0, boxes, 1e-6, arm_pairs=False) is not None:
        return 0.0
    low, high = 0.0, 600.0
    for _ in range(40):
        middle = (low + high) / 2.0
        if backend.evaluate(transforms, 0.0, boxes, middle, arm_pairs=False) is None:
            low = middle
        else:
            high = middle
    return low


def _solids(solids: "list[Solid]") -> "tuple[AxisAlignedBox, ...]":
    from src.robot.safety._capsule import TurnedBox

    return tuple(AxisAlignedBox(center_mm=np.asarray(s.centre, dtype=np.float64),
                                half_extents_mm=np.asarray(s.half, dtype=np.float64), name=f"solid_{i}",
                                turned=None if s.yaw_deg == 0.0 else TurnedBox(
                                    half_extents_mm=np.asarray(s.half, dtype=np.float64),
                                    yaw_rad=math.radians(s.yaw_deg)))
                 for i, s in enumerate(solids))


@lru_cache(maxsize=16)
def _bin(real_mm: float, yaw_deg: float = 30.0) -> "tuple[float, tuple[Solid, ...]]":
    """Where a 300 x 200 mm bin with a 40 mm rim, turned ``yaw_deg``, stands ``real_mm`` from the arm's meshes at Q."""
    backend = _backend(_arm())
    near, far = -250.0, -800.0
    for _ in range(50):
        middle = (near + far) / 2.0
        if _distance(backend, Q, _solids(open_bin((0.0, middle), (300.0, 200.0), 40.0, yaw_deg=yaw_deg))) > real_mm:
            far = middle
        else:
            near = middle
    y = (near + far) / 2.0
    return y, tuple(open_bin((0.0, y), (300.0, 200.0), 40.0, yaw_deg=yaw_deg))


@lru_cache(maxsize=16)
def _seen(real_mm: float, yaw_deg: float = 30.0) -> "tuple[AxisAlignedBox, ...]":
    """The boxes the real camera world builds of that bin, seen from 900 mm straight over it, as the guard holds them."""
    y, solids = _bin(real_mm, yaw_deg)
    refresh = _refresh(list(solids), (0.0, y))
    assert refresh.ok, refresh.render()
    return tuple(refresh.guard_boxes)


#: The bench the planner holds, its top at the base plate: 38.3 mm under the shoulder mesh at Q, past the reach and the
#: line clearance both.
_BENCH = AxisAlignedBox(center_mm=np.array([0.0, 0.0, -25.0]), half_extents_mm=np.array([1000.0, 1000.0, 25.0]),
                        name="support_plane")
#: A part far from the arm, which the camera saw too: a box to set aside that refuses nothing.
_FAR_PART = AxisAlignedBox(center_mm=np.array([500.0, 400.0, 30.0]), half_extents_mm=np.array([30.0, 30.0, 30.0]),
                           name="seen_09_part")


class HandKnownOpen:
    """A hand known empty and open, as the owner's toggle reads once a person said its jaws stand open: connected, its
    count standing and saying open. The camera's boxes are set aside for no other hand (the owner, 2026-10-01; what other
    hands say is ``test_the_cameras_boxes_are_set_aside_only_for_a_hand_known_open.py``)."""

    toggles_without_sensor = True
    jaws_closed = False
    edge_unknown = False
    is_connected = True
    min_width_mm = 5.0
    max_width_mm = 49.99

    def why_jaws_unknown(self) -> str:
        return ""

    def jaws_open_for_a_pick(self) -> str:
        return ""


#: The hand every cell of this file carries unless a test hands it another.
KNOWN_OPEN = HandKnownOpen()


class SeenWorldPlanner(BandPlanner):
    """cuRobo beside the base: its world is the boxes it holds, met where the exact meshes come within :data:`REACH_MM`
    plus the clearance asked of a box it holds and has not set aside; the robot itself is the band double's.

    ``judge_joint_path`` takes ``ignore_perceived`` as the glue does: every camera box it holds, by name, or it refuses as
    the sidecar refuses. ``carries_part`` is what the glue says of a part the planner may hold. Every report is recorded
    with the names it set aside, ``()`` where none.
    """

    def __init__(self, arm: Any, world: "tuple[AxisAlignedBox, ...]", *, carries_part: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._backend = _backend(arm)
        self.world = {box.name: box for box in world}
        self.carries_part = carries_part
        self.aside: tuple[str, ...] = ()

    def in_world(self, config: "list[float] | tuple[float, ...]", clearance_mm: float) -> bool:
        held = tuple(box for name, box in self.world.items() if name not in self.aside)
        if not held:
            return False
        transforms = ur_link_transforms_mm("ur10", np.asarray(config, dtype=np.float64))
        return self._backend.evaluate(transforms, 0.0, held, REACH_MM + float(clearance_mm), arm_pairs=False) is not None

    def judge_joint_path(self, samples: Any, *, clearance_mm: float = 0.0, name_pairs: bool = True,
                         ignore_perceived: "tuple[str, ...] | None" = None) -> PathJudgement:
        names = () if ignore_perceived is None else tuple(sorted(ignore_perceived))
        if ignore_perceived is not None and list(names) != sorted(n for n in self.world if n.startswith("seen_")):
            raise CuroboUnavailableError(f"the planner holds other boxes the camera saw than {list(names)}")
        self.aside = names
        try:
            judged = super().judge_joint_path(samples, clearance_mm=clearance_mm, name_pairs=name_pairs)
        finally:
            self.aside = ()
        self.calls[-1] = (*self.calls[-1], names)
        return PathJudgement(checked=judged.checked, clearance_mm=judged.clearance_mm, refused=judged.refused,
                             pairs_named=judged.pairs_named, perceived_ignored=names)

    def set_aside(self) -> "list[tuple[Any, ...]]":
        """Every report asked with the camera's boxes set aside."""
        return [call for call in self.named("judge") if call[4]]

    def detach_payload(self) -> bool:
        """The glue's detach, confirmed: it holds no part after it."""
        self.calls.append(("detach",))
        self.carries_part = False
        return True


def _cell(real_mm: float = 30.0, *, yaw_deg: float = 30.0, world: "tuple[AxisAlignedBox, ...] | None" = None,
          seen: "tuple[AxisAlignedBox, ...] | None" = None, fixtures: "list[dict[str, Any]] | None" = None,
          planner_cls: "type[SeenWorldPlanner]" = SeenWorldPlanner, carried_part_mm: "float | None" = None,
          mesh_first: bool = False, **kwargs: Any) -> "tuple[Any, SeenWorldPlanner]":
    """The arm at Q with a bin ``real_mm`` from its meshes the camera saw, held by both authorities, carrying a hand
    known empty and open (:data:`KNOWN_OPEN`); ``carried_part_mm`` and ``mesh_first`` as :func:`_arm` takes them."""
    arm = _arm(fixtures, carried_part_mm=carried_part_mm, mesh_first=mesh_first)
    boxes = _seen(real_mm, yaw_deg) if seen is None else seen
    arm._preflight.set_perceived_obstacles(boxes)
    planner = planner_cls(arm, (_BENCH, *boxes) if world is None else world, **kwargs)
    planner.here = list(Q)
    arm._curobo_ur = planner
    arm.set_hand(KNOWN_OPEN)
    return arm, planner


def _line(arm: Any, target: "list[float]") -> Any:
    with arm.without_camera_world(_DECLINED):
        return arm.move_to_joints_on_the_line(JointPositions(tuple(target)))


def _decide(arm: Any, planner: SeenWorldPlanner, configs: "list[list[float]]", clearance_mm: float) -> Any:
    verdict = planner.check_joint_path(configs, refresh=False, clearance_mm=clearance_mm)
    assert not verdict.valid, "the planner was meant to refuse these"
    return arm._exact_guard_decides(planner, configs, verdict, clearance_mm=clearance_mm,
                                    command=MotionCommand.MOVE_JOINTS)


def _from(arm: Any, planner: SeenWorldPlanner, config: "list[float]") -> None:
    """Stand the arm, and the planner's double, at ``config``."""
    arm._conn.get_joint_positions.return_value = list(config)
    planner.here = list(config)


def _declared_wall(real_mm: float) -> "tuple[AxisAlignedBox, dict[str, Any]]":
    """The near wall of the square bin ``real_mm`` from the arm's meshes at Q, declared at its measured place: the box
    both authorities hold, and the fixture that declares it."""
    _, solids = _bin(real_mm, 0.0)
    wall = solids[3]
    box = AxisAlignedBox(center_mm=np.asarray(wall.centre, dtype=np.float64),
                         half_extents_mm=np.asarray(wall.half, dtype=np.float64), name="bin_wall")
    return box, {"name": "bin_wall", "center_mm": list(wall.centre), "half_extents_mm": list(wall.half)}


#: A lamp declared over the base, 320 to 380 mm up beside the upper arm: the exact guard keeps its 10 mm from it along the
#: whole line from AWAY to OFF (12 mm at the nearest), and the planner's line clearance meets it on a stretch of that line
#: inside the samples the camera's boxes refuse, never at its first.
_LAMP_CENTRE = (100.0 * math.sin(math.radians(-37.0)), -100.0 * math.cos(math.radians(-37.0)), 350.0)
_LAMP = AxisAlignedBox(center_mm=np.asarray(_LAMP_CENTRE, dtype=np.float64),
                       half_extents_mm=np.array([10.0, 10.0, 30.0]), name="lamp")


class TheScenesPremisesTests(unittest.TestCase):
    def test_the_bin_30_mm_away_is_9_mm_from_the_cameras_boxes_and_inside_the_planners_reach(self) -> None:
        """The numbers this file rests on: the exact guard keeps 5 mm and accepts 9.1; the planner's 27 mm meets it."""
        arm = _arm()
        backend = _backend(arm)
        _, solids = _bin(30.0)
        self.assertAlmostEqual(_distance(backend, Q, _solids(list(solids))), 30.0, delta=0.05)
        to_seen = _distance(backend, Q, _seen(30.0))
        self.assertGreater(to_seen, 5.0)
        self.assertLess(to_seen, REACH_MM)
        self.assertGreater(_distance(backend, Q, (_BENCH,)), REACH_MM + _LINE_MM, "the bench refuses nothing at Q")
        self.assertEqual(float(arm.config.safety.planned_motion.line_clearance_mm), _LINE_MM)
        self.assertEqual(float(arm.config.safety.self_collision.perceived_min_distance_mm), 5.0)


class TheCamerasBoxesAloneAreTheExactGuardsTests(unittest.TestCase):
    def test_a_straight_line_beside_a_bin_30_mm_from_the_housing_runs(self) -> None:
        """⭐ _judge_legs: the planner refuses every sample of the line on its world (and on the band); asked again with
        the camera's boxes set aside it clears the world; the exact guard accepts each at 5 mm. One moveJ."""
        arm, planner = _cell()
        result = _line(arm, Q_ON)
        self.assertTrue(result.ok, result.message)
        arm._conn.moveJ.assert_called_once()
        (check,) = [c for c in planner.named("check") if len(c[1]) > 1]
        self.assertEqual(check[2], _LINE_MM, "a straight line is held at the line clearance")
        self.assertTrue(all(planner.in_world(config, check[2]) for config in check[1]))
        (again,) = planner.set_aside()
        self.assertEqual(again[2], check[2], "asked again at the clearance it was refused at")
        self.assertEqual(list(again[4]), sorted(box.name for box in _seen(30.0)))
        self.assertEqual(again[1], check[1], "every sample the world refused, and only those")

    def test_a_cartesian_goal_beside_the_bin_is_admitted_at_the_screen(self) -> None:
        """⭐ The goal screen: the goal configuration the planner refuses on its world, at no clearance, is the exact
        guard's; the line to it too; the move runs on that goal, one line."""
        arm, planner = _cell()
        arm._conn.get_joint_positions.return_value = list(Q_ON)
        planner.here = list(Q_ON)
        with arm.without_camera_world(_DECLINED):
            result = arm.move(pose_where_it_ends(arm, Q))
        self.assertTrue(result.ok, result.message)
        screened = [c for c in planner.named("check") if len(c[1]) == 1]
        self.assertTrue(screened and screened[0][2] == 0.0 and planner.in_world(screened[0][1][0], 0.0))
        self.assertTrue([c for c in planner.set_aside() if c[2] == 0.0], "the screen asked with the boxes set aside")
        (executed,) = planner.named("execute")
        self.assertLess(max(abs(a - b) for a, b in zip(executed[1][-1], Q)), 1e-6)
        self.assertFalse(planner.named("plan"))

    def test_a_tool_line_beside_the_bin_runs(self) -> None:
        """⭐ moveL: every sample solved, the exact gate, then the planner, then this admission; one moveL."""
        arm, planner = _cell()
        arm.ik = lambda pose, *, seed=None: JointPositions(tuple(Q))
        arm._motion = MagicMock()
        arm._motion.move_to.return_value = True
        flange = np.asarray(ur_link_transforms_mm("ur10", np.asarray(Q))[-1])
        arm._motion.get_current_pose.return_value = pose_to_urpose(Pose.from_matrix(flange, frame=Frame.BASE))
        start = arm.get_tcp_pose()
        with arm.without_camera_world(_DECLINED):
            result = arm.move(start, linear=True)
        self.assertTrue(result.ok, result.message)
        arm._motion.move_to.assert_called_once()
        (check,) = planner.named("check")
        (again,) = planner.set_aside()
        self.assertEqual(again[1], check[1])

    def test_the_bin_at_40_and_47_7_mm_is_admitted_too(self) -> None:
        for real_mm in (40.0, 47.7):
            with self.subTest(real_mm=real_mm):
                arm, planner = _cell(real_mm)
                for clearance in (0.0, _LINE_MM):
                    self.assertIsNone(_decide(arm, planner, [Q], clearance))


class WhatStaysThePlannersTests(unittest.TestCase):
    def test_the_same_bin_while_a_part_is_carried_is_refused(self) -> None:
        """⭐ The carried part is the planner's alone: nothing is set aside while the arm or the planner may hold one."""
        for what, carried in (("the planner holds one", {"carries_part": True}), ("the arm holds one", {})):
            with self.subTest(what=what):
                arm, planner = _cell(**carried)
                if not carried:
                    arm._attached_payload = (120.0, 5.0)
                result = _line(arm, Q_ON)
                self.assertIs(result.status, MotionStatus.SELF_COLLISION_REJECTED)
                arm._conn.moveJ.assert_not_called()
                self.assertEqual(planner.set_aside(), [], "the planner was never asked with the boxes set aside")
                for said in ("the planner refused this straight joint line at sample 0", "the planner's world",
                             "a part is carried"):
                    self.assertIn(said, result.message or "")

    def test_a_planner_that_cannot_say_it_holds_no_part_is_refused(self) -> None:
        arm, planner = _cell()
        del planner.carries_part
        self.assertFalse(hasattr(planner, "carries_part"))
        standing = _decide(arm, planner, [Q], 0.0)
        assert standing is not None
        self.assertIn("cannot say", standing.reason)
        self.assertEqual(planner.set_aside(), [])

    def test_the_same_bin_declared_stays_the_planners(self) -> None:
        """⭐ A declared fixture is never set aside. The square bin's near wall 30 mm from the meshes, declared at its
        measured place: the exact guard keeps its 10 mm and accepts it, the planner's line clearance meets it, and the
        refusal stands on the planner's world even with the camera's own boxes set aside. Seen, the same bin runs."""
        y, solids = _bin(30.0, 0.0)
        wall = solids[3]
        self.assertAlmostEqual(_distance(_backend(_arm()), Q, _solids([wall])), 30.0, delta=0.05)
        declared = AxisAlignedBox(center_mm=np.asarray(wall.centre, dtype=np.float64),
                                  half_extents_mm=np.asarray(wall.half, dtype=np.float64), name="bin_wall")
        arm, planner = _cell(seen=(_FAR_PART,), world=(_BENCH, declared, _FAR_PART),
                             fixtures=[{"name": "bin_wall", "center_mm": list(wall.centre),
                                        "half_extents_mm": list(wall.half)}])
        self.assertIsNone(arm._preflight.gate_joint_target(JointPositions(tuple(Q)), arm=arm),
                          "the exact guard accepts the declared wall at its 10 mm")
        result = _line(arm, Q_ON)
        self.assertIs(result.status, MotionStatus.SELF_COLLISION_REJECTED)
        arm._conn.moveJ.assert_not_called()
        (again,) = planner.set_aside()
        self.assertEqual(again[4], ("seen_09_part",), "the camera's own box was set aside, the declared wall never")
        for said in ("the planner's world refuses it with the boxes the camera saw set aside too", "declared",
                     "stay the planner's"):
            self.assertIn(said, result.message or "")

        seen_arm, _ = _cell(seen=_seen(30.0, 0.0))
        self.assertTrue(_line(seen_arm, Q_ON).ok, "the same bin, seen and not declared, runs")

    def test_the_bench_stays_the_planners(self) -> None:
        """A bench the planner meets is the planner's whatever the camera saw: with the bin's boxes set aside the
        planner still meets the bench (raised 20 mm over the base plate, 18.3 mm under the shoulder mesh)."""
        raised = AxisAlignedBox(center_mm=np.array([0.0, 0.0, -5.0]), half_extents_mm=np.array([1000.0, 1000.0, 25.0]),
                                name="support_plane")
        arm, planner = _cell(world=(raised, *_seen(30.0)))
        standing = _decide(arm, planner, [Q], 0.0)
        assert standing is not None and standing.row is not None
        self.assertEqual(standing.row.index, 0)
        self.assertIn("the bench", standing.reason)
        self.assertIn("set aside too", standing.reason)
        self.assertEqual(len(planner.set_aside()), 1)

    def test_nothing_is_set_aside_where_the_exact_mesh_engine_does_not_run(self) -> None:
        """⭐ The capsule proxy decides nothing the planner refused: without the exact engine the world stands, and the
        planner is never asked with the camera's boxes set aside."""
        arm, planner = _cell()
        arm._preflight.exact_pairs = lambda arm=None: None
        standing = _decide(arm, planner, [Q], 0.0)
        assert standing is not None
        self.assertIn("no exact mesh guard runs here", standing.reason)
        self.assertEqual(planner.set_aside(), [])
        self.assertEqual(planner.named("judge"), [], "no report was even asked for")

    def test_a_world_refusal_with_no_box_the_camera_saw_stands(self) -> None:
        arm, planner = _cell(seen=(), world=(_BENCH, *_seen(30.0)))
        standing = _decide(arm, planner, [Q], 0.0)
        assert standing is not None
        self.assertIn("holds no box the camera saw", standing.reason)
        self.assertEqual(planner.set_aside(), [])

    def test_boxes_no_camera_world_named_are_never_set_aside(self) -> None:
        """Only boxes the camera world named (``seen_``) are ever set aside: a guard holding another name, or one name
        twice, keeps the planner's world, and the planner is never asked."""
        seen = _seen(30.0)
        renamed = tuple(AxisAlignedBox(center_mm=box.center_mm, half_extents_mm=box.half_extents_mm, name=name,
                                       turned=box.turned)
                        for box, name in zip(seen, ["bin_wall", *[b.name for b in seen[1:]]]))
        twice = tuple(AxisAlignedBox(center_mm=box.center_mm, half_extents_mm=box.half_extents_mm, name=seen[0].name,
                                     turned=box.turned) for box in seen)
        for what, boxes in (("another name", renamed), ("one name twice", twice)):
            with self.subTest(what=what):
                arm, planner = _cell(seen=boxes, world=(_BENCH, *seen))
                standing = _decide(arm, planner, [Q], 0.0)
                assert standing is not None
                self.assertIn("did not name as its own", standing.reason)
                self.assertEqual(planner.set_aside(), [])

    def test_the_exact_guard_has_the_last_word(self) -> None:
        """⭐ A bin 22 mm from the meshes is 1.7 mm from the camera's boxes: the planner's world clears with them set
        aside, and the exact guard, judging the very sample again, refuses it at 5 mm, naming the box."""
        arm, planner = _cell(22.0)
        standing = _decide(arm, planner, [Q], 0.0)
        assert standing is not None
        for said in ("the exact guard refuses it too", "fixture:seen_", "< 5.000 mm"):
            self.assertIn(said, standing.reason)
        self.assertIs(standing.status, MotionStatus.SELF_COLLISION_REJECTED)
        self.assertEqual(len(planner.set_aside()), 1, "the guard spoke after the planner's second judgement")

    def test_no_second_judgement_where_the_planner_refused_only_the_robot_itself(self) -> None:
        """⭐ THE CONTROL: a refusal on the band alone is F1's, decided as before, and the planner is never asked with
        anything set aside."""
        arm, planner = _cell(world=(_BENCH, _FAR_PART), seen=(_FAR_PART,))
        self.assertTrue(_line(arm, Q_ON).ok)
        self.assertTrue(planner.named("judge"))
        self.assertEqual(planner.set_aside(), [])


class ShiftedSecondReport(SeenWorldPlanner):
    """A second judgement that opens on another sample than was asked."""

    def judge_joint_path(self, samples: Any, **asked: Any) -> PathJudgement:
        judged = super().judge_joint_path(samples, **asked)
        if not judged.perceived_ignored:
            return judged
        return PathJudgement(checked=judged.checked, clearance_mm=judged.clearance_mm,
                             refused=(RefusedSample(index=judged.checked, bound_ok=True, self_ok=True,
                                                    world_ok=False, pairs=()),),
                             pairs_named=judged.pairs_named, perceived_ignored=judged.perceived_ignored)


class MiscountedSecondReport(SeenWorldPlanner):
    def judge_joint_path(self, samples: Any, **asked: Any) -> PathJudgement:
        judged = super().judge_joint_path(samples, **asked)
        if not judged.perceived_ignored:
            return judged
        return PathJudgement(checked=judged.checked + 1, clearance_mm=judged.clearance_mm, refused=judged.refused,
                             pairs_named=judged.pairs_named, perceived_ignored=judged.perceived_ignored)


class OtherBoxesSetAside(SeenWorldPlanner):
    def judge_joint_path(self, samples: Any, **asked: Any) -> PathJudgement:
        judged = super().judge_joint_path(samples, **asked)
        if not judged.perceived_ignored:
            return judged
        return PathJudgement(checked=judged.checked, clearance_mm=judged.clearance_mm, refused=judged.refused,
                             pairs_named=judged.pairs_named, perceived_ignored=judged.perceived_ignored[:-1])


class ItsOwnTermChangedWithTheWorld(SeenWorldPlanner):
    """A second judgement that finds the robot itself clear where the first refused it: no report on these samples."""

    def judge_joint_path(self, samples: Any, **asked: Any) -> PathJudgement:
        judged = super().judge_joint_path(samples, **asked)
        if not judged.perceived_ignored:
            return judged
        return PathJudgement(checked=judged.checked, clearance_mm=judged.clearance_mm, refused=(),
                             pairs_named=judged.pairs_named, perceived_ignored=judged.perceived_ignored)


class UnavailableTheSecondTime(SeenWorldPlanner):
    def judge_joint_path(self, samples: Any, **asked: Any) -> PathJudgement:
        if asked.get("ignore_perceived") is not None:
            raise CuroboUnavailableError("the sidecar exited")
        return super().judge_joint_path(samples, **asked)


class AMalformedSecondReportIsNoReportTests(unittest.TestCase):
    def test_every_second_report_that_is_not_one_leaves_the_refusal_standing(self) -> None:
        """⭐ FAIL CLOSED: a second judgement that does not account for the samples asked, that set aside other boxes,
        that finds the robot itself otherwise than the first, or that never came, admits nothing and moves nothing."""
        for double, said in ((ShiftedSecondReport, "does not account"), (MiscountedSecondReport, "does not account"),
                             (OtherBoxesSetAside, "other boxes"), (ItsOwnTermChangedWithTheWorld, "disagree"),
                             (UnavailableTheSecondTime, "could not judge it again")):
            with self.subTest(double=double.__name__):
                arm, planner = _cell(planner_cls=double)
                result = _line(arm, Q_ON)
                self.assertIs(result.status, MotionStatus.SELF_COLLISION_REJECTED)
                arm._conn.moveJ.assert_not_called()
                self.assertIn(said, result.message or "")
                self.assertIn("the planner's world", result.message or "")


class OffTheBandThePlannersWorldAloneRefusesTests(unittest.TestCase):
    """⭐ The ordinary case at a cell (verifier W, 2026-10-01): a pose beside a seen bin out of the cushion band. The planner
    refuses it on its world alone and types its verdict WORLD, so every decision here goes through the world term, the
    second judgement and the exact guard's last word, and nothing of the band."""

    def test_the_planner_refuses_these_poses_on_its_world_alone(self) -> None:
        """The premises: out of the band, refused on the world alone at both clearances, typed WORLD; the guard accepts."""
        arm, planner = _cell()
        self.assertIsNone(arm._preflight.gate_joint_target(JointPositions(tuple(OFF)), arm=arm))
        for config in (OFF, OFF_ON):
            self.assertFalse(planner.in_band(config))
            for clearance in (0.0, _LINE_MM):
                with self.subTest(config=config, clearance=clearance):
                    verdict = planner.check_joint_path([config], refresh=False, clearance_mm=clearance)
                    self.assertIs(getattr(verdict.refusal, "kind", None), StateRefusalKind.WORLD)
                    (row,) = planner.judge_joint_path([config], clearance_mm=clearance).refused
                    self.assertEqual((row.bound_ok, row.self_ok, row.world_ok), (True, True, False))

    def test_a_world_typed_refusal_beside_a_bin_30_40_and_47_7_mm_away_is_admitted(self) -> None:
        """⭐ A verdict the planner typed WORLD is the exact guard's where only the camera's boxes refused it: at a plan's
        clearance and at a line's, the planner asked once more with them set aside, at the clearance it refused at."""
        for real_mm in (30.0, 40.0, 47.7):
            arm, planner = _cell(real_mm)
            for clearance in (0.0, _LINE_MM):
                with self.subTest(real_mm=real_mm, clearance=clearance):
                    verdict = planner.check_joint_path([OFF], refresh=False, clearance_mm=clearance)
                    self.assertIs(getattr(verdict.refusal, "kind", None), StateRefusalKind.WORLD)
                    self.assertIsNone(arm._exact_guard_decides(planner, [OFF], verdict, clearance_mm=clearance,
                                                               command=MotionCommand.MOVE_JOINTS))
                    again = planner.set_aside()[-1]
                    self.assertEqual((again[1], again[2]), ([OFF], clearance))

    def test_a_straight_line_beside_the_bin_runs_where_only_its_world_refused_it(self) -> None:
        """_judge_legs: every sample of the line refused on the world alone, asked again, admitted; one moveJ."""
        arm, planner = _cell()
        _from(arm, planner, OFF)
        result = _line(arm, OFF_ON)
        self.assertTrue(result.ok, result.message)
        arm._conn.moveJ.assert_called_once()
        (check,) = planner.named("check")
        self.assertTrue(all(planner.in_world(c, _LINE_MM) and not planner.in_band(c) for c in check[1]))
        (again,) = planner.set_aside()
        self.assertEqual(again[1], check[1])

    def test_the_exact_guard_has_the_last_word_on_a_world_typed_refusal(self) -> None:
        """⭐ A bin 22 mm from the meshes: with the camera's boxes set aside the planner's world clears, and the exact guard,
        judging the very sample again, refuses it at 5 mm on the box it names. The robot itself was never in question."""
        arm, planner = _cell(22.0)
        verdict = planner.check_joint_path([OFF], refresh=False, clearance_mm=0.0)
        self.assertIs(getattr(verdict.refusal, "kind", None), StateRefusalKind.WORLD)
        standing = _decide(arm, planner, [OFF], 0.0)
        assert standing is not None and standing.row is not None
        self.assertTrue(standing.row.self_ok)
        for said in ("the exact guard refuses it too", "fixture:seen_", "< 5.000 mm"):
            self.assertIn(said, standing.reason)
        self.assertIs(standing.status, MotionStatus.SELF_COLLISION_REJECTED)
        self.assertEqual(len(planner.set_aside()), 1, "the guard spoke after the planner's second judgement")

    def test_what_stays_the_planners_off_the_band(self) -> None:
        """⭐ The carried part the planner models, a declared wall and the bench stay the planner's on a refusal it typed
        WORLD. Jaws closed on a part nobody models are judged as an empty hand is (the owner, 2026-10-05)."""
        for what in ("the planner holds one", "the arm holds one"):
            with self.subTest(what=what):
                arm, planner = _cell(carries_part=what == "the planner holds one")
                if what == "the arm holds one":
                    _modelled_part(arm)
                _from(arm, planner, OFF)
                result = _line(arm, OFF_ON)
                self.assertFalse(result.ok)
                arm._conn.moveJ.assert_not_called()
                self.assertIn("a part is carried", result.message or "")
                self.assertEqual(planner.set_aside(), [], "the planner was never asked with the boxes set aside")
        with self.subTest(what="the jaws closed on one nobody models"):
            arm, planner = _cell()
            self.assertFalse(arm.attach_payload(40.0), "this cell models no carried part")
            _from(arm, planner, OFF)
            self.assertTrue(_line(arm, OFF_ON).ok, "judged as an empty hand is")
            self.assertNotEqual(planner.set_aside(), [])

        wall, fixture = _declared_wall(30.0)
        self.assertAlmostEqual(_distance(_backend(_arm()), OFF, (wall,)), 30.0, delta=0.5)
        arm, planner = _cell(seen=(_FAR_PART,), world=(_BENCH, wall, _FAR_PART), fixtures=[fixture])
        _from(arm, planner, OFF)
        result = _line(arm, OFF_ON)
        self.assertFalse(result.ok)
        arm._conn.moveJ.assert_not_called()
        self.assertIn("the planner's world refuses it with the boxes the camera saw set aside too", result.message or "")
        (again,) = planner.set_aside()
        self.assertEqual(again[4], ("seen_09_part",), "the camera's own box was set aside, the declared wall never")

        raised = AxisAlignedBox(center_mm=np.array([0.0, 0.0, -5.0]), half_extents_mm=np.array([1000.0, 1000.0, 25.0]),
                                name="support_plane")
        arm, planner = _cell(world=(raised, *_seen(30.0)))
        standing = _decide(arm, planner, [OFF], 0.0)
        assert standing is not None and standing.row is not None
        self.assertTrue(standing.row.self_ok)
        self.assertIn("the bench", standing.reason)
        self.assertIn("set aside too", standing.reason)

    def test_a_line_onto_the_bin_is_judged_again_sample_by_sample(self) -> None:
        """⭐ THE POSITION MAPPING. A line from AWAY to OFF meets the bin's camera boxes only on its way, so the samples the
        planner's world refuses start well after its first. The declared lamp is met by the planner's line clearance on a
        stretch inside them: asked again with the camera's boxes set aside, the planner still refuses that stretch, and
        the refusal stands on the stretch's first sample, counted over the whole line, not over the samples asked again."""
        seen = _seen(30.0)
        arm, planner = _cell(world=(_BENCH, *seen, _LAMP),
                             fixtures=[{"name": "lamp", "center_mm": list(_LAMP_CENTRE),
                                        "half_extents_mm": [10.0, 10.0, 30.0]}])
        _from(arm, planner, AWAY)
        result = _line(arm, OFF)
        self.assertFalse(result.ok)
        arm._conn.moveJ.assert_not_called()
        (check,) = planner.named("check")
        configs = check[1]
        refused = [i for i, config in enumerate(configs) if planner.in_world(config, _LINE_MM)]
        planner.aside = tuple(box.name for box in seen)
        by_the_lamp = [i for i in refused if planner.in_world(configs[i], _LINE_MM)]
        planner.aside = ()
        self.assertGreater(refused[0], 0, "the line starts clear of the planner's world")
        self.assertGreater(by_the_lamp[0], refused[0], "the lamp is met inside the stretch, after its first sample")
        self.assertTrue(all(not planner.in_band(config) for config in configs), "the robot itself is never in question")
        (again,) = planner.set_aside()
        self.assertEqual(again[1], [configs[i] for i in refused], "every sample the world refused, and only those")
        self.assertIn(f"at sample {by_the_lamp[0]} of {len(configs)}", result.message or "")
        self.assertIn("the planner's world refuses it with the boxes the camera saw set aside too", result.message or "")


class TheScreenSaysWhatAMoveDoesBesideTheBinTests(unittest.TestCase):
    """⭐ The screen at teaching, at the desk start and at every campaign's start asks the planner what a move asks it
    (verifier W, 2026-10-01): a pose only the camera's boxes refuse is no ERROR, because a move goes there, and the line
    says what runs into it and out of it."""

    def test_a_pose_only_the_cameras_boxes_refuse_is_said_as_such_and_a_move_goes_there(self) -> None:
        count = len(_seen(30.0))
        for name, joints, start in (("off the band", OFF, OFF_ON), ("in the band too", Q, Q_ON)):
            with self.subTest(pose=name):
                arm, planner = _cell()
                screen = arm.screen_configuration(JointPositions(tuple(joints)))
                self.assertIs(screen.verdict, PoseVerdict.SEEN_BOXES)
                self.assertFalse(screen.is_error)
                line = screen.line("BESIDE_BIN")
                self.assertTrue(line.startswith("BESIDE_BIN: beside the boxes the camera saw: "), line)
                for said in (f"the {count} box(es) the camera saw", "5 mm", "Straight lines", "a planned move",
                             "carries a part"):
                    self.assertIn(said, screen.detail)
                if joints is Q:
                    self.assertIn("forearm_link and wrist_2_link", screen.detail, "the band is said beside it")
                (again,) = planner.set_aside()
                self.assertEqual((len(again[1]), again[2]), (1, 0.0))
                arm._conn.moveJ.assert_not_called()
                _from(arm, planner, start)
                self.assertTrue(_line(arm, joints).ok, "the move the screen promises runs")

    def test_what_the_screen_still_calls_an_error_beside_the_bin(self) -> None:
        """Where no move goes, the screen still says ERROR and why: a part carried, a declared wall the planner meets with
        the camera's boxes set aside, the bench under the housing, a bin the exact guard itself refuses."""
        arm, planner = _cell(carries_part=True)
        screen = arm.screen_configuration(JointPositions(tuple(OFF)))
        self.assertIs(screen.verdict, PoseVerdict.PLANNER_REFUSED)
        self.assertTrue(screen.line("BESIDE_BIN").startswith("ERROR BESIDE_BIN: refused by the planner"))
        self.assertIn("a part is carried", screen.detail)
        self.assertEqual(planner.set_aside(), [])

        wall, fixture = _declared_wall(20.0)
        arm, planner = _cell(seen=(_FAR_PART,), world=(_BENCH, wall, _FAR_PART), fixtures=[fixture])
        screen = arm.screen_configuration(JointPositions(tuple(OFF)))
        self.assertIs(screen.verdict, PoseVerdict.PLANNER_REFUSED)
        self.assertIn("set aside too", screen.detail)

        raised = AxisAlignedBox(center_mm=np.array([0.0, 0.0, -5.0]), half_extents_mm=np.array([1000.0, 1000.0, 25.0]),
                                name="support_plane")
        arm, planner = _cell(world=(raised, *_seen(30.0)))
        screen = arm.screen_configuration(JointPositions(tuple(OFF)))
        self.assertIs(screen.verdict, PoseVerdict.PLANNER_REFUSED)
        self.assertIn("the bench", screen.detail)

        arm, planner = _cell(22.0)
        screen = arm.screen_configuration(JointPositions(tuple(OFF)))
        self.assertIs(screen.verdict, PoseVerdict.GUARD_REFUSED)
        self.assertTrue(screen.is_error)


def _modelled_part(arm: Any) -> None:
    """The jaws closed on a part the planner models, as a grasp's attach on a cell that models one leaves the arm."""
    arm._closed_on_part = True
    arm._attached_payload = (120.0, 5.0)


class AnArmThatClosesOnAPartBesideTheBinIsHeldThereTests(unittest.TestCase):
    """⭐ What the owner has to know (verifier W, 2026-10-01, measured on the GPU with a real attach): the camera's boxes
    are set aside only for an empty hand. An arm admitted beside the bin empty-handed that closes on a part the planner
    models is held there: every way out starts at the pose the planner's world refuses, and nothing is set aside while
    that part is carried. Nothing moves; the refusal says how a person gets it out. A grasp therefore judges its lift
    carrying before it closes and does not close there (the owner, 2026-10-01:
    ``test_a_grasp_judges_its_lift_carrying_before_it_closes``); these hold the arm that closed there all the same.

    A part the cell models nowhere is judged as an empty hand is (the owner, 2026-10-05: "mehr Griffe"): nobody judges it
    against the camera's boxes either way, and the exact guard still judges the arm and the hand."""

    def test_a_part_the_cell_does_not_model_is_judged_as_an_empty_hand_is(self) -> None:
        """⭐ A cell whose planning_world.payload models no part (here no length is declared) closes its jaws on one, and
        the line beside the bin runs as it runs empty-handed: the boxes set aside, the exact guard deciding."""
        arm, planner = _cell()
        self.assertIsNotNone(arm.payload_declined_reason())
        self.assertFalse(arm.attach_payload(40.0))
        self.assertTrue(arm._closed_on_part, "the arm still knows its jaws closed on a part")
        self.assertIs(arm.payload_model(), PayloadModel.NONE)
        self.assertFalse(planner.carries_part, "the planner was handed nothing")
        result = _line(arm, Q_ON)
        self.assertTrue(result.ok, result.message)
        self.assertNotEqual(planner.set_aside(), [], "the camera's boxes were set aside, as for an empty hand")

    def test_an_empty_hand_goes_in_and_a_carried_part_does_not_come_out_on_its_own(self) -> None:
        arm, planner = _cell()
        _from(arm, planner, AWAY)
        self.assertTrue(_line(arm, OFF).ok, "empty-handed, the line into the pose beside the bin runs")
        _from(arm, planner, OFF)
        _modelled_part(arm)  # what a grasp's attach leaves on a cell that models the part
        before = arm._conn.moveJ.call_count
        for target in (AWAY, OFF_ON):
            with self.subTest(target=target):
                result = _line(arm, target)
                self.assertFalse(result.ok)
                for said in ("at sample 0 of", "the planner's world", "a part is carried", "Robot.release"):
                    self.assertIn(said, result.message or "")
        self.assertEqual(arm._conn.moveJ.call_count, before, "nothing moved")
        self.assertTrue(arm.detach_payload(), "what a release does once the jaws opened")
        self.assertTrue(_line(arm, AWAY).ok, "empty-handed again, it leaves")

    def test_the_tool_line_out_is_held_the_same_way(self) -> None:
        """The retreat after a close is a moveL: carrying, it is refused at its first sample as every line out is."""
        arm, planner = _cell()
        arm.ik = lambda pose, *, seed=None: JointPositions(tuple(Q))
        arm._motion = MagicMock()
        arm._motion.move_to.return_value = True
        flange = np.asarray(ur_link_transforms_mm("ur10", np.asarray(Q))[-1])
        arm._motion.get_current_pose.return_value = pose_to_urpose(Pose.from_matrix(flange, frame=Frame.BASE))
        _modelled_part(arm)
        start = arm.get_tcp_pose()
        with arm.without_camera_world(_DECLINED):
            result = arm.move(start, linear=True)
        self.assertFalse(result.ok)
        arm._motion.move_to.assert_not_called()
        for said in ("at sample 0 of", "a part is carried", "Robot.release"):
            self.assertIn(said, result.message or "")


if __name__ == "__main__":
    unittest.main()
