"""The camera world lets the owner's pick finish, and still refuses what nobody saw (review of 2026-09-23).

The owner's cell: a UR10, a Robotiq Hand-E, and an Intel D415 on the wrist beside the gripper, tilted about 45
degrees outward. A review of the camera world (scratchpad review4, lens "world") found three ways it stopped the
cell or let obstacles go:

  * the refresh for the line back up out of a grasp judged the whole frame. At the grasp the camera stands about
    150 mm over the bench, closer than a D415 measures, so 27 % of the frame held a depth at 848 x 480: BLIND, and
    the retreat raised with the part clamped. The hand's line is out of that camera's view at every pose of the pick;
  * the band round a declared fixture was the bench band, which robot.yaml tells a cell to raise by the sink of its
    slab, so a tote declared on a cell with its slab sunk 50 mm took every part within 60 mm of it;
  * every wrist frame declared its rig's shutter-motion refusal threshold as its placement error although the arm
    stood still, and the bench band grew from 5 mm to 10-16 mm: a plate 8 mm tall left the world.

The frames are rendered through a pinhole with the D415's field of view, 320 x 180, cut at its minimum range
(Intel's datasheet: about 310 mm at 848 x 480 and 450 mm at 1280 x 720, not measured here).
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.safety.planning.depth_source import RigDepthSource
from src.robot.safety.planning.live_world import (
    CameraView,
    DepthSnapshot,
    LivePlannerWorld,
    WorldVerdict,
    refresh_planner_world,
)
from src.robot.safety.planning.perceived import LinkCapsule, SelfEnvelope, WorldBuildLimits, WorldBuildTuning
from src.robot.safety.planning.world import planner_cuboid

_W, _H, _F = 320, 180, 925.0 / 4.0
_K = np.array([[_F, 0.0, _W / 2.0], [0.0, _F, _H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
_MIN_RANGE_MM = {"848 x 480": 310.0, "1280 x 720": 450.0}
#: The UR10's box, with z opened below the bench: on the cell the body's reach keeps these points.
_LIMITS = WorldBuildLimits(x_mm=(-760.7, 760.7), y_mm=(-760.7, 760.7), z_mm=(-100.0, 748.4), support_plane_top_mm=0.0)
_BENCH = (planner_cuboid("support_plane", (0.0, 0.0, -25.0), (3000.0, 3000.0, 50.0)),)
#: A body standing out of every scene here: one capsule at the base.
_SELF = SelfEnvelope(
    frames_mm=(np.eye(4),),
    capsules=(LinkCapsule(frame=0, start_mm=(0.0, 0.0, 0.0), end_mm=(0.0, 0.0, 100.0), radius_mm=50.0),),
)
#: The part the pick closes on: a 40 mm cube on the bench.
_CUBE = ((-20.0, -720.0, 0.0), (20.0, -680.0, 40.0))
_GRASP = Pose.tool_down(0.0, -700.0, 20.0, closing_axis="-y")
_STANDOFF = Pose.tool_down(0.0, -700.0, 100.0, closing_axis="-y")


def _mount() -> np.ndarray:
    """CAMERA to TOOL: 60 mm along the tool's +x, 130 mm behind the TCP, turned 45 degrees toward +x (K3's stand-in)."""
    half = math.sqrt(0.5)
    mount = np.eye(4)
    mount[:3, :3] = [[half, 0.0, half], [0.0, 1.0, 0.0], [-half, 0.0, half]]
    mount[:3, 3] = (60.0, 0.0, -130.0)
    return mount


def _look(position_mm: tuple[float, float, float], direction: tuple[float, float, float]) -> np.ndarray:
    """CAMERA to BASE for a camera at ``position_mm`` looking along ``direction``, image x along base x."""
    z = np.asarray(direction, dtype=np.float64) / np.linalg.norm(direction)
    x = np.array([1.0, 0.0, 0.0]) - z * z[0]
    x /= np.linalg.norm(x)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    transform[:3, 3] = position_mm
    return transform


def _render(camera_to_base: np.ndarray, min_range_mm: float, *boxes: tuple) -> np.ndarray:
    """The depth of the bench and ``boxes`` along each ray, 0 where it is nearer than ``min_range_mm`` or meets nothing."""
    cols, rows = np.meshgrid(np.arange(_W, dtype=np.float64), np.arange(_H, dtype=np.float64))
    rays = np.stack([(cols - _K[0, 2]) / _F, (rows - _K[1, 2]) / _F, np.ones_like(cols)], axis=-1)
    rays = rays @ camera_to_base[:3, :3].T
    origin = camera_to_base[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = -origin[2] / rays[..., 2]
        depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, np.inf)
        for low, high in boxes:
            t_low, t_high = (np.asarray(low) - origin) / rays, (np.asarray(high) - origin) / rays
            near = np.nanmax(np.minimum(t_low, t_high), axis=-1)
            far = np.nanmin(np.maximum(t_low, t_high), axis=-1)
            depth = np.minimum(depth, np.where((far >= near) & (near > 0.0), near, np.inf))
    return np.where(np.isfinite(depth) & (depth >= min_range_mm), depth, 0.0)


def _covered(boxes: tuple, point: tuple[float, float, float]) -> list[str]:
    """The perceived boxes that hold ``point``."""
    names = []
    for box in boxes:
        c, s = math.cos(box.yaw_rad), math.sin(box.yaw_rad)
        d = np.asarray(point, dtype=np.float64) - np.asarray(box.center_mm, dtype=np.float64)
        local = (c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2])
        if all(abs(v) <= h / 2.0 for v, h in zip(local, box.dims_mm)):
            names.append(box.name)
    return names


class _Wrist:
    """The wrist D415 at one tool pose, counting how often it is asked."""

    def __init__(self, tool: Pose, depth: np.ndarray) -> None:
        self.tool, self.depth, self.grabs = tool.to_matrix(), depth, 0

    def grab_surface_depth(self) -> DepthSnapshot:
        self.grabs += 1
        return DepthSnapshot(depth_mm=self.depth, intrinsics=_K, timestamp=100.0, tool_to_base_mm=self.tool)


class _Planner:
    def set_world(self, cuboids, meshes=None):  # noqa: ANN001, ANN201
        return len(cuboids)


def _wrist_world(tool: Pose, min_range_mm: float, depth: np.ndarray | None = None) -> tuple[LivePlannerWorld, _Wrist]:
    camera = _Wrist(tool, _render(tool.to_matrix() @ _mount(), min_range_mm, _CUBE) if depth is None else depth)
    world = LivePlannerWorld(
        cameras=(CameraView(name="wrist", depth_source=camera, camera_to_tool=_mount()),),
        declared=_BENCH, limits=_LIMITS, tuning=WorldBuildTuning(max_boxes=8),
    )
    return world, camera


def _refresh(world: LivePlannerWorld, goal: Pose | tuple[float, float, float]):  # noqa: ANN202
    point = goal.position_mm if isinstance(goal, Pose) else goal
    return refresh_planner_world(
        source=world, client=_Planner(), self_envelope=_SELF, near_point_mm=[float(v) for v in point], now=100.05,
    )


# ---------------------------------------------------------------------------------------------------
# The pick's lines: judged where they go, not by the whole frame
# ---------------------------------------------------------------------------------------------------


class ThePicksLinesArePlannedTests(unittest.TestCase):
    """At the grasp the retreat raised CameraWorldUnavailable on 27 % of the frame, and nothing got the Hand-E out."""

    def test_the_line_back_up_out_of_the_grasp_is_planned_and_its_stamp_says_what_was_not_seen(self) -> None:
        for mode, min_range in _MIN_RANGE_MM.items():
            with self.subTest(mode):
                world, _ = _wrist_world(_GRASP, min_range)

                refresh = _refresh(world, _STANDOFF)

                self.assertTrue(refresh.ok, refresh.render())
                self.assertEqual((), refresh.goal_seen_by, "the camera looks outward, away from the hand's line")
                stamp = refresh.camera_world()
                assert stamp is not None and stamp.unseen is not None
                self.assertTrue(stamp.unseen.goal_out_of_view)
                ((camera, holes),) = stamp.unseen.no_depth
                self.assertEqual("wrist", camera)
                self.assertGreater(holes, 0.5, "most of the frame is nearer than the camera measures, and the stamp says so")

    def test_the_line_down_from_the_standoff_is_planned(self) -> None:
        for mode, min_range in _MIN_RANGE_MM.items():
            with self.subTest(mode):
                world, _ = _wrist_world(_STANDOFF, min_range)

                refresh = _refresh(world, _GRASP)

                self.assertTrue(refresh.ok, refresh.render())
                self.assertEqual((), refresh.goal_seen_by)


class AWristFrameStillRefusesWhatItDidNotSeeTests(unittest.TestCase):
    """The controls: a frame that saw nothing, a goal in the holes, and a frame asked about no motion."""

    def test_a_frame_with_no_depth_at_the_grasp_raises_after_its_attempts(self) -> None:
        world, camera = _wrist_world(_GRASP, 310.0, depth=np.zeros((_H, _W)))

        with self.assertRaises(CameraWorldUnavailable) as caught:
            _refresh(world, _STANDOFF)

        self.assertIs(WorldVerdict.BLIND, caught.exception.verdict)
        self.assertEqual(1 + world.fresh_frame_attempts, camera.grabs)

    def test_a_goal_the_camera_looks_at_inside_its_minimum_range_is_unseen(self) -> None:
        world, camera = _wrist_world(_GRASP, 310.0)
        placed = _GRASP.to_matrix() @ _mount()
        goal = tuple(float(v) for v in placed[:3, 3] + 200.0 * placed[:3, 2])

        refresh = _refresh(world, goal)

        self.assertFalse(refresh.ok)
        self.assertIs(WorldVerdict.UNSEEN, refresh.verdict, refresh.render())
        self.assertIsNone(refresh.camera_world())
        self.assertEqual(1, camera.grabs, "where the camera stands is not a camera fault: refused, not asked again")

    def test_asked_about_no_motion_the_whole_frame_is_judged(self) -> None:
        world, _ = _wrist_world(_GRASP, 310.0)

        snapshot = world.world_for(self_envelope=_SELF, now=100.05)

        self.assertIs(WorldVerdict.BLIND, snapshot.verdict, snapshot.render())
        self.assertIn("27% of the pixels hold a depth", snapshot.reason)

    def test_a_fixed_camera_mostly_holes_is_blind_whatever_the_goal(self) -> None:
        """A fixed camera is never carried toward what it sees: a frame of it mostly holes is a camera fault."""
        overhead = _look((0.0, -300.0, 1000.0), (0.0, 0.0, -1.0))
        depth = _render(overhead, 310.0)
        depth[:, int(0.3 * _W):] = 0.0
        camera = _Wrist(_GRASP, depth)
        world = LivePlannerWorld(
            cameras=(CameraView(name="overhead", depth_source=camera, camera_to_base=overhead),),
            declared=_BENCH, limits=_LIMITS, tuning=WorldBuildTuning(max_boxes=8),
        )

        snapshot = world.world_for(self_envelope=_SELF, near_point_mm=(-500.0, -300.0, 100.0), now=100.05)

        self.assertIs(WorldVerdict.BLIND, snapshot.verdict, snapshot.render())
        self.assertIn("30% of the pixels hold a depth", snapshot.reason)


# ---------------------------------------------------------------------------------------------------
# A declared fixture keeps its own band, whatever the slab's clearance
# ---------------------------------------------------------------------------------------------------


class ADeclaredToteOnASunkSlabTests(unittest.TestCase):
    """With the slab sunk 50 mm and the clearance raised to 55, a declared tote took every part within 60 mm of it."""

    #: A tote on the bench: a floor 10 mm thick and the wall on its +y side, 120 mm tall.
    _FLOOR = ((-150.0, -750.0, 0.0), (150.0, -550.0, 10.0))
    _WALL = ((-150.0, -560.0, 0.0), (150.0, -550.0, 120.0))
    #: A 40 mm cube on the tote floor, 25 mm from the wall.
    _PART = ((-20.0, -625.0, 10.0), (20.0, -585.0, 50.0))

    def _boxes(self, slab_top_mm: float, clearance_mm: float) -> tuple:
        def cuboid(name: str, box: tuple) -> dict:
            low, high = np.asarray(box[0]), np.asarray(box[1])
            return planner_cuboid(name, tuple((low + high) / 2.0), tuple(high - low))

        camera_to_base = _look((0.0, -650.0, 700.0), (0.0, 0.0, -1.0))
        source = _Wrist(_GRASP, _render(camera_to_base, 310.0, self._FLOOR, self._WALL, self._PART))
        slab = planner_cuboid("support_plane", (0.0, 0.0, slab_top_mm - 25.0), (3000.0, 3000.0, 50.0))
        limits = WorldBuildLimits(x_mm=_LIMITS.x_mm, y_mm=_LIMITS.y_mm, z_mm=_LIMITS.z_mm,
                                  support_plane_top_mm=slab_top_mm, plane_clearance_mm=clearance_mm)
        world = LivePlannerWorld(
            cameras=(CameraView(name="overhead", depth_source=source, camera_to_base=camera_to_base),),
            declared=(slab, cuboid("tote_floor", self._FLOOR), cuboid("tote_wall", self._WALL)),
            limits=limits, tuning=WorldBuildTuning(max_boxes=8),
        )
        snapshot = world.world_for(self_envelope=_SELF, now=100.05)
        self.assertIs(WorldVerdict.FRESH, snapshot.verdict, snapshot.render())
        assert snapshot.perceived is not None
        return snapshot.perceived.boxes

    def test_the_part_beside_the_declared_tote_stays_an_obstacle_on_a_sunk_slab(self) -> None:
        for slab, clearance in ((0.0, 5.0), (-50.0, 55.0)):
            with self.subTest(slab_top_mm=slab, plane_clearance_mm=clearance):
                boxes = self._boxes(slab, clearance)

                self.assertTrue(_covered(boxes, (0.0, -605.0, 30.0)), [(b.name, b.center_mm, b.dims_mm) for b in boxes])
                self.assertFalse(_covered(boxes, (100.0, -700.0, 30.0)), "the tote's hollow stays hollow")


# ---------------------------------------------------------------------------------------------------
# A still wrist frame declares the motion it measured, not the rig's refusal threshold
# ---------------------------------------------------------------------------------------------------


class _Rig:
    """A wrist rig handle: the same depth on every grab."""

    rig_id = "wrist"

    def __init__(self, depth: np.ndarray) -> None:
        self.depth = depth

    def grab(self) -> SimpleNamespace:
        return SimpleNamespace(depth=self.depth)

    def get_intrinsics(self) -> np.ndarray:
        return _K


class AStillWristFrameKeepsALowPartTests(unittest.TestCase):
    """The band grew by a 1 mm / 0.5 degree threshold the still arm never moved, and an 8 mm plate left the world."""

    def test_a_plate_8_mm_tall_beside_the_target_stays_an_obstacle(self) -> None:
        camera_to_base = _look((0.0, -346.4, 353.6), (0.0, -1.0, -1.0))
        tool = Pose.from_matrix(camera_to_base @ np.linalg.inv(_mount()), frame=Frame.BASE)
        plate = ((-60.0, -760.0, 0.0), (60.0, -680.0, 8.0))
        source = RigDepthSource(_Rig(_render(camera_to_base, 310.0, plate)), tool_pose=lambda: tool,
                                motion_tolerance=(1.0, 0.5))
        world = LivePlannerWorld(
            cameras=(CameraView(name="wrist", depth_source=source, camera_to_tool=_mount()),),
            declared=_BENCH, limits=_LIMITS, tuning=WorldBuildTuning(max_boxes=8), max_age_ms=1e9,
        )

        snapshot = world.world_for(self_envelope=_SELF)

        self.assertIs(WorldVerdict.FRESH, snapshot.verdict, snapshot.render())
        assert snapshot.perceived is not None
        low, high = snapshot.perceived.bench_band_mm["wrist"]
        self.assertAlmostEqual(5.0, low, places=6)
        self.assertAlmostEqual(5.0, high, places=6)
        self.assertTrue(_covered(snapshot.perceived.boxes, (0.0, -720.0, 4.0)), snapshot.render())


if __name__ == "__main__":
    unittest.main()
