"""The camera world vouches for what its cameras measured, and says what they did not.

The owner's cell, 2026-09-23: a UR10 with a Robotiq Hand-E and an Intel D415 on the wrist, tilted about 45
degrees forward. An audit of that cell's planning world (scratchpad audit3, lens "world") found five ways the
world handed the planner space nobody saw, or refused space everybody could see:

  * a pixel with no depth became free space and was never counted, and a frame counted as seen when a single
    pixel held a depth;
  * a frame whose near field was blind while its far half held depth vouched PLANNED for where the hand goes;
  * the bench came back as one obstacle as soon as the camera's placement was off by more than the fixed 5 mm
    plane clearance, which a wrist camera's own declared shutter motion already allows;
  * obstacles were culled to the TCP's workspace box, although the links and the D415 swing past it;
  * the slab sunk below the bench, as robot.yaml recommended for low picks, turned the bench into one obstacle.

Every scene here is rendered through the pinhole the converter inverts, so the geometry is known in
millimetres. Names that are new with this change are imported inside the tests, so the file loads against the
tree before it and each test fails there on its own assertion or its own missing name.
"""

from __future__ import annotations

import json
import math
import unittest

import numpy as np

from src.robot.core.camera_world import CameraWorldStamp, CameraWorldUse
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.safety.planning.live_world import (
    CameraView,
    DepthSnapshot,
    LivePlannerWorld,
    WorldVerdict,
    refresh_planner_world,
)
from src.robot.safety.planning.perceived import (
    DepthView,
    LinkCapsule,
    PerceptionGeometryError,
    SelfEnvelope,
    WorldBuildLimits,
    WorldBuildTuning,
    build_perceived_boxes,
)
from src.robot.safety.planning.world import planner_cuboid

#: A small pinhole, so a test renders its own frames in a few milliseconds.
_W, _H, _F = 160, 120, 140.0
_K = np.array([[_F, 0.0, _W / 2.0], [0.0, _F, _H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
#: The UR10 profile's TCP box (config/robot/robot.ur10.yaml), with the bench at z 0.
_UR10_LIMITS = WorldBuildLimits(
    x_mm=(-760.7, 760.7), y_mm=(-760.7, 760.7), z_mm=(100.0, 748.4), support_plane_top_mm=0.0,
)
_LIMITS = WorldBuildLimits(x_mm=(-760.0, 760.0), y_mm=(-760.0, 760.0), z_mm=(-100.0, 900.0), support_plane_top_mm=0.0)
_BENCH = (planner_cuboid("support_plane", (0.0, 0.0, -25.0), (3000.0, 3000.0, 50.0)),)
#: A body standing out of every scene here: one capsule at the base.
_SELF = SelfEnvelope(
    frames_mm=(np.eye(4),),
    capsules=(LinkCapsule(frame=0, start_mm=(0.0, 0.0, 0.0), end_mm=(0.0, 0.0, 100.0), radius_mm=50.0),),
)


# ---------------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------------


def _look(position_mm: tuple[float, float, float], direction: tuple[float, float, float]) -> np.ndarray:
    """CAMERA to BASE for a camera at ``position_mm`` looking along ``direction``, image x along base x."""
    z = np.asarray(direction, dtype=np.float64)
    z /= np.linalg.norm(z)
    x = np.array([1.0, 0.0, 0.0]) - z * z[0]
    x /= np.linalg.norm(x)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    transform[:3, 3] = position_mm
    return transform


#: Straight down from a metre up: base x is image x, and a pixel at 1000 mm is the bench.
_OVERHEAD = _look((0.0, 0.0, 1000.0), (0.0, 0.0, -1.0))


def _rays(camera_to_base: np.ndarray) -> np.ndarray:
    """Per pixel, the BASE direction of its ray, scaled so its camera z is 1."""
    cols, rows = np.meshgrid(np.arange(_W, dtype=np.float64), np.arange(_H, dtype=np.float64))
    camera = np.stack([(cols - _K[0, 2]) / _F, (rows - _K[1, 2]) / _F, np.ones_like(cols)], axis=-1)
    return camera @ camera_to_base[:3, :3].T


def _plane(camera_to_base: np.ndarray, z_mm: float = 0.0) -> np.ndarray:
    """The depth of a horizontal plane at ``z_mm``, 0 where a ray never meets it."""
    rays = _rays(camera_to_base)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (z_mm - camera_to_base[2, 3]) / rays[..., 2]
    return np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0)


def _box(
    camera_to_base: np.ndarray, low_mm: tuple[float, float, float], high_mm: tuple[float, float, float],
) -> np.ndarray:
    """The depth of an axis-aligned box's nearest face along each ray, 0 where a ray misses it (slab method)."""
    rays = _rays(camera_to_base)
    origin = camera_to_base[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_low = (np.asarray(low_mm) - origin) / rays
        t_high = (np.asarray(high_mm) - origin) / rays
    near = np.nanmax(np.minimum(t_low, t_high), axis=-1)
    far = np.nanmin(np.maximum(t_low, t_high), axis=-1)
    return np.where((far >= near) & (near > 0.0), near, 0.0)


def _nearest(*depths: np.ndarray) -> np.ndarray:
    """What a camera measures where several surfaces stand: the nearest one each ray meets."""
    stack = np.stack([np.where(d > 0.0, d, np.inf) for d in depths])
    nearest = stack.min(axis=0)
    return np.where(np.isfinite(nearest), nearest, 0.0)


def _turned(camera_to_base: np.ndarray, degrees: float) -> np.ndarray:
    """The same camera turned about its own x axis: a placement off by ``degrees``."""
    angle = math.radians(degrees)
    turn = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(angle), -math.sin(angle)], [0.0, math.sin(angle), math.cos(angle)]])
    out = camera_to_base.copy()
    out[:3, :3] = camera_to_base[:3, :3] @ turn
    return out


class _Camera:
    """A depth source under the test's control, counting how often it is asked."""

    def __init__(self, snapshot: DepthSnapshot) -> None:
        self.snapshot = snapshot
        self.grabs = 0

    def grab_surface_depth(self) -> DepthSnapshot:
        self.grabs += 1
        return self.snapshot


class _Planner:
    """A planner client that confirms whatever it is sent."""

    def set_world(self, cuboids, meshes=None):  # noqa: ANN001, ANN201
        return len(cuboids)


def _overhead_world(depth: np.ndarray, **kwargs: object) -> tuple[LivePlannerWorld, _Camera]:
    camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_K, timestamp=100.0))
    settings: dict[str, object] = {
        "cameras": (CameraView(name="overhead", depth_source=camera, camera_to_base=_OVERHEAD),),
        "declared": _BENCH, "limits": _LIMITS, "tuning": WorldBuildTuning(max_boxes=4), "max_age_ms": 500.0,
    }
    settings.update(kwargs)
    return LivePlannerWorld(**settings), camera  # type: ignore[arg-type]


def _holes_right_of(depth: np.ndarray, share: float) -> np.ndarray:
    """The frame with its columns from ``1 - share`` of the width onward holding no depth."""
    out = depth.copy()
    out[:, int(round(_W * (1.0 - share))):] = 0.0
    return out


# ---------------------------------------------------------------------------------------------------
# Holes: counted, reported, and a frame of them is not a world
# ---------------------------------------------------------------------------------------------------


class AHoleIsCountedTests(unittest.TestCase):
    """perceived.py dropped a pixel with no depth through `valid = keep & isfinite & depth > 0`, with no reason."""

    def _scene(self) -> np.ndarray:
        # A block 80 mm tall on the left half of the image.
        return _nearest(_plane(_OVERHEAD), _box(_OVERHEAD, (-190.0, -30.0, 0.0), (-110.0, 30.0, 80.0)))

    def test_the_control_a_block_on_a_full_frame_is_one_box(self) -> None:
        world = build_perceived_boxes(views=(DepthView(self._scene(), _K, _OVERHEAD, name="overhead"),), limits=_LIMITS)

        self.assertEqual(1, len(world.boxes), world.render())
        self.assertNotIn("no_depth", world.dropped_points)

    def test_a_block_in_the_holes_is_lost_and_the_holes_are_counted_by_name(self) -> None:
        depth = self._scene()
        depth[:, : _W // 2] = 0.0

        world = build_perceived_boxes(views=(DepthView(depth, _K, _OVERHEAD, name="overhead"),), limits=_LIMITS)

        self.assertEqual((), world.boxes, "the block stood in the holes and nothing measured it")
        self.assertEqual(_H * _W // 2, world.dropped_points.get("no_depth"),
                         "every pixel with no depth is counted, in pixels of the full frame")

    def test_each_views_share_of_depth_is_reported_and_rendered(self) -> None:
        depth = self._scene()
        depth[:, : _W // 2] = 0.0

        world = build_perceived_boxes(views=(DepthView(depth, _K, _OVERHEAD, name="overhead"),), limits=_LIMITS)

        self.assertEqual({"overhead": 0.5}, world.depth_coverage)
        self.assertIn("overhead: 50% of the image held no depth", world.render())
        self.assertEqual({"overhead": 0.5}, json.loads(json.dumps(world.to_dict()))["depth_coverage"])

    def test_a_masked_pixel_is_left_out_not_a_hole(self) -> None:
        depth = self._scene()
        depth[:, : _W // 2] = 0.0
        masked = np.zeros(depth.shape, dtype=bool)
        masked[:, : _W // 2] = True

        world = build_perceived_boxes(
            views=(DepthView(depth, _K, _OVERHEAD, exclude_masks=(masked,), name="overhead"),), limits=_LIMITS,
        )

        self.assertNotIn("no_depth", world.dropped_points)
        self.assertEqual({"overhead": 1.0}, world.depth_coverage)

    def test_the_count_ignores_the_stride(self) -> None:
        depth = self._scene()
        depth[:, : _W // 2] = 0.0

        world = build_perceived_boxes(
            views=(DepthView(depth, _K, _OVERHEAD, name="overhead"),), limits=_LIMITS,
            tuning=WorldBuildTuning(pixel_stride=3),
        )

        self.assertEqual(_H * _W // 2, world.dropped_points.get("no_depth"))


class AFrameOfHolesIsBlindTests(unittest.TestCase):
    """live_world judged a frame blind only when not one pixel held a depth: 19 % counted as FRESH."""

    def test_a_frame_mostly_holes_is_blind_and_says_how_much(self) -> None:
        world, _ = _overhead_world(_holes_right_of(_plane(_OVERHEAD), 0.7))

        snapshot = world.world_for(self_envelope=_SELF, now=100.1)

        self.assertIs(WorldVerdict.BLIND, snapshot.verdict, snapshot.render())
        self.assertEqual("overhead", snapshot.camera)
        self.assertIn("30% of the pixels hold a depth", snapshot.reason)
        self.assertIn("at least 50%", snapshot.reason)

    def test_a_refresh_on_a_frame_mostly_holes_raises_after_its_attempts(self) -> None:
        world, camera = _overhead_world(_holes_right_of(_plane(_OVERHEAD), 0.7), fresh_frame_attempts=2)

        with self.assertRaises(CameraWorldUnavailable) as caught:
            refresh_planner_world(source=world, client=_Planner(), self_envelope=_SELF, now=100.1)

        self.assertIs(WorldVerdict.BLIND, caught.exception.verdict)
        self.assertEqual(3, camera.grabs, "a blind frame is asked again, and never served from the cache")

    def test_the_control_a_frame_with_holes_below_the_limit_is_fresh_and_its_stamp_says_so(self) -> None:
        world, _ = _overhead_world(_holes_right_of(_plane(_OVERHEAD), 0.4))

        refresh = refresh_planner_world(source=world, client=_Planner(), self_envelope=_SELF, now=100.1)

        self.assertTrue(refresh.ok, refresh.render())
        stamp = refresh.camera_world()
        assert stamp is not None and stamp.unseen is not None
        ((camera, share),) = stamp.unseen.no_depth
        self.assertEqual("overhead", camera)
        self.assertAlmostEqual(0.4, share, places=6)
        self.assertIn("not seen, planned as free: no depth on 40% of overhead", stamp.render())

    def test_the_limit_is_a_share_above_zero(self) -> None:
        for bad in (0.0, -0.1, 1.5):
            with self.subTest(bad), self.assertRaises(PerceptionGeometryError):
                _overhead_world(_plane(_OVERHEAD), min_depth_coverage=bad)


# ---------------------------------------------------------------------------------------------------
# The goal: where the hand goes has to have been measured by a camera that looks at it
# ---------------------------------------------------------------------------------------------------


class TheGoalHasToBeSeenTests(unittest.TestCase):
    """A frame whose near field was blind while its far half held depth vouched PLANNED for where the hand goes."""

    #: Holes on the right 40 % of the image, x > about 180 mm on the bench: below the frame limit.
    @staticmethod
    def _frame() -> np.ndarray:
        return _holes_right_of(_plane(_OVERHEAD), 0.4)

    def test_a_goal_in_the_holes_is_unseen_and_names_the_camera(self) -> None:
        world, _ = _overhead_world(self._frame())

        snapshot = world.world_for(self_envelope=_SELF, near_point_mm=(250.0, 0.0, 100.0), now=100.1)

        self.assertIs(WorldVerdict.UNSEEN, snapshot.verdict, snapshot.render())
        self.assertFalse(snapshot.usable)
        self.assertEqual("overhead", snapshot.camera)
        self.assertIn("hold a depth on 'overhead' 0% of the 200 mm about it", snapshot.reason)
        self.assertEqual(["support_plane"], [box["name"] for box in snapshot.cuboids])

    def test_one_camera_that_measured_the_goal_is_enough(self) -> None:
        """The views are fused: a second camera's holes about the goal take nothing from what the first measured."""
        holes = _Camera(DepthSnapshot(depth_mm=self._frame(), intrinsics=_K, timestamp=100.0))
        whole = _Camera(DepthSnapshot(depth_mm=_plane(_OVERHEAD), intrinsics=_K, timestamp=100.0))
        world = LivePlannerWorld(
            cameras=(CameraView(name="close", depth_source=holes, camera_to_base=_OVERHEAD),
                     CameraView(name="far", depth_source=whole, camera_to_base=_OVERHEAD)),
            declared=_BENCH, limits=_LIMITS, tuning=WorldBuildTuning(max_boxes=4),
        )

        snapshot = world.world_for(self_envelope=_SELF, near_point_mm=(250.0, 0.0, 100.0), now=100.1)

        self.assertIs(WorldVerdict.FRESH, snapshot.verdict, snapshot.render())
        self.assertEqual(("far",), snapshot.goal_seen_by)

    def test_an_unseen_goal_refuses_the_motion_and_does_not_ask_the_camera_again(self) -> None:
        """Where the camera stands is not a camera fault: a refusal, not a raise, and no second grab."""
        world, camera = _overhead_world(self._frame())

        refresh = refresh_planner_world(
            source=world, client=_Planner(), self_envelope=_SELF, near_point_mm=(250.0, 0.0, 100.0), now=100.1,
        )

        self.assertFalse(refresh.ok)
        self.assertIs(WorldVerdict.UNSEEN, refresh.verdict)
        self.assertIsNone(refresh.camera_world())
        self.assertEqual(1, camera.grabs)

    def test_the_control_a_goal_the_camera_measured_is_fresh_and_named(self) -> None:
        world, _ = _overhead_world(self._frame())

        refresh = refresh_planner_world(
            source=world, client=_Planner(), self_envelope=_SELF, near_point_mm=(-250.0, 0.0, 100.0), now=100.1,
        )

        self.assertTrue(refresh.ok, refresh.render())
        self.assertEqual(("overhead",), refresh.goal_seen_by)
        stamp = refresh.camera_world()
        assert stamp is not None and stamp.unseen is not None
        self.assertFalse(stamp.unseen.goal_out_of_view)

    def test_a_goal_no_camera_looks_at_is_planned_and_the_stamp_says_it_was_not_seen(self) -> None:
        cases = {"beside the image": (2000.0, 0.0, 100.0), "behind the camera": (0.0, 0.0, 1500.0)}
        for label, goal in cases.items():
            with self.subTest(label):
                world, _ = _overhead_world(_plane(_OVERHEAD))

                refresh = refresh_planner_world(
                    source=world, client=_Planner(), self_envelope=_SELF, near_point_mm=goal, now=100.1,
                )

                self.assertTrue(refresh.ok, refresh.render())
                self.assertEqual((), refresh.goal_seen_by)
                stamp = refresh.camera_world()
                assert stamp is not None and stamp.unseen is not None
                self.assertTrue(stamp.unseen.goal_out_of_view)
                self.assertEqual((), stamp.unseen.no_depth)
                self.assertIn("the goal in no camera's view", stamp.render())
                self.assertIn("not seen, planned as free: the goal in no camera's view", refresh.render())
                self.assertEqual([], refresh.to_dict()["goal_seen_by"])
                snapshot = world.world_for(self_envelope=_SELF, near_point_mm=goal, now=100.1)
                self.assertIn("the goal is in no camera's view", snapshot.render())

    def test_the_control_a_full_frame_and_no_goal_stamps_what_it_always_did(self) -> None:
        world, _ = _overhead_world(_plane(_OVERHEAD))

        refresh = refresh_planner_world(source=world, client=_Planner(), self_envelope=_SELF, now=100.1)

        self.assertIsNone(refresh.goal_seen_by)
        self.assertEqual(CameraWorldStamp.planned(cameras=("overhead",), captured_at_s=100.0), refresh.camera_world())


class TheOwnersWristViewTests(unittest.TestCase):
    """The D415 on the wrist, 45 degrees down, with its minimum range: nothing nearer than it has a depth.

    The minimum ranges are Intel's datasheet values for the D415, about 450 mm at 1280 x 720 and about
    310 mm at 848 x 480, not measured here. The frames are the bench alone, cut at the minimum range.
    """

    @staticmethod
    def _snapshot(height_mm: float, min_range_mm: float, goal: tuple[float, float, float]):  # noqa: ANN205
        view = _look((0.0, -400.0, height_mm), (0.0, -1.0, -1.0))
        bench = _plane(view)
        depth = np.where(bench >= min_range_mm, bench, 0.0)
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_K, timestamp=100.0))
        world = LivePlannerWorld(
            cameras=(CameraView(name="wrist", depth_source=camera, camera_to_base=view),),
            declared=_BENCH, limits=_LIMITS, tuning=WorldBuildTuning(max_boxes=4),
        )
        return world.world_for(self_envelope=_SELF, near_point_mm=goal, now=100.1)

    def test_300_mm_over_the_bench_at_1280_x_720_the_frame_is_mostly_holes(self) -> None:
        snapshot = self._snapshot(300.0, 450.0, (0.0, -700.0, 40.0))

        self.assertIs(WorldVerdict.BLIND, snapshot.verdict, snapshot.render())
        self.assertIn("42% of the pixels hold a depth", snapshot.reason)

    def test_360_mm_over_the_bench_at_1280_x_720_the_near_goal_is_unseen_and_the_far_one_seen(self) -> None:
        near = self._snapshot(360.0, 450.0, (0.0, -610.0, 40.0))
        far = self._snapshot(360.0, 450.0, (0.0, -760.0, 40.0))

        self.assertIs(WorldVerdict.UNSEEN, near.verdict, near.render())
        self.assertIn("'wrist' 40%", near.reason)
        self.assertIs(WorldVerdict.FRESH, far.verdict, far.render())
        self.assertEqual(("wrist",), far.goal_seen_by)

    def test_at_848_x_480_both_views_see_the_goal(self) -> None:
        for height, goal in ((300.0, (0.0, -700.0, 40.0)), (360.0, (0.0, -610.0, 40.0))):
            with self.subTest(height=height):
                snapshot = self._snapshot(height, 310.0, goal)

                self.assertIs(WorldVerdict.FRESH, snapshot.verdict, snapshot.render())
                self.assertEqual(("wrist",), snapshot.goal_seen_by)


# ---------------------------------------------------------------------------------------------------
# The stamp's unseen space
# ---------------------------------------------------------------------------------------------------


class TheUnseenSpaceTests(unittest.TestCase):
    def test_it_names_something_or_is_not_built(self) -> None:
        from src.robot.core.camera_world import UnseenSpace

        with self.assertRaises(ValueError):
            UnseenSpace()
        for bad in ((("wrist", 0.0),), (("wrist", 1.5),), (("", 0.2),), (("wrist", 0.2), ("wrist", 0.3))):
            with self.subTest(bad), self.assertRaises(ValueError):
                UnseenSpace(no_depth=bad)
        with self.assertRaises(TypeError):
            UnseenSpace(no_depth=[("wrist", 0.2)])  # type: ignore[arg-type]

    def test_only_a_planned_stamp_carries_it(self) -> None:
        from src.robot.core.camera_world import UnseenSpace

        unseen = UnseenSpace(goal_out_of_view=True)
        with self.assertRaises(ValueError):
            CameraWorldStamp(use=CameraWorldUse.UNPLANNED, reason="ik", unseen=unseen)
        with self.assertRaises(TypeError):
            CameraWorldStamp.planned(cameras=("wrist",), captured_at_s=1.0, unseen="the goal")  # type: ignore[arg-type]

    def test_it_renders_one_ascii_line_and_survives_json(self) -> None:
        from src.robot.core.camera_world import UnseenSpace

        stamp = CameraWorldStamp.planned(
            cameras=("wrist",), captured_at_s=1.0,
            unseen=UnseenSpace(no_depth=(("wrist", 0.125),), goal_out_of_view=True),
        )

        text = stamp.render()
        self.assertTrue(text.isascii())
        self.assertTrue(text.endswith("not seen, planned as free: no depth on 12% of wrist, the goal in no camera's view"))
        data = json.loads(json.dumps(stamp.to_dict()))
        self.assertEqual({"no_depth": {"wrist": 0.125}, "goal_out_of_view": True}, data["unseen"])
        self.assertNotEqual(stamp, CameraWorldStamp.planned(cameras=("wrist",), captured_at_s=1.0))


# ---------------------------------------------------------------------------------------------------
# The bench band follows the error the camera declares
# ---------------------------------------------------------------------------------------------------

#: A wrist-like view, 45 degrees down from 400 mm, seeing the bench from about 0.4 to 1.2 m.
_TILTED = _look((0.0, -200.0, 400.0), (0.0, -1.0, -1.0))


class TheBenchBandTests(unittest.TestCase):
    """A fixed 5 mm clearance is below what a wrist camera's placement may be off by, so the bench came back.

    Measured with this scene 2026-09-23: placed 0.7 degrees off, the bench comes back as one box 787 x 168 mm.
    """

    def _bench_seen_off(self, **error: float) -> object:
        return build_perceived_boxes(
            views=(DepthView(_plane(_TILTED), _K, _turned(_TILTED, -0.7), name="wrist", **error),),  # type: ignore[arg-type]
            limits=_LIMITS,
        )

    def test_the_trap_a_camera_placed_off_and_declaring_nothing_sees_the_bench_as_an_obstacle(self) -> None:
        world = self._bench_seen_off()

        self.assertEqual(1, len(world.boxes), world.render())  # type: ignore[attr-defined]

    def test_a_camera_that_declares_its_error_keeps_the_bench_the_bench(self) -> None:
        world = self._bench_seen_off(placement_error_mm=1.0, placement_error_rad=math.radians(0.7))

        self.assertEqual((), world.boxes, world.render())  # type: ignore[attr-defined]

    def test_the_band_grows_with_range_and_is_reported(self) -> None:
        world = build_perceived_boxes(
            views=(DepthView(_plane(_TILTED), _K, _TILTED, name="wrist",
                             placement_error_mm=1.0, placement_error_rad=math.radians(0.7)),),
            limits=_LIMITS,
        )

        low, high = world.bench_band_mm["wrist"]
        nearest = float(np.min(_plane(_TILTED)))
        self.assertAlmostEqual(5.0 + 1.0 + math.radians(0.7) * nearest, low, delta=0.5)
        self.assertGreater(high, low + 5.0, "a point a metre out is placed less well than one 0.4 m out")
        self.assertIn("wrist: anything up to", world.render())

    def test_the_control_a_view_that_declares_nothing_keeps_the_configured_clearance(self) -> None:
        world = build_perceived_boxes(views=(DepthView(_plane(_OVERHEAD), _K, _OVERHEAD, name="overhead"),), limits=_LIMITS)

        self.assertEqual({"overhead": (5.0, 5.0)}, world.bench_band_mm)

    def test_the_band_is_capped_so_a_part_above_it_stays_an_obstacle(self) -> None:
        from src.robot.safety.planning.perceived import MAX_DECLARED_BAND_MM

        depth = _nearest(_plane(_TILTED), _box(_TILTED, (-40.0, -700.0, 0.0), (40.0, -620.0, 40.0)))

        world = build_perceived_boxes(
            views=(DepthView(depth, _K, _TILTED, name="wrist", placement_error_mm=50.0,
                             placement_error_rad=math.radians(10.0)),),
            limits=_LIMITS,
        )

        self.assertEqual(1, len(world.boxes), world.render())
        self.assertLessEqual(world.bench_band_mm["wrist"][1], 5.0 + MAX_DECLARED_BAND_MM + 1e-9)

    def test_a_placement_error_below_zero_is_refused(self) -> None:
        with self.assertRaises(PerceptionGeometryError):
            build_perceived_boxes(
                views=(DepthView(_plane(_OVERHEAD), _K, _OVERHEAD, placement_error_mm=-1.0),), limits=_LIMITS,
            )

    def test_a_wrist_frame_is_judged_against_its_rigs_declared_tolerance_and_its_lever(self) -> None:
        """The live world widens a wrist view's band by the frame's declared error and the camera's lever."""
        camera_to_tool = np.eye(4)
        camera_to_tool[:3, 3] = (0.0, 60.0, 80.0)  # 100 mm from the TCP
        tool_to_base = _turned(_TILTED, -0.7) @ np.linalg.inv(camera_to_tool)
        for declared, boxes in (((0.0, 0.0), 1), ((1.0, 0.7), 0)):
            with self.subTest(declared=declared):
                source = _Camera(DepthSnapshot(
                    depth_mm=_plane(_TILTED), intrinsics=_K, timestamp=100.0, tool_to_base_mm=tool_to_base,
                    placement_error_mm=declared[0], placement_error_deg=declared[1],
                ))
                world = LivePlannerWorld(
                    cameras=(CameraView(name="wrist", depth_source=source, camera_to_tool=camera_to_tool),),
                    declared=_BENCH, limits=_LIMITS, tuning=WorldBuildTuning(max_boxes=4),
                )

                snapshot = world.world_for(self_envelope=_SELF, now=100.1)

                self.assertIs(WorldVerdict.FRESH, snapshot.verdict, snapshot.render())
                assert snapshot.perceived is not None
                self.assertEqual(boxes, len(snapshot.perceived.boxes), snapshot.render())
                if declared[1]:
                    low, _ = snapshot.perceived.bench_band_mm["wrist"]
                    lever = math.radians(0.7) * 100.0
                    self.assertGreaterEqual(low, 5.0 + 1.0 + lever)


class ARigDeclaresTheMotionItMeasuredTests(unittest.TestCase):
    """Review of 2026-09-23: every wrist reading declared the rig's refusal threshold, which a still arm never moved."""

    def test_a_still_wrist_declares_none_one_that_crept_what_it_measured_and_a_fixed_one_none(self) -> None:
        from types import SimpleNamespace

        from src.geometry import Frame, Pose
        from src.robot.safety.planning.depth_source import RigDepthSource

        class _Handle:
            rig_id = "wrist"

            def grab(self):  # noqa: ANN202
                return SimpleNamespace(depth=np.full((4, 4), 500.0))

            def get_intrinsics(self):  # noqa: ANN202
                return _K

        def pose(x_mm: float, yaw_deg: float = 0.0) -> Pose:
            half = math.radians(yaw_deg) / 2.0
            return Pose(position_mm=np.array([x_mm, -500.0, 300.0]),
                        quaternion_xyzw=np.array([0.0, 0.0, math.sin(half), math.cos(half)]), frame=Frame.BASE)

        def reader(*poses: Pose):  # noqa: ANN202
            queue = list(poses)
            return lambda: queue.pop(0) if len(queue) > 1 else queue[0]

        still = RigDepthSource(_Handle(), tool_pose=reader(pose(0.0)), motion_tolerance=(1.5, 0.25))
        crept = RigDepthSource(_Handle(), tool_pose=reader(pose(0.0), pose(0.4, 0.2)), motion_tolerance=(1.5, 0.25))
        readings = [still.grab_surface_depth(), crept.grab_surface_depth(), RigDepthSource(_Handle()).grab_surface_depth()]

        assert all(reading is not None for reading in readings)
        declared = [(r.placement_error_mm, r.placement_error_deg) for r in readings]  # type: ignore[union-attr]
        self.assertEqual((0.0, 0.0), declared[0])
        self.assertAlmostEqual(0.4, declared[1][0], places=9)
        self.assertAlmostEqual(0.2, declared[1][1], places=6)
        self.assertEqual((0.0, 0.0), declared[2])


# ---------------------------------------------------------------------------------------------------
# The reach: what the robot's body can meet, not what its TCP may reach
# ---------------------------------------------------------------------------------------------------


def _ur10_hande_envelope(joints: np.ndarray) -> SelfEnvelope:
    """The owner's arm and hand as the self filter builds them: the committed UR10 bundle and a Hand-E on 20 mm."""
    from src.config.schema.robot import RobotConfig
    from src.robot.safety._ur_kinematics import ur_link_transforms_mm
    from src.robot.safety.planning.hand import planner_hand
    from src.robot.safety.planning.self_envelope import arm_capsules, hand_spheres

    hand = planner_hand(RobotConfig.model_validate({"vendor": "ur", "gripper": {
        "model": "robotiq_hande", "coupling_plates": [{"name": "plate", "thickness_mm": 20.0}]}}))
    capsules = arm_capsules("ur10")
    spheres = hand_spheres(hand, "ur10")  # type: ignore[arg-type]
    frames = ur_link_transforms_mm("ur10", np.asarray(joints, dtype=np.float64))
    assert capsules is not None and spheres is not None and frames is not None
    return SelfEnvelope(frames_mm=tuple(frames), capsules=tuple(capsules) + tuple(spheres))


#: The arm reaching out over the bench toward base -Y, where the owner works.
_REACHING = np.array([-1.57, -1.2, 1.4, -1.77, -1.57, 0.0])


class TheReachIsTheBodysTests(unittest.TestCase):
    """reservation.py culled perceived points to the TCP's workspace box: a fence at y -820 on the UR10 was
    1507 points OUTSIDE_LIMITS and no obstacle, while its wrist links and the D415 swing past y -760."""

    def test_the_reach_holds_the_body_in_every_configuration(self) -> None:
        reach = _ur10_hande_envelope(_REACHING).reach()
        rng = np.random.default_rng(3)
        farthest = 0.0
        for joints in rng.uniform(-np.pi, np.pi, (300, 6)):
            envelope = _ur10_hande_envelope(joints)
            self.assertAlmostEqual(reach.radius_mm, envelope.reach().radius_mm, places=6,
                                   msg="the bound does not depend on where the arm stands")
            for capsule in envelope.capsules:
                frame = envelope.frames_mm[capsule.frame]
                for end in (capsule.start_mm, capsule.end_mm):
                    point = frame[:3, :3] @ np.asarray(end) + frame[:3, 3]
                    farthest = max(farthest, float(np.linalg.norm(point - np.asarray(reach.center_mm))) + capsule.radius_mm)
        self.assertLessEqual(farthest, reach.radius_mm)
        # The numbers the docstring of SelfEnvelope.reach states.
        self.assertAlmostEqual(1793.0, reach.radius_mm, delta=1.0)
        self.assertAlmostEqual(1558.0, farthest, delta=1.0)

    def test_the_padding_grows_it_and_a_bad_capsule_is_refused(self) -> None:
        envelope = _ur10_hande_envelope(_REACHING)
        self.assertAlmostEqual(envelope.reach().radius_mm + 15.0, envelope.reach(padding_mm=15.0).radius_mm)
        with self.assertRaises(PerceptionGeometryError):
            SelfEnvelope(frames_mm=(np.eye(4),), capsules=(LinkCapsule(3, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0),)).reach()

    @staticmethod
    def _fence_world(fence_y_mm: float) -> LivePlannerWorld:
        # The wrist camera looking out along -Y and down at a fence 400 mm tall across the bench.
        view = _look((0.0, -500.0, 500.0), (0.0, -1.0, -0.6))
        fence = _box(view, (-300.0, fence_y_mm - 20.0, 0.0), (300.0, fence_y_mm, 400.0))
        depth = _nearest(_plane(view), fence)
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_K, timestamp=100.0))
        return LivePlannerWorld(
            cameras=(CameraView(name="wrist", depth_source=camera, camera_to_base=view),),
            declared=_BENCH, limits=_UR10_LIMITS, tuning=WorldBuildTuning(max_boxes=4),
        )

    def test_a_fence_past_the_tcp_box_and_inside_the_arms_reach_is_an_obstacle(self) -> None:
        snapshot = self._fence_world(-820.0).world_for(self_envelope=_ur10_hande_envelope(_REACHING), now=100.1)

        self.assertIs(WorldVerdict.FRESH, snapshot.verdict, snapshot.render())
        assert snapshot.perceived is not None
        self.assertEqual(1, len(snapshot.perceived.boxes), snapshot.render())
        box = snapshot.perceived.boxes[0]
        self.assertLess(box.center_mm[1], -760.7, "the fence stands past the TCP box, and it is kept")

    def test_the_control_a_fence_past_the_arms_reach_is_dropped_by_name(self) -> None:
        snapshot = self._fence_world(-2400.0).world_for(self_envelope=_ur10_hande_envelope(_REACHING), now=100.1)

        self.assertIs(WorldVerdict.FRESH, snapshot.verdict, snapshot.render())
        assert snapshot.perceived is not None
        self.assertEqual((), snapshot.perceived.boxes, snapshot.render())
        self.assertGreater(snapshot.perceived.dropped_points["outside_limits"], 0)

    def test_the_builder_alone_keeps_the_box_without_a_reach(self) -> None:
        """The pure builder, handed no reach, culls to the box as before: a caller without a robot keeps its answer."""
        view = _look((0.0, -500.0, 500.0), (0.0, -1.0, -0.6))
        depth = _nearest(_plane(view), _box(view, (-300.0, -840.0, 0.0), (300.0, -820.0, 400.0)))

        world = build_perceived_boxes(views=(DepthView(depth, _K, view, name="wrist"),), limits=_UR10_LIMITS)

        self.assertEqual((), world.boxes, world.render())


# ---------------------------------------------------------------------------------------------------
# The slab: sunk below the bench, the clearance has to follow it down
# ---------------------------------------------------------------------------------------------------


class ASunkSlabTests(unittest.TestCase):
    """robot.yaml advised sinking the slab to about -50 mm for low picks; the perceived floor sank with it."""

    @staticmethod
    def _limits(clearance_mm: float) -> WorldBuildLimits:
        return WorldBuildLimits(x_mm=(-760.0, 760.0), y_mm=(-760.0, 760.0), z_mm=(-100.0, 900.0),
                                support_plane_top_mm=-50.0, plane_clearance_mm=clearance_mm)

    def test_the_trap_a_slab_sunk_50_mm_with_the_clearance_left_at_5_brings_the_bench_back(self) -> None:
        world = build_perceived_boxes(views=(DepthView(_plane(_TILTED), _K, _TILTED, name="wrist"),),
                                      limits=self._limits(5.0))

        self.assertEqual(1, len(world.boxes), world.render())

    def test_the_clearance_raised_by_the_sink_keeps_the_bench_the_bench_and_a_part_a_part(self) -> None:
        bench = build_perceived_boxes(views=(DepthView(_plane(_TILTED), _K, _TILTED, name="wrist"),),
                                      limits=self._limits(55.0))
        depth = _nearest(_plane(_TILTED), _box(_TILTED, (-40.0, -700.0, 0.0), (40.0, -620.0, 40.0)))
        part = build_perceived_boxes(views=(DepthView(depth, _K, _TILTED, name="wrist"),), limits=self._limits(55.0))

        self.assertEqual((), bench.boxes, bench.render())
        self.assertEqual(1, len(part.boxes), part.render())


if __name__ == "__main__":
    unittest.main()
