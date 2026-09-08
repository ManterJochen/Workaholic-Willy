"""The cloud-to-boxes converter, against frames built to break it.

Every test here renders its own depth map through the same pinhole model the converter inverts, so
the expected geometry is known in millimetres rather than asserted against whatever the code
happens to produce. The camera looks straight down at a bench, which is the arrangement a fixed
cell actually has and the one where a depth camera's blind spot bites: it measures the top face of
a part and nothing of its body.

What is being defended here is not the arithmetic. It is the ways a world built from a camera can be
wrong in the direction that hurts: an obstacle that quietly does not arrive, a box that covers less
than the thing it stands for, a bench that arrives as a hundred obstacles and eats the planner's
slots, and the robot registering itself so that it can no longer move.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.robot.safety.planning.perceived import (
    DepthView,
    DropReason,
    PerceptionGeometryError,
    SelfBody,
    WorldBuildLimits,
    WorldBuildTuning,
    build_perceived_boxes,
)

#: A 200 x 200 pinhole, 500 px focal length, principal point in the middle.
_FX = _FY = 500.0
_CX = _CY = 100.0
_SHAPE = (200, 200)

#: The camera hangs 1000 mm over the bench looking down: camera +z goes down to the bench, camera +y
#: runs the other way from base +y. A pixel at 1000 mm of depth is therefore a point on the bench.
_CAMERA_HEIGHT_MM = 1000.0
_CAMERA_TO_BASE = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, _CAMERA_HEIGHT_MM],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

_LIMITS = WorldBuildLimits(
    x_mm=(-400.0, 400.0), y_mm=(-400.0, 400.0), z_mm=(-50.0, 900.0), support_plane_top_mm=0.0
)


def _intrinsics() -> np.ndarray:
    return np.array([[_FX, 0.0, _CX], [0.0, _FY, _CY], [0.0, 0.0, 1.0]], dtype=np.float64)


def _view(depth: np.ndarray, **kwargs: object) -> DepthView:
    """The overhead camera, as one view of the cell."""
    return DepthView(
        surface_depth_mm=depth,
        intrinsics=_intrinsics(),
        camera_to_base=_CAMERA_TO_BASE,
        name="overhead",
        timestamp=100.0,
        **kwargs,  # type: ignore[arg-type]
    )


def _bench_depth() -> np.ndarray:
    """A depth map of nothing but the bench: every pixel at the camera height."""
    return np.full(_SHAPE, _CAMERA_HEIGHT_MM, dtype=np.float64)


def _put_block(
    depth: np.ndarray,
    *,
    centre_xy_mm: tuple[float, float],
    size_xy_mm: tuple[float, float],
    height_mm: float,
) -> np.ndarray:
    """Draw the top face of an upright block into a bench depth map, and return its mask.

    The block's top is `height_mm` above the bench, so the camera sees it at
    `_CAMERA_HEIGHT_MM - height_mm`. Only the top face is drawn, because that is all a camera above
    it can measure, and a test that drew the sides would be testing a sensor nobody has.
    """
    top_depth = _CAMERA_HEIGHT_MM - height_mm
    scale = top_depth / _FX  # millimetres per pixel at the block's top
    half_cols = size_xy_mm[0] / 2.0 / scale
    half_rows = size_xy_mm[1] / 2.0 / scale
    centre_col = _CX + centre_xy_mm[0] / scale
    centre_row = _CY - centre_xy_mm[1] / scale  # base +y is up the image

    rows, cols = np.mgrid[0 : _SHAPE[0], 0 : _SHAPE[1]]
    mask = (np.abs(cols - centre_col) <= half_cols) & (np.abs(rows - centre_row) <= half_rows)
    depth[mask] = top_depth
    return mask


def _diagonal_bar(
    depth: np.ndarray, *, angle_deg: float, length_mm: float, width_mm: float, height_mm: float
) -> np.ndarray:
    """Draw the top face of a bar lying at `angle_deg` to base +x, and return its mask."""
    top_depth = _CAMERA_HEIGHT_MM - height_mm
    scale = top_depth / _FX
    rows, cols = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]]
    x_mm = (cols - _CX) * scale
    y_mm = -(rows - _CY) * scale
    angle = math.radians(angle_deg)
    along = x_mm * math.cos(angle) + y_mm * math.sin(angle)
    across = -x_mm * math.sin(angle) + y_mm * math.cos(angle)
    mask = (np.abs(along) <= length_mm / 2.0) & (np.abs(across) <= width_mm / 2.0)
    depth[mask] = top_depth
    return mask


class ConverterGeometryTests(unittest.TestCase):
    def test_a_block_on_the_bench_becomes_one_box_where_the_block_is(self) -> None:
        depth = _bench_depth()
        _put_block(depth, centre_xy_mm=(100.0, 50.0), size_xy_mm=(120.0, 80.0), height_mm=90.0)

        world = build_perceived_boxes(views=(_view(depth),), limits=_LIMITS)

        self.assertEqual(len(world.boxes), 1, world.render())
        box = world.boxes[0]
        self.assertAlmostEqual(box.center_mm[0], 100.0, delta=15.0)
        self.assertAlmostEqual(box.center_mm[1], 50.0, delta=15.0)
        margin = 2.0 * WorldBuildTuning().margin_mm
        self.assertAlmostEqual(box.dims_mm[0], 120.0 + margin, delta=25.0)
        self.assertAlmostEqual(box.dims_mm[1], 80.0 + margin, delta=25.0)

    def test_the_bench_itself_never_becomes_an_obstacle(self) -> None:
        """Every bench pixel is a point, and the bench is already one declared box."""
        world = build_perceived_boxes(views=(_view(_bench_depth()),), limits=_LIMITS)

        self.assertEqual(world.boxes, ())
        self.assertGreater(world.dropped_points[DropReason.BELOW_PLANE], 1000)

    def test_a_box_reaches_the_bench_because_a_camera_cannot_see_under_a_part(self) -> None:
        depth = _bench_depth()
        _put_block(depth, centre_xy_mm=(0.0, 0.0), size_xy_mm=(100.0, 100.0), height_mm=120.0)

        floored = build_perceived_boxes(
            views=(_view(depth),), limits=_LIMITS, tuning=WorldBuildTuning(floor_to_plane=True)
        ).boxes[0]
        surface_only = build_perceived_boxes(
            views=(_view(depth),), limits=_LIMITS, tuning=WorldBuildTuning(floor_to_plane=False)
        ).boxes[0]

        # Floored: the box spans the block's real height. Surface only: it is the margin and nothing
        # else, which is the sheet a single overhead view actually measures.
        self.assertAlmostEqual(floored.dims_mm[2], 120.0 + 2.0 * 15.0, delta=10.0)
        self.assertAlmostEqual(surface_only.dims_mm[2], 2.0 * 15.0, delta=5.0)
        self.assertLess(surface_only.dims_mm[2], floored.dims_mm[2])

    def test_two_separated_blocks_are_two_obstacles(self) -> None:
        depth = _bench_depth()
        _put_block(depth, centre_xy_mm=(-150.0, 0.0), size_xy_mm=(60.0, 60.0), height_mm=80.0)
        _put_block(depth, centre_xy_mm=(150.0, 0.0), size_xy_mm=(60.0, 60.0), height_mm=80.0)

        world = build_perceived_boxes(views=(_view(depth),), limits=_LIMITS)

        self.assertEqual(len(world.boxes), 2, world.render())
        xs = sorted(round(box.center_mm[0]) for box in world.boxes)
        self.assertAlmostEqual(xs[0], -150.0, delta=20.0)
        self.assertAlmostEqual(xs[1], 150.0, delta=20.0)

    def test_a_long_part_is_boxed_along_its_own_axis(self) -> None:
        """A diagonal part in an axis-aligned box blocks the space beside it, so the box turns."""
        depth = _bench_depth()
        mask = _diagonal_bar(depth, angle_deg=30.0, length_mm=240.0, width_mm=40.0, height_mm=60.0)
        self.assertGreater(int(mask.sum()), 200)

        box = build_perceived_boxes(views=(_view(depth),), limits=_LIMITS).boxes[0]

        # The yaw is a direction, not an orientation: 30 and 210 degrees are the same bar.
        yaw_deg = math.degrees(box.yaw_rad) % 180.0
        self.assertAlmostEqual(yaw_deg, 30.0, delta=8.0)
        # Turned with the bar, the box is long in one axis and thin in the other. Axis-aligned it
        # would be roughly square, which is the space this test exists to stop being blocked.
        long_side, short_side = sorted(box.dims_mm[:2], reverse=True)
        self.assertGreater(long_side / short_side, 2.0)


class TwoCameraTests(unittest.TestCase):
    """A second camera exists to see what the first cannot, and must not double what it can."""

    @staticmethod
    def _side_view(depth: np.ndarray) -> DepthView:
        """A second camera 1000 mm along base +x, looking back at the cell along its own axis.

        Its transform maps camera +z onto base -x, so a patch in its image lands beside the bench
        rather than above it. A depth of 850 mm is therefore a point at base x = 150, and its
        optical axis runs at 100 mm above the bench, which is the height of the block in these
        tests: a camera aimed somewhere else would see a different object, not the same one twice.
        """
        transform = np.array(
            [
                [0.0, 0.0, -1.0, 1000.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 100.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return DepthView(
            surface_depth_mm=depth, intrinsics=_intrinsics(), camera_to_base=transform,
            name="side", timestamp=100.0,
        )

    @staticmethod
    def _side_patch(depth_mm: float = 850.0) -> np.ndarray:
        """The side camera's image: a far wall outside the workspace, and one patch of an object."""
        image = np.full(_SHAPE, 2000.0, dtype=np.float64)
        rows, cols = np.mgrid[0 : _SHAPE[0], 0 : _SHAPE[1]]
        patch = (np.abs(cols - _CX) <= 12) & (np.abs(rows - _CY) <= 12)
        image[patch] = depth_mm
        return image

    def test_one_object_seen_by_both_cameras_is_one_obstacle(self) -> None:
        """Two boxes for one part would be the fusion failing, and it would look like success."""
        overhead = _bench_depth()
        _put_block(overhead, centre_xy_mm=(150.0, 0.0), size_xy_mm=(80.0, 80.0), height_mm=100.0)

        world = build_perceived_boxes(
            views=(_view(overhead), self._side_view(self._side_patch())), limits=_LIMITS
        )

        self.assertEqual(len(world.boxes), 1, world.render())
        self.assertAlmostEqual(world.boxes[0].center_mm[0], 150.0, delta=40.0)

    def test_a_second_camera_can_only_add_geometry(self) -> None:
        """What one camera cannot see is the reason the other one is bolted to the cell."""
        overhead = _bench_depth()  # sees nothing but bench

        alone = build_perceived_boxes(views=(_view(overhead),), limits=_LIMITS)
        together = build_perceived_boxes(
            views=(_view(overhead), self._side_view(self._side_patch())), limits=_LIMITS
        )

        self.assertEqual(alone.boxes, ())
        self.assertEqual(len(together.boxes), 1, together.render())

    def test_the_fused_age_is_the_age_of_the_stalest_view(self) -> None:
        depth = _bench_depth()
        _put_block(depth, centre_xy_mm=(150.0, 0.0), size_xy_mm=(60.0, 60.0), height_mm=80.0)
        older = DepthView(
            surface_depth_mm=depth, intrinsics=_intrinsics(), camera_to_base=_CAMERA_TO_BASE,
            name="side", timestamp=99.0,
        )

        world = build_perceived_boxes(views=(_view(depth), older), limits=_LIMITS)

        self.assertEqual(world.source_timestamp, 99.0)

    def test_one_unstamped_view_makes_the_whole_world_undateable(self) -> None:
        """A world is only as current as the half of it nobody can date."""
        depth = _bench_depth()
        unstamped = DepthView(
            surface_depth_mm=depth, intrinsics=_intrinsics(), camera_to_base=_CAMERA_TO_BASE,
            name="side", timestamp=None,
        )

        world = build_perceived_boxes(views=(_view(depth), unstamped), limits=_LIMITS)

        self.assertIsNone(world.source_timestamp)

    def test_a_view_names_itself_in_a_refusal(self) -> None:
        broken = DepthView(
            surface_depth_mm=_bench_depth(), intrinsics=_intrinsics(), camera_to_base=np.eye(3),
            name="wrist", timestamp=100.0,
        )
        with self.assertRaises(PerceptionGeometryError) as caught:
            build_perceived_boxes(views=(_view(_bench_depth()), broken), limits=_LIMITS)
        self.assertIn("wrist", str(caught.exception))


class ConverterAccountingTests(unittest.TestCase):
    def test_the_excluded_mask_leaves_its_object_out(self) -> None:
        depth = _bench_depth()
        target = _put_block(
            depth, centre_xy_mm=(-150.0, 0.0), size_xy_mm=(60.0, 60.0), height_mm=80.0
        )
        _put_block(depth, centre_xy_mm=(150.0, 0.0), size_xy_mm=(60.0, 60.0), height_mm=80.0)

        world = build_perceived_boxes(
            views=(_view(depth, exclude_masks=(target,)),), limits=_LIMITS
        )

        self.assertEqual(len(world.boxes), 1, world.render())
        self.assertGreater(world.boxes[0].center_mm[0], 0.0)
        self.assertGreater(world.dropped_points[DropReason.EXCLUDED], 0)

    def test_the_slot_budget_keeps_the_nearest_and_reports_the_rest(self) -> None:
        depth = _bench_depth()
        # Kept inside the sensor: a block whose top face runs past the image edge is a block the
        # camera saw half of, and this test is about the budget rather than about clipping.
        _put_block(depth, centre_xy_mm=(-150.0, 0.0), size_xy_mm=(50.0, 50.0), height_mm=80.0)
        _put_block(depth, centre_xy_mm=(0.0, 0.0), size_xy_mm=(50.0, 50.0), height_mm=80.0)
        _put_block(depth, centre_xy_mm=(150.0, 0.0), size_xy_mm=(50.0, 50.0), height_mm=80.0)

        world = build_perceived_boxes(
            views=(_view(depth),), limits=_LIMITS, tuning=WorldBuildTuning(max_boxes=1),
            near_point_mm=(150.0, 0.0, 80.0),
        )

        self.assertEqual(len(world.boxes), 1)
        self.assertAlmostEqual(world.boxes[0].center_mm[0], 150.0, delta=20.0)
        self.assertEqual(world.dropped_obstacle_count, 2)
        self.assertIn("did NOT fit the slot budget", world.render())

    def test_an_obstacle_outside_the_declared_reach_is_dropped_and_counted(self) -> None:
        depth = _bench_depth()
        _put_block(depth, centre_xy_mm=(0.0, 0.0), size_xy_mm=(60.0, 60.0), height_mm=80.0)

        narrow = WorldBuildLimits(
            x_mm=(200.0, 400.0), y_mm=(-400.0, 400.0), z_mm=(-50.0, 900.0),
            support_plane_top_mm=0.0,
        )
        world = build_perceived_boxes(views=(_view(depth),), limits=narrow)

        self.assertEqual(world.boxes, ())
        self.assertGreater(world.dropped_points[DropReason.OUTSIDE_LIMITS], 0)

    def test_a_cluster_takes_the_name_of_the_segmentation_it_sits_in(self) -> None:
        depth = _bench_depth()
        mask = _put_block(depth, centre_xy_mm=(0.0, 0.0), size_xy_mm=(80.0, 80.0), height_mm=80.0)

        world = build_perceived_boxes(
            views=(_view(depth, labelled_masks=(("red cube", mask),)),), limits=_LIMITS
        )

        self.assertEqual(world.boxes[0].label, "red cube")
        self.assertEqual(world.boxes[0].name, "seen_00_red_cube")

    def test_two_runs_over_one_frame_produce_the_same_world(self) -> None:
        """A planner handed a different world for an unchanged scene plans a different path."""
        depth = _bench_depth()
        _put_block(depth, centre_xy_mm=(80.0, -60.0), size_xy_mm=(70.0, 110.0), height_mm=95.0)

        first = build_perceived_boxes(views=(_view(depth),), limits=_LIMITS)
        second = build_perceived_boxes(views=(_view(depth),), limits=_LIMITS)

        self.assertEqual(first.to_dict(), second.to_dict())


class SelfBodyTests(unittest.TestCase):
    """The arm is in the picture, and an arm registered as an obstacle is an arm that cannot move."""

    def test_the_robot_does_not_become_an_obstacle(self) -> None:
        depth = _bench_depth()
        # An upright column standing where the arm is, which is what a camera sees of a wrist over
        # the bench, plus a part on the bench that must survive the filter.
        _put_block(depth, centre_xy_mm=(0.0, 0.0), size_xy_mm=(90.0, 90.0), height_mm=300.0)
        _put_block(depth, centre_xy_mm=(150.0, 0.0), size_xy_mm=(60.0, 60.0), height_mm=80.0)

        body = SelfBody.from_polyline(
            [(0.0, 0.0, 0.0), (0.0, 0.0, 400.0)], radius_mm=120.0, tool_radius_mm=120.0
        )
        world = build_perceived_boxes(views=(_view(depth),), limits=_LIMITS, self_body=body)

        self.assertEqual(len(world.boxes), 1, world.render())
        self.assertAlmostEqual(world.boxes[0].center_mm[0], 150.0, delta=20.0)
        self.assertGreater(world.dropped_points[DropReason.SELF], 0)

    def test_without_the_filter_the_same_frame_registers_the_arm(self) -> None:
        """The negative half: proving the filter is what removes it, not the scene."""
        depth = _bench_depth()
        _put_block(depth, centre_xy_mm=(0.0, 0.0), size_xy_mm=(90.0, 90.0), height_mm=300.0)
        _put_block(depth, centre_xy_mm=(150.0, 0.0), size_xy_mm=(60.0, 60.0), height_mm=80.0)

        world = build_perceived_boxes(views=(_view(depth),), limits=_LIMITS)

        self.assertEqual(len(world.boxes), 2, world.render())

    def test_a_capsule_covers_the_space_between_its_ends(self) -> None:
        body = SelfBody.from_polyline([(0.0, 0.0, 0.0), (500.0, 0.0, 0.0)], radius_mm=100.0)
        points = np.array(
            [
                [250.0, 0.0, 0.0],      # on the axis, halfway
                [250.0, 99.0, 0.0],     # inside the radius
                [250.0, 101.0, 0.0],    # just outside
                [-101.0, 0.0, 0.0],     # past the near end
                [601.0, 0.0, 0.0],      # past the far end
            ]
        )
        np.testing.assert_array_equal(
            body.contains(points), np.array([True, True, False, False, False])
        )

    def test_a_body_that_is_not_capsules_is_refused(self) -> None:
        with self.assertRaises(PerceptionGeometryError):
            SelfBody.from_polyline([(0.0, 0.0, 0.0)], radius_mm=100.0)


class ConverterRefusalTests(unittest.TestCase):
    def test_no_support_plane_is_refused_rather_than_guessed(self) -> None:
        limits = WorldBuildLimits(
            x_mm=(-400.0, 400.0), y_mm=(-400.0, 400.0), z_mm=(-50.0, 900.0),
            support_plane_top_mm=None,
        )
        with self.assertRaises(PerceptionGeometryError) as caught:
            build_perceived_boxes(views=(_view(_bench_depth()),), limits=limits)
        self.assertIn("support_plane", str(caught.exception))

    def test_no_view_at_all_is_refused(self) -> None:
        with self.assertRaises(PerceptionGeometryError):
            build_perceived_boxes(views=(), limits=_LIMITS)

    def test_a_transform_of_the_wrong_shape_is_refused(self) -> None:
        view = DepthView(
            surface_depth_mm=_bench_depth(), intrinsics=_intrinsics(), camera_to_base=np.eye(3)
        )
        with self.assertRaises(PerceptionGeometryError):
            build_perceived_boxes(views=(view,), limits=_LIMITS)

    def test_a_mask_of_the_wrong_shape_is_refused(self) -> None:
        with self.assertRaises(PerceptionGeometryError):
            build_perceived_boxes(
                views=(_view(_bench_depth(), exclude_masks=(np.ones((4, 4), dtype=bool),)),),
                limits=_LIMITS,
            )

    def test_a_cluster_grid_finer_than_the_cloud_is_refused(self) -> None:
        """It would split one object into one cluster per point, and nothing would say so."""
        with self.assertRaises(PerceptionGeometryError):
            WorldBuildTuning(voxel_size_mm=20.0, cluster_voxel_mm=5.0)

    def test_an_empty_frame_yields_an_empty_world_rather_than_an_error(self) -> None:
        """No depth at all is a camera problem for the caller to judge, not a geometry error."""
        world = build_perceived_boxes(
            views=(_view(np.zeros(_SHAPE, dtype=np.float64)),), limits=_LIMITS
        )

        self.assertTrue(world.is_empty)
        self.assertEqual(world.considered_points, 0)
        self.assertIn("no perceived obstacle", world.render())


if __name__ == "__main__":
    unittest.main()
