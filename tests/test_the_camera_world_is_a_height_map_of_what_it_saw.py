"""The camera world is a height map of what the camera saw, not one block per cluster (owner, 2026-09-30).

Until now every cluster of points 25 mm apart became one box, grown 15 mm and carried down to the bench: an open bin
came back as a solid block over its inside, a low part beside a tall one took the tall one's height, and a square bin
turned on the bench was boxed square with BASE. The owner places bins beside the robot's base, where the UR10's
shoulder housing sweeps 52 mm over the base plate, and every one of those blocks refused the upper arm.

Now each cluster is laid on a grid in the bench plane (cells of at least the 25 mm the clustering runs on), turned the
way the cluster lies. Each cell keeps the highest point seen in it, the cells merge into rectangles of about one height, and each
rectangle is one box from the bench up to its top plus the 15 mm margin. A bin is a ring of walls with its inside
free, two parts stay two boxes, a turned part is a turned box, and every point the camera saw stays inside a box by
the margin. Past the slot budget the boxes merge further, only ever into boxes that hold what they replace, and the
world refuses only what still does not fit, saying so.

Every scene is ray cast straight down through a pinhole (``tests/_seen_scenes.py``). Names new with this change are
imported inside the tests, so this file loads against the tree before it and each test fails there on its own.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.robot.safety.planning.live_world import CameraView, DepthSnapshot, LivePlannerWorld, refresh_planner_world
from src.robot.safety.planning.perceived import (
    DepthView,
    DropReason,
    LinkCapsule,
    SelfEnvelope,
    WorldBuildTuning,
    build_perceived_boxes,
)
from src.robot.safety.planning.world import planner_cuboid
from tests._seen_scenes import (
    K,
    LIMITS,
    Solid,
    clearance_inside,
    covered_by,
    deepest,
    looking_down,
    open_bin,
    render,
    seen_points,
)

#: The margin every box is grown by, the shipped 15 mm.
_MARGIN = WorldBuildTuning().margin_mm


def _seen(solids: "list[Solid]", camera_xy: "tuple[float, float]", **tuning: object) -> "tuple[object, np.ndarray]":
    """The world a camera 900 mm over ``camera_xy`` builds of ``solids``, and every point it saw above the band."""
    camera = looking_down(*camera_xy)
    depth = render(camera, solids)
    view = DepthView(surface_depth_mm=depth, intrinsics=K, camera_to_base=camera, name="overhead", timestamp=100.0)
    world = build_perceived_boxes(views=(view,), limits=LIMITS, tuning=WorldBuildTuning(**tuning))  # type: ignore[arg-type]
    return world, seen_points(camera, depth)


def _grid(x: "tuple[float, float]", y: "tuple[float, float]", z: "tuple[float, ...]", step: float = 10.0) -> np.ndarray:
    xs, ys, zs = np.meshgrid(np.arange(x[0], x[1] + 1e-9, step), np.arange(y[0], y[1] + 1e-9, step), np.asarray(z))
    return np.column_stack((xs.ravel(), ys.ravel(), zs.ravel()))


class AnOpenBinIsItsWallsTests(unittest.TestCase):
    """⛔ The owner's bins: an open bin came back as one block over its inside, and its inside was never free."""

    _BIN = open_bin((0.0, -420.0), (300.0, 200.0), 40.0)

    def test_an_open_bin_is_a_ring_of_walls_with_a_free_inside(self) -> None:
        world, seen = _seen(self._BIN, (0.0, -420.0))
        boxes = world.boxes  # type: ignore[attr-defined]

        # The inside, 45 mm and more from every wall, from the floor to the rim: nothing covers it.
        inside = _grid((-100.0, 100.0), (-470.0, -370.0), (5.0, 20.0, 35.0))
        self.assertEqual(covered_by(boxes, inside), [], world.render())  # type: ignore[attr-defined]
        # And every point the camera saw of the walls is inside a box.
        self.assertTrue(bool((deepest(boxes, seen) > 0.0).all()), "a wall point the camera saw is in no box")
        self.assertGreaterEqual(len(boxes), 4, world.render())  # type: ignore[attr-defined]

    def test_a_wall_keeps_its_own_thickness_and_its_corners_are_boxes_of_their_own(self) -> None:
        """A corner cell holds the other wall's points, and a wall that took it was boxed a cell thick over its whole
        length: 12 mm into the bin on the grasp bench (2026-10-05), where the fingers of a part beside the wall go."""
        world, seen = _seen(self._BIN, (0.0, -420.0))
        boxes = world.boxes  # type: ignore[attr-defined]

        # Along the middle of the +x wall, its inner face at x 145: 5 mm past the margin is free, at every height.
        beside = _grid((145.0 - _MARGIN - 5.0, 145.0 - _MARGIN - 5.0), (-470.0, -370.0), (5.0, 20.0, 35.0))
        self.assertEqual(covered_by(boxes, beside), [], world.render())  # type: ignore[attr-defined]
        # And the corners stay covered: every point the camera saw is inside a box.
        self.assertTrue(bool((deepest(boxes, seen) > 0.0).all()), "a wall point the camera saw is in no box")

    def test_a_turned_open_bin_is_a_ring_of_turned_walls(self) -> None:
        turned = open_bin((0.0, -420.0), (300.0, 200.0), 40.0, yaw_deg=30.0)
        world, seen = _seen(turned, (0.0, -420.0))
        boxes = world.boxes  # type: ignore[attr-defined]

        c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
        local = _grid((-100.0, 100.0), (-50.0, 50.0), (5.0, 20.0, 35.0))
        inside = np.column_stack((c * local[:, 0] - s * local[:, 1], -420.0 + s * local[:, 0] + c * local[:, 1],
                                  local[:, 2]))
        self.assertEqual(covered_by(boxes, inside), [], world.render())  # type: ignore[attr-defined]
        self.assertTrue(bool((deepest(boxes, seen) > 0.0).all()))
        for box in boxes:
            with self.subTest(box=box.name):
                # A quarter turn is the same box, so the turn is read modulo 90 degrees.
                self.assertAlmostEqual(math.degrees(box.yaw_rad) % 90.0, 30.0, delta=1.0)


class NeighboursStaySeparateTests(unittest.TestCase):
    def test_two_parts_20_mm_apart_stay_two_boxes(self) -> None:
        """⛔ Before, the two clustered on the 25 mm grid and the low one took the tall one's height."""
        low = Solid((-40.0, -400.0, 20.0), (30.0, 30.0, 20.0))    # x -70..-10, 40 mm tall
        tall = Solid((40.0, -400.0, 45.0), (30.0, 30.0, 45.0))    # x 10..70, 90 mm tall, 20 mm beside it
        world, seen = _seen([low, tall], (0.0, -400.0))
        boxes = world.boxes  # type: ignore[attr-defined]

        self.assertGreaterEqual(len(boxes), 2, world.render())  # type: ignore[attr-defined]
        over_the_low_one = np.array([[-40.0, -400.0, 70.0], [-55.0, -415.0, 70.0], [-40.0, -400.0, 100.0]])
        self.assertEqual(covered_by(boxes, over_the_low_one), [], world.render())  # type: ignore[attr-defined]
        holding_the_low_one = [box for box in boxes if bool((clearance_inside(box, [[-40.0, -400.0, 39.0]]) > 0).all())]
        self.assertTrue(holding_the_low_one)
        for box in holding_the_low_one:
            top = box.center_mm[2] + box.dims_mm[2] / 2.0
            self.assertLessEqual(top, 40.0 + _MARGIN + 2.0, "the low part took the tall part's height")
        self.assertTrue(bool((deepest(boxes, seen) > 0.0).all()))


def _tilted(target: "tuple[float, float, float]", azimuth_deg: float, *, tilt_deg: float = 45.0,
            distance_mm: float = 600.0) -> np.ndarray:
    """The owner's camera: ``tilt_deg`` off straight down, ``distance_mm`` from ``target``, coming from ``azimuth_deg``."""
    from tests._seen_scenes import looking_at

    tilt, azimuth = math.radians(tilt_deg), math.radians(azimuth_deg)
    eye = np.asarray(target) + distance_mm * np.array([math.sin(tilt) * math.cos(azimuth),
                                                       math.sin(tilt) * math.sin(azimuth), math.cos(tilt)])
    return looking_at(tuple(eye), target)  # type: ignore[arg-type]


class NeighboursOfOneClusterKeepTheirHeightsTests(unittest.TestCase):
    def test_two_parts_a_tilted_camera_sees_as_one_cluster_keep_their_own_heights(self) -> None:
        """The owner's camera looks down at 45 degrees (review of 2026-09-30). From the low part's side the floor
        between the two is behind the low part, and the two touch: one cluster, and only the height each cell keeps
        keeps the low one low. Made red by merging cells of any height into one rectangle."""
        from src.robot.safety.planning.perceived import _cluster

        tall = Solid((40.0, -400.0, 45.0), (30.0, 30.0, 45.0))
        for shift, azimuth in ((0.0, 180.0), (0.0, 90.0), (7.5, 180.0), (7.5, -90.0)):
            low = Solid((-40.0 + shift, -400.0, 20.0), (30.0, 30.0, 20.0))    # 20 mm and 5 mm from the tall one
            camera = _tilted((0.0, -400.0, 0.0), azimuth)
            depth = render(camera, [low, tall])
            with self.subTest(gap_mm=20.0 - 2.0 * shift, azimuth_deg=azimuth):
                self.assertEqual(len(set(_cluster(seen_points(camera, depth), 25.0).tolist())), 1,
                                 "the control: the camera sees the two as one cluster")
                view = DepthView(surface_depth_mm=depth, intrinsics=K, camera_to_base=camera, name="tilted",
                                 timestamp=100.0)
                world = build_perceived_boxes(views=(view,), limits=LIMITS, tuning=WorldBuildTuning())
                over_the_low_one = np.array([[-40.0, -400.0, 70.0], [-55.0, -415.0, 70.0], [-40.0, -400.0, 100.0]])
                self.assertEqual(covered_by(world.boxes, over_the_low_one + [shift, 0.0, 0.0]), [], world.render())
                holding = [box for box in world.boxes
                           if bool((clearance_inside(box, [[-40.0 + shift, -400.0, 39.0]]) > 0).all())]
                self.assertTrue(holding)
                for box in holding:
                    self.assertLessEqual(box.center_mm[2] + box.dims_mm[2] / 2.0, 40.0 + _MARGIN + 2.0,
                                         "the low part took the tall part's height")


class ATurnedPartIsATurnedBoxTests(unittest.TestCase):
    def test_a_square_part_turned_30_degrees_is_boxed_in_its_own_turn(self) -> None:
        """⛔ A square footprint had no principal direction and was boxed square with BASE, 1.37 times as wide."""
        part = Solid((0.0, -450.0, 25.0), (40.0, 40.0, 25.0), yaw_deg=30.0)
        world, _ = _seen([part], (0.0, -450.0))
        (box,) = world.boxes  # type: ignore[attr-defined]

        self.assertAlmostEqual(math.degrees(box.yaw_rad) % 90.0, 30.0, delta=1.0)
        for side in box.dims_mm[:2]:
            self.assertAlmostEqual(side, 80.0 + 2.0 * _MARGIN, delta=8.0)

    def test_a_bin_seen_along_two_walls_is_boxed_along_its_walls(self) -> None:
        """Two walls of a bin meeting at a corner, turned 20 degrees: their principal direction runs 25 degrees off
        both, their smallest rectangle along them."""
        from src.robot.safety.planning.perceived import _oriented_box

        yaw = math.radians(20.0)
        along = np.concatenate([np.linspace(0.0, 300.0, 61), np.zeros(41)])
        across = np.concatenate([np.zeros(61), np.linspace(0.0, 200.0, 41)])
        points = np.column_stack((along * math.cos(yaw) - across * math.sin(yaw),
                                  along * math.sin(yaw) + across * math.cos(yaw), np.full(along.size, 40.0)))
        _, dims, fitted = _oriented_box(points, 0.0)

        self.assertAlmostEqual(math.degrees(fitted) % 90.0, 20.0, delta=0.5)
        self.assertAlmostEqual(max(dims[:2]), 300.0, delta=0.5)
        self.assertAlmostEqual(min(dims[:2]), 200.0, delta=0.5)

    def test_a_round_part_is_boxed_square_with_base(self) -> None:
        """No turn for what has none: a disc's smallest rectangle is no smaller turned, so the box stays put."""
        from src.robot.safety.planning.perceived import _oriented_box

        angle = np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False)
        points = np.column_stack((60.0 * np.cos(angle), 60.0 * np.sin(angle), np.full(angle.size, 30.0)))
        _, _, fitted = _oriented_box(points, 0.0)
        self.assertEqual(fitted, 0.0)


class EveryBoxStandsOnTheBenchTests(unittest.TestCase):
    def test_a_box_runs_from_the_plane_to_its_top_and_the_margin(self) -> None:
        """⛔ Before, the margin was added below the bench too: a box from 15 mm under the plane."""
        world, _ = _seen([Solid((100.0, -450.0, 60.0), (50.0, 40.0, 60.0))], (100.0, -450.0))
        (box,) = world.boxes  # type: ignore[attr-defined]

        self.assertAlmostEqual(box.center_mm[2] - box.dims_mm[2] / 2.0, 0.0, places=6)
        self.assertAlmostEqual(box.center_mm[2] + box.dims_mm[2] / 2.0, 120.0 + _MARGIN, delta=1.0)


class EveryPointStaysInsideABoxTests(unittest.TestCase):
    """The invariant the guard's 5 mm rests on: what the camera saw lies inside a box by the margin."""

    def test_every_seen_point_is_inside_a_box_by_the_margin_on_every_side_and_the_top(self) -> None:
        from src.robot.safety.planning.height_map import coarsen, height_map_columns

        rng = np.random.default_rng(11)
        for trial in range(20):
            with self.subTest(trial=trial):
                count = int(rng.integers(40, 900))
                points = np.column_stack((rng.uniform(-150.0, 150.0, count), rng.uniform(-90.0, 90.0, count),
                                          rng.choice([8.0, 25.0, 40.0, 90.0], count) + rng.normal(0.0, 2.0, count)))
                points[:, 2] = np.maximum(points[:, 2], 6.0)
                yaw = float(rng.uniform(-math.pi / 2.0, math.pi / 2.0))
                columns = height_map_columns(points, yaw_rad=yaw, floor_mm=0.0, margin_mm=15.0, cell_mm=25.0,
                                             step_mm=10.0)
                budget = int(rng.integers(1, max(2, len(columns))))
                merged = coarsen([columns], budget)
                self.assertLessEqual(len(columns), budget)
                self.assertGreaterEqual(merged, 0)
                c, s = math.cos(yaw), math.sin(yaw)
                u, v = c * points[:, 0] + s * points[:, 1], -s * points[:, 0] + c * points[:, 1]
                best = np.full(points.shape[0], -np.inf)
                for column in columns:
                    clear = np.minimum.reduce([u - column.low[0], column.high[0] - u, v - column.low[1],
                                               column.high[1] - v, column.high[2] - points[:, 2]])
                    clear = np.where(points[:, 2] >= column.low[2], clear, -np.inf)
                    best = np.maximum(best, clear)
                self.assertGreaterEqual(float(best.min()), 15.0 - 1e-6, "a point lies less than the margin inside")
                for column in columns:
                    self.assertAlmostEqual(column.low[2], 0.0, places=9, msg="a box that does not stand on the plane")

    def test_every_pixel_of_a_cluttered_bench_is_inside_a_box(self) -> None:
        clutter = [
            *open_bin((0.0, -430.0), (300.0, 200.0), 40.0, yaw_deg=12.0),
            Solid((-40.0, -420.0, 15.0), (20.0, 30.0, 15.0), yaw_deg=40.0),
            Solid((60.0, -450.0, 35.0), (25.0, 25.0, 35.0)),
            Solid((230.0, -430.0, 55.0), (30.0, 60.0, 55.0), yaw_deg=-25.0),
        ]
        world, seen = _seen(clutter, (40.0, -430.0))
        self.assertTrue(bool((deepest(world.boxes, seen) > 0.0).all()), world.render())  # type: ignore[attr-defined]

    def test_the_same_frame_gives_the_same_world_twice(self) -> None:
        scene = open_bin((0.0, -430.0), (300.0, 200.0), 40.0, yaw_deg=12.0)
        first, _ = _seen(scene, (0.0, -430.0))
        second, _ = _seen(scene, (0.0, -430.0))
        self.assertEqual(first.to_dict(), second.to_dict())  # type: ignore[attr-defined]


class ANoisyBinFloorTests(unittest.TestCase):
    """A real D415 reads a bin's floor with noise (review of 2026-09-30): with 1.5 mm of it the 3 mm floor of the bin
    the tests draw rises past the shipped 5 mm bench band and comes back as low slabs over the inside. The inside is
    free once the band holds the floor and three sigma of noise, the remedy the docs name."""

    def test_the_inside_is_free_once_the_bench_band_holds_the_floor_and_its_noise(self) -> None:
        from src.robot.safety.planning.perceived import WorldBuildLimits

        scene = open_bin((0.0, -420.0), (300.0, 200.0), 40.0)
        camera = looking_down(0.0, -420.0)
        clean = render(camera, scene)
        noisy = np.where(clean > 0.0, clean + np.random.default_rng(7).normal(0.0, 1.5, clean.shape), 0.0)
        view = DepthView(surface_depth_mm=noisy, intrinsics=K, camera_to_base=camera, name="overhead", timestamp=100.0)
        inside = _grid((-100.0, 100.0), (-470.0, -370.0), (10.0, 20.0, 35.0), step=20.0)

        def covered(clearance_mm: float) -> float:
            limits = WorldBuildLimits(x_mm=LIMITS.x_mm, y_mm=LIMITS.y_mm, z_mm=LIMITS.z_mm, support_plane_top_mm=0.0,
                                      plane_clearance_mm=clearance_mm)
            world = build_perceived_boxes(views=(view,), limits=limits, tuning=WorldBuildTuning(pixel_stride=2))
            return float((deepest(world.boxes, inside) > 0.0).mean())

        self.assertGreater(covered(5.0), 0.2, "the control: the noise lifts the floor out of the shipped band")
        self.assertEqual(covered(3.0 + 3.0 * 1.5 + 0.5), 0.0)


class WhatTheRobotHidIsFilledFromWhatWasSeenTests(unittest.TestCase):
    """``height_map_columns(hidden=...)`` and ``bridge_columns`` on their own (review of 2026-09-30): what they fill,
    and the promises they keep. ``hidden`` stands in for the cameras: it says which BASE points the robot hid."""

    #: A wall along x, 5 mm thick and 40 mm tall, seen from x = -150 to 150 but for its middle 100 mm.
    _GAP = np.array([[x, y, 40.0] for x in [*np.arange(-150.0, -49.0, 5.0), *np.arange(50.0, 151.0, 5.0)]
                     for y in (0.0, 5.0)])

    @staticmethod
    def _columns(points: np.ndarray, **fill: object) -> list:
        from src.robot.safety.planning.height_map import height_map_columns

        return height_map_columns(points, yaw_rad=0.0, floor_mm=0.0, margin_mm=15.0, cell_mm=25.0, step_mm=10.0,
                                  **fill)  # type: ignore[arg-type]

    @staticmethod
    def _covers(columns: list, point: "tuple[float, float, float]") -> bool:
        return any(bool(np.all(column.low <= point) and np.all(column.high >= point)) for column in columns)

    def test_a_stretch_the_robot_hid_inside_what_was_seen_stands_as_high_as_what_is_beside_it(self) -> None:
        middle = (0.0, 2.5, 30.0)
        self.assertFalse(self._covers(self._columns(self._GAP), middle), "the control: unseen space is free")
        hid = self._columns(self._GAP, hidden=lambda p: np.abs(p[:, 0]) < 50.0)
        self.assertTrue(self._covers(hid, middle))
        self.assertTrue(self._covers(hid, (0.0, 2.5, 40.0 + 14.9)))
        self.assertFalse(self._covers(hid, (0.0, 2.5, 40.0 + 15.1)), "filled higher than what was seen beside it")
        self.assertGreater(sum(column.hidden_cells for column in hid), 0)

    def test_a_stretch_is_never_filled_through_the_hand_that_hid_it(self) -> None:
        """The hand down in a bin hid the floor beside the part, the floor was filled as high as the wall beside it,
        and the guard held a column 212 mm tall through the hand: every way out was refused (the grasp bench,
        2026-10-05). The hand stands in the hidden middle from 12 mm up: the fill there keeps its box a margin under it,
        a fill clear of it by more than its box and the margin is kept, and so is a fill the hand stands over."""

        def on_the_hand(points: np.ndarray) -> np.ndarray:
            return (np.abs(points[:, 0] - 25.0) <= 5.0) & (np.abs(points[:, 1] - 2.5) <= 7.5) & (points[:, 2] >= 12.0)

        def hidden(points: np.ndarray) -> np.ndarray:
            return np.abs(points[:, 0]) < 50.0

        through = self._columns(self._GAP, hidden=hidden)
        self.assertTrue(self._covers(through, (25.0, 2.5, 30.0)), "the control: the fill stands in the hand")

        kept = self._columns(self._GAP, hidden=hidden, robot_on=on_the_hand)

        self.assertFalse(self._covers(kept, (25.0, 2.5, 30.0)), "the fill still runs through the hand")
        self.assertTrue(self._covers(kept, (-40.0, 2.5, 30.0)), "a fill the hand stands clear of was dropped")
        # A hand that stands over the fill keeps it: a rim under the shoulder housing.
        over = self._columns(self._GAP, hidden=hidden,
                             robot_on=lambda p: (np.abs(p[:, 0] - 25.0) <= 5.0) & (p[:, 2] > 60.0))
        self.assertTrue(self._covers(over, (25.0, 2.5, 30.0)))

    def test_a_stretch_nothing_hid_is_as_it_was(self) -> None:
        plain = self._columns(self._GAP)
        unhidden = self._columns(self._GAP, hidden=lambda p: np.zeros(p.shape[0], dtype=bool), reach_mm=150.0,
                                 taken_mm=self._GAP)
        self.assertEqual([(tuple(c.low), tuple(c.high), c.members.tolist()) for c in plain],
                         [(tuple(c.low), tuple(c.high), c.members.tolist()) for c in unhidden])

    def test_the_fill_inside_never_reaches_past_the_one_box_the_cluster_would_have_been(self) -> None:
        """Hidden everywhere, and nothing taken: every box lies inside the points' extent in their turn grown by the
        margin, no higher than their highest point and the margin, and every seen point stays inside a box."""
        from src.robot.safety.planning.height_map import height_map_columns

        rng = np.random.default_rng(23)
        for trial in range(12):
            with self.subTest(trial=trial):
                points = np.column_stack((rng.uniform(-150.0, 150.0, 300), rng.uniform(-90.0, 90.0, 300),
                                          rng.choice([8.0, 25.0, 40.0, 90.0], 300)))
                yaw = float(rng.uniform(-math.pi / 2.0, math.pi / 2.0))
                columns = height_map_columns(points, yaw_rad=yaw, floor_mm=0.0, margin_mm=15.0, cell_mm=25.0,
                                             step_mm=10.0, hidden=lambda p: np.ones(p.shape[0], dtype=bool),
                                             reach_mm=150.0)
                c, s = math.cos(yaw), math.sin(yaw)
                u, v = c * points[:, 0] + s * points[:, 1], -s * points[:, 0] + c * points[:, 1]
                low = np.array([u.min() - 15.0, v.min() - 15.0, 0.0]) - 1e-6
                high = np.array([u.max() + 15.0, v.max() + 15.0, points[:, 2].max() + 15.0]) + 1e-6
                for column in columns:
                    self.assertTrue(bool(np.all(column.low >= low) and np.all(column.high <= high)))
                best = np.full(points.shape[0], -np.inf)
                for column in columns:
                    clear = np.minimum.reduce([u - column.low[0], column.high[0] - u, v - column.low[1],
                                               column.high[1] - v, column.high[2] - points[:, 2]])
                    best = np.maximum(best, clear)
                self.assertGreaterEqual(float(best.min()), 15.0 - 1e-6)

    def test_a_row_runs_on_only_where_the_self_filter_took_something_no_higher_than_it(self) -> None:
        """A wall seen from x = -150 to 0; the robot hides x = 0 to 150. It runs on where the self filter took points
        at its height (the hand's spheres, x 0 to 100), not where it took only points over it (a housing, 60 mm up),
        and not where it took nothing."""
        wall = np.array([[x, y, 40.0] for x in np.arange(-150.0, 1.0, 5.0) for y in (0.0, 5.0)])

        def hidden(p: np.ndarray) -> np.ndarray:
            return p[:, 0] > 0.0

        took = np.array([[x, 2.5, z] for x in np.arange(5.0, 100.0, 5.0) for z in (10.0, 30.0)])
        over = np.array([[x, 2.5, 60.0] for x in np.arange(5.0, 100.0, 5.0)])
        ran = self._columns(wall, hidden=hidden, reach_mm=150.0, taken_mm=took)
        self.assertTrue(self._covers(ran, (60.0, 2.5, 30.0)))
        self.assertLessEqual(max(column.high[0] for column in ran), 100.0 + 25.0 + 15.0, "ran past what was taken")
        self.assertFalse(self._covers(ran, (60.0, 2.5, 40.0 + 15.1)), "ran on higher than the wall")
        for taken in (over, None):
            with self.subTest(taken="over the wall" if taken is not None else "nothing"):
                self.assertFalse(self._covers(self._columns(wall, hidden=hidden, reach_mm=150.0, taken_mm=taken),
                                              (60.0, 2.5, 30.0)))

    def test_an_object_is_never_widened(self) -> None:
        """A wall one cell deep, hidden and taken all round: it runs on along itself and never across."""
        wall = np.array([[x, y, 40.0] for x in np.arange(-150.0, 151.0, 5.0) for y in (0.0, 5.0)])
        everywhere = np.array([[x, y, 20.0] for x in np.arange(-300.0, 301.0, 10.0) for y in np.arange(-200.0, 201.0, 10.0)])
        columns = self._columns(wall, hidden=lambda p: np.ones(p.shape[0], dtype=bool), reach_mm=150.0,
                                taken_mm=everywhere)
        for column in columns:
            self.assertGreaterEqual(column.low[1], 0.0 - 15.0 - 1e-6)
            self.assertLessEqual(column.high[1], 5.0 + 15.0 + 1e-6)

    def test_the_shadow_between_two_parts_is_bridged_no_further_than_they_reach(self) -> None:
        from src.robot.safety.planning.height_map import bridge_columns

        left = np.array([[x, y, 40.0] for x in np.arange(-150.0, -59.0, 5.0) for y in (0.0, 5.0)])
        right = np.array([[x, y, 40.0] for x in np.arange(60.0, 151.0, 5.0) for y in (0.0, 5.0)])
        # the robot's shadow over the gap and on past the right part
        def hidden(p: np.ndarray) -> np.ndarray:
            return p[:, 0] > -60.0

        bridged = bridge_columns([left, right], floor_mm=0.0, margin_mm=15.0, cell_mm=25.0, step_mm=10.0,
                                 hidden=hidden, reach_mm=150.0)
        self.assertTrue(self._covers(bridged, (0.0, 2.5, 30.0)))
        for column in bridged:
            self.assertEqual(column.members.size, 0)
            self.assertLessEqual(column.high[0], 150.0 + 15.0 + 1e-6, "bridged past the parts it joins")
            self.assertGreaterEqual(column.low[1], 0.0 - 15.0 - 1e-6)
            self.assertLessEqual(column.high[1], 5.0 + 15.0 + 1e-6)
            self.assertLessEqual(column.high[2], 40.0 + 15.0 + 1e-6)
        self.assertEqual(bridge_columns([left, right], floor_mm=0.0, margin_mm=15.0, cell_mm=25.0, step_mm=10.0,
                                        hidden=lambda p: np.zeros(p.shape[0], dtype=bool), reach_mm=150.0), [],
                         "bridged what nothing hid")
        self.assertEqual(bridge_columns([right], floor_mm=0.0, margin_mm=15.0, cell_mm=25.0, step_mm=10.0,
                                        hidden=hidden, reach_mm=150.0), [], "one part's own shadow was bridged")

    def test_a_bridge_is_never_built_through_the_hand_that_hid_it(self) -> None:
        """The Hand-E at a cube in a tray hid the floor between the cube and the wall from the wrist camera, the bridge
        stood as high as the wall through the fingers, and every way out was refused (the grasp bench, 2026-10-06). The
        hand stands in the gap from 12 mm up: the bridge there keeps its box a margin under it; a hand down to the floor
        leaves no bridge there; the stretch the hand stands clear of is bridged as before."""
        from src.robot.safety.planning.height_map import bridge_columns

        left = np.array([[x, y, 40.0] for x in np.arange(-150.0, -59.0, 5.0) for y in (0.0, 5.0)])
        right = np.array([[x, y, 40.0] for x in np.arange(60.0, 151.0, 5.0) for y in (0.0, 5.0)])

        def hidden(p: np.ndarray) -> np.ndarray:
            return np.abs(p[:, 0]) < 60.0

        def hand_from(z_mm: float):  # noqa: ANN202
            return lambda p: (np.abs(p[:, 0] - 25.0) <= 5.0) & (np.abs(p[:, 1] - 2.5) <= 7.5) & (p[:, 2] >= z_mm)

        def bridge(robot_on: object = None) -> list:
            return bridge_columns([left, right], floor_mm=0.0, margin_mm=15.0, cell_mm=25.0, step_mm=10.0,
                                  hidden=hidden, reach_mm=150.0, robot_on=robot_on)  # type: ignore[arg-type]

        self.assertTrue(self._covers(bridge(), (25.0, 2.5, 30.0)), "the control: the bridge stands in the hand")
        held = bridge(hand_from(12.0))
        self.assertFalse(self._covers(held, (25.0, 2.5, 12.0)), "the bridge still runs through the hand")
        self.assertFalse(self._covers(held, (25.0, 2.5, 12.0 - 15.0 + 0.1)), "its box comes within a margin of it")
        self.assertTrue(self._covers(held, (-45.0, 2.5, 30.0)), "a stretch the hand stands clear of went")
        self.assertFalse(self._covers(bridge(hand_from(0.0)), (25.0, 2.5, 1.0)), "a bridge under the floor stands")
        for column in held:
            self.assertEqual(column.members.size, 0)
            self.assertGreater(column.hidden_cells, 0)

    def test_a_fill_the_hand_hangs_just_over_keeps_its_box_a_margin_under_it(self) -> None:
        """The box over a fill reaches a margin over it, and the hand hung inside that margin: the Hand-E's housing 0.9
        mm deep in a fill and the retreat refused (the grasp bench, 2026-10-06). The fill under a hand 10 mm over it is
        lowered until its box keeps a margin under the hand; under a hand higher than its box and the margin it is
        kept."""

        def hidden(points: np.ndarray) -> np.ndarray:
            return np.abs(points[:, 0]) < 50.0

        def hand_from(z_mm: float):  # noqa: ANN202
            return lambda p: (np.abs(p[:, 0] - 25.0) <= 5.0) & (np.abs(p[:, 1] - 2.5) <= 7.5) & (p[:, 2] >= z_mm)

        under = self._columns(self._GAP, hidden=hidden, robot_on=hand_from(50.0))
        self.assertFalse(self._covers(under, (25.0, 2.5, 50.0 - 15.0 + 0.1)), "its box comes within a margin of it")
        self.assertTrue(self._covers(under, (-40.0, 2.5, 30.0)), "a fill the hand stands clear of was dropped")
        high = self._columns(self._GAP, hidden=hidden, robot_on=hand_from(40.0 + 2 * 15.0 + 5.0))
        self.assertTrue(self._covers(high, (25.0, 2.5, 40.0 + 14.9)), "a fill under a hand clear of its box went")

    def test_a_row_never_runs_on_through_the_hand(self) -> None:
        """A neighbour's row ran on a whole cell into the cell the fingers stood in, on one point the filter had taken
        there, the fingers 10 mm inside its box (the grasp bench, 2026-10-06). The row runs on as before where the hand
        stands clear of it, and keeps its box a margin under the hand where the hand stands in it."""
        wall = np.array([[x, y, 40.0] for x in np.arange(-150.0, 1.0, 5.0) for y in (0.0, 5.0)])

        def hidden(p: np.ndarray) -> np.ndarray:
            return p[:, 0] > 0.0

        took = np.array([[x, 2.5, z] for x in np.arange(5.0, 100.0, 5.0) for z in (10.0, 30.0)])

        def hand_from(z_mm: float):  # noqa: ANN202
            return lambda p: (np.abs(p[:, 0] - 25.0) <= 5.0) & (np.abs(p[:, 1] - 2.5) <= 7.5) & (p[:, 2] >= z_mm)

        clear = self._columns(wall, hidden=hidden, reach_mm=150.0, taken_mm=took, robot_on=hand_from(200.0))
        self.assertTrue(self._covers(clear, (60.0, 2.5, 30.0)), "the control: the row runs on past a hand clear of it")
        held = self._columns(wall, hidden=hidden, reach_mm=150.0, taken_mm=took, robot_on=hand_from(12.0))
        self.assertFalse(self._covers(held, (25.0, 2.5, 12.0)), "the row ran on through the hand")
        self.assertFalse(self._covers(held, (25.0, 2.5, 12.0 - 15.0 + 0.1)), "its box comes within a margin of it")

    def test_two_things_of_two_heights_are_no_wall_to_run_on(self) -> None:
        """A tray's wall, 130 mm, and a small cylinder beside it, 105 mm, one cluster of two rows: the row across them
        ran on past the cylinder as high as the wall, into the cell where the hand stood at the part, on the part's own
        top the self filter had taken for the hand (the grasp bench, 2026-10-06). Two cells of two heights are no wall:
        nothing runs on. Of one height they are, and it runs on as before."""
        def cluster(second_mm: float) -> np.ndarray:
            return np.array([[x, y, 130.0 if y < 25.0 else second_mm]
                             for x in np.arange(0.0, 51.0, 5.0) for y in (*np.arange(0.0, 23.0, 5.0),
                                                                          *np.arange(30.0, 51.0, 5.0))])

        def hidden(p: np.ndarray) -> np.ndarray:
            return p[:, 1] > 50.0

        took = np.array([[x, y, 90.0] for x in np.arange(5.0, 50.0, 5.0) for y in np.arange(55.0, 75.0, 5.0)])
        # Past the cylinder's own box, 15 mm round its last point at y = 50: in the cell a row runs on into.
        past = (25.0, 72.0, 100.0)
        two = self._columns(cluster(105.0), hidden=hidden, reach_mm=150.0, taken_mm=took)
        self.assertFalse(self._covers(two, past), "the wall ran on past the cylinder")
        one = self._columns(cluster(128.0), hidden=hidden, reach_mm=150.0, taken_mm=took)
        self.assertTrue(self._covers(one, past), "the control: one wall runs on")


class TheBudgetMergesBeforeItRefusesTests(unittest.TestCase):
    _BIN = open_bin((0.0, -420.0), (300.0, 200.0), 40.0)

    def test_over_the_budget_the_boxes_merge_to_fit_and_still_hold_every_point(self) -> None:
        """⛔ The walls of one bin are more boxes than two slots: they merge into two, never dropped."""
        roomy, _ = _seen(self._BIN, (0.0, -420.0))
        tight, seen = _seen(self._BIN, (0.0, -420.0), max_boxes=2)

        self.assertGreater(len(roomy.boxes), 2)  # type: ignore[attr-defined]
        self.assertEqual(len(tight.boxes), 2, tight.render())  # type: ignore[attr-defined]
        self.assertEqual(tight.dropped_obstacle_count, 0)  # type: ignore[attr-defined]
        self.assertEqual(tight.merged_to_fit, len(roomy.boxes) - 2)  # type: ignore[attr-defined]
        self.assertTrue(bool((deepest(tight.boxes, seen) > 0.0).all()))  # type: ignore[attr-defined]
        self.assertIn("merged", tight.render())  # type: ignore[attr-defined]
        self.assertEqual(tight.to_dict()["merged_to_fit"], len(roomy.boxes) - 2)  # type: ignore[attr-defined]

    def test_a_merge_holds_what_it_replaced(self) -> None:
        from src.robot.safety.planning.height_map import coarsen, height_map_columns

        rng = np.random.default_rng(5)
        points = np.column_stack((rng.uniform(-200.0, 200.0, 600), rng.uniform(-60.0, 60.0, 600),
                                  rng.choice([10.0, 30.0, 70.0], 600)))
        columns = height_map_columns(points, yaw_rad=0.3, floor_mm=0.0, margin_mm=15.0, cell_mm=25.0, step_mm=10.0)
        before = [(np.array(column.low), np.array(column.high)) for column in columns]
        coarsen([columns], 3)
        for low, high in before:
            self.assertTrue(any(bool(np.all(column.low <= low + 1e-9) and np.all(column.high >= high - 1e-9))
                                for column in columns), "a merged box does not hold a box it replaced")

    def test_with_a_goal_the_boxes_far_from_it_merge_first(self) -> None:
        """In a pile the boxes merged wherever the least volume was, and those beside the part grew past what the
        calculator had planned the fingers against (the grasp bench, 2026-10-06). Two rows of posts alike, one at the
        goal and one 400 mm off: with the goal named, the far row merges and the near one stays as it was seen; merged
        or not, every box still holds what it held."""
        from src.robot.safety.planning.height_map import Column, coarsen

        def row(x_mm: float) -> list[Column]:
            return [Column(members=np.zeros(0, dtype=np.int64), low=np.array([x_mm - 10.0, y - 10.0, 0.0]),
                           high=np.array([x_mm + 10.0, y + 10.0, 40.0]), hidden_cells=0)
                    for y in np.arange(-100.0, 101.0, 40.0)]

        near, far = row(0.0), row(400.0)
        before = [(column.low.copy(), column.high.copy()) for column in near + far]
        taken = coarsen([near, far], len(near) + 2, near_mm=np.array([0.0, 0.0, 20.0]), yaws=[0.0, 0.0])
        self.assertEqual(taken, 4)
        self.assertEqual(len(near), 6, "a box beside the goal was merged while far ones were left")
        self.assertEqual(len(far), 2)
        for low, high in before:
            self.assertTrue(any(bool(np.all(c.low <= low + 1e-9) and np.all(c.high >= high - 1e-9))
                                for c in near + far), "a merged box does not hold a box it replaced")
        # A part turned half round reads the goal in its own turn: its row at 400 mm along its own axis lies at
        # x = -400 mm in BASE, where the goal is, and stays; the unturned row at the origin merges.
        turned, plain = row(400.0), row(0.0)
        coarsen([turned, plain], len(turned) + 2, near_mm=np.array([-400.0, 0.0, 20.0]), yaws=[math.pi, 0.0])
        self.assertEqual(len(turned), 6, "the goal was not read in the part's own turn")
        self.assertEqual(len(plain), 2)

    def test_what_does_not_fit_as_one_box_per_object_is_refused_with_a_sentence(self) -> None:
        scene = [Solid((x, -450.0, 30.0), (25.0, 25.0, 30.0)) for x in (-150.0, 0.0, 150.0)]
        depth = render(looking_down(0.0, -450.0), scene)

        class _Camera:
            def grab_surface_depth(self) -> DepthSnapshot:
                return DepthSnapshot(depth_mm=depth, intrinsics=K, timestamp=100.0)

        class _Client:
            def set_world(self, boxes: list) -> int:
                return len(boxes)

        world = LivePlannerWorld(
            cameras=(CameraView(name="overhead", depth_source=_Camera(), camera_to_base=looking_down(0.0, -450.0)),),
            declared=(planner_cuboid("support_plane", (0.0, 0.0, -25.0), (2000.0, 2000.0, 50.0)),),
            limits=LIMITS, tuning=WorldBuildTuning(max_boxes=2), max_age_ms=500.0,
        )
        body = SelfEnvelope(frames_mm=(np.eye(4),), capsules=(
            LinkCapsule(frame=0, start_mm=(0.0, 0.0, 0.0), end_mm=(0.0, 0.0, 100.0), radius_mm=50.0),))
        refresh = refresh_planner_world(source=world, client=_Client(), self_envelope=body,
                                        near_point_mm=(150.0, -450.0, 60.0), now=100.1)

        self.assertFalse(refresh.ok)
        self.assertEqual(refresh.dropped_obstacles, 1)
        self.assertEqual(refresh.guard_boxes, ())
        self.assertIn("one box per object", refresh.reason)
        self.assertIn("safety.planning_world.perceived.max_boxes", refresh.reason)
        self.assertIn(DropReason.NO_SLOT, world.world_for(self_envelope=body, now=100.1).perceived.dropped_clusters)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
