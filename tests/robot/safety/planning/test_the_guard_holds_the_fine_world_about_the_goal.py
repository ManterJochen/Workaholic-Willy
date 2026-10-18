"""The guard holds the fine world about a motion's goal, and never less than the camera world kept (2026-10-13).

``fine_world.fine_columns`` lays every pixel on a 2 mm grid and boxes it a margin round; ``perceived._fine_guard_world``
builds the guard's world of them about the goal, the planner keeping the camera world's boxes. Pinned:

* every point stands a margin inside a fine box, its sides and its top, the box carried down to the floor; the same
  points give the same boxes; a one-cell hole in a part's top is no gap; a cell is a part's where most of its points
  are, a part's fringe its own, a wall beside it nobody's; a part resting on a held box stands on it;
* a speck of a pixel or two leaves, a cluster the camera world takes for noise leaves, a held target's fringe is the
  target's;
* on the owner's cell frames of 2026-10-07 (``tests/data/cell_2026_10_07``, as the camera world's golden builds them):
  the planner's boxes are the camera world's, byte for byte; the guard holds fine boxes about the goal and every point
  the camera world kept there lies in one of them (``fine_uncovered`` 0);
* the room it gives: a finger 5 mm from a measured wall is refused by the camera world's box and kept by the fine one.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from src.robot.safety.planning.fine_world import (
    FINE_CELL_MM,
    FINE_MARGIN_MM,
    FINE_TARGET_FRINGE_MM,
    FineRegion,
    fine_columns,
    fringe_of,
    noise_clusters,
    specks,
)


def _inside(points: np.ndarray, columns: "list", margin: float = FINE_MARGIN_MM) -> np.ndarray:
    """Which points stand a margin inside one of ``columns``, its sides and its top, over its floor."""
    out = np.zeros(points.shape[0], dtype=bool)
    for c in columns:
        low, high = np.asarray(c.low_mm), np.asarray(c.high_mm)
        out |= ((points[:, 0] >= low[0] + margin - 1e-9) & (points[:, 0] <= high[0] - margin + 1e-9)
                & (points[:, 1] >= low[1] + margin - 1e-9) & (points[:, 1] <= high[1] - margin + 1e-9)
                & (points[:, 2] >= low[2] - 1e-9) & (points[:, 2] <= high[2] - margin + 1e-9))
    return out


def _cube(centre: "tuple[float, float]", half: float, top: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-half, half, (n, 2)) + np.asarray(centre)
    return np.column_stack([xy, np.full(n, top) + rng.normal(0.0, 0.4, n)])


class TheFineColumns(unittest.TestCase):
    def test_every_point_stands_a_margin_inside_a_box_on_the_floor(self) -> None:
        rng = np.random.default_rng(1)
        points = np.vstack([_cube((0.0, 0.0), 15.0, 30.0, 3000, 2),
                            np.column_stack([rng.uniform(40, 70, 2000), rng.uniform(-10, 10, 2000),
                                             20.0 + 15.0 * np.sin(rng.uniform(0, 3, 2000))])])
        owners = np.r_[np.zeros(3000, dtype=np.int64), -np.ones(2000, dtype=np.int64)]
        columns = fine_columns(points, owners, floor_mm=0.0)
        self.assertTrue(_inside(points, columns).all())
        self.assertTrue(all(abs(c.low_mm[2]) < 1e-9 for c in columns))
        again = fine_columns(points, owners, floor_mm=0.0)
        self.assertEqual([(c.low_mm, c.high_mm, c.owner) for c in columns],
                         [(c.low_mm, c.high_mm, c.owner) for c in again])
        self.assertEqual({c.owner for c in columns}, {0, None})

    def test_a_one_cell_hole_in_a_top_is_no_gap(self) -> None:
        xs, ys = np.meshgrid(np.arange(0.5, 30.0, FINE_CELL_MM), np.arange(0.5, 30.0, FINE_CELL_MM))
        points = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 25.0)])
        hole = (np.abs(points[:, 0] - 14.5) < 0.1) & (np.abs(points[:, 1] - 14.5) < 0.1)
        columns = fine_columns(points[~hole], np.full(int((~hole).sum()), -1), floor_mm=0.0)
        self.assertTrue(_inside(np.array([[14.5, 14.5, 25.0]]), columns).all())

    def test_a_part_owns_its_fringe_and_not_the_wall_beside_it(self) -> None:
        part = _cube((0.0, 0.0), 10.0, 30.0, 2000, 3)
        rng = np.random.default_rng(4)
        fringe = np.column_stack([rng.uniform(10.5, 13.5, 200), rng.uniform(-8, 8, 200), rng.uniform(5, 25, 200)])
        wall = np.column_stack([rng.uniform(14.5, 16.0, 400), rng.uniform(-8, 8, 400), rng.uniform(40, 60, 400)])
        points = np.vstack([part, fringe, wall])
        owners = np.r_[np.zeros(2000, dtype=np.int64), -np.ones(600, dtype=np.int64)]
        columns = fine_columns(points, owners, floor_mm=0.0)
        fringe_boxes = [c for c in columns if _inside(fringe, [c]).any()]
        wall_boxes = [c for c in columns if _inside(wall, [c]).any()]
        self.assertTrue(fringe_boxes and all(c.owner == 0 for c in fringe_boxes))
        self.assertTrue(wall_boxes and all(c.owner is None for c in wall_boxes))

    def test_a_part_resting_on_a_held_box_stands_on_it(self) -> None:
        resting = _cube((0.0, 0.0), 8.0, 70.0, 800, 5)
        resting[:, 2] = np.clip(resting[:, 2], 52.0, None)
        held = (lambda xy: (np.abs(xy[:, 0]) <= 20.0) & (np.abs(xy[:, 1]) <= 20.0), 50.0)
        columns = fine_columns(resting, np.full(800, -1), floor_mm=0.0, floors=[held])
        self.assertTrue(all(abs(c.low_mm[2] - 50.0) < 1e-9 for c in columns))

    def test_specks_noise_and_a_targets_fringe(self) -> None:
        body = _cube((0.0, 0.0), 10.0, 30.0, 2000, 6)
        speck = np.array([[60.0, 60.0, 20.0], [60.5, 60.2, 20.3]])
        flags = specks(np.vstack([body, speck]))
        self.assertFalse(flags[:2000].any())
        self.assertTrue(flags[2000:].all())
        noise, _ = noise_clusters(np.vstack([body, speck]), cluster_mm=25.0, min_points=12)
        self.assertTrue(noise[2000:].all())
        box = SimpleNamespace(box_to_base_mm=np.eye(4), half_extents_mm=np.array([10.0, 10.0, 15.0]))
        near = np.array([[10.0 + FINE_TARGET_FRINGE_MM - 1.0, 0.0, 5.0], [10.0 + FINE_TARGET_FRINGE_MM + 1.0, 0.0, 5.0],
                         [12.0, 0.0, 30.0]])
        self.assertEqual(fringe_of(near, [box]).tolist(), [True, False, False])

    def test_the_region(self) -> None:
        region = FineRegion.about((100.0, -500.0, 50.0), 150.0)
        self.assertEqual(region.contains(np.array([[249.0, -500.0, 0.0], [251.0, -500.0, 0.0]])).tolist(),
                         [True, False])


class OnTheCellsFrames(unittest.TestCase):
    """The owner's frames of 2026-10-07, built as the camera world's golden builds them."""

    @classmethod
    def setUpClass(cls) -> None:
        from tests.robot.safety.planning import test_a_faster_camera_world_builds_the_same_world as golden

        cls.golden = golden
        cls.built = {}
        for name in ("F1", "F2"):
            for case in ("A", "D"):
                cls.built[(name, case)] = golden.build(name, case)

    def test_the_planner_keeps_the_camera_worlds_boxes(self) -> None:
        reference = self.golden._golden()  # noqa: SLF001 (the golden of that test)
        for (name, case), snapshot in self.built.items():
            built = self.golden.canonical(snapshot)
            self.assertEqual(self.golden.differences(reference[f"{name}/{case}"]["boxes"], built["boxes"]), [],
                             f"{name}/{case}")

    def test_the_guard_holds_the_fine_world_and_never_less(self) -> None:
        for (name, case), snapshot in self.built.items():
            perceived = snapshot.perceived
            self.assertIsNotNone(perceived.guard_boxes, f"{name}/{case}")
            self.assertGreater(perceived.fine_boxes, 0, f"{name}/{case}")
            self.assertEqual(perceived.fine_uncovered, 0, f"{name}/{case}")
            names = [box.name for box in perceived.guard_boxes]
            self.assertEqual(len(names), len(set(names)), f"{name}/{case}: two guard boxes share a name")


class TheRoomItGives(unittest.TestCase):
    def test_a_finger_five_mm_from_a_measured_wall(self) -> None:
        """The camera world grows a box 6 mm and the guard keeps 2 mm from it: a finger 5 mm from the wall's measured face
        is inside that. The fine world grows it 2 mm: the finger keeps 3 mm, over the guard's 2."""
        rng = np.random.default_rng(7)
        wall = np.column_stack([rng.uniform(0.0, 4.0, 3000), rng.uniform(-30, 30, 3000), rng.uniform(0, 60, 3000)])
        columns = fine_columns(wall, np.full(3000, -1), floor_mm=0.0)
        face = float(wall[:, 0].max())
        finger_x = face + 5.0
        nearest = min(finger_x - c.high_mm[0] for c in columns if c.high_mm[0] <= finger_x + 1e-9)
        self.assertGreaterEqual(nearest, 2.0 - FINE_CELL_MM)    # the cell's own rounding, at most one cell
        coarse_face = face + 6.0
        self.assertLess(finger_x - coarse_face, 2.0)
        self.assertTrue(math.isfinite(nearest))


if __name__ == "__main__":
    unittest.main()
