"""A pad holds a part under where it is widest and square to its outline (the owner's cell, 2026-10-15).

Asked of every grasp the cell executed that day, two readings separated the grasps that held from those that came up
empty: whether the pads reached under where the part is widest (``body_grasps._under_the_widest``), and, on the colour
mask's sharp outline rather than the 2 mm grid, whether a pad met a face of the part square or an edge of it obliquely
(``body_grasps._squared``). Pinned on the owner's Hand-E, on synthetic parts read as a camera straight above reads them:

* a lying cylinder whose measured points are its upper half, and whose visual hull stands as a wall from the support to
  its top at its silhouette: every grasp across it has its pads 2 mm under its equator, read from the measured points;
  without them, the hull makes it a block, and grasps stand over the equator;
* a box is a wall to its top: the rule asks nothing more of it;
* a box turned 22.5 degrees: the grasps square to its faces rank first;
* a face the pad meets 40 degrees off square at the finger's edge is refused; the outline read on the mask lands where
  the mask is.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.robot.grasping.generation import body_grasps as bg
from src.robot.grasping.generation.footprint_hull import mask_outline_xy
from tests.robot.grasping.generation.test_the_hand_grasps_the_parts_own_body import _jaw, _surface


def _lying_upper_half(xy: np.ndarray) -> np.ndarray:
    """A cylinder 40 mm across lying along x, its axis 20 mm over the support: what a camera above measures of it."""
    return np.where((np.abs(xy[:, 0]) <= 25.0) & (np.abs(xy[:, 1]) <= 20.0),
                    20.0 + np.sqrt(np.maximum(400.0 - xy[:, 1] ** 2, 0.0)), np.nan)


def _hull_skirt(*, half_length: float = 25.0, radius: float = 20.0, top: float = 40.0, step: float = 1.0) -> np.ndarray:
    """The visual hull's points at the cylinder's silhouette: a wall from the support to its top along both long
    sides, as the calculator places them for the mask's rim (``footprint_hull.hull_points``)."""
    xs = np.arange(-half_length, half_length + 1e-9, step)
    zs = np.arange(2.0, top + 1e-9, step)
    gx, gz = np.meshgrid(xs, zs)
    return np.vstack([np.column_stack([gx.ravel(), np.full(gx.size, side * radius), gz.ravel()]) for side in (-1, 1)])


def _yaw(candidate) -> float:
    return math.degrees(math.atan2(float(candidate.closing_axis[1]), float(candidate.closing_axis[0]))) % 180.0


class UnderTheWidestTests(unittest.TestCase):
    def test_a_lying_cylinder_is_held_under_its_equator_read_from_its_measured_points(self) -> None:
        jaw = _jaw()
        measured = _surface(_lying_upper_half, 30.0, n=16000)
        cloud = np.vstack([measured, _hull_skirt()])
        found = bg.body_grasps(cloud, support_height_mm=0.0, jaw=jaw, max_candidates=12, measured_base_mm=measured)
        across = [c for c in found if abs(_yaw(c) - 90.0) < 10.0]
        self.assertTrue(across, "no grasp across the cylinder")
        for candidate in across:
            self.assertLessEqual(float(candidate.position_mm[2]) - jaw.pad_ahead_mm, 20.0 - bg.UNDER_THE_WIDEST_MM + 0.5)
        # A grasp across with its pads 4 mm over the equator: refused on the measured points; read from the hull as
        # well, the cylinder is a block to its top, and it would stand.
        body = bg.body_map(cloud, 0.0, floor_margin_mm=2.0)
        high = bg._Found(anchor=np.array([0.0, 0.0, 24.0 + jaw.pad_ahead_mm]), axis=np.array([0.0, 1.0, 0.0]),
                         width_mm=40.0, angle_rad=0.0, score=1.0, clearance_mm=20.0, robust=True)
        counted: dict[str, int] = {}
        self.assertEqual([], bg._under_the_widest(
            [high], body, measured, jaw, count=lambda cause, n: counted.__setitem__(cause, counted.get(cause, 0) + n)))
        self.assertEqual({"cone": 1}, counted)
        self.assertEqual([high], bg._under_the_widest([high], body, body.points, jaw, count=lambda cause, n: None))

    def test_a_box_is_a_wall_to_its_top_and_asked_nothing_more(self) -> None:
        jaw = _jaw()
        box = _surface(lambda xy: np.where((np.abs(xy[:, 0]) <= 15.0) & (np.abs(xy[:, 1]) <= 15.0), 40.0, np.nan), 20.0)
        refused: dict[str, int] = {}
        with_rule = bg.body_grasps(box, support_height_mm=0.0, jaw=jaw, max_candidates=12, refusals=refused)
        original = bg._under_the_widest
        bg._under_the_widest = lambda batch, body, measured, jaw, *, count: batch
        try:
            without = bg.body_grasps(box, support_height_mm=0.0, jaw=jaw, max_candidates=12)
        finally:
            bg._under_the_widest = original
        self.assertEqual([tuple(np.round(c.position_mm, 3)) for c in without],
                         [tuple(np.round(c.position_mm, 3)) for c in with_rule])


class SquareToTheOutlineTests(unittest.TestCase):
    @staticmethod
    def _turned_box(degrees: float):
        turn = math.radians(degrees)
        c, s = math.cos(turn), math.sin(turn)

        def height(xy: np.ndarray) -> np.ndarray:
            u = c * xy[:, 0] + s * xy[:, 1]
            v = -s * xy[:, 0] + c * xy[:, 1]
            return np.where((np.abs(u) <= 17.0) & (np.abs(v) <= 12.0), 30.0, np.nan)

        side = np.linspace(-1.0, 1.0, 120)
        edges = np.vstack([np.column_stack([np.full(side.size, 17.0 * sign), 12.0 * side]) for sign in (-1, 1)]
                          + [np.column_stack([17.0 * side, np.full(side.size, 12.0 * sign)]) for sign in (-1, 1)])
        outline = edges @ np.array([[c, s], [-s, c]])
        return height, outline

    def test_the_grasps_square_to_a_turned_boxs_faces_rank_first(self) -> None:
        jaw = _jaw()
        height, outline = self._turned_box(22.5)
        cloud = _surface(height, 25.0)
        found = bg.body_grasps(cloud, support_height_mm=0.0, jaw=jaw, max_candidates=12, outline_base_mm=outline)
        self.assertTrue(found)
        first = _yaw(found[0])
        off = min(abs((first - normal + 90.0) % 180.0 - 90.0) for normal in (22.5, 112.5))
        self.assertLessEqual(off, bg.SQUARE_DEG, f"the first grasp closes at {first:.1f} deg")

    def test_a_face_met_forty_degrees_off_at_the_fingers_edge_is_refused(self) -> None:
        jaw = _jaw()
        found = bg._Found(anchor=np.zeros(3), axis=np.array([1.0, 0.0, 0.0]), width_mm=30.0, angle_rad=0.0, score=1.0,
                          clearance_mm=5.0, robust=True)
        v = np.linspace(-14.0, 14.0, 60)
        # The far face runs 40 degrees off square across the finger: it reaches furthest at the finger's edge.
        face = np.column_stack([15.0 + math.tan(math.radians(40.0)) * v, v])
        near = np.column_stack([np.full(v.size, -15.0), v])
        counted: dict[str, int] = {}
        kept = bg._squared([found], np.vstack([face, near]), jaw,
                           count=lambda cause, n: counted.__setitem__(cause, counted.get(cause, 0) + n))
        self.assertEqual([], kept)
        self.assertEqual({"cone": 1}, counted)
        square = bg._squared([found], np.vstack([np.column_stack([np.full(v.size, 15.0), v]), near]), jaw,
                             count=lambda cause, n: None)
        self.assertEqual(1, len(square))
        self.assertLess(square[0].turn_deg, 1.0)

    def test_the_outline_read_on_the_mask_lands_where_the_mask_is(self) -> None:
        """A 40 x 20 px mask seen straight down from 600 mm: its edge at the part's top, 30 mm up, in BASE."""
        k = np.array([[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]])
        camera_to_base = np.eye(4)
        camera_to_base[:3, :3] = np.diag([1.0, -1.0, -1.0])
        camera_to_base[:3, 3] = (100.0, -500.0, 600.0)
        mask = np.zeros((480, 640), dtype=bool)
        mask[230:250, 300:340] = True
        outline = mask_outline_xy(mask, k, camera_to_base, 30.0)
        scale = (600.0 - 30.0) / 900.0
        self.assertAlmostEqual(100.0 + (300 - 320) * scale, float(outline[:, 0].min()), delta=0.01)
        self.assertAlmostEqual(100.0 + (339 - 320) * scale, float(outline[:, 0].max()), delta=0.01)
        self.assertAlmostEqual(-500.0 - (249 - 240) * scale, float(outline[:, 1].min()), delta=0.01)
        self.assertEqual((0, 2), mask_outline_xy(np.zeros((4, 4), bool), k, camera_to_base, 30.0).shape)


if __name__ == "__main__":
    unittest.main()
