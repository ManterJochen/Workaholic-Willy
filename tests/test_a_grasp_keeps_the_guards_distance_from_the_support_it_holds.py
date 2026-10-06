"""A grasp keeps the guard's distance from the support's solid the guard holds, and sits as low as that admits.

The camera world holds a support as a solid whose top stands over the surface's reading by what the reading stands over
its plane, the band and the allowance (``support_surfaces.SupportSolid.top_at``), and the guard keeps its distance from
that top. SFE planned the hand's height on the reading alone: once its grasps went lower (the owner, 2026-10-05: "wir
müssen tiefer gehen"), the calculator's post-hoc filter refused all 12 grasps of a 30 mm cylinder lying on the mat, whose
solid stood 7.3 mm over the reading, and three shapes the bench had picked the evening before got no grasp (the grasp
bench, 2026-10-06). ``support_footprint.HandFloor`` hands SFE the solids: no point of the hand comes nearer their top
than the guard's distance, and the height solve starts there.
"""

from __future__ import annotations

import math
import unittest
from functools import lru_cache

import numpy as np

from src.robot.grasping.generation.support_footprint import (
    HandFloor,
    SupportFootprintJaw,
    generate_support_footprint_grasps,
)
from src.robot.safety.planning.support_surfaces import SupportSolid

#: The reading of the support under the part, BASE z, millimetres.
READING_MM = 56.0
#: How far the held solid's top stands over the reading: p90 excess 2, band 3, allowance 2.
OVER_THE_READING_MM = 7.0
#: The guard's distance from a camera box.
GUARD_MM = 3.0
#: The support clearance over the reading the cell asks: 1 mm since 2026-10-06 (the owner: "bis auf 1 mm"), 3 before.
CLEARANCE_MM = 1.0


def _solid(top_over_reading_mm: float = OVER_THE_READING_MM, *, tilt_x: float = 0.0) -> SupportSolid:
    """A 600 x 600 mm piece of the mat whose held top stands ``top_over_reading_mm`` over the reading at its centre, its
    top face rising ``tilt_x`` millimetres a millimetre along x, as the reading does."""
    excess, allowance = 2.0, 2.0
    band = top_over_reading_mm - excess - allowance
    depth = 40.0
    top = READING_MM + top_over_reading_mm
    angle = -math.atan(tilt_x)
    turn = np.array([[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]])
    return SupportSolid(
        name="seen_s00_support0", kind="support", surface=0,
        centre_mm=np.array([0.0, -650.0, top]) - turn[:, 2] * (depth / 2.0),
        half_extents_mm=np.array([300.0, 300.0, depth / 2.0]), rotation=turn,
        local_plane=np.array([READING_MM, tilt_x, 0.0]), excess_mm=excess, band_mm=band, allowance_mm=allowance)


def _box_cloud(height_mm: float, *, size_mm: float = 40.0, step: float = 2.0) -> np.ndarray:
    """A ``size_mm`` square box ``height_mm`` tall on the reading at (0, -650): its top and its four sides."""
    half = size_mm / 2.0
    along = np.arange(-half, half + 1e-9, step)
    gx, gy = np.meshgrid(along, along, indexing="ij")
    top = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, READING_MM + height_mm)])
    zs = np.arange(READING_MM + 1.0, READING_MM + height_mm, step)
    sides = []
    for z in zs:
        for value in along:
            sides.extend([(value, -half, z), (value, half, z), (-half, value, z), (half, value, z)])
    cloud = np.vstack([top, np.asarray(sides, dtype=np.float64)])
    return cloud + np.array([0.0, -650.0, 0.0])


@lru_cache(maxsize=None)
def _jaw() -> SupportFootprintJaw:
    """The owner's Hand-E as the shipped ``hande`` profile builds it: 50 mm of stroke, the tip 10.45 mm under the grasp,
    the support clearance 1 mm."""
    from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import hande_cell

    jaw = SupportFootprintJaw.from_robot_config(hande_cell())
    assert jaw.table_clearance_mm == CLEARANCE_MM, jaw
    return jaw


def _grasps(cloud: np.ndarray, floor: HandFloor | None, refusals: dict[str, int] | None = None):  # noqa: ANN202
    return generate_support_footprint_grasps(cloud, support_height_mm=READING_MM, jaw=_jaw(), max_candidates=40,
                                             hand_floor=floor, refusals=refusals)


class TheFloorIsTheHeldTopAndTheDistance(unittest.TestCase):
    def test_over_the_solid_it_is_its_top_and_the_distance(self) -> None:
        floor = HandFloor(solids=(_solid(),), distance_mm=GUARD_MM)
        at = floor.at(np.array([[0.0, -650.0], [120.0, -500.0]]))
        np.testing.assert_allclose(at, READING_MM + OVER_THE_READING_MM + GUARD_MM)

    def test_where_no_solid_lies_there_is_none(self) -> None:
        floor = HandFloor(solids=(_solid(),), distance_mm=GUARD_MM)
        self.assertEqual(float(floor.at(np.array([[2000.0, 0.0]]))[0]), -math.inf)
        self.assertFalse(floor.under(np.array([[2000.0, 0.0, 0.0]])))

    def test_a_tilted_solid_lifts_it_where_its_top_rises(self) -> None:
        floor = HandFloor(solids=(_solid(tilt_x=0.02),), distance_mm=GUARD_MM)
        low, high = floor.at(np.array([[-100.0, -650.0], [100.0, -650.0]]))
        self.assertAlmostEqual(float(high - low), 4.0, places=6)

    def test_it_is_the_box_the_guard_holds(self) -> None:
        # The support model's own top over the reading and the box's top face are one height.
        solid = _solid(tilt_x=0.01)
        xy = np.array([[-150.0, -700.0], [0.0, -650.0], [120.0, -540.0]])
        floor = HandFloor(solids=(solid,), distance_mm=GUARD_MM)
        np.testing.assert_allclose(floor.at(xy), solid.top_at(xy) + GUARD_MM, atol=0.05)

    def test_the_highest_solid_over_a_point_holds_it(self) -> None:
        floor = HandFloor(solids=(_solid(), _solid(12.0)), distance_mm=GUARD_MM)
        self.assertAlmostEqual(float(floor.at(np.array([[0.0, -650.0]]))[0]), READING_MM + 12.0 + GUARD_MM)

    def test_with_no_solid_nothing_is_under_it(self) -> None:
        floor = HandFloor(solids=(), distance_mm=GUARD_MM)
        self.assertEqual(floor.highest_mm, -math.inf)
        self.assertFalse(floor.under(np.array([[0.0, -650.0, -1000.0]])))


class AGraspKeepsTheGuardsDistanceFromTheHeldSupport(unittest.TestCase):
    """A 30 mm box on the mat, the bench's lying cylinder's height: the solid stands 7 mm over the reading."""

    def test_on_the_reading_alone_the_grasps_come_within_the_guards_distance(self) -> None:
        # The defect: the reading and the clearance admit fingertips the guard refuses.
        grasps = _grasps(_box_cloud(30.0), None)
        self.assertTrue(grasps)
        held_floor = READING_MM + OVER_THE_READING_MM + GUARD_MM
        self.assertTrue(any(READING_MM + g.clearance_mm < held_floor for g in grasps))

    def test_on_the_held_solid_every_grasp_keeps_the_distance(self) -> None:
        floor = HandFloor(solids=(_solid(),), distance_mm=GUARD_MM)
        grasps = _grasps(_box_cloud(30.0), floor)
        self.assertTrue(grasps, "the part stands high enough for the hand over the held solid")
        held_floor = READING_MM + OVER_THE_READING_MM + GUARD_MM
        for grasp in grasps:
            self.assertGreaterEqual(READING_MM + grasp.clearance_mm, held_floor - 1e-6)

    def test_the_lowest_grasp_sits_as_low_as_the_guard_admits(self) -> None:
        # The height solve starts at the held solid: the lowest fingertip lands within a millimetre of its floor.
        floor = HandFloor(solids=(_solid(),), distance_mm=GUARD_MM)
        grasps = _grasps(_box_cloud(30.0), floor)
        held_floor = READING_MM + OVER_THE_READING_MM + GUARD_MM
        lowest = min(READING_MM + grasp.clearance_mm for grasp in grasps)
        self.assertLess(lowest - held_floor, 1.0)

    def test_a_part_the_held_floor_leaves_no_room_on_counts_under_the_table(self) -> None:
        # 22 mm tall: the reading alone leaves a finger room on it, the held solid does not, and the support says so.
        self.assertTrue(_grasps(_box_cloud(22.0), None))
        refusals: dict[str, int] = {}
        grasps = _grasps(_box_cloud(22.0), HandFloor(solids=(_solid(),), distance_mm=GUARD_MM), refusals)
        self.assertEqual(grasps, [])
        self.assertGreater(refusals["table"], 0)
        self.assertEqual(0, sum(count for cause, count in refusals.items() if cause.startswith(("seen", "declared"))))

    def test_without_a_floor_the_generator_is_unchanged(self) -> None:
        cloud = _box_cloud(30.0)
        a = _grasps(cloud, None)
        b = generate_support_footprint_grasps(cloud, support_height_mm=READING_MM, jaw=_jaw(), max_candidates=40)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            np.testing.assert_allclose(x.position_mm, y.position_mm)
            self.assertAlmostEqual(x.score, y.score)


if __name__ == "__main__":
    unittest.main()
