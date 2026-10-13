"""A part standing on another part is planned on its foot, not on the tail its mask ran down onto the mat.

The owner's cell, 2026-10-12, "the gray cube on top": the upper of two stacked 40 mm cubes on the 55 mm mat was cut
with 3 % of its pixels run down the lower cube's side onto the mat. Its lowest point read the mat, 41 mm under its
foot, and its grasps were planned on the mat, on one part 82 mm tall (``footprint hull ... support 55.5 mm, top 137.4
mm``). The support resolver now takes a cloud's foot where it stands on another part: the lowest 10 mm band holding 3 %
of its points, 15 mm or more over the declared height, with 25 mm of the part over it
(``support_resolver._lowest_mm``). A part on the support keeps its lowest point, as every reference case did.

The clouds are drawn to the cell's fused cloud of the upper cube (the replay bank's
``pick-349a9e6847a6-20261012T081043Z``): 76 % on its top face at 136 to 139 mm, 19 % down its sides from 95 mm, and the
tail from 55 to 90 mm.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.collision import resolve_support_plane

#: The mat the cubes stand on, as the camera world finds it under them.
MAT_MM = 55.5


def _heights(*bands: tuple[float, float, int], seed: int = 7) -> np.ndarray:
    """A cloud of ``(low, high, count)`` bands of heights, BASE mm, spread evenly over xy."""
    rng = np.random.default_rng(seed)
    z = np.concatenate([rng.uniform(low, high, count) for low, high, count in bands])
    xy = rng.uniform(-20.0, 20.0, (z.size, 2))
    return np.column_stack((xy, z))


def _plane_mm(cloud: np.ndarray, declared: float = MAT_MM, **keywords: float) -> float:
    return resolve_support_plane(declared_height_mm=declared, target_clouds_base_mm=[cloud], **keywords).height_mm


class APartOnAPartIsPlannedOnItsFootTests(unittest.TestCase):
    def test_the_cells_upper_cube_is_planned_on_the_lower_cubes_top(self) -> None:
        upper = _heights((136.0, 139.0, 7600), (95.0, 136.0, 1900), (55.0, 90.0, 320))

        self.assertAlmostEqual(95.0, _plane_mm(upper), delta=2.0)

    def test_a_longer_tail_down_the_lower_part_still_finds_the_foot(self) -> None:
        """8 % run down the lower cube: no 10 mm of it holds 3 % of the cloud."""
        upper = _heights((136.0, 139.0, 7300), (95.0, 136.0, 1900), (55.0, 90.0, 800))

        self.assertAlmostEqual(95.0, _plane_mm(upper), delta=2.0)

    def test_a_part_on_a_part_in_a_bin_is_planned_on_its_foot_over_the_floor(self) -> None:
        upper = _heights((136.0, 139.0, 7600), (95.0, 136.0, 1900), (20.0, 90.0, 320))

        self.assertAlmostEqual(95.0, _plane_mm(upper, declared=0.0, container_floor_mm=20.0), delta=2.0)


class APartOnTheSupportKeepsItsLowestPointTests(unittest.TestCase):
    def test_a_cube_on_the_mat_seen_down_its_side_stays_on_the_mat(self) -> None:
        cube = _heights((94.0, 97.0, 6000), (55.5, 94.0, 2600), (54.0, 56.0, 100))

        self.assertAlmostEqual(MAT_MM, _plane_mm(cube), delta=0.6)

    def test_a_tall_part_whose_side_was_seen_thinly_stays_on_the_mat(self) -> None:
        """A tenth of its points down 60 mm of side: no band of it is dense, and the top alone is no part's height."""
        tall = _heights((114.0, 116.0, 9000), (55.5, 114.0, 1000))

        self.assertAlmostEqual(MAT_MM, _plane_mm(tall), delta=0.6)

    def test_a_low_part_standing_less_than_15_mm_over_the_declared_height_keeps_its_lowest_point(self) -> None:
        """A part on a thin plate the world did not find: its lowest point, as before."""
        on_a_plate = _heights((100.0, 103.0, 6000), (66.0, 100.0, 2600), (56.0, 66.0, 60))

        self.assertAlmostEqual(56.0, _plane_mm(on_a_plate), delta=0.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
