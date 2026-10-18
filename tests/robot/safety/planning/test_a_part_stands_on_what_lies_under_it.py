"""A part stands on what lies under it, never on a surface beside it that its cloud reaches over.

On the owner's cell (2026-10-12 10:14) a crate stood turned over on the mat as a large obstacle, its bottom 185 mm up.
From the side looks the mask of the part beside it ran onto the crate, the support model read the crate's bottom under
some of the part's points, and the highest reading under the part's foot was taken as its support: every grasp of the
part was refused, the part standing "under its support". Handed the part's own points, ``SupportModel.height_under``
now takes no reading over the lowest tenth of their heights and the surface's band (``support_ceiling_mm``); handed x
and y alone it reads as before.

Read at the middle of the part's heights the ceiling let the tops of a dense pile, one surface 81 mm up round a standing
cylinder, stand under it (the owner's cell, 2026-10-15, 14:36 to 14:41): the cylinder's top filled the middle of its
heights, the looks saw its side down to the mat at 61 mm, and six grasps of its top 7 mm slipped in a row.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.safety.planning.support_surfaces import (
    UNDER_A_PART_MM,
    SupportModel,
    SupportSolid,
    support_ceiling_mm,
)

MAT_MM = 55.0
CRATE_MM = 185.0


def _solid(name: str, surface: int, top_mm: float, x_mm: "tuple[float, float]",
           y_mm: "tuple[float, float]") -> SupportSolid:
    """A level solid from the bench up to ``top_mm`` over BASE ``x_mm`` x ``y_mm``, read at ``top_mm``."""
    centre = np.array([(x_mm[0] + x_mm[1]) / 2.0, (y_mm[0] + y_mm[1]) / 2.0, top_mm / 2.0])
    half = np.array([(x_mm[1] - x_mm[0]) / 2.0, (y_mm[1] - y_mm[0]) / 2.0, top_mm / 2.0])
    return SupportSolid(name=name, kind="support", surface=surface, centre_mm=centre, half_extents_mm=half,
                        rotation=np.eye(3), local_plane=np.array([top_mm, 0.0, 0.0]), excess_mm=0.0, band_mm=5.0,
                        allowance_mm=2.0)


#: The mat, and the crate's bottom over it beside the part.
MODEL = SupportModel(surfaces=(), solids=(_solid("mat", 0, MAT_MM, (-300.0, 300.0), (-800.0, -400.0)),
                                          _solid("crate", 1, CRATE_MM, (50.0, 250.0), (-750.0, -450.0))),
                     bench_top_mm=0.0, bench_band_mm=5.0, allowance_mm=2.0)


def _part(rng: np.random.Generator, *, bled: int) -> np.ndarray:
    """A 40 mm cube on the mat beside the crate, seen whole, and ``bled`` points of a mask that ran onto the crate."""
    own = np.column_stack((rng.uniform(0.0, 40.0, 300), rng.uniform(-620.0, -580.0, 300),
                           rng.uniform(MAT_MM + 2.0, MAT_MM + 40.0, 300)))
    onto = np.column_stack((rng.uniform(55.0, 75.0, bled), rng.uniform(-620.0, -580.0, bled), np.full(bled, CRATE_MM)))
    return np.concatenate((own, onto))


class APartStandsOnWhatLiesUnderItTests(unittest.TestCase):
    def test_a_mask_run_onto_the_crate_beside_the_part_leaves_the_part_on_the_mat(self) -> None:
        points = _part(np.random.default_rng(7), bled=120)
        self.assertEqual(MAT_MM, MODEL.height_under(points))
        # What it read before, and still reads from x and y alone: the crate's bottom.
        self.assertEqual(CRATE_MM, MODEL.height_under(points[:, :2]))

    def test_a_part_standing_on_the_crate_stands_on_the_crate(self) -> None:
        rng = np.random.default_rng(3)
        points = np.column_stack((rng.uniform(100.0, 140.0, 200), rng.uniform(-620.0, -580.0, 200),
                                  rng.uniform(CRATE_MM + 2.0, CRATE_MM + 40.0, 200)))
        self.assertEqual(CRATE_MM, MODEL.height_under(points))

    def test_a_flat_part_on_the_mat_keeps_the_mat(self) -> None:
        rng = np.random.default_rng(5)
        points = np.column_stack((rng.uniform(0.0, 40.0, 200), rng.uniform(-620.0, -580.0, 200),
                                  rng.uniform(MAT_MM + 0.5, MAT_MM + 2.0, 200)))
        self.assertEqual(MAT_MM, MODEL.height_under(points))

    def test_a_standing_cylinder_in_a_pile_stands_on_the_mat_not_on_the_tops_round_it(self) -> None:
        """The Thursday cylinder: its top fills most of its points, the looks saw its side down to the mat, and the tops
        of the pile round it read as one surface over the mat, under its foot too."""
        rng = np.random.default_rng(11)
        tops = SupportModel(surfaces=(), solids=(_solid("mat", 0, MAT_MM, (-300.0, 300.0), (-800.0, -400.0)),
                                                 _solid("tops", 1, MAT_MM + 23.0, (-100.0, 100.0), (-700.0, -500.0))),
                            bench_top_mm=0.0, bench_band_mm=5.0, allowance_mm=2.0)
        angle = rng.uniform(0.0, 2.0 * np.pi, 900)
        radius = np.sqrt(rng.uniform(0.0, 1.0, 900)) * 17.0
        top = np.column_stack((radius * np.cos(angle), -600.0 + radius * np.sin(angle), np.full(900, MAT_MM + 36.0)))
        side = np.column_stack((17.0 * np.cos(angle[:400]), -600.0 + 17.0 * np.sin(angle[:400]),
                                rng.uniform(MAT_MM + 3.0, MAT_MM + 36.0, 400)))
        points = np.concatenate((top, side))
        self.assertGreater(float(np.median(points[:, 2])), MAT_MM + 23.0 + UNDER_A_PART_MM)
        self.assertEqual(MAT_MM, tops.height_under(points))

    def test_the_ceiling_is_the_lowest_tenth_of_the_part_and_the_band(self) -> None:
        points = np.array([[0.0, 0.0, 60.0], [0.0, 0.0, 70.0], [0.0, 0.0, 200.0]])
        self.assertAlmostEqual(62.0 + UNDER_A_PART_MM, support_ceiling_mm(points))
        self.assertIsNone(support_ceiling_mm(np.zeros((0, 3))))

    def test_a_part_under_no_support_has_none(self) -> None:
        points = np.array([[900.0, 900.0, 60.0], [905.0, 905.0, 70.0]])
        self.assertIsNone(MODEL.height_under(points))


if __name__ == "__main__":
    unittest.main()
