"""Where the coarse grid finds few grasps, SFE asks every orientation the hand may grip in (the owner, 2026-10-06: "die
Orientierung feiner ... den kompletten Freiheitsgrad", and "sollten wir dann nicht noch die geneigte Schliessachse
nutzen?").

The coarse grid is the footprint's two face axes (a round one's fan every 30 degrees), and the approach tilted about the
closing axis every 15 degrees. Where it finds fewer than three grasps, the fine grid runs after it:

* the approach tilted every 7.5 degrees;
* each face's closing axis turned 10 and 20 degrees either way about the vertical, a round part's fan every 15 degrees;
* each face's closing axis tilted 10 and 20 degrees out of the horizontal, the hand rolled about its binormal so one
  finger stands higher than the other, on the coarse tilts.

A pad then closes off a vertical face's normal by the turn or the roll, inside the friction cone (26.6 degrees at the
shipped 0.5), and the score weighs the cone's slack, so a face's own axis still ranks first. A line tilted out of the
horizontal leaves the part through its top or its foot as well as a side, and its pads end there. Pinned here on the
owner's Hand-E and a 40 mm box on the mat's reading, the fine grid forced on where it must be shown.
"""

from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np

from src.robot.grasping.generation import support_footprint as sf
from tests.test_a_grasp_keeps_the_guards_distance_from_the_support_it_holds import READING_MM, _box_cloud, _jaw

#: The friction cone's half angle at the shipped 0.5, degrees.
CONE_DEG = math.degrees(math.atan(0.5))


def _grasps(*, fine: bool, cloud: "np.ndarray | None" = None) -> list:
    """The 40 mm box's grasps, the fine grid forced on where ``fine``, else as SFE decides."""
    cloud = _box_cloud(40.0) if cloud is None else cloud
    with mock.patch.object(sf, "_FINE_BELOW", 10_000 if fine else sf._FINE_BELOW):
        return sf.generate_support_footprint_grasps(cloud, support_height_mm=READING_MM, jaw=_jaw(),
                                                    max_candidates=400)


def _turn_deg(candidate: object) -> float:
    """How far the closing axis turns about the vertical off the nearest of the box's face axes (x and y here)."""
    axis = np.asarray(candidate.closing_axis, dtype=np.float64)  # type: ignore[attr-defined]
    angle = math.degrees(math.atan2(abs(axis[1]), abs(axis[0])))
    return min(angle, 90.0 - angle)


def _roll_deg(candidate: object) -> float:
    """How far the closing axis tilts out of the horizontal."""
    return math.degrees(math.asin(min(1.0, abs(float(candidate.closing_axis[2])))))  # type: ignore[attr-defined]


class TheFineGridRunsWhereTheCoarseFindsFewTests(unittest.TestCase):
    def test_a_box_the_coarse_grid_grips_keeps_the_coarse_grids_grasps(self) -> None:
        found = _grasps(fine=False)
        self.assertGreaterEqual(len(found), sf._FINE_BELOW)
        self.assertTrue(all(_roll_deg(c) < 0.01 and _turn_deg(c) < 0.01 for c in found),
                        "the fine grid ran beside grasps enough")

    def test_forced_on_it_turns_and_rolls_the_closing_axis_inside_the_cone(self) -> None:
        found = _grasps(fine=True)
        turned = [c for c in found if _turn_deg(c) > 5.0]
        rolled = [c for c in found if _roll_deg(c) > 5.0]
        self.assertTrue(turned, "no closing axis turned about the vertical")
        self.assertTrue(rolled, "no closing axis tilted out of the horizontal")
        for candidate in found:
            self.assertLessEqual(math.degrees(candidate.contact_angle_rad), CONE_DEG + 1e-6)
        self.assertEqual({10, 20}, {round(_turn_deg(c)) for c in turned})
        self.assertEqual({10, 20}, {round(_roll_deg(c)) for c in rolled})

    def test_a_face_axis_still_ranks_first(self) -> None:
        found = _grasps(fine=True)
        self.assertLess(_turn_deg(found[0]), 0.01)
        self.assertLess(_roll_deg(found[0]), 0.01)

    def test_a_rolled_grasp_keeps_both_fingers_over_the_support(self) -> None:
        jaw = _jaw()
        for candidate in [c for c in _grasps(fine=True) if _roll_deg(c) > 5.0]:
            with self.subTest(roll=round(_roll_deg(candidate)), z=round(float(candidate.position_mm[2]), 1)):
                self.assertGreaterEqual(candidate.clearance_mm, jaw.table_clearance_mm - 1e-6)


class TheRolledHandTests(unittest.TestCase):
    def test_a_roll_tilts_the_closing_axis_and_keeps_the_frame_orthonormal(self) -> None:
        axis = np.array([1.0, 0.0, 0.0])
        approach = np.array([0.0, 0.0, -1.0])
        for roll in (10.0, -20.0):
            with self.subTest(roll=roll):
                turned_axis, turned_approach = sf._rolled(axis, approach, roll)
                self.assertAlmostEqual(abs(float(turned_axis[2])), math.sin(math.radians(abs(roll))), places=9)
                self.assertAlmostEqual(float(turned_axis @ turned_approach), 0.0, places=9)
                self.assertAlmostEqual(float(np.linalg.norm(turned_axis)), 1.0, places=9)
                self.assertAlmostEqual(float(np.linalg.norm(turned_approach)), 1.0, places=9)
        same_axis, same_approach = sf._rolled(axis, approach, 0.0)
        np.testing.assert_array_equal(axis, same_axis)
        np.testing.assert_array_equal(approach, same_approach)

    def test_the_pads_of_an_inclined_line_end_at_the_parts_top(self) -> None:
        """A closing line tilted 20 degrees through an anchor 4 mm under the box's top leaves the box through the top on
        its rising side: its contact there lies on the top, not past it on the side's plane."""
        prism = sf.reconstruct_support_prism(_box_cloud(40.0), support_height_mm=READING_MM)
        assert prism is not None
        axis, approach = sf._rolled(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), 20.0)
        binormal = np.cross(approach, axis)
        anchor = np.array([0.0, -650.0, prism.z1 - 4.0])
        contacts = sf._pad_contacts(prism, anchor, axis, approach, binormal / np.linalg.norm(binormal), _jaw())
        assert contacts is not None
        _enter, _exit, contact_b, contact_a, _low = contacts
        for contact in (contact_a, contact_b):
            self.assertLessEqual(float(contact[2]), prism.z1 + 1e-6)
            self.assertGreaterEqual(float(contact[2]), prism.z0 - 1e-6)


if __name__ == "__main__":
    unittest.main()
