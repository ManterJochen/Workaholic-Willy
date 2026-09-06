"""The silhouette candidate's width must belong to the axis it actually closes on.

⛔ THE DEFECT. `_candidate_generator` assigned `width_mm = extent_minor_px` UNCONDITIONALLY, above the
branch that picks the closing axis. That branch closes across the MINOR direction for an elongated
silhouette — correct, and the reason the branch was added — but along the PRINCIPAL (MAJOR) direction
for a near-square one. The two contacts are then placed at `centre ± closing_axis * width/2` and
shipped as `distance_mm = width_mm`, so on a near-square mask BOTH CONTACTS LAND INSIDE THE OBJECT and
the commanded width understates the span the gripper must close.

MEASURED on 503 ground-truth instance masks from 30 `v1_proof` scenes with the real `MaskAnalyzer`:
**40.0 %** take the near-square branch, and on those the major-minor extent gap is a median 10.7 px —
15.0 % of the major extent. The silhouette rung's own numbers agree: MAE 10.72 mm against the true
span, bias **−7.50 mm**, and **72.6 %** of candidates commanded NARROWER than the object, against
39.4 % on the `sfe_fused` rung.

It survived because the elongated branch was added later and correctly, and its comment recorded that
near-square "keeps the principal-axis closing byte-identical" — byte-identical to a legacy pairing
that was already mismatched.

⚠ REACHABILITY: this fires only when the support-footprint stage stands down, which needs a
CAMERA→BASE transform and a BASE-frame support plane. That is exactly a cell mid-bring-up — the
September state — which is why an off-the-shipped-path defect is still worth two lines.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.generation._candidate_generator import _ELONGATED_ASPECT_MAX
from src.robot.grasping.generation.calculator import GraspCalculator
from src.robot.grasping.generation.mask_analyzer import MaskAnalyzer


def _rect_mask(width_px: int, height_px: int, canvas: int = 200) -> np.ndarray:
    """An axis-aligned filled rectangle, so the PCA extents are exactly the side lengths."""
    mask = np.zeros((canvas, canvas), dtype=bool)
    cy, cx = canvas // 2, canvas // 2
    mask[cy - height_px // 2:cy + height_px // 2, cx - width_px // 2:cx + width_px // 2] = True
    return mask


def _analysis(width_px: int, height_px: int):
    found = MaskAnalyzer().analyze(_rect_mask(width_px, height_px))
    assert found is not None, "the fixture produced no analysable mask"
    return found


class TheExtentsAreWhatWeThinkTests(unittest.TestCase):
    """If the fixture's own geometry were wrong, every assertion below would be about nothing."""

    def test_a_long_rectangle_has_the_expected_major_and_minor(self) -> None:
        found = _analysis(120, 30)
        self.assertGreater(found.extent_major_px, found.extent_minor_px)
        self.assertAlmostEqual(found.extent_major_px, 120, delta=6)
        self.assertAlmostEqual(found.extent_minor_px, 30, delta=6)

    def test_a_near_square_lands_ABOVE_the_elongated_threshold(self) -> None:
        found = _analysis(100, 88)
        aspect = found.extent_minor_px / found.extent_major_px
        self.assertGreater(aspect, _ELONGATED_ASPECT_MAX,
                           "this fixture takes the elongated branch and proves nothing")

    def test_a_long_rectangle_lands_BELOW_it(self) -> None:
        found = _analysis(120, 30)
        self.assertLessEqual(found.extent_minor_px / found.extent_major_px, _ELONGATED_ASPECT_MAX)


class TheWidthFollowsTheClosingAxisTests(unittest.TestCase):
    """The property, asserted on the real generator through a full candidate build."""

    @staticmethod
    def _candidate_width(width_px: int, height_px: int, *, min_mm: float = 0.0,
                         max_mm: float = 10_000.0) -> tuple[float, float, float]:
        """``(commanded width, major extent, minor extent)``, through the REAL generator.

        Built via `GraspCalculator` rather than by hand, so every other knob is the shipped default
        and the only thing under test is which EXTENT becomes the width. `pixel_to_mm=1.0` makes the
        two units comparable directly.
        """
        found = _analysis(width_px, height_px)
        depth = np.full((200, 200), 500.0, dtype=np.float64)
        camera = np.array([[500.0, 0.0, 100.0], [0.0, 500.0, 100.0], [0.0, 0.0, 1.0]])
        calculator = GraspCalculator(camera_matrix=camera,
                                     min_grip_width_mm=min_mm, max_grip_width_mm=max_mm)
        poses = calculator._generator.silhouette_contact_poses(   # noqa: SLF001 - the method under test
            analysis=found, depth_map=depth, median_depth_mm=500.0, pixel_to_mm=1.0,
            point_3d_cam=None, scale_to_mm=1.0, depth_confidence=1.0, up_cam=None,
        )
        widths = {round(float(pose.grip_width_mm), 3) for pose in poses}
        commanded = float(next(iter(widths))) if widths else float("nan")
        return commanded, float(found.extent_major_px), float(found.extent_minor_px)

    def test_a_NEAR_SQUARE_mask_commands_the_MAJOR_extent(self) -> None:
        """⛔ THE BUG. It closes along the principal (major) axis, so the width must be the major
        extent. It used to command the minor one, putting both contacts inside the object."""
        commanded, major, minor = self._candidate_width(100, 88)
        self.assertAlmostEqual(commanded, major, delta=2.0,
                               msg=f"commanded {commanded:.1f}, major {major:.1f}, minor {minor:.1f}")
        self.assertNotAlmostEqual(commanded, minor, delta=2.0)

    def test_an_ELONGATED_mask_still_commands_the_MINOR_extent(self) -> None:
        """The branch that was already right must be byte-identical: it closes ACROSS the short side,
        which is the whole reason that branch exists (an on-edge box regressed 5/5 -> 0/5 without it)."""
        commanded, major, minor = self._candidate_width(120, 30)
        self.assertAlmostEqual(commanded, minor, delta=2.0,
                               msg=f"commanded {commanded:.1f}, major {major:.1f}, minor {minor:.1f}")
        self.assertNotAlmostEqual(commanded, major, delta=2.0)

    def test_a_PERFECT_square_is_unchanged_either_way(self) -> None:
        """Where the two extents coincide the repair cannot be observed — stated so a reader does not
        take a passing square as evidence."""
        commanded, major, minor = self._candidate_width(90, 90)
        self.assertAlmostEqual(major, minor, delta=3.0)
        self.assertAlmostEqual(commanded, major, delta=3.0)

    def test_the_commanded_width_is_never_NARROWER_than_the_span_it_closes(self) -> None:
        """The failure in one sentence: a width narrower than the object means the fingers are
        commanded to a separation the object does not permit, and the contacts sit inside it."""
        for width_px, height_px in ((100, 88), (120, 30), (90, 90), (140, 120), (60, 20)):
            with self.subTest(size=(width_px, height_px)):
                commanded, major, minor = self._candidate_width(width_px, height_px)
                closes_on = major if (minor / major) > _ELONGATED_ASPECT_MAX else minor
                self.assertGreaterEqual(commanded, closes_on - 2.0,
                                        f"commanded {commanded:.1f} < span {closes_on:.1f}")


class TheAperturePathTests(unittest.TestCase):
    """⚠ THE CONSEQUENCE, AND IT CAUGHT THIS TEST FIRST. The first version asserted that a near-square
    object wider than the jaw would COME BACK with the major span commanded. It does not -- the
    aperture filter now refuses it outright ("all 3 synthetic pair(s) fell outside the grip range
    (width 99.0 mm, allowed 5.0-95.0 mm)"), because the span really is 99 mm and the jaw really does
    open 95. That refusal IS the repair: before it, the same object produced a candidate at the 88 mm
    MINOR extent, which fits the jaw and drives the fingers into the object."""

    def test_an_object_wider_than_the_jaw_is_REFUSED_rather_than_commanded_too_narrow(self) -> None:
        commanded, major, minor = TheWidthFollowsTheClosingAxisTests._candidate_width(
            100, 88, min_mm=5.0, max_mm=95.0)          # aperture BETWEEN the two extents
        self.assertTrue(np.isnan(commanded),
                        f"a {major:.0f} mm span came back for a 95 mm jaw as {commanded:.1f} mm")
        self.assertLess(minor, 95.0, "the fixture no longer straddles the aperture")
        self.assertGreater(major, 95.0, "the fixture no longer straddles the aperture")

    def test_the_SAME_object_is_accepted_once_the_jaw_can_actually_span_it(self) -> None:
        """Otherwise the test above would pass for a generator that refuses everything."""
        commanded, major, _minor = TheWidthFollowsTheClosingAxisTests._candidate_width(
            100, 88, min_mm=5.0, max_mm=120.0)
        self.assertFalse(np.isnan(commanded), "nothing came back even with a jaw wide enough")
        self.assertAlmostEqual(commanded, major, delta=2.0)


if __name__ == "__main__":
    unittest.main()
