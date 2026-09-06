"""Camera optics + placement math — the pure-numpy half of the tilted-rig work (no Isaac needed).

The sim cell used to author one camera hanging straight over the scene with whatever lens USD defaults
to. Neither assumption survives contact with the real cell: its two D435s are mounted off to the side
and tilted, and a D435 sees a much wider field than Isaac's default lens. These tests pin the geometry
and the optics conversions that the config now drives.
"""

from __future__ import annotations

import math
import unittest

from src.willy_sim.scene.cameras import (
    D435_RGB_HFOV_DEG,
    aperture_for_hfov,
    camera_elevation_deg,
    camera_position_for_elevation,
    hfov_deg_from_intrinsics,
    vfov_deg_from_hfov,
)


class FieldOfViewMathTests(unittest.TestCase):
    def test_aperture_and_intrinsics_are_exact_inverses(self) -> None:
        """Authoring a lens and measuring it back must agree, or "we set 69.4 deg" says nothing about
        what the renderer will actually do."""
        for hfov in (30.0, 47.0, D435_RGB_HFOV_DEG, 120.0):
            focal = 24.0
            aperture = aperture_for_hfov(focal, hfov)
            fx = 640 * focal / aperture  # exactly how USD/Isaac derive fx
            self.assertAlmostEqual(hfov_deg_from_intrinsics(fx, 640), hfov, places=9)

    def test_aperture_is_unit_agnostic(self) -> None:
        """Aperture and focal length are always expressed in the SAME unit, so a global rescale (USD's
        raw attributes vs Isaac's scaled getters) cancels. This is why the setter can read the focal
        length back from the camera without knowing which convention it is in."""
        self.assertAlmostEqual(
            aperture_for_hfov(240.0, 69.4) / aperture_for_hfov(24.0, 69.4), 10.0, places=9
        )

    def test_vertical_fov_is_derived_from_the_render_aspect_not_the_datasheet(self) -> None:
        """A D435 quotes 69.4 x 42.5 deg at 16:9; the sim renders 4:3. Taking both numbers from the
        datasheet would make them mutually inconsistent and stretch the image, so the vertical FOV
        follows from the horizontal one and the actual resolution."""
        self.assertAlmostEqual(vfov_deg_from_hfov(D435_RGB_HFOV_DEG, (640, 480)), 54.87, places=1)
        self.assertAlmostEqual(vfov_deg_from_hfov(D435_RGB_HFOV_DEG, (1920, 1080)), 42.56, places=1)
        # ...and the 16:9 derivation does land on the datasheet's own vertical figure.
        self.assertAlmostEqual(vfov_deg_from_hfov(D435_RGB_HFOV_DEG, (1920, 1080)), 42.5, delta=0.5)

    def test_vertical_fov_is_the_conservative_cone(self) -> None:
        """The coverage audit judges "in frame" with the vertical FOV so its verdict holds whatever the
        image roll is. That only works if vertical is genuinely the narrower of the two."""
        self.assertLess(vfov_deg_from_hfov(D435_RGB_HFOV_DEG, (640, 480)), D435_RGB_HFOV_DEG)

    def test_degenerate_lenses_are_rejected_not_silently_clamped(self) -> None:
        with self.assertRaises(ValueError):
            aperture_for_hfov(0.0, 60.0)
        with self.assertRaises(ValueError):
            aperture_for_hfov(24.0, 180.0)  # tan(90 deg) -> infinite aperture
        with self.assertRaises(ValueError):
            hfov_deg_from_intrinsics(0.0, 640)


class TiltedPlacementTests(unittest.TestCase):
    def test_placement_and_elevation_are_exact_inverses(self) -> None:
        aim = (350.0, 0.0, 50.0)
        for elevation in (20.0, 45.0, 70.0, 89.0):
            pos = camera_position_for_elevation(
                aim, elevation_deg=elevation, azimuth_deg=-90.0, distance_mm=900.0
            )
            self.assertAlmostEqual(camera_elevation_deg(pos, aim), elevation, places=9)

    def test_elevation_is_measured_from_the_table_plane(self) -> None:
        """A rig is described on site as an angle up from the table ("about 70 degrees from the table
        edge"), not as a deviation from vertical. 90 deg is straight down; 0 deg is horizontal."""
        aim = (350.0, 0.0, 50.0)
        nadir = camera_position_for_elevation(aim, elevation_deg=90.0, azimuth_deg=0.0, distance_mm=900.0)
        self.assertAlmostEqual(nadir[0], aim[0], places=6)      # directly above the aim point
        self.assertAlmostEqual(nadir[1], aim[1], places=6)
        self.assertAlmostEqual(nadir[2], aim[2] + 900.0, places=6)
        self.assertAlmostEqual(camera_elevation_deg((350.0, -900.0, 50.0), aim), 0.0, places=6)

    def test_azimuth_places_a_pair_on_opposite_sides(self) -> None:
        """The two-sided rig: whichever side the arm reaches in from, the other camera still sees."""
        aim = (350.0, 0.0, 50.0)
        left = camera_position_for_elevation(aim, elevation_deg=70.0, azimuth_deg=-90.0, distance_mm=900.0)
        right = camera_position_for_elevation(aim, elevation_deg=70.0, azimuth_deg=90.0, distance_mm=900.0)
        self.assertLess(left[1], aim[1])
        self.assertGreater(right[1], aim[1])
        self.assertAlmostEqual(left[1], -right[1], places=6)
        self.assertAlmostEqual(left[2], right[2], places=6)
        # ...and the stand-off is honoured exactly.
        self.assertAlmostEqual(
            math.dist(left, aim), 900.0, places=6,
        )

    def test_a_nadir_camera_reports_90_degrees(self) -> None:
        self.assertEqual(camera_elevation_deg((450.0, 0.0, 1000.0), (450.0, 0.0, 50.0)), 90.0)

    def test_zero_distance_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            camera_position_for_elevation((0.0, 0.0, 0.0), elevation_deg=70.0, azimuth_deg=0.0, distance_mm=0.0)


if __name__ == "__main__":
    unittest.main()
