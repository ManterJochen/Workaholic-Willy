"""The Tier-0 gate, and the proof that it can REJECT — which is the only thing that makes it a gate.

⛔ A GATE NOBODY HAS SEEN REJECT ANYTHING IS DECORATION. So every check here is run twice: against a
backend that renders the exact geometry (must pass) and against one broken in a specific, plausible way
(must fail). The broken ones are not inventions — each is a defect that has actually shipped somewhere:
a principal point off by one pixel, a depth in the wrong unit, a mask whose ids belong to a different
frame.

⭑ THE OBLIQUITY IS THE INSTRUMENT. `test_the_OVERHEAD_view_cannot_see_the_defect` is the load-bearing
test in this file: it shows the same broken backend passing perfectly when graded from straight down.
A principal-point shift cannot change recovered Z on a plane normal to the optical axis, so a contract
test written on the overhead view is not a weaker version of this gate — it is a test that cannot fail.
The repo's other two depth checks share the same blindness for a different reason: they compare a
backend against itself.
"""

from __future__ import annotations

import unittest

import numpy as np

from datagen.render.camera import project_to_pixels
from datagen.render.equivalence import (
    GEOMETRY_TOLERANCE_MM,
    check_depth_reconstructs_geometry,
    check_landmark_reconstructs,
    check_mask_matches_depth,
    check_points_match_analytic_rays,
    check_repeatable,
    oblique_camera,
    run_tier0,
)

_PLANE_Z_MM = 0.0


def _render_plane(camera_to_base: np.ndarray, intrinsics: np.ndarray,
                  resolution: tuple[int, int] = (640, 480), *,
                  plane_z_mm: float = _PLANE_Z_MM,
                  principal_offset_px: tuple[float, float] = (0.0, 0.0),
                  depth_scale: float = 1.0) -> np.ndarray:
    """An EXACT analytic depth image of the plane z = plane_z_mm, as a perfect backend would produce.

    The optional distortions are how a broken backend is simulated: `principal_offset_px` renders as if
    the sensor's centre were elsewhere (the defect the oblique view exists to catch), `depth_scale`
    returns metres-as-millimetres or similar.
    """
    width, height = resolution
    k = np.asarray(intrinsics, dtype=np.float64)
    columns, rows = np.meshgrid(np.arange(width, dtype=np.float64),
                                np.arange(height, dtype=np.float64))
    # Ray directions in CAMERA, built with the SHIFTED centre when a defect is being simulated.
    x = (columns - (k[0, 2] + principal_offset_px[0])) / k[0, 0]
    y = (rows - (k[1, 2] + principal_offset_px[1])) / k[1, 1]
    directions = np.stack([x, y, np.ones_like(x)], axis=-1)
    rotation, origin = camera_to_base[:3, :3], camera_to_base[:3, 3]
    world = directions @ rotation.T                              # (H, W, 3) in BASE
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (plane_z_mm - origin[2]) / world[..., 2]
    depth = np.where(np.isfinite(t) & (t > 0.0), t, 0.0)
    return (depth * depth_scale).astype(np.float64)


def _mask_of(depth: np.ndarray) -> np.ndarray:
    return depth > 0.0


class TheGateAcceptsAnExactBackendTests(unittest.TestCase):
    """If the harness cannot pass a perfect backend it is measuring itself, not the engine."""

    def test_an_exact_plane_render_reconstructs_to_below_the_bar(self) -> None:
        camera_to_base, k = oblique_camera()
        depth = _render_plane(camera_to_base, k)
        finding = check_depth_reconstructs_geometry(depth, _mask_of(depth), camera_to_base, k,
                                                    plane_z_mm=_PLANE_Z_MM)
        self.assertTrue(finding.passed, finding.detail)
        self.assertLess(finding.measured, GEOMETRY_TOLERANCE_MM)

    def test_an_empty_render_is_a_FAILURE_and_not_a_pass(self) -> None:
        """⚠ The degenerate case that would otherwise sail through: no pixels means max() over nothing,
        which is not zero error -- it is no evidence."""
        camera_to_base, k = oblique_camera()
        blank = np.zeros((16, 16), dtype=np.float64)
        finding = check_depth_reconstructs_geometry(blank, _mask_of(blank), camera_to_base, k,
                                                    plane_z_mm=_PLANE_Z_MM)
        self.assertFalse(finding.passed)
        self.assertIn("rendered nothing", finding.detail)


class TheGateRejectsABrokenBackendTests(unittest.TestCase):
    """⛔ THE HALF THAT MAKES IT A GATE."""

    def test_a_principal_point_off_by_ONE_PIXEL_is_caught(self) -> None:
        """MEASURED, and the direction is not arbitrary: the repo's `look_at` with up = (0,0,1) gives
        the camera an EXACTLY horizontal x-axis (R[2,0] = -0.000000), so a COLUMN-direction offset does
        not change the depth VALUE at all -- there is no defect in the image to catch. A ROW offset of
        one pixel is 1.714 mm."""
        camera_to_base, k = oblique_camera()
        depth = _render_plane(camera_to_base, k, principal_offset_px=(0.0, 1.0))
        finding = check_depth_reconstructs_geometry(depth, _mask_of(depth), camera_to_base, k,
                                                    plane_z_mm=_PLANE_Z_MM)
        self.assertFalse(finding.passed,
                         f"a one-pixel principal-point offset went unnoticed ({finding.measured:.4f} mm)")
        self.assertGreater(finding.measured, GEOMETRY_TOLERANCE_MM)

    def test_the_OVERHEAD_view_cannot_see_the_defect(self) -> None:
        """⭑ THE LOAD-BEARING TEST IN THIS FILE. The SAME broken backend, graded from straight down,
        passes perfectly -- because a principal-point shift cannot change recovered Z on a plane normal
        to the optical axis. A contract test written on the overhead view is not a weaker gate; it is a
        gate that cannot fail."""
        overhead, k = oblique_camera(eye_mm=(0.0, 0.0, 800.0), look_at_mm=(0.0, 0.0, 0.0))
        depth = _render_plane(overhead, k, principal_offset_px=(0.0, 1.0))
        finding = check_depth_reconstructs_geometry(depth, _mask_of(depth), overhead, k,
                                                    plane_z_mm=_PLANE_Z_MM)
        self.assertTrue(finding.passed, "the overhead view somehow saw it -- re-derive this claim")
        self.assertLess(finding.measured, 1e-9,
                        "the overhead error is not merely small, it is structurally zero")

    def test_a_bigger_offset_is_caught_more_strongly_still(self) -> None:
        """Monotonicity: the check must scale with the defect, not merely trip on one fixture."""
        camera_to_base, k = oblique_camera()
        errors = []
        for offset in (1.0, 4.0, 16.0):
            depth = _render_plane(camera_to_base, k, principal_offset_px=(0.0, offset))
            errors.append(check_depth_reconstructs_geometry(
                depth, _mask_of(depth), camera_to_base, k, plane_z_mm=_PLANE_Z_MM).measured)
        self.assertEqual(errors, sorted(errors), errors)

    def test_a_depth_in_the_WRONG_UNIT_is_caught(self) -> None:
        """Metres where millimetres were promised. Trivially catchable and trivially shipped."""
        camera_to_base, k = oblique_camera()
        depth = _render_plane(camera_to_base, k, depth_scale=0.001)
        finding = check_depth_reconstructs_geometry(depth, _mask_of(depth), camera_to_base, k,
                                                    plane_z_mm=_PLANE_Z_MM)
        self.assertFalse(finding.passed)

    def test_a_plane_at_the_WRONG_HEIGHT_is_caught(self) -> None:
        """The backend renders a consistent world that is not the world it was asked for."""
        camera_to_base, k = oblique_camera()
        depth = _render_plane(camera_to_base, k, plane_z_mm=5.0)
        finding = check_depth_reconstructs_geometry(depth, _mask_of(depth), camera_to_base, k,
                                                    plane_z_mm=_PLANE_Z_MM)
        self.assertFalse(finding.passed)
        self.assertAlmostEqual(finding.measured, 5.0, places=3)


class TheMaskAndTheDepthMustAgreeTests(unittest.TestCase):
    """Two independent claims a backend makes about one frame."""

    @staticmethod
    def _frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        background = np.full((32, 32), 800.0)
        depth = background.copy()
        depth[10:20, 12:24] = 700.0
        instance = np.zeros((32, 32), dtype=np.int32)
        instance[10:20, 12:24] = 3
        return instance, depth, background

    def test_an_agreeing_frame_passes(self) -> None:
        instance, depth, background = self._frame()
        self.assertTrue(check_mask_matches_depth(instance, depth, background).passed)

    def test_a_mask_SHIFTED_BY_ONE_PIXEL_is_caught(self) -> None:
        """The signature of an id space that belongs to a different frame -- the reason datagen
        abandoned Isaac's own segmentation annotator and derives masks by depth differencing."""
        instance, depth, background = self._frame()
        finding = check_mask_matches_depth(np.roll(instance, 1, axis=1), depth, background)
        self.assertFalse(finding.passed)
        self.assertLess(finding.measured, 1.0)

    def test_an_EMPTY_frame_is_a_failure_and_not_a_perfect_score(self) -> None:
        """0/0 is not 1.0. A backend that rendered nothing must not read as perfect agreement."""
        blank = np.zeros((8, 8), dtype=np.int32)
        flat = np.full((8, 8), 500.0)
        finding = check_mask_matches_depth(blank, flat, flat)
        self.assertFalse(finding.passed)
        self.assertIn("proves nothing", finding.detail)


class OnlyAWorldReferenceCatchesASelfConsistentCameraTests(unittest.TestCase):
    """THE LIMIT THAT MEASUREMENT FOUND, and it changed the design of this gate.

    A backend that renders through a shifted principal point AND reports that same shifted point is
    consistent with itself. Both other checks compute their expectation from the K the backend hands
    over, so both agree with it perfectly -- measured, 0.0000 mm at a 4 px offset in cx. You cannot
    catch a camera using only that camera's own parameters. The reference has to come from the WORLD:
    a body whose position the LAYOUT chose, not one the renderer produced.
    """

    @staticmethod
    def _landmark(resolution=(320, 240)):
        from datagen.render.camera import unproject_to_base

        camera_to_base, k = oblique_camera(resolution=resolution)
        depth = _render_plane(camera_to_base, k, resolution)
        points = unproject_to_base(depth, _mask_of(depth), camera_to_base, k)
        centre = np.array([60.0, -40.0, 0.0])
        inside = (np.abs(points[:, 0] - centre[0]) < 40.0) & (np.abs(points[:, 1] - centre[1]) < 40.0)
        rows, columns = np.nonzero(_mask_of(depth))
        mask = np.zeros_like(depth, dtype=bool)
        mask[rows[inside], columns[inside]] = True
        return depth, mask, camera_to_base, k, centre

    def test_an_exact_backend_lands_the_landmark(self) -> None:
        depth, mask, camera_to_base, k, centre = self._landmark()
        finding = check_landmark_reconstructs(depth, mask, camera_to_base, k,
                                              landmark_centre_mm=centre)
        self.assertTrue(finding.passed, finding.detail)

    def test_a_SELF_CONSISTENT_camera_error_is_caught_HERE_and_nowhere_else(self) -> None:
        depth, mask, camera_to_base, k, centre = self._landmark()
        wrong = k.copy()
        wrong[0, 2] += 1.0
        caught = check_landmark_reconstructs(depth, mask, camera_to_base, wrong,
                                             landmark_centre_mm=centre)
        self.assertFalse(caught.passed,
                         f"the world reference missed it ({caught.measured:.4f} mm)")

        # ...and the proof that the other two cannot: same defect, same frame, both silent.
        blind_ray = check_points_match_analytic_rays(depth, _mask_of(depth), camera_to_base, wrong,
                                                     plane_z_mm=_PLANE_Z_MM)
        blind_plane = check_depth_reconstructs_geometry(depth, _mask_of(depth), camera_to_base,
                                                        wrong, plane_z_mm=_PLANE_Z_MM)
        self.assertTrue(blind_ray.passed, "re-derive the claim: the ray check saw it after all")
        self.assertTrue(blind_plane.passed, "re-derive the claim: the plane check saw it after all")
        self.assertLess(blind_ray.measured, 1e-9)

    def test_it_scales_with_the_defect(self) -> None:
        depth, mask, camera_to_base, k, centre = self._landmark()
        errors = []
        for offset in (1.0, 4.0):
            wrong = k.copy()
            wrong[0, 2] += offset
            errors.append(check_landmark_reconstructs(depth, mask, camera_to_base, wrong,
                                                      landmark_centre_mm=centre).measured)
        self.assertEqual(errors, sorted(errors), errors)

    def test_a_landmark_with_no_pixels_is_a_failure(self) -> None:
        depth, _unused, camera_to_base, k, centre = self._landmark()
        finding = check_landmark_reconstructs(depth, np.zeros_like(depth, dtype=bool),
                                              camera_to_base, k, landmark_centre_mm=centre)
        self.assertFalse(finding.passed)


class RepeatabilityTests(unittest.TestCase):
    def test_a_deterministic_backend_passes(self) -> None:
        camera_to_base, k = oblique_camera(resolution=(64, 48))
        self.assertTrue(check_repeatable(lambda: _render_plane(camera_to_base, k, (64, 48))).passed)

    def test_a_backend_that_jitters_is_caught(self) -> None:
        rng = np.random.default_rng(0)
        self.assertFalse(check_repeatable(lambda: rng.normal(size=(4, 4))).passed)

    def test_a_backend_whose_SHAPE_moves_is_caught_without_broadcasting(self) -> None:
        """⚠ Subtracting a (4,4) from a (4,1) would BROADCAST and report a small number instead of a
        failure. Shape is checked before the difference."""
        sizes = iter([(4, 4), (4, 1)])
        finding = check_repeatable(lambda: np.zeros(next(sizes)))
        self.assertFalse(finding.passed)
        self.assertIn("shape moved", finding.detail)


class TheReportIsUnambiguousTests(unittest.TestCase):
    def test_one_failure_makes_the_backend_inadmissible(self) -> None:
        """No partial credit: each check answers a question with a right answer."""
        camera_to_base, k = oblique_camera()
        good = _render_plane(camera_to_base, k)
        bad = _render_plane(camera_to_base, k, principal_offset_px=(0.0, 1.0))
        findings = [
            check_depth_reconstructs_geometry(good, _mask_of(good), camera_to_base, k,
                                              plane_z_mm=_PLANE_Z_MM),
            check_depth_reconstructs_geometry(bad, _mask_of(bad), camera_to_base, k,
                                              plane_z_mm=_PLANE_Z_MM),
        ]
        admissible, report = run_tier0(findings)
        self.assertFalse(admissible)
        self.assertIn("NOT ADMISSIBLE", report)

    def test_an_all_green_run_says_so(self) -> None:
        camera_to_base, k = oblique_camera()
        good = _render_plane(camera_to_base, k)
        admissible, report = run_tier0([check_depth_reconstructs_geometry(
            good, _mask_of(good), camera_to_base, k, plane_z_mm=_PLANE_Z_MM)])
        self.assertTrue(admissible)
        self.assertIn("admissible on Tier 0", report)

    def test_the_report_is_ascii(self) -> None:
        """It is printed to a cp1252 console; a non-ASCII character crashes the run that reports it."""
        camera_to_base, k = oblique_camera()
        good = _render_plane(camera_to_base, k)
        _, report = run_tier0([check_depth_reconstructs_geometry(
            good, _mask_of(good), camera_to_base, k, plane_z_mm=_PLANE_Z_MM)])
        self.assertTrue(report.isascii(), report)


class TheHarnessAgreesWithTheRepoCameraTests(unittest.TestCase):
    """The fixture must be the repo's own convention, or this whole file grades a private world."""

    def test_the_synthetic_render_round_trips_through_project_to_pixels(self) -> None:
        camera_to_base, k = oblique_camera(resolution=(128, 96))
        depth = _render_plane(camera_to_base, k, (128, 96))
        from datagen.render.camera import unproject_to_base

        points = unproject_to_base(depth, _mask_of(depth), camera_to_base, k)
        self.assertGreater(len(points), 1000)
        pixels = project_to_pixels(points, camera_to_base, k)
        self.assertLess(float(np.nanmax(np.abs(pixels - np.round(pixels)))), 1e-6,
                        "the fixture's rays do not land on pixel centres in the repo's convention")


if __name__ == "__main__":
    unittest.main()
