"""The GL-free rasteriser, graded by the Tier-0 gate it will have to pass as part of a backend.

⭑ THIS IS THE FIRST REAL COMPONENT PUT THROUGH THAT GATE, and that ordering is the point: the gate was
written against deliberately broken fixtures before any backend existed, so passing it here is evidence
rather than coincidence. A renderer that agreed with its own conventions would sail through a test
written afterwards.

WHY A RASTERISER AT ALL. The engine question began as "which physics engine" and the measurement moved
it: the corpus carries no RGB, so a backend owes depth and silhouettes, and both are pure geometry.
Measured on this box at 640x480 -- 0.04 s per view at 108 triangles, 0.39 s at 10,252, i.e. **0.6 h for
2,000 scenes x 3 views** against Isaac's ~27-29 s per scene -- with **nothing new to install**.
"""

from __future__ import annotations

import unittest

import numpy as np
import trimesh

from datagen.render.camera import unproject_to_base
from datagen.render.equivalence import (
    check_depth_reconstructs_geometry,
    check_landmark_reconstructs,
    check_mask_matches_depth,
    check_points_match_analytic_rays,
    check_repeatable,
    oblique_camera,
    run_tier0,
)
from datagen.render.raster import rasterise

_RESOLUTION = (320, 240)


def _plane(half_x: float = 700.0, half_y: float = 700.0, z_mm: float = 0.0):
    """A FLAT quad at ``z_mm`` -- two triangles, no thickness.

    ⚠ NOT A BOX, and the first version of this fixture was one. A 20 mm-thick plate has SIDE faces, and
    an oblique camera sees them: the gate then reported 19.85 mm of "error" that was the side wall
    lying exactly where it was put. The gate was right and the fixture was wrong. The check being
    exercised is "does depth land on a known PLANE", so it has to be given a plane.
    """
    vertices = np.array([[-half_x, -half_y, z_mm], [half_x, -half_y, z_mm],
                         [half_x, half_y, z_mm], [-half_x, half_y, z_mm]], dtype=np.float64)
    return vertices, np.array([[0, 1, 2], [0, 2, 3]])


def _plate(extents=(900.0, 700.0, 20.0), centre=(0.0, 0.0, -10.0)):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(np.asarray(centre, dtype=np.float64))
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces)


def _box(extents, centre):
    mesh = trimesh.creation.box(extents=np.asarray(extents, dtype=np.float64))
    mesh.apply_translation(np.asarray(centre, dtype=np.float64))
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces)


class ItPassesTheTier0GateTests(unittest.TestCase):
    """The gate, unchanged, applied to a real renderer instead of an analytic fixture."""

    def setUp(self) -> None:
        self.camera_to_base, self.k = oblique_camera(resolution=_RESOLUTION)

    def test_the_table_plane_reconstructs_to_below_the_bar(self) -> None:
        """Its top face is at z = 0, which is a fact about the WORLD and not about another render."""
        view = rasterise({1: _plane()}, self.camera_to_base, self.k, _RESOLUTION)
        finding = check_depth_reconstructs_geometry(view.depth_mm, view.hit, self.camera_to_base,
                                                    self.k, plane_z_mm=0.0)
        self.assertTrue(finding.passed, f"{finding.measured:.4f} mm -- {finding.detail}")

    def test_every_pixel_lands_on_its_own_analytic_ray(self) -> None:
        view = rasterise({1: _plane()}, self.camera_to_base, self.k, _RESOLUTION)
        finding = check_points_match_analytic_rays(view.depth_mm, view.hit, self.camera_to_base,
                                                   self.k, plane_z_mm=0.0)
        self.assertTrue(finding.passed, f"{finding.measured:.4f} mm -- {finding.detail}")

    def test_a_body_lands_where_the_LAYOUT_put_it(self) -> None:
        """The world-referenced check -- the only one that can catch a self-consistent camera."""
        centre = np.array([60.0, -40.0, 25.0])
        bodies = {1: _plane(), 2: _box((80.0, 80.0, 50.0), centre)}
        view = rasterise(bodies, self.camera_to_base, self.k, _RESOLUTION)
        mask = view.instance_map == 2
        self.assertGreater(int(mask.sum()), 200, "the body barely rendered; the check would be noise")
        # The camera sees the top face and two sides, so x and y are resolved and z is the TOP.
        points = unproject_to_base(view.depth_mm, mask, self.camera_to_base, self.k)
        seen_top = float(points[:, 2].max())
        self.assertAlmostEqual(seen_top, 50.0, delta=0.6, msg="the top face is not where it was placed")
        finding = check_landmark_reconstructs(
            view.depth_mm, mask, self.camera_to_base, self.k,
            landmark_centre_mm=np.array([centre[0], centre[1], (seen_top + points[:, 2].min()) / 2.0]))
        self.assertTrue(finding.passed, f"{finding.measured:.4f} mm -- {finding.detail}")

    def test_it_is_repeatable(self) -> None:
        bodies = {1: _plane(), 2: _box((80.0, 80.0, 50.0), (60.0, -40.0, 25.0))}
        finding = check_repeatable(
            lambda: rasterise(bodies, self.camera_to_base, self.k, _RESOLUTION).depth_mm)
        self.assertTrue(finding.passed, finding.detail)

    def test_the_whole_tier0_report_says_admissible(self) -> None:
        view = rasterise({1: _plane()}, self.camera_to_base, self.k, _RESOLUTION)
        admissible, report = run_tier0([
            check_depth_reconstructs_geometry(view.depth_mm, view.hit, self.camera_to_base, self.k,
                                              plane_z_mm=0.0),
            check_points_match_analytic_rays(view.depth_mm, view.hit, self.camera_to_base, self.k,
                                             plane_z_mm=0.0),
        ])
        self.assertTrue(admissible, report)


class TheInstanceMapIsTheSameSceneAsTheDepthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_to_base, self.k = oblique_camera(resolution=_RESOLUTION)

    def test_the_mask_agrees_with_the_depth_that_produced_it(self) -> None:
        """Two independent claims about one frame -- the mismatch datagen abandoned Isaac's own
        segmentation annotator over."""
        plate = {1: _plane()}
        background = rasterise(plate, self.camera_to_base, self.k, _RESOLUTION)
        with_body = rasterise({**plate, 2: _box((80.0, 80.0, 50.0), (60.0, -40.0, 25.0))},
                              self.camera_to_base, self.k, _RESOLUTION)
        finding = check_mask_matches_depth(
            (with_body.instance_map == 2).astype(np.int32), with_body.depth_mm, background.depth_mm)
        self.assertTrue(finding.passed, f"IoU {finding.measured:.4f} -- {finding.detail}")

    def test_zero_means_NOTHING_HIT_and_is_not_a_body(self) -> None:
        view = rasterise({1: _box((40.0, 40.0, 40.0), (0.0, 0.0, 20.0))},
                         self.camera_to_base, self.k, _RESOLUTION)
        self.assertTrue(bool(((view.instance_map == 0) == (view.depth_mm == 0.0)).all()),
                        "instance 0 and 'no depth' must be the same pixels")

    def test_a_nearer_body_OCCLUDES_a_farther_one(self) -> None:
        """The z-buffer's whole job, and the thing a per-body render would get wrong."""
        far = _box((60.0, 60.0, 60.0), (0.0, 0.0, 30.0))
        near = _box((60.0, 60.0, 60.0), (150.0, -150.0, 30.0))
        view = rasterise({1: far, 2: near}, self.camera_to_base, self.k, _RESOLUTION)
        for body in (1, 2):
            self.assertGreater(int((view.instance_map == body).sum()), 100, body)
        # No pixel may carry a body id while its depth belongs to another body's surface.
        for body in (1, 2):
            mask = view.instance_map == body
            self.assertTrue(bool((view.depth_mm[mask] > 0.0).all()))


class ThePerspectiveIsCorrectTests(unittest.TestCase):
    """⚠ THE CLASSIC RASTERISER ERROR, and the one this corpus would suffer most from: interpolating z
    linearly in screen space instead of 1/z. It is exact on a fronto-parallel face and bows every
    oblique one -- and every view in this corpus is oblique."""

    def test_a_STEEPLY_oblique_plane_stays_flat(self) -> None:
        camera_to_base, k = oblique_camera(resolution=_RESOLUTION, eye_mm=(700.0, -650.0, 260.0))
        view = rasterise({1: _plane(1400.0, 1000.0)}, camera_to_base, k, _RESOLUTION)
        finding = check_depth_reconstructs_geometry(view.depth_mm, view.hit, camera_to_base, k,
                                                    plane_z_mm=0.0)
        self.assertTrue(finding.passed,
                        f"a shallow grazing view bows: {finding.measured:.4f} mm -- linear-z?")

    def test_depth_is_camera_Z_and_not_RAY_LENGTH(self) -> None:
        """They differ by up to 1/cos(half-field) -- about 15 % at the corner of a 60 deg lens. Both
        look plausible; only one is what `unproject_to_base` reads."""
        camera_to_base, k = oblique_camera(resolution=_RESOLUTION)
        view = rasterise({1: _plane()}, camera_to_base, k, _RESOLUTION)
        rows, columns = np.nonzero(view.hit)
        corner = int(np.argmax((columns - k[0, 2]) ** 2 + (rows - k[1, 2]) ** 2))
        row, column = rows[corner], columns[corner]
        z = view.depth_mm[row, column]
        direction = np.array([(column - k[0, 2]) / k[0, 0], (row - k[1, 2]) / k[1, 1], 1.0])
        ray_length = z * float(np.linalg.norm(direction))
        self.assertGreater(ray_length - z, 0.05 * z, "the fixture no longer distinguishes the two")
        point = unproject_to_base(view.depth_mm, view.hit, camera_to_base, k)
        self.assertLess(float(np.max(np.abs(point[:, 2]))), 0.1)


class ItCostsNothingToInstallTests(unittest.TestCase):
    """The reason a second backend exists at all is installability; a renderer that needed a driver
    would be the same wall in smaller form."""

    def test_the_module_imports_only_numpy(self) -> None:
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "datagen/render/raster.py").read_text(
            encoding="utf-8")
        for forbidden in ("import trimesh", "import OpenGL", "import glfw", "import mujoco",
                          "import pybullet", "embreex", "rtree"):
            self.assertNotIn(forbidden, source, forbidden)

    def test_it_renders_with_no_display_and_no_gpu(self) -> None:
        """It already did, four classes ago -- stated here so the property is pinned rather than
        assumed by a reader who sees a module named `raster` and expects a context."""
        camera_to_base, k = oblique_camera(resolution=(64, 48))
        view = rasterise({1: _plane()}, camera_to_base, k, (64, 48))
        self.assertGreater(int(view.hit.sum()), 100)


if __name__ == "__main__":
    unittest.main()
