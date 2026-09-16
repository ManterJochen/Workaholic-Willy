"""A sphere fit that covers the body, and reaches past it by no more than it was told (B6).

A collision sphere model is two errors at once, and only one of them is safe. Where the spheres do not COVER the body,
the planner clears configurations the exact mesh guard then refuses, or worse, the arm is somewhere the model says is
empty: a false clear. Where they reach PAST the body, the planner refuses configurations that are fine: a false
collide, which costs picks and nothing else.

The vendor's map and the fits tried before it optimise a compromise and measure the result afterwards. This fitter
makes the safe half a property of the construction instead: every surface sample ends up inside some sphere by at
least the sample spacing, and no sphere reaches further past the surface than the reach it was given. What is then
traded is COUNT against reach, which is a planning cost rather than a safety one.

The algorithm is tested here against bodies whose distance function is exact arithmetic, so the test measures the
fitter rather than a mesh library: a box, a plate thin enough that marching inward leaves it, and a sphere.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]


def _cover():
    path = _ROOT / "scripts" / "curobo" / "_cover_fit.py"
    spec = importlib.util.spec_from_file_location("_cover_fit_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Box:
    """An axis aligned box, with the exact unsigned distance and inside test for any point."""

    def __init__(self, half: "tuple[float, float, float]") -> None:
        self.half = np.asarray(half, dtype=np.float64)

    def distance(self, points: np.ndarray) -> np.ndarray:
        p = np.abs(np.asarray(points, dtype=np.float64)) - self.half
        outside = np.linalg.norm(np.maximum(p, 0.0), axis=1)
        inside = np.minimum(p.max(axis=1), 0.0)
        return np.abs(outside + inside)

    def inside(self, points: np.ndarray) -> np.ndarray:
        return np.all(np.abs(np.asarray(points, dtype=np.float64)) <= self.half + 1e-12, axis=1)

    def samples(self, spacing: float, seed: int = 0) -> "tuple[np.ndarray, np.ndarray]":
        """Points on the six faces on a grid of ``spacing``, with the outward normal of each."""
        rng = np.random.default_rng(seed)
        points, normals = [], []
        for axis in range(3):
            u, v = [i for i in range(3) if i != axis]
            grid_u = np.arange(-self.half[u], self.half[u] + 1e-9, spacing)
            grid_v = np.arange(-self.half[v], self.half[v] + 1e-9, spacing)
            uu, vv = np.meshgrid(grid_u, grid_v, indexing="ij")
            for sign in (-1.0, 1.0):
                block = np.zeros((uu.size, 3))
                block[:, u] = uu.ravel()
                block[:, v] = vv.ravel()
                block[:, axis] = sign * self.half[axis]
                normal = np.zeros((uu.size, 3))
                normal[:, axis] = sign
                points.append(block)
                normals.append(normal)
        points = np.vstack(points)
        normals = np.vstack(normals)
        order = rng.permutation(len(points))
        return points[order], normals[order]


class _Sphere:
    def __init__(self, radius: float) -> None:
        self.radius = float(radius)

    def distance(self, points: np.ndarray) -> np.ndarray:
        return np.abs(np.linalg.norm(np.asarray(points, dtype=np.float64), axis=1) - self.radius)

    def inside(self, points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.asarray(points, dtype=np.float64), axis=1) <= self.radius + 1e-12

    def samples(self, count: int, seed: int = 0) -> "tuple[np.ndarray, np.ndarray]":
        rng = np.random.default_rng(seed)
        directions = rng.normal(size=(count, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return directions * self.radius, directions


class TheFitCoversTheBodyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _cover()

    def _fit(self, body, points, normals, *, reach, spacing, cap=400):
        return self.module.cover_body(
            points, normals, distance=body.distance, inside=body.inside,
            reach_m=reach, spacing_m=spacing, cap=cap, seed=20260916,
        )

    def _reach_past(self, body, fit) -> float:
        """The furthest any point of any sphere's surface sits outside the body."""
        directions = self.module.unit_directions()
        worst = 0.0
        for centre, radius in zip(fit.centres_m, fit.radii_m):
            surface = centre[None, :] + radius * directions
            outside = ~body.inside(surface)
            if outside.any():
                worst = max(worst, float(body.distance(surface[outside]).max()))
        return worst

    def test_every_surface_point_ends_up_inside_a_sphere(self) -> None:
        """⭐ THE SAFE HALF, by construction rather than by measurement afterwards."""
        box = _Box((0.03, 0.02, 0.05))
        points, normals = box.samples(0.002)
        fit = self._fit(box, points, normals, reach=0.012, spacing=0.002)
        assert fit is not None

        gaps = np.min(np.linalg.norm(points[:, None, :] - fit.centres_m[None, :, :], axis=2)
                      - fit.radii_m[None, :], axis=1)
        self.assertLessEqual(float(gaps.max()), -0.002 + 1e-9,
                             "a surface point sits outside every sphere, which is a false clear waiting to happen")
        self.assertAlmostEqual(fit.uncovered_max_m, float(gaps.max()), places=9)

    def test_no_sphere_reaches_further_past_the_body_than_it_was_told(self) -> None:
        """⭐ THE OTHER HALF. Reach is what a false collide is made of, and it is the number the caller chooses."""
        box = _Box((0.03, 0.02, 0.05))
        points, normals = box.samples(0.002)
        for reach in (0.008, 0.012, 0.020):
            with self.subTest(reach_mm=reach * 1000.0):
                fit = self._fit(box, points, normals, reach=reach, spacing=0.002)
                assert fit is not None
                self.assertLessEqual(self._reach_past(box, fit), reach + 1e-6)

    def test_a_smaller_reach_costs_more_spheres(self) -> None:
        """The trade the caller is making, measured rather than asserted in prose."""
        box = _Box((0.03, 0.02, 0.05))
        points, normals = box.samples(0.002)
        counts = {}
        for reach in (0.008, 0.020):
            fit = self._fit(box, points, normals, reach=reach, spacing=0.002, cap=400)
            assert fit is not None
            counts[reach] = len(fit.radii_m)
        self.assertGreater(counts[0.008], counts[0.020], counts)

    def test_a_cap_it_cannot_meet_is_refused_rather_than_half_covered(self) -> None:
        """⭐ THE CONTROL. Half a cover is the dangerous answer: it looks like a fit and leaves holes."""
        box = _Box((0.03, 0.02, 0.05))
        points, normals = box.samples(0.002)
        # Measured: this body needs 99 spheres at 8 mm of reach, so four is a cap it cannot meet.
        self.assertIsNone(self._fit(box, points, normals, reach=0.008, spacing=0.002, cap=4))

    def test_a_reach_too_tight_for_the_sampling_is_refused_before_it_is_attempted(self) -> None:
        """⭐ THE CONTROL on the regime itself. Below twice the spacing a corner sphere covers its own seed by
        exactly zero, and floating point decides whether the body has a hole in it. Measured while writing
        this: at 6 mm of reach on a 4 mm grid, sixteen corner samples stayed uncovered after fifty rounds.
        """
        box = _Box((0.03, 0.02, 0.05))
        points, normals = box.samples(0.004)
        with self.assertRaises(ValueError) as caught:
            self._fit(box, points, normals, reach=0.006, spacing=0.004)
        self.assertIn("6.0 mm", str(caught.exception))
        self.assertIn("4.0 mm", str(caught.exception))

    def test_a_curved_body_is_covered_too(self) -> None:
        ball = _Sphere(0.05)
        points, normals = ball.samples(4000)
        fit = self._fit(ball, points, normals, reach=0.008, spacing=0.002)
        assert fit is not None
        self.assertLessEqual(self._reach_past(ball, fit), 0.008 + 1e-6)
        self.assertEqual(len(fit), 1, "a ball is one sphere, and a fitter that cannot see that is fitting noise")
        gaps = np.min(np.linalg.norm(points[:, None, :] - fit.centres_m[None, :, :], axis=2)
                      - fit.radii_m[None, :], axis=1)
        self.assertLess(float(gaps.max()), 0.0)

    def test_the_sphere_kept_is_the_LARGEST_admissible_one_not_the_deepest(self) -> None:
        """⭐ Marching inward, the distance to the body rises to the medial axis and falls again, so the
        deepest step a candidate may take is PAST the biggest sphere it could have had. Both are equally safe,
        because a centre inside at unsigned distance u carries the whole ball B(c, u) inside the body and a
        sphere of u + reach around it lies at most reach outside. What the choice costs is coverage.

        A ball of radius R is the clean case: the best centre is its own, radius R + reach. The deepest
        admissible step sits (reach - spacing) / 2 past it, where the radius is smaller by the same amount.
        Measured on the ur5e upper arm, that difference was 477 spheres against 194 at 12 mm of reach.
        """
        ball = _Sphere(0.05)
        points, normals = ball.samples(4000)

        fit = self._fit(ball, points, normals, reach=0.008, spacing=0.002)

        assert fit is not None
        self.assertGreaterEqual(
            float(fit.radii_m.max()), 0.05 + 0.008 - 0.002,
            "the march kept a sphere smaller than the body allows: it took the deepest step rather than the "
            "one with the largest inscribed ball")
        self.assertLessEqual(self._reach_past(ball, fit), 0.008 + 1e-6,
                             "and the bigger sphere still reaches no further past the body")

    def test_a_body_thinner_than_the_reach_is_still_covered(self) -> None:
        """⭐ THE CONTROL a march inward gets wrong. A plate 6 mm thick, fitted at 12 mm of reach: a centre that
        walks straight through it lands outside, where the distance GROWS with every step, and the sphere it would
        take from there covers nothing near the point it started at."""
        plate = _Box((0.04, 0.003, 0.04))
        points, normals = plate.samples(0.002)
        fit = self._fit(plate, points, normals, reach=0.012, spacing=0.002, cap=200)
        assert fit is not None

        gaps = np.min(np.linalg.norm(points[:, None, :] - fit.centres_m[None, :, :], axis=2)
                      - fit.radii_m[None, :], axis=1)
        self.assertLess(float(gaps.max()), 0.0)
        self.assertLessEqual(self._reach_past(plate, fit), 0.012 + 1e-6)

    def test_two_runs_of_one_recipe_are_the_same_spheres(self) -> None:
        """A safety artefact that differs between two runs of one recipe cannot be reviewed."""
        box = _Box((0.03, 0.02, 0.05))
        points, normals = box.samples(0.002)
        first = self._fit(box, points, normals, reach=0.012, spacing=0.002)
        second = self._fit(box, points, normals, reach=0.012, spacing=0.002)
        assert first is not None and second is not None
        np.testing.assert_array_equal(first.centres_m, second.centres_m)
        np.testing.assert_array_equal(first.radii_m, second.radii_m)


if __name__ == "__main__":
    unittest.main()
