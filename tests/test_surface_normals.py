"""`estimate_surface_normals` — the batched path, held against the per-point definition.

⚠ WHY THIS FILE EXISTS. This function is used by the corpus builder, by the analytic contact search
and by the learned generator's runtime, and until now **one** test file touched it at all. It was also
the single largest cost in the runtime cloud build: profiled on a VGA frame it was **91 %** of it
(0.645 s of 0.709 s), because `_pca_normal` ran once per point from Python — 9,451 times, each doing a
3×3 `eigh`.

It is batched now. `_pca_normal` and `_sorted_limited_neighbors` were NOT deleted: they are the
readable statement of what the batch computes, and the tests below use them as the ORACLE. That is
what keeps a vectorised rewrite from drifting away from its own definition — and it is the reason
keeping them is not dead code.

MEASURED on 308,324 points from real corpus clouds, both orientation modes: valid masks identical,
**zero** normals disagreeing by more than a degree (worst 0.174°), confidence and curvature within
1.5e-05 — float32 rounding from a different summation order. 13.5 s → 2.5 s, a 5.5× speedup, and the
runtime cloud build went 423 ms → 150 ms.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.geometry._spatial import RadiusIndex
from src.robot.grasping.geometry.normals import (
    NormalEstimationConfig,
    _pca_normal,
    _sorted_limited_neighbors,
    estimate_surface_normals,
)


def _reference(points: np.ndarray, cfg: NormalEstimationConfig):
    """The per-point definition, composed from the helpers the batch replaced.

    Deliberately a transcription of the loop that used to live in `estimate_surface_normals`, so a
    disagreement here means the batch changed the ANSWER rather than the arithmetic order.
    """
    n = len(points)
    normals = np.zeros((n, 3), dtype=np.float64)
    curvature = np.full((n,), np.inf)
    valid = np.zeros((n,), dtype=bool)
    index = RadiusIndex(points)
    camera = np.asarray(cfg.camera_position_mm, dtype=np.float64)
    for i, point in enumerate(points):
        near = index.query_radius(point, cfg.radius_mm)
        if near.size < cfg.min_neighbors:
            continue
        near = _sorted_limited_neighbors(points, point, near, cfg.max_neighbors)
        if near.size < cfg.min_neighbors:
            continue
        try:
            normal, local = _pca_normal(points[near])
        except ValueError:
            continue
        if cfg.max_curvature is not None and local > cfg.max_curvature:
            continue
        if cfg.orient_towards_camera and float(np.dot(normal, camera - point)) < 0.0:
            normal = -normal
        normals[i], curvature[i], valid[i] = normal, local, True
    return normals, curvature, valid


def _surface(rows: int = 40, noise: float = 0.3, seed: int = 0) -> np.ndarray:
    """A noisy curved sheet — what a depth frame actually gives, unlike a perfect lattice."""
    rng = np.random.default_rng(seed)
    u, v = np.meshgrid(np.linspace(-50, 50, rows), np.linspace(-50, 50, rows))
    z = np.sin(u / 18.0) * 7.0 + rng.normal(0.0, noise, u.shape)
    return np.column_stack([u.ravel(), v.ravel(), z.ravel()])


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between two sets of LINES — a normal and its negation are the same normal."""
    dot = np.abs(np.einsum("ij,ij->i", a.astype(np.float64), b.astype(np.float64)))
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


class AgreesWithTheDefinitionTests(unittest.TestCase):
    def _check(self, points: np.ndarray, **kw) -> None:
        cfg = NormalEstimationConfig(**kw)
        got = estimate_surface_normals(points, cfg)
        want_n, want_k, want_v = _reference(points, cfg)
        np.testing.assert_array_equal(got.valid_mask, want_v, "a different set of points is valid")
        both = got.valid_mask & want_v
        if not both.any():
            return
        worst = float(_angle_deg(got.normals[both], want_n[both]).max())
        self.assertLess(worst, 1.0, f"batched normals differ from the definition by {worst:.3f} deg")
        np.testing.assert_allclose(got.curvature[both], want_k[both], atol=1e-4)

    def test_a_noisy_surface(self) -> None:
        self._check(_surface(), radius_mm=15.0)

    def test_with_the_camera_orientation_the_corpus_and_runtime_both_use(self) -> None:
        self._check(_surface(), radius_mm=15.0, orient_towards_camera=True,
                    camera_position_mm=(0.0, 0.0, 900.0))

    def test_a_curvature_ceiling_rejects_the_same_points(self) -> None:
        """The batch turns a `continue` into a mask; it must reject exactly what the loop rejected."""
        self._check(_surface(noise=3.0), radius_mm=15.0, max_curvature=0.05)

    def test_a_tight_radius_that_starves_most_points(self) -> None:
        """`min_neighbors` is checked twice in the loop and once in the batch; they must agree."""
        self._check(_surface(), radius_mm=3.5)

    def test_a_sparse_cloud_where_nothing_qualifies(self) -> None:
        rng = np.random.default_rng(1)
        self._check(rng.normal(size=(40, 3)) * 500.0, radius_mm=5.0)


class NeighbourTableTests(unittest.TestCase):
    """The batched query has to return what the per-point query plus a sort plus a cap returned."""

    def test_it_matches_the_single_point_query(self) -> None:
        points = _surface(rows=25)
        index = RadiusIndex(points)
        table, live = index.neighbour_table(12.0, 16)
        for i in (0, 7, 123, len(points) - 1):
            reference = _sorted_limited_neighbors(points, points[i],
                                                  index.query_radius(points[i], 12.0), 16)
            np.testing.assert_array_equal(np.sort(table[i][live[i]]), np.sort(reference))

    def test_padding_never_reaches_a_computation(self) -> None:
        """Padding is index 0 so the gather stays a plain fancy-index; `live` is what excludes it."""
        points = _surface(rows=12)
        table, live = RadiusIndex(points).neighbour_table(4.0, 64)
        self.assertTrue((table[~live] == 0).all())
        self.assertLess(int(live.sum()), live.size, "the fixture must actually produce padding")

    def test_every_live_neighbour_is_inside_the_radius(self) -> None:
        points = _surface(rows=20)
        table, live = RadiusIndex(points).neighbour_table(10.0, 32)
        distance = np.linalg.norm(points[table] - points[:, None, :], axis=2)
        self.assertLessEqual(float(distance[live].max()), 10.0 + 1e-9)

    def test_the_NUMPY_FALLBACK_agrees_with_the_scipy_path(self) -> None:
        """SciPy is an accelerator, not a dependency. Two back-ends that disagree would make the
        corpus depend on what happened to be installed."""
        points = _surface(rows=18)
        fast = RadiusIndex(points)
        slow = RadiusIndex(points)
        slow._tree = None
        a_table, a_live = fast.neighbour_table(11.0, 24)
        b_table, b_live = slow.neighbour_table(11.0, 24)
        np.testing.assert_array_equal(a_live.sum(axis=1), b_live.sum(axis=1))
        for i in range(0, len(points), 37):
            np.testing.assert_array_equal(np.sort(a_table[i][a_live[i]]),
                                          np.sort(b_table[i][b_live[i]]))


class AssumptionTests(unittest.TestCase):
    def test_eigh_returns_ASCENDING_eigenvalues(self) -> None:
        """The batch drops the `argsort` the per-point version did, on this contract alone — and that
        sort cost 0.082 s of a 0.477 s call. If numpy ever stops guaranteeing it, the smallest
        principal axis stops being column 0 and every normal silently becomes a tangent."""
        rng = np.random.default_rng(0)
        blocks = rng.normal(size=(20000, 8, 3))
        blocks[::7] = blocks[::7, :1]                    # rank-deficient, the degenerate case
        centred = blocks - blocks.mean(axis=1, keepdims=True)
        values, _ = np.linalg.eigh(np.einsum("mki,mkj->mij", centred, centred))
        self.assertTrue(bool((np.diff(values, axis=1) >= 0.0).all()))


class EdgeTests(unittest.TestCase):
    def test_an_empty_cloud_returns_empty_arrays(self) -> None:
        result = estimate_surface_normals(np.zeros((0, 3)))
        self.assertEqual(result.size, 0)
        self.assertEqual(result.valid_count, 0)

    def test_a_DEGENERATE_neighbourhood_is_reported_as_a_PERFECT_PLANE(self) -> None:
        """⚠ PINS A TRAP, not a desirable behaviour — and pins it because the batch inherited it
        exactly rather than inventing it.

        Thirty identical points have no surface at all. The covariance is zero, `eigh` returns the
        identity, and the smallest 'principal axis' is whatever column 0 happens to be. Curvature then
        comes out 0.0, which reads as PERFECTLY PLANAR, so planarity is 1.0 and confidence is 1.0 —
        maximum certainty about geometry that is not there. The per-point version does the same
        (verified directly), so this is pre-existing.

        It reaches decisions: `contacts/antipodal.py`, `contacts/dense_sampler.py` and
        the success-model scorers all read `confidence`. MEASURED over 153,781 real
        corpus neighbourhoods, 6 have a second eigenvalue below 1e-6 of the total — a LINE, not a
        surface — and each of those reported a confident normal. Rare, real, and tracked separately
        from the performance work that this file is about.
        """
        result = estimate_surface_normals(np.zeros((30, 3)), radius_mm=15.0)
        self.assertEqual(result.valid_count, 30)
        self.assertAlmostEqual(float(result.curvature[0]), 0.0, places=6)
        self.assertAlmostEqual(float(result.confidence[0]), 1.0, places=6)

    def test_a_flat_plane_gives_the_plane_normal(self) -> None:
        rng = np.random.default_rng(3)
        xy = rng.uniform(-40, 40, (600, 2))
        points = np.column_stack([xy, np.zeros(len(xy))])
        result = estimate_surface_normals(points, radius_mm=15.0)
        self.assertGreater(result.valid_count, 500)
        axis = np.abs(result.normals[result.valid_mask][:, 2])
        self.assertGreater(float(axis.min()), 0.999)


if __name__ == "__main__":
    unittest.main()
