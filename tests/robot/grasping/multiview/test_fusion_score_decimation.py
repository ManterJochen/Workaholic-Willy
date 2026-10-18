"""Scoring-only decimation: 584x cheaper, and the geometry that reaches the generator is untouched.

MEASURED 2026-08-19 driving the real stack over a datagen scene (`bin_000009`, three views, five
objects). One pick took 187 s and `_fused_scene` was 182.7 s of it. The cause is documented in the code
that has it: ``_overlap_fraction`` states its own precondition -- *"the clouds are a few thousand points
at most"* -- and one datagen object's cloud is **37,119 points**, an order of magnitude past it. Chunked
brute force over 37k x 37k is ~1.4e9 distance evaluations.

The precondition was reasonable for the scenes it was written against. A path-traced data-generation
corpus simply is not those scenes.

    voxel_mm   seconds   speedup   assignment vs 0.0   fused clouds
       0.0      182.16      1.0x   (reference)         (reference)
       2.0        1.92     94.8x   identical           identical
       4.0        0.31    583.8x   identical           identical
       8.0        0.19    968.9x   identical           identical
      16.0        0.16   1167.7x   identical           identical

**The fused clouds are identical by construction, not by luck**, and that is the design: the decimation
feeds only the association SCORING, while ``fuse_scene_clouds`` rebuilds each object's cloud from the
ORIGINAL full-resolution inputs. A decimation applied to the output instead would quietly coarsen every
grasp. The tests below pin that separation.

**Default 0.0 = off.** The fixed cameras of a cell score their full clouds; the looks of a wrist camera score
on 3 mm voxels (``association.WRIST_VIEW_SCORE_VOXEL_MM``), whose views are a few thousand points no longer.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.multiview.association import (
    ViewCandidates,
    _voxel_decimate,
    assign_view,
    fuse_scene_clouds,
)


def _slab(origin: np.ndarray, *, n: int = 4000, extent: float = 60.0, seed: int = 0) -> np.ndarray:
    """A dense point slab at ``origin`` -- the shape a real depth view of one object produces."""
    rng = np.random.default_rng(seed)
    return origin + rng.uniform(-extent / 2.0, extent / 2.0, size=(n, 3))


class DecimatorTests(unittest.TestCase):
    def test_off_returns_the_input_untouched(self) -> None:
        points = _slab(np.array([0.0, 0.0, 0.0]))
        for voxel in (0.0, -1.0):
            with self.subTest(voxel=voxel):
                self.assertIs(_voxel_decimate(points, voxel), points)

    def test_it_keeps_observed_points_not_invented_ones(self) -> None:
        # A cell's centroid is a point nothing ever saw, and the overlap is a statement about
        # observed surface. Every survivor must be a row of the input.
        points = _slab(np.array([100.0, 0.0, 0.0]), n=2000, seed=1)
        kept = _voxel_decimate(points, 5.0)
        self.assertLess(len(kept), len(points))
        as_rows = {tuple(row) for row in points}
        for row in kept:
            self.assertIn(tuple(row), as_rows)

    def test_it_actually_thins(self) -> None:
        points = _slab(np.array([0.0, 0.0, 0.0]), n=20000, extent=60.0, seed=2)
        self.assertLess(len(_voxel_decimate(points, 8.0)), len(points) / 4)

    def test_an_empty_cloud_survives(self) -> None:
        self.assertEqual(_voxel_decimate(np.zeros((0, 3)), 4.0).shape, (0, 3))


class TheMatchIsUnchangedTests(unittest.TestCase):
    """Two well-separated objects must associate the same way, decimated or not."""

    def setUp(self) -> None:
        self.primaries = [
            _slab(np.array([0.0, 0.0, 0.0]), seed=10),
            _slab(np.array([300.0, 0.0, 0.0]), seed=11),
        ]
        # The other camera's view: same objects, in the OPPOSITE order, so a correct association has
        # to actually match rather than pass positions through.
        self.view = ViewCandidates("other", (
            _slab(np.array([300.0, 0.0, 0.0]), seed=12),
            _slab(np.array([0.0, 0.0, 0.0]), seed=13),
        ))

    def _assignment(self, voxel: float) -> tuple:
        return tuple(assign_view(self.primaries, self.view, score_voxel_mm=voxel).assignment)

    def test_the_assignment_survives_decimation(self) -> None:
        reference = self._assignment(0.0)
        self.assertEqual(reference, (1, 0))  # crossed, as constructed
        for voxel in (2.0, 4.0, 8.0):
            with self.subTest(voxel=voxel):
                self.assertEqual(self._assignment(voxel), reference)

    def test_the_fused_clouds_are_identical_because_they_come_from_the_originals(self) -> None:
        # The property the whole design rests on. If this ever fails, the decimation has leaked out of
        # the scoring and into the geometry, and every grasp planned on it is coarser than it says.
        off, _ = fuse_scene_clouds(self.primaries, [self.view])
        on, _ = fuse_scene_clouds(self.primaries, [self.view], score_voxel_mm=4.0)
        self.assertEqual(len(off), len(on))
        for a, b in zip(off, on):
            np.testing.assert_array_equal(a, b)


class TheDefaultIsOffTests(unittest.TestCase):

    def test_not_setting_it_matches_setting_it_to_zero(self) -> None:
        primaries = [_slab(np.array([0.0, 0.0, 0.0]), seed=30)]
        view = ViewCandidates("other", (_slab(np.array([0.0, 0.0, 0.0]), seed=31),))
        default = assign_view(primaries, view)
        explicit = assign_view(primaries, view, score_voxel_mm=0.0)
        self.assertEqual(default.scores, explicit.scores)
        self.assertEqual(default.assignment, explicit.assignment)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
