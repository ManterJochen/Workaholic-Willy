"""RadiusIndex: the SciPy-accelerated and the pure-NumPy fallback must agree bit-for-bit.

The brute-force branch is otherwise ``pragma: no cover`` (SciPy is always present in CI), yet the
antipodal search relies on both back-ends producing the same neighbour set. These tests force the
fallback so that property is actually exercised.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.contacts import antipodal
from src.robot.grasping.contacts.antipodal import find_antipodal_pairs
from src.robot.grasping.geometry import RadiusIndex

# Two facing faces at x=0 and x=40 -> several valid antipodal pairs across the gap.
_POINTS = np.array(
    [
        [0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [0.0, 8.0, 8.0],
        [40.0, 0.0, 0.0], [40.0, 10.0, 0.0], [40.0, 0.0, 10.0], [40.0, 8.0, 8.0],
    ]
)
_NORMALS = np.array(
    [[1.0, 0.0, 0.0]] * 4 + [[-1.0, 0.0, 0.0]] * 4
)


class _BruteRadiusIndex(RadiusIndex):
    """RadiusIndex with the KD-tree disabled, forcing the NumPy brute-force path."""

    def __init__(self, points: np.ndarray) -> None:
        super().__init__(points)
        self._tree = None


class RadiusIndexBackendAgreementTests(unittest.TestCase):
    def test_query_radius_backends_agree(self) -> None:
        tree = RadiusIndex(_POINTS)
        brute = _BruteRadiusIndex(_POINTS)
        for q in _POINTS:
            got_tree = sorted(int(i) for i in tree.query_radius(q, 45.0))
            got_brute = sorted(int(i) for i in brute.query_radius(q, 45.0))
            self.assertEqual(got_tree, got_brute)

    def test_query_knn_backends_agree(self) -> None:
        tree = RadiusIndex(_POINTS)
        brute = _BruteRadiusIndex(_POINTS)
        for q in _POINTS:
            got_tree = sorted(int(i) for i in tree.query_knn(q, 3))
            got_brute = sorted(int(i) for i in brute.query_knn(q, 3))
            self.assertEqual(got_tree, got_brute)


class AntipodalDeterminismWithoutScipyTests(unittest.TestCase):
    def test_pairs_identical_with_and_without_kdtree(self) -> None:
        with_tree = find_antipodal_pairs(_POINTS, _NORMALS, min_width_mm=5.0, max_width_mm=150.0)
        self.assertGreater(len(with_tree), 0)  # the fixture must actually produce pairs
        original = antipodal.RadiusIndex
        try:
            antipodal.RadiusIndex = _BruteRadiusIndex  # type: ignore[misc]
            without_tree = find_antipodal_pairs(_POINTS, _NORMALS, min_width_mm=5.0, max_width_mm=150.0)
        finally:
            antipodal.RadiusIndex = original  # type: ignore[misc]
        self.assertEqual([p.to_dict() for p in with_tree], [p.to_dict() for p in without_tree])


if __name__ == "__main__":
    unittest.main()
