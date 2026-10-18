"""The shared ContactPoint substrate: validation policy, the caller-aliasing fix, and composition."""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.contacts import ContactPair, ContactPoint


class ContactPointTests(unittest.TestCase):
    def test_valid_point_and_unit_normal(self) -> None:
        cp = ContactPoint(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 1.0]))
        self.assertTrue(np.allclose(cp.point_mm, [1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(np.linalg.norm(cp.normal)), 1.0)
        self.assertEqual(cp.to_dict(), {"point_mm": [1.0, 2.0, 3.0], "normal": [0.0, 0.0, 1.0]})

    def test_near_unit_normal_is_accepted_and_normalised(self) -> None:
        # Within the 1% tolerance -> accepted, stored exactly unit.
        cp = ContactPoint(np.array([0.0, 0.0, 0.0]), np.array([1.005, 0.0, 0.0]))
        self.assertAlmostEqual(float(np.linalg.norm(cp.normal)), 1.0, places=12)

    def test_non_unit_normal_rejected(self) -> None:
        for bad in (np.array([2.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0]), np.zeros(3)):
            with self.assertRaises(ValueError):
                ContactPoint(np.zeros(3), bad)

    def test_bad_shape_or_nonfinite_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ContactPoint(np.zeros(2), np.array([1.0, 0.0, 0.0]))
        with self.assertRaises(ValueError):
            ContactPoint(np.array([np.inf, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))

    def test_frozen_normal_is_readonly(self) -> None:
        cp = ContactPoint(np.zeros(3), np.array([1.0, 0.0, 0.0]))
        with self.assertRaises(ValueError):
            cp.normal[0] = 0.0


class CallerAliasingRegressionTests(unittest.TestCase):
    """The old ContactPair froze the CALLER's array in place (np.asarray without copy)."""

    def test_contact_point_does_not_freeze_caller_array(self) -> None:
        point = np.array([1.0, 2.0, 3.0])
        ContactPoint(point, np.array([1.0, 0.0, 0.0]))
        point[0] = 9.0  # must not raise; the caller keeps a writable array
        self.assertEqual(point[0], 9.0)

    def test_contact_pair_does_not_freeze_caller_arrays(self) -> None:
        point_a = np.array([0.0, 0.0, 0.0])
        point_b = np.array([10.0, 0.0, 0.0])
        ContactPair(
            point_a=point_a,
            point_b=point_b,
            normal_a=np.array([-1.0, 0.0, 0.0]),
            normal_b=np.array([1.0, 0.0, 0.0]),
            distance_mm=10.0,
            antipodal_score=1.0,
        )
        point_a[0] = 5.0  # was the latent bug: 'assignment destination is read-only'
        self.assertEqual(point_a[0], 5.0)


class ContactPairComposesContactPointTests(unittest.TestCase):
    def test_contact_views_round_trip(self) -> None:
        pair = ContactPair(
            point_a=np.array([0.0, 0.0, 0.0]),
            point_b=np.array([10.0, 0.0, 0.0]),
            normal_a=np.array([-1.0, 0.0, 0.0]),
            normal_b=np.array([1.0, 0.0, 0.0]),
            distance_mm=10.0,
            antipodal_score=1.0,
        )
        self.assertIsInstance(pair.contact_a, ContactPoint)
        self.assertTrue(np.array_equal(pair.contact_a.point_mm, pair.point_a))
        self.assertTrue(np.array_equal(pair.contact_b.normal, pair.normal_b))


if __name__ == "__main__":
    unittest.main()
