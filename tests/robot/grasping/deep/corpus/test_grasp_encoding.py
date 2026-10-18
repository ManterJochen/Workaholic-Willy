"""The grasp encoding — a cap of approach directions, rotation mod pi, and a round trip that closes.

This file exists because the round trip has now caught three separate things, two of them mine:

  1. p95 66.7 deg on v1_proof's labels. NOT an encoding bug -- 32.2 % of the analytic reference's
     accepted labels were reached from under the table, a defect nothing else had noticed.
  2. max 36.4 deg after those were repaired. That one WAS this module: the approach set was the lower
     HEMISPHERE, and 1.24 % of admissible labels tilt past horizontal, up to 120 deg.
  3. p95 getting WORSE (9.2 -> 17.7) when the hemisphere became a cap. The bound was written
     `cos(max_tilt)` where the geometry needs `-cos(max_tilt)`, so `max_tilt=135` produced a 45 deg
     cap and crammed 100 bins 7.4 deg apart into a sixth of the intended domain.

None of the three is visible in a loss curve. A wrong convention produces a corpus that trains
perfectly and describes nothing.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.deep.corpus.grasp_encoding import (
    APPROACH_BINS,
    APPROACH_MAX_TILT_DEG,
    ROTATION_BINS,
    approach_bin,
    approach_directions,
    decode_grasp,
    encode_grasp,
    graspability_field,
    rotation_frame,
)


def _tilt_deg(directions: np.ndarray) -> np.ndarray:
    """Angle from straight down. Straight down is (0,0,-1), so this is ``acos(-z)``."""
    return np.degrees(np.arccos(np.clip(-np.asarray(directions)[:, 2], -1.0, 1.0)))


class ApproachCapTests(unittest.TestCase):
    def test_the_cap_reaches_the_tilt_it_was_asked_for(self) -> None:
        """The sign error that made `max_tilt=135` mean 45. Checked at four sizes, not one."""
        for tilt in (45.0, 90.0, 135.0):
            with self.subTest(tilt=tilt):
                angles = _tilt_deg(approach_directions(400, tilt))
                self.assertLessEqual(angles.max(), tilt + 1.0)
                self.assertGreater(angles.max(), tilt - 3.0,
                                   "the cap must REACH its bound, not stop short of it")

    def test_ninety_degrees_is_exactly_the_lower_hemisphere(self) -> None:
        """The anchor case: a 90 deg cap is what the first version of this module built by hand."""
        directions = approach_directions(200, 90.0)
        self.assertLessEqual(float(directions[:, 2].max()), 1e-9)

    def test_every_direction_is_a_unit_vector(self) -> None:
        lengths = np.linalg.norm(approach_directions(APPROACH_BINS), axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-12)

    def test_the_spiral_is_deterministic(self) -> None:
        """The bin index is stored in every target; two spirals would make a corpus meaningless."""
        np.testing.assert_array_equal(approach_directions(64), approach_directions(64))

    def test_the_bins_are_evenly_spread(self) -> None:
        """Equal-area sampling. A spiral that crowded the pole would waste half its resolution."""
        directions = approach_directions(APPROACH_BINS)
        cosine = directions @ directions.T
        np.fill_diagonal(cosine, -1.0)
        spacing = np.degrees(np.arccos(np.clip(cosine.max(axis=1), -1.0, 1.0)))
        self.assertLess(spacing.max() / spacing.min(), 3.0,
                        "nearest-neighbour spacing should not vary wildly across the cap")

    def test_a_refused_count_or_tilt_raises_rather_than_returning_nonsense(self) -> None:
        for kwargs in ({"count": 0}, {"max_tilt_deg": 0.0}, {"max_tilt_deg": 200.0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                approach_directions(**kwargs)  # type: ignore[arg-type]


class AdmissibilityTests(unittest.TestCase):
    def test_a_grasp_from_straight_underneath_is_flagged_out_of_domain(self) -> None:
        """`argmax` always answers. Without this flag it would answer with a 45 deg lie."""
        _bins, admissible = approach_bin(np.array([[0.0, 0.0, 1.0]]))
        self.assertFalse(bool(admissible[0]))

    def test_a_straight_down_grasp_is_in_domain(self) -> None:
        _bins, admissible = approach_bin(np.array([[0.0, 0.0, -1.0]]))
        self.assertTrue(bool(admissible[0]))

    def test_every_bin_centre_is_in_its_own_domain(self) -> None:
        """A direction that IS a bin cannot be out-of-domain; if it is, the tolerance is wrong."""
        _bins, admissible = approach_bin(approach_directions(APPROACH_BINS))
        self.assertTrue(bool(admissible.all()))

    def test_encode_carries_admissibility_out_rather_than_dropping_rows(self) -> None:
        """Dropping is the caller's call, and the COUNT dropped is a number worth reporting."""
        approach = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
        axis = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        bins, rotations, admissible = encode_grasp(approach, axis)
        self.assertEqual(len(bins), 2)
        self.assertEqual(len(rotations), 2)
        self.assertEqual(admissible.tolist(), [True, False])


class RoundTripTests(unittest.TestCase):
    """Encode then decode. The error must be bounded by the discretisation and nothing else."""

    def _grasps(self, count: int = 2000) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(20260824)
        # Sampled over the cap itself, not over the sphere: this measures the encoding, not the
        # admissibility filter.
        directions = approach_directions(count, APPROACH_MAX_TILT_DEG)
        tangent, binormal = rotation_frame(directions)
        angle = rng.uniform(0.0, np.pi, size=count)
        axis = tangent * np.cos(angle)[:, None] + binormal * np.sin(angle)[:, None]
        return directions, axis / np.linalg.norm(axis, axis=1, keepdims=True)

    def test_the_approach_survives_to_within_the_bin_spacing(self) -> None:
        approach, axis = self._grasps()
        bins, rotations, admissible = encode_grasp(approach, axis)
        self.assertTrue(bool(admissible.all()))
        decoded, _ = decode_grasp(bins, rotations)
        error = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", decoded, approach), -1.0, 1.0)))
        # Measured on 483 real labels: median 7.49, p95 8.88, max 13.26 deg against a 17.8 deg
        # nearest-neighbour spacing. 20 is that bound with room; a regression past it means the
        # spiral, the cap or the frame moved.
        self.assertLess(float(error.max()), 20.0)

    def test_the_closing_axis_survives_to_within_the_rotation_bin(self) -> None:
        approach, axis = self._grasps()
        bins, rotations, _ = encode_grasp(approach, axis)
        _, decoded = decode_grasp(bins, rotations)
        # |cos|, because the axis and its negative are the same grasp.
        error = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", decoded, axis)), 0.0, 1.0)))
        self.assertLess(float(error.max()), 180.0 / ROTATION_BINS)

    def test_an_axis_and_its_negative_encode_identically(self) -> None:
        """The jaw is symmetric. If these differ, every grasp's probability mass is split in two."""
        approach, axis = self._grasps(200)
        _, forward, _ = encode_grasp(approach, axis)
        _, backward, _ = encode_grasp(approach, -axis)
        np.testing.assert_array_equal(forward, backward)

    def test_a_closing_axis_is_projected_rather_than_assumed_perpendicular(self) -> None:
        """Real labels are orthogonal to about 1e-6, not exactly. Tilting one must not move its bin."""
        approach = np.array([[0.0, 0.0, -1.0]])
        clean = np.array([[1.0, 0.0, 0.0]])
        tilted = np.array([[1.0, 0.0, 0.02]])          # 1.1 deg out of plane
        _, a, _ = encode_grasp(approach, clean)
        _, b, _ = encode_grasp(approach, tilted)
        np.testing.assert_array_equal(a, b)


class RotationFrameTests(unittest.TestCase):
    def test_the_frame_is_orthonormal_everywhere_on_the_cap(self) -> None:
        directions = approach_directions(500, APPROACH_MAX_TILT_DEG)
        tangent, binormal = rotation_frame(directions)
        np.testing.assert_allclose(np.linalg.norm(tangent, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(binormal, axis=1), 1.0, atol=1e-12)
        for a, b in ((directions, tangent), (directions, binormal), (tangent, binormal)):
            np.testing.assert_allclose(np.einsum("ij,ij->i", a, b), 0.0, atol=1e-12)

    def test_it_survives_a_straight_down_approach(self) -> None:
        """The single most common grasp in this project, and the one a fixed +z cross product kills."""
        tangent, binormal = rotation_frame(np.array([[0.0, 0.0, -1.0]]))
        self.assertTrue(np.isfinite(tangent).all() and np.isfinite(binormal).all())
        np.testing.assert_allclose(np.linalg.norm(tangent, axis=1), 1.0, atol=1e-12)


class GraspabilityFieldTests(unittest.TestCase):
    def test_points_near_a_contact_light_up_and_the_rest_do_not(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        field = graspability_field(points, np.array([[0.0, 0.0, 0.0]]), radius_mm=10.0)
        self.assertEqual(field.tolist(), [1.0, 1.0, 0.0])

    def test_an_empty_input_returns_an_empty_field_rather_than_raising(self) -> None:
        self.assertEqual(len(graspability_field(np.zeros((0, 3)), np.zeros((0, 3)))), 0)
        self.assertEqual(graspability_field(np.zeros((3, 3)), np.zeros((0, 3))).tolist(),
                         [0.0, 0.0, 0.0])

    def test_the_field_is_about_CONTACTS_not_grasp_centres(self) -> None:
        """A centre floats between the pads. Seeds sampled there would sit in free space."""
        surface = np.array([[-20.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        centre = np.array([[0.0, 0.0, 0.0]])
        self.assertEqual(graspability_field(surface, centre, radius_mm=10.0).tolist(), [0.0, 0.0])
        self.assertEqual(graspability_field(surface, surface, radius_mm=10.0).tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
