"""Phase F: analytical force-closure certificate tests."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.robot.grasping import (
    ForceClosureCertificate,
    GraspCalculator,
    certify_contact_pair,
)
from src.robot.grasping.contacts import ContactPair
from src.robot.grasping.scoring.force_closure import friction_threshold


def _pair(
    p_a: tuple[float, float, float],
    p_b: tuple[float, float, float],
    n_a: tuple[float, float, float],
    n_b: tuple[float, float, float],
) -> ContactPair:
    p_a_arr = np.asarray(p_a, dtype=np.float64)
    p_b_arr = np.asarray(p_b, dtype=np.float64)
    n_a_arr = np.asarray(n_a, dtype=np.float64)
    n_b_arr = np.asarray(n_b, dtype=np.float64)
    n_a_arr = n_a_arr / np.linalg.norm(n_a_arr)
    n_b_arr = n_b_arr / np.linalg.norm(n_b_arr)
    distance = float(np.linalg.norm(p_b_arr - p_a_arr))
    # antipodal_score / axis_alignment / normal_opposition are not used
    # by the certificate; we still have to provide valid values.
    return ContactPair(
        point_a=p_a_arr,
        point_b=p_b_arr,
        normal_a=n_a_arr,
        normal_b=n_b_arr,
        distance_mm=distance,
        antipodal_score=1.0,
        axis_alignment=1.0,
        normal_opposition=float(np.clip(-np.dot(n_a_arr, n_b_arr), 0.0, 1.0)),
    )


class FrictionThresholdTests(unittest.TestCase):
    def test_zero_friction_is_unit(self) -> None:
        self.assertAlmostEqual(friction_threshold(0.0), 1.0)

    def test_threshold_matches_cos_atan_mu(self) -> None:
        for mu in (0.1, 0.4, 0.8, 1.5):
            self.assertAlmostEqual(
                friction_threshold(mu),
                math.cos(math.atan(mu)),
                places=12,
            )

    def test_negative_friction_rejected(self) -> None:
        with self.assertRaises(ValueError):
            friction_threshold(-0.1)

    def test_nonfinite_friction_rejected(self) -> None:
        with self.assertRaises(ValueError):
            friction_threshold(float("nan"))
        with self.assertRaises(ValueError):
            friction_threshold(float("inf"))


class CertifyContactPairTests(unittest.TestCase):
    def test_perfect_antipodal_certified(self) -> None:
        # Outward normals point away from object: at p_a (x=-50) the
        # outward normal is -x, at p_b (x=+50) the outward normal is +x.
        pair = _pair((-50.0, 0.0, 0.0), (50.0, 0.0, 0.0), (-1, 0, 0), (1, 0, 0))
        cert = certify_contact_pair(pair, friction_coefficient=0.4)
        self.assertIsInstance(cert, ForceClosureCertificate)
        self.assertTrue(cert.is_certified)
        self.assertAlmostEqual(cert.cos_alpha_a, 1.0, places=12)
        self.assertAlmostEqual(cert.cos_alpha_b, 1.0, places=12)
        self.assertGreater(cert.margin, 0.0)

    def test_perfect_antipodal_certified_at_zero_friction(self) -> None:
        pair = _pair((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1, 0, 0), (1, 0, 0))
        cert = certify_contact_pair(pair, friction_coefficient=0.0)
        self.assertTrue(cert.is_certified)
        # With mu=0 the threshold is exactly 1.0, so margin must be ~0.
        self.assertAlmostEqual(cert.margin, 0.0, places=12)

    def test_outside_friction_cone_not_certified(self) -> None:
        # Tilt both normals 60 deg away from the closing axis. With mu=0.4
        # the friction half-angle is atan(0.4) ~= 21.8 deg, so 60 deg is
        # well outside the cone.
        angle = math.radians(60.0)
        n_a = (-math.cos(angle), math.sin(angle), 0.0)
        n_b = (math.cos(angle), math.sin(angle), 0.0)
        pair = _pair((-50.0, 0.0, 0.0), (50.0, 0.0, 0.0), n_a, n_b)
        cert = certify_contact_pair(pair, friction_coefficient=0.4)
        self.assertFalse(cert.is_certified)
        self.assertLess(cert.margin, 0.0)

    def test_edge_of_cone_marginally_certified(self) -> None:
        # Tilt both normals to exactly the friction half-angle. The test
        # should sit on the boundary (margin ~= 0) and be certified
        # (the inequality is >=).
        mu = 0.4
        alpha = math.atan(mu)
        n_a = (-math.cos(alpha), math.sin(alpha), 0.0)
        n_b = (math.cos(alpha), math.sin(alpha), 0.0)
        pair = _pair((-10.0, 0.0, 0.0), (10.0, 0.0, 0.0), n_a, n_b)
        cert = certify_contact_pair(pair, friction_coefficient=mu)
        self.assertTrue(cert.is_certified)
        self.assertAlmostEqual(cert.margin, 0.0, places=10)

    def test_asymmetric_normal_failure_dominates_margin(self) -> None:
        # Contact a is perfectly aligned, contact b is far off-cone.
        # The certificate must report not-certified and the margin must
        # be driven by the worse (b) cosine.
        bad_angle = math.radians(75.0)
        n_a = (-1.0, 0.0, 0.0)
        n_b = (math.cos(bad_angle), math.sin(bad_angle), 0.0)
        pair = _pair((-30.0, 0.0, 0.0), (30.0, 0.0, 0.0), n_a, n_b)
        cert = certify_contact_pair(pair, friction_coefficient=0.4)
        self.assertFalse(cert.is_certified)
        self.assertAlmostEqual(cert.cos_alpha_a, 1.0, places=12)
        self.assertAlmostEqual(
            cert.cos_alpha_b, math.cos(bad_angle), places=12
        )
        self.assertAlmostEqual(
            cert.margin, cert.cos_alpha_b - cert.cos_threshold, places=12
        )

    def test_to_dict_round_trip(self) -> None:
        pair = _pair((-5.0, 0.0, 0.0), (5.0, 0.0, 0.0), (-1, 0, 0), (1, 0, 0))
        cert = certify_contact_pair(pair, friction_coefficient=0.5)
        payload = cert.to_dict()
        self.assertEqual(
            set(payload.keys()),
            {
                "is_certified",
                "friction_coefficient",
                "cos_threshold",
                "cos_alpha_a",
                "cos_alpha_b",
                "margin",
            },
        )
        self.assertTrue(payload["is_certified"])
        self.assertAlmostEqual(payload["friction_coefficient"], 0.5)

    def test_invalid_pair_type_rejected(self) -> None:
        with self.assertRaises(TypeError):
            certify_contact_pair("not a pair", friction_coefficient=0.4)  # type: ignore[arg-type]


class GraspCalculatorFrictionWiringTests(unittest.TestCase):
    """Confirm Phase F integration is strictly additive."""

    def test_default_constructor_leaves_certification_disabled(self) -> None:
        calc = GraspCalculator()
        self.assertIsNone(calc.friction_coefficient)

    def test_invalid_friction_coefficient_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GraspCalculator(friction_coefficient=-0.1)
        with self.assertRaises(ValueError):
            GraspCalculator(friction_coefficient=5.0)
        with self.assertRaises(ValueError):
            GraspCalculator(friction_coefficient=float("nan"))

    def test_friction_coefficient_accepted_in_range(self) -> None:
        calc = GraspCalculator(friction_coefficient=0.4)
        self.assertAlmostEqual(calc.friction_coefficient or 0.0, 0.4)

    def test_helper_is_noop_when_friction_disabled(self) -> None:
        calc = GraspCalculator()  # friction_coefficient = None
        # Helper should run and not touch pose metadata.
        from src.robot.grasping.planning import generate_grasp_poses

        pair = _pair(
            (-30.0, 0.0, 0.0), (30.0, 0.0, 0.0), (-1, 0, 0), (1, 0, 0)
        )
        poses = generate_grasp_poses([pair])
        self.assertEqual(len(poses), 1)
        before = dict(poses[0].metadata)
        certified = calc._apply_force_closure_certificates([pair], poses)
        self.assertEqual(certified, 0)
        self.assertNotIn("force_closure", poses[0].metadata)
        self.assertNotIn("force_closure_certified", poses[0].metadata)
        self.assertEqual(poses[0].metadata, before)

    def test_helper_stamps_certificate_when_enabled(self) -> None:
        from src.robot.grasping.planning import generate_grasp_poses

        calc = GraspCalculator(friction_coefficient=0.4)
        pair = _pair(
            (-30.0, 0.0, 0.0), (30.0, 0.0, 0.0), (-1, 0, 0), (1, 0, 0)
        )
        poses = generate_grasp_poses([pair])
        certified = calc._apply_force_closure_certificates([pair], poses)
        self.assertEqual(certified, 1)
        meta = poses[0].metadata
        self.assertTrue(meta["force_closure_certified"])
        self.assertEqual(meta["force_closure"]["is_certified"], True)
        self.assertEqual(meta["confidence_kind"], "analytical")

    def test_helper_keeps_heuristic_when_not_certified(self) -> None:
        from src.robot.grasping.planning import generate_grasp_poses

        calc = GraspCalculator(friction_coefficient=0.1)
        # Tilt 60deg outside the friction cone (atan(0.1) ~= 5.7 deg).
        a = math.radians(60.0)
        pair = _pair(
            (-30.0, 0.0, 0.0),
            (30.0, 0.0, 0.0),
            (-math.cos(a), math.sin(a), 0.0),
            (math.cos(a), math.sin(a), 0.0),
        )
        poses = generate_grasp_poses([pair])
        # Mark the heuristic kind the way the calculator pipeline does.
        poses[0].metadata.setdefault("confidence_kind", "heuristic")
        certified = calc._apply_force_closure_certificates([pair], poses)
        self.assertEqual(certified, 0)
        meta = poses[0].metadata
        self.assertFalse(meta["force_closure_certified"])
        # Non-certified candidates retain their heuristic marker.
        self.assertEqual(meta["confidence_kind"], "heuristic")


if __name__ == "__main__":
    unittest.main()
