"""The sim ETH sweep says the gate it holds a solve to, not a stricter one (found in the S6 Isaac bundle, 2026-09-26).

``run_eth_calibrate`` accepts a marginal residual for a rendered ArUco marker, on purpose: a small marker seen obliquely
from the overhead camera lands near 3 mm. Its RESULT line and its log line said "quality good+" whatever the marker, so
a marginal ArUco solve printed ``GATE (<8 mm, <1.5 deg, quality good+): PASS``, and a reader took the calibration for a
good one (measured on 2026-09-26: 10 of 22 stations, rmse 3.44 mm, marginal). The verdict and the words that state its
rule now come from one function, so they cannot say two things.
"""

from __future__ import annotations

import unittest

from src.willy_sim.run_eth_calibrate import _gate


class TheGateSaysTheRuleItChecksTests(unittest.TestCase):

    def test_a_marginal_aruco_solve_passes_and_says_marginal_is_enough(self) -> None:
        ok, rule = _gate("aruco", 6.286, 0.487, "marginal")
        self.assertTrue(ok)
        self.assertEqual(rule, "<8 mm, <1.5 deg, quality marginal+")

    def test_ground_truth_still_needs_a_good_residual_and_says_so(self) -> None:
        ok, rule = _gate("ground_truth", 0.0, 0.0, "marginal")
        self.assertFalse(ok)
        self.assertEqual(rule, "<1 mm, <1.5 deg, quality good+")
        self.assertTrue(_gate("ground_truth", 0.5, 0.2, "excellent")[0])

    def test_the_bounds_hold_whatever_the_quality(self) -> None:
        self.assertFalse(_gate("aruco", 8.0, 0.1, "excellent")[0])
        self.assertFalse(_gate("aruco", 1.0, 1.5, "excellent")[0])
        self.assertFalse(_gate("aruco", 1.0, 0.1, "poor")[0])
        self.assertFalse(_gate("ground_truth", 1.0, 0.0, "excellent")[0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
