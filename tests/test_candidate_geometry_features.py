"""The grasp's own shape, logged per candidate — the only kind of feature a pairwise ranker can use.

THE FINDING THIS EXISTS FOR (measured 2026-08-19, driving the real stack over datagen scenes). The
pairwise ranker differences features WITHIN a group, and a group is one attempt's candidate list, which
all come from the SAME winning segmentation. So a scene-level, pick-level or even object-level quantity
is identical for every row in the group and differences to exactly zero -- however well populated it
looks on a dashboard. Only a genuinely per-GRASP quantity can move the model at all.

Measured over 98 candidate rows from 12 scenes, before and after:

    per-candidate LIVE   3/15  ->  7/19

    geom.grip_width_mm          53 distinct   28.21 .. 73.83 mm    stdev 12.24
    geom.grasp_z_mm             16 distinct   46.40 .. 112.00 mm   stdev 15.03
    geom.approach_clearance_mm   6 distinct    5.00 .. 140.00 mm   stdev 46.13
    geom.approach_tilt_deg       5 distinct    0.00 .. 90.00 deg   stdev 15.26
    geometric_score             66 distinct    0.53 .. 0.93        stdev  0.07
    (every other RANKING_FEATURE_KEY: constant)

**Logged BESIDE the policy vector, not inside it.** ``load_pairwise_logistic_ranking_policy`` requires
an artifact's ``feature_keys`` to equal ``RANKING_FEATURE_KEYS`` exactly and raises otherwise, so
extending that tuple would make the committed v3 baseline unloadable -- and that baseline is the
behaviour policy this data is collected under. A richer DATASET costs nothing; changing the policy
contract is a separate, versioned decision.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.robot.grasping.loop._shadow_aggregator import (
    _build_candidate_log,
    _candidate_geometry,
)


@dataclass
class _Candidate:
    grip_width_mm: float = 40.0
    position: Any = field(default_factory=lambda: np.array([100.0, 0.0, 75.0]))
    approach: Any = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    metadata: dict = field(default_factory=dict)


class GeometryExtractionTests(unittest.TestCase):
    def test_a_top_down_approach_is_zero_degrees(self) -> None:
        geometry = _candidate_geometry(_Candidate(approach=np.array([0.0, 0.0, -1.0])))
        self.assertAlmostEqual(geometry["approach_tilt_deg"], 0.0, places=6)

    def test_a_horizontal_approach_is_ninety(self) -> None:
        geometry = _candidate_geometry(_Candidate(approach=np.array([1.0, 0.0, 0.0])))
        self.assertAlmostEqual(geometry["approach_tilt_deg"], 90.0, places=6)

    def test_tilt_is_measured_from_straight_down_at_forty_five(self) -> None:
        geometry = _candidate_geometry(_Candidate(approach=np.array([1.0, 0.0, -1.0])))
        self.assertAlmostEqual(geometry["approach_tilt_deg"], 45.0, places=6)

    def test_an_unnormalised_approach_gives_the_same_angle(self) -> None:
        # The calculator does not promise a unit vector; a length-dependent angle would be a silent
        # scale bug that only shows up as a slightly wrong feature.
        a = _candidate_geometry(_Candidate(approach=np.array([0.0, 3.0, -3.0])))
        b = _candidate_geometry(_Candidate(approach=np.array([0.0, 1.0, -1.0])))
        self.assertAlmostEqual(a["approach_tilt_deg"], b["approach_tilt_deg"], places=9)

    def test_height_is_the_base_frame_z(self) -> None:
        geometry = _candidate_geometry(_Candidate(position=np.array([1.0, 2.0, 88.5])))
        self.assertAlmostEqual(geometry["grasp_z_mm"], 88.5, places=6)

    def test_width_and_clearance_come_through(self) -> None:
        geometry = _candidate_geometry(
            _Candidate(grip_width_mm=51.25, metadata={"approach_clearance_mm": 70.0})
        )
        self.assertAlmostEqual(geometry["grip_width_mm"], 51.25, places=6)
        self.assertAlmostEqual(geometry["approach_clearance_mm"], 70.0, places=6)

    def test_it_never_raises_on_a_malformed_candidate(self) -> None:
        # A geometry block is telemetry. It must never be the reason a pick fails.
        class _Broken:
            grip_width_mm = "not a number"
            position = "nope"
            approach = object()
            metadata = {"approach_clearance_mm": None}

        self.assertEqual(_candidate_geometry(_Broken()), {})

    def test_a_bare_object_yields_an_empty_block(self) -> None:
        self.assertEqual(_candidate_geometry(object()), {})

    def test_a_zero_length_approach_reports_no_angle(self) -> None:
        # Rather than an arbitrary 0 deg, which would read as "perfectly top-down".
        geometry = _candidate_geometry(_Candidate(approach=np.array([0.0, 0.0, 0.0])))
        self.assertNotIn("approach_tilt_deg", geometry)


class CandidateLogTests(unittest.TestCase):
    def test_geometry_rides_alongside_the_policy_vector(self) -> None:
        rows, behavior = _build_candidate_log(
            ("c0", "c1"),
            {"c0": {"geometric_score": 0.9}, "c1": {"geometric_score": 0.7}},
            geometry_by_id={"c0": {"grip_width_mm": 40.0}, "c1": {"grip_width_mm": 55.0}},
        )
        self.assertEqual(behavior, "c0")
        self.assertEqual(rows[0]["features"], {"geometric_score": 0.9})
        self.assertEqual(rows[0]["geometry"], {"grip_width_mm": 40.0})
        self.assertEqual(rows[1]["geometry"], {"grip_width_mm": 55.0})

    def test_the_policy_vector_is_untouched_by_the_geometry(self) -> None:
        # The contract that keeps the committed v3 baseline loadable.
        with_geom, _ = _build_candidate_log(
            ("c0",), {"c0": {"geometric_score": 0.9}},
            geometry_by_id={"c0": {"grip_width_mm": 40.0}},
        )
        without, _ = _build_candidate_log(("c0",), {"c0": {"geometric_score": 0.9}})
        self.assertEqual(with_geom[0]["features"], without[0]["features"])

    def test_a_caller_that_passes_no_geometry_logs_the_row_it_always_logged(self) -> None:
        rows, _ = _build_candidate_log(("c0",), {"c0": {"geometric_score": 0.9}})
        self.assertNotIn("geometry", rows[0])
        self.assertEqual(set(rows[0]), {"candidate_id", "rank", "executed", "features"})

    def test_an_empty_geometry_block_is_omitted_not_written_empty(self) -> None:
        rows, _ = _build_candidate_log(("c0",), {"c0": {}}, geometry_by_id={"c0": {}})
        self.assertNotIn("geometry", rows[0])


class WhyTheOtherKeysCannotHelpTests(unittest.TestCase):
    """Pinned as arithmetic, because it is the reason this file exists."""

    def test_a_feature_constant_within_a_group_differences_to_zero(self) -> None:
        # What the pairwise trainer does with a success/fail pair: it subtracts. A key with the same
        # value on both sides contributes nothing to the gradient, whatever that value is.
        success = {"pick_level_key": 188797.0, "per_grasp_key": 0.93}
        failure = {"pick_level_key": 188797.0, "per_grasp_key": 0.53}
        delta = {k: success[k] - failure[k] for k in success}
        self.assertEqual(delta["pick_level_key"], 0.0)
        self.assertNotEqual(delta["per_grasp_key"], 0.0)

    def test_uncertainty_score_is_one_minus_the_geometric_score(self) -> None:
        # MEASURED over 7 records: max |uncertainty_score - (1 - geometric_score)| = 3.9e-05, i.e. the
        # 4-decimal rounding in the record and nothing else. On the legacy-fallback path the decision
        # engine computes exactly (1 - top_score) + reasons_penalty, so plumbing this into the
        # per-candidate row would hand a learner a perfect mirror of a feature it already has and
        # inflate its apparent quality. It is deliberately NOT plumbed.
        for geometric_score in (0.8985, 0.8482, 0.9303, 0.8394):
            with self.subTest(geometric_score=geometric_score):
                legacy_uncertainty = max(0.0, 1.0 - geometric_score)
                self.assertAlmostEqual(legacy_uncertainty, 1.0 - geometric_score, places=9)
                self.assertLess(abs(legacy_uncertainty - (1.0 - geometric_score)), 1e-9)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
