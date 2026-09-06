"""Phase T2 — :class:`GraspScoreBreakdown` feasibility extension.

Locked invariants (operator decisions Q5-C):

1. New top-level ``feasibility_score: float = 0.0`` field on
   :class:`GraspScoreBreakdown`.
2. New ``feasibility: float = 0.0`` weight on
   :class:`GraspScoreWeights`.
3. ``total_score`` formula stays a weighted average:
   ``(w_g·geo + w_s·stab + w_r·reach + w_f·feas) / (w_g+w_s+w_r+w_f)``.
   With default ``w_f = 0.0`` every legacy test's ``total_score`` is
   byte-identical (proven by the regression test below).
4. ``components`` dict gains a stable ``"feasibility"`` entry only
   when ``feasibility_inputs`` is passed (otherwise the dict is
   unchanged from T0).
5. ``to_dict()`` includes ``feasibility_score`` at the top level so
   downstream log consumers see a stable shape.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.geometry import Frame
from src.robot.grasping.planning.grasp_pose import GraspPose
from src.robot.grasping.planning.reachability import IKQualityMetrics
from src.robot.grasping.scoring import (
    GraspScoreBreakdown,
    GraspScoreWeights,
    rank_grasp_poses,
    score_grasp_pose,
)
from src.robot.grasping.scoring.feasibility_score import (
    FeasibilityInputs,
    FeasibilityScoreConfig,
)
from src.robot.grasping.motion.trajectory_safety import (
    ApproachPathOutcome,
    ApproachPathReport,
)


def _pose(*, position: tuple[float, float, float] = (0.0, 0.0, 200.0)) -> GraspPose:
    return GraspPose(
        position_mm=np.array(position, dtype=np.float64),
        rotation_matrix=np.eye(3),
        grip_width_mm=40.0,
        score=0.8,
        confidence=0.9,
        contacts=(
            np.array([position[0] - 20.0, position[1], position[2]], dtype=np.float64),
            np.array([position[0] + 20.0, position[1], position[2]], dtype=np.float64),
        ),
        frame=Frame.BASE,
    )


class GraspScoreWeightsExtensionTests(unittest.TestCase):
    def test_default_feasibility_weight_is_zero(self) -> None:
        # Default weight must be 0.0 so legacy total_score values are
        # byte-identical to T0/T1.
        w = GraspScoreWeights()
        self.assertEqual(w.feasibility, 0.0)

    def test_positive_feasibility_weight_accepted(self) -> None:
        w = GraspScoreWeights(feasibility=0.2)
        self.assertEqual(w.feasibility, 0.2)

    def test_negative_feasibility_weight_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GraspScoreWeights(feasibility=-0.1)


class GraspScoreBreakdownExtensionTests(unittest.TestCase):
    def test_breakdown_has_feasibility_score_field_default_zero(self) -> None:
        b = score_grasp_pose(_pose())
        # When no feasibility_inputs are passed, the field is present
        # but neutral 0.0 and contributes nothing (weight 0.0).
        self.assertTrue(hasattr(b, "feasibility_score"))
        self.assertEqual(b.feasibility_score, 0.0)
        # components dict still has the original three keys, unchanged.
        self.assertIn("geometric", b.components)
        self.assertIn("stability", b.components)
        self.assertIn("reachability", b.components)
        self.assertNotIn("feasibility", b.components)

    def test_to_dict_includes_feasibility_score(self) -> None:
        b = score_grasp_pose(_pose())
        d = b.to_dict()
        self.assertIn("feasibility_score", d)
        self.assertEqual(d["feasibility_score"], 0.0)

    def test_feasibility_inputs_populates_breakdown(self) -> None:
        inputs = FeasibilityInputs(
            ik_quality=IKQualityMetrics(
                condition_number=10.0,
                min_singular_value=0.1,
                joint_margin_deg=20.0,
            ),
            approach_path=ApproachPathReport(outcome=ApproachPathOutcome.CLEAR),
        )
        cfg = FeasibilityScoreConfig(
            ik_quality_enabled=True,
            joint_margin_enabled=True,
            swept_approach_enabled=True,
        )
        b = score_grasp_pose(
            _pose(),
            feasibility_inputs=inputs,
            feasibility_config=cfg,
        )
        self.assertGreater(b.feasibility_score, 0.5)
        self.assertIn("feasibility", b.components)
        self.assertTrue(
            {"ik_quality", "joint_margin", "approach_clearance"}.issubset(
                b.components["feasibility"].keys()
            )
        )


class GraspScoreBreakdownByteIdenticalityTests(unittest.TestCase):
    """Default-weight regression: every legacy total_score is preserved."""

    def _pre_t2_total(self, b: GraspScoreBreakdown) -> float:
        # Reference formula prior to T2: weighted average of three
        # signals with the original weights.
        w = GraspScoreWeights()
        denom = w.geometric + w.stability + w.reachability
        return float(
            (
                b.geometric_score * w.geometric
                + b.stability_score * w.stability
                + b.reachability_score * w.reachability
            )
            / denom
        )

    def test_total_score_matches_legacy_when_feasibility_weight_zero(self) -> None:
        for x in range(-50, 60, 10):
            pose = _pose(position=(float(x), 0.0, 200.0))
            b = score_grasp_pose(pose)
            with self.subTest(x=x):
                self.assertAlmostEqual(
                    b.total_score,
                    self._pre_t2_total(b),
                    places=12,
                )

    def test_ranking_order_preserved_when_feasibility_weight_zero(self) -> None:
        # Operator non-regression: a configuration with feasibility
        # config attached but weight=0 must produce the *same* order
        # as a configuration without any feasibility inputs.
        poses = [_pose(position=(float(x), 0.0, 200.0)) for x in range(-40, 60, 20)]
        ranked_legacy = rank_grasp_poses(poses)
        ranked_t2 = rank_grasp_poses(
            poses,
            feasibility_inputs_per_pose=[
                FeasibilityInputs(
                    ik_quality=IKQualityMetrics(condition_number=99.0),
                )
                for _ in poses
            ],
            feasibility_config=FeasibilityScoreConfig(
                ik_quality_enabled=True
            ),
            # Weight stays 0.0 by default ⇒ feasibility doesn't move
            # total_score even though it's *computed*.
        )
        legacy_order = [r.pose.position_mm.tolist() for r in ranked_legacy]
        t2_order = [r.pose.position_mm.tolist() for r in ranked_t2]
        self.assertEqual(legacy_order, t2_order)

    def test_positive_feasibility_weight_can_reorder(self) -> None:
        # Sanity: when the weight is non-zero, a high-feasibility
        # candidate can outrank a higher-geometric-score candidate.
        poses = [_pose(position=(0.0, 0.0, 200.0)), _pose(position=(50.0, 0.0, 200.0))]
        # Penalise pose 0 via terrible IK quality, reward pose 1.
        inputs = [
            FeasibilityInputs(
                ik_quality=IKQualityMetrics(
                    condition_number=999.0,
                    joint_margin_deg=0.5,
                )
            ),
            FeasibilityInputs(
                ik_quality=IKQualityMetrics(
                    condition_number=5.0,
                    joint_margin_deg=40.0,
                )
            ),
        ]
        cfg = FeasibilityScoreConfig(
            ik_quality_enabled=True, joint_margin_enabled=True
        )
        ranked = rank_grasp_poses(
            poses,
            weights=GraspScoreWeights(
                geometric=0.05, stability=0.05, reachability=0.05, feasibility=0.85
            ),
            feasibility_inputs_per_pose=inputs,
            feasibility_config=cfg,
        )
        # Pose 1 wins because feasibility dominates the weighting.
        self.assertEqual(
            ranked[0].pose.position_mm.tolist(),
            [50.0, 0.0, 200.0],
        )


if __name__ == "__main__":
    unittest.main()
