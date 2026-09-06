"""Phase T2 schema tests for ``robot.grasping.feasibility``.

The feasibility block is *scaffolding* — the operator's locked
direction (free-text answer to ``feasibility_signals_in_T2``):

  "Please use C as the option to choose. But document it right.
  Therefore, we can implement it later onwards and its not dead
  code, whichs never been seen in this code base again."

⇒ Every field exists in the schema today with a *safe disabled*
default so the structural wiring is durable, but no behavior
changes until the operator flips a flag.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.config.schema.robot import (
    RobotGraspingConfig,
)
from src.config.schema.robot.robot_schema import (
    GraspingFeasibilityConfig,
)


class GraspingFeasibilityConfigDefaultsTests(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        c = GraspingFeasibilityConfig()
        self.assertFalse(c.enabled)
        self.assertEqual(c.weight, 0.0)
        self.assertFalse(c.ik_quality_enabled)
        self.assertFalse(c.joint_margin_enabled)
        self.assertFalse(c.swept_approach_enabled)
        self.assertEqual(c.swept_approach_top_k, 5)
        # Sub-weights default to the locked ratio so operators only
        # need to flip ``enabled`` + per-signal flags + a single
        # top-level ``weight`` to opt in.
        self.assertAlmostEqual(c.ik_quality_weight, 0.4)
        self.assertAlmostEqual(c.joint_margin_weight, 0.3)
        self.assertAlmostEqual(c.swept_approach_weight, 0.3)
        # Apply-modes locks EASY out by default (operator answer Q1-B).
        self.assertEqual(
            set(c.apply_modes),
            {"auto", "dense_clutter", "dense_autonomous"},
        )

    def test_extra_fields_forbidden(self) -> None:
        with self.assertRaises(ValidationError):
            GraspingFeasibilityConfig(unknown_knob=True)  # type: ignore[call-arg]


class GraspingFeasibilityConfigValidationTests(unittest.TestCase):
    def test_weight_bounds(self) -> None:
        GraspingFeasibilityConfig(weight=0.0)
        GraspingFeasibilityConfig(weight=1.0)
        with self.assertRaises(ValidationError):
            GraspingFeasibilityConfig(weight=-0.1)
        with self.assertRaises(ValidationError):
            GraspingFeasibilityConfig(weight=1.5)

    def test_top_k_must_be_positive(self) -> None:
        GraspingFeasibilityConfig(swept_approach_top_k=1)
        with self.assertRaises(ValidationError):
            GraspingFeasibilityConfig(swept_approach_top_k=0)
        with self.assertRaises(ValidationError):
            GraspingFeasibilityConfig(swept_approach_top_k=-3)

    def test_sub_weights_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            GraspingFeasibilityConfig(ik_quality_weight=-0.1)
        with self.assertRaises(ValidationError):
            GraspingFeasibilityConfig(joint_margin_weight=2.0)

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            GraspingFeasibilityConfig(apply_modes=("nonexistent_mode",))


class RobotGraspingConfigFeasibilityIntegrationTests(unittest.TestCase):
    def test_grasping_block_carries_feasibility_default(self) -> None:
        g = RobotGraspingConfig()
        # The new sub-block is a sibling of `decision` / `closed_loop` /
        # `verification` / `dense_recovery`.
        self.assertIsInstance(g.feasibility, GraspingFeasibilityConfig)
        self.assertFalse(g.feasibility.enabled)


if __name__ == "__main__":
    unittest.main()
