"""Phase T3 schema tests for ``robot.grasping.occlusion``.

The occlusion sub-block carries the directional/corridor analysis
knobs. Operator-locked answers in this session:

* Q3: add now under ``robot.grasping.occlusion`` sub-block,
  additive, defaults off.
* Q1: ``apply_modes`` default = ``(auto, dense_clutter,
  dense_autonomous)`` — EASY locked out.
* Q2: ``hard_reject_enabled`` defaults to :data:`False`. T3 ships
  as demote-only.
* Q8: ``directional_enabled`` defaults to :data:`False`. Operator
  must opt in.
* Q5: ``top_k`` defaults to 5 (mirrors swept_approach_top_k).
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.config.schema.robot import RobotGraspingConfig
from src.config.schema.robot.robot_schema import (
    GraspingOcclusionConfig,
)


class GraspingOcclusionConfigDefaultsTests(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        c = GraspingOcclusionConfig()
        self.assertFalse(c.directional_enabled)
        self.assertFalse(c.hard_reject_enabled)
        # EASY locked out by default (Q1).
        self.assertNotIn("easy", c.apply_modes)
        self.assertIn("auto", c.apply_modes)
        self.assertIn("dense_clutter", c.apply_modes)
        self.assertIn("dense_autonomous", c.apply_modes)
        # Locked numeric defaults.
        self.assertGreater(c.corridor_radius_mm, 0.0)
        self.assertGreater(c.corridor_step_mm, 0.0)
        self.assertGreater(c.corridor_max_distance_mm, 0.0)
        self.assertGreaterEqual(c.mask_fusion_weight, 0.0)
        self.assertLessEqual(c.mask_fusion_weight, 1.0)
        self.assertGreaterEqual(c.partial_confidence_threshold, 0.0)
        self.assertGreater(
            c.hard_reject_confidence_threshold,
            c.partial_confidence_threshold,
        )
        self.assertEqual(c.top_k, 5)

    def test_extra_fields_forbidden(self) -> None:
        with self.assertRaises(ValidationError):
            GraspingOcclusionConfig(unknown_knob=True)  # type: ignore[call-arg]


class GraspingOcclusionConfigValidationTests(unittest.TestCase):
    def test_non_positive_radius_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            GraspingOcclusionConfig(corridor_radius_mm=0.0)
        with self.assertRaises(ValidationError):
            GraspingOcclusionConfig(corridor_radius_mm=-1.0)

    def test_non_positive_step_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            GraspingOcclusionConfig(corridor_step_mm=0.0)

    def test_non_positive_max_distance_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            GraspingOcclusionConfig(corridor_max_distance_mm=0.0)

    def test_mask_fusion_weight_outside_unit_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            GraspingOcclusionConfig(mask_fusion_weight=-0.1)
        with self.assertRaises(ValidationError):
            GraspingOcclusionConfig(mask_fusion_weight=1.1)

    def test_confidence_thresholds_in_unit_interval(self) -> None:
        with self.assertRaises(ValidationError):
            GraspingOcclusionConfig(partial_confidence_threshold=-0.1)
        with self.assertRaises(ValidationError):
            GraspingOcclusionConfig(hard_reject_confidence_threshold=1.1)

    def test_top_k_must_be_positive(self) -> None:
        GraspingOcclusionConfig(top_k=1)
        with self.assertRaises(ValidationError):
            GraspingOcclusionConfig(top_k=0)
        with self.assertRaises(ValidationError):
            GraspingOcclusionConfig(top_k=-3)

    def test_unknown_apply_mode_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            GraspingOcclusionConfig(apply_modes=("nope",))


class RobotGraspingConfigOcclusionIntegrationTests(unittest.TestCase):
    def test_grasping_block_carries_occlusion_default(self) -> None:
        g = RobotGraspingConfig()
        self.assertIsInstance(g.occlusion, GraspingOcclusionConfig)
        self.assertFalse(g.occlusion.directional_enabled)
        self.assertFalse(g.occlusion.hard_reject_enabled)


if __name__ == "__main__":
    unittest.main()
