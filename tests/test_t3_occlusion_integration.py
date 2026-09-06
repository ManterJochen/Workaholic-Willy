"""Phase T3 integration — occlusion settings flow from YAML through
:class:`AutonomousGraspService.from_robot_config` to a typed snapshot
that downstream calculator/runtime layers can consume.

Locked invariants:

1. ``robot.grasping.occlusion.directional_enabled = False`` (default)
   ⇒ snapshot reports corridor analysis as inactive, hard-reject off,
   feasibility corridor_risk weight 0.0. T2 byte-identical state
   preserved.
2. ``directional_enabled = True`` and active mode in ``apply_modes``
   ⇒ snapshot reflects the YAML values.
3. ``directional_enabled = True`` and active mode *not* in
   ``apply_modes`` (canonical: EASY) ⇒ snapshot reports corridor
   analysis as inactive for that service. EASY services cannot
   accidentally consume corridor-aware ranking.
4. ``hard_reject_enabled`` defaults False — the new
   ``CORRIDOR_BLOCKED`` reason code is *not* emitted by default.

These tests do not assert calculator behaviour; that is covered by
the analyzer + feasibility integration suites.
"""

from __future__ import annotations

import unittest


class EffectiveGraspingConfigCorridorExtensionTests(unittest.TestCase):
    def test_snapshot_has_corridor_fields(self) -> None:
        from src.robot.execution.autonomous_grasp import (
            EffectiveGraspingConfig,
            GraspMode,
        )

        # The corridor/feasibility phase fields moved into nested sub-configs,
        # so they are no longer top-level ``__dataclass_fields__``. The flat
        # serialization contract (``to_dict``) still emits every historical
        # flat key, so assert against a bare snapshot's serialized keys.
        bare = EffectiveGraspingConfig(
            default_mode=GraspMode.AUTO,
            max_attempts=5,
            closed_loop_enabled=False,
            verification_enabled=False,
            dense_recovery_enabled=False,
            dense_recovery_allowed_actions=(),
        )
        fields = set(bare.to_dict().keys())
        # The 9 new fields locked in the T3 plan.
        for name in (
            "corridor_directional_enabled",
            "corridor_hard_reject_enabled",
            "corridor_top_k",
            "corridor_radius_mm",
            "corridor_step_mm",
            "corridor_max_distance_mm",
            "corridor_mask_fusion_weight",
            "corridor_hard_reject_confidence_threshold",
            "feasibility_corridor_risk_enabled",
            "feasibility_corridor_risk_weight",
        ):
            self.assertIn(name, fields)

    def test_to_dict_includes_corridor_keys(self) -> None:
        from src.robot.execution.autonomous_grasp import (
            EffectiveGraspingConfig,
            GraspMode,
        )

        snap = EffectiveGraspingConfig(
            default_mode=GraspMode.AUTO,
            max_attempts=5,
            closed_loop_enabled=False,
            verification_enabled=False,
            dense_recovery_enabled=False,
            dense_recovery_allowed_actions=(),
        )
        d = snap.to_dict()
        for key in (
            "corridor_directional_enabled",
            "corridor_hard_reject_enabled",
            "corridor_top_k",
            "corridor_radius_mm",
            "corridor_step_mm",
            "corridor_max_distance_mm",
            "corridor_mask_fusion_weight",
            "corridor_hard_reject_confidence_threshold",
            "feasibility_corridor_risk_enabled",
            "feasibility_corridor_risk_weight",
        ):
            self.assertIn(key, d)
        # Defaults preserve byte-equivalent T2 behavior.
        self.assertEqual(d["corridor_directional_enabled"], False)
        self.assertEqual(d["corridor_hard_reject_enabled"], False)
        self.assertEqual(d["feasibility_corridor_risk_enabled"], False)
        self.assertEqual(d["feasibility_corridor_risk_weight"], 0.0)
        self.assertEqual(d["corridor_top_k"], 5)


class OcclusionApplyModesEasyExclusionTests(unittest.TestCase):
    def test_easy_mode_excluded_by_default(self) -> None:
        from src.config.schema.robot.robot_schema import (
            GraspingOcclusionConfig,
        )

        c = GraspingOcclusionConfig(directional_enabled=True)
        self.assertNotIn("easy", c.apply_modes)
        self.assertIn("auto", c.apply_modes)
        self.assertIn("dense_clutter", c.apply_modes)
        self.assertIn("dense_autonomous", c.apply_modes)


if __name__ == "__main__":
    unittest.main()
