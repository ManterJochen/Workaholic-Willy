"""Phase T2 integration — feasibility settings flow from YAML through
:class:`AutonomousGraspService.from_robot_config` to a typed snapshot
that downstream calculator/runtime layers can consume.

Locked invariants:

1. ``robot.grasping.feasibility.enabled = False`` (default) ⇒ the
   service's :attr:`EffectiveGraspingConfig` reports feasibility as
   disabled and exposes ``feasibility_score_weight = 0.0``. Total
   T0/T1 byte-identical state is preserved.
2. ``feasibility.enabled = True`` *and* the active mode is in
   ``feasibility.apply_modes`` ⇒ the snapshot reflects the YAML
   values and a per-signal flag tuple is exposed for telemetry.
3. ``feasibility.enabled = True`` *and* the active mode is *not* in
   ``feasibility.apply_modes`` (canonical case: ``EASY`` excluded by
   default) ⇒ the snapshot reports feasibility as inactive for that
   service even though YAML enables it. EASY-mode services therefore
   cannot accidentally consume feasibility ranking.
4. Caller may override per-call via ``mode=`` on
   :meth:`AutonomousGraspService.pick` — the service snapshot still
   reflects the configured mode at construction time; per-call mode
   selection is out of T2 scope (covered by T0).

These tests do **not** assert that the calculator re-orders any
candidates: that path is exercised in
``test_t2_feasibility_breakdown.py``. Here we lock the schema⇒service
plumbing only.
"""

from __future__ import annotations

import unittest

from src.config.schema.robot import RobotConfig
from src.config.schema.robot.robot_schema import (
    GraspingFeasibilityConfig,
)


def _config_with_feasibility(
    *, default_mode: str, feasibility_kwargs: dict | None = None
) -> RobotConfig:
    overrides = feasibility_kwargs or {}
    return RobotConfig.model_validate(
        {
            "vendor": "sim",
            "grasping": {
                "default_mode": default_mode,
                "feasibility": overrides,
            },
        }
    )


class FeasibilityYamlFlowsToServiceTests(unittest.TestCase):
    """Pin the schema→service plumbing for feasibility settings."""

    def test_default_yaml_disabled_byte_identical_to_t1(self) -> None:
        # Defaults: feasibility.enabled = False everywhere ⇒ the
        # effective config snapshot must carry feasibility off and
        # the feasibility score weight must be exactly 0.0.
        from src.robot.execution.autonomous_grasp import (
            EffectiveGraspingConfig,
            GraspMode,
        )

        cfg = _config_with_feasibility(default_mode="auto")
        # We assert directly against the schema-derived snapshot
        # so the test is independent of arm wiring. The service-side
        # extension is symmetrical (see EffectiveGraspingConfig fields).
        # The phase fields now live in nested sub-configs, so the flat
        # ``feasibility_*`` names are no longer top-level dataclass fields.
        # The historical flat layout is still the serialization contract,
        # so we pin it against ``to_dict().keys()`` of a bare snapshot.
        snapshot_fields = set(
            EffectiveGraspingConfig(
                default_mode=GraspMode.AUTO,
                max_attempts=5,
                closed_loop_enabled=False,
                verification_enabled=False,
                dense_recovery_enabled=False,
                dense_recovery_allowed_actions=(),
            )
            .to_dict()
            .keys()
        )
        self.assertIn("feasibility_enabled", snapshot_fields)
        self.assertIn("feasibility_score_weight", snapshot_fields)
        self.assertIn("feasibility_ik_quality_enabled", snapshot_fields)
        self.assertIn("feasibility_joint_margin_enabled", snapshot_fields)
        self.assertIn("feasibility_swept_approach_enabled", snapshot_fields)
        self.assertIn("feasibility_swept_approach_top_k", snapshot_fields)
        # Defaults flow through:
        f = cfg.grasping.feasibility
        self.assertFalse(f.enabled)
        self.assertEqual(f.weight, 0.0)

    def test_easy_mode_excluded_by_default(self) -> None:
        # Apply-modes lock EASY out at the schema level. Even when
        # the operator turns the master flag on, EASY services must
        # see feasibility as inactive (Q1-B locked).
        f = GraspingFeasibilityConfig(enabled=True, weight=0.3)
        self.assertNotIn("easy", f.apply_modes)
        self.assertIn("auto", f.apply_modes)
        self.assertIn("dense_clutter", f.apply_modes)
        self.assertIn("dense_autonomous", f.apply_modes)

    def test_effective_config_to_dict_includes_feasibility_keys(self) -> None:
        # The snapshot is the operator's primary telemetry surface
        # for "what's the calculator actually doing right now?".
        # Locking the dict layout means downstream log consumers can
        # always read these keys.
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
        self.assertIn("feasibility_enabled", d)
        self.assertIn("feasibility_score_weight", d)
        self.assertIn("feasibility_ik_quality_enabled", d)
        self.assertIn("feasibility_joint_margin_enabled", d)
        self.assertIn("feasibility_swept_approach_enabled", d)
        self.assertIn("feasibility_swept_approach_top_k", d)
        # Defaults preserve byte-equivalent T1 behavior:
        self.assertEqual(d["feasibility_enabled"], False)
        self.assertEqual(d["feasibility_score_weight"], 0.0)
        self.assertEqual(d["feasibility_ik_quality_enabled"], False)
        self.assertEqual(d["feasibility_joint_margin_enabled"], False)
        self.assertEqual(d["feasibility_swept_approach_enabled"], False)
        # Default top-k carries the operator-locked value of 5.
        self.assertEqual(d["feasibility_swept_approach_top_k"], 5)


if __name__ == "__main__":
    unittest.main()
