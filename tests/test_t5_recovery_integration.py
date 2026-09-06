"""Phase T5 — integration tests for ``EffectiveGraspingConfig``.

Mirrors the T4 integration pattern: verify that

1. the snapshot carries the new T5 ``recovery_orchestrator_*`` keys,
2. defaults are byte-identical to T4 (every flag OFF, EASY excluded),
3. the schema's ``apply_modes`` filter permanently excludes EASY.
"""

from __future__ import annotations

import unittest

from src.robot.execution.autonomous_grasp import (
    EffectiveGraspingConfig,
    EffectiveRecoveryOrchestratorConfig,
    GraspMode,
)


_T5_KEYS = (
    "recovery_orchestrator_enabled",
    "recovery_orchestrator_max_actions",
    "recovery_orchestrator_apply_modes",
    "recovery_orchestrator_allowed_actions",
    "recovery_orchestrator_per_action_budget",
)


def _bare_effective() -> EffectiveGraspingConfig:
    return EffectiveGraspingConfig(
        default_mode=GraspMode.AUTO,
        max_attempts=5,
        closed_loop_enabled=False,
        verification_enabled=False,
        dense_recovery_enabled=False,
        dense_recovery_allowed_actions=(),
    )


class EffectiveConfigT5DefaultsTests(unittest.TestCase):
    def test_all_t5_keys_present(self) -> None:
        d = _bare_effective().to_dict()
        for key in _T5_KEYS:
            self.assertIn(key, d, f"missing T5 key {key}")

    def test_defaults_preserve_t4_snapshot(self) -> None:
        d = _bare_effective().to_dict()
        self.assertFalse(d["recovery_orchestrator_enabled"])
        self.assertEqual(d["recovery_orchestrator_max_actions"], 2)
        self.assertEqual(
            d["recovery_orchestrator_apply_modes"],
            ["auto", "dense_clutter", "dense_autonomous"],
        )
        self.assertEqual(d["recovery_orchestrator_allowed_actions"], [])
        self.assertEqual(d["recovery_orchestrator_per_action_budget"], [])

    def test_per_action_budget_jsonifies_to_list_of_pairs(self) -> None:
        eff = EffectiveGraspingConfig(
            default_mode=GraspMode.DENSE_CLUTTER,
            max_attempts=5,
            closed_loop_enabled=False,
            verification_enabled=False,
            dense_recovery_enabled=False,
            dense_recovery_allowed_actions=(),
            recovery_orchestrator=EffectiveRecoveryOrchestratorConfig(
                per_action_budget=(
                    ("rescan", 1),
                    ("next_viewpoint", 2),
                ),
            ),
        )
        d = eff.to_dict()
        self.assertEqual(
            d["recovery_orchestrator_per_action_budget"],
            [["rescan", 1], ["next_viewpoint", 2]],
        )


class SchemaApplyModesLockoutTests(unittest.TestCase):
    def test_easy_excluded_from_default_apply_modes(self) -> None:
        from src.config.schema.robot.robot_schema import (
            GraspingRecoveryConfig,
        )

        rec = GraspingRecoveryConfig(enabled=True)
        self.assertNotIn("easy", rec.apply_modes)
        self.assertEqual(
            rec.apply_modes,
            ("auto", "dense_clutter", "dense_autonomous"),
        )


if __name__ == "__main__":
    unittest.main()
