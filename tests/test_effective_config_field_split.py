"""Byte-identity guard for the ``EffectiveGraspingConfig`` flat dict.

``to_dict()`` emits a flat dict in a fixed order with fixed value transforms; it feeds
``outcome_logging.json_safe`` (serialized with ``sort_keys=False``), so key order is load-bearing for
replay/KPI diffs. The rest of the suite only does subset (``assertIn``) checks on ``to_dict()``, so a
silent key reorder or drop would pass everywhere else. This file is the explicit lock on key order,
count, and the value transforms.
"""

from __future__ import annotations

import json
import unittest

from src.robot.execution.autonomous_grasp import (
    EffectiveGraspingConfig,
    EffectiveRecoveryOrchestratorConfig,
    GraspMode,
)

# The canonical flat key order emitted by ``EffectiveGraspingConfig.to_dict()``. The decision, feasibility,
# corridor, ordering, uncertainty, watchdog and performance keys, and ``success_model_enabled`` and
# ``approach_validation_enabled``, left on 2026-10-16 with their blocks (the owner's cleanup).
EXPECTED_KEYS: tuple[str, ...] = (
    "default_mode",
    "max_attempts",
    "recovery_orchestrator_enabled",
    "recovery_orchestrator_max_actions",
    "recovery_orchestrator_apply_modes",
    "recovery_orchestrator_allowed_actions",
    "recovery_orchestrator_per_action_budget",
    "fusion_enabled",
)


def _bare() -> EffectiveGraspingConfig:
    """A bare snapshot: the 2 core fields set, the recovery sub-config defaulted."""

    return EffectiveGraspingConfig(
        default_mode=GraspMode.AUTO,
        max_attempts=5,
    )


class ToDictByteIdentityTests(unittest.TestCase):
    def test_key_order_and_count_is_canonical(self) -> None:
        keys = tuple(_bare().to_dict().keys())
        self.assertEqual(keys, EXPECTED_KEYS)
        for gone in ("decision_enabled", "feasibility_enabled", "corridor_directional_enabled", "ordering_enabled",
                     "uncertainty_enabled", "watchdog_mode", "performance_enabled", "success_model_enabled",
                     "approach_validation_enabled"):
            self.assertNotIn(gone, keys)

    def test_value_transforms_preserved(self) -> None:
        d = _bare().to_dict()
        # enum -> stable string
        self.assertEqual(d["default_mode"], "auto")
        self.assertIsInstance(d["default_mode"], str)
        # tuple -> list
        self.assertEqual(d["recovery_orchestrator_allowed_actions"], [])
        self.assertEqual(d["recovery_orchestrator_apply_modes"], ["auto", "dense_clutter"])
        self.assertEqual(d["recovery_orchestrator_per_action_budget"], [])
        # the whole thing must be JSON-serializable (json_safe contract)
        json.dumps(d)

    def test_per_action_budget_list_of_pairs_transform(self) -> None:
        eff = EffectiveGraspingConfig(
            default_mode=GraspMode.DENSE_CLUTTER,
            max_attempts=5,
            recovery_orchestrator=EffectiveRecoveryOrchestratorConfig(
                per_action_budget=(("rescan", 1), ("next_target", 2)),
            ),
        )
        self.assertEqual(
            eff.to_dict()["recovery_orchestrator_per_action_budget"],
            [["rescan", 1], ["next_target", 2]],
        )


class NestedStructureTests(unittest.TestCase):
    def test_nested_attrs_agree_with_flat_to_dict(self) -> None:
        eff = EffectiveGraspingConfig(
            default_mode=GraspMode.AUTO,
            max_attempts=5,
            fusion_enabled=True,
            recovery_orchestrator=EffectiveRecoveryOrchestratorConfig(enabled=True, max_actions=3),
        )
        d = eff.to_dict()
        self.assertTrue(d["fusion_enabled"])
        self.assertEqual(eff.recovery_orchestrator.max_actions, 3)
        self.assertEqual(d["recovery_orchestrator_max_actions"], 3)
        self.assertTrue(d["recovery_orchestrator_enabled"])

    def test_snapshot_is_frozen_and_hashable(self) -> None:
        eff = _bare()
        self.assertIsInstance(hash(eff), int)
        with self.assertRaises(Exception):
            eff.recovery_orchestrator = EffectiveRecoveryOrchestratorConfig()  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
