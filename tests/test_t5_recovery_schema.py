"""Phase T5 — RED tests for the recovery YAML/schema surface.

T5 adds an additive ``robot.grasping.recovery.policy`` block. All flags
default OFF so an unmodified T4 YAML loads byte-identically.

The schema validates:

* enabled flag and per-action budget shape,
* allowed_actions strings against the SceneRecoveryAction vocabulary,
* apply_modes membership (no EASY by default, no unknown modes),
* max_recovery_actions non-negative,
* mount point on :class:`RobotGraspingConfig`.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.config.schema.robot.robot_schema import (
    RobotGraspingConfig,
)


def _instantiate(**kwargs):
    return RobotGraspingConfig(**kwargs)


class RecoverySchemaDefaultsTests(unittest.TestCase):
    def test_recovery_block_present_with_defaults(self) -> None:
        cfg = _instantiate()
        rec = cfg.recovery
        self.assertFalse(rec.enabled)
        self.assertEqual(rec.max_recovery_actions, 2)
        self.assertEqual(rec.allowed_actions, ())
        self.assertEqual(rec.per_action_budget, ())
        self.assertEqual(
            rec.apply_modes,
            ("auto", "dense_clutter", "dense_autonomous"),
        )

    def test_easy_not_in_default_apply_modes(self) -> None:
        cfg = _instantiate()
        self.assertNotIn("easy", cfg.recovery.apply_modes)


class RecoverySchemaValidationTests(unittest.TestCase):
    def test_unknown_action_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            _instantiate(
                recovery={
                    "enabled": True,
                    "allowed_actions": ["bogus_action"],
                }
            )

    def test_unknown_apply_mode_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            _instantiate(
                recovery={
                    "apply_modes": ["bogus_mode"],
                }
            )

    def test_negative_max_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            _instantiate(recovery={"max_recovery_actions": -1})

    def test_per_action_budget_unknown_action_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            _instantiate(
                recovery={
                    "per_action_budget": [["bogus_action", 1]],
                }
            )

    def test_per_action_budget_negative_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            _instantiate(
                recovery={
                    "per_action_budget": [["rescan", -1]],
                }
            )

    def test_physical_action_without_fixture_rejected(self) -> None:
        """If allowed_actions contains a physical action, fixture must be set."""

        with self.assertRaises(ValidationError):
            _instantiate(
                recovery={
                    "enabled": True,
                    "allowed_actions": ["nudge_target"],
                    "fixture": None,
                }
            )

    def test_physical_action_with_fixture_accepted(self) -> None:
        cfg = _instantiate(
            recovery={
                "enabled": True,
                "allowed_actions": ["nudge_target"],
                "fixture": {
                    "center_mm": [0.0, 0.0, 100.0],
                    "half_extents_mm": [200.0, 200.0, 50.0],
                    "max_nudge_mm": 5.0,
                },
            }
        )
        self.assertIsNotNone(cfg.recovery.fixture)


class RecoveryNonPhysicalAllowedTests(unittest.TestCase):
    def test_non_physical_actions_accepted_without_fixture(self) -> None:
        cfg = _instantiate(
            recovery={
                "enabled": True,
                "allowed_actions": ["rescan", "next_viewpoint", "next_target"],
            }
        )
        self.assertEqual(
            cfg.recovery.allowed_actions,
            ("rescan", "next_viewpoint", "next_target"),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
