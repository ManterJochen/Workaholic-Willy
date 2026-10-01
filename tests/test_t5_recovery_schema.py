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
            ("auto", "dense_clutter"),
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

    def test_container_agitate_without_fixture_rejected(self) -> None:
        """``container_agitate`` needs the fixture box its waypoints stay inside, a declared container or not."""

        with self.assertRaises(ValidationError) as caught:
            _instantiate(
                recovery={
                    "enabled": True,
                    "allowed_actions": ["container_agitate"],
                    "fixture": None,
                },
                support={"container": {"interior_min_mm": [-200.0, -900.0, 20.0],
                                       "interior_max_mm": [200.0, -500.0, 180.0]}},
            )
        self.assertIn("recovery.fixture", str(caught.exception))

    def test_nudge_target_without_fixture_accepted(self) -> None:
        """The push needs no fixture (owner, 2026-10-01): the automatic push box bounds where it lands."""

        cfg = _instantiate(
            recovery={
                "enabled": True,
                "allowed_actions": ["nudge_target"],
                "fixture": None,
            }
        )
        self.assertIsNone(cfg.recovery.fixture)
        self.assertEqual(cfg.recovery.allowed_actions, ("nudge_target",))

    def test_physical_action_with_fixture_accepted(self) -> None:
        cfg = _instantiate(
            recovery={
                "enabled": True,
                "allowed_actions": ["nudge_target"],
                "fixture": {
                    "center_mm": [0.0, 0.0, 100.0],
                    "half_extents_mm": [200.0, 200.0, 50.0],
                    # 5 mm until 2026-09-29; a ceiling under 10 mm allows no push and is refused at load now.
                    "max_nudge_mm": 30.0,
                },
            }
        )
        self.assertIsNotNone(cfg.recovery.fixture)


class RecoveryNonPhysicalAllowedTests(unittest.TestCase):
    def test_non_physical_actions_accepted_without_fixture(self) -> None:
        cfg = _instantiate(
            recovery={
                "enabled": True,
                "allowed_actions": ["rescan", "next_target"],
            }
        )
        self.assertEqual(
            cfg.recovery.allowed_actions,
            ("rescan", "next_target"),
        )
        # `next_viewpoint`, the third non-physical action until it was merged into `rescan` on
        # 2026-09-29, is refused with the action to name instead.
        with self.assertRaises(ValidationError) as caught:
            _instantiate(recovery={"enabled": True, "allowed_actions": ["next_viewpoint"]})
        self.assertIn("removed on purpose: use rescan", str(caught.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
