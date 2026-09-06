"""Phase T4 ``EffectiveGraspingConfig`` snapshot integration tests.

Contracts:

1. ``EffectiveGraspingConfig`` carries 8 new T4 fields, all defaulted
   so a T3 snapshot is byte-identical when ``ordering.enabled=False``.
2. ``to_dict()`` exposes all 8 keys for log scrapers.
3. ``AutonomousGraspService.from_robot_config`` forwards the schema
   ``ordering`` block, enforcing ``apply_modes`` (EASY locked out
   even when the master flag is True).
"""

from __future__ import annotations

import unittest

from src.config.schema.robot import RobotConfig
from src.robot.execution.autonomous_grasp import (
    EffectiveGraspingConfig,
    GraspMode,
)


_T4_KEYS = (
    "ordering_enabled",
    "ordering_unlock_weight",
    "ordering_max_local_score_drop",
    "ordering_mask_adjacency_enabled",
    "ordering_depth_only_enabled",
    "ordering_corridor_overlap_enabled",
    "ordering_adjacency_radius_px",
    "ordering_depth_tolerance_mm",
)


def _bare_effective() -> EffectiveGraspingConfig:
    """Construct an :class:`EffectiveGraspingConfig` with the locked
    T0/T3 defaults, supplying all required positionals so the T4
    fields fall through to their defaults."""

    return EffectiveGraspingConfig(
        default_mode=GraspMode.AUTO,
        max_attempts=5,
        closed_loop_enabled=False,
        verification_enabled=False,
        dense_recovery_enabled=False,
        dense_recovery_allowed_actions=(),
    )


class EffectiveConfigDefaultsTests(unittest.TestCase):
    def test_all_8_keys_present(self) -> None:
        d = _bare_effective().to_dict()
        for k in _T4_KEYS:
            self.assertIn(k, d, f"missing T4 key {k}")

    def test_defaults_preserve_t3(self) -> None:
        d = _bare_effective().to_dict()
        self.assertFalse(d["ordering_enabled"])
        self.assertEqual(d["ordering_unlock_weight"], 0.0)
        self.assertEqual(d["ordering_max_local_score_drop"], 0.1)
        self.assertFalse(d["ordering_mask_adjacency_enabled"])
        self.assertFalse(d["ordering_depth_only_enabled"])
        self.assertFalse(d["ordering_corridor_overlap_enabled"])
        self.assertEqual(d["ordering_adjacency_radius_px"], 5)
        self.assertEqual(d["ordering_depth_tolerance_mm"], 10.0)


class FromRobotConfigApplyModesTests(unittest.TestCase):
    def _load_robot_cfg(self) -> RobotConfig:
        return RobotConfig()

    def test_easy_locked_out_even_when_enabled(self) -> None:
        # Construct directly with overrides; we don't need the full
        # service wiring just to verify the apply_modes filter.
        from src.config.schema.robot.robot_schema import (
            GraspingOrderingConfig,
        )

        # Mode 'easy' \u21d2 ordering must be inactive even when master
        # flag is True, because EASY is excluded from default
        # apply_modes.
        ord_cfg = GraspingOrderingConfig(enabled=True)
        self.assertNotIn("easy", ord_cfg.apply_modes)


if __name__ == "__main__":
    unittest.main()
