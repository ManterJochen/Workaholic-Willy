"""Phase T4 schema-level tests for ``robot.grasping.ordering``.

Contracts:

1. ``GraspingOrderingConfig`` defaults to OFF and mirrors the T2/T3
   apply-modes default ``(auto, dense_clutter, dense_autonomous)``
   (EASY locked out for the +5% cycle-time budget).
2. Numeric validators reject negatives / out-of-range values at
   construction time.
3. ``apply_modes`` entries must be in the locked
   ``_KNOWN_GRASP_MODES`` set; unknown modes raise.
4. ``RobotGraspingConfig`` exposes ``.ordering`` as a sibling of
   ``.feasibility`` and ``.occlusion``.
5. The nested ``blocker_graph`` sub-block carries its own three
   per-signal flags, all default off.
"""

from __future__ import annotations

import unittest

from src.config.schema.robot.robot_schema import (
    GraspingOrderingConfig,
    RobotGraspingConfig,
)


class DefaultsTests(unittest.TestCase):
    def test_top_level_defaults(self) -> None:
        cfg = GraspingOrderingConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.unlock_weight, 0.0)
        self.assertEqual(cfg.max_local_score_drop, 0.1)
        self.assertEqual(
            cfg.apply_modes,
            ("auto", "dense_clutter", "dense_autonomous"),
        )

    def test_blocker_graph_defaults(self) -> None:
        cfg = GraspingOrderingConfig()
        bg = cfg.blocker_graph
        self.assertFalse(bg.mask_adjacency_enabled)
        self.assertFalse(bg.depth_only_enabled)
        self.assertFalse(bg.corridor_overlap_enabled)
        self.assertEqual(bg.adjacency_radius_px, 5)
        self.assertEqual(bg.depth_tolerance_mm, 10.0)


class ValidatorTests(unittest.TestCase):
    def test_rejects_negative_unlock_weight(self) -> None:
        with self.assertRaises(Exception):
            GraspingOrderingConfig(unlock_weight=-0.1)

    def test_rejects_out_of_range_max_local_drop(self) -> None:
        with self.assertRaises(Exception):
            GraspingOrderingConfig(max_local_score_drop=-0.1)
        with self.assertRaises(Exception):
            GraspingOrderingConfig(max_local_score_drop=1.5)

    def test_rejects_unknown_apply_mode(self) -> None:
        with self.assertRaises(Exception):
            GraspingOrderingConfig(apply_modes=("nonexistent",))

    def test_easy_excluded_from_default_apply_modes(self) -> None:
        cfg = GraspingOrderingConfig()
        self.assertNotIn("easy", cfg.apply_modes)


class MountedOnRobotGraspingTests(unittest.TestCase):
    def test_ordering_is_mounted(self) -> None:
        rg = RobotGraspingConfig()
        self.assertIsInstance(rg.ordering, GraspingOrderingConfig)
        self.assertFalse(rg.ordering.enabled)


if __name__ == "__main__":
    unittest.main()
