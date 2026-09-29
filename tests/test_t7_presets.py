"""Phase T7 — RED tests for operator-safe mode presets.

Q4=B + Q5=A locked three preset overlays; two remain:

* ``easy`` — EASY mode, conservative defaults; recovery off; uncertainty
  layer off (legacy threshold path).
* ``dense_clutter`` — DENSE_CLUTTER mode, recovery on with
  ``rescan`` allowed only (no physical motion by default; it allowed
  ``next_viewpoint`` until that was merged into ``rescan`` on 2026-09-29),
  uncertainty layer ENABLED with the locked AUTO threshold.

``verification_heavy`` (CLOSED_LOOP mode, recovery on) was deleted on 2026-09-29 with the
``closed_loop`` mode it named: without the two-scan refinement it had no purpose left.

Contract:

* ``list_presets()`` returns the two preset names sorted.
* ``load_preset(name)`` returns a plain ``dict`` overlay shaped like
  the ``robot:`` block (so it can be deep-merged into ``robot.yaml``).
* ``apply_preset(base, name)`` returns a new dict where the overlay
  has been deep-merged onto ``base`` (base is not mutated).
* Each preset's overlay loads cleanly into ``RobotConfig`` when
  merged onto a minimal base, demonstrating schema compatibility
  end-to-end.
"""

from __future__ import annotations

import unittest

from src.config.schema.robot import RobotConfig
from src.robot.grasping.replay.presets import (
    apply_preset,
    list_presets,
    load_preset,
)


_MINIMAL_BASE: dict = {
    "vendor": "dummy",
    "gripper": {"vendor": "none"},
    "grasping": {"default_mode": "auto"},
}


class PresetCatalogTests(unittest.TestCase):
    def test_two_presets_shipped(self) -> None:
        self.assertEqual(
            list_presets(),
            ["dense_clutter", "easy"],
        )

    def test_load_preset_returns_dict(self) -> None:
        for name in list_presets():
            overlay = load_preset(name)
            self.assertIsInstance(overlay, dict)
            self.assertIn("grasping", overlay)

    def test_unknown_preset_raises(self) -> None:
        with self.assertRaises(KeyError):
            load_preset("not_a_preset")


class EasyPresetTests(unittest.TestCase):
    def test_default_mode_is_easy(self) -> None:
        merged = apply_preset(_MINIMAL_BASE, "easy")
        self.assertEqual(merged["grasping"]["default_mode"], "easy")

    def test_recovery_off(self) -> None:
        merged = apply_preset(_MINIMAL_BASE, "easy")
        self.assertFalse(merged["grasping"]["recovery"]["enabled"])

    def test_uncertainty_off(self) -> None:
        merged = apply_preset(_MINIMAL_BASE, "easy")
        # When the layer is off the new threshold key is not active;
        # legacy decision threshold path remains in effect.
        self.assertFalse(merged["grasping"]["uncertainty"]["enabled"])

    def test_loads_into_robot_config(self) -> None:
        merged = apply_preset(_MINIMAL_BASE, "easy")
        RobotConfig(**merged)  # must not raise


class DenseClutterPresetTests(unittest.TestCase):
    def test_default_mode_is_dense_clutter(self) -> None:
        merged = apply_preset(_MINIMAL_BASE, "dense_clutter")
        self.assertEqual(merged["grasping"]["default_mode"], "dense_clutter")

    def test_recovery_on_with_rescan_only(self) -> None:
        merged = apply_preset(_MINIMAL_BASE, "dense_clutter")
        recovery = merged["grasping"]["recovery"]
        self.assertTrue(recovery["enabled"])
        self.assertEqual(list(recovery["allowed_actions"]), ["rescan"])

    def test_uncertainty_enabled_with_locked_threshold(self) -> None:
        merged = apply_preset(_MINIMAL_BASE, "dense_clutter")
        unc = merged["grasping"]["uncertainty"]
        self.assertTrue(unc["enabled"])
        self.assertAlmostEqual(unc["fail_closed_threshold"], 0.4)

    def test_loads_into_robot_config(self) -> None:
        merged = apply_preset(_MINIMAL_BASE, "dense_clutter")
        RobotConfig(**merged)


class ApplyPresetSemanticsTests(unittest.TestCase):
    def test_apply_preset_does_not_mutate_base(self) -> None:
        snapshot = dict(_MINIMAL_BASE)
        snapshot_grasping = dict(_MINIMAL_BASE["grasping"])
        apply_preset(_MINIMAL_BASE, "easy")
        self.assertEqual(_MINIMAL_BASE["grasping"], snapshot_grasping)
        self.assertEqual(set(_MINIMAL_BASE), set(snapshot))

    def test_deep_merge_preserves_unrelated_keys(self) -> None:
        base = {
            "vendor": "dummy",
            "gripper": {"vendor": "none"},
            "grasping": {
                "default_mode": "auto",
                "max_attempts": 9,
            },
        }
        merged = apply_preset(base, "easy")
        # max_attempts was not overridden by the preset
        self.assertEqual(merged["grasping"]["max_attempts"], 9)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
