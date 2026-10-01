"""The detector plan: recipe v1, the smoke and full tiers, the caller's own choices winning, and the refusals."""

from __future__ import annotations

import unittest

from src.models.detection.closed_set.training.plan import DetectorPlan, DetectorPlanOverrides, build_plan
from src.models.detection.closed_set.training.recipes import RECIPES, TIERS


class PlanTests(unittest.TestCase):
    def test_the_defaults_are_recipe_v1_at_the_full_tier(self) -> None:
        plan, applied = build_plan(recipe="v1", tier="full")
        defaults = DetectorPlan()
        for key, value in {**RECIPES["v1"], **{k: v for k, v in TIERS["full"].items() if k != "why"}}.items():
            self.assertEqual(value, getattr(plan, key), key)
            self.assertEqual(value, getattr(defaults, key), f"{key}: the plan default and the recipe disagree")
        self.assertEqual(("v1", "full", []), (plan.recipe, plan.tier, applied["overridden"]))

    def test_the_smoke_tier_narrows_the_run(self) -> None:
        plan, _ = build_plan(recipe="v1", tier="smoke")
        self.assertEqual((2, 64, False, 0), (plan.epochs, plan.max_train_images, plan.multiscale, plan.patience))

    def test_what_the_caller_chose_wins_over_the_tier(self) -> None:
        plan, applied = build_plan(recipe="v1", tier="smoke", overrides=DetectorPlanOverrides(epochs=5, patience=3))
        self.assertEqual((5, 3), (plan.epochs, plan.patience))
        self.assertEqual(["epochs=5", "patience=3"], sorted(applied["overridden"]))
        self.assertNotIn("epochs", applied)

    def test_only_chosen_fields_are_forwarded(self) -> None:
        self.assertEqual({"batch": 4}, DetectorPlanOverrides(batch=4).forwarded())
        self.assertEqual({"workers": None}, DetectorPlanOverrides(workers=None).forwarded())

    def test_an_unknown_recipe_or_tier_is_refused_by_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown detector recipe 'v2'; known: v1"):
            build_plan(recipe="v2")
        with self.assertRaisesRegex(ValueError, "unknown training tier 'fast'; known: full, smoke"):
            build_plan(tier="fast")

    def test_settings_no_run_can_use_are_refused(self) -> None:
        cases = {
            "image_size": (DetectorPlanOverrides(image_size=100), "multiple of 32"),
            "amp": (DetectorPlanOverrides(amp="fp8"), "unknown amp"),
            "epochs": (DetectorPlanOverrides(epochs=0), "epochs must be at least 1"),
            "val_fraction": (DetectorPlanOverrides(val_fraction=1.0), "below 1"),
            "ema_decay": (DetectorPlanOverrides(ema_decay=1.0), "between 0 and 1"),
            "patience": (DetectorPlanOverrides(patience=-1), "never stops early"),
        }
        for name, (overrides, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                build_plan(overrides=overrides)

    def test_the_strong_augmentations_stop_for_the_last_tenth(self) -> None:
        self.assertEqual(45, DetectorPlan(epochs=50).stop_augment_epoch())
        self.assertEqual(1, DetectorPlan(epochs=2).stop_augment_epoch())
        self.assertEqual(10, DetectorPlan(epochs=10, no_aug_epochs=0).stop_augment_epoch())
        self.assertEqual(7, DetectorPlan(epochs=10, no_aug_epochs=3).stop_augment_epoch())
        self.assertEqual(0, DetectorPlan(epochs=10, augment=False, multiscale=False).stop_augment_epoch())


if __name__ == "__main__":
    unittest.main()
