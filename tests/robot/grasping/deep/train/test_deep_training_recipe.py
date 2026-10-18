"""A named, versioned recipe carries the proven settings, and the artifact says which one it was.

⛔ THE PROBLEM IT SOLVES. Every individual default in `train-set` is what the arms already measured
ran with, so moving them would make every earlier run irreproducible without explicit flags. But a
customer typing nothing then gets the research defaults rather than what this project has learned.
A recipe is the third option: a NAMED BUNDLE, asked for by one flag, frozen once it ships, and
written into the artifact so a model can say what produced it.

⚠ A RECIPE CONTAINS NO UNSETTLED ARM. `slot_mixing`, `axis_mode`, `--crop-mm` and `--generative` are
all being measured. Pinning one inside a version number would hand a customer a coin flip wearing a
guarantee.

⚠ AND IT NEVER OVERRIDES A FLAG THE OPERATOR TYPED. Comparing against argparse's defaults cannot
tell "not given" from "given, and equal to the default", so the check reads `sys.argv`.
"""

from __future__ import annotations

import unittest
from dataclasses import fields
from pathlib import Path

from src.robot.grasping.deep.train.plan import PlanOverrides, build_plan

from src.robot.grasping.deep.train.recipes import (
    RECIPES,
    TIERS,
    describe,
    recipe,
    settings,
    tier,
)
from src.robot.grasping.deep.train.trainer import SetTrainingPlan


class WhatIsInItTests(unittest.TestCase):

    def test_v1_turns_on_the_two_things_that_are_justified(self) -> None:
        """`refit` so the shipped model sees every part the customer owns (without it, MEASURED,
        14.8 % of their catalogue never reaches the weights), and the collapse floor so a head that
        answers every seed identically is caught at epoch three instead of hours downstream."""
        self.assertEqual(recipe("v1"), {"refit": True, "collapse_floor_deg": 1.0})

    def test_no_recipe_pins_an_UNSETTLED_arm(self) -> None:
        """⭐ THE RULE THAT KEEPS A VERSION NUMBER MEANINGFUL. Every one of these is an open
        comparison; a recipe that chose one would be presenting a guess as a decision."""
        open_arms = {"slot_mixing", "axis_mode", "crop_mm", "generative", "target"}
        for name, body in RECIPES.items():
            with self.subTest(name):
                self.assertEqual(open_arms & set(body), set(),
                                 f"recipe {name} pins an arm that is still being measured")

    def test_every_recipe_key_is_a_real_plan_field(self) -> None:
        """A recipe key with no field behind it is applied to nothing and fails silently."""
        known = {f.name for f in fields(SetTrainingPlan)}
        for name, body in RECIPES.items():
            for key in body:
                with self.subTest(recipe=name, key=key):
                    self.assertIn(key, known)


class RefusalTests(unittest.TestCase):

    def test_an_unknown_recipe_is_REFUSED_not_ignored(self) -> None:
        """A customer who typed `v2` before it exists must not get `v1`'s behaviour under `v2`'s
        name, which is the exact failure a version string exists to prevent."""
        with self.assertRaises(ValueError) as caught:
            recipe("v2")
        self.assertIn("v1", str(caught.exception), "the refusal does not say what does exist")

    def test_an_unknown_tier_is_refused_too(self) -> None:
        with self.assertRaises(ValueError):
            tier("overnight")


class TheTiersTests(unittest.TestCase):

    def test_smoke_turns_the_refit_OFF(self) -> None:
        """⚠ THE TIER WINS OVER THE RECIPE, and it has to: a run that only proves the chain closes
        must not pay for a second training pass, which would make the fast tier slower than the thing
        it exists to precede."""
        self.assertTrue(recipe("v1")["refit"])
        self.assertFalse(settings("v1", "smoke")["refit"])

    def test_full_leaves_the_refit_ON(self) -> None:
        self.assertTrue(settings("v1", "full")["refit"])

    def test_smoke_is_the_same_run_stopped_early(self) -> None:
        """Not a different configuration. Otherwise the fast tier would prove the chain closes for a
        model nobody is going to train."""
        self.assertEqual(set(TIERS["smoke"]) - {"why"}, {"epochs", "train_units", "refit"})

    def test_every_tier_says_what_it_does_NOT_prove(self) -> None:
        """⚠ The failure this prevents is a customer taking a twenty-minute smoke run for the
        finished product."""
        self.assertIn("NOTHING", TIERS["smoke"]["why"])


class TheDescriptionTests(unittest.TestCase):

    def test_it_shows_the_MERGED_result_not_both_layers(self) -> None:
        """⛔ Printing the recipe's `refit=True` beside the tier's `refit=False` tells a reader two
        things and lets them pick the wrong one."""
        line = describe("v1", "smoke")
        self.assertIn("refit=False", line)
        self.assertNotIn("refit=True", line)

    def test_it_stays_ASCII(self) -> None:
        """⚠ NOT COSMETIC. This is printed and the terminal is cp1252, so a decorated marker turns a
        training run's first line into a UnicodeEncodeError."""
        for name in RECIPES:
            for tier_name in [None, *TIERS]:
                with self.subTest(recipe=name, tier=tier_name):
                    line = describe(name, tier_name)
                    self.assertTrue(line.isascii(), line)


class TheArtifactRemembersTests(unittest.TestCase):

    def test_the_plan_carries_the_recipe_name(self) -> None:
        """So two `.pt` files are two things a person can tell apart."""
        self.assertIn("recipe", {f.name for f in fields(SetTrainingPlan)})
        self.assertIsNone(SetTrainingPlan().recipe, "a hand-assembled run must not claim a recipe")

    def test_the_recipe_field_changes_NOTHING_about_the_run(self) -> None:
        """⚠ RECORD ONLY. The settings are applied to the args before the plan is built, so the plan
        field is provenance. If it also drove behaviour there would be two paths to the same effect."""
        plain = SetTrainingPlan()
        stamped = SetTrainingPlan(recipe="v1")
        for field in fields(SetTrainingPlan):
            if field.name == "recipe":
                continue
            with self.subTest(field.name):
                self.assertEqual(getattr(plain, field.name), getattr(stamped, field.name))


class TheBannerTests(unittest.TestCase):
    """⛔ The first version of the banner printed the recipe's settings verbatim, so a run given an
    explicit `--epochs 1` announced `epochs=2` and then ran one. A banner nobody can trust is worse
    than no banner, because it is the line a person quotes afterwards.

    ⛔⛔ THESE TWO WERE REWRITTEN ON 2026-09-04, and the reason is worth keeping. They asserted on
    the SOURCE of `_cmd_train_set`: that it contained `overridden.append`, and that the recipe block
    contained `sys.argv`. Both were true of a mechanism that has since been removed, and neither
    would have noticed if the behaviour had broken while the strings stayed. The property they were
    reaching for survives and is now tested directly, on the merge itself.
    """

    def test_a_value_EQUAL_to_the_default_still_counts_as_chosen(self) -> None:
        """⭑ THE PROPERTY THE ARGV SCAN EXISTED FOR, now guaranteed by construction.

        Comparing against argparse's defaults cannot tell "not given" from "given, and equal to the
        default", so a customer who typed the default value would have had it silently replaced by
        the recipe. The old fix was to scan `sys.argv` for the flag, which works from a shell and is
        meaningless from code: under pytest or a service process `sys.argv` holds the runner's
        arguments. `PlanOverrides` carries `UNSET` for "not chosen", so the distinction is DATA and
        holds in both worlds.

        12 is `SetTrainingPlan`'s own default for `epochs`, and `--tier smoke` sets 2.
        """
        plan, notes = build_plan(tier="smoke", overrides=PlanOverrides(epochs=12))
        self.assertEqual(12, plan.epochs, "a value equal to the default was treated as absent")
        self.assertIn("epochs=12", notes["overridden"])

    def test_the_same_setting_left_alone_DOES_take_the_tier(self) -> None:
        """The control. Without it the test above passes for a merge that never applies anything."""
        plan, notes = build_plan(tier="smoke")
        self.assertEqual(2, plan.epochs)
        self.assertEqual(2, notes["epochs"])
        self.assertEqual([], notes["overridden"])

    def test_the_command_prints_what_your_flags_overrode(self) -> None:
        """The banner is the operator-facing half, so read it off the command rather than the merge.

        ⚠ BY AST, on the call, not by searching the file for a string: the earlier version matched
        `overridden.append`, a call that moved into `build_plan` and left the assertion looking for
        a mechanism rather than a message.
        """
        import ast as _ast

        source = Path(
            "src/robot/grasping/deep/__main__.py").read_text(encoding="utf-8")
        command = next(node for node in _ast.walk(_ast.parse(source))
                       if isinstance(node, _ast.FunctionDef) and node.name == "_cmd_train_set")
        text = _ast.unparse(command)
        self.assertIn("your flags win:", text)
        self.assertIn("overridden", text)

    def test_the_command_no_longer_scans_argv(self) -> None:
        """⚠ BY AST. A text search would match the comment that explains the removal, which is the
        trap this branch hit four times in one day."""
        import ast as _ast

        source = Path(
            "src/robot/grasping/deep/__main__.py").read_text(encoding="utf-8")
        command = next(node for node in _ast.walk(_ast.parse(source))
                       if isinstance(node, _ast.FunctionDef) and node.name == "_cmd_train_set")
        self.assertEqual([], [n for n in _ast.walk(command)
                              if isinstance(n, _ast.Attribute) and n.attr == "argv"])


if __name__ == "__main__":
    unittest.main()
