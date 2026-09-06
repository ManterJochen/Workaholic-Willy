"""The corpus half of a named recipe, and the two things the runbook produced and then discarded.

⛔ **STEP 1 SCREENED WHICH OF THE CUSTOMER'S PARTS A JAW CAN GRASP, AND NOTHING READ THE RESULT.**
`prepare-assets` prints the wiring instruction in its own output ("point your config's
`assets.jaw_screen_path` at those"), and the runbook went straight from Step 1 to `build --config`.
So the bank was filtered by a box proxy on the mesh HULL, while a jaw closes on a LINE through the
object. MEASURED 2026-09-01 over the whole library: the proxy called 514 meshes graspable where the
labeller finds 331, and REFUSED 50 that do have grasps; one pot carrying 1,044 labels missed the span
limit by ONE millimetre.

⛔ **AND OBJECTS WERE STOOD UP RATHER THAN DUMPED.** `families.flat_orientation` defaults to
`upright`, which is a stable equilibrium a solver leaves alone: of 405 MuJoCo-settled objects, 90.1 %
end within 30 degrees of upright. A tipped object is 3.0x as likely to admit a jaw grasp, 17.6 %
against 5.9 %, worth six times the jaw labels per scene and a net +57 % graspable objects per scene.
It compounds with the engine, because MuJoCo is both the engine that leaves things upright and the one
a customer without an NVIDIA box uses.

⚠ THE DEFAULTS STAY. Both are what every corpus on disk was built with, and moving either would
silently change what a rebuild produces. The recipe is the named bundle that says which of them a NEW
corpus should not keep.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from datagen.config import DatagenConfig
from datagen.recipes import CORPUS_RECIPES, PLACEHOLDERS, config_for, corpus_recipe, steps


class WhatIsInItTests(unittest.TestCase):

    def test_v1_dumps_objects_rather_than_standing_them_up(self) -> None:
        """⭐ The largest single lever in the corpus, and the default is the other value."""
        self.assertEqual(corpus_recipe("v1")["families"]["flat_orientation"], "random")

    def test_the_default_is_still_upright_so_old_corpora_rebuild(self) -> None:
        """The control. If the default had moved, the recipe would be redundant and every corpus on
        disk would rebuild into something else."""
        self.assertEqual(DatagenConfig().families.flat_orientation, "upright")

    def test_v1_wires_the_screen_step_1_produces(self) -> None:
        """⛔ The runbook produced it and never used it."""
        self.assertIn("assets.jaw_screen_path", PLACEHOLDERS["v1"])

    def test_the_screen_key_defaults_to_EMPTY_which_is_the_proxy(self) -> None:
        """Empty keeps the box proxy, so a config that does not name a screen is not obviously
        broken, which is exactly why it went unnoticed."""
        self.assertEqual(DatagenConfig().assets.jaw_screen_path, "")


class TheWrittenConfigTests(unittest.TestCase):

    def test_it_validates_as_a_real_config(self) -> None:
        """⭐ A starter config that does not load is worse than none: the customer's first command
        fails and the message is about pydantic."""
        config = DatagenConfig(**config_for("v1"))

        self.assertEqual(config.families.flat_orientation, "random")
        self.assertTrue(config.assets.jaw_screen_path)

    def test_it_is_MINIMAL_rather_than_a_dump_of_every_default(self) -> None:
        """A file repeating two hundred defaults hides the handful of lines that were chosen, and a
        customer editing it cannot tell which of them matter."""
        body = config_for("v1")

        self.assertLessEqual(len(body), 3, f"the starter config has grown a tail: {sorted(body)}")
        self.assertEqual(set(body), {"families", "assets"})

    def test_init_config_writes_and_REFUSES_to_overwrite(self) -> None:
        """⚠ That file is where the customer's edits live. A command that silently replaced them
        would destroy the one thing in the tree they authored."""
        from datagen.__main__ import main

        with TemporaryDirectory() as name:
            target = Path(name) / "cfg.json"
            self.assertEqual(main(["init-config", "--recipe", "v1", "--out", str(target)]), 0)
            body = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(body["families"]["flat_orientation"], "random")

            second = main(["init-config", "--recipe", "v1", "--out", str(target)])
            self.assertNotEqual(second, 0, "a second run overwrote an edited config")

    def test_an_unknown_recipe_is_refused(self) -> None:
        from datagen.__main__ import main

        with TemporaryDirectory() as name:
            target = Path(name) / "cfg.json"
            self.assertNotEqual(main(["init-config", "--recipe", "v9", "--out", str(target)]), 0)
            self.assertFalse(target.exists(), "a refused recipe still wrote a file")


class TheStepsTests(unittest.TestCase):

    def test_kinds_is_on_the_corpus_step_and_NOT_on_the_label_step(self) -> None:
        """⛔ `label_dataset` takes no `kinds` and always writes BOTH, so the flag decided nothing
        there while the runbook explained it with a measurement. `build-cloud-corpus` defaults to
        `jaw` and would drop every suction label the step before wrote."""
        commands = {command.split()[3]: command for command, _why in steps("v1")}

        self.assertNotIn("--kinds", commands["label-grasps"])
        self.assertIn("--kinds both", commands["build-cloud-corpus"])

    def test_the_label_step_still_asks_for_the_grid_density(self) -> None:
        """The control: stripping the wrong flag must not strip the right one. Uniform azimuth
        sampling reaches the top-down approach only by coincidence."""
        commands = {command.split()[3]: command for command, _why in steps("v1")}

        self.assertIn("--density grid", commands["label-grasps"])

    def test_every_step_says_WHY_its_flags_are_not_the_defaults(self) -> None:
        """A command list without reasons is a spell. The next person cannot tell which flag they
        may drop for their corpus."""
        for command, why in steps("v1"):
            with self.subTest(command):
                self.assertGreater(len(why), 40, "this step's reason is a stub")


class TheTwoHalvesAgreeTests(unittest.TestCase):
    """⚠ `backend` may never import `datagen`, so the recipe exists twice, once on each side. That
    is the price of the dependency rule, and this is what keeps the two from drifting apart."""

    def test_both_halves_know_the_same_recipe_names(self) -> None:
        from src.robot.grasping.deep.train.recipes import RECIPES

        self.assertEqual(set(RECIPES), set(CORPUS_RECIPES),
                         "a recipe exists on one side only, so `--recipe v1` means different things "
                         "to the corpus builder and to the trainer")

    def test_the_corpus_half_points_at_the_training_half(self) -> None:
        """A customer who ran `init-config` must be told the command that consumes its output."""
        from datagen.__main__ import main

        with TemporaryDirectory() as name:
            target = Path(name) / "cfg.json"
            main(["init-config", "--recipe", "v1", "--out", str(target)])

        source = Path("datagen/__main__.py").read_text(encoding="utf-8")
        self.assertIn("train-set --recipe", source)


if __name__ == "__main__":
    unittest.main()
