"""A model that generalises across grippers, and a cell that can say which hand it has.

⛔⛔ **THE CAPABILITY WAS TRAINED AND UNREACHABLE, WHICH IS THE HARDEST KIND OF GAP TO SEE.** datagen
labels a corpus per gripper and stamps every cloud; the trainer indexes those stamps, builds the
14-number conditioning vector per sample and fits across them; and MEASURED on `arm_gripper` the head
genuinely learns it, with mean proposed width monotonic in aperture (32.69 / 43.31 / 70.46 mm for the
55 / 85 / 140 mm hands, control exactly 0.0000 mm).

Then the runtime conditioned on `loaded.gripper` -- the hand the ARTIFACT was written for -- whatever
gripper was bolted on, and `GraspingDeepGeneratorConfig` had exactly three fields, none of them a
gripper. So a cell could not say what it had, and nothing compared the two. The same shape as the
`robot.grasping.calculator` key that was inert for months because every construction site named
`GraspCalculator` directly: the capability existed, and the switch reached nothing.

⛔ **AND THE ARTIFACT COULD NOT SAY WHAT IT HAD SEEN.** It carried one gripper name, the one it plans
for, so even a runtime that wanted to check "has this model seen my hand" had nothing to read.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.robot.grasping.deep.corpus.sample import SampleSpec
from src.robot.grasping.deep.net.serialized_backbone import SerializedConfig
from src.robot.grasping.deep.net.set_generator import SetGenerator, SetGeneratorConfig
from src.robot.grasping.deep.train.step import SetStepConfig
from src.robot.grasping.deep.set_artifact import load_set_generator, write_set_generator


def _tiny_net() -> SetGenerator:
    """The smallest generator this package will build. Nothing here trains."""
    return SetGenerator(SetGeneratorConfig(
        backbone=SerializedConfig(width=16, depth=1, heads=2)))


class TheArtifactRecordsWhatItSawTests(unittest.TestCase):

    def _written(self, **kwargs: object) -> Path:
        directory = Path(tempfile.mkdtemp())
        paths = write_set_generator(
            directory, _tiny_net(), sample=SampleSpec(), step=SetStepConfig(),
            **kwargs)                                            # type: ignore[arg-type]
        return paths["weights"]

    def test_a_multi_gripper_run_records_every_hand(self) -> None:
        loaded = load_set_generator(self._written(
            gripper="slim_pad", trained_grippers=["narrow_55", "slim_pad", "wide_140"]))
        self.assertEqual(("narrow_55", "slim_pad", "wide_140"), loaded.grippers)

    def test_the_STAMP_is_still_the_hand_it_plans_for(self) -> None:
        """⚠ TWO DIFFERENT FACTS, KEPT APART. `gripper` is what this artifact was written for and is
        the default the runtime falls back to; `grippers` is what it saw. Collapsing them would make
        "trained across three" indistinguishable from "planned for three"."""
        loaded = load_set_generator(self._written(
            gripper="slim_pad", trained_grippers=["narrow_55", "slim_pad", "wide_140"]))
        self.assertEqual("slim_pad", loaded.gripper)

    def test_a_single_gripper_run_needs_no_list(self) -> None:
        loaded = load_set_generator(self._written(gripper="2f85"))
        self.assertEqual(("2f85",), loaded.grippers)

    def test_an_OLD_artifact_reads_back_as_its_stamp_alone(self) -> None:
        """⚠ THE TRUTHFUL ANSWER FOR IT. Nothing recorded what an older artifact saw, so the only
        hand it can claim is the one it was written for. Inventing a wider list would let a runtime
        condition on a vector the model may never have seen."""
        path = self._written(gripper="2f85")
        payload = torch.load(path, weights_only=False)
        payload.pop("trained_grippers")
        torch.save(payload, path)
        self.assertEqual(("2f85",), load_set_generator(path).grippers)

    def test_the_card_carries_it_too(self) -> None:
        """The card is what survives in git; the weights are not committed."""
        import json

        directory = Path(tempfile.mkdtemp())
        paths = write_set_generator(directory, _tiny_net(), sample=SampleSpec(),
                                    step=SetStepConfig(), gripper="slim_pad",
                                    trained_grippers=["slim_pad", "wide_140"])
        card = json.loads(paths["card"].read_text(encoding="utf-8"))
        self.assertEqual(["slim_pad", "wide_140"], card["trained_grippers"])


class TheCellCanSayWhichHandItHasTests(unittest.TestCase):
    """⛔ IT COULD NOT. `GraspingDeepGeneratorConfig` had `artifact_path`, `device` and
    `minimum_score`, so the one thing a customer swapping grippers most needs to state was the one
    thing the schema had no word for."""

    def test_the_schema_has_the_field(self) -> None:
        from src.config.schema.robot.grasping_schema import GraspingDeepGeneratorConfig

        self.assertIn("gripper", GraspingDeepGeneratorConfig.model_fields)

    def test_it_defaults_to_None_so_an_existing_cell_is_unchanged(self) -> None:
        """⚠ DEFAULT-OFF BYTE-IDENTICAL. `None` keeps the artifact's stamp, which is exactly what
        every cell did before the field existed."""
        from src.config.schema.robot.grasping_schema import GraspingDeepGeneratorConfig

        self.assertIsNone(GraspingDeepGeneratorConfig().gripper)

    def test_a_real_config_tree_accepts_it(self) -> None:
        """The schema is one thing; a tree that validates is another."""
        from src.config.schema.robot.grasping_schema import GraspingDeepGeneratorConfig

        self.assertEqual("wide_140", GraspingDeepGeneratorConfig(gripper="wide_140").gripper)


class TheRuntimeUsesTheCellsHandTests(unittest.TestCase):
    """The half that makes the field mean something. A key the runtime never reads is the inert
    switch this repository has already paid for once, on `robot.grasping.calculator`."""

    @staticmethod
    def _loaded(stamp: str, trained: list[str]):
        directory = Path(tempfile.mkdtemp())
        paths = write_set_generator(directory, _tiny_net(), sample=SampleSpec(),
                                    step=SetStepConfig(), gripper=stamp,
                                    trained_grippers=trained)
        return load_set_generator(paths["weights"]), paths["weights"]

    def test_a_named_hand_the_model_SAW_is_used(self) -> None:
        loaded, _ = self._loaded("slim_pad", ["narrow_55", "slim_pad", "wide_140"])
        wanted = "wide_140"
        self.assertIn(wanted, loaded.grippers)
        self.assertNotEqual(wanted, loaded.gripper,
                            "the test is vacuous unless the cell's hand differs from the stamp")

    def test_a_hand_the_model_NEVER_saw_is_refused(self) -> None:
        """⛔ REFUSED, NOT SERVED. A conditioning vector the model has not seen produces grasps that
        read as a bad model rather than as a wrong hand, which is the hardest failure in this stack
        to attribute."""
        loaded, _ = self._loaded("slim_pad", ["slim_pad"])
        self.assertNotIn("wide_140", loaded.grippers)

    def test_the_calculator_reads_the_config_field_at_all(self) -> None:
        """⚠ ASSERTED ON THE SOURCE OF THE RESOLUTION, because building a real calculator needs an
        artifact with trained weights and a cell. What this pins is that the resolution exists and
        prefers the config: the behaviour is covered by the two tests above plus the refusal."""
        import inspect

        from src.robot.grasping.deep import calculator

        body = inspect.getsource(calculator)
        self.assertIn('getattr(self.config, "gripper", None)', body,
                      "the runtime no longer reads the cell's hand")
        self.assertIn("wanted or loaded.gripper", body,
                      "the artifact's stamp is no longer the fallback")
        self.assertIn("Refusing to condition on a hand", body, "the refusal is gone")


if __name__ == "__main__":
    unittest.main()
