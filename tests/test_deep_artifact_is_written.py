"""A trained model has to be loadable by something, and for twenty-five arms it was not.

⛔⛔ THE DEFECT. `set_artifact` has been fully written and tested since the architecture restart: a
self-describing format with four named refusals, so a binned model cannot be loaded into a set
decoder and a state dict cannot be loaded into a config it did not travel with. It had NO CALL SITE.

The training loop wrote a bare `torch.save({"model": state_dict, "plan": repr(plan), ...})`, and
`load_set_generator` REFUSES that by kind. So nothing a run produced could be deployed, and the plan's
PRIMARY metric, the referee success rate, had no model to propose grasps with. A whole arc was
measuring a model that could not leave the folder it was trained in.

Found by an audit of the plan against the code on 2026-09-03. These tests hold the round trip open.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.robot.grasping.deep.corpus.sample import SampleSpec
from src.robot.grasping.deep.net.slot_head import SlotHeadConfig
from src.robot.grasping.deep.net.set_generator import SetGenerator, SetGeneratorConfig
from src.robot.grasping.deep.set_artifact import load_set_generator
from src.robot.grasping.deep.train.trainer import SetTrainingPlan, _write_artifact


def _plan() -> SetTrainingPlan:
    return SetTrainingPlan(sample=SampleSpec(grasp_set=True, points=256),
                           model=SetGeneratorConfig(
                               head=SlotHeadConfig(slots=2, width=32, gripper_width=8,
                                                    query_width=8),
                               graspability_width=8))


class RoundTripTests(unittest.TestCase):

    def test_a_written_artifact_loads_back(self) -> None:
        """⭐ THE WHOLE POINT. The fold checkpoint cannot do this, which is why it was not enough."""
        plan = _plan()
        torch.manual_seed(0)
        net = SetGenerator(plan.model)
        with tempfile.TemporaryDirectory() as folder:
            block = _write_artifact(Path(folder), net, plan, ["2f85"], None)
            self.assertTrue(block["written"], block)
            self.assertEqual(block["gripper"], "2f85")
            loaded = load_set_generator(block["weights"])
        self.assertEqual(loaded.gripper, "2f85")
        for name, tensor in net.state_dict().items():
            with self.subTest(name):
                self.assertTrue(torch.equal(loaded.net.state_dict()[name].cpu(), tensor.cpu()))

    def test_the_card_is_written_beside_the_weights(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as folder:
            block = _write_artifact(Path(folder), SetGenerator(plan.model), plan, ["2f85"], None)
            self.assertTrue(Path(block["card"]).is_file())
            self.assertTrue(Path(block["weights"]).is_file())


class GripperTests(unittest.TestCase):
    """⛔ The artifact stamps which hand the model plans for, and the loader resolves it to the
    conditioning vector. Guessing it would put the 2F-85's geometry on a model trained for a 140 mm
    hand without a word, which is exactly what importing a foreign corpus now produces."""

    def test_a_single_gripper_corpus_needs_no_choice(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as folder:
            block = _write_artifact(Path(folder), SetGenerator(plan.model), plan,
                                    ["wide_140", "wide_140"], None)
        self.assertEqual(block["gripper"], "wide_140")

    def test_a_VARIED_corpus_refuses_rather_than_defaulting(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as folder:
            block = _write_artifact(Path(folder), SetGenerator(plan.model), plan,
                                    ["2f85", "wide_140"], None)
        self.assertFalse(block["written"])
        self.assertIn("2 grippers", block["reason"])
        self.assertIn("artifact_gripper", block["reason"])

    def test_a_named_gripper_wins_over_the_corpus(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as folder:
            block = _write_artifact(Path(folder), SetGenerator(plan.model), plan,
                                    ["2f85", "wide_140"], "narrow_55")
        self.assertTrue(block["written"])
        self.assertEqual(block["gripper"], "narrow_55")

    def test_an_unknown_gripper_is_reported_and_not_raised(self) -> None:
        """A run that has just spent hours training should not lose its weights to a typo in a flag,
        so the failure is recorded in the report and the fold checkpoint still exists."""
        plan = _plan()
        with tempfile.TemporaryDirectory() as folder:
            block = _write_artifact(Path(folder), SetGenerator(plan.model), plan, ["2f85"],
                                    "no_such_hand")
        self.assertFalse(block["written"])
        self.assertIn("unknown gripper", block["reason"])


class ReportTests(unittest.TestCase):

    def test_the_absence_of_an_artifact_is_recorded_rather_than_left_blank(self) -> None:
        """"No artifact was written" is a fact a card should carry, not an absence a reader has to
        notice. An absence is also what a writer that forgot produces."""
        plan = _plan()
        with tempfile.TemporaryDirectory() as folder:
            block = _write_artifact(Path(folder), SetGenerator(plan.model), plan,
                                    ["2f85", "narrow_55"], None)
        self.assertIn("written", block)
        self.assertIn("reason", block)

    def test_the_loop_calls_it_and_the_flag_is_reachable(self) -> None:
        loop = Path("src/robot/grasping/deep/train/trainer.py").read_text(encoding="utf-8")
        self.assertIn("_write_artifact(", loop)
        self.assertIn('report.setdefault("artifact"', loop)
        cli = Path("src/robot/grasping/deep/__main__.py").read_text(encoding="utf-8")
        self.assertIn('"--artifact-gripper"', cli)
        self.assertIn("artifact_gripper=args.artifact_gripper", cli)


if __name__ == "__main__":
    unittest.main()
