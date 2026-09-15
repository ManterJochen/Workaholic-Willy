"""An artifact plans only for a hand it trained across, names hands as the resolver does, and keeps their numbers.

Lane (i) D5. The trainer stamped whatever `artifact_gripper` it was handed, so an artifact could claim a hand no sample
conditioned on. The writer and the loader knew only `JAW_GEOMETRY`'s names, so no artifact could be written for a
registry hand. And nothing recorded what a hand measured when the model was trained, so re-measuring it in the
registry would silently condition the net on a hand it never saw. Now the run refuses the unseen stamp, the writer and
the loader resolve through `deep.hands` in the cell's tree, and the artifact stores each trained hand's numbers, which
the loader checks.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from src.robot.grasping.deep.corpus.sample import SampleSpec
from src.robot.grasping.deep.hands import resolve_hand
from src.robot.grasping.deep.set_artifact import load_set_generator, write_set_generator
from src.robot.grasping.deep.train.step import SetStepConfig
from tests.test_deep_artifact_is_written import _plan
from tests.test_deep_set_artifact import _net

_DATA = Path(__file__).resolve().parents[1] / "config"
_HANDE = "robotiq_hande"
_TRAINED = ("robotiq_2f85", _HANDE)


def _write(directory: Path, gripper: str = _HANDE, trained: tuple[str, ...] = _TRAINED) -> Path:
    return write_set_generator(directory, _net(), sample=SampleSpec(grasp_set=True, points=512),
                               step=SetStepConfig(seeds=16), gripper=gripper, trained_grippers=list(trained))["weights"]


class ARegistryHandIsAnArtifactHandTests(unittest.TestCase):
    def test_an_artifact_for_a_registry_hand_writes_and_loads(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            loaded = load_set_generator(_write(Path(name)))
        self.assertEqual(loaded.gripper, _HANDE)
        self.assertEqual(loaded.grippers, _TRAINED)

    def test_the_artifact_keeps_each_trained_hands_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            payload = torch.load(_write(Path(name)), weights_only=True)
        for hand in _TRAINED:
            with self.subTest(hand=hand):
                self.assertEqual(payload["hand_numbers"][hand], dict(resolve_hand(hand).numbers))

    def test_a_hand_re_measured_since_the_run_refuses_at_load(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = _write(Path(name))
            payload = torch.load(path, weights_only=True)
            payload["hand_numbers"][_HANDE]["aperture_mm"] = 50.5
            torch.save(payload, path)
            with self.assertRaises(ValueError) as caught:
                load_set_generator(path)
        self.assertIn(_HANDE, str(caught.exception))
        self.assertIn("aperture_mm", str(caught.exception))

    def test_an_artifact_written_before_the_numbers_still_loads(self) -> None:
        """The control: nothing recorded them, so nothing is checked."""
        with tempfile.TemporaryDirectory() as name:
            path = write_set_generator(Path(name), _net(), sample=SampleSpec(grasp_set=True, points=512),
                                       step=SetStepConfig(seeds=16), gripper="2f85")["weights"]
            payload = torch.load(path, weights_only=True)
            payload.pop("hand_numbers", None)
            torch.save(payload, path)
            self.assertEqual(load_set_generator(path).gripper, "2f85")


class TheCellsTreeAnswersTests(unittest.TestCase):
    def _tree_without_the_hande(self, directory: Path) -> Path:
        root = directory / "data"
        shutil.copytree(_DATA / "grippers", root / "grippers")
        (root / "grippers" / f"{_HANDE}.yaml").unlink()
        return root

    def test_the_loader_resolves_in_the_cells_tree(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = _write(Path(name))
            with self.assertRaises(ValueError) as caught:
                load_set_generator(path, data_dir=self._tree_without_the_hande(Path(name)))  # type: ignore[call-arg]
        self.assertIn("cannot", str(caught.exception))

    def test_the_calculator_loads_in_its_tree(self) -> None:
        from src.robot.grasping.deep.calculator import DeepCalculatorConfig, DeepGraspCalculator

        with tempfile.TemporaryDirectory() as name:
            path = _write(Path(name))
            tree = self._tree_without_the_hande(Path(name))
            calculator = DeepGraspCalculator(DeepCalculatorConfig(artifact_path=str(path), device="cpu",
                                                                  data_dir=str(tree)))
            with self.assertRaises(ValueError):
                calculator.preload()


class TheRunPlansOnlyForAHandItSawTests(unittest.TestCase):
    def test_a_stamp_the_run_never_saw_is_refused(self) -> None:
        from src.robot.grasping.deep.net.set_generator import SetGenerator
        from src.robot.grasping.deep.train.trainer import _write_artifact

        plan = _plan()
        with tempfile.TemporaryDirectory() as folder:
            block = _write_artifact(Path(folder), SetGenerator(plan.model), plan, ["2f85", "wide_140"], "narrow_55")
            written = any(Path(folder).glob("*.pt"))
        self.assertFalse(block["written"])
        self.assertIn("narrow_55", block["reason"])
        self.assertFalse(written)

    def test_the_model_name_of_a_hand_it_saw_is_accepted(self) -> None:
        """A pre-registry corpus is stamped `2f85`; naming the hand by its model name is naming the same hand."""
        from src.robot.grasping.deep.net.set_generator import SetGenerator
        from src.robot.grasping.deep.train.trainer import _write_artifact

        plan = _plan()
        with tempfile.TemporaryDirectory() as folder:
            block = _write_artifact(Path(folder), SetGenerator(plan.model), plan, ["2f85"], "robotiq_2f85")
        self.assertTrue(block["written"], block)
        self.assertEqual(block["gripper"], "robotiq_2f85")


if __name__ == "__main__":
    unittest.main()
