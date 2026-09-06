"""The K-slot generator behind the same calculator seam as the binned one.

⛔⛔ **THE SEAM EXISTED AS A REFUSAL AND NOTHING ELSE.** `train-set` writes
`kind="set_grasp_generator"`; `DeepGraspCalculator` accepted only `"grasp_generator"`. So the entire
restarted spine produced weights no cell could load, its only consumer anywhere was `propose.py`, and
owner decision 6 (ship `deep` only if it beats `sfe_fused` on the ladder plus on-box parity) was
unrunnable no matter how good the model became: the ladder grades a CALCULATOR, and no calculator
could hold them.

⚠ BOTH FAMILIES, BY OWNER DECISION 2026-09-04. The binned generator is the only one that has ever been
through the ladder, so removing it would delete the comparison; the set generator is the one being
trained. One config key, one calculator, a branch on `kind`. These tests pin that both still work and
that neither can be mistaken for the other.

⚠ WHAT THIS DOES NOT SHOW. An untrained net proposes nothing useful. What is pinned here is the
CONTRACT: it loads, it never throws, its grasps are in the BASE frame, they say which family produced
them, and a wrong artifact is still refused by name.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.robot.grasping.deep.calculator import DeepCalculatorConfig, DeepGraspCalculator
from src.robot.grasping.deep.net.set_generator import SetGenerator, SetGeneratorConfig
from src.robot.grasping.deep.set_artifact import write_set_generator
from src.robot.grasping.types.grasp_point import GraspFrame

_K = ((900.0, 0.0, 80.0), (0.0, 900.0, 80.0), (0.0, 0.0, 1.0))


def _tiny_net() -> SetGenerator:
    """The smallest set generator that still exercises every stage of the decode."""
    torch.manual_seed(0)
    config = SetGeneratorConfig()
    backbone = type(config.backbone)(width=32, depth=2, heads=2)
    head = type(config.head)(slots=4, width=32)
    return SetGenerator(type(config)(backbone=backbone, head=head))


def _artifact(directory: Path, *, gripper: str = "2f85") -> Path:
    from src.robot.grasping.deep.train.trainer import SetTrainingPlan

    plan = SetTrainingPlan()
    paths = write_set_generator(directory, _tiny_net(), sample=plan.sample, step=plan.step,
                                gripper=gripper)
    return paths["weights"]


def _frame(size: int = 160) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.full((size, size), 700.0)
    depth[40:120, 40:120] = 660.0
    mask = np.zeros((size, size), dtype=bool)
    mask[40:120, 40:120] = True
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = [400.0, 0.0, 1000.0]
    return depth, mask, transform


def _calculator(path: Path, **over) -> DeepGraspCalculator:
    return DeepGraspCalculator(DeepCalculatorConfig(
        artifact_path=str(path), camera_matrix=_K, device="cpu", minimum_score=0.0, **over))


class LoadTests(unittest.TestCase):

    def test_the_runtime_LOADS_what_train_set_writes(self) -> None:
        """⛔ THE DEFECT ITSELF. This assertion failed by construction until 2026-09-04."""
        with tempfile.TemporaryDirectory() as name:
            calculator = _calculator(_artifact(Path(name)))
            calculator.preload()
            self.assertEqual(calculator._family, "set")           # noqa: SLF001
            self.assertEqual(calculator._set_gripper, "2f85")     # noqa: SLF001

    def test_the_gripper_comes_from_the_ARTIFACT_and_is_not_guessed(self) -> None:
        """⚠ A wrong conditioning vector produces grasps that read as a bad model rather than as a
        wrong hand, which is the hardest kind of defect to attribute."""
        with tempfile.TemporaryDirectory() as name:
            calculator = _calculator(_artifact(Path(name), gripper="wide_140"))
            calculator.preload()
            self.assertEqual(calculator._set_gripper, "wide_140")  # noqa: SLF001

    def test_the_SAMPLE_SPEC_travels_with_the_weights(self) -> None:
        """Serving a net a different sample than it was fitted to is skew that reads as a bad model.
        The binned path already did this; the set path must not be the exception."""
        with tempfile.TemporaryDirectory() as name:
            calculator = _calculator(_artifact(Path(name)))
            calculator.preload()
            self.assertIn("graspability_radius_mm", calculator._sample_spec)  # noqa: SLF001

    def test_an_unknown_kind_is_still_refused_and_NAMES_BOTH_families(self) -> None:
        """The control. If the branch accepted anything, every test above would pass on a calculator
        that had stopped checking."""
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "wrong.pt"
            torch.save({"kind": "grasp_gbt_ranker", "artifact_version": 1}, path)
            calculator = _calculator(path)
            with self.assertRaises(ValueError) as caught:
                calculator.preload()
        message = str(caught.exception)
        self.assertIn("grasp_generator", message)
        self.assertIn("set_grasp_generator", message)


class ContractTests(unittest.TestCase):

    def test_it_proposes_in_the_BASE_frame_and_says_which_family(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            calculator = _calculator(_artifact(Path(name)))
            depth, mask, transform = _frame()
            result = calculator.compute_result(mask, depth, transform)
        self.assertTrue(result.candidates, result.reasons)
        for grasp in result.candidates:
            self.assertEqual(grasp.frame, GraspFrame.BASE)
            self.assertEqual(grasp.metadata["generator"], "deep")
            self.assertEqual(grasp.metadata["family"], "set")
            # The head is not calibrated and every candidate says so wherever it travels.
            self.assertFalse(grasp.metadata["score_is_calibrated"])

    def test_the_telemetry_names_the_family(self) -> None:
        """⚠ Two architectures now sit behind one config key and report the same metrics. A rate that
        does not say which one produced it cannot be compared against anything."""
        with tempfile.TemporaryDirectory() as name:
            calculator = _calculator(_artifact(Path(name)))
            depth, mask, transform = _frame()
            calculator.compute_result(mask, depth, transform)
            self.assertEqual(calculator.last_telemetry["family"], "set")

    def test_every_candidate_carries_its_SLOT(self) -> None:
        """The K slots are the architecture. A slot that never appears in the output is the shape of
        defect this arc has already met once: every `top1` it measured was slot 0."""
        with tempfile.TemporaryDirectory() as name:
            calculator = _calculator(_artifact(Path(name)))
            depth, mask, transform = _frame()
            result = calculator.compute_result(mask, depth, transform)
        for grasp in result.candidates:
            self.assertIn("slot", grasp.metadata)
            self.assertIn(grasp.metadata["slot"], range(4))

    def test_it_NEVER_THROWS_on_a_degenerate_frame(self) -> None:
        """An optional generator that raises takes the pick down with it. The protocol forbids it."""
        with tempfile.TemporaryDirectory() as name:
            calculator = _calculator(_artifact(Path(name)))
            empty = np.zeros((160, 160), dtype=bool)
            result = calculator.compute_result(empty, np.full((160, 160), 700.0), np.eye(4))
        self.assertFalse(result.candidates)
        self.assertTrue(result.reasons)

    def test_compute_and_compute_result_agree(self) -> None:
        """The ladder calls `compute()`, the runtime calls `compute_result()`. Both are required, and
        the ladder is how owner decision 6 gets evaluated at all."""
        with tempfile.TemporaryDirectory() as name:
            calculator = _calculator(_artifact(Path(name)))
            depth, mask, transform = _frame()
            candidates = calculator.compute(mask, depth, transform)
            result = calculator.compute_result(mask, depth, transform)
        self.assertEqual(len(candidates), len(result.candidates))

    def test_a_width_the_jaw_cannot_open_to_is_counted_not_dropped(self) -> None:
        """⚠ 'The net proposed nothing' and 'the net proposed nothing THIS JAW CAN HOLD' call for
        opposite repairs, so the second must be visible in the telemetry."""
        with tempfile.TemporaryDirectory() as name:
            calculator = _calculator(_artifact(Path(name)))
            depth, mask, transform = _frame()
            calculator.compute_result(mask, depth, transform)
            self.assertIn("rejected_grip_width", calculator.last_telemetry)


class TheRetiredFamilyIsRefusedTests(unittest.TestCase):
    """⛔ THE INVERSE OF THE TEST THAT USED TO LIVE HERE.

    Until 2026-09-04 this file asserted that the BINNED generator still loaded through the same
    calculator, because branching rather than replacing was the whole point of the seam. The owner
    retired that architecture, so the assertion is now the opposite one, and it matters more than
    the old one did: somebody with a `grasp_generator_v1.pt` on disk must be TOLD what happened.
    A bare "unexpected kind" would send them looking for a loader that was deleted on purpose, and a
    silent fallback to the analytic stack would let them read the ladder as a model result.
    """

    @staticmethod
    def _binned_artifact(directory: Path) -> Path:
        """A file stamped the way the retired trainer stamped its artifacts.

        Written by hand rather than imported: the module that held these constants was deleted, and
        importing them back would resurrect exactly what this test exists to keep buried.
        """
        import torch

        path = directory / "grasp_generator_v1.pt"
        torch.save({"kind": "grasp_generator", "artifact_version": 1, "state_dict": {}}, path)
        return path

    def test_the_calculator_names_the_retirement_rather_than_the_kind(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            calculator = _calculator(self._binned_artifact(Path(name)))
            with self.assertRaises(ValueError) as caught:
                calculator.preload()
        message = str(caught.exception)
        self.assertIn("BINNED", message)
        self.assertIn("train-set", message, "the refusal has to say what to do instead")

    def test_the_factory_preflight_refuses_it_too(self) -> None:
        """Both halves of the seam, because either one alone still admits the file.

        The calculator and the factory each carry their own check, and this repository has already
        shipped a state where they disagreed: the preflight accepted only the binned kind while the
        calculator had learned to load the set one, so a correct artifact was refused by the half
        nobody was looking at.
        """
        from src.robot.grasping.calculator_factory import (  # noqa: PLC0415
            _refuse_unless_artifact,
        )

        with tempfile.TemporaryDirectory() as name:
            path = self._binned_artifact(Path(name))
            with self.assertRaises(ValueError) as caught:
                _refuse_unless_artifact(str(path))
        self.assertIn("BINNED", str(caught.exception))
        self.assertIn("retired", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
