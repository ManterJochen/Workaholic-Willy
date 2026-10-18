"""Starting a run from somebody else's weights, which is what a customer with their own cell needs.

⭐⭐ FINE-TUNING IS NOT RESUMING, and conflating them is how this capability was missing for months
while looking present. `train --resume` continues one interrupted experiment and REFUSES on any
difference in the plan, the architecture, the corpus by content, the device or the torch version. That
strictness is right for what it does, and it makes the flag useless for the case a customer actually
has: take a model trained on our corpus and continue it on their own objects, which is a changed
corpus by definition.

So `--init-from` starts a NEW run. Fresh optimiser, fresh schedule, fresh split; only the
initialisation moves, and the report says where it came from.

⛔ The trap these tests are mostly about is `load_state_dict(..., strict=False)`. It is the obvious
way to load a checkpoint from a slightly different model and it silently leaves every mismatched
tensor at its random initialisation, producing a model half inherited and half fresh that nobody
chose, which trains and reports plausible numbers.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.robot.grasping.deep.net.slot_head import SlotHeadConfig
from src.robot.grasping.deep.net.set_generator import SetGenerator, SetGeneratorConfig
from src.robot.grasping.deep.train.trainer import _load_weights


def _small(slots: int = 2) -> SetGeneratorConfig:
    """A generator small enough for a unit test, with the head shape under test."""
    return SetGeneratorConfig(head=SlotHeadConfig(slots=slots, width=32, gripper_width=8,
                                                  query_width=8),
                              graspability_width=8)


def _checkpoint(folder: Path, config: SetGeneratorConfig, *, epoch: int = 5) -> Path:
    torch.manual_seed(0)
    net = SetGenerator(config)
    path = folder / "fold0.pt"
    torch.save({"model": net.state_dict(), "plan": "SetTrainingPlan(epochs=6)", "fold": 0,
                "epoch": epoch}, path)
    return path


class LoadTests(unittest.TestCase):

    def test_weights_are_copied_and_the_source_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            config = _small()
            path = _checkpoint(Path(folder), config)
            torch.manual_seed(99)
            fresh = SetGenerator(config)
            before = {k: v.clone() for k, v in fresh.state_dict().items()}
            inherited = _load_weights(fresh, path, "cpu")
        self.assertEqual(inherited["tensors"], len(before))
        self.assertEqual(inherited["trained_epochs"], 6)
        self.assertIn("fold0.pt", inherited["path"])
        moved = sum(1 for k, v in fresh.state_dict().items()
                    if not torch.equal(v, before[k]))
        self.assertGreater(moved, 0, "nothing was actually loaded")

    def test_a_state_dict_key_is_accepted_too(self) -> None:
        """The artifact writer uses `state_dict`; the fold checkpoint uses `model`. Both are ours."""
        with tempfile.TemporaryDirectory() as folder:
            config = _small()
            torch.manual_seed(0)
            net = SetGenerator(config)
            path = Path(folder) / "artifact.pt"
            torch.save({"state_dict": net.state_dict()}, path)
            _load_weights(SetGenerator(config), path, "cpu")


class RefusalTests(unittest.TestCase):
    """⛔ A shape that does not match is a question for a person, not a tensor to leave random."""

    def test_a_checkpoint_from_a_different_head_is_REFUSED_not_partly_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = _checkpoint(Path(folder), _small(slots=2))
            with self.assertRaises(ValueError) as caught:
                _load_weights(SetGenerator(_small(slots=5)), path, "cpu")
        message = str(caught.exception)
        self.assertIn("does not fit this architecture", message)
        self.assertIn("different shape", message)

    def test_a_checkpoint_from_a_different_width_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = _checkpoint(Path(folder), _small())
            wider = SetGeneratorConfig(head=SlotHeadConfig(slots=2, width=64, gripper_width=8,
                                                           query_width=8), graspability_width=8)
            with self.assertRaises(ValueError):
                _load_weights(SetGenerator(wider), path, "cpu")

    def test_a_file_that_is_not_a_checkpoint_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "other.pt"
            torch.save({"something": 1}, path)
            with self.assertRaises(ValueError) as caught:
                _load_weights(SetGenerator(_small()), path, "cpu")
            self.assertIn("neither `model` nor `state_dict`", str(caught.exception))

    def test_a_missing_file_refuses_by_path(self) -> None:
        with self.assertRaises(FileNotFoundError):
            _load_weights(SetGenerator(_small()), Path("nowhere") / "absent.pt", "cpu")


class FreezeTests(unittest.TestCase):

    def test_freezing_the_backbone_leaves_the_heads_trainable(self) -> None:
        net = SetGenerator(_small())
        for parameter in net.backbone.parameters():
            parameter.requires_grad_(False)
        backbone = {id(p) for p in net.backbone.parameters()}
        trainable = [p for p in net.parameters() if p.requires_grad]
        self.assertTrue(trainable, "freezing the backbone left nothing to train")
        self.assertFalse(any(id(p) in backbone for p in trainable))

    def test_a_frozen_tensor_does_NOT_drift_under_weight_decay(self) -> None:
        """⛔ WRITTEN TO CONFIRM A GUARD AND IT REFUTED THE GUARD'S REASON. I claimed that handing a
        frozen tensor to AdamW lets decoupled weight decay shrink it toward zero while every log line
        says it is frozen, and built the optimiser from the trainable parameters to prevent that.

        MEASURED: it does not happen. The step skips any parameter whose gradient is None, so a
        frozen tensor is untouched either way. Kept as a test because the belief is plausible enough
        to come back, and because the filter it justified is still worth having for a smaller reason:
        it makes `trainable_parameters` mean what it says.
        """
        frozen = torch.nn.Parameter(torch.ones(4), requires_grad=False)
        optimiser = torch.optim.AdamW([frozen], lr=0.1, weight_decay=0.5)
        for _ in range(5):
            optimiser.step()
        self.assertTrue(torch.allclose(frozen.data, torch.ones(4)),
                        "a frozen parameter moved; the earlier worry was right after all")


class SurfaceTests(unittest.TestCase):

    def test_the_flag_is_reachable_and_says_it_is_not_a_resume(self) -> None:
        text = Path("src/robot/grasping/deep/__main__.py").read_text(encoding="utf-8")
        self.assertIn('"--init-from"', text)
        self.assertIn('"--freeze-backbone"', text)
        self.assertIn("IS NOT A RESUME", text)
        self.assertIn("init_from=args.init_from", text)

    def test_the_report_stamps_the_source_even_when_there_is_none(self) -> None:
        """⛔ A card that mentions the source only when there is one leaves a reader to infer "from
        scratch" from an absence, and an absence is also what a writer that forgot produces."""
        text = Path("src/robot/grasping/deep/train/trainer.py").read_text(encoding="utf-8")
        self.assertIn('"initialised_from": str(init_from) if init_from is not None else None', text)
        self.assertIn('"frozen_backbone": bool(freeze_backbone)', text)


if __name__ == "__main__":
    unittest.main()
