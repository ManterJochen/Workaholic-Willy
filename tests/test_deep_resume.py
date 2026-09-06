"""Continuing a run, and refusing to continue one that is not the same run.

⭐⭐ THE DISTINCTION THIS FILE EXISTS TO KEEP. Two flags start a run from existing weights and they
mean opposite things:

    --resume      continues ONE experiment. Refuses any change to the plan, because the whole claim
                  of a resumed curve is that it is the curve it started.
    --init-from   starts a NEW experiment from somebody's weights. Deliberately ALLOWS a changed
                  corpus, which is exactly what a customer training on their own objects needs.

Conflating them is how this repository already lost a capability: `train --resume` refuses a changed
corpus by content, which is correct for a resume and made it useless for fine-tuning, and for months
there was no other flag.

⛔ And weights alone cannot resume a run. AdamW's moments are half its state, the cosine schedule
reads the epoch it is on, and the two seed streams decide which units and seeds the next epoch draws.
Restoring only the weights is a warm start, which is a thing worth having and is not this.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.robot.grasping.deep.corpus.sample import SampleSpec
from src.robot.grasping.deep.net.slot_head import SlotHeadConfig
from src.robot.grasping.deep.net.serialized_backbone import SerializedConfig
from src.robot.grasping.deep.net.set_generator import SetGenerator, SetGeneratorConfig
from src.robot.grasping.deep.train.trainer import (
    SetTrainingPlan, _plan_fingerprint, _resume_fold)


def _plan(**kwargs) -> SetTrainingPlan:
    base = dict(epochs=4, sample=SampleSpec(grasp_set=True, points=256),
                model=SetGeneratorConfig(
                    backbone=SerializedConfig(width=32, depth=1, heads=4),
                    head=SlotHeadConfig(slots=2, width=32, gripper_width=8, query_width=8),
                    graspability_width=8))
    base.update(kwargs)
    return SetTrainingPlan(**base)


def _pieces(plan: SetTrainingPlan):
    torch.manual_seed(0)
    net = SetGenerator(plan.model)
    optimiser = torch.optim.AdamW(net.parameters(), lr=plan.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=plan.epochs)
    return net, optimiser, schedule


def _write(folder: Path, plan: SetTrainingPlan, *, epoch: int = 1, **override) -> Path:
    net, optimiser, schedule = _pieces(plan)
    for _ in range(epoch + 1):
        schedule.step()
    rng = np.random.default_rng(plan.seed)
    generator = torch.Generator().manual_seed(plan.seed)
    payload = {"model": net.state_dict(), "optimiser": optimiser.state_dict(),
               "schedule": schedule.state_dict(), "plan": _plan_fingerprint(plan),
               "plan_text": repr(plan), "numpy_state": rng.bit_generator.state,
               "torch_state": generator.get_state(), "fold": 0, "epoch": epoch}
    payload.update(override)
    path = folder / "fold0.pt"
    torch.save(payload, path)
    return path


class ResumeTests(unittest.TestCase):

    def test_it_returns_the_next_epoch_and_restores_every_piece(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as folder:
            path = _write(Path(folder), plan, epoch=2)
            net, optimiser, schedule = _pieces(plan)
            before = schedule.get_last_lr()[0]
            rng = np.random.default_rng(999)
            generator = torch.Generator().manual_seed(999)
            first = _resume_fold(path, net, optimiser, schedule, plan, rng, generator, "cpu")
        self.assertEqual(first, 3)
        self.assertNotEqual(schedule.get_last_lr()[0], before,
                            "the schedule was not moved to where the run had reached")

    def test_the_seed_streams_are_restored_so_the_next_epoch_draws_what_it_would_have(self) -> None:
        """⛔ WITHOUT THIS A RESUMED RUN SEES DIFFERENT UNITS AND DIFFERENT SEEDS from the run it
        claims to continue, so the curve has a discontinuity nobody put there."""
        plan = _plan()
        with tempfile.TemporaryDirectory() as folder:
            source = np.random.default_rng(plan.seed)
            source.random(17)                                   # advance it, as an epoch would
            path = _write(Path(folder), plan)
            torch.save({**torch.load(path, weights_only=False),
                        "numpy_state": source.bit_generator.state}, path)
            expected = source.random(5)

            net, optimiser, schedule = _pieces(plan)
            rng = np.random.default_rng(12345)
            _resume_fold(path, net, optimiser, schedule, plan, rng,
                         torch.Generator().manual_seed(1), "cpu")
            self.assertTrue(np.allclose(rng.random(5), expected))


class RefusalTests(unittest.TestCase):
    """⛔ A resume into a changed world is a different experiment wearing the old run's card."""

    def test_a_changed_plan_is_refused_and_the_changed_field_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = _write(Path(folder), _plan())
            other = _plan(batch=8)
            net, optimiser, schedule = _pieces(other)
            with self.assertRaises(ValueError) as caught:
                _resume_fold(path, net, optimiser, schedule, other, np.random.default_rng(0),
                             torch.Generator(), "cpu")
        self.assertIn("batch", str(caught.exception))
        self.assertIn("--init-from", str(caught.exception))

    def test_a_changed_epoch_count_is_refused_because_it_changes_the_SCHEDULE(self) -> None:
        """⚠ The one that looks harmless. `CosineAnnealingLR` takes `T_max` from `epochs`, so
        continuing a twelve-epoch schedule as a twenty-epoch one is a different learning-rate curve
        under the old run's name."""
        with tempfile.TemporaryDirectory() as folder:
            path = _write(Path(folder), _plan(epochs=4))
            longer = _plan(epochs=20)
            net, optimiser, schedule = _pieces(longer)
            with self.assertRaises(ValueError) as caught:
                _resume_fold(path, net, optimiser, schedule, longer, np.random.default_rng(0),
                             torch.Generator(), "cpu")
        self.assertIn("epochs", str(caught.exception))

    def test_a_changed_MODEL_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = _write(Path(folder), _plan())
            wider = _plan(model=SetGeneratorConfig(
                backbone=SerializedConfig(width=64, depth=1, heads=4),
                head=SlotHeadConfig(slots=2, width=32, gripper_width=8, query_width=8),
                graspability_width=8))
            net, optimiser, schedule = _pieces(wider)
            with self.assertRaises(ValueError) as caught:
                _resume_fold(path, net, optimiser, schedule, wider, np.random.default_rng(0),
                             torch.Generator(), "cpu")
        self.assertIn("model", str(caught.exception))

    def test_an_OLD_checkpoint_says_why_it_cannot_be_continued(self) -> None:
        """Every checkpoint written before this landed stores its plan as a display string, so a
        changed plan could not be detected. Refused by name rather than compared against a repr."""
        plan = _plan()
        with tempfile.TemporaryDirectory() as folder:
            path = _write(Path(folder), plan)
            payload = torch.load(path, weights_only=False)
            payload["plan"] = repr(plan)                 # what every pre-2026-09-03 run wrote
            torch.save(payload, path)
            net, optimiser, schedule = _pieces(plan)
            with self.assertRaises(ValueError) as caught:
                _resume_fold(path, net, optimiser, schedule, plan, np.random.default_rng(0),
                             torch.Generator(), "cpu")
        self.assertIn("display string", str(caught.exception))
        self.assertIn("--init-from", str(caught.exception))

    def test_a_checkpoint_without_optimiser_state_is_refused(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as folder:
            path = _write(Path(folder), plan)
            payload = torch.load(path, weights_only=False)
            del payload["optimiser"]
            torch.save(payload, path)
            net, optimiser, schedule = _pieces(plan)
            with self.assertRaises(ValueError) as caught:
                _resume_fold(path, net, optimiser, schedule, plan, np.random.default_rng(0),
                             torch.Generator(), "cpu")
        self.assertIn("optimiser", str(caught.exception))
        self.assertIn("warm-started", str(caught.exception))

    def test_a_missing_checkpoint_says_there_is_nothing_to_resume(self) -> None:
        plan = _plan()
        net, optimiser, schedule = _pieces(plan)
        with self.assertRaises(FileNotFoundError):
            _resume_fold(Path("nowhere") / "fold0.pt", net, optimiser, schedule, plan,
                         np.random.default_rng(0), torch.Generator(), "cpu")


class FingerprintTests(unittest.TestCase):

    def test_the_plan_is_stored_as_data_and_not_as_a_repr(self) -> None:
        """A repr cannot be compared field by field, so a resume could only be eyeballed."""
        block = _plan_fingerprint(_plan())
        self.assertIsInstance(block, dict)
        self.assertIsInstance(block["model"], dict)
        self.assertIn("epochs", block)

    def test_two_identical_plans_fingerprint_identically(self) -> None:
        self.assertEqual(_plan_fingerprint(_plan()), _plan_fingerprint(_plan()))

    def test_the_flag_is_reachable_and_distinguishes_itself_from_init_from(self) -> None:
        text = Path("src/robot/grasping/deep/__main__.py").read_text(encoding="utf-8")
        self.assertIn('"--resume"', text)
        self.assertIn("resume=args.resume", text)
        self.assertIn("Use --init-from", text)


if __name__ == "__main__":
    unittest.main()
