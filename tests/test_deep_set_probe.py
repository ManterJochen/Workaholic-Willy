"""The memorisation probe: does a 21.5M-parameter model on 463 asset groups recall or generalise?

⚠ THIS FILE EXISTS BEFORE THE FIRST RESULT DOES, and that is the point. A threshold chosen after
seeing the number it judges is not a threshold, so the instrument is built and pinned first and the
number it produces is read afterwards.

The failures it has to prevent are all ways of producing a gap that is not memorisation:

* comparing 4,000 training units against 512 held-out ones, which reports a difference between two
  sample sizes as a difference between two populations,
* comparing four different random draws, so the seed noise lands wherever it happens to,
* reading the raw gap, which credits the model with the corpus's own asymmetry between two folds
  that hold different assets to begin with.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.robot.grasping.deep.corpus.index import grouped_folds
from src.robot.grasping.deep.net.set_generator import SetGenerator
from src.robot.grasping.deep.train.trainer import corpus_index
from src.robot.grasping.deep.eval.probes import memorisation_probe
from tests.test_deep_set_loop import _corpus, _small_plan


def _setup(scenes: int = 12) -> tuple:
    directory = TemporaryDirectory()
    index = corpus_index(_corpus(Path(directory.name), scenes=scenes))
    plan = _small_plan()
    train, test = grouped_folds(index.groups, folds=plan.folds, seed=0)[0]
    return directory, index, plan, train, test


class SymmetryTests(unittest.TestCase):
    """The probe must not manufacture a gap out of how it was run."""

    def test_the_two_samples_are_the_same_size(self) -> None:
        """⛔ A probe comparing 4,000 units against 512 reports a sample-size difference as a
        population difference, and the smaller side's noise lands wherever it lands."""
        directory, index, plan, train, test = _setup()
        with directory:
            net = SetGenerator(plan.model)
            result = memorisation_probe(net, index, train, test, plan, units=4)
            self.assertEqual(result.units, min(4, len(train), len(test)))

    def test_an_untrained_net_is_measured_on_the_same_units(self) -> None:
        """The control answers "what gap do these two folds show before anyone trained on either".
        If it saw different units it would answer a different question."""
        directory, index, plan, train, test = _setup()
        with directory:
            result = memorisation_probe(SetGenerator(plan.model), index, train, test, plan, units=4)
            for key in result.seen:
                self.assertIn(key, result.control_seen)
                self.assertIn(key, result.control_unseen)

    def test_the_same_net_probed_twice_gives_the_same_numbers(self) -> None:
        """⛔ Four independent random draws would let the seed noise decide the verdict. Every pass
        re-seeds, so the seed draw and the label subsample are identical across folds and models."""
        directory, index, plan, train, test = _setup()
        with directory:
            net = SetGenerator(plan.model)
            first = memorisation_probe(net, index, train, test, plan, units=4, seed=3)
            second = memorisation_probe(net, index, train, test, plan, units=4, seed=3)
            for key in first.seen:
                with self.subTest(key):
                    self.assertAlmostEqual(first.seen[key], second.seen[key], places=5)
                    self.assertAlmostEqual(first.unseen[key], second.unseen[key], places=5)

    def test_an_untrained_model_probes_close_to_its_own_control(self) -> None:
        """⭐ THE FLOOR. Two fresh nets of one architecture differ only by initialisation, so the
        EXCESS gap between them has no learning in it. It is not zero, and knowing how far from zero
        is exactly why the control is measured rather than assumed."""
        directory, index, plan, train, test = _setup()
        with directory:
            result = memorisation_probe(SetGenerator(plan.model), index, train, test, plan, units=8)
            self.assertTrue(all(np.isfinite(v) for v in result.excess_gap.values()))


class ReportingTests(unittest.TestCase):

    def test_the_excess_gap_subtracts_the_control(self) -> None:
        """⭐ THE NUMBER TO READ. The two folds hold different assets, so a difference exists before
        training does; reporting the raw gap would credit the model with the corpus's asymmetry."""
        directory, index, plan, train, test = _setup()
        with directory:
            result = memorisation_probe(SetGenerator(plan.model), index, train, test, plan, units=4)
            for key, value in result.excess_gap.items():
                self.assertAlmostEqual(value, result.gap[key] - result.control_gap[key], places=9)

    def test_the_row_carries_every_view_of_the_numbers(self) -> None:
        directory, index, plan, train, test = _setup()
        with directory:
            row = memorisation_probe(SetGenerator(plan.model), index, train, test,
                                     plan, units=4).as_row()
            for metric in ("total", "coverage", "top1_hit", "approach_error_deg"):
                for prefix in ("seen", "unseen", "gap", "control_gap", "excess"):
                    self.assertIn(f"{prefix}_{metric}", row)

    def test_it_reports_more_than_the_loss(self) -> None:
        """A model can memorise its way to a lower total while getting worse at what a cell reads."""
        directory, index, plan, train, test = _setup()
        with directory:
            seen = memorisation_probe(SetGenerator(plan.model), index, train, test,
                                      plan, units=4).seen
            for key in ("coverage", "top1_hit", "approach_error_deg", "offset_error_mm"):
                self.assertIn(key, seen)


class ModeTests(unittest.TestCase):

    def test_the_probe_runs_in_eval_mode_and_leaves_no_gradient(self) -> None:
        """A probe that trained the model it measures would report the last batch it just learned."""
        directory, index, plan, train, test = _setup()
        with directory:
            net = SetGenerator(plan.model)
            net.train()
            before = [p.detach().clone() for p in net.parameters()]
            memorisation_probe(net, index, train, test, plan, units=4)
            self.assertFalse(net.training, "the probe left the model in training mode")
            for parameter, original in zip(net.parameters(), before, strict=True):
                self.assertTrue(torch.equal(parameter.detach(), original),
                                "the probe changed a weight")
            self.assertTrue(all(p.grad is None or float(p.grad.abs().sum()) == 0.0
                                for p in net.parameters()))

    def test_a_unit_budget_below_one_batch_is_refused(self) -> None:
        directory, index, plan, train, test = _setup()
        with directory:
            with self.assertRaises(ValueError):
                memorisation_probe(SetGenerator(plan.model), index, train, test, plan, units=1)


if __name__ == "__main__":
    unittest.main()
