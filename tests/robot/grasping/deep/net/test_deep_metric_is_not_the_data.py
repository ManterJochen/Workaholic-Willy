"""Does a reported metric measure the MODEL, or does it measure the corpus?

⛔⛔ THE DEFECT THIS FILE EXISTS FOR SHIPPED, and it shipped past every other test in this suite.

`coverage` was `labels matched / labels valid`. The one-to-one matching is UNCONSTRAINED, so it binds
`min(K, L)` pairs however badly they fit, and that ratio is therefore a pure function of the label
counts and K. MEASURED: three unrelated random models scored **0.6906 each, to four decimals**. A
metric that cannot tell a random net from a perfect one is not a weak metric, it is not a metric, and
it had been reported as one of the four headline numbers for the whole architecture restart.

The test is a differential: run several UNRELATED models through the same batch and read which
columns move. Anything identical across them is a property of the data.

⚠ `slots_filled` is deliberately still constant. It is `min(K, L)` averaged over seeds and is kept as
a DIAGNOSTIC -- how many slots the data gives a head to fill is worth watching -- so it is listed as
an expected constant rather than removed. If a column ever joins or leaves that list, that is a
finding and this file is where it gets recorded.
"""

from __future__ import annotations

import unittest

import torch

from src.robot.grasping.deep.net.assignment import optimal_one_to_one
from src.robot.grasping.deep.net.rotation import director_matrix
from src.robot.grasping.deep.net.set_loss import (
    GraspSetPrediction,
    GraspSetTarget,
    grasp_set_cost,
)
from src.robot.grasping.deep.train.step import SetStepConfig, _metrics

#: Columns that are properties of the corpus and are reported as diagnostics, not as scores.
_EXPECTED_CONSTANTS = {"slots_filled"}

#: Columns that must respond to what the model predicts.
#:
#: ⭐ `top1_hit_30mm` and `top1_hit_40mm` were added by a later commit and this test REFUSED them
#: until they were listed, which is what it is for. They exist because an offline probe measured a
#: median offset error of 19.5 mm as the best achievable from oracle inputs, so the 20 mm criterion
#: sits at the edge of the reachable and a head improving from 40 mm to 22 mm would report a flat
#: zero for a whole run.
#: ⭐⭐ `axis_error_deg` was added on 2026-09-03 and this test REFUSED it too, which is the second
#: time it has caught its author. It belongs here rather than with the constants: a model predicting
#: a better closing axis scores lower on it, and three unrelated random models must therefore differ.
#:
#: It is reported under BOTH targets, deliberately. The run can now train on the approach or on the
#: closing axis, and a number that existed in only one of those arms could not compare them, which
#: is the whole reason the switch exists. MEASURED motivation: at one seed the approach spreads
#: 52.93 degrees across its labels and the closing axis spreads 0.00.
_MUST_MOVE = {"coverage", "top1_hit", "top1_hit_30mm", "top1_hit_40mm",
              "approach_error_deg", "axis_error_deg", "offset_error_mm",
              # ⛔ THE COLLAPSE DIAGNOSTICS BELONG HERE, NOT WITH THE CONSTANTS. They are functions
              # of the MODEL's output alone, so unrelated models must disagree on them; a spread that
              # was identical across three random models would be reading the data rather than the
              # head, which is the exact failure this file exists to catch. That they are diagnostics
              # and not scores does not exempt them: `slots_filled` is a diagnostic too and it sits
              # in the constants for the opposite reason, that it is a pure function of the labels.
              "slot_spread_deg", "seed_spread_deg", "width_spread_mm"}


def _params_of(axis: torch.Tensor) -> torch.Tensor:
    matrix = director_matrix(axis)
    return torch.stack([matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2],
                        matrix[..., 1, 1], matrix[..., 1, 2], matrix[..., 2, 2]], dim=-1)


def _target(seed: int = 0, seeds: int = 48, labels: int = 9) -> GraspSetTarget:
    """A target shaped like the real one, and the shape matters.

    ⚠ `gather_set_targets` PACKS the valid labels at the FRONT: slot `j` is valid exactly when
    `j < take`. A fixture that scattered the valid flags at random would put an invalid label in
    position zero on some seeds, and the oracle, which fills its most confident slot from label zero,
    would then answer with padding. The first version of this file did that and read a perfect head
    as scoring 0.5, which is a property of the fixture and not of the metric.
    """
    generator = torch.Generator().manual_seed(seed)
    approach = torch.randn(seeds, labels, 3, generator=generator)
    axis = torch.randn(seeds, labels, 3, generator=generator)
    counts = torch.randint(0, labels + 1, (seeds,), generator=generator)
    valid = torch.arange(labels).unsqueeze(0) < counts.unsqueeze(1)
    return GraspSetTarget(
        approach=approach / approach.norm(dim=-1, keepdim=True),
        axis=axis / axis.norm(dim=-1, keepdim=True),
        offset_m=torch.randn(seeds, labels, 3, generator=generator) * 0.02,
        width_m=torch.rand(seeds, labels, generator=generator) * 0.08,
        valid=valid)


def _random_model(seed: int, seeds: int = 48, slots: int = 4) -> GraspSetPrediction:
    generator = torch.Generator().manual_seed(seed)
    return GraspSetPrediction(
        approach=torch.randn(seeds, slots, 3, generator=generator),
        axis_params=torch.randn(seeds, slots, 6, generator=generator),
        offset_m=torch.randn(seeds, slots, 3, generator=generator) * 0.02,
        width_m=torch.rand(seeds, slots, generator=generator) * 0.08,
        confidence=torch.randn(seeds, slots, generator=generator))


def _oracle(target: GraspSetTarget, slots: int = 4) -> GraspSetPrediction:
    """A perfect head: its slots ARE the first labels, most confident first."""
    take = slice(0, slots)
    return GraspSetPrediction(
        approach=target.approach[:, take], axis_params=_params_of(target.axis[:, take]),
        offset_m=target.offset_m[:, take], width_m=target.width_m[:, take],
        confidence=torch.linspace(2.0, -2.0, slots).expand(target.approach.shape[0],
                                                           slots).contiguous())


def _scores(prediction: GraspSetPrediction, target: GraspSetTarget) -> dict[str, float]:
    matching = optimal_one_to_one(grasp_set_cost(prediction, target), target.valid)
    raw = _metrics(prediction, target, matching, SetStepConfig())
    out: dict[str, float] = {}
    for key, value in raw.items():
        if key.endswith("_sum"):
            base = key[:-4]
            count = float(raw[f"{base}_count"])
            out[base] = float(value) / count if count else float("nan")
    return out


class DifferentialTests(unittest.TestCase):

    def test_every_score_moves_between_unrelated_models(self) -> None:
        """⭐ THE TEST THAT WOULD HAVE CAUGHT IT. Three unrelated random models on one batch: a column
        that does not move is measuring the corpus."""
        target = _target()
        runs = [_scores(_random_model(seed), target) for seed in (1, 2, 3)]
        for key in _MUST_MOVE:
            with self.subTest(key):
                values = [run[key] for run in runs]
                self.assertGreater(max(values) - min(values), 1e-9,
                                   f"{key} is identical across unrelated models: {values}")

    def test_the_known_constants_are_still_constant_and_still_listed(self) -> None:
        """If one of these ever starts moving, that is a finding, not a fix to absorb quietly."""
        target = _target()
        runs = [_scores(_random_model(seed), target) for seed in (1, 2, 3)]
        for key in _EXPECTED_CONSTANTS:
            with self.subTest(key):
                values = [run[key] for run in runs]
                self.assertLess(max(values) - min(values), 1e-9,
                                f"{key} moved; it was listed as a corpus property")

    def test_no_reported_score_is_missing_from_either_list(self) -> None:
        """A new metric has to be classified deliberately. One added without a decision is one nobody
        has asked whether it measures anything."""
        reported = set(_scores(_random_model(1), _target()))
        self.assertEqual(reported, _MUST_MOVE | _EXPECTED_CONSTANTS)


class RangeTests(unittest.TestCase):
    """A metric also has to span its range: near the floor for noise, at the ceiling for perfection."""

    def test_a_random_model_scores_near_the_floor(self) -> None:
        scores = _scores(_random_model(7), _target())
        self.assertLess(scores["coverage"], 0.05)
        self.assertLess(scores["top1_hit"], 0.05)

    def test_a_perfect_model_reaches_the_ceiling(self) -> None:
        """⭐ `top1_hit` must reach exactly 1.0. Coverage cannot: with one-to-one matching, K slots
        bind at most K of a seed's labels, so its ceiling is a property of K against the data and has
        to be quoted with every coverage number."""
        target = _target()
        scores = _scores(_oracle(target), target)
        self.assertEqual(scores["top1_hit"], 1.0)
        self.assertLess(scores["approach_error_deg"], 1e-3)
        self.assertGreater(scores["coverage"], 0.3)
        self.assertLess(scores["coverage"], 1.0)

    def test_the_perfect_model_beats_the_random_one_on_every_score(self) -> None:
        target = _target()
        perfect, random = _scores(_oracle(target), target), _scores(_random_model(7), target)
        self.assertGreater(perfect["coverage"], random["coverage"])
        self.assertGreater(perfect["top1_hit"], random["top1_hit"])
        self.assertLess(perfect["approach_error_deg"], random["approach_error_deg"])
        self.assertLess(perfect["offset_error_mm"], random["offset_error_mm"])


if __name__ == "__main__":
    unittest.main()
