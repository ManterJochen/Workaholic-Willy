"""Does the head still tell its inputs apart, or has it learned one answer?

⛔⛔ **THE INSTRUMENT THAT WAS MISSING WHILE AN ARM COLLAPSED IN FULL VIEW.** The first model this
project put in front of a physics referee proposed 320 grasps carrying ELEVEN distinct approach
vectors, a within-scene spread whose median was 0.000 degrees, widths spanning 30.97 to 32.13 mm and
confidences spanning 0.3891 to 0.3957. It had learned a constant. Every quality metric reported that
honestly as a low number, and none of them could say WHY, so the collapse cost a training run, a
proposal pass and a physics pass before anyone read the poses themselves.

⚠ NOT A QUALITY METRIC IN EITHER DIRECTION, and the tests below are built to keep that straight. A
large spread is not skill: a random head has the largest spread of all. It is only ever read as "the
head can still tell its inputs apart", and the near-zero end is the finding.
"""

from __future__ import annotations

import unittest

import torch

from src.robot.grasping.deep.net.set_loss import GraspSetPrediction
from src.robot.grasping.deep.train.step import _spread


def _prediction(approach: torch.Tensor, width_m: torch.Tensor | None = None
                ) -> GraspSetPrediction:
    seeds, slots = approach.shape[:2]
    return GraspSetPrediction(
        approach=approach,
        axis_params=torch.zeros(seeds, slots, 6),
        offset_m=torch.zeros(seeds, slots, 3),
        width_m=torch.full((seeds, slots), 0.05) if width_m is None else width_m,
        confidence=torch.zeros(seeds, slots))


def _chosen(prediction: GraspSetPrediction) -> torch.Tensor:
    """The slot the head is most confident in, the same one `_metrics` reports on."""
    rows = torch.arange(prediction.confidence.shape[0])
    return prediction.approach[rows, prediction.confidence.argmax(dim=1)]


def _scores(prediction: GraspSetPrediction) -> dict[str, float]:
    raw = _spread(prediction, _chosen(prediction))
    return {key[: -len("_sum")]: float(raw[key]) / float(raw[key[: -len("_sum")] + "_count"])
            for key in raw if key.endswith("_sum")}


class CollapseTests(unittest.TestCase):

    def test_ONE_DIRECTION_FOR_EVERY_SEED_reads_as_zero(self) -> None:
        """⛔ THE MEASURED FAILURE, reproduced. 64 seeds, 4 slots, one direction throughout."""
        approach = torch.tensor([0.469, -0.075, -0.880]).expand(64, 4, 3).contiguous()
        scores = _scores(_prediction(approach))
        self.assertLess(scores["seed_spread_deg"], 1e-3)
        self.assertLess(scores["slot_spread_deg"], 1e-3)

    def test_slots_frozen_but_SEEDS_varying_is_caught_by_the_right_number(self) -> None:
        """⛔ The two failures are different and must not be reported as one. `affine` mixing froze
        the SLOTS while the seeds still differed, measured at a slot spread of 9.6e-09 with argmax
        slot 0 for 64 of 64 seeds. A single "spread" number would have averaged that away."""
        torch.manual_seed(0)
        per_seed = torch.nn.functional.normalize(torch.randn(64, 1, 3), dim=-1)
        approach = per_seed.expand(64, 4, 3).contiguous()      # slots identical, seeds differ
        scores = _scores(_prediction(approach))
        self.assertLess(scores["slot_spread_deg"], 1e-3, "the frozen slots were not caught")
        self.assertGreater(scores["seed_spread_deg"], 30.0, "the varying seeds were called frozen")

    def test_seeds_frozen_but_SLOTS_varying_is_caught_the_other_way(self) -> None:
        """The mirror case, so neither number is quietly reading the other."""
        torch.manual_seed(1)
        per_slot = torch.nn.functional.normalize(torch.randn(1, 4, 3), dim=-1)
        approach = per_slot.expand(64, 4, 3).contiguous()      # seeds identical, slots differ
        scores = _scores(_prediction(approach))
        self.assertGreater(scores["slot_spread_deg"], 20.0)
        self.assertLess(scores["seed_spread_deg"], 1e-3)

    def test_a_HEALTHY_head_is_far_from_the_floor(self) -> None:
        """The control. A detector that fires on everything is not a detector."""
        torch.manual_seed(2)
        approach = torch.nn.functional.normalize(torch.randn(64, 4, 3), dim=-1)
        scores = _scores(_prediction(approach))
        self.assertGreater(scores["slot_spread_deg"], 30.0)
        self.assertGreater(scores["seed_spread_deg"], 30.0)

    def test_the_WIDTH_head_gets_the_same_question(self) -> None:
        """Measured on the collapsed run: widths spanned 30.97 to 32.13 mm across 320 grasps, a
        constant wearing three significant figures."""
        approach = torch.nn.functional.normalize(torch.randn(64, 4, 3), dim=-1)
        flat = _scores(_prediction(approach, width_m=torch.full((64, 4), 0.031)))
        varied = _scores(_prediction(approach, width_m=torch.rand(64, 4) * 0.1))
        self.assertLess(flat["width_spread_mm"], 1e-6)
        self.assertGreater(varied["width_spread_mm"], 5.0)

    def test_a_CANCELLING_batch_does_not_poison_the_epoch(self) -> None:
        """⚠ A mean direction can be the zero vector when directions cancel exactly. Only the
        near-zero end of this metric is ever read, so the value there does not matter, but a NaN
        would propagate into the epoch's sum and take every other seed's number with it."""
        approach = torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 0.0, -1.0]]])
        scores = _scores(_prediction(approach))
        for key, value in scores.items():
            with self.subTest(key):
                self.assertEqual(value, value, f"{key} is NaN")

    def test_a_SINGLE_slot_head_reports_zero_slot_spread_without_dividing_by_zero(self) -> None:
        """K = 1 is a real configuration (`--slots 1` is the other arm of the K sweep), and a slot
        spread is undefined there rather than alarming."""
        approach = torch.nn.functional.normalize(torch.randn(16, 1, 3), dim=-1)
        scores = _scores(_prediction(approach))
        self.assertEqual(scores["slot_spread_deg"], 0.0)
        self.assertGreater(scores["seed_spread_deg"], 30.0)


class WiringTests(unittest.TestCase):

    def test_the_kill_switch_is_OFF_by_default(self) -> None:
        """⚠ A stop condition is a behaviour change, and every seam here ships default-off."""
        from src.robot.grasping.deep.train.trainer import SetTrainingPlan

        self.assertEqual(SetTrainingPlan().collapse_floor_deg, 0.0)

    def test_the_flag_reaches_the_plan(self) -> None:
        from src.robot.grasping.deep.__main__ import build_parser

        args = build_parser().parse_args(
            ["train-set", "--clouds", "c", "--out", "o", "--collapse-floor-deg", "1.5"])
        self.assertEqual(args.collapse_floor_deg, 1.5)
        from pathlib import Path

        source = Path("src/robot/grasping/deep/__main__.py").read_text(encoding="utf-8")
        self.assertIn("collapse_floor_deg=args.collapse_floor_deg", source)

    def test_it_does_not_fire_before_the_head_has_had_a_chance(self) -> None:
        """⚠ An untrained head starts near a constant. Killing at epoch zero would refuse every
        run, so the check begins at epoch three."""
        from pathlib import Path

        source = Path("src/robot/grasping/deep/train/trainer.py").read_text(encoding="utf-8")
        self.assertIn("epoch >= 2", source)

    def test_a_stop_is_RECORDED_and_not_only_logged(self) -> None:
        """A run that stopped for this reason is a result about the configuration, and a reader of
        `epochs.json` alone must be able to see it."""
        from pathlib import Path

        source = Path("src/robot/grasping/deep/train/trainer.py").read_text(encoding="utf-8")
        self.assertIn('report.setdefault("collapsed", [])', source)


if __name__ == "__main__":
    unittest.main()
