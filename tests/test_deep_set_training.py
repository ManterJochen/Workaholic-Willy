"""One step of the set generator, end to end on a real sample.

⚠ THIS IS A WIRING TEST AND IT IS SUPPOSED TO BE. Every part underneath was built and measured on its
own; what has never been checked before this file is that they fit together. The failures a wiring
test has to catch are the ones that produce a number rather than an error:

* a seed drawn from a field the gradient flows back through, which teaches the WHERE stage to propose
  places that are EASY rather than places worth grasping,
* a metric computed over the head's best slot rather than its most confident one, which reports an
  oracle over the head's own output,
* a term that reaches the total twice, or not at all.

The sample comes from `build_sample` rather than from a fixture, so the key names, the frames and the
dtypes are the real ones. A wiring test against invented tensors checks the test's idea of the
interface.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.robot.grasping.deep.corpus.sample import SampleSpec, build_sample
from src.robot.grasping.deep.net.gripper import GRIPPER_VECTOR_DIM, swept_volume_vector
from src.robot.grasping.deep.net.slot_head import SlotHeadConfig
from src.robot.grasping.deep.net.serialized_backbone import SerializedConfig
from src.robot.grasping.deep.net.set_generator import SetGeneratorConfig, SetGenerator
from src.robot.grasping.deep.train.step import SetStepConfig, set_training_step
from tests.test_deep_dataset import _scene

_JAW = {
    "aperture_mm": 85.0, "min_width_mm": 5.0, "finger_ahead_mm": 28.72,
    "finger_behind_mm": 33.37, "finger_thickness_mm": 31.35, "finger_width_mm": 27.0,
    "pad_span_mm": 38.0, "friction_coefficient": 0.5,
}


def _net(slots: int = 3) -> SetGenerator:
    """Small enough to run in a test, same shape as the real one."""
    return SetGenerator(SetGeneratorConfig(
        backbone=SerializedConfig(width=32, depth=2, heads=4, window=64),
        head=SlotHeadConfig(slots=slots, width=64, gripper_width=8, query_width=8)))


def _grippers(count: int) -> torch.Tensor:
    """One row per sample, which is what the step takes so a batch may mix grippers."""
    return swept_volume_vector(**_JAW).expand(count, GRIPPER_VECTOR_DIM)


def _samples(count: int = 2) -> list[dict]:
    spec = SampleSpec(grasp_set=True, points=512)
    rng = np.random.default_rng(4)
    return [build_sample(_scene(), rng, spec, target_instance=None) for _ in range(count)]


class StepTests(unittest.TestCase):

    def test_a_step_reports_every_term_and_every_metric(self) -> None:
        out = set_training_step(_net(), _samples(), _grippers(2))
        for key in ("total", "approach", "axis", "offset", "width", "confidence", "graspability",
                    "coverage_sum", "coverage_count", "top1_hit_sum", "top1_hit_count",
                    "approach_error_deg_sum", "offset_error_mm_sum", "slots_filled_sum",
                    "truncated", "seeds_labelled", "seeds_predicted", "seeds_random"):
            with self.subTest(key):
                self.assertIn(key, out)
                self.assertTrue(torch.isfinite(out[key]).all(), f"{key} is not finite")

    def test_the_total_carries_the_where_stage_exactly_once(self) -> None:
        """⛔ A term that reaches the total twice moves the optimiser twice as hard for a reason
        nobody can see in a curve."""
        net, samples, gripper = _net(), _samples(), _grippers(2)
        torch.manual_seed(0)
        plain = set_training_step(net, samples, gripper,
                                  config=SetStepConfig(graspability_weight=0.0))
        torch.manual_seed(0)
        weighted = set_training_step(net, samples, gripper,
                                     config=SetStepConfig(graspability_weight=1.0))
        difference = float(weighted["total"] - plain["total"])
        self.assertAlmostEqual(difference, float(weighted["graspability"]), places=4)

    def test_gradients_reach_the_backbone_and_both_heads(self) -> None:
        net = _net()
        set_training_step(net, _samples(), _grippers(2))["total"].backward()
        for name, module in (("backbone", net.backbone.embed),
                             ("graspability", net.graspability[0]),
                             ("head", net.head.output)):
            with self.subTest(name):
                self.assertIsNotNone(module.weight.grad)
                self.assertGreater(float(module.weight.grad.abs().sum()), 0.0)

    def test_the_seed_draw_carries_no_gradient(self) -> None:
        """⛔ THE SUBTLE ONE. If the draw were differentiable the WHERE head would learn to propose
        seeds that are easy to answer instead of seeds worth grasping, which is a reward for looking
        away. Checked by removing the WHAT loss entirely: the field must still be trained, and only
        by its own term."""
        net = _net()
        out = set_training_step(net, _samples(), _grippers(2),
                                config=SetStepConfig(graspability_weight=0.0))
        out["graspability"].backward(retain_graph=True)
        own = float(net.graspability[0].weight.grad.abs().sum())
        net.zero_grad()
        out["approach"].backward()
        through_seeds = net.graspability[0].weight.grad
        self.assertGreater(own, 0.0)
        self.assertTrue(through_seeds is None or float(through_seeds.abs().sum()) == 0.0,
                        "the pose loss reached the graspability head through the seed draw")

    def test_a_gripper_block_that_does_not_match_the_batch_is_refused(self) -> None:
        """⛔ ONE ROW PER SAMPLE. A single vector silently broadcast is how an input that should vary
        becomes one that never does, and on a corpus holding several jaws it would train the head on
        a hand the labels did not come from. That is worse than no conditioning at all: a constant
        input teaches nothing, a wrong one teaches a relationship that is not there."""
        net = _net()
        with self.assertRaises(ValueError):
            set_training_step(net, _samples(), swept_volume_vector(**_JAW))
        with self.assertRaises(ValueError):
            set_training_step(net, _samples(), _grippers(5))

    def test_a_mixed_batch_gives_each_sample_its_own_gripper(self) -> None:
        """⭐ The case the whole seam exists for: same geometry, different hand, different correct
        answer. If the rows were not gathered per seed this would be indistinguishable from a batch
        where every sample carries the same jaw."""
        net = _net()
        wide = swept_volume_vector(**{**_JAW, "aperture_mm": 140.0})
        mixed = torch.stack([swept_volume_vector(**_JAW), wide])
        torch.manual_seed(0)
        a = float(set_training_step(net, _samples(), mixed)["total"])
        torch.manual_seed(0)
        b = float(set_training_step(net, _samples(), _grippers(2))["total"])
        self.assertNotAlmostEqual(a, b, places=6)

    def test_an_empty_batch_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            set_training_step(_net(), [], _grippers(2))

    def test_the_reported_seed_counts_add_up(self) -> None:
        out = set_training_step(_net(), _samples(), _grippers(2),
                                config=SetStepConfig(seeds=24))
        drawn = sum(float(out[f"seeds_{s}"]) for s in ("labelled", "predicted", "random"))
        self.assertGreater(drawn, 0.0)
        self.assertLessEqual(drawn, 24 * 2)

    def test_one_slot_and_many_slots_both_run(self) -> None:
        """`slots = 1` is the old behaviour and has to stay reachable, or the sweep has no floor."""
        for slots in (1, 5):
            with self.subTest(slots=slots):
                out = set_training_step(_net(slots), _samples(1), _grippers(1))
                self.assertTrue(torch.isfinite(out["total"]))
                self.assertLessEqual(float(out["slots_filled_sum"] / out["slots_filled_count"]),
                                     slots)


class RaggedBatchTests(unittest.TestCase):
    """⛔ MEASURED 0.08 % OF THE CORPUS, AND IT KILLED A RUN TWO MINUTES IN.

    `build_sample` returns `min(budget, points available)`. Two of 2,500 v5 clouds hold fewer than
    the 8,192-point budget, so a plain `torch.stack` refuses them, and a rate that low survives every
    test and every smoke run before taking out a six-hour job on its first batch.
    """

    def _ragged(self) -> list[dict]:
        rng = np.random.default_rng(9)
        short = build_sample(_scene(), rng, SampleSpec(grasp_set=True, points=200),
                             target_instance=None)
        full = build_sample(_scene(), rng, SampleSpec(grasp_set=True, points=400),
                            target_instance=None)
        self.assertNotEqual(len(short["points_m"]), len(full["points_m"]))
        return [short, full]

    def test_a_batch_of_unequal_clouds_runs(self) -> None:
        out = set_training_step(_net(), self._ragged(), _grippers(2))
        self.assertTrue(torch.isfinite(out["total"]))

    def test_the_padding_is_never_supervised(self) -> None:
        """⭐ `supervise` is padded with FALSE while the points are repeated cyclically, and the two
        must not be confused. A duplicated supervised point is a second seed candidate at the SAME
        place: the draw would call them distinct because their indices differ, and the loss and the
        coverage metric would both count that position twice."""
        from src.robot.grasping.deep.train.step import _padded_batch

        samples = self._ragged()
        _, _, supervise = _padded_batch(samples, torch.device("cpu"))
        short_count = len(samples[0]["points_m"])
        self.assertFalse(bool(supervise[0, short_count:].any()),
                         "a padded row was marked supervised")
        self.assertTrue(torch.equal(
            supervise[0, :short_count],
            torch.as_tensor(np.asarray(samples[0]["supervise"]), dtype=torch.bool)))

    def test_the_points_are_repeated_cyclically_rather_than_from_the_last_one(self) -> None:
        """Padding with the last point would pile hundreds of copies at one place and put a spike
        into the cloud that is not in the scene. A cyclic repeat spreads them through it."""
        from src.robot.grasping.deep.train.step import _padded_batch

        samples = self._ragged()
        points, _, _ = _padded_batch(samples, torch.device("cpu"))
        short_count = len(samples[0]["points_m"])
        padded = points[0, short_count:]
        self.assertGreater(len(padded), 1)
        self.assertGreater(float(padded.std(dim=0).max()), 1e-4,
                           "every padded point is the same one")


class MetricTests(unittest.TestCase):

    def test_top1_reads_the_most_confident_slot_not_the_best_one(self) -> None:
        """⭐ THE MEASUREMENT SHAPE THIS ARC HAS PUBLISHED WRONG ONCE. A cell receives ranked
        candidates and takes the first. Scoring the best of K reports an oracle over the head's own
        output and flatters every arm equally."""
        source = __import__("inspect").getsource(
            __import__("src.robot.grasping.deep.train.step",
                       fromlist=["_metrics"])._metrics)
        self.assertIn("confidence.argmax(dim=1)", source)
        self.assertNotIn("angle.min()", source)

    def test_a_perfect_head_would_score_one(self) -> None:
        """The metric has to be reachable. A hit threshold nobody can meet reports zero forever and
        reads as a broken model."""
        from src.robot.grasping.deep.net.rotation import director_matrix
        from src.robot.grasping.deep.net.set_loss import (
            GraspSetPrediction,
            GraspSetTarget,
        )
        from src.robot.grasping.deep.train.step import _metrics

        approach = torch.tensor([[0.0, 0.0, -1.0]]).unsqueeze(1)
        axis = torch.tensor([[1.0, 0.0, 0.0]]).unsqueeze(1)
        offset = torch.zeros(1, 1, 3)
        matrix = director_matrix(axis)
        params = torch.stack([matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2],
                              matrix[..., 1, 1], matrix[..., 1, 2], matrix[..., 2, 2]], dim=-1)
        prediction = GraspSetPrediction(approach=approach, axis_params=params, offset_m=offset,
                                        width_m=torch.tensor([[0.05]]),
                                        confidence=torch.tensor([[2.0]]))
        target = GraspSetTarget(approach=approach, axis=axis, offset_m=offset,
                                width_m=torch.tensor([[0.05]]),
                                valid=torch.ones(1, 1, dtype=torch.bool))
        from src.robot.grasping.deep.net.assignment import optimal_one_to_one
        from src.robot.grasping.deep.net.set_loss import grasp_set_cost

        matching = optimal_one_to_one(grasp_set_cost(prediction, target), target.valid)
        metrics = _metrics(prediction, target, matching, SetStepConfig())
        # ⛔ SUM OVER COUNT, because the metric is undefined where the count is zero. Reading the
        # sum alone would call a batch with no labelled seed a perfect score.
        self.assertEqual(float(metrics["top1_hit_sum"] / metrics["top1_hit_count"]), 1.0)
        self.assertLess(float(metrics["approach_error_deg_sum"]
                              / metrics["approach_error_deg_count"]), 1e-3)
        self.assertLess(float(metrics["offset_error_mm_sum"]
                              / metrics["offset_error_mm_count"]), 1e-3)


if __name__ == "__main__":
    unittest.main()
