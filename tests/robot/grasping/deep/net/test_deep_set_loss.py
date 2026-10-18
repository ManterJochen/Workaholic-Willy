"""Scoring K predicted grasps against the several a seed admits.

⚠ THE FAILURE THIS FILE IS FOR is a matching decided by the wrong thing. The several grasps a point
admits sit, by construction, at almost the same PLACE and differ in APPROACH: measured on v5, 75.2 %
of supervised points hold more than one approach further than 15 degrees apart, against 40.4 % for
the closing axis. A cost dominated by the offset term would pair slots with labels by position and be
structurally blind to the distinction the whole head exists to make. It would also look fine: the
loss would fall, the coverage would look reasonable, and every slot would answer the same mode.

So the tests below check BEHAVIOUR under a constructed disagreement rather than the weight values.
The values are a starting point and will be swept; what may not change is which term decides.
"""

from __future__ import annotations

import math
import unittest

import torch

from src.robot.grasping.deep.net.assignment import optimal_one_to_one
from src.robot.grasping.deep.net.rotation import director_matrix
from src.robot.grasping.deep.net.set_loss import (
    GraspSetPrediction,
    GraspSetTarget,
    SetLossWeights,
    grasp_set_cost,
    grasp_set_loss,
)


def _params_of(axis: torch.Tensor) -> torch.Tensor:
    matrix = director_matrix(axis)
    return torch.stack([matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2],
                        matrix[..., 1, 1], matrix[..., 1, 2], matrix[..., 2, 2]], dim=-1)


def _prediction(approach: torch.Tensor, axis: torch.Tensor, offset: torch.Tensor,
                width: torch.Tensor, confidence: torch.Tensor | None = None
                ) -> GraspSetPrediction:
    return GraspSetPrediction(
        approach=approach, axis_params=_params_of(axis), offset_m=offset, width_m=width,
        confidence=confidence if confidence is not None else torch.zeros_like(width))


def _target(approach: torch.Tensor, axis: torch.Tensor, offset: torch.Tensor,
            width: torch.Tensor, valid: torch.Tensor | None = None) -> GraspSetTarget:
    return GraspSetTarget(
        approach=approach, axis=axis, offset_m=offset, width_m=width,
        valid=valid if valid is not None else torch.ones_like(width, dtype=torch.bool))


class CostTests(unittest.TestCase):

    def test_an_exact_answer_costs_almost_nothing(self) -> None:
        approach = torch.tensor([[[0.0, 0.0, -1.0]]])
        axis = torch.tensor([[[1.0, 0.0, 0.0]]])
        offset = torch.tensor([[[0.001, 0.0, 0.002]]])
        width = torch.tensor([[0.04]])
        cost = grasp_set_cost(_prediction(approach, axis, offset, width),
                              _target(approach, axis, offset, width))
        self.assertLess(float(cost), 1e-5)

    def test_the_axis_term_ignores_the_label_sign(self) -> None:
        """⛔ The corpus stores the other sign on 64 % of labels. A signed axis term would make the
        matching a function of bookkeeping rather than of geometry."""
        approach = torch.tensor([[[0.0, 0.0, -1.0]]])
        axis = torch.tensor([[[1.0, 0.0, 0.0]]])
        offset = torch.zeros(1, 1, 3)
        width = torch.tensor([[0.04]])
        prediction = _prediction(approach, axis, offset, width)
        positive = grasp_set_cost(prediction, _target(approach, axis, offset, width))
        negative = grasp_set_cost(prediction, _target(approach, -axis, offset, width))
        self.assertTrue(torch.allclose(positive, negative, atol=1e-6))


class WhichTermDecidesTests(unittest.TestCase):
    """⭐ The tests the module exists for. Two labels are made to disagree and the matching is read."""

    def _two_labels_at_one_place(self) -> GraspSetTarget:
        """Two grasps at the SAME position with approaches 90 degrees apart, which is the shape the
        corpus actually holds: the several grasps at a point differ in direction, not in place."""
        approach = torch.tensor([[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]])
        axis = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
        offset = torch.zeros(1, 2, 3)
        width = torch.tensor([[0.04, 0.04]])
        return _target(approach, axis, offset, width)

    def test_a_slot_is_matched_by_its_approach_not_by_its_position(self) -> None:
        target = self._two_labels_at_one_place()
        # One slot answers the SECOND label's direction exactly, but sits 30 mm away. If the offset
        # decided, it would be matched to the first label, which it is 90 degrees wrong about.
        prediction = _prediction(
            approach=torch.tensor([[[1.0, 0.0, 0.0]]]),
            axis=torch.tensor([[[0.0, 0.0, 1.0]]]),
            offset=torch.tensor([[[0.03, 0.0, 0.0]]]),
            width=torch.tensor([[0.04]]))
        cost = grasp_set_cost(prediction, target)
        matching = optimal_one_to_one(cost, target.valid)
        self.assertEqual(int(matching.slot_label[0, 0]), 1,
                         "the matching was decided by position rather than by approach")

    def test_position_still_breaks_a_tie_between_equal_approaches(self) -> None:
        """The offset is not ignored, it is outranked. With the directions equal it must decide."""
        approach = torch.tensor([[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]])
        axis = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        offset = torch.tensor([[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]])
        width = torch.tensor([[0.04, 0.04]])
        target = _target(approach, axis, offset, width)
        prediction = _prediction(
            approach=torch.tensor([[[0.0, 0.0, -1.0]]]),
            axis=torch.tensor([[[1.0, 0.0, 0.0]]]),
            offset=torch.tensor([[[0.048, 0.0, 0.0]]]),
            width=torch.tensor([[0.04]]))
        matching = optimal_one_to_one(grasp_set_cost(prediction, target), target.valid)
        self.assertEqual(int(matching.slot_label[0, 0]), 1)

    def test_two_slots_cover_two_modes_instead_of_both_taking_the_easier_one(self) -> None:
        """The one-to-one guard, seen through the loss rather than through the matcher."""
        target = self._two_labels_at_one_place()
        prediction = _prediction(
            approach=torch.tensor([[[0.0, 0.0, -1.0], [0.1, 0.0, -1.0]]]),
            axis=torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
            offset=torch.zeros(1, 2, 3),
            width=torch.tensor([[0.04, 0.04]]))
        matching = optimal_one_to_one(grasp_set_cost(prediction, target), target.valid)
        self.assertEqual(sorted(matching.slot_label[0].tolist()), [0, 1])
        # Coverage moved to `_metrics`, where the hit thresholds are; see
        # `test_deep_metric_is_not_the_data` for why. What is checked here is the MATCHING.
        self.assertEqual(sorted(matching.slot_label[0].tolist()), [0, 1])


class CommensurabilityTests(unittest.TestCase):
    """⛔⛔ THE DEFECT THAT MADE A LIVE RUN LOOK LIKE A FAILED ARCHITECTURE.

    The four terms are two angles, a distance and a width. The COST matrix divided its lengths by
    `_LENGTH_SCALE_M` so a weight of 1.0 meant "as important as"; the LOSS passed the same constant
    as smooth-L1's `beta`, which moves the transition point and does NOT make a metre commensurable
    with a radian.

    MEASURED at epoch 4 of a live run, the contributions to the total were axis 41.4 %, approach
    31.2 %, confidence 15.1 %, graspability 11.3 %, **offset 1.0 %, width 0.1 %**. So the offset head
    never left its initialisation, sat at 35.1 mm of error against a random baseline's 34.3, and
    `top1_hit` -- which requires an offset within 20 mm -- was capped near zero however good the
    approach became. The flat curve was read as "the architecture does not work".
    """

    def _typical(self) -> tuple:
        """One typical wrongness per channel: 50 degrees of approach, a perpendicular axis, 36 mm of
        offset (the corpus median) and 20 mm of width."""
        angle = torch.deg2rad(torch.tensor(50.0))
        target = GraspSetTarget(
            approach=torch.tensor([[[0.0, 0.0, -1.0]]]), axis=torch.tensor([[[1.0, 0.0, 0.0]]]),
            offset_m=torch.zeros(1, 1, 3), width_m=torch.tensor([[0.05]]),
            valid=torch.ones(1, 1, dtype=torch.bool))
        prediction = GraspSetPrediction(
            approach=torch.tensor([[[float(torch.sin(angle)), 0.0, -float(torch.cos(angle))]]]),
            axis_params=_params_of(torch.tensor([[[0.0, 1.0, 0.0]]])),
            offset_m=torch.tensor([[[0.036, 0.0, 0.0]]]), width_m=torch.tensor([[0.070]]),
            confidence=torch.tensor([[0.0]]))
        return prediction, target

    def test_a_typical_offset_error_is_not_a_rounding_error_beside_the_angles(self) -> None:
        """⭐ THE GUARD. The offset must be within an order of magnitude of the approach for a typical
        wrongness in each, or the optimiser has no reason to fix a head the hit criterion depends on.
        """
        terms = grasp_set_loss(*self._typical())
        weights = SetLossWeights()
        approach = float(terms["approach"]) * weights.loss_approach
        offset = float(terms["offset"]) * weights.loss_offset
        self.assertGreater(offset, approach / 10.0,
                           f"offset {offset:.4f} is negligible beside approach {approach:.4f}")
        self.assertLess(offset, approach * 10.0)

    def test_a_metre_of_error_is_not_measured_in_metres(self) -> None:
        """Doubling the offset error must more than double its loss in the quadratic region, and the
        absolute scale must be order one rather than order 0.01."""
        prediction, target = self._typical()
        near = float(grasp_set_loss(prediction, target)["offset"])
        far = GraspSetPrediction(
            approach=prediction.approach, axis_params=prediction.axis_params,
            offset_m=prediction.offset_m * 2.0, width_m=prediction.width_m,
            confidence=prediction.confidence)
        self.assertGreater(near, 0.05, "a 36 mm error costs less than a rounding error")
        self.assertGreater(float(grasp_set_loss(far, target)["offset"]), near * 2.0)

    def test_every_term_is_order_one_for_a_maximal_error(self) -> None:
        """A term whose maximum is 0.01 cannot compete with one whose maximum is 2, whatever its
        weight says."""
        terms = grasp_set_loss(*self._typical())
        for key in ("approach", "axis", "offset"):
            with self.subTest(key):
                self.assertGreater(float(terms[key]), 0.05)
                self.assertLess(float(terms[key]), 10.0)


class LossTests(unittest.TestCase):

    def _batch(self, slots: int = 3, labels: int = 4) -> tuple:
        generator = torch.Generator().manual_seed(5)
        approach = torch.randn(2, slots, 3, generator=generator)
        axis = torch.randn(2, slots, 3, generator=generator)
        axis = axis / axis.norm(dim=-1, keepdim=True)
        prediction = _prediction(
            approach, axis, torch.randn(2, slots, 3, generator=generator) * 0.02,
            torch.rand(2, slots, generator=generator) * 0.08,
            torch.randn(2, slots, generator=generator))
        label_axis = torch.randn(2, labels, 3, generator=generator)
        label_axis = label_axis / label_axis.norm(dim=-1, keepdim=True)
        target = _target(torch.randn(2, labels, 3, generator=generator), label_axis,
                         torch.randn(2, labels, 3, generator=generator) * 0.02,
                         torch.rand(2, labels, generator=generator) * 0.08)
        return prediction, target

    def test_every_term_is_reported_separately(self) -> None:
        """A set head can improve its total while getting worse at the one thing a cell cares about,
        and a run that logged only the sum could not tell."""
        terms = grasp_set_loss(*self._batch())
        for key in ("approach", "axis", "offset", "width", "confidence", "total"):
            self.assertIn(key, terms)
            self.assertEqual(terms[key].shape, ())

    def test_the_total_is_the_weighted_sum_of_the_parts(self) -> None:
        weights = SetLossWeights()
        terms = grasp_set_loss(*self._batch(), weights=weights)
        expected = (weights.loss_approach * terms["approach"]
                    + weights.loss_axis * terms["axis"]
                    + weights.loss_offset * terms["offset"]
                    + weights.loss_width * terms["width"]
                    + weights.loss_confidence * terms["confidence"])
        self.assertAlmostEqual(float(terms["total"]), float(expected), places=6)

    def test_a_perfect_prediction_has_no_pose_loss_left(self) -> None:
        approach = torch.tensor([[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]])
        axis = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
        offset = torch.tensor([[[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]]])
        width = torch.tensor([[0.03, 0.05]])
        terms = grasp_set_loss(_prediction(approach, axis, offset, width),
                               _target(approach, axis, offset, width))
        for key in ("approach", "axis", "offset", "width"):
            with self.subTest(key):
                self.assertLess(float(terms[key]), 1e-6)

    def test_an_unmatched_slot_is_supervised_on_its_confidence_alone(self) -> None:
        """⛔ Pushing an empty slot's POSE anywhere would invent a target, and pushing it toward the
        nearest label is the mode-dropping failure the one-to-one matching exists to prevent."""
        approach = torch.tensor([[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]])
        axis = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
        offset = torch.zeros(1, 2, 3)
        width = torch.tensor([[0.04, 0.04]])
        target = _target(approach[:, :1], axis[:, :1], offset[:, :1], width[:, :1])

        # ⚠ The matched slot is deliberately slightly WRONG. `1 - cos` is smooth at its minimum, so
        # an exact prediction has zero gradient for a correct reason, and a test built on one would
        # read "the matched slot got no gradient either" and prove nothing.
        free = torch.tensor([[[0.1, 0.0, -1.0], [1.0, 0.0, 0.0]]], requires_grad=True)
        prediction = GraspSetPrediction(
            approach=free, axis_params=_params_of(axis), offset_m=offset, width_m=width,
            confidence=torch.zeros(1, 2))
        grasp_set_loss(prediction, target)["approach"].backward()
        self.assertGreater(float(free.grad[0, 0].abs().sum()), 0.0)
        self.assertEqual(float(free.grad[0, 1].abs().sum()), 0.0,
                         "the unmatched slot received a pose gradient")

    def test_a_seed_with_no_labels_still_produces_a_finite_loss(self) -> None:
        """v5 holds objects no grasp reaches. A batch of them must train the confidence and not
        divide by zero on its way there."""
        prediction, target = self._batch()
        empty = GraspSetTarget(approach=target.approach, axis=target.axis,
                               offset_m=target.offset_m, width_m=target.width_m,
                               valid=torch.zeros_like(target.valid))
        terms = grasp_set_loss(prediction, empty)
        for key, value in terms.items():
            with self.subTest(key):
                self.assertTrue(math.isfinite(float(value)))
        self.assertEqual(float(terms["approach"]), 0.0)
        self.assertGreater(float(terms["confidence"]), 0.0)

    def test_the_confidence_pushes_unmatched_slots_down_and_matched_ones_up(self) -> None:
        approach = torch.tensor([[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]])
        axis = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
        offset = torch.zeros(1, 2, 3)
        width = torch.tensor([[0.04, 0.04]])
        logits = torch.zeros(1, 2, requires_grad=True)
        prediction = GraspSetPrediction(
            approach=approach, axis_params=_params_of(axis), offset_m=offset, width_m=width,
            confidence=logits)
        target = _target(approach[:, :1], axis[:, :1], offset[:, :1], width[:, :1])
        grasp_set_loss(prediction, target)["confidence"].backward()
        self.assertLess(float(logits.grad[0, 0]), 0.0)     # matched: pushed up
        self.assertGreater(float(logits.grad[0, 1]), 0.0)  # unmatched: pushed down

    def test_the_loss_and_the_coverage_come_from_one_matching(self) -> None:
        """Passing the matching in is how a caller keeps the loss and the metric from disagreeing
        about one batch."""
        prediction, target = self._batch()
        shared = optimal_one_to_one(grasp_set_cost(prediction, target), target.valid)
        with_shared = grasp_set_loss(prediction, target, matching=shared)
        computed = grasp_set_loss(prediction, target)
        for key in with_shared:
            with self.subTest(key):
                self.assertAlmostEqual(float(with_shared[key]), float(computed[key]), places=6)

    def test_gradients_reach_every_head(self) -> None:
        prediction, target = self._batch()
        tensors = {
            "approach": prediction.approach.detach().requires_grad_(True),
            "axis": prediction.axis_params.detach().requires_grad_(True),
            "offset": prediction.offset_m.detach().requires_grad_(True),
            "width": prediction.width_m.detach().requires_grad_(True),
            "confidence": prediction.confidence.detach().requires_grad_(True),
        }
        grasp_set_loss(GraspSetPrediction(
            approach=tensors["approach"], axis_params=tensors["axis"],
            offset_m=tensors["offset"], width_m=tensors["width"],
            confidence=tensors["confidence"]), target)["total"].backward()
        for name, tensor in tensors.items():
            with self.subTest(name):
                self.assertIsNotNone(tensor.grad)
                self.assertGreater(float(tensor.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
