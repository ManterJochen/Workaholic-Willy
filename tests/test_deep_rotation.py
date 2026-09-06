"""The grasp orientation representation: does it actually remove the ambiguity it claims to?

⚠ WHY THIS FILE IS SHARPER THAN A SHAPE CHECK. The defect it guards against is not a crash and not a
wrong shape. It is a loss whose optimum, for a point that has seen a grasp written down both ways,
sits BETWEEN the two correct answers. That is invisible to any test that feeds one label and checks
the output matches, because with one label everything works.

So two of these tests do not assert a property, they RUN a fit against a deliberately two-signed label
set and read where it lands. That is the only shape of test that can tell the repair from the defect,
and this repository has already paid 24 arms for the difference.
"""

from __future__ import annotations

import math
import unittest

import torch

from src.robot.grasping.deep.net.rotation import (
    DIRECTOR_NORM,
    angular_error_deg,
    approach_loss,
    decode_frame,
    director_confidence,
    director_decode,
    director_loss,
    director_matrix,
    director_params,
)


def _axes(count: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(count, 3, generator=generator, dtype=torch.float64)
    return raw / raw.norm(dim=-1, keepdim=True)


def _params_of(axis: torch.Tensor) -> torch.Tensor:
    """The six numbers whose `director_params` is exactly `director_matrix(axis)`."""
    matrix = director_matrix(axis)
    return torch.stack([matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2],
                        matrix[..., 1, 1], matrix[..., 1, 2], matrix[..., 2, 2]], dim=-1)


class RepresentationTests(unittest.TestCase):

    def test_a_director_cannot_tell_an_axis_from_its_negation(self) -> None:
        axes = _axes(64)
        self.assertTrue(torch.allclose(director_matrix(axes), director_matrix(-axes)))

    def test_the_target_norm_is_the_documented_constant(self) -> None:
        norms = director_matrix(_axes(32)).flatten(start_dim=-2).norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.full_like(norms, DIRECTOR_NORM)))

    def test_params_are_symmetric_and_traceless(self) -> None:
        raw = torch.randn(16, 6, dtype=torch.float64)
        matrices = director_params(raw)
        self.assertTrue(torch.allclose(matrices, matrices.transpose(-1, -2)))
        trace = matrices.diagonal(dim1=-2, dim2=-1).sum(-1)
        self.assertTrue(torch.allclose(trace, torch.zeros_like(trace), atol=1e-12))

    def test_decode_recovers_the_axis_up_to_sign(self) -> None:
        axes = _axes(64, seed=1)
        recovered = director_decode(_params_of(axes))
        self.assertLess(float(angular_error_deg(recovered, axes, antipodal=True).max()), 1e-4)

    def test_the_loss_is_zero_at_the_label_and_at_its_negation(self) -> None:
        axes = _axes(32, seed=2)
        params = _params_of(axes)
        for target in (axes, -axes):
            loss = director_loss(params, target, reduction="none")
            self.assertLess(float(loss.max()), 1e-20)

    def test_six_free_numbers_are_not_more_than_the_representation_needs(self) -> None:
        """Adding a multiple of the identity must not change the loss: the trace is not a degree of
        freedom the head can spend, and if it were, the head could reduce its loss by moving somewhere
        the target can never be."""
        axes = _axes(8, seed=3)
        params = _params_of(axes)
        shifted = params + torch.tensor([2.5, 0.0, 0.0, 2.5, 0.0, 2.5], dtype=torch.float64)
        self.assertTrue(torch.allclose(director_loss(params, axes, reduction="none"),
                                       director_loss(shifted, axes, reduction="none")))


class TheReasonForTheDesignTests(unittest.TestCase):
    """The two tests the file exists for. Both RUN something rather than assert a property."""

    def test_a_perpendicular_prediction_still_has_a_gradient(self) -> None:
        """⭐ THE DIRECT COMPARISON against the obvious alternative.

        `1 - (v_hat . v)^2` is also sign-invariant and is the loss most people reach for. Its gradient
        VANISHES exactly where the prediction is perpendicular to the label, which is the worst
        possible prediction and the one that most needs to be pushed. The director loss has no such
        stationary point.
        """
        target = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        perpendicular = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)

        squared_cosine = 1.0 - ((perpendicular / perpendicular.norm()) * target).sum(-1).pow(2)
        squared_cosine.backward()
        self.assertLess(float(perpendicular.grad.abs().max()), 1e-12,
                        "the alternative was expected to have NO gradient here")

        params = _params_of(perpendicular.detach()).clone().requires_grad_(True)
        director_loss(params, target).backward()
        self.assertGreater(float(params.grad.abs().max()), 0.1,
                           "the director loss must push a perpendicular prediction")

    def test_a_two_signed_label_set_converges_to_the_axis_not_to_the_average(self) -> None:
        """⭐⭐ THE MEASUREMENT THAT SEPARATES THE REPAIR FROM THE DEFECT.

        The corpus stores the same physical grasp with either sign; MEASURED, the decoded axis comes
        back negated on 64 % of labels. So fit a free head against a label set that is half `+v` and
        half `-v` and read where it lands.

        The director lands on the axis. The sign-preserving regression lands NOWHERE, and for a
        stronger reason than being inaccurate: on this label set its loss is exactly constant, because
        the mean of `+v` and `-v` is the zero vector and the loss reads only that mean. Zero gradient
        everywhere. It is not a worse fit, it is not a fit at all.
        """
        axis = torch.tensor([0.0, 0.6, 0.8], dtype=torch.float64)
        labels = torch.stack([axis] * 50 + [-axis] * 50)

        params = torch.zeros(6, dtype=torch.float64, requires_grad=True)
        optimiser = torch.optim.Adam([params], lr=0.05)
        for _ in range(600):
            optimiser.zero_grad()
            director_loss(params.expand(len(labels), 6), labels).backward()
            optimiser.step()
        error = float(angular_error_deg(director_decode(params.detach()), axis, antipodal=True))
        self.assertLess(error, 1.0, f"the director settled {error:.2f} deg off the true axis")

        vector = torch.randn(3, dtype=torch.float64, requires_grad=True)
        before = (vector / vector.norm()).detach().clone()
        optimiser = torch.optim.Adam([vector], lr=0.05)
        for _ in range(600):
            optimiser.zero_grad()
            loss = (1.0 - (vector / vector.norm() * labels).sum(-1)).mean()
            loss.backward()
            optimiser.step()
        self.assertAlmostEqual(float(loss), 1.0, places=9,
                               msg="the sign-preserving loss should be flat on a two-signed set")
        self.assertTrue(torch.allclose(vector.detach() / vector.detach().norm(), before,
                                       atol=1e-9),
                        "with no gradient the direction must not have moved at all")

    def test_taking_the_better_branch_emits_a_confident_invalid_axis(self) -> None:
        """⭐⭐⭐ THE TEST THAT SEPARATES THE DIRECTOR FROM WHAT PEOPLE ACTUALLY DO.

        The usual repair is to take the smaller of the two branch losses, which is the same thing as
        `1 - |cos|`. On the two-SIGNED set above that works, so the test before this one does not
        separate them and must not be read as if it did.

        Where it fails is a point that admits two genuinely DIFFERENT closing lines, which this
        repository measured at **74.7 %** of supervised points. Analytically, on labels `+-e1` and
        `+-e2`: aligning with either axis costs 0.5, while the bisector costs `1 - 1/sqrt(2) = 0.293`.
        So the branch loss is MINIMISED 45 degrees away from every correct answer, and it emits that
        bisector with no indication that anything is wrong.

        The director cannot resolve the two axes either, and this file has never claimed it could. The
        difference is that it SAYS SO: its confidence falls to about a half, which is exactly the
        signal the outside ablation gates on.
        """
        first = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        second = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        labels = torch.stack([first] * 25 + [-first] * 25 + [second] * 25 + [-second] * 25)

        vector = torch.tensor([0.9, 0.3, 0.1], dtype=torch.float64, requires_grad=True)
        optimiser = torch.optim.Adam([vector], lr=0.05)
        for _ in range(1500):
            optimiser.zero_grad()
            (1.0 - (vector / vector.norm() * labels).sum(-1).abs()).mean().backward()
            optimiser.step()
        branch = vector.detach()
        for label in (first, second):
            self.assertAlmostEqual(
                float(angular_error_deg(branch, label, antipodal=True)), 45.0, delta=2.0,
                msg="the branch loss was expected to settle on the bisector of the two axes")

        params = torch.zeros(6, dtype=torch.float64, requires_grad=True)
        optimiser = torch.optim.Adam([params], lr=0.05)
        for _ in range(1500):
            optimiser.zero_grad()
            director_loss(params.expand(len(labels), 6), labels).backward()
            optimiser.step()
        self.assertAlmostEqual(float(director_confidence(params.detach())), 0.5, delta=0.05,
                               msg="the director must report that it is hedging, not hide it")


class ConfidenceTests(unittest.TestCase):

    def test_an_exact_director_is_fully_confident(self) -> None:
        confidence = director_confidence(_params_of(_axes(16, seed=4)))
        self.assertTrue(torch.allclose(confidence, torch.ones_like(confidence)))

    def test_hedging_between_two_axes_lowers_the_confidence(self) -> None:
        """The mean of two directors 90 degrees apart has a smaller norm than either, so a head with
        nothing to say reports it. That is the concentration signal, and it costs nothing."""
        first = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        second = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        blended = 0.5 * (_params_of(first) + _params_of(second))
        self.assertLess(float(director_confidence(blended)), 0.75)
        self.assertGreater(float(director_confidence(blended)), 0.0)


class ApproachTests(unittest.TestCase):

    def test_the_approach_loss_is_not_sign_invariant(self) -> None:
        """⛔ THE ASYMMETRY IS THE POINT. Reaching for the object and reaching away from it are
        different grasps, so the approach keeps its sign where the closing axis loses it."""
        approach = _axes(32, seed=5)
        self.assertLess(float(approach_loss(approach, approach)), 1e-12)
        self.assertGreater(float(approach_loss(approach, -approach)), 1.9)


class FrameTests(unittest.TestCase):

    def test_the_decoded_frame_is_orthonormal(self) -> None:
        approach = _axes(64, seed=6)
        axis = _axes(64, seed=7)
        a, c, b = decode_frame(approach, axis)
        for left, right in ((a, c), (a, b), (c, b)):
            self.assertLess(float((left * right).sum(-1).abs().max()), 1e-10)
        for vector in (a, c, b):
            self.assertTrue(torch.allclose(vector.norm(dim=-1),
                                           torch.ones(64, dtype=torch.float64)))

    def test_the_approach_survives_the_orthogonalisation(self) -> None:
        """The approach is the better determined head, so it is kept and the axis is moved."""
        approach = _axes(32, seed=8)
        a, _, _ = decode_frame(approach, _axes(32, seed=9))
        self.assertTrue(torch.allclose(a, approach))

    def test_an_axis_parallel_to_the_approach_does_not_produce_nan(self) -> None:
        approach = _axes(16, seed=10)
        a, c, b = decode_frame(approach, approach.clone())
        for vector in (a, c, b):
            self.assertFalse(bool(torch.isnan(vector).any()))
            self.assertTrue(torch.allclose(vector.norm(dim=-1),
                                           torch.ones(16, dtype=torch.float64)))

    def test_the_axis_aligned_pole_is_covered(self) -> None:
        """The fallback perpendicular is built from the first two components, so an approach along
        +z is the case where that construction itself degenerates."""
        approach = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=torch.float64)
        a, c, b = decode_frame(approach, approach.clone())
        for vector in (a, c, b):
            self.assertFalse(bool(torch.isnan(vector).any()))
        self.assertLess(float((a * c).sum(-1).abs().max()), 1e-10)


class MetricTests(unittest.TestCase):

    def test_folding_turns_a_negated_axis_from_180_degrees_into_zero(self) -> None:
        axis = _axes(32, seed=11)
        self.assertLess(float(angular_error_deg(-axis, axis, antipodal=True).max()), 1e-6)
        self.assertGreater(float(angular_error_deg(-axis, axis, antipodal=False).min()),
                           179.9)

    def test_a_right_angle_reads_as_ninety_either_way(self) -> None:
        first = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        second = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        for antipodal in (True, False):
            self.assertAlmostEqual(
                float(angular_error_deg(first, second, antipodal=antipodal)), 90.0, places=6)

    def test_the_fold_is_exactly_ninety_degrees_wide(self) -> None:
        angles = torch.linspace(0.0, math.pi, 37, dtype=torch.float64)
        predicted = torch.stack([torch.cos(angles), torch.sin(angles),
                                 torch.zeros_like(angles)], dim=-1)
        target = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        folded = angular_error_deg(predicted, target, antipodal=True)
        self.assertLessEqual(float(folded.max()), 90.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
