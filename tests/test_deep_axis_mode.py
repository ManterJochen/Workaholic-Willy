"""Two ways to write down a closing axis, and the measurement that made the second one necessary.

⭐⭐ WHY THE SECOND ONE EXISTS. The architecture plan describes both and calls the cheap one the
default: "predict a 3-vector, loss `1 - (v_hat . v)^2`. Manifestly sign-invariant. Perpendicular is
the WORST point of this loss rather than an attractor, so mode averaging is eliminated. Ten lines, no
numerical risk." Only the other one got built, and for twenty-five arms nobody noticed.

⛔⛔ **AND THE MEASUREMENT I BUILT IT ON WAS CONFOUNDED, WHICH THIS FILE'S OWN TEST FOUND.** A corpus
run had the APPROACH head at 1.70 degrees and the AXIS head at 31.32 by epoch three, and I read that
as eighteen times worse for the same job. It is not the same job: the approach control's target is
`-normal`, a LINEAR function of an input channel, and the axis control's is
`normalise(cross(normal, z))`, which is not. The comparison moved the representation and the target
together and attributed the whole gap to one of them.

The direct comparison, below, puts the director AHEAD at small scale: 0.49 degrees against 1.17. So
the representation question is open and is settled by an arm that holds the target fixed and moves
only `--axis-mode`, which is what the flag exists for.

What these tests guard is what can be asserted without that arm: both modes are genuinely
sign-invariant, both decode to a unit vector, neither is broken on a target it can express, and the
default stays byte-identical so the twenty-five measured arms remain comparable.
"""

from __future__ import annotations

import unittest

import torch

from src.robot.grasping.deep.net.slot_head import SlotGraspHead, SlotHeadConfig
from src.robot.grasping.deep.net.rotation import (
    AXIS_MODES, axis_decode, axis_loss, vector_axis_decode, vector_axis_loss)


def _axes(count: int = 64, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(count, 3), dim=-1)


class SignTests(unittest.TestCase):
    """⛔ THE PROPERTY THE WHOLE REPRESENTATION EXISTS FOR. A closing axis and its negation are the
    same grasp, so a loss that can tell them apart teaches a head to guess a coin flip."""

    def test_both_modes_are_sign_invariant(self) -> None:
        raw = torch.randn(64, 6)
        axis = _axes()
        for mode in AXIS_MODES:
            with self.subTest(mode):
                self.assertAlmostEqual(float(axis_loss(raw, axis, mode=mode)),
                                       float(axis_loss(raw, -axis, mode=mode)), places=6)

    def test_the_vector_loss_is_zero_on_a_perfect_prediction_of_either_sign(self) -> None:
        axis = _axes()
        raw = torch.cat([axis, torch.randn(64, 3)], dim=-1)
        self.assertLess(float(vector_axis_loss(raw, axis)), 1e-6)
        self.assertLess(float(vector_axis_loss(raw, -axis)), 1e-6)

    def test_perpendicular_is_the_WORST_point_and_not_an_attractor(self) -> None:
        """⭐ The property that rules out mode averaging. A loss whose minimum sat between two
        opposed labels would pull a head onto the bisector, which is a pose neither label describes.
        Here the bisector is the maximum."""
        axis = torch.tensor([[0.0, 0.0, 1.0]])
        aligned = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        square = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        halfway = torch.tensor([[0.7071, 0.0, 0.7071, 0.0, 0.0, 0.0]])
        self.assertLess(float(vector_axis_loss(aligned, axis)), 1e-6)
        self.assertAlmostEqual(float(vector_axis_loss(square, axis)), 1.0, places=4)
        self.assertLess(float(vector_axis_loss(halfway, axis)),
                        float(vector_axis_loss(square, axis)))


class DecodeTests(unittest.TestCase):

    def test_both_modes_decode_to_a_unit_vector(self) -> None:
        raw = torch.randn(32, 4, 6)
        for mode in AXIS_MODES:
            with self.subTest(mode):
                decoded = axis_decode(raw, mode=mode)
                self.assertEqual(decoded.shape, (32, 4, 3))
                self.assertTrue(torch.allclose(decoded.norm(dim=-1), torch.ones(32, 4), atol=1e-5))

    def test_the_vector_decode_round_trips_exactly(self) -> None:
        """No eigendecomposition, so this is arithmetic rather than an approximation.

        ⚠ THAT IS A PROPERTY, NOT AN ADVANTAGE. An earlier version of this line went on to say it is
        "part of why it is slower to learn", which nothing measured supports and which the test at
        the bottom of this file contradicts."""
        axis = _axes(32, seed=3)
        raw = torch.cat([axis * 4.2, torch.randn(32, 3)], dim=-1)
        self.assertTrue(torch.allclose(vector_axis_decode(raw), axis, atol=1e-5))

    def test_a_zero_prediction_decodes_without_a_nan(self) -> None:
        decoded = vector_axis_decode(torch.zeros(8, 6))
        self.assertFalse(bool(torch.isnan(decoded).any()))


class HeadTests(unittest.TestCase):

    @staticmethod
    def _run(mode: str):
        torch.manual_seed(0)
        head = SlotGraspHead(16, SlotHeadConfig(slots=4, axis_mode=mode))
        torch.manual_seed(1)
        with torch.no_grad():
            return head(torch.randn(32, 16), torch.randn(32, 14))

    def test_the_raw_outputs_are_identical_between_the_modes(self) -> None:
        """⭐ SO THE COMPARISON IS NOT CONFOUNDED. Both modes keep all fourteen outputs, so the
        shapes, the parameter count and the RNG stream are the same and any difference in a curve is
        the representation rather than a different model."""
        director, vector = self._run("director"), self._run("vector")
        for name in ("approach", "axis_params", "offset_m", "width_m", "confidence"):
            with self.subTest(name):
                self.assertTrue(torch.equal(getattr(director, name), getattr(vector, name)))

    def test_the_prediction_carries_its_own_mode(self) -> None:
        """⛔ CARRIED, NOT PASSED ALONGSIDE. The loss, the metrics and the runtime decode all read
        the axis slots, and a mode threaded through three separate arguments is a mode three callers
        can disagree about. One would eventually supervise a director and score a vector, which is a
        defect that produces entirely plausible numbers."""
        for mode in AXIS_MODES:
            with self.subTest(mode):
                self.assertEqual(self._run(mode).axis_mode, mode)

    def test_the_default_is_still_the_measured_one(self) -> None:
        """Twenty-five arms were measured on `director`. Changing this default silently would make
        every one of their numbers a comparison against a different model."""
        self.assertEqual(SlotHeadConfig().axis_mode, "director")
        self.assertEqual(AXIS_MODES[0], "director")

    def test_an_unknown_mode_refuses_by_name(self) -> None:
        with self.assertRaises(ValueError) as caught:
            SlotGraspHead(16, SlotHeadConfig(axis_mode="whatever_sounds_plausible"))
        self.assertIn("axis_mode", str(caught.exception))
        with self.assertRaises(ValueError):
            axis_loss(torch.randn(4, 6), _axes(4), mode="nonsense")
        with self.assertRaises(ValueError):
            axis_decode(torch.randn(4, 6), mode="nonsense")


class LearnabilityTests(unittest.TestCase):

    def test_both_modes_fit_a_direct_target_and_NEITHER_dominates(self) -> None:
        """⛔⛔ THIS TEST WAS WRITTEN TO CONFIRM A HYPOTHESIS AND IT REFUTED IT. Kept in that shape,
        because the refutation is the useful part.

        The hypothesis was that the director representation is intrinsically slower to learn than a
        plain 3-vector, inferred from a corpus run where the APPROACH head reached 1.70 degrees and
        the AXIS head 31.32 on what I called targets of identical difficulty. MEASURED here, on a
        direct target with both modes given the same budget and initialisation, the ordering is the
        other way round: director 0.49 degrees against vector 1.17.

        So the corpus comparison was CONFOUNDED, and the confound is in the targets rather than in
        the representations. The approach control's target is `-normal`, a linear function of an
        input channel. The axis control's is `normalise(cross(normal, z))`, which is not. Comparing
        them measured the two things at once and attributed the whole gap to one of them.

        What this test can honestly assert is that neither representation is BROKEN: both drive a
        small head to within a couple of degrees on a target it can express. The representation
        question is settled by an arm that holds the target fixed and moves only the mode, which is
        what `--axis-mode` exists for.
        """
        errors = {}
        for mode in AXIS_MODES:
            torch.manual_seed(4)
            net = torch.nn.Sequential(torch.nn.Linear(3, 64), torch.nn.ReLU(),
                                      torch.nn.Linear(64, 6))
            optimiser = torch.optim.Adam(net.parameters(), lr=0.01)
            torch.manual_seed(5)
            target = torch.nn.functional.normalize(torch.randn(256, 3), dim=-1)
            for _ in range(300):
                optimiser.zero_grad()
                axis_loss(net(target), target, mode=mode).backward()
                optimiser.step()
            with torch.no_grad():
                decoded = axis_decode(net(target), mode=mode)
                cosine = (decoded * target).sum(-1).abs().clamp(max=1.0)
                errors[mode] = float(torch.rad2deg(torch.arccos(cosine)).mean())
        for mode, error in errors.items():
            with self.subTest(mode):
                self.assertLess(error, 5.0,
                                f"{mode} could not fit a target it can express: {errors}")


if __name__ == "__main__":
    unittest.main()
