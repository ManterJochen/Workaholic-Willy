"""The differential test the plan's risk table asks for, and could not be written until today.

⭐⭐ WHY IT COULD NOT BE WRITTEN BEFORE. Our own corpus carries ONE gripper, so the fourteen
conditioning numbers were the same on every sample. `test_deep_gripper_vector` says so in its own
words: with a constant input the encoder still trains, nothing distinguishes the samples, the
embedding is a learned constant, and "the seam is wiring without a capability". That test PINS the
constancy so a future differential test has something to be different from. This is that test.

What made it possible is the foreign import, which stamps `wide_140` on every scene it writes against
our corpus's `2f85`. A run over both now sees two apertures, and the risk row in section 7 of the
architecture plan, "the gripper seam is inert: a differential test shows no output change", finally
has an instrument.

⛔ WHAT A LIVE SEAM MEANS AND WHAT IT DOES NOT. These tests establish that the gripper input REACHES
the output and moves it. They do not establish that the model USES it correctly, that it generalises
to a gripper it never saw, or that a width prediction respects an aperture. Those are training
results and this is a wiring check. Confusing the two is precisely the inert-switch mistake in its
mirror image: declaring a capability because a signal is present.
"""

from __future__ import annotations

import unittest

import torch

from src.robot.grasping.deep.net.gripper import (
    GRIPPER_VECTOR_DIM, JAW_GEOMETRY, gripper_vector)
from src.robot.grasping.deep.net.slot_head import SlotGraspHead, SlotHeadConfig
from src.robot.grasping.deep.net.set_generator import SetGenerator, SetGeneratorConfig


def _head(seed: int = 0) -> SlotGraspHead:
    torch.manual_seed(seed)
    return SlotGraspHead(16, SlotHeadConfig(slots=4, width=64, gripper_width=16, query_width=8))


def _features(count: int = 32, seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(count, 16)


class DifferentialTests(unittest.TestCase):
    """⭐ THE ROW FROM THE PLAN'S RISK TABLE: same scene, different hand, different answer."""

    def test_two_apertures_give_two_different_heads_output(self) -> None:
        head = _head().eval()
        features = _features()
        with torch.no_grad():
            narrow = head(features, gripper_vector("2f85").expand(len(features), GRIPPER_VECTOR_DIM))
            wide = head(features, gripper_vector("wide_140").expand(len(features),
                                                                    GRIPPER_VECTOR_DIM))
        for name in ("approach", "axis_params", "offset_m", "width_m", "confidence"):
            with self.subTest(name):
                moved = float((getattr(narrow, name) - getattr(wide, name)).abs().max())
                self.assertGreater(moved, 1e-5,
                                   f"{name} is identical for a 55 mm and a 140 mm hand; the "
                                   f"conditioning seam is inert")

    def test_every_profile_pair_separates(self) -> None:
        """Not just the extremes. Two profiles that share an aperture must still differ, because the
        vector carries pad geometry and friction as well as the opening."""
        head = _head().eval()
        features = _features(16)
        outputs = {}
        with torch.no_grad():
            for name in JAW_GEOMETRY:
                vector = gripper_vector(name).expand(len(features), GRIPPER_VECTOR_DIM)
                outputs[name] = head(features, vector).width_m
        names = sorted(outputs)
        for first in range(len(names)):
            for second in range(first + 1, len(names)):
                a, b = names[first], names[second]
                with self.subTest(f"{a} vs {b}"):
                    self.assertGreater(float((outputs[a] - outputs[b]).abs().max()), 1e-6,
                                       f"{a} and {b} are indistinguishable to the head")

    def test_the_whole_generator_separates_them_too(self) -> None:
        """The head is where the vector enters, but a seam that dies in the composition is still
        dead. Checked end to end on the assembled generator rather than on the head alone."""
        torch.manual_seed(0)
        net = SetGenerator(SetGeneratorConfig(
            head=SlotHeadConfig(slots=2, width=32, gripper_width=8, query_width=8),
            graspability_width=8)).eval()
        torch.manual_seed(2)
        points = torch.randn(1, 512, 3) * 0.1
        features = torch.randn(1, 512, net.backbone.config.in_features)
        seeds = 8
        batch_index = torch.zeros(seeds, dtype=torch.long)
        point_index = torch.arange(seeds)
        with torch.no_grad():
            encoded = net.encode(points, features)
            narrow = net.propose(encoded, batch_index, point_index,
                                 gripper_vector("2f85").expand(seeds, GRIPPER_VECTOR_DIM))
            wide = net.propose(encoded, batch_index, point_index,
                               gripper_vector("wide_140").expand(seeds, GRIPPER_VECTOR_DIM))
        self.assertGreater(float((narrow.width_m - wide.width_m).abs().max()), 1e-6,
                           "the composed generator ignores its gripper input")


class HonestyTests(unittest.TestCase):
    """⛔ A live seam is not a capability, and this file must not be read as claiming one."""

    def test_an_untrained_head_already_separates_them(self) -> None:
        """⚠ THE POINT THAT KEEPS THIS TEST HONEST. The differential above passes on a head with
        RANDOM weights, because a different input through a random matrix gives a different output.
        So it proves the wiring and nothing about learning. A reader who takes it for evidence that
        the model adapts to a gripper has made the inert-switch mistake in its mirror image."""
        head = _head(seed=7).eval()
        features = _features(8, seed=9)
        with torch.no_grad():
            first = head(features, gripper_vector("2f85").expand(8, GRIPPER_VECTOR_DIM)).width_m
            second = head(features, gripper_vector("narrow_55").expand(8,
                                                                       GRIPPER_VECTOR_DIM)).width_m
        self.assertGreater(float((first - second).abs().max()), 1e-6)

    def test_the_corpus_side_of_the_seam_is_what_was_missing(self) -> None:
        """The wiring was never the problem: our corpus carried one gripper, so every sample fed the
        same fourteen numbers and the embedding was a learned constant. The foreign importer stamps
        a different profile, which is what gives a RUN two values to tell apart."""
        from src.robot.grasping.deep.foreign.grasp_anything import DEFAULT_GRIPPER

        self.assertIn(DEFAULT_GRIPPER, JAW_GEOMETRY)
        self.assertNotEqual(DEFAULT_GRIPPER, "2f85")
        self.assertNotEqual(float(JAW_GEOMETRY[DEFAULT_GRIPPER]["aperture_mm"]),
                            float(JAW_GEOMETRY["2f85"]["aperture_mm"]))


if __name__ == "__main__":
    unittest.main()
