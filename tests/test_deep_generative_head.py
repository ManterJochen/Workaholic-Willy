"""Stage 4b: the head that MODELS the multimodality instead of quotienting it out.

⭐⭐ THE MEASUREMENT IT EXISTS FOR. On v5 over 56,272 supervised points, 86.7 % reach more than one
label and 75.2 % admit more than one approach further than fifteen degrees apart, with a median point
holding four distinct approaches and a ninetieth percentile of fifteen.

The deterministic head answers that with K slots and a one-to-one matching, which is bounded: K slots
bind at most K labels, so its coverage ceiling at K = 4 is 0.385 rather than 1.0. That ceiling is a
property of K against the data, not of the model. A generative head has no such bound.

⚠ NOTHING HERE IS A RESULT. These tests establish that the shape works: it runs, its samples differ,
its two noise schedules are both read, and it refuses inputs it cannot interpret. Whether it beats the
deterministic head is an arm on identical seeds, crops and folds, and no arm has run.
"""

from __future__ import annotations

import unittest

import torch

from src.robot.grasping.deep.net.generative_head import (
    POSE_DIM, GenerativeGraspHead, GenerativeHeadConfig)
from src.robot.grasping.deep.net.gripper import GRIPPER_VECTOR_DIM


def _head(**kwargs) -> GenerativeGraspHead:
    torch.manual_seed(0)
    base = dict(width=64, depth=2, steps=8, samples=4, gripper_width=8, time_width=16)
    base.update(kwargs)
    return GenerativeGraspHead(32, GenerativeHeadConfig(**base))


def _inputs(count: int = 8):
    torch.manual_seed(1)
    return torch.randn(count, 32), torch.randn(count, GRIPPER_VECTOR_DIM)


class ShapeTests(unittest.TestCase):

    def test_it_predicts_noise_shaped_like_the_pose(self) -> None:
        head = _head()
        features, gripper = _inputs()
        level = torch.full((8,), 3.0)
        out = head(features, gripper, torch.randn(8, POSE_DIM), level, level)
        self.assertEqual(tuple(out.shape), (8, POSE_DIM))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_the_pose_carries_ten_numbers(self) -> None:
        """An approach (3), a closing axis (3), an offset (3) and a width (1). The axis travels as a
        plain vector rather than a director on purpose: the director exists to make a LOSS
        sign-invariant, and this head predicts noise. Sign invariance falls out of the sampling."""
        self.assertEqual(POSE_DIM, 10)


class SamplingTests(unittest.TestCase):

    def test_the_samples_DIFFER(self) -> None:
        """⭐ THE WHOLE POINT. A generative head returning one pose four times models nothing, and
        would pass every shape test in this file."""
        head = _head()
        features, gripper = _inputs()
        drawn = head.sample(features, gripper, count=4)
        self.assertEqual(tuple(drawn.shape), (8, 4, POSE_DIM))
        spread = float((drawn - drawn.mean(dim=1, keepdim=True)).abs().mean())
        self.assertGreater(spread, 1e-3, f"the four samples per seed are identical: {spread}")

    def test_the_count_is_a_BUDGET_and_not_an_architectural_bound(self) -> None:
        """The deterministic head emits exactly K. This is the difference being tested, so it has to
        be visible: the same head answers with two, four or sixteen."""
        head = _head()
        features, gripper = _inputs(4)
        for count in (1, 2, 16):
            with self.subTest(count):
                self.assertEqual(tuple(head.sample(features, gripper, count=count).shape),
                                 (4, count, POSE_DIM))

    def test_sampling_is_finite_and_reproducible_under_a_generator(self) -> None:
        head = _head()
        features, gripper = _inputs()
        first = head.sample(features, gripper, count=3,
                            generator=torch.Generator().manual_seed(7))
        second = head.sample(features, gripper, count=3,
                             generator=torch.Generator().manual_seed(7))
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertTrue(torch.equal(first, second))


class ConditioningTests(unittest.TestCase):

    def test_BOTH_noise_levels_are_read(self) -> None:
        """⛔ THE TRANSLATION AND THE ROTATION ARE NOISED SEPARATELY, and a head that read only one of
        the two levels would look correct and denoise the other at the wrong rate. Outside work
        measured separate schedules as better than one; our own numbers make it obvious, since an
        offset is metres and an axis is a direction."""
        head = _head()
        features, gripper = _inputs()
        pose = torch.randn(8, POSE_DIM)
        low = torch.full((8,), 1.0)
        high = torch.full((8,), 7.0)
        base = head(features, gripper, pose, low, low)
        moved_translation = head(features, gripper, pose, high, low)
        moved_rotation = head(features, gripper, pose, low, high)
        self.assertGreater(float((moved_translation - base).abs().max()), 1e-5,
                           "the translation noise level is ignored")
        self.assertGreater(float((moved_rotation - base).abs().max()), 1e-5,
                           "the rotation noise level is ignored")

    def test_the_gripper_reaches_the_output(self) -> None:
        """The same seam the deterministic head has, checked the same way. A live seam is wiring and
        not a capability; see `test_deep_gripper_seam_is_live` for why that distinction matters."""
        from src.robot.grasping.deep.net.gripper import gripper_vector

        head = _head()
        features, _ = _inputs()
        pose = torch.randn(8, POSE_DIM)
        level = torch.full((8,), 3.0)
        narrow = head(features, gripper_vector("2f85").expand(8, GRIPPER_VECTOR_DIM), pose,
                      level, level)
        wide = head(features, gripper_vector("wide_140").expand(8, GRIPPER_VECTOR_DIM), pose,
                    level, level)
        self.assertGreater(float((narrow - wide).abs().max()), 1e-6)


class RefusalTests(unittest.TestCase):

    def test_a_wrong_feature_width_refuses_by_name(self) -> None:
        head = _head()
        with self.assertRaises(ValueError) as caught:
            head(torch.randn(4, 5), torch.randn(4, GRIPPER_VECTOR_DIM),
                 torch.randn(4, POSE_DIM), torch.zeros(4), torch.zeros(4))
        self.assertIn("features must be", str(caught.exception))

    def test_a_BROADCAST_gripper_refuses(self) -> None:
        """⚠ The same refusal the deterministic head makes, and for the same reason: broadcasting a
        bare (14,) would be convenient and would hide the case the seam exists to expose, an input
        that should vary per sample and never does."""
        head = _head()
        features, _ = _inputs()
        with self.assertRaises(ValueError):
            head(features, torch.randn(GRIPPER_VECTOR_DIM), torch.randn(8, POSE_DIM),
                 torch.zeros(8), torch.zeros(8))

    def test_a_pose_of_the_wrong_width_refuses(self) -> None:
        head = _head()
        features, gripper = _inputs()
        with self.assertRaises(ValueError):
            head(features, gripper, torch.randn(8, POSE_DIM + 1), torch.zeros(8), torch.zeros(8))

    def test_zero_steps_refuses_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            GenerativeGraspHead(32, GenerativeHeadConfig(steps=0))


class HonestyTests(unittest.TestCase):

    def test_the_module_says_it_has_measured_nothing(self) -> None:
        """⚠ Every number in its docstring is about the DATA or about the deterministic head. A
        reader must not take this file for evidence that a generative head works here."""
        from pathlib import Path

        text = Path("src/robot/grasping/deep/net/generative_head.py").read_text(
            encoding="utf-8")
        self.assertIn("NOTHING HERE IS MEASURED YET", text)
        self.assertIn("IT IS AN ARM, NOT A REPLACEMENT", text)


if __name__ == "__main__":
    unittest.main()
