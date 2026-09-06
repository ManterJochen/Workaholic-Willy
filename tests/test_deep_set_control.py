"""A target the stack must be able to learn, so a flat curve can be attributed.

⛔ WHY THIS EXISTS. Two explanations fit the architecture arm's flat curve equally well: the
architecture cannot learn a 6-DoF grasp head, or the corpus cannot teach one. Nothing measured
separates them, so every next step -- more assets, a bigger backbone, a different head -- is a bet on
one of the two with no evidence for either.

The control deletes every difficulty the real target has and keeps everything else. What it must NOT
do is also make the supervision denser or the seeds easier, because then it would be answering two
questions at once and could attribute neither. These tests pin that boundary.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.deep.corpus.sample import SampleSpec, build_sample
from src.robot.grasping.deep.train.synthetic_control import CONTROL_LEVELS, synthetic_labels
from tests.test_deep_dataset import _scene


def _sample() -> dict:
    return build_sample(_scene(), np.random.default_rng(2),
                        SampleSpec(grasp_set=True, points=512), target_instance=None)


class TargetTests(unittest.TestCase):

    def test_the_approach_is_the_inward_normal_exactly(self) -> None:
        """⭐ THE POINT OF THE WHOLE CONTROL. The answer is a linear function of an input channel at
        the same point, so a head that cannot reach it is broken independently of any data question.
        """
        sample = synthetic_labels(_sample())
        points = sample["set_pair_point"]
        normals = np.asarray(sample["features"])[:, :3][points]
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
        self.assertTrue(np.allclose(sample["set_grasp_approach"], -normals, atol=1e-5))

    def test_there_is_exactly_one_label_per_supervised_point(self) -> None:
        """No multimodality, so nothing for a head to average. That is the difficulty being removed.
        """
        sample = synthetic_labels(_sample())
        points = sample["set_pair_point"]
        self.assertEqual(len(points), len(np.unique(points)))
        self.assertEqual(len(sample["set_pair_grasp"]), len(points))
        self.assertEqual(len(np.unique(sample["set_pair_grasp"])), len(points))

    def test_the_axis_is_perpendicular_to_the_approach(self) -> None:
        sample = synthetic_labels(_sample())
        dot = np.einsum("ij,ij->i", sample["set_grasp_approach"], sample["set_grasp_axis"])
        self.assertLess(float(np.abs(dot).max()), 1e-5)
        norms = np.linalg.norm(sample["set_grasp_axis"], axis=-1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

    def test_the_offset_is_a_fixed_step_along_the_approach(self) -> None:
        """One direction, no two-contact coin flip. The real corpus's offset sign is positive 44.3 %
        of the time, and removing that is most of what makes this control easy."""
        sample = synthetic_labels(_sample())
        offset = (sample["set_grasp_position_m"]
                  - np.asarray(sample["points_m"])[sample["set_pair_point"]])
        lengths = np.linalg.norm(offset, axis=-1)
        self.assertTrue(np.allclose(lengths, 0.025, atol=1e-5))
        along = np.einsum("ij,ij->i", offset / lengths[:, None], sample["set_grasp_approach"])
        self.assertTrue(np.allclose(along, 1.0, atol=1e-5))

    def test_the_perpendicular_axis_is_continuous(self) -> None:
        """⚠ A rule that branched on the largest component would jump between neighbouring points,
        putting a discontinuity into the control that the real target does not have."""
        from src.robot.grasping.deep.train.synthetic_control import _perpendicular

        base = np.array([[0.0, 0.0, 1.0]])
        nudged = np.array([[1e-4, 0.0, 1.0]])
        nudged = nudged / np.linalg.norm(nudged)
        first, second = _perpendicular(base), _perpendicular(nudged)
        self.assertGreater(float(np.abs((first * second).sum())), 0.99)

    def test_a_normal_aligned_with_the_reference_still_gets_an_axis(self) -> None:
        from src.robot.grasping.deep.train.synthetic_control import _perpendicular

        axis = _perpendicular(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]))
        self.assertFalse(bool(np.isnan(axis).any()))
        self.assertTrue(np.allclose(np.linalg.norm(axis, axis=-1), 1.0, atol=1e-6))


class BoundaryTests(unittest.TestCase):
    """⛔ The control replaces the ANSWERS and nothing else."""

    def test_the_supervised_set_is_unchanged(self) -> None:
        """A control that also made the supervision dense would be answering two questions at once.
        The seed mixture, the label density and the class balance must be what the real run sees."""
        plain = _sample()
        control = synthetic_labels(plain)
        self.assertTrue(np.array_equal(plain["supervise"], control["supervise"]))
        self.assertTrue(np.array_equal(plain["points_m"], control["points_m"]))
        self.assertTrue(np.array_equal(plain["features"], control["features"]))

    def test_the_caller_s_sample_is_not_modified(self) -> None:
        plain = _sample()
        before = np.asarray(plain["set_grasp_approach"]).copy()
        synthetic_labels(plain)
        self.assertTrue(np.array_equal(before, plain["set_grasp_approach"]))

    def test_every_level_is_reachable_and_an_unknown_one_refuses(self) -> None:
        for level in CONTROL_LEVELS:
            with self.subTest(level):
                self.assertIn("set_grasp_approach", synthetic_labels(_sample(), level))
        with self.assertRaises(ValueError):
            synthetic_labels(_sample(), "whatever_sounds_plausible")

    def test_the_local_level_needs_the_neighbourhood_and_the_normal_level_does_not(self) -> None:
        """If `normal` passes and `local` fails, the failure is in the ENCODER rather than the head.
        The two levels have to differ, or that distinction is unavailable."""
        sample = _sample()
        plain = synthetic_labels(sample, "normal")
        local = synthetic_labels(sample, "local")
        self.assertEqual(len(np.unique(plain["set_grasp_width_m"])), 1)
        self.assertFalse(np.allclose(plain["set_grasp_width_m"], local["set_grasp_width_m"]))

    def test_a_sample_with_nothing_supervised_carries_NO_labels(self) -> None:
        """⛔ NOT the original ones. Handing back the corpus's real grasp table would make a batch
        train partly on the control and partly on the thing the control exists to replace, and
        nothing anywhere would say so. Caught by this test on the first run."""
        sample = _sample()
        self.assertGreater(len(sample["set_pair_point"]), 0, "the fixture had no real labels")
        sample["supervise"] = np.zeros_like(np.asarray(sample["supervise"]))
        out = synthetic_labels(sample)
        self.assertEqual(len(out["set_pair_point"]), 0)
        self.assertEqual(len(out["set_grasp_approach"]), 0)
        self.assertEqual(out["set_grasp_position_m"].shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
