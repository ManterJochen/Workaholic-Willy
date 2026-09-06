"""The sample contract, checked against the failures that have actually cost this project time.

⭐ THE TEST THAT MATTERS MOST IS `test_our_own_producer_satisfies_its_own_contract`. A contract
written from a reading of the code, rather than from its output, describes what somebody believed the
producer did. If our own generator fails the contract then the contract is wrong, and an importer
written against it would be built to match a fiction.

Every other test here pins one failure mode. They are deliberately about MEANING and not about types:
a shape check catches a transposed array, and none of the four frame defects this repository has
shipped was a transposed array.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.deep.corpus.sample import SampleSpec, build_sample
from src.robot.grasping.deep.foreign.contract import (
    SAMPLE_KEYS, describe_sample, validate_sample)
from tests.test_deep_dataset import _scene


def _good() -> dict:
    """A tiny sample that satisfies the contract, built by hand so a test can break one thing."""
    points = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]],
                      dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0]] * 4, dtype=np.float32)
    return {
        "points_m": points,
        "features": normals,
        "supervise": np.array([True, True, False, False]),
        "set_pair_point": np.array([0, 1], dtype=np.int32),
        "set_pair_grasp": np.array([0, 0], dtype=np.int32),
        "set_grasp_position_m": np.array([[0.025, 0.0, 0.0]], dtype=np.float32),
        "set_grasp_approach": np.array([[0.0, 0.0, -1.0]], dtype=np.float32),
        "set_grasp_axis": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        "set_grasp_width_m": np.array([0.05], dtype=np.float32),
    }


def _problems(sample: dict) -> dict[str, str]:
    return {p.key: p.problem for p in validate_sample(sample)}


class ContractTests(unittest.TestCase):

    def test_a_well_formed_sample_passes(self) -> None:
        self.assertEqual(validate_sample(_good()), [])

    def test_our_own_producer_satisfies_its_own_contract(self) -> None:
        """⭐ THE ANCHOR. If this fails, the contract is wrong and not the producer."""
        sample = build_sample(_scene(), np.random.default_rng(4),
                              SampleSpec(grasp_set=True, points=512), target_instance=None)
        fatal = [p for p in validate_sample(sample) if p.fatal]
        self.assertEqual(fatal, [], f"our own generator breaks the contract: {fatal}")

    def test_every_documented_key_is_required(self) -> None:
        for key in SAMPLE_KEYS:
            with self.subTest(key):
                sample = _good()
                del sample[key]
                self.assertEqual(_problems(sample).get(key), "missing")


class UnitTests(unittest.TestCase):
    """⛔ The class of defect that passes every other check. A corpus in the wrong unit is
    geometrically self-consistent and a network trains on it happily."""

    def test_a_cloud_in_millimetres_is_caught(self) -> None:
        sample = _good()
        sample["points_m"] = np.asarray(sample["points_m"]) * 1000.0
        self.assertEqual(_problems(sample).get("points_m"), "probably not metres")

    def test_a_width_in_millimetres_is_caught(self) -> None:
        sample = _good()
        sample["set_grasp_width_m"] = np.array([50.0], dtype=np.float32)
        self.assertEqual(_problems(sample).get("set_grasp_width_m"), "probably not metres")

    def test_a_non_positive_opening_is_caught(self) -> None:
        sample = _good()
        sample["set_grasp_width_m"] = np.array([0.0], dtype=np.float32)
        self.assertEqual(_problems(sample).get("set_grasp_width_m"), "non-positive opening")


class ConventionTests(unittest.TestCase):

    def test_directions_must_be_unit_length(self) -> None:
        sample = _good()
        sample["set_grasp_approach"] = np.array([[0.0, 0.0, -2.0]], dtype=np.float32)
        self.assertEqual(_problems(sample).get("set_grasp_approach"), "not unit length")

    def test_a_direction_of_zero_length_is_named_separately(self) -> None:
        sample = _good()
        sample["set_grasp_axis"] = np.zeros((1, 3), dtype=np.float32)
        self.assertEqual(_problems(sample).get("set_grasp_axis"), "zero-length direction")

    def test_a_swapped_approach_and_closing_axis_is_caught(self) -> None:
        """The jaws close ACROSS the approach. A dataset that stores two columns of one rotation
        without saying which is which fails here the moment the wrong pair is picked."""
        sample = _good()
        sample["set_grasp_axis"] = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        self.assertEqual(_problems(sample).get("set_grasp_axis"),
                         "not perpendicular to the approach")


class FrameTests(unittest.TestCase):
    """⛔⛔ The one that shipped. A grasp table in a different frame from its cloud is off by a
    constant per scene, which looks like a bias a head could almost learn around."""

    def test_a_grasp_table_in_another_frame_is_caught(self) -> None:
        sample = _good()
        sample["set_grasp_position_m"] = np.asarray(sample["set_grasp_position_m"]) + 1.0
        self.assertEqual(_problems(sample).get("set_grasp_position_m"),
                         "probably a different frame")

    def test_a_small_displacement_is_reported_without_refusing(self) -> None:
        """Suspicion is reported and does not refuse, because a validator that refuses on suspicion
        gets switched off and then catches nothing at all."""
        sample = _good()
        sample["set_grasp_position_m"] = np.asarray(sample["set_grasp_position_m"]) + 0.14
        found = [p for p in validate_sample(sample) if p.key == "set_grasp_position_m"]
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0].fatal)


class IndexTests(unittest.TestCase):

    def test_a_pair_pointing_past_the_cloud_is_caught(self) -> None:
        sample = _good()
        sample["set_pair_point"] = np.array([0, 99], dtype=np.int32)
        self.assertEqual(_problems(sample).get("set_pair_point"), "index out of range")

    def test_a_pair_pointing_past_the_grasp_table_is_caught(self) -> None:
        sample = _good()
        sample["set_pair_grasp"] = np.array([0, 7], dtype=np.int32)
        self.assertEqual(_problems(sample).get("set_pair_grasp"), "index out of range")

    def test_pairs_on_unsupervised_points_are_caught(self) -> None:
        """The loss drops these and the metric may not, and then the two disagree about the
        denominator. That exact disagreement produced a metric which could not separate two
        unrelated random models, and it took six retractions to find."""
        sample = _good()
        sample["set_pair_point"] = np.array([0, 2], dtype=np.int32)
        self.assertEqual(_problems(sample).get("set_pair_point"), "pairs on unsupervised points")

    def test_pairs_without_a_grasp_table_are_caught(self) -> None:
        sample = _good()
        sample["set_grasp_position_m"] = np.zeros((0, 3), dtype=np.float32)
        sample["set_grasp_approach"] = np.zeros((0, 3), dtype=np.float32)
        sample["set_grasp_axis"] = np.zeros((0, 3), dtype=np.float32)
        sample["set_grasp_width_m"] = np.zeros(0, dtype=np.float32)
        self.assertEqual(_problems(sample).get("set_pair_grasp"), "pairs without a grasp table")


class DescribeTests(unittest.TestCase):

    def test_it_reports_the_number_that_decides_whether_slots_are_worth_anything(self) -> None:
        """⭐ Labels per point. An imported corpus with a mean of 1.0 has already collapsed its own
        multimodality, and no set head could recover what the publisher threw away."""
        sample = _good()
        sample["set_pair_point"] = np.array([0, 0, 1], dtype=np.int32)
        sample["set_pair_grasp"] = np.array([0, 0, 0], dtype=np.int32)
        described = describe_sample(sample)
        self.assertEqual(described["labels_per_point"]["max"], 2)
        self.assertAlmostEqual(described["labels_per_point"]["share_multi"], 0.5)

    def test_it_survives_a_sample_with_no_grasps(self) -> None:
        sample = _good()
        for key in ("set_pair_point", "set_pair_grasp"):
            sample[key] = np.zeros(0, dtype=np.int32)
        for key, shape in (("set_grasp_position_m", (0, 3)), ("set_grasp_approach", (0, 3)),
                           ("set_grasp_axis", (0, 3)), ("set_grasp_width_m", (0,))):
            sample[key] = np.zeros(shape, dtype=np.float32)
        described = describe_sample(sample)
        self.assertEqual(described["grasps"], 0)
        self.assertNotIn("labels_per_point", described)


if __name__ == "__main__":
    unittest.main()
