"""The SET of grasps a point admits, instead of the one that happens to be nearest.

⚠ WHAT THIS GUARDS is not a shape. `_assign_grasps` keeps the nearest contact per point, which is an
implementation detail standing in for a design decision nobody made, and everything downstream
inherits it: a head regressed against one label per point lands between the modes on the approach,
the offset and the width alike.

MEASURED on v5 with `datagen/eval/label_set_shape.py`, 80 scenes and 56,272 supervised
points: **86.7 %** of them reach more than one label, **75.2 %** admit more than one approach further
than 15 degrees apart, and the median point holds **4 distinct approaches**. The old figure this arc
quoted, 74.7 %, came from v3; the corpus changed by an order of magnitude and the number did not.

⛔ The first thing every test here checks is that the flag OFF changes nothing. This lands in the
sampler every measured arm shares, and a corpus that shifted under it would silently invalidate
twenty-four of them.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.deep.corpus.sample import SampleSpec, build_sample
from tests.test_deep_dataset import _scene

_SET_KEYS = ("set_pair_point", "set_pair_grasp", "set_grasp_position_m",
             "set_grasp_approach", "set_grasp_axis", "set_grasp_width_m")


def _sample(**spec_kwargs: object) -> dict:
    return build_sample(_scene(), np.random.default_rng(3), SampleSpec(**spec_kwargs),
                        target_instance=None)


class ByteIdentityTests(unittest.TestCase):
    """⛔ Default off, and off must be indistinguishable from before the flag existed."""

    def test_the_arrays_are_absent_by_default(self) -> None:
        sample = _sample()
        for key in _SET_KEYS:
            self.assertNotIn(key, sample)

    def test_turning_it_on_leaves_every_existing_array_untouched(self) -> None:
        """The one that matters. The pair pass now runs under an `or`, so it reaches cases it did not
        before, and it must not be able to touch anything the old path produced."""
        off = _sample()
        on = _sample(grasp_set=True)
        for key, value in off.items():
            with self.subTest(key):
                self.assertIn(key, on)
                if isinstance(value, np.ndarray):
                    self.assertTrue(np.array_equal(value, on[key], equal_nan=True))
                else:
                    self.assertEqual(value, on[key])

    def test_it_composes_with_the_multi_hot_path(self) -> None:
        """Both flags read the same second radius pass. Turning one on must not change the other."""
        multi = _sample(approach_multilabel=True)
        both = _sample(approach_multilabel=True, grasp_set=True)
        self.assertTrue(np.array_equal(multi["approach_set"], both["approach_set"]))


class ContentTests(unittest.TestCase):

    def test_every_pair_points_at_a_real_point_and_a_real_grasp(self) -> None:
        sample = _sample(grasp_set=True)
        points = len(sample["points_mm"]) if "points_mm" in sample else len(sample["depth_m"])
        grasps = len(sample["set_grasp_width_m"])
        self.assertTrue((sample["set_pair_point"] >= 0).all())
        self.assertTrue((sample["set_pair_point"] < points).all())
        self.assertTrue((sample["set_pair_grasp"] >= 0).all())
        self.assertTrue((sample["set_pair_grasp"] < grasps).all())

    def test_only_supervised_points_appear(self) -> None:
        """Dropped here rather than in the loss, so the two paths cannot disagree about which rows
        carry a target."""
        sample = _sample(grasp_set=True)
        self.assertTrue(sample["supervise"][sample["set_pair_point"]].all())

    def test_the_table_is_in_metres_and_the_directions_are_unit(self) -> None:
        sample = _sample(grasp_set=True)
        self.assertLess(float(np.abs(sample["set_grasp_position_m"]).max()), 10.0)
        for key in ("set_grasp_approach", "set_grasp_axis"):
            with self.subTest(key):
                norms = np.linalg.norm(sample[key], axis=-1)
                self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

    def test_the_approach_and_the_axis_are_perpendicular(self) -> None:
        sample = _sample(grasp_set=True)
        dot = np.einsum("ij,ij->i", sample["set_grasp_approach"], sample["set_grasp_axis"])
        self.assertLess(float(np.abs(dot).max()), 1e-5)

    def test_the_set_contains_the_label_the_single_path_chose(self) -> None:
        """⭐ THE TWO PATHS MUST AGREE ABOUT WHAT EXISTS and differ only in how much of it they keep.
        If the nearest-contact label were absent from the set, the set would be a different question
        rather than a superset of the old one."""
        sample = _sample(grasp_set=True)
        supervised = np.flatnonzero(sample["supervise"] & (sample["approach_bin"] >= 0))
        self.assertTrue(supervised.size, "the fixture produced no supervised point")
        reached = set(sample["set_pair_point"].tolist())
        self.assertTrue(set(supervised.tolist()) <= reached,
                        "a point with a single-path label reaches no grasp in the set")


class AugmentationTests(unittest.TestCase):
    """⛔ The table is emitted AFTER the augmentation, and mixing the two frames is invisible."""

    def test_the_offset_survives_a_rotation(self) -> None:
        """A rotated cloud with an unrotated grasp table produces a target wrong by exactly the
        augmentation, which looks like a hard problem rather than a bug. Checked by the one quantity
        a z-rotation about the scene centre cannot change: the LENGTH of the seed-to-centre offset."""
        scene = _scene()
        plain = build_sample(scene, np.random.default_rng(11),
                             SampleSpec(grasp_set=True, rotate_z=False), target_instance=None)
        rotated = build_sample(scene, np.random.default_rng(11),
                               SampleSpec(grasp_set=True, rotate_z=True), target_instance=None)
        for sample in (plain, rotated):
            self.assertTrue(sample["set_pair_point"].size)

        def offsets(sample: dict) -> np.ndarray:
            centres = sample["set_grasp_position_m"][sample["set_pair_grasp"]]
            seeds = sample["points_m"][sample["set_pair_point"]]
            return np.linalg.norm(centres - seeds, axis=-1)

        self.assertTrue(np.allclose(np.sort(offsets(plain)), np.sort(offsets(rotated)), atol=1e-4))

    def test_the_position_is_in_the_same_frame_as_the_points(self) -> None:
        """⛔⛔ THE FOURTH FRAME DEFECT IN THIS REPOSITORY, and the reason this test exists.

        `points_m` is the cloud with its xy mean removed and the support height taken off z. The
        grasp table arrives in the raw scene frame. Emitted unshifted, every seed-to-centre offset is
        wrong by a per-scene constant, which is exactly the shape a head can almost learn around and
        nobody notices.

        Checked against the geometry rather than against the shift: a labelled grasp centre sits
        close to the points its own contacts reach, so an offset of tens of centimetres means the two
        arrays are not in the same frame.
        """
        sample = _sample(grasp_set=True)
        centres = sample["set_grasp_position_m"][sample["set_pair_grasp"]]
        seeds = sample["points_m"][sample["set_pair_point"]]
        offsets = np.linalg.norm(centres - seeds, axis=-1)
        self.assertLess(float(offsets.max()), 0.10,
                        "the grasp table and the point cloud are in different frames")


if __name__ == "__main__":
    unittest.main()
