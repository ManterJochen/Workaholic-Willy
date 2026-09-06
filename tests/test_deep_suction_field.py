"""Suction reaches the WHERE stage, and nothing else.

⭐⭐ **THE MEASUREMENT.** The architecture plan names "68.5 % of training units carry zero labelled
points" as the thing starving the heads, and that number is measured on JAW labels alone. Run through
`build_sample` on a corpus extracted with `--kinds both`, over 181 units of v5/s0:

    units with NO jaw positive .............. 140   77.3 %
      of those, RESCUED by suction ..........  86   61.4 % of the empty ones
    units with NO label of either kind ......  54   29.8 %

    positive POINTS, jaw only ............... 19,397
    positive POINTS, jaw or suction ......... 30,153   (+55.5 %)

⛔ **AND IT MUST NOT REACH THE POSE HEADS.** A cup has no closing axis and no opening; the corpus
stores exactly that, `closing_axis [0,0,0]` and `width_mm 0.0`, and our own sample contract refuses
both by name. A jaw head trained on a suction label would be fitting a zero axis. What changes is the
WHERE stage, which today learns from jaw labels alone and is therefore taught to call a suction-only
object EMPTY, which is false for any cell carrying both end effectors.

⚠ ABSENT unless the corpus was built with `--kinds both`, so every existing corpus and every existing
reader is byte-identical.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.deep.corpus.sample import build_sample
from src.robot.grasping.deep.train.trainer import SetTrainingPlan


def _scene(*, suction: np.ndarray | None = None, jaw_contacts: bool = True) -> dict:
    """A minimal corpus-shaped scene: one object as a small grid of points."""
    grid = np.stack(np.meshgrid(np.linspace(0, 60, 7), np.linspace(0, 60, 7),
                                np.linspace(0, 30, 3), indexing="ij"), axis=-1).reshape(-1, 3)
    count = len(grid)
    scene: dict = {
        "scene_id": "t", "points_mm": grid,
        "normals": np.tile([0.0, 0.0, 1.0], (count, 1)),
        "normal_valid": np.ones(count, dtype=bool),
        "view_count": np.ones(count, dtype=np.int16),
        "instance_id": np.zeros(count, dtype=np.int16),
        "object_instance": np.asarray([0], dtype=np.int16),
        "object_asset_id": np.asarray(["asset_a"], dtype="<U80"),
        "grasp_position_mm": np.asarray([[30.0, 30.0, 15.0]]) if jaw_contacts else np.zeros((0, 3)),
        "grasp_approach": np.asarray([[0.0, 0.0, -1.0]]) if jaw_contacts else np.zeros((0, 3)),
        "grasp_axis": np.asarray([[1.0, 0.0, 0.0]]) if jaw_contacts else np.zeros((0, 3)),
        "grasp_width_mm": np.asarray([40.0]) if jaw_contacts else np.zeros(0),
        "grasp_instance": np.asarray([0], dtype=np.int16) if jaw_contacts
        else np.zeros(0, dtype=np.int16),
        "contact_points_mm": np.asarray([[10.0, 30.0, 15.0], [50.0, 30.0, 15.0]])
        if jaw_contacts else np.zeros((0, 3)),
        "contact_grasp_index": np.asarray([0, 0], dtype=np.int32) if jaw_contacts
        else np.zeros(0, dtype=np.int32),
    }
    if suction is not None:
        scene["suction_position_mm"] = suction
        scene["suction_approach"] = np.tile([0.0, 0.0, -1.0], (len(suction), 1))
        scene["suction_instance"] = np.zeros(len(suction), dtype=np.int16)
    return scene


def _sample(scene: dict) -> dict:
    return build_sample(scene, np.random.default_rng(0), SetTrainingPlan().sample,
                        target_instance=0)


class FieldTests(unittest.TestCase):

    def test_a_suction_ONLY_object_is_no_longer_empty(self) -> None:
        """⛔ THE DEFECT. Without this the WHERE stage is taught that an object it could pick with a
        cup carries nothing worth attempting."""
        sample = _sample(_scene(suction=np.asarray([[30.0, 30.0, 30.0]]), jaw_contacts=False))
        self.assertEqual(float(np.asarray(sample["graspability"]).sum()), 0.0)
        self.assertGreater(float(np.asarray(sample["graspability_suction"]).sum()), 0.0)
        self.assertGreater(float(np.asarray(sample["graspable_any"]).sum()), 0.0)

    def test_the_JAW_field_is_untouched_by_suction(self) -> None:
        """⭐ Six readers already treat `graspability` as "somewhere a JAW touched", and silently
        widening it would change what all of them mean without any of them saying so."""
        without = _sample(_scene())
        with_suction = _sample(_scene(suction=np.asarray([[30.0, 30.0, 30.0]])))
        self.assertTrue(np.array_equal(np.asarray(without["graspability"]),
                                       np.asarray(with_suction["graspability"])))

    def test_the_union_is_the_MAXIMUM_of_the_two(self) -> None:
        sample = _sample(_scene(suction=np.asarray([[5.0, 5.0, 30.0]])))
        jaw = np.asarray(sample["graspability"])
        suction = np.asarray(sample["graspability_suction"])
        self.assertTrue(np.array_equal(np.asarray(sample["graspable_any"]),
                                       np.maximum(jaw, suction)))

    def test_a_corpus_WITHOUT_suction_is_byte_identical(self) -> None:
        """⚠ Every corpus built before `--kinds both` has no suction arrays, and must read exactly as
        it always did."""
        sample = _sample(_scene())
        self.assertEqual(float(np.asarray(sample["graspability_suction"]).sum()), 0.0)
        self.assertTrue(np.array_equal(np.asarray(sample["graspable_any"]),
                                       np.asarray(sample["graspability"])))

    def test_suction_on_ANOTHER_object_does_not_count(self) -> None:
        """⛔ The field is about the TARGET. A cup landing on a neighbour is not a reason to call this
        object graspable, and the same filter already applies to jaw contacts."""
        scene = _scene(suction=np.asarray([[30.0, 30.0, 30.0]]))
        scene["suction_instance"] = np.asarray([7], dtype=np.int16)
        sample = _sample(scene)
        self.assertEqual(float(np.asarray(sample["graspability_suction"]).sum()), 0.0)


class PoseHeadTests(unittest.TestCase):

    def test_suction_NEVER_reaches_the_pose_supervision(self) -> None:
        """⛔⛔ A cup has no closing axis and no opening; the corpus stores `[0,0,0]` and `0.0`, and
        our own sample contract refuses both by name. A jaw head trained on one would be fitting a
        zero axis."""
        without = _sample(_scene())
        with_suction = _sample(_scene(suction=np.asarray([[5.0, 5.0, 30.0]])))
        for key in ("set_pair_point", "set_pair_grasp", "set_grasp_axis", "set_grasp_width_m"):
            if key in without:
                with self.subTest(key):
                    self.assertTrue(np.array_equal(np.asarray(without[key]),
                                                   np.asarray(with_suction[key])),
                                    f"{key} changed when suction labels were added")

    def test_the_WHERE_stage_reads_the_suction_field(self) -> None:
        """The wiring, checked in the trainer rather than assumed from the sample."""
        from pathlib import Path

        source = Path("src/robot/grasping/deep/train/step.py").read_text(encoding="utf-8")
        self.assertIn('sample.get("graspability_suction")', source)
        self.assertIn("field_target", source)


if __name__ == "__main__":
    unittest.main()
