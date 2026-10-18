"""Which object a proposal is aimed at, and how far it is from actually being on one.

⛔⛔ MEASURED ON THE FIRST PROPOSAL RUN THAT REACHED A REFEREE: 128 of 320 proposals, 40 %, came back
as instance -1 or -2, and every one of them was refused downstream as "instance -1 is not in this
scene". A cloud carries the table and the bin walls with instance ids below zero, and the nearest
point to a proposal is very often one of them. So the referee judged 178 trials out of 320 and the
denominator of the plan's primary metric was chosen by a defect rather than by a decision.

⭐ AND THE DISTANCE IS THE OTHER HALF. A grasp that closed on an object and slipped is a grasp that
FAILED. A grasp 200 mm from every object never had the chance: the model aimed at nothing. A single
hold rate merges the two and answers neither, so the distance travels with every proposal. MEASURED on
the same run after the fix: 29.1 % sit more than 42.5 mm from any object, and only 21.6 % are within
10 mm of one.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.robot.grasping.deep.eval.propose import _OFF_OBJECT_MM, _nearest_instance


def _scene(points, instances) -> dict:
    return {"points_mm": np.asarray(points, dtype=np.float64),
            "instance_id": np.asarray(instances, dtype=np.int64)}


class InstanceTests(unittest.TestCase):

    def test_a_TABLE_point_is_never_the_answer(self) -> None:
        """⛔ THE DEFECT ITSELF. The table sits closer than the object and used to win."""
        scene = _scene([[0, 0, 0], [10, 0, 0]], [-1, 3])
        instance, distance = _nearest_instance(scene, np.array([1.0, 0.0, 0.0]))
        self.assertEqual(instance, 3)
        self.assertAlmostEqual(distance, 9.0)

    def test_the_nearest_OBJECT_wins_among_several(self) -> None:
        scene = _scene([[0, 0, 0], [100, 0, 0], [-1, 0, 0]], [1, 2, -2])
        self.assertEqual(_nearest_instance(scene, np.array([90.0, 0.0, 0.0]))[0], 2)

    def test_the_distance_is_to_the_OBJECT_and_not_to_the_nearest_point(self) -> None:
        """A proposal sitting on the table with an object 50 mm away is 50 mm from an object, not
        0 mm from a point. The number has to mean the thing the metric splits on."""
        scene = _scene([[0, 0, 0], [50, 0, 0]], [-1, 0])
        _, distance = _nearest_instance(scene, np.array([0.0, 0.0, 0.0]))
        self.assertAlmostEqual(distance, 50.0)

    def test_a_scene_with_NO_object_points_says_so_rather_than_guessing(self) -> None:
        """⚠ NaN, not zero. A distance of zero would read as a perfectly placed grasp, which is the
        opposite of what an empty scene means."""
        instance, distance = _nearest_instance(_scene([[0, 0, 0]], [-1]), np.zeros(3))
        self.assertEqual(instance, 0)
        self.assertTrue(math.isnan(distance))

    def test_a_cloud_with_no_instance_channel_does_not_crash(self) -> None:
        instance, distance = _nearest_instance({"points_mm": np.zeros((3, 3))}, np.zeros(3))
        self.assertEqual(instance, 0)
        self.assertTrue(math.isnan(distance))

    def test_the_off_object_threshold_is_HALF_AN_APERTURE(self) -> None:
        """Not a round number picked to make a rate look good. A grasp centre further than half the
        jaw's opening from every object point cannot have the object between its fingers."""
        self.assertAlmostEqual(_OFF_OBJECT_MM, 85.0 / 2)


class ProposalTests(unittest.TestCase):

    def test_the_distance_is_written_to_the_file(self) -> None:
        """The referee and the scorer both read the file, so a field the writer drops is a field
        nothing downstream can split on."""
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from src.robot.grasping.deep.eval.propose import Proposal, write_proposals

        proposal = Proposal(scene_id="s", instance_id=2, position_mm=(1.0, 2.0, 3.0),
                            approach=(0.0, 0.0, -1.0), closing_axis=(1.0, 0.0, 0.0),
                            width_mm=40.0, confidence=0.5, seed_index=0, slot=0, rank=0,
                            object_distance_mm=17.5)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.jsonl"
            summary = write_proposals(path, [proposal])
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(row["object_distance_mm"], 17.5)
        self.assertEqual(summary["on_object"], 1)
        self.assertIn("distinct_share", summary)

    def test_an_off_object_proposal_is_counted_as_such(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from src.robot.grasping.deep.eval.propose import Proposal, write_proposals

        far = Proposal(scene_id="s", instance_id=2, position_mm=(1.0, 2.0, 3.0),
                       approach=(0.0, 0.0, -1.0), closing_axis=(1.0, 0.0, 0.0), width_mm=40.0,
                       confidence=0.5, seed_index=0, slot=0, rank=0, object_distance_mm=120.0)
        with TemporaryDirectory() as tmp:
            summary = write_proposals(Path(tmp) / "p.jsonl", [far])
        self.assertEqual(summary["on_object"], 0)


if __name__ == "__main__":
    unittest.main()
