"""What the part's body asks of many grasps at once answers what it answers of one, and says "unsure" only on a limit.

The body (``body_grasps``) asks the guard's floor (``HandFloor.under_many``) and the camera's boxes
(``SeenEnvelope.refuses_many``) for a batch of its grasps at once. Each answers what the one-grasp call answers for the
same points wherever no number lies within 1e-6 of its threshold, made over many points whose products may round their
last bit another way, and "unsure" (2) only there, where the body asks the one-grasp call instead.

The camera's boxes are measured only where they could refuse: a pair whose gap along a face normal stands at the limit
is never nearer (a projection never lengthens a distance, checked on 200,000 random pairs), and a pair whose hand point
nearest the box stands nearer than the limit surely is. (The other batched answers, the seen test, the obstacle grids,
the room and the part, were SFE's batched builds' alone and left with its search on 2026-10-16.)
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.robot.grasping.generation import scene_obstacles as so
from src.robot.grasping.generation.hand import HandFloor
from src.robot.grasping.generation.scene_obstacles import SeenEnvelope, box_distances_mm
from tests.test_the_cheaper_builds_answer_to_the_bit import _solid

#: What a batched answer says where only the one-grasp call decides.
UNSURE = 2


def _turns(rng: np.random.Generator, count: int) -> np.ndarray:
    """``count`` random rotations, their columns the axes."""
    q = rng.normal(size=(count, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def _agrees(test: unittest.TestCase, many: np.ndarray, one: list[bool], what: str) -> int:
    """Where the many-at-once answer is sure, it is the one-at-a-time answer; how many it was unsure of."""
    for k, (said_many, said_one) in enumerate(zip(many.tolist(), one)):
        if said_many != UNSURE:
            test.assertEqual(int(said_one), said_many, f"{what} {k}")
    return int((many == UNSURE).sum())


class TheCameraBoxesTests(unittest.TestCase):
    def test_a_face_normals_gap_never_exceeds_the_boxes_distance(self) -> None:
        rng = np.random.default_rng(42)
        worst, skipped = -math.inf, 0
        for _ in range(20):
            n = 10_000
            ca = rng.uniform(-200.0, 200.0, (n, 3)) + np.array([-130.0, -700.0, 60.0])
            cb = ca + rng.normal(size=(n, 3)) * 60.0
            ha, hb = rng.uniform(1.0, 60.0, (n, 3)), rng.uniform(1.0, 60.0, (n, 3))
            ra, rb = _turns(rng, n), _turns(rng, n)
            exact = box_distances_mm(ca, ra, ha, cb, rb, hb)
            lower = so._face_gaps(ca, ra, ha, cb, rb, hb)
            apart = exact > 0.0
            worst = max(worst, float((lower - exact)[apart].max()))
            for limit in (1.0, 4.0):
                skipped += int(((lower >= limit + so._MANY_SURE) & (exact < limit)).sum())
            upper = so._nearest_point_distances(ca, ra, ha, cb, rb, hb)
            self.assertTrue(bool((upper >= exact - 1e-9).all()), "the nearest point bounds it from above")
        self.assertLessEqual(worst, 1e-9)
        self.assertEqual(0, skipped)

    def test_many_hands_at_once_are_refused_where_one_is(self) -> None:
        """Seeded hands with rolls among whole, soft, turned and tipped boxes, as ``refuses`` judges them one at a
        time; the cull by face normals and by the nearest point changes no answer."""
        from src.robot.grasping.collision.gripper_model import jaw_envelope
        from src.robot.safety.planning.perceived import SeenBox
        from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import hande_cell

        cfg = hande_cell()
        hand = jaw_envelope(cfg)
        rng = np.random.default_rng(20261009)
        unsure, refused = 0, 0
        for scene in range(12):
            turns = _turns(rng, 8)
            if scene % 3 == 0:
                turns = np.repeat(np.eye(3)[None], 8, axis=0)
            boxes = [SeenBox(centre_mm=tuple(rng.uniform([-200.0, -200.0, 20.0], [200.0, 200.0, 80.0])),
                             rotation=tuple(map(tuple, turns[k])), half_extents_mm=tuple(rng.uniform(5.0, 40.0, 3)),
                             soft_mm=float(rng.choice([0.0, 0.0, 2.0]))) for k in range(8)]
            envelope = SeenEnvelope.of(boxes, gripper_model=hand, open_width_mm=float(cfg.gripper.max_width_mm),
                                       distance_mm=3.0)
            assert envelope is not None
            at = rng.uniform([-200.0, -200.0, 20.0], [200.0, 200.0, 160.0], (300, 3))
            hands = _turns(rng, 300)
            many = envelope.refuses_many(at, hands)
            one = [envelope.refuses(at[k], hands[k]) for k in range(300)]
            unsure += _agrees(self, many, one, f"scene {scene} hand")
            refused += sum(one)
            # Without either cull: the same answer.
            no_cull = so._nearer_many(at[:, None, :] + np.matmul(envelope._swept_centres[None],
                                                                  hands.transpose(0, 2, 1)),
                                      hands, envelope._swept_halves, envelope._swept_radii, envelope._box_centres,
                                      envelope._box_turns, envelope._box_halves, envelope._box_radii,
                                      envelope._whole_pairs, envelope.distance_mm + so.WHOLE_SLACK_MM,
                                      np.zeros(300, dtype=bool))
            culled = so._nearer_many(at[:, None, :] + np.matmul(envelope._swept_centres[None],
                                                                 hands.transpose(0, 2, 1)),
                                     hands, envelope._swept_halves, envelope._swept_radii, envelope._box_centres,
                                     envelope._box_turns, envelope._box_halves, envelope._box_radii,
                                     envelope._whole_pairs, envelope.distance_mm + so.WHOLE_SLACK_MM,
                                     np.ones(300, dtype=bool))
            self.assertEqual(no_cull.tolist(), culled.tolist())
        self.assertGreater(refused, 900, "the scenes refuse hands")
        self.assertLess(refused, 2700, "and keep some")
        self.assertEqual(0, unsure, "no random hand stands on a limit")

    def test_boxes_that_are_not_turned_squarely_are_measured_without_the_culls(self) -> None:
        from src.robot.safety.planning.perceived import SeenBox

        sheared = SeenBox(centre_mm=(0.0, 0.0, 50.0), rotation=((1.0, 0.2, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                          half_extents_mm=(10.0, 10.0, 10.0))
        envelope = SeenEnvelope(boxes=(sheared,), hand_centres=np.zeros((1, 3)), hand_halves=np.ones((1, 3)),
                                distance_mm=3.0)
        self.assertFalse(envelope._square)
        at = np.random.default_rng(3).uniform([-40.0, -40.0, 20.0], [40.0, 40.0, 80.0], (400, 3))
        hands = np.repeat(np.eye(3)[None], 400, axis=0)
        many = envelope.refuses_many(at, hands)
        _agrees(self, many, [envelope.refuses(at[k], hands[k]) for k in range(400)], "hand")


class TheFloorTests(unittest.TestCase):
    def test_many_builds_at_once_stand_over_the_floor_where_one_does(self) -> None:
        rng = np.random.default_rng(13)
        floors = [HandFloor(solids=(), distance_mm=3.0)]
        for fingers_to_the_reading in (False, True):
            for drop in (0.0, 1.0):
                solids = tuple(_solid(rng, tilt_deg=tilt) for tilt in (0.0, 2.0, 12.0, 95.0))
                floors.append(HandFloor(solids=solids, distance_mm=float(rng.uniform(1.0, 5.0)),
                                           fingers_to_the_reading=fingers_to_the_reading, finger_floor_drop_mm=drop))
        for floor in floors:
            points = rng.uniform([-250.0, -250.0, 0.0], [250.0, 250.0, 120.0], (200, 20, 3))
            asked = rng.random((200, 20)) < 0.9
            for fingers in (False, True):
                many = floor.under_many(points, asked, fingers=fingers)
                one = [floor.under(points[k][asked[k]], fingers=fingers) for k in range(200)]
                self.assertEqual(0, _agrees(self, many, one, "build"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
