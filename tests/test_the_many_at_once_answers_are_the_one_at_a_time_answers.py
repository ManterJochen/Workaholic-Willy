"""What SFE asks of many builds at once answers what it answers of one, and says "unsure" only on a threshold.

``support_footprint._build_many`` (``robot.grasping.batched_builds``) asks the guard's floor (``HandFloor.under_many``),
the camera's boxes (``SeenEnvelope.refuses_many``), the seen test (``SeenInTheFrame.unseen_many``), the obstacle grids
(``_ObstacleSet.met_many``), the room (``_Room.least_many``) and the part (``_inside_many``) for every build of a
closing line at once. Each answers what the one-build call answers for the same points: the grids and the room
exactly, the rest, made over many points whose products may round their last bit another way, wherever no number lies
within 1e-6 of its threshold, and "unsure" (2) only there, where SFE asks the one-build call instead.

The camera's boxes are measured only where they could refuse: a pair whose gap along a face normal stands at the limit
is never nearer (a projection never lengthens a distance, checked on 200,000 random pairs), and a pair whose hand point
nearest the box stands nearer than the limit surely is. And where this machine's numpy would round a stacked product
otherwise, or making the builds at once fails, SFE makes them one at a time and says so once: the same grasps.
"""

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest import mock

import numpy as np

from src.robot.grasping.generation import scene_obstacles as so
from src.robot.grasping.generation import support_footprint as sf
from src.robot.grasping.generation.scene_obstacles import SeenEnvelope, SeenInTheFrame, box_distances_mm
from tests.test_a_lines_builds_made_at_once_answer_to_the_bit import _grasps, _same
from tests.test_sfe_units_in_any_order_give_the_search_its_answer import scenes
from tests.test_the_cheaper_builds_answer_to_the_bit import _solid


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
        if said_many != sf._UNSURE:
            test.assertEqual(int(said_one), said_many, f"{what} {k}")
    return int((many == sf._UNSURE).sum())


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
        from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
        from src.robot.safety.planning.perceived import SeenBox
        from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import hande_cell

        cfg = hande_cell()
        hand = build_gripper_geometry(cfg.grasping.gripper_geometry)
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


class TheSeenTestTests(unittest.TestCase):
    def test_many_builds_at_once_are_seen_where_one_is(self) -> None:
        rng = np.random.default_rng(5)
        to_camera = np.eye(4)
        to_camera[:3, :3] = _turns(rng, 1)[0]
        to_camera[:3, 3] = [10.0, -20.0, 600.0]
        depth = rng.uniform(450.0, 800.0, (240, 320))
        depth[rng.random((240, 320)) < 0.1] = 0.0
        depth[rng.random((240, 320)) < 0.05] = np.nan
        seen = SeenInTheFrame(depth_mm=depth, to_camera=to_camera, fx=280.0, fy=280.0, cx=160.0, cy=120.0)
        points = rng.uniform(-150.0, 150.0, (500, 12, 3))
        asked = rng.random((500, 12)) < 0.8
        many = seen.unseen_many(points, asked)
        one = [not bool(seen(points[k][asked[k]]).all()) for k in range(500)]
        self.assertEqual(0, _agrees(self, many, one, "build"))
        self.assertTrue(any(one) and not all(one), "the frame sees some builds whole and misses some")


class TheFloorTests(unittest.TestCase):
    def test_many_builds_at_once_stand_over_the_floor_where_one_does(self) -> None:
        rng = np.random.default_rng(13)
        floors = [sf.HandFloor(solids=(), distance_mm=3.0)]
        for fingers_to_the_reading in (False, True):
            for drop in (0.0, 1.0):
                solids = tuple(_solid(rng, tilt_deg=tilt) for tilt in (0.0, 2.0, 12.0, 95.0))
                floors.append(sf.HandFloor(solids=solids, distance_mm=float(rng.uniform(1.0, 5.0)),
                                           fingers_to_the_reading=fingers_to_the_reading, finger_floor_drop_mm=drop))
        for floor in floors:
            points = rng.uniform([-250.0, -250.0, 0.0], [250.0, 250.0, 120.0], (200, 20, 3))
            asked = rng.random((200, 20)) < 0.9
            for fingers in (False, True):
                many = floor.under_many(points, asked, fingers=fingers)
                one = [floor.under(points[k][asked[k]], fingers=fingers) for k in range(200)]
                self.assertEqual(0, _agrees(self, many, one, "build"))


class TheGridsTheRoomAndThePartTests(unittest.TestCase):
    def test_the_grids_the_room_and_the_part_answer_exactly_what_they_answer_one_build_at_a_time(self) -> None:
        rng = np.random.default_rng(17)
        for name, (cloud, keywords) in scenes().items():
            plan = sf.plan_support_footprint(sf.SfeInputs(cloud, counting=True, **keywords))
            assert plan is not None
            obstacles, _side, _every = plan.grids()
            lo, hi = cloud.min(axis=0) - 40.0, cloud.max(axis=0) + 40.0
            points = rng.uniform(lo, hi, (150, 30, 3))
            asked = rng.random((150, 30)) < 0.8
            with self.subTest(scene=name):
                for counting in (False, True):
                    hit, sets = obstacles.met_many(points, asked, counting=counting)
                    for k in range(150):
                        mine = points[k][asked[k]]
                        self.assertEqual(obstacles.hits(mine), bool(hit[k]))
                        if counting:
                            self.assertEqual(obstacles.met(mine),
                                             tuple(n for n, on in zip(("seen", "declared", "own"), sets[k]) if on))
                for tree in (True, False):
                    room = sf._Room(keywords.get("obstacle_points_base_mm"), plan.rigid)
                    if not tree:
                        room.tree = None
                    least = room.least_many(points, asked)
                    for k in range(150):
                        self.assertEqual(np.float64(room.least_mm(points[k][asked[k]])).tobytes(),
                                         np.float64(least[k]).tobytes())
                margin = -1.0 - plan.prism.inflate
                inside = sf._inside_many(plan.prism, plan.prism.normals.T, points, asked, margin)
                one = [bool(plan.prism.contains(points[k][asked[k]], margin_mm=margin).any()) for k in range(150)]
                self.assertEqual(0, _agrees(self, inside, one, "build"))


# ---------------------------------------------------------------------------------------------------------------------
# One at a time where at once cannot be had
# ---------------------------------------------------------------------------------------------------------------------


class OneAtATimeWhereAtOnceCannotBeHadTests(unittest.TestCase):
    def test_this_machines_numpy_answers_to_the_bit(self) -> None:
        from tests.test_sfe_says_why_it_refused import default_jaw, hande_jaw

        self.assertEqual("", sf.batched_builds_hold())
        self.assertEqual("", sf.batched_builds_hold(hande_jaw()))
        self.assertEqual("", sf.batched_builds_hold(default_jaw()))

    def _one_at_a_time(self, patch: Any, said: str) -> None:
        cloud, keywords = scenes()["a bar boxed in"]
        expected = _grasps(cloud, keywords, batched=False)
        with patch, mock.patch.object(sf, "_SAID", set()), \
                self.assertLogs("src.robot.grasping.generation.support_footprint", "WARNING") as told:
            got = _grasps(cloud, keywords, batched=True)
            again = _grasps(cloud, keywords, batched=True)
        _same(self, expected, got, "the first part")
        _same(self, expected, again, "the next")
        self.assertEqual(1, len(told.records), "said once")
        self.assertIn(said, told.output[0])
        self.assertIn("the grasps are the same", told.output[0])

    def test_a_numpy_whose_stacked_products_round_otherwise_has_every_build_made_alone(self) -> None:
        sf._stacked_products_hold.cache_clear()
        self.addCleanup(sf._stacked_products_hold.cache_clear)
        self._one_at_a_time(mock.patch.object(sf, "_stacked_products_hold", return_value="a reason of this machine"),
                            "a reason of this machine")

    def test_a_pass_that_fails_has_every_build_made_alone_and_counts_nothing_twice(self) -> None:
        def broken(*args: Any, **keywords: Any) -> Any:
            raise MemoryError("no room for the line")

        self._one_at_a_time(mock.patch.object(sf, "_build_many", side_effect=broken), "MemoryError: no room")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
