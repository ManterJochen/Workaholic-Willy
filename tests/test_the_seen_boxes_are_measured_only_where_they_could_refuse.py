"""The seen boxes are measured only where they could refuse, and the verdict is the one of every pair measured.

``SeenEnvelope.refuses`` keeps the open hand the guard's distance and ``WHOLE_SLACK_MM`` off every whole box the camera
world builds of a neighbour, and a finger ``SOFT_SLACK_MM`` off a named part's measured surface. SFE asks it of every
build it gets that far with, and on the owner's cell (2026-10-08) a call measured the hand's three boxes against 38
boxes, every pair, which was 95 % of its time. A pair whose bounding spheres stand apart by the limit or more cannot come
nearer than the limit, so it is no longer measured: the verdict must stay
``least_mm < distance_mm + WHOLE_SLACK_MM or into_parts_mm < SOFT_SLACK_MM``, the test of every pair, which
``least_mm`` and ``into_parts_mm`` still make for their other callers.

Pinned on the owner's Hand-E: seeded scenes of whole, soft and turned boxes about a grasp, poses tilted, turned and
rolled as SFE's fine search rolls them, and the edges 0.01 mm either side of each limit, turned with the hand.
"""

from __future__ import annotations

import math
import unittest
from functools import lru_cache
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from src.robot.grasping.generation import scene_obstacles
from src.robot.grasping.generation.scene_obstacles import SOFT_SLACK_MM, WHOLE_SLACK_MM, SeenEnvelope
from src.robot.grasping.generation.support_footprint import _rolled

GUARD_MM = 3.0
#: Where the grasps stand: over the owner's mat, 56 mm over the bench.
CENTRE = np.array([0.0, -650.0, 56.0])


@lru_cache(maxsize=None)
def _hand() -> "tuple[Any, float]":
    from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
    from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import hande_cell

    cfg = hande_cell()
    return build_gripper_geometry(cfg.grasping.gripper_geometry), float(cfg.gripper.max_width_mm)


def _turn(yaw: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    about_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    about_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return about_z @ about_x


def _box(centre: Any, half: Any, *, soft_mm: float = 0.0, rotation: "np.ndarray | None" = None) -> Any:
    return SimpleNamespace(centre_mm=np.asarray(centre, dtype=np.float64),
                           rotation=np.eye(3) if rotation is None else rotation,
                           half_extents_mm=np.asarray(half, dtype=np.float64), soft_mm=soft_mm)


def _envelope(boxes: "list[Any]") -> SeenEnvelope:
    hand, open_mm = _hand()
    envelope = SeenEnvelope.of(boxes, gripper_model=hand, open_width_mm=open_mm, distance_mm=GUARD_MM)
    assert envelope is not None
    return envelope


def _every_pair(envelope: SeenEnvelope, position: np.ndarray, rotation: np.ndarray) -> bool:
    """The verdict of every pair measured: what ``refuses`` said before it left any pair out."""
    return (envelope.least_mm(position, rotation) < envelope.distance_mm + WHOLE_SLACK_MM
            or envelope.into_parts_mm(position, rotation) < SOFT_SLACK_MM)


def _scene(rng: np.random.Generator, gap_mm: float, extra: int) -> SeenEnvelope:
    """Four neighbours ``gap_mm`` off a 40 mm part's sides, every other one soft, each turned a little about the vertical
    and one tipped, and ``extra`` more scattered past them, half of them soft."""
    boxes = []
    for number, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        off = 20.0 + gap_mm + 26.0
        boxes.append(_box(CENTRE + (dx * off, dy * off, 20.0), (26.0, 26.0, 48.0), soft_mm=8.0 if number % 2 else 0.0,
                          rotation=_turn(rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2) if number == 3 else 0.0)))
    for _ in range(extra):
        reach, bearing = rng.uniform(gap_mm + 60.0, gap_mm + 220.0), rng.uniform(0.0, 2.0 * math.pi)
        boxes.append(_box(CENTRE + (reach * math.cos(bearing), reach * math.sin(bearing), rng.uniform(5.0, 40.0)),
                          rng.uniform(6.0, 30.0, 3), soft_mm=8.0 if rng.random() < 0.5 else 0.0,
                          rotation=_turn(rng.uniform(0.0, math.pi), rng.uniform(-0.4, 0.4))))
    return _envelope(boxes)


def _poses(rng: np.random.Generator, count: int) -> "list[tuple[np.ndarray, np.ndarray]]":
    """Grasps about the part: tilted on SFE's fine ladder, turned about the vertical, rolled as its fine search rolls."""
    out = []
    for _ in range(count):
        tilt = math.radians(float(rng.choice([0.0, 7.5, 15.0, 30.0, 45.0, 60.0, 90.0])))
        yaw = rng.uniform(0.0, 2.0 * math.pi)
        roll = float(rng.choice([0.0, 10.0, -10.0, 20.0, -20.0]))
        axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        down = np.array([0.0, 0.0, -1.0])
        approach = down * math.cos(tilt) + np.cross(axis, down) * math.sin(tilt)
        axis, approach = _rolled(axis, approach, roll)
        binormal = np.cross(approach, axis)
        at = CENTRE + (rng.uniform(-8.0, 8.0), rng.uniform(-8.0, 8.0), rng.uniform(12.0, 40.0))
        out.append((at, np.column_stack([axis, binormal / np.linalg.norm(binormal), approach])))
    return out


class ThePrunedVerdictIsTheVerdictOfEveryPairTests(unittest.TestCase):
    def test_seeded_poses_with_rolls_soft_and_turned_boxes_get_the_same_verdict(self) -> None:
        rng = np.random.default_rng(20261008)
        refused = admitted = 0
        for gap, extra in ((8.0, 4), (14.0, 12), (22.0, 34), (35.0, 34), (60.0, 20), (90.0, 30), (130.0, 34)):
            envelope = _scene(rng, gap, extra)
            for number, (position, rotation) in enumerate(_poses(rng, 150)):
                with self.subTest(gap=gap, pose=number):
                    expected = _every_pair(envelope, position, rotation)
                    self.assertEqual(expected, envelope.refuses(position, rotation))
                    refused += int(expected)
                    admitted += int(not expected)
        # Both verdicts are asked many times, or the comparison proves nothing.
        self.assertGreater(refused, 100)
        self.assertGreater(admitted, 100)

    def test_a_soft_box_alone_and_a_hand_with_no_finger_listed_are_judged_as_every_pair_judges_them(self) -> None:
        rng = np.random.default_rng(7)
        part = _box(CENTRE + (46.0, 0.0, 20.0), (26.0, 26.0, 48.0), soft_mm=8.0)
        envelope = _envelope([part])
        bare = SeenEnvelope(boxes=envelope.boxes, hand_centres=envelope.hand_centres, hand_halves=envelope.hand_halves,
                            distance_mm=envelope.distance_mm, way_in_mm=envelope.way_in_mm)
        for number, (position, rotation) in enumerate(_poses(rng, 120)):
            with self.subTest(pose=number):
                self.assertEqual(_every_pair(envelope, position, rotation), envelope.refuses(position, rotation))
                self.assertEqual(_every_pair(bare, position, rotation), bare.refuses(position, rotation))


class TheEdgesAreDecidedAsBeforeTests(unittest.TestCase):
    """0.01 mm either side of each limit, the hand and the box turned together: a turn keeps every distance."""

    TURNS = ((0.0, 0.0), (0.7, 0.0), (2.1, 0.3), (-1.2, -0.25))

    @staticmethod
    def _outer_x() -> float:
        hand, open_mm = _hand()
        return max(float(b.max_corner_mm[0]) for b in hand.collision_boxes(open_mm))

    def _turned(self, box: Any, yaw: float, pitch: float) -> "tuple[Any, np.ndarray, np.ndarray]":
        """``box``, laid out beside the hand straight down at ``CENTRE``, and the hand, turned together about it."""
        turn = _turn(yaw, pitch)
        down = np.column_stack([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        moved = _box(CENTRE + turn @ (np.asarray(box.centre_mm) - CENTRE), box.half_extents_mm, soft_mm=box.soft_mm,
                     rotation=turn @ np.asarray(box.rotation))
        return moved, CENTRE.copy(), turn @ down

    def test_a_wall_a_hundredth_inside_the_slack_is_refused_and_one_a_hundredth_past_it_is_not(self) -> None:
        limit = GUARD_MM + WHOLE_SLACK_MM
        for yaw, pitch in self.TURNS:
            for offset, refused in ((-0.01, True), (0.01, False)):
                with self.subTest(yaw=yaw, pitch=pitch, offset=offset):
                    face = self._outer_x() + limit + offset
                    wall = _box(CENTRE + (face + 10.0, 0.0, 100.0), (10.0, 80.0, 150.0))
                    box, position, rotation = self._turned(wall, yaw, pitch)
                    envelope = _envelope([box])
                    self.assertAlmostEqual(envelope.least_mm(position, rotation), limit + offset, places=6)
                    self.assertEqual(refused, _every_pair(envelope, position, rotation))
                    self.assertEqual(refused, envelope.refuses(position, rotation))

    def test_a_named_part_a_hundredth_inside_its_slack_is_refused_and_one_a_hundredth_past_it_is_not(self) -> None:
        soft = 8.0
        for yaw, pitch in self.TURNS:
            for offset, refused in ((-0.01, True), (0.01, False)):
                with self.subTest(yaw=yaw, pitch=pitch, offset=offset):
                    face = self._outer_x() + SOFT_SLACK_MM + offset
                    # Beside the fingertips, its top under the palm, as the slack test of 2026-10-06 lays it out.
                    part = _box(CENTRE + (face + 20.0 - soft, 0.0, -20.0), (20.0, 20.0, 40.0), soft_mm=soft)
                    box, position, rotation = self._turned(part, yaw, pitch)
                    envelope = _envelope([box])
                    self.assertAlmostEqual(envelope.into_parts_mm(position, rotation), SOFT_SLACK_MM + offset,
                                           places=6)
                    self.assertEqual(refused, _every_pair(envelope, position, rotation))
                    self.assertEqual(refused, envelope.refuses(position, rotation))


class OnlyThePairsThatCouldRefuseAreMeasuredTests(unittest.TestCase):
    def test_boxes_whose_spheres_stand_past_the_limit_are_never_measured(self) -> None:
        """One neighbour beside the hand, and 30 boxes all round it inside the reach every pair used to be measured at
        (the hand's own reach, the box's and the distance with its slack) and past the limit from every box of the
        hand: the 30 are never measured, the verdict is the one of every pair."""
        hand, open_mm = _hand()
        envelope = _envelope([_box(CENTRE, (1.0, 1.0, 1.0))])
        front, back = min(envelope.way_in_mm), max(envelope.way_in_mm)
        swept_centres = envelope.hand_centres - np.array([0.0, 0.0, (front + back) / 2.0])
        swept_halves = envelope.hand_halves + np.array([0.0, 0.0, (back - front) / 2.0])
        reach_hand = float(np.max(np.linalg.norm(swept_centres, axis=1) + np.linalg.norm(swept_halves, axis=1)))
        half = np.array([12.0, 12.0, 30.0])
        off = reach_hand + float(np.linalg.norm(half)) + GUARD_MM + WHOLE_SLACK_MM - 0.1
        position, rotation = _poses(np.random.default_rng(11), 1)[0]
        bearings = np.random.default_rng(3).normal(size=(30, 3))
        far = [_box(position + off * b / np.linalg.norm(b), half) for b in bearings]
        near = _box(position + (0.0, 0.0, 90.0), (20.0, 20.0, 20.0))
        envelope = _envelope([near, *far])
        measured: list[int] = []
        real = scene_obstacles.box_distances_mm

        def counted(*arrays: Any) -> np.ndarray:
            measured.append(int(np.asarray(arrays[0]).reshape(-1, 3).shape[0]))
            return real(*arrays)

        with mock.patch.object(scene_obstacles, "box_distances_mm", side_effect=counted):
            verdict = envelope.refuses(position, rotation)
            pruned, measured[:] = sum(measured), []
            expected = _every_pair(envelope, position, rotation)
        self.assertEqual(expected, verdict)
        hand_boxes = envelope.hand_centres.shape[0]
        self.assertGreaterEqual(sum(measured), hand_boxes * 31, "every pair was not measured where it used to be")
        self.assertLessEqual(pruned, hand_boxes, "a box past the limit from every box of the hand was measured")
        del hand, open_mm


if __name__ == "__main__":
    unittest.main()
