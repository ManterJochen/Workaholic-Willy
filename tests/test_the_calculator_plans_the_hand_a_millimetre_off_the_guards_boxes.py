"""The calculator plans the open hand a millimetre past the guard's distance from a camera box (the grasp bench,
2026-10-06).

The guard's boxes are built from every frame the pick kept and merged to fit its slots, the calculator's from the one
frame it judged, so the two stand a little apart. On a tray the guard refused the line down to four grasps in a row 0.18
to 0.68 mm short of its 3 mm, and every refused grasp took one of the pick's tries. ``SeenEnvelope`` now keeps the
guard's distance and ``WHOLE_SLACK_MM`` from a whole box, as it keeps a finger ``SOFT_SLACK_MM`` off a named part's
surface. And it asks the hand at every place of its way in, from the grasp back to the standoff, in one distance: the
hand's boxes stand square to the approach, so the way each sweeps is a box again.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.robot.grasping.generation.scene_obstacles import SOFT_SLACK_MM, WHOLE_SLACK_MM, SeenEnvelope

GUARD_MM = 3.0
#: The open hand straight down at 100 mm: its closing axis BASE x, its approach BASE -z.
AT = np.array([0.0, 0.0, 100.0])
DOWN = np.column_stack([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def _hand() -> "tuple[Any, float]":
    from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
    from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import hande_cell

    cfg = hande_cell()
    return build_gripper_geometry(cfg.grasping.gripper_geometry), float(cfg.gripper.max_width_mm)


def _box(centre: "tuple[float, float, float]", half: "tuple[float, float, float]", soft_mm: float = 0.0) -> Any:
    return SimpleNamespace(centre_mm=np.asarray(centre, dtype=np.float64), rotation=np.eye(3),
                           half_extents_mm=np.asarray(half, dtype=np.float64), soft_mm=soft_mm)


def _envelope(boxes: "list[Any]", way_in_mm: "tuple[float, ...]" = (0.0, 40.0, 80.0)) -> SeenEnvelope:
    hand, open_mm = _hand()
    envelope = SeenEnvelope.of(boxes, gripper_model=hand, open_width_mm=open_mm, distance_mm=GUARD_MM,
                               way_in_mm=way_in_mm)
    assert envelope is not None
    return envelope


def _outer_x() -> float:
    """How far the open fingers reach along +x from the grasp centre, millimetres."""
    hand, open_mm = _hand()
    return max(float(b.max_corner_mm[0]) for b in hand.collision_boxes(open_mm))


class AWholeBoxKeepsTheSlackTests(unittest.TestCase):
    def _wall(self, gap_mm: float) -> Any:
        """A wall beside the open finger on +x, ``gap_mm`` off it, as tall as the hand's whole way in."""
        face = _outer_x() + gap_mm
        return _box((face + 10.0, 0.0, 150.0), (10.0, 80.0, 150.0))

    def test_a_wall_the_guard_admits_by_less_than_the_slack_is_refused(self) -> None:
        envelope = _envelope([self._wall(GUARD_MM + WHOLE_SLACK_MM / 2.0)])
        self.assertAlmostEqual(envelope.least_mm(AT, DOWN), GUARD_MM + WHOLE_SLACK_MM / 2.0, places=6)
        self.assertTrue(envelope.refuses(AT, DOWN))

    def test_a_wall_past_the_slack_is_not(self) -> None:
        envelope = _envelope([self._wall(GUARD_MM + WHOLE_SLACK_MM + 0.5)])
        self.assertFalse(envelope.refuses(AT, DOWN))

    def test_a_named_part_keeps_its_own_slack(self) -> None:
        """A finger may come to a named part's measured surface, less ``SOFT_SLACK_MM``; the wall's slack is not its."""
        soft = 8.0
        face = _outer_x() + SOFT_SLACK_MM + 0.5
        # Beside the fingertips, its top under the palm.
        part = _box((face + 20.0 - soft, 0.0, 80.0), (20.0, 20.0, 40.0), soft_mm=soft)
        envelope = _envelope([part])
        self.assertAlmostEqual(envelope.into_parts_mm(AT, DOWN), SOFT_SLACK_MM + 0.5, places=6)
        self.assertFalse(envelope.refuses(AT, DOWN))


class TheWayInIsEveryPlaceTests(unittest.TestCase):
    def test_one_distance_answers_for_every_place_from_the_grasp_to_the_standoff(self) -> None:
        rng = np.random.default_rng(7)
        for trial in range(40):
            with self.subTest(trial=trial):
                centre = (float(rng.uniform(-90.0, 90.0)), float(rng.uniform(-90.0, 90.0)),
                          float(rng.uniform(60.0, 260.0)))
                half = tuple(float(v) for v in rng.uniform(2.0, 30.0, 3))
                boxes = [_box(centre, half)]  # type: ignore[arg-type]
                swept = _envelope(boxes, (0.0, 80.0)).least_mm(AT, DOWN)
                places = [_envelope(boxes, (float(b),)).least_mm(AT, DOWN) for b in np.arange(0.0, 80.01, 0.5)]
                if not np.isfinite(min(places)):
                    self.assertFalse(np.isfinite(swept) and swept < 20.0)
                    continue
                self.assertLessEqual(swept, min(places) + 1e-6)
                self.assertLessEqual(min(places) - swept, 0.5 + 1e-6)

    def test_the_ends_of_the_way_are_its_nearest_and_its_farthest(self) -> None:
        """A box just past the standoff stays out of the way in; one inside it at the farthest place is met."""
        hand, open_mm = _hand()
        top = max(-float(b.min_corner_mm[2]) for b in hand.collision_boxes(open_mm))  # the palm's back, up the approach
        over = _box((0.0, 0.0, AT[2] + 80.0 + top + GUARD_MM + WHOLE_SLACK_MM + 5.0 + 5.0), (40.0, 40.0, 5.0))
        self.assertFalse(_envelope([over], (0.0, 80.0)).refuses(AT, DOWN))
        inside = _box((0.0, 0.0, AT[2] + 80.0 + top + 2.0 + 5.0), (40.0, 40.0, 5.0))
        self.assertTrue(_envelope([inside], (0.0, 80.0)).refuses(AT, DOWN))
        self.assertFalse(_envelope([inside], (0.0, 40.0)).refuses(AT, DOWN))


if __name__ == "__main__":
    unittest.main()
