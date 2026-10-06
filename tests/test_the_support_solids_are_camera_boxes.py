"""The solids of what the parts stand on are boxes the camera saw (fix plan Track S, test 9).

They carry the camera world's name prefix, ``seen_s<k>_support<i>`` and ``seen_s<k>_bench``, so where only the camera's
boxes refuse the planner's world, W's admission sets them aside by name with the rest and the exact guard decides them
(``curobo_client._camera_box_names``); and the exact guard judges them as it judges every box the camera saw, at
``perceived_min_distance_mm``, not at the ``min_distance_mm`` a declared fixture keeps. The declared bench slab stays
the planner's.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.safety.planning._curobo_protocol import PERCEIVED_PREFIX
from src.robot.safety.planning.curobo_client import _camera_box_names
from src.robot.safety.planning.live_world import _guard_boxes
from tests import _cell_2026_10_01 as cell
from tests.test_the_guard_holds_the_bench import _hand_over_the_bench


def _world(owner: cell.OwnerCell, depth: "np.ndarray | None" = None):  # noqa: ANN202
    look = cell.look("P1")
    snapshot = owner.world(look, depth).world_for(self_envelope=owner.envelope(np.radians(cell.LOOK_0_DEG)),
                                                  near_point_mm=(-165.0, -679.0, 300.0))
    assert snapshot.perceived is not None, snapshot.reason
    return snapshot


class TheSolidsAreCameraBoxesTests(unittest.TestCase):
    def test_every_solid_carries_the_camera_worlds_prefix_and_is_set_aside_by_name(self) -> None:
        snapshot = _world(cell.owner_cell())
        solids = [box.name for box in snapshot.perceived.boxes if box.kind != "seen"]
        self.assertTrue(any(name.endswith("_bench") for name in solids))
        self.assertGreaterEqual(sum("_support" in name for name in solids), 4)
        self.assertTrue(all(name.startswith(PERCEIVED_PREFIX) for name in solids))
        every = [box.name for box in snapshot.perceived.boxes]
        self.assertEqual(_camera_box_names(every), tuple(sorted(every)))
        # The declared slab is the planner's: it is never set aside with the camera's boxes.
        declared = [box["name"] for box in snapshot.cuboids[: snapshot.declared_count]]
        self.assertIn("support_plane", declared)
        with self.assertRaises(ValueError):
            _camera_box_names([*solids, "support_plane"])

    def test_the_guard_judges_them_at_the_distance_it_keeps_from_what_the_camera_saw(self) -> None:
        """A cell holding declared fixtures at 10 mm and the camera's boxes at 5 mm, its fingers kept from the whole
        solid too (``fingers_to_the_support_reading`` off): 7 mm over the bench's solid is admitted, 3 mm is refused at
        5 mm."""
        robot = cell.owner_robot(perceived={"fingers_to_the_support_reading": False})
        robot["safety"]["self_collision"]["min_distance_mm"] = 10.0
        owner = cell.OwnerCell(robot)
        if not owner.has_the_engine():
            self.skipTest("no exact mesh engine or UR10 bundle on this box")
        self.assertEqual(RobotConfig.model_validate(robot).safety.self_collision.perceived_min_distance_mm, 5.0)
        look = cell.look("P1")
        bench = cell.level_table(look, -4.0)  # read low, so the bench's own solid is what holds it
        boxes = _guard_boxes(_world(owner, bench).perceived)
        top = [box for box in boxes if box.name.endswith("_bench")][0]
        top_mm = float(top.center_mm[2] + top.half_extents_mm[2])
        self.assertAlmostEqual(top_mm, 7.0, places=6)
        self.assertEqual(owner.refused(boxes, _hand_over_the_bench(top_mm + 7.0, 0.0)), "")
        refusal = owner.refused(boxes, _hand_over_the_bench(top_mm + 3.0, 0.0))
        self.assertIn("_bench", refusal)
        self.assertIn("< 5.000 mm", refusal)
        self.assertTrue(math.isfinite(top_mm))

    def test_the_fingers_come_to_1_mm_of_the_bench_and_no_nearer(self) -> None:
        """The owner, 2026-10-06: "bis auf 1 mm, solange er sie nicht berührt", the bench as the mat. The same cell at
        the shipped finger floor: the guard keeps its 5 mm from the solid's top lowered for the fingers by the band, the
        allowance and the 4 mm the floor lies under its distance, so the fingers come to 1 mm over the bench and no
        nearer."""
        robot = cell.owner_robot()
        robot["safety"]["self_collision"]["min_distance_mm"] = 10.0
        owner = cell.OwnerCell(robot)
        if not owner.has_the_engine():
            self.skipTest("no exact mesh engine or UR10 bundle on this box")
        self.assertEqual(RobotConfig.model_validate(robot).safety.planning_world.perceived.finger_floor_mm, 1.0)
        bench = cell.level_table(cell.look("P1"), -4.0)
        boxes = _guard_boxes(_world(owner, bench).perceived)
        top = [box for box in boxes if box.name.endswith("_bench")][0]
        top_mm = float(top.center_mm[2] + top.half_extents_mm[2])
        floor_mm = top_mm - float(top.finger_top_mm) + 5.0
        self.assertAlmostEqual(floor_mm, 1.0, places=6)
        self.assertEqual(owner.refused(boxes, _hand_over_the_bench(floor_mm + 1.0, 0.0)), "")
        refusal = owner.refused(boxes, _hand_over_the_bench(floor_mm - 0.5, 0.0))
        self.assertIn("_bench", refusal)
        self.assertIn("finger", refusal)


if __name__ == "__main__":
    unittest.main()
