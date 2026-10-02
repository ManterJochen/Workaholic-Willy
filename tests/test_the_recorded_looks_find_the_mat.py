"""The owner's recorded looks find the mat and the yellow bin's floor (fix plan Track S, test 13).

Two of the owner's looks of 2026-10-01 (P1, P5: LOOK[0], the Zollstock in a pile on the 55 mm foam mat, the yellow bin
beside it), built into the world the cell builds from the owner's profile. Before, the mat came back as strips of
obstacles 15 mm over its reading and the slots overflowed: 73 and 79 boxes merged to fit the 64. Now the stack finds
the mat itself, as one surface tilted the degree the camera reads it, and holds it as at most four solids; the bin's
floor is a support of its own and the bin's walls stay obstacles.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from tests import _cell_2026_10_01 as cell

#: Where the yellow bin stands beside the mat in these looks, BASE millimetres (its floor, as the camera reads it).
_BIN_X, _BIN_Y = (100.0, 350.0), (-690.0, -490.0)


def _world(name: str, **perceived: object) -> Any:
    owner = cell.owner_cell()
    look = cell.look(name)
    config = None
    if perceived:
        from src.config.schema.robot import RobotConfig

        config = RobotConfig.model_validate(cell.owner_robot(perceived=perceived))
    world = owner.world(look, config=config)
    world.offer_segmentation(camera="EIH_Cam", target_points_base_mm=look.target_points_base_mm,
                             target_label="Einen Zollstock", hold=True)
    snapshot = world.world_for(self_envelope=owner.envelope(np.radians(cell.LOOK_0_DEG)),
                               near_point_mm=(-165.0, -679.0, 300.0))
    assert snapshot.perceived is not None, snapshot.reason
    return snapshot.perceived


class TheRecordedLooksFindTheMatTests(unittest.TestCase):
    worlds: dict[str, Any] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.worlds = {name: _world(name) for name in cell.LOOKS}

    def test_the_mat_is_one_surface_tilted_as_the_camera_reads_it(self) -> None:
        for name, world in self.worlds.items():
            with self.subTest(look=name):
                self.assertIsNotNone(world.supports)
                mats = [s for s in world.supports.surfaces if s.z_range_mm[0] > 40.0]
                self.assertEqual(len(mats), 1, world.supports.render())
                mat = mats[0]
                self.assertGreaterEqual(mat.z_range_mm[0], 50.0)
                self.assertLessEqual(mat.z_range_mm[1], 63.0)
                self.assertGreaterEqual(mat.tilt_deg, 1.0)
                self.assertLessEqual(mat.tilt_deg, 1.2)
                self.assertGreater(mat.area_mm2, 0.09e6)
                self.assertLessEqual(len(mat.solids), 4)
                held = [box for box in world.boxes if box.name in mat.solids]
                self.assertEqual(len(held), len(mat.solids))
                self.assertTrue(all(box.kind == "support" and box.rotation is not None for box in held))

    def test_the_merges_drop_and_the_bench_is_held(self) -> None:
        for name, world in self.worlds.items():
            with self.subTest(look=name):
                self.assertLessEqual(world.merged_to_fit, 67)
                self.assertEqual(world.dropped_obstacle_count, 0)
                bench = [box for box in world.boxes if box.kind == "bench"]
                self.assertEqual(len(bench), 1)
                self.assertAlmostEqual(bench[0].center_mm[2] + bench[0].dims_mm[2] / 2.0, 7.0, places=6)
        self.assertLessEqual(self.worlds["P1"].merged_to_fit, 66)

    def test_before_the_mat_was_strips_and_the_slots_overflowed(self) -> None:
        before = _world("P1", support_surfaces=False)
        self.assertIsNone(before.supports)
        self.assertEqual(before.merged_to_fit, 73)
        self.assertGreater(before.merged_to_fit, self.worlds["P1"].merged_to_fit + 15)

    def test_the_yellow_bins_floor_is_a_support_and_its_walls_stay_obstacles(self) -> None:
        for name, world in self.worlds.items():
            with self.subTest(look=name):
                floors = []
                for surface in world.supports.surfaces:
                    solids = [s for s in world.supports.support_solids if s.surface == surface.index]
                    centres = np.array([s.centre_mm for s in solids])
                    inside = ((centres[:, 0] > _BIN_X[0]) & (centres[:, 0] < _BIN_X[1])
                              & (centres[:, 1] > _BIN_Y[0]) & (centres[:, 1] < _BIN_Y[1]))
                    if inside.all() and surface.z_range_mm[1] < 15.0:
                        floors.append((surface, solids))
                self.assertEqual(len(floors), 1, world.supports.render())
                surface, solids = floors[0]
                self.assertGreater(surface.area_mm2, 0.01e6)
                walls = []
                for solid in solids:
                    reach = solid.enclosing_half_extents_mm
                    for box in world.boxes:
                        if box.kind != "seen":
                            continue
                        half = np.asarray(box.enclosing_half_extents_mm)
                        beside = np.all(np.abs(np.asarray(box.center_mm[:2]) - solid.centre_mm[:2])
                                        < reach[:2] + half[:2] + 10.0)
                        if beside and box.center_mm[2] + half[2] - 15.0 > 60.0:
                            walls.append(box.name)
                self.assertGreaterEqual(len(set(walls)), 3, "the bin's walls left the world")


if __name__ == "__main__":
    unittest.main()
