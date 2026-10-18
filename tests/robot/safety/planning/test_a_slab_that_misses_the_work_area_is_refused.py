"""A declared bench slab that misses the work area is refused at build (fix plan S2, Track S, test 11).

The owner's slab stood at (500, 500) with an extent of 1000 x 1000 mm, behind the robot, while every part lay at y -900
to -270: the planner held a bench where nothing stood and none under the parts, and the guard none at all. A cell is now
refused at build where its slab covers none of the workspace box, naming the slab's centre and extent and the box; one
covering part of it is said; the Monday config, (-10, -585) with 900 x 700, covers it all.
"""

from __future__ import annotations

import logging
import unittest

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.execution.camera_world_wiring import SupportSlabMissesTheWorkspace, _live_planner_world
from src.robot.safety.planning.live_world import CameraView
from tests import _cell_2026_10_01 as cell


class _Silent:
    def grab_surface_depth(self) -> None:
        return None


def _config(centre: "tuple[float, float]", extent: "tuple[float, float]") -> RobotConfig:
    return RobotConfig.model_validate(cell.owner_robot(support_plane={
        "height_mm": 0.0, "extent_mm": list(extent), "thickness_mm": 50.0, "center_mm": list(centre)}))


def _build(config: RobotConfig):  # noqa: ANN202
    return _live_planner_world(config, [CameraView(name="EIH_Cam", depth_source=_Silent(), camera_to_tool=np.eye(4))])


class ASlabThatMissesTheWorkAreaTests(unittest.TestCase):
    def test_the_owners_slab_behind_the_robot_is_refused_naming_where_it_is(self) -> None:
        with self.assertRaises(SupportSlabMissesTheWorkspace) as caught:
            _build(_config((500.0, 500.0), (1000.0, 1000.0)))
        said = str(caught.exception)
        self.assertIn("centre (500, 500)", said)
        self.assertIn("extent 1000 x 1000", said)
        self.assertIn("x -420 to 400", said)
        self.assertIn("y -900 to -270", said)
        self.assertIn("covers none of it", said)
        self.assertIsInstance(caught.exception, ValueError)

    def test_the_monday_slab_under_the_work_area_builds(self) -> None:
        logger = logging.getLogger("src.robot.execution.camera_world_wiring")
        with self.assertNoLogs(logger, logging.WARNING):
            world = _build(_config((-10.0, -585.0), (900.0, 700.0)))
        self.assertTrue(world.tuning.support_surfaces)

    def test_a_slab_covering_part_of_the_work_area_is_said(self) -> None:
        with self.assertLogs("src.robot.execution.camera_world_wiring", logging.WARNING) as said:
            _build(_config((300.0, -585.0), (600.0, 700.0)))
        self.assertIn("covers only part of it", said.records[0].getMessage())


if __name__ == "__main__":
    unittest.main()
