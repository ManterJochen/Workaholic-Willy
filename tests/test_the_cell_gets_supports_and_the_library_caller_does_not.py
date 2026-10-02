"""The cell finds what it stands on; a library caller gets the world it always got (fix plan Track S, test 12).

A cell built from its tree carries ``planning_world.perceived.support_surfaces`` (on) and ``support_allowance_mm`` (2)
into its world, which then holds the surfaces and the bench as solids. ``WorldBuildTuning`` is the library's own: its
default leaves the supports off, so a caller that builds a tuning by hand gets the world of before, byte for byte. The
fingerprints below are that world's on the owner's two recorded looks, with and without a held target, as commit
1d91e3a builds it (every float to a millionth).
"""

from __future__ import annotations

import hashlib
import json
import unittest
from typing import Any

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.execution.camera_world_wiring import _live_planner_world
from src.robot.safety.planning.live_world import CameraView
from src.robot.safety.planning.perceived import WorldBuildTuning, build_perceived_boxes, target_keep_out_box
from tests import _cell_2026_10_01 as cell

#: ``sha256`` of the world's ``to_dict()``, floats to six places, keys sorted, as 1d91e3a builds it: by look, without and
#: with the look's target held out.
_BEFORE = {
    ("P1", False): "91ffe553c87979e7", ("P1", True): "5c79e555cddd9546",
    ("P5", False): "b3d00391b5f9f8f3", ("P5", True): "04ae67d7efb652d4",
}


def _canon(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {str(key): _canon(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canon(item) for item in value]
    return value


class _Silent:
    def grab_surface_depth(self) -> None:
        return None


class TheCellGetsSupportsTests(unittest.TestCase):
    def test_a_world_built_from_the_tree_finds_the_surfaces_and_holds_the_bench(self) -> None:
        config = RobotConfig.model_validate(cell.owner_robot())
        world = _live_planner_world(config, [CameraView(name="EIH_Cam", depth_source=_Silent(),
                                                        camera_to_tool=np.eye(4))])
        self.assertTrue(world.tuning.support_surfaces)
        self.assertEqual(world.tuning.support_allowance_mm, 2.0)
        owner = cell.owner_cell()
        snapshot = owner.world(cell.look("P1")).world_for(self_envelope=owner.envelope(np.radians(cell.LOOK_0_DEG)),
                                                          near_point_mm=(-165.0, -679.0, 300.0))
        perceived = snapshot.perceived
        assert perceived is not None and perceived.supports is not None
        self.assertTrue(perceived.supports.surfaces)
        kinds = [box.kind for box in perceived.boxes]
        self.assertEqual(kinds.count("bench"), 1)
        self.assertGreaterEqual(kinds.count("support"), 4)
        self.assertIn("support", perceived.dropped_points)
        self.assertIn("supports", perceived.to_dict())

    def test_a_cell_can_turn_them_off(self) -> None:
        config = RobotConfig.model_validate(cell.owner_robot(perceived={"support_surfaces": False}))
        world = _live_planner_world(config, [CameraView(name="EIH_Cam", depth_source=_Silent(),
                                                        camera_to_tool=np.eye(4))])
        self.assertFalse(world.tuning.support_surfaces)


class TheLibraryCallerGetsTheWorldOfBeforeTests(unittest.TestCase):
    def test_the_librarys_own_tuning_leaves_them_off(self) -> None:
        tuning = WorldBuildTuning()
        self.assertFalse(tuning.support_surfaces)
        self.assertEqual(tuning.support_allowance_mm, 0.0)

    def test_the_world_is_byte_for_byte_what_it_was(self) -> None:
        for name in cell.LOOKS:
            look = cell.look(name)
            tuning = WorldBuildTuning(pixel_stride=2)
            target = target_keep_out_box(look.target_points_base_mm, name="target", limits=cell.OWNER_LIMITS,
                                         tuning=tuning)
            for held in (False, True):
                with self.subTest(look=name, target_held=held):
                    world = build_perceived_boxes(views=[cell.depth_view(look)], limits=cell.OWNER_LIMITS,
                                                  tuning=tuning, near_point_mm=(-165.0, -679.0, 300.0),
                                                  keep_out=(target,) if held and target is not None else ())
                    text = json.dumps(_canon(world.to_dict()), sort_keys=True).encode()
                    self.assertEqual(hashlib.sha256(text).hexdigest()[:16], _BEFORE[(name, held)])
                    self.assertIsNone(world.supports)


if __name__ == "__main__":
    unittest.main()
