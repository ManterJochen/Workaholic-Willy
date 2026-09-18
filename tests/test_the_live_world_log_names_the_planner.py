"""The build log said the camera world is read before every plan. On an `ik` cell nothing plans.

`_wire_live_planner_world` hands the arm a world and logged "refreshed before every plan" whatever the
planner. The UR arm reads that world only inside its cuRobo planner, so on `motion_planner: ik` the line
announced a protection that never runs, which is exactly the kind of sentence this library may not print.

A stereo rig is told at the same place (Step 4e). It has no depth of its own, so a world built on it would
answer no frame to every refresh and every planned motion would raise. An ik cell gets no world and a
warning that names the rig; a cuRobo cell, whose every motion would then be refused, does not build (an
owner decision, held in tests/test_camera_world_wiring.py).

The sentences are `CameraWorldWiring.render()`, logged once by the cell. The rig here is a calibrated
RGB-D rig of a cell that enables the world, and for the stereo row its owner answers a real `StereoFrame`.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.camera.setup.image_taking.frames import StereoFrame
from src.geometry import Frame, Transform
from src.robot.execution.autonomous_grasp import cells

_RIG = "realsense_d435"


def _cell(planner: str) -> RobotConfig:
    return RobotConfig.model_validate({
        "vendor": "ur", "ur": {"motion_planner": planner},
        "safety": {"payload": {"enforce": False}, "planning_world": {
            "enabled": True,
            "support_plane": {"height_mm": 0.0, "extent_mm": [1600.0, 1600.0], "thickness_mm": 50.0},
        }},
    })


class _Owner:
    """The rig's open camera: its handle answers ``frame``, and its calibration is a fixed CAMERA to BASE."""

    def __init__(self, frame: object) -> None:
        self.rig_id = _RIG
        self._frame = frame

    def handle(self) -> SimpleNamespace:
        return SimpleNamespace(rig_id=_RIG, grab=lambda: self._frame, get_intrinsics=lambda: np.eye(3))

    def calibration(self) -> SimpleNamespace:
        return SimpleNamespace(
            mounting_mode="eye_to_hand",
            camera_to_base=lambda: Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE),
        )


class TheWiringLogNamesThePlannerTests(unittest.TestCase):
    def _wire(self, planner: str, *, frame: object = None) -> tuple[str, SimpleNamespace]:
        arm = SimpleNamespace(set_live_planner_world=MagicMock())
        service = SimpleNamespace(runtime=SimpleNamespace(orchestrator=SimpleNamespace(arm=arm)))
        # An RGB-D rig: the wiring grabs one frame to tell a stereo pair from a camera with depth.
        owner = _Owner(SimpleNamespace(depth=np.ones((4, 4))) if frame is None else frame)
        perception = SimpleNamespace(streamer=SimpleNamespace(camera=owner))
        rig = SimpleNamespace(rig_id=_RIG, enabled=True, source="rgbd",
                              extrinsics=SimpleNamespace(mounting_mode="eye_to_hand"))
        app = SimpleNamespace(camera=SimpleNamespace(cameras=SimpleNamespace(rigs=[rig], primary_rig_id=_RIG)))
        with self.assertLogs(cells.__name__, level="INFO") as logs:
            cells._wire_live_planner_world(_cell(planner), service, perception, app_cfg=app)
        return "\n".join(logs.output), arm

    def test_an_ik_cell_is_not_told_its_world_is_read_before_every_plan(self) -> None:
        text, arm = self._wire("ik")
        arm.set_live_planner_world.assert_called_once()
        self.assertNotIn("refreshed before every plan", text)
        self.assertIn("'ik'", text, "the log should name the planner that makes the world unread")

    def test_a_curobo_cell_still_says_so(self) -> None:
        """The control: the sentence is true there, and removing it everywhere would pass the test above."""
        text, arm = self._wire("curobo")
        arm.set_live_planner_world.assert_called_once()
        self.assertIn("refreshed before every plan", text)

    def test_a_stereo_rig_wires_no_world_and_says_why(self) -> None:
        pair = np.zeros((4, 4, 3), dtype=np.uint8)

        text, arm = self._wire("ik", frame=StereoFrame(left=pair, right=pair))

        arm.set_live_planner_world.assert_not_called()
        self.assertIn("stereo", text)
        self.assertIn("realsense_d435", text)
        self.assertNotIn("refreshed before every plan", text)


if __name__ == "__main__":
    unittest.main()
