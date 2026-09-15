"""The build log said the camera world is read before every plan. On an `ik` cell nothing plans.

`_wire_live_planner_world` hands the arm a world and logged "refreshed before every plan" whatever the
planner. The UR arm reads that world only inside its cuRobo planner, so on `motion_planner: ik` the line
announced a protection that never runs, which is exactly the kind of sentence this library may not print.

A stereo rig is told at the same place (Step 4e). It has no depth of its own, so a world built on it would
answer no frame to every refresh and every planned motion would raise; the cell gets no world and a
warning that names the rig instead.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.config.schema.robot import RobotConfig
from src.camera.setup.image_taking.frames import StereoFrame
from src.robot.execution.autonomous_grasp import cells

_LIVE = "src.robot.execution.autonomous_grasp.live_world"


def _cell(planner: str):
    cfg = RobotConfig.model_validate({
        "vendor": "ur", "ur": {"motion_planner": planner}, "safety": {"payload": {"enforce": False}},
    })
    arm = SimpleNamespace(set_live_planner_world=MagicMock())
    orchestrator = SimpleNamespace(arm=arm, frame_resolver=object())
    service = SimpleNamespace(runtime=SimpleNamespace(orchestrator=orchestrator))
    # An RGB-D rig: the wiring grabs one frame to tell a stereo pair from a camera with depth.
    streamer = SimpleNamespace(rig_id="realsense_d435", grab=lambda: SimpleNamespace(depth=np.ones((4, 4))))
    perception = SimpleNamespace(streamer=streamer)
    return cfg, service, perception, arm


class TheWiringLogNamesThePlannerTests(unittest.TestCase):
    def _wire(self, planner: str, *, frame: object = None) -> tuple[str, SimpleNamespace]:
        cfg, service, perception, arm = _cell(planner)
        if frame is not None:
            perception.streamer.grab = lambda: frame
        world = SimpleNamespace(cameras=("realsense_d435",))
        with (
            patch(f"{_LIVE}.static_camera_to_base_mm", return_value=np.eye(4)),
            patch(f"{_LIVE}.RigDepthSource", side_effect=lambda streamer: streamer),
            patch(f"{_LIVE}.build_live_planner_world", return_value=world),
            self.assertLogs(cells.__name__, level="INFO") as logs,
        ):
            cells._wire_live_planner_world(cfg, service, perception)
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

        text, arm = self._wire("curobo", frame=StereoFrame(left=pair, right=pair))

        arm.set_live_planner_world.assert_not_called()
        self.assertIn("stereo", text)
        self.assertIn("realsense_d435", text)
        self.assertNotIn("refreshed before every plan", text)


if __name__ == "__main__":
    unittest.main()
