"""The build log said the camera world is read before every plan. On an `ik` cell nothing plans.

`_wire_live_planner_world` hands the arm a world and logged "refreshed before every plan" whatever the
planner. The UR arm reads that world only inside its cuRobo planner, so on `motion_planner: ik` the line
announced a protection that never runs, which is exactly the kind of sentence this library may not print.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.execution.autonomous_grasp import cells

_LIVE = "src.robot.execution.autonomous_grasp.live_world"


def _cell(planner: str):
    cfg = RobotConfig.model_validate({
        "vendor": "ur", "ur": {"motion_planner": planner}, "safety": {"payload": {"enforce": False}},
    })
    arm = SimpleNamespace(set_live_planner_world=MagicMock())
    orchestrator = SimpleNamespace(arm=arm, frame_resolver=object())
    service = SimpleNamespace(runtime=SimpleNamespace(orchestrator=orchestrator))
    perception = SimpleNamespace(streamer=SimpleNamespace(rig_id="realsense_d435"))
    return cfg, service, perception, arm


class TheWiringLogNamesThePlannerTests(unittest.TestCase):
    def _wire(self, planner: str) -> tuple[str, SimpleNamespace]:
        cfg, service, perception, arm = _cell(planner)
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


if __name__ == "__main__":
    unittest.main()
