"""Every cell's camera source judges a part's colour (the owner, 2026-10-08).

``on`` was the owner's choice and the default of ``robot.grasping.colour_check`` (2026-10-08): the camera source judges
a part's colour on its pixels and a part of another colour is no target. The key left on 2026-10-17 with its ``log``
and its twin ``colour_check_clipped`` (the owner's cleanup; the refusal is pinned in
``tests/robot/safety/planning/test_the_cell_fix_keys_load_and_refuse.py``): a cell builds its primary source as the source stands, judging.
"""

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.perception import RealSenseVisionPerceptionSource
from src.robot.perception.colour_check import ColourCheck


class _Handle:
    def get_intrinsics(self) -> np.ndarray:
        return np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 180.0], [0.0, 0.0, 1.0]])


class _Spec:
    @classmethod
    def from_config(cls, models: Any) -> "_Spec":
        return cls()

    def build(self) -> str:
        return "backend"


class TheCheckTests(unittest.TestCase):
    def test_a_source_judges_unless_it_is_told_not_to_and_there_is_no_third_way(self) -> None:
        default = inspect.signature(RealSenseVisionPerceptionSource).parameters["colour_check"].default
        self.assertIs(ColourCheck.ON, default)
        self.assertEqual({"on", "off"}, {member.value for member in ColourCheck})


class TheBuildTests(unittest.TestCase):
    def test_the_cells_primary_source_is_built_judging(self) -> None:
        from src.robot.execution.autonomous_grasp.cells import _build_on_open_cameras

        handed: dict[str, Any] = {}

        def source(**keywords: Any) -> str:
            handed.update(keywords)
            return "source"

        # A camera section that declares no other camera: the cell fuses none, so only the primary's source is built.
        app = SimpleNamespace(models=None, camera=SimpleNamespace(cameras=SimpleNamespace(rigs=(), primary_rig_id="overhead")))
        _build_on_open_cameras(
            RobotConfig(), app, prompt="object",
            cameras={"overhead": SimpleNamespace(handle=_Handle)}, rig_id="overhead", np=np, perception_spec=_Spec,
            build_calculator=lambda *_args, **_keywords: SimpleNamespace(), vision_source=source)

        self.assertEqual(("backend", "object"), (handed["backend"], handed["prompt"]))
        self.assertNotIn("colour_check", handed, "the source judges as it stands")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
