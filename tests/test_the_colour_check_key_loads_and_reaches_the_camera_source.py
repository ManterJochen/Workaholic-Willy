"""The colour check's key loads and reaches the camera source a real cell builds (``robot.grasping.colour_check``).

``on`` is the owner's choice and the default (2026-10-08): the camera source judges a part's colour on its pixels and
a part of another colour is no target. ``log`` judges and logs and changes nothing, ``off`` judges nothing. YAML reads a
bare ``on`` and ``off`` as booleans, so the two booleans are the two words. The build hands the key to the cell's
primary source; without that wire a cell would report the switch and not have it.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np
from pydantic import ValidationError

from src.config.schema.robot import RobotConfig
from src.config.schema.robot.grasping_schema import RobotGraspingConfig


class TheKeyLoadsTests(unittest.TestCase):
    def test_the_check_is_on_unless_a_cell_says_otherwise(self) -> None:
        self.assertEqual("on", RobotGraspingConfig().colour_check)
        self.assertEqual("on", RobotConfig().grasping.colour_check)

    def test_the_three_words_load_and_a_bare_yaml_switch_is_the_word(self) -> None:
        for value, loaded in (("on", "on"), ("log", "log"), ("off", "off"), (True, "on"), (False, "off")):
            with self.subTest(value=value):
                self.assertEqual(loaded, RobotGraspingConfig(colour_check=value).colour_check)

    def test_any_other_word_is_refused_at_load(self) -> None:
        for value in ("sometimes", "ON", "", 1):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                RobotGraspingConfig(colour_check=value)


class _Handle:
    def get_intrinsics(self) -> np.ndarray:
        return np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 180.0], [0.0, 0.0, 1.0]])


class _Spec:
    @classmethod
    def from_config(cls, models: Any) -> "_Spec":
        return cls()

    def build(self) -> str:
        return "backend"


class TheBuildHandsItToTheSourceTests(unittest.TestCase):
    def _built_with(self, value: str, clipped: str = "exclude") -> dict[str, Any]:
        from src.robot.execution.autonomous_grasp.cells import _build_on_open_cameras

        handed: dict[str, Any] = {}

        def source(**keywords: Any) -> str:
            handed.update(keywords)
            return "source"

        robot = RobotConfig(grasping=RobotGraspingConfig(colour_check=value, colour_check_clipped=clipped))
        _build_on_open_cameras(
            robot, SimpleNamespace(models=None), prompt="object",
            cameras={"overhead": SimpleNamespace(handle=_Handle)}, rig_id="overhead", np=np, perception_spec=_Spec,
            build_calculator=lambda *_args, **_keywords: SimpleNamespace(), vision_source=source)
        return handed

    def test_the_cell_s_primary_source_is_built_with_the_key(self) -> None:
        for value in ("on", "log", "off"):
            with self.subTest(value=value):
                handed = self._built_with(value)
                self.assertEqual(value, handed["colour_check"])
                self.assertEqual(("backend", "object"), (handed["backend"], handed["prompt"]))

    def test_whether_clipped_pixels_count_reaches_the_source_too(self) -> None:
        """``robot.grasping.colour_check_clipped``: ``exclude`` as shipped (the owner, 2026-10-09), ``keep`` as before."""
        for clipped in ("exclude", "keep"):
            with self.subTest(clipped=clipped):
                self.assertEqual(clipped, self._built_with("on", clipped)["colour_check_clipped"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
