"""A declared tool frame approaches along its own offset, on every profile that drives a UR.

The `ursim` profile declared the grasp centre 132 mm along flange +Z, which is the UR tool axis, and then
the Isaac rotation (-90 deg about flange X), which turns the approach onto flange +Y. Each half came from
a different convention. Every Cartesian command composes that frame now, where once only `move_linear`
did, so on URSim a top-down target was reached with the flange turned 90 degrees off its tool axis. The
owner set the rotation to identity on 2026-09-11.

Derived rather than listed: every profile layer is loaded, and every one that ends on the UR driver with
a declared tool frame is checked. A tool that is genuinely mounted off its axis states its measured angle
here instead of loosening the check for everyone.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.config import load_robot_config
from src.config.loader import available_profiles
from src.robot.drivers.ur.tool_frame import tool_frame_matrix

#: How far a declared approach axis may lean away from the tool offset before it is two conventions.
_MAX_ANGLE_DEG = 1.0


class EveryDeclaredUrToolFrameApproachesAlongItsOffsetTests(unittest.TestCase):
    def test_the_approach_axis_and_the_tool_offset_agree(self) -> None:
        checked: list[str | None] = []
        for layer in (None, *available_profiles()):
            robot = load_robot_config(profile=layer)
            frame = robot.gripper.tool_frame
            if robot.vendor != "ur" or frame.source == "undeclared":
                continue
            offset = np.asarray(frame.offset_mm, dtype=np.float64)
            approach = tool_frame_matrix(frame.offset_mm, frame.rotation_quat_xyzw)[:3, 2]
            cosine = float(np.dot(approach, offset / np.linalg.norm(offset)))
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            checked.append(layer)
            with self.subTest(profile=layer):
                self.assertLess(
                    angle, _MAX_ANGLE_DEG,
                    f"profile {layer!r}: the grasp centre sits along {frame.offset_mm} from the flange, "
                    f"but the declared rotation points the approach {angle:.1f} deg away from it",
                )
        self.assertIn("ursim", checked, "the check reached no profile that drives a UR with a declared frame")


if __name__ == "__main__":
    unittest.main()
