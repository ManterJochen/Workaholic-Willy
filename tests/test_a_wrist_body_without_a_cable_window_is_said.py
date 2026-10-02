"""A wrist camera on an arm whose joints may wind its cable is said once, at build.

The owner's cell (fix plan S3): wrist 3 was driven 200 degrees from home with the camera's cable over the wrist.
``safety.joint_limits.within_half_turn_of_home`` keeps every full-turn joint within half a turn of home, so a cable run
along the arm is never wound further; it is off by default. A cell that carries a wrist body and leaves it off gets one
WARNING when its bodies are resolved, naming the camera and the key. It refuses nothing: side approaches are on by the
owner's decision and some of their lines need the other wrist branch, so the window comes on after the branch check.
"""

from __future__ import annotations

import logging
import unittest

from tests._wrist_body import WristOwner, wrist_cell

_LOGGER = "src.robot.execution.wrist_bodies"


class TheCableWindowIsSaidTests(unittest.TestCase):
    def test_a_wrist_body_with_the_window_off_is_one_warning(self) -> None:
        from src.robot.execution.wrist_bodies import WristBodies

        with self.assertLogs(_LOGGER, level=logging.WARNING) as said:
            bodies = WristBodies.from_owners(wrist_cell(), [WristOwner()])
        self.assertEqual(1, len(bodies.bodies))
        lines = [line for line in said.output if "within_half_turn_of_home" in line]
        self.assertEqual(1, len(lines), said.output)
        self.assertIn(bodies.bodies[0].link_name, lines[0])
        self.assertIn("nothing is refused", lines[0])

    def test_with_the_window_on_nothing_is_said(self) -> None:
        from src.robot.execution.wrist_bodies import WristBodies

        cell = wrist_cell()
        limits = cell.safety.joint_limits.model_copy(update={"within_half_turn_of_home": True})
        cell = cell.model_copy(update={"safety": cell.safety.model_copy(update={"joint_limits": limits})})
        logger = logging.getLogger(_LOGGER)
        with self.assertNoLogs(logger, level=logging.WARNING):
            bodies = WristBodies.from_owners(cell, [WristOwner()])
        self.assertEqual(1, len(bodies.bodies))

    def test_an_arm_that_carries_no_camera_is_told_nothing(self) -> None:
        from src.robot.execution.wrist_bodies import WristBodies

        logger = logging.getLogger(_LOGGER)
        with self.assertNoLogs(logger, level=logging.WARNING):
            bodies = WristBodies.from_owners(wrist_cell(), [])
        self.assertEqual((), bodies.bodies)


if __name__ == "__main__":
    unittest.main()
