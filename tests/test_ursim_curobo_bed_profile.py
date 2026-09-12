"""The URSim bed can run with the planner on, and the layer that turns it on is one file.

`ursim` exercises the driver against a real UR controller and plans nothing, because when it was
written nothing planned. Step 3 made every verb on a cuRobo cell either planned or path checked, and
that is the half no bed had measured: a real controller answering a real sidecar, with no hardware in
the room.

The layers compose rather than repeat. `ursim,ursim_curobo` is the UR5e bed;
`ursim,ursim_curobo,ursim_ur3` is the same bed on the smaller arm, which needs its own planner margin
because a UR3e finds no plan at the UR5e figure.
"""

from __future__ import annotations

import unittest

from src.config import load_robot_config


def _robot(profile: str):
    return load_robot_config(profile=profile)


class TheBedLayersComposeTests(unittest.TestCase):
    def test_ursim_alone_still_plans_nothing(self) -> None:
        """The control. Without it the two below could be describing the base tree."""
        robot = _robot("ursim")
        self.assertEqual("ik", robot.ur.motion_planner)
        self.assertEqual("ur5e", robot.ur.model)

    def test_the_ur5e_bed_turns_the_planner_on(self) -> None:
        robot = _robot("ursim,ursim_curobo")
        self.assertEqual("ur", robot.vendor)
        self.assertEqual("127.0.0.1", robot.ur.ip)
        self.assertEqual("curobo", robot.ur.motion_planner)
        self.assertEqual(10.0, robot.safety.self_collision.planner_margin_mm)

    def test_the_ur3e_bed_is_the_same_bed_on_the_smaller_arm(self) -> None:
        robot = _robot("ursim,ursim_curobo,ursim_ur3")
        self.assertEqual("ur3e", robot.ur.model)
        self.assertEqual("curobo", robot.ur.motion_planner)
        self.assertEqual(
            4.0, robot.safety.self_collision.planner_margin_mm,
            "a UR3e finds no plan at the UR5e margin, so the layer order has to end here",
        )

    def test_the_ur3_layer_over_plain_ursim_is_unchanged_in_what_matters(self) -> None:
        """Its margin is inert without the planner, and that is why it can sit in both chains."""
        robot = _robot("ursim,ursim_ur3")
        self.assertEqual("ur3e", robot.ur.model)
        self.assertEqual("ik", robot.ur.motion_planner)


if __name__ == "__main__":
    unittest.main()
