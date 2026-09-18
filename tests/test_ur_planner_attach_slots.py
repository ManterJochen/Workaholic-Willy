"""The attach slots a UR planner reserves are the ones its evidence lookup names: one number, from the reservation.

Step 8e (O1 A). The planner used to enable its attach spheres from ``payload.enabled`` alone while the evidence lookup
and the desk read the reservation, which counted them only inside an enabled planning world. A UR with the world off and
the payload on then reserved sixteen spheres and looked up a file measured with none: the composed hash differs and
the planner is refused. Both read the reservation now.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

from src.config.schema.robot import RobotConfig
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.safety.planning.reservation import PlannerReservation


def _arm(*, world: bool, length_mm: "float | None") -> URRobotArm:
    payload: dict[str, Any] = {"enabled": True, "length_mm": length_mm}
    arm = URRobotArm(RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"motion_planner": "curobo"},
        "gripper": {"model": "robotiq_2f85"},
        "safety": {
            "payload": {"enforce": False},
            "self_collision": {"planner_margin_mm": 4.0},
            "planning_world": {
                "enabled": world,
                "support_plane": {"height_mm": 0.0, "extent_mm": [1600.0, 1600.0], "thickness_mm": 50.0},
                "payload": payload,
            },
        },
    }))
    arm._conn = MagicMock()
    arm._curobo_client_factory = lambda: MagicMock()  # type: ignore[assignment]
    return arm


class TheAttachSlotsAreOneNumberTests(unittest.TestCase):
    def test_the_attach_slots_the_planner_reserves_are_the_ones_it_looks_up(self) -> None:
        for world in (False, True):
            for length_mm, slots in ((60.0, 16), (None, 0)):
                with self.subTest(world=world, length_mm=length_mm):
                    arm = _arm(world=world, length_mm=length_mm)
                    looked_up = PlannerReservation.from_config(robot_cfg=arm.config).sphere_slots

                    planner = arm._curobo_ur_planner()

                    self.assertEqual(slots, looked_up)
                    self.assertEqual(looked_up, planner._attach_spheres)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
