"""A planned arm reads its tool frame and its payload from the tool mounted on it, and from nothing else (P0i i5).

The hand was fixed in four places, and the connect read the tool frame and the payload straight from the tree. A
``MountedTool`` (``safety.planning.mounted_tool``) now holds the hand, the tool frame and the payload as one value,
``MountedTool.from_config`` reads it from the tree, the UR driver builds its arm with it, and the arm pushes, checks and
composes what it holds. So a tool that differs from the tree is the one the controller hears of.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any

import numpy as np

from src.robot.safety.planning.mounted_tool import MountedTool
from tests._arm_kit import kit_tree, planned_arm
from tests._fake_transport import chain_of, fake_transport

_OFFSET = (0.0, 132.0, 0.0)
_QUAT = (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
_Q = [0.0, -1.2, 1.3, -0.4, 1.5, 0.2]


def _tree(**payload: Any) -> Any:
    return kit_tree("ik", safety={"payload": {"enforce": True, "mass_kg": 1.5, "cog_mm": (0.0, 0.0, 60.0), **payload}},
                    gripper={"tool_frame": {"source": "willy", "offset_mm": _OFFSET, "rotation_quat_xyzw": _QUAT}})


def _fake() -> Any:
    fake = fake_transport(joints=_Q, kinematics=chain_of("ur5e"))
    fake.connected = False
    return fake


class TheArmReadsItsMountedToolTests(unittest.TestCase):
    def test_the_connect_pushes_the_payload_of_the_tool_it_holds(self) -> None:
        config = _tree()
        heavier = replace(MountedTool.from_config(config),
                          payload=config.safety.payload.model_copy(update={"mass_kg": 2.25}))
        fake = _fake()
        arm = planned_arm(config, transport=fake, tool=heavier)
        arm.connect()
        self.assertEqual([call.args for call in fake.calls("set_payload")], [(2.25, [0.0, 0.0, 60.0])])
        self.assertIs(arm.mounted_tool, heavier)

    def test_the_tool_frame_the_arm_composes_is_the_one_it_holds(self) -> None:
        config = _tree()
        shorter = replace(MountedTool.from_config(config),
                          tool_frame=config.gripper.tool_frame.model_copy(update={"offset_mm": (0.0, 100.0, 0.0)}))
        arm = planned_arm(config, transport=_fake(), tool=shorter)
        declared = arm._declared_tool_matrix()
        self.assertIsNotNone(declared)
        np.testing.assert_allclose(np.asarray(declared)[:3, 3], [0.0, 100.0, 0.0])

    def test_the_tool_says_what_it_is(self) -> None:
        import json

        tool = MountedTool.from_config(_tree())
        self.assertEqual(tool.render(), "tool  robotiq_2f85, tool frame willy (0.0, 132.0, 0.0) mm, 1.5 kg at "
                                        "(0.0, 0.0, 60.0) mm")
        self.assertEqual(json.loads(json.dumps(tool.to_dict()))["hand"], "robotiq_2f85")


if __name__ == "__main__":
    unittest.main()
