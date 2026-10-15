"""A tool reaches a planned arm after its build by one path, mount_tool, and never while the arm is connected (P0i i5).

Owner decision 5 (2026-09-26): one ``mount_tool()`` path at build and connect, and a second mount while the cell is
connected refused with the reason "a tool change is not built yet"; a tool change stays a later function. On an arm
that is not connected the guard models the new hand, refused as its build refuses it, a planner built for the old hand
is stopped, and the next connect pushes the new tool's payload.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any

from src.contracts import UNSET
from src.robot.safety.planning.mounted_tool import MountedTool, ToolChangeRefused
from tests._arm_kit import kit_tree, planned_arm
from tests._fake_transport import chain_of, fake_transport

_OFFSET = (0.0, 132.0, 0.0)
_QUAT = (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
_Q = [0.0, -1.2, 1.3, -0.4, 1.5, 0.2]


def _tree(hand: str = "robotiq_2f85", **gripper: Any) -> Any:
    return kit_tree("ik", safety={"payload": {"enforce": True, "mass_kg": 1.5, "cog_mm": (0.0, 0.0, 60.0)},
                                  "self_collision": {"backend": "fcl"}},
                    gripper={"model": hand, "tool_frame": {"source": "willy", "offset_mm": _OFFSET,
                                                          "rotation_quat_xyzw": _QUAT}, **gripper})


def _arm(*, connected: bool) -> tuple[Any, Any]:
    fake = fake_transport(joints=_Q, kinematics=chain_of("ur5e"))
    fake.connected = connected
    return planned_arm(_tree(), transport=fake), fake


def _heavier(arm: Any) -> MountedTool:
    tool = arm.mounted_tool
    return replace(tool, payload=tool.payload.model_copy(update={"mass_kg": 2.25}))


class AToolIsMountedByOnePathTests(unittest.TestCase):
    def test_a_tool_is_not_changed_on_a_connected_arm(self) -> None:
        arm, _fake = _arm(connected=True)
        before = arm.mounted_tool
        with self.assertRaises(ToolChangeRefused) as refused:
            arm.mount_tool(_heavier(arm))
        self.assertTrue(str(refused.exception).startswith("a tool change is not built yet"))
        self.assertIs(arm.mounted_tool, before)

    def test_a_tool_mounted_on_an_arm_not_connected_is_the_one_the_next_connect_pushes(self) -> None:
        arm, fake = _arm(connected=False)
        heavier = _heavier(arm)
        arm.mount_tool(heavier)
        self.assertIs(arm.mounted_tool, heavier)
        arm.connect()
        self.assertEqual([call.args for call in fake.calls("set_payload")], [(2.25, [0.0, 0.0, 60.0])])

    def test_the_guard_models_the_mounted_hand(self) -> None:
        arm, _fake = _arm(connected=False)
        hande = MountedTool.from_config(_tree("robotiq_hande",
                                              coupling_plates=[{"name": "hande_adapter", "thickness_mm": 20.0}]))
        arm.mount_tool(hande)
        self.assertEqual(arm.safety_preflight.planner_hand(arm).model, "robotiq_hande")

    def test_the_mounted_hand_stands_fully_open_until_it_is_told_a_width(self) -> None:
        """The opening the old hand was told (lever G, 2026-10-13) is not the new hand's: its fingers stand as
        baked, fully open, the wider side."""
        arm, _fake = _arm(connected=False)
        guard = next(g for g in arm.safety_preflight.guards if g.name == "self_collision")
        guard.set_finger_opening(40.0)
        arm.mount_tool(MountedTool.from_config(_tree("robotiq_hande",
                                                     coupling_plates=[{"name": "hande_adapter", "thickness_mm": 20.0}])))
        self.assertIsNone(guard.finger_opening_mm)

    def test_a_planner_built_for_the_old_hand_is_stopped(self) -> None:
        arm, _fake = _arm(connected=False)
        stopped: list[str] = []
        arm.stop_planner = lambda: stopped.append("stop")  # type: ignore[method-assign]
        arm.mount_tool(_heavier(arm))
        self.assertEqual(stopped, ["stop"])

    def test_a_hand_the_build_would_refuse_is_refused_and_nothing_changes(self) -> None:
        """A guard reading hand geometry refuses a hand nobody named, at a mount with the build's own sentence."""
        arm, _fake = _arm(connected=False)
        before = arm.mounted_tool
        with self.assertRaises(ToolChangeRefused) as refused:
            arm.mount_tool(replace(before, hand=UNSET))
        self.assertIn("robot.gripper.model is unset", str(refused.exception))
        self.assertIs(arm.mounted_tool, before)
        self.assertEqual(arm.safety_preflight.planner_hand(arm).model, "robotiq_2f85")


if __name__ == "__main__":
    unittest.main()
