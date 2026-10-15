"""A tool mounted on a planned arm is what every guard of that arm judges, or the mount is refused (P0i i5 review).

The review of 2026-09-26 found four readers ``mount_tool`` did not reach: the payload guard kept the tree's payload, so
a mount of 50 kg on a 5 kg envelope was accepted and pushed at the next connect; the wrist cameras the arm holds were
checked against the tool frame they were calibrated with only when they were handed in; a hand placed by one tool frame
could be mounted with another; and a capsule guard models the tool from the tree's dimensions, which no mount changes.
Each now follows the mount or refuses it, with nothing changed where it refuses. The planner is stopped before anything
changes, and a tool frame a controller reported for the old tool is forgotten.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any

import numpy as np

from src.geometry.matrix import position_quaternion_to_matrix
from src.robot.safety.planning.mounted_tool import MountedTool, ToolChangeRefused
from tests._arm_kit import kit_tree, planned_arm
from tests._fake_transport import chain_of, fake_transport

_OFFSET = (0.0, 132.0, 0.0)
_QUAT = (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
_Q = [0.0, -1.2, 1.3, -0.4, 1.5, 0.2]


def _tree(*, backend: str = "fcl", enforce: bool = True, hand: str = "robotiq_2f85", source: str = "willy") -> Any:
    return kit_tree("ik", safety={"payload": {"enforce": enforce, "mass_kg": 1.5, "cog_mm": (0.0, 0.0, 60.0)},
                                  "self_collision": {"backend": backend}},
                    gripper={"model": hand, "tool_frame": {"source": source, "offset_mm": _OFFSET,
                                                          "rotation_quat_xyzw": _QUAT},
                             # The Hand-E's sphere map starts at its own mounting face, so its plate is measured.
                             **({"coupling_plates": [{"name": "hande_adapter", "thickness_mm": 20.0}]}
                                if hand == "robotiq_hande" else {})})


def _arm(config: Any) -> tuple[Any, Any]:
    # On `controller` the TCP register holds the declared frame, set on the pendant; on `willy` it is a bare flange.
    on_controller = config.gripper.tool_frame.source == "controller"
    register = position_quaternion_to_matrix(np.asarray(_OFFSET), np.asarray(_QUAT)) if on_controller else None
    fake = fake_transport(joints=_Q, kinematics=chain_of("ur5e"), tcp_register=register)
    fake.connected = False
    return planned_arm(config, transport=fake), fake


def _payload_guard(arm: Any) -> Any:
    return next(guard for guard in arm.safety_preflight.guards if guard.name == "payload")


class _Body:
    """A wrist camera body as the arm asks it: whether the flange to TCP it was placed with is the frame given."""

    def __init__(self, record: np.ndarray) -> None:
        self._record = np.asarray(record, dtype=float)

    def record_refusal(self, source: str, frame: Any) -> str | None:
        if frame is not None and np.allclose(np.asarray(frame, dtype=float), self._record):
            return None
        return "wrist camera 'wrist': its flange to TCP record is not the frame this cell applies now"

    def envelope_spheres_mm(self) -> list[Any]:
        return []


class AMountedToolIsWhatEveryGuardJudgesTests(unittest.TestCase):
    def test_the_payload_guard_judges_the_mounted_payload(self) -> None:
        arm, _fake = _arm(_tree())
        heavier = replace(arm.mounted_tool, payload=arm.mounted_tool.payload.model_copy(update={"mass_kg": 4.0}))
        arm.mount_tool(heavier)
        self.assertIs(_payload_guard(arm).payload, heavier.payload)

    def test_a_payload_the_tree_could_not_declare_is_refused_and_nothing_changes(self) -> None:
        arm, fake = _arm(_tree())
        before = arm.mounted_tool
        over = replace(before, payload=before.payload.model_copy(update={"mass_kg": 50.0}))
        with self.assertRaises(ToolChangeRefused) as refused:
            arm.mount_tool(over)
        self.assertIn("max_mass_kg", str(refused.exception))
        self.assertIs(arm.mounted_tool, before)
        self.assertIs(_payload_guard(arm).payload, before.payload)
        arm.connect()
        self.assertEqual([call.args for call in fake.calls("set_payload")], [(1.5, [0.0, 0.0, 60.0])])

    def test_a_mount_keeps_whether_the_payload_is_guarded(self) -> None:
        arm, _fake = _arm(_tree(enforce=False))
        enforced = replace(arm.mounted_tool, payload=arm.mounted_tool.payload.model_copy(update={"enforce": True}))
        with self.assertRaises(ToolChangeRefused) as refused:
            arm.mount_tool(enforced)
        self.assertIn("safety.payload.enforce", str(refused.exception))
        self.assertFalse(arm.mounted_tool.payload.enforce)

    def test_a_frame_the_wrist_cameras_were_not_placed_with_is_refused(self) -> None:
        arm, _fake = _arm(_tree())
        arm.set_wrist_bodies([_Body(arm._declared_tool_matrix())])
        before = arm.mounted_tool
        shorter = replace(before, tool_frame=before.tool_frame.model_copy(update={"offset_mm": (0.0, 100.0, 0.0)}))
        with self.assertRaises(ToolChangeRefused) as refused:
            arm.mount_tool(shorter)
        self.assertIn("flange to TCP record", str(refused.exception))
        self.assertIs(arm.mounted_tool, before)

    def test_the_same_frame_passes_the_wrist_cameras(self) -> None:
        arm, _fake = _arm(_tree())
        arm.set_wrist_bodies([_Body(arm._declared_tool_matrix())])
        heavier = replace(arm.mounted_tool, payload=arm.mounted_tool.payload.model_copy(update={"mass_kg": 2.0}))
        arm.mount_tool(heavier)
        self.assertIs(arm.mounted_tool, heavier)

    def test_a_hand_placed_by_another_frame_is_not_a_tool(self) -> None:
        tool = MountedTool.from_config(_tree())
        turned = tool.tool_frame.model_copy(update={"rotation_quat_xyzw": (0.0, 0.0, 0.0, 1.0)})
        with self.assertRaises(ValueError) as refused:
            replace(tool, tool_frame=turned)
        self.assertIn("another tool frame", str(refused.exception))

    def test_a_capsule_guard_is_not_handed_another_hand(self) -> None:
        arm, _fake = _arm(_tree(backend="capsule"))
        before = arm.mounted_tool
        with self.assertRaises(ToolChangeRefused) as refused:
            arm.mount_tool(MountedTool.from_config(_tree(backend="capsule", hand="robotiq_hande")))
        self.assertIn("capsule", str(refused.exception))
        self.assertIs(arm.mounted_tool, before)

    def test_a_capsule_guard_takes_the_same_hand_with_another_payload(self) -> None:
        arm, _fake = _arm(_tree(backend="capsule"))
        heavier = replace(arm.mounted_tool, payload=arm.mounted_tool.payload.model_copy(update={"mass_kg": 2.0}))
        arm.mount_tool(heavier)
        self.assertIs(arm.mounted_tool, heavier)

    def test_the_planner_is_stopped_before_anything_changes(self) -> None:
        arm, _fake = _arm(_tree())
        before = arm.mounted_tool
        hand_before = arm.safety_preflight.planner_hand(arm)

        def refuses() -> None:
            raise RuntimeError("the sidecar did not stop")

        arm.stop_planner = refuses  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            arm.mount_tool(MountedTool.from_config(_tree(hand="robotiq_hande")))
        self.assertIs(arm.mounted_tool, before)
        self.assertIs(arm.safety_preflight.planner_hand(arm), hand_before)

    def test_a_tool_frame_the_controller_reported_for_the_old_tool_is_forgotten(self) -> None:
        arm, fake = _arm(_tree(source="controller"))
        arm.connect()
        self.assertIsNotNone(arm._observed_tool_frame)
        fake.connected = False  # the link dropped, and nobody called disconnect()
        arm.mount_tool(replace(arm.mounted_tool,
                               payload=arm.mounted_tool.payload.model_copy(update={"mass_kg": 2.0})))
        self.assertIsNone(arm._observed_tool_frame)


if __name__ == "__main__":
    unittest.main()
