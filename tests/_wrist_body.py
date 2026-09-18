"""Shared doubles: a UR cuRobo cell with a declared tool frame, and a wrist camera that declares its body.

The rig and its calibration are what ``execution.wrist_bodies`` reads: a rig with ``body`` and ``extrinsics``, and an
owner whose calibration answers ``camera_to_tool()``, ``flange_to_tcp`` and ``artifact_path`` as ``RigCalibration``
does. The world half comes from tests/test_camera_world_wiring.py, so a robot built with this owner also builds a live
world.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from src.config.schema.robot import RobotConfig
from src.calibration.serialization import FlangeToTcp
from src.contracts import UNSET
from src.robot.drivers.ur.tool_frame import tool_frame_matrix

#: The shipped 2F-85 frame, which places the hand on a UR flange.
TOOL_OFFSET_MM = (0.0, 132.0, 0.0)
TOOL_QUAT = (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)


def wrist_cell(*, source: str = "willy", planner: str = "curobo", backend: str = "fcl") -> RobotConfig:
    """A UR cell that enables the live planner world, names the 2F-85 and declares its tool frame."""
    return RobotConfig.model_validate({
        "vendor": "ur", "ur": {"motion_planner": planner},
        "safety": {
            "payload": {"enforce": False},
            "self_collision": {"backend": backend},
            "planning_world": {
                "enabled": True,
                "support_plane": {"height_mm": 0.0, "extent_mm": [1600.0, 1600.0], "thickness_mm": 50.0},
                "perceived": {"enabled": True, "voxel_field_mm": 0.0, "max_age_ms": 5000.0},
            },
        },
        "gripper": {"model": "robotiq_2f85", "tool_frame": {
            "source": source, "offset_mm": list(TOOL_OFFSET_MM), "rotation_quat_xyzw": list(TOOL_QUAT)}},
    })


def declared_frame() -> np.ndarray:
    return tool_frame_matrix(TOOL_OFFSET_MM, TOOL_QUAT)


def camera_to_tool() -> Any:
    from tests.test_camera_world_wiring import _CAMERA_TO_TOOL

    return _CAMERA_TO_TOOL


def wrist_rig(rig_id: str = "wrist", *, body: bool = True, enabled: bool = True, calibrated: bool = True,
              model: str = "realsense_d435", margin_mm: float = 5.0, bracket: Any = None,
              record_tolerance: "tuple[float, float] | None" = (1.0, 0.5)) -> SimpleNamespace:
    """A wrist rig as the schema loads it, cut to what the resolution and the world plan read."""
    extrinsics = None
    if calibrated:
        extrinsics = SimpleNamespace(
            mounting_mode="eye_in_hand", artifact_path=f"eih_{rig_id}.json",
            shutter_motion_tolerance_mm=1.0, shutter_motion_tolerance_deg=0.5,
            record_tolerance_mm=None if record_tolerance is None else record_tolerance[0],
            record_tolerance_deg=None if record_tolerance is None else record_tolerance[1])
    declared = SimpleNamespace(model=model, margin_mm=margin_mm, bracket=bracket) if body else None
    return SimpleNamespace(rig_id=rig_id, enabled=enabled, source="rgbd", extrinsics=extrinsics, body=declared)


class WristOwner:
    """An open wrist camera: the rig, a handle the world reads, and a calibration with or without its record."""

    def __init__(self, rig_id: str = "wrist", *, record: Any = "declared", source: str = "willy", **rig: Any) -> None:
        from tests.test_camera_world_wiring import _Owner

        self.rig_id = rig_id
        self.rig = wrist_rig(rig_id, **rig)
        self._owner = _Owner(rig_id, wrist=True)
        if isinstance(record, str) and record == "declared":
            self.record: Any = FlangeToTcp.from_matrix(source, declared_frame())
        elif record is None:
            self.record = UNSET
        else:
            self.record = FlangeToTcp.from_matrix(source, record)

    def handle(self) -> Any:
        return self._owner.handle()

    def calibration(self) -> SimpleNamespace:
        return SimpleNamespace(
            rig_id=self.rig_id, mounting_mode="eye_in_hand", camera_to_tool=camera_to_tool,
            shutter_motion_tolerance_mm=1.0, shutter_motion_tolerance_deg=0.5, flange_to_tcp=self.record,
            artifact_path=f"eih_{self.rig_id}.json")


def camera_section(*rigs: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(cameras=SimpleNamespace(rigs=list(rigs), primary_rig_id=rigs[0].rig_id))


def wrist_body(**owner: Any) -> Any:
    """The one body a default wrist owner resolves to on a willy cell."""
    from src.robot.execution.wrist_bodies import WristBodies

    return WristBodies.from_owners(wrist_cell(), [WristOwner(**owner)]).bodies[0]
