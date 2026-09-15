"""A rig's calibration, loaded from the one key that declares it: ``camera.cameras.rigs[<id>].extrinsics``.

Every reader of a fixed camera's CAMERA to BASE (fusion, the planner world, the locator, the
preflight) and of a wrist camera's CAMERA to TOOL loads it through `RigCalibration.from_config`, so
there is one loader, and `load_extrinsics` and `load_cam_to_tool` have one production caller. A file
that does not load is refused here, when the cell is built, naming the key and the path.

It imports nothing from ``src.camera`` or ``src.robot``, so the robot may load it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.calibration.serialization import load_cam_to_tool, load_extrinsics
from src.geometry import Transform

__all__ = ["RigCalibration", "RigCalibrationError", "RigNotCalibrated"]

MountingMode = Literal["eye_to_hand", "eye_in_hand"]


class RigCalibrationError(ValueError):
    """A declared rig calibration that cannot be used: its artifact does not load, or it was asked
    for a transform its mounting does not have."""


class RigNotCalibrated(RigCalibrationError):
    """A rig was asked for its calibration and declares none."""


def _key(rig_id: str) -> str:
    return f"camera.cameras.rigs[{rig_id!r}].extrinsics"


@dataclass(frozen=True)
class RigCalibration:
    """One rig's mounting, its artifact and the transform the artifact holds, and for a wrist rig the
    arm motion it tolerates at the shutter."""

    rig_id: str
    mounting_mode: MountingMode
    artifact_path: str
    #: CAMERA to BASE for an eye_to_hand rig, CAMERA to TOOL for an eye_in_hand rig.
    transform: Transform
    shutter_motion_tolerance_mm: float | None = None
    shutter_motion_tolerance_deg: float | None = None

    @classmethod
    def from_config(cls, rig_id: str, extrinsics_cfg: Any) -> RigCalibration:
        """The calibration ``extrinsics_cfg`` declares for ``rig_id``, with its artifact loaded now."""
        if extrinsics_cfg is None:
            raise RigNotCalibrated(
                f"rig {rig_id!r} declares no calibration. Calibrate it (python -m "
                f"src.robot.execution.real_cell.calibrate --rig {rig_id}) and set {_key(rig_id)} to "
                "the artifact that writes.")
        mode = extrinsics_cfg.mounting_mode
        path = str(extrinsics_cfg.artifact_path)
        try:
            transform = load_extrinsics(path).transform if mode == "eye_to_hand" else load_cam_to_tool(path)
        except Exception as exc:  # noqa: BLE001 (refused where the fix goes, whatever the file did)
            raise RigCalibrationError(
                # The path as written, not its repr: on Windows a repr doubles every backslash, and
                # a path copied out of this sentence would name a file that does not exist.
                f"{_key(rig_id)} names {path} ({mode}), which does not load: {type(exc).__name__}: {exc}"
            ) from exc
        return cls(
            rig_id=rig_id,
            mounting_mode=mode,
            artifact_path=path,
            transform=transform,
            shutter_motion_tolerance_mm=extrinsics_cfg.shutter_motion_tolerance_mm,
            shutter_motion_tolerance_deg=extrinsics_cfg.shutter_motion_tolerance_deg,
        )

    def camera_to_base(self) -> Transform:
        """A fixed camera's CAMERA to BASE. A wrist camera has none, because the arm carries it."""
        if self.mounting_mode != "eye_to_hand":
            raise RigCalibrationError(
                f"rig {self.rig_id!r} is eye_in_hand: a wrist camera has no fixed CAMERA to BASE; ask its "
                "frames for the tool pose at the shutter.")
        return self.transform

    def camera_to_tool(self) -> Transform:
        """A wrist camera's CAMERA to TOOL. A fixed camera has none."""
        if self.mounting_mode != "eye_in_hand":
            raise RigCalibrationError(
                f"rig {self.rig_id!r} is eye_to_hand: a fixed camera has no CAMERA to TOOL; its artifact is "
                "CAMERA to BASE.")
        return self.transform

    def render(self) -> str:
        tolerance = ("" if self.mounting_mode == "eye_to_hand" else
                     f", shutter motion within {self.shutter_motion_tolerance_mm} mm and "
                     f"{self.shutter_motion_tolerance_deg} deg")
        return f"rig {self.rig_id!r} calibrated {self.mounting_mode} from {self.artifact_path}{tolerance}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rig_id": self.rig_id,
            "mounting_mode": self.mounting_mode,
            "artifact_path": self.artifact_path,
            "shutter_motion_tolerance_mm": self.shutter_motion_tolerance_mm,
            "shutter_motion_tolerance_deg": self.shutter_motion_tolerance_deg,
        }
