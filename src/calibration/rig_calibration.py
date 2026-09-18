"""A rig's calibration, loaded from the one key that declares it: ``camera.cameras.rigs[<id>].extrinsics``.

Every reader of a fixed camera's CAMERA to BASE (fusion, the planner world, the locator, the
preflight) and of a wrist camera's CAMERA to TOOL loads it through `RigCalibration.from_config`, so
there is one loader, and `load_extrinsics` and `load_cam_to_tool_artifact` have one production
caller. A wrist camera's artifact may carry the flange to TCP its solve was made against, and the
calibration keeps it beside the transform. A file that does not load is refused here, when the cell
is built, naming the key and the path.

At import it loads nothing from ``src.camera`` or ``src.robot``, so the robot may load it. The
flange to TCP check (:func:`flange_to_tcp_refusal`) reads the UR tool frame helpers when it runs,
because a wrist camera's pick frame and its body are both placed against the frame it compares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.calibration.serialization import FlangeToTcp, load_cam_to_tool_artifact, load_extrinsics
from src.contracts import UNSET, Maybe, chosen
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
    #: The flange to TCP a wrist camera's solve was made against. ``UNSET`` for a fixed camera and for
    #: an artifact that carries no record.
    flange_to_tcp: Maybe[FlangeToTcp] = UNSET
    #: How far the cell's flange to TCP may lie from the record before a body placed from it is refused.
    record_tolerance_mm: float | None = None
    record_tolerance_deg: float | None = None

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
        record: Maybe[FlangeToTcp] = UNSET
        try:
            if mode == "eye_to_hand":
                transform = load_extrinsics(path).transform
            else:
                transform, record = load_cam_to_tool_artifact(path)
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
            flange_to_tcp=record,
            record_tolerance_mm=getattr(extrinsics_cfg, "record_tolerance_mm", None),
            record_tolerance_deg=getattr(extrinsics_cfg, "record_tolerance_deg", None),
        )

    def flange_to_tcp_refusal(self, tool_frame: Any) -> "str | None":
        """Why this wrist rig was not calibrated against the flange to TCP this cell declares, or None.

        The answer of the module function :func:`flange_to_tcp_refusal` for this calibration.
        """
        return flange_to_tcp_refusal(self, tool_frame)

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
        if self.mounting_mode == "eye_in_hand":
            tolerance += (f", flange to TCP recorded on {self.flange_to_tcp.source}" if chosen(self.flange_to_tcp)
                          else ", flange to TCP not recorded")
        return f"rig {self.rig_id!r} calibrated {self.mounting_mode} from {self.artifact_path}{tolerance}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rig_id": self.rig_id,
            "mounting_mode": self.mounting_mode,
            "artifact_path": self.artifact_path,
            "shutter_motion_tolerance_mm": self.shutter_motion_tolerance_mm,
            "shutter_motion_tolerance_deg": self.shutter_motion_tolerance_deg,
            "flange_to_tcp": self.flange_to_tcp.to_dict() if chosen(self.flange_to_tcp) else None,
            "record_tolerance_mm": self.record_tolerance_mm,
            "record_tolerance_deg": self.record_tolerance_deg,
        }


def flange_to_tcp_record_refusal(
    rig_id: str, *, record_source: str, record_mm: Any, source: str, frame: Any,
    record_tolerance_mm: "float | None", record_tolerance_deg: "float | None",
) -> "str | None":
    """Why ``frame``, the flange to TCP a ``source`` cell applies, is not the rig's record, or None.

    A wrist camera's body and its pick frame are both placed from the record, so a frame that moved
    since makes both stale. On a ``willy`` cell the record was written from the declared numbers and
    has to be them; on a ``polyscope`` cell the frame is derived at connect and may lie within the
    rig's record tolerances.
    """
    import numpy as np

    from src.robot.drivers.ur.tool_frame import compare_tool_frames

    key = f"camera.cameras.rigs[{rig_id!r}]"
    if source != record_source:
        return (f"{key} was calibrated on a {record_source} cell and this cell's tool frame source is "
                f"{source}, so its body and its pick frame were placed against another tool frame: calibrate the "
                "rig again")
    if frame is None:
        return f"{key} is placed from a recorded flange to TCP, and this cell cannot say which frame it applies"
    d_t, d_r = compare_tool_frames(np.asarray(frame, dtype=np.float64), np.asarray(record_mm, dtype=np.float64))
    if source == "willy":
        limit_mm, limit_deg = 1e-6, 1e-6
    else:
        if record_tolerance_mm is None or record_tolerance_deg is None:
            return (f"{key}.extrinsics names no record_tolerance_mm and record_tolerance_deg, so a polyscope cell "
                    "cannot say how far its derived tool frame may lie from the one this rig was calibrated "
                    "against")
        limit_mm, limit_deg = record_tolerance_mm, record_tolerance_deg
    if d_t > limit_mm or d_r > limit_deg:
        return (f"{key} was calibrated against a flange to TCP {d_t:.3f} mm and {d_r:.3f} deg away from the one "
                f"this cell applies (allowed {limit_mm:g} mm and {limit_deg:g} deg), so its body and its pick "
                f"frame are both stale: calibrate the rig again (python -m "
                f"src.robot.execution.real_cell.calibrate --rig {rig_id} --mode eye_in_hand)")
    return None


def flange_to_tcp_refusal(calibration: Any, tool_frame: Any) -> "str | None":
    """Why a wrist rig's calibration was not solved against the flange to TCP this cell declares, or ``None``.

    ``calibration`` answers as :class:`RigCalibration` does (``rig_id``, ``artifact_path``,
    ``flange_to_tcp``, the record tolerances), and ``tool_frame`` is ``robot.gripper.tool_frame``. A
    camera on the wrist is placed by its CAMERA to TOOL composed with the tool pose, so every frame
    of it, the pick frame and a body alike, is placed against the flange to TCP the solve was made
    against. A calibration that records none, one recorded on a cell of another tool frame source,
    and one whose record lies further from the declared frame than the cell accepts are refused,
    naming the command that calibrates the rig again. A fixed camera has no tool in its transform
    and is never refused.
    """
    if str(getattr(calibration, "mounting_mode", "eye_in_hand")) != "eye_in_hand":
        return None
    rig_id = str(calibration.rig_id)
    key = f"camera.cameras.rigs[{rig_id!r}]"
    again = f"python -m src.robot.execution.real_cell.calibrate --rig {rig_id} --mode eye_in_hand"
    record = getattr(calibration, "flange_to_tcp", UNSET)
    if not chosen(record) or record is None:
        return (f"{key} is placed from the flange to TCP its calibration was solved against, and "
                f"{calibration.artifact_path} records none: it was written before the record existed. Calibrate the "
                f"rig again ({again})")
    source = getattr(tool_frame, "source", None)
    if source not in ("willy", "polyscope"):
        return (f"{key} is placed from a recorded flange to TCP, and robot.gripper.tool_frame.source is "
                f"{source!r}, so this cell declares no tool frame to hold the record to")
    import numpy as np

    from src.robot.drivers.ur.tool_frame import compare_tool_frames, tool_frame_matrix

    declared = tool_frame_matrix(tool_frame.offset_mm, tool_frame.rotation_quat_xyzw)
    if source == "willy" or record.source != "polyscope":
        return flange_to_tcp_record_refusal(
            rig_id, record_source=str(record.source), record_mm=record.matrix(), source=str(source), frame=declared,
            record_tolerance_mm=getattr(calibration, "record_tolerance_mm", None),
            record_tolerance_deg=getattr(calibration, "record_tolerance_deg", None),
        )
    d_t, d_r = compare_tool_frames(np.asarray(record.matrix()), declared)
    limit = float(tool_frame.verify_tolerance_mm)
    if d_t <= limit and d_r <= 5.0:
        return None
    return (f"{key} was calibrated against a flange to TCP {d_t:.3f} mm and {d_r:.3f} deg away from the declared "
            f"tool frame, further than a connect accepts ({limit:g} mm and 5 deg), so its body and its pick frame "
            f"are stale: calibrate the rig again ({again})")
