"""The wrist cameras a cell's arm carries, resolved once from the camera section, for every door that builds an arm.

A camera the arm carries can hit the cell unnoticed unless every collision model carries its housing. A rig
declares that body (``camera.cameras.rigs[<id>].body``); this module turns the declaration into
:class:`~src.robot.safety.planning.body_link.WristBody` values, or refuses the cell saying why, and the real cell
build, ``Robot``, ``PlannerStart``, the calibration CLI and the desk checklist all ask it the same question.

Only a cell that reads geometry resolves bodies: one whose planner is cuRobo, or whose self collision guard reads
hand geometry (:func:`~src.robot.safety.planning.hand.wrist_body_reader`). On such a cell:

* an enabled eye_in_hand rig without a body is refused, naming the key;
* a body whose rig declares no calibration is refused, because nothing places it, and the calibration CLI
  sweeps such a rig only with a stated reason;
* a calibration without the flange to TCP it was solved against is refused, and so is one whose record is not
  the cell's tool frame: exactly the declared frame on a ``willy`` cell, within the connect tolerance on a
  ``polyscope`` cell at a desk, where the driver compares it again with the rig's record tolerances when it
  connects;
* a camera the repository's registry does not hold, or that the tree describes differently, is refused.

A body is resolved whether or not its rig is enabled: a switched off camera still hangs on the arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.contracts import chosen

if TYPE_CHECKING:
    from src.robot.safety.planning.body_link import WristBody

__all__ = ["WristBodies", "WristBodyRequired"]


class WristBodyRequired(ValueError):
    """A cell that reads geometry cannot place, or was not told, the body of a camera its arm carries."""


def _key(rig_id: str) -> str:
    return f"camera.cameras.rigs[{rig_id!r}]"


@dataclass(frozen=True)
class WristBodies:
    """The wrist camera bodies one cell's arm carries, and why they were or were not read."""

    bodies: tuple["WristBody", ...]
    #: Why this cell reads geometry, as :func:`wrist_body_reader` says it, or ``None`` for a cell that reads none
    #: and so resolved no body.
    reader: "str | None"

    @classmethod
    def from_config(cls, robot_cfg: Any, camera_cfg: Any, *, data_dir: "str | Path | None" = None) -> "WristBodies":
        """The bodies ``camera_cfg`` declares for the arm ``robot_cfg`` describes, or :class:`WristBodyRequired`.

        ``camera_cfg`` is the camera section of the same tree, ``data_dir`` that tree's root, whose camera
        registry is compared with the repository's; ``None`` is the repository's tree. Every calibration
        artifact is loaded now, through ``RigCalibration``.
        """
        from src.calibration.rig_calibration import RigCalibration, RigCalibrationError

        def calibration_of(rig: Any) -> Any:
            try:
                return RigCalibration.from_config(rig.rig_id, rig.extrinsics)
            except RigCalibrationError as exc:
                raise WristBodyRequired(f"{_key(rig.rig_id)}.body cannot be placed: {exc}") from exc

        rigs = tuple(getattr(getattr(camera_cfg, "cameras", None), "rigs", None) or ())
        return cls._resolve(robot_cfg, rigs, calibration_of, data_dir=data_dir)

    @classmethod
    def from_owners(cls, robot_cfg: Any, owners: "Any") -> "WristBodies":
        """The bodies the open camera owners' rigs declare, placed from each owner's calibration.

        An owner answers ``rig`` and ``calibration()`` as ``Camera`` does. Only the rigs handed in are read, so
        an eye_in_hand rig the caller did not hand in is not refused here; the build that opens every camera is.
        """
        by_rig = {str(owner.rig_id): owner for owner in owners}

        def calibration_of(rig: Any) -> Any:
            try:
                return by_rig[str(rig.rig_id)].calibration()
            except Exception as exc:  # noqa: BLE001 (a calibration that does not load places nothing)
                raise WristBodyRequired(f"{_key(rig.rig_id)}.body cannot be placed: {exc}") from exc

        return cls._resolve(robot_cfg, tuple(owner.rig for owner in owners), calibration_of, data_dir=None)

    @classmethod
    def _resolve(cls, robot_cfg: Any, rigs: "tuple[Any, ...]", calibration_of: Any, *,
                 data_dir: "str | Path | None") -> "WristBodies":
        from src.config.cameras import load_camera, tree_camera_refusal
        from src.calibration.rig_calibration import flange_to_tcp_refusal
        from src.config.loader import ConfigError
        from src.robot.safety.planning.body_link import WristBody
        from src.robot.safety.planning.hand import wrist_body_reader

        reader = wrist_body_reader(robot_cfg)
        if reader is None:
            return cls(bodies=(), reader=None)
        bodies: list[WristBody] = []
        for rig in rigs:
            key = _key(rig.rig_id)
            body = getattr(rig, "body", None)
            extrinsics = getattr(rig, "extrinsics", None)
            if body is None:
                if rig.enabled and extrinsics is not None and extrinsics.mounting_mode == "eye_in_hand":
                    raise WristBodyRequired(
                        f"{key} is an enabled eye_in_hand camera the arm carries and declares no body, and this cell "
                        f"reads geometry ({reader}): its housing would be invisible to the planner, the exact guard "
                        f"and the self filter. Declare {key}.body with the camera model from config/cameras/, "
                        "margin_mm and bracket")
                continue
            refusal = tree_camera_refusal(body.model, data_dir=data_dir)
            if refusal is not None:
                raise WristBodyRequired(f"{key}.body: {refusal}")
            try:
                spec = load_camera(body.model)
            except ConfigError as exc:
                raise WristBodyRequired(f"{key}.body: {exc}") from exc
            if extrinsics is None:
                raise WristBodyRequired(
                    f"{key}.body declares a camera the arm carries and the rig declares no calibration, so nothing "
                    "places the body on the flange. Calibrate it first: python -m "
                    f"src.robot.execution.real_cell.calibrate --rig {rig.rig_id} --mode eye_in_hand "
                    '--unmodelled-wrist-body "<why the sweep may run without it>"')
            calibration = calibration_of(rig)
            record = calibration.flange_to_tcp
            if not chosen(record):
                raise WristBodyRequired(
                    f"{key}.body is placed from the flange to TCP its calibration was solved against, and "
                    f"{calibration.artifact_path} records none: it was written before the record existed. Calibrate "
                    "the rig again")
            stale = flange_to_tcp_refusal(calibration, robot_cfg.gripper.tool_frame)
            if stale is not None:
                raise WristBodyRequired(stale)
            placed = WristBody.from_parts(
                rig_id=rig.rig_id, spec=spec, bracket=body.bracket, margin_mm=body.margin_mm,
                camera_to_tool=calibration.camera_to_tool(), flange_to_tcp=record,
                artifact_path=calibration.artifact_path,
                record_tolerance_mm=getattr(extrinsics, "record_tolerance_mm", None),
                record_tolerance_deg=getattr(extrinsics, "record_tolerance_deg", None),
            )
            cover = placed.cover_refusal()
            if cover is not None:
                raise WristBodyRequired(f"{key}.body: {cover}")
            bodies.append(placed)
        return cls(bodies=tuple(bodies), reader=reader)

    def hand_to(self, arm: Any) -> None:
        """Hand the bodies to ``arm`` through its ``set_wrist_bodies``, or refuse an arm that cannot take them."""
        setter = getattr(arm, "set_wrist_bodies", None)
        if callable(setter):
            setter(self.bodies)
            return
        if self.bodies:
            raise WristBodyRequired(
                f"this cell carries the wrist camera(s) {[body.link_name for body in self.bodies]} and reads geometry "
                f"({self.reader}), and {type(arm).__name__} takes no wrist bodies, so its planner and guard would not "
                "hold them")

    def line(self) -> str:
        """One ASCII line saying which wrist cameras the arm carries."""
        if self.reader is None:
            return "wrist cameras  not read: this cell's planner and guard read no geometry"
        if not self.bodies:
            return "wrist cameras  none declared"
        return "wrist cameras  " + "; ".join(body.render() for body in self.bodies)

    def render(self) -> str:
        return self.line().encode("ascii", "backslashreplace").decode("ascii")

    def to_dict(self) -> "dict[str, Any]":
        return {"reader": self.reader, "bodies": [body.to_dict() for body in self.bodies]}
