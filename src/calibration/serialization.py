"""
JSON serialisation for :class:`Extrinsics` and for the eye-in-hand
``CAMERA -> TOOL`` transform.

Wire format
-----------
::

    {
        "schema": "willy.calibration.extrinsics/1",
        "transform": { ... willy.geometry.transform/1 ... },
        "rmse_mm": 1.234,
        "max_error_mm": 2.5,
        "num_samples": 42,
        "captured_at": "2026-05-08T13:44:18+00:00",
        "rig_id": "rig-0",
        "quality": "excellent",
    }

Both writers render the payload into a ``.tmp`` sibling and move it over the
target, so a reader never sees half a file. The schema version is checked on
load; a mismatch raises :class:`ExtrinsicsError` rather than coercing.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.contracts import UNSET, Maybe, chosen
from src.geometry import Frame, GeometryError, Transform, transform_from_dict, transform_to_dict
from src.utility.log_cfg import create_logger

from .constants import CALIBRATION_LOG_DIR, SERIALIZATION_LOG_FILE
from .exceptions import ExtrinsicsError
from .extrinsics import EXTRINSICS_SCHEMA, Extrinsics

logger = create_logger(
    "CalibrationSerialization", SERIALIZATION_LOG_FILE, log_dir=CALIBRATION_LOG_DIR
)

#: Wire schema for a persisted eye-in-hand CAMERA->TOOL calibration transform. :data:`EXTRINSICS_SCHEMA`
#: is locked to CAMERA->BASE and cannot carry one, so a wrist camera gets its own typed artifact, which
#: the multi-camera registry loads into an EyeInHandFrameResolver.
CAM_TO_TOOL_SCHEMA: str = "willy.calibration.cam_to_tool/1"

#: The same artifact with the flange to TCP the solve was made against recorded beside it. The tool
#: pose the routine read was the flange times that frame, so CAMERA to TOOL holds only while the cell
#: holds that frame. Written when the caller hands the record in; a caller without one writes ``/1``,
#: byte for byte.
CAM_TO_TOOL_SCHEMA_V2: str = "willy.calibration.cam_to_tool/2"

__all__ = [
    "CAM_TO_TOOL_SCHEMA",
    "CAM_TO_TOOL_SCHEMA_V2",
    "EXTRINSICS_SCHEMA",
    "FlangeToTcp",
    "extrinsics_from_dict",
    "extrinsics_to_dict",
    "load_cam_to_tool",
    "load_cam_to_tool_artifact",
    "load_extrinsics",
    "save_cam_to_tool",
    "save_extrinsics",
]


@dataclass(frozen=True)
class FlangeToTcp:
    """The flange to TCP an eye-in-hand solve was made against, and where the cell took it from.

    ``source`` is the tool frame mode: ``willy`` when the driver composes the declared frame itself,
    so the record is the declared numbers; ``polyscope`` when the controller applies its own tool
    setting, so the record is the frame the driver derived from the controller at connect.
    ``matrix_mm`` is the 4x4 homogeneous transform as rows, translation in millimetres. A record
    that is not a finite rigid transform raises :class:`ExtrinsicsError`.
    """

    source: Literal["willy", "polyscope"]
    matrix_mm: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if self.source not in ("willy", "polyscope"):
            raise ExtrinsicsError(f"FlangeToTcp: source must be 'willy' or 'polyscope', got {self.source!r}")
        rows = self.matrix_mm
        if len(rows) != 4 or any(len(row) != 4 for row in rows):
            raise ExtrinsicsError("FlangeToTcp: matrix_mm must be four rows of four numbers")
        if not all(math.isfinite(value) for row in rows for value in row):
            raise ExtrinsicsError("FlangeToTcp: matrix_mm holds a number that is not finite")
        matrix = np.asarray(rows, dtype=np.float64)
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12):
            raise ExtrinsicsError(f"FlangeToTcp: the last row must be 0 0 0 1, got {list(matrix[3])}")
        rotation = matrix[:3, :3]
        if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9)):
            raise ExtrinsicsError("FlangeToTcp: matrix_mm is not a rigid transform (its rotation is not orthonormal)")

    @classmethod
    def from_matrix(cls, source: Any, matrix: Any) -> "FlangeToTcp":
        """The record of a 4x4 array-like."""
        array = np.asarray(matrix, dtype=np.float64)
        if array.shape != (4, 4):
            raise ExtrinsicsError(f"FlangeToTcp: expected a 4x4 matrix, got shape {array.shape}")
        return cls(source=source, matrix_mm=tuple(tuple(float(v) for v in row) for row in array))

    def matrix(self) -> np.ndarray:
        """The record as a 4x4 float64 array."""
        return np.asarray(self.matrix_mm, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "matrix_mm": [list(row) for row in self.matrix_mm]}


def extrinsics_to_dict(ext: Extrinsics) -> dict[str, Any]:
    """Serialise :class:`Extrinsics` to a JSON-friendly dictionary."""
    return {
        "schema": EXTRINSICS_SCHEMA,
        "transform": transform_to_dict(ext.transform),
        "rmse_mm": float(ext.rmse_mm),
        "max_error_mm": float(ext.max_error_mm),
        "num_samples": int(ext.num_samples),
        "captured_at": ext.captured_at.isoformat(),
        "rig_id": ext.rig_id,
        "quality": ext.quality,
    }


def extrinsics_from_dict(data: Mapping[str, Any]) -> Extrinsics:
    """Deserialise :class:`Extrinsics` from :func:`extrinsics_to_dict` output.

    A foreign schema, a missing key, a non-string ``captured_at`` and a payload the
    dataclass itself rejects all raise :class:`ExtrinsicsError`.
    """
    schema = data.get("schema")
    if schema != EXTRINSICS_SCHEMA:
        raise ExtrinsicsError(
            f"extrinsics_from_dict: expected schema {EXTRINSICS_SCHEMA!r}, got {schema!r}"
        )

    required = ("transform", "rmse_mm", "max_error_mm", "num_samples", "captured_at", "rig_id")
    missing = [k for k in required if k not in data]
    if missing:
        raise ExtrinsicsError(f"extrinsics_from_dict: missing required keys {missing!r}")

    try:
        transform = transform_from_dict(data["transform"])
    except (GeometryError, TypeError, ValueError, KeyError) as exc:
        raise ExtrinsicsError("extrinsics_from_dict: invalid transform payload") from exc

    captured_raw = data["captured_at"]
    if not isinstance(captured_raw, str):
        raise ExtrinsicsError(
            f"extrinsics_from_dict: captured_at must be ISO-8601 string, got {type(captured_raw).__name__}"
        )
    try:
        captured_at = datetime.fromisoformat(captured_raw)
    except ValueError as exc:
        raise ExtrinsicsError(f"extrinsics_from_dict: invalid captured_at {captured_raw!r}") from exc

    try:
        return Extrinsics(
            transform=transform,
            rmse_mm=float(data["rmse_mm"]),
            max_error_mm=float(data["max_error_mm"]),
            num_samples=int(data["num_samples"]),
            captured_at=captured_at,
            rig_id=str(data["rig_id"]),
            quality=data.get("quality", "unknown"),
        )
    except (TypeError, ValueError, GeometryError) as exc:
        raise ExtrinsicsError("extrinsics_from_dict: invalid extrinsics payload") from exc


def save_extrinsics(path: str | Path, ext: Extrinsics) -> Path:
    """Atomically write :class:`Extrinsics` as schema-versioned JSON, returning the target path.

    Missing parent directories are created. Keys are sorted, so two identical solves
    produce identical bytes.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(extrinsics_to_dict(ext), indent=2, sort_keys=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(target)
    # Ties this file to the solve that produced it, so a bad pick traces back to a calibration run.
    logger.info(
        "Extrinsics written: %s (rig=%s, samples=%d, rmse=%.3f mm, quality=%s, %d bytes).",
        target,
        ext.rig_id,
        int(ext.num_samples),
        float(ext.rmse_mm),
        ext.quality,
        len(payload.encode("utf-8")),
    )
    return target


def load_extrinsics(path: str | Path) -> Extrinsics:
    """Load :class:`Extrinsics` from a JSON file written by :func:`save_extrinsics`.

    Invalid JSON and a non-object top level raise :class:`ExtrinsicsError`; the payload
    itself is checked by :func:`extrinsics_from_dict`.
    """
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExtrinsicsError(f"load_extrinsics: {path!s} is not valid JSON") from exc
    if not isinstance(data, Mapping):
        raise ExtrinsicsError(f"load_extrinsics: top-level JSON must be an object, got {type(data).__name__}")
    ext = extrinsics_from_dict(data)
    logger.info(
        "Extrinsics loaded: %s (rig=%s, rmse=%.3f mm, quality=%s, captured %s).",
        path,
        ext.rig_id,
        float(ext.rmse_mm),
        ext.quality,
        ext.captured_at.isoformat(),
    )
    return ext


# ----------------------------------------------------------------------
# Eye-in-hand CAMERA->TOOL calibration transform for a wrist camera
# ----------------------------------------------------------------------
def save_cam_to_tool(
    path: str | Path, transform: Transform, *, rig_id: str, flange_to_tcp: Maybe[FlangeToTcp] = UNSET,
) -> Path:
    """Atomically persist an eye-in-hand ``CAMERA -> TOOL`` calibration transform as versioned JSON.

    ``transform`` must be ``Transform(from_frame=CAMERA, to_frame=TOOL)`` with translation in mm:
    the static half an :class:`EyeInHandFrameResolver` composes with the live TCP. Wrong frames
    or a blank ``rig_id`` raise :class:`ExtrinsicsError`.

    A ``flange_to_tcp`` handed in writes ``/2`` with the record beside the transform. Left unset,
    the function writes ``/1``, byte for byte the artifact without a record.
    """
    if transform.from_frame is not Frame.CAMERA or transform.to_frame is not Frame.TOOL:
        raise ExtrinsicsError(
            "save_cam_to_tool requires Transform(CAMERA -> TOOL); got "
            f"{transform.from_frame.value} -> {transform.to_frame.value}"
        )
    if not str(rig_id).strip():
        raise ExtrinsicsError("save_cam_to_tool: rig_id must be a non-empty string")
    body: dict[str, Any] = {"schema": CAM_TO_TOOL_SCHEMA, "transform": transform_to_dict(transform),
                            "rig_id": str(rig_id)}
    if chosen(flange_to_tcp):
        body = {**body, "schema": CAM_TO_TOOL_SCHEMA_V2, "flange_to_tcp": flange_to_tcp.to_dict()}
    payload = json.dumps(body, indent=2, sort_keys=True)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(target)
    logger.info(
        "CAMERA->TOOL transform written: %s (rig=%s, %d bytes).",
        target,
        str(rig_id),
        len(payload.encode("utf-8")),
    )
    return target


def load_cam_to_tool(path: str | Path) -> Transform:
    """Load an eye-in-hand ``CAMERA -> TOOL`` transform written by :func:`save_cam_to_tool`, ``/1`` or ``/2``.

    The schema and the stored frame pair are both checked, so a CAMERA->BASE artifact written by
    :func:`save_extrinsics` cannot enter as a wrist calibration.
    """
    return _read_cam_to_tool(path)[0]


def load_cam_to_tool_artifact(path: str | Path) -> tuple[Transform, Maybe[FlangeToTcp]]:
    """The ``CAMERA -> TOOL`` transform and the flange to TCP it was solved against, ``UNSET`` for a ``/1`` artifact."""
    return _read_cam_to_tool(path)


def _read_cam_to_tool(path: str | Path) -> tuple[Transform, Maybe[FlangeToTcp]]:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExtrinsicsError(f"load_cam_to_tool: {path!s} is not valid JSON") from exc
    schema = data.get("schema") if isinstance(data, Mapping) else None
    if schema not in (CAM_TO_TOOL_SCHEMA, CAM_TO_TOOL_SCHEMA_V2):
        raise ExtrinsicsError(
            f"load_cam_to_tool: expected schema {CAM_TO_TOOL_SCHEMA!r} or {CAM_TO_TOOL_SCHEMA_V2!r}, got {schema!r}"
        )
    record: Maybe[FlangeToTcp] = UNSET
    if schema == CAM_TO_TOOL_SCHEMA_V2:
        held = data.get("flange_to_tcp")
        if not isinstance(held, Mapping):
            raise ExtrinsicsError(f"load_cam_to_tool: {path!s} is {CAM_TO_TOOL_SCHEMA_V2!r} and holds no flange_to_tcp")
        try:
            record = FlangeToTcp.from_matrix(held.get("source"), held.get("matrix_mm"))
        except (TypeError, ValueError) as exc:
            raise ExtrinsicsError(f"load_cam_to_tool: {path!s} holds an invalid flange_to_tcp") from exc
    try:
        transform = transform_from_dict(data["transform"])
    except (GeometryError, TypeError, ValueError, KeyError) as exc:
        raise ExtrinsicsError("load_cam_to_tool: invalid transform payload") from exc
    if transform.from_frame is not Frame.CAMERA or transform.to_frame is not Frame.TOOL:
        raise ExtrinsicsError(
            "load_cam_to_tool: stored transform must be CAMERA -> TOOL; got "
            f"{transform.from_frame.value} -> {transform.to_frame.value}"
        )
    logger.info("CAMERA->TOOL transform loaded: %s (rig=%s, flange to TCP %s).", path, data.get("rig_id"),
                record.source if chosen(record) else "not recorded")
    return transform, record
