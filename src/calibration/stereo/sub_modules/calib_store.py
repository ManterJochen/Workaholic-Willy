"""JSON persistence for stereo calibration parameters.

Writes the small per-camera intrinsics and rectification parameters. The
remap tables run to megabytes and are not stored; :class:`CalibrationResult`
recomputes them on load. Each file carries :data:`STEREO_CALIB_SCHEMA`, the
same versioning the extrinsics serializer applies, and a load refuses any
other value.

The file says nothing of where the pair stands in the cell. It once carried an
optional ``T_cam_to_base`` beside the parameters, a second source of the pose
that no reader in the pick path took: since 2026-10-17 a stereo rig is placed
by its own calibration, ``camera.cameras.rigs[<id>].extrinsics``, as an RGB-D
rig is. A file written before still loads, and that key in it is not read.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.calibration.exceptions import StereoCalibrationError
from src.calibration.stereo.config import CalibrationResult

logger = logging.getLogger(__name__)

STEREO_CALIB_SCHEMA = "willy.calibration.stereo/1"

_PARAM_KEYS = ("camL", "distL", "rectL", "projL", "camR", "distR", "rectR", "projR", "Q")

__all__ = ["STEREO_CALIB_SCHEMA", "StereoCalibrationStore"]


class StereoCalibrationStore:
    """Reads and writes a stereo pair's calibration parameters as JSON."""

    def save(self, filepath: str, result: CalibrationResult) -> None:
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "schema": STEREO_CALIB_SCHEMA,
            "frame_size": [int(result.frame_size[0]), int(result.frame_size[1])],
        }
        for key in _PARAM_KEYS:
            payload[key] = np.asarray(getattr(result, key), dtype=np.float64).tolist()
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, filepath: str) -> CalibrationResult:
        try:
            payload = json.loads(Path(filepath).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StereoCalibrationError(f"cannot read stereo calibration '{filepath}': {exc}") from exc
        if payload.get("schema") != STEREO_CALIB_SCHEMA:
            raise StereoCalibrationError(
                f"stereo calibration schema mismatch: {payload.get('schema')!r} != {STEREO_CALIB_SCHEMA!r}"
            )
        try:
            p = {key: np.asarray(payload[key], dtype=np.float64) for key in _PARAM_KEYS}
            width, height = (int(v) for v in payload["frame_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StereoCalibrationError(f"stereo calibration file has missing/invalid fields: {exc}") from exc
        # A file written before 2026-10-17 may carry `T_cam_to_base` too: not read, the rig's own extrinsics place it.
        return CalibrationResult(
            camL=p["camL"], distL=p["distL"], rectL=p["rectL"], projL=p["projL"],
            camR=p["camR"], distR=p["distR"], rectR=p["rectR"], projR=p["projR"],
            Q=p["Q"], frame_size=(width, height),
        )
