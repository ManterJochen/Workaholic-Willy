"""Calibration APIs for stereo vision, eye-hand solving and persistence.

Domain logic only: camera device I/O, robot vendor drivers, FastAPI routes and
application pipelines stay outside this boundary. OpenCV stays inside the
stereo subpackage and the sweep's target estimators (``targets.py``, and
``rgbd_marker_source.py`` through them), none of which this package imports;
the hand-eye solvers and the typed persistence use NumPy and
:mod:`src.geometry` alone.
"""

from __future__ import annotations

from .exceptions import (
    CalibrationDataError,
    CalibrationError,
    CalibrationSolveError,
    ExtrinsicsError,
    StereoCalibrationError,
)
from .extrinsics import EXTRINSICS_SCHEMA, Extrinsics
from .eye_hand import (
    EYE_HAND_DATASET_SCHEMA,
    EyeHandCalibrationResult,
    EyeHandCalibrationSettings,
    EyeHandDataset,
    EyeHandSample,
    EyeInHandCalibrator,
    EyeToHandCalibrator,
    MountingMode,
)
from .helpers import proj_to_K, unit_scaling, validate_image_pair_shapes
from .quality import (
    DEFAULT_BANDS_MM,
    DEFAULT_BANDS_PX,
    QUALITY_LABELS,
    QualityBandsMm,
    QualityBandsPx,
    QualityLabel,
    classify_rmse,
)
from .serialization import (
    CAM_TO_TOOL_SCHEMA,
    extrinsics_from_dict,
    extrinsics_to_dict,
    load_cam_to_tool,
    load_extrinsics,
    save_cam_to_tool,
    save_extrinsics,
)
from .solver import HandEyeAXXB, UmeyamaRigid

__all__ = [
    "CAM_TO_TOOL_SCHEMA",
    "DEFAULT_BANDS_MM",
    "DEFAULT_BANDS_PX",
    "EXTRINSICS_SCHEMA",
    "EYE_HAND_DATASET_SCHEMA",
    "QUALITY_LABELS",
    # Exceptions
    "CalibrationDataError",
    "CalibrationError",
    "CalibrationSolveError",
    # Core type
    "Extrinsics",
    "ExtrinsicsError",
    "EyeHandCalibrationResult",
    "EyeHandCalibrationSettings",
    "EyeHandDataset",
    "EyeHandSample",
    "EyeInHandCalibrator",
    "EyeToHandCalibrator",
    "MountingMode",
    "proj_to_K",
    "QualityBandsMm",
    "QualityBandsPx",
    "StereoCalibrationError",
    # Quality bands
    "QualityLabel",
    "classify_rmse",
    "extrinsics_from_dict",
    # Serialisation
    "extrinsics_to_dict",
    "load_cam_to_tool",
    "load_extrinsics",
    "save_cam_to_tool",
    "save_extrinsics",
    "unit_scaling",
    "validate_image_pair_shapes",
    # Solvers
    "HandEyeAXXB",
    "UmeyamaRigid",
]
