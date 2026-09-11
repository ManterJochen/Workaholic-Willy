"""Public schema surface for the configuration package.

Import from here rather than reaching into the ``camera``, ``models`` or
``robot`` submodules. Those are re-exported for convenience and their
layout is not part of the stable contract.
"""

from ._base import StrictModel, validate_aruco_dict_name
from .app import (
    AppConfig,
    CameraConfig,
    ModelsConfig,
    PerceptionModelsConfig,
    primary_camera_calibration_conflict,
)
from .camera import (
    ArucoDictName,
    BaseRigConfig,
    CalibrationConfig,
    CameraSystemConfig,
    EyeHandRoutineConfig,
    EyeInHandWorkflowConfig,
    EyeToHandWorkflowConfig,
    HandEyeConfig,
    QualityConfig,
    RGBDCalibPaths,
    RGBDDeviceRigConfig,
    SingleDeviceRigConfig,
    StereoCalibPaths,
    StereoMatcherConfig,
    WebcamPairRigConfig,
    WlsFilterConfig,
)
from .models import (
    InferenceOptimization,
    ObjectDetectorConfig,
    SegmenterConfig,
    SpeechToTextConfig,
)
from .robot import (
    DummyConfig,
    GripperConfig,
    MotionLimitsConfig,
    RobotCalibrationConfig,
    RobotCalibrationQualityBandsMm,
    RobotConfig,
    RobotSafetyConfig,
    SafePoseConfig,
    SimCameraSchema,
    SimConfig,
    URConfig,
    WorkspaceLimitsConfig,
)
from .runtime import (
    ImageEncodingConfig,
    RuntimeConfig,
)

__all__ = [
    # Root + runtime
    "AppConfig",
    # Camera
    "ArucoDictName",
    "BaseRigConfig",
    "CalibrationConfig",
    "CameraConfig",
    "CameraSystemConfig",
    "DummyConfig",
    "EyeHandRoutineConfig",
    "EyeInHandWorkflowConfig",
    "EyeToHandWorkflowConfig",
    "HandEyeConfig",
    # Models
    # Robot
    "GripperConfig",
    "ImageEncodingConfig",
    "InferenceOptimization",
    "ModelsConfig",
    "MotionLimitsConfig",
    "ObjectDetectorConfig",
    "PerceptionModelsConfig",
    "QualityConfig",
    "RGBDCalibPaths",
    "RGBDDeviceRigConfig",
    "RobotCalibrationConfig",
    "RobotCalibrationQualityBandsMm",
    "RobotConfig",
    "RobotSafetyConfig",
    "RuntimeConfig",
    "SafePoseConfig",
    "SegmenterConfig",
    "SimCameraSchema",
    "SimConfig",
    "SingleDeviceRigConfig",
    "SpeechToTextConfig",
    "StereoCalibPaths",
    "StereoMatcherConfig",
    # Base
    "StrictModel",
    "URConfig",
    "WebcamPairRigConfig",
    "WlsFilterConfig",
    "WorkspaceLimitsConfig",
    "primary_camera_calibration_conflict",
    "validate_aruco_dict_name",
]
