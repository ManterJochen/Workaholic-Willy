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
    WebcamPairRigConfig,
)
from .models import (
    ObjectDetectorConfig,
    SegmenterConfig,
    SpeechToTextConfig,
)
from .robot import (
    DummyConfig,
    GripperConfig,
    RobotCalibrationConfig,
    RobotCalibrationQualityBandsMm,
    RobotConfig,
    RobotSafetyConfig,
    SimCameraSchema,
    SimConfig,
    URConfig,
    WorkspaceLimitsConfig,
)
from .runtime import (
    GreetingConfig,
    GreetingWave,
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
    "GreetingConfig",
    "GreetingWave",
    "GripperConfig",
    "ImageEncodingConfig",
    "ModelsConfig",
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
    "SegmenterConfig",
    "SimCameraSchema",
    "SimConfig",
    "SingleDeviceRigConfig",
    "SpeechToTextConfig",
    "StereoCalibPaths",
    # Base
    "StrictModel",
    "URConfig",
    "WebcamPairRigConfig",
    "WorkspaceLimitsConfig",
    "validate_aruco_dict_name",
]
