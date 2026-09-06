"""Robot package: vendor-neutral arm and gripper abstractions plus concrete drivers.

Quick start::

    from src.config.loader import load_config
    from src.robot.core import RobotVendor
    from src.robot.drivers import create_arm

    cfg = load_config()
    with create_arm(RobotVendor.from_string(cfg.robot.vendor), config=cfg.robot) as bot:
        pose = bot.get_tcp_pose()         # vendor-neutral Pose, Frame.BASE
        bot.move_linear(target_pose)      # Pose-typed
        bot.move_home()                   # driver-provided home move

Architecture
------------
* :mod:`.core`         vendor-neutral protocols and typed primitives.
* :mod:`.drivers`      vendor-specific arm drivers.
* :mod:`.grippers`     vendor-specific gripper drivers.
* :mod:`.safety`       workspace and motion guards.
* :mod:`.execution`    motion, pose provisioning, calibration routines.
* :mod:`.core.capabilities` public capability descriptors.

Lazy re-exports
---------------
The vendor-neutral types under :mod:`.core` are re-exported eagerly, so most
callers can write ``from src.robot import RobotArm, JointPositions``. The
concrete UR-facing names are re-exported lazily through :pep:`562`
``__getattr__``, so importing :mod:`src.robot.core` for vendor-free work does not
pull in the UR driver chain.

Numerics contract
-----------------
Inside this package translations are in millimetres and rotations follow the
geometry subsystem: XYZW quaternions in ``float64``. Axis-angle URPose
conversions happen at the :mod:`src.geometry.conversions` boundary and nowhere
else.
"""

from __future__ import annotations

# Eager: vendor-free types only.
from .constants import HOME_JOINTS_DEFAULT, ROBOT_LOG_DIR, ROBOT_LOG_FILE
from .core import (
    Gripper,
    GripperVendor,
    JointPositions,
    RobotArm,
    RobotCapabilities,
    RobotConnectionError,
    RobotEmergencyStop,
    RobotError,
    RobotKinematicsError,
    RobotMotionRejected,
    RobotSingularityRisk,
    RobotVendor,
)

# Lazy proxies for the concrete UR-facing names.
# Each entry: attribute_name -> (relative_module, attribute_in_module).
_LAZY: dict[str, tuple[str, str]] = {
    "Robot": (".drivers.ur.arm", "URRobotArm"),
    "URRobotArm": (".drivers.ur.arm", "URRobotArm"),
    "URConnection": (".drivers.ur", "URConnection"),
    "GripperController": (".grippers", "GripperController"),
    "URPose": (".drivers.ur.pose", "URPose"),
    "WorkspaceGuard": (".safety", "WorkspaceGuard"),
    "SingularityThresholds": (".safety", "SingularityThresholds"),
    "SingularityReport": (".safety", "SingularityReport"),
    "SingularityGuard": (".safety", "SingularityGuard"),
    "SafetyDecision": (".safety", "SafetyDecision"),
    "SafetyReason": (".safety", "SafetyReason"),
    "SafetyContext": (".safety", "SafetyContext"),
    "SafetyGuard": (".safety", "SafetyGuard"),
    "SafetyPreflight": (".safety", "SafetyPreflight"),
    "WorkspaceSafetyGuard": (".safety", "WorkspaceSafetyGuard"),
    "JointLimitGuard": (".safety", "JointLimitGuard"),
    "IKQualityGuard": (".safety", "IKQualityGuard"),
    "MotionContinuityGuard": (".safety", "MotionContinuityGuard"),
    "PayloadGuard": (".safety", "PayloadGuard"),
    "SelfCollisionGuard": (".safety", "SelfCollisionGuard"),
    "analyze_joint_singularity": (".safety", "analyze_joint_singularity"),
    "analyze_pose_singularity": (".safety", "analyze_pose_singularity"),
    "assert_pose_not_singular": (".safety", "assert_pose_not_singular"),
    "MotionController": (".drivers.ur.motion", "MotionController"),
    "PoseProvider": (".execution", "PoseProvider"),
    "CalibrationRoutine": (".execution", "CalibrationRoutine"),
    "CalibrationResult": (".execution", "CalibrationResult"),
    "RobotCalibrationEvent": (".events", "RobotCalibrationEvent"),
    "RobotCalibrationEventListener": (".events", "RobotCalibrationEventListener"),
}


def __getattr__(name: str):
    """Resolve the heavy, UR-bound names on first access, per :pep:`562`."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'src.robot' has no attribute {name!r}")
    module_name, attr = target
    from importlib import import_module

    module = import_module(module_name, package=__name__)
    value = getattr(module, attr)
    globals()[name] = value  # cache, so the next access is direct
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(_LAZY.keys())))


__all__ = [
    "HOME_JOINTS_DEFAULT",
    "ROBOT_LOG_DIR",
    "ROBOT_LOG_FILE",
    "CalibrationResult",
    "CalibrationRoutine",
    "Gripper",
    "GripperController",
    "GripperVendor",
    "JointPositions",
    "MotionController",
    "PoseProvider",
    # Lazy UR-facing facade + helpers
    "Robot",
    # Vendor-neutral core
    "RobotArm",
    "RobotCalibrationEvent",
    "RobotCalibrationEventListener",
    "RobotCapabilities",
    "RobotConnectionError",
    "RobotEmergencyStop",
    "RobotError",
    "RobotKinematicsError",
    "RobotMotionRejected",
    "RobotSingularityRisk",
    "RobotVendor",
    "SafetyContext",
    "SafetyDecision",
    "SafetyGuard",
    "SafetyPreflight",
    "SafetyReason",
    "SelfCollisionGuard",
    "JointLimitGuard",
    "IKQualityGuard",
    "MotionContinuityGuard",
    "PayloadGuard",
    "SingularityGuard",
    "SingularityReport",
    "SingularityThresholds",
    "URConnection",
    "URPose",
    "URRobotArm",
    "WorkspaceGuard",
    "WorkspaceSafetyGuard",
    "analyze_joint_singularity",
    "analyze_pose_singularity",
    "assert_pose_not_singular",
]
