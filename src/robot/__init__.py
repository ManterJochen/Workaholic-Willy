"""Robot package: vendor-neutral arm and gripper abstractions plus concrete drivers.

Quick start::

    from src.config.loader import load_config
    from src.robot.core import RobotVendor
    from src.robot.drivers import create_arm

    cfg = load_config()
    with create_arm(RobotVendor.from_string(cfg.robot.vendor), config=cfg.robot) as bot:
        pose = bot.get_tcp_pose()         # vendor-neutral Pose, Frame.BASE
        # A cuRobo UR arm refuses a motion with no live camera world and no decline.
        with bot.without_camera_world("bench move: no camera watches the cell"):
            bot.move_linear(target_pose)  # Pose-typed
            bot.move_home()               # driver-provided home move

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
callers can write ``from src.robot import RobotArm, JointPositions``. The heavier
vendor-neutral names (the safety guards, ``Robot``, the calibration routine) are
re-exported lazily through :pep:`562` ``__getattr__``. No driver class is: a
vendor's classes live in :mod:`src.robot.drivers` and reach a caller as a
:class:`RobotArm` from ``create_arm``.

Numerics contract
-----------------
Inside this package translations are in millimetres and rotations follow the
geometry subsystem: XYZW quaternions in ``float64``. Axis-angle URPose
conversions happen at the UR driver's boundary,
:mod:`src.robot.drivers.ur.pose_adapter`, and nowhere else.
"""

from __future__ import annotations

# Eager: vendor-free types only.
from .constants import ROBOT_LOG_DIR, ROBOT_LOG_FILE
from .core import (
    ArmFault,
    ArmFaultClass,
    Gripper,
    GripperVendor,
    JointPositions,
    RobotArm,
    RobotCapabilities,
    RobotConnectionError,
    RobotError,
    RobotKinematicsError,
    RobotMotionRejected,
    RobotSingularityRisk,
    RobotVendor,
)

# Lazy proxies for the vendor-neutral names. No driver is reachable from here: a vendor's classes live in
# `robot.drivers.<vendor>` and reach the stack as a `RobotArm` from `create_arm`, so the package root cannot
# become the door through which one vendor's types spread (the four UR entries left on 2026-09-20).
# Each entry: attribute_name -> (relative_module, attribute_in_module).
_LAZY: dict[str, tuple[str, str]] = {
    "Robot": (".execution", "Robot"),
    "GripperController": (".grippers", "GripperController"),
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
    "ROBOT_LOG_DIR",
    "ROBOT_LOG_FILE",
    "CalibrationResult",
    "CalibrationRoutine",
    "Gripper",
    "GripperController",
    "GripperVendor",
    "JointPositions",
    "PoseProvider",
    # The arm and the gripper as one noun (execution/robot.py)
    "Robot",
    # Vendor-neutral core
    "RobotArm",
    "RobotCalibrationEvent",
    "RobotCalibrationEventListener",
    "RobotCapabilities",
    "RobotConnectionError",
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
    "WorkspaceGuard",
    "WorkspaceSafetyGuard",
    "analyze_joint_singularity",
    "analyze_pose_singularity",
    "assert_pose_not_singular",
]
