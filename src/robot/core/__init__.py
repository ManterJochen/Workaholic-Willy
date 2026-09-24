"""Vendor-neutral core abstractions for the Workaholic-Willy robot subsystem.

This package is the abstract surface every supported robot driver satisfies (UR,
KUKA, Franka, ROS 2, MoveIt, cuRobo, simulators), and the only part of the ``robot``
package that pipelines, planners and the application layer depend on directly.

Public exports:

* :class:`RobotArm`, :class:`Gripper`, :class:`ObjectDetectingGripper` and
  :class:`StoppableGripper` are the driver `Protocol`s.
* :class:`TogglesWithoutSensor` is a hand whose every command is one pulse that flips its
  jaws, with nothing to read them back; :func:`toggle_without_sensor_of` finds one, so a pick
  asks it before the arm moves instead of commanding an open, with no driver import.
* :class:`RobotVendor`, :class:`GripperVendor` are the canonical driver identifiers.
* :class:`JointPositions` is the typed joint-vector wrapper, in radians.
* :class:`RobotCapabilities` holds the declarative driver feature flags.
* :class:`MotionResult`, :class:`MotionStatus`, :class:`MotionCommand` are the typed
  motion-outcome contract.
* :class:`CameraWorldStamp`, :class:`CameraWorldUse` and :class:`CameraWorldDecline`
  say whether a camera world stood behind a motion, carried on its result;
  :func:`without_camera_world`, :func:`active_decline`, :func:`resolve_camera_world`,
  :func:`stamp_result` and the :class:`DeclinesCameraWorld` capability say how a driver
  declines, reads a decline and stamps its motions, and :func:`camera_world_refusal` with
  ``NO_CAMERA_WORLD_MESSAGE`` says which stamped motions a driver refuses before they move.
* :class:`RobotError` and its subclasses are the vendor-neutral error hierarchy.
* :class:`SupportsFreedrive` is hand guiding as a vendor capability: a :class:`FreedriveSession`
  frees the arm for a person and reads it as :class:`FreedriveSample`, and
  :class:`ControllerPayload` is the payload the controller compensates for while it is free.

The numerics contract mirrors :mod:`src.geometry`: translations in millimetres,
orientations as unit XYZW quaternions with a canonical sign, joint angles in radians,
all of them ``float64``. Every public ndarray this layer returns is read-only.
"""

from __future__ import annotations

from .arm_capabilities import (
    DigitalIOPort,
    RobotMode,
    RobotStatus,
    SafetyMode,
    SupportsDigitalIO,
    SupportsForceTorque,
    SupportsRobotStatus,
    Wrench,
)
from .camera_world import (
    DECLINE_ON_A_LIVE_WORLD_MESSAGE,
    NO_CAMERA_WORLD_MESSAGE,
    CameraWorldDecline,
    CameraWorldStamp,
    CameraWorldUse,
    DeclinesCameraWorld,
    active_decline,
    camera_world_refusal,
    resolve_camera_world,
    stamp_result,
    without_camera_world,
)
from .capabilities import RobotCapabilities
from .errors import (
    PerceptionFrameMoved,
    CameraWorldUnavailable,
    IsaacNotAvailableError,
    RobotConnectionError,
    RobotEmergencyStop,
    RobotError,
    RobotKinematicsError,
    RobotMotionRejected,
    RobotSingularityRisk,
)
from .freedrive import (
    ControllerPayload,
    FreedriveSample,
    FreedriveSession,
    SupportsFreedrive,
)
from .gripper import (
    Gripper,
    ObjectDetectingGripper,
    StoppableGripper,
    TogglesWithoutSensor,
    TwoStateGripper,
    toggle_without_sensor_of,
)
from .gripper_vendor import GripperVendor
from .joint_positions import JointPositions
from .motion_result import (
    NO_PLAN_FAIL_SAFE_MESSAGE,
    MotionCommand,
    MotionResult,
    MotionStatus,
)
from .robot_arm import RobotArm
from .vendor import RobotVendor

__all__ = [
    "CameraWorldDecline",
    "CameraWorldStamp",
    "CameraWorldUse",
    "ControllerPayload",
    "DECLINE_ON_A_LIVE_WORLD_MESSAGE",
    "NO_CAMERA_WORLD_MESSAGE",
    "DeclinesCameraWorld",
    "CameraWorldUnavailable",
    "DigitalIOPort",
    "FreedriveSample",
    "FreedriveSession",
    "Gripper",
    "GripperVendor",
    "IsaacNotAvailableError",
    "JointPositions",
    "MotionCommand",
    "MotionResult",
    "MotionStatus",
    "NO_PLAN_FAIL_SAFE_MESSAGE",
    "ObjectDetectingGripper",
    "StoppableGripper",
    "TogglesWithoutSensor",
    "TwoStateGripper",
    "RobotArm",
    "RobotCapabilities",
    "RobotConnectionError",
    "RobotEmergencyStop",
    "RobotError",
    "RobotKinematicsError",
    "RobotMode",
    "RobotMotionRejected",
    "RobotSingularityRisk",
    "RobotStatus",
    "RobotVendor",
    "SafetyMode",
    "SupportsDigitalIO",
    "SupportsForceTorque",
    "SupportsFreedrive",
    "SupportsRobotStatus",
    "Wrench",
    "active_decline",
    "camera_world_refusal",
    "resolve_camera_world",
    "stamp_result",
    "toggle_without_sensor_of",
    "without_camera_world",
]
