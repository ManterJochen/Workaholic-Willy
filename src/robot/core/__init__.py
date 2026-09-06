"""Vendor-neutral core abstractions for the Workaholic-Willy robot subsystem.

This package is the abstract surface every supported robot driver satisfies (UR,
KUKA, Franka, ROS 2, MoveIt, cuRobo, simulators), and the only part of the ``robot``
package that pipelines, planners and the application layer depend on directly.

Public exports:

* :class:`RobotArm`, :class:`Gripper`, :class:`ObjectDetectingGripper` are the driver
  `Protocol`s.
* :class:`RobotVendor`, :class:`GripperVendor` are the canonical driver identifiers.
* :class:`JointPositions` is the typed joint-vector wrapper, in radians.
* :class:`RobotCapabilities` holds the declarative driver feature flags.
* :class:`MotionResult`, :class:`MotionStatus`, :class:`MotionCommand` are the typed
  motion-outcome contract.
* :class:`RobotError` and its subclasses are the vendor-neutral error hierarchy.

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
from .capabilities import RobotCapabilities
from .errors import (
    IsaacNotAvailableError,
    RobotConnectionError,
    RobotEmergencyStop,
    RobotError,
    RobotKinematicsError,
    RobotMotionRejected,
    RobotSingularityRisk,
)
from .gripper import Gripper, ObjectDetectingGripper
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
    "DigitalIOPort",
    "Gripper",
    "GripperVendor",
    "IsaacNotAvailableError",
    "JointPositions",
    "MotionCommand",
    "MotionResult",
    "MotionStatus",
    "NO_PLAN_FAIL_SAFE_MESSAGE",
    "ObjectDetectingGripper",
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
    "SupportsRobotStatus",
    "Wrench",
]
