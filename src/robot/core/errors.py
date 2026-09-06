"""Vendor-neutral error hierarchy for the Workaholic-Willy robot subsystem.

Every driver translates its vendor-specific failures (``ur_rtde`` exceptions, ROS 2
service errors, ``franky`` faults) into one of these types before re-raising, so
application and pipeline code catches only the abstract types defined here.

Hierarchy::

    RobotError
        +-- RobotConnectionError    (cannot reach controller, dropped link)
        +-- RobotKinematicsError    (FK / IK failure, unreachable target)
        +-- RobotMotionRejected     (workspace guard / safety pre-check denied)
        |       `-- RobotSingularityRisk   (target too close to a singularity)
        +-- RobotEmergencyStop      (e-stop / protective stop active)
        `-- IsaacNotAvailableError  (sim/SDK backend not installed on this host)
"""

from __future__ import annotations

__all__ = [
    "IsaacNotAvailableError",
    "RobotConnectionError",
    "RobotEmergencyStop",
    "RobotError",
    "RobotKinematicsError",
    "RobotMotionRejected",
    "RobotSingularityRisk",
]


class RobotError(Exception):
    """Base class for all vendor-neutral robot failures."""


class RobotConnectionError(RobotError):
    """Raised when the controller link cannot be opened or has dropped."""


class RobotKinematicsError(RobotError):
    """Raised when forward or inverse kinematics fail, or a target is unreachable."""


class RobotMotionRejected(RobotError):
    """Raised when a motion request is rejected before it reaches the driver.

    The causes are a workspace-guard rejection, a joint-limit pre-check failure, a
    soft-limit violation, or an input-validation error.
    """


class RobotSingularityRisk(RobotMotionRejected):
    """Raised when a target is rejected because singularity risk is too high."""


class RobotEmergencyStop(RobotError):
    """Raised when the controller reports an active e-stop or protective stop."""


class IsaacNotAvailableError(RobotError, RuntimeError):
    """Raised when a simulator or SDK feature is requested on a host without it.

    This is the canonical "the sim backend is not installed here" error, for instance
    Isaac Sim on a macOS host. It subclasses :class:`RobotError` so it belongs to the
    vendor-neutral taxonomy, and :class:`RuntimeError` so an existing
    ``except RuntimeError`` guard keeps catching it. A caller catches this type
    directly and surfaces a :class:`~src.robot.core.MotionResult` carrying
    :attr:`~src.robot.core.MotionStatus.UNSUPPORTED`.
    """
