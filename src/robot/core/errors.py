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
        +-- CameraWorldUnavailable  (a camera could not vouch for the cell after its attempts)
        `-- IsaacNotAvailableError  (sim/SDK backend not installed on this host)
"""

from __future__ import annotations

__all__ = [
    "CameraWorldUnavailable",
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

    ``result`` carries the typed :class:`~src.robot.core.motion_result.MotionResult` the
    refusal came from, where one exists. The bool-and-raise verbs and their typed twins
    refuse for the same reasons and through the same gates, so the raise carries the
    status, the target and the camera-world stamp instead of one string. It is ``None``
    for a refusal raised before any gate produced a result, such as a wrong number of
    joints.
    """

    def __init__(self, *args: object, result: object | None = None) -> None:
        super().__init__(*args)
        #: The refusal as the typed verbs report it, or ``None``. Typed as ``object`` so
        #: the error module stays importable from anywhere: ``motion_result`` imports the
        #: camera world, which imports back into core, and a cycle here would be paid by
        #: every module in the package.
        self.result = result


class RobotSingularityRisk(RobotMotionRejected):
    """Raised when a target is rejected because singularity risk is too high."""


class RobotEmergencyStop(RobotError):
    """Raised when the controller reports an active e-stop or protective stop."""


class CameraWorldUnavailable(RobotError):
    """Raised when a camera could not vouch for the cell after its fresh-frame attempts.

    This is a fault of the cell rather than the refusal of one motion: the camera answered nothing,
    a blind frame, or only frames older than the cell allows, and asking it again did not change
    that. It raises out of every verb and out of ``pick()``, where an ordinary refusal returns a
    status, so a campaign stops on a dead camera instead of retrying it as though a grasp had
    missed.

    It is neither a ``CuroboUnavailableError``, which driver sites turn into CONTROLLER_REJECTED,
    nor a :class:`RobotMotionRejected`, which is the ordinary refusal.
    """

    def __init__(
        self,
        *,
        camera: str,
        verdict: object,
        attempts: int,
        reason: str,
        refresh: object | None = None,
    ) -> None:
        #: The camera that could not vouch, as the cell names it.
        self.camera = str(camera)
        #: What it answered on the last reading: no frame, blind or stale.
        self.verdict = verdict
        #: How many readings were asked of it, the first included.
        self.attempts = int(attempts)
        #: The sentence the world source gave for the last reading.
        self.reason = str(reason)
        #: The refused refresh as a report carries it, or ``None``. Typed as ``object`` for the
        #: same import cycle rule as :attr:`RobotMotionRejected.result`.
        self.refresh = refresh
        super().__init__(
            f"camera {self.camera!r} could not vouch for the cell after {self.attempts} reading(s) "
            f"({verdict}): {self.reason}"
        )


class IsaacNotAvailableError(RobotError, RuntimeError):
    """Raised when a simulator or SDK feature is requested on a host without it.

    This is the canonical "the sim backend is not installed here" error, for instance
    Isaac Sim on a macOS host. It subclasses :class:`RobotError` so it belongs to the
    vendor-neutral taxonomy, and :class:`RuntimeError` so an existing
    ``except RuntimeError`` guard keeps catching it. A caller catches this type
    directly and surfaces a :class:`~src.robot.core.MotionResult` carrying
    :attr:`~src.robot.core.MotionStatus.UNSUPPORTED`.
    """
