"""Typed execution results for the vendor-neutral motion contract.

:meth:`RobotArm.move` returns the canonical typed result defined here. The
bool-returning high-level methods (``move_to``, ``move_home``,
``wait_until_steady``) remain as compatibility shims, and a bool collapses every
non-success outcome into ``False``, so a workspace rejection cannot be told from an
IK failure or a controller fault.

Contract
--------
* :class:`MotionStatus` is the closed set of outcome categories a driver classifies
  into.
* :class:`MotionCommand` names the high-level command that produced the result, so a
  policy or runtime report carries context without parsing strings.
* :class:`MotionResult` is the immutable wire type returned by ``RobotArm.move``.

When may a driver raise?
~~~~~~~~~~~~~~~~~~~~~~~~
An ordinary execution outcome (workspace rejection, IK failure, controller refusal,
timeout, marker not detected) is returned as a typed :class:`MotionResult` and never
raised. A driver may raise on a connection or transport fault if it prefers the
exception path, so a caller can rely on a typed result for everything that is not a
true runtime fault. :meth:`MotionResult.from_bool` and :meth:`MotionResult.executed`
build the dataclass, so a driver author does not hand-roll one per code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.geometry import Pose

    from .joint_positions import JointPositions

__all__ = [
    "MotionCommand",
    "MotionResult",
    "MotionStatus",
]


class MotionStatus(StrEnum):
    """Closed set of motion-result categories.

    The string values are stable and safe to log or emit on an event stream, and a
    downstream consumer may switch on the enum directly.
    """

    EXECUTED = "executed"
    """The command was accepted and completed without fault."""

    WORKSPACE_REJECTED = "workspace_rejected"
    """A pre-flight workspace or safety guard refused the target."""

    IK_FAILED = "ik_failed"
    """The driver could not find a valid joint solution for the target."""

    CONTROLLER_REJECTED = "controller_rejected"
    """The controller refused the command, for instance on a protective stop."""

    TIMEOUT = "timeout"
    """The command did not complete within the configured time budget."""

    CONNECTION_ERROR = "connection_error"
    """The transport or link to the controller is down or faulted."""

    UNSUPPORTED = "unsupported"
    """This driver does not implement the requested command kind."""

    INVALID_TARGET = "invalid_target"
    """The target argument is malformed, for instance a wrong frame or a NaN."""

    CANCELLED = "cancelled"
    """The caller cancelled the move before it completed."""

    UNKNOWN = "unknown"
    """The driver could not classify the failure. Avoid where possible."""

    # ------------------------------------------------------------------
    # Safety-guard rejections
    # ------------------------------------------------------------------
    #
    # The vendor-neutral :class:`src.robot.safety.SafetyPreflight` pipeline produces
    # these categories. A driver surfaces them verbatim when the preflight
    # short-circuits, so an operator or an orchestration layer branches on the
    # precise cause instead of scraping a log.

    JOINT_LIMIT_REJECTED = "joint_limit_rejected"
    """A joint target violates a configured hard limit, or the configured safety
    margin to that limit."""

    IK_QUALITY_REJECTED = "ik_quality_rejected"
    """The IK solution exists but fails a quality check: NaN, wrong DoF, an
    excessive joint jump, or proximity to a singularity or a joint limit."""

    SELF_COLLISION_REJECTED = "self_collision_rejected"
    """The commanded configuration would intersect the arm with itself, its base,
    its tool, or a declared fixture."""

    PAYLOAD_REJECTED = "payload_rejected"
    """The configured payload is outside the allowed mass, centre-of-gravity or
    inertia envelope."""

    CONTINUITY_REJECTED = "continuity_rejected"
    """The motion-continuity guard refused an abrupt joint or orientation jump
    between consecutive commands."""


NO_PLAN_FAIL_SAFE_MESSAGE = (
    "cuRobo found no collision-free plan; failing safe (no blind motion)."
)
"""The one motion message that means the arm never moved.

``MotionStatus.TIMEOUT`` covers two events that must not be confused. A
motion-planner refusal happens before any command reaches the controller: nothing
moved, the cell is exactly where it was, and the honest response is to look again. A
genuine execution timeout means the command was accepted and did not finish in its
budget, so the arm may still be in motion, and acting on that state without a stop is
not an orchestration layer's decision to make.

The status alone cannot tell the two apart, so the message carries the distinction
and both the sim and the UR driver emit this exact constant for the refusal. A
consumer that branches on nothing having moved matches on this, never on ``TIMEOUT``
alone.
"""


class MotionCommand(StrEnum):
    """High-level command kind that produced a :class:`MotionResult`."""

    MOVE_TO = "move_to"
    MOVE_HOME = "move_home"
    MOVE_JOINTS = "move_joints"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class MotionResult:
    """Immutable typed outcome of one motion command.

    Parameters
    ----------
    status
        Closed-set outcome category. ``EXECUTED`` is the only success value;
        everything else is a flavour of failure.
    command
        Which high-level command produced this result.
    target_pose
        The :class:`Pose` that was commanded, where one applies.
    target_joints
        The :class:`JointPositions` that were commanded, where they apply.
    message
        Optional detail for a log or a UI. May be empty.
    exception
        Optional underlying exception for a ``CONNECTION_ERROR`` or ``UNKNOWN``
        fault, where the original traceback helps diagnosis. May be ``None``.

    Notes
    -----
    The class is frozen, so it passes through layers without accidental mutation.
    Construct it through :meth:`executed`, :meth:`failed` or :meth:`from_bool` rather
    than the bare constructor, which reads more clearly at a call site.
    """

    status: MotionStatus
    command: MotionCommand
    target_pose: "Pose | None" = None
    target_joints: "JointPositions | None" = None
    message: str = ""
    exception: BaseException | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    @property
    def ok(self) -> bool:
        """``True`` only where the command executed successfully."""
        return self.status is MotionStatus.EXECUTED

    def __bool__(self) -> bool:  # pragma: no cover (trivial)
        """Let ``if result:`` test success directly."""
        return self.ok

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def executed(
        cls,
        command: MotionCommand,
        *,
        target_pose: "Pose | None" = None,
        target_joints: "JointPositions | None" = None,
        message: str = "",
    ) -> "MotionResult":
        """Build a successful result."""
        return cls(
            status=MotionStatus.EXECUTED,
            command=command,
            target_pose=target_pose,
            target_joints=target_joints,
            message=message,
        )

    @classmethod
    def failed(
        cls,
        status: MotionStatus,
        command: MotionCommand,
        *,
        target_pose: "Pose | None" = None,
        target_joints: "JointPositions | None" = None,
        message: str = "",
        exception: BaseException | None = None,
    ) -> "MotionResult":
        """Build a failure result with an explicit ``status``."""
        if status is MotionStatus.EXECUTED:
            raise ValueError(
                "MotionResult.failed requires a non-EXECUTED status; "
                "use MotionResult.executed for success."
            )
        return cls(
            status=status,
            command=command,
            target_pose=target_pose,
            target_joints=target_joints,
            message=message,
            exception=exception,
        )

    @classmethod
    def from_bool(
        cls,
        ok: bool,
        command: MotionCommand,
        *,
        target_pose: "Pose | None" = None,
        target_joints: "JointPositions | None" = None,
        failure_status: MotionStatus = MotionStatus.CONTROLLER_REJECTED,
        message: str = "",
    ) -> "MotionResult":
        """Bridge a bool return into a typed result.

        ``failure_status`` is the category assigned when ``ok`` is ``False``. The
        default :attr:`MotionStatus.CONTROLLER_REJECTED` is the most common cause for
        a driver that returns ``False`` without a richer classification. A caller
        that has already proved a more specific cause, such as ``WORKSPACE_REJECTED``
        from a pre-flight check, sets ``failure_status`` explicitly.
        """
        if ok:
            return cls.executed(
                command,
                target_pose=target_pose,
                target_joints=target_joints,
                message=message,
            )
        return cls.failed(
            failure_status,
            command,
            target_pose=target_pose,
            target_joints=target_joints,
            message=message,
        )
