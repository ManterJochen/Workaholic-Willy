"""SafetyDecision, the typed outcome of a single safety-guard evaluation.

The safety-preflight pipeline (joint limit, IK quality, self-collision, payload,
motion continuity, workspace) runs before every commanded motion, and each guard
reports its verdict in a form that is unambiguous, so a caller parses no strings;
structured, so it carries enough context for a log, a UI or an event; and directly
mappable onto the closed :class:`MotionStatus` set, so a typed :class:`MotionResult`
is assembled at the driver boundary without further classification.

:class:`SafetyDecision` is that wire type. :class:`SafetyReason` is the closed set of
verdict categories, one per guard family plus ``OK`` and ``UNAVAILABLE``.

Numerics contract
-----------------
``SafetyDecision`` carries a free-form ``detail: dict[str, str]``, so a guard attaches
structured context such as the offending joint index or the measured singular value
without losing precision. A guard formats a numeric value into a string with enough
precision for diagnostics tooling to re-parse it. The dict is for a log or an event
payload and not for cross-guard arithmetic.

When does a guard return ``UNAVAILABLE``?
-----------------------------------------
A guard reports :attr:`SafetyReason.UNAVAILABLE` when it lacks the data to decide and
the operator has not opted in to failing closed for that condition, for instance:

* the self-collision guard is configured for the ``fcl`` backend and the configured
  mesh directory is empty;
* the joint-limit guard has no static fallback limits and the connected driver exposes
  no telemetry, as with KUKA EKI.

``UNAVAILABLE`` does not itself accept the motion. :meth:`SafetyPreflight.evaluate`
decides whether to fail open or fail closed from the guard ``enforce`` flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.robot.core import MotionStatus

__all__ = [
    "SafetyDecision",
    "SafetyReason",
]


class SafetyReason(str, Enum):
    """Closed set of safety-decision reason categories.

    The string values are stable and safe to log or emit on an event stream. Every
    non-``OK`` reason maps to exactly one :class:`MotionStatus` through
    :func:`safety_reason_to_motion_status`, so the driver boundary synthesises a typed
    :class:`MotionResult` without further classification.
    """

    OK = "ok"
    """The guard accepted the commanded motion."""

    WORKSPACE = "workspace"
    """Rejected by the Cartesian workspace box or by orientation diversity."""

    JOINT_LIMIT = "joint_limit"
    """Joint hard limit (or configured margin) would be violated."""

    IK_QUALITY = "ik_quality"
    """The IK solution exists but fails a quality check: a singularity, a jump, or
    proximity to a limit."""

    SELF_COLLISION = "self_collision"
    """Commanded configuration would intersect the arm / tool / fixture."""

    PAYLOAD = "payload"
    """The configured payload is outside the allowed mass, centre-of-gravity or
    inertia envelope."""

    CONTINUITY = "continuity"
    """Abrupt joint or orientation step between consecutive commands."""

    UNAVAILABLE = "unavailable"
    """The guard could not evaluate, for want of telemetry or an asset. The preflight
    decides whether to fail open or fail closed from the guard ``enforce`` flag."""


# ---------------------------------------------------------------------
# SafetyReason -> MotionStatus mapping
# ---------------------------------------------------------------------
#
# This table is the single source of truth for translating a safety verdict into the
# closed :class:`MotionStatus` set the driver boundary emits on a typed
# :class:`MotionResult`. Every non-``OK`` reason has an entry.

_REASON_TO_STATUS: dict[SafetyReason, MotionStatus] = {
    SafetyReason.WORKSPACE: MotionStatus.WORKSPACE_REJECTED,
    SafetyReason.JOINT_LIMIT: MotionStatus.JOINT_LIMIT_REJECTED,
    SafetyReason.IK_QUALITY: MotionStatus.IK_QUALITY_REJECTED,
    SafetyReason.SELF_COLLISION: MotionStatus.SELF_COLLISION_REJECTED,
    SafetyReason.PAYLOAD: MotionStatus.PAYLOAD_REJECTED,
    SafetyReason.CONTINUITY: MotionStatus.CONTINUITY_REJECTED,
    # ``UNAVAILABLE`` maps to CONTROLLER_REJECTED by default, because the only way it
    # reaches the driver boundary is an ``enforce``-set guard failing closed without
    # enough information to name the underlying cause. A caller may pass an explicit
    # override through ``SafetyDecision.unavailable(motion_status=...)``.
    SafetyReason.UNAVAILABLE: MotionStatus.CONTROLLER_REJECTED,
}


def safety_reason_to_motion_status(reason: SafetyReason) -> MotionStatus:
    """Translate a non-``OK`` :class:`SafetyReason` into a :class:`MotionStatus`.

    Raises
    ------
    ValueError
        If ``reason`` is :attr:`SafetyReason.OK`, which represents acceptance and has
        no failure status.
    """
    if reason is SafetyReason.OK:
        raise ValueError(
            "SafetyReason.OK has no MotionStatus mapping; the motion "
            "is accepted and no MotionResult should be synthesised."
        )
    try:
        return _REASON_TO_STATUS[reason]
    except KeyError as exc:  # pragma: no cover (exhaustive enum)
        raise ValueError(
            f"No MotionStatus mapping for SafetyReason {reason!r}."
        ) from exc


@dataclass(frozen=True)
class SafetyDecision:
    """Immutable verdict produced by a single safety guard.

    Parameters
    ----------
    accepted
        ``True`` only where the guard accepts the commanded motion.
    reason
        Closed-set category of the verdict. It is :attr:`SafetyReason.OK` when
        ``accepted`` is ``True``.
    guard
        Short identifier of the producing guard, such as ``"workspace"`` or
        ``"joint_limit"``, so a log or a UI can say which guard refused.
    message
        Message suitable for direct display in an operator UI or a log. It may be
        empty for ``accepted=True``.
    detail
        Free-form structured detail such as the offending joint index or the measured
        singular value. It is a ``dict[str, str]``, so it serialises straight into a
        structured event payload without further coercion.
    motion_status_override
        Optional :class:`MotionStatus` for the driver boundary, in place of the default
        mapping for ``reason``. It is consulted on rejection only and ignored when
        ``accepted`` is ``True``. :meth:`unavailable` uses it, so a guard that fails
        closed on missing data still emits a precise driver status where one is known.

    Construction
    ------------
    Call :meth:`accept`, :meth:`reject` or :meth:`unavailable` rather than the bare
    constructor.
    """

    accepted: bool
    reason: SafetyReason
    guard: str
    message: str = ""
    detail: dict[str, str] = field(default_factory=dict)
    motion_status_override: MotionStatus | None = None

    def __post_init__(self) -> None:
        if self.accepted and self.reason is not SafetyReason.OK:
            raise ValueError(
                "SafetyDecision.accepted=True requires SafetyReason.OK; "
                f"got reason={self.reason!r}."
            )
        if not self.accepted and self.reason is SafetyReason.OK:
            raise ValueError(
                "SafetyDecision.accepted=False forbids SafetyReason.OK; "
                "use one of the rejection reasons."
            )
        if not isinstance(self.guard, str) or not self.guard:
            raise ValueError(
                "SafetyDecision.guard must be a non-empty string."
            )

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    @property
    def rejected(self) -> bool:
        """``True`` only where this decision refuses the motion."""
        return not self.accepted

    def __bool__(self) -> bool:  # pragma: no cover (trivial)
        """Let ``if decision: ...`` mean that the decision was accepted."""
        return self.accepted

    @property
    def motion_status(self) -> MotionStatus | None:
        """The :class:`MotionStatus` to surface at the driver boundary.

        It is ``None`` where the decision is accepted, since no result is needed. It is
        the explicit override where one is set, and otherwise the default mapping for
        :attr:`reason`.
        """
        if self.accepted:
            return None
        if self.motion_status_override is not None:
            return self.motion_status_override
        return safety_reason_to_motion_status(self.reason)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def accept(cls, guard: str, *, message: str = "") -> "SafetyDecision":
        """Build an accepted decision for ``guard``."""
        return cls(
            accepted=True,
            reason=SafetyReason.OK,
            guard=guard,
            message=message,
        )

    @classmethod
    def reject(
        cls,
        guard: str,
        reason: SafetyReason,
        *,
        message: str = "",
        detail: dict[str, str] | None = None,
        motion_status_override: MotionStatus | None = None,
    ) -> "SafetyDecision":
        """Build a rejection decision.

        ``reason`` must not be :attr:`SafetyReason.OK`.
        """
        if reason is SafetyReason.OK:
            raise ValueError(
                "SafetyDecision.reject requires a non-OK reason; use "
                "SafetyDecision.accept for acceptance."
            )
        return cls(
            accepted=False,
            reason=reason,
            guard=guard,
            message=message,
            detail=dict(detail) if detail else {},
            motion_status_override=motion_status_override,
        )

    @classmethod
    def unavailable(
        cls,
        guard: str,
        *,
        message: str = "",
        detail: dict[str, str] | None = None,
        motion_status_override: MotionStatus | None = None,
    ) -> "SafetyDecision":
        """Build an ``UNAVAILABLE`` decision.

        The preflight orchestrator decides whether it fails open or fails closed from
        the guard ``enforce`` flag.
        """
        return cls(
            accepted=False,
            reason=SafetyReason.UNAVAILABLE,
            guard=guard,
            message=message,
            detail=dict(detail) if detail else {},
            motion_status_override=motion_status_override,
        )
