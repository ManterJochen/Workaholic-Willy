"""Frame resolver: the typed contract for ``T_cam_to_base`` per frame.

:class:`src.robot.grasping.types.perception.PerceptionFrame` carries
intrinsics but not the camera-to-base transform. With no resolver wired,
:meth:`src.robot.grasping.generation.calculator.GraspCalculator.compute_result`
is handed no transform and candidates come back one of two ways:

* implicitly converted by the ``camera_matrix`` the calculator was built
  with plus an unrelated transform passed at orchestrator construction
  time, which is a footgun for an eye-in-hand setup whose transform
  changes with the TCP, or
* in :attr:`src.robot.grasping.types.grasp_point.GraspFrame.CAMERA`, in
  which case the execution policy would silently try to move the arm to
  a camera-frame pose, which is a safety hole.

The :class:`FrameResolver` Protocol closes that gap. The orchestrator
asks a resolver for the current ``T_cam_to_base`` for the moment the
frame was captured, forwards it into the calculator, and the execution
policy fails closed when no resolver is wired and a candidate would
still be camera-frame.

Fail-closed contract
--------------------
If a ``PerceptionFrame`` lacks a usable ``T_cam_to_base`` and no
:class:`FrameResolver` is wired, the high-level service must refuse to
execute rather than fall back to the ``camera_matrix`` the calculator
was built with. This module owns half of that contract, the resolver
Protocol and three built-ins; the orchestrator and the execution policy
own the other half, the forwarding and the fail-closed guard.

This module imports only vendor-neutral surfaces and is safe to import
on macOS with no hardware drivers installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Union, runtime_checkable

import numpy as np

from src.geometry import Frame, Pose, Transform
from src.robot.core import RobotArm
from src.robot.grasping.types.perception import PerceptionFrame

__all__ = [
    "EyeInHandFrameResolver",
    "FrameResolutionFailure",
    "FrameResolver",
    "IdentityFrameResolver",
    "RESOLVE_REASON_BAD_TRANSFORM",
    "RESOLVE_REASON_EXCEPTION",
    "RESOLVE_REASON_NONE_RETURNED",
    "RESOLVE_REASON_NO_RESOLVER",
    "RESOLVE_REASON_WRONG_FRAME",
    "StaticCameraToBaseResolver",
    "resolve_or_none",
]


@runtime_checkable
class FrameResolver(Protocol):
    """Resolve ``T_cam_to_base`` for a given :class:`PerceptionFrame`.

    Implementations must return a :class:`Transform` whose
    ``from_frame`` is :attr:`Frame.CAMERA` and ``to_frame`` is
    :attr:`Frame.BASE`. The resolver is responsible for whatever
    bookkeeping its mounting topology requires: an eye-to-hand resolver
    ignores ``arm``, while an eye-in-hand resolver consults
    :meth:`RobotArm.get_tcp_pose` at call time and composes it with the
    camera-to-tool calibration.

    The resolver must not mutate the frame or the arm, and it must be
    callable repeatedly within one attempt, because the orchestrator may
    invoke it for the initial capture, for a refinement recapture, and
    for active perception viewpoints.
    """

    def camera_to_base_for_frame(
        self,
        frame: PerceptionFrame,
        *,
        arm: RobotArm,
    ) -> Transform: ...


# ---------------------------------------------------------------------------
# Built-in resolvers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StaticCameraToBaseResolver:
    """Eye-to-hand resolver: a fixed ``T_cam_to_base``.

    Use when the camera is bolted to a static fixture and the
    calibration is invariant to the arm pose. The resolver returns
    the same transform on every call and ignores ``arm`` entirely.
    """

    transform: Transform

    def __post_init__(self) -> None:
        if (
            self.transform.from_frame is not Frame.CAMERA
            or self.transform.to_frame is not Frame.BASE
        ):
            raise ValueError(
                "StaticCameraToBaseResolver.transform must be "
                "Transform(CAMERA -> BASE); got "
                f"{self.transform.from_frame.name} -> "
                f"{self.transform.to_frame.name}"
            )

    def camera_to_base_for_frame(
        self,
        frame: PerceptionFrame,  # noqa: ARG002 (unused for eye-to-hand)
        *,
        arm: RobotArm,  # noqa: ARG002 (unused for eye-to-hand)
    ) -> Transform:
        return self.transform


@dataclass(frozen=True, slots=True)
class EyeInHandFrameResolver:
    """Eye-in-hand resolver: compose live TCP pose with camera-on-tool.

    The camera is rigidly mounted to the flange, and so to the TCP. The
    static calibration ``t_cam_to_tool``, from ``CAMERA`` to ``TOOL``, is
    supplied at construction. At resolution time the resolver:

    1. reads the current TCP pose from the arm
       (:meth:`RobotArm.get_tcp_pose`),
    2. reinterprets that pose as ``T_tool_to_base``, since the TCP pose
       in the base frame is the rigid transform that maps tool-frame
       data into the base frame,
    3. composes ``T_cam_to_tool @ T_tool_to_base`` to obtain the live
       ``T_cam_to_base``.

    Failure modes are explicit:

    * Wrong calibration frames raise :class:`ValueError` at
      construction.
    * A TCP pose that is not in :attr:`Frame.BASE` raises
      :class:`ValueError` at resolution. This catches the common
      footgun of a driver returning a tool-frame TCP read.
    """

    t_cam_to_tool: Transform

    def __post_init__(self) -> None:
        if (
            self.t_cam_to_tool.from_frame is not Frame.CAMERA
            or self.t_cam_to_tool.to_frame is not Frame.TOOL
        ):
            raise ValueError(
                "EyeInHandFrameResolver.t_cam_to_tool must be "
                "Transform(CAMERA -> TOOL); got "
                f"{self.t_cam_to_tool.from_frame.name} -> "
                f"{self.t_cam_to_tool.to_frame.name}"
            )

    def camera_to_base_for_frame(
        self,
        frame: PerceptionFrame,
        *,
        arm: RobotArm,
    ) -> Transform:
        """``CAMERA`` to ``BASE`` for the moment this frame was captured, not for the moment of the call.

        The frame is read, not ignored. For a camera bolted to the wrist the transform depends on
        where the tool was when the shutter opened, and the closed-loop path moves the arm between
        capture and resolve by design and then re-perceives. Reading the TCP at resolve time instead
        puts every millimetre the tool travelled in between into the grasp, in a frame nothing
        downstream checks.

        A producer that does not stamp `tool_pose` leaves it `None`, and the resolver then reads the
        arm directly. That fallback is what keeps an unstamped source working; stamping is what makes
        the answer correct.
        """
        tcp_pose = frame.tool_pose if getattr(frame, "tool_pose", None) is not None \
            else arm.get_tcp_pose()
        if not isinstance(tcp_pose, Pose):
            raise TypeError(
                "RobotArm.get_tcp_pose() must return a Pose; got "
                f"{type(tcp_pose).__name__}"
            )
        if tcp_pose.frame is not Frame.BASE:
            raise ValueError(
                "EyeInHandFrameResolver requires arm.get_tcp_pose() in "
                f"Frame.BASE; got {tcp_pose.frame.name}. This usually "
                "means the driver returned a tool-frame reading by "
                "mistake."
            )
        # The TCP pose in base coordinates is a rigid transform that
        # maps tool-frame data into base-frame data: T_tool_to_base.
        t_tool_to_base = Transform.from_matrix(
            np.asarray(tcp_pose.to_matrix(), dtype=np.float64),
            from_frame=Frame.TOOL,
            to_frame=Frame.BASE,
        )
        # Transform.compose semantics: ``a.compose(b)`` returns a
        # transform from ``a.from_frame`` to ``b.to_frame``, meaning
        # apply ``a`` first and then ``b``. Composing
        # ``T_cam_to_tool`` with ``T_tool_to_base`` therefore yields
        # the wanted ``T_cam_to_base``.
        return self.t_cam_to_tool.compose(t_tool_to_base)


@dataclass(frozen=True, slots=True)
class IdentityFrameResolver:
    """Explicit "camera frame == base frame" resolver.

    Use only for synthetic rigs and simulation scenes where the
    perception system reports points already expressed in the base frame
    of the robot. A production deployment must use one of the calibrated
    resolvers. This class exists so that such a rig can wire a resolver
    explicitly rather than rely on the absent-resolver fallback, which
    makes the intent visible where it is wired.
    """

    def camera_to_base_for_frame(
        self,
        frame: PerceptionFrame,  # noqa: ARG002 (unused)
        *,
        arm: RobotArm,  # noqa: ARG002 (unused)
    ) -> Transform:
        return Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)


# ---------------------------------------------------------------------------
# Typed convenience: non-raising resolution wrapper for shadow callers
# ---------------------------------------------------------------------------

# Reason strings, a frozen wire contract. A consumer such as
# multi-view fusion or a telemetry overlay must match against these
# constants rather than against the human-readable ``message``.
RESOLVE_REASON_NO_RESOLVER = "no_resolver"
RESOLVE_REASON_EXCEPTION = "exception"
RESOLVE_REASON_NONE_RETURNED = "none_returned"
RESOLVE_REASON_BAD_TRANSFORM = "bad_transform"
RESOLVE_REASON_WRONG_FRAME = "wrong_frame"


@dataclass(frozen=True, slots=True)
class FrameResolutionFailure:
    """Typed failure carrier returned by :func:`resolve_or_none`.

    ``reason`` is one of the ``RESOLVE_REASON_*`` module-level
    constants and is the stable wire field. ``message`` is a
    human-readable diagnostic for logs and telemetry only. Do not branch
    on it.

    This carrier exists so that a shadow-path consumer, such as
    multi-view fusion or a read-only observer, can ask for the transform
    if it is cheap and safe to produce without entangling itself in the
    fail-closed branch of the execution policy. Anything that executes
    motion must use the resolver directly and let exceptions propagate,
    as the fail-closed contract requires.
    """

    reason: str
    message: str = ""


def resolve_or_none(
    resolver: Optional[FrameResolver],
    frame: PerceptionFrame,
    *,
    arm: RobotArm,
) -> Union[Transform, FrameResolutionFailure]:
    """Best-effort, non-raising wrapper around :meth:`FrameResolver.camera_to_base_for_frame`.

    Returns either:

    * a :class:`Transform` from ``CAMERA`` to ``BASE`` on success, or
    * a :class:`FrameResolutionFailure` describing why no transform
      could be produced.

    This helper never raises, which is why it is for shadow-path and
    observer callers only, such as fusion ingest or a telemetry overlay.
    The execution policy must call the resolver directly, so that a
    hardware or calibration failure surfaces on the fail-closed
    exception path.
    """

    if resolver is None:
        return FrameResolutionFailure(
            reason=RESOLVE_REASON_NO_RESOLVER,
            message="no frame resolver wired",
        )
    try:
        t = resolver.camera_to_base_for_frame(frame, arm=arm)
    except Exception as exc:  # noqa: BLE001 (intentional broad guard for shadow paths)
        return FrameResolutionFailure(
            reason=RESOLVE_REASON_EXCEPTION,
            message=f"{type(exc).__name__}: {exc}",
        )
    if t is None:
        return FrameResolutionFailure(
            reason=RESOLVE_REASON_NONE_RETURNED,
            message="resolver returned None",
        )
    if not isinstance(t, Transform):
        return FrameResolutionFailure(
            reason=RESOLVE_REASON_BAD_TRANSFORM,
            message=f"resolver returned {type(t).__name__}, expected Transform",
        )
    if t.from_frame is not Frame.CAMERA or t.to_frame is not Frame.BASE:
        return FrameResolutionFailure(
            reason=RESOLVE_REASON_WRONG_FRAME,
            message=(
                f"resolver returned Transform({t.from_frame.name} -> "
                f"{t.to_frame.name}); expected CAMERA -> BASE"
            ),
        )
    return t
