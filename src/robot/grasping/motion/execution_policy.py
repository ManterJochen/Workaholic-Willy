"""GraspExecutionPolicy, the vendor-neutral strategy for approach, grasp and retreat.

The motion and gripper choreography lives here rather than in
:class:`BinPickingOrchestrator`. The orchestrator decides which grasp to
attempt, and the policy owns how the arm and the optional gripper realise
that grasp.

Capability-aware close verification
-----------------------------------
The policy queries :meth:`ObjectDetectingGripper.is_object_detected` if
and only if the configured gripper advertises the
:class:`ObjectDetectingGripper` capability. A gripper without that
capability is trusted after the close command and the policy returns
:attr:`PolicyOutcome.EXECUTED`.

Numerics
--------
* TCP positions are :class:`Pose`: millimetres, an XYZW quaternion and a :class:`Frame`.
* Gripper widths and motion offsets are millimetres, as floats.
* Forces are newtons, as optional floats.

This module imports no vendor driver. It talks only to the
:class:`RobotArm` and :class:`Gripper` Protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.quaternion import from_rotation_matrix
from src.robot.core import (
    Gripper,
    MotionCommand,
    MotionResult,
    MotionStatus,
    ObjectDetectingGripper,
    RobotArm,
)
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint

__all__ = [
    "GraspExecutionPolicy",
    "PolicyOutcome",
    "PolicyReport",
]


class PolicyOutcome(str, Enum):
    """Terminal status of :meth:`GraspExecutionPolicy.execute`."""

    EXECUTED = "executed"
    """Approach, close and retreat completed, and the object is held or trusted to be."""

    OBJECT_NOT_DETECTED = "object_not_detected"
    """Close succeeded mechanically but the gripper reports an empty jaw."""

    MOTION_FAILED = "motion_failed"
    """:meth:`RobotArm.move_to` raised. The exception is in the report."""

    CAMERA_FRAME_REJECTED = "camera_frame_rejected"
    """Fail-closed guard: the policy refused to execute a camera-frame
    :class:`GraspPoint` because ``require_base_frame_grasp`` is enabled
    and no :class:`FrameResolver` was wired into the upstream
    orchestrator. No motion was commanded.
    """

    APPROACH_PATH_BLOCKED = "approach_path_blocked"
    """The approach and retreat sweep of every ranked candidate collided with a neighbour in the
    scene obstacle cloud, as judged by the swept-volume validator, so no candidate was executed.
    The dense-mode approach-validation fallback in the orchestrator surfaces this. No motion was
    commanded.
    """


@dataclass(frozen=True, slots=True)
class PolicyReport:
    """Outcome of a :meth:`GraspExecutionPolicy.execute` call.

    Attributes
    ----------
    outcome
        Terminal :class:`PolicyOutcome`.
    waypoints
        Ordered tuple of :class:`Pose` instances the policy commanded.
    object_detected
        :data:`None` when the gripper has no detection capability or no
        gripper was configured; otherwise the reported boolean.
    error
        Optional :class:`Exception` instance for :attr:`PolicyOutcome.MOTION_FAILED`.
    motion_status
        Categorical :class:`MotionStatus` from the typed
        :meth:`RobotArm.move` surface. ``None`` when the driver exposes
        only the older bool ``move_to`` path, which carries no typed
        information. Populated whenever the driver implements the
        typed move contract, including the success case
        :attr:`MotionStatus.EXECUTED`, so a consumer can confirm a clean
        execution without reading ``outcome`` alone.
    motion_message
        Optional human-readable detail forwarded from the underlying
        :class:`MotionResult.message`. Empty string when not provided.
    """

    outcome: PolicyOutcome
    waypoints: tuple[Pose, ...] = ()
    object_detected: bool | None = None
    error: Exception | None = None
    motion_status: MotionStatus | None = None
    motion_message: str = ""


@dataclass
class GraspExecutionPolicy:
    """Approach / grasp / retreat strategy.

    Parameters
    ----------
    arm
        Any :class:`RobotArm` driver.
    gripper
        Optional :class:`Gripper`. When :data:`None`, the policy only
        drives the arm (waypoints are still emitted; no close command).
    standoff_mm
        Distance from the grasp point along the reversed approach axis
        for the pre-grasp waypoint. The policy interpolates linearly
        from pre-grasp to the grasp point in :attr:`approach_steps`
        waypoints, inclusive of both endpoints.
    retreat_mm
        Vertical lift (along world +Z) commanded after closing.
    approach_steps
        Number of inclusive waypoints between pre-grasp and the grasp
        point. Must be ``>= 2``.
    pre_open_width_mm
        Optional jaw width commanded before the approach when a gripper
        is present. ``None`` skips the pre-open command.
    close_width_mm
        Jaw width commanded at the grasp point. When :data:`None`, falls
        back to ``max(gripper.min_width_mm, grasp.grip_width_mm - 1.0)``
        at execution time.
    close_speed
        Optional gripper speed in ``[0.0, 1.0]``.
    close_force_n
        Optional gripper force in newtons.
    """

    arm: RobotArm
    gripper: Gripper | None = None
    standoff_mm: float = 80.0
    retreat_mm: float = 100.0
    approach_steps: int = 4
    # Split the post-close vertical lift into ``retreat_steps`` interpolated waypoints, each driven and
    # settled, since move_joint settles after every move. A single 100 mm lift swings a loosely gripped
    # wide object on a compliant grip: the object pendulums up to about 150 mm during the retreat and
    # trips the arm convergence into execution_failed. Chunked lifts let the pendulum damp between
    # steps. The default of 1 is the single retreat waypoint.
    retreat_steps: int = 1
    pre_open_width_mm: float | None = None
    close_width_mm: float | None = None
    # Squeeze margin in millimetres, used only when ``close_width_mm is None``: the jaw closes to
    # ``max(min_width, grasp.grip_width_mm - close_squeeze_mm)``, so the close adapts to each object's
    # measured grip width without per-object tuning, a firm squeeze below the surface being what holds
    # it. The default 1.0 is a hug of one millimetre. A larger margin, 11 for instance, is what the
    # dense YCB pick uses so that a thin box is clamped rather than merely touched.
    close_squeeze_mm: float = 1.0
    close_speed: float | None = None
    close_force_n: float | None = None
    # Opt-in fail-closed frame guard. When ``True`` the policy refuses
    # to execute a camera-frame :class:`GraspPoint` and returns
    # :attr:`PolicyOutcome.CAMERA_FRAME_REJECTED` without commanding any
    # motion. The default ``False`` leaves every caller as it is. Only
    # :class:`AutonomousGraspService` turns it on, and only when an
    # operator wires a :class:`FrameResolver`.
    require_base_frame_grasp: bool = False
    # Pre-move steady-state temporal gate, from DwellSafetyConfig. When True the policy blocks on
    # ``arm.wait_until_steady(steady_timeout_s)`` before every commanded move and fails closed on
    # timeout, with no motion. The default False leaves a caller that never threads the dwell config,
    # or a double that does not implement ``wait_until_steady``, unchanged. The autonomous_grasp
    # composition root turns it on from ``cfg.robot.safety.dwell``, mirroring
    # ``require_base_frame_grasp``. ``dwell_after_stop_s`` is not consulted here, as no Stop path is
    # exercised.
    require_steady_before_motion: bool = False
    steady_timeout_s: float = 5.0
    # Opt-in grasp-orientation alignment for symmetric top-down grasps. When True the policy rigidly
    # yaws the orientation of each grasp about base-Z so the gripper closing axis falls on the trackable
    # base-X, keeping the position and the near-vertical approach. A top-down close along base-Y is not
    # trackable by the UR5e overhead move, where the standoff move times out, measured and not a
    # singularity, but a square cube is graspable from any horizontal yaw, so aligning to base-X
    # recovers it. The default False leaves the waypoints unchanged. It is valid for symmetric objects
    # only: YCB, with its fixed graspable axis, needs a shape-aware yaw to the nearest reachable
    # orientation instead.
    align_closing_to_base_x: bool = False
    # When True the approach is driven as a single goal, the grasp pose itself, so the global planner
    # (cuRobo) plans the whole collision-free descent instead of tracking the interpolated waypoints
    # from standoff to grasp. The waypoint decomposition is a blind-IK construct: cuRobo plans each tiny
    # segment in isolation and the final tight one near the obstacles returns no plan, whereas one goal
    # from park to grasp solves, measured as mp_l2_reach park to grasp EXECUTED against NONE for the
    # per-segment descent. The default False drives every approach waypoint. The gripper is still
    # pre-opened before the move and the retreat lifts are unchanged.
    planner_owns_approach: bool = False

    def __post_init__(self) -> None:
        if self.approach_steps < 2:
            raise ValueError(
                f"approach_steps must be >= 2, got {self.approach_steps}"
            )
        if self.standoff_mm < 0.0:
            raise ValueError(f"standoff_mm must be >= 0, got {self.standoff_mm}")
        if self.retreat_mm < 0.0:
            raise ValueError(f"retreat_mm must be >= 0, got {self.retreat_mm}")
        if self.retreat_steps < 1:
            raise ValueError(f"retreat_steps must be >= 1, got {self.retreat_steps}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, grasp: GraspPoint) -> PolicyReport:
        """Drive the arm and the gripper through the full grasp sequence."""
        # Fail-closed guard: when enabled, a camera-frame grasp is never
        # executed, because doing so would silently move the arm to a
        # camera-frame pose. The guard runs before the waypoints are
        # built, so no preparation work happens either.
        if self.require_base_frame_grasp and grasp.frame is not GraspFrame.BASE:
            return PolicyReport(
                outcome=PolicyOutcome.CAMERA_FRAME_REJECTED,
                waypoints=(),
                motion_message=(
                    "refused to execute a camera-frame grasp with no "
                    "FrameResolver wired (require_base_frame_grasp=True)"
                ),
            )
        # Forget any part from a previous pick. Detaching here rather than on release means a
        # stale attachment cannot survive a failed pick, a recovery, or a runner that never released:
        # the planner is only ever told about a part between a confirmed close and the next attempt.
        detach = getattr(self.arm, "detach_payload", None)
        if callable(detach):
            detach()
        waypoints = self._build_waypoints(grasp)

        # Pre-open the gripper before driving the approach so the jaws
        # are clear at the grasp point.
        if self.gripper is not None and self.pre_open_width_mm is not None:
            self.gripper.set_width_mm(
                float(self.pre_open_width_mm),
                speed=self.close_speed,
                force=None,
            )

        commanded: list[Pose] = []
        last_status: MotionStatus | None = None
        last_message: str = ""
        approach = waypoints[:-self.retreat_steps]  # all but the retreat lift(s)
        if self.planner_owns_approach:
            # cuRobo plans the full collision-free descent to the grasp itself, so only the grasp is
            # driven, as the last approach waypoint, and the interpolated standoff and approach
            # segments the blind-IK path needs are skipped.
            approach = approach[-1:]
        for pose in approach:
            try:
                result = self._drive_to(pose)
            except Exception as exc:  # noqa: BLE001 (propagate via report)
                return PolicyReport(
                    outcome=PolicyOutcome.MOTION_FAILED,
                    waypoints=tuple(commanded),
                    error=exc,
                    motion_status=last_status,
                    motion_message=last_message,
                )
            if result is not None:
                last_status = result.status
                last_message = result.message
                if not result.ok:
                    return PolicyReport(
                        outcome=PolicyOutcome.MOTION_FAILED,
                        waypoints=tuple(commanded),
                        error=cast("Exception | None", result.exception),
                        motion_status=result.status,
                        motion_message=result.message,
                    )
            commanded.append(pose)

        # Close the gripper at the grasp point.
        object_detected: bool | None = None
        if self.gripper is not None:
            target_width = self._resolve_close_width(grasp)
            self.gripper.set_width_mm(
                target_width,
                speed=self.close_speed,
                force=self.close_force_n,
            )
            if isinstance(self.gripper, ObjectDetectingGripper):
                object_detected = bool(self.gripper.is_object_detected())
                if not object_detected:
                    return PolicyReport(
                        outcome=PolicyOutcome.OBJECT_NOT_DETECTED,
                        waypoints=tuple(commanded),
                        object_detected=False,
                        motion_status=last_status,
                        motion_message=last_message,
                    )

        # The planner learns what it is carrying, before the first motion that carries it. A lift,
        # transit or place planned as if the hand were empty ignores the part, and on a cell lifting a
        # part out of a bin that part is the geometry most likely to meet a wall.
        attach = getattr(self.arm, "attach_payload", None)
        if callable(attach):
            # ⛔ THE COMMANDED WIDTH IS NOT THE WIDTH OF THE PART. On the adaptive branch the command
            # is `grip_width - close_squeeze_mm`, a number the jaws deliberately never reach, so the
            # planner was carrying the size of something that was never measured. Reading the jaws is
            # a real measurement and, since a close now waits for the fingers, a settled one.
            attach(self._measured_or_commanded_width(grasp))

        # Retreat: command each interpolated lift waypoint, one per retreat_step; the default of 1 is
        # the single full lift.
        for pose in waypoints[-self.retreat_steps:]:
            try:
                result = self._drive_to(pose)
            except Exception as exc:  # noqa: BLE001 (propagate via report)
                return PolicyReport(
                    outcome=PolicyOutcome.MOTION_FAILED,
                    waypoints=tuple(commanded),
                    object_detected=object_detected,
                    error=exc,
                    motion_status=last_status,
                    motion_message=last_message,
                )
            if result is not None:
                last_status = result.status
                last_message = result.message
                if not result.ok:
                    return PolicyReport(
                        outcome=PolicyOutcome.MOTION_FAILED,
                        waypoints=tuple(commanded),
                        object_detected=object_detected,
                        error=cast("Exception | None", result.exception),
                        motion_status=result.status,
                        motion_message=result.message,
                    )
            commanded.append(pose)

        return PolicyReport(
            outcome=PolicyOutcome.EXECUTED,
            waypoints=tuple(commanded),
            object_detected=object_detected,
            motion_status=last_status,
            motion_message=last_message,
        )

    # ------------------------------------------------------------------
    # Motion adapter
    # ------------------------------------------------------------------

    def _drive_to(self, pose: Pose) -> MotionResult | None:
        """Drive ``arm`` to ``pose`` via the best available surface.

        Prefers the typed :meth:`RobotArm.move` contract, which every
        production driver implements. Falls back to the older
        :meth:`RobotArm.move_to` surface, which returns a bool or raises,
        so a double that does not implement the typed contract keeps
        working; :class:`PolicyReport.motion_status` records which path
        was taken and therefore what diagnostics are available.

        Returns the :class:`MotionResult` from the typed path, or
        ``None`` when the ``move_to`` path is taken, which carries no
        typed information.
        """
        # Pre-move steady gate, a fail-closed temporal check outside the per-target SafetyPreflight
        # pipeline. It is skipped when the gate is off or the driver has no steady signal, in which
        # case the getattr yields None and the move passes through. A timeout returns a typed timeout
        # result before any motion, which the execute() loop maps to PolicyOutcome.MOTION_FAILED.
        if self.require_steady_before_motion:
            wait_fn = getattr(self.arm, "wait_until_steady", None)
            if callable(wait_fn) and not wait_fn(float(self.steady_timeout_s)):
                return MotionResult.failed(
                    MotionStatus.TIMEOUT,
                    MotionCommand.OTHER,
                    target_pose=pose,
                    message=(
                        "pre-move steady gate timed out after "
                        f"{self.steady_timeout_s:.2f}s (require_steady_before_motion)"
                    ),
                )
        typed_move = getattr(self.arm, "move", None)
        if callable(typed_move):
            return typed_move(pose)
        self.arm.move_to(pose)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _measured_or_commanded_width(self, grasp: GraspPoint) -> float:
        """What the jaws actually hold, falling back to what they were told.

        The fallback is not a nicety: a gripper that cannot report a width, or one whose read fails
        mid-cell, must still attach SOMETHING or the planner goes back to lifting the part as if the
        hand were empty. A commanded width is a worse number than a measured one and a much better
        number than none.
        """
        read = getattr(self.gripper, "get_width_mm", None)
        if callable(read):
            try:
                measured = float(read())
            except Exception:  # noqa: BLE001 - a failed read is not a failed pick
                measured = float("nan")
            if measured == measured and measured > 0.0:  # not NaN
                return measured
        return self._resolve_close_width(grasp)

    def _resolve_close_width(self, grasp: GraspPoint) -> float:
        """Pick the jaw width to command at the grasp point."""
        if self.close_width_mm is not None:
            return float(self.close_width_mm)
        # Adaptive: squeeze ``close_squeeze_mm`` below the predicted grip width, 1 mm by default, but
        # never below the minimum opening of the gripper. A larger margin clamps thin objects firmly.
        gripper = self.gripper
        min_w = getattr(gripper, "min_width_mm", 0.0) or 0.0
        return float(max(min_w, float(grasp.grip_width_mm) - float(self.close_squeeze_mm)))

    def _build_waypoints(self, grasp: GraspPoint) -> tuple[Pose, ...]:
        """Build the [approach_0..approach_N-1, retreat] waypoint sequence."""
        closing = np.asarray(grasp.axis, dtype=np.float64)
        approach_unit = np.asarray(grasp.approach, dtype=np.float64)
        target = np.asarray(grasp.position, dtype=np.float64)
        if self.align_closing_to_base_x:  # yaw the close onto the reachable base-X sign for this side
            prefer_plus_x = float(target[1]) < 0.0  # UR5e: -Y reaches with +X close, +Y with -X
            closing, approach_unit = _yaw_axes_to_base_x(closing, approach_unit, prefer_plus_x=prefer_plus_x)
        quat = _quaternion_from_axes(closing, approach_unit)
        pre = target - approach_unit * float(self.standoff_mm)
        frame = Frame(grasp.frame.value)
        poses: list[Pose] = []
        for index, fraction in enumerate(np.linspace(0.0, 1.0, num=self.approach_steps)):
            position = pre + (target - pre) * float(fraction)
            poses.append(
                Pose(
                    position_mm=position,
                    quaternion_xyzw=quat.copy(),
                    frame=frame,
                    label=f"approach_{index:02d}",
                )
            )
        # Retreat: ``retreat_steps`` interpolated vertical lifts from the grasp point up to grasp plus
        # retreat_mm, for k of 1 to N. With retreat_steps=1 that is a single waypoint at the full lift,
        # labelled "retreat".
        for k in range(1, self.retreat_steps + 1):
            fraction = float(k) / float(self.retreat_steps)
            retreat_position = target + np.array([0.0, 0.0, float(self.retreat_mm) * fraction])
            label = "retreat" if self.retreat_steps == 1 else f"retreat_{k:02d}"
            poses.append(
                Pose(
                    position_mm=retreat_position,
                    quaternion_xyzw=quat.copy(),
                    frame=frame,
                    label=label,
                )
            )
        return tuple(poses)


def _quaternion_from_axes(closing: np.ndarray, approach: np.ndarray) -> np.ndarray:
    """Unit XYZW quaternion from the closing and approach axes of a grasp.

    Rotation columns are ``[closing, binormal, approach]``, with the binormal recovered from
    ``approach x closing`` so the matrix is right-handed and orthonormal.
    """
    closing = np.asarray(closing, dtype=np.float64)
    approach = np.asarray(approach, dtype=np.float64)
    binormal = np.cross(approach, closing)
    binormal_norm = float(np.linalg.norm(binormal))
    if binormal_norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    binormal /= binormal_norm
    R = np.column_stack([closing, binormal, approach])
    return np.asarray(from_rotation_matrix(R), dtype=np.float64)


def _grasp_point_to_quaternion(grasp: GraspPoint) -> np.ndarray:
    """Rebuild a unit XYZW quaternion from the closing and approach axes of a ``GraspPoint``."""
    return _quaternion_from_axes(
        np.asarray(grasp.axis, dtype=np.float64), np.asarray(grasp.approach, dtype=np.float64)
    )


def _yaw_axes_to_base_x(
    closing: np.ndarray, approach: np.ndarray, *, prefer_plus_x: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly yaw ``(closing, approach)`` about base-Z so the closing axis points along base +X, with
    ``prefer_plus_x``, or along base -X, keeping the approach near vertical.

    The target sign is given rather than derived from whichever of +/-X is geometrically nearer. The
    UR5e overhead move converges for a base-X top-down close and times out for a base-Y one, and for a
    natural closing near base-Y, right on the decision boundary, the nearer sign is unstable and lands
    on the unreachable wrist flip. The reachable sign depends on position, measured as the -Y side
    reaching with a +X close and the +Y side with a -X close, so the caller passes the side as
    ``prefer_plus_x``. A degenerate, vertical closing axis is returned unchanged. See
    :attr:`GraspExecutionPolicy.align_closing_to_base_x`. Picking the sign by testing IK joint-limit
    reachability per candidate would be general; the side heuristic covers the overhead workspace of
    the sim and symmetric objects.
    """
    closing = np.asarray(closing, dtype=np.float64)
    approach = np.asarray(approach, dtype=np.float64)
    ch = np.array([closing[0], closing[1], 0.0], dtype=np.float64)
    n = float(np.linalg.norm(ch))
    if n < 1e-6:  # a near-vertical closing axis has no meaningful horizontal yaw
        return closing, approach
    ang = float(np.arctan2(ch[1], ch[0]))
    target = 0.0 if prefer_plus_x else np.pi  # base +X or base -X (the reachable side)
    theta = target - ang
    c, s = float(np.cos(theta)), float(np.sin(theta))
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ closing, rz @ approach
