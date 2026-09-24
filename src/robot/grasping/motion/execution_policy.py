"""GraspExecutionPolicy, the vendor-neutral strategy for approach, grasp and retreat.

The motion and gripper choreography lives here rather than in
:class:`BinPickingOrchestrator`. The orchestrator decides which grasp to
attempt, and the policy owns how the arm and the optional gripper realise
that grasp.

Capability-aware close verification
-----------------------------------
The policy queries :meth:`ObjectDetectingGripper.is_object_detected` if
and only if the configured gripper advertises the
:class:`ObjectDetectingGripper` capability, and a ``False`` ends the pick
as :attr:`PolicyOutcome.OBJECT_NOT_DETECTED`. A ``True`` is only a
measurement where the gripper's :class:`ReportsHoldEvidence` does not say
UNMEASURED: a jaw with no feedback wired answers ``is_object_detected``
with its own command, so :attr:`PolicyReport.object_detected` is then
``None``, not ``True``. A gripper without either capability is trusted
after the close command and the policy returns :attr:`PolicyOutcome.EXECUTED`
with ``object_detected`` ``None``.

The controller is asked before anything is commanded
----------------------------------------------------
On an arm that reports its controller (:class:`SupportsRobotStatus`, the UR
driver), a policy with a gripper reads the controller before the detach and
the pre-open and again before the close. A controller that cannot move ends
the pick as :attr:`PolicyOutcome.MOTION_FAILED` with
:attr:`MotionStatus.CONTROLLER_REJECTED` and nothing commanded, the jaws
included.

A hand that toggles with no sensor
----------------------------------
A :class:`TogglesWithoutSensor` hand (a ``jaw_io`` single_toggle) is never
pulsed before the arm moves: the pre-open is skipped whatever
``pre_open_width_mm`` says, and the hand is asked instead whether its jaws
stand open (``jaws_open_for_a_pick``), which asks a person where it believes
them closed. A pick nobody can vouch for ends as
:attr:`PolicyOutcome.GRIPPER_FAULT` before any motion. At the part it gets
exactly one close.

A gripper that raises
---------------------
A driver that raises while the jaws are commanded, a :class:`RobotError`
above all, ends the pick as :attr:`PolicyOutcome.GRIPPER_FAULT` with the
exception on the report, never as an exception out of :meth:`execute`.

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
    CameraWorldStamp,
    Gripper,
    MotionCommand,
    MotionResult,
    MotionStatus,
    ObjectDetectingGripper,
    RobotArm,
    SupportsRobotStatus,
)
from src.robot.core.arm_capabilities import LineMotion, line_motion_of
from src.robot.core.camera_world import weakest_camera_world
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.core.gripper import (
    HoldEvidence,
    OpensAndCloses,
    ReportsHoldEvidence,
    hold_evidence_of,
    toggle_without_sensor_of,
    width_is_measured_of,
)
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint

__all__ = [
    "GraspExecutionPolicy",
    "PolicyOutcome",
    "PolicyReport",
    "weakest_camera_world",
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

    GRIPPER_FAULT = "gripper_fault"
    """The gripper raised while it was commanded (``error`` carries it), or a hand that toggles
    with no sensor would not start the pick: it believed its jaws closed and nobody at a terminal
    said otherwise, or a person aborted. ``motion_message`` says which. Nothing was commanded
    after it.
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
        :data:`None` when no gripper was configured, when the gripper has
        no detection capability, or when it says it measured nothing about
        the hold (``hold_evidence()`` UNMEASURED: a jaw with no feedback
        wired, whose ``is_object_detected`` is its own command). Otherwise
        what the gripper measured: ``True`` held, ``False`` empty.
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
    camera_worlds
        One :class:`CameraWorldStamp` per typed motion, in command order,
        the stamp of a motion that failed included. Empty when the driver
        only exposes the legacy bool ``move_to`` path or no motion was
        commanded. :attr:`camera_world` reads the weakest one.
    """

    outcome: PolicyOutcome
    waypoints: tuple[Pose, ...] = ()
    object_detected: bool | None = None
    error: Exception | None = None
    motion_status: MotionStatus | None = None
    motion_message: str = ""
    camera_worlds: tuple[CameraWorldStamp, ...] = ()
    #: What the arm said it keeps of a straight line, read before anything was commanded. ``None``
    #: for an arm that does not say, which keeps the interpolated approach. Not carried into the
    #: grasp record.
    line_motion: LineMotion | None = None

    @property
    def camera_world(self) -> CameraWorldStamp | None:
        """The weakest stamp of :attr:`camera_worlds` (:func:`weakest_camera_world`)."""
        return weakest_camera_world(self.camera_worlds)


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
        is present. ``None`` skips the pre-open command, and so does a
        hand that toggles with no sensor, whatever this says.
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
        # What the arm keeps of a straight line, read once before anything is commanded. An arm that
        # says it keeps lines gets a planned move to the standoff, one line to the grasp and line
        # lifts: the final descent is the motion a finger meets a part or a wall on, and it must not
        # be planned around the part it reaches into. An arm that keeps none is refused here, before
        # the pre-open, as the robot's hand verbs refuse it. An arm that does not say keeps the
        # interpolated waypoints.
        reading = line_motion_of(self.arm)
        line_motion = None if reading is None else reading.motion
        if reading is not None and reading.motion is LineMotion.NOT_KEPT:
            return PolicyReport(
                outcome=PolicyOutcome.MOTION_FAILED,
                waypoints=(),
                motion_status=MotionStatus.UNSUPPORTED,
                motion_message=f"this arm keeps no straight line for the final descent: {reading.reason}",
                line_motion=line_motion,
            )
        # Where the jaws will be told anything, a controller that cannot move is asked about before anything is
        # commanded, the planner's part included. After a protective stop mid-pick, the next attempt of a
        # campaign used to detach and pulse a toggle hand open before its first motion asked the controller
        # anything, so the part dropped from wherever the arm had stopped (owner-cell audit, reproduced with
        # fakes, 2026-09-23). A policy with no gripper commands no I/O, and its first motion meets the
        # controller as before.
        refused = _controller_refusal(self.arm) if self.gripper is not None else ""
        if refused:
            return PolicyReport(
                outcome=PolicyOutcome.MOTION_FAILED,
                waypoints=(),
                motion_status=MotionStatus.CONTROLLER_REJECTED,
                motion_message=refused,
                line_motion=line_motion,
            )
        toggle = toggle_without_sensor_of(self.gripper)
        if toggle is not None:
            # Never a pulse before the arm moves (owner's decision, 2026-09-24): a toggle is asked, and asks a person
            # where it believes its jaws closed, instead of being pulsed open. A pick nobody can vouch for ends here.
            try:
                why = toggle.jaws_open_for_a_pick()
            except Exception as exc:  # noqa: BLE001 (a gripper fault is the report, never an escaping raise)
                return _gripper_fault(exc, line_motion=line_motion)
            if why:
                return PolicyReport(outcome=PolicyOutcome.GRIPPER_FAULT, motion_message=why, line_motion=line_motion)
        # Forget any part from a previous pick. Detaching here rather than on release means a
        # stale attachment cannot survive a failed pick, a recovery, or a runner that never released:
        # the planner is only ever told about a part between a confirmed close and the next attempt.
        detach = getattr(self.arm, "detach_payload", None)
        if callable(detach):
            detach()
        waypoints = self._build_waypoints(grasp)

        # Pre-open the gripper before driving the approach so the jaws
        # are clear at the grasp point.
        if self.gripper is not None and self.pre_open_width_mm is not None and toggle is None:
            try:
                if isinstance(self.gripper, OpensAndCloses):
                    # A two-state gripper told what it means: a pre-open is an open whatever its width reads as.
                    self.gripper.set_closed(False)
                else:
                    self.gripper.set_width_mm(
                        float(self.pre_open_width_mm),
                        speed=self.close_speed,
                        force=None,
                    )
            except Exception as exc:  # noqa: BLE001 (a gripper fault is the report, never an escaping raise)
                return _gripper_fault(exc, line_motion=line_motion)

        commanded: list[Pose] = []
        last_status: MotionStatus | None = None
        last_message: str = ""
        # One camera-world stamp per typed motion, collected wherever a status is, so a report of
        # a failed pick still says what stood behind the motion that failed.
        stamps: list[CameraWorldStamp] = []
        approach = waypoints[:-self.retreat_steps]  # all but the retreat lift(s)
        legs: list[tuple[Pose, bool]] = [(pose, False) for pose in approach]
        if reading is not None:
            # The standoff, planned, then one line to the grasp.
            legs = [(approach[0], False), (approach[-1], True)]
        for pose, linear in legs:
            try:
                result = self._drive_to(pose, linear=linear)
            except CameraWorldUnavailable:
                # A camera that could not vouch for the cell is a fault of the cell, so it leaves the
                # pick and PickRun stops the campaign on it rather than retrying a grasp.
                raise
            except Exception as exc:  # noqa: BLE001 (propagate via report)
                return PolicyReport(
                    outcome=PolicyOutcome.MOTION_FAILED,
                    waypoints=tuple(commanded),
                    error=exc,
                    motion_status=last_status,
                    motion_message=last_message,
                    camera_worlds=tuple(stamps),
                    line_motion=line_motion,
                )
            if result is not None:
                last_status = result.status
                last_message = result.message
                stamps.append(result.camera_world)
                if not result.ok:
                    return PolicyReport(
                        outcome=PolicyOutcome.MOTION_FAILED,
                        waypoints=tuple(commanded),
                        error=cast("Exception | None", result.exception),
                        motion_status=result.status,
                        motion_message=result.message,
                        camera_worlds=tuple(stamps),
                        line_motion=line_motion,
                    )
            commanded.append(pose)

        # Close the gripper at the grasp point.
        object_detected: bool | None = None
        if self.gripper is not None:
            # Asked again at the part: a stop that fell between the line's end and the close leaves an arm that
            # cannot lift what the jaws would take.
            refused = _controller_refusal(self.arm)
            if refused:
                return PolicyReport(
                    outcome=PolicyOutcome.MOTION_FAILED,
                    waypoints=tuple(commanded),
                    motion_status=MotionStatus.CONTROLLER_REJECTED,
                    motion_message=refused,
                    camera_worlds=tuple(stamps),
                    line_motion=line_motion,
                )
            target_width = self._resolve_close_width(grasp)
            try:
                if isinstance(self.gripper, OpensAndCloses):
                    # The grasp width read against closed_below_mm opens a two-state gripper when the part is wider
                    # than the threshold, and the loop then reported object_not_detected on jaws it never closed
                    # (URSim, the owner's toggle cell, 2026-09-23). A grasp is a close.
                    self.gripper.set_closed(True)
                else:
                    self.gripper.set_width_mm(
                        target_width,
                        speed=self.close_speed,
                        force=self.close_force_n,
                    )
                # What the gripper measured, not what it echoes: a jaw with no feedback wired answers
                # is_object_detected with its own close, and the pick service recorded that as a detected part
                # on every close (owner-cell audit, 2026-09-23). Unmeasured is None, and the pick goes on as the
                # trusted close it always was.
                object_detected = _hold_after_close(self.gripper)
            except Exception as exc:  # noqa: BLE001 (a gripper fault is the report, never an escaping raise)
                return _gripper_fault(exc, line_motion=line_motion, waypoints=tuple(commanded),
                                      motion_status=last_status, camera_worlds=tuple(stamps))
            if object_detected is False:
                return PolicyReport(
                    outcome=PolicyOutcome.OBJECT_NOT_DETECTED,
                    waypoints=tuple(commanded),
                    object_detected=False,
                    motion_status=last_status,
                    motion_message=last_message,
                    camera_worlds=tuple(stamps),
                    line_motion=line_motion,
                )

        # The planner learns what it is carrying, before the first motion that carries it. A lift,
        # transit or place planned as if the hand were empty ignores the part, and on a cell lifting a
        # part out of a bin that part is the geometry most likely to meet a wall.
        attach = getattr(self.arm, "attach_payload", None)
        if callable(attach):
            # The commanded width is not the width of the part. On the adaptive branch the command
            # is `grip_width` less `close_squeeze_mm`, a number the jaws never reach by design, so it
            # is the size of something nobody measured. Reading the jaws is a measurement and, since
            # a close waits for the fingers, a settled one.
            attach(self._measured_or_commanded_width(grasp))

        # Retreat: command each interpolated lift waypoint, one per retreat_step; the default of 1 is
        # the single full lift.
        for pose in waypoints[-self.retreat_steps:]:
            try:
                result = self._drive_to(pose, linear=reading is not None)
            except CameraWorldUnavailable:
                raise  # out of the pick, for the reason the approach gives
            except Exception as exc:  # noqa: BLE001 (propagate via report)
                return PolicyReport(
                    outcome=PolicyOutcome.MOTION_FAILED,
                    waypoints=tuple(commanded),
                    object_detected=object_detected,
                    error=exc,
                    motion_status=last_status,
                    motion_message=last_message,
                    camera_worlds=tuple(stamps),
                    line_motion=line_motion,
                )
            if result is not None:
                last_status = result.status
                last_message = result.message
                stamps.append(result.camera_world)
                if not result.ok:
                    return PolicyReport(
                        outcome=PolicyOutcome.MOTION_FAILED,
                        waypoints=tuple(commanded),
                        object_detected=object_detected,
                        error=cast("Exception | None", result.exception),
                        motion_status=result.status,
                        motion_message=result.message,
                        camera_worlds=tuple(stamps),
                        line_motion=line_motion,
                    )
            commanded.append(pose)

        return PolicyReport(
            outcome=PolicyOutcome.EXECUTED,
            waypoints=tuple(commanded),
            object_detected=object_detected,
            motion_status=last_status,
            motion_message=last_message,
            camera_worlds=tuple(stamps),
            line_motion=line_motion,
        )

    # ------------------------------------------------------------------
    # Motion adapter
    # ------------------------------------------------------------------

    def _drive_to(self, pose: Pose, *, linear: bool = False) -> MotionResult | None:
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
            # Only a line passes the keyword, so a planned move is the plain call any arm accepts.
            return typed_move(pose, linear=True) if linear else typed_move(pose)
        self.arm.move_to(pose)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _measured_or_commanded_width(self, grasp: GraspPoint) -> float:
        """What the jaws actually hold, falling back to what they were told.

        The fallback is not a nicety: a gripper that cannot report a width, or one whose read fails
        mid-cell, must still attach something or the planner goes back to lifting the part as if the
        hand were empty. A commanded width is a worse number than a measured one and a much better
        number than none.

        The jaws are read only where the gripper says its width is measured (``MeasuresWidth``), as
        the robot's grasp verb does. A jaw driven over digital I/O reports the band it was commanded
        to, its closed band 5 mm on the owner's Hand-E, and the carried part was attached 5 mm wide
        for a 40 mm grasp, so the planner lifted it out of a bin about 34 mm too narrow across the
        jaws (owner-cell audit, reproduced 2026-09-23).
        """
        read = getattr(self.gripper, "get_width_mm", None)
        if callable(read) and width_is_measured_of(self.gripper):
            try:
                measured = float(read())
            except Exception:  # noqa: BLE001 (a failed read is not a failed pick)
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


def _controller_refusal(arm: object) -> str:
    """Why ``arm``'s controller cannot act now, or ``""`` where it can, or where the arm does not say.

    Only an arm that implements ``SupportsRobotStatus`` (the UR driver) is asked; every other arm passes, as it
    passes the pick loop's diagnosis. The criterion is ``not is_operational``, the one the pick loop names a failed
    motion ``CONTROLLER_NOT_OPERATIONAL`` with, so a refusal here and that diagnosis describe one state: a controller
    in REDUCED safety mode is refused too. A read that raises is not caught: a controller that cannot be asked is a
    fault of the cell, and the pick leaves on it with nothing commanded.
    """
    if not isinstance(arm, SupportsRobotStatus):
        return ""
    status = arm.get_robot_status()
    if status.is_operational:
        return ""
    detail = f": {status.message}" if status.message else ""
    return (
        f"the controller cannot move (robot_mode={status.robot_mode.value}, safety_mode={status.safety_mode.value}, "
        f"protective_stop={status.protective_stopped}, emergency_stop={status.emergency_stopped}{detail}), so "
        "nothing was commanded, the jaws included; clear the stop where the arm is visible, then run again"
    )


def _gripper_fault(
    exc: Exception,
    *,
    line_motion: LineMotion | None,
    waypoints: tuple[Pose, ...] = (),
    motion_status: MotionStatus | None = None,
    camera_worlds: tuple[CameraWorldStamp, ...] = (),
) -> PolicyReport:
    """The report of a gripper that raised while it was commanded: the pick ends here, with nothing commanded after it.

    A driver's ``RobotError`` (a toggle that cannot say where its jaws stand, an I/O write the controller refused) used
    to leave :meth:`GraspExecutionPolicy.execute` as an exception, and the pick loop lost every motion before it.
    """
    return PolicyReport(
        outcome=PolicyOutcome.GRIPPER_FAULT,
        waypoints=waypoints,
        error=exc,
        motion_status=motion_status,
        motion_message=f"the gripper raised: {type(exc).__name__}: {exc}",
        camera_worlds=camera_worlds,
        line_motion=line_motion,
    )


def _hold_after_close(gripper: object) -> bool | None:
    """What ``gripper`` measured about a hold after a close: ``True`` held, ``False`` empty, ``None`` unmeasured.

    ``is_object_detected`` is asked first where the gripper has it, and a ``False`` stays final as it always was. Its
    ``True`` counts only where ``hold_evidence()`` does not say otherwise: EMPTY is ``False``, UNMEASURED ``None``,
    because a jaw with no feedback wired, or a vacuum with no switch, answers ``is_object_detected`` with its
    command. A gripper with neither capability measured nothing, ``None``.
    """
    detecting = isinstance(gripper, ObjectDetectingGripper)
    if isinstance(gripper, ObjectDetectingGripper) and not bool(gripper.is_object_detected()):
        return False
    if isinstance(gripper, ReportsHoldEvidence):
        hold = hold_evidence_of(gripper)
        if hold is HoldEvidence.EMPTY:
            return False
        return True if hold is HoldEvidence.HELD else None
    return True if detecting else None


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
