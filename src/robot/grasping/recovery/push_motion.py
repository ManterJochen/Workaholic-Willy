"""Drive one planned push of a failed part: the motion side of ``nudge_target``, run inside the pick attempt.

:func:`~src.robot.grasping.recovery.push_planner.plan_push` says where the hand goes; :func:`execute_push` drives it
there, under the owner's rules of 2026-09-29, and answers with a :class:`PushOutcome` that says what happened. It
imports no pick loop and no service: the pick attempt hands it the arm, the plan, the gripper and the part's
segmentation offer, and records what comes back.

**Before anything moves**, each of these is read, and none of them asks anybody. The first that fails ends the push
with a ``refused_*`` outcome and zero motion, so the recovery loop falls through to its next action
(:func:`~src.robot.grasping.recovery.policy.refused_before_motion`):

* the plan is a :class:`~src.robot.grasping.recovery.push_planner.PushPlan` (not the planner's refusal), and each
  contact leg runs at or below the owner's speed for it (:data:`LEG_SPEED_CAPS_M_S`);
* the plan is the shape of a push, whoever built it: it pushes 10 to 50 mm (:data:`SHORTEST_PUSH_MM`,
  :data:`LONGEST_PUSH_MM`), comes down at most 5 deg from straight down, and every station it drives lies within
  1 mm of where its own axes put it: P0 80 mm straight above the contact start, the push leg the 10 mm contact gap
  plus the push along its direction, 5 mm back, 80 mm straight up (:data:`APPROACH_RISE_MM`,
  :data:`CONTACT_GAP_MM`, :data:`BACK_OFF_MM`, :data:`STATION_TOLERANCE_MM`). Its tool axes make a pose;
* the arm has the typed ``move`` and says it judges every sample of a straight line before it drives it
  (:attr:`~src.robot.core.arm_capabilities.LineMotion.CHECKED`), and its live camera world is wired;
* the offer carries the part's BASE points, and the plan's part stands inside their footprint;
* the controller can move (an arm that reports its controller is asked; the same criterion the grasp uses);
* the jaws stand open by a read-only check of a connected hand. A hand that toggles with no sensor (the owner's
  Hand-E on tool DO0) is read off its count: ``jaws_closed`` is False and ``why_jaws_unknown()`` answers ``""``. A
  gripper that measures its width is read. A hand that is not connected vouches for nothing (its count ended at the
  disconnect and the next connect asks), nothing else counts, and ``jaws_open_for_a_pick``, which can ask a person,
  is never called. A toggle's count nobody can vouch for (``refused_jaws_unknown``) ends the pick at the caller, as a
  gripper fault, and so does a gripper that measures its width found not connected or whose width read fails; a count
  or a width that says closed is a refusal like the others;
* the arm is at rest, where the cell asks for that before every motion (``safety.dwell``), as for every grasp and
  look motion.

**The motion**, all of it inside :func:`~src.robot.core.keep_out.keeping_out` with the part's points plus its
swept travel (:func:`~src.robot.grasping.recovery.push_planner.swept_target_points_mm`, as far as the push leg
drives), so the part is no obstacle anywhere along its push. A world that does not take that offer is a refusal:

1. P0, in the air above the contact start, like a grasp approach: the arm's plain ``move``. On the owner's cuRobo
   UR that is the nearest configuration inside the cable window, the straight joint line first, a capped cuRobo
   detour only where the line is blocked, the camera world refreshed before the gate.
2. The four contact legs of :attr:`PushPlan.legs` (down, push, back 5 mm, up), each
   ``arm.move(pose, linear=True, vel=..., acc=...)``: a judged straight line at the plan's explicit speed. Before
   each, the controller is read again, a toggle's count is read again (the owner, 2026-09-30), and the arm waited on
   to rest where the cell asks for that.

Then the controller and a toggle's count are read once more and where the arm stands is read, and it returns. The move
back to the look is the caller's, as after a grasp approach, and so is looking again.

**After something may have moved**, nothing more moves. A P0 refused before anything was sent leaves the arm
where it stood, and that is still a refusal the loop falls through on. So is the down leg refused before anything
was sent (``refused_down_not_sent``, the lead's ruling of 2026-09-29, told to the owner, who may overrule it): the arm
stands in the air at P0, nothing touched the part, and the caller moves it back to the look like a grasp approach
before it goes on (the outcome's ``legs_done`` says the arm left the look). "Before anything was sent" is the owner's
one rule for it (``generated_view.REFUSED_BEFORE_SENDING``, and the planner's no-plan refusal): a line the arm refuses
with another status, ``unsupported`` among them, may have been commanded as far as that rule can tell, and stops the
push where the arm is. Any other failure of P0, any refusal or failure of a contact leg once
that leg was commanded, the down leg that may have been sent included, a controller that stops or cannot be read at
any point after P0 (the end of the up leg included), and a toggle's count nobody can vouch for before a contact leg
or once the up leg ended (DO0 switched at the pendant while the push drives) stop the push where the arm is, with no
further call to any motion verb and no planned escape: ``unsafe_recovery_refused``, or
``controller_not_operational`` where the controller says it cannot move (a protective stop). A stop is cleared by a
person: ``recover_from_protective_stop`` is never called, and the campaign ends. A camera world that cannot vouch
for the cell (``CameraWorldUnavailable``) is raised out, as every motion of a pick raises it: nothing more was
commanded, and the campaign stops on it.

The gripper is only read: DO0 is never written, so a push cannot lose the toggle's count. Once the keep-out scope
was opened the live world forgets every offer when it closes (the live world's rule), so after any push, a refusal
included, the caller offers the pick's target again before its next motion.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.closing_axis import ClosingAxis
from src.geometry.exceptions import GeometryError
from src.geometry.quaternion import from_rotation_matrix, to_rotation_matrix
from src.robot.core import NO_PLAN_FAIL_SAFE_MESSAGE, MotionResult, MotionStatus, SupportsRobotStatus
from src.robot.core.arm_capabilities import LineMotion, LineReading, halt_state_of, halted_refusal, line_motion_of
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.core.gripper import toggle_without_sensor_of, why_toggle_count_unknown, width_is_measured_of
from src.robot.core.keep_out import SegmentationOffer, keeping_out
from src.robot.execution.motion import steady_timeout_of
from src.robot.grasping.recovery.policy import refused_before_motion
from src.robot.grasping.recovery.push_planner import PushPlan, PushRefusal, Vec3, swept_target_points_mm

__all__ = [
    "APPROACH_LEG",
    "APPROACH_RISE_MM",
    "BACK_OFF_MM",
    "CONTACT_GAP_MM",
    "CONTACT_LEGS",
    "JAWS_OPEN_TOLERANCE_MM",
    "LEG_ACCEL_CAP_M_S2",
    "LEG_SPEED_CAPS_M_S",
    "LONGEST_PUSH_MM",
    "MAX_APPROACH_TILT_DEG",
    "OFFER_FOOTPRINT_MARGIN_MM",
    "SHORTEST_PUSH_MM",
    "STATION_TOLERANCE_MM",
    "PushOutcome",
    "PushOutcomeCode",
    "execute_push",
    "push_budget_key",
    "push_tool_quaternion",
]

_MM_PER_M = 1000.0

#: The motion to P0, in the air above the contact start.
APPROACH_LEG = "approach"
#: The contact legs, in the order :attr:`PushPlan.legs` gives them and this module drives them.
CONTACT_LEGS: tuple[str, ...] = ("down", "push", "back", "up")
#: The fastest each contact leg may run, in m/s: the owner's numbers (2026-09-29), down, back and up at 50 mm/s and
#: the push itself at 25 mm/s. Written here as literals, not read from the planner, so a planner constant that drifts
#: is refused rather than driven. The same holds for every number of a push's shape below.
LEG_SPEED_CAPS_M_S: Mapping[str, float] = MappingProxyType({
    "down": 50.0 / _MM_PER_M,
    "push": 25.0 / _MM_PER_M,
    "back": 50.0 / _MM_PER_M,
    "up": 50.0 / _MM_PER_M,
})
#: The largest acceleration any contact leg may ask for, in m/s^2: the push's 0.1, which the others share.
LEG_ACCEL_CAP_M_S2 = 0.1
#: The shortest push, in mm (owner, 2026-09-29): a push opens at most its own length, and less than this cannot
#: open room for a finger.
SHORTEST_PUSH_MM = 10.0
#: The longest push, in mm: the owner's hard cap (2026-09-29). A longer one is refused, never shortened.
LONGEST_PUSH_MM = 50.0
#: How far the finger backs off along the push line before it lifts, in mm: the owner's 5 mm.
BACK_OFF_MM = 5.0
#: The leading finger comes down this far behind the part, so the push leg runs this plus the push distance, in mm:
#: the planner's contact gap.
CONTACT_GAP_MM = 10.0
#: P0 and the lift point stand this far above the contact height, in mm: the planner's rise, the grasp's standoff.
APPROACH_RISE_MM = 80.0
#: Every station the push drives lies within this of where the plan's own axes and distance put it, in mm.
STATION_TOLERANCE_MM = 1.0
#: The approach is at most this far from straight down (BASE -Z), in degrees: the planner pushes only on a support
#: within 5 deg of level, and comes down along that support's downward normal.
MAX_APPROACH_TILT_DEG = 5.0
#: The plan's part centre lies inside the offer's points' footprint (BASE XY) grown by this, in mm; outside it the
#: offer would keep another part out of the world than the one the plan pushes.
OFFER_FOOTPRINT_MARGIN_MM = 5.0
#: A gripper that measures its width counts as open within this of its widest opening, in mm.
JAWS_OPEN_TOLERANCE_MM = 2.0
#: The plan's closing and approach axes may be this far from perpendicular (as a cosine) and still make a tool pose.
_AXES_PERPENDICULAR_COS = 1e-3
#: The plan's closing and approach axes are unit vectors to within this.
_UNIT_TOLERANCE = 1e-3
_EPS_MM = 1e-6
#: How a guard or the planner refuses a motion before anything is sent: the arm stands where it stood. With the
#: planner's own no-plan refusal (``NO_PLAN_FAIL_SAFE_MESSAGE``) these are the only failures of P0 that leave the push
#: a refusal before motion. The pick loop's look motions use the same set.
_REFUSED_BEFORE_SENDING: frozenset[MotionStatus] = frozenset({
    MotionStatus.WORKSPACE_REJECTED,
    MotionStatus.IK_FAILED,
    MotionStatus.IK_QUALITY_REJECTED,
    MotionStatus.JOINT_LIMIT_REJECTED,
    MotionStatus.SELF_COLLISION_REJECTED,
    MotionStatus.PAYLOAD_REJECTED,
    MotionStatus.CONTINUITY_REJECTED,
})
_STOP_WORDS = "The arm stays where it stopped, nothing more was commanded, and a person decides what happens next."


class PushOutcomeCode(StrEnum):
    """What one :func:`execute_push` came to. Every ``refused_*`` value is reached before anything moved."""

    #: P0 and all four contact legs ran and the controller can still move; the arm stands at the lift point.
    PUSHED = "pushed"
    #: Something may have moved and then a motion was refused or failed, the arm did not come to rest, or the
    #: controller could not be read: the arm stopped where it is, a person decides. The service outcome of the same
    #: name.
    UNSAFE_RECOVERY_REFUSED = "unsafe_recovery_refused"
    #: The controller cannot move (a protective or emergency stop) after something may have moved. The failure
    #: reason of the same name, which no recovery action follows. A person clears the stop.
    CONTROLLER_NOT_OPERATIONAL = "controller_not_operational"
    #: The planner refused, or what was handed in is no plan.
    REFUSED_NO_PLAN = "refused_no_plan"
    #: The plan is not the shape of a push (P0 straight above the contact start, the push leg the contact gap plus
    #: the push along its direction, 5 mm back, straight up, coming down at most 5 deg from straight down), or its
    #: numbers or tool axes make no tool pose.
    REFUSED_PLAN_MALFORMED = "refused_plan_malformed"
    #: The plan pushes more than the owner's 50 mm.
    REFUSED_PUSH_TOO_LONG = "refused_push_too_long"
    #: The plan pushes less than 10 mm, which cannot open room for a finger.
    REFUSED_PUSH_TOO_SHORT = "refused_push_too_short"
    #: A contact leg asks for more speed or acceleration than the owner allows it, or the legs are not down, push,
    #: back and up in that order.
    REFUSED_LEG_TOO_FAST = "refused_leg_too_fast"
    #: The arm has no typed ``move``.
    REFUSED_NO_TYPED_MOVE = "refused_no_typed_move"
    #: The arm does not say it judges every sample of a straight line before it drives it, or cannot be read.
    REFUSED_LINES_NOT_JUDGED = "refused_lines_not_judged"
    #: The arm has no live camera world to keep the part out of.
    REFUSED_NO_LIVE_WORLD = "refused_no_live_world"
    #: The offer carries no BASE points of the part.
    REFUSED_NO_TARGET_POINTS = "refused_no_target_points"
    #: The plan's part stands outside the offer's points: the offer is of another part.
    REFUSED_OFFER_NOT_THE_PART = "refused_offer_not_the_part"
    #: The controller cannot move, or could not be read, before anything moved. Nothing was commanded, and there is
    #: nothing to fall through to: the caller ends the pick there, as a stopped controller ends it anywhere.
    REFUSED_CONTROLLER_STOPPED = "refused_controller_stopped"
    #: There is no gripper to read the jaws of.
    REFUSED_NO_GRIPPER = "refused_no_gripper"
    #: The toggle's count, or the measured width, says the jaws do not stand open.
    REFUSED_JAWS_CLOSED = "refused_jaws_closed"
    #: Nobody can say the jaws stand open: the count is unknown, DO0 was switched by hand, the hand is not connected,
    #: a read failed, or the gripper cannot say. On a hand that toggles with no sensor, and on a gripper that measures
    #: its width, the caller ends the pick there as a gripper fault (the owner, 2026-09-30): nobody vouches for that hand
    #: any more. Only a gripper that neither counts nor measures falls through on it.
    REFUSED_JAWS_UNKNOWN = "refused_jaws_unknown"
    #: The arm did not come to rest before P0 where the cell asks for that (``safety.dwell``).
    REFUSED_ARM_NOT_STEADY = "refused_arm_not_steady"
    #: The arm's live camera world did not take the part and its push path to keep out.
    REFUSED_KEEP_OUT_NOT_TAKEN = "refused_keep_out_not_taken"
    #: P0 was refused before anything was sent (a guard or the planner): the arm stands where it stood.
    REFUSED_APPROACH_NOT_SENT = "refused_approach_not_sent"
    #: The line down beside the part was refused before anything was sent (the arm's line judge, a guard or the
    #: planner): the arm stands in the air at P0, where it moved, and nothing touched the part. The caller moves it
    #: back to the look like a grasp approach, and then falls through as on any refusal before motion.
    REFUSED_DOWN_NOT_SENT = "refused_down_not_sent"


_STOPPED = frozenset({PushOutcomeCode.UNSAFE_RECOVERY_REFUSED, PushOutcomeCode.CONTROLLER_NOT_OPERATIONAL})


def push_budget_key(plan: PushPlan) -> tuple[Vec3, Vec3]:
    """The part's centre and its predicted landing, the key a push is counted by
    (:class:`~src.robot.grasping.recovery.push_budgets.PushBudgets`).

    The same centre goes to ``check`` before the push and to ``record`` after it, and the landing is the centre moved
    by the push along its direction: ``plan.target_centre_mm + plan.push_distance_mm * plan.direction``.
    """

    centre = np.asarray(plan.target_centre_mm, dtype=np.float64)
    landing = centre + float(plan.push_distance_mm) * np.asarray(plan.direction, dtype=np.float64)
    return _vec3(centre), _vec3(landing)


@dataclass(frozen=True, slots=True)
class PushOutcome:
    """What :func:`execute_push` did, and why it stopped where it did.

    ``leg`` names the motion the outcome is about: ``approach`` (P0) or a contact leg, the one that was refused or
    failed, or the one the controller stopped (or the arm did not rest, or a toggle's count went unknown) before;
    ``up`` where the controller was found stopped or unreadable, or a toggle's count nobody can vouch for, once the last
    leg ended; ``None`` for a refusal before any motion and for a finished push.
    ``legs_done`` are the motions that ran to the end, in order. ``results`` holds every typed result the arm
    returned, the failing one included, so the camera-world stamps travel with it. ``tcp_after`` is where the arm
    stands after the push, read once it ended (``None`` where nothing moved or the read failed). ``controller`` is
    what the controller said when it was asked after a failure.
    """

    code: PushOutcomeCode
    reason: str
    plan: Optional[PushPlan] = None
    refusal: Optional[PushRefusal] = None
    leg: Optional[str] = None
    legs_done: tuple[str, ...] = ()
    results: tuple[MotionResult, ...] = ()
    motion_status: Optional[MotionStatus] = None
    motion_message: str = ""
    error: Optional[BaseException] = field(default=None, repr=False, compare=False)
    tcp_after: Optional[Pose] = None
    controller: str = ""

    @property
    def executed(self) -> bool:
        """True where the whole push ran and the controller can still move."""
        return self.code is PushOutcomeCode.PUSHED

    @property
    def refused_before_motion(self) -> bool:
        """True where the push touched nothing and the loop falls through to its next action. Nothing moved, except
        on ``refused_down_not_sent``, where the arm went to P0 in the air (``legs_done``) and the caller moves it back
        to the look first."""
        return refused_before_motion(self.code)

    @property
    def motion_started(self) -> bool:
        """True where the push may have moved the part: it counts against its budgets (``PushBudgets.record``)."""
        return not self.refused_before_motion

    @property
    def stopped(self) -> bool:
        """True where the push stopped after something may have moved: the arm stays where it is, a person decides.
        No further motion follows, not even back to the look, and the campaign ends."""
        return self.code in _STOPPED

    @property
    def budget_centre_mm(self) -> Optional[Vec3]:
        """The centre to count this push by, ``None`` without a plan (:func:`push_budget_key`)."""
        return None if self.plan is None else push_budget_key(self.plan)[0]

    @property
    def budget_landing_mm(self) -> Optional[Vec3]:
        """The predicted landing to count this push by, ``None`` without a plan (:func:`push_budget_key`)."""
        return None if self.plan is None else push_budget_key(self.plan)[1]

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-serialisable values, for telemetry."""

        def _r(values: Optional[Vec3]) -> Optional[list[float]]:
            return None if values is None else [round(float(v), 3) for v in values]

        return {
            "code": self.code.value,
            "reason": self.reason,
            "leg": self.leg,
            "legs_done": list(self.legs_done),
            "motion_status": None if self.motion_status is None else self.motion_status.value,
            "motion_message": self.motion_message,
            "error": None if self.error is None else f"{type(self.error).__name__}: {self.error}",
            "controller": self.controller,
            "motion_started": self.motion_started,
            "budget_centre_mm": _r(self.budget_centre_mm),
            "budget_landing_mm": _r(self.budget_landing_mm),
            "tcp_after_mm": None if self.tcp_after is None else _r(_vec3(self.tcp_after.position_mm)),
            "plan": None if self.plan is None else self.plan.to_dict(),
            "refusal": None if self.refusal is None else self.refusal.to_dict(),
        }


def push_tool_quaternion(
    plan: PushPlan, *, current_tool_x: Optional[Any] = None, natural: Optional[ClosingAxis] = None,
) -> Optional[np.ndarray]:
    """The tool orientation for every pose of the push, as a canonical XYZW quaternion, or ``None`` where the plan's
    axes make none.

    The tool frame is the grasp frame: X closes, Z approaches (``plan.approach``, straight down onto the support), Y
    completes it. X lies along ``plan.direction`` either way, since either finger may lead. Where the cell names how
    its hand and camera naturally stand (``natural``, ``robot.natural_closing_axis``), the way round whose tool +X lies
    nearer that direction at the part is taken, as for every camera grasp (``closing_axis.turned_nearer``). Otherwise,
    and where the push runs straight across it, the sign nearer ``current_tool_x`` (the tool's X in BASE where the arm
    stands) is taken, so the wrist turns at most a quarter turn.
    """

    closing = np.asarray(plan.direction, dtype=np.float64)
    approach = np.asarray(plan.approach, dtype=np.float64)
    if closing.shape != (3,) or approach.shape != (3,) or not np.all(np.isfinite(closing)) \
            or not np.all(np.isfinite(approach)):
        return None
    a_norm, c_norm = float(np.linalg.norm(approach)), float(np.linalg.norm(closing))
    if a_norm < 1e-9 or c_norm < 1e-9:
        return None
    approach, closing = approach / a_norm, closing / c_norm
    if abs(float(closing @ approach)) > _AXES_PERPENDICULAR_COS:
        return None
    closing = closing - float(closing @ approach) * approach
    closing = closing / float(np.linalg.norm(closing))
    if natural is not None:
        from src.robot.grasping.geometry.closing_axis import turned_nearer  # noqa: PLC0415

        if turned_nearer(natural, plan.target_centre_mm, closing):
            closing, current_tool_x = -closing, None
        elif turned_nearer(natural, plan.target_centre_mm, -closing):
            current_tool_x = None
    if current_tool_x is not None:
        here = np.asarray(current_tool_x, dtype=np.float64).reshape(-1)
        if here.shape == (3,) and np.all(np.isfinite(here)) and float(closing @ here) < 0.0:
            closing = -closing
    binormal = np.cross(approach, closing)
    rotation = np.column_stack([closing, binormal, approach])
    return np.asarray(from_rotation_matrix(rotation), dtype=np.float64)


def execute_push(
    arm: Any,
    plan: Union[PushPlan, PushRefusal, Any],
    *,
    gripper: Any,
    offer: SegmentationOffer,
    natural_closing_axis: Optional[ClosingAxis] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> PushOutcome:
    """Drive one push: P0 like a grasp approach, then the four judged contact lines, all while the part and its
    swept travel are kept out of the arm's live camera world. See the module docstring for every rule.

    ``arm`` is the cell's :class:`~src.robot.core.RobotArm`. ``plan`` is what
    :func:`~src.robot.grasping.recovery.push_planner.plan_push` answered; its refusal is answered with a refusal here.
    ``gripper`` is the cell's gripper, only ever read. ``offer`` is the pick's segmentation offer of the part (its
    masks for their camera, its BASE points and the frame's shutter time, as the pick holds it for a grasp); the push
    holds the same offer with the part's points swept along the push. ``natural_closing_axis`` is how the cell's hand
    and camera naturally stand (``robot.natural_closing_axis``): the push closes the way round nearer it
    (:func:`push_tool_quaternion`); ``None`` keeps the way round nearer where the tool stands. ``should_cancel`` is the
    pick's stop check: asked between the legs, as the controller and the toggle's count are, a stop asked for ends the
    push where the arm stands, nothing more commanded, and a person decides (Track P's item 5); a check that raises
    counts as asked.

    Every failure is a :class:`PushOutcome`. It raises only ``CameraWorldUnavailable``, which a motion raises where
    the camera world cannot vouch for the cell (nothing more is commanded, and the keep-out scope closes on the way
    out), what the live world's ``forget_segmentation`` raises as that scope closes (the live world only clears its
    own memory there), and a ``BaseException`` that is no ``Exception``, such as a person's ``KeyboardInterrupt``.
    """

    ready = _before_motion(arm, plan, gripper=gripper, offer=offer, natural=natural_closing_axis)
    if isinstance(ready, PushOutcome):
        return ready
    push_plan, poses, held, steady_s = ready

    with contextlib.ExitStack() as scope:
        try:
            scope.enter_context(keeping_out(arm, held))
        except CameraWorldUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 (a world that refuses the offer refuses the push, nothing moved)
            return _refused(PushOutcomeCode.REFUSED_KEEP_OUT_NOT_TAKEN,
                            f"The arm's camera world did not take the part and its push path to keep out "
                            f"({type(exc).__name__}: {exc}), so nothing was commanded.", plan=push_plan)
        return _drive_push(arm, push_plan, poses, steady_s, gripper=gripper, should_cancel=should_cancel)


def _drive_push(arm: Any, plan: PushPlan, poses: tuple[Pose, ...], steady_s: Optional[float], *,
                gripper: Any = None, should_cancel: Optional[Callable[[], bool]] = None) -> PushOutcome:
    """P0, then the four contact legs, inside the open keep-out scope: the outcome, and nothing moved after a stop.
    Before each contact leg, and once the up leg ended, a toggle's count is read again (``gripper``): one nobody can
    vouch for stops the push where the arm stands, as a controller that cannot be read does."""

    move = arm.move
    results: list[MotionResult] = []
    done: list[str] = []

    result, error = _drive(move, poses[0], linear=False)
    if result is not None:
        results.append(result)
    if error is None and result is not None and result.ok:
        done.append(APPROACH_LEG)
    elif error is None and result is not None and _sent_nothing(result):
        return PushOutcome(
            code=PushOutcomeCode.REFUSED_APPROACH_NOT_SENT,
            reason=(f"The move to P0 above the push was refused before anything was sent ({result.status.value}: "
                    f"{result.message}); the arm stands where it stood."),
            plan=plan, leg=APPROACH_LEG, results=tuple(results), motion_status=result.status,
            motion_message=result.message,
        )
    else:
        return _stop(arm, plan, APPROACH_LEG, result, error, done, results)

    for leg, pose in zip(plan.legs, poses[1:]):
        halted = _halt(arm, plan, f"Before the {leg.name} leg", leg.name, done, results, steady_s=steady_s,
                       gripper=gripper, should_cancel=should_cancel)
        if halted is not None:
            return halted
        result, error = _drive(move, pose, linear=True, vel=leg.vel_m_s, acc=leg.acc_m_s2)
        if result is not None:
            results.append(result)
        if error is None and result is not None and result.ok:
            done.append(leg.name)
            continue
        if leg.name == CONTACT_LEGS[0] and error is None and result is not None and _sent_nothing(result):
            # The down leg was refused before anything was sent: the arm is in the air at P0 and nothing touched the
            # part. A fall-through, not a stop (the lead's ruling of 2026-09-29, pending the owner): the caller moves the
            # arm back to the look. Unless a stop was asked for: then the arm stays in the air (Track P's item 5).
            if _asked_to_stop(should_cancel):
                return _finish(arm, PushOutcome(
                    code=PushOutcomeCode.UNSAFE_RECOVERY_REFUSED,
                    reason=(f"The line down beside the part was refused before anything was sent ({result.status.value}: "
                            f"{result.message}), and a stop was asked for: the arm stays in the air at P0. {_STOP_WORDS}"),
                    plan=plan, leg=leg.name, legs_done=tuple(done), results=tuple(results),
                    motion_status=result.status, motion_message=result.message,
                ))
            return _finish(arm, PushOutcome(
                code=PushOutcomeCode.REFUSED_DOWN_NOT_SENT,
                reason=(f"The line down beside the part was refused before anything was sent ({result.status.value}: "
                        f"{result.message}); the arm stands in the air at P0 and nothing touched the part, so it goes "
                        "back to the look."),
                plan=plan, leg=leg.name, legs_done=tuple(done), results=tuple(results), motion_status=result.status,
                motion_message=result.message,
            ))
        return _stop(arm, plan, leg.name, result, error, done, results)

    halted = _halt(arm, plan, "After the up leg", CONTACT_LEGS[-1], done, results, steady_s=None, gripper=gripper,
                   should_cancel=should_cancel)
    if halted is not None:
        return halted
    return _finish(arm, PushOutcome(
        code=PushOutcomeCode.PUSHED,
        reason=(f"The part was pushed {plan.push_distance_mm:g} mm and the hand lifted clear; the move back to "
                "the look is the caller's."),
        plan=plan, legs_done=tuple(done), results=tuple(results),
        motion_status=results[-1].status if results else None,
        motion_message=results[-1].message if results else "",
    ))


# --- before anything moves -------------------------------------------------------------------------------------


def _refused(code: PushOutcomeCode, reason: str, *, plan: Optional[PushPlan] = None,
             refusal: Optional[PushRefusal] = None) -> PushOutcome:
    return PushOutcome(code=code, reason=reason, plan=plan, refusal=refusal)


def _before_motion(
    arm: Any, plan: Any, *, gripper: Any, offer: SegmentationOffer, natural: Optional[ClosingAxis] = None,
) -> Union[PushOutcome, tuple[PushPlan, tuple[Pose, ...], SegmentationOffer, Optional[float]]]:
    """Every precondition, read and never asked: the refusal, or the plan, its five poses, the offer to hold and the
    steady gate's timeout (``None`` where the cell asks for none)."""

    if isinstance(plan, PushRefusal):
        return _refused(PushOutcomeCode.REFUSED_NO_PLAN,
                        f"The push planner refused ({plan.code}): {plan.sentence}", refusal=plan)
    if not isinstance(plan, PushPlan):
        return _refused(PushOutcomeCode.REFUSED_NO_PLAN,
                        f"{type(plan).__name__} is not a push plan, so nothing is pushed.")
    too_fast = _legs_problem(plan)
    if too_fast:
        return _refused(PushOutcomeCode.REFUSED_LEG_TOO_FAST, too_fast, plan=plan)
    shape = _shape_problem(plan)
    if shape is not None:
        return _refused(shape[0], shape[1], plan=plan)

    move = getattr(arm, "move", None)
    if not callable(move):
        return _refused(PushOutcomeCode.REFUSED_NO_TYPED_MOVE,
                        "The arm has no typed move, so no leg of the push could be judged, and nothing is pushed.",
                        plan=plan)
    try:
        reading = line_motion_of(arm)
    except Exception as exc:  # noqa: BLE001 (an arm that cannot say what it keeps of a line is refused, not raised)
        return _refused(PushOutcomeCode.REFUSED_LINES_NOT_JUDGED,
                        f"What the arm keeps of a straight line could not be read ({type(exc).__name__}: {exc}), "
                        "so nothing is pushed.", plan=plan)
    if not isinstance(reading, LineReading) or reading.motion is not LineMotion.CHECKED:
        said = (f"keeps a straight line as {reading.motion.value} ({reading.reason})"
                if isinstance(reading, LineReading) else "does not say what it keeps of a straight line")
        return _refused(PushOutcomeCode.REFUSED_LINES_NOT_JUDGED,
                        f"The arm {said}; the push's contact legs run only as lines judged sample by sample before "
                        "they move, so nothing is pushed.", plan=plan)
    if getattr(arm, "live_planner_world", None) is None:
        return _refused(PushOutcomeCode.REFUSED_NO_LIVE_WORLD,
                        "The arm has no live camera world, so the part could not be kept out of it along its push "
                        "and nothing is pushed.", plan=plan)
    points = _offer_points(offer)
    if points is None:
        return _refused(PushOutcomeCode.REFUSED_NO_TARGET_POINTS,
                        "The offer carries no finite BASE points of the part (N x 3), so its push path could not be "
                        "kept out of the camera world and nothing is pushed.", plan=plan)
    elsewhere = _offer_problem(plan, points)
    if elsewhere:
        return _refused(PushOutcomeCode.REFUSED_OFFER_NOT_THE_PART, elsewhere, plan=plan)

    words, read_error = _controller_reading(arm)
    if words:
        return _refused(PushOutcomeCode.REFUSED_CONTROLLER_STOPPED,
                        f"{words}. Nothing was commanded.", plan=plan)
    if read_error:
        return _refused(PushOutcomeCode.REFUSED_CONTROLLER_STOPPED,
                        f"The controller could not be read ({read_error}), so nothing was commanded.", plan=plan)

    jaws = _jaws_refusal(gripper)
    if jaws is not None:
        return _refused(jaws[0], jaws[1], plan=plan)

    try:
        quaternion = push_tool_quaternion(plan, current_tool_x=_tool_x(arm), natural=natural)
        if quaternion is None:
            return _refused(PushOutcomeCode.REFUSED_PLAN_MALFORMED,
                            f"The plan's closing axis {plan.direction} and approach {plan.approach} make no tool "
                            "pose, so nothing is pushed.", plan=plan)
        stations = (("push_p0", plan.approach_tcp_mm),) + tuple(
            (f"push_{leg.name}", leg.end_tcp_mm) for leg in plan.legs)
        poses = tuple(Pose(position_mm=np.asarray(position, dtype=np.float64), quaternion_xyzw=quaternion.copy(),
                           frame=Frame.BASE, label=label) for label, position in stations)
        # The part moves as far as the push leg drives past the contact gap: the shape check holds that to the
        # plan's push distance within the stations' tolerance, and the sweep takes the longer of the two.
        driven = float(np.linalg.norm(poses[2].position_mm - poses[1].position_mm)) - CONTACT_GAP_MM
        swept = swept_target_points_mm(points, plan.direction, max(float(plan.push_distance_mm), driven))
        held = dataclasses.replace(offer, target_points_base_mm=np.vstack([points, swept]))
    except (TypeError, ValueError, GeometryError) as exc:
        return _refused(PushOutcomeCode.REFUSED_PLAN_MALFORMED,
                        f"The plan's stations make no pose ({exc}), so nothing is pushed.", plan=plan)

    steady_s = steady_timeout_of(arm)
    restless = _not_at_rest(arm, steady_s)
    if restless:
        return _refused(PushOutcomeCode.REFUSED_ARM_NOT_STEADY, f"{restless}, so nothing was commanded.", plan=plan)
    return plan, poses, held, steady_s


def _legs_problem(plan: PushPlan) -> str:
    """Why the plan's contact legs may not run, or ``""``: the four legs in order, each at or below its cap."""

    legs = plan.legs
    names = tuple(leg.name for leg in legs)
    if names != CONTACT_LEGS:
        return f"The plan's contact legs are {list(names)}, not {list(CONTACT_LEGS)}, so nothing is pushed."
    for leg in legs:
        cap = LEG_SPEED_CAPS_M_S[leg.name]
        speed, accel = float(leg.vel_m_s), float(leg.acc_m_s2)
        if not (math.isfinite(speed) and 0.0 < speed <= cap + 1e-12):
            return (f"The {leg.name} leg asks for {speed:g} m/s and the owner allows it at most {cap:g} m/s, so "
                    "nothing is pushed.")
        if not (math.isfinite(accel) and 0.0 < accel <= LEG_ACCEL_CAP_M_S2 + 1e-12):
            return (f"The {leg.name} leg asks for {accel:g} m/s^2 and a contact leg may accelerate at most "
                    f"{LEG_ACCEL_CAP_M_S2:g} m/s^2, so nothing is pushed.")
    return ""


def _shape_problem(plan: PushPlan) -> Optional[tuple[PushOutcomeCode, str]]:
    """Why the plan is not the shape of a push the owner allows, or ``None``. Checked on the plan as handed in,
    whoever built it: :func:`~src.robot.grasping.recovery.push_planner.plan_push` makes only plans that pass."""

    malformed = PushOutcomeCode.REFUSED_PLAN_MALFORMED
    try:
        distance = float(plan.push_distance_mm)
        direction = _unit(plan.direction, "closing axis")
        approach = _unit(plan.approach, "approach")
        here = _finite3(plan.approach_tcp_mm, "P0")
        _finite3(plan.target_centre_mm, "part centre")
        legs = tuple((leg.name, _finite3(leg.start_tcp_mm, f"{leg.name} start"),
                      _finite3(leg.end_tcp_mm, f"{leg.name} end")) for leg in plan.legs)
    except (TypeError, ValueError, IndexError) as exc:
        return malformed, f"The plan's numbers make no push ({exc}), so nothing is pushed."
    if not math.isfinite(distance):
        return malformed, f"The plan's push distance of {distance!r} mm is not a number, so nothing is pushed."
    if distance > LONGEST_PUSH_MM + _EPS_MM:
        return (PushOutcomeCode.REFUSED_PUSH_TOO_LONG,
                f"The plan pushes {distance:g} mm and the owner's hard cap is {LONGEST_PUSH_MM:g} mm, so nothing is "
                "pushed.")
    if distance < SHORTEST_PUSH_MM - _EPS_MM:
        return (PushOutcomeCode.REFUSED_PUSH_TOO_SHORT,
                f"The plan pushes {distance:g} mm and a push shorter than {SHORTEST_PUSH_MM:g} mm cannot open room "
                "for a finger, so nothing is pushed.")
    if abs(float(direction @ approach)) > _AXES_PERPENDICULAR_COS:
        return malformed, (f"The plan's closing axis {plan.direction} is not square to its approach "
                           f"{plan.approach}, so they make no tool pose and nothing is pushed.")
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(approach[2])))))
    if tilt > MAX_APPROACH_TILT_DEG + 1e-9:
        return malformed, (f"The plan comes down {tilt:.1f} deg from straight down and a push comes down at most "
                           f"{MAX_APPROACH_TILT_DEG:g} deg from it, so nothing is pushed.")
    expected = {
        "down": (APPROACH_RISE_MM * approach, f"{APPROACH_RISE_MM:g} mm straight down along its approach"),
        "push": ((CONTACT_GAP_MM + distance) * direction,
                 f"the {CONTACT_GAP_MM:g} mm contact gap plus its {distance:g} mm push along its direction"),
        "back": (-BACK_OFF_MM * direction, f"{BACK_OFF_MM:g} mm back along its push"),
        "up": (-APPROACH_RISE_MM * approach, f"{APPROACH_RISE_MM:g} mm straight up along its approach"),
    }
    previous = "P0"
    for name, start, end in legs:
        if float(np.linalg.norm(start - here)) > STATION_TOLERANCE_MM:
            return malformed, (f"The plan's {name} leg starts {float(np.linalg.norm(start - here)):.1f} mm from "
                               f"where {previous} ends, so nothing is pushed.")
        run, said = expected[name]
        off = float(np.linalg.norm((end - here) - run))
        if off > STATION_TOLERANCE_MM:
            return malformed, (f"The plan's {name} leg runs {float(np.linalg.norm(end - here)):.1f} mm and ends "
                               f"{off:.1f} mm from where {said} puts it (at most {STATION_TOLERANCE_MM:g} mm), so "
                               "nothing is pushed.")
        here, previous = end, f"the {name} leg"
    return None


def _unit(values: Any, name: str) -> np.ndarray:
    vector = _finite3(values, name)
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > _UNIT_TOLERANCE:
        raise ValueError(f"the {name} {tuple(values)} is not a unit vector (length {norm:g})")
    return vector / norm


def _finite3(values: Any, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"the {name} {values!r} is not three finite numbers")
    return vector


def _offer_points(offer: SegmentationOffer) -> Optional[np.ndarray]:
    """The offer's finite BASE points of the part as an (N, 3) array, or ``None`` where it carries none."""

    raw = getattr(offer, "target_points_base_mm", None)
    if raw is None:
        return None
    try:
        points = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    points = points[np.all(np.isfinite(points), axis=1)]
    return points if len(points) else None


def _offer_problem(plan: PushPlan, points: np.ndarray) -> str:
    """Why the offer is not of the part the plan pushes, or ``""``: the plan's part centre lies inside the offer's
    footprint in BASE XY, grown by :data:`OFFER_FOOTPRINT_MARGIN_MM`."""

    low = points[:, :2].min(axis=0) - OFFER_FOOTPRINT_MARGIN_MM
    high = points[:, :2].max(axis=0) + OFFER_FOOTPRINT_MARGIN_MM
    centre = np.asarray(plan.target_centre_mm, dtype=np.float64)[:2]
    if bool(np.all(centre >= low) and np.all(centre <= high)):
        return ""
    return (f"The plan pushes a part at ({centre[0]:.0f}, {centre[1]:.0f}) mm and the offer's points span x "
            f"{low[0] + OFFER_FOOTPRINT_MARGIN_MM:.0f} to {high[0] - OFFER_FOOTPRINT_MARGIN_MM:.0f} mm, y "
            f"{low[1] + OFFER_FOOTPRINT_MARGIN_MM:.0f} to {high[1] - OFFER_FOOTPRINT_MARGIN_MM:.0f} mm: the offer "
            "is of another part, and keeping that one out would leave the pushed one in the world, so nothing is "
            "pushed.")


def _controller_reading(arm: Any) -> tuple[str, str]:
    """(why the controller cannot move or ``""``, why it could not be read or ``""``). An arm that does not report
    its controller passes, as it passes the grasp; a halted arm never does. The halt latch is read first, of any arm
    that carries one (the dummy included), then a status that says so in words, and a halt is said in its own words:
    a person confirms the cell is clear, then Restart, never clears a stop that is not there."""

    halted = halt_state_of(arm)
    if halted is not None:
        return _halted_words(halted.reason), ""
    if not isinstance(arm, SupportsRobotStatus):
        return "", ""
    try:
        status = arm.get_robot_status()
        said = getattr(status, "halted", "")
        if isinstance(said, str) and said:
            return _halted_words(said), ""
        if status.is_operational:
            return "", ""
        detail = f": {status.message}" if status.message else ""
        words = (f"The controller cannot move (robot_mode={status.robot_mode.value}, "
                 f"safety_mode={status.safety_mode.value}, protective_stop={status.protective_stopped}, "
                 f"emergency_stop={status.emergency_stopped}{detail}); a person clears the stop where the arm is "
                 "visible")
    except Exception as exc:  # noqa: BLE001 (a controller that cannot be read is said, never raised past the push)
        return "", f"{type(exc).__name__}: {exc}"
    return words, ""


def _halted_words(reason: str) -> str:
    """The push's words for a halted arm, said as its stopped controller's are: one sentence, capitalised."""
    said = halted_refusal(reason, "the push goes no further")
    return said[:1].upper() + said[1:]


def _not_at_rest(arm: Any, timeout_s: Optional[float]) -> str:
    """Why the arm is not at rest, or ``""``: waited on only where the cell asks for it (``timeout_s`` set)."""

    if timeout_s is None:
        return ""
    try:
        if arm.wait_until_steady(timeout_s):
            return ""
    except Exception as exc:  # noqa: BLE001 (a wait that fails is an arm nobody can say is at rest)
        return f"The arm could not be waited on to come to rest ({type(exc).__name__}: {exc}, safety.dwell)"
    return f"The arm did not come to rest within {timeout_s:g} s (safety.dwell)"


def _jaws_refusal(gripper: Any) -> Optional[tuple[PushOutcomeCode, str]]:
    """Why the jaws cannot be said to stand open, read without asking or sending anything; ``None`` where they do.
    A read that raises is a jaw nobody can vouch for."""

    try:
        return _jaws_reading(gripper)
    except Exception as exc:  # noqa: BLE001 (a failed read refuses the push, never escapes it)
        return (PushOutcomeCode.REFUSED_JAWS_UNKNOWN,
                f"The jaws could not be read ({type(exc).__name__}: {exc}), so nothing is pushed and nobody is asked.")


def _jaws_reading(gripper: Any) -> Optional[tuple[PushOutcomeCode, str]]:
    if gripper is None:
        return (PushOutcomeCode.REFUSED_NO_GRIPPER,
                "There is no gripper to read, so nobody can say the jaws stand open, and nothing is pushed.")
    toggle = toggle_without_sensor_of(gripper)
    if toggle is not None:
        if getattr(gripper, "is_connected", False) is not True:
            return (PushOutcomeCode.REFUSED_JAWS_UNKNOWN,
                    "The hand is not connected, so its count stands for nothing: the count ends at a disconnect and "
                    "the next connect asks a person where the jaws stand. Nothing is pushed and nobody is asked.")
        why_unknown = getattr(toggle, "why_jaws_unknown", None)
        if not callable(why_unknown):
            return (PushOutcomeCode.REFUSED_JAWS_UNKNOWN,
                    "This toggle hand cannot say whether its count stands (no why_jaws_unknown), so nothing is "
                    "pushed.")
        if bool(toggle.edge_unknown):
            why = str(why_unknown() or "its count is unknown")
            return (PushOutcomeCode.REFUSED_JAWS_UNKNOWN,
                    f"Nobody can say where the jaws stand ({why}), so nothing is pushed and nobody is asked.")
        why = str(why_unknown())
        if why:
            return (PushOutcomeCode.REFUSED_JAWS_UNKNOWN,
                    f"Nobody can say where the jaws stand ({why}), so nothing is pushed and nobody is asked.")
        if bool(toggle.jaws_closed):
            return (PushOutcomeCode.REFUSED_JAWS_CLOSED,
                    "The count says the jaws stand closed. A push runs with open jaws and never switches them, so "
                    "nothing is pushed and nobody is asked.")
        return None
    if width_is_measured_of(gripper):
        if getattr(gripper, "is_connected", False) is not True:
            return (PushOutcomeCode.REFUSED_JAWS_UNKNOWN,
                    "The gripper is not connected, so no width it reports says where the jaws stand, and nothing "
                    "is pushed.")
        width = float(gripper.get_width_mm())
        widest = float(gripper.max_width_mm)
        if not (math.isfinite(width) and math.isfinite(widest)) or width < widest - JAWS_OPEN_TOLERANCE_MM:
            return (PushOutcomeCode.REFUSED_JAWS_CLOSED,
                    f"The jaws read {width:g} mm open and a push needs them open ({widest:g} mm, within "
                    f"{JAWS_OPEN_TOLERANCE_MM:g} mm), so nothing is pushed.")
        return None
    return (PushOutcomeCode.REFUSED_JAWS_UNKNOWN,
            f"The gripper ({type(gripper).__name__}) neither counts its changes nor measures its width, so nobody can "
            "say its jaws stand open, and nothing is pushed.")


def _tool_x(arm: Any) -> Optional[np.ndarray]:
    """The tool's X in BASE where the arm stands, read; ``None`` where the arm cannot say."""

    read = getattr(arm, "get_tcp_pose", None)
    if not callable(read):
        return None
    try:
        pose = read()
        return np.asarray(to_rotation_matrix(np.asarray(pose.quaternion_xyzw, dtype=np.float64))[:, 0])
    except Exception:  # noqa: BLE001 (the sign of the closing axis is a preference, never a reason to fail)
        return None


# --- while it moves --------------------------------------------------------------------------------------------


def _drive(move: Any, pose: Pose, *, linear: bool, vel: Optional[float] = None,
           acc: Optional[float] = None) -> tuple[Optional[MotionResult], Optional[BaseException]]:
    """One motion: its typed result, or what it raised. ``CameraWorldUnavailable`` is raised on, as a pick raises it."""

    try:
        if linear:
            result = move(pose, linear=True, vel=vel, acc=acc)
        else:
            result = move(pose)
    except CameraWorldUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 (a failed motion is an outcome, never an escaping raise)
        return None, exc
    if not isinstance(result, MotionResult):
        return None, TypeError(f"the arm's move answered {type(result).__name__}, not a MotionResult")
    return result, None


def _sent_nothing(result: MotionResult) -> bool:
    """Whether a failed motion was refused before anything was sent (:data:`_REFUSED_BEFORE_SENDING`)."""

    if result.status is MotionStatus.TIMEOUT:
        return result.message == NO_PLAN_FAIL_SAFE_MESSAGE
    return result.status in _REFUSED_BEFORE_SENDING


def _halt(arm: Any, plan: PushPlan, when: str, leg: str, done: list[str], results: list[MotionResult], *,
          steady_s: Optional[float], gripper: Any = None,
          should_cancel: Optional[Callable[[], bool]] = None) -> Optional[PushOutcome]:
    """The stop between two motions, or ``None`` where the next may run: a stop asked for (``should_cancel``, Track P's
    item 5), the controller is read, a toggle's count is read where ``gripper`` is handed in (before each contact leg
    and once the up leg ended; nobody can vouch for it once DO0 was switched at the pendant while the push drove, the
    owner, 2026-09-30), and the arm waited on to rest where ``steady_s`` is set.
    Nothing is commanded either way, and nobody is asked."""

    if _asked_to_stop(should_cancel):
        return _finish(arm, PushOutcome(
            code=PushOutcomeCode.UNSAFE_RECOVERY_REFUSED, reason=f"{when}: a stop was asked for. {_STOP_WORDS}",
            plan=plan, leg=leg, legs_done=tuple(done), results=tuple(results),
        ))
    words, read_error = _controller_reading(arm)
    said = words or (f"The controller could not be read ({read_error})" if read_error else "")
    unknown = "" if said or gripper is None else why_toggle_count_unknown(gripper)
    jaws = f"Nobody can say where the jaws stand ({unknown})" if unknown else ""
    restless = "" if said or jaws else _not_at_rest(arm, steady_s)
    if not (said or jaws or restless):
        return None
    return _finish(arm, PushOutcome(
        code=PushOutcomeCode.CONTROLLER_NOT_OPERATIONAL if words else PushOutcomeCode.UNSAFE_RECOVERY_REFUSED,
        reason=f"{when}: {said or jaws or restless}. {_STOP_WORDS}",
        plan=plan, leg=leg, legs_done=tuple(done), results=tuple(results), controller=said,
    ))


def _asked_to_stop(should_cancel: Optional[Callable[[], bool]]) -> bool:
    """Whether the pick's stop check says a stop was asked for; a check that raises counts as asked."""

    if should_cancel is None:
        return False
    try:
        return bool(should_cancel())
    except Exception:  # noqa: BLE001 (a stop check that cannot answer is no permission to move on)
        return True


def _stop(arm: Any, plan: PushPlan, leg: str, result: Optional[MotionResult], error: Optional[BaseException],
          done: list[str], results: list[MotionResult]) -> PushOutcome:
    """The push stops where the arm is: nothing more is commanded. The controller says which kind of stop it is."""

    words, read_error = _controller_reading(arm)
    said = words or (f"The controller could not be read ({read_error})" if read_error else "")
    if result is not None:
        what = f"{result.status.value}: {result.message}" if result.message else result.status.value
    elif error is not None:
        what = f"it raised {type(error).__name__}: {error}"
    else:  # pragma: no cover (defensive: _drive answers one or the other)
        what = "no result"
    where = "before any contact" if leg == APPROACH_LEG else "after the push's first contact leg was commanded"
    reason = f"The {leg} motion did not finish ({what}), {where}. {_STOP_WORDS}"
    if said:
        reason = f"{reason} {said}."
    return _finish(arm, PushOutcome(
        code=PushOutcomeCode.CONTROLLER_NOT_OPERATIONAL if words else PushOutcomeCode.UNSAFE_RECOVERY_REFUSED,
        reason=reason, plan=plan, leg=leg, legs_done=tuple(done), results=tuple(results),
        motion_status=None if result is None else result.status,
        motion_message="" if result is None else result.message, error=error, controller=said,
    ))


def _finish(arm: Any, outcome: PushOutcome) -> PushOutcome:
    """``outcome`` with where the arm stands now, read (a read, never a motion)."""

    read = getattr(arm, "get_tcp_pose", None)
    if not callable(read):
        return outcome
    try:
        where = read()
    except Exception:  # noqa: BLE001 (where the arm stands is a report, never a reason to fail)
        return outcome
    return dataclasses.replace(outcome, tcp_after=where if isinstance(where, Pose) else None)


def _vec3(values: Any) -> Vec3:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return (float(array[0]), float(array[1]), float(array[2]))
