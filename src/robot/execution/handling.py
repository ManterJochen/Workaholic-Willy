"""What a robot's hand does as one verb: close on a part, open off it, and say what was measured.

``Robot.grasp``, ``Robot.release`` and ``Robot.is_holding`` delegate here. A verb commands the gripper, reads back the
width, whether that width was measured, and what the gripper measured about the hold, and hands a part it holds to the
arm's carried part model where the arm has one. Every reading is a capability the gripper or the arm opts into
(``core.gripper.ReportsHoldEvidence``, ``MeasuresWidth``, ``core.arm_capabilities.CarriesPayload``), so a driver that
cannot say is read as not measured and not modelled rather than guessed.

* A hold the gripper measured, or that nothing could measure, is handed to the arm as a carried part; a measured empty
  close hands nothing.
* A release opens to the hand's width. A part still measured in the jaws keeps its model and the release is not
  confirmed; an unmeasured release detaches and says it was not checked.
* A hand that toggles with no sensor (``core.gripper.TogglesWithoutSensor``, a ``jaw_io`` single_toggle) is told open
  or close and never a width: a report carries no width, says the hold and the release were not checked because there
  is no sensor, and says where the jaws already stood and nothing was sent.
* No force is commanded.
* A gripper that raises is stopped once where it can be stopped, and the report carries the fault.
* Nothing is commanded on a robot with no gripper, a gripper that holds nothing, an arm or gripper whose link is not
  open, or an arm whose controller cannot move (``SupportsRobotStatus`` reading not operational, or a status read that
  raises): after a protective stop the jaws stay as they are, whatever they hold.

``Robot.pick`` and ``Robot.place`` put the hand verbs at the end of a straight descent. Before any command they refuse,
in this order: a robot that cannot hold, a link that is not open, a pose not in BASE, a camera world the arm's own
motion would be refused for (``ReadsCameraWorld``), an arm whose motions do not go through cuRobo and the exact mesh
guard (``motion.route_of``: a desk arm runs, a UR on the ik planner, a KUKA and an arm that does not say are refused),
and last, as the one question put to the controller, a controller that cannot move. A pick on a hand that toggles with
no sensor then asks it whether its jaws stand open (``jaws_open_for_a_pick``) instead of opening them: never a change
before the arm moves, and the pick is refused where the hand believes them closed and nobody at a terminal says
otherwise. Then the motions: a planned move to
the standoff, a line down to the pose, the hand verb, which asks the controller again at the part, and a line out of it,
each move carrying the caller's decline, each preceded by the arm's own steady gate where its tree asks for one
(``safety.dwell``); an arm that gates its own sends (``gate_at`` ``send``) waits for steady right before it sends instead,
after it judged the move. The line out of a pick goes back up to the standoff, or, for a grasp more than
:data:`LIFT_STRAIGHT_UP_BEYOND_DEG` off vertical, straight up (BASE +Z) by the standoff and at least
:data:`MIN_STRAIGHT_LIFT_MM`: a side grasp does not drag its part sideways along the support (side grasps are on, the
owner, 2026-10-01). Before the hand verb of a pick, an arm that judges a line as if its jaws held a part
(``JudgesCarriedLines``, the UR driver) is asked about that line out with the part in them, the caller's decline with
it: where it would refuse it, the jaws stay open, the arm goes back up the line it came down, and the pick ends
``CARRIED_RETREAT_REFUSED`` (the owner, 2026-10-01). An empty hand always backs out the way it came. A refused motion
ends the verb with nothing commanded after it, and so does a camera that could not vouch for the cell, as its own
outcome rather than a raise. A pick holds its keep-out offer in the arm's live world from before the detach to after its
last motion, and forgets it on any exit. A place holds its own, what the part is set down on, from before its first
motion to after its last, the release between them, and forgets it on any exit: the part comes down to within the line
clearance of what a camera located under it (owner's example 13, 2026-09-24). A place handed a drop over a box's rim
(``instead``, for a part set down below it) lets the part go there where the guard or the planner refuses the line into
the box before anything was sent: the line from the same standoff to ``instead``, judged as every line is, and the
refused line is no motion of the place (the owner, 2026-10-08 night).

The owner's held worlds (``safety.planning_world.hold``, 2026-10-09), each off as shipped: at ``standoff`` the line in
or down is judged in the world its route was judged in, with no frame at the standoff; at ``drop`` the release and the
line out of a place hold the world the line in was judged in, with no frame after the release. Where the arm judges the
next leg (``robot.motion.judge_next_leg``), a place's release leaves the jaws' stroke to the verb where the hand can
(``wait_settled``): the one change goes out, the part is forgotten, and in the world held at the drop the line out is
judged meanwhile, as it will run, and the joint move a task declared after the place (``expecting_next``) from where the
line out ends; nothing is sent before the stroke is over.

A pick's report says whether another candidate may follow (``another_candidate_may_follow``): only after a guard or the
planner refused its first motion, nothing sent and nothing commanded to the jaws, or after ``CARRIED_RETREAT_REFUSED``
with the empty hand backed out open. Every verb that fails leaves a WARNING with its message in the robot log.

The module imports nothing above ``robot.core`` but its sibling :mod:`~src.robot.execution.motion`, which imports
nothing above it either, and the robot package's logger factory, and connects nothing: the verbs run inside
``Robot.connected()``.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Final, cast

import numpy as np

from src.contracts import UNSET, Maybe, chosen
from src.geometry import Frame, Pose
from src.robot.core import MotionResult, MotionStatus, StoppableGripper, SupportsRobotStatus, TwoStateGripper
from src.robot.core.gripper import OpensAndCloses
from src.robot.core.arm_capabilities import (
    CarriesPayload,
    JudgesCarriedLines,
    LineMotion,
    LineReading,
    PayloadModel,
    RobotStatus,
    halt_state_of,
    halted_refusal,
)
from src.robot.core.camera_world import (
    CameraWorldDecline,
    CameraWorldStamp,
    ReadsCameraWorld,
    camera_world_refusal,
    weakest_camera_world,
)
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.core.gripper import HoldEvidence, hold_evidence_of, toggle_without_sensor_of, width_is_measured_of
from src.robot.core.keep_out import SegmentationOffer, keeping_out
from src.robot.execution.motion import gates_its_own_sends, route_of, steady_timeout_of
from src.robot.constants import create_robot_logger
from src.robot.core.motion_result import NO_PLAN_FAIL_SAFE_MESSAGE

__all__ = [
    "LIFT_STRAIGHT_UP_BEYOND_DEG",
    "MIN_STRAIGHT_LIFT_MM",
    "HandOutcome",
    "HandReport",
    "HandVerb",
    "HandlingOutcome",
    "HandlingReport",
    "HandlingVerb",
    "PayloadState",
    "grasp",
    "is_holding",
    "pick",
    "place",
    "release",
]

#: Every verb that fails says so here, and in the robot log beside every other robot event.
_LOG = create_robot_logger(__name__, "handling.log")

#: How far off vertical a grasp may stand, degrees, and still back out of the part along its approach. Past it the part
#: lifts straight up: backing out of a side grasp along its approach drags the part sideways along the support.
LIFT_STRAIGHT_UP_BEYOND_DEG: Final[float] = 10.0
#: The least a part lifts straight up off a tilted grasp, millimetres, whatever the standoff: the owner's standoff is
#: 10 mm, and a part 10 mm off the support is still among what stands on it.
MIN_STRAIGHT_LIFT_MM: Final[float] = 60.0

#: The failures a guard or the planner answers before anything is sent: ``generated_view.REFUSED_BEFORE_SENDING``, the
#: set the wrist pick and the pick loop read, restated because this module imports nothing above ``robot.core`` (a test
#: pins the two equal). A TIMEOUT is the planner's refusal only in its own words, ``NO_PLAN_FAIL_SAFE_MESSAGE``.
_REFUSED_BEFORE_SENDING: Final[frozenset[MotionStatus]] = frozenset({
    MotionStatus.WORKSPACE_REJECTED,
    MotionStatus.IK_FAILED,
    MotionStatus.IK_QUALITY_REJECTED,
    MotionStatus.JOINT_LIMIT_REJECTED,
    MotionStatus.SELF_COLLISION_REJECTED,
    MotionStatus.PAYLOAD_REJECTED,
    MotionStatus.CONTINUITY_REJECTED,
})


class HandVerb(StrEnum):
    """Which hand verb a report is about."""

    GRASP = "grasp"
    RELEASE = "release"


class HandOutcome(StrEnum):
    """How a hand verb (``Robot.grasp``, ``Robot.release``) ended.

    Attributes:
        GRASPED: The jaws closed and nothing measured them empty.
        NOTHING_HELD: The jaws closed and the hand measured nothing held.
        RELEASED: The jaws opened and nothing measured a part still in them.
        RELEASE_NOT_CONFIRMED: The jaws opened and the hand still measures a part: the carried part stays modelled.
        GRIPPER_FAULT: The hand raised; it was stopped where it can be.
        REFUSED: Nothing was commanded.
    """

    #: The jaws closed and nothing measured them empty.
    GRASPED = "grasped"
    #: The jaws closed and the gripper measured nothing held.
    NOTHING_HELD = "nothing_held"
    #: The jaws opened and nothing measured a part still in them.
    RELEASED = "released"
    #: The jaws opened and the gripper still measures a part: the carried part stays modelled.
    RELEASE_NOT_CONFIRMED = "release_not_confirmed"
    #: The gripper raised; it was stopped where it can be.
    GRIPPER_FAULT = "gripper_fault"
    #: Nothing was commanded.
    REFUSED = "refused"


class PayloadState(StrEnum):
    """What a hand verb did to the arm's model of a carried part."""

    #: The planner carries the part and the self filter takes it out.
    ATTACHED = "attached"
    #: The self filter takes the part out; the planner declined it and routes as if the gripper were empty.
    FILTER_ONLY = "filter_only"
    #: Nothing models the part; ``payload_reason`` says why.
    NOT_MODELLED = "not_modelled"
    #: The carried part was forgotten.
    DETACHED = "detached"
    #: The arm was asked to forget the part and its planner did not confirm.
    DETACH_FAILED = "detach_failed"
    #: The verb left the model as it was.
    UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class HandReport:
    """What one hand verb commanded, measured and did to the arm's model of a carried part.

    Attributes:
        verb (HandVerb): ``grasp`` or ``release``.
        outcome (HandOutcome): How it ended.
        commanded_width_mm (float | None): The width commanded, millimetres; ``None`` where nothing was (default: None).
        reported_width_mm (float | None): The width the hand reported, millimetres; ``None`` where it reports none
            (default: None).
        width_measured (bool): Whether the reported width is a measurement (default: False).
        hold (HoldEvidence): What the hand measured about a hold: ``HELD``, ``EMPTY`` or ``UNMEASURED``
            (default: UNMEASURED).
        payload (PayloadState): What the verb did to the carried part model: ``ATTACHED``, ``FILTER_ONLY``,
            ``NOT_MODELLED``, ``DETACHED``, ``DETACH_FAILED`` or ``UNCHANGED`` (default: UNCHANGED).
        payload_reason (str): Why the part is not modelled, where it is not (default: "").
        error (str): What the hand raised, for ``GRIPPER_FAULT`` (default: "").
        no_sensor (bool): The hand has no sensor at all (it toggles): nothing about the width, the hold or the release
            was checked (default: False).
        note (str): What the hand said about the command, such as that the jaws already stood there and nothing was sent
            (default: "").
    """

    verb: HandVerb
    outcome: HandOutcome
    commanded_width_mm: float | None = None
    reported_width_mm: float | None = None
    width_measured: bool = False
    hold: HoldEvidence = HoldEvidence.UNMEASURED
    payload: PayloadState = PayloadState.UNCHANGED
    payload_reason: str = ""
    error: str = ""
    #: The hand has no sensor at all (it toggles): nothing about the width, the hold or the release was checked.
    no_sensor: bool = False
    #: What the hand said about the command, such as that the jaws already stood there and nothing was sent.
    note: str = ""

    @property
    def ok(self) -> bool:
        """Whether the verb did what it was asked: the jaws closed on something or opened off it."""
        return self.outcome in (HandOutcome.GRASPED, HandOutcome.RELEASED)

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The hand verb as a person reads it.

        Returns:
            str: ASCII, no trailing newline. ``print(report)`` shows the same.
        """
        head = f"{self.verb.value}  {self.outcome.value.upper()}"
        if self.outcome is HandOutcome.REFUSED:
            return f"{head}  nothing commanded: {self.error}"
        parts = []
        if self.commanded_width_mm is not None:
            parts.append(f"width {self.commanded_width_mm:.2f} mm commanded")
        if self.reported_width_mm is not None:
            read = "measured" if self.width_measured else "reported, width not measured"
            parts.append(f"{self.reported_width_mm:.2f} mm {read}")
        if self.note:
            parts.append(self.note)
        if self.no_sensor:
            hold = "release not checked (no sensor)" if self.verb is HandVerb.RELEASE else "hold not checked (no sensor)"
        elif self.hold is HoldEvidence.UNMEASURED:
            hold = "hold not measured" + (", release not checked" if self.verb is HandVerb.RELEASE else "")
        else:
            hold = f"hold {self.hold.value.upper()}"
        parts.append(hold)
        payload = f"payload {self.payload.value.upper()}"
        if self.payload_reason:
            payload += f": {self.payload_reason}"
        parts.append(payload)
        lines = [f"{head}  " + "; ".join(parts)]
        if self.error:
            lines.append(f"  fault: {self.error}")
        if self.payload in (PayloadState.ATTACHED, PayloadState.FILTER_ONLY):
            lines.append("  the carried part is a declared envelope (safety.planning_world.payload), and a second held "
                         "part is not modelled")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The hand verb as plain data.

        Returns:
            dict[str, Any]: ``json.dumps`` safe; the enums as their values.
        """
        return {
            "verb": self.verb.value,
            "outcome": self.outcome.value,
            "ok": self.ok,
            "commanded_width_mm": self.commanded_width_mm,
            "reported_width_mm": self.reported_width_mm,
            "width_measured": self.width_measured,
            "hold": self.hold.value,
            "payload": self.payload.value,
            "payload_reason": self.payload_reason,
            "error": self.error,
            "no_sensor": self.no_sensor,
            "note": self.note,
        }


def grasp(robot: Any, width_mm: float) -> HandReport:
    """Close ``robot``'s gripper to ``width_mm`` and hand what it holds to the arm's carried part model."""
    return _said(_grasp(robot, width_mm))


def release(robot: Any) -> HandReport:
    """Open ``robot``'s gripper to the hand's width and forget the carried part unless one is still measured."""
    return _said(_release(robot))


def _said(report: HandReport) -> HandReport:
    """``report``, a WARNING with what it says in the robot log first where the verb failed, on one line."""
    if not report.ok:
        _LOG.warning("%s", report.render().replace("\n", " |"))
    return report


def _grasp(robot: Any, width_mm: float) -> HandReport:
    """:func:`grasp`, saying nothing: a pick's own close is said with the pick."""
    refused = _refusal(robot, HandVerb.GRASP)
    if refused is not None:
        return refused
    gripper = robot.gripper
    commanded = float(width_mm)
    why = "" if isinstance(gripper, OpensAndCloses) else _width_refusal(gripper, commanded, close=True, what="a grasp at")
    if why:
        return HandReport(verb=HandVerb.GRASP, outcome=HandOutcome.REFUSED, error=why)
    toggle = toggle_without_sensor_of(gripper)
    note = _already(toggle, closed=True)
    try:
        _command(gripper, commanded, close=True)
        reported = float(gripper.get_width_mm())
        measured = width_is_measured_of(gripper)
        hold = hold_evidence_of(gripper)
    except Exception as exc:  # noqa: BLE001 (a gripper fault is the report, after the jaws are stopped)
        return _fault(gripper, HandVerb.GRASP, commanded, exc)
    if hold is HoldEvidence.EMPTY:
        return HandReport(verb=HandVerb.GRASP, outcome=HandOutcome.NOTHING_HELD, commanded_width_mm=commanded,
                          reported_width_mm=reported, width_measured=measured, hold=hold)
    # A toggle's part is carried at the width the caller named: the width it reports is a band, not the part.
    state, reason = _attach(robot.arm, reported if measured else commanded)
    if toggle is not None:
        return HandReport(verb=HandVerb.GRASP, outcome=HandOutcome.GRASPED, hold=hold, payload=state,
                          payload_reason=reason, no_sensor=True, note=note)
    return HandReport(verb=HandVerb.GRASP, outcome=HandOutcome.GRASPED, commanded_width_mm=commanded,
                      reported_width_mm=reported, width_measured=measured, hold=hold, payload=state,
                      payload_reason=reason)


def _release(robot: Any, *, meanwhile: "Callable[[], None] | None" = None) -> HandReport:
    """:func:`release`, saying nothing: a place's own release is said with the place.

    ``meanwhile`` runs while the jaws' stroke is waited out, where the hand leaves its stroke to the verb
    (``wait_settled``): the place judges its line out there (``robot.motion.judge_next_leg``). The one change has gone
    out, the hand has been read and the part forgotten, and nothing is sent until the stroke is over. Where the hand
    leaves nothing, ``meanwhile`` does not run, and the release is the one it always was.
    """
    refused = _refusal(robot, HandVerb.RELEASE)
    if refused is not None:
        return refused
    gripper = robot.gripper
    commanded = float(gripper.max_width_mm)
    why = "" if isinstance(gripper, OpensAndCloses) else _width_refusal(gripper, commanded, close=False,
                                                                       what="a release to")
    if why:
        return HandReport(verb=HandVerb.RELEASE, outcome=HandOutcome.REFUSED, error=why)
    toggle = toggle_without_sensor_of(gripper)
    note = _already(toggle, closed=False)
    try:
        settles_at = _command(gripper, commanded, close=False, leave_the_stroke=meanwhile is not None)
        reported = float(gripper.get_width_mm())
        measured = width_is_measured_of(gripper)
        hold = hold_evidence_of(gripper)
    except Exception as exc:  # noqa: BLE001 (a gripper fault is the report, after the jaws are stopped)
        return _fault(gripper, HandVerb.RELEASE, commanded, exc)
    try:
        if hold is HoldEvidence.HELD:
            return HandReport(verb=HandVerb.RELEASE, outcome=HandOutcome.RELEASE_NOT_CONFIRMED,
                              commanded_width_mm=commanded, reported_width_mm=reported, width_measured=measured,
                              hold=hold, payload_reason="the gripper still measures a part, so its model is kept")
        state, reason = _detach(robot.arm)
        if settles_at is not None and meanwhile is not None:
            _while_the_jaws_travel(meanwhile)
    finally:
        _settled(gripper, settles_at)
    if toggle is not None:
        return HandReport(verb=HandVerb.RELEASE, outcome=HandOutcome.RELEASED, hold=hold, payload=state,
                          payload_reason=reason, no_sensor=True, note=note)
    return HandReport(verb=HandVerb.RELEASE, outcome=HandOutcome.RELEASED, commanded_width_mm=commanded,
                      reported_width_mm=reported, width_measured=measured, hold=hold, payload=state,
                      payload_reason=reason)


def is_holding(robot: Any) -> HoldEvidence:
    """What ``robot``'s gripper measured about a hold, commanding nothing; UNMEASURED with no open gripper."""
    gripper = getattr(robot, "gripper", None)
    if gripper is None or not bool(getattr(gripper, "is_connected", False)):
        return HoldEvidence.UNMEASURED
    return hold_evidence_of(gripper)


def _refusal(robot: Any, verb: HandVerb) -> HandReport | None:
    gripper = getattr(robot, "gripper", None)
    why = ""
    if gripper is None:
        why = "this robot has no gripper"
    elif bool(getattr(gripper, "holds_nothing", False)):
        why = f"the gripper in hand ({type(gripper).__name__}) holds nothing: no end-effector was built"
    elif not bool(getattr(robot.arm, "is_connected", False)):
        why = "the arm is not connected; call the hand verbs inside robot.connected()"
    elif not bool(getattr(gripper, "is_connected", False)):
        why = "the gripper is not connected; call the hand verbs inside robot.connected()"
    else:
        why = _controller_refusal(robot.arm)
    return None if not why else HandReport(verb=verb, outcome=HandOutcome.REFUSED, error=why)


def _controller_refusal(arm: Any) -> str:
    """Why ``arm``'s controller cannot act now, or ``""`` where it can, or where the arm does not say.

    Asked before every gripper command, because a hand's I/O still switches while the arm is stopped: after a protective
    stop mid-pick a toggle hand's pre-open pulsed and dropped the part from wherever the arm stood, before anything asked
    the controller (owner-cell audit, reproduced with fakes, 2026-09-23). Only an arm that implements
    ``SupportsRobotStatus`` (the UR driver) is asked; every other arm passes. The criterion is ``not is_operational``,
    the one the pick loop and ``GraspExecutionPolicy`` use, so REDUCED safety mode is refused too. A read that raises
    refuses as well: a verb promises a report, and a controller that cannot be asked cannot be said to move.

    The halt latch is asked first, of any arm that carries one (``SupportsHalt``, the dummy included), and of the status
    (``RobotStatus.halted``): a halted arm is refused with the halt's own sentence. A halt is not a controller stop, so
    it never reads "clear the stop": a person confirms the cell is clear, then Restart. Where the controller is stopped
    as well, the sentence says so beside the halt.
    """
    halted = halt_state_of(arm)
    if not isinstance(arm, SupportsRobotStatus):
        return _halted_sentence(halted.reason, None) if halted is not None else ""
    try:
        status = arm.get_robot_status()
        # Read as strictly as the latch: words, or not halted. A double that answers every attribute, and a status
        # from before the halt with no such field, are not halted.
        said = getattr(status, "halted", "")
    except Exception as exc:  # noqa: BLE001 (the verb reports, and nothing is commanded)
        if halted is not None:
            return _halted_sentence(halted.reason, None)
        return (f"the controller's state could not be read ({type(exc).__name__}: {exc}), so nothing was commanded, "
                "the jaws included")
    reason = halted.reason if halted is not None else said if isinstance(said, str) else ""
    if reason:
        stopped = isinstance(status, RobotStatus) and not status.controller_operational
        return _halted_sentence(reason, status if stopped else None)
    if status.is_operational:
        return ""
    detail = f": {status.message}" if status.message else ""
    return (f"the controller cannot move (robot_mode={status.robot_mode.value}, safety_mode={status.safety_mode.value}, "
            f"protective_stop={status.protective_stopped}, emergency_stop={status.emergency_stopped}{detail}), so "
            "nothing was commanded, the jaws included; clear the stop where the arm is visible, then run again")


def _halted_sentence(reason: str, stopped: "RobotStatus | None") -> str:
    """The refusal of a halted arm; ``stopped`` is the controller's status where it cannot move either."""
    sentence = halted_refusal(reason, "nothing was commanded, the jaws included")
    if stopped is None:
        return sentence
    detail = f": {stopped.message}" if stopped.message else ""
    return (f"{sentence}. The controller cannot move either (robot_mode={stopped.robot_mode.value}, "
            f"safety_mode={stopped.safety_mode.value}, protective_stop={stopped.protective_stopped}, "
            f"emergency_stop={stopped.emergency_stopped}{detail}): a protective stop is released at the teach pendant, "
            "where the arm is visible, before that")


def _command(gripper: Any, width_mm: float, *, close: bool, leave_the_stroke: bool = False) -> float | None:
    """Command the gripper for a verb that knows whether it closes: by intent where the gripper takes it, else by width.

    A gripper that takes open and close as what they are (``OpensAndCloses``) is told which, and ``width_mm`` is not
    sent: read against ``closed_below_mm`` it could turn the verb round (owner's toggle cell, 2026-09-23). Any other
    gripper gets the width, which :func:`_width_refusal` has already been asked about.

    ``leave_the_stroke`` asks a hand that can leave its stroke to the verb (``wait_settled``, a ``jaw_io`` hand that waits
    on no switch) to send its one change and come back: the moment the stroke is over is returned, ``time.monotonic()``
    seconds, for :func:`_settled`. ``None`` where the command waited out its stroke itself, as every other command does.
    """
    if isinstance(gripper, OpensAndCloses):
        if leave_the_stroke and callable(getattr(gripper, "wait_settled", None)):
            settles_at = cast("Any", gripper).set_closed(close, wait=False)  # the hand's own keyword (``jaw_io``)
            return float(settles_at) if isinstance(settles_at, (int, float)) and not isinstance(settles_at, bool) \
                else None
        gripper.set_closed(close)
    else:
        gripper.set_width_mm(width_mm)
    return None


def _settled(gripper: Any, settles_at: float | None) -> None:
    """Wait out the stroke ``gripper`` left to the verb (:func:`_command`), until ``settles_at``; nothing where none."""
    if settles_at is not None:
        gripper.wait_settled(settles_at)


def _while_the_jaws_travel(meanwhile: "Callable[[], None]") -> None:
    """Run what a verb judges while the jaws' stroke is waited out. It sends nothing; a judgement ahead that raised is
    said in the log and kept nowhere, and the leg it was about is judged as it runs."""
    try:
        meanwhile()
    except Exception as exc:  # noqa: BLE001 (a judgement ahead never ends the verb: its leg is judged as it runs)
        _LOG.warning("what was judged while the jaws travelled raised (%s: %s): the next leg is judged as it runs",
                     type(exc).__name__, exc)


def _judges_next_legs(arm: Any) -> bool:
    """Whether ``arm`` judges the next leg while the jaws' stroke is waited out (``robot.motion.judge_next_leg``), read
    as ``is True``, so a double that answers every attribute does not."""
    return getattr(arm, "judges_next_legs", False) is True


def _the_next_leg_judged(arm: Any, after: Pose, *, junction: str, now: bool) -> None:
    """Have ``arm`` judge the joint move declared after this verb from where the line to ``after`` ends
    (``judge_the_next_leg``): ``now``, while the jaws' stroke is waited out, else during that line where the arm judges
    legs in motion. On an arm that judges no next leg, nothing; it sends nothing either way."""
    judge = getattr(arm, "judge_the_next_leg", None)
    if _judges_next_legs(arm) and callable(judge):
        judge(after, junction=junction, now=now)


def _width_refusal(gripper: Any, width_mm: float, *, close: bool, what: str) -> str:
    """Why ``width_mm`` would do the opposite of what the verb means on a gripper with two states, or ``""``.

    A ``TwoStateGripper`` closes at or below its ``closed_below_mm`` and opens above it, whatever the verb that sent
    the width meant: with the default of 5 mm, a pick of a 40 mm part grasps at 39 mm and opens the jaws, and with
    no sensor it reports EXECUTED (found diagnosing the owner's toggle cell, 2026-09-23). ``what`` names the command
    in the sentence, for example "a grasp at".
    """
    if not isinstance(gripper, TwoStateGripper):
        return ""
    below = float(gripper.closed_below_mm)
    if (width_mm <= below) is close:
        return ""
    if close:
        return (f"the gripper has two states and closes only at or below its closed_below_mm, {below} mm, so {what} "
                f"{width_mm} mm would open it: set robot.gripper.jaw_io.closed_below_mm at or above {width_mm} and "
                f"below max_width_mm")
    return (f"the gripper has two states and closes at or below its closed_below_mm, {below} mm, so {what} "
            f"{width_mm} mm would close it: set robot.gripper.jaw_io.closed_below_mm below {width_mm}")


def _already(toggle: Any, *, closed: bool) -> str:
    """For a hand that toggles, where its count says the jaws already stand as asked: no change goes out, said so."""
    if toggle is None or bool(toggle.jaws_closed) is not closed:
        return ""
    return f"already {'closed' if closed else 'open'}: no change"


def _fault(gripper: Any, verb: HandVerb, commanded: float | None, exc: BaseException) -> HandReport:
    if isinstance(gripper, StoppableGripper):
        try:
            gripper.stop()
        except Exception as stop_exc:  # noqa: BLE001 (a failed halt must not hide the fault it follows)
            exc = RuntimeError(f"{exc}; the stop that followed failed too: {stop_exc}")
    return HandReport(verb=verb, outcome=HandOutcome.GRIPPER_FAULT, commanded_width_mm=commanded,
                      error=f"{type(exc).__name__}: {exc}")


def _attach(arm: Any, width_mm: float) -> tuple[PayloadState, str]:
    if not isinstance(arm, CarriesPayload):
        return PayloadState.NOT_MODELLED, f"{type(arm).__name__} models no carried part"
    declined = arm.payload_declined_reason()
    if declined is not None:
        return PayloadState.NOT_MODELLED, declined
    arm.attach_payload(width_mm)
    model = arm.payload_model()
    if model is PayloadModel.PLANNER_AND_FILTER:
        return PayloadState.ATTACHED, ""
    if model is PayloadModel.FILTER_ONLY:
        return PayloadState.FILTER_ONLY, ("the planner declined the part and routes as if the gripper were empty; the "
                                          "self filter takes it out of what the cameras see")
    return PayloadState.NOT_MODELLED, "the attach modelled nothing"


def _detach(arm: Any) -> tuple[PayloadState, str]:
    if not isinstance(arm, CarriesPayload):
        return PayloadState.NOT_MODELLED, f"{type(arm).__name__} models no carried part"
    if arm.detach_payload():
        return PayloadState.DETACHED, ""
    return PayloadState.DETACH_FAILED, "the planner did not confirm it forgot the carried part"


# ---------------------------------------------------------------------------------------------------
# pick and place
# ---------------------------------------------------------------------------------------------------


class HandlingVerb(StrEnum):
    """Which handling verb a report is about."""

    PICK = "pick"
    PLACE = "place"


class HandlingOutcome(StrEnum):
    """How a pick or a place (``Robot.pick``, ``Robot.place``) ended.

    Attributes:
        EXECUTED: Every motion ran and the hand verb did what it was asked.
        MOTION_REFUSED: A motion was refused, raised or timed out at the steady gate; nothing was commanded after it.
        NOTHING_HELD: The pick closed on nothing measured, opened again and backed out to the standoff.
        RELEASE_NOT_CONFIRMED: The place opened and the hand still measures a part; the arm stays where it stands.
        GRIPPER_FAULT: The hand raised; it was stopped where it can be, and nothing was commanded after it.
        CAMERA_WORLD_UNAVAILABLE: A camera could not vouch for the cell; nothing was commanded after it, and a caller
            stops rather than trying again.
        CARRIED_RETREAT_REFUSED: At the part, before the jaws closed, the line out judged as if they held the part would
            be refused (or could not be judged): the jaws stayed open, and the arm went back up to the standoff.
        REFUSED: Nothing was commanded.
    """

    #: Every motion ran and the hand verb did what it was asked.
    EXECUTED = "executed"
    #: A motion was refused, raised or timed out at the steady gate; nothing was commanded after it.
    MOTION_REFUSED = "motion_refused"
    #: The pick closed on nothing measured, opened again and backed out to the standoff.
    NOTHING_HELD = "nothing_held"
    #: The place opened and the gripper still measures a part; the arm stays where it stands.
    RELEASE_NOT_CONFIRMED = "release_not_confirmed"
    #: The gripper raised; it was stopped where it can be, and nothing was commanded after it.
    GRIPPER_FAULT = "gripper_fault"
    #: A camera could not vouch for the cell through its fresh-frame attempts; nothing was commanded after it, and a
    #: caller stops rather than trying again. The verb reports it rather than raising it.
    CAMERA_WORLD_UNAVAILABLE = "camera_world_unavailable"
    #: At the part, before the jaws closed, the arm judged the line out as if they held the part and would refuse it,
    #: or could not judge it: the jaws stayed open, and the arm went back up the line it came down, to the standoff.
    CARRIED_RETREAT_REFUSED = "carried_retreat_refused"
    #: Nothing was commanded.
    REFUSED = "refused"


#: How each line motion reads in a report.
_LINE_WORDS = {
    LineMotion.CHECKED: "a line judged sample by sample before it runs",
    LineMotion.CONTROLLER_LINE: "a line the controller draws, judged at its end only",
    LineMotion.TELEPORT: "no controller: the pose is set",
    LineMotion.NOT_KEPT: "not kept",
}


@dataclass(frozen=True, slots=True)
class HandlingReport:
    """What one pick or place commanded, what stood behind each motion, and what the hand did.

    Attributes:
        verb (HandlingVerb): ``pick`` or ``place``.
        outcome (HandlingOutcome): How it ended.
        poses (tuple[Pose, ...]): Every pose a motion of the verb went to, in order (default: ()).
        statuses (tuple[MotionStatus, ...]): Each motion's status, in the same order (default: ()).
        camera_worlds (tuple[CameraWorldStamp, ...]): What camera world stood behind each motion (default: ()).
        line (LineReading | None): What the arm keeps of the line motions; ``None`` where there was none
            (default: None).
        keep_out_held (bool): Whether the ``keep_out`` was held out of the arm's world through the motions
            (default: False).
        hand (HandReport | None): The hand verb's report; ``None`` where the hand was not reached (default: None).
        message (str): Why it ended where it did not succeed (default: "").
        another_candidate_may_follow (bool): Whether a program may try another candidate after this pick: true only when
            a guard or the planner refused its first motion with nothing sent, or after ``CARRIED_RETREAT_REFUSED`` with
            the empty hand backed out open; always false for a place. A program that does looks around again first
            (default: False).
    """

    verb: HandlingVerb
    outcome: HandlingOutcome
    poses: tuple[Pose, ...] = ()
    statuses: tuple[MotionStatus, ...] = ()
    camera_worlds: tuple[CameraWorldStamp, ...] = ()
    line: LineReading | None = None
    keep_out_held: bool = False
    hand: HandReport | None = None
    message: str = ""
    #: Whether a program may try another candidate after this pick (the cell-fix plan's contract 4): true only after a
    #: guard or the planner refused the pick's first motion, nothing sent and nothing commanded to the jaws (a hand told
    #: to pre-open was commanded), or after ``CARRIED_RETREAT_REFUSED`` with the empty hand backed out open. A program
    #: that does looks around again first. Always false for a place.
    another_candidate_may_follow: bool = False

    @property
    def ok(self) -> bool:
        """Whether every motion ran and the hand verb did what it was asked."""
        return self.outcome is HandlingOutcome.EXECUTED

    @property
    def weakest_camera_world(self) -> CameraWorldStamp | None:
        """The stamp a reader of the whole verb has to see: the first that does not vouch, else the last."""
        return weakest_camera_world(self.camera_worlds)

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The pick or place as a person reads it.

        Returns:
            str: ASCII, no trailing newline. ``print(report)`` shows the same.
        """
        head = f"{self.verb.value}  {self.outcome.value.upper()}  {len(self.poses)} motion(s) commanded"
        lines = [head + (f": {self.message}" if self.message else "")]
        if self.line is not None:
            words = _LINE_WORDS[self.line.motion]
            lines.append(f"  line  {self.line.motion.value.upper()}: {words}; {self.line.reason}")
        weakest = self.weakest_camera_world
        if weakest is not None:
            lines.append(f"  {weakest.render()}")
        if self.keep_out_held:
            lines.append("  the target was held out of the planner world through every motion")
        if self.outcome is HandlingOutcome.CAMERA_WORLD_UNAVAILABLE:
            lines.append("  a camera could not vouch for the cell: look at the camera rather than trying again")
        if self.another_candidate_may_follow:
            lines.append("  another candidate may follow: the jaws stand open and the part was not taken; look around "
                         "again first")
        if self.hand is not None:
            lines.extend(f"  {line}" for line in self.hand.render().split("\n"))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The pick or place as plain data.

        Returns:
            dict[str, Any]: ``json.dumps`` safe; the poses as ``position_mm`` and ``quaternion_xyzw``.
        """
        weakest = self.weakest_camera_world
        return {
            "verb": self.verb.value,
            "outcome": self.outcome.value,
            "ok": self.ok,
            "poses_mm": [[float(v) for v in pose.position_mm] for pose in self.poses],
            "statuses": [status.value for status in self.statuses],
            "camera_worlds": [stamp.to_dict() for stamp in self.camera_worlds],
            "weakest_camera_world": None if weakest is None else weakest.to_dict(),
            "line": None if self.line is None else {"motion": self.line.motion.value, "reason": self.line.reason},
            "keep_out_held": self.keep_out_held,
            "hand": None if self.hand is None else self.hand.to_dict(),
            "message": self.message,
            "another_candidate_may_follow": self.another_candidate_may_follow,
        }


def pick(
    robot: Any,
    pose: Pose,
    width_mm: float,
    *,
    standoff_mm: float = 80.0,
    squeeze_mm: float = 1.0,
    pre_open_mm: "Maybe[float | None]" = UNSET,
    camera_world: Maybe[CameraWorldDecline] = UNSET,
    keep_out: Maybe[SegmentationOffer] = UNSET,
) -> HandlingReport:
    """Pick the part at ``pose``, ``width_mm`` across: standoff, a line in, grasp, a line out (straight up for a grasp
    more than :data:`LIFT_STRAIGHT_UP_BEYOND_DEG` off vertical)."""
    return _said_handling(_Handling(robot, HandlingVerb.PICK, pose, standoff_mm, camera_world).pick(
        width_mm=float(width_mm), squeeze_mm=float(squeeze_mm), pre_open_mm=pre_open_mm, keep_out=keep_out))


def place(
    robot: Any,
    pose: Pose,
    *,
    standoff_mm: float = 80.0,
    camera_world: Maybe[CameraWorldDecline] = UNSET,
    keep_out: Maybe[SegmentationOffer] = UNSET,
    instead: Maybe[Pose] = UNSET,
) -> HandlingReport:
    """Place the held part at ``pose``: standoff, a line in, release, a line out unless the release is not confirmed.

    ``instead`` is where the part is let go where the guard or the planner refuses the line in to ``pose`` before
    anything was sent: the drop over a box's rim, for a part set down below it (``robot.place.release_in_a_box``), from
    the same standoff. The refused line is no motion of the place; the line to ``instead`` is judged as every line is."""
    return _said_handling(_Handling(robot, HandlingVerb.PLACE, pose, standoff_mm, camera_world).place(
        keep_out=keep_out, instead=instead))


def _said_handling(report: HandlingReport) -> HandlingReport:
    """``report``, a WARNING with its message in the robot log first where the pick or place failed."""
    if not report.ok:
        _LOG.warning("%s %s after %d motion(s): %s%s", report.verb.value, report.outcome.value, len(report.poses),
                     report.message or "no reason was given",
                     "; another candidate may follow" if report.another_candidate_may_follow else "")
    return report


class _Refused(Exception):
    """A step ended the verb; carries the report."""

    def __init__(self, report: HandlingReport) -> None:
        super().__init__(report.message)
        self.report = report


class _Handling:
    """One pick or place: the refusals, then the motions, recorded as they are commanded."""

    def __init__(self, robot: Any, verb: HandlingVerb, pose: Pose, standoff_mm: float,
                 camera_world: Maybe[CameraWorldDecline]) -> None:
        self.robot = robot
        self.arm = robot.arm
        self.gripper = robot.gripper
        self.verb = verb
        self.pose = pose
        self.standoff_mm = float(standoff_mm)
        self.camera_world = camera_world
        self.poses: list[Pose] = []
        self.statuses: list[MotionStatus] = []
        self.stamps: list[CameraWorldStamp] = []
        self.line: LineReading | None = None
        self.keep_out_held = False
        #: Whether anything was commanded to the jaws: the pre-open, the close, the open after an empty close.
        self.jaws_commanded = False
        #: Whether the place judged ahead while its release's stroke was waited out (:meth:`_while_the_jaws_open`).
        self.judged_while_the_jaws_opened = False
        #: Where a place lets its part go where the line in to ``pose`` is refused before anything was sent
        #: (:meth:`_line_in`); ``None`` for none.
        self.instead: Pose | None = None

    # ---- the verbs --------------------------------------------------------------------------

    def pick(self, *, width_mm: float, squeeze_mm: float, pre_open_mm: "Maybe[float | None]",
             keep_out: Maybe[SegmentationOffer]) -> HandlingReport:
        refused = self._refusal()
        if refused is not None:
            return refused
        scope: AbstractContextManager[Any] = keeping_out(self.arm, keep_out) if chosen(keep_out) else nullcontext()
        try:
            with scope as held:
                self.keep_out_held = bool(getattr(held, "world_wired", False))
                return self._pick_body(width_mm=width_mm, squeeze_mm=squeeze_mm, pre_open_mm=pre_open_mm)
        except _Refused as ended:
            return ended.report

    def place(self, *, keep_out: Maybe[SegmentationOffer], instead: Maybe[Pose] = UNSET) -> HandlingReport:
        refused = self._refusal()
        if refused is not None:
            return refused
        self.instead = instead if chosen(instead) else None
        if self.instead is not None and self.instead.frame is not Frame.BASE:
            return self._report(HandlingOutcome.REFUSED, message=(
                f"the drop over the rim is in {self.instead.frame.value!r}, and a place takes a pose in BASE"))
        # Asked before the arm moves, not when release() asks it at the tray.
        why = "" if isinstance(self.gripper, OpensAndCloses) else _width_refusal(
            self.gripper, float(self.gripper.max_width_mm), close=False, what="the place's release to")
        if why:
            return self._report(HandlingOutcome.REFUSED, message=why)
        # What the part is set down on is held out of the world from here, after every refusal and before the first
        # motion, to after the line out: the part comes down to within the line clearance of it, and a world that
        # still held it would refuse the line in. The offer is forgotten on every exit, a refused motion included.
        scope: AbstractContextManager[Any] = keeping_out(self.arm, keep_out) if chosen(keep_out) else nullcontext()
        try:
            with scope as held:
                self.keep_out_held = bool(getattr(held, "world_wired", False))
                return self._place_body()
        except _Refused as ended:
            return ended.report

    def _place_body(self) -> HandlingReport:
        standoff = self._standoff()
        self._move(standoff, linear=False)
        # The owner's switches (safety.planning_world.hold, 2026-10-09): the line in judged in the world its route was
        # judged in, with no frame at the standoff, and the line out in the world the line in was judged in, with no
        # frame after the release. Each is a block that changes nothing where its switch is off.
        with self._held_at("standoff", "the line in to a place, judged in the world its route to the standoff was "
                                       "judged in"):
            self._line_in()
        with self._held_at("drop", "the release and the line out of a place, judged in the world the line in was "
                                   "judged in"):
            self.jaws_commanded = True
            hand = _release(self.robot, meanwhile=self._while_the_jaws_open(standoff))
            if hand.outcome is HandOutcome.GRIPPER_FAULT:
                return self._report(HandlingOutcome.GRIPPER_FAULT, hand=hand, message=hand.error)
            if hand.outcome is HandOutcome.RELEASE_NOT_CONFIRMED:
                return self._report(HandlingOutcome.RELEASE_NOT_CONFIRMED, hand=hand, message=(
                    "the gripper still measures a part after opening, so the arm stays where it stands"))
            if hand.outcome is not HandOutcome.RELEASED:
                return self._report(HandlingOutcome.REFUSED, hand=hand, message=hand.error)
            if not self.judged_while_the_jaws_opened:
                # No stroke was left to the verb: the joint move declared after the place is judged during the line out,
                # where the arm judges legs in motion (robot.motion.judge_next_leg in_settles_and_motion).
                _the_next_leg_judged(self.arm, standoff, junction="drop", now=False)
            self._move(standoff, linear=True)
            return self._report(HandlingOutcome.EXECUTED, hand=hand)

    def _line_in(self) -> None:
        """The place's line in, to ``pose``. Where the place carries a drop over a box's rim (``instead``) and the guard or
        the planner refuses that line before anything was sent, the part goes over the rim instead: the line from the
        standoff to ``instead``, judged as every line is (the owner, 2026-10-08 night: where the opening is too narrow,
        today's drop over the rim). The refused line is no motion of the place, as nothing was sent and the arm stands at
        the standoff; any other end of it ends the verb as before."""
        try:
            self._move(self.pose, linear=True)
        except _Refused as refused:
            if self.instead is None or not self._refused_before_sending(refused.report.message):
                raise
            _LOG.info("the line into the box, to %s, was refused before anything was sent (%s): the part is let go over "
                      "the rim instead", self.pose.label or "the drop below its rim", refused.report.message)
            del self.poses[-1], self.statuses[-1], self.stamps[-1]
            self._move(self.instead, linear=True)

    def _refused_before_sending(self, message: str) -> bool:
        """Whether the motion just recorded was refused by a guard or the planner before anything was sent: a refusal
        the arm answered with a status (:data:`_REFUSED_BEFORE_SENDING`, or the planner's own TIMEOUT sentence)."""
        if not self.statuses or len(self.statuses) != len(self.poses):
            return False   # the motion raised, and says nothing of what was sent
        status = self.statuses[-1]
        if status is MotionStatus.TIMEOUT:
            return message == NO_PLAN_FAIL_SAFE_MESSAGE
        return status in _REFUSED_BEFORE_SENDING

    def _held_at(self, junction: str, reason: str) -> AbstractContextManager[Any]:
        """The block in which the arm holds its world across ``junction`` (``safety.planning_world.hold``,
        ``holds_world_at``), where it does and the verb carries no decline; elsewhere a block that changes nothing."""
        holds = getattr(self.arm, "holds_world_at", None)
        if chosen(self.camera_world) or not callable(holds) or holds(junction) is not True:
            return nullcontext()
        return self.arm.held_world(reason)

    def _while_the_jaws_open(self, standoff: Pose) -> "Callable[[], None] | None":
        """What the place judges while the jaws open, where the arm judges the next leg (``robot.motion.judge_next_leg``)
        and the verb carries no decline; ``None`` elsewhere, and the release waits out its stroke as ever.

        In the world held across the drop (``safety.planning_world.hold.drop``): the line out to ``standoff``, from where
        the arm stands, as it will run (``judge_line_ahead``), and where it would run, the joint move a task declared
        after the place from where the line out ends (``judge_the_next_leg``). Outside that world nothing is judged ahead:
        the line out takes its frame after the release, as ever. Nothing is sent before the stroke is over.
        """
        if not _judges_next_legs(self.arm) or chosen(self.camera_world):
            return None
        arm = self.arm

        def meanwhile() -> None:
            self.judged_while_the_jaws_opened = True
            holds = getattr(arm, "holds_world_at", None)
            judge_line = getattr(arm, "judge_line_ahead", None)
            if not callable(holds) or holds("drop") is not True or not callable(judge_line):
                return
            if judge_line(standoff) is None:
                _the_next_leg_judged(arm, standoff, junction="drop", now=True)

        return meanwhile

    def _pick_body(self, *, width_mm: float, squeeze_mm: float, pre_open_mm: "Maybe[float | None]") -> HandlingReport:
        opening = float(self.gripper.max_width_mm) if not chosen(pre_open_mm) else pre_open_mm
        close_to = max(float(self.gripper.min_width_mm), width_mm - squeeze_mm)
        # Both widths are asked what they mean before the arm moves, not at the part. A gripper told open and close
        # by intent has nothing to ask: no width reaches it.
        why = "" if isinstance(self.gripper, OpensAndCloses) else (
            (_width_refusal(self.gripper, float(opening), close=False, what="the pick's pre-open to")
             if opening is not None else "")
            or _width_refusal(self.gripper, close_to, close=True, what="the pick's grasp at"))
        if why:
            return self._report(HandlingOutcome.REFUSED, message=why)
        toggle = toggle_without_sensor_of(self.gripper)
        if toggle is not None:
            # Never a change before the arm moves (owner's decision, 2026-09-24): every change moves a toggle's jaws, and
            # one on a wrong count closed the owner's jaws before the part. The hand is asked instead, and it asks a
            # person where it believes its jaws closed; a pick nobody can vouch for ends here, before any motion.
            try:
                why = toggle.jaws_open_for_a_pick()
            except Exception as exc:  # noqa: BLE001 (a gripper fault is the report, after the jaws are stopped)
                hand = _fault(self.gripper, HandVerb.GRASP, None, exc)
                return self._report(HandlingOutcome.GRIPPER_FAULT, hand=hand, message=hand.error)
            if why:
                return self._report(HandlingOutcome.REFUSED, message=why)
            opening = None
        detach = getattr(self.arm, "detach_payload", None)
        if callable(detach):
            detach()
        if opening is not None:
            self._command_gripper(float(opening), close=False)
        standoff = self._standoff()
        lift = self._lift(standoff)
        self._move(standoff, linear=False)
        # The owner's O2 (safety.planning_world.hold.standoff): the line down judged in the world its route was judged in,
        # with no frame at the standoff; a block that changes nothing where the switch is off.
        with self._held_at("standoff", "the line down to a pick, judged in the world its route to the standoff was "
                                       "judged in"):
            self._move(self.pose, linear=True)
        why = self._line_out_refusal(lift, self._line_out_width(close_to, width_mm))
        if why:
            # The jaws stay open, and the arm goes back up the line it came down (the owner, 2026-10-01).
            said = f"{why.rstrip('.')}. The jaws stayed open"
            try:
                self._move(standoff, linear=True)
            except _Refused as backing:
                raise _Refused(replace(backing.report, message=(
                    f"{said}; the line back up to the standoff failed too: {backing.report.message}"))) from None
            return self._report(HandlingOutcome.CARRIED_RETREAT_REFUSED, message=(
                f"{said}, and the arm went back up the line it came down, to the standoff, empty-handed"))
        self.jaws_commanded = True
        hand = _grasp(self.robot, close_to)
        if hand.outcome is HandOutcome.GRIPPER_FAULT:
            return self._report(HandlingOutcome.GRIPPER_FAULT, hand=hand, message=hand.error)
        if hand.outcome is HandOutcome.NOTHING_HELD:
            # An empty hand backs out the way it came: there is nothing to lift.
            self._command_gripper(float(self.gripper.max_width_mm), close=False)
            self._move(standoff, linear=True)
            return self._report(HandlingOutcome.NOTHING_HELD, hand=hand, message=(
                "the gripper measured nothing held, so it opened and backed out to the standoff"))
        if hand.outcome is not HandOutcome.GRASPED:
            return self._report(HandlingOutcome.REFUSED, hand=hand, message=hand.error)
        self._move(lift, linear=True)
        return self._report(HandlingOutcome.EXECUTED, hand=hand)

    # ---- before any command -----------------------------------------------------------------

    def _refusal(self) -> HandlingReport | None:
        gripper = self.gripper
        if gripper is None:
            return self._report(HandlingOutcome.REFUSED, message="this robot has no gripper")
        if bool(getattr(gripper, "holds_nothing", False)):
            return self._report(HandlingOutcome.REFUSED, message=(
                f"the gripper in hand ({type(gripper).__name__}) holds nothing: no end-effector was built"))
        if not bool(getattr(self.arm, "is_connected", False)) or not bool(getattr(gripper, "is_connected", False)):
            return self._report(HandlingOutcome.REFUSED, message=(
                f"the arm or the gripper is not connected; call {self.verb.value} inside robot.connected()"))
        if self.pose.frame is not Frame.BASE:
            return self._report(HandlingOutcome.REFUSED, message=(
                f"the pose is in {self.pose.frame.value!r}, and {self.verb.value} takes a pose in BASE"))
        if isinstance(self.arm, ReadsCameraWorld):
            stamp = self.arm.camera_world_for_move(self.camera_world)
            sentence = camera_world_refusal(stamp, live_world_wired=self.arm.live_camera_world_wired())
            if sentence is not None:
                self.stamps.append(stamp)
                return self._report(HandlingOutcome.REFUSED, message=sentence)
        # Everything that moves the arm goes through cuRobo and the exact mesh guard with the camera world, so a pick
        # or a place reads the route its motions take, as Robot.move does. A desk arm runs; a UR on the ik planner is
        # refused like a KUKA.
        route = route_of(self.arm)
        self.line = route.line
        if not route.runs:
            return self._report(HandlingOutcome.REFUSED, message=route.reason)
        # Last, because it is the one question that goes to the controller, and before the detach and the pre-open:
        # the jaws of a stopped arm stay as they are, whatever they hold.
        why = _controller_refusal(self.arm)
        if why:
            return self._report(HandlingOutcome.REFUSED, message=why)
        return None

    # ---- the steps ----------------------------------------------------------------------------

    def _standoff(self) -> Pose:
        """``pose`` moved back along its own +Z, the approach, by the standoff, turned as the pose is."""
        matrix = np.asarray(self.pose.to_matrix(), dtype=np.float64)
        back = matrix[:3, 3] - matrix[:3, 2] * self.standoff_mm
        return Pose(position_mm=back, quaternion_xyzw=np.asarray(self.pose.quaternion_xyzw, dtype=np.float64),
                    frame=Frame.BASE, label="standoff")

    def _lift(self, standoff: Pose) -> Pose:
        """Where a pick's part goes after the close: ``standoff``, back along the approach, for a grasp within
        :data:`LIFT_STRAIGHT_UP_BEYOND_DEG` of vertical; else straight up, BASE +Z, by the standoff and at least
        :data:`MIN_STRAIGHT_LIFT_MM`, turned as the pose is."""
        matrix = np.asarray(self.pose.to_matrix(), dtype=np.float64)
        approach = matrix[:3, 2] / max(float(np.linalg.norm(matrix[:3, 2])), 1e-12)
        off_vertical = float(np.degrees(np.arccos(np.clip(-float(approach[2]), -1.0, 1.0))))
        if off_vertical <= LIFT_STRAIGHT_UP_BEYOND_DEG:
            return standoff
        up = matrix[:3, 3] + np.array([0.0, 0.0, max(self.standoff_mm, MIN_STRAIGHT_LIFT_MM)])
        return Pose(position_mm=up, quaternion_xyzw=np.asarray(self.pose.quaternion_xyzw, dtype=np.float64),
                    frame=Frame.BASE, label="lift")

    def _move(self, target: Pose, *, linear: bool) -> None:
        timeout = self._steady_timeout()
        if timeout is not None and not self.arm.wait_until_steady(timeout):
            self.poses.append(target)
            self.statuses.append(MotionStatus.TIMEOUT)
            raise _Refused(self._report(HandlingOutcome.MOTION_REFUSED, message=(
                f"the arm did not come to rest within {timeout:.2f} s before the next motion (safety.dwell)")))
        keywords: dict[str, Any] = {"linear": linear}
        if chosen(self.camera_world):
            keywords["camera_world"] = self.camera_world
        try:
            result = self.arm.move(target, **keywords)
        except CameraWorldUnavailable as exc:
            # The verb promises a report, so a camera that could not vouch ends it as its own outcome rather than a
            # raise that loses the motions before it.
            self.poses.append(target)
            raise _Refused(self._report(HandlingOutcome.CAMERA_WORLD_UNAVAILABLE, message=str(exc)))
        except Exception as exc:  # noqa: BLE001 (a raising driver ends the verb, reported)
            self.poses.append(target)
            raise _Refused(self._report(HandlingOutcome.MOTION_REFUSED, message=f"{type(exc).__name__}: {exc}"))
        self.poses.append(target)
        if isinstance(result, MotionResult):
            self.statuses.append(result.status)
            self.stamps.append(result.camera_world)
            if not result.ok:
                raise _Refused(self._report(HandlingOutcome.MOTION_REFUSED, message=result.message or ""))

    def _line_out_width(self, close_to: float, width_mm: float) -> float:
        """The width the line out is judged carrying: what the attach after the close carries, where that is known
        ahead. A hand that does not measure its width is attached at the width it is told (:func:`grasp`), so it is
        judged at exactly that, the box the attach hands the planner; a hand that measures is attached at what it
        measures, and the part's own width stands in for it, never narrower than the jaws are told."""
        return max(close_to, float(width_mm)) if width_is_measured_of(self.gripper) else close_to

    def _line_out_refusal(self, standoff: Pose, width_mm: float) -> str:
        """Why the line back up to ``standoff``, judged at the part as if the jaws held a part ``width_mm`` across, would
        be refused, or could not be judged; ``""`` where it would run, and on an arm that does not judge it
        (``JudgesCarriedLines``). Nothing moves. It carries the verb's decline, as every motion of it does, and a camera
        that cannot vouch for the cell ends the verb as its own outcome, as on every motion of it."""
        if not isinstance(self.arm, JudgesCarriedLines):
            return ""
        keywords: dict[str, Any] = {"grip_width_mm": width_mm}
        if chosen(self.camera_world):
            keywords["camera_world"] = self.camera_world
        try:
            refused = self.arm.carried_line_refusal(standoff, **keywords)
        except CameraWorldUnavailable as exc:
            raise _Refused(self._report(HandlingOutcome.CAMERA_WORLD_UNAVAILABLE, message=str(exc))) from None
        except Exception as exc:  # noqa: BLE001 (a line out nobody could judge is no line the jaws close for)
            return f"the line out could not be judged as if the jaws held the part ({type(exc).__name__}: {exc})"
        if not isinstance(refused, MotionResult) or refused.ok:
            return ""
        return (f"the line out, judged at the part as if the jaws held the part {width_mm:g} mm across, would be "
                f"refused: {refused.message}")

    def _command_gripper(self, width_mm: float, *, close: bool) -> None:
        self.jaws_commanded = True
        try:
            _command(self.gripper, width_mm, close=close)
        except Exception as exc:  # noqa: BLE001 (a gripper fault ends the verb, after the jaws are stopped)
            hand = _fault(self.gripper, HandVerb.RELEASE, width_mm, exc)
            raise _Refused(self._report(HandlingOutcome.GRIPPER_FAULT, hand=hand, message=hand.error))

    def _steady_timeout(self) -> float | None:
        """The arm's own steady gate, where its tree asks for one and it can wait; ``None`` otherwise, and for an arm that
        gates its own sends (``safety.dwell.gate_at`` ``send``), which waits right before it sends, after its judgement."""
        return None if gates_its_own_sends(self.arm) else steady_timeout_of(self.arm)

    def _report(self, outcome: HandlingOutcome, *, hand: HandReport | None = None, message: str = "") -> HandlingReport:
        return HandlingReport(
            verb=self.verb, outcome=outcome, poses=tuple(self.poses), statuses=tuple(self.statuses),
            camera_worlds=tuple(self.stamps), line=self.line, keep_out_held=self.keep_out_held, hand=hand,
            message=message, another_candidate_may_follow=self._another_may_follow(outcome, message),
        )

    def _another_may_follow(self, outcome: HandlingOutcome, message: str) -> bool:
        """Contract 4 of the cell-fix plan, read off what this pick commanded: the carried lift refused and the empty
        hand backed out open, or the pick's first motion refused by a guard or the planner before anything was sent,
        with nothing commanded to the jaws. Everything else, a place included, is false."""
        if self.verb is not HandlingVerb.PICK:
            return False
        if outcome is HandlingOutcome.CARRIED_RETREAT_REFUSED:
            return True
        if outcome is not HandlingOutcome.MOTION_REFUSED or self.jaws_commanded:
            return False
        if len(self.poses) != 1 or len(self.statuses) != 1:
            return False   # a motion ran before the refused one, or the refused one raised and says nothing
        status = self.statuses[0]
        if status is MotionStatus.TIMEOUT:
            return message == NO_PLAN_FAIL_SAFE_MESSAGE
        return status in _REFUSED_BEFORE_SENDING
