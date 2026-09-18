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
* No force is commanded.
* A gripper that raises is stopped once where it can be stopped, and the report carries the fault.
* Nothing is commanded on a robot with no gripper, a gripper that holds nothing, or an arm or gripper whose link is not
  open.

``Robot.pick`` and ``Robot.place`` put the hand verbs at the end of a straight descent. Before any command they refuse,
in this order: a robot that cannot hold, a link that is not open, a pose not in BASE, a camera world the arm's own
motion would be refused for (``ReadsCameraWorld``), and an arm that keeps no line or does not say (``KeepsLines``). Then
the motions: a planned move to the standoff, a line down to the pose, the hand verb, and a line back up to the
standoff, each move carrying the caller's decline, each preceded by the arm's own steady gate where its tree asks for
one (``safety.dwell``). A refused motion ends the verb with nothing commanded after it. A pick holds its keep-out offer
in the arm's live world from before the detach to after its last motion, and forgets it on any exit.

The module imports nothing above ``robot.core`` and connects nothing: the verbs run inside ``Robot.connected()``.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from src.contracts import UNSET, Maybe, chosen
from src.geometry import Frame, Pose
from src.robot.core import MotionResult, MotionStatus, StoppableGripper
from src.robot.core.arm_capabilities import CarriesPayload, LineMotion, LineReading, PayloadModel, line_motion_of
from src.robot.core.camera_world import (
    CameraWorldDecline,
    CameraWorldStamp,
    ReadsCameraWorld,
    camera_world_refusal,
    weakest_camera_world,
)
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.core.gripper import HoldEvidence, hold_evidence_of, width_is_measured_of
from src.robot.core.keep_out import SegmentationOffer, keeping_out

__all__ = [
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


class HandVerb(StrEnum):
    """Which hand verb a report is about."""

    GRASP = "grasp"
    RELEASE = "release"


class HandOutcome(StrEnum):
    """How a hand verb ended."""

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
    """What one hand verb commanded, measured and did to the carried part model."""

    verb: HandVerb
    outcome: HandOutcome
    commanded_width_mm: float | None = None
    reported_width_mm: float | None = None
    width_measured: bool = False
    hold: HoldEvidence = HoldEvidence.UNMEASURED
    payload: PayloadState = PayloadState.UNCHANGED
    payload_reason: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        """Whether the verb did what it was asked: the jaws closed on something or opened off it."""
        return self.outcome in (HandOutcome.GRASPED, HandOutcome.RELEASED)

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        head = f"{self.verb.value}  {self.outcome.value.upper()}"
        if self.outcome is HandOutcome.REFUSED:
            return f"{head}  nothing commanded: {self.error}"
        parts = []
        if self.commanded_width_mm is not None:
            parts.append(f"width {self.commanded_width_mm:.2f} mm commanded")
        if self.reported_width_mm is not None:
            read = "measured" if self.width_measured else "reported, width not measured"
            parts.append(f"{self.reported_width_mm:.2f} mm {read}")
        hold = "hold not measured" if self.hold is HoldEvidence.UNMEASURED else f"hold {self.hold.value.upper()}"
        if self.verb is HandVerb.RELEASE and self.hold is HoldEvidence.UNMEASURED:
            hold += ", release not checked"
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
        """Plain data, ``json.dumps`` safe."""
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
        }


def grasp(robot: Any, width_mm: float) -> HandReport:
    """Close ``robot``'s gripper to ``width_mm`` and hand what it holds to the arm's carried part model."""
    refused = _refusal(robot, HandVerb.GRASP)
    if refused is not None:
        return refused
    gripper = robot.gripper
    commanded = float(width_mm)
    try:
        gripper.set_width_mm(commanded)
        reported = float(gripper.get_width_mm())
        measured = width_is_measured_of(gripper)
        hold = hold_evidence_of(gripper)
    except Exception as exc:  # noqa: BLE001 (a gripper fault is the report, after the jaws are stopped)
        return _fault(gripper, HandVerb.GRASP, commanded, exc)
    if hold is HoldEvidence.EMPTY:
        return HandReport(verb=HandVerb.GRASP, outcome=HandOutcome.NOTHING_HELD, commanded_width_mm=commanded,
                          reported_width_mm=reported, width_measured=measured, hold=hold)
    state, reason = _attach(robot.arm, reported if measured else commanded)
    return HandReport(verb=HandVerb.GRASP, outcome=HandOutcome.GRASPED, commanded_width_mm=commanded,
                      reported_width_mm=reported, width_measured=measured, hold=hold, payload=state,
                      payload_reason=reason)


def release(robot: Any) -> HandReport:
    """Open ``robot``'s gripper to the hand's width and forget the carried part unless one is still measured."""
    refused = _refusal(robot, HandVerb.RELEASE)
    if refused is not None:
        return refused
    gripper = robot.gripper
    commanded = float(gripper.max_width_mm)
    try:
        gripper.set_width_mm(commanded)
        reported = float(gripper.get_width_mm())
        measured = width_is_measured_of(gripper)
        hold = hold_evidence_of(gripper)
    except Exception as exc:  # noqa: BLE001 (a gripper fault is the report, after the jaws are stopped)
        return _fault(gripper, HandVerb.RELEASE, commanded, exc)
    if hold is HoldEvidence.HELD:
        return HandReport(verb=HandVerb.RELEASE, outcome=HandOutcome.RELEASE_NOT_CONFIRMED,
                          commanded_width_mm=commanded, reported_width_mm=reported, width_measured=measured,
                          hold=hold, payload_reason="the gripper still measures a part, so its model is kept")
    state, reason = _detach(robot.arm)
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
    return None if not why else HandReport(verb=verb, outcome=HandOutcome.REFUSED, error=why)


def _fault(gripper: Any, verb: HandVerb, commanded: float, exc: BaseException) -> HandReport:
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
    """How a pick or a place ended."""

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
    """What one pick or place commanded, what stood behind each motion, and what the hand did."""

    verb: HandlingVerb
    outcome: HandlingOutcome
    poses: tuple[Pose, ...] = ()
    statuses: tuple[MotionStatus, ...] = ()
    camera_worlds: tuple[CameraWorldStamp, ...] = ()
    line: LineReading | None = None
    keep_out_held: bool = False
    hand: HandReport | None = None
    message: str = ""

    @property
    def ok(self) -> bool:
        """Whether every motion ran and the hand verb did what it was asked."""
        return self.outcome is HandlingOutcome.EXECUTED

    @property
    def weakest_camera_world(self) -> CameraWorldStamp | None:
        """The stamp a reader of the whole verb has to see: the first that does not vouch, else the last."""
        return weakest_camera_world(self.camera_worlds)

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
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
        if self.hand is not None:
            lines.extend(f"  {line}" for line in self.hand.render().split("\n"))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe."""
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
    """Pick the part at ``pose``, ``width_mm`` across: standoff, a line in, grasp, a line out."""
    return _Handling(robot, HandlingVerb.PICK, pose, standoff_mm, camera_world).pick(
        width_mm=float(width_mm), squeeze_mm=float(squeeze_mm), pre_open_mm=pre_open_mm, keep_out=keep_out)


def place(
    robot: Any,
    pose: Pose,
    *,
    standoff_mm: float = 80.0,
    camera_world: Maybe[CameraWorldDecline] = UNSET,
) -> HandlingReport:
    """Place the held part at ``pose``: standoff, a line in, release, a line out unless the release is not confirmed."""
    return _Handling(robot, HandlingVerb.PLACE, pose, standoff_mm, camera_world).place()


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

    def place(self) -> HandlingReport:
        refused = self._refusal()
        if refused is not None:
            return refused
        try:
            standoff = self._standoff()
            self._move(standoff, linear=False)
            self._move(self.pose, linear=True)
            hand = release(self.robot)
            if hand.outcome is HandOutcome.GRIPPER_FAULT:
                return self._report(HandlingOutcome.GRIPPER_FAULT, hand=hand, message=hand.error)
            if hand.outcome is HandOutcome.RELEASE_NOT_CONFIRMED:
                return self._report(HandlingOutcome.RELEASE_NOT_CONFIRMED, hand=hand, message=(
                    "the gripper still measures a part after opening, so the arm stays where it stands"))
            if hand.outcome is not HandOutcome.RELEASED:
                return self._report(HandlingOutcome.REFUSED, hand=hand, message=hand.error)
            self._move(standoff, linear=True)
            return self._report(HandlingOutcome.EXECUTED, hand=hand)
        except _Refused as ended:
            return ended.report

    def _pick_body(self, *, width_mm: float, squeeze_mm: float, pre_open_mm: "Maybe[float | None]") -> HandlingReport:
        detach = getattr(self.arm, "detach_payload", None)
        if callable(detach):
            detach()
        opening = float(self.gripper.max_width_mm) if not chosen(pre_open_mm) else pre_open_mm
        if opening is not None:
            self._command_gripper(float(opening))
        standoff = self._standoff()
        self._move(standoff, linear=False)
        self._move(self.pose, linear=True)
        close_to = max(float(self.gripper.min_width_mm), width_mm - squeeze_mm)
        hand = grasp(self.robot, close_to)
        if hand.outcome is HandOutcome.GRIPPER_FAULT:
            return self._report(HandlingOutcome.GRIPPER_FAULT, hand=hand, message=hand.error)
        if hand.outcome is HandOutcome.NOTHING_HELD:
            self._command_gripper(float(self.gripper.max_width_mm))
            self._move(standoff, linear=True)
            return self._report(HandlingOutcome.NOTHING_HELD, hand=hand, message=(
                "the gripper measured nothing held, so it opened and backed out to the standoff"))
        if hand.outcome is not HandOutcome.GRASPED:
            return self._report(HandlingOutcome.REFUSED, hand=hand, message=hand.error)
        self._move(standoff, linear=True)
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
        reading = line_motion_of(self.arm)
        if reading is None:
            return self._report(HandlingOutcome.REFUSED, message=(
                f"{type(self.arm).__name__} does not say what it keeps of a straight line, so the descent cannot be "
                "promised"))
        self.line = reading
        if reading.motion is LineMotion.NOT_KEPT:
            return self._report(HandlingOutcome.REFUSED, message=f"this arm keeps no straight line: {reading.reason}")
        return None

    # ---- the steps ----------------------------------------------------------------------------

    def _standoff(self) -> Pose:
        """``pose`` moved back along its own +Z, the approach, by the standoff, turned as the pose is."""
        matrix = np.asarray(self.pose.to_matrix(), dtype=np.float64)
        back = matrix[:3, 3] - matrix[:3, 2] * self.standoff_mm
        return Pose(position_mm=back, quaternion_xyzw=np.asarray(self.pose.quaternion_xyzw, dtype=np.float64),
                    frame=Frame.BASE, label="standoff")

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
        except CameraWorldUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 (a raising driver ends the verb, reported)
            self.poses.append(target)
            raise _Refused(self._report(HandlingOutcome.MOTION_REFUSED, message=f"{type(exc).__name__}: {exc}"))
        self.poses.append(target)
        if isinstance(result, MotionResult):
            self.statuses.append(result.status)
            self.stamps.append(result.camera_world)
            if not result.ok:
                raise _Refused(self._report(HandlingOutcome.MOTION_REFUSED, message=result.message or ""))

    def _command_gripper(self, width_mm: float) -> None:
        try:
            self.gripper.set_width_mm(width_mm)
        except Exception as exc:  # noqa: BLE001 (a gripper fault ends the verb, after the jaws are stopped)
            hand = _fault(self.gripper, HandVerb.RELEASE, width_mm, exc)
            raise _Refused(self._report(HandlingOutcome.GRIPPER_FAULT, hand=hand, message=hand.error))

    def _steady_timeout(self) -> float | None:
        """The arm's own steady gate, where its tree asks for one and it can wait; ``None`` otherwise."""
        dwell = getattr(getattr(getattr(self.arm, "config", None), "safety", None), "dwell", None)
        if dwell is None or not callable(getattr(self.arm, "wait_until_steady", None)):
            return None
        if not bool(getattr(dwell, "require_steady_before_motion", False)):
            return None
        return float(getattr(dwell, "steady_timeout_s", 5.0))

    def _report(self, outcome: HandlingOutcome, *, hand: HandReport | None = None, message: str = "") -> HandlingReport:
        return HandlingReport(
            verb=self.verb, outcome=outcome, poses=tuple(self.poses), statuses=tuple(self.statuses),
            camera_worlds=tuple(self.stamps), line=self.line, keep_out_held=self.keep_out_held, hand=hand,
            message=message,
        )
