"""What a robot's arm does as one verb: move to a pose, move to joints, go home, and say what stood behind it.

``Robot.move``, ``Robot.move_joints`` and ``Robot.home`` delegate here. Everything that moves the arm goes through
cuRobo and the exact mesh guard (Coal) with the camera world, so each verb reads, before any command, how the arm's
motions reach it (:func:`route_of`):

* PLANNED: cuRobo plans or checks every motion of this arm, and the exact mesh guard judges its path sample by sample.
  The verb runs through the arm's own checked verbs, whose camera world is mandatory or declined.
* UNPLANNED: a desk arm, the dummy or the kinematic mock. It has no controller, the pose is set, and nothing real moves.
  The verb runs, and the motion's camera world says UNPLANNED.
* REFUSED: an arm that drives a controller and plans nothing (a UR on the ik planner, a KUKA), a cuRobo arm whose paths
  cannot be judged, or an arm that does not say. The verb commands nothing.

Before any command a verb also refuses a link that is not open, a pose not in BASE, and a camera world the arm's own
motion would be refused for (``ReadsCameraWorld``), in that order and before the route. It then waits for the arm's
own steady gate where its tree asks for one (``safety.dwell``). A decline reaches ``move`` and ``move_joints`` as the
arm's own keyword, and ``home`` as a block around the arm's ``move_home``, which takes no keyword.

Every verb returns a frozen :class:`MotionReport`. A camera that could not vouch for the cell
(``CameraWorldUnavailable``) and a driver fault (``RobotError``) end the verb as an outcome rather than a raise. A
programmer's error still raises: a decline that is not a :class:`~src.robot.core.camera_world.CameraWorldDecline`, and
an arm whose typed verb hands back something that is not a ``MotionResult``.

``move_home`` answers with a bool on every driver, so a home the arm refuses reads UNKNOWN here, and the driver's log
holds the gate that refused it.

The module imports nothing above ``robot.core`` and connects nothing: the verbs run inside ``Robot.connected()``.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.contracts import UNSET, Maybe, chosen
from src.geometry import Frame, Pose
from src.robot.core.arm_capabilities import LineMotion, LineReading, line_motion_of
from src.robot.core.camera_world import (
    CameraWorldDecline,
    CameraWorldStamp,
    ReadsCameraWorld,
    camera_world_refusal,
    without_camera_world,
)
from src.robot.core.capabilities import RobotCapabilities
from src.robot.core.errors import (
    CameraWorldUnavailable,
    RobotConnectionError,
    RobotEmergencyStop,
    RobotError,
    RobotKinematicsError,
    RobotMotionRejected,
)
from src.robot.core.joint_positions import JointPositions
from src.robot.core.motion_result import MotionCommand, MotionResult, MotionStatus

__all__ = [
    "MotionOutcome",
    "MotionReport",
    "MotionRoute",
    "MotionVerb",
    "RouteReading",
    "decline_of",
    "home",
    "move",
    "move_joints",
    "route_of",
    "steady_timeout_of",
]


class MotionVerb(StrEnum):
    """Which motion verb a report is about."""

    MOVE = "move"
    MOVE_JOINTS = "move_joints"
    HOME = "home"


#: The arm command each verb reaches.
_COMMANDS = {
    MotionVerb.MOVE: MotionCommand.MOVE_TO,
    MotionVerb.MOVE_JOINTS: MotionCommand.MOVE_JOINTS,
    MotionVerb.HOME: MotionCommand.MOVE_HOME,
}


class MotionOutcome(StrEnum):
    """How a motion verb ended."""

    #: The arm ran the motion.
    EXECUTED = "executed"
    #: The arm's own verb was asked and refused the motion, or its driver raised; the result says why.
    MOTION_REFUSED = "motion_refused"
    #: A camera could not vouch for the cell through its fresh-frame attempts. Nothing was commanded, and a caller stops
    #: rather than trying again.
    CAMERA_WORLD_UNAVAILABLE = "camera_world_unavailable"
    #: Refused before any command: a link that is not open, a pose not in BASE, a camera world the arm would refuse, an
    #: arm whose motions do not go through cuRobo and the exact mesh guard, or an arm that did not come to rest.
    REFUSED = "refused"


class MotionRoute(StrEnum):
    """How a robot verb's motions reach the arm, read off the arm before any command."""

    #: cuRobo plans or checks every motion against the camera world or a decline, and the exact mesh guard judges its
    #: path sample by sample.
    PLANNED = "planned"
    #: A desk arm with no controller: the pose is set, nothing plans it, and nothing real moves.
    UNPLANNED = "unplanned"
    #: A motion would reach a controller unplanned or unjudged, or the arm does not say. Nothing is commanded.
    REFUSED = "refused"


def _ascii(text: str) -> str:
    """``text`` with every character outside ASCII written as its escape, so ``render()`` stays ASCII."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _line_dict(line: LineReading | None) -> dict[str, str] | None:
    return None if line is None else {"motion": line.motion.value, "reason": line.reason}


@dataclass(frozen=True, slots=True)
class RouteReading:
    """A route, the sentence that says why, and the line reading it was read from."""

    route: MotionRoute
    reason: str
    #: What the arm keeps of a straight line (``KeepsLines``), or ``None`` for an arm that does not say.
    line: LineReading | None = None

    @property
    def runs(self) -> bool:
        """Whether a motion on this route may be commanded: PLANNED or UNPLANNED."""
        return self.route is not MotionRoute.REFUSED

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Describe this to a person, as one line of ASCII, no trailing newline."""
        text = f"{self.route.value.upper()}: {_ascii(self.reason)}"
        return text if self.line is None else f"{text} (line {self.line.motion.value.upper()})"

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe."""
        return {"route": self.route.value, "runs": self.runs, "reason": self.reason, "line": _line_dict(self.line)}


def _simulated(arm: object) -> bool:
    capabilities = getattr(arm, "capabilities", None)
    return isinstance(capabilities, RobotCapabilities) and bool(capabilities.is_simulated)


def route_of(arm: object) -> RouteReading:
    """How ``arm``'s motions reach it, read before any command. It opens no connection and starts no planner.

    The arm's line reading decides (``KeepsLines``), because it is the arm's own answer to whether a path of it is
    judged: CHECKED on a cuRobo arm whose exact mesh guard judges every sample, TELEPORT on an arm with no controller.
    PLANNED also needs the arm to say that it plans its paths (``plans_paths``), and UNPLANNED needs capabilities that
    say the arm is simulated, so a caller's own arm that claims a teleport on a real controller is refused.
    """
    line = line_motion_of(arm)
    if line is None:
        return RouteReading(MotionRoute.REFUSED, (
            f"{type(arm).__name__} does not say what it keeps of a path (KeepsLines), so nothing shows that its motions "
            "go through cuRobo and the exact mesh guard"))
    plans = bool(getattr(arm, "plans_paths", False))
    if line.motion is LineMotion.CHECKED and plans:
        return RouteReading(MotionRoute.PLANNED, line.reason, line)
    if line.motion is LineMotion.TELEPORT and _simulated(arm):
        return RouteReading(MotionRoute.UNPLANNED, f"a desk arm, and nothing real moves: {line.reason}", line)
    if plans:
        return RouteReading(MotionRoute.REFUSED, f"cuRobo plans this arm and no path of it can be judged: {line.reason}",
                            line)
    return RouteReading(MotionRoute.REFUSED, (
        f"nothing plans the motions of this arm against the camera world or judges their paths: {line.reason}"), line)


def steady_timeout_of(arm: object) -> float | None:
    """The arm's own steady gate, where its tree asks for one and it can wait; ``None`` otherwise."""
    dwell = getattr(getattr(getattr(arm, "config", None), "safety", None), "dwell", None)
    if dwell is None or not callable(getattr(arm, "wait_until_steady", None)):
        return None
    if not bool(getattr(dwell, "require_steady_before_motion", False)):
        return None
    return float(getattr(dwell, "steady_timeout_s", 5.0))


def decline_of(
    decline: Maybe[str], camera_world: Maybe[CameraWorldDecline] = UNSET,
) -> Maybe[CameraWorldDecline]:
    """The decline a verb carries: ``decline`` as a reason, or ``camera_world``, the same decline as an object.

    ``camera_world`` is kept as an alias of ``decline``. Unset when neither was chosen. Both at once, a decline that is
    not text and a blank reason are a programmer's error and raise (``TypeError``, ``TypeError``, ``ValueError``),
    before anything moves.
    """
    if chosen(decline) and chosen(camera_world):
        raise TypeError("pass decline='why this motion needs no camera world' or camera_world=..., not both")
    if chosen(camera_world):
        if not isinstance(camera_world, CameraWorldDecline):
            raise TypeError(
                f"camera_world is a CameraWorldDecline, not {camera_world!r}; pass decline='why this motion needs no "
                f"camera world' instead")
        return camera_world
    if not chosen(decline):
        return UNSET
    if not isinstance(decline, str):
        raise TypeError(f"decline is the reason as text, not {decline!r}")
    return CameraWorldDecline(decline)


@dataclass(frozen=True, slots=True)
class MotionReport:
    """What one motion verb asked of the arm, how the arm's motions reach it, and what the arm said."""

    verb: MotionVerb
    outcome: MotionOutcome
    #: How this arm's motions reach it, read before any command.
    route: RouteReading
    #: The arm's typed result, or the one the verb refused with before any command: the status, the message, the target
    #: and the camera world stamp.
    result: MotionResult
    #: What the arm keeps of the straight line a ``move(linear=True)`` asked for; ``None`` for every other motion.
    line: LineReading | None = None

    @property
    def ok(self) -> bool:
        """Whether the arm ran the motion."""
        return self.outcome is MotionOutcome.EXECUTED

    @property
    def status(self) -> MotionStatus:
        return self.result.status

    @property
    def message(self) -> str:
        return self.result.message

    @property
    def camera_world(self) -> CameraWorldStamp:
        return self.result.camera_world

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        lines = [f"{self.verb.value}  {self.outcome.value.upper()}"]
        lines.extend(f"  {line}" for line in self.result.render().split("\n"))
        lines.append(f"  route  {self.route.render()}")
        if self.line is not None:
            lines.append(f"  line  {self.line.motion.value.upper()}: {_ascii(self.line.reason)}")
        if self.outcome is MotionOutcome.CAMERA_WORLD_UNAVAILABLE:
            lines.append("  a camera could not vouch for the cell: look at the camera rather than trying again")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe."""
        return {
            "verb": self.verb.value,
            "outcome": self.outcome.value,
            "ok": self.ok,
            "route": self.route.to_dict(),
            "line": _line_dict(self.line),
            "result": self.result.to_dict(),
        }


def move(
    arm: Any,
    pose: Pose,
    *,
    decline: Maybe[CameraWorldDecline] = UNSET,
    linear: bool = False,
    vel: Maybe[float] = UNSET,
    acc: Maybe[float] = UNSET,
) -> MotionReport:
    """Move ``arm`` to ``pose`` (BASE), as a straight line when ``linear``, through its own typed ``move``.

    ``vel`` and ``acc`` reach the arm only when chosen, so the driver's own default speed stands otherwise.
    """
    keywords: dict[str, Any] = {"linear": bool(linear)}
    if chosen(vel):
        keywords["vel"] = float(vel)
    if chosen(acc):
        keywords["acc"] = float(acc)
    if chosen(decline):
        keywords["camera_world"] = decline
    return _Motion(arm, MotionVerb.MOVE, decline, target_pose=pose, linear=bool(linear)).run(
        lambda: arm.move(pose, **keywords))


def move_joints(arm: Any, joints: JointPositions, *, decline: Maybe[CameraWorldDecline] = UNSET) -> MotionReport:
    """Move ``arm`` to ``joints`` (radians) through its own typed ``move_to_joints``."""
    keywords: dict[str, Any] = {"camera_world": decline} if chosen(decline) else {}
    return _Motion(arm, MotionVerb.MOVE_JOINTS, decline, target_joints=joints).run(
        lambda: arm.move_to_joints(joints, **keywords))


def home(arm: Any, *, decline: Maybe[CameraWorldDecline] = UNSET) -> MotionReport:
    """Move ``arm`` to its configured home through its own gated ``move_home``, inside the decline when one is given."""

    def command() -> bool:
        scope: AbstractContextManager[Any] = (
            without_camera_world(arm, decline.reason) if chosen(decline) else nullcontext())
        with scope:
            return bool(arm.move_home())

    return _Motion(arm, MotionVerb.HOME, decline).run(command)


#: What a home the arm refused says: ``move_home`` answers with a bool on every driver.
_HOME_REFUSED = ("the arm refused the move home, and its move_home answers only with False: the driver's log names "
                 "the gate that refused it")


class _Motion:
    """One motion verb: the refusals, then the steady gate, then the arm's own verb, read into one report."""

    def __init__(self, arm: Any, verb: MotionVerb, decline: Maybe[CameraWorldDecline], *,
                 target_pose: Pose | None = None, target_joints: JointPositions | None = None,
                 linear: bool = False) -> None:
        self.arm = arm
        self.verb = verb
        self.command = _COMMANDS[verb]
        self.decline = decline
        self.target_pose = target_pose
        self.target_joints = target_joints
        self.linear = linear

    def run(self, command: Callable[[], object]) -> MotionReport:
        route = route_of(self.arm)
        line = route.line if self.linear else None
        stamp = self._stamp_before(route)
        refused = self._refusal(route, stamp)
        if refused is not None:
            return MotionReport(self.verb, MotionOutcome.REFUSED, route, refused, line)
        try:
            timeout = steady_timeout_of(self.arm)
            if timeout is not None and not self.arm.wait_until_steady(timeout):
                return MotionReport(self.verb, MotionOutcome.REFUSED, route, self._failed(
                    MotionStatus.TIMEOUT, f"the arm did not come to rest within {timeout:.2f} s before the motion "
                    "(safety.dwell)", stamp), line)
            answer = command()
        except CameraWorldUnavailable as exc:
            return MotionReport(self.verb, MotionOutcome.CAMERA_WORLD_UNAVAILABLE, route,
                                self._failed(MotionStatus.UNSUPPORTED, str(exc), stamp, exc), line)
        except RobotError as exc:
            return MotionReport(self.verb, MotionOutcome.MOTION_REFUSED, route, self._raised(exc, stamp), line)
        result = self._result_of(answer, stamp)
        outcome = MotionOutcome.EXECUTED if result.ok else MotionOutcome.MOTION_REFUSED
        return MotionReport(self.verb, outcome, route, result, line)

    # ---- before any command -----------------------------------------------------------------

    def _stamp_before(self, route: RouteReading) -> CameraWorldStamp:
        """The camera world a motion would carry, read as the arm is now; UNPLANNED on a desk arm that does not say."""
        if isinstance(self.arm, ReadsCameraWorld):
            return self.arm.camera_world_for_move(self.decline)
        if route.route is MotionRoute.UNPLANNED:
            return CameraWorldStamp.unplanned(route.reason)
        return CameraWorldStamp.unstated()

    def _refusal(self, route: RouteReading, stamp: CameraWorldStamp) -> MotionResult | None:
        if not bool(getattr(self.arm, "is_connected", False)):
            return self._failed(MotionStatus.CONNECTION_ERROR, (
                f"the arm is not connected; call {self.verb.value} inside robot.connected()"), stamp)
        if self.target_pose is not None and self.target_pose.frame is not Frame.BASE:
            return self._failed(MotionStatus.INVALID_TARGET, (
                f"the pose is in {self.target_pose.frame.value!r}, and {self.verb.value} takes a pose in BASE"), stamp)
        if isinstance(self.arm, ReadsCameraWorld):
            sentence = camera_world_refusal(stamp, live_world_wired=self.arm.live_camera_world_wired())
            if sentence is not None:
                return self._failed(MotionStatus.UNSUPPORTED, sentence, stamp)
        if not route.runs:
            return self._failed(MotionStatus.UNSUPPORTED, route.reason, stamp)
        return None

    # ---- what the arm said ------------------------------------------------------------------

    def _failed(self, status: MotionStatus, message: str, stamp: CameraWorldStamp,
                exc: BaseException | None = None) -> MotionResult:
        return MotionResult.failed(status, self.command, target_pose=self.target_pose,
                                   target_joints=self.target_joints, message=message, exception=exc,
                                   camera_world=stamp)

    def _raised(self, exc: RobotError, stamp: CameraWorldStamp) -> MotionResult:
        """A driver's raise as a result: the typed refusal it carries where it has one, else its type as a status."""
        carried = exc.result if isinstance(exc, RobotMotionRejected) else None
        if isinstance(carried, MotionResult):
            return carried
        if isinstance(exc, RobotConnectionError):
            status = MotionStatus.CONNECTION_ERROR
        elif isinstance(exc, RobotKinematicsError):
            status = MotionStatus.IK_FAILED
        elif isinstance(exc, RobotEmergencyStop):
            status = MotionStatus.CONTROLLER_REJECTED
        else:
            status = MotionStatus.UNKNOWN
        return self._failed(status, f"{type(exc).__name__}: {exc}", stamp, exc)

    def _result_of(self, answer: object, stamp: CameraWorldStamp) -> MotionResult:
        if isinstance(answer, MotionResult):
            return answer
        if self.verb is MotionVerb.HOME and isinstance(answer, bool):
            if answer:
                return MotionResult.executed(self.command, message="move_home", camera_world=stamp)
            return self._failed(MotionStatus.UNKNOWN, _HOME_REFUSED, stamp)
        raise TypeError(
            f"{type(self.arm).__name__} answered {self.verb.value} with {type(answer).__name__}, not a MotionResult: "
            "its typed verbs break the RobotArm contract")
