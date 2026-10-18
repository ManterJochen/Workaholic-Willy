"""Optional capability extensions for arm drivers.

:class:`~src.robot.core.RobotArm` stays a small, strictly vendor-neutral Protocol
with a contract-locked member set. The extras a controller can offer (digital and
analog I/O, live force and torque, live robot and safety status, what a straight line
keeps, a move home that says why it was refused, the configuration a pose goes to, a joint
move on its straight joint line and nothing else, a grasp's lift judged before the jaws close, where the controller
puts its flange, what a UR cell judges and holds ahead of its motions, and a cell shared with other arms) are
declared here as separate ``runtime_checkable`` Protocols,
mirroring :class:`~src.robot.core.gripper.ObjectDetectingGripper`. Hand guiding follows the same
pattern from its own module, :mod:`~src.robot.core.freedrive`: an arm a person may move by
hand advertises ``SupportsFreedrive`` (the UR driver, on teach mode), and one that does not,
the sim arm among them, keeps the calibration stations it drives to itself.

A driver opts in by implementing one. A caller feature-checks with
``isinstance(arm, SupportsForceTorque)`` and falls back where the capability is
absent, so a driver that implements none keeps working and nothing is added to the
neutral :class:`RobotArm` surface.

The halt latch is one of them (:class:`SupportsHalt`): "halt now" in the console latches the arm, and a latched arm
sends nothing and switches no output until a person says the cell is clear. It folds into
:attr:`RobotStatus.is_operational`, so every gate that asks the controller refuses a halted arm, while
:attr:`RobotStatus.controller_operational` keeps the controller's own answer: a halt is not a controller stop.

Units follow the rest of the ``robot`` layer: forces in newtons, torques in
newton-metres, every wrench frame-tagged with :class:`~src.geometry.Frame`. The enums
are the vendor-neutral projection of a controller native mode, such as the UR
``getRobotMode`` and ``getSafetyMode`` integers, so another vendor's driver can
implement the same Protocols against its own controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from src.contracts import UNSET
from src.geometry import Frame

from .errors import RobotMotionRejected

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Sequence

    from src.contracts import Maybe
    from src.geometry import Pose

    from .camera_world import CameraWorldDecline
    from .cell_gate import CellGate
    from .errors import ArmFault
    from .joint_positions import JointPositions
    from .motion_result import MotionResult

__all__ = [
    "AppliesToolFrame",
    "BrakesWhenHalted",
    "ArmHalted",
    "CarriesWristBodies",
    "ChoosesConfigurations",
    "ControllerProvenance",
    "DigitalIOPort",
    "DrivesJointLines",
    "DrivesJointPaths",
    "HALT_BRAKE_OUTCOMES",
    "HaltState",
    "HoldsItsFingersAsOpened",
    "HoldsItsWorld",
    "HoldsItsWorldAtJunctions",
    "HomesTyped",
    "JudgesCarriedLines",
    "JudgesGraspsAhead",
    "JudgesLinesAhead",
    "JudgesNextLegs",
    "JudgesTheLineDown",
    "KeepsItsBranch",
    "KeepsLines",
    "KnowsItsFlange",
    "KnowsItsHand",
    "LineMotion",
    "LineReading",
    "ReadsItsStatusQuickly",
    "ReportsControllerProvenance",
    "ReportsWristBodies",
    "ScreensJointGoals",
    "RobotMode",
    "RobotStatus",
    "SafetyMode",
    "ScreensConfigurations",
    "SharesItsCell",
    "StepsItsOwnPhysics",
    "SupportsDigitalIO",
    "SupportsForceTorque",
    "SupportsHalt",
    "SupportsRobotStatus",
    "SupportsWatching",
    "Wrench",
    "brakes_in_motion_of",
    "controller_flanges_of",
    "end_pulse",
    "halt_state_of",
    "halted_refusal",
    "line_motion_of",
]


class DigitalIOPort(StrEnum):
    """Which digital I/O bank a pin lives on. UR has standard, configurable and tool."""

    STANDARD = "standard"
    CONFIGURABLE = "configurable"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class Wrench:
    """A 6-DoF force and torque reading.

    The forces ``fx, fy, fz`` are newtons and the torques ``tx, ty, tz`` are
    newton-metres, expressed in ``frame``. The vendor-neutral default is the robot
    BASE frame, which is what UR ``getActualTCPForce`` reports.
    """

    fx: float
    fy: float
    fz: float
    tx: float
    ty: float
    tz: float
    frame: Frame = Frame.BASE

    @property
    def force(self) -> tuple[float, float, float]:
        """The linear force component ``(fx, fy, fz)`` in newtons."""
        return (self.fx, self.fy, self.fz)

    @property
    def torque(self) -> tuple[float, float, float]:
        """The moment component ``(tx, ty, tz)`` in newton-metres."""
        return (self.tx, self.ty, self.tz)

    @property
    def force_magnitude(self) -> float:
        """Euclidean magnitude of the linear force in newtons, the hand-over signal."""
        return (self.fx * self.fx + self.fy * self.fy + self.fz * self.fz) ** 0.5


class RobotMode(StrEnum):
    """Vendor-neutral robot operating mode, the projection of UR ``getRobotMode``."""

    DISCONNECTED = "disconnected"
    CONFIRM_SAFETY = "confirm_safety"
    BOOTING = "booting"
    POWER_OFF = "power_off"
    POWER_ON = "power_on"
    IDLE = "idle"
    BACKDRIVE = "backdrive"
    RUNNING = "running"
    UPDATING_FIRMWARE = "updating_firmware"
    UNKNOWN = "unknown"


class SafetyMode(StrEnum):
    """Vendor-neutral safety mode, the projection of UR ``getSafetyMode``."""

    NORMAL = "normal"
    REDUCED = "reduced"
    PROTECTIVE_STOP = "protective_stop"
    RECOVERY = "recovery"
    SAFEGUARD_STOP = "safeguard_stop"
    SYSTEM_EMERGENCY_STOP = "system_emergency_stop"
    ROBOT_EMERGENCY_STOP = "robot_emergency_stop"
    VIOLATION = "violation"
    FAULT = "fault"
    UNKNOWN = "unknown"

    @property
    def is_stopped(self) -> bool:
        """True for a mode where the arm is halted or faulted.

        NORMAL, REDUCED and RECOVERY are the modes this excludes.
        """
        return self in {
            SafetyMode.PROTECTIVE_STOP,
            SafetyMode.SAFEGUARD_STOP,
            SafetyMode.SYSTEM_EMERGENCY_STOP,
            SafetyMode.ROBOT_EMERGENCY_STOP,
            SafetyMode.VIOLATION,
            SafetyMode.FAULT,
        }


@dataclass(frozen=True, slots=True)
class RobotStatus:
    """A snapshot of the controller's live robot and safety state, and of the arm's halt latch.

    ``message`` carries the controller's own text, such as the UR dashboard
    ``safetystatus`` string, so an operator reads a fault rather than a code.
    ``halted`` is why the arm's halt latch is set (:class:`SupportsHalt`), ``""`` while it is not: it is the arm's,
    not the controller's, and it is read beside the controller's fields rather than in them.

    ``program_running`` says whether the program on the controller that takes this cell's motion commands runs (on a
    UR the RTDE control script). It is required and keyword-only, so a producer that cannot say does not build a
    status that reads as ready (P0e E13): measured on URSim, a UR in RUNNING and NORMAL with no stop latched took no
    command at all once that script had stopped.
    """

    robot_mode: RobotMode
    safety_mode: SafetyMode
    protective_stopped: bool
    emergency_stopped: bool
    message: str = ""
    halted: str = ""
    program_running: bool = field(kw_only=True)

    @property
    def controller_ready(self) -> bool:
        """The controller's state alone: powered, running, in NORMAL safety and not stopped, its program and the halt
        latch aside."""
        return (
            self.robot_mode == RobotMode.RUNNING
            and self.safety_mode == SafetyMode.NORMAL
            and not self.protective_stopped
            and not self.emergency_stopped
        )

    @property
    def controller_operational(self) -> bool:
        """The controller's own answer: ready, and running the program that takes this cell's commands. The halt latch
        aside.

        What every console gate reads, after it read the latch: a halted arm on a healthy controller is halted, and
        must never read as a protective stop, whose remedy (the pendant) is not the halt's (a person confirms the cell
        is clear).
        """
        return self.controller_ready and self.program_running

    @property
    def is_operational(self) -> bool:
        """True only when the controller is operational and the arm is not halted.

        The one criterion every library gate that asks the controller uses (the hand verbs, the pick loop, the grasp
        policy, the push, the locator, freedrive), so a halted arm is refused by all of them with no edit to any.
        """
        return self.controller_operational and not self.halted

    @property
    def is_stopped(self) -> bool:
        """True when a protective or emergency stop is active, or the safety mode is a stop."""
        return self.protective_stopped or self.emergency_stopped or self.safety_mode.is_stopped


@runtime_checkable
class SupportsDigitalIO(Protocol):
    """Capability extension: read and write the controller's digital and analog I/O.

    ``pin`` indexes the bank named by ``port``: on UR, 0 to 7 for standard and
    configurable, 0 to 1 for tool. ``value`` is the logic level. A driver advertising
    this lets a pipeline drive an external actuator or read a sensor through the
    controller I/O.
    """

    def set_digital_output(
        self, pin: int, value: bool, *, port: DigitalIOPort = DigitalIOPort.STANDARD
    ) -> None:
        """Drive a digital output pin high (``True``) or low (``False``)."""
        ...

    def get_digital_input(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        """Read a digital input pin's logic level."""
        ...

    def get_digital_output(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        """Read back a digital output pin's commanded logic level."""
        ...

    def set_analog_output(self, pin: int, value: float, *, current: bool = False) -> None:
        """Set an analog output. ``value`` is volts, or amps where ``current=True``."""
        ...


@runtime_checkable
class SupportsForceTorque(Protocol):
    """Capability extension: read live TCP force and torque plus per-joint torques.

    The use is collaborative hand-over detection: a spike in
    :attr:`Wrench.force_magnitude`, or in a joint torque, means a person is pushing or
    pulling the tool. A vendor with no wrist F/T channel does not implement it.
    """

    def get_tcp_wrench(self) -> Wrench:
        """The current force and torque at the TCP, in N and Nm, in the BASE frame."""
        ...

    def get_joint_torques(self) -> tuple[float, ...]:
        """The current torque at each joint in newton-metres, in base-to-tool order."""
        ...


@runtime_checkable
class CarriesWristBodies(Protocol):
    """Capability extension: the arm holds the wrist cameras mounted on its own flange.

    A camera on the wrist travels with the arm, so the planner routes around it and the self filter takes it out of
    what the cameras see. An arm that cannot take them is one a cell with a wrist camera may not be built on, and
    the refusal says so by name (:class:`~src.robot.execution.wrist_bodies.WristBodies`). Until 2026-09-20 the caller
    asked with ``getattr``, so an arm without the member never received the cameras, with nothing saying so.

    The bodies are ``safety.planning.body_link.WristBody`` values; ``core`` sits below ``safety`` and types them
    loosely rather than importing upward.
    """

    def set_wrist_bodies(self, bodies: "Sequence[Any]") -> None:
        """Hand this arm the wrist cameras it carries, before its planner or guard is built. Empty clears them."""
        ...


@runtime_checkable
class AppliesToolFrame(Protocol):
    """Capability extension: the arm says which flange-to-TCP frame its poses stand on, right now.

    A hand-eye solve is recorded against the frame applied while it was measured, so an arm that cannot say which
    one it applied leaves the record unanchored, and the artifact says so instead of guessing.
    """

    @property
    def active_tool_frame(self) -> "np.ndarray | None":
        """The flange to TCP this arm applies now, as a 4x4 in millimetres, or ``None`` where it is not known."""
        ...


@dataclass(frozen=True, slots=True)
class ControllerProvenance:
    """What the controller says about itself: evidence a person can judge, never a guess.

    ``is_simulator`` is three-valued and never ``False``. A driver sets it ``True`` only where the pair it holds
    proves a simulator; anything short of proof stays ``None``, because "we could not tell" and "it is a real
    controller" are different answers, and a console that renders them alike tells an operator a physical arm is
    connected when nothing checked. Measured 2026-08-19: a URSim UR3e rendered as a physical arm, because the only
    flag on the panel was ``is_simulated``, which is about the driver and is rightly ``False`` for URSim.

    The evidence itself is reported whether or not it proves anything, so a person can read the serial and the
    address and judge them.

    Attributes:
        serial (str | None): The controller's own serial number, ``None`` where it did not answer (default: None).
        host (str | None): The address the driver talks to (default: None).
        is_simulator (bool | None): ``True`` only where the driver proved a simulator, ``None`` otherwise; never
            ``False`` (default: None).
    """

    serial: str | None = None
    host: str | None = None
    is_simulator: bool | None = None

    def __post_init__(self) -> None:
        if self.is_simulator is False:
            raise ValueError(
                "ControllerProvenance.is_simulator is True or None: absence of proof is absence of a claim, "
                "not a claim that the controller is real."
            )

    def says_nothing(self) -> bool:
        """Whether no evidence was gathered at all, so a caller adds no keys rather than empty ones.

        Returns:
            bool: ``True`` where serial, host and verdict are all unknown.
        """
        return self.serial is None and self.host is None and self.is_simulator is None

    def render(self) -> str:
        """Describe this to a person.

        Returns:
            str: One line, ASCII, no trailing newline.
        """
        verdict = "simulator (proven)" if self.is_simulator else "simulator unproven"
        return f"serial {self.serial or 'unknown'}, host {self.host or 'unknown'}, {verdict}"


@runtime_checkable
class ReportsControllerProvenance(Protocol):
    """Capability extension: the arm asks its controller what it is, and never raises doing it.

    The console reached for ``arm._conn`` to answer this, which is a UR attribute: a second vendor would have
    rendered "cannot tell" forever, and nothing would have said so. The rule that turns evidence into proof is the
    controller's own, so it lives in the driver that knows the controller.
    """

    def controller_provenance(self) -> ControllerProvenance:
        """What this arm's controller says about itself. Best effort; an unreachable controller says nothing."""
        ...


@runtime_checkable
class SupportsRobotStatus(Protocol):
    """Capability extension: read the live robot and safety state, and let a person acknowledge a fault.

    This lets a pipeline say a controller fault (:class:`~src.robot.core.errors.ArmFault`, with the safety mode and the
    controller's own text) instead of a bare motion rejection. Clearing it is a person's action and never the stack's.
    """

    def get_robot_status(self) -> RobotStatus:
        """A snapshot of the controller's robot mode, safety mode and stop flags."""
        ...

    def acknowledge_fault(self, fault: "ArmFault") -> bool:
        """A person acknowledges ``fault`` at this controller; ``True`` where it acknowledged.

        Recovery is never automatic (the owner, 2026-09-19): this is reached from a script run at the cell, where whoever
        clears a stop can see the arm, and a boundary test forbids a call from the execution, grasping and safety layers
        and from the console. A fault whose ``acknowledgeable`` is ``False`` answers ``False``.
        """
        ...


#: What a halt did to the move in flight, :attr:`HaltState.brake`. ``none``: no move was in flight. ``pending``: one was,
#: and it has not ended yet (being braked, or running to its end with the brake off). ``braked``: braked under control,
#: and the arm stood still. ``unconfirmed``: braked, stopped where it could not be watched, or ended in a fault (the
#: brake off included), and the arm was not seen to stand still; if it still moves, the emergency stop is the answer.
#: ``ran_out``: it ended without a brake: ran to
#: its end with the brake off, had already finished when the halt came, or the controller stopped it.
HALT_BRAKE_OUTCOMES: frozenset[str] = frozenset({"none", "pending", "braked", "unconfirmed", "ran_out"})


@dataclass(frozen=True, slots=True)
class HaltState:
    """The arm's halt latch while it is set: why, when, and what became of the move that was in flight.

    ``requested_at`` is Unix seconds. ``in_motion`` says a move was in flight when the halt was requested. ``braked``
    says that move was braked under control (``stopJ``/``stopL`` on a UR), and
    ``brake_s`` how long from the request until the arm stood still; ``False`` and ``None`` where no move was braked,
    because none was in flight or because the arm lets the move in flight run to its end (the dummy).
    ``brake`` says what became of that move as far as it is known now (:data:`HALT_BRAKE_OUTCOMES`), so a reader can
    tell a brake still in progress from one that failed and from a move that ended without one; left out, it follows
    from the other fields. The thread that moved the arm replaces the record once its move ended, so a reader never
    sees half of one.
    """

    reason: str
    requested_at: float
    in_motion: bool = False
    braked: bool = False
    brake_s: float | None = None
    brake: str = ""

    def __post_init__(self) -> None:
        if not self.brake:
            derived = "braked" if self.braked else "pending" if self.in_motion else "none"
            object.__setattr__(self, "brake", derived)
        elif self.brake not in HALT_BRAKE_OUTCOMES:
            raise ValueError(f"HaltState.brake {self.brake!r} is none of {sorted(HALT_BRAKE_OUTCOMES)}")

    def render(self) -> str:
        """One line for a person: why the arm is halted, and what became of the move in flight."""
        if self.brake == "braked":
            took = f" in {self.brake_s:.2f} s" if self.brake_s is not None else ""
            motion = f"the move in flight was braked under control and the arm stood still{took}"
        elif self.brake == "unconfirmed":
            # Braked and never seen to stand, stopped where it could not be watched, or ended in a fault with the brake
            # off: what all of them share is that nobody saw the arm stand still.
            motion = ("the arm was not seen to stand still after the move in flight: if it still moves, press the "
                      "emergency stop")
        elif self.brake == "ran_out":
            motion = "the move in flight ended without a brake, and nothing after it is sent"
        elif self.brake == "pending":
            motion = "the move in flight has not ended yet, and nothing after it is sent"
        else:
            motion = "no move was in flight"
        return f"halted ({self.reason}): {motion}"

    def to_dict(self) -> dict[str, object]:
        return {"reason": self.reason, "requested_at": self.requested_at, "in_motion": self.in_motion,
                "braked": self.braked, "brake_s": self.brake_s, "brake": self.brake}


class MoveEnd(StrEnum):
    """How the last move a link to a controller was asked for ended, for the sentence a refused verb says.

    Read by the thread that asked for the move, right after it returned, never across threads. The UR connection
    keeps it (``URConnection.last_move_end``); a transport that keeps the halt latch reports it
    (:class:`~src.robot.core.transport.HoldsAHaltLatch`). Here since P0e E7, where the UR connection defined it.
    """

    #: No move was asked for yet.
    NONE = "none"
    #: The move was sent as it always was, and the answer is the controller's own (brakes off, or a caller's asynchronous move).
    SENT = "sent"
    #: A watched move stood at its target: joints within 2e-3 rad, a line's TCP within 1 mm and 2e-3 rad.
    ARRIVED = "arrived"
    #: The halt latch was set: nothing was sent.
    REFUSED_HALTED = "refused_halted"
    #: The halt braked the move in flight (stopJ or stopL on a UR) and the arm stood still.
    BRAKED = "braked"
    #: The halt's brake was sent and the arm was not seen to stand still, or the brake itself failed.
    BRAKE_UNCONFIRMED = "brake_unconfirmed"
    #: A watched move ended away from its target with nobody braking it.
    ENDED_SHORT = "ended_short"
    #: A protective or emergency stop, or the end of the control program, ended a watched move.
    STOPPED = "stopped"
    #: The controller refused to start the move, or never showed it running; a move it never showed was stopped.
    NOT_STARTED = "not_started"


#: The ends of a move the halt explains: a verb that meets one answers CANCELLED, not a controller refusal.
HALT_ENDS: frozenset[MoveEnd] = frozenset({MoveEnd.REFUSED_HALTED, MoveEnd.BRAKED, MoveEnd.BRAKE_UNCONFIRMED})


@runtime_checkable
class SupportsHalt(Protocol):
    """Capability extension: the halt latch ("halt now"); methods only, so a check reads no property.

    ``halt(reason)`` latches the arm and sends nothing from the thread that calls it: from then on every motion is
    refused with nothing sent and every output switch is refused (:class:`ArmHalted`), and only :meth:`clear_halt`,
    called once a person has confirmed the cell is clear, gives them back. It never raises, and a second halt keeps
    the first record. :meth:`halt_state` is the latch while it is set: a lock-free read that never raises, because the
    console reads it on every poll.

    What becomes of a move already in flight is the arm's: the UR arm brakes it under control; the dummy has nothing in
    flight.
    :func:`brakes_in_motion_of` says which.
    """

    def halt(self, reason: str) -> HaltState:
        """Latch the arm; the record of the latch, the first one where it was set already."""
        ...

    def clear_halt(self) -> None:
        """End the latch: motion and outputs are given back. Called once a person confirmed the cell is clear."""
        ...

    def halt_state(self) -> HaltState | None:
        """The latch while it is set, else ``None``. Never raises, never blocks."""
        ...


class ArmHalted(RobotMotionRejected):
    """Raised where a halted arm is asked to switch an output, or to move by a verb that raises: nothing was sent.

    A :class:`RobotMotionRejected`, so a caller that handles a refused motion handles it too.
    """


def halted_refusal(reason: str, what: str) -> str:
    """The sentence a halted arm refuses with: why it is halted, what was (not) sent, and what ends it.

    ``what`` says what became of the command, such as "nothing was sent to the controller". A halt is not a controller
    stop, so it never says to clear one: a person confirms the cell is clear, which ends the latch, and the way back
    is a Restart.
    """
    return f"the arm is halted ({reason}): {what}; a person confirms the cell is clear, then Restart"


def end_pulse(io: SupportsDigitalIO, pin: int, *, port: DigitalIOPort) -> None:
    """End a pulse on ``io``'s output ``pin``: drive it low, however the pulse ended.

    For a bistable valve's coil and a vacuum blow-off, which are pulsed high and must go back low: a coil left energised
    until the cell is cleared is what cooks it, and a blow-off left on blows until then. Never for a toggle hand, whose
    every change of its output moves the jaws. The plain write, as always; where a halt refused it (:class:`ArmHalted`,
    nothing was sent), through the arm's own door for exactly this, ``end_output_pulse``: the one write a halted arm
    makes, and only ever low. An arm without that door raises the refusal, as the plain write did.
    """
    try:
        io.set_digital_output(pin, False, port=port)
    except ArmHalted:
        end = getattr(io, "end_output_pulse", None)
        if not callable(end):
            raise
        end(pin, port=port)


def halt_state_of(arm: object) -> HaltState | None:
    """The halt latch of ``arm`` while it is set, or ``None`` for an arm that is not halted or has no latch.

    Strict: only a :class:`HaltState` counts, so a double that answers every attribute reads as not halted. A latch
    that cannot be read is no latch here; every gate that refuses a halted arm reads it this way, and every UR verb
    refuses on its own connection's latch as well.
    """
    if not isinstance(arm, SupportsHalt):
        return None
    try:
        state = arm.halt_state()
    except Exception:  # noqa: BLE001 (a halt_state that raises is a broken double, never a latch to obey)
        return None
    return state if isinstance(state, HaltState) else None


@runtime_checkable
class BrakesWhenHalted(Protocol):
    """Capability extension: the arm says whether a halt brakes the move in flight, or lets it run to its end.

    Beside :class:`SupportsHalt` and not in it, because a latch is the same question on every arm and what becomes of
    the move in flight is not: the UR brakes it under control, the dummy has nothing in flight. The console says which on its brake panel (:func:`brakes_in_motion_of`).
    """

    def brakes_in_motion(self) -> bool:
        """Whether a halt brakes a move in flight here, rather than letting it run to its end."""
        ...


@runtime_checkable
class StepsItsOwnPhysics(Protocol):
    """Capability extension: the arm's ``wait_until_steady`` steps its simulation itself (the Isaac sim).

    Nothing waits on such an arm for rest before a motion (``execution.motion.steady_timeout_of``): as a gate inside the
    pick loop the wait fights the loop's own stepping, and the picks fail. On a real arm the same call reads the
    controller and steps nothing. The sim profile's ``safety.dwell`` said so until 2026-10-17 (the owner's cleanup).
    """

    @property
    def steps_its_own_physics(self) -> bool:
        """``True`` on such an arm."""
        ...


def brakes_in_motion_of(arm: object) -> bool:
    """Whether ``arm`` brakes a move in flight when it is halted (every UR).

    ``False`` for an arm that latches and lets the move in flight run to its end, and for an arm with no latch.
    """
    method = getattr(arm, "brakes_in_motion", None)
    if not callable(method):
        return False
    try:
        return method() is True
    except Exception:  # noqa: BLE001 (an arm that cannot say brakes nothing anybody can count on)
        return False


@runtime_checkable
class SupportsWatching(Protocol):
    """Capability extension: watch the controller without driving it (P0i, owner 2026-09-26).

    ``watch()`` opens a link on which the status reads and a person's acknowledgement answer and which opens on a
    controller in a protective stop, where a control program will not start; ``is_connected`` stays ``False`` on it, so
    every motion verb refuses with nothing sent, and ``disconnect()`` closes it. The fault door watches: a stop is what
    it is for, and a full connect is refused by exactly that stop (measured on URSim, 2026-09-26).
    """

    @property
    def is_watching(self) -> bool: ...

    def watch(self) -> None:
        """Open the watch; a no-op on an arm that is connected or already watching."""
        ...


class LineMotion(StrEnum):
    """What a ``move(pose, linear=True)`` on an arm keeps of the line it was asked for."""

    #: Every sample of the line is judged before any of it moves, and the arm drives it.
    CHECKED = "checked"
    #: The controller draws the line, and only its end is judged.
    CONTROLLER_LINE = "controller_line"
    #: No controller: the pose is set.
    TELEPORT = "teleport"
    #: The line is not kept, because the flag is dropped or the path cannot be judged. The
    #: reading says which.
    NOT_KEPT = "not_kept"


@dataclass(frozen=True, slots=True)
class LineReading:
    """A line motion and the sentence that says why."""

    motion: LineMotion
    reason: str


@runtime_checkable
class KeepsLines(Protocol):
    """Capability extension: the arm says, before it moves, what it keeps of a straight line."""

    def line_motion(self) -> LineReading:
        """What a ``move(pose, linear=True)`` on this arm keeps of the line, and why."""
        ...


@runtime_checkable
class HomesTyped(Protocol):
    """Capability extension: the arm goes home as a typed verb, and says which gate refused it.

    ``move_home`` answers with a bool on every driver, so every refusal it can meet reads the same
    ``False``. An arm that implements this answers the same move with the
    :class:`~src.robot.core.motion_result.MotionResult` its gates produced: the status, the sentence
    and the home configuration. Its ``move_home`` is this verb's ``ok``.

    It is not a member of :class:`~src.robot.core.RobotArm` on purpose. The drivers subclass that
    Protocol explicitly, and a class that does inherits every member it does not define as a stub
    that returns ``None``, so a new member there would answer ``None`` on every driver that does not
    implement it.
    """

    def move_to_home(self) -> "MotionResult":
        """Move to the configured home, gated, and return the typed result of that one move."""
        ...

    @property
    def home_joint_positions(self) -> "Sequence[float]":
        """The joints :meth:`move_to_home` drives to, in radians. The joint-limit guard centres a cable's
        half-turn window on them (every real arm's)."""
        ...


@runtime_checkable
class ScreensJointGoals(Protocol):
    """Capability extension: the arm says what a goal it plans to is screened against before any motion.

    The workspace box its guard holds a grasp centre to and that guard's margin, the joint window its planner
    chooses a goal configuration in, and where its tool stands at a joint vector by its own chain. A calibration
    station is screened by the numbers the arm will judge it with, instead of by a copy of its configuration.
    """

    @property
    def workspace_limits(self) -> Any:
        """The box the workspace guard holds every grasp centre to (``robot.motion.workspace``), millimetres."""
        ...

    @property
    def workspace_margin_mm(self) -> float:
        """How far inside that box the guard keeps a grasp centre, millimetres."""
        ...

    def goal_joint_window(self) -> "tuple[Sequence[float], Sequence[float]]":
        """Where a chosen goal's joints may lie, radians: the planner's window narrowed to what the guard admits."""
        ...

    def tcp_at_joints_mm(self, joints: Any) -> "np.ndarray | None":
        """The TCP at ``joints`` as a 4x4 in millimetres by the arm's own chain and tool frame, or ``None``."""
        ...


@runtime_checkable
class ChoosesConfigurations(Protocol):
    """Capability extension: the arm names the joint configuration a pose goes to, unmoved.

    ``RobotArm.ik`` answers with whatever configuration a solver lands on, and says nothing
    about whether a motion there would be allowed. An arm that implements this answers the
    question a caller asks before it spends a motion: the configuration this arm would choose
    for ``pose`` (a TCP pose in BASE), as its own gates judge it. That is its inverse
    kinematics, the joint window it chooses in (on a cell with a cable, half a turn either
    side of home), the branch it holds first, and the endpoint gate: the workspace box on the
    TCP, the joint limits, self-collision and the payload. The generated view of a wrist pick
    screens every angle of its orbit this way, so only an angle the arm can stand at costs a
    judged motion.

    Nothing moves, no camera world is refreshed, and nothing is remembered as commanded: the
    motion to the answer is still judged whole, its path and the world included, when it
    runs. A pose with no admissible configuration raises
    :class:`~src.robot.core.errors.RobotKinematicsError`, whose message names what refused
    it: the inverse kinematics (out of reach), the window, or the gate. An arm that cannot
    read where it stands, or reads something that is no configuration of it, raises
    :class:`~src.robot.core.errors.RobotConnectionError`, which is no statement about the
    pose.

    It is not a member of :class:`~src.robot.core.RobotArm`, for the reason
    :class:`HomesTyped` gives; and an arm whose ``ik`` is a stand-in (the dummy's) does not
    implement it, so a stand-in never places a camera.
    """

    def nearest_configuration(self, pose: "Pose") -> "JointPositions":
        """The configuration ``pose`` goes to on this arm, as its gates judge it; raises if none."""
        ...


@runtime_checkable
class DrivesJointLines(Protocol):
    """Capability extension: the arm moves to joints on the straight joint line from where it stands, or not at all.

    ``RobotArm.move_to_joints`` runs the straight joint line where it is clear and, on an arm that plans, asks the
    planner for a way around it where it is not. The owner allows that for a taught pose and never for the motions a
    pick makes up on its own (2026-09-29): the one view a wrist pick generates, and its move back to the look that saw
    the part, must never drive through a retract pose or take a detour nobody taught. An arm that implements this runs
    such a motion as its ``move_to_joints`` runs a clear line, judged the same way against the same world, and refuses a
    line that is not clear with the refusal its judge gave, nothing sent and nothing planned around it. An arm whose
    paths nobody judges refuses every such motion: a line nobody judged is not the line that was allowed.

    It is not a member of :class:`~src.robot.core.RobotArm`, for the reason :class:`HomesTyped` gives. An arm that
    does not implement it has no motion a caller may take for a straight joint line alone.
    """

    def move_to_joints_on_the_line(self, joints: "JointPositions") -> "MotionResult":
        """Move to ``joints`` on the straight joint line from where the arm stands; refused, nothing sent, if not clear."""
        ...


@runtime_checkable
class DrivesJointPaths(Protocol):
    """Capability extension: the arm runs straight joint lines through several waypoints as ONE motion, judged whole.

    Every leg is judged before anything is sent, as :class:`DrivesJointLines` judges one line, against the world one
    refresh built; a leg that is not clear refuses the whole path, nothing sent and nothing planned around it. A path
    that passes is sent waypoint after waypoint with nothing judged or planned between them, so the arm turns at each
    and runs on: the wave at a greeting (the owner, 2026-10-07: "das muss eine konsistente Bewegung sein"), whose
    swings stood five seconds apart while each was judged on its own. Not a member of
    :class:`~src.robot.core.RobotArm`, for the reason :class:`HomesTyped` gives.
    """

    def move_through_joints_on_the_line(self, waypoints: "Sequence[JointPositions]") -> "MotionResult":
        """Run the straight joint lines through ``waypoints`` as one motion, judged whole first; refused, nothing sent,
        if a leg is not clear."""
        ...


@runtime_checkable
class HoldsItsWorld(Protocol):
    """Capability extension: the arm judges every line inside a block against the world its last refresh built, with no
    new frame.

    A grasp goes down a judged line to the part, and from there it closes and lifts, or backs out the way it came with
    the jaws open. The camera world a line refreshes at the part, seen from there, held boxes the one the line down was
    judged in did not, and the arm stood in a bin refused every way back up, both the lift and the line it had just come
    down (the grasp bench, 2026-10-06). The owner's answer: the way back up from the part is judged against the world
    that let the line down run, which is the world the last refresh built (2026-10-06: "Ja, beides"). Inside the block
    every motion of the arm is judged against that world and stamped with the refresh that built it; an arm with no
    world refreshed yet refreshes as ever. It is not a member of :class:`~src.robot.core.RobotArm`, for the reason
    :class:`HomesTyped` gives.
    """

    def held_world(self, reason: str) -> "AbstractContextManager[None]":
        """A block in which every line is judged against the world the last refresh built; ``reason`` says why."""
        ...


@runtime_checkable
class JudgesCarriedLines(Protocol):
    """Capability extension: the arm judges the lift a grasp will make, at the part, before the jaws close on it.

    A grasp closes at the part, then lifts it on a straight line. A lift refused once the jaws hold the part would hold
    the arm there, the part in its jaws, every way out refused at its first sample. So a grasp asks first, at the part
    with its jaws still open, whether the lift would run, and closes only where it would; where not, it backs out
    empty-handed on the line it came in on (the owner, 2026-10-01). No part in the jaws is modelled (the owner,
    2026-10-05, every cell since 2026-10-17): a closed hand is judged as an empty one, so the lift is judged as it will
    run. Asking moves nothing and commands nothing. It carries the decline the lift would carry (``camera_world``), as
    every motion of the grasp does.

    It is not a member of :class:`~src.robot.core.RobotArm`, for the reason :class:`HomesTyped` gives. A grasp on an arm
    that does not implement it (the dummy, the sim) closes and lifts as before.
    """

    def carried_line_refusal(self, pose: "Pose", *,
                             camera_world: "Maybe[CameraWorldDecline]" = UNSET) -> "MotionResult | None":
        """The refusal ``move(pose, linear=True, camera_world=camera_world)`` would meet from where the arm stands once
        the jaws close there, judged now; ``None`` where that line would run. Nothing moves."""
        ...


@runtime_checkable
class JudgesGraspsAhead(Protocol):
    """Capability extension: the arm judges a whole grasp from where it stands, before it leaves for it.

    A grasp is a planned move to its standoff, a line down to the part and a lift, and each is judged as it runs: a
    grasp the guard or the planner refuses at the part has taken the arm there first. Asked ahead, each of the three is
    judged once, in the order that judges each once (the judge chain, 2026-10-08): the line down first, from the
    configuration the move to the standoff would end on (:class:`ChoosesConfigurations`), before any route is planned;
    then the lift from the grasp (:class:`JudgesCarriedLines`), in the world the line down was judged in; the move to
    the standoff from where the arm stands last, because a route costs seconds and a line refused at a part on the
    flange is every configuration's refusal. Where the route ends on the configuration the line was judged from, each
    was judged once, and an arm may run that route as it was judged where nothing it was judged on changed since (the UR
    driver does); where it ends anywhere else, the line and the lift are judged again from where it ends, each from
    where the one before ends. A pick asks it of each grasp it may take, a blocker's (the owner, 2026-10-03) and the
    part's (URSim, 2026-10-03: a lift the cable window refused once the hand stood at the part), and drives to the first
    every judgement admits. Asking moves nothing and commands nothing.

    It is not a member of :class:`~src.robot.core.RobotArm`, for the reason :class:`HomesTyped` gives. An arm that
    does not implement it has its grasps judged as they run, as before.
    """

    def grasp_refusal_ahead(self, *, standoff: "Pose", grasp: "Pose", lift: "Pose") -> str:
        """Why the grasp would be refused before its jaws close, judged now from where the arm stands; ``""`` where every
        judgement passes, and where this arm cannot judge it ahead. Nothing moves."""
        ...


@runtime_checkable
class JudgesLinesAhead(Protocol):
    """Capability extension: the arm judges a planned move and the straight lines after it from where it stands.

    A push is a planned move to P0 above the part and four judged straight lines (down, push, back, up), each judged as
    it runs: a push leg the guard refuses once the arm came down beside the part leaves the arm standing there, a person
    to decide. Asked ahead, the move to ``approach`` is judged as :meth:`move` would route it, and each line from where
    the one before ends, every one in the world it would run in (URSim, 2026-10-03). Asking moves nothing.

    It is not a member of :class:`~src.robot.core.RobotArm`, for the reason :class:`HomesTyped` gives. An arm that
    does not implement it has its lines judged as they run, as before.
    """

    def lines_refusal_ahead(self, *, approach: "Pose", lines: "Sequence[Pose]") -> str:
        """Why the move to ``approach`` or one of ``lines`` after it would be refused, judged now from where the arm
        stands; ``""`` where every judgement passes, and where this arm cannot judge them ahead. Nothing moves."""
        ...


@runtime_checkable
class JudgesNextLegs(Protocol):
    """Capability extension: the arm judges the next leg while the jaws' stroke is waited out (2026-10-08).

    A grasp closes, a place opens, and the jaws take their stroke while the arm stands. On an arm that judges the next
    leg (every cuRobo arm), a verb has the hand leave the stroke to it (``set_closed(wait=False)``) and asks the arm
    meanwhile: the line out of a place (:meth:`judge_line_ahead`), and the joint move declared next
    (:meth:`expecting_next`, :meth:`judge_the_next_leg`), the carry to a bin's look after a pick and the return after a
    place. A leg judged ahead runs as it was judged where nothing it was judged on changed. Judging moves nothing. It is
    not a member of :class:`~src.robot.core.RobotArm`, for the reason :class:`HomesTyped` gives; an arm that does not
    implement it, or reads ``judges_next_legs`` false, has every leg judged as it runs.
    """

    @property
    def judges_next_legs(self) -> bool:
        """Whether a verb judges the next leg while the jaws' stroke is waited out."""
        ...

    def expecting_next(self, joints: "JointPositions") -> "AbstractContextManager[None]":
        """A block within which the joint move to ``joints`` is the leg after the verbs running in it."""
        ...

    def judge_the_next_leg(self, after: "Pose", *, junction: str, now: bool = True) -> str:
        """Judge the declared joint move from where the line to ``after`` leaves the arm and keep it for that move;
        ``""`` where it was kept or left to the line, else why not. Nothing moves."""
        ...

    def judge_line_ahead(self, pose: "Pose") -> "MotionResult | None":
        """Judge the line to ``pose`` from where the arm stands, now, and keep it for that one line; ``None`` where it
        would run, else the refusal it would meet. Nothing moves."""
        ...


@runtime_checkable
class HoldsItsWorldAtJunctions(Protocol):
    """Capability extension: the arm holds its world across a junction of a grasp or a place, rather than taking a new
    frame there: at the ``standoff``, through the ``carry`` and through the ``drop`` (the owner's O2 of 2026-10-09, every
    cell since 2026-10-17).

    Beside :class:`HoldsItsWorld`, which holds the world for a block a caller names: these say where the arm holds it,
    and hold the approach a grasp judged ahead only where its route stands ready to run as judged. It
    is not a member of :class:`~src.robot.core.RobotArm`, for the reason :class:`HomesTyped` gives; an arm that does not
    implement it refreshes at every junction, as before.
    """

    def holds_world_at(self, junction: str) -> bool:
        """Whether this arm holds its world across ``junction``, where a live camera world is wired."""
        ...

    def holding_the_approach(self, standoff: "Pose", reason: str) -> "AbstractContextManager[bool]":
        """A block that holds the world the grasp judged ahead to ``standoff`` was judged in; yields whether it holds."""
        ...


@runtime_checkable
class JudgesTheLineDown(Protocol):
    """Capability extension: the arm judges the line down from a standoff to a grasp in the world it has, with no new
    frame, from the configuration the route to the standoff ends on.

    A pick asks it of the next grasp of a look once the one before was refused before anything was sent (the owner,
    2026-10-12: the guard's boxes decide what the next grasp is spared). Asking moves nothing. It is not a member of
    :class:`~src.robot.core.RobotArm`, for the reason :class:`HomesTyped` gives; a pick on an arm that does not implement
    it judges the next grasp whole, as before.
    """

    def line_down_refusal(self, *, standoff: "Pose", grasp: "Pose") -> str:
        """Why the line down from ``standoff`` to ``grasp`` would be refused, or ``""`` where it would run, where
        nothing is judged ahead, and where no configuration reaches the standoff. Nothing moves."""
        ...


@runtime_checkable
class KeepsItsBranch(Protocol):
    """Capability extension: the arm keeps the branch it holds for the Cartesian moves of a block (the owner's R2 of
    2026-10-08: a change of branch is the last resort).

    A pick tries every grasp of a look inside such a block first, and leaves the grasps it refuses for their branch until
    after them; it reads, on the closed form alone, whether a pose has a configuration on the arm's branch at all. It is
    not a member of :class:`~src.robot.core.RobotArm`, for the reason :class:`HomesTyped` gives; a pick on an arm that
    does not implement it tries its grasps in their order, as before.
    """

    def keeping_its_branch(self, overshoot_deg: float = 15.0) -> "AbstractContextManager[Any]":
        """A block in which every Cartesian move keeps the branch the arm holds; yields what the block refused for it."""
        ...

    def has_a_goal_on_its_branch(self, pose: "Pose", here: "Sequence[float]") -> "bool | None":
        """Whether ``pose`` has a configuration on the branch the arm holds at ``here``; ``None`` where it cannot tell."""
        ...


@runtime_checkable
class ScreensConfigurations(Protocol):
    """Capability extension: the arm says what the exact guard and the planner make of joints before any move goes there.

    A pose is screened where it is taught, where a campaign starts and at the desk (the owner, 2026-09-30), so a pose
    only the planner's padded spheres refuse is known before a pick meets it. The verdict is a
    ``safety.planning.band.PoseScreen``; ``core`` sits below ``safety`` and types it loosely rather than importing
    upward. It is not a member of :class:`~src.robot.core.RobotArm`, for the reason :class:`HomesTyped` gives; a pose on
    an arm that does not implement it is judged when a motion goes there.
    """

    def screen_configuration(self, joints: "JointPositions", *, ask_planner: bool = True) -> Any:
        """What the guard and the planner say about ``joints``, as a ``PoseScreen``. Nothing moves."""
        ...


@runtime_checkable
class ReadsItsStatusQuickly(Protocol):
    """Capability extension: the arm reads the controller's state from what it already receives, with no round trip.

    :meth:`SupportsRobotStatus.get_robot_status` may ask the controller for its text on every call (the UR's dashboard);
    a ready bar that polls asks this instead, whose ``message`` stays empty. Beside that Protocol and not in it, because
    a status read is every status arm's and a cheap one is not.
    """

    def quick_robot_status(self) -> RobotStatus:
        """The controller's fields and the halt latch, from the receive stream; raises on a closed connection."""
        ...


@runtime_checkable
class KnowsItsHand(Protocol):
    """Capability extension: the arm is handed the hand it carries, to read and never to command.

    What the arm judged ahead runs as it was judged only while this hand reads as it did then
    (``core.gripper.why_not_known_open``). ``connect_cell`` hands it once both are up. The camera's boxes are set aside
    whatever it reads: no cell models a part in the jaws (the owner, 2026-10-05; every cell since 2026-10-17).
    """

    def set_hand(self, gripper: object | None) -> None:
        """Hand this arm the hand it carries; ``None`` for none."""
        ...


@runtime_checkable
class HoldsItsFingersAsOpened(Protocol):
    """Capability extension: the arm's guard holds the hand's fingers as wide as the hand was told to open them.

    A grasp planned for jaws that open only as needed (lever G, a Robotiq hand, 2026-10-13) is judged and driven with the
    fingers where the hand reads them back, and not at the full stroke they were baked at (``GraspExecutionPolicy``).
    An arm that does not implement this judges them fully open, which is never nearer the part than they are.
    """

    def set_finger_opening(self, width_mm: float | None) -> None:
        """Hold the fingers ``width_mm`` apart between their inner faces in this arm's guard; ``None`` fully open."""
        ...


@runtime_checkable
class ReportsWristBodies(Protocol):
    """Capability extension: the arm says which wrist cameras it carries, as its guard holds them.

    The pick places their housing on each grasp to choose its wrist turn (``wrist_turn``). Beside
    :class:`CarriesWristBodies`, which hands them in, because an arm that takes them may hold them in a planner it does
    not show. The bodies are ``safety.planning.body_link.WristBody`` values, typed loosely as there.
    """

    def wrist_bodies(self) -> "tuple[Any, ...]":
        """The wrist cameras this arm carries; empty for none."""
        ...


@runtime_checkable
class KnowsItsFlange(Protocol):
    """Capability extension: the arm says where its controller puts the flange at any joints, on the controller's own
    kinematics.

    A controller calibrated at the factory computes its poses on kinematics of its own, and puts its flange millimetres
    from where the arm's nominal table puts it at the same joints: 6.0 to 6.3 mm and 0.6 degrees on the owner's UR10 CB3
    (2026-10-08). A camera on the arm is placed by the pose the controller reports, so the world it sees is the
    controller's. The exact guard places what hangs from the flange, the hand, its fingers and a wrist camera, where this
    says, and the links by the nominal table (the owner, 2026-10-12: everything at the flange), so a hand is judged where
    it stands in the world the camera saw. A UR reads its controller's rows from its primary interface; an arm whose
    controller hands out its model, or a description calibrated against it, can say the same. It is not a member of
    :class:`~src.robot.core.RobotArm`, for the reason :class:`HomesTyped` gives. An arm that does not implement it, or
    cannot say now, has its flange placed by the nominal table, as before.
    """

    def controller_flanges_mm(self, joints_rad: "np.ndarray") -> "np.ndarray | None":
        """The flange at every row of ``joints_rad``, ``(N, dof)`` radians, as the controller computes it: ``(N, 4, 4)``,
        the rotation unitless and the translation in millimetres, in the controller's base frame, each row answered alone
        as it is answered among others; ``None`` where this arm cannot say now."""
        ...


@runtime_checkable
class SharesItsCell(Protocol):
    """Capability extension: the arm shares its cell with other arms (owner, 2026-10-15).

    The cell puts its sends behind a gate that reserves the space each send needs (``core.cell_gate``), hands it the
    other arms where they stand or move, for its guard, its planner and its cameras' self filter, and reads its own body
    for theirs: where it stands (no part in its jaws: no cell models one). An arm that does not implement it stays in a
    cell of its own.
    """

    def join_cell(self, gate: "CellGate") -> None:
        """Put every send of this arm behind ``gate``, before its planner is built."""
        ...

    def leave_cell(self) -> None:
        """Take this arm out of its cell again: no gate before its sends, no other arm held."""
        ...

    def hold_partners(self, partners: Any) -> None:
        """Hold the other arms of the cell (a ``safety.partner_arm.PartnerWorld``), or none for ``None``."""
        ...

    def reserve_partner_slots(self, count: int) -> None:
        """Reserve room in the planner's world for ``count`` boxes of the other arms, before it starts."""
        ...

    def own_envelope(self) -> Any:
        """This arm's own body where it stands, as a self filter takes it out of a view, or ``None``."""
        ...


def controller_flanges_of(arm: object, joints_rad: "np.ndarray") -> "np.ndarray | None":
    """Where ``arm``'s controller puts its flange at every row of ``joints_rad`` (:class:`KnowsItsFlange`), or ``None``
    for an arm that does not say or cannot now.

    Strict: only ``(N, 4, 4)`` finite numbers for the ``N`` rows asked count, so a double that answers every attribute
    says nothing here, and neither does an answer that raises.
    """
    if not isinstance(arm, KnowsItsFlange):
        return None
    rows = np.asarray(joints_rad, dtype=np.float64)
    try:
        flanges = np.asarray(arm.controller_flanges_mm(rows), dtype=np.float64)
    except Exception:  # noqa: BLE001 (an arm that cannot say has its flange placed by the nominal table)
        return None
    if rows.ndim != 2 or flanges.shape != (rows.shape[0], 4, 4) or not bool(np.isfinite(flanges).all()):
        return None
    return flanges


def line_motion_of(arm: object) -> LineReading | None:
    """What ``arm`` keeps of a straight line, or ``None`` for an arm that does not say."""
    if isinstance(arm, KeepsLines):
        return arm.line_motion()
    return None
