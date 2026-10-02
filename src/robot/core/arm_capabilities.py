"""Optional capability extensions for arm drivers.

:class:`~src.robot.core.RobotArm` stays a small, strictly vendor-neutral Protocol
with a contract-locked member set. The extras a controller can offer (digital and
analog I/O, live force and torque, live robot and safety status, what a straight line
keeps, a model of the carried part, a move home that says why it was refused, the
configuration a pose goes to, a joint move on its straight joint line and nothing else, a line judged as if the
jaws held a part) are declared here as
separate ``runtime_checkable`` Protocols,
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
``getRobotMode`` and ``getSafetyMode`` integers, so a KUKA or Franka driver can
implement the same Protocols against its own controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from src.contracts import UNSET
from src.geometry import Frame

from .errors import RobotMotionRejected

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.contracts import Maybe
    from src.geometry import Pose

    from .camera_world import CameraWorldDecline
    from .joint_positions import JointPositions
    from .motion_result import MotionResult

__all__ = [
    "ArmHalted",
    "CarriesPayload",
    "ChoosesConfigurations",
    "DigitalIOPort",
    "DrivesJointLines",
    "HALT_BRAKE_OUTCOMES",
    "HaltState",
    "HomesTyped",
    "JudgesCarriedLines",
    "KeepsLines",
    "LineMotion",
    "LineReading",
    "PayloadModel",
    "RobotMode",
    "RobotStatus",
    "SafetyMode",
    "SupportsDigitalIO",
    "SupportsForceTorque",
    "SupportsHalt",
    "SupportsRobotStatus",
    "Wrench",
    "brakes_in_motion_of",
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
    """

    robot_mode: RobotMode
    safety_mode: SafetyMode
    protective_stopped: bool
    emergency_stopped: bool
    message: str = ""
    halted: str = ""

    @property
    def controller_operational(self) -> bool:
        """The controller's own answer: powered, running, in NORMAL safety and not stopped. The halt latch aside.

        What every console gate reads, after it read the latch: a halted arm on a healthy controller is halted, and
        must never read as a protective stop, whose remedy (the pendant) is not the halt's (a person confirms the cell
        is clear).
        """
        return (
            self.robot_mode == RobotMode.RUNNING
            and self.safety_mode == SafetyMode.NORMAL
            and not self.protective_stopped
            and not self.emergency_stopped
        )

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
class SupportsRobotStatus(Protocol):
    """Capability extension: read the live robot and safety state, and clear a stop.

    This lets a pipeline surface a controller fault with its safety mode and text
    instead of a bare motion rejection, and clear a protective stop so the cell
    resumes work.
    """

    def get_robot_status(self) -> RobotStatus:
        """A snapshot of the controller's robot mode, safety mode and stop flags."""
        ...

    def recover_from_protective_stop(self) -> bool:
        """Try to clear an active protective stop. ``True`` if the controller acknowledged."""
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
    says that move was braked under control (``stopJ``/``stopL`` on a UR with ``robot.ur.brake_on_halt``), and
    ``brake_s`` how long from the request until the arm stood still; ``False`` and ``None`` where no move was braked,
    because none was in flight or because the arm lets the move in flight run to its end (brakes off, the dummy).
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


@runtime_checkable
class SupportsHalt(Protocol):
    """Capability extension: the halt latch ("halt now"); methods only, so a check reads no property.

    ``halt(reason)`` latches the arm and sends nothing from the thread that calls it: from then on every motion is
    refused with nothing sent and every output switch is refused (:class:`ArmHalted`), and only :meth:`clear_halt`,
    called once a person has confirmed the cell is clear, gives them back. It never raises, and a second halt keeps
    the first record. :meth:`halt_state` is the latch while it is set: a lock-free read that never raises, because the
    console reads it on every poll.

    What becomes of a move already in flight is the arm's: the UR arm brakes it under control where
    ``robot.ur.brake_on_halt`` says so, and lets it run to its end otherwise; the dummy has nothing in flight.
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


def brakes_in_motion_of(arm: object) -> bool:
    """Whether ``arm`` brakes a move in flight when it is halted (the UR with ``robot.ur.brake_on_halt``).

    ``False`` for an arm that latches and lets the move in flight run to its end, and for an arm with no latch.
    """
    method = getattr(arm, "brakes_in_motion", None)
    if not callable(method):
        return False
    try:
        return method() is True
    except Exception:  # noqa: BLE001 (an arm that cannot say brakes nothing anybody can count on)
        return False


class PayloadModel(StrEnum):
    """What models the part the gripper carries now."""

    #: The planner carries the part and the self filter takes it out of what the cameras see.
    PLANNER_AND_FILTER = "planner_and_filter"
    #: The self filter takes the part out and the planner declined to carry it, so it routes as if
    #: the gripper were empty.
    FILTER_ONLY = "filter_only"
    #: Nothing models a part.
    NONE = "none"


@runtime_checkable
class CarriesPayload(Protocol):
    """Capability extension: the arm models a part in its gripper, and says why when it cannot.

    ``payload_declined_reason`` is the static answer, read off the arm's configuration and hand
    before any attach and without starting a planner. ``payload_model`` is what the last attach
    left in force.
    """

    def attach_payload(self, grip_width_mm: float) -> bool:
        """Model a part roughly ``grip_width_mm`` across. ``True`` when the planner carries it."""
        ...

    def detach_payload(self) -> bool:
        """Forget the carried part."""
        ...

    def payload_declined_reason(self) -> str | None:
        """Why this arm models no carried part, naming the key or the hand; ``None`` where it can."""
        ...

    def payload_model(self) -> PayloadModel:
        """What models the part the gripper carries now."""
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
class JudgesCarriedLines(Protocol):
    """Capability extension: the arm judges a straight line as if its jaws held a part, before they close on one.

    A grasp closes at the part, then lifts it on a straight line, and the lift is judged once the jaws hold the part. A
    part changes what an arm allows: on the UR driver the carried part is the planner's alone, and nothing of the
    camera's world is set aside while the hand holds one (Option 1). An empty hand admitted beside a bin the camera saw
    that closed there would hold the arm there, the part in its jaws, every way out refused at its first sample. So a
    grasp asks first, at the part with its jaws still open, whether the lift would run with the part in them, and
    closes only where it would; where not, it backs out empty-handed on the line it came in on (the owner, 2026-10-01).
    Asking moves nothing and commands nothing. It carries the decline the lift would carry (``camera_world``), as every
    motion of the grasp does.

    It is not a member of :class:`~src.robot.core.RobotArm`, for the reason :class:`HomesTyped` gives. A grasp on an arm
    that does not implement it (the dummy, the sim) closes and lifts as before.
    """

    def carried_line_refusal(self, pose: "Pose", *, grip_width_mm: float,
                             camera_world: "Maybe[CameraWorldDecline]" = UNSET) -> "MotionResult | None":
        """The refusal ``move(pose, linear=True, camera_world=camera_world)`` would meet from where the arm stands with a
        part about ``grip_width_mm`` across in its jaws, judged now; ``None`` where that line would run. Nothing moves."""
        ...


def line_motion_of(arm: object) -> LineReading | None:
    """What ``arm`` keeps of a straight line, or ``None`` for an arm that does not say."""
    if isinstance(arm, KeepsLines):
        return arm.line_motion()
    return None
