"""Optional capability extensions for arm drivers.

:class:`~src.robot.core.RobotArm` stays a small, strictly vendor-neutral Protocol
with a contract-locked member set. The extras a controller can offer (digital and
analog I/O, live force and torque, live robot and safety status, what a straight line
keeps, a model of the carried part, a move home that says why it was refused) are
declared here as separate ``runtime_checkable`` Protocols, mirroring
:class:`~src.robot.core.gripper.ObjectDetectingGripper`. Hand guiding follows the same
pattern from its own module, :mod:`~src.robot.core.freedrive`: an arm a person may move by
hand advertises ``SupportsFreedrive`` (the UR driver, on teach mode), and one that does not,
the sim arm among them, keeps the calibration stations it drives to itself.

A driver opts in by implementing one. A caller feature-checks with
``isinstance(arm, SupportsForceTorque)`` and falls back where the capability is
absent, so a driver that implements none keeps working and nothing is added to the
neutral :class:`RobotArm` surface.

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

from src.geometry import Frame

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from .motion_result import MotionResult

__all__ = [
    "CarriesPayload",
    "DigitalIOPort",
    "HomesTyped",
    "KeepsLines",
    "LineMotion",
    "LineReading",
    "PayloadModel",
    "RobotMode",
    "RobotStatus",
    "SafetyMode",
    "SupportsDigitalIO",
    "SupportsForceTorque",
    "SupportsRobotStatus",
    "Wrench",
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
    """A snapshot of the controller's live robot and safety state.

    ``message`` carries the controller's own text, such as the UR dashboard
    ``safetystatus`` string, so an operator reads a fault rather than a code.
    """

    robot_mode: RobotMode
    safety_mode: SafetyMode
    protective_stopped: bool
    emergency_stopped: bool
    message: str = ""

    @property
    def is_operational(self) -> bool:
        """True only when the arm is powered, running, in NORMAL safety and not stopped."""
        return (
            self.robot_mode == RobotMode.RUNNING
            and self.safety_mode == SafetyMode.NORMAL
            and not self.protective_stopped
            and not self.emergency_stopped
        )

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


def line_motion_of(arm: object) -> LineReading | None:
    """What ``arm`` keeps of a straight line, or ``None`` for an arm that does not say."""
    if isinstance(arm, KeepsLines):
        return arm.line_motion()
    return None
