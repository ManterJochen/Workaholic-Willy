""":class:`NullGripper`, the explicit "no gripper attached" implementation.

It lets pipelines stay branch-free: rather than checking ``if gripper is None``, code
calls ``gripper.set_width_mm(...)`` and this implementation records the command
silently. A width query returns the configured nominal opening, which defaults to the
maximum. A command is accepted and reaches no hardware.

A NullGripper is also what a caller gets when a real one could not be built, and that
case has to be distinguishable from a cell that has no end-effector on purpose. A
config asking for a Robotiq on a non-UR arm, a vacuum on an arm with no digital I/O,
or a vendor with no driver each produced a working NullGripper and a log line, so the
cell connected, reported success, closed on nothing and lifted nothing.
:class:`GripperSubstitution` carries that fact on the object itself, where a caller
reads it without parsing logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from src.robot.core import Gripper

__all__ = ["GripperSubstitution", "NullGripper", "SubstitutionReason"]


class SubstitutionReason(StrEnum):
    """Why a real gripper could not be built, one member per fallback the factory takes."""

    #: ``gripper.vendor`` is not a name this stack knows.
    UNKNOWN_VENDOR = "unknown_vendor"
    #: Robotiq requested on an arm that is not a UR. The gripper lives on the UR
    #: controller's tool I/O.
    ROBOTIQ_NEEDS_UR = "robotiq_needs_ur"
    #: Vacuum requested, but the arm does not advertise ``SupportsDigitalIO``.
    VACUUM_NEEDS_DIGITAL_IO = "vacuum_needs_digital_io"
    #: A digital-I/O jaw requested, but the arm does not advertise ``SupportsDigitalIO``.
    #: Separate from the vacuum member so the operator message names the right
    #: end-effector, since the two share a cause and nothing else.
    JAW_IO_NEEDS_DIGITAL_IO = "jaw_io_needs_digital_io"
    #: A recognised vendor with no driver in this repo, such as franka_hand or schunk.
    NO_DRIVER = "no_driver"


@dataclass(frozen=True, slots=True)
class GripperSubstitution:
    """What a substituted gripper carries: what was asked for, why it was refused, what to do."""

    reason: SubstitutionReason
    #: The ``gripper.vendor`` value the config asked for.
    requested: str
    #: One sentence an operator can act on.
    detail: str
    fix: str


class NullGripper:
    """No-op gripper. Implements :class:`Gripper` structurally."""

    def __init__(
        self,
        *,
        min_width_mm: float = 0.0,
        max_width_mm: float = 0.0,
        substitution: GripperSubstitution | None = None,
    ) -> None:
        self._min = float(min_width_mm)
        self._max = float(max_width_mm)
        self._connected = False
        #: ``None`` means this cell genuinely has no end-effector configured, which is
        #: a legitimate state. Anything else means a real gripper was asked for and
        #: could not be built.
        self.substitution = substitution

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def min_width_mm(self) -> float:
        return self._min

    @property
    def max_width_mm(self) -> float:
        return self._max

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def activate(self) -> None:
        return

    def set_width_mm(
        self,
        width_mm: float,
        *,
        speed: float | None = None,
        force: float | None = None,
    ) -> None:
        del width_mm, speed, force

    def get_width_mm(self) -> float:
        return self._max


assert isinstance(NullGripper(), Gripper), (
    "NullGripper does not satisfy the Gripper Protocol"
)
