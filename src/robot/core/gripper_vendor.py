""":class:`GripperVendor`: canonical identifiers for gripper drivers.

This mirrors :class:`RobotVendor`. Every gripper driver under
:mod:`src.robot.grippers` registers a factory under one of these members.
Pipelines do not read this enum; they go through the
:class:`~src.robot.core.Gripper` Protocol.

Adding a gripper is a two-line change here plus a driver module or subpackage under
``src/robot/grippers/<name>.py``.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["GripperVendor"]


class GripperVendor(StrEnum):
    """Canonical vendor identifiers for gripper drivers.

    Values are lowercase and whitespace-free, so they round-trip through YAML and
    JSON config without quoting.

    Members
    -------
    ROBOTIQ
        Robotiq HE and HE-X over the SDU ``robotiq_gripper`` driver.
    FRANKA_HAND
        Franka Hand over libfranka. No driver exists yet.
    SCHUNK
        Schunk EGK and EGN. No driver exists yet.
    VACUUM
        A suction end-effector actuated over the controller's digital I/O.
        Deliberately not manufacturer-specific: an ejector on an output pin, with an
        optional vacuum switch on an input, is the whole interface, so a cell can be
        configured for suction before the cup has been chosen. See
        ``grippers/vacuum.py``.
    JAW_IO
        A parallel-jaw gripper actuated over the controller's digital I/O, the
        vendor-neutral twin of :attr:`VACUUM`. It is named for what it is (jaws) and
        how it speaks (I/O) rather than for a manufacturer, because a pneumatic or
        electric two-finger gripper on a UR is one or two output pins plus, usually,
        reed switches on inputs. :attr:`ROBOTIQ` is jaws too, but speaks a socket
        protocol the URCap opens rather than I/O. See ``grippers/jaw_io.py``.
    ONROBOT
        An OnRobot RG2 or RG6 reached over Modbus TCP through the OnRobot Compute
        Box. It differs from :attr:`ROBOTIQ` in every way that matters: the Compute
        Box is a separate device on its own network address rather than something the
        arm hosts, there is no activation stroke and no speed register, and the width
        is an opening in tenths of a millimetre, so a larger number means more open,
        the opposite of Robotiq's 0-255 counts. See ``grippers/onrobot.py``.

        RG2 and RG6 only. The 2FG7 shares the family name and not the register map,
        and no public map for it could be sourced; the vacuum tools (VG10, VGC10) and
        the 3FG15 are different devices again. A driver that accepted them would be
        guessing at addresses.
    DUMMY
        Pure-Python sim gripper for offline development.
    NONE
        Explicit "no gripper attached": a no-op implementation that still satisfies
        the :class:`Gripper` Protocol, so pipelines stay branch-free.
    """

    ROBOTIQ = "robotiq"
    FRANKA_HAND = "franka_hand"
    SCHUNK = "schunk"
    VACUUM = "vacuum"
    JAW_IO = "jaw_io"
    ONROBOT = "onrobot"
    DUMMY = "dummy"
    NONE = "none"

    def __repr__(self) -> str:  # pragma: no cover (cosmetic)
        return f"GripperVendor.{self.name}"

    @classmethod
    def from_string(cls, value: str) -> GripperVendor:
        """Coerce a free-form, case-insensitive string into a member.

        Raises :class:`ValueError` for an unknown vendor, listing the valid options
        in the message.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(
                f"GripperVendor expects a string, got {type(value).__name__}"
            )
        normalised = value.strip().lower()
        try:
            return cls(normalised)
        except ValueError as exc:
            valid = ", ".join(v.value for v in cls)
            raise ValueError(
                f"unknown gripper vendor {value!r}; valid: {valid}"
            ) from exc
