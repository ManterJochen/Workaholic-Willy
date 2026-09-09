""":class:`GripperVendor`: canonical identifiers for gripper drivers.

This mirrors :class:`RobotVendor`. Every gripper driver under
:mod:`src.robot.grippers` registers a factory under one of these members.
Pipelines do not read this enum; they go through the
:class:`~src.robot.core.Gripper` Protocol.

Adding a gripper is a two-line change here plus a driver module or subpackage under
``src/robot/grippers/<name>.py``. A member added without that driver belongs in
``_RESERVED_VENDORS`` at the bottom of this file, or the refusal in
:meth:`GripperVendor.from_string` will offer a name that builds no end-effector.
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
        Franka Hand over libfranka. A reserved slot: no driver in this repo, and
        listed in ``_RESERVED_VENDORS`` so the refusal below says so.
    SCHUNK
        Schunk EGK and EGN. A reserved slot, same as :attr:`FRANKA_HAND`.
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

        Raises :class:`ValueError` for an unknown vendor. The message separates the
        vendors this checkout can build from the reserved slots, because those are
        different mistakes with different fixes (see ``_RESERVED_VENDORS`` below).
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
            buildable = ", ".join(v.value for v in cls if v.value not in _RESERVED_VENDORS)
            message = f"unknown gripper vendor {value!r}; buildable: {buildable}"
            if _RESERVED_VENDORS:
                reserved = ", ".join(v.value for v in cls if v.value in _RESERVED_VENDORS)
                message += (
                    f"; reserved (no driver here): {reserved}. A cell configured for a reserved "
                    "name comes up with a NullGripper and no end-effector."
                )
            raise ValueError(message) from exc


#: Members that exist so a config can be written against a driver this repo does not have. They
#: are deliberate slots, the same way ``franka`` and ``ros2`` are slots in :class:`RobotVendor`.
#:
#: Measured 2026-09-09: the refusal above built its list from the enum and so advertised both of
#: these as valid. The enum carried eight members, ``available_gripper_vendors()`` six. A typo was
#: answered with ``robotiq, franka_hand, schunk, vacuum, jaw_io, onrobot, dummy, none``; a config
#: copying one of the two reserved names passes schema validation (they are real members, and that
#: is correct) and then falls through to the ``SubstitutionReason.NO_DRIVER`` substitution in
#: ``execution/runtime_pick.py``. The cell connects, reports every pick a success, and holds
#: nothing. Splitting the refusal in two is what ``drivers/host.py`` (``Host.require``) already
#: does on the arm side: "not a name" and "a name with no driver here" are different mistakes.
#:
#: Hand-kept, because this is the bottom of the stack. The registry that owns the fact lives two
#: layers up (``grippers/registry.py``), and ``core`` imports nothing above itself. The same kind
#: of hand-kept membership test in ``doctor.py`` reported first ``vacuum`` and then ``jaw_io`` as
#: having no driver while both had one, so this one is pinned against the registry in both
#: directions by ``tests/test_gripper_vendor_refusal_names_what_can_be_built.py``: a driver that
#: lands without leaving this set, or a member added without entering it, goes red.
_RESERVED_VENDORS: frozenset[str] = frozenset({"franka_hand", "schunk"})
