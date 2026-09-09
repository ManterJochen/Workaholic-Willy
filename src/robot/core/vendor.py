""":class:`RobotVendor`: canonical identifiers for arm-driver vendors.

Used by:

* :class:`src.config.schema.robot.RobotConfig`, where an operator selects a driver
  through the ``vendor`` field and the value is checked against this enum as the
  config loads.
* :mod:`src.robot.drivers.registry`, where driver factories register under one of
  these identifiers.
* :class:`src.robot.core.RobotCapabilities`, whose ``vendor`` field is deliberately
  free-form. It is validated for lowercase and whitespace only, never against this
  enum, so a custom rig can name a driver outside it. Enum membership is enforced at
  the config layer and not here.

Adding a vendor is a deliberate two-line change here plus a driver package under
``src/robot/drivers/<name>/``. Pipelines do not read this enum; they go through the
:class:`~src.robot.core.RobotArm` Protocol. A member added without that driver belongs
in ``_RESERVED_VENDORS`` at the bottom of this file, or the refusal in
:meth:`RobotVendor.from_string` will offer a name that builds no arm.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["RobotVendor"]


class RobotVendor(StrEnum):
    """Canonical vendor identifiers for arm drivers.

    Values are lowercase and whitespace-free, so they round-trip through YAML and
    JSON config without quoting.

    Members
    -------
    UR
        Universal Robots over RTDE. See ``drivers/ur/``.
    KUKA
        KUKA over EthernetKRL, which is TCP with XML and needs no vendor SDK on this
        side. See ``drivers/kuka/``.
    FRANKA
        Franka Emika over libfranka or franky. A reserved slot: ``drivers/franka/`` is
        a package stub that registers nothing, and the name is listed in
        ``_RESERVED_VENDORS`` so the refusal below says so.
    ROS2
        ROS 2 and MoveIt over ``rclpy``. A reserved slot, same as :attr:`FRANKA`.
    SIM
        The Isaac-backed sim arm. Import-safe on a host without Isaac: the SDK is
        touched at ``connect()``, not at construction. See ``drivers/sim/``.
    DUMMY
        Pure-Python sim arm for tests and offline development.
    """

    UR = "ur"
    KUKA = "kuka"
    FRANKA = "franka"
    ROS2 = "ros2"
    SIM = "sim"
    DUMMY = "dummy"

    def __repr__(self) -> str:  # pragma: no cover (cosmetic)
        return f"RobotVendor.{self.name}"

    @classmethod
    def from_string(cls, value: str) -> RobotVendor:
        """Coerce a free-form, case-insensitive string into a member.

        Raises :class:`ValueError` for an unknown vendor. The message separates the
        vendors this checkout can build from the reserved slots, because those are
        different mistakes with different fixes (see ``_RESERVED_VENDORS`` below).
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(
                f"RobotVendor expects a string, got {type(value).__name__}"
            )
        normalised = value.strip().lower()
        try:
            return cls(normalised)
        except ValueError as exc:
            buildable = ", ".join(v.value for v in cls if v.value not in _RESERVED_VENDORS)
            message = f"unknown robot vendor {value!r}; buildable: {buildable}"
            if _RESERVED_VENDORS:
                reserved = ", ".join(v.value for v in cls if v.value in _RESERVED_VENDORS)
                message += (
                    f"; reserved (no driver here): {reserved}. A cell configured for a reserved "
                    "name passes config validation and is then refused by create_arm; no arm is "
                    "substituted."
                )
            raise ValueError(message) from exc


#: Members that exist so a config can be written against a driver this repo does not have. They
#: are deliberate slots: ``drivers/franka/__init__.py`` and ``drivers/ros2/__init__.py`` are
#: package stubs that say no driver is registered, and register nothing.
#:
#: Measured 2026-09-09: the refusal above built its list from the enum and so advertised both of
#: these as valid. The enum carries six members, ``available_vendors()`` four. A typo was answered
#: with ``ur, kuka, franka, ros2, sim, dummy``, and a config copying one of the two reserved names
#: passes schema validation (they are real members, and that is correct).
#:
#: What happens next is not the gripper's story, which is why this was a message repair and not a
#: behaviour one. A reserved gripper name substitutes a ``NullGripper`` and the cell comes up
#: holding nothing while reporting success. An arm has no such fallback:
#: ``create_arm(RobotVendor.FRANKA)`` raises ``RobotConnectionError: no driver registered for
#: vendor 'franka'; available: dummy, kuka, sim, ur``, and ``Host.require`` refuses even earlier.
#: Nothing substitutes and nothing moves, so what the old message cost was the hour between
#: copying ``franka`` out of a list that called it valid and learning at build time that it was
#: not.
#:
#: Hand-kept, because this is the bottom of the stack. The registry that owns the fact lives two
#: layers up (``drivers/registry.py``, which imports this module), and ``core`` imports nothing
#: above itself. The same kind of hand-kept membership test in ``doctor.py`` reported first
#: ``vacuum`` and then ``jaw_io`` as having no driver while both had one, so this one is pinned
#: against the registry in both directions by
#: ``tests/test_robot_vendor_refusal_names_what_can_be_built.py``: a driver that lands without
#: leaving this set, or a member added without entering it, goes red.
_RESERVED_VENDORS: frozenset[str] = frozenset({"franka", "ros2"})
