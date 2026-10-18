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
``src/robot/drivers/<name>/`` that registers its factory. Every member has a driver:
``tests/robot/drivers/test_robot_vendor_refusal_names_what_can_be_built.py`` goes red for one that has
none. Pipelines do not read this enum; they go through the
:class:`~src.robot.core.RobotArm` Protocol.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

__all__ = ["RobotVendor"]

#: Names that were reserved slots with no driver behind them until they were removed on 2026-10-17 (the owner's
#: cleanup). A refusal of one says so, so a tree written for one of them is not answered as a typo.
_REMOVED: Final = frozenset({"kuka", "franka", "ros2"})


class RobotVendor(StrEnum):
    """Canonical vendor identifiers for arm drivers.

    Values are lowercase and whitespace-free, so they round-trip through YAML and
    JSON config without quoting.

    Members
    -------
    UR
        Universal Robots over RTDE. See ``drivers/ur/``.
    SIM
        The Isaac-backed sim arm. Import-safe on a host without Isaac: the SDK is
        touched at ``connect()``, not at construction. See ``drivers/sim/``.
    DUMMY
        Pure-Python sim arm for tests and offline development.
    """

    UR = "ur"
    SIM = "sim"
    DUMMY = "dummy"

    def __repr__(self) -> str:  # pragma: no cover (cosmetic)
        return f"RobotVendor.{self.name}"

    @classmethod
    def from_string(cls, value: str) -> RobotVendor:
        """Coerce a free-form, case-insensitive string into a member.

        Raises :class:`ValueError` for an unknown vendor, naming every vendor this checkout
        builds, and saying so where the name is one of those removed on 2026-10-17.
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
            message = f"unknown robot vendor {value!r}; buildable: {', '.join(v.value for v in cls)}"
            if normalised in _REMOVED:
                message += (f"; {normalised!r} was a reserved name with no driver behind it and was removed on "
                            "2026-10-17 (the owner's cleanup)")
            raise ValueError(message) from exc
