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
:class:`~src.robot.core.RobotArm` Protocol.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["RobotVendor"]


class RobotVendor(StrEnum):
    """Canonical vendor identifiers for arm drivers.

    Values are lowercase and whitespace-free, so they round-trip through YAML and
    JSON config without quoting.
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

        Raises :class:`ValueError` for an unknown vendor, listing the valid options
        in the message.
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
            valid = ", ".join(v.value for v in cls)
            raise ValueError(
                f"unknown robot vendor {value!r}; valid: {valid}"
            ) from exc
