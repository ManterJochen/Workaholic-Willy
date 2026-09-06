"""Declarative capability descriptor for a :class:`RobotArm` driver.

Each driver advertises its feature set as a frozen :class:`RobotCapabilities`.
Pipeline and planner code branches on these flags rather than running
``isinstance(driver, ...)`` against concrete driver classes: it skips
``move_linear`` where the driver has no Cartesian move, and falls back to an
offline FK where ``has_native_fk`` is False.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RobotCapabilities"]


@dataclass(frozen=True, slots=True)
class RobotCapabilities:
    """Feature flags advertised by a robot-arm driver.

    Parameters
    ----------
    vendor : str
        Free-form vendor or driver identifier (``"ur"``, ``"franka"``, ``"ros2"``,
        ``"sim"``, ``"dummy"``). Lower-case, no whitespace.
    model : str
        Arm model identifier (``"ur5e"``, ``"panda"``, ``"iiwa14"``). Free-form,
        may be empty.
    dof : int
        Degrees of freedom. Validated against :class:`JointPositions` lengths at
        the protocol boundary.
    supports_joint_move : bool
        Driver implements :meth:`RobotArm.move_joint`.
    supports_linear_move : bool
        Driver implements the Cartesian :meth:`RobotArm.move_linear`.
    supports_async_move : bool
        Driver can dispatch non-blocking moves, the UR
        ``moveJ(asynchronous=True)`` path.
    has_native_fk : bool
        Driver provides controller-side forward kinematics.
    has_native_ik : bool
        Driver provides controller-side inverse kinematics.
    has_force_control : bool
        Driver exposes a Cartesian force-control or admittance-control primitive.
        Reserved for a force-control surface that does not exist yet, and defaults
        to ``False``.
    is_simulated : bool
        ``True`` where the driver is purely software, with no real hardware.
    """

    vendor: str
    model: str = ""
    dof: int = 6
    supports_joint_move: bool = True
    supports_linear_move: bool = True
    supports_async_move: bool = False
    has_native_fk: bool = False
    has_native_ik: bool = False
    has_force_control: bool = False
    is_simulated: bool = False

    def __post_init__(self) -> None:
        if not self.vendor:
            raise ValueError("RobotCapabilities.vendor must be non-empty.")
        if " " in self.vendor or self.vendor != self.vendor.lower():
            raise ValueError(
                f"RobotCapabilities.vendor must be lowercase and whitespace-free; "
                f"got {self.vendor!r}."
            )
        if self.dof <= 0:
            raise ValueError(f"RobotCapabilities.dof must be > 0; got {self.dof}.")
