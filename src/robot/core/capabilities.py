"""Declarative capability descriptor for a :class:`RobotArm` driver.

Each driver states its identity and a few facts as a frozen :class:`RobotCapabilities`:
the vendor, the model, the joint count, the controller it drives (the cell lock is
keyed on them), whether it computes its own forward kinematics (the IK quality
guard's singularity check runs only where it does) and whether it is software only.

What an arm offers beyond :class:`RobotArm` (force readings, hand guiding, digital
I/O, kept lines) is not a flag here: it is asked through the capability Protocols in
``arm_capabilities`` and ``freedrive``. Every flag here is read by something (P0i i6,
2026-09-26: ``supports_joint_move``, ``supports_linear_move``, ``has_native_ik`` and
``has_force_control`` were set by every driver, read by nothing, and deleted).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RobotCapabilities"]


# Keyword-only, so the field order is not part of the contract: the value carries the arm's identity (the controller
# address, and a ``dof`` that comes from the arm description rather than a default of six), and a positional
# construction would make each of those a silent re-binding of somebody else's argument.
@dataclass(frozen=True, slots=True, kw_only=True)
class RobotCapabilities:
    """The identity of a robot-arm driver and the few facts it states as flags.

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
        the protocol boundary. No default since P0e: a driver reads it from its
        arm's description, and six was a UR's number that every other arm would
        have inherited in silence.
    controller : str or None
        The address of the controller this driver talks to, or ``None`` where none
        is declared or none exists. The cell lock is keyed on it.
    owns_a_controller : bool
        Whether this driver drives a controller of its own. A simulator and a desk
        dummy do not. Kept apart from ``controller`` (the owner, in aurora,
        2026-09-20): "drives none" and "drives one nobody has named yet" are
        different facts, and the second must refuse where the first takes no lock.
    has_native_fk : bool
        Driver provides forward kinematics (:meth:`RobotArm.fk`) the
        safety layer may call; the IK quality guard's singularity check
        runs only where it does.
    is_simulated : bool
        ``True`` where the driver is purely software, with no real hardware.
    """

    vendor: str
    model: str = ""
    dof: int
    controller: str | None
    owns_a_controller: bool
    has_native_fk: bool = False
    is_simulated: bool = False

    def __post_init__(self) -> None:
        if not self.vendor:
            raise ValueError("RobotCapabilities.vendor must be non-empty.")
        if " " in self.vendor or self.vendor != self.vendor.lower():
            raise ValueError(
                f"RobotCapabilities.vendor must be lowercase and whitespace-free; "
                f"got {self.vendor!r}."
            )
        # The same rule as the vendor's (P0g R4): the arm registry and the planner envelope take a model by its exact
        # lowercase name, the DH chain lowercases it, so a mixed-case model solved IK and then failed at the planner.
        if " " in self.model or self.model != self.model.lower():
            raise ValueError(
                f"RobotCapabilities.model must be lowercase and whitespace-free; "
                f"got {self.model!r}."
            )
        if self.dof <= 0:
            raise ValueError(f"RobotCapabilities.dof must be > 0; got {self.dof}.")
        if self.controller is not None and not self.owns_a_controller:
            raise ValueError(
                f"RobotCapabilities names the controller {self.controller!r} and says it owns none: a driver that "
                f"drives no controller of its own has no address to name."
            )
        if self.controller is not None and not self.controller.strip():
            raise ValueError("RobotCapabilities.controller is blank; None says no address is declared.")
