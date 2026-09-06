""":class:`Gripper`: vendor-neutral abstract surface for a parallel-jaw gripper.

Drivers under :mod:`src.robot.drivers` (Robotiq HE, Robotiq HE-X, Franka Hand, sim,
dummy) implement this Protocol. The arm package exposes a configured
:class:`Gripper` alongside the :class:`RobotArm`, and the two are independent: a
driver with no gripper attached returns a no-op implementation rather than ``None``.

Numerics contract
-----------------
* Widths are millimetres, as ``float``.
* ``force`` is normalised to ``[0.0, 1.0]``. The mapping onto the underlying force
  scale is driver-dependent, and the Robotiq driver maps it onto 0-255 counts. It is
  not newtons.
* Speeds are normalised to ``[0.0, 1.0]``, again a driver-dependent mapping onto
  counts or millimetres per second. A pipeline that needs physical units reads
  :class:`RobotCapabilities`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["Gripper", "ObjectDetectingGripper"]


@runtime_checkable
class Gripper(Protocol):
    """Vendor-neutral gripper interface."""

    # ---- introspection --------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """``True`` while a live link to the gripper is open."""
        ...

    @property
    def min_width_mm(self) -> float:
        """Smallest commandable jaw opening, in millimetres."""
        ...

    @property
    def max_width_mm(self) -> float:
        """Largest commandable jaw opening, in millimetres."""
        ...

    # ---- lifecycle ------------------------------------------------------

    def connect(self) -> None:
        """Open the link to the gripper. Idempotent."""
        ...

    def disconnect(self) -> None:
        """Close the link. Idempotent."""
        ...

    def activate(self) -> None:
        """Run the vendor-specific calibration or activation routine."""
        ...

    # ---- commands -------------------------------------------------------

    def set_width_mm(
        self,
        width_mm: float,
        *,
        speed: float | None = None,
        force: float | None = None,
    ) -> None:
        """Command the jaw opening to ``width_mm``.

        Parameters
        ----------
        width_mm
            Target opening. The driver clamps it into
            ``[min_width_mm, max_width_mm]``.
        speed
            Optional normalised speed in ``[0.0, 1.0]``.
        force
            Optional normalised grasp force in ``[0.0, 1.0]``, not newtons, as the
            numerics contract above sets out. ``None`` keeps the driver default.
        """
        ...

    def get_width_mm(self) -> float:
        """Current jaw opening, in millimetres."""
        ...


@runtime_checkable
class ObjectDetectingGripper(Protocol):
    """Capability extension: the gripper reports whether it is holding an object.

    A driver that advertises this Protocol opts in to post-close verification in
    :class:`GraspExecutionPolicy`. A driver with no feedback channel does not
    implement the method, and the policy then trusts the close command, which is the
    documented default.

    The check runs after the gripper has commanded its close target width. It returns
    ``True`` where the jaws have engaged the object and ``False`` where the grasp
    slipped, missed or closed on nothing.
    """

    def is_object_detected(self) -> bool:
        """Return ``True`` if the gripper is currently holding an object."""
        ...
