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

Readings a caller takes after a command
---------------------------------------
* :class:`ReportsHoldEvidence` says what the gripper measured about a hold: HELD, EMPTY, or
  UNMEASURED where nothing was measured. On several drivers ``is_object_detected`` answers with
  what the driver knows, which is the command where nothing is wired, so a verb that has to say
  whether a hold was measured reads this instead.
* :class:`MeasuresWidth` says whether ``get_width_mm`` is a measurement or the band the jaws were
  commanded to.

A hand that toggles
-------------------
* :class:`TogglesWithoutSensor` is a hand whose every command is one pulse that flips its jaws, with no
  sensor to say where they stand. A pick asks it before the arm moves instead of commanding an open
  (:func:`toggle_without_sensor_of`), so the pick code needs no driver import.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

__all__ = [
    "Gripper",
    "HoldEvidence",
    "MeasuresWidth",
    "ObjectDetectingGripper",
    "OpensAndCloses",
    "ReportsHoldEvidence",
    "StoppableGripper",
    "TogglesWithoutSensor",
    "TwoStateGripper",
    "hold_evidence_of",
    "toggle_without_sensor_of",
    "width_is_measured_of",
]


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


@runtime_checkable
class StoppableGripper(Protocol):
    """Capability extension: the jaws can be halted where they are.

    A halt is not an ``open()``. Opening is a motion to a new target, and stopping is the absence
    of motion. On a Robotiq the two are different registers: ``SPE 0`` is the minimum speed, not a
    stop, and only clearing ``GTO`` halts a travel. A caller that stops a gripper by commanding a
    width has commanded another motion.

    The transport's halt (:meth:`robotiq_socket.RobotiqSocket.stop`) is reached through this
    Protocol. Without it an aborted close goes on closing while the cell believes the attempt is
    over. A driver without a halt does not implement this, and a caller must treat its absence as
    jaws that cannot be stopped.
    """

    def stop(self) -> None:
        """Halt the jaws where they are. Must not command a new width."""
        ...


@runtime_checkable
class TwoStateGripper(Protocol):
    """Capability extension: the gripper has two states, and reads a commanded width as the closed one at or below
    ``closed_below_mm`` and as open above it.

    ``jaw_io`` is one. The width-based Protocol hides it: a grasp above the threshold commands an OPEN and a release at
    or below it a CLOSE, neither raises, and a gripper with no sensor then reports the verb done. Both were found
    diagnosing the owner's toggle cell on 2026-09-23, so the hand verbs ask this before they command.
    """

    @property
    def closed_below_mm(self) -> float:
        """The commanded width at or below which the gripper closes."""
        ...


@runtime_checkable
class OpensAndCloses(Protocol):
    """Capability extension: the gripper takes open and close as what they are, not as a width.

    A verb that knows what it means, a pick's pre-open, its grasp, a release, says it here, so no width read against
    ``closed_below_mm`` can turn it round. On the owner's toggle cell (2026-09-23) a grasp at the part's width was an
    open and a release at the hand's width could be a close, with nothing raised. ``jaw_io`` is one; a gripper that is
    a :class:`TwoStateGripper` and not this one is still asked what a width means before it is commanded.
    """

    def set_closed(self, closed: bool) -> None:
        """Close the jaws when ``closed``, open them otherwise, and return once they have got there."""
        ...


class HoldEvidence(StrEnum):
    """What a gripper measured about a hold after its last command."""

    #: The gripper measured a part held: a stall on something, a vacuum switch, a part sensor.
    HELD = "held"
    #: The gripper measured nothing held: the jaws reached their target, or the switch or sensor
    #: reads off.
    EMPTY = "empty"
    #: Nothing was measured: no sensor is wired, the jaws are open or travelling, or the driver
    #: cannot say.
    UNMEASURED = "unmeasured"


@runtime_checkable
class ReportsHoldEvidence(Protocol):
    """Capability extension: the gripper says what it measured about a hold, never its command."""

    def hold_evidence(self) -> HoldEvidence:
        """HELD or EMPTY where a measurement says so, UNMEASURED otherwise."""
        ...


@runtime_checkable
class MeasuresWidth(Protocol):
    """Capability extension: the gripper says whether ``get_width_mm`` is measured or commanded."""

    def width_is_measured(self) -> bool:
        """``True`` where ``get_width_mm`` reads a position sensor."""
        ...


@runtime_checkable
class TogglesWithoutSensor(Protocol):
    """Capability extension: every command is one pulse that flips the jaws, and nothing measures where they stand.

    ``jaw_io`` with ``actuation: single_toggle`` is one: the owner's Hand-E on the Robotiq I/O Coupling, one tool
    output, 24 V, no feedback. The program counts its own pulses from where a person said the jaws stood when the
    gripper connected, so a pulse sent on a wrong count moves the jaws the wrong way and nothing notices. Three rules
    follow, and every pick path keeps them:

    * No pulse before the arm moves at the start of a pick. A pick asks :meth:`jaws_open_for_a_pick` instead of
      commanding an open: where the count says open it answers at once, and where it says closed it asks the person
      again rather than pulsing, and refuses the pick when nobody can be asked.
    * Exactly one pulse at the part (the close) and one at the release, each only where the count says the jaws stand
      the other way.
    * No width is checked and no hold is measured: a close counts as a grasp, and a report says it was not checked
      because there is no sensor.

    ``toggles_without_sensor`` is read as ``is True``, so a double that answers every attribute is not taken for one.
    """

    @property
    def toggles_without_sensor(self) -> bool:
        """``True`` where every command is a pulse that flips the jaws and nothing reads them back."""
        ...

    @property
    def jaws_closed(self) -> bool:
        """Where the program believes the jaws stand: its own count of its pulses, never a measurement."""
        ...

    @property
    def edge_unknown(self) -> bool:
        """``True`` where a pulse failed so that nobody can say whether the jaws flipped, until a person says again."""
        ...

    def jaws_open_for_a_pick(self) -> str:
        """Before a pick moves the arm: ``""`` where the jaws stand open, else why the pick must not start.

        Never pulses on its own. Where the count says closed it asks a person again, and a person who says closed
        may choose one pulse to open them there and then.
        """
        ...


def toggle_without_sensor_of(gripper: object) -> TogglesWithoutSensor | None:
    """``gripper`` where it is a hand that toggles with no sensor (:class:`TogglesWithoutSensor`), else ``None``."""
    if isinstance(gripper, TogglesWithoutSensor) and getattr(gripper, "toggles_without_sensor", False) is True:
        return gripper
    return None


def hold_evidence_of(gripper: object) -> HoldEvidence:
    """What ``gripper`` measured about a hold; UNMEASURED for a gripper that cannot say."""
    if isinstance(gripper, ReportsHoldEvidence):
        return HoldEvidence(gripper.hold_evidence())
    return HoldEvidence.UNMEASURED


def width_is_measured_of(gripper: object) -> bool:
    """Whether ``gripper`` measures its width; ``False`` for a gripper that cannot say."""
    return isinstance(gripper, MeasuresWidth) and bool(gripper.width_is_measured())
