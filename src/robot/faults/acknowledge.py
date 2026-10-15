"""A person acknowledging one fault at the cell, and the report of what happened (P0i i4).

``FaultAcknowledgement`` is the library twin of ``python -m src.robot.faults acknowledge``: built by
``from_tree(loaded, confirm=...)``, one verb ``run()``, a frozen ``AcknowledgementReport``. It asks the robot's
``acknowledge fault`` ability row first, takes the cell lock and connects the arm alone, reads the controller's state,
and names the fault it describes. A fault no software clears (an emergency button, a controller powered off) is said
with what a person does about it, and nothing is sent. Any other is put to the person, in the fault's own sentence,
through ``confirm``, which has no default: nothing is acknowledged that nobody answered.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.config.tree import LoadedTree
    from src.robot.core.arm_capabilities import SupportsRobotStatus
    from src.robot.execution.robot import Robot

__all__ = ["AcknowledgementOutcome", "AcknowledgementReport", "FaultAcknowledgement", "acknowledge_command"]

logger = logging.getLogger(__name__)

#: Why the door builds an arm whose wrist camera body cannot be placed yet (P0i i4 review): it never moves it.
_NOTHING_MOVES = ("acknowledging a fault moves nothing: the arm is connected, its state read and its fault "
                  "acknowledged, and it is never commanded")


def acknowledge_command(profile: str | None) -> str:
    """The command a person at the cell runs to acknowledge the fault of the cell ``profile`` names, as shown to them."""
    chain = f" --profile {profile}" if profile else ""
    return f"python -m src.robot.faults acknowledge{chain}"


class AcknowledgementOutcome(StrEnum):
    """What one acknowledgement came to."""

    #: The controller took the acknowledgement.
    ACKNOWLEDGED = "acknowledged"
    #: The arm reports no fault: nothing was sent.
    NO_FAULT = "no_fault"
    #: A fault no software clears, an emergency button or a controller powered off: nothing was sent.
    NOT_ACKNOWLEDGEABLE = "not_acknowledgeable"
    #: The person did not confirm: nothing was sent.
    DECLINED = "declined"
    #: The acknowledgement was sent and the controller did not take it.
    NOT_TAKEN = "not_taken"
    #: Refused before the controller was asked: the arm offers no acknowledgement, the cell is held, the connect failed.
    REFUSED = "refused"


_EXIT_OK = frozenset({AcknowledgementOutcome.ACKNOWLEDGED, AcknowledgementOutcome.NO_FAULT})


@dataclass(frozen=True, slots=True)
class AcknowledgementReport:
    """One acknowledgement: its outcome, the fault as the controller described it, and what a person does next."""

    outcome: AcknowledgementOutcome
    #: The fault, in its own sentence (``ArmFault.render``), or empty where none was read.
    fault: str = ""
    #: What a person does next, in the imperative, or empty where nothing is left to do.
    what_to_do: str = ""
    #: The refusal, verbatim, for REFUSED.
    detail: str = ""

    @property
    def exit_code(self) -> int:
        """0 when the fault was acknowledged or there was none, 1 otherwise."""
        return 0 if self.outcome in _EXIT_OK else 1

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Describe this to a person: the outcome, then the fault, the refusal and what to do. ASCII, no newline."""
        lines = [f"acknowledge fault  {self.outcome.value.upper()}"]
        if self.fault:
            lines.append(f"  fault  {self.fault}")
        if self.detail:
            lines.append(f"  refused  {self.detail}")
        if self.what_to_do:
            lines.append(f"  next  {self.what_to_do}")
        return "\n".join(line.encode("ascii", "backslashreplace").decode("ascii") for line in lines)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe: the same fields ``render()`` prints."""
        return {"outcome": self.outcome.value, "exit_code": self.exit_code, "fault": self.fault,
                "what_to_do": self.what_to_do, "detail": self.detail}


@dataclass(frozen=True)
class FaultAcknowledgement:
    """One acknowledgement, to be run by a person standing at the cell."""

    robot: "Robot"
    #: Asked with the fault's own sentence; ``True`` only when the person at the cell said yes. No default.
    confirm: Callable[[str], bool]

    @classmethod
    def from_tree(cls, loaded: "LoadedTree", *, confirm: Callable[[str], bool]) -> "FaultAcknowledgement":
        """The acknowledgement for the arm a loaded tree describes, built alone (no gripper). Connects nothing.

        A wrist camera whose body cannot be placed yet does not stop it (found by review, 2026-09-26): the arm is never
        commanded here, and a stop on a cell whose camera is not calibrated yet is a stop a person has to clear too.
        """
        from src.robot.execution.robot import Robot

        return cls(robot=Robot.from_tree(loaded, gripper=None, unmodelled_wrist_body=_NOTHING_MOVES),
                   confirm=confirm)

    def run(self) -> AcknowledgementReport:
        """Ask the ability row, watch under the cell lock, read the fault, ask the person, acknowledge. Never raises
        for a refusal; the watch is closed and the lock given back on every way out.

        It watches, it does not connect (measured on URSim, 2026-09-26, 1 of 4). A full connect starts the
        controller's control program, and a controller in a protective stop starts none, so the door was refused on
        exactly the stop it is for. The watch (``Robot.watching``) reads the state and releases a stop without one, and
        on it no control program runs by design: a controller that is ready and runs none has no fault a person clears
        here, and the next connect starts the program.
        """
        from src.robot.core import ArmFault, ArmFaultClass
        from src.robot.execution.abilities import Ability, row_of

        # Its own row, and no other (found by review, 2026-09-26): a row of another function that cannot be read, a
        # wrist body that does not resolve for the pick, is no reason to leave a stop standing.
        row = row_of(Ability.ACKNOWLEDGE_FAULT, arm=self.robot.arm)
        if not row.available:
            return AcknowledgementReport(AcknowledgementOutcome.REFUSED, detail=row.reason)
        try:
            with self.robot.watching():
                # The row said it: this arm reads its controller's state, takes an acknowledgement, and can be watched.
                arm = cast("SupportsRobotStatus", self.robot.arm)
                fault = ArmFault.from_status(arm.get_robot_status())
                if fault is None or fault.kind is ArmFaultClass.PROGRAM_STOPPED:
                    return AcknowledgementReport(AcknowledgementOutcome.NO_FAULT)
                said = fault.render()
                if not fault.acknowledgeable:
                    return AcknowledgementReport(AcknowledgementOutcome.NOT_ACKNOWLEDGEABLE, fault=said,
                                                 what_to_do=fault.what_to_do())
                if not self.confirm(said):
                    return AcknowledgementReport(AcknowledgementOutcome.DECLINED, fault=said,
                                                 what_to_do="nothing was acknowledged: " + fault.what_to_do())
                logger.warning("A person at the cell acknowledged a %s.", fault.kind.value)
                if not arm.acknowledge_fault(fault):
                    return AcknowledgementReport(AcknowledgementOutcome.NOT_TAKEN, fault=said,
                                                 what_to_do=fault.what_to_do())
                return AcknowledgementReport(AcknowledgementOutcome.ACKNOWLEDGED, fault=said,
                                             what_to_do="start the run again once the arm stands where it should")
        except Exception as exc:  # noqa: BLE001 - a held cell or a refused watch is this report's answer
            return AcknowledgementReport(AcknowledgementOutcome.REFUSED, detail=f"{type(exc).__name__}: {exc}")
