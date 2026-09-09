"""Ask any arm what safety it enforces, and get an answer that can be printed.

The defect this exists to end is a report of a cleared safety layer from a run that
never reached one. `SafetyPreflight` lives in the driver, the dummy driver carries none
(`drivers/dummy/arm.py` says so), and a run with `policy.preflight` set to `None`
looks exactly like a run that passed six guards.
`scripts/examples/api/01_first_cell/one_pick_end_to_end.py` records the case in its own words. The question is
the one every caller of the library API asks: hand over an arm, and be told what it
will refuse.

This is read at runtime and printed rather than checked against the driver registry,
for a structural reason. A library caller supplies their own arm object, which
enumerating the registry cannot see. The attestation has to travel with the run and
appear in the report, or it describes the drivers that shipped rather than the cell
that ran.

An arm that does not answer is not treated as safe.
:attr:`SafetyPosture.UNSTATED` is what a caller own `RobotArm` gets, and
:attr:`SafetyAttestation.enforced` is `False` for it. Reading silence as consent is the
failure mode this module is named after.

It attests and it does not enforce. Nothing here can stop a motion.
`SafetyPreflight` is gated inside the driver own `move` (`drivers/ur/arm.py:499-527`,
`:613`), which is the only place that can refuse a command. This says what is in the
pipeline, so a run that enforced nothing cannot be mistaken for one that enforced
everything.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.robot.safety.preflight import SafetyPreflight

if TYPE_CHECKING:  # pragma: no cover (typing only)
    pass

__all__ = ["SafetyAttestation", "SafetyGated", "SafetyPosture"]


@runtime_checkable
class SafetyGated(Protocol):
    """Capability extension: this arm can say which safety pipeline it runs.

    It is modelled on :class:`~src.robot.core.gripper.ObjectDetectingGripper` at
    `core/gripper.py:93`, which is this repository shape for a capability some drivers
    have and the rest simply do not implement. A driver opts in by exposing
    ``safety_preflight``, and one that does not is reported as
    :attr:`SafetyPosture.UNSTATED`, never as safe.

    Returning `None` is an answer, and it differs from not answering. A driver that
    deliberately carries no preflight, such as the dummy, implements this and returns
    `None`, which reads as :attr:`SafetyPosture.UNGATED`, a stated fact. Not
    implementing it reads as UNSTATED, meaning nobody said. Both are unsafe to drive
    and an operator can tell them apart, because the first is a decision and the second
    is a gap.
    """

    @property
    def safety_preflight(self) -> "SafetyPreflight | None":
        """The guard pipeline every motion of this arm passes through, or ``None`` for none."""
        ...


class SafetyPosture(StrEnum):
    """What an arm answered when asked what it enforces."""

    #: A preflight with at least one guard. The names are in :attr:`SafetyAttestation.guards`.
    GATED = "gated"
    #: The arm answered, and the answer is that nothing gates its motion. That is the
    #: dummy driver, and a sim arm built without a preflight.
    UNGATED = "ungated"
    #: The arm does not implement :class:`SafetyGated`. It is read as unsafe on purpose,
    #: because silence is not consent, and a caller-supplied arm is exactly the case
    #: enumerating the driver registry cannot see.
    UNSTATED = "unstated"
    #: A preflight object whose guard pipeline is empty. It has its own value because it
    #: is the most misleading state of the four: everything is wired, `evaluate()` runs,
    #: and it can refuse nothing. A pipeline that passes every input is
    #: indistinguishable from safety in every log.
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class SafetyAttestation:
    """What an arm said it enforces. It satisfies both halves of the report contract.

    It carries both `render()` and `to_dict()`, which elsewhere in this repository are
    disjoint, and that is why `src/contracts/reporting.py` defines them as separate
    Protocols. This one needs both: an operator reads the text before driving a cell,
    and the record log stores the mapping, so the safety posture of a run stays
    recoverable from telemetry later.
    """

    #: The arm class name rather than the vendor, because two cells can run the same
    #: vendor with different guards.
    arm: str
    posture: SafetyPosture
    #: Guard names in execution order. Empty unless the posture is GATED.
    guards: tuple[str, ...] = ()
    #: The canonical guard families that are not in the pipeline.
    #:
    #: The absence is the safety fact. `guards` says what will run, so an operator reads
    #: a count and stops; what a count cannot show is which family was switched off, and
    #: an `enforce: false` block removes that family surface entirely rather than
    #: silencing its log.
    omitted: tuple[str, ...] = ()
    #: Whether the guards ever judge the middle of a planned path, or only its endpoints.
    checks_trajectories: bool = False

    @property
    def enforced(self) -> bool:
        """Will something refuse a bad command?

        True for GATED alone. `UNSTATED` and `EMPTY` both look like working pipelines
        from the outside and refuse nothing, so both are false here. A caller branches
        on this property and prints the posture.
        """
        return self.posture is SafetyPosture.GATED and bool(self.guards)

    @classmethod
    def of(cls, arm: object) -> "SafetyAttestation":
        """Ask ``arm`` what it enforces. Never raises, never imports a vendor SDK.

        A broken answer is still an answer. Where the property raises, as it does on a
        half-constructed driver or a double that models the attribute badly, that is
        reported as UNSTATED rather than propagated. This is called on the path where an
        operator is finding out whether it is safe to start, and an exception there
        would replace the answer with a traceback.

        The whole interrogation sits inside one `try`, and both of its lines need to.
        `runtime_checkable` protocol matching goes through `hasattr`, which evaluates
        the property, so an `isinstance` outside the `try` raises from inside itself for
        a driver whose `safety_preflight` raises, and the handler never runs. Reading
        `guard_names` outside it raises `AttributeError` one line later for an arm that
        answers with something that is not a preflight at all.

        An arm that answers with something unusable is reported UNSTATED and not GATED.
        For a safety question that is the same as having said nothing, and the
        fail-closed reading is the only safe one.
        """
        name = type(arm).__name__
        try:
            if not isinstance(arm, SafetyGated):
                return cls(arm=name, posture=SafetyPosture.UNSTATED)
            preflight = arm.safety_preflight
            if preflight is None:
                return cls(arm=name, posture=SafetyPosture.UNGATED)
            names = tuple(preflight.guard_names)
            trajectories = bool(preflight.checks_trajectories)
            # `getattr` with a default, because a caller can supply any object with a
            # `safety_preflight`, and an attestation that raised on an unfamiliar one
            # would fail exactly where it is asked whether it is safe to start.
            omitted = tuple(getattr(preflight, "omitted_guards", ()) or ())
        except Exception:  # noqa: BLE001 (a driver that cannot answer has not answered)
            return cls(arm=name, posture=SafetyPosture.UNSTATED)
        return cls(
            arm=name,
            posture=SafetyPosture.GATED if names else SafetyPosture.EMPTY,
            guards=names,
            omitted=omitted,
            checks_trajectories=trajectories,
        )

    def render(self) -> str:
        """One block an operator reads before driving a cell. ASCII, no trailing newline."""
        head = f"  safety     {self.posture.value.upper():<9} {self.arm}"
        if self.posture is SafetyPosture.GATED:
            path = "endpoints and paths" if self.checks_trajectories else "endpoints only"
            lines = [
                head,
                f"             {len(self.guards)} guard(s), {path}",
                f"             {', '.join(self.guards)}",
            ]
            if self.omitted:
                # The line an operator needs. A guard count reads as complete, and
                # naming the family that is switched off is what makes it a reading
                # rather than a reassurance. An `enforce: false` block removes that
                # surface entirely.
                lines.append(f"             not enforced: {', '.join(self.omitted)}")
            return "\n".join(lines)
        reason = {
            SafetyPosture.UNGATED: "this arm states that nothing gates its motion",
            SafetyPosture.UNSTATED: ("this arm does not report a safety pipeline, so none is "
                                     "assumed"),
            SafetyPosture.EMPTY: ("a preflight is wired and its guard list is empty, so it refuses "
                                  "nothing"),
        }[self.posture]
        return "\n".join((head, f"             {reason}"))

    def to_dict(self) -> dict[str, Any]:
        """Plain data. It survives `json.dumps` with no custom encoder."""
        return {
            "arm": self.arm,
            "posture": self.posture.value,
            "guards": list(self.guards),
            "omitted": list(self.omitted),
            "checks_trajectories": self.checks_trajectories,
            "enforced": self.enforced,
        }
