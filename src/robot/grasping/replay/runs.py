"""Rolling up a record log from Python: the KPIs, and whether the telemetry behind them is sound.

A roll-up is three questions, not one: the KPIs, whether every record carries the telemetry those
KPIs read, and whether any record carries a field of the wrong type. `compute_kpis` answers only
the first. The audits travel with the numbers here because a KPI computed over records that are
missing telemetry is not wrong, it is unfounded, and those are different things a reader must be
able to tell apart. `KpiRollup.sound` separates them, the same distinction as
`ReadinessVerdict.UNREADABLE`. The exit code is 2 when either audit finds anything.

This is the offline tail. Nothing here imports the pick path, opens a cell or loads a policy. It
reads a log somebody else wrote.

The imports inside `kpis()` stay local rather than moving to module level: `replay.kpi`,
`replay.telemetry_catalog` and `telemetry.outcome_logging` each pull a large module tree and drag
`cv2` in behind them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from src.contracts import UNSET, Maybe, chosen

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord

__all__ = [
    "GateKeyStatus",
    "KpiRollup",
    "RecordLog",
    "SoakGate",
    "SoakSource",
    "SoakVerdict",
    "TelemetryOffender",
    "TelemetryVerdict",
]


class TelemetryVerdict(StrEnum):
    """Whether the records behind a roll-up carry what the KPIs assume.

    Not a severity ladder, and not a judgement about grasp quality. A log can be perfectly sound
    and describe a terrible pick rate; a log can be unsound and describe a wonderful one. This says
    only whether the numbers rest on complete records.
    """

    #: Every record carries the fields the KPIs read.
    SOUND = "sound"
    #: At least one record is missing a telemetry field, or carries one of the wrong type. The KPIs
    #: are still computed, because a partial answer with its caveat beats no answer.
    INCOMPLETE = "incomplete"
    #: The log could not be read at all.
    UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class TelemetryOffender:
    """One record that does not carry what the KPIs assume."""

    attempt_id: str
    #: Field names absent from the record.
    missing: tuple[str, ...] = ()
    #: Field names present with the wrong type.
    bad_types: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "missing": list(self.missing),
            "bad_fields": list(self.bad_types),
        }


@dataclass(frozen=True, slots=True)
class KpiRollup:
    """The KPIs, and whether the records behind them carry what those KPIs assume."""

    verdict: TelemetryVerdict
    source: str
    records: int
    #: The `KpiSummary`'s own mapping, minus every rate these records cannot measure. Empty when
    #: the log could not be read.
    kpi: Mapping[str, Any] = field(default_factory=dict)
    #: Maps a KPI name to why these records cannot measure it, from `kpi.unmeasurable_kpis`. Every
    #: name here is absent from `kpi` above, and that pairing is the point: a rate whose denominator
    #: is empty computes to a structural 0.0 and reads as a measurement of zero.
    unmeasurable: Mapping[str, str] = field(default_factory=dict)
    #: Records missing a telemetry field, in record order, exactly as `audit_records` returned them.
    missing_telemetry: tuple[TelemetryOffender, ...] = ()
    #: Records carrying a field of the wrong type, in record order, exactly as `audit_extra_records`
    #: returned them.
    wrong_types: tuple[TelemetryOffender, ...] = ()
    #: Why it was unreadable. Empty otherwise.
    detail: str = ""

    @property
    def sound(self) -> bool:
        return self.verdict is TelemetryVerdict.SOUND

    @property
    def offenders(self) -> tuple[TelemetryOffender, ...]:
        """Both audits merged, one entry per attempt, for reading.

        A derived view; `missing_telemetry` and `wrong_types` are the truth. Merging is lossy
        twice over: it drops the record order the audits emit, so a payload rebuilt from it would
        not match the one the CLI prints, and keying by attempt id collapses two records that
        share one. The two audits answer two questions and the CLI payload keeps two keys.
        """
        by_id: dict[str, TelemetryOffender] = {}
        for offender in (*self.missing_telemetry, *self.wrong_types):
            existing = by_id.get(offender.attempt_id)
            by_id[offender.attempt_id] = (
                offender
                if existing is None
                else TelemetryOffender(
                    attempt_id=offender.attempt_id,
                    missing=existing.missing or offender.missing,
                    bad_types=existing.bad_types or offender.bad_types,
                )
            )
        return tuple(by_id.values())

    @property
    def exit_code(self) -> int:
        """The exit code on the report, matching the CLI: 0 for a sound log, 2 for anything else."""
        return 0 if self.sound else 2

    def render(self) -> str:
        """The verdict, the numbers, and what is missing behind them. ASCII, no trailing newline."""
        if self.verdict is TelemetryVerdict.UNREADABLE:
            return f"  UNREADABLE: {self.source}{f': {self.detail}' if self.detail else ''}"
        lines = [f"  {self.verdict.value.upper()}: {self.records} record(s) from {self.source}"]
        lines.extend(f"    {key:<34}{value}" for key, value in sorted(dict(self.kpi).items()))
        # Printed, never omitted. An absent line reads as an oversight; a named rate reads as a fact
        # about these records. The reason itself belongs in the payload, not the terminal.
        for key in sorted(self.unmeasurable):
            lines.append(f"    {key:<34}not measurable from these records")
        if self.offenders:
            # The caveat prints with the numbers: a KPI over incomplete records is unfounded
            # rather than wrong, and the rate alone does not say which of those it is.
            lines.append(f"    {len(self.offenders)} record(s) do not carry what these KPIs read:")
            for offender in self.offenders[:5]:
                what = ", ".join((*offender.missing, *offender.bad_types))
                lines.append(f"      {offender.attempt_id}: {what}")
            if len(self.offenders) > 5:
                lines.append(f"      ... and {len(self.offenders) - 5} more")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, `json.dumps`-safe with no custom encoder."""
        return {
            "verdict": self.verdict.value,
            "source": self.source,
            "records": self.records,
            "sound": self.sound,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "kpi": dict(self.kpi),
            "unmeasurable": dict(self.unmeasurable),
            "missing_telemetry": [o.to_dict() for o in self.missing_telemetry],
            "wrong_types": [o.to_dict() for o in self.wrong_types],
        }


@dataclass(frozen=True, slots=True)
class RecordLog:
    """A log of attempt records, from a cell, a simulation run or a replay pack, rolled up into KPIs.

    ```python
    print(RecordLog.from_jsonl("picks.jsonl").kpis())
    ```

    Attributes:
        path (Path | None): The JSONL file; ``None`` for records in memory.
        records (tuple[GraspAttemptRecord, ...] | None): The records in memory; ``None`` for a file, read at
            :meth:`kpis`.
    """

    path: Path | None = None
    records: "tuple[GraspAttemptRecord, ...] | None" = None

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "RecordLog":
        """A JSONL log on disk. Nothing is read until :meth:`kpis`, so an unreadable log arrives as an UNREADABLE
        verdict rather than an exception.

        Args:
            path (str | Path): The log, one record per line.

        Returns:
            RecordLog: The log.
        """
        return cls(path=Path(path))

    @classmethod
    def from_records(cls, records: "Sequence[GraspAttemptRecord]") -> "RecordLog":
        """Records already in memory, from a run that has just finished.

        Args:
            records (Sequence[GraspAttemptRecord]): The records.

        Returns:
            RecordLog: The log.
        """
        return cls(records=tuple(records))

    def kpis(self) -> KpiRollup:
        """The KPIs over this log, with the audits that say whether they rest on anything: a rate whose records lack the
        fields behind it is unfounded, not just wrong.

        Returns:
            KpiRollup: The audit's ``verdict``, how many ``records``, the KPIs by name (``kpi``), what could not be
                measured and why (``unmeasurable``), and the records missing telemetry; prints as itself.
        """
        from src.robot.grasping.replay.kpi import (  # noqa: PLC0415
            compute_kpis,
            unmeasurable_kpis,
        )
        from src.robot.grasping.replay.telemetry_catalog import (  # noqa: PLC0415
            audit_extra_records,
            audit_records,
        )
        from src.robot.grasping.telemetry.outcome_logging import (  # noqa: PLC0415
            iter_jsonl,
        )

        records = self.records
        source = str(self.path) if self.path is not None else f"{len(records or ())} record(s)"
        if records is None:
            if self.path is None:  # pragma: no cover (neither factory can produce this)
                return KpiRollup(TelemetryVerdict.UNREADABLE, "nothing", 0,
                                 detail="no path and no records")
            if not self.path.is_file():
                return KpiRollup(TelemetryVerdict.UNREADABLE, source, 0,
                                 detail="no such record log")
            try:
                records = tuple(iter_jsonl(self.path))
            except Exception as exc:  # noqa: BLE001 (an unreadable log is a verdict, not a crash)
                return KpiRollup(TelemetryVerdict.UNREADABLE, source, 0,
                                 detail=f"{type(exc).__name__}: {exc}")

        rows = tuple(records)
        summary = compute_kpis(rows)
        # Order is part of the answer. Both audits walk the records and append, so both come back
        # in record order; each list is kept exactly as returned so a caller can rebuild the CLI
        # payload byte for byte.
        missing = tuple(
            TelemetryOffender(attempt_id=aid, missing=tuple(fields))
            for aid, fields in audit_records(rows)
        )
        wrong = tuple(
            TelemetryOffender(attempt_id=aid, bad_types=tuple(fields))
            for aid, fields in audit_extra_records(rows)
        )
        # The withheld rates come out of `kpi`, they are not merely annotated beside it. A payload
        # that carried both `false_positive_grasp_rate: 0.0` and a note saying it cannot be measured
        # would still be read by whichever half the reader saw first, and the number wins that.
        withheld = unmeasurable_kpis(rows, summary)
        published = {
            key: value for key, value in summary.to_dict().items() if key not in withheld
        }
        return KpiRollup(
            verdict=(
                TelemetryVerdict.INCOMPLETE if (missing or wrong) else TelemetryVerdict.SOUND
            ),
            source=source,
            records=len(rows),
            kpi=published,
            unmeasurable=withheld,
            missing_telemetry=missing,
            wrong_types=wrong,
        )


# =============================================================================================
# The gate
# =============================================================================================


class SoakSource(StrEnum):
    """What a soak verdict was computed over, which is what it is and is not evidence of.

    `evaluate_soak_gate_over_records` returns the same `dict[str, bool]` whether it ran over a
    real log or a sim log, and nothing in that return value says which. The population is recorded
    here instead: a green gate over the wrong population is how a stack gets credited with work it
    did not do. The synthetic stream and the canonical packs left on 2026-10-16 with the synthetic soak.
    """

    #: A captured `GraspAttemptRecord` log from a real cell. Can fail, and is meant to.
    REAL_RECORDS = "real_records"
    #: A log from an Isaac-physics run. Real picks, not hardware-representative, and
    #: `false_positive_grasp_rate` is structurally 0: no secondary verifier exists, in sim or on a
    #: cell, since the post-grasp verification stage left on 2026-09-29.
    SIM_RECORDS = "sim_records"


class GateKeyStatus(StrEnum):
    """One gate key's outcome. Every key of the gate is judged from the records alone."""

    PASSED = "passed"
    FAILED = "failed"


#: What each door's verdict is evidence of, in one line, for `render()`.
#:
#: Not the CLI's `note` field. Those are wire values inside a JSON payload, addressed to whoever
#: reads the report file; this is addressed to whoever reads the terminal. Neither is derived from
#: the other, and neither replaces the other.
_MEASURES: dict[SoakSource, str] = {
    SoakSource.REAL_RECORDS: (
        "record-intrinsic integrity over a real log. This can fail."
    ),
    SoakSource.SIM_RECORDS: (
        "record-intrinsic integrity over real sim picks. pick_success_rate is reported, not gated. "
        "Not hardware-representative."
    ),
}


@dataclass(frozen=True, slots=True)
class SoakVerdict:
    """A soak gate's outcome: what was judged, what it was judged over, and what failed."""

    source: SoakSource
    origin: str
    keys: Mapping[str, GateKeyStatus] = field(default_factory=dict)
    violations: tuple[str, ...] = ()
    attempts: int = 0
    kpi: Mapping[str, Any] = field(default_factory=dict)
    #: Why the input could not be read at all, empty when it could. Not the same as a failed gate,
    #: and that is the whole reason it is a separate field: a gate that ran and refused judged
    #: something, a gate handed a path that does not exist judged nothing. Measured: both record
    #: doors died with a raw `FileNotFoundError` traceback on a mistyped path while `RecordLog`, one
    #: class up this file, has answered the same mistake with `TelemetryVerdict.UNREADABLE` and exit
    #: 2 since it was written.
    unreadable: str = ""

    @property
    def measures(self) -> str:
        """One line on what this verdict is, and is not, evidence of."""
        return _MEASURES[self.source]

    @property
    def passes(self) -> bool:
        return not self.violations and not self.unreadable

    @property
    def exit_code(self) -> int:
        """Derived from the violations, never stored beside them: ``0 if not violations else 1``,
        which is what both CLI modes return. A stored copy could disagree with the list it
        summarises.

        An unreadable input exits 2, not 1, matching `KpiRollup.exit_code` for that same condition:
        1 means this gate judged the records and refused them, and a caller that cannot tell those
        apart will hunt for a quality problem in a file it never opened.
        """
        if self.unreadable:
            return 2
        return 0 if self.passes else 1

    def gate_wire(self) -> dict[str, Any]:
        """The gate in the exact shape the CLI prints: ``True`` / ``False``, plus the derived ``passes``.

        One stored fact, two views. `keys` is the truth; this is the wire projection and
        :meth:`to_dict` is the Python one. Neither can drift from the other, because neither is
        stored.
        """
        wire: dict[str, Any] = {name: status is GateKeyStatus.PASSED for name, status in self.keys.items()}
        wire["passes"] = self.passes
        return wire

    def render(self) -> str:
        """The verdict, what it measures, and every violation. ASCII, no trailing newline."""
        if self.unreadable:
            return f"  UNREADABLE: {self.origin}: {self.unreadable}"
        head = "PASS" if self.passes else "FAIL"
        lines = [
            f"  {head}: {self.attempts} attempt(s) from {self.origin}",
            f"    measures: {self.measures}",
        ]
        if self.keys:
            lines.append(f"    {len(self.keys)} gate key(s) judged")
        for name, status in sorted(self.keys.items()):
            if status is GateKeyStatus.FAILED:
                lines.append(f"      FAILED  {name}")
        for violation in self.violations:
            lines.append(f"    violation: {violation}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, `json.dumps`-safe with no custom encoder."""
        return {
            "source": self.source.value,
            "origin": self.origin,
            "measures": self.measures,
            "passes": self.passes,
            "exit_code": self.exit_code,
            "attempts": self.attempts,
            "judged_keys": len(self.keys),
            "keys": {name: status.value for name, status in self.keys.items()},
            "violations": list(self.violations),
            "kpi": dict(self.kpi),
            "unreadable": self.unreadable,
        }


@dataclass(frozen=True, slots=True)
class SoakGate:
    """The soak gate, with one named door per population it can judge.

        from src.robot.grasping.replay.runs import SoakGate

        verdict = SoakGate.over_records("logs/run.jsonl").evaluate()
        print(verdict.render())
        raise SystemExit(verdict.exit_code)

    Two named doors, never a flag: the two populations differ in what they are evidence of, not
    in a parameter. The door a caller typed is recorded on the verdict as `SoakSource` and printed
    by `render()`.
    """

    source: SoakSource
    records_path: Path | None = None
    #: Not defaulted here. `evaluate_soak_gate_over_records` declares ``min_attempts=2000``; a copy of the number in
    #: this class would be a second declaration of one fact, agreeing today and silently diverging later.
    min_attempts: "Maybe[int]" = UNSET

    # --- the two doors -------------------------------------------------------------------------

    @classmethod
    def over_records(cls, path: str | Path) -> "SoakGate":
        """A captured log from a real cell."""
        return cls(source=SoakSource.REAL_RECORDS, records_path=Path(path))

    @classmethod
    def over_sim_records(
        cls, path: str | Path, *, min_attempts: "Maybe[int]" = UNSET
    ) -> "SoakGate":
        """An Isaac-physics log: real picks, not hardware-representative. `pick_success_rate` is reported and not
        gated, and the verdict says so."""
        return cls(
            source=SoakSource.SIM_RECORDS, records_path=Path(path), min_attempts=min_attempts
        )

    # --- the verb ------------------------------------------------------------------------------

    def evaluate(self) -> SoakVerdict:
        """Run the gate. Reads; writes nothing, anywhere."""
        return self._evaluate_records()

    # --- the shape behind both doors -----------------------------------------------------------

    def _evaluate_records(self) -> SoakVerdict:
        from src.robot.grasping.replay.kpi import compute_kpis  # noqa: PLC0415
        from src.robot.grasping.replay.soak import (  # noqa: PLC0415
            evaluate_soak_gate_over_records,
        )
        from src.robot.grasping.telemetry.outcome_logging import (  # noqa: PLC0415
            iter_jsonl,
        )

        assert self.records_path is not None  # both record doors set it
        # The read is the part that fails, and it failed as a traceback. `RecordLog.kpis` has always
        # turned a missing log into a verdict; this door raised `FileNotFoundError` straight through
        # both CLI modes that call it, so a mistyped path printed a stack trace instead of the
        # sentence every neighbouring mode prints.
        if not self.records_path.is_file():
            return SoakVerdict(
                source=self.source,
                origin=str(self.records_path),
                violations=("record log unreadable",),
                unreadable="no such record log",
            )
        try:
            records = tuple(iter_jsonl(self.records_path))
        except (OSError, ValueError) as exc:
            # A log that exists and cannot be parsed is the same kind of answer: nothing was judged.
            return SoakVerdict(
                source=self.source,
                origin=str(self.records_path),
                violations=("record log unreadable",),
                unreadable=f"{type(exc).__name__}: {exc}",
            )
        # Forwarded only when chosen, so an unspecified floor reaches the callee's own 2000 rather
        # than a copy of it made here.
        extra: dict[str, Any] = {}
        if chosen(self.min_attempts):
            extra["min_attempts"] = self.min_attempts
        gate, violations = evaluate_soak_gate_over_records(records, **extra)
        keys = {
            name: (GateKeyStatus.PASSED if value else GateKeyStatus.FAILED)
            for name, value in gate.items()
        }
        return SoakVerdict(
            source=self.source,
            origin=str(self.records_path),
            keys=keys,
            violations=tuple(violations),
            attempts=len(records),
            kpi=compute_kpis(records).to_dict(),
        )
