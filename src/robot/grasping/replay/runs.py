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
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

from src.contracts import UNSET, Maybe, chosen

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord

__all__ = [
    "Baseline",
    "BaselineMeasurement",
    "GateKeyStatus",
    "KpiRollup",
    "PACK_DEPENDENT_GATE_KEYS",
    "RecordLog",
    "SoakGate",
    "SoakSource",
    "SoakVerdict",
    "TelemetryOffender",
    "TelemetryVerdict",
    "committed_baseline_pick_rate",
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
    #: The `KpiSummary`'s own mapping, verbatim. Empty when the log could not be read.
    kpi: Mapping[str, Any] = field(default_factory=dict)
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
            "missing_telemetry": [o.to_dict() for o in self.missing_telemetry],
            "wrong_types": [o.to_dict() for o in self.wrong_types],
        }


@dataclass(frozen=True, slots=True)
class RecordLog:
    """A `GraspAttemptRecord` log, from a cell, a sim runner or a replay pack.

        from src.robot.grasping.replay.runs import RecordLog

        rollup = RecordLog.from_jsonl("logs/run.jsonl").kpis()
        print(rollup.render())
        if not rollup.sound:
            ...   # the numbers are unfounded, not wrong

    Reads a log and computes. It opens no cell, loads no policy and never writes.
    """

    path: Path | None = None
    records: "tuple[GraspAttemptRecord, ...] | None" = None

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "RecordLog":
        """A JSONL log on disk. Nothing is read until :meth:`kpis`.

        The factory does not open the file, so an unreadable log arrives as an UNREADABLE verdict
        from `kpis()` rather than as an exception raised during construction.
        """
        return cls(path=Path(path))

    @classmethod
    def from_records(cls, records: "Sequence[GraspAttemptRecord]") -> "RecordLog":
        """Records already in memory, from a run that has just finished."""
        return cls(records=tuple(records))

    def kpis(self) -> KpiRollup:
        """The KPIs over this log, with the telemetry audits that say whether they rest on anything.

        Three questions, not one. `compute_kpis` alone returns numbers with no indication that the
        records behind them are missing fields, which is the difference between a rate that is
        wrong and one that is unfounded.
        """
        from src.robot.grasping.replay.kpi import compute_kpis  # noqa: PLC0415
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
        return KpiRollup(
            verdict=(
                TelemetryVerdict.INCOMPLETE if (missing or wrong) else TelemetryVerdict.SOUND
            ),
            source=source,
            records=len(rows),
            kpi=summary.to_dict(),
            missing_telemetry=missing,
            wrong_types=wrong,
        )


# =============================================================================================
# The gate, and the baseline it measures against
# =============================================================================================


class SoakSource(StrEnum):
    """What a soak verdict was computed over, which is what it is and is not evidence of.

    `evaluate_soak_gate_over_records` returns the same `dict[str, bool]` whether it ran over a
    real log, a sim log or a generated stream, and nothing in that return value says which. The
    population is recorded here instead: a green gate over the wrong population is how a stack
    gets credited with work it did not do.
    """

    #: A captured `GraspAttemptRecord` log from a real cell. Can fail, and is meant to.
    REAL_RECORDS = "real_records"
    #: A log from an Isaac-physics run. Real picks, not hardware-representative, and
    #: `false_positive_grasp_rate` is structurally 0 because sim has no secondary verifier.
    SIM_RECORDS = "sim_records"
    #: Records generated on the spot from scenario specs. Proves telemetry and KPI consistency, and
    #: says nothing about grasp quality.
    SYNTHETIC_STREAM = "synthetic_stream"
    #: The canonical packs on disk, as written by ``--regenerate-canonical``. The U12 contract
    #: self-check, which passes by construction: a regression tripwire, not a quality measurement.
    CANONICAL_PACKS = "canonical_packs"


class GateKeyStatus(StrEnum):
    """One gate key's outcome.

    Three states, not two. A key that could not be judged is not a key that passed, and folding
    them together is how a gate reports a clean bill of health over four checks it never ran.
    """

    PASSED = "passed"
    FAILED = "failed"
    #: The check needs context this door does not have. Over a bare record log there are no on-disk
    #: canonical packs, so the pack-dependent keys have nothing to judge.
    NOT_APPLICABLE = "not_applicable"


#: The four keys that need the on-disk canonical packs. Both record-facing CLI modes overwrite
#: exactly these with ``"not_applicable"`` after the gate has run, and the doors here read this
#: same tuple.
PACK_DEPENDENT_GATE_KEYS: tuple[str, ...] = (
    "slo_packs_pass",
    "drift_gate_pass",
    "ood_gate_pass",
    "easy_attempt_wall_time_within_budget",
)

#: What each door's verdict is evidence of, in one line, for `render()`.
#:
#: Not the CLI's `note` field. Those are wire values inside a JSON payload, addressed to whoever
#: reads the report file; this is addressed to whoever reads the terminal. Neither is derived from
#: the other, and neither replaces the other.
_MEASURES: dict[SoakSource, str] = {
    SoakSource.REAL_RECORDS: (
        "record-intrinsic integrity over a real log. This can fail, unlike the canonical self-check."
    ),
    SoakSource.SIM_RECORDS: (
        "record-intrinsic integrity over real sim picks. pick_success_rate is reported, not gated: "
        "there is no comparable sim baseline yet. Not hardware-representative."
    ),
    SoakSource.SYNTHETIC_STREAM: (
        "telemetry and KPI consistency over a generated stream. Says nothing about grasp quality."
    ),
    SoakSource.CANONICAL_PACKS: (
        "the U12 contract self-check over the committed packs. Passes by construction: a regression "
        "tripwire, not a quality measurement."
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
    #: The rate this gate compared against, or ``None`` when no comparison was made. `None` here
    #: is a real value meaning "deliberately not compared", not an absent argument.
    baseline_pick_rate: float | None = None
    #: The door's own full machine payload, where it has one beyond the fields above. The canonical
    #: door carries the whole soak report here; the others carry nothing.
    report: Mapping[str, Any] = field(default_factory=dict)

    @property
    def measures(self) -> str:
        """One line on what this verdict is, and is not, evidence of."""
        return _MEASURES[self.source]

    @property
    def passes(self) -> bool:
        return not self.violations

    @property
    def exit_code(self) -> int:
        """Derived from the violations, never stored beside them: ``0 if not violations else 1``,
        which is what all four CLI modes return. A stored copy could disagree with the list it
        summarises.
        """
        return 0 if self.passes else 1

    @property
    def judged(self) -> int:
        """How many keys were actually decided. A gate whose keys are all not_applicable passes
        loudly while judging nothing, and this is the number that gives that away.
        """
        return sum(1 for s in self.keys.values() if s is not GateKeyStatus.NOT_APPLICABLE)

    def gate_wire(self) -> dict[str, Any]:
        """The gate in the exact shape the CLI prints: ``True`` / ``False`` /
        ``"not_applicable"``, plus the derived ``passes``.

        One stored fact, two views. `keys` is the truth; this is the wire projection and
        :meth:`to_dict` is the Python one. Neither can drift from the other, because neither is
        stored.
        """
        wire: dict[str, Any] = {}
        for name, status in self.keys.items():
            if status is GateKeyStatus.NOT_APPLICABLE:
                wire[name] = GateKeyStatus.NOT_APPLICABLE.value
            else:
                wire[name] = status is GateKeyStatus.PASSED
        wire["passes"] = self.passes
        return wire

    def render(self) -> str:
        """The verdict, what it measures, and every violation. ASCII, no trailing newline."""
        head = "PASS" if self.passes else "FAIL"
        lines = [
            f"  {head}: {self.attempts} attempt(s) from {self.origin}",
            f"    measures: {self.measures}",
        ]
        if self.keys:
            skipped = len(self.keys) - self.judged
            tail = f", {skipped} not applicable" if skipped else ""
            lines.append(f"    {self.judged} of {len(self.keys)} gate key(s) judged{tail}")
        if self.baseline_pick_rate is not None:
            lines.append(f"    baseline pick rate: {self.baseline_pick_rate:.4f}")
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
            "judged_keys": self.judged,
            "keys": {name: status.value for name, status in self.keys.items()},
            "violations": list(self.violations),
            "baseline_pick_rate": self.baseline_pick_rate,
            "kpi": dict(self.kpi),
        }


@dataclass(frozen=True, slots=True)
class SoakGate:
    """The soak gate, with one named door per population it can judge.

        from src.robot.grasping.replay.runs import SoakGate

        verdict = SoakGate.over_records("logs/run.jsonl").evaluate()
        print(verdict.render())
        raise SystemExit(verdict.exit_code)

    Four named doors, never a flag: the four populations differ in what they are evidence of, not
    in a parameter. The door a caller typed is recorded on the verdict as `SoakSource` and printed
    by `render()`.
    """

    source: SoakSource
    records_path: Path | None = None
    thresholds_path: Path | None = None
    #: Not defaulted here. `evaluate_soak_gate_over_records` declares ``min_attempts=2000`` and
    #: `build_soak_report` declares ``total_attempts=2400``; a copy of either number in this class
    #: would be a second declaration of one fact, agreeing today and silently diverging later.
    min_attempts: "Maybe[int]" = UNSET
    total_attempts: "Maybe[int]" = UNSET

    # --- the four doors ------------------------------------------------------------------------

    @classmethod
    def over_records(cls, path: str | Path) -> "SoakGate":
        """A captured log from a real cell, judged against the committed baseline pick rate.

        `evaluate_soak_gate_over_records` takes ``baseline_pick_rate`` as a required keyword and
        does not know where one comes from; `committed_baseline_pick_rate` reads it from
        `docs/baselines/u_plus_baseline_v1.json` under the repo root. `None` there is legal and
        silently skips the non-regression check.
        """
        return cls(source=SoakSource.REAL_RECORDS, records_path=Path(path))

    @classmethod
    def over_sim_records(
        cls, path: str | Path, *, min_attempts: "Maybe[int]" = UNSET
    ) -> "SoakGate":
        """An Isaac-physics log: real picks, not hardware-representative.

        No baseline comparison, on purpose. The committed baseline is a synthetic distribution, so
        comparing sim pick rate against it would be a cross-population claim. `pick_success_rate` is
        reported and not gated, and the verdict says so.
        """
        return cls(
            source=SoakSource.SIM_RECORDS, records_path=Path(path), min_attempts=min_attempts
        )

    @classmethod
    def over_synthetic(cls, thresholds_path: str | Path) -> "SoakGate":
        """Records generated at evaluation time, against a KPI thresholds YAML.

        Judges telemetry and KPI consistency. It cannot judge grasp quality, because it invented the
        grasps.
        """
        return cls(source=SoakSource.SYNTHETIC_STREAM, thresholds_path=Path(thresholds_path))

    @classmethod
    def over_canonical_packs(cls, *, total_attempts: "Maybe[int]" = UNSET) -> "SoakGate":
        """The canonical packs on disk: the U12 contract self-check.

        It passes by construction: a tripwire that fires when the telemetry contract changes
        underneath the packs. It is not a statement about how well the stack grasps anything.
        """
        return cls(source=SoakSource.CANONICAL_PACKS, total_attempts=total_attempts)

    # --- the verb ------------------------------------------------------------------------------

    def evaluate(self) -> SoakVerdict:
        """Run the gate. Reads; writes nothing, anywhere."""
        if self.source is SoakSource.SYNTHETIC_STREAM:
            return self._evaluate_synthetic()
        if self.source is SoakSource.CANONICAL_PACKS:
            return self._evaluate_canonical()
        return self._evaluate_records()

    # --- the three shapes behind the four doors ------------------------------------------------

    def _evaluate_records(self) -> SoakVerdict:
        from src.robot.grasping.replay.kpi import compute_kpis  # noqa: PLC0415
        from src.robot.grasping.replay.soak import (  # noqa: PLC0415
            evaluate_soak_gate_over_records,
        )
        from src.robot.grasping.telemetry.outcome_logging import (  # noqa: PLC0415
            iter_jsonl,
        )

        assert self.records_path is not None  # both record doors set it
        records = tuple(iter_jsonl(self.records_path))
        baseline = (
            committed_baseline_pick_rate()
            if self.source is SoakSource.REAL_RECORDS
            else None
        )
        # Forwarded only when chosen, so an unspecified floor reaches the callee's own 2000 rather
        # than a copy of it made here.
        extra: dict[str, Any] = {}
        if chosen(self.min_attempts):
            extra["min_attempts"] = self.min_attempts
        gate, violations = evaluate_soak_gate_over_records(
            records, baseline_pick_rate=baseline, **extra
        )
        # The pack-dependent keys are added here, not filtered out of the result:
        # `evaluate_soak_gate` returns the seven record-intrinsic keys only, and the four
        # pack-dependent ones never appear in `gate`. Adding them is what makes this door report
        # the eleven keys the CLI reports.
        keys = {
            name: (GateKeyStatus.PASSED if value else GateKeyStatus.FAILED)
            for name, value in gate.items()
        }
        for name in PACK_DEPENDENT_GATE_KEYS:
            keys[name] = GateKeyStatus.NOT_APPLICABLE
        return SoakVerdict(
            source=self.source,
            origin=str(self.records_path),
            keys=keys,
            violations=tuple(violations),
            attempts=len(records),
            kpi=compute_kpis(records).to_dict(),
            baseline_pick_rate=baseline,
        )

    def _evaluate_synthetic(self) -> SoakVerdict:
        from src.robot.grasping.replay.kpi import compute_kpis  # noqa: PLC0415
        from src.robot.grasping.replay.telemetry_catalog import (  # noqa: PLC0415
            audit_records,
        )

        assert self.thresholds_path is not None
        thresholds = _load_kpi_thresholds(self.thresholds_path)
        records = _synthetic_soak_stream(int(thresholds.soak.min_attempts))
        summary = compute_kpis(records)
        offenders = audit_records(records)
        untyped = _untyped_outcomes(records)

        # The key names are the real gate's names, not new ones. All three predicates are the same
        # predicates `evaluate_soak_gate` applies; only the dead-loop threshold differs, because it
        # comes from the YAML rather than the locked constant. Reusing the names is what lets a
        # caller compare this door against the others key by key.
        dead_loop_ok = summary.dead_loop_rate <= thresholds.soak.dead_loop_rate_max
        keys = {
            "dead_loop_rate_within_gate": (
                GateKeyStatus.PASSED if dead_loop_ok else GateKeyStatus.FAILED
            ),
            "telemetry_offenders_zero": (
                GateKeyStatus.PASSED if not offenders else GateKeyStatus.FAILED
            ),
            "untyped_outcomes_zero": (
                GateKeyStatus.PASSED if not untyped else GateKeyStatus.FAILED
            ),
        }
        violations: list[str] = []
        if not dead_loop_ok:
            violations.append(
                f"dead_loop_rate {summary.dead_loop_rate:.4f} > "
                f"{thresholds.soak.dead_loop_rate_max:.4f}"
            )
        if offenders:
            violations.append(f"telemetry_offenders={len(offenders)}")
        if untyped:
            violations.append(f"untyped_outcomes={len(untyped)}")
        return SoakVerdict(
            source=self.source,
            origin=str(self.thresholds_path),
            keys=keys,
            violations=tuple(violations),
            attempts=len(records),
            kpi=summary.to_dict(),
        )

    def _evaluate_canonical(self) -> SoakVerdict:
        from src.robot.grasping.replay.canonical_datasets import (  # noqa: PLC0415
            repo_root_from_module,
        )
        from src.robot.grasping.replay.soak import build_soak_report  # noqa: PLC0415

        extra: dict[str, Any] = {}
        if chosen(self.total_attempts):
            extra["total_attempts"] = self.total_attempts
        repo_root = repo_root_from_module()
        payload, violations = build_soak_report(repo_root, **extra)
        # `build_soak_report` is typed `dict[str, object]`, so both reads need a cast. Narrowed
        # here rather than loosening the builder, which every other caller relies on as it is.
        gate = dict(cast("Mapping[str, Any]", payload.get("gate") or {}))
        gate.pop("passes", None)  # derived on the verdict, never stored beside the violations
        keys = {
            name: (
                GateKeyStatus.NOT_APPLICABLE
                if value == GateKeyStatus.NOT_APPLICABLE.value
                else (GateKeyStatus.PASSED if value else GateKeyStatus.FAILED)
            )
            for name, value in gate.items()
        }
        return SoakVerdict(
            source=self.source,
            origin=str(repo_root),
            keys=keys,
            violations=tuple(violations),
            attempts=int(cast("int", payload.get("total_attempts") or 0)),
            report=payload,
        )


@dataclass(frozen=True, slots=True)
class BaselineMeasurement:
    """The KPI / SLO / telemetry baseline over the canonical packs, as measured now."""

    #: The builder's payload verbatim. Not reshaped, not re-derived: this dict is compared against
    #: a committed one, so a view that rearranged it would make the comparison meaningless.
    report: Mapping[str, Any]
    audit_offenders: int

    @property
    def clean(self) -> bool:
        return self.audit_offenders == 0

    @property
    def exit_code(self) -> int:
        """2 for a dirty audit, not 1: `--baseline-report` answers a different question from the
        soak gates, whose 1 means a violated threshold. CI reads the number.
        """
        return 0 if self.clean else 2

    def write(self, path: str | Path) -> Path:
        """Write the report where the CLI writes it: sorted keys, indent 2, one trailing newline.

        Serialises the report already built by :meth:`Baseline.measure`. `write_baseline_report`
        rebuilds it from the packs instead, which is a second build and a second chance to differ.
        """
        import json  # noqa: PLC0415

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(dict(self.report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return out

    def render(self) -> str:
        """The packs, their record counts and their audit offenders. ASCII, no trailing newline."""
        packs = list(self.report.get("packs") or [])
        head = "CLEAN" if self.clean else f"{self.audit_offenders} AUDIT OFFENDER(S)"
        lines = [f"  {head}: {len(packs)} canonical pack(s)"]
        for pack in packs:
            kpi = dict(pack.get("kpi") or {})
            lines.append(
                f"    {str(pack.get('name', '?')):<32}"
                f"{int(pack.get('record_count', 0)):>6} record(s)  "
                f"pick {float(kpi.get('pick_success_rate', 0.0)):.4f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The report verbatim, with the offender total the CLI derives beside it."""
        return {"report": dict(self.report), "total_audit_offenders": self.audit_offenders}


@dataclass(frozen=True, slots=True)
class Baseline:
    """The committed KPI / SLO / telemetry baseline, and the packs it is measured over.

        from src.robot.grasping.replay.runs import Baseline

        measured = Baseline.canonical().measure()
        print(measured.render())

    One door only: the baseline is defined as the canonical packs in this repository, and a door
    taking arbitrary packs would let a caller produce a file called a baseline that measures
    something else.
    """

    @classmethod
    def canonical(cls) -> "Baseline":
        """The canonical packs under this repository, as written by `--regenerate-canonical`."""
        return cls()

    def measure(self) -> BaselineMeasurement:
        """Build the report. Reads the packs; writes nothing until
        :meth:`BaselineMeasurement.write` is called.
        """
        from src.robot.grasping.replay.baseline_report import (  # noqa: PLC0415
            build_baseline_report,
        )
        from src.robot.grasping.replay.canonical_datasets import (  # noqa: PLC0415
            repo_root_from_module,
        )

        report = build_baseline_report(repo_root_from_module())
        offenders = sum(
            int(pack["telemetry_offender_count"]) + int(pack["extra_type_offender_count"])
            for pack in cast("list[dict[str, Any]]", report["packs"])
        )
        return BaselineMeasurement(report=report, audit_offenders=int(offenders))


# --- shared helpers ----------------------------------------------------------------------------


def committed_baseline_pick_rate() -> float | None:
    """The aggregate pick success rate from the committed baseline, or ``None`` if unreadable.

    Reads `docs/baselines/u_plus_baseline_v1.json` under the repo root and aggregates it.
    ``None`` comes back when that file is absent, malformed or not a dict. ``None`` reaching
    `evaluate_soak_gate_over_records` is a legal value meaning the non-regression check is
    skipped, so the gate stays green without making the comparison.
    """
    import json  # noqa: PLC0415

    from src.robot.grasping.replay.canonical_datasets import (  # noqa: PLC0415
        repo_root_from_module,
    )
    from src.robot.grasping.replay.soak import (  # noqa: PLC0415
        _aggregate_baseline_pick_success_rate,
    )

    path = repo_root_from_module() / "docs" / "baselines" / "u_plus_baseline_v1.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return _aggregate_baseline_pick_success_rate(payload)


def _load_kpi_thresholds(path: Path) -> Any:
    import yaml  # noqa: PLC0415

    from src.config.schema.robot.kpi_schema import KpiThresholdsConfig  # noqa: PLC0415

    with path.open() as fh:
        return KpiThresholdsConfig.model_validate(yaml.safe_load(fh))


def _untyped_outcomes(records: "Sequence[GraspAttemptRecord]") -> list[str]:
    from src.robot.execution.autonomous_grasp import (  # noqa: PLC0415
        AutonomousGraspOutcome,
    )

    valid = frozenset(v.value for v in AutonomousGraspOutcome)
    return [r.attempt_id for r in records if r.final_outcome not in valid]


def _synthetic_soak_stream(total_attempts: int) -> "tuple[GraspAttemptRecord, ...]":
    """The three scenarios `--soak` generates, at the thresholds file's attempt count.

    The seeds and weights are the CLI's, digit for digit. Changing one changes which records are
    generated, which changes the KPIs, which changes whether the gate passes. There is no
    "reasonable default" available here: the only correct values are the ones already in use.
    """
    from src.robot.grasping.replay.soak import (  # noqa: PLC0415
        SoakScenarioSpec,
        generate_soak_records,
    )

    per = total_attempts // 3
    remainder = total_attempts - per * 3
    return (
        *generate_soak_records(
            SoakScenarioSpec(
                name="easy",
                mode="easy",
                attempts=per,
                failure_class_weights={"succeeded": 19.0, "no_valid_grasp": 1.0},
                recovery_success_rate=0.0,
                cycle_time_mean_s=1.5,
                cycle_time_jitter_s=0.2,
                seed=11,
            )
        ),
        *generate_soak_records(
            SoakScenarioSpec(
                name="auto",
                mode="auto",
                attempts=per,
                failure_class_weights={
                    "succeeded": 14.0,
                    "execution_failed": 2.0,
                    "no_valid_grasp": 2.0,
                    "decision_fail_closed": 1.0,
                },
                recovery_success_rate=0.5,
                cycle_time_mean_s=2.5,
                cycle_time_jitter_s=0.5,
                seed=22,
            )
        ),
        *generate_soak_records(
            SoakScenarioSpec(
                name="dense",
                mode="dense_clutter",
                attempts=per + remainder,
                failure_class_weights={
                    "succeeded": 990.0,
                    "execution_failed": 5.0,
                    "no_valid_grasp": 4.0,
                    "recovery_exhausted": 1.0,
                },
                recovery_success_rate=0.6,
                cycle_time_mean_s=3.5,
                cycle_time_jitter_s=0.7,
                seed=33,
            )
        ),
    )
