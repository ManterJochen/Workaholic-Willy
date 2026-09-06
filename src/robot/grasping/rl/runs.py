"""The library face of the offline RL commands: a noun, one verb, a typed report.

`check-dataset` judges a record log before anyone trains on it. It opens no cell, loads no policy
and trains nothing.

`assess_records` is the pure function over the records and `format_readiness` is the renderer;
:class:`RecordLogCheck` adds what surrounds them, which is the two ways reading a log fails and the
three exit codes.

The verdict is reported, never raised. `_commands_dataset.py` states that this readiness check is
reported and deliberately not gating, so it is the easiest thing in this pipeline to walk straight
past; a library that raised on an unreadable log would make that decision for its caller, while
:attr:`ReadinessVerdict.UNREADABLE` leaves it where it was.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.robot.grasping.rl._io import load_jsonl
from src.robot.grasping.rl.readiness import (
    DatasetReadiness,
    assess_records,
    format_readiness,
)

__all__ = ["ReadinessVerdict", "RecordLogCheck", "RecordLogReport"]


class ReadinessVerdict(StrEnum):
    """What the check concluded, and the exit code each verdict carries.

    The three are not a severity ladder. `UNREADABLE` is not "worse than not trainable": it means
    the question was never asked, because there was nothing to ask it of. A caller that collapses
    them into a boolean reports a missing file and a bad dataset as the same outcome.
    """

    #: No blocking finding. Exit 0.
    TRAINABLE = "trainable"
    #: The log was read and judged, and something blocks training on it. Exit 1.
    NOT_TRAINABLE = "not_trainable"
    #: The log could not be read at all: no such file, or not parseable. Exit 2.
    UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class RecordLogReport:
    """What the check found. Satisfies both halves of the report contract."""

    verdict: ReadinessVerdict
    #: What was judged, for the report to name: a path, or a description of an in-memory sequence.
    source: str
    #: The numbers behind the verdict. ``None`` only when the log could not be read.
    readiness: DatasetReadiness | None = None
    #: Why it was unreadable. Empty otherwise.
    detail: str = ""

    @property
    def trainable(self) -> bool:
        return self.verdict is ReadinessVerdict.TRAINABLE

    @property
    def exit_code(self) -> int:
        """The process exit code for this verdict.

        On the report, not in the CLI, so no command re-derives it: `_cmd_check_dataset` returns
        `report.exit_code` and owns nothing but its own operator wording. `DoctorReport.exit_code`
        in `safety/planning/doctor.py` does the same.
        """
        return {
            ReadinessVerdict.TRAINABLE: 0,
            ReadinessVerdict.NOT_TRAINABLE: 1,
            ReadinessVerdict.UNREADABLE: 2,
        }[self.verdict]

    def render(self) -> str:
        """The whole report: the verdict, the counts, and the per-column detail behind them."""
        if self.readiness is None:
            return f"  UNREADABLE: {self.source}{f': {self.detail}' if self.detail else ''}"
        return format_readiness(self.readiness, verbose=True)

    def summary(self) -> str:
        """The verdict and its counts, without the per-column tables.

        A fragment, which is why it is not called `render`. The convention is that `render()`
        returns the whole object and takes no arguments; a shorter view keeps its own name. Both go
        through `format_readiness`, so there is still exactly one implementation of this text.
        """
        if self.readiness is None:
            return self.render()
        return format_readiness(self.readiness, verbose=False)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, `json.dumps`-safe with no custom encoder."""
        readiness = self.readiness
        return {
            "verdict": self.verdict.value,
            "source": self.source,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "records": readiness.records if readiness else 0,
            "groups": readiness.groups if readiness else 0,
            "pairs": readiness.pairs if readiness else 0,
            "successes": readiness.successes if readiness else 0,
            "failures": readiness.failures if readiness else 0,
            "candidate_rows": readiness.candidate_rows if readiness else 0,
            "blocking": list(readiness.blocking) if readiness else [],
            "warnings": list(readiness.warnings) if readiness else [],
            "notes": list(readiness.notes) if readiness else [],
            "live_per_candidate": list(readiness.live_per_candidate) if readiness else [],
        }


@dataclass(frozen=True, slots=True)
class RecordLogCheck:
    """Judge a `GraspAttemptRecord` log before anyone trains on it.

        from src.robot.grasping.rl.runs import RecordLogCheck

        report = RecordLogCheck.from_records_log("logs/run.jsonl").assess()
        print(report.render())
        if not report.trainable:
            ...

    Runs on any record log, wherever it came from: a customer's own cell, a sim runner, a replay
    pack. It opens no cell, loads no policy and trains nothing.
    """

    #: Exactly one of these is set. The path is read by :meth:`assess`, not by the factory, because
    #: reading is the part that can fail and a factory that raised would move the failure to
    #: construction, where the caller has no report to put it in.
    path: Path | None = None
    records: tuple[Mapping[str, Any], ...] | None = None

    @classmethod
    def from_records_log(cls, path: str | Path) -> "RecordLogCheck":
        """A JSONL log on disk. Nothing is read until :meth:`assess`."""
        return cls(path=Path(path))

    @classmethod
    def from_records(cls, records: Sequence[Mapping[str, Any]]) -> "RecordLogCheck":
        """Records already in memory, from a run that has just finished, say."""
        return cls(records=tuple(records))

    def assess(self) -> RecordLogReport:
        """Read what is needed, judge it, and report. Never raises.

        A missing file and an unparseable one both reach the caller as `UNREADABLE`, with the
        reason in `detail` and exit code 2, which is what the CLI prints for them.
        """
        records = self.records
        source = str(self.path) if self.path is not None else f"{len(records or ())} record(s)"

        if records is None:
            if self.path is None:  # pragma: no cover (neither factory can produce this)
                return RecordLogReport(ReadinessVerdict.UNREADABLE, "nothing",
                                       detail="no path and no records")
            if not self.path.is_file():
                return RecordLogReport(ReadinessVerdict.UNREADABLE, source,
                                       detail="no such record log")
            try:
                records = tuple(load_jsonl(self.path))
            except Exception as exc:  # noqa: BLE001 (a log we cannot read is a verdict, not a crash)
                return RecordLogReport(ReadinessVerdict.UNREADABLE, source,
                                       detail=f"{type(exc).__name__}: {exc}")

        readiness = assess_records(list(records))
        verdict = (ReadinessVerdict.TRAINABLE if readiness.trainable
                   else ReadinessVerdict.NOT_TRAINABLE)
        return RecordLogReport(verdict, source, readiness=readiness)
