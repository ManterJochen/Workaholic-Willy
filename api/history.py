"""What actually happened, from the file that survives the process.

Two sources, and the console never blurs them:

* Runs live in memory (``api.runs``). They are rich, carrying candidate counts, scores and the motion
  status chain, and they die when the server does.
* Records live in a JSONL file. They survive restarts and they include what the CLI runner wrote, so
  one history covers both tools. They are also thinner than they look: on a default attempt the
  production serializer fills ``profile``, ``initial_telemetry``, ``execution``, ``selected_grasp``,
  ``recovery_actions`` and ``extra``. ``verification`` and ``refinement`` gained writers on
  2026-09-10 and fill only on an attempt where those layers actually ran, which no shipped config
  turns on. ``frame``, ``target``, ``initial_grasp`` and ``refined_grasp`` have no writer at all.

Anything a UI shows therefore says which of the two it came from, because "the chosen grasp was at
x=312" and "the run succeeded" are claims of very different strength and only one of them survives a
restart.

Some KPIs have no honest source, and they are named rather than zeroed. That is the load-bearing idea
here: ``_ratio(x, 0)`` returns ``0.0``, so a rate computed over an empty denominator arrives looking
like a measurement of zero. On a dashboard that reads as perfect or as broken, and both are claims
nobody made. Four cases, three of them decided per record set rather than once. The reasons and the
denominator tests live in ``replay/kpi.py``, so the offline CLI over the same log says the same
thing; this console consumes them:

* ``false_positive_grasp_rate`` is always withheld. It counts grasps that reported success and were
  actually empty, and observing one needs an independent post-grasp re-check this stack does not have,
  so the field it reads is never written.
* ``dense_recovery_success_rate`` is withheld when no record both ran in a dense mode and recorded a
  recovery action. Both halves, because that is the conjunction ``compute_kpis`` divides by; a guard
  on the mode alone publishes a structural 0.0 over an all-dense log that never recovered. The
  shipped default mode is ``auto``, so on a console-driven cell the denominator is empty anyway.
* ``first_attempt_success_rate`` is withheld when no record carries a recovery action. It differs from
  ``pick_success_rate`` only by excluding attempts that recovered, so with none it is the same number
  printed twice, implying a measured recovery cost that was never measured.
* ``median_cycle_time_s`` is withheld when it is ``None``. No writer on the production pick path sets
  ``extra['cycle_time_s']``: the key is not in the frozen telemetry catalog at all, and the writers
  that do exist are synthetic or sim-side. The honest duration here is ``median_attempt_seconds``.

``extra['attempt_wall_time_s']`` is stamped on every attempt, by ``pick()`` itself, regardless of
config. It is ``median_attempt_seconds`` below, and it is the number to quote when somebody asks how
long a pick takes.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from api.constants import API_LOG_DIR, HISTORY_LOG_FILE
from src.robot.grasping.replay.kpi import (
    UNMEASURABLE_KPIS,
    unmeasurable_kpis,
)
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover
    from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord

__all__ = [
    "UNMEASURABLE_KPIS",
    "load_records",
    "records_csv",
    "rollup",
    "runs_csv",
]

logger = create_logger("ConsoleHistory", HISTORY_LOG_FILE, log_dir=API_LOG_DIR)

#: ``(source, total_attempts)`` of the last roll-up that was logged, so a polled caller does not write
#: the same line forever. The demo screen refreshes the KPI panel every 10 seconds, and a per-request
#: line there would be about 8600 identical entries a day. One line each time the numbers moved is
#: what an operator wants from this log. It gates nothing but the log call: a lost race writes one
#: extra line and changes no answer.
_LAST_ROLLUP_LOGGED: tuple[str, int] | None = None

#: ``UNMEASURABLE_KPIS`` and the per-record-set reasons are imported, not declared here, and it used
#: to be the other way round. They were written for this console and lived only in it, so
#: ``python -m src.robot.grasping.replay --records`` printed ``false_positive_grasp_rate: 0.0`` as a
#: measurement over the exact records this console refused to put a number on: two answers about one
#: log. The definitions now sit beside the arithmetic they qualify (``replay/kpi.py``) and both
#: readers quote the same sentence. This name stays exported from here because the schema and the
#: screens read it.


@dataclass(frozen=True, slots=True)
class Rollup:
    """KPIs over a set of records, with the ones that cannot be measured named rather than zeroed."""

    total_attempts: int
    #: Only the KPIs with an honest source. Keys mirror ``KpiSummary`` field names.
    kpis: dict[str, Any]
    #: Maps a KPI name to why it is absent. Rendered where the number would have been.
    unmeasurable: dict[str, str]
    #: Maps an outcome string to how many attempts ended that way. The typed reason surface, counted.
    outcomes: dict[str, int]
    source: str


def load_records(path: "str | Path | None") -> list["GraspAttemptRecord"]:
    """Every record in ``path``.

    A missing file or ``None`` yields an empty list, which is a real state, not an error: a cell that
    has not logged anything yet is the normal state of a fresh bench, and a history view that returned
    500 for it would send an operator hunting for a fault that is not there.
    """
    if path is None:
        return []
    target = Path(path)
    if not target.exists():
        # Debug, not warning: a bench that has not picked anything yet is the normal first state, and
        # a warning here would train an operator to ignore this file.
        logger.debug("No record log at %s yet; history is empty.", target)
        return []

    from src.robot.grasping.telemetry.outcome_logging import iter_jsonl

    records = list(iter_jsonl(target))
    logger.debug(
        "Read %d record(s) (%d bytes) from %s.", len(records), target.stat().st_size, target
    )
    return records


def rollup(records: list["GraspAttemptRecord"], *, source: str) -> Rollup:
    """Roll ``records`` up with the same function the offline KPI gate uses.

    Not a second implementation: a console that computed its own success rate would eventually
    disagree with ``python -m src.robot.grasping.replay --records``, and then nobody could say which
    number was the real one.
    """
    from src.robot.grasping.replay import compute_kpis

    summary = compute_kpis(records)
    kpis: dict[str, Any] = {
        field: getattr(summary, field)
        for field in (
            "pick_success_rate",
            "first_attempt_success_rate",
            "dead_loop_rate",
            "safety_rejection_rate",
            "median_cycle_time_s",
            "dense_recovery_success_rate",
        )
    }

    # The duration these records do support. Stamped by pick() itself on every attempt, so unlike
    # cycle time it is present on a console cell, a CLI runner and a sim run alike.
    samples = []
    for record in records:
        value = (record.extra or {}).get("attempt_wall_time_s")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            samples.append(float(value))
    median_attempt = _median(samples)
    if median_attempt is not None:
        kpis["median_attempt_seconds"] = round(median_attempt, 3)

    unmeasurable = unmeasurable_kpis(records, summary)
    for name in unmeasurable:
        kpis.pop(name, None)

    outcomes: dict[str, int] = {}
    for record in records:
        key = str(record.final_outcome)
        outcomes[key] = outcomes.get(key, 0) + 1

    # The withheld names are the load-bearing half: a KPI that is absent because its denominator is
    # empty looks, on a screen, exactly like one nobody asked for. Naming them per roll-up means the
    # log says which of the two happened on this record set.
    global _LAST_ROLLUP_LOGGED
    fingerprint = (source, summary.total_attempts)
    if fingerprint != _LAST_ROLLUP_LOGGED:
        _LAST_ROLLUP_LOGGED = fingerprint
        logger.info(
            "Rolled up %d attempt(s) from %s: %d KPI(s) published, %d withheld (%s).",
            summary.total_attempts,
            source,
            len(kpis),
            len(unmeasurable),
            ", ".join(sorted(unmeasurable)) or "none",
        )
    return Rollup(
        total_attempts=summary.total_attempts,
        kpis=kpis,
        unmeasurable=unmeasurable,
        outcomes=dict(sorted(outcomes.items(), key=lambda kv: -kv[1])),
        source=source,
    )


def _median(values: list[float]) -> float | None:
    """The middle value, or ``None`` for an empty set: never 0.0, which would read as a measurement."""
    if not values:
        return None
    import statistics

    return float(statistics.median(values))


def runs_csv(runs: list[Any]) -> str:
    """One row per run: the report-level view. Times are seconds since the epoch, as stored."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        ["run_id", "prompt", "state", "requested_picks", "attempted", "succeeded",
         "started_at", "finished_at", "duration_s", "outcomes", "error"]
    )
    for run in runs:
        duration = (
            round(run.finished_at - run.started_at, 3) if run.finished_at is not None else ""
        )
        writer.writerow([
            run.id, run.prompt, str(run.state), run.requested_picks, run.attempted, run.succeeded,
            run.started_at, run.finished_at if run.finished_at is not None else "",
            duration, "|".join(run.outcomes), run.error,
        ])
    body = buffer.getvalue()
    logger.debug("Exported %d run(s) as CSV (%d bytes).", len(runs), len(body))
    return body


def records_csv(records: list["GraspAttemptRecord"]) -> str:
    """One row per grasp attempt: the analysis view, flattened out of the JSONL.

    Only the blocks the production serializer actually populates get columns. Emitting an empty
    ``selected_grasp`` column for every row would suggest the data exists and was missing this time,
    rather than that it is never written on the live path.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        ["timestamp", "attempt_id", "mode", "final_outcome", "safety_rejected",
         "robot_vendor", "robot_model", "attempt_wall_time_s", "low_level_outcome"]
    )
    for record in records:
        extra = dict(record.extra or {})
        writer.writerow([
            record.timestamp, record.attempt_id, str(record.mode), str(record.final_outcome),
            extra.get("safety_rejected", ""),
            extra.get("robot_vendor", ""), extra.get("robot_model", ""),
            extra.get("attempt_wall_time_s", ""), extra.get("low_level_outcome", ""),
        ])
    body = buffer.getvalue()
    logger.debug("Exported %d grasp attempt(s) as CSV (%d bytes).", len(records), len(body))
    return body
