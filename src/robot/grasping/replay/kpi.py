"""KPI primitive over :class:`GraspAttemptRecord` sequences.

The harness is stateless: it consumes the production logging format
and emits an immutable summary.

Every KPI definition below mirrors the locked KPI contract and
states it in more detail than that contract did. Cycle-time is read
from ``record.extra["cycle_time_s"]`` when present; the field is
optional and the median is ``None`` when no record reports a value.

Failure-class semantics:

* ``pick_success_rate``: ``final_outcome == "succeeded"``.
* ``first_attempt_success_rate``: succeeded with empty
  ``recovery_actions``.
* ``dead_loop_rate``: ``final_outcome == "recovery_exhausted"``.
  This is the only outcome that signals an attempt ran past its
  budget; ``decision_fail_closed`` is a typed terminal and
  explicitly not a dead loop.
* ``safety_rejection_rate``: ``extra["safety_rejected"] is True``.
* ``dense_recovery_success_rate``: among dense-mode attempts that
  invoked at least one recovery action, the fraction that
  ultimately succeeded.
* ``false_positive_grasp_rate``: among reported successes, the
  fraction where ``extra["verification_failed_after_success"]`` is
  truthy (operator/audit-time disagreement with the success
  signal).

Some of these rates have no source, and this module names them rather than returning a number.
:func:`_ratio` returns ``0.0`` for an empty denominator, so a rate nothing feeds arrives looking like
a measurement of zero: on a report that reads as perfect, or as broken, and both are claims nobody
made. :data:`UNMEASURABLE_KPIS` and :func:`unmeasurable_kpis` are the honest half, and they live here
rather than in the caller that first needed them. The operator console (``api/history.py``) grew them
first and was for a while the only surface that said so, while
``python -m src.robot.grasping.replay --records`` printed ``false_positive_grasp_rate: 0.0`` as a
measurement over the same records. One definition, both readers.

:meth:`KpiSummary.to_dict` still emits every key. It is the frozen wire shape the committed soak and
baseline reports are built from, and those reports name the same rates in their own provenance
blocks. Withholding happens where a number is presented to somebody, not in the arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord


_DENSE_MODES: frozenset[str] = frozenset(
    {"dense_clutter", "dense_autonomous"}
)


#: Maps a KPI name to why no number can be shown for it, whatever the records say. Not a display
#: nicety: the value this computes to is not "unknown", it is a confident zero derived from a field
#: nothing on this stack writes.
UNMEASURABLE_KPIS: dict[str, str] = {
    "false_positive_grasp_rate": (
        "No number can be given. This rate counts grasps that reported success and were actually "
        "empty, and observing one requires an independent post-grasp re-check that this stack does "
        "not have, so the field it is computed from is never set and the arithmetic returns a "
        "structural 0.0. A 0.0% false-positive rate on a screen is a claim nobody made."
    ),
}

#: The same idea, decided per record set: these three can be measured on records that carry what
#: they need. On a set that does not, the arithmetic still returns a number, and that number is an
#: empty denominator wearing the clothes of a measurement.
CONDITIONALLY_UNMEASURABLE: dict[str, str] = {
    "dense_recovery_success_rate": (
        "Not measurable from these records: none of them both ran in a dense mode and recorded a "
        "recovery action, which is what this rate divides by. The denominator is empty, so the rate "
        "computes to a structural 0.0, reading as 'recovery never works' rather than 'recovery was "
        "never asked to'. The shipped default mode is 'auto'."
    ),
    "first_attempt_success_rate": (
        "Not measurable from these records: none of them recorded a recovery action. This rate "
        "differs from the pick success rate only by excluding attempts that needed one, so with none "
        "it is the same number printed twice, implying a measured cost of recovery that was never "
        "measured."
    ),
    "median_cycle_time_s": (
        "Not measurable from these records: no record carries a cycle time. No writer on the "
        "production pick path sets extra['cycle_time_s']: it is not in the frozen telemetry "
        "catalog, and the writers that do exist are synthetic or sim-side. The honest duration for "
        "these records is median_attempt_seconds."
    ),
}


def _has_support(
    name: str,
    records: Sequence[GraspAttemptRecord],
    summary: "KpiSummary",
) -> bool:
    """Does this record set actually contain what the named KPI divides by?

    Deliberately checks the denominator, not the result. A recovery rate of 0.0 over 200 dense
    attempts that all needed recovery is a real and alarming measurement; the same 0.0 over none of
    them is not a measurement at all, and only looking at the input can tell the two apart.

    Each branch mirrors its counter in :func:`compute_kpis` exactly, and that is the whole
    correctness condition here. The first cut of this function, written in the console layer, got
    ``dense_recovery_success_rate`` wrong in a way worth keeping written down: ``compute_kpis``
    counts a record into that denominator only when it is ``mode in _DENSE_MODES`` and
    ``bool(recovery_actions)``, and the guard tested the mode half alone. A log of 200 dense-mode
    attempts that never recovered, which several shipped logs are, for example
    ``logs/s4_sim_soak.jsonl`` at 330/330/0, therefore passed the guard and published
    ``_ratio(0, 0) = 0.0``. Worse, it contradicted itself inside one response: the same set had
    ``first_attempt_success_rate`` withheld for having no recovery actions while this rate was
    published over that identical empty input. A guard that checks half a conjunction is not a guard.
    """
    if name == "dense_recovery_success_rate":
        return any(
            str(record.mode) in _DENSE_MODES and bool(record.recovery_actions)
            for record in records
        )
    if name == "first_attempt_success_rate":
        return any(record.recovery_actions for record in records)
    if name == "median_cycle_time_s":
        return summary.median_cycle_time_s is not None
    return True


def unmeasurable_kpis(
    records: Iterable[GraspAttemptRecord] | Sequence[GraspAttemptRecord],
    summary: "Optional[KpiSummary]" = None,
) -> dict[str, str]:
    """Maps a KPI name to why these records cannot measure it. Empty means every rate has a source.

    The names are the load-bearing half. A KPI absent because its denominator is empty looks, on a
    screen or in a JSON payload, exactly like one nobody asked for; a KPI present with a structural
    0.0 looks like a measurement. Naming them per record set is what tells those three apart.

    ``summary`` is accepted so a caller that already computed one does not pay for a second pass.
    """
    rows = tuple(records)
    resolved = summary if summary is not None else compute_kpis(rows)
    withheld = dict(UNMEASURABLE_KPIS)
    for name, reason in CONDITIONALLY_UNMEASURABLE.items():
        if not _has_support(name, rows, resolved):
            withheld[name] = reason
    return withheld


@dataclass(frozen=True, slots=True)
class KpiSummary:
    """Immutable, JSON-safe KPI snapshot."""

    total_attempts: int
    pick_success_rate: float
    first_attempt_success_rate: float
    dead_loop_rate: float
    safety_rejection_rate: float
    median_cycle_time_s: Optional[float]
    dense_recovery_success_rate: float
    false_positive_grasp_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_attempts": int(self.total_attempts),
            "pick_success_rate": float(self.pick_success_rate),
            "first_attempt_success_rate": float(
                self.first_attempt_success_rate
            ),
            "dead_loop_rate": float(self.dead_loop_rate),
            "safety_rejection_rate": float(self.safety_rejection_rate),
            "median_cycle_time_s": (
                None
                if self.median_cycle_time_s is None
                else float(self.median_cycle_time_s)
            ),
            "dense_recovery_success_rate": float(
                self.dense_recovery_success_rate
            ),
            "false_positive_grasp_rate": float(
                self.false_positive_grasp_rate
            ),
        }


def _ratio(num: int, denom: int) -> float:
    return float(num) / float(denom) if denom else 0.0


def _truthy_extra(extra: Mapping[str, Any], key: str) -> bool:
    return bool(extra.get(key))


def compute_kpis(
    records: Iterable[GraspAttemptRecord] | Sequence[GraspAttemptRecord],
) -> KpiSummary:
    """Return a :class:`KpiSummary` over ``records``.

    Empty input gives all-zero rates and a ``None`` median.
    """

    records = tuple(records)
    total = len(records)
    if total == 0:
        return KpiSummary(
            total_attempts=0,
            pick_success_rate=0.0,
            first_attempt_success_rate=0.0,
            dead_loop_rate=0.0,
            safety_rejection_rate=0.0,
            median_cycle_time_s=None,
            dense_recovery_success_rate=0.0,
            false_positive_grasp_rate=0.0,
        )

    succeeded = 0
    first_try_success = 0
    dead_loop = 0
    safety_rejected = 0
    cycle_times: list[float] = []
    dense_with_recovery = 0
    dense_recovery_success = 0
    reported_success = 0
    false_positive = 0

    for r in records:
        extra = dict(r.extra) if r.extra else {}
        is_success = r.final_outcome == "succeeded"
        has_recovery = bool(r.recovery_actions)
        if is_success:
            succeeded += 1
            if not has_recovery:
                first_try_success += 1
            reported_success += 1
            if _truthy_extra(extra, "verification_failed_after_success"):
                false_positive += 1
        if r.final_outcome == "recovery_exhausted":
            dead_loop += 1
        if _truthy_extra(extra, "safety_rejected"):
            safety_rejected += 1
        ct = extra.get("cycle_time_s")
        if isinstance(ct, (int, float)) and ct == ct:  # not NaN
            cycle_times.append(float(ct))
        if r.mode in _DENSE_MODES and has_recovery:
            dense_with_recovery += 1
            if is_success:
                dense_recovery_success += 1

    return KpiSummary(
        total_attempts=total,
        pick_success_rate=_ratio(succeeded, total),
        first_attempt_success_rate=_ratio(first_try_success, total),
        dead_loop_rate=_ratio(dead_loop, total),
        safety_rejection_rate=_ratio(safety_rejected, total),
        median_cycle_time_s=(median(cycle_times) if cycle_times else None),
        dense_recovery_success_rate=_ratio(
            dense_recovery_success, dense_with_recovery
        ),
        false_positive_grasp_rate=_ratio(false_positive, reported_success),
    )
