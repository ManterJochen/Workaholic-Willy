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
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord


_DENSE_MODES: frozenset[str] = frozenset(
    {"dense_clutter", "dense_autonomous"}
)


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
