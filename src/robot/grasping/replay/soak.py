"""The soak gate over a record log: the thresholds a run of picks is judged by, from its records alone.

``--records-gate`` judges a real cell's log and ``--sim-soak-report`` an Isaac-physics one (:class:`~.runs.SoakGate`).
Each of the six keys is judgeable from the records alone: enough attempts, every outcome typed, no unbounded retry
loop, every record carrying the telemetry its outcome requires and no field of the wrong type, and the dead-loop rate
under its ceiling. The synthetic soak, its canonical packs and the baseline it compared the pick rate against left on
2026-10-16 (the owner's cleanup): they judged invented picks and said nothing about grasping.
"""

from __future__ import annotations

from typing import Sequence

from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome
from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord
from src.robot.grasping.replay.kpi import compute_kpis
from src.robot.grasping.replay.telemetry_catalog import (
    TELEMETRY_CATALOG,
    audit_records,
    audit_extra_records,
)

#: Every outcome the catalog knows, the retired ones by their strings included, so a record logged before an outcome
#: left is still typed.
_VALID_OUTCOMES: frozenset[str] = frozenset(
    v.value for v in AutonomousGraspOutcome
) | frozenset(TELEMETRY_CATALOG)

#: Minimum attempts the soak gate requires.
SOAK_MIN_ATTEMPTS: int = 2000

#: A recovery_actions count above this is treated as an unbounded
#: retry loop for the gate (no real cell should plan more than a
#: handful of recovery moves on a single pick).
UNBOUNDED_RECOVERY_THRESHOLD: int = 8

#: Ceiling on dead_loop_rate across the soak window.
DEAD_LOOP_RATE_MAX: float = 0.005


def evaluate_soak_gate(
    *,
    n_records: int,
    dead_loop_rate: float,
    untyped_count: int,
    unbounded_count: int,
    telemetry_offender_count: int,
    extra_offender_count: int,
    min_attempts: int = SOAK_MIN_ATTEMPTS,
    dead_loop_max: float = DEAD_LOOP_RATE_MAX,
) -> tuple[dict[str, bool], list[str]]:
    """The soak gate's six thresholds over counts already taken from a log. Pure: no I/O.

    Args:
        n_records (int): How many records the log holds.
        dead_loop_rate (float): The log's dead-loop rate (``KpiSummary.dead_loop_rate``).
        untyped_count (int): Records whose outcome no catalog entry types.
        unbounded_count (int): Records with more than :data:`UNBOUNDED_RECOVERY_THRESHOLD` recovery actions.
        telemetry_offender_count (int): Records missing a telemetry field their outcome requires.
        extra_offender_count (int): Records carrying an ``extra`` field of the wrong type.
        min_attempts (int): The floor of attempts; :data:`SOAK_MIN_ATTEMPTS` unless a caller states its own.
        dead_loop_max (float): The ceiling of the dead-loop rate.

    Returns:
        tuple[dict[str, bool], list[str]]: Each key and whether it passed, and one sentence per violation.
    """

    gate: dict[str, bool] = {
        "min_attempts_met": n_records >= min_attempts,
        "untyped_outcomes_zero": untyped_count == 0,
        "unbounded_retry_loops_zero": unbounded_count == 0,
        "telemetry_offenders_zero": telemetry_offender_count == 0,
        "extra_type_offenders_zero": extra_offender_count == 0,
        "dead_loop_rate_within_gate": dead_loop_rate <= dead_loop_max,
    }
    violations: list[str] = []
    if not gate["min_attempts_met"]:
        violations.append(f"min_attempts {n_records} < {min_attempts}")
    if not gate["untyped_outcomes_zero"]:
        violations.append(f"untyped_outcomes={untyped_count}")
    if not gate["unbounded_retry_loops_zero"]:
        violations.append(f"unbounded_retry_loops={unbounded_count}")
    if not gate["telemetry_offenders_zero"]:
        violations.append(f"telemetry_offenders={telemetry_offender_count}")
    if not gate["extra_type_offenders_zero"]:
        violations.append(f"extra_type_offenders={extra_offender_count}")
    if not gate["dead_loop_rate_within_gate"]:
        violations.append(
            f"dead_loop_rate {dead_loop_rate:.4f} > {dead_loop_max:.4f}"
        )
    return gate, violations


def evaluate_soak_gate_over_records(
    records: Sequence[GraspAttemptRecord],
    *,
    min_attempts: int = SOAK_MIN_ATTEMPTS,
) -> tuple[dict[str, bool], list[str]]:
    """The soak gate over a GraspAttemptRecord log (the ``--records-gate`` and ``--sim-soak-report`` paths).

    Args:
        records (Sequence[GraspAttemptRecord]): The log.
        min_attempts (int): The floor of attempts; the ``--sim-soak-report`` path passes a smaller sim floor.

    Returns:
        tuple[dict[str, bool], list[str]]: As :func:`evaluate_soak_gate`.
    """

    summary = compute_kpis(records)
    telemetry_offenders = audit_records(records)
    extra_offenders = audit_extra_records(records)
    untyped = [
        r.attempt_id for r in records if r.final_outcome not in _VALID_OUTCOMES
    ]
    unbounded = [
        r.attempt_id
        for r in records
        if len(getattr(r, "recovery_actions", ()) or ())
        > UNBOUNDED_RECOVERY_THRESHOLD
    ]
    return evaluate_soak_gate(
        n_records=len(records),
        dead_loop_rate=summary.dead_loop_rate,
        untyped_count=len(untyped),
        unbounded_count=len(unbounded),
        telemetry_offender_count=len(telemetry_offenders),
        extra_offender_count=len(extra_offenders),
        min_attempts=min_attempts,
    )
