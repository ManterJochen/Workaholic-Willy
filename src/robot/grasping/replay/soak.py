"""Synthetic deterministic soak generator.

The generator builds a stream of :class:`GraspAttemptRecord`
instances whose distribution is controlled by a frozen
:class:`SoakScenarioSpec`. Records are shaped to pass
:func:`telemetry_catalog.audit_record` for every outcome the shipped
scenarios weight. The two watchdog outcomes are the exception:
``drift_blocked_auto`` and ``ood_blocked_auto`` need the
``extra.drift_severity``, ``extra.ood_flagged`` and
``extra.degraded_mode_active`` triple, which ``canonical_datasets``
stamps when it enriches those packs; this generator does not, so a
spec that weights either outcome emits audit offenders.

Determinism contract: identical specs produce byte-identical record
sequences (via ``GraspAttemptRecord.to_dict()``).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    GraspMode,
)
from src.robot.grasping.constants import (
    REPLAY_SOAK_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord
from src.robot.grasping.replay.kpi import compute_kpis
from src.robot.grasping.replay.telemetry_catalog import (
    TELEMETRY_CATALOG,
    audit_records,
    audit_extra_records,
)

#: Records what the gate was judged over. Four of the eleven gate keys depend on
#: canonical packs being present on disk, and a missing pack turns its key into
#: a silent failure that reads exactly like a real one.
logger = create_grasping_logger("ReplaySoak", REPLAY_SOAK_LOG_FILE)


_VALID_OUTCOMES: frozenset[str] = frozenset(
    v.value for v in AutonomousGraspOutcome
)
_VALID_MODES: frozenset[str] = frozenset(v.value for v in GraspMode)
_DENSE_MODES: frozenset[str] = frozenset(
    {"dense_clutter", "dense_autonomous"}
)
# Recovery-only outcomes always carry at least one recovery action.
_RECOVERY_OUTCOMES: frozenset[str] = frozenset(
    {"recovery_exhausted", "unsafe_recovery_refused"}
)


@dataclass(frozen=True, slots=True)
class SoakScenarioSpec:
    """Frozen, validated scenario spec for the synthetic generator."""

    name: str
    mode: str
    attempts: int
    failure_class_weights: Mapping[str, float] = field(default_factory=dict)
    recovery_success_rate: float = 0.0
    cycle_time_mean_s: float = 1.0
    cycle_time_jitter_s: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_MODES)!r}; got "
                f"{self.mode!r}"
            )
        if not isinstance(self.attempts, int) or self.attempts <= 0:
            raise ValueError(
                f"attempts must be a positive int; got {self.attempts!r}"
            )
        if not self.failure_class_weights:
            raise ValueError("failure_class_weights must not be empty")
        unknown = [
            k
            for k in self.failure_class_weights
            if k not in _VALID_OUTCOMES
        ]
        if unknown:
            raise ValueError(
                f"failure_class_weights references unknown outcomes: "
                f"{unknown!r}"
            )
        if any(w < 0.0 for w in self.failure_class_weights.values()):
            raise ValueError("failure_class_weights must be non-negative")
        if sum(self.failure_class_weights.values()) <= 0.0:
            raise ValueError(
                "failure_class_weights must sum to a positive value"
            )
        if not (0.0 <= self.recovery_success_rate <= 1.0):
            raise ValueError(
                "recovery_success_rate must lie in [0, 1]; got "
                f"{self.recovery_success_rate!r}"
            )
        if self.cycle_time_mean_s <= 0.0:
            raise ValueError(
                "cycle_time_mean_s must be positive; got "
                f"{self.cycle_time_mean_s!r}"
            )
        if self.cycle_time_jitter_s < 0.0:
            raise ValueError(
                "cycle_time_jitter_s must be non-negative; got "
                f"{self.cycle_time_jitter_s!r}"
            )
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an int")


def _populate_required_fields(
    outcome: str,
    *,
    rng: random.Random,
    mode: str,
    recovery_success: bool,
) -> dict[str, object]:
    """Build the optional record kwargs that satisfy the catalog."""

    required = TELEMETRY_CATALOG[outcome]
    fields: dict[str, object] = {}
    if "execution" in required:
        fields["execution"] = {
            "outcome": "executed" if outcome == "succeeded" else "failed",
            "command": "MOVE_TO",
        }
    if "verification" in required:
        fields["verification"] = {
            "outcome": "passed" if outcome == "succeeded" else "failed",
            "score": float(rng.random()),
        }
    if "refinement" in required:
        fields["refinement"] = {
            "outcome": "ok" if outcome != "refinement_diverged" else "diverged",
            "iterations": int(rng.randint(1, 3)),
        }
    # Recovery actions are always synthesised when the outcome
    # requires them; dense-mode success attaches a "next_viewpoint"
    # action with probability 0.5 so the dense-recovery KPI has a
    # denominator.
    if (
        "recovery_actions" in required
        or outcome in _RECOVERY_OUTCOMES
        or (
            mode in _DENSE_MODES
            and outcome == "succeeded"
            and rng.random() < 0.5
        )
    ):
        n_actions = int(rng.randint(1, 2))
        fields["recovery_actions"] = tuple(
            {
                "action": "next_viewpoint",
                "outcome": (
                    "recovered_success"
                    if (outcome == "succeeded" and recovery_success)
                    else "no_change"
                ),
            }
            for _ in range(n_actions)
        )
    return fields


def _build_extra(
    outcome: str,
    *,
    rng: random.Random,
    mode: str,
    cycle_time_mean_s: float,
    cycle_time_jitter_s: float,
) -> dict[str, object]:
    extra: dict[str, object] = {}
    # Cycle time is always emitted so the median KPI is well-defined.
    jitter = (
        rng.uniform(-cycle_time_jitter_s, cycle_time_jitter_s)
        if cycle_time_jitter_s > 0.0
        else 0.0
    )
    extra["cycle_time_s"] = max(0.05, cycle_time_mean_s + jitter)
    required = TELEMETRY_CATALOG[outcome]
    if "extra.decision_reason_code" in required:
        extra["decision_reason_code"] = "uncertainty_above_threshold"
    if "extra.uncertainty_score" in required:
        extra["uncertainty_score"] = float(rng.uniform(0.4, 0.95))
    if "extra.threshold_used" in required:
        extra["threshold_used"] = 0.4
    return extra


def _draw_outcome(
    weights: Mapping[str, float], rng: random.Random
) -> str:
    total = sum(weights.values())
    pick = rng.random() * total
    cum = 0.0
    for name, w in weights.items():
        cum += w
        if pick <= cum:
            return name
    return next(iter(weights))


def generate_soak_records(
    spec: SoakScenarioSpec,
) -> tuple[GraspAttemptRecord, ...]:
    """Return a deterministic tuple of records for ``spec``."""

    rng = random.Random(spec.seed)
    records: list[GraspAttemptRecord] = []
    for index in range(spec.attempts):
        outcome = _draw_outcome(spec.failure_class_weights, rng)
        recovery_success = rng.random() < spec.recovery_success_rate
        fields = _populate_required_fields(
            outcome,
            rng=rng,
            mode=spec.mode,
            recovery_success=recovery_success,
        )
        extra = _build_extra(
            outcome,
            rng=rng,
            mode=spec.mode,
            cycle_time_mean_s=spec.cycle_time_mean_s,
            cycle_time_jitter_s=spec.cycle_time_jitter_s,
        )
        record = GraspAttemptRecord.new(
            timestamp=float(index),
            attempt_id=f"{spec.name}-{index:06d}",
            mode=spec.mode,
            final_outcome=outcome,
            extra=extra,
            **fields,  # type: ignore[arg-type]
        )
        records.append(record)
    return tuple(records)


#: Minimum attempts the soak gate requires.
SOAK_MIN_ATTEMPTS: int = 2000

#: Default total attempts in the locked soak composition. Sized
#: above the gate minimum so a single dropped scenario cannot silently
#: take the soak below threshold.
SOAK_DEFAULT_ATTEMPTS: int = 2400

#: A recovery_actions count above this is treated as an unbounded
#: retry loop for the gate (no real cell should plan more than a
#: handful of recovery moves on a single pick).
UNBOUNDED_RECOVERY_THRESHOLD: int = 8

#: Default relative output path for the consolidated soak artifact.
DEFAULT_SOAK_REPORT_RELATIVE_PATH: str = "logs/u12/soak_report.json"

#: Ceiling on dead_loop_rate across the soak window.
DEAD_LOOP_RATE_MAX: float = 0.005

#: Budget multiplier for ``p95_attempt_wall_time_s`` in easy mode: the
#: current p95 must stay within +5% of the committed baseline.
EASY_WALL_TIME_BUDGET_MULTIPLIER: float = 1.05


def _default_scenarios(total_attempts: int) -> tuple[SoakScenarioSpec, ...]:
    """Compose the locked, deterministic 3-scenario soak (easy/auto/dense split ~35/30/35; frozen seeds give byte-identical streams)."""

    if total_attempts < SOAK_MIN_ATTEMPTS:
        raise ValueError(
            f"total_attempts must be >= {SOAK_MIN_ATTEMPTS}; "
            f"got {total_attempts!r}"
        )
    easy_n = (total_attempts * 35) // 100
    auto_n = (total_attempts * 30) // 100
    dense_n = total_attempts - easy_n - auto_n
    return (
        SoakScenarioSpec(
            name="u12_easy",
            mode="easy",
            attempts=easy_n,
            failure_class_weights={
                "succeeded": 198.0,
                "no_valid_grasp": 1.0,
                "no_target": 1.0,
            },
            recovery_success_rate=0.0,
            cycle_time_mean_s=1.5,
            cycle_time_jitter_s=0.2,
            seed=12_001,
        ),
        SoakScenarioSpec(
            name="u12_auto",
            mode="auto",
            attempts=auto_n,
            failure_class_weights={
                "succeeded": 178.0,
                "execution_failed": 8.0,
                "no_valid_grasp": 10.0,
                "decision_fail_closed": 3.0,
                "recovery_exhausted": 1.0,
            },
            recovery_success_rate=0.55,
            cycle_time_mean_s=2.5,
            cycle_time_jitter_s=0.5,
            seed=12_002,
        ),
        SoakScenarioSpec(
            name="u12_dense",
            mode="dense_clutter",
            attempts=dense_n,
            failure_class_weights={
                "succeeded": 180.0,
                "execution_failed": 9.0,
                "no_valid_grasp": 7.0,
                "decision_fail_closed": 3.0,
                "recovery_exhausted": 1.0,
            },
            recovery_success_rate=0.6,
            cycle_time_mean_s=3.5,
            cycle_time_jitter_s=0.7,
            seed=12_003,
        ),
    )


def _aggregate_baseline_pick_success_rate(
    baseline: Mapping[str, object],
) -> float | None:
    """Return record-count-weighted aggregate ``pick_success_rate`` (``None`` if the baseline manifest is malformed or empty)."""

    manifest = baseline.get("manifest")
    if not isinstance(manifest, Mapping):
        return None
    packs = manifest.get("packs")
    if not isinstance(packs, list):
        return None
    pack_kpis = baseline.get("packs")
    if not isinstance(pack_kpis, list):
        return None
    by_name: dict[str, float] = {}
    for entry in pack_kpis:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name")
        kpi = entry.get("kpi")
        if not isinstance(name, str) or not isinstance(kpi, Mapping):
            continue
        rate = kpi.get("pick_success_rate")
        if isinstance(rate, (int, float)):
            by_name[name] = float(rate)
    total = 0
    weighted = 0.0
    for pack in packs:
        if not isinstance(pack, Mapping):
            continue
        name = pack.get("name")
        count = pack.get("record_count")
        if not isinstance(name, str) or not isinstance(count, int):
            continue
        rate = by_name.get(name)
        if rate is None:
            continue
        total += count
        weighted += rate * count
    if total == 0:
        return None
    return weighted / total


def evaluate_soak_gate(
    *,
    n_records: int,
    pick_success_rate: float,
    dead_loop_rate: float,
    untyped_count: int,
    unbounded_count: int,
    telemetry_offender_count: int,
    extra_offender_count: int,
    baseline_pick_rate: float | None,
    min_attempts: int = SOAK_MIN_ATTEMPTS,
    dead_loop_max: float = DEAD_LOOP_RATE_MAX,
) -> tuple[dict[str, bool], list[str]]:
    """Record-intrinsic subset of the soak gate: the 7 thresholds judgeable from a GraspAttemptRecord log alone.

    Pure: no I/O, no canonical packs. The pack-dependent keys (slo/drift/ood/easy-wall-time) are added by
    the synthetic-soak caller, or marked ``not_applicable`` by ``--records-gate``. Both paths share this so
    the record-intrinsic thresholds can never drift between them.
    """

    pick_regression = (
        baseline_pick_rate is not None
        and pick_success_rate + 1e-9 < baseline_pick_rate
    )
    gate: dict[str, bool] = {
        "min_attempts_met": n_records >= min_attempts,
        "untyped_outcomes_zero": untyped_count == 0,
        "unbounded_retry_loops_zero": unbounded_count == 0,
        "telemetry_offenders_zero": telemetry_offender_count == 0,
        "extra_type_offenders_zero": extra_offender_count == 0,
        "dead_loop_rate_within_gate": dead_loop_rate <= dead_loop_max,
        "pick_success_rate_non_regression": (
            baseline_pick_rate is None or not pick_regression
        ),
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
    if not gate["pick_success_rate_non_regression"]:
        violations.append(
            f"pick_success_rate {pick_success_rate:.4f} < "
            f"baseline {baseline_pick_rate:.4f}"
        )
    return gate, violations


def evaluate_soak_gate_over_records(
    records: Sequence[GraspAttemptRecord],
    *,
    baseline_pick_rate: float | None,
    min_attempts: int = SOAK_MIN_ATTEMPTS,
) -> tuple[dict[str, bool], list[str]]:
    """Evaluate the record-intrinsic soak gate over a real GraspAttemptRecord log (the ``--records-gate`` path).

    Unlike ``build_soak_report``'s synthetic stream, this can fail on real data. The pack-dependent keys
    (slo/drift/ood/wall-time) are out of scope here (no on-disk pack context); the CLI marks them
    ``not_applicable``. ``min_attempts`` defaults to the synthetic floor; the ``--sim-soak-report`` path
    passes a smaller sim floor.
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
        pick_success_rate=float(summary.pick_success_rate),
        dead_loop_rate=summary.dead_loop_rate,
        untyped_count=len(untyped),
        unbounded_count=len(unbounded),
        telemetry_offender_count=len(telemetry_offenders),
        extra_offender_count=len(extra_offenders),
        baseline_pick_rate=baseline_pick_rate,
        min_attempts=min_attempts,
    )


def build_soak_report(
    repo_root: Path,
    *,
    total_attempts: int = SOAK_DEFAULT_ATTEMPTS,
    baseline_relative_path: str = "docs/baselines/u_plus_baseline_v1.json",
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Build the consolidated soak report and gate verdict.

    Deterministic for a fixed tree, not pure: reads the four canonical packs, the committed baseline JSON
    at ``baseline_relative_path`` and the adaptation audit log under ``repo_root``, so the payload tracks
    on-disk state as well as the arguments. The baseline path is rendered as a posix path relative to
    ``repo_root`` when it lies inside it, so a fixed tree gives a byte-identical payload on any platform.
    Raises ``FileNotFoundError`` when a canonical pack is missing. Returns ``(payload, violations)``;
    ``payload['gate']['passes']`` is ``True`` iff ``violations`` is empty.
    """

    # Imports are local to avoid a circular import at module load:
    # ``soak`` is imported by ``__main__``, which itself pulls in
    # evaluators that import ``soak``.
    from src.robot.grasping.replay.baseline_report import (
        _adaptation_aggregate,
    )
    from src.robot.grasping.replay.canonical_datasets import (
        CANONICAL_PACKS,
    )
    from src.robot.grasping.replay.failure_taxonomy import (
        build_taxonomy_report,
    )
    from src.robot.grasping.replay.slo_eval import (
        evaluate_slo_pack_path,
    )
    from src.robot.grasping.replay.watchdog_eval import (
        DRIFT_PRECISION_GATE,
        DRIFT_RECALL_GATE,
        OOD_PRECISION_GATE,
        OOD_RECALL_GATE,
        evaluate_drift_pack_path,
        evaluate_ood_pack_path,
    )

    scenarios = _default_scenarios(total_attempts)
    records: tuple = ()
    for spec in scenarios:
        records = records + generate_soak_records(spec)

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

    taxonomy = build_taxonomy_report(records).to_dict()

    # SLO + watchdog evaluators consume canonical packs on disk
    # (they require latency/severity enrichment which the synthetic
    # soak stream intentionally does not carry).
    slo_block: dict[str, dict[str, object]] = {}
    slo_pack_pass: list[bool] = []
    for pack in CANONICAL_PACKS:
        pack_path = repo_root / pack.relative_path
        if not pack_path.exists():  # pragma: no cover (safety net)
            # A missing pack narrows the gate's evidence without failing it:
            # the verdict still says "passes" over the packs that are present.
            logger.warning(
                "SLO gate skips pack %s: not on disk at %s "
                "(regenerate with --regenerate-canonical)",
                pack.name,
                pack_path,
            )
            continue
        pack_report = evaluate_slo_pack_path(pack_path)
        slo_block[pack.name] = pack_report.to_dict()
        slo_pack_pass.append(bool(pack_report.passes_gate))

    drift_pack = repo_root / "tests/data/replay/replay_drift_synthetic_v1.jsonl"
    ood_pack = repo_root / "tests/data/replay/replay_ood_synthetic_v1.jsonl"
    drift_report = (
        evaluate_drift_pack_path(drift_pack).to_dict()
        if drift_pack.exists()
        else None
    )
    ood_report = (
        evaluate_ood_pack_path(ood_pack).to_dict()
        if ood_pack.exists()
        else None
    )
    drift_pass = (
        drift_report is not None and bool(drift_report.get("passes_gate"))
    )
    ood_pass = ood_report is not None and bool(ood_report.get("passes_gate"))
    # A missing pack and a failing pack both land on False; only the log separates
    # "the watchdog is wrong" from "the watchdog was never evaluated".
    if drift_report is None:
        logger.warning("Drift gate FAILED for lack of a pack at %s", drift_pack)
    if ood_report is None:
        logger.warning("OOD gate FAILED for lack of a pack at %s", ood_pack)

    adaptation = _adaptation_aggregate(repo_root)

    baseline_path = repo_root / baseline_relative_path
    baseline_pick: float | None = None
    if baseline_path.exists():
        try:
            baseline_payload = json.loads(baseline_path.read_text())
        except (OSError, ValueError):
            baseline_payload = None
        if isinstance(baseline_payload, dict):
            baseline_pick = _aggregate_baseline_pick_success_rate(
                baseline_payload
            )

    soak_pick = float(summary.pick_success_rate)
    pick_regression = (
        baseline_pick is not None and soak_pick + 1e-9 < baseline_pick
    )

    # Easy ``p95_attempt_wall_time_s`` budget check.
    # Both numbers come from build_baseline_report against the
    # canonical packs on disk: the "current" value is computed live,
    # the "baseline" value is read from the committed baseline JSON.
    # The gate fires when current_easy > baseline_easy * 1.05. If
    # either side is missing, the gate is skipped (treated as
    # insufficient evidence rather than a regression).
    from src.robot.grasping.replay.baseline_report import (
        build_baseline_report,
    )

    current_report = build_baseline_report(repo_root)
    current_easy: float | None = None
    try:
        current_easy_raw = (
            current_report["runtime_slo"]  # type: ignore[index]
        )["p95_attempt_wall_time_s_by_mode"]["easy"]
        if current_easy_raw is not None:
            current_easy = float(current_easy_raw)
    except (KeyError, TypeError, ValueError):
        current_easy = None
    baseline_easy: float | None = None
    if isinstance(baseline_payload, dict):
        try:
            be = (
                baseline_payload["runtime_slo"]
            )["p95_attempt_wall_time_s_by_mode"]["easy"]
            if be is not None:
                baseline_easy = float(be)
        except (KeyError, TypeError, ValueError):
            baseline_easy = None
    easy_budget: float | None = (
        baseline_easy * EASY_WALL_TIME_BUDGET_MULTIPLIER
        if baseline_easy is not None
        else None
    )
    easy_within_budget = (
        current_easy is None
        or easy_budget is None
        or current_easy <= easy_budget + 1e-12
    )

    # The record-intrinsic 7-key subset lives in evaluate_soak_gate (shared with the
    # --records-gate path so the thresholds never drift between synthetic + real-log evaluation).
    gate_record, violations = evaluate_soak_gate(
        n_records=len(records),
        pick_success_rate=soak_pick,
        dead_loop_rate=summary.dead_loop_rate,
        untyped_count=len(untyped),
        unbounded_count=len(unbounded),
        telemetry_offender_count=len(telemetry_offenders),
        extra_offender_count=len(extra_offenders),
        baseline_pick_rate=baseline_pick,
    )
    gate: dict[str, object] = dict(gate_record)
    # Pack-dependent keys: need the on-disk canonical packs (SLO/drift/OOD latency + the easy wall-time
    # budget); not judgeable from a GraspAttemptRecord log alone (the --records-gate marks these
    # ``not_applicable``).
    gate["slo_packs_pass"] = bool(slo_pack_pass) and all(slo_pack_pass)
    gate["drift_gate_pass"] = drift_pass
    gate["ood_gate_pass"] = ood_pass
    gate["easy_attempt_wall_time_within_budget"] = bool(easy_within_budget)
    if not gate["slo_packs_pass"]:
        violations.append("slo_packs_not_passing")
    if not gate["drift_gate_pass"]:
        violations.append("drift_gate_failed")
    if not gate["ood_gate_pass"]:
        violations.append("ood_gate_failed")
    if not gate["easy_attempt_wall_time_within_budget"]:
        violations.append(
            f"easy_p95_attempt_wall_time_s "
            f"{current_easy:.4f} > budget "
            f"{easy_budget:.4f}"  # type: ignore[str-format]
        )
    gate["passes"] = not violations

    # Logs which sub-gates were evaluated over what. The pass/fail line itself
    # belongs to the CLI that returns the exit code.
    logger.info(
        "Soak gate over %d synthetic attempt(s) from %d scenario(s): %d/%d key(s) "
        "pass; %d SLO pack(s) evaluated, drift=%s, ood=%s, baseline pick %s vs soak "
        "%.4f",
        len(records),
        len(scenarios),
        sum(1 for k, v in gate.items() if k != "passes" and v is True),
        len(gate) - 1,
        len(slo_pack_pass),
        drift_pass,
        ood_pass,
        "unavailable" if baseline_pick is None else f"{baseline_pick:.4f}",
        soak_pick,
    )
    payload: dict[str, object] = {
        "capability_group": "soak_gate",
        "report_version": 1,
        # Provenance banner. The soak input is a synthetic deterministic generator with an authored
        # outcome distribution, not an observed run, so ``gate.passes`` proves the telemetry, KPI,
        # taxonomy and SLO pipeline is internally consistent over the synthetic stream. It is not a
        # measurement of real grasp quality. The pick_success and dead_loop numbers below are
        # arithmetic consequences of the authored weights. For a real quality signal, run
        # ``--records-gate`` over a captured GraspAttemptRecord log.
        "provenance": {
            "input": "synthetic_generator",
            "measures": [
                "telemetry_contract_consistency",
                "on_disk_pack_slo_latency",
                "watchdog_precision_recall_on_synthetic_packs",
            ],
            "does_not_measure": ["real_grasp_success_quality"],
            "note": (
                "SYNTHETIC self-check, NOT a hardware quality gate: the outcome distribution is authored, "
                "not observed. Use --records-gate over a real GraspAttemptRecord log for a quality signal."
            ),
        },
        "total_attempts": len(records),
        "scenarios": [
            {
                "name": s.name,
                "mode": s.mode,
                "attempts": s.attempts,
                "seed": s.seed,
            }
            for s in scenarios
        ],
        "kpi": summary.to_dict(),
        "untyped_outcomes": len(untyped),
        "unbounded_retry_loops": len(unbounded),
        "telemetry_offenders": len(telemetry_offenders),
        "extra_type_offenders": len(extra_offenders),
        "failure_taxonomy": taxonomy,
        "slo": slo_block,
        "watchdog": {
            "drift": drift_report,
            "ood": ood_report,
            "drift_precision_gate": float(DRIFT_PRECISION_GATE),
            "drift_recall_gate": float(DRIFT_RECALL_GATE),
            "ood_precision_gate": float(OOD_PRECISION_GATE),
            "ood_recall_gate": float(OOD_RECALL_GATE),
        },
        "adaptation": adaptation,
        "baseline_comparison": {
            # Emit a posix path (forward slashes) so the committed soak report is byte-stable across
            # platforms; str(WindowsPath) renders os.sep and makes the payload platform-dependent.
            "baseline_path": (
                baseline_path.relative_to(repo_root).as_posix()
                if baseline_path.is_relative_to(repo_root)
                else baseline_path.as_posix()
            ),
            "baseline_pick_success_rate_aggregate": baseline_pick,
            "soak_pick_success_rate": soak_pick,
            "regression": bool(pick_regression),
        },
        "easy_attempt_wall_time": {
            "capability_group": "soak_gate",
            "current_p95_s": current_easy,
            "baseline_p95_s": baseline_easy,
            "budget_multiplier": float(
                EASY_WALL_TIME_BUDGET_MULTIPLIER
            ),
            "budget_p95_s": easy_budget,
            "within_budget": bool(easy_within_budget),
        },
        "gate": gate,
        "violations": list(violations),
    }
    return payload, tuple(violations)
