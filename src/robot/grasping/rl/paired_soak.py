"""Paired RL-on / RL-off soak comparator.

Scope
-----
The operational hardening layer on top of the canary + specialist
routers: a deterministic synthetic soak that drives the LinUCB policy
(via :class:`ActiveCanaryRouter` and :class:`DeformableSpecialistRouter`)
against a deterministic baseline and produces a paired KPI report.

The harness is additive: it never imports from the runtime callsites
(autonomous_grasp, recovery_orchestrator, pick_loop). The routers are
constructed standalone, fed synthetic recovery requests, and their
decisions are translated into :class:`GraspAttemptRecord` rows so the
existing :func:`compute_kpis` aggregator can score both arms with the
locked KPI semantics.

Locks
-----
* hybrid: own generator + reuse :mod:`replay.kpi`.
* paired JSONs + comparison file written by :func:`run_paired_soak`.
* soft verdict: the parity verdict is honest about the baseline's
  actual delta on synthetic traffic.
* runnable :func:`verify_baseline_parity`.
* additive only: no edits to runtime/safety/replay schema.

The module is pure-stdlib (no numpy, no torch). All artefacts are
deterministic given ``(seed, attempts_per_arm)`` and the policy +
promotion report.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord
from src.robot.grasping.replay.kpi import KpiSummary, compute_kpis
from src.robot.grasping.rl.honesty import build_paired_soak_honesty
from src.robot.grasping.rl.canary_router import (
    ActiveCanaryRouter,
    CanaryConfig,
    CanaryDecision,
    RL_MODE_GEOMETRY_ONLY,
    RL_MODE_RL_ACTIVE,
    RL_MODE_RL_SHADOW,
    load_active_canary_router,
)
from src.robot.grasping.rl.recovery_policy import (
    DEFAULT_FALLBACK_TABLE,
    RECOVERY_ACTIONS_SET,
    RecoveryRequest,
    RecoveryStateKey,
)
from src.robot.grasping.rl.sequencing_policy import (
    FAILURE_CLASS_COLLISION,
    FAILURE_CLASS_DEFORMABLE,
    FAILURE_CLASS_EMPTY_AIR,
    FAILURE_CLASS_OCCLUSION,
    FAILURE_CLASS_SLIP,
)
from src.robot.grasping.rl.specialist_router import (
    DeformableSpecialistRouter,
    SpecialistConfig,
    SpecialistDecision,
    load_deformable_specialist_router,
)

from src.robot.grasping.constants import (
    RL_PAIRED_SOAK_LOG_FILE,
    create_grasping_logger,
)

#: The parity check is the safety half of this gate: on a fallback or with RL off,
#: the applied action must equal the baseline. A violation is returned as a string
#: list, not raised, and would otherwise be visible only to whoever opens the JSON.
logger = create_grasping_logger("RLPairedSoak", RL_PAIRED_SOAK_LOG_FILE)


# ---------------------------------------------------------------------------
# Schema constants.
# ---------------------------------------------------------------------------

#: Emitted artifacts carry reward_model / dataset_provenance /
#: gate_verdict_interpretation honesty stamps.
PAIRED_SOAK_SCHEMA_VERSION: int = 3
PAIRED_SOAK_MIN_ATTEMPTS_PER_ARM: int = 200

# Honesty stamp: the paired-soak lift is measured against a hardcoded synthetic outcome model
# (``_synthetic_outcome`` / ``_synthetic_cycle_time`` below), not real-hardware or high-fidelity-sim
# outcomes. Every emitted artifact carries this so a reader cannot mistake the deltas for real lift.
SOAK_REWARD_MODEL: str = "synthetic_hardcoded"
SOAK_REWARD_INTERPRETATION: str = (
    "mechanics-only: deltas characterise router/rollback MECHANICS under a deterministic synthetic "
    "outcome model, NOT real-hardware lift. Real lift needs captured replay data."
)
PAIRED_SOAK_DEFAULT_ATTEMPTS_PER_ARM: int = 600

PARITY_PASS: str = "pass"
PARITY_WARN: str = "warn"
PARITY_FAIL: str = "fail"
PARITY_VERDICTS: tuple[str, ...] = (
    PARITY_PASS,
    PARITY_WARN,
    PARITY_FAIL,
)


# Soft thresholds (soft verdict).
#:    - pass requires >=1 of {cycle >= 5% better, retries >= 10% reduced}
#:      and no pick-success regression beyond 2pp and dead_loop
#:      growth <= 0.5pp.
#:    - fail if pick-success regresses by > 2pp or dead_loop grows
#:      by > 0.5pp or safety_rejection grows by > 0.1pp.
#:    - otherwise warn.
PASS_CYCLE_IMPROVEMENT_FRAC: float = 0.05
PASS_RETRY_REDUCTION_FRAC: float = 0.10
FAIL_PICK_SUCCESS_REGRESSION_PP: float = 0.02
FAIL_DEAD_LOOP_GROWTH_PP: float = 0.005
FAIL_SAFETY_GROWTH_PP: float = 0.001


# Scenario mix: covers every state-key cell the routers care about.
_FAILURE_CLASS_MIX: tuple[str, ...] = (
    FAILURE_CLASS_EMPTY_AIR,
    FAILURE_CLASS_OCCLUSION,
    FAILURE_CLASS_SLIP,
    FAILURE_CLASS_COLLISION,
    FAILURE_CLASS_DEFORMABLE,  # specialist-eligible when dense
)
_MODE_MIX: tuple[str, ...] = (
    "dense_clutter",
    "dense_autonomous",
    "easy_pick",
    "auto_pick",
)


# ---------------------------------------------------------------------------
# Determinism helpers.
# ---------------------------------------------------------------------------


def _hash_u32(*parts: str) -> int:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _det_uniform(*parts: str) -> float:
    return _hash_u32(*parts) / float(0xFFFFFFFF)


# ---------------------------------------------------------------------------
# Record + report dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SoakRecord:
    """One soak record. Both arms share this schema."""

    attempt_id: str
    mode: str
    failure_class: str
    baseline_action: str
    applied_action: str
    rl_action_proposed: Optional[str]
    override: bool
    fallback_triggered: bool
    fallback_reason_code: Optional[str]
    rl_router_path: str  # "baseline" | "base" | "specialist_deformable"
    rl_mode: str
    final_outcome: str  # "succeeded" | "failed" | "recovery_exhausted"
    cycle_time_s: float
    recovery_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "mode": self.mode,
            "failure_class": self.failure_class,
            "baseline_action": self.baseline_action,
            "applied_action": self.applied_action,
            "rl_action_proposed": self.rl_action_proposed,
            "override": bool(self.override),
            "fallback_triggered": bool(self.fallback_triggered),
            "fallback_reason_code": self.fallback_reason_code,
            "rl_router_path": self.rl_router_path,
            "rl_mode": self.rl_mode,
            "final_outcome": self.final_outcome,
            "cycle_time_s": float(self.cycle_time_s),
            "recovery_count": int(self.recovery_count),
        }


@dataclass(frozen=True, slots=True)
class SoakArmReport:
    """KPI snapshot for one arm of the paired soak."""

    arm: str  # "rl_off" | "rl_on"
    rl_mode: str
    attempts: int
    kpis: KpiSummary
    override_rate: float
    fallback_rate: float
    median_cycle_time_s: float
    retry_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "rl_mode": self.rl_mode,
            "attempts": int(self.attempts),
            "kpis": self.kpis.to_dict(),
            "override_rate": float(self.override_rate),
            "fallback_rate": float(self.fallback_rate),
            "median_cycle_time_s": float(self.median_cycle_time_s),
            "retry_rate": float(self.retry_rate),
        }


@dataclass(frozen=True, slots=True)
class ParityResult:
    """Output of :func:`verify_baseline_parity`."""

    verdict: str  # PARITY_PASS | PARITY_FAIL
    violations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "violations": list(self.violations),
        }


@dataclass(frozen=True, slots=True)
class SoakComparisonReport:
    """Paired-soak delta report. ``verdict`` is soft."""

    schema_version: int
    seed: int
    attempts_per_arm: int
    rl_off: SoakArmReport
    rl_on: SoakArmReport
    delta_pick_success_pp: float
    delta_first_attempt_pp: float
    delta_dead_loop_pp: float
    delta_safety_rejection_pp: float
    delta_cycle_time_frac: float
    delta_retry_rate_frac: float
    rl_override_rate: float
    rl_fallback_rate: float
    baseline_parity: ParityResult
    verdict: str
    verdict_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "report_kind": "paired_soak_comparison",
            "seed": int(self.seed),
            "attempts_per_arm": int(self.attempts_per_arm),
            "rl_off": self.rl_off.to_dict(),
            "rl_on": self.rl_on.to_dict(),
            "deltas": {
                "pick_success_pp": float(self.delta_pick_success_pp),
                "first_attempt_pp": float(self.delta_first_attempt_pp),
                "dead_loop_pp": float(self.delta_dead_loop_pp),
                "safety_rejection_pp": float(self.delta_safety_rejection_pp),
                "cycle_time_frac": float(self.delta_cycle_time_frac),
                "retry_rate_frac": float(self.delta_retry_rate_frac),
            },
            "rl_router_stats": {
                "override_rate": float(self.rl_override_rate),
                "fallback_rate": float(self.rl_fallback_rate),
            },
            "baseline_parity": self.baseline_parity.to_dict(),
            "verdict": self.verdict,
            "verdict_reasons": list(self.verdict_reasons),
            # The deltas above are not real lift: they come from a hardcoded synthetic reward model.
            "reward_model": SOAK_REWARD_MODEL,
            "interpretation": SOAK_REWARD_INTERPRETATION,
            # Adds dataset_provenance + gate_verdict_interpretation.
            **build_paired_soak_honesty(),
        }


# ---------------------------------------------------------------------------
# Synthetic outcome model (deterministic, seed-driven).
# ---------------------------------------------------------------------------


def _baseline_action_for(failure_class: str) -> str:
    """Deterministic baseline = fallback table per failure class."""

    return DEFAULT_FALLBACK_TABLE[failure_class]


def _is_specialist_blocked(failure_class: str, action: str, dense: bool) -> bool:
    """Mirror the specialist mask: ``perturb_and_retry`` is blocked on
    deformable+dense (the only specialist-eligible cell).
    """

    return (
        dense
        and failure_class == FAILURE_CLASS_DEFORMABLE
        and action == "perturb_and_retry"
    )


def _synthetic_outcome(
    *,
    attempt_id: str,
    failure_class: str,
    action: str,
    dense: bool,
    seed: int,
) -> str:
    """Deterministic synthetic outcome model.

    Encodes a prior about which actions match which failure classes.
    The LinUCB may or may not recover this prior given the small soak
    window; the soft verdict keeps things honest.

    Reward shape:
      * baseline action (fallback table) gets a 0.60 base success rate.
      * a hand-coded "ideal" action gets 0.75.
      * any other valid action gets 0.40.
      * blocked-on-specialist actions never appear (router masked).
      * a draw above 0.97 terminates as ``recovery_exhausted`` (dead
        loop) when the action is neither the ideal nor the baseline.
    """

    ideal = {
        FAILURE_CLASS_EMPTY_AIR: "re_segment",
        FAILURE_CLASS_OCCLUSION: "reobserve",
        FAILURE_CLASS_SLIP: "replan_grasp",
        FAILURE_CLASS_COLLISION: "replan_grasp",
        # On dense+deformable, perturb_and_retry is masked, so the
        # next-best is replan_grasp.
        FAILURE_CLASS_DEFORMABLE: (
            "replan_grasp" if dense else "perturb_and_retry"
        ),
    }.get(failure_class, "reobserve")
    baseline = _baseline_action_for(failure_class)

    if action == ideal:
        p_success = 0.75
    elif action == baseline:
        p_success = 0.60
    else:
        p_success = 0.40

    r = _det_uniform("outcome", str(seed), attempt_id, action, failure_class)
    if r < p_success:
        return "succeeded"
    # Dead-loop tail for the worst action only.
    if action != ideal and action != baseline and r > 0.97:
        return "recovery_exhausted"
    return "failed"


def _synthetic_cycle_time(
    *, attempt_id: str, action: str, seed: int
) -> float:
    """Deterministic cycle time in seconds.

    The baseline is 2.0 seconds, plus jitter and a per-action cost.
    Recoveries that re-segment or perturb cost more, matching the
    cycle-time KPI.
    """

    base = 2.0
    cost = {
        "reobserve": 0.4,
        "re_segment": 0.7,
        "replan_grasp": 0.5,
        "perturb_and_retry": 0.9,
        "abort_recovery": 0.1,
    }.get(action, 0.5)
    jitter = _det_uniform("cycle", str(seed), attempt_id) * 0.3
    return base + cost + jitter


# ---------------------------------------------------------------------------
# Scenario generation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SoakScenarioSpec:
    """Deterministic soak scenario.

    ``seed`` keys every synthetic draw. The same seed and
    ``attempts_per_arm`` give byte-identical record lists.
    """

    attempts_per_arm: int = PAIRED_SOAK_DEFAULT_ATTEMPTS_PER_ARM
    seed: int = 20251015

    def __post_init__(self) -> None:
        if self.attempts_per_arm < PAIRED_SOAK_MIN_ATTEMPTS_PER_ARM:
            raise ValueError(
                f"attempts_per_arm must be >= "
                f"{PAIRED_SOAK_MIN_ATTEMPTS_PER_ARM}; "
                f"got {self.attempts_per_arm!r}"
            )
        if self.attempts_per_arm > 100_000:
            raise ValueError(
                "attempts_per_arm absurdly large; soak is synthetic, "
                f"got {self.attempts_per_arm!r}"
            )


def _build_state_key(
    *, failure_class: str, mode: str, attempt_idx: int
) -> RecoveryStateKey:
    dense = "dense" if mode.startswith("dense") else "non_dense"
    return RecoveryStateKey(
        failure_class_bucket=failure_class,
        attempt_index_bucket="0" if attempt_idx == 0 else "1",
        reobserve_count_bucket="0",
        dense_bucket=dense,
        last_outcome_bucket="failed",
    )


def _iter_synthetic_attempts(
    spec: SoakScenarioSpec,
) -> list[tuple[str, str, str, RecoveryStateKey]]:
    """Return a deterministic stream of
    ``(attempt_id, mode, failure_class, state_key)`` tuples,
    cycling deterministically through (mode x failure_class)."""

    out: list[tuple[str, str, str, RecoveryStateKey]] = []
    for i in range(spec.attempts_per_arm):
        mode = _MODE_MIX[i % len(_MODE_MIX)]
        fc = _FAILURE_CLASS_MIX[(i // len(_MODE_MIX)) % len(_FAILURE_CLASS_MIX)]
        attempt_id = f"paired-{spec.seed:08x}-{i:06d}"
        sk = _build_state_key(
            failure_class=fc, mode=mode, attempt_idx=i % 2
        )
        out.append((attempt_id, mode, fc, sk))
    return out


# ---------------------------------------------------------------------------
# Arm generators.
# ---------------------------------------------------------------------------


def generate_rl_off_arm(spec: SoakScenarioSpec) -> tuple[SoakRecord, ...]:
    """Generate the deterministic baseline (geometry_only) arm.

    No router is constructed. Every action is the fallback-table
    action for the failure class. ``override`` is always False;
    ``rl_action_proposed`` is None.
    """

    out: list[SoakRecord] = []
    for attempt_id, mode, fc, _sk in _iter_synthetic_attempts(spec):
        baseline = _baseline_action_for(fc)
        dense = mode.startswith("dense")
        outcome = _synthetic_outcome(
            attempt_id=attempt_id,
            failure_class=fc,
            action=baseline,
            dense=dense,
            seed=spec.seed,
        )
        ct = _synthetic_cycle_time(
            attempt_id=attempt_id, action=baseline, seed=spec.seed
        )
        out.append(
            SoakRecord(
                attempt_id=attempt_id,
                mode=mode,
                failure_class=fc,
                baseline_action=baseline,
                applied_action=baseline,
                rl_action_proposed=None,
                override=False,
                fallback_triggered=False,
                fallback_reason_code=None,
                rl_router_path="baseline",
                rl_mode=RL_MODE_GEOMETRY_ONLY,
                final_outcome=outcome,
                cycle_time_s=ct,
                recovery_count=1,
            )
        )
    return tuple(out)


def generate_rl_on_arm(
    spec: SoakScenarioSpec,
    *,
    base_router: ActiveCanaryRouter,
    specialist_router: DeformableSpecialistRouter,
) -> tuple[SoakRecord, ...]:
    """Generate the RL-on arm and apply the router decisions.

    Each request goes to the specialist router when it is eligible and
    to the canary router otherwise. The outcome is reported back to the
    router that decided, so its windowed KPI estimator advances and
    auto-fallback can fire if regret spikes.
    """

    out: list[SoakRecord] = []
    for attempt_id, mode, fc, sk in _iter_synthetic_attempts(spec):
        baseline = _baseline_action_for(fc)
        dense = mode.startswith("dense")
        request = RecoveryRequest(attempt_id=attempt_id, state_key=sk)

        decision: SpecialistDecision | CanaryDecision
        router_for_outcome: DeformableSpecialistRouter | ActiveCanaryRouter
        if specialist_router.is_routed(sk):
            decision = specialist_router.choose_recovery(
                request=request, baseline_action=baseline
            )
            router_for_outcome = specialist_router
            router_path = decision.rl_router_path
            rl_mode = decision.rl_mode
            applied = decision.applied_action
            # Specialist mask should keep `perturb_and_retry` out
            # for deformable+dense; verify defensively.
            if _is_specialist_blocked(fc, applied, dense):
                # Mask escape: clamp to the baseline so the soak record
                # stays consistent with the specialist's promise.
                applied = baseline
        else:
            decision = base_router.choose_recovery(
                request=request, baseline_action=baseline, mode=mode
            )
            router_for_outcome = base_router
            router_path = decision.rl_router_path
            rl_mode = decision.rl_mode
            applied = decision.applied_action

        if applied not in RECOVERY_ACTIONS_SET:
            raise RuntimeError(  # pragma: no cover (invariant guard)
                f"router returned unknown action {applied!r}"
            )

        outcome = _synthetic_outcome(
            attempt_id=attempt_id,
            failure_class=fc,
            action=applied,
            dense=dense,
            seed=spec.seed,
        )
        ct = _synthetic_cycle_time(
            attempt_id=attempt_id, action=applied, seed=spec.seed
        )

        # Report the canary attempt's outcome to the router;
        # recovery_exhausted counts as "failed" for the regret estimator.
        outcome_for_router = "succeeded" if outcome == "succeeded" else "failed"
        router_for_outcome.record_outcome(decision.audit_id, outcome_for_router)

        out.append(
            SoakRecord(
                attempt_id=attempt_id,
                mode=mode,
                failure_class=fc,
                baseline_action=baseline,
                applied_action=applied,
                rl_action_proposed=decision.rl_action_proposed,
                override=bool(decision.override),
                fallback_triggered=bool(decision.fallback_triggered),
                fallback_reason_code=decision.fallback_reason_code,
                rl_router_path=router_path,
                rl_mode=(
                    RL_MODE_RL_ACTIVE
                    if not decision.fallback_triggered
                    else RL_MODE_RL_SHADOW
                ) if rl_mode == RL_MODE_RL_ACTIVE else rl_mode,
                final_outcome=outcome,
                cycle_time_s=ct,
                recovery_count=1,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# GraspAttemptRecord adapter (reuses compute_kpis).
# ---------------------------------------------------------------------------


def _to_grasp_attempt_records(
    records: Sequence[SoakRecord],
) -> tuple[GraspAttemptRecord, ...]:
    out: list[GraspAttemptRecord] = []
    for r in records:
        recovery_actions = (
            ({"action": r.applied_action},) if r.recovery_count > 0 else ()
        )
        extra: dict[str, Any] = {
            "cycle_time_s": float(r.cycle_time_s),
            "rl_router_path": r.rl_router_path,
            "rl_mode": r.rl_mode,
            "rl_action_proposed": r.rl_action_proposed,
            "rl_action_applied": r.applied_action,
            "baseline_action": r.baseline_action,
            "override": bool(r.override),
            "fallback_triggered": bool(r.fallback_triggered),
            "fallback_reason_code": r.fallback_reason_code,
        }
        out.append(
            GraspAttemptRecord(
                timestamp=1_700_000_000.0,  # frozen for determinism
                attempt_id=r.attempt_id,
                mode=r.mode,
                final_outcome=r.final_outcome,
                recovery_actions=recovery_actions,
                extra=extra,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Arm report builder.
# ---------------------------------------------------------------------------


def _retry_rate(records: Sequence[SoakRecord]) -> float:
    if not records:
        return 0.0
    retried = sum(1 for r in records if r.recovery_count > 0)
    return retried / len(records)


def build_arm_report(
    *, arm: str, rl_mode: str, records: Sequence[SoakRecord]
) -> SoakArmReport:
    if arm not in ("rl_off", "rl_on"):
        raise ValueError(f"arm must be 'rl_off' or 'rl_on'; got {arm!r}")
    grasp_records = _to_grasp_attempt_records(records)
    kpis = compute_kpis(grasp_records)
    n = max(len(records), 1)
    override_count = sum(1 for r in records if r.override)
    fallback_count = sum(1 for r in records if r.fallback_triggered)
    cycle_median = (
        kpis.median_cycle_time_s
        if kpis.median_cycle_time_s is not None
        else 0.0
    )
    return SoakArmReport(
        arm=arm,
        rl_mode=rl_mode,
        attempts=len(records),
        kpis=kpis,
        override_rate=override_count / n,
        fallback_rate=fallback_count / n,
        median_cycle_time_s=float(cycle_median),
        retry_rate=_retry_rate(records),
    )


# ---------------------------------------------------------------------------
# Baseline parity verifier.
# ---------------------------------------------------------------------------


def verify_baseline_parity(
    records: Sequence[SoakRecord],
) -> ParityResult:
    """Assert that every record which should have taken the
    baseline path actually did.

    A record is parity-bound when:

    * ``rl_mode == "geometry_only"`` (RL disabled at this attempt), or
    * ``fallback_triggered is True`` (router rolled back to baseline).

    The requirement for those records is:

    * ``applied_action == baseline_action``,
    * ``override is False``,
    * ``rl_action_proposed`` is None, checked only on the
      ``fallback_triggered`` records; the router skips
      ``policy.propose`` once it has rolled back, so a record that is
      merely ``geometry_only`` may carry a proposal and has to satisfy
      only the first two.

    Returns :class:`ParityResult` with pass or fail +
    violation strings keyed by ``attempt_id``.
    """

    violations: list[str] = []
    for r in records:
        bound = (
            r.rl_mode == RL_MODE_GEOMETRY_ONLY or r.fallback_triggered
        )
        if not bound:
            continue
        if r.applied_action != r.baseline_action:
            violations.append(
                f"{r.attempt_id}: applied={r.applied_action!r} "
                f"!= baseline={r.baseline_action!r}"
            )
        if r.override:
            violations.append(f"{r.attempt_id}: override is True")
        if r.rl_action_proposed is not None and r.fallback_triggered:
            # Proposed is allowed in non-fallback rl_shadow rows;
            # parity only requires applied==baseline there. Under
            # fallback rl_action_proposed must be None, because the
            # router skips policy.propose entirely.
            violations.append(
                f"{r.attempt_id}: rl_action_proposed set during fallback"
            )
    if violations:
        logger.error(
            "Baseline parity FAILED on %d of %d record(s); RL changed an action it "
            "was not allowed to: %s",
            len(violations),
            len(records),
            "; ".join(violations[:10])
            + (" ..." if len(violations) > 10 else ""),
        )
        return ParityResult(
            verdict=PARITY_FAIL, violations=tuple(violations)
        )
    logger.info("Baseline parity holds over %d record(s)", len(records))
    return ParityResult(verdict=PARITY_PASS, violations=())


# ---------------------------------------------------------------------------
# Verdict scoring.
# ---------------------------------------------------------------------------


def _score_verdict(
    *,
    rl_off: SoakArmReport,
    rl_on: SoakArmReport,
    parity: ParityResult,
) -> tuple[str, tuple[str, ...], dict[str, float]]:
    reasons: list[str] = []

    pick_pp = rl_on.kpis.pick_success_rate - rl_off.kpis.pick_success_rate
    first_pp = (
        rl_on.kpis.first_attempt_success_rate
        - rl_off.kpis.first_attempt_success_rate
    )
    dead_pp = rl_on.kpis.dead_loop_rate - rl_off.kpis.dead_loop_rate
    safety_pp = (
        rl_on.kpis.safety_rejection_rate
        - rl_off.kpis.safety_rejection_rate
    )

    cycle_frac: float
    if rl_off.median_cycle_time_s > 0.0:
        cycle_frac = (
            (rl_off.median_cycle_time_s - rl_on.median_cycle_time_s)
            / rl_off.median_cycle_time_s
        )
    else:
        cycle_frac = 0.0

    retry_frac: float
    if rl_off.retry_rate > 0.0:
        retry_frac = (
            (rl_off.retry_rate - rl_on.retry_rate) / rl_off.retry_rate
        )
    else:
        retry_frac = 0.0

    deltas = {
        "pick_pp": pick_pp,
        "first_pp": first_pp,
        "dead_pp": dead_pp,
        "safety_pp": safety_pp,
        "cycle_frac": cycle_frac,
        "retry_frac": retry_frac,
    }

    # Fail conditions.
    if pick_pp < -FAIL_PICK_SUCCESS_REGRESSION_PP:
        reasons.append(
            f"pick_success regressed by {-pick_pp:.4f} "
            f"(> {FAIL_PICK_SUCCESS_REGRESSION_PP})"
        )
    if dead_pp > FAIL_DEAD_LOOP_GROWTH_PP:
        reasons.append(
            f"dead_loop grew by {dead_pp:.4f} "
            f"(> {FAIL_DEAD_LOOP_GROWTH_PP})"
        )
    if safety_pp > FAIL_SAFETY_GROWTH_PP:
        reasons.append(
            f"safety_rejection grew by {safety_pp:.4f} "
            f"(> {FAIL_SAFETY_GROWTH_PP})"
        )
    if parity.verdict == PARITY_FAIL:
        reasons.append(
            f"baseline parity violations: {len(parity.violations)}"
        )

    if reasons:
        return PARITY_FAIL, tuple(reasons), deltas

    # Pass conditions.
    if (
        cycle_frac >= PASS_CYCLE_IMPROVEMENT_FRAC
        or retry_frac >= PASS_RETRY_REDUCTION_FRAC
    ):
        reasons.append(
            f"RL improves at least one of "
            f"cycle (frac={cycle_frac:.4f}) / "
            f"retry (frac={retry_frac:.4f})"
        )
        return PARITY_PASS, tuple(reasons), deltas

    # Otherwise warn.
    reasons.append(
        "no regression beyond fail thresholds, but neither pass "
        "threshold met (cycle/retry improvement insufficient)"
    )
    return PARITY_WARN, tuple(reasons), deltas


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def run_paired_soak(
    *,
    policy_artifact_path: Path | str,
    promotion_report_path: Path | str,
    spec: Optional[SoakScenarioSpec] = None,
    output_dir: Optional[Path | str] = None,
    base_canary_pct: float = 30.0,
    specialist_canary_pct: float = 100.0,
) -> SoakComparisonReport:
    """Run both arms and score the delta between them.

    Files written to ``output_dir`` when it is provided:

    * ``paired_soak_rl_off_v1.json``
    * ``paired_soak_rl_on_v1.json``
    * ``paired_soak_comparison_v1.json``

    All three are deterministic given the same ``spec`` and policy
    artefacts.
    """

    if spec is None:
        spec = SoakScenarioSpec()

    base_router = load_active_canary_router(
        policy_artifact_path=Path(policy_artifact_path),
        promotion_report_path=Path(promotion_report_path),
        config=CanaryConfig(canary_pct=float(base_canary_pct)),
        rl_router_path="base",
    )
    specialist_router = load_deformable_specialist_router(
        policy_artifact_path=Path(policy_artifact_path),
        promotion_report_path=Path(promotion_report_path),
        config=SpecialistConfig(canary_pct=float(specialist_canary_pct)),
    )

    rl_off_records = generate_rl_off_arm(spec)
    rl_on_records = generate_rl_on_arm(
        spec, base_router=base_router, specialist_router=specialist_router
    )

    rl_off_report = build_arm_report(
        arm="rl_off", rl_mode=RL_MODE_GEOMETRY_ONLY, records=rl_off_records
    )
    rl_on_report = build_arm_report(
        arm="rl_on", rl_mode=RL_MODE_RL_ACTIVE, records=rl_on_records
    )

    parity_records = tuple(rl_off_records) + tuple(rl_on_records)
    parity = verify_baseline_parity(parity_records)

    verdict, reasons, deltas = _score_verdict(
        rl_off=rl_off_report, rl_on=rl_on_report, parity=parity
    )

    report = SoakComparisonReport(
        schema_version=PAIRED_SOAK_SCHEMA_VERSION,
        seed=spec.seed,
        attempts_per_arm=spec.attempts_per_arm,
        rl_off=rl_off_report,
        rl_on=rl_on_report,
        delta_pick_success_pp=deltas["pick_pp"],
        delta_first_attempt_pp=deltas["first_pp"],
        delta_dead_loop_pp=deltas["dead_pp"],
        delta_safety_rejection_pp=deltas["safety_pp"],
        delta_cycle_time_frac=deltas["cycle_frac"],
        delta_retry_rate_frac=deltas["retry_frac"],
        rl_override_rate=rl_on_report.override_rate,
        rl_fallback_rate=rl_on_report.fallback_rate,
        baseline_parity=parity,
        verdict=verdict,
        verdict_reasons=reasons,
    )

    log = logger.info if verdict == PARITY_PASS else logger.warning
    log(
        "Paired soak verdict %s over %d attempt(s) per arm (seed %d, base canary "
        "%.1f%%, specialist %.1f%%): pick %+.4fpp, first-attempt %+.4fpp, dead-loop "
        "%+.4fpp, cycle %+.3f, override rate %.3f, fallback rate %.3f%s",
        verdict,
        spec.attempts_per_arm,
        spec.seed,
        float(base_canary_pct),
        float(specialist_canary_pct),
        deltas["pick_pp"],
        deltas["first_pp"],
        deltas["dead_pp"],
        deltas["cycle_frac"],
        rl_on_report.override_rate,
        rl_on_report.fallback_rate,
        "" if not reasons else ": " + "; ".join(reasons),
    )
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_json(
            out / "paired_soak_rl_off_v1.json",
            {
                "schema_version": PAIRED_SOAK_SCHEMA_VERSION,
                "report_kind": "paired_soak_rl_off_arm",
                "reward_model": SOAK_REWARD_MODEL,
                "seed": spec.seed,
                "attempts_per_arm": spec.attempts_per_arm,
                "arm": rl_off_report.to_dict(),
                "records": [r.to_dict() for r in rl_off_records],
            },
        )
        _write_json(
            out / "paired_soak_rl_on_v1.json",
            {
                "schema_version": PAIRED_SOAK_SCHEMA_VERSION,
                "report_kind": "paired_soak_rl_on_arm",
                "reward_model": SOAK_REWARD_MODEL,
                "seed": spec.seed,
                "attempts_per_arm": spec.attempts_per_arm,
                "base_canary_pct": float(base_canary_pct),
                "specialist_canary_pct": float(specialist_canary_pct),
                "arm": rl_on_report.to_dict(),
                "records": [r.to_dict() for r in rl_on_records],
            },
        )
        _write_json(
            out / "paired_soak_comparison_v1.json",
            report.to_dict(),
        )

    return report


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s (%d bytes)", path, len(text.encode("utf-8")))


__all__ = (
    "PAIRED_SOAK_SCHEMA_VERSION",
    "SOAK_REWARD_MODEL",
    "SOAK_REWARD_INTERPRETATION",
    "PAIRED_SOAK_MIN_ATTEMPTS_PER_ARM",
    "PAIRED_SOAK_DEFAULT_ATTEMPTS_PER_ARM",
    "PARITY_PASS",
    "PARITY_WARN",
    "PARITY_FAIL",
    "PARITY_VERDICTS",
    "PASS_CYCLE_IMPROVEMENT_FRAC",
    "PASS_RETRY_REDUCTION_FRAC",
    "FAIL_PICK_SUCCESS_REGRESSION_PP",
    "FAIL_DEAD_LOOP_GROWTH_PP",
    "FAIL_SAFETY_GROWTH_PP",
    "SoakRecord",
    "SoakArmReport",
    "ParityResult",
    "SoakComparisonReport",
    "SoakScenarioSpec",
    "generate_rl_off_arm",
    "generate_rl_on_arm",
    "build_arm_report",
    "verify_baseline_parity",
    "run_paired_soak",
)
