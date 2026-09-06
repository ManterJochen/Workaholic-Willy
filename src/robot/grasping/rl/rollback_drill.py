"""Rollback drill harness.

Exercises the :class:`ActiveCanaryRouter` and
:class:`DeformableSpecialistRouter` kill switches end-to-end and
emits a typed :class:`RollbackDrillReport`.

Five drill kinds are run, the first four per router and the last
across both:

1. ``operator_engage_and_reset``
       Operator triggers ``engage_fallback``; subsequent decisions
       must return baseline with ``fallback_triggered=True``;
       ``reset_canary`` brings the router back online.
2. ``regret_rate_auto_fallback``
       Inject canary traffic that always overrides and always fails,
       so the windowed regret estimator exceeds ``max_regret_rate``
       and fires ``regret_rate_exceeded`` automatically.
3. ``override_rate_auto_fallback``
       Inject canary traffic that always overrides + always succeeds.
       Override rate climbs past ``max_override_rate`` and fires
       ``override_rate_exceeded``.
4. ``policy_propose_error``
       Wrap the loaded policy with a stub that raises on
       ``propose_recovery``. The router catches and engages
       ``policy_propose_error``.
5. ``post_fallback_baseline_parity``
       After any fallback, every subsequent decision applies
       the baseline action verbatim, with no override and no RL
       proposal, checked via :func:`verify_baseline_parity` from
       :mod:`paired_soak`.

The drills are deterministic and pure-stdlib, and read only the
policy artifact and promotion report they are given.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.robot.grasping.constants import (
    RL_ROLLBACK_DRILL_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.rl.honesty import build_rollback_drill_honesty
from src.robot.grasping.rl.canary_router import (
    ActiveCanaryRouter,
    CanaryConfig,
    CanaryDecision,
    FALLBACK_REASON_OPERATOR,
    FALLBACK_REASON_OVERRIDE_RATE,
    FALLBACK_REASON_POLICY_ERROR,
    FALLBACK_REASON_REGRET_RATE,
    load_active_canary_router,
)
from src.robot.grasping.rl.recovery_policy import (
    DEFAULT_FALLBACK_TABLE,
    RECOVERY_ACTION_REOBSERVE,
    LinUCBRecoveryPolicy,
    RecoveryActionScore,
    RecoveryRequest,
    RecoverySelection,
    RecoveryStateKey,
    RECOVERY_ACTIONS,
    load_linucb_recovery_policy,
)
from src.robot.grasping.rl.sequencing_policy import (
    FAILURE_CLASS_DEFORMABLE,
    FAILURE_CLASS_EMPTY_AIR,
    FAILURE_CLASS_SLIP,
)
from src.robot.grasping.rl.specialist_router import (
    DeformableSpecialistRouter,
    SpecialistConfig,
    SpecialistDecision,
    SPECIALIST_FALLBACK_OPERATOR,
    SPECIALIST_FALLBACK_OVERRIDE_RATE,
    SPECIALIST_FALLBACK_POLICY_ERROR,
    SPECIALIST_FALLBACK_REGRET_RATE,
    load_deformable_specialist_router,
)
from src.robot.grasping.rl.paired_soak import (
    SoakRecord,
    PARITY_PASS,
    verify_baseline_parity,
)


# ---------------------------------------------------------------------------
# Schema constants.
# ---------------------------------------------------------------------------

#: Report carries reward_model / interpretation / policy_provenance honesty stamps.
ROLLBACK_DRILL_SCHEMA_VERSION: int = 2

DRILL_OPERATOR_ENGAGE_AND_RESET: str = "operator_engage_and_reset"
DRILL_REGRET_AUTO_FALLBACK: str = "regret_rate_auto_fallback"
DRILL_OVERRIDE_AUTO_FALLBACK: str = "override_rate_auto_fallback"
DRILL_POLICY_PROPOSE_ERROR: str = "policy_propose_error"
DRILL_POST_FALLBACK_PARITY: str = "post_fallback_baseline_parity"

DRILL_KINDS: tuple[str, ...] = (
    DRILL_OPERATOR_ENGAGE_AND_RESET,
    DRILL_REGRET_AUTO_FALLBACK,
    DRILL_OVERRIDE_AUTO_FALLBACK,
    DRILL_POLICY_PROPOSE_ERROR,
    DRILL_POST_FALLBACK_PARITY,
)

DRILL_PASS: str = "pass"
DRILL_FAIL: str = "fail"

ROUTER_BASE: str = "active_canary"
ROUTER_SPECIALIST: str = "deformable_specialist"


# ---------------------------------------------------------------------------
# Drill outcome dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DrillOutcome:
    router: str
    drill: str
    verdict: str
    expected_reason_code: Optional[str]
    observed_reason_code: Optional[str]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "router": self.router,
            "drill": self.drill,
            "verdict": self.verdict,
            "expected_reason_code": self.expected_reason_code,
            "observed_reason_code": self.observed_reason_code,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class RollbackDrillReport:
    schema_version: int
    outcomes: tuple[DrillOutcome, ...]
    overall_verdict: str
    policy_artifact_path: str = ""
    promotion_report_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "report_kind": "rollback_drill_report",
            "overall_verdict": self.overall_verdict,
            "outcomes": [o.to_dict() for o in self.outcomes],
            # This report proves kill-switch/fallback mechanics, not policy quality or real lift.
            **build_rollback_drill_honesty(
                policy_artifact_path=self.policy_artifact_path,
                promotion_report_path=self.promotion_report_path,
            ),
        }


# ---------------------------------------------------------------------------
# Stubs for the policy-error drill.
# ---------------------------------------------------------------------------


class _PolicyError(RuntimeError):
    """Sentinel raised by the wrapper stub."""


class _AlwaysRaisingPolicyWrapper:
    """Duck-typed policy mimic that raises on ``propose_recovery``.

    Exposes ``policy_id`` and ``version`` from the underlying real
    policy so the router's promotion-report validation passes.
    """

    def __init__(self, real: LinUCBRecoveryPolicy) -> None:
        self._real = real

    @property
    def policy_id(self) -> str:
        return self._real.policy_id

    @property
    def version(self) -> int:
        return int(self._real.version)

    def propose_recovery(
        self, request: RecoveryRequest
    ) -> RecoverySelection:  # noqa: D401
        del request
        raise _PolicyError("synthetic policy-propose failure (rollback drill)")


class _AlwaysOverridePolicyWrapper:
    """Duck-typed policy that always proposes an action other than
    the baseline so :class:`CanaryDecision.override` is True
    for every canary attempt.

    Carries the real policy's ``policy_id`` / ``version`` so router
    validation succeeds. ``propose_recovery`` returns a valid
    :class:`RecoverySelection` with a non-baseline top action.
    """

    def __init__(
        self, real: LinUCBRecoveryPolicy, override_action: str = "reobserve"
    ) -> None:
        self._real = real
        if override_action not in DEFAULT_FALLBACK_TABLE.values() and (
            override_action not in {a for a in DEFAULT_FALLBACK_TABLE.values()}
        ):
            pass  # accept any RECOVERY_ACTIONS member
        self._override = override_action

    @property
    def policy_id(self) -> str:
        return self._real.policy_id

    @property
    def version(self) -> int:
        return int(self._real.version)

    def propose_recovery(
        self, request: RecoveryRequest
    ) -> RecoverySelection:
        # Pick an action that differs from the baseline for the
        # request's failure class to guarantee ``override=True``.
        baseline = DEFAULT_FALLBACK_TABLE[request.state_key.failure_class_bucket]
        action = self._override if self._override != baseline else next(
            a for a in RECOVERY_ACTIONS if a != baseline
        )
        # Build a complete per-action ranking + scores so
        # RecoverySelection validates.
        ranked = (action,) + tuple(a for a in RECOVERY_ACTIONS if a != action)
        scores = tuple(
            RecoveryActionScore(
                action=a,
                expected_reward=(1.0 if a == action else 0.0),
                ucb=(1.0 if a == action else 0.0),
                min_support_over_active=10,
            )
            for a in RECOVERY_ACTIONS
        )
        return RecoverySelection(
            policy_id=self._real.policy_id,
            policy_version=int(self._real.version),
            action=action,
            ranked_actions=ranked,
            scores=scores,
            used_fallback=False,
        )


# ---------------------------------------------------------------------------
# Helpers for synthetic traffic.
# ---------------------------------------------------------------------------


def _base_request(idx: int, *, failure_class: str = FAILURE_CLASS_EMPTY_AIR) -> RecoveryRequest:
    return RecoveryRequest(
        attempt_id=f"rollback-drill-base-{idx:06d}",
        state_key=RecoveryStateKey(
            failure_class_bucket=failure_class,
            attempt_index_bucket="0",
            reobserve_count_bucket="0",
            dense_bucket="non_dense",
            last_outcome_bucket="failed",
        ),
    )


def _specialist_request(idx: int) -> RecoveryRequest:
    return RecoveryRequest(
        attempt_id=f"rollback-drill-spec-{idx:06d}",
        state_key=RecoveryStateKey(
            failure_class_bucket=FAILURE_CLASS_DEFORMABLE,
            attempt_index_bucket="0",
            reobserve_count_bucket="0",
            dense_bucket="dense",
            last_outcome_bucket="failed",
        ),
    )


def _decision_to_record(
    decision: CanaryDecision | SpecialistDecision,
    *,
    mode: str,
    failure_class: str,
    final_outcome: str = "succeeded",
) -> SoakRecord:
    return SoakRecord(
        attempt_id=decision.attempt_id,
        mode=mode,
        failure_class=failure_class,
        baseline_action=decision.baseline_action,
        applied_action=decision.applied_action,
        rl_action_proposed=decision.rl_action_proposed,
        override=bool(decision.override),
        fallback_triggered=bool(decision.fallback_triggered),
        fallback_reason_code=decision.fallback_reason_code,
        rl_router_path=decision.rl_router_path,
        rl_mode=decision.rl_mode,
        final_outcome=final_outcome,
        cycle_time_s=2.5,
        recovery_count=1,
    )


# ---------------------------------------------------------------------------
# Per-drill primitives for ActiveCanaryRouter.
# ---------------------------------------------------------------------------


def _drill_canary_operator_engage_and_reset(
    *, policy_path: Path, promotion_path: Path
) -> DrillOutcome:
    router = load_active_canary_router(
        policy_artifact_path=policy_path,
        promotion_report_path=promotion_path,
        config=CanaryConfig(canary_pct=50.0, warmup_n=0),
    )
    router.engage_fallback(FALLBACK_REASON_OPERATOR)
    if not router.is_fallback_active:
        return DrillOutcome(
            router=ROUTER_BASE,
            drill=DRILL_OPERATOR_ENGAGE_AND_RESET,
            verdict=DRILL_FAIL,
            expected_reason_code=FALLBACK_REASON_OPERATOR,
            observed_reason_code=router.fallback_reason,
            detail="engage_fallback did not flip is_fallback_active",
        )
    if router.fallback_reason != FALLBACK_REASON_OPERATOR:
        return DrillOutcome(
            router=ROUTER_BASE,
            drill=DRILL_OPERATOR_ENGAGE_AND_RESET,
            verdict=DRILL_FAIL,
            expected_reason_code=FALLBACK_REASON_OPERATOR,
            observed_reason_code=router.fallback_reason,
            detail="fallback_reason mismatch after engage_fallback",
        )
    # Every subsequent decision must apply baseline + fallback_triggered.
    for i in range(5):
        d = router.choose_recovery(
            request=_base_request(i),
            baseline_action="reobserve",
            mode="dense_clutter",
        )
        if d.applied_action != d.baseline_action or d.override or not d.fallback_triggered:
            return DrillOutcome(
                router=ROUTER_BASE,
                drill=DRILL_OPERATOR_ENGAGE_AND_RESET,
                verdict=DRILL_FAIL,
                expected_reason_code=FALLBACK_REASON_OPERATOR,
                observed_reason_code=d.fallback_reason_code,
                detail=f"post-engage decision {i} did not honour baseline parity",
            )
    # Reset must clear fallback.
    router.reset_canary()
    if router.is_fallback_active or router.fallback_reason is not None:
        return DrillOutcome(
            router=ROUTER_BASE,
            drill=DRILL_OPERATOR_ENGAGE_AND_RESET,
            verdict=DRILL_FAIL,
            expected_reason_code=None,
            observed_reason_code=router.fallback_reason,
            detail="reset_canary did not clear fallback state",
        )
    return DrillOutcome(
        router=ROUTER_BASE,
        drill=DRILL_OPERATOR_ENGAGE_AND_RESET,
        verdict=DRILL_PASS,
        expected_reason_code=FALLBACK_REASON_OPERATOR,
        observed_reason_code=FALLBACK_REASON_OPERATOR,
        detail="engage + 5 baseline decisions + reset OK",
    )


def _drill_canary_regret_auto_fallback(
    *, policy_path: Path, promotion_path: Path
) -> DrillOutcome:
    real = load_linucb_recovery_policy(policy_path)
    overriding = _AlwaysOverridePolicyWrapper(real, override_action="reobserve")
    router = ActiveCanaryRouter(
        policy=overriding,  # type: ignore[arg-type]
        promotion_report_path=promotion_path,
        config=CanaryConfig(
            canary_pct=100.0, warmup_n=5, window_size=30, max_regret_rate=0.20
        ),
    )
    # Drive enough overriding+failing traffic to exceed regret cap.
    for i in range(40):
        d = router.choose_recovery(
            request=_base_request(i, failure_class=FAILURE_CLASS_SLIP),
            baseline_action="replan_grasp",  # baseline for slip
            mode="dense_clutter",
        )
        # All overrides are reported as failures.
        router.record_outcome(d.audit_id, "failed")
        if router.is_fallback_active:
            break
    if not router.is_fallback_active:
        return DrillOutcome(
            router=ROUTER_BASE,
            drill=DRILL_REGRET_AUTO_FALLBACK,
            verdict=DRILL_FAIL,
            expected_reason_code=FALLBACK_REASON_REGRET_RATE,
            observed_reason_code=None,
            detail="40 overriding+failing canary attempts did not trigger fallback",
        )
    if router.fallback_reason != FALLBACK_REASON_REGRET_RATE:
        return DrillOutcome(
            router=ROUTER_BASE,
            drill=DRILL_REGRET_AUTO_FALLBACK,
            verdict=DRILL_FAIL,
            expected_reason_code=FALLBACK_REASON_REGRET_RATE,
            observed_reason_code=router.fallback_reason,
            detail="auto-fallback fired with wrong reason_code",
        )
    return DrillOutcome(
        router=ROUTER_BASE,
        drill=DRILL_REGRET_AUTO_FALLBACK,
        verdict=DRILL_PASS,
        expected_reason_code=FALLBACK_REASON_REGRET_RATE,
        observed_reason_code=router.fallback_reason,
        detail="regret-rate cap engaged auto-fallback",
    )


def _drill_canary_override_auto_fallback(
    *, policy_path: Path, promotion_path: Path
) -> DrillOutcome:
    real = load_linucb_recovery_policy(policy_path)
    overriding = _AlwaysOverridePolicyWrapper(real, override_action="re_segment")
    router = ActiveCanaryRouter(
        policy=overriding,  # type: ignore[arg-type]
        promotion_report_path=promotion_path,
        config=CanaryConfig(
            canary_pct=100.0,
            warmup_n=5,
            window_size=30,
            max_regret_rate=1.0,  # disable regret trigger
            max_override_rate=0.50,
        ),
    )
    for i in range(40):
        d = router.choose_recovery(
            request=_base_request(i, failure_class=FAILURE_CLASS_SLIP),
            baseline_action="replan_grasp",
            mode="dense_clutter",
        )
        # Mark every override as succeeded so regret stays 0.
        router.record_outcome(d.audit_id, "succeeded")
        if router.is_fallback_active:
            break
    if not router.is_fallback_active:
        return DrillOutcome(
            router=ROUTER_BASE,
            drill=DRILL_OVERRIDE_AUTO_FALLBACK,
            verdict=DRILL_FAIL,
            expected_reason_code=FALLBACK_REASON_OVERRIDE_RATE,
            observed_reason_code=None,
            detail="40 overriding+succeeding canary attempts did not trigger fallback",
        )
    if router.fallback_reason != FALLBACK_REASON_OVERRIDE_RATE:
        return DrillOutcome(
            router=ROUTER_BASE,
            drill=DRILL_OVERRIDE_AUTO_FALLBACK,
            verdict=DRILL_FAIL,
            expected_reason_code=FALLBACK_REASON_OVERRIDE_RATE,
            observed_reason_code=router.fallback_reason,
            detail="auto-fallback fired with wrong reason_code",
        )
    return DrillOutcome(
        router=ROUTER_BASE,
        drill=DRILL_OVERRIDE_AUTO_FALLBACK,
        verdict=DRILL_PASS,
        expected_reason_code=FALLBACK_REASON_OVERRIDE_RATE,
        observed_reason_code=router.fallback_reason,
        detail="override-rate cap engaged auto-fallback",
    )


def _drill_canary_policy_propose_error(
    *, policy_path: Path, promotion_path: Path
) -> DrillOutcome:
    real = load_linucb_recovery_policy(policy_path)
    bad = _AlwaysRaisingPolicyWrapper(real)
    router = ActiveCanaryRouter(
        policy=bad,  # type: ignore[arg-type]
        promotion_report_path=promotion_path,
        config=CanaryConfig(canary_pct=100.0, warmup_n=0, window_size=20),
    )
    d = router.choose_recovery(
        request=_base_request(0),
        baseline_action=RECOVERY_ACTION_REOBSERVE,
        mode="dense_clutter",
    )
    if not router.is_fallback_active:
        return DrillOutcome(
            router=ROUTER_BASE,
            drill=DRILL_POLICY_PROPOSE_ERROR,
            verdict=DRILL_FAIL,
            expected_reason_code=FALLBACK_REASON_POLICY_ERROR,
            observed_reason_code=None,
            detail="policy raise did not engage fallback",
        )
    if (
        router.fallback_reason != FALLBACK_REASON_POLICY_ERROR
        or d.fallback_reason_code != FALLBACK_REASON_POLICY_ERROR
    ):
        return DrillOutcome(
            router=ROUTER_BASE,
            drill=DRILL_POLICY_PROPOSE_ERROR,
            verdict=DRILL_FAIL,
            expected_reason_code=FALLBACK_REASON_POLICY_ERROR,
            observed_reason_code=router.fallback_reason,
            detail="policy-error fallback fired with wrong reason_code",
        )
    return DrillOutcome(
        router=ROUTER_BASE,
        drill=DRILL_POLICY_PROPOSE_ERROR,
        verdict=DRILL_PASS,
        expected_reason_code=FALLBACK_REASON_POLICY_ERROR,
        observed_reason_code=router.fallback_reason,
        detail="policy.propose_recovery raised -> fallback engaged",
    )


# ---------------------------------------------------------------------------
# Per-drill primitives for DeformableSpecialistRouter.
# ---------------------------------------------------------------------------


def _drill_specialist_operator_engage_and_reset(
    *, policy_path: Path, promotion_path: Path
) -> DrillOutcome:
    router = load_deformable_specialist_router(
        policy_artifact_path=policy_path,
        promotion_report_path=promotion_path,
        config=SpecialistConfig(canary_pct=100.0, warmup_n=0),
    )
    router.engage_fallback(SPECIALIST_FALLBACK_OPERATOR)
    if (
        not router.is_fallback_active
        or router.fallback_reason != SPECIALIST_FALLBACK_OPERATOR
    ):
        return DrillOutcome(
            router=ROUTER_SPECIALIST,
            drill=DRILL_OPERATOR_ENGAGE_AND_RESET,
            verdict=DRILL_FAIL,
            expected_reason_code=SPECIALIST_FALLBACK_OPERATOR,
            observed_reason_code=router.fallback_reason,
            detail="engage_fallback did not flip is_fallback_active",
        )
    for i in range(5):
        d = router.choose_recovery(
            request=_specialist_request(i),
            baseline_action="perturb_and_retry",  # baseline for deformable
        )
        if d.applied_action != d.baseline_action or d.override or not d.fallback_triggered:
            return DrillOutcome(
                router=ROUTER_SPECIALIST,
                drill=DRILL_OPERATOR_ENGAGE_AND_RESET,
                verdict=DRILL_FAIL,
                expected_reason_code=SPECIALIST_FALLBACK_OPERATOR,
                observed_reason_code=d.fallback_reason_code,
                detail=f"post-engage decision {i} broke baseline parity",
            )
    router.reset_canary()
    if router.is_fallback_active or router.fallback_reason is not None:
        return DrillOutcome(
            router=ROUTER_SPECIALIST,
            drill=DRILL_OPERATOR_ENGAGE_AND_RESET,
            verdict=DRILL_FAIL,
            expected_reason_code=None,
            observed_reason_code=router.fallback_reason,
            detail="reset_canary did not clear fallback state",
        )
    return DrillOutcome(
        router=ROUTER_SPECIALIST,
        drill=DRILL_OPERATOR_ENGAGE_AND_RESET,
        verdict=DRILL_PASS,
        expected_reason_code=SPECIALIST_FALLBACK_OPERATOR,
        observed_reason_code=SPECIALIST_FALLBACK_OPERATOR,
        detail="engage + 5 baseline decisions + reset OK",
    )


def _drill_specialist_regret_auto_fallback(
    *, policy_path: Path, promotion_path: Path
) -> DrillOutcome:
    real = load_linucb_recovery_policy(policy_path)
    # `re_segment` is non-blocked on dense+deformable and differs
    # from the baseline `perturb_and_retry`, so it guarantees override.
    overriding = _AlwaysOverridePolicyWrapper(real, override_action="re_segment")
    router = DeformableSpecialistRouter(
        policy=overriding,  # type: ignore[arg-type]
        promotion_report_path=promotion_path,
        config=SpecialistConfig(
            canary_pct=100.0,
            warmup_n=5,
            window_size=30,
            max_regret_rate=0.20,
        ),
    )
    for i in range(40):
        d = router.choose_recovery(
            request=_specialist_request(i),
            baseline_action="perturb_and_retry",
        )
        router.record_outcome(d.audit_id, "failed")
        if router.is_fallback_active:
            break
    if (
        not router.is_fallback_active
        or router.fallback_reason != SPECIALIST_FALLBACK_REGRET_RATE
    ):
        return DrillOutcome(
            router=ROUTER_SPECIALIST,
            drill=DRILL_REGRET_AUTO_FALLBACK,
            verdict=DRILL_FAIL,
            expected_reason_code=SPECIALIST_FALLBACK_REGRET_RATE,
            observed_reason_code=router.fallback_reason,
            detail="specialist regret-rate cap did not engage fallback",
        )
    return DrillOutcome(
        router=ROUTER_SPECIALIST,
        drill=DRILL_REGRET_AUTO_FALLBACK,
        verdict=DRILL_PASS,
        expected_reason_code=SPECIALIST_FALLBACK_REGRET_RATE,
        observed_reason_code=router.fallback_reason,
        detail="regret-rate cap engaged specialist fallback",
    )


def _drill_specialist_override_auto_fallback(
    *, policy_path: Path, promotion_path: Path
) -> DrillOutcome:
    real = load_linucb_recovery_policy(policy_path)
    overriding = _AlwaysOverridePolicyWrapper(real, override_action="re_segment")
    router = DeformableSpecialistRouter(
        policy=overriding,  # type: ignore[arg-type]
        promotion_report_path=promotion_path,
        config=SpecialistConfig(
            canary_pct=100.0,
            warmup_n=5,
            window_size=30,
            max_regret_rate=1.0,
            max_override_rate=0.50,
        ),
    )
    for i in range(40):
        d = router.choose_recovery(
            request=_specialist_request(i),
            baseline_action="perturb_and_retry",
        )
        router.record_outcome(d.audit_id, "succeeded")
        if router.is_fallback_active:
            break
    if (
        not router.is_fallback_active
        or router.fallback_reason != SPECIALIST_FALLBACK_OVERRIDE_RATE
    ):
        return DrillOutcome(
            router=ROUTER_SPECIALIST,
            drill=DRILL_OVERRIDE_AUTO_FALLBACK,
            verdict=DRILL_FAIL,
            expected_reason_code=SPECIALIST_FALLBACK_OVERRIDE_RATE,
            observed_reason_code=router.fallback_reason,
            detail="specialist override-rate cap did not engage fallback",
        )
    return DrillOutcome(
        router=ROUTER_SPECIALIST,
        drill=DRILL_OVERRIDE_AUTO_FALLBACK,
        verdict=DRILL_PASS,
        expected_reason_code=SPECIALIST_FALLBACK_OVERRIDE_RATE,
        observed_reason_code=router.fallback_reason,
        detail="override-rate cap engaged specialist fallback",
    )


def _drill_specialist_policy_propose_error(
    *, policy_path: Path, promotion_path: Path
) -> DrillOutcome:
    real = load_linucb_recovery_policy(policy_path)
    bad = _AlwaysRaisingPolicyWrapper(real)
    router = DeformableSpecialistRouter(
        policy=bad,  # type: ignore[arg-type]
        promotion_report_path=promotion_path,
        config=SpecialistConfig(canary_pct=100.0, warmup_n=0, window_size=20),
    )
    d = router.choose_recovery(
        request=_specialist_request(0),
        baseline_action="perturb_and_retry",
    )
    if (
        not router.is_fallback_active
        or router.fallback_reason != SPECIALIST_FALLBACK_POLICY_ERROR
        or d.fallback_reason_code != SPECIALIST_FALLBACK_POLICY_ERROR
    ):
        return DrillOutcome(
            router=ROUTER_SPECIALIST,
            drill=DRILL_POLICY_PROPOSE_ERROR,
            verdict=DRILL_FAIL,
            expected_reason_code=SPECIALIST_FALLBACK_POLICY_ERROR,
            observed_reason_code=router.fallback_reason,
            detail="specialist did not engage policy-error fallback",
        )
    return DrillOutcome(
        router=ROUTER_SPECIALIST,
        drill=DRILL_POLICY_PROPOSE_ERROR,
        verdict=DRILL_PASS,
        expected_reason_code=SPECIALIST_FALLBACK_POLICY_ERROR,
        observed_reason_code=router.fallback_reason,
        detail="policy.propose_recovery raised -> specialist fallback engaged",
    )


# ---------------------------------------------------------------------------
# Post-fallback parity drill (cross-router).
# ---------------------------------------------------------------------------


def _drill_post_fallback_parity(
    *, policy_path: Path, promotion_path: Path
) -> DrillOutcome:
    """Engage both operator fallbacks, run 20 mixed requests through
    each, and verify :func:`verify_baseline_parity` passes on every
    record (applied == baseline, no override, no proposal).
    """

    base = load_active_canary_router(
        policy_artifact_path=policy_path,
        promotion_report_path=promotion_path,
        config=CanaryConfig(canary_pct=50.0, warmup_n=0),
    )
    spec = load_deformable_specialist_router(
        policy_artifact_path=policy_path,
        promotion_report_path=promotion_path,
        config=SpecialistConfig(canary_pct=100.0, warmup_n=0),
    )
    base.engage_fallback(FALLBACK_REASON_OPERATOR)
    spec.engage_fallback(SPECIALIST_FALLBACK_OPERATOR)

    records: list[SoakRecord] = []
    d: CanaryDecision | SpecialistDecision
    for i in range(20):
        d = base.choose_recovery(
            request=_base_request(i),
            baseline_action="re_segment",  # baseline for empty_air
            mode="dense_clutter",
        )
        records.append(
            _decision_to_record(
                d, mode="dense_clutter", failure_class=FAILURE_CLASS_EMPTY_AIR
            )
        )
    for i in range(20):
        d = spec.choose_recovery(
            request=_specialist_request(i),
            baseline_action="perturb_and_retry",
        )
        records.append(
            _decision_to_record(
                d, mode="dense_clutter", failure_class=FAILURE_CLASS_DEFORMABLE
            )
        )

    parity = verify_baseline_parity(records)
    if parity.verdict != PARITY_PASS:
        return DrillOutcome(
            router="cross_router",
            drill=DRILL_POST_FALLBACK_PARITY,
            verdict=DRILL_FAIL,
            expected_reason_code=None,
            observed_reason_code=None,
            detail=(
                f"verify_baseline_parity FAIL: "
                f"{len(parity.violations)} violations"
            ),
        )
    return DrillOutcome(
        router="cross_router",
        drill=DRILL_POST_FALLBACK_PARITY,
        verdict=DRILL_PASS,
        expected_reason_code=None,
        observed_reason_code=None,
        detail="40 post-fallback decisions all baseline-equivalent",
    )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


#: This is the rehearsal of the kill switch. A failed drill means the switch that
#: is supposed to stop an RL policy on a real cell did not stop it, so each
#: failing drill is named, not just counted in a verdict.
logger = create_grasping_logger("RLRollbackDrill", RL_ROLLBACK_DRILL_LOG_FILE)


def run_rollback_drills(
    *,
    policy_artifact_path: Path | str,
    promotion_report_path: Path | str,
    output_path: Optional[Path | str] = None,
) -> RollbackDrillReport:
    """Run all router + cross-router drills and return a typed report.

    If ``output_path`` is provided, the report is also written as
    deterministic JSON.
    """

    pp = Path(policy_artifact_path)
    rp = Path(promotion_report_path)

    outcomes: list[DrillOutcome] = [
        _drill_canary_operator_engage_and_reset(policy_path=pp, promotion_path=rp),
        _drill_canary_regret_auto_fallback(policy_path=pp, promotion_path=rp),
        _drill_canary_override_auto_fallback(policy_path=pp, promotion_path=rp),
        _drill_canary_policy_propose_error(policy_path=pp, promotion_path=rp),
        _drill_specialist_operator_engage_and_reset(policy_path=pp, promotion_path=rp),
        _drill_specialist_regret_auto_fallback(policy_path=pp, promotion_path=rp),
        _drill_specialist_override_auto_fallback(policy_path=pp, promotion_path=rp),
        _drill_specialist_policy_propose_error(policy_path=pp, promotion_path=rp),
        _drill_post_fallback_parity(policy_path=pp, promotion_path=rp),
    ]

    overall = (
        DRILL_PASS
        if all(o.verdict == DRILL_PASS for o in outcomes)
        else DRILL_FAIL
    )
    report = RollbackDrillReport(
        schema_version=ROLLBACK_DRILL_SCHEMA_VERSION,
        outcomes=tuple(outcomes),
        overall_verdict=overall,
        policy_artifact_path=str(pp),
        promotion_report_path=str(rp),
    )
    failed = [o for o in outcomes if o.verdict != DRILL_PASS]
    if failed:
        logger.error(
            "Rollback drill FAILED %d/%d: %s (policy %s)",
            len(failed),
            len(outcomes),
            "; ".join(f"{o.router}/{o.drill}" for o in failed),
            pp,
        )
    else:
        logger.info(
            "Rollback drill passed %d/%d for policy %s", len(outcomes), len(outcomes), pp
        )
    if output_path is not None:
        op = Path(output_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
        op.write_text(body, encoding="utf-8")
        logger.info(
            "Wrote rollback-drill report to %s (%d bytes)",
            op,
            len(body.encode("utf-8")),
        )
    return report


__all__ = (
    "ROLLBACK_DRILL_SCHEMA_VERSION",
    "DRILL_OPERATOR_ENGAGE_AND_RESET",
    "DRILL_REGRET_AUTO_FALLBACK",
    "DRILL_OVERRIDE_AUTO_FALLBACK",
    "DRILL_POLICY_PROPOSE_ERROR",
    "DRILL_POST_FALLBACK_PARITY",
    "DRILL_KINDS",
    "DRILL_PASS",
    "DRILL_FAIL",
    "ROUTER_BASE",
    "ROUTER_SPECIALIST",
    "DrillOutcome",
    "RollbackDrillReport",
    "run_rollback_drills",
)
