"""``rl_active`` bounded canary router (additive, offline-callable).

Wraps a promoted RL policy artifact (the recovery family only) and
exposes the canary contract that the runtime can opt into:

* deterministic hash-based attempt sampling (stable across replays),
* bounded mode scope (dense by default),
* online KPI window (override-rate + regret-rate),
* auto-fallback with operator-only reset,
* typed ``CanaryDecision`` carrying every telemetry field.

Design constraints (all conservative):

* Recovery family only; construction rejects any other family.
* Default canary cap = 10% of eligible attempts.
* Default scope = dense; auto/easy get baseline regardless of cap.
* Sampling = ``int(sha1(attempt_id), 16) % 10000 < cap_x100``; fully
  deterministic, no RNG state, replay-stable.
* Fallback triggers (configurable, windowed): regret-rate over
  threshold or override-rate over threshold after a warmup of
  ``warmup_n`` canary attempts and over a sliding window of
  ``window_size`` recent attempts.
* Fallback persists until the operator calls ``reset_canary()``; no
  auto-recovery, so the rollback toggle stays one switch.
* Regret definition (no true counterfactual) = ``override == True AND
  outcome == "failed"``. Conservative upper-bound proxy; documented in
  :class:`CanaryStats`.
* No runtime callsite integration here: this is the infrastructure +
  the rollback drill, and the runtime stays untouched.
* Promotion-report verdict must be ``"pass"``; fail and abstain are
  hard rejected on construction. No ``force=True`` escape.

This module is stdlib-only, deterministic, and never touches
``robot.safety`` or any motor-control surface.
"""

from __future__ import annotations

import hashlib
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Mapping, Optional

from src.config.schema.robot.rl_schema import (
    RL_MODE_GEOMETRY_ONLY,
    RL_MODE_RL_ACTIVE,
    RL_MODE_RL_EXPERIMENTAL,
    RL_MODE_RL_SHADOW,
)

from src.robot.grasping.constants import (
    RL_CANARY_ROUTER_LOG_FILE,
    create_grasping_logger,
)

from .action_mask import ActionMaskContext
from .promotion import (
    POLICY_FAMILY_RECOVERY,
    PROMOTION_VERDICT_PASS,
    PromotionInputError,
    load_promotion_report,
)
from .recovery_policy import (
    DEFAULT_MAX_RECOVERY_ATTEMPTS,
    RECOVERY_ACTIONS_SET,
    LinUCBRecoveryPolicy,
    RecoveryRequest,
    apply_recovery_anti_loop_gate,
    load_linucb_recovery_policy,
)


#: The full ``rl_mode`` vocabulary, imported from the schema so the mode strings have one source
#: of truth (the schema enum) rather than a re-declared copy. ``CanaryDecision.rl_mode`` is always
#: ``RL_MODE_RL_ACTIVE``; this tuple is the mirror other consumers import.
RL_MODES: tuple[str, ...] = (
    RL_MODE_GEOMETRY_ONLY,
    RL_MODE_RL_SHADOW,
    RL_MODE_RL_ACTIVE,
    RL_MODE_RL_EXPERIMENTAL,
)

#: Default eligible scope = dense modes only. The easy and auto modes
#: get the baseline regardless of canary cap; easy stays conservative.
DEFAULT_ELIGIBLE_MODES: tuple[str, ...] = (
    "dense",
    "dense_clutter",
    "dense_recovery",
)

#: Fallback reason codes (typed, machine-parseable); every fallback
#: emits exactly one code.
FALLBACK_REASON_REGRET_RATE: str = "regret_rate_exceeded"
FALLBACK_REASON_OVERRIDE_RATE: str = "override_rate_exceeded"
FALLBACK_REASON_OPERATOR: str = "operator_engaged"
FALLBACK_REASON_POLICY_ERROR: str = "policy_propose_error"
FALLBACK_REASONS: tuple[str, ...] = (
    FALLBACK_REASON_REGRET_RATE,
    FALLBACK_REASON_OVERRIDE_RATE,
    FALLBACK_REASON_OPERATOR,
    FALLBACK_REASON_POLICY_ERROR,
)

#: Bucket cardinality for deterministic canary hashing. 10000 lets the
#: operator dial the cap in 0.01% increments.
_CANARY_HASH_BUCKETS: int = 10000


#: The canary is one of two RL surfaces allowed to change a recovery action on a real
#: cell; the deformable specialist in ``specialist_router`` is the other, held off by
#: its default ``SpecialistConfig.canary_pct = 0.0``. The kill switch has no automatic
#: recovery, so "when did it fall back and on what evidence" is an operator question,
#: not a telemetry one. The policy exception below is logged here with its traceback
#: and then dropped; this log is the only place it survives.
logger = create_grasping_logger("RLCanaryRouter", RL_CANARY_ROUTER_LOG_FILE)


class CanaryInputError(ValueError):
    """Raised when canary configuration / promotion contract is invalid."""


@dataclass(frozen=True)
class CanaryConfig:
    """Bounded-canary configuration; all defaults conservative. Fallback triggers stay off until
    ``warmup_n`` canary attempts are seen, which avoids spurious early rollbacks."""

    canary_pct: float = 10.0
    eligible_modes: tuple[str, ...] = DEFAULT_ELIGIBLE_MODES
    window_size: int = 50
    warmup_n: int = 20
    max_regret_rate: float = 0.30
    max_override_rate: float = 0.95
    #: Defence-in-depth attempt budget for the anti-loop gate: once ``attempt_index`` reaches this,
    #: any non-abort RL proposal is clipped to ``abort_recovery`` (mirrors the shadow path's gate).
    max_recovery_attempts: int = DEFAULT_MAX_RECOVERY_ATTEMPTS

    def __post_init__(self) -> None:
        if not (0.0 <= self.canary_pct <= 100.0):
            raise CanaryInputError(
                f"canary_pct must be in [0,100]; got {self.canary_pct!r}"
            )
        if self.window_size <= 0:
            raise CanaryInputError(
                f"window_size must be > 0; got {self.window_size!r}"
            )
        if self.warmup_n < 0:
            raise CanaryInputError(
                f"warmup_n must be >= 0; got {self.warmup_n!r}"
            )
        if not (0.0 <= self.max_regret_rate <= 1.0):
            raise CanaryInputError(
                "max_regret_rate must be in [0,1]; "
                f"got {self.max_regret_rate!r}"
            )
        if not (0.0 <= self.max_override_rate <= 1.0):
            raise CanaryInputError(
                "max_override_rate must be in [0,1]; "
                f"got {self.max_override_rate!r}"
            )
        if not self.eligible_modes:
            raise CanaryInputError("eligible_modes must be non-empty")
        if self.max_recovery_attempts <= 0:
            raise CanaryInputError(
                f"max_recovery_attempts must be > 0; got {self.max_recovery_attempts!r}"
            )


@dataclass(frozen=True)
class CanaryDecision:
    """One canary decision carrying every telemetry field.

    ``applied_action`` is what the runtime takes: the baseline action unless an RL override
    applies, and always the baseline once ``fallback_triggered`` is set."""

    attempt_id: str
    audit_id: str
    rl_mode: str
    rl_policy_id: str
    rl_artifact_version: int
    rl_router_path: str  # "base" | "specialist"
    scope_eligible: bool
    in_canary: bool
    baseline_action: str
    rl_action_proposed: Optional[str]
    rl_action_blocked_by_mask: bool
    rl_reason_features: tuple[tuple[str, float], ...]
    rl_confidence: float
    applied_action: str
    override: bool
    fallback_triggered: bool
    fallback_reason_code: Optional[str]
    #: Defence-in-depth anti-loop gate outcome (attempt-budget clip). ``anti_loop_clipped`` is True
    #: when the gate rewrites the applied action to ``abort_recovery``; ``anti_loop_clip_reason`` is
    #: one of ``"none"`` / ``"invalid_proposed_action"`` / ``"max_recovery_attempts_reached"``.
    anti_loop_clipped: bool = False
    anti_loop_clip_reason: str = "none"


@dataclass(frozen=True)
class CanaryStats:
    """Sliding-window stats over recent canary attempts; ``regret_rate`` is a conservative upper bound
    (an override that later failed; no true counterfactual is available offline)."""

    window_attempts: int
    override_count: int
    regret_count: int
    override_rate: float
    regret_rate: float
    warmed_up: bool


def _attempt_in_canary(attempt_id: str, canary_pct: float) -> bool:
    """Deterministic, replay-stable canary admission (``int(sha1(attempt_id),16) % 10000 < canary_pct*100``)."""

    if canary_pct <= 0.0:
        return False
    if canary_pct >= 100.0:
        return True
    h = hashlib.sha1(attempt_id.encode("utf-8")).digest()
    bucket = int.from_bytes(h[:4], "big") % _CANARY_HASH_BUCKETS
    threshold = int(round(canary_pct * 100.0))
    return bucket < threshold


@dataclass
class _WindowEntry:
    audit_id: str
    override: bool
    outcome: Optional[str] = None  # "succeeded" | "failed" | None


class ActiveCanaryRouter:
    """Bounded-canary active router for a promoted RL policy; additive (never enters the runtime by
    itself). Callers apply ``decision.applied_action`` and report outcomes via
    :meth:`record_outcome`."""

    def __init__(
        self,
        *,
        policy: LinUCBRecoveryPolicy,
        promotion_report_path: Path,
        config: Optional[CanaryConfig] = None,
        rl_router_path: str = "base",
    ) -> None:
        if config is None:
            config = CanaryConfig()
        self._config = config
        self._policy = policy
        self._rl_router_path = str(rl_router_path)
        report = load_promotion_report(promotion_report_path)
        if report.get("policy_family") != POLICY_FAMILY_RECOVERY:
            raise CanaryInputError(
                "the active-canary router only supports the LinUCB recovery family; "
                f"promotion report family={report.get('policy_family')!r}"
            )
        if report.get("policy_id") != policy.policy_id:
            raise CanaryInputError(
                "policy_id mismatch: promotion report claims "
                f"{report.get('policy_id')!r}, "
                f"policy artifact is {policy.policy_id!r}"
            )
        if report.get("verdict") != PROMOTION_VERDICT_PASS:
            raise CanaryInputError(
                "the active-canary router requires a promotion report with "
                f"verdict={PROMOTION_VERDICT_PASS!r}; "
                f"got {report.get('verdict')!r}"
            )
        self._promotion_report = report
        self._window: Deque[_WindowEntry] = deque(maxlen=config.window_size)
        self._audit_index: dict[str, _WindowEntry] = {}
        self._canary_attempts_seen: int = 0
        self._fallback_active: bool = False
        self._fallback_reason: Optional[str] = None
        logger.info(
            "Canary armed for policy %s from promotion report %s: cap %.2f%% of "
            "eligible attempts, scope %s, window %d, warmup %d, max regret %.3f, "
            "max override %.3f",
            policy.policy_id,
            promotion_report_path,
            config.canary_pct,
            ", ".join(sorted(config.eligible_modes)),
            config.window_size,
            config.warmup_n,
            config.max_regret_rate,
            config.max_override_rate,
        )

    # public read-only properties

    @property
    def config(self) -> CanaryConfig:
        return self._config

    @property
    def policy(self) -> LinUCBRecoveryPolicy:
        return self._policy

    @property
    def is_fallback_active(self) -> bool:
        return self._fallback_active

    @property
    def fallback_reason(self) -> Optional[str]:
        return self._fallback_reason

    @property
    def promotion_report(self) -> Mapping[str, Any]:
        return self._promotion_report

    @property
    def canary_attempts_seen(self) -> int:
        return self._canary_attempts_seen

    def stats(self) -> CanaryStats:
        n = len(self._window)
        override_count = sum(1 for e in self._window if e.override)
        regret_count = sum(
            1
            for e in self._window
            if e.override and e.outcome == "failed"
        )
        # Override rate is over the whole window. Regret rate is the overrides whose
        # attempt then failed, over the overrides observed in the window. With no
        # overrides in the window the regret rate is 0.
        override_rate = override_count / n if n > 0 else 0.0
        regret_rate = regret_count / override_count if override_count > 0 else 0.0
        warmed_up = self._canary_attempts_seen >= self._config.warmup_n
        return CanaryStats(
            window_attempts=n,
            override_count=override_count,
            regret_count=regret_count,
            override_rate=override_rate,
            regret_rate=regret_rate,
            warmed_up=warmed_up,
        )

    # main decision entry point

    def choose_recovery(
        self,
        *,
        request: RecoveryRequest,
        baseline_action: str,
        mode: str,
        attempt_index: int = 0,
        mask_context: ActionMaskContext | None = None,
    ) -> CanaryDecision:
        """Return the canary decision for one recovery dispatch; ``baseline_action`` is the deterministic
        action the caller would take without RL, which the canary never replaces as source of truth.

        Two fail-closed gates sit in front of the applied action (the RL policy can never elevate a
        proposal past them): the six-channel action mask (via ``mask_context``; a degraded scene forbids
        any RL override) and the defence-in-depth anti-loop gate (``attempt_index`` vs the config's
        ``max_recovery_attempts``). Both default to no-ops (``attempt_index=0``, ``mask_context=None``)."""

        if baseline_action not in RECOVERY_ACTIONS_SET:
            raise CanaryInputError(
                f"baseline_action={baseline_action!r} not a known "
                "recovery action"
            )
        audit_id = uuid.uuid4().hex
        scope_eligible = mode in self._config.eligible_modes
        in_canary = scope_eligible and _attempt_in_canary(
            request.attempt_id, self._config.canary_pct
        )

        # Default to baseline. Mutate only when RL is reached and a
        # valid action is proposed.
        rl_proposed: Optional[str] = None
        rl_blocked: bool = False
        rl_confidence: float = 0.0
        rl_features: tuple[tuple[str, float], ...] = ()
        anti_loop_clipped: bool = False
        anti_loop_clip_reason: str = "none"

        applied = baseline_action
        override = False
        fallback_triggered = self._fallback_active
        fallback_reason: Optional[str] = (
            self._fallback_reason if self._fallback_active else None
        )

        if in_canary and not self._fallback_active:
            try:
                sel = self._policy.propose_recovery(request)
            except Exception as exc:  # noqa: BLE001 (defensive)
                logger.exception(
                    "Canary policy raised on attempt %s; falling back to the "
                    "deterministic baseline %r: %s",
                    request.attempt_id,
                    baseline_action,
                    exc,
                )
                self._engage_fallback_internal(FALLBACK_REASON_POLICY_ERROR)
                fallback_triggered = True
                fallback_reason = FALLBACK_REASON_POLICY_ERROR
                _ = exc
            else:
                rl_proposed = sel.action
                # Safety precedence: the mask and the anti-loop gate both sit in front of the applied
                # action, so the RL policy can never elevate a proposal past them:
                #
                # (1) Six-channel action mask (highest authority). Only the scene-level
                #     ``degraded_mode`` channel gates a recovery action; the other five channels mask
                #     grasp candidates (a different action space), so they do not apply at this seam.
                #     A degraded scene forbids any RL override; the deterministic baseline stands.
                # (2) Defence-in-depth anti-loop gate: once the recovery attempt budget is spent (or an
                #     invalid token slips through) the proposal is clipped to ``abort_recovery``.
                scene_blocked = mask_context is not None and mask_context.degraded_mode_active
                if scene_blocked:
                    rl_blocked = True  # applied stays baseline; no override.
                else:
                    gate = apply_recovery_anti_loop_gate(
                        proposed_action=sel.action,
                        attempt_index=attempt_index,
                        max_recovery_attempts=self._config.max_recovery_attempts,
                    )
                    anti_loop_clipped = gate.clipped
                    anti_loop_clip_reason = gate.clip_reason
                    applied = gate.action
                    override = applied != baseline_action
                # Surface the top-action expected reward as a compact confidence proxy; the
                # full per-action breakdown goes in ``rl_reason_features``.
                for s in sel.scores:
                    if s.action == sel.action:
                        rl_confidence = float(s.expected_reward)
                        break
                rl_features = tuple(
                    (s.action, float(s.expected_reward)) for s in sel.scores
                )

        # Record the canary attempt for windowed KPI tracking.
        if in_canary and not fallback_triggered:
            entry = _WindowEntry(audit_id=audit_id, override=override)
            self._window_record(entry)
            self._canary_attempts_seen += 1

        # Re-evaluate auto-fallback after recording: a regret-rate
        # spike must take effect on the very next call.
        if not self._fallback_active:
            self._maybe_engage_auto_fallback()

        return CanaryDecision(
            attempt_id=request.attempt_id,
            audit_id=audit_id,
            rl_mode=RL_MODE_RL_ACTIVE,
            rl_policy_id=self._policy.policy_id,
            rl_artifact_version=int(self._policy.version),
            rl_router_path=self._rl_router_path,
            scope_eligible=scope_eligible,
            in_canary=in_canary,
            baseline_action=baseline_action,
            rl_action_proposed=rl_proposed,
            rl_action_blocked_by_mask=rl_blocked,
            rl_reason_features=rl_features,
            rl_confidence=rl_confidence,
            applied_action=applied,
            override=override,
            fallback_triggered=fallback_triggered,
            fallback_reason_code=fallback_reason,
            anti_loop_clipped=anti_loop_clipped,
            anti_loop_clip_reason=anti_loop_clip_reason,
        )

    # outcome reporting

    def record_outcome(self, audit_id: str, outcome: str) -> None:
        """Attach an outcome (``"succeeded"``/``"failed"``) to a previously-issued canary decision; an
        unknown ``audit_id`` is a silent no-op so callers can blindly report every attempt."""

        if outcome not in ("succeeded", "failed"):
            raise CanaryInputError(
                f"outcome must be 'succeeded' or 'failed'; got {outcome!r}"
            )
        entry = self._audit_index.get(audit_id)
        if entry is None:
            return
        entry.outcome = outcome
        if not self._fallback_active:
            self._maybe_engage_auto_fallback()

    # explicit fallback / reset

    def engage_fallback(
        self, reason_code: str = FALLBACK_REASON_OPERATOR
    ) -> None:
        """Operator-initiated kill switch."""

        if reason_code not in FALLBACK_REASONS:
            raise CanaryInputError(
                f"unknown fallback reason_code={reason_code!r}"
            )
        self._engage_fallback_internal(reason_code)

    def reset_canary(self) -> None:
        """Clear fallback state and window: the operator-only one switch that brings the canary back
        online. No automatic recovery is ever attempted."""

        logger.info(
            "Canary reset by operator (was %s, reason %s); window and counters cleared",
            "in fallback" if self._fallback_active else "live",
            self._fallback_reason,
        )
        self._fallback_active = False
        self._fallback_reason = None
        self._window.clear()
        self._audit_index.clear()
        self._canary_attempts_seen = 0

    # internals

    def _window_record(self, entry: _WindowEntry) -> None:
        if len(self._window) == self._window.maxlen:
            evicted = self._window[0]
            self._audit_index.pop(evicted.audit_id, None)
        self._window.append(entry)
        self._audit_index[entry.audit_id] = entry

    def _engage_fallback_internal(self, reason_code: str) -> None:
        if self._fallback_active:
            return
        # Every fallback path (operator, auto-KPI, policy error) funnels through
        # here, so the evidence is recorded exactly once and always with the
        # window stats that justify it.
        stats = self.stats()
        logger.warning(
            "Canary fallback engaged (%s) after %d canary attempt(s): override rate "
            "%.3f (max %.3f), regret rate %.3f (max %.3f) over a %d-attempt window. "
            "Stays off until an operator calls reset_canary().",
            reason_code,
            self._canary_attempts_seen,
            stats.override_rate,
            self._config.max_override_rate,
            stats.regret_rate,
            self._config.max_regret_rate,
            stats.window_attempts,
        )
        self._fallback_active = True
        self._fallback_reason = reason_code

    def _maybe_engage_auto_fallback(self) -> None:
        s = self.stats()
        if not s.warmed_up:
            return
        if s.override_count == 0 and s.window_attempts == 0:
            return
        if s.regret_rate > self._config.max_regret_rate:
            self._engage_fallback_internal(FALLBACK_REASON_REGRET_RATE)
            return
        if s.override_rate > self._config.max_override_rate:
            self._engage_fallback_internal(FALLBACK_REASON_OVERRIDE_RATE)
            return


def load_active_canary_router(
    *,
    policy_artifact_path: Path,
    promotion_report_path: Path,
    config: Optional[CanaryConfig] = None,
    rl_router_path: str = "base",
) -> ActiveCanaryRouter:
    """Load the recovery policy + its promotion report into a router."""

    try:
        policy = load_linucb_recovery_policy(policy_artifact_path)
    except (OSError, ValueError) as exc:
        raise CanaryInputError(
            f"failed to load LinUCB recovery policy at {policy_artifact_path}: "
            f"{exc}"
        ) from exc
    try:
        return ActiveCanaryRouter(
            policy=policy,
            promotion_report_path=promotion_report_path,
            config=config,
            rl_router_path=rl_router_path,
        )
    except PromotionInputError as exc:
        raise CanaryInputError(str(exc)) from exc


__all__ = (
    "RL_MODE_GEOMETRY_ONLY",
    "RL_MODE_RL_SHADOW",
    "RL_MODE_RL_ACTIVE",
    "RL_MODE_RL_EXPERIMENTAL",
    "RL_MODES",
    "DEFAULT_ELIGIBLE_MODES",
    "FALLBACK_REASON_REGRET_RATE",
    "FALLBACK_REASON_OVERRIDE_RATE",
    "FALLBACK_REASON_OPERATOR",
    "FALLBACK_REASON_POLICY_ERROR",
    "FALLBACK_REASONS",
    "CanaryInputError",
    "CanaryConfig",
    "CanaryDecision",
    "CanaryStats",
    "ActiveCanaryRouter",
    "load_active_canary_router",
)
