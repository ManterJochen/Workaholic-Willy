"""Deformable specialist router + ``rl_experimental`` lane.

Sits alongside :class:`src.robot.grasping.rl.canary_router.ActiveCanaryRouter`
and is engaged only when the typed state signals indicate a
deformable failure on a dense scene. The specialist wraps the same
promotion-gated LinUCB recovery policy but applies a tighter action
mask (e.g. forbid ``perturb_and_retry`` on deformables, which risks
damaging fabric / wires / bags) and a stricter regret cap
(default 0.20 vs. base 0.30).

Design constraints (all conservative):

* trigger: routed iff
  ``state_key.failure_class_bucket == "deformable_misclassification"
  and state_key.dense_bucket == "dense"``.
* rollout: shadow + bounded canary; default ``canary_pct = 0.0``
  (pure shadow until operator opts in).
* ``rl_experimental``: constant + ``route_rl_experimental(...)``
  raise path only; no artifact, no fake hardware claims.
* artifact: reuse the promotion-gated LinUCB recovery policy; do
  not invent a new family. Specialist enforces an additional
  action mask (default: forbid ``perturb_and_retry``) and a stricter
  ``max_regret_rate = 0.20``.
* kill switch: fully independent from the base canary; reason
  codes are namespaced (``specialist_*``); ``engage_fallback`` /
  ``reset_canary`` here do not touch the base router and vice versa.

This module is stdlib-only, deterministic, additive, and never
imports any motor-control surface.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Mapping, Optional

from src.robot.grasping.constants import (
    RL_SPECIALIST_ROUTER_LOG_FILE,
    create_grasping_logger,
)

from .canary_router import (
    RL_MODE_RL_ACTIVE,
    RL_MODE_RL_EXPERIMENTAL,
    _attempt_in_canary,
)
from .promotion import (
    POLICY_FAMILY_RECOVERY,
    PROMOTION_VERDICT_PASS,
    PromotionInputError,
    load_promotion_report,
)
from .recovery_policy import (
    RECOVERY_ACTIONS_SET,
    LinUCBRecoveryPolicy,
    RecoveryRequest,
    RecoveryStateKey,
    load_linucb_recovery_policy,
)


# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------


#: Trigger state-key constants.
SPECIALIST_FAILURE_CLASS_DEFORMABLE: str = "deformable_misclassification"
SPECIALIST_DENSE_BUCKET: str = "dense"

#: Default tighter action mask for deformable scenes. ``perturb_and_retry``
#: physically perturbs the object; on a deformable that is a damage risk
#: and the post-perturbation state is unpredictable.
DEFAULT_SPECIALIST_BLOCKED_ACTIONS: tuple[str, ...] = ("perturb_and_retry",)

#: Namespaced fallback reason codes.
SPECIALIST_FALLBACK_REGRET_RATE: str = "specialist_regret_rate_exceeded"
SPECIALIST_FALLBACK_OVERRIDE_RATE: str = "specialist_override_rate_exceeded"
SPECIALIST_FALLBACK_OPERATOR: str = "specialist_operator_engaged"
SPECIALIST_FALLBACK_POLICY_ERROR: str = "specialist_policy_propose_error"
SPECIALIST_FALLBACK_MASK_BLOCKED_ALL: str = (
    "specialist_action_mask_left_no_options"
)
SPECIALIST_FALLBACK_REASONS: tuple[str, ...] = (
    SPECIALIST_FALLBACK_REGRET_RATE,
    SPECIALIST_FALLBACK_OVERRIDE_RATE,
    SPECIALIST_FALLBACK_OPERATOR,
    SPECIALIST_FALLBACK_POLICY_ERROR,
    SPECIALIST_FALLBACK_MASK_BLOCKED_ALL,
)

#: ``rl_experimental`` lane: typed reason for the NotImplementedError;
#: never ships an artifact.
EXPERIMENTAL_NOT_IMPLEMENTED_REASON: str = "rl_experimental_not_implemented"

#: Router-path tag emitted in every specialist decision.
SPECIALIST_ROUTER_PATH: str = "specialist_deformable"


# ---------------------------------------------------------------------------
# Exceptions.
# ---------------------------------------------------------------------------


#: Its own file, not the base canary's: this router has an independent kill switch
#: with namespaced reason codes, and reading the two fallback histories interleaved
#: is exactly how one gets mistaken for the other. The extra action mask it applies
#: (blocking e.g. perturb_and_retry on fabric) is also unique to this lane.
logger = create_grasping_logger("RLSpecialistRouter", RL_SPECIALIST_ROUTER_LOG_FILE)


class SpecialistInputError(ValueError):
    """Raised when specialist config / promotion contract is invalid."""


class ExperimentalLaneNotImplementedError(NotImplementedError):
    """Raised by :func:`route_rl_experimental`; no artifact ships.

    Carries the typed reason code so callers can branch on
    ``exc.reason_code`` rather than parsing the message.
    """

    def __init__(
        self,
        message: str = "rl_experimental lane has no policy artifact in the deformable specialist",
        *,
        reason_code: str = EXPERIMENTAL_NOT_IMPLEMENTED_REASON,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecialistConfig:
    """Specialist canary config. Defaults are strict:

    * ``canary_pct = 0.0``: pure shadow until operator opts in,
    * ``max_regret_rate = 0.20``: stricter than base 0.30,
    * blocked-action mask removes ``perturb_and_retry`` from the
      deformable action menu.
    """

    canary_pct: float = 0.0
    blocked_actions: tuple[str, ...] = DEFAULT_SPECIALIST_BLOCKED_ACTIONS
    window_size: int = 50
    warmup_n: int = 20
    max_regret_rate: float = 0.20
    max_override_rate: float = 0.95

    def __post_init__(self) -> None:
        if not (0.0 <= self.canary_pct <= 100.0):
            raise SpecialistInputError(
                f"canary_pct must be in [0,100]; got {self.canary_pct!r}"
            )
        if self.window_size <= 0:
            raise SpecialistInputError(
                f"window_size must be > 0; got {self.window_size!r}"
            )
        if self.warmup_n < 0:
            raise SpecialistInputError(
                f"warmup_n must be >= 0; got {self.warmup_n!r}"
            )
        if not (0.0 <= self.max_regret_rate <= 1.0):
            raise SpecialistInputError(
                "max_regret_rate must be in [0,1]; "
                f"got {self.max_regret_rate!r}"
            )
        if not (0.0 <= self.max_override_rate <= 1.0):
            raise SpecialistInputError(
                "max_override_rate must be in [0,1]; "
                f"got {self.max_override_rate!r}"
            )
        bad = [a for a in self.blocked_actions if a not in RECOVERY_ACTIONS_SET]
        if bad:
            raise SpecialistInputError(
                f"blocked_actions has unknown action(s): {bad!r}"
            )
        if set(self.blocked_actions) == RECOVERY_ACTIONS_SET:
            raise SpecialistInputError(
                "blocked_actions cannot mask out every recovery action"
            )


# ---------------------------------------------------------------------------
# Decision / Stats records.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecialistDecision:
    """One specialist decision. Mirrors :class:`CanaryDecision` plus
    two routing-specific fields:

    * ``routed_to_specialist``: did the typed state hit the trigger?
    * ``mask_filtered_action``: did the action mask change the
      applied action away from the policy's top proposal?
    """

    attempt_id: str
    audit_id: str
    rl_mode: str
    rl_policy_id: str
    rl_artifact_version: int
    rl_router_path: str  # SPECIALIST_ROUTER_PATH
    routed_to_specialist: bool
    in_canary: bool
    baseline_action: str
    rl_action_proposed: Optional[str]
    rl_action_blocked_by_mask: bool
    mask_filtered_action: bool
    rl_reason_features: tuple[tuple[str, float], ...]
    rl_confidence: float
    applied_action: str
    override: bool
    fallback_triggered: bool
    fallback_reason_code: Optional[str]


@dataclass(frozen=True)
class SpecialistStats:
    """Sliding-window stats over recent canary attempts on the specialist path.

    ``regret_rate`` is the same conservative upper bound used by :class:`CanaryStats`:
    ``override AND outcome == "failed"`` over ``override_count``.
    """

    window_attempts: int
    override_count: int
    regret_count: int
    override_rate: float
    regret_rate: float
    warmed_up: bool


# ---------------------------------------------------------------------------
# Routing predicate.
# ---------------------------------------------------------------------------


def is_specialist_eligible(state_key: RecoveryStateKey) -> bool:
    """``True`` iff this attempt should be routed to the deformable
    specialist (deformable failure on a dense scene)."""

    return (
        state_key.failure_class_bucket == SPECIALIST_FAILURE_CLASS_DEFORMABLE
        and state_key.dense_bucket == SPECIALIST_DENSE_BUCKET
    )


# ---------------------------------------------------------------------------
# Router internals.
# ---------------------------------------------------------------------------


@dataclass
class _WindowEntry:
    audit_id: str
    override: bool
    outcome: Optional[str] = None  # "succeeded" | "failed" | None


def _apply_action_mask(
    *,
    proposed: str,
    ranked_actions: tuple[str, ...],
    blocked_actions: frozenset[str],
) -> tuple[Optional[str], bool, bool]:
    """Return ``(masked_action_or_None, mask_filtered, mask_blocked_all)``.

    * If ``proposed`` is not blocked: returns ``(proposed, False, False)``.
    * If ``proposed`` is blocked: walks ``ranked_actions`` in order and
      returns the first non-blocked action with ``mask_filtered=True``.
    * If every action in ``ranked_actions`` is blocked (only possible
      with a mis-configured mask covering all recovery actions): returns
      ``(None, True, True)`` so the caller can engage fallback.
    """

    if proposed not in blocked_actions:
        return proposed, False, False
    for candidate in ranked_actions:
        if candidate not in blocked_actions:
            return candidate, True, False
    return None, True, True


# ---------------------------------------------------------------------------
# Router.
# ---------------------------------------------------------------------------


class DeformableSpecialistRouter:
    """Bounded-canary specialist router for deformable+dense recoveries.

    The router is additive: it never enters the runtime by itself.
    Runtime callers call :meth:`choose_recovery` and
    apply ``decision.applied_action`` directly. Outcomes are reported
    back via :meth:`record_outcome` keyed by ``decision.audit_id``.
    """

    def __init__(
        self,
        *,
        policy: LinUCBRecoveryPolicy,
        promotion_report_path: Path,
        config: Optional[SpecialistConfig] = None,
    ) -> None:
        if config is None:
            config = SpecialistConfig()
        self._config = config
        self._policy = policy
        self._blocked: frozenset[str] = frozenset(config.blocked_actions)

        try:
            report = load_promotion_report(promotion_report_path)
        except PromotionInputError as exc:
            raise SpecialistInputError(str(exc)) from exc
        if report.get("policy_family") != POLICY_FAMILY_RECOVERY:
            raise SpecialistInputError(
                "the deformable specialist only supports the LinUCB recovery family; "
                f"promotion report family={report.get('policy_family')!r}"
            )
        if report.get("policy_id") != policy.policy_id:
            raise SpecialistInputError(
                "policy_id mismatch: promotion report claims "
                f"{report.get('policy_id')!r}, "
                f"policy artifact is {policy.policy_id!r}"
            )
        if report.get("verdict") != PROMOTION_VERDICT_PASS:
            raise SpecialistInputError(
                "the deformable specialist requires a promotion report with "
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
            "Deformable specialist armed for policy %s from %s: cap %.2f%%, blocked "
            "action(s) %s, max regret %.3f, max override %.3f, window %d",
            policy.policy_id,
            promotion_report_path,
            config.canary_pct,
            ", ".join(sorted(self._blocked)) or "none",
            config.max_regret_rate,
            config.max_override_rate,
            config.window_size,
        )

    # --- read-only props ---------------------------------------------------

    @property
    def config(self) -> SpecialistConfig:
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

    def stats(self) -> SpecialistStats:
        n = len(self._window)
        override_count = sum(1 for e in self._window if e.override)
        regret_count = sum(
            1
            for e in self._window
            if e.override and e.outcome == "failed"
        )
        override_rate = override_count / n if n > 0 else 0.0
        regret_rate = (
            regret_count / override_count if override_count > 0 else 0.0
        )
        warmed_up = self._canary_attempts_seen >= self._config.warmup_n
        return SpecialistStats(
            window_attempts=n,
            override_count=override_count,
            regret_count=regret_count,
            override_rate=override_rate,
            regret_rate=regret_rate,
            warmed_up=warmed_up,
        )

    # --- routing predicate -------------------------------------------------

    def is_routed(self, state_key: RecoveryStateKey) -> bool:
        return is_specialist_eligible(state_key)

    # --- main decision -----------------------------------------------------

    def choose_recovery(
        self,
        *,
        request: RecoveryRequest,
        baseline_action: str,
    ) -> SpecialistDecision:
        """Return the specialist's canary decision for one recovery dispatch.

        ``baseline_action`` is the deterministic recovery action the caller would take
        under the runtime baseline. The specialist never replaces it as the source of
        truth.
        """

        if baseline_action not in RECOVERY_ACTIONS_SET:
            raise SpecialistInputError(
                f"baseline_action={baseline_action!r} not a known "
                "recovery action"
            )
        audit_id = uuid.uuid4().hex
        routed = self.is_routed(request.state_key)
        in_canary = routed and _attempt_in_canary(
            request.attempt_id, self._config.canary_pct
        )

        rl_proposed: Optional[str] = None
        rl_blocked: bool = False
        mask_filtered: bool = False
        rl_confidence: float = 0.0
        rl_features: tuple[tuple[str, float], ...] = ()
        applied = baseline_action
        override = False
        fallback_triggered = self._fallback_active
        fallback_reason: Optional[str] = (
            self._fallback_reason if self._fallback_active else None
        )

        if routed and not self._fallback_active:
            try:
                sel = self._policy.propose_recovery(request)
            except Exception:  # noqa: BLE001 (defensive)
                logger.exception(
                    "Specialist policy raised on attempt %s; falling back to the "
                    "deterministic baseline %r",
                    request.attempt_id,
                    baseline_action,
                )
                self._engage_fallback_internal(SPECIALIST_FALLBACK_POLICY_ERROR)
                fallback_triggered = True
                fallback_reason = SPECIALIST_FALLBACK_POLICY_ERROR
            else:
                rl_proposed = sel.action
                rl_features = tuple(
                    (s.action, float(s.expected_reward)) for s in sel.scores
                )
                for s in sel.scores:
                    if s.action == sel.action:
                        rl_confidence = float(s.expected_reward)
                        break
                masked_action, mask_filtered, mask_blocked_all = (
                    _apply_action_mask(
                        proposed=sel.action,
                        ranked_actions=sel.ranked_actions,
                        blocked_actions=self._blocked,
                    )
                )
                if mask_blocked_all:
                    self._engage_fallback_internal(
                        SPECIALIST_FALLBACK_MASK_BLOCKED_ALL
                    )
                    fallback_triggered = True
                    fallback_reason = SPECIALIST_FALLBACK_MASK_BLOCKED_ALL
                    rl_blocked = True
                else:
                    if in_canary:
                        # Active canary path: apply the (masked) RL action.
                        assert masked_action is not None
                        applied = masked_action
                        override = applied != baseline_action

        # Record only the attempts that actually applied an RL action
        # (in_canary and not fallback-triggered) so override/regret
        # stats reflect real influence, not shadow-only observations.
        if (
            routed
            and in_canary
            and not fallback_triggered
        ):
            entry = _WindowEntry(audit_id=audit_id, override=override)
            self._window_record(entry)
            self._canary_attempts_seen += 1

        if not self._fallback_active:
            self._maybe_engage_auto_fallback()

        return SpecialistDecision(
            attempt_id=request.attempt_id,
            audit_id=audit_id,
            rl_mode=RL_MODE_RL_ACTIVE,
            rl_policy_id=self._policy.policy_id,
            rl_artifact_version=int(self._policy.version),
            rl_router_path=SPECIALIST_ROUTER_PATH,
            routed_to_specialist=routed,
            in_canary=in_canary,
            baseline_action=baseline_action,
            rl_action_proposed=rl_proposed,
            rl_action_blocked_by_mask=rl_blocked,
            mask_filtered_action=mask_filtered,
            rl_reason_features=rl_features,
            rl_confidence=rl_confidence,
            applied_action=applied,
            override=override,
            fallback_triggered=fallback_triggered,
            fallback_reason_code=fallback_reason,
        )

    # --- outcome reporting -------------------------------------------------

    def record_outcome(self, audit_id: str, outcome: str) -> None:
        if outcome not in ("succeeded", "failed"):
            raise SpecialistInputError(
                f"outcome must be 'succeeded' or 'failed'; got {outcome!r}"
            )
        entry = self._audit_index.get(audit_id)
        if entry is None:
            return
        entry.outcome = outcome
        if not self._fallback_active:
            self._maybe_engage_auto_fallback()

    # --- explicit fallback / reset ---------------------------------------

    def engage_fallback(
        self, reason_code: str = SPECIALIST_FALLBACK_OPERATOR
    ) -> None:
        if reason_code not in SPECIALIST_FALLBACK_REASONS:
            raise SpecialistInputError(
                f"unknown specialist fallback reason_code={reason_code!r}"
            )
        self._engage_fallback_internal(reason_code)

    def reset_canary(self) -> None:
        logger.info(
            "Specialist reset by operator (was %s, reason %s); this does not touch "
            "the base canary",
            "in fallback" if self._fallback_active else "live",
            self._fallback_reason,
        )
        self._fallback_active = False
        self._fallback_reason = None
        self._window.clear()
        self._audit_index.clear()
        self._canary_attempts_seen = 0

    # --- internals --------------------------------------------------------

    def _window_record(self, entry: _WindowEntry) -> None:
        if len(self._window) == self._window.maxlen:
            evicted = self._window[0]
            self._audit_index.pop(evicted.audit_id, None)
        self._window.append(entry)
        self._audit_index[entry.audit_id] = entry

    def _engage_fallback_internal(self, reason_code: str) -> None:
        if self._fallback_active:
            return
        # One funnel for every specialist fallback path, with the window evidence.
        stats = self.stats()
        logger.warning(
            "Specialist fallback engaged (%s) after %d canary attempt(s): override "
            "rate %.3f (max %.3f), regret rate %.3f (max %.3f) over a %d-attempt "
            "window. Stays off until reset_canary().",
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
        if s.window_attempts == 0:
            return
        if s.regret_rate > self._config.max_regret_rate:
            self._engage_fallback_internal(SPECIALIST_FALLBACK_REGRET_RATE)
            return
        if s.override_rate > self._config.max_override_rate:
            self._engage_fallback_internal(SPECIALIST_FALLBACK_OVERRIDE_RATE)
            return


# ---------------------------------------------------------------------------
# Convenience constructor.
# ---------------------------------------------------------------------------


def load_deformable_specialist_router(
    *,
    policy_artifact_path: Path,
    promotion_report_path: Path,
    config: Optional[SpecialistConfig] = None,
) -> DeformableSpecialistRouter:
    """Load the LinUCB recovery policy + promotion report into a specialist router."""

    try:
        policy = load_linucb_recovery_policy(policy_artifact_path)
    except (OSError, ValueError) as exc:
        raise SpecialistInputError(
            f"failed to load LinUCB recovery policy at {policy_artifact_path}: "
            f"{exc}"
        ) from exc
    try:
        return DeformableSpecialistRouter(
            policy=policy,
            promotion_report_path=promotion_report_path,
            config=config,
        )
    except PromotionInputError as exc:
        raise SpecialistInputError(str(exc)) from exc


# ---------------------------------------------------------------------------
# rl_experimental lane: scaffold only.
# ---------------------------------------------------------------------------


def route_rl_experimental(*args: Any, **kwargs: Any) -> "Any":
    """``rl_experimental`` lane entry point.

    Deliberately ships no artifact here. The lane exists as a
    typed constant (:data:`RL_MODE_RL_EXPERIMENTAL`, defined in
    ``src.config.schema.robot.rl_schema`` and re-exported by
    ``canary_router``) so config schemas, telemetry, and CLI surfaces
    can refer to it without inventing capabilities the codebase
    cannot honor. Calling this function raises
    :class:`ExperimentalLaneNotImplementedError` with the typed
    reason ``rl_experimental_not_implemented`` so callers can branch
    on the reason code rather than parsing a message.
    """

    _ = args, kwargs
    raise ExperimentalLaneNotImplementedError()


__all__ = (
    # constants
    "SPECIALIST_FAILURE_CLASS_DEFORMABLE",
    "SPECIALIST_DENSE_BUCKET",
    "DEFAULT_SPECIALIST_BLOCKED_ACTIONS",
    "SPECIALIST_FALLBACK_REGRET_RATE",
    "SPECIALIST_FALLBACK_OVERRIDE_RATE",
    "SPECIALIST_FALLBACK_OPERATOR",
    "SPECIALIST_FALLBACK_POLICY_ERROR",
    "SPECIALIST_FALLBACK_MASK_BLOCKED_ALL",
    "SPECIALIST_FALLBACK_REASONS",
    "EXPERIMENTAL_NOT_IMPLEMENTED_REASON",
    "SPECIALIST_ROUTER_PATH",
    # errors
    "SpecialistInputError",
    "ExperimentalLaneNotImplementedError",
    # types
    "SpecialistConfig",
    "SpecialistDecision",
    "SpecialistStats",
    "DeformableSpecialistRouter",
    # predicates / loaders / lanes
    "is_specialist_eligible",
    "load_deformable_specialist_router",
    "route_rl_experimental",
    # re-export for telemetry conformance
    "RL_MODE_RL_EXPERIMENTAL",
)
