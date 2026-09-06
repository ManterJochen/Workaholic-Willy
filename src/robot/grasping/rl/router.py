"""Shadow router umbrella.

The shadow router is the single seam that the runtime calls when
``robot.rl.mode`` is one of the RL-active modes. It is the only
dispatch class the runtime imports from
:mod:`src.robot.grasping.rl` for inference, so a new dispatcher
attaches to a new slot here instead of changing
:mod:`src.robot.execution.autonomous_grasp`.

``candidate_policy`` is the one required dispatcher; every other slot
(``ranking_policy``, ``sequencing_policy``, ``perception_policy``,
``recovery_policy``) is reserved with a frozen ``None`` default so the
contract is locked while implementations land incrementally.

Authority lock: the router never overrides safety, geometry, or
deterministic decisions. Its only effect is to emit a
:class:`ShadowRouterTelemetry` carrier on
``AutonomousGraspReport.shadow_router_telemetry``: zero behavior
change, full audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from src.robot.grasping.constants import (
    RL_ROUTER_LOG_FILE,
    create_grasping_logger,
)

from .action_mask import (
    ActionMaskContext,
    apply_action_mask,
    mask_summary,
)
from .candidate_policy import (
    CandidateFeatures,
    CandidatePolicy,
    CandidateSelection,
    CandidateSelectionRequest,
)
from .ranking_policy import (
    RankingCandidateFeatures,
    RankingPolicy,
    RankingSelection,
    RankingSelectionRequest,
)
from .sequencing_policy import (
    SEQUENCING_ACTIONS_SET,
    SequencingDecision,
    SequencingPolicy,
    SequencingRequest,
    SequencingStateKey,
    apply_anti_loop_gate,
)
from .perception_budget_policy import (
    PerceptionBudgetPolicy,
    PerceptionBudgetRequest,
    PerceptionBudgetSelection,
    PerceptionBudgetStateKey,
)
from .recovery_policy import (
    RECOVERY_ACTIONS,
    RECOVERY_ACTIONS_SET,
    RECOVERY_ACTION_ABORT_RECOVERY,
    RecoveryActionScore,
    RecoveryPolicy,
    RecoveryRequest,
    RecoverySelection,
    RecoveryStateKey,
    apply_recovery_anti_loop_gate,
)
from ._router_telemetry import (
    BREAKDOWN_TOP_N,
    KENDALL_TAU_BOUNDS,
    _clamp,
    build_candidate_breakdown,
    emit_candidate_shadow_extras,
    emit_ranking_extras,
    emit_sequencing_extras,
    emit_perception_extras,
    emit_recovery_extras,
)


#: Router-path token. Mirrors the value the telemetry catalog accepts
#: on ``record.extra['rl_router_path']``.
ROUTER_PATH_RL_SHADOW: str = "rl_shadow"

#: Fallback-path token: on a router exception the caller falls back to
#: the baseline and logs a structured reason. Allowed by the catalog.
ROUTER_PATH_FALLBACK_BASELINE: str = "fallback_baseline"


#: Every shadow path here catches any policy exception and converts it into a
#: typed fallback row whose reason code is built from the exception class name
#: and never the message text; three paths prefix it with a path token
#: (``router_exception:``, ``ranking_exception:``, ``sequencing_exception:``).
#: The log is the only place the message and traceback survive: in telemetry
#: alone, a shadow policy that throws on every attempt looks exactly like one
#: that is simply off.
logger = create_grasping_logger("RLShadowRouter", RL_ROUTER_LOG_FILE)


@dataclass(frozen=True)
class CandidateAgreement:
    """Disagreement summary between policy proposal and baseline.

    Attributes
    ----------
    top1_agree
        ``True`` iff the policy's top-1 (post-prune) candidate id
        matches ``deterministic_top1_id``. ``False`` when either side
        produced no candidate at all (one-side-missing is treated as
        disagreement so canary thresholds catch silent regressions).
    kendall_tau
        Kendall's tau rank-correlation across the common candidate
        set. Clamped to ``[-1.0, 1.0]``. ``0.0`` when fewer than two
        candidates are common.
    pruned_count
        Number of proposals the policy flagged ``pruned=True``.
    """

    top1_agree: bool
    kendall_tau: float
    pruned_count: int


@dataclass(frozen=True)
class RankingShadowTelemetry:
    """Ranking sub-block carried on :class:`ShadowRouterTelemetry`.

    Rides alongside the candidate-selection telemetry; :class:`PickReport`
    and its schema are left untouched.
    """

    router_path: str
    policy_id: str
    policy_version: int
    selection: RankingSelection
    regret_top1: bool
    kendall_tau: float
    fallback_triggered: bool = False
    fallback_reason_code: str = "none"


@dataclass(frozen=True)
class ShadowRouterTelemetry:
    """Typed shadow-router telemetry attached to the grasp report.

    Mutating any field requires a contract version bump
    (:data:`SHADOW_ROUTER_TELEMETRY_VERSION`). The runtime never
    reads back from this carrier: it is logging-only by construction.

    The ranking sub-block (:attr:`ranking_telemetry`) is :data:`None`
    when no ranking policy is wired or when the seam was never reached
    for this attempt.
    """

    router_path: str
    policy_id: str
    policy_version: int
    selection: CandidateSelection
    mask_summary: Mapping[str, int]
    agreement: CandidateAgreement
    fallback_triggered: bool = False
    fallback_reason_code: str = "none"
    # Default ``None`` keeps every caller byte-compatible: no
    # positional shift.
    ranking_telemetry: Optional[RankingShadowTelemetry] = None
    sequencing_telemetry: "Optional[SequencingShadowTelemetry]" = None
    # The perception sub-block lives only on this internal carrier; it is
    # not mirrored into the replay-record ``extra.*`` schema or the
    # telemetry catalog.
    perception_telemetry: "Optional[PerceptionShadowTelemetry]" = None
    # The recovery sub-block lives only on this internal carrier; it is
    # not mirrored into the replay-record ``extra.*`` schema or the
    # telemetry catalog.
    recovery_telemetry: "Optional[RecoveryShadowTelemetry]" = None


#: Bumped on any change to :class:`ShadowRouterTelemetry` shape or semantics.
SHADOW_ROUTER_TELEMETRY_VERSION: int = 5


@dataclass(frozen=True)
class SequencingShadowTelemetry:
    """Sequencing sub-block carried on :class:`ShadowRouterTelemetry`.

    The per-attempt :class:`SequencingDecision` rows ride in
    :attr:`decisions` (tuple, deterministic order = invocation order).
    The extras dict summarises only the last failure attempt;
    consumers that need the full trace read this carrier.
    """

    router_path: str
    policy_id: str
    policy_version: int
    decisions: tuple[SequencingDecision, ...] = ()
    fallback_triggered: bool = False
    fallback_reason_code: str = "none"


@dataclass(frozen=True)
class PerceptionShadowTelemetry:
    """Perception sub-block carried on :class:`ShadowRouterTelemetry`.

    Emitted by :meth:`ShadowRouter.run_perception_shadow` per attempt
    when a perception-budget policy is wired. The runtime never
    reads back from this carrier: it is logging-only by construction
    (no replay-record ``extra.*`` mirror).
    """

    router_path: str
    policy_id: str
    policy_version: int
    selection: PerceptionBudgetSelection
    state_key: PerceptionBudgetStateKey
    baseline_action: str
    agree_with_baseline: bool
    fallback_triggered: bool = False
    fallback_reason_code: str = "none"


@dataclass(frozen=True)
class RecoveryShadowTelemetry:
    """Recovery sub-block carried on :class:`ShadowRouterTelemetry`.

    Emitted by :meth:`ShadowRouter.run_recovery_shadow` per recovery
    invocation when a recovery policy is wired. The runtime never
    reads back from this carrier: it is logging-only by construction
    (no replay-record ``extra.*`` mirror).

    :attr:`gated_action` is the proposed action after the
    defence-in-depth anti-loop gate has run; :attr:`gate_clipped`
    indicates the gate overrode the policy proposal (e.g. forced
    ``abort_recovery``).
    """

    router_path: str
    policy_id: str
    policy_version: int
    selection: RecoverySelection
    state_key: RecoveryStateKey
    baseline_action: str
    gated_action: str
    attempt_index: int
    max_recovery_attempts: int
    agree_with_baseline: bool
    gate_clipped: bool = False
    gate_clip_reason: str = "none"
    fallback_triggered: bool = False
    fallback_reason_code: str = "none"


@dataclass(frozen=True)
class SequencingAttemptState:
    """Router input for one post-attempt sequencing invocation.

    Carries everything :meth:`ShadowRouter.run_sequencing_shadow`
    needs to (a) build the state key, (b) apply the anti-loop gate,
    and (c) compute baseline agreement. Every field arrives in final
    form from the producer at the ``pick_loop`` seam: the two
    ``*_clipped`` fields are clipped there, the raw counterparts feed
    the gate, and this class clips nothing.
    """

    attempt_index: int
    last_outcome_label: str
    attempt_index_clipped: int
    commit_reobserve_count_clipped: int
    commit_reobserve_count: int
    last_failure_class: str
    max_attempts: int
    max_reobserve_attempts: int
    baseline_action: str


@dataclass(frozen=True)
class ShadowRouter:
    """Umbrella router for the shadow dispatchers.

    Only :attr:`candidate_policy` is required. Reserved slots accept
    any object (``None`` default) so subclassing is not needed when
    more producers land; the runtime call surface is fixed.
    """

    candidate_policy: CandidatePolicy
    # Typed slots; ``None`` is the silent shadow-off path, populated
    # incrementally.
    ranking_policy: Optional[RankingPolicy] = None
    sequencing_policy: Optional[SequencingPolicy] = None
    perception_policy: Optional[PerceptionBudgetPolicy] = None
    recovery_policy: Optional[RecoveryPolicy] = None
    router_path: str = ROUTER_PATH_RL_SHADOW

    def run_perception_shadow(
        self,
        *,
        attempt_id: str,
        state_key: PerceptionBudgetStateKey,
        baseline_action: str,
    ) -> "Optional[PerceptionShadowTelemetry]":
        """Run the perception-budget shadow path.

        Returns ``None`` when no perception policy is wired (silent
        shadow-off path).

        Any exception from the policy is caught and converted into a
        typed fallback telemetry row (``fallback_triggered=True``)
        whose action is the baseline: defence-in-depth so a buggy
        policy can never crash the runtime.
        """

        if self.perception_policy is None:
            return None
        from .perception_budget_policy import (  # local import to avoid cycle
            PERCEPTION_ACTIONS_SET,
            PerceptionBudgetSelection,
        )

        if baseline_action not in PERCEPTION_ACTIONS_SET:
            # Defence-in-depth: a misbehaving caller is funneled to a
            # canonical fallback row so downstream consumers never see
            # an invalid baseline_action token.
            baseline_action_safe = "continue"
        else:
            baseline_action_safe = baseline_action
        try:
            selection = self.perception_policy.propose_perception_budget(
                PerceptionBudgetRequest(
                    attempt_id=attempt_id, state_key=state_key
                )
            )
            return PerceptionShadowTelemetry(
                router_path=self.router_path,
                policy_id=selection.policy_id,
                policy_version=selection.policy_version,
                selection=selection,
                state_key=state_key,
                baseline_action=baseline_action_safe,
                agree_with_baseline=(
                    selection.action == baseline_action_safe
                ),
                fallback_triggered=False,
                fallback_reason_code="none",
            )
        except Exception as exc:  # noqa: BLE001 (bounded by typed fallback)
            logger.exception(
                "Perception-budget shadow fell back to baseline %r for attempt %s: %s",
                baseline_action_safe,
                attempt_id,
                exc,
            )
            fallback_selection = PerceptionBudgetSelection(
                policy_id=getattr(self.perception_policy, "name", "unknown")
                + "@"
                + str(getattr(self.perception_policy, "version", 0)),
                policy_version=int(
                    getattr(self.perception_policy, "version", 0)
                ),
                action=baseline_action_safe,
                expected_reward_stop=0.0,
                expected_reward_continue=0.0,
                ucb_stop=0.0,
                ucb_continue=0.0,
                support_stop=0,
                support_continue=0,
                used_fallback=True,
            )
            return PerceptionShadowTelemetry(
                router_path=ROUTER_PATH_FALLBACK_BASELINE,
                policy_id=fallback_selection.policy_id,
                policy_version=fallback_selection.policy_version,
                selection=fallback_selection,
                state_key=state_key,
                baseline_action=baseline_action_safe,
                agree_with_baseline=True,  # by construction in fallback
                fallback_triggered=True,
                fallback_reason_code=type(exc).__name__,
            )

    def run_recovery_shadow(
        self,
        *,
        attempt_id: str,
        state_key: RecoveryStateKey,
        baseline_action: str,
        attempt_index: int,
        max_recovery_attempts: int,
    ) -> "Optional[RecoveryShadowTelemetry]":
        """Run the recovery-optimization shadow path.

        Returns ``None`` when no recovery policy is wired (silent
        shadow-off path). The runtime never reads back from the
        returned carrier: it is logging-only.

        The defence-in-depth anti-loop gate
        (:func:`apply_recovery_anti_loop_gate`) runs on every proposal,
        forcing ``abort_recovery`` once ``attempt_index >=
        max_recovery_attempts``. Any exception from the policy is
        caught and converted into a typed fallback telemetry row whose
        action is the (gated) baseline.
        """

        if self.recovery_policy is None:
            return None

        # Defence-in-depth: a ``baseline_action`` outside
        # ``RECOVERY_ACTIONS_SET`` is replaced by the canonical
        # ``abort_recovery`` token, so downstream consumers never see
        # an invalid value.
        if baseline_action not in RECOVERY_ACTIONS_SET:
            baseline_action_safe = RECOVERY_ACTION_ABORT_RECOVERY
        else:
            baseline_action_safe = baseline_action
        try:
            selection = self.recovery_policy.propose_recovery(
                RecoveryRequest(
                    attempt_id=attempt_id, state_key=state_key
                )
            )
            gate = apply_recovery_anti_loop_gate(
                proposed_action=selection.action,
                attempt_index=attempt_index,
                max_recovery_attempts=max_recovery_attempts,
            )
            return RecoveryShadowTelemetry(
                router_path=self.router_path,
                policy_id=selection.policy_id,
                policy_version=selection.policy_version,
                selection=selection,
                state_key=state_key,
                baseline_action=baseline_action_safe,
                gated_action=gate.action,
                attempt_index=int(attempt_index),
                max_recovery_attempts=int(max_recovery_attempts),
                agree_with_baseline=(gate.action == baseline_action_safe),
                gate_clipped=gate.clipped,
                gate_clip_reason=gate.clip_reason,
                fallback_triggered=False,
                fallback_reason_code="none",
            )
        except Exception as exc:  # noqa: BLE001 (bounded by typed fallback)
            logger.exception(
                "Recovery shadow fell back to baseline %r for attempt %s: %s",
                baseline_action_safe,
                attempt_id,
                exc,
            )
            # Apply the gate to the baseline as well so a gate clip
            # is visible even when the policy crashed.
            gate = apply_recovery_anti_loop_gate(
                proposed_action=baseline_action_safe,
                attempt_index=attempt_index,
                max_recovery_attempts=max_recovery_attempts,
            )
            fallback_selection = RecoverySelection(
                policy_id=getattr(self.recovery_policy, "name", "unknown")
                + "@"
                + str(getattr(self.recovery_policy, "version", 0)),
                policy_version=int(
                    getattr(self.recovery_policy, "version", 0)
                ),
                action=gate.action,
                ranked_actions=(gate.action,)
                + tuple(a for a in RECOVERY_ACTIONS if a != gate.action),
                scores=tuple(
                    RecoveryActionScore(
                        action=a,
                        expected_reward=0.0,
                        ucb=0.0,
                        min_support_over_active=0,
                    )
                    for a in RECOVERY_ACTIONS
                ),
                used_fallback=True,
            )
            return RecoveryShadowTelemetry(
                router_path=ROUTER_PATH_FALLBACK_BASELINE,
                policy_id=fallback_selection.policy_id,
                policy_version=fallback_selection.policy_version,
                selection=fallback_selection,
                state_key=state_key,
                baseline_action=baseline_action_safe,
                gated_action=gate.action,
                attempt_index=int(attempt_index),
                max_recovery_attempts=int(max_recovery_attempts),
                agree_with_baseline=True,  # by construction in fallback
                gate_clipped=gate.clipped,
                gate_clip_reason=gate.clip_reason,
                fallback_triggered=True,
                fallback_reason_code=type(exc).__name__,
            )

    def run_candidate_selection(
        self,
        *,
        attempt_id: str,
        candidate_ids: Sequence[str],
        per_candidate_features: Mapping[str, Mapping[str, object]],
        mask_context: ActionMaskContext,
        deterministic_top1_id: Optional[str],
    ) -> ShadowRouterTelemetry:
        """Run the candidate-selection shadow path.

        The router applies the action mask first, then forwards the
        unmasked candidates to :attr:`candidate_policy`. Any exception
        is caught and converted to a typed fallback telemetry row so
        the runtime can never crash on a policy bug.
        """

        masked_rows = apply_action_mask(candidate_ids, mask_context)
        ms = dict(mask_summary(masked_rows))

        try:
            unmasked = tuple(row.candidate_id for row in masked_rows if row.keep)
            features = tuple(
                CandidateFeatures(
                    candidate_id=cid,
                    features=per_candidate_features.get(cid, {}),
                )
                for cid in unmasked
            )
            request = CandidateSelectionRequest(
                attempt_id=attempt_id,
                candidates=features,
                deterministic_top1_id=deterministic_top1_id,
            )
            selection = self.candidate_policy.propose(request)
            # Safety contract: the router must reject any proposal
            # whose id is not in the unmasked request. A misbehaving
            # policy is treated as a fallback. The mask already keeps
            # masked ids from reaching the policy; this is the
            # defence-in-depth re-check on the way back.
            unmasked_set = set(unmasked)
            for p in selection.proposals:
                if p.candidate_id not in unmasked_set:
                    raise ValueError(
                        f"policy {selection.policy_id!r} returned "
                        f"masked or unknown candidate "
                        f"{p.candidate_id!r}"
                    )
            agreement = _compute_agreement(
                selection=selection,
                deterministic_top1_id=deterministic_top1_id,
            )
            return ShadowRouterTelemetry(
                router_path=self.router_path,
                policy_id=selection.policy_id,
                policy_version=selection.policy_version,
                selection=selection,
                mask_summary=ms,
                agreement=agreement,
                fallback_triggered=False,
                fallback_reason_code="none",
            )
        except Exception as exc:  # noqa: BLE001 (bounded by typed fallback)
            # Reason code is the exception class name so operators can
            # grep for it in soak logs without leaking message text.
            logger.exception(
                "Candidate selection fell back to the deterministic order for "
                "attempt %s (%d candidate(s), %d unmasked): %s",
                attempt_id,
                len(candidate_ids),
                sum(1 for row in masked_rows if row.keep),
                exc,
            )
            reason = type(exc).__name__
            empty_selection = CandidateSelection(
                policy_id=getattr(
                    self.candidate_policy, "name", "unknown"
                ),
                policy_version=int(
                    getattr(self.candidate_policy, "version", 0)
                ),
                proposals=(),
            )
            return ShadowRouterTelemetry(
                router_path=ROUTER_PATH_FALLBACK_BASELINE,
                policy_id=empty_selection.policy_id,
                policy_version=empty_selection.policy_version,
                selection=empty_selection,
                mask_summary=ms,
                agreement=CandidateAgreement(
                    top1_agree=False,
                    kendall_tau=0.0,
                    pruned_count=0,
                ),
                fallback_triggered=True,
                fallback_reason_code=f"router_exception:{reason}",
            )

    def run_ranking_shadow(
        self,
        *,
        attempt_id: str,
        candidate_ids: Sequence[str],
        per_candidate_features: Mapping[str, Mapping[str, object]],
        deterministic_ranking: Sequence[str],
    ) -> "RankingShadowTelemetry":
        """Run the ranking shadow path.

        Fail-safe: any exception raised by the policy is converted to
        a typed fallback :class:`RankingShadowTelemetry` row with
        :attr:`fallback_reason_code` ``"ranking_exception:<ExcName>"``.
        When :attr:`ranking_policy` is :data:`None` the caller must not
        invoke this method; the runtime checks for the policy slot
        before calling.
        """

        policy = self.ranking_policy
        if policy is None:
            # Defensive: a missing policy yields an empty selection and
            # a fallback row with reason ``ranking_exception:NoPolicy``
            # rather than an exception.
            empty = RankingSelection(
                policy_id="unset",
                policy_version=0,
                proposals=(),
                regret_top1=False,
                kendall_tau=0.0,
            )
            return RankingShadowTelemetry(
                router_path=ROUTER_PATH_FALLBACK_BASELINE,
                policy_id="unset",
                policy_version=0,
                selection=empty,
                regret_top1=False,
                kendall_tau=0.0,
                fallback_triggered=True,
                fallback_reason_code="ranking_exception:NoPolicy",
            )
        try:
            features = tuple(
                RankingCandidateFeatures(
                    candidate_id=cid,
                    features=per_candidate_features.get(cid, {}),
                )
                for cid in candidate_ids
            )
            request = RankingSelectionRequest(
                attempt_id=attempt_id,
                candidates=features,
                deterministic_ranking=tuple(deterministic_ranking),
            )
            selection = policy.propose_ranking(request)
            # Safety contract: defence-in-depth check that the policy
            # only returns ids the request supplied.
            known = set(candidate_ids)
            for p in selection.proposals:
                if p.candidate_id not in known:
                    raise ValueError(
                        f"ranking policy {selection.policy_id!r} "
                        f"returned unknown candidate "
                        f"{p.candidate_id!r}"
                    )
            return RankingShadowTelemetry(
                router_path=self.router_path,
                policy_id=selection.policy_id,
                policy_version=selection.policy_version,
                selection=selection,
                regret_top1=bool(selection.regret_top1),
                kendall_tau=_clamp(
                    float(selection.kendall_tau), *KENDALL_TAU_BOUNDS
                ),
                fallback_triggered=False,
                fallback_reason_code="none",
            )
        except Exception as exc:  # noqa: BLE001 (bounded by typed fallback)
            logger.exception(
                "Ranking shadow fell back to the deterministic order for attempt "
                "%s: %s",
                attempt_id,
                exc,
            )
            reason = type(exc).__name__
            empty = RankingSelection(
                policy_id=getattr(policy, "name", "unknown"),
                policy_version=int(getattr(policy, "version", 0)),
                proposals=(),
                regret_top1=False,
                kendall_tau=0.0,
            )
            return RankingShadowTelemetry(
                router_path=ROUTER_PATH_FALLBACK_BASELINE,
                policy_id=empty.policy_id,
                policy_version=empty.policy_version,
                selection=empty,
                regret_top1=False,
                kendall_tau=0.0,
                fallback_triggered=True,
                fallback_reason_code=f"ranking_exception:{reason}",
            )

    def run_sequencing_shadow(
        self,
        *,
        attempt_id: str,
        states: "Sequence[SequencingAttemptState]",
    ) -> "SequencingShadowTelemetry":
        """Run the sequencing shadow path.

        Invoked once per pick with the per-failure-attempt state list
        (success outcomes are excluded by the producer at the pick_loop
        seam, never here). The router:

        1. consults :attr:`sequencing_policy` for each state to get a
           :class:`SequencingSelection` proposal,
        2. clips the proposal through :func:`apply_anti_loop_gate`
           (consumer-side defence-in-depth),
        3. emits one :class:`SequencingDecision` row per state with
           ``proposed_action`` / ``gated_action`` recorded separately.

        Fail-safe: any exception aborts the run for this pick and
        returns a typed fallback with ``fallback_reason_code``
        ``"sequencing_exception:<ExcName>"``. When
        :attr:`sequencing_policy` is :data:`None` the contract is the
        caller does not invoke this method; a typed fallback row is
        still built defensively.
        """

        policy = self.sequencing_policy
        if policy is None:
            return SequencingShadowTelemetry(
                router_path=ROUTER_PATH_FALLBACK_BASELINE,
                policy_id="unset",
                policy_version=0,
                decisions=(),
                fallback_triggered=True,
                fallback_reason_code="sequencing_exception:NoPolicy",
            )
        try:
            decisions: list[SequencingDecision] = []
            policy_id = getattr(policy, "name", "unknown")
            policy_version = int(getattr(policy, "version", 0))
            for state in states:
                state_key = SequencingStateKey(
                    last_outcome_label=state.last_outcome_label,
                    attempt_index_clipped=state.attempt_index_clipped,
                    commit_reobserve_count_clipped=(
                        state.commit_reobserve_count_clipped
                    ),
                    last_failure_class=state.last_failure_class,
                )
                request = SequencingRequest(
                    attempt_id=f"{attempt_id}:{state.attempt_index}",
                    state_key=state_key,
                )
                selection = policy.propose_sequencing(request)
                policy_id = selection.policy_id
                policy_version = int(selection.policy_version)
                if selection.action not in SEQUENCING_ACTIONS_SET:
                    raise ValueError(
                        f"sequencing policy {selection.policy_id!r} "
                        f"returned unknown action {selection.action!r}"
                    )
                gated_action, clipped = apply_anti_loop_gate(
                    proposed_action=selection.action,
                    attempt_index=state.attempt_index,
                    commit_reobserve_count=state.commit_reobserve_count,
                    max_attempts=state.max_attempts,
                    max_reobserve_attempts=state.max_reobserve_attempts,
                )
                if state.baseline_action not in SEQUENCING_ACTIONS_SET:
                    raise ValueError(
                        f"baseline_action {state.baseline_action!r} "
                        "not in the sequencing action enum"
                    )
                decisions.append(
                    SequencingDecision(
                        attempt_index=int(state.attempt_index),
                        state_key=state_key,
                        proposed_action=selection.action,
                        gated_action=gated_action,
                        baseline_action=state.baseline_action,
                        agree_with_baseline=(
                            gated_action == state.baseline_action
                        ),
                        anti_loop_clipped=bool(clipped),
                        support_count=int(selection.support_count),
                        used_fallback=bool(selection.used_fallback),
                    )
                )
            return SequencingShadowTelemetry(
                router_path=self.router_path,
                policy_id=policy_id,
                policy_version=policy_version,
                decisions=tuple(decisions),
                fallback_triggered=False,
                fallback_reason_code="none",
            )
        except Exception as exc:  # noqa: BLE001 (bounded by typed fallback)
            logger.exception(
                "Sequencing shadow fell back to baseline for attempt %s: %s",
                attempt_id,
                exc,
            )
            reason = type(exc).__name__
            return SequencingShadowTelemetry(
                router_path=ROUTER_PATH_FALLBACK_BASELINE,
                policy_id=getattr(policy, "name", "unknown"),
                policy_version=int(getattr(policy, "version", 0)),
                decisions=(),
                fallback_triggered=True,
                fallback_reason_code=f"sequencing_exception:{reason}",
            )


# ---------------------------------------------------------------------------
# Telemetry helpers.
# ---------------------------------------------------------------------------


def _compute_agreement(
    *,
    selection: CandidateSelection,
    deterministic_top1_id: Optional[str],
) -> CandidateAgreement:
    """Compute the agreement triple from a selection."""

    pruned_count = sum(1 for p in selection.proposals if p.pruned)
    top1 = selection.top1_id
    if deterministic_top1_id is None or top1 is None:
        top1_agree = False
    else:
        top1_agree = top1 == deterministic_top1_id
    # Kendall tau across the common set. The baseline ordering outside
    # top-1 is not available at this seam, so tau is reported as 0.0
    # here; the ranking policy fills it in with the full baseline
    # permutation. The field stays type-stable either way.
    tau = 0.0
    return CandidateAgreement(
        top1_agree=bool(top1_agree),
        kendall_tau=_clamp(tau, *KENDALL_TAU_BOUNDS),
        pruned_count=int(pruned_count),
    )


__all__ = (
    "BREAKDOWN_TOP_N",
    "CandidateAgreement",
    "KENDALL_TAU_BOUNDS",
    "ROUTER_PATH_FALLBACK_BASELINE",
    "ROUTER_PATH_RL_SHADOW",
    "SHADOW_ROUTER_TELEMETRY_VERSION",
    "PerceptionShadowTelemetry",
    "RankingShadowTelemetry",
    "RecoveryShadowTelemetry",
    "SequencingAttemptState",
    "SequencingShadowTelemetry",
    "ShadowRouter",
    "ShadowRouterTelemetry",
    "build_candidate_breakdown",
    "emit_candidate_shadow_extras",
    "emit_ranking_extras",
    "emit_sequencing_extras",
    "emit_perception_extras",
    "emit_recovery_extras",
)
