"""Promotion OPE triple-extraction: shape logged replay records into ``EvaluationTriple`` rows for ``_promotion_estimators``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .ope import _build_sequencing_pairs
from .perception_budget_policy import (
    PERCEPTION_ACTIONS,
    LinUCBPerceptionBudgetPolicy,
    PerceptionBudgetRequest,
)
from .recovery_policy import LinUCBRecoveryPolicy, RecoveryRequest
from .sequencing_policy import (
    LookupTableSequencingPolicy,
    SequencingRequest,
)
from .train_perception_budget import (
    _is_in_training_scope as perception_record_in_training_scope,
    extract_action_label as extract_perception_action_label,
    extract_perception_state as extract_perception_state,
)
from .train_recovery import (
    _record_in_training_scope as recovery_record_in_training_scope,
    extract_recovery_tuples as extract_recovery_tuples,
)


@dataclass(frozen=True)
class EvaluationTriple:
    """One (behavior_action, target_action, reward) row used for OPE."""

    behavior_action: str
    target_action: str
    target_expected_reward: float  # Circular DM: the policy's own expected-reward self-estimate
    reward: float
    # The discrete state bucket (``state_key.as_string_key()``) the row was observed in. Every
    # extractor must set it; the "" default is there only for triples built to exercise the WIS and
    # self-report-DM math, which do not read the state bucket. The independent tabular DM
    # (``fit_tabular_dm``) buckets on this.
    state_key: str = ""


@dataclass(frozen=True)
class TripleExtractionResult:
    triples: tuple[EvaluationTriple, ...]
    num_records_total: int
    num_records_in_scope: int
    num_records_dropped: int


def extract_perception_triples(
    *,
    policy: LinUCBPerceptionBudgetPolicy,
    records: Sequence[Mapping[str, Any]],
    training_scope: str,
) -> TripleExtractionResult:
    """Extract evaluation triples for the perception-budget policy.

    The behaviour action comes from the trainer's label. The target
    action and its expected reward come from the policy's
    ``propose_perception_budget``.
    """

    triples: list[EvaluationTriple] = []
    total = 0
    in_scope = 0
    dropped = 0
    for rec in records:
        total += 1
        if not perception_record_in_training_scope(rec, training_scope):
            continue
        in_scope += 1
        try:
            state = extract_perception_state(rec)
        except Exception:  # noqa: BLE001 (defensive; trainer is source of truth)
            dropped += 1
            continue
        try:
            behavior_action = extract_perception_action_label(rec)
        except Exception:  # noqa: BLE001
            dropped += 1
            continue
        if behavior_action not in PERCEPTION_ACTIONS:
            dropped += 1
            continue
        request = PerceptionBudgetRequest(
            attempt_id=str(rec.get("attempt_id", "promotion")),
            state_key=state,
        )
        sel = policy.propose_perception_budget(request)
        # DM estimate for the target action.
        if sel.action == "stop":
            expected = float(sel.expected_reward_stop)
        else:
            expected = float(sel.expected_reward_continue)
        reward = _reward_from_record(rec)
        triples.append(
            EvaluationTriple(
                behavior_action=behavior_action,
                target_action=sel.action,
                target_expected_reward=expected,
                reward=reward,
                state_key=state.as_string_key(),
            )
        )
    return TripleExtractionResult(
        triples=tuple(triples),
        num_records_total=total,
        num_records_in_scope=in_scope,
        num_records_dropped=dropped,
    )


def extract_recovery_triples(
    *,
    policy: LinUCBRecoveryPolicy,
    records: Sequence[Mapping[str, Any]],
    training_scope: str,
) -> TripleExtractionResult:
    """Extract evaluation triples for the recovery policy.

    The behaviour action is the per-step logged action; the target
    action comes from the policy's ``propose_recovery``.
    """

    triples: list[EvaluationTriple] = []
    total = 0
    in_scope = 0
    dropped = 0
    for rec in records:
        total += 1
        if not recovery_record_in_training_scope(rec, training_scope):
            continue
        in_scope += 1
        rec_tuples, _dropped_unknown_token, _dropped_no_actions = (
            extract_recovery_tuples(rec)
        )
        for t in rec_tuples:
            request = RecoveryRequest(
                attempt_id=str(rec.get("attempt_id", "promotion")),
                state_key=t.state_key,
            )
            sel = policy.propose_recovery(request)
            expected = 0.0
            for score in sel.scores:
                if score.action == sel.action:
                    expected = float(score.expected_reward)
                    break
            triples.append(
                EvaluationTriple(
                    behavior_action=t.action,
                    target_action=sel.action,
                    target_expected_reward=expected,
                    reward=float(t.reward),
                    state_key=t.state_key.as_string_key(),
                )
            )
    return TripleExtractionResult(
        triples=tuple(triples),
        num_records_total=total,
        num_records_in_scope=in_scope,
        num_records_dropped=dropped,
    )


def extract_sequencing_triples(
    *,
    policy: LookupTableSequencingPolicy,
    records: Sequence[Mapping[str, Any]],
    training_scope: str,
) -> TripleExtractionResult:
    """Extract evaluation triples for the sequencing policy by reusing ``ope._build_sequencing_pairs`` (failure-only, successor-derived ``(state, behaviour-action, reward)`` pairs) unmodified: its OPE report output is hash-pinned. Reward is the cycle-time-weighted ``OPE_REWARD_COEFFICIENTS_V1``; the lookup table exposes no per-action expected reward, so the circular ``target_expected_reward`` stays 0.0 and the independent tabular DM is the real DM."""

    _ = training_scope  # recorded by the caller for provenance; pairing is over all groups.
    triples: list[EvaluationTriple] = []
    pairs = _build_sequencing_pairs(records)
    for state, behaviour_action, reward, _nxt in pairs:
        sel = policy.propose_sequencing(
            SequencingRequest(attempt_id="promotion", state_key=state)
        )
        triples.append(
            EvaluationTriple(
                behavior_action=behaviour_action,
                target_action=sel.action,
                target_expected_reward=0.0,
                reward=float(reward),
                state_key=state.as_string_key(),
            )
        )
    total = len(records)
    return TripleExtractionResult(
        triples=tuple(triples),
        num_records_total=total,
        num_records_in_scope=total,
        num_records_dropped=total - len(triples),
    )


def _reward_from_record(rec: Mapping[str, Any]) -> float:
    """Binary attempt reward: ``1.0`` on success, ``0.0`` otherwise, matching the trainers' canonical-pack labeling."""

    final = rec.get("final_outcome")
    return 1.0 if final == "succeeded" else 0.0
