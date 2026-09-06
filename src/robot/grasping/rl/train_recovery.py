"""Trainer for the LinUCB recovery optimization policy.

Pure-stdlib trainer that ingests replay-pack records, derives per-recovery
(state, action, reward) tuples, fits diagonal-A LinUCB statistics, and
serialises a deterministic baseline artifact.

Action labels: inferred from each record's ``recovery_actions`` list
using a frozen token map (``recovery_actions_token_map_v1``). Records
whose ``recovery_actions`` are empty or whose tokens do not map to the
action set are dropped (never silently relabelled); the drop count is
reported via :class:`RecoveryTrainResult`.

Training scope: dense-only by default. Pass ``training_scope="all"``
(CLI overridable) to fit on every record.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

from src.robot.grasping.constants import (
    RL_TRAIN_RECOVERY_LOG_FILE,
    create_grasping_logger,
)

from ._artifact_io import write_artifact_json
from ._io import load_jsonl
from .recovery_policy import (
    DEFAULT_FALLBACK_TABLE,
    DEFAULT_LINUCB_ALPHA,
    DEFAULT_LINUCB_LAMBDA,
    DEFAULT_MIN_SUPPORT_THRESHOLD,
    DENSE_BUCKET_DENSE,
    FAILURE_CLASS_BUCKETS,
    LAST_OUTCOME_FAILED,
    LAST_OUTCOME_RECOVERED,
    LAST_OUTCOME_SUCCEEDED,
    LAST_OUTCOME_UNKNOWN,
    ONEHOT_FAMILIES,
    RECOVERY_ACTIONS,
    RECOVERY_ACTIONS_SET,
    RECOVERY_ACTION_ABORT_RECOVERY,
    RECOVERY_ACTION_CONTAINER_AGITATE,
    RECOVERY_ACTION_NEXT_TARGET,
    RECOVERY_ACTION_NUDGE_TARGET,
    RECOVERY_ACTION_PERTURB_AND_RETRY,
    RECOVERY_ACTION_REOBSERVE,
    RECOVERY_ACTION_REPLAN_GRASP,
    RECOVERY_ACTION_RE_SEGMENT,
    RECOVERY_ARTIFACT_SCHEMA_VERSION,
    LinUCBRecoveryPolicy,
    RecoveryRequest,
    RecoveryStateKey,
    bucket_attempt_index,
    bucket_dense,
    bucket_reobserve_count,
    onehot_dim,
    onehot_index_layout,
    state_key_to_onehot,
)
from .honesty import (
    INPUT_SYNTHETIC_DETERMINISTIC,
    REWARD_LOGGED_OUTCOMES,
    make_dataset_provenance,
    single_action_degeneracy_note,
)
from .sequencing_policy import (
    FAILURE_CLASS_COLLISION,
    FAILURE_CLASS_DRIFT,
    FAILURE_CLASS_EMPTY_AIR,
    FAILURE_CLASS_OCCLUSION,
    FAILURE_CLASS_SLIP,
    FAILURE_CLASS_UNKNOWN,
)

#: A deterministic behaviour policy only ever picks a few arms, so the remaining
#: actions end with zero support and the runtime falls back on every state cell.
#: Only the per-action counts show it, so the log line names every action that
#: was never observed and warns whenever there is one.
logger = create_grasping_logger("RLTrainRecovery", RL_TRAIN_RECOVERY_LOG_FILE)


ACTION_LABEL_SOURCE: str = "recovery_actions_token_map_v1"

# Honesty stamps. `honesty.py` is the shared source of truth for the reward-model enum and the
# provenance shape.
RECOVERY_REWARD_INTERPRETATION: str = (
    "off-policy LinUCB over the LOGGED binary recovery outcome (recovered_success), NOT a synthetic "
    "outcome model; it ranks which recovery action to try. The committed baseline trains on the SYNTHETIC "
    "canonical recovery packs (see dataset_provenance.input_source), so the coefficients characterise "
    "mechanics only; NOT real-hardware lift. Real lift needs captured replay data + a passing offline promotion gate."
)
RECOVERY_DATASET_FIT_FOR: str = "recovery-action selection in SHADOW (off-policy)"
RECOVERY_DATASET_NOT_FIT_FOR: tuple[str, ...] = (
    "real-hardware lift",
    "active runtime control without a passing offline promotion gate",
)

#: Frozen token map from the arbitrary ``recovery_actions[i].action``
#: strings observed in canonical packs and runtime traces onto the
#: action enum. Never widen silently; unknown tokens drop the record.
RECOVERY_TOKEN_MAP: Mapping[str, str] = {
    # reobserve family
    "reobserve": RECOVERY_ACTION_REOBSERVE,
    "next_viewpoint": RECOVERY_ACTION_REOBSERVE,
    "active_viewpoint": RECOVERY_ACTION_REOBSERVE,
    "view_change": RECOVERY_ACTION_REOBSERVE,
    # re_segment family
    "re_segment": RECOVERY_ACTION_RE_SEGMENT,
    "resegment": RECOVERY_ACTION_RE_SEGMENT,
    "rescan": RECOVERY_ACTION_RE_SEGMENT,
    # replan_grasp family
    "replan_grasp": RECOVERY_ACTION_REPLAN_GRASP,
    "regenerate_grasp": RECOVERY_ACTION_REPLAN_GRASP,
    "refine": RECOVERY_ACTION_REPLAN_GRASP,
    "regrasp": RECOVERY_ACTION_REPLAN_GRASP,
    # perturb_and_retry family
    "perturb_and_retry": RECOVERY_ACTION_PERTURB_AND_RETRY,
    "lateral_perturbation": RECOVERY_ACTION_PERTURB_AND_RETRY,
    "perturb": RECOVERY_ACTION_PERTURB_AND_RETRY,
    "retry": RECOVERY_ACTION_PERTURB_AND_RETRY,
    # abort_recovery family
    "abort_recovery": RECOVERY_ACTION_ABORT_RECOVERY,
    "abort": RECOVERY_ACTION_ABORT_RECOVERY,
    "give_up": RECOVERY_ACTION_ABORT_RECOVERY,
    # The three concrete sim recovery actions, identity-mapped.
    "next_target": RECOVERY_ACTION_NEXT_TARGET,
    "nudge_target": RECOVERY_ACTION_NUDGE_TARGET,
    "container_agitate": RECOVERY_ACTION_CONTAINER_AGITATE,
}

TRAINING_SCOPE_DENSE: str = "dense"
TRAINING_SCOPE_ALL: str = "all"
TRAINING_SCOPES: tuple[str, ...] = (TRAINING_SCOPE_DENSE, TRAINING_SCOPE_ALL)
TRAINING_SCOPES_SET: frozenset[str] = frozenset(TRAINING_SCOPES)


def _bucket_dense_for_training(mode_value: Any) -> str:
    return bucket_dense(mode_value)


def _record_in_training_scope(record: Mapping[str, Any], scope: str) -> bool:
    if scope == TRAINING_SCOPE_ALL:
        return True
    return _bucket_dense_for_training(record.get("mode")) == DENSE_BUCKET_DENSE


def _infer_failure_class(record: Mapping[str, Any]) -> str:
    """Infer the failure_class enum from canonical-pack signals.

    Canonical packs carry no ``extra.failure_class`` field, so the
    class is backfilled conservatively from ``final_outcome``. A trace
    that does set ``extra.failure_class`` short-circuits the
    heuristic.
    """

    extra = record.get("extra") or {}
    fc = extra.get("failure_class")
    if isinstance(fc, str) and fc in FAILURE_CLASS_BUCKETS:
        return fc
    final_outcome = record.get("final_outcome")
    if final_outcome == "execution_failed":
        return FAILURE_CLASS_SLIP
    if final_outcome == "no_valid_grasp":
        return FAILURE_CLASS_EMPTY_AIR
    if final_outcome == "decision_fail_closed":
        return FAILURE_CLASS_DRIFT
    if final_outcome == "recovery_exhausted":
        return FAILURE_CLASS_UNKNOWN
    # safety-rejected / collision-detected may appear in real traces.
    if isinstance(final_outcome, str) and "collision" in final_outcome:
        return FAILURE_CLASS_COLLISION
    if isinstance(final_outcome, str) and "occlusion" in final_outcome:
        return FAILURE_CLASS_OCCLUSION
    return FAILURE_CLASS_UNKNOWN


def _bucket_outcome_label(outcome: Any) -> str:
    if not isinstance(outcome, str):
        return LAST_OUTCOME_UNKNOWN
    if outcome in ("recovered_success", "success", "succeeded"):
        return LAST_OUTCOME_SUCCEEDED
    if outcome in ("recovered_failure", "failed", "failure"):
        return LAST_OUTCOME_FAILED
    if "recover" in outcome:
        return LAST_OUTCOME_RECOVERED
    return LAST_OUTCOME_UNKNOWN


def _reward_from_outcome(outcome: Any) -> float:
    if isinstance(outcome, str) and outcome in ("recovered_success",):
        return 1.0
    return 0.0


@dataclass(frozen=True)
class RecoveryTuple:
    """One (state, action, reward) training row."""

    state_key: RecoveryStateKey
    action: str
    reward: float


def extract_recovery_tuples(
    record: Mapping[str, Any],
) -> tuple[tuple[RecoveryTuple, ...], int, int]:
    """Extract per-attempt training tuples from a single replay record.

    Returns ``(tuples, n_dropped_unknown_token, n_dropped_no_actions)``.
    """

    actions = record.get("recovery_actions") or ()
    if not isinstance(actions, (list, tuple)) or len(actions) == 0:
        return ((), 0, 1)

    failure_class = _infer_failure_class(record)
    dense_bucket = _bucket_dense_for_training(record.get("mode"))
    out: list[RecoveryTuple] = []
    dropped_unknown_token = 0
    reobserve_seen = 0
    last_outcome_bucket = LAST_OUTCOME_UNKNOWN
    for i, entry in enumerate(actions):
        if not isinstance(entry, Mapping):
            dropped_unknown_token += 1
            continue
        raw = entry.get("action")
        if not isinstance(raw, str):
            dropped_unknown_token += 1
            continue
        mapped = RECOVERY_TOKEN_MAP.get(raw)
        if mapped is None:
            dropped_unknown_token += 1
            continue
        try:
            state = RecoveryStateKey(
                failure_class_bucket=failure_class,
                attempt_index_bucket=bucket_attempt_index(i),
                reobserve_count_bucket=bucket_reobserve_count(reobserve_seen),
                dense_bucket=dense_bucket,
                last_outcome_bucket=last_outcome_bucket,
            )
        except ValueError:
            dropped_unknown_token += 1
            continue
        reward = _reward_from_outcome(entry.get("outcome"))
        out.append(RecoveryTuple(state_key=state, action=mapped, reward=reward))
        if mapped == RECOVERY_ACTION_REOBSERVE:
            reobserve_seen += 1
        last_outcome_bucket = _bucket_outcome_label(entry.get("outcome"))

    return (tuple(out), dropped_unknown_token, 0)


@dataclass(frozen=True)
class RecoveryTrainResult:
    num_records_total: int
    num_records_in_scope: int
    num_records_dropped_no_actions: int
    num_tuples_extracted: int
    num_tuples_dropped_unknown_token: int
    num_per_action: Mapping[str, int]
    mean_reward_per_action: Mapping[str, float]
    num_distinct_state_keys: int
    training_scope: str
    seed: int

    def __post_init__(self) -> None:
        if set(self.num_per_action.keys()) != RECOVERY_ACTIONS_SET:
            raise ValueError("num_per_action keys must equal RECOVERY_ACTIONS")
        if set(self.mean_reward_per_action.keys()) != RECOVERY_ACTIONS_SET:
            raise ValueError(
                "mean_reward_per_action keys must equal RECOVERY_ACTIONS"
            )


def train_linucb_recovery_policy(
    records: Iterable[Mapping[str, Any]],
    *,
    training_scope: str = TRAINING_SCOPE_DENSE,
    alpha: float = DEFAULT_LINUCB_ALPHA,
    ridge_lambda: float = DEFAULT_LINUCB_LAMBDA,
    min_support_threshold: int = DEFAULT_MIN_SUPPORT_THRESHOLD,
    dataset_id: str = "",
    dataset_hash: str = "",
    seed: int = 0,
) -> tuple[LinUCBRecoveryPolicy, RecoveryTrainResult]:
    """Fit the LinUCB recovery policy."""

    if training_scope not in TRAINING_SCOPES_SET:
        raise ValueError(
            f"training_scope={training_scope!r} not in {TRAINING_SCOPES!r}"
        )
    if alpha < 0.0:
        raise ValueError("alpha must be >= 0")
    if ridge_lambda <= 0.0:
        raise ValueError("ridge_lambda must be > 0")
    if min_support_threshold < 0:
        raise ValueError("min_support_threshold must be >= 0")

    dim = onehot_dim()

    A_diag: dict[str, list[float]] = {
        a: [ridge_lambda] * dim for a in RECOVERY_ACTIONS
    }
    b_vec: dict[str, list[float]] = {a: [0.0] * dim for a in RECOVERY_ACTIONS}
    sup: dict[str, list[int]] = {a: [0] * dim for a in RECOVERY_ACTIONS}

    num_per_action: Counter[str] = Counter()
    reward_sum_per_action: dict[str, float] = {a: 0.0 for a in RECOVERY_ACTIONS}
    distinct_keys: set[str] = set()

    num_records_total = 0
    num_records_in_scope = 0
    num_records_dropped_no_actions = 0
    num_tuples_extracted = 0
    num_tuples_dropped_unknown_token = 0

    for rec in records:
        num_records_total += 1
        if not _record_in_training_scope(rec, training_scope):
            continue
        num_records_in_scope += 1
        tuples, dropped_token, dropped_no_actions = extract_recovery_tuples(rec)
        num_tuples_dropped_unknown_token += dropped_token
        num_records_dropped_no_actions += dropped_no_actions
        for t in tuples:
            num_tuples_extracted += 1
            num_per_action[t.action] += 1
            reward_sum_per_action[t.action] += t.reward
            distinct_keys.add(t.state_key.as_string_key())
            x = state_key_to_onehot(t.state_key)
            A_a = A_diag[t.action]
            b_a = b_vec[t.action]
            s_a = sup[t.action]
            for i, xi in enumerate(x):
                if xi == 0:
                    continue
                # Diagonal closed-form update: A_ii += x_i^2 (=1 for one-hot),
                # b_i += x_i * r, support_i += 1.
                A_a[i] += 1.0
                b_a[i] += t.reward
                s_a[i] += 1

    mean_reward_per_action = {
        a: (reward_sum_per_action[a] / num_per_action[a])
        if num_per_action[a] > 0
        else 0.0
        for a in RECOVERY_ACTIONS
    }

    policy = LinUCBRecoveryPolicy(
        A_diag_per_action={a: tuple(A_diag[a]) for a in RECOVERY_ACTIONS},
        b_per_action={a: tuple(b_vec[a]) for a in RECOVERY_ACTIONS},
        feature_support_per_action={a: tuple(sup[a]) for a in RECOVERY_ACTIONS},
        fallback_table=dict(DEFAULT_FALLBACK_TABLE),
        alpha=alpha,
        ridge_lambda=ridge_lambda,
        min_support_threshold=min_support_threshold,
        training_scope=DENSE_BUCKET_DENSE
        if training_scope == TRAINING_SCOPE_DENSE
        else "all",
        artifact_dataset_id=dataset_id,
        artifact_dataset_hash=dataset_hash,
    )

    result = RecoveryTrainResult(
        num_records_total=num_records_total,
        num_records_in_scope=num_records_in_scope,
        num_records_dropped_no_actions=num_records_dropped_no_actions,
        num_tuples_extracted=num_tuples_extracted,
        num_tuples_dropped_unknown_token=num_tuples_dropped_unknown_token,
        num_per_action={a: int(num_per_action[a]) for a in RECOVERY_ACTIONS},
        mean_reward_per_action=mean_reward_per_action,
        num_distinct_state_keys=len(distinct_keys),
        training_scope=training_scope,
        seed=seed,
    )
    unseen = [a for a in RECOVERY_ACTIONS if result.num_per_action[a] == 0]
    log = logger.warning if unseen else logger.info
    log(
        "Trained recovery policy on %d/%d in-scope record(s) (scope %s, dataset %r): "
        "%d tuple(s) over %d distinct state cell(s); dropped %d record(s) with no "
        "actions and %d tuple(s) with an unknown token%s",
        result.num_records_in_scope,
        result.num_records_total,
        training_scope,
        dataset_id or "?",
        result.num_tuples_extracted,
        result.num_distinct_state_keys,
        result.num_records_dropped_no_actions,
        result.num_tuples_dropped_unknown_token,
        ""
        if not unseen
        else f"; {len(unseen)} action(s) never observed ({', '.join(unseen)}) "
        "-> zero support -> the runtime gate falls back on every cell",
    )
    return policy, result


def estimate_recovery_kpis(
    records: Iterable[Mapping[str, Any]],
    policy: LinUCBRecoveryPolicy,
    training_scope: str,
) -> Mapping[str, Any]:
    """Lightweight offline KPI estimate for the artifact manifest.

    Reports the proposed-action distribution, the fallback rate and
    the rate at which a proposal matches the logged action, over the
    in-scope tuples. No hardware claim is made; this is a shadow-only
    proxy for downstream review.
    """

    proposed_counter: Counter[str] = Counter()
    fallback_used = 0
    n_tuples = 0
    agree_with_observed = 0
    for rec in records:
        if not _record_in_training_scope(rec, training_scope):
            continue
        tuples, _, _ = extract_recovery_tuples(rec)
        for t in tuples:
            sel = policy.propose_recovery(
                RecoveryRequest(attempt_id="kpi", state_key=t.state_key)
            )
            proposed_counter[sel.action] += 1
            if sel.used_fallback:
                fallback_used += 1
            if sel.action == t.action:
                agree_with_observed += 1
            n_tuples += 1
    proposed_dist = {
        a: (proposed_counter[a] / n_tuples) if n_tuples > 0 else 0.0
        for a in RECOVERY_ACTIONS
    }
    return {
        "num_tuples_evaluated": n_tuples,
        "proposed_action_distribution": proposed_dist,
        "fallback_rate": (fallback_used / n_tuples) if n_tuples > 0 else 0.0,
        "agreement_with_observed_action_rate": (
            (agree_with_observed / n_tuples) if n_tuples > 0 else 0.0
        ),
    }


def build_recovery_artifact_json(
    *,
    policy: LinUCBRecoveryPolicy,
    train_result: RecoveryTrainResult,
    seed: int,
    dataset_id: str,
    dataset_hash: str,
    offline_kpi_estimates: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical artifact dict (deterministic / json-safe)."""

    # An honest marker emitted iff the policy collapsed to about one action: one action carries every tuple
    # and the rest fall back to the hand-authored table. Absent when the fit is non-degenerate.
    degeneracy = single_action_degeneracy_note(
        train_result.num_per_action,
        fallback_rate=float(offline_kpi_estimates.get("fallback_rate", 0.0) or 0.0),
    )
    return {
        "schema_version": RECOVERY_ARTIFACT_SCHEMA_VERSION,
        "reward_model": REWARD_LOGGED_OUTCOMES,
        "reward_interpretation": RECOVERY_REWARD_INTERPRETATION,
        "dataset_provenance": make_dataset_provenance(
            input_source=INPUT_SYNTHETIC_DETERMINISTIC,
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            leakage_proven=None,
            fit_for=RECOVERY_DATASET_FIT_FOR,
            not_fit_for=RECOVERY_DATASET_NOT_FIT_FOR,
        ),
        **({"degeneracy_note": degeneracy} if degeneracy is not None else {}),
        "policy_name": policy.name,
        "policy_version": policy.version,
        "policy_kind": "recovery",
        "actions": list(RECOVERY_ACTIONS),
        "onehot_layout": list(onehot_index_layout()),
        "onehot_families": [
            {"name": name, "buckets": list(buckets)}
            for name, buckets in ONEHOT_FAMILIES
        ],
        "training_scope": policy.training_scope,
        "alpha": policy.alpha,
        "ridge_lambda": policy.ridge_lambda,
        "min_support_threshold": policy.min_support_threshold,
        "A_diag_per_action": {
            a: list(policy.A_diag_per_action[a]) for a in RECOVERY_ACTIONS
        },
        "b_per_action": {
            a: list(policy.b_per_action[a]) for a in RECOVERY_ACTIONS
        },
        "feature_support_per_action": {
            a: list(policy.feature_support_per_action[a]) for a in RECOVERY_ACTIONS
        },
        "fallback_table": {fc: policy.fallback_table[fc] for fc in FAILURE_CLASS_BUCKETS},
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash,
        "seed": seed,
        "training_metadata": {
            "action_label_source": ACTION_LABEL_SOURCE,
            "num_records_total": train_result.num_records_total,
            "num_records_in_scope": train_result.num_records_in_scope,
            "num_records_dropped_no_actions": train_result.num_records_dropped_no_actions,
            "num_tuples_extracted": train_result.num_tuples_extracted,
            "num_tuples_dropped_unknown_token": train_result.num_tuples_dropped_unknown_token,
            "num_per_action": {
                a: train_result.num_per_action[a] for a in RECOVERY_ACTIONS
            },
            "mean_reward_per_action": {
                a: train_result.mean_reward_per_action[a] for a in RECOVERY_ACTIONS
            },
            "num_distinct_state_keys": train_result.num_distinct_state_keys,
        },
        "offline_kpi_estimates": dict(offline_kpi_estimates),
    }


def _hash_pack_paths(paths: Sequence[str | Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
        with Path(p).open("rb") as f:
            for chunk in iter(cast("Callable[[], bytes]", lambda f=f: f.read(65536)), b""):
                h.update(chunk)
    return h.hexdigest()


def train_recovery_from_packs(
    *,
    pack_paths: Sequence[str | Path],
    seed: int = 0,
    training_scope: str = TRAINING_SCOPE_DENSE,
    alpha: float = DEFAULT_LINUCB_ALPHA,
    ridge_lambda: float = DEFAULT_LINUCB_LAMBDA,
    min_support_threshold: int = DEFAULT_MIN_SUPPORT_THRESHOLD,
    dataset_id: str = "",
) -> tuple[LinUCBRecoveryPolicy, RecoveryTrainResult, dict[str, Any]]:
    """Train from one or more JSONL replay packs; return (policy, result, artifact)."""

    records: list[dict[str, Any]] = []
    for p in pack_paths:
        records.extend(load_jsonl(p))
    dataset_hash = _hash_pack_paths(pack_paths)
    effective_dataset_id = dataset_id or "+".join(
        Path(p).name for p in pack_paths
    )
    policy, result = train_linucb_recovery_policy(
        records,
        training_scope=training_scope,
        alpha=alpha,
        ridge_lambda=ridge_lambda,
        min_support_threshold=min_support_threshold,
        dataset_id=effective_dataset_id,
        dataset_hash=dataset_hash,
        seed=seed,
    )
    kpis = estimate_recovery_kpis(records, policy, training_scope)
    artifact = build_recovery_artifact_json(
        policy=policy,
        train_result=result,
        seed=seed,
        dataset_id=effective_dataset_id,
        dataset_hash=dataset_hash,
        offline_kpi_estimates=kpis,
    )
    return policy, result, artifact


def write_recovery_artifact(artifact: Mapping[str, Any], path: str | Path) -> str:
    """Write artifact deterministically; return sha256 hex of the bytes written."""

    return write_artifact_json(artifact, path)


__all__ = (
    "ACTION_LABEL_SOURCE",
    "RECOVERY_TOKEN_MAP",
    "TRAINING_SCOPES",
    "TRAINING_SCOPE_ALL",
    "TRAINING_SCOPE_DENSE",
    "RecoveryTrainResult",
    "RecoveryTuple",
    "build_recovery_artifact_json",
    "estimate_recovery_kpis",
    "extract_recovery_tuples",
    "train_linucb_recovery_policy",
    "train_recovery_from_packs",
    "write_recovery_artifact",
)
