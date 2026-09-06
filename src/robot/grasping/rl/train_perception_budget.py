"""Offline trainer for :class:`LinUCBPerceptionBudgetPolicy`.

Walks the records of a canonical replay pack, or the train split of
a dataset manifest, derives a 4-tuple state key and a binary action
label per record, pairs each record with a reward under
:data:`OPE_REWARD_COEFFICIENTS_V1`, and aggregates the resulting
``(context, action, reward)`` tuples into the per-action LinUCB
diagonal sufficient statistics ``A_a = lambda I + Sigma x x^T``, ``b_a = Sigma r * x``.

Because every context vector is one-hot, ``A_a`` is diagonal and the
posterior closed-form has no matrix inversion (just per-entry
``theta_i = b_i / A_i``). Pure stdlib, deterministic, exact.

Action label inference:
    The canonical packs do not log per-view stop or continue
    decisions, so the per-attempt aggregate label comes from
    ``extra.fusion_latency_ms``:

    * ``fusion_latency_ms`` present: the attempt took the multi-view
      path, so the label is :data:`PERCEPTION_ACTION_CONTINUE`.
    * ``fusion_latency_ms`` absent: the attempt committed on the
      single-view path, so the label is
      :data:`PERCEPTION_ACTION_STOP`.

    The artifact records this assumption in
    ``training_metadata.action_label_source`` so downstream auditors
    can verify the bias.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.robot.grasping.constants import (
    RL_TRAIN_PERCEPTION_BUDGET_LOG_FILE,
    create_grasping_logger,
)

from ._artifact_io import write_artifact_json
from .dataset import load_jsonl
from .honesty import build_perception_honesty
from .ope import (
    OPE_REWARD_COEFFICIENTS_V1,
    compute_record_reward,
)
from .perception_budget_policy import (
    DEFAULT_FALLBACK_TABLE,
    DEFAULT_LINUCB_ALPHA,
    DEFAULT_LINUCB_LAMBDA,
    DEFAULT_MIN_SUPPORT_THRESHOLD,
    LinUCBPerceptionBudgetPolicy,
    MODE_BUCKET_DENSE,
    PERCEPTION_ACTIONS,
    PERCEPTION_ACTION_CONTINUE,
    PERCEPTION_ACTION_STOP,
    PERCEPTION_BUDGET_ARTIFACT_SCHEMA_VERSION,
    PerceptionBudgetStateKey,
    bucket_candidate_count,
    bucket_mode,
    bucket_occlusion,
    bucket_views_seen,
    onehot_dim,
    onehot_index_layout,
    state_key_to_onehot,
)

#: This trainer drops records for two different reasons, out of dense scope and
#: no cycle time, and a run that kept almost nothing still produces a valid
#: artifact. The log line carries the kept and dropped counts and warns when
#: nothing was kept or only one arm was observed.
logger = create_grasping_logger(
    "RLTrainPerceptionBudget", RL_TRAIN_PERCEPTION_BUDGET_LOG_FILE
)


#: Value written to ``training_metadata.action_label_source``. It names how
#: the per-record action label is inferred; the module docstring states it.
ACTION_LABEL_SOURCE_FUSION_LATENCY: str = "fusion_latency_ms_presence"


@dataclass(frozen=True)
class PerceptionBudgetTrainResult:
    """Typed summary of a LinUCB training run."""

    num_records: int
    num_records_kept: int
    num_records_dropped_dense_scope: int
    num_records_dropped_missing_cycle_time: int
    num_action_stop: int
    num_action_continue: int
    num_cells_observed: int
    mean_reward_stop: float
    mean_reward_continue: float


# ---------------------------------------------------------------------------
# Per-record extraction.
# ---------------------------------------------------------------------------


def _extract_record_mode(rec: Mapping[str, Any]) -> str:
    """Read the mode token from a record (top-level or extras)."""

    top = rec.get("mode")
    if isinstance(top, str) and top:
        return top
    extra = rec.get("extra") or {}
    if isinstance(extra, Mapping):
        em = extra.get("mode")
        if isinstance(em, str) and em:
            return em
    return ""


def extract_perception_state(
    rec: Mapping[str, Any],
) -> PerceptionBudgetStateKey:
    """Derive the 4-tuple state key for a per-attempt training record.

    Deterministic and total: missing fields collapse to safe defaults
    (``views_seen=0``, ``candidate_count=0``, ``occlusion=unknown``,
    ``mode=unknown``).
    """

    extra = rec.get("extra") or {}
    if not isinstance(extra, Mapping):
        extra = {}
    views = extra.get("views_seen")
    if not isinstance(views, int):
        views = 0
    cand = extra.get("candidate_count")
    if not isinstance(cand, int):
        cand = 0
    occ = extra.get("expected_occlusion_ratio")
    if occ is None:
        occ = extra.get("occlusion_ratio")
    mode_raw = _extract_record_mode(rec)
    return PerceptionBudgetStateKey(
        views_seen_bucket=bucket_views_seen(views),
        last_candidate_count_bucket=bucket_candidate_count(cand),
        last_occlusion_bucket=bucket_occlusion(occ),
        mode_bucket=bucket_mode(mode_raw),
    )


def extract_action_label(rec: Mapping[str, Any]) -> str:
    """Synthesise the per-attempt aggregate stop/continue label.

    See module docstring for the rationale. The label source is
    documented in the artifact training metadata.
    """

    extra = rec.get("extra") or {}
    if not isinstance(extra, Mapping):
        return PERCEPTION_ACTION_STOP
    if "fusion_latency_ms" in extra and extra["fusion_latency_ms"] is not None:
        return PERCEPTION_ACTION_CONTINUE
    return PERCEPTION_ACTION_STOP


def _is_in_training_scope(rec: Mapping[str, Any], scope: str) -> bool:
    """Keep only dense-mode records when ``scope == "dense"``."""

    if scope == "unknown":
        return True  # the wildcard scope accepts every record
    mode = bucket_mode(_extract_record_mode(rec))
    return mode == scope


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------


def train_linucb_perception_budget_policy(
    records: Sequence[Mapping[str, Any]],
    *,
    training_scope: str = MODE_BUCKET_DENSE,
    alpha: float = DEFAULT_LINUCB_ALPHA,
    ridge_lambda: float = DEFAULT_LINUCB_LAMBDA,
    min_support_threshold: int = DEFAULT_MIN_SUPPORT_THRESHOLD,
    dataset_id: str = "",
    dataset_hash: str = "",
) -> tuple[LinUCBPerceptionBudgetPolicy, PerceptionBudgetTrainResult]:
    """Fit per-action diagonal LinUCB sufficient statistics on ``records``.

    Records outside ``training_scope`` are dropped. Records that
    cannot produce a reward, meaning ``extra.cycle_time_s`` is
    missing, are also dropped. Both counts reach the result.
    """

    d = onehot_dim()
    A_per_action: dict[str, list[float]] = {
        a: [float(ridge_lambda)] * d for a in PERCEPTION_ACTIONS
    }
    b_per_action: dict[str, list[float]] = {
        a: [0.0] * d for a in PERCEPTION_ACTIONS
    }
    sup_per_action: dict[str, list[int]] = {
        a: [0] * d for a in PERCEPTION_ACTIONS
    }

    cells_observed: set[str] = set()
    n_stop = 0
    n_cont = 0
    sum_reward_stop = 0.0
    sum_reward_cont = 0.0
    dropped_scope = 0
    dropped_missing_cycle = 0
    kept = 0

    for rec in records:
        if not _is_in_training_scope(rec, training_scope):
            dropped_scope += 1
            continue
        extra = rec.get("extra") or {}
        if not isinstance(extra, Mapping) or "cycle_time_s" not in extra:
            dropped_missing_cycle += 1
            continue
        try:
            reward = compute_record_reward(rec)
        except Exception:  # noqa: BLE001 (strict reward eval, drop malformed)
            dropped_missing_cycle += 1
            continue
        state = extract_perception_state(rec)
        action = extract_action_label(rec)
        onehot = state_key_to_onehot(state)
        cells_observed.add(state.as_string_key())
        kept += 1
        if action == PERCEPTION_ACTION_STOP:
            n_stop += 1
            sum_reward_stop += reward
        else:
            n_cont += 1
            sum_reward_cont += reward
        A = A_per_action[action]
        b = b_per_action[action]
        sup = sup_per_action[action]
        for i, xi in enumerate(onehot):
            if xi == 0:
                continue
            A[i] += 1.0  # xi == 1, so xi * xi == 1
            b[i] += reward
            sup[i] += 1

    policy = LinUCBPerceptionBudgetPolicy(
        A_diag_per_action={a: tuple(v) for a, v in A_per_action.items()},
        b_per_action={a: tuple(v) for a, v in b_per_action.items()},
        feature_support_per_action={
            a: tuple(v) for a, v in sup_per_action.items()
        },
        fallback_table=dict(DEFAULT_FALLBACK_TABLE),
        alpha=alpha,
        ridge_lambda=ridge_lambda,
        min_support_threshold=int(min_support_threshold),
        training_scope=training_scope,
        artifact_dataset_id=dataset_id,
        artifact_dataset_hash=dataset_hash,
    )
    mean_stop = sum_reward_stop / float(n_stop) if n_stop > 0 else 0.0
    mean_cont = sum_reward_cont / float(n_cont) if n_cont > 0 else 0.0
    result = PerceptionBudgetTrainResult(
        num_records=len(records),
        num_records_kept=kept,
        num_records_dropped_dense_scope=dropped_scope,
        num_records_dropped_missing_cycle_time=dropped_missing_cycle,
        num_action_stop=n_stop,
        num_action_continue=n_cont,
        num_cells_observed=len(cells_observed),
        mean_reward_stop=mean_stop,
        mean_reward_continue=mean_cont,
    )
    single_armed = result.num_action_stop == 0 or result.num_action_continue == 0
    log = logger.warning if (not result.num_records_kept or single_armed) else logger.info
    log(
        "Trained perception-budget policy on %d/%d record(s) (scope %s, dataset %r): "
        "dropped %d out of scope and %d without a cycle time; %d stop / %d continue "
        "over %d state cell(s); mean reward stop %.4f, continue %.4f%s",
        result.num_records_kept,
        result.num_records,
        training_scope,
        dataset_id or "?",
        result.num_records_dropped_dense_scope,
        result.num_records_dropped_missing_cycle_time,
        result.num_action_stop,
        result.num_action_continue,
        result.num_cells_observed,
        result.mean_reward_stop,
        result.mean_reward_continue,
        "; one arm never observed: no comparison is possible" if single_armed else "",
    )
    return policy, result


# ---------------------------------------------------------------------------
# Offline KPI estimator (perception-cost reduction vs baseline).
# ---------------------------------------------------------------------------


def estimate_perception_budget_kpis(
    records: Sequence[Mapping[str, Any]],
    policy: LinUCBPerceptionBudgetPolicy,
    *,
    training_scope: str = MODE_BUCKET_DENSE,
) -> dict[str, Any]:
    """Offline KPI estimator.

    Walks the in-scope records, runs the policy on each record's
    derived state, and computes:

    * ``baseline_mean_perception_cost``: mean ``extra.perception_cost``
      as logged. A value that is missing, non-numeric or exactly 0.0
      is replaced by the synthetic proxy, 2.0 when
      ``fusion_latency_ms`` is present and 1.0 otherwise, so a real
      zero cost cannot be told apart from a missing one.
    * ``baseline_mean_reward``: mean reward under the as-logged
      behaviour.
    * ``policy_proposed_stop_rate``: fraction of in-scope records the
      policy would have stopped at view 0, the only synthetic state
      available here.
    * ``policy_proposed_continue_rate``.
    * ``estimated_perception_cost_reduction_pct``: a proposed
      :data:`PERCEPTION_ACTION_STOP` costs 1.0, one view; a proposed
      :data:`PERCEPTION_ACTION_CONTINUE` costs the as-logged value,
      because counterfactual multi-view cost cannot be simulated.
      The estimate is a lower bound on savings under the synthetic
      perception_cost model.

    All numbers are emitted with full float precision and are
    surfaced in the artifact's ``offline_kpi_estimates`` block.
    """

    in_scope = [
        rec for rec in records if _is_in_training_scope(rec, training_scope)
    ]
    if not in_scope:
        return {
            "in_scope_records": 0,
            "baseline_mean_perception_cost": 0.0,
            "baseline_mean_reward": 0.0,
            "policy_proposed_stop_rate": 0.0,
            "policy_proposed_continue_rate": 0.0,
            "estimated_perception_cost_reduction_pct": 0.0,
        }

    from .perception_budget_policy import PerceptionBudgetRequest  # avoid cycle

    sum_baseline_cost = 0.0
    sum_baseline_reward = 0.0
    n_stop = 0
    n_continue = 0
    sum_policy_cost = 0.0
    n = 0

    for rec in in_scope:
        extra = rec.get("extra") or {}
        if not isinstance(extra, Mapping) or "cycle_time_s" not in extra:
            continue
        try:
            reward = compute_record_reward(rec)
        except Exception:  # noqa: BLE001 (drop malformed for KPI estimate)
            continue
        n += 1
        baseline_cost = extra.get("perception_cost", 0.0)
        if not isinstance(baseline_cost, (int, float)):
            baseline_cost = 0.0
        # A perception_cost that is missing, non-numeric or exactly
        # 0.0 falls back to the synthetic proxy: 2.0 if multi-view
        # path (fusion_latency_ms present) else 1.0.
        if baseline_cost == 0.0:
            baseline_cost = (
                2.0
                if "fusion_latency_ms" in extra
                and extra["fusion_latency_ms"] is not None
                else 1.0
            )
        sum_baseline_cost += float(baseline_cost)
        sum_baseline_reward += reward
        state = extract_perception_state(rec)
        selection = policy.propose_perception_budget(
            PerceptionBudgetRequest(
                attempt_id=str(rec.get("attempt_id", "")),
                state_key=state,
            )
        )
        if selection.action == PERCEPTION_ACTION_STOP:
            n_stop += 1
            sum_policy_cost += 1.0
        else:
            n_continue += 1
            sum_policy_cost += float(baseline_cost)

    if n == 0:
        return {
            "in_scope_records": 0,
            "baseline_mean_perception_cost": 0.0,
            "baseline_mean_reward": 0.0,
            "policy_proposed_stop_rate": 0.0,
            "policy_proposed_continue_rate": 0.0,
            "estimated_perception_cost_reduction_pct": 0.0,
        }
    baseline_mean_cost = sum_baseline_cost / float(n)
    policy_mean_cost = sum_policy_cost / float(n)
    if baseline_mean_cost > 0.0:
        reduction_pct = (
            100.0 * (baseline_mean_cost - policy_mean_cost) / baseline_mean_cost
        )
    else:
        reduction_pct = 0.0
    return {
        "in_scope_records": int(n),
        "baseline_mean_perception_cost": float(baseline_mean_cost),
        "baseline_mean_reward": float(sum_baseline_reward / float(n)),
        "policy_proposed_stop_rate": float(n_stop) / float(n),
        "policy_proposed_continue_rate": float(n_continue) / float(n),
        "estimated_perception_cost_reduction_pct": float(reduction_pct),
    }


# ---------------------------------------------------------------------------
# Artifact serialisation.
# ---------------------------------------------------------------------------


def build_perception_budget_artifact_json(
    *,
    policy: LinUCBPerceptionBudgetPolicy,
    train_result: PerceptionBudgetTrainResult,
    seed: int,
    dataset_id: str,
    dataset_hash: str,
    offline_kpi_estimates: Mapping[str, Any],
    action_label_source: str = ACTION_LABEL_SOURCE_FUSION_LATENCY,
) -> dict[str, Any]:
    """Serialise a trained perception-budget policy into JSON shape."""

    A_blob = {a: list(v) for a, v in policy.A_diag_per_action.items()}
    b_blob = {a: list(v) for a, v in policy.b_per_action.items()}
    sup_blob = {a: list(v) for a, v in policy.feature_support_per_action.items()}
    fallback_blob = dict(policy.fallback_table)
    return {
        "schema_version": PERCEPTION_BUDGET_ARTIFACT_SCHEMA_VERSION,
        **build_perception_honesty(
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            num_records_kept=int(train_result.num_records_kept),
            # Surface a degeneracy_note when the training collapsed to about one action, so a reader
            # cannot mistake a single-action LinUCB fit for a meaningfully trained stop or continue
            # policy.
            num_per_action={
                PERCEPTION_ACTION_STOP: int(train_result.num_action_stop),
                PERCEPTION_ACTION_CONTINUE: int(train_result.num_action_continue),
            },
        ),
        "policy_name": policy.name,
        "policy_version": policy.version,
        "policy_kind": "perception_budget",
        "actions": list(PERCEPTION_ACTIONS),
        "onehot_layout": list(onehot_index_layout()),
        "training_scope": policy.training_scope,
        "alpha": float(policy.alpha),
        "ridge_lambda": float(policy.ridge_lambda),
        "min_support_threshold": int(policy.min_support_threshold),
        "A_diag_per_action": A_blob,
        "b_per_action": b_blob,
        "feature_support_per_action": sup_blob,
        "fallback_table": fallback_blob,
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash,
        "seed": int(seed),
        "training_metadata": {
            "algorithm": "linucb_diagonal_onehot",
            "action_label_source": action_label_source,
            "reward_coefficients": dict(OPE_REWARD_COEFFICIENTS_V1),
            "num_records": int(train_result.num_records),
            "num_records_kept": int(train_result.num_records_kept),
            "num_records_dropped_dense_scope": int(
                train_result.num_records_dropped_dense_scope
            ),
            "num_records_dropped_missing_cycle_time": int(
                train_result.num_records_dropped_missing_cycle_time
            ),
            "num_action_stop": int(train_result.num_action_stop),
            "num_action_continue": int(train_result.num_action_continue),
            "num_cells_observed": int(train_result.num_cells_observed),
            "mean_reward_stop": float(train_result.mean_reward_stop),
            "mean_reward_continue": float(train_result.mean_reward_continue),
        },
        "offline_kpi_estimates": dict(offline_kpi_estimates),
    }


def train_perception_budget_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 1,
    training_scope: str = MODE_BUCKET_DENSE,
    alpha: float = DEFAULT_LINUCB_ALPHA,
    ridge_lambda: float = DEFAULT_LINUCB_LAMBDA,
    min_support_threshold: int = DEFAULT_MIN_SUPPORT_THRESHOLD,
    dataset_id: str = "",
    dataset_hash: str = "",
) -> tuple[
    LinUCBPerceptionBudgetPolicy,
    PerceptionBudgetTrainResult,
    dict[str, Any],
]:
    """Train from an in-memory record sequence."""

    policy, result = train_linucb_perception_budget_policy(
        records,
        training_scope=training_scope,
        alpha=alpha,
        ridge_lambda=ridge_lambda,
        min_support_threshold=min_support_threshold,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
    )
    kpis = estimate_perception_budget_kpis(
        records, policy, training_scope=training_scope
    )
    artifact = build_perception_budget_artifact_json(
        policy=policy,
        train_result=result,
        seed=seed,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
        offline_kpi_estimates=kpis,
    )
    return policy, result, artifact


def train_perception_budget_from_manifest(
    *,
    manifest_path: str | Path,
    seed: int = 1,
    training_scope: str = MODE_BUCKET_DENSE,
    alpha: float = DEFAULT_LINUCB_ALPHA,
    ridge_lambda: float = DEFAULT_LINUCB_LAMBDA,
    min_support_threshold: int = DEFAULT_MIN_SUPPORT_THRESHOLD,
) -> tuple[
    LinUCBPerceptionBudgetPolicy,
    PerceptionBudgetTrainResult,
    dict[str, Any],
]:
    """Train from a dataset manifest's train split."""

    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    splits = manifest["split_files"]
    repo_root = manifest_path.parent.parent.parent
    train_rel = splits["train"]["jsonl"]
    train_path = repo_root / train_rel
    if not train_path.exists():
        train_path = Path(train_rel)
    records = list(load_jsonl(train_path))
    dataset_id = str(manifest.get("dataset_id", ""))
    sources = manifest.get("sources") or []
    canonical_sources = sorted(
        [(s.get("path", ""), s.get("sha256", "")) for s in sources]
    )
    dataset_hash = hashlib.sha256(
        json.dumps(canonical_sources, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return train_perception_budget_from_records(
        records,
        seed=seed,
        training_scope=training_scope,
        alpha=alpha,
        ridge_lambda=ridge_lambda,
        min_support_threshold=min_support_threshold,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
    )


def train_perception_budget_from_packs(
    *,
    pack_paths: Sequence[str | Path],
    seed: int = 1,
    training_scope: str = MODE_BUCKET_DENSE,
    alpha: float = DEFAULT_LINUCB_ALPHA,
    ridge_lambda: float = DEFAULT_LINUCB_LAMBDA,
    min_support_threshold: int = DEFAULT_MIN_SUPPORT_THRESHOLD,
) -> tuple[
    LinUCBPerceptionBudgetPolicy,
    PerceptionBudgetTrainResult,
    dict[str, Any],
]:
    """Train from one or more JSONL replay packs."""

    all_records: list[Mapping[str, Any]] = []
    source_entries: list[tuple[str, str]] = []
    for p in pack_paths:
        pp = Path(p)
        recs = list(load_jsonl(pp))
        all_records.extend(recs)
        h = hashlib.sha256()
        with pp.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        source_entries.append((str(pp), h.hexdigest()))
    source_entries.sort()
    dataset_hash = hashlib.sha256(
        json.dumps(source_entries, sort_keys=True).encode("utf-8")
    ).hexdigest()
    dataset_id = "+".join(Path(p).name for p in pack_paths)
    return train_perception_budget_from_records(
        all_records,
        seed=seed,
        training_scope=training_scope,
        alpha=alpha,
        ridge_lambda=ridge_lambda,
        min_support_threshold=min_support_threshold,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
    )


def write_perception_budget_artifact(
    artifact: Mapping[str, Any], path: str | Path
) -> str:
    """Serialise the artifact and return its sha256 digest."""

    return write_artifact_json(artifact, path)


__all__ = (
    "ACTION_LABEL_SOURCE_FUSION_LATENCY",
    "PerceptionBudgetTrainResult",
    "build_perception_budget_artifact_json",
    "estimate_perception_budget_kpis",
    "extract_action_label",
    "extract_perception_state",
    "train_linucb_perception_budget_policy",
    "train_perception_budget_from_manifest",
    "train_perception_budget_from_packs",
    "train_perception_budget_from_records",
    "write_perception_budget_artifact",
)
