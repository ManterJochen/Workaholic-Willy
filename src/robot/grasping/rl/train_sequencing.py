"""Offline trainer for :class:`LookupTableSequencingPolicy`.

Empirical-frequency aggregation: walk records of the train split,
group by scene/episode, derive a 4-tuple state key per record, pair
each record with its in-group successor's executed action and
did-succeed outcome, then pick for each state cell the action with
the highest empirical success rate, ties broken by the canonical
:data:`SEQUENCING_ACTIONS` order. Cells with cumulative support below
``min_support_threshold`` are not baked into the lookup table; at
inference time the lookup-table policy falls back to the hand-authored
:data:`DEFAULT_FALLBACK_TABLE` keyed on failure class.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.robot.grasping.constants import (
    RL_TRAIN_SEQUENCING_LOG_FILE,
    create_grasping_logger,
)

from ._artifact_io import write_artifact_json
from .dataset import (
    OUTCOME_CLASS_SUCCESS,
    derive_outcome_class,
    load_jsonl,
)
from .honesty import build_sequencing_honesty
from .sequencing_policy import (
    ATTEMPT_INDEX_MAX,
    COMMIT_REOBSERVE_COUNT_MAX,
    DEFAULT_FALLBACK_TABLE,
    FAILURE_CLASSES,
    FAILURE_CLASS_NONE,
    FAILURE_CLASS_UNKNOWN,
    LookupTableSequencingPolicy,
    OUTCOME_LABEL_FAILURE,
    OUTCOME_LABEL_SUCCESS,
    OUTCOME_LABELS,
    SEQUENCING_ACTIONS,
    SEQUENCING_ACTION_GRASP,
    SEQUENCING_ACTION_RECOVER,
    SEQUENCING_ACTION_REOBSERVE,
    SEQUENCING_ARTIFACT_SCHEMA_VERSION,
    SequencingStateKey,
    clip_attempt_index,
    clip_commit_reobserve_count,
)
from .train_ranking import _group_key as _ranking_group_key


#: Default minimum cell-support threshold for the production CLI.
DEFAULT_MIN_SUPPORT_THRESHOLD: int = 5


#: A committed lookup cell and an observed but under-supported one look the same
#: to every downstream consumer; the under-supported one silently falls back. The
#: log line carries both counts and warns when no cell was committed at all.
logger = create_grasping_logger("RLTrainSequencing", RL_TRAIN_SEQUENCING_LOG_FILE)


@dataclass(frozen=True)
class SequencingTrainResult:
    """Typed summary of a lookup-table training run."""

    num_records: int
    num_groups: int
    num_pairs: int
    num_cells_observed: int
    num_cells_committed: int
    num_cells_below_threshold: int
    fallback_table_dataset_classes: tuple[str, ...]


# ---------------------------------------------------------------------------
# Per-record extraction.
# ---------------------------------------------------------------------------


def _extract_outcome_label(rec: Mapping[str, Any]) -> str:
    """Map a record's terminal outcome to the binary label set."""

    cls = derive_outcome_class(rec)
    if cls == OUTCOME_CLASS_SUCCESS:
        return OUTCOME_LABEL_SUCCESS
    return OUTCOME_LABEL_FAILURE


def _extract_failure_class(rec: Mapping[str, Any]) -> str:
    """Read the failure class from a record's extras (or default).

    Success records map to :data:`FAILURE_CLASS_NONE`. Records that
    lack a classification map to :data:`FAILURE_CLASS_UNKNOWN`.
    """

    label = _extract_outcome_label(rec)
    if label == OUTCOME_LABEL_SUCCESS:
        return FAILURE_CLASS_NONE
    extra = rec.get("extra") or {}
    if not isinstance(extra, Mapping):
        extra = {}
    tax = extra.get("failure_taxonomy_class")
    if isinstance(tax, str) and tax in FAILURE_CLASSES:
        return tax
    expected = extra.get("expected_root_cause")
    if isinstance(expected, str) and expected in FAILURE_CLASSES:
        return expected
    return FAILURE_CLASS_UNKNOWN


def _extract_attempt_index(rec: Mapping[str, Any]) -> int:
    extra = rec.get("extra") or {}
    for k in ("attempt_index", "attempt_idx", "pick_attempt_index"):
        if isinstance(rec.get(k), int):
            return clip_attempt_index(int(rec[k]))
        if isinstance(extra.get(k), int):
            return clip_attempt_index(int(extra[k]))
    return 0


def _extract_commit_reobserve_count(rec: Mapping[str, Any]) -> int:
    extra = rec.get("extra") or {}
    for k in (
        "commit_reobserve_count",
        "reobserve_attempts",
        "reobserve_count",
    ):
        if isinstance(rec.get(k), int):
            return clip_commit_reobserve_count(int(rec[k]))
        if isinstance(extra.get(k), int):
            return clip_commit_reobserve_count(int(extra[k]))
    return 0


def _record_state_key(rec: Mapping[str, Any]) -> SequencingStateKey:
    return SequencingStateKey(
        last_outcome_label=_extract_outcome_label(rec),
        attempt_index_clipped=_extract_attempt_index(rec),
        commit_reobserve_count_clipped=_extract_commit_reobserve_count(rec),
        last_failure_class=_extract_failure_class(rec),
    )


def _derive_action_from_successor(next_rec: Mapping[str, Any]) -> str:
    """Map a successor record onto a sequencing action.

    A heuristic used only for empirical training; runtime never
    consults it. Success successors map to
    :data:`SEQUENCING_ACTION_GRASP`; successors carrying recovery
    evidence map to :data:`SEQUENCING_ACTION_RECOVER`; every other
    successor maps to :data:`SEQUENCING_ACTION_REOBSERVE`, the
    cheapest perception-refresh action.
    """

    if _extract_outcome_label(next_rec) == OUTCOME_LABEL_SUCCESS:
        return SEQUENCING_ACTION_GRASP
    extra = next_rec.get("extra") or {}
    if not isinstance(extra, Mapping):
        extra = {}
    if (
        extra.get("recovery_class")
        or extra.get("recovery_action")
        or extra.get("recovery_attempted") is True
    ):
        return SEQUENCING_ACTION_RECOVER
    return SEQUENCING_ACTION_REOBSERVE


def _did_succeed(rec: Mapping[str, Any]) -> bool:
    return _extract_outcome_label(rec) == OUTCOME_LABEL_SUCCESS


# ---------------------------------------------------------------------------
# Empirical aggregation.
# ---------------------------------------------------------------------------


def _aggregate_pairs(
    records: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, dict[str, int]]],
    int,
    int,
    int,
]:
    """Group records and tally ``state_key -> action -> {n, n_success}``.

    Returns ``(table, num_groups, num_pairs, num_groups_with_pairs)``.
    Only failure-side state keys are emitted; success cells are never
    proposed at runtime.
    """

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[_ranking_group_key(rec)].append(rec)

    table: dict[str, dict[str, dict[str, int]]] = {}
    num_pairs = 0
    for group_recs in groups.values():
        if len(group_recs) < 2:
            continue
        for i in range(len(group_recs) - 1):
            cur = group_recs[i]
            nxt = group_recs[i + 1]
            cur_label = _extract_outcome_label(cur)
            # Success outcomes never invoke the seam.
            if cur_label == OUTCOME_LABEL_SUCCESS:
                continue
            state_key = _record_state_key(cur).as_string_key()
            action = _derive_action_from_successor(nxt)
            cell = table.setdefault(state_key, {})
            agg = cell.setdefault(action, {"n": 0, "n_success": 0})
            agg["n"] += 1
            if _did_succeed(nxt):
                agg["n_success"] += 1
            num_pairs += 1
    return table, len(groups), num_pairs, sum(
        1 for g in groups.values() if len(g) >= 2
    )


def _select_action_for_cell(
    per_action: Mapping[str, Mapping[str, int]],
) -> tuple[str, int]:
    """Pick the highest-empirical-success action; total support."""

    best_action = ""
    best_rate = -1.0
    total = 0
    for action in SEQUENCING_ACTIONS:  # canonical order = deterministic tie-break
        agg = per_action.get(action)
        if agg is None:
            continue
        n = int(agg["n"])
        total += n
        if n == 0:
            continue
        rate = float(agg["n_success"]) / float(n)
        if rate > best_rate or (rate == best_rate and best_action == ""):
            best_action = action
            best_rate = rate
    if not best_action:
        # No labelled successor in this cell: leave it uncommitted so
        # the fallback table serves it at inference time.
        return "", total
    return best_action, total


def train_lookup_table_sequencing_policy(
    records: Sequence[Mapping[str, Any]],
    *,
    min_support_threshold: int = DEFAULT_MIN_SUPPORT_THRESHOLD,
    dataset_id: str = "",
    dataset_hash: str = "",
) -> tuple[LookupTableSequencingPolicy, SequencingTrainResult]:
    """Empirical-frequency lookup-table trainer."""

    table_raw, num_groups, num_pairs, _ = _aggregate_pairs(records)
    lookup_table: dict[str, dict[str, Any]] = {}
    num_below = 0
    for state_key in sorted(table_raw.keys()):
        per_action = table_raw[state_key]
        action, support = _select_action_for_cell(per_action)
        if not action:
            continue
        if support < min_support_threshold:
            num_below += 1
            continue
        lookup_table[state_key] = {"action": action, "support": support}
    policy = LookupTableSequencingPolicy(
        lookup_table=lookup_table,
        fallback_table=dict(DEFAULT_FALLBACK_TABLE),
        min_support_threshold=int(min_support_threshold),
        artifact_dataset_id=dataset_id,
        artifact_dataset_hash=dataset_hash,
    )
    result = SequencingTrainResult(
        num_records=len(records),
        num_groups=num_groups,
        num_pairs=num_pairs,
        num_cells_observed=len(table_raw),
        num_cells_committed=len(lookup_table),
        num_cells_below_threshold=num_below,
        fallback_table_dataset_classes=FAILURE_CLASSES,
    )
    log = logger.warning if not result.num_cells_committed else logger.info
    log(
        "Trained sequencing policy on %d record(s) from dataset %r: %d group(s), "
        "%d pair(s) -> %d/%d state cell(s) committed (%d below the support "
        "threshold %d -> hand-authored fallback)",
        result.num_records,
        dataset_id or "?",
        result.num_groups,
        result.num_pairs,
        result.num_cells_committed,
        result.num_cells_observed,
        result.num_cells_below_threshold,
        int(min_support_threshold),
    )
    return policy, result


# ---------------------------------------------------------------------------
# Artifact serialisation.
# ---------------------------------------------------------------------------


def build_sequencing_artifact_json(
    *,
    policy: LookupTableSequencingPolicy,
    train_result: SequencingTrainResult,
    seed: int,
    dataset_id: str,
    dataset_hash: str,
) -> dict[str, Any]:
    """Serialise a trained sequencing policy into the artifact JSON shape."""

    lookup_blob = {
        key: {"action": cell["action"], "support": int(cell["support"])}
        for key, cell in policy.lookup_table.items()
    }
    fallback_blob = dict(policy.fallback_table)
    return {
        "schema_version": SEQUENCING_ARTIFACT_SCHEMA_VERSION,
        **build_sequencing_honesty(
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            num_cells_committed=int(train_result.num_cells_committed),
            num_cells_below_threshold=int(train_result.num_cells_below_threshold),
        ),
        "policy_name": policy.name,
        "policy_version": policy.version,
        "policy_kind": "sequencing",
        "actions": list(SEQUENCING_ACTIONS),
        "failure_classes": list(FAILURE_CLASSES),
        "state_schema": {
            "outcome_labels": list(OUTCOME_LABELS),
            "attempt_index_max": ATTEMPT_INDEX_MAX,
            "commit_reobserve_count_max": COMMIT_REOBSERVE_COUNT_MAX,
        },
        "min_support_threshold": int(policy.min_support_threshold),
        "lookup_table": lookup_blob,
        "fallback_table": fallback_blob,
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash,
        "seed": int(seed),
        "training": {
            "algorithm": "empirical_frequency_min_support",
            "num_records": int(train_result.num_records),
            "num_groups": int(train_result.num_groups),
            "num_pairs": int(train_result.num_pairs),
            "num_cells_observed": int(train_result.num_cells_observed),
            "num_cells_committed": int(train_result.num_cells_committed),
            "num_cells_below_threshold": int(
                train_result.num_cells_below_threshold
            ),
        },
    }


def train_sequencing_from_manifest(
    *,
    manifest_path: str | Path,
    seed: int = 1,
    min_support_threshold: int = DEFAULT_MIN_SUPPORT_THRESHOLD,
) -> tuple[
    LookupTableSequencingPolicy, SequencingTrainResult, dict[str, Any]
]:
    """Train against the train split of a dataset manifest."""

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
    policy, result = train_lookup_table_sequencing_policy(
        records,
        min_support_threshold=min_support_threshold,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
    )
    artifact = build_sequencing_artifact_json(
        policy=policy,
        train_result=result,
        seed=seed,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
    )
    return policy, result, artifact


def write_sequencing_artifact(
    artifact: Mapping[str, Any], path: str | Path
) -> str:
    """Serialise the artifact and return its sha256 digest."""

    return write_artifact_json(artifact, path)


__all__ = (
    "DEFAULT_MIN_SUPPORT_THRESHOLD",
    "SequencingTrainResult",
    "build_sequencing_artifact_json",
    "train_lookup_table_sequencing_policy",
    "train_sequencing_from_manifest",
    "write_sequencing_artifact",
)
