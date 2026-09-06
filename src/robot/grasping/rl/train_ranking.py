"""Offline trainer for :class:`PairwiseLogisticRankingPolicy`.

Pure-stdlib Newton-IRLS over the pairwise logistic loss. Each
training row is a delta between a success record and a fail record
drawn from the same scene/attempt group. Deterministic for a fixed
dataset and seed combination: it reuses the SPD Cholesky and ridge
choices of :mod:`train_candidate`, so the artifact regenerates
byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.robot.grasping.constants import (
    RL_TRAIN_RANKING_LOG_FILE,
    create_grasping_logger,
)

from ._artifact_io import write_artifact_json
from ._features import _executed_candidate_features
from .candidate_policy import _project_feature
from .dataset import OUTCOME_CLASS_SUCCESS, derive_outcome_class, load_jsonl
from .honesty import build_ranking_honesty
from .ranking_policy import (
    PairwiseLogisticRankingPolicy,
    RANKING_ARTIFACT_SCHEMA_VERSION,
    RANKING_FEATURE_KEYS,
)
from .train_candidate import (
    CONVERGENCE_TOL,
    MAX_NEWTON_ITERS,
    RIDGE_LAMBDA,
    _outer_add,
    _solve_spd,
)


#: Per-group pair cap. Prevents O(n^2) combinatorial blowup when a
#: scene has many records of both classes. ``8 * 8 = 64`` is the
#: largest balanced group admitted before :func:`_build_pairs`
#: sub-samples with a deterministic stride, which keeps the artifact
#: byte-identical for a given manifest.
MAX_PAIRS_PER_GROUP: int = 64


#: Logs the trained pairwise accuracy beside its baseline, at warning level when
#: the trained value does not exceed the baseline or the weights come out all
#: zero. A ranker whose features all project to zero otherwise looks exactly
#: like a trained one.
logger = create_grasping_logger("RLTrainRanking", RL_TRAIN_RANKING_LOG_FILE)


@dataclass(frozen=True)
class RankingTrainResult:
    """Typed pairwise-logistic training summary."""

    weights: tuple[float, ...]
    bias: float
    num_pairs: int
    num_groups_with_pairs: int
    final_log_loss: float
    iterations: int
    converged: bool
    ndcg_at_1: float
    pairwise_accuracy: float
    ndcg_at_1_baseline: float
    pairwise_accuracy_baseline: float


def _record_features(rec: Mapping[str, Any]) -> list[float]:
    """Project one record into the 15-key feature row.

    Highest-priority feature source first, the same precedence as
    :func:`train_candidate._build_feature_matrix`:

    1. ``record['rl_state_features']``
    2. ``record['extra']``
    3. ``_executed_candidate_features(record)``
    4. ``record`` top level.
    """

    rl_feats = rec.get("rl_state_features") or {}
    extra = rec.get("extra") or {}
    # Sim records carry the executed candidate's features in extra['rl_candidate_features'], a per-candidate
    # list, not as flat keys. Without this source the projection reads all-zeros on sim data. It sits below
    # the flat keys and is empty, and so inert, for the canonical replay packs, which carry no
    # rl_candidate_features.
    exec_feats = _executed_candidate_features(rec)
    row: list[float] = []
    for key in RANKING_FEATURE_KEYS:
        if key in rl_feats:
            row.append(_project_feature(rl_feats[key]))
        elif key in extra:
            row.append(_project_feature(extra[key]))
        elif key in exec_feats:
            row.append(_project_feature(exec_feats[key]))
        else:
            row.append(_project_feature(rec.get(key)))
    return row


def _group_key(rec: Mapping[str, Any]) -> str:
    """Best-effort scene/episode key for pair grouping.

    ``scene_family_id``, the coarse scene family of scene kind plus target identity, takes priority so
    success and fail pairs form across episodes of the same target object. The per-episode ``scene_id``
    is unique and leakage-safe, and on its own yields no pairs at all. Records without
    ``scene_family_id``, the canonical packs among them, fall through to ``scene_id``.
    """

    extra = rec.get("extra") or {}
    for k in ("scene_family_id", "scene_id", "episode_id", "session_id", "bin_id"):
        v = rec.get(k) or extra.get(k)
        if isinstance(v, str) and v:
            return f"{k}:{v}"
    aid = rec.get("attempt_id") or extra.get("attempt_id")
    if isinstance(aid, str) and aid:
        # Best-effort: strip a numeric suffix so consecutive attempts
        # of the same scene cluster together when no scene_id was
        # recorded. It never merges across distinct attempt-id
        # prefixes.
        return f"attempt_prefix:{aid.rsplit('-', 1)[0]}"
    return "_global"


def _is_success(rec: Mapping[str, Any]) -> bool:
    return derive_outcome_class(rec) == OUTCOME_CLASS_SUCCESS


def _build_pairs(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[list[float], list[float], str, str]], int]:
    """Build (success_features, fail_features, success_id, fail_id) tuples.

    Deterministic ordering: groups iterated in sorted key order;
    within a group, records sorted by ``(attempt_id, index)``. When a
    group has more than :data:`MAX_PAIRS_PER_GROUP` pairs, the
    sub-sampling is a deterministic stride.
    """

    groups: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for idx, rec in enumerate(records):
        key = _group_key(rec)
        groups.setdefault(key, []).append((idx, rec))

    pairs: list[tuple[list[float], list[float], str, str]] = []
    groups_with_pairs = 0
    for key in sorted(groups.keys()):
        members = groups[key]
        # Stable attempt-id ordering inside the group so the
        # sub-sample stride below is deterministic.
        members.sort(
            key=lambda im: (
                str(im[1].get("attempt_id") or ""),
                im[0],
            )
        )
        succ = [m for m in members if _is_success(m[1])]
        fail = [m for m in members if not _is_success(m[1])]
        if not succ or not fail:
            continue
        groups_with_pairs += 1
        local: list[tuple[list[float], list[float], str, str]] = []
        for si, srec in succ:
            for fi, frec in fail:
                local.append(
                    (
                        _record_features(srec),
                        _record_features(frec),
                        f"{key}#s{si}",
                        f"{key}#f{fi}",
                    )
                )
        if len(local) > MAX_PAIRS_PER_GROUP:
            stride = len(local) / MAX_PAIRS_PER_GROUP
            picked: list[tuple[list[float], list[float], str, str]] = []
            for j in range(MAX_PAIRS_PER_GROUP):
                picked.append(local[int(j * stride)])
            local = picked
        pairs.extend(local)
    return pairs, groups_with_pairs


def _train_pairwise_newton_irls(
    pairs: Sequence[tuple[list[float], list[float], str, str]],
) -> RankingTrainResult:
    """Pairwise logistic regression: minimize Sigma log(1+exp(-dot(w, Delta x))) + ridge.

    Solved by Newton-IRLS, closed-form per step. ``bias`` is fitted at
    zero because the bias term cancels in the pairwise delta; the slot
    stays in the artifact for shape symmetry with the logistic
    baseline.
    """

    n_pairs = len(pairs)
    if n_pairs == 0:
        raise ValueError(
            "cannot train the pairwise-logistic ranker on zero pairs"
        )
    n_features = len(RANKING_FEATURE_KEYS)
    theta = [0.0] * n_features
    prev_loss = float("inf")
    iterations = 0
    converged = False
    for it in range(MAX_NEWTON_ITERS):
        iterations = it + 1
        grad = [0.0] * n_features
        H = [[0.0] * n_features for _ in range(n_features)]
        loss = 0.0
        for spos, sneg, _sid, _fid in pairs:
            dx = [spos[j] - sneg[j] for j in range(n_features)]
            z = sum(theta[j] * dx[j] for j in range(n_features))
            # log(1 + exp(-z)); stable for large |z|.
            if z > 50.0:
                loss += 0.0  # log1p(exp(-large)) ~= 0
                p_neg = 0.0
            elif z < -50.0:
                loss += -z
                p_neg = 1.0
            else:
                loss += math.log1p(math.exp(-z))
                p_neg = 1.0 / (1.0 + math.exp(z))
            # Gradient: -p_neg * dx
            for j in range(n_features):
                grad[j] -= p_neg * dx[j]
            # Hessian: p_neg * (1-p_neg) * dx dx^T
            w = p_neg * (1.0 - p_neg)
            _outer_add(H, dx, w)
        for j in range(n_features):
            H[j][j] += RIDGE_LAMBDA
            grad[j] += RIDGE_LAMBDA * theta[j]
        loss /= n_pairs
        for j in range(n_features):
            grad[j] /= n_pairs
            for k in range(n_features):
                H[j][k] /= n_pairs
        try:
            delta = _solve_spd([row[:] for row in H], list(grad))
        except ValueError:
            break
        for j in range(n_features):
            theta[j] -= delta[j]
        if abs(prev_loss - loss) < CONVERGENCE_TOL:
            converged = True
            break
        prev_loss = loss
    weights = tuple(theta)
    bias = 0.0
    # Metrics computed on the same pair set; the ranker ships shadow-only.
    pairwise_acc = _pairwise_accuracy(pairs, weights)
    baseline_acc = 0.5
    return RankingTrainResult(
        weights=weights,
        bias=bias,
        num_pairs=n_pairs,
        num_groups_with_pairs=0,  # caller fills
        final_log_loss=float(prev_loss if converged else loss),
        iterations=iterations,
        converged=converged,
        ndcg_at_1=0.0,  # caller fills (needs grouped records)
        pairwise_accuracy=float(pairwise_acc),
        ndcg_at_1_baseline=0.0,  # caller fills
        pairwise_accuracy_baseline=float(baseline_acc),
    )


def _pairwise_accuracy(
    pairs: Sequence[tuple[list[float], list[float], str, str]],
    weights: Sequence[float],
) -> float:
    """Fraction of pairs where score(success) > score(fail)."""

    if not pairs:
        return 0.0
    correct = 0
    n_features = len(RANKING_FEATURE_KEYS)
    for spos, sneg, _sid, _fid in pairs:
        zs = sum(weights[j] * spos[j] for j in range(n_features))
        zf = sum(weights[j] * sneg[j] for j in range(n_features))
        if zs > zf:
            correct += 1
        elif zs == zf:
            correct += 0  # ties broken against the model
    return correct / float(len(pairs))


def _ndcg_at_1(
    records: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    bias: float,
) -> float:
    """Mean NDCG@1 across scene/attempt groups (relevance = 1.0 iff success).

    NDCG@1 reduces to "top-1 record's binary success label" since
    ``DCG@1 = rel_1`` and ``IDCG@1 = 1`` when at least one positive
    exists in the group.
    """

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for rec in records:
        groups.setdefault(_group_key(rec), []).append(rec)
    n_features = len(RANKING_FEATURE_KEYS)
    scores: list[float] = []
    for key in sorted(groups.keys()):
        members = groups[key]
        if len(members) < 2:
            continue
        if not any(_is_success(m) for m in members):
            continue
        scored: list[tuple[float, str, Mapping[str, Any]]] = []
        for m in members:
            feats = _record_features(m)
            z = bias + sum(weights[j] * feats[j] for j in range(n_features))
            scored.append((z, str(m.get("attempt_id") or ""), m))
        scored.sort(key=lambda r: (-r[0], r[1]))
        top1 = scored[0][2]
        scores.append(1.0 if _is_success(top1) else 0.0)
    if not scores:
        return 0.0
    return sum(scores) / float(len(scores))


def _ndcg_at_1_baseline(
    records: Sequence[Mapping[str, Any]],
) -> float:
    """Baseline NDCG@1: deterministic ordering by predicted_success_probability."""

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for rec in records:
        groups.setdefault(_group_key(rec), []).append(rec)
    scores: list[float] = []
    for key in sorted(groups.keys()):
        members = groups[key]
        if len(members) < 2:
            continue
        if not any(_is_success(m) for m in members):
            continue

        def _baseline_score(m: Mapping[str, Any]) -> float:
            extra = m.get("extra") or {}
            v = m.get("predicted_success_probability")
            if v is None:
                v = extra.get("predicted_success_probability")
            return _project_feature(v)

        ordered = sorted(
            members,
            key=lambda m: (
                -_baseline_score(m),
                str(m.get("attempt_id") or ""),
            ),
        )
        scores.append(1.0 if _is_success(ordered[0]) else 0.0)
    if not scores:
        return 0.0
    return sum(scores) / float(len(scores))


def train_pairwise_logistic_ranker(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 1,
    policy_name: str = "v3_ranking_baseline_v1",
    policy_version: int = 1,
    dataset_id: str = "",
    dataset_hash: str = "",
) -> tuple[PairwiseLogisticRankingPolicy, RankingTrainResult]:
    """Train a :class:`PairwiseLogisticRankingPolicy` on dataset records."""

    pairs, groups_with_pairs = _build_pairs(records)
    result = _train_pairwise_newton_irls(pairs)
    ndcg = _ndcg_at_1(records, result.weights, result.bias)
    ndcg_base = _ndcg_at_1_baseline(records)
    # Re-emit RankingTrainResult with the group/ndcg fields filled.
    result = RankingTrainResult(
        weights=result.weights,
        bias=result.bias,
        num_pairs=result.num_pairs,
        num_groups_with_pairs=int(groups_with_pairs),
        final_log_loss=result.final_log_loss,
        iterations=result.iterations,
        converged=result.converged,
        ndcg_at_1=float(ndcg),
        pairwise_accuracy=result.pairwise_accuracy,
        ndcg_at_1_baseline=float(ndcg_base),
        pairwise_accuracy_baseline=result.pairwise_accuracy_baseline,
    )
    policy = PairwiseLogisticRankingPolicy(
        weights=result.weights,
        bias=result.bias,
        name=policy_name,
        version=policy_version,
        artifact_dataset_id=dataset_id,
        artifact_dataset_hash=dataset_hash,
    )
    _ = seed  # Newton-IRLS is closed-form; seed only recorded for audit.
    no_lift = (
        result.pairwise_accuracy <= result.pairwise_accuracy_baseline
        or not any(w != 0.0 for w in result.weights)
    )
    log = logger.warning if (not result.converged or no_lift) else logger.info
    log(
        "Trained ranking policy %s on %d pair(s) from %d group(s) of dataset %r: "
        "log loss %.6f after %d iteration(s), converged=%s; pairwise accuracy "
        "%.4f vs baseline %.4f, NDCG@1 %.4f vs baseline %.4f",
        policy.policy_id,
        result.num_pairs,
        result.num_groups_with_pairs,
        dataset_id or "?",
        result.final_log_loss,
        result.iterations,
        result.converged,
        result.pairwise_accuracy,
        result.pairwise_accuracy_baseline,
        result.ndcg_at_1,
        result.ndcg_at_1_baseline,
    )
    return policy, result


def build_ranking_artifact_json(
    *,
    policy: PairwiseLogisticRankingPolicy,
    train_result: RankingTrainResult,
    seed: int,
    dataset_id: str,
    dataset_hash: str,
) -> dict[str, Any]:
    """Serialise a trained ranking policy into the artifact JSON shape."""

    return {
        "schema_version": RANKING_ARTIFACT_SCHEMA_VERSION,
        **build_ranking_honesty(
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            pairwise_accuracy=float(train_result.pairwise_accuracy),
            pairwise_accuracy_baseline=float(train_result.pairwise_accuracy_baseline),
        ),
        "policy_name": policy.name,
        "policy_version": policy.version,
        "policy_kind": "ranking",
        "feature_keys": list(RANKING_FEATURE_KEYS),
        "weights": [float(w) for w in policy.weights],
        "bias": float(policy.bias),
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash,
        "seed": int(seed),
        "training": {
            "algorithm": "pairwise_newton_irls_l2",
            "max_iters": MAX_NEWTON_ITERS,
            "ridge_lambda": RIDGE_LAMBDA,
            "convergence_tol": CONVERGENCE_TOL,
            "max_pairs_per_group": MAX_PAIRS_PER_GROUP,
            "iterations": int(train_result.iterations),
            "converged": bool(train_result.converged),
            "num_pairs": int(train_result.num_pairs),
            "num_groups_with_pairs": int(train_result.num_groups_with_pairs),
            "final_log_loss": float(train_result.final_log_loss),
            "ndcg_at_1": float(train_result.ndcg_at_1),
            "pairwise_accuracy": float(train_result.pairwise_accuracy),
            "ndcg_at_1_baseline": float(train_result.ndcg_at_1_baseline),
            "pairwise_accuracy_baseline": float(
                train_result.pairwise_accuracy_baseline
            ),
        },
    }


def train_ranking_from_manifest(
    *,
    manifest_path: str | Path,
    seed: int = 1,
) -> tuple[
    PairwiseLogisticRankingPolicy, RankingTrainResult, dict[str, Any]
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
    policy, result = train_pairwise_logistic_ranker(
        records,
        seed=seed,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
    )
    artifact = build_ranking_artifact_json(
        policy=policy,
        train_result=result,
        seed=seed,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
    )
    return policy, result, artifact


def write_ranking_artifact(
    artifact: Mapping[str, Any], path: str | Path
) -> str:
    """Serialise the artifact and return its sha256 digest."""

    return write_artifact_json(artifact, path)


__all__ = (
    "MAX_PAIRS_PER_GROUP",
    "RankingTrainResult",
    "build_ranking_artifact_json",
    "train_pairwise_logistic_ranker",
    "train_ranking_from_manifest",
    "write_ranking_artifact",
)
