"""Offline trainer for :class:`LogisticCandidatePolicy`.

Pure-stdlib Newton-IRLS logistic regression over the 12 frozen SAR
state-feature keys. Deterministic for a fixed dataset and seed
combination, no external dependencies.

The Newton step is closed-form, so there is no learning rate, no
schedule and no platform-dependent SIMD difference: one dataset
yields identical artifact bytes on every CPU. ``seed`` is recorded in
the artifact as provenance; the fit itself consumes no randomness.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.robot.grasping.constants import (
    RL_TRAIN_CANDIDATE_LOG_FILE,
    create_grasping_logger,
)

from ._artifact_io import write_artifact_json
from ._features import _executed_candidate_features
from .candidate_policy import (
    CANDIDATE_FEATURE_KEYS,
    DEFAULT_PRUNE_THRESHOLD,
    LOGISTIC_ARTIFACT_SCHEMA_VERSION,
    LogisticCandidatePolicy,
    _project_feature,
    _sigmoid,
)
from .dataset import OUTCOME_CLASS_SUCCESS, derive_outcome_class, load_jsonl
from .honesty import build_candidate_honesty


#: Logs every run, at warning level when the fit did not converge or the weights
#: come out all zero. A degenerate policy is written as an ordinary artifact and
#: then abstains at runtime without announcing it.
logger = create_grasping_logger("RLTrainCandidate", RL_TRAIN_CANDIDATE_LOG_FILE)


#: Newton-IRLS convergence settings. Tight enough to give a stable
#: artifact, loose enough to terminate on degenerate datasets.
MAX_NEWTON_ITERS: int = 25
CONVERGENCE_TOL: float = 1e-7
RIDGE_LAMBDA: float = 1e-3  # tiny L2 to keep the Hessian PD


@dataclass(frozen=True)
class TrainResult:
    """Typed offline-training summary attached to the artifact JSON."""

    weights: tuple[float, ...]
    bias: float
    num_samples: int
    num_positive: int
    final_log_loss: float
    iterations: int
    converged: bool


def _build_feature_matrix(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[list[float]], list[float]]:
    """Project records into (X, y). X rows are 12-wide; y in {0.0, 1.0}.

    Feature dict source per record, highest priority first:
    1. ``record['rl_state_features']``, set when the SAR extractor
       pre-projected the record.
    2. ``record['extra']``, the telemetry bag, which carries most
       feature keys.
    3. ``_executed_candidate_features(record)``, the executed
       candidate's own features.
    4. ``record`` top level.

    Bools project to 1/0 before ints (projection rule).
    """

    X: list[list[float]] = []
    y: list[float] = []
    for rec in records:
        rl_feats = rec.get("rl_state_features") or {}
        extra = rec.get("extra") or {}
        # Sim records log the executed candidate's features in extra['rl_candidate_features'], a per-candidate
        # list, not as flat keys. This source sits below the flat keys so the projection reads real values
        # instead of all-zeros; it is empty, and so inert, for the canonical replay packs.
        exec_feats = _executed_candidate_features(rec)
        row: list[float] = []
        for key in CANDIDATE_FEATURE_KEYS:
            if key in rl_feats:
                row.append(_project_feature(rl_feats[key]))
            elif key in extra:
                row.append(_project_feature(extra[key]))
            elif key in exec_feats:
                row.append(_project_feature(exec_feats[key]))
            else:
                row.append(_project_feature(rec.get(key)))
        X.append(row)
        label = 1.0 if derive_outcome_class(rec) == OUTCOME_CLASS_SUCCESS else 0.0
        y.append(label)
    return X, y


def _matvec(A: Sequence[Sequence[float]], v: Sequence[float]) -> list[float]:
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def _outer_add(
    H: list[list[float]], x: Sequence[float], w: float
) -> None:
    n = len(x)
    for i in range(n):
        xi = x[i] * w
        for j in range(n):
            H[i][j] += xi * x[j]


def _solve_spd(
    A: list[list[float]], b: list[float]
) -> list[float]:
    """Solve A x = b via Cholesky + back-substitution (A SPD)."""

    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0.0:
                    raise ValueError("Cholesky failed: matrix not SPD")
                L[i][j] = s**0.5
            else:
                L[i][j] = s / L[j][j]
    # Solve L y = b
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    # Solve L^T x = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (
            y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))
        ) / L[i][i]
    return x


def _train_newton_irls(
    X: Sequence[Sequence[float]], y: Sequence[float]
) -> TrainResult:
    """Newton-irls logistic regression with L2 ridge."""

    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("cannot train the logistic candidate policy on zero records")
    n_features = len(CANDIDATE_FEATURE_KEYS)
    # Augment with bias column on the right.
    n_params = n_features + 1
    # theta layout: [w_0, ..., w_{n-1}, bias]
    theta = [0.0] * n_params
    prev_loss = float("inf")
    iterations = 0
    converged = False
    for it in range(MAX_NEWTON_ITERS):
        iterations = it + 1
        # Compute p_i = sigmoid(dot(x_i, w) + b)
        grad = [0.0] * n_params
        H = [[0.0] * n_params for _ in range(n_params)]
        loss = 0.0
        for i in range(n_samples):
            xi = list(X[i]) + [1.0]
            z = sum(theta[j] * xi[j] for j in range(n_params))
            p = _sigmoid(z)
            # Clamp for numerical safety inside log.
            eps = 1e-12
            p_clamped = min(max(p, eps), 1.0 - eps)
            yi_val = y[i]
            r = p_clamped - yi_val
            w = p_clamped * (1.0 - p_clamped)
            # Stable cross-entropy per sample: log(1+exp(z)) - y*z.
            if z > 50.0:
                loss += z - yi_val * z
            elif z < -50.0:
                loss += 0.0 - yi_val * z
            else:
                loss += math.log1p(math.exp(z)) - yi_val * z
            for j in range(n_params):
                grad[j] += r * xi[j]
            _outer_add(H, xi, w)
        # L2 ridge on weights only (not bias) keeps the Hessian PD
        # without biasing the intercept.
        for j in range(n_features):
            H[j][j] += RIDGE_LAMBDA
            grad[j] += RIDGE_LAMBDA * theta[j]
        # Average gradients/loss across samples for tol comparability.
        loss /= n_samples
        for j in range(n_params):
            grad[j] /= n_samples
            for k in range(n_params):
                H[j][k] /= n_samples
        # Newton step: theta -= H^{-1} grad. Solve H * delta = grad.
        try:
            delta = _solve_spd([row[:] for row in H], list(grad))
        except ValueError:
            # Singular Hessian: bail out with the current theta.
            break
        for j in range(n_params):
            theta[j] -= delta[j]
        if abs(prev_loss - loss) < CONVERGENCE_TOL:
            converged = True
            break
        prev_loss = loss

    weights = tuple(theta[:-1])
    bias = theta[-1]
    # Compute final log-loss (already averaged).
    return TrainResult(
        weights=weights,
        bias=bias,
        num_samples=n_samples,
        num_positive=int(sum(y)),
        final_log_loss=float(prev_loss if converged else loss),
        iterations=iterations,
        converged=converged,
    )


def train_logistic_candidate_policy(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 1,
    policy_name: str = "v2_candidate_baseline_v1",
    policy_version: int = 1,
    prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
    dataset_id: str = "",
    dataset_hash: str = "",
) -> tuple[LogisticCandidatePolicy, TrainResult]:
    """Train a :class:`LogisticCandidatePolicy` on dataset records.

    Returns the trained policy and the typed training summary;
    :func:`build_artifact_json` folds both into the artifact JSON.
    """

    X, y = _build_feature_matrix(records)
    result = _train_newton_irls(X, y)
    policy = LogisticCandidatePolicy(
        weights=result.weights,
        bias=result.bias,
        name=policy_name,
        version=policy_version,
        prune_threshold=prune_threshold,
        artifact_dataset_id=dataset_id,
        artifact_dataset_hash=dataset_hash,
    )
    # Newton-IRLS is closed-form and consumes no randomness, so
    # ``seed`` reaches the artifact as provenance only.
    _ = seed
    degenerate = not any(w != 0.0 for w in result.weights)
    log = (
        logger.warning if (not result.converged or degenerate) else logger.info
    )
    log(
        "Trained candidate policy %s on %d sample(s) (%d positive) from dataset %r: "
        "log loss %.6f after %d iteration(s), converged=%s%s",
        policy.policy_id,
        result.num_samples,
        result.num_positive,
        dataset_id or "?",
        result.final_log_loss,
        result.iterations,
        result.converged,
        "; all-zero weights: the policy will abstain" if degenerate else "",
    )
    return policy, result


def build_artifact_json(
    *,
    policy: LogisticCandidatePolicy,
    train_result: TrainResult,
    seed: int,
    dataset_id: str,
    dataset_hash: str,
) -> dict[str, Any]:
    """Serialise a trained policy into the artifact JSON shape."""

    return {
        "schema_version": LOGISTIC_ARTIFACT_SCHEMA_VERSION,
        **build_candidate_honesty(dataset_id=dataset_id, dataset_hash=dataset_hash),
        "policy_name": policy.name,
        "policy_version": policy.version,
        "policy_kind": "candidate_selection",
        "feature_keys": list(CANDIDATE_FEATURE_KEYS),
        "weights": [float(w) for w in policy.weights],
        "bias": float(policy.bias),
        "prune_threshold": float(policy.prune_threshold),
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash,
        "seed": int(seed),
        "training": {
            "algorithm": "newton_irls_l2",
            "max_iters": MAX_NEWTON_ITERS,
            "ridge_lambda": RIDGE_LAMBDA,
            "convergence_tol": CONVERGENCE_TOL,
            "iterations": int(train_result.iterations),
            "converged": bool(train_result.converged),
            "num_samples": int(train_result.num_samples),
            "num_positive": int(train_result.num_positive),
            "final_log_loss": float(train_result.final_log_loss),
        },
    }


def train_from_manifest(
    *,
    manifest_path: str | Path,
    seed: int = 1,
    prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
) -> tuple[LogisticCandidatePolicy, TrainResult, dict[str, Any]]:
    """Train against the train split of a dataset manifest.

    The trainer never touches the val or test splits; leakage
    protection is enforced at the API boundary, not by trainer
    convention.
    """

    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    splits = manifest["split_files"]
    # Split paths inside a manifest are relative to the repository root, and
    # the root is derived from the manifest's own location. That assumes the
    # committed layout named by ``_cli_common.COMMITTED_MANIFEST_DIR_REL``.
    repo_root = manifest_path.parent.parent.parent
    train_rel = splits["train"]["jsonl"]
    train_path = repo_root / train_rel
    if not train_path.exists():
        # Fallback: try treating split path as absolute.
        train_path = Path(train_rel)
    records = list(load_jsonl(train_path))
    dataset_id = str(manifest.get("dataset_id", ""))
    # ``sources`` is the manifest key for the input pack list. Hashing
    # the sorted (path, sha256) pairs anchors dataset identity to
    # provenance and keeps re-orderings from changing the hash.
    sources = manifest.get("sources") or []
    canonical_sources = sorted(
        [(s.get("path", ""), s.get("sha256", "")) for s in sources]
    )
    dataset_hash = hashlib.sha256(
        json.dumps(canonical_sources, sort_keys=True).encode("utf-8")
    ).hexdigest()
    policy, result = train_logistic_candidate_policy(
        records,
        seed=seed,
        prune_threshold=prune_threshold,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
    )
    artifact = build_artifact_json(
        policy=policy,
        train_result=result,
        seed=seed,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
    )
    return policy, result, artifact


def write_artifact(artifact: Mapping[str, Any], path: str | Path) -> str:
    """Serialise the artifact and return its sha256 digest."""

    return write_artifact_json(artifact, path)


__all__ = (
    "CONVERGENCE_TOL",
    "MAX_NEWTON_ITERS",
    "RIDGE_LAMBDA",
    "TrainResult",
    "build_artifact_json",
    "train_from_manifest",
    "train_logistic_candidate_policy",
    "write_artifact",
)
