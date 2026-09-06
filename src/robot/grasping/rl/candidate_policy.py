"""Candidate-selection policy contract + baselines.

Two policies ship:

* :class:`RandomCandidatePolicy`: deterministic random reorder seeded
  by ``sha256(attempt_id || policy_id)``. Serves as the control and
  smoke baseline, so the disagreement-vs-deterministic telemetry has
  a known-noise floor.
* :class:`LogisticCandidatePolicy`: deterministic logistic-regression
  scorer over the 12 frozen SAR state-feature keys. Trained offline
  by :mod:`src.robot.grasping.rl.train_candidate`, which writes the
  JSON artifact under ``docs/baselines/rl_policies/`` that this
  module loads.

Both policies receive a post-mask candidate list. Masked candidates
must never appear in the request, and the policy contract refuses to
return any masked candidate at top-1.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from src.robot.grasping.constants import (
    RL_CANDIDATE_POLICY_LOG_FILE,
    create_grasping_logger,
)

from ._artifact_io import hash_artifact as hash_logistic_artifact
from ._features import _project_feature, _sigmoid

#: Which weights a shadow run scored with, and what corpus they came from. The
#: proposal itself is per-attempt and belongs to the router's telemetry; the load
#: happens once and is the only place the provenance exists.
logger = create_grasping_logger("RLCandidatePolicy", RL_CANDIDATE_POLICY_LOG_FILE)


#: Frozen feature key order. Must match
#: ``src.robot.grasping.rl.sar.STATE_FEATURE_KEYS`` exactly:
#: :class:`LogisticCandidatePolicy` projects each candidate's feature
#: dict through this order and the trained weights are positional.
CANDIDATE_FEATURE_KEYS: tuple[str, ...] = (
    "uncertainty_score",
    "uncertainty_disagreement",
    "fused_view_count",
    "fusion_evidence_quality",
    "predicted_success_probability",
    "decision_latency_ms",
    "ranking_latency_ms",
    "fusion_latency_ms",
    "drift_severity",
    "ood_flagged",
    "degraded_mode_active",
    "multi_view_occlusion_reduced",
)

#: Default prune threshold for the logistic baseline. Candidates whose
#: predicted score is strictly below the threshold are flagged as
#: ``pruned=True`` in the policy proposal. ``0.0`` keeps every
#: candidate, so pruning is off by default: the reorder and prune
#: surface exists, and shadow telemetry characterises the score
#: distribution before any prune becomes an active decision.
DEFAULT_PRUNE_THRESHOLD: float = 0.0


@dataclass(frozen=True)
class CandidateFeatures:
    """Per-candidate feature row; keys must be a subset of :data:`CANDIDATE_FEATURE_KEYS`."""

    candidate_id: str
    features: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateSelectionRequest:
    """Immutable per-attempt policy input: the ``attempt_id`` (used for
    RNG seeding), the ordered tuple of already-unmasked ``candidates``,
    and the deterministic baseline's top-1 id (``None`` if none)."""

    attempt_id: str
    candidates: tuple[CandidateFeatures, ...]
    deterministic_top1_id: str | None = None


@dataclass(frozen=True)
class ProposedCandidate:
    """One row of the policy's proposed ordering: ``candidate_id`` at
    zero-based ``rank`` with raw ``score`` in [0, 1]; ``pruned`` is
    ``True`` iff the policy proposes dropping it before re-ranking."""

    candidate_id: str
    rank: int
    score: float
    pruned: bool = False


@dataclass(frozen=True)
class CandidateSelection:
    """Immutable typed output of :meth:`CandidatePolicy.propose`."""

    policy_id: str
    policy_version: int
    proposals: tuple[ProposedCandidate, ...]

    @property
    def top1_id(self) -> str | None:
        """Top-ranked, non-pruned candidate id (or ``None``)."""

        for p in self.proposals:
            if not p.pruned:
                return p.candidate_id
        return None


class CandidatePolicy(Protocol):
    """Frozen Protocol for candidate-selection dispatchers: deterministic for a fixed ``attempt_id``
    and side-effect free; every returned ``candidate_id`` must appear in the request."""

    name: str
    version: int

    def propose(
        self, request: CandidateSelectionRequest
    ) -> CandidateSelection: ...


@dataclass(frozen=True)
class RandomCandidatePolicy:
    """Deterministic random reorder; never prunes.

    Seeded by ``sha256(attempt_id || ':' || policy_id)`` so the noise
    floor is reproducible and RNG-isolated across processes.
    """

    name: str = "v2_random_baseline_v1"
    version: int = 1

    @property
    def policy_id(self) -> str:
        return f"{self.name}@{self.version}"

    def propose(
        self, request: CandidateSelectionRequest
    ) -> CandidateSelection:
        seed_material = f"{request.attempt_id}:{self.policy_id}".encode("utf-8")
        seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8], "big", signed=False
        )
        rng = random.Random(seed)
        order = list(request.candidates)
        rng.shuffle(order)
        proposals: list[ProposedCandidate] = []
        for rank, cand in enumerate(order):
            proposals.append(
                ProposedCandidate(
                    candidate_id=cand.candidate_id,
                    rank=rank,
                    # Uniform draws in [0, 1] for symmetry with the
                    # logistic baseline's sigmoid output. Drawn from
                    # the same deterministic RNG so re-runs are
                    # byte-identical.
                    score=rng.random(),
                    pruned=False,
                )
            )
        return CandidateSelection(
            policy_id=self.policy_id,
            policy_version=self.version,
            proposals=tuple(proposals),
        )


@dataclass(frozen=True)
class LogisticCandidatePolicy:
    """Deterministic logistic-regression scorer over the SAR features; ``weights`` is positional and
    aligned with :data:`CANDIDATE_FEATURE_KEYS`."""

    weights: tuple[float, ...]
    bias: float
    name: str = "v2_candidate_baseline_v1"
    version: int = 1
    prune_threshold: float = DEFAULT_PRUNE_THRESHOLD
    artifact_dataset_id: str = ""
    artifact_dataset_hash: str = ""

    def __post_init__(self) -> None:
        if len(self.weights) != len(CANDIDATE_FEATURE_KEYS):
            raise ValueError(
                "LogisticCandidatePolicy.weights length "
                f"({len(self.weights)}) must match "
                f"CANDIDATE_FEATURE_KEYS length "
                f"({len(CANDIDATE_FEATURE_KEYS)})"
            )
        if not (0.0 <= float(self.prune_threshold) <= 1.0):
            raise ValueError(
                "LogisticCandidatePolicy.prune_threshold must be in "
                f"[0.0, 1.0]; got {self.prune_threshold!r}"
            )

    @property
    def policy_id(self) -> str:
        return f"{self.name}@{self.version}"

    def score_features(self, features: Mapping[str, Any]) -> float:
        z = float(self.bias)
        for key, w in zip(CANDIDATE_FEATURE_KEYS, self.weights):
            z += w * _project_feature(features.get(key))
        return _sigmoid(z)

    def propose(
        self, request: CandidateSelectionRequest
    ) -> CandidateSelection:
        # Score every candidate; sort by (-score, candidate_id) for
        # ties so the ordering is fully deterministic across runs and
        # platforms.
        scored: list[tuple[float, str, CandidateFeatures]] = []
        for cand in request.candidates:
            s = self.score_features(cand.features)
            scored.append((s, cand.candidate_id, cand))
        scored.sort(key=lambda row: (-row[0], row[1]))

        proposals: list[ProposedCandidate] = []
        for rank, (score, cid, _cand) in enumerate(scored):
            proposals.append(
                ProposedCandidate(
                    candidate_id=cid,
                    rank=rank,
                    score=score,
                    pruned=(score < float(self.prune_threshold)),
                )
            )
        return CandidateSelection(
            policy_id=self.policy_id,
            policy_version=self.version,
            proposals=tuple(proposals),
        )


#: Logistic-policy artifact schema version. Bumped when the on-disk
#: JSON schema changes in a non-backward-compatible way.
LOGISTIC_ARTIFACT_SCHEMA_VERSION: int = 2


def load_logistic_candidate_policy(path: str | Path) -> LogisticCandidatePolicy:
    """Load a :class:`LogisticCandidatePolicy` from a JSON artifact.

    A ``feature_keys`` or ``schema_version`` mismatch raises
    ``ValueError``, so a stale artifact can never load silently.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        blob: Mapping[str, Any] = json.load(f)

    schema_version = int(blob.get("schema_version", 0))
    if schema_version != LOGISTIC_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"LogisticCandidatePolicy artifact schema_version "
            f"{schema_version} does not match the frozen contract "
            f"{LOGISTIC_ARTIFACT_SCHEMA_VERSION}"
        )
    feature_keys = tuple(blob["feature_keys"])
    if feature_keys != CANDIDATE_FEATURE_KEYS:
        raise ValueError(
            "LogisticCandidatePolicy artifact feature_keys do not "
            "match the frozen contract"
        )
    weights = tuple(float(w) for w in blob["weights"])
    policy = LogisticCandidatePolicy(
        weights=weights,
        bias=float(blob["bias"]),
        name=str(blob.get("policy_name", "v2_candidate_baseline_v1")),
        version=int(blob.get("policy_version", 1)),
        prune_threshold=float(
            blob.get("prune_threshold", DEFAULT_PRUNE_THRESHOLD)
        ),
        artifact_dataset_id=str(blob.get("dataset_id", "")),
        artifact_dataset_hash=str(blob.get("dataset_hash", "")),
    )
    # A committed policy can be degenerate (all-zero weights = an honest abstain);
    # that is invisible at the callsite, so it is called out at load time.
    degenerate = not any(w != 0.0 for w in weights)
    log = logger.warning if degenerate else logger.info
    log(
        "Loaded candidate policy %s from %s: %d weight(s)%s, prune_threshold %.3f, "
        "dataset %s (%s)",
        policy.policy_id,
        path,
        len(weights),
        " (all zero -> degenerate, the policy abstains)" if degenerate else "",
        policy.prune_threshold,
        policy.artifact_dataset_id or "?",
        policy.artifact_dataset_hash[:12] or "?",
    )
    return policy


__all__ = (
    "CANDIDATE_FEATURE_KEYS",
    "DEFAULT_PRUNE_THRESHOLD",
    "LOGISTIC_ARTIFACT_SCHEMA_VERSION",
    "CandidateFeatures",
    "CandidatePolicy",
    "CandidateSelection",
    "CandidateSelectionRequest",
    "LogisticCandidatePolicy",
    "ProposedCandidate",
    "RandomCandidatePolicy",
    "hash_logistic_artifact",
    "load_logistic_candidate_policy",
)
