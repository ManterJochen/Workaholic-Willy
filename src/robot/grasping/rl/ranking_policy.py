"""Ranking-policy contract + pairwise-logistic baseline.

Ships one live ranker dispatcher; LinUCB and bandit families are
deferred. The dispatcher is:

* :class:`PairwiseLogisticRankingPolicy`: deterministic, pure-stdlib,
  trained via :mod:`src.robot.grasping.rl.train_ranking` with
  Newton-IRLS on synthesised success>fail pairs.

The ranker scores each candidate via the same sigmoid surface as the
candidate policy, but over a separate 15-key feature contract (the
12 SAR keys plus 3 blend signals). The Protocol :class:`RankingPolicy`
is a distinct type from :class:`CandidatePolicy`, with no shared
structural type.

Authority lock:

* shadow only; never overrides safety, geometry or deterministic
  ranking,
* deterministic for a fixed dataset/seed/feature row,
* loader rejects schema or feature-keys drift (no silent staleness).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from src.robot.grasping.constants import (
    RL_RANKING_POLICY_LOG_FILE,
    create_grasping_logger,
)

from ._artifact_io import hash_artifact as hash_ranking_artifact
from .candidate_policy import (
    CANDIDATE_FEATURE_KEYS,
    _project_feature,
    _sigmoid,
)

#: Which artifact is loaded, and from which dataset, is the first thing to check
#: when a rerank looks wrong. The load happens once and is the only place that
#: provenance is recorded.
logger = create_grasping_logger("RLRankingPolicy", RL_RANKING_POLICY_LOG_FILE)


#: Frozen feature-key order. 12 SAR keys followed by the 3 blend
#: signals. Positional alignment with trained weights is a hard
#: contract: never reorder; bump the schema version instead.
RANKING_FEATURE_KEYS: tuple[str, ...] = CANDIDATE_FEATURE_KEYS + (
    "geometric_score",
    "feasibility_score",
    "shadow_predicted_success_probability",
)


@dataclass(frozen=True)
class RankingCandidateFeatures:
    """Per-candidate feature row supplied to the ranking policy.

    Keys must be a subset of :data:`RANKING_FEATURE_KEYS`. Missing
    values default to ``0.0`` via the SAR projection rule.
    """

    candidate_id: str
    features: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankingSelectionRequest:
    """Immutable per-attempt input to a :class:`RankingPolicy`.

    Parameters
    ----------
    attempt_id
        Stable attempt identifier, for provenance and audit only; the
        deterministic pairwise-logistic ranker does not consume it.
    candidates
        Candidate rows in deterministic baseline order (i.e. the
        order produced by the blend before the shadow ranker
        rescores).
    deterministic_ranking
        The deterministic baseline ordering as a tuple of candidate
        ids (agreement reference is the post-blend order).
        ``len(deterministic_ranking)`` must equal ``len(candidates)``;
        the constructor does not enforce it because shadow callers
        may legitimately probe with a mismatched length when the
        baseline produced fewer ids than candidates.
    """

    attempt_id: str
    candidates: tuple[RankingCandidateFeatures, ...]
    deterministic_ranking: tuple[str, ...]


@dataclass(frozen=True)
class RankedCandidate:
    """One row of the ranker's proposed ordering."""

    candidate_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class RankingSelection:
    """Immutable typed output of :meth:`RankingPolicy.propose_ranking`.

    Attributes
    ----------
    policy_id / policy_version
        Provenance for the trained artifact.
    proposals
        Ranker's full ordering, ``rank=0`` first.
    regret_top1
        ``True`` iff the ranker's top-1 candidate id differs from
        ``deterministic_ranking[0]``. ``False`` when either side
        produced no candidate (one-side-missing is treated as
        agreement-by-default so canary thresholds key off the
        explicit ``regret_top1=True`` signal).
    kendall_tau
        Full Kendall tau-b (tie correction on) computed across the
        common candidate set. Clamped to ``[-1.0, 1.0]``. ``0.0``
        when fewer than two ids are common.
    """

    policy_id: str
    policy_version: int
    proposals: tuple[RankedCandidate, ...]
    regret_top1: bool
    kendall_tau: float

    @property
    def top1_id(self) -> str | None:
        if not self.proposals:
            return None
        return self.proposals[0].candidate_id


class RankingPolicy(Protocol):
    """Frozen Protocol for ranking dispatchers.

    Implementations must be deterministic for a fixed request and
    side-effect free. Distinct from :class:`CandidatePolicy`: the
    request shape carries a ``deterministic_ranking`` tuple (not just
    a top-1 id) so the policy can compute Kendall tau-b honestly.
    """

    name: str
    version: int

    def propose_ranking(
        self, request: RankingSelectionRequest
    ) -> RankingSelection: ...


# ---------------------------------------------------------------------------
# Kendall tau-b (full tie-corrected variant).
# ---------------------------------------------------------------------------


def kendall_tau_b(
    order_a: Sequence[str], order_b: Sequence[str]
) -> float:
    """Compute Kendall tau-b between two rankings over a common id set.

    Implements the standard tau-b definition:

        tau_b = (C - D) / sqrt((n0 - n1) * (n0 - n2))

    where:

    * ``n0 = n*(n-1)/2``,
    * ``n1`` = pairs tied in ``order_a``,
    * ``n2`` = pairs tied in ``order_b``,
    * ``C`` = concordant pairs across both orderings,
    * ``D`` = discordant pairs.

    The "ranks" are derived from the index of each id in the input
    order; this is the convention used throughout the ranking telemetry
    (post-blend order gives ranks 0, 1, 2, and so on). Returns ``0.0``
    when the common set has fewer than 2 ids or when the denominator
    degenerates to zero (all ties on one side).
    """

    common = [cid for cid in order_a if cid in set(order_b)]
    n = len(common)
    if n < 2:
        return 0.0
    rank_a = {cid: i for i, cid in enumerate(order_a) if cid in common}
    rank_b = {cid: i for i, cid in enumerate(order_b) if cid in common}
    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0
    for i in range(n):
        ci = common[i]
        for j in range(i + 1, n):
            cj = common[j]
            da = rank_a[ci] - rank_a[cj]
            db = rank_b[ci] - rank_b[cj]
            if da == 0 and db == 0:
                ties_a += 1
                ties_b += 1
                continue
            if da == 0:
                ties_a += 1
                continue
            if db == 0:
                ties_b += 1
                continue
            if (da > 0 and db > 0) or (da < 0 and db < 0):
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) // 2
    denom_sq = (n0 - ties_a) * (n0 - ties_b)
    if denom_sq <= 0:
        return 0.0
    tau = (concordant - discordant) / math.sqrt(denom_sq)
    if tau < -1.0:
        return -1.0
    if tau > 1.0:
        return 1.0
    return tau


# ---------------------------------------------------------------------------
# Pairwise-logistic ranker.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairwiseLogisticRankingPolicy:
    """Deterministic sigmoid scorer aligned with :data:`RANKING_FEATURE_KEYS`.

    The model is trained with the pairwise logistic loss, one sample
    per ordered success>fail pair within an attempt; see
    :mod:`src.robot.grasping.rl.train_ranking`. At inference the model
    exposes a pointwise score per candidate, and sorting by that score
    induces the proposed ranking.

    Sorting tie-break is ``(-score, candidate_id)`` so the order is
    fully deterministic across runs and platforms.
    """

    weights: tuple[float, ...]
    bias: float
    name: str = "v3_ranking_baseline_v1"
    version: int = 1
    artifact_dataset_id: str = ""
    artifact_dataset_hash: str = ""

    def __post_init__(self) -> None:
        if len(self.weights) != len(RANKING_FEATURE_KEYS):
            raise ValueError(
                "PairwiseLogisticRankingPolicy.weights length "
                f"({len(self.weights)}) must match "
                f"RANKING_FEATURE_KEYS length "
                f"({len(RANKING_FEATURE_KEYS)})"
            )

    @property
    def policy_id(self) -> str:
        return f"{self.name}@{self.version}"

    def score_features(self, features: Mapping[str, Any]) -> float:
        z = float(self.bias)
        for key, w in zip(RANKING_FEATURE_KEYS, self.weights):
            z += w * _project_feature(features.get(key))
        return _sigmoid(z)

    def propose_ranking(
        self, request: RankingSelectionRequest
    ) -> RankingSelection:
        scored: list[tuple[float, str]] = []
        for cand in request.candidates:
            s = self.score_features(cand.features)
            scored.append((s, cand.candidate_id))
        scored.sort(key=lambda row: (-row[0], row[1]))

        proposals = tuple(
            RankedCandidate(candidate_id=cid, rank=rank, score=score)
            for rank, (score, cid) in enumerate(scored)
        )
        proposed_order = tuple(p.candidate_id for p in proposals)
        det_order = tuple(request.deterministic_ranking)
        # regret_top1 is True iff top-1 disagrees. It is False when one
        # side is missing: no regret can be claimed with nothing to compare.
        if proposals and det_order:
            regret_top1 = proposals[0].candidate_id != det_order[0]
        else:
            regret_top1 = False
        tau = kendall_tau_b(proposed_order, det_order)
        return RankingSelection(
            policy_id=self.policy_id,
            policy_version=self.version,
            proposals=proposals,
            regret_top1=bool(regret_top1),
            kendall_tau=float(tau),
        )


# ---------------------------------------------------------------------------
# Artifact loader.
# ---------------------------------------------------------------------------


#: Ranker-artifact schema version. Bump on any incompatible on-disk
#: JSON change. Artifacts at this version carry the ``reward_model``,
#: ``reward_interpretation`` and ``dataset_provenance`` honesty stamps.
RANKING_ARTIFACT_SCHEMA_VERSION: int = 2


def load_pairwise_logistic_ranking_policy(
    path: str | Path,
) -> PairwiseLogisticRankingPolicy:
    """Load a :class:`PairwiseLogisticRankingPolicy` from a JSON artifact.

    Rejects schema-version drift and feature-keys drift so a stale
    artifact can never be silently loaded.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        blob: Mapping[str, Any] = json.load(f)
    schema_version = int(blob.get("schema_version", 0))
    if schema_version != RANKING_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"PairwiseLogisticRankingPolicy artifact schema_version "
            f"{schema_version} does not match the frozen contract "
            f"{RANKING_ARTIFACT_SCHEMA_VERSION}"
        )
    feature_keys = tuple(blob["feature_keys"])
    if feature_keys != RANKING_FEATURE_KEYS:
        raise ValueError(
            "PairwiseLogisticRankingPolicy artifact feature_keys do "
            "not match the frozen contract"
        )
    weights = tuple(float(w) for w in blob["weights"])
    policy = PairwiseLogisticRankingPolicy(
        weights=weights,
        bias=float(blob["bias"]),
        name=str(blob.get("policy_name", "v3_ranking_baseline_v1")),
        version=int(blob.get("policy_version", 1)),
        artifact_dataset_id=str(blob.get("dataset_id", "")),
        artifact_dataset_hash=str(blob.get("dataset_hash", "")),
    )
    degenerate = not any(w != 0.0 for w in weights)
    log = logger.warning if degenerate else logger.info
    log(
        "Loaded ranking policy %s from %s: %d weight(s)%s, dataset %s (%s)",
        policy.policy_id,
        path,
        len(weights),
        " (all zero -> degenerate, the rerank is an identity)" if degenerate else "",
        policy.artifact_dataset_id or "?",
        policy.artifact_dataset_hash[:12] or "?",
    )
    return policy


__all__ = (
    "RANKING_ARTIFACT_SCHEMA_VERSION",
    "RANKING_FEATURE_KEYS",
    "PairwiseLogisticRankingPolicy",
    "RankedCandidate",
    "RankingCandidateFeatures",
    "RankingPolicy",
    "RankingSelection",
    "RankingSelectionRequest",
    "hash_ranking_artifact",
    "kendall_tau_b",
    "load_pairwise_logistic_ranking_policy",
)
