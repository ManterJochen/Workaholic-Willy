"""Active-perception budget optimizer (shadow-only).

Ships the perception-budget policy contract and a deterministic
LinUCB contextual-bandit baseline that decides, per attempt, whether
the runtime should ``STOP`` (commit and proceed to grasp with current
views) or ``CONTINUE`` (request another viewpoint from
:class:`ScoringViewpointPlanner`).

Authority locks (identical to the other shadow policies):

* shadow only; never overrides the deterministic active-perception
  planner, and the runtime keeps its existing view budget,
* deterministic for a fixed state key + artifact,
* loader rejects schema-version drift (no silent staleness),
* fail-closed: when the cell support is below threshold or the
  artifact is missing, the policy returns a hand-authored fallback
  action (``CONTINUE`` if ``views_seen < 2`` else ``STOP``).

Design:

* Shadow wire-in at the router seam; the replay-record ``extra.*``
  schema is not extended.
* LinUCB contextual bandit. Discrete-context (one-hot) form keeps the
  ``A_a`` matrices diagonal, so the pure-stdlib closed-form inversion
  stays exact and no numpy dependency is needed.
* 4-tuple state key ``(views_seen_bucket, last_candidate_count_bucket,
  last_occlusion_bucket, mode_bucket)``.
* Binary action set ``{STOP, CONTINUE}``.
* Reward = ``OPE_REWARD_COEFFICIENTS_V1``.
* The artifact surfaces ``training_scope`` (default ``"dense"``) so
  the router can refuse a wrong-scope load.
* Fallback: ``CONTINUE`` if ``views_seen < 2`` else ``STOP``.
* Per-cell minimum support = 5.
* No record-extras / catalog additions; the policy's runtime output
  lives only in the typed
  :class:`src.robot.grasping.rl.router.PerceptionShadowTelemetry`
  carrier.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from src.robot.grasping.constants import (
    RL_PERCEPTION_BUDGET_POLICY_LOG_FILE,
    create_grasping_logger,
)

from ._artifact_io import hash_artifact as hash_perception_budget_artifact
from ._linucb import score_linucb_action

#: This policy spends cycle time (an extra view instead of committing now), so
#: the artifact and the support threshold in force are recorded once, at load.
logger = create_grasping_logger(
    "RLPerceptionBudgetPolicy", RL_PERCEPTION_BUDGET_POLICY_LOG_FILE
)


# ---------------------------------------------------------------------------
# Action enum (binary stop / continue).
# ---------------------------------------------------------------------------

PERCEPTION_ACTION_STOP: str = "stop"
PERCEPTION_ACTION_CONTINUE: str = "continue"

#: Deterministic ordering of the action enum. The artifact's
#: ``actions`` block must match this order; never reorder.
PERCEPTION_ACTIONS: tuple[str, ...] = (
    PERCEPTION_ACTION_STOP,
    PERCEPTION_ACTION_CONTINUE,
)

#: Frozen set form for membership checks.
PERCEPTION_ACTIONS_SET: frozenset[str] = frozenset(PERCEPTION_ACTIONS)


# ---------------------------------------------------------------------------
# Mode-bucket enum (4th state dimension).
# ---------------------------------------------------------------------------

MODE_BUCKET_EASY: str = "easy"
MODE_BUCKET_AUTO: str = "auto"
MODE_BUCKET_DENSE: str = "dense"
MODE_BUCKET_UNKNOWN: str = "unknown"

MODE_BUCKETS: tuple[str, ...] = (
    MODE_BUCKET_EASY,
    MODE_BUCKET_AUTO,
    MODE_BUCKET_DENSE,
    MODE_BUCKET_UNKNOWN,
)
MODE_BUCKETS_SET: frozenset[str] = frozenset(MODE_BUCKETS)


def bucket_mode(value: Any) -> str:
    """Map a raw mode string to one of :data:`MODE_BUCKETS`.

    ``None``, missing, or unrecognised inputs collapse to
    :data:`MODE_BUCKET_UNKNOWN`; the mapping is deterministic and total.
    """

    if not isinstance(value, str):
        return MODE_BUCKET_UNKNOWN
    v = value.strip().lower()
    # Direct match.
    if v in MODE_BUCKETS_SET and v != MODE_BUCKET_UNKNOWN:
        return v
    # Prefix-based mapping for canonical pack mode strings
    # (``"dense_clutter"`` becomes ``"dense"``, ``"easy_canonical"``
    # becomes ``"easy"``). Deterministic and total.
    if v.startswith("easy"):
        return MODE_BUCKET_EASY
    if v.startswith("auto"):
        return MODE_BUCKET_AUTO
    if v.startswith("dense"):
        return MODE_BUCKET_DENSE
    return MODE_BUCKET_UNKNOWN


# ---------------------------------------------------------------------------
# Views-seen bucket.
# ---------------------------------------------------------------------------

VIEWS_SEEN_BUCKET_0: str = "0"
VIEWS_SEEN_BUCKET_1: str = "1"
VIEWS_SEEN_BUCKET_2: str = "2"
VIEWS_SEEN_BUCKET_3P: str = "3+"

VIEWS_SEEN_BUCKETS: tuple[str, ...] = (
    VIEWS_SEEN_BUCKET_0,
    VIEWS_SEEN_BUCKET_1,
    VIEWS_SEEN_BUCKET_2,
    VIEWS_SEEN_BUCKET_3P,
)
VIEWS_SEEN_BUCKETS_SET: frozenset[str] = frozenset(VIEWS_SEEN_BUCKETS)


def bucket_views_seen(value: Any) -> str:
    """Map an integer view-count to one of :data:`VIEWS_SEEN_BUCKETS`.

    Negative / non-integer inputs collapse to ``"0"`` (deterministic
    and total; never raises).
    """

    if not isinstance(value, int) or isinstance(value, bool):
        # ``bool`` is an ``int`` subclass, so guard explicitly.
        return VIEWS_SEEN_BUCKET_0
    if value <= 0:
        return VIEWS_SEEN_BUCKET_0
    if value == 1:
        return VIEWS_SEEN_BUCKET_1
    if value == 2:
        return VIEWS_SEEN_BUCKET_2
    return VIEWS_SEEN_BUCKET_3P


# ---------------------------------------------------------------------------
# Candidate-count bucket.
# ---------------------------------------------------------------------------

CANDIDATE_COUNT_BUCKET_0: str = "0"
CANDIDATE_COUNT_BUCKET_1_2: str = "1-2"
CANDIDATE_COUNT_BUCKET_3_5: str = "3-5"
CANDIDATE_COUNT_BUCKET_6P: str = "6+"

CANDIDATE_COUNT_BUCKETS: tuple[str, ...] = (
    CANDIDATE_COUNT_BUCKET_0,
    CANDIDATE_COUNT_BUCKET_1_2,
    CANDIDATE_COUNT_BUCKET_3_5,
    CANDIDATE_COUNT_BUCKET_6P,
)
CANDIDATE_COUNT_BUCKETS_SET: frozenset[str] = frozenset(CANDIDATE_COUNT_BUCKETS)


def bucket_candidate_count(value: Any) -> str:
    """Map a grasp-candidate count to one of :data:`CANDIDATE_COUNT_BUCKETS`.

    ``None``, missing, or non-integer inputs collapse to ``"0"``.
    """

    if not isinstance(value, int) or isinstance(value, bool):
        return CANDIDATE_COUNT_BUCKET_0
    if value <= 0:
        return CANDIDATE_COUNT_BUCKET_0
    if value <= 2:
        return CANDIDATE_COUNT_BUCKET_1_2
    if value <= 5:
        return CANDIDATE_COUNT_BUCKET_3_5
    return CANDIDATE_COUNT_BUCKET_6P


# ---------------------------------------------------------------------------
# Occlusion bucket.
# ---------------------------------------------------------------------------

OCCLUSION_BUCKET_LOW: str = "low"     # < 0.30
OCCLUSION_BUCKET_MID: str = "mid"     # [0.30, 0.70)
OCCLUSION_BUCKET_HIGH: str = "high"   # >= 0.70
OCCLUSION_BUCKET_UNKNOWN: str = "unknown"

OCCLUSION_BUCKETS: tuple[str, ...] = (
    OCCLUSION_BUCKET_LOW,
    OCCLUSION_BUCKET_MID,
    OCCLUSION_BUCKET_HIGH,
    OCCLUSION_BUCKET_UNKNOWN,
)
OCCLUSION_BUCKETS_SET: frozenset[str] = frozenset(OCCLUSION_BUCKETS)

OCCLUSION_LOW_UPPER_EXCL: float = 0.30
OCCLUSION_MID_UPPER_EXCL: float = 0.70


def bucket_occlusion(value: Any) -> str:
    """Map a ``[0, 1]`` occlusion ratio to one of :data:`OCCLUSION_BUCKETS`.

    ``None``, non-numeric and NaN inputs collapse to ``"unknown"``;
    the mapping never raises.
    """

    if value is None:
        return OCCLUSION_BUCKET_UNKNOWN
    if isinstance(value, bool):
        return OCCLUSION_BUCKET_UNKNOWN
    if not isinstance(value, (int, float)):
        return OCCLUSION_BUCKET_UNKNOWN
    fv = float(value)
    if math.isnan(fv) or math.isinf(fv):
        return OCCLUSION_BUCKET_UNKNOWN
    if fv < OCCLUSION_LOW_UPPER_EXCL:
        return OCCLUSION_BUCKET_LOW
    if fv < OCCLUSION_MID_UPPER_EXCL:
        return OCCLUSION_BUCKET_MID
    return OCCLUSION_BUCKET_HIGH


# ---------------------------------------------------------------------------
# 4-tuple state key.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerceptionBudgetStateKey:
    """Discrete 4-tuple state key.

    The string serialisation (:meth:`as_string_key`) is the artifact
    cell key; never reorder the fields.
    """

    views_seen_bucket: str
    last_candidate_count_bucket: str
    last_occlusion_bucket: str
    mode_bucket: str

    def __post_init__(self) -> None:
        if self.views_seen_bucket not in VIEWS_SEEN_BUCKETS_SET:
            raise ValueError(
                f"views_seen_bucket={self.views_seen_bucket!r} not in "
                f"{VIEWS_SEEN_BUCKETS!r}"
            )
        if self.last_candidate_count_bucket not in CANDIDATE_COUNT_BUCKETS_SET:
            raise ValueError(
                f"last_candidate_count_bucket="
                f"{self.last_candidate_count_bucket!r} not in "
                f"{CANDIDATE_COUNT_BUCKETS!r}"
            )
        if self.last_occlusion_bucket not in OCCLUSION_BUCKETS_SET:
            raise ValueError(
                f"last_occlusion_bucket={self.last_occlusion_bucket!r} "
                f"not in {OCCLUSION_BUCKETS!r}"
            )
        if self.mode_bucket not in MODE_BUCKETS_SET:
            raise ValueError(
                f"mode_bucket={self.mode_bucket!r} not in {MODE_BUCKETS!r}"
            )

    def as_string_key(self) -> str:
        """Stable artifact cell key (``"v|c|o|m"``)."""

        return (
            f"{self.views_seen_bucket}|"
            f"{self.last_candidate_count_bucket}|"
            f"{self.last_occlusion_bucket}|"
            f"{self.mode_bucket}"
        )


# ---------------------------------------------------------------------------
# One-hot encoding (LinUCB context).
# ---------------------------------------------------------------------------

#: Deterministic concatenation order of the one-hot bucket families.
#: Never reorder: the artifact's per-action ``A_diag`` and ``b``
#: vectors are indexed against this layout.
ONEHOT_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("views_seen", VIEWS_SEEN_BUCKETS),
    ("last_candidate_count", CANDIDATE_COUNT_BUCKETS),
    ("last_occlusion", OCCLUSION_BUCKETS),
    ("mode", MODE_BUCKETS),
)


def onehot_dim() -> int:
    """Length of the one-hot context vector."""

    return sum(len(buckets) for _, buckets in ONEHOT_FAMILIES)


def onehot_index_layout() -> tuple[str, ...]:
    """Return the ordered list of ``"family.bucket"`` labels (length = D)."""

    out: list[str] = []
    for fam, buckets in ONEHOT_FAMILIES:
        for b in buckets:
            out.append(f"{fam}.{b}")
    return tuple(out)


def state_key_to_onehot(key: PerceptionBudgetStateKey) -> tuple[int, ...]:
    """Encode a state key as a one-hot context vector (length = D).

    Exactly 4 entries are ``1`` (one per bucket family), the rest are
    ``0``. Deterministic and total: every valid state key has a
    well-defined encoding.
    """

    layout = onehot_index_layout()
    active = (
        f"views_seen.{key.views_seen_bucket}",
        f"last_candidate_count.{key.last_candidate_count_bucket}",
        f"last_occlusion.{key.last_occlusion_bucket}",
        f"mode.{key.mode_bucket}",
    )
    active_set = set(active)
    return tuple(1 if label in active_set else 0 for label in layout)


# ---------------------------------------------------------------------------
# Locked LinUCB hyperparameters.
# ---------------------------------------------------------------------------

#: LinUCB exploration coefficient alpha. Surfaced in the artifact so it
#: is part of the audit trail; reproducible at inference time.
DEFAULT_LINUCB_ALPHA: float = 1.0

#: Ridge regularisation. Keeps ``A_a`` positive-definite even when a
#: feature is never observed for that action.
DEFAULT_LINUCB_LAMBDA: float = 1.0

#: Default minimum per-cell (state, action) support before the lookup
#: answer is used. Below this, the fallback table serves the request.
DEFAULT_MIN_SUPPORT_THRESHOLD: int = 5


# ---------------------------------------------------------------------------
# Hand-authored fallback table.
# ---------------------------------------------------------------------------

#: Conservative default: continue gathering until at least 2 views,
#: then stop. This mirrors the deterministic
#: :class:`ScoringViewpointPlanner` behaviour for low view counts and
#: short-circuits perception cost beyond the second view.
DEFAULT_FALLBACK_TABLE: Mapping[str, str] = {
    VIEWS_SEEN_BUCKET_0: PERCEPTION_ACTION_CONTINUE,
    VIEWS_SEEN_BUCKET_1: PERCEPTION_ACTION_CONTINUE,
    VIEWS_SEEN_BUCKET_2: PERCEPTION_ACTION_STOP,
    VIEWS_SEEN_BUCKET_3P: PERCEPTION_ACTION_STOP,
}


def _validate_fallback_table(table: Mapping[str, str]) -> None:
    missing = sorted(VIEWS_SEEN_BUCKETS_SET - set(table.keys()))
    if missing:
        raise ValueError(
            f"fallback table missing entries for views_seen buckets: "
            f"{missing!r}"
        )
    for vb, action in table.items():
        if vb not in VIEWS_SEEN_BUCKETS_SET:
            raise ValueError(
                f"fallback table contains unknown views_seen bucket {vb!r}"
            )
        if action not in PERCEPTION_ACTIONS_SET:
            raise ValueError(
                f"fallback table action {action!r} for bucket {vb!r} "
                f"not in {PERCEPTION_ACTIONS!r}"
            )


# ---------------------------------------------------------------------------
# Policy interface.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerceptionBudgetRequest:
    """Per-attempt input to a :class:`PerceptionBudgetPolicy`."""

    attempt_id: str
    state_key: PerceptionBudgetStateKey


@dataclass(frozen=True)
class PerceptionBudgetSelection:
    """Producer-side proposal (pre-gate)."""

    policy_id: str
    policy_version: int
    action: str
    expected_reward_stop: float
    expected_reward_continue: float
    ucb_stop: float
    ucb_continue: float
    support_stop: int
    support_continue: int
    used_fallback: bool

    def __post_init__(self) -> None:
        if self.action not in PERCEPTION_ACTIONS_SET:
            raise ValueError(
                f"action={self.action!r} not in {PERCEPTION_ACTIONS!r}"
            )


class PerceptionBudgetPolicy(Protocol):
    """Frozen Protocol for perception-budget dispatchers.

    Distinct from the other policy protocols so the umbrella router
    can hold all of them without structural aliasing.
    """

    name: str
    version: int

    def propose_perception_budget(
        self, request: PerceptionBudgetRequest
    ) -> PerceptionBudgetSelection: ...


# ---------------------------------------------------------------------------
# LinUCB lookup policy.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinUCBPerceptionBudgetPolicy:
    """LinUCB contextual bandit on one-hot 4-tuple context.

    Storage form per action ``a`` (length-D vectors, D = :func:`onehot_dim`):

    * ``A_diag_per_action[a]``: diagonal entries of
      ``A_a = lambda I + Sigma x x^T``. Since every context vector is
      one-hot, ``x x^T`` is also diagonal, so the full matrix never
      needs to materialise.
    * ``b_per_action[a]``: the cumulative reward vector
      ``Sigma r * x``.

    The per-cell support is the sum of ``A_diag`` minus the ridge
    contribution, i.e. the count of training records that activated
    that one-hot feature for that action.

    Per-action posterior mean weights:
        ``theta_a[i] = b_a[i] / A_a[i]``.

    UCB score for an action under context ``x``:
        ``score = Sigma x[i] * theta_a[i] + alpha * sqrt(Sigma x[i]^2 / A_a[i])``.

    For a one-hot context the score reduces to the sum of the 4
    active feature contributions, deterministic and exact.
    """

    A_diag_per_action: Mapping[str, tuple[float, ...]]
    b_per_action: Mapping[str, tuple[float, ...]]
    feature_support_per_action: Mapping[str, tuple[int, ...]]
    fallback_table: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_FALLBACK_TABLE)
    )
    alpha: float = DEFAULT_LINUCB_ALPHA
    ridge_lambda: float = DEFAULT_LINUCB_LAMBDA
    min_support_threshold: int = DEFAULT_MIN_SUPPORT_THRESHOLD
    training_scope: str = MODE_BUCKET_DENSE
    name: str = "v5_perception_budget_baseline_v1"
    version: int = 1
    artifact_dataset_id: str = ""
    artifact_dataset_hash: str = ""

    def __post_init__(self) -> None:
        d = onehot_dim()
        for action in PERCEPTION_ACTIONS:
            A = self.A_diag_per_action.get(action)
            b = self.b_per_action.get(action)
            sup = self.feature_support_per_action.get(action)
            if A is None or b is None or sup is None:
                raise ValueError(
                    f"missing per-action vectors for action={action!r}"
                )
            if len(A) != d or len(b) != d or len(sup) != d:
                raise ValueError(
                    f"per-action vector length mismatch for {action!r}: "
                    f"got A={len(A)}, b={len(b)}, sup={len(sup)}; want {d}"
                )
            for ai in A:
                if not isinstance(ai, (int, float)) or float(ai) <= 0.0:
                    raise ValueError(
                        f"A_diag entry {ai!r} for action={action!r} must "
                        f"be a positive float (ridge enforces > 0)"
                    )
            for bi in b:
                if not isinstance(bi, (int, float)):
                    raise ValueError(
                        f"b entry {bi!r} for action={action!r} must be numeric"
                    )
            for s in sup:
                if not isinstance(s, int) or s < 0:
                    raise ValueError(
                        f"feature_support entry {s!r} for action={action!r} "
                        f"must be a non-negative int"
                    )
        _validate_fallback_table(self.fallback_table)
        if self.alpha < 0.0:
            raise ValueError(f"alpha={self.alpha!r} must be non-negative")
        if self.ridge_lambda <= 0.0:
            raise ValueError(
                f"ridge_lambda={self.ridge_lambda!r} must be positive"
            )
        if self.min_support_threshold < 0:
            raise ValueError(
                f"min_support_threshold={self.min_support_threshold!r} "
                f"must be non-negative"
            )
        if self.training_scope not in MODE_BUCKETS_SET:
            raise ValueError(
                f"training_scope={self.training_scope!r} not in "
                f"{MODE_BUCKETS!r}"
            )

    @property
    def policy_id(self) -> str:
        return f"{self.name}@{self.version}"

    # ------------------------------------------------------------------
    # Scoring.
    # ------------------------------------------------------------------

    def _score_action(
        self, action: str, onehot: tuple[int, ...]
    ) -> tuple[float, float, int]:
        """Return ``(expected_reward, ucb, support)`` for ``(action, x)``.

        ``support`` here is the minimum per-feature support across the
        4 active features, a conservative measure used by the fallback
        gate.
        """

        return score_linucb_action(
            A_diag=self.A_diag_per_action[action],
            b=self.b_per_action[action],
            support=self.feature_support_per_action[action],
            onehot=onehot,
            alpha=self.alpha,
        )

    def propose_perception_budget(
        self, request: PerceptionBudgetRequest
    ) -> PerceptionBudgetSelection:
        onehot = state_key_to_onehot(request.state_key)
        exp_stop, ucb_stop, sup_stop = self._score_action(
            PERCEPTION_ACTION_STOP, onehot
        )
        exp_cont, ucb_cont, sup_cont = self._score_action(
            PERCEPTION_ACTION_CONTINUE, onehot
        )

        # Fallback gate: if either action has support below threshold
        # for this state, fall back to the hand-authored table.
        if (
            sup_stop < self.min_support_threshold
            or sup_cont < self.min_support_threshold
        ):
            fallback_action = self.fallback_table[
                request.state_key.views_seen_bucket
            ]
            return PerceptionBudgetSelection(
                policy_id=self.policy_id,
                policy_version=self.version,
                action=fallback_action,
                expected_reward_stop=exp_stop,
                expected_reward_continue=exp_cont,
                ucb_stop=ucb_stop,
                ucb_continue=ucb_cont,
                support_stop=sup_stop,
                support_continue=sup_cont,
                used_fallback=True,
            )

        # Pick the action with the highest UCB score; deterministic
        # alphabetical tie-break, so "continue" (lexically before
        # "stop") wins ties.
        if ucb_stop > ucb_cont:
            chosen = PERCEPTION_ACTION_STOP
        elif ucb_cont > ucb_stop:
            chosen = PERCEPTION_ACTION_CONTINUE
        else:
            chosen = PERCEPTION_ACTION_CONTINUE  # lex tie-break
        return PerceptionBudgetSelection(
            policy_id=self.policy_id,
            policy_version=self.version,
            action=chosen,
            expected_reward_stop=exp_stop,
            expected_reward_continue=exp_cont,
            ucb_stop=ucb_stop,
            ucb_continue=ucb_cont,
            support_stop=sup_stop,
            support_continue=sup_cont,
            used_fallback=False,
        )


# ---------------------------------------------------------------------------
# Artifact (de)serialisation.
# ---------------------------------------------------------------------------


#: Perception-budget artifact schema version. Bump on any
#: incompatible on-disk JSON change.
PERCEPTION_BUDGET_ARTIFACT_SCHEMA_VERSION: int = 2


def load_linucb_perception_budget_policy(
    path: str | Path,
) -> LinUCBPerceptionBudgetPolicy:
    """Load a :class:`LinUCBPerceptionBudgetPolicy` from an artifact.

    Rejects schema-version drift and action / family drift so a
    stale artifact can never be silently loaded.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        blob: Mapping[str, Any] = json.load(f)
    schema_version = int(blob.get("schema_version", 0))
    if schema_version != PERCEPTION_BUDGET_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "LinUCBPerceptionBudgetPolicy artifact schema_version "
            f"{schema_version} does not match the frozen contract "
            f"{PERCEPTION_BUDGET_ARTIFACT_SCHEMA_VERSION}"
        )
    actions_blob = tuple(blob.get("actions", ()))
    if actions_blob != PERCEPTION_ACTIONS:
        raise ValueError(
            "LinUCBPerceptionBudgetPolicy artifact 'actions' does not "
            "match the frozen enum"
        )
    layout_blob = tuple(blob.get("onehot_layout", ()))
    if layout_blob != onehot_index_layout():
        raise ValueError(
            "LinUCBPerceptionBudgetPolicy artifact 'onehot_layout' does "
            "not match the frozen layout; feature drift detected"
        )
    A_blob = blob.get("A_diag_per_action", {}) or {}
    b_blob = blob.get("b_per_action", {}) or {}
    sup_blob = blob.get("feature_support_per_action", {}) or {}
    A_per_action: dict[str, tuple[float, ...]] = {}
    b_per_action: dict[str, tuple[float, ...]] = {}
    sup_per_action: dict[str, tuple[int, ...]] = {}
    for action in PERCEPTION_ACTIONS:
        if action not in A_blob or action not in b_blob or action not in sup_blob:
            raise ValueError(
                f"LinUCBPerceptionBudgetPolicy artifact missing per-action "
                f"data for {action!r}"
            )
        A_per_action[action] = tuple(float(v) for v in A_blob[action])
        b_per_action[action] = tuple(float(v) for v in b_blob[action])
        sup_per_action[action] = tuple(int(v) for v in sup_blob[action])
    fallback_blob = blob.get("fallback_table", {}) or {}
    fallback_table = {str(k): str(v) for k, v in fallback_blob.items()}
    logger.info(
        "Loaded perception-budget policy %s@%s from %s: %d action(s) x %d "
        "feature(s), alpha %.3f, min_support %s, scope %s, dataset %s (%s)",
        blob.get("policy_name", "v5_perception_budget_baseline_v1"),
        blob.get("policy_version", 1),
        path,
        len(PERCEPTION_ACTIONS),
        onehot_dim(),
        float(blob.get("alpha", DEFAULT_LINUCB_ALPHA)),
        blob.get("min_support_threshold", DEFAULT_MIN_SUPPORT_THRESHOLD),
        blob.get("training_scope", MODE_BUCKET_DENSE),
        blob.get("dataset_id", "?"),
        str(blob.get("dataset_hash", ""))[:12] or "?",
    )
    return LinUCBPerceptionBudgetPolicy(
        A_diag_per_action=A_per_action,
        b_per_action=b_per_action,
        feature_support_per_action=sup_per_action,
        fallback_table=fallback_table,
        alpha=float(blob.get("alpha", DEFAULT_LINUCB_ALPHA)),
        ridge_lambda=float(blob.get("ridge_lambda", DEFAULT_LINUCB_LAMBDA)),
        min_support_threshold=int(
            blob.get("min_support_threshold", DEFAULT_MIN_SUPPORT_THRESHOLD)
        ),
        training_scope=str(blob.get("training_scope", MODE_BUCKET_DENSE)),
        name=str(blob.get("policy_name", "v5_perception_budget_baseline_v1")),
        version=int(blob.get("policy_version", 1)),
        artifact_dataset_id=str(blob.get("dataset_id", "")),
        artifact_dataset_hash=str(blob.get("dataset_hash", "")),
    )


__all__ = (
    "CANDIDATE_COUNT_BUCKETS",
    "CANDIDATE_COUNT_BUCKETS_SET",
    "CANDIDATE_COUNT_BUCKET_0",
    "CANDIDATE_COUNT_BUCKET_1_2",
    "CANDIDATE_COUNT_BUCKET_3_5",
    "CANDIDATE_COUNT_BUCKET_6P",
    "DEFAULT_FALLBACK_TABLE",
    "DEFAULT_LINUCB_ALPHA",
    "DEFAULT_LINUCB_LAMBDA",
    "DEFAULT_MIN_SUPPORT_THRESHOLD",
    "LinUCBPerceptionBudgetPolicy",
    "MODE_BUCKETS",
    "MODE_BUCKETS_SET",
    "MODE_BUCKET_AUTO",
    "MODE_BUCKET_DENSE",
    "MODE_BUCKET_EASY",
    "MODE_BUCKET_UNKNOWN",
    "OCCLUSION_BUCKETS",
    "OCCLUSION_BUCKETS_SET",
    "OCCLUSION_BUCKET_HIGH",
    "OCCLUSION_BUCKET_LOW",
    "OCCLUSION_BUCKET_MID",
    "OCCLUSION_BUCKET_UNKNOWN",
    "OCCLUSION_LOW_UPPER_EXCL",
    "OCCLUSION_MID_UPPER_EXCL",
    "ONEHOT_FAMILIES",
    "PERCEPTION_ACTIONS",
    "PERCEPTION_ACTIONS_SET",
    "PERCEPTION_ACTION_CONTINUE",
    "PERCEPTION_ACTION_STOP",
    "PERCEPTION_BUDGET_ARTIFACT_SCHEMA_VERSION",
    "PerceptionBudgetPolicy",
    "PerceptionBudgetRequest",
    "PerceptionBudgetSelection",
    "PerceptionBudgetStateKey",
    "VIEWS_SEEN_BUCKETS",
    "VIEWS_SEEN_BUCKETS_SET",
    "VIEWS_SEEN_BUCKET_0",
    "VIEWS_SEEN_BUCKET_1",
    "VIEWS_SEEN_BUCKET_2",
    "VIEWS_SEEN_BUCKET_3P",
    "LinUCBPerceptionBudgetPolicy",
    "bucket_candidate_count",
    "bucket_mode",
    "bucket_occlusion",
    "bucket_views_seen",
    "hash_perception_budget_artifact",
    "load_linucb_perception_budget_policy",
    "onehot_dim",
    "onehot_index_layout",
    "state_key_to_onehot",
)
