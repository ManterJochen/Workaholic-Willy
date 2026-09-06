"""Recovery optimization policy contract + LinUCB baseline.

Authority lock:

* This is a shadow-only seam. The deterministic recovery orchestrator
  owns every motion; the policy only reorders and scores existing
  recovery sub-actions and never invents one.
* No edits to :mod:`src.robot.grasping.recovery.orchestrator`,
  :mod:`src.robot.grasping.loop.pick_loop`, or
  :mod:`src.robot.execution.autonomous_grasp`. The runtime touches it
  exclusively through :class:`ShadowRouter.run_recovery_shadow` and
  the typed :class:`RecoveryShadowTelemetry` carrier on
  :class:`ShadowRouterTelemetry`.
* No new fields on the replay-record ``extra.*`` schema and no
  additions to ``telemetry_catalog.py``.
* Defence-in-depth anti-loop gate forces ``abort_recovery`` once the
  attempt budget is exhausted.

Action space: the eight actions of :data:`RECOVERY_ACTIONS`, a
deterministic, bounded ordering choice over existing recovery
primitives.

State key: discrete 5-tuple ``(failure_class, attempt_index_clipped,
reobserve_count_clipped, dense_or_not, last_outcome_label)``.

Learner: LinUCB on the one-hot encoding of that key, diagonal-A
closed-form, pure stdlib. ``alpha=1.0``, ``lambda=1.0``, ``min_support=5``.
Below the min-support gate the policy falls back to a hand-authored
table keyed by ``failure_class``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from src.robot.grasping.constants import (
    RL_RECOVERY_POLICY_LOG_FILE,
    create_grasping_logger,
)

from ._artifact_io import hash_artifact as hash_recovery_artifact
from ._linucb import score_linucb_action
from .sequencing_policy import (
    FAILURE_CLASS_COLLISION,
    FAILURE_CLASS_DEFORMABLE,
    FAILURE_CLASS_DRIFT,
    FAILURE_CLASS_EMPTY_AIR,
    FAILURE_CLASS_NONE,
    FAILURE_CLASS_OCCLUSION,
    FAILURE_CLASS_SLIP,
    FAILURE_CLASS_UNKNOWN,
    FAILURE_CLASSES,
)


# ---------------------------------------------------------------------------
# Recovery sub-action enum.
# ---------------------------------------------------------------------------

RECOVERY_ACTION_REOBSERVE: str = "reobserve"
RECOVERY_ACTION_RE_SEGMENT: str = "re_segment"
RECOVERY_ACTION_REPLAN_GRASP: str = "replan_grasp"
RECOVERY_ACTION_PERTURB_AND_RETRY: str = "perturb_and_retry"
RECOVERY_ACTION_ABORT_RECOVERY: str = "abort_recovery"
# The sim's ``SceneRecoveryPolicy`` emits a richer recovery vocabulary than the five abstract
# actions above. These three concrete actions are appended and never reordered, so the policy learns
# over the same action space the deterministic recovery policy uses instead of dropping those steps.
# Artifacts that do not carry them give them zero training support, so they fall back until recovery
# episodes supply data.
RECOVERY_ACTION_NEXT_TARGET: str = "next_target"
RECOVERY_ACTION_NUDGE_TARGET: str = "nudge_target"
RECOVERY_ACTION_CONTAINER_AGITATE: str = "container_agitate"

#: Canonical ordering. Never reorder: artifact ``A_diag`` and ``b``
#: vectors are keyed by this ordering and downstream lex tie-breakers
#: rely on it. The three concrete sim actions sit after the five
#: abstract ones so existing artifact rows keep their positions.
RECOVERY_ACTIONS: tuple[str, ...] = (
    RECOVERY_ACTION_REOBSERVE,
    RECOVERY_ACTION_RE_SEGMENT,
    RECOVERY_ACTION_REPLAN_GRASP,
    RECOVERY_ACTION_PERTURB_AND_RETRY,
    RECOVERY_ACTION_ABORT_RECOVERY,
    RECOVERY_ACTION_NEXT_TARGET,
    RECOVERY_ACTION_NUDGE_TARGET,
    RECOVERY_ACTION_CONTAINER_AGITATE,
)
RECOVERY_ACTIONS_SET: frozenset[str] = frozenset(RECOVERY_ACTIONS)


# ---------------------------------------------------------------------------
# State-key bucket families.
# ---------------------------------------------------------------------------

# failure_class: reuse the locked failure-class enum verbatim so
# replay records stay schema-compatible across versions.
FAILURE_CLASS_BUCKETS: tuple[str, ...] = FAILURE_CLASSES
FAILURE_CLASS_BUCKETS_SET: frozenset[str] = frozenset(FAILURE_CLASS_BUCKETS)

# attempt_index: clipped to {0, 1, 2, 3+}
ATTEMPT_INDEX_BUCKET_0: str = "0"
ATTEMPT_INDEX_BUCKET_1: str = "1"
ATTEMPT_INDEX_BUCKET_2: str = "2"
ATTEMPT_INDEX_BUCKET_3P: str = "3+"
ATTEMPT_INDEX_BUCKETS: tuple[str, ...] = (
    ATTEMPT_INDEX_BUCKET_0,
    ATTEMPT_INDEX_BUCKET_1,
    ATTEMPT_INDEX_BUCKET_2,
    ATTEMPT_INDEX_BUCKET_3P,
)
ATTEMPT_INDEX_BUCKETS_SET: frozenset[str] = frozenset(ATTEMPT_INDEX_BUCKETS)


def bucket_attempt_index(value: Any) -> str:
    if not isinstance(value, int) or isinstance(value, bool):
        return ATTEMPT_INDEX_BUCKET_0
    if value <= 0:
        return ATTEMPT_INDEX_BUCKET_0
    if value == 1:
        return ATTEMPT_INDEX_BUCKET_1
    if value == 2:
        return ATTEMPT_INDEX_BUCKET_2
    return ATTEMPT_INDEX_BUCKET_3P


# reobserve_count: clipped to {0, 1, 2, 3+}
REOBSERVE_COUNT_BUCKET_0: str = "0"
REOBSERVE_COUNT_BUCKET_1: str = "1"
REOBSERVE_COUNT_BUCKET_2: str = "2"
REOBSERVE_COUNT_BUCKET_3P: str = "3+"
REOBSERVE_COUNT_BUCKETS: tuple[str, ...] = (
    REOBSERVE_COUNT_BUCKET_0,
    REOBSERVE_COUNT_BUCKET_1,
    REOBSERVE_COUNT_BUCKET_2,
    REOBSERVE_COUNT_BUCKET_3P,
)
REOBSERVE_COUNT_BUCKETS_SET: frozenset[str] = frozenset(REOBSERVE_COUNT_BUCKETS)


def bucket_reobserve_count(value: Any) -> str:
    if not isinstance(value, int) or isinstance(value, bool):
        return REOBSERVE_COUNT_BUCKET_0
    if value <= 0:
        return REOBSERVE_COUNT_BUCKET_0
    if value == 1:
        return REOBSERVE_COUNT_BUCKET_1
    if value == 2:
        return REOBSERVE_COUNT_BUCKET_2
    return REOBSERVE_COUNT_BUCKET_3P


# dense_or_not: {dense, non_dense}
DENSE_BUCKET_DENSE: str = "dense"
DENSE_BUCKET_NON_DENSE: str = "non_dense"
DENSE_BUCKETS: tuple[str, ...] = (DENSE_BUCKET_DENSE, DENSE_BUCKET_NON_DENSE)
DENSE_BUCKETS_SET: frozenset[str] = frozenset(DENSE_BUCKETS)


def bucket_dense(mode_value: Any) -> str:
    """Map a runtime ``mode`` token to dense vs non-dense (prefix match)."""

    if not isinstance(mode_value, str):
        return DENSE_BUCKET_NON_DENSE
    if mode_value.startswith("dense"):
        return DENSE_BUCKET_DENSE
    return DENSE_BUCKET_NON_DENSE


# last_outcome_label: bounded set
LAST_OUTCOME_SUCCEEDED: str = "succeeded"
LAST_OUTCOME_FAILED: str = "failed"
LAST_OUTCOME_RECOVERED: str = "recovered"
LAST_OUTCOME_UNKNOWN: str = "unknown"
LAST_OUTCOME_BUCKETS: tuple[str, ...] = (
    LAST_OUTCOME_SUCCEEDED,
    LAST_OUTCOME_FAILED,
    LAST_OUTCOME_RECOVERED,
    LAST_OUTCOME_UNKNOWN,
)
LAST_OUTCOME_BUCKETS_SET: frozenset[str] = frozenset(LAST_OUTCOME_BUCKETS)


def bucket_last_outcome(value: Any) -> str:
    if not isinstance(value, str):
        return LAST_OUTCOME_UNKNOWN
    if value in LAST_OUTCOME_BUCKETS_SET:
        return value
    # Common synonyms surfaced by the telemetry catalog.
    if value in ("success",):
        return LAST_OUTCOME_SUCCEEDED
    if value in ("failure", "failed_attempt"):
        return LAST_OUTCOME_FAILED
    return LAST_OUTCOME_UNKNOWN


# ---------------------------------------------------------------------------
# State key (5-tuple).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryStateKey:
    """Discrete 5-tuple state key for the recovery policy."""

    failure_class_bucket: str
    attempt_index_bucket: str
    reobserve_count_bucket: str
    dense_bucket: str
    last_outcome_bucket: str

    def __post_init__(self) -> None:
        if self.failure_class_bucket not in FAILURE_CLASS_BUCKETS_SET:
            raise ValueError(
                f"failure_class_bucket={self.failure_class_bucket!r} not in "
                f"{FAILURE_CLASS_BUCKETS!r}"
            )
        if self.attempt_index_bucket not in ATTEMPT_INDEX_BUCKETS_SET:
            raise ValueError(
                f"attempt_index_bucket={self.attempt_index_bucket!r} not in "
                f"{ATTEMPT_INDEX_BUCKETS!r}"
            )
        if self.reobserve_count_bucket not in REOBSERVE_COUNT_BUCKETS_SET:
            raise ValueError(
                f"reobserve_count_bucket={self.reobserve_count_bucket!r} not in "
                f"{REOBSERVE_COUNT_BUCKETS!r}"
            )
        if self.dense_bucket not in DENSE_BUCKETS_SET:
            raise ValueError(
                f"dense_bucket={self.dense_bucket!r} not in {DENSE_BUCKETS!r}"
            )
        if self.last_outcome_bucket not in LAST_OUTCOME_BUCKETS_SET:
            raise ValueError(
                f"last_outcome_bucket={self.last_outcome_bucket!r} not in "
                f"{LAST_OUTCOME_BUCKETS!r}"
            )

    def as_string_key(self) -> str:
        return (
            f"{self.failure_class_bucket}|"
            f"{self.attempt_index_bucket}|"
            f"{self.reobserve_count_bucket}|"
            f"{self.dense_bucket}|"
            f"{self.last_outcome_bucket}"
        )


# ---------------------------------------------------------------------------
# One-hot encoding.
# ---------------------------------------------------------------------------

ONEHOT_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("failure_class", FAILURE_CLASS_BUCKETS),
    ("attempt_index", ATTEMPT_INDEX_BUCKETS),
    ("reobserve_count", REOBSERVE_COUNT_BUCKETS),
    ("dense", DENSE_BUCKETS),
    ("last_outcome", LAST_OUTCOME_BUCKETS),
)


def onehot_dim() -> int:
    return sum(len(buckets) for _, buckets in ONEHOT_FAMILIES)


def onehot_index_layout() -> tuple[str, ...]:
    out: list[str] = []
    for fam, buckets in ONEHOT_FAMILIES:
        for b in buckets:
            out.append(f"{fam}.{b}")
    return tuple(out)


def state_key_to_onehot(key: RecoveryStateKey) -> tuple[int, ...]:
    layout = onehot_index_layout()
    active = (
        f"failure_class.{key.failure_class_bucket}",
        f"attempt_index.{key.attempt_index_bucket}",
        f"reobserve_count.{key.reobserve_count_bucket}",
        f"dense.{key.dense_bucket}",
        f"last_outcome.{key.last_outcome_bucket}",
    )
    active_set = set(active)
    return tuple(1 if label in active_set else 0 for label in layout)


# ---------------------------------------------------------------------------
# Hyperparameters + fallback table.
# ---------------------------------------------------------------------------

DEFAULT_LINUCB_ALPHA: float = 1.0
DEFAULT_LINUCB_LAMBDA: float = 1.0
DEFAULT_MIN_SUPPORT_THRESHOLD: int = 5
DEFAULT_MAX_RECOVERY_ATTEMPTS: int = 3

#: Hand-authored fallback table keyed by ``failure_class_bucket``.
#: Encodes the deterministic-conservative default that maps each
#: typed failure class to the safest deterministic recovery sub-action.
#: Used both when the policy artifact is missing and when min-support
#: is not met for the active 5-tuple cell.
DEFAULT_FALLBACK_TABLE: Mapping[str, str] = {
    FAILURE_CLASS_NONE: RECOVERY_ACTION_REOBSERVE,
    FAILURE_CLASS_UNKNOWN: RECOVERY_ACTION_REOBSERVE,
    FAILURE_CLASS_SLIP: RECOVERY_ACTION_REPLAN_GRASP,
    FAILURE_CLASS_EMPTY_AIR: RECOVERY_ACTION_RE_SEGMENT,
    FAILURE_CLASS_COLLISION: RECOVERY_ACTION_REPLAN_GRASP,
    FAILURE_CLASS_OCCLUSION: RECOVERY_ACTION_REOBSERVE,
    FAILURE_CLASS_DEFORMABLE: RECOVERY_ACTION_PERTURB_AND_RETRY,
    FAILURE_CLASS_DRIFT: RECOVERY_ACTION_ABORT_RECOVERY,
}


# ---------------------------------------------------------------------------
# Request / Selection types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryRequest:
    attempt_id: str
    state_key: RecoveryStateKey


@dataclass(frozen=True)
class RecoveryActionScore:
    """Per-action score row carried inside :class:`RecoverySelection`."""

    action: str
    expected_reward: float
    ucb: float
    min_support_over_active: int

    def __post_init__(self) -> None:
        if self.action not in RECOVERY_ACTIONS_SET:
            raise ValueError(
                f"action={self.action!r} not in {RECOVERY_ACTIONS!r}"
            )


@dataclass(frozen=True)
class RecoverySelection:
    """Output of :meth:`RecoveryPolicy.propose_recovery`.

    Carries the ranked action ordering + per-action scores so the
    shadow router can emit a complete audit row for every recovery
    invocation.
    """

    policy_id: str
    policy_version: int
    action: str
    ranked_actions: tuple[str, ...]
    scores: tuple[RecoveryActionScore, ...]
    used_fallback: bool

    def __post_init__(self) -> None:
        if self.action not in RECOVERY_ACTIONS_SET:
            raise ValueError(
                f"action={self.action!r} not in {RECOVERY_ACTIONS!r}"
            )
        if set(self.ranked_actions) != RECOVERY_ACTIONS_SET:
            raise ValueError(
                "ranked_actions must be a permutation of RECOVERY_ACTIONS; "
                f"got {self.ranked_actions!r}"
            )
        if len(self.scores) != len(RECOVERY_ACTIONS):
            raise ValueError(
                f"scores length {len(self.scores)} != "
                f"{len(RECOVERY_ACTIONS)} actions"
            )


class RecoveryPolicy(Protocol):
    name: str
    version: int

    def propose_recovery(self, request: RecoveryRequest) -> RecoverySelection:
        ...


# ---------------------------------------------------------------------------
# LinUCB recovery policy (diagonal-A closed-form, pure stdlib).
# ---------------------------------------------------------------------------


def _validate_per_action_vectors(
    name: str, mapping: Mapping[str, Sequence[float] | Sequence[int]], dim: int
) -> None:
    if set(mapping.keys()) != RECOVERY_ACTIONS_SET:
        raise ValueError(
            f"{name} keys must equal RECOVERY_ACTIONS; got {set(mapping.keys())!r}"
        )
    for a, v in mapping.items():
        if len(v) != dim:
            raise ValueError(
                f"{name}[{a!r}] length {len(v)} != onehot_dim {dim}"
            )


@dataclass(frozen=True)
class LinUCBRecoveryPolicy:
    """Deterministic LinUCB recovery-action policy."""

    A_diag_per_action: Mapping[str, tuple[float, ...]]
    b_per_action: Mapping[str, tuple[float, ...]]
    feature_support_per_action: Mapping[str, tuple[int, ...]]
    fallback_table: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_FALLBACK_TABLE)
    )
    alpha: float = DEFAULT_LINUCB_ALPHA
    ridge_lambda: float = DEFAULT_LINUCB_LAMBDA
    min_support_threshold: int = DEFAULT_MIN_SUPPORT_THRESHOLD
    training_scope: str = DENSE_BUCKET_DENSE
    name: str = "v6_recovery_baseline_v1"
    version: int = 1
    artifact_dataset_id: str = ""
    artifact_dataset_hash: str = ""

    def __post_init__(self) -> None:
        dim = onehot_dim()
        _validate_per_action_vectors("A_diag_per_action", self.A_diag_per_action, dim)
        _validate_per_action_vectors("b_per_action", self.b_per_action, dim)
        _validate_per_action_vectors(
            "feature_support_per_action", self.feature_support_per_action, dim
        )
        for a, av in self.A_diag_per_action.items():
            for x in av:
                if not (x > 0.0):
                    raise ValueError(
                        f"A_diag_per_action[{a!r}] must be strictly positive; "
                        f"got non-positive entry {x}"
                    )
        for a, sv in self.feature_support_per_action.items():
            for s in sv:
                if s < 0:
                    raise ValueError(
                        f"feature_support_per_action[{a!r}] must be non-negative"
                    )
        # Fallback table must cover every failure class to keep
        # ``propose_recovery`` total.
        missing = FAILURE_CLASS_BUCKETS_SET - set(self.fallback_table.keys())
        if missing:
            raise ValueError(
                f"fallback_table missing failure classes: {sorted(missing)!r}"
            )
        for fc, a in self.fallback_table.items():
            if a not in RECOVERY_ACTIONS_SET:
                raise ValueError(
                    f"fallback_table[{fc!r}]={a!r} not in {RECOVERY_ACTIONS!r}"
                )
        if self.min_support_threshold < 0:
            raise ValueError("min_support_threshold must be >= 0")

    @property
    def policy_id(self) -> str:
        return f"{self.name}@{self.version}"

    def _score_action(
        self, action: str, onehot: Sequence[int]
    ) -> tuple[float, float, int]:
        """Return ``(expected, ucb, min_support_over_active)`` for one action."""

        return score_linucb_action(
            A_diag=self.A_diag_per_action[action],
            b=self.b_per_action[action],
            support=self.feature_support_per_action[action],
            onehot=onehot,
            alpha=self.alpha,
        )

    def propose_recovery(self, request: RecoveryRequest) -> RecoverySelection:
        onehot = state_key_to_onehot(request.state_key)

        scored: list[RecoveryActionScore] = []
        per_action_min_sup: dict[str, int] = {}
        for a in RECOVERY_ACTIONS:
            expected, ucb, min_sup = self._score_action(a, onehot)
            scored.append(
                RecoveryActionScore(
                    action=a,
                    expected_reward=expected,
                    ucb=ucb,
                    min_support_over_active=min_sup,
                )
            )
            per_action_min_sup[a] = min_sup

        # Fallback gate: if any action's min-support across the 5
        # active features is below the threshold, fall back to the
        # hand-authored table (a learned vote on a cell where at least
        # one arm is unsupported is unreliable).
        #
        # The min() is over all 8 RECOVERY_ACTIONS, so a deterministic behaviour policy leaves every
        # arm it never picks at zero support and the gate falls back at any data volume. That is a
        # coverage requirement, not a data shortage: only deliberate exploration (a randomised
        # recovery action) or a gate redesign changes it. All-arms gating is the conservatism an
        # offline policy needs, because LinUCB optimism would otherwise prefer an unsupported arm.
        # Do not relax the gate: the seam is shadow-only, so relaxing it buys no behaviour and
        # costs byte-identity.
        gate_min_sup = min(per_action_min_sup.values()) if per_action_min_sup else 0
        used_fallback = gate_min_sup < self.min_support_threshold

        if used_fallback:
            fc = request.state_key.failure_class_bucket
            chosen = self.fallback_table.get(
                fc, RECOVERY_ACTION_ABORT_RECOVERY
            )
            logger.debug(
                "Recovery fallback: min support %d < %d -> hand-authored action %r "
                "for failure class %r (learned arms not trusted)",
                gate_min_sup,
                self.min_support_threshold,
                chosen,
                fc,
            )
            # Deterministic ranking under fallback: chosen first, then
            # the remaining actions in canonical order.
            tail = tuple(a for a in RECOVERY_ACTIONS if a != chosen)
            ranked = (chosen,) + tail
        else:
            # UCB pick; lex tie-break uses RECOVERY_ACTIONS order
            # (stable, deterministic).
            def sort_key(s: RecoveryActionScore) -> tuple[float, int]:
                return (-s.ucb, RECOVERY_ACTIONS.index(s.action))

            ranked_scores = sorted(scored, key=sort_key)
            ranked = tuple(s.action for s in ranked_scores)
            chosen = ranked[0]

        return RecoverySelection(
            policy_id=self.policy_id,
            policy_version=self.version,
            action=chosen,
            ranked_actions=ranked,
            scores=tuple(scored),
            used_fallback=used_fallback,
        )


#: Two things here are decided rather than raised and leave no other trace: the
#: min-support fallback (the learned arm was not trusted) and the anti-loop clip
#: (the budget forced an abort). Both read downstream as an ordinary recovery
#: choice. The fallback logs at debug because falling back is the ordinary path;
#: the clip logs at warning because a bound was enforced.
logger = create_grasping_logger("RLRecoveryPolicy", RL_RECOVERY_POLICY_LOG_FILE)


# ---------------------------------------------------------------------------
# Defence-in-depth anti-loop gate.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AntiLoopGateResult:
    """Outcome of :func:`apply_recovery_anti_loop_gate`."""

    action: str
    clipped: bool
    clip_reason: str


def apply_recovery_anti_loop_gate(
    *,
    proposed_action: str,
    attempt_index: int,
    max_recovery_attempts: int = DEFAULT_MAX_RECOVERY_ATTEMPTS,
) -> AntiLoopGateResult:
    """Force ``abort_recovery`` once the attempt budget is exhausted.

    Defence-in-depth gate that runs on top of any policy proposal.
    ``attempt_index`` is the next attempt number (0-based for the
    first recovery attempt). When ``attempt_index >= max_recovery_attempts``
    any non-abort proposal is rewritten to ``abort_recovery`` with
    reason ``"max_recovery_attempts_reached"``.

    Invalid proposed_action tokens are also clipped to ``abort_recovery``
    with reason ``"invalid_proposed_action"`` so a buggy upstream
    cannot inject an unknown token into the runtime fold.
    """

    if proposed_action not in RECOVERY_ACTIONS_SET:
        # An unknown token reaching the runtime fold is an upstream defect, not a
        # policy decision; it must not disappear behind a clean abort.
        logger.error(
            "Anti-loop gate clipped an invalid proposed recovery action %r to %r",
            proposed_action,
            RECOVERY_ACTION_ABORT_RECOVERY,
        )
        return AntiLoopGateResult(
            action=RECOVERY_ACTION_ABORT_RECOVERY,
            clipped=True,
            clip_reason="invalid_proposed_action",
        )
    if attempt_index >= max_recovery_attempts and proposed_action != RECOVERY_ACTION_ABORT_RECOVERY:
        logger.warning(
            "Anti-loop gate clipped %r to %r: attempt %d of a %d-attempt budget",
            proposed_action,
            RECOVERY_ACTION_ABORT_RECOVERY,
            attempt_index,
            max_recovery_attempts,
        )
        return AntiLoopGateResult(
            action=RECOVERY_ACTION_ABORT_RECOVERY,
            clipped=True,
            clip_reason="max_recovery_attempts_reached",
        )
    return AntiLoopGateResult(
        action=proposed_action, clipped=False, clip_reason="none"
    )


# ---------------------------------------------------------------------------
# Artifact schema + loader.
# ---------------------------------------------------------------------------

#: Artifact carries reward_model / dataset_provenance / degeneracy_note honesty stamps.
RECOVERY_ARTIFACT_SCHEMA_VERSION: int = 2


def load_linucb_recovery_policy(path: str | Path) -> LinUCBRecoveryPolicy:
    """Deserialise a recovery-policy artifact with strict drift checks."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        blob = json.load(f)

    sv = blob.get("schema_version")
    if sv != RECOVERY_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version drift: got {sv!r}, expected "
            f"{RECOVERY_ARTIFACT_SCHEMA_VERSION!r}"
        )

    actions = tuple(blob.get("actions") or ())
    if actions != RECOVERY_ACTIONS:
        raise ValueError(
            f"actions drift: artifact={actions!r}, code={RECOVERY_ACTIONS!r}"
        )
    layout = tuple(blob.get("onehot_layout") or ())
    if layout != onehot_index_layout():
        raise ValueError(
            "onehot_layout drift between artifact and policy code; refusing to load"
        )

    A = {a: tuple(float(x) for x in blob["A_diag_per_action"][a]) for a in RECOVERY_ACTIONS}
    b = {a: tuple(float(x) for x in blob["b_per_action"][a]) for a in RECOVERY_ACTIONS}
    sup = {
        a: tuple(int(x) for x in blob["feature_support_per_action"][a])
        for a in RECOVERY_ACTIONS
    }
    fallback = {str(k): str(v) for k, v in blob["fallback_table"].items()}

    logger.info(
        "Loaded recovery policy %s@%s from %s: %d action(s) x %d feature(s), "
        "alpha %.3f, min_support %s, scope %s, dataset %s (%s)",
        blob.get("policy_name", "v6_recovery_baseline_v1"),
        blob.get("policy_version", 1),
        path,
        len(RECOVERY_ACTIONS),
        onehot_dim(),
        float(blob.get("alpha", DEFAULT_LINUCB_ALPHA)),
        blob.get("min_support_threshold", DEFAULT_MIN_SUPPORT_THRESHOLD),
        blob.get("training_scope", DENSE_BUCKET_DENSE),
        blob.get("dataset_id", "?"),
        str(blob.get("dataset_hash", ""))[:12] or "?",
    )
    return LinUCBRecoveryPolicy(
        A_diag_per_action=A,
        b_per_action=b,
        feature_support_per_action=sup,
        fallback_table=fallback,
        alpha=float(blob.get("alpha", DEFAULT_LINUCB_ALPHA)),
        ridge_lambda=float(blob.get("ridge_lambda", DEFAULT_LINUCB_LAMBDA)),
        min_support_threshold=int(
            blob.get("min_support_threshold", DEFAULT_MIN_SUPPORT_THRESHOLD)
        ),
        training_scope=str(blob.get("training_scope", DENSE_BUCKET_DENSE)),
        name=str(blob.get("policy_name", "v6_recovery_baseline_v1")),
        version=int(blob.get("policy_version", 1)),
        artifact_dataset_id=str(blob.get("dataset_id", "")),
        artifact_dataset_hash=str(blob.get("dataset_hash", "")),
    )


__all__ = (
    "ATTEMPT_INDEX_BUCKETS",
    "DEFAULT_FALLBACK_TABLE",
    "DEFAULT_LINUCB_ALPHA",
    "DEFAULT_LINUCB_LAMBDA",
    "DEFAULT_MAX_RECOVERY_ATTEMPTS",
    "DEFAULT_MIN_SUPPORT_THRESHOLD",
    "DENSE_BUCKETS",
    "DENSE_BUCKET_DENSE",
    "DENSE_BUCKET_NON_DENSE",
    "FAILURE_CLASS_BUCKETS",
    "LAST_OUTCOME_BUCKETS",
    "LAST_OUTCOME_FAILED",
    "LAST_OUTCOME_RECOVERED",
    "LAST_OUTCOME_SUCCEEDED",
    "LAST_OUTCOME_UNKNOWN",
    "ONEHOT_FAMILIES",
    "RECOVERY_ACTIONS",
    "RECOVERY_ACTIONS_SET",
    "RECOVERY_ACTION_ABORT_RECOVERY",
    "RECOVERY_ACTION_PERTURB_AND_RETRY",
    "RECOVERY_ACTION_REOBSERVE",
    "RECOVERY_ACTION_REPLAN_GRASP",
    "RECOVERY_ACTION_RE_SEGMENT",
    "RECOVERY_ARTIFACT_SCHEMA_VERSION",
    "REOBSERVE_COUNT_BUCKETS",
    "AntiLoopGateResult",
    "LinUCBRecoveryPolicy",
    "RecoveryActionScore",
    "RecoveryPolicy",
    "RecoveryRequest",
    "RecoverySelection",
    "RecoveryStateKey",
    "apply_recovery_anti_loop_gate",
    "bucket_attempt_index",
    "bucket_dense",
    "bucket_last_outcome",
    "bucket_reobserve_count",
    "hash_recovery_artifact",
    "load_linucb_recovery_policy",
    "onehot_dim",
    "onehot_index_layout",
    "state_key_to_onehot",
)
