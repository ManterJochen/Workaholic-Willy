"""Sequencing-policy contract + deterministic lookup-table baseline.

Ships one live sequencing dispatcher: a deterministic lookup table
from state to action over the failure-side post-attempt action enum
``{GRASP, REOBSERVE, RECOVER, ABORT}``.

Authority lock:

* shadow only; never overrides safety, geometry or deterministic
  sequencing,
* deterministic for a fixed state key + artifact,
* loader rejects schema-version drift (no silent staleness).

Contract notes:

* Success outcomes never invoke the seam (the helper is a
  failure-only producer); the lookup table need not enumerate success
  cells.
* ``"grasp"`` post-attempt means "proceed with next attempt after the
  deterministic perception refresh": the action the baseline already
  takes on most failures.
* The baseline is the deterministic loop's own next-iter decision,
  derived in a pure helper at the pick_loop seam, not here.
* State key shape: ``(last_outcome_label, attempt_index in [0,9],
  commit_reobserve_count in [0,3], last_failure_class)``.
* Failure-class enum: the 6 taxonomy classes plus ``"none"`` and
  ``"unknown"``, a frozen string set mirrored in the artifact's
  ``state_schema`` block.
* "Training" is empirical-frequency over observed transitions from a
  state to its next action, with a minimum support threshold; cells
  below threshold fall back to a hand-authored mapping baked in the
  artifact.
* The anti-loop gate lives inside
  :meth:`ShadowRouter.run_sequencing_shadow` and emits the gated
  action alongside the producer's proposed action.
* Per-attempt :class:`SequencingDecision` rows aggregate into a
  ``decisions`` tuple on :class:`SequencingShadowTelemetry`; the extras
  dict summarises the last failure attempt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from src.robot.grasping.constants import (
    RL_SEQUENCING_POLICY_LOG_FILE,
    create_grasping_logger,
)

from ._artifact_io import hash_artifact as hash_sequencing_artifact

#: The anti-loop clip is a bound being enforced, and it reads downstream as an
#: ordinary abort, so it is logged here; the per-attempt decision row already
#: travels as telemetry.
logger = create_grasping_logger("RLSequencingPolicy", RL_SEQUENCING_POLICY_LOG_FILE)


# ---------------------------------------------------------------------------
# Action enum (4-action post-attempt set).
# ---------------------------------------------------------------------------

SEQUENCING_ACTION_GRASP: str = "grasp"
SEQUENCING_ACTION_REOBSERVE: str = "reobserve"
SEQUENCING_ACTION_RECOVER: str = "recover"
SEQUENCING_ACTION_ABORT: str = "abort"

#: Deterministic ordering of the action enum. The artifact's
#: ``actions`` block must match this order; never reorder.
SEQUENCING_ACTIONS: tuple[str, ...] = (
    SEQUENCING_ACTION_GRASP,
    SEQUENCING_ACTION_REOBSERVE,
    SEQUENCING_ACTION_RECOVER,
    SEQUENCING_ACTION_ABORT,
)

#: Frozen set form for membership checks.
SEQUENCING_ACTIONS_SET: frozenset[str] = frozenset(SEQUENCING_ACTIONS)


# ---------------------------------------------------------------------------
# Outcome + failure-class label enums.
# ---------------------------------------------------------------------------

OUTCOME_LABEL_SUCCESS: str = "success"
OUTCOME_LABEL_FAILURE: str = "failure"
#: Labels emitted on the last attempt's outcome at the seam. The seam
#: never fires on ``"success"``; the success label appears only in the
#: artifact's documented state_schema for completeness.
OUTCOME_LABELS: tuple[str, ...] = (
    OUTCOME_LABEL_SUCCESS,
    OUTCOME_LABEL_FAILURE,
)

FAILURE_CLASS_NONE: str = "none"
FAILURE_CLASS_UNKNOWN: str = "unknown"
FAILURE_CLASS_SLIP: str = "slip_after_grasp"
FAILURE_CLASS_EMPTY_AIR: str = "empty_air_grasp"
FAILURE_CLASS_COLLISION: str = "collision_rejection"
FAILURE_CLASS_OCCLUSION: str = "occlusion_misread"
FAILURE_CLASS_DEFORMABLE: str = "deformable_misclassification"
FAILURE_CLASS_DRIFT: str = "calibration_drift_suspected"

#: Frozen 8-class enum: 6 taxonomy classes + ``none`` (success-side,
#: present only for state_schema completeness) + ``unknown`` (live
#: runtime when the taxonomy was not computed for the attempt).
FAILURE_CLASSES: tuple[str, ...] = (
    FAILURE_CLASS_NONE,
    FAILURE_CLASS_UNKNOWN,
    FAILURE_CLASS_SLIP,
    FAILURE_CLASS_EMPTY_AIR,
    FAILURE_CLASS_COLLISION,
    FAILURE_CLASS_OCCLUSION,
    FAILURE_CLASS_DEFORMABLE,
    FAILURE_CLASS_DRIFT,
)
FAILURE_CLASSES_SET: frozenset[str] = frozenset(FAILURE_CLASSES)


# ---------------------------------------------------------------------------
# State-key clip ranges.
# ---------------------------------------------------------------------------

#: Inclusive upper bound on the clipped ``attempt_index`` cell key.
#: Pick loops above this depth re-use the same cell; small key
#: cardinality keeps the table fully enumerable.
ATTEMPT_INDEX_MAX: int = 9

#: Inclusive upper bound on the clipped ``commit_reobserve_count``
#: cell key.
COMMIT_REOBSERVE_COUNT_MAX: int = 3


def clip_attempt_index(value: int) -> int:
    if value < 0:
        return 0
    if value > ATTEMPT_INDEX_MAX:
        return ATTEMPT_INDEX_MAX
    return int(value)


def clip_commit_reobserve_count(value: int) -> int:
    if value < 0:
        return 0
    if value > COMMIT_REOBSERVE_COUNT_MAX:
        return COMMIT_REOBSERVE_COUNT_MAX
    return int(value)


@dataclass(frozen=True)
class SequencingStateKey:
    """Discrete 4-tuple state key.

    The string serialisation (:meth:`as_string_key`) is the artifact
    cell key; never reorder the fields.
    """

    last_outcome_label: str
    attempt_index_clipped: int
    commit_reobserve_count_clipped: int
    last_failure_class: str

    def __post_init__(self) -> None:
        if self.last_outcome_label not in OUTCOME_LABELS:
            raise ValueError(
                f"last_outcome_label={self.last_outcome_label!r} not in "
                f"{OUTCOME_LABELS!r}"
            )
        if not (0 <= self.attempt_index_clipped <= ATTEMPT_INDEX_MAX):
            raise ValueError(
                f"attempt_index_clipped out of [0,{ATTEMPT_INDEX_MAX}]"
            )
        if not (
            0
            <= self.commit_reobserve_count_clipped
            <= COMMIT_REOBSERVE_COUNT_MAX
        ):
            raise ValueError(
                "commit_reobserve_count_clipped out of "
                f"[0,{COMMIT_REOBSERVE_COUNT_MAX}]"
            )
        if self.last_failure_class not in FAILURE_CLASSES_SET:
            raise ValueError(
                f"last_failure_class={self.last_failure_class!r} not in "
                f"{FAILURE_CLASSES!r}"
            )

    def as_string_key(self) -> str:
        """Stable artifact cell key (``"outcome|idx|cnt|class"``)."""

        return (
            f"{self.last_outcome_label}|{self.attempt_index_clipped}|"
            f"{self.commit_reobserve_count_clipped}|"
            f"{self.last_failure_class}"
        )


# ---------------------------------------------------------------------------
# Policy interface.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequencingRequest:
    """Per-attempt input to a :class:`SequencingPolicy`."""

    attempt_id: str
    state_key: SequencingStateKey


@dataclass(frozen=True)
class SequencingSelection:
    """Producer-side proposal (pre-gate)."""

    policy_id: str
    policy_version: int
    action: str
    support_count: int
    used_fallback: bool

    def __post_init__(self) -> None:
        if self.action not in SEQUENCING_ACTIONS_SET:
            raise ValueError(
                f"action={self.action!r} not in {SEQUENCING_ACTIONS!r}"
            )


class SequencingPolicy(Protocol):
    """Frozen Protocol for sequencing dispatchers.

    Distinct from the other policy protocols, including
    :class:`CandidatePolicy` and :class:`RankingPolicy`, so the
    umbrella router can hold all of them without structural aliasing.
    """

    name: str
    version: int

    def propose_sequencing(
        self, request: SequencingRequest
    ) -> SequencingSelection: ...


# ---------------------------------------------------------------------------
# Anti-loop gate (consumer-side defence-in-depth).
# ---------------------------------------------------------------------------


def apply_anti_loop_gate(
    *,
    proposed_action: str,
    attempt_index: int,
    commit_reobserve_count: int,
    max_attempts: int,
    max_reobserve_attempts: int,
) -> tuple[str, bool]:
    """Clip a producer proposal to enforce anti-loop bounds.

    Rules (deterministic, side-effect free):

    1. ``REOBSERVE`` once ``commit_reobserve_count >= max_reobserve_attempts``
       is clipped to ``ABORT``: it cannot reobserve any further, so no
       infinite reobserve loop is possible.
    2. Any non-``ABORT`` proposal on the last allowed attempt index
       (``attempt_index >= max_attempts - 1``) is clipped to ``ABORT``:
       a terminal attempt cannot ask for another attempt.

    Returns ``(gated_action, anti_loop_clipped)``.
    """

    if proposed_action not in SEQUENCING_ACTIONS_SET:
        # Defence-in-depth: a misbehaving producer is clipped to abort.
        logger.error(
            "Anti-loop gate clipped an invalid proposed sequencing action %r to %r",
            proposed_action,
            SEQUENCING_ACTION_ABORT,
        )
        return SEQUENCING_ACTION_ABORT, True
    if (
        proposed_action == SEQUENCING_ACTION_REOBSERVE
        and commit_reobserve_count >= max_reobserve_attempts
    ):
        logger.warning(
            "Anti-loop gate clipped %r to %r: %d reobservation(s) already spent of "
            "%d allowed",
            proposed_action,
            SEQUENCING_ACTION_ABORT,
            commit_reobserve_count,
            max_reobserve_attempts,
        )
        return SEQUENCING_ACTION_ABORT, True
    if (
        proposed_action != SEQUENCING_ACTION_ABORT
        and attempt_index >= max_attempts - 1
    ):
        logger.warning(
            "Anti-loop gate clipped %r to %r on the last allowed attempt "
            "(index %d of %d)",
            proposed_action,
            SEQUENCING_ACTION_ABORT,
            attempt_index,
            max_attempts,
        )
        return SEQUENCING_ACTION_ABORT, True
    return proposed_action, False


# ---------------------------------------------------------------------------
# Decision row (per-attempt), emitted by the router for telemetry.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequencingDecision:
    """One per-attempt sequencing telemetry row."""

    attempt_index: int
    state_key: SequencingStateKey
    proposed_action: str
    gated_action: str
    baseline_action: str
    agree_with_baseline: bool
    anti_loop_clipped: bool
    support_count: int
    used_fallback: bool

    def __post_init__(self) -> None:
        for a in (self.proposed_action, self.gated_action, self.baseline_action):
            if a not in SEQUENCING_ACTIONS_SET:
                raise ValueError(
                    f"action {a!r} not in {SEQUENCING_ACTIONS!r}"
                )


# ---------------------------------------------------------------------------
# Hand-authored fallback table (used when a cell has support below
# ``min_support_threshold``).
# ---------------------------------------------------------------------------

#: Conservative fallback chosen per failure class. Slip and collision
#: bias toward physical recovery; drift aborts because the
#: deterministic stack will already have flagged it; deformable,
#: occlusion and empty-air bias toward perception refresh. ``unknown``
#: defaults to reobserve (the cheapest signal-gathering action) and
#: ``none`` proceeds with the next grasp.
DEFAULT_FALLBACK_TABLE: Mapping[str, str] = {
    FAILURE_CLASS_NONE: SEQUENCING_ACTION_GRASP,
    FAILURE_CLASS_UNKNOWN: SEQUENCING_ACTION_REOBSERVE,
    FAILURE_CLASS_SLIP: SEQUENCING_ACTION_RECOVER,
    FAILURE_CLASS_EMPTY_AIR: SEQUENCING_ACTION_REOBSERVE,
    FAILURE_CLASS_COLLISION: SEQUENCING_ACTION_RECOVER,
    FAILURE_CLASS_OCCLUSION: SEQUENCING_ACTION_REOBSERVE,
    FAILURE_CLASS_DEFORMABLE: SEQUENCING_ACTION_REOBSERVE,
    FAILURE_CLASS_DRIFT: SEQUENCING_ACTION_ABORT,
}


def _validate_fallback_table(table: Mapping[str, str]) -> None:
    missing = sorted(FAILURE_CLASSES_SET - set(table.keys()))
    if missing:
        raise ValueError(
            f"fallback table missing entries for failure classes: "
            f"{missing!r}"
        )
    for cls, action in table.items():
        if cls not in FAILURE_CLASSES_SET:
            raise ValueError(
                f"fallback table contains unknown failure class {cls!r}"
            )
        if action not in SEQUENCING_ACTIONS_SET:
            raise ValueError(
                f"fallback table action {action!r} for class {cls!r} "
                f"not in {SEQUENCING_ACTIONS!r}"
            )


# ---------------------------------------------------------------------------
# Lookup-table policy.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LookupTableSequencingPolicy:
    """Deterministic state-to-action policy with empirical-frequency cells.

    Cells with empirical support below :attr:`min_support_threshold`
    are served from :attr:`fallback_table` keyed on the state key's
    failure class.
    """

    # Maps cell_key (as_string_key) to {"action": str, "support": int}.
    lookup_table: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    fallback_table: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_FALLBACK_TABLE)
    )
    min_support_threshold: int = 5
    name: str = "v4_sequencing_baseline_v1"
    version: int = 1
    artifact_dataset_id: str = ""
    artifact_dataset_hash: str = ""

    def __post_init__(self) -> None:
        _validate_fallback_table(self.fallback_table)
        for cell_key, cell in self.lookup_table.items():
            action = cell.get("action") if isinstance(cell, Mapping) else None
            if action not in SEQUENCING_ACTIONS_SET:
                raise ValueError(
                    f"lookup_table cell {cell_key!r} action "
                    f"{action!r} not in {SEQUENCING_ACTIONS!r}"
                )
            support = cell.get("support") if isinstance(cell, Mapping) else None
            if not isinstance(support, int) or support < 0:
                raise ValueError(
                    f"lookup_table cell {cell_key!r} support "
                    f"{support!r} is not a non-negative int"
                )

    @property
    def policy_id(self) -> str:
        return f"{self.name}@{self.version}"

    def propose_sequencing(
        self, request: SequencingRequest
    ) -> SequencingSelection:
        cell_key = request.state_key.as_string_key()
        cell = self.lookup_table.get(cell_key)
        if (
            cell is not None
            and int(cell.get("support", 0)) >= self.min_support_threshold
        ):
            return SequencingSelection(
                policy_id=self.policy_id,
                policy_version=self.version,
                action=str(cell["action"]),
                support_count=int(cell["support"]),
                used_fallback=False,
            )
        action = self.fallback_table[request.state_key.last_failure_class]
        return SequencingSelection(
            policy_id=self.policy_id,
            policy_version=self.version,
            action=action,
            support_count=int(cell.get("support", 0)) if cell else 0,
            used_fallback=True,
        )


# ---------------------------------------------------------------------------
# Artifact loader.
# ---------------------------------------------------------------------------


#: Sequencing-artifact schema version. Bump on any incompatible on-disk
#: JSON change. Carries the ``reward_model``, ``reward_interpretation``,
#: ``dataset_provenance`` and ``fallback_honesty`` stamps.
SEQUENCING_ARTIFACT_SCHEMA_VERSION: int = 2


def load_lookup_table_sequencing_policy(
    path: str | Path,
) -> LookupTableSequencingPolicy:
    """Load a :class:`LookupTableSequencingPolicy` from an artifact.

    Rejects schema-version drift and action / failure-class drift
    so a stale artifact can never be silently loaded.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        blob: Mapping[str, Any] = json.load(f)
    schema_version = int(blob.get("schema_version", 0))
    if schema_version != SEQUENCING_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "LookupTableSequencingPolicy artifact schema_version "
            f"{schema_version} does not match the frozen contract "
            f"{SEQUENCING_ARTIFACT_SCHEMA_VERSION}"
        )
    actions_blob = tuple(blob.get("actions", ()))
    if actions_blob != SEQUENCING_ACTIONS:
        raise ValueError(
            "LookupTableSequencingPolicy artifact 'actions' does not "
            "match the frozen enum"
        )
    failure_classes_blob = tuple(blob.get("failure_classes", ()))
    if failure_classes_blob != FAILURE_CLASSES:
        raise ValueError(
            "LookupTableSequencingPolicy artifact 'failure_classes' "
            "does not match the frozen enum"
        )
    raw_table = blob.get("lookup_table", {}) or {}
    lookup_table: dict[str, dict[str, Any]] = {}
    for key, cell in raw_table.items():
        lookup_table[str(key)] = {
            "action": str(cell["action"]),
            "support": int(cell["support"]),
        }
    fallback_blob = blob.get("fallback_table", {}) or {}
    fallback_table = {str(k): str(v) for k, v in fallback_blob.items()}
    logger.info(
        "Loaded sequencing policy %s@%s from %s: %d lookup cell(s), %d fallback "
        "entry(ies), min_support %s, dataset %s (%s)",
        blob.get("policy_name", "v4_sequencing_baseline_v1"),
        blob.get("policy_version", 1),
        path,
        len(lookup_table),
        len(fallback_table),
        blob.get("min_support_threshold", 5),
        blob.get("dataset_id", "?"),
        str(blob.get("dataset_hash", ""))[:12] or "?",
    )
    return LookupTableSequencingPolicy(
        lookup_table=lookup_table,
        fallback_table=fallback_table,
        min_support_threshold=int(blob.get("min_support_threshold", 5)),
        name=str(blob.get("policy_name", "v4_sequencing_baseline_v1")),
        version=int(blob.get("policy_version", 1)),
        artifact_dataset_id=str(blob.get("dataset_id", "")),
        artifact_dataset_hash=str(blob.get("dataset_hash", "")),
    )


__all__ = (
    "ATTEMPT_INDEX_MAX",
    "COMMIT_REOBSERVE_COUNT_MAX",
    "DEFAULT_FALLBACK_TABLE",
    "FAILURE_CLASSES",
    "FAILURE_CLASSES_SET",
    "FAILURE_CLASS_COLLISION",
    "FAILURE_CLASS_DEFORMABLE",
    "FAILURE_CLASS_DRIFT",
    "FAILURE_CLASS_EMPTY_AIR",
    "FAILURE_CLASS_NONE",
    "FAILURE_CLASS_OCCLUSION",
    "FAILURE_CLASS_SLIP",
    "FAILURE_CLASS_UNKNOWN",
    "LookupTableSequencingPolicy",
    "OUTCOME_LABELS",
    "OUTCOME_LABEL_FAILURE",
    "OUTCOME_LABEL_SUCCESS",
    "SEQUENCING_ACTIONS",
    "SEQUENCING_ACTIONS_SET",
    "SEQUENCING_ACTION_ABORT",
    "SEQUENCING_ACTION_GRASP",
    "SEQUENCING_ACTION_RECOVER",
    "SEQUENCING_ACTION_REOBSERVE",
    "SEQUENCING_ARTIFACT_SCHEMA_VERSION",
    "SequencingDecision",
    "SequencingPolicy",
    "SequencingRequest",
    "SequencingSelection",
    "SequencingStateKey",
    "apply_anti_loop_gate",
    "clip_attempt_index",
    "clip_commit_reobserve_count",
    "hash_sequencing_artifact",
    "load_lookup_table_sequencing_policy",
)
