"""The recovery a pick runs after a failed attempt: the typed contract.

When a pick attempt fails, the cell may look again (``RESCAN``) before it gives up, and in ``dense_clutter`` the push
of a boxed-in part (``NUDGE_TARGET``) runs inside the pick attempt itself
(:mod:`~src.robot.grasping.recovery.push_motion`), where the part, its neighbours and the keep-out are known. This
module ships the typed vocabulary and the operator-bounded policy both read; the loop that drives a rescan is
:mod:`~src.robot.grasping.recovery.orchestrator`.

``NEXT_VIEWPOINT`` was merged into ``RESCAN`` on 2026-09-29: with the viewpoint planners gone nothing moved the camera
for it, so it re-perceived from where the camera stood exactly as ``RESCAN`` does. ``NEXT_TARGET``, the rescan that
skipped the part a pick failed on, and ``CONTAINER_AGITATE`` with its fixture envelope left on 2026-10-16 (the owner's
cleanup): the owner's cell named neither, and no cell declared the container the agitation needed. A record logged
before then keeps its strings; a campaign that asks for one is refused with the sentence of
``src/config/schema/_removed.py``.

Scope and non-goals
-------------------

* Recovery is allowed only when the active :class:`GraspBehaviorProfile` has a non-empty
  :attr:`recovery_allowed_actions` tuple. The locked contract makes ``EASY`` provably free of recovery motion: an easy
  profile has an empty tuple, so nothing here plans for it, whatever the policy says.
* The recovery loop plans ``RESCAN`` alone, which moves nothing itself: the next pick re-perceives the scene. The push
  is never planned there; :func:`push_permitted` is the gate the service hands the pick attempt instead.
* A cell built from config runs the owner's fixed recovery (``CELL_RECOVERY`` in
  ``src/robot/execution/autonomous_grasp/config.py``): a rescan once and a push twice a pick, at most two actions, in
  ``auto`` and ``dense_clutter``.
* Every ``refused_*`` outcome a push reports is reached before the arm is commanded (:func:`refused_before_motion`).

Public surface
--------------

* :class:`SceneRecoveryAction`: typed enum of allowed actions.
* :class:`SceneRecoveryPolicy`: operator-bounded configuration.
* :class:`SceneRecoveryContext`: frozen aggregate of inputs.
* :class:`SceneRecoveryPlan`: frozen aggregate of a single decision.
* :func:`refused_before_motion`: whether an outcome was reached before anything moved.
* :func:`push_permitted`: whether the profile and the policy let a push run at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Mapping

from src.robot.grasping.types.feedback import GraspFailureReason

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.execution.autonomous_grasp import (
        AutonomousGraspOutcome,
        GraspBehaviorProfile,
    )


__all__ = [
    "SceneRecoveryAction",
    "SceneRecoveryContext",
    "SceneRecoveryPlan",
    "SceneRecoveryPolicy",
    "push_permitted",
    "refused_before_motion",
]


class SceneRecoveryAction(StrEnum):
    """Typed recovery action vocabulary.

    Values are stable strings to match :attr:`GraspBehaviorProfile.recovery_allowed_actions` exactly (the profile uses
    plain strings to avoid a circular import). The profile-level allow-list is the outer gate; the policy-level
    allow-list is the inner gate. Both must include an action for it to be planned.

    ``NEXT_VIEWPOINT`` (``"next_viewpoint"``) left on 2026-09-29, merged into :attr:`RESCAN`; ``NEXT_TARGET``
    (``"next_target"``) and ``CONTAINER_AGITATE`` (``"container_agitate"``) left on 2026-10-16, and ``ABORT`` with them,
    which nothing planned.
    """

    NONE = "none"
    RESCAN = "rescan"
    NUDGE_TARGET = "nudge_target"


@dataclass(frozen=True, slots=True)
class SceneRecoveryPolicy:
    """Bounded operator configuration for scene recovery.

    Attributes
    ----------
    enabled
        Master switch. When :data:`False` the loop plans nothing and no pick pushes.
    allowed_actions
        Inner allow-list. The outer allow-list lives on the active :class:`GraspBehaviorProfile`; both must include an
        action for it to be planned. Defaults to ``RESCAN``, the action that moves nothing.
    max_recovery_actions
        Upper bound on the number of recovery actions one pick runs before giving up. The loop honours it through
        :attr:`SceneRecoveryContext.history`.
    per_action_budget
        Per-action attempt cap. An empty mapping means only ``max_recovery_actions`` applies.
    apply_modes
        Modes for which the orchestrator is permitted to drive recovery. Easy is permanently excluded by default.
    """

    enabled: bool = False
    allowed_actions: tuple[SceneRecoveryAction, ...] = (
        SceneRecoveryAction.RESCAN,
    )
    max_recovery_actions: int = 2
    per_action_budget: Mapping[SceneRecoveryAction, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    apply_modes: tuple[str, ...] = (
        "auto",
        "dense_clutter",
    )

    def __post_init__(self) -> None:
        if self.max_recovery_actions < 0:
            raise ValueError(
                "max_recovery_actions must be non-negative; "
                f"got {self.max_recovery_actions}"
            )
        seen: set[SceneRecoveryAction] = set()
        for action in self.allowed_actions:
            if not isinstance(action, SceneRecoveryAction):
                raise ValueError(
                    "allowed_actions entries must be SceneRecoveryAction "
                    f"members; got {action!r}"
                )
            if action in seen:
                raise ValueError(
                    f"allowed_actions contains duplicate entry {action}"
                )
            seen.add(action)
        if SceneRecoveryAction.NONE in seen:
            raise ValueError(
                "SceneRecoveryAction.NONE must not appear in allowed_actions; "
                "it is the typed 'do nothing' sentinel"
            )
        for key, val in dict(self.per_action_budget).items():
            if not isinstance(key, SceneRecoveryAction):
                raise ValueError(
                    "per_action_budget keys must be SceneRecoveryAction "
                    f"members; got {key!r}"
                )
            if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                raise ValueError(
                    "per_action_budget values must be non-negative ints; "
                    f"got {key.value}={val!r}"
                )
        for mode in self.apply_modes:
            if not isinstance(mode, str) or not mode:
                raise ValueError(
                    "apply_modes entries must be non-empty strings; "
                    f"got {mode!r}"
                )

    def permits(self, action: SceneRecoveryAction) -> bool:
        """Return :data:`True` iff ``action`` is permitted by this policy."""

        if not self.enabled:
            return False
        return action in self.allowed_actions


@dataclass(frozen=True, slots=True)
class SceneRecoveryContext:
    """Frozen aggregate of everything the orchestrator needs to plan."""

    profile: "GraspBehaviorProfile"
    policy: SceneRecoveryPolicy
    last_outcome: "AutonomousGraspOutcome"
    failure_reasons: tuple[GraspFailureReason, ...] = ()
    history: tuple[SceneRecoveryAction, ...] = ()
    #: Richer per-(action, class) history for anti-loop bookkeeping, which only the orchestrator in
    #: :mod:`src.robot.grasping.recovery.orchestrator` reads.
    typed_history: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class SceneRecoveryPlan:
    """Frozen aggregate result of a single planning call.

    Attributes
    ----------
    action
        Typed action to take. :attr:`SceneRecoveryAction.NONE` means no recovery is planned; the caller treats this as
        "no recovery available".
    reason
        Short machine-readable string explaining the decision. Stable keys; safe to log to JSONL.
    telemetry
        Free-form key/value bag. JSON-serializable values only.
    """

    action: SceneRecoveryAction
    reason: str = ""
    telemetry: Mapping[str, Any] = field(default_factory=dict)


def _profile_permits(profile: "GraspBehaviorProfile", action: SceneRecoveryAction) -> bool:
    """Outer-gate check against the profile allow-list (plain-string compare avoids a circular import)."""

    return action.value in tuple(profile.recovery_allowed_actions)


def push_permitted(profile: "GraspBehaviorProfile", policy: SceneRecoveryPolicy) -> bool:
    """Whether the mode's profile and the operator's policy let a push (``NUDGE_TARGET``) run at all.

    The push runs inside the pick attempt; this is the gate the service hands it. Every one of these must hold, and
    none of them asks a person:

    * the policy is enabled and the mode is in its ``apply_modes``;
    * the mode's built-in profile lists ``nudge_target`` (dense_clutter only);
    * the policy's ``allowed_actions`` lists it;
    * neither ``max_recovery_actions`` nor its ``per_action_budget`` is zero.

    The automatic push box bounds where the part lands. The push budgets (1 per part, 2 per pick, 5 per campaign) and
    the push's own preconditions come on top.
    """

    action = SceneRecoveryAction.NUDGE_TARGET
    if not policy.permits(action):
        return False
    if profile.mode.value not in policy.apply_modes or not _profile_permits(profile, action):
        return False
    if policy.max_recovery_actions <= 0:
        return False
    budget = policy.per_action_budget.get(action)
    return budget is None or int(budget) > 0


#: Every ``refused_*`` outcome is reached before anything moved. Nothing may use this prefix for an outcome reached
#: after a move was commanded.
_REFUSED_PREFIX = "refused_"


def refused_before_motion(outcome: str) -> bool:
    """True for an outcome reached before the arm was commanded."""

    return str(outcome).startswith(_REFUSED_PREFIX)
