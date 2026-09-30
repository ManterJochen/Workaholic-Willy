"""Dense-clutter recovery policy.

When a pick attempt fails in a dense scene, the operator may want the
system to take further action before giving up: rescan, try another
target, nudge the target slightly, or, under strict operator sign-off,
agitate a known container. This module ships the typed contract for
that decision and the two physical strategies.

``NEXT_VIEWPOINT`` was merged into ``RESCAN`` on 2026-09-29: with the
viewpoint planners gone nothing moved the camera for it, so it
re-perceived from where the camera stood exactly as ``RESCAN`` does. A
config that still names it is refused at load with that sentence; a
record logged before then keeps its string.

Scope and non-goals
-------------------

* This module is a planner. It returns a typed
  :class:`SceneRecoveryPlan` describing what should happen next; it
  does not drive the robot itself. Motion is executed by
  :func:`execute_recovery_motion`, which runs through the same
  :class:`SafetyPreflight`-aware typed :meth:`RobotArm.move` surface
  the rest of the grasping pipeline uses and refuses to proceed when
  the driver reports any status other than ``EXECUTED``. It drives
  only ``CONTAINER_AGITATE``: the push that ``NUDGE_TARGET`` stands for
  runs inside the pick attempt (:mod:`push_motion`), and the executor
  refuses a nudge before anything moves.
* Recovery is allowed only when the active
  :class:`GraspBehaviorProfile` has a non-empty
  :attr:`recovery_allowed_actions` tuple. The locked contract
  makes ``EASY`` provably free of recovery motion: an easy profile
  has an empty tuple, so every strategy in this module short-circuits
  to :attr:`SceneRecoveryAction.NONE` regardless of the policy.
* Physical recovery actions (:attr:`SceneRecoveryAction.NUDGE_TARGET`,
  :attr:`SceneRecoveryAction.CONTAINER_AGITATE`) require an explicit
  :class:`FixtureEnvelope`. Without one the policy refuses to be
  constructed; this prevents a future config drift from quietly
  enabling shake/push motion outside a known-safe workspace.
* :class:`ContainerAgitateStrategy` is disabled by default: it
  returns :attr:`SceneRecoveryAction.NONE` until the operator explicitly
  opts in via a non-zero :attr:`FixtureEnvelope.max_agitate_amplitude_mm`.
  When armed, the executor drives three bounded waypoints through the
  SafetyPreflight-gated :meth:`RobotArm.move` surface: a there-and-back
  oscillation while :attr:`FixtureEnvelope.agitate_contact_depth_mm` is
  at its ``0.0`` default, and above it a contact redistribute that
  descends, sweeps and retracts. The executor cuts the amplitude to
  :attr:`FixtureEnvelope.max_agitate_amplitude_mm`. It then checks every
  waypoint against the fixture box before it moves any, and refuses the
  whole plan when one lies outside. Any non-EXECUTED :class:`MotionResult`
  aborts, so the safety layer can still veto each move and a config drift
  cannot inject an out-of-box waypoint. A config names
  ``container_agitate`` only on a cell that declares a container (refused
  at load otherwise, 2026-09-29).
* Every ``refused_*`` outcome of :func:`execute_recovery_motion` is
  reached before the arm is commanded (:func:`refused_before_motion`).
  The recovery loop relies on that to fall through to the next action
  after a refusal. ``aborted_motion_failed`` is the one outcome after
  motion.
* Retry orchestration (a failed pick, then a plan, an execution and a
  retried pick) is wired into :class:`AutonomousGraspService` via
  ``run_recovery_loop`` and ``_run_with_recovery``, and it is
  default-off: it only runs when ``robot.grasping.recovery.enabled`` is
  set, otherwise the service short-circuits to a single byte-identical
  pick. The ``RuntimePickService`` path never drives the loop. That loop's
  outer gate is the mode's built-in profile, which no config key widens:
  ``RESCAN`` and ``NEXT_TARGET`` in auto and dense_clutter,
  ``NUDGE_TARGET`` in dense_clutter only, and ``CONTAINER_AGITATE`` in no
  built-in profile. The perception-based push that ``NUDGE_TARGET`` now
  stands for runs inside the pick attempt, not in this executor;
  :func:`push_permitted` is the gate the service hands it. The executor
  refuses a nudge (``refused_push_runs_in_the_pick``) before anything
  moves, so the loop falls through to its next action. The scene-blind
  nudge it used to drive, a TCP offset from wherever the arm stood,
  planned +X by ``SmallNudgeStrategy``, left on 2026-09-29: the owner
  ruled blind moves out on the real cell. The agitate strategy here
  moves the arm only for a caller that widens the profile and builds the
  orchestrator with it (the simulator runner ``run_dense_pick --g6``). The service's ``recovery_policy`` and
  ``recovery_strategy`` slots, which held what
  ``robot.grasping.dense_recovery`` built and no pick consulted, left
  with that block on 2026-09-29, and with them the strategies only it
  built (``ActivePerceptionRecoveryStrategy``,
  ``NextTargetRecoveryStrategy``, ``NoRecoveryStrategy``).

Public surface
--------------

* :class:`SceneRecoveryAction`: typed enum of allowed actions.
* :class:`FixtureEnvelope`: bounded workspace for physical recovery.
* :class:`SceneRecoveryPolicy`: operator-bounded configuration.
* :class:`SceneRecoveryContext`: frozen aggregate of inputs.
* :class:`SceneRecoveryPlan`: frozen aggregate of a single decision.
* :class:`SceneRecoveryReport`: frozen aggregate of an executed plan.
* :class:`SceneRecoveryStrategy`: Protocol.
* :class:`ContainerAgitateStrategy`: the built-in physical strategy.
* :func:`execute_recovery_motion`: typed motion executor for plans
  whose :meth:`SceneRecoveryPlan.is_motion_action` is :data:`True`.
* :func:`refused_before_motion`: whether an executor outcome was reached
  before anything moved.
* :func:`push_permitted`: whether the profile and the policy let a push
  run at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import MotionStatus, RobotArm
from src.robot.grasping.types.feedback import GraspFailureReason
from src.robot.grasping.types.grasp_point import GraspPoint

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.execution.autonomous_grasp import (
        AutonomousGraspOutcome,
        GraspBehaviorProfile,
    )
    from src.robot.grasping.types.perception import PerceptionFrame


__all__ = [
    "ContainerAgitateStrategy",
    "FixtureEnvelope",
    "REFUSED_PUSH_RUNS_IN_THE_PICK",
    "SceneRecoveryAction",
    "SceneRecoveryContext",
    "SceneRecoveryPlan",
    "SceneRecoveryPolicy",
    "SceneRecoveryReport",
    "SceneRecoveryStrategy",
    "execute_recovery_motion",
    "push_permitted",
    "refused_before_motion",
]


class SceneRecoveryAction(StrEnum):
    """Typed recovery action vocabulary.

    Values are stable strings to match
    :attr:`GraspBehaviorProfile.recovery_allowed_actions` exactly (the
    profile uses plain strings to avoid a circular import). The
    profile-level allow-list is the outer gate; the policy-level
    allow-list is the inner gate. Both must include an action for a
    strategy to be allowed to plan it.

    ``NEXT_VIEWPOINT`` (``"next_viewpoint"``) left on 2026-09-29, merged
    into :attr:`RESCAN`, which is what it did once nothing moved the
    camera for it.
    """

    NONE = "none"
    RESCAN = "rescan"
    NEXT_TARGET = "next_target"
    NUDGE_TARGET = "nudge_target"
    CONTAINER_AGITATE = "container_agitate"
    ABORT = "abort"


# Actions that command physical motion of the robot. These must never
# be planned without an explicit :class:`FixtureEnvelope`; the policy
# enforces this at construction time.
_PHYSICAL_ACTIONS: frozenset[SceneRecoveryAction] = frozenset(
    {
        SceneRecoveryAction.NUDGE_TARGET,
        SceneRecoveryAction.CONTAINER_AGITATE,
    }
)


@dataclass(frozen=True, slots=True)
class FixtureEnvelope:
    """Bounded workspace inside which physical recovery may move.

    The envelope is an axis-aligned box in :attr:`Frame.BASE`,
    expressed via its centre and half-extents in millimetres. Every
    waypoint the executor commands as part of a recovery plan must lie
    inside this box. A plan with one outside is refused before anything
    moves, with :class:`SceneRecoveryReport.outcome` set to
    ``"refused_envelope_violation"``.

    Attributes
    ----------
    center_mm
        ``(x, y, z)`` envelope centre in :attr:`Frame.BASE`.
    half_extents_mm
        ``(dx, dy, dz)`` non-negative half extents. The box is
        ``[center - half, center + half]`` per axis.
    max_nudge_mm
        The longest push the cell allows,
        ``recovery.fixture.max_nudge_mm``: 50 mm by default and at most
        (owner, 2026-09-29; it was 5 mm, less than the Hand-E finger's
        10.8 mm thickness, so no push could open room for a finger). The
        push runs inside the pick attempt and settles its distance with
        :func:`~src.robot.grasping.recovery.push_planner.resolve_push_distance`
        (``recovery.fixture.push_distance_mm``, 30 mm, when nobody asks);
        this executor drives no nudge.
    max_agitate_amplitude_mm
        Largest permissible single-step agitation amplitude; the executor
        cuts a larger planned amplitude to it. Default ``0.0`` keeps
        :class:`ContainerAgitateStrategy` disabled until an operator
        explicitly raises this number.
    """

    center_mm: tuple[float, float, float]
    half_extents_mm: tuple[float, float, float]
    max_nudge_mm: float = 50.0
    max_agitate_amplitude_mm: float = 0.0
    # When >0 the CONTAINER_AGITATE executor runs a contact redistribute instead of the air-shake
    # oscillation: it descends ``agitate_contact_depth_mm`` toward the object layer and sweeps the approach
    # corridor (a directed +X push that contacts + clears a movable blocker) so a re-pick succeeds. Default
    # 0.0 keeps the there-and-back oscillation. ``agitate_sweep_offset_mm`` offsets the sweep start along +X
    # off the target so the descent clears the target itself (uses only the known approach corridor, no
    # blocker ground-truth).
    agitate_contact_depth_mm: float = 0.0
    agitate_sweep_offset_mm: float = 0.0

    def __post_init__(self) -> None:
        if len(self.center_mm) != 3:
            raise ValueError(
                "center_mm must have exactly 3 components; "
                f"got {self.center_mm!r}"
            )
        if len(self.half_extents_mm) != 3:
            raise ValueError(
                "half_extents_mm must have exactly 3 components; "
                f"got {self.half_extents_mm!r}"
            )
        if any(h < 0.0 for h in self.half_extents_mm):
            raise ValueError(
                "half_extents_mm must be non-negative on every axis; "
                f"got {self.half_extents_mm!r}"
            )
        if self.max_nudge_mm < 0.0:
            raise ValueError(
                f"max_nudge_mm must be non-negative; got {self.max_nudge_mm}"
            )
        if self.max_agitate_amplitude_mm < 0.0:
            raise ValueError(
                "max_agitate_amplitude_mm must be non-negative; "
                f"got {self.max_agitate_amplitude_mm}"
            )
        if self.agitate_contact_depth_mm < 0.0:
            raise ValueError(
                "agitate_contact_depth_mm must be non-negative; "
                f"got {self.agitate_contact_depth_mm}"
            )

    def contains(self, position_mm: Sequence[float]) -> bool:
        """Return :data:`True` iff ``position_mm`` lies inside the box."""

        if len(position_mm) != 3:
            return False
        for axis, value in enumerate(position_mm):
            lo = self.center_mm[axis] - self.half_extents_mm[axis]
            hi = self.center_mm[axis] + self.half_extents_mm[axis]
            if value < lo or value > hi:
                return False
        return True


@dataclass(frozen=True, slots=True)
class SceneRecoveryPolicy:
    """Bounded operator configuration for scene recovery.

    Attributes
    ----------
    enabled
        Master switch. When :data:`False` every strategy in this
        module returns :attr:`SceneRecoveryAction.NONE`.
    allowed_actions
        Inner allow-list. Strategies refuse to plan actions outside
        this set. The outer allow-list lives on the active
        :class:`GraspBehaviorProfile`; both must include an action for
        it to be planned. Defaults to the cheapest non-motion action
        (``RESCAN``) so a freshly enabled policy without a configured
        fixture is still safe.
    max_recovery_actions
        Upper bound on the number of recovery actions the caller
        will execute before giving up. Strategies honour this via
        :attr:`SceneRecoveryContext.history`: when the history is
        already at the bound, every strategy returns ``NONE``.
    fixture
        Required for any physical action (``NUDGE_TARGET`` /
        ``CONTAINER_AGITATE``). Construction fails when a physical
        action sits in :attr:`allowed_actions` without a fixture.
    """

    enabled: bool = False
    allowed_actions: tuple[SceneRecoveryAction, ...] = (
        SceneRecoveryAction.RESCAN,
    )
    max_recovery_actions: int = 2
    fixture: Optional[FixtureEnvelope] = None
    #: Per-action attempt cap. An empty mapping means only
    #: ``max_recovery_actions`` applies.
    per_action_budget: Mapping[SceneRecoveryAction, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    #: Modes for which the orchestrator is permitted to drive recovery.
    #: Easy is permanently excluded by default. ``dense_autonomous`` left the
    #: default with the mode on 2026-09-29.
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
        physical_in_use = seen & _PHYSICAL_ACTIONS
        if physical_in_use and self.fixture is None:
            raise ValueError(
                "physical recovery actions require a FixtureEnvelope; "
                f"got {sorted(a.value for a in physical_in_use)} without "
                "fixture"
            )
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
    """Frozen aggregate of everything a strategy needs to plan.

    Fields are :data:`None` / empty when the upstream service could
    not collect that signal. Strategies degrade gracefully: missing
    inputs yield :attr:`SceneRecoveryAction.NONE` rather than a
    crash.
    """

    profile: "GraspBehaviorProfile"
    policy: SceneRecoveryPolicy
    last_outcome: "AutonomousGraspOutcome"
    failure_reasons: tuple[GraspFailureReason, ...] = ()
    last_frame: Optional["PerceptionFrame"] = None
    # `target_identity` left on 2026-09-29 with the two-scan refinement that defined its type. No
    # caller ever set it and no strategy read it.
    last_grasp: Optional[GraspPoint] = None
    current_tcp: Optional[Pose] = None
    history: tuple[SceneRecoveryAction, ...] = ()
    #: Richer per-(action, class) history for anti-loop bookkeeping.
    #: Strategies must not introspect this tuple; only the orchestrator in
    #: :mod:`src.robot.grasping.recovery.orchestrator` consumes it.
    typed_history: tuple[Any, ...] = ()
    #: When :data:`True` the orchestrator reorders
    #: each failure-class action list so that the perception action
    #: (``RESCAN``) is tried before the others. Group order across
    #: failure classes is preserved.
    aggressive_recovery_bias: bool = False


@dataclass(frozen=True, slots=True)
class SceneRecoveryPlan:
    """Frozen aggregate result of a single planning call.

    Attributes
    ----------
    action
        Typed action to take. :attr:`SceneRecoveryAction.NONE` means
        the strategy declines to act (gate, exhaustion, missing
        signals); the caller treats this as "no recovery available".
    reason
        Short machine-readable string explaining the decision. Stable
        keys; safe to log to JSONL.
    nudge_offset_mm
        ``(dx, dy, dz)`` offset for :attr:`SceneRecoveryAction.NUDGE_TARGET`.
        Nothing drives it any more: the push runs inside the pick attempt,
        and the executor refuses a nudge plan whatever it carries. Kept so
        a plan and its telemetry row keep their shape. ``None`` for other
        actions.
    agitate_amplitude_mm
        Amplitude for :attr:`SceneRecoveryAction.CONTAINER_AGITATE`.
        ``0.0`` for other actions or a disabled fixture.
    telemetry
        Free-form key/value bag. JSON-serializable values only.
    """

    action: SceneRecoveryAction
    reason: str = ""
    nudge_offset_mm: Optional[tuple[float, float, float]] = None
    agitate_amplitude_mm: float = 0.0
    telemetry: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_motion_action(self) -> bool:
        """Whether executing this plan commands robot motion."""

        return self.action in _PHYSICAL_ACTIONS


@dataclass(frozen=True, slots=True)
class SceneRecoveryReport:
    """Frozen aggregate of the result of executing a plan.

    Attributes
    ----------
    plan
        The :class:`SceneRecoveryPlan` that was supplied to the
        executor.
    executed
        :data:`True` iff motion (or a no-op completion for non-motion
        plans) ran to completion. Always :data:`False` for
        :attr:`SceneRecoveryAction.NONE`.
    outcome
        Stable string: one of ``"completed"``, ``"skipped_no_action"``,
        ``"refused_envelope_violation"``, ``"refused_no_tcp"``,
        ``"refused_push_runs_in_the_pick"``,
        ``"refused_agitate_disabled"``, ``"refused_no_typed_move"``,
        ``"refused_unknown_action"``, ``"aborted_motion_failed"``. Every
        ``refused_*`` is reached before the arm is commanded;
        ``aborted_motion_failed`` is the only outcome after motion.
    telemetry
        Free-form key/value bag. JSON-serializable values only.
    """

    plan: SceneRecoveryPlan
    executed: bool
    outcome: str
    telemetry: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class SceneRecoveryStrategy(Protocol):
    """Vendor-neutral recovery strategy."""

    def plan(self, context: SceneRecoveryContext) -> SceneRecoveryPlan:
        ...


# ---------------------------------------------------------------------------
# Shared gates
# ---------------------------------------------------------------------------


def _profile_permits(profile: "GraspBehaviorProfile", action: SceneRecoveryAction) -> bool:
    """Outer-gate check against the profile allow-list (plain-string compare avoids a circular import)."""

    return action.value in tuple(profile.recovery_allowed_actions)


def _gate(
    context: SceneRecoveryContext, action: SceneRecoveryAction
) -> Optional[SceneRecoveryPlan]:
    """Common gating logic; returns a NONE plan when blocked, else None."""

    policy = context.policy
    if not policy.enabled:
        return SceneRecoveryPlan(
            action=SceneRecoveryAction.NONE,
            reason="policy_disabled",
            telemetry={"requested_action": str(action)},
        )
    if not _profile_permits(context.profile, action):
        return SceneRecoveryPlan(
            action=SceneRecoveryAction.NONE,
            reason="profile_disallows_action",
            telemetry={
                "requested_action": str(action),
                "profile_recovery_allowed_actions": tuple(
                    context.profile.recovery_allowed_actions
                ),
            },
        )
    if not policy.permits(action):
        return SceneRecoveryPlan(
            action=SceneRecoveryAction.NONE,
            reason="policy_disallows_action",
            telemetry={
                "requested_action": str(action),
                "policy_allowed_actions": tuple(
                    a.value for a in policy.allowed_actions
                ),
            },
        )
    if len(context.history) >= policy.max_recovery_actions:
        return SceneRecoveryPlan(
            action=SceneRecoveryAction.NONE,
            reason="recovery_budget_exhausted",
            telemetry={
                "history": tuple(a.value for a in context.history),
                "max_recovery_actions": policy.max_recovery_actions,
            },
        )
    return None


def push_permitted(profile: "GraspBehaviorProfile", policy: SceneRecoveryPolicy) -> bool:
    """Whether the mode's profile and the operator's policy let a push (``NUDGE_TARGET``) run at all.

    The push runs inside the pick attempt, not in :func:`execute_recovery_motion`; this is the gate the service
    hands it. Every one of these must hold, and none of them asks a person:

    * the policy is enabled and the mode is in its ``apply_modes``;
    * the mode's built-in profile lists ``nudge_target`` (dense_clutter only);
    * ``recovery.allowed_actions`` lists it, with a fixture declared;
    * neither ``max_recovery_actions`` nor its ``per_action_budget`` is zero.

    The push budgets (1 per part, 2 per pick, 5 per campaign) and the push's own preconditions come on top.
    """

    action = SceneRecoveryAction.NUDGE_TARGET
    if not policy.permits(action) or policy.fixture is None:
        return False
    if profile.mode.value not in policy.apply_modes or not _profile_permits(profile, action):
        return False
    if policy.max_recovery_actions <= 0:
        return False
    budget = policy.per_action_budget.get(action)
    return budget is None or int(budget) > 0


#: Every ``refused_*`` outcome is reached before anything moved. Neither the executor nor the recovery loop
#: may use this prefix for an outcome reached after a move was commanded.
_REFUSED_PREFIX = "refused_"


def refused_before_motion(outcome: str) -> bool:
    """True for an outcome reached before the arm was commanded: the loop falls through to its next action."""

    return str(outcome).startswith(_REFUSED_PREFIX)


#: The executor's answer to a ``NUDGE_TARGET`` plan: the push runs inside the pick attempt, never here, so the
#: loop's nudge is refused before anything moves and the loop falls through to its next action.
REFUSED_PUSH_RUNS_IN_THE_PICK = "refused_push_runs_in_the_pick"


@dataclass(frozen=True, slots=True)
class ContainerAgitateStrategy:
    """Container-agitate strategy (disabled by default).

    Plans a :attr:`SceneRecoveryAction.CONTAINER_AGITATE` only when the
    policy permits it and the fixture has a non-zero
    :attr:`FixtureEnvelope.max_agitate_amplitude_mm`; otherwise it returns
    :attr:`SceneRecoveryAction.NONE` (off by default). When armed, the
    executor drives three bounded waypoints through the
    SafetyPreflight-gated :meth:`RobotArm.move` surface: a there-and-back
    oscillation while :attr:`FixtureEnvelope.agitate_contact_depth_mm` is
    at its ``0.0`` default, and above it a contact redistribute that
    descends, sweeps and retracts. The executor cuts the amplitude to
    :attr:`FixtureEnvelope.max_agitate_amplitude_mm`, checks every
    waypoint against the fixture box before it moves any (one outside
    refuses the whole plan; nothing is moved into the box), and aborts on
    any non-EXECUTED motion.
    """

    def plan(self, context: SceneRecoveryContext) -> SceneRecoveryPlan:
        blocked = _gate(context, SceneRecoveryAction.CONTAINER_AGITATE)
        if blocked is not None:
            return blocked
        fixture = context.policy.fixture
        if fixture is None:  # pragma: no cover (defensive)
            return SceneRecoveryPlan(
                action=SceneRecoveryAction.NONE,
                reason="no_fixture_envelope",
                telemetry={"strategy": "container_agitate"},
            )
        if fixture.max_agitate_amplitude_mm <= 0.0:
            return SceneRecoveryPlan(
                action=SceneRecoveryAction.NONE,
                reason="agitate_amplitude_disabled",
                telemetry={
                    "strategy": "container_agitate",
                    "max_agitate_amplitude_mm": fixture.max_agitate_amplitude_mm,
                },
            )
        return SceneRecoveryPlan(
            action=SceneRecoveryAction.CONTAINER_AGITATE,
            reason="agitate_amplitude_configured",
            agitate_amplitude_mm=float(fixture.max_agitate_amplitude_mm),
            telemetry={
                "strategy": "container_agitate",
                "max_agitate_amplitude_mm": fixture.max_agitate_amplitude_mm,
            },
        )


def _motion_executed(status: object) -> bool:
    """True iff a typed-move ``result.status`` is :attr:`MotionStatus.EXECUTED` (identity check)."""

    return status is MotionStatus.EXECUTED


def execute_recovery_motion(
    *,
    arm: RobotArm,
    plan: SceneRecoveryPlan,
    policy: SceneRecoveryPolicy,
    current_tcp: Optional[Pose] = None,
) -> SceneRecoveryReport:
    """Drive ``arm`` through the motion side of ``plan``.

    * For :attr:`SceneRecoveryAction.NONE` returns
      ``executed=False, outcome="skipped_no_action"``.
    * For non-motion actions (``RESCAN`` / ``NEXT_TARGET`` / ``ABORT``) returns
      ``executed=True, outcome="completed"`` without touching the arm;
      the caller is responsible for re-acquiring perception or
      switching target.
    * For :attr:`SceneRecoveryAction.NUDGE_TARGET` refuses with
      :data:`REFUSED_PUSH_RUNS_IN_THE_PICK` and touches nothing: the push
      runs inside the pick attempt (``push_motion.execute_push``), where
      the part and its neighbours are known. The scene-blind offset this
      executor used to drive left on 2026-09-29.
    * For :attr:`SceneRecoveryAction.CONTAINER_AGITATE` drives three
      bounded waypoints via :meth:`RobotArm.move`, the amplitude cut to
      :attr:`FixtureEnvelope.max_agitate_amplitude_mm`: a there-and-back
      oscillation (``+amplitude``/``-amplitude`` along x, then back to
      the start TCP) while
      :attr:`FixtureEnvelope.agitate_contact_depth_mm` is ``0.0``,
      otherwise a descend, sweep ``+amplitude`` along +X and retract
      that ends offset from the start TCP. Every waypoint is checked
      against the fixture box before the first one moves. Refuses
      with ``"refused_no_tcp"`` / ``"refused_agitate_disabled"`` (no
      fixture or amplitude<=0) / ``"refused_envelope_violation"``, and
      aborts on any non-EXECUTED status with ``"aborted_motion_failed"``.

    Every refusal is reached before the arm is commanded.
    """

    if plan.action is SceneRecoveryAction.NONE:
        return SceneRecoveryReport(
            plan=plan,
            executed=False,
            outcome="skipped_no_action",
            telemetry={"reason": plan.reason},
        )
    if plan.action in (
        SceneRecoveryAction.RESCAN,
        SceneRecoveryAction.NEXT_TARGET,
        SceneRecoveryAction.ABORT,
    ):
        return SceneRecoveryReport(
            plan=plan,
            executed=True,
            outcome="completed",
            telemetry={"action": str(plan.action)},
        )
    if plan.action is SceneRecoveryAction.CONTAINER_AGITATE:
        # A bounded three-waypoint motion that redistributes clutter, driven through the
        # SafetyPreflight-gated arm.move surface. The amplitude is cut to the fixture's limit and every
        # waypoint is checked against the fixture envelope before the first one moves (defence-in-depth), so
        # a hand-built plan or config drift cannot inject an out-of-box waypoint, a refusal always means
        # nothing moved, and the safety layer can still veto each move.
        if current_tcp is None:
            return SceneRecoveryReport(
                plan=plan,
                executed=False,
                outcome="refused_no_tcp",
                telemetry={"action": str(plan.action)},
            )
        fixture = policy.fixture
        requested = float(plan.agitate_amplitude_mm)
        amplitude = requested if fixture is None else min(requested, float(fixture.max_agitate_amplitude_mm))
        # A NaN amplitude would pass every envelope comparison (each one is False), so it is refused here.
        if fixture is None or not np.isfinite(amplitude) or amplitude <= 0.0:
            return SceneRecoveryReport(
                plan=plan,
                executed=False,
                outcome="refused_agitate_disabled",
                telemetry={"action": str(plan.action), "amplitude_mm": amplitude,
                           "requested_amplitude_mm": requested},
            )
        typed_move = getattr(arm, "move", None)
        if not callable(typed_move):
            return SceneRecoveryReport(
                plan=plan,
                executed=False,
                outcome="refused_no_typed_move",
                telemetry={"action": str(plan.action)},
            )
        start = np.asarray(current_tcp.position_mm, dtype=np.float64)
        quat = np.asarray(current_tcp.quaternion_xyzw, dtype=np.float64).copy()
        depth = float(getattr(fixture, "agitate_contact_depth_mm", 0.0))
        if depth > 0.0:
            # Contact redistribute: descend toward the object layer (offset +X off the target so the
            # descent clears the target), then sweep +X through the approach corridor as a directed
            # push that contacts and clears a movable blocker, then retract up clear of the clutter.
            # This uses only the known +X approach corridor, with no blocker ground-truth. Each
            # waypoint is still envelope-clamped.
            sweep_offset = float(getattr(fixture, "agitate_sweep_offset_mm", 0.0))
            deltas = (
                (sweep_offset, 0.0, -depth),
                (sweep_offset + amplitude, 0.0, -depth),
                (sweep_offset + amplitude, 0.0, 0.0),
            )
        else:
            # Air-shake oscillation: +amplitude along x, then -amplitude, then back to the start TCP.
            deltas = ((amplitude, 0.0, 0.0), (-amplitude, 0.0, 0.0), (0.0, 0.0, 0.0))
        destinations = [start + np.asarray(delta, dtype=np.float64) for delta in deltas]
        for index, destination in enumerate(destinations):
            if not fixture.contains(destination.tolist()):
                return SceneRecoveryReport(
                    plan=plan,
                    executed=False,
                    outcome="refused_envelope_violation",
                    telemetry={
                        "action": str(plan.action),
                        "waypoint_index": index,
                        "destination_mm": tuple(float(x) for x in destination),
                    },
                )
        for index, destination in enumerate(destinations):
            target_pose = Pose(
                position_mm=destination,
                quaternion_xyzw=quat.copy(),
                frame=Frame.BASE,
                label="recovery_agitate",
            )
            result = typed_move(target_pose)
            status = getattr(result, "status", None)
            executed = _motion_executed(status)
            if not executed:
                return SceneRecoveryReport(
                    plan=plan,
                    executed=False,
                    outcome="aborted_motion_failed",
                    telemetry={
                        "action": str(plan.action),
                        "waypoint_index": index,
                        "motion_status": str(status),
                    },
                )
        return SceneRecoveryReport(
            plan=plan,
            executed=True,
            outcome="completed",
            telemetry={
                "action": str(plan.action),
                "waypoints": len(deltas),
                "amplitude_mm": amplitude,
                "requested_amplitude_mm": requested,
                "clamped": amplitude < requested,
                "agitate_mode": "contact_redistribute" if depth > 0.0 else "oscillation",
                "contact_depth_mm": depth,
            },
        )
    if plan.action is SceneRecoveryAction.NUDGE_TARGET:
        # The push runs inside the pick attempt, where the part, its neighbours and the keep-out are known
        # (push_motion.execute_push). The scene-blind offset this branch used to drive from wherever the arm
        # stood left on 2026-09-29 (owner: no blind moves on the real cell), so a nudge here is refused before
        # anything moves and the loop falls through to its next action.
        return SceneRecoveryReport(
            plan=plan,
            executed=False,
            outcome=REFUSED_PUSH_RUNS_IN_THE_PICK,
            telemetry={"action": str(plan.action)},
        )
    # Unknown action: refuse rather than silently completing.
    return SceneRecoveryReport(  # pragma: no cover (defensive)
        plan=plan,
        executed=False,
        outcome="refused_unknown_action",
        telemetry={"action": str(plan.action)},
    )
