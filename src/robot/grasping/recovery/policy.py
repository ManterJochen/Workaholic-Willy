"""Dense-clutter recovery policy.

When a pick attempt fails in a dense scene, the operator may want the
system to take further action before giving up: rescan, move the
camera to a new viewpoint, nudge the target slightly, or, under
strict operator sign-off, agitate a known container. This module
ships the typed contract for that decision and a small family of
default strategies.

Scope and non-goals
-------------------

* This module is a planner. It returns a typed
  :class:`SceneRecoveryPlan` describing what should happen next; it
  does not drive the robot itself. Motion is executed by
  :func:`execute_recovery_motion`, which runs through the same
  :class:`SafetyPreflight`-aware typed :meth:`RobotArm.move` surface
  the rest of the grasping pipeline uses and refuses to proceed when
  the driver reports any status other than ``EXECUTED``.
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
  enabling shake/nudge motion outside a known-safe workspace.
* :class:`ContainerAgitateStrategy` is disabled by default: it
  returns :attr:`SceneRecoveryAction.NONE` until the operator explicitly
  opts in via a non-zero :attr:`FixtureEnvelope.max_agitate_amplitude_mm`.
  When armed, the executor drives three bounded waypoints through the
  SafetyPreflight-gated :meth:`RobotArm.move` surface: a there-and-back
  oscillation while :attr:`FixtureEnvelope.agitate_contact_depth_mm` is
  at its ``0.0`` default, and above it a contact redistribute that
  descends, sweeps and retracts. Every waypoint is re-clamped against
  the fixture envelope and any non-EXECUTED :class:`MotionResult`
  aborts, so the safety layer can still veto each move and a config
  drift cannot inject an out-of-box waypoint.
* Retry orchestration (a failed pick, then a plan, an execution and a
  retried pick) is wired into :class:`AutonomousGraspService` via
  ``run_recovery_loop`` and ``_run_with_recovery``, and it is
  default-off: it only runs when ``robot.grasping.recovery.enabled`` is
  set, otherwise the service short-circuits to a single byte-identical
  pick. The ``recovery_policy`` and ``recovery_strategy`` slots carry the
  opt-in machinery. The ``RuntimePickService`` path never drives the loop.

Public surface
--------------

* :class:`SceneRecoveryAction`: typed enum of allowed actions.
* :class:`FixtureEnvelope`: bounded workspace for physical recovery.
* :class:`SceneRecoveryPolicy`: operator-bounded configuration.
* :class:`SceneRecoveryContext`: frozen aggregate of inputs.
* :class:`SceneRecoveryPlan`: frozen aggregate of a single decision.
* :class:`SceneRecoveryReport`: frozen aggregate of an executed plan.
* :class:`SceneRecoveryStrategy`: Protocol.
* :class:`NoRecoveryStrategy`, :class:`ActivePerceptionRecoveryStrategy`,
  :class:`NextTargetRecoveryStrategy`, :class:`SmallNudgeStrategy`,
  :class:`ContainerAgitateStrategy`: built-ins.
* :func:`execute_recovery_motion`: typed motion executor for plans
  whose :meth:`SceneRecoveryPlan.is_motion_action` is :data:`True`.
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
from src.robot.grasping.closed_loop.refinement import TargetIdentity

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.execution.autonomous_grasp import (
        AutonomousGraspOutcome,
        GraspBehaviorProfile,
    )
    from src.robot.grasping.types.perception import PerceptionFrame


__all__ = [
    "ActivePerceptionRecoveryStrategy",
    "ContainerAgitateStrategy",
    "FixtureEnvelope",
    "NextTargetRecoveryStrategy",
    "NoRecoveryStrategy",
    "SceneRecoveryAction",
    "SceneRecoveryContext",
    "SceneRecoveryPlan",
    "SceneRecoveryPolicy",
    "SceneRecoveryReport",
    "SceneRecoveryStrategy",
    "SmallNudgeStrategy",
    "execute_recovery_motion",
]


class SceneRecoveryAction(StrEnum):
    """Typed recovery action vocabulary.

    Values are stable strings to match
    :attr:`GraspBehaviorProfile.recovery_allowed_actions` exactly (the
    profile uses plain strings to avoid a circular import). The
    profile-level allow-list is the outer gate; the policy-level
    allow-list is the inner gate. Both must include an action for a
    strategy to be allowed to plan it.
    """

    NONE = "none"
    RESCAN = "rescan"
    NEXT_VIEWPOINT = "next_viewpoint"
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
    expressed via its centre and half-extents in millimetres. Any
    motion the executor commands as part of a recovery plan is
    clamped against this box; a plan that would step outside is
    refused with :class:`SceneRecoveryReport.outcome` set to
    ``"refused_envelope_violation"``.

    Attributes
    ----------
    center_mm
        ``(x, y, z)`` envelope centre in :attr:`Frame.BASE`.
    half_extents_mm
        ``(dx, dy, dz)`` non-negative half extents. The box is
        ``[center - half, center + half]`` per axis.
    max_nudge_mm
        Largest permissible per-axis nudge magnitude. The strategy
        clamps its planned offset to this bound; the executor
        re-validates after clamping.
    max_agitate_amplitude_mm
        Largest permissible single-step agitation amplitude. Default
        ``0.0`` keeps :class:`ContainerAgitateStrategy` disabled until
        an operator explicitly raises this number.
    """

    center_mm: tuple[float, float, float]
    half_extents_mm: tuple[float, float, float]
    max_nudge_mm: float = 5.0
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
        it to be planned. Defaults to the two cheapest non-motion
        actions (``RESCAN`` + ``NEXT_VIEWPOINT``) so a freshly
        enabled policy without a configured fixture is still safe.
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
        SceneRecoveryAction.NEXT_VIEWPOINT,
    )
    max_recovery_actions: int = 2
    fixture: Optional[FixtureEnvelope] = None
    #: Per-action attempt cap. An empty mapping means only
    #: ``max_recovery_actions`` applies.
    per_action_budget: Mapping[SceneRecoveryAction, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    #: Modes for which the orchestrator is permitted to drive recovery.
    #: Easy is permanently excluded by default.
    apply_modes: tuple[str, ...] = (
        "auto",
        "dense_clutter",
        "dense_autonomous",
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
    target_identity: Optional[TargetIdentity] = None
    last_grasp: Optional[GraspPoint] = None
    current_tcp: Optional[Pose] = None
    history: tuple[SceneRecoveryAction, ...] = ()
    #: Richer per-(action, class) history for anti-loop bookkeeping.
    #: Strategies must not introspect this tuple; only the orchestrator in
    #: :mod:`src.robot.grasping.recovery.orchestrator` consumes it.
    typed_history: tuple[Any, ...] = ()
    #: When :data:`True` the orchestrator reorders
    #: each failure-class action list so that perception actions
    #: (``NEXT_VIEWPOINT`` / ``RESCAN``) are tried before motion-style
    #: actions. Group order across failure classes is preserved.
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
        Already clamped by the strategy against
        :attr:`FixtureEnvelope.max_nudge_mm`. ``None`` for other
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
        ``"refused_no_offset"``, ``"refused_agitate_disabled"``,
        ``"refused_no_typed_move"``, ``"refused_unknown_action"``,
        ``"aborted_motion_failed"``.
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


@dataclass(frozen=True, slots=True)
class NoRecoveryStrategy:
    """Always returns :attr:`SceneRecoveryAction.NONE`.

    Used as the default when an operator wires the recovery slot but
    has not yet picked a strategy. The service treats this exactly
    like no strategy at all.
    """

    def plan(self, context: SceneRecoveryContext) -> SceneRecoveryPlan:
        return SceneRecoveryPlan(
            action=SceneRecoveryAction.NONE,
            reason="no_recovery_strategy",
            telemetry={"strategy": "no_recovery"},
        )


@dataclass(frozen=True, slots=True)
class ActivePerceptionRecoveryStrategy:
    """Escalate ``RESCAN`` then ``NEXT_VIEWPOINT``.

    The strategy looks at :attr:`SceneRecoveryContext.history`:

    * if ``RESCAN`` is permitted and has not been tried yet, plan
      ``RESCAN``,
    * else if ``NEXT_VIEWPOINT`` is permitted, plan ``NEXT_VIEWPOINT``,
    * else NONE.

    The strategy is stateless: re-running it with the history
    appended drives the escalation.
    """

    def plan(self, context: SceneRecoveryContext) -> SceneRecoveryPlan:
        # The shared gate is action-specific; it runs for each
        # candidate in priority order to surface the correct reason
        # ("policy_disallows_action" vs "profile_disallows_action").
        for action in (
            SceneRecoveryAction.RESCAN,
            SceneRecoveryAction.NEXT_VIEWPOINT,
        ):
            if action in context.history:
                continue
            blocked = _gate(context, action)
            if blocked is None:
                return SceneRecoveryPlan(
                    action=action,
                    reason=f"escalation:{action.value}",
                    telemetry={
                        "strategy": "active_perception",
                        "history": tuple(a.value for a in context.history),
                    },
                )
        return SceneRecoveryPlan(
            action=SceneRecoveryAction.NONE,
            reason="active_perception_exhausted",
            telemetry={
                "strategy": "active_perception",
                "history": tuple(a.value for a in context.history),
            },
        )


@dataclass(frozen=True, slots=True)
class NextTargetRecoveryStrategy:
    """Plan a switch to a different target in the current frame.

    Only meaningful for multi-segmentation scenes. The strategy
    requires a :class:`PerceptionFrame` with more than one
    segmentation; otherwise it returns NONE. This is the cheapest
    non-motion recovery: no perception re-acquisition is needed.
    """

    def plan(self, context: SceneRecoveryContext) -> SceneRecoveryPlan:
        blocked = _gate(context, SceneRecoveryAction.NEXT_TARGET)
        if blocked is not None:
            return blocked
        frame = context.last_frame
        if frame is None or len(frame.segmentations) < 2:
            return SceneRecoveryPlan(
                action=SceneRecoveryAction.NONE,
                reason="no_alternative_target_in_frame",
                telemetry={
                    "strategy": "next_target",
                    "n_segmentations": (
                        0 if frame is None else len(frame.segmentations)
                    ),
                },
            )
        return SceneRecoveryPlan(
            action=SceneRecoveryAction.NEXT_TARGET,
            reason="alternative_target_available",
            telemetry={
                "strategy": "next_target",
                "n_segmentations": len(frame.segmentations),
            },
        )


@dataclass(frozen=True, slots=True)
class SmallNudgeStrategy:
    """Plan a bounded nudge along a chosen axis.

    The nudge magnitude is the policy fixture's
    :attr:`FixtureEnvelope.max_nudge_mm`. Direction is taken from
    :attr:`offset_axis` (a unit-ish 3-vector); the strategy
    re-normalises and scales it. The resulting offset is rejected
    when the displaced position would fall outside the fixture
    envelope.

    The strategy does not pick the "best" nudge direction from the
    scene. An active-perception scoring pass can replace it without
    changing this surface.
    """

    offset_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def plan(self, context: SceneRecoveryContext) -> SceneRecoveryPlan:
        blocked = _gate(context, SceneRecoveryAction.NUDGE_TARGET)
        if blocked is not None:
            return blocked
        fixture = context.policy.fixture
        # _gate already accepted the action, which means the policy
        # has a fixture (construction enforces it). The check below
        # is a defensive runtime guard that also narrows the type.
        if fixture is None:  # pragma: no cover (defensive)
            return SceneRecoveryPlan(
                action=SceneRecoveryAction.NONE,
                reason="no_fixture_envelope",
                telemetry={"strategy": "small_nudge"},
            )
        axis = np.asarray(self.offset_axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm <= 0.0:
            return SceneRecoveryPlan(
                action=SceneRecoveryAction.NONE,
                reason="zero_offset_axis",
                telemetry={"strategy": "small_nudge"},
            )
        unit = axis / norm
        scaled = unit * float(fixture.max_nudge_mm)
        # Validate that the displacement lands inside the envelope.
        # Without a known target position the current TCP is the
        # fallback anchor. If neither is available, refuse.
        anchor: Optional[np.ndarray] = None
        if context.last_grasp is not None:
            anchor = np.asarray(context.last_grasp.position, dtype=np.float64)
        elif context.current_tcp is not None:
            anchor = np.asarray(
                context.current_tcp.position_mm, dtype=np.float64
            )
        if anchor is None:
            return SceneRecoveryPlan(
                action=SceneRecoveryAction.NONE,
                reason="no_anchor_pose",
                telemetry={"strategy": "small_nudge"},
            )
        target_position = (anchor + scaled).tolist()
        if not fixture.contains(target_position):
            return SceneRecoveryPlan(
                action=SceneRecoveryAction.NONE,
                reason="nudge_would_leave_envelope",
                telemetry={
                    "strategy": "small_nudge",
                    "candidate_position_mm": tuple(target_position),
                    "fixture_center_mm": fixture.center_mm,
                    "fixture_half_extents_mm": fixture.half_extents_mm,
                },
            )
        return SceneRecoveryPlan(
            action=SceneRecoveryAction.NUDGE_TARGET,
            reason="bounded_nudge",
            nudge_offset_mm=(float(scaled[0]), float(scaled[1]), float(scaled[2])),
            telemetry={
                "strategy": "small_nudge",
                "magnitude_mm": float(fixture.max_nudge_mm),
                "anchor_position_mm": tuple(float(x) for x in anchor),
                "target_position_mm": tuple(target_position),
            },
        )


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
    descends, sweeps and retracts. Every waypoint is re-clamped against
    the fixture envelope and any non-EXECUTED motion aborts.
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
    * For non-motion actions (``RESCAN`` / ``NEXT_VIEWPOINT`` /
      ``NEXT_TARGET`` / ``ABORT``) returns
      ``executed=True, outcome="completed"`` without touching the arm;
      the caller is responsible for re-acquiring perception or
      switching target.
    * For :attr:`SceneRecoveryAction.NUDGE_TARGET` commands a single
      typed :meth:`RobotArm.move` to ``current_tcp + nudge``. Any
      :class:`MotionResult` whose status is not ``EXECUTED`` aborts
      the recovery with ``outcome="aborted_motion_failed"``. Without
      ``current_tcp`` the executor refuses with
      ``outcome="refused_no_tcp"``.
    * For :attr:`SceneRecoveryAction.CONTAINER_AGITATE` drives three
      bounded waypoints via :meth:`RobotArm.move`, re-clamping every
      waypoint against the fixture envelope: a there-and-back
      oscillation (``+amplitude``/``-amplitude`` along x, then back to
      the start TCP) while
      :attr:`FixtureEnvelope.agitate_contact_depth_mm` is ``0.0``,
      otherwise a descend, sweep ``+amplitude`` along +X and retract
      that ends offset from the start TCP. Refuses
      with ``"refused_no_tcp"`` / ``"refused_agitate_disabled"`` (no
      fixture or amplitude<=0) / ``"refused_envelope_violation"``, and
      aborts on any non-EXECUTED status with ``"aborted_motion_failed"``.
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
        SceneRecoveryAction.NEXT_VIEWPOINT,
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
        # A bounded three-waypoint motion that redistributes clutter, driven through the same
        # SafetyPreflight-gated arm.move surface as the nudge. Every waypoint is re-validated against the
        # fixture envelope in the executor (defence-in-depth, like NUDGE_TARGET), so a hand-built plan or
        # config drift cannot inject an out-of-box waypoint, and the safety layer can still veto each move.
        if current_tcp is None:
            return SceneRecoveryReport(
                plan=plan,
                executed=False,
                outcome="refused_no_tcp",
                telemetry={"action": str(plan.action)},
            )
        fixture = policy.fixture
        amplitude = float(plan.agitate_amplitude_mm)
        if fixture is None or amplitude <= 0.0:
            return SceneRecoveryReport(
                plan=plan,
                executed=False,
                outcome="refused_agitate_disabled",
                telemetry={"action": str(plan.action), "amplitude_mm": amplitude},
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
        for index, delta in enumerate(deltas):
            destination = start + np.asarray(delta, dtype=np.float64)
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
                "agitate_mode": "contact_redistribute" if depth > 0.0 else "oscillation",
                "contact_depth_mm": depth,
            },
        )
    if plan.action is SceneRecoveryAction.NUDGE_TARGET:
        if current_tcp is None:
            return SceneRecoveryReport(
                plan=plan,
                executed=False,
                outcome="refused_no_tcp",
                telemetry={"action": str(plan.action)},
            )
        if plan.nudge_offset_mm is None:
            return SceneRecoveryReport(
                plan=plan,
                executed=False,
                outcome="refused_no_offset",
                telemetry={"action": str(plan.action)},
            )
        fixture = policy.fixture
        offset = np.asarray(plan.nudge_offset_mm, dtype=np.float64)
        # Re-validate that the destination falls inside the fixture envelope.
        # The strategy clamps as well; checking again here keeps the safety
        # guarantee local to the motion.
        destination = (
            np.asarray(current_tcp.position_mm, dtype=np.float64) + offset
        )
        if fixture is not None and not fixture.contains(destination.tolist()):
            return SceneRecoveryReport(
                plan=plan,
                executed=False,
                outcome="refused_envelope_violation",
                telemetry={
                    "action": str(plan.action),
                    "destination_mm": tuple(float(x) for x in destination),
                },
            )
        target_pose = Pose(
            position_mm=destination,
            quaternion_xyzw=np.asarray(
                current_tcp.quaternion_xyzw, dtype=np.float64
            ).copy(),
            frame=Frame.BASE,
            label="recovery_nudge",
        )
        typed_move = getattr(arm, "move", None)
        if not callable(typed_move):
            return SceneRecoveryReport(
                plan=plan,
                executed=False,
                outcome="refused_no_typed_move",
                telemetry={"action": str(plan.action)},
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
                    "motion_status": str(status),
                },
            )
        return SceneRecoveryReport(
            plan=plan,
            executed=True,
            outcome="completed",
            telemetry={
                "action": str(plan.action),
                "destination_mm": tuple(float(x) for x in destination),
                "motion_status": str(status),
            },
        )
    # Unknown action: refuse rather than silently completing.
    return SceneRecoveryReport(  # pragma: no cover (defensive)
        plan=plan,
        executed=False,
        outcome="refused_unknown_action",
        telemetry={"action": str(plan.action)},
    )
