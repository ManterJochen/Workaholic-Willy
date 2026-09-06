"""Recovery intelligence orchestrator.

The orchestration layer on top of the typed recovery carriers in
:mod:`src.robot.grasping.recovery.policy`, which supply the typed
action vocabulary, the operator-bounded policy, the fixture
envelope, the per-action strategy implementations and the motion
executor. This module is the pure decision driver that, given:

* the locked :class:`GraspBehaviorProfile`,
* the operator-supplied :class:`SceneRecoveryPolicy`,
* the typed failure-reason set from the last pick attempt, and
* the accumulated recovery history,

selects the next :class:`SceneRecoveryPlan` to execute, or returns
:data:`None` when no further recovery is permissible.

The orchestrator is byte-identically inert when ``policy.enabled`` is
:data:`False` or the resolved mode is not in
``policy.apply_modes``. Easy is permanently excluded from the
default ``apply_modes``; this is a second hard guard on top of the
easy profile having an empty ``recovery_allowed_actions`` tuple.

Public surface
--------------

* :class:`RecoveryDispatcher`: pure failure-class to action mapping.
* :class:`RecoveryHistoryEntry`: typed (action, failure-class,
  outcome) record used for anti-loop bookkeeping.
* :class:`RecoveryTrailEntry`, :class:`RecoveryTrail`: typed
  telemetry attached to the per-pick outcome.
* :class:`RecoveryOrchestrator`: frozen decision driver.
* :func:`run_recovery_loop`: bounded loop wrapper around an
  external ``pick`` callable. Drives the orchestrator, the executor
  and perception re-acquisition. Returns the final report and the
  typed trail.

When armed, the agitate motion drives bounded, envelope-clamped waypoints
through the SafetyPreflight-gated arm.move surface, the same executor path
as the nudge: a there-and-back oscillation by default, a contact
redistribute when the fixture sets a non-zero agitate contact depth. The
orchestrator never plans an action the policy or profile does not list;
defaults are chosen so a recovery YAML produces a byte-identical snapshot
until the operator opts in (CONTAINER_AGITATE stays gated behind a fixture
and a non-zero amplitude).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Protocol,
    TYPE_CHECKING,
    Tuple,
)

from src.robot.grasping.constants import (
    RECOVERY_ORCHESTRATOR_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.types.feedback import GraspFailureReason
from src.robot.grasping.recovery.policy import (
    _PHYSICAL_ACTIONS,
    SceneRecoveryAction,
    SceneRecoveryContext,
    SceneRecoveryPlan,
    SceneRecoveryPolicy,
    SceneRecoveryStrategy,
    execute_recovery_motion,
)
from src.robot.grasping.recovery.trail_serialize import (  # noqa: F401 (re-exported for by-name importers)
    recovery_actions_from_trail,
)

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.execution.autonomous_grasp import (
        GraspBehaviorProfile,
    )
    from src.robot.core import RobotArm


#: Recovery is a bounded loop (``max_recovery_actions``, typically <= 3), so the
#: driver logs one line per executed action and one for the terminal reason.
logger = create_grasping_logger(
    "RecoveryOrchestrator", RECOVERY_ORCHESTRATOR_LOG_FILE
)


__all__ = [
    "RecoveryDispatcher",
    "RecoveryHistoryEntry",
    "RecoveryOrchestrator",
    "RecoveryTrail",
    "RecoveryTrailEntry",
    "VALID_TERMINAL_REASONS",
    "reorder_actions_for_uncertainty",
    "run_recovery_loop",
]


# Locked default map from failure class to action priority.
_DEFAULT_MAP: Mapping[GraspFailureReason, Tuple[SceneRecoveryAction, ...]] = {
    # IK, reachability and table conflict: try a different target first,
    # then a rescan to refresh the scene.
    GraspFailureReason.IK_FAILED: (
        SceneRecoveryAction.NEXT_TARGET,
        SceneRecoveryAction.RESCAN,
    ),
    GraspFailureReason.ALL_OUT_OF_WORKSPACE: (
        SceneRecoveryAction.NEXT_TARGET,
        SceneRecoveryAction.RESCAN,
    ),
    GraspFailureReason.ALL_TABLE_CONFLICT: (
        SceneRecoveryAction.NEXT_TARGET,
        SceneRecoveryAction.RESCAN,
    ),
    # Perception-quality complaints: rescan first, then move camera.
    GraspFailureReason.RESCAN_RECOMMENDED: (
        SceneRecoveryAction.RESCAN,
        SceneRecoveryAction.NEXT_VIEWPOINT,
    ),
    GraspFailureReason.LOW_DEPTH_CONFIDENCE: (
        SceneRecoveryAction.RESCAN,
        SceneRecoveryAction.NEXT_VIEWPOINT,
    ),
    GraspFailureReason.LOW_MASK_CONFIDENCE: (
        SceneRecoveryAction.RESCAN,
        SceneRecoveryAction.NEXT_VIEWPOINT,
    ),
    GraspFailureReason.EMPTY_MASK: (
        SceneRecoveryAction.RESCAN,
        SceneRecoveryAction.NEXT_VIEWPOINT,
    ),
    GraspFailureReason.MASK_TOO_SMALL: (
        SceneRecoveryAction.RESCAN,
        SceneRecoveryAction.NEXT_VIEWPOINT,
    ),
    GraspFailureReason.NO_VALID_DEPTH: (
        SceneRecoveryAction.RESCAN,
        SceneRecoveryAction.NEXT_VIEWPOINT,
    ),
    # Occlusion: move viewpoint first.
    GraspFailureReason.ACTIVE_PERCEPTION_RECOMMENDED: (
        SceneRecoveryAction.NEXT_VIEWPOINT,
        SceneRecoveryAction.RESCAN,
    ),
    GraspFailureReason.HEAVY_OCCLUSION: (
        SceneRecoveryAction.NEXT_VIEWPOINT,
        SceneRecoveryAction.RESCAN,
    ),
    # No candidates, collisions or no valid grasp: try alternative
    # targets, then a bounded nudge (only if fixture present), then
    # change viewpoint.
    GraspFailureReason.NO_CANDIDATES_GENERATED: (
        SceneRecoveryAction.NEXT_TARGET,
        SceneRecoveryAction.NUDGE_TARGET,
        SceneRecoveryAction.NEXT_VIEWPOINT,
    ),
    GraspFailureReason.ALL_COLLIDED: (
        # A clutter jam is the case agitation is for: try a bounded nudge, then an envelope-clamped
        # agitate, before changing viewpoint. CONTAINER_AGITATE stays double-gated (profile + policy
        # allow-list + fixture present + non-zero amplitude), and no built-in profile lists it in
        # recovery_allowed_actions, so the outer gate always drops it; the entry records the intent
        # for a profile that does list it.
        SceneRecoveryAction.NEXT_TARGET,
        SceneRecoveryAction.NUDGE_TARGET,
        SceneRecoveryAction.CONTAINER_AGITATE,
        SceneRecoveryAction.NEXT_VIEWPOINT,
    ),
    GraspFailureReason.NO_VALID_GRASP: (
        SceneRecoveryAction.NEXT_TARGET,
        SceneRecoveryAction.NUDGE_TARGET,
        SceneRecoveryAction.NEXT_VIEWPOINT,
    ),
    # Refinement-class failures: rescan to re-acquire the target,
    # then try a different one.
    GraspFailureReason.TARGET_LOST_DURING_REFINE: (
        SceneRecoveryAction.RESCAN,
        SceneRecoveryAction.NEXT_TARGET,
    ),
    GraspFailureReason.REFINEMENT_DIVERGED: (
        SceneRecoveryAction.RESCAN,
        SceneRecoveryAction.NEXT_TARGET,
    ),
    # Explicit "try next candidate" hint.
    GraspFailureReason.TRY_NEXT_CANDIDATE: (
        SceneRecoveryAction.NEXT_TARGET,
    ),
    # The planner refused to route to a valid grasp and nothing moved.
    # Rescan and nothing else, as a safety rule rather than a tuning one:
    # a refusal is a fail-closed outcome, re-perceiving costs nothing and
    # changes what the planner is asked next, while every other action
    # here either commands motion (NUDGE_TARGET, CONTAINER_AGITATE, and
    # NEXT_VIEWPOINT in the pick loop's own recovery) or silently
    # redirects the cell to an object the operator did not ask for
    # (NEXT_TARGET). Falling through to one of those after the cell has
    # just declined to move is how a refusal becomes a motion.
    GraspFailureReason.MOTION_PLAN_REFUSED: (
        SceneRecoveryAction.RESCAN,
    ),
    # Escalations: no recovery is appropriate, surface to the
    # operator instead.
    #
    # A controller that cannot move belongs here: every action in this table either re-perceives
    # (pointless, the scene is fine) or commands motion (refused, and the last thing a stopped cell
    # should be asked for). It needs a human.
    GraspFailureReason.CONTROLLER_NOT_OPERATIONAL: (),
    # A target label perception never returned belongs here too: not unsafe (the cell is fine) and
    # not fail-closed (nothing refused anything), only pointless. The detector is deterministic on a
    # static frame, so a rescan re-derives the same labels; NEXT_TARGET would hand the operator an
    # object they did not ask for, the exact substitution the hard label gate exists to prevent. The
    # operator reads the labels that were seen and re-prompts.
    GraspFailureReason.TARGET_LABEL_NOT_FOUND: (),
    GraspFailureReason.TOPOLOGY_RISK_REJECTED: (),
    GraspFailureReason.SEMANTIC_REJECTED: (),
    GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED: (),
}


@dataclass(frozen=True, slots=True)
class RecoveryDispatcher:
    """Pure dispatcher from failure class to action priority.

    The dispatcher is operator-overridable: passing a non-empty
    ``overrides`` mapping replaces the default action sequence for
    each listed failure class. Unlisted classes retain the locked
    defaults. Setting a class to an empty tuple disables recovery
    for that class (useful for operator policy where certain failure
    types must always escalate).
    """

    overrides: Mapping[GraspFailureReason, Tuple[SceneRecoveryAction, ...]] = (
        field(default_factory=dict)
    )

    def actions_for(
        self, reason: GraspFailureReason
    ) -> Tuple[SceneRecoveryAction, ...]:
        if reason in self.overrides:
            return tuple(self.overrides[reason])
        return _DEFAULT_MAP.get(reason, ())


# ---------------------------------------------------------------------------
# History + trail types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecoveryHistoryEntry:
    """Single anti-loop bookkeeping entry.

    The orchestrator refuses to plan the same
    ``(action, failure_class)`` pair twice within one pick().
    """

    action: SceneRecoveryAction
    failure_class: GraspFailureReason
    outcome: str


VALID_TERMINAL_REASONS: frozenset[str] = frozenset(
    {
        "recovered_success",
        "exhausted_budget",
        "anti_loop_blocked",
        "escalated_no_recovery",
        "recovery_disabled",
    }
)


@dataclass(frozen=True, slots=True)
class RecoveryTrailEntry:
    """One step in the orchestration trail.

    Recorded after the plan has been executed (or refused) so the
    ``executed`` and ``outcome`` fields reflect actual driver state.
    """

    attempt_index: int
    failure_reason: GraspFailureReason
    plan_action: SceneRecoveryAction
    plan_reason: str
    executed: bool
    outcome: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_index": int(self.attempt_index),
            "failure_reason": str(self.failure_reason.value),
            "plan_action": str(self.plan_action.value),
            "plan_reason": str(self.plan_reason),
            "executed": bool(self.executed),
            "outcome": str(self.outcome),
        }


@dataclass(frozen=True, slots=True)
class RecoveryTrail:
    """Typed trail attached to the final outcome of a recovery loop.

    ``terminal_reason`` is one of :data:`VALID_TERMINAL_REASONS`.
    """

    entries: Tuple[RecoveryTrailEntry, ...]
    terminal_reason: str

    def __post_init__(self) -> None:
        if self.terminal_reason not in VALID_TERMINAL_REASONS:
            raise ValueError(
                f"terminal_reason must be one of "
                f"{sorted(VALID_TERMINAL_REASONS)!r}; got "
                f"{self.terminal_reason!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminal_reason": str(self.terminal_reason),
            "entries": [e.to_dict() for e in self.entries],
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


# Perception-style actions that gather more information about the
# scene. When ``aggressive_recovery_bias`` is set the orchestrator
# reorders each failure-class action list so these come first; group
# order across failure classes is preserved.
_PERCEPTION_ACTIONS: frozenset[SceneRecoveryAction] = frozenset(
    {SceneRecoveryAction.NEXT_VIEWPOINT, SceneRecoveryAction.RESCAN}
)


def reorder_actions_for_uncertainty(
    actions: Tuple[SceneRecoveryAction, ...],
    *,
    aggressive: bool,
) -> Tuple[SceneRecoveryAction, ...]:
    """Optionally push perception actions to the front of ``actions``.

    When ``aggressive`` is :data:`False` the input tuple is returned
    unchanged (byte-identical to the input ordering). When :data:`True` the
    relative order within the perception group and within the
    non-perception group is preserved; only the two groups are
    swapped in priority.
    """

    if not aggressive:
        return actions
    perception: list[SceneRecoveryAction] = []
    other: list[SceneRecoveryAction] = []
    for a in actions:
        if a in _PERCEPTION_ACTIONS:
            perception.append(a)
        else:
            other.append(a)
    if not perception:
        return actions
    return tuple(perception + other)


@dataclass(frozen=True, slots=True)
class RecoveryOrchestrator:
    """Pure decision driver for the recovery state machine.

    Given a :class:`SceneRecoveryContext` the orchestrator returns
    either a concrete :class:`SceneRecoveryPlan` (action != NONE) or
    :data:`None`, meaning no further recovery is permissible and the
    caller terminates.

    The orchestrator delegates the content of the plan to a
    per-action strategy looked up in :attr:`strategies` unless
    :attr:`bypass_strategies` is :data:`True`, in which case it
    synthesises a minimal plan directly and consults no strategy, so
    the dispatcher mapping alone decides the action.
    """

    dispatcher: RecoveryDispatcher
    strategies: Mapping[SceneRecoveryAction, SceneRecoveryStrategy]
    bypass_strategies: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def attach_history_entry(
        context: SceneRecoveryContext,
        entry: RecoveryHistoryEntry,
    ) -> SceneRecoveryContext:
        """Return a new context with ``entry`` appended to both ``history`` and ``typed_history``."""

        new_history = context.history + (entry.action,)
        new_typed = context.typed_history + (entry,)
        return replace(context, history=new_history, typed_history=new_typed)

    def context_with_history_entry(
        self,
        context: SceneRecoveryContext,
        entry: RecoveryHistoryEntry,
    ) -> SceneRecoveryContext:
        return self.attach_history_entry(context, entry)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def next_step(
        self,
        context: SceneRecoveryContext,
        *,
        typed_history: Tuple[RecoveryHistoryEntry, ...] = (),
    ) -> Optional[SceneRecoveryPlan]:
        """Return the next plan, or :data:`None` to terminate.

        Parameters
        ----------
        context
            The standard :class:`SceneRecoveryContext`.
        typed_history
            Optional richer history with failure-class anti-loop
            tracking. When supplied the orchestrator refuses to plan
            an action that has already been tried for the same
            failure class.
        """

        policy = context.policy
        if not policy.enabled:
            return None
        mode_name = context.profile.mode.value
        if mode_name not in policy.apply_modes:
            return None
        if not context.failure_reasons:
            return None
        if len(context.history) >= policy.max_recovery_actions:
            return None

        # Prefer the explicit kwarg; fall back to the context's
        # ``typed_history`` sidecar, which ``attach_history_entry`` fills.
        merged_typed = tuple(typed_history) + tuple(context.typed_history)
        tried_pairs: set[Tuple[SceneRecoveryAction, GraspFailureReason]] = {
            (e.action, e.failure_class)
            for e in merged_typed
            if hasattr(e, "action") and hasattr(e, "failure_class")
        }
        # Per-action budget consumption count: how many history actions
        # equal each action.
        action_counts: dict[SceneRecoveryAction, int] = {}
        for h in context.history:
            action_counts[h] = action_counts.get(h, 0) + 1

        for reason in context.failure_reasons:
            for action in reorder_actions_for_uncertainty(
                self.dispatcher.actions_for(reason),
                aggressive=bool(context.aggressive_recovery_bias),
            ):
                # Outer gate: profile must allow this action.
                if action.value not in tuple(
                    context.profile.recovery_allowed_actions
                ):
                    continue
                # Inner gate: policy must allow this action.
                if action not in policy.allowed_actions:
                    continue
                # Anti-loop: same (action, class) pair already tried.
                if (action, reason) in tried_pairs:
                    continue
                # Per-action budget.
                budget = self._per_action_budget(policy, action)
                if budget is not None and action_counts.get(action, 0) >= budget:
                    continue
                # For physical actions, require a fixture.
                if action in _PHYSICAL_ACTIONS and policy.fixture is None:
                    continue
                plan = self._materialise_plan(action, context)
                if plan.action is SceneRecoveryAction.NONE:
                    # Strategy refused; try the next action in the
                    # priority list (graceful fall-through).
                    continue
                return plan
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _per_action_budget(
        policy: SceneRecoveryPolicy,
        action: SceneRecoveryAction,
    ) -> Optional[int]:
        budget = policy.per_action_budget.get(action)
        if budget is None:
            return None
        return int(budget)

    def _materialise_plan(
        self,
        action: SceneRecoveryAction,
        context: SceneRecoveryContext,
    ) -> SceneRecoveryPlan:
        if self.bypass_strategies:
            return SceneRecoveryPlan(
                action=action,
                reason=f"dispatcher:{action.value}",
                telemetry={"orchestrator": "bypass_strategies"},
            )
        strategy = self.strategies.get(action)
        if strategy is None:
            return SceneRecoveryPlan(
                action=SceneRecoveryAction.NONE,
                reason="no_strategy_for_action",
                telemetry={"requested_action": str(action)},
            )
        return strategy.plan(context)


# ---------------------------------------------------------------------------
# Loop driver
# ---------------------------------------------------------------------------


class _PickReportProtocol(Protocol):
    """Structural protocol for the report shape used by the loop.

    The driver only reads ``outcome`` and ``failure_reasons`` from
    the report. Anything else passes through unchanged.
    """

    outcome: Any
    failure_reasons: Tuple[GraspFailureReason, ...]


def _is_success(report: Any) -> bool:
    outcome = getattr(report, "outcome", None)
    return outcome is not None and getattr(outcome, "value", str(outcome)) == "succeeded"


def _extract_failure_reasons(report: Any) -> Tuple[GraspFailureReason, ...]:
    reasons = getattr(report, "failure_reasons", None)
    if not reasons:
        return ()
    out: list[GraspFailureReason] = []
    for r in reasons:
        if isinstance(r, GraspFailureReason):
            out.append(r)
    return tuple(out)


def run_recovery_loop(
    *,
    pick: Callable[[], Any],
    profile: "GraspBehaviorProfile",
    policy: SceneRecoveryPolicy,
    orchestrator: RecoveryOrchestrator,
    frame_acquirer: Callable[[], None],
    arm: Optional["RobotArm"] = None,
    aggressive_recovery_bias: bool | Callable[[Any], bool] = False,
) -> tuple[Any, RecoveryTrail]:
    """Bounded closed-loop recovery driver.

    The driver:

    1. Calls ``pick()``.
    2. If the outcome is success, returns immediately with a
       ``recovered_success`` trail.
    3. Otherwise extracts typed failure reasons, asks the orchestrator
       for the next plan, executes the plan (motion via
       :func:`execute_recovery_motion`, perception re-acquisition via
       ``frame_acquirer``), appends to the trail, and retries.
    4. Terminates with one of the typed
       :data:`VALID_TERMINAL_REASONS`.

    The loop is bounded by ``policy.max_recovery_actions`` and the
    orchestrator's anti-loop rule.

    ``aggressive_recovery_bias`` may be a plain ``bool`` or a callable
    ``(last_report) -> bool`` re-evaluated each iteration. The callable form
    lets the driver derive the bias from the most recent attempt (e.g. fused
    uncertainty above a threshold) so perception actions are prioritised only
    while the scene is uncertain.
    """

    # Disabled fast path: emit a typed trail without ever calling the
    # orchestrator.
    if not policy.enabled or profile.mode.value not in policy.apply_modes:
        report = pick()
        if _is_success(report):
            return report, RecoveryTrail(entries=(), terminal_reason="recovered_success")
        logger.debug(
            "Recovery inert for mode=%s (enabled=%s, apply_modes=%s); pick failure escalates",
            profile.mode.value,
            policy.enabled,
            tuple(policy.apply_modes),
        )
        return report, RecoveryTrail(
            entries=(), terminal_reason="recovery_disabled"
        )

    typed_history: list[RecoveryHistoryEntry] = []
    action_history: list[SceneRecoveryAction] = []
    trail_entries: list[RecoveryTrailEntry] = []
    last_report = pick()

    while True:
        if _is_success(last_report):
            return last_report, RecoveryTrail(
                entries=tuple(trail_entries),
                terminal_reason="recovered_success",
            )
        failure_reasons = _extract_failure_reasons(last_report)
        if not failure_reasons:
            logger.warning(
                "Pick failed with no typed failure reason after %d recovery action(s) "
                "- nothing to recover from, escalating",
                len(trail_entries),
            )
            return last_report, RecoveryTrail(
                entries=tuple(trail_entries),
                terminal_reason="escalated_no_recovery",
            )

        bias = (
            bool(aggressive_recovery_bias(last_report))
            if callable(aggressive_recovery_bias)
            else bool(aggressive_recovery_bias)
        )
        ctx = SceneRecoveryContext(
            profile=profile,
            policy=policy,
            last_outcome=getattr(last_report, "outcome"),
            failure_reasons=failure_reasons,
            history=tuple(action_history),
            aggressive_recovery_bias=bias,
        )
        plan = orchestrator.next_step(
            ctx, typed_history=tuple(typed_history)
        )
        if plan is None:
            terminal = _terminal_reason_for_none(
                policy=policy,
                action_history=tuple(action_history),
                typed_history=tuple(typed_history),
                failure_reasons=failure_reasons,
                orchestrator=orchestrator,
            )
            logger.info(
                "Recovery finished after %d action(s): %s (reasons=%s)",
                len(trail_entries),
                terminal,
                ", ".join(r.value for r in failure_reasons),
            )
            return last_report, RecoveryTrail(
                entries=tuple(trail_entries),
                terminal_reason=terminal,
            )

        # Execute the plan.
        executed, outcome = _execute_plan(
            plan=plan, policy=policy, arm=arm, frame_acquirer=frame_acquirer
        )
        primary_failure = failure_reasons[0]
        logger.info(
            "Recovery action %d: %s for %s (%s) -> executed=%s outcome=%s",
            len(trail_entries),
            plan.action.value,
            primary_failure.value,
            plan.reason,
            executed,
            outcome,
        )
        trail_entries.append(
            RecoveryTrailEntry(
                attempt_index=len(trail_entries),
                failure_reason=primary_failure,
                plan_action=plan.action,
                plan_reason=plan.reason,
                executed=executed,
                outcome=outcome,
            )
        )
        action_history.append(plan.action)
        typed_history.append(
            RecoveryHistoryEntry(
                action=plan.action,
                failure_class=primary_failure,
                outcome=outcome,
            )
        )
        if not executed:
            # Refused or aborted: terminate with escalation.
            logger.warning(
                "Recovery action %s was not executed (%s); escalating",
                plan.action.value,
                outcome,
            )
            return last_report, RecoveryTrail(
                entries=tuple(trail_entries),
                terminal_reason="escalated_no_recovery",
            )
        last_report = pick()


def _terminal_reason_for_none(
    *,
    policy: SceneRecoveryPolicy,
    action_history: Tuple[SceneRecoveryAction, ...],
    typed_history: Tuple[RecoveryHistoryEntry, ...],
    failure_reasons: Tuple[GraspFailureReason, ...],
    orchestrator: RecoveryOrchestrator,
) -> str:
    """Classify why the orchestrator returned ``None`` for telemetry.

    Priority: ``exhausted_budget`` > ``anti_loop_blocked`` > ``escalated_no_recovery``.
    """

    if len(action_history) >= policy.max_recovery_actions:
        return "exhausted_budget"
    # Anti-loop: whether every candidate action for every failure reason was
    # already tried for the same class.
    tried_pairs = {
        (e.action, e.failure_class) for e in typed_history
    }
    saw_blocked_pair = False
    for reason in failure_reasons:
        for action in orchestrator.dispatcher.actions_for(reason):
            if (action, reason) in tried_pairs:
                saw_blocked_pair = True
    if saw_blocked_pair:
        return "anti_loop_blocked"
    return "escalated_no_recovery"


def _execute_plan(
    *,
    plan: SceneRecoveryPlan,
    policy: SceneRecoveryPolicy,
    arm: Optional["RobotArm"],
    frame_acquirer: Callable[[], None],
) -> tuple[bool, str]:
    """Run the plan's side-effect; return (executed, outcome_string)."""

    if plan.action in (
        SceneRecoveryAction.RESCAN,
        SceneRecoveryAction.NEXT_VIEWPOINT,
    ):
        frame_acquirer()
        return True, "completed"
    if plan.action is SceneRecoveryAction.NEXT_TARGET:
        # No perception side effect; the next pick() call re-ranks.
        return True, "completed"
    if plan.action in (
        SceneRecoveryAction.NUDGE_TARGET,
        SceneRecoveryAction.CONTAINER_AGITATE,
    ):
        if arm is None:
            logger.warning(
                "Physical recovery %s requested without an arm - refused",
                plan.action.value,
            )
            return False, "refused_no_arm"
        # Both physical actions route through the SafetyPreflight-gated executor,
        # which needs a start pose, so read the live TCP first.
        get_tcp = getattr(arm, "get_tcp_pose", None)
        current_tcp = get_tcp() if callable(get_tcp) else None
        report = execute_recovery_motion(
            arm=arm, plan=plan, policy=policy, current_tcp=current_tcp
        )
        return bool(report.executed), str(report.outcome)
    logger.warning("No executor for recovery action %s - refused", plan.action.value)
    return False, "refused_unknown_action"
