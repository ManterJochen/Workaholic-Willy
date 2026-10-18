"""Recovery orchestrator: the bounded loop that looks again after a failed pick.

The orchestration layer on top of the typed recovery carriers in :mod:`src.robot.grasping.recovery.policy`. This
module is the pure decision driver that, given:

* the locked :class:`GraspBehaviorProfile`,
* the operator-supplied :class:`SceneRecoveryPolicy`,
* the typed failure-reason set from the last pick attempt, and
* the accumulated recovery history,

selects the next :class:`SceneRecoveryPlan` to run, or returns :data:`None` when no further recovery is permissible.

The orchestrator is inert when ``policy.enabled`` is :data:`False` or the resolved mode is not in
``policy.apply_modes``. Easy is permanently excluded from the default ``apply_modes``; this is a second hard guard on
top of the easy profile having an empty ``recovery_allowed_actions`` tuple.

The loop plans one action, ``RESCAN``, which moves nothing itself: the next pick re-perceives the scene. The push
(``NUDGE_TARGET``) runs inside the pick attempt
(:func:`~src.robot.grasping.recovery.push_motion.execute_push`), where the part, its neighbours and the keep-out are
known, and this loop never plans it. ``NEXT_TARGET``, the rescan that skipped the part a pick failed on, and
``CONTAINER_AGITATE``, which moved the arm inside a declared fixture box, left on 2026-10-16 (the owner's cleanup), and
with them the strategies that planned an action and the refusals a planned action could meet before anything moved.

Public surface
--------------

* :class:`RecoveryDispatcher`: pure failure-class to action mapping.
* :class:`RecoveryHistoryEntry`: typed (action, failure-class, outcome) record used for anti-loop bookkeeping.
* :class:`RecoveryTrailEntry`, :class:`RecoveryTrail`: typed telemetry attached to the per-pick outcome.
* :class:`RecoveryOrchestrator`: frozen decision driver.
* :func:`run_recovery_loop`: bounded loop wrapper around an external ``pick`` callable. Drives the orchestrator and
  perception re-acquisition. Returns the final report and the typed trail.
* :func:`terminal_outcome_for_trail`: the service outcome a finished loop stands for (``recovery_exhausted``), or
  ``None`` to keep the last pick's own.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    SceneRecoveryAction,
    SceneRecoveryContext,
    SceneRecoveryPlan,
    SceneRecoveryPolicy,
)

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.execution.autonomous_grasp import (
        GraspBehaviorProfile,
    )


#: Recovery is a bounded loop (``max_recovery_actions``, typically <= 3), so the
#: driver logs one line per executed action and one for the terminal reason.
logger = create_grasping_logger(
    "RecoveryOrchestrator", RECOVERY_ORCHESTRATOR_LOG_FILE
)


__all__ = [
    "OUTCOME_RECOVERY_EXHAUSTED",
    "RecoveryDispatcher",
    "RecoveryHistoryEntry",
    "RecoveryOrchestrator",
    "RecoveryTrail",
    "RecoveryTrailEntry",
    "VALID_TERMINAL_REASONS",
    "run_recovery_loop",
    "terminal_outcome_for_trail",
]


# Locked map from failure class to the action it asks for. ``NEXT_VIEWPOINT`` left every row on 2026-09-29, merged
# into ``RESCAN``; ``NEXT_TARGET``, ``CONTAINER_AGITATE`` and the push left them on 2026-10-16, the first two with their
# code and the push because it runs inside the pick attempt, never from this loop.
_DEFAULT_MAP: Mapping[GraspFailureReason, Tuple[SceneRecoveryAction, ...]] = {
    # IK, reachability and table conflict: a rescan refreshes the scene the next pick plans on.
    GraspFailureReason.IK_FAILED: (SceneRecoveryAction.RESCAN,),
    GraspFailureReason.ALL_OUT_OF_WORKSPACE: (SceneRecoveryAction.RESCAN,),
    GraspFailureReason.ALL_TABLE_CONFLICT: (SceneRecoveryAction.RESCAN,),
    # Perception-quality complaints: look again.
    GraspFailureReason.RESCAN_RECOMMENDED: (SceneRecoveryAction.RESCAN,),
    GraspFailureReason.LOW_DEPTH_CONFIDENCE: (SceneRecoveryAction.RESCAN,),
    GraspFailureReason.LOW_MASK_CONFIDENCE: (SceneRecoveryAction.RESCAN,),
    GraspFailureReason.EMPTY_MASK: (SceneRecoveryAction.RESCAN,),
    GraspFailureReason.MASK_TOO_SMALL: (SceneRecoveryAction.RESCAN,),
    GraspFailureReason.NO_VALID_DEPTH: (SceneRecoveryAction.RESCAN,),
    # Occlusion: look again. Another view comes from a look pose or a second camera, not from here.
    GraspFailureReason.ACTIVE_PERCEPTION_RECOMMENDED: (SceneRecoveryAction.RESCAN,),
    GraspFailureReason.HEAVY_OCCLUSION: (SceneRecoveryAction.RESCAN,),
    # No candidates or no valid grasp: look again. A clutter jam (``ALL_COLLIDED``) is the one case a push is for, and
    # the pick attempt pushes before it gives up (push_planner.neighbour_evidence); what reaches this loop rescans.
    GraspFailureReason.NO_CANDIDATES_GENERATED: (SceneRecoveryAction.RESCAN,),
    GraspFailureReason.ALL_COLLIDED: (SceneRecoveryAction.RESCAN,),
    GraspFailureReason.NO_VALID_GRASP: (SceneRecoveryAction.RESCAN,),
    # Refinement-class failures. No pick reports either since the two-scan refinement left on 2026-09-29; the rows
    # stay so a failure of that kind replayed from an older record still maps.
    GraspFailureReason.TARGET_LOST_DURING_REFINE: (SceneRecoveryAction.RESCAN,),
    GraspFailureReason.REFINEMENT_DIVERGED: (SceneRecoveryAction.RESCAN,),
    # The planner refused to route to a valid grasp and nothing moved. A rescan costs nothing and changes what the
    # planner is asked next; nothing here commands motion after the cell has just declined to move.
    GraspFailureReason.MOTION_PLAN_REFUSED: (SceneRecoveryAction.RESCAN,),
    # Escalations: no recovery is appropriate, surface to the operator instead.
    #
    # A controller that cannot move belongs here: a rescan is pointless, the scene is fine, and it needs a human.
    GraspFailureReason.CONTROLLER_NOT_OPERATIONAL: (),
    # A target label perception never returned belongs here too: not unsafe (the cell is fine) and not fail-closed
    # (nothing refused anything), only pointless. The detector is deterministic on a static frame, so a rescan
    # re-derives the same labels. It never hands the operator an object they did not ask for, which the hard label
    # gate exists to prevent. The operator reads the labels that were seen and re-prompts.
    GraspFailureReason.TARGET_LABEL_NOT_FOUND: (),
    GraspFailureReason.TOPOLOGY_RISK_REJECTED: (),
    GraspFailureReason.SEMANTIC_REJECTED: (),
    # The hint that came with a semantic rejection: the next candidate is the pick loop's own business.
    GraspFailureReason.TRY_NEXT_CANDIDATE: (),
    GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED: (),
}


@dataclass(frozen=True, slots=True)
class RecoveryDispatcher:
    """Pure dispatcher from failure class to the actions it asks for, in order."""

    def actions_for(
        self, reason: GraspFailureReason
    ) -> Tuple[SceneRecoveryAction, ...]:
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
    """One step in the orchestration trail, recorded after the plan ran."""

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


#: The one action this loop runs: it moves nothing, and the next pick re-perceives the scene.
_PLANNED_HERE: frozenset[SceneRecoveryAction] = frozenset({SceneRecoveryAction.RESCAN})


@dataclass(frozen=True, slots=True)
class RecoveryOrchestrator:
    """Pure decision driver for the recovery state machine.

    Given a :class:`SceneRecoveryContext` the orchestrator returns either a concrete :class:`SceneRecoveryPlan` or
    :data:`None`, meaning no further recovery is permissible and the caller terminates. It plans an action of the
    failure's row that both the profile and the policy allow, that was not yet tried for that failure class and whose
    budget is left, and only ``RESCAN``, the one action this loop runs.
    """

    dispatcher: RecoveryDispatcher

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
            Optional richer history with failure-class anti-loop tracking. When supplied the orchestrator refuses to
            plan an action that has already been tried for the same failure class.
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

        # Prefer the explicit kwarg; fall back to the context's ``typed_history`` sidecar.
        merged_typed = tuple(typed_history) + tuple(context.typed_history)
        tried_pairs: set[Tuple[SceneRecoveryAction, GraspFailureReason]] = {
            (e.action, e.failure_class)
            for e in merged_typed
            if hasattr(e, "action") and hasattr(e, "failure_class")
        }
        # Per-action budget consumption count: how many history actions equal each action.
        action_counts: dict[SceneRecoveryAction, int] = {}
        for h in context.history:
            action_counts[h] = action_counts.get(h, 0) + 1

        for reason in context.failure_reasons:
            for action in self.dispatcher.actions_for(reason):
                if action not in _PLANNED_HERE:
                    continue
                # Outer gate: profile must allow this action.
                if action.value not in tuple(context.profile.recovery_allowed_actions):
                    continue
                # Inner gate: policy must allow this action.
                if action not in policy.allowed_actions:
                    continue
                # Anti-loop: same (action, class) pair already tried.
                if (action, reason) in tried_pairs:
                    continue
                # Per-action budget.
                budget = policy.per_action_budget.get(action)
                if budget is not None and action_counts.get(action, 0) >= int(budget):
                    continue
                return SceneRecoveryPlan(
                    action=action,
                    reason=f"dispatcher:{action.value}",
                    telemetry={"orchestrator": "direct_action"},
                )
        return None


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
) -> tuple[Any, RecoveryTrail]:
    """Bounded recovery driver: pick, look again, pick again.

    The driver:

    1. Calls ``pick()``.
    2. If the outcome is success, returns immediately with a ``recovered_success`` trail.
    3. Otherwise extracts typed failure reasons, asks the orchestrator for the next plan, re-acquires perception
       (``frame_acquirer``), appends to the trail, and retries.
    4. Terminates with one of the typed :data:`VALID_TERMINAL_REASONS`.

    The loop is bounded by ``policy.max_recovery_actions``, the per-action budget and the orchestrator's anti-loop
    rule.
    """

    # Disabled fast path: emit a typed trail without ever calling the orchestrator.
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

        ctx = SceneRecoveryContext(
            profile=profile,
            policy=policy,
            last_outcome=getattr(last_report, "outcome"),
            failure_reasons=failure_reasons,
            history=tuple(action_history),
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

        # The plan is a rescan: the next pick re-perceives the scene.
        frame_acquirer()
        outcome = "completed"
        primary_failure = failure_reasons[0]
        logger.info(
            "Recovery action %d: %s for %s (%s) -> executed=%s outcome=%s",
            len(trail_entries),
            plan.action.value,
            primary_failure.value,
            plan.reason,
            True,
            outcome,
        )
        trail_entries.append(
            RecoveryTrailEntry(
                attempt_index=len(trail_entries),
                failure_reason=primary_failure,
                plan_action=plan.action,
                plan_reason=plan.reason,
                executed=True,
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
        last_report = pick()


#: The :class:`AutonomousGraspOutcome` value a finished recovery loop can stand for, spelled here so this package
#: imports no service module (a test pins it against the enum).
OUTCOME_RECOVERY_EXHAUSTED = "recovery_exhausted"


def terminal_outcome_for_trail(trail: RecoveryTrail) -> Optional[str]:
    """The service outcome a finished recovery loop stands for, or ``None`` to keep the last pick's own.

    * ``recovery_exhausted``: at least one recovery action ran and the loop ended because none was left
      (``exhausted_budget``, ``anti_loop_blocked``).
    * ``None`` otherwise: success, a disabled loop, or one that ended on a failure no action recovers (a stopped
      controller, a label nobody saw). That pick's own outcome says more.
    """

    if trail.terminal_reason in ("exhausted_budget", "anti_loop_blocked") and any(
        entry.executed for entry in trail.entries
    ):
        return OUTCOME_RECOVERY_EXHAUSTED
    return None


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
    # Anti-loop: whether an action that ran was the one blocked for a failure reason it was already tried for.
    tried_pairs = {(e.action, e.failure_class) for e in typed_history}
    for reason in failure_reasons:
        for action in orchestrator.dispatcher.actions_for(reason):
            if (action, reason) in tried_pairs:
                return "anti_loop_blocked"
    return "escalated_no_recovery"
