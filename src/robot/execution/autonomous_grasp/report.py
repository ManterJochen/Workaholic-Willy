"""High-level pick-outcome enum and the typed ``AutonomousGraspReport`` aggregate.

Depends only on the config value objects (:mod:`.config`) and the downstream
telemetry types the report composes, never on the service, so the dependency
stays one-directional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Optional

from src.robot.execution.runtime_pick import PickSessionReport
from src.robot.grasping.decision import DecisionReport
from src.robot.grasping.multiview.fusion import FusionTelemetry
from src.robot.grasping.loop.pick_loop import PickOutcome
from src.robot.grasping.rl.router import ShadowRouterTelemetry
from src.robot.grasping.scoring.success_probability import (
    RankingBlendTelemetry,
    ShadowSuccessTelemetry,
    UncertaintyRerankTelemetry,
)
from src.robot.grasping.uncertainty import UncertaintySnapshot

from .config import EffectiveGraspingConfig, GraspBehaviorProfile, GraspMode

if TYPE_CHECKING:
    from src.robot.grasping.loop.pick_loop import CommitDecision


class AutonomousGraspOutcome(StrEnum):
    """Terminal status of an :meth:`AutonomousGraspService.pick` call.

    Kept separate from :class:`PickOutcome` so the high-level service can add
    outcome categories (refinement, verification, recovery) without widening the
    orchestrator-level taxonomy.
    """

    SUCCEEDED = "succeeded"
    NO_TARGET = "no_target"
    NO_VALID_GRASP = "no_valid_grasp"
    EXECUTION_FAILED = "execution_failed"
    # Reserved for later phases; declared now so callers can already
    # write exhaustive switches against the enum.
    REFINEMENT_FAILED = "refinement_failed"
    VERIFICATION_FAILED = "verification_failed"
    RECOVERY_EXHAUSTED = "recovery_exhausted"
    UNSAFE_RECOVERY_REFUSED = "unsafe_recovery_refused"
    # Fail-closed frame contract: a successful candidate was produced
    # but the execution policy refused because no :class:`FrameResolver`
    # was wired and the grasp would have been commanded in camera frame.
    MISSING_CAMERA_FRAME = "missing_camera_frame"
    # Refiner could not re-identify the target in the second-scan frame.
    # Distinct from :attr:`NO_VALID_GRASP` so the operator can
    # distinguish "never had a candidate" from "lost sight of it after
    # the standoff move".
    TARGET_LOST_DURING_REFINE = "target_lost_during_refine"
    # Refined grasp exceeded at least one configured correction bound.
    # Refuse-to-execute signal.
    REFINEMENT_DIVERGED = "refinement_diverged"
    # Honest "this mode is not yet implemented" signal for CLOSED_LOOP /
    # DENSE_AUTONOMOUS until refinement/verification/recovery land.
    MODE_NOT_AVAILABLE = "mode_not_available"
    # AUTO fail-closed decision layer terminal outcomes.
    # ``DECISION_FAIL_CLOSED`` is emitted when the pre-execution
    # :class:`DecisionEngine` refuses to dispatch on real hardware
    # (low confidence + exhausted re-observation budget or no
    # viewpoint planner). ``DECISION_RECOVER_PENDING`` is the typed
    # "recovery requested" signal for the no-candidates branch: the
    # decision layer does not execute the recovery state machine, so
    # the report carries the typed :class:`DecisionReport` and tells
    # the operator that physical recovery is required.
    DECISION_FAIL_CLOSED = "decision_fail_closed"
    DECISION_RECOVER_PENDING = "decision_recover_pending"
    # Uncertainty fusion driven fail-closed. Distinct from
    # :attr:`DECISION_FAIL_CLOSED` so operators can attribute fail-
    # closures to the fused-channel pathway specifically. Carries
    # :class:`DecisionReasonCode.UNCERTAINTY_FAIL_CLOSED` on the
    # decision report.
    UNCERTAINTY_FAIL_CLOSED = "uncertainty_fail_closed"
    # Mandatory multi-view commit gate refused the winning candidate and
    # the configured re-observation budget was exhausted. Distinct from
    # :attr:`NO_VALID_GRASP` so the operator can tell apart "no candidate
    # at all" from "had a candidate but the fused evidence was
    # insufficient".
    NO_COMMIT_INSUFFICIENT_FUSION = "no_commit_insufficient_fusion"
    # Drift + OOD watchdog blocks AUTO on real hardware at the HIGH and
    # SEVERE drift severities (hardware lock). Distinct outcomes so replay
    # can separate drift-driven from OOD-driven fail-closures and from
    # generic decision fail-closures.
    DRIFT_BLOCKED_AUTO = "drift_blocked_auto"
    OOD_BLOCKED_AUTO = "ood_blocked_auto"
    # An operator asked the run to stop between attempts. A first-class outcome rather than a
    # flavour of EXECUTION_FAILED: a stop is a decision, not a failed grasp, so it must not count
    # against the measured pick rate. Replay and the KPI roll-up separate "the cell could not"
    # from "a person said no".
    CANCELLED = "cancelled"


# Mapping from low-level orchestrator outcomes to high-level outcomes.
# Anything not in this table is treated as ``EXECUTION_FAILED`` so a
# new low-level outcome never silently maps to ``SUCCEEDED``.
_PICK_TO_AUTONOMOUS: Mapping[PickOutcome, AutonomousGraspOutcome] = {
    PickOutcome.EXECUTED: AutonomousGraspOutcome.SUCCEEDED,
    PickOutcome.NO_PERCEPTION: AutonomousGraspOutcome.NO_TARGET,
    PickOutcome.RESCANNED_EXHAUSTED: AutonomousGraspOutcome.NO_VALID_GRASP,
    PickOutcome.RELOCATED_EXHAUSTED: AutonomousGraspOutcome.NO_VALID_GRASP,
    PickOutcome.OBJECT_NOT_DETECTED: AutonomousGraspOutcome.VERIFICATION_FAILED,
    PickOutcome.EXECUTION_FAILED: AutonomousGraspOutcome.EXECUTION_FAILED,
    PickOutcome.ABORTED: AutonomousGraspOutcome.EXECUTION_FAILED,
    PickOutcome.CANCELLED: AutonomousGraspOutcome.CANCELLED,
    # A stopped or powered-down controller maps to CANCELLED for the same reason CANCELLED itself
    # exists: the cell stopped, the grasp did not fail, and EXECUTION_FAILED would file a machine
    # state an operator must act on under "bad grasps". The autonomous enum is deliberately not
    # widened: at that level "the run did not complete because the cell stopped" is one fact, and
    # the PickOutcome underneath keeps the distinction for anyone who needs it.
    PickOutcome.CONTROLLER_NOT_OPERATIONAL: AutonomousGraspOutcome.CANCELLED,
    PickOutcome.CAMERA_FRAME_REJECTED: AutonomousGraspOutcome.MISSING_CAMERA_FRAME,
    PickOutcome.NO_COMMIT_INSUFFICIENT_FUSION: AutonomousGraspOutcome.NO_COMMIT_INSUFFICIENT_FUSION,
}


@dataclass(frozen=True, slots=True)
class AutonomousGraspReport:
    """Typed aggregate result of an :meth:`AutonomousGraspService.pick` call.

    This report composes :class:`PickSessionReport` rather than
    widening it. Callers that only care about the orchestrator-level
    detail can read :attr:`pick_report` and ignore the rest.

    Attributes
    ----------
    outcome
        High-level :class:`AutonomousGraspOutcome`.
    mode
        :class:`GraspMode` the operator requested for this attempt.
    profile
        Snapshot of the :class:`GraspBehaviorProfile` that was in
        effect for the attempt. Preserves the locked toggles even
        if a future operator reconfigures the service.
    pick_report
        Underlying :class:`PickSessionReport` produced by the wrapped
        :class:`RuntimePickService`. :data:`None` only when the service
        refused to dispatch (e.g. ``MODE_NOT_AVAILABLE``).
    telemetry
        Free-form key/value bag for forward-compatible diagnostics.
        Keys are stable strings; values are JSON-serializable.
    effective_config
        Snapshot of the ``robot.grasping`` block that wired
        the service. :data:`None` for services built via
        :meth:`AutonomousGraspService.from_components`, and when the
        caller passed an explicit ``mode`` override to
        :meth:`AutonomousGraspService.from_robot_config` without a
        declared ``grasping`` block. The snapshot is stable for the
        lifetime of the service; the report propagates the reference
        so downstream consumers can correlate behaviour to the
        configuration in effect.
    """

    outcome: AutonomousGraspOutcome
    mode: GraspMode
    profile: GraspBehaviorProfile
    pick_report: Optional[PickSessionReport] = None
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    effective_config: Optional[EffectiveGraspingConfig] = None
    # Typed :class:`DecisionReport` from the pre-execution decision
    # layer. :data:`None` when no engine was wired or the active mode
    # is out of scope (e.g. :attr:`GraspMode.EASY`).
    decision: Optional[DecisionReport] = None
    # Typed uncertainty fusion snapshot; populated on every report.
    # When the layer is off the default factory ships a
    # ``UncertaintySnapshot.disabled(...)`` carrier (fused=0.0,
    # fused_available=False, fail_closed=False).
    uncertainty: UncertaintySnapshot = field(
        default_factory=lambda: UncertaintySnapshot.disabled(
            fail_closed_threshold=0.4,
        )
    )
    # Shadow-mode success-probability telemetry for the executed grasp
    # (or the attempted-and-failed grasp). ``None`` whenever the service
    # was built without an artifact or the underlying pick never scored
    # a candidate. Always present (and non-``None``) when the artifact is
    # loaded and at least one candidate reached execution.
    shadow_success_telemetry: Optional[ShadowSuccessTelemetry] = None
    # Bounded probability-blend reranker audit telemetry forwarded
    # verbatim from :class:`PickReport`. ``None`` whenever the service
    # was built without ``ranking_blend_enabled=True`` in
    # ``success_model`` (the silent byte-identical path). When
    # non-``None``, ``applied`` indicates whether the blend actually
    # fired and ``skip_reason`` names the gate.
    ranking_blend_telemetry: Optional[RankingBlendTelemetry] = None
    # Per-candidate uncertainty re-rank telemetry (mirrors ranking_blend_telemetry). ``None``
    # for legacy/disabled deployments and on the pre-rerank refine early-returns.
    uncertainty_rerank_telemetry: Optional[UncertaintyRerankTelemetry] = None
    # Shadow-only multi-view fusion telemetry forwarded verbatim from
    # :class:`PickReport`. ``None`` whenever ``fusion.enabled=False`` in
    # the robot config or the service was built without a frame resolver.
    # Zero decision influence by construction (orchestrator never reads
    # it).
    fusion_telemetry: Optional[FusionTelemetry] = None
    # Typed multi-view commit-gate decision forwarded verbatim from
    # :class:`PickReport`. ``None`` whenever the service exited before
    # any candidate was selected. When the gate is configured but
    # disabled the decision still surfaces with ``allowed=True`` and a
    # typed ``reason`` so operators can confirm "gate was running and
    # approved" versus "gate was off".
    commit_decision: Optional["CommitDecision"] = None
    # Typed candidate-selection shadow-router telemetry. ``None``
    # whenever the service was built without a :class:`ShadowRouter`,
    # which is the default; operator opt-in via ``robot.rl.mode ==
    # "rl_shadow"``. Always ``None`` for the ``geometry_only`` /
    # ``hybrid_ml`` modes. Carrier-only by construction: the
    # orchestrator never reads it.
    shadow_router_telemetry: Optional[ShadowRouterTelemetry] = None
    # Per-step recovery trail
    # (``{action, outcome, step_result, executed, failure_reason, plan_reason}`` per step). Empty ``()`` on
    # the default path; populated by ``_attach_recovery_trail`` only when the opt-in recovery loop ran. The
    # serializer copies it into ``GraspAttemptRecord.recovery_actions`` (an existing top-level field) so
    # offline ``train_recovery`` can walk the per-step (action, reward) sequence. Carrier-only for the pick path.
    recovery_actions: tuple[Mapping[str, Any], ...] = ()

    @property
    def succeeded(self) -> bool:
        """:data:`True` iff the grasp completed successfully."""

        return self.outcome is AutonomousGraspOutcome.SUCCEEDED

    def failure_summary(self) -> str:
        """Why this attempt did not succeed, in one operator-readable line.

        There is no ``failure_reason`` field on this report: the outcome enum plus the telemetry
        bag is the reason surface. The lookup lives here so the CLI and the console give the same
        answer rather than each guessing at it.

        The line carries the outcome plus whichever of three sources is present:

        * ``telemetry['low_level_outcome']``, what the pick loop terminated on, which is
          finer-grained than the service-level outcome (``no_grasp_found`` vs ``approach_path_blocked``);
        * ``telemetry['runtime_error_type']``, present only when the attempt raised;
        * the last attempt's typed ``reasons``, why the candidates were rejected.
        """
        if self.succeeded:
            return ""
        parts: list[str] = [str(self.outcome)]
        low_level = self.telemetry.get("low_level_outcome")
        if low_level and str(low_level) != str(self.outcome):
            parts.append(f"low_level={low_level}")
        if error_type := self.telemetry.get("runtime_error_type"):
            parts.append(f"raised={error_type}")
        attempts = getattr(self.pick_report, "attempts", ()) or ()
        if attempts and (reasons := getattr(attempts[-1], "reasons", ())):
            parts.append("reasons=" + ",".join(str(r) for r in reasons))
        return "  ".join(parts)

    #: The optional layers, and the attribute that proves each one ran rather than being configured.
    #:
    #: Each entry is a field that is ``None`` when the layer did not run, never a config flag, so
    #: this table cannot say "on" for something that produced nothing.
    _LAYERS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("decision", "decision"),
        ("fusion", "fusion_telemetry"),
        ("success-model", "shadow_success_telemetry"),
        ("blend", "ranking_blend_telemetry"),
        ("rl-shadow", "shadow_router_telemetry"),
    )

    def layers_that_ran(self) -> tuple[str, ...]:
        """The optional layers that actually produced something on this attempt.

        The commit gate is not on the ``None`` table above. :class:`CommitDecision` is attached to
        ``PickReport.commit_decision`` on every pick with a winning candidate, and a disabled or
        skipped gate still reports ``allowed=True`` with a typed ``reason``. So
        ``commit_decision is not None`` means a candidate won, not that the gate ran.

        The gate counts as having participated only when its ``reason`` is outside the
        ``skipped_*`` family. A reason this method has never seen counts as participation rather
        than as a skip: a new skip reason reading as "it ran" is the over-reporting this method
        exists to prevent, and the opposite mistake is only noise.

        A fragment, so it is not called ``render``. :meth:`render` composes it.
        """
        ran = [name for name, attribute in self._LAYERS
               if getattr(self, attribute, None) is not None]
        commit = self.commit_decision
        if commit is not None and not str(getattr(commit, "reason", "")).startswith("skipped_"):
            ran.append("commit")
        if self.uncertainty.fused_available:
            ran.append("uncertainty")
        if self.recovery_actions:
            ran.append("recovery")
        return tuple(ran)

    def render(self) -> str:
        """The whole attempt, for a person. ASCII, no trailing newline, no arguments.

        Every advanced block in this stack ships `enabled: false`, so the default pick is
        open-loop: no AUTO decision gate, no closed-loop refine, no fusion commit, no learned
        success model, no RL. The layers line is what keeps a report from reading the same whether
        one layer ran or six.

        It names what produced something, read off fields that are ``None`` when the layer did not
        run, never off the config that asked for it. "(none)" is the expected answer on a default
        cell and is printed rather than omitted, so the line reads as a fact rather than as an
        oversight.

        The failure line is :meth:`failure_summary`, not a second lookup, so the CLI and the
        console give one answer.
        """
        pick = self.pick_report
        head = f"  outcome    {str(self.outcome).upper():<28} mode={self.mode}"
        lines = [head]
        if not self.succeeded:
            lines.append(f"             {self.failure_summary()}")
        if pick is not None:
            where = "simulated" if getattr(pick, "is_simulated", False) else "real"
            lines.append(
                f"  cell       {getattr(pick, 'robot_vendor', '?')} ({where}), "
                f"gripper {'present' if getattr(pick, 'gripper_present', False) else 'absent'}"
            )
            chosen = getattr(pick, "target_index", None)
            lines.append(
                f"  candidates {getattr(pick, 'candidate_count', 0)}"
                + (f", executed #{chosen} at score {getattr(pick, 'selected_score', 0.0):.3f}"
                   if chosen is not None else ", none executed")
            )
        ran = self.layers_that_ran()
        lines.append(f"  layers     {', '.join(ran) if ran else '(none)'}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The wire half. Plain data, `json.dumps`-safe with no custom encoder.

        A view, never a second computation. Every nested block is asked for its own ``to_dict``
        rather than being re-derived here; a number computed here that :meth:`render` does not
        compute would make the two halves two answers about one attempt.

        ``effective_config`` is deliberately not inlined. It is the flat 82-key telemetry contract,
        it has its own ``to_dict``, and a consumer that wants it can ask the report for it. Inlining
        it would make this dict far larger than the attempt it describes.
        """
        pick = self.pick_report
        return {
            "outcome": str(self.outcome),
            "mode": str(self.mode),
            "succeeded": self.succeeded,
            "failure_summary": self.failure_summary(),
            "layers_that_ran": list(self.layers_that_ran()),
            "robot_vendor": getattr(pick, "robot_vendor", None),
            "is_simulated": getattr(pick, "is_simulated", None),
            "gripper_present": getattr(pick, "gripper_present", None),
            "candidate_count": getattr(pick, "candidate_count", 0),
            "target_index": getattr(pick, "target_index", None),
            "selected_score": float(getattr(pick, "selected_score", 0.0) or 0.0),
            "uncertainty": self.uncertainty.to_dict(),
            "decision": self.decision.to_dict() if self.decision is not None else None,
            "recovery_actions": [dict(action) for action in self.recovery_actions],
            "telemetry": dict(self.telemetry),
        }
