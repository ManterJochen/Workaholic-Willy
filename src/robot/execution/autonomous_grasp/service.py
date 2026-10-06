"""Autonomous grasp service: the operator-facing "one call to pick" surface.

This module is the high-level facade an operator or upstream API layer calls to
perform a grasp without having to assemble :class:`GraspCalculator`,
:class:`BinPickingOrchestrator`, execution policies and recovery strategies by
hand.

A per-call ``mode`` whose sampler differs from the one the service was built
with is refused with an honest :class:`AutonomousGraspOutcome`
(``MODE_NOT_AVAILABLE``) rather than run under the wrong sampler. The two-scan
pre-grasp refinement and the ``closed_loop`` / ``dense_autonomous`` modes that
ran it were removed on 2026-09-29, and the post-grasp verification stage and the
``dense_recovery`` slots the same day: no pick path consulted them. Whether a
close holds a part is the execution policy's own post-close check of the
gripper's hold evidence, and the recovery a pick runs is
``robot.grasping.recovery``.

Contract:

* :class:`GraspMode` wraps :class:`GraspSamplingMode`; the low-level enum is
  untouched and existing callers keep working.
* :attr:`GraspMode.EASY` is byte-equivalent to the :class:`RuntimePickService`
  happy path with ``SINGLE_OBJECT`` sampling.
* :class:`AutonomousGraspReport` composes :class:`PickSessionReport`; it does
  not widen it.
* The decision and shadow-router slots are exposed as typed fields with no-op
  defaults.

This module imports only vendor-neutral surfaces. It is safe to import on macOS
with no hardware drivers installed.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional


from src.contracts import UNSET, Maybe, chosen
from src.robot.constants import create_robot_logger
from src.robot.core import (
    NO_PLAN_FAIL_SAFE_MESSAGE,
    Gripper,
    JointPositions,
    MotionStatus,
    RobotArm,
    RobotError,
)
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.core.gripper import toggle_without_sensor_of, why_toggle_count_unknown
from src.robot.execution.looks import Look, LookPose, look_label, looks_of, move_to_look
from src.robot.execution.motion import MotionOutcome, MotionReport
from src.robot.grasping.types.feedback import GraspFailureReason
from src.robot.execution.runtime_pick import (
    RuntimePickService,
)
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
from src.robot.grasping.motion.frame_resolver import FrameResolver
from src.robot.grasping.motion.grasp_motion import (
    GraspMotion,
    build_execution_policy,
    foreign_policy_refusal,
)
from src.robot.grasping.types.perception import MultiCameraPerceptionSource
from src.robot.grasping.generation.calculator import GraspCalculator
from src.robot.grasping.loop.pick_loop import (
    PerceptionSource,
    PickOutcome,
    judged_faces_turned_away,
)
from src.robot.grasping.recovery.policy import (
    SceneRecoveryAction,
    SceneRecoveryPolicy,
)
from src.robot.grasping.recovery.orchestrator import (
    OUTCOME_RECOVERY_EXHAUSTED,
    OUTCOME_UNSAFE_RECOVERY_REFUSED,
    RecoveryDispatcher,
    RecoveryOrchestrator,
    RecoveryTrail,
    RecoveryTrailEntry,
    run_recovery_loop,
    terminal_outcome_for_trail,
)

from .builders import (
    apply_orchestrator_overlays,
    build_config_frame_resolver,
    build_decision_layer,
    build_effective_config,
    build_runtime,
)
from .config import (
    EffectiveGraspingConfig,
    GraspBehaviorProfile,
    GraspMode,
    GraspModeInput,
    _profile_for,
    resolve_grasp_mode,
)
from .latency import LatencyTelemetryCoordinator
from .prompt import PickPrompt
from .report import (
    _PICK_TO_AUTONOMOUS,
    AutonomousGraspOutcome,
    AutonomousGraspReport,
)
from .shadow import attach_shadow_router, maybe_build_shadow_router
from .watchdog import WatchdogCoordinator
from src.robot.grasping.decision import (
    DecisionAction,
    DecisionEngine,
    DecisionPolicy,
    DecisionReasonCode,
    DecisionReport,
)
from src.robot.grasping.telemetry.latency_tracker import (
    LatencyStage,
    LatencyTracker,
)
from src.robot.grasping.uncertainty import (
    UncertaintyCalibration,
    UncertaintyChannelValues,
    UncertaintySnapshot,
    UncertaintyWeights,
    fuse_uncertainty,
    should_bias_recovery_for_uncertainty,
)
from src.robot.execution.calibration_watchdog import (
    WatchdogAction,
    WatchdogPolicy,
    WatchdogReport,
    WatchdogSample,
)
from src.robot.events import (
    RobotWatchdogEventListener,
)
# The shadow router is the only RL-package surface that lands in the runtime.
# ``ShadowRouter`` is a typed dataclass field on the service; its telemetry twin
# ``ShadowRouterTelemetry`` is a field on the report and is imported there. The
# wiring and post-processing logic lives in ``shadow``, imported above. The
# import is cheap (no torch or numpy dependency) and stays inert unless the
# operator opts into ``robot.rl.mode == "rl_shadow"``.
from src.robot.grasping.rl.router import (
    ShadowRouter,
)

if TYPE_CHECKING:
    from src.config.schema import CameraConfig
    from src.config.schema.robot import RobotConfig
    from src.geometry.closing_axis import ClosingAxis
    from src.robot.execution.handling import HandlingReport
    from src.robot.grasping.loop.pick_loop import LookedAround
    from src.robot.grasping.recovery.push_gate import PickPush, PushCampaign, PushCell, PushGate
    from src.robot.grasping.recovery.push_planner import AxisBox, PushRefusal
    from src.robot.grasping.loop.progress import (
        PickProgressListener,
        ShouldCancel,
    )


__all__ = [
    "AutonomousGraspOutcome",
    "AutonomousGraspReport",
    "AutonomousGraspService",
    "DecisionAction",
    "DecisionEngine",
    "DecisionPolicy",
    "DecisionReport",
    "EffectiveGraspingConfig",
    "GraspBehaviorProfile",
    "GraspMode",
    "GraspModeInput",
    "found_nothing",
    "only_excluded",
    "only_kept_out",
    "resolve_grasp_mode",
]


# The `UNSET` sentinel and its `chosen()` guard come from `src/contracts`: a caller that did not
# choose a value passes `UNSET`, never `None` and never a comparison against the default.
# `chosen()` is a `TypeGuard` over `isinstance` rather than an `is` comparison, because
# `Any = object()` is precise at run time and narrows nothing for a type checker. A `TypeGuard`
# narrows only the positive branch, so the value-using arm goes first at every call site below.

# The "uncertainty inactive" sentinels returned by ``_resolve_uncertainty_runtime`` when the operator has not
# opted into fusion (or the active mode is out of ``apply_modes``): a 0.0 fail-closed threshold never trips,
# a 1.01 disagreement threshold never trips (the fused score is clamped to [0, 1]), and a 0.0 ranking penalty is
# a no-op. Together they reproduce the exact baseline decision behaviour.
_UNCERTAINTY_INACTIVE_FAIL_CLOSED_THRESHOLD = 0.0
_UNCERTAINTY_INACTIVE_DISAGREEMENT_THRESHOLD = 1.01
_UNCERTAINTY_INACTIVE_RANKING_PENALTY_WEIGHT = 0.0


def _is_motion_plan_refusal(attempt: Any) -> bool:
    """True when this attempt failed because the planner refused and nothing moved.

    Both halves are required. ``motion_status`` narrows to the timeout family; the message
    constant separates "the planner found no route, no command was sent" from "the command was
    accepted and overran its budget". Only the first is a state an orchestration layer may act on
    without a stop, so the second must not reach the recovery driver as a refusal.

    Duck-typed and total: an attempt that carries neither field yields ``False``.
    """

    status = getattr(attempt, "motion_status", None)
    if status is None or str(status) != str(MotionStatus.TIMEOUT):
        return False
    return str(getattr(attempt, "motion_message", "") or "") == NO_PLAN_FAIL_SAFE_MESSAGE


class _RecoveryReportAdapter:
    """Duck-typed view over an :class:`AutonomousGraspReport` exposing the
    ``outcome`` + ``failure_reasons`` surface that :func:`run_recovery_loop` consumes.
    Failure reasons come from the last attempt of the wrapped report's pick session.

    ``PickAttempt.reasons`` says why no grasp was found; it is empty whenever a grasp was found,
    scored and accepted and the failure happened afterwards, in motion. Without this adapter a
    motion-planner refusal reaches the recovery driver as an empty tuple, the driver has nothing
    to recover from, and it terminates ``escalated_no_recovery`` with zero actions. This is the
    only place that sees the pick report and the recovery driver's contract at once.
    """

    __slots__ = ("report", "outcome", "failure_reasons")

    def __init__(self, report: "AutonomousGraspReport") -> None:
        self.report = report
        self.outcome = report.outcome
        reasons: tuple[Any, ...] = ()
        psr = report.pick_report
        # A recovery that stopped where the arm stands (a push that stopped) leaves nothing to recover from: the loop
        # ends on it with no reason to act on, and no further pick drives the arm back to a look. A person decides.
        if getattr(report, "needs_person", False) is not True and psr is not None and psr.attempts:
            last = psr.attempts[-1]
            reasons = tuple(last.reasons)
            if not reasons and _is_motion_plan_refusal(last) and getattr(last, "action", "") != "push":
                # Narrow on purpose. It fires only when the attempt carries no reason of its own
                # (so it can never displace a real one) and only for the refusal the drivers emit
                # before any command reaches the controller. ``MotionStatus.TIMEOUT`` alone is not
                # enough: on real hardware it also covers a command that was accepted and ran out
                # of budget, where the arm may still be moving, and re-perceiving that is not an
                # orchestration decision to make. The message constant is the discriminator both
                # drivers share; see ``NO_PLAN_FAIL_SAFE_MESSAGE``.
                reasons = (GraspFailureReason.MOTION_PLAN_REFUSED,)
        self.failure_reasons = reasons



#: The service's own lines, the pick path's among them, to the robot log and ``grasp_service.log`` (RC6 of the
#: cell-fix plan: they reached no file).
_LOG: logging.Logger = create_robot_logger(__name__, "grasp_service.log")


def _warn_if_records_will_not_be_trainable(robot_cfg: Any, record_log_path: Any) -> None:
    """Warn once, at boot, when a cell is about to log records a ranker can never learn from.

    The per-candidate feature rows (`extra.rl_candidate_features`, the only thing a pairwise ranker
    can rank) are written exclusively when the ranking shadow ran, and the shadow is built only for
    ``robot.rl.mode == "rl_shadow"``. A cell with any other mode logs records that look healthy: the
    outcome is there, the KPIs roll up, the file grows every pick. The shortfall surfaces only when
    somebody trains and the trainer reports a converged fit over nothing, so the warning fires at
    boot rather than at training time.

    A warning, never a refusal. Record logging also feeds the KPI roll-up, the failure taxonomy and
    every post-mortem a bring-up depends on, and a cell that never intends to train an RL policy has
    good reasons to log; refusing here would be a fail-closed gesture with no safety behind it.

    `python -m src.robot.grasping.rl check-dataset --records <log>` answers the same question
    about a log that already exists.
    """
    mode = str(getattr(getattr(robot_cfg, "rl", None), "mode", "") or "")
    if mode == "rl_shadow":
        return
    _LOG.warning(
        "grasping.record_log_path is set (%s) but robot.rl.mode is %r, not 'rl_shadow': this cell "
        "will log no per-candidate features (extra.rl_candidate_features), which are written only "
        "when the ranking shadow runs. The records stay valid for KPIs, the failure taxonomy and "
        "diagnosis, but a pairwise ranker trained on them has nothing to rank. Set robot.rl.mode: "
        "rl_shadow (with policy_id + artifact_path) if this corpus is meant for training, and check "
        "any existing log with `python -m src.robot.grasping.rl check-dataset --records %s`.",
        record_log_path, mode or "<unset>", record_log_path,
    )


#: What :meth:`AutonomousGraspService.pick` reports instead of raising: a fault of the cell. The robot
#: stack's own errors (a controller link, an e-stop, a camera that could not vouch, a wrist frame taken
#: while the tool moved), a device that stopped delivering and a fused camera the cell refuses to go
#: without (both arrive as ``RuntimeError``, the type the vendor SDKs raise), and an I/O link
#: (``OSError``). Everything else is a programmer error and raises: a ``TypeError`` or an
#: ``AttributeError`` from a wrong double, a ``ValueError`` from a wiring contract, an assertion.
_CELL_FAULTS: tuple[type[Exception], ...] = (RobotError, RuntimeError, OSError)
#: The two ``RuntimeError`` subclasses Python raises for a programmer error rather than for a cell.
_NOT_CELL_FAULTS: tuple[type[Exception], ...] = (NotImplementedError, RecursionError)


def _grounding_sources(orchestrator: Any) -> list[Any]:
    """The perception sources of a cell that ground a phrase: the primary first, then each fused camera.

    Duck-typed on ``set_prompt``. The live camera source has it; a source that grounds no phrase (the
    rehearsal scene, a ground-truth sim source) does not, and keeps its frames.
    """
    candidates: list[Any] = [getattr(orchestrator, "perception", None)]
    rig = getattr(orchestrator, "multi_camera_perception", None)
    candidates.extend((getattr(rig, "sources", None) or {}).values())
    return [source for source in candidates
            if source is not None and callable(getattr(source, "set_prompt", None))]


def _recovery_action(name: str) -> SceneRecoveryAction:
    """The :class:`SceneRecoveryAction` a configured name spells, refusing one removed on purpose.

    Only an unknown name looks the removed table up, so the table's module is imported on the refusal
    alone. An unknown name that was never an action raises the enum's own :class:`ValueError`.
    """

    try:
        return SceneRecoveryAction(name)
    except ValueError:
        from src.config.schema._removed import REMOVED_RECOVERY_ACTIONS  # noqa: PLC0415

        said = REMOVED_RECOVERY_ACTIONS.get(str(name).strip().lower())
        if said is None:
            raise
        raise ValueError(
            f"recovery_orchestrator names the recovery action {name!r}, removed on purpose: {said}"
        ) from None


@dataclass
class AutonomousGraspService:
    """High-level operator-facing grasp service.

    The service wraps an existing :class:`RuntimePickService` and adds:

    * a typed :class:`GraspMode` selector with locked behavior profiles,
    * a high-level :class:`AutonomousGraspReport` outcome,
    * honest refusal of a per-call mode whose sampler the wrapped runtime
      was not built with: no silent degradation.

    The wrapped :class:`RuntimePickService` is the same underlying
    facade used elsewhere. A caller that already has a configured
    :class:`RuntimePickService` wraps it directly via the default
    constructor. A caller that only has components or a
    :class:`RobotConfig` uses :meth:`from_components` or
    :meth:`from_robot_config`, which mirror the wrapped facade's
    constructors one-for-one so no new construction contract is
    invented here.

    The default :class:`GraspMode` is :attr:`GraspMode.AUTO` to match
    the resolution of ``None``.
    """

    runtime: RuntimePickService
    default_mode: GraspMode = GraspMode.AUTO
    # The refinement slots (``refinement_policy``, ``refiner``) left on 2026-09-29 with the
    # two-scan pre-grasp refinement and the two modes that demanded it. The verification slots
    # (``verification_policy``, ``verifier``) and the dense-recovery slots (``recovery_policy``,
    # ``recovery_strategy``) left the same day: no pick path consulted either pair. The hold a pick
    # reports is the execution policy's own post-close check, and the recovery loop builds its
    # policy from ``effective_config`` on every pick (``_build_recovery_orchestrator_policy``).
    # Snapshot of the ``robot.grasping`` config block the service was
    # wired from. :data:`None` for the legacy ``from_components`` path
    # so existing call sites do not have to opt into config-driven
    # wiring just to construct the service.
    effective_config: Optional[EffectiveGraspingConfig] = None
    # AUTO fail-closed decision layer. Both slots are :data:`None` for
    # legacy services so the wrapped pick path runs verbatim.
    # ``from_robot_config`` auto-builds these when
    # ``robot.grasping.decision.enabled`` is :data:`True`.
    decision_policy: Optional[DecisionPolicy] = None
    decision_engine: Optional[DecisionEngine] = None
    # Drift + OOD watchdog rolling history. One sample per completed
    # decision tick is appended to this list (capped at a generous hard
    # ceiling so unbounded growth on a long-running service is
    # impossible). The watchdog itself trims to ``policy.window_size``
    # internally before evaluation. Runtime wiring fills the list as
    # ``pick()`` is called; a caller may pre-populate it to seed a
    # drift trend.
    watchdog_history: list[WatchdogSample] = field(default_factory=list)
    # Optional listener invoked on watchdog state transitions (drift
    # escalating from ``NONE`` to ``MODERATE``, OOD flag rising edge,
    # degraded-mode rising edge, enforced BLOCK_AUTO fired). The
    # listener is best-effort: exceptions raised inside it are swallowed
    # so a buggy subscriber cannot abort the grasp loop.
    watchdog_event_listener: Optional[RobotWatchdogEventListener] = None
    # Optional candidate-selection shadow router. ``None`` by default.
    # Wired by :meth:`from_robot_config` when ``robot.rl.mode ==
    # "rl_shadow"`` and the logistic-policy artifact is available on
    # disk. The service never consults this for execution decisions; it
    # is invoked post-pick to annotate :class:`AutonomousGraspReport`
    # with :attr:`shadow_router_telemetry` and nothing else.
    shadow_router: Optional[ShadowRouter] = None
    #: Where the camera looks from when a program names no look: ``robot.look_joint_positions_deg`` of the cell
    #: profile, read once by :meth:`from_robot_config` into :class:`~src.robot.core.JointPositions`, in order.
    #: ``PickRun`` and the console hand them to every pick whose program names none (a program's own looks override
    #: them), on a fixed camera's cell as on a wrist camera's: a fixed camera's arm moves to them too, and its pick
    #: stops at the first that finds something. :meth:`pick` itself is handed its looks and reads none from here.
    #: Empty for a cell that declares none, and for :meth:`from_components`.
    configured_looks: tuple[LookPose, ...] = ()
    #: What the cell is as a push of a failed part needs it (``nudge_target``, pushed inside the pick attempt): the open
    #: hand from the gripper registry, the workspace, the clearance the arm's line judge keeps and a declared container
    #: (:class:`~src.robot.grasping.recovery.push_gate.PushCell`). :meth:`from_robot_config` reads it where
    #: ``grasping.recovery.allowed_actions`` names ``nudge_target``, and keeps the refusal where it cannot be read;
    #: ``None`` otherwise and for :meth:`from_components`, and then nothing is pushed.
    push_cell: "PushCell | PushRefusal | None" = None

    def __post_init__(self) -> None:
        # Per-attempt latency/SLO coordinator.
        self._latency = LatencyTelemetryCoordinator()
        # Drift/OOD watchdog coordinator. The rolling sample history +
        # event listener stay on the service.
        self._watchdog = WatchdogCoordinator()
        # Opt-in production record logging. ``None`` unless a deployment calls
        # :meth:`enable_record_logging`; ``from_robot_config`` wires it from
        # ``grasping.record_log_path``. When set, :meth:`pick` appends one
        # ``GraspAttemptRecord`` JSONL line per attempt. Off by default, and off
        # is byte-identical.
        self._record_log_path: Optional[Path] = None
        #: Stamped into each record's ``extra`` (robot vendor/model) so two robots' data are not mixed.
        self._record_provenance: Optional[dict[str, Any]] = None
        #: Set by :meth:`set_cancel_check`. Consulted by this service's own retry loop, which sits
        #: above the orchestrator's attempt loop and would otherwise start a fresh pick right after a
        #: cancelled one. Default ``None`` leaves the loop unchanged.
        self._should_cancel: Optional["ShouldCancel"] = None
        #: The campaign the picks belong to (:meth:`start_campaign`): its push budgets, the parts next_target skips
        #: and the push distance. Started on the first pick where no caller started one.
        self._campaign: Optional["PushCampaign"] = None
        #: Why the last pick stopped where the arm stands, a person to decide (its report's ``needs_person``), in that
        #: report's words; ``""`` while nothing waits for a person. Set, it refuses every pick with nothing asked or
        #: commanded until :meth:`start_campaign` or :meth:`acknowledge_needs_person` clears it.
        self._needs_person: str = ""
        #: Why this service's picks cannot push, each said once (a WARNING per reason, not per pick).
        self._push_said: set[str] = set()
        # Push the wired :class:`ShadowRouter` (if any) onto the
        # underlying orchestrator so the ranking-shadow producer can
        # fire at the pick_loop seam. Never raises: a runtime stack that
        # exposes no orchestrator is skipped silently, and the post-pick
        # reader is also defensive.
        try:
            orchestrator = getattr(
                getattr(self, "runtime", None), "orchestrator", None
            )
            if orchestrator is not None and self.shadow_router is not None:
                # ``shadow_router`` is a plain dataclass field on
                # ``BinPickingOrchestrator``; direct assignment is the
                # same idiom the overlay wiring uses
                # (e.g. ``runtime.orchestrator.frame_resolver = ...``).
                orchestrator.shadow_router = self.shadow_router
        except Exception:  # noqa: BLE001 (shadow-router attach is fail-safe: degrade to no-shadow, never break setup)
            return

    # Construction helpers

    @classmethod
    def from_components(
        cls,
        *,
        arm: RobotArm,
        calculator: GraspCalculator,
        perception: PerceptionSource,
        mode: GraspMode | str | None = None,
        gripper: Gripper | None = None,
        max_attempts: int = 5,
        standoff_mm: float = 80.0,
        retreat_mm: float = 100.0,
        policy: GraspExecutionPolicy | None = None,
        motion: "Maybe[GraspMotion]" = UNSET,
        frame_resolver: FrameResolver | None = None,
        decision_policy: DecisionPolicy | None = None,
        decision_engine: DecisionEngine | None = None,
        record_log_path: "str | Path | None" = None,
    ) -> "AutonomousGraspService":
        """Build the service directly from raw components.

        ``motion`` is how the pick moves (:class:`GraspMotion`): the service builds the one policy it
        drives from it, on ``arm`` and ``gripper``, with the base frame guard where ``frame_resolver``
        is wired, the dwell gate ``arm.config`` asks for, and the jaws opened before every approach.
        ``policy`` is the older spelling and is still accepted: a policy whose arm or hand is not
        ``arm`` or ``gripper`` is refused (``ValueError``), because it would move a robot this service
        never checked, and passing both is a ``TypeError``.

        ``mode`` selects the locked behavior profile that the service
        applies on every :meth:`pick` call unless that call passes
        an explicit per-call ``mode`` override. The
        :class:`GraspSamplingMode` consumed by the wrapped
        :class:`RuntimePickService` is always derived from the
        profile; a free-form sampling mode is not accepted here, which
        is what prevents profile/sampling drift.

        When ``frame_resolver`` is provided the service:

        * forwards the resolver to the underlying
          :class:`BinPickingOrchestrator` so the calculator receives
          a per-frame ``T_cam_to_base`` and produces base-frame
          candidates, and
        * configures the default :class:`GraspExecutionPolicy` with
          ``require_base_frame_grasp=True`` so a camera-frame grasp
          is refused before motion, closing the frame contract
          end-to-end. Passing an explicit ``policy`` keeps full control
          with the caller; the service does not mutate a caller-supplied
          policy.
        """

        if policy is not None and chosen(motion):
            raise TypeError("pass motion=GraspMotion(...) or policy=..., not both")
        if policy is not None:
            refusal = foreign_policy_refusal(policy, arm=arm, gripper=gripper)
            if refusal:
                raise ValueError(refusal)
        resolved_mode = resolve_grasp_mode(mode)
        profile = _profile_for(resolved_mode)
        # Build a default policy that fails closed when a resolver is
        # wired. An explicit caller-supplied policy is honoured
        # verbatim; mutating an externally owned object would be a
        # silent footgun.
        effective_policy = policy
        if chosen(motion):
            # The caller's motion, on this service's arm and hand, with every guard: the base frame
            # guard where a resolver is wired, the dwell gate the arm's tree asks for, and the jaws
            # opened before every approach.
            standoff_mm, retreat_mm = motion.standoff_and_retreat(standoff_mm, retreat_mm)
            effective_policy = build_execution_policy(
                motion, arm=arm, gripper=gripper, standoff_mm=standoff_mm, retreat_mm=retreat_mm,
                base_frame_required=frame_resolver is not None,
                dwell=getattr(getattr(getattr(arm, "config", None), "safety", None), "dwell", None),
            )
        elif effective_policy is None and frame_resolver is not None:
            effective_policy = GraspExecutionPolicy(
                arm=arm,
                gripper=gripper,
                standoff_mm=standoff_mm,
                retreat_mm=retreat_mm,
                require_base_frame_grasp=True,
                # Without this the jaws are never opened on the real path. `pre_open_width_mm`
                # defaults to None; the willy_sim runners set it, as does the policy
                # `from_robot_config` builds (builders.py), and nothing on this path releases
                # either. The arm then descends with the jaws wherever the previous close left
                # them, and the next close to a wider target reaches the gripper as an opening:
                # measured, it dropped the held part onto the new grasp point and gripped nothing,
                # while the attempt was recorded as EXECUTED.
                #
                # This does not make the service place parts, and it is not meant to. It moves the
                # unavoidable opening from the grasp point to the cell's own start pose, before the
                # approach, where it is expected. A caller that needs the part kept must place it
                # between picks.
                #
                # Derived from the hand rather than a constant: the 2F-85 opens 85 mm and the Hand-E
                # 49.99, and a number written here would be right for one of them.
                pre_open_width_mm=(
                    float(gripper.max_width_mm)
                    if gripper is not None and getattr(gripper, "max_width_mm", None) is not None
                    else None
                ),
            )
        runtime = RuntimePickService.from_components(
            arm=arm,
            calculator=calculator,
            perception=perception,
            gripper=gripper,
            max_attempts=max_attempts,
            grasp_sampling_mode=profile.sampling_mode,
            standoff_mm=standoff_mm,
            retreat_mm=retreat_mm,
            policy=effective_policy,
        )
        if frame_resolver is not None:
            # Attach the resolver to the orchestrator the runtime just
            # built. ``frame_resolver`` is part of the orchestrator's
            # public dataclass surface, so direct assignment is the
            # supported idiom.
            runtime.orchestrator.frame_resolver = frame_resolver
        service = cls(
            runtime=runtime,
            default_mode=resolved_mode,
            decision_policy=(
                decision_engine.policy
                if decision_engine is not None and decision_policy is None
                else decision_policy
            ),
            decision_engine=decision_engine,
        )
        # Opt into production record logging when a path is supplied (None / empty is off).
        if record_log_path:
            service.enable_record_logging(record_log_path)
        return service

    @classmethod
    def from_robot_config(
        cls,
        robot_cfg: RobotConfig,
        *,
        calculator: GraspCalculator,
        perception: PerceptionSource,
        mode: GraspMode | str | None = None,
        max_attempts: "Maybe[int]" = UNSET,
        standoff_mm: float = 80.0,
        retreat_mm: float = 100.0,
        policy: GraspExecutionPolicy | None = None,
        motion: "Maybe[GraspMotion]" = UNSET,
        frame_resolver: FrameResolver | None = None,
        decision_policy: DecisionPolicy | None = None,
        decision_engine: DecisionEngine | None = None,
        arm: "RobotArm | None" = None,
        gripper: "Gripper | None" = None,
        multi_camera_perception: "MultiCameraPerceptionSource | None" = None,
        #: Which camera id is the cell's primary, when the caller knows. It is not config the root
        #: can read: which rig a cell opens is `camera.cameras.primary_rig_id`, in the camera
        #: section, and this class is handed a `RobotConfig`. The overlays need it to leave the
        #: primary out of the list of cameras a fused pick waits for, because the primary delivers
        #: through `perception` and never through the extra-camera rig, so a primary that is expected
        #: there is permanently missing. `None` expects every camera the map names, which is what
        #: every caller got before.
        primary_camera_id: str | None = None,
        #: One calculator per camera id, for a cell that may promote an object from a camera
        #: other than the primary. A calculator is bound to one camera's intrinsics at
        #: construction, so an object seen only by a second camera cannot be computed with the
        #: primary's: it would not fail, it would place a plausible grasp in the wrong lens.
        #: `None` is every cell that cannot promote, which is every cell by default.
        camera_calculators: "dict[str, Any] | None" = None,
        #: The camera section of the tree `robot_cfg` came from. Each camera's calibration is
        #: declared on its rig there, `camera.cameras.rigs[<id>].extrinsics`, so the primary's
        #: resolver and a fused cell's resolver map are built from it. `UNSET` builds no resolver
        #: from config, which a real vendor refuses below unless `frame_resolver` is passed.
        camera: "Maybe[CameraConfig]" = UNSET,
    ) -> "AutonomousGraspService":
        """Build the service from a validated ``RobotConfig`` tree.

        ``arm`` / ``gripper`` accept a live device handle that config cannot describe: the Isaac
        gripper needs the simulator session its arm already lives on. Without that argument such a
        cell has to build through ``from_components``, which leaves ``effective_config=None`` and
        therefore silences the config-driven orchestrator overlays unless a runner re-enables them
        by hand, so the simulator measures a hand-wired stack while a real cell runs the
        config-driven one. Supplying the handle here lets every other decision come from config.

        Mirrors :meth:`RuntimePickService.from_robot_config` and adds
        the config-driven wiring contract:

        * Strict fail-closed. When ``mode`` is not supplied and
          the operator did not explicitly declare a ``grasping`` block
          on the ``RobotConfig``, this method raises
          :class:`ValueError`. Production deployments must opt in to
          autonomous grasping explicitly; relying on the schema default
          would let a misconfigured YAML silently boot a degraded
          mode. Callers that pass ``mode`` explicitly bypass the
          requirement: a caller that already knows which mode it wants
          does not need to populate the full block.
        * Mode and ``max_attempts`` resolution. When the caller
          does not pass these, the values flow from
          ``robot_cfg.grasping.default_mode`` and
          ``robot_cfg.grasping.max_attempts``. Explicit kwargs always
          win.
        * Recovery. ``robot_cfg.grasping.recovery`` reaches the pick through
          the effective-config snapshot. ``container_agitate`` needs the
          fixture envelope declared beside it, ``recovery.fixture``, which
          the schema refuses to go without; the push (``nudge_target``)
          needs none, and a declared one only narrows where it may land
          (owner, 2026-10-01). Where ``nudge_target`` is allowed,
          the cell's push inputs are read here too (:attr:`push_cell`: the
          hand from the gripper registry, the workspace, the clearance the
          arm's line judge keeps, a declared container). The
          ``verification`` and ``dense_recovery`` sub-policies, and the
          ``recovery_fixture`` argument the second one took, left on
          2026-09-29.

        ``frame_resolver`` is plumbed through identically to
        :meth:`from_components`. See that method's docstring for the
        full fail-closed contract.

        ``motion`` (:class:`GraspMotion`) is how the pick moves, built into the one policy on the arm
        and hand this method resolves, with every guard. ``policy`` is still accepted and is refused (``ValueError``)
        unless it drives the arm and hand this method resolved, which only a caller that also passed
        ``arm`` and ``gripper`` can hold.
        """

        if policy is not None and chosen(motion):
            raise TypeError("pass motion=GraspMotion(...) or policy=..., not both")
        if chosen(motion):
            standoff_mm, retreat_mm = motion.standoff_and_retreat(standoff_mm, retreat_mm)

        # Detect whether the operator explicitly declared the ``grasping``
        # block. ``model_fields_set`` is the pydantic-v2 surface that
        # reports which fields were provided to the model constructor,
        # as opposed to filled in by the schema default.
        grasping_declared = "grasping" in getattr(
            robot_cfg, "model_fields_set", set()
        )
        if mode is None and not grasping_declared:
            raise ValueError(
                "AutonomousGraspService.from_robot_config requires "
                "either an explicit ``mode`` argument or a "
                "``robot.grasping`` block explicitly declared on the "
                "RobotConfig. The schema default is intentionally not "
                "trusted on production hosts (Phase T0 fail-closed)."
            )

        grasping_cfg = getattr(robot_cfg, "grasping", None) if grasping_declared else None

        # Resolve mode + max_attempts from config when the caller did not
        # provide them. ``resolve_grasp_mode`` validates string inputs.
        if mode is None:
            mode = grasping_cfg.default_mode  # type: ignore[union-attr]
        resolved_mode = resolve_grasp_mode(mode)
        profile = _profile_for(resolved_mode)

        # The value-using arm comes first because `chosen` is a `TypeGuard`, which narrows the
        # positive branch only. Reversed, `int(max_attempts)` would still see `int | _Unset` and
        # mypy would refuse it.
        if chosen(max_attempts):
            resolved_max_attempts = int(max_attempts)
        else:
            resolved_max_attempts = (
                int(grasping_cfg.max_attempts) if grasping_cfg is not None else 5
            )

        # Decision-layer auto-build.
        resolved_decision_policy, resolved_decision_engine = build_decision_layer(
            grasping_cfg,
            decision_policy=decision_policy,
            decision_engine=decision_engine,
        )

        # Immutable effective-config snapshot (``None`` on the legacy
        # ``mode``-override path).
        effective_config = build_effective_config(
            grasping_cfg,
            resolved_mode=resolved_mode,
            resolved_max_attempts=resolved_max_attempts,
        )

        # Build the primary camera's resolver from its rig's declared calibration when the caller
        # did not supply one. This is what turns a config-built cell's grasps into the base frame.
        # A caller-supplied frame_resolver (sim) wins; no camera section, or a primary rig with no
        # calibration, returns None; a declared artifact that does not load raises (fail-closed).
        if frame_resolver is None:
            frame_resolver = build_config_frame_resolver(grasping_cfg, camera=camera)

        # Fail closed on a real cell with no resolver at all. Without one, `require_base_frame_grasp`
        # is never switched on, the grasp stays `frame=camera`, and every single motion comes back
        # `MotionStatus.INVALID_TARGET` (`URRobotArm.move` requires `Frame.BASE`). That is safe, and
        # it is also indistinguishable at the bench from "the cell is broken": 100% rejection, no
        # hint that the cause is a missing calibration artifact. No shipped camera section declares
        # a rig's calibration, so this is the default state of a freshly configured real cell. Sim
        # and dummy are exempt: they pass their resolver in code, or run without one on purpose. A
        # real wrist camera needs no resolver in code: its CAMERA to TOOL is declared on its rig and
        # built into an eye-in-hand resolver above.
        vendor_name = str(getattr(getattr(robot_cfg, "vendor", ""), "value", getattr(robot_cfg, "vendor", "")))
        if frame_resolver is None and vendor_name.lower() not in ("sim", "dummy"):
            key = (f"camera.cameras.rigs[{camera.cameras.primary_rig_id!r}].extrinsics" if chosen(camera)
                   else "camera.cameras.rigs[<primary rig id>].extrinsics")
            raise ValueError(
                f"no CAMERA->BASE frame resolver for a {vendor_name!r} cell. Perception reports grasps "
                "in the camera frame; without a resolver they are never transformed into the robot's "
                "base frame, so the driver refuses every motion as INVALID_TARGET, which at the bench "
                "looks like a broken cell, not a missing calibration. Fix it one of two ways: (1) declare "
                f"the primary camera's calibration on its rig, {key}, with the mounting_mode and the "
                "artifact_path the calibration CLI wrote, and hand this root the camera section "
                "(build_real_cell does), or (2) pass frame_resolver= explicitly."
            )

        # Underlying runtime pick service (+ fail-closed frame-guard patch).
        runtime = build_runtime(
            robot_cfg,
            calculator=calculator,
            perception=perception,
            resolved_max_attempts=resolved_max_attempts,
            profile=profile,
            standoff_mm=standoff_mm,
            retreat_mm=retreat_mm,
            policy=policy,
            motion=motion,
            frame_resolver=frame_resolver,
            arm=arm,
            gripper=gripper,
        )

        # Orchestrator overlays applied in place.
        from src.robot.safety.planning.hand import hand_past_the_tcp_mm  # noqa: PLC0415

        apply_orchestrator_overlays(
            runtime, grasping_cfg, resolved_mode=resolved_mode,
            primary_camera_id=primary_camera_id, camera=camera,
            hand_past_the_tcp_mm=hand_past_the_tcp_mm(robot_cfg),
            palm_thickness_mm=_registry_palm_thickness_mm(robot_cfg),
        )

        # The half of multi-camera fusion that config cannot carry. The overlays above wire
        # `fusion_geometry_config`, the per-camera resolver map and the list of cameras the config
        # asked for. The rig is a live device handle, so it arrives here the same way `perception`
        # and `arm` do. Without it a physical cell can name three calibrated cameras, load three
        # artifacts fail-closed, and still run single-view.
        #
        # None is the byte-identical default and stays legal with fusion configured: an operator
        # who enabled the block on a cell whose rig is not wired yet gets the loop's warning that
        # fusion is enabled and the cell is running single-view, once per pick, which is the honest
        # answer. Refusing here would break the sim runners, which wire the rig afterwards.
        if multi_camera_perception is not None:
            if runtime.orchestrator is None:
                raise ValueError(
                    "multi_camera_perception was supplied but this service has no orchestrator to "
                    "hold it; the cameras would be opened and never observed."
                )
            runtime.orchestrator.multi_camera_perception = multi_camera_perception

        if runtime.orchestrator is not None:
            # Both of these are labels the orchestrator cannot derive: which camera `perception`
            # streams from lives in the camera section of the config, and this class is handed a
            # `RobotConfig`. Set unconditionally so a caller that passes neither gets the empty
            # string and an absent map, which is what every caller got before they existed.
            runtime.orchestrator.primary_camera_id = primary_camera_id or ""
            runtime.orchestrator.camera_calculators = camera_calculators

        service = cls(
            runtime=runtime,
            default_mode=resolved_mode,
            effective_config=effective_config,
            decision_policy=resolved_decision_policy,
            decision_engine=resolved_decision_engine,
            shadow_router=maybe_build_shadow_router(robot_cfg),
            # In degrees in the tree, as the pendant shows them; the one place they become joints.
            configured_looks=tuple(
                JointPositions.deg(*row) for row in getattr(robot_cfg, "look_joint_positions_deg", None) or ()),
            push_cell=_push_cell_of(robot_cfg, grasping_cfg),
        )
        # Opt into production record logging from config (grasping.record_log_path; default
        # None / empty is off). The loader already applied ${ENV} substitution, so an unset env
        # collapses to "", which counts as off here.
        record_log_path = getattr(grasping_cfg, "record_log_path", None) or None
        if record_log_path:
            # Stamp robot identity so a UR3e cell's records are never silently pooled with the UR5e's.
            # The model comes from the vendor-appropriate field, not `cell_identity()`, which is
            # willy_sim-only and hardcodes `robot_vendor="sim"`. Scene and session provenance are
            # deliberately not stamped: in production those would be fabricated strings and would
            # flip the leakage audit from an honest "skipped" to a dishonest green pass.
            vendor_str = str(getattr(robot_cfg, "vendor", "") or "")
            model = {
                "ur": getattr(getattr(robot_cfg, "ur", None), "model", None),
                "sim": getattr(getattr(robot_cfg, "sim", None), "robot_model", None),
                "kuka": getattr(getattr(robot_cfg, "kuka", None), "model", None),
            }.get(vendor_str)
            service.enable_record_logging(
                record_log_path,
                provenance={"robot_vendor": vendor_str, "robot_model": model or "unknown"},
            )
            _warn_if_records_will_not_be_trainable(robot_cfg, record_log_path)
        return service

    # Public API

    def _maybe_attach_shadow_router(
        self, report: AutonomousGraspReport
    ) -> AutonomousGraspReport:
        """Shadow-router post-processor; delegates to :func:`shadow.attach_shadow_router`.

        Fail-safe; never raises. Shadow routing has zero influence on the executed
        grasp: it only annotates :attr:`AutonomousGraspReport.shadow_router_telemetry`
        and the RL extras dict on ``report.telemetry``.
        """

        return attach_shadow_router(
            shadow_router=self.shadow_router,
            runtime=getattr(self, "runtime", None),
            report=report,
        )

    def pick(
        self,
        *,
        mode: GraspMode | str | None = None,
        look: "Maybe[Look]" = UNSET,
        both_faces: bool = False,
        multi_view: bool = True,
    ) -> AutonomousGraspReport:
        """Run one autonomous pick attempt and return a typed report.

        Parameters
        ----------
        look
            Where the camera looks from before it perceives: one
            :class:`~src.robot.core.JointPositions` (``JointPositions.deg``
            takes degrees), ``"home"``, or several in order
            (:mod:`src.robot.execution.looks`). The arm moves to each
            through its own judged ``move_to_joints`` (or the gated home)
            and the pick perceives there.

            A camera on the wrist hands its looks to the pick loop
            (``BinPickingOrchestrator.look_around``), which fuses each look
            with the ones before it and stops at the first whose grasp is
            valid and carries no rescan reason: what the looks are for is a
            safe grasp point, never a whole scan of the part. A grasp still
            uncertain, or no candidate, goes on to the next look. A look the
            planner refused before anything was sent is skipped and said; a
            pick that reaches none of its looks ends ``EXECUTION_FAILED``
            (``CANCELLED`` on a stopped controller) with nothing perceived,
            and so does a look motion that may have moved the arm. The
            report's ``looks`` names the looks perceived from,
            ``looks_fused`` those fused into the grasp's cloud and
            ``jaw_faces_seen`` which contact faces of the chosen grasp were
            seen. A fixed camera handed looks moves to each in turn and
            stops at the first that finds something, as it always did.

            A wrist pick handed looks has had its rescans: its looks and the
            one view they may generate. Where the cell arms the recovery loop
            (``grasping.recovery``), that loop rescans nothing for it, so a
            rescan never drives them again; only ``next_target`` runs the
            looks again within this call, for another part of the label. A
            pick handed no look rescans where it stands, as it always did.

            Nothing is said to the hand before or between looks, and a
            camera that could not vouch on the way is a fault. Unset
            perceives from where the arm stands: a wrist camera then takes
            a fresh frame there when its grasp calls for another look, the
            one retry a pick told not to move has. ``PickRun`` and the
            console hand every pick ``configured_looks``, else ``"home"`` on
            a wrist camera, when the program names no look. A list that
            names nothing, or an entry that is not a look, raises before
            anything moves.
        both_faces
            Ask that both jaw contact faces of the chosen grasp be seen
            before gripping: the owner's switch for safety-critical
            processes. A wrist pick then looks on until a look shows both,
            its one generated view included. One whose views show both
            nowhere is not gripped: it ends ``NO_VALID_GRASP``, the face not
            seen named in its failure line and ``both_faces`` set on its
            report, whether the pick loop ends it (the open loop, which
            ``PickRun`` and the console run) or the decision layer's pick
            refuses it here. A fixed camera cannot look around, so its pick
            grips only a grasp its cameras (fused, where the cell fuses
            them) showed both faces of, and ends the same way otherwise. The
            grasp gripped is then the one whose faces were judged: the
            reranks and the approach check's fall-back to another candidate
            stand down. The cell's natural orientation
            (``robot.natural_closing_axis``) and a motion naming a closing axis
            (``GraspMotion(closing_axis=...)``) choose the way round before the
            faces are judged, so both work with it. A motion that turns every
            grasp about base Z before it closes
            (``GraspMotion(align_closing_to_base_x=True)``) would close
            the jaws on faces nobody judged, so asking for both together raises
            ``ValueError`` before anything moves. Off, the default and the fast
            one, a look is good enough when its grasp is valid and certain. A
            suction cup has no jaw faces to see. Handed to this pick alone: the
            next pick starts with it off.
        multi_view
            The looks as handed, the default. ``False`` is "Multi-View aus" (the
            owner's Q11, 2026-09-30): the pick looks from the first of its looks
            only, and a wrist pick generates no view after it
            (``BinPickingOrchestrator.generated_view``), so it grips a grasp that
            look found safe or none; a fixed camera handed looks tries the first
            one only. Handed to this pick alone, and taken back when it ends,
            whatever ended it.
        mode
            Optional per-call override for the :class:`GraspMode`. When
            omitted the service uses :attr:`default_mode`. The override
            only changes the behavior profile used to interpret the
            attempt; it does not rebuild the wrapped
            :class:`RuntimePickService`. Switching modes at call time
            therefore cannot change which low-level
            :class:`GraspSamplingMode` the orchestrator uses on this
            attempt, because the wrapped service was bound to a sampling
            mode at construction. The override is restricted to modes
            whose profile uses the same sampling mode as the default;
            anything else is refused with a typed
            :attr:`AutonomousGraspOutcome.MODE_NOT_AVAILABLE` outcome,
            so the report never misstates which sampler ran.

        Returns
        -------
        AutonomousGraspReport
            The high-level typed report. Whenever
            ``pick_report is not None`` the wrapped
            :class:`RuntimePickService` executed; whenever it
            is :data:`None` the service refused to dispatch and the
            telemetry explains why.

            A fault of the cell (a controller link that dropped, an
            e-stop, a camera that could not vouch, a device that stopped
            delivering) is not raised: the outcome is
            ``EXECUTION_FAILED``, :attr:`AutonomousGraspReport.fault`
            carries it and ``pick_report`` is :data:`None`. A programmer
            error still raises.

            A service with a gripper asks the controller first, as
            ``Robot.pick`` does: one that cannot move, or whose state
            cannot be read, ends the pick ``CANCELLED`` before the hand
            is asked anything, with nothing perceived or commanded and
            :attr:`AutonomousGraspReport.controller_stopped` set, so a
            campaign stops on it.

            A hand that toggles with no sensor is asked next, before
            anything of the pick moves; one that will not start (it
            believes its jaws closed and nobody at a terminal said
            otherwise) is ``EXECUTION_FAILED`` with
            :attr:`AutonomousGraspReport.gripper_fault` saying why.

            Every pick belongs to a campaign (:meth:`start_campaign`):
            where the cell's recovery allows it, a wrist pick in
            ``dense_clutter`` pushes a part every grasp of which collided
            with a neighbour within 25 mm (``nudge_target``, inside the
            pick attempt, the jaws read and never switched), within the
            campaign's budgets, and ``next_target`` skips a failed part of
            the label for this pick and the next two. A toggle's count
            nobody can vouch for when the push reads it, or a gripper that
            measures its width found not connected or unreadable then, ends
            the pick as a gripper fault, nothing pushed; so does a toggle's
            count nobody can vouch for before ``next_target`` drives the
            looks again, before any look is driven again. One nobody can
            vouch for before a contact leg of the push, or once the arm is
            up, stops the push where the arm stands.
            A push that stopped
            once something may have moved leaves the arm where it is and
            the report :attr:`AutonomousGraspReport.needs_person`; a
            campaign stops on it, and so does this service: every later
            pick is refused ``UNSAFE_RECOVERY_REFUSED``, needing a person,
            with nothing asked, perceived or commanded, until a person
            decides (:meth:`start_campaign`, which every ``PickRun`` and
            console run calls, or :meth:`acknowledge_needs_person`).
        """

        # A wrong look list is the program's error, raised before anything moves.
        looks = looks_of(look) if chosen(look) else ()
        if not multi_view:
            # Multi-View aus (Q11): the first look alone; the pick loop is told to generate no view after it.
            looks = looks[:1]
        # So is both_faces with a motion that turns every grasp off the faces that were judged, raised before the
        # controller or the hand is asked anything (the pick loop refuses the same, `judged_faces_turned_away`).
        turned = judged_faces_turned_away(getattr(self.runtime, "orchestrator", None)) if both_faces else ""
        if turned:
            raise ValueError(turned)
        # What the last pick's looks came to is not this pick's (`looked_around`): a pick that ends before it looks,
        # on any of the refusals below, leaves none to read, so its views are never taken for the last pick's.
        _looking: Any = getattr(self.runtime, "orchestrator", None)
        if getattr(_looking, "looked_around", None) is not None:
            _looking.looked_around = None
        if getattr(_looking, "ranked_looked", None) is not None:
            _looking.ranked_looked = None
        # A cancel that arrived between two picks is honoured here, before any work starts: a caller
        # running a campaign loops on pick(), and refusing to start is the cheapest possible stop.
        # Default ``None`` costs one attribute read and leaves the body below unchanged.
        if self._should_cancel is not None and self._should_cancel():
            return self._cancelled_report(mode=mode)
        # A recovery of an earlier pick stopped where the arm stands (a push that stopped once something may have moved):
        # a person decides first. The next pick would drive the arm to its look from there, the escape the owner ruled
        # out (2026-09-29), so none starts, and nothing is read off the cell, asked or commanded.
        if self._needs_person:
            return self._needs_person_report(mode=mode)
        # The controller is asked before the hand, as Robot.pick asks it: a toggle that believes its jaws closed asks
        # the person, and a person's 'p' switches DO0, which on an arm still protective-stopped with the part in the
        # jaws drops it from wherever the lift stopped (found auditing the owner's toggle cell, 2026-09-24). A stopped
        # controller, or one whose state cannot be read, ends the pick here with nobody asked and nothing commanded.
        stopped = self._controller_refusal()
        if stopped:
            return self._controller_refused_report(stopped, mode=mode)
        # A hand that toggles with no sensor is asked before anything of the pick moves, a look included (owner's
        # decision, 2026-09-24): never a change there, and on jaws it believes closed it asks the person again, or the
        # pick ends here with nobody to ask. The policy asks once more at the approach.
        try:
            hand = self._hand_refusal()
        except _CELL_FAULTS as exc:
            if isinstance(exc, _NOT_CELL_FAULTS):
                raise
            return self._fault_report(exc, mode=mode)
        if hand:
            return self._hand_refused_report(hand, mode=mode)
        # A pick of the campaign starts: its own two pushes are available again, and the parts next_target skips age.
        self.campaign.start_pick()
        # Install a per-attempt LatencyTracker so the decision seam can record
        # its span via ``self._current_latency_tracker``, and the ranking and
        # fusion seams theirs through the orchestrator. The tracker is always
        # installed (recording is cheap); the typed ``performance_enabled``
        # knob only gates enforcement and SLO_BREACH event emission.
        # The wall-time captured here covers the full pick() body and is
        # exported as ``attempt_wall_time_s`` on the returned report.
        self._current_latency_tracker: Optional[LatencyTracker] = LatencyTracker()
        # Hand it to the orchestrator so the ranking and fusion spans are recorded on every path,
        # including the default open-loop pick.
        _orch = getattr(self.runtime, "orchestrator", None)
        if _orch is not None:
            _orch.latency_tracker = self._current_latency_tracker
        # One attempt_id per pick(), shared across latency / shadow / record-logging
        # telemetry. The decision path mints its own ``auto-<uuid>`` deeper in; this pick-scoped id
        # is the fallback stamped when no path set one. Without it every record logged from the
        # default open-loop path collides on the literal "pick".
        pick_attempt_id = f"pick-{uuid.uuid4().hex[:12]}"
        wall_t0_ns = time.monotonic_ns()
        fault: Optional[Exception] = None
        # What the pick loop kept of its pushes before this pick: a fault below leaves the pushes of the pick that
        # raised there, and only those are this pick's.
        pushes_before = _pushes_of(_orch)
        try:
            report = self._look_and_pick(looks, mode=mode, both_faces=both_faces, multi_view=multi_view)
        except _CELL_FAULTS as exc:
            if isinstance(exc, _NOT_CELL_FAULTS):
                raise
            # A fault of the cell is this pick's answer, not an exception out of it: a verb that
            # promises a report never raises for a domain refusal, and a programmer error still does.
            # A campaign still stops on it, because the report carries the fault and `PickRun`
            # reads it. A push the fault stopped where the arm stands (a camera world that could not vouch mid-push)
            # is said on it too, so the report needs a person.
            fault = exc
            report = _with_the_pushes_of_a_fault(
                self._fault_report(exc, mode=mode),
                tuple(push for push in _pushes_of(_orch) if not any(push is kept for kept in pushes_before)))
        finally:
            # Always tear the tracker down on the way out; subsequent
            # pick() calls install a fresh one.
            tracker = self._current_latency_tracker
            self._current_latency_tracker = None
        wall_elapsed_s = (time.monotonic_ns() - wall_t0_ns) / 1e9
        report = self._inject_latency_telemetry(
            report=report,
            tracker=tracker,
            wall_elapsed_s=wall_elapsed_s,
            attempt_id=pick_attempt_id,
        )
        if getattr(report, "needs_person", False) is True:
            # The arm stands where a recovery stopped it: no pick of this service starts until a person decides.
            self._needs_person = str(report.telemetry.get("push_stopped") or report.failure_summary())
        if fault is not None:
            # What the raise left behind, and nothing more: no SLO sample, no shadow annotation and no
            # record. The corpus counts grasps, and a camera that stopped delivering is not a grasp
            # that missed.
            return report
        # Update rolling latency history and emit SLO_BREACH on
        # rising-edge breaches. Always a no-op unless the operator opted
        # in via ``performance.enabled`` and
        # ``performance.emit_breach_events``.
        self._latency.update_and_emit(
            config=self.effective_config,
            event_listener=self.watchdog_event_listener,
            tracker=tracker,
            attempt_id=str(report.telemetry.get("attempt_id", "")),
            effective_mode=report.mode,
        )
        # Opt-in shadow-router post-processor. Pure carrier; never
        # influences ``report`` when shadow is off.
        report = self._maybe_attach_shadow_router(report)
        self._maybe_log_record(report)
        return report

    def _look_and_pick(
        self, looks: "tuple[LookPose, ...]", *, mode: GraspMode | str | None, both_faces: bool = False,
        multi_view: bool = True,
    ) -> AutonomousGraspReport:
        """One attempt from its looks: a wrist camera's handed to the pick loop, a fixed camera's tried in turn here.

        Inside :meth:`pick`'s fault guard and its latency span, so a look is part of the attempt it serves: a camera
        that could not vouch on the way is the attempt's fault, and one record is logged for the attempt, not one per
        look.

        A camera on the wrist: :meth:`_look_around_and_pick`. A fixed camera, the service's own loop: from each look in
        turn until one finds something (:func:`_found_nothing`), or from where the arm stands with none. A stop is
        asked for before every look, the first included (the hand may have asked a person in between): one asked for
        while a look perceives ends the pick there, ``CANCELLED`` with the looks the arm was sent to, not the one it
        would have been sent to next. The pick loop asks too, but only once the arm has reached that next look.

        ``both_faces`` means one thing on every camera: a fixed camera's pick loop is handed it as a wrist camera's is
        (``orch.both_faces``, taken back in a ``finally``), and grips only a grasp its cameras showed both contact faces
        of; it cannot look around for the other, so the pick ends ``NO_VALID_GRASP`` naming it.
        """
        if self.perceives_from_the_wrist:
            return self._look_around_and_pick(looks, mode=mode, both_faces=both_faces, multi_view=multi_view)
        if not both_faces:
            return self._look_from_a_fixed_camera(looks, mode=mode)
        orchestrator = self.runtime.orchestrator
        orchestrator.both_faces = True
        try:
            report = self._look_from_a_fixed_camera(looks, mode=mode)
        finally:
            orchestrator.both_faces = False
        return replace(report, both_faces=True)

    def _look_from_a_fixed_camera(
        self, looks: "tuple[LookPose, ...]", *, mode: GraspMode | str | None,
    ) -> AutonomousGraspReport:
        """A fixed camera's attempt: from each of ``looks`` in turn until one finds something, or where it stands."""
        if not looks:
            return self._run_with_recovery(mode=mode)
        arm = self.runtime.orchestrator.arm
        tried: list[str] = []
        report: Optional[AutonomousGraspReport] = None
        for pose in looks:
            if self._should_cancel is not None and self._should_cancel():
                return replace(self._cancelled_report(mode=mode), looks=tuple(tried), telemetry={
                    "cancelled_before_start": not tried, "stage": "look", "cancelled_before_look": look_label(pose)})
            tried.append(look_label(pose))
            moved = move_to_look(arm, pose)
            if moved.outcome is MotionOutcome.CAMERA_WORLD_UNAVAILABLE and isinstance(
                    moved.result.exception, CameraWorldUnavailable):
                raise moved.result.exception
            if not moved.ok:
                return self._look_not_reached(moved, tuple(tried), mode=mode)
            report = self._run_with_recovery(mode=mode)
            if not _found_nothing(report):
                break
        assert report is not None  # narrowed: `looks` is not empty and every path above returned or picked
        return replace(report, looks=tuple(tried))

    def _look_around_and_pick(
        self, looks: "tuple[LookPose, ...]", *, mode: GraspMode | str | None, both_faces: bool,
        multi_view: bool = True,
    ) -> AutonomousGraspReport:
        """One attempt of a wrist camera: its looks handed to the pick loop, which looks from each, fused, and picks.

        The pick loop owns the looking (``BinPickingOrchestrator.look_around``): it moves to each look, perceives there,
        fuses the look with the ones before it and with the fixed cameras, and stops at the first look whose grasp is
        safe (plan A1; addendum 1). The service hands it this pick's looks and switch (``orch.looks``,
        ``orch.both_faces``) and takes them back in a ``finally``, whatever ended the pick, so the next pick inherits
        neither, and no frame of this pick stays held (``close_looks``). The report is then stamped with what the looks
        came to (:func:`_report_of_looking`). With no look the pose the arm stands at is the one look.

        A stop is asked for before the first look, as the fixed camera's loop asks it (the hand may have asked a person
        since the pick began); the pick loop asks before every later one.

        A pick handed looks has had its rescans: its looks, each fused, and the one view they may generate, the very
        last resort (addendum 3). The recovery loop's rescan re-runs the pick, which would drive every look and a new
        generated view again, so it rescans nothing for such a pick (:meth:`_run_with_recovery`), the same rule as the
        pick loop's own, which never comes back to a pick handed looks for a second attempt. A pick handed no look
        rescans where it stands, as it always did.
        """
        if looks and self._should_cancel is not None and self._should_cancel():
            return replace(self._cancelled_report(mode=mode), telemetry={
                "cancelled_before_start": True, "stage": "look", "cancelled_before_look": look_label(looks[0])})
        orchestrator = self.runtime.orchestrator
        # Only this pick's looks are said on its report: :meth:`pick` let the last pick's go before it began, so a
        # decision that ends before any look leaves none.
        orchestrator.looks = tuple(looks)
        orchestrator.both_faces = bool(both_faces)
        # Multi-View aus (Q11): no view is generated after the one look this pick was handed.
        orchestrator.generated_view = bool(multi_view)
        try:
            report = self._run_with_recovery(mode=mode, rescans=not looks)
        finally:
            orchestrator.close_looks()
            orchestrator.looks = ()
            orchestrator.both_faces = False
            orchestrator.generated_view = True
        return _report_of_looking(report, orchestrator.looked_around, handed=bool(looks), both_faces=both_faces)

    def _looks_stopped_report(
        self, looked: "LookedAround", *, effective_mode: GraspMode,
    ) -> AutonomousGraspReport:
        """The report of a decided pick whose looks ended on a motion: nothing perceived there, nothing else commanded.

        ``CANCELLED`` with the pick loop's ``CONTROLLER_NOT_OPERATIONAL`` underneath where the controller cannot move,
        as the loop said it when the looking stopped (not asked again: a second read can disagree with the first);
        ``EXECUTION_FAILED`` otherwise. The look it did not reach, and why, are stamped by :func:`_report_of_looking`.
        """
        stopped = ({"low_level_outcome": str(PickOutcome.CONTROLLER_NOT_OPERATIONAL), "controller": looked.controller}
                   if looked.controller else {})
        return AutonomousGraspReport(
            outcome=AutonomousGraspOutcome.CANCELLED if stopped else AutonomousGraspOutcome.EXECUTION_FAILED,
            mode=effective_mode,
            profile=_profile_for(effective_mode),
            effective_config=self.effective_config,
            telemetry={"stage": "look", **stopped},
        )

    def _look_not_reached(
        self, moved: MotionReport, tried: tuple[str, ...], *, mode: GraspMode | str | None,
    ) -> AutonomousGraspReport:
        """The report of an attempt whose look the arm did not reach: nothing perceived, nothing else commanded.

        ``CANCELLED`` with the pick loop's own words where the controller cannot move, so a campaign stops on it as it
        stops on a pick that ended on a stopped controller; ``EXECUTION_FAILED`` otherwise.
        """
        effective_mode = resolve_grasp_mode(mode) if mode is not None else self.default_mode
        stopped = _controller_stop_telemetry(self.runtime.orchestrator)
        return AutonomousGraspReport(
            outcome=AutonomousGraspOutcome.CANCELLED if stopped else AutonomousGraspOutcome.EXECUTION_FAILED,
            mode=effective_mode,
            profile=_profile_for(effective_mode),
            effective_config=self.effective_config,
            telemetry={"stage": "look", "look_refused": f"{moved.status.value}: {moved.message}", **stopped},
            looks=tried,
        )

    @property
    def looked_around(self) -> "LookedAround | None":
        """What the looks of the last wrist pick came to, as the pick loop keeps it; ``None`` for a fixed camera, and
        after a pick that ended before it looked (each pick lets the last one's go as it begins).

        The looks visited and fused, the judgement the pick went on with (its target cloud, fused over the looks) and
        every look's frame whole (``views``: RGB, depth, the tool pose stamped at its shutter, the intrinsics), which
        is what ``PickRun(record_views=True)`` keeps. Readable after the pick: taking its looks back holds nothing.
        """
        return getattr(getattr(self.runtime, "orchestrator", None), "looked_around", None)

    @property
    def ranked_looked(self) -> "LookedAround | None":
        """The last of the last pick's look sequences whose judgement ranked grasps (the pick loop's ``ranked_looked``):
        what a failed pick's debug pictures draw its grasps from once its last look ranked none. ``None`` as for
        :attr:`looked_around`, and where no look of the pick ranked a grasp."""
        return getattr(getattr(self.runtime, "orchestrator", None), "ranked_looked", None)

    @property
    def perceives_from_the_wrist(self) -> bool:
        """Whether this cell's picks perceive through a camera on the wrist: its frames are placed by the tool pose.

        Read off the resolver the build wired, as the build itself decides which frames to stamp: an eye-in-hand
        resolver composes the tool pose into every grasp, a fixed camera's never asks. A wrist camera sees what the
        arm points it at, so ``PickRun`` hands its picks a look (home, when the program names none).
        """
        from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver  # noqa: PLC0415

        orchestrator = getattr(self.runtime, "orchestrator", None)
        return isinstance(getattr(orchestrator, "frame_resolver", None), EyeInHandFrameResolver)

    def put_back(self, report: AutonomousGraspReport) -> "HandlingReport":
        """Put the part a pick lifted back where it was grasped: a planned move to the standoff, a line in, release, out.

        The pose is the one the tool closed at (:attr:`AutonomousGraspReport.grasp_pose`), the standoff the pick's own
        (the policy's ``standoff_mm``), and the verb :meth:`Robot.place <src.robot.execution.robot.Robot.place>` on
        this service's arm and hand, so every refusal of a place stands and the release is the hand's own (on a toggle
        hand, one change). What lets a campaign run many times on one part. A report that did not succeed, or carries no
        grasp pose, is refused with nothing commanded.
        """
        from src.robot.execution.handling import HandlingOutcome, HandlingReport, HandlingVerb  # noqa: PLC0415
        from src.robot.execution.robot import Robot  # noqa: PLC0415

        pose = report.grasp_pose
        if not report.succeeded or pose is None:
            why = "the pick did not succeed" if not report.succeeded else "the pick carries no grasp pose"
            return HandlingReport(verb=HandlingVerb.PLACE, outcome=HandlingOutcome.REFUSED,
                                  message=f"nothing to put back: {why}")
        orchestrator = self.runtime.orchestrator
        standoff_mm = float(getattr(orchestrator.policy, "standoff_mm", 80.0))
        # No lock of its own: the connect that brought this service up holds the cell's.
        robot = Robot.from_parts(arm=orchestrator.arm, gripper=orchestrator.gripper, lock_key=None)
        return robot.place(pose, standoff_mm=standoff_mm)

    def enable_record_logging(
        self, log_path: "str | Path | None", *, provenance: "Mapping[str, Any] | None" = None
    ) -> None:
        """Opt into appending one ``GraspAttemptRecord`` JSONL line per :meth:`pick` to ``log_path``:
        the production data source the soak / KPI / RL layers read. Pass ``None`` to disable. Off by
        default, so an un-configured service logs nothing and is byte-identical.

        ``provenance`` is stamped into every record's ``extra`` bag (e.g. ``robot_vendor`` / ``robot_model``)
        so a corpus from two robots is not silently mixed, which matters the moment a UR3e cell and the
        UR5e cell both log. ``extra`` is additive: :func:`audit_record` checks only required-key
        presence and :func:`audit_extra_record` type-checks only the keys it already knows, so new
        keys leave the frozen contract intact."""

        self._record_log_path = Path(log_path) if log_path is not None else None
        self._record_provenance = dict(provenance) if provenance else None

    @property
    def last_debug_image_png(self) -> "bytes | None":
        """The most recent grasp-point debug PNG the wrapped calculator rendered, or ``None``.

        The :class:`GraspCalculator` renders this overlay (segmentation mask + projected gripper
        wireframes + contact arrows + per-candidate rank/score) on every ``compute`` that was handed an
        ``rgb_image``, so after a :meth:`pick` it holds the just-grasped scene. This read-only accessor
        is the seam the sim runners dump to disk (the grasp-point viewer "in action"), and it is also
        what ``/v1/overlay`` and the console's viewfinder serve."""

        orch = getattr(self.runtime, "orchestrator", None)
        calculator = getattr(orch, "calculator", None)
        return getattr(calculator, "last_debug_image_png", None)

    @property
    def debug_image_rendering_enabled(self) -> bool:
        """Whether the overlay above is currently being rendered.

        The read half of :meth:`enable_debug_image_rendering`. A blank viewfinder during a pick
        means either "rendering is off" or "nothing segmented yet", and only this flag separates
        the two.
        """
        orch = getattr(self.runtime, "orchestrator", None)
        calculator = getattr(orch, "calculator", None)
        return bool(getattr(calculator, "render_debug_images", False))

    def enable_debug_image_rendering(self, enabled: bool = True) -> None:
        """Opt into rendering the grasp-point overlay on every pick (read via
        :attr:`last_debug_image_png`). Flips the wrapped calculator's ``render_debug_images`` switch so
        the live pick path forwards the perception-frame rgb into the calculator. Off by default,
        which is byte-identical and costs no render time; best-effort (no-op if the runtime exposes
        no calculator)."""

        orch = getattr(self.runtime, "orchestrator", None)
        calculator = getattr(orch, "calculator", None)
        if calculator is not None:
            calculator.render_debug_images = enabled

    def _cancelled_report(self, *, mode: "GraspMode | str | None") -> AutonomousGraspReport:
        """The report for a pick that was refused before it began.

        A typed ``CANCELLED`` outcome rather than a failure: a stop is a decision, and a stop that
        counted as an execution failure would lower the measured pick rate on every press of the
        button. Carries no pick_report because no pick ran; an empty attempt trail in the record
        corpus is one replay could not tell from a real one.
        """
        effective_mode = resolve_grasp_mode(mode) if mode is not None else self.default_mode
        return AutonomousGraspReport(
            outcome=AutonomousGraspOutcome.CANCELLED,
            mode=effective_mode,
            profile=_profile_for(effective_mode),
            effective_config=self.effective_config,
            telemetry={"cancelled_before_start": True},
        )

    def controller_refusal(self) -> str:
        """Why the controller cannot start a pick, or ``""``: the question every pick asks first, for a caller that
        drives the arm between picks (a task: before its place, after a motion that failed). In ``Robot.pick``'s words;
        a service with no gripper is not asked. Reads the controller and commands nothing."""
        return self._controller_refusal()

    def _controller_refusal(self) -> str:
        """Why the controller cannot start a pick, or ``""``: asked before the hand, of a service with a gripper.

        The question ``Robot.pick`` asks (:func:`~src.robot.execution.handling._controller_refusal`), in its words: an
        arm that implements ``SupportsRobotStatus`` (the UR driver) is asked, ``not is_operational`` refuses (REDUCED
        safety mode too), and a read that raises refuses as well, because a controller that cannot be asked cannot be
        said to move. Every other arm passes. A service with no gripper is not asked, as the policy does not ask: it
        has no jaws to keep, and its first motion meets the controller as before.
        """
        from src.robot.execution.handling import _controller_refusal as controller_refusal  # noqa: PLC0415

        orchestrator = getattr(self.runtime, "orchestrator", None)
        if getattr(orchestrator, "gripper", None) is None:
            return ""
        return controller_refusal(getattr(orchestrator, "arm", None))

    def _controller_refused_report(self, why: str, *, mode: "GraspMode | str | None") -> AutonomousGraspReport:
        """The report for a pick a controller that cannot move ended before it began: nobody asked, nothing commanded.

        CANCELLED with the pick loop's ``CONTROLLER_NOT_OPERATIONAL`` underneath, in the two keys
        :func:`_controller_stop_telemetry` writes (``low_level_outcome`` and ``controller``), so
        :attr:`AutonomousGraspReport.controller_stopped` reads it as it reads a stop the pick met mid-motion, and
        ``PickRun`` and the console stop on it. ``why`` is the sentence the controller was refused with. Written here,
        not read again through the pick loop's diagnosis: a second read can disagree with the first (one that raises
        says nothing), and the report would then not stop a campaign. No ``pick_report``: no pick ran.
        """
        effective_mode = resolve_grasp_mode(mode) if mode is not None else self.default_mode
        return AutonomousGraspReport(
            outcome=AutonomousGraspOutcome.CANCELLED,
            mode=effective_mode,
            profile=_profile_for(effective_mode),
            effective_config=self.effective_config,
            telemetry={"stage": "before_start", "low_level_outcome": str(PickOutcome.CONTROLLER_NOT_OPERATIONAL),
                       "controller": why},
        )

    def _needs_person_report(self, *, mode: "GraspMode | str | None") -> AutonomousGraspReport:
        """The report for a pick refused because a recovery of an earlier pick stopped where the arm stands.

        Nothing is read off the cell, asked or commanded. ``UNSAFE_RECOVERY_REFUSED``, so it needs a person as the report
        it follows did (:attr:`AutonomousGraspReport.needs_person`), with that report's words
        (``stopped_where_the_arm_stands``), and ``PickRun`` and the console stop on it as they stop on that report. No
        ``pick_report`` and no record: no pick ran. :meth:`start_campaign` or :meth:`acknowledge_needs_person` lets the
        next pick start.
        """
        effective_mode = resolve_grasp_mode(mode) if mode is not None else self.default_mode
        return AutonomousGraspReport(
            outcome=AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED,
            mode=effective_mode,
            profile=_profile_for(effective_mode),
            effective_config=self.effective_config,
            telemetry={"stage": "before_start", "stopped_where_the_arm_stands": self._needs_person},
        )

    @property
    def stopped_where_the_arm_stands(self) -> str:
        """Why a pick of this service stopped where the arm stands, a person to decide, in its report's words; ``""``
        while nothing waits for a person. While it is set, :meth:`pick` starts nothing."""
        return self._needs_person

    def acknowledge_needs_person(self) -> None:
        """A person decided what happens after a recovery that stopped where the arm stands (a report's
        :attr:`~AutonomousGraspReport.needs_person`): the next pick may start. :meth:`start_campaign` does the same, and
        every ``PickRun`` and console run starts a campaign."""
        if self._needs_person:
            _LOG.info("a person decided after a recovery that stopped where the arm stands; picks may start again")
        self._needs_person = ""

    def _hand_refusal(self) -> str:
        """Why the hand will not start a pick, or ``""``: asked of a hand that toggles with no sensor, and no other."""
        toggle = toggle_without_sensor_of(getattr(getattr(self.runtime, "orchestrator", None), "gripper", None))
        return toggle.jaws_open_for_a_pick() if toggle is not None else ""

    def _hand_refused_report(
        self, why: str, *, mode: "GraspMode | str | None", stage: str = "before_start",
    ) -> AutonomousGraspReport:
        """The report for a pick a hand that toggles would not start: nothing moved and nothing was sent.

        EXECUTION_FAILED with the pick loop's ``gripper_fault`` underneath, as the policy's own refusal reports it
        (:attr:`AutonomousGraspReport.gripper_fault`), so a campaign stops on either. No ``pick_report``: no pick ran.
        ``stage`` says where: ``before_start``, or ``before_the_re_pick`` where the pick's recovery would have picked
        again (:meth:`_run_with_recovery`). There ``re_pick_refused`` says so too, ``gripper_fault``: the recovery
        trail's last step names the action whose pick this refused, and that pick never ran.
        """
        effective_mode = resolve_grasp_mode(mode) if mode is not None else self.default_mode
        telemetry = {"stage": stage, "low_level_outcome": str(PickOutcome.GRIPPER_FAULT), "gripper_fault": why}
        if stage == "before_the_re_pick":
            telemetry["re_pick_refused"] = PickOutcome.GRIPPER_FAULT.value
        return AutonomousGraspReport(
            outcome=AutonomousGraspOutcome.EXECUTION_FAILED,
            mode=effective_mode,
            profile=_profile_for(effective_mode),
            effective_config=self.effective_config,
            telemetry=telemetry,
        )

    def _fault_report(self, fault: Exception, *, mode: "GraspMode | str | None") -> AutonomousGraspReport:
        """The report for a pick a fault of the cell ended.

        EXECUTION_FAILED, the outcome the pick loop's own aborted attempt maps to, with the fault on the
        report and no ``pick_report``: nothing below this service returned, so there is no attempt trail
        to carry, and an invented one would put a trail into the corpus that replay could not tell from
        a real one.
        """
        effective_mode = resolve_grasp_mode(mode) if mode is not None else self.default_mode
        return AutonomousGraspReport(
            outcome=AutonomousGraspOutcome.EXECUTION_FAILED,
            mode=effective_mode,
            profile=_profile_for(effective_mode),
            effective_config=self.effective_config,
            fault=fault,
        )

    def attach_progress_listener(self, listener: "PickProgressListener | None") -> None:
        """Opt into live progress events from inside the pick loop. ``None`` detaches.

        Without a listener ``pick()`` is a blocking call that reports nothing until it returns, which
        is enough for a CLI printing a result line and not enough for an operator console watching an
        arm move.

        The listener is called inline, on the thread driving the robot. Exceptions are swallowed
        (a subscriber must not be able to abort a motion in flight) but a slow listener still blocks
        the loop, so a console hands the event to a queue and returns.

        Post-construction like :meth:`enable_record_logging` and :meth:`enable_debug_image_rendering`,
        rather than a config key: it describes who is watching, not how the cell behaves, and it would
        mean nothing in a YAML file a CLI user reads. Off by default and byte-identical.
        """
        orch = getattr(self.runtime, "orchestrator", None)
        if orch is not None:
            orch.on_progress = listener

    def set_cancel_check(self, should_cancel: "ShouldCancel | None") -> None:
        """Let a caller stop a run between attempts. ``None`` clears it.

        "Stop" here means do not begin the next attempt. It never interrupts a motion already in
        flight: this code has no safe way to do that, and the thing that does is the physical stop
        button. A pick with five attempts therefore ends after the one that is running, instead of
        after all five.

        Unset, the attempt loop runs unchanged. When it fires the run is no longer byte-identical
        and it terminates with the typed ``CANCELLED`` outcome rather than a failure, so an
        operator's decision never lowers the measured pick rate.

        Set in two places on purpose. The orchestrator's attempt loop is one of them; this service is
        the other, because :meth:`pick` is itself callable in a loop by a caller running a campaign,
        and a cancel that arrives between two picks must not be silently forgotten just because the
        orchestrator never got asked.
        """
        orch = getattr(self.runtime, "orchestrator", None)
        if orch is not None:
            orch.should_cancel = should_cancel
        self._should_cancel = should_cancel

    def set_target_label(self, label: "str | None") -> None:
        """Set a hard label-target for a prompt-driven dense pick: only the segmentation carrying
        this label is an executable target; the others stay as neighbour clutter (so the dense sampler +
        the re-rank/approach validators still see them). Pass ``None`` to clear. Unset by default and
        byte-identical. Best-effort (no-op if the runtime exposes no orchestrator)."""

        orch = getattr(self.runtime, "orchestrator", None)
        if orch is not None:
            orch.target_label = label

    def set_prompt(self, prompt: "str | PickPrompt") -> PickPrompt:
        """Say what the next picks look for, and return what they looked for until now.

        A text is what an operator typed or spoke (:meth:`PickPrompt.from_text`). Every camera whose
        source grounds a phrase grounds it from the next frame on and maps its detector's words onto it,
        and the label filter (:meth:`set_target_label`) makes only an object it grounded a target. Pass
        the returned :class:`PickPrompt` back to put all three back. Nothing reopens and nothing
        reloads: the detector is handed the phrase on every frame, which is what lets a campaign change
        it.

        A source that grounds no phrase (the rehearsal scene, a ground-truth sim source) keeps its
        frames and the label filter still applies, so such a cell reports the prompted label as not
        found rather than picking whatever it shows. Refused before anything changes: a text that names
        nothing, and a prompt without a phrase for a cell that grounds one.
        """
        wanted = PickPrompt.from_text(prompt) if isinstance(prompt, str) else prompt
        orchestrator = self.runtime.orchestrator
        sources = _grounding_sources(orchestrator)
        if sources and not wanted.phrase:
            raise ValueError(
                "this cell grounds a phrase and the prompt names none; a prompt taken from a cell whose "
                "perception grounds nothing cannot be put back on one that grounds a phrase"
            )
        first = sources[0] if sources else None
        previous = PickPrompt(
            phrase=str(getattr(first, "prompt", "") or ""),
            target_label=orchestrator.target_label,
            object_labels=tuple(getattr(first, "object_labels", ()) or ()),
        )
        for source in sources:
            source.set_prompt(wanted.phrase, object_labels=wanted.object_labels)
        orchestrator.target_label = wanted.target_label
        return previous

    def grounds_a_phrase(self) -> bool:
        """Whether a camera of this cell grounds a phrase (a detector reads one), so a prompt says what its picks look
        for; ``False`` for a cell whose perception grounds none, such as the rehearsal scene, which picks what it
        shows. A task refuses an empty object on a cell that grounds a phrase unless the operator said "anything"."""
        return bool(_grounding_sources(getattr(self.runtime, "orchestrator", None)))

    def set_closing_axis(self, axis: Any) -> "ClosingAxis | None":
        """Have the next picks grip only grasps that close along ``axis``, and return the axis they closed along until
        now; ``None`` lets them close along any, as the cell's motion named.

        The opt-in filter a task's Advanced drawer sets for that task alone (``GraspMotion(closing_axis=...)`` is the
        program's way): ``axis`` is a name ``Pose.tool_down`` takes (``"-y"``), an orientation, or a ``ClosingAxis``,
        read by ``closing_axis_of`` and set on the execution policy, which the pick loop reads before anything reads a
        candidate (it chooses, it never twists). Refused before anything changes: a value that names no axis
        (``ValueError``, ``TypeError``), and any axis beside a motion that turns every grasp about base Z before it
        closes (``align_closing_to_base_x``), which would close the jaws off the axis named. Pass the returned value
        back to put it back.
        """
        policy = getattr(self.runtime.orchestrator, "policy", None)
        wanted = self._closing_axis_wanted(axis)
        previous = getattr(policy, "closing_axis", None)
        policy.closing_axis = wanted  # type: ignore[union-attr]
        return previous

    def closing_axis_refusal(self, axis: Any) -> str:
        """Why :meth:`set_closing_axis` would refuse ``axis``, in its words, read with nothing changed; ``""`` where it
        takes it. What a task asks before it says or sets anything, so its refusal comes before its first event."""
        try:
            self._closing_axis_wanted(axis)
        except (TypeError, ValueError) as exc:
            return str(exc) or f"{axis!r} names no closing axis"
        return ""

    def _closing_axis_wanted(self, axis: Any) -> "ClosingAxis | None":
        """``axis`` read as :meth:`set_closing_axis` takes it, or its refusal raised (``ValueError``, ``TypeError``)."""
        from src.robot.grasping.geometry.closing_axis import closing_axis_of  # noqa: PLC0415
        from src.robot.grasping.motion.execution_policy import closing_axis_twisted  # noqa: PLC0415

        policy = getattr(self.runtime.orchestrator, "policy", None)
        wanted = None if axis is None else closing_axis_of(axis)
        twisted = closing_axis_twisted(SimpleNamespace(
            align_closing_to_base_x=bool(getattr(policy, "align_closing_to_base_x", False)), closing_axis=wanted))
        if twisted:
            raise ValueError(twisted)
        return wanted

    def detector_failures(self) -> int:
        """How many times the detectors of this cell's grounding cameras failed and answered nothing, so far: each
        backend counted once, whichever cameras share it.

        A backend swallows a detector's exception and perceives nothing (``TwoStageBackend``), so a pick that met one
        reads as "nothing there"; a caller that reads this before and after a pick tells the two apart (a task's
        ``detector_failed``). Read off each backend's ``failures`` count, where it keeps one; a backend that keeps none
        counts nothing.
        """
        total = 0
        counted: list[Any] = []
        for source in _grounding_sources(getattr(self.runtime, "orchestrator", None)):
            backend = getattr(source, "backend", None)
            if backend is None or any(backend is seen for seen in counted):
                continue
            counted.append(backend)
            failures = getattr(backend, "failures", 0)
            if isinstance(failures, int) and not isinstance(failures, bool) and failures > 0:
                total += failures
        return total

    def push_distance(self, push_mm: "Maybe[float]" = UNSET) -> float:
        """How far a push of a campaign asking ``push_mm`` moves its part, in mm: ``ValueError`` with the sentence
        where the request is refused.

        Unset, the config's ``recovery.fixture.push_distance_mm`` (30 mm unless the cell says otherwise). A request is
        taken as asked up to the cell's ceiling, ``recovery.fixture.max_nudge_mm`` (50 mm unless the cell says less),
        and refused above it or above the owner's hard cap of 50 mm, never shortened; one under 10 mm is refused too,
        since it cannot open room for a finger (``push_planner.resolve_push_distance``). A cell that declares no fixture
        takes the defaults: 30 mm unset, and the hard cap of 50 mm as its ceiling.
        """
        from src.robot.grasping.recovery.push_planner import (  # noqa: PLC0415
            DEFAULT_PUSH_DISTANCE_MM,
            PushRefusal,
            resolve_push_distance,
        )

        recovery = getattr(self.effective_config, "recovery_orchestrator", None)
        default = float(getattr(recovery, "push_distance_mm", DEFAULT_PUSH_DISTANCE_MM))
        ceiling = self.push_ceiling()
        if chosen(push_mm):
            try:
                requested: Optional[float] = float(push_mm)
            except (TypeError, ValueError):
                raise ValueError(f"a push distance is a number of mm, not {push_mm!r}") from None
        else:
            requested = None
        resolved = resolve_push_distance(requested, default_mm=default, ceiling_mm=ceiling)
        if isinstance(resolved, PushRefusal):
            raise ValueError(resolved.sentence)
        return float(resolved)

    def push_ceiling(self) -> float:
        """The longest push the cell allows, in mm: ``recovery.fixture.max_nudge_mm``, or the owner's hard cap of 50 mm
        where no fixture is declared."""
        from src.robot.grasping.recovery.push_planner import PUSH_DISTANCE_CAP_MM  # noqa: PLC0415

        recovery = getattr(self.effective_config, "recovery_orchestrator", None)
        fixture = getattr(recovery, "fixture", None)
        return min(float(fixture[2]), PUSH_DISTANCE_CAP_MM) if fixture is not None else PUSH_DISTANCE_CAP_MM

    def start_campaign(self, *, push_mm: "Maybe[float]" = UNSET,
                       critical_parts: "Maybe[bool | None]" = UNSET,
                       recovery_actions: "Maybe[tuple[str, ...] | None]" = UNSET,
                       blocker_is_the_pick: "Maybe[bool]" = UNSET) -> "PushCampaign":
        """Start a campaign of picks: fresh push budgets (1 per part, 2 per pick, 5 per campaign), no part skipped, and
        the push distance ``push_mm`` settles to (:meth:`push_distance`, whose ``ValueError`` it raises before
        anything changes). Where nobody asked for a distance, a push that opens too little room may go as far as the
        cell allows (:meth:`push_ceiling`); one asked for is taken as asked. ``critical_parts`` is the run's word on the
        owner's switch (``recovery.critical_parts``); unset or ``None``, the cell's stands. ``blocker_is_the_pick`` says a
        blocker a pick takes away is the part it picks, which its caller sets down where its parts go (a task that takes
        every part into one place, the owner, 2026-10-06); unset, it is set aside. ``PickRun`` and a console run
        each start one; a pick no caller started one for starts it. Starting one is a person's decision too: after a
        recovery that stopped where the arm stands (:attr:`stopped_where_the_arm_stands`), picks may start again."""
        from src.robot.grasping.recovery.push_gate import PushCampaign  # noqa: PLC0415

        critical = critical_parts if chosen(critical_parts) and critical_parts is not None else None
        actions = recovery_actions if chosen(recovery_actions) and recovery_actions is not None else None
        if actions is not None:
            for name in actions:
                _recovery_action(name)  # an unknown action is refused before anything changes
        self._campaign = PushCampaign(distance_mm=self.push_distance(push_mm),
                                      longest_mm=None if chosen(push_mm) else self.push_ceiling(),
                                      critical_parts=None if critical is None else bool(critical),
                                      recovery_actions=None if actions is None else tuple(actions),
                                      blocker_is_the_pick=bool(blocker_is_the_pick) if chosen(blocker_is_the_pick)
                                      else False)
        self.acknowledge_needs_person()
        return self._campaign

    @property
    def campaign(self) -> "PushCampaign":
        """The campaign the picks belong to: the one :meth:`start_campaign` started, else one started now with the
        config's push distance. Where the config's own distance is refused (a hand-built snapshot, since the schema
        refuses one at load), the campaign still counts and skips, and pushes nothing."""
        if self._campaign is None:
            from src.robot.grasping.recovery.push_gate import PushCampaign  # noqa: PLC0415

            try:
                self._campaign = PushCampaign(distance_mm=self.push_distance(), longest_mm=self.push_ceiling())
            except ValueError as refused:
                self._say_once("distance", "no part is pushed: %s", refused)
                self._campaign = PushCampaign(distance_mm=None)
        return self._campaign

    def _say_once(self, key: str, message: str, *args: Any) -> None:
        """A WARNING about why picks cannot push, said once per service for each ``key``."""
        if key not in self._push_said:
            self._push_said.add(key)
            _LOG.warning(message, *args)

    def _extra_with_route(self) -> Optional[dict[str, Any]]:
        """Record provenance, plus which perception route grounded this pick when one did.

        Stamped into ``extra`` rather than onto the record itself: :class:`GraspAttemptRecord` is a
        frozen contract with consumers in KPI, taxonomy and RL, and ``extra`` is the additive channel
        those consumers already tolerate. A routed cell gains two keys and an un-routed cell's
        records stay byte-identical, which is what keeps the soak baseline valid.

        The route is the first field to group by when a pick rate moves: whether one route did
        better on the hard prompts is unanswerable from a log that never recorded which model
        looked at the frame.
        """
        base = self._record_provenance
        try:
            perception = getattr(getattr(self, "runtime", None), "orchestrator", None)
            route = getattr(getattr(perception, "perception", None), "last_route", None)
        except Exception:  # noqa: BLE001 (provenance must never break logging, let alone a pick)
            route = None
        fusion = self._fusion_geometry_extras()
        if not route:
            # None when there is genuinely nothing to stamp, so a cell that configures no provenance
            # and runs no fusion logs an unchanged record.
            if not base and not fusion:
                return None
            return {**(base or {}), **fusion}
        return {
            **(base or {}), **fusion,
            "perception_route": route[0], "perception_route_reason": route[1],
        }

    def _fusion_geometry_extras(self) -> dict[str, Any]:
        """Produce the three ``multi_view_fusion`` catalog fields from geometry fusion.

        ``fused_view_count``, ``fusion_evidence_quality`` and ``multi_view_occlusion_reduced`` are
        declared in ``telemetry_catalog.py`` and read in ``shadow.py``. Geometry fusion computes
        exactly that information every pick and stamps it on
        ``orchestrator._fusion_geometry_telemetry``; this method joins the two vocabularies.

        The catalog declares types, not meanings, so the meanings are stated here:

        * ``fused_view_count``: how many views contributed a usable view this pick: one per camera on a
          fixed cell; on a wrist camera each look of the pick that took part counts as a view of its
          own, beside the fixed cameras fused with it.
        * ``fusion_evidence_quality``: the mean association score over the pairs that won their
          assignment, i.e. how confidently the views agreed an object was the same object.
        * ``multi_view_occlusion_reduced``: whether any object gained surface from another view,
          which is the literal question "did multi-view see what one view could not".

        All three are scene-level. They are diagnosis and telemetry, and they cannot help the
        pairwise ranker, which differences features within one scene's candidate list: a value
        identical for every candidate differences to exactly zero.

        Empty (and therefore byte-identical) whenever geometry fusion did not run.
        """
        try:
            orchestrator = getattr(getattr(self, "runtime", None), "orchestrator", None)
            telemetry = getattr(orchestrator, "_fusion_geometry_telemetry", None)
            if not telemetry or not isinstance(telemetry, Mapping):
                return {}
            views = telemetry.get("fused_views_used", 0)
            quality = telemetry.get("fused_evidence_quality", 0.0)
            objects = telemetry.get("fused_objects", 0)
            # Inside the guard, not after it. The conversions are where a non-numeric value bites:
            # an orchestrator whose telemetry mapping returns a non-number makes ``int()`` raise.
            # The docstring promises this never breaks logging, and that promise has to cover the
            # arithmetic, not just the attribute lookups.
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       for v in (views, quality, objects)):
                return {}
            return {
                "fused_view_count": int(views),
                "fusion_evidence_quality": float(quality),
                "multi_view_occlusion_reduced": bool(int(objects) > 0),
            }
        except Exception:  # noqa: BLE001 (telemetry must never break a pick, let alone logging)
            return {}

    def _maybe_log_record(self, report: "AutonomousGraspReport") -> None:
        path = self._record_log_path
        if path is None:
            return
        try:
            from .record_logging import log_record

            attempt_id = str(report.telemetry.get("attempt_id", "")) or "pick"
            log_record(
                report, attempt_id=attempt_id, log_path=path,
                extra=self._extra_with_route(),
            )
        except Exception:  # noqa: BLE001 (a logging failure must never break the pick)
            pass

    @staticmethod
    def _inject_latency_telemetry(
        *,
        report: AutonomousGraspReport,
        tracker: Optional[LatencyTracker],
        wall_elapsed_s: float,
        attempt_id: str,
    ) -> AutonomousGraspReport:
        """Fold per-stage latency + wall-time (+ the shared attempt_id) into a report's telemetry.

        Stages that never opened a span (e.g. fusion on a pick that fused
        no other camera's view) are absent from the snapshot, so the SLO
        gate skips nulls instead of treating them as passing zeros.
        ``attempt_wall_time_s`` is always emitted because the pick()
        wall-clock is meaningful for every attempt.

        ``attempt_id``: keep the decision path's ``auto-<uuid>`` when it already stamped one,
        else stamp the pick-scoped ``attempt_id`` so latency / shadow / record-logging all share a
        single, unique id.
        """

        merged = dict(report.telemetry)
        if tracker is not None:
            merged.update(tracker.snapshot())
        merged["attempt_wall_time_s"] = float(wall_elapsed_s)
        if not merged.get("attempt_id"):
            merged["attempt_id"] = attempt_id
        return replace(report, telemetry=merged)

    def _pick_inner(
        self,
        *,
        mode: GraspMode | str | None = None,
    ) -> AutonomousGraspReport:
        """Body of :meth:`pick`; latency telemetry is injected by the wrapper."""

        effective_mode = resolve_grasp_mode(mode) if mode is not None else self.default_mode
        profile = _profile_for(effective_mode)
        default_profile = _profile_for(self.default_mode)

        # Per-call override is only safe if the wrapped runtime was
        # built with a compatible sampling mode. Otherwise the report
        # would claim behavior the underlying orchestrator cannot
        # deliver.
        if profile.sampling_mode is not default_profile.sampling_mode:
            return AutonomousGraspReport(
                outcome=AutonomousGraspOutcome.MODE_NOT_AVAILABLE,
                mode=effective_mode,
                profile=profile,
                pick_report=None,
                telemetry={
                    "reason": "per_call_mode_requires_different_sampling_mode",
                    "configured_sampling_mode": str(default_profile.sampling_mode),
                    "requested_sampling_mode": str(profile.sampling_mode),
                    "hint": (
                        "rebuild the service with the desired mode via "
                        "from_components/from_robot_config, or stay on "
                        "modes that share the configured sampling mode."
                    ),
                },
                effective_config=self.effective_config,
            )

        # Pre-execution decision layer. ``EASY`` is locked out of scope
        # so the ``EASY`` happy path stays byte-equivalent to the trust
        # path. When the engine is unwired or the policy is disabled,
        # the legacy pick path runs verbatim.
        decision_in_scope = (
            self.decision_engine is not None
            and self.decision_policy is not None
            and self.decision_policy.enabled
            and effective_mode is not GraspMode.EASY
        )
        if decision_in_scope:
            return self._pick_with_decision(
                effective_mode=effective_mode, profile=profile
            )

        return self._execute_pick(
            effective_mode=effective_mode, profile=profile, decision=None
        )

    # Recovery-orchestrator runtime integration (opt-in)

    def _build_recovery_orchestrator_policy(self) -> SceneRecoveryPolicy:
        """Build the :class:`SceneRecoveryPolicy` from the
        ``recovery_orchestrator_*`` operator config.

        Returns a disabled policy (the byte-identical single-pick fast path)
        unless ``grasping.recovery.enabled`` is true and the resolved mode is
        listed in ``grasping.recovery.apply_modes``, whose default excludes
        ``easy``.

        A snapshot built by hand (a simulator runner's ``dataclasses.replace``) skips the schema, so
        a recovery action removed on purpose (``next_viewpoint``, merged into ``rescan`` on
        2026-09-29) raises :class:`ValueError` here with the schema's own sentence, not as an unknown
        enum value.
        """

        cfg = self.effective_config
        asked = getattr(getattr(self, "_campaign", None), "recovery_actions", None)
        if cfg is None or (asked is None and not cfg.recovery_orchestrator.enabled):
            return SceneRecoveryPolicy(enabled=False)
        if asked is not None and not asked:
            return SceneRecoveryPolicy(enabled=False)
        # The run's own ticks (the console, 2026-10-05) override the cell's list for this campaign alone.
        allowed = tuple(
            _recovery_action(a) for a in (asked if asked is not None else cfg.recovery_orchestrator.allowed_actions)
        )
        budget = {
            _recovery_action(name): int(count)
            for name, count in cfg.recovery_orchestrator.per_action_budget
        }
        # The envelope, rebuilt from what the operator declared. `SceneRecoveryPolicy` refuses to
        # construct `container_agitate` without one, so a config that declares it and a call site that
        # passes no envelope raise on every pick(). The push needs none; a declared one narrows its box.
        envelope = None
        declared = cfg.recovery_orchestrator.fixture
        if declared is not None:
            from src.robot.grasping.recovery.policy import FixtureEnvelope

            centre, half_extents, max_nudge = declared
            envelope = FixtureEnvelope(
                center_mm=centre,
                half_extents_mm=half_extents,
                max_nudge_mm=max_nudge,
            )

        return SceneRecoveryPolicy(
            enabled=True,
            allowed_actions=allowed,
            max_recovery_actions=int(cfg.recovery_orchestrator.max_actions),
            per_action_budget=budget,
            apply_modes=tuple(cfg.recovery_orchestrator.apply_modes),
            fixture=envelope,
        )

    def _run_with_recovery(
        self, *, mode: GraspMode | str | None, rescans: bool = True,
    ) -> AutonomousGraspReport:
        """Run one pick attempt, optionally wrapped in the recovery loop.

        When ``recovery_orchestrator.enabled`` is :data:`False` (the default)
        this is byte-identical to a single :meth:`_pick_inner` call. When
        enabled (and the effective mode is in the configured ``apply_modes``),
        the attempt is driven through :func:`run_recovery_loop`; the
        ``uncertainty_recovery_aggressive_threshold`` knob biases the recovery
        action ordering toward perception whenever the most recent attempt's
        fused uncertainty exceeds it.

        ``rescans`` false takes ``rescan`` out of the actions the loop may plan: a wrist pick handed looks, whose looks
        and one generated view were its rescans (:meth:`_look_around_and_pick`). The loop still runs on what is left.

        The owner's recovery (2026-09-29), wired here:

        * The loop plans ``rescan`` and ``next_target`` directly and every other action through a strategy, of which
          there is none for the push: the push runs inside the pick attempt (the pick loop's ``push_gate``), and a
          ``nudge_target`` this loop plans is refused before anything moves and falls through.
        * ``next_target`` is a rescan that skips the part that failed: the pick loop names it (``failed_part``), it is
          added to the campaign's exclusion zones (same label only, this pick and the next two), and the pick runs
          again, a wrist pick handed looks with its looks again, for another part of the label.
        * Before ``next_target`` or a rescan drives the looks again (a wrist pick handed looks), a hand that toggles
          with no sensor is read (``why_toggle_count_unknown``: nothing asked, nothing sent). A count nobody can vouch
          for ends the pick there as a gripper fault (``stage`` ``before_the_re_pick``, ``re_pick_refused``
          ``gripper_fault``: the trail's last step never picked), before any look is driven again, and ``PickRun`` and
          the console stop on it (the owner, 2026-09-30). A re-pick that drives no look, a fixed camera's or that of a
          wrist pick handed no look, reads nothing more: its own check before the arm moves asks where the jaws stand,
          as at every pick start.
        * The pick loop is handed the campaign's zones and, where the policy lets this pick push
          (:func:`~src.robot.grasping.recovery.policy.push_permitted`: dense_clutter only), a push gate
          (:meth:`_push_gate`), both taken back after the pick.
        * A push that stopped where the arm stands ends the loop there (``needs_person``), ``unsafe_recovery_refused``
          unless the controller stopped. A loop that ran at least one action and ended for want of another is
          ``recovery_exhausted``, and one whose recovery motion failed ``unsafe_recovery_refused``
          (``terminal_outcome_for_trail``).
        """

        policy = self._build_recovery_orchestrator_policy()
        campaign = self.campaign
        loop = self.runtime.orchestrator
        if not policy.enabled:
            with _handed(loop, zones=campaign.zones, gate=None, blocker_is_the_pick=campaign.blocker_is_the_pick):
                return self._pick_inner(mode=mode)

        cfg = self.effective_config
        assert cfg is not None  # guaranteed by policy.enabled above
        effective_mode = (
            resolve_grasp_mode(mode) if mode is not None else self.default_mode
        )
        profile = _profile_for(effective_mode)
        if profile.mode.value not in policy.apply_modes:
            with _handed(loop, zones=campaign.zones, gate=None, blocker_is_the_pick=campaign.blocker_is_the_pick):
                return self._pick_inner(mode=mode)
        if not rescans and SceneRecoveryAction.RESCAN in policy.allowed_actions:
            _LOG.info("recovery rescans nothing for a wrist pick handed looks: its looks, and the one view they may "
                      "generate, were its rescans, and driving them again is not a recovery")
            policy = replace(policy, allowed_actions=tuple(
                action for action in policy.allowed_actions if action is not SceneRecoveryAction.RESCAN))

        threshold = float(cfg.uncertainty.recovery_aggressive_threshold)
        orchestrator = RecoveryOrchestrator(
            dispatcher=RecoveryDispatcher(), strategies={},
            direct_actions=frozenset({SceneRecoveryAction.RESCAN, SceneRecoveryAction.NEXT_TARGET}),
        )
        arm = getattr(loop, "arm", None)
        gate = self._push_gate(profile, policy, campaign)
        pushes_by_pick: list[tuple["PickPush", ...]] = []
        # What picks again: next_target where it skipped the failed part (`_skip_failed_part`), a rescan otherwise.
        again = ["a rescan"]

        def _pick_once() -> _RecoveryReportAdapter:
            if pushes_by_pick and self.perceives_from_the_wrist and bool(getattr(loop, "looks", None)):
                # next_target or a rescan drives the looks again (the owner, 2026-09-30): a toggle's count nobody can
                # vouch for (DO0 switched at the pendant since the pick began, a count lost or unreadable) ends the pick
                # here, as a gripper fault, before any look is driven again. Read only: nobody is asked, nothing is
                # sent. A re-pick that drives no look (a fixed camera's, a wrist pick's handed none) reads nothing
                # more: its own check before the arm moves asks where the jaws stand, as at every pick start.
                unknown = why_toggle_count_unknown(getattr(loop, "gripper", None))
                if unknown:
                    said = (f"the pick ends before {again[0]} picks again, with nothing commanded: nobody can say where "
                            f"the jaws stand ({unknown}), and nobody is asked")
                    _LOG.error("%s", said)
                    pushes_by_pick.append(())
                    return _RecoveryReportAdapter(self._hand_refused_report(said, mode=mode, stage="before_the_re_pick"))
            again[0] = "a rescan"
            report = self._pick_inner(mode=mode)
            pushed = _pushes_of(loop)
            pushes_by_pick.append(pushed)
            return _RecoveryReportAdapter(_with_pushes(report, pushed))

        def _aggressive(adapter: _RecoveryReportAdapter) -> bool:
            # Single source of truth: the canonical gate requires fused_available and
            # fused >= threshold, applied to the report's own UncertaintySnapshot.
            snapshot = getattr(adapter.report, "uncertainty", None)
            return should_bias_recovery_for_uncertainty(
                snapshot, recovery_aggressive_threshold=threshold,
            )

        def _skip_failed_part(adapter: _RecoveryReportAdapter) -> bool:
            # The part the pick that just failed went for, as the pick loop names it: skipped by the next picks.
            from src.robot.grasping.recovery.push_gate import FailedPart  # noqa: PLC0415

            part = getattr(loop, "failed_part", None)
            if not isinstance(part, FailedPart):
                return False
            zone = campaign.zones.exclude(label=part.label, centre_mm=part.centre_mm,
                                          footprint_diagonal_mm=part.footprint_diagonal_mm)
            _LOG.info("next_target skips the %r at (%.0f, %.0f) mm, within %.0f mm, for this pick and the next %d",
                      part.label, zone.centre_xy_mm[0], zone.centre_xy_mm[1], zone.radius_mm,
                      zone.last_pick - zone.added_pick)
            again[0] = "next_target"
            return True

        with _handed(loop, zones=campaign.zones, gate=gate, blocker_is_the_pick=campaign.blocker_is_the_pick):
            final, trail = run_recovery_loop(
                pick=_pick_once,
                profile=profile,
                policy=policy,
                orchestrator=orchestrator,
                frame_acquirer=lambda: None,  # the next _pick_inner re-perceives the scene
                arm=arm,
                aggressive_recovery_bias=_aggressive,
                skip_failed_part=_skip_failed_part,
            )
        report = self._attach_recovery_trail(final.report, trail, pushes=tuple(pushes_by_pick))
        return _with_the_recovery_outcome(report, trail)

    def _push_gate(
        self, profile: GraspBehaviorProfile, policy: SceneRecoveryPolicy, campaign: "PushCampaign",
    ) -> "PushGate | None":
        """What lets this pick push its failed part, or ``None``: the policy must permit it
        (:func:`~src.robot.grasping.recovery.policy.push_permitted`: enabled, the mode's profile lists ``nudge_target``,
        which only dense_clutter's does, ``recovery.allowed_actions`` lists it, a budget above zero), the cell's push
        inputs must be known (:attr:`push_cell`), the campaign must have a push distance, and a declared fixture box
        must leave room. No fixture is needed: without one the automatic push box alone bounds the push. Why a pick
        that is permitted cannot push is said once, as a WARNING."""
        from src.robot.grasping.recovery.policy import push_permitted  # noqa: PLC0415
        from src.robot.grasping.recovery.push_gate import PushCell, PushGate  # noqa: PLC0415

        if not push_permitted(profile, policy):
            return None
        cell = self.push_cell
        if not isinstance(cell, PushCell):
            why = getattr(cell, "sentence", "") or "the cell's hand, workspace and clearance were not read (push_cell)"
            self._say_once("cell", "nudge_target is allowed and no part is pushed: %s", why)
            return None
        if campaign.distance_mm is None:
            return None
        operator_box = _operator_box(policy.fixture)
        if isinstance(operator_box, str):
            self._say_once("fixture", "nudge_target is allowed and no part is pushed: %s", operator_box)
            return None
        cfg = self.effective_config
        critical = campaign.critical_parts
        if critical is None:
            critical = bool(getattr(cfg.recovery_orchestrator, "critical_parts", False)) if cfg is not None else False
        return PushGate(
            budgets=campaign.budgets, distance_mm=campaign.distance_mm, cell=cell, operator_box=operator_box,
            on_approach_blocked=bool(getattr(cfg, "approach_validation_enabled", False)),
            blocker_grasp_tries=int(cfg.recovery_orchestrator.blocker_grasp_tries) if cfg is not None else 3,
            critical_parts=critical, longest_mm=campaign.longest_mm,
        )

    @staticmethod
    def _attach_recovery_trail(
        report: AutonomousGraspReport, trail: RecoveryTrail, *, pushes: "tuple[tuple[PickPush, ...], ...]" = (),
    ) -> AutonomousGraspReport:
        """Fold the recovery-trail summary into a report's telemetry for operator
        visibility. Only called on the opt-in recovery path, so the default path
        stays byte-identical. Values are simple types (str + list[str]) so the
        telemetry audit (which checks required-key presence, not extras) is safe.

        ``pushes`` are what the pushes of each pick of the loop came to, pick by pick: every push that moved anything
        is a ``nudge_target`` step of the trail the record carries (``recovery_actions``), in the order it ran, and
        every push the picks considered is on the telemetry (``pushes``).
        """

        merged = dict(report.telemetry)
        merged["recovery_trail_terminal_reason"] = trail.terminal_reason
        merged["recovery_trail_actions"] = [
            entry.plan_action.value for entry in trail.entries
        ]
        considered = [push.to_dict() for pick in pushes for push in pick]
        if considered:
            merged["pushes"] = considered
        # Carry the per-step recovery trail on the typed report field so the serializer writes it into
        # GraspAttemptRecord.recovery_actions (the per-step action+reward source for offline recovery training).
        return replace(
            report,
            telemetry=merged,
            recovery_actions=_recovery_rows(trail, pushes),
        )

    # Fused-uncertainty helpers

    def _build_uncertainty_calibration(self) -> UncertaintyCalibration:
        """Construct the runtime calibration from operator config.

        Combines the per-channel weight vector
        (``effective_config.uncertainty.weights``) with an optional artifact
        path, which is honoured only as the identity-monotone shape. Channels
        for which the config omits a weight default to ``0.0``, so downstream
        fusion ignores them entirely.
        """

        cfg = self.effective_config
        weights_dict: dict[str, float] = (
            dict(cfg.uncertainty.weights) if cfg is not None else {}
        )

        def _w(name: str) -> float:
            value = weights_dict.get(name, 0.0)
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        weights = UncertaintyWeights(
            depth_confidence=_w("depth_confidence"),
            mask_confidence=_w("mask_confidence"),
            occlusion_corridor_risk=_w("occlusion_corridor_risk"),
            feasibility_margin=_w("feasibility_margin"),
            verification_residual=_w("verification_residual"),
            topology_risk=_w("topology_risk"),
            semantic_confidence=_w("semantic_confidence"),
        )
        # Every channel keeps the identity map: JSON calibration artifacts
        # are not loaded here.
        return UncertaintyCalibration(weights=weights, maps={}, calibration_id=None)

    def _build_uncertainty_channel_values(
        self,
        *,
        frame: Any,
        grasp_result: Any,
    ) -> UncertaintyChannelValues:
        """Assemble per-tick channel observations from live runtime state.

        No channel has a producer, so every channel stays :data:`None`:
        ``fused_available`` is :data:`False` and the engine reverts to the
        baseline. Channels can be filled in one at a time (depth-confidence
        stats from the perception frame, mask-confidence from segmentation
        backends).
        """

        return UncertaintyChannelValues()

    # Decision-layer pre-flight loop

    def _pick_with_decision(
        self,
        *,
        effective_mode: GraspMode,
        profile: GraspBehaviorProfile,
    ) -> AutonomousGraspReport:
        """One perception tick decided by :class:`DecisionEngine`.

        Steps:

        1. Evaluate the drift and OOD watchdog. An enforced
           ``BLOCK_AUTO`` ends the pick with a synthetic ``FAIL_CLOSED``
           before anything is perceived.
        2. Acquire a perception frame and compute best candidates via
           the orchestrator's helper, which honours the configured
           :class:`FrameResolver`. A camera on the wrist looks around
           instead (``BinPickingOrchestrator.look_around``): the looks
           the pick was handed, each fused with the ones before, and the
           engine decides on the grasp they judged. A look sequence that
           stopped on a motion or a stop request ends the pick as a
           refused look or a cancel, with nothing else commanded.
        3. Hand the typed inputs to
           :meth:`DecisionEngine.decide`.
        4. Dispatch:

           * ``GRASP_NOW``: delegate to :meth:`_execute_pick` and
             attach the decision telemetry. A wrist pick's looks are
             handed on (``go_on_with``), so the pick goes on with the
             grasp that was decided on and does not look again. A wrist
             pick that asked for both jaw contact faces (``both_faces``)
             and whose looks did not see both is not gripped: a typed
             ``NO_VALID_GRASP`` report, the engine's decision kept on it.
           * ``RECOVER``: return a typed
             ``DECISION_RECOVER_PENDING`` report. No physical recovery
             action is executed here.
           * ``FAIL_CLOSED``: return a typed
             ``DECISION_FAIL_CLOSED`` report with structured
             telemetry. No pick attempt is dispatched.

        A fixed camera's decision never moves the arm. A wrist camera's
        moves it through the pick's looks before anything is decided:
        each a judged motion, with nothing said to the hand. The bounded
        loop that asked a viewpoint planner for another pose on
        ``MOVE_CAMERA`` and commanded the arm there, without checking the
        motion's result, was removed on 2026-09-29 with the planners.
        """

        # Engine + policy are narrowed by the caller-side gate.
        assert self.decision_engine is not None
        assert self.decision_policy is not None
        engine = self.decision_engine
        orch = self.runtime.orchestrator
        is_simulated = bool(orch.arm.capabilities.is_simulated)

        # Stable per-call attempt id. Used purely for telemetry
        # correlation; the wrapped runtime constructs its own
        # attempt records inside :class:`PickSessionReport`.
        attempt_id = f"auto-{uuid.uuid4().hex[:12]}"

        # Drift + OOD watchdog policy is built once per pick(). When
        # :data:`None` the watchdog is inert: a ``from_components``
        # service carries no effective config.
        watchdog_policy = self._build_watchdog_policy()

        # Fused-uncertainty surfaces. The calibration is built once per
        # pick (cheap; no I/O when no artifact path is set). The runtime
        # gate ``uncertainty_active`` is :data:`True` only when the
        # operator opted in via robot.grasping.uncertainty.enabled and
        # the current mode is in apply_modes. ``EASY`` is permanently
        # excluded from the default apply_modes tuple.
        (
            calibration,
            fail_closed_threshold,
            disagreement_threshold,
            ranking_penalty_weight,
            uncertainty_active,
        ) = self._resolve_uncertainty_runtime(effective_mode)

        # Pre-decide watchdog gate. Evaluates over the rolling
        # history accumulated from prior attempts. Returns
        # :data:`None` when the watchdog has no policy wired. The
        # gate fires only when ``report.enforced`` is :data:`True`,
        # which already encodes the hardware lock (real hardware +
        # canary/active + mode in policy.block_modes) and the
        # ``EASY`` exclusion, since schema validation rejects
        # ``EASY`` from ``block_modes``.
        watchdog_report, terminal_report = self._handle_watchdog_pretick(
            policy=watchdog_policy,
            effective_mode=effective_mode,
            is_simulated=is_simulated,
            attempt_id=attempt_id,
            profile=profile,
        )
        if terminal_report is not None:
            return terminal_report

        looked: "LookedAround | None" = None
        frame: Optional[Any]
        grasp_result: Optional[Any]
        if self.perceives_from_the_wrist:
            # A camera on the wrist decides on its looks, fused (plan A2): one look sequence, whose judgement the pick
            # goes on with on GRASP_NOW. What the looks came to is stamped on the report by the caller.
            looked = orch.look_around()
            if looked.cancelled:
                return self._cancelled_report(mode=effective_mode)
            if looked.stopped is not None:
                return self._looks_stopped_report(looked, effective_mode=effective_mode)
            judged = looked.judged
            frame = judged.frame if judged is not None else None
            grasp_result = judged.result if judged is not None else None
        else:
            frame = orch.perception.acquire()
            grasp_result = self._rank_grasp_for_decision(frame)

        uncertainty_snapshot = self._build_uncertainty_snapshot_for_tick(
            frame=frame,
            grasp_result=grasp_result,
            calibration=calibration,
            uncertainty_active=uncertainty_active,
            fail_closed_threshold=fail_closed_threshold,
            disagreement_threshold=disagreement_threshold,
            attempt_id=attempt_id,
        )

        decision = self._decide_once(
            engine=engine,
            grasp_result=grasp_result,
            effective_mode=effective_mode,
            attempt_id=attempt_id,
            is_simulated=is_simulated,
            uncertainty_snapshot=uncertainty_snapshot,
            uncertainty_active=uncertainty_active,
            ranking_penalty_weight=ranking_penalty_weight,
        )
        # Record one watchdog sample per decision (keeps the rolling
        # history fresh for the next pick()).
        self._record_watchdog_sample(
            self._build_watchdog_sample_from_decision(decision)
        )

        if decision.action is DecisionAction.GRASP_NOW:
            if looked is not None and _both_faces_unseen(orch, looked):
                # The owner's switch for safety-critical processes (addendum 1): no grip on a grasp whose jaw contact
                # faces the looks did not both see. The engine's word is kept on the report beside the refusal.
                faces = looked.jaw_faces_seen
                _LOG.warning(
                    "%s (looked from %s), and the pick asked for both jaw contact faces (both_faces): it does not grip",
                    "the contact faces of the chosen grasp could not be judged" if faces is None else " and ".join(
                        f"the contact face at jaw {jaw} of the chosen grasp was not seen"
                        for jaw, seen in zip((1, 2), faces) if not seen),
                    ", ".join(looked.visited) or "no look")
                return self._terminal_decision_report(
                    effective_mode=effective_mode,
                    profile=profile,
                    decision=decision,
                    watchdog_report=watchdog_report,
                    n_segmentations=len(frame.segmentations) if frame is not None else 0,
                    reasons=tuple(getattr(grasp_result, "reasons", ()) or ()),
                    refused_as=AutonomousGraspOutcome.NO_VALID_GRASP,
                )
            if looked is not None:
                # The looks just decided on are the ones the pick goes on with: the pick loop does not look again.
                orch.go_on_with(looked)
            return self._execute_pick(
                effective_mode=effective_mode,
                profile=profile,
                decision=decision,
                watchdog_report=watchdog_report,
            )

        # RECOVER or FAIL_CLOSED: terminate without dispatching. What the frame held goes with it, so a
        # pick handed several looks can tell a view that held nothing from an object it could not grasp.
        # A wrist pick's frame is the look its grasp was judged on; no look that saw anything held nothing.
        return self._terminal_decision_report(
            effective_mode=effective_mode,
            profile=profile,
            decision=decision,
            watchdog_report=watchdog_report,
            n_segmentations=len(frame.segmentations) if frame is not None else 0,
            reasons=tuple(getattr(grasp_result, "reasons", ()) or ()),
        )

    # Decision helpers

    def _resolve_uncertainty_runtime(
        self, effective_mode: GraspMode
    ) -> tuple[Optional[UncertaintyCalibration], float, float, float, bool]:
        """The per-pick uncertainty runtime: (calibration, fail_closed_threshold,
        disagreement_threshold, ranking_penalty_weight, uncertainty_active).
        ``uncertainty_active`` is True only when the operator opted in
        (effective_config.uncertainty.enabled) and the mode is in apply_modes;
        ``EASY`` is permanently excluded. Inactive yields the inert sentinels
        (None / 0.0 / 1.01 / 0.0 / False)."""
        ucfg = self.effective_config
        uncertainty_enabled_cfg = bool(
            ucfg is not None and ucfg.uncertainty.enabled
        )
        uncertainty_modes = (
            frozenset(ucfg.uncertainty.apply_modes) if ucfg is not None else frozenset()
        )
        uncertainty_active = bool(
            uncertainty_enabled_cfg
            and effective_mode.value in uncertainty_modes
        )
        if uncertainty_active:
            # ``uncertainty_active`` implies ``ucfg is not None`` (see guards above);
            # assert so the type checker can narrow ``ucfg`` here.
            assert ucfg is not None
            calibration: Optional[UncertaintyCalibration] = (
                self._build_uncertainty_calibration()
            )
            fail_closed_threshold = float(ucfg.uncertainty.fail_closed_threshold)
            disagreement_threshold = float(
                ucfg.uncertainty.channel_disagreement_threshold
            )
            ranking_penalty_weight = float(
                ucfg.uncertainty.ranking_penalty_weight
            )
        else:
            calibration = None
            fail_closed_threshold = _UNCERTAINTY_INACTIVE_FAIL_CLOSED_THRESHOLD
            disagreement_threshold = _UNCERTAINTY_INACTIVE_DISAGREEMENT_THRESHOLD
            ranking_penalty_weight = _UNCERTAINTY_INACTIVE_RANKING_PENALTY_WEIGHT
        return (
            calibration,
            fail_closed_threshold,
            disagreement_threshold,
            ranking_penalty_weight,
            uncertainty_active,
        )

    def _rank_grasp_for_decision(self, frame: Any) -> Optional[Any]:
        """Rank the frame's segmentations into a best candidate inside the ranking latency span.
        Empty segmentations short-circuit to None without opening a span (null contract)."""
        if not frame.segmentations:
            return None
        orch = self.runtime.orchestrator
        # The ranking span lives inside ``_best_result_over_segmentations``, the one
        # call every pick path makes; opening one here as well would nest a span in a
        # span and count the same work twice. See the carrier comment in pick_loop.py.
        grasp_result, _idx, _ord_decision = orch._best_result_over_segmentations(frame)
        return grasp_result

    def _build_uncertainty_snapshot_for_tick(
        self,
        *,
        frame: Any,
        grasp_result: Optional[Any],
        calibration: Optional[UncertaintyCalibration],
        uncertainty_active: bool,
        fail_closed_threshold: float,
        disagreement_threshold: float,
        attempt_id: str,
    ) -> Optional[UncertaintySnapshot]:
        """The per-tick fused-uncertainty snapshot. None when inactive, and the engine
        then uses the baseline. Prints the ``[U8] uncertainty_no_signal`` diagnostic
        when no producer wired a channel this tick; the snapshot is still attached to
        the decision report."""
        if not (uncertainty_active and calibration is not None):
            return None
        channel_values = self._build_uncertainty_channel_values(
            frame=frame, grasp_result=grasp_result
        )
        uncertainty_snapshot = fuse_uncertainty(
            channel_values,
            calibration,
            fail_closed_threshold=fail_closed_threshold,
            channel_disagreement_threshold=disagreement_threshold,
        )
        if not uncertainty_snapshot.fused_available:
            print(f"[U8] uncertainty_no_signal: attempt_id={attempt_id}")
        return uncertainty_snapshot

    def _decide_once(
        self,
        *,
        engine: DecisionEngine,
        grasp_result: Optional[Any],
        effective_mode: GraspMode,
        attempt_id: str,
        is_simulated: bool,
        uncertainty_snapshot: Optional[UncertaintySnapshot],
        uncertainty_active: bool,
        ranking_penalty_weight: float,
    ) -> DecisionReport:
        """One engine.decide(), inside the decision latency span when a tracker is
        installed. The watchdog sample is recorded at the call site."""
        tracker = self._current_latency_tracker
        if tracker is not None:
            with tracker.span(LatencyStage.DECISION):
                return engine.decide(
                    grasp_result=grasp_result,
                    mode=effective_mode.value,
                    attempt_id=attempt_id,
                    is_simulated=is_simulated,
                    uncertainty_snapshot=uncertainty_snapshot,
                    uncertainty_active=uncertainty_active,
                    ranking_penalty_weight=ranking_penalty_weight,
                )
        return engine.decide(
            grasp_result=grasp_result,
            mode=effective_mode.value,
            attempt_id=attempt_id,
            is_simulated=is_simulated,
            uncertainty_snapshot=uncertainty_snapshot,
            uncertainty_active=uncertainty_active,
            ranking_penalty_weight=ranking_penalty_weight,
        )

    def _handle_watchdog_pretick(
        self,
        *,
        policy: Optional[WatchdogPolicy],
        effective_mode: GraspMode,
        is_simulated: bool,
        attempt_id: str,
        profile: GraspBehaviorProfile,
    ) -> tuple[
        Optional[WatchdogReport],
        Optional[AutonomousGraspReport],
    ]:
        """The pre-decide watchdog gate. Returns ``(watchdog_report, terminal_report)``:
        ``watchdog_report`` is threaded downstream, since the GRASP_NOW dispatch and the
        terminal reports carry it; ``terminal_report`` is non-None only when an enforced
        BLOCK_AUTO fires, and that branch records the fail-closure sample and builds the
        terminal report."""
        watchdog_report = self._evaluate_watchdog_pretick(
            policy=policy,
            effective_mode=effective_mode,
            is_simulated=is_simulated,
        )
        if watchdog_report is not None:
            self._fire_watchdog_transitions(
                curr=watchdog_report,
                attempt_id=attempt_id,
                effective_mode=effective_mode,
            )
        if (
            watchdog_report is not None
            and watchdog_report.enforced
            and watchdog_report.recommended_action
            is WatchdogAction.BLOCK_AUTO
        ):
            synthetic = self._synthesize_watchdog_block_decision(
                report=watchdog_report,
                effective_mode=effective_mode,
                attempt_id=attempt_id,
            )
            # Record a sample so subsequent ticks see this fail-closure in the rolling rate.
            self._record_watchdog_sample(
                self._build_watchdog_sample_from_decision(synthetic)
            )
            terminal = self._terminal_decision_report(
                effective_mode=effective_mode,
                profile=profile,
                decision=synthetic,
                watchdog_report=watchdog_report,
            )
            return watchdog_report, terminal
        return watchdog_report, None

    # Watchdog (drift + OOD). Logic lives in watchdog.WatchdogCoordinator
    # (self._watchdog). These shims delegate to it and keep the pick()-path
    # call sites and the ``service.watchdog_history`` injection point stable.

    def _build_watchdog_policy(self) -> Optional[WatchdogPolicy]:
        return self._watchdog.build_policy(self.effective_config)

    def _evaluate_watchdog_pretick(
        self,
        *,
        policy: Optional[WatchdogPolicy],
        effective_mode: GraspMode,
        is_simulated: bool,
    ) -> Optional[WatchdogReport]:
        return self._watchdog.evaluate_pretick(
            history=self.watchdog_history,
            policy=policy,
            effective_mode=effective_mode,
            is_simulated=is_simulated,
        )

    def _watchdog_telemetry_dict(
        self, report: WatchdogReport
    ) -> dict[str, Any]:
        return self._watchdog.telemetry_dict(report)

    def _synthesize_watchdog_block_decision(
        self,
        *,
        report: WatchdogReport,
        effective_mode: GraspMode,
        attempt_id: str,
    ) -> DecisionReport:
        return self._watchdog.synthesize_block_decision(
            report=report,
            effective_mode=effective_mode,
            attempt_id=attempt_id,
        )

    def _record_watchdog_sample(self, sample: WatchdogSample) -> None:
        self._watchdog.record_sample(self.watchdog_history, sample)

    @staticmethod
    def _build_watchdog_sample_from_decision(
        decision: DecisionReport,
    ) -> WatchdogSample:
        return WatchdogCoordinator.build_sample_from_decision(decision)

    def _fire_watchdog_transitions(
        self,
        *,
        curr: WatchdogReport,
        attempt_id: str,
        effective_mode: GraspMode,
    ) -> None:
        # A pick evaluates the watchdog once, so every flag this evaluation raises is a rising edge
        # for the pick (see WatchdogCoordinator._detect_transitions).
        self._watchdog.fire_transitions(
            event_listener=self.watchdog_event_listener,
            curr=curr,
            attempt_id=attempt_id,
            effective_mode=effective_mode,
        )

    def _base_telemetry(self, profile: GraspBehaviorProfile) -> dict[str, Any]:
        """The shared leading telemetry key (``sampling_mode``) for every AutonomousGraspReport
        this service builds: one source, so the projection cannot drift across the legacy
        and decision paths. Callers add their path-specific keys after, so the insertion
        order keeps ``sampling_mode`` first."""
        return {"sampling_mode": str(profile.sampling_mode)}

    def _terminal_decision_report(
        self,
        *,
        effective_mode: GraspMode,
        profile: GraspBehaviorProfile,
        decision: DecisionReport,
        watchdog_report: Optional[WatchdogReport] = None,
        n_segmentations: Optional[int] = None,
        reasons: Optional[tuple[Any, ...]] = None,
        refused_as: Optional[AutonomousGraspOutcome] = None,
    ) -> AutonomousGraspReport:
        """Wrap a non-executing decision into an :class:`AutonomousGraspReport`.

        ``n_segmentations`` and ``reasons`` are what the decided frame held: how many objects, and why its ranking
        grasped none (``target_label_not_found`` where no object carried the prompted label). The engine answers an
        empty frame and an object it cannot grasp with the same ``RECOVER`` for want of candidates; these two keys
        are what tells a look that found nothing from one that found something (``_found_nothing``). Left out where
        no frame was decided on (a watchdog block before perception).

        ``refused_as`` is the service's own outcome where it refused a decision to grasp (a wrist pick that asked for
        both jaw contact faces and did not see both); left out, the decision's action names it.
        """

        if refused_as is not None:
            outcome = refused_as
        elif decision.action is DecisionAction.RECOVER:
            outcome = AutonomousGraspOutcome.DECISION_RECOVER_PENDING
        elif decision.action is DecisionAction.FAIL_CLOSED:
            # Prefer the uncertainty-specific terminal outcome when the
            # fail-closure was driven by fused signal
            # (UNCERTAINTY_FAIL_CLOSED) or channel disagreement
            # (CHANNEL_DISAGREEMENT). The two watchdog reason codes map
            # to dedicated outcomes so replay can attribute drift and
            # OOD fail-closures separately. DECISION_FAIL_CLOSED is
            # reserved for the ``(1 - top_score) + reasons_penalty``
            # path.
            if decision.reason_code is DecisionReasonCode.DRIFT_BLOCKED_AUTO:
                outcome = AutonomousGraspOutcome.DRIFT_BLOCKED_AUTO
            elif decision.reason_code is DecisionReasonCode.OOD_BLOCKED_AUTO:
                outcome = AutonomousGraspOutcome.OOD_BLOCKED_AUTO
            elif decision.reason_code in (
                DecisionReasonCode.UNCERTAINTY_FAIL_CLOSED,
                DecisionReasonCode.CHANNEL_DISAGREEMENT,
            ):
                outcome = AutonomousGraspOutcome.UNCERTAINTY_FAIL_CLOSED
            else:
                outcome = AutonomousGraspOutcome.DECISION_FAIL_CLOSED
        else:  # pragma: no cover (defensive; the caller filters above)
            outcome = AutonomousGraspOutcome.MODE_NOT_AVAILABLE

        telemetry = self._base_telemetry(profile)
        telemetry.update(decision.to_dict())
        if watchdog_report is not None:
            telemetry.update(self._watchdog_telemetry_dict(watchdog_report))
        if n_segmentations is not None:
            telemetry["n_segmentations"] = int(n_segmentations)
        if reasons is not None:
            telemetry["reasons"] = tuple(str(r) for r in reasons)
        return AutonomousGraspReport(
            outcome=outcome,
            mode=effective_mode,
            profile=replace(profile),
            pick_report=None,
            telemetry=telemetry,
            effective_config=self.effective_config,
            decision=decision,
        )

    # The pick body, called directly by the decision pre-flight on
    # ``GRASP_NOW``.

    def _execute_pick(
        self,
        *,
        effective_mode: GraspMode,
        profile: GraspBehaviorProfile,
        decision: Optional[DecisionReport],
        watchdog_report: Optional[WatchdogReport] = None,
    ) -> AutonomousGraspReport:
        """Run one open-loop attempt (``runtime.run_attempt()``) and fold the pre-flight in.

        ``decision`` is :data:`None` when no decision pre-flight ran and
        a typed :class:`DecisionReport` when the pre-flight selected
        ``GRASP_NOW``. When present, its telemetry is merged into
        every returned :class:`AutonomousGraspReport` and the typed
        report carries the engine's :attr:`decision` field.

        ``watchdog_report`` is :data:`None` when the watchdog is unwired
        and a typed :class:`WatchdogReport` when it is wired. When
        present (and the watchdog mode is not ``disabled``) its
        telemetry keys are merged into the returned report.
        """

        report = self._run_legacy_attempt(
            effective_mode=effective_mode, profile=profile
        )
        if decision is None and watchdog_report is None:
            return report
        merged_telemetry: dict[str, Any] = dict(report.telemetry)
        if decision is not None:
            merged_telemetry.update(decision.to_dict())
        if watchdog_report is not None:
            merged_telemetry.update(
                self._watchdog_telemetry_dict(watchdog_report)
            )
        return replace(report, telemetry=merged_telemetry, decision=decision)

    def _run_legacy_attempt(
        self,
        *,
        effective_mode: GraspMode,
        profile: GraspBehaviorProfile,
    ) -> AutonomousGraspReport:
        """The open-loop attempt: ``runtime.run_attempt()``, the mapping from
        PickOutcome to AutonomousGraspOutcome, and the report build.

        The fusion span is not timed here. The orchestrator records it around
        the geometry fusion itself (``BinPickingOrchestrator._fused_scene``), on
        this path and on every other, and only when another camera's view was
        fused."""
        pick_report = self.runtime.run_attempt()
        outcome = _PICK_TO_AUTONOMOUS.get(
            pick_report.outcome,
            AutonomousGraspOutcome.EXECUTION_FAILED,
        )

        telemetry = self._base_telemetry(profile)
        telemetry["low_level_outcome"] = str(pick_report.outcome)
        if pick_report.error is not None:
            # Carry the type name only; the exception itself stays on
            # the underlying PickSessionReport. Telemetry must remain
            # JSON-serializable.
            telemetry["runtime_error_type"] = type(pick_report.error).__name__

        return AutonomousGraspReport(
            outcome=outcome,
            mode=effective_mode,
            # Snapshot the profile so a later config reload cannot
            # rewrite history on this report.
            profile=replace(profile),
            pick_report=pick_report,
            telemetry=telemetry,
            effective_config=self.effective_config,
            shadow_success_telemetry=pick_report.shadow_success_telemetry,
            ranking_blend_telemetry=pick_report.ranking_blend_telemetry,
            uncertainty_rerank_telemetry=pick_report.uncertainty_rerank_telemetry,
        )


def _report_of_looking(
    report: AutonomousGraspReport, looked: "LookedAround | None", *, handed: bool, both_faces: bool = False,
) -> AutonomousGraspReport:
    """``report`` with what a wrist pick's looks came to (``orch.looked_around``), or as it is when nothing looked.

    ``looks`` are the looks perceived from and ``looks_fused`` those whose views make up the grasp's cloud, the one it
    was ranked on first, both only on a pick handed looks: a pick handed none names none, as it always did.
    ``jaw_faces_seen`` and ``hand_eye_gap_mm`` are the chosen grasp's contact faces seen and the hand-eye check, and
    ``both_faces`` whether the pick asked for both faces. A look sequence that ended on a motion says the look it did
    not reach and why (``refused_look``, ``look_refused``); one a stop ended, how far it got
    (``cancelled_before_start``). A look the planner refused before anything was sent, skipped while the pick went on
    to a look it reached, is said as ``"look: why"`` in ``looks_skipped`` (addendum 2 skips it; the report names it).
    The one view the looks generated as their last resort is ``generated_view_deg``, its turn about the part (its look
    is among ``looks``), and a move back to the look the grasp was ranked on that was refused, so that the approach
    started from where the arm stood, is ``move_back_refused``. A generated view or a move back that ended the pick on
    its motion is said as a look not reached is. The fail-closed ending of ``both_faces`` reaches the report as its
    outcome, ``NO_VALID_GRASP``, and :meth:`AutonomousGraspReport.failure_summary` names the face from
    ``jaw_faces_seen``.
    """
    if looked is None:
        return replace(report, both_faces=True) if both_faces else report
    telemetry = dict(report.telemetry)
    if handed and looked.cancelled:
        telemetry.update({"cancelled_before_start": not looked.visited, "stage": "look"})
    if looked.stopped is not None:
        telemetry.update({
            "stage": "look", "refused_look": looked.stopped_look,
            "look_refused": f"{looked.stopped.status.value}: {looked.stopped.message}",
        })
    if looked.refused and looked.visited:
        # Not where no look was reached: the refusal that ended the pick is `look_refused` above.
        telemetry["looks_skipped"] = [f"{look}: {why}" for look, why in looked.refused]
    if looked.move_back_refused:
        # The approach started from where the arm stood, not from the look its grasp was ranked on.
        telemetry["move_back_refused"] = looked.move_back_refused
    return replace(
        report,
        telemetry=telemetry,
        looks=tuple(looked.visited) if handed else (),
        looks_fused=tuple(looked.looks_fused) if handed else (),
        jaw_faces_seen=looked.jaw_faces_seen,
        hand_eye_gap_mm=looked.hand_eye_gap_mm,
        both_faces=bool(both_faces),
        generated_view_deg=looked.generated_view_deg if handed else None,
    )


def _both_faces_unseen(orchestrator: Any, looked: "LookedAround") -> bool:
    """Whether a pick that asked for both jaw contact faces (``orch.both_faces``) would grip a grasp whose faces its
    looks did not both see: the owner's switch for safety-critical processes then refuses the grip (addendum 1).

    As the pick loop judges a look good: a hand with no jaw faces, a suction cup, has none to see; faces that could not
    be judged are not seen, so the switch fails closed.
    """
    if getattr(orchestrator, "both_faces", False) is not True:
        return False
    from src.robot.grasping.collision.gripper_model import SuctionCupGripperModel  # noqa: PLC0415

    if isinstance(getattr(orchestrator, "gripper_model", None), SuctionCupGripperModel):
        return False
    return looked.jaw_faces_seen != (True, True)


def found_nothing(report: AutonomousGraspReport) -> bool:
    """Whether a pick found nothing to pick from where it looked: no object at all, or none with the prompted label.

    The service's own rule (:func:`_found_nothing`), public for a caller that counts empty looks (a task ends
    ``nothing_left`` after two in a row). A part seen with no grasp found is something, not nothing.
    """
    return _found_nothing(report)


def only_excluded(report: AutonomousGraspReport) -> bool:
    """Whether a pick saw only parts its campaign keeps out: the parts ``next_target`` skips, or a region its task keeps
    out (the bin it places into, the circle about its drop).

    The pick loop ends such an attempt with the zones' sentence (``PickAttempt.excluded``) and the service reports it
    ``no_valid_grasp``, which :func:`found_nothing` does not count. Read off the pick report's last attempt, so a report
    that ran no pick says no. A task counts as an empty look only :func:`only_kept_out`, the narrower of the two.
    """
    excluded = getattr(_last_attempt(report), "excluded", None)
    return isinstance(excluded, str) and bool(excluded)


def only_kept_out(report: AutonomousGraspReport) -> bool:
    """Whether a pick saw only parts its task keeps out, every one of them standing in a region of the task (the bin it
    places into, the circle about its drop): a look with nothing left for the task to pick, which a task counts as an
    empty look, not a failed pick.

    A pick that saw a part a zone ``next_target`` made after a failed pick skips, alone or beside the task's own, is
    :func:`only_excluded` and not this: that part still stands there to be picked, and the look is a failed pick. Read
    off the pick loop's last attempt (``PickAttempt.excluded_by_regions``).
    """
    return only_excluded(report) and getattr(_last_attempt(report), "excluded_by_regions", False) is True


def _last_attempt(report: AutonomousGraspReport) -> Any:
    """The pick loop's last attempt of ``report``, or ``None`` where it ran none."""
    attempts = getattr(getattr(report, "pick_report", None), "attempts", ()) or ()
    return attempts[-1] if attempts else None


def _found_nothing(report: AutonomousGraspReport) -> bool:
    """Whether a pick found nothing to pick from where it looked: no object at all, or none with the prompted label.

    What sends a fixed camera's pick on to its next look. Anything else was found and is answered from where it was
    seen: a grasp the calculator refused, a motion that failed, a stop. A wrist camera's looks are the pick loop's
    (:meth:`AutonomousGraspService._look_around_and_pick`), which goes on by its own rule: to the next look, fused,
    until a grasp is safe. A wrist pick handed no look has no next look, and rescans where it stands instead.

    Each pick path says it in its own words. The pick loop's: ``NO_TARGET``, or a last attempt refused for want of the
    label. The decision layer's: ``DECISION_RECOVER_PENDING`` (no candidates) on a frame that held no object, or whose
    ranking carried ``target_label_not_found`` (:meth:`AutonomousGraspService._terminal_decision_report` keeps both).
    A report that does not say which (a watchdog block has no frame) is not empty: it ends the looking where it stands.
    The two-scan refinement's own words (``NO_VALID_GRASP`` at its first compute) left with it on 2026-09-29.
    """
    if report.outcome is AutonomousGraspOutcome.NO_TARGET:
        return True
    # The pick loop's own word underneath, where a recovery loop that ran out of actions reports recovery_exhausted.
    if getattr(report.pick_report, "outcome", None) is PickOutcome.NO_PERCEPTION:
        return True
    not_found = GraspFailureReason.TARGET_LABEL_NOT_FOUND.value
    telemetry = report.telemetry or {}
    said = tuple(str(reason) for reason in (telemetry.get("reasons") or ()))
    if report.pick_report is None:
        if report.outcome is AutonomousGraspOutcome.DECISION_RECOVER_PENDING:
            return telemetry.get("n_segmentations") == 0 or not_found in said
        return False
    attempts = getattr(report.pick_report, "attempts", ()) or ()
    reasons = tuple(getattr(attempts[-1], "reasons", ()) or ()) if attempts else ()
    return GraspFailureReason.TARGET_LABEL_NOT_FOUND in reasons


def _controller_stop_telemetry(orchestrator: Any) -> dict[str, Any]:
    """The telemetry of a look the arm did not reach on a controller that cannot move; ``{}`` otherwise.

    Asked of the pick loop's own diagnosis (``BinPickingOrchestrator._controller_cannot_move``), so a look that failed
    names a stopped cell in the words, and with the ``low_level_outcome``, the open-loop path does, and
    :attr:`AutonomousGraspReport.controller_stopped` reads both alike. Asked only after a motion failed. An
    orchestrator without the diagnosis, or one whose diagnosis raises, says nothing: a diagnosis must never replace
    the failure it explains.
    """
    diagnose = getattr(orchestrator, "_controller_cannot_move", None)
    if not callable(diagnose):
        return {}
    try:
        reason = diagnose()
    except Exception:  # noqa: BLE001 (the failure stands as it was)
        return {}
    if not isinstance(reason, str) or not reason:
        return {}
    return {"low_level_outcome": str(PickOutcome.CONTROLLER_NOT_OPERATIONAL), "controller": reason}


@contextmanager
def _handed(orchestrator: Any, *, zones: Any, gate: Any, blocker_is_the_pick: bool = False) -> Iterator[None]:
    """Hand the pick loop the campaign's exclusion zones, this pick's push gate and whether a blocker it takes away is
    the part it picks, and take them back after it, as the looks and ``both_faces`` are handed: the next pick inherits
    none of them."""
    before = (getattr(orchestrator, "exclusion_zones", None), getattr(orchestrator, "push_gate", None),
              getattr(orchestrator, "blocker_is_the_pick", False))
    orchestrator.exclusion_zones = zones
    orchestrator.push_gate = gate
    if hasattr(orchestrator, "blocker_is_the_pick"):
        orchestrator.blocker_is_the_pick = bool(blocker_is_the_pick)
    try:
        yield
    finally:
        orchestrator.exclusion_zones, orchestrator.push_gate = before[0], before[1]
        if hasattr(orchestrator, "blocker_is_the_pick"):
            orchestrator.blocker_is_the_pick = before[2]


def _pushes_of(orchestrator: Any) -> "tuple[PickPush, ...]":
    """What the pushes of the pick that just ran came to, as the pick loop keeps them (``pushes``); none from a loop
    that keeps none."""
    from src.robot.grasping.recovery.push_gate import PickPush  # noqa: PLC0415

    kept = getattr(orchestrator, "pushes", ())
    return tuple(push for push in kept if isinstance(push, PickPush)) if isinstance(kept, tuple) else ()


def _with_pushes(report: AutonomousGraspReport, pushes: "tuple[PickPush, ...]") -> AutonomousGraspReport:
    """``report`` with what its pick's pushes came to: every push it considered on the telemetry (``pushes``), and a
    push that stopped where the arm stands said (``push_stopped``), the pick then ``unsafe_recovery_refused`` unless
    its controller stopped (``CANCELLED``, :attr:`AutonomousGraspReport.controller_stopped`). Either way the report
    needs a person (:attr:`AutonomousGraspReport.needs_person`), and nothing more moves."""
    if not pushes:
        return report
    telemetry = dict(report.telemetry)
    telemetry["pushes"] = [push.to_dict() for push in pushes]
    stopped = next((push for push in pushes if push.stopped), None)
    if stopped is None:
        return replace(report, telemetry=telemetry)
    telemetry["push_stopped"] = stopped.sentence
    outcome = report.outcome if stopped.controller_stopped else AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED
    return replace(report, telemetry=telemetry, outcome=outcome)


def _with_the_pushes_of_a_fault(
    report: AutonomousGraspReport, pushes: "tuple[PickPush, ...]",
) -> AutonomousGraspReport:
    """A fault's ``report`` with the pushes of the pick that raised it: every push it considered on the telemetry
    (``pushes``), and a push the fault stopped where the arm stands said (``push_stopped``), so the report needs a person
    (:attr:`AutonomousGraspReport.needs_person`). The outcome stays the fault's."""
    if not pushes:
        return report
    telemetry = dict(report.telemetry)
    telemetry["pushes"] = [push.to_dict() for push in pushes]
    stopped = next((push for push in pushes if push.stopped), None)
    if stopped is not None:
        telemetry["push_stopped"] = stopped.sentence
    return replace(report, telemetry=telemetry)


def _with_the_recovery_outcome(report: AutonomousGraspReport, trail: RecoveryTrail) -> AutonomousGraspReport:
    """``report`` with the outcome the recovery loop stands for, where it stands for one
    (``terminal_outcome_for_trail``): ``unsafe_recovery_refused`` where a recovery motion started and failed,
    ``recovery_exhausted`` where at least one action ran and the loop ended for want of another. A report that
    succeeded, stopped for a person (a push that stopped, a stopped controller, a hand that needs one), was cancelled
    or carries a fault keeps its own, and so does one that found nothing (``no_target``): an empty bench is no dead
    loop, whatever the recovery rescanned (the KPI roll-up counts ``recovery_exhausted`` as one), and its own outcome
    says more, the rule ``terminal_outcome_for_trail`` keeps for a failure no action recovers."""
    outcome = getattr(report, "outcome", None)
    hand = getattr(report, "gripper_fault", "")
    kept = (AutonomousGraspOutcome.SUCCEEDED, AutonomousGraspOutcome.CANCELLED, AutonomousGraspOutcome.NO_TARGET)
    if (outcome in kept
            or getattr(report, "needs_person", False) is True or getattr(report, "controller_stopped", False) is True
            or (isinstance(hand, str) and hand) or getattr(report, "fault", None) is not None):
        return report
    terminal = terminal_outcome_for_trail(trail)
    if terminal == OUTCOME_UNSAFE_RECOVERY_REFUSED:
        return replace(report, outcome=AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED)
    if terminal == OUTCOME_RECOVERY_EXHAUSTED:
        return replace(report, outcome=AutonomousGraspOutcome.RECOVERY_EXHAUSTED)
    return report


def _trail_row(entry: RecoveryTrailEntry) -> dict[str, Any]:
    """One step of the recovery loop, as ``recovery_actions_from_trail`` writes it."""
    return {
        "action": str(entry.plan_action.value),
        "outcome": str(entry.outcome),
        "step_result": str(entry.outcome),
        "executed": bool(entry.executed),
        "failure_reason": str(entry.failure_reason.value),
        "plan_reason": str(entry.plan_reason),
    }


def _push_row(push: "PickPush") -> dict[str, Any]:
    """One push whose arm left the look, as a ``nudge_target`` step of the trail (the same six keys)."""
    return {
        "action": SceneRecoveryAction.NUDGE_TARGET.value,
        "outcome": push.code,
        "step_result": push.code,
        "executed": push.pushed,
        "failure_reason": push.trigger,
        "plan_reason": "push_in_the_pick",
    }


def _recovery_rows(
    trail: RecoveryTrail, pushes: "tuple[tuple[PickPush, ...], ...]",
) -> tuple[dict[str, Any], ...]:
    """The recovery a pick ran, step by step, as the record's ``recovery_actions`` carries it.

    The loop's steps (``recovery_actions_from_trail``'s rows) with every push whose arm left the look among them (one
    that moved the part, one that stopped where the arm stands, and P0 in the air above a part nothing touched), in the
    order they ran: the pushes of the loop's first pick, then the loop's steps up to the one that ran the pick again,
    then that pick's pushes, and so on. As ``recovery_actions_from_trail`` has it, the last step's outcome reads
    ``recovered_success`` where the recovery ended in a success. With no push this is that function's output exactly.
    """
    entries = list(trail.entries)
    rows: list[dict[str, Any]] = []
    index = 0
    for pick in pushes:
        rows.extend(_push_row(push) for push in pick if push.arm_moved or push.stopped)
        while index < len(entries):
            entry = entries[index]
            index += 1
            rows.append(_trail_row(entry))
            if entry.executed:
                break
    rows.extend(_trail_row(entry) for entry in entries[index:])
    if rows and trail.terminal_reason == "recovered_success":
        rows[-1]["outcome"] = "recovered_success"
    return tuple(rows)


def _operator_box(fixture: Any) -> "AxisBox | None | str":
    """The declared fixture box (``recovery.fixture``) as the push planner's operator box, which only narrows where a
    push may land; ``None`` without a fixture, and why not, as a sentence, where the box leaves no room in BASE XY."""
    if fixture is None:
        return None
    from src.robot.grasping.recovery.push_planner import AxisBox  # noqa: PLC0415

    (cx, cy, cz), (hx, hy, hz) = fixture.center_mm, fixture.half_extents_mm
    if not (float(hx) > 0.0 and float(hy) > 0.0):
        return (f"the recovery fixture box (recovery.fixture) is {2.0 * float(hx):g} x {2.0 * float(hy):g} mm in BASE "
                "XY, which leaves a pushed part nowhere to land")
    # Only its XY narrows the push box; a flat box is given a millimetre of height so it is a box.
    half_z = max(float(hz), 1.0)
    return AxisBox((float(cx) - float(hx), float(cy) - float(hy), float(cz) - half_z),
                   (float(cx) + float(hx), float(cy) + float(hy), float(cz) + half_z))


def _push_cell_of(robot_cfg: Any, grasping_cfg: Any) -> "PushCell | PushRefusal | None":
    """What the cell is as a push needs it, read wherever the config has a recovery block: a run may allow the push
    for itself from the console (the owner, 2026-10-05) where the config's ``allowed_actions`` does not name it."""
    recovery = getattr(grasping_cfg, "recovery", None)
    if recovery is None:
        return None
    from src.robot.grasping.recovery.push_gate import PushCell  # noqa: PLC0415

    return PushCell.from_robot_config(robot_cfg)


def _registry_palm_thickness_mm(robot_cfg: Any) -> "float | None":
    """The housing along the closing axis the gripper registry measured for ``robot.gripper.model``
    (``palm_thickness_mm``), as the push reads it (``push_hand``); ``None`` where no registry hand is named or it says
    none."""
    model = str(getattr(getattr(robot_cfg, "gripper", None), "model", "") or "")
    if not model:
        return None
    from src.config.grippers import load_gripper  # noqa: PLC0415 (config is read only when a cell is built)

    try:
        palm = getattr(load_gripper(model, aliases=False).jaw, "palm_thickness_mm", None)
    except Exception:  # noqa: BLE001 (a hand the registry does not hold keeps the palm as wide as its fingers)
        return None
    return None if palm is None else float(palm)
