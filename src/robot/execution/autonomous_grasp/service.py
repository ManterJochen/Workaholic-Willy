"""Autonomous grasp service: the operator-facing "one call to pick" surface.

This module is the high-level facade an operator or upstream API layer calls to
perform a grasp without having to assemble :class:`GraspCalculator`,
:class:`BinPickingOrchestrator`, viewpoint planners, execution policies,
verification policies, and recovery strategies by hand.

A mode whose machinery is not wired (two-scan refinement, post-grasp
verification, dense-clutter scene recovery) is refused with an honest
:class:`AutonomousGraspOutcome` (``MODE_NOT_AVAILABLE``). ``CLOSED_LOOP`` and
``DENSE_AUTONOMOUS`` are never silently degraded to behave like ``AUTO``: that
would misstate to the operator which guarantees are in effect.

Contract:

* :class:`GraspMode` wraps :class:`GraspSamplingMode`; the low-level enum is
  untouched and existing callers keep working.
* :attr:`GraspMode.EASY` is byte-equivalent to the :class:`RuntimePickService`
  happy path with ``SINGLE_OBJECT`` sampling and no verification.
* :class:`AutonomousGraspReport` composes :class:`PickSessionReport`; it does
  not widen it.
* Frame-completeness, refinement, verification, and recovery slots are exposed
  as Protocol-typed fields with no-op defaults.

This module imports only vendor-neutral surfaces. It is safe to import on macOS
with no hardware drivers installed.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

import numpy as np

from src.contracts import UNSET, Maybe, chosen
from src.geometry import Frame, Pose
from src.robot.core import (
    NO_PLAN_FAIL_SAFE_MESSAGE,
    Gripper,
    MotionStatus,
    RobotArm,
)
from src.robot.grasping.types.feedback import GraspFailureReason
from src.robot.execution.runtime_pick import (
    PickSessionReport,
    RuntimePickService,
)
from src.robot.grasping.motion.execution_policy import (
    GraspExecutionPolicy,
    PolicyOutcome,
    _grasp_point_to_quaternion,
)
from src.robot.grasping.motion.frame_resolver import FrameResolver
from src.robot.grasping.types.perception import MultiCameraPerceptionSource
from src.robot.grasping.generation.calculator import GraspCalculator
from src.robot.grasping.types.grasp_point import GraspPoint
from src.robot.grasping.loop.pick_loop import (
    PerceptionSource,
    ViewpointPlanner,
)
from src.robot.grasping.closed_loop.refinement import (
    PreGraspRefiner,
    RefinementOutcome,
    RefinementPolicy,
    RefinementReport,
    target_identity_from_segmentation,
)
from src.robot.grasping.closed_loop.verification import (
    GraspVerificationContext,
    GraspVerificationPolicy,
    GraspVerificationReport,
    GraspVerifier,
    VerificationOutcome,
)
from src.robot.grasping.recovery.policy import (
    FixtureEnvelope,
    SceneRecoveryAction,
    SceneRecoveryPolicy,
    SceneRecoveryStrategy,
)
from src.robot.grasping.recovery.orchestrator import (
    RecoveryDispatcher,
    RecoveryOrchestrator,
    RecoveryTrail,
    recovery_actions_from_trail,
    run_recovery_loop,
)

from .builders import (
    apply_orchestrator_overlays,
    assert_closed_loop_actors_wired,
    build_closed_loop_actors,
    build_config_frame_resolver,
    build_config_viewpoint_planner,
    build_decision_layer,
    build_effective_config,
    build_runtime,
    build_subpolicies,
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
    from src.config.schema.robot import RobotConfig
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
        if psr is not None and psr.attempts:
            last = psr.attempts[-1]
            reasons = tuple(last.reasons)
            if not reasons and _is_motion_plan_refusal(last):
                # Narrow on purpose. It fires only when the attempt carries no reason of its own
                # (so it can never displace a real one) and only for the refusal the drivers emit
                # before any command reaches the controller. ``MotionStatus.TIMEOUT`` alone is not
                # enough: on real hardware it also covers a command that was accepted and ran out
                # of budget, where the arm may still be moving, and re-perceiving that is not an
                # orchestration decision to make. The message constant is the discriminator both
                # drivers share; see ``NO_PLAN_FAIL_SAFE_MESSAGE``.
                reasons = (GraspFailureReason.MOTION_PLAN_REFUSED,)
        self.failure_reasons = reasons



#: Boot-time diagnostics only. Nothing on the pick path logs through this.
_LOG = logging.getLogger(__name__)


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


@dataclass
class AutonomousGraspService:
    """High-level operator-facing grasp service.

    The service wraps an existing :class:`RuntimePickService` and adds:

    * a typed :class:`GraspMode` selector with locked behavior profiles,
    * a high-level :class:`AutonomousGraspReport` outcome,
    * honest refusal for modes whose required machinery (refinement,
      verification, recovery) is not wired: no silent
      degradation.

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
    # Refinement wiring. ``refinement_policy`` is the operator's bounded
    # configuration; ``refiner`` is the concrete two-scan computation
    # (default :class:`DefaultPreGraspRefiner`). When either is
    # :data:`None` the service refuses :attr:`GraspMode.CLOSED_LOOP`
    # and :attr:`GraspMode.DENSE_AUTONOMOUS` with the same honest
    # :attr:`AutonomousGraspOutcome.MODE_NOT_AVAILABLE` outcome:
    # no silent degradation.
    refinement_policy: Optional[RefinementPolicy] = None
    refiner: Optional[PreGraspRefiner] = None
    # Verification wiring. ``verification_policy`` is the operator's
    # bounded configuration; ``verifier`` is the concrete
    # post-grasp check. When the active profile demands verification
    # but either is :data:`None` (or the policy is disabled) the
    # service refuses with the same honest
    # :attr:`AutonomousGraspOutcome.MODE_NOT_AVAILABLE` outcome as
    # the refinement gate: no silent degradation.
    verification_policy: Optional[GraspVerificationPolicy] = None
    verifier: Optional[GraspVerifier] = None
    # Recovery slots. The recovery planner is exposed on the service
    # so an operator can wire it via ``from_components`` /
    # ``from_robot_config`` without rebuilding the constructor surface.
    # These slots do not ship an autonomous retry loop; they exist so
    # config can flow through and the wiring surface stays stable.
    recovery_policy: Optional[SceneRecoveryPolicy] = None
    recovery_strategy: Optional[SceneRecoveryStrategy] = None
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
        viewpoint_planner: ViewpointPlanner | None = None,
        max_attempts: int = 5,
        standoff_mm: float = 80.0,
        retreat_mm: float = 100.0,
        policy: GraspExecutionPolicy | None = None,
        frame_resolver: FrameResolver | None = None,
        refinement_policy: RefinementPolicy | None = None,
        refiner: PreGraspRefiner | None = None,
        verification_policy: GraspVerificationPolicy | None = None,
        verifier: GraspVerifier | None = None,
        recovery_policy: SceneRecoveryPolicy | None = None,
        recovery_strategy: SceneRecoveryStrategy | None = None,
        decision_policy: DecisionPolicy | None = None,
        decision_engine: DecisionEngine | None = None,
        record_log_path: "str | Path | None" = None,
    ) -> "AutonomousGraspService":
        """Build the service directly from raw components.

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

        resolved_mode = resolve_grasp_mode(mode)
        profile = _profile_for(resolved_mode)
        # Build a default policy that fails closed when a resolver is
        # wired. An explicit caller-supplied policy is honoured
        # verbatim; mutating an externally owned object would be a
        # silent footgun.
        effective_policy = policy
        if effective_policy is None and frame_resolver is not None:
            effective_policy = GraspExecutionPolicy(
                arm=arm,
                gripper=gripper,
                standoff_mm=standoff_mm,
                retreat_mm=retreat_mm,
                require_base_frame_grasp=True,
            )
        runtime = RuntimePickService.from_components(
            arm=arm,
            calculator=calculator,
            perception=perception,
            gripper=gripper,
            viewpoint_planner=viewpoint_planner,
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
            refinement_policy=refinement_policy,
            refiner=refiner,
            verification_policy=verification_policy,
            verifier=verifier,
            recovery_policy=recovery_policy,
            recovery_strategy=recovery_strategy,
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
        viewpoint_planner: ViewpointPlanner | None = None,
        max_attempts: "Maybe[int]" = UNSET,
        standoff_mm: float = 80.0,
        retreat_mm: float = 100.0,
        policy: GraspExecutionPolicy | None = None,
        frame_resolver: FrameResolver | None = None,
        refinement_policy: RefinementPolicy | None = None,
        refiner: PreGraspRefiner | None = None,
        verification_policy: GraspVerificationPolicy | None = None,
        verifier: GraspVerifier | None = None,
        recovery_policy: SceneRecoveryPolicy | None = None,
        recovery_strategy: SceneRecoveryStrategy | None = None,
        recovery_fixture: FixtureEnvelope | None = None,
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
        * Sub-policy auto-build. When the caller does not supply a
          :class:`RefinementPolicy`,
          :class:`GraspVerificationPolicy`, or
          :class:`SceneRecoveryPolicy` and the corresponding
          ``robot_cfg.grasping.*`` sub-block is enabled, the matching
          policy object is materialised from the sub-block values.
          Disabled sub-blocks leave the slot as :data:`None` so the
          honest :attr:`AutonomousGraspOutcome.MODE_NOT_AVAILABLE`
          gates continue to fire. Concrete strategies
          (:class:`PreGraspRefiner`, :class:`GraspVerifier`,
          :class:`SceneRecoveryStrategy`) remain caller-owned: they
          carry hardware coupling the schema cannot express.
        * Physical recovery actions. When
          ``robot_cfg.grasping.dense_recovery`` requests a physical
          action (``nudge_target`` / ``container_agitate``) without
          a caller-supplied ``recovery_fixture``, this method raises
          :class:`ValueError`. The fixture envelope is hardware-shaped
          and intentionally not in the config schema.

        ``frame_resolver`` is plumbed through identically to
        :meth:`from_components`. See that method's docstring for the
        full fail-closed contract.
        """

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

        # Sub-policy auto-build (refinement / verification / recovery).
        (
            resolved_refinement_policy,
            resolved_verification_policy,
            resolved_recovery_policy,
        ) = build_subpolicies(
            grasping_cfg,
            refinement_policy=refinement_policy,
            verification_policy=verification_policy,
            recovery_policy=recovery_policy,
            standoff_mm=standoff_mm,
            recovery_fixture=recovery_fixture,
        )

        # The objects those policies need in order to do anything. A policy without its actor is
        # a flag that reads as on over a service that cannot act on it: with
        # `closed_loop.enabled: true` in YAML and no refiner, every pick is refused with
        # `mode_requires_refinement_but_no_refiner_wired`.
        resolved_refiner, resolved_verifier, resolved_recovery_strategy = (
            build_closed_loop_actors(
                grasping_cfg,
                refinement_policy=resolved_refinement_policy,
                verification_policy=resolved_verification_policy,
                recovery_policy=resolved_recovery_policy,
                refiner=refiner,
                verifier=verifier,
                recovery_strategy=recovery_strategy,
            )
        )
        assert_closed_loop_actors_wired(
            refinement_policy=resolved_refinement_policy,
            verification_policy=resolved_verification_policy,
            recovery_policy=resolved_recovery_policy,
            refiner=resolved_refiner,
            verifier=resolved_verifier,
            recovery_strategy=resolved_recovery_strategy,
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

        # Auto-build a static camera-to-base resolver from the configured calibration artifact when
        # the caller did not supply one. This is what makes the fusion substrate + the commit gate
        # reachable from production config. A caller-supplied frame_resolver (sim / eye-in-hand)
        # wins; fusion-disabled / no-artifact returns None (byte-identical); a set-but-unloadable
        # path raises (fail-closed).
        if frame_resolver is None:
            frame_resolver = build_config_frame_resolver(grasping_cfg)

        # Fail closed on a real cell with no resolver at all. Without one, `require_base_frame_grasp`
        # is never switched on, the grasp stays `frame=camera`, and every single motion comes back
        # `MotionStatus.INVALID_TARGET` (`URRobotArm.move` requires `Frame.BASE`). That is safe, and
        # it is also indistinguishable at the bench from "the cell is broken": 100% rejection, no
        # hint that the cause is a missing calibration artifact. The shipped
        # `config/robot/robot.yaml` enables no fusion and names no artifact path, so this is the
        # default state of a freshly configured real cell. Sim/dummy are exempt: they legitimately
        # pass a live resolver in code (an eye-in-hand cell's TCP-composed resolver cannot be
        # serialized to an artifact).
        vendor_name = str(getattr(getattr(robot_cfg, "vendor", ""), "value", getattr(robot_cfg, "vendor", "")))
        if frame_resolver is None and vendor_name.lower() not in ("sim", "dummy"):
            raise ValueError(
                f"no CAMERA->BASE frame resolver for a {vendor_name!r} cell. Perception reports grasps "
                "in the camera frame; without a resolver they are never transformed into the robot's "
                "base frame, so the driver refuses every motion as INVALID_TARGET, which at the bench "
                "looks like a broken cell, not a missing calibration. Fix it one of two ways: (1) set "
                "grasping.fusion.enabled: true + grasping.fusion.extrinsics_artifact_path to the "
                "artifact your eye-to-hand calibration wrote (run_eth_calibrate writes "
                "logs/calibration/<robot_model>/eth_<camera>.json), or (2) pass frame_resolver= "
                "explicitly, which is what an eye-in-hand cell does because a live TCP-composed "
                "resolver cannot be serialized."
            )

        # Auto-build a viewpoint planner so the commit-gate reobserve relocates the
        # camera to a diverse viewpoint in production (the relocate only fires when a planner is
        # wired). A caller-supplied planner (sim) wins; the three-flag gate (fusion + commit_policy
        # + active_perception_use_fusion) all-off returns None (byte-identical, and the relocate
        # stays inert).
        if viewpoint_planner is None:
            viewpoint_planner = build_config_viewpoint_planner(grasping_cfg, robot_cfg)

        # Underlying runtime pick service (+ fail-closed frame-guard patch).
        runtime = build_runtime(
            robot_cfg,
            calculator=calculator,
            perception=perception,
            viewpoint_planner=viewpoint_planner,
            resolved_max_attempts=resolved_max_attempts,
            profile=profile,
            standoff_mm=standoff_mm,
            retreat_mm=retreat_mm,
            policy=policy,
            frame_resolver=frame_resolver,
            arm=arm,
            gripper=gripper,
        )

        # Orchestrator overlays applied in place.
        apply_orchestrator_overlays(
            runtime, grasping_cfg, resolved_mode=resolved_mode,
            primary_camera_id=primary_camera_id,
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

        service = cls(
            runtime=runtime,
            default_mode=resolved_mode,
            refinement_policy=resolved_refinement_policy,
            refiner=resolved_refiner,
            verification_policy=resolved_verification_policy,
            verifier=resolved_verifier,
            recovery_policy=resolved_recovery_policy,
            recovery_strategy=resolved_recovery_strategy,
            effective_config=effective_config,
            decision_policy=resolved_decision_policy,
            decision_engine=resolved_decision_engine,
            shadow_router=maybe_build_shadow_router(robot_cfg),
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
    ) -> AutonomousGraspReport:
        """Run one autonomous pick attempt and return a typed report.

        Parameters
        ----------
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
        """

        # A cancel that arrived between two picks is honoured here, before any work starts: a caller
        # running a campaign loops on pick(), and refusing to start is the cheapest possible stop.
        # Default ``None`` costs one attribute read and leaves the body below unchanged.
        if self._should_cancel is not None and self._should_cancel():
            return self._cancelled_report(mode=mode)
        # Install a per-attempt LatencyTracker so the decision / ranking
        # / fusion seams can record their spans via
        # ``self._current_latency_tracker``. The tracker is always
        # installed (recording is cheap); the typed ``performance_enabled``
        # knob only gates enforcement and SLO_BREACH event emission.
        # The wall-time captured here covers the full pick() body and is
        # exported as ``attempt_wall_time_s`` on the returned report.
        self._current_latency_tracker: Optional[LatencyTracker] = LatencyTracker()
        # Hand it to the orchestrator so the ranking span is recorded on every path, including the
        # default open-loop pick.
        _orch = getattr(self.runtime, "orchestrator", None)
        if _orch is not None:
            _orch.latency_tracker = self._current_latency_tracker
        # One attempt_id per pick(), shared across latency / shadow / record-logging
        # telemetry. The decision path mints its own ``auto-<uuid>`` deeper in; this pick-scoped id
        # is the fallback stamped when no path set one. Without it every record logged from the
        # default open-loop path collides on the literal "pick".
        pick_attempt_id = f"pick-{uuid.uuid4().hex[:12]}"
        wall_t0_ns = time.monotonic_ns()
        try:
            report = self._run_with_recovery(mode=mode)
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

        * ``fused_view_count``: how many cameras contributed a usable view this pick.
        * ``fusion_evidence_quality``: the mean association score over the pairs that won their
          assignment, i.e. how confidently the cameras agreed an object was the same object.
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

        Stages that never opened a span (e.g. fusion on an ``EASY``-only
        attempt) are absent from the snapshot, so the SLO gate skips
        nulls instead of treating them as passing zeros.
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
        """

        cfg = self.effective_config
        if cfg is None or not cfg.recovery_orchestrator.enabled:
            return SceneRecoveryPolicy(enabled=False)
        allowed = tuple(
            SceneRecoveryAction(a) for a in cfg.recovery_orchestrator.allowed_actions
        )
        budget = {
            SceneRecoveryAction(name): int(count)
            for name, count in cfg.recovery_orchestrator.per_action_budget
        }
        # The envelope, rebuilt from what the operator declared. `SceneRecoveryPolicy` refuses to
        # construct a physical action without one, so a config that declares a physical action and
        # a call site that passes no envelope raise on every pick().
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
        self, *, mode: GraspMode | str | None
    ) -> AutonomousGraspReport:
        """Run one pick attempt, optionally wrapped in the recovery loop.

        When ``recovery_orchestrator.enabled`` is :data:`False` (the default)
        this is byte-identical to a single :meth:`_pick_inner` call. When
        enabled (and the effective mode is in the configured ``apply_modes``),
        the attempt is driven through :func:`run_recovery_loop`; the
        ``uncertainty_recovery_aggressive_threshold`` knob biases the recovery
        action ordering toward perception whenever the most recent attempt's
        fused uncertainty exceeds it.
        """

        policy = self._build_recovery_orchestrator_policy()
        if not policy.enabled:
            return self._pick_inner(mode=mode)

        cfg = self.effective_config
        assert cfg is not None  # guaranteed by policy.enabled above
        effective_mode = (
            resolve_grasp_mode(mode) if mode is not None else self.default_mode
        )
        profile = _profile_for(effective_mode)
        if profile.mode.value not in policy.apply_modes:
            return self._pick_inner(mode=mode)

        threshold = float(cfg.uncertainty.recovery_aggressive_threshold)
        orchestrator = RecoveryOrchestrator(
            dispatcher=RecoveryDispatcher(), strategies={}, bypass_strategies=True
        )
        arm = getattr(self.runtime.orchestrator, "arm", None)

        def _pick_once() -> _RecoveryReportAdapter:
            return _RecoveryReportAdapter(self._pick_inner(mode=mode))

        def _aggressive(adapter: _RecoveryReportAdapter) -> bool:
            # Single source of truth: the canonical gate requires fused_available and
            # fused >= threshold, applied to the report's own UncertaintySnapshot.
            snapshot = getattr(adapter.report, "uncertainty", None)
            return should_bias_recovery_for_uncertainty(
                snapshot, recovery_aggressive_threshold=threshold,
            )

        final, trail = run_recovery_loop(
            pick=_pick_once,
            profile=profile,
            policy=policy,
            orchestrator=orchestrator,
            frame_acquirer=lambda: None,  # the next _pick_inner re-perceives the scene
            arm=arm,
            aggressive_recovery_bias=_aggressive,
        )
        return self._attach_recovery_trail(final.report, trail)

    @staticmethod
    def _attach_recovery_trail(
        report: AutonomousGraspReport, trail: RecoveryTrail
    ) -> AutonomousGraspReport:
        """Fold the recovery-trail summary into a report's telemetry for operator
        visibility. Only called on the opt-in recovery path, so the default path
        stays byte-identical. Values are simple types (str + list[str]) so the
        telemetry audit (which checks required-key presence, not extras) is safe.
        """

        merged = dict(report.telemetry)
        merged["recovery_trail_terminal_reason"] = trail.terminal_reason
        merged["recovery_trail_actions"] = [
            entry.plan_action.value for entry in trail.entries
        ]
        # Carry the per-step recovery trail on the typed report field so the serializer writes it into
        # GraspAttemptRecord.recovery_actions (the per-step action+reward source for offline recovery training).
        return replace(
            report,
            telemetry=merged,
            recovery_actions=recovery_actions_from_trail(trail),
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
        """Bounded re-observation loop driven by :class:`DecisionEngine`.

        Steps:

        1. Acquire a perception frame and compute best candidates via
           the orchestrator's helper, which honours the configured
           :class:`FrameResolver`.
        2. Hand the typed inputs to
           :meth:`DecisionEngine.decide`.
        3. Dispatch:

           * ``GRASP_NOW``: delegate to :meth:`_execute_pick` and
             attach the decision telemetry.
           * ``MOVE_CAMERA``: ask the wired viewpoint planner for the
             next pose, command the arm, bump the counter, loop.
           * ``RECOVER``: return a typed
             ``DECISION_RECOVER_PENDING`` report. No physical recovery
             action is executed here.
           * ``FAIL_CLOSED``: return a typed
             ``DECISION_FAIL_CLOSED`` report with structured
             telemetry. No pick attempt is dispatched.

        The loop terminates at the first non-``MOVE_CAMERA`` decision
        or when the bounded budget is reached: the engine emits
        ``FAIL_CLOSED`` or a permissive ``GRASP_NOW`` at that point, so
        this method needs no explicit ceiling.
        """

        # Engine + policy are narrowed by the caller-side gate.
        assert self.decision_engine is not None
        assert self.decision_policy is not None
        engine = self.decision_engine
        orch = self.runtime.orchestrator
        arm = orch.arm
        is_simulated = bool(arm.capabilities.is_simulated)
        planner = orch.viewpoint_planner

        # Stable per-call attempt id. Used purely for telemetry
        # correlation; the wrapped runtime constructs its own
        # attempt records inside :class:`PickSessionReport`.
        attempt_id = f"auto-{uuid.uuid4().hex[:12]}"
        reobservation_count = 0
        viewpoints_visited: list[Pose] = []

        # Drift + OOD watchdog policy is built once per pick(). When
        # :data:`None` the watchdog is inert: a ``from_components``
        # service carries no effective config.
        watchdog_policy = self._build_watchdog_policy()
        prev_watchdog_report: Optional[WatchdogReport] = None

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

        while True:
            # Pre-decide watchdog gate. Evaluates over the rolling
            # history accumulated from prior attempts. Returns
            # :data:`None` when the watchdog has no policy wired. The
            # gate fires only when ``report.enforced`` is :data:`True`,
            # which already encodes the hardware lock (real hardware +
            # canary/active + mode in policy.block_modes) and the
            # ``EASY`` exclusion, since schema validation rejects
            # ``EASY`` from ``block_modes``.
            (
                watchdog_report,
                prev_watchdog_report,
                terminal_report,
            ) = self._handle_watchdog_pretick(
                policy=watchdog_policy,
                effective_mode=effective_mode,
                is_simulated=is_simulated,
                prev_watchdog_report=prev_watchdog_report,
                attempt_id=attempt_id,
                reobservation_count=reobservation_count,
                profile=profile,
            )
            if terminal_report is not None:
                return terminal_report

            frame = orch.perception.acquire()
            grasp_result: Optional[Any] = self._rank_grasp_for_decision(frame)

            uncertainty_snapshot = self._build_uncertainty_snapshot_for_tick(
                frame=frame,
                grasp_result=grasp_result,
                calibration=calibration,
                uncertainty_active=uncertainty_active,
                fail_closed_threshold=fail_closed_threshold,
                disagreement_threshold=disagreement_threshold,
                attempt_id=attempt_id,
                reobservation_count=reobservation_count,
            )

            decision = self._decide_once(
                engine=engine,
                grasp_result=grasp_result,
                effective_mode=effective_mode,
                attempt_id=attempt_id,
                reobservation_count=reobservation_count,
                is_simulated=is_simulated,
                viewpoint_planner_available=planner is not None,
                uncertainty_snapshot=uncertainty_snapshot,
                uncertainty_active=uncertainty_active,
                ranking_penalty_weight=ranking_penalty_weight,
            )
            # Record one watchdog sample per decision tick (keeps the
            # rolling history fresh for the next pick()); kept at the call-site so
            # the primary decide + the re-decide each record exactly once.
            self._record_watchdog_sample(
                self._build_watchdog_sample_from_decision(decision)
            )

            if decision.action is DecisionAction.GRASP_NOW:
                inner = self._execute_pick(
                    effective_mode=effective_mode,
                    profile=profile,
                    decision=decision,
                    watchdog_report=watchdog_report,
                )
                return inner

            if decision.action is DecisionAction.MOVE_CAMERA:
                # The engine emits MOVE_CAMERA only when a planner is
                # wired and the budget allows another step; this still
                # defends against a planner that returns :data:`None`
                # (suggestions exhausted).
                assert planner is not None  # narrowed by engine
                next_vp = planner.next_viewpoint(
                    current_tcp=arm.get_tcp_pose(),
                    history=tuple(viewpoints_visited),
                )
                if next_vp is None:
                    # Re-decide treating the planner as unavailable so
                    # the engine selects the right FAIL_CLOSED reason
                    # code and structured telemetry.
                    final = self._decide_once(
                        engine=engine,
                        grasp_result=grasp_result,
                        effective_mode=effective_mode,
                        attempt_id=attempt_id,
                        reobservation_count=reobservation_count,
                        is_simulated=is_simulated,
                        viewpoint_planner_available=False,
                        uncertainty_snapshot=uncertainty_snapshot,
                        uncertainty_active=uncertainty_active,
                        ranking_penalty_weight=ranking_penalty_weight,
                    )
                    self._record_watchdog_sample(
                        self._build_watchdog_sample_from_decision(final)
                    )
                    return self._terminal_decision_report(
                        effective_mode=effective_mode,
                        profile=profile,
                        decision=final,
                        watchdog_report=watchdog_report,
                    )
                arm.move(next_vp)
                viewpoints_visited.append(next_vp)
                reobservation_count += 1
                continue

            # RECOVER or FAIL_CLOSED: terminate without dispatching.
            return self._terminal_decision_report(
                effective_mode=effective_mode,
                profile=profile,
                decision=decision,
                watchdog_report=watchdog_report,
            )

    # Decision-loop helpers

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
        reobservation_count: int,
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
            print(
                "[U8] uncertainty_no_signal: "
                f"attempt_id={attempt_id} "
                f"reobservation_count={reobservation_count}"
            )
        return uncertainty_snapshot

    def _decide_once(
        self,
        *,
        engine: DecisionEngine,
        grasp_result: Optional[Any],
        effective_mode: GraspMode,
        attempt_id: str,
        reobservation_count: int,
        is_simulated: bool,
        viewpoint_planner_available: bool,
        uncertainty_snapshot: Optional[UncertaintySnapshot],
        uncertainty_active: bool,
        ranking_penalty_weight: float,
    ) -> DecisionReport:
        """One engine.decide(), inside the decision latency span when a tracker is
        installed. Shared by the primary decide and by the MOVE_CAMERA
        planner-exhausted re-decide, which differ only in
        ``viewpoint_planner_available``. The per-tick watchdog sample is recorded at
        each call site, so every decide records exactly once."""
        tracker = self._current_latency_tracker
        if tracker is not None:
            with tracker.span(LatencyStage.DECISION):
                return engine.decide(
                    grasp_result=grasp_result,
                    mode=effective_mode.value,
                    attempt_id=attempt_id,
                    reobservation_count=reobservation_count,
                    is_simulated=is_simulated,
                    viewpoint_planner_available=viewpoint_planner_available,
                    uncertainty_snapshot=uncertainty_snapshot,
                    uncertainty_active=uncertainty_active,
                    ranking_penalty_weight=ranking_penalty_weight,
                )
        return engine.decide(
            grasp_result=grasp_result,
            mode=effective_mode.value,
            attempt_id=attempt_id,
            reobservation_count=reobservation_count,
            is_simulated=is_simulated,
            viewpoint_planner_available=viewpoint_planner_available,
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
        prev_watchdog_report: Optional[WatchdogReport],
        attempt_id: str,
        reobservation_count: int,
        profile: GraspBehaviorProfile,
    ) -> tuple[
        Optional[WatchdogReport],
        Optional[WatchdogReport],
        Optional[AutonomousGraspReport],
    ]:
        """The pre-decide watchdog gate. Returns ``(watchdog_report, new_prev_watchdog_report,
        terminal_report)``: ``watchdog_report`` is threaded downstream, since the GRASP_NOW
        dispatch and the terminal reports carry it; ``new_prev_watchdog_report`` carries the
        transition state across MOVE_CAMERA re-observations; ``terminal_report`` is non-None
        only when an enforced BLOCK_AUTO fires, and that branch records the fail-closure
        sample and builds the terminal report."""
        watchdog_report = self._evaluate_watchdog_pretick(
            policy=policy,
            effective_mode=effective_mode,
            is_simulated=is_simulated,
        )
        if watchdog_report is not None:
            self._fire_watchdog_transitions(
                prev=prev_watchdog_report,
                curr=watchdog_report,
                attempt_id=attempt_id,
                reobservation_count=reobservation_count,
                effective_mode=effective_mode,
            )
            prev_watchdog_report = watchdog_report
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
                reobservation_count=reobservation_count,
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
            return watchdog_report, prev_watchdog_report, terminal
        return watchdog_report, prev_watchdog_report, None

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
        reobservation_count: int,
    ) -> DecisionReport:
        return self._watchdog.synthesize_block_decision(
            report=report,
            effective_mode=effective_mode,
            attempt_id=attempt_id,
            reobservation_count=reobservation_count,
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
        prev: Optional[WatchdogReport],
        curr: WatchdogReport,
        attempt_id: str,
        reobservation_count: int,
        effective_mode: GraspMode,
    ) -> None:
        self._watchdog.fire_transitions(
            event_listener=self.watchdog_event_listener,
            prev=prev,
            curr=curr,
            attempt_id=attempt_id,
            reobservation_count=reobservation_count,
            effective_mode=effective_mode,
        )

    def _base_telemetry(self, profile: GraspBehaviorProfile) -> dict[str, Any]:
        """The shared leading telemetry key (``sampling_mode``) for every AutonomousGraspReport
        this service builds: one source, so the projection cannot drift across the legacy,
        decision and refine paths. Callers add their path-specific keys after, so the
        insertion order keeps ``sampling_mode`` first."""
        return {"sampling_mode": str(profile.sampling_mode)}

    def _terminal_decision_report(
        self,
        *,
        effective_mode: GraspMode,
        profile: GraspBehaviorProfile,
        decision: DecisionReport,
        watchdog_report: Optional[WatchdogReport] = None,
    ) -> AutonomousGraspReport:
        """Wrap a non-executing decision into an :class:`AutonomousGraspReport`."""

        if decision.action is DecisionAction.RECOVER:
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
        """Execute the refinement gate, then run_attempt().

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

        report = self._execute_pick_body(
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

    def _readiness_gate(
        self,
        *,
        effective_mode: GraspMode,
        profile: GraspBehaviorProfile,
        phase_gate: str,
        required: bool,
        ready: bool,
        reason: str,
        hint: str,
    ) -> Optional[AutonomousGraspReport]:
        """The shared refinement and verification readiness gate. When the active profile
        requires a capability whose wiring is absent (``not ready``), it returns a
        MODE_NOT_AVAILABLE report whose telemetry carries reason, refinement_required,
        verification_required, phase_gate and hint. Returns ``None`` to continue.
        The ``hint`` text is frozen by the honesty contract; keep it verbatim."""
        if not (required and not ready):
            return None
        return AutonomousGraspReport(
            outcome=AutonomousGraspOutcome.MODE_NOT_AVAILABLE,
            mode=effective_mode,
            profile=profile,
            pick_report=None,
            telemetry={
                "reason": reason,
                "refinement_required": profile.refinement_enabled,
                "verification_required": profile.verification_enabled,
                "phase_gate": phase_gate,
                "hint": hint,
            },
            effective_config=self.effective_config,
        )

    def _execute_pick_body(
        self,
        *,
        effective_mode: GraspMode,
        profile: GraspBehaviorProfile,
    ) -> AutonomousGraspReport:
        # Refinement and verification are operator opt-in via the ``refinement_policy``/
        # ``refiner`` and ``verification_policy``/``verifier`` constructor parameters. When
        # the active profile requires a capability whose wiring is absent, the service
        # returns MODE_NOT_AVAILABLE instead of downgrading to an open-loop or close-only
        # run. Both gates share _readiness_gate.
        refine_ready = (
            self.refiner is not None
            and self.refinement_policy is not None
            and self.refinement_policy.enabled
        )
        gate = self._readiness_gate(
            effective_mode=effective_mode,
            profile=profile,
            phase_gate="S3_refinement",
            required=profile.refinement_enabled,
            ready=refine_ready,
            reason="mode_requires_refinement_but_no_refiner_wired",
            hint=(
                "pass refinement_policy=RefinementPolicy(enabled=True) "
                "and refiner=DefaultPreGraspRefiner(policy=...) to "
                "from_components / from_robot_config."
            ),
        )
        if gate is not None:
            return gate
        verify_ready = (
            self.verifier is not None
            and self.verification_policy is not None
            and self.verification_policy.enabled
        )
        gate = self._readiness_gate(
            effective_mode=effective_mode,
            profile=profile,
            phase_gate="S4_verification",
            required=profile.verification_enabled,
            ready=verify_ready,
            reason="mode_requires_verification_but_no_verifier_wired",
            hint=(
                "pass verification_policy=GraspVerificationPolicy("
                "enabled=True) and a GraspVerifier (e.g. "
                "CompositeGraspVerifier of ObjectDetectingGripperVerifier "
                "+ WidthDeltaGripperVerifier + "
                "VisionTargetDisplacementVerifier) to "
                "from_components / from_robot_config."
            ),
        )
        if gate is not None:
            return gate
        if profile.refinement_enabled and refine_ready:
            return self._pick_with_refinement(
                effective_mode=effective_mode,
                profile=profile,
            )
        return self._run_legacy_attempt(effective_mode=effective_mode, profile=profile)

    def _run_legacy_attempt(
        self,
        *,
        effective_mode: GraspMode,
        profile: GraspBehaviorProfile,
    ) -> AutonomousGraspReport:
        """The open-loop attempt: the fusion span, ``runtime.run_attempt()``, the
        mapping from PickOutcome to AutonomousGraspOutcome, and the report build."""
        # Fusion stage span. The run_attempt() call is timed and the span
        # is committed only when the orchestrator performed multi-view
        # fusion on this attempt (``views_attempted > 0`` on the resulting
        # FusionTelemetry). It is a coarse upper bound on fusion compute,
        # not a producer-side timer inside SceneFusion. The snapshot omits
        # ``fusion_latency_ms`` entirely when fusion did not run, for
        # example ``EASY`` mode and single-view paths.
        _t0_attempt_ns = time.monotonic_ns()
        pick_report = self.runtime.run_attempt()
        _attempt_elapsed_ms = (time.monotonic_ns() - _t0_attempt_ns) / 1e6
        tracker = self._current_latency_tracker
        if tracker is not None:
            fusion_tm = getattr(pick_report, "fusion_telemetry", None)
            if (
                fusion_tm is not None
                and getattr(fusion_tm, "views_attempted", 0) > 0
            ):
                tracker.record(LatencyStage.FUSION, _attempt_elapsed_ms)
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
            fusion_telemetry=pick_report.fusion_telemetry,
            commit_decision=pick_report.commit_decision,
        )

    # Two-scan refinement orchestration

    def _pick_with_refinement(
        self,
        *,
        effective_mode: GraspMode,
        profile: GraspBehaviorProfile,
    ) -> AutonomousGraspReport:
        """Drive the two-scan workflow for refinement-capable modes.

        Steps:

        1. Acquire an initial :class:`PerceptionFrame`.
        2. Compute the best initial grasp via the orchestrator's own
           ``_best_result_over_segmentations`` helper, which shares the
           :class:`FrameResolver` plumbing and the segmentation-wise
           neighbour-mask bookkeeping with the legacy path.
        3. Build a :class:`TargetIdentity` from the segmentation the
           winning candidate came from.
        4. Drive the arm to a standoff pose ``standoff_mm`` behind the
           candidate along its approach direction, which is above the
           candidate only for a top-down approach.
        5. Acquire a refined :class:`PerceptionFrame`.
        6. Run :attr:`refiner` and take its :class:`RefinementReport`.
        7. On :attr:`RefinementOutcome.ACCEPTED`, hand the refined
           grasp to the orchestrator's :class:`GraspExecutionPolicy`
           for the final motion, then run post-grasp verification.
        8. Map the result to an :class:`AutonomousGraspReport`.

        Scene recovery is not driven from here. When the profile demands
        a capability this method does not run, that is surfaced in
        telemetry rather than reported as success.
        """

        # Local aliases to keep the body readable.
        assert self.refiner is not None  # narrowed by caller
        assert self.refinement_policy is not None  # narrowed by caller
        refiner = self.refiner
        policy_cfg = self.refinement_policy
        orch = self.runtime.orchestrator

        snapshot_profile = replace(profile)

        # Full ranking consistency: the refine path applies the same ranking pipeline
        # (annotate, then blend, then uncertainty re-rank) before consuming
        # ``initial_result.best``, so both pick paths choose the same grasp. These two
        # hold the resulting telemetry for the ``_report`` closure; typed ``Any``
        # because they pass straight through into the typed report fields.
        refine_blend_tel: Any = None
        refine_rerank_tel: Any = None

        def _report(
            outcome: AutonomousGraspOutcome,
            *,
            pick_report: Optional[PickSessionReport] = None,
            telemetry: Optional[dict] = None,
        ) -> AutonomousGraspReport:
            base_telemetry = self._base_telemetry(profile)
            # The label: "hold" when the refiner re-validates on the initial frame
            # (static camera, no second viewpoint), "two_scan" for the standoff and
            # re-perceive path.
            base_telemetry["refinement_mode"] = (
                "two_scan"
                if (self.refinement_policy is None or self.refinement_policy.reperceive)
                else "hold"
            )
            base_telemetry["verification_enabled"] = (
                self.verification_policy is not None
                and self.verification_policy.enabled
            )
            if telemetry:
                base_telemetry.update(telemetry)
            return AutonomousGraspReport(
                outcome=outcome,
                mode=effective_mode,
                profile=snapshot_profile,
                pick_report=pick_report,
                telemetry=base_telemetry,
                effective_config=self.effective_config,
                shadow_success_telemetry=(
                    pick_report.shadow_success_telemetry
                    if pick_report is not None
                    else None
                ),
                ranking_blend_telemetry=(
                    pick_report.ranking_blend_telemetry
                    if pick_report is not None
                    else refine_blend_tel
                ),
                uncertainty_rerank_telemetry=(
                    pick_report.uncertainty_rerank_telemetry
                    if pick_report is not None
                    else refine_rerank_tel
                ),
                fusion_telemetry=(
                    pick_report.fusion_telemetry
                    if pick_report is not None
                    else None
                ),
                commit_decision=(
                    pick_report.commit_decision
                    if pick_report is not None
                    else None
                ),
            )

        # Step 1: acquire initial frame.
        initial_frame = orch.perception.acquire()
        if not initial_frame.segmentations:
            return _report(
                AutonomousGraspOutcome.NO_TARGET,
                telemetry={"stage": "initial_frame", "n_segmentations": 0},
            )

        # Step 2: best initial result plus which segmentation it came from. The
        # ranking span is opened inside the orchestrator, once, for every path, so
        # nothing is recorded here.
        initial_result, target_index, _ord_decision = (
            orch._best_result_over_segmentations(initial_frame)
        )
        # Apply annotate, blend and uncertainty re-rank to the refine path's
        # candidates before ``.best`` is consumed, so it picks the same grasp the
        # normal path would. refine_blend_tel and refine_rerank_tel must be assigned
        # here, before the validity gate: the _report closure reads them on every
        # subsequent early return.
        initial_result, refine_blend_tel, refine_rerank_tel = (
            self._apply_refine_ranking_consistency(orch, initial_result)
        )
        if (
            initial_result is None
            or initial_result.best is None
            or target_index is None
        ):
            return _report(
                AutonomousGraspOutcome.NO_VALID_GRASP,
                telemetry={
                    "stage": "initial_compute",
                    "reasons": tuple(
                        str(r)
                        for r in (initial_result.reasons if initial_result else ())
                    ),
                },
            )

        initial_grasp = initial_result.best
        target_seg = initial_frame.segmentations[target_index]

        # Step 3: build a target fingerprint for the tracker. The initial
        # camera_to_base is resolved here, while the arm is still at the initial pose
        # (an eye-in-hand camera moves at standoff), so the identity can carry the
        # target's 3D base-frame centroid. The WorldSpacePoseTracker then matches by
        # 3D pose, which is viewpoint-invariant; the default IoU tracker ignores the
        # extra field (byte-identical).
        initial_camera_to_base = None
        if orch.frame_resolver is not None:
            initial_camera_to_base = orch.frame_resolver.camera_to_base_for_frame(
                initial_frame, arm=orch.arm
            )
        try:
            identity = target_identity_from_segmentation(
                target_seg,
                depth_map=initial_frame.depth_map,
                intrinsics=initial_frame.intrinsics,
                camera_to_base=initial_camera_to_base,
            )
        except ValueError as exc:
            # Defensive: the calculator just produced a candidate on
            # this mask, so it should not be empty. If it is, treat
            # it as a perception bug rather than crashing the service.
            return _report(
                AutonomousGraspOutcome.NO_TARGET,
                telemetry={
                    "stage": "target_identity",
                    "error": str(exc),
                },
            )

        # A static eye-to-hand camera (the dense overhead) cannot yield a second viewpoint,
        # so the two-scan standoff and re-perceive is a no-op there and it degrades the
        # attempt: the standoff puts the arm above the object, which either occludes the
        # overhead camera (high standoff) or times out the move from park to standoff (low
        # standoff). With ``reperceive=False`` the refiner runs as a hold on the initial
        # frame: no arm move, no re-perceive; it re-validates the grasp and keeps the full
        # perceive, refine, verify, recover loop. ``reperceive=True`` (default) is the
        # two-scan path.
        if policy_cfg.reperceive:
            # Step 4: drive to standoff behind the initial candidate.
            standoff_pose, standoff_err = self._build_standoff_pose(
                initial_grasp,
                policy_cfg.standoff_mm,
            )
            if standoff_pose is None:
                return _report(
                    AutonomousGraspOutcome.EXECUTION_FAILED,
                    telemetry={
                        "stage": "standoff_build",
                        "error": standoff_err or "unknown",
                    },
                )
            move_result = self._drive_arm_to(standoff_pose)
            if move_result is False:
                return _report(
                    AutonomousGraspOutcome.EXECUTION_FAILED,
                    telemetry={
                        "stage": "standoff_move",
                        "error": "arm.move returned non-executed status",
                    },
                )

            # Step 5: acquire refined frame.
            refined_frame = orch.perception.acquire()
            if not refined_frame.segmentations:
                return _report(
                    AutonomousGraspOutcome.TARGET_LOST_DURING_REFINE,
                    telemetry={"stage": "refined_frame", "n_segmentations": 0},
                )

            # Resolve camera_to_base for the refined frame (eye-in-hand
            # camera has moved with the TCP).
            camera_to_base = None
            if orch.frame_resolver is not None:
                camera_to_base = orch.frame_resolver.camera_to_base_for_frame(
                    refined_frame, arm=orch.arm
                )
        else:
            # Hold: no arm move, no re-perceive; refine-validate on the initial frame (static-camera safe).
            refined_frame = initial_frame
            camera_to_base = initial_camera_to_base

        # Step 6: run the refiner.
        refinement: RefinementReport = refiner.refine(
            initial_grasp=initial_grasp,
            initial_frame=initial_frame,
            refined_frame=refined_frame,
            target_identity=identity,
            calculator=orch.calculator,
            camera_to_base=camera_to_base,
        )

        if refinement.outcome is RefinementOutcome.TARGET_LOST:
            return _report(
                AutonomousGraspOutcome.TARGET_LOST_DURING_REFINE,
                telemetry={
                    "stage": "refiner",
                    "refinement_telemetry": dict(refinement.telemetry),
                },
            )
        if refinement.outcome is RefinementOutcome.DIVERGED:
            return _report(
                AutonomousGraspOutcome.REFINEMENT_DIVERGED,
                telemetry={
                    "stage": "refiner",
                    "position_delta_mm": refinement.position_delta_mm,
                    "orientation_delta_deg": refinement.orientation_delta_deg,
                    "grip_width_delta_mm": refinement.grip_width_delta_mm,
                    "refinement_telemetry": dict(refinement.telemetry),
                },
            )
        if refinement.outcome is RefinementOutcome.NO_GRASP:
            return _report(
                AutonomousGraspOutcome.NO_VALID_GRASP,
                telemetry={
                    "stage": "refiner",
                    "refinement_telemetry": dict(refinement.telemetry),
                },
            )
        if refinement.outcome is not RefinementOutcome.ACCEPTED:
            # Future-proofing: any unrecognised outcome is treated as
            # a refinement failure rather than silently succeeding.
            return _report(
                AutonomousGraspOutcome.REFINEMENT_FAILED,
                telemetry={
                    "stage": "refiner",
                    "unrecognised_outcome": str(refinement.outcome),
                },
            )

        assert refinement.refined_grasp is not None  # ACCEPTED invariant

        # Step 7: hand the refined grasp to the execution policy. Going
        # through ``runtime.run_attempt`` would restart perception and
        # re-pick a candidate, discarding the refinement.
        exec_policy = orch.policy
        if exec_policy is None:
            return _report(
                AutonomousGraspOutcome.EXECUTION_FAILED,
                telemetry={
                    "stage": "execution",
                    "error": "orchestrator has no GraspExecutionPolicy wired",
                },
            )
        # Snapshot the pre-close jaw width so verifiers can reason about
        # how far the jaws travelled. Best-effort: a gripper that does
        # not expose ``get_width_mm`` yields :data:`None` and the
        # width-delta verifier falls back to ``INCONCLUSIVE``.
        gripper = getattr(exec_policy, "gripper", None)
        pre_close_width_mm = _safe_gripper_width(gripper)
        policy_report = exec_policy.execute(refinement.refined_grasp)

        outcome = _policy_outcome_to_autonomous(policy_report.outcome)
        executed_telemetry: dict[str, Any] = {
            "stage": "executed",
            "policy_outcome": str(policy_report.outcome),
            "position_delta_mm": refinement.position_delta_mm,
            "orientation_delta_deg": refinement.orientation_delta_deg,
            "grip_width_delta_mm": refinement.grip_width_delta_mm,
            "match_iou": refinement.match_iou,
        }

        # Verification runs only when the execution policy executed the
        # grasp. Any earlier failure already carries its own outcome;
        # re-running a verifier would paper over the real error.
        outcome = self._run_post_grasp_verification(
            outcome=outcome,
            executed_telemetry=executed_telemetry,
            refined_grasp=refinement.refined_grasp,
            gripper=gripper,
            pre_close_width_mm=pre_close_width_mm,
            exec_policy=exec_policy,
            identity=identity,
            orch=orch,
        )

        return _report(outcome, telemetry=executed_telemetry)

    # Refinement-loop helpers

    def _apply_refine_ranking_consistency(
        self, orch: Any, initial_result: Any
    ) -> tuple[Any, Any, Any]:
        """Full ranking consistency for the refine path: annotate, then blend, then
        uncertainty re-rank, applied to the candidates before ``.best`` is consumed so
        both pick paths choose the same grasp. Each stage is a no-op unless its config is
        wired; it changes candidate metadata and candidate order, never
        ``GraspPoint.score``. Returns ``(initial_result, blend_tel, rerank_tel)``: the
        caller assigns the telemetry before the validity gate, since the _report closure
        reads it on every subsequent early return."""
        refine_blend_tel: Any = None
        refine_rerank_tel: Any = None
        _ctx = orch.shadow_success_context
        if _ctx is not None and initial_result is not None and initial_result.candidates:
            from src.robot.grasping.scoring.success_probability import (
                annotate_grasp_points_with_shadow_probability,
                maybe_blend_rerank_candidates,
                maybe_uncertainty_rerank_candidates,
            )

            annotate_grasp_points_with_shadow_probability(initial_result.candidates, ctx=_ctx)
            _blend_cands, refine_blend_tel = maybe_blend_rerank_candidates(
                initial_result.candidates, ctx=_ctx, config=orch.ranking_blend_config
            )
            if refine_blend_tel is not None and refine_blend_tel.applied:
                initial_result = replace(
                    initial_result,
                    candidates=_blend_cands,
                    top_score=float(_blend_cands[0].score) if _blend_cands else 0.0,
                )
            _rerank_cands, refine_rerank_tel = maybe_uncertainty_rerank_candidates(
                initial_result.candidates, ctx=_ctx, config=orch.uncertainty_rerank_config
            )
            if refine_rerank_tel is not None and refine_rerank_tel.applied:
                initial_result = replace(
                    initial_result,
                    candidates=_rerank_cands,
                    top_score=float(_rerank_cands[0].score) if _rerank_cands else 0.0,
                )
        return initial_result, refine_blend_tel, refine_rerank_tel

    def _run_post_grasp_verification(
        self,
        *,
        outcome: AutonomousGraspOutcome,
        executed_telemetry: dict[str, Any],
        refined_grasp: Any,
        gripper: Any,
        pre_close_width_mm: Optional[float],
        exec_policy: Any,
        identity: Any,
        orch: Any,
    ) -> AutonomousGraspOutcome:
        """Post-grasp verification. Guard first: it runs only when the policy executed the
        grasp (outcome ``SUCCEEDED``), a verifier is wired and the policy is enabled;
        otherwise it returns ``outcome`` untouched and performs no post-lift acquire.
        Mutates ``executed_telemetry`` in place with the verification result and returns
        the outcome, which may be VERIFICATION_FAILED."""
        if not (
            outcome is AutonomousGraspOutcome.SUCCEEDED
            and self.verifier is not None
            and self.verification_policy is not None
            and self.verification_policy.enabled
        ):
            return outcome
        assert self.verifier is not None
        assert self.verification_policy is not None
        post_close_width_mm = _safe_gripper_width(gripper)
        commanded_close_width_mm: Optional[float] = None
        if gripper is not None:
            try:
                commanded_close_width_mm = exec_policy._resolve_close_width(
                    refined_grasp
                )
            except Exception:  # pragma: no cover (defensive)
                commanded_close_width_mm = None
        post_lift_frame = None
        if self.verification_policy.post_lift_vision_check:
            try:
                post_lift_frame = orch.perception.acquire()
            except Exception as exc:  # noqa: BLE001 (keep verifier total)
                executed_telemetry["post_lift_acquire_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        context = GraspVerificationContext(
            grasp=refined_grasp,
            policy=self.verification_policy,
            gripper=gripper,
            pre_close_width_mm=pre_close_width_mm,
            post_close_width_mm=post_close_width_mm,
            commanded_close_width_mm=commanded_close_width_mm,
            post_lift_frame=post_lift_frame,
            target_identity=identity,
        )
        verification: GraspVerificationReport = self.verifier.verify(context)
        executed_telemetry["verification_outcome"] = str(verification.outcome)
        executed_telemetry["verification_reason"] = verification.reason
        executed_telemetry["verification_telemetry"] = dict(verification.telemetry)
        if verification.outcome is VerificationOutcome.FAILED or (
            verification.outcome is VerificationOutcome.INCONCLUSIVE
            and self.verification_policy.fail_closed
        ):
            return AutonomousGraspOutcome.VERIFICATION_FAILED
        return outcome

    # Motion helpers

    def _build_standoff_pose(
        self,
        grasp: GraspPoint,
        standoff_mm: float,
    ) -> tuple[Optional[Pose], Optional[str]]:
        """Build the standoff :class:`Pose` for the refinement scan.

        The standoff sits ``standoff_mm`` behind the grasp along its
        approach direction, so the gripper is backed off with the
        eye-in-hand camera roughly above the target. Orientation is
        derived from the grasp's :attr:`approach` and :attr:`axis`
        vectors, matching :class:`GraspExecutionPolicy`'s
        ``_build_waypoints`` so the final motion does not re-orient the
        wrist.

        Returns ``(pose, None)`` on success and ``(None, error_msg)``
        on failure, so the caller can surface the error in telemetry
        instead of letting an exception reach :meth:`pick`.
        """

        try:
            quat = _grasp_point_to_quaternion(grasp)
            approach_unit = np.asarray(grasp.approach, dtype=np.float64)
            target = np.asarray(grasp.position, dtype=np.float64)
            standoff_position = target - approach_unit * float(standoff_mm)
            frame = Frame(grasp.frame.value)
            return (
                Pose(
                    position_mm=standoff_position,
                    quaternion_xyzw=quat,
                    frame=frame,
                    label="refinement_standoff",
                ),
                None,
            )
        except Exception as exc:  # pragma: no cover (defensive)
            return None, f"{type(exc).__name__}: {exc}"

    def _drive_arm_to(self, pose: Pose) -> bool:
        """Drive the wrapped arm to ``pose`` for the refinement scan.

        Prefers the typed :meth:`RobotArm.move` surface and treats any
        :class:`MotionResult` whose status is not ``EXECUTED`` as a
        failure, including :class:`SafetyPreflight` rejections. Falls
        back to the :meth:`RobotArm.move_to` bool-or-raise surface for
        arms that do not expose :meth:`RobotArm.move`.
        """

        arm = self.runtime.orchestrator.arm
        typed_move = getattr(arm, "move", None)
        if callable(typed_move):
            result = typed_move(pose)
            status = getattr(result, "status", None)
            if status is None:
                return False
            # Identity check (not a str-match): MotionStatus is a StrEnum whose value renders as "executed".
            return status is MotionStatus.EXECUTED
        legacy = getattr(arm, "move_to", None)
        if not callable(legacy):
            return False
        try:
            legacy(pose)
        except Exception:  # pragma: no cover (defensive)
            return False
        return True


# Policy outcome to autonomous outcome mapping


_POLICY_TO_AUTONOMOUS: Mapping[PolicyOutcome, AutonomousGraspOutcome] = {
    PolicyOutcome.EXECUTED: AutonomousGraspOutcome.SUCCEEDED,
    PolicyOutcome.MOTION_FAILED: AutonomousGraspOutcome.EXECUTION_FAILED,
    PolicyOutcome.OBJECT_NOT_DETECTED: AutonomousGraspOutcome.VERIFICATION_FAILED,
    PolicyOutcome.CAMERA_FRAME_REJECTED: AutonomousGraspOutcome.MISSING_CAMERA_FRAME,
}


def _policy_outcome_to_autonomous(outcome: PolicyOutcome) -> AutonomousGraspOutcome:
    """Map a :class:`PolicyOutcome` to an :class:`AutonomousGraspOutcome`.

    Anything not explicitly listed is treated as
    :attr:`AutonomousGraspOutcome.EXECUTION_FAILED` so a new low-level
    :class:`PolicyOutcome` value never silently maps to ``SUCCEEDED``.
    """

    return _POLICY_TO_AUTONOMOUS.get(
        outcome, AutonomousGraspOutcome.EXECUTION_FAILED
    )


# Gripper helpers


def _safe_gripper_width(gripper: Optional[Any]) -> Optional[float]:
    """Best-effort read of :meth:`Gripper.get_width_mm`.

    Returns :data:`None` when the gripper is absent, the method does
    not exist, or the call raises, so the caller can surface the
    failure as :attr:`VerificationOutcome.INCONCLUSIVE` instead of
    crashing the service.
    """

    if gripper is None:
        return None
    getter = getattr(gripper, "get_width_mm", None)
    if not callable(getter):
        return None
    try:
        return float(getter())
    except Exception:  # noqa: BLE001 (defensive)
        return None
