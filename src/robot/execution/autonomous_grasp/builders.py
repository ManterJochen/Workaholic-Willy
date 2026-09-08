"""Per-phase builder helpers for ``AutonomousGraspService.from_robot_config``.

Each helper is a pure function of its config inputs.
:func:`apply_orchestrator_overlays` is the exception: it performs one
well-scoped mutation of the runtime's orchestrator. A disabled sub-block, or a
mode outside ``apply_modes``, leaves the corresponding slot at its off default;
a misconfigured block raises fail-closed rather than degrading.

These helpers import their building blocks from the grasping and runtime layers
and from the package's own :mod:`.config` value objects. The dependency is
one-directional: nothing here imports the service.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, cast

from src.robot.constants import GRASP_BUILDERS_LOG_FILE, create_robot_logger
from src.robot.execution.runtime_pick import RuntimePickService
from src.robot.grasping.collision import (
    GripperGeometryStrategy,
    ParallelJawGripperModel,
    SuctionCupGripperModel,
)
from src.robot.grasping.decision import DecisionEngine, DecisionPolicy
from src.robot.grasping.multiview.fusion import FusionConfig, SceneFusion
from src.robot.grasping.loop.pick_loop import CommitPolicy
from src.robot.grasping.scoring.corridor import CorridorAnalysisConfig
from src.robot.grasping.scoring.feasibility_score import FeasibilityScoreConfig
from src.robot.grasping.loop.target_selector import (
    BlockerGraphConfig,
    TargetOrderingConfig,
)
from src.robot.grasping.recovery.policy import (
    FixtureEnvelope,
    SceneRecoveryAction,
    SceneRecoveryPolicy,
    SceneRecoveryStrategy,
)
from src.robot.grasping.closed_loop.refinement import PreGraspRefiner, RefinementPolicy
from src.robot.grasping.scoring.success_probability import (
    RankingBlendConfig,
    UncertaintyRerankConfig,
    try_load_shadow_success_context,
)
from src.robot.grasping.closed_loop.verification import (
    GraspVerificationPolicy,
    GraspVerifier,
)

from .config import (
    EffectiveCorridorConfig,
    EffectiveDecisionConfig,
    EffectiveFeasibilityConfig,
    EffectiveGraspingConfig,
    EffectiveOrderingConfig,
    EffectivePerformanceConfig,
    EffectiveRecoveryOrchestratorConfig,
    EffectiveUncertaintyConfig,
    EffectiveWatchdogConfig,
)

if TYPE_CHECKING:
    from src.config.schema.robot import RobotConfig, RobotGraspingConfig
    from src.config.schema.robot.grasping_schema import GraspingGripperGeometryConfig
    from src.robot.core import Gripper, RobotArm
    from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
    from src.robot.grasping.motion.frame_resolver import FrameResolver
    from src.robot.grasping.generation.calculator import GraspCalculator
    from src.robot.grasping.loop.pick_loop import (
        PerceptionSource,
        ViewpointPlanner,
    )

    from .config import GraspBehaviorProfile, GraspMode


# The default pick path is open-loop and nearly every advanced ``grasping.*`` block ships disabled,
# so what a boot actually wired is not visible from the config tree alone. These helpers run once
# per service build and are the only code that knows, so they write the answer to this log.
logger = create_robot_logger("AutonomousGraspBuilders", GRASP_BUILDERS_LOG_FILE)

#: Orchestrator slots that an overlay fills, in the order ``apply_orchestrator_overlays`` sets them.
#: Logged as one aggregate line rather than one line per slot.
_OVERLAY_SLOTS: tuple[str, ...] = (
    "shadow_success_context",
    "ranking_blend_config",
    "uncertainty_rerank_config",
    "scene_fusion",
    "fusion_geometry_config",
    "commit_policy",
    "corridor_config",
    "feasibility_config",
    "target_ordering",
    "approach_path_policy",
)


def build_gripper_geometry(
    config: GraspingGripperGeometryConfig,
) -> GripperGeometryStrategy:
    """Map the ``grasping.gripper_geometry`` config block onto a collision envelope for the calculator.

    Pass the result as ``GraspCalculator(gripper_model=...)`` so grasp candidates are filtered
    against the real end-effector shape (parallel-jaw fingers or a suction cup). The default block
    (``kind: parallel_jaw`` with the shipped dimensions) produces the same envelope as
    ``ParallelJawGripperModel()`` built with no arguments.
    """
    if config.kind == "suction":
        s = config.suction
        return SuctionCupGripperModel(
            cup_radius_mm=s.cup_radius_mm,
            cup_height_mm=s.cup_height_mm,
            contact_tip_depth_mm=s.contact_tip_depth_mm,
            shaft_radius_mm=s.shaft_radius_mm,
            shaft_length_mm=s.shaft_length_mm,
            mount_radius_mm=s.mount_radius_mm,
            mount_depth_mm=s.mount_depth_mm,
            outer_margin_mm=config.outer_margin_mm,
        )
    j = config.parallel_jaw
    return ParallelJawGripperModel(
        finger_length_mm=j.finger_length_mm,
        finger_thickness_mm=j.finger_thickness_mm,
        finger_width_mm=j.finger_width_mm,
        finger_pad_overlap_mm=j.finger_pad_overlap_mm,
        fingertip_depth_mm=j.fingertip_depth_mm,
        pad_length_mm=j.pad_length_mm,
        pad_ahead_mm=j.pad_ahead_mm,
        palm_depth_mm=j.palm_depth_mm,
        palm_width_mm=j.palm_width_mm,
        outer_margin_mm=config.outer_margin_mm,
    )


def build_subpolicies(
    grasping_cfg: Optional[RobotGraspingConfig],
    *,
    refinement_policy: Optional[RefinementPolicy],
    verification_policy: Optional[GraspVerificationPolicy],
    recovery_policy: Optional[SceneRecoveryPolicy],
    standoff_mm: float,
    recovery_fixture: Optional[FixtureEnvelope],
) -> tuple[
    Optional[RefinementPolicy],
    Optional[GraspVerificationPolicy],
    Optional[SceneRecoveryPolicy],
]:
    """Sub-policy auto-build (refinement / verification / recovery).

    Each block is materialised only when the caller did not pass an explicit
    policy and the corresponding ``grasping_cfg.*`` block is enabled. Disabled
    blocks leave the slot at :data:`None` so the ``MODE_NOT_AVAILABLE`` gates
    fire. Raises :class:`ValueError` for an unknown recovery action, or for a
    physical recovery action requested without a ``recovery_fixture``.
    """

    resolved_refinement_policy = refinement_policy
    resolved_verification_policy = verification_policy
    resolved_recovery_policy = recovery_policy

    if grasping_cfg is not None:
        if resolved_refinement_policy is None and grasping_cfg.closed_loop.enabled:
            cl = grasping_cfg.closed_loop
            resolved_refinement_policy = RefinementPolicy(
                enabled=True,
                standoff_mm=standoff_mm,
                max_position_correction_mm=float(cl.max_position_correction_mm),
                max_orientation_correction_deg=float(
                    cl.max_orientation_correction_deg
                ),
                max_grip_width_correction_mm=float(
                    cl.max_grip_width_correction_mm
                ),
                target_match_iou_threshold=float(cl.target_match_iou_threshold),
                # Whether refinement takes the second scan (standoff move plus re-acquire) or
                # re-validates on the frame it already has. ``service.py`` gates the standoff
                # move on ``policy.reperceive``. A static overhead camera wants it off: moving
                # to standoff buys nothing when the viewpoint does not change, and costs a move
                # plus an acquire per attempt.
                reperceive=bool(cl.pregrasp_rescan),
            )
        if (
            resolved_verification_policy is None
            and grasping_cfg.verification.enabled
        ):
            vc = grasping_cfg.verification
            vp_kwargs: dict[str, Any] = {
                "enabled": True,
                "require_object_detected": bool(vc.require_object_detected),
                "width_delta_min_mm": float(vc.width_delta_min_mm),
                "post_lift_vision_check": bool(vc.post_lift_vision_check),
                "vision_displacement_iou_max": float(
                    vc.vision_displacement_iou_max
                ),
                "fail_closed": bool(vc.fail_closed),
            }
            # Optional schema field: propagate only when set, so the
            # policy default of ``None`` is not overridden.
            max_delta = getattr(vc, "width_delta_max_mm", None)
            if max_delta is not None:
                vp_kwargs["width_delta_max_mm"] = float(max_delta)
            resolved_verification_policy = GraspVerificationPolicy(**vp_kwargs)
        if (
            resolved_recovery_policy is None
            and grasping_cfg.dense_recovery.enabled
        ):
            dr = grasping_cfg.dense_recovery
            resolved_actions: list[SceneRecoveryAction] = []
            for name in dr.allowed_actions:
                try:
                    resolved_actions.append(SceneRecoveryAction(name))
                except ValueError as exc:
                    raise ValueError(
                        f"robot.grasping.dense_recovery.allowed_actions "
                        f"contains unknown action {name!r}. "
                        f"Allowed: {[a.value for a in SceneRecoveryAction if a is not SceneRecoveryAction.NONE]}"
                    ) from exc
            # SceneRecoveryPolicy enforces the physical-action /
            # fixture invariant in its __post_init__. The pre-check
            # here raises an actionable error that names the missing
            # ``recovery_fixture`` kwarg directly.
            _PHYSICAL = {
                SceneRecoveryAction.NUDGE_TARGET,
                SceneRecoveryAction.CONTAINER_AGITATE,
            }
            if any(a in _PHYSICAL for a in resolved_actions) and recovery_fixture is None:
                raise ValueError(
                    "robot.grasping.dense_recovery requests a "
                    "physical action (nudge_target / "
                    "container_agitate) but no ``recovery_fixture`` "
                    "was supplied. Pass a FixtureEnvelope to "
                    "AutonomousGraspService.from_robot_config "
                    "(Phase T0 fail-closed)."
                )
            resolved_recovery_policy = SceneRecoveryPolicy(
                enabled=True,
                allowed_actions=tuple(resolved_actions),
                max_recovery_actions=int(dr.max_recovery_actions),
                fixture=recovery_fixture,
            )

    return (
        resolved_refinement_policy,
        resolved_verification_policy,
        resolved_recovery_policy,
    )


def build_closed_loop_actors(
    grasping_cfg: Optional[RobotGraspingConfig],
    *,
    refinement_policy: Optional[RefinementPolicy],
    verification_policy: Optional[GraspVerificationPolicy],
    recovery_policy: Optional[SceneRecoveryPolicy],
    refiner: Optional[PreGraspRefiner],
    verifier: Optional[GraspVerifier],
    recovery_strategy: Optional[SceneRecoveryStrategy],
) -> tuple[
    Optional[PreGraspRefiner],
    Optional[GraspVerifier],
    Optional[SceneRecoveryStrategy],
]:
    """The runtime objects the three closed-loop policies need in order to act.

    ``build_subpolicies`` turns ``closed_loop`` / ``verification`` / ``dense_recovery``
    into policies; this builds the ``refiner``, ``verifier`` and ``recovery_strategy``
    that consume them, so an enabled policy has something behind it rather than
    reporting ``mode_requires_refinement_but_no_refiner_wired`` at pick time.

    None of these objects is hardware-coupled: ``DefaultPreGraspRefiner`` needs only
    its policy (the tracker defaults), and the verifiers and all three recovery
    strategies take no required arguments.

    An explicitly passed object always wins over an auto-built one.
    """

    if grasping_cfg is None:
        return refiner, verifier, recovery_strategy

    resolved_refiner = refiner
    resolved_verifier = verifier
    resolved_strategy = recovery_strategy

    if resolved_refiner is None and refinement_policy is not None and refinement_policy.enabled:
        from src.robot.grasping.closed_loop.refinement import DefaultPreGraspRefiner

        resolved_refiner = DefaultPreGraspRefiner(policy=refinement_policy)

    if (
        resolved_verifier is None
        and verification_policy is not None
        and verification_policy.enabled
    ):
        from src.robot.grasping.closed_loop.verification import (
            CompositeGraspVerifier,
            ObjectDetectingGripperVerifier,
            VisionTargetDisplacementVerifier,
            WidthDeltaGripperVerifier,
        )

        # Both gripper verifiers, because they read different things: one asks the gripper whether
        # it believes it holds something, the other measures how far the jaws actually closed. A
        # gripper that offers neither makes both inconclusive, which ``require_all_conclusive``
        # then decides; its schema field states why it defaults to passing and saying so.
        verifiers: list[Any] = [ObjectDetectingGripperVerifier(), WidthDeltaGripperVerifier()]

        # And the scene verifier, when the operator asked for the post-lift frame. With
        # ``post_lift_vision_check`` on, the service re-acquires a full perception frame after the
        # lift and passes the target identity into the verification context; this verifier is what
        # reads them and applies ``vision_displacement_iou_max``.
        #
        # It matters most on the cell that has the least to say otherwise: a jaw gripper driven
        # over digital I/O reports neither "object detected" nor a jaw width, so both gripper
        # verifiers return inconclusive by construction there. Looking at the scene is then the
        # only way to find out whether the object actually left the table.
        if bool(grasping_cfg.verification.post_lift_vision_check):
            verifiers.append(VisionTargetDisplacementVerifier())

        resolved_verifier = CompositeGraspVerifier(
            verifiers=tuple(verifiers),
            require_all_conclusive=bool(
                getattr(grasping_cfg.verification, "require_all_conclusive", False)
            ),
        )

    if resolved_strategy is None and recovery_policy is not None and recovery_policy.enabled:
        from src.robot.grasping.recovery.policy import (
            ActivePerceptionRecoveryStrategy,
            NextTargetRecoveryStrategy,
            NoRecoveryStrategy,
        )

        choice = str(getattr(grasping_cfg.dense_recovery, "strategy", "active_perception"))
        strategies: dict[str, SceneRecoveryStrategy] = {
            "active_perception": ActivePerceptionRecoveryStrategy(),
            "next_target": NextTargetRecoveryStrategy(),
            "none": NoRecoveryStrategy(),
        }
        resolved_strategy = strategies[choice]

    auto_built = [
        name
        for name, before, after in (
            ("refiner", refiner, resolved_refiner),
            ("verifier", verifier, resolved_verifier),
            ("recovery_strategy", recovery_strategy, resolved_strategy),
        )
        if before is None and after is not None
    ]
    if auto_built:
        # Naming the actors config assembled makes the enabled/wired pair checkable in the log; an
        # enabled policy with nothing behind it is otherwise invisible until pick time.
        logger.info("closed-loop actors auto-built from config: %s", ", ".join(auto_built))
    return resolved_refiner, resolved_verifier, resolved_strategy


def assert_closed_loop_actors_wired(
    *,
    refinement_policy: Optional[RefinementPolicy],
    verification_policy: Optional[GraspVerificationPolicy],
    recovery_policy: Optional[SceneRecoveryPolicy],
    refiner: Optional[PreGraspRefiner],
    verifier: Optional[GraspVerifier],
    recovery_strategy: Optional[SceneRecoveryStrategy],
) -> None:
    """Refuse to build a service whose enabled policy has nothing to drive it.

    Without this check the service builds cleanly and then reports, per attempt at pick time, that
    the requested mode is not available. Raising here puts the complaint at the boot, where the
    misconfiguration is and while somebody is still looking.
    """

    missing = [
        name
        for name, policy, actor in (
            ("closed_loop", refinement_policy, refiner),
            ("verification", verification_policy, verifier),
            ("dense_recovery", recovery_policy, recovery_strategy),
        )
        if policy is not None and getattr(policy, "enabled", False) and actor is None
    ]
    if missing:
        raise ValueError(
            "robot.grasping."
            + " and robot.grasping.".join(missing)
            + " is enabled but has no runtime object to drive it. Config normally builds one "
            "(builders.build_closed_loop_actors); a caller that passes its own policies must pass "
            "the matching refiner / verifier / recovery_strategy too."
        )


def build_decision_layer(
    grasping_cfg: Optional[RobotGraspingConfig],
    *,
    decision_policy: Optional[DecisionPolicy],
    decision_engine: Optional[DecisionEngine],
) -> tuple[Optional[DecisionPolicy], Optional[DecisionEngine]]:
    """Decision-layer auto-build.

    Caller-supplied policy and engine instances always win over schema-derived
    ones. A disabled decision block leaves both slots at :data:`None`, so the
    pick path runs with no decision gate.
    """

    resolved_decision_policy = decision_policy
    resolved_decision_engine = decision_engine
    if resolved_decision_engine is not None and resolved_decision_policy is None:
        # A caller-supplied engine implicitly defines the policy.
        resolved_decision_policy = resolved_decision_engine.policy
    if (
        resolved_decision_engine is None
        and grasping_cfg is not None
        and grasping_cfg.decision.enabled
    ):
        dc = grasping_cfg.decision
        if resolved_decision_policy is None:
            resolved_decision_policy = DecisionPolicy(
                enabled=True,
                auto_uncertainty_threshold=float(dc.auto_uncertainty_threshold),
                max_reobservations=int(dc.max_reobservations),
                reasons_penalty=float(dc.reasons_penalty),
                fail_closed_on_real_hardware=bool(dc.fail_closed_on_real_hardware),
            )
        resolved_decision_engine = DecisionEngine(policy=resolved_decision_policy)
    return resolved_decision_policy, resolved_decision_engine


def build_effective_config(
    grasping_cfg: Optional[RobotGraspingConfig],
    *,
    resolved_mode: "GraspMode",
    resolved_max_attempts: int,
) -> Optional["EffectiveGraspingConfig"]:
    """Build the immutable effective-config snapshot.

    Returns :data:`None` when no ``grasping`` block was consulted, which leaves
    :attr:`AutonomousGraspService.effective_config` at :data:`None` on the
    ``mode``-override path. The ``apply_modes`` filters are enforced here so
    easy-mode services never see feasibility / occlusion / ordering / recovery
    overlays even when the operator turns a master flag on.
    """

    if grasping_cfg is None:
        return None

    # Feasibility-aware ranking overlay. The ``apply_modes`` filter is
    # enforced here, which keeps easy mode's +5% cycle-time budget
    # (``median_cycle_time_increase_pct_max``) intact.
    feas_cfg = grasping_cfg.feasibility
    feas_active = bool(feas_cfg.enabled) and (
        resolved_mode.value in feas_cfg.apply_modes
    )
    return EffectiveGraspingConfig(
        default_mode=resolved_mode,
        max_attempts=resolved_max_attempts,
        closed_loop_enabled=bool(grasping_cfg.closed_loop.enabled),
        verification_enabled=bool(grasping_cfg.verification.enabled),
        dense_recovery_enabled=bool(grasping_cfg.dense_recovery.enabled),
        dense_recovery_allowed_actions=tuple(
            grasping_cfg.dense_recovery.allowed_actions
        ),
        # The three keys the flat contract does not cover (config.py field docstrings say why).
        # Effective state, the same rule the feasibility, corridor and ordering overlays use: a
        # block outside its ``apply_modes`` reads False even when the YAML says true, because a
        # record has to answer whether the block was acting on this attempt. ``fusion`` has no
        # ``apply_modes``, so it reports the configured value; whether its substrate is wired
        # additionally depends on a CAMERA->BASE resolver, which is a runtime fact no config
        # snapshot can see.
        success_model_enabled=bool(
            grasping_cfg.success_model.enabled
            and resolved_mode.value in grasping_cfg.success_model.apply_modes
        ),
        fusion_enabled=bool(grasping_cfg.fusion.enabled),
        approach_validation_enabled=bool(
            grasping_cfg.approach_validation.enabled
            and resolved_mode.value in grasping_cfg.approach_validation.apply_modes
        ),
        decision=EffectiveDecisionConfig(
            enabled=bool(grasping_cfg.decision.enabled),
            auto_uncertainty_threshold=float(
                grasping_cfg.decision.auto_uncertainty_threshold
            ),
            max_reobservations=int(
                grasping_cfg.decision.max_reobservations
            ),
        ),
        feasibility=EffectiveFeasibilityConfig(
            enabled=feas_active,
            score_weight=(
                float(feas_cfg.weight) if feas_active else 0.0
            ),
            ik_quality_enabled=(
                bool(feas_cfg.ik_quality_enabled) if feas_active else False
            ),
            joint_margin_enabled=(
                bool(feas_cfg.joint_margin_enabled)
                if feas_active
                else False
            ),
            swept_approach_enabled=(
                bool(feas_cfg.swept_approach_enabled)
                if feas_active
                else False
            ),
            swept_approach_top_k=int(
                feas_cfg.swept_approach_top_k
            ),
            # The 4th feasibility signal is gated by the corridor
            # apply-modes filter so it cannot demote candidates
            # when the analyzer is not running.
            corridor_risk_enabled=(
                feas_active
                and bool(grasping_cfg.occlusion.directional_enabled)
                and (
                    resolved_mode.value
                    in grasping_cfg.occlusion.apply_modes
                )
            ),
            corridor_risk_weight=0.0,
        ),
        # Occlusion / corridor overlay. The ``apply_modes`` filter on
        # ``occlusion`` is enforced here, so an easy-mode service never
        # runs the analyzer.
        corridor=EffectiveCorridorConfig(
            directional_enabled=(
                bool(grasping_cfg.occlusion.directional_enabled)
                and (
                    resolved_mode.value
                    in grasping_cfg.occlusion.apply_modes
                )
            ),
            hard_reject_enabled=(
                bool(grasping_cfg.occlusion.hard_reject_enabled)
                and bool(grasping_cfg.occlusion.directional_enabled)
                and (
                    resolved_mode.value
                    in grasping_cfg.occlusion.apply_modes
                )
            ),
            top_k=int(grasping_cfg.occlusion.top_k),
            radius_mm=float(
                grasping_cfg.occlusion.corridor_radius_mm
            ),
            step_mm=float(
                grasping_cfg.occlusion.corridor_step_mm
            ),
            max_distance_mm=float(
                grasping_cfg.occlusion.corridor_max_distance_mm
            ),
            mask_fusion_weight=float(
                grasping_cfg.occlusion.mask_fusion_weight
            ),
            hard_reject_confidence_threshold=float(
                grasping_cfg.occlusion.hard_reject_confidence_threshold
            ),
        ),
        # Clutter-aware target ordering. Easy mode is locked out by
        # ``apply_modes`` regardless of the master flag; when inactive
        # every signal reads off.
        ordering=EffectiveOrderingConfig(
            enabled=(
                bool(grasping_cfg.ordering.enabled)
                and (
                    resolved_mode.value
                    in grasping_cfg.ordering.apply_modes
                )
            ),
            unlock_weight=(
                float(grasping_cfg.ordering.unlock_weight)
                if bool(grasping_cfg.ordering.enabled)
                and (
                    resolved_mode.value
                    in grasping_cfg.ordering.apply_modes
                )
                else 0.0
            ),
            max_local_score_drop=float(
                grasping_cfg.ordering.max_local_score_drop
            ),
            mask_adjacency_enabled=(
                bool(
                    grasping_cfg.ordering.blocker_graph.mask_adjacency_enabled
                )
                and bool(grasping_cfg.ordering.enabled)
                and (
                    resolved_mode.value
                    in grasping_cfg.ordering.apply_modes
                )
            ),
            depth_only_enabled=(
                bool(grasping_cfg.ordering.blocker_graph.depth_only_enabled)
                and bool(grasping_cfg.ordering.enabled)
                and (
                    resolved_mode.value
                    in grasping_cfg.ordering.apply_modes
                )
            ),
            corridor_overlap_enabled=(
                bool(
                    grasping_cfg.ordering.blocker_graph.corridor_overlap_enabled
                )
                and bool(grasping_cfg.ordering.enabled)
                and (
                    resolved_mode.value
                    in grasping_cfg.ordering.apply_modes
                )
            ),
            adjacency_radius_px=int(
                grasping_cfg.ordering.blocker_graph.adjacency_radius_px
            ),
            depth_tolerance_mm=float(
                grasping_cfg.ordering.blocker_graph.depth_tolerance_mm
            ),
        ),
        # Recovery orchestrator. Easy is permanently locked out by
        # ``apply_modes``. A resolved mode outside ``apply_modes``
        # collapses every flag to its off default.
        recovery_orchestrator=EffectiveRecoveryOrchestratorConfig(
            enabled=(
                bool(grasping_cfg.recovery.enabled)
                and (
                    resolved_mode.value
                    in grasping_cfg.recovery.apply_modes
                )
            ),
            max_actions=(
                int(grasping_cfg.recovery.max_recovery_actions)
                if bool(grasping_cfg.recovery.enabled)
                and (
                    resolved_mode.value
                    in grasping_cfg.recovery.apply_modes
                )
                else 2
            ),
            apply_modes=tuple(
                grasping_cfg.recovery.apply_modes
            ),
            allowed_actions=(
                tuple(grasping_cfg.recovery.allowed_actions)
                if bool(grasping_cfg.recovery.enabled)
                and (
                    resolved_mode.value
                    in grasping_cfg.recovery.apply_modes
                )
                else ()
            ),
            per_action_budget=(
                tuple(
                    (str(action), int(count))
                    for action, count in grasping_cfg.recovery.per_action_budget
                )
                if bool(grasping_cfg.recovery.enabled)
                and (
                    resolved_mode.value
                    in grasping_cfg.recovery.apply_modes
                )
                else ()
            ),
            # Emitted unconditionally, even when the mode gate above zeroes the actions: it is the
            # operator's declared bound, and a bound is not dropped because a gate is closed. The
            # consumer builds an envelope only when it has physical actions.
            fixture=(
                (
                    # Unpacked rather than comprehended: the schema pins both as 3-tuples, and a
                    # generator would widen them to `tuple[float, ...]`, which is exactly the shape
                    # a 2- or 4-element mistake would take.
                    (
                        float(grasping_cfg.recovery.fixture.center_mm[0]),
                        float(grasping_cfg.recovery.fixture.center_mm[1]),
                        float(grasping_cfg.recovery.fixture.center_mm[2]),
                    ),
                    (
                        float(grasping_cfg.recovery.fixture.half_extents_mm[0]),
                        float(grasping_cfg.recovery.fixture.half_extents_mm[1]),
                        float(grasping_cfg.recovery.fixture.half_extents_mm[2]),
                    ),
                    float(grasping_cfg.recovery.fixture.max_nudge_mm),
                )
                if grasping_cfg.recovery.fixture is not None
                else None
            ),
        ),
        uncertainty=EffectiveUncertaintyConfig(
            enabled=bool(grasping_cfg.uncertainty.enabled),
            # When the layer is enabled its own key wins; otherwise the
            # value mirrors ``decision.auto_uncertainty_threshold``.
            fail_closed_threshold=(
                float(grasping_cfg.uncertainty.fail_closed_threshold)
                if bool(grasping_cfg.uncertainty.enabled)
                else float(grasping_cfg.decision.auto_uncertainty_threshold)
            ),
            apply_modes=tuple(
                grasping_cfg.uncertainty.apply_modes
            ),
            weights=(
                (
                    "depth_confidence",
                    float(grasping_cfg.uncertainty.weights.depth_confidence),
                ),
                (
                    "mask_confidence",
                    float(grasping_cfg.uncertainty.weights.mask_confidence),
                ),
                (
                    "occlusion_corridor_risk",
                    float(
                        grasping_cfg.uncertainty.weights.occlusion_corridor_risk
                    ),
                ),
                (
                    "feasibility_margin",
                    float(grasping_cfg.uncertainty.weights.feasibility_margin),
                ),
                (
                    "verification_residual",
                    float(
                        grasping_cfg.uncertainty.weights.verification_residual
                    ),
                ),
                (
                    "topology_risk",
                    float(grasping_cfg.uncertainty.weights.topology_risk),
                ),
                (
                    "semantic_confidence",
                    float(
                        grasping_cfg.uncertainty.weights.semantic_confidence
                    ),
                ),
            ),
            ranking_penalty_weight=float(
                grasping_cfg.uncertainty.ranking_penalty_weight
            ),
            recovery_aggressive_threshold=float(
                grasping_cfg.uncertainty.recovery_aggressive_threshold
            ),
            channel_disagreement_threshold=float(
                grasping_cfg.uncertainty.channel_disagreement_threshold
            ),
            calibration_artifact_path=(
                grasping_cfg.uncertainty.calibration_artifact_path
            ),
        ),
        # Drift + OOD watchdog overlay.
        watchdog=EffectiveWatchdogConfig(
            mode=str(grasping_cfg.watchdog.mode),
            window_size=int(grasping_cfg.watchdog.window_size),
            min_samples_for_trend=int(
                grasping_cfg.watchdog.min_samples_for_trend
            ),
            calibration_delta_moderate_mm=float(
                grasping_cfg.watchdog.calibration_delta_moderate_mm
            ),
            calibration_delta_high_mm=float(
                grasping_cfg.watchdog.calibration_delta_high_mm
            ),
            calibration_delta_severe_mm=float(
                grasping_cfg.watchdog.calibration_delta_severe_mm
            ),
            depth_confidence_shift_moderate=float(
                grasping_cfg.watchdog.depth_confidence_shift_moderate
            ),
            depth_confidence_shift_high=float(
                grasping_cfg.watchdog.depth_confidence_shift_high
            ),
            depth_confidence_shift_severe=float(
                grasping_cfg.watchdog.depth_confidence_shift_severe
            ),
            fail_closed_rate_moderate=float(
                grasping_cfg.watchdog.fail_closed_rate_moderate
            ),
            fail_closed_rate_high=float(
                grasping_cfg.watchdog.fail_closed_rate_high
            ),
            fail_closed_rate_severe=float(
                grasping_cfg.watchdog.fail_closed_rate_severe
            ),
            hand_eye_residual_trend_moderate_mm=float(
                grasping_cfg.watchdog.hand_eye_residual_trend_moderate_mm
            ),
            hand_eye_residual_trend_high_mm=float(
                grasping_cfg.watchdog.hand_eye_residual_trend_high_mm
            ),
            hand_eye_residual_trend_severe_mm=float(
                grasping_cfg.watchdog.hand_eye_residual_trend_severe_mm
            ),
            verification_residual_trend_moderate_mm=float(
                grasping_cfg.watchdog.verification_residual_trend_moderate_mm
            ),
            verification_residual_trend_high_mm=float(
                grasping_cfg.watchdog.verification_residual_trend_high_mm
            ),
            verification_residual_trend_severe_mm=float(
                grasping_cfg.watchdog.verification_residual_trend_severe_mm
            ),
            ood_score_moderate=float(
                grasping_cfg.watchdog.ood_score_moderate
            ),
            ood_score_high=float(
                grasping_cfg.watchdog.ood_score_high
            ),
            ood_score_severe=float(
                grasping_cfg.watchdog.ood_score_severe
            ),
            block_modes=tuple(grasping_cfg.watchdog.block_modes),
        ),
        performance=EffectivePerformanceConfig(
            enabled=bool(grasping_cfg.performance.enabled),
            decision_latency_slo_ms=float(
                grasping_cfg.performance.decision_latency_slo_ms
            ),
            ranking_latency_slo_ms=float(
                grasping_cfg.performance.ranking_latency_slo_ms
            ),
            fusion_latency_slo_ms=float(
                grasping_cfg.performance.fusion_latency_slo_ms
            ),
            decision_timeout_ms=float(
                grasping_cfg.performance.decision_timeout_ms
            ),
            ranking_timeout_ms=float(
                grasping_cfg.performance.ranking_timeout_ms
            ),
            fusion_timeout_ms=float(
                grasping_cfg.performance.fusion_timeout_ms
            ),
            model_fallback_policy=str(
                grasping_cfg.performance.model_fallback_policy
            ),
            fusion_fallback_policy=str(
                grasping_cfg.performance.fusion_fallback_policy
            ),
            emit_breach_events=bool(
                grasping_cfg.performance.emit_breach_events
            ),
            breach_window_size=int(
                grasping_cfg.performance.breach_window_size
            ),
        ),
    )


def build_config_frame_resolver(grasping_cfg: Optional[RobotGraspingConfig]) -> Optional["FrameResolver"]:
    """Build a static CAMERA->BASE resolver from ``grasping.fusion.extrinsics_artifact_path``.

    Returns ``None`` (byte-identical) when ``grasping_cfg`` is ``None``, fusion is disabled,
    or no artifact path is configured. Fail-closed: a configured-but-unloadable path raises, so a fusion
    deployment with a missing/stale/invalid calibration artifact cannot silently run with an unreachable
    commit gate. Eye-in-hand cells leave the path unset and pass a ``frame_resolver`` in code (the
    live TCP-composed resolver cannot be serialized).
    """
    if grasping_cfg is None:
        return None
    fusion_cfg = getattr(grasping_cfg, "fusion", None)
    if fusion_cfg is None or not bool(getattr(fusion_cfg, "enabled", False)):
        return None
    path = getattr(fusion_cfg, "extrinsics_artifact_path", None)
    if not path:
        return None
    from src.calibration.serialization import load_extrinsics
    from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver

    try:
        ext = load_extrinsics(path)
    except Exception as exc:  # noqa: BLE001 (re-raised as a fail-closed config error)
        raise RuntimeError(
            f"grasping.fusion.extrinsics_artifact_path {path!r} failed to load: {exc}. Fusion is "
            "enabled but the calibration artifact is missing/stale/invalid -> the U6 commit gate would "
            "be unreachable. Fix the path or set grasping.fusion.enabled: false."
        ) from exc
    return StaticCameraToBaseResolver(transform=ext.transform)


def build_config_frame_resolvers(
    grasping_cfg: Optional[RobotGraspingConfig],
) -> dict[str, "FrameResolver"]:
    """Build a ``{camera_id -> FrameResolver}`` map from the multi-camera ``grasping.fusion.cameras`` map.

    The multi-view counterpart of :func:`build_config_frame_resolver`: each enabled camera is loaded
    and resolved individually. Returns ``{}`` (byte-identical) when ``grasping_cfg`` is ``None``,
    fusion is disabled, or the ``cameras`` map is empty, which leaves the single
    ``extrinsics_artifact_path`` path unchanged. Disabled cameras are skipped. By ``mounting_mode``:
    ``eye_to_hand`` loads the CAMERA->BASE :class:`Extrinsics` into a
    :class:`StaticCameraToBaseResolver`; ``eye_in_hand`` loads the CAMERA->TOOL transform into an
    :class:`EyeInHandFrameResolver`. Fail-closed: an enabled camera with a missing, stale or invalid
    artifact raises, because an incomplete multi-view resolver map must not run silently.
    """
    if grasping_cfg is None:
        return {}
    fusion_cfg = getattr(grasping_cfg, "fusion", None)
    if fusion_cfg is None or not bool(getattr(fusion_cfg, "enabled", False)):
        return {}
    cameras = getattr(fusion_cfg, "cameras", None) or {}
    if not cameras:
        return {}
    from src.calibration.serialization import load_cam_to_tool, load_extrinsics
    from src.robot.grasping.motion.frame_resolver import (
        EyeInHandFrameResolver,
        StaticCameraToBaseResolver,
    )

    resolvers: dict[str, "FrameResolver"] = {}
    for cam_id, cam_cfg in cameras.items():
        if not bool(getattr(cam_cfg, "enabled", True)):
            continue
        path = cam_cfg.extrinsics_artifact_path
        mode = cam_cfg.mounting_mode
        try:
            if mode == "eye_in_hand":
                resolvers[cam_id] = EyeInHandFrameResolver(t_cam_to_tool=load_cam_to_tool(path))
            else:
                resolvers[cam_id] = StaticCameraToBaseResolver(transform=load_extrinsics(path).transform)
        except Exception as exc:  # noqa: BLE001 (re-raised as a fail-closed per-camera config error)
            raise RuntimeError(
                f"grasping.fusion.cameras[{cam_id!r}] ({mode}) failed to load {path!r}: {exc}. Fusion is "
                "enabled but this camera's calibration is missing/stale/invalid -> the multi-view resolver "
                "map would be incomplete. Fix the artifact, set the camera enabled:false, or fusion.enabled:false."
            ) from exc
    return resolvers


def build_config_viewpoint_planner(
    grasping_cfg: Optional[RobotGraspingConfig], robot_cfg: RobotConfig
) -> Optional["ViewpointPlanner"]:
    """Auto-build a ScoringViewpointPlanner so the commit-gate reobserve relocates
    the camera to a diverse viewpoint in production (otherwise the reobserve re-ingests the same
    static frame and only inflates views_accepted without growing the corridor evidence).

    Returns ``None`` (byte-identical) unless all three of ``grasping.fusion.enabled``,
    ``grasping.fusion.commit_policy.enabled`` and ``grasping.fusion.active_perception_use_fusion``
    are True; ``active_perception_use_fusion`` is the dedicated, named off switch for this planner.
    A caller-supplied ``viewpoint_planner`` wins upstream. The planner's viewpoint budget is tied to
    ``commit_policy.max_reobserve_attempts``, the single source of truth for the reobserve budget,
    and its safety_check is a ``WorkspaceBoxSafetyCheck`` derived from ``robot_cfg.workspace_limits``
    so a proposed camera pose outside the declared workspace is refused (fail-closed backstop; falls
    back to the planner default if no workspace box is configured).
    """
    if grasping_cfg is None:
        return None
    fusion_cfg = getattr(grasping_cfg, "fusion", None)
    if fusion_cfg is None or not bool(getattr(fusion_cfg, "enabled", False)):
        return None
    if not bool(getattr(fusion_cfg, "active_perception_use_fusion", False)):
        return None
    commit_cfg = getattr(fusion_cfg, "commit_policy", None)
    if commit_cfg is None or not bool(getattr(commit_cfg, "enabled", False)):
        return None
    from src.robot.grasping.closed_loop.active_perception import (
        ScoringViewpointPlanner,
        ViewScoringPolicy,
        WorkspaceBoxSafetyCheck,
    )

    budget = int(getattr(commit_cfg, "max_reobserve_attempts", 1))
    policy = ViewScoringPolicy(max_viewpoints=max(budget, 1))
    wl = getattr(robot_cfg, "workspace_limits", None)
    if wl is not None:
        center = (
            (float(wl.x_min) + float(wl.x_max)) / 2.0,
            (float(wl.y_min) + float(wl.y_max)) / 2.0,
            (float(wl.z_min) + float(wl.z_max)) / 2.0,
        )
        half = (
            (float(wl.x_max) - float(wl.x_min)) / 2.0,
            (float(wl.y_max) - float(wl.y_min)) / 2.0,
            (float(wl.z_max) - float(wl.z_min)) / 2.0,
        )
        return ScoringViewpointPlanner(
            policy=policy,
            safety_check=WorkspaceBoxSafetyCheck(center_mm=center, half_extents_mm=half),
        )
    return ScoringViewpointPlanner(policy=policy)


def build_runtime(
    robot_cfg: RobotConfig,
    *,
    calculator: "GraspCalculator",
    perception: "PerceptionSource",
    viewpoint_planner: Optional["ViewpointPlanner"],
    resolved_max_attempts: int,
    profile: "GraspBehaviorProfile",
    standoff_mm: float,
    retreat_mm: float,
    policy: Optional["GraspExecutionPolicy"],
    frame_resolver: Optional["FrameResolver"],
    arm: Optional["RobotArm"] = None,
    gripper: Optional["Gripper"] = None,
) -> RuntimePickService:
    """Build the underlying :class:`RuntimePickService` and patch its frame guard.

    ``from_robot_config`` builds the arm and gripper from the config tree, so a
    fail-closed default policy cannot be pre-constructed the way ``from_components``
    does. The runtime builds its own default policy first, and
    ``require_base_frame_grasp`` is then set on it when a resolver is wired and the
    caller supplied no explicit policy. A caller-supplied policy is left untouched.
    """

    runtime = RuntimePickService.from_robot_config(
        robot_cfg,
        calculator=calculator,
        perception=perception,
        viewpoint_planner=viewpoint_planner,
        max_attempts=resolved_max_attempts,
        grasp_sampling_mode=profile.sampling_mode,
        standoff_mm=standoff_mm,
        retreat_mm=retreat_mm,
        policy=policy,
        arm=arm,
        gripper=gripper,
    )
    logger.info(
        "runtime pick service built: arm=%s gripper=%s sampling=%s max_attempts=%d "
        "standoff=%.1f mm retreat=%.1f mm frame_resolver=%s",
        type(arm).__name__ if arm is not None else "from-config",
        type(gripper).__name__ if gripper is not None else "from-config",
        profile.sampling_mode.value, int(resolved_max_attempts), float(standoff_mm), float(retreat_mm),
        type(frame_resolver).__name__ if frame_resolver is not None else "none",
    )
    if frame_resolver is not None:
        runtime.orchestrator.frame_resolver = frame_resolver
    if policy is None:
        # The orchestrator built its own default policy in __post_init__; the fail-closed guards are
        # wired onto it here. require_base_frame_grasp only when a resolver is wired; the dwell
        # steady-gate is independent of the resolver.
        built_policy = runtime.orchestrator.policy
        if built_policy is not None:
            if frame_resolver is not None:
                built_policy.require_base_frame_grasp = True
            # Enable the pre-move steady-state gate from robot_cfg.safety.dwell. Duck-typed via
            # getattr so the execution layer stays free of config-schema imports (strict downward stack).
            dwell = getattr(getattr(robot_cfg, "safety", None), "dwell", None)
            if dwell is not None:
                built_policy.require_steady_before_motion = bool(
                    getattr(dwell, "require_steady_before_motion", False)
                )
                built_policy.steady_timeout_s = float(getattr(dwell, "steady_timeout_s", 5.0))
    return runtime


def apply_orchestrator_overlays(
    runtime: RuntimePickService,
    grasping_cfg: Optional[RobotGraspingConfig],
    *,
    resolved_mode: "GraspMode",
    primary_camera_id: str | None = None,
) -> None:
    """Apply the orchestrator overlays in place.

    No-op when no ``grasping`` block was consulted. Each overlay is fail-safe and
    leaves its orchestrator slot at the default (``None``) when its sub-block is
    disabled, so a deployment that opted into none of these overlays runs the
    byte-identical path.
    """

    if grasping_cfg is None:
        logger.debug("no grasping config block: orchestrator overlays skipped entirely")
        return

    # What the gripper is, and what the parts stand on. Until these are set, ``collision_checker``
    # filters against a default jaw envelope and returns early on the table check, so a candidate
    # that drives a finger under the support is never rejected.
    #
    # Set first, before any of the optional overlays below: a pick path that does not know its own
    # end effector or its own table is wrong regardless of which advanced block is enabled.
    runtime.orchestrator.gripper_model = build_gripper_geometry(grasping_cfg.gripper_geometry)
    runtime.orchestrator.support_config = grasping_cfg.support

    # The learned grasp ranker, in shadow. Loaded eagerly and fail-safe, exactly as the
    # success-probability model below is: a missing artifact, a spec that disagrees with the model's
    # own column list, or any I/O error yields ``None`` plus one warning, and ``None`` in this slot
    # is the byte-identical path: nothing is scored and no telemetry key appears.
    #
    # Disabled by default. It changes no ordering and no decision; it is watched on real picks
    # through `would_change_top1` and nothing more.
    from src.robot.grasping.deep.ranker.context import DeepRankerContext  # noqa: PLC0415

    runtime.orchestrator.deep_ranker_context = DeepRankerContext.from_config(
        getattr(grasping_cfg, "deep_ranker", None))

    # Eagerly load the success-probability shadow model. The loader is
    # fail-safe: a missing artifact, schema mismatch, or any I/O error
    # yields ``None`` and a single ``WARNING`` log entry. ``None`` is
    # then plumbed into the orchestrator slot, which means the runtime
    # path is byte-identical (no metadata mutation, no telemetry on the
    # report). The mode label is the raw resolved mode value so the
    # predictor's one-hot bucketing sees the operator's intent rather
    # than the schema default.
    sm_cfg = grasping_cfg.success_model
    # Thread the configured lifecycle_phase (duck-typed via getattr so the execution layer stays free of
    # config-schema imports; defaults to "shadow", byte-identical). canary/active still fail closed to
    # ctx=None unless the artifact carries a valid promotion.json.
    lifecycle_phase = str(getattr(sm_cfg, "lifecycle_phase", "shadow"))
    shadow_ctx = try_load_shadow_success_context(
        enabled=bool(sm_cfg.enabled),
        artifact_dir=str(sm_cfg.artifact_dir),
        mode_label=resolved_mode.value,
        version_label="v1",
        lifecycle_phase=lifecycle_phase,
    )
    if shadow_ctx is not None:
        runtime.orchestrator.shadow_success_context = shadow_ctx
    # Bounded probability-blend reranker config, wired only when the
    # operator opted in via ``success_model.ranking_blend_enabled=True``.
    # The carrier validates the contract (weight in [0.0, 0.5],
    # dense-only modes). With ``enabled=False`` the orchestrator slot
    # stays at its default ``None`` so
    # :func:`maybe_blend_rerank_candidates` returns ``(candidates,
    # None)``: the silent byte-identical path.
    if bool(sm_cfg.ranking_blend_enabled):
        runtime.orchestrator.ranking_blend_config = RankingBlendConfig(
            enabled=True,
            weight=float(sm_cfg.ranking_blend_weight),
            modes=tuple(sm_cfg.ranking_blend_modes),
        )
    # Per-candidate uncertainty re-rank config, the consumer of the
    # corridor-risk producer. Wired only when ``grasping.uncertainty.rerank_enabled``.
    # The producer (GraspCalculator.corridor_risk_per_candidate) is turned on by the
    # calculator's owner at construction (the sim runners read the same flag); when
    # the producer is off the re-rank no-ops (skip_reason=missing_uncertainty).
    # Default-off leaves the slot ``None``, which is the byte-identical path.
    unc_cfg = grasping_cfg.uncertainty
    if bool(getattr(unc_cfg, "rerank_enabled", False)):
        runtime.orchestrator.uncertainty_rerank_config = UncertaintyRerankConfig(
            enabled=True,
            weight=float(unc_cfg.rerank_weight),
            modes=tuple(unc_cfg.rerank_modes),
        )
    # Shadow-only multi-view fusion substrate. Wired only when
    # ``grasping.fusion.enabled=True`` and a frame resolver is available
    # on the orchestrator. Without a resolver the substrate's strict
    # frame contract would reject every ingest, so the slot stays
    # ``None`` and preserves the byte-identical path. The orchestrator's
    # ingest seam re-checks ``config.enabled`` and the substrate itself
    # enforces frame and intrinsic contracts on every call; this
    # function never raises.
    fusion_cfg = grasping_cfg.fusion
    if (
        bool(fusion_cfg.enabled)
        and runtime.orchestrator.frame_resolver is not None
    ):
        runtime.orchestrator.scene_fusion = SceneFusion(
            config=FusionConfig(
                enabled=True,
                max_views=int(fusion_cfg.max_views),
                max_view_age_s=float(fusion_cfg.max_view_age_s),
                voxel_size_mm=float(fusion_cfg.voxel_size_mm),
                # The schema validates roi_extent_mm to exactly 3 components; keep the tuple() coercion
                # (runtime-identical) and cast to the fixed-arity type FusionConfig expects.
                roi_extent_mm=cast(
                    "tuple[float, float, float]", tuple(fusion_cfg.roi_extent_mm)
                ),
                max_voxels=int(fusion_cfg.max_voxels),
                depth_min_mm=float(fusion_cfg.depth_min_mm),
                depth_max_mm=float(fusion_cfg.depth_max_mm),
            )
        )
    # Multi-camera geometry fusion. Independent of the ``fusion.enabled`` substrate above: that one
    # runs the shadow voxel grid, this one changes the grasp candidates themselves by giving the
    # generator each object's surface as every camera that can identify it sees it: measured on
    # the datagen reference, top-1 is 43.50 % single-view against 55.93 % fused.
    #
    # The per-camera resolver map is built here rather than in the loop because it is fail-closed
    # at construction: an enabled camera whose calibration artifact will not load raises while an
    # operator is watching, instead of degrading to single-view later. The perception rig itself is
    # not config, it is a live device handle, so the caller passes it in; until it does, the loop
    # logs that it is running single-view rather than pretending to fuse.
    geometry_cfg = getattr(fusion_cfg, "geometry", None)
    if geometry_cfg is not None and bool(getattr(geometry_cfg, "enabled", False)):
        runtime.orchestrator.fusion_geometry_config = geometry_cfg
        runtime.orchestrator.camera_frame_resolvers = build_config_frame_resolvers(grasping_cfg)
        # What the config asked for, separately from what resolved. See
        # `BinPickingOrchestrator.configured_camera_ids`: the refusal policy needs the requested
        # set, not the resolver map, which is empty whenever `fusion.enabled` is false. Enabled
        # cameras only: a camera switched off was not asked for.
        _fusion = getattr(grasping_cfg, "fusion", None)
        _cameras = getattr(_fusion, "cameras", None) or {}
        # Every camera except the primary. The map names the cell's whole camera inventory, the
        # primary included, so that one list answers "how many cameras does this cell have" and the
        # simulator and a real cell mean the same thing by it. What this attribute means is narrower:
        # which cameras a pick waits for. The primary delivers through `perception`, never through
        # the extra-camera rig, so expecting it there makes it permanently missing, which is a
        # warning on every pick under `degrade` and a raised refusal on every pick under `refuse`.
        runtime.orchestrator.configured_camera_ids = tuple(
            sorted(cam_id for cam_id, cam in _cameras.items()
                   if bool(getattr(cam, "enabled", True)) and cam_id != primary_camera_id))
    # The shortfall is reported once at construction, not once per pick.
    #
    # `on_camera_unavailable` governs the case where a camera the config asked for did not deliver.
    # It cannot govern the case where no second camera was ever asked for, because a cell with one
    # camera is a legitimate cell and refusing it would be wrong. Single-view is a measured cost
    # rather than a neutral choice: on the datagen reference, top-1 goes 43.50 % single-view
    # against 55.93 % fused. An operator who believes they built a multi-camera cell and
    # built a one-camera cell has no other signal.
    #
    # One line at construction: a warning below two calibrated cameras, info otherwise. It sits
    # next to where the resolvers are built and fires whether or not fusion geometry is enabled, so
    # the case of fusion enabled with no second camera is named.
    _fusion_all = getattr(grasping_cfg, "fusion", None)
    _all_cameras = getattr(_fusion_all, "cameras", None) or {}
    _calibrated = sorted(cam_id for cam_id, cam in _all_cameras.items()
                         if bool(getattr(cam, "enabled", True)))
    # An eye-in-hand rig is a second viewpoint from one device, so it counts. The wrist camera sees
    # the far side of a part the fixed camera cannot, which is the whole reason fusion helps.
    _eih = sorted(cam_id for cam_id in _calibrated
                  if str(getattr(_all_cameras[cam_id], "mounting_mode", "")) == "eye_in_hand")
    if len(_calibrated) < 2:
        logger.warning(
            "this cell has %d calibrated camera(s) (%s) and will grasp SINGLE-VIEW. MEASURED on the "
            "datagen reference, fusing a second view moves top-1 from 43.50 %% to 55.93 %%. Add a "
            "camera to grasping.fusion.cameras with its own calibration artifact, or accept "
            "single-view deliberately.",
            len(_calibrated), ", ".join(_calibrated) or "none")
    else:
        logger.info("this cell has %d calibrated camera(s): %s (%d eye-in-hand)",
                    len(_calibrated), ", ".join(_calibrated), len(_eih))

    # Mode-aware commit gate. Always set ``mode_label`` so the
    # orchestrator can route mode-aware gates correctly. When the
    # schema-level commit policy is enabled, build the runtime carrier;
    # the orchestrator's gate helper still defends against missing
    # fusion via ``SKIPPED_NO_FUSION``.
    runtime.orchestrator.mode_label = str(resolved_mode.value)
    commit_cfg = fusion_cfg.commit_policy
    if bool(commit_cfg.enabled):
        runtime.orchestrator.commit_policy = CommitPolicy(
            enabled=True,
            min_views_accepted=int(commit_cfg.min_views_accepted),
            min_corridor_hit_fraction=float(
                commit_cfg.min_corridor_hit_fraction
            ),
            corridor_radius_mm=float(commit_cfg.corridor_radius_mm),
            corridor_length_mm=float(commit_cfg.corridor_length_mm),
            max_reobserve_attempts=int(
                commit_cfg.max_reobserve_attempts
            ),
            apply_modes=frozenset(commit_cfg.apply_modes),
        )

    # Clutter-aware target ordering, assigned to ``target_ordering`` below. The orchestrator
    # carries the ordering path: a ``target_ordering`` slot, a blocker graph, and `select_target`,
    # which re-picks the winner when a lower-scoring grasp unlocks more of the scene. Every runtime
    # field maps 1:1 onto the schema block, same names and same defaults, and ``apply_modes`` is
    # enforced here so a block outside its modes stays byte-identical rather than half-wired.
    #
    # Ordering cannot be judged on one pick. It changes which object is taken first, so a
    # single-target scene shows nothing by construction and a lift count over one pick is the wrong
    # metric entirely. The measurement is a full bin-clearing sequence.
    #
    # Directional occlusion, the scoring half only. `analyze_corridor` and the per-candidate stamp
    # live in the calculator; this overlay switches them on and sets their geometry, which the call
    # site otherwise takes from a default-constructed `CorridorAnalysisConfig()`. The runtime
    # fields map 1:1 onto the schema block, `blocked_confidence_threshold` included.
    #
    # `hard_reject_enabled` is deliberately not wired here and stays in UNWIRED_SWITCHES. It is
    # what lets the score refuse a candidate, and a run that moves both the ranking and the
    # rejection answers neither question. Wire it once the score itself is trusted.
    occ_cfg = grasping_cfg.occlusion
    if bool(occ_cfg.directional_enabled) and resolved_mode.value in occ_cfg.apply_modes:
        runtime.orchestrator.corridor_config = CorridorAnalysisConfig(
            radius_mm=float(occ_cfg.corridor_radius_mm),
            step_mm=float(occ_cfg.corridor_step_mm),
            max_distance_mm=float(occ_cfg.corridor_max_distance_mm),
            mask_fusion_weight=float(occ_cfg.mask_fusion_weight),
            partial_confidence_threshold=float(occ_cfg.partial_confidence_threshold),
            blocked_confidence_threshold=float(occ_cfg.hard_reject_confidence_threshold),
        )

    # Feasibility, the reachability re-rank. Its consumer is otherwise a GraspCalculator
    # constructor argument; supplied per call from here, like corridor_config, so wiring it once
    # wires it for every caller.
    #
    # Expect it to cost picks, and enable it on that understanding. calculator.py records the trap:
    # the ik_quality sub-signal demotes a good overhead top-down grasp, because such a grasp is
    # itself near-singular (high Jacobian condition number), so demoting a high condition number
    # demotes exactly the grasp that is wanted. It is wired so the claim can be re-measured through
    # config. It stays `enabled: false` in robot.yaml.
    feas_cfg_block = grasping_cfg.feasibility
    if (
        bool(feas_cfg_block.enabled)
        and float(feas_cfg_block.weight) > 0.0
        and resolved_mode.value in feas_cfg_block.apply_modes
    ):
        runtime.orchestrator.feasibility_config = FeasibilityScoreConfig(
            ik_quality_enabled=bool(feas_cfg_block.ik_quality_enabled),
            ik_quality_weight=float(feas_cfg_block.ik_quality_weight),
            joint_margin_enabled=bool(feas_cfg_block.joint_margin_enabled),
            joint_margin_weight=float(feas_cfg_block.joint_margin_weight),
            swept_approach_enabled=bool(feas_cfg_block.swept_approach_enabled),
            swept_approach_weight=float(feas_cfg_block.swept_approach_weight),
        )
        runtime.orchestrator.feasibility_weight = float(feas_cfg_block.weight)

    ord_cfg = grasping_cfg.ordering
    if bool(ord_cfg.enabled) and resolved_mode.value in ord_cfg.apply_modes:
        runtime.orchestrator.target_ordering = TargetOrderingConfig(
            enabled=True,
            unlock_weight=float(ord_cfg.unlock_weight),
            max_local_score_drop=float(ord_cfg.max_local_score_drop),
            blocker_graph=BlockerGraphConfig(
                mask_adjacency_enabled=bool(ord_cfg.blocker_graph.mask_adjacency_enabled),
                depth_only_enabled=bool(ord_cfg.blocker_graph.depth_only_enabled),
                corridor_overlap_enabled=bool(ord_cfg.blocker_graph.corridor_overlap_enabled),
                adjacency_radius_px=int(ord_cfg.blocker_graph.adjacency_radius_px),
                depth_tolerance_mm=float(ord_cfg.blocker_graph.depth_tolerance_mm),
            ),
        )

    # Dense-mode swept-volume approach/retreat validator. Set the carrier + the dense-only
    # apply_modes when enabled; the orchestrator gates per-pick on ``mode_label in _approach_path_modes``,
    # so non-dense / single-object picks stay byte-identical even when this is wired. Independent of
    # fusion (it needs only the per-frame neighbour masks the calculator already consumes).
    av_cfg = getattr(grasping_cfg, "approach_validation", None)
    if av_cfg is not None and bool(getattr(av_cfg, "enabled", False)):
        from src.robot.grasping.motion.trajectory_safety import ApproachPathPolicy

        runtime.orchestrator.approach_path_policy = ApproachPathPolicy(
            standoff_mm=float(av_cfg.standoff_mm),
            retreat_mm=float(av_cfg.retreat_mm),
            num_approach_samples=int(av_cfg.num_approach_samples),
            num_retreat_samples=int(av_cfg.num_retreat_samples),
            collision_margin_mm=float(av_cfg.collision_margin_mm),
        )
        runtime.orchestrator._approach_path_modes = frozenset(av_cfg.apply_modes)

    live = [slot for slot in _OVERLAY_SLOTS if getattr(runtime.orchestrator, slot, None) is not None]
    logger.info(
        "orchestrator overlays for mode=%s: %s",
        resolved_mode.value, ", ".join(live) if live else "none (default open-loop pick path)",
    )
