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

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional

from src.contracts import UNSET, Maybe, chosen
from src.robot.constants import GRASP_BUILDERS_LOG_FILE, create_robot_logger
from src.robot.execution.runtime_pick import RuntimePickService
from src.robot.grasping.collision.gripper_model import HandSteps, jaw_envelope
from src.robot.grasping.collision.support_resolver import DeclaredSupport, bench_height_mm
from src.robot.grasping.motion.grasp_motion import (
    GraspMotion,
    build_execution_policy,
    foreign_policy_refusal,
)

from .config import (
    CELL_RECOVERY,
    EffectiveGraspingConfig,
)

if TYPE_CHECKING:
    from src.config.schema import CameraConfig
    from src.config.schema.robot import RobotConfig, RobotGraspingConfig
    from src.robot.core import Gripper, RobotArm
    from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
    from src.robot.grasping.motion.frame_resolver import FrameResolver
    from src.robot.grasping.generation.calculator import GraspCalculator
    from src.robot.grasping.loop.pick_loop import PerceptionSource

    from .config import GraspBehaviorProfile, GraspMode


# The default pick path is open-loop and nearly every advanced ``grasping.*`` block ships disabled,
# so what a boot actually wired is not visible from the config tree alone. These helpers run once
# per service build and are the only code that knows, so they write the answer to this log.
logger = create_robot_logger("AutonomousGraspBuilders", GRASP_BUILDERS_LOG_FILE)

#: Orchestrator slots that an overlay fills, in the order ``apply_orchestrator_overlays`` sets them.
#: Logged as one aggregate line rather than one line per slot.
_OVERLAY_SLOTS: tuple[str, ...] = (
    "camera_frame_resolvers",
)


# ``build_subpolicies``, ``build_closed_loop_actors`` and ``assert_closed_loop_actors_wired`` left on
# 2026-09-29 with the two blocks they turned into runtime objects: ``robot.grasping.verification``
# (a verifier no pick path consulted) and ``robot.grasping.dense_recovery`` (a recovery policy and
# strategy stored on the service and consulted by no pick). The recovery a pick runs is fixed since 2026-10-16
# (``CELL_RECOVERY``), and the service reads it from the effective-config snapshot below.


def build_effective_config(
    grasping_cfg: Optional[RobotGraspingConfig],
    *,
    resolved_mode: "GraspMode",
    resolved_max_attempts: int,
    critical_parts: bool = False,
    fusion: bool = False,
) -> Optional["EffectiveGraspingConfig"]:
    """Build the immutable effective-config snapshot.

    Returns :data:`None` when no ``grasping`` block was consulted, which leaves
    :attr:`AutonomousGraspService.effective_config` at :data:`None` on the
    ``mode``-override path. The recovery is the cell's fixed one (:data:`CELL_RECOVERY`), whose ``apply_modes`` filter
    is enforced here, so an easy-mode service never sees a recovery action. ``critical_parts`` is the cell's
    ``runtime.task.critical_parts``, and ``fusion`` whether its camera section fuses two or more cameras
    (:func:`~src.robot.execution.camera_fusion.fused_cameras`).
    """

    if grasping_cfg is None:
        return None

    recovery = CELL_RECOVERY
    if resolved_mode.value not in recovery.apply_modes:
        # A resolved mode outside ``apply_modes`` collapses every action to off.
        recovery = replace(recovery, enabled=False, allowed_actions=(), per_action_budget=())
    return EffectiveGraspingConfig(
        default_mode=resolved_mode,
        max_attempts=resolved_max_attempts,
        # Whether the camera section fuses its cameras; whether another camera's view was actually fused depends on the
        # rig delivering, a runtime fact no config snapshot can see.
        fusion_enabled=bool(fusion),
        recovery_orchestrator=replace(recovery, critical_parts=bool(critical_parts)),
    )


def _resolver_for_rig(rig_id: str, extrinsics: Any) -> "FrameResolver":
    """The resolver a rig's declared calibration gives, loaded through the one loader.

    A calibration that does not load is refused with a ``RuntimeError`` naming the rig key.
    """
    from src.calibration.rig_calibration import RigArtifactMissing, RigCalibration, RigCalibrationError
    from src.robot.grasping.motion.frame_resolver import (
        EyeInHandFrameResolver,
        StaticCameraToBaseResolver,
    )

    try:
        calibration = RigCalibration.from_config(rig_id, extrinsics)
    except RigCalibrationError as exc:
        # A missing artifact's refusal already says to sweep or leave the block out; any other says only
        # what the file did.
        fix = "" if isinstance(exc, RigArtifactMissing) else (
            ": fix the artifact, or remove the rig's extrinsics block until the camera is calibrated")
        raise RuntimeError(
            f"{str(exc).rstrip('.')}. The cell is refused at construction rather than run with a camera it cannot "
            f"place{fix}."
        ) from exc
    if calibration.mounting_mode == "eye_in_hand":
        return EyeInHandFrameResolver(t_cam_to_tool=calibration.camera_to_tool())
    return StaticCameraToBaseResolver(transform=calibration.camera_to_base())


def build_config_frame_resolver(
    grasping_cfg: Optional[RobotGraspingConfig],
    *,
    camera: "Maybe[CameraConfig]" = UNSET,
) -> Optional["FrameResolver"]:
    """The primary camera's frame resolver, built from its rig's calibration.

    The calibration is declared on the rig, ``camera.cameras.rigs[<primary>].extrinsics``. An
    eye_to_hand rig gives a :class:`StaticCameraToBaseResolver`; an eye_in_hand rig gives an
    :class:`EyeInHandFrameResolver`, which composes the tool pose. Every cell builds it, one camera or
    several: this resolver is what turns every grasp into the base frame, and a fused cell builds the
    other cameras' beside it (:func:`build_config_frame_resolvers`). ``grasping_cfg`` is kept for its
    callers and not read, because the calibration is not declared in the grasping block.

    Returns ``None`` when no camera section is handed in or the primary rig declares no
    calibration. Fail-closed: a declared artifact that does not load raises at construction, naming
    the rig key.
    """
    if not chosen(camera):
        return None
    primary = camera.cameras.primary_rig_id
    rig = next((r for r in camera.cameras.rigs if r.rig_id == primary), None)
    extrinsics = getattr(rig, "extrinsics", None)
    if extrinsics is None:
        return None
    return _resolver_for_rig(primary, extrinsics)


def build_config_frame_resolvers(camera: "Maybe[CameraConfig]" = UNSET) -> dict[str, "FrameResolver"]:
    """A ``{camera_id -> FrameResolver}`` map for every camera the cell fuses, the primary among them.

    The cameras are :func:`~src.robot.execution.camera_fusion.fused_cameras`: every enabled rig that declares its
    calibration, RGB-D or stereo, where two or more do (the owner, 2026-10-16: no switch). Each resolver is built from
    its rig's ``camera.cameras.rigs[<id>].extrinsics``, through the one loader and by mounting, as
    :func:`build_config_frame_resolver` does. ``{}`` where the cell fuses none, or no camera section is handed in.

    Fail-closed: a declared artifact that does not load raises at construction, naming the rig key.
    """
    from src.robot.execution.camera_fusion import fused_cameras  # noqa: PLC0415

    if not chosen(camera):
        return {}
    rigs = {rig.rig_id: rig for rig in camera.cameras.rigs}
    return {cam_id: _resolver_for_rig(cam_id, rigs[cam_id].extrinsics)
            for cam_id in fused_cameras(camera.cameras.rigs, camera.cameras.primary_rig_id)}


def build_runtime(
    robot_cfg: RobotConfig,
    *,
    calculator: "GraspCalculator",
    perception: "PerceptionSource",
    resolved_max_attempts: int,
    profile: "GraspBehaviorProfile",
    standoff_mm: float,
    retreat_mm: float,
    policy: Optional["GraspExecutionPolicy"],
    frame_resolver: Optional["FrameResolver"],
    arm: Optional["RobotArm"] = None,
    gripper: Optional["Gripper"] = None,
    motion: "Maybe[GraspMotion]" = UNSET,
) -> RuntimePickService:
    """Build the underlying :class:`RuntimePickService` and patch its frame guard.

    ``from_robot_config`` builds the arm and gripper from the config tree, so a
    fail-closed policy cannot be pre-constructed the way ``from_components`` does.
    Once the runtime has resolved them, the one policy it drives is built on that arm
    and hand from ``motion`` (a default :class:`GraspMotion` when unset) with every
    guard: ``require_base_frame_grasp`` when a resolver is wired, the dwell steady
    gate, and the jaws opened to the hand's ``max_width_mm`` before the approach. A
    caller-supplied policy drives as given when it holds the runtime's own arm and
    hand, and is refused with ``ValueError`` when it holds another.
    """

    runtime = RuntimePickService.from_robot_config(
        robot_cfg,
        calculator=calculator,
        perception=perception,
        max_attempts=resolved_max_attempts,
        standoff_mm=standoff_mm,
        retreat_mm=retreat_mm,
        policy=policy,
        arm=arm,
        gripper=gripper,
    )
    if frame_resolver is not None:
        runtime.orchestrator.frame_resolver = frame_resolver
    orchestrator = runtime.orchestrator
    if policy is not None:
        # A policy drives the arm and hand it was built with, so one around another arm or hand would
        # move a robot this cell never checked. Refused for as long as ``policy=`` is accepted at all.
        refusal = foreign_policy_refusal(policy, arm=orchestrator.arm, gripper=orchestrator.gripper)
        if refusal:
            raise ValueError(refusal)
    else:
        # The one policy, built on the arm and hand this cell resolved, with every guard, whether the
        # caller chose a motion or not: the base frame guard only when a resolver is wired (the steady
        # gate is the arm's own since P0e); and the jaws opened to the hand's width before the approach. Without the pre-open the arm descends with the jaws
        # wherever the previous close left them, and a close onto a wider part reaches the gripper as
        # an opening. The width is the hand's own, not a constant: the 2F-85 opens 85 mm and the
        # Hand-E 49.99.
        orchestrator.policy = build_execution_policy(
            motion if chosen(motion) else GraspMotion(),
            arm=orchestrator.arm, gripper=orchestrator.gripper, standoff_mm=standoff_mm, retreat_mm=retreat_mm,
            base_frame_required=frame_resolver is not None,
        )
    policy_built = runtime.orchestrator.policy
    pre_open = getattr(policy_built, "pre_open_width_mm", None) if policy_built is not None else None
    logger.info(
        "runtime pick service built: arm=%s gripper=%s mode=%s max_attempts=%d "
        "standoff=%.1f mm retreat=%.1f mm frame_resolver=%s pre_open=%s",
        type(arm).__name__ if arm is not None else "from-config",
        type(gripper).__name__ if gripper is not None else "from-config",
        profile.mode.value, int(resolved_max_attempts), float(standoff_mm), float(retreat_mm),
        type(frame_resolver).__name__ if frame_resolver is not None else "none",
        f"{float(pre_open):.2f} mm" if pre_open is not None else "none",
    )
    return runtime


def apply_orchestrator_overlays(
    runtime: RuntimePickService,
    grasping_cfg: Optional[RobotGraspingConfig],
    *,
    resolved_mode: "GraspMode",
    #: The camera section of the same tree, whose rigs declare each fused camera's calibration.
    camera: "Maybe[CameraConfig]" = UNSET,
    #: How far past the TCP the guard holds the jaw (``planning.hand.hand_past_the_tcp_mm``): the calculator's jaw stands
    #: there too (``gripper_model.jaw_envelope``).
    hand_past_the_tcp_mm: float = 0.0,
    #: The registry's housing along the closing axis (``gripper_model.jaw_envelope``); ``None`` where it says none.
    palm_thickness_mm: float | None = None,
    #: The fingers and the housing as the hand's guard bundle measures them (``gripper_model.jaw_envelope``); ``None``
    #: where none was measured.
    hand_steps: "HandSteps | None" = None,
    #: The tree the cell is built from: the hand its calculator plans (``gripper_model.jaw_envelope``) and the bench its
    #: parts stand on (``support_resolver.bench_height_mm``). ``None`` plans the 2F-85 over a bench at 0 mm.
    robot_cfg: "RobotConfig | None" = None,
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
    runtime.orchestrator.gripper_model = jaw_envelope(robot_cfg, past_the_tcp_mm=hand_past_the_tcp_mm,
                                                      palm_thickness_mm=palm_thickness_mm, hand_steps=hand_steps)
    if hand_past_the_tcp_mm > 0.0:
        logger.info("the guard holds the jaw's grasp centre %.2f mm past the declared TCP: the calculator plans the hand "
                    "there too (the preflight's grasp centre row names the difference)", float(hand_past_the_tcp_mm))
    runtime.orchestrator.declared_support = DeclaredSupport(height_mm=bench_height_mm(robot_cfg),
                                                            container=grasping_cfg.support.container)
    # A task follows its parts from pick to pick (the owner, 2026-10-09), and a frame carries the parts beside the named
    # ones, read off the depth (the owner, 2026-10-12): every built cell's since 2026-10-17, as the owner's ran them.
    runtime.orchestrator.follow_parts = True
    runtime.orchestrator.name_neighbours = True

    # The learned grasp ranker, in shadow. Loaded eagerly and fail-safe: a missing artifact, a spec that disagrees with
    # the model's own column list, or any I/O error yields ``None`` plus one warning, and ``None`` in this slot is the
    # byte-identical path: nothing is scored and no telemetry key appears.
    #
    # Disabled by default. It changes no ordering and no decision; it is watched on real picks
    # through `would_change_top1` and nothing more.
    from src.robot.grasping.deep.ranker.context import DeepRankerContext  # noqa: PLC0415

    runtime.orchestrator.deep_ranker_context = DeepRankerContext.from_config(
        getattr(grasping_cfg, "deep_ranker", None))

    # Multi-camera fusion, by itself wherever the camera section declares two or more enabled, calibrated rigs, RGB-D or
    # stereo (the owner, 2026-10-16: no switch). It changes the grasp candidates themselves by giving the generator each
    # object's surface as every camera that can identify it sees it: measured on the datagen reference, top-1 is
    # 43.50 % single-view against 55.93 % fused.
    #
    # The per-camera resolver map is built here rather than in the loop because it is fail-closed at construction: a
    # camera whose calibration artifact will not load raises while an operator is watching, instead of degrading to
    # single-view later. The other cameras' rig is a live device handle, so the caller passes it in; until it does,
    # the loop logs that it is running single-view rather than pretending to fuse.
    from src.robot.execution.camera_fusion import CameraFusionPlan, gives_depth  # noqa: PLC0415

    if chosen(camera):
        plan = CameraFusionPlan.from_config(camera.cameras.rigs, primary_rig_id=camera.cameras.primary_rig_id)
        if plan.cameras:
            runtime.orchestrator.camera_frame_resolvers = build_config_frame_resolvers(camera)
            logger.info("%s", plan.render())
        else:
            # One line at construction. A cell with one camera is a legitimate cell, and a wrist camera fuses its own
            # looks; a cell that declares two or more enabled rigs that give depth and fuses none was meant to, and has
            # no other signal. A stereo pair whose stereo calibration is not written gives none: the base profile's.
            enabled = [rig for rig in camera.cameras.rigs if bool(rig.enabled) and gives_depth(rig)]
            if len(enabled) >= 2:
                logger.warning("%s MEASURED on the datagen reference, fusing a second view moves top-1 from 43.50 %% to "
                               "55.93 %%.", plan.refusal())
            else:
                logger.info("%s", plan.render())

    live = [slot for slot in _OVERLAY_SLOTS if getattr(runtime.orchestrator, slot, None) is not None]
    logger.info(
        "orchestrator overlays for mode=%s: %s",
        resolved_mode.value, ", ".join(live) if live else "none (default open-loop pick path)",
    )
