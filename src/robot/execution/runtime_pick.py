"""
Runtime pick facade: one pick attempt behind a single synchronous call.

This module sits between any future API or UI layer and the grasping
orchestrator. Its job is narrow:

1. Own one full perception, grasp and execution attempt synchronously.
2. Carry vendor identity (real or simulated) onto the result so a
   caller can tell which backend produced the report.
3. Aggregate the typed :class:`MotionStatus` chain, plus coarse timing
   markers, into a single :class:`PickSessionReport`.

Design notes
------------
* The facade depends only on the vendor-neutral surfaces
  (:class:`RobotArm`, :class:`Gripper`, :class:`PerceptionSource`,
  :class:`GraspCalculator`, :class:`BinPickingOrchestrator`).
* No vendor SDK import lives here. Construction takes already-built
  components, or a validated ``RobotConfig`` tree through
  :meth:`RuntimePickService.from_robot_config`.
* The facade is synchronous and in-process. It is importable on macOS
  without Isaac installed because it touches no simulator code path
  directly; when a caller hands it an :class:`IsaacRobotArm`, that
  driver's lazy SDK import decides what happens.
* The session report surfaces the typed motion status chain, so
  orchestration and debugging tooling need not scrape logs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, TYPE_CHECKING

from src.robot.core import (
    Gripper,
    MotionStatus,
    RobotArm,
    RobotCapabilities,
    RobotVendor,
)
from src.robot.grasping.motion.execution_policy import (
    GraspExecutionPolicy,
    PolicyReport,
)
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.generation.calculator import GraspCalculator
from src.robot.grasping.types.modes import GraspSamplingMode
from src.robot.grasping.loop.pick_loop import (
    BinPickingOrchestrator,
    PerceptionSource,
    PickAttempt,
    PickOutcome,
    PickReport,
    ViewpointPlanner,
)

if TYPE_CHECKING:  # pragma: no cover (import only for typing)
    from src.config.schema.robot.tool_frame_schema import ToolFrameConfig
    # Telemetry carriers, referenced only from string-form field
    # annotations (PEP 563 ``from __future__ import annotations`` is in
    # effect at module top). Keeping the import behind ``TYPE_CHECKING``
    # means ``runtime_pick`` never grows a runtime dependency on the
    # predictor module: the wiring surface stays ``pick_loop``
    # (consumer) and ``autonomous_grasp`` (loader).
    from src.robot.grasping.scoring.success_probability import (
        RankingBlendTelemetry,
        ShadowSuccessTelemetry,
        UncertaintyRerankTelemetry,
    )
    from src.robot.grasping.multiview.fusion import (
        FusionTelemetry,
    )
    from src.robot.grasping.loop.pick_loop import CommitDecision

    # Typing-only imports for the sim-config bridge below. Both are pure
    # (Pydantic schema + a plain driver dataclass, no Isaac/vendor SDK), and live
    # behind TYPE_CHECKING so the runtime import graph is unchanged.
    from src.config.schema.robot.sim_schema import SimConfig
    from src.robot.drivers.sim.config import SimRobotConfig


_LOG = logging.getLogger(__name__)

__all__ = [
    "PickSessionReport",
    "PickTimings",
    "RuntimePickService",
    "build_sim_driver_config",
]


def build_sim_driver_config(
    schema_sim: "SimConfig", tool_frame: "ToolFrameConfig | None" = None
) -> "SimRobotConfig":
    """Convert a Pydantic ``SimConfig`` into the driver-side ``SimRobotConfig`` dataclass.

    The single conversion point shared by ``RuntimePickService.from_robot_config`` and the
    ``src.willy_sim`` runners (so the Pydantic config tree is the one source of truth for
    the sim cell). Copies only the driver-relevant fields; the Pydantic-only scene-authoring extras
    (``assets_root``/``gripper_variant``/``scene_setup``/camera mount+near-clip) are read directly
    off the schema by willy_sim and intentionally do not cross into the lean driver dataclass.
    """
    from src.robot.drivers.sim.config import SimCameraConfig, SimRobotConfig

    cameras: dict[str, SimCameraConfig] = {
        name: SimCameraConfig(
            prim_path=cam.prim_path,
            mounting_mode=cam.mounting_mode,
        )
        for name, cam in schema_sim.cameras.items()
    }
    return SimRobotConfig(
        backend=schema_sim.backend,
        enabled=schema_sim.enabled,
        mock_mode=schema_sim.mock_mode,
        robot_model=schema_sim.robot_model,
        scene=schema_sim.scene,
        robot_prim_path=schema_sim.robot_prim_path,
        gripper_prim_path=schema_sim.gripper_prim_path,
        home_joint_positions=(
            tuple(schema_sim.home_joint_positions)
            if schema_sim.home_joint_positions is not None
            else None
        ),
        cameras=cameras,
        step_dt_s=schema_sim.step_dt_s,
        settle_timeout_s=schema_sim.settle_timeout_s,
        # One source of truth for flange->TCP: the gripper block the real drivers read too.
        # There is no `robot.sim.tcp_offset_mm`: a scalar paired with a quaternion hardcoded in
        # the sim driver would be two values for one transform, one of them invisible to config,
        # which lets `gripper_mount` swap a gripper's width profile while the arm keeps the
        # 2F-85's geometry (measured: EZU-35 8 mm out, EGU-50 17 mm).
        tool_offset_mm=(
            (float(tool_frame.offset_mm[0]), float(tool_frame.offset_mm[1]),
             float(tool_frame.offset_mm[2])) if tool_frame is not None else (0.0, 0.0, 0.0)
        ),
        tool_rotation_quat_xyzw=(
            (float(tool_frame.rotation_quat_xyzw[0]), float(tool_frame.rotation_quat_xyzw[1]),
             float(tool_frame.rotation_quat_xyzw[2]), float(tool_frame.rotation_quat_xyzw[3]))
            if tool_frame is not None else (0.0, 0.0, 0.0, 1.0)
        ),
        headless=schema_sim.headless,
    )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PickTimings:
    """Coarse wall-clock markers for a single :meth:`RuntimePickService.run_attempt` call.

    All values are in seconds. ``total_s`` covers the entire call duration; ``orchestrator_s`` is the
    ``BinPickingOrchestrator.run()`` span, and today the two are equal. There is no separate
    ``perception_s`` field: perception time is part of ``orchestrator_s`` because the orchestrator
    owns perception acquisition, and surfacing it separately needs a per-stage sub-timer on
    ``PickReport``.
    """

    total_s: float = 0.0
    orchestrator_s: float = 0.0


@dataclass(frozen=True, slots=True)
class PickSessionReport:
    """Typed aggregate result of one :meth:`RuntimePickService.run_attempt` call.

    Attributes
    ----------
    outcome
        Terminal :class:`PickOutcome` from the orchestrator.
    robot_vendor
        Vendor label as reported by :attr:`RobotCapabilities.vendor`
        (e.g. ``"ur"``, ``"kuka"``, ``"sim"``).
    is_simulated
        Mirror of :attr:`RobotCapabilities.is_simulated`. Lets callers
        gate "real hardware only" actions without re-checking the
        vendor string.
    gripper_present
        :data:`True` iff a gripper was wired into the policy. Object
        detection on grippers without that capability is reported as
        :data:`None` in :attr:`object_detected`.
    candidate_count
        Number of grasp candidates evaluated on the last perception
        frame seen by the orchestrator. ``0`` when no candidates were
        generated or perception was empty.
    target_index
        Segmentation index of the grasp chosen for execution, or
        :data:`None` if the loop never reached execution.
    selected_score
        Score of the executed grasp, or ``0.0`` when none ran.
    motion_status_chain
        Ordered tuple of every typed :class:`MotionStatus` observed
        along the attempt. Empty when the underlying driver only
        exposed the bool ``move_to`` path, which stays supported: an
        empty chain means the driver predates the typed contract, not
        that the attempt failed.
    motion_message
        Free-text detail from the last :class:`MotionResult`, useful
        for surfacing controller-side rejection messages.
    object_detected
        Threaded from :attr:`PolicyReport.object_detected`. :data:`None`
        when the gripper has no detection capability.
    attempts
        Full attempt history forwarded from the orchestrator, so
        debugging tools can see every rescan, relocate and next-target
        action without re-running the loop.
    executed_grasp
        The :class:`GraspResult` that was actually executed, or
        :data:`None` when execution did not happen.
    timings
        Wall-clock markers; see :class:`PickTimings`.
    error
        Last unexpected exception observed by the execution policy, if
        any. Typed motion failures do not populate this field: they
        live in :attr:`motion_status_chain`, so a non-``None`` value
        always indicates an out-of-band failure.
    """

    outcome: PickOutcome
    robot_vendor: str
    is_simulated: bool
    gripper_present: bool
    candidate_count: int = 0
    #: The executed grasp's own telemetry, taken from the pick loop.
    #:
    #: This is what fills `GraspAttemptRecord.initial_telemetry`. Without it the calculator's
    #: telemetry stops at `GraspResult.telemetry` inside the pick loop and reaches no record, so
    #: the `deep_ranker_*` keys the shadow ranker stamps onto this dict cannot be read back from a
    #: single record however many picks run.
    #:
    #: Empty when nothing executed, so a run that reaches no grasp is byte-identical.
    calculator_telemetry: Mapping[str, Any] = field(default_factory=dict)
    target_index: Optional[int] = None
    selected_score: float = 0.0
    motion_status_chain: tuple[MotionStatus, ...] = ()
    motion_message: str = ""
    object_detected: Optional[bool] = None
    attempts: tuple[PickAttempt, ...] = ()
    executed_grasp: Optional[GraspResult] = None
    timings: PickTimings = field(default_factory=PickTimings)
    error: Optional[BaseException] = None
    # Shadow-mode success-probability telemetry, forwarded verbatim
    # from :class:`PickReport`. ``None`` whenever the orchestrator was
    # built without a :class:`ShadowSuccessContext` or no candidate was
    # scored.
    shadow_success_telemetry: "ShadowSuccessTelemetry | None" = None
    # Bounded probability-blend reranker audit telemetry, forwarded
    # verbatim from :class:`PickReport`. ``None`` whenever the
    # orchestrator was built without a ``RankingBlendConfig`` (or
    # ``ranking_blend_config.enabled is False``); a non-``None`` record
    # describes either the applied rerank or the reason it was skipped.
    ranking_blend_telemetry: "RankingBlendTelemetry | None" = None
    # Per-candidate uncertainty re-rank telemetry, forwarded verbatim from the PickReport.
    uncertainty_rerank_telemetry: "UncertaintyRerankTelemetry | None" = None
    # Shadow-only multi-view fusion telemetry, forwarded verbatim from
    # :class:`PickReport`. ``None`` whenever the orchestrator was built
    # without a ``SceneFusion`` carrier (or it was disabled). The field
    # has zero influence on ``outcome``, ``executed_grasp`` and
    # ``attempts``.
    fusion_telemetry: "FusionTelemetry | None" = None
    # Typed commit-gate decision, forwarded verbatim from
    # :class:`PickReport`. ``None`` when the orchestrator exited
    # before a candidate was selected (e.g. NO_PERCEPTION). When the
    # gate is disabled this field still surfaces ``allowed=True`` so
    # consumers can distinguish "no commit policy" from "policy ran
    # and approved".
    commit_decision: "CommitDecision | None" = None

    @property
    def succeeded(self) -> bool:
        """Convenience flag: :data:`True` iff the grasp was executed."""

        return self.outcome is PickOutcome.EXECUTED


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


@dataclass
class RuntimePickService:
    """Single-attempt pick service used as the internal runtime entry point.

    The facade is minimal: it wraps a fully-built
    :class:`BinPickingOrchestrator` and produces a typed
    :class:`PickSessionReport`. Construction can:

    * take a pre-built orchestrator via the ``orchestrator`` field,
    * accept the orchestrator's ingredients and build one internally
      via :meth:`from_components`, or
    * read a validated ``RobotConfig`` tree via
      :meth:`from_robot_config`.
    """

    orchestrator: BinPickingOrchestrator

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_components(
        cls,
        *,
        arm: RobotArm,
        calculator: GraspCalculator,
        perception: PerceptionSource,
        gripper: Gripper | None = None,
        viewpoint_planner: ViewpointPlanner | None = None,
        max_attempts: int = 5,
        dense_sampling: bool = True,
        grasp_sampling_mode: GraspSamplingMode | bool | str | None = None,
        standoff_mm: float = 80.0,
        retreat_mm: float = 100.0,
        policy: GraspExecutionPolicy | None = None,
    ) -> "RuntimePickService":
        """Build a :class:`RuntimePickService` from raw components.

        Mirrors :class:`BinPickingOrchestrator`'s public constructor so
        callers do not need to import the orchestrator class directly
        once they have a facade in hand. The optional ``policy``
        argument is honoured if supplied; otherwise the orchestrator
        builds its own default. ``grasp_sampling_mode`` is the
        production-preferred knob; when omitted the ``dense_sampling``
        bool still drives behavior.
        """

        orchestrator = BinPickingOrchestrator(
            arm=arm,
            calculator=calculator,
            perception=perception,
            viewpoint_planner=viewpoint_planner,
            max_attempts=max_attempts,
            dense_sampling=dense_sampling,
            grasp_sampling_mode=grasp_sampling_mode,
            standoff_mm=standoff_mm,
            retreat_mm=retreat_mm,
            gripper=gripper,
            policy=policy,
        )
        return cls(orchestrator=orchestrator)

    # ------------------------------------------------------------------
    # Config-driven construction
    # ------------------------------------------------------------------

    @classmethod
    def from_robot_config(
        cls,
        robot_cfg,
        *,
        calculator: GraspCalculator,
        grasp_sampling_mode: GraspSamplingMode | bool | str | None = None,
        perception: PerceptionSource,
        viewpoint_planner: ViewpointPlanner | None = None,
        max_attempts: int = 5,
        dense_sampling: bool = True,
        standoff_mm: float = 80.0,
        retreat_mm: float = 100.0,
        policy: GraspExecutionPolicy | None = None,
        arm: RobotArm | None = None,
        gripper: Gripper | None = None,
    ) -> "RuntimePickService":
        """Build a :class:`RuntimePickService` straight from a validated
        ``RobotConfig`` tree.

        ``arm`` and ``gripper`` are live-device escape hatches. Some end-effectors are not
        describable by config because they are not data: an ``IsaacGripper`` needs the simulator
        session its arm already lives on, exactly as the multi-camera rig needs a live camera
        handle. Passing the handle here lets everything else come from config. The alternative is
        ``from_components``, which builds no ``effective_config``, so the sim would measure a
        hand-wired stack while a real cell ran the config-driven one.

        A supplied handle replaces only its own construction step; nothing else is skipped. The
        arm-vendor SDK readiness gate is skipped for a supplied arm, because a caller holding a
        constructed arm has already proved the SDK is there.

        The vendor block in the schema is the single
        source of truth: this method reads ``robot_cfg.vendor`` and
        dispatches to the matching driver factory via
        :func:`src.robot.drivers.create_arm` /
        :func:`src.robot.grippers.create_gripper`. For
        ``RobotVendor.SIM`` the schema ``SimConfig`` is converted into
        the driver-side ``SimRobotConfig`` dataclass so the existing
        ``create_arm(RobotVendor.SIM, config=...)`` contract is honoured
        without leaking Pydantic types into the driver layer.

        Parameters
        ----------
        robot_cfg
            Validated ``RobotConfig`` instance from
            :mod:`src.config.schema.robot`.
        calculator
            Already-constructed :class:`GraspCalculator`.
        perception
            Already-constructed :class:`PerceptionSource`.
        viewpoint_planner, max_attempts, dense_sampling, standoff_mm,
        retreat_mm, policy
            Forwarded verbatim to
            :class:`BinPickingOrchestrator` (see
            :meth:`from_components`).

        Returns
        -------
        RuntimePickService
            A ready-to-run facade. ``arm.connect()`` is not called
            here; callers stay in charge of the connect and disconnect
            lifecycle.
        """

        from src.robot.drivers import create_arm
        from src.robot.grippers import (
            GripperSubstitution,
            SubstitutionReason,
            create_gripper,
        )
        from src.robot.core import GripperVendor

        vendor = RobotVendor.from_string(robot_cfg.vendor)

        # Startup gate: fail early with a clear "host not ready for vendor X" message if the
        # configured arm vendor's SDK is missing, instead of late inside connect(). Mock paths that
        # drive no real device (the sim in mock_mode, the dummy) are skipped, so a host without the
        # vendor SDK can still run them.
        from src.robot.drivers.doctor import require_arm_vendor_ready

        sim_mock = vendor is RobotVendor.SIM and bool(robot_cfg.sim.mock_mode)
        require_arm_vendor_ready(
            vendor,
            mock_mode=sim_mock or vendor is RobotVendor.DUMMY or arm is not None,
        )

        # --- Arm construction --------------------------------------------------
        if arm is not None:
            pass  # caller supplied a live handle; every other step still runs from config
        elif vendor is RobotVendor.SIM:
            arm = create_arm(
                RobotVendor.SIM,
                config=build_sim_driver_config(robot_cfg.sim, robot_cfg.gripper.tool_frame),
            )
        elif vendor is RobotVendor.DUMMY:
            # Dummy arm ignores the config tree entirely.
            arm = create_arm(RobotVendor.DUMMY)
        else:
            # UR, KUKA, etc. take the full schema config tree.
            arm = create_arm(vendor, config=robot_cfg)

        # --- Gripper construction ----------------------------------------------
        #
        # Five config combinations cannot produce a real end-effector, one per `SubstitutionReason`
        # member. Resolving them to a working NullGripper and a log line is the most dangerous
        # silent state in this build path: the cell connects, every pick reports success, the jaws
        # close on nothing and lift nothing. `_substitute` below routes all five through one place
        # that warns and records the reason on the object, so a caller sees it without parsing
        # logs. A sixth case goes through it too.
        gripper_cfg = robot_cfg.gripper

        def _substitute(reason: SubstitutionReason, detail: str, fix: str) -> Gripper:
            _LOG.warning("from_robot_config: %s Built a NullGripper (no real end-effector). %s",
                         detail, fix)
            return create_gripper(
                GripperVendor.NONE,
                min_width_mm=gripper_cfg.min_width_mm,
                max_width_mm=gripper_cfg.max_width_mm,
                substitution=GripperSubstitution(
                    reason=reason, requested=str(gripper_cfg.vendor), detail=detail, fix=fix,
                ),
            )

        substituted: Gripper | None = None
        if gripper is not None:
            # A live end-effector the config cannot describe (the Isaac gripper on a shared
            # session). Skipping the block below also skips its NullGripper substitution, which is
            # the point: the sim's `gripper.vendor: robotiq` on a `sim` arm would otherwise
            # substitute a gripper that closes on nothing and reports success.
            return cls.from_components(
                arm=arm, calculator=calculator, perception=perception, gripper=gripper,
                viewpoint_planner=viewpoint_planner, max_attempts=max_attempts,
                grasp_sampling_mode=grasp_sampling_mode, dense_sampling=dense_sampling,
                standoff_mm=standoff_mm, retreat_mm=retreat_mm, policy=policy,
            )
        try:
            gripper_vendor = GripperVendor.from_string(gripper_cfg.vendor)
        except ValueError:
            # An unrecognised gripper.vendor string is a misconfig; tolerate it as "no gripper"
            # (the facade stays usable) but warn instead of silently dropping the end-effector.
            substituted = _substitute(
                SubstitutionReason.UNKNOWN_VENDOR,
                f"gripper.vendor={gripper_cfg.vendor!r} is not a gripper this stack knows.",
                "Set gripper.vendor to one of: none, dummy, robotiq, vacuum, jaw_io, onrobot.",
            )
            gripper_vendor = GripperVendor.NONE

        if gripper_vendor is GripperVendor.NONE:
            # Either a cell that genuinely has no end-effector (substitution stays None, a legitimate
            # configuration) or the unknown-vendor fallback above, which already built one with a reason.
            gripper = substituted if substituted is not None else create_gripper(
                GripperVendor.NONE,
                min_width_mm=gripper_cfg.min_width_mm,
                max_width_mm=gripper_cfg.max_width_mm,
            )
        elif gripper_vendor is GripperVendor.DUMMY:
            gripper = create_gripper(
                GripperVendor.DUMMY,
                min_width_mm=gripper_cfg.min_width_mm,
                max_width_mm=gripper_cfg.max_width_mm,
            )
        elif gripper_vendor is GripperVendor.ROBOTIQ:
            # Robotiq lives on the UR controller's tool I/O. If the
            # selected arm vendor is not UR, the operator likely
            # mis-configured the YAML; fall back to no-gripper rather
            # than crashing at connect time.
            if vendor is RobotVendor.UR:
                gripper = create_gripper(
                    GripperVendor.ROBOTIQ,
                    config=gripper_cfg,
                    ip=robot_cfg.ur.ip,
                )
            else:
                # A config requesting Robotiq on a non-UR arm (notably the Isaac sim, whose `sim`
                # profile sets gripper.vendor=robotiq + arm vendor=sim) cannot build a real gripper
                # through from_robot_config. Warn instead of silently booting gripper-less; the sim
                # builds its IsaacGripper via AutonomousGraspService.from_components (shared session).
                gripper = _substitute(
                    SubstitutionReason.ROBOTIQ_NEEDS_UR,
                    f"gripper.vendor='robotiq' but the arm vendor is {vendor.value!r}, not UR; a "
                    f"Robotiq lives on the UR controller's tool I/O and cannot be reached from here.",
                    "For the Isaac sim use AutonomousGraspService.from_components (the IsaacGripper "
                    "needs the shared session). On a real cell, set robot.vendor: ur.",
                )
        elif gripper_vendor is GripperVendor.VACUUM:
            # Suction over the arm's digital I/O: no SDK, so it builds from config (the driver, its
            # factory, the enum member and VacuumGripperConfig all ship). The arm must satisfy
            # SupportsDigitalIO (the real UR driver does); anything else falls back to NullGripper + warn,
            # mirroring the Robotiq-on-non-UR branch above rather than crashing at connect.
            #
            # Lifecycle contract: VacuumGripper.connect() immediately drives set_digital_output, and
            # from_robot_config does not call arm.connect() (the caller owns the lifecycle). So a real
            # cell must call arm.connect() before gripper.connect(), or the vacuum connect raises with no
            # I/O behind it.
            from src.robot.core import SupportsDigitalIO

            if isinstance(arm, SupportsDigitalIO):
                vac = gripper_cfg.vacuum
                gripper = create_gripper(
                    GripperVendor.VACUUM,
                    io=arm,
                    config=gripper_cfg,
                    vacuum_output_pin=vac.vacuum_output_pin,
                    blow_off_output_pin=vac.blow_off_output_pin,
                    vacuum_ok_input_pin=vac.vacuum_ok_input_pin,
                    io_port=vac.io_port,
                    engage_timeout_s=vac.engage_timeout_s,
                    blow_off_s=vac.blow_off_s,
                    vacuum_on_below_mm=vac.vacuum_on_below_mm,
                    min_width_mm=gripper_cfg.min_width_mm,
                    max_width_mm=gripper_cfg.max_width_mm,
                )
            else:
                gripper = _substitute(
                    SubstitutionReason.VACUUM_NEEDS_DIGITAL_IO,
                    f"gripper.vendor='vacuum' but the arm vendor {vendor.value!r} does not advertise "
                    f"SupportsDigitalIO; suction switches the controller's digital I/O, which only "
                    f"the real UR driver exposes.",
                    "Set robot.vendor: ur for a real suction cell, or gripper.vendor: none.",
                )
        elif gripper_vendor is GripperVendor.JAW_IO:
            # Parallel jaws over the arm's digital I/O: same story as suction directly above, same
            # lifecycle contract (arm.connect() before gripper.connect(), because the driver touches
            # the controller's I/O as soon as it connects), and the same fallback when the
            # configured arm cannot switch a pin.
            #
            # This is the vendor for a gripper wired to the controller. A Robotiq is jaws too and
            # is not this: it speaks a socket the URCap opens, which is why it has its own branch
            # and its own failure mode.
            from src.robot.core import SupportsDigitalIO

            if isinstance(arm, SupportsDigitalIO):
                jaw = gripper_cfg.jaw_io
                gripper = create_gripper(
                    GripperVendor.JAW_IO,
                    io=arm,
                    config=gripper_cfg,
                    actuation=jaw.actuation,
                    close_output_pin=jaw.close_output_pin,
                    open_output_pin=jaw.open_output_pin,
                    pulse_s=jaw.pulse_s,
                    part_present_input_pin=jaw.part_present_input_pin,
                    closed_confirm_input_pin=jaw.closed_confirm_input_pin,
                    open_confirm_input_pin=jaw.open_confirm_input_pin,
                    io_port=jaw.io_port,
                    close_timeout_s=jaw.close_timeout_s,
                    close_settle_s=jaw.close_settle_s,
                    closed_below_mm=jaw.closed_below_mm,
                    open_on_connect_without_feedback=jaw.open_on_connect_without_feedback,
                    min_width_mm=gripper_cfg.min_width_mm,
                    max_width_mm=gripper_cfg.max_width_mm,
                )
            else:
                gripper = _substitute(
                    SubstitutionReason.JAW_IO_NEEDS_DIGITAL_IO,
                    f"gripper.vendor='jaw_io' but the arm vendor {vendor.value!r} does not advertise "
                    f"SupportsDigitalIO; a solenoid jaw switches the controller's digital I/O, "
                    f"which only the real UR driver exposes.",
                    "Set robot.vendor: ur for a real I/O jaw cell, or gripper.vendor: none.",
                )
        elif gripper_vendor is GripperVendor.ONROBOT:
            # A gripper vendor in the enum with no branch here falls into the `else` below: a
            # `NullGripper` is substituted, a warning is logged, and the cell connects, reports
            # every pick a success and holds nothing. Every new enum member needs its branch
            # written here.
            #
            # There is no `SupportsDigitalIO` precondition, unlike the two I/O branches, and none
            # is missing: a Compute Box is a separate device on its own network address, so this
            # gripper has no dependency on the arm vendor at all. That is also why `host` comes from
            # the gripper's own config and not from `robot.ur.ip` the way the Robotiq branch does.
            rg = gripper_cfg.onrobot
            gripper = create_gripper(
                GripperVendor.ONROBOT,
                config=gripper_cfg,
                host=rg.host,
                port=rg.port,
                unit=rg.unit_id,
                default_force_n=rg.default_force_n,
                use_fingertip_offset=rg.use_fingertip_offset,
            )
        else:  # pragma: no cover (future vendors)
            # A recognised gripper vendor with no registered driver (FRANKA_HAND, schunk, future)
            # falls back to NullGripper; warn so a config-only build is not silently gripper-less.
            gripper = _substitute(
                SubstitutionReason.NO_DRIVER,
                f"gripper.vendor={gripper_vendor.value!r} is a recognised name with no driver in this "
                f"repo.",
                "Use robotiq, vacuum, jaw_io or onrobot, or write the driver and register it in "
                "grippers/registry.py.",
            )

        return cls.from_components(
            arm=arm,
            calculator=calculator,
            perception=perception,
            gripper=gripper,
            viewpoint_planner=viewpoint_planner,
            max_attempts=max_attempts,
            grasp_sampling_mode=grasp_sampling_mode,
            dense_sampling=dense_sampling,
            standoff_mm=standoff_mm,
            retreat_mm=retreat_mm,
            policy=policy,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_attempt(self) -> PickSessionReport:
        """Run one full pick attempt and return a typed session report.

        The call is synchronous. Any exception raised by the
        orchestrator propagates: unexpected runtime faults are still
        raised, while ordinary motion outcomes are typed.
        """

        start = time.perf_counter()
        pick_report = self.orchestrator.run()
        orchestrator_s = time.perf_counter() - start
        return self._compose(pick_report, orchestrator_s=orchestrator_s)

    # ------------------------------------------------------------------
    # Report assembly
    # ------------------------------------------------------------------

    def _compose(
        self,
        pick_report: PickReport,
        *,
        orchestrator_s: float,
    ) -> PickSessionReport:
        capabilities = self._capabilities()
        policy_report = self.orchestrator._last_policy_report  # noqa: SLF001
        chain, message, object_detected, error = self._extract_policy_fields(
            policy_report,
        )

        target_index, selected_score, candidate_count = self._executed_target_metrics(
            pick_report,
        )

        return PickSessionReport(
            outcome=pick_report.outcome,
            robot_vendor=capabilities.vendor if capabilities else "unknown",
            is_simulated=bool(capabilities.is_simulated) if capabilities else False,
            gripper_present=self.orchestrator.gripper is not None,
            candidate_count=candidate_count,
            # `selected_telemetry` first. `executed_grasp` exists only on a successful pick, and the
            # shadow ranker's opinion is about the candidates rather than about the motion, so a
            # failed execution must still carry it.
            calculator_telemetry=dict(
                getattr(pick_report, "selected_telemetry", None)
                or getattr(getattr(pick_report, "executed_grasp", None), "telemetry", None) or {}),
            target_index=target_index,
            selected_score=selected_score,
            motion_status_chain=chain,
            motion_message=message,
            object_detected=object_detected,
            attempts=pick_report.attempts,
            executed_grasp=pick_report.executed_grasp,
            timings=PickTimings(
                total_s=orchestrator_s,
                orchestrator_s=orchestrator_s,
            ),
            error=error,
            shadow_success_telemetry=pick_report.shadow_success_telemetry,
            ranking_blend_telemetry=pick_report.ranking_blend_telemetry,
            uncertainty_rerank_telemetry=pick_report.uncertainty_rerank_telemetry,
            fusion_telemetry=pick_report.fusion_telemetry,
            commit_decision=pick_report.commit_decision,
        )

    def _capabilities(self) -> RobotCapabilities | None:
        """Return :class:`RobotCapabilities` if the arm exposes them.

        A driver that does not implement the property yields
        :data:`None`, and the facade falls back to ``"unknown"`` and
        ``is_simulated=False``.
        """

        caps = getattr(self.orchestrator.arm, "capabilities", None)
        if isinstance(caps, RobotCapabilities):
            return caps
        return None

    @staticmethod
    def _extract_policy_fields(
        policy_report: PolicyReport | None,
    ) -> tuple[tuple[MotionStatus, ...], str, bool | None, BaseException | None]:
        """Pull the typed motion fields off the orchestrator's last policy report.

        The orchestrator stores only the most recent policy report,
        because each pick attempt fires a single policy call (the loop
        retries at the orchestrator level by re-running perception, not
        by replaying execution). The contract therefore yields at most
        one :class:`PolicyReport` per session, and its
        :attr:`PolicyReport.motion_status` is the canonical chain
        entry.
        """

        if policy_report is None:
            return (), "", None, None

        chain: tuple[MotionStatus, ...]
        if policy_report.motion_status is None:
            chain = ()
        else:
            chain = (policy_report.motion_status,)

        return (
            chain,
            policy_report.motion_message,
            policy_report.object_detected,
            policy_report.error,
        )

    @staticmethod
    def _executed_target_metrics(
        pick_report: PickReport,
    ) -> tuple[Optional[int], float, int]:
        """Derive ``target_index``, ``selected_score``, ``candidate_count``.

        ``candidate_count`` is the length of the executed grasp's ranked candidate list, and
        ``0`` whenever no grasp executed. ``target_index`` comes from the last attempt in either
        case, and the executed grasp's own ``top_score`` is used as ``selected_score`` when there
        is one.
        """

        if pick_report.executed_grasp is not None:
            last = pick_report.attempts[-1] if pick_report.attempts else None
            target_index = last.target_index if last is not None else None
            return (
                target_index,
                float(pick_report.executed_grasp.top_score),
                len(pick_report.executed_grasp.candidates)
                if hasattr(pick_report.executed_grasp, "candidates")
                else 0,
            )

        if not pick_report.attempts:
            return None, 0.0, 0
        last = pick_report.attempts[-1]
        return last.target_index, float(last.score), 0
