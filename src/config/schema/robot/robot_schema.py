"""Schemas for the vendor-selected robot configuration tree."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from .._base import StrictModel

from .ur_schema import URConfig
from .kuka_schema import KukaConfig, KukaEkiConfig
from .dummy_schema import DummyConfig
from .sim_schema import (
    SimCameraSchema,
    SimConfig,
    SimGateConfig,
    SimMarkerConfig,
    SimObjectConfig,
    SimSceneConfig,
    SimTableConfig,
    SimViewpointConfig,
)
from .calibration_schema import RobotCalibrationConfig, RobotCalibrationQualityBandsMm

from .safety_schema import (
    DwellSafetyConfig,
    FixtureBoxConfig,
    IkQualitySafetyConfig,
    JointLimitSafetyConfig,
    LimitsSafetyConfig,
    MotionContinuitySafetyConfig,
    PayloadSafetyConfig,
    RobotSafetyConfig,
    AttachedPayloadConfig,
    PerceivedWorldConfig,
    PlanningWorldConfig,
    PlannerMeshConfig,
    SelfCollisionSafetyConfig,
    SupportPlaneConfig,
    TrajectoryCheckConfig,
)
from .tool_frame_schema import ToolFrameConfig

from .grasping_schema import (
    BlockerGraphSchemaConfig,
    GraspingClosedLoopConfig,
    GraspingDecisionConfig,
    GraspingDenseRecoveryConfig,
    GraspingFeasibilityConfig,
    GraspingOcclusionConfig,
    GraspingOrderingConfig,
    GraspingPerformanceConfig,
    GraspingRecoveryConfig,
    GraspingRecoveryFixtureConfig,
    GraspingSuccessModelConfig,
    GraspingUncertaintyConfig,
    GraspingVerificationConfig,
    GraspingWatchdogConfig,
    RobotGraspingApproachValidationConfig,
    RobotGraspingCommitPolicyConfig,
    RobotGraspingConfig,
    RobotGraspingFusionConfig,
    UncertaintyChannelWeightsConfig,
)

from .rl_schema import (
    RLExperimentalConfig,
    RL_ACTIVE_MODES,
    RL_MODE_GEOMETRY_ONLY,
    RL_MODE_HYBRID_ML,
    RL_MODE_RL_ACTIVE,
    RL_MODE_RL_EXPERIMENTAL,
    RL_MODE_RL_SHADOW,
    RL_MODE_VALUES,
    RobotRLConfig,
)

__all__ = [
    "BlockerGraphSchemaConfig",
    "DummyConfig",
    "DwellSafetyConfig",
    "FixtureBoxConfig",
    "GraspingClosedLoopConfig",
    "GraspingDecisionConfig",
    "GraspingDenseRecoveryConfig",
    "GraspingFeasibilityConfig",
    "GraspingOcclusionConfig",
    "GraspingOrderingConfig",
    "GraspingPerformanceConfig",
    "GraspingRecoveryConfig",
    "GraspingRecoveryFixtureConfig",
    "GraspingSuccessModelConfig",
    "GraspingUncertaintyConfig",
    "GraspingVerificationConfig",
    "GraspingWatchdogConfig",
    "GripperConfig",
    "IkQualitySafetyConfig",
    "JointLimitSafetyConfig",
    "KukaConfig",
    "KukaEkiConfig",
    "LimitsSafetyConfig",
    "MotionContinuitySafetyConfig",
    "MotionLimitsConfig",
    "PayloadSafetyConfig",
    "RLExperimentalConfig",
    "RL_ACTIVE_MODES",
    "RL_MODE_GEOMETRY_ONLY",
    "RL_MODE_HYBRID_ML",
    "RL_MODE_RL_ACTIVE",
    "RL_MODE_RL_EXPERIMENTAL",
    "RL_MODE_RL_SHADOW",
    "RL_MODE_VALUES",
    "RobotCalibrationConfig",
    "RobotCalibrationQualityBandsMm",
    "RobotConfig",
    "RobotGraspingApproachValidationConfig",
    "RobotGraspingCommitPolicyConfig",
    "RobotGraspingConfig",
    "RobotGraspingFusionConfig",
    "RobotRLConfig",
    "RobotSafetyConfig",
    "SafePoseConfig",
    "AttachedPayloadConfig",
    "PerceivedWorldConfig",
    "PlannerMeshConfig",
    "PlanningWorldConfig",
    "SelfCollisionSafetyConfig",
    "SupportPlaneConfig",
    "TrajectoryCheckConfig",
    "SimCameraSchema",
    "SimConfig",
    "SimGateConfig",
    "SimMarkerConfig",
    "SimObjectConfig",
    "SimSceneConfig",
    "SimTableConfig",
    "SimViewpointConfig",
    "URConfig",
    "OnRobotGripperConfig",
    "VacuumGripperConfig",
    "UncertaintyChannelWeightsConfig",
    "WorkspaceLimitsConfig",
]


class MotionLimitsConfig(StrictModel):
    """Hard upper bounds on commanded velocity and acceleration."""

    max_velocity: float = Field(default=1.0, gt=0.0, le=3.14)
    max_acceleration: float = Field(default=0.5, gt=0.0, le=5.0)


class SafePoseConfig(StrictModel):
    """Cartesian position the robot can retreat to at operator request.

    Position only. The single reader is the willy_sim reach doctor (``harness/reach.py``), which
    asks whether the point lies inside the arm's envelope, and orientation does not enter that
    question. Nothing moves to this pose; orientation belongs with the code that would.
    """

    x: float = 0.0
    y: float = -300.0
    z: float = 400.0


class WorkspaceLimitsConfig(StrictModel):
    """Cartesian workspace bounding box in millimetres."""

    x_min: float = -500.0
    x_max: float = 500.0
    y_min: float = -500.0
    y_max: float = 500.0
    z_min: float = 0.0
    z_max: float = 500.0

    @model_validator(mode="after")
    def _check_ordering(self) -> WorkspaceLimitsConfig:
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be < x_max ({self.x_max})")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be < y_max ({self.y_max})")
        if self.z_min >= self.z_max:
            raise ValueError(f"z_min ({self.z_min}) must be < z_max ({self.z_max})")
        return self


class VacuumGripperConfig(StrictModel):
    """Wiring and timing for a suction end-effector on the controller's digital I/O.

    Consulted only when ``gripper.vendor == "vacuum"``. Every field is a number measured on the
    actual cell: which pin the ejector is on, which bank it is in, whether a vacuum switch is
    wired, how long that ejector takes to build a seal. The defaults describe the common case
    (tool I/O, pin 0, no switch) and are not a claim about any particular cell.
    """

    #: Output pin that switches the ejector or pump on.
    vacuum_output_pin: int = Field(default=0, ge=0, le=7)
    #: Optional output pulsed on release. Residual vacuum holds a light part on the cup after the
    #: ejector stops, so the part lets go somewhere unintended; this pulse pushes it off where it
    #: was meant to land.
    blow_off_output_pin: int | None = Field(default=None, ge=0, le=7)
    #: Optional input from a vacuum switch, giving the real post-close verification the jaw path
    #: lacks. Without it the driver can only report what it commanded.
    vacuum_ok_input_pin: int | None = Field(default=None, ge=0, le=7)
    #: Which I/O bank the pins live on. A tool-mounted ejector is usually on the tool block.
    io_port: Literal["standard", "configurable", "tool"] = "tool"
    #: How long to wait for the switch to confirm a seal. A timeout is a missed grasp, not a fault.
    engage_timeout_s: float = Field(default=1.0, gt=0.0, le=30.0)
    #: Blow-off pulse length on release.
    blow_off_s: float = Field(default=0.15, ge=0.0, le=5.0)
    #: Commanded width at/below which the driver engages vacuum. Mirrors the simulated cup so the
    #: two interpret the width-based Gripper Protocol identically.
    vacuum_on_below_mm: float = Field(default=5.0, gt=0.0)


class JawIOGripperConfig(StrictModel):
    """Wiring and timing for a parallel-jaw gripper on the controller's digital I/O.

    Consulted only when ``gripper.vendor == "jaw_io"``. As in :class:`VacuumGripperConfig`, every
    field is a number measured on the actual cell: which pin closes the jaws, whether the reed
    switch is active-high, how long the cylinder takes to travel. The defaults describe the common
    pneumatic case (tool I/O, single solenoid, no feedback) and are not a claim about any
    particular cell.

    A digital-I/O jaw is binary. It cannot travel to 40 mm, so the width-based ``Gripper`` Protocol
    is reinterpreted here exactly as the suction driver reinterprets it; see ``closed_below_mm``.
    """

    #: How the jaws are driven.
    #:
    #: ``single_solenoid``: one output, high = close, low = open (spring return). The common
    #: pneumatic case. On power loss the spring opens and a held part falls.
    #:
    #: ``double_solenoid``: two pulsed outputs, one per direction, on a bistable valve. The valve
    #: holds the part on power loss, which is the safer failure, but the jaw state is then not
    #: inferable from the outputs, so feedback pins matter more here.
    actuation: Literal["single_solenoid", "double_solenoid"] = "single_solenoid"
    #: Output that closes the jaws: held high while closed under ``single_solenoid``, pulsed
    #: under ``double_solenoid``.
    close_output_pin: int = Field(default=0, ge=0, le=7)
    #: Output that opens the jaws. Required for ``double_solenoid``; unused (and must stay unset)
    #: for ``single_solenoid``, where "open" is simply dropping ``close_output_pin``.
    open_output_pin: int | None = Field(default=None, ge=0, le=7)
    #: Pulse length for ``double_solenoid``. A bistable valve latches, so the coil is energised only
    #: long enough to throw it; holding it high is what cooks the coil.
    pulse_s: float = Field(default=0.2, gt=0.0, le=5.0)
    #: Optional input from a dedicated part-present sensor. The simplest feedback: one pin, read
    #: directly.
    part_present_input_pin: int | None = Field(default=None, ge=0, le=7)
    #: Optional reed switch that reads true when the jaws are fully closed. Fully closed after a
    #: close command means the jaws met each other, so the grasp is empty.
    closed_confirm_input_pin: int | None = Field(default=None, ge=0, le=7)
    #: Optional reed switch that reads true when the jaws are fully open. With both switches wired
    #: the driver can tell "closed on nothing" from "closed on a part": neither switch active means
    #: the jaws stopped in between, so something is between them. That is the strongest post-grasp
    #: signal this path offers.
    open_confirm_input_pin: int | None = Field(default=None, ge=0, le=7)
    #: Which I/O bank the pins live on. A tool-mounted gripper is usually on the tool block.
    io_port: Literal["standard", "configurable", "tool"] = "tool"
    #: How long to wait for the jaws to reach a settled state after a close. A timeout is a missed
    #: grasp, not a fault; the verification stage decides, exactly as for suction.
    close_timeout_s: float = Field(default=1.0, gt=0.0, le=30.0)
    #: Fixed travel wait used when no feedback pin is wired: there is nothing to poll, so the driver
    #: can only wait. Also the settle pause after the jaws leave the open switch.
    close_settle_s: float = Field(default=0.3, ge=0.0, le=10.0)
    #: Commanded width at/below which the driver closes. Mirrors ``vacuum_on_below_mm`` so both I/O
    #: end-effectors interpret the width-based Protocol identically.
    closed_below_mm: float = Field(default=5.0, gt=0.0)
    #: What ``connect()`` does when no feedback is wired. The default, False, is not to actuate.
    #:
    #: With feedback the rule is to open only when the sensor proves the jaws are empty, and to
    #: hold and warn when it does not, rather than dropping an unknown workpiece wherever the arm
    #: happens to be. Without feedback "proven empty" is unreachable, so the same rule means not
    #: actuating. True gives the suction driver's behaviour instead: assert a known state on
    #: connect, accepting that a held part is released.
    open_on_connect_without_feedback: bool = False

    @model_validator(mode="after")
    def _check_actuation_pins(self) -> "JawIOGripperConfig":
        # A double solenoid with no open pin cannot open; it would latch closed on the first grasp
        # and never let go. Refuse at config load rather than at the first release.
        if self.actuation == "double_solenoid" and self.open_output_pin is None:
            raise ValueError(
                "gripper.jaw_io: actuation 'double_solenoid' needs `open_output_pin`; a bistable "
                "valve has no spring to open it, so without that pin the jaws could never release."
            )
        if self.actuation == "single_solenoid" and self.open_output_pin is not None:
            raise ValueError(
                "gripper.jaw_io: actuation 'single_solenoid' opens by dropping `close_output_pin`, "
                "so `open_output_pin` is never driven. Remove it, or set actuation to "
                "'double_solenoid' if the valve really has two coils."
            )
        pins = [self.close_output_pin, self.open_output_pin]
        if self.open_output_pin is not None and self.close_output_pin == self.open_output_pin:
            raise ValueError(
                f"gripper.jaw_io: close_output_pin and open_output_pin are both {pins[0]}; one pin "
                "cannot drive both coils."
            )
        return self


class OnRobotGripperConfig(StrictModel):
    """An OnRobot RG2 / RG6 reached over Modbus TCP through the OnRobot Compute Box.

    The Compute Box is a separate device with its own address, so :attr:`host` is config of its own
    rather than the ``robot.ur.ip`` the Robotiq branch reuses for the URCap daemon on the arm's
    controller. Pointing it at the arm addresses a machine that has never heard of the gripper.

    There is no speed field: RG2/RG6 carry no speed register anywhere in the writable map (force,
    width, control), and OnRobot's own library exposes only a read-only ``rg_get_speed``.

    RG2 and RG6 only. The 2FG7 shares the family name and not the register map, VG10/VGC10 are
    vacuum and 3FG15 is three-fingered; configuring one of those here drives unverified addresses.
    """

    #: The Compute Box's own IP, not the robot's, and not to be assumed. The factory default is
    #: 192.168.1.1, but the documented Dynamic IP mode only falls back to it after a 60-second DHCP
    #: timeout, so a box on a DHCP network can be anywhere. Read it from the Web Client.
    host: str = Field(default="192.168.1.1", min_length=1)
    #: Modbus TCP port on the Compute Box. 502 is OnRobot's documented value, alongside a limit
    #: of one concurrent connection.
    port: int = Field(default=502, ge=1, le=65535)
    #: Modbus unit id, which selects the tool and is chosen by the mounting, not by the gripper:
    #: 65 through a Quick Changer or a HEX-E/H QC, 66 for the primary side of a Dual Quick Changer
    #: and 67 for the secondary. An RG2-FT answers on 65 too with an incompatible map and Modbus
    #: alone cannot tell them apart, so confirm the model in the Web Client before commanding.
    unit_id: int = Field(default=65, ge=1, le=247)
    #: Gripping force in newtons, used when a caller passes none. Newtons are the native unit here:
    #: the register is tenths of a newton, unlike Robotiq's opaque 0-255 count whose physical span
    #: differs per model. The default 20 N is gentle against the RG2's range of roughly 3-40 N and
    #: the RG6's 25-120 N.
    default_force_n: float = Field(default=20.0, gt=0.0, le=200.0)
    #: Whether commanded widths are interpreted with the configured fingertip offset (control value
    #: 16) or without it (control value 1). Only meaningful when non-standard fingertips are fitted
    #: and their offset has been written to the gripper.
    use_fingertip_offset: bool = False


class GripperConfig(StrictModel):
    """Gripper vendor, physical opening limits and the per-vendor wiring blocks.

    ``vendor`` selects which driver the robot runtime instantiates and is validated at config load
    against the ``GripperVendor`` enum, so an unknown vendor is rejected there rather than at
    ``create_gripper``.
    """

    vendor: str = Field(default="robotiq", min_length=1)
    #: Physical opening of the mounted gripper, in mm. The default 85 mm is the Robotiq 2F-85,
    #: the end-effector this project ships. The Robotiq driver anchors its count map on this
    #: value, so it must be the real physical open width, not a policy ceiling.
    max_width_mm: float = Field(default=85.0, gt=0.0)
    #: Smallest meaningful grip, a policy floor, not the physical closed width (the 2F-85 closes to
    #: 0 mm). The driver's count map is anchored on 0, so this only clamps commanded widths.
    min_width_mm: float = Field(default=5.0, ge=0.0)
    #: The physical closed width: what ``get_width_mm()`` reads when the jaws are shut on nothing.
    #: 0.0 is the Robotiq 2F-85, fingers touching.
    #:
    #: Separate from ``min_width_mm``, and the separation is load-bearing.
    #: ``WidthDeltaGripperVerifier``, the only grasp verifier a jaw cell can use since the Robotiq
    #: driver exposes no object-detection capability, decides "the jaws collapsed on nothing" as
    #: ``post_close <= min_width + width_delta_min_mm`` and takes ``min_width`` from this field.
    #: The policy floor 5.0 in its place puts that threshold at 7 mm, where a genuinely held 6 mm
    #: part reports as an empty grasp; the physical 0.0 passes the same case. The count map in
    #: ``robotiq.py`` makes the same distinction and is anchored on the physical width too.
    closed_width_mm: float = Field(default=0.0, ge=0.0)
    #: Wiring for a suction end-effector; inert unless ``vendor == "vacuum"``.
    vacuum: VacuumGripperConfig = Field(default_factory=VacuumGripperConfig)
    #: Wiring for a parallel-jaw end-effector on the controller's digital I/O; inert unless
    #: ``vendor == "jaw_io"``. Separate from ``vacuum`` so a cell can declare both and swap the
    #: vendor string, which is exactly how a jaw/suction cell is commissioned.
    jaw_io: JawIOGripperConfig = Field(default_factory=JawIOGripperConfig)
    #: An OnRobot RG2/RG6 on a Compute Box; inert unless ``vendor == "onrobot"``. Separate from the
    #: two I/O blocks above so a cell can declare several end-effectors and commission by swapping
    #: the vendor string.
    onrobot: OnRobotGripperConfig = Field(default_factory=OnRobotGripperConfig)
    #: Flange -> grasp-centre transform for this end-effector. Hangs off the gripper, not the robot,
    #: because it is a property of what is bolted on, and this repo carries four different ones and
    #: swaps between two of them mid-run. The default ``source: undeclared`` is inert here and loads
    #: anywhere; the real-arm driver is what refuses it.
    tool_frame: ToolFrameConfig = Field(default_factory=ToolFrameConfig)

    @field_validator("vendor")
    @classmethod
    def _validate_gripper_vendor(cls, v: str) -> str:
        # Lazy import keeps the config layer free of robot-runtime imports at module load; the enum
        # is a pure StrEnum with no SDK behind it. Coerces case-insensitively, rejects unknowns.
        from src.robot.core.gripper_vendor import GripperVendor

        try:
            return GripperVendor.from_string(v).value
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    @model_validator(mode="after")
    def _check_ordering(self) -> GripperConfig:
        if self.min_width_mm >= self.max_width_mm:
            raise ValueError(
                f"min_width_mm ({self.min_width_mm}) must be < "
                f"max_width_mm ({self.max_width_mm})"
            )
        # ⛔ THE THIRD WIDTH JOINED THE MODEL AND NEVER JOINED THIS RULE. `closed_width_mm` anchors
        # the driver's count map: `_mm_to_count` spans `max_width_mm - closed_width_mm` and guards a
        # non-positive span with `else 0.0`, which does not raise -- it collapses the entire map onto
        # the closed end. MEASURED 2026-09-10 with closed 60.0 against max 50.0: 50 mm, 25 mm, 5 mm
        # and `open()` all came out as count 255, a full close at full speed, and the config loaded
        # clean. It is refused here rather than in the driver because a cell that cannot open its
        # hand is not a cell, and the earliest honest refusal is the cheapest one.
        if self.closed_width_mm >= self.max_width_mm:
            raise ValueError(
                f"closed_width_mm ({self.closed_width_mm}) must be < max_width_mm "
                f"({self.max_width_mm}): it is the PHYSICAL gap with the jaws shut, so a value at or "
                f"above the stroke leaves no travel to map. The driver's count map spans "
                f"max_width_mm - closed_width_mm and would collapse, turning every commanded width, "
                f"open() included, into a full close."
            )
        return self


class RobotConfig(StrictModel):
    """Top-level robot configuration tree.

    ``vendor`` selects which driver the robot runtime instantiates and defaults to ``ur``.
    Vendor-specific transport settings live in named sibling blocks (``ur``, ``sim``, ``dummy``,
    ``kuka``) so each driver owns its own typed surface: a UR cell's transport under ``ur``, a
    KUKA cell's controller address under ``kuka.controller_ip``. Only the block matching
    ``vendor`` is consulted at runtime; the others stay valid but unused.

    The config layer imports no robot runtime module, so validation works on hosts that only edit
    or lint YAML.
    """

    vendor: str = Field(default="ur", min_length=1)
    ur: URConfig = Field(default_factory=URConfig)
    sim: SimConfig = Field(default_factory=SimConfig)
    dummy: DummyConfig = Field(default_factory=DummyConfig)
    kuka: KukaConfig = Field(default_factory=KukaConfig)
    motion_limits: MotionLimitsConfig = Field(default_factory=MotionLimitsConfig)
    workspace_limits: WorkspaceLimitsConfig = Field(default_factory=WorkspaceLimitsConfig)
    gripper: GripperConfig = Field(default_factory=GripperConfig)
    #: Joint configuration the arm returns to on ``move_home()``, in radians. ``None`` uses
    #: :data:`HOME_JOINTS_DEFAULT`, which is a UR5e pose: on a UR3e it puts the grasp centre at
    #: z = 561.9 mm and r = 466.9 mm, 93.4% of that arm's 500 mm reach and past the 85% this project
    #: treats as near-singular, while a UR3e cell's own ``workspace_limits`` stop at z = 320. A cell
    #: on any other arm gives its own value, or its first motion leaves the declared workspace.
    #: ``sim.home_joint_positions`` is the sim-side twin of this field.
    home_joint_positions: tuple[float, ...] | None = None

    safe_pose: SafePoseConfig = Field(default_factory=SafePoseConfig)
    calibration: RobotCalibrationConfig = Field(default_factory=RobotCalibrationConfig)
    safety: RobotSafetyConfig = Field(default_factory=RobotSafetyConfig)
    grasping: RobotGraspingConfig = Field(default_factory=RobotGraspingConfig)
    # RL optimisation extension layer. The defaults, mode='hybrid_ml' with no artifact loaded,
    # run no RL at all and preserve production behaviour byte-identically. RL-active modes are
    # schema-gated and runtime-rejected; see src/robot/grasping/rl.
    rl: RobotRLConfig = Field(default_factory=RobotRLConfig)

    @model_validator(mode="after")
    def _check_self_collision_model_matches_sim_robot(self) -> "RobotConfig":
        """On an enabled sim cell, ``kinematics_model`` must equal ``sim.robot_model``.

        ``safety.self_collision.kinematics_model`` and ``sim.robot_model`` are independent
        hand-edited keys, and the self-collision guard feeds the first into the UR DH table
        (``safety/_ur_kinematics.py``) to derive per-link transforms. A cell naming a different
        model in each evaluates every motion against the other arm's link lengths (ur5e a2/a3 =
        -425/-392.2 mm vs ur3e -243.55/-213.2 mm) and returns wrong self-collision verdicts
        silently, so config load refuses it. Inert when the sim block is disabled, which is the
        real-robot config.
        """
        sc = self.safety.self_collision
        if self.sim.enabled and sc.kinematics_model and sc.kinematics_model != self.sim.robot_model:
            raise ValueError(
                f"safety.self_collision.kinematics_model={sc.kinematics_model!r} does not match "
                f"sim.robot_model={self.sim.robot_model!r}. The self-collision guard would derive link "
                "transforms from the wrong robot's DH chain. Set both to the same model."
            )
        return self

    @model_validator(mode="after")
    def _check_self_collision_model_matches_ur_robot(self) -> "RobotConfig":
        """The same coupling for a real UR cell: ``kinematics_model`` must equal ``ur.model``.

        Identical mechanism to the sim check above, with a physical arm on the other end: the guard
        would derive link transforms from another robot's DH chain and return wrong self-collision
        verdicts for real motion. Gated on ``vendor == "ur"``, so a sim or dummy cell, whose ``ur``
        block is inert boilerplate, is unaffected.
        """
        sc = self.safety.self_collision
        if self.vendor == "ur" and sc.kinematics_model and sc.kinematics_model != self.ur.model:
            raise ValueError(
                f"safety.self_collision.kinematics_model={sc.kinematics_model!r} does not match "
                f"ur.model={self.ur.model!r}. On real hardware the self-collision guard would derive link "
                "transforms from the wrong robot's DH chain. Set both to the same model."
            )
        return self

    @model_validator(mode="after")
    def _check_self_collision_model_is_meaningful_for_this_vendor(self) -> "RobotConfig":
        """``kinematics_model`` selects a UR DH table, so it means nothing on a non-UR, non-sim cell.

        The two rules above couple the key to ``sim.robot_model`` and to ``ur.model``; neither
        fires on a cell that is neither. Profiles compose, so that gap is reachable:
        ``WILLY_PROFILE=sim,web`` loads with ``vendor='kuka'`` and ``kinematics_model='ur5e'``, and
        the guard then evaluates a KUKA arm against UR5e link lengths.

        Only the bundled UR tables exist, so no value here is correct for another vendor. Unset is
        always accepted and leaves the guard on the capsule path, or failing closed if the mesh
        backend was demanded.
        """
        model = self.safety.self_collision.kinematics_model
        if model and self.vendor not in ("ur", "sim"):
            raise ValueError(
                f"safety.self_collision.kinematics_model={model!r} is set on a vendor={self.vendor!r} "
                f"cell. That key selects a Universal Robots DH table; there is none for this vendor, so "
                f"the self-collision guard would evaluate this arm against another robot's link lengths. "
                f"Leave it unset."
            )
        return self

    @model_validator(mode="after")
    def _check_safe_pose_in_workspace(self) -> "RobotConfig":
        """The safe (retreat) pose must lie inside ``workspace_limits``.

        The ``SafetyPreflight`` WorkspaceGuard rejects any motion whose target is outside
        ``workspace_limits``, so a safe_pose outside the box would make the retreat-to-safe motion
        itself fail closed. Only position is gated: :class:`SafePoseConfig` carries no orientation.
        """
        sp, wl = self.safe_pose, self.workspace_limits
        if not (wl.x_min <= sp.x <= wl.x_max
                and wl.y_min <= sp.y <= wl.y_max
                and wl.z_min <= sp.z <= wl.z_max):
            raise ValueError(
                f"safe_pose ({sp.x}, {sp.y}, {sp.z}) mm is outside workspace_limits "
                f"x[{wl.x_min}, {wl.x_max}] y[{wl.y_min}, {wl.y_max}] z[{wl.z_min}, {wl.z_max}]: "
                "the WorkspaceGuard would reject the retreat-to-safe motion."
            )
        return self

    @field_validator("vendor")
    @classmethod
    def _validate_vendor(cls, v: str) -> str:
        """Validate ``vendor`` against the ``RobotVendor`` enum at config load, so a typo like
        'ur5' is rejected here rather than late at ``create_arm``."""
        from src.robot.core.vendor import RobotVendor

        try:
            return RobotVendor.from_string(v).value
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

