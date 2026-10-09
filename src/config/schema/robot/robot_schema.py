"""Schemas for the vendor-selected robot configuration tree."""

from __future__ import annotations

import keyword
import math
import re
from typing import Final, Literal

from pydantic import Field, field_validator, model_validator

from .._base import StrictModel
from ..grippers.gripper_schema import MODEL_NAME_PATTERN

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
    home_in_radians,
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
    PlannerMeshConfig,
    PlanningWorldConfig,
    SelfCollisionSafetyConfig,
    SupportPlaneConfig,
)
from .tool_frame_schema import ToolFrameConfig
from .place_schema import PlaceGridConfig, RobotPlaceConfig

from .grasping_schema import (
    BlockerGraphSchemaConfig,
    GraspingDecisionConfig,
    GraspingFeasibilityConfig,
    GraspingOcclusionConfig,
    GraspingOrderingConfig,
    GraspingPerformanceConfig,
    GraspingRecoveryConfig,
    GraspingRecoveryFixtureConfig,
    GraspingSuccessModelConfig,
    GraspingUncertaintyConfig,
    GraspingWatchdogConfig,
    RobotGraspingApproachValidationConfig,
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
    "GraspingDecisionConfig",
    "GraspingFeasibilityConfig",
    "GraspingOcclusionConfig",
    "GraspingOrderingConfig",
    "GraspingPerformanceConfig",
    "GraspingRecoveryConfig",
    "GraspingRecoveryFixtureConfig",
    "GraspingSuccessModelConfig",
    "GraspingUncertaintyConfig",
    "GraspingWatchdogConfig",
    "GripperConfig",
    "IkQualitySafetyConfig",
    "JointLimitSafetyConfig",
    "KukaConfig",
    "KukaEkiConfig",
    "LimitsSafetyConfig",
    "MotionContinuitySafetyConfig",
    "MotionLimitsConfig",
    "RobotMotionConfig",
    "NamedPoseConfig",
    "POSE_LABEL_MAX_CHARS",
    "POSE_NAME_MAX_CHARS",
    "PayloadSafetyConfig",
    "PlaceGridConfig",
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
    "RobotGraspingConfig",
    "RobotGraspingFusionConfig",
    "RobotPlaceConfig",
    "RobotRLConfig",
    "RobotSafetyConfig",
    "SafePoseConfig",
    "AttachedPayloadConfig",
    "PerceivedWorldConfig",
    "PlannerMeshConfig",
    "PlanningWorldConfig",
    "SelfCollisionSafetyConfig",
    "SupportPlaneConfig",
    "SimCameraSchema",
    "SimConfig",
    "SimGateConfig",
    "SimMarkerConfig",
    "SimObjectConfig",
    "SimSceneConfig",
    "SimTableConfig",
    "URConfig",
    "OnRobotGripperConfig",
    "VacuumGripperConfig",
    "UncertaintyChannelWeightsConfig",
    "WorkspaceLimitsConfig",
    "pose_label_refusal",
    "pose_name_refusal",
    "pose_words",
]


class MotionLimitsConfig(StrictModel):
    """Hard upper bounds on commanded velocity and acceleration."""

    max_velocity: float = Field(default=1.0, gt=0.0, le=3.14)
    max_acceleration: float = Field(default=0.5, gt=0.0, le=5.0)


class RobotMotionConfig(StrictModel):
    """When the next leg of a pick or a place is judged: as it runs, or ahead while the arm waits or moves.

    ``judge_next_leg`` (map4 motion C7, the owner, 2026-10-09):

    * ``"off"``, the default and today's behaviour: every leg is judged when the arm stands at its start, and the jaws'
      stroke is waited out before anything else is asked.
    * ``"in_settles"``: the change of the jaws goes out once, as ever, and while their stroke is waited out the next leg
      is judged: the line out of a place, and, in a world held across the stroke (``safety.planning_world.hold``), the
      joint move a task declared after the pick or the place. Nothing is sent before the stroke is over.
    * ``"in_settles_and_motion"``: as ``"in_settles"``, and a declared joint move not judged in a stroke is judged on a
      second thread while the line before it runs, on an arm that watches its sends (``robot.ur.brake_on_halt``: a halt
      brakes the line in flight). Elsewhere it falls back to ``"in_settles"`` and says so once in the log.

    A leg judged ahead runs only where the arm stands within 0.5 mm of where it was judged from and nothing else it was
    judged on changed since (the world's refresh, the carried part, the hand, the camera's boxes, a halt); otherwise it
    is judged again where the arm stands. Judging ahead sends nothing and switches no output.
    """

    judge_next_leg: Literal["off", "in_settles", "in_settles_and_motion"] = Field(default="off")


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


#: How many digital pins the UR tool connector has each way: outputs 0 and 1, inputs 0 and 1. The standard and
#: configurable banks in the control box have 0 to 7 (``src/robot/drivers/ur/connection.py``, which reads the tool
#: bank back at global ids 16 and 17).
_TOOL_BANK_PINS = 2

#: The shortest pulse the load accepts where a pulse is what moves the jaws (``double_solenoid``; a ``single_toggle`` is
#: switched, not pulsed): several controller cycles, 8 ms each on a CB3. The driver built directly takes any pulse; this
#: floor is the config's.
_MIN_PULSE_S = 0.05


def _refuse_pins_the_tool_bank_lacks(block: str, io_port: str, pins: "dict[str, int | None]") -> None:
    """Refuse a pin on ``io_port: tool`` that the UR tool connector does not have: it has pins 0 and 1 each way.

    The fields allow 0 to 7 because the standard and configurable banks have eight pins each,
    and ``tool`` is the default bank, so a pin number measured in the control box and left on
    the default loaded, and every write to it was a pin the wrist does not have (review of
    2026-09-24). Refused at load, rather than at the first write of a program already moving.
    """
    if io_port != "tool":
        return
    for name, pin in pins.items():
        if pin is not None and pin >= _TOOL_BANK_PINS:
            kind = "outputs" if "output" in name else "inputs"
            raise ValueError(
                f"gripper.{block}: `{name}: {pin}` is on io_port 'tool', and the UR tool connector has digital "
                f"{kind} 0 and 1 only (src/robot/drivers/ur/connection.py), so {pin} is no pin on that bank. Use 0 "
                "or 1, or set io_port to 'standard' or 'configurable' if the wire lands in the control box."
            )


class VacuumGripperConfig(StrictModel):
    """Wiring and timing for a suction end-effector on the controller's digital I/O.

    Consulted only when ``gripper.vendor == "vacuum"``. Every field is a number measured on the
    actual cell: which pin the ejector is on, which bank it is in, whether a vacuum switch is
    wired, how long that ejector takes to build a seal. The defaults describe the common case
    (tool I/O, pin 0, no switch) and are not a claim about any particular cell.
    """

    #: Output pin that switches the ejector or pump on. Every pin here is 0 to 7 on the standard and
    #: configurable banks, and 0 or 1 on the tool bank, the default, which has two pins each way.
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

    @model_validator(mode="after")
    def _check_tool_bank_pins(self) -> "VacuumGripperConfig":
        _refuse_pins_the_tool_bank_lacks("vacuum", self.io_port, {
            "vacuum_output_pin": self.vacuum_output_pin,
            "blow_off_output_pin": self.blow_off_output_pin,
            "vacuum_ok_input_pin": self.vacuum_ok_input_pin,
        })
        return self


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
    #:
    #: ``single_toggle``: one output, where every change of it moves the jaws once, switched on as
    #: much as switched off (the owner's Hand-E on the Robotiq I/O Coupling, confirmed at the pendant
    #: 2026-09-28), closed to open and back, and nothing is read back (a feedback input is refused).
    #: A command is one change, left where it went; nothing is pulsed. The program counts its own
    #: changes from where a person says the jaws stand when the gripper connects, before anything
    #: moves, and the connect writes nothing; nothing is kept between programs. It switches only
    #: where its count says the jaws stand the other way from the request, never before the arm
    #: moves at the start of a pick, and refuses a command on an output somebody switched by hand
    #: since the last one. It takes no width: ``closed_below_mm`` does not apply to it.
    actuation: Literal["single_solenoid", "double_solenoid", "single_toggle"] = "single_solenoid"
    #: Output that closes the jaws: held high while closed under ``single_solenoid``, pulsed
    #: under ``double_solenoid``, and under ``single_toggle`` the one pin, switched once to close
    #: and once to open. Every pin here is 0 to 7 on the standard and configurable banks, and 0 or 1 on the
    #: tool bank, the default, which has two pins each way.
    close_output_pin: int = Field(default=0, ge=0, le=7)
    #: Output that opens the jaws. Required for ``double_solenoid``; unused (and must stay unset)
    #: for ``single_solenoid``, where "open" is simply dropping ``close_output_pin``, and for
    #: ``single_toggle``, where "open" is the next change of it.
    open_output_pin: int | None = Field(default=None, ge=0, le=7)
    #: Pulse length for ``double_solenoid``. A bistable valve latches, so the coil is energised
    #: only long enough to throw it; holding it high is what cooks the coil. At least 0.05 s,
    #: several controller cycles (8 ms each on a CB3): the controller applies an output once a
    #: cycle, so a shorter pulse may never reach the device. Unused by ``single_solenoid``, which
    #: holds a level, and by ``single_toggle``, which changes its output once per command and
    #: never pulses (a pulse moved the owner's jaws twice).
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
    #: grasp, not a fault; the execution policy's post-close hold check decides, exactly as for
    #: suction.
    close_timeout_s: float = Field(default=1.0, gt=0.0, le=30.0)
    #: The jaws' travel time, both ways. With no feedback pin wired it is the whole wait: there is nothing to poll,
    #: so the driver can only wait. With one, a reading that also occurs mid-stroke is taken only after it. An open
    #: with no open switch waits it too, so a place does not back out of jaws still opening. A ``single_toggle``
    #: waits it after every change, and after the change a person asks for at connect.
    close_settle_s: float = Field(default=0.3, ge=0.0, le=10.0)
    #: Commanded width at/below which the driver closes. Mirrors ``vacuum_on_below_mm`` so both I/O
    #: end-effectors interpret the width-based Protocol identically. Unused by ``single_toggle``,
    #: which refuses a width.
    closed_below_mm: float = Field(default=5.0, gt=0.0)
    #: What ``connect()`` does when no feedback is wired. The default, False, is not to actuate.
    #:
    #: With feedback the rule is to open only when the sensor proves the jaws are empty, and to
    #: hold and warn when it does not, rather than dropping an unknown workpiece wherever the arm
    #: happens to be. Without feedback "proven empty" is unreachable, so the same rule means not
    #: actuating. True gives the suction driver's behaviour instead: assert a known state on
    #: connect, accepting that a held part is released. Refused for ``single_toggle``, which cannot
    #: assert a state, only move the jaws the other way.
    open_on_connect_without_feedback: bool = False
    #: Whether connecting asks a person, before anything moves, whether the jaws stand open. Closed
    #: is answered with one command that opens them, or an abort that refuses the connect, and with
    #: no terminal to ask at the connect is refused. ``None``, the default, asks for
    #: ``single_toggle``, which nothing else can tell where its jaws stand, and not for the
    #: solenoids; ``true`` asks for a solenoid too. ``false`` is refused for ``single_toggle``.
    confirm_open_at_start: bool | None = None

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
        if self.actuation == "single_toggle" and self.open_output_pin is not None:
            raise ValueError(
                "gripper.jaw_io: actuation 'single_toggle' opens by the next change of "
                "`close_output_pin`, so `open_output_pin` is never driven. Remove it, or set actuation "
                "to 'double_solenoid' if the valve really has two coils."
            )
        # A toggle can only move the jaws the other way, not put them somewhere: a change on jaws that
        # already stand open closes them, so asserting open on connect is the one thing it must not do.
        if self.actuation == "single_toggle" and self.open_on_connect_without_feedback:
            raise ValueError(
                "gripper.jaw_io: actuation 'single_toggle' cannot assert 'open' on connect: every change "
                "of its output moves the jaws, so one on jaws that already stand open would close them. Remove "
                "`open_on_connect_without_feedback`; connect() asks a person where the jaws stand instead."
            )
        # A toggle reads nothing back: the program counts its changes from where a person said the jaws
        # stood, so an input is a wire nobody reads, and a cell that believes it is sensed is not.
        if self.actuation == "single_toggle":
            wired = [name for name in ("part_present_input_pin", "closed_confirm_input_pin", "open_confirm_input_pin")
                     if getattr(self, name) is not None]
            if wired:
                raise ValueError(
                    f"gripper.jaw_io: actuation 'single_toggle' reads no sensor, so `{wired[0]}` would be a wire "
                    "nobody reads: the program counts its own changes from where a person said the jaws stood at "
                    "connect. Remove it, or set actuation to 'single_solenoid' or 'double_solenoid' if the valve "
                    "really holds a level or has two coils."
                )
            if self.confirm_open_at_start is False:
                raise ValueError(
                    "gripper.jaw_io: actuation 'single_toggle' always asks at connect where its jaws stand: nothing "
                    "else can tell the program, and a count started from a wrong guess inverts every command after "
                    "it. Remove `confirm_open_at_start: false`."
                )
        pins = [self.close_output_pin, self.open_output_pin]
        if self.open_output_pin is not None and self.close_output_pin == self.open_output_pin:
            raise ValueError(
                f"gripper.jaw_io: close_output_pin and open_output_pin are both {pins[0]}; one pin "
                "cannot drive both coils."
            )
        _refuse_pins_the_tool_bank_lacks("jaw_io", self.io_port, {
            "close_output_pin": self.close_output_pin,
            "open_output_pin": self.open_output_pin,
            "part_present_input_pin": self.part_present_input_pin,
            "closed_confirm_input_pin": self.closed_confirm_input_pin,
            "open_confirm_input_pin": self.open_confirm_input_pin,
        })
        # Where a pulse is what moves the jaws, a pulse of a controller cycle or two may never reach the device, or
        # reach it too short to register (review of 2026-09-24). A toggle is switched, not pulsed, and never reads it.
        if self.actuation == "double_solenoid" and self.pulse_s < _MIN_PULSE_S:
            raise ValueError(
                f"gripper.jaw_io: `pulse_s: {self.pulse_s}` is shorter than {_MIN_PULSE_S} s, the shortest pulse the "
                f"load accepts for actuation '{self.actuation}', where the pulse is what moves the jaws: the "
                "controller applies an output once per controller cycle (8 ms on a CB3) and reports it back a cycle "
                "or two later, so a pulse of a few cycles may never reach the device, or reach it too short to "
                f"register. Set pulse_s to at least {_MIN_PULSE_S}, and measure the pulse the device needs with "
                "`python -m src.robot.drivers.ur --pulse PIN --for SECONDS --yes`: one call must throw the valve once."
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


class CouplingPlateConfig(StrictModel):
    """One plate between the arm flange and the hand's own mounting face: what holds the hand out, and what it is.

    The thickness places the hand and the cross section collides, and neither number is written twice. The
    stack's thicknesses sum to where the hand sits. The cross section is the two half extents across the
    approach, so a 90 mm quick change coupler is ``[45.0, 45.0]``.

    A plate without a cross section becomes no body, and the cell says its name rather than inventing a
    width. Measured by ``scripts/curobo/probe_plate_body.py``: the Hand-E's 20 mm plate over the 1,483 judged
    poses, against the exact guard's 10 mm, turns 0 to 2 of about 950 clear poses per arm at a UR flange
    radius, the worst grazing at 8.944 mm; at a coupler's 45 mm it is 0 to 25. So an unmeasured plate is a
    small hole and an unmeasured coupler is not, and the difference is what the caller declares.
    """

    #: What it is, so a cell that declares no cross section can be told which plate is missing one.
    name: str = Field(min_length=1)
    #: How far it holds the hand out along the approach, in millimetres. It is the same bench
    #: measurement as the plate term in ``tool_frame.offset_mm``.
    thickness_mm: float = Field(gt=0.0)
    #: The two half extents across the approach, in the hand model's own axes (closing, then
    #: binormal). ``None``, the default, is a plate nobody measured across: it still holds the hand out
    #: and it becomes no body.
    cross_section_mm: tuple[float, float] | None = Field(default=None)

    @model_validator(mode="after")
    def _check_cross_section(self) -> CouplingPlateConfig:
        if self.cross_section_mm is not None and min(self.cross_section_mm) <= 0.0:
            raise ValueError(
                f"plate {self.name!r}: cross_section_mm holds the two half extents across the approach and both "
                f"must be positive; got {list(self.cross_section_mm)}. A plate of no width is not a body, and a "
                f"plate nobody measured across declares no cross section at all rather than zero."
            )
        return self


class GripperConfig(StrictModel):
    """Gripper vendor, physical opening limits and the per-vendor wiring blocks.

    ``vendor`` selects which driver the robot runtime instantiates and is validated at config load
    against the ``GripperVendor`` enum, so an unknown vendor is rejected there rather than at
    ``create_gripper``.
    """

    vendor: str = Field(default="robotiq", min_length=1)
    #: The registry name of the hand bolted on: the stem of a file under ``config/grippers/``
    #: (``robotiq_2f85``), loaded by ``src.config.grippers.load_gripper``. ``None``, the default,
    #: changes nothing. A set value is lower case letters, digits and underscores, so it can never read
    #: as a profile overlay or reach outside the registry. A name no registry file defines still
    #: passes the schema, which is validated without a data directory, and the loader refuses it. The
    #: loader also fills the hand's widths and collision envelope from its file where the profile
    #: chain leaves them unset (``src/config/hand_numbers.py``). The guard loads its mesh bundle by
    #: this name (``safety.planning.hand.planner_hand``), and a cell whose guard reads hand geometry
    #: refuses to build while it is unset.
    model: str | None = Field(default=None, pattern=MODEL_NAME_PATTERN)
    #: The plates between the arm flange and the hand's own mounting face. Their thicknesses sum to
    #: where a hand whose sphere map starts at its mounting face sits on the flange, and the self
    #: collision guard adds that sum to the hand's meshes. ``None``, the default, is no measurement:
    #: such a hand refuses at build until the plates are written, and ``[]`` says it is bolted
    #: straight to the flange. A hand whose map already sits at the flange (the 2F-85, placed by its
    #: arm asset) refuses any plate.
    #:
    #: Each plate also carries what it is, where somebody measured it: a plate with a
    #: ``cross_section_mm`` becomes collision geometry through the declared body writer, and one
    #: without becomes a named gap instead. The thickness is written once, in the plate, so it is
    #: never checked against a second statement of itself.
    coupling_plates: list[CouplingPlateConfig] | None = Field(default=None)
    #: Physical opening of the mounted gripper, in mm. The default 85 mm is the Robotiq 2F-85,
    #: the end-effector this project ships. The Robotiq driver anchors its count map on this
    #: value, so it must be the real physical open width, not a policy ceiling. A cell that names a
    #: hand takes it from the hand's ``jaw.aperture_mm`` unless it states it, and a stated value that
    #: differs is refused.
    max_width_mm: float = Field(default=85.0, gt=0.0)
    #: Smallest meaningful grip, a policy floor, not the physical closed width (the 2F-85 closes to
    #: 0 mm). The driver's count map is anchored on 0, so this only clamps commanded widths. A cell
    #: that names a hand takes the hand's floor unless it states one between that floor and the
    #: aperture.
    min_width_mm: float = Field(default=5.0, ge=0.0)
    #: The physical closed width: what ``get_width_mm()`` reads when the jaws are shut on nothing.
    #: 0.0 is the Robotiq 2F-85, fingers touching. A cell that names a hand takes the hand's own,
    #: and a stated value that differs is refused.
    #:
    #: Separate from ``min_width_mm``, and the separation is load-bearing: the count map in
    #: ``robotiq.py`` is anchored on this physical width, and anchoring it on the policy floor
    #: instead would move every commanded width off the mark. The width-delta grasp verifier
    #: that also read it was removed on 2026-09-29 with the post-grasp verification stage;
    #: whether a close holds a part is the gripper's own hold evidence, read by the execution
    #: policy after every close.
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

    @property
    def coupling_mm(self) -> float | None:
        """How far the plate stack holds the hand off the flange, or ``None`` where nobody has measured it.

        The one place the stack is summed, so a reader never adds the thicknesses itself and no second
        answer can exist. ``None`` and ``0.0`` are different answers: undeclared against bolted straight on.
        """
        if self.coupling_plates is None:
            return None
        return float(sum(plate.thickness_mm for plate in self.coupling_plates))

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
        # The third width belongs in this rule too. `closed_width_mm` anchors the driver's count map:
        # `_mm_to_count` spans max_width_mm minus closed_width_mm and guards a non-positive span with
        # `else 0.0`, which does not raise but collapses the entire map onto the closed end. With
        # closed 60.0 against max 50.0, 50 mm, 25 mm, 5 mm and `open()` all came out as count 255, a
        # full close at full speed, while the config loaded clean. It is refused here rather than in
        # the driver because a cell that cannot open its hand is not a cell, and the earliest refusal
        # is the cheapest one.
        if self.closed_width_mm >= self.max_width_mm:
            raise ValueError(
                f"closed_width_mm ({self.closed_width_mm}) must be < max_width_mm "
                f"({self.max_width_mm}): it is the physical gap with the jaws shut, so a value at or "
                f"above the stroke leaves no travel to map. The driver's count map spans "
                f"max_width_mm - closed_width_mm and would collapse, turning every commanded width, "
                f"open() included, into a full close."
            )
        # A jaw_io gripper reads every commanded width as open or closed, closed at or below closed_below_mm. A release
        # and a pick's pre-open command max_width_mm and a grasp commands at least min_width_mm, so a threshold outside
        # that band makes every release close the jaws, or leaves no grasp that closes them. With 84, the 2F-85's
        # number, on a Hand-E's 49.99 a pick closes the jaws at its pre-open, the symptom the owner's toggle cell
        # reported on 2026-09-23. A single_toggle takes no width at all, so the band does not apply to it.
        if self.vendor == "jaw_io" and self.jaw_io.actuation != "single_toggle":
            below = self.jaw_io.closed_below_mm
            if below >= self.max_width_mm:
                raise ValueError(
                    f"gripper.jaw_io.closed_below_mm ({below}) must be < max_width_mm ({self.max_width_mm}): jaw_io "
                    f"closes at or below it, and a release and a pick's pre-open command max_width_mm, so every "
                    f"release would close the jaws. Set it just below max_width_mm, above the widest part less the "
                    f"squeeze."
                )
            if below < self.min_width_mm:
                raise ValueError(
                    f"gripper.jaw_io.closed_below_mm ({below}) must be >= min_width_mm ({self.min_width_mm}): a pick "
                    f"grasps at min_width_mm or wider, so below it a pick's grasp would never close the jaws."
                )
        return self


#: The largest joint value a look may name, degrees either way: a full turn. No joint of a cell is taught past it, and a
#: look past it is a typo or another unit, refused at load rather than driven to.
LOOK_MAX_ABS_DEG: Final[float] = 360.0

#: The longest name of a taught pose, in characters. The name is the YAML key and the word the command reader hands on.
POSE_NAME_MAX_CHARS: Final[int] = 32
#: The longest label of a taught pose, in characters: what the chat and the cards say ("Ablage links").
POSE_LABEL_MAX_CHARS: Final[int] = 40
#: A comment mark where the guided writer looks for one: after whitespace. A label holding one would be cut there by a
#: later rewrite of its line (``src/config/edit.py``, ``_rewrite``).
_COMMENT_MARK = re.compile(r"\s#")
#: The words the loader's YAML (PyYAML's ``SafeLoader``) reads as a boolean or as nothing, refused in any case. The pose
#: door writes a pose's name bare, as its key (``yes:``), so a pose under one of them would read back as ``True``,
#: ``False`` or ``None`` and could never be written: the arm would have been freed for nothing.
_YAML_WORDS: Final[frozenset[str]] = frozenset({"yes", "no", "true", "false", "on", "off", "null"})


def pose_name_refusal(name: object) -> str:
    """Why ``name`` cannot name a pose taught in the console, or ``""`` where it can.

    A name is the YAML key under ``robot.named_poses`` and the word the command reader hands back for the label a
    person said, so it is an ASCII identifier (letters, digits and underscores, not starting with a digit) of at most
    :data:`POSE_NAME_MAX_CHARS` characters, no Python keyword, and never ``home`` in any case: Home is the arm's own
    ``robot.home_joint_positions``, configured there and taught nowhere else. It is no word YAML reads as true, false
    or null (``yes``, ``off``, ``null``, in any case: the key would read back as no name), and it does not have the
    shape of the loader's own words (``__null__``, the overlay reset, which would turn a default place naming it into
    none). The one rule the schema, the pose writer and teaching read (``teach.name_refusal``).
    """
    if not isinstance(name, str) or not name:
        return "a pose needs a name: an ASCII identifier such as drop_left"
    if not (name.isascii() and name.isidentifier()):
        return (f"{name!r} is no pose name: a name is an ASCII identifier, letters, digits and underscores, not "
                "starting with a digit (drop_left, park_2)")
    if len(name) > POSE_NAME_MAX_CHARS:
        return f"{name!r} is longer than the {POSE_NAME_MAX_CHARS} characters a pose name may have"
    if keyword.iskeyword(name):
        return f"{name!r} is a Python keyword, and a pose name is none"
    if name.lower() == "home":
        return (f"{name!r} is Home, the arm's own home pose (robot.home_joint_positions): it is configured there and "
                "taught nowhere else")
    if name.casefold() in _YAML_WORDS:
        return (f"{name!r} is a word YAML reads as true, false or null, in any case, not as a name: the pose door "
                "writes the name as the pose's key, and the pose would read back under none. Choose another name")
    if name.startswith("__") and name.endswith("__"):
        return (f"{name!r} has the shape of the config loader's own words (__null__ resets a value in an overlay), "
                "and a pose name is none of them")
    return ""


def pose_words(name: str, label: str = "") -> frozenset[str]:
    """The words a pose answers to when a person names it: its name and its label, each without case and with every run
    of whitespace one space.

    The command reader is handed the poses by their labels, an unlabelled pose by its name, and a person may say
    either, so no two poses of a tree share a word: :class:`RobotConfig` refuses two that do at load, and the console's
    store refuses a pose that would before the arm is freed.
    """
    return frozenset(word for word in (_folded(name), _folded(label)) if word)


def _folded(text: str) -> str:
    """``text`` as a person's word is compared: without case, every run of whitespace one space."""
    return " ".join(str(text).split()).casefold()


def pose_label_refusal(label: object) -> str:
    """Why ``label`` cannot be what the chat, the cards and the command reader call a taught pose, or ``""``.

    Free text, Unicode and spaces allowed, of at most :data:`POSE_LABEL_MAX_CHARS` characters, holding no newline or
    other character a line cannot show and no ``" #"``: the guided writer rewrites a pose's lines in place, and it takes
    a ``#`` after whitespace for the start of a comment. An empty label is refused here, where the console writes one; a
    pose written by hand may leave its label out, and its name is said instead.
    """
    if not isinstance(label, str) or not label.strip():
        return "a pose needs a label: what the chat and the cards call it, such as Ablage links"
    if len(label) > POSE_LABEL_MAX_CHARS:
        return f"the label {label!r} is longer than the {POSE_LABEL_MAX_CHARS} characters a label may have"
    if _COMMENT_MARK.search(label):
        return (f"the label {label!r} holds a '#' after a space, which a later rewrite of its line would take for the "
                "start of a comment")
    if not label.isprintable():
        return f"the label {label!r} holds a newline or another character a line cannot show"
    return ""


class NamedPoseConfig(StrictModel):
    """A pose taught by hand in the console: where a task puts its part, or where it returns to (owner decision 15).

    The console frees the arm, a person guides it there, and the pose is screened by the exact guard and the planner
    while the arm holds; only a pose both clear, or one in the planner's cushion band, is written, into the last layer
    of the profile chain (the owner, Q10). A place pose says where the part's BOTTOM is let go: a task raises the tool
    over it by the part's hang. The name it is kept under (``robot.named_poses.<name>``) follows
    :func:`pose_name_refusal`.
    """

    #: The joints in DEGREES, one value per joint, as the pendant shows them: the looks' rules hold (each a finite
    #: number of at most a full turn either way, as many as the home names, and as many as every other pose names).
    joints_deg: tuple[float, ...]
    #: What the chat, the cards and the command reader call the pose ("Ablage links"): free text of at most 40
    #: characters, no newline and no " #" (:func:`pose_label_refusal`), and no word another pose answers to
    #: (:func:`pose_words`: its name, or its label). Empty: the name is said.
    label: str = ""
    #: When it was taught, the local time with its offset (ISO 8601), as the console wrote it; ``None`` for a pose
    #: written by hand.
    taught_at: str | None = None
    #: What the exact guard and the planner said when it was taught: ``clear`` (both clear it) or ``band`` (in the
    #: planner's cushion band: straight lines run into it and out of it, and a planned move takes a short leg first).
    #: An unscreened or refused pose is never written. ``None``: written by hand and not screened in the console; a
    #: task screens it before anything moves. Layers merge a pose KEY BY KEY, as every mapping of the tree: a pose a
    #: later layer names with its joints alone keeps the screen a lower layer gave other joints, so a screen read off a
    #: tree proves nothing about the joints beside it, and a task screens every pose before it moves. The pose door
    #: writes all five keys at once, so a pose it wrote never mixes layers.
    screen: Literal["clear", "band"] | None = None
    #: The screen's own words, for a band pose; free text, for a person reading the YAML.
    note: str = ""

    @field_validator("label")
    @classmethod
    def _label_is_a_line(cls, value: str) -> str:
        """A label written must be one a line can hold and a rewrite keeps; an empty one is no label."""
        refused = pose_label_refusal(value) if value else ""
        if refused:
            raise ValueError(refused)
        return value


def _called(pose: NamedPoseConfig) -> str:
    """What a refusal says a pose is called: its label as written, or that it has none (its name is said)."""
    return f"label {pose.label!r}" if pose.label else "no label"


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
    #: When the next leg of a pick or a place is judged (:class:`RobotMotionConfig`).
    motion: RobotMotionConfig = Field(default_factory=RobotMotionConfig)
    workspace_limits: WorkspaceLimitsConfig = Field(default_factory=WorkspaceLimitsConfig)
    gripper: GripperConfig = Field(default_factory=GripperConfig)
    #: Joint configuration the arm returns to on ``move_home()``, in radians. ``None`` uses
    #: :data:`HOME_JOINTS_DEFAULT`, which is a UR5e pose: on a UR3e it puts the grasp centre at
    #: z = 561.9 mm and r = 466.9 mm, 93.4% of that arm's 500 mm reach and past the 85% this project
    #: treats as near-singular, while a UR3e cell's own ``workspace_limits`` stop at z = 320. A cell
    #: on any other arm gives its own value, or its first motion leaves the declared workspace.
    #: ``sim.home_joint_positions`` is the sim-side twin of this field.
    home_joint_positions: tuple[float, ...] | None = None
    #: The same home in DEGREES, as the pendant shows it and ``python -m src.robot.drivers.ur --where`` prints it.
    #: Exactly one of the two may be given. Read once, at load: the loaded config carries the home in radians in
    #: ``home_joint_positions`` and ``None`` here, so every consumer reads the one unit it always read.
    home_joint_positions_deg: tuple[float, ...] | None = None
    #: Where the camera looks from before each pick when its program names no look: one list per look, in DEGREES,
    #: one value per joint, as the pendant shows them and ``python -m src.robot.drivers.ur --where`` prints them,
    #: visited in this order. A wrist camera's pick fuses each look with the ones before it and stops at the first
    #: whose grasp is safe. A fixed camera's arm is moved to them too, before it perceives, and its pick stops at the
    #: first look that finds something. Kept in degrees in the loaded config, and read once by the pick service into
    #: joint positions (``configured_looks``); there is deliberately no radians twin, because a look is where the arm
    #: goes next and its unit must be said. A program's own looks override these. ``None`` configures none: a wrist
    #: camera then looks from home, and a fixed camera does not move to look.
    look_joint_positions_deg: tuple[tuple[float, ...], ...] | None = None
    #: How the hand and its camera naturally stand: the direction the jaws close along where nothing else says (the
    #: owner's decision, 2026-09-30). A name of ``CLOSING_AXES``, the names ``Pose.tool_down`` takes (``x``, ``-x``,
    #: ``y``, ``-y``, ``radial``, ``-radial``, ``tangential``, ``-tangential``, a leading "+" allowed; ``radial`` and
    #: ``tangential`` read at each place), or a quaternion ``[x, y, z, w]``, as a taught pose gives it, whose tool +X
    #: laid onto the base XY plane is that direction: only its heading counts, not its pitch out of the horizontal.
    #: Camera grasps stay free, any closing direction and any tilt: of a grasp's two equivalent wrist turns, half a turn
    #: about its approach apart (the same two contact faces, the jaws swapped), the pick, its push, the Locator and
    #: ``Scene.grasps`` take the one whose tool +X lies nearer this direction at the grasp's place; none is left out or
    #: tilted, and a program's own ``closing_axis`` takes precedence. Beside the simulator's twist
    #: (``GraspMotion(align_closing_to_base_x=True)``), which chooses the way round itself, it turns no grasp (a push
    #: still takes the natural way round). A pose built
    #: through the cell, ``robot.tool_down(x, y, z)``, closes along it where the program names no axis. ``None``, the
    #: default, turns no grasp, and ``robot.tool_down`` closes along ``x``, as ``Pose.tool_down`` does. Refused at load:
    #: an unknown name, a quaternion that is none, and one whose tool +X stands within 10 degrees of the vertical, which
    #: names no direction.
    natural_closing_axis: str | tuple[float, float, float, float] | None = None
    #: Poses taught by hand in the console (:class:`NamedPoseConfig`), under their NAMES: a place a task puts its part
    #: at, or a pose it returns to. Written by the console's pose writer alone, into the last layer of the profile
    #: chain, and only once the exact guard and the planner cleared the pose or found it in the planner's band; the
    #: console never renames or deletes one (edit them here). Refused at load: a name that is no pose name
    #: (``pose_name_refusal``: ``home`` among them), joints that break the looks' rules, and two poses under one label.
    #: Home stays ``home_joint_positions``.
    named_poses: dict[str, NamedPoseConfig] = Field(default_factory=dict)
    #: The pose a task places at when its command names no target: a name in ``named_poses``, or ``None`` for none, and
    #: a task that names no place is then refused. Chosen in the console.
    default_place_pose: str | None = None
    #: How a task sets its parts down (:class:`RobotPlaceConfig`): into a box below its rim, side by side on a flat
    #: place, the hang from what the pick measured, the carry over the rim. Every default is what a task did before.
    place: RobotPlaceConfig = Field(default_factory=RobotPlaceConfig)

    safe_pose: SafePoseConfig = Field(default_factory=SafePoseConfig)
    calibration: RobotCalibrationConfig = Field(default_factory=RobotCalibrationConfig)
    safety: RobotSafetyConfig = Field(default_factory=RobotSafetyConfig)
    grasping: RobotGraspingConfig = Field(default_factory=RobotGraspingConfig)
    # RL optimisation extension layer. The defaults, mode='hybrid_ml' with no artifact loaded,
    # run no RL at all and preserve production behaviour byte-identically. RL-active modes are
    # schema-gated and runtime-rejected; see src/robot/grasping/rl.
    rl: RobotRLConfig = Field(default_factory=RobotRLConfig)

    @model_validator(mode="after")
    def _home_in_degrees_loads_as_radians(self) -> "RobotConfig":
        """``home_joint_positions_deg`` becomes ``home_joint_positions`` in radians; both at once is refused.

        First among the validators, so every rule after it reads the home in the unit it always read.
        """
        home = home_in_radians(self.home_joint_positions, self.home_joint_positions_deg, "robot.home_joint_positions")
        if self.home_joint_positions_deg is not None:
            object.__setattr__(self, "home_joint_positions", home)
            object.__setattr__(self, "home_joint_positions_deg", None)
            # Stated as the radians it now is, so what the model says was set matches what it holds.
            stated = (self.model_fields_set - {"home_joint_positions_deg"}) | {"home_joint_positions"}
            object.__setattr__(self, "__pydantic_fields_set__", stated)
        return self

    @model_validator(mode="after")
    def _looks_are_joints_in_degrees(self) -> "RobotConfig":
        """``look_joint_positions_deg`` names at least one look, each a finite joint vector in degrees, all one length.

        Refused at load: a list that names no look (leave the key out instead), a look that names no joint or holds a
        value that is not a finite number, looks of different lengths, a length unlike the home's where the home is
        set (after the home above is read in radians), and a value past a full turn either way
        (:data:`LOOK_MAX_ABS_DEG`). Radians written into this key cannot be told from small degrees here; the real
        cell's desk check says a look that reads so.
        """
        looks = self.look_joint_positions_deg
        if looks is None:
            return self
        key = "robot.look_joint_positions_deg"
        if not looks:
            raise ValueError(f"{key} names no look: give one list of joint degrees per look, or leave the key out")
        for number, look in enumerate(looks, start=1):
            if not look:
                raise ValueError(f"{key}: look {number} names no joint; give one value in degrees per joint")
            if not all(math.isfinite(value) for value in look):
                raise ValueError(f"{key}: look {number} holds a value that is not a finite number: {list(look)}")
            if any(abs(value) > LOOK_MAX_ABS_DEG for value in look):
                raise ValueError(
                    f"{key}: look {number} turns a joint past a full turn ({list(look)} deg, at most "
                    f"{LOOK_MAX_ABS_DEG:g} either way); write each joint in degrees as the pendant shows it")
        lengths = sorted({len(look) for look in looks})
        if len(lengths) > 1:
            raise ValueError(
                f"{key}: the looks name {', '.join(str(n) for n in lengths)} joints; every look names one value in "
                "degrees per joint of the arm")
        home = self.home_joint_positions
        if home is not None and lengths[0] != len(home):
            raise ValueError(
                f"{key}: each look names {lengths[0]} joints and the home (robot.home_joint_positions) {len(home)}; a "
                "look names one value per joint of the arm, as the home does")
        return self

    @model_validator(mode="after")
    def _named_poses_are_poses_a_task_can_go_to(self) -> "RobotConfig":
        """``named_poses`` holds poses under pose names, each a finite joint vector in degrees; the default names one.

        Refused at load: a name :func:`pose_name_refusal` refuses; a pose that names no joint, holds a value that is not
        a finite number or turns a joint past a full turn (:data:`LOOK_MAX_ABS_DEG`); poses of different lengths, and a
        length unlike the home's where the home is set (read in radians by the rule above); two poses that answer to
        one word (:func:`pose_words`: a pose answers to its name and to its label, compared without case and with
        spaces folded), since the command reader hands a pose on by the word a person said and an unlabelled pose is
        said by its name; and a ``default_place_pose`` that names no pose of the tree.
        """
        key = "robot.named_poses"
        lengths: dict[int, str] = {}
        answers: dict[str, str] = {}
        for name, pose in self.named_poses.items():
            refused = pose_name_refusal(name)
            if refused:
                raise ValueError(f"{key}: {refused}")
            joints = pose.joints_deg
            if not joints:
                raise ValueError(f"{key}.{name} names no joint; give one value in degrees per joint")
            if not all(math.isfinite(value) for value in joints):
                raise ValueError(f"{key}.{name} holds a joint value that is not a finite number: {list(joints)}")
            if any(abs(value) > LOOK_MAX_ABS_DEG for value in joints):
                raise ValueError(
                    f"{key}.{name} turns a joint past a full turn ({list(joints)} deg, at most {LOOK_MAX_ABS_DEG:g} "
                    "either way); write each joint in degrees as the pendant shows it")
            lengths.setdefault(len(joints), name)
            for word in sorted(pose_words(name, pose.label)):
                first = answers.setdefault(word, name)
                if first != name:
                    raise ValueError(
                        f"{key}: {first} ({_called(self.named_poses[first])}) and {name} ({_called(pose)}) both answer "
                        f"to {word!r}; a pose answers to its name and to its label (without case, spaces folded), and "
                        "the command reader hands a pose on by the word a person said, so two poses answering to one "
                        "word could send a part to either. Give one another label")
        if len(lengths) > 1:
            named = ", ".join(f"{name} {length}" for length, name in sorted(lengths.items()))
            raise ValueError(f"{key}: the poses name different numbers of joints ({named}); every pose names one value "
                             "in degrees per joint of the arm")
        home = self.home_joint_positions
        if home is not None and lengths and next(iter(lengths)) != len(home):
            length, name = next(iter(lengths.items()))
            raise ValueError(
                f"{key}.{name} names {length} joints and the home (robot.home_joint_positions) {len(home)}; a pose "
                "names one value per joint of the arm, as the home does")
        place = self.default_place_pose
        if place is not None and place not in self.named_poses:
            known = ", ".join(sorted(self.named_poses)) or "none"
            raise ValueError(f"robot.default_place_pose is {place!r}, which names no pose of robot.named_poses "
                             f"(poses: {known}); teach it first, or name one of them")
        return self

    @field_validator("natural_closing_axis")
    @classmethod
    def _natural_closing_axis_names_one(
        cls, value: "str | tuple[float, float, float, float] | None",
    ) -> "str | tuple[float, float, float, float] | None":
        """``natural_closing_axis`` is read by the one reader of a closing axis (``closing_axis.closing_axis_of``), so a
        value the pick could not read is refused here, at load, with that reader's sentence."""
        if value is None:
            return value
        # Imported here, and only for a cell that names one. The reader lives with the poses it reads (src.geometry),
        # not with the grasps it turns, so reading a tree loads no grasping package and no OpenCV.
        from src.geometry.closing_axis import closing_axis_of  # noqa: PLC0415

        try:
            closing_axis_of(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"robot.natural_closing_axis: {exc}") from None
        return value

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

