"""The arm and the gripper a robot config describes, built without a pick service.

Three functions:

* :func:`resolve_arm` runs the arm-vendor readiness gate and builds the arm ``robot.vendor`` names,
  or hands a supplied arm through;
* :func:`build_gripper` builds the end-effector ``robot.gripper`` names for the arm in hand, or a
  ``NullGripper`` that records why no real one could be built;
* :func:`build_sim_driver_config` converts the Pydantic ``SimConfig`` into the driver's
  ``SimRobotConfig``, for ``resolve_arm`` and for the ``willy_sim`` runners.

``RuntimePickService.from_robot_config`` builds its arm and gripper here, and so does
:class:`~src.robot.execution.robot.Robot`, which needs only the two handles and gets the ones a pick
service gets from the same tree.

Every driver, gripper and readiness import is function-local. Importing this module loads
``src.robot.core`` and none of the grasping stack, and a caller that patches
``src.robot.drivers.doctor.require_arm_vendor_ready`` reaches the gate, because the name is looked up
when the function runs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.robot.core import Gripper, RobotArm, RobotCapabilities, RobotVendor

if TYPE_CHECKING:
    from src.config.schema.robot import RobotConfig
    from src.config.schema.robot.sim_schema import SimConfig
    from src.config.schema.robot.tool_frame_schema import ToolFrameConfig
    from src.robot.drivers.sim.config import SimRobotConfig


_LOG = logging.getLogger(__name__)

__all__ = ["build_gripper", "build_sim_driver_config", "resolve_arm"]


def resolve_arm(robot_cfg: "RobotConfig", *, arm: RobotArm | None) -> RobotArm:
    """The arm ``robot_cfg`` describes, or ``arm`` itself when a live handle is supplied.

    ``arm`` is ``None`` when no handle is in hand, and it has no default so a caller states it. The
    readiness gate runs either way unless the arm drives no device of the configured vendor: the sim
    in mock mode, the dummy, and a supplied arm that reports another vendor. Nothing is connected
    here.
    """

    from src.robot.drivers import create_arm

    vendor = RobotVendor.from_string(robot_cfg.vendor)

    # Startup gate: fail early with a clear "host not ready for vendor X" message if the configured
    # arm vendor's SDK is missing, instead of late inside connect(). Mock paths that drive no real
    # device (the sim in mock_mode, the dummy) are skipped, so a host without the vendor SDK can
    # still run them.
    from src.robot.drivers.doctor import require_arm_vendor_ready

    sim_mock = vendor is RobotVendor.SIM and bool(robot_cfg.sim.mock_mode)
    # A supplied arm does not count as mock mode on its own, because holding a constructed arm proves
    # nothing about the SDK: `URRobotArm` constructs on a host without `ur_rtde`, since the SDK import
    # sits inside `connect()`. A handle proves only that `create_arm` does not run here; whether a
    # real device gets driven is decided by what the arm is. An arm advertising another vendor is a
    # stand-in and drives nothing of this one; an arm advertising this vendor is the driver, and its
    # connect() still needs the SDK the gate is asking about.
    #
    # `RobotCapabilities.vendor`, not `isinstance` against a driver class: reading the class would
    # import the vendor module (`isaacsim` for the sim) into the composition root, and
    # `core/capabilities.py` says in as many words that pipeline code branches on these flags
    # instead. Every driver declares it (sim/arm.py `vendor="sim"`, ur/arm.py `vendor="ur"`), and the
    # dataclass enforces the lower-case form the enum values use. Read through `getattr` the way
    # `RuntimePickService._capabilities` does: a partial stand-in that implements no capabilities is
    # not any vendor's driver, so it counts as a stand-in.
    supplied_caps = getattr(arm, "capabilities", None)
    supplied_arm_vendor = (
        supplied_caps.vendor if isinstance(supplied_caps, RobotCapabilities) else None
    )
    stand_in_arm = arm is not None and supplied_arm_vendor != vendor.value
    require_arm_vendor_ready(
        vendor,
        mock_mode=sim_mock or vendor is RobotVendor.DUMMY or stand_in_arm,
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
    return arm


def _vendor_in_hand(arm: object) -> str:
    """The vendor ``arm`` reports, or ``"unknown"`` for an object that reports no capabilities.

    Read through ``getattr`` so the execution layer needs no import edge to a vendor driver.
    """
    caps = getattr(arm, "capabilities", None)
    return caps.vendor if isinstance(caps, RobotCapabilities) else "unknown"


def build_gripper(robot_cfg: "RobotConfig", *, arm: RobotArm) -> Gripper:
    """The end-effector ``robot_cfg.gripper`` names, built for ``arm``.

    A tree that cannot produce the gripper it names yields a ``NullGripper`` that carries a
    ``GripperSubstitution`` and logs a warning, so the refusal in ``connect_cell`` reads the reason
    off the object. Nothing is connected here: the caller connects the arm before the gripper.
    """

    from src.robot.grippers import (
        GripperSubstitution,
        SubstitutionReason,
        create_gripper,
    )
    from src.robot.core import GripperVendor

    # --- Gripper construction ----------------------------------------------
    #
    # Five config combinations cannot produce a real end-effector, one per `SubstitutionReason`
    # member. Resolving them to a working NullGripper and a log line is the most dangerous silent
    # state in this build path: the cell connects, every pick reports success, the jaws close on
    # nothing and lift nothing. `_substitute` below routes all five through one place that warns and
    # records the reason on the object, so a caller sees it without parsing logs. A sixth case goes
    # through it too.
    gripper_cfg = robot_cfg.gripper

    def _substitute(reason: SubstitutionReason, detail: str, fix: str) -> Gripper:
        _LOG.warning("build_gripper: %s Built a NullGripper (no real end-effector). %s",
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
    try:
        gripper_vendor = GripperVendor.from_string(gripper_cfg.vendor)
    except ValueError:
        # An unrecognised gripper.vendor string is a misconfig; tolerate it as "no gripper" (the
        # facade stays usable) but warn instead of silently dropping the end-effector.
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
        # Robotiq lives on the UR controller's tool I/O, reached over a socket the URCap opens at
        # that controller's address. Two separate facts have to hold.
        #
        # The arm in hand is asked, not `robot_cfg.vendor`. A supplied handle replaces the arm
        # construction step, so from that point the config does not describe the arm the cell runs
        # on: asked instead, a `vendor: ur` tree beside a `DummyRobotArm` would yield a real Robotiq
        # driver aimed at a real controller address, next to an arm holding no controller
        # connection, and nothing on the built cell would say so. The two I/O branches below ask the
        # handle (`isinstance(arm, SupportsDigitalIO)`) for the same reason.
        #
        # The address comes off the arm as well, and that takes no config field of its own.
        # `URRobotArm.__init__` keeps the tree it was built from (`self.config = config`), so
        # `arm.config.ur.ip` is the controller this arm talks to. The tree is the wrong source for a
        # supplied arm: a tree naming another vendor leaves `robot.ur.ip` at the schema default
        # 192.168.1.100, a plausible address on a real subnet that nobody chose. Read through
        # `getattr` rather than by importing the UR driver: the execution layer has no import edge to
        # a vendor driver and this must not add one.
        #
        # When `resolve_arm` built the arm itself the two sources are the same object
        # (`create_arm(RobotVendor.UR, config=robot_cfg)` hands `robot_cfg` straight to
        # `URRobotArm`), so the shipped no-handle UR cell reads the address it always read.
        arm_vendor = _vendor_in_hand(arm)
        arm_ip = getattr(getattr(getattr(arm, "config", None), "ur", None), "ip", None)
        if arm_vendor != RobotVendor.UR.value:
            # A config requesting Robotiq on a non-UR arm (notably the Isaac sim, whose `sim`
            # profile sets gripper.vendor=robotiq + arm vendor=sim) cannot build a real gripper
            # through from_robot_config. Warn instead of silently booting gripper-less; the sim
            # builds its IsaacGripper via AutonomousGraspService.from_components (shared session).
            gripper = _substitute(
                SubstitutionReason.ROBOTIQ_NEEDS_UR,
                f"gripper.vendor='robotiq' but the arm in hand reports vendor {arm_vendor!r}, "
                f"which is not a UR. A Robotiq lives on the UR controller's tool I/O, so an arm "
                f"that is not on such a controller cannot reach one.",
                "For the Isaac sim use AutonomousGraspService.from_components (the IsaacGripper "
                "needs the shared session). On a real cell, set robot.vendor: ur and either let "
                "this build the arm or hand in the UR arm you built yourself.",
            )
        elif not arm_ip:
            # The other half of the same refusal, and it is stated separately because an operator
            # cannot act on the two the same way. Falling back to `robot_cfg.ur.ip` here is the
            # thing that must not happen: on a tree that names no UR that value is the schema
            # default, and pointing a real driver at an address nobody chose is exactly what this
            # half of the refusal prevents.
            gripper = _substitute(
                SubstitutionReason.ROBOTIQ_NEEDS_UR,
                f"gripper.vendor='robotiq' and the arm in hand does report vendor "
                f"{RobotVendor.UR.value!r}, but it exposes no controller address: reading "
                f"arm.config.ur.ip off it found nothing. The Robotiq is reached over a socket on "
                f"that arm's controller, and robot.ur.ip in this tree ({robot_cfg.ur.ip!r}) "
                f"describes whatever the config names, not the arm that was handed in.",
                "Hand in an arm built from a config tree (URRobotArm keeps the one it was "
                "constructed with), or supply no arm at all and set robot.vendor: ur so this "
                "builds the arm from this tree and both halves come from the same place.",
            )
        else:
            gripper = create_gripper(
                GripperVendor.ROBOTIQ,
                config=gripper_cfg,
                ip=str(arm_ip),
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
                f"gripper.vendor='vacuum' but the arm in hand reports vendor "
                f"{_vendor_in_hand(arm)!r} and does not advertise SupportsDigitalIO. Suction "
                f"switches the controller's digital I/O, which only the real UR driver exposes.",
                "Set robot.vendor: ur for a real suction cell, or gripper.vendor: none.",
            )
    elif gripper_vendor is GripperVendor.JAW_IO:
        # Parallel jaws over the arm's digital I/O: same story as suction directly above, same
        # lifecycle contract (arm.connect() before gripper.connect(), because the driver touches
        # the controller's I/O as soon as it connects), and the same fallback when the configured
        # arm cannot switch a pin.
        #
        # This is the vendor for a gripper wired to the controller. A Robotiq is jaws too and is
        # not this: it speaks a socket the URCap opens, which is why it has its own branch and its
        # own failure mode.
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
                f"gripper.vendor='jaw_io' but the arm in hand reports vendor "
                f"{_vendor_in_hand(arm)!r} and does not advertise SupportsDigitalIO. A solenoid "
                f"jaw switches the controller's digital I/O, which only the real UR driver "
                f"exposes.",
                "Set robot.vendor: ur for a real I/O jaw cell, or gripper.vendor: none.",
            )
    elif gripper_vendor is GripperVendor.ONROBOT:
        # A gripper vendor in the enum with no branch here falls into the `else` below: a
        # `NullGripper` is substituted, a warning is logged, and the cell connects, reports every
        # pick a success and holds nothing. Every new enum member needs its branch written here.
        #
        # There is no `SupportsDigitalIO` precondition, unlike the two I/O branches, and none is
        # missing: a Compute Box is a separate device on its own network address, so this gripper
        # has no dependency on the arm vendor at all. That is also why `host` comes from the
        # gripper's own config and not from `robot.ur.ip` the way the Robotiq branch does.
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
        # A recognised gripper vendor with no registered driver (FRANKA_HAND, schunk, future) falls
        # back to NullGripper; warn so a config-only build is not silently gripper-less.
        gripper = _substitute(
            SubstitutionReason.NO_DRIVER,
            f"gripper.vendor={gripper_vendor.value!r} is a recognised name with no driver in this "
            f"repo.",
            "Use robotiq, vacuum, jaw_io or onrobot, or write the driver and register it in "
            "grippers/registry.py.",
        )

    return gripper


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
