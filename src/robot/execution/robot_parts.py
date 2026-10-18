"""The arm and the gripper a robot config describes, built without a pick service.

Five functions:

* :func:`resolve_arm` runs the arm-vendor readiness gate and builds the arm ``robot.vendor`` names,
  or hands a supplied arm through;
* :func:`construct_arm` builds that arm with no gate and connects nothing: what a desk asks of the arm a cell would
  drive (P0i);
* :func:`arm_facts` reads off an arm what a gripper driver needs of it, for the build and for the desk alike;
* :func:`build_gripper` builds the end-effector ``robot.gripper`` names for the arm in hand, or a
  ``NullGripper`` that records why no real one could be built, from :func:`gripper_driver_verdict`,
  which the desk's ``gripper driver`` row asks too;
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
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.robot.core import Gripper, RobotArm, RobotCapabilities, RobotVendor

if TYPE_CHECKING:
    from src.config.schema.robot import RobotConfig
    from src.robot.core import GripperVendor
    from src.robot.grippers import ArmFacts, SubstitutionReason
    from src.config.schema.robot.sim_schema import SimConfig
    from src.config.schema.robot.tool_frame_schema import ToolFrameConfig
    from src.robot.drivers.sim.config import SimRobotConfig
    from src.robot.safety.planning.reservation import PlannerReservation


_LOG = logging.getLogger(__name__)

__all__ = ["GripperDriverVerdict", "arm_facts", "build_gripper", "build_sim_driver_config", "construct_arm",
           "driverless_verdict", "gripper_driver_verdict", "resolve_arm"]


def resolve_arm(robot_cfg: "RobotConfig", *, arm: RobotArm | None) -> RobotArm:
    """The arm ``robot_cfg`` describes, or ``arm`` itself when a live handle is supplied.

    ``arm`` is ``None`` when no handle is in hand, and it has no default so a caller states it. The
    readiness gate runs either way unless the arm drives no device of the configured vendor: the sim
    in mock mode, the dummy, and a supplied arm that reports another vendor. Nothing is connected
    here.
    """

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
    # A supplied handle is used as it is; every other step still runs from config.
    return arm if arm is not None else construct_arm(robot_cfg)


def construct_arm(robot_cfg: "RobotConfig") -> RobotArm:
    """The arm ``robot_cfg`` selects, constructed and never connected, with no readiness gate on its SDK.

    What a desk asks of the arm a cell would drive (P0i, owner decision 2): every driver constructs without its SDK and
    first needs its controller at ``connect``, so the arm built here answers its Protocols and commands nothing. A tree
    its driver refuses raises as that driver raises. :func:`resolve_arm` runs the vendor's readiness gate, then this.
    """
    from src.robot.drivers import create_arm

    vendor = RobotVendor.from_string(robot_cfg.vendor)
    if vendor is RobotVendor.SIM:
        from src.robot.safety.planning.reservation import PlannerReservation

        return create_arm(
            RobotVendor.SIM,
            config=build_sim_driver_config(
                robot_cfg.sim, robot_cfg.gripper.tool_frame,
                planner_reservation=PlannerReservation.from_config(robot_cfg=robot_cfg),
            ),
        )
    if vendor is RobotVendor.DUMMY:
        # Dummy arm ignores the config tree entirely.
        return create_arm(RobotVendor.DUMMY)
    # The UR takes the full schema config tree.
    return create_arm(vendor, config=robot_cfg)


def arm_facts(arm: object) -> "ArmFacts":
    """What a gripper driver needs of ``arm``, read off it: the vendor it reports, whether it switches digital I/O
    (``SupportsDigitalIO``) and the controller its identity names.

    The build asks it of the handle in hand and the desk's ``gripper driver`` row of the arm the tree builds, so the two
    decide on the same facts. Read through ``getattr`` rather than by importing the UR driver: the execution layer has
    no import edge to a vendor driver and this must not add one.
    """
    from src.robot.core import SupportsDigitalIO
    from src.robot.grippers import ArmFacts

    return ArmFacts(vendor=_vendor_in_hand(arm), has_digital_io=isinstance(arm, SupportsDigitalIO),
                    controller=_controller_of(arm))


def _controller_of(arm: object) -> str | None:
    """The controller address ``arm`` names in its own identity, or ``None`` where it names none or reports none."""
    caps = getattr(arm, "capabilities", None)
    return caps.controller if isinstance(caps, RobotCapabilities) else None


def _vendor_in_hand(arm: object) -> str:
    """The vendor ``arm`` reports, or ``"unknown"`` for an object that reports no capabilities.

    Read through ``getattr`` so the execution layer needs no import edge to a vendor driver.
    """
    caps = getattr(arm, "capabilities", None)
    return caps.vendor if isinstance(caps, RobotCapabilities) else "unknown"


@dataclass(frozen=True)
class GripperDriverVerdict:
    """Which driver a gripper config gets on an arm, decided before anything is built.

    One decision with two readers: :func:`build_gripper` constructs from it, and the desk's
    ``gripper driver`` row states it for the arm a config builds, so the desk blocks exactly where
    the build would substitute.
    """

    #: The vendor the build constructs: the requested one, or ``none`` where it substitutes.
    vendor: "GripperVendor"
    #: ``robot.gripper.vendor`` as the config wrote it.
    requested: str
    #: Why no real end-effector can be built, or ``None`` where the requested driver is built.
    reason: "SubstitutionReason | None" = None
    detail: str = ""
    fix: str = ""


def gripper_driver_verdict(
    robot_cfg: "RobotConfig", *, arm_vendor: str, arm_has_digital_io: bool, arm_ip: "str | None",
) -> GripperDriverVerdict:
    """The driver ``robot_cfg.gripper`` gets on an arm that reports ``arm_vendor``, switches pins or not, at ``arm_ip``.

    Pure: the facts about the arm are arguments, read off an arm by :func:`arm_facts`, so the desk asks it with the facts
    of the arm the tree builds and the build with those of the handle in hand. What each driver needs of the arm is its
    own spec's to say (P0g slice 8): this asks the spec of the vendor the tree names, and says itself only what no spec
    can, a name with no driver behind it (:func:`driverless_verdict`).
    """
    from src.robot.core import GripperVendor
    from src.robot.grippers import SUBSTITUTE_VENDOR, ArmFacts, gripper_driver

    driverless = driverless_verdict(robot_cfg)
    if driverless is not None:
        return driverless
    requested = str(robot_cfg.gripper.vendor)
    gripper_vendor = GripperVendor.from_string(robot_cfg.gripper.vendor)
    driver = gripper_driver(gripper_vendor)
    if driver is None:  # pragma: no cover - driverless_verdict answers every vendor with no driver
        raise RuntimeError(f"no driver for {gripper_vendor.value!r}, and driverless_verdict admitted it")
    refusal = driver.refuse(robot_cfg, ArmFacts(vendor=arm_vendor, has_digital_io=arm_has_digital_io,
                                                controller=arm_ip))
    if refusal is not None:
        return GripperDriverVerdict(SUBSTITUTE_VENDOR, requested, refusal.reason, refusal.detail, refusal.fix)
    return GripperDriverVerdict(gripper_vendor, requested)


def driverless_verdict(robot_cfg: "RobotConfig") -> "GripperDriverVerdict | None":
    """The verdict for a ``robot.gripper.vendor`` no arm makes a driver of, or ``None`` where a driver exists.

    A name this stack does not know, and a name it knows with no driver behind it, are decided by the tree alone, so a
    desk states them whatever the arm is, even an arm that does not build (review of P0i i1). Where a driver exists,
    what it needs of the arm is its spec's to say, and :func:`gripper_driver_verdict` asks it.
    """
    from src.robot.core import GripperVendor
    from src.robot.grippers import SUBSTITUTE_VENDOR, SubstitutionReason, gripper_driver

    gripper_cfg = robot_cfg.gripper
    requested = str(gripper_cfg.vendor)
    try:
        gripper_vendor = GripperVendor.from_string(gripper_cfg.vendor)
    except ValueError:
        # An unrecognised gripper.vendor string is a misconfig; tolerate it as "no gripper" (the
        # facade stays usable) but warn instead of silently dropping the end-effector.
        return GripperDriverVerdict(
            SUBSTITUTE_VENDOR, requested, SubstitutionReason.UNKNOWN_VENDOR,
            f"gripper.vendor={gripper_cfg.vendor!r} is not a gripper this stack knows.",
            "Set gripper.vendor to one of: none, dummy, robotiq, vacuum, jaw_io, onrobot.",
        )
    driver = gripper_driver(gripper_vendor)
    if driver is None:
        # A recognised gripper vendor with no registered driver (FRANKA_HAND, schunk, future) falls
        # back to NullGripper; warn so a config-only build is not silently gripper-less.
        return GripperDriverVerdict(
            SUBSTITUTE_VENDOR, requested, SubstitutionReason.NO_DRIVER,
            f"gripper.vendor={gripper_vendor.value!r} is a recognised name with no driver in this "
            f"repo.",
            "Use robotiq, vacuum, jaw_io or onrobot, or write the driver and register it in "
            "grippers/registry.py.",
        )
    return None


def drives_jaws_over_io(robot_cfg: "RobotConfig") -> bool:
    """Whether the configured hand's driver drives jaws over the arm's digital I/O, which the bench's jaw verbs drive."""
    from src.robot.grippers import gripper_driver

    driver = gripper_driver(str(robot_cfg.gripper.vendor))
    return bool(driver is not None and driver.drives_jaws_over_io)


def build_gripper(robot_cfg: "RobotConfig", *, arm: RobotArm) -> Gripper:
    """The end-effector ``robot_cfg.gripper`` names, built for ``arm``.

    A tree that cannot produce the gripper it names yields a ``NullGripper`` that carries a
    ``GripperSubstitution`` and logs a warning, so the refusal in ``connect_cell`` reads the reason
    off the object. Nothing is connected here: the caller connects the arm before the gripper.

    Which driver is built is :func:`gripper_driver_verdict`, asked with what the handle in hand
    reports, and the desk asks the same function for the arm a config builds. How it is built is that
    driver's spec (P0g slice 8).
    """

    from src.robot.grippers import GripperSubstitution, build_substitute, gripper_driver

    # Five config combinations cannot produce a real end-effector, one per `SubstitutionReason`
    # member. Resolving them to a working NullGripper and a log line is the most dangerous silent
    # state in this build path: the cell connects, every pick reports success, the jaws close on
    # nothing and lift nothing. Each is a verdict with a reason, built in one place below that warns
    # and records the reason on the object, so a caller sees it without parsing logs. A sixth case
    # goes through it too. The arm is read through `getattr` rather than by importing the UR
    # driver: the execution layer has no import edge to a vendor driver and this must not add one.
    # The arm's facts are `arm_facts`, which the desk's row asks too.
    facts = arm_facts(arm)
    verdict = gripper_driver_verdict(
        robot_cfg,
        arm_vendor=facts.vendor,
        arm_has_digital_io=facts.has_digital_io,
        arm_ip=facts.controller,
    )
    if verdict.reason is not None:
        _LOG.warning("build_gripper: %s Built a NullGripper (no real end-effector). %s",
                     verdict.detail, verdict.fix)
        return build_substitute(robot_cfg, GripperSubstitution(
            reason=verdict.reason, requested=verdict.requested, detail=verdict.detail, fix=verdict.fix,
        ))
    driver = gripper_driver(verdict.vendor)
    if driver is None:  # pragma: no cover (a verdict and its specs out of step)
        raise RuntimeError(f"gripper_driver_verdict admitted {verdict.vendor.value!r} and no spec builds it")
    return driver.build(robot_cfg, arm, facts)


def build_sim_driver_config(
    schema_sim: "SimConfig",
    tool_frame: "ToolFrameConfig | None" = None,
    *,
    planner_reservation: "PlannerReservation | None" = None,
) -> "SimRobotConfig":
    """Convert a Pydantic ``SimConfig`` into the driver-side ``SimRobotConfig`` dataclass.

    The single conversion point shared by ``RuntimePickService.from_robot_config`` and the
    ``src.willy_sim`` runners (so the Pydantic config tree is the one source of truth for
    the sim cell). Copies only the driver-relevant fields; the Pydantic-only scene-authoring extras
    (``assets_root``/``scene_setup``/camera mount+near-clip) are read directly
    off the schema by willy_sim and intentionally do not cross into the lean driver dataclass.

    ``planner_reservation`` is what the planner sidecar allocates. It depends on the declared world
    rather than on the sim block, so the callers, which hold the whole robot config, pass
    ``PlannerReservation.from_config`` of it.
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
        # which lets a mounted gripper swap its width profile while the arm keeps the
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
        planner_reservation=planner_reservation,
    )
