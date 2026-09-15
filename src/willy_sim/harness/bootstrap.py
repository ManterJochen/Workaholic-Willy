"""Shared Isaac sim-cell bootstrap for the willy_sim runners.

Every pick, calibration and diagnostic runner opens its Isaac session with the same prefix: load
the sim config, build the fail-closed arm, start the session, push the asset root into Kit's
settings, author the combined scene, connect the arm, then build and connect the gripper.
:func:`bootstrap_sim_cell` is that prefix and returns a typed :class:`SimCell`. Each runner adds
its own perception, calculator, resolver, policy and service assembly on top of the booted cell.

Importable where ``isaacsim`` is absent: only ``IsaacRobotArm`` and ``build_combined_scene`` are
imported at module scope, and both lazy-import ``isaacsim`` internally. ``carb``, ``omni.usd`` and
the gripper are imported inside the function instead.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from src.config import ConfigError
from src.config.schema.robot import RobotConfig
from src.contracts import UNSET, Maybe, chosen
from src.robot.drivers.sim.arm import IsaacRobotArm
from src.robot.drivers.sim.robot_models import ur_model_spec
from src.utility.log_cfg import create_logger
from src.willy_sim.config import (
    load_sim_config,
    require_robot,
    sim_driver_config,
    sim_safety_preflight,
)
from src.willy_sim.constants import BOOTSTRAP_LOG_FILE, WILLY_SIM_LOG_DIR
from src.willy_sim.grippers import MOUNTED_GRIPPERS, MountedGripperSpec, sim_mount_for
from src.willy_sim.harness.coverage import warn_if_out_of_frame
from src.willy_sim.harness.reach import warn_if_unreachable
from src.willy_sim.scene import SceneHandles, build_combined_scene

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.app import AppConfig
    from src.config.schema.robot.safety_schema import DwellSafetyConfig
    from src.config.schema.robot.sim_schema import SimConfig
    from src.robot.grippers.sim import IsaacGripper

#: Which cell came up, on file. The banners below print to stdout because an Isaac boot buries a
#: log line, but stdout lives only in the shell that ran it, so the same facts are logged here too.
_LOG = create_logger("SimCellBootstrap", log_file=BOOTSTRAP_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR)

__all__ = [
    "ENV_ALLOW_DEGRADED",
    "DegradedMotionStackError",
    "SimCell",
    "announce_planning_environment",
    "bootstrap_sim_cell",
    "motion_stack_banner",
    "require_motion_stack",
    "resolve_runner_planner",
]


def announce_planning_environment(
    planner: str, *, robot_config: str | None = None, kinematics_model: str = "ur5e",
) -> str:
    """Print the cuRobo and Coal anchoring status once at boot, and say when a run will degrade.

    The motion stack builds on two external engines: the cuRobo trajectory planner and the Coal/fcl
    exact-mesh collision backend (see :mod:`src.robot.safety.planning.environment`). When a runner
    asks for the ``"curobo"`` planner and the cuRobo env is absent, the arm falls back to blind IK
    without a word. Blind IK can choose a self-colliding arm branch that the fail-closed guard then
    correctly rejects, which surfaces as ``execution_failed`` and reads as a grasp-quality problem
    rather than as a missing engine. Reusing the safety-layer probe here makes that visible instead
    of burying it in the Isaac boot log. Returns the announced text as well as printing it, so the
    banner can be read back without Isaac.

    ``robot_config`` and ``kinematics_model`` make the banner describe this cell. Both engines are
    per-robot: the planner loads ``{model}.yml``, the exact-mesh guard loads
    ``{model}_collision_meshes.npz``. Left at the defaults the banner reports the env-var reading
    and ``ur5e``, which names the wrong robot in a cell driving anything else, even though the
    driver still resolves the right config. A caller that has a configured cell passes both.
    """
    from src.robot.safety.planning.environment import (
        ENV_CUROBO_PYTHON,
        probe_planning_environment,
    )

    env = probe_planning_environment(robot_config=robot_config, kinematics_model=kinematics_model)
    lines = [env.render()]
    if planner == "curobo" and not env.curobo.available:
        lines.append(
            f"  !! motion_planner='curobo' requested but the cuRobo env is MISSING -> this run uses BLIND IK.\n"
            f"     Reach poses that self-collide on blind IK are (correctly) safety-rejected, so picks may fail.\n"
            f"     Set {ENV_CUROBO_PYTHON} to the cuRobo env python for representative on-box runs."
        )
    text = "\n".join(lines)
    print(text, flush=True)
    # Debug: the full report is on stdout already. What is worth keeping in the file is the one-line
    # answer to "was this run planned by cuRobo or by blind IK", which changes what the arm proposes.
    _LOG.debug(
        "planning environment for planner=%r robot_config=%s kinematics=%s: curobo=%s collision=%s",
        planner, robot_config, kinematics_model, env.curobo.available, env.collision.available,
    )
    return text


class DegradedMotionStackError(RuntimeError):
    """The cell asked for cuRobo / exact-mesh collision and at least one of them is not installed."""


#: Escape hatch for :func:`require_motion_stack`. Set to ``1`` to run anyway, knowingly degraded.
ENV_ALLOW_DEGRADED = "WILLY_ALLOW_DEGRADED_MOTION"


def motion_stack_banner(env: object, *, planner: str, missing: list[str]) -> str:
    """The block printed when an engine the cell depends on is absent.

    A box rather than a log line: an Isaac boot emits thousands of lines, and a one-line warning
    about a missing engine disappears among them.
    """
    from src.robot.safety.planning.environment import ENV_COAL_PREFIX, ENV_CUROBO_PYTHON

    hints = {
        "cuRobo planner": (
            f"set {ENV_CUROBO_PYTHON} to the cuRobo env's python.exe "
            f"(see ext_deps/README.md). Without it the arm plans with BLIND IK: no collision-aware "
            f"routing, so it proposes self-colliding branches the guard then rejects; picks fail for "
            f"a reason that looks like bad grasping."
        ),
        "exact-mesh collision (Coal/fcl)": (
            f"set {ENV_COAL_PREFIX} to the Coal env prefix (see ext_deps/README.md). Without it the "
            f"self-collision guard falls back to CAPSULE proxies, whose default 60 mm link radius was "
            f"measured to over-reject legitimate reach-down grasps; and, worse, capsules were "
            f"deceived 3x where the mesh check was not."
        ),
    }
    width = 96
    out = ["", "#" * width, "#" + " MOTION STACK INCOMPLETE ".center(width - 2, "=") + "#"]
    for name in missing:
        out.append(f"#  MISSING: {name}".ljust(width - 1) + "#")
        for line in _wrap(hints[name], width - 8):
            out.append(f"#      {line}".ljust(width - 1) + "#")
    out.append("#" + "-" * (width - 2) + "#")
    for line in _wrap(
        f"This cell is configured to route EVERY motion through cuRobo and the exact-mesh collision "
        f"engine. Running without them does not merely make the cell slower or slightly different; it "
        f"changes which motions are proposed and which are accepted, so any pick rate measured now is "
        f"NOT the pick rate of the configured system, and must not be reported as one. Install the "
        f"missing engine, or set {ENV_ALLOW_DEGRADED}=1 to proceed knowingly.",
        width - 6,
    ):
        out.append(f"#  {line}".ljust(width - 1) + "#")
    out.append("#" * width)
    out.append("")
    return "\n".join(out)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def resolve_runner_planner(requested: str, *, env_available: bool) -> str:
    """The planner a runner will actually use, or a refusal. Never a quieter one than was asked for.

    A run that asks for `curobo` on a box without the cuRobo environment is refused rather
    than built on the blind `ik` path. A pick rate measured that way describes a different
    motion stack than the cell was configured with, and the number cannot say so on its own.

    Asking for `ik` explicitly still gives `ik`, whatever is installed: that is somebody
    deciding, and a decision does not depend on what happens to be on the box.
    """
    if requested != "curobo" or env_available:
        return requested
    raise DegradedMotionStackError(
        "this run asks for motion_planner='curobo' and the cuRobo environment is not available on "
        "this box. Every motion would be planned and path checked by a sidecar that cannot start, so "
        "the run is refused rather than built on the blind ik path: a pick rate measured that way "
        "describes a different motion stack. Install the environment (see ext_deps/README.md), or "
        "pass --motion-planner ik to say that this run is deliberately unplanned."
    )


def require_motion_stack(
    planner: str,
    *,
    robot_config: str | None = None,
    kinematics_model: str = "ur5e",
    exact_mesh_collision: bool,
) -> list[str]:
    """Fail closed when an engine this cell routes through is not installed. Returns what was missing.

    The safety argument is that motion is collision-planned by cuRobo and collision-checked against
    exact meshes by Coal/fcl. A missing engine replaces that argument rather than weakening it:
    blind IK proposes self-colliding branches, and the capsule proxy passes configurations the mesh
    check rejects. The default is therefore to refuse rather than to produce numbers that describe a
    different system.

    ``exact_mesh_collision`` says whether the cell's safety config asked for the mesh backend
    (``self_collision.backend == "fcl"``); a cell that never asked is not held to it.

    Set :data:`ENV_ALLOW_DEGRADED` to run anyway. The banner still prints, every time.
    """
    import os

    from src.robot.safety.planning.environment import probe_planning_environment

    env = probe_planning_environment(robot_config=robot_config, kinematics_model=kinematics_model)
    missing: list[str] = []
    if planner == "curobo" and not env.curobo.available:
        missing.append("cuRobo planner")
    if exact_mesh_collision and not env.collision.available:
        missing.append("exact-mesh collision (Coal/fcl)")
    if not missing:
        return []
    print(motion_stack_banner(env, planner=planner, missing=missing), flush=True)
    if os.environ.get(ENV_ALLOW_DEGRADED, "").strip() not in ("1", "true", "TRUE", "yes"):
        # Raised, so it gets no log line of its own: the traceback is the record.
        raise DegradedMotionStackError(
            f"missing motion-stack engine(s): {', '.join(missing)}. "
            f"Install them, or set {ENV_ALLOW_DEGRADED}=1 to run knowingly degraded."
        )
    # This branch returns the failure instead of raising it, so the run proceeds and its numbers
    # describe a different motion stack than the configured one. Logged as well as printed, because
    # the banner exists only in the shell that ran it.
    _LOG.error(
        "proceeding with a DEGRADED motion stack: %s missing, %s is set. Pick rates from this run do "
        "NOT describe the configured system.",
        ", ".join(missing), ENV_ALLOW_DEGRADED,
    )
    return missing


@dataclass(frozen=True, slots=True)
class SimCell:
    """The booted Isaac sim cell shared by every runner's ``build_service``.

    Carries the live arm + gripper + scene handles plus the loaded, ``robot``-narrowed config, so
    a runner reads ``cell.sim`` / ``cell.robot.gripper`` / ``cell.dwell`` without re-deriving them.
    """

    arm: IsaacRobotArm
    gripper: IsaacGripper
    handles: SceneHandles
    cfg: AppConfig
    robot: RobotConfig
    sim: SimConfig
    dwell: DwellSafetyConfig
    #: Engines the cell routes through that are not installed (empty == fully anchored). Non-empty only
    #: when the operator explicitly opted into a degraded run; carried so a runner can stamp it onto its
    #: results, because a pick rate measured without them describes a different system.
    degraded_engines: tuple[str, ...] = ()
    #: The standalone gripper Isaac mounted, derived from the hand and the arm asset, or ``None`` where
    #: the asset's baked variant carries the hand. A record stamps its name (``cell_identity``).
    mount: MountedGripperSpec | None = None


def bootstrap_sim_cell(
    data_dir: str | None,
    *,
    headless: bool,
    safety: bool = True,
    scene_kwargs: Mapping[str, Any] | None = None,
    hand: str | None = None,
    robot_model: str | None = None,
    extra_profiles: Sequence[str] | None = None,
    post_scene_hook: Callable[[Any], None] | None = None,
    motion_planner: Maybe[str] = UNSET,
) -> SimCell:
    """Boot the combined arm and gripper Isaac cell from the sim config tree.

    Loads and validates the sim ``cfg``, builds the fail-closed :class:`IsaacRobotArm`
    (``safety=False`` skips the preflight, as ``inspect_wrist_cam`` does), starts the session,
    pushes the asset root into Kit's settings, authors the combined scene (``scene_kwargs`` go
    verbatim to :func:`build_combined_scene`: ``objects_override``, ``camera_position_mm``, the
    ArUco marker kwargs), connects the arm, then constructs and connects the gripper. Returns a
    :class:`SimCell`. Left at the defaults the cell is a UR5e carrying the hand the tree names;
    ``robot_model`` stacks the matching robot layer.

    ``hand`` runs this hand, a registry name, instead of the one the tree names: the robot is
    revalidated with it, and the mount and the tool frame follow from it. ``None`` keeps the tree's
    hand.

    ``post_scene_hook`` is a callback ``(stage) -> None`` invoked after the scene is built and
    before ``arm.connect()``, which is the first play. A runner uses it to author extra physics
    prims that must exist before play, such as the Isaac surface gripper, which the gripper
    extension registers only at play. Leaving it ``None`` changes nothing.
    """
    # ``robot_model`` selects a second profile layer ("sim,ur3e"), so the whole robot-dependent
    # config, meaning the reach-limited scene geometry, safe_pose and the safety kinematics model,
    # comes from that layer's YAML rather than from per-runner patching. ``None`` loads plain "sim".
    started = time.perf_counter()
    _LOG.info(
        "booting sim cell: data_dir=%s robot_model=%s profiles=%s headless=%s safety=%s "
        "hand=%s",
        data_dir, robot_model, list(extra_profiles or ()), headless, safety, hand,
    )
    cfg = load_sim_config(data_dir, robot_model=robot_model, extra_profiles=extra_profiles)
    robot = require_robot(cfg)
    if robot_model is not None and robot.sim.robot_model != robot_model:
        # Fail closed: the layer exists, because the loader validated it, but it does not claim the
        # robot. Driving a UR3e cell whose config still says "ur5e" picks the wrong Lula and cuRobo
        # kinematics and the wrong USD, with bad motion as the only symptom.
        raise ValueError(
            f"profile layer {robot_model!r} does not set robot.sim.robot_model={robot_model!r} "
            f"(config says {robot.sim.robot_model!r}). Fix config/robot/robot.{robot_model}.yaml."
        )
    # A per-run hand, validated. The robot is rebuilt through the schema rather than with
    # ``model_copy``, which validates nothing and would let a name no registry holds reach the Isaac
    # boot. The camera section is untouched, so the one cross-section rule on AppConfig, the primary
    # camera calibrated in one place, cannot change here.
    if hand is not None:
        merged = robot.model_dump()
        merged["gripper"]["model"] = hand
        robot = RobotConfig.model_validate(merged)
    # Which gripper Isaac puts on the arm, derived rather than configured: a second key for it could
    # name another hand than robot.gripper.model. Refuses, before the boot, a cell that names no hand,
    # a name the cell's own registry does not hold, and a hand the sim has no mount for.
    mount = sim_mount_for(robot.sim.robot_model, robot.gripper.model, data_dir=data_dir)
    hand_spec = mount if mount is not None else MOUNTED_GRIPPERS.get(str(robot.gripper.model))
    if hand_spec is None:
        raise ConfigError(
            f"the {robot.sim.robot_model!r} asset bakes the hand {robot.gripper.model!r}, and the sim holds no "
            f"measured tool frame for it: add its MountedGripperSpec to willy_sim.grippers"
        )
    # A different gripper is a different tool frame, not only a different width profile: a mount that
    # swapped only the profile would leave the arm composing flange->TCP with the 2F-85's 132 mm, 12.1 mm
    # out for a mounted EGU-50 and 18.0 mm for an EZU-35. The rotation is shared: every mount spec puts
    # its approach on wrist +Y, so only the translation differs. A hand passed here writes the frame its
    # mount composes, and a frame the tree declares has to agree with it.
    expected = tuple(float(v) for v in hand_spec.flange_to_tcp_offset_mm)
    if hand is not None:
        merged = robot.model_dump()
        merged["gripper"]["tool_frame"]["offset_mm"] = list(expected)
        robot = RobotConfig.model_validate(merged)
    declared = tuple(float(v) for v in (robot.gripper.tool_frame.offset_mm or ()))
    if len(declared) != len(expected) or any(abs(a - b) > 1e-6 for a, b in zip(declared, expected)):
        raise ConfigError(
            f"the sim cell declares robot.gripper.tool_frame.offset_mm {list(declared)}, and the hand "
            f"{robot.gripper.model!r} on the {robot.sim.robot_model!r} asset puts its grasp centre at "
            f"{list(expected)} mm from the flange. Declare {list(expected)}, or pass the hand to the bootstrap "
            f"with hand=, which writes it."
        )
    cfg = cfg.model_copy(update={"robot": robot})
    # Geometry sanity before the roughly 60 s Isaac boot: every scene position here was authored for
    # a UR5e (850 mm reach). On a shorter arm the same numbers are physically untouchable, and the
    # symptom is an opaque IK or plan failure, or a run that reads as bad grasping but is geometry.
    warn_if_unreachable(robot)
    # The same question for the optics: an aimed camera (the tilted rig) is placed independently of
    # the robot layer that anchors the scene, so nothing guarantees the two agree about where the
    # workspace is. A scene outside the frame reads as a perception failure, not a geometry one.
    warn_if_out_of_frame(robot)
    dwell = robot.safety.dwell
    sim = robot.sim
    preflight = sim_safety_preflight(cfg) if safety else None
    driver_cfg = sim_driver_config(cfg, headless=headless)
    # Engine check before the boot as well: it is a filesystem probe, so an operator who has not
    # pointed at the cuRobo and Coal envs learns it in milliseconds rather than after the Isaac
    # start-up and a run whose numbers describe a different motion stack.
    from src.robot.drivers.sim.robot_models import curobo_robot_yml

    # The descriptor the arm loads, by arm and hand. With no hand named there is none, and the banner
    # says so rather than falling back to the environment's ur5e.yml.
    from src.robot.drivers.sim.robot_models import NO_DESCRIPTOR

    _robot_yml = (
        curobo_robot_yml(sim.robot_model, robot.gripper.model) if robot.gripper.model else NO_DESCRIPTOR
    )
    # kinematics_model is optional in the schema; an unset one means the guard has no DH chain
    # configured at all, so report the bundle for the robot the cell actually drives.
    _kin_model = robot.safety.self_collision.kinematics_model or sim.robot_model
    # The planner this run uses: the caller's, else the tree's. It is read here rather than off
    # ``driver_cfg`` so that the engine check, the banner and the arm all speak about the same one,
    # because a runner that resolves ``ik`` otherwise leaves this gate asking about ``curobo``.
    _planner = motion_planner if chosen(motion_planner) else driver_cfg.motion_planner
    if _planner not in ("ik", "rmpflow", "curobo"):
        raise ValueError(
            f"motion_planner={_planner!r} is not one this driver knows; it takes 'ik', 'rmpflow' or "
            f"'curobo'"
        )
    if _planner != driver_cfg.motion_planner:
        driver_cfg = replace(driver_cfg, motion_planner=_planner)  # type: ignore[arg-type]
    degraded = tuple(
        require_motion_stack(
            _planner,
            robot_config=_robot_yml,
            kinematics_model=_kin_model,
            exact_mesh_collision=(robot.safety.self_collision.backend or "").lower() == "fcl",
        )
    )
    arm = IsaacRobotArm(driver_cfg, safety_preflight=preflight)
    # The cell's own geometry, underneath whatever the runner registers. ``set_world`` replaces the
    # planner's world, so a runner that declares its bin walls would drop the bench with them. The
    # declaration is the one the guard reads, so a cell has a single description of its furniture.
    from src.robot.safety.planning.world import build_planner_cuboids, build_planner_meshes

    _world_cfg = getattr(robot.safety, "planning_world", None)
    _reservation = driver_cfg.planner_reservation
    _base_world = build_planner_cuboids(
        _world_cfg, robot.safety.self_collision.fixtures,
        **({"max_cuboids": _reservation.cuboid_slots} if _reservation is not None else {}),
    )
    _base_meshes = build_planner_meshes(_world_cfg)
    if _base_world:
        arm.set_planner_base_world(_base_world, _base_meshes)
        _LOG.info(
            "planner base world: %d obstacle(s) and %d mesh(es) from config",
            len(_base_world), len(_base_meshes),
        )
    arm.session.start()
    import carb  # type: ignore[import-not-found]

    if sim.assets_root:
        carb.settings.get_settings().set("/persistent/isaac/asset_root/default", sim.assets_root)
    handles = build_combined_scene(arm.session, sim, mount=mount, **(scene_kwargs or {}))
    if post_scene_hook is not None:  # author extra prims (e.g. the surface gripper) before the first play
        import omni.usd  # type: ignore[import-not-found]

        post_scene_hook(omni.usd.get_context().get_stage())
    arm.connect()

    from src.robot.grippers.sim import IsaacGripper

    # The hand's own GripperProfile, mounted or baked. A mounted gripper drives its joint merged into
    # the arm's articulation; a baked hand is the one its mount spec describes, the 2F-85 spec
    # reproducing the baked variant's joint frame, so the baked 2F-85 keeps ROBOTIQ_2F85_PROFILE.
    gripper_profile = hand_spec.profile
    gripper = IsaacGripper(
        session=arm.session, gripper_prim_path=sim.gripper_prim_path, profile=gripper_profile,
    )
    gripper.connect()
    # Announce the cuRobo and Coal anchoring once, right before the runner's run loop. A "curobo"
    # run with a missing cuRobo env refuses, so this is not the last warning before a degraded run;
    # it stays because Coal is the other engine, and its absence is still a quieter check rather
    # than a refusal.
    announce_planning_environment(
        driver_cfg.motion_planner, robot_config=_robot_yml, kinematics_model=_kin_model,
    )
    # One line that identifies the cell a later result belongs to: which robot, which end-effector,
    # how many objects, and whether the ``camera_to_base`` the calculator reasons in exists at all.
    _LOG.info(
        "sim cell up in %.1f s: robot=%s gripper=%s planner=%s objects=%d camera_to_base=%s%s",
        time.perf_counter() - started, sim.robot_model,
        mount.name if mount is not None else f"baked:{ur_model_spec(sim.robot_model).baked_gripper_variant}",
        driver_cfg.motion_planner,
        len(handles.object_specs), "SET" if handles.camera_to_base is not None else "NONE",
        f" degraded={list(degraded)}" if degraded else "",
    )
    return SimCell(
        arm=arm, gripper=gripper, handles=handles, cfg=cfg, robot=robot, sim=sim, dwell=dwell,
        degraded_engines=degraded, mount=mount,
    )
