"""Pure-Python configuration dataclasses for the Isaac sim driver.

They avoid every Isaac SDK import deliberately, so they are safe to construct, validate
and serialise on a host without Isaac installed. The SDK loads only when the driver
connects, which :mod:`.session` handles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "SimCameraConfig",
    "SimRobotConfig",
]


CameraMountingMode = Literal["eye_to_hand", "eye_in_hand"]


@dataclass(frozen=True, slots=True)
class SimCameraConfig:
    """One camera attached to the simulated scene.

    Attributes
    ----------
    prim_path
        The simulator prim path that owns the camera sensor.
    mounting_mode
        ``"eye_to_hand"`` for a camera fixed in the world frame and ``"eye_in_hand"``
        for one attached to the end-effector.
        :class:`src.robot.execution.calibration.CalibrationRoutine` reads it to pick the
        matching calibrator.
    """

    prim_path: str
    mounting_mode: CameraMountingMode = "eye_to_hand"


@dataclass(frozen=True, slots=True)
class SimRobotConfig:
    """The top-level config block for one Isaac-backed robot cell.

    Attributes
    ----------
    backend
        Always ``"isaac"``. It is a discriminator, so a PyBullet or MuJoCo backend can
        occupy the same slot.
    enabled
        The master switch. At ``False`` the registry raises rather than spinning up an
        Isaac session silently.
    scene
        A scene identifier or a USD path. The driver opens it on connect.
    robot_prim_path
        The simulator prim path of the articulation this stack drives.
    gripper_prim_path
        The optional prim path of the gripper articulation or asset.
    cameras
        A mapping of camera name to :class:`SimCameraConfig`, empty where the cell runs
        without simulated cameras.
    home_joint_positions
        Explicit home joints. The driver falls back to the scene-defined home where they
        are omitted.
    step_dt_s
        The deterministic simulator step, in seconds. A driver uses it for both motion
        stepping and ``wait_until_steady``.
    settle_timeout_s
        The longest sim time in seconds the driver waits for a steady signal before
        returning :attr:`MotionStatus.TIMEOUT`.
    motion_planner
        The approach-phase planner. ``"ik"`` is the blind pure-IK straight line,
        ``"rmpflow"`` drives the reactive collision-aware policy first and then snaps to
        IK, and ``"curobo"`` is the default described at the field itself.
    headless
        Run Isaac without a viewport, which is what a batch run wants.
    mock_mode
        At ``True`` the driver runs as a pure-Python kinematic mock: no Isaac SDK is
        touched, ``connect()`` succeeds immediately, and a motion method updates an
        internal TCP and joint cache and returns :attr:`MotionStatus.EXECUTED`. That is
        the offline mode, and it lets a downstream project drive the
        ``RobotVendor.SIM`` driver without owning an Isaac workstation. The mock
        kinematics are identity on purpose: ``move_linear(pose)`` assigns the TCP.
    """

    backend: Literal["isaac"] = "isaac"
    enabled: bool = False
    # Which UR model this sim cell drives. The key selects the Lula solver config, the
    # Isaac USD, the cuRobo {key}.yml and the safety kinematics_model, which
    # src.robot.drivers.sim.robot_models sets out. The default "ur5e" leaves an existing
    # cell unchanged.
    robot_model: str = "ur5e"
    scene: str | None = None
    robot_prim_path: str | None = None
    gripper_prim_path: str | None = None
    cameras: dict[str, SimCameraConfig] = field(default_factory=dict)
    home_joint_positions: tuple[float, ...] | None = None
    step_dt_s: float = 1.0 / 60.0
    settle_timeout_s: float = 5.0
    # Flange to TCP, the full rigid transform: from the Lula EE frame, which is tool0
    # and the flange, to the grasp centre of the gripper. fk, ik and get_tcp_pose work in
    # this TCP frame, so motion targets the grasp point rather than the flange. The
    # identity targets tool0 directly.
    #
    # It is one transform in one field on purpose. Two values describing one transform,
    # with one of them invisible to config, is how a cell silently inherits the wrong
    # tool: `gripper_mount` selects the width profile of a different gripper while the
    # arm keeps the 2F-85 offset, which measures 8 mm out on an EZU-35 and 17 mm on an
    # EGU-50. Both halves come from `robot.gripper.tool_frame`, which the real drivers
    # read too.
    tool_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tool_rotation_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    # The approach-phase motion planner, one of three.
    #
    # "ik" is the precise pure-IK path with an interpolated move_joint, which is a blind
    # straight line in joint space. "rmpflow" drives the Lula RMPflow reactive policy
    # toward the pose first, collision-aware once scene obstacles are registered through
    # IsaacRobotArm.register_planner_obstacles, and then snaps to the exact IK solution.
    # "curobo" is a global collision-free trajectory from a process-isolated cuRobo
    # planning server: cuRobo cannot share the Isaac process because their warp versions
    # clash, so it runs in its own environment and the driver talks to it over stdio,
    # which curobo_client.py handles. On that path the driver executes the whole cuRobo
    # trajectory and never snaps blindly, because the planner owns the final motion.
    #
    # The default is "curobo", so every sim cell plans real collision-aware trajectories,
    # gated by the Coal-mesh self-collision preflight that the sim profile configures as
    # backend=fcl with kinematics_model=ur5e. A host without the cuRobo environment stays
    # green because the driver falls back to "ik" both at build time, through
    # curobo_env_available() in the dense build_service, and at move time, through the
    # _resolve_motion_planner probe, and mock_mode short-circuits before curobo is
    # reached at all. Validated on-box: dense curobo with Coal and vision is 10/10, all
    # at 99.5 mm. An earlier 5/10 was measured to be the ik_quality joint-jump guard
    # falsely rejecting the continuous planned config of cuRobo rather than
    # self-collision fidelity, and skipping the continuity guards on the cuRobo path,
    # which owns trajectory continuity, fixed it while the static Coal self-collision,
    # workspace and joint-limit guards still gate the final config. The arm
    # self-collision spheres are mesh-fit to the Coal mesh, from Lula and the surface, so
    # the cuRobo planning stays Coal-faithful. A runner can still force "ik" or
    # "rmpflow" per cell.
    motion_planner: Literal["ik", "rmpflow", "curobo"] = "curobo"
    headless: bool = True
    mock_mode: bool = False
