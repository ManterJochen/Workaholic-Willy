"""The Isaac Sim-backed :class:`RobotArm` driver.

The kinematics core is real: ``connect()`` wraps the ``SingleArticulation``, builds a
Lula kinematics solver, and ``get_tcp_pose``, ``get_joint_positions``, ``fk`` and ``ik``
read live Isaac state through the ``adapter.py`` boundary, in millimetres with XYZW
quaternions in ``Frame.BASE``.

Three guarantees hold throughout:

1. Importing the module is safe on a host without Isaac. No Isaac symbol is touched at
   import time, and every ``isaacsim.*`` import is lazy, inside the non-mock branch of
   ``connect()``.
2. ``mock_mode=True`` is a pure-Python kinematic mock: no Isaac is touched, and the
   motion methods update an internal TCP and joint cache. A host without Isaac runs
   entirely on that path.
3. Every motion method honours the typed :class:`MotionResult` contract, and a read
   raises a typed error such as ``RobotConnectionError`` rather than leaking an
   ``ImportError``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from src.geometry import Frame, FrameMismatchError, Pose
from src.geometry.quaternion import (
    IDENTITY_QUAT_XYZW,
    conjugate,
    multiply,
    rotate_vector,
)

from ...core import (
    NO_PLAN_FAIL_SAFE_MESSAGE,
    IsaacNotAvailableError,
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotArm,
    RobotCapabilities,
    RobotConnectionError,
    RobotKinematicsError,
    RobotMotionRejected,
)
from ...safety import SafetyPreflight
from ...safety._ur_kinematics import ur_link_origins_mm
from ...safety.continuous_monitor import ContinuousCollisionMonitor, ContinuousGuardAbort
from ...safety.planning import CuroboPlanClient, CuroboUnavailableError
from ...safety.planning.live_world import WorldRefresh, refresh_planner_world
from ...safety.planning.world import merge_planner_worlds

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from ...safety.planning.live_world import LivePlannerWorld
from .adapter import (
    willy_joints_to_isaac,
    willy_pose_to_isaac,
    isaac_joints_to_willy,
    isaac_pose_to_willy,
    isaac_rotmat_to_wxyz,
)
from ._isaac_protocols import (
    IsaacArmSubset,
    IsaacArticulation,
    IsaacKinematicsSolver,
    IsaacMotionPolicy,
    IsaacRmpFlow,
)
from .config import SimRobotConfig
from .session import IsaacSimSession

__all__ = ["ISAAC_CAPABILITIES", "IsaacRobotArm"]

_LOGGER = logging.getLogger(__name__)


# The end-effector frame, shared by every UR e-series model. The Lula robot key is not a
# module constant: it is derived per connect from SimRobotConfig.robot_model through
# sim.robot_models.ur_model_spec, which is what lets one driver serve "UR5e", "UR3e" and
# the rest.
_EE_FRAME = "tool0"

# Settle detection: the largest joint speed in rad/s below which the arm counts as
# stopped. Validated on-box: the UR5e residual velocity at rest is near zero, so
# 1e-2 rad/s sits well above the noise floor and settles a 0.3 rad joint move in about
# 24 steps, roughly 0.4 s of sim time.
_SETTLE_VEL_THRESHOLD = 1e-2

# The success gate of move(), in millimetres. RMPflow alone converges to about 19 mm,
# its reactive steady state, so move() drives RMPflow toward the target and then snaps
# to the exact IK solution, which is sub-millimetre. This is the generous EXECUTED
# tolerance the refined TCP lands within.
_MOVE_POS_TOL_MM = 5.0

# The largest joint step in radians when interpolating a move_joint command. The
# position drive stalls on a single large point-to-point command, so move_joint walks
# waypoints no larger than this.
_MAX_JOINT_STEP_RAD = 0.03

# How many times move() re-resolves IK and drives again before giving up. A large move,
# such as one from a parked arm, or a load-perturbed move can stall short once, and
# re-resolving IK from the new configuration and driving again recovers it. A move that
# converges, which is the common case, returns on the first try.
_MOVE_MAX_ATTEMPTS = 3

# Retry a move only where it landed within this distance of the target, which is a near
# miss of settling or precision that a re-resolve can close. A far miss means a wrong or
# unreachable IK branch, and retrying the same target does not help, so it fails fast
# rather than grinding through full re-drives, which keeps a failing pick cheap. Raising
# this to retry the retreat measured worse: it does not re-converge and adds
# instability.
_MOVE_RETRY_MAX_ERR_MM = 40.0

# A hard cap on move_joint interpolation waypoints, so a wild or far IK solution cannot
# explode into thousands of physics steps. 250 * _MAX_JOINT_STEP_RAD is 7.5 rad, which
# covers any reasonable single move.
_MAX_INTERP_STEPS = 250

# Orientation matters for grasping: an IK branch can hit the target position with the
# gripper rotated, and a wrong closing axis shoves the object instead of gripping it. So
# _resolve_ik scores position and orientation together and move() verifies both. These
# are the tolerance and the millimetre-per-degree trade-off weight.
_MOVE_ORI_TOL_DEG = 6.0
_ORI_ERR_WEIGHT_MM_PER_DEG = 3.0


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Smallest rotation angle (degrees) between two XYZW quaternions."""
    dot = abs(float(np.dot(np.asarray(q1, dtype=np.float64), np.asarray(q2, dtype=np.float64))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, dot))))

# The RMPflow approach tolerance in metres: stop driving the reactive policy once this
# close and let the IK refine finish. About 30 mm sits comfortably outside the roughly
# 19 mm steady state of RMPflow.
_RMP_APPROACH_TOL_M = 0.03

# cuRobo trajectory execution. Per waypoint, step until the arm is within this joint
# tolerance of it, so the arm tracks the collision-free path, capped at this many physics
# steps, so a hard-to-reach waypoint cannot hang the follow. Tracking to a tolerance is
# what makes the follow robust across arbitrary cuRobo plans.
_CUROBO_WP_TOL_RAD = 0.03
_CUROBO_MAX_STEPS_PER_WP = 12

# The six arm joints, by name. The Isaac robot is a combined 12-DoF articulation: the
# arm, the Robotiq 2F-85 finger_joint and its five mimics. The driver addresses these six
# by name through an ArticulationSubset, which makes it robust to the gripper joints and
# their dof ordering.
_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


# The gripper mount transform for a Robotiq 2F-85 on the UR5e flange, measured on-box:
# the gripper approach axis is the flange +Y, where the flange is the Lula EE `tool0`,
# its closing axis is the flange +X, and the grasp centre, the pad centre, sits
# `tcp_offset_mm` along the flange +Y. The grasp and TCP frame here is X for closing, Y
# for the binormal and Z for approach, so the flange-to-TCP rotation is a fixed -90 deg
# about the flange X, which maps the flange +Y onto the TCP +Z, the approach. fk and
# get_tcp_pose compose flange to TCP, and ik composes TCP to flange before handing the
# flange target to Lula.
_FLANGE_TO_TCP_QUAT_XYZW = np.array([-0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float64)


def _flange_to_tcp(
    flange_pos_mm: np.ndarray,
    flange_quat_xyzw: np.ndarray,
    offset_mm: "tuple[float, float, float]",
    rotation_quat_xyzw: "tuple[float, float, float, float]",
):
    """Map a flange pose, meaning tool0, to the gripper grasp-centre pose, meaning the TCP.

    Both halves are arguments. A transform that is half config and half source is one a
    config-driven cell cannot state: a scalar offset here with a module-level quaternion
    beside it would let a mounted gripper change its width profile while the arm kept
    the 2F-85 geometry.
    """
    flange_quat = np.asarray(flange_quat_xyzw, dtype=np.float64)
    tcp_quat = multiply(flange_quat, np.asarray(rotation_quat_xyzw, dtype=np.float64))
    tcp_pos = np.asarray(flange_pos_mm, dtype=np.float64) + rotate_vector(
        flange_quat, np.asarray(offset_mm, dtype=np.float64)
    )
    return tcp_pos, tcp_quat


def _tcp_to_flange(
    tcp_pos_mm: np.ndarray,
    tcp_quat_xyzw: np.ndarray,
    offset_mm: "tuple[float, float, float]",
    rotation_quat_xyzw: "tuple[float, float, float, float]",
):
    """Map a gripper grasp-centre (TCP) target to the flange (tool0) target for Lula IK."""
    tcp_quat = np.asarray(tcp_quat_xyzw, dtype=np.float64)
    flange_quat = multiply(tcp_quat, conjugate(np.asarray(rotation_quat_xyzw, dtype=np.float64)))
    flange_pos = np.asarray(tcp_pos_mm, dtype=np.float64) - rotate_vector(
        flange_quat, np.asarray(offset_mm, dtype=np.float64)
    )
    return flange_pos, flange_quat


# The capability surface for the real Isaac driver, which has native FK and IK.
# ``IsaacRobotArm.capabilities`` downgrades ``has_native_fk`` and ``has_native_ik`` to
# False in mock_mode, because the pure-Python mock has no kinematics, so a safety guard
# never calls fk or ik on a mock arm.
ISAAC_CAPABILITIES = RobotCapabilities(
    vendor="sim",
    model="isaac-sim",
    dof=6,
    supports_joint_move=True,
    supports_linear_move=True,
    supports_async_move=False,
    has_native_fk=True,
    has_native_ik=True,
    has_force_control=False,
    is_simulated=True,
)


class IsaacRobotArm(RobotArm):
    """The Isaac-backed :class:`RobotArm` adapter.

    Parameters
    ----------
    config
        The :class:`SimRobotConfig` describing the scene, the prim paths and the runtime
        knobs. Construction does not touch Isaac.
    safety_preflight
        An optional vendor-neutral :class:`SafetyPreflight`. Where one is supplied,
        ``move()`` routes through it before commanding motion.
    """

    def __init__(
        self,
        config: SimRobotConfig,
        *,
        safety_preflight: "SafetyPreflight | None" = None,
    ) -> None:
        self._config = config
        self._session = IsaacSimSession(config)
        self._connected = False
        self._preflight: "SafetyPreflight | None" = safety_preflight

        # The real Isaac handles, populated in connect() and None otherwise. They are
        # typed through the structural Protocol stubs, so attribute access on the
        # lazily imported Isaac objects type-checks without importing isaacsim here.
        self._articulation: IsaacArticulation | None = None
        self._arm_subset: IsaacArmSubset | None = None  # ArticulationSubset over the 6 _ARM_JOINT_NAMES
        self._kin_solver: IsaacKinematicsSolver | None = None
        self._rmpflow: IsaacRmpFlow | None = None
        self._amp: IsaacMotionPolicy | None = None
        self._dof_names: list[str] | None = None
        # Opt-in. At True, _resolve_ik prefers the natural shoulder-pan branch, where
        # joint_0 is atan2(target_y, target_x), over the flipped branch seeded from the
        # park pi. The flipped one reaches the same TCP and folds the wrist into the
        # forearm, measured at 43 mm of self-penetration. The default of False leaves the
        # seed set and the scoring unchanged.
        self._natural_aim_seed = False
        # The opt-in continuous collision-avoidance guard, where ``None`` is off and
        # leaves behaviour unchanged. Where it is set, as the preflight is, move_joint
        # checks every interpolation waypoint and halts on a margin breach or a
        # fail-safe. It is software avoidance and not certified safety.
        self._continuous_monitor: ContinuousCollisionMonitor | None = None
        # The motion-planning seam. "ik" is the blind pure-IK straight line. "rmpflow"
        # drives the reactive collision-aware policy toward the pose before the IK snap,
        # avoiding the obstacles registered through register_planner_obstacles. It is
        # overridable, as _natural_aim_seed and _continuous_monitor are, so a runner
        # flips it before the config carries it.
        self._motion_planner: str = config.motion_planner
        # The lazily spawned cuRobo planning client, a process-isolated server, created
        # only where motion_planner is "curobo" and a move() is issued. ``None`` is off,
        # and disconnect() closes it.
        self._curobo_client: CuroboPlanClient | None = None
        # Opt-in: plan move_joint() through cuRobo rather than interpolating a straight
        # line in joint space. It is off by default because it changes the trajectory of
        # every move_joint call, and a new seam ships default-off until a runner measures
        # it. A runner sets it, as it sets _natural_aim_seed and _continuous_monitor,
        # before the config carries it.
        self._plan_joint_moves: bool = False
        # The cuRobo fallback. Where motion_planner is "curobo" and the sidecar or its
        # environment is unavailable, for want of an environment or a python or because
        # the server boot fails, the first move() probes once, logs a warning, latches
        # this flag and degrades to the blind "ik" path. That is what makes cuRobo by
        # default safe on a box without the cuRobo environment instead of failing every
        # motion.
        self._curobo_unavailable: bool = False
        #: Why it became unavailable, kept so a runner can report the cause and not just the symptom.
        self._curobo_unavailable_reason: str = ""
        #: Obstacles re-sent underneath every scene registration; see set_planner_base_world.
        self._planner_base_world: list[dict] = []
        #: The cell as the cameras see it, asked immediately before every plan. `None` is the
        #: unchanged path: the planner keeps whatever world a runner last registered.
        self._live_world: "LivePlannerWorld | None" = None
        #: What the most recent refresh did, for a report and for an operator asking why.
        self._last_world_refresh: "WorldRefresh | None" = None

        # The mock-mode kinematic cache, which is also the placeholder state before
        # connect. In mock_mode the motion methods update these in place and the reads
        # return them.
        self._tcp: Pose = Pose(
            position_mm=np.array([400.0, 0.0, 300.0], dtype=np.float64),
            quaternion_xyzw=IDENTITY_QUAT_XYZW.copy(),
            frame=Frame.BASE,
            label="isaac-home-placeholder",
        )
        self._home_tcp: Pose = self._tcp
        home = config.home_joint_positions
        self._joints = JointPositions(
            np.asarray(
                home if home is not None else (0.0,) * ISAAC_CAPABILITIES.dof,
                dtype=np.float64,
            )
        )
        self._home_joints = self._joints

    @property
    def safety_preflight(self) -> "SafetyPreflight | None":
        """The guard pipeline every motion of this arm passes through.

        It implements :class:`~src.robot.safety.attestation.SafetyGated`, so a caller
        asks what this arm will refuse without reaching into `_preflight`. That matters
        for the library API: a caller supplies their own arm, which enumerating the
        driver registry cannot see, so the answer travels with the object.
        """
        return self._preflight

    @property
    def mock_mode(self) -> bool:
        """Whether this arm runs the pure-Python kinematic mock (immutable per session)."""
        return self._config.mock_mode

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> RobotCapabilities:
        """The honest capability surface.

        Real Isaac has native FK and IK. In ``mock_mode`` the pure-Python mock has no
        kinematics, so ``has_native_fk`` and ``has_native_ik`` report False; otherwise a
        safety guard would call ``fk`` or ``ik`` on a mock arm and fail.
        """
        if self._config.mock_mode:
            return replace(ISAAC_CAPABILITIES, has_native_fk=False, has_native_ik=False)
        return ISAAC_CAPABILITIES

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def session(self) -> IsaacSimSession:
        """Expose the owning :class:`IsaacSimSession` for tests / tooling."""
        return self._session

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Bring the Isaac session up and, outside mock mode, wrap the articulation and kinematics.

        Raises
        ------
        IsaacNotAvailableError
            On a host without the Isaac SDK, outside mock mode.
        RobotConnectionError
            Where ``robot_prim_path`` is not configured for the non-mock path.
        """
        self._session.start()
        if not self._config.mock_mode:
            self._bring_up_articulation()
        self._connected = True

    def _bring_up_articulation(self) -> None:
        """Wrap the articulation and build the Lula kinematics solver, outside mock mode.

        Every ``isaacsim.*`` import is lazy here, so the module stays importable on a
        host without Isaac.
        """
        if not self._config.robot_prim_path:
            raise RobotConnectionError(
                "IsaacRobotArm.connect (non-mock) requires SimRobotConfig.robot_prim_path."
            )
        from isaacsim.core.prims import SingleArticulation  # type: ignore[import-not-found]
        from isaacsim.robot_motion.motion_generation import (  # type: ignore[import-not-found]
            interface_config_loader,
        )
        from isaacsim.robot_motion.motion_generation.lula.kinematics import (  # type: ignore[import-not-found]
            LulaKinematicsSolver,
        )

        articulation = SingleArticulation(prim_path=self._config.robot_prim_path)
        articulation.initialize()

        # The robot is a combined 12-DoF articulation, the arm plus the 2F-85 gripper
        # variant. Addressing the six arm joints by name through an ArticulationSubset
        # makes every arm read and write ignore the gripper DoFs and stay robust to their
        # dof ordering.
        from isaacsim.core.api.articulations import ArticulationSubset  # type: ignore[import-not-found]

        dof_names = list(articulation.dof_names)
        missing = [n for n in _ARM_JOINT_NAMES if n not in dof_names]
        if missing:
            raise IsaacNotAvailableError(
                f"Arm joints {missing} not in the articulation dof_names {dof_names}."
            )
        arm_subset = ArticulationSubset(articulation, list(_ARM_JOINT_NAMES))

        if self._config.home_joint_positions is not None:
            home = np.asarray(self._config.home_joint_positions, dtype=np.float64)
            arm_subset.set_joint_positions(home)         # teleport the state
            arm_subset.apply_action(joint_positions=home)  # And hold it as the drive target, so the
            self._session.step_n(2)                       # arm does not drift during later stepping

        # The Lula solver and the RMPflow policy for the configured UR model, where
        # robot_model selects the Lula supported-config key such as "UR5e" or "UR3e".
        # Every UR e-series shares the link and joint names, so only the Lula config, its
        # link lengths, differs, and the rest of the driver is model-agnostic.
        from src.robot.drivers.sim.robot_models import ur_model_spec

        lula_key = ur_model_spec(self._config.robot_model).lula_key
        lula_cfg = interface_config_loader.load_supported_lula_kinematics_solver_config(lula_key)
        solver = LulaKinematicsSolver(**lula_cfg)
        frames = list(solver.get_all_frame_names())
        if _EE_FRAME not in frames:
            raise IsaacNotAvailableError(
                f"End-effector frame {_EE_FRAME!r} not found in the {lula_key} Lula description "
                f"(frames: {frames})."
            )

        # The RMPflow motion policy for the Cartesian move() path. The config already
        # carries end_effector_frame_name='tool0', and ArticulationMotionPolicy binds it
        # to this articulation at the fixed step_dt_s.
        from isaacsim.robot_motion.motion_generation import (  # type: ignore[import-not-found]
            ArticulationMotionPolicy,
        )
        from isaacsim.robot_motion.motion_generation.lula.motion_policies import (  # type: ignore[import-not-found]
            RmpFlow,
        )

        rmp_cfg = interface_config_loader.load_supported_motion_policy_config(lula_key, "RMPflow")
        rmpflow = RmpFlow(**rmp_cfg)
        amp = ArticulationMotionPolicy(
            articulation, rmpflow, default_physics_dt=self._config.step_dt_s
        )

        self._articulation = articulation
        self._arm_subset = arm_subset
        self._kin_solver = solver
        self._rmpflow = rmpflow
        self._amp = amp
        self._dof_names = dof_names

    def disconnect(self) -> None:
        if self._curobo_client is not None:
            self._curobo_client.close()  # shut down the process-isolated cuRobo server
            self._curobo_client = None
        self._articulation = None
        self._arm_subset = None
        self._kin_solver = None
        self._rmpflow = None
        self._amp = None
        self._dof_names = None
        self._session.stop()
        self._connected = False

    # ------------------------------------------------------------------
    # State (live reads in non-mock; cached in mock)
    # ------------------------------------------------------------------

    def get_tcp_pose(self) -> Pose:
        if not self._connected:
            raise RobotConnectionError("IsaacRobotArm is not connected.")
        if self._config.mock_mode or self._arm_subset is None or self._kin_solver is None:
            return self._tcp
        joints = np.asarray(self._arm_subset.get_joint_positions(), dtype=np.float64)
        pos_m, rot = self._kin_solver.compute_forward_kinematics(_EE_FRAME, joints)
        tool0 = isaac_pose_to_willy(np.asarray(pos_m), isaac_rotmat_to_wxyz(rot), label="isaac-tcp")
        tcp_pos, tcp_quat = _flange_to_tcp(
            tool0.position_mm, tool0.quaternion_xyzw,
            self._config.tool_offset_mm, self._config.tool_rotation_quat_xyzw,
        )
        return Pose(position_mm=tcp_pos, quaternion_xyzw=tcp_quat, frame=Frame.BASE, label="isaac-tcp")

    def get_joint_positions(self) -> JointPositions:
        if not self._connected:
            raise RobotConnectionError("IsaacRobotArm is not connected.")
        if self._config.mock_mode or self._arm_subset is None:
            return self._joints
        return isaac_joints_to_willy(self._arm_subset.get_joint_positions())

    # ------------------------------------------------------------------
    # Kinematics (real in non-mock; mock has none)
    # ------------------------------------------------------------------

    def _check_dof(self, joints: JointPositions, what: str) -> None:
        n = int(np.asarray(joints.values).reshape(-1).shape[0])
        if n != ISAAC_CAPABILITIES.dof:
            raise RobotMotionRejected(
                f"IsaacRobotArm.{what} expects {ISAAC_CAPABILITIES.dof} joints; got {n}."
            )

    def fk(self, joints: JointPositions) -> Pose:
        if self._config.mock_mode:
            raise IsaacNotAvailableError(
                "IsaacRobotArm.fk requires a non-mock Isaac session; mock_mode has no kinematics."
            )
        if not self._connected or self._kin_solver is None:
            raise RobotConnectionError("IsaacRobotArm.fk requires a connected articulation.")
        self._check_dof(joints, "fk")
        pos_m, rot = self._kin_solver.compute_forward_kinematics(
            _EE_FRAME, willy_joints_to_isaac(joints)
        )
        tool0 = isaac_pose_to_willy(np.asarray(pos_m), isaac_rotmat_to_wxyz(rot), label="isaac-fk")
        tcp_pos, tcp_quat = _flange_to_tcp(
            tool0.position_mm, tool0.quaternion_xyzw,
            self._config.tool_offset_mm, self._config.tool_rotation_quat_xyzw,
        )
        return Pose(position_mm=tcp_pos, quaternion_xyzw=tcp_quat, frame=Frame.BASE, label="isaac-fk")

    def ik(
        self,
        pose: Pose,
        *,
        seed: JointPositions | None = None,
    ) -> JointPositions:
        # Validate the frame before any conversion, for contract parity with the other
        # drivers.
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"IsaacRobotArm.ik requires Frame.BASE; got {pose.frame!r}."
            )
        if self._config.mock_mode:
            raise IsaacNotAvailableError(
                "IsaacRobotArm.ik requires a non-mock Isaac session; mock_mode has no kinematics."
            )
        if not self._connected or self._kin_solver is None:
            raise RobotConnectionError("IsaacRobotArm.ik requires a connected articulation.")

        # The pose is the TCP target, the grasp centre, so it is mapped through the
        # gripper mount onto the flange target, tool0, that Lula solves for.
        flange_pos, flange_quat = _tcp_to_flange(
            np.asarray(pose.position_mm, dtype=np.float64),
            np.asarray(pose.quaternion_xyzw, dtype=np.float64),
            self._config.tool_offset_mm, self._config.tool_rotation_quat_xyzw,
        )
        tool0_target = Pose(
            position_mm=flange_pos, quaternion_xyzw=flange_quat, frame=Frame.BASE, label=pose.label
        )
        target_pos_m, target_wxyz = willy_pose_to_isaac(tool0_target)
        kwargs: dict[str, object] = {}
        if seed is not None:
            self._check_dof(seed, "ik seed")
            kwargs["warm_start"] = willy_joints_to_isaac(seed)

        joints, success = self._kin_solver.compute_inverse_kinematics(
            _EE_FRAME, target_pos_m, target_wxyz, **kwargs
        )
        if not success or joints is None:
            raise RobotKinematicsError(
                f"IsaacRobotArm.ik did not converge for pose {pose.label!r}."
            )
        joints = np.asarray(joints, dtype=np.float64).reshape(-1)
        if joints.shape[0] != ISAAC_CAPABILITIES.dof:
            raise RobotKinematicsError(
                f"IK returned {joints.shape[0]} joints; expected {ISAAC_CAPABILITIES.dof}."
            )
        return isaac_joints_to_willy(joints)

    # ------------------------------------------------------------------
    # Motion (skeleton: Step 4b/4d will implement the non-mock path)
    # ------------------------------------------------------------------

    def reset_safety_continuity(self) -> None:
        """Invalidate the motion-continuity reference of the SafetyPreflight, its previous target.

        Call it after a deliberate trajectory discontinuity, such as parking the arm out
        of an overhead camera view through ``move_joint`` for perception, so the next
        ``move()`` is not compared across the park, which the continuity guard would
        otherwise read as a huge spurious joint and TCP step. It does nothing without a
        preflight. A ``move_joint`` driven internally by move() does not touch it; only
        explicit repositioning by a caller does.
        """
        if self._preflight is not None:
            self._preflight.reset()

    def move_to_joints(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> MotionResult:
        """The typed joint move: gate the destination through the preflight, then drive.

        It returns the typed rejection where :meth:`move_joint` raises it. The gate is
        the same one, run once here rather than twice by delegating to the public
        method.
        """
        if self._preflight is not None:
            rejected = self._preflight.gate_joint_target(joints, arm=self)
            if rejected is not None:
                return rejected
        self._drive_joints(joints, velocity=velocity, acceleration=acceleration)
        return MotionResult.executed(
            MotionCommand.MOVE_JOINTS, target_joints=joints, message="move_to_joints",
        )

    def move_joint(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        """A gated joint-space move.

        The Protocol promises ``RobotMotionRejected`` where a safety pre-check denies
        the move, and runner code reaches this surface directly rather than only through
        :meth:`move_to_joints`. :meth:`move_home` routes through here too, which is what
        closes it.

        The gate runs before the mock branch on purpose. A configuration the guard would
        refuse on the articulation is one it refuses on the mock as well, because a mock
        that accepts what the real driver rejects hides regressions.
        """
        if self._preflight is not None:
            rejected = self._preflight.gate_joint_target(joints, arm=self)
            if rejected is not None:
                raise RobotMotionRejected(
                    f"move_joint refused by the safety preflight: {rejected.message}"
                )
        self._drive_joints(joints, velocity=velocity, acceleration=acceleration)

    def _drive_joints(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        """Command the joints with no safety gate. Callers must have gated the destination first."""
        if self.mock_mode:
            # The identity mock: commit the commanded joints. The TCP is not touched,
            # because the mock has no FK, and a caller needing a consistent TCP uses
            # move_linear or move.
            self._joints = joints
            return
        if not self._connected or self._arm_subset is None:
            raise RobotConnectionError(
                "IsaacRobotArm.move_joint requires a connected articulation."
            )
        self._check_dof(joints, "move_joint")
        # Drive the six arm joints alone: the subset maps them onto their indices in the
        # combined articulation and leaves the gripper DoFs untouched. velocity and
        # acceleration are accepted for Protocol parity and do not apply, because the
        # drive is position-controlled and walks fixed _MAX_JOINT_STEP_RAD waypoints at
        # the fixed step_dt_s cadence. Honouring them would need a velocity or torque
        # control mode.
        #
        # A large move is interpolated into small increments, because one big
        # point-to-point command makes the position drive stall partway, reaching
        # equilibrium short of the target and staying there. Confirmed on-box: a home to
        # grasp command settled with a 1.5 rad elbow error and never recovered. Walking
        # waypoints of about `_MAX_JOINT_STEP_RAD` keeps every command drivable, so the
        # arm tracks the whole path, and a final settle converges the last of it.
        target = np.asarray(willy_joints_to_isaac(joints), dtype=np.float64)
        if self._arm_subset is None:
            return
        start = np.asarray(self._arm_subset.get_joint_positions(), dtype=np.float64)
        mon = self._continuous_monitor  # H3.9g B2: opt-in (None == off == byte-identical)

        # The planned joint move, opt-in. The straight line below is exactly that, a
        # linear interpolation in joint space, which knows nothing about geometry and can
        # sweep one link through another. That is not theoretical: the first thing the
        # continuous guard did once it had a mesh backend was refuse this path in
        # run_fused_pick, at forearm against lfinger, 5.828 mm.
        #
        # With `plan_joint_moves` the path comes from cuRobo instead, in joint space, so
        # the goal configuration is the one that was asked for rather than any IK branch
        # reaching the same tool pose. It is off by default, because it changes the
        # trajectory of every move_joint call and each runner that adopts it owes its own
        # on-box measurement.
        waypoints: list[np.ndarray] | None = None
        if self._plan_joint_moves and self._motion_planner == "curobo":
            planned = self._plan_joint_path(start, target)
            if planned is None:
                # Fail closed. Interpolating here silently would hand back exactly the
                # blind path this option exists to replace, at the moment it is known to
                # be unsafe.
                raise CuroboUnavailableError(
                    f"cuRobo found no collision-free joint path to {np.round(target, 3).tolist()}. "
                    f"Refusing to interpolate blindly; that is the path the guard already rejects."
                )
            waypoints = [np.asarray(q, dtype=np.float64) for q in planned]

        if waypoints is None:
            n_steps = max(1, int(np.ceil(float(np.max(np.abs(target - start))) / _MAX_JOINT_STEP_RAD)))
            n_steps = min(n_steps, _MAX_INTERP_STEPS)
            waypoints = [start + (target - start) * (i / n_steps) for i in range(1, n_steps + 1)]

        for waypoint in waypoints:
            if mon is not None:  # check the next config before moving into it; halt (never apply) on stop
                verdict = mon.check(waypoint)  # waypoint is already UR-order joints (subset == UR order)
                if verdict.stop:
                    raise ContinuousGuardAbort(verdict)
            self._arm_subset.apply_action(joint_positions=waypoint)
            self._session.step()
        if mon is not None:
            verdict = mon.check(target)
            if verdict.stop:
                raise ContinuousGuardAbort(verdict)
        self._arm_subset.apply_action(joint_positions=target)
        self.wait_until_steady(timeout_s=self._config.settle_timeout_s)

    def move_linear(
        self,
        pose: Pose,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        if self.mock_mode:
            if pose.frame is not Frame.BASE:
                raise FrameMismatchError(
                    "IsaacRobotArm.move_linear requires Frame.BASE; "
                    f"got {pose.frame!r}."
                )
            self._tcp = pose
            return
        result = self.move(pose, linear=True, vel=velocity, acc=acceleration)
        if result.status is not MotionStatus.EXECUTED:
            raise RobotMotionRejected(
                f"IsaacRobotArm.move_linear failed: {result.status.value}: {result.message}"
            )

    def stop(self) -> None:
        """A best-effort halt that holds the arm at its current configuration. Always safe.

        In mock mode, or before connect, it does nothing. Otherwise it re-commands the
        current joint positions, so the position drive holds station instead of tracking
        a stale target, then drops the preflight continuity memo, because a stop is a
        deliberate trajectory discontinuity. It never raises.

        It does not interrupt a ``move()`` already iterating its waypoint loop. The sim
        has no hardware e-stop, and this only re-targets the drive so the next step
        holds.
        """
        if self.mock_mode or not self._connected or self._arm_subset is None:
            return None
        try:
            current = np.asarray(self._arm_subset.get_joint_positions(), dtype=np.float64)
            self._arm_subset.apply_action(joint_positions=current)  # hold station
        except Exception:  # noqa: BLE001 (stop() must never raise (RobotArm Protocol contract))
            pass
        if self._preflight is not None:
            self._preflight.reset()
        return None

    # ------------------------------------------------------------------
    # High-level pipeline surface (Phase N typed contract)
    # ------------------------------------------------------------------

    def is_inside_workspace(self, pose: Pose) -> bool:
        """Always ``True``, because this driver owns no workspace box.

        ``SimRobotConfig`` carries no Cartesian limits. The box belongs to the
        ``WorkspaceGuard`` inside the injected ``SafetyPreflight``, which
        ``sim_safety_preflight`` builds from ``cfg.robot.workspace_limits``, and it is
        enforced on every Cartesian command in :meth:`move`. The ``RobotArm`` Protocol
        requires a driver with no configured workspace to return ``True`` here, and the
        Frame.BASE check is kept for contract parity with the other drivers.
        """
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                "IsaacRobotArm.is_inside_workspace requires Frame.BASE; "
                f"got {pose.frame!r}."
            )
        return True

    def move_to(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> bool:
        if self.mock_mode:
            if not self._connected or pose.frame is not Frame.BASE:
                return False
            self._tcp = pose
            return True
        return (
            self.move(pose, linear=linear, vel=vel, acc=acc, register=register).status
            is MotionStatus.EXECUTED
        )

    def move_home(self) -> bool:
        """Drive to the configured home, gated like every other joint move.

        The gate runs before the mock branch and before the real one. The home
        configuration is a place the arm will sit, and a mock that committed one the
        guard would refuse is how a regression stays invisible off-box. It returns
        ``bool`` like its two siblings, and the typed reason is logged.
        """
        if self._preflight is not None and self._home_joints is not None:
            home = JointPositions(np.asarray(self._home_joints, dtype=np.float64))
            rejected = self._preflight.gate_joint_target(home, arm=self)
            if rejected is not None:
                _LOGGER.error(
                    "move_home REFUSED by %s: %s", rejected.status, rejected.message,
                )
                return False
        if self.mock_mode:
            if not self._connected:
                return False
            self._tcp = self._home_tcp
            self._joints = self._home_joints
            return True
        # Drive to the configured home through the interpolated move_joint, which is the
        # path the runners use for homing between runs. A home move is a deliberate
        # trajectory restart, so the preflight continuity memo is cleared and the next
        # move() is not compared across it, mirroring reset_safety_continuity.
        if not self._connected or self._arm_subset is None or self._home_joints is None:
            return False
        if self._preflight is not None:
            self._preflight.reset()
        self.move_joint(self._home_joints)
        return True

    def wait_until_steady(
        self,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> bool:
        """Step the sim until the arm stops, or until ``timeout_s`` sim seconds elapse.

        It returns ``True`` once the largest absolute joint velocity is below
        ``_SETTLE_VEL_THRESHOLD`` in rad/s, and ``False`` on timeout. In mock mode, and
        before connect, nothing is in flight, so it returns ``True`` immediately.
        ``poll_interval_s`` is accepted for interface parity, because in sim the check
        runs on every physics step, at the fixed ``step_dt_s`` cadence.
        """
        if self._config.mock_mode or self._arm_subset is None:
            return True
        dt = self._config.step_dt_s if self._config.step_dt_s > 0 else 1.0 / 60.0
        max_steps = max(1, int(timeout_s / dt))
        for _ in range(max_steps):
            self._session.step()
            vel = np.asarray(self._arm_subset.get_joint_velocities(), dtype=np.float64)
            if float(np.max(np.abs(vel))) < _SETTLE_VEL_THRESHOLD:
                return True
        return False

    def _resolve_ik(self, pose: Pose) -> JointPositions:
        """Resolve IK for a TCP target robustly, across several seeds.

        A single warm start from the current configuration can land Lula on a far or
        joint-limited IK branch the drive cannot physically reach, as a top-down grasp
        whose closing axis forces an awkward wrist does. So it tries the current and home
        configurations and a handful of wrist-flipped seeds, which reach the other IK
        families, and keeps the solution whose FK best matches the target. That makes
        ``move()`` robust to any grasp orientation. It raises where none converges.
        """
        current = np.asarray(self.get_joint_positions().values, dtype=np.float64)
        pi = float(np.pi)
        bases = [current]
        if self._natural_aim_seed:
            # A forward-reach base aimed at the target XY, so the natural, non-self-
            # colliding IK branch is tried first. It then wins the early-out break at
            # every waypoint and the whole descent stays in the clean branch, at 0 mm,
            # instead of the flipped one seeded from the park pi, at 43 mm.
            tx, ty = float(pose.position_mm[0]), float(pose.position_mm[1])
            bases.insert(0, np.array([np.arctan2(ty, tx), -1.6, 1.6, -1.5, -1.57, 0.0], dtype=np.float64))
        if self._config.home_joint_positions is not None:
            bases.append(np.asarray(self._config.home_joint_positions, dtype=np.float64))
        seeds: list[np.ndarray] = []
        for base in bases:
            # Sweep wrist_3, the free rotation about the approach axis for a top-down
            # grasp. The reachable IK branch can sit at any wrist_3 and Lula returns only
            # the branch nearest the seed, so the full circle is covered to be sure of
            # finding a reachable, drivable one.
            for k in range(12):
                s = base.copy()
                s[5] = base[5] + k * (2.0 * pi / 12.0)
                seeds.append(s)
            seeds.append(base + np.array([0.0, 0.0, 0.0, pi, 0.0, 0.0]))  # wrist_1 flip
            seeds.append(base + np.array([0.0, 0.0, 0.0, 0.0, pi, 0.0]))  # wrist_2 flip

        target_pos = np.asarray(pose.position_mm, dtype=np.float64)
        target_quat = np.asarray(pose.quaternion_xyzw, dtype=np.float64)
        best: JointPositions | None = None
        best_err = float("inf")
        for seed in seeds:
            try:
                joints = self.ik(pose, seed=JointPositions(seed))
            except RobotKinematicsError:
                continue
            fk_pose = self.fk(joints)
            pos_err = float(np.linalg.norm(np.asarray(fk_pose.position_mm) - target_pos))
            ori_err = _quat_angle_deg(fk_pose.quaternion_xyzw, target_quat)
            # A combined cost, so a branch that hits the position with the gripper
            # rotated loses to one that matches both, because a wrong closing axis shoves
            # the object instead of gripping it. The accurate grasp orientation often
            # forces a wrist_3 flip of about pi from the home configuration, and that is
            # unavoidable for a correct grasp: the pi-equivalent orientation is not an
            # equivalent grasp for the 2F-85 here, measured as a -10 mm shove. The
            # IK-jump and continuity thresholds of the sim profile are widened to accept
            # the flip, which the interpolated drive walks safely.
            combined = pos_err + _ORI_ERR_WEIGHT_MM_PER_DEG * ori_err
            if combined < best_err:
                best_err, best = combined, joints
                if pos_err < _MOVE_POS_TOL_MM and ori_err < _MOVE_ORI_TOL_DEG:
                    break
        if best is None:
            raise RobotKinematicsError(
                f"IsaacRobotArm: no IK seed converged for pose {pose.label!r}."
            )
        # Unwind each revolute joint by plus or minus 2*pi to the branch nearest the
        # current configuration. A revolute joint at theta and at theta+2*pi is the
        # identical arm pose and FK, so this never changes the grasp, and it avoids
        # commanding a needless full turn such as a 360 deg wrist_3 wind or a 235 deg
        # elbow swing reachable as -125 deg. That keeps the interpolated motion short and
        # inside the motion-continuity and IK-jump envelope. A joint that would leave the
        # plus or minus 2*pi limit keeps the raw IK value, because the joint-limit guard
        # owns that envelope.
        best_arr = np.asarray(best.values, dtype=np.float64)
        two_pi = 2.0 * float(np.pi)
        unwound = best_arr - two_pi * np.round((best_arr - current) / two_pi)
        out_of_range = np.abs(unwound) > two_pi
        unwound[out_of_range] = best_arr[out_of_range]
        return JointPositions(unwound)

    def _drive_rmpflow(self, pose: Pose) -> bool:
        """Drive RMPflow toward a BASE-frame pose, a smooth collision-aware approach.

        It returns ``True`` once the end-effector is within ``_RMP_APPROACH_TOL_M`` of
        the target, meaning RMPflow routed there, and ``move()`` then refines to the
        exact IK solution, which removes the roughly 19 mm steady-state offset of
        RMPflow.

        It returns ``False`` where the ``settle_timeout_s`` step budget elapses without
        converging, meaning the reactive policy is blocked or trapped in a local
        minimum, as it is when a registered obstacle fully blocks the direct line, which
        ``mp_iso_avoid.py`` measures. The caller treats ``False`` as a fail-safe and does
        not blind-snap through the obstacle, which would defeat the avoidance.
        """
        # Bind the Isaac handles to locals, so the non-None narrowing holds across the
        # method calls in the loop. The five handles are set together in
        # _bring_up_articulation, so this guard does nothing on a connected path, and it
        # mirrors the RobotConnectionError convention of this file for an unconnected
        # articulation. move() pre-checks the same handles.
        rmpflow, amp = self._rmpflow, self._amp
        articulation, arm_subset, kin_solver = self._articulation, self._arm_subset, self._kin_solver
        if (
            rmpflow is None or amp is None or articulation is None
            or arm_subset is None or kin_solver is None
        ):
            raise RobotConnectionError(
                "IsaacRobotArm._drive_rmpflow requires a connected articulation."
            )
        # The RMPflow end-effector is tool0, the Lula frame, so the TCP target is mapped
        # through the gripper mount onto the tool0 target and the convergence check below
        # is consistent.
        flange_pos, flange_quat = _tcp_to_flange(
            np.asarray(pose.position_mm, dtype=np.float64),
            np.asarray(pose.quaternion_xyzw, dtype=np.float64),
            self._config.tool_offset_mm, self._config.tool_rotation_quat_xyzw,
        )
        tool0_pose = Pose(
            position_mm=flange_pos, quaternion_xyzw=flange_quat, frame=Frame.BASE, label=pose.label
        )
        target_pos_m, target_wxyz = willy_pose_to_isaac(tool0_pose)
        rmpflow.set_end_effector_target(
            target_position=target_pos_m, target_orientation=target_wxyz
        )
        dt = self._config.step_dt_s if self._config.step_dt_s > 0 else 1.0 / 60.0
        budget = max(1, int(self._config.settle_timeout_s / dt))
        for _ in range(budget):
            rmpflow.update_world()
            action = amp.get_next_articulation_action(dt)
            articulation.apply_action(action)
            self._session.step()
            q = np.asarray(arm_subset.get_joint_positions(), dtype=np.float64)
            pos_m, _ = kin_solver.compute_forward_kinematics(_EE_FRAME, q)
            if float(np.linalg.norm(np.asarray(pos_m) - target_pos_m)) < _RMP_APPROACH_TOL_M:
                return True
        return False

    def register_planner_obstacles(self, obstacles: "Iterable[Any]") -> int:
        """Register scene obstacles into the planner world model, the shared scene-to-planner seam.

        ``obstacles`` are Isaac core-API objects, such as a ``FixedCuboid``,
        ``DynamicCuboid`` or ``DynamicCylinder`` wrapping the bin walls and the
        neighbour-object prims, and the RMPflow approach then routes around them. The
        runner builds the objects, because this driver never imports Isaac, so they are
        duck-typed as ``Any``. It returns the count registered, and returns 0 without
        doing anything where no planner is up.

        Registration is loud. Where the planner rejects an obstacle, for a wrong type or
        any other reason, the underlying ``add_obstacle`` raises and surfaces the
        problem rather than leaving RMPflow obstacle-blind, which would read as RMPflow
        not helping. The ``update_world()`` inside ``_drive_rmpflow`` refreshes the poses
        of the registered obstacles at every step.
        """
        rmpflow = self._rmpflow
        if rmpflow is None:
            return 0
        added = 0
        for obstacle in obstacles:
            rmpflow.add_obstacle(obstacle)
            added += 1
        return added

    def _guard_self_collision_margin_mm(self) -> float:
        """The clearance to ask the planner for, read off the live guard of this arm.

        0.0 leaves the planner alone. It is read off the guard rather than re-read from
        config, so a runner that swapped the guard in code is still described truthfully.

        It is ``planner_margin_mm`` and not the ``min_distance_mm`` the guard enforces,
        because how much margin a planner can absorb depends on how tightly its spheres
        fit that robot. Deriving it from the enforced distance is what takes a UR3e cell
        from 10/10 to 0/10: it plans to 6 mm and finds no plan at 10 mm.
        """
        preflight = self._preflight
        if preflight is None:
            return 0.0
        for guard in getattr(preflight, "guards", ()) or ():
            margin = getattr(guard, "_planner_margin_mm", None)
            if isinstance(margin, (int, float)) and margin > 0.0:
                return float(margin)
        return 0.0

    def _plan_joint_path(
        self, start: "np.ndarray", target: "np.ndarray"
    ) -> list[list[float]] | None:
        """A collision-free joint path from ``start`` to ``target``, or ``None`` where cuRobo finds none.

        It returns ``None`` rather than raising for an unavailable planner too, so the
        one caller decides what an absent plan means, and it decides to refuse. Keeping
        that judgement in one place is the point: a helper that fell back to
        interpolation quietly would defeat the option entirely.
        """
        try:
            client = self._get_curobo_client()
        except Exception as exc:  # noqa: BLE001 (no planner is "no plan", reported by the caller)
            _LOGGER.warning("planned joint move requested but cuRobo is unavailable: %s", exc)
            return None
        stale = self._refresh_planner_world()
        if stale:
            # The one caller refuses on a `None` path, which is what a world nobody can
            # vouch for has to produce: a joint move plans through the same cell a pose
            # move does.
            _LOGGER.error("planned joint move refused. %s", stale)
            return None
        try:
            return client.plan_joint([float(q) for q in start], [float(q) for q in target])
        except CuroboUnavailableError as exc:
            _LOGGER.warning("cuRobo joint planning failed: %s", exc)
            return None

    def _get_curobo_client(self) -> CuroboPlanClient:
        """Lazily spawn + JIT-warm the process-isolated cuRobo planning server (blocks on the first call)."""
        if self._curobo_client is None:
            import os

            from src.robot.drivers.sim.robot_models import curobo_robot_yml
            from src.robot.safety.planning.environment import ENV_CUROBO_ROBOT

            # The cuRobo robot config is the {key}.yml of the configured model. The
            # config wins on purpose: a shell-scoped WILLY_CUROBO_ROBOT would otherwise
            # plan one robot cell against another robot geometry silently, such as a ur3e
            # cell planned with ur5e link lengths, with no visible symptom. A disagreeing
            # variable is therefore ignored loudly, and planning a different robot means
            # setting robot_model.
            robot_yml = curobo_robot_yml(self._config.robot_model)
            env_yml = os.environ.get(ENV_CUROBO_ROBOT)
            if env_yml and env_yml != robot_yml:
                _LOGGER.warning(
                    "%s=%r disagrees with the configured robot_model=%r (-> %r) and is IGNORED: planning a "
                    "cell against another robot's geometry is never safe. Set robot_model instead.",
                    ENV_CUROBO_ROBOT, env_yml, self._config.robot_model, robot_yml,
                )
            # Hand the planner the clearance the safety guard will demand of the
            # configuration it returns. Without it the two engines judge the same path by
            # different geometry and cuRobo keeps offering configurations the guard
            # refuses, measured on-box as 9.44 to 9.47 mm plans against a 10.000 mm guard
            # margin, which reads as a bad grasp rather than a rejected one.
            client = CuroboPlanClient(
                robot_config=robot_yml, self_collision_margin_mm=self._guard_self_collision_margin_mm(),
            )
            client.start()
            self._curobo_client = client
        return self._curobo_client

    def _resolve_motion_planner(self) -> str:
        """The effective planner for this move, with the cuRobo fallback.

        It returns ``self._motion_planner`` verbatim unless that is ``"curobo"`` and the
        cuRobo sidecar or environment cannot be brought up. Then it probes once, warns,
        latches ``_curobo_unavailable`` and returns ``"ik"``, so a ``"curobo"`` default
        degrades to the blind path on a host without the cuRobo environment instead of
        failing every motion. Once latched, a later move skips the probe.
        """
        if self._motion_planner != "curobo" or self.mock_mode:
            return self._motion_planner
        if self._curobo_unavailable:
            return "ik"
        try:
            self._get_curobo_client()  # idempotent: spawns + JIT-warms the server once, else raises
        except CuroboUnavailableError as exc:
            self._curobo_unavailable = True
            # Kept for the caller to read rather than only for a log line to scroll past.
            # A silent degradation here is invisible where it matters: the cell keeps
            # running, the blind IK path proposes configurations the self-collision guard
            # correctly refuses, and a validation run reports 0/10 with nothing tying
            # that back to a planner that never started. Measured once when an OS policy
            # blocked the sidecar CUDA extension, where the only evidence was this one
            # warning among thousands of lines.
            self._curobo_unavailable_reason = str(exc)
            _LOGGER.warning(
                "cuRobo planner unavailable (%s); falling back to the blind IK path for this arm.", exc,
            )
            return "ik"
        return "curobo"

    @property
    def curobo_degraded(self) -> bool:
        """``True`` where this arm was configured for cuRobo and is running blind IK instead.

        A runner that reports a pass rate reports this next to it, because the two
        numbers mean different things and a rate measured on the fallback path is not a
        measurement of the configured cell.
        """
        return self._motion_planner == "curobo" and self._curobo_unavailable

    @property
    def curobo_degraded_reason(self) -> str:
        """Why the planner could not start. Empty while nothing has degraded."""
        return self._curobo_unavailable_reason

    def set_planner_base_world(self, cuboids: "list[dict]") -> None:
        """The obstacles that survive every later registration: the bench, the fixtures, the cell.

        ``set_world`` replaces the planner world rather than extending it, so a runner
        registering its bin walls deletes everything that was there before, including
        the table the sidecar boots with. Anything declared here is re-sent underneath
        every later :meth:`set_curobo_world`, so scene obstacles are added to the cell
        rather than in place of it.
        """
        self._planner_base_world = [dict(c) for c in cuboids]

    def set_curobo_world(self, cuboids: list[dict]) -> int:
        """Register the scene obstacles into the cuRobo collision world.

        It is the scene-to-planner world model for the ``"curobo"`` planner, the
        cross-process analogue of register_planner_obstacles for RMPflow.

        ``cuboids`` are ``{"name", "dims_m":[x,y,z], "pose":[px,py,pz,qw,qx,qy,qz]}`` in
        the BASE frame, in metres, with a WXYZ quaternion, and the runner serialises the
        bin walls and neighbour bodies. It spawns the planning server where one is
        needed, and a later ``move()`` plan routes around these.

        The cell own base world, from :meth:`set_planner_base_world`, is sent underneath
        them, because ``set_world`` replaces rather than extends and a runner must not be
        able to delete the bench by declaring a wall. It returns the count registered,
        and says so where that is fewer than the count sent: the client returns 0 and
        keeps planning against the previous world on failure, and an obstacle the
        planner never received is one it routes straight through.
        """
        merged = merge_planner_worlds(self._planner_base_world, cuboids)
        try:
            count = self._get_curobo_client().set_world(merged)
        except CuroboUnavailableError:
            return 0
        if count != len(merged):
            _LOGGER.error(
                "the planner confirmed %d of %d obstacle(s): it is planning against a world that is "
                "missing part of this cell, and an obstacle it never received is one it will route "
                "straight through", count, len(merged),
            )
        return count

    @property
    def live_planner_world(self) -> "LivePlannerWorld | None":
        """What this arm asks before it plans, or `None` where nothing was handed in.

        Read by whatever holds the scene, which is the pick loop: it is the only thing that
        knows which object an attempt is reaching for, and that object has to be left out of
        the world or the planner refuses to approach it.
        """
        return self._live_world

    def set_live_planner_world(self, world: "LivePlannerWorld | None") -> None:
        """Hand this arm the thing that answers what the cell looks like right now.

        Handed in rather than built here, because a driver may not reach up into
        perception: the composition root owns the cameras and the transforms and this arm
        only ever asks. Setting `None` restores the behaviour of registering a world once
        and planning against it forever.
        """
        self._live_world = world

    @property
    def last_world_refresh(self) -> "WorldRefresh | None":
        """What the most recent refresh did, or `None` where none has run."""
        return self._last_world_refresh

    def _self_link_origins_mm(self) -> "list[list[float]] | None":
        """Where this arm's own links are, in BASE millimetres, for taking the robot out of the view.

        `None` where the model has no bundled kinematic chain or the arm is not connected.
        The world source treats that as a cell that cannot describe itself and refuses to
        build a perceived world, which is the right answer: a camera that can see the robot
        and a robot that cannot say where it is produce an obstacle exactly where the arm
        is standing.
        """
        if self._arm_subset is None:
            return None
        joints = np.asarray(self._arm_subset.get_joint_positions(), dtype=np.float64)
        ordered = np.asarray(
            [float(joints[_ARM_JOINT_NAMES.index(name)]) for name in _ARM_JOINT_NAMES],
            dtype=np.float64,
        )
        origins = ur_link_origins_mm(str(self._config.robot_model), ordered)
        # Plain numbers across the boundary: the world source is pure geometry and takes
        # points, not this repository's arrays.
        return None if origins is None else [[float(v) for v in point] for point in origins]

    def _refresh_planner_world(self, *, near_point_mm: "Sequence[float] | None" = None) -> str:
        """Refresh the planner world before a plan. An empty string means it may go ahead.

        It returns the reason instead of raising, because both callers already have a shape
        for refusing a motion and neither wants an exception mid-pick. Without a live world
        configured this does nothing and returns nothing.
        """
        if self._live_world is None:
            return ""
        refresh = refresh_planner_world(
            source=self._live_world,
            client=self._get_curobo_client(),
            link_origins_mm=self._self_link_origins_mm(),
            near_point_mm=near_point_mm,
        )
        self._last_world_refresh = refresh
        # The guard hears about the same obstacles, including the empty list when the world
        # could not be vouched for: a guard left holding boxes from a refused refresh is
        # checking a cell that no longer exists.
        if self._preflight is not None:
            self._preflight.set_perceived_obstacles(refresh.guard_boxes)
        if not refresh.ok:
            _LOGGER.error("%s", refresh.render())
            return refresh.render()
        _LOGGER.info("%s", refresh.render())
        return ""

    def _drive_curobo(
        self, pose: "Pose", *, current_joints: "JointPositions | None" = None
    ) -> MotionResult:
        """Plan a collision-free trajectory to ``pose`` with cuRobo and execute it.

        The planner owns the final motion. This maps the grasp TCP pose onto tool0, the
        cuRobo and Lula end-effector, and millimetres with XYZW onto metres with WXYZ,
        sends the current joints and the goal to the warm server, and follows the
        returned joint trajectory. It is ``EXECUTED`` where the final TCP is within
        tolerance, a fail-safe ``TIMEOUT`` where cuRobo found no collision-free plan, and
        ``CONTROLLER_REJECTED`` where the planning server is unavailable. No failure
        produces blind motion, because the planner owns the path.

        The safety preflight runs here rather than in ``move``, on the actual planned
        final configuration of cuRobo, so the high-fidelity Coal self-collision,
        joint-limit and workspace guards validate the branch that executes. The
        ``_resolve_ik`` of this arm can pick a self-colliding IK branch cuRobo never
        uses, which is measured.
        """
        if self._arm_subset is None:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, MotionCommand.MOVE_TO, target_pose=pose,
                message="IsaacRobotArm._drive_curobo requires a connected articulation.",
            )
        # Grasp TCP to tool0, the Lula end-effector, then millimetres to metres and XYZW
        # to WXYZ, which is the cuRobo goal convention.
        flange_pos_mm, flange_quat_xyzw = _tcp_to_flange(
            np.asarray(pose.position_mm, dtype=np.float64),
            np.asarray(pose.quaternion_xyzw, dtype=np.float64),
            self._config.tool_offset_mm, self._config.tool_rotation_quat_xyzw,
        )
        goal_pos_m = (np.asarray(flange_pos_mm, dtype=np.float64) / 1000.0).tolist()
        fqx, fqy, fqz, fqw = (float(v) for v in flange_quat_xyzw)
        goal_quat_wxyz = [fqw, fqx, fqy, fqz]
        arm_q = np.asarray(self._arm_subset.get_joint_positions(), dtype=np.float64)  # _ARM_JOINT_NAMES order
        try:
            client = self._get_curobo_client()
            stale = self._refresh_planner_world(near_point_mm=[float(v) for v in pose.position_mm])
            if stale:
                return MotionResult.failed(
                    MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                    message=f"planner world not refreshed, so this move is refused. {stale}",
                )
            start = [float(arm_q[_ARM_JOINT_NAMES.index(n)]) for n in client.joint_names]  # -> server order
            traj = client.plan(start, goal_pos_m, goal_quat_wxyz)
        except CuroboUnavailableError as exc:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                message=f"cuRobo planner unavailable: {exc}",
            )
        if traj is None:
            # It is fail-safe and opaque: downstream this is a bare timeout,
            # indistinguishable from a bad grasp. Which pose the planner could not route
            # to, and from which start configuration, is what separates a planner
            # misconfigured for this robot from a pose that is genuinely blocked, so both
            # are stated.
            _LOGGER.warning(
                "cuRobo found NO plan: TCP target=%s -> flange goal=%s quat_wxyz=%s from start joints=%s "
                "(server order %s); failing safe, no blind motion.",
                np.round(np.asarray(pose.position_mm), 1).tolist(),
                np.round(np.asarray(goal_pos_m) * 1000.0, 1).tolist(),
                [round(v, 4) for v in goal_quat_wxyz],
                np.round(arm_q, 3).tolist(), [round(v, 3) for v in start],
            )
            return MotionResult.failed(
                MotionStatus.TIMEOUT, MotionCommand.MOVE_TO, target_pose=pose,
                message=NO_PLAN_FAIL_SAFE_MESSAGE,
            )
        # The safety gate on the planned final configuration of cuRobo, the branch that
        # executes: the Coal self-collision, joint-limit and workspace guards on the real
        # motion rather than on the independent _resolve_ik branch. cuRobo guarantees a
        # self-collision-free path, and this re-asserts it under the fail-closed
        # contract, catching a configuration that trips a sim-only limit or box.
        if self._preflight is not None:
            order = [client.joint_names.index(n) for n in _ARM_JOINT_NAMES]  # server order -> _ARM_JOINT_NAMES
            final_q = JointPositions(np.asarray([traj[-1][i] for i in order], dtype=np.float64))
            # cuRobo owns trajectory continuity, having planned a smooth, dynamically
            # feasible, collision-checked path, so the continuity guards are skipped: the
            # ik_quality joint-jump and the motion_continuity step size. A large net
            # joint delta against the start, such as a shoulder wrap between +pi and -pi
            # that is no physical rotation at all, is not a blind interpolator
            # teleporting. The static guards, workspace, joint-limit, self-collision and
            # payload, still gate the final configuration, which keeps the Coal
            # self-collision gate. Measured: the ik_quality joint-jump was the real
            # blocker on the cuRobo path and self-collision never fired.
            rejected = self._preflight_reject(
                pose, target_joints=final_q, current_joints=current_joints,
                skip_guards=frozenset({"ik_quality", "motion_continuity"}),
            )
            if rejected is not None:
                return rejected
            # And the path, where one is asked for. `_execute_curobo_trajectory` applies
            # every waypoint straight to the articulation and never routes through
            # `move_joint`, which is the only place the in-motion monitor is consulted,
            # so without this the middle of a cuRobo plan executes unexamined on this
            # driver too.
            ordered = [[float(wp[i]) for i in order] for wp in traj]
            rejected = self._preflight.gate_trajectory(ordered, arm=self)
            if rejected is not None:
                return rejected
        self._execute_curobo_trajectory(traj, client.joint_names, client.dt)
        tcp = self.get_tcp_pose()
        err_mm = float(np.linalg.norm(np.asarray(tcp.position_mm) - np.asarray(pose.position_mm)))
        ori_err_deg = _quat_angle_deg(tcp.quaternion_xyzw, pose.quaternion_xyzw)
        if err_mm <= _MOVE_POS_TOL_MM and ori_err_deg <= _MOVE_ORI_TOL_DEG:
            return MotionResult.executed(
                MotionCommand.MOVE_TO, target_pose=pose,
                message=f"cuRobo trajectory reached target ({err_mm:.1f} mm, {ori_err_deg:.1f} deg).",
            )
        _LOGGER.warning(
            "cuRobo trajectory ended %.1f mm / %.1f deg from target=%s (tol %.1f mm, %.1f deg); the arm "
            "followed the plan but the plan did not land on the goal.",
            err_mm, ori_err_deg, np.round(np.asarray(pose.position_mm), 1).tolist(),
            _MOVE_POS_TOL_MM, _MOVE_ORI_TOL_DEG,
        )
        return MotionResult.failed(
            MotionStatus.TIMEOUT, MotionCommand.MOVE_TO, target_pose=pose,
            message=f"cuRobo trajectory ended {err_mm:.1f} mm from target (> {_MOVE_POS_TOL_MM} mm).",
        )

    def _execute_curobo_trajectory(
        self, traj: list[list[float]], server_joint_names: list[str], traj_dt: float
    ) -> None:
        """Follow the cuRobo joint trajectory faithfully, with no per-waypoint IK and no interpolation.

        cuRobo guarantees the path is collision-free, so the arm tracks it rather than
        rushing through. Each waypoint is commanded as the drive target and stepped until
        the arm reaches it, within ``_CUROBO_WP_TOL_RAD``, or a per-waypoint step cap
        expires, so the arm physically follows the planned path at every configuration. A
        fixed number of steps per waypoint instead lets the drive lag, and on a contorted
        plan the arm ends far from the goal. A final settle on the last waypoint lands
        the TCP exactly. ``traj_dt`` is the time step of the plan and is informational,
        because tracking to a tolerance is what makes the follow robust.
        """
        if self._arm_subset is None:
            return
        idx = [server_joint_names.index(n) for n in _ARM_JOINT_NAMES]  # server order -> _ARM_JOINT_NAMES order
        last_q: np.ndarray | None = None
        for wp in traj:
            last_q = np.asarray([wp[i] for i in idx], dtype=np.float64)
            self._arm_subset.apply_action(joint_positions=last_q)
            for _ in range(_CUROBO_MAX_STEPS_PER_WP):
                self._session.step()
                cur = np.asarray(self._arm_subset.get_joint_positions(), dtype=np.float64)
                if float(np.max(np.abs(cur - last_q))) < _CUROBO_WP_TOL_RAD:
                    break  # reached this waypoint, so advance and keep the arm on the planned path
        if last_q is not None:  # re-assert the final config, then settle so the TCP lands exactly
            self._arm_subset.apply_action(joint_positions=last_q)
        self.wait_until_steady(self._config.settle_timeout_s)

    def move(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> MotionResult:
        """The typed move: Cartesian motion through RMPflow with an IK refine.

        A frame or connection fault is returned as a typed result and never raised.
        Outside mock mode it pre-resolves IK, giving ``IK_FAILED`` where the pose is
        unreachable, runs the preflight, drives RMPflow toward the pose as a smooth
        collision-aware approach, and then snaps to the exact IK solution. It is
        ``EXECUTED`` where the refined TCP is within ``_MOVE_POS_TOL_MM``, and
        ``TIMEOUT`` otherwise.
        """
        if pose.frame is not Frame.BASE:
            return MotionResult.failed(
                MotionStatus.INVALID_TARGET,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=(
                    "IsaacRobotArm.move requires Frame.BASE; got "
                    f"{pose.frame!r}."
                ),
            )
        if not self._connected:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message="IsaacRobotArm is not connected.",
            )
        # In mock_mode the kinematics are the identity. The preflight still runs, because
        # a pose-based guard can reject, and there is no articulation or IK, so the
        # joint-based guards see target_joints as None.
        if self.mock_mode:
            rejected = self._preflight_reject(pose, target_joints=None, current_joints=None)
            if rejected is not None:
                return rejected
            self._tcp = pose
            return MotionResult.executed(
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message="mock_mode",
            )
        # Outside mock mode with the articulation or motion policy never brought up, as
        # happens where the connected flag was set without a real connect(), this
        # surfaces a typed CONNECTION_ERROR rather than leaking a RobotConnectionError
        # from the kinematics path.
        if self._articulation is None or self._kin_solver is None or self._rmpflow is None:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message="IsaacRobotArm.move: articulation/motion policy not initialized (call connect()).",
            )
        # Pre-resolve IK before the preflight, so the joint-based guards, joint-limit,
        # IK-quality and self-collision, can evaluate at all: without target_joints they
        # report unavailable and, under enforce, fail closed. It warm-starts from the
        # current joints. Then the preflight runs with the resolved joints and the
        # current joints, and a rejection returns its typed status. The drive loop below
        # reuses this solution for the first attempt.
        try:
            target_joints = self._resolve_ik(pose)
        except RobotKinematicsError as exc:
            return MotionResult.failed(
                MotionStatus.IK_FAILED,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=f"IsaacRobotArm.move: IK failed: {exc}",
            )
        try:
            current_joints = self.get_joint_positions()
        except Exception:  # noqa: BLE001 (telemetry hiccup: the IK-jump check skips on None)
            current_joints = None
        # Resolve the effective planner once, falling back from curobo to ik where the
        # sidecar or its environment is unavailable, and then route on it, so a "curobo"
        # default is safe wherever the cuRobo environment is absent.
        planner = self._resolve_motion_planner()
        # cuRobo plans a smooth, in-limits, collision-free joint trajectory itself, so
        # the motion_continuity guard misfires across cuRobo moves. It memoises the
        # pre-resolved IK, which is used only for this preflight, while cuRobo executes a
        # different 2pi-equivalent IK branch, so the pre-resolved IK of the next move
        # steps about 360 deg against the stale memo and the guard falsely rejects.
        # cuRobo owns continuity, so the memo is cleared first, for the reason
        # _JOINT_MOVE_SKIP_GUARDS skips continuity on a deliberate joint command.
        if planner == "curobo":
            if self._preflight is not None:
                self._preflight.reset()
            # cuRobo owns collision, both self and world, so the preflight runs inside
            # _drive_curobo on the actual planned final configuration rather than on the
            # _resolve_ik above, which can pick a self-colliding IK branch cuRobo would
            # never execute. Measured: the corridor-clear -Y top-down close self-collides
            # on the _resolve_ik branch and cuRobo reaches it clean. That keeps the
            # high-fidelity Coal self-collision check on the configuration that executes
            # rather than falsely rejecting on an unused branch.
            return self._drive_curobo(pose, current_joints=current_joints)
        rejected = self._preflight_reject(pose, target_joints=target_joints, current_joints=current_joints)
        if rejected is not None:
            return rejected
        # The planner seam. The default "ik" skips it entirely, and the "curobo" branch
        # is handled above, where it owns its preflight on the planned configuration.
        # "rmpflow" drives the reactive collision-aware policy toward the pose first,
        # routing around the obstacles register_planner_obstacles registered.
        #
        # The IK snap is gated on RMPflow converging. Having routed there,
        # _drive_to_target refines the last 19 mm or so to the exact grasp, a short,
        # near-target, obstacle-free motion. Not having converged, blocked or in a local
        # minimum as mp_iso_avoid.py measures, this fails safe: a blind snap would ram
        # through the obstacle and defeat the avoidance, measured as a 292 mm shove. It
        # is collision-aware where it can route and an honest timeout where it cannot,
        # and the harder case is the job of cuRobo.
        if planner == "rmpflow" and self._rmpflow is not None:
            if not self._drive_rmpflow(pose):
                return MotionResult.failed(
                    MotionStatus.TIMEOUT,
                    MotionCommand.MOVE_TO,
                    target_pose=pose,
                    message=(
                        "RMPflow approach did not converge (obstacle-blocked / local minimum); "
                        "failing safe rather than blind-snapping through the obstacle."
                    ),
                )
        return self._drive_to_target(pose, target_joints)

    def _drive_to_target(
        self,
        pose: "Pose",
        target_joints: "JointPositions | None",
    ) -> MotionResult:
        """Drive to ``pose`` in joint space, retrying IK where an attempt does not converge.

        ``target_joints`` is the pre-resolved, safety-gated seed for the first attempt,
        and ``None`` forces a fresh resolve. It returns ``EXECUTED`` once the refined TCP
        is within tolerance, ``IK_FAILED`` where a re-resolve raises, and ``TIMEOUT``
        where no attempt converges.
        """
        # Drive in joint space, retrying where an attempt does not converge. A large
        # move, such as one from a parked arm, or a load-perturbed move can stall short
        # the first time, and re-resolving IK from the new configuration and driving
        # again recovers it. RMPflow, in `_drive_rmpflow`, is kept for cluttered scenes,
        # and the picks use this precise pure-IK path. The first attempt reuses the
        # pre-resolved, safety-gated solution, and a retry re-resolves.
        err_mm = float("inf")
        for _attempt in range(_MOVE_MAX_ATTEMPTS):
            if target_joints is None:
                try:
                    target_joints = self._resolve_ik(pose)
                except RobotKinematicsError as exc:
                    return MotionResult.failed(
                        MotionStatus.IK_FAILED,
                        MotionCommand.MOVE_TO,
                        target_pose=pose,
                        message=f"IsaacRobotArm.move: IK failed: {exc}",
                    )
            self.move_joint(target_joints)
            tcp = self.get_tcp_pose()
            err_mm = float(
                np.linalg.norm(np.asarray(tcp.position_mm) - np.asarray(pose.position_mm))
            )
            ori_err_deg = _quat_angle_deg(tcp.quaternion_xyzw, pose.quaternion_xyzw)
            if err_mm <= _MOVE_POS_TOL_MM and ori_err_deg <= _MOVE_ORI_TOL_DEG:
                return MotionResult.executed(
                    MotionCommand.MOVE_TO,
                    target_pose=pose,
                    message=f"IK + move_joint reached target ({err_mm:.1f} mm, {ori_err_deg:.1f} deg).",
                )
            if err_mm > _MOVE_RETRY_MAX_ERR_MM:
                break  # a far miss: a re-resolve of the same target will not recover, so fail fast
            target_joints = None  # near miss: re-resolve IK from the new configuration next attempt
        # A move that does not converge is the most opaque failure this driver produces:
        # it surfaces as a bare timeout, which reads as a bad grasp when it means the arm
        # could not reach the pose the grasp asked for. The distance and the
        # configuration it stalled in are what separate the two, so they are logged
        # rather than sealed into a message nobody reads.
        _LOGGER.warning(
            "move did NOT converge: %.1f mm from target (tol %.1f mm) after %d attempt(s); "
            "target=%s stalled at joints=%s",
            err_mm, _MOVE_POS_TOL_MM, _MOVE_MAX_ATTEMPTS,
            np.round(np.asarray(pose.position_mm), 1).tolist(),
            np.round(np.asarray(self.get_joint_positions().values), 3).tolist(),
        )
        return MotionResult.failed(
            MotionStatus.TIMEOUT,
            MotionCommand.MOVE_TO,
            target_pose=pose,
            message=f"IsaacRobotArm.move did not converge ({err_mm:.1f} mm > {_MOVE_POS_TOL_MM} mm).",
        )

    def _preflight_reject(
        self,
        pose: "Pose",
        *,
        target_joints: "JointPositions | None" = None,
        current_joints: "JointPositions | None" = None,
        skip_guards: "frozenset[str]" = frozenset(),
    ) -> "MotionResult | None":
        """Run the safety preflight for a Cartesian target.

        It returns a typed rejection :class:`MotionResult` where a guard rejects and
        ``None`` otherwise, including where no preflight was injected. ``target_joints``
        lets the joint-based guards, joint-limit, IK-quality and self-collision,
        evaluate; passing ``None`` makes them report unavailable, which is what the mock
        does because it has no IK. ``skip_guards``, empty by default, bypasses named
        guards and is used only by the cuRobo path, to skip the trajectory-continuity
        guards, which :meth:`_drive_curobo` explains.
        """
        if self._preflight is None:
            return None
        ctx = self._preflight.context_for_pose(
            pose,
            command=MotionCommand.MOVE_TO,
            target_joints=target_joints,
            current_joints=current_joints,
            arm=self,
        )
        decision = self._preflight.evaluate(ctx, skip_guards=skip_guards)
        return SafetyPreflight.as_motion_result(
            decision, MotionCommand.MOVE_TO, target_pose=pose,
        )
